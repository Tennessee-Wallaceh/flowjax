"""Contains transformers, which are bijections that have methods that facilitate
parameterisation with neural networks. All transformers have the "Transformer"
suffix, to avoid potential name clashes with bijections.
"""

import jax
import jax.numpy as jnp
from flowjax.bijections import Transformer
from functools import partial
from jax.scipy.special import erf, erfinv
from jax.scipy import stats as jstats
from flowjax.utils import Array

class AffineTransformer(Transformer):
    "Affine transformation compatible with neural network parameterisation."

    def transform(self, x, loc, scale):
        return x * scale + loc

    def transform_and_log_abs_det_jacobian(self, x, loc, scale):
        return x * scale + loc, jnp.log(scale).sum()

    def inverse(self, y, loc, scale):
        return (y - loc) / scale

    def inverse_and_log_abs_det_jacobian(self, y, loc, scale):
        return self.inverse(y, loc, scale), -jnp.log(scale).sum()

    def num_params(self, dim):
        return dim * 2

    def get_ranks(self, dim):
        return jnp.tile(jnp.arange(dim), 2)

    def get_args(self, params):
        loc, log_scale = params.split(2)
        return loc, jnp.exp(log_scale)


class ScaleTransformer(Transformer):
    "Scale transformation compatible with neural network parameterisation."

    def transform(self, x, scale):
        return x * scale

    def transform_and_log_abs_det_jacobian(self, x, scale):
        return x * scale, jnp.log(scale)

    def inverse(self, y, scale):
        return y / scale

    def inverse_and_log_abs_det_jacobian(self, y, scale):
        return y / scale, -jnp.log(scale)

    def num_params(self, dim):
        return dim

    def get_ranks(self, dim):
        return jnp.tile(jnp.arange(dim), 1)

    def get_args(self, params):
        return (jnp.exp(params), )


class RationalQuadraticSplineTransformer(Transformer):
    """
    RationalQuadraticSplineTransformer (https://arxiv.org/abs/1906.04032). Ouside the interval
    [-B, B], the identity transform is used. Each row of parameter matrices
    (x_pos, y_pos, derivatives) corresponds to a column in x. 

    Args:
        K (int): Number of inner knots
        B: (int): Interval to transform [-B, B]
    """
    def __init__(self, K, B, softmax_adjust=1e-2, min_derivative=1e-3):
        """
        RationalQuadraticSplineTransformer (https://arxiv.org/abs/1906.04032). Ouside the interval
        [-B, B], the identity transform is used. Each row of parameter matrices
        (x_pos, y_pos, derivatives) corresponds to a column in x.

        Args:
            K (int): Number of inner knots
            B: (int): Interval to transform [-B, B]
            softmax_adjust: (float): Controls minimum bin width and height by rescaling softmax output, e.g. 0=no adjustment, 1=average softmax output with evenly spaced widths, >1 promotes more evenly spaced widths. See `real_to_increasing_on_interval`.
            min_derivative: (float): Minimum derivative.
        """
        self.K = K
        self.B = B
        self.softmax_adjust = softmax_adjust
        self.min_derivative = min_derivative

    @partial(jax.vmap, in_axes=[None, 0, 0, 0, 0])
    def transform(self, x, x_pos, y_pos, derivatives):
        k = jnp.searchsorted(x_pos, x) - 1
        xi = (x - x_pos[k]) / (x_pos[k + 1] - x_pos[k])
        sk = (y_pos[k + 1] - y_pos[k]) / (x_pos[k + 1] - x_pos[k])
        dk, dk1, yk, yk1 = derivatives[k], derivatives[k + 1], y_pos[k], y_pos[k + 1]
        num = (yk1 - yk) * (sk * xi**2 + dk * xi * (1 - xi))
        den = sk + (dk1 + dk - 2 * sk) * xi * (1 - xi)
        return yk + num / den  # eq. 4

    def transform_and_log_abs_det_jacobian(self, x, x_pos, y_pos, derivatives):
        y = self.transform(x, x_pos, y_pos, derivatives)
        derivative = self.derivative(x, x_pos, y_pos, derivatives)
        return y, jnp.log(derivative).sum()

    @partial(jax.vmap, in_axes=[None, 0, 0, 0, 0])
    def inverse(self, y, x_pos, y_pos, derivatives):
        outside = jnp.logical_or(y < -self.B, y > self.B)
        return jnp.where(
            outside, y, self._inverse(y, x_pos, y_pos, derivatives)
        )
    
    def _inverse(self, y, x_pos, y_pos, derivatives):
        k = jnp.searchsorted(y_pos, y) - 1
        xk, xk1, yk, yk1 = x_pos[k], x_pos[k + 1], y_pos[k], y_pos[k + 1]
        sk = (yk1 - yk) / (xk1 - xk)
        y_delta_s_term = (y - yk) * (derivatives[k + 1] + derivatives[k] - 2 * sk)
        a = (yk1 - yk) * (sk - derivatives[k]) + y_delta_s_term
        b = (yk1 - yk) * derivatives[k] - y_delta_s_term
        c = -sk * (y - yk)
        sqrt_term = jnp.sqrt(b**2 - 4 * a * c)
        xi = (2 * c) / (-b - sqrt_term)
        return xi * (xk1 - xk) + xk

    def inverse_and_log_abs_det_jacobian(self, y, x_pos, y_pos, derivatives):
        outside = jnp.logical_or(y < -self.B, y > self.B)
        x = self.inverse(y, x_pos, y_pos, derivatives)
        _x = jnp.where(
            outside, jnp.zeros_like(x), x
        )
        lad = jnp.where(
            outside,
            jnp.zeros_like(y),
            -jnp.log(self.derivative(_x, x_pos, y_pos, derivatives))
        )
        return x, lad

    def _num_params_per_dim(self):
        if self.left_trainable:
            return (self.K * 3)
        else:
            return (self.K * 3 - 1)

    def num_params(self, dim: int):
        return self._num_params_per_dim() * dim

    def get_ranks(self, dim: int):
        return jnp.repeat(jnp.arange(dim), self._num_params_per_dim())

    def get_args(self, params):
        params = params.reshape((-1, self._num_params_per_dim()))
        return jax.vmap(self._get_args)(params)

    def _get_args(self, params):
        "Gets the arguments for a single dimension of x (defined for 1d)."
        # if we are left trainable, we have an extra param and pad only right side
        if self.left_trainable:
            derivatives = jax.nn.softplus(params[self.K * 2:]) + self.min_derivative
            derivatives = jnp.pad(derivatives, (1, 2), constant_values=1)
        else:
            derivatives = jax.nn.softplus(params[self.K * 2:]) + self.min_derivative
            derivatives = jnp.pad(derivatives, 2, constant_values=1)

        x_pos = real_to_increasing_on_interval(
            params[: self.K], self.B, self.softmax_adjust
        )
        y_pos = real_to_increasing_on_interval(
            params[self.K : self.K * 2], self.B, self.softmax_adjust
        )


        # Padding sets up linear spline from the edge of the bounding box to B * 1e4
        pos_pad = jnp.array([self.B, 1e4 * self.B])
        x_pos = jnp.hstack((-jnp.flip(pos_pad), x_pos, pos_pad))
        y_pos = jnp.hstack((-jnp.flip(pos_pad), y_pos, pos_pad))
        derivatives = jnp.pad(derivatives, 2, constant_values=1)

        return x_pos, y_pos, derivatives

    @partial(jax.vmap, in_axes=[None, 0, 0, 0, 0])
    def derivative(self, x, x_pos, y_pos, derivatives):  # eq. 5
        k = jnp.searchsorted(x_pos, x) - 1
        xi = (x - x_pos[k]) / (x_pos[k + 1] - x_pos[k])
        sk = (y_pos[k + 1] - y_pos[k]) / (x_pos[k + 1] - x_pos[k])
        dk, dk1 = derivatives[k], derivatives[k + 1]
        num = sk**2 * (dk1 * xi**2 + 2 * sk * xi * (1 - xi) + dk * (1 - xi) ** 2)
        den = (sk + (dk1 + dk - 2 * sk) * xi * (1 - xi)) ** 2
        return num / den

class InverseNormCDF(Transformer):
    def transform(self, u, loc, scale):
        return loc + scale * jnp.sqrt(2) * erfinv(2 * u - 1)

    def transform_and_log_abs_det_jacobian(self, u, loc, scale):
        raise NotImplementedError

    def inverse(self, x, loc, scale):
        std_x = (x - loc) / scale
        return 0.5 * (1 + erf(std_x / jnp.sqrt(2))) 

    def inverse_and_log_abs_det_jacobian(self, x, loc, scale):
        std_x = (x - loc) / scale
        x = 0.5 * (1 + erf(std_x / jnp.sqrt(2)))
        return x, jstats.norm.logpdf(std_x) - jnp.log(scale)

    def num_params(self, dim):
        return dim * 2

    def get_ranks(self, dim):
        return jnp.tile(jnp.arange(dim), 2)

    def get_args(self, params):
        loc, log_scale = params.split(2)
        return loc, jnp.exp(log_scale)


class ErfInv(Transformer):
    def transform(self, u):
        return erfinv(u)

    def transform_and_log_abs_det_jacobian(self, u):
        raise NotImplementedError

    def inverse(self, x):
        return erf(x) 

    def inverse_and_log_abs_det_jacobian(self, x):
        return erf(x), jnp.log(2 / jnp.sqrt(jnp.pi)) - jnp.square(x) 

    def num_params(self, dim):
        return 0

    def get_ranks(self, dim):
        return jnp.tile(jnp.arange(dim), 2)

    def get_args(self, params):
        return tuple()


class Logit(Transformer):
    """
    U -> X logit
    X -> U sigmoid
    """
    MIN_SIG = 1e-15
    MAX_SIG = 1 - 1e-7
    def transform(self, u):
        return jnp.log((u + 1) / (1 - u))

    def transform_and_log_abs_det_jacobian(self, u):
        raise NotImplementedError

    def inverse(self, x):
        return 2 * jax.nn.sigmoid(x) - 1

    def inverse_and_log_abs_det_jacobian(self, x):
        sig = jnp.clip(
            jax.nn.sigmoid(x), 
            a_min=self.MIN_SIG, 
            a_max=self.MAX_SIG,
        )
        return 2 * sig - 1, jnp.log(2 * sig * (1 - sig))

    def num_params(self, dim):
        return 0

    def get_ranks(self, dim):
        return jnp.tile(jnp.arange(dim), 2)

    def get_args(self, params):
        return tuple()


class Exponential(Transformer):
    """
    Z -> X exp
    X -> Z log
    """
    def transform(self, z):
        return jnp.exp(z)

    def transform_and_log_abs_det_jacobian(self, u):
        raise NotImplementedError

    def inverse(self, x):
        return jnp.log(x)

    def inverse_and_log_abs_det_jacobian(self, x):
        return jnp.log(x), jnp.log(1 / x)

    def num_params(self, dim):
        return 0

    def get_ranks(self, dim):
        return jnp.tile(jnp.arange(dim), 2)

    def get_args(self, params):
        return tuple()


class Softplus(Transformer):
    """
    Z -> X exp
    X -> Z log
    """
    THRESHOLD=10
    def transform(self, z):
        return jnp.where(
            z > self.THRESHOLD,
            z,
            jax.nn.softplus(z)
        )

    def transform_and_log_abs_det_jacobian(self, z):
        x = jnp.where(
            z > self.THRESHOLD,
            z,
            jax.nn.softplus(z)
        )
        log_abs_det_jacobian = jnp.where(
            z > self.THRESHOLD,
            z,
            z - jax.nn.softplus(z)
        )
        return x, log_abs_det_jacobian

    def inverse(self, x):
        return jnp.where(
            x > self.THRESHOLD,
            x,
            jnp.log(jnp.exp(x))
        )

    def inverse_and_log_abs_det_jacobian(self, x):
        z = jnp.where(
            x > self.THRESHOLD,
            x,
            jnp.log(jnp.exp(x))
        )
        log_abs_det_jacobian = x - z
        return z, log_abs_det_jacobian

    def num_params(self, dim):
        return 0

    def get_ranks(self, dim):
        return jnp.tile(jnp.arange(dim), 0)

    def get_args(self, params):
        return tuple()


def real_to_increasing_on_interval(
    arr: Array, B: float = 1, softmax_adjust: float = 1e-2
):
    """Transform unconstrained parameter vector to monotonically increasing positions on [-B, B].

    Args:
        arr (Array): Parameter vector.
        B (float, optional): Interval to transform output. Defaults to 1.
        softmax_adjust (float, optional): Rescales softmax output using (widths + softmax_adjust/widths.size) / (1 + softmax_adjust). e.g. 0=no adjustment, 1=average softmax output with evenly spaced widths, >1 promotes more evenly spaced widths.
    """
    if softmax_adjust < 0:
        raise ValueError("softmax_adjust should be >= 0.")
    widths = jax.nn.softmax(arr)
    widths = (widths + softmax_adjust / widths.size) / (1 + softmax_adjust)
    widths = widths.at[0].set(widths[0] / 2)
    return 2 * B * jnp.cumsum(widths) - B
