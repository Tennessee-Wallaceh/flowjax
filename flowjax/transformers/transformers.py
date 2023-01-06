"""
Transformers available in the module ``flowjax.transformers``. Transformers are simple
bijections that can be parameterised by neural networks, e.g. as in a
:py:class:`~flowjax.flows.CouplingFlow` or :py:class:`~flowjax.flows.MaskedAutoregressiveFlow`.
Note, the use of the word "transformers" is unrelated to its use in self-attention
models!
"""

from functools import partial

import jax
import jax.numpy as jnp

from flowjax.transformers.abc import Transformer
from flowjax.utils import real_to_increasing_on_interval


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


class ScaleTransformer(AffineTransformer):
    "A Scale transformer is the same as an Affine transformer with a zero location."
    def num_params(self, dim):
        return dim

    def get_ranks(self, dim):
        return jnp.tile(jnp.arange(dim), 1)

    def get_args(self, params):
        return jnp.zeros_like(params), jnp.exp(params)


class RationalQuadraticSplineTransformer(Transformer):
    """RationalQuadraticSplineTransformer (https://arxiv.org/abs/1906.04032)."""
    K: int
    B: int
    softmax_adjust: float
    min_derivative: float
    left_trainable: bool

    def __init__(self, K, B, softmax_adjust=1e-2, min_derivative=1e-3, left_trainable=False):
        """
        Each row of parameter matrices (x_pos, y_pos, derivatives) corresponds to a column in x.
        Ouside the interval [-B, B], the identity transform is used. 

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
        self.left_trainable = left_trainable

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
        k = jnp.searchsorted(y_pos, y) - 1
        xk, xk1, yk, yk1 = x_pos[k], x_pos[k + 1], y_pos[k], y_pos[k + 1]
        sk = (yk1 - yk) / (xk1 - xk)
        y_delta_s_term = (y - yk) * (derivatives[k + 1] + derivatives[k] - 2 * sk)
        a = (yk1 - yk) * (sk - derivatives[k]) + y_delta_s_term
        b = (yk1 - yk) * derivatives[k] - y_delta_s_term
        c = -sk * (y - yk)
        sqrt_term = jnp.sqrt(b**2 - 4 * a * c)
        xi = (2 * c) / (-b - sqrt_term)
        x = xi * (xk1 - xk) + xk
        return x

    def inverse_and_log_abs_det_jacobian(self, y, x_pos, y_pos, derivatives):
        x = self.inverse(y, x_pos, y_pos, derivatives)
        derivative = self.derivative(x, x_pos, y_pos, derivatives)
        return x, -jnp.log(derivative).sum()

    def _params_per_dim(self):
        num_params = (self.K * 3 - 1)
        if self.left_trainable:
            num_params += 1 # extra parameter for left derivative
        return num_params

    def num_params(self, dim: int):
        return self._params_per_dim() * dim

    def get_ranks(self, dim: int):
        return jnp.repeat(jnp.arange(dim), self._params_per_dim())

    def get_args(self, params):
        params = params.reshape((-1, self._params_per_dim()))
        return jax.vmap(self._get_args)(params)

    def _get_args(self, params):
        "Gets the arguments for a single dimension of x (defined for 1d)."
        x_pos = real_to_increasing_on_interval(
            params[: self.K], self.B, self.softmax_adjust
        )
        y_pos = real_to_increasing_on_interval(
            params[self.K : self.K * 2], self.B, self.softmax_adjust
        )
        derivatives = jax.nn.softplus(params[self.K * 2 :]) + self.min_derivative

        # Padding sets up linear spline from the edge of the bounding box to B * 1e4
        pos_pad = jnp.array([self.B, 1e4 * self.B])
        x_pos = jnp.hstack((-jnp.flip(pos_pad), x_pos, pos_pad))
        y_pos = jnp.hstack((-jnp.flip(pos_pad), y_pos, pos_pad))
        if self.left_trainable:
            derivatives = jnp.pad(derivatives, (1, 2), constant_values=1)            
        else:
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
