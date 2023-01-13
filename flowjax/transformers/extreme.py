from flowjax.transformers import Transformer
import jax.numpy as jnp
import jax
from jax.scipy.special import erfc, ndtri
from functools import partial
from tensorflow_probability.substrates import jax as tfp
from jax.scipy.special import gammainc, gammaln

def _erfcinv(x):
    return -ndtri(0.5 * x) / jnp.sqrt(2)

def pos_domain(params, min_val):
    tail_params = min_val + jnp.exp(params)
    return tail_params.split(2) # pos, neg

def min_max_domain(params, min_val, max_val):
    tail_params = jax.nn.sigmoid(params)
    tail_params *= max_val - min_val
    tail_params += min_val
    return tail_params.split(2) # pos, neg

class ExtremeValueActivation(Transformer):
    _get_args: callable
    """
    ExtremeValueActivation (D. Prangle, T. Hickling)

    This transform is Reals -> Reals.
    """
    MIN_ERF_INV = 5e-7
    def __init__(self, min_tail_param=1e-3, max_tail_param=1):
        if max_tail_param is None:
            self._get_args = lambda params: pos_domain(params, min_tail_param)
        else:
            self._get_args = lambda params: min_max_domain(params, min_tail_param, max_tail_param)

    @partial(jax.vmap, in_axes=[None, 0, 0, 0])
    def transform(self, u, pos_tail, neg_tail):
        """
        From light tailed real to heavy real
        """
        sign = jnp.sign(u)
        tail_param = jnp.where(sign > 0, pos_tail, neg_tail)
        g = erfc(jnp.abs(u) / jnp.sqrt(2))

        transformed = sign / tail_param
        transformed *= jnp.power(g, -tail_param) - 1
        return transformed

    def transform_and_log_abs_det_jacobian(self, u, pos_tail, neg_tail):
        sign = jnp.sign(u)
        tail_param = jnp.where(sign > 0, pos_tail, neg_tail)
        g = erfc(jnp.abs(u) / jnp.sqrt(2))

        transformed = sign / tail_param
        transformed *= jnp.power(g, -tail_param) - 1

        lad = jnp.log(g) * (-tail_param -1)
        lad -= 0.5 * jnp.square(u) 
        lad += jnp.log(jnp.sqrt(2) / jnp.sqrt(jnp.pi))

        return transformed, jnp.sum(lad)

    @partial(jax.vmap, in_axes=[None, 0, 0, 0])
    def inverse(self, x, pos_tail, neg_tail):
        sign = jnp.sign(x)
        tail_param = jnp.where(sign > 0, pos_tail, neg_tail)

        g = jnp.power(1 + tail_param * jnp.abs(x), -1 / tail_param)
        g = jnp.clip(g, a_min=self.MIN_ERF_INV) # Should be in (0, 1]

        transformed = sign * jnp.sqrt(2) * _erfcinv(g)

        return transformed

    def inverse_and_log_abs_det_jacobian(self, x, pos_tail, neg_tail):
        sign = jnp.sign(x)
        tail_param = jnp.where(sign > 0, pos_tail, neg_tail)

        inner = 1 + tail_param * jnp.abs(x)
        g = jnp.power(inner, -1 / tail_param)
        g = jnp.clip(g, a_min=self.MIN_ERF_INV) # Should be in (0, 1]

        transformed = sign * jnp.sqrt(2) * _erfcinv(g)
    
        lad = (-1 - 1 / tail_param) * jnp.log(inner)
        lad += jnp.log(0.5 * jnp.sqrt(2) * jnp.sqrt(jnp.pi))
        lad += jnp.square(_erfcinv(g))

        return transformed, jnp.sum(lad)

    def num_params(self, dim: int) -> int:
        return dim * 2
    
    def get_ranks(self, dim: int):
        return jnp.tile(jnp.arange(dim), 2)

    def get_args(self, *args, **kwargs):
        return self._get_args(*args, **kwargs)


class TailTransformation(Transformer):
    _get_args: callable
    """
    TailTransformation (D. Prangle, T. Hickling)

    This transform is (-1, 1) -> Reals.
    """
    def __init__( self, min_tail_param=1e-3, max_tail_param=1):
        if max_tail_param is None:
            self._get_args = lambda params: pos_domain(params, min_tail_param)
        else:
            self._get_args = lambda params: min_max_domain(params, min_tail_param, max_tail_param)

    @partial(jax.vmap, in_axes=[None, 0, 0, 0])
    def transform(self, u, pos_tail, neg_tail):
        sign = jnp.sign(u)
        tail_param = jnp.where(sign > 0, pos_tail, neg_tail)

        transformed = sign / tail_param
        transformed *= jnp.power(1 - jnp.abs(u), -tail_param) - 1
        
        return transformed

    @partial(jax.vmap, in_axes=[None, 0, 0, 0])
    def transform_and_log_abs_det_jacobian(self, u, pos_tail, neg_tail):
        sign = jnp.sign(u)
        tail_param = jnp.where(sign > 0, pos_tail, neg_tail)

        transformed = sign / tail_param
        transformed *= jnp.power(1 - jnp.abs(u), -tail_param) - 1

        lad = (-tail_param - 1) * jnp.log(1 - jnp.abs(u))
        
        return transformed, lad

    @partial(jax.vmap, in_axes=[None, 0, 0, 0])
    def inverse(self, x, pos_tail, neg_tail):
        sign = jnp.sign(x)
        tail_param = jnp.where(sign > 0, pos_tail, neg_tail)
        neg_tail_inv = -1 / tail_param

        transformed = jnp.power(1 + sign * tail_param * x, neg_tail_inv)
        transformed = sign * (1 - transformed)

        return transformed

    @partial(jax.vmap, in_axes=[None, 0, 0, 0])
    def inverse_and_log_abs_det_jacobian(self, x, pos_tail, neg_tail):
        sign = jnp.sign(x)
        tail_param = jnp.where(sign > 0, pos_tail, neg_tail)
        neg_tail_inv = -1 / tail_param

        transformed = jnp.power(1 + sign * tail_param * x, neg_tail_inv)
        transformed = sign * (1 - transformed)

        lad = (neg_tail_inv - 1) * jnp.log(1 + sign * tail_param * x)

        return transformed, lad

    def num_params(self, dim: int) -> int:
        return dim * 2
    
    def get_ranks(self, dim: int):
        return jnp.repeat(jnp.arange(dim), 2)

    def get_args(self, *args, **kwargs):
        return self._get_args(*args, **kwargs)


class SwitchTransform(Transformer):
    """
    SwitchTransform (D. Prangle, T. Hickling)

    This transform is Reals -> Reals.
    """
    MIN_ERF_INV = 5e-7
    @partial(jax.vmap, in_axes=[None, 0, 0, 0])
    def transform(self, z, pos_tail, neg_tail):
        sign = jnp.sign(z)
        tail_param = jnp.where(sign > 0, pos_tail, neg_tail)
        heavy_param = jnp.abs(tail_param)
        light_param = -jnp.abs(tail_param)

        # The heavy transformation
        g = erfc(jnp.abs(z) / jnp.sqrt(2))
        heavy_transformed = sign / heavy_param
        heavy_transformed *= jnp.power(g, -tail_param) - 1

        # The light transformation
        light_transformed = sign * jnp.power(jnp.abs(z), 2 + light_param)
        
        return jnp.where(
            tail_param > 0, 
            heavy_transformed,
            light_transformed
        )

    @partial(jax.vmap, in_axes=[None, 0, 0, 0])
    def transform_and_log_abs_det_jacobian(self, u, pos_tail, neg_tail):
        raise NotImplementedError

    @partial(jax.vmap, in_axes=[None, 0, 0, 0])
    def inverse(self, x, pos_tail, neg_tail):
        sign = jnp.sign(x)
        tail_param = jnp.where(sign > 0, pos_tail, neg_tail)
        heavy_param = jnp.abs(tail_param)
        light_param = -jnp.abs(tail_param)

        g = jnp.power(1 + heavy_param * jnp.abs(x), -1 / heavy_param)
        g = jnp.clip(g, a_min=self.MIN_ERF_INV) # Should be in (0, 1]
        heavy_transformed = sign * jnp.sqrt(2) * _erfcinv(g)

        x_adj = jnp.clip(jnp.abs(x), 1e-6)
        light_transformed = sign * jnp.power(x_adj, 1 / (2 + light_param))

        return jnp.where(
            tail_param > 0, 
            heavy_transformed, 
            light_transformed
        )

    @partial(jax.vmap, in_axes=[None, 0, 0, 0])
    def inverse_and_log_abs_det_jacobian(self, x, pos_tail, neg_tail):
        sign = jnp.sign(x)
        tail_param = jnp.where(sign > 0, pos_tail, neg_tail)
        heavy_param = jnp.abs(tail_param)
        light_param = -jnp.abs(tail_param)

        inner = 1 + heavy_param * jnp.abs(x)
        g = jnp.power(inner, -1 / heavy_param)
        g = jnp.clip(g, a_min=self.MIN_ERF_INV) # Should be in (0, 1]
        heavy_transformed = sign * jnp.sqrt(2) * _erfcinv(g)

        heavy_lad = (-1 - 1 / heavy_param) * jnp.log(inner)
        heavy_lad += jnp.log(0.5 * jnp.sqrt(2) * jnp.sqrt(jnp.pi))
        heavy_lad += jnp.square(_erfcinv(g))

        x_adj = jnp.clip(jnp.abs(x), 1e-6)
        light_transformed = sign * jnp.power(x_adj, 1 / (2 + light_param))
        light_lad = -jnp.log(2 + light_param)
        light_lad -= (1 + light_param) * jnp.log(x_adj) / (2 + light_param)
        
        # select whether we wanted the heavy or light transformation
        transformed = jnp.where(tail_param > 0, heavy_transformed, light_transformed)
        lad = jnp.where(tail_param > 0, heavy_lad, light_lad)
        
        return transformed, lad

    def num_params(self, dim: int) -> int:
        return dim * 2
    
    def get_ranks(self, dim: int):
        return jnp.repeat(jnp.arange(dim), 2)

    def get_args(self, params):
        """
        Parameter should be in [-1, inf]
        """
        params = jnp.exp(params.reshape((-1, 2))) - 1
        return params[:, 0], params[:, 1]


class ShiftChi(Transformer):
    """
    ShiftChi (D. Prangle, T. Hickling)

    This transform is (-1, 1) -> Reals.

    The parameter is unconstrained.
    """
    @partial(jax.vmap, in_axes=[None, 0, 0, 0])
    def transform(self, u, pos_tail, neg_tail):
        sign = jnp.sign(u)
        tail_param = jnp.where(sign > 0, pos_tail, neg_tail)

        g = (jnp.abs(u) - 1) * (1 - gammainc(tail_param, 1)) + 1
        x = tfp.math.igammainv(tail_param, g) - 1

        return sign * x

    @partial(jax.vmap, in_axes=[None, 0, 0, 0])
    def transform_and_log_abs_det_jacobian(self, u, pos_tail, neg_tail):
        raise NotImplementedError 

    @partial(jax.vmap, in_axes=[None, 0, 0, 0])
    def inverse(self, x, pos_tail, neg_tail):
        sign = jnp.sign(x)
        tail_param = jnp.where(sign > 0, pos_tail, neg_tail)

        transformed = gammainc(tail_param, jnp.abs(x) + 1) - 1
        transformed /= 1 - gammainc(tail_param,  1)
        transformed += 1

        return sign * transformed

    @partial(jax.vmap, in_axes=[None, 0, 0, 0])
    def inverse_and_log_abs_det_jacobian(self, x, pos_tail, neg_tail):
        sign = jnp.sign(x)
        tail_param = jnp.where(sign > 0, pos_tail, neg_tail)

        transformed = gammainc(tail_param, jnp.abs(x) + 1) - 1
        transformed /= 1 - gammainc(tail_param,  1)
        transformed += 1

        lad = (tail_param - 1) * jnp.log(jnp.abs(x) + 1)
        lad -= (jnp.abs(x) + 1)
        lad -= gammaln(tail_param) + jnp.log(1 - gammainc(tail_param, 1))
        
        return sign * transformed, lad

    def num_params(self, dim: int) -> int:
        return dim * 2
    
    def get_ranks(self, dim: int):
        return jnp.repeat(jnp.arange(dim), 2)

    def get_args(self, params):
        params = jax.nn.sigmoid(params.reshape((-1, 2)))
        return params[:, 0], params[:, 1]

class Power(Transformer):
    eps: float

    def __init__(self, eps):
        self.eps = eps

    """
    Power (D. Prangle, T. Hickling)

    This transform is Reals -> Reals.

    The parameter is in [0, -1].
    """
    @partial(jax.vmap, in_axes=[None, 0, 0, 0])
    def transform(self, z, pos_tail, neg_tail):
        sign = jnp.sign(z)
        tail_param = jnp.where(sign > 0, pos_tail, neg_tail)
        transformed = sign * jnp.power(jnp.abs(z), tail_param)
        return transformed

    @partial(jax.vmap, in_axes=[None, 0, 0, 0])
    def transform_and_log_abs_det_jacobian(self, u, pos_tail, neg_tail):
        raise NotImplementedError 

    @partial(jax.vmap, in_axes=[None, 0, 0, 0])
    def inverse(self, x, pos_tail, neg_tail):
        sign = jnp.sign(x)
        x_adj = jnp.clip(jnp.abs(x), 1e-6)
        tail_param = jnp.where(sign > 0, pos_tail, neg_tail)
        transformed = sign * jnp.power(x_adj, 1 / tail_param)
        return transformed

    @partial(jax.vmap, in_axes=[None, 0, 0, 0])
    def inverse_and_log_abs_det_jacobian(self, x, pos_tail, neg_tail):
        sign = jnp.sign(x)
        abs_x = jnp.abs(x)

        tail_param = jnp.where(sign > 0, pos_tail, neg_tail)
        
        transformed = jnp.power(abs_x + self.eps, 1 / tail_param) - self.eps
        transformed = sign * transformed

        lad = -jnp.log(tail_param)
        lad += (1 / tail_param - 1) * jnp.log(abs_x + self.eps)

        return  transformed, lad

    def num_params(self, dim: int) -> int:
        return dim * 2
    
    def get_ranks(self, dim: int):
        return jnp.repeat(jnp.arange(dim), 2)

    def get_args(self, params):
        params = -jax.nn.sigmoid(params.reshape((-1, 2)))
        return params[:, 0], params[:, 1]