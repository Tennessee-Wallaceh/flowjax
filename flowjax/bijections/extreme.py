from flowjax.bijections.abc import Transformer
import jax.numpy as jnp
import jax
from jax.scipy.special import erfc, ndtri, betainc
from functools import partial
from jaxopt import Bisection

def _erfcinv(x):
    return -ndtri(0.5 * x) / jnp.sqrt(2)

def pos_domain(params, min_val):
    params = params.reshape((-1, 2))
    tail_params = min_val + jnp.exp(params)
    return tail_params[:, 0], tail_params[:, 1]

def min_max_domain(params, min_val, max_val):
    params = params.reshape((-1, 2))
    tail_params = jax.nn.sigmoid(params)
    tail_params *= max_val - min_val
    tail_params += min_val
    return tail_params[:, 0], tail_params[:, 1]

class ExtremeValueActivation(Transformer):
    """
    ExtremeValueActivation (D. Prangle, T. Hickling)

    This transform is Reals -> Reals.
    """
    # MIN_ERF_INV = jnp.finfo(jnp.float32).smallest_normal
    MIN_ERF_INV = 5e-7
    def __init__( self, min_tail_param=1e-3, max_tail_param=1):
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

    @partial(jax.vmap, in_axes=[None, 0, 0, 0])
    def transform_and_log_abs_det_jacobian(self, u, pos_tail, neg_tail):
        sign = jnp.sign(u)
        tail_param = jnp.where(sign > 0, pos_tail, neg_tail)
        g = erfc(jnp.abs(u) / jnp.sqrt(2))

        transformed = sign / tail_param
        transformed *= jnp.power(g, -tail_param) - 1

        lad = jnp.log(g) * (-tail_param -1)
        lad -= 0.5 * jnp.square(u) 
        lad += jnp.log(jnp.sqrt(2) / jnp.sqrt(jnp.pi))

        return transformed, lad

    @partial(jax.vmap, in_axes=[None, 0, 0, 0])
    def inverse(self, x, pos_tail, neg_tail):
        sign = jnp.sign(x)
        tail_param = jnp.where(sign > 0, pos_tail, neg_tail)

        g = jnp.power(1 + tail_param * jnp.abs(x), -1 / tail_param)
        g = jnp.clip(g, a_min=self.MIN_ERF_INV) # Should be in (0, 1]

        transformed = sign * jnp.sqrt(2) * _erfcinv(g)

        return transformed

    @partial(jax.vmap, in_axes=[None, 0, 0, 0])
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

        return transformed, lad

    def num_params(self, dim: int) -> int:
        return dim * 2
    
    def get_ranks(self, dim: int):
        return jnp.repeat(jnp.arange(dim), 2)

    def get_args(self, *args, **kwargs):
        return self._get_args(*args, **kwargs)


class FullActivation(Transformer):
    """
    FullActivation (D. Prangle, T. Hickling)

    This transform is Reals -> Reals.
    """
    # MIN_ERF_INV = jnp.finfo(jnp.float32).smallest_normal
    # MIN_ERF_INV = 5e-7
    def __init__( self, min_tail_param=1e-3, max_tail_param=1):
        if max_tail_param is None:
            self._get_args = lambda params: pos_domain(params, min_tail_param)
        else:
            self._get_args = lambda params: min_max_domain(params, min_tail_param, max_tail_param)

    @partial(jax.vmap, in_axes=[None, 0, 0, 0, 0])
    def transform(self, z, pos_tail, neg_tail, df):
        """
        From light tailed real to heavy real
        """
        sign = jnp.sign(z)
        tail_param = jnp.where(sign > 0, pos_tail, neg_tail)

        u = df / (jnp.square(z) + df)
        g = betainc(0.5 * df, 0.5, u)

        transformed = sign / tail_param
        transformed *= jnp.power(g, -tail_param) - 1

        return transformed

    @partial(jax.vmap, in_axes=[None, 0, 0, 0, 0])
    def transform_and_log_abs_det_jacobian(self, z, pos_tail, neg_tail, df):
        sign = jnp.sign(z)
        tail_param = jnp.where(sign > 0, pos_tail, neg_tail)

        u = df / (jnp.square(z) + df)
        g = betainc(0.5 * df, 0.5, u)

        transformed = sign / tail_param
        transformed *= jnp.power(g, -tail_param) - 1

        sq_z = jnp.square(z) + df

        lad = -(0.5 * df - 1) * jnp.log(1 - df / sq_z)
        lad += 0.5 * jnp.log(df / sq_z)
        lad += 2 * jnp.log(sq_z)
        lad -= jnp.log(df * z)

        return transformed, lad

    @partial(jax.vmap, in_axes=[None, 0, 0, 0, 0])
    def inverse(self, x, pos_tail, neg_tail, df):
        sign = jnp.sign(x)
        tail_param = jnp.where(sign > 0, pos_tail, neg_tail)

        inner = 1 + tail_param * jnp.abs(x)
        g = jnp.power(inner, -1 / tail_param)

        target = lambda u: betainc(0.5 * df, 0.5, u)  - g # Ie the z which gives x

        bisec = Bisection(
            optimality_fun=target,
            lower=0, 
            upper=10000
        )
        u = bisec.run().params
        g_inv = jnp.sqrt(1 + 1 / u)

        transformed = sign * jnp.sqrt(2) * g_inv

        return transformed

    @partial(jax.vmap, in_axes=[None, 0, 0, 0, 0])
    def inverse_and_log_abs_det_jacobian(self, x, pos_tail, neg_tail, df):
        sign = jnp.sign(x)
        tail_param = jnp.where(sign > 0, pos_tail, neg_tail)

        inner = 1 + tail_param * jnp.abs(x)
        g = jnp.power(inner, -1 / tail_param)
        g = jnp.clip(g, a_min=self.MIN_ERF_INV) # Should be in (0, 1]

        target = lambda u: betainc(0.5 * df, 0.5, u)  - g # Ie the z which gives x

        bisec = Bisection(
            optimality_fun=target,
            lower=0, 
            upper=10000
        )
        u = bisec.run().params
        g_inv = jnp.sqrt(1 + 1 / u)

        transformed = sign * jnp.sqrt(2) * g_inv
    
        lad = (-1 - 1 / tail_param) * jnp.log(inner)
        lad += jnp.log(0.5 * jnp.sqrt(2) * jnp.sqrt(jnp.pi))
        lad += jnp.square(_erfcinv(g))

        return transformed, lad

    def num_params(self, dim: int) -> int:
        return dim * 3
    
    def get_ranks(self, dim: int):
        return jnp.repeat(jnp.arange(dim), 3)

    def get_args(self, params):
        return self._get_args(params[:, :2]), jnp.exp(params[:, 2])


class TailTransformation(Transformer):
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
        pass

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


class Kuma(Transformer):
    """
    Kumasawary transform

    This transform is (-1, 1) -> (-1, 1).
    """
    @partial(jax.vmap, in_axes=[None, 0, 0, 0, 0, 0])
    def transform(self, u, a_neg, b_neg, a_pos, b_pos):
        sign = jnp.sign(u)
        a = jnp.where(sign > 0, a_pos, a_neg)
        b = jnp.where(sign > 0, b_pos, b_neg)

        transformed = jnp.power(1 - jnp.abs(u), 1 / b)
        transformed = sign * jnp.power(1 - transformed, 1 / a)
        transformed = jnp.clip(transformed, -1 + 1e-7, 1 - 1e-7)
        return transformed

    @partial(jax.vmap, in_axes=[None, 0, 0, 0, 0, 0])
    def transform_and_log_abs_det_jacobian(self, u, a_neg, b_neg, a_pos, b_pos):
        pass

    @partial(jax.vmap, in_axes=[None, 0, 0, 0, 0, 0])
    def inverse(self, x, a_neg, b_neg, a_pos, b_pos):
        sign = jnp.sign(x)
        x = jnp.clip(x, -1 + 1e-7, 1 - 1e-7)
        a = jnp.where(sign > 0, a_pos, a_neg)
        b = jnp.where(sign > 0, b_pos, b_neg)

        x_abs = jnp.abs(x)
        transformed = 1 - jnp.power(x_abs, a)
        transformed = sign * (1 - jnp.power(transformed, b))

        return transformed

    @partial(jax.vmap, in_axes=[None, 0, 0, 0, 0, 0])
    def inverse_and_log_abs_det_jacobian(self, x,  a_neg, b_neg, a_pos, b_pos):
        sign = jnp.sign(x)
        x = jnp.clip(x, -1 + 1e-7, 1 - 1e-7)
        a = jnp.where(sign > 0, a_pos, a_neg)
        b = jnp.where(sign > 0, b_pos, b_neg)
        x_abs = jnp.abs(x)

        transformed = 1 - jnp.power(x_abs, a)
        transformed = sign * (1 - jnp.power(transformed, b))
        
        d_dx = jnp.power(1 - jnp.power(x_abs, a), b - 1)
        d_dx *= jnp.power(x_abs, a - 1)
        d_dx *= a * b

        return transformed, jnp.log(d_dx)

    def num_params(self, dim: int) -> int:
        return dim * 4
    
    def get_ranks(self, dim: int):
        return jnp.repeat(jnp.arange(dim), 4)

    def get_args(self, params):
        params = params.reshape((-1, 4))
        # transformed = jax.nn.sigmoid(params)
        transformed = jnp.exp(params)
        return (
            transformed[:, 0], 
            transformed[:, 1], 
            transformed[:, 2], 
            transformed[:, 3]
        )