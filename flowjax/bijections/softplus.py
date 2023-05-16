from flowjax.bijections import Bijection
import jax.numpy as jnp
from jax.nn import softplus
import jax

def softplus_inverse(x):
    threshold = jnp.log(jnp.finfo(x).eps) + 2.
    is_too_small = x < jnp.exp(threshold)
    is_too_large = x > -threshold
    too_small_value = jnp.log(x)
    too_large_value = x
    # This `where` will ultimately be a NOP because we won't select this
    # codepath whenever we used the surrogate `ones_like`.
    x = jnp.where(is_too_small | is_too_large, jnp.ones([], x.dtype), x)
    y = x + jnp.log(-jnp.expm1(-x))  # == log(expm1(x))
    y = jnp.where(
        is_too_small,
        too_small_value,
        jnp.where(is_too_large, too_large_value, y)
    )
    return y

class Softplus(Bijection):
    """
    Stable transformation for
    """
    _min: float
    def __init__(self, min):
        self._min = min
        self.shape = ()
        self.cond_shape = None

    @property
    def min(self):
        return jax.lax.stop_gradient(self._min)

    def transform(self, z):
        x_add = softplus(z)
        return self.min + x_add

    def transform_and_log_abs_det_jacobian(self, z):
        raise NotImplementedError

    def inverse(self, x):
        adj_x = x - self.min
        return softplus_inverse(adj_x)

    def inverse_and_log_abs_det_jacobian(self, x):
        raise NotImplementedError