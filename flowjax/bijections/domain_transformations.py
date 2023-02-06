"""
Some interesting disucssion on the topic of domain transformations: can be found
here - https://arxiv.org/pdf/2301.08297.pdf
"""
import jax
import jax.numpy as jnp
from jax.scipy.special import expit, logit
from flowjax.bijections import Bijection
from flowjax.utils import Array
from jax.nn import softplus
"""
Scalar real -> [a, inf]
"""
def exp(theta, minval=1e-6, tol=1e-12):
    return minval + jnp.clip(jnp.exp(theta), tol)

def softplus(theta, minval=1e-6, tol=1e-12):
    return minval + jnp.clip(jnp.log(1 + jnp.exp(theta)), tol)

class Softplus(Bijection):
    """
    Stable
    """
    _min: Array
    STABLE_REGION=10
    def __init__(self, min):
        self._min = min
        self.shape = ()
        self.cond_shape = None

    @property
    def min(self):
        return jax.lax.stop_gradient(self._min)

    def transform(self, z):
        x_add = jnp.where(
            z < self.STABLE_REGION,
            softplus(z),
            z
        )
        return self.min + x_add

    def transform_and_log_abs_det_jacobian(self, z):
        x_add = jnp.where(
            z < self.STABLE_REGION,
            softplus(z),
            z
        )
        x = self.min + x_add
        lad = jnp.where(
            z < self.STABLE_REGION,
            x - jnp.log(1 + jnp.exp(x)),
            0
        )
        return x, lad

    def inverse(self, x):
        adj_x = x - self.min
        z = jnp.where(
            adj_x < self.STABLE_REGION,
            jnp.log(jnp.exp(adj_x) - 1),
            adj_x
        )
        return z

    def inverse_and_log_abs_det_jacobian(self, x):
        adj_x = x - self.min
        z = jnp.where(
            adj_x < self.STABLE_REGION,
            jnp.log(jnp.exp(adj_x) - 1),
            adj_x
        )
        lad = jnp.where(
            adj_x < self.STABLE_REGION,
            adj_x - jnp.log(jnp.exp(adj_x) - 1),
            0,
        )
        return z, lad

"""
Scalar real -> [a, b]
"""
class Expit(Bijection):
    """
    Stable only for z in [-50, 50].
    If targetting fixed interval consider using splines.
    """
    _a: Array
    _b: Array
    def __init__(self, a, b):
        self._a = a
        self._b = b
        self.shape = ()
        self.cond_shape = None

    @property
    def a(self):
        return jax.lax.stop_gradient(self._a)

    @property
    def b(self):
        return jax.lax.stop_gradient(self._b)

    def transform(self, z):
        return self.a + (self.b - self.a) * expit(z)

    def transform_and_log_abs_det_jacobian(self, z):
        x = self.a + (self.b - self.a) * expit(z)
        lad = jnp.log(self.b - self.a) + z - 2 * jnp.log(1 + jnp.exp(-z))
        return x, lad

    def inverse(self, x):
        adj_x = (x - self.a) / (self.b - self.a)
        return logit(adj_x)

    def inverse_and_log_abs_det_jacobian(self, x):
        adj_x = (x - self.a) / (self.b - self.a)
        z = logit(adj_x)
        return z, -jnp.log(self.b - self.a) - z + 2 * jnp.log(1 + jnp.exp(-z))

