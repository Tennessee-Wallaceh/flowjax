from flowjax.bijections.abc import Transformer
import jax
import jax.numpy as jnp
from flowjax.bijections.abc import Bijection
from flowjax.utils import Array
from jax import random

class Univariate(Bijection):
    """
    A class implementing a Bijection for the unconditional univariate case.
    """
    transformer: Transformer
    cond_dim: int
    params: Array
    def __init__(self, transformer: Transformer, key):
        self.transformer = transformer
        self.cond_dim = 0

        # Initialise the transform parameters
        self.params = random.normal(key, shape=(transformer.num_params(1),))

    def transform(self, z: Array, condition=None):
        transform_args = self.transformer.get_args(self.params)
        return self.transformer.transform(z,  *transform_args)

    def transform_and_log_abs_det_jacobian(self, z: Array, condition=None):
        transform_args = self.transformer.get_args(self.params)
        x, log_abs_det = self.transformer.transform_and_log_abs_det_jacobian(
            z, *transform_args
        )
        return x, log_abs_det

    def inverse(self, x: Array, condition=None):
        transform_args = self.transformer.get_args(self.params)
        return self.transformer.inverse(x, *transform_args)

    def inverse_and_log_abs_det_jacobian(self, x: Array, condition=None):
        transform_args = self.transformer.get_args(self.params)
        return self.transformer.inverse_and_log_abs_det_jacobian(x, *transform_args)


class Fixed(Bijection):
    """
    A class implementing a Bijection for the 
    unconditional univariate case.
    """
    transformer: Transformer
    cond_dim: int
    _args: Array
    def __init__(self, transformer: Transformer, *args):
        self.transformer = transformer
        self.cond_dim = 0
        self._args = args

    @property
    def args(self):
        return jax.lax.stop_gradient(self._args)

    def transform(self, z: Array, condition=None):
        return self.transformer.transform(z, *self.args)

    def transform_and_log_abs_det_jacobian(self, z: Array, condition=None):
        x, log_abs_det = self.transformer.transform_and_log_abs_det_jacobian(
            z, *self.args
        )
        return x, log_abs_det

    def inverse(self, x: Array, condition=None):
        return self.transformer.inverse(x, *self.args)

    def inverse_and_log_abs_det_jacobian(self, x: Array, condition=None):
        return self.transformer.inverse_and_log_abs_det_jacobian(x, *self.args)