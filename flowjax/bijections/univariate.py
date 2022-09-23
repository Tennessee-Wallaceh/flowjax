from flowjax.bijections.abc import Transformer
import jax.numpy as jnp
from flowjax.bijections.abc import Bijection
from flowjax.utils import Array

class Univariate(Bijection):
    """
    A class implementing a Bijection for the 
    unconditional univariate case.
    """
    transformer: Transformer
    cond_dim: int
    dim: int
    params: Array
    def __init__(self, transformer: Transformer, dim: int):
        self.transformer = transformer
        self.cond_dim = 0
        self.dim = dim

        # Initialise the transform parameters
        self.params = jnp.zeros(transformer.num_params(self.dim))

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
        return self.bijection.inverse(x, *transform_args)

    def inverse_and_log_abs_det_jacobian(self, x: Array, condition=None):
        transform_args = self.transformer.get_args(self.params)
        return self.transformer.inverse_and_log_abs_det_jacobian(x, *transform_args)