from flowjax.utils import Array
from typing import List
from abc import ABC, abstractmethod

from equinox import Module

class Transformer(ABC, Module):
    """Bijection which facilitates parameterisation with a neural network output
    (e.g. as in coupling flows, or masked autoressive flows). Should not contain
    (directly) trainable parameters."""

    @abstractmethod
    def transform(self, x: Array, *args: Array) -> Array:
        """Apply transformation."""

    @abstractmethod
    def transform_and_log_abs_det_jacobian(self, x: Array, *args: Array) -> tuple:
        """Apply transformation and compute log absolute value of the Jacobian determinant."""

    @abstractmethod
    def inverse(self, y: Array, *args: Array) -> Array:
        """Invert the transformation."""

    def inverse_and_log_abs_det_jacobian(self, x: Array, *args: Array) -> tuple:
        """Invert the transformation and compute the log absolute value of the Jacobian determinant."""

    @abstractmethod
    def num_params(self, dim: int) -> int:
        "Total number of parameters required for bijection."

    @abstractmethod
    def get_ranks(self, dim: int) -> Array:
        "The ranks of the parameters, i.e. which dimension of the input the parameters correspond to."

    @abstractmethod
    def get_args(self, params: Array) -> List[Array]:
        "Transform unconstrained vector of params (e.g. nn output) into args for transformation."