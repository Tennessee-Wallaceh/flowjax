"""Utilities and typing protocols for explicit parameterizations."""

from __future__ import annotations

from typing import Any, Protocol, TypeVar, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
from jaxtyping import Array

TLatent = TypeVar("TLatent")
TValue = TypeVar("TValue")


def _is_tracer(x: Array) -> bool:
    return isinstance(x, jax.core.Tracer)


@runtime_checkable
class Parameterization(Protocol[TLatent, TValue]):
    """Protocol for explicit constrained parameter handling.

    Implementations must expose:
    - ``latent``: the underlying state.
    - ``value``: a deterministic value computed from latent state.
    """

    latent: TLatent

    @property
    def value(self) -> TValue:
        """Return the deterministic value associated with ``latent``."""


def parameterize(pytree: Any):
    """Materialize parameterizations across a pytree.

    Leaves implementing :class:`Parameterization` are preserved.
    """

    return tree_map(lambda leaf: leaf, pytree, is_leaf=_is_parameter_leaf)


def _is_parameter_leaf(leaf) -> bool:
    return isinstance(leaf, Parameterization)


def non_trainable(pytree: Any):
    """Apply stop-gradient to all inexact array leaves in a pytree."""

    return tree_map(
        lambda leaf: jax.lax.stop_gradient(leaf) if eqx.is_inexact_array(leaf) else leaf,
        pytree,
    )


class PositiveParameter(eqx.Module):
    """A positive parameterization using softplus."""

    latent: Array
    min_value: float

    def __init__(self, value: Array, *, min_value: float = 0.0):
        value = jnp.asarray(value, dtype=float)
        if (not _is_tracer(value)) and jnp.any(value <= min_value):
            raise ValueError(
                "PositiveParameter expected all values to be greater than min_value "
                f"({min_value}); got minimum value {jnp.min(value)}.",
            )
        self.latent = _inv_softplus(value - min_value)
        self.min_value = min_value

    @property
    def value(self) -> Array:
        latent = self.latent
        return jax.nn.softplus(latent) + self.min_value


class TriangularParameter(eqx.Module):
    """A triangular matrix parameterization with positive diagonal entries."""

    latent: Array
    lower: bool

    def __init__(self, matrix: Array, *, lower: bool = True):
        matrix = jnp.asarray(matrix, dtype=float)
        if matrix.shape[-1] != matrix.shape[-2]:
            raise ValueError("TriangularParameter expected a square matrix.")
        diag = jnp.diagonal(matrix, axis1=-2, axis2=-1)
        if (not _is_tracer(diag)) and jnp.any(diag <= 0):
            raise ValueError(
                "TriangularParameter expected strictly positive diagonal entries; "
                f"got minimum diagonal value {jnp.min(diag)}.",
            )

        self.lower = lower
        self.latent = self._replace_diagonal(matrix, _inv_softplus(diag))

    @property
    def value(self) -> Array:
        latent = self.latent
        triangular = jnp.tril(latent) if self.lower else jnp.triu(latent)
        diag = jnp.diagonal(triangular, axis1=-2, axis2=-1)
        positive_diag = jax.nn.softplus(diag)
        return self._replace_diagonal(triangular, positive_diag)

    def _replace_diagonal(self, matrix: Array, diagonal: Array) -> Array:
        dim = matrix.shape[-1]
        eye = jnp.eye(dim, dtype=matrix.dtype)
        return matrix * (1 - eye) + diagonal[..., None] * eye


class UnitVectorParameter(eqx.Module):
    """A parameterization mapping vectors to the unit sphere."""

    latent: Array

    def __init__(self, value: Array):
        value = jnp.asarray(value, dtype=float)
        if value.ndim != 1:
            raise ValueError("UnitVectorParameter expected a 1-dimensional vector.")
        if (not _is_tracer(value)) and jnp.all(value == 0):
            raise ValueError("UnitVectorParameter expected a non-zero vector.")
        self.latent = value

    @property
    def value(self) -> Array:
        latent = self.latent
        return latent / jnp.linalg.norm(latent)


class IncreasingIntervalParameter(eqx.Module):
    """A parameterization for increasing positions on a closed interval."""

    latent: Array
    interval: tuple[float, float]
    min_width: float

    def __init__(
        self,
        latent: Array,
        *,
        interval: tuple[int | float, int | float],
        min_width: float | int = 1e-3,
    ):
        latent = jnp.asarray(latent, dtype=float)
        if latent.ndim != 1:
            raise ValueError(
                "IncreasingIntervalParameter expected a 1-dimensional array."
            )
        if latent.shape[0] < 1:
            raise ValueError("IncreasingIntervalParameter expected at least one bin.")

        lo, hi = float(interval[0]), float(interval[1])
        if not hi > lo:
            raise ValueError(
                "IncreasingIntervalParameter expected interval[1] > interval[0]."
            )

        num_bins = latent.shape[0]
        min_width = float(min_width)
        if (hi - lo) <= (num_bins * min_width):
            raise ValueError(
                "Interval too small for min_width constraint. "
                f"Interval width is {hi - lo}, but requires > {num_bins * min_width}.",
            )

        self.latent = latent
        self.interval = (lo, hi)
        self.min_width = min_width

    @property
    def value(self) -> Array:
        latent = self.latent

        lo, hi = self.interval
        num_bins = latent.shape[0]
        widths = jax.nn.softmax(latent) * (hi - lo - num_bins * self.min_width)
        widths = widths + self.min_width
        cumsum = jnp.cumsum(widths, axis=-1)
        return lo + jnp.concatenate(
            (jnp.zeros((1,), dtype=latent.dtype), cumsum),
            axis=-1,
        )


class LogSimplexParameter(eqx.Module):
    """A parameterization returning log-normalized positive weights."""

    latent: Array

    def __init__(self, weights: Array):
        weights = jnp.asarray(weights, dtype=float)
        if (not _is_tracer(weights)) and jnp.any(weights <= 0):
            raise ValueError("LogSimplexParameter expected strictly positive weights.")
        self.latent = jnp.log(weights)

    @property
    def value(self) -> Array:
        latent = self.latent
        return jax.nn.log_softmax(latent)


class MaskedWeightParameter(eqx.Module):
    """Masked weight parameterization for autoregressive networks."""

    latent: Array
    mask: Array

    def __init__(self, weights: Array, mask: Array):
        weights = jnp.asarray(weights)
        mask = jnp.asarray(mask)
        if weights.shape != mask.shape:
            raise ValueError(
                "MaskedWeightParameter expected weights and mask to have the same shape; "
                f"got {weights.shape} and {mask.shape}.",
            )
        if mask.dtype != jnp.bool_:
            if not jnp.all((mask == 0) | (mask == 1)):
                raise ValueError(
                    "MaskedWeightParameter expected a boolean mask or a {0, 1} mask."
                )
            mask = mask.astype(bool)
        self.latent = weights
        self.mask = mask

    @property
    def value(self) -> Array:
        latent = self.latent
        return jnp.where(self.mask, latent, 0)

def _inv_softplus(x: Array) -> Array:
    """Inverse softplus for positive x."""
    return x + jnp.log(-jnp.expm1(-x))
