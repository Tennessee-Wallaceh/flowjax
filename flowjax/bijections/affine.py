"""Affine bijections."""

from collections.abc import Callable
from typing import ClassVar

import jax
import jax.nn as jnn
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
from jaxtyping import Array, ArrayLike, Shaped, PRNGKeyArray
import equinox as eqx

from flowjax.bijections.bijection import AbstractBijection
from flowjax.parameters import PositiveParameter, TriangularParameter
from flowjax.utils import arraylike_to_array





class Affine(AbstractBijection):
    r"""Elementwise affine transformation :math:`y = a \cdot x + b`.

    ``loc`` and ``scale`` should broadcast to the desired shape of the bijection.
    By default, we constrain the scale parameter to be postive using ``softplus``, but
    other parameterizations can be achieved by replacing the scale parameter after
    construction e.g. using ``eqx.tree_at``.

    Args:
        loc: Location parameter. Defaults to 0.
        scale: Scale parameter. Defaults to 1.
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    loc: Array
    scale: PositiveParameter

    def __init__(
        self,
        loc: ArrayLike = 0,
        scale: ArrayLike = 1,
    ):
        self.loc, scale = jnp.broadcast_arrays(
            *(arraylike_to_array(a, dtype=float) for a in (loc, scale)),
        )
        self.shape = scale.shape
        self.scale = PositiveParameter(scale)

    def transform_and_log_det(self, x, condition=None):
        scale = self.scale.value
        return x * scale + self.loc, jnp.log(jnp.abs(scale)).sum()

    def inverse_and_log_det(self, y, condition=None):
        scale = self.scale.value
        return (y - self.loc) / scale, -jnp.log(jnp.abs(scale)).sum()


class Loc(AbstractBijection):
    r"""Location transformation :math:`y = a \cdot x + b`.

    Args:
        loc: Scale parameter. Defaults to 1.
    """

    loc: Array
    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None

    def __init__(self, loc: ArrayLike):
        self.loc = arraylike_to_array(loc, dtype=float)
        self.shape = self.loc.shape

    def transform_and_log_det(self, x, condition=None):
        return x + self.loc, jnp.zeros(())

    def inverse_and_log_det(self, y, condition=None):
        return y - self.loc, jnp.zeros(())


class Scale(AbstractBijection):
    r"""Scale transformation :math:`y = a \cdot x`.

    Args:
        scale: Scale parameter. Defaults to 1.
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    scale: PositiveParameter

    def __init__(
        self,
        scale: ArrayLike,
    ):
        scale = arraylike_to_array(scale, "scale", dtype=float)
        self.scale = PositiveParameter(scale)
        self.shape = jnp.shape(scale)

    def transform_and_log_det(self, x, condition=None):
        scale = self.scale.value
        return x * scale, jnp.log(jnp.abs(scale)).sum()

    def inverse_and_log_det(self, y, condition=None):
        scale = self.scale.value
        return y / scale, -jnp.log(jnp.abs(scale)).sum()


class TriangularAffine(AbstractBijection):
    r"""A triangular affine transformation.

    Transformation has the form :math:`Ax + b`, where :math:`A` is a lower or upper
    triangular matrix, and :math:`b` is the bias vector. We assume the diagonal
    entries are positive, and constrain the values using softplus. Other
    parameterizations can be achieved by e.g. replacing ``self.triangular``
    after construction.

    Args:
        loc: Location parameter. If this is scalar, it is broadcast to the dimension
            inferred from arr.
        arr: Triangular matrix.
        lower: Whether the mask should select the lower or upper
            triangular matrix (other elements ignored). Defaults to True (lower).
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    loc: Array
    triangular: TriangularParameter
    lower: bool

    def __init__(
        self,
        loc: Shaped[ArrayLike, " #dim"],
        arr: Shaped[Array, "dim dim"],
        *,
        lower: bool = True,
    ):
        loc, arr = (arraylike_to_array(a, dtype=float) for a in (loc, arr))
        if (arr.ndim != 2) or (arr.shape[0] != arr.shape[1]):
            raise ValueError("arr must be a square, 2-dimensional matrix.")
        dim = arr.shape[0]
        self.triangular = TriangularParameter(arr, lower=lower)
        self.lower = lower
        self.shape = (dim,)
        self.loc = jnp.broadcast_to(loc, (dim,))

    def transform_and_log_det(self, x, condition=None):
        triangular = self.triangular.value
        y = triangular @ x + self.loc
        return y, jnp.log(jnp.abs(jnp.diag(triangular))).sum()

    def inverse_and_log_det(self, y, condition=None):
        triangular = self.triangular.value
        x = solve_triangular(triangular, y - self.loc, lower=self.lower)
        return x, -jnp.log(jnp.abs(jnp.diag(triangular))).sum()


class AdditiveCondition(AbstractBijection):
    """Given a callable ``f``, carries out the transformation ``y = x + f(condition)``.

    If used to transform a distribution, this allows the "location" to be changed as a
    function of the conditioning variables. Note that the callable can be a callable
    module with trainable parameters.

    Args:
        module: A callable (e.g. a function or callable module) that maps array with
            shape cond_shape, to a shape that is broadcastable with the shape of the
            bijection.
        shape: The shape of the bijection.
        cond_shape: The condition shape of the bijection.

    Example:
        Conditioning using a linear transformation

        .. doctest::

            >>> from flowjax.bijections import AdditiveCondition
            >>> from equinox.nn import Linear
            >>> import jax.numpy as jnp
            >>> import jax.random as jr
            >>> bijection = AdditiveCondition(
            ...     Linear(2, 3, key=jr.key(0)), shape=(3,), cond_shape=(2,)
            ...     )
            >>> y = bijection.transform(jnp.ones(3), condition=jnp.ones(2))

    """

    shape: tuple[int, ...]
    cond_shape: tuple[int, ...]
    module: Callable[[ArrayLike], ArrayLike]

    def __init__(
        self,
        module: Callable[[ArrayLike], ArrayLike],
        shape: tuple[int, ...],
        cond_shape: tuple[int, ...],
    ):
        self.module = module
        self.shape = shape
        self.cond_shape = cond_shape

    def transform_and_log_det(self, x, condition=None):
        return x + self.module(condition), jnp.zeros(())

    def inverse_and_log_det(self, y, condition=None):
        return y - self.module(condition), jnp.zeros(())


class LULinear(AbstractBijection):
    shape: tuple[int, ...]
    cond_shape = None

    lower_entries: Array
    upper_entries: Array
    unconstrained_upper_diag: Array
    bias: Array
    eps: float = eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        dim: int,
        *,
        identity_init: bool = True,
        eps: float = 1e-3,
    ):
        self.shape = (dim,)
        self.eps = eps

        n_triangular_entries = dim * (dim - 1) // 2

        if identity_init:
            self.lower_entries = jnp.zeros(n_triangular_entries)
            self.upper_entries = jnp.zeros(n_triangular_entries)
            self.unconstrained_upper_diag = jnp.full(
                dim,
                jnp.log(jnp.expm1(1.0 - eps)),
            )
        else:
            lower_key, upper_key, diag_key = jax.random.split(key, 3)
            bound = 1.0 / jnp.sqrt(dim)
            self.lower_entries = jax.random.uniform(
                lower_key,
                (n_triangular_entries,),
                minval=-bound,
                maxval=bound,
            )
            self.upper_entries = jax.random.uniform(
                upper_key,
                (n_triangular_entries,),
                minval=-bound,
                maxval=bound,
            )
            self.unconstrained_upper_diag = jax.random.uniform(
                diag_key,
                (dim,),
                minval=-bound,
                maxval=bound,
            )

        self.bias = jnp.zeros(dim)

    @property
    def lower_indices(self) -> tuple[Array, Array]:
        return jnp.tril_indices(self.shape[0], k=-1)

    @property
    def upper_indices(self) -> tuple[Array, Array]:
        return jnp.triu_indices(self.shape[0], k=1)

    @property
    def diag_indices(self) -> tuple[Array, Array]:
        return jnp.diag_indices(self.shape[0])
    
    @property
    def upper_diag(self):
        return jnn.softplus(self.unconstrained_upper_diag) + self.eps

    def _lower_upper(self):
        dim = self.shape[0]

        lower = jnp.zeros((dim, dim))
        lower = lower.at[self.lower_indices].set(self.lower_entries)
        lower = lower.at[self.diag_indices].set(1.0)

        upper = jnp.zeros((dim, dim))
        upper = upper.at[self.upper_indices].set(self.upper_entries)
        upper = upper.at[self.diag_indices].set(self.upper_diag)

        return lower, upper

    def weight(self):
        lower, upper = self._lower_upper()
        return lower @ upper

    def transform_and_log_det(self, x, condition=None):
        lower, upper = self._lower_upper()
        y = lower @ (upper @ x) + self.bias
        log_det = jnp.sum(jnp.log(self.upper_diag))
        return y, log_det

    def inverse_and_log_det(self, y, condition=None):
        lower, upper = self._lower_upper()
        z = y - self.bias
        z = jax.scipy.linalg.solve_triangular(
            lower,
            z,
            lower=True,
            unit_diagonal=True,
        )
        x = jax.scipy.linalg.solve_triangular(
            upper,
            z,
            lower=False,
        )
        log_det = -jnp.sum(jnp.log(self.upper_diag))
        return x, log_det


class UnitLULinear(AbstractBijection):
    shape: tuple[int, ...]
    cond_shape = None

    lower_entries: Array
    upper_entries: Array
    unconstrained_upper_diag: Array
    bias: Array
    eps: float = eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        dim: int,
        *,
        identity_init: bool = True,
        eps: float = 1e-3,
    ):
        self.shape = (dim,)
        self.eps = eps

        n_triangular_entries = dim * (dim - 1) // 2

        if identity_init:
            self.lower_entries = jnp.zeros(n_triangular_entries)
            self.upper_entries = jnp.zeros(n_triangular_entries)
            self.unconstrained_upper_diag = jnp.full(
                dim,
                jnp.log(jnp.expm1(1.0 - eps)),
            )
        else:
            lower_key, upper_key, diag_key = jax.random.split(key, 3)
            bound = 1.0 / jnp.sqrt(dim)
            self.lower_entries = jax.random.uniform(
                lower_key,
                (n_triangular_entries,),
                minval=-bound,
                maxval=bound,
            )
            self.upper_entries = jax.random.uniform(
                upper_key,
                (n_triangular_entries,),
                minval=-bound,
                maxval=bound,
            )
            self.unconstrained_upper_diag = jax.random.uniform(
                diag_key,
                (dim,),
                minval=-bound,
                maxval=bound,
            )

        self.bias = jnp.zeros(dim)

    @property
    def lower_indices(self) -> tuple[Array, Array]:
        return jnp.tril_indices(self.shape[0], k=-1)

    @property
    def upper_indices(self) -> tuple[Array, Array]:
        return jnp.triu_indices(self.shape[0], k=1)

    @property
    def diag_indices(self) -> tuple[Array, Array]:
        return jnp.diag_indices(self.shape[0])
    
    @property
    def upper_diag(self):
        upper_diag = jnn.softplus(self.unconstrained_upper_diag) + self.eps
        log_upper_diag = jnp.log(upper_diag)
        return jnp.exp(log_upper_diag - jnp.mean(log_upper_diag))

    def _lower_upper(self):
        dim = self.shape[0]

        lower = jnp.zeros((dim, dim))
        lower = lower.at[self.lower_indices].set(self.lower_entries)
        lower = lower.at[self.diag_indices].set(1.0)

        upper = jnp.zeros((dim, dim))
        upper = upper.at[self.upper_indices].set(self.upper_entries)
        upper = upper.at[self.diag_indices].set(self.upper_diag)

        return lower, upper

    def weight(self):
        lower, upper = self._lower_upper()
        return lower @ upper

    def transform_and_log_det(self, x, condition=None):
        lower, upper = self._lower_upper()
        y = lower @ (upper @ x) + self.bias
        log_det = jnp.zeros((), dtype=x.dtype)
        return y, log_det

    def inverse_and_log_det(self, y, condition=None):
        lower, upper = self._lower_upper()
        z = y - self.bias
        z = jax.scipy.linalg.solve_triangular(
            lower,
            z,
            lower=True,
            unit_diagonal=True,
        )
        x = jax.scipy.linalg.solve_triangular(
            upper,
            z,
            lower=False,
        )
        log_det = jnp.zeros((), dtype=y.dtype)
        return x, log_det