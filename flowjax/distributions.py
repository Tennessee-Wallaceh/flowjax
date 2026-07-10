"""Distributions from flowjax.distributions."""

import inspect
from abc import abstractmethod
from collections.abc import Callable
from functools import wraps
from math import prod
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from equinox import AbstractVar, AbstractClassVar
from jax import dtypes
from jax.numpy import linalg
from jax.scipy import stats as jstats
from jax.scipy.special import logsumexp
from jax.tree_util import tree_map
from jaxtyping import Array, ArrayLike, PRNGKeyArray, Shaped
from flowjax.parameters import non_trainable

from flowjax.bijections import (
    AbstractBijection,
    Affine,
    Chain,
    Exp,
    Scale,
    TriangularAffine,
)
from flowjax.parameters import LogSimplexParameter, PositiveParameter
from flowjax.utils import (
    arraylike_to_array,
    merge_cond_shapes,
)

class AbstractDistribution(eqx.Module):
    """Abstract distribution class.

    Distributions are registered as JAX PyTrees (as they are Equinox modules), and as
    such are compatible with normal JAX operations.

    Methods act on a single sample and, for conditional distributions, a single
    conditioning variable. Use ``jax.vmap`` or ``eqx.filter_vmap`` to explicitly
    vectorise operations over additional batch dimensions.

    Concrete subclasses can be implemented as follows:

    - Inherit from ``AbstractDistribution``.
    - Define the abstract attributes ``shape`` and ``cond_shape``.
      ``cond_shape`` should be ``None`` for unconditional distributions.
    - Implement ``_sample``, which returns a single sample with shape ``shape``.
    - Implement ``_log_prob``, which returns the scalar log probability of a
      single sample.

    Stateful distributions may additionally override ``_sample_with_state`` and
    ``_sample_and_log_prob_with_state``.

    Attributes:
        shape: Shape of a single sample from the distribution.
        cond_shape: Shape of a single conditioning variable, or ``None`` for
            unconditional distributions.
    """

    shape: AbstractVar[tuple[int, ...]]
    cond_shape: AbstractVar[tuple[int, ...] | None]

    @abstractmethod
    def _log_prob(
        self,
        x: Array,
        condition: Array | None = None,
    ) -> Array:
        """Evaluate the log probability of a single sample."""

    def _log_prob_with_state(
        self,
        x: Array,
        condition: Array | None = None,
        *,
        key: PRNGKeyArray | None = None,
        state: eqx.nn.State | None = None,
        inference: bool = True,
    ) -> tuple[Array, eqx.nn.State | None]:
        """Evaluate log probability with optional key/state support.

        Stateless distributions should not need to override this.
        """
        del key, inference
        return self._log_prob(x, condition), state

    @abstractmethod
    def _sample(
        self,
        key: PRNGKeyArray,
        condition: Array | None = None,
    ) -> Array:
        """Draw a single sample from the distribution."""

    def _sample_with_state(
        self,
        key: PRNGKeyArray,
        condition: Array | None = None,
        *,
        state: eqx.nn.State | None = None,
        inference: bool = True,
    ) -> tuple[Array, eqx.nn.State | None]:
        """Draw a single sample with optional state support.

        Stateless distributions should not need to override this.
        """
        del inference
        return self._sample(key, condition), state

    def _sample_and_log_prob(
        self,
        key: PRNGKeyArray,
        condition: Array | None = None,
    ) -> tuple[Array, Array]:
        """Draw a single sample and return its log probability."""
        x = self._sample(key, condition)
        return x, self._log_prob(x, condition)

    def _sample_and_log_prob_with_state(
        self,
        key: PRNGKeyArray,
        condition: Array | None = None,
        *,
        state: eqx.nn.State | None = None,
        inference: bool = True,
    ) -> tuple[tuple[Array, Array], eqx.nn.State | None]:
        """Draw a sample and return its log probability with state support."""
        sample_key, log_prob_key = jr.split(key)

        x, state = self._sample_with_state(
            sample_key,
            condition,
            state=state,
            inference=inference,
        )
        log_prob, state = self._log_prob_with_state(
            x,
            condition,
            key=log_prob_key,
            state=state,
            inference=inference,
        )

        return (x, log_prob), state

    def log_prob(
        self,
        x: ArrayLike,
        condition: ArrayLike | None = None,
    ) -> Array:
        """Evaluate the log probability of a single sample.

        Args:
            x: Sample with shape matching ``distribution.shape``.
            condition: Conditioning variable with shape matching
                ``distribution.cond_shape``. Required for conditional distributions.

        Returns:
            Scalar log probability.
        """
        x = arraylike_to_array(x, err_name="x", dtype=float)
        self._check_shape("x", x, self.shape)

        condition = self._process_condition(condition)
        return self._log_prob(x, condition)

    def log_prob_with_state(
        self,
        x: ArrayLike,
        condition: ArrayLike | None = None,
        key: PRNGKeyArray | None = None,
        state: eqx.nn.State | None = None,
        *,
        inference: bool = True,
    ) -> tuple[Array, eqx.nn.State | None]:
        """Evaluate log probability with optional key/state support."""
        x = arraylike_to_array(x, err_name="x", dtype=float)
        self._check_shape("x", x, self.shape)

        condition = self._process_condition(condition)

        return self._log_prob_with_state(
            x,
            condition,
            key=key,
            state=state,
            inference=inference,
        )

    def sample(
        self,
        key: PRNGKeyArray,
        condition: ArrayLike | None = None,
    ) -> Array:
        """Draw a single sample from the distribution.

        Args:
            key: JAX random key.
            condition: Conditioning variable with shape matching
                ``distribution.cond_shape``. Required for conditional distributions.

        Returns:
            Sample with shape matching ``distribution.shape``.
        """
        self._check_key(key)
        condition = self._process_condition(condition)

        return self._sample(key, condition)

    def sample_with_state(
        self,
        key: PRNGKeyArray,
        condition: ArrayLike | None = None,
        *,
        state: eqx.nn.State | None = None,
        inference: bool = True,
    ) -> tuple[Array, eqx.nn.State | None]:
        """Draw a single sample with optional state support.

        Args:
            key: JAX random key.
            condition: Conditioning variable with shape matching
                ``distribution.cond_shape``. Required for conditional distributions.
            state: Optional Equinox state.
            inference: Whether to run in inference/evaluation mode.

        Returns:
            The sample and updated state.
        """
        self._check_key(key)
        condition = self._process_condition(condition)

        return self._sample_with_state(
            key,
            condition,
            state=state,
            inference=inference,
        )

    def sample_and_log_prob(
        self,
        key: PRNGKeyArray,
        condition: ArrayLike | None = None,
    ) -> tuple[Array, Array]:
        """Draw a single sample and return its log probability.

        For transformed distributions, especially flows, this will generally be more
        efficient than calling ``sample`` and ``log_prob`` separately.

        Args:
            key: JAX random key.
            condition: Conditioning variable with shape matching
                ``distribution.cond_shape``. Required for conditional distributions.

        Returns:
            The sample and its scalar log probability.
        """
        self._check_key(key)
        condition = self._process_condition(condition)

        return self._sample_and_log_prob(key, condition)

    def sample_and_log_prob_with_state(
        self,
        key: PRNGKeyArray,
        condition: ArrayLike | None = None,
        *,
        state: eqx.nn.State | None = None,
        inference: bool = True,
    ) -> tuple[tuple[Array, Array], eqx.nn.State | None]:
        """Draw a single sample and return its log probability with state support.

        Args:
            key: JAX random key.
            condition: Conditioning variable with shape matching
                ``distribution.cond_shape``. Required for conditional distributions.
            state: Optional Equinox state.
            inference: Whether to run in inference/evaluation mode.

        Returns:
            The sample and its scalar log probability, together with updated state.
        """
        self._check_key(key)
        condition = self._process_condition(condition)

        return self._sample_and_log_prob_with_state(
            key,
            condition,
            state=state,
            inference=inference,
        )

    def _process_condition(
        self,
        condition: ArrayLike | None,
    ) -> Array | None:
        if self.cond_shape is None:
            if condition is not None:
                raise ValueError(
                    "Cannot pass condition to an unconditional distribution."
                )
            return None

        if condition is None:
            raise ValueError("Condition required for a conditional distribution.")

        condition = arraylike_to_array(
            condition,
            err_name="condition",
            dtype=float,
        )
        self._check_shape("condition", condition, self.cond_shape)

        return condition

    @staticmethod
    def _check_shape(
        name: str,
        array: Array,
        expected_shape: tuple[int, ...],
    ) -> None:
        if array.shape != expected_shape:
            raise ValueError(
                f"Expected {name} to have shape {expected_shape}; got {array.shape}."
            )

    @staticmethod
    def _check_key(key: PRNGKeyArray) -> None:
        if not dtypes.issubdtype(key.dtype, dtypes.prng_key):
            raise TypeError("New-style typed JAX PRNG keys required.")

    @property
    def ndim(self) -> int:
        """Number of dimensions in the distribution; the length of ``shape``."""
        return len(self.shape)

    @property
    def cond_ndim(self) -> None | int:
        """Number of dimensions of the conditioning variable."""
        return None if self.cond_shape is None else len(self.cond_shape)


class AbstractTransformed(AbstractDistribution):
    """Abstract class representing transformed distributions.

    The forward bijection is used for sampling and the inverse bijection for density
    evaluation. Concrete implementations should subclass ``AbstractTransformed`` and
    define the abstract attributes ``base_dist`` and ``bijection``.

    .. warning::
        It is the user's responsibility to ensure the bijection is valid across the
        entire support of the distribution. Failure to do so may result in non-finite
        values or an incorrectly normalized density.

    Attributes:
        base_dist: The base distribution.
        bijection: The transformation to apply.
    """

    base_dist: AbstractVar[AbstractDistribution]
    bijection: AbstractVar[AbstractBijection]

    def __check_init__(self):
        """Check for compatible shapes between base distribution and bijection."""
        if (
            self.base_dist.cond_shape is not None
            and self.bijection.cond_shape is not None
            and self.base_dist.cond_shape != self.bijection.cond_shape
        ):
            raise ValueError(
                "The base distribution and bijection are both conditional "
                "but have mismatched cond_shape attributes. Base distribution has "
                f"{self.base_dist.cond_shape}, and the bijection has "
                f"{self.bijection.cond_shape}."
            )

        if self.base_dist.shape != self.bijection.shape:
            raise ValueError(
                "The base distribution and bijection have mismatched shapes. "
                f"Base distribution has {self.base_dist.shape}, and the bijection "
                f"has {self.bijection.shape}."
            )

    def _log_prob(
        self,
        x: Array,
        condition: Array | None = None,
    ) -> Array:
        z, log_abs_det = self.bijection.inverse_and_log_det(x, condition)
        log_prob = self.base_dist._log_prob(z, condition) + log_abs_det

        # If log_prob is nan, assume x lies outside the transform support.
        return jnp.where(jnp.isnan(log_prob), -jnp.inf, log_prob)

    def _log_prob_with_state(
        self,
        x: Array,
        condition: Array | None = None,
        *,
        key: PRNGKeyArray | None = None,
        state: eqx.nn.State | None = None,
        inference: bool = True,
    ) -> tuple[Array, eqx.nn.State | None]:
        if key is None:
            bijection_key = None
            base_key = None
        else:
            bijection_key, base_key = jr.split(key)

        (z, log_abs_det), state = self.bijection.inverse_and_log_det_with_state(
            x,
            condition,
            key=bijection_key,
            state=state,
            inference=inference,
        )
        base_log_prob, state = self.base_dist._log_prob_with_state(
            z,
            condition,
            key=base_key,
            state=state,
            inference=inference,
        )

        log_prob = base_log_prob + log_abs_det

        # If log_prob is nan, assume x lies outside the transform support.
        return jnp.where(jnp.isnan(log_prob), -jnp.inf, log_prob), state

    def _sample(
        self,
        key: PRNGKeyArray,
        condition: Array | None = None,
    ) -> Array:
        base_sample = self.base_dist._sample(key, condition)
        return self.bijection.transform(base_sample, condition)

    def _sample_with_state(
        self,
        key: PRNGKeyArray,
        condition: Array | None = None,
        *,
        state: eqx.nn.State | None = None,
        inference: bool = True,
    ) -> tuple[Array, eqx.nn.State | None]:
        base_key, bijection_key = jr.split(key)

        base_sample, state = self.base_dist._sample_with_state(
            base_key,
            condition,
            state=state,
            inference=inference,
        )
        sample, state = self.bijection.transform_with_state(
            base_sample,
            condition,
            key=bijection_key,
            state=state,
            inference=inference,
        )
        return sample, state

    def _sample_and_log_prob(
        self,
        key: PRNGKeyArray,
        condition: Array | None = None,
    ) -> tuple[Array, Array]:
        # Override to avoid computing the inverse transformation.
        base_sample, base_log_prob = self.base_dist._sample_and_log_prob(
            key,
            condition,
        )
        sample, forward_log_det = self.bijection.transform_and_log_det(
            base_sample,
            condition,
        )
        return sample, base_log_prob - forward_log_det

    def _sample_and_log_prob_with_state(
        self,
        key: PRNGKeyArray,
        condition: Array | None = None,
        *,
        state: eqx.nn.State | None = None,
        inference: bool = True,
    ) -> tuple[tuple[Array, Array], eqx.nn.State | None]:
        base_key, bijection_key = jr.split(key)

        (base_sample, base_log_prob), state = (
            self.base_dist._sample_and_log_prob_with_state(
                base_key,
                condition,
                state=state,
                inference=inference,
            )
        )

        (sample, forward_log_det), state = (
            self.bijection.transform_and_log_det_with_state(
                base_sample,
                condition,
                key=bijection_key,
                state=state,
                inference=inference,
            )
        )

        return (sample, base_log_prob - forward_log_det), state

    def merge_transforms(self):
        """Unnest nested transformed distributions.

        Returns an equivalent distribution with nested transformed distributions
        unravelled, such that the returned base distribution is not itself an
        ``AbstractTransformed`` instance.
        """
        if not isinstance(self.base_dist, AbstractTransformed):
            return self

        base_dist = self.base_dist
        bijections = [self.bijection]

        while isinstance(base_dist, AbstractTransformed):
            bijections.append(base_dist.bijection)
            base_dist = base_dist.base_dist

        bijection = Chain(list(reversed(bijections))).merge_chains()
        return Transformed(base_dist, bijection)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.base_dist.shape

    @property
    def cond_shape(self) -> tuple[int, ...] | None:
        return merge_cond_shapes(
            (self.bijection.cond_shape, self.base_dist.cond_shape)
        )

    
class Transformed(AbstractTransformed):
    """Form a distribution like object using a base distribution and a bijection.

    We take the forward bijection for use in sampling, and the inverse
    bijection for use in density evaluation.

    .. warning::
            It is the users responsibility to ensure the bijection is valid across the
            entire support of the distribution. Failure to do so may result in
            non-finite values or incorrectly normalized densities.

    Args:
        base_dist: Base distribution.
        bijection: Bijection to transform distribution.

    Example:
        .. doctest::

            >>> from flowjax.distributions import StandardNormal, Transformed
            >>> from flowjax.bijections import Affine
            >>> normal = StandardNormal()
            >>> bijection = Affine(1)
            >>> transformed = Transformed(normal, bijection)
    """

    base_dist: AbstractDistribution
    bijection: AbstractBijection

    # manual init because Pylance doesn't understand AbstractVar
    def __init__(self, base_dist: AbstractDistribution, bijection: AbstractBijection):
        self.base_dist = base_dist
        self.bijection = bijection


class AbstractLocScaleDistribution(AbstractTransformed):
    """Abstract distribution class for affine transformed distributions."""

    base_dist: AbstractVar[AbstractDistribution]
    bijection: AbstractVar[Affine]

    @property
    def loc(self):
        """Location of the distribution."""
        return self.bijection.loc

    @property
    def scale(self):
        """Scale of the distribution."""
        return self.bijection.scale.value


class StandardNormal(AbstractDistribution):
    """Standard normal distribution.

    Note unlike :class:`Normal`, this has no trainable parameters.

    Args:
        shape: The shape of the distribution. Defaults to ().
    """

    shape: tuple[int, ...] = ()
    cond_shape: ClassVar[None] = None

    def _log_prob(self, x, condition=None):
        return jstats.norm.logpdf(x).sum()

    def _sample(self, key, condition=None):
        return jr.normal(key, self.shape)


class Normal(AbstractLocScaleDistribution):
    """An independent Normal distribution with mean and std for each dimension.

    ``loc`` and ``scale`` should broadcast to the desired shape of the distribution.

    Args:
        loc: Means. Defaults to 0. Defaults to 0.
        scale: Standard deviations. Defaults to 1.
    """

    base_dist: StandardNormal
    bijection: Affine

    def __init__(self, loc: ArrayLike = 0, scale: ArrayLike = 1):
        self.base_dist = StandardNormal(
            jnp.broadcast_shapes(jnp.shape(loc), jnp.shape(scale)),
        )
        self.bijection = Affine(loc=loc, scale=scale)


class LogNormal(AbstractTransformed):
    """Log normal distribution.

    ``loc`` and ``scale`` here refers to the underlying normal distribution.

    Args:
        loc: Location paramter. Defaults to 0.
        scale: Scale parameter. Defaults to 1.
    """

    base_dist: Normal
    bijection: Exp

    def __init__(self, loc: ArrayLike = 0, scale: ArrayLike = 1):
        self.base_dist = Normal(loc, scale)
        self.bijection = Exp(self.base_dist.shape)


class MultivariateNormal(AbstractTransformed):
    """Multivariate normal distribution.

    Internally this is parameterised using the Cholesky decomposition of the covariance
    matrix.

    Args:
        loc: The location/mean parameter vector. If this is scalar it is broadcast to
            the dimension implied by the covariance matrix.
        covariance: Covariance matrix.
    """

    base_dist: StandardNormal
    bijection: TriangularAffine

    def __init__(
        self,
        loc: Shaped[ArrayLike, "#dim"],
        covariance: Shaped[Array, "dim dim"],
    ):
        self.bijection = TriangularAffine(loc, linalg.cholesky(covariance))
        self.base_dist = StandardNormal(self.bijection.shape)

    @property
    def loc(self):
        """Location (mean) of the distribution."""
        return self.bijection.loc

    @property
    def covariance(self):
        """The covariance matrix."""
        cholesky = self.bijection.triangular.value
        return cholesky @ cholesky.T


class _StandardUniform(AbstractDistribution):
    r"""Standard Uniform distribution."""

    shape: tuple[int, ...] = ()
    cond_shape: ClassVar[None] = None

    def _log_prob(self, x, condition=None):
        return jstats.uniform.logpdf(x).sum()

    def _sample(self, key, condition=None):
        return jr.uniform(key, shape=self.shape)


class Uniform(AbstractLocScaleDistribution):
    """Uniform distribution.

    ``minval`` and ``maxval`` should broadcast to the desired distribution shape.

    Args:
        minval: Minimum values.
        maxval: Maximum values.
    """

    base_dist: _StandardUniform
    bijection: Affine

    def __init__(self, minval: ArrayLike, maxval: ArrayLike):
        shape = jnp.broadcast_shapes(jnp.shape(minval), jnp.shape(maxval))
        minval, maxval = eqx.error_if(
            (minval, maxval), maxval <= minval, "minval must be less than the maxval."
        )
        self.base_dist = _StandardUniform(shape)
        self.bijection = non_trainable(Affine(loc=minval, scale=maxval - minval))

    @property
    def minval(self):
        """Minimum value of the uniform distribution."""
        return self.bijection.loc

    @property
    def maxval(self):
        """Maximum value of the uniform distribution."""
        return self.bijection.loc + self.bijection.scale.value


class _StandardGumbel(AbstractDistribution):
    """Standard gumbel distribution."""

    shape: tuple[int, ...] = ()
    cond_shape: ClassVar[None] = None

    def _log_prob(self, x, condition=None):
        return -(x + jnp.exp(-x)).sum()

    def _sample(self, key, condition=None):
        return jr.gumbel(key, shape=self.shape)


class Gumbel(AbstractLocScaleDistribution):
    """Gumbel distribution.

    ``loc`` and ``scale`` should broadcast to the dimension of the distribution.

    Args:
        loc: Location paramter. Defaults to 0.
        scale: Scale parameter. Defaults to 1.
    """

    base_dist: _StandardGumbel
    bijection: Affine

    def __init__(self, loc: ArrayLike = 0, scale: ArrayLike = 1):
        self.base_dist = _StandardGumbel(
            jnp.broadcast_shapes(jnp.shape(loc), jnp.shape(scale)),
        )
        self.bijection = Affine(loc, scale)


class _StandardCauchy(AbstractDistribution):
    """Implements standard cauchy distribution (loc=0, scale=1)."""

    shape: tuple[int, ...] = ()
    cond_shape: ClassVar[None] = None

    def _log_prob(self, x, condition=None):
        return jstats.cauchy.logpdf(x).sum()

    def _sample(self, key, condition=None):
        return jr.cauchy(key, shape=self.shape)


class Cauchy(AbstractLocScaleDistribution):
    """Cauchy distribution.

    ``loc`` and ``scale`` should broadcast to the dimension of the distribution.

    Args:
        loc: Location paramter. Defaults to 0.
        scale: Scale parameter. Defaults to 1.
    """

    base_dist: _StandardCauchy
    bijection: Affine

    def __init__(self, loc: ArrayLike = 0, scale: ArrayLike = 1):
        self.base_dist = _StandardCauchy(
            jnp.broadcast_shapes(jnp.shape(loc), jnp.shape(scale)),
        )
        self.bijection = Affine(loc, scale)


class _StandardStudentT(AbstractDistribution):
    """Implements student T distribution with specified degrees of freedom."""

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    df: PositiveParameter

    def __init__(self, df: ArrayLike):
        df = arraylike_to_array(df, dtype=float)
        df = eqx.error_if(df, df <= 0, "Degrees of freedom values must be positive.")
        self.shape = jnp.shape(df)
        self.df = PositiveParameter(df, min_value=0.5)

    def _log_prob(self, x, condition=None):
        return jstats.t.logpdf(x, df=self.df.value).sum()

    def _sample(self, key, condition=None):
        return jr.t(key, df=self.df.value, shape=self.shape)


class StudentT(AbstractLocScaleDistribution):
    """Student T distribution.

    ``df``, ``loc`` and ``scale`` broadcast to the dimension of the distribution.

    Args:
        df: The degrees of freedom.
        loc: Location parameter. Defaults to 0.
        scale: Scale parameter. Defaults to 1.
    """

    base_dist: _StandardStudentT
    bijection: Affine

    def __init__(self, df: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1):
        df, loc, scale = jnp.broadcast_arrays(df, loc, scale)
        self.base_dist = _StandardStudentT(df)
        self.bijection = Affine(loc, scale)

    @property
    def df(self):
        """The degrees of freedom of the distribution."""
        return self.base_dist.df


class _StandardLaplace(AbstractDistribution):
    """Implements standard laplace distribution (loc=0, scale=1)."""

    shape: tuple[int, ...] = ()
    cond_shape: ClassVar[None] = None

    def _log_prob(self, x, condition=None):
        return jstats.laplace.logpdf(x).sum()

    def _sample(self, key, condition=None):
        return jr.laplace(key, shape=self.shape)


class Laplace(AbstractLocScaleDistribution):
    """Laplace distribution.

    ``loc`` and ``scale`` should broadcast to the dimension of the distribution.

    Args:
        loc: Location paramter. Defaults to 0.
        scale: Scale parameter. Defaults to 1.
    """

    base_dist: _StandardLaplace
    bijection: Affine

    def __init__(self, loc: ArrayLike = 0, scale: ArrayLike = 1):
        shape = jnp.broadcast_shapes(jnp.shape(loc), jnp.shape(scale))
        self.base_dist = _StandardLaplace(shape)
        self.bijection = Affine(loc, scale)


class _StandardExponential(AbstractDistribution):
    shape: tuple[int, ...] = ()
    cond_shape: ClassVar[None] = None

    def _log_prob(self, x, condition=None):
        return jstats.expon.logpdf(x).sum()

    def _sample(self, key, condition=None):
        return jr.exponential(key, shape=self.shape)


class Exponential(AbstractTransformed):
    """Exponential distribution.

    Args:
        rate: The rate parameter (1 / scale).
    """

    base_dist: _StandardExponential
    bijection: Scale

    def __init__(self, rate: ArrayLike = 1):
        self.base_dist = _StandardExponential(jnp.shape(rate))
        self.bijection = Scale(1 / rate)

    @property
    def rate(self):
        return 1 / self.bijection.scale.value


class _StandardLogistic(AbstractDistribution):
    shape: tuple[int, ...] = ()
    cond_shape: ClassVar[None] = None

    def _sample(self, key, condition=None):
        return jr.logistic(key, self.shape)

    def _log_prob(self, x, condition=None):
        return jstats.logistic.logpdf(x).sum()


class Logistic(AbstractLocScaleDistribution):
    """Logistic distribution.

    ``loc`` and ``scale`` should broadcast to the shape of the distribution.

    Args:
        loc: Location parameter. Defaults to 0.
        scale: Scale parameter. Defaults to 1.
    """

    base_dist: _StandardLogistic
    bijection: Affine

    def __init__(self, loc: ArrayLike = 0, scale: ArrayLike = 1):
        self.base_dist = _StandardLogistic(
            shape=jnp.broadcast_shapes(jnp.shape(loc), jnp.shape(scale)),
        )
        self.bijection = Affine(loc=loc, scale=scale)


class VmapMixture(AbstractDistribution):
    """Create a mixture distribution.

    Given a distribution in which the arrays have a leading dimension with size matching
    the number of components, and a set of weights, create a mixture distribution.

    Example:
        .. doctest::

            >>> # Creating a 3 component, 2D gaussian mixture
            >>> from flowjax.distributions import Normal, VmapMixture
            >>> import equinox as eqx
            >>> import jax.numpy as jnp
            >>> normals = eqx.filter_vmap(Normal)(jnp.zeros((3, 2)))
            >>> mixture = VmapMixture(normals, weights=jnp.ones(3))
            >>> mixture.shape
            (2,)

    Args:
        dist: Distribution with a leading dimension in arrays with size equal to the
            number of mixture components. Often it is convenient to construct this with
            with a pattern like ``eqx.filter_vmap(MyDistribution)(my_params)``.
        weights: The positive, but possibly unnormalized component weights.
    """

    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None
    log_normalized_weights: LogSimplexParameter
    dist: AbstractDistribution

    def __init__(
        self,
        dist: AbstractDistribution,
        weights: ArrayLike,
    ):
        weights = eqx.error_if(weights, weights <= 0, "Weights must be positive.")
        self.dist = dist
        self.log_normalized_weights = LogSimplexParameter(weights)
        self.shape = dist.shape
        self.cond_shape = dist.cond_shape

    def _log_prob(self, x, condition=None):
        log_probs = eqx.filter_vmap(lambda d: d._log_prob(x, condition))(self.dist)
        return logsumexp(log_probs + self.log_normalized_weights.value)

    def _sample(self, key, condition=None):
        key1, key2 = jr.split(key)
        component = jr.categorical(key1, self.log_normalized_weights.value)
        component_dist = tree_map(
            lambda leaf: leaf[component] if isinstance(leaf, Array) else leaf,
            tree=self.dist,
        )
        return component_dist._sample(key2, condition)


class _StandardGamma(AbstractDistribution):
    concentration: PositiveParameter
    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None

    def __init__(self, concentration: ArrayLike):
        self.concentration = PositiveParameter(
            arraylike_to_array(concentration, dtype=float)
        )
        self.shape = jnp.shape(concentration)

    def _sample(self, key, condition=None):
        return jr.gamma(key, self.concentration.value)

    def _log_prob(self, x, condition=None):
        return jstats.gamma.logpdf(x, self.concentration.value).sum()


class Gamma(AbstractTransformed):
    """Gamma distribution.

    Args:
        concentration: Positive concentration parameter.
        scale: The scale (inverse of rate) parameter.
    """

    base_dist: _StandardGamma
    bijection: Scale

    def __init__(self, concentration: ArrayLike, scale: ArrayLike):
        concentration, scale = jnp.broadcast_arrays(concentration, scale)
        self.base_dist = _StandardGamma(concentration)
        self.bijection = Scale(scale)


class Beta(AbstractDistribution):
    """Beta distribution.

    Args:
        alpha: The alpha shape parameter.
        beta: The beta shape parameter.
    """

    alpha: PositiveParameter
    beta: PositiveParameter
    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None

    def __init__(self, alpha: ArrayLike, beta: ArrayLike):
        alpha, beta = jnp.broadcast_arrays(
            arraylike_to_array(alpha, dtype=float),
            arraylike_to_array(beta, dtype=float),
        )
        self.alpha = PositiveParameter(alpha)
        self.beta = PositiveParameter(beta)
        self.shape = alpha.shape

    def _sample(self, key, condition=None):
        return jr.beta(key, self.alpha.value, self.beta.value)

    def _log_prob(self, x, condition=None):
        return jstats.beta.logpdf(x, self.alpha.value, self.beta.value).sum()
