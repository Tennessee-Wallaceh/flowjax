# Distribution object (for flows and base distributions)

from abc import ABC, abstractmethod
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random
from jax.random import KeyArray
from jax.scipy import stats as jstats

from flowjax.bijections import Affine, Bijection
from flowjax.utils import Array, broadcast_arrays_1d
<<<<<<< HEAD
from typing import Any
import equinox as eqx
from jax.scipy.special import ndtri, gammainc
from flowjax.bijections.univariate import Fixed
from flowjax.bijections.extreme import TailTransformation
from jax.scipy.stats import beta

# Tensorflow probability substrates
import tensorflow_probability as tfp
tquantile = tfp.substrates.jax.distributions.student_t.quantile
=======
>>>>>>> 8d7d0230bb4a876f198cf6a5ac94492b590020cf

# To construct a distribution, we define _log_prob and _sample, which take in vector arguments.
# More friendly methods are then created from these, supporting batches of inputs.
# Note that unconditional distributions should allow, but ignore the passing of conditional variables
# (to facilitate easy composing of conditional and unconditional distributions)


class Distribution(eqx.Module, ABC):
    """Distribution base class."""

    dim: int
    cond_dim: int

    @abstractmethod
    def _log_prob(self, x: Array, condition: Optional[Array] = None):
        "Evaluate the log probability of point x."
        pass

    @abstractmethod
    def _sample(self, key: KeyArray, condition: Optional[Array] = None):
        "Sample a point from the distribution."
        pass

    @property
    def conditional(self):
        "Whether the distribution is an unconditional distribution or not."
        return True if self.cond_dim > 0 else False

    def sample(
        self,
        key: KeyArray,
        condition: Optional[Array] = None,
        n: Optional[int] = None,
    ) -> Array:
        """Sample from a distribution.

        Args:
            key (KeyArray): Jax PRNGKey.
            condition (Optional[Array], optional): Conditioning variables. If the conditioning variable has
                a leading batch dimension, `n` is inferred from the leading axis. Defaults to None.
            n (Optional[int], optional): Number of samples. Defaults to None.

        Returns:
            Array: Jax array of samples.
        """
        self._argcheck_condition(condition)

        if n is None:
            if condition is None:
                return self._sample(key)
            elif condition.ndim == 1:
                return self._sample(key, condition)
            else:
                n = condition.shape[0]
                in_axes = (0, 0)  # type: tuple[Any, Any]
        else:
            if condition is not None and condition.ndim != 1:
                raise ValueError("condition must be 1d if n is provided.")
            in_axes = (0, None)

        keys = random.split(key, n)
        return jax.vmap(self._sample, in_axes)(keys, condition)

    def log_prob(self, x: Array, condition: Optional[Array] = None):
        """Evaluate the log probability. If a matrix/matrices are passed,
        we vmap (vectorise) over the leading axis.

        Args:
            x (Array): Points at which to evaluate density.
            condition (Optional[Array], optional): Conditioning variables. Defaults to None.

        Returns:
            Array: Jax array of log probabilities.
        """
        self._argcheck_x(x)
        self._argcheck_condition(condition)

        if condition is None:
            if x.ndim == 1:
                return self._log_prob(x)
            else:
                return jax.vmap(self._log_prob)(x)
        else:
            if (x.ndim == 1) and (condition.ndim == 1):
                return self._log_prob(x, condition)
            else:
                in_axes = [0 if a.ndim == 2 else None for a in (x, condition)]
                return jax.vmap(self._log_prob, in_axes)(x, condition)

    def icdf(self):
        raise NotImplementedError

    def _argcheck_x(self, x: Array):
        if x.ndim not in (1, 2):
            raise ValueError("x.ndim should be 1 or 2")

        if x.shape[-1] != self.dim:
            raise ValueError(f"Expected x.shape[-1]=={self.dim}, got {x.shape}.")

    def _argcheck_condition(self, condition: Optional[Array] = None):
        if condition is None:
            if self.conditional:
                raise ValueError(f"condition must be provided.")
        else:
            if condition.ndim not in (1, 2):
                raise ValueError("condition.ndim should be 1 or 2")
            if condition.shape[-1] != self.cond_dim:
                raise ValueError(f"Expected condition.shape[-1]=={self.cond_dim}.")


class Transformed(Distribution):
    """
    Form a distribution object defined by a base distribution and a bijection.
    For Z ~ base and bijection T this distribution is of the random variable X = T(Z). 
    The forward bijection maps samples from the base to the final (used for sampling).
    The inverse bijection maps X onto the base distribution (used for density evaluation).
    """
    base_dist: Distribution
    bijection: Bijection
    dim: int
    cond_dim: int
<<<<<<< HEAD
    def __init__( self, base_dist: Distribution, bijection: Bijection):
        """
=======

    def __init__(
        self,
        base_dist: Distribution,
        bijection: Bijection,
    ):
        """
        Form a distribution like object using a base distribution and a
        bijection. We take the forward bijection for use in sampling, and the inverse
        bijection for use in density evaluation.

>>>>>>> 8d7d0230bb4a876f198cf6a5ac94492b590020cf
        Args:
            base_dist (Distribution): Base distribution.
            bijection (Bijection): Bijection to transform distribution.
        """
        self.base_dist = base_dist
        self.bijection = bijection
        self.dim = self.base_dist.dim
        self.cond_dim = max(self.bijection.cond_dim, self.base_dist.cond_dim)

    def _log_prob(self, x: Array, condition: Optional[Array] = None):
        z, log_abs_det = self.bijection.inverse_and_log_abs_det_jacobian(x, condition)
        p_z = self.base_dist._log_prob(z, condition)
        return p_z + log_abs_det

    def _sample(self, key: KeyArray, condition: Optional[Array] = None):
        assert x.shape == (self.dim,)
        z = self.base_dist._sample(key, condition)
        x = self.bijection.transform(z, condition)
        return x

    def quantile(self, u, condition=None):
        base_quantiles = self.base_dist.quantile(u)
        return jax.vmap(self.bijection.transform)(base_quantiles, condition)


class StandardNormal(Distribution):

    def __init__(self, dim: int):
        """
        Implements a standard normal distribution, condition is ignored.

        Args:
            dim (int): Dimension of the normal distribution.
        """
        self.dim = dim
        self.cond_dim = 0

    def _log_prob(self, x: Array, condition: Optional[Array] = None):
        return jnp.clip(
            jstats.norm.logpdf(x).sum(),
            a_min=jnp.log(1e-37)
        )

    def _sample(self, key: KeyArray, condition: Optional[Array] = None):
        return random.normal(key, (self.dim,))

    def __repr__(self):
        return f'<FJ N(0, 1)>'

    def quantile(self, u):
        return ndtri(u)


class Normal(Transformed):
    """
    Implements an independent Normal distribution with mean and std for
    each dimension. `loc` and `scale` should be broadcastable.
    """

    def __init__(self, loc: Array, scale: Array = 1.0):
        """
        Args:
            loc (Array): Array of the means of each dimension.
            scale (Array): Array of the standard deviations of each dimension.
        """
        loc, scale = broadcast_arrays_1d(loc, scale)
        base_dist = StandardNormal(loc.shape[0])
        bijection = Affine(loc=loc, scale=scale)
        super().__init__(base_dist, bijection)

    @property
    def loc(self):
        return self.bijection.loc

    @property
    def scale(self):
        return self.bijection.scale


class _StandardUniform(Distribution):
    """
    Implements a standard independent Uniform distribution, ie X ~ Uniform([0, 1]^dim).
    """

    def __init__(self, dim):
        self.dim = dim
        self.cond_dim = 0

    def _log_prob(self, x: Array, condition: Optional[Array] = None):
        return jstats.uniform.logpdf(x).sum()

    def _sample(self, key: KeyArray, condition: Optional[Array] = None):
        return random.uniform(key, shape=(self.dim,))

    def __repr__(self):
        return f'<FJ N({self.mean}, {self.std})>'

    def quantile(self, u):
        return ndtri(u)

class Uniform(Transformed):
    """
    Implements an independent uniform distribution
    between min and max for each dimension. `minval` and `maxval` should be broadcastable.
    """

    def __init__(self, minval: Array, maxval: Array):
        """
        Args:
            minval (Array): ith entry gives the min of the ith dimension
            maxval (Array): ith entry gives the max of the ith dimension
        """
        minval, maxval = broadcast_arrays_1d(minval, maxval)
        if jnp.any(maxval < minval):
            raise ValueError("Minimums must be less than maximums.")
        base_dist = _StandardUniform(minval.shape[0])
        bijection = Affine(loc=minval, scale=maxval - minval)
        super().__init__(base_dist, bijection)

    @property
    def minval(self):
        return self.bijection.loc

    @property
    def maxval(self):
        return self.bijection.loc + self.bijection.scale

    def quantile(self, u):
        return (u  * (self.max - self.min)) + self.min


class _StandardGumbel(Distribution):
    """Standard gumbel distribution (https://en.wikipedia.org/wiki/Gumbel_distribution).
    """

    def __init__(self, dim):
        
        self.dim = dim
        self.cond_dim = 0

    def _log_prob(self, x: Array, condition: Optional[Array] = None):
        return -(x + jnp.exp(-x)).sum()

    def _sample(self, key: KeyArray, condition: Optional[Array] = None):
        return random.gumbel(key, shape=(self.dim,))

    def __repr__(self):
        return f'<FJ Gumbel(0, 1)>'

    def quantile(self, u):
        raise NotImplementedError


class Gumbel(Transformed):
    """Gumbel distribution (https://en.wikipedia.org/wiki/Gumbel_distribution)"""

    def __init__(self, loc: Array, scale: Array = 1.0):
        """
        `loc` and `scale` should broadcast to the dimension of the distribution.

        Args:
            loc (Array): Location paramter. 
            scale (Array, optional): Scale parameter. Defaults to 1.0.
        """
        loc, scale = broadcast_arrays_1d(loc, scale)
        base_dist = _StandardGumbel(loc.shape[0])
        bijection = Affine(loc, scale)
        super().__init__(base_dist, bijection)

    @property
    def loc(self):
        return self.bijection.loc

    @property
    def scale(self):
        return self.bijection.scale


class _StandardCauchy(Distribution):
    """
    Implements standard cauchy distribution (loc=0, scale=1)
    Ref: https://en.wikipedia.org/wiki/Cauchy_distribution
    """
    def __init__(self, dim):
        self.dim = dim
        self.cond_dim = 0

    def _log_prob(self, x: Array, condition: Optional[Array] = None):
        return jstats.cauchy.logpdf(x).sum()

    def _sample(self, key: KeyArray, condition: Optional[Array] = None):
        return random.cauchy(key, shape=(self.dim,))

    def __repr__(self):
        return f'<FJ Cauchy(0, 1)>'

    def quantile(self, u):
        return jnp.tan(jnp.pi  * (u - 0.5))


class Cauchy(Transformed):
    """
    Cauchy distribution (https://en.wikipedia.org/wiki/Cauchy_distribution).
    """
    def __init__(self, loc: Array, scale: Array = 1.0):
        """
        `loc` and `scale` should broadcast to the dimension of the distribution.

        Args:
            loc (Array): Location paramter. 
            scale (Array, optional): Scale parameter. Defaults to 1.0.
        """
        loc, scale = broadcast_arrays_1d(loc, scale)
        base_dist = _StandardCauchy(loc.shape[0])
        bijection = Affine(loc, scale)
        super().__init__(base_dist, bijection)

    @property
    def loc(self):
        return self.bijection.loc

    @property
    def scale(self):
        return self.bijection.scale


class _StandardStudentT(Distribution):
    """
    Implements student T distribution with specified degrees of freedom.
    """
    log_df: Array
    def __init__(self, df: Array):
        self.dim = df.shape[0]
        self.cond_dim = 0
        self.log_df = jnp.log(df)

    def _log_prob(self, x: Array, condition: Optional[Array] = None):
        return jstats.t.logpdf(x, df=self.df).sum()

    def _sample(self, key: KeyArray, condition: Optional[Array] = None):
        return random.t(key, df=self.df, shape=(self.dim,))

    @property
    def df(self):
        return jnp.exp(self.log_df)


class StudentT(Transformed):
<<<<<<< HEAD
    "Student T distribution. `loc` and `scale` should be broadcastable."
    def __init__(self, df: Array, loc: Array, scale: Array = 1.0):
=======
    """Student T distribution (https://en.wikipedia.org/wiki/Student%27s_t-distribution)."""

    def __init__(self, df: Array, loc: Array = 0.0, scale: Array = 1.0):
        """
        `df`, `loc` and `scale` broadcast to the dimension of the distribution.

        Args:
            df (Array): The degrees of freedom.
            loc (Array): Location parameter. Defaults to 0.0.
            scale (Array, optional): Scale parameter. Defaults to 1.0.
        """
>>>>>>> 8d7d0230bb4a876f198cf6a5ac94492b590020cf
        df, loc, scale = broadcast_arrays_1d(df, loc, scale)
        self.dim = df.shape[0]
        self.cond_dim = 0
        base_dist = _StandardStudentT(df)
        bijection = Affine(loc, scale)
        super().__init__(base_dist, bijection)

    @property
    def loc(self):
        return self.bijection.loc

    @property
    def scale(self):
        return self.bijection.scale

    @property
    def df(self):
        return self.base_dist.df


class TwoSidedGPD(Distribution):
    neg_tail: float
    pos_tail: float
    dist: Distribution
    def __init__(self, dim, neg_tail=1., pos_tail=1.):
        self.dim = dim
        self.cond_dim = 0
        self.neg_tail = neg_tail
        self.pos_tail = pos_tail
        self.dist = Transformed(
            base_dist=Uniform(1, -1, 1), 
            bijection=Fixed(
                TailTransformation(None, None), 
                jnp.array([pos_tail]), jnp.array([neg_tail])
            )
        )

    def _log_prob(self, x: Array, condition: Optional[Array] = None):
        return self.dist._log_prob(x)

    def _sample(self, key: KeyArray, condition: Optional[Array] = None):
        return self.dist._sample(key)

    def __repr__(self):
        return (
            '<FJ TwoSideGPD('
            f'neg_tail={self.neg_tail:.2f} | '
            f'pos_tail={self.pos_tail:.2f} | '
            ')>'
        )


class HalfStudentT(Distribution):
    unc_df: Array
    def __init__(self, dim, df=1.):
        self.dim = dim
        self.cond_dim = 0
        self.unc_df = jnp.log(jnp.array([df]))

    def _log_prob(self, x: Array, condition: Optional[Array] = None):
        return jnp.clip(
            jnp.log(2) + jstats.t.logpdf(x, df=jnp.exp(self.unc_df)),
            a_min=jnp.log(1e-37)
        ) 

    def _sample(self, key: KeyArray, condition: Optional[Array] = None):
        df = jnp.exp(self.unc_df)
        beta = random.beta(key, 0.5 * df, 0.5, shape=(self.dim,))
        x = jnp.sqrt(df / beta - df)
        return x

    def __repr__(self):
        return (
            '<FJ HalfStudentT('
            f'neg_tail={self.neg_tail:.2f} | '
            f'pos_tail={self.pos_tail:.2f} | '
            ')>'
        )