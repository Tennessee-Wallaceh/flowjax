# Distribution object (for flows and base distributions)

from abc import ABC, abstractmethod
from typing import Optional
from flowjax.bijections.abc import Bijection
from jax import random
from jax.scipy import stats as jstats
import jax
import jax.numpy as jnp
from jax.random import KeyArray
from flowjax.utils import Array
from typing import Any
import equinox as eqx
from jax.scipy.special import ndtri
from flowjax.bijections.univariate import Fixed
from flowjax.bijections.extreme import TailTransformation
from jax.scipy.stats import beta

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
        self, key: KeyArray, condition: Optional[Array] = None, n: Optional[int] = None,
    ) -> Array:
        """Sample from a distribution. The condition can be a vector, or a matrix.
        - If condition.ndim==1, n allows repeated sampling (a single sample is drawn if n is not provided).
        - If condition.ndim==2, axis 0 is treated as batch dimension, (one sample is drawn for each row).

        Args:
            key (KeyArray): Jax PRNGKey.
            condition (Optional[Array], optional): Conditioning variables. Defaults to None.
            n (Optional[int], optional): Number of samples (if condition.ndim==1). Defaults to None.

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

    def _argcheck_x(self, x: Array):
        if x.ndim not in (1,2):
            raise ValueError("x.ndim should be 1 or 2")

        if x.shape[-1] != self.dim:
            raise ValueError(f"Expected x.shape[-1]=={self.dim}.")

    def _argcheck_condition(self, condition: Optional[Array] = None):
        if condition is None:
            if self.conditional:
                raise ValueError(f"condition must be provided.")
        else:
            if condition.ndim not in (1,2):
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
    def __init__( self, base_dist: Distribution, bijection: Bijection):
        """
        Args:
            base_dist (Distribution): Base distribution.
            bijection (Bijection): Bijection defined in "normalising" direction.
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
        z = self.base_dist._sample(key, condition)
        x = self.bijection.transform(z, condition)
        return x

    def quantile(self, u, condition=None):
        q_func = getattr(self, 'quantile', None)
        assert q_func is not None, 'Quantile not implemented!'
        return jax.vmap(self.bijection.transform)(self.base_dist.quantile(u), condition)


class StandardNormal(Distribution):
    """
    Implements a standard normal distribution, condition is ignored.
    """
    def __init__(self, dim: int):
        """
        Args:
            dim (int): Dimension of the normal distribution.
        """
        self.dim = dim
        self.cond_dim = 0

    def _log_prob(self, x: Array, condition: Optional[Array] = None):
        assert x.shape == (self.dim,)
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


class Normal(Distribution):
    """
    Implements a normal distribution with mean and std.
    """
    mean: float
    std: float
    def __init__(self, dim: int, mean: float=0.0, std: float=1.0):
        """
        Args:
            dim (int): Dimension of the normal distribution.
        """
        self.dim = dim
        self.cond_dim = 0
        self.mean = mean
        self.std = std

    def _log_prob(self, x: Array, condition: Optional[Array] = None):
        assert x.shape == (self.dim,)
        std_x = (x - self.mean) / self.std
        return jstats.norm.logpdf(std_x).sum()

    def _sample(self, key: KeyArray, condition: Optional[Array] = None):
        std_x = random.normal(key, (self.dim,))
        return self.std * std_x + self.mean

    def __repr__(self):
        return f'<FJ N({self.mean}, {self.std})>'

class Uniform(Distribution):
    """
    Implements a Uniform distribution defined over a min and max val.
    X ~ Uniform([min, max])
    """
    min: float
    max: float
    def __init__(self, dim, min=0.0, max=1.0):
        """
        Args:
            min (float): 
            max (float): 
        """
        self.dim = dim
        self.cond_dim = 0
        self.min = min
        self.max = max

    def _log_prob(self, x: Array, condition: Optional[Array] = None):
        assert x.shape == (self.dim,)
        return jstats.uniform.logpdf(
            x, loc=self.min, scale=self.max - self.min
        ).sum()

    def _sample(self, key: KeyArray, condition: Optional[Array] = None):
        return random.uniform(
            key, shape=(self.dim,), minval=self.min, maxval=self.max
        )

    def __repr__(self):
        return f'<FlowJax Uniform([{self.min}, {self.max})>'

    def quantile(self, u):
        return (u  * (self.max - self.min)) + self.min

class Gumbel(Distribution):
    """
    Implements standard gumbel distribution (loc=0, scale=1)
    Ref: https://en.wikipedia.org/wiki/Gumbel_distribution
    """
    def __init__(self, dim):
        self.dim = dim
        self.cond_dim = 0

    def _log_prob(self, x: Array, condition: Optional[Array] = None):
        assert x.shape == (self.dim,)
        return -(x + jnp.exp(-x)).sum()

    def _sample(self, key: KeyArray, condition: Optional[Array] = None):
        return random.gumbel(key, shape=(self.dim,))

    def __repr__(self):
        return f'<FJ Gumbel(0, 1)>'

    def quantile(self, u):
        raise NotImplementedError
        
class Cauchy(Distribution):
    """
    Implements standard cauchy distribution (loc=0, scale=1)
    Ref: https://en.wikipedia.org/wiki/Cauchy_distribution
    """
    def __init__(self, dim):
        self.dim = dim
        self.cond_dim = 0

    def _log_prob(self, x: Array, condition: Optional[Array] = None):
        assert x.shape == (self.dim,)
        return jstats.cauchy.logpdf(x).sum()

    def _sample(self, key: KeyArray, condition: Optional[Array] = None):
        return random.cauchy(key, shape=(self.dim,))

    def __repr__(self):
        return f'<FJ Cauchy(0, 1)>'

    def quantile(self, u):
        return jnp.tan(jnp.pi  * (u - 0.5))

class StudentT(Distribution):
    """
    Implements student T distribution with specified degree of freedom.
    """
    unc_df: Array
    def __init__(self, dim, df=30.):
        self.dim = dim
        self.cond_dim = 0
        self.unc_df = jnp.log(jnp.array([df]))

    def _log_prob(self, x: Array, condition: Optional[Array] = None):
        assert x.shape == (self.dim,)
        # return jstats.t.logpdf(x, df=self.df).sum()
        return jnp.clip(
            jstats.t.logpdf(x, df=jnp.exp(self.unc_df)).sum(),
            a_min=jnp.log(1e-37)
        )

    def _sample(self, key: KeyArray, condition: Optional[Array] = None):
        return random.t(key, df=jnp.exp(self.unc_df), shape=(self.dim,))

    def __repr__(self):
        return f'<FJ StudentT(df={jnp.exp(self.unc_df).item():.2f})>'


class TwoSidedPareto(Distribution):
    neg_tail: float
    pos_tail: float
    def __init__(self, dim, neg_tail=1., pos_tail=1.):
        self.dim = dim
        self.cond_dim = 0
        self.neg_tail = neg_tail
        self.pos_tail = pos_tail

    def _log_prob(self, x: Array, condition: Optional[Array] = None):
        return jnp.log(jnp.where(
            x < 0,
            0.5 * jstats.pareto.pdf(1 - x, self.neg_tail),
            0.5 * jstats.pareto.pdf(x + 1, self.pos_tail)    
        ))

    def _sample(self, key: KeyArray, condition: Optional[Array] = None):
        key_1, key_2, key_3 = random.split(key, 3)
        return jnp.where(
            random.bernoulli(key_1, jnp.array([0.5])),
            1 - random.pareto(key_2, self.neg_tail),
            random.pareto(key_3, self.pos_tail) - 1
        )

    def __repr__(self):
        return (
            '<FJ Pareto('
            f'neg_tail={self.neg_tail:.2f} | '
            f'pos_tail={self.pos_tail:.2f} | '
            ')>'
        )


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
    

class Beta(Distribution):
    _a: Array
    _b: Array
    def __init__(self, dim, a=1., b=1.):
        self.dim = dim
        self.cond_dim = 0
        self._a = jnp.log(jnp.array(a))
        self._b = jnp.log(jnp.array(b))

    @property
    def a(self):
        return jnp.exp(self._a)

    @property
    def b(self):
        return jnp.exp(self._b)

    def _log_prob(self, x: Array, condition: Optional[Array] = None):
        return beta.logpdf(x, self.a, self.b)

    def _sample(self, key: KeyArray, condition: Optional[Array] = None):
        return random.beta(key, self.a, self.b, shape=(self.dim,))

    def __repr__(self):
        return (
            '<FJ Beta('
                f'a={self.a:.2f} | '
                f'b={self.b:.2f} | '
            ')>'
        )


class DoubleBeta(Distribution):
    _a_neg: Array
    _a_pos: Array
    _b_neg: Array
    _b_pos: Array
    def __init__(self, dim, a_neg=1., a_pos=1., b_neg=1., b_pos=1.):
        self.dim = dim
        self.cond_dim = 0
        self._a_neg = jnp.log(jnp.array(a_neg))
        self._a_pos = jnp.log(jnp.array(a_pos))
        self._b_neg = jnp.log(jnp.array(b_neg))
        self._b_pos = jnp.log(jnp.array(b_pos))

    @property
    def a_neg(self):
        return jnp.exp(self._a_neg)
    
    @property
    def a_pos(self):
        return jnp.exp(self._a_pos)

    @property
    def b_neg(self):
        return jnp.exp(self._b_neg)

    @property
    def b_pos(self):
        return jnp.exp(self._b_pos)

    def _log_prob(self, x: Array, condition: Optional[Array] = None):
        return jnp.log(jnp.where(
            x < 0,
            0.5 * beta.pdf(-x, self.a_neg, self.b_neg),
            0.5 * beta.pdf(x, self.a_pos, self.b_pos)
        ))

    def _sample(self, key: KeyArray, condition: Optional[Array] = None):
        key_1, key_2 = random.split(key)
        raw_samp = jnp.where(
            random.bernoulli(key_1, jnp.array([0.5])),
            -random.beta(key_2, self.a_neg, self.b_neg, shape=(self.dim,)),
            random.beta(key_2, self.a_pos, self.b_pos, shape=(self.dim,))
        )
        return jnp.clip(raw_samp, a_min=-1 + 1e-7, a_max=1 - 1e-7)

    def __repr__(self):
        return (
            '<FJ DBeta()>'
        )