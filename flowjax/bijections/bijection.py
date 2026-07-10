"""Abstract base classes for bijections.

Note when implementing bijections, by convention we try to i) implement the "transform"
methods as the faster/more intuitive approach (compared to the inverse methods); and ii)
implement only the forward methods if an inverse is not available. The ``Invert``
bijection can be used to invert the orientation if a fast inverse is desired (e.g.
maximum likelihood fitting of flows).
"""

from abc import abstractmethod
import typing

import equinox as eqx
from equinox import AbstractVar, AbstractClassVar
from jaxtyping import Array, ArrayLike, PRNGKeyArray
import jax.random as jr

from flowjax.utils import arraylike_to_array


class AbstractBijection(eqx.Module):
    """Bijection abstract class.

    Similar to :py:class:`~flowjax.distributions.AbstractDistribution`, bijections have
    a ``shape`` and a ``cond_shape`` attribute. To allow easy composition of bijections,
    all bijections support passing conditioning variables, even if ignored.

    Bijections are registered as JAX PyTrees through Equinox, so are compatible with
    normal JAX operations. Methods act on a single input and, for conditional
    bijections, a single conditioning variable. Use ``jax.vmap`` or
    ``eqx.filter_vmap`` to explicitly vectorise operations over additional batch
    dimensions.

    The standard methods are deterministic and represent evaluation of the actual
    bijection. The ``*_with_state`` methods provide an optional path for modules that
    need training/evaluation mode, random keys, or Equinox state. Stateless bijections
    inherit default implementations that delegate to the deterministic methods.

    Implementing a bijection:

    - Inherit from ``AbstractBijection``.
    - Define the attributes ``shape`` and ``cond_shape``. A ``cond_shape`` of
      ``None`` represents an unconditional bijection.
    - Implement ``transform_and_log_det`` and ``inverse_and_log_det``.
    - Optionally override ``transform_and_log_det_with_state`` and
      ``inverse_and_log_det_with_state`` if the bijection requires random keys,
      inference mode, or state.
    """

    shape: AbstractVar[tuple[int, ...]]
    cond_shape: AbstractVar[tuple[int, ...] | None]
    # these need to be AbstractVar rather than AbstractClassVar, 
    # since e.g Chain only knows at init time which it satisfies.
    stochastic: AbstractVar[bool]
    stateful: AbstractVar[bool]

class AbstractDeterministicBijection[Condition: typing.Union[Array | None]](AbstractBijection):
    stochastic: bool = False
    stateful: bool = False
    
    @abstractmethod
    def transform_and_log_det(
        self,
        x: ArrayLike,
        *,
        condition: Condition = None,
    ) -> tuple[Array, Array]:
        """Apply transformation and compute the log absolute Jacobian determinant.

        Args:
            x: Input with shape matching ``bijection.shape``.
            condition: Conditioning variable with shape matching
                ``bijection.cond_shape``. Required for conditional bijections.
        """

    @abstractmethod
    def inverse_and_log_det(
        self,
        y: ArrayLike,
        *,
        condition: Condition = None,
    ) -> tuple[Array, Array]:
        """Compute the inverse and corresponding log absolute Jacobian determinant.

        Args:
            y: Input with shape matching ``bijection.shape``.
            condition: Conditioning variable with shape matching
                ``bijection.cond_shape``. Required for conditional bijections.
        """


    def transform(
        self,
        x: ArrayLike,
        *,
        condition: Condition = None,
    ) -> Array:
        """Apply the forward transformation.

        Args:
            x: Input with shape matching ``bijection.shape``.
            condition: Conditioning variable with shape matching
                ``bijection.cond_shape``. Required for conditional bijections.

        Returns:
            Transformed input.
        """
        return self.transform_and_log_det(x, condition=condition)[0]

    def inverse(
        self,
        y: ArrayLike,
        *,
        condition: Condition = None,
    ) -> Array:
        """Compute the inverse transformation.

        Args:
            y: Input with shape matching ``bijection.shape``.
            condition: Conditioning variable with shape matching
                ``bijection.cond_shape``. Required for conditional bijections.

        Returns:
            Inverse-transformed input.
        """
        return self.inverse_and_log_det(y, condition=condition)[0]

class AbstractStochasticBijection[Condition: typing.Union[Array | None]](AbstractBijection):
    """
    The interpretation of this class is a bit subtle.
    Essentially, for every fixed realization of the randomness, 
    the resulting map is bijective and has a corresponding inverse and log determinant.
    This isn't the case for something like a VAE.
    """
    stochastic = True
    stateful = False

    @abstractmethod
    def transform_and_log_det(
        self,
        x: ArrayLike,
        *,
        condition: Condition = None,
        key: PRNGKeyArray,
        inference: bool = True,
    ) -> tuple[Array, Array]:
        """Apply a stochastic transformation and compute the log determinant."""

    @abstractmethod
    def inverse_and_log_det(
        self,
        y: ArrayLike,
        *,
        condition: Condition = None,
        key: PRNGKeyArray,
        inference: bool = True,
    ) -> tuple[Array, Array]:
        """Compute the stochastic inverse and corresponding log determinant."""
    
class AbstractStatefulBijection[Condition: typing.Union[Array | None]](AbstractBijection):
    stochastic = False
    stateful = True

    @abstractmethod
    def transform_and_log_det(
        self,
        x: ArrayLike,
        *,
        condition: Condition = None,
        state: eqx.nn.State,
        inference: bool = True,
    ) -> tuple[tuple[Array, Array], eqx.nn.State]:
        """Apply a stateful transformation and compute the log determinant."""

    @abstractmethod
    def inverse_and_log_det(
        self,
        y: ArrayLike,
        *,
        condition: Condition = None,
        state: eqx.nn.State,
        inference: bool = True,
    ) -> tuple[tuple[Array, Array], eqx.nn.State]:
        """Compute the stateful inverse and corresponding log determinant."""

    def transform(
        self,
        x: ArrayLike,
        *,
        condition: Condition = None,
        state: eqx.nn.State,
        inference: bool = True,
    ) -> tuple[Array, eqx.nn.State]:
        (y, _), state = self.transform_and_log_det(
            x,
            condition=condition,
            state=state,
            inference=inference,
        )
        return y, state

    def inverse(
        self,
        y: ArrayLike,
        *,
        condition: Condition = None,
        state: eqx.nn.State,
        inference: bool = True,
    ) -> tuple[Array, eqx.nn.State]:
        (x, _), state = self.inverse_and_log_det(
            y,
            condition=condition,
            state=state,
            inference=inference,
        )
        return x, state

class AbstractStochasticStatefulBijection[
    Condition: typing.Union[Array | None]
](AbstractBijection):
    stochastic = True
    stateful = True

    @abstractmethod
    def transform_and_log_det(
        self,
        x: ArrayLike,
        *,
        condition: Condition = None,
        key: PRNGKeyArray,
        state: eqx.nn.State,
        inference: bool = True,
    ) -> tuple[tuple[Array, Array], eqx.nn.State]:
        """Apply a stochastic stateful transformation and compute the log determinant."""

    @abstractmethod
    def inverse_and_log_det(
        self,
        y: ArrayLike,
        *,
        condition: Condition = None,
        key: PRNGKeyArray,
        state: eqx.nn.State,
        inference: bool = True,
    ) -> tuple[tuple[Array, Array], eqx.nn.State]:
        """Compute the stochastic stateful inverse and corresponding log determinant."""

    def transform(
        self,
        x: ArrayLike,
        *,
        condition: Condition = None,
        key: PRNGKeyArray,
        state: eqx.nn.State,
        inference: bool = True,
    ) -> tuple[Array, eqx.nn.State]:
        (y, _), state = self.transform_and_log_det(
            x,
            condition=condition,
            key=key,
            state=state,
            inference=inference,
        )
        return y, state

    def inverse(
        self,
        y: ArrayLike,
        *,
        condition: Condition = None,
        key: PRNGKeyArray,
        state: eqx.nn.State,
        inference: bool = True,
    ) -> tuple[Array, eqx.nn.State]:
        (x, _), state = self.inverse_and_log_det(
            y,
            condition=condition,
            key=key,
            state=state,
            inference=inference,
        )
        return x, state
    
def _check_and_cast(
    bijection: AbstractBijection,
    x: ArrayLike,
    condition: ArrayLike | None = None,
) -> tuple[Array, Array | None]:
    """Cast and validate arguments for a bijection operation."""
    x = arraylike_to_array(x)

    if x.shape != bijection.shape:
        raise ValueError(
            f"Expected input shape {bijection.shape}; got {x.shape}."
        )

    if condition is not None:
        condition = arraylike_to_array(condition, err_name="condition")
    elif bijection.cond_shape is not None:
        raise ValueError("Expected condition to be provided.")

    if (
        # implies that  bijection.cond_shape is not None
        # from above
        condition is not None 
        and condition.shape != bijection.cond_shape
    ):
        raise ValueError(
            f"Expected condition.shape {bijection.cond_shape}; got "
            f"{condition.shape}."
        )

    return x, condition


@typing.overload
def transform_and_log_det(
    bijection: AbstractDeterministicBijection,
    x: ArrayLike,
    *,
    condition: ArrayLike | None = None,
    key: None = None,
    state: None = None,
    inference: bool = True,
) -> tuple[Array, Array]:
    ...


@typing.overload
def transform_and_log_det(
    bijection: AbstractStochasticBijection,
    x: ArrayLike,
    *,
    condition: ArrayLike | None = None,
    key: PRNGKeyArray,
    state: None = None,
    inference: bool = True,
) -> tuple[Array, Array]:
    ...


@typing.overload
def transform_and_log_det(
    bijection: AbstractStatefulBijection,
    x: ArrayLike,
    *,
    condition: ArrayLike | None = None,
    key: None = None,
    state: eqx.nn.State,
    inference: bool = True,
) -> tuple[tuple[Array, Array], eqx.nn.State]:
    ...


@typing.overload
def transform_and_log_det(
    bijection: AbstractStochasticStatefulBijection,
    x: ArrayLike,
    *,
    condition: ArrayLike | None = None,
    key: PRNGKeyArray,
    state: eqx.nn.State,
    inference: bool = True,
) -> tuple[tuple[Array, Array], eqx.nn.State]:
    ...

def transform_and_log_det(
    bijection: AbstractBijection,
    x: ArrayLike,
    *,
    condition: ArrayLike | None = None,
    key: PRNGKeyArray | None = None,
    state: eqx.nn.State | None = None,
    inference: bool = True,
):
    """Transform an input and return the log absolute Jacobian determinant."""
    x, condition = _check_and_cast(bijection, x, condition)

    match bijection:
        case AbstractStochasticStatefulBijection():
            if key is None:
                raise ValueError(
                    "A key is required for a stochastic bijection."
                )
            if state is None:
                raise ValueError(
                    "State is required for a stateful bijection."
                )

            return bijection.transform_and_log_det(
                x,
                condition=condition,
                key=key,
                state=state,
                inference=inference,
            )

        case AbstractStochasticBijection():
            if key is None:
                raise ValueError(
                    "A key is required for a stochastic bijection."
                )

            return bijection.transform_and_log_det(
                x,
                condition=condition,
                key=key,
                inference=inference,
            )

        case AbstractStatefulBijection():
            if state is None:
                raise ValueError(
                    "State is required for a stateful bijection."
                )

            return bijection.transform_and_log_det(
                x,
                condition=condition,
                state=state,
                inference=inference,
            )

        case AbstractDeterministicBijection():
            return bijection.transform_and_log_det(
                x,
                condition=condition,
            )

        case _:
            raise TypeError(
                f"Unsupported bijection type: {type(bijection).__name__}."
            )
        
@typing.overload
def inverse_and_log_det(
    bijection: AbstractDeterministicBijection,
    y: ArrayLike,
    *,
    condition: ArrayLike | None = None,
    key: None = None,
    state: None = None,
    inference: bool = True,
) -> tuple[Array, Array]:
    ...


@typing.overload
def inverse_and_log_det(
    bijection: AbstractStochasticBijection,
    y: ArrayLike,
    *,
    condition: ArrayLike | None = None,
    key: PRNGKeyArray,
    state: None = None,
    inference: bool = True,
) -> tuple[Array, Array]:
    ...


@typing.overload
def inverse_and_log_det(
    bijection: AbstractStatefulBijection,
    y: ArrayLike,
    *,
    condition: ArrayLike | None = None,
    key: None = None,
    state: eqx.nn.State,
    inference: bool = True,
) -> tuple[tuple[Array, Array], eqx.nn.State]:
    ...


@typing.overload
def inverse_and_log_det(
    bijection: AbstractStochasticStatefulBijection,
    y: ArrayLike,
    *,
    condition: ArrayLike | None = None,
    key: PRNGKeyArray,
    state: eqx.nn.State,
    inference: bool = True,
) -> tuple[tuple[Array, Array], eqx.nn.State]:
    ...


def inverse_and_log_det(
    bijection: AbstractBijection,
    y: ArrayLike,
    *,
    condition: ArrayLike | None = None,
    key: PRNGKeyArray | None = None,
    state: eqx.nn.State | None = None,
    inference: bool = True,
) -> (
    tuple[Array, Array]
    | tuple[tuple[Array, Array], eqx.nn.State]
):
    """Invert an input and return the log absolute Jacobian determinant."""
    y, condition = _check_and_cast(bijection, y, condition)

    match bijection:
        case AbstractStochasticStatefulBijection():
            if key is None:
                raise ValueError(
                    "A key is required for a stochastic bijection."
                )
            if state is None:
                raise ValueError(
                    "State is required for a stateful bijection."
                )

            return bijection.inverse_and_log_det(
                y,
                condition=condition,
                key=key,
                state=state,
                inference=inference,
            )

        case AbstractStochasticBijection():
            if key is None:
                raise ValueError(
                    "A key is required for a stochastic bijection."
                )

            return bijection.inverse_and_log_det(
                y,
                condition=condition,
                key=key,
                inference=inference,
            )

        case AbstractStatefulBijection():
            if state is None:
                raise ValueError(
                    "State is required for a stateful bijection."
                )

            return bijection.inverse_and_log_det(
                y,
                condition=condition,
                state=state,
                inference=inference,
            )

        case AbstractDeterministicBijection():
            return bijection.inverse_and_log_det(
                y,
                condition=condition,
            )

        case _:
            raise TypeError(
                f"Unsupported bijection type: {type(bijection).__name__}."
            )
        

@typing.overload
def transform_and_log_det_direct(
    bijection: AbstractDeterministicBijection,
    x: Array,
    *,
    condition: Array | None,
    key: None = None,
    state: None = None,
    inference: bool,
) -> tuple[Array, Array]:
    ...


@typing.overload
def transform_and_log_det_direct(
    bijection: AbstractStochasticBijection,
    x: Array,
    *,
    condition: Array | None,
    key: PRNGKeyArray,
    state: None = None,
    inference: bool,
) -> tuple[Array, Array]:
    ...


@typing.overload
def transform_and_log_det_direct(
    bijection: AbstractStatefulBijection,
    x: Array,
    *,
    condition: Array | None,
    key: None = None,
    state: eqx.nn.State,
    inference: bool,
) -> tuple[tuple[Array, Array], eqx.nn.State]:
    ...


@typing.overload
def transform_and_log_det_direct(
    bijection: AbstractStochasticStatefulBijection,
    x: Array,
    *,
    condition: Array | None,
    key: PRNGKeyArray,
    state: eqx.nn.State,
    inference: bool,
) -> tuple[tuple[Array, Array], eqx.nn.State]:
    ...


def transform_and_log_det_direct(
    bijection: AbstractBijection,
    x: Array,
    *,
    condition: Array | None,
    key: PRNGKeyArray | None = None,
    state: eqx.nn.State | None = None,
    inference: bool,
) -> (
    tuple[Array, Array]
    | tuple[tuple[Array, Array], eqx.nn.State]
):
    match bijection:
        case AbstractStochasticStatefulBijection():
            assert key is not None
            assert state is not None

            return bijection.transform_and_log_det(
                x,
                condition=condition,
                key=key,
                state=state,
                inference=inference,
            )

        case AbstractStochasticBijection():
            assert key is not None

            return bijection.transform_and_log_det(
                x,
                condition=condition,
                key=key,
                inference=inference,
            )

        case AbstractStatefulBijection():
            assert state is not None

            return bijection.transform_and_log_det(
                x,
                condition=condition,
                state=state,
                inference=inference,
            )

        case AbstractDeterministicBijection():
            return bijection.transform_and_log_det(
                x,
                condition=condition,
            )

        case _:
            raise TypeError(
                f"Unsupported bijection type: {type(bijection).__name__}."
            )


@typing.overload
def inverse_and_log_det_direct(
    bijection: AbstractDeterministicBijection,
    y: Array,
    *,
    condition: Array | None,
    key: None = None,
    state: None = None,
    inference: bool,
) -> tuple[Array, Array]:
    ...


@typing.overload
def inverse_and_log_det_direct(
    bijection: AbstractStochasticBijection,
    y: Array,
    *,
    condition: Array | None,
    key: PRNGKeyArray,
    state: None = None,
    inference: bool,
) -> tuple[Array, Array]:
    ...


@typing.overload
def inverse_and_log_det_direct(
    bijection: AbstractStatefulBijection,
    y: Array,
    *,
    condition: Array | None,
    key: None = None,
    state: eqx.nn.State,
    inference: bool,
) -> tuple[tuple[Array, Array], eqx.nn.State]:
    ...


@typing.overload
def inverse_and_log_det_direct(
    bijection: AbstractStochasticStatefulBijection,
    y: Array,
    *,
    condition: Array | None,
    key: PRNGKeyArray,
    state: eqx.nn.State,
    inference: bool,
) -> tuple[tuple[Array, Array], eqx.nn.State]:
    ...


def inverse_and_log_det_direct(
    bijection: AbstractBijection,
    y: Array,
    *,
    condition: Array | None,
    key: PRNGKeyArray | None = None,
    state: eqx.nn.State | None = None,
    inference: bool,
) -> (
    tuple[Array, Array]
    | tuple[tuple[Array, Array], eqx.nn.State]
):
    match bijection:
        case AbstractStochasticStatefulBijection():
            assert key is not None
            assert state is not None

            return bijection.inverse_and_log_det(
                y,
                condition=condition,
                key=key,
                state=state,
                inference=inference,
            )

        case AbstractStochasticBijection():
            assert key is not None

            return bijection.inverse_and_log_det(
                y,
                condition=condition,
                key=key,
                inference=inference,
            )

        case AbstractStatefulBijection():
            assert state is not None

            return bijection.inverse_and_log_det(
                y,
                condition=condition,
                state=state,
                inference=inference,
            )

        case AbstractDeterministicBijection():
            return bijection.inverse_and_log_det(
                y,
                condition=condition,
            )

        case _:
            raise TypeError(
                f"Unsupported bijection type: {type(bijection).__name__}."
            )
        


@typing.overload
def _transform_and_log_det_direct(
    bijection: AbstractDeterministicBijection,
    x: Array,
    *,
    condition: Array | None = None,
    key: None = None,
    state: None = None,
    inference: bool = True,
) -> tuple[Array, Array, None]:
    ...


@typing.overload
def _transform_and_log_det_direct(
    bijection: AbstractStochasticBijection,
    x: Array,
    *,
    condition: Array | None = None,
    key: PRNGKeyArray,
    state: None = None,
    inference: bool = True,
) -> tuple[Array, Array, None]:
    ...


@typing.overload
def _transform_and_log_det_direct(
    bijection: AbstractStatefulBijection,
    x: Array,
    *,
    condition: Array | None = None,
    key: None = None,
    state: eqx.nn.State,
    inference: bool = True,
) -> tuple[Array, Array, eqx.nn.State]:
    ...


@typing.overload
def _transform_and_log_det_direct(
    bijection: AbstractStochasticStatefulBijection,
    x: Array,
    *,
    condition: Array | None = None,
    key: PRNGKeyArray,
    state: eqx.nn.State,
    inference: bool = True,
) -> tuple[Array, Array, eqx.nn.State]:
    ...


def _transform_and_log_det_direct(
    bijection: AbstractBijection,
    x: Array,
    *,
    condition: Array | None = None,
    key: PRNGKeyArray | None = None,
    state: eqx.nn.State | None = None,
    inference: bool = True,
) -> tuple[Array, Array, eqx.nn.State | None]:
    """Apply a bijection directly, normalising the result for composition."""
    match bijection:
        case AbstractStochasticStatefulBijection():
            assert key is not None
            assert state is not None

            (x, log_abs_det_jac), state = bijection.transform_and_log_det(
                x,
                condition=condition,
                key=key,
                state=state,
                inference=inference,
            )

        case AbstractStochasticBijection():
            assert key is not None

            x, log_abs_det_jac = bijection.transform_and_log_det(
                x,
                condition=condition,
                key=key,
                inference=inference,
            )

        case AbstractStatefulBijection():
            assert state is not None

            (x, log_abs_det_jac), state = bijection.transform_and_log_det(
                x,
                condition=condition,
                state=state,
                inference=inference,
            )

        case AbstractDeterministicBijection():
            x, log_abs_det_jac = bijection.transform_and_log_det(
                x,
                condition=condition,
            )

        case _:
            raise TypeError(
                f"Unsupported bijection type: {type(bijection).__name__}."
            )

    return x, log_abs_det_jac, state


@typing.overload
def _inverse_and_log_det_direct(
    bijection: AbstractDeterministicBijection,
    y: Array,
    *,
    condition: Array | None = None,
    key: None = None,
    state: None = None,
    inference: bool = True,
) -> tuple[Array, Array, None]:
    ...


@typing.overload
def _inverse_and_log_det_direct(
    bijection: AbstractStochasticBijection,
    y: Array,
    *,
    condition: Array | None = None,
    key: PRNGKeyArray,
    state: None = None,
    inference: bool = True,
) -> tuple[Array, Array, None]:
    ...


@typing.overload
def _inverse_and_log_det_direct(
    bijection: AbstractStatefulBijection,
    y: Array,
    *,
    condition: Array | None = None,
    key: None = None,
    state: eqx.nn.State,
    inference: bool = True,
) -> tuple[Array, Array, eqx.nn.State]:
    ...


@typing.overload
def _inverse_and_log_det_direct(
    bijection: AbstractStochasticStatefulBijection,
    y: Array,
    *,
    condition: Array | None = None,
    key: PRNGKeyArray,
    state: eqx.nn.State,
    inference: bool = True,
) -> tuple[Array, Array, eqx.nn.State]:
    ...


def _inverse_and_log_det_direct(
    bijection: AbstractBijection,
    y: Array,
    *,
    condition: Array | None = None,
    key: PRNGKeyArray | None = None,
    state: eqx.nn.State | None = None,
    inference: bool = True,
) -> tuple[Array, Array, eqx.nn.State | None]:
    """Apply a bijection inverse directly, normalising the result for composition."""
    match bijection:
        case AbstractStochasticStatefulBijection():
            assert key is not None
            assert state is not None

            (y, log_abs_det_jac), state = bijection.inverse_and_log_det(
                y,
                condition=condition,
                key=key,
                state=state,
                inference=inference,
            )

        case AbstractStochasticBijection():
            assert key is not None

            y, log_abs_det_jac = bijection.inverse_and_log_det(
                y,
                condition=condition,
                key=key,
                inference=inference,
            )

        case AbstractStatefulBijection():
            assert state is not None

            (y, log_abs_det_jac), state = bijection.inverse_and_log_det(
                y,
                condition=condition,
                state=state,
                inference=inference,
            )

        case AbstractDeterministicBijection():
            y, log_abs_det_jac = bijection.inverse_and_log_det(
                y,
                condition=condition,
            )

        case _:
            raise TypeError(
                f"Unsupported bijection type: {type(bijection).__name__}."
            )

    return y, log_abs_det_jac, state

def _split_keys(
    bijections: tuple[AbstractBijection, ...],
    key: PRNGKeyArray | None,
) -> tuple[PRNGKeyArray | None, ...]:
    """Split a key across stochastic bijections only."""
    n_stochastic = sum(bijection.stochastic for bijection in bijections)

    if n_stochastic == 0:
        return (None,) * len(bijections)

    if key is None:
        raise ValueError("A key is required for stochastic bijections.")

    stochastic_keys = iter(jr.split(key, n_stochastic))

    return tuple(
        next(stochastic_keys) if bijection.stochastic else None
        for bijection in bijections
    )

