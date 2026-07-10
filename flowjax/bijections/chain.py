"""Chain bijection which allows sequential application of arbitrary bijections."""

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import PRNGKeyArray, Array, ArrayLike

from flowjax.bijections.bijection import (
    AbstractBijection,
    AbstractDeterministicBijection,
    AbstractStochasticBijection,
    AbstractStatefulBijection,
    AbstractStochasticStatefulBijection,
    _transform_and_log_det_direct,
    _inverse_and_log_det_direct,
    _split_keys,
)
from flowjax.utils import check_shapes_match, merge_cond_shapes

def chained_transform_and_log_det(
    bijections: tuple[AbstractBijection, ...],
    x: Array,
    *,
    condition: Array | None = None,
    key: PRNGKeyArray | None = None,
    state: eqx.nn.State | None = None,
    inference: bool = True,
) -> (
    tuple[Array, Array]
    | tuple[tuple[Array, Array], eqx.nn.State]
):
    """Apply a sequence of bijections in the forward direction."""
    log_abs_det_jac = jnp.zeros(())
    keys = _split_keys(bijections, key)

    for bijection, key_i in zip(bijections, keys):
        x, log_abs_det_jac_i, state = _transform_and_log_det_direct(
            bijection,
            x,
            condition=condition,
            key=key_i,
            state=state,
            inference=inference,
        )

        log_abs_det_jac += log_abs_det_jac_i.sum()

    if any(bijection.stateful for bijection in bijections):
        assert state is not None
        return (x, log_abs_det_jac), state

    return x, log_abs_det_jac

def chained_inverse_and_log_det(
    bijections: tuple[AbstractBijection, ...],
    y: Array,
    *,
    condition: Array | None = None,
    key: PRNGKeyArray | None = None,
    state: eqx.nn.State | None = None,
    inference: bool = True,
) -> (
    tuple[Array, Array]
    | tuple[tuple[Array, Array], eqx.nn.State]
):
    """Apply a sequence of bijections in the inverse direction."""
    log_abs_det_jac = jnp.zeros(())
    keys = _split_keys(bijections, key)

    for bijection, key_i in zip(
        reversed(bijections),
        reversed(keys),
    ):
        y, log_abs_det_jac_i, state = _inverse_and_log_det_direct(
            bijection,
            y,
            condition=condition,
            key=key_i,
            state=state,
            inference=inference,
        )

        log_abs_det_jac += log_abs_det_jac_i.sum()

    if any(bijection.stateful for bijection in bijections):
        assert state is not None
        return (y, log_abs_det_jac), state

    return y, log_abs_det_jac


class _AbstractChain:
    bijections: tuple[AbstractBijection, ...]

    def __getitem__(self, i: int | slice) -> AbstractBijection | "_AbstractChain":
        if isinstance(i, int):
            return self.bijections[i]

        if isinstance(i, slice):
            return chain(self.bijections[i])

        raise TypeError(
            f"Indexing with type {type(i)} is not supported."
        )

    def __iter__(self):
        yield from self.bijections

    def __len__(self):
        return len(self.bijections)
    
class DeterministicChain(
    AbstractDeterministicBijection,
):
    bijections: tuple[AbstractDeterministicBijection, ...]

    def transform_and_log_det(
        self,
        x: ArrayLike,
        *,
        condition: Array | None = None,
    ) -> tuple[Array, Array]:
        return chained_transform_and_log_det(
            self.bijections,
            x,
            condition=condition,
        )

    def inverse_and_log_det(
        self,
        y: ArrayLike,
        *,
        condition: Array | None = None,
    ) -> tuple[Array, Array]:
        return chained_inverse_and_log_det(
            self.bijections,
            y,
            condition=condition,
        )
    
class StochasticChain(
    AbstractStochasticBijection[Array | None],
):
    bijections: tuple[AbstractBijection, ...]

    def transform_and_log_det(
        self,
        x: ArrayLike,
        *,
        condition: Array | None = None,
        key: PRNGKeyArray,
        inference: bool = True,
    ) -> tuple[Array, Array]:
        return chained_transform_and_log_det(
            self.bijections,
            x,
            condition=condition,
            key=key,
            inference=inference,
        )

    def inverse_and_log_det(
        self,
        y: ArrayLike,
        *,
        condition: Array | None = None,
        key: PRNGKeyArray,
        inference: bool = True,
    ) -> tuple[Array, Array]:
        return chained_inverse_and_log_det(
            self.bijections,
            y,
            condition=condition,
            key=key,
            inference=inference,
        )
    
class StatefulChain(
    AbstractStatefulBijection[Array | None],
):
    bijections: tuple[AbstractBijection, ...]

    def transform_and_log_det(
        self,
        x: ArrayLike,
        *,
        condition: Array | None = None,
        state: eqx.nn.State,
        inference: bool = True,
    ) -> tuple[tuple[Array, Array], eqx.nn.State]:
        return chained_transform_and_log_det(
            self.bijections,
            x,
            condition=condition,
            state=state,
            inference=inference,
        )

    def inverse_and_log_det(
        self,
        y: ArrayLike,
        *,
        condition: Array | None = None,
        state: eqx.nn.State,
        inference: bool = True,
    ) -> tuple[tuple[Array, Array], eqx.nn.State]:
        return chained_inverse_and_log_det(
            self.bijections,
            y,
            condition=condition,
            state=state,
            inference=inference,
        )

class StochasticStatefulChain(
    AbstractStochasticStatefulBijection[Array | None],
):
    bijections: tuple[AbstractBijection, ...]

    def transform_and_log_det(
        self,
        x: ArrayLike,
        *,
        condition: Array | None = None,
        key: PRNGKeyArray,
        state: eqx.nn.State,
        inference: bool = True,
    ) -> tuple[tuple[Array, Array], eqx.nn.State]:
        return chained_transform_and_log_det(
            self.bijections,
            x,
            condition=condition,
            key=key,
            state=state,
            inference=inference,
        )

    def inverse_and_log_det(
        self,
        y: ArrayLike,
        *,
        condition: Array | None = None,
        key: PRNGKeyArray,
        state: eqx.nn.State,
        inference: bool = True,
    ) -> tuple[tuple[Array, Array], eqx.nn.State]:
        return chained_inverse_and_log_det(
            self.bijections,
            y,
            condition=condition,
            key=key,
            state=state,
            inference=inference,
        )

def merge_chains(bijections):
    """Return an equivalent chain with nested chains flattened."""

    while any(isinstance(b, _AbstractChain) for b in bijections):
        merged = []

        for bijection in bijections:
            if isinstance(bijection, _AbstractChain):
                merged.extend(bijection.bijections)
            else:
                merged.append(bijection)

        bijections = tuple(merged)

    return chain(bijections)
    
def chain(
    bijections: Sequence[AbstractBijection],
) -> (
    DeterministicChain
    | StochasticChain
    | StatefulChain
    | StochasticStatefulChain
):
    bijections = tuple(merge_chains(bijections))

    check_shapes_match([b.shape for b in bijections])

    shape = bijections[0].shape
    cond_shape = merge_cond_shapes([b.cond_shape for b in bijections])

    match (
        any(b.stochastic for b in bijections),
        any(b.stateful for b in bijections),
    ):
        case False, False:
            return DeterministicChain(
                shape=shape,
                cond_shape=cond_shape,
                bijections=bijections,
            )

        case True, False:
            return StochasticChain(
                shape=shape,
                cond_shape=cond_shape,
                bijections=bijections,
            )

        case False, True:
            return StatefulChain(
                shape=shape,
                cond_shape=cond_shape,
                bijections=bijections,
            )

        case True, True:
            return StochasticStatefulChain(
                shape=shape,
                cond_shape=cond_shape,
                bijections=bijections,
            )

        case _:
            raise TypeError()