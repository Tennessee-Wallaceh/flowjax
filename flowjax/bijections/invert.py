"""Module contains ways to invert bijections"""

from collections.abc import Sequence
from itertools import accumulate

import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

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

class DeterministicInvert(
    AbstractDeterministicBijection[Array | None],
):
    bijection: AbstractDeterministicBijection
    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None

    def transform_and_log_det(
        self,
        x: ArrayLike,
        *,
        condition: Array | None = None,
    ) -> tuple[Array, Array]:
        return self.bijection.inverse_and_log_det(
            x,
            condition=condition,
        )

    def inverse_and_log_det(
        self,
        y: ArrayLike,
        *,
        condition: Array | None = None,
    ) -> tuple[Array, Array]:
        return self.bijection.transform_and_log_det(
            y,
            condition=condition,
        )
    
class StochasticInvert(
    AbstractStochasticBijection[Array | None],
):
    bijection: AbstractStochasticBijection
    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None

    def transform_and_log_det(
        self,
        x: ArrayLike,
        *,
        condition: Array | None = None,
        key: PRNGKeyArray,
        inference: bool = True,
    ) -> tuple[Array, Array]:
        return self.bijection.inverse_and_log_det(
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
        return self.bijection.transform_and_log_det(
            y,
            condition=condition,
            key=key,
            inference=inference,
        )

class StatefulInvert(
    AbstractStatefulBijection[Array | None],
):
    bijection: AbstractStatefulBijection
    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None

    def transform_and_log_det(
        self,
        x: ArrayLike,
        *,
        condition: Array | None = None,
        state: eqx.nn.State,
        inference: bool = True,
    ) -> tuple[tuple[Array, Array], eqx.nn.State]:
        return self.bijection.inverse_and_log_det(
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
        return self.bijection.transform_and_log_det(
            y,
            condition=condition,
            state=state,
            inference=inference,
        )

class StochasticStatefulInvert(
    AbstractStochasticStatefulBijection[Array | None],
):
    bijection: AbstractStochasticStatefulBijection
    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None

    def transform_and_log_det(
        self,
        x: Array,
        *,
        condition: Array | None = None,
        key: PRNGKeyArray,
        state: eqx.nn.State,
        inference: bool = True,
    ) -> tuple[tuple[Array, Array], eqx.nn.State]:
        return self.bijection.inverse_and_log_det(
            x,
            condition=condition,
            key=key,
            state=state,
            inference=inference,
        )

    def inverse_and_log_det(
        self,
        y: Array,
        *,
        condition: Array | None = None,
        key: PRNGKeyArray,
        state: eqx.nn.State,
        inference: bool = True,
    ) -> tuple[tuple[Array, Array], eqx.nn.State]:
        return self.bijection.transform_and_log_det(
            y,
            condition=condition,
            key=key,
            state=state,
            inference=inference,
        )
    
def invert(
    bijection: AbstractBijection,
) -> (
    DeterministicInvert
    | StochasticInvert
    | StatefulInvert
    | StochasticStatefulInvert
):
    match bijection:
        case AbstractStochasticStatefulBijection():
            return StochasticStatefulInvert(
                bijection=bijection,
                shape=bijection.shape,
                cond_shape=bijection.cond_shape
            )

        case AbstractStochasticBijection():
            return StochasticInvert(
                bijection=bijection,
                shape=bijection.shape,
                cond_shape=bijection.cond_shape
            )

        case AbstractStatefulBijection():
            return StatefulInvert(
                bijection=bijection,
                shape=bijection.shape,
                cond_shape=bijection.cond_shape
            )

        case AbstractDeterministicBijection():
            return DeterministicInvert(
                bijection=bijection,
                shape=bijection.shape,
                cond_shape=bijection.cond_shape
            )

        case _:
            raise TypeError(
                f"Unsupported bijection type: {type(bijection).__name__}."
            )