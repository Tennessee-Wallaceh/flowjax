"""Module contains bijections formed by "stacking" other bijections."""

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

def stacked_transform_and_log_det(
    bijections: tuple[AbstractBijection, ...],
    x: Array,
    *,
    axis: int,
    condition: Array | None = None,
    key: PRNGKeyArray | None = None,
    state: eqx.nn.State | None = None,
    inference: bool = True,
):
    x_parts = (
        x_i.squeeze(axis=axis)
        for x_i in jnp.split(x, len(bijections), axis=axis)
    )
    keys = _split_keys(bijections, key)

    y_parts = []
    log_abs_det_jac = jnp.zeros(())

    for bijection, x_i, key_i in zip(
        bijections,
        x_parts,
        keys,
        strict=True,
    ):
        y_i, log_abs_det_jac_i, state = _transform_and_log_det_direct(
            bijection,
            x_i,
            condition=condition,
            key=key_i,
            state=state,
            inference=inference,
        )

        y_parts.append(y_i)
        log_abs_det_jac += log_abs_det_jac_i.sum()

    y = jnp.stack(y_parts, axis=axis)

    if any(b.stateful for b in bijections):
        assert state is not None
        return (y, log_abs_det_jac), state

    return y, log_abs_det_jac

def stacked_inverse_and_log_det(
    bijections: tuple[AbstractBijection, ...],
    y: Array,
    *,
    axis: int,
    condition: Array | None = None,
    key: PRNGKeyArray | None = None,
    state: eqx.nn.State | None = None,
    inference: bool = True,
):
    y_parts = (
        y_i.squeeze(axis=axis)
        for y_i in jnp.split(y, len(bijections), axis=axis)
    )
    keys = _split_keys(bijections, key)

    x_parts = []
    log_abs_det_jac = jnp.zeros(())

    for bijection, y_i, key_i in zip(
        bijections,
        y_parts,
        keys,
        strict=True,
    ):
        x_i, log_abs_det_jac_i, state = _inverse_and_log_det_direct(
            bijection,
            y_i,
            condition=condition,
            key=key_i,
            state=state,
            inference=inference,
        )

        x_parts.append(x_i)
        log_abs_det_jac += log_abs_det_jac_i.sum()

    x = jnp.stack(x_parts, axis=axis)

    if any(b.stateful for b in bijections):
        assert state is not None
        return (x, log_abs_det_jac), state

    return x, log_abs_det_jac

class DeterministicStack(
    AbstractDeterministicBijection[Array | None],
):
    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None
    bijections: tuple[AbstractDeterministicBijection, ...]
    axis: int

    def transform_and_log_det(
        self,
        x: Array,
        *,
        condition: Array | None = None,
    ) -> tuple[Array, Array]:
        return stacked_transform_and_log_det(
            self.bijections,
            x,
            axis=self.axis,
            condition=condition,
        )

    def inverse_and_log_det(
        self,
        y: Array,
        *,
        condition: Array | None = None,
    ) -> tuple[Array, Array]:
        return stacked_inverse_and_log_det(
            self.bijections,
            y,
            axis=self.axis,
            condition=condition,
        )


class StochasticStack(
    AbstractStochasticBijection[Array | None],
):
    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None
    bijections: tuple[AbstractBijection, ...]
    axis: int

    def transform_and_log_det(
        self,
        x: Array,
        *,
        condition: Array | None = None,
        key: PRNGKeyArray,
        inference: bool = True,
    ) -> tuple[Array, Array]:
        return stacked_transform_and_log_det(
            self.bijections,
            x,
            axis=self.axis,
            condition=condition,
            key=key,
            inference=inference,
        )

    def inverse_and_log_det(
        self,
        y: Array,
        *,
        condition: Array | None = None,
        key: PRNGKeyArray,
        inference: bool = True,
    ) -> tuple[Array, Array]:
        return stacked_inverse_and_log_det(
            self.bijections,
            y,
            axis=self.axis,
            condition=condition,
            key=key,
            inference=inference,
        )


class StatefulStack(
    AbstractStatefulBijection[Array | None],
):
    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None
    bijections: tuple[AbstractBijection, ...]
    axis: int

    def transform_and_log_det(
        self,
        x: Array,
        *,
        condition: Array | None = None,
        state: eqx.nn.State,
        inference: bool = True,
    ) -> tuple[tuple[Array, Array], eqx.nn.State]:
        return stacked_transform_and_log_det(
            self.bijections,
            x,
            axis=self.axis,
            condition=condition,
            state=state,
            inference=inference,
        )

    def inverse_and_log_det(
        self,
        y: Array,
        *,
        condition: Array | None = None,
        state: eqx.nn.State,
        inference: bool = True,
    ) -> tuple[tuple[Array, Array], eqx.nn.State]:
        return stacked_inverse_and_log_det(
            self.bijections,
            y,
            axis=self.axis,
            condition=condition,
            state=state,
            inference=inference,
        )


class StochasticStatefulStack(
    AbstractStochasticStatefulBijection[Array | None],
):
    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None
    bijections: tuple[AbstractBijection, ...]
    axis: int

    def transform_and_log_det(
        self,
        x: Array,
        *,
        condition: Array | None = None,
        key: PRNGKeyArray,
        state: eqx.nn.State,
        inference: bool = True,
    ) -> tuple[tuple[Array, Array], eqx.nn.State]:
        return stacked_transform_and_log_det(
            self.bijections,
            x,
            axis=self.axis,
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
        return stacked_inverse_and_log_det(
            self.bijections,
            y,
            axis=self.axis,
            condition=condition,
            key=key,
            state=state,
            inference=inference,
        )

def stack(
    bijections: Sequence[AbstractBijection],
    axis: int = 0,
) -> (
    DeterministicStack
    | StochasticStack
    | StatefulStack
    | StochasticStatefulStack
):
    bijections = tuple(bijections)

    shapes = [b.shape for b in bijections]
    check_shapes_match(shapes)

    shape = shapes[0][:axis] + (len(bijections),) + shapes[0][axis:]
    cond_shape = merge_cond_shapes([b.cond_shape for b in bijections])

    match (
        any(b.stochastic for b in bijections),
        any(b.stateful for b in bijections),
    ):
        case False, False:
            return DeterministicStack(
                shape=shape,
                cond_shape=cond_shape,
                bijections=bijections,
                axis=axis,
            )

        case True, False:
            return StochasticStack(
                shape=shape,
                cond_shape=cond_shape,
                bijections=bijections,
                axis=axis,
            )

        case False, True:
            return StatefulStack(
                shape=shape,
                cond_shape=cond_shape,
                bijections=bijections,
                axis=axis,
            )

        case True, True:
            return StochasticStatefulStack(
                shape=shape,
                cond_shape=cond_shape,
                bijections=bijections,
                axis=axis,
            )

        case _:
            raise TypeError()