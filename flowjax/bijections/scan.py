"""Bijections that wrap JAX function transforms (scan and vmap)."""

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
from jax.lax import scan
from jax.tree_util import tree_leaves, tree_map
from jaxtyping import PyTree, Array, PRNGKeyArray, ArrayLike
import jax.random as jr

from flowjax.bijections.bijection import AbstractBijection
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

def _scan_length(bijection: AbstractBijection) -> int:
    params = eqx.filter(bijection, eqx.is_array)
    leaves = tree_leaves(params)

    if not leaves:
        raise ValueError(
            "Cannot scan a bijection with no array leaves."
        )

    return leaves[0].shape[0]

def _filter_scan(
    f,
    init,
    bijection,
    *scan_xs,
    reverse: bool = False,
):
    params, static = eqx.partition(
        bijection,
        filter_spec=eqx.is_array,
    )

    def _scan_fn(carry, xs):
        params_i, *scan_xs_i = xs
        bijection_i = eqx.combine(params_i, static)

        return f(
            carry,
            bijection_i,
            *scan_xs_i,
        )

    return scan(
        _scan_fn,
        init,
        (params, *scan_xs),
        reverse=reverse,
    )


def scanned_transform_and_log_det(
    bijection: AbstractBijection,
    x: Array,
    *,
    condition: Array | None = None,
    key: PRNGKeyArray | None = None,
    state: eqx.nn.State | None = None,
    inference: bool = True,
):
    if bijection.stochastic:
        if key is None:
            raise ValueError(
                "A key is required for a stochastic bijection."
            )

        keys = jrandom.split(key, _scan_length(bijection))

        def step(carry, bijection_i, key_i):
            x, log_abs_det_jac, state = carry

            x, log_abs_det_jac_i, state = _transform_and_log_det_direct(
                bijection_i,
                x,
                condition=condition,
                key=key_i,
                state=state,
                inference=inference,
            )

            return (
                x,
                log_abs_det_jac + log_abs_det_jac_i.sum(),
                state,
            ), None

        (y, log_abs_det_jac, state), _ = _filter_scan(
            step,
            (x, jnp.zeros(()), state),
            bijection,
            keys,
        )

    else:
        def step(carry, bijection_i):
            x, log_abs_det_jac, state = carry

            x, log_abs_det_jac_i, state = _transform_and_log_det_direct(
                bijection_i,
                x,
                condition=condition,
                state=state,
                inference=inference,
            )

            return (
                x,
                log_abs_det_jac + log_abs_det_jac_i.sum(),
                state,
            ), None

        (y, log_abs_det_jac, state), _ = _filter_scan(
            step,
            (x, jnp.zeros(()), state),
            bijection,
        )

    if bijection.stateful:
        assert state is not None
        return (y, log_abs_det_jac), state

    return y, log_abs_det_jac
def scanned_inverse_and_log_det(
    bijection: AbstractBijection,
    y: Array,
    *,
    condition: Array | None = None,
    key: PRNGKeyArray | None = None,
    state: eqx.nn.State | None = None,
    inference: bool = True,
):
    if bijection.stochastic:
        if key is None:
            raise ValueError(
                "A key is required for a stochastic bijection."
            )

        keys = jrandom.split(key, _scan_length(bijection))

        def step(carry, bijection_i, key_i):
            y, log_abs_det_jac, state = carry

            y, log_abs_det_jac_i, state = _inverse_and_log_det_direct(
                bijection_i,
                y,
                condition=condition,
                key=key_i,
                state=state,
                inference=inference,
            )

            return (
                y,
                log_abs_det_jac + log_abs_det_jac_i.sum(),
                state,
            ), None

        (x, log_abs_det_jac, state), _ = _filter_scan(
            step,
            (y, jnp.zeros(()), state),
            bijection,
            keys,
            reverse=True,
        )

    else:
        def step(carry, bijection_i):
            y, log_abs_det_jac, state = carry

            y, log_abs_det_jac_i, state = _inverse_and_log_det_direct(
                bijection_i,
                y,
                condition=condition,
                state=state,
                inference=inference,
            )

            return (
                y,
                log_abs_det_jac + log_abs_det_jac_i.sum(),
                state,
            ), None

        (x, log_abs_det_jac, state), _ = _filter_scan(
            step,
            (y, jnp.zeros(()), state),
            bijection,
            reverse=True,
        )

    if bijection.stateful:
        assert state is not None
        return (x, log_abs_det_jac), state

    return x, log_abs_det_jac


class DeterministicScan(
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
        return scanned_transform_and_log_det(
            self.bijection,
            x,
            condition=condition,
        )

    def inverse_and_log_det(
        self,
        y: ArrayLike,
        *,
        condition: Array | None = None,
    ) -> tuple[Array, Array]:
        return scanned_inverse_and_log_det(
            self.bijection,
            y,
            condition=condition,
        )
    
class StatefulScan(
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
        return scanned_transform_and_log_det(
            self.bijection,
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
        return scanned_inverse_and_log_det(
            self.bijection,
            y,
            condition=condition,
            state=state,
            inference=inference,
        )

class StochasticScan(
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
        return scanned_transform_and_log_det(
            self.bijection,
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
        return scanned_inverse_and_log_det(
            self.bijection,
            y,
            condition=condition,
            key=key,
            inference=inference,
        )
     
class StochasticStatefulScan(
    AbstractStochasticStatefulBijection[Array | None],
):
    bijection: AbstractStochasticStatefulBijection
    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None

    def transform_and_log_det(
        self,
        x: ArrayLike,
        *,
        condition: Array | None = None,
        key: PRNGKeyArray,
        state: eqx.nn.State,
        inference: bool = True,
    ) -> tuple[tuple[Array, Array], eqx.nn.State]:
        return scanned_transform_and_log_det(
            self.bijection,
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
        return scanned_inverse_and_log_det(
            self.bijection,
            y,
            condition=condition,
            key=key,
            state=state,
            inference=inference,
        )
    
def scan_bijection(
    bijection: AbstractBijection,
) -> (
    DeterministicScan
    | StochasticScan
    | StatefulScan
    | StochasticStatefulScan
):
    match bijection:
        case AbstractStochasticStatefulBijection():
            return StochasticStatefulScan(
                bijection=bijection,
                shape=bijection.shape,
                cond_shape=bijection.cond_shape,
            )

        case AbstractStochasticBijection():
            return StochasticScan(
                bijection=bijection,
                shape=bijection.shape,
                cond_shape=bijection.cond_shape,
            )

        case AbstractStatefulBijection():
            return StatefulScan(
                bijection=bijection,
                shape=bijection.shape,
                cond_shape=bijection.cond_shape,
            )

        case AbstractDeterministicBijection():
            return DeterministicScan(
                bijection=bijection,
                shape=bijection.shape,
                cond_shape=bijection.cond_shape,
            )

        case _:
            raise TypeError(
                f"Unsupported bijection type: {type(bijection).__name__}."
            )