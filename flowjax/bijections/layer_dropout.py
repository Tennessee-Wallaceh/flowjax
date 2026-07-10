import jax.numpy as jnp
import jax.random as jr
from flowjax.bijections import (
    Chain, 
    AbstractBijection,
)
import equinox as eqx
from typing import Sequence
from jaxtyping import PRNGKeyArray, Array

class DropoutChain(Chain):
    """A Chain with optional stochastic-depth style layer dropout.

    Dropout is only used when ``key`` and ``drop_prob`` are supplied to
    ``transform_and_log_det`` or ``inverse_and_log_det``. Otherwise this behaves
    like a standard Chain.

    Args:
        bijections: Sequence of bijections.
        dropout_eligible: Boolean sequence indicating which bijections may be skipped.
            Non-eligible bijections are always active.
        force_first: Whether the first bijection is always active.
        force_last: Whether the last bijection is always active.
    """

    dropout_eligible: tuple[bool, ...] = eqx.field(static=True)
    force_first: bool = eqx.field(static=True)
    force_last: bool = eqx.field(static=True)

    def __init__(
        self,
        bijections: Sequence[AbstractBijection],
        *,
        dropout_eligible: Sequence[bool] | None = None,
        force_first: bool = False,
        force_last: bool = True,
    ):
        super().__init__(bijections)

        if dropout_eligible is None:
            dropout_eligible = (True,) * len(self.bijections)
        else:
            dropout_eligible = tuple(bool(x) for x in dropout_eligible)

        if len(dropout_eligible) != len(self.bijections):
            raise ValueError(
                "`dropout_eligible` must have the same length as `bijections`."
            )

        self.dropout_eligible = dropout_eligible
        self.force_first = force_first
        self.force_last = force_last

    def sample_active_mask(
        self,
        key: PRNGKeyArray,
        *,
        drop_prob: float,
    ) -> Array:
        if not 0 <= drop_prob < 1:
            raise ValueError("`drop_prob` must satisfy 0 <= drop_prob < 1.")

        eligible = jnp.asarray(self.dropout_eligible)
        active = jr.bernoulli(
            key,
            p=1.0 - drop_prob,
            shape=(len(self.bijections),),
        )

        # Ineligible layers are always active.
        active = jnp.logical_or(active, ~eligible)

        if self.force_first:
            active = active.at[0].set(True)

        if self.force_last:
            active = active.at[-1].set(True)

        return active

    def transform_and_log_det(
        self,
        x: Array,
        condition: Array | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Array, Array]:
        if key is None or drop_prob is None or drop_prob == 0:
            return super().transform_and_log_det(x, condition)

        active = self.sample_active_mask(key, drop_prob=drop_prob)
        return self._transform_and_log_det_with_mask(x, condition, active)

    def inverse_and_log_det(
        self,
        y: Array,
        condition: Array | None = None,
        *,
        key: PRNGKeyArray | None = None,
        drop_prob: float | None = None,
    ) -> tuple[Array, Array]:
        if key is None or drop_prob is None or drop_prob == 0:
            return super().inverse_and_log_det(y, condition)

        active = self.sample_active_mask(key, drop_prob=drop_prob)
        return self._inverse_and_log_det_with_mask(y, condition, active)

    def _transform_and_log_det_with_mask(
        self,
        x: Array,
        condition: Array | None,
        active: Array,
    ) -> tuple[Array, Array]:
        log_abs_det_jac = jnp.zeros(())

        for i, bijection in enumerate(self.bijections):
            x_new, log_abs_det_jac_i = bijection.transform_and_log_det(
                x,
                condition,
            )

            active_i = active[i]
            x = jnp.where(active_i, x_new, x)
            log_abs_det_jac = log_abs_det_jac + jnp.where(
                active_i,
                log_abs_det_jac_i.sum(),
                0.0,
            )

        return x, log_abs_det_jac

    def _inverse_and_log_det_with_mask(
        self,
        y: Array,
        condition: Array | None,
        active: Array,
    ) -> tuple[Array, Array]:
        log_abs_det_jac = jnp.zeros(())

        for i, bijection in reversed(tuple(enumerate(self.bijections))):
            y_new, log_abs_det_jac_i = bijection.inverse_and_log_det(
                y,
                condition,
            )

            active_i = active[i]
            y = jnp.where(active_i, y_new, y)
            log_abs_det_jac = log_abs_det_jac + jnp.where(
                active_i,
                log_abs_det_jac_i.sum(),
                0.0,
            )

        return y, log_abs_det_jac

    def __getitem__(self, i: int | slice) -> AbstractBijection:
        if isinstance(i, int):
            return self.bijections[i]

        if isinstance(i, slice):
            return DropoutChain(
                self.bijections[i],
                dropout_eligible=self.dropout_eligible[i],
                force_first=self.force_first,
                force_last=self.force_last,
            )

        raise TypeError(f"Indexing with type {type(i)} is not supported.")

    def merge_chains(self):
        """Returns an equivalent DropoutChain with nested chains flattened."""
        bijections = []
        dropout_eligible = []

        for b, eligible in zip(
            self.bijections,
            self.dropout_eligible,
            strict=True,
        ):
            if isinstance(b, DropoutChain):
                bijections.extend(b.bijections)
                dropout_eligible.extend(b.dropout_eligible)
            elif isinstance(b, Chain):
                bijections.extend(b.bijections)
                dropout_eligible.extend([eligible] * len(b.bijections))
            else:
                bijections.append(b)
                dropout_eligible.append(eligible)

        return DropoutChain(
            bijections,
            dropout_eligible=dropout_eligible,
            force_first=self.force_first,
            force_last=self.force_last,
        )