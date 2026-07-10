"""Masked autoregressive network and bijection."""

from collections.abc import Callable
from functools import partial
from typing import Protocol, Literal, ClassVar, runtime_checkable
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.nn as jnn
import jax.random as jr
import jax.numpy as jnp
from jaxtyping import Array, Int, PRNGKeyArray

from flowjax.bijections.bijection import AbstractStochasticStatefulBijection
from flowjax.bijections.jax_transforms import Vmap
from flowjax.masks import rank_based_mask
from flowjax.parameters import MaskedWeightParameter
from flowjax.utils import get_ravelled_pytree_constructor

@dataclass(kw_only=True, frozen=True, slots=True)
class _BaseMaskedNNConfig:
    width: int
    dropout: float
    activation: Callable = jnn.relu

    def __post_init__(self):
        if not 0 <= self.dropout < 1:
            raise ValueError(
                f"Dropout must be in [0, 1), got {self.dropout:.3f}."
            )

@dataclass(kw_only=True, frozen=True, slots=True)
class FeedForwardMaskedNNConfig(_BaseMaskedNNConfig):
    depth: int

@dataclass(kw_only=True, frozen=True, slots=True)
class ResidualMaskedNNConfig(_BaseMaskedNNConfig):
    num_blocks: int = 1
    block_size: int = 2


MaskedNNConfig = FeedForwardMaskedNNConfig | ResidualMaskedNNConfig





@runtime_checkable
class SupportsMaskedWeight(Protocol):
    @property
    def value(self) -> Array: ...


class MaskedLinear(eqx.Module):
    weight: SupportsMaskedWeight
    bias: Array | None

    def __call__(self, x: Array) -> Array:
        if x.ndim != 1:
            raise ValueError(f"Expected x.ndim == 1; got {x.ndim}.")
        weight = self.weight.value
        if weight.ndim != 2:
            raise ValueError(f"Expected weight.ndim == 2; got {weight.ndim}.")
        if weight.shape[1] != x.shape[0]:
            raise ValueError(
                "Input dimension mismatch in MaskedLinear: "
                f"expected x.shape[0] == {weight.shape[1]}, got {x.shape[0]}.",
            )
        y = weight @ x
        if self.bias is not None:
            if self.bias.shape != (weight.shape[0],):
                raise ValueError(
                    "Bias shape mismatch in MaskedLinear: "
                    f"expected {(weight.shape[0],)}, got {self.bias.shape}.",
                )
            y = y + self.bias
        return y

def _masked_ranks(
    dim: int,
    out_dim: int,
    cond_dim: int | None,
    width: int,
) -> tuple[Array, Array, Array]:
    # If dim=1, hidden ranks all zero -> all weights masked out in final layer
    # we give conditioning variables rank -1 (no masking of edges to output)
    # If dim=1, hidden ranks all -1 -> all outputs only depend on condition
    if cond_dim is None:
        in_ranks = jnp.arange(dim)

        if dim == 1:
            hidden_ranks = jnp.zeros(width, dtype=int)
        else:
            hidden_ranks = jnp.arange(width) % (dim - 1)
    else:
        in_ranks = jnp.hstack(
            (jnp.arange(dim), -jnp.ones(cond_dim, dtype=int))
        )
        hidden_ranks = (jnp.arange(width) % dim) - 1

    out_ranks = jnp.repeat(jnp.arange(dim), out_dim)

    return in_ranks, hidden_ranks, out_ranks

class MaskedMLP(eqx.Module):
    layers: tuple[MaskedLinear, ...]
    activation: Callable
    final_activation: Callable
    use_final_bias: bool
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        dim: int,
        out_dim: int,
        cond_dim: int | None = None,
        *,
        config: FeedForwardMaskedNNConfig,
        key: PRNGKeyArray,
    ):
        in_ranks, hidden_ranks, out_ranks = _masked_ranks(
            dim,
            out_dim,
            cond_dim,
            config.width,
        )

        mlp = eqx.nn.MLP(
            in_size=len(in_ranks),
            out_size=len(out_ranks),
            width_size=len(hidden_ranks),
            depth=config.depth,
            activation=config.activation,
            key=key,
        )

        ranks = [
            in_ranks,
            *[hidden_ranks] * (len(mlp.layers) - 1),
            out_ranks,
        ]

        self.layers = tuple(
            MaskedLinear(
                weight=MaskedWeightParameter(
                    linear.weight,
                    rank_based_mask(
                        ranks[i],
                        ranks[i + 1],
                        eq=i != len(mlp.layers) - 1,
                    ),
                ),
                bias=linear.bias,
            )
            for i, linear in enumerate(mlp.layers)
        )

        self.activation = mlp.activation
        self.final_activation = mlp.final_activation
        self.use_final_bias = mlp.use_final_bias
        self.dropout = eqx.nn.Dropout(config.dropout)

    def __call__(
        self,
        x: Array,
        *,
        key: PRNGKeyArray | None = None,
        inference: bool = True,
    ) -> Array:
        hidden_layers = self.layers[1:-1]

        keys: PRNGKeyArray | list[None]
        if key is None:
            keys = [None] * len(hidden_layers)
        else:
            keys = jr.split(key, len(hidden_layers))

        x = self.activation(self.layers[0](x))

        for layer, dropout_key in zip(hidden_layers, keys, strict=True):
            x = self.activation(layer(x))
            x = self.dropout(
                x,
                key=dropout_key,
                inference=inference,
            )

        x = self.layers[-1](x)

        if self.final_activation is not None:
            x = self.final_activation(x)

        return x

class MaskedResidualMLP(eqx.Module):
    layers: tuple[MaskedLinear, ...]
    activation: Callable
    final_activation: Callable
    use_final_bias: bool
    dropout: eqx.nn.Dropout
    block_size: int

    def __init__(
        self,
        dim: int,
        out_dim: int,
        cond_dim: int | None = None,
        *,
        config: ResidualMaskedNNConfig,
        key: PRNGKeyArray,
    ):
        in_ranks, hidden_ranks, out_ranks = _masked_ranks(
            dim,
            out_dim,
            cond_dim,
            config.width,
        )

        mlp_key, residual_init_key = jr.split(key)

        num_hidden_layers = config.num_blocks * config.block_size

        mlp = eqx.nn.MLP(
            in_size=len(in_ranks),
            out_size=len(out_ranks),
            width_size=len(hidden_ranks),
            depth=num_hidden_layers,
            activation=config.activation,
            key=mlp_key,
        )

        ranks = [
            in_ranks,
            *[hidden_ranks] * (len(mlp.layers) - 1),
            out_ranks,
        ]

        residual_final_indices = set(
            range(
                config.block_size,
                num_hidden_layers + 1,
                config.block_size,
            )
        )

        residual_init_keys = iter(
            jr.split(
                residual_init_key,
                2 * len(residual_final_indices),
            )
        )

        masked_layers = []

        for i, linear in enumerate(mlp.layers):
            weight = linear.weight
            bias = linear.bias

            if i in residual_final_indices:
                weight = jr.uniform(
                    next(residual_init_keys),
                    weight.shape,
                    minval=-1e-3,
                    maxval=1e-3,
                )

                if bias is not None:
                    bias = jr.uniform(
                        next(residual_init_keys),
                        bias.shape,
                        minval=-1e-3,
                        maxval=1e-3,
                    )

            mask = rank_based_mask(
                ranks[i],
                ranks[i + 1],
                eq=i != len(mlp.layers) - 1,
            )

            masked_layers.append(
                MaskedLinear(
                    weight=MaskedWeightParameter(weight, mask),
                    bias=bias,
                )
            )

        self.layers = tuple(masked_layers)
        self.activation = mlp.activation
        self.final_activation = mlp.final_activation
        self.use_final_bias = mlp.use_final_bias
        self.dropout = eqx.nn.Dropout(config.dropout)
        self.block_size = config.block_size

    def __call__(
        self,
        x: Array,
        *,
        key: PRNGKeyArray | None = None,
        inference: bool = True,
    ) -> Array:
        hidden_layers = self.layers[1:-1]
        num_blocks = len(hidden_layers) // self.block_size

        keys: PRNGKeyArray | list[None]
        if key is None:
            keys = [None] * num_blocks
        else:
            keys = jr.split(key, num_blocks)

        x = self.layers[0](x)

        for block_idx, dropout_key in enumerate(keys):
            start = block_idx * self.block_size
            block = hidden_layers[start : start + self.block_size]

            residual = x

            for layer_idx, layer in enumerate(block):
                x = self.activation(x)

                if layer_idx == self.block_size - 1:
                    x = self.dropout(
                        x,
                        key=dropout_key,
                        inference=inference,
                    )

                x = layer(x)

            x = x + residual

        x = self.layers[-1](x)

        if self.final_activation is not None:
            x = self.final_activation(x)

        return x


class MaskedAutoregressive(
    AbstractStochasticStatefulBijection[Array | None]
):
    """Masked autoregressive bijection."""

    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None
    transformer_constructor: Callable
    masked_autoregressive_mlp: MaskedMLP | MaskedResidualMLP

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        transformer: AbstractBijection,
        dim: int,
        nn_config: MaskedNNConfig,
        cond_dim: int | None = None,
    ) -> None:
        if transformer.shape != () or transformer.cond_shape is not None:
            raise ValueError(
                "Only unconditional transformers with shape () are supported.",
            )

        constructor, num_params = get_ravelled_pytree_constructor(
            transformer,
            filter_spec=eqx.is_inexact_array,
        )

        self.masked_autoregressive_mlp = masked_autoregressive_mlp(
            dim,
            num_params,
            cond_dim,
            config=nn_config,
            key=key,
        )

        self.transformer_constructor = constructor
        self.shape = (dim,)
        self.cond_shape = None if cond_dim is None else (cond_dim,)

    def transform_and_log_det(
        self,
        x: ArrayLike,
        *,
        condition: Array | None = None,
        key: PRNGKeyArray,
        state: eqx.nn.State,
        inference: bool = True,
    ) -> tuple[tuple[Array, Array], eqx.nn.State]:
        nn_input = x if condition is None else jnp.hstack((x, condition))

        transformer_params, state = self.masked_autoregressive_mlp(
            nn_input,
            key=key,
            state=state,
            inference=inference,
        )
        transformer = self._flat_params_to_transformer(transformer_params)

        return transformer.transform_and_log_det(x), state

    def inverse_and_log_det(
        self,
        y: ArrayLike,
        *,
        condition: Array | None = None,
        key: PRNGKeyArray,
        state: eqx.nn.State,
        inference: bool = True,
    ) -> tuple[tuple[Array, Array], eqx.nn.State]:
        init = (y, 0, state)

        fn = partial(
            self.inv_scan_fn,
            condition=condition,
            key=key,
            inference=inference,
        )

        (x, _, state), _ = jax.lax.scan(
            fn,
            init,
            None,
            length=len(y),
        )

        (_, log_det), state = self.transform_and_log_det(
            x,
            condition=condition,
            key=key,
            state=state,
            inference=inference,
        )

        return (x, -log_det), state

    def inv_scan_fn(
        self,
        init,
        _,
        *,
        condition,
        key,
        inference,
    ):
        """One step in computing the inverse."""
        y, rank, state = init

        nn_input = y if condition is None else jnp.hstack((y, condition))

        transformer_params, state = self.masked_autoregressive_mlp(
            nn_input,
            key=key,
            state=state,
            inference=inference,
        )
        transformer = self._flat_params_to_transformer(transformer_params)

        x = transformer.inverse(y)
        x = y.at[rank].set(x[rank])

        return (x, rank + 1, state), None

    def _flat_params_to_transformer(self, params: Array):
        """Reshape to dim X params_per_dim, then vmap."""
        dim = self.shape[-1]
        transformer_params = jnp.reshape(params, (dim, -1))

        return eqx.filter_vmap(
            self.transformer_constructor
        )(transformer_params)
    

def masked_autoregressive_mlp(
    dim: int,
    out_dim: int,
    cond_dim: int | None = None,
    *,
    config: MaskedNNConfig,
    key: PRNGKeyArray,
) -> MaskedMLP | MaskedResidualMLP:
    

    match config:
        case FeedForwardMaskedNNConfig():
            return MaskedMLP(
                dim,
                out_dim,
                cond_dim,
                config=config,
                key=key,
            )

        case ResidualMaskedNNConfig():
            return MaskedResidualMLP(
                dim,
                out_dim,
                cond_dim,
                config=config,
                key=key,
            )