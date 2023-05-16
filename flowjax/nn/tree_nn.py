from typing import Callable, List

import jax.nn as jnn
import jax.numpy as jnp
from equinox import Module
from equinox.nn import Linear
from jax import random
from jax.random import KeyArray

from flowjax.masks import rank_based_mask
from flowjax.utils import Array, _identity
from flowjax.nn.masked_autoregressive import MaskedLinear, AutoregressiveMLP

class TreeAutoregressiveMLP():
    autoregressive_nn: Callable
    flat_to_tree: Callable

    def __init__(
        self,
        in_ranks: Array,
        out_ranks_tree: Array,
        width: int,
        depth: int,
        activation: Callable = jnn.relu,
        final_activation: Callable = _identity,
        *,
        key
    ) -> None:
        """An autoregressive multilayer perceptron, similar to ``equinox.nn.composed.MLP``.
        Connections will only exist where in_ranks < out_ranks.

        Args:
            in_ranks (Array): Ranks of the inputs.
            hidden_ranks (Array): Ranks of the hidden layer(s).
            out_ranks (Array): Ranks of the outputs.
            depth (int): Number of hidden layers.
            activation (Callable, optional): Activation function. Defaults to jnn.relu.
            final_activation (Callable, optional): Final activation function. Defaults to _identity.
            key (KeyArray): Jax PRNGKey
        """
        out_ranks, unravel = jax.flatten_util.ravel_pytree(out_ranks_tree)

        masks = []
        if depth == 0:
            masks.append(rank_based_mask(in_ranks, out_ranks, eq=False))
        else:
            masks.append(rank_based_mask(in_ranks, hidden_ranks, eq=True))
            for _ in range(depth - 1):
                masks.append(rank_based_mask(hidden_ranks, hidden_ranks, eq=True))
            masks.append(rank_based_mask(hidden_ranks, out_ranks, eq=False))

        keys = random.split(key, len(masks))
        layers = [MaskedLinear(mask, key=key) for mask, key in zip(masks, keys)]

        self.layers = layers
        self.in_size = len(in_ranks)
        self.out_size = len(out_ranks)
        self.width_size = len(hidden_ranks)
        self.depth = depth
        self.activation = activation
        self.final_activation = final_activation
        self.flat_to_tree = unravel

    def __call__(self, x: Array):
        """Forward pass.
        Args:
            x: A JAX array with shape (in_size,).
        """
        for layer in self.layers[:-1]:
            x = x + self.activation(layer(x))
        x = self.layers[-1](x)
        x = self.final_activation(x)
        return self.flat_to_tree(x)