"""Masked autoregressive network and bijection."""

from functools import partial
from typing import Callable, Union, Type
import jax.random as jr
import jax
import jax.nn as jnn
import jax.numpy as jnp
from jax.random import KeyArray
import equinox as eqx

from flowjax.bijections import Bijection
from flowjax.nn import AutoregressiveMLP

from flowjax.utils import Array, tile_until_length

class TreeAutoregressiveMLP(AutoregressiveMLP):
    flat_to_tree: callable
    def __init__(self, *args, out_ranks_tree, **kwargs):
        out_ranks, self.flat_to_tree = jax.flatten_util.ravel_pytree(out_ranks_tree)
        super().__init__(*args, out_ranks=out_ranks, **kwargs)

    def __call__(self, x):
        return self.flat_to_tree(super().__call__(x))

class TreeMaskedAutoregressive(Bijection):
    target_bijection_static: Bijection
    autoregressive_mlp: AutoregressiveMLP
    def __init__(
        self,
        key,
        target_bijection: Type[Bijection],
        dim: int,
        cond_dim: Union[None, int],
        nn_width: int,
        nn_depth: int,
        nn_activation: Callable = jnn.relu,
    ) -> None:
        """
        Implementation of a masked autoregressive bijection (https://arxiv.org/abs/1705.07057v4).
        The bijection is parameterised by a neural network, with weights masked to ensure
        an autoregressive structure.

        Args:
            key (KeyArray): Jax PRNGKey
            bijection_definition (Bijection): Bijection with shape () to be parameterised by the autoregressive network.
            dim (int): Dimension.
            cond_dim (Union[None, int]): Dimension of any conditioning variables.
            nn_width (int): Neural network width.
            nn_depth (int): Neural network depth.
            nn_activation (Callable, optional): Neural network activation. Defaults to jnn.relu.
        """
        if target_bijection.shape != () or target_bijection.cond_shape is not None:
            raise ValueError(
                "Currently, only unconditional scalar bijections are supported."
            )
        
        # in_ranks
        if cond_dim is None:
            self.cond_shape = None
            in_ranks = jnp.arange(dim)
        else:
            self.cond_shape = (cond_dim,)
            # we give conditioning variables rank -1 (no masking of edges to output)
            in_ranks = jnp.hstack((jnp.arange(dim), -jnp.ones(cond_dim)))

        # hidden_ranks
        hidden_ranks = tile_until_length(jnp.arange(dim), nn_width)

        # out_ranks
        target_bijection_params, target_bijection_static = eqx.partition(target_bijection, eqx.is_inexact_array)
        out_ranks_tree = jax.tree_util.tree_map(lambda leaf: jnp.arange(dim), target_bijection_params)
        self.target_bijection_static = target_bijection_static

        # given input, produces PyTree of the target bijection parameters 
        self.autoregressive_mlp = TreeAutoregressiveMLP(
            in_ranks=in_ranks,
            hidden_ranks=hidden_ranks,
            out_ranks_tree=out_ranks_tree,
            depth=nn_depth,
            key=key,
        )

        self.shape = (dim,)
        self.cond_shape = None if cond_dim is None else (cond_dim,)

    def transform(self, x, condition=()):
        nn_input = jnp.hstack((x, *condition))
        bijection = eqx.combine(
            self.target_bijection_static, 
            self.autoregressive_mlp(nn_input),
        )
        return bijection.transform(x)

    def transform_and_log_abs_det_jacobian(self, x, condition=()):
        nn_input = jnp.hstack((x, *condition))
        bijection = eqx.combine(
            self.target_bijection_static, 
            self.autoregressive_mlp(nn_input),
        )
        return bijection.transform_and_log_abs_det_jacobian(x)

    def inverse(self, y, condition=()):
        init = (y, 0)
        fn = partial(self.inv_scan_fn, condition=condition)
        (x, _), _ = jax.lax.scan(fn, init, None, length=len(y))
        return x

    def inv_scan_fn(self, init, _, condition=()):
        "One 'step' in computing the inverse"
        y, rank = init
        nn_input = jnp.hstack((y, *condition))
        bijection = eqx.combine(
            self.target_bijection_static, 
            self.autoregressive_mlp(nn_input),
        )
        x = bijection.inverse(y)
        x = y.at[rank].set(x[rank])
        return (x, rank + 1), None

    def inverse_and_log_abs_det_jacobian(self, y, *condition):
        x = self.inverse(y, *condition)
        log_det = self.transform_and_log_abs_det_jacobian(x, *condition)[1]
        return x, -log_det