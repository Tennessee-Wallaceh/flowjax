import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
from flowjax.bijections.abc import Bijection
from flowjax.utils import Array

class Conditional(Bijection):
    transformer_constructor: Callable
    autoregressive_mlp: AutoregressiveMLP
    def __init__(
        self,
        key: KeyArray,
        transformer: Bijection,
        dim: int,
        cond_dim: Union[None, int],
        nn_width: int,
        nn_depth: int,
        nn_activation: Callable = jnn.relu,
    ) -> None:
        """Masked autoregressive bijection implementation (https://arxiv.org/abs/1705.07057v4).
        The transformer is parameterised by a neural network, with weights masked to ensure
        an autoregressive structure.

        Args:
            key (KeyArray): Jax PRNGKey
            transformer (Bijection): Bijection with shape () to be parameterised by the autoregressive network.
            dim (int): Dimension.
            cond_dim (Union[None, int]): Dimension of any conditioning variables.
            nn_width (int): Neural network width.
            nn_depth (int): Neural network depth.
            nn_activation (Callable, optional): Neural network activation. Defaults to jnn.relu.
        """
        if transformer.shape != () or transformer.cond_shape is not None:
            raise ValueError(
                "Currently, only unconditional transformers with shape () are supported."
            )

        constructor, transformer_init_params = get_ravelled_bijection_constructor(
            transformer
        )

        autoregressive_mlp = eqx.nn.MLP(
            in_size=cond_dim,
            out_size=out_size,
            width_size=nn_width, 
            depth=nn_depth,
            activation=jax.nn.relu,
            final_activation=lambda x: x,
            key=key,
        )

        self.transformer_constructor = constructor
        self.shape = (dim,)
        self.cond_shape = None if cond_dim is None else (cond_dim,)

    def transform(self, x, condition=None):
        nn_input = x if condition is None else jnp.hstack((x, condition))
        transformer_params = self.autoregressive_mlp(nn_input)
        transformer = self._flat_params_to_transformer(transformer_params)
        return transformer.transform(x)

    def transform_and_log_abs_det_jacobian(self, x, condition=None):
        nn_input = x if condition is None else jnp.hstack((x, condition))
        transformer_params = self.autoregressive_mlp(nn_input)
        transformer = self._flat_params_to_transformer(transformer_params)
        return transformer.transform_and_log_abs_det_jacobian(x)

    def inverse(self, y, condition=None):
        init = (y, 0)
        fn = partial(self.inv_scan_fn, condition=condition)
        (x, _), _ = jax.lax.scan(fn, init, None, length=len(y))
        return x

    def inv_scan_fn(self, init, _, condition):
        "One 'step' in computing the inverse"
        y, rank = init
        nn_input = y if condition is None else jnp.hstack((y, condition))
        transformer_params = self.autoregressive_mlp(nn_input)
        transformer = self._flat_params_to_transformer(transformer_params)
        x = transformer.inverse(y)
        x = y.at[rank].set(x[rank])
        return (x, rank + 1), None

    def inverse_and_log_abs_det_jacobian(self, y, condition=None):
        x = self.inverse(y, condition)
        log_det = self.transform_and_log_abs_det_jacobian(x, condition)[1]
        return x, -log_det

    def _flat_params_to_transformer(
        self, params: Array
    ):  # TODO code repetition with MAF
        "Reshape to dim X params_per_dim, then vmap."
        dim = self.shape[-1]
        transformer_params = jnp.reshape(params, (dim, -1))
        transformer = eqx.filter_vmap(self.transformer_constructor)(transformer_params)
        return Vmap(transformer, (dim,))
