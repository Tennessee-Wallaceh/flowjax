import equinox as eqx
from typing import List
import jax
import jax.numpy as jnp
from flowjax.bijections.abc import Bijection, Transformer

class Monotonic(eqx.Module):
    weight: jax.numpy.ndarray
    bias: jax.numpy.ndarray
    active_s: jax.numpy.ndarray

    def __init__(self, in_size, out_size, key):
        wkey, bkey, skey = jax.random.split(key, 3)
        self.weight = jax.random.normal(wkey, (out_size, in_size))
        self.bias = jax.random.normal(bkey, (out_size,))
        self.active_s = jax.random.normal(skey, (out_size,))

    def activation(self, x, s):
        return s * jax.nn.elu(x) - (1 - s) * jax.nn.elu(-x)

    def __call__(self, x):
        pos_weight = jnp.exp(self.weight)
        active_s = jax.nn.sigmoid(self.active_s)
        return self.activation(pos_weight @ x + self.bias, active_s)

class CompositeMonotonic(eqx.Module):
    layers: List[eqx.Module]

    def __init__(self, num_layers, layer_width, key):
        layer_keys = jax.random.split(key, num_layers)
        self.layers = [Monotonic(1, layer_width, layer_keys[0])]
        for i in range(1, num_layers):
            self.layers.append(Monotonic(layer_width, layer_width, layer_keys[i]))
        self.layers.append(Monotonic(layer_width, 1, layer_keys[i]))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0]


class MonoNN(Bijection):
    """
    U -> X Nothing
    X -> U Mono NN
    """
    transformer: Transformer
    cond_dim: int
    nn: eqx.Module
    nn_value_and_grad: eqx.Module
    def __init__(self, key, depth=3, width=5):
        self.nn = CompositeMonotonic(depth, width, key)
        self.nn_value_and_grad = jax.value_and_grad(self.nn)
        self.cond_dim = 0
        self.transformer = None

    def transform(self, u):
        raise NotImplementedError

    def transform_and_log_abs_det_jacobian(self, u):
        raise NotImplementedError

    def inverse(self, x):
        return self.nn(x)

    def inverse_and_log_abs_det_jacobian(self, x, params):
        out, grad = self.nn_value_and_grad(x)
        return out.reshape(-1), jnp.log(grad)

    def num_params(self, dim):
        return 0

    def get_ranks(self, dim):
        return jnp.tile(jnp.arange(dim), 2)

    def get_args(self, params):
        return tuple()