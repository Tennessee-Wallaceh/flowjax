"""Residual scalar bijections using paper-style sign-adaptive split units."""

from typing import ClassVar, Protocol

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray

from flowjax.bijections.bijection import AbstractBijection
from flowjax.parameters import PositiveParameter
from flowjax.root_finding import bisection_search


class _Activation(Protocol):
    def __call__(self, x: Array) -> Array: ...


class _MonotonicLayer(Protocol):
    weight_raw: Array
    bias: Array
    out_scale: PositiveParameter

    def forward_and_jacobian(self, x: Array) -> tuple[Array, Array]: ...


class _SplitMonotonicLayer:
    weight_raw: Array
    bias: Array
    out_scale: PositiveParameter
    base_activation: _Activation

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        in_dim: int,
        out_dim: int,
        target_slope: float,
        w_scale: float,
        base_activation: _Activation,
    ):
        if in_dim < 1 or out_dim < 1:
            raise ValueError(f"in_dim and out_dim must be >=1. Got in_dim={in_dim}, out_dim={out_dim}.")
        if target_slope <= 0:
            raise ValueError(f"target_slope must be > 0. Got {target_slope}.")
        if w_scale <= 0:
            raise ValueError(f"w_scale must be > 0. Got {w_scale}.")

        self.base_activation = base_activation
        w = w_scale * jr.normal(key, (out_dim, in_dim))
        self.weight_raw = w
        self.bias = jnp.zeros((out_dim,))

        mean_abs_w = jnp.mean(jnp.abs(w), axis=1)
        eps = 1e-8
        v0 = target_slope / (mean_abs_w + eps)
        if not bool(jnp.all(jnp.isfinite(v0))):
            raise ValueError("Computed invalid out-scale initialization.")
        self.out_scale = PositiveParameter(jnp.clip(v0, a_min=eps))

    def forward_and_jacobian(self, x: Array) -> tuple[Array, Array]:
        w = self.weight_raw
        b = self.bias
        v = self.out_scale.value

        w_pos = jnp.maximum(w, 0.0)
        w_neg = jnp.minimum(w, 0.0)
        pre_pos = w_pos @ x + b
        pre_neg = w_neg @ x + b

        act_pos = self.base_activation(pre_pos)
        act_neg = self.base_activation(pre_neg)
        y = v * (act_pos - act_neg)

        act_prime_pos = 0.5 * (1 + pre_pos / jnp.sqrt(pre_pos**2 + 4.0))
        act_prime_neg = 0.5 * (1 + pre_neg / jnp.sqrt(pre_neg**2 + 4.0))
        jac = (v[:, None] * (act_prime_pos[:, None] * w_pos - act_prime_neg[:, None] * w_neg))
        return y, jac


class MonotonicResidual(AbstractBijection):
    r"""Scalar residual bijection :math:`T(x)=\sigma x + g(x)`.

    Uses the split construction over input-side weights
    :math:`g(x)=\sum_i v_i [\phi(w_i^+x+b_i)-\phi(w_i^-x+b_i)] + c`.
    """

    shape: ClassVar[tuple[int, ...]] = ()
    cond_shape: ClassVar[None] = None

    sigma: PositiveParameter
    out_scale: PositiveParameter
    in_weight_raw: Array
    bias: Array
    intercept: Array
    base_activation: _Activation

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        features: int = 32,
        rho: float = 0.75,
        w_scale: float = 0.1,
    ):
        if features < 1:
            raise ValueError("features must be a positive integer.")
        if not (0.0 < rho < 1.0):
            raise ValueError(f"rho must be in (0, 1). Got {rho}.")
        if w_scale <= 0:
            raise ValueError(f"w_scale must be > 0. Got {w_scale}.")

        self.base_activation = jax.nn.squareplus
        target_residual_slope = 1.0 - rho

        k1 = jr.split(key, 1)[0]
        w = w_scale * jr.normal(k1, (features,))
        b = jnp.zeros((features,))

        mean_abs_w = jnp.mean(jnp.abs(w))
        eps = 1e-8
        v0 = 2.0 * target_residual_slope / (features * mean_abs_w + eps)
        if not bool(jnp.isfinite(v0)):
            raise ValueError(
                "Computed invalid v0 during initialization; check rho, features and w_scale. "
                f"Got v0={v0}, mean_abs_w={mean_abs_w}."
            )
        v = v0 * jnp.ones((features,))

        self.in_weight_raw = w
        self.bias = b

        # Calibrate v to hit residual slope target using the exact local derivative at x=0.
        w_pos = jnp.maximum(w, 0.0)
        w_neg = jnp.minimum(w, 0.0)
        pre_pos0 = b
        pre_neg0 = b
        act_prime0 = 0.5 * (1 + b / jnp.sqrt(b**2 + 4.0))
        dgdx_terms0 = v * (act_prime0 * w_pos - act_prime0 * w_neg)
        dgdx0 = jnp.sum(dgdx_terms0)
        rescale = target_residual_slope / (dgdx0 + eps)
        v = v * rescale

        self.out_scale = PositiveParameter(jnp.clip(v, a_min=eps))

        g0 = jnp.sum(self.out_scale.value * (self.base_activation(pre_pos0) - self.base_activation(pre_neg0)))
        self.intercept = -g0
        self.sigma = PositiveParameter(jnp.asarray(rho), min_value=1e-6)

    def _g_and_log_grad(self, x: Array) -> tuple[Array, Array]:
        w = self.in_weight_raw
        b = self.bias
        v = self.out_scale.value

        w_pos = jnp.maximum(w, 0.0)
        w_neg = jnp.minimum(w, 0.0)
        pre_pos = w_pos * x + b
        pre_neg = w_neg * x + b

        act_pos = self.base_activation(pre_pos)
        act_neg = self.base_activation(pre_neg)
        g = jnp.sum(v * (act_pos - act_neg)) + self.intercept

        act_prime_pos = 0.5 * (1 + pre_pos / jnp.sqrt(pre_pos**2 + 4.0))
        act_prime_neg = 0.5 * (1 + pre_neg / jnp.sqrt(pre_neg**2 + 4.0))
        dgdx_terms = v * (act_prime_pos * w_pos - act_prime_neg * w_neg)

        tiny = jnp.finfo(w.dtype).tiny
        log_dgdx = jax.scipy.special.logsumexp(jnp.log(jnp.clip(dgdx_terms, a_min=tiny)))
        return g, log_dgdx

    def transform_and_log_det(self, x, condition=None):
        g, log_dgdx = self._g_and_log_grad(x)
        log_dydx = jnp.logaddexp(jnp.log(self.sigma.value), log_dgdx)
        y = self.sigma.value * x + g
        return y, log_dydx

    def inverse_and_log_det(self, y, condition=None):
        root, _ = bisection_search(
            lambda x: self.transform(x) - y,
            lower=jnp.asarray(-1.0),
            upper=jnp.asarray(1.0),
        )
        _, log_det = self.transform_and_log_det(root)
        return root, -log_det


class DeepMonotonicResidual(AbstractBijection):
    r"""Deep scalar residual bijection :math:`T(x)=\sigma x + h(x)`, with :math:`h:\mathbb{R}\to\mathbb{R}`."""

    shape: ClassVar[tuple[int, ...]] = ()
    cond_shape: ClassVar[None] = None

    sigma: PositiveParameter
    intercept: Array
    input_layer: _MonotonicLayer
    hidden_layers: tuple[_MonotonicLayer, ...]
    output_layer: _MonotonicLayer

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        width: int = 32,
        num_hidden_layers: int = 1,
        rho: float = 0.75,
        w_scale: float = 0.1,
    ):
        if width < 1:
            raise ValueError(f"width must be >= 1. Got {width}.")
        if num_hidden_layers < 0:
            raise ValueError(f"num_hidden_layers must be >= 0. Got {num_hidden_layers}.")
        if not (0.0 < rho < 1.0):
            raise ValueError(f"rho must be in (0, 1). Got {rho}.")
        if w_scale <= 0:
            raise ValueError(f"w_scale must be > 0. Got {w_scale}.")

        self.sigma = PositiveParameter(jnp.asarray(rho), min_value=1e-6)
        self.intercept = jnp.asarray(0.0)
        self.base_activation = jax.nn.squareplus

        n_layers = num_hidden_layers + 2
        keys = jr.split(key, n_layers)
        target_residual_slope = 1.0 - rho
        per_layer_slope = target_residual_slope ** (1.0 / max(n_layers, 1))

        self.input_layer = _SplitMonotonicLayer(
            keys[0],
            in_dim=1,
            out_dim=width,
            target_slope=per_layer_slope,
            w_scale=w_scale,
            base_activation=self.base_activation,
        )
        self.hidden_layers = tuple(
            _SplitMonotonicLayer(
                keys[i + 1],
                in_dim=width,
                out_dim=width,
                target_slope=per_layer_slope,
                w_scale=w_scale,
                base_activation=self.base_activation,
            )
            for i in range(num_hidden_layers)
        )
        self.output_layer = _SplitMonotonicLayer(
            keys[-1],
            in_dim=width,
            out_dim=1,
            target_slope=per_layer_slope,
            w_scale=w_scale,
            base_activation=self.base_activation,
        )

        g0, _ = self._h_and_dhdx(jnp.asarray(0.0))
        self.intercept = -g0

    def _h_and_dhdx(self, x: Array) -> tuple[Array, Array]:
        z, jac = self.input_layer.forward_and_jacobian(jnp.atleast_1d(x))
        for layer in self.hidden_layers:
            z_new, jac_layer = layer.forward_and_jacobian(z)
            jac = jac_layer @ jac
            z = z_new
        out, jac_out = self.output_layer.forward_and_jacobian(z)
        dhdx = (jac_out @ jac).squeeze()
        return out.squeeze(), dhdx

    def transform_and_log_det(self, x, condition=None):
        h, dhdx = self._h_and_dhdx(x)
        dydx = self.sigma.value + dhdx
        tiny = jnp.finfo(jnp.asarray(dydx).dtype).tiny
        log_dydx = jnp.log(jnp.clip(dydx, a_min=tiny))
        y = self.sigma.value * x + h + self.intercept
        return y, log_dydx

    def inverse_and_log_det(self, y, condition=None):
        root, _ = bisection_search(
            lambda x: self.transform(x) - y,
            lower=jnp.asarray(-1.0),
            upper=jnp.asarray(1.0),
        )
        _, log_det = self.transform_and_log_det(root)
        return root, -log_det
