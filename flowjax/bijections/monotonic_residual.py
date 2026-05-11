"""Residual scalar bijections using paper-style sign-adaptive split units."""

from typing import ClassVar, Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray

from flowjax.bijections.bijection import AbstractBijection
from flowjax.parameters import PositiveParameter
from flowjax.root_finding import bisection_search


@runtime_checkable
class _Activation(Protocol):
    def __call__(self, x: Array) -> Array: ...


@runtime_checkable
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


class TriangularDeepMonotonicResidual(AbstractBijection):
    """Multivariate triangular deep monotone residual flow ``T(x)=sigma*x+h(x)``."""

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None

    sigma: PositiveParameter
    hidden_layers: tuple["_MaskedMonotoneLayer", ...]
    head: "_MaskedMonotoneLayer"
    intercept: Array
    base_activation: _Activation
    sigma_is_scalar: bool
    inverse_lower: Array
    inverse_upper: Array

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        dim: int,
        channels: int = 16,
        num_hidden_layers: int = 2,
        rho: float = 0.95,
        w_scale_self: float = 0.1,
        w_scale_past: float = 1e-3,
        per_dim_sigma: bool = True,
        inverse_bracket: tuple[float, float] = (-5.0, 5.0),
    ):
        if dim < 1:
            raise ValueError(f"dim must be >= 1. Got {dim}.")
        if channels < 1:
            raise ValueError(f"channels must be >= 1. Got {channels}.")
        if num_hidden_layers < 1:
            raise ValueError(f"num_hidden_layers must be >= 1. Got {num_hidden_layers}.")
        if not (0.0 < rho < 1.0):
            raise ValueError(f"rho must be in (0, 1). Got {rho}.")
        if inverse_bracket[0] >= inverse_bracket[1]:
            raise ValueError(
                f"inverse_bracket must satisfy lower < upper. Got {inverse_bracket}."
            )
        self.shape = (dim,)
        self.base_activation = jax.nn.squareplus
        self.sigma_is_scalar = not per_dim_sigma
        sigma0 = jnp.asarray(rho) if self.sigma_is_scalar else jnp.full((dim,), rho)
        self.sigma = PositiveParameter(sigma0, min_value=1e-6)
        self.inverse_lower = jnp.asarray(inverse_bracket[0])
        self.inverse_upper = jnp.asarray(inverse_bracket[1])

        keys = jr.split(key, num_hidden_layers + 1)
        self.hidden_layers = tuple(
            _MaskedMonotoneLayer(
                keys[i],
                dim=dim,
                in_channels=1 if i == 0 else channels,
                out_channels=channels,
                base_activation=self.base_activation,
                w_past_scale=w_scale_past,
                w_self_scale=w_scale_self,
                out_scale_init=1.0,
            )
            for i in range(num_hidden_layers)
        )
        self.head = _MaskedMonotoneLayer(
            keys[-1],
            dim=dim,
            in_channels=channels,
            out_channels=1,
            base_activation=self.base_activation,
            w_past_scale=w_scale_past,
            w_self_scale=w_scale_self,
            out_scale_init=1e-3,
        )
        self._calibrate_identity_scales()
        self.intercept = -self._h_and_diag_dh(jnp.zeros((dim,)))[0]

    def _h_and_diag_dh(self, x: Array) -> tuple[Array, Array]:
        z = x[:, None]
        diag = jnp.ones_like(z)
        for layer in self.hidden_layers:
            z, diag = layer.forward_and_diag_grad(z, diag)
        h, diag = self.head.forward_and_diag_grad(z, diag)
        return h.squeeze(axis=-1), diag.squeeze(axis=-1)

    def _calibrate_identity_scales(self):
        _, diag_dh0 = self._h_and_diag_dh(jnp.zeros((self.shape[0],)))
        target = 1.0 - self.sigma.value
        ratio = target / jnp.clip(diag_dh0, min=1e-6)
        new_value = jnp.clip(self.head.out_scale.value * ratio[:, None], min=1e-6)
        self.head = eqx.tree_at(
            lambda l: l.out_scale,
            self.head,
            PositiveParameter(new_value),
        )

    def transform_and_log_det(self, x, condition=None):
        h, diag_dh = self._h_and_diag_dh(x)
        h = h + self.intercept
        sigma = self.sigma.value
        y = sigma * x + h
        diag = sigma + diag_dh
        tiny = jnp.finfo(diag.dtype).tiny
        return y, jnp.sum(jnp.log(jnp.clip(diag, min=tiny)))

    def inverse_and_log_det(self, y, condition=None):
        x = jnp.zeros_like(y)
        for i in range(self.shape[0]):
            def scalar_eq(xi):
                x_trial = x.at[i].set(xi)
                return self.transform(x_trial)[i] - y[i]

            xi, _ = bisection_search(
                scalar_eq,
                lower=self.inverse_lower,
                upper=self.inverse_upper,
            )
            x = x.at[i].set(xi)
        _, logdet = self.transform_and_log_det(x)
        return x, -logdet


@runtime_checkable
class _MaskedLayerProtocol(Protocol):
    def forward_and_diag_grad(self, z: Array, diag_in: Array) -> tuple[Array, Array]: ...


class _MaskedMonotoneLayer(eqx.Module):
    w_past: Array
    w_self_raw: Array
    bias: Array
    out_scale: PositiveParameter
    strict_lower_mask: Array
    base_activation: _Activation

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        dim: int,
        in_channels: int,
        out_channels: int,
        base_activation: _Activation,
        w_past_scale: float,
        w_self_scale: float,
        out_scale_init: float,
    ):
        k1, k2 = jr.split(key, 2)
        self.w_past = w_past_scale * jr.normal(k1, (dim, dim, in_channels, out_channels))
        self.w_self_raw = w_self_scale * jr.normal(k2, (dim, in_channels, out_channels))
        self.bias = jnp.zeros((dim, out_channels))
        self.out_scale = PositiveParameter(jnp.full((dim, out_channels), out_scale_init))
        self.strict_lower_mask = jnp.tril(jnp.ones((dim, dim)), k=-1)
        self.base_activation = base_activation

    def forward_and_diag_grad(self, z: Array, diag_in: Array) -> tuple[Array, Array]:
        w_past = self.w_past * self.strict_lower_mask[:, :, None, None]
        past_term = jnp.einsum("djco,jc->do", w_past, z)
        w_self_pos = jnp.maximum(self.w_self_raw, 0.0)
        w_self_neg = jnp.minimum(self.w_self_raw, 0.0)
        self_pos = jnp.einsum("dco,dc->do", w_self_pos, z)
        self_neg = jnp.einsum("dco,dc->do", w_self_neg, z)
        pre_pos = past_term + self_pos + self.bias
        pre_neg = past_term + self_neg + self.bias
        act_pos = self.base_activation(pre_pos)
        act_neg = self.base_activation(pre_neg)
        z_next = self.out_scale.value * (act_pos - act_neg)

        act_prime_pos = 0.5 * (1 + pre_pos / jnp.sqrt(pre_pos**2 + 4.0))
        act_prime_neg = 0.5 * (1 + pre_neg / jnp.sqrt(pre_neg**2 + 4.0))
        self_pos_diag = jnp.einsum("dco,dc->do", w_self_pos, diag_in)
        self_neg_diag = jnp.einsum("dco,dc->do", -w_self_neg, diag_in)
        diag_out = self.out_scale.value * (
            act_prime_pos * self_pos_diag + act_prime_neg * self_neg_diag
        )
        return z_next, diag_out
