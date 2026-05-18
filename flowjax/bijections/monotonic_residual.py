"""Residual scalar bijections using paper-style sign-adaptive split units."""

from typing import ClassVar, Protocol, Literal, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray

from flowjax.bijections.bijection import AbstractBijection
from flowjax.parameters import PositiveParameter
from flowjax.root_finding import bisection_search
import math
import numpy as np
from scipy.stats import qmc


class BoundedPositiveParameter(eqx.Module):
    raw: Array
    min_value: float = eqx.field(static=True)
    max_value: float = eqx.field(static=True)

    def __init__(
        self,
        value: Array,
        *,
        min_value: float = 1e-6,
        max_value: float = 1.0,
    ):
        if not (0.0 <= min_value < max_value):
            raise ValueError(
                f"Expected 0 <= min_value < max_value. Got "
                f"min_value={min_value}, max_value={max_value}."
            )

        value = jnp.asarray(value)
        p = (value - min_value) / (max_value - min_value)
        p = jnp.clip(p, 1e-6, 1.0 - 1e-6)

        self.raw = jax.scipy.special.logit(p)
        self.min_value = min_value
        self.max_value = max_value

    @property
    def value(self) -> Array:
        p = jax.nn.sigmoid(self.raw)
        return self.min_value + (self.max_value - self.min_value) * p

def _sobol_centres(
    key: PRNGKeyArray,
    *,
    n: int,
    centre_dim: int,
    lower: float,
    upper: float,
) -> Array:

    seed = int(jr.randint(key, (), 0, np.iinfo(np.int32).max))
    sampler = qmc.Sobol(d=centre_dim, scramble=True, seed=seed)

    m = int(math.ceil(math.log2(n)))
    u = sampler.random_base2(m=m)[:n]

    centres = lower + (upper - lower) * u
    return jnp.asarray(centres)

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

def _make_scale_gate_schedule(
    *,
    num_hidden_layers: int,
    max_log_gate: float | None,
    scale_schedule: Literal["conservative", "balanced", "free"],
    hidden_out_scale_init: float | None,
    hidden_out_scale_max: float | None,
    head_out_scale_init: float | None,
    head_out_scale_max: float | None,
) -> dict[str, float]:
    n_monotone_layers = num_hidden_layers + 1

    if scale_schedule == "conservative":
        block_branch_budget = 1.5
        default_gate_max = 1.10
        hidden_cap = 0.75
        head_cap = 1.25
        default_hidden_init = 0.35
        default_head_init = 1e-3

    elif scale_schedule == "balanced":
        block_branch_budget = 2.0
        default_gate_max = 1.25
        hidden_cap = 1.0
        head_cap = 2.0
        default_hidden_init = 0.5
        default_head_init = 1e-3

    elif scale_schedule == "free":
        block_branch_budget = 4.0
        default_gate_max = 2.0
        hidden_cap = 2.0
        head_cap = 4.0
        default_hidden_init = 1.0
        default_head_init = 1e-3

    else:
        raise ValueError(
            "scale_schedule must be one of "
            "{'conservative', 'balanced', 'free'}. "
            f"Got {scale_schedule}."
        )

    # Crude per-layer derivative multiplier budget.
    per_layer_budget = block_branch_budget ** (1.0 / n_monotone_layers)

    if max_log_gate is None:
        gate_max = min(default_gate_max, per_layer_budget)
        max_log_gate = math.log(gate_max)
    else:
        if max_log_gate < 0.0:
            raise ValueError(f"max_log_gate must be non-negative. Got {max_log_gate}.")
        gate_max = math.exp(max_log_gate)

    # Leave room for the gate. If the user asks for a large gate, this forces
    # the scale max down rather than allowing both to compound freely.
    raw_scale_max = per_layer_budget / gate_max
    raw_scale_max = max(raw_scale_max, 1e-3)

    if hidden_out_scale_max is None:
        hidden_out_scale_max = min(hidden_cap, raw_scale_max)

    if head_out_scale_max is None:
        head_out_scale_max = min(head_cap, raw_scale_max)

    hidden_out_scale_max = max(float(hidden_out_scale_max), 1e-3)
    head_out_scale_max = max(float(head_out_scale_max), 1e-3)

    if hidden_out_scale_init is None:
        hidden_out_scale_init = min(default_hidden_init, 0.5 * hidden_out_scale_max)

    if head_out_scale_init is None:
        head_out_scale_init = min(default_head_init, 0.5 * head_out_scale_max)

    hidden_out_scale_init = min(float(hidden_out_scale_init), 0.95 * hidden_out_scale_max)
    head_out_scale_init = min(float(head_out_scale_init), 0.95 * head_out_scale_max)

    return {
        "max_log_gate": float(max_log_gate),
        "hidden_out_scale_init": float(hidden_out_scale_init),
        "hidden_out_scale_max": float(hidden_out_scale_max),
        "head_out_scale_init": float(head_out_scale_init),
        "head_out_scale_max": float(head_out_scale_max),
    }


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
        w_scale_context: float = 1e-3,
        context_channels: int | None = None,
        max_log_gate: float | None = None,
        per_dim_sigma: bool = True,
        centre_init_range: tuple[float, float] | None = (-5.0, 5.0),
        inverse_bracket: tuple[float, float] = (-5.0, 5.0),
        context_rank: int = 5,
        scale_schedule: Literal["conservative", "balanced", "free"] = "balanced",
        hidden_out_scale_init: float | None = None,
        hidden_out_scale_max: float | None = None,
        head_out_scale_init: float | None = None,
        head_out_scale_max: float | None = None,
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

        schedule = _make_scale_gate_schedule(
            num_hidden_layers=num_hidden_layers,
            max_log_gate=max_log_gate,
            scale_schedule=scale_schedule,
            hidden_out_scale_init=hidden_out_scale_init,
            hidden_out_scale_max=hidden_out_scale_max,
            head_out_scale_init=head_out_scale_init,
            head_out_scale_max=head_out_scale_max,
        )

        max_log_gate = schedule["max_log_gate"]
        hidden_out_scale_init = schedule["hidden_out_scale_init"]
        hidden_out_scale_max = schedule["hidden_out_scale_max"]
        head_out_scale_init = schedule["head_out_scale_init"]
        head_out_scale_max = schedule["head_out_scale_max"]

        if context_channels is None:
            context_channels = channels

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
                context_channels=context_channels,
                base_activation=self.base_activation,
                context_activation=jax.nn.swish,
                w_context_scale=w_scale_context,
                w_self_scale=w_scale_self,
                out_scale_init=hidden_out_scale_init,
                out_scale_max=hidden_out_scale_max,
                max_log_gate=max_log_gate,
                use_context_loc=True,
                use_context_gate=max_log_gate > 0.0,
                centre_init_range=centre_init_range,
                context_rank=context_rank,
            )
            for i in range(num_hidden_layers)
        )

        self.head = _MaskedMonotoneLayer(
            keys[-1],
            dim=dim,
            in_channels=channels,
            out_channels=1,
            context_channels=context_channels,
            base_activation=self.base_activation,
            context_activation=jax.nn.swish,
            w_context_scale=w_scale_context,
            w_self_scale=w_scale_self,
            out_scale_init=head_out_scale_init,
            out_scale_max=head_out_scale_max,
            max_log_gate=max_log_gate,
            use_context_loc=True,
            use_context_gate=max_log_gate > 0.0,
            centre_init_range=centre_init_range,
            context_rank=context_rank,
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

        new_value = self.head.out_scale.value * ratio[:, None]
        new_value = jnp.clip(
            new_value,
            min=self.head.out_scale.min_value,
            max=0.95 * self.head.out_scale.max_value,
        )

        self.head = eqx.tree_at(
            lambda l: l.out_scale,
            self.head,
            BoundedPositiveParameter(
                new_value,
                min_value=self.head.out_scale.min_value,
                max_value=self.head.out_scale.max_value,
            ),
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
    

class _MaskedMonotoneLayer(eqx.Module):
    # Strictly-past low-rank prefix context path.
    w_context_project: Array
    w_context_mix: Array
    context_bias: Array

    # Context projections.
    w_context_pos: Array
    w_context_neg: Array
    w_context_gate: Array
    w_context_loc: Array

    # Self-coordinate monotone path.
    w_self_raw: Array

    bias_pos: Array
    bias_neg: Array
    out_scale: BoundedPositiveParameter

    base_activation: _Activation
    context_activation: _Activation

    dim: int
    context_rank: int = eqx.field(static=True)
    max_log_gate: float = eqx.field(static=True)
    use_context_loc: bool = eqx.field(static=True)
    use_context_gate: bool = eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        dim: int,
        in_channels: int,
        out_channels: int,
        context_channels: int,
        base_activation: _Activation,
        context_activation: _Activation,
        w_context_scale: float,
        w_self_scale: float,
        out_scale_init: float,
        out_scale_max: float,
        max_log_gate: float,
        use_context_loc: bool = True,
        use_context_gate: bool = True,
        centre_init_range: tuple[float, float] | None = None,
        context_rank: int | None = None,
    ):
        (
            k_context_project,
            k_context_mix,
            k_context_pos,
            k_context_neg,
            k_context_gate,
            k_context_loc,
            k_self,
            k_centres,
        ) = jr.split(key, 8)

        self.dim = dim
        self.context_rank = context_channels if context_rank is None else context_rank
        self.max_log_gate = max_log_gate
        self.use_context_loc = use_context_loc
        self.use_context_gate = use_context_gate

        self.w_context_project = (
            w_context_scale
            / jnp.sqrt(jnp.asarray(in_channels, dtype=jnp.float32))
            * jr.normal(k_context_project, (dim, in_channels, self.context_rank))
        )

        self.w_context_mix = (
            w_context_scale
            / jnp.sqrt(jnp.asarray(self.context_rank, dtype=jnp.float32))
            * jr.normal(k_context_mix, (dim, self.context_rank, context_channels))
        )

        self.context_bias = jnp.zeros((dim, context_channels))

        proj_scale = w_context_scale / jnp.sqrt(
            jnp.asarray(context_channels, dtype=jnp.float32)
        )

        self.w_context_pos = proj_scale * jr.normal(
            k_context_pos, (dim, context_channels, out_channels)
        )
        self.w_context_neg = proj_scale * jr.normal(
            k_context_neg, (dim, context_channels, out_channels)
        )

        self.w_context_gate = 0.1 * proj_scale * jr.normal(
            k_context_gate, (dim, context_channels, out_channels)
        )
        self.w_context_loc = 0.1 * proj_scale * jr.normal(
            k_context_loc, (dim, context_channels, out_channels)
        )

        self.w_self_raw = (
            w_self_scale
            / jnp.sqrt(jnp.asarray(in_channels, dtype=jnp.float32))
            * jr.normal(k_self, (dim, in_channels, out_channels))
        )

        if centre_init_range is not None:
            lower, upper = centre_init_range

            centres = _sobol_centres(
                k_centres,
                n=dim * out_channels,
                centre_dim=in_channels,
                lower=lower,
                upper=upper,
            )

            centres = centres.reshape(dim, out_channels, in_channels)
            centres = jnp.swapaxes(centres, 1, 2)

            w_self_pos = jnp.maximum(self.w_self_raw, 0.0)
            w_self_neg = jnp.minimum(self.w_self_raw, 0.0)

            self.bias_pos = -jnp.einsum("dco,dco->do", w_self_pos, centres)
            self.bias_neg = -jnp.einsum("dco,dco->do", w_self_neg, centres)
        else:
            self.bias_pos = jnp.zeros((dim, out_channels))
            self.bias_neg = jnp.zeros((dim, out_channels))

        self.out_scale = BoundedPositiveParameter(
            jnp.full((dim, out_channels), out_scale_init),
            min_value=1e-6,
            max_value=out_scale_max,
        )

        self.base_activation = base_activation
        self.context_activation = context_activation
    
    def forward_and_diag_grad(self, z: Array, diag_in: Array) -> tuple[Array, Array]:
        # Strictly-past low-rank context: C_d(z_<d).
        projected = jnp.einsum("dcr,dc->dr", self.w_context_project, z)

        prefix = jnp.cumsum(projected, axis=0)
        prefix = jnp.concatenate(
            [jnp.zeros_like(prefix[:1]), prefix[:-1]],
            axis=0,
        )

        context_pre = (
            jnp.einsum("drk,dr->dk", self.w_context_mix, prefix)
            + self.context_bias
        )
        context = self.context_activation(context_pre)

        # Context gives branch-specific shifts.
        context_pos = jnp.einsum("dko,dk->do", self.w_context_pos, context)
        context_neg = jnp.einsum("dko,dk->do", self.w_context_neg, context)

        # Optional conditional location: mu_d(z_<d).
        context_loc = jnp.einsum("dko,dk->do", self.w_context_loc, context)
        if not self.use_context_loc:
            context_loc = jnp.zeros_like(context_loc)

        # Optional positive conditional gate: sigma_d(z_<d).
        raw_gate = jnp.einsum("dko,dk->do", self.w_context_gate, context)
        log_gate = self.max_log_gate * jnp.tanh(raw_gate)
        gate = jnp.exp(log_gate)
        if not self.use_context_gate:
            gate = jnp.ones_like(gate)

        # Self-coordinate monotone path.
        w_self_pos = jnp.maximum(self.w_self_raw, 0.0)
        w_self_neg = jnp.minimum(self.w_self_raw, 0.0)

        self_pos = jnp.einsum("dco,dc->do", w_self_pos, z)
        self_neg = jnp.einsum("dco,dc->do", w_self_neg, z)

        pre_pos = self_pos + context_pos + self.bias_pos
        pre_neg = self_neg + context_neg + self.bias_neg

        act_pos = self.base_activation(pre_pos)
        act_neg = self.base_activation(pre_neg)

        unit = act_pos - act_neg
        z_next = context_loc + self.out_scale.value * gate * unit

        act_prime_pos = 0.5 * (1 + pre_pos / jnp.sqrt(pre_pos**2 + 4.0))
        act_prime_neg = 0.5 * (1 + pre_neg / jnp.sqrt(pre_neg**2 + 4.0))

        self_pos_diag = jnp.einsum("dco,dc->do", w_self_pos, diag_in)
        self_neg_diag = jnp.einsum("dco,dc->do", -w_self_neg, diag_in)

        diag_out = self.out_scale.value * gate * (
            act_prime_pos * self_pos_diag
            + act_prime_neg * self_neg_diag
        )

        return z_next, diag_out