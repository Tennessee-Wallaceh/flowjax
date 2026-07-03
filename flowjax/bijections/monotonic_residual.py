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

@runtime_checkable
class _Activation(Protocol):
    def __call__(self, x: Array) -> Array: ...
    


class _BatchedSplitMonotonicLayer(eqx.Module):
    weight_raw: Array
    bias_pos: Array
    bias_neg: Array
    out_scale: BoundedPositiveParameter
    base_activation: _Activation

    dim: int = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        dim: int,
        in_channels: int,
        out_channels: int,
        target_slope: float,
        w_scale: float,
        out_scale_max: float,
        base_activation: _Activation,
        centre_init_range: tuple[float, float] | None = None,
    ):
        if dim < 1:
            raise ValueError(f"dim must be >= 1. Got {dim}.")
        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1. Got {in_channels}.")
        if out_channels < 1:
            raise ValueError(f"out_channels must be >= 1. Got {out_channels}.")
        if target_slope <= 0:
            raise ValueError(f"target_slope must be > 0. Got {target_slope}.")
        if w_scale <= 0:
            raise ValueError(f"w_scale must be > 0. Got {w_scale}.")
        if out_scale_max <= 0:
            raise ValueError(f"out_scale_max must be > 0. Got {out_scale_max}.")

        k_w, k_centres = jr.split(key, 2)

        self.dim = dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_activation = base_activation

        w = w_scale * jr.normal(k_w, (dim, out_channels, in_channels))
        self.weight_raw = w

        w_pos = jnp.maximum(w, 0.0)
        w_neg = jnp.minimum(w, 0.0)

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

            self.bias_pos = -jnp.einsum("doi,doi->do", w_pos, centres)
            self.bias_neg = -jnp.einsum("doi,doi->do", w_neg, centres)
        else:
            self.bias_pos = jnp.zeros((dim, out_channels))
            self.bias_neg = jnp.zeros((dim, out_channels))

        eps = 1e-8

        # At zero / near a centre, squareplus'(0) = 0.5, so the rough
        # per-output derivative magnitude is:
        #
        #     0.5 * out_scale * sum(abs(w), axis=-1)
        #
        # This initializes each output channel to have approximately
        # `target_slope` local derivative scale.
        sum_abs_w = jnp.sum(jnp.abs(w), axis=-1)
        v0 = 2.0 * target_slope / (sum_abs_w + eps)
        v0 = jnp.clip(v0, min=eps, max=0.95 * out_scale_max)

        self.out_scale = BoundedPositiveParameter(
            v0,
            min_value=1e-6,
            max_value=out_scale_max,
        )

    @staticmethod
    def _squareplus_and_prime(x: Array) -> tuple[Array, Array]:
        root = jnp.sqrt(x * x + 4.0)
        act = 0.5 * (x + root)
        act_prime = 0.5 * (1.0 + x / root)
        return act, act_prime

    def forward_and_jacobian(
        self,
        z: Array,
        jac_in: Array,
    ) -> tuple[Array, Array]:
        """Forward pass and scalar-input Jacobian propagation.

        Args:
            z: shape (dim, in_channels)
            jac_in: shape (dim, in_channels), where jac_in[d, c] is
                d z[d, c] / d x[d].

        Returns:
            z_next: shape (dim, out_channels)
            jac_out: shape (dim, out_channels), where jac_out[d, o] is
                d z_next[d, o] / d x[d].
        """
        w = self.weight_raw
        v = self.out_scale.value

        w_pos = jnp.maximum(w, 0.0)
        w_neg = jnp.minimum(w, 0.0)

        pre_pos = jnp.einsum("doi,di->do", w_pos, z) + self.bias_pos
        pre_neg = jnp.einsum("doi,di->do", w_neg, z) + self.bias_neg

        act_pos, act_prime_pos = self._squareplus_and_prime(pre_pos)
        act_neg, act_prime_neg = self._squareplus_and_prime(pre_neg)

        z_next = v * (act_pos - act_neg)

        jac_pos = jnp.einsum("doi,di->do", w_pos, jac_in)
        jac_neg = jnp.einsum("doi,di->do", -w_neg, jac_in)

        jac_out = v * (
            act_prime_pos * jac_pos
            + act_prime_neg * jac_neg
        )

        return z_next, jac_out


class DeepMarginalMonotonicResidual(AbstractBijection):
    r"""Elementwise deep monotone residual bijection.

    For each dimension d:

        T_d(x_d) = sigma_d * x_d + h_d(x_d)

    where h_d is an independent deep monotone network shared in shape but
    not parameters across dimensions.

    Example with num_hidden_layers=2 and width=5:

        1 -> 5 -> 5 -> 1
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None

    sigma: PositiveParameter
    hidden_layers: tuple[_BatchedSplitMonotonicLayer, ...]
    head: _BatchedSplitMonotonicLayer
    intercept: Array
    inverse_lower: Array
    inverse_upper: Array
    base_activation: _Activation

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        dim: int,
        width: int = 5,
        num_hidden_layers: int = 2,
        rho: float = 0.05,
        w_scale: float = 0.1,
        hidden_out_scale_max: float = 1.0,
        head_out_scale_max: float = 2.0,
        inverse_bracket: tuple[float, float] = (-20.0, 20.0),
        per_dim_sigma: bool = True,
        input_centre_init_range: tuple[float, float] | None = (-5.0, 5.0),
        hidden_centre_init_range: tuple[float, float] | None = (-2.0, 2.0),
        head_centre_init_range: tuple[float, float] | None = None,
    ):
        if dim < 1:
            raise ValueError(f"dim must be >= 1. Got {dim}.")
        if width < 1:
            raise ValueError(f"width must be >= 1. Got {width}.")
        if num_hidden_layers < 1:
            raise ValueError(
                f"num_hidden_layers must be >= 1. Got {num_hidden_layers}."
            )
        if not (0.0 < rho < 1.0):
            raise ValueError(f"rho must be in (0, 1). Got {rho}.")
        if w_scale <= 0:
            raise ValueError(f"w_scale must be > 0. Got {w_scale}.")
        if hidden_out_scale_max <= 0:
            raise ValueError(
                f"hidden_out_scale_max must be > 0. Got {hidden_out_scale_max}."
            )
        if head_out_scale_max <= 0:
            raise ValueError(
                f"head_out_scale_max must be > 0. Got {head_out_scale_max}."
            )
        if inverse_bracket[0] >= inverse_bracket[1]:
            raise ValueError(
                f"Expected inverse_bracket lower < upper. Got {inverse_bracket}."
            )

        self.shape = (dim,)
        self.base_activation = jax.nn.squareplus

        self.inverse_lower = jnp.asarray(inverse_bracket[0])
        self.inverse_upper = jnp.asarray(inverse_bracket[1])

        sigma0 = jnp.full((dim,), rho) if per_dim_sigma else jnp.asarray(rho)
        self.sigma = PositiveParameter(sigma0, min_value=1e-6)

        keys = jr.split(key, num_hidden_layers + 1)

        # Hidden layers are feature maps, so do not make each hidden layer
        # carry the whole residual slope budget. The head is calibrated later.
        hidden_target_slope = 1.0
        target_residual_slope = 1.0 - rho

        self.hidden_layers = tuple(
            _BatchedSplitMonotonicLayer(
                keys[i],
                dim=dim,
                in_channels=1 if i == 0 else width,
                out_channels=width,
                target_slope=hidden_target_slope,
                w_scale=w_scale,
                out_scale_max=hidden_out_scale_max,
                base_activation=self.base_activation,
                centre_init_range=(
                    input_centre_init_range
                    if i == 0
                    else hidden_centre_init_range
                ),
            )
            for i in range(num_hidden_layers)
        )

        self.head = _BatchedSplitMonotonicLayer(
            keys[-1],
            dim=dim,
            in_channels=width,
            out_channels=1,
            target_slope=target_residual_slope,
            w_scale=w_scale,
            out_scale_max=head_out_scale_max,
            base_activation=self.base_activation,
            centre_init_range=head_centre_init_range,
        )

        self._calibrate_identity_slope()

        h0, _ = self._h_and_grad(jnp.zeros((dim,)))
        self.intercept = -h0

    def _h_and_grad(self, x: Array) -> tuple[Array, Array]:
        """Evaluate h(x) and dh/dx elementwise.

        Args:
            x: shape (dim,)

        Returns:
            h: shape (dim,)
            dhdx: shape (dim,)
        """
        z = x[:, None]
        jac = jnp.ones_like(z)

        for layer in self.hidden_layers:
            z, jac = layer.forward_and_jacobian(z, jac)

        h, dhdx = self.head.forward_and_jacobian(z, jac)

        return h.squeeze(axis=-1), dhdx.squeeze(axis=-1)

    def _calibrate_identity_slope(self):
        """Calibrate head scale so initial derivative is near identity at zero.

        The intended initialization is:

            sigma + dh/dx ≈ 1

        at x = 0.
        """
        x0 = jnp.zeros((self.shape[0],))
        _, dhdx0 = self._h_and_grad(x0)

        sigma = jnp.broadcast_to(self.sigma.value, self.shape)
        target = 1.0 - sigma

        ratio = target / jnp.clip(dhdx0, min=1e-8)

        new_value = self.head.out_scale.value * ratio[:, None]
        new_value = jnp.clip(
            new_value,
            min=self.head.out_scale.min_value,
            max=0.95 * self.head.out_scale.max_value,
        )

        self.head = eqx.tree_at(
            lambda layer: layer.out_scale,
            self.head,
            BoundedPositiveParameter(
                new_value,
                min_value=self.head.out_scale.min_value,
                max_value=self.head.out_scale.max_value,
            ),
        )

    def transform_and_log_det(self, x, condition=None):

        if x.shape != self.shape:
            raise ValueError(f"Expected x.shape={self.shape}, got {x.shape}.")

        h, dhdx = self._h_and_grad(x)
        h = h + self.intercept

        sigma = jnp.broadcast_to(self.sigma.value, self.shape)

        y = sigma * x + h
        dydx = sigma + dhdx

        tiny = jnp.finfo(x.dtype).tiny
        logdet = jnp.sum(jnp.log(jnp.clip(dydx, min=tiny)))

        return y, logdet

    def inverse_and_log_det(self, y, condition=None):
        """Vectorized bisection inverse.

        Since the transform is elementwise, this bisects all dimensions
        simultaneously. This is much better than vmapping scalar bisection
        that calls a full transform per coordinate.
        """

        if y.shape != self.shape:
            raise ValueError(f"Expected y.shape={self.shape}, got {y.shape}.")

        lower = jnp.full_like(y, self.inverse_lower)
        upper = jnp.full_like(y, self.inverse_upper)

        def body(carry, _):
            lower, upper = carry
            mid = 0.5 * (lower + upper)

            f_lower = self.transform(lower)[0] - y
            f_mid = self.transform(mid)[0] - y

            same_sign = jnp.sign(f_lower) == jnp.sign(f_mid)

            lower = jnp.where(same_sign, mid, lower)
            upper = jnp.where(same_sign, upper, mid)

            return (lower, upper), None

        (lower, upper), _ = jax.lax.scan(
            body,
            (lower, upper),
            xs=None,
            length=32,
        )

        x = 0.5 * (lower + upper)
        _, logdet = self.transform_and_log_det(x)

        return x, -logdet

    def derivative_summary(self, x: Array) -> dict[str, Array]:
        """Small diagnostic helper."""

        _, dhdx = self._h_and_grad(x)
        sigma = jnp.broadcast_to(self.sigma.value, self.shape)
        dydx = sigma + dhdx

        log_dydx = jnp.log(jnp.clip(dydx, min=jnp.finfo(x.dtype).tiny))

        return {
            "mean_log_dydx": jnp.mean(log_dydx),
            "std_log_dydx": jnp.std(log_dydx),
            "max_abs_log_dydx": jnp.max(jnp.abs(log_dydx)),
            "min_dydx": jnp.min(dydx),
            "max_dydx": jnp.max(dydx),
        }
    
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