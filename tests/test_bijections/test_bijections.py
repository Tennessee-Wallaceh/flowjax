"General tests for bijections (including transformers)."
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
import jax.random as jr

from flowjax.bijections import (
    AdditiveLinearCondition,
    Affine,
    BlockAutoregressiveNetwork,
    Chain,
    Coupling,
    EmbedCondition,
    Flip,
    MaskedAutoregressive,
    Partial,
    Permute,
    Scan,
    Tanh,
    TanhLinearTails,
    TriangularAffine,
    RationalQuadraticSpline,
)

from jax.config import config

config.update("jax_enable_x64", True)


dim = 5
cond_dim = 2
key = jr.PRNGKey(0)
pos_def_triangles = jnp.full((dim, dim), 0.5) + jnp.diag(jnp.ones(dim))


def get_maf_layer(key):
    return MaskedAutoregressive(
        key, Affine(), dim, cond_dim=cond_dim, nn_width=5, nn_depth=5
    )


# What's wrong with making it match exactly?
# Specify elementwise, and that each parameter leading axis corresponds to the dimension

bijections = {
    "Flip": Flip(),
    "Permute": Permute(jnp.flip(jnp.arange(dim))),
    "Permute (3D)": Permute(jnp.reshape(jr.permutation(key, jnp.arange(2*3*4)), (2,3,4))),
    "Partial (int)": Partial(Affine(jnp.array(2), jnp.array(2)), 0),
    "Partial (bool array)": Partial(Flip(), jnp.array([True, False] * 2 + [True])),
    "Partial (int array)": Partial(Flip(), jnp.array([0, 4])),
    "Partial (slice)": Partial(Affine(jnp.zeros(3)), slice(0, 3)),
    "Affine": Affine(jnp.ones(dim), jnp.full(dim, 2)),
    "Tanh": Tanh(),
    "TanhLinearTails": TanhLinearTails(1),
    "TriangularAffine (lower)": TriangularAffine(jnp.arange(dim), pos_def_triangles),
    "TriangularAffine (upper)": TriangularAffine(
        jnp.arange(dim), pos_def_triangles, lower=False
    ),
    "TriangularAffine (weight_norm)": TriangularAffine(
        jnp.arange(dim), pos_def_triangles, weight_normalisation=True
    ),
    "RationalQuadraticSpline": RationalQuadraticSpline(knots=4, interval=1, shape=(5,)),
    "Coupling (unconditional)": Coupling(
        key,
        Affine(),
        d=dim // 2,
        D=dim,
        cond_dim=None,
        nn_width=10,
        nn_depth=2,
    ),
    "Coupling (conditional)": Coupling(
        key,
        Affine(),
        d=dim // 2,
        D=dim,
        cond_dim=cond_dim,
        nn_width=10,
        nn_depth=2,
    ),
    "MaskedAutoregressive_Affine (unconditional)": MaskedAutoregressive(
        key, Affine(), cond_dim=0, dim=dim, nn_width=10, nn_depth=2
    ),
    "MaskedAutoregressive_Affine (conditional)": MaskedAutoregressive(
        key, Affine(), cond_dim=cond_dim, dim=dim, nn_width=10, nn_depth=2
    ),
    "MaskedAutoregressive_RationalQuadraticSpline (unconditional)": MaskedAutoregressive(
        key,
        RationalQuadraticSpline(5, 3),
        dim=dim,
        cond_dim=0,
        nn_width=10,
        nn_depth=2,
    ),
    "BlockAutoregressiveNetwork (unconditional)": BlockAutoregressiveNetwork(
        key, dim=dim, cond_dim=0, block_dim=3, depth=1
    ),
    "BlockAutoregressiveNetwork (conditional)": BlockAutoregressiveNetwork(
        key, dim=dim, cond_dim=cond_dim, block_dim=3, depth=1
    ),
    "AdditiveLinearCondition": AdditiveLinearCondition(
        jr.uniform(key, (dim, cond_dim))
    ),
    "EmbedCondition": EmbedCondition(
        BlockAutoregressiveNetwork(key, dim=dim, cond_dim=1, block_dim=3, depth=1),
        eqx.nn.MLP(2, 1, 3, 1, key=key),
        (cond_dim,),  # Raw
    ),
    "Chain": Chain([Flip(), Affine(jnp.ones(dim), jnp.full(dim, 2))]),
    "Scan": Scan(eqx.filter_vmap(get_maf_layer)(jr.split(key, 3))),
}


@pytest.mark.parametrize("bijection", bijections.values(), ids=bijections.keys())
def test_transform_inverse(bijection):
    """Tests transform and inverse methods."""
    shape = bijection.shape if bijection.shape is not None else (dim,)
    x = jr.normal(jr.PRNGKey(0), shape)
    if bijection.cond_shape is not None:
        cond = jr.normal(jr.PRNGKey(0), bijection.cond_shape)
    else:
        cond = None
    y = bijection.transform(x, cond)
    try:
        x_reconstructed = bijection.inverse(y, cond)
        assert x == pytest.approx(x_reconstructed, abs=1e-4)
    except NotImplementedError:
        pass


@pytest.mark.parametrize("bijection", bijections.values(), ids=bijections.keys())
def test_transform_inverse_and_log_dets(bijection):
    """Tests the transform_and_log_abs_det_jacobian and inverse_and_log_abs_det_jacobian methods,
    by 1) checking invertibility and 2) comparing log dets to those obtained with
    automatic differentiation."""
    shape = bijection.shape if bijection.shape is not None else (dim,)
    x = jr.normal(jr.PRNGKey(0), shape)

    if bijection.cond_shape is not None:
        cond = jr.normal(jr.PRNGKey(0), bijection.cond_shape)
    else:
        cond = None

    # We flatten the function so auto_jacobian is calculated correctly
    def flat_transform(x_flat, cond):
        x = x_flat.reshape(shape)
        y = bijection.transform(x, cond)
        return y.ravel()
    
    auto_jacobian = jax.jacobian(flat_transform)(x.ravel(), cond)
    auto_log_det = jnp.log(jnp.abs(jnp.linalg.det(auto_jacobian)))
    y, logdet = bijection.transform_and_log_abs_det_jacobian(x, cond)
    assert logdet == pytest.approx(auto_log_det, abs=1e-4)

    try:
        x_reconstructed, logdetinv = bijection.inverse_and_log_abs_det_jacobian(y, cond)
        assert logdetinv == pytest.approx(-auto_log_det, abs=1e-4)
        assert x == pytest.approx(x_reconstructed, abs=1e-4)

    except NotImplementedError:
        pass
