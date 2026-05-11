import jax
import jax.numpy as jnp
import jax.random as jr

from flowjax.bijections import TriangularDeepMonotonicResidual


def test_triangular_jacobian_and_positive_diag():
    bij = TriangularDeepMonotonicResidual(jr.key(0), dim=4, channels=8)
    x = jr.normal(jr.key(1), (4,))
    jac = jax.jacfwd(bij.transform)(x)
    assert jnp.allclose(jnp.triu(jac, k=1), 0.0, atol=1e-6)
    assert jnp.all(jnp.diag(jac) > 0)


def test_logdet_matches_full_jacobian():
    bij = TriangularDeepMonotonicResidual(jr.key(2), dim=4, channels=8)
    x = jr.normal(jr.key(3), (4,))
    _, logdet = bij.transform_and_log_det(x)
    jac = jax.jacfwd(bij.transform)(x)
    expected = jnp.linalg.slogdet(jac)[1]
    assert jnp.allclose(logdet, expected, atol=1e-6)


def test_inverse_consistency():
    bij = TriangularDeepMonotonicResidual(jr.key(4), dim=3, channels=8)
    x = jr.normal(jr.key(5), (3,))
    y = bij.transform(x)
    x_inv = bij.inverse(y)
    assert jnp.allclose(x, x_inv, atol=1e-4)
    y_inv = bij.transform(x_inv)
    assert jnp.allclose(y, y_inv, atol=1e-4)


def test_propagated_diag_matches_autodiff_diag():
    bij = TriangularDeepMonotonicResidual(jr.key(8), dim=4, channels=6)
    x = jr.normal(jr.key(9), (4,))
    _, diag_dh = bij._h_and_diag_dh(x)
    jac_h = jax.jacfwd(lambda u: bij._h_and_diag_dh(u)[0])(x)
    assert jnp.allclose(diag_dh, jnp.diag(jac_h), atol=1e-5, rtol=1e-5)


def test_near_identity_initialization():
    bij = TriangularDeepMonotonicResidual(jr.key(6), dim=5, channels=8)
    x = jr.normal(jr.key(7), (5,))
    y = bij.transform(x)
    assert jnp.max(jnp.abs(y - x)) < 0.15
    jac = jax.jacfwd(bij.transform)(x)
    assert jnp.linalg.norm(jac - jnp.eye(5)) < 0.5
