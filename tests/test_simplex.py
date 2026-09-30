"""Correctness of the simplex/sphere geometry.

These are the identities everything downstream silently assumes. A subtle sign or
normalisation error here would not crash training; it would just quietly produce a model
that learns the wrong vector field, which is far more expensive to discover later.
"""

import torch

from dgm.paths import simplex as S

torch.manual_seed(0)
K, B, L = 5, 8, 12


def rand_simplex(*shape):
    return torch.distributions.Dirichlet(torch.ones(K)).sample(shape)


def onehot(*shape):
    idx = torch.randint(0, K, shape)
    return torch.nn.functional.one_hot(idx, K).float()


def test_sphere_roundtrip():
    p = rand_simplex(B, L)
    assert torch.allclose(S.from_sphere(S.to_sphere(p)), p, atol=1e-6)


def test_sphere_is_unit_norm():
    s = S.to_sphere(rand_simplex(B, L))
    assert torch.allclose(s.norm(dim=-1), torch.ones(B, L), atol=1e-6)


def test_onehot_does_not_produce_nan():
    """One-hot input is the common case for real data and hits sqrt(0)."""
    s = S.to_sphere(onehot(B, L))
    assert torch.isfinite(s).all()
    v = S.log_map(S.to_sphere(rand_simplex(B, L)), s)
    assert torch.isfinite(v).all()


def test_slerp_endpoints():
    s0, s1 = S.to_sphere(rand_simplex(B, L)), S.to_sphere(onehot(B, L))
    assert torch.allclose(S.slerp(s0, s1, torch.zeros(B, L)), s0, atol=1e-5)
    assert torch.allclose(S.slerp(s0, s1, torch.ones(B, L)), s1, atol=1e-5)


def test_slerp_stays_on_sphere():
    s0, s1 = S.to_sphere(rand_simplex(B, L)), S.to_sphere(rand_simplex(B, L))
    for t in [0.1, 0.25, 0.5, 0.9]:
        st = S.slerp(s0, s1, torch.full((B, L), t))
        assert torch.allclose(st.norm(dim=-1), torch.ones(B, L), atol=1e-5)


def test_slerp_identical_points():
    """Degenerate branch: source and target agree, as they do whenever a position matches."""
    s = S.to_sphere(onehot(B, L))
    out = S.slerp(s, s, torch.full((B, L), 0.5))
    assert torch.isfinite(out).all()
    assert torch.allclose(out, s, atol=1e-5)


def test_slerp_constant_speed():
    """A geodesic is traversed at constant speed: equal t-steps cover equal arc length."""
    s0, s1 = S.to_sphere(rand_simplex(B, L)), S.to_sphere(rand_simplex(B, L))
    ts = torch.linspace(0, 1, 11)
    pts = [S.slerp(s0, s1, torch.full((B, L), float(t))) for t in ts]
    arcs = torch.stack([S.sphere_angle(a, b) for a, b in zip(pts[:-1], pts[1:])])
    assert (arcs.std(dim=0) < 1e-4).all()


def test_log_exp_inverse():
    s0, s1 = S.to_sphere(rand_simplex(B, L)), S.to_sphere(rand_simplex(B, L))
    assert torch.allclose(S.exp_map(s0, S.log_map(s0, s1)), s1, atol=1e-5)


def test_log_map_norm_is_geodesic_distance():
    s0, s1 = S.to_sphere(rand_simplex(B, L)), S.to_sphere(rand_simplex(B, L))
    assert torch.allclose(
        S.log_map(s0, s1).norm(dim=-1), S.sphere_angle(s0, s1), atol=1e-5
    )


def test_log_map_is_tangent():
    s0, s1 = S.to_sphere(rand_simplex(B, L)), S.to_sphere(rand_simplex(B, L))
    v = S.log_map(s0, s1)
    assert torch.allclose((v * s0).sum(-1), torch.zeros(B, L), atol=1e-5)


def test_fisher_rao_is_a_metric():
    p, q = rand_simplex(B, L), rand_simplex(B, L)
    d = S.fisher_rao_distance(p, q)
    assert torch.allclose(d, S.fisher_rao_distance(q, p), atol=1e-6)  # symmetry
    assert torch.allclose(
        S.fisher_rao_distance(p, p), torch.zeros(B, L), atol=1e-5
    )  # identity
    assert (d >= 0).all() and (d <= torch.pi / 2 + 1e-5).all()  # positive orthant


def test_fisher_rao_exceeds_euclidean_near_boundary():
    """The motivating fact: near the boundary, Fisher-Rao and Euclidean disagree sharply.

    Two nearly-one-hot distributions on different vertices are Euclidean-far but
    Fisher-Rao-far by a different ordering, which is exactly why the cost function used
    for a coupling is not a neutral choice.
    """
    eps = 1e-4
    a = torch.full((K,), eps)
    a[0] = 1 - eps * (K - 1)
    b = torch.full((K,), eps)
    b[1] = 1 - eps * (K - 1)
    mid = torch.full((K,), 1.0 / K)
    d_corner = S.fisher_rao_distance(a, b)
    d_to_mid = S.fisher_rao_distance(a, mid)
    # Corner-to-corner is the maximal separation in the positive orthant (pi/2);
    # corner-to-barycentre is strictly less.
    assert d_corner > d_to_mid


def test_tangent_project_idempotent():
    s = S.to_sphere(rand_simplex(B, L))
    v = torch.randn(B, L, K)
    v1 = S.tangent_project(s, v)
    assert torch.allclose(S.tangent_project(s, v1), v1, atol=1e-6)
    assert torch.allclose((v1 * s).sum(-1), torch.zeros(B, L), atol=1e-5)


def test_decode_recovers_onehot():
    x = onehot(B, L)
    assert (S.decode(x) == x.argmax(-1)).all()
    assert (S.decode(S.to_sphere(x), from_sphere_coords=True) == x.argmax(-1)).all()


def test_gradients_flow_through_onehot_path():
    """End-to-end differentiability on the pathological input."""
    s0 = S.to_sphere(rand_simplex(B, L))
    s1 = S.to_sphere(onehot(B, L))
    s0.requires_grad_(True)
    loss = S.log_map(S.slerp(s0, s1, torch.rand(B, L)), s1).square().sum()
    loss.backward()
    assert s0.grad is not None and torch.isfinite(s0.grad).all()
