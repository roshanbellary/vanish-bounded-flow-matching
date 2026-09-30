"""Geometry of the probability simplex.

Modality 1 lives on the interior of the simplex

    Delta^{K-1} = { p in R^K : p_i >= 0, sum_i p_i = 1 },

carrying the Fisher-Rao metric. The reason we do not just run Euclidean flow matching on
simplex coordinates is that the Fisher-Rao metric blows up at the boundary: near a vertex,
a vanishing step in probability is an unbounded step in the natural geometry of the space.
A linear interpolant therefore spends most of its arclength in a region where the model has
to produce enormous velocities, which is the documented source of instability in naive
simplex flow matching.

The standard fix is the square-root map, which is an isometry from the Fisher-Rao simplex to
the positive orthant of a Euclidean sphere:

    phi(p) = sqrt(p),     phi^{-1}(s) = s^2,      ||phi(p)||_2 = 1.

Under this map the Fisher-Rao geodesic between two distributions becomes an ordinary
great-circle arc, so interpolation is slerp and the geodesic distance is an angle. Sphere
paths (e.g. the Fisher-Flow baseline) are defined here and pushed back to the simplex for
decoding.

Conventions used throughout:
  * ``p`` denotes simplex points, shape (..., K), rows summing to 1.
  * ``s`` denotes sphere points, shape (..., K), unit L2 norm, non-negative entries.
  * The unit-radius convention is used, so Fisher-Rao distance is ``arccos<s0, s1>`` in
    [0, pi/2] for the positive orthant. Some papers use radius 2 and report twice this.
"""

from __future__ import annotations

import torch
from torch import Tensor

# Below this angle two sphere points are treated as coincident and slerp falls back to
# a linear blend. Chosen well above float32 epsilon for arccos, which loses precision
# badly as the inner product approaches 1.
_ANGLE_EPS = 1e-6


def to_sphere(p: Tensor, eps: float = 1e-8) -> Tensor:
    """Map simplex points to the positive orthant of the unit sphere.

    ``eps`` guards the square root against exact zeros, which are common when the input
    is one-hot. Without it the gradient of sqrt at 0 is infinite and training diverges on
    the very first batch of real (one-hot) sequence data.
    """
    p = p.clamp_min(eps)
    p = p / p.sum(dim=-1, keepdim=True)
    return torch.sqrt(p)


def from_sphere(s: Tensor) -> Tensor:
    """Map sphere points back to the simplex. Inverse of :func:`to_sphere`."""
    p = s.square()
    return p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def project_to_sphere(s: Tensor, eps: float = 1e-12) -> Tensor:
    """Renormalise onto the unit sphere.

    ODE integration leaves the sphere at every step because an Euler update follows the
    tangent, not the manifold. Every sampler step must re-project or the trajectory
    spirals off the manifold and the decoded probabilities stop being meaningful.
    """
    return s / s.norm(dim=-1, keepdim=True).clamp_min(eps)


def fisher_rao_distance(p: Tensor, q: Tensor) -> Tensor:
    """Fisher-Rao (great-circle) distance between simplex points, in [0, pi/2].

    This is the cost function that a geometry-aware coupling would use in place of the
    Euclidean distance between one-hot vectors.
    """
    s0, s1 = to_sphere(p), to_sphere(q)
    return sphere_angle(s0, s1)


def sphere_angle(s0: Tensor, s1: Tensor) -> Tensor:
    """Great-circle angle between unit vectors, in [0, pi].

    Uses the half-angle form ``2*atan2(||s0 - s1||, ||s0 + s1||)`` rather than
    ``arccos<s0, s1>``. The arccos form is ill-conditioned near zero angle: its derivative
    is unbounded there, so guarding it with a clamp imposes a hard floor on the returned
    distance (clamping the inner product at 1 - 1e-7 floors the angle at ~4.5e-4, and every
    pair closer than that collapses to the same value). Since this function is the cost
    used for geometry-aware couplings, such a floor would silently make nearby pairs
    indistinguishable. The atan2 form is exact at 0 and well-conditioned throughout.
    """
    diff = (s0 - s1).norm(dim=-1)
    summ = (s0 + s1).norm(dim=-1)
    return 2.0 * torch.atan2(diff, summ)


def slerp(s0: Tensor, s1: Tensor, t: Tensor) -> Tensor:
    """Geodesic interpolation on the sphere: the Fisher-Rao path on the simplex.

    ``t`` broadcasts against the batch dimensions of ``s0``/``s1`` and is measured with
    t=0 at ``s0`` and t=1 at ``s1``.

    The near-parallel branch matters more here than in graphics uses of slerp: real
    sequence data is one-hot, so whenever source and target agree on a position the two
    sphere points coincide exactly and ``sin(theta)`` underflows to zero.
    """
    if t.dim() < s0.dim():
        t = t.reshape(t.shape + (1,) * (s0.dim() - t.dim()))

    theta = sphere_angle(s0, s1).unsqueeze(-1)
    sin_theta = torch.sin(theta)

    near = theta.abs() < _ANGLE_EPS
    # torch.where evaluates both branches, so the denominator is sanitised first to keep
    # NaNs from the degenerate branch out of the backward pass.
    safe_sin = torch.where(near, torch.ones_like(sin_theta), sin_theta)
    w0 = torch.sin((1.0 - t) * theta) / safe_sin
    w1 = torch.sin(t * theta) / safe_sin
    geo = w0 * s0 + w1 * s1
    lin = (1.0 - t) * s0 + t * s1

    return project_to_sphere(torch.where(near, lin, geo))


def log_map(s0: Tensor, s1: Tensor) -> Tensor:
    """Logarithm map at ``s0`` applied to ``s1``: the tangent vector pointing along the
    geodesic from ``s0`` to ``s1``, with norm equal to the geodesic distance.

    This is the conditional target velocity for geodesic flow matching on the sphere:
    a model trained to regress ``log_map(s_t, s_1) / (1 - t)`` learns the vector field
    whose flow transports the source distribution to the data distribution.
    """
    theta = sphere_angle(s0, s1).unsqueeze(-1)
    # Component of s1 orthogonal to s0, i.e. the direction of travel in the tangent space.
    resid = s1 - (s0 * s1).sum(dim=-1, keepdim=True) * s0
    norm = resid.norm(dim=-1, keepdim=True)
    near = norm < _ANGLE_EPS
    safe_norm = torch.where(near, torch.ones_like(norm), norm)
    return torch.where(near, torch.zeros_like(resid), theta * resid / safe_norm)


def exp_map(s: Tensor, v: Tensor) -> Tensor:
    """Exponential map at ``s`` along tangent vector ``v``.

    Used by the geodesic (manifold-respecting) sampler step, as opposed to a Euclidean
    Euler step followed by re-projection. Keeping both available matters: whether the
    sampler respects the geometry is itself an ablation axis.
    """
    v = v - (s * v).sum(dim=-1, keepdim=True) * s  # enforce tangency
    norm = v.norm(dim=-1, keepdim=True)
    near = norm < _ANGLE_EPS
    safe_norm = torch.where(near, torch.ones_like(norm), norm)
    out = torch.cos(norm) * s + torch.sin(norm) * v / safe_norm
    return project_to_sphere(torch.where(near, s, out))


def tangent_project(s: Tensor, v: Tensor) -> Tensor:
    """Project an ambient vector onto the tangent space at ``s``.

    The network emits an unconstrained R^K vector; only its tangential component is a
    valid velocity on the sphere. Applying this inside the loss rather than only at
    sampling time stops the model from wasting capacity on the radial component.
    """
    return v - (s * v).sum(dim=-1, keepdim=True) * s


def decode(p_or_s: Tensor, from_sphere_coords: bool = False) -> Tensor:
    """Decode continuous state to discrete token indices by argmax.

    The gap between the continuous state and this argmax is a real and under-reported
    error source on this modality: a model can have excellent continuous loss while its
    decoded sequences are poor, because argmax discards all the mass placed off the
    winning vertex.
    """
    p = from_sphere(p_or_s) if from_sphere_coords else p_or_s
    return p.argmax(dim=-1)


def uniform_simplex(shape: tuple[int, ...], k: int, device=None, dtype=None) -> Tensor:
    """The barycentre of the simplex, the natural uninformative source point."""
    return torch.full((*shape, k), 1.0 / k, device=device, dtype=dtype)


def sample_dirichlet_source(
    shape: tuple[int, ...], k: int, alpha: float = 1.0, device=None, dtype=None
) -> Tensor:
    """Sample a source distribution on the simplex.

    ``alpha = 1`` is uniform over the simplex. Smaller alpha concentrates mass near the
    vertices (more one-hot-like sources); larger alpha concentrates near the barycentre.
    The source distribution is a coupling-relevant design choice, so it is exposed rather
    than hard-coded.
    """
    conc = torch.full((*shape, k), float(alpha), device=device, dtype=dtype)
    return torch.distributions.Dirichlet(conc).sample()
