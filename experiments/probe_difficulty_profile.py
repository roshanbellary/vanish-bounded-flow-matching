"""Probe: where along t is flow matching on the simplex actually hard?

Tests preregistered predictions P1/P2; the motivating measurement for PACE
(`pace_modality1.py`, which imports from this module).

Two established results predict OPPOSITE answers:

  * Stark et al. (arXiv:2402.05841) Prop. 1 -- for LINEAR flow matching on the simplex, the
    model posterior has support on at most k-1 vertices once t > 1/k, and for t > 1/2 it is
    exactly the argmax operator. Difficulty should therefore be concentrated EARLY, and the
    structure should MOVE WITH K, since vertices are eliminated at t = 1/K, 1/(K-1), ..., 1/2.
  * Fisher-Rao geometry (Davis et al., arXiv:2405.14664) -- one-hot data forces the flow to
    terminate at a vertex and the Fisher-Rao velocity norm diverges there. Difficulty should
    therefore be concentrated LATE.

These apply to different paths, so we measure BOTH paths rather than assuming one.

Two difficulty signals, both read off the trained model:

  loss(t)  -- held-out conditional FM regression loss in t-bins. What the training objective
              actually finds hard. Note the conditional target on the geodesic has norm
              theta(s0,s1), constant in t, so a rising loss is a real effect and not the
              1/(1-t) factor blowing up.
  accel(t) -- norm of the time-derivative of the learned field along sampling trajectories,
              by finite differences. This is what sets Euler truncation error, so it is the
              quantity that decides where sampling steps should go.

Setting: L=1 (a single categorical position). Chosen so the target distribution is known in
closed form for ANY K, making the KL exact rather than proxied, and so the measurement
isolates the geometry rather than the network's ability to model inter-position structure.

Run: python experiments/probe_difficulty_profile.py
Writes results/probe_difficulty_profile.json (overwritten).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dgm.paths import simplex as S  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT_JSON = REPO / "results" / "probe_difficulty_profile.json"
OUT_FIG = REPO / "paper" / "figures" / "difficulty_profile.png"

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
NBINS = 40


# ---------------------------------------------------------------- target distribution


def make_target(k: int, seed: int) -> Tensor:
    """A sparse-ish categorical target. The cube makes mass concentrate on a few categories,
    which is what makes the problem non-trivial -- a uniform target is learnable by a
    constant field and would tell us nothing about where the difficulty lies."""
    g = torch.Generator().manual_seed(seed)
    p = torch.rand(k, generator=g).pow(3.0)
    return (p / p.sum()).to(DEV)


def uniform_simplex_sample(n: int, k: int) -> Tensor:
    """Dirichlet(1,...,1), i.e. uniform on the simplex, via normalised Exp(1) variates.

    torch.distributions.Dirichlet has no MPS kernel (aten::_sample_dirichlet), and routing
    through the CPU fallback for every training batch would dominate the runtime. The
    exponential construction is exact for alpha=1, not an approximation.
    """
    e = -torch.rand(n, k, device=DEV).clamp_min(1e-12).log()
    return e / e.sum(-1, keepdim=True)


# ---------------------------------------------------------------- model


class VelocityNet(nn.Module):
    """Small MLP: (x_t, t) -> velocity. Deliberately tiny; we are measuring the geometry of
    the problem, not chasing sample quality."""

    def __init__(self, k: int, hidden: int = 256, n_freq: int = 16):
        super().__init__()
        self.n_freq = n_freq
        self.net = nn.Sequential(
            nn.Linear(k + 2 * n_freq, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, k),
        )

    def time_embed(self, t: Tensor) -> Tensor:
        freqs = torch.arange(self.n_freq, device=t.device, dtype=t.dtype)
        ang = t[:, None] * (2.0 ** freqs)[None, :] * math.pi
        return torch.cat([ang.sin(), ang.cos()], dim=-1)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return self.net(torch.cat([x, self.time_embed(t)], dim=-1))


# ---------------------------------------------------------------- the two paths
#
# Each path provides: sample a state at time t, and the conditional target velocity there.


def path_linear(x0: Tensor, x1: Tensor, t: Tensor):
    """Linear interpolation in simplex coordinates -- the setting Stark's Prop. 1 covers."""
    xt = (1 - t[:, None]) * x0 + t[:, None] * x1
    return xt, x1 - x0


def path_geodesic(x0: Tensor, x1: Tensor, t: Tensor):
    """Fisher-Rao geodesic, i.e. slerp on the sphere after the square-root map."""
    s0, s1 = S.to_sphere(x0), S.to_sphere(x1)
    st = S.slerp(s0, s1, t)
    # log_map(s_t, s_1) has norm (1-t)*theta(s0,s1), so this target has norm theta(s0,s1):
    # constant along the path. Any trend in the loss is therefore not an artefact of 1/(1-t).
    u = S.log_map(st, s1) / (1 - t[:, None]).clamp_min(1e-4)
    return st, u


PATHS = {"linear": path_linear, "geodesic": path_geodesic}


def sample_pair(target: Tensor, n: int, k: int):
    """Source ~ Dirichlet(1) on the simplex; target ~ the data distribution, one-hot."""
    x0 = uniform_simplex_sample(n, k)
    idx = torch.multinomial(target.expand(n, k), 1).squeeze(-1)
    x1 = torch.nn.functional.one_hot(idx, k).float()
    return x0, x1


# ---------------------------------------------------------------- train


def train(target: Tensor, k: int, path: str, steps: int, seed: int) -> VelocityNet:
    torch.manual_seed(seed)
    model = VelocityNet(k).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    fn = PATHS[path]
    for _ in range(steps):
        x0, x1 = sample_pair(target, 512, k)
        t = torch.rand(512, device=DEV)  # uniform t: the default we are interrogating
        xt, u = fn(x0, x1, t)
        v = model(xt, t)
        if path == "geodesic":
            v = S.tangent_project(xt, v)
        loss = (v - u).square().sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval()


# ---------------------------------------------------------------- measure


@torch.no_grad()
def loss_profile(model, target, k, path, n=200_000) -> np.ndarray:
    """Held-out conditional regression loss, binned in t."""
    fn = PATHS[path]
    sums = torch.zeros(NBINS, device=DEV)
    cnts = torch.zeros(NBINS, device=DEV)
    for _ in range(n // 20_000):
        x0, x1 = sample_pair(target, 20_000, k)
        t = torch.rand(20_000, device=DEV)
        xt, u = fn(x0, x1, t)
        v = model(xt, t)
        if path == "geodesic":
            v = S.tangent_project(xt, v)
        per = (v - u).square().sum(-1)
        b = (t * NBINS).long().clamp(0, NBINS - 1)
        sums.index_add_(0, b, per)
        cnts.index_add_(0, b, torch.ones_like(per))
    return (sums / cnts.clamp_min(1)).cpu().numpy()


@torch.no_grad()
def accel_profile(model, k, path, n=20_000, steps=200) -> np.ndarray:
    """||dv/dt|| along sampling trajectories, by finite differences.

    This is the quantity that sets Euler truncation error, so it is what should decide where
    sampling steps go. Integrated on a uniform grid so the measurement itself is unbiased
    with respect to the allocation we are about to propose.
    """
    x = uniform_simplex_sample(n, k)
    if path == "geodesic":
        x = S.to_sphere(x)
    h = 1.0 / steps
    acc = np.zeros(NBINS)
    cnt = np.zeros(NBINS)
    prev = None
    for i in range(steps):
        t = torch.full((n,), i * h, device=DEV)
        v = model(x, t)
        if path == "geodesic":
            v = S.tangent_project(x, v)
        if prev is not None:
            # Ambient finite difference. On the sphere the two vectors live in tangent spaces
            # at nearby points; for h=1/200 the parallel-transport correction is second order
            # and below the effect size we care about.
            a = (v - prev).norm(dim=-1) / h
            b = min(int((i * h) * NBINS), NBINS - 1)
            acc[b] += a.sum().item()
            cnt[b] += n
        prev = v
        x = S.exp_map(x, h * v) if path == "geodesic" else x + h * v
    return acc / np.maximum(cnt, 1)


@torch.no_grad()
def exact_kl(model, target, k, path, nfe: int, n=100_000) -> float:
    """Exact KL(p_data || p_model). Exact because L=1: the model's output distribution is a
    histogram over K categories and the target is known in closed form."""
    x = uniform_simplex_sample(n, k)
    if path == "geodesic":
        x = S.to_sphere(x)
    h = 1.0 / nfe
    for i in range(nfe):
        t = torch.full((n,), i * h, device=DEV)
        v = model(x, t)
        if path == "geodesic":
            v = S.tangent_project(x, v)
        x = S.exp_map(x, h * v) if path == "geodesic" else x + h * v
    idx = x.argmax(-1)
    q = torch.bincount(idx, minlength=k).float()
    q = (q + 1.0) / (q.sum() + k)  # Laplace smoothing so KL stays finite on a missed mode
    p = target
    return float((p * (p.clamp_min(1e-12).log() - q.log())).sum())


# ---------------------------------------------------------------- report


def summarise(prof: np.ndarray) -> dict:
    """P1 asks for max/min >= 2.0; P2 asks whether the mode sits away from t=0.5."""
    centers = (np.arange(NBINS) + 0.5) / NBINS
    sm = np.convolve(prof, np.ones(3) / 3, mode="same")  # mild smoothing before taking a mode
    return {
        "ratio_max_min": float(prof.max() / max(prof.min(), 1e-12)),
        "mode_t": float(centers[int(np.argmax(sm))]),
        "mass_first_half": float(prof[: NBINS // 2].sum() / prof.sum()),
        "profile": prof.tolist(),
    }


def main() -> int:
    ks = (4, 20, 60)
    steps = {4: 4000, 20: 6000, 60: 8000}
    out: dict = {"nbins": NBINS, "device": str(DEV), "runs": []}

    print(f"device={DEV}\n")
    hdr = f"{'path':>9} {'K':>4} | {'ratio':>7} {'mode_t':>7} {'mass<0.5':>9} | {'KL@5':>8} {'KL@100':>8}"
    print(hdr)
    print("-" * len(hdr))

    for path in ("linear", "geodesic"):
        for k in ks:
            target = make_target(k, seed=0)
            model = train(target, k, path, steps[k], seed=0)
            lp = loss_profile(model, target, k, path)
            ap = accel_profile(model, k, path)
            row = {
                "path": path,
                "k": k,
                "train_steps": steps[k],
                "loss": summarise(lp),
                "accel": summarise(ap),
                "kl": {str(n): exact_kl(model, target, k, path, n) for n in (5, 10, 20, 100)},
            }
            out["runs"].append(row)
            print(f"{path:>9} {k:>4} | {row['loss']['ratio_max_min']:>7.2f}"
                  f" {row['loss']['mode_t']:>7.3f} {row['loss']['mass_first_half']:>9.3f} |"
                  f" {row['kl']['5']:>8.4f} {row['kl']['100']:>8.4f}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT_JSON.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
