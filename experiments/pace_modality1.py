"""PACE go/no-go on Modality 1: does allocating training samples and sampling steps by a MEASURED profile beat uniform
allocation, at matched compute?

Design (two-stage, like DC-FM; the pilot cost is reported):

  Stage 1  Train a pilot model with uniform t. Measure the per-t acceleration profile
           ||dv/dt|| along sampling trajectories. Acceleration, not loss: the per-t loss is
           dominated by the irreducible conditional variance Var(u|x_t), which is maximal at
           t=0 for every conditional FM model regardless of geometry and cannot be reduced by
           spending more samples there. Acceleration is what sets Euler truncation error, so
           it is the quantity a reallocation can actually exploit.

  Stage 2  Retrain from scratch under three t-densities, at IDENTICAL step count, batch size,
           architecture and seed:
             uniform      -- the field's default
             logitnormal  -- the SD3 default, whose SNR motivation has no analogue on
                             the simplex
             matched      -- density proportional to the measured profile, with the 1/p(t)
                             importance weight so the objective is UNCHANGED. This is variance
                             reduction, not a different loss.
           Then evaluate each under two inference grids (uniform, matched); the grid costs no
           training.

Preregistered primary endpoint: exact KL(p_data || p_model) vs NFE. Exact because L=1 -- the
model's output law is a histogram over K categories and the target is known in closed form.

Run: python experiments/pace_modality1.py [--quick]      # --quick: 1 seed, K=4, smoke
Writes results/pace_modality1.json (overwritten).
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dgm.paths import simplex as S  # noqa: E402
from probe_difficulty_profile import (  # noqa: E402
    DEV, NBINS, PATHS, VelocityNet, accel_profile, make_target, sample_pair,
    uniform_simplex_sample,
)

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "pace_modality1.json"

NFES = (2, 5, 10, 20, 50, 100)


# ---------------------------------------------------------------- t densities
#
# Each returns (t, w) where w is the importance weight that keeps the objective equal to the
# uniform-t objective, so a non-uniform density is variance reduction rather than a
# different loss.


def density_uniform(n: int, _p: np.ndarray | None):
    return torch.rand(n, device=DEV), torch.ones(n, device=DEV)


def density_logitnormal(n: int, _p: np.ndarray | None):
    """SD3's choice: t = sigmoid(z), z ~ N(0,1). Concentrates near t=0.5."""
    z = torch.randn(n, device=DEV)
    t = torch.sigmoid(z)
    # density of t: p(t) = phi(logit(t)) / (t(1-t))
    tc = t.clamp(1e-6, 1 - 1e-6)
    logit = torch.log(tc / (1 - tc))
    phi = torch.exp(-0.5 * logit**2) / np.sqrt(2 * np.pi)
    p = phi / (tc * (1 - tc))
    return t, 1.0 / p.clamp_min(1e-6)


def _cdf_from_profile(prof: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.maximum(prof.astype(np.float64), 1e-12)
    p = p / p.sum()  # per-bin probability mass
    return p, np.concatenate([[0.0], np.cumsum(p)])


def density_matched(n: int, prof: np.ndarray):
    """Sample t with density proportional to the measured profile, piecewise-uniform in bins."""
    p, cdf = _cdf_from_profile(prof)
    u = torch.rand(n, device=DEV).cpu().numpy()
    b = np.clip(np.searchsorted(cdf, u, side="right") - 1, 0, NBINS - 1)
    frac = (u - cdf[b]) / np.maximum(p[b], 1e-12)
    t = (b + frac) / NBINS
    dens = p[b] * NBINS  # density of the piecewise-uniform law at t, wrt Lebesgue on [0,1]
    return (torch.tensor(t, device=DEV, dtype=torch.float32),
            torch.tensor(1.0 / np.maximum(dens, 1e-6), device=DEV, dtype=torch.float32))


DENSITIES = {
    "uniform": density_uniform,
    "logitnormal": density_logitnormal,
    "matched": density_matched,
}


# ---------------------------------------------------------------- inference grids


def grid_uniform(nfe: int, _p: np.ndarray | None) -> np.ndarray:
    return np.linspace(0.0, 1.0, nfe + 1)


def grid_matched(nfe: int, prof: np.ndarray) -> np.ndarray:
    """Equal-error grid: step boundaries placed so each step covers equal accumulated
    acceleration. Where the field bends hard, steps get short."""
    p, cdf = _cdf_from_profile(prof)
    targets = np.linspace(0.0, 1.0, nfe + 1)
    edges = np.interp(targets, cdf, np.arange(NBINS + 1) / NBINS)
    edges[0], edges[-1] = 0.0, 1.0
    return np.maximum.accumulate(edges)


GRIDS = {"uniform": grid_uniform, "matched": grid_matched}


# ---------------------------------------------------------------- train / eval


def train(target, k, path, density, prof, steps, seed, batch=512):
    torch.manual_seed(seed)
    model = VelocityNet(k).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    fn, dens = PATHS[path], DENSITIES[density]
    for _ in range(steps):
        x0, x1 = sample_pair(target, batch, k)
        t, w = dens(batch, prof)
        xt, u = fn(x0, x1, t)
        v = model(xt, t)
        if path == "geodesic":
            v = S.tangent_project(xt, v)
        # w is the 1/p(t) unbiasing weight: the expected objective equals the uniform-t one.
        loss = (w * (v - u).square().sum(-1)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval()


@torch.no_grad()
def kl_on_grid(model, target, k, path, edges: np.ndarray, n=200_000) -> float:
    x = uniform_simplex_sample(n, k)
    if path == "geodesic":
        x = S.to_sphere(x)
    for i in range(len(edges) - 1):
        t0, h = float(edges[i]), float(edges[i + 1] - edges[i])
        if h <= 0:
            continue
        t = torch.full((n,), t0, device=DEV)
        v = model(x, t)
        if path == "geodesic":
            v = S.tangent_project(x, v)
        x = S.exp_map(x, h * v) if path == "geodesic" else x + h * v
    q = torch.bincount(x.argmax(-1), minlength=k).float()
    q = (q + 1.0) / (q.sum() + k)
    p = target
    return float((p * (p.clamp_min(1e-12).log() - q.log())).sum())


# ---------------------------------------------------------------- driver


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="1 seed, K=4 only: a smoke run")
    a = ap.parse_args()

    ks = (4,) if a.quick else (4, 60)
    seeds = (0,) if a.quick else (0, 1, 2, 3, 4)
    paths = ("geodesic",) if a.quick else ("linear", "geodesic")
    steps = {4: 4000, 60: 8000}
    t_start = time.time()

    out = {"nfes": list(NFES), "seeds": list(seeds), "runs": [], "pilots": []}

    for path, k in itertools.product(paths, ks):
        target = make_target(k, seed=0)
        # --- Stage 1: pilot. One uniform-t model per (path, K); its profile is reused for
        # every matched condition. Cost is 1/(len(densities)*len(seeds)) of the total.
        pilot = train(target, k, path, "uniform", None, steps[k], seed=99)
        prof = accel_profile(pilot, k, path)
        prof = np.convolve(prof, np.ones(5) / 5, mode="same")  # denoise before using as density
        out["pilots"].append({"path": path, "k": k, "accel_profile": prof.tolist()})
        print(f"[pilot] {path} K={k}  accel ratio={prof.max()/max(prof.min(),1e-9):.2f} "
              f"mode_t={(np.argmax(prof)+0.5)/NBINS:.3f}")

        for density, seed in itertools.product(DENSITIES, seeds):
            m = train(target, k, path, density, prof, steps[k], seed)
            for gname, gfn in GRIDS.items():
                for nfe in NFES:
                    out["runs"].append({
                        "path": path, "k": k, "density": density, "grid": gname,
                        "seed": seed, "nfe": nfe,
                        "kl": kl_on_grid(m, target, k, path, gfn(nfe, prof)),
                    })
            print(f"  {path} K={k} {density:>12} seed={seed}  "
                  f"KL@5(unif)={out['runs'][-2*len(NFES)+1]['kl']:.4f}")

    out["wall_seconds"] = round(time.time() - t_start, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT.relative_to(REPO)}  ({out['wall_seconds']:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
