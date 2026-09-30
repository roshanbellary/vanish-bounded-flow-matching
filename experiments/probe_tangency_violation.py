"""Is the unguided simplex violation a real property of the learned field, or Euler error?

The guidance head-to-head found that at gamma=0 -- no guidance at all -- the sampler leaves
the simplex on 21% of steps. Before treating that as a finding, it has to survive the obvious
confound: Euler local truncation error is O(h^2), so if the violations are discretization,
shrinking the step size should kill them.

Decision rule, fixed before running: if the PER-STEP violation rate falls steeply with h it
is an artifact; if it plateaus, the learned velocity field genuinely points off the simplex
and no step size saves it.

Uses the SAME base field as the guidance experiment (K=8, linear path, 6000 steps, seed 0,
via `probe_difficulty_profile.train`), so its NFE=100 row reproduces that experiment's
unguided violation rate. Synthetic, one seed: the paper's step-size claim is limited to this.

Run: python -u experiments/probe_tangency_violation.py
Writes: results/probe_tangency_violation.json, which the paper reads (an existing file is
        renamed aside with its mtime first).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))
from probe_difficulty_profile import (  # noqa: E402
    DEV, make_target, train, uniform_simplex_sample,
)

OUT = REPO / "results" / "probe_tangency_violation.json"
NFES = (25, 50, 100, 200, 400, 800)
N_TRAJ = 50_000


@torch.no_grad()
def sweep(model, k: int, nfe: int, n: int):
    """Unguided Euler with clamp + renormalise, the sampler every Modality 1 arm uses."""
    x = uniform_simplex_sample(n, k)
    h = 1.0 / nfe
    bad_steps, mass, ever = 0, 0.0, torch.zeros(n, dtype=torch.bool, device=DEV)
    for i in range(nfe):
        y = x + h * model(x, torch.full((n,), i * h, device=DEV))
        bad = y.min(-1).values < 0
        bad_steps += int(bad.sum())
        ever |= bad
        mass += float(y.clamp_max(0.0).abs().sum())
        x = y.clamp_min(0.0)
        x = x / x.sum(-1, keepdim=True).clamp_min(1e-12)
    return bad_steps / (n * nfe), mass / (n * nfe), float(ever.float().mean())


def main() -> int:
    k, t0 = 8, time.time()
    torch.manual_seed(0)
    model = train(make_target(k, seed=0), k, "linear", steps=6000, seed=0)
    rows = []
    print(f"{'NFE':>6} {'h':>8} | {'per-step rate':>14} {'mass/step':>10} {'traj ever bad':>14}")
    for nfe in NFES:
        rate, mass, ever = sweep(model, k, nfe, N_TRAJ)
        rows.append({"nfe": nfe, "h": 1.0 / nfe, "step_rate": rate, "mass_per_step": mass,
                     "traj_ever_bad": ever})
        print(f"{nfe:>6} {1/nfe:>8.4f} | {rate:>14.4f} {mass:>10.2e} {ever:>14.4f}", flush=True)
    out = {"k": k, "n_traj": N_TRAJ, "device": str(DEV), "rows": rows,
           "wall_seconds": time.time() - t0}
    if OUT.exists():                 # results are append-only: keep the old file beside it
        OUT.rename(OUT.with_suffix(f".{int(OUT.stat().st_mtime)}.json"))
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT.relative_to(REPO)} ({out['wall_seconds']:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
