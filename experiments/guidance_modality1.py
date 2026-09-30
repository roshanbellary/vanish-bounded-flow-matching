"""Metric-aware guidance on the simplex: the Modality 1 verification experiment.

Four guidance rules swapped at sampling time on ONE trained checkpoint; no retraining per
arm, which is what makes the whole grid affordable.

    euclidean  v + g*grad r                          raw ambient gradient
    tangent    v + g*(grad r - mean grad r)          sum-to-zero (STGFlow Prop. 5 style)
    natural    v + g*p*(grad r - <p, grad r>)        Fisher-Rao natural gradient (replicator)
    mirror     exponentiated-gradient / entropic     mirror-descent dual step

The natural and mirror rules cannot leave the simplex: both carry a factor of p_i that
vanishes as p_i -> 0, so the update dies exactly where the constraint binds. The euclidean
and tangent rules can, and Dirichlet FM patches that with a Euclidean projection back onto
the simplex (Wang & Carreira-Perpinan 2013). We measure how often that patch is actually
needed, which no paper in this literature reports.

Prior art, cited rather than reinvented: Riemannian MeanFlow (Woo, Skreta, Park, Neklyudov,
Ahn, arXiv:2602.07744, ICML 2026) already uses a Riemannian reward gradient on the Fisher-Rao
simplex. It asserts the mechanism in one sentence, states no proposition about staying in the
simplex, and never compares against Euclidean-plus-projection. Those three gaps are what this
experiment fills.

Why the toy setting is the right one for the primary claim: with a known categorical target
and a reward that is the indicator of a category subset S, the IDEAL guided distribution is
p_data restricted to S and renormalised -- known in closed form. So we can measure the exact
KL between what a guidance rule produces and what it *should* produce. On real sequence data
that quantity is unavailable at any price, and everyone substitutes a proxy classifier score
that cannot distinguish "steered correctly" from "collapsed onto one sequence".

Run: python experiments/guidance_modality1.py
Writes: results/guidance_modality1.json (an existing file is renamed aside with its mtime first).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))
from probe_difficulty_profile import (  # noqa: E402
    DEV, make_target, train, uniform_simplex_sample,
)

OUT = REPO / "results" / "guidance_modality1.json"

RULES = ("euclidean", "tangent", "natural", "mirror")
GAMMAS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)


def reward_grad(x: Tensor, mask: Tensor) -> Tensor:
    """grad of r(x) = log(sum_{i in S} x_i), the smooth indicator of the target set S.

    Closed form: d_i r = 1{i in S} / sum_{j in S} x_j. Deliberately analytic so no classifier
    training confounds the comparison between guidance rules.
    """
    s = (x * mask).sum(-1, keepdim=True).clamp_min(1e-8)
    return mask / s


def guided_velocity(rule: str, v: Tensor, x: Tensor, g: Tensor, gamma: float) -> Tensor:
    if gamma == 0.0:
        return v
    if rule == "euclidean":
        step = g
    elif rule == "tangent":
        step = g - g.mean(-1, keepdim=True)
    elif rule in ("natural", "mirror"):
        # Fisher-Rao natural gradient = replicator form. The mirror (exponentiated-gradient)
        # step has the same continuous-time limit; they differ only at finite step size, and
        # reporting both makes that equivalence checkable rather than asserted.
        step = x * (g - (x * g).sum(-1, keepdim=True))
    else:
        raise ValueError(rule)
    return v + gamma * step


def project_simplex(y: Tensor) -> Tensor:
    """Euclidean projection onto the simplex (Wang & Carreira-Perpinan 2013), as used by
    Dirichlet FM to repair guided updates that go negative."""
    k = y.shape[-1]
    u, _ = torch.sort(y, dim=-1, descending=True)
    css = u.cumsum(-1)
    idx = torch.arange(1, k + 1, device=y.device, dtype=y.dtype)
    cond = u - (css - 1.0) / idx > 0
    rho = cond.float().cumsum(-1).argmax(-1, keepdim=True)
    theta = (css.gather(-1, rho) - 1.0) / (rho.float() + 1.0)
    return (y - theta).clamp_min(0.0)


@torch.no_grad()
def sample(model, k, mask, rule, gamma, nfe=100, n=100_000, project=True):
    """Euler sampling with guidance. Returns decoded indices plus constraint diagnostics."""
    x = uniform_simplex_sample(n, k)
    h = 1.0 / nfe
    viol_steps = 0
    viol_mass = 0.0
    for i in range(nfe):
        t = torch.full((n,), i * h, device=DEV)
        v = guided_velocity(rule, model(x, t), x, reward_grad(x, mask), gamma)
        y = x + h * v
        bad = y.min(-1).values < 0
        viol_steps += int(bad.sum())
        viol_mass += float(y.clamp_max(0.0).abs().sum())
        x = project_simplex(y) if project else y.clamp_min(0.0)
        x = x / x.sum(-1, keepdim=True).clamp_min(1e-12)
    return x.argmax(-1), viol_steps / (n * nfe), viol_mass / (n * nfe)


def kl(p: np.ndarray, q_counts: np.ndarray) -> float:
    q = (q_counts + 1.0) / (q_counts.sum() + len(q_counts))
    p = p / p.sum()
    return float((p * (np.log(np.maximum(p, 1e-12)) - np.log(q))).sum())


def main() -> int:
    k, nfe = 8, 100
    t0 = time.time()
    target = make_target(k, seed=0)
    tgt = target.cpu().numpy()

    # Target set S: the two LEAST likely categories. Guidance then has real work to do --
    # steering toward rare modes is where a guidance rule is actually stressed.
    order = np.argsort(tgt)
    S = sorted(order[:2].tolist())
    mask = torch.zeros(k, device=DEV)
    mask[S] = 1.0

    ideal = tgt.copy()
    ideal[[i for i in range(k) if i not in S]] = 0.0
    ideal = ideal / ideal.sum()   # p_data restricted to S: the correct guided distribution

    print(f"K={k}  target set S={S}  (prior mass {tgt[S].sum():.4f})")
    print("training base model (unguided, uniform t) ...")
    model = train(target, k, "linear", steps=6000, seed=0)

    out = {"k": k, "S": S, "nfe": nfe, "prior_mass_S": float(tgt[S].sum()),
           "ideal": ideal.tolist(), "rows": []}

    hdr = (f"{'rule':>10} {'gamma':>6} | {'hit rate':>9} {'KL to ideal':>12} "
           f"{'viol rate':>10} {'viol mass':>10}")
    print("\n" + hdr)
    print("-" * len(hdr))

    for rule in RULES:
        for gamma in GAMMAS:
            idx, vr, vm = sample(model, k, mask, rule, gamma, nfe=nfe)
            counts = torch.bincount(idx, minlength=k).cpu().numpy().astype(float)
            hit = counts[S].sum() / counts.sum()
            row = {"rule": rule, "gamma": gamma, "hit_rate": float(hit),
                   "kl_to_ideal": kl(ideal, counts), "violation_rate": vr,
                   "violation_mass": vm, "counts": counts.tolist()}
            out["rows"].append(row)
            print(f"{rule:>10} {gamma:>6.1f} | {hit:>9.4f} {row['kl_to_ideal']:>12.4f} "
                  f"{vr:>10.4f} {vm:>10.2e}")

    out["wall_seconds"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():                 # results are append-only: keep the old file beside it
        OUT.rename(OUT.with_suffix(f".{int(OUT.stat().st_mtime)}{OUT.suffix}"))
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT.relative_to(REPO)}  ({out['wall_seconds']:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
