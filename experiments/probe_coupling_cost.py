"""Probe: does the coupling COST FUNCTION actually change the OT permutation?

Motivation. A candidate innovation is a geometry-matched minibatch-OT coupling: pair source
and target samples under the Fisher-Rao geodesic cost rather than a Euclidean one. FoldFlow-OT and Fisher-Flow both *use* a matched cost, but nobody appears to have
isolated the cost as the controlled variable and asked whether it matters.

There is a specific reason to fear it does not. Minibatch OT only consumes the cost matrix
through an argmin over permutations. On the sphere, chordal distance and geodesic angle are
related by chord = 2 sin(theta/2), a strictly monotone map. A monotone transform of
individual entries does NOT generally preserve the argmin of a SUM, so the permutations can
differ -- but they may coincide overwhelmingly often in practice, in which case the entire
"matched cost" idea is a no-op on this modality and must be dropped.

This probe answers that with pure numerics: no training, no GPU, no model. Agreement near
100% means the matched cost is a no-op here; low agreement means the idea is live.

Costs compared (all aggregated over sequence positions, squared):
  simplex_euclid : ||p0 - p1||^2          -- naive, ignores the geometry entirely
  sphere_chordal : ||s0 - s1||^2          -- extrinsic/ambient distance after sqrt map
  fisher_rao     : theta(s0, s1)^2        -- the intrinsic geodesic cost

Run: python experiments/probe_coupling_cost.py
Writes: results/probe_coupling_cost.json (overwritten).
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dgm.paths import simplex as S  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "probe_coupling_cost.json"

COSTS = ("simplex_euclid", "sphere_chordal", "fisher_rao")


def cost_matrix(p0: torch.Tensor, p1: torch.Tensor, kind: str) -> np.ndarray:
    """Pairwise (B0, B1) cost, summing a per-position squared distance over the sequence.

    p0, p1: (B, L, K) simplex points.
    """
    if kind == "simplex_euclid":
        d = (p0[:, None] - p1[None, :]).square().sum(-1)  # (B0, B1, L)
    else:
        s0, s1 = S.to_sphere(p0), S.to_sphere(p1)
        a, b = s0[:, None], s1[None, :]
        if kind == "sphere_chordal":
            d = (a - b).square().sum(-1)
        elif kind == "fisher_rao":
            d = S.sphere_angle(a.expand(-1, b.shape[1], -1, -1), b.expand(a.shape[0], -1, -1, -1)).square()
        else:
            raise ValueError(kind)
    return d.sum(-1).double().numpy()  # aggregate over sequence length


def sample_batch(b: int, length: int, k: int, alpha: float, gen: torch.Generator):
    """Source ~ Dirichlet(alpha) on the simplex; target ~ one-hot, as real sequence data is."""
    conc = torch.full((b, length, k), float(alpha))
    p0 = torch.distributions.Dirichlet(conc).sample()
    idx = torch.randint(0, k, (b, length), generator=gen)
    p1 = torch.nn.functional.one_hot(idx, k).float()
    return p0, p1


def run(b: int, length: int, k: int, alpha: float, trials: int, seed: int) -> dict:
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)

    agree = {f"{a}_vs_{c}": [] for a, c in itertools.combinations(COSTS, 2)}
    # Regret: extra TRUE (Fisher-Rao) transport cost paid by using another cost's permutation,
    # as a fraction of the optimal Fisher-Rao cost. This is what actually matters -- a
    # permutation can differ on many entries yet be nearly as good.
    regret = {c: [] for c in COSTS if c != "fisher_rao"}

    for _ in range(trials):
        p0, p1 = sample_batch(b, length, k, alpha, gen)
        mats = {c: cost_matrix(p0, p1, c) for c in COSTS}
        perms = {c: linear_sum_assignment(mats[c])[1] for c in COSTS}

        for a, c in itertools.combinations(COSTS, 2):
            agree[f"{a}_vs_{c}"].append(float((perms[a] == perms[c]).mean()))

        fr = mats["fisher_rao"]
        rows = np.arange(b)
        opt = fr[rows, perms["fisher_rao"]].sum()
        for c in regret:
            regret[c].append(float((fr[rows, perms[c]].sum() - opt) / opt))

    return {
        "batch": b,
        "length": length,
        "k": k,
        "alpha": alpha,
        "trials": trials,
        "agreement": {kk: float(np.mean(v)) for kk, v in agree.items()},
        "fr_regret": {kk: float(np.mean(v)) for kk, v in regret.items()},
    }


def main() -> int:
    k, length, trials = 4, 32, 20  # DNA alphabet, short sequences, enough trials to average
    rows = []
    print(f"{'B':>5} {'alpha':>6} | {'eucl~chord':>10} {'eucl~FR':>8} {'chord~FR':>9} |"
          f" {'regret(eucl)':>12} {'regret(chord)':>13}")
    print("-" * 78)
    for alpha in (0.5, 1.0):
        for b in (8, 32, 128, 256):
            r = run(b, length, k, alpha, trials, seed=0)
            rows.append(r)
            a = r["agreement"]
            g = r["fr_regret"]
            print(f"{b:>5} {alpha:>6} | {a['simplex_euclid_vs_sphere_chordal']:>10.3f}"
                  f" {a['simplex_euclid_vs_fisher_rao']:>8.3f}"
                  f" {a['sphere_chordal_vs_fisher_rao']:>9.3f} |"
                  f" {g['simplex_euclid']:>12.4f} {g['sphere_chordal']:>13.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {OUT.relative_to(Path.cwd())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
