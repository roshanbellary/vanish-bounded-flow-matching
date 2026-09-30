"""Candidate innovation: a soft tangency penalty with tunable weight (falsified; reported).

Predictions Q1-Q4 are preregistered.

Motivation, from our own Pareto measurement rather than a literature gap:

    K=4   unconstrained  violation 0.074  KL 0.0130      hard  violation 0.0097  KL 0.0163
    K=60  unconstrained  violation 0.971  KL 0.0666      hard  violation 0.0107  KL 0.4608

Hard constraint satisfaction (replicator parameterization) costs 25% of KL at K=4 and 690% at
K=60. Leaving the field unconstrained means 97% of steps exit the simplex at K=60. Both
extremes are bad in different ways and nobody has asked whether the middle dominates.

The mechanism. Keep the unconstrained parameterization, and add to the training objective a
penalty on the OUTWARD component of the velocity at the boundary-facing coordinates:

    L = || v - u ||^2  +  lambda * sum_i relu( -v_i )^2 * w_i(x)

where w_i(x) = (1 - x_i)  up-weights coordinates that are close to zero, i.e. exactly where an
outward velocity would push the state out of the simplex. The penalty is zero in the interior
and grows only where the constraint can actually bind, so it does not fight the regression
target where the target is unconstrained.

This is a soft version of the tangency condition that Cont (arXiv:2607.28344) Assumption F4
requires and Corollary 3.3 assumes is automatic. Our measurement says it is not automatic for
learned fields; this asks what it costs to ask for it.

Q2 is the success criterion and is deliberately demanding: there must exist a lambda whose
violation rate is within 2x of the hard constraint AND whose KL is within one standard
deviation of the unconstrained model, at K=60. Interpolating between the two extremes is not
enough; the soft penalty has to dominate both.

Run: python -u experiments/soft_penalty.py
Writes: results/soft_penalty.jsonl (truncated at start; one JSON line per finished cell).
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
    DEV, VelocityNet, make_target, sample_pair, uniform_simplex_sample,
)

OUT = REPO / "results" / "soft_penalty.jsonl"
NFES = (2, 5, 10, 20, 50, 100)
LAMBDAS = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)


def tangency_penalty(v: Tensor, x: Tensor) -> Tensor:
    """Penalise outward velocity, weighted toward coordinates near the boundary.

    relu(-v_i) is the outward (simplex-exiting) component of coordinate i. Weighting by
    (1 - x_i) concentrates the penalty where x_i is small -- the only place an outward
    velocity can actually cause a violation within one step.
    """
    outward = torch.relu(-v)
    return (outward.square() * (1.0 - x)).sum(-1)


def train(target, k, lam, steps, seed, batch=512):
    torch.manual_seed(seed)
    net = VelocityNet(k).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    for _ in range(steps):
        x0, x1 = sample_pair(target, batch, k)
        t = torch.rand(batch, device=DEV)
        xt = (1 - t[:, None]) * x0 + t[:, None] * x1
        u = x1 - x0
        v = net(xt, t)
        loss = (v - u).square().sum(-1).mean()
        if lam > 0:
            loss = loss + lam * tangency_penalty(v, xt).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return net.eval()


@torch.no_grad()
def evaluate(net, target, k, nfe, n=50_000):
    x = uniform_simplex_sample(n, k)
    h = 1.0 / nfe
    bad = 0
    for i in range(nfe):
        t = torch.full((n,), i * h, device=DEV)
        y = x + h * net(x, t)
        bad += int((y.min(-1).values < 0).sum())
        x = y.clamp_min(0.0)
        x = x / x.sum(-1, keepdim=True).clamp_min(1e-12)
    q = torch.bincount(x.argmax(-1), minlength=k).float().cpu().numpy()
    q = (q + 1.0) / (q.sum() + k)
    p = target.cpu().numpy(); p = p / p.sum()
    return float((p * (np.log(np.maximum(p, 1e-12)) - np.log(q))).sum()), bad / (n * nfe)


def main() -> int:
    ks, seeds = (4, 60), (0, 1, 2)
    steps = {4: 4000, 60: 8000}
    t0 = time.time()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("")

    print(f"{'K':>4} {'lambda':>8} {'seed':>5} | {'KL@100':>9} {'KL@10':>9} {'violation':>10}")
    print("-" * 56)
    for k in ks:
        target = make_target(k, seed=0)
        for lam in LAMBDAS:
            for seed in seeds:
                net = train(target, k, lam, steps[k], seed)
                res = {str(n): evaluate(net, target, k, n) for n in NFES}
                kl = {n: res[n][0] for n in res}
                viol = res["100"][1]
                row = {"k": k, "lam": lam, "seed": seed, "kl": kl,
                       "violation_rate_nfe100": viol}
                with OUT.open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
                print(f"{k:>4} {lam:>8.2f} {seed:>5} | {kl['100']:>9.4f} "
                      f"{kl['10']:>9.4f} {viol:>10.4f}")
    print(f"\nwrote {OUT.relative_to(REPO)}  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
