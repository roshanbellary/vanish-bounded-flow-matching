"""Does a constraint-preserving field parameterization work, and what does it cost?

Tests idea 1 of `docs/ideas_v2.md`: on this synthetic benchmark the learned velocity field
points outside the simplex on ~21% of visited states, flat across a 32x range of step sizes,
so it is not discretization error.

Two parameterizations, identical architecture, optimiser, steps, batch and seed:

  unconstrained   v = net(x, t)                      what everyone does; needs clamping
  replicator      w = net(x, t); v = x * (w - <x,w>) v_i -> 0 as x_i -> 0, so the flow
                                                     provably cannot leave the simplex

The framing: classical constrained diffusions (Wright-Fisher, Jacobi, CIR) stay in their
domain because drift AND diffusion degenerate at the boundary -- the Feller condition for
well-posedness without reflection. Learned flows inherit the domain and discard the property.

TWO OUTCOMES, BOTH INFORMATIVE. The replicator form must drive violations to zero; that is
arithmetic, not a hypothesis. The real question is the COST: the constrained family may not
be able to represent the true marginal field, trading constraint violations for approximation
error. If KL is unchanged or better, the constraint is free and clamping was never needed. If
KL is worse, the expressiveness price is the finding.

Run: python -u experiments/constrained_field.py
Writes: results/constrained_field.jsonl (truncated at start; one JSON line per finished cell).
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

OUT = REPO / "results" / "constrained_field.jsonl"
NFES = (2, 5, 10, 20, 50, 100)


def field(net, x: Tensor, t: Tensor, param: str) -> Tensor:
    w = net(x, t)
    if param == "unconstrained":
        return w
    if param == "replicator":
        # x_i multiplies the output, so the velocity vanishes on every face; the <x,w>
        # term keeps the update on the affine hull.
        return x * (w - (x * w).sum(-1, keepdim=True))
    raise ValueError(param)


def train(target, k, param, steps, seed, batch=512):
    torch.manual_seed(seed)
    net = VelocityNet(k).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    for _ in range(steps):
        x0, x1 = sample_pair(target, batch, k)
        t = torch.rand(batch, device=DEV)
        xt = (1 - t[:, None]) * x0 + t[:, None] * x1   # linear path, simplex coordinates
        u = x1 - x0                                     # conditional target
        loss = (field(net, xt, t, param) - u).square().sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return net.eval()


@torch.no_grad()
def evaluate(net, target, k, param, nfe, n=50_000):
    x = uniform_simplex_sample(n, k)
    h = 1.0 / nfe
    bad_steps = 0
    for i in range(nfe):
        t = torch.full((n,), i * h, device=DEV)
        y = x + h * field(net, x, t, param)
        bad_steps += int((y.min(-1).values < 0).sum())
        x = y.clamp_min(0.0)
        x = x / x.sum(-1, keepdim=True).clamp_min(1e-12)
    q = torch.bincount(x.argmax(-1), minlength=k).float().cpu().numpy()
    q = (q + 1.0) / (q.sum() + k)
    p = target.cpu().numpy(); p = p / p.sum()
    kl = float((p * (np.log(np.maximum(p, 1e-12)) - np.log(q))).sum())
    return kl, bad_steps / (n * nfe)


def main() -> int:
    ks, seeds = (4, 60), (0, 1, 2)
    steps = {4: 4000, 60: 8000}
    t0 = time.time()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("")

    print(f"{'K':>4} {'param':>14} {'seed':>5} | " + " ".join(f"KL@{n:<4}" for n in NFES)
          + " | viol")
    print("-" * 88)
    for k in ks:
        target = make_target(k, seed=0)
        for param in ("unconstrained", "replicator"):
            for seed in seeds:
                net = train(target, k, param, steps[k], seed)
                kls, viol = [], None
                for nfe in NFES:
                    kl, v = evaluate(net, target, k, param, nfe)
                    kls.append(kl)
                    if nfe == 100:
                        viol = v
                row = {"k": k, "param": param, "seed": seed,
                       "kl": dict(zip(map(str, NFES), kls)), "violation_rate_nfe100": viol}
                with OUT.open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
                print(f"{k:>4} {param:>14} {seed:>5} | "
                      + " ".join(f"{x:<7.4f}" for x in kls) + f" | {viol:.4f}")
    print(f"\nwrote {OUT.relative_to(REPO)}  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
