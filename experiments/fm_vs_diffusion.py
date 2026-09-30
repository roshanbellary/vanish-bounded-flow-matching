"""Section 4.2: continuous flow matching versus diffusion, matched protocol.

The decision rule was fixed in the preregistered protocol before any of this ran:
carry forward the family with lower exact KL at NFE=10 on the synthetic benchmark;
ties (overlapping CIs) broken by the real-data task; a second tie goes to flow matching.

Matched by construction: identical architecture (VelocityNet), parameter count, optimiser,
learning rate, training steps, batch size, seeds, evaluation sample count and NFE grid. The
only difference is the generative process.

  flow matching   linear path in simplex coordinates, x_t = (1-t)x_0 + t x_1, regress
                  u = x_1 - x_0, integrate the ODE with Euler.
  diffusion       VP/cosine Gaussian diffusion on simplex coordinates, epsilon-prediction,
                  deterministic DDIM sampling so the NFE axis means the same thing in both.

SECOND PURPOSE. Our central claim is that the LEARNED field is not tangent to the domain;
measured on flow matching alone, that could be an artifact of the flow-matching objective.
Diffusion's iterate is Gaussian and off-simplex BY DESIGN, so the mid-trajectory state is not
comparable -- but the model's x_0-PREDICTION is: it is the model's estimate of a clean data
point, and clean data points live in the simplex. We therefore measure, for both families, how
often the model's estimate of the clean sample falls outside the simplex. If diffusion violates
too, the finding is about learned generative fields rather than about flow matching.

Run: python -u experiments/fm_vs_diffusion.py
Writes: results/fm_vs_diffusion.jsonl (an existing file is renamed aside with its mtime first).
"""

from __future__ import annotations

import json
import math
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

OUT = REPO / "results" / "fm_vs_diffusion.jsonl"
NFES = (2, 5, 10, 20, 50, 100)


# ---------------------------------------------------------------- diffusion schedule


def alpha_bar(s: Tensor) -> Tensor:
    """Cosine schedule (Nichol & Dhariwal). s=0 is clean data, s=1 is noise."""
    f = torch.cos((s + 0.008) / 1.008 * math.pi / 2).clamp_min(1e-4) ** 2
    f0 = math.cos(0.008 / 1.008 * math.pi / 2) ** 2
    return (f / f0).clamp(1e-5, 1.0)


# ---------------------------------------------------------------- training


def train_fm(target, k, steps, seed, batch=512):
    torch.manual_seed(seed)
    net = VelocityNet(k).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    for _ in range(steps):
        x0, x1 = sample_pair(target, batch, k)
        t = torch.rand(batch, device=DEV)
        xt = (1 - t[:, None]) * x0 + t[:, None] * x1
        loss = (net(xt, t) - (x1 - x0)).square().sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return net.eval()


def train_diff(target, k, steps, seed, batch=512):
    """Identical net, optimiser, steps, batch. Only the process differs."""
    torch.manual_seed(seed)
    net = VelocityNet(k).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    for _ in range(steps):
        _, x1 = sample_pair(target, batch, k)
        s = torch.rand(batch, device=DEV)
        ab = alpha_bar(s)[:, None]
        eps = torch.randn_like(x1)
        xs = ab.sqrt() * x1 + (1 - ab).sqrt() * eps
        loss = (net(xs, s) - eps).square().sum(-1).mean()   # epsilon-prediction
        opt.zero_grad(); loss.backward(); opt.step()
    return net.eval()


# ---------------------------------------------------------------- sampling


@torch.no_grad()
def sample_fm(net, k, nfe, n):
    x = uniform_simplex_sample(n, k)
    h = 1.0 / nfe
    bad = 0
    for i in range(nfe):
        t = torch.full((n,), i * h, device=DEV)
        v = net(x, t)
        # x_0-prediction implied by the flow: extrapolate the velocity to t=1.
        # This is the flow-matching analogue of diffusion's x_0-hat and makes the
        # tangency comparison between families apples-to-apples.
        x1_hat = x + (1.0 - i * h) * v
        bad += int((x1_hat.min(-1).values < 0).sum())
        x = (x + h * v).clamp_min(0.0)
        x = x / x.sum(-1, keepdim=True).clamp_min(1e-12)
    return x, bad / (n * nfe)


@torch.no_grad()
def sample_diff(net, k, nfe, n):
    """Deterministic DDIM so NFE means the same thing as in the ODE sampler."""
    x = torch.randn(n, k, device=DEV)
    ss = torch.linspace(1.0, 0.0, nfe + 1, device=DEV)
    bad = 0
    for i in range(nfe):
        s, s_next = ss[i], ss[i + 1]
        ab, ab_n = alpha_bar(s.view(1)), alpha_bar(s_next.view(1))
        eps = net(x, s.expand(n))
        x1_hat = (x - (1 - ab).sqrt() * eps) / ab.sqrt().clamp_min(1e-4)
        bad += int((x1_hat.min(-1).values < 0).sum())
        x = ab_n.sqrt() * x1_hat + (1 - ab_n).sqrt() * eps
    return x, bad / (n * nfe)


@torch.no_grad()
def kl_of(samples, target, k):
    q = torch.bincount(samples.argmax(-1), minlength=k).float().cpu().numpy()
    q = (q + 1.0) / (q.sum() + k)
    p = target.cpu().numpy(); p = p / p.sum()
    return float((p * (np.log(np.maximum(p, 1e-12)) - np.log(q))).sum())


def main() -> int:
    ks, seeds = (4, 60), (0, 1, 2, 3, 4)
    steps = {4: 4000, 60: 8000}
    n_eval = 50_000
    t0 = time.time()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():                 # results are append-only: keep the old file beside it
        OUT.rename(OUT.with_suffix(f".{int(OUT.stat().st_mtime)}{OUT.suffix}"))
    OUT.write_text("")

    print(f"{'K':>4} {'family':>10} {'seed':>5} | "
          + " ".join(f"KL@{n:<5}" for n in NFES) + "| x0 outside simplex")
    print("-" * 96)
    for k in ks:
        target = make_target(k, seed=0)
        for fam in ("flow_matching", "diffusion"):
            for seed in seeds:
                net = (train_fm if fam == "flow_matching" else train_diff)(
                    target, k, steps[k], seed)
                sampler = sample_fm if fam == "flow_matching" else sample_diff
                kls, viol = {}, None
                for nfe in NFES:
                    xs, v = sampler(net, k, nfe, n_eval)
                    kls[str(nfe)] = kl_of(xs, target, k)
                    if nfe == 100:
                        viol = v
                row = {"k": k, "family": fam, "seed": seed, "kl": kls,
                       "x0_outside_rate_nfe100": viol}
                with OUT.open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
                print(f"{k:>4} {fam:>10} {seed:>5} | "
                      + " ".join(f"{kls[str(n)]:<7.4f}" for n in NFES)
                      + f"| {viol:.4f}")
    print(f"\nwrote {OUT.relative_to(REPO)}  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
