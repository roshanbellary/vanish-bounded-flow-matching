"""Modality 2 transfer check with an MLP backbone: does the tangency failure transfer to
box-constrained images? (preregistered Q4; the U-Net study is `modality2_unet.py`.)

The simplex and the unit box are both bounded domains whose boundary a learned field can
cross, and both are patched ad hoc in practice (simplex projection in Dirichlet FM, dynamic
thresholding in Imagen). The geometries are otherwise unrelated: the simplex is a
(K-1)-dimensional affine slice with Fisher-Rao structure; the box is a full-dimensional
product of intervals.

Arms, matched on everything except the parameterisation of the field:

  unconstrained   v = net(x, t)                 sampler clamps
  hard            v = x * (1 - x) * net(x, t)   vanishes at BOTH faces, so the flow cannot
                                                leave [0,1] in continuous time
  endpoint        v = (sigmoid(net) - x)/(1-t)  box analogue of the simplex endpoint arm
  soft            unconstrained + lambda * outward-velocity penalty

The source is uniform on [0,1]^d, so the whole trajectory starts in the domain and any
excursion is the model's doing.

The FD feature extractor is trained here and its test accuracy is reported alongside every
number it produces (the field's FBD classifiers have 11.2-11.5% test accuracy).

Run: python -u experiments/modality2_images.py   (no flags; overwrites its output)
Writes results/modality2_images.jsonl.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
# DGM_DEVICE overrides the device, e.g. DGM_DEVICE=cpu to run alongside GPU jobs.
DEV = torch.device(os.environ.get("DGM_DEVICE") or
                   ("cuda" if torch.cuda.is_available()
                    else "mps" if torch.backends.mps.is_available() else "cpu"))
OUT = REPO / "results" / "modality2_images.jsonl"
NFES = (2, 5, 10, 20, 50, 100)
D = 784


# ---------------------------------------------------------------- data


def load(n_train=20_000):
    x = np.load(REPO / "data" / "mnist_train.npy").astype(np.float32) / 255.0
    y = np.load(REPO / "data" / "mnist_labels.npy").astype(np.int64)
    # Strictly interior: exact 0 and 1 make the hard parameterisation's x(1-x) factor vanish
    # identically on the data itself, which would be a degenerate target rather than a test.
    x = x * 0.98 + 0.01
    return (torch.tensor(x[:n_train].reshape(-1, D)), torch.tensor(y[:n_train]),
            torch.tensor(x[n_train:n_train + 5000].reshape(-1, D)),
            torch.tensor(y[n_train:n_train + 5000]))


# ---------------------------------------------------------------- feature extractor (metric)


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(128, 10)

    def features(self, x): return self.body(x.view(-1, 1, 28, 28))
    def forward(self, x): return self.head(self.features(x))


def train_classifier(xtr, ytr, xte, yte, steps=2000):
    torch.manual_seed(0)
    m = Classifier().to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    xtr, ytr = xtr.to(DEV), ytr.to(DEV)
    for _ in range(steps):
        i = torch.randint(0, len(xtr), (256,), device=DEV)
        loss = nn.functional.cross_entropy(m(xtr[i]), ytr[i])
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        acc = (m(xte.to(DEV)).argmax(-1) == yte.to(DEV)).float().mean().item()
    return m, acc


@torch.no_grad()
def frechet(m, a: Tensor, b: Tensor) -> float:
    """Frechet distance between Gaussians fitted to classifier features. The FID construction,
    with our own validated extractor in place of Inception."""
    fa = m.features(a.to(DEV)).cpu().numpy().astype(np.float64)
    fb = m.features(b.to(DEV)).cpu().numpy().astype(np.float64)
    mu_a, mu_b = fa.mean(0), fb.mean(0)
    ca, cb = np.cov(fa, rowvar=False), np.cov(fb, rowvar=False)
    # sqrtm(ca @ cb) via eigendecomposition of a symmetrised product
    s, v = np.linalg.eigh(ca)
    s = np.clip(s, 0, None)
    half = v @ np.diag(np.sqrt(s)) @ v.T
    e = np.linalg.eigvalsh(half @ cb @ half)
    return float(((mu_a - mu_b) ** 2).sum() + np.trace(ca) + np.trace(cb)
                 - 2 * np.sqrt(np.clip(e, 0, None)).sum())


# ---------------------------------------------------------------- velocity model


class Net(nn.Module):
    def __init__(self, d=D, h=1024, n_freq=16):
        super().__init__()
        self.n_freq = n_freq
        self.f = nn.Sequential(nn.Linear(d + 2 * n_freq, h), nn.SiLU(),
                               nn.Linear(h, h), nn.SiLU(),
                               nn.Linear(h, h), nn.SiLU(), nn.Linear(h, d))

    def forward(self, x, t):
        fr = torch.arange(self.n_freq, device=t.device, dtype=t.dtype)
        a = t[:, None] * (2.0 ** fr)[None] * math.pi
        return self.f(torch.cat([x, a.sin(), a.cos()], -1))


def velocity(net, x, t, arm):
    w = net(x, t)
    if arm == "hard":
        # x(1-x) vanishes at both faces of the box, the direct analogue of the simplex
        # replicator form's x_i factor vanishing at each face. Constrains the STEP.
        return x * (1 - x) * w
    if arm == "endpoint":
        # Box analogue of the simplex endpoint parameterisation: the net predicts a point
        # INSIDE [0,1]^d via sigmoid and the velocity aims at it, so the implied clean-sample
        # estimate x + (1-t)v is valid by construction.
        return (torch.sigmoid(w) - x) / (1.0 - t[:, None]).clamp_min(1e-3)
    return w


def outward_penalty(v, x):
    """Velocity pushing below 0 where x is small, or above 1 where x is large."""
    return (torch.relu(-v).square() * (1 - x) + torch.relu(v).square() * x).sum(-1)


# ---------------------------------------------------------------- train / sample


def train_fm(data, arm, lam, steps, seed, batch=256):
    torch.manual_seed(seed)
    net = Net().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=2e-4)
    data = data.to(DEV)
    for _ in range(steps):
        i = torch.randint(0, len(data), (batch,), device=DEV)
        x1 = data[i]
        x0 = torch.rand(batch, D, device=DEV)      # uniform on the box
        t = torch.rand(batch, device=DEV)
        xt = (1 - t[:, None]) * x0 + t[:, None] * x1
        v = velocity(net, xt, t, arm)
        loss = (v - (x1 - x0)).square().sum(-1).mean()
        if lam > 0:
            loss = loss + lam * outward_penalty(v, xt).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return net.eval()


@torch.no_grad()
def sample(net, arm, nfe, n=5000):
    x = torch.rand(n, D, device=DEV)
    h = 1.0 / nfe
    step_bad = 0
    x0_bad = 0
    for i in range(nfe):
        t = torch.full((n,), i * h, device=DEV)
        v = velocity(net, x, t, arm)
        x1_hat = x + (1.0 - i * h) * v                     # implied clean-sample estimate
        x0_bad += int(((x1_hat < 0) | (x1_hat > 1)).any(-1).sum())
        y = x + h * v
        step_bad += int(((y < 0) | (y > 1)).any(-1).sum())
        x = y.clamp(0.0, 1.0)
    return x, step_bad / (n * nfe), x0_bad / (n * nfe)


def main() -> int:
    t0 = time.time()
    xtr, ytr, xte, yte = load()
    clf, acc = train_classifier(xtr, ytr, xte, yte)
    print(f"feature extractor test accuracy: {acc*100:.2f}%  "
          f"(vs 11.2-11.5% for the FBD classifiers this field relies on)\n")

    ref = xte[:5000]
    arms = [("unconstrained", 0.0), ("hard", 0.0), ("endpoint", 0.0), ("soft", 0.1)]
    seeds = (0, 1, 2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"classifier_test_accuracy": acc}) + "\n")

    print(f"{'arm':>14} {'lam':>5} {'seed':>5} | {'FD@100':>9} {'FD@10':>9} "
          f"{'step viol':>10} {'x0 outside':>11}")
    print("-" * 74)
    for arm, lam in arms:
        for seed in seeds:
            net = train_fm(xtr, arm, lam, steps=6000, seed=seed)
            fds, sv, ov = {}, None, None
            for nfe in NFES:
                xs, s_, o_ = sample(net, arm, nfe)
                fds[str(nfe)] = frechet(clf, xs.cpu(), ref)
                if nfe == 100:
                    sv, ov = s_, o_
            row = {"arm": arm, "lam": lam, "seed": seed, "fd": fds,
                   "step_violation_nfe100": sv, "x0_outside_nfe100": ov}
            with OUT.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(f"{arm:>14} {lam:>5.2f} {seed:>5} | {fds['100']:>9.3f} {fds['10']:>9.3f} "
                  f"{sv:>10.4f} {ov:>11.4f}")
    print(f"\nwrote {OUT.relative_to(REPO)}  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
