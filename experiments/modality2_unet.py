"""Modality 2: flow matching on MNIST in the box [0,1]^784 with a U-Net backbone, the
transferred vanishing factor, and three external baselines.

ARMS (all matched on architecture, optimiser, schedule, batch, steps, seeds, eval count)

  ours / prior-art parameterisations
    unconstrained   v = net(x,t), sampler clamps                    the default
    mult            v = x(1-x) * net(x,t)                           the transferred mechanism

  external baselines, implemented as sampler/parameterisation variants in this harness
    reflected       reflect at the boundary instead of clamping     Lou & Ermon, ICML 2023
    mirror          flow in an unconstrained dual space via a
                    logit mirror map, push forward by sigmoid       Liu et al., NeurIPS 2023
    dynthresh       Imagen's dynamic thresholding: clamp to the
                    s-th percentile of |x| and rescale              Saharia et al., 2022

These are our implementations of published ideas, not the authors' code; a gap may reflect
our implementation and the paper says so.

Metrics: Frechet distance in a small-classifier feature space at each NFE, plus violation
rate, density and coverage at NFE 100.

Run: python -u experiments/modality2_unet.py --tier smoke
     python -u experiments/modality2_unet.py --tier full --arm mult --seeds 3,4,5,6,7
Writes results/modality2_unet[_<arm>[_s<seeds>]].jsonl (m2fixed_* with --fixed-clf), and the
first 64 samples of each arm's first seed to results/samples_[m2fixed_]<arm>_s<seed>.npy.
"""

from __future__ import annotations

import argparse
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
DEV = torch.device(os.environ.get("DGM_DEVICE") or
                   ("cuda" if torch.cuda.is_available()
                    else "mps" if torch.backends.mps.is_available() else "cpu"))
OUT = REPO / "results" / "modality2_unet.jsonl"
NFES = (10, 50, 100)

TIERS = {
    "smoke": dict(steps=30, n_train=512, n_eval=256, seeds=(0,), ch=16),
    "full":  dict(steps=6000, n_train=20000, n_eval=5000, seeds=(0, 1, 2), ch=48),
}


# ---------------------------------------------------------------- U-Net


def timestep_embedding(t: Tensor, dim: int) -> Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t[:, None] * freqs[None] * 1000.0
    return torch.cat([a.sin(), a.cos()], dim=-1)


class Block(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.t = nn.Linear(tdim, cout)
        self.n1 = nn.GroupNorm(8, cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = torch.nn.functional.silu(self.n1(self.c1(x)))
        h = h + self.t(temb)[:, :, None, None]
        h = torch.nn.functional.silu(self.n2(self.c2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    """Small U-Net, the standard image backbone."""

    def __init__(self, ch=48, tdim=128):
        super().__init__()
        self.tdim = tdim
        self.tmlp = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.inp = nn.Conv2d(1, ch, 3, padding=1)
        self.d1, self.d2 = Block(ch, ch, tdim), Block(ch, ch * 2, tdim)
        self.mid = Block(ch * 2, ch * 2, tdim)
        self.u2, self.u1 = Block(ch * 4, ch, tdim), Block(ch * 2, ch, tdim)
        self.out = nn.Conv2d(ch, 1, 3, padding=1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.pool, self.up = nn.AvgPool2d(2), nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x, t):
        temb = self.tmlp(timestep_embedding(t, self.tdim))
        x = x.view(-1, 1, 28, 28)
        h0 = self.inp(x)
        h1 = self.d1(h0, temb)
        h2 = self.d2(self.pool(h1), temb)
        m = self.mid(self.pool(h2), temb)
        u2 = self.u2(torch.cat([self.up(m), h2], 1), temb)
        u1 = self.u1(torch.cat([self.up(u2), h1], 1), temb)
        return self.out(u1).reshape(x.shape[0], -1)


# ---------------------------------------------------------------- arms


MIRROR_SCALE = 4.0  # logit(x) for x in [0.01,0.99] spans about +-4.6; rescale to ~unit


def to_dual(x: Tensor) -> Tensor:
    """Mirror map: the constrained primal [0,1] -> the unconstrained dual R."""
    return torch.logit(x.clamp(1e-4, 1 - 1e-4)) / MIRROR_SCALE


def from_dual(y: Tensor) -> Tensor:
    return torch.sigmoid(y * MIRROR_SCALE)


def velocity(net, x, t, arm):
    w = net(x, t)
    if arm == "mult":
        # The transferred mechanism: vanishes at both faces of every interval.
        return x * (1 - x) * w
    return w


def step(x, v, h, arm):
    y = x + h * v
    if arm == "reflected":
        # Lou & Ermon: reflect at the boundary rather than projecting onto it. Reflection is
        # measure-preserving where clamping is not.
        y = torch.where(y < 0, -y, y)
        y = torch.where(y > 1, 2.0 - y, y)
        return y.clamp(0.0, 1.0)
    if arm == "dynthresh":
        # Imagen: clamp to the s-th percentile of |x| and rescale, rather than to 1.
        s = torch.quantile(y.abs().flatten(1), 0.995, dim=1).clamp_min(1.0)[:, None]
        return (y.clamp(-s, s) / s).clamp(0.0, 1.0)
    return y.clamp(0.0, 1.0)


# ---------------------------------------------------------------- data / metric


def load(n_train):
    x = np.load(REPO / "data" / "mnist_train.npy").astype(np.float32) / 255.0
    y = np.load(REPO / "data" / "mnist_labels.npy").astype(np.int64)
    x = x * 0.98 + 0.01
    D = 784
    return (torch.tensor(x[:n_train].reshape(-1, D)), torch.tensor(y[:n_train]),
            torch.tensor(x[n_train:n_train + 5000].reshape(-1, D)),
            torch.tensor(y[n_train:n_train + 5000]))


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(128, 10)

    def features(self, x): return self.body(x.view(-1, 1, 28, 28))
    def forward(self, x): return self.head(self.features(x))


def train_classifier(xtr, ytr, xte, yte, steps=2500):
    torch.manual_seed(0)
    m = Classifier().to(DEV); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
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
def feats(m, x, batch=1000):
    return torch.cat([m.features(x[i:i + batch].to(DEV)).cpu()
                      for i in range(0, len(x), batch)])


@torch.no_grad()
def density_coverage(fa: Tensor, fb: Tensor, k: int = 5):
    """Density and coverage (Naeem et al., 2020) -- the diversity/coverage axis that
    section 4.6 asks for alongside quality and efficiency. `fb` is real, `fa` generated.

    Each real point carries a ball whose radius is the distance to its k-th real neighbour.
    Coverage is the fraction of real balls containing at least one generated sample, so it
    penalises a model that collapses onto part of the data; density counts hits per ball and
    so rewards generated mass landing where real mass is. Coverage is the headline because it
    is the one that is robust to outliers.
    """
    # float32 throughout: MPS has no float64 and the A40 runs it at 1/32 rate, while these
    # are only distance comparisons in a 128-d feature space.
    fa, fb = fa.to(DEV).float(), fb.to(DEV).float()
    drr = torch.cdist(fb, fb)
    # kth neighbour excluding self, hence k+1 and the last column.
    radii = drr.topk(k + 1, largest=False).values[:, -1]
    dfr = torch.cdist(fa, fb)              # generated x real
    inside = dfr <= radii[None, :]
    coverage = inside.any(0).float().mean().item()
    density = inside.sum().item() / (k * len(fa))
    return density, coverage


@torch.no_grad()
def frechet(m, a, b):
    fa = feats(m, a).numpy().astype(np.float64)
    fb = feats(m, b).numpy().astype(np.float64)
    mu_a, mu_b = fa.mean(0), fb.mean(0)
    ca, cb = np.cov(fa, rowvar=False), np.cov(fb, rowvar=False)
    s, v = np.linalg.eigh(ca); s = np.clip(s, 0, None)
    half = v @ np.diag(np.sqrt(s)) @ v.T
    e = np.linalg.eigvalsh(half @ cb @ half)
    return float(((mu_a - mu_b) ** 2).sum() + np.trace(ca) + np.trace(cb)
                 - 2 * np.sqrt(np.clip(e, 0, None)).sum())


# ---------------------------------------------------------------- train / sample


def train_fm(data, arm, cfg, seed, batch=128):
    torch.manual_seed(seed)
    net = UNet(cfg["ch"]).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=2e-4)
    data = data.to(DEV)
    # Liu et al.: mirror diffusion does not constrain the field, it changes variables. The
    # whole flow lives in the dual space y = logit(x)/s, where the source is Gaussian and no
    # constraint exists; sigmoid returns it to the box at sampling time. Feasibility is
    # exact for a reason unrelated to the vanishing factor.
    if arm == "mirror":
        data = to_dual(data)
    for _ in range(cfg["steps"]):
        i = torch.randint(0, len(data), (batch,), device=DEV)
        x1 = data[i]
        x0 = (torch.randn(batch, 784, device=DEV) if arm == "mirror"
              else torch.rand(batch, 784, device=DEV))
        t = torch.rand(batch, device=DEV)
        xt = (1 - t[:, None]) * x0 + t[:, None] * x1
        loss = (velocity(net, xt, t, arm) - (x1 - x0)).square().sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return net.eval()


@torch.no_grad()
def sample(net, arm, nfe, n, batch=1000):
    outs, bad, tot = [], 0, 0
    for s0 in range(0, n, batch):
        b = min(batch, n - s0)
        dual = arm == "mirror"
        x = (torch.randn(b, 784, device=DEV) if dual else torch.rand(b, 784, device=DEV))
        h = 1.0 / nfe
        for i in range(nfe):
            t = torch.full((b,), i * h, device=DEV)
            v = velocity(net, x, t, arm)
            if dual:
                # No constraint to violate in the dual space, and none in the primal after
                # sigmoid: mirror is feasible by construction, so its rate is exactly 0.
                x = x + h * v
                tot += b
            else:
                raw = x + h * v
                bad += int(((raw < 0) | (raw > 1)).any(-1).sum()); tot += b
                x = step(x, v, h, arm)
        outs.append((from_dual(x) if dual else x).cpu())
    return torch.cat(outs), bad / tot


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="smoke", choices=list(TIERS))
    ap.add_argument("--arm", default=None)
    # Seeds can be split across processes; the registered full run uses 8 seeds (0-7).
    ap.add_argument("--seeds", default=None,
                    help="comma-separated seed list, overriding the tier default")
    # The protocol fixes ONE feature extractor, but without --fixed-clf every process trains
    # its own (GPU nondeterminism), so paired FDs would mix feature spaces. --train-classifier
    # trains it once and saves it; --fixed-clf loads it and writes m2fixed_* files.
    ap.add_argument("--train-classifier", action="store_true",
                    help="train the shared FD extractor once, save it, and exit")
    ap.add_argument("--fixed-clf", action="store_true",
                    help="load the shared FD extractor instead of training one")
    a = ap.parse_args()
    cfg = dict(TIERS[a.tier])
    if a.seeds:
        cfg["seeds"] = tuple(int(x) for x in a.seeds.split(","))
    t0 = time.time()
    xtr, ytr, xte, yte = load(cfg["n_train"])
    clf_path = REPO / "results" / (("smoke_" if a.tier == "smoke" else "") + "fd_classifier.pt")
    if a.train_classifier:
        if clf_path.exists():
            raise SystemExit(f"{clf_path} exists; the extractor is trained exactly once.")
        with torch.enable_grad():
            clf, acc = train_classifier(xtr, ytr, xte, yte)
        torch.save({"state_dict": {k: v.cpu() for k, v in clf.state_dict().items()},
                    "test_accuracy": acc, "tier": a.tier}, clf_path)
        print(f"saved {clf_path.name}  accuracy {acc*100:.2f}%")
        return 0
    if a.fixed_clf:
        if not clf_path.exists():
            raise SystemExit(f"{clf_path} missing: run --train-classifier once first.")
        blob = torch.load(clf_path, map_location="cpu")
        clf = Classifier(); clf.load_state_dict(blob["state_dict"]); clf = clf.to(DEV).eval()
        acc = blob["test_accuracy"]
    else:
        clf, acc = train_classifier(xtr, ytr, xte, yte)
    print(f"device={DEV} tier={a.tier}  feature extractor accuracy {acc*100:.2f}%\n")

    arms = ["unconstrained", "mult", "reflected", "mirror", "dynthresh"]
    if a.arm:
        arms = [x for x in arms if x == a.arm]
    tag = a.arm if not a.seeds else f"{a.arm}_s{a.seeds.replace(',', '-')}"
    out = OUT if not a.arm else OUT.with_name(f"modality2_unet_{tag}.jsonl")
    if a.fixed_clf:
        out = OUT.with_name(f"m2fixed_{tag if a.arm else 'all'}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Results are append-only. A smoke run gets a name outside the `modality2_unet_*` glob the
    # table code reads; a full run refuses to overwrite results that already exist.
    if a.tier == "smoke":
        out = out.with_name("smoke_" + out.name)
    elif out.exists() and out.stat().st_size > 0:
        raise SystemExit(f"{out} already holds results; move it aside before re-running.")
    out.write_text(json.dumps({"classifier_test_accuracy": acc}) + "\n")

    ref = xte[:cfg["n_eval"]]
    fref = feats(clf, ref)
    print(f"{'arm':>14} {'seed':>5} | {'FD@100':>9} {'FD@10':>9} {'cover':>7} "
          f"{'dens':>7} {'violation':>10}")
    print("-" * 70)
    for arm in arms:
        for seed in cfg["seeds"]:
            net = train_fm(xtr, arm, cfg, seed)
            fds, viol, cov, den = {}, None, None, None
            for nfe in NFES:
                xs, v = sample(net, arm, nfe, cfg["n_eval"])
                fds[str(nfe)] = frechet(clf, xs, ref)
                if nfe == 100:
                    viol = v
                    den, cov = density_coverage(feats(clf, xs), fref)
                    if seed == cfg["seeds"][0]:
                        # A fixed, non-cherry-picked grid: the first 64 samples drawn, saved
                        # before any inspection. The seed is in the name so batches of the
                        # same arm with different seed ranges do not overwrite each other.
                        np.save(out.parent / (f"samples_{'m2fixed_' if a.fixed_clf else ''}"
                                              f"{arm}_s{seed}.npy"),
                                xs[:64].numpy().astype(np.float32))
            with out.open("a") as fh:
                fh.write(json.dumps({"arm": arm, "seed": seed, "fd": fds,
                                     "violation_nfe100": viol, "coverage_nfe100": cov,
                                     "density_nfe100": den}) + "\n")
            print(f"{arm:>14} {seed:>5} | {fds['100']:>9.1f} {fds['10']:>9.1f} "
                  f"{cov:>7.4f} {den:>7.4f} {viol:>10.4f}")
    print(f"\nwrote {out.name}  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
