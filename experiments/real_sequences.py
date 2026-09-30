"""Modality 1 main harness: flow matching on the probability simplex for real biological
sequences, with our arms, ablations and the external baselines under one matched protocol.

Three real datasets at length 128, matched on sequence count, architecture, optimiser,
steps, batch and seeds, so alphabet size is not confounded with length, dataset size or task:

    dna_k4       K=4   human genomic windows (Ensembl GRCh38 chr1)
    protein_k20  K=20  reviewed human proteins (UniProt SwissProt)
    codon_k64    K=64  chr1 sequence read as triplets

ARMS (see `velocity`, `path_sample`, `denoiser_path`)
    unconstrained   v = net(x,t); the sampler clamps
    hard            v = x * (w - <x,w>)            replicator form (Boll et al.)
    mult_only       v = x * w                      face-vanishing factor alone
    center_only     v = w - mean(w)                centering alone
    endpoint*       CatFlow endpoint parameterisation and its variants
    external        fisher_sph, gumbel_ce, dirichlet_ce (corrected; reported in the paper)
                    fisher, gumbel, dirichlet (mis-specified; kept for reproducibility)

METRICS, oracle-free by design (the field's headline metrics run on classifiers with
11.2-11.5% test accuracy, so we avoid learned evaluators):
    step_violation   fraction of Euler steps whose state leaves the simplex
    x0_outside       fraction of steps whose implied clean-sample estimate is out of domain
    marginal_kl      KL between generated and reference per-position marginals (exact)
    kmer_r           Pearson correlation of 3-mer frequencies vs reference

The backbone is the dilated 1-D CNN used by Dirichlet FM and Fisher-Flow.

Run: python -u experiments/real_sequences.py --tier smoke   # local, <2 min
     python -u experiments/real_sequences.py --tier full --only dna_k4 --arm mult_only
Writes results/<prefix>_<dataset>[_<arm>][_s<seeds>].jsonl, prefix "endpoint" (or "heldout"
for --tier heldout); smoke runs write to a separate *_smoke file.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import scipy.special
import torch
import torch.nn as nn
from torch import Tensor

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from dgm.runctx import RunContext  # noqa: E402
from dgm.paths import simplex as S  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available()
                   else "mps" if torch.backends.mps.is_available() else "cpu")
OUT = REPO / "results" / "real_sequences.jsonl"
NFES = (10, 50, 100)

TIERS = {
    "smoke": dict(steps=40, n_train=512, n_heldout=256, n_eval=512, seeds=(0,), hidden=64,
                  layers=4),
    "dev":   dict(steps=2000, n_train=4000, n_eval=4000, seeds=(0,), hidden=128, layers=8),
    "full":  dict(steps=8000, n_train=16351, n_eval=4000, seeds=(0, 1, 2, 3, 4, 5, 6, 7),
                  hidden=128, layers=8),
    # "full" scores samples against the TRAINING set. "heldout" is the same config trained on
    # 12,000 sequences and scored against 4,351 held-out ones (protein has 16,351 in all; DNA
    # and codon are matched to it). Metrics against the training set are kept as *_train.
    "heldout": dict(steps=8000, n_train=12000, n_heldout=4351, n_eval=4000,
                    seeds=(0, 1, 2, 3, 4, 5, 6, 7), hidden=128, layers=8),
    # Wider, deeper net to test the "K=20 is under-trained" objection. Batch 512: the model is
    # compute-bound on an A40 (~5900 samples/s), so larger batches buy nothing.
    "gpu":   dict(steps=8000, n_train=16351, n_eval=8000, seeds=(0, 1, 2),
                  hidden=256, layers=12, batch=512),
}


# ---------------------------------------------------------------- model


class DilatedCNN(nn.Module):
    """The backbone this literature uses: stacked dilated 1-D convolutions with residual
    connections and LayerNorm, conditioned on t. Dilation doubles each layer so the receptive
    field covers the sequence without attention."""

    def __init__(self, k: int, hidden: int, layers: int, n_freq: int = 16):
        super().__init__()
        self.n_freq = n_freq
        self.inp = nn.Conv1d(k, hidden, 1)
        self.t_proj = nn.Linear(2 * n_freq, hidden)
        self.blocks = nn.ModuleList()
        for i in range(layers):
            d = 2 ** (i % 5)
            self.blocks.append(nn.ModuleDict({
                "norm": nn.GroupNorm(1, hidden),
                "conv": nn.Conv1d(hidden, hidden, 5, padding=2 * d, dilation=d),
            }))
        self.out = nn.Conv1d(hidden, k, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        # x: (B, L, K) -> conv wants (B, K, L)
        h = self.inp(x.transpose(1, 2))
        fr = torch.arange(self.n_freq, device=t.device, dtype=t.dtype)
        a = t[:, None] * (2.0 ** fr)[None] * math.pi
        h = h + self.t_proj(torch.cat([a.sin(), a.cos()], -1))[:, :, None]
        for b in self.blocks:
            h = h + b["conv"](torch.relu(b["norm"](h)))
        return self.out(h).transpose(1, 2)


# ---------------------------------------------------------------- external baselines
#
# No released checkpoint exists for our datasets, so each method is re-implemented from its
# published formulation as an arm inside this harness -- not a reproduction of its published
# numbers. Architecture, optimiser, schedule, batch, steps, seeds, eval sample count and NFE
# grid are identical across all arms; only the path (and, for DENOISER_ARMS, the objective)
# differs.
#
#   dirichlet   Stark et al., Dirichlet Flow Matching, arXiv:2402.05841
#   fisher      Davis et al., Fisher Flow Matching, arXiv:2405.14664 (NeurIPS 2024)
#   gumbel      Tang et al., Gumbel-Softmax Flow Matching, arXiv:2503.17361
#
# `fisher`, `dirichlet` and `gumbel` are mis-specified (see SPHERE_ARMS, DENOISER_ARMS);
# `fisher_sph`, `dirichlet_ce` and `gumbel_ce` are the corrected arms the paper reports.


def sample_gamma(conc: Tensor, rounds: int = 8) -> Tensor:
    """Marsaglia-Tsang Gamma sampler for shape >= 1, device-independent.

    torch._standard_gamma has no MPS kernel, and routing every training batch through a CPU
    fallback would dominate the runtime. Marsaglia-Tsang is exact (not an approximation) and
    vectorises; acceptance is ~98% per round, so a handful of rounds leaves a negligible
    residue, which we fill with the mean to keep the tensor finite.
    """
    d = conc - 1.0 / 3.0
    c = 1.0 / torch.sqrt(9.0 * d)
    out = torch.zeros_like(conc)
    done = torch.zeros_like(conc, dtype=torch.bool)
    for _ in range(rounds):
        z = torch.randn_like(conc)
        v = (1.0 + c * z).pow(3)
        u = torch.rand_like(conc)
        ok = (v > 0) & (torch.log(u.clamp_min(1e-12))
                        < 0.5 * z * z + d - d * v + d * torch.log(v.clamp_min(1e-12)))
        take = ok & ~done
        out = torch.where(take, d * v, out)
        done = done | take
        if bool(done.all()):
            break
    return torch.where(done, out, conc)          # residue -> the distribution's mean


def path_sample(x0: Tensor, x1: Tensor, t: Tensor, path: str):
    """Return (x_t, conditional target velocity) for the named probability path."""
    tb = t[:, None, None]
    if path == "linear":
        xt = (1 - tb) * x0 + tb * x1
        return xt, x1 - x0

    if path == "fisher":
        # Fisher-Rao geodesic: sqrt map to the sphere, slerp, pull back. u = log_map/(1-t).
        s0, s1 = S.to_sphere(x0), S.to_sphere(x1)
        st = S.slerp(s0, s1, t[:, None])
        u = S.log_map(st, s1) / (1 - tb).clamp_min(1e-3)
        return S.from_sphere(st), S.tangent_project(st, u)

    if path == "gumbel":
        # Temperature-annealed Gumbel-Softmax interpolant, tau(t) = tau_max * exp(-lam t).
        # Conditional field (their Eq. 13): u = (lam/tau) * x_k * (e_k - x).
        tau_max, lam = 10.0, 3.0
        tau = tau_max * torch.exp(-lam * tb)
        logits = torch.log(x1.clamp_min(1e-8)) / tau + torch.log(x0.clamp_min(1e-12))
        xt = torch.softmax(logits, dim=-1)
        xk = (xt * x1).sum(-1, keepdim=True)
        return xt, (lam / tau) * xk * (x1 - xt)

    if path == "dirichlet":
        # Conditional path p_t(x | x_1=e_i) = Dir(1 + t_s * e_i), t_s = t/(1-t) so that
        # t->1 concentrates on the vertex (Stark et al. integrate to a large finite time).
        ts = (tb / (1 - tb).clamp_min(1e-3)).clamp(max=8.0)
        conc = torch.ones_like(x1) + ts * x1
        g = sample_gamma(conc).clamp_min(1e-12)
        xt = g / g.sum(-1, keepdim=True)
        # Their field points at the target vertex, rescaled so it vanishes both AT the vertex
        # and on the opposite face -- the property that keeps mass from draining off the face.
        xi = (xt * x1).sum(-1, keepdim=True)
        c = (1.0 - xi).clamp_min(1e-6) * (ts + 1.0)
        return xt, c * (x1 - xt)

    raise ValueError(path)


BASELINE_PATHS = {"dirichlet", "fisher", "gumbel"}

# Corrected Fisher-Flow arm. `fisher` trains on the sphere -- its target is ds/dt, tangent to
# the sphere at s = sqrt(x) -- but is sampled by the simplex-coordinate Euler sampler, which
# treats that output as dx/dt. `fisher_sph` trains identically and integrates on the sphere
# with the exponential map, as Davis et al. do. `fisher` is kept so its rows stay reproducible.
SPHERE_ARMS = {"fisher_sph": "fisher"}

# Corrected Gumbel-Softmax FM and Dirichlet FM arms. `gumbel` and `dirichlet` regress on
# velocity targets that are not the published methods: the
# Gumbel target is 18.42x too small for its own path (-log 1e-8 = 18.42) and the Dirichlet target
# is neither Stark et al.'s C(x_i, alpha) nor the velocity of the path it samples. Both papers
# train a DENOISER with cross-entropy and sample with the posterior-weighted conditional field;
# these arms do exactly that. The old arms are kept so their rows stay reproducible.
DENOISER_ARMS = {"gumbel_ce", "dirichlet_ce"}
GUMBEL_TAU_MAX, GUMBEL_LAM, GUMBEL_BETA = 10.0, 3.0, 2.0      # Tang et al. Eq. 8-9, Sec. 6
DIRICHLET_ALPHA_SCALE, DIRICHLET_ALPHA_MAX = 2.0, 8.0          # Stark et al. reference defaults
_CGRID = np.linspace(0.0, 1.0, 1000)                           # reference c_factor b-grid
_ALPHA_SPACING = 0.001                                         # reference alpha_spacing
CFACTOR_STATS = {"bad": 0, "tot": 0}                           # non-finite C values, recorded

KNOWN_ARMS = {"unconstrained", "hard", "mult_only", "center_only", "endpoint",
              "endpoint_rw", "endpoint_convex", *BASELINE_PATHS, *SPHERE_ARMS, *DENOISER_ARMS}


def gumbel_tau(t):
    return GUMBEL_TAU_MAX * torch.exp(-GUMBEL_LAM * t)


def denoiser_path(x1: Tensor, t: Tensor, arm: str):
    """Training input for DENOISER_ARMS: (x_t, network time input). The loss is cross-entropy
    of net(x_t, time) against x1, so no velocity target is needed.

    gumbel_ce     x_t = softmax((x1 + g/beta) / tau(t)), g ~ Gumbel(0,1), t ~ U(0,1)
                  (Tang et al., arXiv:2503.17361, Eq. 8-9, Alg. 1)
    dirichlet_ce  alpha = 1 + alpha_scale * Exp(1); x_t ~ Dir(1 + (alpha-1) x1); time input
                  (alpha-1)/(alpha_max-1), so the sampler's alpha = 1 + 7t maps onto t
                  (Stark et al., arXiv:2402.05841, Eq. 14; reference sample_cond_prob_path)
    """
    if arm == "gumbel_ce":
        u = torch.rand_like(x1).clamp(1e-12, 1.0 - 1e-7)
        g = -torch.log(-torch.log(u))
        tau = gumbel_tau(t)[:, None, None]
        return torch.softmax((x1 + g / GUMBEL_BETA) / tau, dim=-1), t
    if arm == "dirichlet_ce":
        e = -torch.rand(t.shape[0], device=x1.device).clamp_min(1e-12).log()
        alpha = 1.0 + DIRICHLET_ALPHA_SCALE * e
        g = sample_gamma(1.0 + (alpha[:, None, None] - 1.0) * x1).clamp_min(1e-12)
        return g / g.sum(-1, keepdim=True), (alpha - 1.0) / (DIRICHLET_ALPHA_MAX - 1.0)
    raise ValueError(arm)


@functools.lru_cache(maxsize=None)
def _betainc_dalpha(k: int, ia: int) -> np.ndarray:
    """Forward-difference d/dalpha of I_b(alpha, K-1) on the b-grid at grid alpha 1 + ia*0.001,
    exactly the row the reference DirichletConditionalFlow looks up by nearest alpha."""
    a = 1.0 + ia * _ALPHA_SPACING
    return (scipy.special.betainc(a + _ALPHA_SPACING, k - 1, _CGRID)
            - scipy.special.betainc(a, k - 1, _CGRID)) / _ALPHA_SPACING


def dirichlet_c(x: Tensor, alpha: float) -> Tensor:
    """Stark et al. C(x_i, alpha) (Eq. 15-17), computed as the reference c_factor, in float64.
    Non-finite values are set to 0 (the reference's allow_nan_cfactor path) and counted."""
    k = x.shape[-1]
    cdev = x.device if x.device.type == "cuda" else torch.device("cpu")   # MPS has no float64
    xd = x.to(cdev).to(torch.float64)
    dI = torch.from_numpy(_betainc_dalpha(k, int(round((alpha - 1.0) / _ALPHA_SPACING)))).to(cdev)
    pos = xd.clamp(0.0, 1.0) * (len(_CGRID) - 1)
    i0 = pos.floor().long().clamp(0, len(_CGRID) - 2)
    fr = pos - i0
    interp = -(dI[i0] * (1.0 - fr) + dI[i0 + 1] * fr)
    bfun = float(scipy.special.beta(alpha, k - 1))
    out2 = torch.where(xd < 1, bfun / (1.0 - xd) ** (k - 1), torch.zeros_like(xd))
    xa = xd ** (alpha - 1.0)
    out = torch.where(xa > 0, out2 / xa, torch.zeros_like(xd))
    c = interp * out
    bad = ~torch.isfinite(c)
    CFACTOR_STATS["bad"] += int(bad.sum()); CFACTOR_STATS["tot"] += c.numel()
    return torch.where(bad, torch.zeros_like(c), c).to(x.device, x.dtype)


def denoiser_velocity(net, x: Tensor, t: Tensor, arm: str) -> Tensor:
    """dx/dt in harness time t in [0,1] for DENOISER_ARMS: the posterior-weighted conditional
    field, sum_k p(k|x,t) u(x | x1 = e_k).

    gumbel_ce     u = (lam/tau) x_k (e_k - x)            (Tang et al. Eq. 12-13), so
                  v = (lam/tau) (p*x - x <p,x>)
    dirichlet_ce  u = C(x_k, alpha) (e_k - x), alpha = 1 + 7t, d alpha/dt = 7  (Stark Eq. 10, 15)
                  v = 7 (p*C - x <p,C>)
    """
    p = torch.softmax(net(x, t), dim=-1)
    tt = float(t[0])
    if arm == "gumbel_ce":
        px = p * x
        return (GUMBEL_LAM / (GUMBEL_TAU_MAX * math.exp(-GUMBEL_LAM * tt))) * (
            px - x * px.sum(-1, keepdim=True))
    if arm == "dirichlet_ce":
        alpha = 1.0 + (DIRICHLET_ALPHA_MAX - 1.0) * tt
        pc = p * dirichlet_c(x, alpha)
        return (DIRICHLET_ALPHA_MAX - 1.0) * (pc - x * pc.sum(-1, keepdim=True))
    raise ValueError(arm)


def velocity(net, x, t, arm):
    if arm in DENOISER_ARMS:
        return denoiser_velocity(net, x, t, arm)
    arm, _ = split_arm(arm)
    w = net(x, t)
    if arm == "hard":
        # Replicator form (Boll et al., arXiv:2406.04527): constrains the STEP, but leaves the
        # implied ENDPOINT out of domain ~100% of the time. Its two pieces are decomposed by
        # the two arms below.
        return x * (w - (x * w).sum(-1, keepdim=True))
    if arm == "mult_only":
        # Piece 1 alone: the multiplicative x_i factor that makes the field vanish on each
        # face. Constraint WITHOUT the centering, so sum_i v_i = <x,w> != 0 and the update
        # drifts off the affine hull.
        return x * w
    if arm == "center_only":
        # Piece 2 alone: the centering that gives sum_i v_i = 0. Affine-hull preserving
        # WITHOUT the face-vanishing factor, so it does not constrain the simplex at all.
        # A gain here would mean a projection effect, not a constraint effect.
        return w - w.mean(-1, keepdim=True)
    if arm == "endpoint":
        # Endpoint parameterisation (CatFlow, Eijkelboom et al. NeurIPS 2024, arXiv:2406.04843):
        # the net predicts a point IN the simplex and the velocity aims at it, so the implied
        # clean-sample estimate x + (1-t)v = p is valid by construction. Not claimed as ours;
        # tests whether ENDPOINT validity, as opposed to step validity, drives quality at large K.
        p = torch.softmax(w, dim=-1)
        return (p - x) / (1.0 - t[:, None, None]).clamp_min(1e-3)
    return w


def outward_penalty(v, x):
    return (torch.relu(-v).square() * (1 - x)).sum(-1).mean()


def project_simplex(y: Tensor) -> Tensor:
    """Euclidean projection onto the simplex (Wang & Carreira-Perpinan 2013), batched."""
    k = y.shape[-1]
    u, _ = torch.sort(y, dim=-1, descending=True)
    css = u.cumsum(-1)
    idx = torch.arange(1, k + 1, device=y.device, dtype=y.dtype)
    cond = u - (css - 1.0) / idx > 0
    rho = cond.to(y.dtype).cumsum(-1).argmax(-1, keepdim=True)
    theta = (css.gather(-1, rho) - 1.0) / (rho.to(y.dtype) + 1.0)
    return (y - theta).clamp_min(0.0)


def integrate_step(x: Tensor, v: Tensor, t: float, h: float, integrator: str) -> Tensor:
    """One sampler step.

    euler   x + h*v, then clamp+renormalise. The field's implied endpoint is discarded.

    convex  step toward the IMPLIED ENDPOINT by lambda = min(h/(1-t), 1):
                x <- (1 - lambda) x + lambda * proj(x + (1-t) v)
            Since both arguments are simplex points and lambda in [0,1], the result is a
            convex combination of simplex points and therefore exactly in the simplex at ANY
            step size.

    Why this matters. For the endpoint parameterisation v = (p - x)/(1-t), an Euler step is
        x + h v = (1 - h/(1-t)) x + (h/(1-t)) p,
    a convex combination ONLY while h <= 1-t. With uniform steps h = 1/N, once t > 1 - 1/N
    the coefficient exceeds 1 and the update extrapolates PAST p, straight out of the domain.
    The parameterisation guarantees a valid endpoint and the integrator throws it away in the
    final steps.
    """
    if integrator == "euler":
        y = x + h * v
    else:
        lam = min(h / max(1.0 - t, 1e-6), 1.0)
        p = project_simplex(x + (1.0 - t) * v)
        y = (1.0 - lam) * x + lam * p
    y = y.clamp_min(0.0)
    return y / y.sum(-1, keepdim=True).clamp_min(1e-12)


def split_arm(arm: str) -> tuple[str, str]:
    """'endpoint_convex' -> ('endpoint','convex'); 'endpoint_rw' -> ('endpoint','euler')."""
    if arm.endswith("_convex"):
        return arm[:-7], "convex"
    if arm.endswith("_rw"):
        return arm[:-3], "euler"
    return arm, "euler"


def arm_path(arm: str) -> str:
    """Baseline arms carry their own probability path; ours all use the linear path."""
    if arm in SPHERE_ARMS:
        return SPHERE_ARMS[arm]
    return arm if arm in BASELINE_PATHS else "linear"


def sphere_step(net, s: Tensor, t: Tensor, h: float):
    """One Fisher-Flow sampler step on the sphere. Returns (s_next, v, x).

    x = s^2 is the simplex point the network was trained on; its output is regressed onto a
    sphere-tangent target, so it is projected to the tangent space at s and followed along the
    great circle with the exponential map. s^2 of a unit vector is on the simplex exactly.
    """
    x = S.from_sphere(s)
    v = S.tangent_project(s, net(x, t))
    return S.exp_map(s, h * v), v, x


# ---------------------------------------------------------------- data / metrics


def load(name: str, k: int, n: int, seed: int = 0):
    arr = np.load(REPO / "data" / f"{name}.npy").astype(np.int64)
    rng = np.random.default_rng(seed)
    arr = arr[rng.permutation(len(arr))[:n]]
    return torch.tensor(arr)


def load_split(name: str, n_train: int, n_heldout: int, seed: int = 0):
    """Disjoint (train, held-out). With data v2 the split is fixed by `prepare_real_data_v2.py`
    -- by protein for protein, by 100 kb genomic block for DNA and triplets, deduplicated -- and
    here each pool is permuted (seed 0) and truncated to the registered sizes. Without the v2
    files, falls back to the v1 single-permutation split."""
    v2 = REPO / "data" / "v2"
    if (v2 / f"{name}_train.npy").exists():
        rng = np.random.default_rng(seed)
        tr = np.load(v2 / f"{name}_train.npy").astype(np.int64)
        ho = np.load(v2 / f"{name}_heldout.npy").astype(np.int64)
        if n_train > len(tr) or n_heldout > len(ho):
            raise ValueError(f"{name}: pools {len(tr)}/{len(ho)} < {n_train}/{n_heldout}")
        return (torch.tensor(tr[rng.permutation(len(tr))[:n_train]]),
                torch.tensor(ho[rng.permutation(len(ho))[:n_heldout]]))
    arr = np.load(REPO / "data" / f"{name}.npy").astype(np.int64)
    perm = np.random.default_rng(seed).permutation(len(arr))
    if n_train + n_heldout > len(arr):
        raise ValueError(f"{name}: {len(arr)} sequences < {n_train} + {n_heldout}")
    return (torch.tensor(arr[perm[:n_train]]),
            torch.tensor(arr[perm[n_train:n_train + n_heldout]]))


def score(gen: np.ndarray, ref_m: np.ndarray, ref_f: np.ndarray, k: int) -> tuple[float, float]:
    """(marginal KL, 3-mer r) of decoded samples against a reference's statistics."""
    gm, gf = marginals(gen, k), kmer_freq(gen, k)
    return (float((ref_m * (np.log(ref_m) - np.log(gm))).sum(1).mean()),
            float(np.corrcoef(ref_f, gf)[0, 1]))


def onehot(idx: Tensor, k: int) -> Tensor:
    return torch.nn.functional.one_hot(idx, k).float()


def marginals(idx: np.ndarray, k: int) -> np.ndarray:
    """Per-position symbol frequencies, (L, K)."""
    L = idx.shape[1]
    m = np.zeros((L, k))
    for j in range(L):
        m[j] = np.bincount(idx[:, j], minlength=k)
    return (m + 1.0) / (m.sum(1, keepdims=True) + k)


def kmer_freq(idx: np.ndarray, k: int, r: int = 3) -> np.ndarray:
    """3-mer frequency vector; oracle-free and standard in the DNA design literature."""
    codes = np.zeros(len(idx) * (idx.shape[1] - r + 1), dtype=np.int64)
    p = 0
    for j in range(idx.shape[1] - r + 1):
        c = np.zeros(len(idx), dtype=np.int64)
        for o in range(r):
            c = c * k + idx[:, j + o]
        codes[p:p + len(idx)] = c
        p += len(idx)
    f = np.bincount(codes, minlength=k ** r).astype(float)
    return f / f.sum()


# ---------------------------------------------------------------- train / sample


def train(data_idx, k, arm, lam, cfg, seed):
    torch.manual_seed(seed)
    net = DilatedCNN(k, cfg["hidden"], cfg["layers"]).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    x1_all = onehot(data_idx, k).to(DEV)
    bs = min(cfg.get("batch", 128), len(x1_all))
    for _ in range(cfg["steps"]):
        i = torch.randint(0, len(x1_all), (bs,), device=DEV)
        x1 = x1_all[i]
        e = -torch.rand(bs, x1.shape[1], k, device=DEV).clamp_min(1e-12).log()
        x0 = e / e.sum(-1, keepdim=True)              # uniform on the simplex, per position
        t = torch.rand(bs, device=DEV)
        if arm in DENOISER_ARMS:
            # Published objective for these methods: cross-entropy denoiser on x1.
            xt, tn = denoiser_path(x1, t, arm)
            logits = net(xt, tn)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, k), x1.argmax(-1).reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            continue
        xt, target = path_sample(x0, x1, t, arm_path(arm))
        v = velocity(net, xt, t, arm)
        per = (v - target).square().sum(-1)
        if arm.startswith("endpoint_rw"):
            # The endpoint parameterisation v = (p-x)/(1-t) amplifies endpoint error by
            # 1/(1-t)^2, so the plain FM loss concentrates weight on t->1, after the decoded
            # output is already fixed (commitment_time). Multiplying by (1-t)^2 removes exactly
            # that amplification, recovering the objective the unconstrained arm optimises.
            per = per * (1.0 - t).square()
        loss = per.mean()
        if lam > 0:
            loss = loss + lam * outward_penalty(v, xt)
        opt.zero_grad(); loss.backward(); opt.step()
    return net.eval()


@torch.no_grad()
def commitment_time(net, k, L, arm, nfe=100, n=2000):
    """Mean fraction of the trajectory after which a position's decoded symbol never changes.

    For the unconstrained arm this is ~0.05 at every K: the model commits almost immediately,
    from a state that is still essentially noise. An arm that delays commitment keeps options
    open longer.
    """
    e = -torch.rand(n, L, k, device=DEV).clamp_min(1e-12).log()
    x = e / e.sum(-1, keepdim=True)
    h = 1.0 / nfe
    dec = []
    _, integ = split_arm(arm)
    s = S.to_sphere(x) if arm in SPHERE_ARMS else None
    for i in range(nfe):
        tt = i * h
        t = torch.full((n,), tt, device=DEV)
        if arm in SPHERE_ARMS:
            s, _, _ = sphere_step(net, s, t, h)
            x = S.from_sphere(s)
        else:
            x = integrate_step(x, velocity(net, x, t, arm), tt, h, integ)
        dec.append(x.argmax(-1).cpu().numpy())
    dec = np.stack(dec)
    differs = dec != dec[-1][None]
    last = np.where(differs, np.arange(nfe)[:, None, None], -1).max(0)
    return float(((last + 1) / nfe).mean())


@torch.no_grad()
def sample(net, k, L, arm, nfe, n, batch=4000, posterior=False):
    """Decoded samples (argmax of the final state), step-violation and x0-outside rates.

    posterior=True (DENOISER_ARMS only) also returns the argmax of the network's final
    posterior p(x1 | x_1, t=1). Both Dirichlet FM and Gumbel-Softmax FM decode this way: their
    paths never reach a vertex (Dir(1+7e_k) and softmax(2 x1 + g) at the end), so argmax of the
    STATE is capped below the data by construction.
    """
    outs, step_bad, x0_bad, tot = [], 0, 0, 0
    posts = []
    for s in range(0, n, batch):
        b = min(batch, n - s)
        e = -torch.rand(b, L, k, device=DEV).clamp_min(1e-12).log()
        x = e / e.sum(-1, keepdim=True)
        h = 1.0 / nfe
        _, integ = split_arm(arm)
        for i in range(nfe):
            tt = i * h
            t = torch.full((b,), tt, device=DEV)
            v = velocity(net, x, t, arm)
            x1_hat = x + (1.0 - tt) * v
            x0_bad += int((x1_hat.min(-1).values < 0).any(-1).sum())
            # step violation is measured on the RAW Euler update in both integrators, so the
            # number means the same thing across arms: "would the naive step have exited?"
            step_bad += int(((x + h * v).min(-1).values < 0).any(-1).sum())
            x = integrate_step(x, v, tt, h, integ)
            tot += b
        outs.append(x.argmax(-1).cpu().numpy())
        if posterior:
            posts.append(net(x, torch.ones(b, device=DEV)).argmax(-1).cpu().numpy())
    if posterior:
        return np.concatenate(outs), step_bad / tot, x0_bad / tot, np.concatenate(posts)
    return np.concatenate(outs), step_bad / tot, x0_bad / tot


@torch.no_grad()
def sample_sphere(net, k, L, nfe, n, batch=4000):
    """Sampler for SPHERE_ARMS. Same source and NFE grid as `sample`, integrated on the sphere.

    Returns (decoded, metrics) where metrics are per-step rates over all (sample, step) pairs:
      step_violation         the method's own update, x = s'^2 after the exp-map step, leaves
                             the simplex (0 by construction; computed as a check)
      step_violation_euclid  a Euclidean Euler step x + h*dx/dt with the implied simplex
                             velocity dx/dt = 2 s*v would leave it -- comparable with the other
                             arms' step_violation
      x0_outside             the method's own clean-sample estimate exp_map(s,(1-t)v)^2 leaves
                             the simplex (0 by construction)
      x0_outside_euclid      the Euclidean extrapolation x + (1-t) 2 s*v leaves it
    """
    outs = []
    cnt = {"step_violation": 0, "step_violation_euclid": 0,
           "x0_outside": 0, "x0_outside_euclid": 0}
    tot = 0

    def bad(y):
        return int((y.min(-1).values < 0).any(-1).sum())

    for st in range(0, n, batch):
        b = min(batch, n - st)
        e = -torch.rand(b, L, k, device=DEV).clamp_min(1e-12).log()
        x = e / e.sum(-1, keepdim=True)
        s = S.to_sphere(x)
        h = 1.0 / nfe
        for i in range(nfe):
            tt = i * h
            t = torch.full((b,), tt, device=DEV)
            s_next, v, x = sphere_step(net, s, t, h)
            xdot = 2.0 * s * v                       # implied simplex velocity d(s^2)/dt
            cnt["x0_outside"] += bad(S.exp_map(s, (1.0 - tt) * v).square())
            cnt["x0_outside_euclid"] += bad(x + (1.0 - tt) * xdot)
            cnt["step_violation_euclid"] += bad(x + h * xdot)
            cnt["step_violation"] += bad(s_next.square())
            s = s_next
            tot += b
        outs.append(S.from_sphere(s).argmax(-1).cpu().numpy())
    return np.concatenate(outs), {kk: vv / tot for kk, vv in cnt.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="smoke", choices=list(TIERS))
    # Datasets (and, with --arm, arms) are independent, so they can run as concurrent
    # processes on one GPU; one process leaves an A40 mostly idle.
    ap.add_argument("--only", default=None, help="restrict to one dataset name")
    ap.add_argument("--arm", default=None, help="restrict to one arm")
    # Seed subset, e.g. --seeds 0,1,2,3, so one (dataset, arm) can be split across processes.
    # Each subset writes its own file (suffix _s0-1-2-3); cfg["seeds"] itself is not changed,
    # so a cell's run_id does not depend on how the seeds were split.
    ap.add_argument("--seeds", default=None, help="comma-separated seed subset")
    # A rerun (e.g. checkpoint-saving, for real-data guidance) writes under its own prefix, so
    # its rows can never be read as, or deduplicated into, the registered rows.
    ap.add_argument("--out-prefix", default=None, help="override the result-file prefix")
    ap.add_argument("--save-ckpt", action="store_true", help="save each trained network")
    a = ap.parse_args()                     # parse BEFORE any file is touched
    if a.arm is not None and a.arm not in KNOWN_ARMS:
        ap.error(f"unknown arm {a.arm!r}")
    cfg = TIERS[a.tier]
    seeds = tuple(cfg["seeds"])
    if a.seeds:
        seeds = tuple(int(x) for x in a.seeds.split(","))
        if not set(seeds) <= set(cfg["seeds"]):
            ap.error(f"--seeds {seeds} not a subset of the tier's seeds {cfg['seeds']}")
    t0 = time.time()
    global OUT
    prefix = a.out_prefix or ("heldout" if a.tier == "heldout" else "endpoint")
    if a.only:                      # one file per concurrent process; no interleaved writes
        tag = f"{a.only}" + (f"_{a.arm}" if a.arm else "")
        if a.seeds:
            tag += "_s" + "-".join(map(str, seeds))
        if a.tier == "smoke":
            # A smoke run truncates its output; give it its own file so it can never
            # truncate a full-tier result file of the same (dataset, arm).
            tag += "_smoke"
        OUT = REPO / "results" / f"{prefix}_{tag}.jsonl"
    elif a.tier == "smoke":
        OUT = REPO / "results" / "real_sequences_smoke.jsonl"   # never truncate the full file
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if a.tier == "smoke":
        OUT.write_text("")

    # Three alphabets at matched length and sample count. The codon set is chromosome-1
    # sequence read three bases at a time (384-base windows); its build script is lost, so we
    # cannot confirm its windows coincide with the DNA set's.
    datasets = [("dna_k4", 4), ("protein_k20", 20), ("codon_k64", 64)]
    if a.only:
        datasets = [d for d in datasets if d[0] == a.only]
    # Default: the two decomposition arms; --arm selects any other.
    arms = [("mult_only", 0.0), ("center_only", 0.0)]
    if a.arm:
        arms = [(a.arm, 0.0)]

    print(f"device={DEV}  tier={a.tier}  cfg={cfg}\n")
    print(f"{'dataset':>12} {'K':>3} {'arm':>14} {'seed':>4} | {'step viol':>10} "
          f"{'x0 out':>8} {'marg KL':>9} {'3mer r':>8}")
    print("-" * 78)

    for name, k in datasets:
        if "n_heldout" in cfg:
            idx, ho = load_split(name, cfg["n_train"], cfg["n_heldout"])
            ref = ho.numpy()                          # primary: the held-out reference
            tr_m, tr_f = marginals(idx.numpy(), k), kmer_freq(idx.numpy(), k)
        else:
            idx = load(name, k, cfg["n_train"])
            ref = idx.numpy()
        ref_m, ref_f = marginals(ref, k), kmer_freq(ref, k)
        L = idx.shape[1]
        for arm, lam in arms:
            for seed in seeds:
                # "gpu" and "heldout" are full-tier runs; device is recorded separately.
                ledger_tier = "full" if a.tier in ("gpu", "heldout") else a.tier
                ctx = RunContext.create(
                    {"exp": "real_sequences", "dataset": name, "k": k, "arm": arm,
                     "lam": lam, "tier": a.tier, **cfg}, seed, ledger_tier)
                if ctx.already_done():
                    print(f"  skip {ctx.run_id} (already in ledger)"); continue
                net = train(idx, k, arm, lam, cfg, seed)
                if a.save_ckpt and a.tier != "smoke":
                    cd = REPO / "results" / "checkpoints"
                    cd.mkdir(parents=True, exist_ok=True)
                    dv = "v2" if (REPO / "data" / "v2" / f"{name}_train.npy").exists() else "v1"
                    torch.save({k_: v.cpu() for k_, v in net.state_dict().items()},
                               cd / f"{name}_{arm}_s{seed}_{a.tier}_{dv}_{prefix}.pt")
                res, saved = {}, {}
                for nfe in NFES:
                    if arm in SPHERE_ARMS:
                        gen, viol = sample_sphere(net, k, L, nfe, cfg["n_eval"])
                    else:
                        CFACTOR_STATS.update(bad=0, tot=0)
                        if arm in DENOISER_ARMS:
                            gen, sv, ov, post = sample(net, k, L, arm, nfe, cfg["n_eval"],
                                                       posterior=True)
                        else:
                            gen, sv, ov = sample(net, k, L, arm, nfe, cfg["n_eval"])
                        viol = {"step_violation": sv, "x0_outside": ov}
                        if arm == "dirichlet_ce":
                            viol["cfactor_nonfinite"] = (CFACTOR_STATS["bad"]
                                                         / max(CFACTOR_STATS["tot"], 1))
                    mkl, r = score(gen, ref_m, ref_f, k)
                    res[str(nfe)] = {**viol, "marginal_kl": mkl, "kmer_r": r}
                    saved[f"nfe{nfe}"] = gen.astype(np.uint8)
                    if arm in DENOISER_ARMS:
                        # The method's own decoding; the state-argmax columns stay beside it.
                        res[str(nfe)]["marginal_kl_post"], res[str(nfe)]["kmer_r_post"] = \
                            score(post, ref_m, ref_f, k)
                        saved[f"nfe{nfe}_post"] = post.astype(np.uint8)
                    if "n_heldout" in cfg:
                        res[str(nfe)]["marginal_kl_train"], res[str(nfe)]["kmer_r_train"] = \
                            score(gen, tr_m, tr_f, k)
                if "n_heldout" in cfg and a.tier != "smoke":
                    sd = REPO / "results" / "m1_samples"
                    sd.mkdir(parents=True, exist_ok=True)
                    dv = "v2" if (REPO / "data" / "v2" / f"{name}_train.npy").exists() else "v1"
                    np.savez_compressed(sd / f"{name}_{arm}_s{seed}_{a.tier}_{dv}"
                                             f"{'_' + a.out_prefix if a.out_prefix else ''}_all.npz",
                                        **saved)
                row = {"dataset": name, "k": k, "arm": arm, "lam": lam, "seed": seed,
                       "tier": a.tier, "per_nfe": res,
                       "data": "v2" if (REPO / "data" / "v2" / f"{name}_train.npy").exists()
                       else "v1",
                       "commitment_t": commitment_time(net, k, L, arm)}
                with OUT.open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
                ctx.finish(res, on_gpu=(DEV.type == "cuda"), device=str(DEV))
                m = res["100"]
                print(f"{name:>12} {k:>3} {arm:>14} {seed:>4} | {m['step_violation']:>10.4f} "
                      f"{m['x0_outside']:>8.4f} {m['marginal_kl']:>9.4f} {m['kmer_r']:>8.4f}")

    print(f"\nwrote {OUT.relative_to(REPO)}  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
