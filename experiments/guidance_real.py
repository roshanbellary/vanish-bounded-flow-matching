"""Reward guidance on the REAL Modality 1 models (protocol amendment 2026-09-29, real-data guidance).

`guidance_modality1.py` uses a synthetic K=8 single-position target, where the ideal guided law
is known exactly; the course instructions ask for guidance demonstrated on Modality 1 itself.
This script steers the frozen flow-matching checkpoints trained on data v2 (held-out split)
toward a sequence property, using the guidance rules of the synthetic experiment:

    DNA      raise GC content          S = {C, G}
    PROTEIN  raise hydrophobic fraction S = {A, I, L, M, F, V, W}

Reward, per position, the same form as the synthetic one: r(x) = sum_l log(sum_{i in S} x_{l,i}),
so grad r = 1{i in S} / sum_{j in S} x_{l,j} position by position. Guided drift
v_theta + gamma * G(x) grad r, gamma constant in t; G is the identity (Euclidean, then simplex
projection), the centering operator (tangent) or Diag(x) - x x^T (natural gradient); the registered
amendment runs Euclidean only (RULES). Nothing is retrained; only sampling changes.

Metrics, per (dataset, seed, rule, gamma), 4,000 samples at NFE 100:
    target        mean fraction of positions in S; share of sequences above the held-out 90th
                  percentile of that fraction (so the data themselves hit ~10%)
    quality       3-mer r against the held-out set, and against the held-out top decile (a
                  real-data stand-in for the ideal guided law)
    diversity     unique fraction, mean pairwise normalised Hamming (20,000 pairs)
    cost          step-violation rate (raw guided Euler update), sampling seconds

Run: python -u experiments/guidance_real.py --tier smoke|full [--only dna_k4] [--seeds 0,1]
Writes (append): results/[smoke_]guidance_real_<dataset|all>[_s<seeds>].jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))
from real_sequences import DEV, TIERS, DilatedCNN, kmer_freq, load_split, velocity  # noqa: E402
from guidance_modality1 import guided_velocity, project_simplex  # noqa: E402

RULES = ("euclidean",)          # registered amendment: Euclidean only
GAMMAS = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)      # log-spaced; fixed in the preregistration
PROT_ALPHA = "ACDEFGHIKLMNPQRSTVWY"
# (K, symbols in S). The "hit" threshold is the held-out 90th percentile of the S-fraction, so an
# unguided model that matches the data hits about 10%.
TARGETS = {
    "dna_k4": (4, [1, 2]),                                        # ACGT: C=1, G=2
    "protein_k20": (20, [PROT_ALPHA.index(c) for c in "AILMFVW"]),
}
N, NFE, PAIRS = 4000, 100, 20_000


def reward_grad(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-position gradient of sum_l log(sum_{i in S} x_{l,i})."""
    return mask / (x * mask).sum(-1, keepdim=True).clamp_min(1e-8)


@torch.no_grad()
def sample(net, k, L, mask, rule, gamma, n=N, nfe=NFE, seed=0, arm="mult_only"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    e = -torch.rand(n, L, k, generator=g).clamp_min(1e-12).log().to(DEV)
    x = e / e.sum(-1, keepdim=True)                      # Dir(1) source, as in training
    h, bad, tot = 1.0 / nfe, 0, 0
    for i in range(nfe):
        t = torch.full((n,), i * h, device=DEV)
        # The learned field through the harness's own parameterisation (x * w for mult_only).
        v = guided_velocity(rule, velocity(net, x, t, arm), x, reward_grad(x, mask), gamma)
        y = x + h * v
        bad += int((y.min(-1).values < 0).any(-1).sum()); tot += n
        # Euclidean projection for every rule, as in the synthetic §4.3 experiment; for the
        # natural rule it is a no-op whenever the update stays inside.
        x = project_simplex(y)
        x = x / x.sum(-1, keepdim=True).clamp_min(1e-12)
    return x.argmax(-1).cpu().numpy().astype(np.uint8), bad / tot


def metrics(gen, S, thr, ref_f, ref_hi_f, k, seed):
    frac = np.isin(gen, S).mean(1)
    r = lambda f: float(np.corrcoef(f, kmer_freq(gen.astype(np.int64), k))[0, 1])
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(gen), PAIRS); j = (i + rng.integers(1, len(gen), PAIRS)) % len(gen)
    return {"target_mean": float(frac.mean()), "target_hit": float((frac > thr).mean()),
            "kmer_r_heldout": r(ref_f), "kmer_r_heldout_hi": r(ref_hi_f),
            "unique": len({row.tobytes() for row in gen}) / len(gen),
            "hamming": float((gen[i] != gen[j]).mean())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="smoke", choices=("smoke", "full"))
    ap.add_argument("--only", default=None, choices=list(TARGETS))
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--arm", default="mult_only", help="which frozen model to guide")
    ap.add_argument("--ckpt-pattern", default="{name}_{arm}_s{seed}_heldout_v2_ckpt_heldout.pt",
                    help="checkpoint file name under results/checkpoints")
    a = ap.parse_args()
    cfg = TIERS["heldout"]
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else list(range(8))
    n = 256 if a.tier == "smoke" else N
    tag = (a.only or "all") + ("_s" + a.seeds.replace(",", "-") if a.seeds else "")
    out = REPO / "results" / f"{'smoke_' if a.tier == 'smoke' else ''}guidance_real_{tag}.jsonl"
    for name, (k, S) in TARGETS.items():
        if a.only and name != a.only:
            continue
        tr, ho = load_split(name, cfg["n_train"], cfg["n_heldout"])
        ho = ho.numpy()
        ho_frac = np.isin(ho, S).mean(1)
        thr = float(np.quantile(ho_frac, 0.9))
        ref_f, ref_hi_f = kmer_freq(ho, k), kmer_freq(ho[ho_frac > thr], k)
        mask = torch.zeros(k, device=DEV); mask[S] = 1.0
        for seed in seeds:
            ck = REPO / "results" / "checkpoints" / a.ckpt_pattern.format(name=name, arm=a.arm, seed=seed)
            sd = torch.load(ck, map_location="cpu")
            hidden = sd["inp.weight"].shape[0]
            layers = sum(1 for key in sd if key.endswith(".conv.weight"))
            net = DilatedCNN(k, hidden, layers).to(DEV); net.load_state_dict(sd); net.eval()
            for rule in RULES:
                for gamma in GAMMAS:
                    if gamma == 0.0 and rule != RULES[0]:
                        continue                          # unguided is rule-independent
                    t0 = time.time()
                    gen, viol = sample(net, k, tr.shape[1], mask, rule, gamma, n=n, seed=seed,
                                       arm=a.arm)
                    row = {"dataset": name, "seed": seed, "arm": a.arm,
                           "rule": rule if gamma else "unguided",
                           "gamma": gamma, "tier": a.tier, "step_violation": viol,
                           "sample_seconds": round(time.time() - t0, 2),
                           **metrics(gen, S, thr, ref_f, ref_hi_f, k, seed),
                           "heldout_target_mean": float(ho_frac.mean()),
                           "heldout_target_hit": float((ho_frac > thr).mean()), "threshold": thr}
                    with out.open("a") as fh:
                        fh.write(json.dumps(row) + "\n")
                    print(f"{name} s{seed} {row['rule']:>9} g={gamma:<4} target {row['target_mean']:.3f} "
                          f"hit {row['target_hit']:.3f} r_ho {row['kmer_r_heldout']:.3f} "
                          f"r_hi {row['kmer_r_heldout_hi']:.3f} ham {row['hamming']:.3f} "
                          f"viol {viol:.3f} ({row['sample_seconds']}s)", flush=True)
    print(f"wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
