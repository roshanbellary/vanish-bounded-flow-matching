"""Modality 1 diversity metrics (exploratory, protocol note 2026-09-29).

The fit metrics (marginal KL, 3-mer r) cannot see a generator that repeats a small set of
sequences. From the saved NFE-100 samples of the held-out v2 cells, per seed:

    unique     distinct sequences / samples
    hamming    mean normalised Hamming distance over 20,000 random pairs of distinct samples
    train_copy fraction of samples identical to a sequence in that run's training set

References: held-out real sequences (8 random subsamples of 4,000) and the unigram sampler
(training-set frequencies, 8 seeds). Closeness to the real value is what matters, not size:
Hamming distance is driven largely by composition.

Reads results/m1_samples/ (kept locally, not committed); writes results/m1_diversity.json.
Run: python scripts/diversity_m1.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
from real_sequences import TIERS, load_split  # noqa: E402

SAMP = REPO / "results" / "m1_samples"
OUT = REPO / "results" / "m1_diversity.json"
DATASETS = (("dna_k4", 4), ("protein_k20", 20), ("codon_k64", 64))
ARMS = ("unconstrained", "mult_only", "center_only", "hard", "endpoint", "endpoint_rw",
        "fisher_sph", "gumbel_ce", "dirichlet_ce")
DENOISER = ("gumbel_ce", "dirichlet_ce")
N, PAIRS, SEEDS = 4000, 20_000, range(8)


def metrics(x: np.ndarray, train_set: set, seed: int) -> dict:
    x = np.asarray(x, np.uint8)[:N]
    rows = {r.tobytes() for r in x}
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(x), PAIRS)
    j = (i + rng.integers(1, len(x), PAIRS)) % len(x)          # j != i
    ham = float((x[i] != x[j]).mean())
    copies = float(np.mean([r.tobytes() in train_set for r in x]))
    return {"unique": len(rows) / len(x), "hamming": ham, "train_copy": copies}


def load_samples(ds: str, arm: str, seed: int) -> dict[str, np.ndarray]:
    """{'': nfe100 decode, 'state': state decode for denoiser arms}."""
    if arm in DENOISER:
        z = np.load(SAMP / f"{ds}_{arm}_s{seed}_heldout_v2_all.npz")
        return {"": z["nfe100_post"], "state": z["nfe100"]}
    return {"": np.load(SAMP / f"{ds}_{arm}_s{seed}.npz")["nfe100"]}


def main() -> int:
    cfg = TIERS["heldout"]
    out = {"n": N, "pairs": PAIRS, "seeds": list(SEEDS), "datasets": {}}
    for ds, k in DATASETS:
        tr, ho = load_split(ds, cfg["n_train"], cfg["n_heldout"])
        tr, ho = tr.numpy().astype(np.uint8), ho.numpy().astype(np.uint8)
        train_set = {r.tobytes() for r in tr}
        res: dict[str, list[dict]] = {}
        for seed in SEEDS:
            rng = np.random.default_rng(1000 + seed)
            res.setdefault("real_heldout", []).append(
                metrics(ho[rng.permutation(len(ho))[:N]], train_set, seed))
            p = np.bincount(tr.ravel(), minlength=k) / tr.size
            uni = np.random.default_rng(seed).choice(k, size=(N, tr.shape[1]), p=p)
            res.setdefault("unigram", []).append(metrics(uni, train_set, seed))
            for arm in ARMS:
                for tag, x in load_samples(ds, arm, seed).items():
                    res.setdefault(arm + (f"_{tag}" if tag else ""), []).append(
                        metrics(x, train_set, seed))
            # protocol addendum: the FM-vs-diffusion families from the sample-saving rerun
            # (fmdiff_real_*_heldout_div), at NFE 100 and at NFE 10 where 4.2 is decided.
            for fam in ("diffusion", "flow_matching"):
                f = SAMP / f"{ds}_{fam}_s{seed}_heldout_v2_div_all.npz"
                if f.exists():
                    z = np.load(f)
                    for nfe in ("100", "10"):
                        res.setdefault(f"{fam}_rerun_nfe{nfe}", []).append(
                            metrics(z[f"nfe{nfe}"], train_set, seed))
        summ = {}
        for name, lst in res.items():
            summ[name] = {m: {"mean": float(np.mean([d[m] for d in lst])),
                              "sd": float(np.std([d[m] for d in lst])),
                              "per_seed": [d[m] for d in lst]} for m in lst[0]}
        out["datasets"][ds] = summ
        print(f"\n{ds}")
        for name, s in summ.items():
            print(f"  {name:>20}  unique {s['unique']['mean']:.4f}  hamming "
                  f"{s['hamming']['mean']:.4f}+-{s['hamming']['sd']:.4f}  "
                  f"train copies {s['train_copy']['mean']:.4f}")
    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
