"""Section 4.2 on REAL sequences: flow matching versus diffusion on the dilated CNN.

`fm_vs_diffusion.py` runs §4.2 on a synthetic categorical target with a small MLP; the
registered protocol names the real task and the 20-layer dilated CNN. This is the confirmation
registered in the protocol amendment of 2026-09-24: the same two processes on the three
Modality 1 datasets and the backbone every other Modality 1 table uses. It does not change
the base-model choice and is reported whichever way it falls.

  flow matching   the harness's `unconstrained` arm: linear path from uniform-on-simplex,
                  Euler, clamp + renormalise. Retrained here so the families are paired.
  diffusion       VP Gaussian diffusion on one-hot simplex coordinates, cosine schedule,
                  epsilon-prediction, deterministic DDIM, argmax decode -- the same process
                  as the synthetic §4.2 run, on the CNN.

Everything else (data subset, architecture, optimiser, lr, steps, batch, seeds, n_eval, NFE
grid, metrics) is shared with `real_sequences.py` by import, not by copy.

Output is append-only and never truncated; each tier writes its own file. Seeds already
present in the output are skipped, so a relaunch resumes.

Run: python -u experiments/fm_vs_diffusion_real.py --tier smoke
     python -u experiments/fm_vs_diffusion_real.py --tier full --only dna_k4 --family diffusion
Writes: results/fmdiff_real_<dataset>_<family>_<tier><suffix>.jsonl; above smoke tier also
        per-NFE samples in results/m1_samples/ and weights in results/checkpoints/.
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
sys.path.insert(0, str(REPO / "experiments"))
from fm_vs_diffusion import alpha_bar  # noqa: E402
from real_sequences import (  # noqa: E402
    DEV, NFES, TIERS, DilatedCNN, kmer_freq, load, load_split, marginals, onehot, sample, score,
    train,
)

DATASETS = (("dna_k4", 4), ("protein_k20", 20), ("codon_k64", 64))
FAMILIES = ("flow_matching", "diffusion")


def train_diff(data_idx, k, cfg, seed):
    """Identical net, optimiser, lr, steps and batch to `real_sequences.train`."""
    torch.manual_seed(seed)
    net = DilatedCNN(k, cfg["hidden"], cfg["layers"]).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    x1_all = onehot(data_idx, k).to(DEV)
    bs = min(cfg.get("batch", 128), len(x1_all))
    for _ in range(cfg["steps"]):
        x1 = x1_all[torch.randint(0, len(x1_all), (bs,), device=DEV)]
        s = torch.rand(bs, device=DEV)
        ab = alpha_bar(s)[:, None, None]
        eps = torch.randn_like(x1)
        xs = ab.sqrt() * x1 + (1 - ab).sqrt() * eps
        loss = (net(xs, s) - eps).square().sum(-1).mean()      # epsilon-prediction
        opt.zero_grad(); loss.backward(); opt.step()
    return net.eval()


@torch.no_grad()
def sample_diff(net, k, L, nfe, n, batch=4000):
    """Deterministic DDIM. x0_outside is counted exactly as `real_sequences.sample` counts it:
    a (sequence, step) pair is bad if any position's clean-sample estimate leaves the simplex.
    Step violation is undefined here -- the diffusion iterate is off-simplex by design."""
    outs, x0_bad, tot = [], 0, 0
    ss = torch.linspace(1.0, 0.0, nfe + 1, device=DEV)
    for st in range(0, n, batch):
        b = min(batch, n - st)
        x = torch.randn(b, L, k, device=DEV)
        for i in range(nfe):
            ab, ab_n = alpha_bar(ss[i].view(1)), alpha_bar(ss[i + 1].view(1))
            eps = net(x, ss[i].expand(b))
            x1_hat = (x - (1 - ab).sqrt() * eps) / ab.sqrt().clamp_min(1e-4)
            x0_bad += int((x1_hat.min(-1).values < 0).any(-1).sum())
            tot += b
            x = ab_n.sqrt() * x1_hat + (1 - ab_n).sqrt() * eps
        outs.append(x.argmax(-1).cpu().numpy())
    return np.concatenate(outs), None, x0_bad / tot


def done_seeds(path: Path) -> set[int]:
    if not path.exists():
        return set()
    return {json.loads(l)["seed"] for l in path.read_text().splitlines() if l.strip()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="smoke", choices=("smoke", "full", "heldout"))
    ap.add_argument("--only", default=None, choices=[d for d, _ in DATASETS])
    ap.add_argument("--family", default=None, choices=FAMILIES)
    ap.add_argument("--seeds", default=None, help="comma-separated subset, e.g. 0,1,2,3")
    # A rerun (e.g. --out-suffix _div) writes to its own files so the registered rows are never
    # appended to or skipped (protocol addendum 2026-09-29).
    ap.add_argument("--out-suffix", default="", help="suffix for output file names")
    a = ap.parse_args()
    cfg = TIERS[a.tier]
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else list(cfg["seeds"])
    datasets = [d for d in DATASETS if a.only in (None, d[0])]
    families = [f for f in FAMILIES if a.family in (None, f)]
    t0 = time.time()
    print(f"device={DEV}  tier={a.tier}  cfg={cfg}  seeds={seeds}", flush=True)

    for name, k in datasets:
        # Score against a held-out split when the tier has one (protocol amendment 2026-09-29).
        if "n_heldout" in cfg:
            idx, ho = load_split(name, cfg["n_train"], cfg["n_heldout"])
            ref = ho.numpy()
            tr_m, tr_f = marginals(idx.numpy(), k), kmer_freq(idx.numpy(), k)
        else:
            idx = load(name, k, cfg["n_train"])
            ref = idx.numpy()
        ref_m, ref_f = marginals(ref, k), kmer_freq(ref, k)
        L = idx.shape[1]
        for fam in families:
            out = REPO / "results" / f"fmdiff_real_{name}_{fam}_{a.tier}{a.out_suffix}.jsonl"
            out.parent.mkdir(parents=True, exist_ok=True)
            have = done_seeds(out)
            for seed in seeds:
                if seed in have:
                    print(f"  skip {name} {fam} seed {seed} (already in {out.name})", flush=True)
                    continue
                tc = time.time()
                if fam == "flow_matching":
                    net = train(idx, k, "unconstrained", 0.0, cfg, seed)
                else:
                    net = train_diff(idx, k, cfg, seed)
                res, saved = {}, {}
                for nfe in NFES:
                    if fam == "flow_matching":
                        gen, sv, ov = sample(net, k, L, "unconstrained", nfe, cfg["n_eval"])
                    else:
                        gen, sv, ov = sample_diff(net, k, L, nfe, cfg["n_eval"])
                    saved[f"nfe{nfe}"] = gen.astype(np.uint8)
                    mkl, r = score(gen, ref_m, ref_f, k)
                    res[str(nfe)] = {"marginal_kl": mkl, "kmer_r": r,
                                     "x0_outside": ov, "step_violation": sv}
                    if "n_heldout" in cfg:
                        res[str(nfe)]["marginal_kl_train"], res[str(nfe)]["kmer_r_train"] = \
                            score(gen, tr_m, tr_f, k)
                if a.tier != "smoke":
                    dv = "v2" if (REPO / "data" / "v2" / f"{name}_train.npy").exists() else "v1"
                    sd = REPO / "results" / "m1_samples"
                    sd.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        sd / f"{name}_{fam}_s{seed}_{a.tier}_{dv}{a.out_suffix}_all.npz", **saved)
                    cd = REPO / "results" / "checkpoints"
                    cd.mkdir(parents=True, exist_ok=True)
                    torch.save({k_: v.cpu() for k_, v in net.state_dict().items()},
                               cd / f"{name}_{fam}_s{seed}_{a.tier}_{dv}{a.out_suffix}.pt")
                row = {"dataset": name, "k": k, "family": fam, "seed": seed, "tier": a.tier,
                       "data": "v2" if (REPO / "data" / "v2" / f"{name}_train.npy").exists()
                       else "v1",
                       "device": str(DEV), "wall_s": round(time.time() - tc, 1), "per_nfe": res}
                with out.open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
                m = res["10"]
                print(f"{name:>12} {fam:>14} seed {seed} | KL@10 {m['marginal_kl']:.4f}  "
                      f"3mer r {m['kmer_r']:+.4f}  x0 out {m['x0_outside']:.3f}  "
                      f"({row['wall_s']:.0f}s)", flush=True)

    print(f"done ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
