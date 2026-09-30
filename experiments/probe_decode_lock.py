"""Feasibility probe: do sequence positions lock their decoded output early?

The proposed mechanism stops evaluating a coordinate once its discrete output can no longer
change. That only pays if positions actually settle well before t=1. This measures the
HEADROOM before any mechanism is built -- if the lock curve is flat until t=0.95, the idea is
dead and no novelty question matters.

Definition used here: a position is "locked at time t" if its argmax from t onward never
changes again for the rest of the trajectory. This is the ORACLE lock (computed with
hindsight), so it is an upper bound on what any runtime predicate could certify. Reporting
the ceiling first is the point: a mechanism cannot beat it.

Run: python -u experiments/probe_decode_lock.py
Writes: results/probe_decode_lock.json (an existing file is renamed aside with its mtime first).
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "experiments"))
import real_sequences as R

OUT = REPO / "results" / "probe_decode_lock.json"


@torch.no_grad()
def lock_curve(net, k, L, nfe=100, n=2000):
    """Record argmax at every step, then find each position's last change."""
    e = -torch.rand(n, L, k, device=R.DEV).clamp_min(1e-12).log()
    x = e / e.sum(-1, keepdim=True)
    h = 1.0 / nfe
    dec = []
    for i in range(nfe):
        t = torch.full((n,), i * h, device=R.DEV)
        v = R.velocity(net, x, t, "unconstrained")
        x = (x + h * v).clamp_min(0.0)
        x = x / x.sum(-1, keepdim=True).clamp_min(1e-12)
        dec.append(x.argmax(-1).cpu().numpy())
    dec = np.stack(dec)                       # (nfe, n, L)
    final = dec[-1]
    # last step at which the decode differed from the final decode
    differs = dec != final[None]              # (nfe, n, L)
    idx = np.arange(nfe)[:, None, None]
    last_change = np.where(differs, idx, -1).max(0)      # (n, L)
    lock_t = (last_change + 1) / nfe                     # fraction of trajectory
    return lock_t.ravel()


def main() -> int:
    cfg = dict(steps=3000, n_train=8000, n_eval=2000, hidden=128, layers=8, batch=256)
    out = {}
    print(f"device={R.DEV}  (oracle lock = upper bound on any runtime predicate)\n")
    print(f"{'dataset':>12} {'K':>4} | " + " ".join(f"t<={q:<5}" for q in (0.25,0.5,0.75,0.9))
          + " | mean lock t")
    print("-" * 74)
    for name, k in (("dna_k4", 4), ("protein_k20", 20), ("codon_k64", 64)):
        idx = R.load(name, k, cfg["n_train"])
        net = R.train(idx, k, "unconstrained", 0.0, cfg, 0)
        lt = lock_curve(net, k, idx.shape[1])
        fr = {q: float((lt <= q).mean()) for q in (0.25, 0.5, 0.75, 0.9)}
        out[name] = {"k": k, "frac_locked_by": fr, "mean_lock_t": float(lt.mean()),
                     "median_lock_t": float(np.median(lt))}
        print(f"{name:>12} {k:>4} | " + " ".join(f"{fr[q]:<6.3f}" for q in (0.25,0.5,0.75,0.9))
              + f" | {lt.mean():.3f}")
    if OUT.exists():                 # results are append-only: keep the old file beside it
        OUT.rename(OUT.with_suffix(f".{int(OUT.stat().st_mtime)}{OUT.suffix}"))
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT.relative_to(REPO)}")
    print("\nInterpretation: 'frac locked by t' is the fraction of positions whose decoded")
    print("symbol never changes after t. Compute saved is bounded by the area under that curve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
