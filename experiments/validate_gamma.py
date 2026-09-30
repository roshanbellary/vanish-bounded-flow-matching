"""Validate the Marsaglia-Tsang Gamma sampler used by the Dirichlet baseline against scipy.

`real_sequences.sample_gamma` replaces torch's Gamma sampler (no MPS kernel). This script runs
the Kolmogorov-Smirnov test against scipy's Gamma CDF whose p-values the paper quotes.

Run: python -u experiments/validate_gamma.py
Writes: results/validate_gamma.json (an existing file is renamed aside with its mtime first).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
from real_sequences import sample_gamma  # noqa: E402

OUT = REPO / "results" / "validate_gamma.json"
SHAPES = (1.0, 2.5, 9.0)
N = 100_000


def main() -> int:
    torch.manual_seed(0)
    rows = []
    for a in SHAPES:
        x = sample_gamma(torch.full((N,), a)).numpy()
        ks = stats.kstest(x, stats.gamma(a).cdf)
        rows.append({"shape": a, "n": N, "ks_stat": float(ks.statistic), "p": float(ks.pvalue)})
        print(f"shape {a:>4}: KS D={ks.statistic:.5f}  p={ks.pvalue:.3f}")
    if OUT.exists():                 # results are append-only: keep the old file beside it
        OUT.rename(OUT.with_suffix(f".{int(OUT.stat().st_mtime)}{OUT.suffix}"))
    OUT.write_text(json.dumps({"rows": rows}, indent=2))
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
