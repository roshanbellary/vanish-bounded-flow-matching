"""Run identity and the append-only run ledger (results/ledger.jsonl).

Used by `experiments/real_sequences.py` for per-run directories under results/runs/. The
ledger is a partial record, not the source of the paper's numbers, which come from the
per-experiment result files. The two invariants:

  * ``run_id`` is a pure function of (resolved config, seed, code revision), so the same
    experiment requested twice resolves to the same row and is skipped the second time.
  * ``append`` never rewrites history. Failed and embarrassing runs stay in the ledger.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
LEDGER = REPO / "results" / "ledger.jsonl"
RUNS = REPO / "results" / "runs"

TIERS = ("smoke", "dev", "full")


def git_sha(dirty_ok: bool = True) -> str:
    """Short revision of the CODE that produced a run.

    A dirty tree gets a ``+dirty`` suffix rather than an error: smoke tests legitimately
    run mid-edit. But a ``full``-tier run with a dirty tree is refused in
    ``RunContext.create`` — paper numbers must come from committed code.

    Changes under ``results/`` do not count as dirty. The guard exists to pin the *code*,
    and results are the one thing a run is *supposed* to modify; counting them would make
    the first cell of a multi-cell run block every later one.
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True
        ).strip()
        porcelain = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO, text=True
        ).splitlines()
    except subprocess.CalledProcessError:
        return "nogit"
    # porcelain lines are "XY <path>"; a rename is "XY <old> -> <new>".
    dirty = [
        ln for ln in porcelain
        if ln.strip() and not ln[3:].split(" -> ")[-1].strip('"').startswith("results/")
    ]
    if dirty and dirty_ok:
        return f"{sha}+dirty"
    return sha


def canonical(obj: Any) -> str:
    """Stable JSON rendering, so key order can never change a run_id."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def compute_run_id(config: dict, seed: int, sha: str) -> str:
    payload = canonical({"config": config, "seed": seed, "git": sha})
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def load_ledger() -> list[dict]:
    if not LEDGER.exists():
        return []
    rows = []
    for line in LEDGER.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def existing_ids() -> set[str]:
    return {r["run_id"] for r in load_ledger()}


def append(row: dict) -> None:
    """Append one row. The only write path into the ledger."""
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as fh:
        fh.write(canonical(row) + "\n")


@dataclass
class RunContext:
    run_id: str
    config: dict
    seed: int
    tier: str
    git: str
    dir: Path
    started: float = field(default_factory=time.time)

    @classmethod
    def create(cls, config: dict, seed: int, tier: str) -> "RunContext":
        if tier not in TIERS:
            raise ValueError(f"tier must be one of {TIERS}, got {tier!r}")
        sha = git_sha()
        if tier == "full" and sha.endswith("+dirty"):
            raise RuntimeError(
                "refusing a full-tier run from a dirty tree: paper numbers must be "
                "reproducible from a commit. Commit your changes first."
            )
        run_id = compute_run_id(config, seed, sha)
        d = RUNS / run_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.resolved.json").write_text(canonical(config))
        return cls(run_id=run_id, config=config, seed=seed, tier=tier, git=sha, dir=d)

    def already_done(self) -> bool:
        return self.run_id in existing_ids()

    def finish(self, metrics: dict, status: str = "ok", **extra: Any) -> dict:
        """Write metrics and append the ledger row. Call exactly once per run."""
        elapsed = time.time() - self.started
        row = {
            "run_id": self.run_id,
            "status": status,
            "tier": self.tier,
            "seed": self.seed,
            "git": self.git,
            "config": self.config,
            "metrics": metrics,
            "wall_seconds": round(elapsed, 1),
            "gpu_hours": round(elapsed / 3600.0, 4) if extra.get("on_gpu") else 0.0,
            "host": platform.node(),
            "device": extra.get("device", os.environ.get("DGM_DEVICE", "unknown")),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        row.update({k: v for k, v in extra.items() if k not in row})
        (self.dir / "metrics.json").write_text(json.dumps(row, indent=2, default=str))
        append(row)
        return row
