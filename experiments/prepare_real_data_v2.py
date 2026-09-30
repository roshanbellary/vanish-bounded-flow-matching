"""Modality 1 data, version 2 (protocol amendment 2026-09-29, "data v2"). Builds a disjoint
train / held-out split for each of the three alphabets and writes

    data/v2/{name}_train.npy     int8 (N_train_pool, 128)
    data/v2/{name}_heldout.npy   int8 (N_heldout_pool, 128)
    data/v2/manifest.json        sources, counts, split rule, sha256 of every array

Why a v2. The v1 sets had three defects:
  * protein kept only each protein's first 128 residues (99.7% start with Met; isoforms share
    N-termini, so 252 rows were duplicates and 43 leaked across a held-out split);
  * DNA and codon were split window by window, so repeat copies and neighbouring windows could
    fall on both sides;
  * the codon set's build script was lost, so what it contained could not be stated.

The rules, identical in spirit for every dataset:
  * PROTEIN (K=20): reviewed human SwissProt, length 128-1000 (as v1), each protein tiled into
    non-overlapping 128-residue windows from its N-terminus; a window with any non-standard
    residue is dropped. The split is BY PROTEIN: 20% of accessions (seeded) are held out, so no
    protein contributes to both sides.
  * DNA (K=4) and TRIPLET (K=64, file name kept as `codon_k64`): ONE contiguous region of GRCh38
    chromosome 1 starting at 1,000,000. DNA tiles it into 128-base windows; the triplet set
    tiles the same region into 384-base windows read as 128 non-overlapping trinucleotides.
    These are 3-mer tokens of genomic DNA, NOT reading-frame codons. Windows containing N are
    dropped. The split is BY GENOMIC BLOCK of 100 kb: 20% of blocks (seeded) are held out, and
    the same blocks are held out for both tokenisations, so DNA and triplet differ only in
    tokenisation and window length.
  * Every dataset: exact duplicate windows are removed within each side, and any held-out window
    identical to a training window is removed from the held-out side.

Raw downloads are cached in data/v2/raw so the build is reproducible offline once fetched.

Run: python -u experiments/prepare_real_data_v2.py
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "v2"
RAW = OUT / "raw"
L = 128
DNA_ALPHA = "ACGT"
PROT_ALPHA = "ACDEFGHIKLMNPQRSTVWY"
TRIPLETS = [a + b + c for a in DNA_ALPHA for b in DNA_ALPHA for c in DNA_ALPHA]   # K=64
HELD_FRAC = 0.20
SEED = 0
CHR_START, CHR_LEN = 1_000_000, 9_000_000      # 9 Mb: enough 384-base windows for the triplets
BLOCK = 100_000                                 # genomic block for the DNA/triplet split
UA = {"User-Agent": "cis6270-project/0.2"}


def get(url: str) -> tuple[str, str]:
    """(body, Link header), retrying transient server errors with backoff."""
    import time
    for attempt in range(8):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                        timeout=180) as r:
                return r.read().decode(), r.headers.get("Link", "") or ""
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            code = getattr(e, "code", None)
            if code is not None and code < 500 and code != 429:
                raise
            print(f"  retry {attempt + 1} after {e}", flush=True)
            time.sleep(min(60, 5 * 2 ** attempt))
    raise RuntimeError(f"giving up on {url}")


def fetch_proteins() -> list[tuple[str, str]]:
    """(accession, sequence) for reviewed human proteins of length 128-1000, cached."""
    cache = RAW / "uniprot_human_128_1000.fasta"
    if not cache.exists():
        q = f"reviewed:true AND organism_id:9606 AND length:[{L} TO 1000]"
        url = "https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode(
            {"query": q, "format": "fasta", "size": 500})
        parts = []
        while url:
            body, link = get(url)
            parts.append(body)
            url = link.split(";")[0].strip("<> ") if 'rel="next"' in link else None
            print(f"  uniprot pages: {len(parts)}", flush=True)
        cache.write_text("".join(parts))
    out, acc, cur = [], None, []
    for line in cache.read_text().splitlines():
        if line.startswith(">"):
            if acc:
                out.append((acc, "".join(cur)))
            acc, cur = line.split("|")[1], []
        else:
            cur.append(line.strip())
    if acc:
        out.append((acc, "".join(cur)))
    return out


def fetch_chr1() -> str:
    """GRCh38 chr1:[CHR_START, CHR_START+CHR_LEN), upper-cased, cached."""
    cache = RAW / f"chr1_{CHR_START}_{CHR_START + CHR_LEN}.txt"
    if not cache.exists():
        parts, s, step = [], CHR_START, 400_000
        while s < CHR_START + CHR_LEN:
            e = min(s + step, CHR_START + CHR_LEN) - 1
            body, _ = get(f"https://rest.ensembl.org/sequence/region/human/1:{s}..{e}"
                          f"?content-type=text/plain")
            parts.append(body.strip().upper())
            s = e + 1
            print(f"  chr1: {sum(map(len, parts)):,} bases", flush=True)
        cache.write_text("".join(parts))
    return cache.read_text()


def dedup_split(tr: list[tuple], ho: list[tuple]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Rows are tuples of token ids. Exact duplicates removed within each side; held-out rows
    that also occur in train removed from held-out. Order preserved (first occurrence)."""
    def uniq(rows):
        seen, keep = set(), []
        for r in rows:
            if r not in seen:
                seen.add(r); keep.append(r)
        return keep, seen
    tr_u, tr_set = uniq(tr)
    ho_u, _ = uniq(ho)
    ho_c = [r for r in ho_u if r not in tr_set]
    stats = {"train_raw": len(tr), "train_dedup": len(tr_u), "heldout_raw": len(ho),
             "heldout_dedup": len(ho_u), "heldout_after_cross_removal": len(ho_c)}
    return np.array(tr_u, np.int8), np.array(ho_c, np.int8), stats


def build_protein(rng) -> dict:
    prots = fetch_proteins()
    idx = {c: i for i, c in enumerate(PROT_ALPHA)}
    accs = sorted({a for a, _ in prots})
    held = set(rng.permutation(accs)[: int(round(HELD_FRAC * len(accs)))])
    tr, ho, dropped = [], [], 0
    for a, s in prots:
        for i in range(0, len(s) - L + 1, L):
            w = s[i:i + L]
            if any(c not in idx for c in w):
                dropped += 1
                continue
            (ho if a in held else tr).append(tuple(idx[c] for c in w))
    a_tr, a_ho, st = dedup_split(tr, ho)
    st.update(proteins=len(accs), heldout_proteins=len(held), windows_nonstandard_dropped=dropped)
    return {"protein_k20": (a_tr, a_ho, st)}


def build_genomic(rng) -> dict:
    seq = fetch_chr1()
    n_blocks = (len(seq) + BLOCK - 1) // BLOCK
    held = set(rng.permutation(n_blocks)[: int(round(HELD_FRAC * n_blocks))].tolist())
    out = {}
    for name, span in (("dna_k4", L), ("codon_k64", 3 * L)):
        tr, ho, dropped, straddle = [], [], 0, 0
        for i in range(0, len(seq) - span + 1, span):
            w = seq[i:i + span]
            if set(w) - set(DNA_ALPHA):
                dropped += 1
                continue
            b0, b1 = i // BLOCK, (i + span - 1) // BLOCK
            if (b0 in held) != (b1 in held):          # window crosses a split boundary
                straddle += 1
                continue
            if span == L:
                row = tuple("ACGT".index(c) for c in w)
            else:
                row = tuple(TRIPLETS.index(w[j:j + 3]) for j in range(0, span, 3))
            (ho if b0 in held else tr).append(row)
        a_tr, a_ho, st = dedup_split(tr, ho)
        st.update(region=f"chr1:{CHR_START}-{CHR_START + len(seq) - 1}", window_bases=span,
                  block_bases=BLOCK, blocks=n_blocks, heldout_blocks=len(held),
                  windows_with_N_dropped=dropped, windows_straddling_split_dropped=straddle)
        out[name] = (a_tr, a_ho, st)
    return out


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    built = {**build_protein(rng), **build_genomic(rng)}
    manifest = {"rule": __doc__.split("The rules")[1].split("Raw downloads")[0].strip(),
                "seed": SEED, "heldout_fraction": HELD_FRAC, "length": L, "datasets": {}}
    for name, (tr, ho, st) in built.items():
        for side, arr in (("train", tr), ("heldout", ho)):
            p = OUT / f"{name}_{side}.npy"
            np.save(p, arr)
            st[f"{side}_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
        manifest["datasets"][name] = st
        print(f"{name:>12}: train {tr.shape}  held-out {ho.shape}  {st}", flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
