"""Fetch two REAL biological sequence datasets with matched length and preprocessing.

Much of the continuous-simplex literature validates at K=4 DNA, while proteins need K=20.
Comparing across alphabet size needs a controlled pair: identical sequence length, identical
preprocessing, identical model. Comparing a published DNA benchmark against a published
protein benchmark would confound alphabet size with length, dataset size, and task.

  DNA      K=4   real human genomic sequence (Ensembl REST, GRCh38 chr1), windowed
  PROTEIN  K=20  real reviewed human proteins (UniProt SwissProt)

Both are cropped to the same length L and integer-encoded.

Run: python -u experiments/prepare_real_data.py
Writes: data/dna_k4.npy, data/protein_k20.npy (int8 index arrays, shape (N, L)).
"""

from __future__ import annotations

import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
L = 128                      # sequence length, identical for both modalities
DNA_ALPHA = "ACGT"           # K=4
PROT_ALPHA = "ACDEFGHIKLMNPQRSTVWY"   # K=20, the standard amino acids


def fetch_protein(target=25_000) -> list[str]:
    """Reviewed human proteins, paginated through the UniProt REST API."""
    seqs: list[str] = []
    q = f"reviewed:true AND organism_id:9606 AND length:[{L} TO 1000]"
    url = "https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode(
        {"query": q, "format": "fasta", "size": 500})
    while url and len(seqs) < target:
        req = urllib.request.Request(url, headers={"User-Agent": "cis6270-project/0.1"})
        with urllib.request.urlopen(req, timeout=120) as r:
            body = r.read().decode()
            link = r.headers.get("Link", "")
        cur: list[str] = []
        for line in body.splitlines():
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur)); cur = []
            else:
                cur.append(line.strip())
        if cur:
            seqs.append("".join(cur))
        url = link.split(";")[0].strip("<> ") if 'rel="next"' in link else None
        print(f"  protein: {len(seqs)}")
    return seqs


def fetch_dna(n_windows=25_000) -> list[str]:
    """Real human genomic sequence, sliced into non-overlapping windows.

    Chromosome 1 is fetched in chunks; Ensembl caps a single request, so we walk a region
    large enough to yield the window count we need after discarding N-containing windows.
    """
    seqs: list[str] = []
    start = 1_000_000
    chunk = 400_000        # well under the Ensembl per-request limit
    while len(seqs) < n_windows:
        end = start + chunk - 1
        url = (f"https://rest.ensembl.org/sequence/region/human/1:{start}..{end}"
               f"?content-type=text/plain")
        req = urllib.request.Request(url, headers={"User-Agent": "cis6270-project/0.1"})
        with urllib.request.urlopen(req, timeout=120) as r:
            s = r.read().decode().strip().upper()
        for i in range(0, len(s) - L, L):
            w = s[i:i + L]
            if set(w) <= set(DNA_ALPHA):     # drop windows containing N or soft-masked gaps
                seqs.append(w)
        start = end + 1
        print(f"  dna: {len(seqs)}")
        if len(s) < chunk // 2:
            break
    return seqs[:n_windows]


def encode(seqs: list[str], alpha: str) -> np.ndarray:
    idx = {c: i for i, c in enumerate(alpha)}
    out = []
    for s in seqs:
        s = s[:L]
        if len(s) < L or any(c not in idx for c in s):
            continue
        out.append([idx[c] for c in s])
    return np.array(out, dtype=np.int8)


def main() -> int:
    DATA.mkdir(exist_ok=True)

    print("fetching protein (K=20) ...")
    prot = encode(fetch_protein(), PROT_ALPHA)
    np.save(DATA / "protein_k20.npy", prot)
    print(f"  -> {prot.shape} saved\n")

    print("fetching DNA (K=4) ...")
    dna = encode(fetch_dna(), DNA_ALPHA)
    np.save(DATA / "dna_k4.npy", dna)
    print(f"  -> {dna.shape} saved\n")

    n = min(len(prot), len(dna))
    print(f"matched sample count available: {n}  (length {L} for both)")
    for name, arr, alpha in (("dna_k4", dna, DNA_ALPHA), ("protein_k20", prot, PROT_ALPHA)):
        counts = np.bincount(arr.ravel(), minlength=len(alpha)).astype(float)
        p = counts / counts.sum()
        # Marginal entropy in nats: an upper bound on what a position-independent model
        # could achieve, and a sanity check that the two datasets are genuinely different.
        ent = float(-(p * np.log(np.maximum(p, 1e-12))).sum())
        print(f"  {name:>12}: K={len(alpha)}  marginal entropy {ent:.4f} nats "
              f"(uniform would be {np.log(len(alpha)):.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
