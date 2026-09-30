# VANISH: Vanishing-Factor Ablation of Non-tangent Flows in Simplex and Hypercube

CIS 6270 (Fall 2026) Project 1: Roshan Bellary, Audhav Durai, Chang-Uk Jeong, Pedram Bayat.

The project asks whether continuous generative models leave bounded domains, and whether a field
that vanishes on the boundary helps. Modality 1 is sequences on the probability simplex (human DNA,
protein and codon windows, K = 4, 20, 64). Modality 2 is MNIST images in the unit box
`[0,1]^784`. The paper is `paper/main.pdf` and the defense deck is `slides/defense.pdf`.

## Repository layout

| Path | Contents |
|---|---|
| `experiments/` | One script per experiment (see the mapping below). |
| `src/dgm/` | Shared code. `dgm/paths/simplex.py` has the sphere/simplex maps used by Fisher-Flow; `dgm/runctx.py` has run bookkeeping. |
| `data/` | Datasets. Not committed (gitignored); see *Data preparation*. |
| `results/` | Every result file behind every number in the paper. Append-only: failed and superseded runs stay on disk. |
| `scripts/make_tables.py` | Regenerates every table and every in-text number (`paper/tables/*.tex`, including `nums.tex`). |
| `scripts/make_figures.py` | Regenerates every figure (`paper/figures/*.pdf`). |
| `paper/` | LaTeX source (official CIS 6270 NeurIPS 2026 template, `neurips_2026.sty`). |
| `slides/` | Defense deck (Beamer). `slides/README.md` has the speaking split. |
| `docs/ideas_v2.md` | Candidate-innovation notes. |
| `tests/` | Unit tests for `dgm/paths/simplex.py` (`pytest tests`). |

## Environment

Python >= 3.10 with `torch`, `numpy`, `scipy` and `matplotlib`. We used Python 3.12, PyTorch 2.6
(local, Apple MPS) and PyTorch 2.8 / CUDA 12.8 on rented single GPUs (NVIDIA A40; one run on an RTX
A6000).

```
pip install torch numpy scipy matplotlib
pip install -e .            # makes src/dgm importable (the scripts also add src/ to sys.path)
```

The paper needs a TeX distribution with `pdflatex` and `bibtex`.

## Data preparation

```
python -u experiments/prepare_real_data.py      # DNA (Ensembl GRCh38 chr1) and protein (UniProt SwissProt)
```

This fetches from the Ensembl and UniProt REST APIs and writes `data/dna_k4.npy` and
`data/protein_k20.npy`. Current API releases can differ from the ones we fetched. To reproduce the
paper exactly, use the arrays we used and check them against these SHA-256 hashes:

```
6b11050a8aaf8a5bacfb73b3f36390c63a87fb55eec1e700a25042f97cd59b20  data/dna_k4.npy
9079f748ebbeeabf0c50a6280d0080c7a85f15bfc5dee61cdc536e6ee813aebe  data/protein_k20.npy
155ec3ceacbc7271214cabbbcb1a6321ba19e606d93e279e1a253ce4b46862d8  data/codon_k64.npy
c5b45806d970e809a5f376f9b8461b7ca3b3e0a321282d72d06d00ab6c6c418f  data/mnist_train.npy
10226b938104d9231ead4e9a0627f17e66cd18baf31a28784f1b72147057decf  data/mnist_labels.npy
```

**Codon set.** `data/codon_k64.npy` is chromosome-1 sequence read three bases at a time
(384-base windows). The script that built it was not retained, and the paper says so. We cannot
confirm that its windows coincide with the DNA set's.

**MNIST.** Put the standard IDX files in `data/MNIST/raw/`, for example via
`torchvision.datasets.MNIST(root="data", download=True)`, then run:

```python
import gzip, numpy as np
def idx(p):
    d = gzip.open(p).read(); nd = d[3]
    dims = [int.from_bytes(d[4 + 4*i:8 + 4*i], "big") for i in range(nd)]
    return np.frombuffer(d, np.uint8, offset=4 + 4*nd).reshape(dims)
np.save("data/mnist_train.npy", idx("data/MNIST/raw/train-images-idx3-ubyte.gz"))
np.save("data/mnist_labels.npy", idx("data/MNIST/raw/train-labels-idx1-ubyte.gz").astype(np.int64))
```

This reproduces the two arrays above byte for byte.

## Running the experiments

Results files are **append-only**; the scripts are written so that re-running one does not
destroy committed results.
- `real_sequences.py`, `fm_vs_diffusion_real.py` and `modality2_unet.py` take `--tier smoke`
  (local, under two minutes: checks shapes and finite outputs) or `--tier full` (the numbers in
  the paper). Smoke runs write to separate files that the table code ignores. A full run of
  `modality2_unet.py` refuses to overwrite a results file that already exists.
- The other scripts take no flags and run their full experiment. If their output file already
  exists, they move it aside (renamed with a timestamp) before writing a new one.

Avoid `--help`; read the argparse block instead.

| Paper | Script | Command | Writes |
|---|---|---|---|
| Table 1 (§4.2, FM vs diffusion, real data) | `experiments/fm_vs_diffusion_real.py` | `--tier full [--only DATASET] [--family flow_matching\|diffusion] [--seeds 0,1,...]` | `results/fmdiff_real_<ds>_<family>_full.jsonl` (appends) |
| Table S1 (§4.2, preregistered synthetic run) | `experiments/fm_vs_diffusion.py` | no flags | `results/fm_vs_diffusion.jsonl` |
| Tables 2 and S3 (§4.3, guidance) | `experiments/guidance_modality1.py` | no flags | `results/guidance_modality1.json` |
| Tables 3, 4 and S2; Figs S1–S4 (§4.4–4.5) | `experiments/real_sequences.py` | `--tier full --only DATASET --arm ARM [--seeds 0,1,2,3]` | `results/endpoint_<ds>_<arm>[_s<seeds>].jsonl` (appends) |
| Table 5; Figs S5–S6 (§4.6) | `experiments/modality2_unet.py` | **two runs per arm:** `--tier full --arm ARM` (seeds 0–2), then `--tier full --arm ARM --seeds 3,4,5,6,7` | `results/modality2_unet_<arm>.jsonl`, `results/modality2_unet_<arm>_s3-4-5-6-7.jsonl`, `results/samples_*.npy` |
| §4.4 step-size sweep | `experiments/probe_tangency_violation.py` | no flags | `results/probe_tangency_violation.json` |
| Appendix, commitment time | `experiments/probe_decode_lock.py` | no flags | `results/probe_decode_lock.json` |
| Appendix, KS check of the Gamma sampler | `experiments/validate_gamma.py` | no flags | `results/validate_gamma.json` |

Here `DATASET` is one of `dna_k4`, `protein_k20` or `codon_k64`.

- **Modality 1 arms:**
  - `unconstrained`: linear FM, the base.
  - `mult_only`: ours, `v = x ⊙ w`.
  - `center_only`.
  - `hard`: the full replicator form.
  - `dirichlet`, `fisher_sph` and `gumbel`: the external baselines, which are our re-implementations.
  - `endpoint` and `endpoint_rw`: falsified interventions.

  `fisher` is an earlier Fisher-Flow run whose sampler integrated a sphere velocity in simplex
  coordinates. It stays on disk and is disclosed in appendix A.4; `fisher_sph`
  replaces it in every table.
- **Modality 2 arms:** `unconstrained` (base), `mult` (ours), `reflected`, `mirror` and `dynthresh`.

Timing is in the paper (§4.1.2). About 6–9 processes can run concurrently on one GPU; 15 thrashed
it.

## Regenerating the paper

```
python3 scripts/make_tables.py      # every table and every in-text number -> paper/tables/
python3 scripts/make_figures.py     # every figure -> paper/figures/
cd paper  && pdflatex main && bibtex main && pdflatex main && pdflatex main
cd slides && pdflatex defense && pdflatex defense
```

No number in the paper or the slides is typed by hand. Running prose numbers are LaTeX macros
generated into `paper/tables/nums.tex`.

## External code

No external code was adapted. The three Modality 1 baselines (Dirichlet FM, Fisher-Flow and
Gumbel-Softmax FM) and the three Modality 2 baselines (reflection, mirror map and dynamic
thresholding) are our own re-implementations of the published formulations, run inside our
harness. They are not the authors' code, and no number in the paper is quoted from another paper.
