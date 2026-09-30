"""Regenerate every paper table from the results files. No number is typed by hand.

Every table, and every number quoted in running text (as macros in `paper/tables/nums.tex`),
is computed from `results/`, so each value traces back to the run that produced it.
Prints a readable console form and writes LaTeX fragments under `paper/tables/`.

Run: python scripts/make_tables.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
TEX = REPO / "paper" / "tables"
RNG = np.random.default_rng(0)

DATASETS = (("dna_k4", 4, "DNA"), ("protein_k20", 20, "Protein"), ("codon_k64", 64, "Codon"))
ARMS = ("unconstrained", "hard", "endpoint")


def fisher_arm(rows) -> str:
    """The Fisher-Flow row. `fisher` integrated a sphere velocity in simplex coordinates (a
    harness bug); `fisher_sph` samples on the sphere as published. Use the corrected arm once
    its rows exist; the buggy rows stay on disk."""
    return "fisher_sph" if any(r["arm"] == "fisher_sph" for r in rows) else "fisher"


# Corrected external arms (protocol amendment 2026-09-29): `gumbel` and `dirichlet` regress on
# targets that are not the published methods; `gumbel_ce` / `dirichlet_ce` train the published
# cross-entropy denoisers. A corrected arm is used only once all 24 of its cells exist, so a
# partial batch can never be mixed into a table.
CORRECTED = {"gumbel": "gumbel_ce", "dirichlet": "dirichlet_ce"}


def ext_arm(rows, name: str) -> str:
    if name == "fisher":
        return fisher_arm(rows)
    new = CORRECTED[name]
    n = len({(r["dataset"], r["seed"]) for r in rows if r["arm"] == new})
    return new if n == len(DATASETS) * 8 else name


def m2_files() -> tuple[list[str], bool]:
    """Modality 2 result files. The rerun with ONE shared FD extractor (protocol amendment
    2026-09-29) replaces the per-process-extractor batch once all 40 cells exist."""
    fixed = sorted(glob.glob(str(RES / "m2fixed_*.jsonl")))
    n = sum(1 for f in fixed for l in open(f) if '"seed"' in l)
    if n == 40:
        return fixed, True
    return sorted(glob.glob(str(RES / "modality2_unet_*.jsonl"))), False


def t_ci(d) -> tuple[float, float, float]:
    """Student-t 95% interval of a mean. At 8 seeds the percentile bootstrap under-covers, so
    this is reported beside it as a robustness check; the bootstrap stays the registered rule."""
    from scipy import stats
    d = np.asarray(d, float)
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), d.mean() - h, d.mean() + h


def unigram_reference(n_seeds: int = 8):
    """A zero-parameter reference: every position drawn i.i.d. from the training set's overall
    symbol frequencies. Scored with the harness's own metric code against the same reference
    set, so a trained model that does not beat it has learned composition and nothing else."""
    import sys
    sys.path.insert(0, str(REPO / "experiments"))
    from real_sequences import TIERS, kmer_freq, load, load_split, marginals
    cfg = TIERS["heldout" if HELDOUT else "full"]
    out = {}
    for ds, k, _ in DATASETS:
        if HELDOUT:     # training-set frequencies, scored against the held-out reference
            tr, ho = load_split(ds, cfg["n_train"], cfg["n_heldout"])
            train, ref = tr.numpy(), ho.numpy()
        else:
            ref = train = load(ds, k, cfg["n_train"]).numpy()
        ref_m, ref_f = marginals(ref, k), kmer_freq(ref, k)
        p = np.bincount(train.ravel(), minlength=k) / train.size
        rs, kls = [], []
        for seed in range(n_seeds):
            g = np.random.default_rng(seed).choice(k, size=(cfg["n_eval"], ref.shape[1]), p=p)
            gm, gf = marginals(g, k), kmer_freq(g, k)
            kls.append(float((ref_m * (np.log(ref_m) - np.log(gm))).sum(1).mean()))
            rs.append(float(np.corrcoef(ref_f, gf)[0, 1]))
        out[ds] = (np.mean(rs), np.std(rs), np.mean(kls))
    return out


def boot(d: np.ndarray, n: int = 20000):
    """Paired bootstrap mean and 95% CI."""
    d = np.asarray(d, float)
    b = [RNG.choice(d, len(d), replace=True).mean() for _ in range(n)]
    return d.mean(), *np.percentile(b, [2.5, 97.5])


# protocol amendments 2026-09-29: Modality 1 is re-evaluated on a held-out split of rebuilt data
# (tier "heldout", data "v2"). Once all 216 cells exist they replace the training-reference rows
# everywhere; until then the old rows are used, so a partial batch is never mixed in.
HELDOUT = False
DENOISER = ("gumbel_ce", "dirichlet_ce")
N_HELDOUT_CELLS = 9 * 3 * 8


def promote_posterior(rows: list[dict]) -> list[dict]:
    """The denoiser baselines decode by argmax of their final posterior (published practice;
    protocol amendment 2026-09-29, decoding). Their `*_post` metrics become the reported ones and
    the state-argmax metrics are kept as `*_state`."""
    for r in rows:
        if r["arm"] in DENOISER:
            for m in r["per_nfe"].values():
                if "kmer_r_post" in m:
                    m["kmer_r_state"], m["marginal_kl_state"] = m["kmer_r"], m["marginal_kl"]
                    m["kmer_r"], m["marginal_kl"] = m["kmer_r_post"], m["marginal_kl_post"]
    return rows


def load_rows() -> tuple[list[dict], bool]:
    ho = [r for r in load_jsonl("heldout_*.jsonl", tier="heldout") if r.get("data") == "v2"]
    if len(ho) == N_HELDOUT_CELLS:
        return promote_posterior(ho), True
    return load_jsonl("endpoint_*_*.jsonl"), False


def load_jsonl(pattern: str, tier: str = "full") -> list[dict]:
    """Load sequence rows, keeping one row per (dataset, arm, seed).

    Result files are append-only, so re-entering a batch after a crash can append a second
    row for a cell that already ran. Averaging over those duplicates would weight some seeds
    twice. The last row wins, matching the dedup in `make_figures.py`.
    """
    uniq = {}
    for f in sorted(glob.glob(str(RES / pattern))):
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            # Full tier only: smoke runs write *_smoke.jsonl files that match the same glob, and
            # last-row-wins dedup would let a smoke row replace a full-tier one.
            if r.get("tier") != tier:
                continue
            uniq[(r["dataset"], r["arm"], r["seed"])] = r
    return list(uniq.values())


def paired(rows, ds: str, arm_a: str, arm_b: str, key: str, nfe: str = "100"):
    """Values for two arms aligned on the seeds they have in common.

    Arms do not always have the same seed set -- a partially completed batch leaves an arm
    with fewer seeds -- and subtracting two ragged vectors either raises or, worse, silently
    pairs seed 0 with seed 3. Returns (a, b) over the sorted common seeds.
    """
    A = {r["seed"]: r["per_nfe"][nfe][key] for r in rows
         if r["dataset"] == ds and r["arm"] == arm_a}
    B = {r["seed"]: r["per_nfe"][nfe][key] for r in rows
         if r["dataset"] == ds and r["arm"] == arm_b}
    s = sorted(set(A) & set(B))
    return np.array([A[i] for i in s]), np.array([B[i] for i in s])


def write_tex(name: str, body: str) -> None:
    TEX.mkdir(parents=True, exist_ok=True)
    (TEX / f"{name}.tex").write_text(body)


# ---------------------------------------------------------------- Boundary concentration


def boundary_stats() -> None:
    """Where the Modality 2 data actually sit relative to the boundary.

    The Section 4.7 explanation rests on these numbers, so they are computed rather than
    asserted, and they move if the preprocessing does.
    """
    f = REPO / "data" / "mnist_train.npy"
    if not f.exists():
        print("\n(MNIST not present; skipping boundary stats)"); return
    x = np.load(f).astype(np.float32) / 255.0
    x = x * 0.98 + 0.01                      # exactly the harness preprocessing
    lo = float((x < 0.05).mean())
    hi = float((x > 0.95).mean())
    g = x * (1 - x)
    print("\n" + "=" * 78)
    print("Modality 2 boundary concentration (drives the Section 4.7 explanation)")
    print("=" * 78)
    print(f"  below 0.05: {lo*100:.1f}%   above 0.95: {hi*100:.1f}%   "
          f"interior: {(1-lo-hi)*100:.1f}%")
    print(f"  mean |x-1/2| = {np.abs(x - 0.5).mean():.3f} (max 0.5)")
    print(f"  gain x(1-x): median {np.median(g):.4f} of a possible 0.25; "
          f"{float((g < 0.025).mean())*100:.1f}% of pixels below a tenth of max")
    NUMS["MTwoMeanDev"] = f"{np.abs(x - 0.5).mean():.3f}"
    NUMS["MTwoInteriorPct"] = f"{(1 - lo - hi)*100:.1f}"
    NUMS["MTwoThrottledPct"] = f"{float((g < 0.025).mean())*100:.1f}"
    NUMS["MTwoMedianGain"] = f"{np.median(g):.4f}"
    write_tex("m2_boundary",
              f"${lo*100:.1f}\\%$ of pixels below $0.05$ and ${hi*100:.1f}\\%$ above $0.95$, "
              f"leaving ${(1-lo-hi)*100:.1f}\\%$ interior")


# ---------------------------------------------------------------- Table: 4.2 FM vs diffusion


def table_fm_vs_diffusion() -> None:
    """Section 4.2. The family-selection evidence: quality against sampling budget, so the
    choice can be justified on the efficiency axis and not on a single number."""
    f = RES / "fm_vs_diffusion.jsonl"
    if not f.exists():
        print("\n(4.2 results not found)"); return
    rows = [json.loads(l) for l in open(f) if l.strip()]
    ks = sorted({r["k"] for r in rows})
    nfes = ["10", "100"]
    fams = (("flow_matching", "Flow matching"), ("diffusion", "Diffusion (VP, DDIM)"))
    print("\n" + "=" * 86)
    print("TABLE 2 — Modality 1: flow matching versus diffusion (marginal KL, lower better)")
    print("=" * 86)
    hdr = "  ".join(f"K={k} NFE={n}" for k in ks for n in nfes)
    print(f"{'family':>22} | {hdr}   x0 outside per K")
    print("-" * 86)
    # x_0-outside is reported per K: one column averaged over both alphabets describes neither.
    tex = ["\\begin{tabular}{l" + "r" * (len(ks) * (len(nfes) + 1)) + "}", "\\toprule",
           "Method & " + " & ".join(f"$K{{=}}{k}$, NFE {n}" for k in ks for n in nfes)
           + " & " + " & ".join(f"$x_0$ out, $K{{=}}{k}$" for k in ks) + " \\\\", "\\midrule"]
    seeds = {fam: len({r["seed"] for r in rows if r["family"] == fam}) for fam, _ in fams}
    NUMS["SynSeeds"] = str(min(seeds.values()))
    NUMS["SynKsmall"], NUMS["SynKlarge"] = str(ks[0]), str(ks[-1])
    for fam, label in fams:
        s = [r for r in rows if r["family"] == fam]
        if not s:
            continue
        cells, tcells, ocells = [], [], []
        for k in ks:
            sk = [r for r in s if r["k"] == k]
            for n in nfes:
                v = np.array([r["kl"][n] for r in sk])
                cells.append(f"{v.mean():>10.4f}")
                tcells.append(f"{v.mean():.4f} $\\pm$ {v.std():.4f}")
            ov = np.mean([r["x0_outside_rate_nfe100"] for r in sk])
            ocells.append(f"{ov:.3f}")
            tag = ("FM" if fam == "flow_matching" else "Diff") + \
                  ("small" if k == ks[0] else "large")
            NUMS[f"SynX{tag}"] = f"{ov:.3f}"
            NUMS[f"SynHundred{tag}"] = f"{np.mean([r['kl']['100'] for r in sk]):.3f}"
        print(f"{label:>22} | {'  '.join(cells)}   {'  '.join(ocells)}")
        tex.append(f"{label} & " + " & ".join(tcells + ocells) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    write_tex("fmdiff", "\n".join(tex))
    # The decision rule, recomputed rather than restated. The boot() calls and their order are
    # fixed: changing them moves every later interval drawn from the shared RNG.
    for k in ks:
        fm = np.array([r["kl"]["10"] for r in rows
                       if r["family"] == "flow_matching" and r["k"] == k])
        df = np.array([r["kl"]["10"] for r in rows
                       if r["family"] == "diffusion" and r["k"] == k])
        if len(fm) and len(fm) == len(df):
            m, lo, hi = boot(df - fm)
            NUMS["SynCI" + ("small" if k == ks[0] else "large")] = ci_tex(m, lo, hi, 3)
            print(f"  K={k}, NFE=10, diffusion minus FM: {m:+.4f} [{lo:+.4f}, {hi:+.4f}]"
                  f" -> {'FM' if lo > 0 else 'diffusion' if hi < 0 else 'no'} advantage")
    print("Both families leave the simplex at essentially the same rate, so non-tangency is")
    print("not a property of one objective.")


def table_fm_vs_diffusion_real() -> None:
    """Section 4.2 on the real Modality 1 data (protocol amendment 2026-09-24): the same two
    processes on the dilated CNN, 8 seeds, paired by seed.

    Uses its OWN generator. `boot()` draws from the module RNG, so adding calls to it anywhere
    before the other tables would silently move every interval printed after this one.
    """
    rng = np.random.default_rng(0)

    def boot_local(d, n=20000):
        d = np.asarray(d, float)
        b = [rng.choice(d, len(d), replace=True).mean() for _ in range(n)]
        return d.mean(), *np.percentile(b, [2.5, 97.5])

    def load(ds, fam):
        f = RES / f"fmdiff_real_{ds}_{fam}_{'heldout' if HELDOUT else 'full'}.jsonl"
        if not f.exists():
            return {}
        return {r["seed"]: r for r in map(json.loads, filter(str.strip, open(f)))}

    print("\n" + "=" * 86)
    print("TABLE 2b — Modality 1, real sequences: FM vs diffusion (marginal KL, lower better)")
    print("=" * 86)
    tex = ["\\begin{tabular}{lrrrrr}", "\\toprule",
           "Dataset & FM, NFE 10 & Diff., NFE 10 & Diff.$-$FM, NFE 10 & Diff.$-$FM, NFE 100 "
           "& $x_0$ out (FM / Diff.) \\\\", "\\midrule"]
    wins, n_seeds = [], set()
    for ds, k, name in DATASETS:
        fm, df = load(ds, "flow_matching"), load(ds, "diffusion")
        s = sorted(set(fm) & set(df))
        if not s:
            print(f"  ({ds}: no real-data FM-vs-diffusion results)"); return
        n_seeds.add(len(s))
        kl = lambda R, n: np.array([R[i]["per_nfe"][n]["marginal_kl"] for i in s])
        a10, b10, a100, b100 = kl(fm, "10"), kl(df, "10"), kl(fm, "100"), kl(df, "100")
        c10, c100 = boot_local(b10 - a10), boot_local(b100 - a100)
        for n in ("10", "50", "100"):
            wins.append(int((kl(fm, n) < kl(df, n)).sum()))
        xf = np.mean([fm[i]["per_nfe"]["100"]["x0_outside"] for i in s])
        xd = np.mean([df[i]["per_nfe"]["100"]["x0_outside"] for i in s])
        tag = {"dna_k4": "Dna", "protein_k20": "Prot", "codon_k64": "Codon"}[ds]
        NUMS[f"RealCI{tag}"] = ci_tex(*c10, 3)
        NUMS[f"RealCIH{tag}"] = ci_tex(*c100, 3)
        NUMS[f"RealX{tag}FM"], NUMS[f"RealX{tag}Diff"] = f"{xf:.3f}", f"{xd:.3f}"
        if ds == "protein_k20":
            r = lambda R: np.mean([R[i]["per_nfe"]["10"]["kmer_r"] for i in s])
            # Math mode, or a negative value typesets with a text hyphen.
            NUMS["RealProtKmerFM"], NUMS["RealProtKmerDiff"] = f"${r(fm):+.3f}$", f"${r(df):+.3f}$"
        tex.append(f"{name} ($K{{=}}{k}$) & {a10.mean():.4f} $\\pm$ {a10.std():.4f} & "
                   f"{b10.mean():.4f} $\\pm$ {b10.std():.4f} & "
                   f"$\\mathbf{{{c10[0]:+.3f}}}$ [{c10[1]:+.3f}, {c10[2]:+.3f}] & "
                   f"{c100[0]:+.3f} [{c100[1]:+.3f}, {c100[2]:+.3f}] & {xf:.3f} / {xd:.3f} \\\\")
        print(f"  {name:>8} K={k:<3} NFE10 diff-FM {c10[0]:+.4f} [{c10[1]:+.4f}, {c10[2]:+.4f}]"
              f"  NFE100 {c100[0]:+.4f} [{c100[1]:+.4f}, {c100[2]:+.4f}]  x0 out {xf:.3f}/{xd:.3f}")
    tex += ["\\bottomrule", "\\end{tabular}"]
    write_tex("fmdiff_real", "\n".join(tex))
    assert len(n_seeds) == 1, f"ragged seed sets across datasets: {n_seeds}"
    NUMS["RealSeeds"] = str(n_seeds.pop())
    NUMS["RealMinWins"] = str(min(wins))
    print(f"  FM better on at least {min(wins)}/{NUMS['RealSeeds']} seeds at every NFE and dataset")


def prose_numbers(rows) -> None:
    """Every number the paper quotes in running text, computed from the stored results.

    Each value here is a macro in `paper/tables/nums.tex`. Own RNG, called last, so no other
    table's interval moves.
    """
    old_rows = load_jsonl("endpoint_*_*.jsonl")        # training-reference v1 rows
    rng = np.random.default_rng(0)

    def boot_local(d, n=20000):
        d = np.asarray(d, float)
        b = [rng.choice(d, len(d), replace=True).mean() for _ in range(n)]
        return d.mean(), *np.percentile(b, [2.5, 97.5])

    # Later-added intervals draw from their own stream, so no existing interval moves.
    rng_new = np.random.default_rng(1)

    def boot_new(d, n=20000):
        d = np.asarray(d, float)
        b = [rng_new.choice(d, len(d), replace=True).mean() for _ in range(n)]
        return d.mean(), *np.percentile(b, [2.5, 97.5])

    def g(ds, arm, key, nfe="100"):
        return np.array([r["per_nfe"][nfe][key] for r in rows
                         if r["dataset"] == ds and r["arm"] == arm])

    def diff(ds, a, b, nfe="100"):
        x, y = paired(rows, ds, a, b, "kmer_r", nfe)
        return x - y

    def sci(v):
        e = int(np.floor(np.log10(abs(v))))
        return f"{v / 10**e:.2f}\\times10^{{{e}}}"

    DNA, PROT, COD = (d for d, _, _ in DATASETS)
    tags = {DNA: "Dna", PROT: "Prot", COD: "Codon"}
    trip = lambda f: " / ".join(f(ds) for ds, _, _ in DATASETS)

    # --- the headline measurement: base arm, NFE=100
    for ds, t in tags.items():
        sv = g(ds, "unconstrained", "step_violation").mean()
        NUMS[f"BaseViol{t}"], NUMS[f"BaseViolTwo{t}"] = f"{sv:.3f}", f"{sv:.2f}"
        r = g(ds, "unconstrained", "kmer_r").mean()
        NUMS[f"BaseKmer{t}"], NUMS[f"BaseKmerTwo{t}"] = f"{r:.3f}", f"{r:.2f}"
    NUMS["BaseViolPctDna"] = f"{g(DNA, 'unconstrained', 'step_violation').mean()*100:.0f}"
    NUMS["BaseXMin"] = f"{min(g(d, 'unconstrained', 'x0_outside').mean() for d in tags):.3f}"

    # --- decomposition (mult-only and center-only vs base; mult-only vs full replicator)
    NUMS["MultCIProt"] = ci_tex(*boot_local(diff(PROT, "mult_only", "unconstrained")), 3)
    NUMS["MultCICodon"] = ci_tex(*boot_local(diff(COD, "mult_only", "unconstrained")), 3)
    NUMS["MultCICodonTen"] = ci_tex(*boot_local(diff(COD, "mult_only", "unconstrained", "10")), 3)
    NUMS["MultMeanCodonTen"] = f"{diff(COD, 'mult_only', 'unconstrained', '10').mean():+.3f}"
    NUMS["MultMeanCodon"] = f"{diff(COD, 'mult_only', 'unconstrained').mean():+.3f}"
    NUMS["DropCenterCICodon"] = ci_tex(*boot_local(diff(COD, "mult_only", "hard")), 3)
    NUMS["MultViolMax"] = f"{max(g(d, 'mult_only', 'step_violation').mean() for d in tags):.4f}"
    # "Exactly zero" holds only from NFE 50: explicit Euler overshoots at NFE 10.
    v10 = [g(d, "mult_only", "step_violation", "10").mean() * 100 for d in tags]
    NUMS["MultViolTenLo"], NUMS["MultViolTenHi"] = f"{min(v10):.1f}", f"{max(v10):.1f}"
    for n_ in ("10", "50"):
        NUMS[f"BaseViolDna{'Ten' if n_ == '10' else 'Fifty'}"] = \
            f"{g(DNA, 'unconstrained', 'step_violation', n_).mean():.2f}"
    for n_, w_ in (("10", "Ten"), ("50", "Fifty"), ("100", "Hundred")):
        NUMS[f"MultMeanProt{w_}"] = f"{diff(PROT, 'mult_only', 'unconstrained', n_).mean():+.3f}"
    h10, h100 = diff(PROT, "hard", "unconstrained", "10").mean(), \
        diff(PROT, "hard", "unconstrained").mean()
    NUMS["HardTen"], NUMS["HardHundred"] = f"{h10:+.3f}", f"{h100:+.3f}"
    NUMS["HardDecay"] = f"{h10 / h100:.1f}"
    NUMS["ReweightCIProt"] = ci_tex(*boot_local(diff(PROT, "endpoint_rw", "unconstrained")), 3)

    # --- external baselines (means; the table carries the spreads)
    km = lambda arm: lambda ds: f"{g(ds, arm, 'kmer_r').mean():.3f}"
    vi = lambda arm: lambda ds: f"{g(ds, arm, 'step_violation').mean():.3f}"
    FA = fisher_arm(rows)
    NUMS["FisherKmer"], NUMS["OursKmer"] = trip(km(FA)), trip(km("mult_only"))
    GA, DA = ext_arm(rows, "gumbel"), ext_arm(rows, "dirichlet")
    NUMS["BaselinesFixed"] = "1" if (GA, DA) == ("gumbel_ce", "dirichlet_ce") else "0"
    NUMS["FisherViol"], NUMS["GumbelViol"] = trip(vi(FA)), trip(vi(GA))
    NUMS["DirichletViol"] = trip(vi(DA))
    NUMS["GumbelKmer"], NUMS["DirichletKmer"] = trip(km(GA)), trip(km(DA))
    go = lambda ds, arm, key: np.array([r["per_nfe"]["100"][key] for r in old_rows
                                        if r["dataset"] == ds and r["arm"] == arm])
    kmo = lambda arm: lambda ds: f"{go(ds, arm, 'kmer_r').mean():.3f}"
    vio = lambda arm: lambda ds: f"{go(ds, arm, 'step_violation').mean():.3f}"
    NUMS["GumbelKmerOld"], NUMS["DirichletKmerOld"] = trip(kmo("gumbel")), trip(kmo("dirichlet"))
    NUMS["GumbelViolOld"], NUMS["DirichletViolOld"] = trip(vio("gumbel")), trip(vio("dirichlet"))
    if HELDOUT:
        # Denoiser arms decoded from the STATE (the rule registered first; appendix only).
        st = lambda arm: lambda ds: f"{g(ds, arm, 'kmer_r_state').mean():.3f}"
        NUMS["GumbelKmerState"], NUMS["DirichletKmerState"] = trip(st(GA)), trip(st(DA))
        # Memorisation gap: 3-mer r against the training set minus against held-out, NFE 100.
        for arm, lab in (("unconstrained", "Base"), ("mult_only", "Ours")):
            gap = [(g(ds, arm, "kmer_r_train") - g(ds, arm, "kmer_r")).mean() for ds in tags]
            NUMS[f"TrainGap{lab}"] = " / ".join(f"{x:+.3f}" for x in gap)
    if DA == "dirichlet_ce":
        cf = [r["per_nfe"][n_].get("cfactor_nonfinite", 0.0) for r in rows if r["arm"] == DA
              for n_ in ("10", "50", "100")]
        NUMS["DirichletCNonfinite"] = sci(max(cf)) if max(cf) > 0 else "0"
    # Each external method against ours and against the base, 3-mer r at NFE 100, both ways.
    for arm, lab in ((GA, "Gumbel"), (DA, "Dirichlet"), (FA, "Fisher")):
        for ds, t in tags.items():
            dd = diff(ds, arm, "mult_only")
            NUMS[f"{lab}VsOurs{t}"] = ci_tex(*boot_new(dd), 3)
            NUMS[f"{lab}VsOursPos{t}"] = str(int((dd > 0).sum()))
            NUMS[f"{lab}VsBase{t}"] = ci_tex(*boot_new(diff(ds, arm, "unconstrained")), 3)
    NUMS["FisherKmerOld"], NUMS["FisherViolOld"] = trip(kmo("fisher")), trip(vio("fisher"))
    NUMS["FisherFixed"] = "1" if FA == "fisher_sph" else "0"
    NUMS["FisherCodonOld"], NUMS["FisherCodonNew"] = kmo("fisher")(COD), km(FA)(COD)
    if FA == "fisher_sph" and "step_violation_euclid" in next(
            r for r in rows if r["arm"] == FA)["per_nfe"]["100"]:
        NUMS["FisherViolEuclid"] = trip(vi_e := (lambda ds: f"{g(ds, FA, 'step_violation_euclid').mean():.3f}"))
    NUMS["DirichletKmerCodon"] = km(DA)(COD)
    NUMS["DirichletKLCodon"] = f"{g(COD, DA, 'marginal_kl').mean():.2f}"
    for arm, lab in ((GA, "Gumbel"), (DA, "Dirichlet"), (FA, "Fisher")):
        NUMS[f"{lab}KL"] = trip(lambda ds, a=arm: f"{g(ds, a, 'marginal_kl').mean():.3f}")
    fv = [go(d, "fisher", "step_violation").mean() * 100 for d in tags]   # the buggy arm
    NUMS["FisherViolLo"], NUMS["FisherViolHi"] = f"{min(fv):.1f}", f"{max(fv):.0f}"
    # Five bootstrap draws on the old `gumbel`/`dirichlet` rows whose results are unused. They
    # keep the shared stream, and so every later interval in this function, where it is.
    for arm, want in (("gumbel", (DNA, PROT, COD)), ("dirichlet", (DNA, PROT))):
        for ds in want:
            if not HELDOUT:
                boot_local(diff(ds, "mult_only", arm))

    # Which method has the highest mean 3-mer r at each K, computed so the text cannot drift.
    cand = {"Dirichlet FM": DA, "Fisher-Flow": FA, "Gumbel-Softmax FM": GA,
            "linear FM": "unconstrained", "ours": "mult_only"}
    for ds, t in tags.items():
        NUMS[f"BestKmer{t}"] = max(cand, key=lambda c: g(ds, cand[c], "kmer_r").mean())

    # Robustness of the headline intervals: Student-t CI and the count of seeds favouring
    # the effect, beside the registered percentile bootstrap.
    for key, dd in (("MultProt", diff(PROT, "mult_only", "unconstrained")),
                    ("MultCodon", diff(COD, "mult_only", "unconstrained")),
                    ("MultCodonTen", diff(COD, "mult_only", "unconstrained", "10")),
                    ("DropCenterCodon", diff(COD, "mult_only", "hard")),
                    ("ReweightProt", diff(PROT, "endpoint_rw", "unconstrained"))):
        NUMS[f"{key}T"] = ci_tex(*t_ci(dd), 3)
        NUMS[f"{key}Pos"] = str(int((dd > 0).sum()))
        NUMS[f"{key}N"] = str(len(dd))

    # --- step-size sweep (probe_tangency_violation)
    sw = json.load(open(RES / "probe_tangency_violation.json"))
    sr = sw["rows"]
    rates = [r["step_rate"] for r in sr]
    NUMS["SweepK"] = str(sw["k"])
    NUMS["SweepNfeMin"], NUMS["SweepNfeMax"] = str(sr[0]["nfe"]), str(sr[-1]["nfe"])
    NUMS["SweepRatio"] = str(sr[-1]["nfe"] // sr[0]["nfe"])
    NUMS["SweepRateLo"], NUMS["SweepRateHi"] = f"{min(rates):.2f}", f"{max(rates):.2f}"
    NUMS["SweepMassHi"], NUMS["SweepMassLo"] = sci(sr[0]["mass_per_step"]), \
        sci(sr[-1]["mass_per_step"])

    # --- guidance
    gd = json.load(open(RES / "guidance_modality1.json"))
    gd["rows"] = [r for r in gd["rows"] if r["rule"] != "mirror"]      # duplicate of natural
    G = lambda rule, gam: next(r for r in gd["rows"]
                               if r["rule"] == rule and abs(r["gamma"] - gam) < 1e-9)
    rules = sorted({r["rule"] for r in gd["rows"]})
    NUMS["GuidPrior"] = f"{gd['prior_mass_S']:.4f}"
    NUMS["GuidHitMax"] = f"{min(max(r['hit_rate'] for r in gd['rows'] if r['rule'] == u) for u in rules):.3f}"
    # KL to the ideal law at gamma = 8, where the text says every rule has saturated.
    kl8 = [G(u, 8.0)["kl_to_ideal"] for u in rules]
    NUMS["GuidKLLo"], NUMS["GuidKLHi"] = f"{min(kl8):.3f}", f"{max(kl8):.3f}"
    mt, me = G("tangent", 8.0)["violation_mass"], G("euclidean", 8.0)["violation_mass"]
    NUMS["GuidMassRatio"] = f"{mt / me:.0f}"
    NUMS["GuidMassTan"], NUMS["GuidMassEuc"] = sci(mt), sci(me)
    NUMS["GuidViolEuc"] = f"{G('euclidean', 8.0)['violation_rate']:.2f}"
    NUMS["GuidViolNat"] = f"{G('natural', 8.0)['violation_rate']:.2f}"
    first = lambda u: min(r["gamma"] for r in gd["rows"] if r["rule"] == u and r["hit_rate"] >= 0.999)
    NUMS["GuidStrengthRatio"] = f"{first('natural') / first('euclidean'):.0f}"
    NUMS["GuidViolZero"] = f"{G('euclidean', 0.0)['violation_rate']:.2f}"
    NUMS["GuidHitZero"] = f"{G('euclidean', 0.0)['hit_rate']:.4f}"
    NUMS["GuidViolTanEight"] = f"{G('tangent', 8.0)['violation_rate']:.3f}"
    NUMS["GuidViolNatEightThree"] = f"{G('natural', 8.0)['violation_rate']:.3f}"

    # --- the synthetic cost of the hard constraint (constrained_field): replicator vs base KL
    cf = [json.loads(l) for l in open(RES / "constrained_field.jsonl") if l.strip()]
    kmax = max(r["k"] for r in cf)
    kl = lambda prm: np.mean([r["kl"]["100"] for r in cf if r["k"] == kmax and r["param"] == prm])
    NUMS["SynHardKLRatio"] = f"{kl('replicator') / kl('unconstrained'):.0f}"
    NUMS["SynHardK"] = str(kmax)

    # --- Modality 2: the per-seed picture behind the preregistered claim
    m2 = []
    m2f, m2fixed = m2_files()
    for f in m2f:
        m2 += [r for r in (json.loads(l) for l in open(f) if l.strip())][1:]
    fd = lambda arm: {r["seed"]: r["fd"]["100"] for r in m2 if r.get("arm") == arm}
    base, ours = fd("unconstrained"), fd("mult")
    s = sorted(set(base) & set(ours))
    d = np.array([base[i] - ours[i] for i in s])          # positive = ours lowers FD
    NUMS["MTwoViolOurs"] = f"{max(r['violation_nfe100'] for r in m2 if r.get('arm') == 'mult'):.4f}"
    NUMS["MTwoWins"] = str(int((d > 0).sum()))
    fd10 = lambda arm: {r["seed"]: r["fd"]["10"] for r in m2 if r.get("arm") == arm}
    b10, o10 = fd10("unconstrained"), fd10("mult")
    NUMS["MTwoFDTenWins"] = str(sum(o10[i] < b10[i] for i in s))
    col = lambda arm, key: {r["seed"]: r[key] for r in m2 if r.get("arm") == arm}
    for key, tag in (("density_nfe100", "Dens"), ("coverage_nfe100", "Cov")):
        o_, b_ = col("mult", key), col("unconstrained", key)
        NUMS[f"MTwo{tag}Losses"] = str(sum(o_[i] < b_[i] for i in s))
        NUMS[f"MTwo{tag}Ours"], NUMS[f"MTwo{tag}Base"] = \
            f"{np.mean(list(o_.values())):.3f}", f"{np.mean(list(b_.values())):.3f}"
    accs = [json.loads(open(f).readline())["classifier_test_accuracy"] for f in m2f]
    NUMS["MTwoAccLo"], NUMS["MTwoAccHi"] = f"{min(accs)*100:.1f}", f"{max(accs)*100:.1f}"
    NUMS["MTwoAcc"] = f"{np.mean(accs)*100:.1f}"
    NUMS["MTwoNClassifiers"] = str(len(set(accs)) if m2fixed else len(accs))
    NUMS["MTwoFixedClf"] = "1" if m2fixed else "0"
    # The old per-process-extractor batch, for the disclosure sentence.
    old = [json.loads(open(f).readline())["classifier_test_accuracy"]
           for f in sorted(glob.glob(str(RES / "modality2_unet_*.jsonl")))]
    NUMS["MTwoOldAccLo"], NUMS["MTwoOldAccHi"] = f"{min(old)*100:.2f}", f"{max(old)*100:.2f}"
    NUMS["MTwoOldN"] = str(len(old))
    # Robustness: t-interval and seed count for the preregistered primary and for mirror.
    NUMS["MTwoClaimT"] = f"${t_ci(d)[0]:+.1f}\\,[{t_ci(d)[1]:+.1f}, {t_ci(d)[2]:+.1f}]$"
    mir = fd("mirror")
    dm = np.array([mir[i] - ours[i] for i in s if i in mir])
    NUMS["MTwoVsMirrorT"] = f"${t_ci(dm)[0]:+.1f}\\,[{t_ci(dm)[1]:+.1f}, {t_ci(dm)[2]:+.1f}]$"
    NUMS["MTwoMirrorWins"] = str(int((dm < 0).sum()))
    # Which seed the uncurated sample grid (fig7) shows: make_figures takes the first tagged file.
    pre = "samples_m2fixed_" if m2fixed else "samples_"
    tagged = sorted(glob.glob(str(RES / f"{pre}mult_s*.npy")))
    NUMS["MTwoSampleSeed"] = tagged[0].rsplit("_s", 1)[1].split(".")[0] if tagged else "?"
    # Each value in its own math group so the line can break between them.
    NUMS["MTwoPerSeed"] = ", ".join(f"${v:+.1f}$" for v in d)
    NUMS["MTwoDropSeedZero"] = ci_tex(*boot_local(d[1:]), 1)
    NUMS["MTwoDropSeedZeroMean"] = f"{d[1:].mean():+.1f}"
    sd = lambda arm: f"{np.std(list(fd(arm).values())):.1f}"
    NUMS["MTwoSdOurs"], NUMS["MTwoSdBase"], NUMS["MTwoSdMirror"] = sd("mult"), \
        sd("unconstrained"), sd("mirror")
    means = [np.mean(list(fd(a).values())) for a in ("unconstrained", "reflected", "dynthresh")]
    NUMS["MTwoClampSpread"] = f"{int(np.ceil(max(means) - min(means)))}"

    # --- appendix: commitment probe, Gamma sampler check, Modality 1 configuration
    dl = json.load(open(RES / "probe_decode_lock.json"))
    for ds, t in tags.items():
        NUMS[f"Commit{t}"] = f"{dl[ds]['mean_lock_t']:.3f}"
    settled = [dl[ds]["frac_locked_by"]["0.25"] * 100 for ds in tags]
    NUMS["CommitSettledLo"], NUMS["CommitSettledHi"] = f"{min(settled):.0f}", f"{max(settled):.0f}"
    lock = np.mean([dl[ds]["mean_lock_t"] for ds in tags])
    NUMS["CommitStep"], NUMS["CommitWasted"] = f"{lock*100:.0f}", f"{100 - lock*100:.0f}"
    vg = json.load(open(RES / "validate_gamma.json"))["rows"]
    NUMS["GammaShapes"] = "/".join(f"{r['shape']:.1f}" for r in vg)
    NUMS["GammaKSp"] = "/".join(f"{r['p']:.3f}" for r in vg)
    import sys
    sys.path.insert(0, str(REPO / "experiments"))
    from real_sequences import TIERS          # import-safe: nothing is written at import
    c = TIERS[next(iter({r["tier"] for r in rows}))]
    NUMS["MOneSeeds"] = str(len({r["seed"] for r in rows if r["arm"] == "unconstrained"}))
    NUMS["MOneBatch"], NUMS["MOneSteps"] = str(c.get("batch", 128)), str(c["steps"])
    NUMS["MOneWidth"], NUMS["MOneLayers"] = str(c["hidden"]), str(c["layers"])
    NUMS["MOneEval"], NUMS["MOneTrain"] = f"{c['n_eval']:,}".replace(",", "{,}"), \
        f"{c['n_train']:,}".replace(",", "{,}")
    NUMS["MOneHeldout"] = f"{c.get('n_heldout', 0):,}".replace(",", "{,}")
    from real_sequences import DilatedCNN
    npar = [sum(q.numel() for q in DilatedCNN(k, c["hidden"], c["layers"]).parameters())
            for _, k, _ in DATASETS]
    NUMS["MOneParamsLo"], NUMS["MOneParamsHi"] = f"{min(npar)/1e3:.0f}K", f"{max(npar)/1e3:.0f}K"
    import modality2_unet as M2U                 # import-safe: writes only inside main()
    m2c = M2U.TIERS["full"]
    NUMS["MTwoParams"] = f"{sum(q.numel() for q in M2U.UNet(m2c['ch']).parameters())/1e6:.2f}M"
    NUMS["MTwoTrain"] = f"{m2c['n_train']:,}".replace(",", "{,}")
    NUMS["MTwoEval"] = f"{m2c['n_eval']:,}".replace(",", "{,}")
    NUMS["MTwoSteps"], NUMS["MTwoWidth"] = str(m2c["steps"]), str(m2c["ch"])
    wall = [json.loads(l)["wall_s"] for f in glob.glob(str(
                RES / f"fmdiff_real_*_{'heldout' if HELDOUT else 'full'}.jsonl"))
            for l in open(f) if l.strip()]
    NUMS["MOneCellMin"] = f"{np.mean(wall)/60:.1f}"
    NUMS["GuidN"] = f"{int(sum(gd['rows'][0]['counts'])):,}".replace(",", "{,}")
    NUMS["GuidNfe"], NUMS["GuidK"] = str(gd["nfe"]), str(gd["k"])
    # --- the step-violation rate counts a step once ANY of 128*K coordinates is negative, so
    # it rises with K mechanically. The implied per-coordinate rate, assuming independence:
    # 1 - (1 - r)^(1/(128 K)). Undefined when r = 1 (codon), so reported for DNA and protein.
    for ds, t in ((DNA, "Dna"), (PROT, "Prot")):
        k = next(k for d_, k, _ in DATASETS if d_ == ds)
        r = g(ds, "unconstrained", "step_violation").mean()
        NUMS[f"PerCoordViol{t}"] = sci(1.0 - (1.0 - r) ** (1.0 / (128 * k)))
    # Residual violation of the constrained arms at NFE 10 is 1/NFE: one step per sample.
    NUMS["ResidualOneStep"] = str(sum(
        abs(g(ds, "mult_only", "step_violation", "10").mean() - 0.1) < 2e-3 for ds in (DNA, PROT)))

    # --- the larger-backbone run (tier "gpu": width 256, 12 layers, batch 512, 3 seeds).
    # No RNG: means only.
    big = [json.loads(l) for f in sorted(glob.glob(str(RES / "real_sequences_*_k*.jsonl")))
           if "smoke" not in f for l in open(f) if l.strip()]
    big = [r for r in big if r.get("tier") == "gpu"]
    if big:
        bg = lambda ds, arm: np.mean([r["per_nfe"]["100"]["kmer_r"] for r in big
                                      if r["dataset"] == ds and r["arm"] == arm])
        NUMS["BigSeeds"] = str(len({r["seed"] for r in big}))
        NUMS["BigBaseProt"], NUMS["BigBaseCodon"] = f"{bg(PROT, 'unconstrained'):.3f}", \
            f"{bg(COD, 'unconstrained'):.3f}"
        NUMS["BigHardGainProt"] = f"{bg(PROT, 'hard') - bg(PROT, 'unconstrained'):+.3f}"
        NUMS["BigHardGainCodon"] = f"{bg(COD, 'hard') - bg(COD, 'unconstrained'):+.3f}"
        NUMS["OursKmerProt"] = f"{g(PROT, 'mult_only', 'kmer_r').mean():.3f}"
    NUMS["OursKmerCodon"] = f"{g(COD, 'mult_only', 'kmer_r').mean():.3f}"
    print("\n  prose macros: " + ", ".join(sorted(NUMS)))


def ci_tex(m: float, lo: float, hi: float, d: int) -> str:
    return f"${m:+.{d}f}\\,[{lo:+.{d}f}, {hi:+.{d}f}]$"


# Numbers quoted in running text are emitted as macros, so prose cannot drift from the table.
NUMS: dict[str, str] = {}


def write_nums() -> None:
    body = ["% Generated by scripts/make_tables.py. Do not edit."]
    body += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(NUMS.items())]
    write_tex("nums", "\n".join(body))


# ---------------------------------------------------------------- Table: 4.3 guidance


def table_guidance() -> None:
    """Section 4.3. Reports the guided target metric AND an off-target quality metric, so
    guidance is not judged by the score it optimises."""
    f = RES / "guidance_modality1.json"
    if not f.exists():
        print("\n(4.3 results not found)"); return
    d = json.load(open(f))
    # "mirror" runs the identical replicator step as "natural" (guidance_modality1.py), so it is
    # not a fourth rule; its rows are dropped from the tables.
    d["rows"] = [r for r in d["rows"] if r["rule"] != "mirror"]
    rows = d["rows"]
    print("\n" + "=" * 86)
    print(f"TABLE 3 — Modality 1: guidance. Target set S={d['S']} of K={d['k']}, "
          f"NFE={d['nfe']}, prior mass {d['prior_mass_S']:.5f}")
    print("=" * 86)
    print(f"{'rule':>12} {'gamma':>7} | {'hit rate':>10} {'KL to ideal':>12} "
          f"{'step viol.':>11}")
    print("-" * 60)
    full = ["\\begin{tabular}{lrrrr}", "\\toprule",
            "Rule & $\\gamma$ & Hit rate $\\uparrow$ & KL to ideal $\\downarrow$ "
            "& Step viol. $\\downarrow$ \\\\", "\\midrule"]
    last = None
    for r in rows:
        if last is not None and r["rule"] != last:
            full.append("\\midrule")
        last = r["rule"]
        print(f"{r['rule']:>12} {r['gamma']:>7.1f} | {r['hit_rate']:>10.5f} "
              f"{r['kl_to_ideal']:>12.4f} {r['violation_rate']:>11.4f}")
        full.append(f"{r['rule']} & {r['gamma']:.1f} & {r['hit_rate']:.4f} & "
                    f"{r['kl_to_ideal']:.4f} & {r['violation_rate']:.4f} \\\\")
    full += ["\\bottomrule", "\\end{tabular}"]
    write_tex("guidance_full", "\n".join(full))

    # The main-text version: one unguided row, then each rule at a moderate and a strong
    # setting. All 24 rows go to the appendix; the story is the columns, not the sweep.
    def get(rule, gamma):
        return next((r for r in rows
                     if r["rule"] == rule and abs(r["gamma"] - gamma) < 1e-9), None)

    rules = sorted({r["rule"] for r in rows}, key=lambda x: [q["rule"] for q in rows].index(x))
    tex = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
           "& \\multicolumn{3}{c}{$\\gamma = 1$} & \\multicolumn{3}{c}{$\\gamma = 8$} \\\\",
           "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
           "Rule & Hit $\\uparrow$ & KL $\\downarrow$ & Viol. $\\downarrow$ "
           "& Hit $\\uparrow$ & KL $\\downarrow$ & Viol. $\\downarrow$ \\\\", "\\midrule"]
    u = get(rules[0], 0.0)
    if u:
        tex.append(f"Unguided ($\\gamma{{=}}0$) & {u['hit_rate']:.4f} & "
                   f"{u['kl_to_ideal']:.2f} & {u['violation_rate']:.3f} & --- & --- & --- \\\\")
        tex.append("\\midrule")
    for rule in rules:
        a, b = get(rule, 1.0), get(rule, 8.0)
        if not (a and b):
            continue
        tex.append(f"{rule} & {a['hit_rate']:.4f} & {a['kl_to_ideal']:.3f} & "
                   f"{a['violation_rate']:.3f} & {b['hit_rate']:.4f} & "
                   f"{b['kl_to_ideal']:.3f} & {b['violation_rate']:.3f} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    write_tex("guidance", "\n".join(tex))
    base = [r for r in rows if r["gamma"] == 0.0]
    if base:
        print(f"\nUnguided hit rate {base[0]['hit_rate']:.5f} against a prior mass of "
              f"{d['prior_mass_S']:.5f}; guidance raises it by orders of magnitude, and the")
        print("cost is paid in the step-violation rate, which rises with gamma.")


# ---------------------------------------------------------------- Table: 4.5 external


def table_baselines(rows) -> None:
    """Section 4.5. At least three recent external methods plus the internal base model,
    under one protocol. Reported whichever way it falls."""
    ext = ((ext_arm(rows, "dirichlet"), "Dirichlet FM~\\citep{stark2024dirichlet}"),
           (fisher_arm(rows), "Fisher-Flow~\\citep{davis2024fisher}"),
           (ext_arm(rows, "gumbel"), "Gumbel-Softmax FM~\\citep{tang2025gumbel}"),
           ("unconstrained", "Linear FM (internal base)"),
           ("mult_only", "\\textbf{Ours}: $v = x \\odot w$"))

    def g(ds, arm, key, nfe="100"):
        return np.array([r["per_nfe"][nfe][key] for r in rows
                         if r["dataset"] == ds and r["arm"] == arm])

    print("\n" + "=" * 92)
    print("TABLE 5 — Modality 1: comparison with recent methods (3-mer r, higher better)")
    print("=" * 92)
    print(f"{'method':>26} | " + "  ".join(f"{lab:>13}" for _, _, lab in DATASETS)
          + f" {'viol.':>8} {'NFE=10':>8}")
    print("-" * 92)
    # Violation is reported per K: averaging a K=4 rate with a K=64 rate describes neither.
    tex = ["\\begin{tabular}{lrrrr}", "\\toprule",
           "& \\multicolumn{3}{c}{3-mer $r$ $\\uparrow$ (NFE 100)} & Step viol. $\\downarrow$ \\\\",
           "\\cmidrule(lr){2-4}",
           "Method & " + " & ".join(f"{lab} ($K{{=}}{k}$)" for _, k, lab in DATASETS)
           + " & $K{=}4/20/64$ \\\\", "\\midrule"]
    for arm, label in ext:
        cells, tcells, vio, r10 = [], [], [], []
        for ds, _, _ in DATASETS:
            v = g(ds, arm, "kmer_r")
            if not len(v):
                cells, tcells = [], []
                break
            cells.append(f"{v.mean():>7.3f}+-{v.std():<5.3f}")
            tcells.append(f"{v.mean():.3f} $\\pm$ {v.std():.3f}")
            vio.append(g(ds, arm, "step_violation").mean())
            r10.append(g(ds, arm, "kmer_r", "10").mean())
        if not cells:
            continue
        if arm == "unconstrained":
            tex.append("\\midrule")
        print(f"{arm:>26} | " + "  ".join(cells)
              + f" {np.mean(vio):>8.3f} {np.mean(r10):>8.3f}")
        tex.append(f"{label} & " + " & ".join(tcells)
                   + " & " + "/".join(f"{x:.2f}" for x in vio) + " \\\\")
    uni = unigram_reference()
    tex.append("\\midrule")
    tex.append("\\emph{Reference: unigram sampler (no model)} & "
               + " & ".join(f"{uni[ds][0]:.3f} $\\pm$ {uni[ds][1]:.3f}" for ds, _, _ in DATASETS)
               + " & --- \\\\")
    for ds, _, lab in DATASETS:
        NUMS[f"Uni{lab}"] = f"{uni[ds][0]:.3f}"
        NUMS[f"UniKL{lab}"] = f"{uni[ds][2]:.3f}"
    tex += ["\\bottomrule", "\\end{tabular}"]
    write_tex("baselines", "\n".join(tex))
    # Where we stand against each external method, stated in both directions.
    print()
    for arm, label in ext[:3]:
        for ds, k, lab in DATASETS:
            ours, them = paired(rows, ds, "mult_only", arm, "kmer_r")
            if not len(ours):
                continue
            m, lo, hi = boot(ours - them)
            side = "better" if lo > 0 else "worse" if hi < 0 else "indistinguishable"
            print(f"  ours vs {arm:>10} on {lab:>8} (K={k:>2}): "
                  f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]  ours is {side}")


# ---------------------------------------------------------------- Table: 4.4 decomposition


def table_decomposition(rows) -> None:
    """Section 4.4, emitted as LaTeX rather than hand-typed.

    Bolding marks a paired interval against the base arm that excludes zero, and is COMPUTED
    here: a hand-marked table silently stops being true the moment the data move.
    """
    # Short labels: the table shares a row with the guidance table in the 5-page paper.
    arms = (("unconstrained", "base $w$"),
            ("mult_only",     "mult $x \\odot w$"),
            ("center_only",   "center $w - \\bar{w}$"),
            ("hard",          "full replicator"))

    def mean(ds, arm, key):
        v = [r["per_nfe"]["100"][key] for r in rows
             if r["dataset"] == ds and r["arm"] == arm]
        return float(np.mean(v)) if v else None

    tex = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
           "& \\multicolumn{3}{c}{3-mer $r$} & \\multicolumn{3}{c}{step violation} \\\\",
           "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
           "Arm & $K{=}4$ & $K{=}20$ & $K{=}64$ & $K{=}4$ & $K{=}20$ & $K{=}64$ \\\\",
           "\\midrule"]
    print("\n" + "=" * 78)
    print("TABLE 4 — Decomposition of the replicator form (8 seeds, NFE=100)")
    print("=" * 78)
    for arm, label in arms:
        if mean(DATASETS[0][0], arm, "kmer_r") is None:
            continue
        cells = []
        for ds, _, _ in DATASETS:
            m = mean(ds, arm, "kmer_r")
            if arm == "unconstrained":
                cells.append(f"{m:.3f}")
            else:
                a, b = paired(rows, ds, arm, "unconstrained", "kmer_r")
                _, lo, hi = boot(a - b) if len(a) else (0, -1, 1)
                cells.append(f"\\textbf{{{m:.3f}}}" if lo > 0 or hi < 0 else f"{m:.3f}")
        for ds, _, _ in DATASETS:
            v = mean(ds, arm, "step_violation")
            cells.append(f"\\textbf{{{v:.4f}}}" if v < 0.001 else f"{v:.3f}")
        tex.append(f"{label} & " + " & ".join(cells) + " \\\\")
        print(f"  {arm:>14} | " + "  ".join(c.replace('\\textbf{', '*').replace('}', '')
                                            for c in cells))
    tex += ["\\bottomrule", "\\end{tabular}"]
    write_tex("decomp", "\n".join(tex))
    print("Bold = paired 95% interval vs base excludes zero (quality) or exact 0 (violation).")


# ---------------------------------------------------------------- Table: K-scaling (headline)


def table_kscaling(rows) -> None:
    def g(ds, arm, key):
        return np.array([r["per_nfe"]["100"][key] for r in rows
                         if r["dataset"] == ds and r["arm"] == arm])

    print("\n" + "=" * 78)
    print("TABLE 1 — Non-tangency of the learned field scales with alphabet size")
    print("=" * 78)
    print(f"{'':<10}{'K':>4} | {'step violation':>16} {'x0 outside':>12} {'3-mer r':>18}")
    print("-" * 68)
    tex = ["\\begin{tabular}{llrrr}", "\\toprule",
           "Alphabet & $K$ & Step viol. & $x_0$ outside & 3-mer $r$ \\\\", "\\midrule"]
    for ds, k, label in DATASETS:
        sv, ov, r = g(ds, "unconstrained", "step_violation"), \
                    g(ds, "unconstrained", "x0_outside"), g(ds, "unconstrained", "kmer_r")
        print(f"{label:<10}{k:>4} | {sv.mean():>10.4f}+-{sv.std():<5.4f} {ov.mean():>12.4f} "
              f"{r.mean():>11.4f}+-{r.std():<6.4f}")
        tex.append(f"{label} & {k} & {sv.mean():.4f} & {ov.mean():.4f} & "
                   f"{r.mean():.4f} $\\pm$ {r.std():.4f} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    write_tex("kscaling", "\n".join(tex))
    print("\nUnconstrained arm. K=4 and K=64 are the same genomic sequence read one and")
    print("three bases at a time, so alphabet size is isolated from data source and length.")


# ---------------------------------------------------------------- Table: interventions


def table_interventions(rows) -> None:
    def g(ds, arm, key):
        return np.array([r["per_nfe"]["100"][key] for r in rows
                         if r["dataset"] == ds and r["arm"] == arm])

    print("\n" + "=" * 78)
    print("TABLE 2 — Constraint interventions: what each fixes, and what it costs")
    print("=" * 78)
    print(f"{'':<10}{'K':>4} {'arm':>14} | {'step viol':>10} {'x0 out':>8} {'3-mer r':>17}")
    print("-" * 70)
    tex = ["\\begin{tabular}{llrrr}", "\\toprule",
           "Alphabet & Arm & Step viol. & $x_0$ outside & 3-mer $r$ \\\\", "\\midrule"]
    for ds, k, label in DATASETS:
        for arm in ARMS:
            sv, ov, r = g(ds, arm, "step_violation"), g(ds, arm, "x0_outside"), \
                        g(ds, arm, "kmer_r")
            print(f"{label:<10}{k:>4} {arm:>14} | {sv.mean():>10.4f} {ov.mean():>8.4f} "
                  f"{r.mean():>10.4f}+-{r.std():<6.4f}")
            tex.append(f"{label} & {arm} & {sv.mean():.4f} & {ov.mean():.4f} & "
                       f"{r.mean():.4f} $\\pm$ {r.std():.4f} \\\\")
        tex.append("\\midrule")
    tex = tex[:-1] + ["\\bottomrule", "\\end{tabular}"]
    write_tex("interventions", "\n".join(tex))


# ---------------------------------------------------------------- Table: hypothesis tests


def table_hypothesis(rows) -> None:
    def g(ds, arm, key):
        return np.array([r["per_nfe"]["100"][key] for r in rows
                         if r["dataset"] == ds and r["arm"] == arm])

    print("\n" + "=" * 78)
    print("TABLE 3 — Preregistered hypothesis tests (paired bootstrap, 3-mer r)")
    print("=" * 78)
    tex = ["\\begin{tabular}{llrl}", "\\toprule",
           "Comparison & $K$ & $\\Delta$ 3-mer $r$ & Verdict \\\\", "\\midrule"]
    for arm in ("hard", "endpoint"):
        for ds, k, label in DATASETS:
            a, b = paired(rows, ds, arm, "unconstrained", "kmer_r")
            m, lo, hi = boot(a - b)
            v = "improves" if lo > 0 else ("hurts" if hi < 0 else "n.s.")
            print(f"  {arm:>12} vs unconstrained, K={k:<3}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}]"
                  f"  {v}")
            tex.append(f"{arm} vs.\\ unconstrained & {k} & "
                       f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}] & {v} \\\\")
        tex.append("\\midrule")
    tex = tex[:-1] + ["\\bottomrule", "\\end{tabular}"]
    write_tex("hypothesis", "\n".join(tex))
    # The caption is COMPUTED, not asserted, so the stated conclusion cannot go stale.
    sig = []
    for arm in ("hard", "endpoint"):
        for ds, k, _ in DATASETS:
            a, b = paired(rows, ds, arm, "unconstrained", "kmer_r")
            m, lo, hi = boot(a - b)
            if lo > 0 or hi < 0:
                sig.append(f"{arm} at K={k} ({m:+.4f})")
    if sig:
        print("\nSignificant effects: " + "; ".join(sig) + ".")
        print("All other comparisons have CIs containing zero.")
    else:
        print("\nNo comparison reaches significance: every CI contains zero.")
    print("The endpoint arm verifiably fixes the pathology it targets (Table 2) without")
    print("improving quality, which refutes the endpoint hypothesis rather than merely")
    print("failing to support it.")


# ---------------------------------------------------------------- Table: commitment time


def table_commitment(rows) -> None:
    print("\n" + "=" * 78)
    print("TABLE 4 — Commitment time: when the decoded output stops changing")
    print("=" * 78)
    print(f"{'':<10}{'K':>4} | " + " ".join(f"{a:>14}" for a in ARMS))
    print("-" * 64)
    tex = ["\\begin{tabular}{llrrr}", "\\toprule",
           "Alphabet & $K$ & Unconstrained & Step-constrained & Endpoint \\\\", "\\midrule"]
    for ds, k, label in DATASETS:
        vals = [np.mean([r["commitment_t"] for r in rows
                         if r["dataset"] == ds and r["arm"] == a]) for a in ARMS]
        print(f"{label:<10}{k:>4} | " + " ".join(f"{v:>14.4f}" for v in vals))
        tex.append(f"{label} & {k} & " + " & ".join(f"{v:.4f}" for v in vals) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    write_tex("commitment", "\n".join(tex))
    print("\nThe decoded symbol is settled after ~5% of the trajectory. Stark et al.'s")
    print("Proposition 1 predicts t=1/2; the measured value is an order of magnitude earlier.")


# ---------------------------------------------------------------- Table: Modality 2


# Section 4.6 wants quality, diversity/coverage, and efficiency for our transferred method
# plus >=3 recent external methods, with the internal base model as a reference.
M2_ARMS = (
    # Labels say what each arm is: all three are flow matching in our harness. Reflection and
    # thresholding are sampler-side rules applied to the base network; mirror trains FM in the
    # dual (logit) space. None is the published diffusion model.
    ("reflected",     "Reflection sampler~\\citep{lou2023reflected}",           "external"),
    ("mirror",        "Mirror-map FM~\\citep{liu2023mirror}",                   "external"),
    ("dynthresh",     "Dyn.\\ thresholding on $x_t$~\\citep{saharia2022imagen}", "external"),
    ("unconstrained", "Flow matching (internal base)",                   "base"),
    ("mult",          "\\textbf{Ours}: $v = x(1-x)\\odot w$",            "ours"),
)


def table_modality2() -> None:
    """Section 4.6. Reads the U-Net batch only; an MLP-backbone file is deliberately not a
    fallback, because mixing the two backbones in one table would be meaningless."""
    files, _ = m2_files()
    if not files:
        print("\n(Modality 2 U-Net results not found)")
        return
    rows, accs = [], []
    for f in files:
        ls = [json.loads(l) for l in open(f) if l.strip()]
        accs.append(ls[0]["classifier_test_accuracy"])
        rows += [r for r in ls[1:] if "arm" in r]
    if not rows:
        print("\n(Modality 2 U-Net results not found)")
        return
    acc = float(np.mean(accs))
    seeds = sorted({r["seed"] for r in rows})
    NUMS["MTwoSeeds"] = str(len(seeds))

    def col(arm, key, nfe=None):
        s = [r for r in rows if r["arm"] == arm]
        if not s:
            return None
        return np.array([(r["fd"][nfe] if nfe else r[key]) for r in s], float)

    print("\n" + "=" * 92)
    print(f"TABLE 6 — Modality 2 transfer (MNIST, [0,1]^784), U-Net backbone. "
          f"Evaluator accuracy {acc*100:.2f}%, seeds {seeds}")
    print("=" * 92)
    print(f"{'method':>22} | {'FD@100 (q)':>19} {'cover (d)':>10} {'dens':>7} "
          f"{'FD@10 (eff)':>12} {'viol':>7}")
    print("-" * 92)
    tex = ["\\begin{tabular}{lrrrrr}", "\\toprule",
           "Method & FD $\\downarrow$ & Coverage $\\uparrow$ & Density $\\uparrow$ "
           "& FD@10 $\\downarrow$ & Step viol. $\\downarrow$ \\\\", "\\midrule"]
    base_fd = col("unconstrained", None, "100")
    for arm, label, kind in M2_ARMS:
        fd, f10 = col(arm, None, "100"), col(arm, None, "10")
        cv, dn = col(arm, "coverage_nfe100"), col(arm, "density_nfe100")
        vl = col(arm, "violation_nfe100")
        if any(c is None for c in (fd, f10, cv, dn, vl)):
            continue
        assert fd is not None and f10 is not None and cv is not None
        assert dn is not None and vl is not None
        print(f"{arm:>22} | {fd.mean():>10.1f}+-{fd.std():<7.1f} {cv.mean():>10.4f} "
              f"{dn.mean():>7.4f} {f10.mean():>12.1f} {vl.mean():>7.4f}")
        if kind == "base":
            tex.append("\\midrule")
        tex.append(f"{label} & {fd.mean():.1f} $\\pm$ {fd.std():.1f} & {cv.mean():.3f} "
                   f"& {dn.mean():.3f} & {f10.mean():.1f} & {vl.mean():.4f} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    write_tex("modality2", "\n".join(tex))

    # The preregistered decision rule for the transfer claim: paired-by-seed FD@100 against
    # the internal base, 95% interval must exclude zero. Computed, never asserted.
    print()
    ours = col("mult", None, "100")
    if ours is not None and base_fd is not None and len(ours) == len(base_fd):
        m, lo, hi = boot(base_fd - ours)   # positive = ours reduces FD
        verdict = ("SUPPORTED" if lo > 0 else
                   "NOT SUPPORTED (interval contains 0)" if hi > 0 else
                   "REJECTED (ours is worse)")
        print(f"Transfer claim, paired FD@100 reduction vs internal base: "
              f"{m:+.1f} [{lo:+.1f}, {hi:+.1f}] -> {verdict}")
        write_tex("modality2_claim",
                  f"{m:+.1f}\\,[{lo:+.1f}, {hi:+.1f}]")
        NUMS["MTwoClaimMean"] = f"{m:+.1f}"
        NUMS["MTwoClaim"] = f"${m:+.1f}\\,[{lo:+.1f}, {hi:+.1f}]$"
    # And where we stand against the external arms, reported whichever way it falls.
    for arm, label, kind in M2_ARMS:
        if kind != "external":
            continue
        e = col(arm, None, "100")
        if e is None or ours is None or len(e) != len(ours):
            continue
        m, lo, hi = boot(e - ours)
        side = "better" if lo > 0 else "worse" if hi < 0 else "indistinguishable"
        print(f"  vs {arm:>12}: {m:+.1f} [{lo:+.1f}, {hi:+.1f}]  ours is {side}")
        NUMS[f"MTwoVs{arm.capitalize()}"] = f"${m:+.1f}\\,[{lo:+.1f}, {hi:+.1f}]$"
    print("\nThe evaluator's own test accuracy is reported because this field's standard")
    print("metric (FBD) is computed from classifiers with 11.2-11.5% accuracy.")
    print("External arms are our re-implementations in this harness, not the authors' code,")
    print("and are trained at our scale; a gap may reflect the implementation.")


# ---------------------------------------------------------------- 4.3 guidance on real data


def table_guidance_real() -> None:
    """Section 4.3, real data (protocol amendment 2026-09-29): Euclidean reward guidance of the
    frozen innovation model (mult_only) toward GC content (DNA) and hydrophobic fraction
    (protein). Registered rows: unguided and gamma = 0.03 / 0.3 / 3; full grid in the appendix.
    Paired bootstrap of gamma=0.3 vs unguided on its own RNG."""
    files = sorted(glob.glob(str(RES / "guidance_real_*.jsonl")))
    if not files:
        return
    g = [json.loads(l) for f in files for l in open(f) if l.strip()]
    g = [r for r in g if r.get("tier") == "full"]
    rng = np.random.default_rng(3)
    lab = {"dna_k4": "DNA (GC)", "protein_k20": "Protein (hydrophobic)"}
    tag = {"dna_k4": "Dna", "protein_k20": "Prot"}
    col = lambda ds, gm, key: np.array([r[key] for r in sorted(
        (r for r in g if r["dataset"] == ds and abs(r["gamma"] - gm) < 1e-9), key=lambda r: r["seed"])])
    ms = lambda v, dgt=3: f"{v.mean():.{dgt}f} $\\pm$ {v.std():.{dgt}f}"

    def table(gammas, name, strength):
        tex = ["\\begin{tabular}{llrrrrr}", "\\toprule",
               "Property & Guidance ($\\gamma$) & Target frac.\\ $\\uparrow$ & Hit rate $\\uparrow$ "
               "& 3-mer $r$ held-out & 3-mer $r$ top decile $\\uparrow$ & Hamming (diversity) \\\\",
               "\\midrule"]
        for ds in lab:
            real = next(r for r in g if r["dataset"] == ds)
            for i, gm in enumerate(gammas):
                name_ = strength.get(gm, f"{gm:g}") if strength else f"{gm:g}"
                tex.append(("" if i else lab[ds]) + f" & {name_} & {ms(col(ds, gm, 'target_mean'))} & "
                           f"{ms(col(ds, gm, 'target_hit'), 2)} & {ms(col(ds, gm, 'kmer_r_heldout'))} & "
                           f"{ms(col(ds, gm, 'kmer_r_heldout_hi'))} & {ms(col(ds, gm, 'hamming'))} \\\\")
            tex.append(f" & \\emph{{held-out real}} & {real['heldout_target_mean']:.3f} & "
                       f"{real['heldout_target_hit']:.2f} & --- & --- & --- \\\\")
            if ds == "dna_k4":
                tex.append("\\midrule")
        tex += ["\\bottomrule", "\\end{tabular}"]
        write_tex(name, "\n".join(tex))

    # Compact main-text version: target (hit), quality (3-mer r to the real top decile) and
    # diversity (Hamming), means only, both properties side by side.
    tex = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
           "& \\multicolumn{3}{c}{DNA (GC)} & \\multicolumn{3}{c}{Protein (hydrophobic)} \\\\",
           "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
           "$\\gamma$ & Hit$\\uparrow$ & $r_{\\text{top}}\\uparrow$ & Ham. & Hit$\\uparrow$ & "
           "$r_{\\text{top}}\\uparrow$ & Ham. \\\\", "\\midrule"]
    for gm, nm in ((0.0, "0 (unguided)"), (0.03, "0.03 (low)"), (0.3, "0.3 (default)"), (3.0, "3 (high)")):
        cells = []
        for ds in lab:
            cells += [f"{col(ds, gm, 'target_hit').mean():.2f}",
                      f"{col(ds, gm, 'kmer_r_heldout_hi').mean():.2f}", f"{col(ds, gm, 'hamming').mean():.2f}"]
        tex.append(f"{nm} & " + " & ".join(cells) + " \\\\")
    real = {ds: next(r for r in g if r["dataset"] == ds) for ds in lab}
    div = json.load(open(RES / "m1_diversity.json"))["datasets"]
    tex += ["\\midrule", "\\emph{Held-out real} & " + " & ".join(
        f"{real[ds]['heldout_target_hit']:.2f} & --- & "
        f"{div[ds]['real_heldout']['hamming']['mean']:.2f}" for ds in lab) + " \\\\",
        "\\bottomrule", "\\end{tabular}"]
    write_tex("guidance_real_compact", "\n".join(tex))
    table((0.0, 0.03, 0.3, 3.0), "guidance_real",
          {0.0: "unguided", 0.03: "low (0.03)", 0.3: "default (0.3)", 3.0: "high (3)"})
    table(sorted({r["gamma"] for r in g}), "guidance_real_full", None)

    def boot_(d, n=20000):
        b = [rng.choice(d, len(d), replace=True).mean() for _ in range(n)]
        return d.mean(), *np.percentile(b, [2.5, 97.5])
    for ds, t in tag.items():
        for key, m in (("target_hit", "Hit"), ("kmer_r_heldout", "Kmer"), ("hamming", "Ham"),
                       ("kmer_r_heldout_hi", "KmerHi")):
            dd = col(ds, 0.3, key) - col(ds, 0.0, key)
            NUMS[f"GR{m}CI{t}"] = ci_tex(*boot_(dd), 3)
        for gm, w in ((0.0, "Zero"), (0.03, "Low"), (0.1, "Mid"), (0.3, "Def"), (3.0, "High")):
            for key, m in (("target_mean", "Target"), ("target_hit", "Hit"), ("kmer_r_heldout", "Kmer"),
                           ("kmer_r_heldout_hi", "KmerHi"), ("hamming", "Ham")):
                NUMS[f"GR{m}{w}{t}"] = f"{col(ds, gm, key).mean():.2f}"
        NUMS[f"GRViolMax{t}"] = f"{max(r['step_violation'] for r in g if r['dataset'] == ds):.3f}"
        NUMS[f"GRSec{t}"] = f"{np.mean([r['sample_seconds'] for r in g if r['dataset'] == ds]):.0f}"
        best = max(sorted({r["gamma"] for r in g}), key=lambda gm: col(ds, gm, "kmer_r_heldout_hi").mean())
        NUMS[f"GRBestGamma{t}"] = f"{best:g}"
    NUMS["GRSeeds"] = str(len({r["seed"] for r in g}))


# ---------------------------------------------------------------- exploratory diversity


def table_diversity() -> None:
    """Exploratory diversity checks (protocol note 2026-09-29): Modality 1 from
    results/m1_diversity.json (written by scripts/diversity_m1.py from the saved samples), and
    the guidance target-set split from the stored per-symbol counts. No RNG."""
    f = RES / "m1_diversity.json"
    if f.exists():
        d = json.load(open(f))["datasets"]
        rows = (("real_heldout", "\\emph{Held-out real}"), ("unigram", "\\emph{Unigram (no model)}"),
                ("unconstrained", "Linear FM (base)"), ("mult_only", "Ours $x\\odot w$"),
                ("center_only", "Center-only"), ("hard", "Full replicator"),
                ("endpoint", "Endpoint"), ("endpoint_rw", "Endpoint, reweighted"),
                ("dirichlet_ce", "Dirichlet FM"), ("fisher_sph", "Fisher-Flow"),
                ("gumbel_ce", "Gumbel-Softmax FM"),
                ("flow_matching_rerun_nfe10", "Flow matching, NFE 10$^\\dagger$"),
                ("diffusion_rerun_nfe100", "Diffusion (DDIM)$^\\dagger$"),
                ("diffusion_rerun_nfe10", "Diffusion, NFE 10$^\\dagger$"))
        tex = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
               "& \\multicolumn{3}{c}{Pairwise Hamming (NFE 100)} & \\multicolumn{3}{c}{Unique fraction} \\\\",
               "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
               "Arm & DNA & Protein & Codon & DNA & Protein & Codon \\\\", "\\midrule"]
        for key, lab in rows:
            if key not in d[DATASETS[0][0]]:
                continue
            if key == "flow_matching_rerun_nfe10":
                tex.append("\\midrule")
            ham = [f"{d[ds][key]['hamming']['mean']:.3f} $\\pm$ {d[ds][key]['hamming']['sd']:.3f}"
                   for ds, _, _ in DATASETS]
            uni = [f"{d[ds][key]['unique']['mean']:.2f}" for ds, _, _ in DATASETS]
            tex.append(f"{lab} & " + " & ".join(ham + uni) + " \\\\")
            if key == "unigram":
                tex.append("\\midrule")
        tex += ["\\bottomrule", "\\end{tabular}"]
        write_tex("diversity", "\n".join(tex))
        coll = lambda ds, arm: sum(u < 0.01 for u in d[ds][arm]["unique"]["per_seed"])
        NUMS["EndpointCollapseProt"], NUMS["EndpointCollapseCodon"] = \
            str(coll("protein_k20", "endpoint")), str(coll("codon_k64", "endpoint"))
        NUMS["EndpointRwCollapseProt"], NUMS["EndpointRwCollapseCodon"] = \
            str(coll("protein_k20", "endpoint_rw")), str(coll("codon_k64", "endpoint_rw"))
        NUMS["MaxTrainCopy"] = f"{max(v['train_copy']['mean'] for ds in d for v in d[ds].values()):.4f}"
        NUMS["RealHamProt"], NUMS["RealHamCodon"] = \
            f"{d['protein_k20']['real_heldout']['hamming']['mean']:.3f}", \
            f"{d['codon_k64']['real_heldout']['hamming']['mean']:.3f}"
        NUMS["GumbelHamProt"], NUMS["GumbelHamCodon"] = \
            f"{d['protein_k20']['gumbel_ce']['hamming']['mean']:.3f}", \
            f"{d['codon_k64']['gumbel_ce']['hamming']['mean']:.3f}"
        NUMS["GumbelStateHamProt"] = f"{d['protein_k20']['gumbel_ce_state']['hamming']['mean']:.3f}"
        if "diffusion_rerun_nfe100" in d["dna_k4"]:
            trip_ = lambda key: " / ".join(f"{d[ds][key]['hamming']['mean']:.2f}" for ds, _, _ in DATASETS)
            NUMS["DiffHam"], NUMS["DiffHamTen"] = trip_("diffusion_rerun_nfe100"), trip_("diffusion_rerun_nfe10")
            NUMS["FMHam"], NUMS["FMHamTen"] = trip_("unconstrained"), trip_("flow_matching_rerun_nfe10")
            NUMS["RealHam"] = trip_("real_heldout")
    # Guidance: does it collapse onto one target symbol? Share of the rarer target within S.
    g = json.load(open(RES / "guidance_modality1.json"))
    S, ideal = g["S"], g["ideal"]
    rare = min(S, key=lambda i: ideal[i])
    share = lambda c: c[rare] / max(sum(c[i] for i in S), 1)
    NUMS["GuidRareIdeal"] = f"{100 * ideal[rare] / sum(ideal[i] for i in S):.0f}"
    for rule, tag in (("euclidean", "Euc"), ("tangent", "Tan"), ("natural", "Nat")):
        r = next(r for r in g["rows"] if r["rule"] == rule and abs(r["gamma"] - 8.0) < 1e-9)
        NUMS[f"GuidRare{tag}"] = f"{100 * share(r['counts']):.0f}"


def main() -> int:
    global HELDOUT
    rows, HELDOUT = load_rows()
    NUMS["HeldOut"] = "1" if HELDOUT else "0"
    if not rows:
        print("no sequence results found"); return 1
    seeds = sorted({r["seed"] for r in rows})
    print(f"Sequence cells: {len(rows)}  seeds: {seeds}  tier: "
          f"{sorted({r['tier'] for r in rows})}")
    boundary_stats()
    table_fm_vs_diffusion()
    table_guidance()
    table_kscaling(rows)
    table_decomposition(rows)
    table_baselines(rows)
    table_interventions(rows)
    table_hypothesis(rows)
    table_commitment(rows)
    table_modality2()
    table_diversity()                   # exploratory; no RNG
    table_guidance_real()               # own RNG
    table_fm_vs_diffusion_real()        # last, and on its own RNG: moves no other interval
    prose_numbers(rows)                 # likewise
    write_nums()
    print(f"\nLaTeX fragments written to {TEX.relative_to(REPO)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
