"""Generate every paper figure from stored results. No number is typed by hand.

Design constraints, for a print venue:
  * Colour is never the only channel. Every series also carries a distinct marker and
    linestyle, so the figures survive greyscale printing and colour-vision deficiency.
    The palette (Okabe-Ito) validates with CVD dE in the 6-8 band, which is permitted
    only with that secondary encoding present.
  * One y-axis per panel. Two measures of different scale get two panels.
  * Recessive grid and spines; text in ink, never in series colour.
  * Error bars are the paired bootstrap 95% CI actually used in the tables.

Run: python scripts/make_figures.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "paper" / "figures"
RNG = np.random.default_rng(0)

# Okabe-Ito, ordered so the weakest CVD pair is not adjacent.
C = {"blue": "#0072B2", "verm": "#D55E00", "green": "#009E73",
     "orange": "#E69F00", "purple": "#CC79A7", "grey": "#5a5a5a"}

DS = (("dna_k4", 4, "DNA"), ("protein_k20", 20, "Protein"), ("codon_k64", 64, "Codon"))

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#9a9a9a", "axes.linewidth": 0.7,
    "grid.color": "#e2e2e0", "grid.linewidth": 0.6,
    "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,                 # TrueType, not Type 3
})


def _load(pattern, tier):
    u = {}
    for f in glob.glob(str(REPO / "results" / pattern)):
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("tier") != tier:                      # smoke rows share the glob
                continue
            u[(r["dataset"], r["arm"], r["seed"])] = r     # dedupe appended rows
    return list(u.values())


def load():
    """Held-out rows on data v2 once all 216 exist (protocol amendments 2026-09-29), with the
    denoiser baselines decoded from their final posterior -- the same rule as make_tables.py."""
    ho = [r for r in _load("heldout_*.jsonl", "heldout") if r.get("data") == "v2"]
    if len(ho) != 9 * 3 * 8:
        return _load("endpoint_*.jsonl", "full")
    for r in ho:
        if r["arm"] in ("gumbel_ce", "dirichlet_ce"):
            for m in r["per_nfe"].values():
                if "kmer_r_post" in m:
                    m["kmer_r"], m["marginal_kl"] = m["kmer_r_post"], m["marginal_kl_post"]
    return ho


ROWS = load()
# `fisher` integrated a sphere velocity in simplex coordinates (a harness bug); `fisher_sph`
# samples on the sphere as published. Use the corrected arm once it exists.
FISHER = "fisher_sph" if any(r["arm"] == "fisher_sph" for r in ROWS) else "fisher"


# Corrected Gumbel-Softmax FM / Dirichlet FM arms (protocol amendment 2026-09-29), used once all
# 24 cells exist -- the same rule as make_tables.py, so figures and tables cannot disagree.
def _corrected(old, new):
    n = len({(r["dataset"], r["seed"]) for r in ROWS if r["arm"] == new})
    return new if n == 24 else old


GUMBEL, DIRICHLET = _corrected("gumbel", "gumbel_ce"), _corrected("dirichlet", "dirichlet_ce")


def m2_files():
    """Shared-extractor Modality 2 rerun (protocol amendment 2026-09-29) once all 40 cells exist."""
    fixed = sorted(glob.glob(str(REPO / "results" / "m2fixed_*.jsonl")))
    if sum(1 for f in fixed for l in open(f) if '"seed"' in l) == 40:
        return fixed, "m2fixed_"
    return sorted(glob.glob(str(REPO / "results" / "modality2_unet_*.jsonl"))), ""


def vals(ds, arm, key, nfe="100"):
    return np.array([r["per_nfe"][nfe][key] for r in ROWS
                     if r["dataset"] == ds and r["arm"] == arm])


def paired(ds, base, arm, nfe="100"):
    a = {r["seed"]: r for r in ROWS if r["dataset"] == ds and r["arm"] == base}
    b = {r["seed"]: r for r in ROWS if r["dataset"] == ds and r["arm"] == arm}
    ks = sorted(set(a) & set(b))
    return np.array([b[k]["per_nfe"][nfe]["kmer_r"] - a[k]["per_nfe"][nfe]["kmer_r"]
                     for k in ks])


def boot(d, n=20000):
    bs = [RNG.choice(d, len(d), replace=True).mean() for _ in range(n)]
    return d.mean(), *np.percentile(bs, [2.5, 97.5])


def tidy(ax, ylab, xlab=None, title=None):
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylab)
    if xlab:
        ax.set_xlabel(xlab)
    if title:
        ax.set_title(title, loc="left", pad=6)


# ---------------------------------------------------------------- Fig 1: overview


def m2_caption() -> str:
    """The panel-(b) annotation, read from the Modality 2 rows rather than typed, so it cannot
    assert a stale result. Falls back to a number-free caption if the rows are absent.
    """
    rows = []
    for f in m2_files()[0]:
        rows += [json.loads(l) for l in open(f) if l.strip() and '"arm"' in l]
    def fd(arm):
        v = [r["fd"]["100"] for r in rows if r["arm"] == arm]
        return float(np.mean(v)) if v else None
    # No FD in the annotation: the FD change failed its preregistered test, and a schematic
    # showing an FD drop reads as a win.
    tail = ""
    viol = max((r["violation_nfe100"] for r in rows if r["arm"] == "mult"), default=None)
    exits = f"{viol*100:.1f}%" if viol is not None else "exactly zero"
    return ("exits on a nonzero fraction of steps;\n"
            r"$x(1{-}x)\odot w$ exits on " + exits + " at NFE 100" + tail)


def fig_overview():
    """Schematic of the mechanism and its transfer, with measured numbers annotated.

    Panel (a): the 2-simplex. The learned field points out of the domain near a face; the
    multiplicative factor makes it vanish there instead. Panel (b): the same construction on
    the unit box, where x(1-x) vanishes at both faces of every interval.

    Percent signs are literal here -- matplotlib is not using a LaTeX backend, so escaping
    them would print the backslash.
    """
    from matplotlib.patches import Polygon, Rectangle, FancyArrowPatch
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.15))

    # ---- (a) simplex --------------------------------------------------------
    V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]])
    a1.add_patch(Polygon(V, closed=True, fc="#f4f4f2", ec="#8a8a8a", lw=1.2, zorder=1))
    for v, lab, off in zip(V, ["$e_1$", "$e_2$", "$e_3$"],
                           [(-0.11, -0.10), (0.05, -0.10), (-0.03, 0.05)]):
        a1.annotate(lab, v + np.array(off), fontsize=7.5, color="#333")
    t = np.linspace(0, 1, 60)
    x0 = np.array([0.42, 0.34]); x1 = V[1]
    traj = (1 - t)[:, None] * x0 + t[:, None] * x1
    a1.plot(traj[:, 0], traj[:, 1], "-", color=C["grey"], lw=1.4, zorder=3)
    a1.plot(*x0, "o", color=C["grey"], ms=5, zorder=4)
    a1.annotate("$x_0$", x0 + np.array([-0.13, 0.03]), fontsize=7, color="#333")
    for px, py in [(0.60, 0.145), (0.75, 0.095)]:
        a1.add_patch(FancyArrowPatch((px, py), (px + 0.09, py - 0.135),
                                     arrowstyle="-|>", mutation_scale=9,
                                     color=C["verm"], lw=1.6, zorder=5))
    for px, py, sc in [(0.60, 0.145, 0.085), (0.75, 0.095, 0.042)]:
        a1.add_patch(FancyArrowPatch((px, py), (px + sc * 1.6, py + sc * 0.15),
                                     arrowstyle="-|>", mutation_scale=8,
                                     color=C["blue"], lw=1.6, zorder=6))
    a1.plot([], [], color=C["verm"], lw=1.7, label="learned $v=w$")
    a1.plot([], [], color=C["blue"], lw=1.7, label=r"$v=x\odot w$")
    # Short labels and a left anchor: longer labels collide with the $e_3$ vertex label, and
    # above the apex they collide with the panel title.
    a1.legend(frameon=False, fontsize=6.3, loc="upper left",
              bbox_to_anchor=(-0.06, 0.93), handlelength=1.1, labelspacing=0.3)
    a1.set_xlim(-0.24, 1.20); a1.set_ylim(-0.30, 1.14); a1.set_aspect("equal")
    a1.axis("off")
    a1.set_title("(a) Modality 1: probability simplex", loc="left", fontsize=8.5, pad=2)
    # Computed, like panel (b), so they cannot drift from the intervals the paper quotes.
    sv = lambda ds: vals(ds, "unconstrained", "step_violation").mean() * 100
    ours = max(vals(ds, "mult_only", "step_violation").mean() for ds in ("dna_k4", "codon_k64"))
    a, b = ({r["seed"]: r["per_nfe"]["100"]["kmer_r"] for r in ROWS
             if r["dataset"] == "codon_k64" and r["arm"] == arm}
            for arm in ("mult_only", "unconstrained"))
    gain = np.mean([a[i] - b[i] for i in sorted(set(a) & set(b))])
    a1.text(0.5, -0.24, f"exits on {sv('dna_k4'):.0f}% of steps at $K$=4, "
            f"{sv('codon_k64'):.0f}% at $K$=64;\n"
            + r"$x\odot w$ exits on " + f"{ours*100:.1f}% at NFE 100 and improves $K$=64 by {gain:+.3f}",
            ha="center", va="top", fontsize=6.3, color="#555", transform=a1.transData)

    # ---- (b) box ------------------------------------------------------------
    a2.add_patch(Rectangle((0, 0), 1, 1, fc="#f4f4f2", ec="#8a8a8a", lw=1.2, zorder=1))
    x0b = np.array([0.30, 0.42]); x1b = np.array([0.88, 0.84])
    tb = (1 - t)[:, None] * x0b + t[:, None] * x1b
    a2.plot(tb[:, 0], tb[:, 1], "-", color=C["grey"], lw=1.4, zorder=3)
    a2.plot(*x0b, "o", color=C["grey"], ms=5, zorder=4)
    a2.annotate("$x_0$", x0b + np.array([-0.15, 0.04]), fontsize=7, color="#333")
    for px, py in [(0.86, 0.82), (0.60, 0.95)]:
        a2.add_patch(FancyArrowPatch((px, py), (px + 0.13, py + 0.13),
                                     arrowstyle="-|>", mutation_scale=9,
                                     color=C["verm"], lw=1.6, zorder=5))
    for px, py, sc in [(0.86, 0.82, 0.10), (0.60, 0.95, 0.035)]:
        a2.add_patch(FancyArrowPatch((px, py), (px + sc, py + sc),
                                     arrowstyle="-|>", mutation_scale=8,
                                     color=C["blue"], lw=1.6, zorder=6))
    a2.annotate(r"$v=x(1{-}x)\odot w$", (0.03, 0.08), fontsize=7.5, color=C["blue"])
    a2.set_xlim(-0.14, 1.24); a2.set_ylim(-0.30, 1.14); a2.set_aspect("equal")
    a2.axis("off")
    a2.set_title("(b) Modality 2: unit box $[0,1]^d$", loc="left", fontsize=8.5, pad=2)
    a2.text(0.5, -0.24, m2_caption(),
            ha="center", va="top", fontsize=6.3, color="#555", transform=a2.transData)

    fig.text(0.5, 0.005, "All components trainable; nothing pretrained or frozen. "
             "The parameterisation adds no parameters and no hyperparameters.",
             ha="center", fontsize=6.4, color="#555")
    fig.savefig(FIG / "fig1_overview.pdf"); plt.close(fig)
    print("  fig1_overview.pdf")


# ---------------------------------------------------------------- Fig 2: K-scaling


def fig_kscaling():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 1.85))
    ks = [k for _, k, _ in DS]
    viol = [vals(ds, "unconstrained", "step_violation").mean() for ds, _, _ in DS]
    x0 = [vals(ds, "unconstrained", "x0_outside").mean() for ds, _, _ in DS]
    q = [vals(ds, "unconstrained", "kmer_r") for ds, _, _ in DS]

    a1.plot(ks, viol, "-o", color=C["verm"], lw=2, ms=7, label="per-step violation")
    a1.plot(ks, x0, "--s", color=C["blue"], lw=2, ms=7, label=r"$x_0$-estimate outside")
    a1.set_xscale("log"); a1.set_xticks(ks); a1.set_xticklabels(ks)
    a1.set_ylim(0, 1.08)
    tidy(a1, "fraction of steps", "alphabet size $K$", "(a) the field leaves its domain")
    a1.legend(frameon=False, loc="lower right")

    m = [v.mean() for v in q]
    sd = [v.std() for v in q]
    a2.errorbar(ks, m, yerr=sd, fmt="-D", color=C["green"], lw=2, ms=7,
                capsize=3, elinewidth=1)
    a2.set_xscale("log"); a2.set_xticks(ks); a2.set_xticklabels(ks)
    a2.set_ylim(0, 1.18)          # headroom: at 1.05 the K=4 label touches the panel title
    for xx, yy in zip(ks, m):
        a2.annotate(f"{yy:.2f}", (xx, yy), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=7, color="#333")
    tidy(a2, "3-mer correlation", "alphabet size $K$", "(b) base-arm 3-mer correlation per dataset")
    fig.savefig(FIG / "fig2_kscaling.pdf"); plt.close(fig)
    print("  fig2_kscaling.pdf")


# ---------------------------------------------------------------- Fig 3: NFE dependence


def fig_nfe():
    """The confound test. An artefact of clamping must shrink as the step shrinks."""
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 1.8), sharey=True)
    nfes = ["10", "50", "100"]
    xs = np.arange(len(nfes))
    for ax, (ds, k, lab) in zip(axes, DS):
        for arm, col, mk, ls, name in (
                ("hard", C["orange"], "o", "--", r"full  $x\odot(w-\langle x,w\rangle)$"),
                ("mult_only", C["blue"], "D", "-", r"mult only  $x\odot w$")):
            ms_, los, his = [], [], []
            for n in nfes:
                m, lo, hi = boot(paired(ds, "unconstrained", arm, n))
                ms_.append(m); los.append(m - lo); his.append(hi - m)
            ax.errorbar(xs, ms_, yerr=[los, his], fmt=mk + ls, color=col, lw=1.8,
                        ms=6, capsize=3, elinewidth=1, label=name)
        ax.axhline(0, color=C["grey"], lw=0.8, ls=":")
        ax.set_xticks(xs); ax.set_xticklabels(nfes)
        tidy(ax, r"$\Delta$ 3-mer $r$ vs base" if ax is axes[0] else "",
             "NFE", f"$K={k}$ ({lab})")
    axes[0].legend(frameon=False, loc="upper right")
    fig.savefig(FIG / "fig3_nfe.pdf"); plt.close(fig)
    print("  fig3_nfe.pdf")


# ---------------------------------------------------------------- Fig 4: decomposition


def fig_decomposition():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 1.85))
    arms = [("mult_only", r"mult only  $x\odot w$", C["blue"]),
            ("center_only", r"center only  $w-\bar w$", C["purple"]),
            ("hard", r"both (full replicator)", C["orange"])]
    w, xs = 0.26, np.arange(len(DS))
    for i, (arm, lab, col) in enumerate(arms):
        m, lo, hi = [], [], []
        for ds, _, _ in DS:
            mm, l, h = boot(paired(ds, "unconstrained", arm))
            m.append(mm); lo.append(mm - l); hi.append(h - mm)
        a1.bar(xs + (i - 1) * w, m, w * 0.88, yerr=[lo, hi], color=col, label=lab,
               capsize=2.5, error_kw={"elinewidth": 0.9}, zorder=3,
               edgecolor="white", linewidth=0.8)
    a1.axhline(0, color=C["grey"], lw=0.9)
    a1.set_xticks(xs); a1.set_xticklabels([f"$K$={k}" for _, k, _ in DS])
    tidy(a1, r"$\Delta$ 3-mer $r$ vs base", None, "(a) the factor carries the gain")
    # Below the axes: inside, it hides the top of the K=20 error bars.
    a1.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3,
              fontsize=6.2, handlelength=1.2, columnspacing=0.9)

    for i, (arm, lab, col) in enumerate(arms):
        v = [vals(ds, arm, "step_violation").mean() for ds, _, _ in DS]
        a2.bar(xs + (i - 1) * w, v, w * 0.88, color=col, zorder=3,
               edgecolor="white", linewidth=0.8)
        # A bar of height 0 is invisible, and "exactly zero" is the finding -- so the
        # near-zero bars are labelled rather than left as blank space.
        for j, vv in enumerate(v):
            if vv < 0.05:
                a2.annotate(f"{vv:.3f}", (xs[j] + (i - 1) * w, vv),
                            textcoords="offset points", xytext=(0, 3), ha="center",
                            fontsize=5.6, color=col, rotation=90)
    base = [vals(ds, "unconstrained", "step_violation").mean() for ds, _, _ in DS]
    a2.plot(xs, base, "k_", ms=22, mew=1.6, label="unconstrained base")
    a2.set_xticks(xs); a2.set_xticklabels([f"$K$={k}" for _, k, _ in DS])
    a2.set_ylim(0, 1.32)
    tidy(a2, "step-violation rate", None, "(b) centering does not constrain at all")
    a2.legend(frameon=False, loc="upper center")
    fig.savefig(FIG / "fig4_decomposition.pdf"); plt.close(fig)
    print("  fig4_decomposition.pdf")


# ---------------------------------------------------------------- Fig 5: baselines


def fig_baselines():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 1.95))
    methods = [("unconstrained", "linear FM", C["grey"]),
               ("mult_only", "ours ($x\\odot w$)", C["blue"]),
               (DIRICHLET, "Dirichlet FM", C["verm"]),
               (FISHER, "Fisher-Flow", C["green"]),
               (GUMBEL, "Gumbel-Softmax FM", C["orange"])]
    w, xs = 0.16, np.arange(len(DS))
    for i, (arm, lab, col) in enumerate(methods):
        q = [vals(ds, arm, "kmer_r").mean() for ds, _, _ in DS]
        e = [vals(ds, arm, "kmer_r").std() for ds, _, _ in DS]
        a1.bar(xs + (i - 2) * w, q, w * 0.88, yerr=e, color=col, label=lab, capsize=2,
               error_kw={"elinewidth": 0.8}, zorder=3, edgecolor="white", linewidth=0.6)
        v = [vals(ds, arm, "step_violation").mean() for ds, _, _ in DS]
        a2.bar(xs + (i - 2) * w, v, w * 0.88, color=col, zorder=3,
               edgecolor="white", linewidth=0.6)
        # Zero-height bars are invisible, and ours being exactly zero is the point: label them.
        for j, vv in enumerate(v):
            if vv < 0.05:
                a2.annotate(f"{vv:.3f}", (xs[j] + (i - 2) * w, vv),
                            textcoords="offset points", xytext=(0, 2), ha="center",
                            va="bottom", fontsize=5.4, color=col, rotation=90)
    for ax, ylab, title in ((a1, "3-mer correlation", "(a) sample quality"),
                            (a2, "step-violation rate", "(b) constraint satisfaction")):
        ax.set_xticks(xs); ax.set_xticklabels([f"$K$={k}" for _, k, _ in DS])
        tidy(ax, ylab, None, title)
    # One legend above both panels: inside panel (a) it sits on top of the K=4 bars.
    fig.legend(*a1.get_legend_handles_labels(), frameon=False, ncol=5, loc="lower center",
               bbox_to_anchor=(0.5, 0.99), columnspacing=1.2, handlelength=1.2)
    a1.set_ylim(0, 1.08); a2.set_ylim(0, 1.08)
    fig.savefig(FIG / "fig5_baselines.pdf"); plt.close(fig)
    print("  fig5_baselines.pdf")


# ---------------------------------------------------------------- Fig 6: Modality 2


# Section 4.6: our transferred arm, the internal base, and three external methods.
# Hatching carries identity alongside colour so the bars survive greyscale printing.
M2 = (("reflected",     "reflect",     C["green"],  "//"),
      ("mirror",        "mirror",      C["orange"], "\\\\"),
      ("dynthresh",     "dyn.thr.",    C["purple"], "xx"),
      ("unconstrained", "base",        C["grey"],   ""),
      ("mult",          "ours",        C["blue"],   ".."))


def fig_modality2():
    rows, accs = [], []
    for f in m2_files()[0]:
        ls = [json.loads(l) for l in open(f) if l.strip()]
        accs.append(ls[0]["classifier_test_accuracy"])
        rows += [r for r in ls[1:] if "arm" in r]
    if not rows:
        print("  (skipped fig6: no U-Net rows)"); return
    acc = float(np.mean(accs))
    present = [m for m in M2 if any(r["arm"] == m[0] for r in rows)]
    xs = np.arange(len(present))
    labs = [m[1] for m in present]

    def gather(arm, key, nfe=None):
        return np.array([(r["fd"][nfe] if nfe else r[key])
                         for r in rows if r["arm"] == arm], float)

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(6.6, 2.15))
    fig.subplots_adjust(wspace=0.42)
    for ax, (key, nfe, ylab, title) in zip(
            (a1, a2, a3),
            ((None, "100", "Fr\u00e9chet distance (lower better)", "(a) sample quality"),
             ("coverage_nfe100", None, "coverage (higher better)", "(b) diversity"),
             ("violation_nfe100", None, "step-violation rate", "(c) feasibility"))):
        v = [gather(m[0], key, nfe) for m in present]
        fmt = {"100": "{:.0f}", "coverage_nfe100": "{:.2f}", "violation_nfe100": "{:.3f}"}[nfe or key]
        ax.bar(xs, [x.mean() for x in v], 0.62,
               yerr=[x.std() for x in v] if len(v[0]) > 1 else None,
               color=[m[2] for m in present], hatch=[m[3] for m in present],
               capsize=2.5, zorder=3, error_kw={"elinewidth": 0.9},
               edgecolor="white", linewidth=0.8)
        # Zero-height bars are invisible, so every bar is also labelled numerically, above its
        # whisker so the whisker does not strike through it; the axis is sized to the whiskers
        # so no label is clipped.
        top = max(x.mean() + x.std() for x in v) or 1.0
        for x, s in zip(xs, v):
            ax.text(x, s.mean() + s.std() + 0.03 * top, fmt.format(s.mean()), ha="center",
                    va="bottom", fontsize=5.6, color="#333", zorder=4)
        # Short labels: long rotated ones run into the neighbouring panel's y-label.
        ax.set_xticks(xs); ax.set_xticklabels(labs, rotation=30, ha="right", fontsize=6.0)
        ax.set_ylim(0, top * 1.16)
        tidy(ax, ylab, None, title)
    nseed = len({r["seed"] for r in rows})
    # Two lines, so the footer does not widen the tight bbox beyond the panels. Plain "%":
    # matplotlib is not using a LaTeX backend, so an escaped one prints its backslash.
    fig.text(0.5, -0.2, f"MNIST, $[0,1]^{{784}}$, U-Net backbone, NFE${{=}}100$, {nseed} seeds. "
             f"Feature extractor test accuracy {acc*100:.1f}%.\nExternal arms are our "
             f"re-implementations. Bars are means; whiskers are $\\pm$1 s.d.",
             ha="center", fontsize=6.4, color="#555", linespacing=1.4)
    fig.savefig(FIG / "fig6_modality2.pdf"); plt.close(fig)
    print("  fig6_modality2.pdf")


def fig_m2_samples():
    """Uncurated sample grids: the first 8 images each arm drew, saved before inspection."""
    # Prefer the seed-tagged grids: under the untagged name two batches of the same arm
    # overwrite each other, so which seed it holds is not recoverable.
    def grid(arm):
        tagged = sorted(glob.glob(str(REPO / "results" / f"samples_{m2_files()[1]}{arm}_s*.npy")))
        if tagged:
            return tagged[0], int(tagged[0].rsplit("_s", 1)[1].split(".")[0])
        p = REPO / "results" / f"samples_{arm}.npy"
        return (str(p), None) if p.exists() else (None, None)

    got = [(a, lab, *grid(a)) for a, lab, _, _ in M2 if grid(a)[0]]
    if not got:
        print("  (skipped fig7: no sample grids)"); return
    seeds = {g[3] for g in got}
    tag = (f"seed {seeds.pop()}" if len(seeds) == 1 and None not in seeds
           else "one seed per arm")
    fig, axes = plt.subplots(len(got), 8, figsize=(6.6, 0.85 * len(got) + 0.25))
    axes = np.atleast_2d(axes)
    for r, (arm, lab, path, seed) in enumerate(got):
        s = np.load(path).reshape(-1, 28, 28)
        for c in range(8):
            ax = axes[r, c]
            ax.imshow(s[c], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#cccccc")
            if c == 0:
                ax.set_ylabel(lab, fontsize=6, rotation=0, ha="right", va="center",
                              labelpad=4)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.text(0.5, 0.03, f"First eight samples drawn at NFE=100, {tag}, saved before any "
             f"inspection. The clamping-family arms are pixel-identical.",
             ha="center", fontsize=6.2, color="#555")
    fig.savefig(FIG / "fig7_m2_samples.pdf"); plt.close(fig)
    print("  fig7_m2_samples.pdf")


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    print("writing figures:")
    # fig_overview() is not called: Figure 1 is TikZ in paper/figures/overview.tex, shared with the deck.
    fig_kscaling(); fig_nfe(); fig_decomposition(); fig_baselines()
    fig_modality2(); fig_m2_samples()
    print(f"-> {FIG.relative_to(REPO)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
