# Defense deck — CIS 6270 Project 1

Build:

```
cd slides && pdflatex defense && pdflatex defense
```

It takes two passes because the corner labels are absolutely positioned.

## Why the deck shares the paper's tables, numbers and overview figure

Every table on a slide is `\input{}` from `paper/tables/*.tex`, the same fragments the paper
uses. Every number in slide prose is a macro from `paper/tables/nums.tex`. The overview figure
(slide 1f) is `paper/figures/overview.tex`, the paper's Figure 1. Running
`python scripts/make_tables.py` regenerates the fragments and macros, and rebuilding updates the
paper and the deck together, so a slide cannot quietly disagree with the paper. Nothing on a slide
is typed by hand.

The fragments carry `\citep{...}` keys for the paper's bibliography. The deck has no natbib
backend and uses the required numbered on-slide citations instead, so those keys are suppressed
by a `\providecommand` in the preamble rather than stripped from the shared source.

## Requirements this deck is built against

| Requirement | Where it is met |
|---|---|
| Slides run in order 1a–3e | Frames appear in that order. |
| Label in the **top-right** of each slide | `\lab{...}` on every frame, including the title (1a) and the bibliography ("Refs"). |
| Numbered on-slide citations | `[1]`, `[2]`, … inline in the body. |
| Sources listed **bottom-right**, very small font | `\srcs{...}`, 4.8pt grey. |
| Bibliography slide at the end | Final frame, `allowframebreaks`. |
| Hard 15-minute limit, evenly distributed | Four speakers × four slides, about 3.4 min each (see below). |
| No notes or flashcards permitted | The `\note{}` blocks are **preparation only** and do not render. Do not print them, bring them, or present in a notes-on-second-screen mode. |

## Speaking assignment

The hard limit is 15 minutes. Sixteen content slides at about 50 s each is about 13.5 minutes,
which leaves a buffer. Each person speaks four slides. The order still runs 1a → 3e; speakers hand
over at the slide boundaries shown.

| Speaker | Slides | Content | Budget |
|---|---|---|---|
| Roshan | 1a, 1b, 1c | Framing, data, the flow-matching baseline | ~2.5 min |
| Audhav | 1d, 1e, 1f | Diffusion baseline, evaluation protocol, overview | ~2.5 min |
| Chang-Uk | 2a, 2b, 2c | Choosing the base model, guidance, the innovation | ~2.5 min |
| Pedram | 2d, 2e | Ablations, comparison with recent methods | ~1.75 min |
| Audhav | 3a | The transferred method | ~0.85 min |
| Pedram | 3b | Transfer results | ~0.85 min |
| Chang-Uk | 3c | Does the innovation transfer? | ~0.85 min |
| Pedram | 3d | Limitations and failure modes | ~0.85 min |
| Roshan | 3e | Conclusions | ~0.85 min |

Per person this is about 3.4 min for Roshan, Audhav and Chang-Uk, and about 3.45 min for Pedram.
Rehearse against a timer; if anyone runs long, cut words from 2e, 3c or 3d first (they are the
densest).

## Q&A

Questions go to individual students and cannot be delegated. Every member is expected to know
the whole project. These are the questions most likely to probe the weakest points:

1. **"Is the parameterisation yours?"** No. The replicator form is Boll et al. 2024. The
   face-vanishing factor also appears in Wang & Kanamori 2026 (for reflected score learning).
   What is ours is testing the multiplicative factor *without* its centering term, in flow
   matching, on real data, plus the measurement of how often learned fields leave the simplex.
2. **"How can Fisher-Flow violate the simplex? It maps back through p = s²."** It cannot, as
   published. Our first harness integrated its sphere velocity in simplex coordinates, a bug found
   in review. The corrected sampler steps on the sphere; the old rows stay on disk and are
   disclosed (paper appendix A.4).
3. **"A unigram sampler scores 0.888 / 0.221. Are your K=20 and K=64 models learning anything
   beyond composition?"** Barely. That is why the paper reports the unigram row. Our gains over the
   linear base are real in the paired sense but sit at or below that floor. The next experiment is
   a larger backbone.
4. **"Isn't the gain just avoided clamping?"** Partly. At K=20 the gain shrinks with NFE, which
   is the clamping signature. Only at K=64 does it grow, and there it is negative at NFE 10. The
   robust result is the centering term: dropping it helps at K=64 on 8/8 seeds.
5. **"Guidance was shown on what?"** A synthetic single-position K=8 target, one seed. That is
   the only setting where the exact guided law is known in closed form. It replaced the
   classifier-free guidance we had preregistered; the paper says so.
6. **"Are your Modality 2 baselines the published diffusion models?"** No. All three are flow
   matching in our harness. Reflection and thresholding are sampler rules on the base network,
   and mirror trains in logit space. That is also why reflection and thresholding match plain
   clamping.
7. **"Is K really the only thing that changes between DNA and codon?"** Nearly. Since the data
   rebuild (v2), both come from the same chr1 region with the same held-out blocks; they differ
   in tokenisation and window length (384 vs 128 bases). The codon tokens are 3-mers of genomic
   DNA, not reading-frame codons.
8. **"Why did your baseline numbers change so much?"** Our first Dirichlet and Gumbel arms
   trained on wrong velocity targets, and Fisher-Flow was integrated in the wrong coordinates.
   All three were corrected by preregistered reruns and scored on held-out data; the old rows
   are disclosed. The earlier "we beat Gumbel and Dirichlet" was an artefact of those bugs.
