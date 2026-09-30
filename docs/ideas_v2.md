# Idea slate v2 — ten candidates

Written 2026-09-20 after seven candidates died (six to prior art, one to our own probe).

**Method change from v1.** Every v1 idea started from a gap in the literature, and gaps get
filled — three of the kill shots were published 7, 11 and 16 days before we looked. Every idea
below instead starts from a **measurement we made and then tried to destroy**. A fact is much
harder to scoop than an idea, and it survives a defense even if someone published the same
mechanism, because we can show the evidence that motivated it.

## The two measurements everything is built on

**M1 — the learned field is not tangent to its own state space.** On the simplex, the fitted
velocity field points outside the domain on **21% of visited states**. The per-step rate is
flat at 0.21 across a 32x range of step sizes (NFE 25→800) while the escaped *mass* per step
falls linearly with h — so it is emphatically not Euler truncation error. 99.6% of
trajectories violate at least once. Nothing in the conditional FM objective constrains the
fitted field to respect the domain; the conditional *target* is admissible, the *fit* is not.

**M2 — the field's borrowed time-sampling default fails, and the damage scales with alphabet
size.** Logit-normal *t*-sampling (imported from SD3) versus uniform: **-53% at K=4 (n.s.),
-420% [-533, -316] at K=60**. Logit-normal was derived from a signal-to-noise argument for
Gaussian image diffusion. There is no SNR on the simplex. The derivation never transferred;
only the number did.

---

## Evaluation criteria

Each idea is scored on five axes. The binding constraint all day has been **transfer** — the
assignment requires the *same core innovation* on two modalities, and simplex-boundary
mechanisms keep evaporating on SE(3), which has no boundary.

| axis | question |
|---|---|
| grounding | does it follow from M1/M2, or from a literature gap? |
| mechanism | is it a real methodological change, or tuning in disguise? |
| transfer | does it survive on a second modality, or degenerate? |
| falsifiable | can we kill it cheaply, before GPU spend? |
| cost | does it fit nine days and $30? |

---

## 1. Constraint-preserving field parameterization

**Grounding:** M1, directly.

**Mechanism.** Classical processes that live on constrained domains — Wright-Fisher, Jacobi,
Cox-Ingersoll-Ross — stay inside because *both* their drift and their diffusion degenerate at
the boundary. That degeneracy is exactly what Feller's boundary classification requires for
the process to be well-posed without reflection or absorption. Modern learned flows on the
simplex inherit the domain but discard the property: the network emits an unconstrained
vector in R^K and nothing forces it to vanish at a face. M1 is the empirical signature of
that discarded condition.

Fix by construction. Let the network emit w and define the field in replicator form

    v_i = x_i * ( w_i - <x, w> ).

Then v_i → 0 as x_i → 0 and the flow cannot exit. No clamp, no projection, no patch.

**Transfer.** Images are box-constrained to [0,1]^d and diffusion models violate it routinely
— which is why Imagen needed *dynamic thresholding* and why every sampler clamps. The same
construction gives v_i = x_i(1-x_i) * w_i on the box. Two different geometries, one principle:
the learned drift must degenerate where the domain does.

**Falsified if.** Violations do not go to zero (implementation bug), or they do but quality is
worse because the constrained family cannot represent the true marginal field. That
expressiveness risk is real and is the interesting half of the experiment.

**Cost.** Cheap; requires retraining, but our models train in under a minute.

**Prior-art risk.** Mirror diffusion (arXiv:2308.06342, 2310.01236), reflected diffusion (Lou
& Ermon ICML 2023), simplex diffusion via Wright-Fisher (DDSM; unify-diffusion 2512.15923).
**Must check whether anyone constrains the LEARNED DRIFT's parameterization as opposed to
changing the process or the coordinates.**

---

## 2. Alphabet-aware analytic time schedule

**Grounding:** M2, with a mechanism that predicts it.

**Mechanism.** Stark's Proposition 1: for linear simplex FM the posterior loses support on one
vertex at each of t = 1/K, 1/(K-1), …, 1/2. As K grows the *first* decision moves to
t = 1/K → 0, so the informative window shrinks toward zero — while logit-normal keeps piling
mass at t = 0.5. That predicts M2 quantitatively: at K=4, 1/K = 0.25 is near the mode and
logit-normal is harmless; at K=60, 1/K = 0.017 and it is catastrophic.

Derive p(t) analytically from the elimination times rather than measuring a profile. Being
analytic and K-dependent is what dodges the measured-profile prior art that killed PACE.

**Transfer.** Weak, and this is the problem. There is no vertex-elimination structure on
images or SE(3). Possible analogue: quantization levels in 8-bit images, but that is a stretch.

**Falsified if.** The derived schedule does not beat uniform at large K, or does not recover
uniform's behaviour at small K.

---

## 3. Decode-sensitivity loss weighting

**Grounding:** M1 plus the observation that the FM loss is blind to the final rounding step.

**Mechanism.** Every simplex model ends in argmax; every 8-bit image model ends in
quantization. Velocity error that moves mass *within* a decision cell is harmless; identical
error across a cell boundary flips the output. Weight the regression loss by proximity to the
nearest decision boundary, which is closed-form in both cases.

**Transfer.** Genuine: simplex argmax ties and image quantization boundaries are the same
object — the loss ignores a non-differentiable projection that decides the output.

**Prior-art risk. High.** Haxholli (arXiv:2609.10863, 11 days old) gives the exact flip
condition on the simplex. This would read as a direct follow-up.

---

## 4. Boundary-degenerate stochasticity

**Grounding:** M1, the diffusion half of the Feller argument.

**Mechanism.** Stochastic interpolants add gamma(t) dW at sampling with gamma state-independent.
On a constrained domain that noise pushes you out near the boundary. Make gamma state-dependent
so it degenerates at the boundary, as Wright-Fisher's does.

**Prior-art risk. High on the simplex** — Wright-Fisher/Jacobi diffusions have exactly this
property and DDSM uses them. Its value is as the *second arm* of idea 1 (drift and diffusion
must both degenerate), not as a standalone.

---

## 5. Boundary exposure bias

**Grounding:** M1, from the training side.

**Mechanism.** During training x_t is constructed by interpolation and is always admissible.
During sampling it drifts out and gets clamped. So the model is queried at states it never saw
— a train/test mismatch created by the clamping patch itself. Measure the shift, then train on
clamped states to close it.

**Transfer.** Universal; any clamped sampler has it.

**Prior-art risk.** Exposure bias in diffusion is studied. The *boundary-induced* version may
not be. Needs a check.

---

## 6. Heterogeneous product state spaces

**Grounding:** the SE(3) rotation/translation ratio nobody derives.

**Mechanism.** Many real state spaces are products of *different* geometries, and the relative
scaling between factors is hand-tuned everywhere. 3D molecule generation is
R^3n x Delta^K (coordinates x atom types) and EDM hand-weights the two losses; protein
backbones are SO(3)^L x R^3L and FrameFlow hand-tunes an exponential rotation schedule against
a linear translation one. Derive the ratio from each factor's intrinsic diameter.

**Transfer.** Excellent — 3D molecules and protein backbones are both products, both
hand-tuned, and the same derivation applies.

**Risk.** A per-factor scalar may read as hyperparameter tuning, which the rubric explicitly
excludes. Needs to be framed as a derivation with a prediction, not a sweep.

---

## 7. Alphabet-scaled source concentration

**Grounding:** M2, and the geometry of the simplex at large K.

**Mechanism.** The Dirichlet(1) source is uniform on the simplex; as K grows its mass
concentrates near the barycentre (all coordinates approximately 1/K), which is maximally far
from every vertex. So expected transport distance *grows* with K even though the task does
not get harder. Scale the source concentration with K.

**Risk.** Source design is crowded (2606.04092, FlexFlow, Haxholli source geometry). Likely a
one-line control, not a contribution.

---

## 8. Reflected rather than clamped sampling

**Mechanism.** Clamping is a projection that silently changes the target distribution.
Reflection is measure-preserving. Swap it in.

**Prior-art risk. Very high** — Reflected Diffusion Models (Lou & Ermon, ICML 2023) owns this
for images.

---

## 9. Time-partitioned capacity rather than time-partitioned samples

**Mechanism.** The scheduling literature reallocates *samples* across t. Nobody reallocates
*capacity*: separate output heads, or a mixture-of-experts over t, so hard time regions get
parameters rather than draws. Distinct from the scheduling prior art that killed PACE.

**Risk.** Adds parameters, so the matched-compute control is delicate, and the rubric warns
against "a larger standard backbone." Must hold parameter count fixed by splitting, not adding.

---

## 10. Constraint violation as an oracle-free quality metric

**Grounding:** M1, plus our finding that the field's standard metric is broken — the FBD
classifiers shipped with Dirichlet FM have **11.2-11.5% test accuracy**.

**Mechanism.** Violation rate needs no proxy classifier, no oracle, no pretrained embedding.
If it correlates with sample quality it is a free diagnostic for a field whose main metric is
unreliable.

**Risk.** A metric, not a mechanism — cannot occupy the innovation slot. Valuable as a
secondary contribution and as section 4.1 content.

---

## Ranking going into adversarial review

| # | idea | grounding | mechanism | transfer | verdict |
|---|---|---|---|---|---|
| 1 | constraint-preserving field parameterization | M1 | strong | simplex + box images | **lead** |
| 6 | heterogeneous product scaling | SE(3) anomaly | medium | molecules + proteins | **strong alternate** |
| 5 | boundary exposure bias | M1 | medium | universal | backup |
| 3 | decode-sensitivity weighting | M1 | medium | simplex + quantized images | risky (Haxholli) |
| 2 | alphabet-aware schedule | M2 | strong | weak | M1-only result |
| 9 | capacity partitioning | — | medium | universal | unexamined |
| 10 | violation as metric | M1 | n/a | universal | secondary contribution |
| 4, 7, 8 | — | — | — | — | likely arms of 1, not ideas |

Ideas 1, 6, 5, 3, 9 go to adversarial review. Ideas 4, 7, 8 are retained as ablation arms.

---

# Adversarial verdicts — 2026-09-20

Nine of ten dead. Every death was a *mechanism* already published, usually within four
months. Not one *measurement* was taken from us.

| # | idea | verdict | killed by |
|---|---|---|---|
| 1 | constraint-preserving field param | **DEAD x2** | CatFlow (NeurIPS 2024) parameterizes simplex velocities to point into the simplex; Categorical Flow Maps (2602.12233); PolyFlow (ICML 2026, 2606.13400) for arbitrary polytopes. AND falsified by our own data: 7x KL cost at K=60. |
| 3 | decode-margin weighting | WOUNDED, core alive | nobody weights by argmax margin (Haxholli trains plain CE), but encircled by Quantization-Aware Diffusion (ICLR 2026) and 3 other 2026 reweighting papers. **Does not transfer to SE(3).** |
| 5 | boundary exposure bias | **DEAD** | Constraint-Aware Flow Matching (2605.12754, May 2026) -- same diagnosis, same fix |
| 6 | product-space scaling | WOUNDED, alive | Diffuse Everything (ICML 2025) owns decoupled per-modality schedules; CCDD may own SNR-synchronisation. Surviving target: FrameFlow's unexplained c=10 and its train/inference asymmetry. |
| 9 | capacity partitioning | **DEAD** | Denoising Task Routing (ICLR 2024) -- parameter-matched, in the abstract. Also violates our own architecture-independence rule. |
| 10 | violation as oracle-free metric | **DEAD as contribution** | CVR is a standard named metric; Flow Complexity (2607.16361) owns oracle-free dynamics diagnostics and reports acting on it *worsens* FID |

## The asset that survived

**Rama Cont, arXiv:2607.28344 (July 2026)**, "Reflected Diffusions, No-Flux Continuity
Equations and Confined Lagrangian Flows in Bounded Domains." Assumption F4 requires the
velocity to be tangent to the boundary; Corollary 3.3 calls this "automatic whenever the
density is continuous and positive up to the boundary."

Our measurement falsifies that premise for *learned* fields. Cont proves the true field is
tangent and assumes the learned one inherits it. It does not:

  * non-tangent on 21% of visited states at K=8, **97% at K=60**
  * per-step rate FLAT across NFE 25-800 while escaped mass falls linearly in h, so it is
    the fit and not the solver
  * 99.6% of trajectories violate at least once

## Three independent alphabet-size failures, all ours

1. **Non-tangency**: 7% of steps at K=4 -> 97% at K=60.
2. **Imported time-sampling default**: logit-normal vs uniform is -53% (n.s.) at K=4 and
   -420% [-533, -316] at K=60.
3. **Cost of the fix**: hard constraint costs 25% KL at K=4 and 690% at K=60.

Every paper in this literature validates on K=4 DNA. The amino-acid alphabet is K=20.
