# EXP_010c — Window Grid on GPT-2 Medium (pre-registered spec)

**Status:** PRE-REGISTERED — recorded before any run.
**Created:** 2026-07-23
**Parent plan:** `STAGE2_PLAN.md` (Phase 1 extension). Sibling of EXP_010b (window
grid on GPT-2 Small); does not replace it.
**Register:** reporting register. Every outcome recorded, verdict criteria stated
before execution.

---

## 1. Question

Stage 1's sharpest degenerate case is GPT-2 Medium: trained on the same corpus as
GPT-2 Small, yet the full-stack ATR loop (0→23) collapses every one of 125 prompts
to the single token `D`, locked by iteration ~10. The J-space paper's band picture
(input-parsing → workspace → motor) says the full-stack loop is the crudest possible
cut: it splices the motor band into the sensory band every cycle.

**EXP_010c asks: is the `D` collapse an artifact of that band-splicing?** If loops
confined to Medium's putative workspace band behave qualitatively differently from
the full stack — recovering structure the full stack destroys — the collapse is a
property of the *cut*, not the *model*, and the workspace framing gains a causal
foothold. If every window says `D`, the collapse is intrinsic and the band theory
takes a hit on this model.

## 2. Hypothesis (continuing the H-numbering from STAGE2_PLAN.md)

- **H9 (collapse rescue):** GPT-2 Medium's single-basin `D` collapse is
  cut-dependent: at least one layer window (inject i → extract j) with
  0 < i or j < 23 produces a qualitatively different landscape (multiple
  terminals, semantic terminals, or non-convergence) on the same prompts.
  - **H9a (band placement):** the window most different from baseline sits inside
    the paper-mapped workspace band (layers ≈10–21; see §4), not in the sensory
    or motor extremes.

H9 is the Medium instance of H5 (window-dependence); H9a is the directional,
workspace-specific refinement. H5 on Small (EXP_010b) remains a separate, pending
experiment.

## 3. Seeding decision (pre-registered design choice)

**Every arm, regardless of window, seeds from a natural L0 prompt pass.** The
engine runs the prompt normally through the full model, reads `resid_post` at
layer j, and the loop thereafter injects at `resid_pre` of layer i. Rationale:

1. **Regime control.** Stage 1's null model located the basins in the
   *language-driven regime* of the weights. Seeding a mid-stack window any other
   way (e.g. raw token embeddings injected at layer i) would move the seed
   off-distribution and confound *where the loop is cut* with *what regime the
   seed is in*.
2. **Comparability.** All arms, including the 0→23 baseline, start from the
   identical natural forward pass; the loop cut is the only manipulated variable.
3. **Theory alignment.** The workspace picture is that sensory layers *populate*
   the workspace. The question is what iteration does to an already-populated
   workspace state — not whether mid-band layers can parse raw token embeddings
   they never receive in training.

**Recorded subtlety:** iteration 1 of every arm therefore splices a *layer-j*
state into a *layer-i* input. That splice is the treatment itself and is present
identically in the baseline (layer-23 state into layer-0 input), so it is
controlled, not confounded. **Optional follow-up control (not in the
pre-registered grid):** a "no-splice seed" arm that seeds the first injection with
the natural `resid_pre` at layer i instead — run only if results are ambiguous
about seed-transient effects.

## 4. Band mapping

The paper's workspace band is ~L38–L92 on its 0–100 depth reindexing. GPT-2
Medium has 24 layers: 0.38 × 24 = 9.1, 0.92 × 24 = 22.1 → **layers 10–21
inclusive** (rounding inward to stay strictly within the band; 12 layers).
A pleasing coincidence, recorded but not leaned on: 12 layers is exactly GPT-2
Small's full depth, so the placement sweep below doubles as "does any
Small-shaped slice of Medium behave like Small?"

## 5. Design

**Model:** GPT-2 Medium (345M, 24 layers, d_model=1024), via TransformerLens
`HookedTransformer`. CPU.

**Windows (inject i → extract j; layers i..j inclusive run in the loop):**

| Arm | Window | Length | Role |
|---|---|---|---|
| A0 | 0→23 | 24 | Baseline — **reproduction gate**: must reproduce the `D` collapse on this subset, else STOP and document |
| A1 | 0→11 | 12 | Placement sweep: front (sensory + band onset) |
| A2 | 6→17 | 12 | Placement sweep: middle |
| A3 | 12→23 | 12 | Placement sweep: back (band tail + motor) |
| A4 | 10→21 | 12 | **Band-exact** (paper-mapped workspace band) |
| A5 | 8→15 | 8 | Length probe inside the band (is any effect placement or length?) |

Window length is held at 12 for the placement sweep (A1–A4), so placement and
length are not confounded; A5 varies length at fixed (mid-band) placement.
Execution order: A0 first (gate), then A4, then A1–A3, A5 last.

**Prompts (deterministic, recorded at run time):** 25 prompts = round-robin
across the 7 Stage-1 categories, alphabetical within category, drawn from the
125-prompt Stage 1 set. **Deviation recorded:** `prompt_library.py` is absent
from both repos; the 125 prompts and their categories are recovered from the
frozen archive `_DATA/EXP009_stage1_trajectories_2026-07-10.zip` →
`experiments/gpt2_medium/output/stage1_results.pt` (read-only). The recovered
list must be printed and committed with the results.

**Protocol:** gated, identical to Stage 1 / RUNBOOK_PHASE1: `threshold=0.999`,
`patience=3`, `check_every=10`, `check_start=100`, `max_iter=1000`, `gate_lag=1`.
Post-hoc `lag_scan` (lags 1–8) on the final iterates of every non-converged run —
the `Divine` lesson: a lag-1 gate cannot see a limit cycle.

**Readout:** Stage-1-identical `ln_final → W_U` decode (comparability), with R1
confidence metrics (top-logit margin, entropy). **Terminal mean vectors and
last-position vectors saved per (window, prompt)** to
`experiments/exp_010c_windows/output/terminals.pt` — these are the hand-off
artifact for the J-lens phase (RUNBOOK_JLENS_MEDIUM.md).

**Per-arm record:** converged fraction, lock-in iterations, terminal token table,
unique-terminal count, cross-prompt terminal mean-vector cosine matrix, R1
metrics. One summary table across arms.

## 6. Pre-registered outcome readings

| Observation | Reading |
|---|---|
| A0 reproduces `D`; A4 (band) escapes it (≥3 distinct terminals, or semantic terminals, or systematic non-convergence) while A1/A3 mirror baseline | **H9 and H9a supported.** The collapse is cut-dependent and band-placed. A4's window becomes the Phase-2 prior for where Medium's workspace lives. |
| A0 reproduces `D`; some window escapes but not preferentially the band (e.g. A1 or A3 differs most) | **H9 supported, H9a not.** Window-dependence is real but not workspace-shaped as mapped; re-examine the band mapping with the J-lens before interpreting. |
| All arms collapse to `D` (or a single junk token each) | **H9 refuted on Medium.** The collapse is not a band-splicing artifact. Does not alone kill H5 (Small grid pending) but removes the tidiest workspace explanation of the Stage 1 regime table. |
| All non-baseline arms fragment (no consolidation anywhere) | Ambiguous — consult A5 vs A2/A4: if the 8-layer window consolidates where 12-layer windows fragment (or vice versa), length is the driver, not placement. Flag for the no-splice control. |
| A0 fails to reproduce `D` on this subset | **STOP.** Harness fault or subset artifact; document, fix, rerun. No interpretation of A1–A5 permitted. |

Any outcome is recorded. Per the house rule: negatives are findings.

## 7. Cost & environment

25 prompts × 6 arms, gated (min ~120 iters, max 1000). Baseline locks in ~10
iters historically, so A0 is cheap; fragmenting arms are the expensive tail.
Estimate: overnight CPU. Pilot tier (see runbook): 5 prompts × 6 arms,
`max_iter=300` — harness validation and a first directional signal, ~1–2 h CPU.

## 8. Relation to the existing plan

- Extends Phase 1 (bridge experiments, no J-lens needed) with a Medium arm.
- EXP_010b (Small grid) remains as specified in RUNBOOK_PHASE1; running both
  makes the Small/Medium comparison itself the finding.
- Phase-2 refinement logic and the J-lens follow-on live in
  `RUNBOOK_JLENS_MEDIUM.md`.
