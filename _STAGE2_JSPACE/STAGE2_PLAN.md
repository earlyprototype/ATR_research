# Stage 2 — Plan & Requirements (v1)

**Status:** WORKING PLAN — direction chosen (see Decision Log in STAGE2_OUTLINE.md).
**Date:** 2026-07-11
**Question:** Is the ATR attractor landscape a readout of workspace structure?
**Theory under test:** the residual stream has a band structure (input-parsing →
workspace → motor; Gurnee et al. 2026). ATR's full-stack loop splices the motor band
into the sensory band every cycle; the Stage 1 cross-model regime differences and the
GPT-2 Small anomaly may be band-structure differences.

**Register rules inherited from Stage 1:** reporting register throughout this folder;
pre-register hypotheses before runs; every result recorded whether it helps or not;
kill criteria stated up front.

---

## Hypotheses

| ID | Hypothesis | Tested by |
|---|---|---|
| H5 | Layer-window loops (inject i, extract j) produce qualitatively different landscapes *within* the putative workspace band vs *across* band boundaries | EXP_010b |
| H6 | GPT-2 Small's five basin tensors project significantly more onto the J-space than the 18 null-model basins | EXP_011 |
| H7 | A coherent J-lens band structure exists at 124M–410M scale, and its presence/shape predicts each model's ATR regime | EXP_012 |
| H8 | Pythia-410m's fragmentation is depth-dependent: looping layers 0–11 (vs native 0–23) changes convergence behaviour | EXP_010a |
| Q-D | (Question, not hypothesis) What is the `Divine` object — limit cycle, wandering attractor, or decode-region plateau? | EXP-D, EXP_013 |

## Kill criteria (what would end the workspace framing)

- EXP_010 shows **no** qualitative window-dependence, **and** EXP_011 shows basins
  project no differently than noise basins → the band-structure theory is dead for
  these models; Stage 2 closes as a classical-dynamics characterisation and the
  anomaly stands unexplained.
- Any single experiment failing does NOT kill the frame alone; the conjunction does.
- EXP-D (Divine) is unconditional: any outcome is a finding.

---

## Phases

### Phase 0 — Instrument (parallel track, start immediately)

*Directions checked 2026-07-11 — dependencies pinned:*
- **Official release: https://github.com/anthropics/jacobian-lens** (Apache-2.0,
  reference implementation, unmaintained). Fits open-weights HF decoders — "other
  HuggingFace decoders adapt cleanly." Ships `walkthrough.ipynb` (end-to-end) and
  Anthropic's synthetic replication prompt sets in `data/`.
- **API is small:** `model = jlens.from_hf(hf_model, tok)`;
  `lens = jlens.fit(model, prompts=..., checkpoint_path=...)` (checkpointable,
  parallelizable via `JacobianLens.merge()`); `lens.apply(model, prompt, positions=...)`
  → per-layer lens logits. Transport is a per-layer expected Jacobian
  `J_l = E[∂h_final/∂h_l]` — for gpt2-small a 768×768 matrix per layer.
- **Fitting cost:** paper lenses use 1000 × 128-token web-text sequences; quality
  saturates fast, ~100 prompts usable. Cost is dominated by the model's own backward
  pass → **CPU-feasible at 124M–410M scale; no GPU required for this plan.**
- **Neuronpedia shortcut: NO** — hosted J-lens currently covers Qwen 3.6 27B only.
  We fit our own lenses.

1. **P0-1** ✅ resolved (above). Clone the repo; `pip install -e .`.
2. **P0-2** Fit the lens for GPT-2 Small: start at ~100 prompts from a
   pretraining-like corpus (or the repo's `data/` sets), checkpoint, scale to 1000 if
   the validation gate wants it. Then Pythia-160m; Medium/410m as Phase 3 approaches.
3. **P0-3** Validation gate: on gpt2-small, lens readouts at mid-layers must be
   qualitatively more interpretable than logit-lens readouts on multi-step prompts
   (walkthrough patterns: multihop recall, the currency/boot example), with sane
   per-layer progression, before any Stage 2 conclusion uses them.
4. **P0-4** Extend `atr_engine.py` with window parameters (inject-layer i,
   extract-layer j) — small, additive, mirrors the `run_atr_gated` pattern.

### Phase 1 — Bridge experiments (no J-lens required, start immediately)
- **EXP_010a — Depth control (H8).** Pythia-410m, loop layers 0–11 vs native 0–23.
  25-prompt diverse subset, convergence-gated, 1000-iter cap.
  *Decides:* whether fragmentation is depth-driven. ~1–2 h CPU.
- **EXP_010b — Window grid (H5, coarse).** GPT-2 Small, windows
  {0→5, 3→8, 6→11, 0→8, 3→11} vs full-stack 0→11 baseline. 25-prompt subset
  including ≥5 `Divine`-basin prompts, gated.
  *Decides:* whether the landscape depends on where the loop is cut, and whether any
  window is qualitatively richer (candidate workspace band). Overnight CPU.
- **EXP-D — Divine characterisation (Q-D, classical).** From existing gated
  trajectories (`output_gated/gated_results.pt`): R1/R3 confidence audit across all
  34 `Divine` prompts; trajectory geometry (PCA, recurrence structure, step-size
  spectra). No new sweeps needed. *Decides:* limit cycle vs wandering attractor at
  the tensor level. Hours, mostly analysis.

**Decision point 1:** if EXP_010 shows window-dependence → Phase 2 with priors set by
which windows differ. If not → J-space frame survives only if EXP_011 delivers.

### Phase 2 — J-space core (needs Phase 0 gate passed)
- **EXP_011 — J-space overlap (H6).** Project the five real basin tensors and the 18
  null-model basin tensors (both in the frozen Stage 1 archives) onto the J-lens
  subspace vs its complement; compare energy fractions with a permutation null (the
  Stage 1 permutation-test pattern reused).
  *Decides:* whether "semantic basin" = "workspace-captured state" — the mechanistic
  reading of the Stage 1 null result.
- **EXP_013 — J-corrected readout (Q-D).** Re-decode the gated trajectories through
  the J-lens instead of `ln_final → W_U`, `Divine` prompts first.
  *Decides:* whether the Divine dissociation is a readout-coordinate illusion or a
  genuine dynamical object.

**Decision point 2:** census go/no-go. Census only if H5 or H6 supported.

### Phase 3 — Workspace census (conditional)
- **EXP_012 — Band structure across the four Stage 1 models (H7).** J-lens per model;
  compare band presence/shape to the ATR regime table. The heaviest compute in the
  plan; scope after Phase 2 (possibly GPU-budgeted).

---

## Requirements

**Code & instruments**
- Anthropic J-lens replication repo (P0-1) — the critical external dependency.
- TransformerLens (installed); `atr_engine.py` + window extension (P0-4).
- Existing patterns to reuse: `gated_resweep.py` (convergence gating),
  `02b_permutation_test.py` (permutation nulls), `cos_sim_diagnostic` (tensor-level
  verdicts).

**Data (read-only, frozen)**
- Stage 1 trajectories: `experiments/*/output*/**.pt` in the ATR repo working copy;
  off-machine backup `_DATA/EXP009_stage1_trajectories_2026-07-10.zip` (fold repo).
- Canonical numbers: `docs/FINDINGS.md` in the ATR repo (do not re-derive).

**Compute & environment**
- Phase 1 runs entirely on this machine's CPU (est. one overnight total).
- Phase 0/2 estimate pending code inspection; cloud GPU hours are the fallback
  accelerator (cost pass waived by decision 2026-07-10).
- Models cached locally: gpt2-small, gpt2-medium, pythia-160m, pythia-410m.
- **Environment trap (recorded):** Python 3.12 = ML stack (torch, transformer-lens);
  Python 3.11 = viz stack (matplotlib, sklearn, imageio). Run experiments on 3.12,
  figures on 3.11, or unify into one venv as a P0 chore.
- Disk: C: at ~94% — clear space before J-lens artifacts land (do NOT touch the HF
  model cache or the `.pt` archives).

**Working arrangement (the Stage 1 pattern, kept)**
- Fable session: experiment design, verdicts, synthesis, editorial register.
- Opus session(s): runbook execution — one runbook per phase, same ground rules as
  `CROSS_MODEL_RUN_PLAN.md` (commit after each experiment, never touch declared
  frozen inputs, deviations documented in the results summary).
- All Stage 2 material lives in `_STAGE2_JSPACE/`; the public ATR repo is not
  touched until Stage 2 has a publishable story of its own (carve-out at maturity,
  as EXP_009 was).

**Deliverables**
- Per phase: a results summary section (RESULTS_SUMMARY pattern) + updated hypothesis
  dispositions in this folder.
- End state: either a workspace-grounded explanation of the Stage 1 anomaly, or a
  clean kill of the band-structure theory with the anomaly restated — both
  publishable, per the house rule that negatives are findings.
