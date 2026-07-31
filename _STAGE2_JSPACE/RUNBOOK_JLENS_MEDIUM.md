# Runbook — J-lens Follow-on for GPT-2 Medium (Phase 2, Medium track)

**For:** an operator session executing after EXP_010c has reported.
**Depends on:** `EXP_010c_SPEC.md` results (the ATR window grid on Medium) and
Phase 0's instrument gate (`RUNBOOK_PHASE0.md` P0-2/P0-3, extended to Medium).
**Status:** PLANNED — priors in §3 are deliberately left open until EXP_010c reports.
**Register:** reporting register; same ground rules as RUNBOOK_PHASE0/1 (commit after
each experiment, frozen inputs read-only, deviations documented).

---

## 1. Purpose

EXP_010c asks whether Medium's `D` collapse depends on where the ATR loop is cut.
This runbook asks the same question **through the paper's own instrument**: fit a
J-lens for GPT-2 Medium, establish whether the model even *has* a coherent
workspace band at 345M scale, and read the EXP_010c terminal tensors through it.
The three sub-experiments are Medium instances of the already-planned EXP_011/012/013
patterns, so the taxonomy carries over:

| ID | Pattern | Question |
|---|---|---|
| EXP_012m | Band census (EXP_012 pattern, one model) | Does Medium have a coherent J-lens band structure at all? Where does it sit relative to the naive 40–90% mapping used by EXP_010c? |
| EXP_011m | Subspace overlap (EXP_011 pattern) | Do EXP_010c terminal tensors (per window) project onto Medium's J-space differently by window? Does the baseline `D` tensor live *outside* the workspace where escaped-window terminals live *inside* it? |
| EXP_013m | J-corrected readout (EXP_013 pattern) | Re-decode window trajectories through the J-lens instead of `ln_final → W_U`: does the `D` readout hide moving verbalizable content the logit lens cannot see? |

## 2. Instrument step (extends Phase 0 to Medium)

1. Fit the lens: `anthropics/jacobian-lens`, `jlens.from_hf` on gpt2-medium,
   ~100 pretraining-like prompts to start (repo `data/` sets), checkpointed,
   scale toward 1000 only if the validation gate demands it. CPU-feasible at
   345M; backward passes dominate cost.
2. **Validation gate (P0-3, Medium edition):** mid-layer J-lens readouts on
   multi-hop prompts must be qualitatively more interpretable than logit-lens
   readouts at the same layers, with sane per-layer progression. No Stage 2
   conclusion uses the Medium lens before this gate passes. Record examples
   either way — at 345M, *failing* this gate is itself evidence for EXP_012m
   (no coherent workspace to find) and must not be silently retried into
   passing.

## 3. Refinement priors — branch on EXP_010c's outcome

The EXP_010c outcome table (spec §6) sets what to look at first:

| EXP_010c outcome | J-lens focus |
|---|---|
| H9 + H9a supported (band window escapes) | EXP_011m first: confirm escaped-window terminals are J-space-loaded and the baseline `D` terminal is not. EXP_012m checks the empirical band against the escaping window's coordinates. Strongest available reading: ATR and J-lens localise the same band by independent means. |
| H9 supported, H9a not (wrong window escapes) | EXP_012m first: map the actual band; the naive 40–90% mapping is presumed wrong for this model. Re-run the EXP_010c *analysis* (not the sweeps) regrouping windows by the empirical band. |
| H9 refuted (everything says `D`) | EXP_012m decides whether there was ever a band to find: a missing/incoherent band at 345M would *explain* the refutation inside the workspace frame (no workspace, nothing to rescue) and directly feeds H7's cross-model claim. EXP_013m on the baseline trajectory asks whether anything verbalizable moves beneath the constant `D`. |
| Fragmentation everywhere / length confound | EXP_013m on fragmenting trajectories: is fragmentation motion *within* verbalizable directions or outside them? Defer EXP_011m until the ATR picture stabilises (no-splice control from spec §3). |

## 3a. Census hand-forward (2026-07-29 — supersedes the §3 branch table's
## input, not its logic)

`EXP_010c4_SPEC.md`'s census has measured **all 300 valid (i, j) windows**
at the registered protocol, so the branch table above is now entered with
complete data rather than a sampled grid. What the map hands forward
(observations; see the 2026-07-29 section of RESULTS_EXP010C.md):

- **Target set for EXP_011m/013m — 21 cells** with whole-word,
  prompt-dependent terminals. These are **not** a contiguous band and
  include single-layer windows (10→10, 13→13, 20→20, 21→21).

  **Exact target-to-artifact manifest.** 19 of the 21 are census arms;
  **two are not** — they were measured by earlier tiers and are absent
  from `terminals_census/`. Loading only `terminals_census/` silently
  drops 8→21 and 10→21. All paths are relative to
  `experiments/exp_010c_windows/output/`; every tensor file is a dict
  keyed `"<arm>|<prompt_id>"`.

  | Cells | Arm | Artifact |
  |---|---|---|
  | 5→23, 6→23, 8→9, 8→11, 8→16, 9→11, 9→20, 10→10, 10→16, 12→15, 13→13, 13→21, 14→16, 15→17, 15→19, 16→18, 17→20, 20→20, 21→21 (19) | `W<i>_<j>` | `terminals_census/W<i>_<j>.pt` |
  | 8→21 | `O8` | `terminals_scan.pt` |
  | 10→21 | `A4` | `terminals_full.pt` |

  The remaining pre-census cells (not targets) live in
  `terminals_full.pt`, `terminals_scan.pt` and `terminals_infill.pt`;
  `results_*.json` / `results_census/*.json` carry the matching
  per-run records.
- **Contrast set:** 91 punctuation-funnel cells and 26 whole-word funnel
  cells (prompt-*independent*) — the natural comparison classes for a
  projection test, alongside the A0 (0→23) `D` terminals.
- **`D` is a single-cell phenomenon:** exactly one window in 300 produces
  it (0→23). EXP_013m's baseline-trajectory question is therefore about
  that one cut, not about deep windows generally.
- **Readout-reliability prior (corrected by the census — read this, not
  the sampled version):** the pre-census tiers suggested agreement was
  high only at extraction 21 and collapsed at j ∈ {15,17,19}. **That does
  not survive the full census.** Over all 277 census arms, direct-decode
  vs via-tail agreement rises roughly monotonically with extraction depth
  — ≈0/25 at j ≤ 4, ~5/25 at j ≈ 13, 18.9/25 at j=22 — and **j=21's
  census arms average only 6.6/25** (the old prior rested on two cells;
  there are eleven at j=21). The j=23 figure is not a via-tail
  measurement at all (empty tail = mean-vs-last-position check). Full
  table in the 2026-07-29 results section. Treat every sub-22 terminal
  identity as instrument-dependent until the J-lens arbitrates.
- **Dissociation to plan around:** agreement across the 21 target cells
  spans 0/25 to 25/25 (25/25 at 5→23, 6→23, 20→20; 0/25 at 8→11, 8→16,
  15→17, 21→21). The whole-word prompt-dependent class and the
  high-agreement class are **not** the same set, so two-instrument
  agreement cannot be used to validate the target cells — EXP_013m is
  load-bearing, not confirmatory.
- **New ATR-side priority, independent of the J-lens:** six cells at
  i ≤ 1 show systematic non-convergence (drifting, not cycling). Together
  with `D`'s isolation to the 0→23 splice, this makes the hook-point
  control (`resid_post` at i−1 vs `resid_pre` at i; i=1 vs i=0) the
  highest-value remaining control in the planned-controls list.

## 4. Procedures

**EXP_012m — band census.** Apply the fitted lens layer-by-layer to a held-out
prompt set (~50 prompts, none from the fitting set, none from the ATR prompt
subset). Per layer record: readout interpretability (top-k lens tokens vs
logit-lens tokens), agreement-with-final-layer curve, and the layer range where
lens ≫ logit-lens. Deliverable: Medium's empirical band `[L_lo, L_hi]` (or a
recorded "no coherent band" verdict) + comparison against the 10–21 mapping.

**EXP_011m — terminal projection.** Frozen input: the per-target artifacts
named in the §3a manifest — `terminals_census/W<i>_<j>.pt` for 19 targets,
plus `terminals_scan.pt` (arm `O8`, 8→21) and `terminals_full.pt` (arm `A4`,
10→21). There is no aggregate `terminals.pt`; load per the manifest or the
two non-census targets are silently missing. For each (window, prompt)
terminal mean vector at the window's extract layer j: decompose into the J-space
component (sparse nonnegative combination of ≤25 J-lens vectors at layer j, per
the paper's construction) vs complement; record energy fractions. Null:
permutation test with token-shuffled lens vectors (Stage 1 `02b_permutation_test`
pattern — the anisotropy lesson applies; naive nulls flatter overlap). Compare
distributions across windows and against the A0 baseline `D` terminals.

**EXP_013m — J-corrected readout.** Requires dense trajectory capture for the
selected runs (re-run those (window, prompt) pairs with mean-vector capture every
iteration; the EXP-D dense-capture pattern). Decode each captured iterate through
the lens at the extract layer; record top-k lens tokens over time. Verdict
language mirrors Q-D: constant readout / cycling readout / drifting readout, at
the lens level vs the logit level.

## 5. Deliverable

`RESULTS_JLENS_MEDIUM.md` in this folder: per-experiment sections (What ran /
Headline numbers / Decides / Interpretation / Open questions), the fitted-lens
checkpoint path and fitting config, validation-gate evidence, all deviations.
Final line: `MEDIUM J-TRACK COMPLETE — H9-mechanism: <verdict> · EXP_012m band:
<[lo,hi] or none> · EXP_011m overlap: <verdict> · EXP_013m readout: <verdict>`.

Kill-criteria note: this track feeds, but does not replace, the plan-level kill
criteria in STAGE2_PLAN.md — a dead band on Medium plus a dead window grid on
Small plus null EXP_011 overlap on Small is the conjunction that ends the
workspace framing.
