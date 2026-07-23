# Runbook — Phase 1: Bridge Experiments (no J-lens required)

**For:** an operator session executing on this machine. Read fully before acting.
**Plan:** `STAGE2_PLAN.md` (this folder) — Phase 1. Independent of Phase 0; either
order works, and a single session may run this before or after RUNBOOK_PHASE0.
**Working dir:** `C:\Users\Fab2\Desktop\AI\_learn\_fold\03_MECHINIPHYLUM\_STAGE2_JSPACE`

## Ground rules

Same as RUNBOOK_PHASE0 §Ground rules (read them there): ATR repo is READ-ONLY;
commit+push this folder to the private fold repo after each experiment; python 3.12
for runs, matplotlib installed there (or figures via 3.11); disk is tight — document,
don't delete; deviations documented in `RESULTS_PHASE1.md`.

**Engine:** use `experiments\atr_engine2.py` if Phase 0's P0-4 fork exists; otherwise
create it now exactly per P0-4 (copy + window params + equivalence check) — it is the
only Phase 0 artifact this runbook needs.

**Frozen inputs (read-only paths):**
- `..\_LAB_NOTEBOOKS\lucier-repo\experiments\gpt2_small\output_gated\gated_results.pt`
- `..\_LAB_NOTEBOOKS\lucier-repo\experiments\pythia_410m\output_deep\deep_results.pt`
- `..\_LAB_NOTEBOOKS\lucier-repo\prompt_library.py` (import the 125 prompts)
- Canonical Stage 1 numbers: `..\_LAB_NOTEBOOKS\lucier-repo\docs\FINDINGS.md`

Run order: EXP-D (fast, pure analysis) → EXP_010a (~2 h) → EXP_010b (overnight).

---

## EXP-D — The `Divine` object (Q-D)

**Question:** the 34 never-converging prompts are all `Divine` — stable decode over a
moving tensor. Limit cycle, wandering attractor, or decode-region plateau?

1. **Schema first.** Load `gated_results.pt`, print its structure, and RECORD it.
   Determine whether per-iteration mean vectors (or full tensors) are stored for the
   non-converged prompts. If trajectories are dense enough (≥ every 10 iters over a
   long horizon), analyse directly and skip step 2.
2. **Dense re-capture (only if needed).** Using atr_engine2 on gpt2-small, re-run a
   subset with dense capture of `mean_vector` (every iteration to 500):
   - 10 `Divine` prompts (deterministic pick: alphabetically first 10 by prompt ID
     among the 34 — derive the 34 from the gated results, RECORD the list),
   - plus 3 converged controls (first alphabetical from `prolet`, `till`, `Anarch`).
   Save to `experiments\exp_d_divine\divine_trajectories.pt`. (~1–2 h CPU.)
3. **Analysis** (script `experiments\exp_d_divine\analyze_divine.py`):
   - Step-size spectrum: `1 - cos(mean_t, mean_{t+1})` over t — steady, decaying, or
     oscillating?
   - Periodicity: FFT of several mean-vector components and of the step-size series;
     a limit cycle shows spectral peaks. Autocorrelation as cross-check.
   - Recurrence: pairwise cosine matrix of mean vectors across t (recurrence plot);
     banded structure = cycle; drift off-diagonal = wandering.
   - Geometry: PCA (2–3 components) of each trajectory; plot; controls should
     contract to points, `Divine` should show its shape.
   - Readout stability: top-1 token and logit margin along the trajectory
     (engine's R1 metrics) — confirm decode constancy while the tensor moves,
     and RECORD the margin's behaviour.
4. **Verdict** in RESULTS_PHASE1.md — one of: limit cycle / wandering within a decode
   region / slow transient after all / other (describe). Evidence lines per the
   analyses above. Figures saved to `experiments\exp_d_divine\` and committed.

**Decides:** Q-D at the tensor level. Any outcome is a finding.

---

## EXP_010a — Depth control on Pythia-410m (H8)

**Question:** is 410m's fragmentation depth-driven? Loop layers 0→11 vs native 0→23.

1. Prompt set (deterministic, RECORD it): the 8 prompts used in Stage 1's deep run
   (listed in `..\...\pythia_410m\output_deep\config.md` or derivable from
   `deep_results.pt`) **plus** the alphabetically-first 17 additional prompts drawn
   round-robin across the 7 categories of `prompt_library.py` → 25 total.
2. Two arms, same 25 prompts, same seed, gated protocol (cos > 0.999 ×3, checks
   every 10 iters past 100, `max_iter=1000`), checkpoint per prompt:
   - **Arm A (control):** inject 0, extract 23 (native full stack — replicates the
     Stage 1 regime on this subset).
   - **Arm B (treatment):** inject 0, extract 11 (12-layer "GPT-2-Small-shaped" loop).
3. Outputs to `experiments\exp_010a_depth\output\` (same artifact set as Stage 1:
   config, per-prompt terminals, lock-in iterations, dissolution pathways).
4. Compare: converged fraction, unique-terminal count, cross-prompt mean-vector
   similarity, basin consolidation (any shared terminals in Arm B?).

**Decides (RECORD explicitly):** if Arm B consolidates or converges where Arm A
fragments → depth is causal (H8 supported). If both fragment alike → depth alone is
not the driver. Runtime ~2 h CPU total (Arm B is half-depth, cheaper).

---

## EXP_010b — Window grid on GPT-2 Small (H5, coarse)

**Question:** does the attractor landscape depend on where the loop is cut?

1. Prompt set (deterministic, RECORD it): 25 prompts = 5 known-`Divine` prompts
   (alphabetically first among the 34) + 20 others round-robin across categories,
   alphabetical within category, excluding the Divine picks.
2. Windows (inject i → extract j), all with the gated protocol, `max_iter=1000`,
   checkpoint per (window, prompt):
   - `0→11` baseline (must reproduce Stage 1 basins on this subset — treat as a
     reproduction gate for the fork; if it disagrees, STOP and document)
   - `0→5`, `3→8`, `6→11` (6-layer windows: early / mid / late)
   - `0→8`, `3→11` (9-layer windows: front-heavy / back-heavy)
3. Readout convention: classify terminals with the SAME `ln_final → W_U` decode as
   Stage 1 (comparability), and **save the terminal mean vectors** per (window,
   prompt) to `experiments\exp_010b_windows\terminals.pt` — Phase 2's EXP_013 will
   re-decode them through the J-lens.
4. Per window, record: converged fraction, lock-in iterations, terminal tokens,
   basin table, cross-prompt similarity matrix. One summary table across windows.

**Decides (RECORD explicitly):** qualitative window-dependence — do any windows
produce a landscape unlike the full stack (more/fewer basins, semantic vs junk,
different convergence character)? Which window is "richest"? That window is the
candidate workspace band for Phase 2 priors. Runtime: overnight CPU (~6–9 h);
run last, leave it going.

---

## Deliverable

`RESULTS_PHASE1.md`: per-experiment sections in the Stage 1 RESULTS_SUMMARY style
(What ran / Headline numbers / Decides / Interpretation / Open questions), the three
explicit prompt lists, all deviations. Commit and push after EACH experiment, not
just at the end. Final line: `PHASE 1 COMPLETE — H8: <verdict> · H5(coarse):
<verdict> · Q-D: <verdict>`.
