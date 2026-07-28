# EXP_012-PYTHIA — Placebo Window Grid on Pythia-410m (pre-registered spec)

**Status:** PRE-REGISTERED — recorded before any registered run.
**Created:** 2026-07-25
**Issue:** #12. **Parent plan:** `STAGE2_PLAN.md`; closes RESULTS_EXP010C.md
§"Planned controls" item 3 AND the Stage 1 depth control that has been open
since Phase 1 (Control 2 → H8 / EXP_010a in `RUNBOOK_PHASE1.md`).
**Register:** reporting register. Every outcome recorded, verdict criteria
stated before execution. Observations only for the cross-model comparison —
no cross-model interpretation in the results register.

---

## 1. Question

Two threads, one sweep:

1. **Placebo grid (planned-controls item 3).** The Medium window map
   (whole-word terminals only at i ∈ {8,10}; funnels elsewhere; full stack
   unique in producing the `D` collapse) could be specific to gpt2-medium or
   generic to 24-layer decoder stacks of this size. Pythia-410m is the
   placebo: Stage 1 found **no consolidation** there (full-stack ATR: 40+
   fragments, ~0.85 cosine plateau, 9/125 prompts converge; deep 8-prompt run
   at 1000 iters: 8 distinct terminals, cross-prompt sim 0.21 —
   `docs/FINDINGS.md`, public repo). Structured window-position effects on a
   model with no known consolidation would reframe the Medium observations.
2. **H8 depth control (EXP_010a, never run).** Is 410m's fragmentation
   depth-driven? The 0→11 vs 0→23 pair inside the grid is exactly the
   EXP_010a design: same 25 prompts, same seed, same gated protocol.

## 2. Model and provenance

**Model:** EleutherAI/pythia-410m (GPTNeoX, 24 layers, d_model=1024,
d_vocab=50304), via TransformerLens `HookedTransformer.from_pretrained("pythia-410m", hf_model=..., tokenizer=...)`
with the HF model (`GPTNeoXForCausalLM`) and `AutoTokenizer` loaded from a
local dir. CPU. torch 2.13.0, transformer_lens 3.5.1, transformers 5.14.1.

**Acquisition route (recorded):** direct from huggingface.co, allowed since
the 2026-07-25 network-policy update (`REMOTE_ENV_MODEL_ACCESS.md`). Files
`https://huggingface.co/EleutherAI/pythia-410m/resolve/main/{config.json,tokenizer.json,pytorch_model.bin}`
(GPTNeoX ships `tokenizer.json`, not vocab/merges). Downloaded 2026-07-25;
sha256 digests (provenance convention per PR #19 review):

| File | Bytes | sha256 |
|---|---|---|
| config.json | 570 | `d4c11e84a59c8af4d88446bba53b718f7aef740daa070ded08fd6a9a3aca4fc6` |
| tokenizer.json | 2,113,710 | `c24618a1b3e6a38167beff1c72cffd126c3a66254347304b50547d12c5f25624` |
| pytorch_model.bin | 911,449,213 | `786bd9bcde112a970e1614807cbb23999482de65fc05c3e0795198719b5e1543` |

State dict verified before any run: 364 tensors, embed_in [50304, 1024],
24 layers.

**Reproduction gate:** unlike EXP_010c's A0 gate (Medium must reproduce `D`),
Pythia-410m's Stage 1 full-stack behaviour is *non-convergence with
fragmented terminals*. P-A0 is therefore the H8 **control arm**, not a
stop-gate: whatever it does is recorded and becomes the baseline for the H8
comparison. A *consolidated single-token converged* P-A0 would contradict
Stage 1 and is flagged loudly if seen (different protocol from Stage 1's
fixed-horizon runs, so not an automatic STOP — recorded and investigated).

## 3. Design

**Harness:** `experiments/exp_010c_windows/run_exp010c.py` generalised with a
`--model-name` parameter (default `gpt2-medium`; recorded minimal diff). Arm
keys reuse the existing window definitions (single source of truth); the
results register reports them with a `P-` prefix. Artifacts suffix:
`pythia410m`. **Ordering note (spec-first rule, recorded per PR #39
review):** this spec pre-registers the design and therefore precedes the
runner-generalisation diff by design; the `--model-name` diff (plus the
GPTNeoX offline loader and the natural-norm record for seed_j runs) is
committed after this spec and BEFORE any registered arm's artifacts. The
committed `output/prompt_subset_pythia.json` is the execution authority for
the prompt set: the runner re-derives and hard-stops on any mismatch.

**Windows (inject i → extract j, inclusive; both models are 24-layer, so the
Medium grid's absolute layers ARE the same relative depths):**

| Arm | Runner key | Window | Role |
|---|---|---|---|
| P-A0 | A0 | 0→23 | Baseline / native full stack — **H8 control arm** (Stage 1 regime on this subset) |
| P-A1 | A1 | 0→11 | **H8 treatment arm** (EXP_010a Arm B, the 12-layer "GPT-2-Small-shaped" loop) |
| P-A2 | A2 | 6→17 | Placement sweep: middle |
| P-A3 | A3 | 12→23 | Placement sweep: back |
| P-A4 | A4 | 10→21 | Band-exact (the Medium map's band window) |
| P-O8 | O8 | 8→21 | Onset-edge arm at the Medium i ∈ {8,10} whole-word zone |

**Deviation recorded (issue text vs this spec):** issue #12's body lists the
sixth window as 8→15 (Medium arm A5, the length probe). The executing task
directive substitutes **O8 (8→21)** — the onset-edge arm that, with A4,
anchors the Medium map's i ∈ {8,10} whole-word zone and carried the
robustness/energy controls (EXP_010c-ROBUST, -VARIANTS Control B). Recorded
flat as a deviation from the issue body; A5 (8→15) on Pythia is follow-on.

**Prompts (deterministic, recorded):** 25 = the Stage 1 Pythia-410m deep-run
8 **plus** 17 round-robin. Derivation (`derive_prompts_pythia.py`, output
committed as `output/prompt_subset_pythia.json`):

- **Core 8** extracted from the committed backup
  `_DATA/EXP009_stage1_trajectories_2026-07-10.zip` →
  `experiments/pythia_410m/output_deep/deep_config.pt` / `deep_results.pt`.
  Structure record: `deep_results.pt` is a dict keyed by the 8 prompt IDs,
  each value a dict with keys {iterations, last_vectors, mean_vectors,
  last_norms, mean_norms, cosine_sims_last, cosine_sims_mean,
  position_similarity, top_tokens, all_position_tokens}; `deep_config.pt`
  records `prompt_keys` = [A01_physics, B01_napoleon, C01_jack_jill,
  D01_water, E01_politics, F01_anger, G01_punctuation, G13_buffalo],
  layer_start 0, layer_end 23, max_iterations 1000, model pythia-410m.
- **Plus 17** drawn round-robin across the 7 categories of the restored
  `prompt_library.py` (public lucier repo @ main, pulled 2026-07-25;
  provenance-flagged reconstruction, all 125 entries `original`), categories
  in alphabetical order, alphabetical by prompt ID within category,
  excluding the core 8.

The full 25, in execution order:

| # | ID | Category | Prompt |
|---|---|---|---|
| 0 | A01_physics (core8) | Complex | The implications of quantum entanglement suggest that |
| 1 | B01_napoleon (core8) | Narrative | Napoleon crossed the Alps with an army of |
| 2 | C01_jack_jill (core8) | Simple | Jack and Jill went up the hill to |
| 3 | D01_water (core8) | Chemical | H2O NaCl CO2 O2 Fe2O3 CH4 NH3 |
| 4 | E01_politics (core8) | Acronyms | NATO EU UN ASEAN BRICS G7 IMF WTO |
| 5 | F01_anger (core8) | Vulgarity | What the fuck is wrong with you you |
| 6 | G01_punctuation (core8) | Wild | ... --- !!! ??? ,,, ;;; ::: ((( ))) |
| 7 | G13_buffalo (core8) | Wild | buffalo buffalo buffalo buffalo buffalo buffalo buffalo |
| 8 | E02_tech | Acronyms | HTTP API REST JSON SQL TCP UDP SSH |
| 9 | D02_periodic | Chemical | He Ne Ar Kr Xe Rn Og Ts Lv |
| 10 | A02_medical | Complex | A meta-analysis of randomised controlled trials indicates |
| 11 | B02_wwi | Narrative | The assassination of Archduke Franz Ferdinand in Sarajevo |
| 12 | C02_king_cole | Simple | Old King Cole was a merry old soul |
| 13 | F02_insult | Vulgarity | You stupid piece of shit I told you |
| 14 | G02_brackets | Wild | [ ] { } ( ) < > &#124; / \ ^ ~ ` |
| 15 | E03_orgs | Acronyms | FBI CIA NSA DOJ IRS SEC FDA CDC |
| 16 | D03_organic | Chemical | CH3CH2OH COOH C6H12O6 ATP ADP |
| 17 | A03_neuro | Complex | The hippocampal formation plays a critical role in |
| 18 | B03_moon | Narrative | One small step for man one giant leap |
| 19 | C03_mary_lamb | Simple | Mary had a little lamb its fleece was |
| 20 | F03_frustration | Vulgarity | For fucks sake how many times do I |
| 21 | G03_counting | Wild | 1 2 3 4 5 6 7 8 9 |
| 22 | E04_internet | Acronyms | LOL LMAO ROFL IMHO TBH SMH FWIW |
| 23 | D04_equation | Chemical | 2H2 + O2 -> 2H2O delta G = - |
| 24 | A04_climate | Complex | Anthropogenic climate change has accelerated the rate of |

(Text authority is `output/prompt_subset_pythia.json`, generated by the
derivation script from `prompt_library.py`; the table above is a convenience
copy. Any transcription mismatch defers to the JSON.)

**Protocol:** identical to the registered Medium grid: gated, cos > 0.999 ×3
(patience 3), checks every 10 iters past `check_start=100`, `max_iter=1000`,
`gate_lag=1`, seed 42, L0 natural-pass seeding (spec EXP_010c §3: seed =
natural full-pass `resid_post` at j; loop injects at `resid_pre` of i).
Post-hoc `lag_scan` (lags 1–8) recorded per run. Readout: `ln_final → W_U`
at j (Stage-1 convention) plus decode-via-tail (through j+1→23) in the
characterisation step. Execution order: P-A0, P-A1, P-A2, P-A3, P-A4, P-O8;
commit per arm.

**Smoke gate (non-registered, before the sweep):** 2 prompts × {A0, A4},
max_iter=60, real weights — verifies GPTNeoX hook names
(`blocks.{i}.hook_resid_pre` / `hook_resid_post`) resolve in
transformer_lens and the generalised loader works. No verdict weight.

## 4. Energy convention (pre-registered, in light of today's Control B)

The registered sweep runs under the **registered `seed_j` convention**
(loop rescales to the seed's extraction-layer norm) — the same convention as
the registered Medium grid, for comparability.

**Known caveat, stated up front:** EXP_010c-VARIANTS Control B (2026-07-25)
showed that on Medium the i=0 arms re-inject at ~218× (A0) and ~307× (A1)
the natural layer-0 `resid_pre` norm under `seed_j`, and that their landscape
class changes under `natural_i` renorm (funnels → non-converging multi-
terminal), while in-band arms (A4, O8) ran within ~5% of natural norm and
were unchanged. Therefore, for THIS run:

- The natural per-layer `resid_pre` norms are measured and recorded for all
  25 prompts (one un-hooked pass each), and the results register reports the
  **per-arm ratio seed_j-target-norm / natural-resid_pre(i)-norm**.
- **Expected caveat, `seed_j` convention:** the i=0 arms (P-A0, P-A1 — both
  H8 arms) inherit the "injection loudness" caveat: any funnel/fragmentation
  observed there is conditional on the j-scale energy convention until a
  `natural_i` variant runs. The H8 verdict is therefore stated as a verdict
  *under the registered Stage-1-comparable convention* (which is also the
  convention the original EXP_010a design implied, since it predates the
  renorm distinction).
- **Expected caveat, `natural_i` convention (not run here):** would break
  comparability with both Stage 1 Pythia runs and the registered Medium
  grid. **No `natural_i` arms in this registered run — recorded as
  follow-on** (mirror of Control B on Pythia).

## 5. H8 pre-registered reading (verbatim criterion from RUNBOOK_PHASE1 §EXP_010a)

Compare P-A1 (0→11, treatment) vs P-A0 (0→23, control) on: converged
fraction, unique-terminal count, cross-prompt mean-vector similarity, basin
consolidation (any shared terminals in the treatment arm).

| Observation | Verdict (mechanical) |
|---|---|
| P-A1 consolidates or converges where P-A0 fragments | **Depth is causal — H8 supported** |
| Both fragment alike | **Depth alone is not the driver — H8 not supported** |

Edge cases recorded flat: if P-A0 itself consolidates/converges on this
subset (contradicting Stage 1), the H8 comparison is still run mechanically
but flagged as baseline-anomalous; if P-A1 fragments MORE than P-A0, that is
"both fragment alike" for the binary verdict, with the direction recorded.

## 6. Grid pre-registered readings (placebo question; observations only)

Reference map (registered Medium grid, seed_j, from RESULTS_EXP010C.md):
A0 `D`-funnel converged / A1 `','`-funnel 1 basin / A2 low-margin funnel
1 basin / A3 `'"'` 21 basins / A4 whole-word {until, forever, since} 12
basins high margin / O8 whole-word {simultaneously, halfway} 11 basins.

| Observation on Pythia | Reading (recorded flat, no cross-model interpretation) |
|---|---|
| Window-position effects ABSENT: all six arms behave alike (all fragment, or all funnel), no in-band/off-band asymmetry | The Medium window-position structure does not generalise trivially to this 24-layer placebo; consistent with it being model-specific. Recorded as observation only. |
| Window-position effects PRESENT: in-band arms (P-A4, P-O8) differ systematically from off-band (P-A1, P-A2, P-A3) in basin count, convergence, margin, or terminal class | Structured window effects exist on a model with no known full-stack consolidation; the Medium map's "window position matters" component is not Medium-specific. Which specific features coincide (whole-word terminals? basin counts?) is tabulated per arm, flat. |
| Partial/mixed (some arms differ, not band-aligned) | Tabulated per arm; no verdict beyond the H8 line. Feeds the cross-model comparison as observations. |

The results register carries a comparison-ready table (same columns as the
Medium map) and NO cross-model interpretation — that belongs to a later
synthesis pass.

## 7. Artifacts and results

- `output/results_pythia410m.json`, `output/terminals_pythia410m.pt`
  (per-arm checkpointed), `output/prompt_subset_pythia.json`,
  `output/natural_resid_norms_pythia410m.json`,
  `output/terminal_characterisation_pythia410m.json`.
- Results register: **new file** `experiments/exp_010c_windows/RESULTS_EXP012_PYTHIA.md`
  (recorded choice: the Medium register RESULTS_EXP010C.md stays
  single-model; the Pythia register lives beside it, sharing the harness
  dir), plus a pointer + planned-controls item 3 tick in RESULTS_EXP010C.md.
- Explicit `H8:` verdict line (mechanical application of §5).

## 8. Cost

6 arms × 25 prompts, gated, max_iter=1000, CPU. Pythia-410m ≈ 3× Medium's
per-iteration cost; non-converging arms hit the 1000-iter cap. Estimate
25–75 min per arm, 2.5–7.5 h total. Sequential, commit+push per arm.
