# EXP_010b — Window grid on GPT-2 Small (H5, coarse) — pre-registered spec

**Status:** committed BEFORE any run (registered). Issue: #16. Source protocol:
`RUNBOOK_PHASE1.md` §EXP_010b, as amended by the issue #16 comments (Divine
derivation from the committed Stage 1 backup zip; gpt2-small weights via the
remote-verified route). **Direction call:** the issue asked for an explicit
go/no-go before scheduling (Small grid post-Medium-pivot). The operator made
the call: RUN. Recorded here.

**Executing line:** `agent:exp010b-small` (peer-board claim posted before this
spec; no other PR/branch/board thread claims issue #16).

## 1. Question and hypothesis

Does the attractor landscape of GPT-2 **Small** — the only model with semantic
Stage 1 basins — depend on where the loop is cut?

**H5 (STAGE2_PLAN.md, verbatim):** "Layer-window loops (inject i, extract j)
produce qualitatively different landscapes *within* the putative workspace band
vs *across* band boundaries." EXP_010b is the **coarse** test.

**Pre-registered H5-coarse reading (RUNBOOK_PHASE1.md §EXP_010b "Decides",
verbatim):** "qualitative window-dependence — do any windows produce a
landscape unlike the full stack (more/fewer basins, semantic vs junk, different
convergence character)? Which window is 'richest'? That window is the candidate
workspace band for Phase 2 priors."

Mechanical rule for the verdict line (fixed now): H5-coarse is **SUPPORTED** if
at least one sub-stack window differs from the 0→11 baseline on any of the
pre-stated comparison fields (converged fraction, tensor-basin count at the
0.999 cluster threshold, decode-terminal class word/fragment/punctuation,
prompt-dependence of terminals); **NOT SUPPORTED** if all five sub-stack
windows reproduce the baseline landscape on all fields. Anything richer than
the verdict line is observation, not interpretation.

Secondary observation fields (recorded flat, no verdict): what happens to the
Stage 1 `Divine` prompts under each window — do they stay non-converging with a
period-2 signature (lag-1 cosine ≪ 1, lag-2 cosine ≈ 1 in `lag_scan`), start
converging, or change class? `lag_scan` is reported flat for every arm.

## 2. Model and acquisition (recorded)

`gpt2` (Small, 12 layers, d_model 768), downloaded 2026-07-26 from
huggingface.co directly (the preferred route per `REMOTE_ENV_MODEL_ACCESS.md`
after the 2026-07-25 policy update): `https://huggingface.co/gpt2/resolve/main/<file>`.

| file | bytes | sha256 |
|---|---|---|
| config.json | 665 | 0daed7749b4f02b8f76240d5444551d7b08712dab4d0adb8239c56ba823bb7b4 |
| vocab.json | 1042301 | 196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783 |
| merges.txt | 456318 | 1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5 |
| pytorch_model.bin | 548118077 | 7c5d3f4b8b76583b422fcb9189ad6c89d5d97a094541ce8932dce3ecabde1421 |

State-dict verification (performed before this spec was committed): 160
tensors; `wte.weight` = [50257, 768]; transformer blocks `h.0 … h.11` (12
layers); config `n_layer=12, n_embd=768, n_head=12, vocab_size=50257`.

Provenance caveat as for every remote acquisition: the experiment's own
reproduction gate (§4) is the behavioural check on these weights. Offline load
via the runner's `--model-path` + `--model-name gpt2` (recorded diff, §6).

## 3. Prompt set (deterministic, recorded)

**Rule (RUNBOOK_PHASE1.md §EXP_010b step 1):** 25 prompts = 5 known-`Divine`
prompts (alphabetically first among the 34) + 20 others round-robin across
categories, alphabetical within category, excluding the Divine picks.

**Divine derivation source (issue #16 second comment — the canonical file):**
`_DATA/EXP009_stage1_trajectories_2026-07-10.zip` →
`experiments/gpt2_small/output_gated/gated_results.pt` (14,353 bytes).

**Schema (inspected and recorded):** loads with `weights_only=True`; a dict of
125 prompt-id → dict with keys `terminal_token` (str), `terminal_token_id`
(int), `terminal_prob` (float), `lock_in_iter` (int|None), `converged` (bool),
`n_iters` (int), `final_cos_sim_mean` (float), `top_logit_margin` (float),
`entropy` (float). No per-iteration trajectories. The 34 `Divine` prompts are
exactly the 34 with `converged=False` (all ran to n_iters=1000,
final_cos_sim_mean 0.676–0.731 — the period-2 signature under the lag-1 gate);
every converged prompt locked at iteration 120. Basin shares: ` prolet`×54,
` Divine`×34, ` till`×19, ` Anarch`×17, ` solidarity`×1 — matching the
committed `gated_report.md` shares.

**Divine picks (alphabetically first 5 of the 34):** A08_linguistics,
A14_kant, A15_sartre, A17_marx, A21_dickens.

**20 others:** round-robin across the 7 categories of the RESTORED
`prompt_library.py` (public lucier repo @ main, 125 prompts, categories
Acronyms/Chemical/Complex/Narrative/Simple/Vulgarity/Wild), categories in
alphabetical order, alphabetical by prompt ID within category, excluding the 5
Divine picks. **Recorded consequence of the literal exclusion rule:** only the
5 *picks* are excluded, so 6 of the 20 round-robin prompts are themselves
Stage 1 `Divine` (F01_anger, E02_tech, F02_insult, E03_orgs, B03_moon,
F03_frustration) — 11 of the 25 prompts are Divine in total. This is the
runbook's own rule applied verbatim, not a deviation; it widens the Divine
observation base.

**Execution authority:** the committed
`experiments/exp_010c_windows/output/prompt_subset_small.json` (derived by
`derive_prompts_small.py`, which re-derives from the zip + library at run time
and hard-stops on any mismatch with the committed record — the pythia
pattern).

## 4. Windows, execution order, reproduction gate

Arms (inject i → extract j), all injections at `blocks.i.hook_resid_pre`,
extraction at `blocks.j.hook_resid_post`:

| arm | window | role |
|---|---|---|
| SB | 0→11 | full-stack baseline — **reproduction gate** |
| S1 | 0→5 | 6-layer early |
| S2 | 3→8 | 6-layer mid |
| S3 | 6→11 | 6-layer late |
| S4 | 0→8 | 9-layer front-heavy |
| S5 | 3→11 | 9-layer back-heavy |

Execution order: SB first, alone; the gate is evaluated before any other arm
runs. Then S1, S2, S3, S4, S5, sequentially, commit+push per arm.

**Reproduction gate (pre-registered, mechanical):** on the 25-prompt subset, SB
must reproduce the Stage 1 gated classification per prompt: for each prompt,
(`converged`, `terminal_token`) must equal the `gated_results.pt` record — the
expected pattern is ` prolet`×10, ` till`×3, ` Anarch`×1 converged (all at
lock 120), and the 11 Divine prompts non-converged with terminal decode
` Divine`. **Any mismatch → STOP and document** (RUNBOOK_PHASE1 wording);
no sub-stack arm runs until the gate verdict is recorded.

## 5. Protocol (identical to the registered EXP_010c/EXP_012 path)

- Gated protocol, single source of truth `experiments/atr_engine2.py`
  `run_atr_gated`: threshold 0.999, patience 3, check_every 10,
  check_start 100, `max_iter=1000`, gate_lag 1.
- Seeding: natural L0 prompt pass in every arm — the un-hooked forward pass of
  the prompt seeds the loop with `resid_post(j)`.
- Renorm: **seed_j** (registered Stage 1 convention — loop tensor rescaled to
  the seed's extraction-layer norm) for Stage 1 comparability.
- **Per-arm norm-ratio record:** natural per-layer `resid_pre` norms recorded
  once per prompt (`--record-natural-norms`); each result row carries
  `seed_norm_at_j` and `target_norm`; the analysis reports, per arm, the ratio
  seed_j-target-norm / natural-resid_pre-norm-at-i. **Control B caveat
  (recorded up front):** EXP_010c-VARIANTS found the Medium/Pythia i=0 funnels
  exist only under seed_j (re-injection at ≈60–400× natural layer-0 norm).
  The i=0 arms here (SB, S1, S4) carry the same caveat: observations under
  seed_j at i=0 are convention-dependent until a natural_i variant is run.
  SB is nonetheless the correct gate arm — Stage 1 itself ran seed_j at i=0.
- Seed 42 (`torch.manual_seed`; protocol is deterministic in eval mode — seed
  recorded for completeness, known inert per EXP_010c-3b item 2a).
- Decode: Stage 1's `ln_final → W_U` readout (engine unchanged).
- **Terminal mean vectors saved per (window, prompt)** to the terminals
  archive for the Phase 2 J-lens re-decode (EXP_013).

## 6. Harness (recorded diff), tiers, artifacts

Minimal recorded diff to the shared harness, the `--model-name` pattern from
the EXP_012 pythia work (no protocol change):

- `run_exp010c.py`: `_load_small_from_local` (offline gpt2 load, same
  cache-seeding pattern as medium); `--model-name` gains choice `gpt2`;
  `ARMS` gains SB/S1–S5; tiers `small010b` (25 prompts, registered params,
  arms SB,S1..S5) and `small_smoke` (2 prompts, arms SB+S2, max_iter 60 —
  **harness validation only, non-registered, no verdict weight**); subset
  choice `small` (`derive_prompts_small.select_subset_small`); the small tiers
  are bound to `--model-name gpt2` + `--subset small` and imply
  `--record-natural-norms`.
- `derive_prompts_small.py`: new, derivation per §3.
- `analyze_terminals.py`: `--model-name` gains `gpt2`; small-tier artifacts
  bound to `--model-name gpt2` for `--decode-via-tail`. For SB (j=11) the tail
  is empty and via-tail equals the direct decode by construction; for
  S3/S5 (j=11) likewise. Via-tail is informative for S1 (j=5), S2 (j=8),
  S4 (j=8).

Arms are run one invocation each (`--arms <ARM> --out-suffix small010b_<ARM>`),
committed per arm, then merged into `results_small010b.json` +
`terminals_small010b.pt` (concatenation + dict union, recorded) for the
characterisation step. Artifacts live in
`experiments/exp_010c_windows/output/` (the established shared-harness
location, as EXP_012 did); the results register lives at
`experiments/exp_010b_small/RESULTS_EXP010B.md`.

**Results-location call (recorded):** RUNBOOK_PHASE1 names `RESULTS_PHASE1.md`,
from the pre-pivot phase structure where EXP-D/010a/010b shared one deliverable.
EXP-D was resolved elsewhere (issue #16 body) and EXP_010a's successor ran as
EXP_012 with its own register; this repo's established convention is one
results register per experiment line. A per-experiment
`RESULTS_EXP010B.md` is therefore used, carrying the runbook-required verdict
fields (H5-coarse verdict; reproduction-gate verdict stated first).

## 7. Pre-registered outcome table

| observation | reading |
|---|---|
| SB reproduces Stage 1 per-prompt (gate §4) | gate PASS — weights and fork behaviourally equivalent; proceed |
| SB disagrees on any prompt | gate FAIL — STOP, document, no sub-stack arm runs |
| ≥1 window differs from SB on any §1 field | H5-coarse SUPPORTED (name the fields, per window) |
| all windows reproduce SB's landscape | H5-coarse NOT SUPPORTED |
| Divine prompts under a window: non-converged, lag-1 ≪ 1, lag-2 ≈ 1 | "period-2 survives this cut" (flat) |
| Divine prompts converge under a window | "period-2 broken by this cut" (flat) — candidate field for the richest-window call |

Runtime estimate: 6 arms × 25 prompts; Small ≈3× faster than Medium per pass
and half the layers; 1–3 h total unless arms hit the 1000-iter cap broadly
(the 11 Divine prompts are expected to cap in SB by construction).
