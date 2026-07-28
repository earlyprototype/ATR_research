# EXP_012-PYTHIA — Results Record

**Spec:** `../../EXP_012_PYTHIA_SPEC.md` (pre-registered before any run).
**Issue:** #12. Closes RESULTS_EXP010C.md §"Planned controls" item 3 and the
Stage 1 depth control (H8 / EXP_010a, `RUNBOOK_PHASE1.md`).
**Status:** COMPLETE — smoke gate, registered sweep (150/150 records), and
terminal characterisation executed; dated sections below.
**Register:** observations only. The cross-model rows quote the Medium
registers verbatim for comparison-readiness; no cross-model interpretation
is drawn here.

---

## 2026-07-25 — Model acquisition + smoke gate (PASSED)

**Route:** huggingface.co directly (allowed since the 2026-07-25 policy
update, `REMOTE_ENV_MODEL_ACCESS.md`). Files, sizes, sha256 (provenance
convention per PR #19 review): see spec §2 — config.json 570 B
`d4c11e84…ca4fc6`, tokenizer.json 2,113,710 B `c24618a1…f25624`,
pytorch_model.bin 911,449,213 B `786bd9bc…5e1543`. State dict verified: 364
tensors, embed_in [50304, 1024], 24 layers. Stack: torch 2.13.0+cpu,
transformer_lens 3.5.1, transformers 5.14.1.

**Loader (recorded diff):** `run_exp010c.py --model-name pythia-410m
--model-path <dir>` — GPTNeoXForCausalLM + AutoTokenizer from the local dir,
HF cache seeded with config.json, passed into
`HookedTransformer.from_pretrained("pythia-410m", hf_model=…, tokenizer=…)`.
One compatibility shim, recorded in-code: transformers 5.x renamed GPTNeoX's
`embed_out` → `lm_head`; transformer_lens 3.5.1's `convert_neox_weights`
reads the old name, so the loader aliases the same module object under
`embed_out`. No weights copied or altered.

**Smoke (non-registered, no verdict weight):** 2 prompts × {A0 0→23,
A4 10→21}, max_iter=60, 74 s. GPTNeoX `blocks.{i}.hook_resid_pre` /
`hook_resid_post` resolve; A0 hit the smoke cap unconverged (`'\n'`,
`' ."'`), A4 locked at 50 (`'bound'`, `'iversary'`). Artifacts:
`output/results_pythia_smoke.json`, `output/terminals_pythia_smoke.pt`.

## 2026-07-25/26 — REGISTERED SWEEP (6 arms × 25 prompts, gated, max_iter=1000)

**What ran:** `run_exp010c.py --tier pythia --model-name pythia-410m
--model-path <local> --subset pythia --seed 42 --out-suffix pythia410m`.
Single process, arms sequential in spec order (P-A0, P-A1, P-A2, P-A3,
P-A4, P-O8), per-arm checkpoints committed. 150/150 records. Registered
protocol: cos > 0.999 ×3, checks every 10 past 100, max_iter=1000,
gate_lag=1, seed 42, L0 natural-pass seeding, renorm=`seed_j`. Prompts: the
committed `output/prompt_subset_pythia.json` (execution authority; the
runner re-derives and hard-stops on mismatch — spec §3).

**Total runtime 15,918 s (~4.4 h).** Per-arm (from launch 23:13:30 UTC and
checkpoint-commit times, ≤9 min lag): P-A0 ≈ 105 min, P-A1 ≈ 10 min,
P-A2 ≈ 19 min, P-A3 ≈ 90 min, P-A4 ≈ 17 min, P-O8 ≈ 19 min. The two
cap-hitting arms (P-A0, P-A3) dominate.

**Observation table** (`analyze_terminals.py --tier pythia410m
--decode-via-tail --model-name pythia-410m`, cosine clustering at the gate
threshold 0.999; direct decode = `ln_final → W_U` at j):

| Arm | Window | Conv | Lock range | Tensor basins (sizes) | Direct decode | Decode-via-tail (j+1→23) | Agree | Margin μ/max |
|---|---|---|---|---|---|---|---|---|
| P-A0 | 0→23 | 13/25 | 120–910 | 15 ([9,3,1×13], off-diag cos .486) | `'/'` ×14, `'\n'` ×2, 9 singletons | `'/'` ×13, `'\n'` ×3, `' and'`/`' AND'` ×4, 5 others | 19/25 | 0.49/1.40 |
| P-A1 | 0→11 | **25/25** | 120 (all) | **1** ([25], off-diag cos 1.000) | `' bast'` ×25 | `'1'` ×25 | 0/25 | 0.02/0.02 |
| P-A2 | 6→17 | 25/25 | 120 (all) | 3 ([23,1,1]) | `' divide'` ×20, 5 singletons | `'\n'` ×24, `'KK'` ×1 | 0/25 | 0.78/2.81 |
| P-A3 | 12→23 | 17/25 | 120–970 | **25** (all private, off-diag cos .822) | `'\n'` ×21, 4 singletons | `'\n'` ×21, same 4 singletons | 25/25 | 1.15/2.25 |
| P-A4 | 10→21 | 25/25 | 120 (all) | 5 ([8,6,6,3,2]) | `' untimely'` ×12, `'bound'` ×8, `' alive'` ×5 | `' alive'` ×18, `' untimely'` ×4, `'bound'` ×3 | 12/25 | 0.26/0.69 |
| P-O8 | 8→21 | 25/25 | 120 (all) | 4 ([8,8,5,4]) | `'iversary'` ×25 | `'iversary'` ×25 | 25/25 | 0.99/1.46 |

For j=23 arms (P-A0, P-A3) the tail is empty; "via-tail" is the
mean-vs-last-position decode check, as in the Medium register.

**Non-converged runs (lag_scan, lags 1–8):** P-A0's 12 non-converged are
mixed — 4 near-threshold slow drifters (lag-1 cos .98–.995) and 8 genuine
drifters (lag-1 cos .59–.94, no lag scoring ~1.0, so no small-period limit
cycle is hiding from the lag-1 gate; E01_politics bottoms at .51). P-A3's 8
non-converged are all near-threshold slow drifters (lag-1 cos .986–.997).

**Norm ratios (spec §4 record: seed_j target norm / natural `resid_pre`
norm at injection layer i, per arm over 25 prompts):**

| Arm | i | Mean ratio | Min | Max |
|---|---|---|---|---|
| P-A0 | 0 | **63.1** | 47.4 | 86.6 |
| P-A1 | 0 | **402.6** | 313.7 | 505.0 |
| P-A2 | 6 | 1.99 | 1.97 | 2.02 |
| P-A3 | 12 | 0.16 | 0.12 | 0.26 |
| P-A4 | 10 | 0.87 | 0.86 | 0.88 |
| P-O8 | 8 | 0.93 | 0.92 | 0.94 |

Same shape as Medium's Control B measurement: i=0 arms run far above
natural input norm under `seed_j` (and the two H8 arms differ from each
other, 63× vs 403×, because the extraction-layer norms differ); mid-band
arms run near natural (0.87–1.99×); P-A3 runs *below* natural. Per spec §4,
every i=0 observation below carries the energy-convention caveat; the
`natural_i` mirror of this grid is recorded follow-on, not run here.

## H8 verdict (mechanical application of RUNBOOK_PHASE1 §EXP_010a / spec §5)

Comparison on the four pre-registered axes:

| Axis | P-A0 (0→23, control) | P-A1 (0→11, treatment) |
|---|---|---|
| Converged fraction | 13/25 | **25/25** |
| Unique terminals | 11 | **1** (`' bast'`) |
| Cross-prompt mean-vector cos (off-diag mean) | 0.486 | **1.000** |
| Basin consolidation | 15 basins, largest 9 | **1 shared basin (25/25)** |

Treatment consolidates AND converges where control fragments, on every
axis, at the earliest possible gated lock (120) for every prompt.

**H8: SUPPORTED — depth is causal for Pythia-410m's fragmentation (Arm
0→11 consolidates and converges where 0→23 fragments), under the
registered `seed_j` energy convention.** Caveat carried from spec §4,
stated flat: both H8 arms inject at i=0 far above natural norm and at
*different* multiples (63× vs 403×), so under this convention window depth
and injection loudness co-vary; the pre-registered criterion is applied as
written, and the `natural_i` mirror is the recorded follow-on that would
separate them. Baseline note: P-A0 partially converges here (13/25)
where Stage 1's fixed-horizon full-stack run reported 9/125 and no
consolidation — different protocol (gated, 1000 iters, this 25-prompt
subset), recorded, not baseline-anomalous in the spec §2 sense (no
consolidated single-token collapse appeared).

## Comparison-ready map vs the Medium grid (observations only)

Medium rows quote the registered EXP_010c full run (seed_j, this file's
sibling register); Pythia rows are the table above. No interpretation.

| Window | Medium (gpt2-medium) | Pythia-410m |
|---|---|---|
| 0→23 | `D` ×25, converged, 4 tight basins | `'/'` ×14 + 10 others, 13/25 converged, 15 basins |
| 0→11 | `','` ×22 funnel, 1 basin, converged | `' bast'` ×25, 1 basin, converged, margin 0.02 |
| 6→17 | `' state'`/`' "'`, 1 basin, margin 0.04 | `' divide'` ×20, 3 basins ([23,1,1]), margin 0.78 |
| 12→23 | `'"'` ×23, 21 basins | `'\n'` ×21, 25 basins (all private) |
| 10→21 | `' until'`/`' forever'`/`' since'` (3 whole words), 12 basins, margin 4.20 | `' untimely'`/`'bound'`/`' alive'` (3 word-like), 5 basins, margin 0.26 |
| 8→21 | `' simultaneously'` ×17/`' halfway'` ×8, 11 basins | `'iversary'` ×25 (subword), 4 basins |

Flat observations, per the spec §6 axes:

1. Window-position effects are PRESENT on the placebo: the six windows do
   not behave alike (converged fraction 13/25–25/25; basin count 1–25;
   single-terminal vs 11-terminal decodes; margins 0.02–1.15 μ).
2. The specific in-band features of the Medium map replicate only
   partially: 10→21 again yields exactly 3 word-like alphabetic terminals
   with prompt-dependence (feature coincidence), but 8→21 yields a single
   shared subword (`'iversary'`) where Medium had 2 whole words, and the
   highest margins sit off-band (P-A3) rather than in-band.
3. No `D`-class collapse anywhere: Pythia's full stack does not produce a
   single-token converged funnel; its 0→23 is the *most* fragmented decode
   (11 unique terminals) rather than the most consolidated.
4. Direct/via-tail agreement splits by window as on Medium: j=23 arms and
   8→21 agree fully; 0→11 and 6→17 disagree completely (0/25); 10→21
   partial (12/25).

## Deviations (recorded)

1. **Sixth window:** issue #12's body lists 8→15 (A5); the executing task
   directive substituted O8 8→21. Pre-registered in spec §3 before any run;
   A5 on Pythia is follow-on.
2. **Mid-sweep harness fixes (PR #39 review, two rounds):** validation and
   CLI-binding fixes were committed while the sweep ran (commits bf1adc7
   and this section's). The running process used the launch-time code
   (62ecff3) throughout; none of the fixes touch the protocol path, and
   the executed subset was verified ID-identical before and after.
3. **Commit-subject miscounts, corrected here:** P-A0's checkpoint commit
   subject said 15/25 converged (actual 13/25); P-A3's said 18/25 (actual
   17/25). The committed artifacts were correct in both cases; the counts
   in this register are computed from the artifacts.
4. **Natural-norm record:** produced for this seed_j run via the tier
   binding (spec §4 promise), not by a `--renorm natural_i` run.

## Artifacts

`output/results_pythia410m.json` (150 records),
`output/terminals_pythia410m.pt`, `output/prompt_subset_pythia.json`,
`output/natural_resid_norms_pythia410m.json`,
`output/terminal_characterisation_pythia410m.json`,
`output/results_pythia_smoke.json`, `output/terminals_pythia_smoke.pt`.
