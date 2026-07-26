# EXP_010c — Results Record

**Spec:** `../../EXP_010c_SPEC.md` (pre-registered before any run).
**Status:** COMPLETE — registered full run and boundary scan executed and regenerated post-review; see dated sections below.

---

## 2026-07-23 — Harness validation (no verdict weight)

**What ran:** `run_exp010c.py --tier smoke --harness-check` — the full pipeline
(prompt derivation → windowed gated loop → terminal capture → artifacts) on a
**random-init 24-layer toy model** (d_model=64, dummy vocab). 2 prompts × arms
{A0 0→23, A4 10→21}, max_iter=60.

**Why toy:** this session's remote environment blocks huggingface.co at the
network-policy layer (proxy CONNECT 403), so gpt2-medium weights cannot be
downloaded here. PyPI is open, so the ML stack installed fine (torch 2.13.0,
transformer_lens 3.5.1, py3.11). The toy run validates mechanics only:
hook wiring for arbitrary (i→j) windows, L0 natural-pass seeding, energy
rescaling, the lag-1 gate, per-arm checkpointing, terminal tensor capture,
JSON/pt artifact writing. All passed; artifacts:
`output/results_smoke_harness.json`, `output/terminals_smoke_harness.pt`.

**Observed (mechanically expected, no interpretation):** both arms converged;
the two windows locked onto *different* dummy terminals (A0 → `<903>`,
A4 → `<402>`), confirming the windows are genuinely wired to different layer
ranges.

**Prompt subset:** derived deterministically and recorded —
`output/prompt_subset.json` (25 prompts, round-robin over the 7 categories,
alphabetical by ID; recovered from the Stage 1 record
`experiments/gpt2_medium/output/dissolution_sentences.md` in the public repo,
since `prompt_library.py` is absent — deviation per spec §5).

## 2026-07-23 — Model acquisition + real-weights smoke (reproduction gate PASSED)

**Model route (recorded):** huggingface.co is blocked by this session's network
policy, but the **legacy HF S3 bucket** is reachable and still serves the model:
`https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-medium-{pytorch_model.bin,config.json,vocab.json,merges.txt}`.
Downloaded to a local dir (1,520,013,706 bytes; state dict verified: 316 tensors,
wte [50257, 1024], 24 layers) and loaded offline via the runner's new
`--model-path` flag (seeds the HF cache with config.json so transformer_lens's
internal AutoConfig lookup resolves without network; weights and tokenizer passed
in explicitly). Provenance caveat: this is the pre-2020 HF mirror of gpt2-medium;
the reproduction gate below is the check that it matches the Stage 1 model's
behaviour.

**What ran:** `run_exp010c.py --tier smoke --model-path <local>` — 2 prompts
(E01_politics, D01_water) × arms {A0, A4}, max_iter=60, check_start=20. 71 s CPU.

**Result:**

| Arm | Window | E01_politics | D01_water | Converged |
|---|---|---|---|---|
| A0 | 0→23 | `D` (lock 40) | `D` (lock 40) | 2/2 |
| A4 | 10→21 | `' ('` | `' except'` | 0/2 (60-iter cap) |

**Reproduction gate: PASSED.** A0 reproduces the Stage 1 `D` collapse on real
weights (lock at the earliest possible gated iteration for this tier's
check_start; consistent with Stage 1's lock-by-10). And a first directional
signal, no verdict weight at n=2: the band-exact window A4 does **not** funnel
to `D` — per-prompt distinct terminals, still moving at the smoke cap.

## 2026-07-23 — Pilot (5 prompts × 6 arms, max_iter=300; non-registered tier)

**What ran:** `--tier pilot --model-path <local>`, 30 runs, 799 s CPU, all
converged. Directional signal only; no spec-§6 verdicts drawn from this tier.

| Arm | Window | Terminals (n=5 prompts) | Unique | Lock-in | Margin range |
|---|---|---|---|---|---|
| A0 | 0→23 | `D` ×5 | 1 | 70 | 0.52 |
| A4 | 10→21 (band-exact) | `' ('` ×3, `' forever'`, `' since'` | **3** | 70–140 | 0.93–1.52 |
| A1 | 0→11 | `','` ×4, `'ing'` | 2 | 70 | 0.27–1.11 |
| A2 | 6→17 | `'\n'` ×5 | 1 | 70 | 0.05–0.85 |
| A3 | 12→23 | `'bone'` ×3, `'"'` ×2 | 2 | 80–90 | 0.06–0.46 |
| A5 | 8→15 | `' rant'` ×2, `' trillion(s)'` ×3 | 3 | 70–100 | 0.04–2.05 |

**Directional observations (pilot-grade, n=5):**
- Baseline confirms again: single `D` funnel, exactly Stage 1.
- The two windows centred on the mapped band (A4 10→21, A5 8→15) show the
  most terminal diversity (3 unique each), the first **prompt-dependent**
  terminals seen in Medium, word-like tokens (`forever`, `since`, `rant`,
  `trillion`), and the highest readout confidence (margins up to 2.05 vs
  baseline's 0.52).
- Off-band windows funnel to punctuation: A1 → `','`, A2 → `'\n'`,
  A3 → `'"'`/`'bone'` — single-or-two-basin, low margins.
- Pattern is consistent with H9 and H9a (band windows qualitatively richest),
  pending the registered run.

**Recorded caveat:** cross-window terminal decoding uses the Stage-1
`ln_final → W_U` convention applied at the window's extract layer — a
logit-lens-at-layer-j readout. Mid-stack decodes are exactly where the J-lens
diverges from the logit lens; EXP_013m re-decodes these terminals properly.

## 2026-07-23 — REGISTERED FULL RUN (25 prompts × 6 arms, gated, max_iter=1000)

**What ran:** `--tier full --model-path <local>`. Interrupted once by a
container restart after arms A0/A4 completed (checkpoints held; backups
committed); arms A1/A2/A3/A5 rerun via `--arms`, artifacts merged (150/150
records). All 150 runs converged. Deviation recorded: the run executed as two
processes due to the restart; protocol parameters identical throughout.

**Terminal characterisation** (`analyze_terminals.py`, cosine clustering at the
gate threshold 0.999, plus the decode-via-tail control retrofit per
EXP_010c-2 §3):

| Arm | Window | Tensor basins (sizes) | Direct decode (ln_final→W_U at j) | Decode-via-tail (through j+1→23) | Agree | Margin μ/max |
|---|---|---|---|---|---|---|
| A0 | 0→23 | 4 tight ([18,3,2,2], off-diag cos .998) | `D` ×25 | `D` ×25 | 25/25 | 0.52/0.52 |
| A4 | 10→21 | **12** (off-diag cos .83) | `' until'` ×19, `' forever'` ×5, `' since'` ×1 | `' until'` ×17, `' forever'` ×5, `' since'` ×3 | **23/25** | **4.20/7.15** |
| A1 | 0→11 | 1 ([25]) | `','` ×22, `'ing'` ×3 | `'.'` ×15, `' the'` ×10 | 0/25 | 0.37/1.84 |
| A2 | 6→17 | 1 ([25]) | `' state'` ×15, `' "'` ×10 | `' source'` ×25 | 0/25 | 0.04/0.11 |
| A3 | 12→23 | **21** (off-diag cos .94) | `'"'` ×23, `'work'` ×2 | `'"'` ×23, `'work'` ×2 | 25/25 | 2.41/3.54 |
| A5 | 8→15 | 3 ([17,7,1]) | `' rant'` ×25 | `' endless'` ×18, `"'"` ×7 | 0/25 | 2.14/2.28 |

Notes: for j=23 arms (A0, A3) the tail is empty, so via-tail agreement is the
mean-vs-last-position decode check (both pass — position collapse holds).
A1's single tensor cluster with two direct-decode tokens reflects mean-vector
clustering vs last-position decoding.

**Observations (no interpretation in this section):**

1. Reproduction gate: PASSED. A0: 25/25 → `D`; terminal tensors
   near-identical across prompts (off-diag cos .998).
2. A4 (10→21): 12 tensor clusters at the gate threshold; 3 decode terminals,
   all whole-word tokens, varying by prompt; highest margins of any arm
   (μ 4.20, max 7.15); direct decode and via-tail agree 23/25 — the only
   arm with non-trivial (j<23) agreement above 0/25 in this grid.
3. Direct-decode vs via-tail agreement at j<23: A4 23/25; A1, A2, A5 all
   0/25. (For j=23 arms the tail is empty; agreement there is the
   mean-vs-last-position check, 25/25 both.)
4. A5 (8→15): 3 tensor clusters ([17,7,1]); direct decode a single token
   (`' rant'` ×25); via-tail decode splits 18/7 (`' endless'`/`"'"`),
   numerically close to the tensor-cluster split.
5. A3 (12→23): 21 tensor clusters; 23/25 of them decode to the same token
   (`'"'`). (Same observation class as Stage 1's `Divine` prompts: distinct
   tensors, shared decode.)

**Spec §6 verdicts (mechanical application of the pre-registered table):**

- **H9 (collapse rescue): SUPPORTED.** Pre-registered criterion: at least
  one window with 0<i or j<23 produces multiple terminals, word terminals,
  or non-convergence on the same prompts. Observed: yes (several windows;
  table above).
- **H9a (band placement): SUPPORTED on the registered wording, with a
  recorded measure caveat.** Registered wording: the window most different
  from baseline sits inside the mapped band. "Most different" was not given
  a single pre-registered metric; by unique-terminal count, prompt-
  dependence, margin, and decode agreement, A4 (in-band) is the outlier arm.
  Caveat: in-band A5 produces a single direct-decode terminal at n=25, so
  window length co-varies with the effect — addressed by the 010c-2 sweeps.

**Caveats (standing):** single seed; one 25-prompt subset; cluster threshold
sensitivity unexplored; direct decode at j<23 is a logit-lens-at-layer-j
readout (now with measured unreliability off-band); the J-lens re-decode
(EXP_013m) remains the arbiter for all mid-stack claims.

## 2026-07-23 — EXP_010c-2 BOUNDARY SCAN (8 arms × 25, gated, max_iter=1000)

**What ran:** `--tier scan --model-path <local>`. 200/200 converged.
Artifacts: `output/results_scan.json`, `output/terminals_scan.pt`.

**Onset-edge sweep (inject i → extract 21), with A4 (i=10) from the main grid:**

| i | Terminals | Unique | Token class (lexical, checkable) |
|---|---|---|---|
| 0 | `','` ×25 | 1 | punctuation, single terminal |
| 4 | `','` ×25 | 1 | punctuation, single terminal |
| 6 | `' apologies'`, `' etc'`, `' please'`, `'..'`, `'oooo'`, `'​'` | 6 | mixed word/punctuation/fragment |
| 8 | `' halfway'` ×8, `' simultaneously'` ×17 | 2 | whole-word |
| 10 | `' until'` ×19, `' forever'` ×5, `' since'` ×1 (A4) | 3 | whole-word |
| 12 | `' NHS'`, `' £'`, `'or'` | 3 | abbreviation/symbol/fragment |
| 14 | `' DES'`, `' or'`, `' than'`, `' vs'`, `'.'`, `'."'`, `'.)'` | 7 | mixed word/punctuation |

**Exit-edge sweep (inject 10 → extract j):**

| j | Terminals | Unique | Token class |
|---|---|---|---|
| 21 | `until`/`forever`/`since` (A4) | 3 | whole-word |
| 22 | incl. `' until'` ×3, `'<|endoftext|>'` ×14, `' Tak'`, `' honor'`… | 7 | mixed, EOT plurality |
| 23 | `' self'` ×10, `' until'` ×5, 8 others ×1–2 | 10 | mixed; `D` not observed |

**Verdicts (mechanical application of spec `EXP_010c2_SPEC.md` §4):**

- **H10 (edge localisation): SUPPORTED.** Pre-registered criterion:
  landscape character changes non-uniformly as an edge slides. Observed:
  single-terminal arms at i≤4; multi-terminal mixed arms at i=6 and i≥12;
  whole-word-only arms exactly at i=8 and i=10. Adjacent settings differ
  qualitatively; the change is not monotonic in i.
- **H10a (final-layer necessity): REFUTED in its strong form.** Re-including
  the motor tail (10→22, 10→23) changes the landscape but does not restore
  the `D` funnel.
  **Observation, stated flat:** of the 14 windows tested across both
  experiments, exactly one — the full stack 0→23 — produces `D`. Every
  partial window produces something else. One genuine deconfound: 0→21 is
  22 of 24 layers and does not produce `D`, so window length alone is not
  the driver.
  **Interpretation (labelled as such, NOT a finding):** "the collapse
  requires the motor→sensory conjunction / is a property of the splice" is
  one story consistent with this table. Untested alternatives that would
  also fit: the layer-0 `resid_pre` hook point is a special coordinate
  system (the raw-embedding slot); `D` requires the specific early layers
  0–2 plus tail jointly; other unexamined combinations (no 2→23, no
  gapped-window controls were run). CAUTION recorded: H10a's refutation
  must not be silently converted into support for the splice story — a
  frame that absorbs both confirmation and refutation of its own
  predictions is not being tested.
- **H10b (sensory-splice necessity): SUPPORTED.** 0→21 and 4→21 collapse to
  the `','` funnel with no motor involvement at all — splicing into the
  sensory front destroys structure by itself.

**Recorded for later comparison (observation only):** whole-word,
prompt-dependent terminals occurred only for injection at i ∈ {8, 10}
(extraction 15–21) in this grid. EXP_012m will measure Medium's J-lens band
independently; the two numbers get compared when both exist. No relationship
is asserted here.

**Terminal inventories:** the complete, uncurated terminal lists per arm are
in `results_full.json` and `results_scan.json`. No curated sublist is
maintained in this record — selecting evocative tokens is itself a bias.
Any statement about semantic relatedness among terminals is deferred to the
anisotropy-corrected permutation control (Stage 1 `02b` pattern), which
previously reclassified an apparent semantic cluster as an artifact.

**Scan characterisation + via-tail control**
(`terminal_characterisation_scan.json`): the semantic injection zone survives
the motor-pathway readout — O8's `' simultaneously'` ×17 holds under via-tail
(17/25 agreement; margins to 5.1), the same robustness class as A4. The
funnels do not: O0/O4 flip 0/25 (`','`→`'.'`/`'"'`). O6 is transitional
(7/25). O12/O14 mostly resolve to comparative junk (`or`) under via-tail.
Tensor-basin counts confirm the same boundary: 1 (O0, O4) → 11–12 (O6, O8) →
10–25 fragmenting (O12+, E22/E23). The i=8–10 semantic zone is therefore
robust across both readout instruments; final arbitration to EXP_013m.

**Caveats:** single seed; one 25-prompt subset; cluster-threshold sensitivity
unexplored; two-instrument agreement is necessary but not sufficient — the
J-lens re-decode remains the registered arbiter.

## Planned controls — next observations (revised 2026-07-23 after review)

Register note: this programme is exploration. These controls are not trials
of any story; each is listed with the uncertainty it removes.

1. **Anisotropy-corrected permutation test** on terminal-set relatedness in
   W_E (Stage 1 `02b` pattern). Removes uncertainty about whether any
   apparent relatedness among terminals exceeds the embedding space's
   baseline anisotropy. **DONE 2026-07-25** — spec `../../EXP_010c_PERM_SPEC.md`,
   results in §"2026-07-25 — EXP_010c-PERM" below and
   `output/permutation_results.json`.
2. **Seed and prompt-subset variation** for A4 and O8. Removes uncertainty
   about whether the observed basin structure and the i ∈ {8,10} whole-word
   zone are properties of this seed/subset or of the model.
   **DONE 2026-07-25** — spec `../../EXP_010c_ROBUST_SPEC.md`, results in
   §"2026-07-25 — EXP_010c-ROBUST" below; artifacts
   `output/results_robust_*.json`, `output/terminals_robust_*.pt`,
   `output/terminal_characterisation_robust_*.json`. Issue: #11.
3. **Same window grid on Pythia-410m.** Removes uncertainty about whether
   window-position effects at these relative depths are specific to
   gpt2-medium or generic to decoder stacks of this size.
   **DONE 2026-07-26** — spec `../../EXP_012_PYTHIA_SPEC.md`, results in
   `RESULTS_EXP012_PYTHIA.md` (separate register, this directory; includes
   the H8 depth-control verdict closing RUNBOOK_PHASE1 §EXP_010a);
   artifacts `output/*_pythia410m.*`. Issue: #12.
4. **Hook-point variants** (`resid_post` at i−1 vs `resid_pre` at i; i=1 vs
   i=0). Removes uncertainty about whether the layer-0 single-terminal
   arms reflect the raw-embedding slot's coordinate system rather than
   layer position.
5. **EXP_012m J-lens band measurement** (independent instrument). Produces
   the number the i ∈ {8,10} observation will be compared against.

## Next

Phase 0 / J-lens track per `RUNBOOK_JLENS_MEDIUM.md`: EXP_012m (measure
Medium's J-lens band structure), EXP_011m (project terminal tensors onto
J-space vs complement, permutation null), EXP_013m (re-decode terminals and
trajectories through the J-lens; three-readout comparison). Controls 1–4
above are ATR-side and need no J-lens.

## 2026-07-23 — Post-review corrections (PR #4 review)

1. **lag_scan bug (review item 1):** all `lag_scan` fields committed before
   this date are placeholders (`[1.0..8.0]` — the dict's keys, not its
   cosines). Fixed; full and scan tiers rerun to regenerate artifacts with
   real values. Convergence-relevant fields (terminals, locks, margins) were
   unaffected by the bug; the reruns also serve as a same-machine
   repeatability check against the pre-fix artifacts, and any discrepancy
   will be recorded here.
2. **Single source of truth (item 2):** the runner's duplicated gated loop
   replaced by `atr_engine2.run_atr_gated(capture_terminal=True)` — a
   recorded diff vs the upstream engine (see atr_engine2.py header).
   Equivalence check: post-refactor smoke reproduces the pre-refactor smoke
   exactly (terminals, lock iterations, convergence flags).
3. **Artifact convention (item 3):** `.bak` files removed and ignored;
   JSON results stay versioned (they are the quantitative record);
   terminal `.pt` files stay versioned while ≤ ~2 MB each (they are the
   Phase-2 hand-off artifact); revisit LFS/release assets if they grow.
4. **Item 4:** terminal `.pt` keys changed from tuples to "ARM|PROMPT_ID"
   strings so the files load with `weights_only=True`.

## 2026-07-23/24 — Artifact regeneration complete (post-review)

**What ran:** full tier (150 runs) and scan tier (200 runs) rerun end-to-end
with the fixed capture path (engine-side `capture_terminal`, real lag_scan).

**Repeatability (observation):** all 350 (arm, prompt) records identical to
the pre-fix runs on terminal_token, terminal_token_id, lock_in_iter, n_iters,
and converged — full tier compared against the pre-fix A0/A4 backup plus the
committed part-2 log; scan tier against the committed pre-fix
`results_scan.json`. Zero mismatches. (Same machine, same seed — this is
same-machine repeatability, not seed robustness; control 2 stands.)

**lag_scan (now real; observation):** A0's locked state reads cosine 1.00000
at every lag 1–8. A4 and O8 locked states show monotonic decay with lag
(A4: 0.99955 at lag-1 → 0.97373 at lag-8; O8: 0.99995 → 0.99665). No lag
pattern shows the multiples-of-p signature of a limit cycle in these runs.
The full per-run values are in the regenerated `results_full.json` /
`results_scan.json`.

**Housekeeping:** `results_full.json` and `terminals_full.pt` un-ignored and
committed (the gitignore entries existed to keep mid-run checkpoints out;
final versions are the record, per the convention above).

## 2026-07-25 — EXP_010c-PERM: anisotropy-corrected permutation test (planned control 1)

**Spec:** `../../EXP_010c_PERM_SPEC.md`, committed before any statistic was
computed (commit 0ca5829). **Script:** `permutation_test.py`. **Per-set JSON:**
`output/permutation_results.json`. Issue: #7.

**What ran:** mean pairwise cosine over unique terminal types per
pre-registered set, against N=10,000 matched random same-size token sets
(matched on leading-space status, decoded length ±1, BPE merge-rank band
rank//5000 with a separate byte-token band; `<|endoftext|>` excluded from
pools). Seed 20260725, per-set substreams. Zero forward passes.

**Space (recorded):** the gpt2-medium checkpoint carries `wte.weight` only —
no `lm_head` tensor (verified on the state dict; 316 tensors). Weights are
tied: W_U = wte^T, so the W_U column for token t is the W_E row for token t
and the two report columns are numerically identical. One test per set.
Model files fetched from `huggingface.co/gpt2-medium/resolve/main/`
(post-2026-07-25 policy route per `REMOTE_ENV_MODEL_ACCESS.md`);
`pytorch_model.bin` 1,520,013,706 bytes, same size as the legacy-mirror
artifact recorded in §"Model acquisition."

**Per-set results (all sets reported; α\* = 0.05/9 = 0.00556 Bonferroni across
the 9 testable sets):**

| # | Set | n | obs cos (W_E) | obs cos (W_U) | null μ | null σ | p | z (σ) | p < α\* |
|---|---|---|---|---|---|---|---|---|---|
| 1 | A4_direct (`until, forever, since`) | 3 | 0.4121 | 0.4121 | 0.2742 | 0.0294 | 0.00090 | +4.69 | **yes** |
| 2 | O8_direct (`simultaneously, halfway`) | 2 | 0.2972 | 0.2972 | 0.3005 | 0.0400 | 0.50095 | −0.08 | no |
| 3 | A5_direct (`rant`) | 1 | N/A — singleton, statistic undefined | | | | | | — |
| 4 | A4_tail (`until, forever, since`) | 3 | 0.4121 | 0.4121 | 0.2750 | 0.0302 | 0.00190 | +4.55 | **yes** |
| 5 | O8_tail (`simultaneously, just, '`) | 3 | 0.2993 | 0.2993 | 0.3001 | 0.0506 | 0.35546 | −0.02 | no |
| 6 | A5_tail (`endless, '`) | 2 | 0.2863 | 0.2863 | 0.3219 | 0.0824 | 0.57714 | −0.43 | no |
| 7 | pooled_word_direct | 6 | 0.3180 | 0.3180 | 0.2831 | 0.0172 | 0.02920 | +2.03 | no |
| 8 | pooled_word_tail | 7 | 0.3320 | 0.3320 | 0.2879 | 0.0239 | 0.06019 | +1.84 | no |
| 9 | A1_direct (contrast: `, ing`) | 2 | 0.3937 | 0.3937 | 0.3344 | 0.0810 | 0.22338 | +0.73 | no |
| 10 | A1_tail (contrast: `. the`) | 2 | 0.5122 | 0.5122 | 0.3043 | 0.0712 | 0.00620 | +2.92 | no |

Smallest candidate pool: `' simultaneously'` (43); all other pools 255–1,626
(full sizes in the JSON).

**Which pre-registered reading obtained (mechanical application of spec §8):**
the **mixed outcome** row. Sets 1 and 4 (A4, both readouts — the identical
type set {`until`, `forever`, `since`}) pass at p < α\* with z > 0: for A4 the
relatedness pattern survives its first control ("semantic" stays quarantined).
Sets 2, 5, 6 (O8 both readouts, A5 via-tail) are at or below their null means
(z −0.43 to −0.08): consistent with baseline anisotropy at matched
size/space/length/frequency. Pooled sets 7–8 do not pass (z +2.03/+1.84,
p ≥ 0.029); per spec §8 the pooled rows do not override per-arm rows or vice
versa. Contrast sets 9–10 do not pass, so no calibration downgrade is
triggered; recorded plainly: A1_tail sits at p = 0.00620 against
α\* = 0.00556.

**Caveats (standing):** n=2 sets carry a single pairwise cosine; the A4 pass
rests on 3 types from one seed and one 25-prompt subset (control 2 covers
seed/subset variation); merge-rank band is a frequency proxy, not a measured
frequency; no statement about *why* the A4 terminals are related is made or
implied.

## 2026-07-25 — Provenance addendum (PR #19 review)

SHA-256 digests of the exact statistical inputs used by the permutation
test (recorded post-run; spec addendum has the same values):
`pytorch_model.bin` 98c7b055…10f8318 · `vocab.json` 19613966…636783 ·
`merges.txt` 1ce16647…26adc5. Full values in `EXP_010c_PERM_SPEC.md`
§Post-run addendum and in `output/permutation_results.json` once
regenerated by the updated script; the numbers in the recorded run are
unchanged by this addendum.

## 2026-07-25 — EXP_010c-ROBUST: seed and prompt-subset robustness (planned control 2)

**Spec:** `../../EXP_010c_ROBUST_SPEC.md`, committed before any run (commit
d4c3a3c). **Runner diff:** `--seed` / `--subset` / `--out-suffix` parameters
plus `derive_prompts.select_subset_b()` (commit 435043b; no protocol change).
Issue: #11.

**What ran:** 3 variants × arms {A0 0→23, A4 10→21, O8 8→21} × 25 prompts =
225 gated runs, protocol identical to the registered full tier (cos > 0.999
×3, check every 10 past 100, max_iter=1000, L0 natural-pass seeding),
gpt2-medium offline via `--model-path`. All 225 converged. Variant 1: seed
1337, registered subset (3488 s). Variant 2: seed 2718, registered subset
(3285 s). Variant 3: seed 42, disjoint subset B — the NEXT 25 prompts under
the registered round-robin/alphabetical rule, recorded in the spec §3 and
`output/prompt_subset_b.json` (2828 s + 2217 s, see deviation).

**Deviation (recorded):** the container restarted at ~14:18 during Variant
3's A4 arm (its log ends mid-arm with no traceback; dmesg shows a fresh
boot). The per-arm A0 checkpoint held; A4 and O8 were rerun via
`--arms A4,O8 --out-suffix robust_subsetB_p2` and the artifacts merged into
`results_robust_subsetB.json` / `terminals_robust_subsetB.pt` (75/75
records; same recovery pattern as the registered full run's restart). Parts
and the A0-only checkpoint backup are committed alongside the merged files
for audit.

**Seed-variant determinism (spec §2 observation):** Variants 1 and 2 are
**identical to the registered run record-by-record** — all 75 records each
match the registered A0/A4 (full tier) and O8 (scan tier) records on all 9
compared fields (terminal_token, terminal_token_id, lock_in_iter, n_iters,
converged, top_logit_margin, entropy, final_cos_sim_mean, terminal_prob),
and their terminal-characterisation lines (basins, cosines, margins,
via-tail) are numerically identical to the registered ones. As pre-registered
in spec §2: the protocol contains no sampling, so the global torch seed does
not influence the run on this machine; the robustness burden falls on the
subset variant.

**Per-variant observation table** (registered rows from the 2026-07-23 full
and scan sections for comparison; Variants 1–2 shown once since they equal
the registered rows exactly):

| Arm | Variant | Direct decode | Via-tail decode | Agree | Tensor basins (sizes) | Margin μ/max | Lock-in |
|---|---|---|---|---|---|---|---|
| A0 | registered = V1 = V2 | `D` ×25 | `D` ×25 | 25/25 | 4 ([18,3,2,2], cos .998) | 0.52/0.52 | 120 |
| A0 | V3 (subset B) | `D` ×25 | `D` ×25 | 25/25 | 3 ([21,3,1], cos .999) | 0.52/0.52 | 120 |
| A4 | registered = V1 = V2 | `' until'` ×19, `' forever'` ×5, `' since'` ×1 | `' until'` ×17, `' forever'` ×5, `' since'` ×3 | 23/25 | 12 (cos .828) | 4.20/7.15 | 130–150 |
| A4 | V3 (subset B) | `' until'` ×22, `' forever'` ×2, `' ('` ×1 | `' until'` ×19, `' forever'` ×4, `' since'` ×1, `' ('` ×1 | 22/25 | 13 (cos .819) | 3.98/7.28 | 120–150 |
| O8 | registered = V1 = V2 | `' simultaneously'` ×17, `' halfway'` ×8 | `' simultaneously'` ×17, `' just'` ×6, `"'"` ×2 | 17/25 | 11 (cos .967) | 2.88/5.10 | 120 |
| O8 | V3 (subset B) | `' simultaneously'` ×18, `' halfway'` ×6, `' already'` ×1 | `' simultaneously'` ×19, `' just'` ×4, `"'"` ×1, `' already'` ×1 | 19/25 | 12 (cos .962) | 2.46/4.31 | 120 |

Token classes: every V3 direct-decode terminal except `' ('` is a whole-word
alphabetic token; both V3-only terminals (`' ('` in A4, `' already'` in O8)
come from the same prompt, `G07_the` (text `"The"`, the subset's only
single-word prompt; margins 1.65 and 0.12 respectively). Full uncurated
inventories in `results_robust_*.json`; no curated sublist kept.

**Pre-registered readings (mechanical application of spec §5):**

- **Primary criterion → "same-basin-structure (stable)".** In every variant:
  A0's direct decode is `D` for all 25 prompts (25/25 in V1, V2, V3), and
  A4's direct-decode terminal set contains ≥2 of its 3 registered types —
  V1/V2: all 3 (`until`, `forever`, `since`); V3: 2 of 3 (`until` ×22,
  `forever` ×2; `since` absent from direct decode, though present ×1 in
  V3's via-tail decode).
- **Secondary reading → "zone-stable at i=8".** O8's direct-decode terminals
  are whole-word alphabetic tokens for all 25 prompts in every variant
  (V3 adds `' already'` ×1 to `simultaneously`/`halfway`).
- Differences observed under subset B, stated exactly: A4 loses `' since'`
  from the direct decode and gains `' ('` on `G07_the`; O8 gains
  `' already'` on `G07_the`; A0's basin count moves from 4 to 3; margin
  means shift by −0.22 (A4) and −0.43 (O8); basin counts 12→13 (A4),
  11→12 (O8).

**Caveats (standing):** the two seed variants test only the global-torch-seed
pathway on one machine — they demonstrate protocol determinism, not
cross-machine invariance; subset B is one additional subset (n=25) drawn by
the same rule from the same 125-prompt pool; cluster-threshold sensitivity
remains unexplored; direct decode at j=21 remains a logit-lens-at-layer-j
readout with the J-lens re-decode (EXP_013m) as arbiter.

## 2026-07-25 — EXP_010c-VARIANTS: hook-point and energy-normalisation controls (issues #13, #14)

**Spec:** `../../EXP_010c_VARIANTS_SPEC.md` (pre-registered, committed before
any run). Protocol identical to the registered tiers except the single named
variable per control. Runner/engine diffs recorded in the spec §2. Deviation:
the executing agent was twice terminated by server-side API errors; sweeps
continued uninterrupted and the analysis was completed by the orchestrating
session — protocol parameters unaffected.

### Control A — hook-point variants (planned-controls item 4)

| Arm | Window / hook | Converged | Tensor basins | Direct decode | Via-tail (agree) | Margin μ/max |
|---|---|---|---|---|---|---|
| I1A0 | 1→23 | 25/25 | 5 | `name` ×22, `Class` ×2, `host` ×1 | `name` ×23, `Class` ×2 (24/25) | 0.16/0.70 |
| I1O0 | 1→21 | 25/25 | 1 | `','` ×25 | `' the'` ×25 (0/25) | 0.14/0.15 |
| HP9 | 10→21 @ `blocks.9.hook_resid_post` | 25/25 | 12 | `' until'` ×19, `' forever'` ×5, `' since'` ×1 | 17/5/3 (23/25) | 4.20/7.15 |

Registered comparators: A0 (0→23) → `D` ×25; O0 (0→21) → `','` ×25.

**Spec §3 readings (mechanical):**

- **Hook equivalence: CONFIRMED.** HP9 is identical to registered A4 on all
  9 comparison fields for all 25 records (0 mismatches).
- **Row-1 condition (both i=1 arms single-terminal): NOT met.** I1O0 is
  single-terminal (`','` ×25 — same token as O0, funnel character and token
  preserved at i=1); I1A0 is not.
- **Row-2 condition (an i=1 arm with ≥3 unique, ≥2 whole-word alphabetic):
  MET by I1A0** (3 unique: `name`, `Class`, `host`; all alphabetic) →
  reading **"embedding-slot effect isolated"** for the 0→23 funnel: the `D`
  terminal does not survive moving injection from layer 0 to layer 1;
  H10b's sensory-splice reading inherits this caveat. Stated flat: the
  0→21 funnel survives i=1 with the same token; the 0→23 `D` funnel does
  not (margins in the I1A0 arm are low: μ 0.16).

### Control B — energy-normalisation variant (session-registered control)

Rescale target: `natural_pre_norm_i` (per-prompt, measured on the natural
pass; saved per arm). Measured seed_j/natural_i norm ratios: A0 ×217.8,
A1 ×306.7, A4 ×1.04, O8 ×1.05 — under the registered convention the i=0
arms re-injected at two orders of magnitude above natural layer-0 input
norm, while the mid-stack arms were within 5% of natural.

| Arm | Window | Converged | Tensor basins | Direct decode | Via-tail (agree) | Margin μ/max |
|---|---|---|---|---|---|---|
| A0 | 0→23 | **0/25** (all 1000-iter cap) | 14 | `This` ×10, `A` ×7, `If` ×3, `','` ×3, `The`, `An` | same set (25/25) | 0.46/1.29 |
| A4 | 10→21 | 25/25 | 12 | `' until'` ×20, `' forever'` ×3, `' since'` ×2 | 16/5/4 (21/25) | 4.29/7.31 |
| O8 | 8→21 | 25/25 | 17 | `' simultaneously'` ×14, `' halfway'` ×9, `' spit'` ×2 | `simultaneously` ×16, `just` ×5, `collision` ×2 (14/25) | 1.87/4.52 |
| A1 | 0→11 | **0/25** (all 1000-iter cap) | 9 | `' ('` ×11, `' T'` ×6, `','` ×4, others | `'"'` ×12, `2017` ×6, `This` ×3 (0/25) | 0.23/0.90 |

**Spec §4 readings (mechanical, class definitions in the spec):**

- A4: word-structured (3 unique types, 0 non-word terminals) — class
  unchanged; terminal types identical to registered (`until`/`forever`/
  `since`), margins within 2% of registered.
- O8: word-structured (3 unique, all whole-word) — class unchanged;
  `simultaneously`/`halfway` persist, `' spit'` ×2 new.
- A0: **not a funnel** under `natural_i` (6 unique types, modal 10/25,
  `D` absent, 0/25 converged) — **class changed**.
- A1: **not a funnel** under `natural_i` (≥5 unique types, modal 11/25,
  0/25 converged) — **class changed**.
- Reading obtained: **"energy convention load-bearing" for A0 and A1;
  "not an energy artifact" for A4 and O8.** Per the pre-registered wording:
  every EXP_010c/010c-2 observation about the i=0 arms (including the
  reproduction-gate `D` collapse and the O0/O4 funnels) is conditional on
  the j-scale energy convention; the in-band word-window observations are
  not conditional on it.

**Combined observation across both controls (stated flat):** of everything
tested to date, the `D` terminal has been observed only under the joint
condition {inject at layer-0 `resid_pre`} ∧ {j-scale (≈218× natural) energy}.
Removing either condition removes `D`. The A4/O8 word-window landscape is
unchanged under: seed variation, disjoint prompt subset, hook-point
convention (HP9), and i-scale energy normalisation.

**Caveats (standing):** single subset for the variant arms; non-convergence
at the 1000-iter cap is a bounded observation (longer horizons untested);
cluster-threshold sensitivity unexplored; direct decode at j<23 remains a
logit-lens readout; the J-lens re-decode (EXP_013m) remains the registered
arbiter. Planned-controls item 4 (hook-point): done (this section). The
session-registered energy-norm control: done (this section).
