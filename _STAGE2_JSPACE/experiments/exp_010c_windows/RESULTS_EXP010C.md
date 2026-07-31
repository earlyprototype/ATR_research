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

## 2026-07-24 — EXP_010c-3 IN-FILL SWEEP (9 arms × 25, gated, max_iter=1000)

**Spec:** `../../EXP_010c3_SPEC.md` (pre-registered before any run; committed first).

**What ran:** `run_exp010c.py --tier infill --model-path <local>` (executed via a
resume-capable driver that reuses the runner's `run_arm_with_terminal` →
`atr_engine2.run_atr_gated` path unchanged, so the protocol is identical to the
plain `--tier infill`; the driver adds per-arm commit and restart-resume).
9 new windows × 25 prompts = **225 runs, 225/225 converged.** Same 25-prompt
subset as the registered runs (`output/prompt_subset.json`; the derivation was
re-verified this session to reproduce the committed subset exactly). Model: the
S3-mirror gpt2-medium (`pytorch_model.bin` 1,520,013,706 bytes — the same file
recorded in §Model acquisition), loaded offline; torch 2.13.0 /
transformer_lens 3.5.1 (same versions as the registered runs). Deviation
recorded: the sweep was interrupted by a container pause partway through arm
X1017 (no partial written — checkpoints are per-completed-arm) and resumed; the
first seven arms and the last two ran as two processes, protocol identical
throughout. Artifacts: `output/results_infill.json`, `output/terminals_infill.pt`,
`output/terminal_characterisation_infill.json`.

**Recorded count deviation:** the issue header says "10 new windows × 25 prompts
= 250 runs" but its window table enumerates 9 windows (4 injection + 5
extraction); this sweep ran the 9 enumerated, justified cells (spec §3 records
why no 10th was added).

**Injection in-fill (inject i → extract 21).** Token class is the checkable
lexical rule (whole-word = leading-space alphabetic core len≥2; punctuation = no
alphanumeric char; fragment = otherwise):

| Arm | i | Terminals | Unique | Dominant class | Class counts | Conv | Lock-ins |
|---|---|---|---|---|---|---|---|
| I5 | 5 | `'...'`×7, `'…'`×6, `' Congratulations'`×4, `' Welcome'`×3, `'been'`×2, `'!!'`×2 +1 | 7 | punctuation | punctuation:15, whole-word:8, fragment:2 | 25/25 | 120/150/170 |
| I7 | 7 | `'oooooooo'`×25 | 1 | fragment | fragment:25 | 25/25 | 120 |
| I9 | 9 | `'oooooooo'`×18, `'…'`×7 | 2 | fragment | fragment:18, punctuation:7 | 25/25 | 120/130 |
| I11 | 11 | `'<\|endoftext\|>'`×25 | 1 | (EOT special token) | fragment:25 | 25/25 | 120 |

Context rows from the boundary scan (extract 21), surrounding even injection
layers: i=6 → 6 unique, mixed (plurality `'​'`); **i=8 → `' simultaneously'`×17,
`' halfway'`×8 (whole-word);** i=10 (A4) → `' until'`×19, `' forever'`×5,
`' since'`×1 (whole-word); i=12 → `'or'`/`' £'`/`' NHS'` (mixed); i=14 → 7 unique,
plurality `' or'`×15.

**Extraction ladder (inject 8 or 10 → extract j):**

| Arm | Window | Terminals | Unique | Dominant class | Conv | Lock-ins |
|---|---|---|---|---|---|---|
| X817 | 8→17 | `' GOP'`×25 | 1 | whole-word | 25/25 | 120 |
| X819 | 8→19 | `"'d"`×25 | 1 | fragment | 25/25 | 120 |
| X1015 | 10→15 | `' Fas'`×25 | 1 | whole-word | 25/25 | 120 |
| X1017 | 10→17 | `' Bhar'`×25 | 1 | whole-word | 25/25 | 120 |
| X1019 | 10→19 | `'…)'`×15, `' […]'`×10 | 2 | punctuation | 25/25 | 120 |

Context: existing extraction endpoints — inject 8 → j=15 (A5) `' rant'`×25, j=21
(O8) `' simultaneously'`/`' halfway'`; inject 10 → j=21 (A4)
`' until'`/`' forever'`/`' since'`.

**Terminal characterisation (tensor basins + decode-via-tail control;** cosine
clustering at gate threshold 0.999; all 9 windows have j<23 so the via-tail is
non-empty):

| Arm | Window | Tensor basins | Direct decode | Via-tail | Agree | Margin μ/max |
|---|---|---|---|---|---|---|
| I5 | 5→21 | 21 | `'...'`×7, `'…'`×6, `' Congratulations'`×4, `' Welcome'`×3 | `'!'`×7, `'…'`×6, `' once'`×4, `' been'`×4 | 6/25 | 0.418/2.06 |
| I7 | 7→21 | 6 | `'oooooooo'`×25 | `'oooooooo'`×25 | 25/25 | 2.402/2.572 |
| I9 | 9→21 | 8 | `'oooooooo'`×18, `'…'`×7 | `'…'`×25 | 7/25 | 1.693/2.305 |
| I11 | 11→21 | 3 | `'<\|endoftext\|>'`×25 | `'<\|endoftext\|>'`×25 | 25/25 | 1.836/2.172 |
| X817 | 8→17 | 2 | `' GOP'`×25 | `' since'`×25 | 0/25 | 1.153/1.451 |
| X819 | 8→19 | 8 | `"'d"`×25 | `"'d"`×15, `'…'`×10 | 15/25 | 2.568/3.646 |
| X1015 | 10→15 | 3 | `' Fas'`×25 | `' the'`×25 | 0/25 | 1.811/2.261 |
| X1017 | 10→17 | 5 | `' Bhar'`×25 | `' Indian'`×25 | 0/25 | 1.308/1.474 |
| X1019 | 10→19 | 16 | `'…)'`×15, `' […]'`×10 | `','`×13, `' )'`×5, `' ,'`×4 | 1/25 | 1.013/2.054 |

**Observations (no interpretation in this section):**

1. All 225 runs converged (lock-in 120–170; most arms lock at exactly 120,
   the earliest gated iteration for check_start=100).
2. Of the 9 new cells, **6 direct-decode to a single terminal repeated across
   all 25 prompts** (I7 `oooooooo`, I11 `<\|endoftext\|>`, X817 `GOP`,
   X819 `'d`, X1015 `Fas`, X1017 `Bhar`). The other three are multi-token:
   I9 (2 unique, `oooooooo` ×18 / `…` ×7), X1019 (2 unique, punctuation), and
   I5 (7 unique, plurality punctuation). None has a whole-word plurality with
   ≥2 unique whole-word terminals.
   *(Corrected 2026-07-25 after the PR #33 review: this read "7 … and I9 is
   18/7 between two tokens", which counted a two-terminal cell inside a
   strict-single-terminal total. The strict count is 6; I9 is stated
   separately above. No verdict depended on the figure.)*
3. Mechanical whole-word flag (≥2 unique terminals AND plurality class
   whole-word) across all 9 new cells: **0 flagged.**
4. I9 (9→21), the injection midpoint between the two extract-21 whole-word
   cells i=8 and i=10: plurality `oooooooo` (fragment) ×18, `…` ×7; whole-word
   count 0/25.
5. Injection flanks at extract 21: I5 (i=5) plurality punctuation; I7 (i=7)
   single fragment `oooooooo` ×25; I11 (i=11) single `<|endoftext|>` ×25.
6. Extraction ladder below 21: at inject 8, j=17 `GOP` and j=19 `'d`; at
   inject 10, j=15 `Fas`, j=17 `Bhar`, j=19 `…)`/`[…]`. Every rung is a single
   direct terminal (or, at j=19-10, two punctuation tokens).
7. Decode-via-tail agreement at these cells: I7 25/25 and I11 25/25 (a single
   prompt-independent token surviving the tail); X819 15/25; I9 7/25; I5 6/25;
   X1019 1/25; **X817, X1015, X1017 all 0/25 — the direct-decode terminal
   inverts entirely under the tail** (`GOP`→`since`, `Fas`→`the`, `Bhar`→`Indian`).

**Spec §4 verdicts (mechanical application of the pre-registered table):**

- **H12 (injection-zone continuity): REFUTED.** Pre-registered: 9→21 funnelling
  to punctuation/fragments or a single shared terminal refutes continuity.
  Observed: I9 plurality `oooooooo` (fragment), 0/25 whole-word. The extract-21
  whole-word cells at i=8 and i=10 are **separate one-layer islands**, not a
  contiguous band; the layer between them is fragment/punctuation.
- **H12a (injection-zone edges): SUPPORTED on the onset side, with a recorded
  sharpening.** Pre-registered: 5→21 and 7→21 resembling the i≤6
  punctuation/fragment character puts the word-zone onset at i=8, sharp.
  Observed: I5 punctuation-dominant, I7 single fragment — both non-whole-word,
  so onset is at i=8. The specific sub-prediction "7→21 whole-word but 5→21 not"
  is **not** borne out (neither is whole-word); the onset is one layer sharper
  than that reading allowed. On the exit side, I11 (i=11) is a single EOT
  terminal (non-whole-word), so the extract-21 whole-word set is exactly
  i ∈ {8, 10} with i=9 excluded.
- **H13 (extraction ladder): SUPPORTED; edge at j=21, sharp.** Pre-registered:
  whole-word + via-tail-robust at the top of the ladder, lost at some locatable
  j\*. Observed: the whole-word, via-tail-robust character holds only at the
  seed extract layer j=21 (O8 17/25, A4 23/25); every rung below (j ∈ {15,17,19}
  at inject 8 and 10) is a single direct terminal, and via-tail agreement
  collapses (0/25 at X817, X1015, X1017; 1/25 at X1019; 15/25 at X819). The
  robust character does not extend even one layer below the seed extract point.
- **Non-convergence:** none (225/225).
- **Via-tail inversion flag (readout-load-bearing cells for EXP_013m):** X817
  (`GOP`→`since`), X1015 (`Fas`→`the`), X1017 (`Bhar`→`Indian`) invert fully
  (0/25). The direct logit-lens-at-j readout at these cells is not what the
  motor tail makes of the same tensor; EXP_013m arbitrates.

### Updated tested-windows map (hand-off to the J-lens phase)

Whole-word flag = ≥2 unique terminals AND plurality lexical class whole-word.
`via-tail` is direct-vs-tail decode agreement; `basins` is tensor clusters at
0.999. `✓` = flagged; `·` = not flagged; blank = not run.

**Injection axis at extract 21** (bold = new this experiment):

| i | arm | unique | whole-word (of 25) | plurality token | basins | via-tail | flag |
|---|---|---|---|---|---|---|---|
| 0 | O0 | 1 | 0 | `','` ×25 | 1 | 0/25 | · |
| 4 | O4 | 1 | 0 | `','` ×25 | 1 | 0/25 | · |
| **5** | **I5** | **7** | **8** | `'...'` ×7 | **21** | **6/25** | **·** |
| 6 | O6 | 6 | 4 | `'​'` ×12 | 12 | 7/25 | · |
| **7** | **I7** | **1** | **0** | `'oooooooo'` ×25 | **6** | **25/25** | **·** |
| 8 | O8 | 2 | 25 | `' simultaneously'` ×17 | 11 | 17/25 | ✓ |
| **9** | **I9** | **2** | **0** | `'oooooooo'` ×18 | **8** | **7/25** | **·** |
| 10 | A4 | 3 | 25 | `' until'` ×19 | 12 | 23/25 | ✓ |
| **11** | **I11** | **1** | **0** | `'<\|endoftext\|>'` ×25 | **3** | **25/25** | **·** |
| 12 | O12 | 3 | 7 | `'or'` ×11 | 10 | 9/25 | · |
| 14 | O14 | 7 | 19 | `' or'` ×15 | 19 | 17/25 | ✓ |

**Extraction ladders** (bold = new this experiment):

| inject | j | arm | unique | whole-word (of 25) | plurality token | basins | via-tail | flag |
|---|---|---|---|---|---|---|---|---|
| 8 | 15 | A5 | 1 | 25 | `' rant'` ×25 | 3 | 0/25 | · |
| 8 | **17** | **X817** | **1** | **25** | `' GOP'` ×25 | **2** | **0/25** | **·** |
| 8 | **19** | **X819** | **1** | **0** | `"'d"` ×25 | **8** | **15/25** | **·** |
| 8 | 21 | O8 | 2 | 25 | `' simultaneously'` ×17 | 11 | 17/25 | ✓ |
| 10 | **15** | **X1015** | **1** | **25** | `' Fas'` ×25 | **3** | **0/25** | **·** |
| 10 | **17** | **X1017** | **1** | **25** | `' Bhar'` ×25 | **5** | **0/25** | **·** |
| 10 | **19** | **X1019** | **2** | **0** | `'…)'` ×15 | **16** | **1/25** | **·** |
| 10 | 21 | A4 | 3 | 25 | `' until'` ×19 | 12 | 23/25 | ✓ |

**Map observations for the J-lens phase (observation only; no mechanism
asserted):**

- The mechanical whole-word flag marks three injection cells at extract 21:
  i ∈ {8, 10, 14}. The in-fill added **no** new flagged cell on either axis.
- Among the flagged cells the plurality token differs in kind (checkable, not
  interpreted): i=8 `' simultaneously'` and i=10 `' until'` are content words;
  i=14 `' or'` is a function word (and i=14's remaining terminals are
  punctuation/fragments). Recorded for the J-lens phase to adjudicate; not
  resolved here.
- At extract 21 the whole-word flag holds at i=8 and i=10 but **not** at the
  layer between them (i=9) nor at the immediate odd flanks (i=7, i=11): single-
  layer resolution, isolated cells.
- On the extraction axis the flag holds only at j=21 for both flagged
  injections; every rung at j ∈ {15,17,19} is unflagged, and three of them
  (8→17, 10→15, 10→17) invert entirely under the via-tail control.
- Suggested J-lens targets (EXP_012m / EXP_011m / EXP_013m), stated as the cells
  where a robust whole-word signature exists to explain: **(8,21) and (10,21)**
  as isolated single-layer islands, plus **(8,15)** (single direct terminal but
  three tensor basins and a via-tail split, per the registered run). The
  via-tail-inverting cells **(8,17), (10,15), (10,17)** are flagged as readout-
  load-bearing: their direct decode is not tail-robust, so any claim about them
  needs the J-lens re-decode.

**Terminal inventories:** the complete uncurated per-arm terminal lists are in
`results_infill.json`; the full characterisation (basins, direct + via-tail
decode, margins, entropy) is in `terminal_characterisation_infill.json`. No
curated sublist is maintained here.

**Caveats (standing):** single seed; one 25-prompt subset; cluster-threshold
sensitivity unexplored; direct decode at j<23 is a logit-lens-at-j readout whose
off-band unreliability is already measured (and here shows full inversion at
three cells); two-instrument (direct + via-tail) agreement is necessary but not
sufficient. The J-lens re-decode (EXP_013m) remains the registered arbiter for
every mid-stack terminal claim, including all cells in this in-fill.
`terminals_infill.pt` is ~2.0 MB — at the committed-`.pt` size threshold noted
in the PR #4 review (item 3); revisit LFS/release assets if the grid grows.


## 2026-07-24 — EXP_010c-3 IN-FILL SCAN — SUPERSEDED PARALLEL RECORD (retired H11-series numbering)

> **Retained, not deleted.** This section records the **same 225 runs** as the
> EXP_010c-3 IN-FILL SWEEP section immediately above — the two sides' artifacts
> were compared at merge and are bitwise identical. It is the independent
> write-up produced on the `claude/issue-6-akvuxp` line, under that line's own
> hypothesis numbering: **H11 / H11a / H11b**. Those labels are **retired**.
>
> Per the H11 ruling recorded on board discussion #37, H11 belongs to the
> EXP_010d capstone line (PR #5), and the in-fill line's numbering is the
> H12-series used above. The section above is therefore the record of these
> runs; this block is kept because the repo convention is that superseded
> claims stay visible rather than being erased.
>
> The retired labels are deliberately left intact here rather than mechanically
> relabelled to H12/H12a/H13, because the two write-ups pre-registered
> **different criteria, not merely different names** — for example H12a above
> pre-registers a directional sub-prediction ("7→21 whole-word but 5→21 not"),
> whereas H11a below explicitly declines to predict a direction and is evaluated
> against the §3 arm-class rule with a recorded coarseness caveat. Renumbering
> the text would misstate what each side actually pre-registered. Where the two
> records differ on criteria, the section above governs. Tracking: issue #52.


**Spec:** `../../EXP_010c3_SPEC.md` (pre-registered and committed before any
run; task register GitHub issue #6, whose "10 windows / 250 runs" header
arithmetic vs its enumerated 9 windows / 225 runs is recorded there as a
deviation — the enumerated set ran).

**What ran:** `--tier infill --model-path <local>`. 225/225 converged
(lock-in 120–170; 210/225 at the earliest gated lock of 120; 4 at 130,
6 at 150, 5 at 170). Environment
rebuilt in a fresh container and re-verified before running: state dict
316 tensors / wte [50257, 1024] / 24 layers; smoke reproduction gate passed
with **byte-identical committed artifacts** (cross-container repeatability;
same pins torch 2.13.0 / transformer_lens 3.5.1 / py3.11). Deviation: the
sweep executed as three processes — the first died silently at 188/225
(no traceback, no OOM evidence; 7 completed arms already committed by
per-arm checkpoints), X819 rerun as its own process (its 13-run overlap
with the dead process's log is **line-identical** — determinism check),
X817 likewise. Protocol parameters identical throughout. Artifacts:
`results_infill.json`, `terminals_infill.pt`, logs `infill_run*.log`,
`terminal_characterisation_infill.json`.

**Injection in-fill (inject i → extract 21).** New arms marked •; measured
neighbours repeated for the axis to read in order:

| i | Terminals (n=25) | Unique | Token class (per spec §3 rule) |
|---|---|---|---|
| 4 | `','` ×25 | 1 | punctuation funnel |
| •5 | `'...'` ×7, `'…'` ×6, `' Congratulations'` ×4, `' Welcome'` ×3, `'been'` ×2, `'!!'` ×2, `' been'` ×1 | 7 | mixed (word/punctuation/fragment) |
| 6 | `' apologies'`, `' etc'`, `' please'`, `'..'`, `'oooo'`, `'​'` | 6 | mixed |
| •7 | `'oooooooo'` ×25 | 1 | fragment, single terminal (arm class per §3 rule: mixed) |
| 8 | `' simultaneously'` ×17, `' halfway'` ×8 | 2 | whole-word, prompt-dependent |
| •9 | `'oooooooo'` ×18, `'…'` ×7 | 2 | fragment + punctuation |
| 10 | `' until'` ×19, `' forever'` ×5, `' since'` ×1 | 3 | whole-word, prompt-dependent |
| •11 | `'<\|endoftext\|>'` ×25 | 1 | special token, single terminal (arm class per §3 rule: mixed) |
| 12 | `' NHS'`, `' £'`, `'or'` | 3 | mixed (abbreviation/symbol/fragment) |
| 14 | `' DES'`, `' or'`, `' than'`, `' vs'`, `'.'`, `'."'`, `'.)'` | 7 | mixed |

**Extraction columns (rows i=8 and i=10).** New arms •; measured cells
repeated for the axis:

| Window | Terminals (n=25) | Unique | Token class (per spec §3 rule) |
|---|---|---|---|
| 8→15 | `' rant'` ×25 | 1 | whole-word, single terminal |
| •8→17 | `' GOP'` ×25 | 1 | abbreviation, single terminal |
| •8→19 | `"'d"` ×25 | 1 | see classification note below |
| 8→21 | `' simultaneously'` ×17, `' halfway'` ×8 | 2 | whole-word, prompt-dependent |
| •10→15 | `' Fas'` ×25 | 1 | whole-word by the §3 character rule; see note |
| •10→17 | `' Bhar'` ×25 | 1 | whole-word by the §3 character rule; see note |
| •10→19 | `'…)'` ×15, `' […]'` ×10 | 2 | punctuation |
| 10→21 | `' until'` ×19, `' forever'` ×5, `' since'` ×1 | 3 | whole-word, prompt-dependent |

**Classification notes (recorded flat):** the §3 rule is character-based.
(1) `"'d"` contains an apostrophe: it is not *whole-word*, not *punctuation*
(has an alphanumeric), and not *fragment* under the strict "alphabetic
only" wording — it falls to *abbreviation/symbol* by the letter of the
rule, while functionally it is a continuation piece (fragment reading).
Both readings are carried below. (2) `' Fas'` and `' Bhar'` satisfy the
whole-word character test (leading space + alphabetic, not all-caps) while
not being standalone dictionary words; the rule classes them whole-word and
this is recorded as a rule limitation, not adjusted post hoc.

**Characterisation + via-tail control**
(`terminal_characterisation_infill.json`):

| Arm | Window | Tensor basins (top sizes) | Direct decode | Decode-via-tail | Agree | Margin μ/max |
|---|---|---|---|---|---|---|
| I9 | 9→21 | 8 ([9,6,2,2,2…]) | `'oooooooo'` ×18, `'…'` ×7 | `'…'` ×25 | 7/25 | 1.69/2.31 |
| I11 | 11→21 | 3 ([16,7,2]) | `'<\|endoftext\|>'` ×25 | `'<\|endoftext\|>'` ×25 | 25/25 | 1.84/2.17 |
| I7 | 7→21 | 6 ([15,4,2,2,1…]) | `'oooooooo'` ×25 | `'oooooooo'` ×25 | 25/25 | 2.40/2.57 |
| I5 | 5→21 | 21 ([2,2,2,2,1…]) | 7 tokens (table above) | `'!'` ×7, `'…'` ×6, `' once'` ×4, `' been'` ×4, `' Welcome'` ×2, `'been'` ×1, `'.'` ×1 | 6/25 | 0.42/2.06 |
| X1019 | 10→19 | 16 ([3,3,3,2,2…]) | `'…)'` ×15, `' […]'` ×10 | `','` ×13, `' )'` ×5, `' ,'` ×4, `' at'` ×2, `' […]'` ×1 | 1/25 | 1.01/2.05 |
| X1017 | 10→17 | 5 ([13,5,3,3,1]) | `' Bhar'` ×25 | `' Indian'` ×25 | 0/25 | 1.31/1.47 |
| X1015 | 10→15 | 3 ([10,8,7]) | `' Fas'` ×25 | `' the'` ×25 | 0/25 | 1.81/2.26 |
| X819 | 8→19 | 8 ([12,4,2,2,2…]) | `"'d"` ×25 | `"'d"` ×15, `'…'` ×10 | 15/25 | 2.57/3.65 |
| X817 | 8→17 | 2 ([23,2]) | `' GOP'` ×25 | `' since'` ×25 | 0/25 | 1.15/1.45 |

**Observations (no interpretation in this section):**

1. 225/225 converged; no arm required the non-convergence outcome row.
2. I9 (9→21): terminals are fragment + punctuation on all 25 prompts. The
   even-i onset scan's two whole-word arms (i=8, i=10) are therefore not
   connected through i=9 at extraction 21.
3. Every new extraction-column cell (j ∈ {15, 17, 19} in both rows, plus
   both 17/19 cells) produced a single- or two-terminal, prompt-independent
   set. The whole-word **and** prompt-dependent combination observed at
   (8→21) and (10→21) did not occur at any new cell.
4. Direct-vs-via-tail agreement at the new cells: 25/25 only at I7 and I11
   (both funnels agreeing on the same token); 15/25 at X819; ≤7/25
   elsewhere; 0/25 at X1015, X1017, X817. For comparison, the registered
   grid's only high non-trivial agreement remains A4 (10→21, 23/25) and
   O8 (8→21, 17/25).
5. Token identities under via-tail at single-funnel cells (flat statement
   of instrument output, no relationship asserted): X817's `' GOP'` ×25
   reads as `' since'` ×25 through the tail; X1017's `' Bhar'` ×25 as
   `' Indian'` ×25; X1015's `' Fas'` ×25 as `' the'` ×25.
6. Tensor-basin counts at the new cells range 2–21; the two cells with the
   most basins (I5: 21, X1019: 16) are also the two with the lowest margins
   (μ 0.42, 1.01). X817 is near-single-basin (2, sizes [23,2], off-diag
   cos .9993).
7. All new-cell margins (μ 0.42–2.57) sit below the registered A4 value
   (μ 4.20).

**Spec §4 verdicts (mechanical application of the pre-registered table):**

- **H11 (zone contiguity): REFUTED.** Pre-registered criterion: I9
  majority non-whole-word → two-island reading. Observed: 25/25
  non-whole-word at I9. Consequence as pre-registered: 8→21 and 10→21 go
  forward to the J-lens phase as separate target cells.
- **H11a (edge localisation): SUPPORTED under the §3 class rule, with a
  recorded coarseness caveat.** Derivation shown, since the arm-level rule
  and the per-token labels differ: the §3 arm classes are *whole-word*
  (all unique terminals whole-word), *punctuation funnel* (single unique
  terminal of class punctuation), else *mixed*. I7's single `'oooooooo'`
  is a fragment → not whole-word, not a punctuation funnel → arm class
  **mixed**. I11's single `'<\|endoftext\|>'` is a special token → arm
  class **mixed**. I5 (7 unique, mixed tokens) → **mixed**. All three
  flanks therefore share the arm class of their measured outer neighbours
  (O6, O12: mixed); no flank shows a class matching neither neighbour.
  Caveat, stated flat: the *mixed* bucket absorbs qualitatively different
  behaviours here — I7's and I11's single-terminal funnels vs O6's/O12's
  multi-token mixtures — so the class-level verdict is weaker than the
  table-level structure; the per-cell tables above carry what the rule
  compresses, and the axis-table entries now state both labels.
- **H11b (extraction independence): NOT SUPPORTED; refutation row fires at
  (10,19).** Support required all five X arms whole-word — fails (X1019
  punctuation; X817 abbreviation; X819 non-whole-word under either
  reading). The pre-registered refutation criterion (majority
  punctuation/fragment) fires cleanly at (10,19); at (8,19) it fires only
  under the fragment reading of `"'d"` (classification note above); at
  (8,17), (10,15), (10,17) neither pre-registered row fires — those cells'
  single-terminal funnels are recorded flat as outcomes the spec's two
  rows did not anticipate.

**Updated tested-windows map (23 of 300 valid (i≤j) cells measured at the
registered protocol: 6 full-tier + 8 scan + 9 infill).** Cell entries: class shorthand (WW = whole-word,
PD = prompt-dependent, fun = single-terminal funnel):

| i\j | 11 | 15 | 17 | 19 | 21 | 22 | 23 |
|---|---|---|---|---|---|---|---|
| 0 | mixed | | | | punct fun | | `D` fun |
| 4 | | | | | punct fun | | |
| 5 | | | | | mixed | | |
| 6 | | | punct fun (6→17) | | mixed | | |
| 8 | | **WW fun** (`rant`) | abbrev fun (`GOP`) | `'d` fun | **WW PD** | | |
| 9 | | | | | frag/punct | | |
| 10 | | WW-by-rule fun (`Fas`) | WW-by-rule fun (`Bhar`) | punct | **WW PD** | mixed | mixed |
| 11 | | | | | EOT fun | | |
| 12 | | | | | mixed | | mixed (12→23) |
| 14 | | | | | mixed | | |

**Hand-forward to the J-lens phase (issue #6 deliverable 3):** among the
23 measured cells, whole-word **prompt-dependent** terminals occur at
exactly **(8→21)** and **(10→21)** — two isolated cells, not a block and
not a column: i=9 between them is fragment/punctuation, and every measured
extraction depth below 21 in both rows loses either the whole-word class
or the prompt dependence. Whole-word single-terminal funnels additionally
at (8→15) and — by the character rule, with the recorded note — (10→15),
(10→17).
Per-layer lens priority from the two-instrument data: extraction 21 is the
only depth where direct and via-tail readouts agree at high rates (A4
23/25, O8 17/25); agreement collapses at 15–19 (0–15/25). EXP_012m/011m/
013m should therefore target layer-21 states of (8→21) and (10→21) first,
and treat all sub-21 terminal identities as instrument-dependent pending
the J-lens re-decode.

**Post-review corrections (PR #10 review, same day):** (1) earliest-lock
aggregate corrected 214→210 of 225 (recomputed from the artifact: 210 at
120, 4 at 130, 6 at 150, 5 at 170); (2) measured-cell count corrected
32→23 of 300 in the map header and caveats (transposition; the census
count 277 = 300 − 23 was and is consistent); (3) H11a derivation spelled
out after the review flagged that the axis tables' per-token labels
(fragment funnel, special-token funnel) and the verdict's arm classes
(mixed) read as contradictory — recomputation under the pre-registered §3
arm-class rule confirms the verdict unchanged; (4) `<\|endoftext\|>`
escaped in two table rows (rendering). None of these changes any verdict.

**Caveats (standing + new):** single seed; one 25-prompt subset;
cluster-threshold sensitivity unexplored; the §3 token-class rule is
character-based and its *mixed* bucket is coarse (measured above); direct
decode at j<23 is a logit-lens-at-layer-j readout with now-measured
unreliability at j ∈ {15,17,19}; **sampling-resolution caveat:** this map
covers 23/300 valid windows — single-layer structure demonstrated here
(I9, X819) means unmeasured cells cannot be interpolated; the registered
full census `EXP_010c4_SPEC.md` (all 277 remaining cells, same protocol)
addresses this. The J-lens re-decode (EXP_013m) remains the registered
arbiter for all mid-stack terminal claims.

*(end of superseded parallel record — the H11-series labels above are retired; see the H12-series section for the record of these runs.)*

## 2026-07-25 — EXP_010c-3b FOLLOW-UP CHECKS (issue #21)

**Spec:** `../../EXP_010c3b_SPEC.md` (pre-registered before any of these results
existed; committed first). Items ordered so the checks capable of weakening the
2026-07-24 section run first. Model, protocol and subset as registered unless
stated. Artifacts: `funnel_geometry.json`, `tested_windows_map.json`,
`results_det_seed42/1234.json`, `results_subset2.json`, `results_ladder8.json`,
`results_settle.json` (+ matching `.pt`).

### Item 1 — are the single-token funnels a decode-geometry artifact? (no model time)

`analyze_funnel_geometry.py`, state dict only. **S1** = share of 10,000
isotropic random directions whose argmax (through the real `ln_final`, tied
unembedding) is that token; **S2** = percentile of `‖wte[t]‖₂` in the 50257
vocabulary; **S3** = percentile of cosine with the mean embedding-row direction.
Seed 42.

| Set | Token | S1 % | S2 norm pct | S3 cos pct |
|---|---|---|---|---|
| funnel | `'oooooooo'` (I7, I9) | 0.00 | 82.28 | 79.65 |
| funnel | `'<\|endoftext\|>'` (I11) | 0.00 | 0.29 | 37.23 |
| funnel | `' GOP'` (X817) | 0.00 | 5.20 | 55.44 |
| funnel | `"'d"` (X819) | 0.00 | 2.74 | 33.38 |
| funnel | `' Fas'` (X1015) | 0.01 | 92.04 | 9.76 |
| funnel | `' Bhar'` (X1017) | 0.00 | 61.89 | 46.27 |
| funnel (extra) | `'…)'` (X1019) | 0.00 | 84.93 | 37.15 |
| funnel (extra) | `' […]'` (X1019) | 0.00 | 54.70 | 4.85 |
| word contrast | `' until'` | 0.00 | 2.20 | 7.58 |
| word contrast | `' forever'` | 0.00 | 27.85 | 5.22 |
| word contrast | `' since'` | 0.00 | 0.97 | 7.59 |
| word contrast | `' simultaneously'` | 0.00 | 17.25 | 25.69 |
| word contrast | `' halfway'` | 0.00 | 26.18 | 14.70 |
| word contrast | `' rant'` | 0.00 | 26.96 | 74.86 |

Collective S1: funnel **0.01%**, word contrast 0.00%. The census top-20 contains
none of the funnel tokens and is led by unrelated tokens at ≤0.22% each
(`'enegger'`, `' destro'`, `' mathemat'`, `'SPONSORED'`, `'advertisement'`,
`'Interstitial'`, …; full list in the artifact).

**Observations:** no funnel token wins ≥1% of random directions; funnel norm
percentiles are scattered (0.29–92.04) rather than concentrated in the upper
tail; the two values outside the central 90% are **low**-norm; the word-contrast
norm percentiles (0.97–27.85) are if anything lower than the funnel set's.

**Spec §1 verdict (mechanical):** the artifact reading is **not met** (required
≥3 funnel tokens at ≥1% each, or ≥25% collectively; observed 0 and 0.01%). The
"not explained" reading is met on the primary statistic (0.01% ≪ 5%) but its S2
sub-condition is **not literally satisfied**, 2 of 6 tokens falling outside the
central 90%. Recorded under the pre-registered middle row — **quantitative,
partial, no verdict beyond the numbers** — with the observation that both
deviations run in the *low*-norm direction, opposite to the mechanism tested
for. No statistic computed here supports the artifact reading.

**Recorded limitation:** isotropic directions are not distributed like real
residual states. S1 bounds the decoder's reach over *generic* directions only
and does not exclude these tokens winning across the region these loops occupy.
A post-hoc diagnostic on that question is recorded immediately below, labelled
as such.

**POST-HOC diagnostic (NOT pre-registered; no verdict weight against the §1
readings).** `analyze_natural_decode.py`: decode the model's **own natural
states** at each extract layer used by the in-fill arms — one ordinary forward
pass per prompt, no ATR loop — with the same `ln_final → W_U` readout. This
replaces the isotropic null with the states the readout actually meets at that
layer. Registered 25-prompt subset; artifact `natural_decode_posthoc.json`.

| Layer | Natural last-position decode (top entries) | Natural mean-position decode |
|---|---|---|
| 15 | varied, one per prompt | `','` ×14, `'\n'` ×5, `'-'` ×3, … |
| 17 | `' China'`, `' CO'`, `' there'`, `' thousands'`, `' meet'`, … | `','` ×9, `'\n'` ×6, `'-'` ×3, `' the'` ×3 |
| 19 | `' UN'`, `' CO'`, `' quantum'`, `' 300'`, `' meet'`, … | `','` ×6, `'\n'` ×4, `' the'` ×4 |
| 21 | `' WTO'`, `'OH'`, `' quantum'`, `' 300'`, `' the'`, … | `'\n'` ×6, `','` ×6, `' the'` ×5 |

**Observation:** at every one of these layers the natural states decode to
varied, prompt-appropriate tokens (last position) or to common punctuation and
function words (mean position). **None of the funnel tokens — `' GOP'`,
`' Bhar'`, `' Fas'`, `'oooooooo'`, `"'d"` — appears anywhere in the natural
decode at the layer where its arm extracts.** The funnel tokens are therefore
not what this readout generically returns at those layers, on the states the
model itself produces there. Taken with S1, no evidence was found that the
single-token funnels are a property of the decoding step; they remain a
property of the looped dynamics. Stated as an observation; the J-lens re-decode
(EXP_013m) remains the registered arbiter.

### Item 2a — seed variation is a no-op in this harness

I9 (9→21), 3 prompts, seed 42 vs seed 1234, all else identical: every record
identical on `terminal_token`, `terminal_token_id`, `lock_in_iter`, `n_iters`,
`converged`, `top_logit_margin`, `entropy`, and on `final_cos_sim_mean` to 10
decimal places (e.g. `0.9999542832` under both).

**Reading (spec §2a):** the model runs in `eval()` and the gated loop performs
no sampling, so the trajectory is determined by prompt and weights;
`torch.manual_seed` has nothing to act on. **The "single seed" caveat carried in
every section above is therefore not the caveat it appears to be** — the honest
wording is "single prompt subset". #11's registered seed-variation control
cannot vary anything and is flagged there for amendment rather than execution.

### Item 2b — the 9→21 refutation on a disjoint prompt subset

Deterministic round-robin at offset 25 (`derive_prompts.select_subset(25,
offset=25)`; zero overlap with the registered subset; `offset=0` verified to
reproduce the committed subset exactly). 50 runs, 50/50 converged.

| Arm | Window | Terminals | Unique | Plurality class | Whole-word | Locks |
|---|---|---|---|---|---|---|
| A0 | 0→23 | `'D'` ×25 | 1 | fragment | 0/25 | 120 |
| I9 | 9→21 | `'oooooooo'` ×22, `'…'` ×2, `' forever'` ×1 | 3 | fragment | 1/25 | 120, 130 |

**Reproduction gate: PASSED** on the new subset (A0 → `D` 25/25, margin 0.52
throughout), so the I9 result is readable.

**Spec §2b verdict (mechanical): H12's refutation SURVIVES.** The
pre-registered failure condition was ≥2 unique terminals *with a whole-word
plurality*; observed plurality is `'oooooooo'` (fragment) at 22/25, whole-word
1/25. Recorded without curation: `' forever'` — one of A4's registered
terminals — occurs once at 1/25.

**Cross-experiment observation (added on merge; neither experiment states this
alone).** EXP_010c-ROBUST (issue #11, section below) independently ran A4
(10→21) and O8 (8→21) on its "subset B", and EXP_010c-3b ran I9 (9→21) at
`--prompt-offset 25`. These are **the same 25 prompts** — the two entry points
were added in parallel for the same need and are verified equal
(`select_subset_b(n) == select_subset(n, offset=25)`, checked against both
committed audit files). The three cells on that shared disjoint subset:

| i | arm | source | Terminals on subset B | Plurality class |
|---|---|---|---|---|
| 8 | O8 | EXP_010c-ROBUST V3 | `' simultaneously'` ×18, `' halfway'` ×6, `' already'` ×1 | whole-word |
| 9 | I9 | EXP_010c-3b §2b | `'oooooooo'` ×22, `'…'` ×2, `' forever'` ×1 | fragment |
| 10 | A4 | EXP_010c-ROBUST V3 | `' until'` ×22, `' forever'` ×2, `' ('` ×1 | whole-word |

**The whole-word / not / whole-word alternation across i ∈ {8, 9, 10}
reproduces on a prompt set disjoint from the one that produced it.** Stated as
an observation on two independently executed experiments sharing a subset
definition; the standing caveats (one additional subset, single machine,
logit-lens-at-j readout) apply unchanged, and EXP_013m remains the arbiter.

### Item 3 — the whole-word scoring rule, and a correction to the map

Decision taken in the spec before recomputing: **the flag rule is kept
unchanged**; narrowing it after seeing which cell it inconveniences would stop
it being mechanical. A content/function column (spec §3 closed-class list) is
computed by the same code for every cell (`analyze_map.py`). No flag changes.

| i | arm | flagged | plurality token | content/function |
|---|---|---|---|---|
| 8 | O8 | YES | `' simultaneously'` ×17 | content |
| 10 | A4 | YES | `' until'` ×19 | content |
| 14 | O14 | YES | `' or'` ×15 | **function** |

**Correction to the 2026-07-24 hand-off.** That section scored only the
extract-21 injection axis and the ladder rungs below 21. Scored uniformly,
**E23 (10→23) also flags** — 10 unique, 20/25 whole-word, plurality `' self'`
×10 (content), 21 tensor basins — and was omitted from the flagged list handed
forward. Recorded here as an addition, with the caveat the earlier record
already states for j=23: **no via-tail control exists at j=23** (empty tail), so
E23's "25/25" is the mean-vs-last-position check, not readout robustness. E23 is
flagged **and** unarbitrated by the second instrument. Full uniform map (all
three axes, every cell) in `output/tested_windows_map.json`.

### Item 4 — extraction ladder above 21 at injection 8

50 runs, 50/50 converged.

| Arm | Window | Terminals | Unique | Plurality class | Whole-word | Locks |
|---|---|---|---|---|---|---|
| E822 | 8→22 | `' �'` ×25 | 1 | punctuation | 0/25 | 120 |
| E823 | 8→23 | `' �'` ×10, `'<\|endoftext\|>'` ×6, `'…'` ×6, `' to'`, `'\n'`, `' //'` | 6 | punctuation (18/25) | 1/25 | 120–740 |

**Characterisation + via-tail control** (`terminal_characterisation_ladder8.json`,
added 2026-07-25 after the PR #33 review noted the map advertised these columns
for cells that had never been characterised):

| Arm | Window | Tensor basins | Direct decode | Via-tail | Agree | Margin μ |
|---|---|---|---|---|---|---|
| E822 | 8→22 | 2 ([22,3]) | `' �'` ×25 | `' �'` ×25 | 25/25 | 0.891 |
| E823 | 8→23 | 25 (all singletons) | `' �'` ×10, `'<\|endoftext\|>'` ×6, `'…'` ×6 | identical | 25/25 | 0.795 |

Note: E823 is a j=23 arm, so its tail is empty and "25/25" is the
mean-vs-last-position check, not a readout-robustness measure (same caveat as
A0/A3/E23). E822's 25/25 *is* a real one-layer tail check. E823's 25 singleton
tensor basins alongside 6 decode terminals is the same observation class as
A3 and the Stage 1 `Divine` prompts: distinct tensors, shared decode.

**Spec §4 verdict (mechanical): the pre-registered "both degrade" reading is
observed** — neither arm retains the whole-word character, so **j=21 is a sharp
peak at injection 8**, with nothing above or below it retaining the character.

**Asymmetry recorded (observation, not in the pre-registered table):** the two
ladders differ *above* 21. At injection 10, E23 (10→23) flags whole-word (item
3); at injection 8, both E822 and E823 are punctuation-dominant. Also, E823's
lock iterations span 120–740, unlike the uniform 120 seen almost everywhere
else in the grid.

### Item 5 — settle time, and a stopping-rule dependence

Recorded protocol variant: `check_start=10` (earliest reportable lock 30),
arms I7 / I9 / X1017, 5 prompts, all else registered. 15 runs, 15/15 converged.
The registered artifacts are untouched.

Observed locks are **80–90**, not the earliest reportable 30. So gate
satisfaction occurs near iteration 80, and the registered `lock=120` values are
upper bounds set by `check_start=100`, as suspected.

**The more consequential observation.** With only the stopping rule changed,
same arm and same prompts:

| Arm | Terminal agreement (cs=100 vs cs=10) | Registered (lock 120) | Variant (lock 80–90) |
|---|---|---|---|
| I7 (7→21) | **5/5 same** | `'oooooooo'` | `'oooooooo'` |
| X1017 (10→17) | **5/5 same** | `' Bhar'` | `' Bhar'` |
| I9 (9→21) | **0/5 same** | `'oooooooo'` ×4, `'…'` ×1 | `'iren'` ×3, `' would'`, `"'d"` |

**Reading (observation only):** the gated `converged` flag does not imply a
fixed point at every cell. I7 and X1017 return the identical terminal under both
stopping rules; **I9 returns a different terminal on every prompt**, so its
state is still moving between iterations ~80 and ~120 while satisfying the
cosine gate at both. I9's terminal *identity* is stopping-rule dependent; its
lexical *class* (non-whole-word plurality) is stable under both rules, so the
H12 refutation in item 2b is unaffected. Every claim about a specific terminal
token in this record inherits this caveat, and I9 is recorded as a slow-drift
cell.

### Consequences for the 2026-07-24 section (appended, not rewritten)

1. The single-token funnels are **not** shown to be a decode-geometry artifact
   (item 1), subject to the recorded isotropy limitation.
2. The "isolated islands" statement **survives** a disjoint prompt subset
   (item 2b) and the alternative stopping rule (item 5, by lexical class).
3. The flagged-cell list handed to the J-lens phase **gains E23 (10→23)**, which
   the earlier map omitted, and which no via-tail control can arbitrate
   (item 3).
4. Every "single seed" caveat should read **"single prompt subset"** (item 2a).
5. Terminal *identity* claims are stopping-rule dependent at slow-drift cells;
   **I9 is one** (item 5).

**Caveats (standing):** one 25-prompt subset per condition; cluster-threshold
sensitivity still unexplored; direct decode at j<23 remains a
logit-lens-at-layer-j readout; the J-lens re-decode (EXP_013m) remains the
registered arbiter for every mid-stack terminal claim.

### Addendum 2026-07-25 — the J-lens census has since reported (cross-reference)

Recorded here because every section above defers to an instrument that has now
returned a result. `../../RESULTS_JLENS_MEDIUM.md` (issue #15) reports EXP_012m:

- **Medium has no coherent J-lens band** under that experiment's pre-registered
  rule — zero lens-dominant layers.
- **No J-lens correlate of the {8,10}→21 islands.** At the three layers this
  record's map distinguishes, the census reports J agree-final 0.02/0.02/0.02
  and J-better 0.14/0.16/0.12 for i = 8/9/10 — i.e. **8 and 10 are
  indistinguishable from 9 and from their flanks** under that instrument. The
  same non-result holds after shifting the comparison one layer to account for
  the convention difference (ATR inject-*i* splices at `resid_pre(i)`, which is
  lens layer *i−1*'s output): 7/8/9 read 0.20/0.14/0.16 J-better, no islands.

**Consequence for this record, stated flatly:** the ATR-side alternation
(whole-word at i=8 and i=10, not at i=9) is a measurement of this looping
procedure and reproduces on a disjoint prompt subset; it currently has **no
independent corroboration from the second instrument**, and the two instruments
disagree about whether anything distinguishes {8,10} from {9}. Nothing in the
observations above is withdrawn — they were readout-specific from the start, and
that limitation was recorded — but the "the two instruments should localise the
same band" expectation stated in the 2026-07-23 session note is **not** borne
out on Medium.

Still open, and now the remaining route to independent corroboration: EXP_011m
(subspace overlap), which does not go through readout rankings at all. EXP_013m
(re-decode of the frozen terminals) also remains open and inherits the
stopping-rule caveat from item 5 above — at I9 the terminal identity depends on
when the loop is stopped, so a re-decode of a frozen terminal is a re-decode of
one stopping rule's answer.
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

## 2026-07-31 — Erratum: a first, deleted run of this control (restoring the record)

The permutation control above was **registered and executed twice**. Before
the EXP_010c-PERM run of record, a first registration
(`../../PERM_TEST_EXP010c_SPEC.md`, committed `97aeb20`, 2026-07-24 19:24)
was executed the same day: its results section was committed to this file at
`e734ba0` (2026-07-24 19:30; seed 2026, N=10,000, Bonferroni α = 0.00625
across 8 sets). At merge `359c622` (2026-07-25) that section was deleted
from this file — the merge took the main-side version wholesale — with no
supersession marker here or on its spec. That violated the convention that
superseded material stays visible; this note restores the fact of the run.

**What the deleted run found (concordant with the run of record):** A4
direct/via-tail +4.59σ/+4.60σ (p = 0.0022/0.0019, significant after
Bonferroni); O8, A5, the pooled sets and the A1 contrast null. The run of
record above (seed 20260725, α\* = 0.00556) gives A4 +4.69σ/+4.55σ
(p = 0.00090/0.00190) with the same null pattern — the two independent
parameterisations agree on every shared set. One threshold-sensitive value,
recorded flat: A1_tail (p = 0.00620) was tested only in the run of record;
it is significant under the first spec's α = 0.00625 and not under the
second's α\* = 0.00556. The run of record's registered reading stands.

The full deleted section and its artifacts are recoverable at
`git show e734ba0` (results text and `output/permutation_results.json` as
of that commit). This note records existence, verdict, and location; it
does not reconstruct the section. Registered in `../../REGISTER.md`
erratum (b). Executed under TC's in-session direction of 2026-07-31
(delegation recorded in the session that produced PR #55).

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

## 2026-07-26 — EXP_010c-4 census checkpoint: priority block complete (observations only)

**Spec:** `../../EXP_010c4_SPEC.md` (+ §6 amendment). Checkpoint note per
spec §5 — counts and classes only; the full observations section, map, and
H14 evaluation come at census completion. Status: 100/277 arms done (the
complete 4 ≤ i ≤ 14, j ≥ 13 neighbourhood), 177-cell remainder in
progress. Artifacts: per-arm shards `output/results_census/`,
`output/terminals_census/`, all committed. Ops: sweep running across
process relaunches (7 silent process deaths to date, each recovered by
`--resume`; per-arm duty ≈ 92%).

**Convergence:** 2,499/2,500 runs converged (lock 120–170). The single
non-convergence — the first in the programme — is W7_14 (7→14),
D01_water: 1000 iterations, final lag-1 cosine 0.99849 (just under the
0.999 gate), lag_scan monotonically decaying 0.9974 → 0.8792 over lags
1–8 (no multiples-of-p signature), terminal decode `' PLEASE'`. Below the
spec's >5/25 systematic-non-convergence threshold; recorded flat.

**Arm classes over the 100 cells (mechanical, the 010c-3 §3 rule):**
21 whole-word, 53 mixed, 26 punctuation funnels; 58/100 prompt-dependent
(≥2 unique terminals at n=25).

**Whole-word AND prompt-dependent cells (the map's target signature) — 9,
all previously unmeasured:** (5→23) `as`/`name` · (6→23) `dawn`/`once` ·
(8→16) `dozen`/`darn` · (9→20) `enough`/`hopefully`/`survive`/
`heartbeat`/`etc` · (10→16) `etc`/`Fas` · (12→15) `HuffPost`/`Interest` ·
(13→13, single-layer window) `till` ×20 + 4 singles · (13→21)
`Kra`/`until`/`today`/`ever` · (14→16) `impressed`/`anything`/`nor` + 2
singles. Token-identity statements, recorded flat: `' until'` at (13→21)
matches an A4 (10→21) terminal token; `' till'` at (13→13) matches the
token of GPT-2 Small's Stage 1 basin. No relatedness asserted
(anisotropy-corrected permutation control remains pending).

**Whole-word single-terminal funnels — 12:** (5→22) `Happy` · (6→18)
`Republican` · (8→14) `rant` (token shared with measured 8→15) · (8→18)
`lol` · (9→13) `hopefully` · (9→14) and (9→15) `dreaded` · (10→13) `Fas`
(shared with measured 10→15) · (11→14) `Amen` · (11→16) `Afgh` · (12→13)
`evid` · (12→22) `or`.

**Standing note:** all class labels are the character-based rule with its
recorded edges (leading-space alphabetic non-words such as `Fas`, `Kra`,
`Afgh`, `evid` class as whole-word by the rule; the 010c-3 classification
notes apply). Full uncurated inventories live in the shards; no curated
sublist is maintained.

## 2026-07-29 — EXP_010c-4 FULL WINDOW CENSUS COMPLETE (300/300 cells)

**Spec:** `../../EXP_010c4_SPEC.md` (pre-registered 2026-07-24, committed
before any census run; §6 amendment recorded pre-analysis after the PR #10
review). Analysis: `build_final_map.py` (in this directory; refuses a
partial census by default — `--allow-partial` emits provisional
diagnostics with the H14 verdict suppressed).

**What ran:** `--tier census --resume`, 277 previously-unmeasured windows ×
the registered 25-prompt subset = 6,925 runs, gated (0.999 ×3, check_every
10, check_start 100, max_iter 1000), seed 42, L0 natural-pass seeding,
identical protocol to every registered tier. With the 23 cells measured by
the full/scan/infill tiers (not rerun, per the do-not-rerun convention),
**all 300 valid windows 0 ≤ i ≤ j ≤ 23 are now measured**: 7,500 registered
runs total, **7,435 converged**.

**Ops (recorded):** the census executed across many process boundaries —
**14 silent process deaths, 2 container restarts, and 1 filesystem
rollback** (the local worktree reverted ~1 h behind origin; recovered by
fast-forward from the remote, which holds every arm because each is pushed
as it completes). Per-arm shard checkpointing plus artifact-verified
`--resume` meant **zero measured cells were lost or recomputed
inconsistently**; per-arm duty cycle ≈ 92 %. Deviation recorded: many
processes, identical protocol parameters throughout.

**Arm classes over all 300 cells (mechanical, the 010c-3 §3 rule):**
47 whole-word · 162 mixed · 91 punctuation funnels.
**172/300 prompt-dependent** (≥2 unique terminals at n=25).

**`D` inventory:** across all 300 windows, `D` appears as a terminal in
**exactly one cell — (0→23), the full stack.** 1→23, 2→23 and 3→23
(near-full stacks omitting only layers 0, 0–1, 0–2) give `'name'` ×22,
`'_'` ×25 and `' +'` ×23 respectively.

**Non-convergence inventory (8 cells, 65 runs of 7,500):**

| Cell | Non-converged | Note |
|---|---|---|
| 0→4 | 16/25 | systematic (>5/25) |
| 0→5 | 14/25 | systematic |
| 0→1 | 8/25 | systematic |
| 0→6 | 8/25 | systematic |
| 1→3 | 7/25 | systematic |
| 1→2 | 6/25 | systematic |
| 16→23 | 5/25 | below threshold |
| 7→14 | 1/25 | isolated |

All six systematic cells sit at i ≤ 1 in short windows. Where lag_scan was
inspected (7→14) it decayed monotonically over lags 1–8 — no
multiples-of-p signature.

**Whole-word AND prompt-dependent cells — 21 (the J-lens target set):**

| Cell | Terminals (top) | | Cell | Terminals (top) |
|---|---|---|---|---|
| 5→23 | `as` 24, `name` 1 | | 13→13 | `till` 20 +3 |
| 6→23 | `dawn` 23, `once` 2 | | 13→21 | `Kra` 14, `until` 6, `today` 3 |
| 8→9 | `slee` 9, `sensitive` 8 +2 | | 14→16 | `impressed` 12, `anything` 5 +3 |
| 8→11 | `Carb` 15, `incent` 10 | | 15→17 | `manga` 20, `same` 3, `anime` 2 |
| 8→16 | `dozen` 17, `darn` 8 | | 15→19 | `Quebec` 14, `Canadian` 11 |
| 8→21 | `simultaneously` 17, `halfway` 8 | | 16→18 | `Quebec` 23, `same` 2 |
| 9→11 | `Pros` 20, `gmaxwell` 4 +1 | | 17→20 | `but` 16, `lately` 6, `been` 3 |
| 9→20 | `enough` 7, `hopefully` 7 +3 | | 20→20 | `about` 13, `into` 12 |
| 10→10 | `Tooth` 24, `Overt` 1 | | 21→21 | `Mau` 22, `Mace` 2 +1 |
| 10→16 | `etc` 16, `Fas` 9 | | 12→15 | `HuffPost` 23, `Interest` 2 |
| 10→21 | `until` 19, `forever` 5, `since` 1 | | | |

Full uncurated inventories in the per-arm shards. Token-identity
statements only, no relatedness asserted (anisotropy control pending):
`' until'` appears at 10→21, 13→21 and 15→21; `' till'` at 13→13 matches
GPT-2 Small's Stage 1 basin token; `' rant'` at 8→12, 8→14, 8→15;
`' Quebec'` at 15→18, 15→19, 16→18.

**Whole-word single-terminal funnels — 26:** 0→3, 1→1, 1→22, 2→22, 3→22,
5→22, 6→18, 8→14, 8→15, 8→18, 9→12, 9→13, 9→14, 9→15, 10→11, 10→13,
10→15, 10→17, 11→14, 11→16, 12→13, 12→22, 15→18, 16→17, 16→19, 22→22.

**H14 (aliasing materiality) — spec §6 amended criterion: SUPPORTED.**
Eligible cells (census cells with ≥1 already-measured neighbour on the
valid lattice): 50. Cells whose arm class differs from **every** measured
neighbour: **15** — 0→10, 0→12, 4→22, 5→17, 5→22, 6→18, 7→15, 8→18, 8→22,
9→20, 10→14, 11→15, 11→17, 12→22, 13→21. Support required ≥1; observed 15.
(Cells with no measured neighbour are excluded from the evaluation and
appear in the map only, per the amendment.)

**Full map** (rows = inject i, cols = extract j; `W*` = whole-word AND
prompt-dependent, `W` = whole-word funnel, `P` = punctuation funnel,
`m` = mixed; blank = i > j, invalid):

```
      0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
  0   m  P  P  W  m  m  m  m  m  m  P  m  P  P  P  P  P  P  P  P  P  P  P  m
  1      W  m  m  m  m  m  m  m  m  m  m  m  m  P  P  P  P  P  P  P  P  W  m
  2         m  P  P  P  P  P  P  P  P  P  P  P  P  P  P  P  P  P  P  P  W  P
  3            P  P  P  P  P  P  P  P  P  P  P  P  P  P  P  P  P  P  P  W  m
  4               m  m  m  m  m  m  m  m  m  P  P  P  P  P  P  P  P  P  m  m
  5                  m  m  m  m  m  m  m  m  P  P  P  P  P  P  P  m  m  W W*
  6                     m  m  m  m  m  m  m  P  P  P  m  m  W  m  m  m  m W*
  7                        m  m  m  m  m  m  m  m  P  m  m  m  m  m  m  m  m
  8                           m W*  m W*  m  m  W  W W*  m  W  m  m W*  P  m
  9                              m  m W*  W  W  W  W  P  m  m  m W*  m  m  m
 10                                W*  W  P  W  m  W W*  W  m  m  m W*  m  m
 11                                    m  m  m  W  m  W  P  m  m  m  m  m  m
 12                                       m  W  m W*  P  P  m  m  m  m  W  m
 13                                         W*  m  m  m  m  m  m  m W*  P  m
 14                                             m  m W*  m  P  m  m  m  m  m
 15                                                m  m W*  W W*  m  m  m  m
 16                                                   m  W W*  W  m  m  m  m
 17                                                      m  P  m W*  m  P  m
 18                                                         m  m  m  m  m  m
 19                                                            m  m  m  m  m
 20                                                              W*  m  m  m
 21                                                                 W*  m  m
 22                                                                     W  m
 23                                                                        m
```

**Superseded by this section (recorded):** the 010c-3 hand-forward stated
that whole-word prompt-dependent terminals occur "at exactly (8→21) and
(10→21)". That was true of the 23 cells measured then; over all 300 cells
the count is 21. The earlier statement is left in place above as the record
of what the sampled data supported; this section is the complete map.

**Caveats (standing):** single seed; one 25-prompt subset;
cluster-threshold sensitivity unexplored; the §3 token-class rule is
character-based with the recorded edges (leading-space alphabetic
non-words such as `Fas`, `Kra`, `Mau` class as whole-word by the rule);
direct decode at j<23 is a logit-lens-at-layer-j readout — the via-tail
control for this tier is appended below when it completes; the J-lens
re-decode (EXP_013m) remains the registered arbiter for all mid-stack
terminal claims. The layer axes now carry **no sampling assumption**:
every valid (i, j) is measured.

### 2026-07-29 — Census terminal characterisation + decode-via-tail control

`analyze_terminals.py --tier census --decode-via-tail` over all 277 census
arms (6,925 terminal decodes plus cosine clustering at the gate threshold).
Artifact: `output/terminal_characterisation_census.json`. Observations only.

**Tensor basins:** 1–25 per arm, mean 10.6 (25 = fully prompt-private).

**Direct-decode vs decode-via-tail agreement, aggregated by extraction
depth j** (census arms only; **the j=23 row is not a via-tail measurement**
— the tail is empty there, so the figure is the mean-vs-last-position
decode check, same convention as the earlier tiers):

| j | arms | mean agree /25 | arms ≥20/25 |
|---|---|---|---|
| 0–4 | 15 | 0.0–0.3 | 0 |
| 5–9 | 40 | 1.5–3.3 | 0 |
| 10–14 | 64 | 2.8–5.6 | 2 |
| 15–19 | 83 | 4.8–8.9 | 11 |
| 20 | 21 | 9.0 | 3 |
| 21 | 11 | 6.6 | 1 |
| 22 | 22 | 18.9 | 12 |
| 23 | 21 | 24.8 *(not via-tail; see note)* | 21 |

Per-depth means rise monotonically-with-noise from ~0 at j≤4 to 18.9 at
j=22. The pre-census prior (high agreement only at j=21) is **not**
reproduced across the full census: j=21's census arms average 6.6/25,
while j=22 averages 18.9/25.

**The 21 target cells (whole-word AND prompt-dependent), agreement and
basins.** 19 are census-measured; (8→21) 17/25 and (10→21) 23/25 come from
the earlier tiers:

| Cell | Basins | Agree | Cell | Basins | Agree |
|---|---|---|---|---|---|
| 5→23 | 5 | 25/25 | 13→13 | 7 | 20/25 |
| 6→23 | 5 | 25/25 | 13→21 | 13 | 16/25 |
| 8→9 | 25 | 2/25 | 14→16 | 16 | 1/25 |
| 8→11 | 7 | 0/25 | 15→17 | 10 | 0/25 |
| 8→16 | 6 | 0/25 | 15→19 | 7 | 16/25 |
| 9→11 | 13 | 16/25 | 16→18 | 3 | 12/25 |
| 9→20 | 17 | 13/25 | 17→20 | 13 | 18/25 |
| 10→10 | 20 | 0/25 | 20→20 | 13 | 25/25 |
| 10→16 | 8 | 14/25 | 21→21 | 11 | 0/25 |
| 12→15 | 11 | 1/25 | | | |

Agreement within the target set spans the full range (0/25 to 25/25), so
the whole-word prompt-dependent class does **not** coincide with the
high-agreement class. Recorded flat: (8→16) direct `dozen`/`darn` reads as
`' just'` ×25 through the tail; (10→10) direct `Tooth` reads as `' Imp'`;
(15→19) direct `Quebec`/`Canadian` reads as `Quebec`/`Montreal`, and
(16→18) as `Quebec`/`Ottawa`.

**Standing caveat, reinforced:** at j<22 the two readout instruments
disagree on most cells, so mid-stack terminal identities remain
instrument-dependent. The J-lens re-decode (EXP_013m) is the registered
arbiter; this table is the prior it will be compared against.

### Post-review corrections (PR #51 review, 2026-07-29)

The census closing changes were re-submitted as a narrow review-only PR
(#51, base `claude/issue-6-akvuxp`) because PR #10's 554 generated data
files push it past the reviewer's file limit. Four findings, all accepted;
**no measured value changed** — `build_final_map.py` reproduces 300/300
cells, 47/162/91 arm classes, the 21-cell target set, 8 non-convergence
cells, `D` at (0,23) only, and H14 SUPPORTED 15/50 exactly as recorded
above.

1. **Partial-census guard was opt-in (major).** This section claimed the
   analysis script "refuses to run on a partial census", but the guard sat
   behind an optional `--require-complete` flag: run without it, a partial
   artifact set emitted a final map and an H14 verdict indistinguishable
   from the real one. The record over-claimed the code. Inverted: refusal
   is now the default and `--allow-partial` is the explicit override,
   which suppresses the H14 verdict (eligibility and the "differs from
   every measured neighbour" test both read neighbours, so a partial
   lattice can flip a verdict as cells land) and stamps the map
   PROVISIONAL. Wording above corrected to match.
2. **Machine-specific artifact path (major).** `BASE` was an absolute
   `/home/user/...` path, so the analysis was unrunnable off this
   container. Now derived from `__file__`, matching `analyze_terminals.py`.
3. **Incomplete target-to-artifact manifest (major).** The runbook's §3a
   hand-forward listed all 21 target cells against `terminals_census/`,
   but only **19** are census arms: 8→21 is arm `O8` in
   `terminals_scan.pt` and 10→21 is arm `A4` in `terminals_full.pt`. An
   operator loading only `terminals_census/` would have silently dropped
   two targets. §3a now carries an exact per-target manifest. The same
   edit fixed a stale EXP_011m frozen-input reference to
   `output/terminals.pt`, a file that does not exist.
4. **Over-broad review filters (minor).** `.coderabbit.yaml` excluded
   `**/output/...` repo-wide, which would have silently un-reviewed the
   artifacts of every future experiment. Scoped to the `exp_010c_windows`
   tree.

Findings 1–3 are defects in the *instrument and its documentation*, not in
the measurements; the census artifacts they read are unchanged and were
never regenerated.
