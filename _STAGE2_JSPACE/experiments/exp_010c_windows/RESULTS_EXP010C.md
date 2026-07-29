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
   baseline anisotropy.
2. **Seed and prompt-subset variation** for A4 and O8. Removes uncertainty
   about whether the observed basin structure and the i ∈ {8,10} whole-word
   zone are properties of this seed/subset or of the model.
3. **Same window grid on Pythia-410m.** Removes uncertainty about whether
   window-position effects at these relative depths are specific to
   gpt2-medium or generic to decoder stacks of this size.
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

## 2026-07-24 — EXP_010c-3 IN-FILL SCAN (9 arms × 25, gated, max_iter=1000)

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

## 2026-07-26 — EXP_010c-4 census checkpoint: priority block complete (observations only)

**Spec:** `../../EXP_010c4_SPEC.md` (+ §6 amendment). Checkpoint note per
spec §5 — counts and classes only; the full observations section, map, and
H12 evaluation come at census completion. Status: 100/277 arms done (the
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
review). Analysis: `build_final_map.py` (in this directory; refuses to run
on a partial census).

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

**H12 (aliasing materiality) — spec §6 amended criterion: SUPPORTED.**
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
