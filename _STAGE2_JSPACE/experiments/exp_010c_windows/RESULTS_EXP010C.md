# EXP_010c — Results Record

**Spec:** `../../EXP_010c_SPEC.md` (pre-registered before any run).
**Status:** harness validated; registered run NOT yet executed.

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
