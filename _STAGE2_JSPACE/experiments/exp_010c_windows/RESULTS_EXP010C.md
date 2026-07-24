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

**Headline findings:**

1. **Reproduction gate: PASSED.** A0: 25/25 → `D`, near-identical tensors.
2. **The band window (A4) is qualitatively unlike every other arm:** 12
   distinct tensor basins (a genuinely diverse landscape, not a funnel),
   decoding to a small semantically coherent temporal-durative set
   (`until`/`forever`/`since`), with by far the highest readout confidence —
   and it is the **only** arm whose mid-stack decode survives the
   decode-via-tail control (23/25). The motor pathway itself endorses the
   band window's readout.
3. **The final-layer concern, answered empirically:** mid-stack decodes at
   j=11, 15, 17 flip completely under via-tail (0/25 agreement) — the
   `ln_final` calibration mismatch is real and large *except* in the band
   window, where the readout is robust to it. This both vindicates the concern
   and shows it does not touch the A4 result.
4. **The temporal theme extends across the band:** A5's via-tail decode is
   `' endless'` ×18 — the *other* in-band window also speaks duration once
   read through the model's own pathway, and its via-tail split (18/7)
   tracks its tensor-basin split ([17,7,1]). In-band iteration settles into
   duration-words: *until, forever, since, endless*.
5. **A3 (12→23) is a mass `Divine`-like dissociation:** 21 distinct tensor
   fixed points sharing one decode (`'"'`) — many private dynamics, one
   public word.

**Spec §6 verdicts (pre-registered readings):**

- **H9 (collapse rescue): SUPPORTED.** The `D` collapse is cut-dependent:
  no window reproduces it, and window placement produces four qualitatively
  distinct landscape types (uniform funnel / junk funnel / diverse-semantic /
  many-private-one-decode).
- **H9a (band placement): SUPPORTED, with a recorded nuance.** The
  paper-mapped band-exact window is the richest by every measure (basin
  count, prompt-dependence, semantic coherence, margin, readout robustness).
  Nuance: in-band A5 (8 layers) funnels to a single direct-decode token at
  full scale, so placement is necessary but window length co-determines
  richness — exactly the question EXP_010c-2's edge sweeps are registered
  to resolve.

**Caveats (standing):** single seed; one 25-prompt subset; cluster threshold
sensitivity unexplored; direct decode at j<23 is a logit-lens-at-layer-j
readout (now with measured unreliability off-band); the J-lens re-decode
(EXP_013m) remains the arbiter for all mid-stack claims.

## 2026-07-23 — EXP_010c-2 BOUNDARY SCAN (8 arms × 25, gated, max_iter=1000)

**What ran:** `--tier scan --model-path <local>`. 200/200 converged.
Artifacts: `output/results_scan.json`, `output/terminals_scan.pt`.

**Onset-edge sweep (inject i → extract 21), with A4 (i=10) from the main grid:**

| i | Terminals | Unique | Character |
|---|---|---|---|
| 0 | `','` ×25 | 1 | junk funnel |
| 4 | `','` ×25 | 1 | junk funnel |
| 6 | `' apologies'`, `' etc'`, `' please'`, `'..'`, `'oooo'`, `'​'` | 6 | transitional junk |
| **8** | **`' halfway'`, `' simultaneously'`** | 2 | **semantic: temporal/positional** |
| **10** | **`' until'`, `' forever'`, `' since'`** (A4) | 3 | **semantic: temporal-durative** |
| 12 | `' NHS'`, `' £'`, `'or'` | 3 | lexical junk |
| 14 | `' DES'`, `' or'`, `' than'`, `' vs'`, `'.'`, `'."'`, `'.)'` | 7 | comparative/punctuation fragments |

**Exit-edge sweep (inject 10 → extract j):**

| j | Terminals | Unique | Character |
|---|---|---|---|
| 21 | `until`/`forever`/`since` (A4) | 3 | semantic, via-tail-robust |
| 22 | incl. `' until'`, `'<|endoftext|>'`, `' Tak'`, `' honor'`… | 7 | degrading, temporal remnant |
| 23 | incl. `' self'` ×10, `' until'` ×5, misc junk | 10 | fragmented; **no `D`** |

**Verdicts (spec `EXP_010c2_SPEC.md` §4):**

- **H10 (edge localisation): SUPPORTED.** Transitions are sharp, not smooth:
  junk funnel (i≤4) → transitional junk (i=6) → semantic temporal regime
  (i=8–10) → junk again (i≥12). The landscape's character is a non-monotonic
  function of the injection point with an identifiable semantic window.
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

**The ATR-derived prediction handed to the J-lens census (EXP_012m):**
word-like, prompt-dependent terminals occur only for injection at layers
8–10 (extraction 15–21). Whether that zone has anything to do with a
*workspace* is exactly what the census must test — the coincidence with the
paper's 38% onset mapping (layer ≈9) is consistent-with on one model, one
seed, one prompt subset, and the mapping's rounding was our own choice. The
"ignition" reading (band onset as the special injection point) is recorded
as interpretation only. **The prediction is falsifiable in both directions:
if the J-lens band census puts Medium's onset far from layers 8–10, or finds
no coherent band, the workspace reading of the injection zone dies.**

**Token-pattern note (recorded observation, hypothesis NOT registered):**
the word-like arms' terminals read as temporal/positional vocabulary
(*until, forever, since, endless, simultaneously, halfway*). CAUTION: this
is eyeballed semantic clustering over ~6 token types — the exact move Stage
1's `W_E` permutation test killed once already (all-warm neighbourhoods were
an anisotropy artifact). Before this pattern is called a theme anywhere, it
needs the permutation-null treatment: are these tokens closer to each other
in embedding space than matched random terminal sets, under the anisotropy-
corrected null? Until then it is a curiosity, not a finding.

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

## Registered disconfirmers (added 2026-07-23, after review)

This programme is exploration, not validation. The following tests would
*hurt* the workspace reading, and are registered so the frame can lose:

1. **Anisotropy null for the token pattern:** permutation test of the
   "temporal" terminal set against matched random terminal draws in W_E
   (Stage 1 `02b` pattern). Kills the theme if it fails.
2. **Seed and subset robustness:** rerun A4 and O8 with a different seed and
   a disjoint 25-prompt subset. If the basin structure or the 8–10 zone
   moves, the localisation claim weakens accordingly.
3. **Placebo windows on a control model:** run the same window grid on
   Pythia-410m (whose full-stack regime is fragmentation, not collapse). If
   band-mapped windows there also produce "richest" landscapes at ~38%
   depth, the effect is generic depth-arithmetic, not workspace structure.
4. **Hook-point control:** inject at `resid_post` of layer i−1 instead of
   `resid_pre` of layer i (identical computation, different hook) and at
   i=1 vs i=0 specifically, to test whether the layer-0 funnels are an
   artifact of the raw-embedding slot rather than "sensory splicing."
5. **The J-lens census itself (EXP_012m):** an onset far from layers 8–10,
   or no coherent band at 345M, falsifies the workspace reading of the
   injection zone outright.

If 1–2 fail, the token-theme and localisation claims are withdrawn. If 3–5
fail, the workspace framing of EXP_010c reverts to a classical-dynamics
description (window-dependent attractor landscapes, mechanism unknown) — per
the Stage 2 plan's standing kill criteria.

## Next

Phase 0 / J-lens track per `RUNBOOK_JLENS_MEDIUM.md`, with branch priors now
set by both experiments: EXP_011m first (are the A4/O8 temporal terminals
J-space-loaded where the funnels are not?), EXP_012m census testing the
**layers ≈7–11 (~30–45% depth) band-onset prediction**, EXP_013m re-decode of
the temporal-family terminals and the `D`/`','` funnels.
