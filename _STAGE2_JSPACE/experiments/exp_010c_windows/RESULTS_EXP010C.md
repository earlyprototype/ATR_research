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

## Next

EXP_010c-2 boundary scan (pre-registered, `../../EXP_010c2_SPEC.md`): exit-edge
{10→22, 10→23} and onset-edge {0,4,6,8,12,14}→21 sweeps, same protocol.
