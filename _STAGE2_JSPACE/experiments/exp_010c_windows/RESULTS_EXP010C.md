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

## Pending

1. ~~Smoke with real weights~~ ✅ above.
2. **Pilot:** `--tier pilot` — 5 prompts × 6 arms, max_iter=300. Directional
   signal only (non-registered tier). IN PROGRESS this session.
3. **Registered run:** `--tier full` — 25 × 6, gated, max_iter=1000. Overnight
   CPU. Verdicts per spec §6 recorded HERE.
