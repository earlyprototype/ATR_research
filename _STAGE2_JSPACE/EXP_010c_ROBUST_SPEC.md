# EXP_010c-ROBUST — Seed and Prompt-Subset Robustness for the Word-Forming Windows (pre-registered spec)

**Status:** PRE-REGISTERED — committed before any run.
**Created:** 2026-07-25
**Parent:** `EXP_010c_SPEC.md` / `EXP_010c2_SPEC.md`. Executes planned-controls
item 2 of `experiments/exp_010c_windows/RESULTS_EXP010C.md` (issue #11).

---

## 1. Question

All EXP_010c / 010c-2 results are single-seed (42), single-subset (the
registered 25-prompt round-robin subset). The two strongest observations —
A4 (10→21)'s 3 whole-word prompt-dependent terminals with the grid's highest
margins, and the i ∈ {8, 10} whole-word injection zone (O8 8→21) — could in
principle be properties of that seed or that subset rather than of the model.
This control decides which.

## 2. Where the seed enters (recorded before running)

The gated ATR protocol (`atr_engine2.run_atr_gated`) contains **no sampling**:
no dropout (eval mode), no stochastic decoding, no random initialisation of
the loop state (L0 natural-pass seeding is a deterministic forward pass of the
prompt). The only RNG touchpoint in the runner is `torch.manual_seed(<seed>)`
set before model load. The pre-registered expectation therefore includes the
possibility that the seed variants reproduce the registered run **exactly**
(bit-level or token-level); if that happens it is recorded as the observation
— "the protocol is deterministic w.r.t. the global torch seed on this
machine" — and the burden of robustness shifts entirely to the subset variant
(Variant 3), which changes the actual input data. No variant is skipped on
this expectation: exact repetition must be demonstrated, not assumed.

## 3. Design

**Model:** gpt2-medium, offline load via `--model-path` (same 1,520,013,706-byte
checkpoint as the registered runs; already on disk this session).

**Arms (3):** A0 (0→23, baseline / reproduction anchor), A4 (10→21), O8 (8→21).

**Protocol:** identical to the registered full tier — gated (cos > 0.999 ×3,
checks every 10 past check_start=100), max_iter=1000, L0 natural-pass seeding,
terminal mean+last vectors saved per (arm, prompt). Runner:
`run_exp010c.py` with minimally added `--seed` / `--subset` / `--out-suffix`
parameters (recorded diff, no refactor).

**Variants (3 × 3 arms × 25 prompts = 225 runs):**

| Variant | Seed | Prompt subset | Artifact suffix |
|---|---|---|---|
| 1 | 1337 | registered 25-prompt subset | `robust_seed1337` |
| 2 | 2718 | registered 25-prompt subset | `robust_seed2718` |
| 3 | 42 | disjoint subset B (below) | `robust_subsetB` |

**Subset B derivation rule (deterministic, no hand-picking):** extend the
registered rule — round-robin over the 7 categories, alphabetical by prompt ID
within category — to 50 prompts and take positions 26–50, i.e. the NEXT 25
prompts after the registered subset under the identical ordering
(`derive_prompts.select_subset(50)[25:]`, added as `select_subset_b()`).
Verified disjoint from the registered subset before registration. Full list:

| # | ID | Category |
|---|---|---|
| 1 | C04_humpty | Simple |
| 2 | F04_argument | Vulgarity |
| 3 | G04_fibonacci | Wild |
| 4 | E05_finance | Acronyms |
| 5 | D05_amino | Chemical |
| 6 | A05_evolution | Complex |
| 7 | B05_mlk | Narrative |
| 8 | C05_twinkle | Simple |
| 9 | F05_rant | Vulgarity |
| 10 | G05_primes | Wild |
| 11 | E06_medical | Acronyms |
| 12 | D06_physics_eq | Chemical |
| 13 | A06_epistemology | Complex |
| 14 | B06_sources | Narrative |
| 15 | C06_dog | Simple |
| 16 | F06_dismissal | Vulgarity |
| 17 | G06_binary | Wild |
| 18 | E07_military | Acronyms |
| 19 | D07_dna | Chemical |
| 20 | A07_sociology | Complex |
| 21 | B07_breaking | Narrative |
| 22 | C07_cat_mat | Simple |
| 23 | F07_shock | Vulgarity |
| 24 | G07_the | Wild |
| 25 | E08_academic | Acronyms |

(Prompt texts are the Stage 1 record's verbatim strings, recovered from
`dissolution_sentences.md` exactly as for the registered subset; the executed
list is additionally saved to `output/prompt_subset_b.json` at run time for
audit.)

**Category-composition note (recorded):** the registered subset holds 4
prompts each of Acronyms/Chemical/Complex/Narrative and 3 each of
Simple/Vulgarity/Wild; subset B holds the complementary 3/3/3/3/4/4/4 plus a
4th Acronyms — compositions differ by construction of the round-robin split,
not by choice.

## 4. Analysis

Per variant, `analyze_terminals.py --tier robust_<variant> --decode-via-tail
--model-path <local>`: tensor-basin clustering at the 0.999 gate threshold,
direct decode, decode-via-tail (through j+1→23), agreement, margins, entropy.
Artifacts per variant: `results_robust_<variant>.json` +
`terminals_robust_<variant>.pt` + `terminal_characterisation_robust_<variant>.json`,
committed after each variant completes (≤ ~2 MB per convention; if a `.pt`
exceeds it, note and commit JSON only).

## 5. Pre-registered readings (defined before any run)

Registered-run reference values (from `RESULTS_EXP010C.md`, 2026-07-23
sections): A0 → `D` ×25; A4 → {`' until'` ×19, `' forever'` ×5, `' since'` ×1};
O8 → {`' simultaneously'` ×17, `' halfway'` ×8}.

**Primary criterion (basin structure):**

| Observation | Reading |
|---|---|
| In EVERY variant: A0's direct decode is `D` for all 25 prompts AND A4's direct-decode terminal set contains ≥2 of its 3 registered terminal types (`' until'`, `' forever'`, `' since'`) | **Same-basin-structure ("stable")** |
| Anything else | **Variant-dependent** — describe exactly what differed (which variant, which arm, which terminals), no adjectives |

**Secondary reading (whole-word injection zone), recorded separately and not
folded into the primary criterion:** O8's direct-decode terminals are
whole-word tokens (alphabetic token text, ignoring the leading space) for all
25 prompts in every variant → "zone-stable at i=8"; otherwise list the
non-word terminals per variant.

**Also recorded per variant (observations, no thresholds):** unique-terminal
counts, lock-in iterations, margin μ/max, via-tail agreement, tensor-basin
counts, and — for Variants 1–2 — whether the records are identical to the
registered run (per §2).

Interpretation beyond the mechanical readings above is deferred to a session
note; the results section stays observations-only.

## 6. Execution order

Variant 1 → Variant 2 → Variant 3, arms in order A0, A4, O8 within each;
commit and push after each variant (per-arm checkpointing inside the runner
guards against container restarts). Est. 2–4 h CPU total.
