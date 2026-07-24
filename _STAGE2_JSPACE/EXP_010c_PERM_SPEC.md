# EXP_010c-PERM — Anisotropy-corrected permutation test on terminal-token relatedness (pre-registered spec)

**Status:** PRE-REGISTERED — recorded before the analysis runs. This file is
committed on its own, before the script or any results JSON.
**Created:** 2026-07-24
**Parent:** `experiments/exp_010c_windows/RESULTS_EXP010C.md` §"Planned controls
— next observations", item 1. Implements GitHub issue #7.
**Method precedent:** Stage 1's `02b_permutation_test.py` (`W_E` anisotropy
null), which reclassified the GPT-2 Small "all-warm basins" reading as an
embedding-anisotropy artifact (decision log 2026-07-11; public repo
`docs/FINDINGS.md`).

---

## 1. Question

Are the terminal tokens produced by the GPT-2 Medium window loops (EXP_010c /
010c-2) closer to each other in embedding space than **matched** random token
sets, under an anisotropy-corrected null?

The word-producing arms landed on tokens that *read* as a temporal/durative
family (`until`, `forever`, `since`, `endless`, `simultaneously`, `halfway`).
That pattern is an eyeballed clustering, explicitly quarantined in RESULTS
pending this test. Embedding spaces are anisotropic — random token pairs
already have high baseline cosine — so "these tokens look related" is cheap
until a **matched** null says otherwise. This test decides signal vs artifact.

This is a pure analysis of committed artifacts plus the gpt2-medium embedding
matrices. **Zero model-forward-pass time.**

## 2. Pre-registered token sets under test

All sets are the **full, uncurated** terminal inventories per arm — never a
hand-picked sublist. Statistic is computed over **unique token types** (a token
that decodes to the same string is counted once regardless of basin occupancy;
occupancy reflects dynamics, not semantics — decided up front, unweighted).

Token IDs for direct-decode sets are taken verbatim from the per-record
`terminal_token_id` fields in `results_full.json` / `results_scan.json`.
Via-tail sets come from `terminal_characterisation_{full,scan}.json`
(`tail_decode_terminals`); their strings are re-encoded to IDs with the
gpt2-medium tokenizer (asserted single-token; cross-checked against
results-derived IDs where the same string appears there).

Sets with only one unique type cannot yield a pairwise cosine and are excluded
from the test (recorded, not silently dropped): direct-decode A0 (`D`), A5
(`rant`), O0/O4 (`,`).

**Word-producing arms — direct decode (ln_final→W_U at extract layer j):**

| ID | Arm / window | Unique types |
|---|---|---|
| `WS_A4` | A4 10→21 | until, forever, since (3) |
| `WS_O8` | O8 8→21 | halfway, simultaneously (2) |
| `WS_POOL` | A4 ∪ O8 ∪ A5 (pooled word-arms) | until, forever, since, halfway, simultaneously, rant (6) |

**Word-producing arms — via-tail decode (through j+1→23):**

| ID | Arm | Unique types |
|---|---|---|
| `WT_A4` | A4 | until, forever, since (3) |
| `WT_O8` | O8 | simultaneously, just, `'` (3) |
| `WT_A5` | A5 | endless, `'` (2) |
| `WT_POOL` | A4 ∪ O8 ∪ A5 via-tail | union of the above |

**Contrast sets (funnel / off-band arms — should sit near the null):**

| ID | Arm / window | Unique types | Role |
|---|---|---|---|
| `CS_A1` | A1 0→11 direct | `ing`, `,` (2) | punctuation funnel |
| `CS_A3` | A3 12→23 direct | `"`, `work` (2) | punctuation/high-basin arm |
| `CS_O14` | O14 14→21 direct | or, `."`, vs, `.)`, DES, `.`, than (7) | off-band mixed, size-matched to `WS_POOL` |

The `WS_POOL` set deliberately includes `rant` (not temporal) — that is the
honest uncurated pool. Removing it would be exactly the evocative-token
selection bias this test exists to defeat.

## 3. Space

GPT-2 ties input and output embeddings (`lm_head.weight == wte.weight`), so in
the raw state dict `W_E` and `W_U` are the *same* matrix. The two spaces are
made distinct here in the way the actual decode makes them distinct:

- **W_E space** — cosine between raw `wte` rows: `cos(wte[a], wte[b])`.
- **W_U space** — cosine in the ln_final-gain-weighted space the terminals were
  actually decoded through. The token-discriminative part of the final decode
  is `logit_t = (γ ⊙ x̂ + β) · wte[t]`, so the effective unembedding direction
  for token t is `u_t = γ ⊙ wte[t]` with `γ = ln_f.weight`. W_U-space cosine is
  `cos(γ⊙wte[a], γ⊙wte[b])`. This is the "arguably right" space named in the
  issue (terminals are decoded through W_U), and it is genuinely distinct from
  W_E because of the γ weighting.

Both spaces are reported side by side for every set.

## 4. Statistic

Mean pairwise cosine similarity within the set, over all `C(k,2)` unordered
pairs of the k unique token types. Unweighted (see §2).

## 5. Null (the anisotropy correction)

For each set, **N = 10,000** matched random token sets. A null set replaces each
observed token with one random vocab token **matched** on the properties that
drive baseline (anisotropic) similarity:

1. **Leading-space status** — exact match on whether the token's byte-level
   string begins with the GPT-2 space marker `Ġ`.
2. **String length** — decoded length within **±1 character** of the observed
   token.
3. **Frequency band** — BPE **merge rank** within the same quintile band as the
   observed token (merge rank = the line index at which the token's byte-pair is
   formed in `merges.txt`; earlier merge ⇒ more frequent. Byte-level base tokens
   that are never formed by a merge get rank 0 and fall in the most-frequent
   band). Merge rank is the issue-sanctioned frequency proxy.

Matching is per-token: the candidate pool for observed token t is
`{v : samespace(v)=samespace(t) ∧ |len(v)−len(t)|≤1 ∧ band(v)=band(t)}`,
excluding the observed token itself and (within one null set) tokens already
drawn for that set — so a null set has k distinct types, like the observed set.
If a token's matched pool is smaller than a floor (< 20 candidates), the
frequency band is relaxed first, then the length window (to ±2), and the
relaxation is **recorded per token** in the results JSON. An unmatched null
flatters the result; matching is exactly the correction Stage 1's lesson
demands.

Special tokens (`<|endoftext|>`, id 50256) are excluded from both observed sets
and the candidate vocabulary.

## 6. Readout

Per set, per space:

- **observed** mean pairwise cosine;
- **null_mean**, **null_sd** over the 10,000 matched sets;
- **effect size** z = (observed − null_mean) / null_sd (in null SDs);
- **empirical p-value** (one-sided, testing observed > null):
  p = (1 + #{null ≥ observed}) / (1 + N).

**Multiple comparisons:** every set × space is one test. With 10 sets × 2
spaces = 20 tests, the Bonferroni-adjusted threshold is α = 0.05 / 20 = 0.0025.
All tests are reported with this threshold; no set is dropped post hoc.

## 7. Pre-registered readings (written before running)

| Observation | Reading |
|---|---|
| Word-arm sets: observed ≫ null (p < 0.0025, z large positive) | The relatedness pattern **survives its first control**. Still not licensed as "semantic" — that word stays quarantined pending the J-lens; the finding is only "closer than a matched anisotropy null." |
| Word-arm sets: observed ≈ null (p not significant) | The apparent temporal/durative family is an **anisotropy/matching artifact**. Recorded plainly; the token-pattern thread closes the way Stage 1's all-warm claim did. |
| Contrast sets significant too | Weakens any word-arm-specific claim — the test is flagging a generic property of loop terminals, not the word family. Reported as such. |
| Mixed (significant in one space, not the other) | Report both; the W_U (decode) space is the more decision-relevant of the two, but neither is suppressed. |

Either direction is a finding. The result calibrates how much attention the
token pattern earns in the J-lens phase; it gates and blocks nothing.

## 8. Conventions

- RNG seed fixed and recorded in the results JSON (`seed`).
- Deterministic throughout; the same seed reproduces every number.
- Results go to the observations register (`RESULTS_EXP010C.md`, dated section):
  numbers, not adjectives. Interpretation, if any, is labelled as such.
- Committed after the run alongside the spec: the analysis script
  (`permutation_test.py`) and a per-set results JSON
  (`output/perm_test_results.json`).
- Planned-controls item 1 in RESULTS is marked done with a pointer to that
  section.
