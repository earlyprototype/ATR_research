# Permutation Test — Terminal-Token Relatedness (pre-registered spec)

> **SUPERSEDED — 2026-07-31.** The executed registration of record for the
> issue #7 permutation control is **`EXP_010c_PERM_SPEC.md`** (committed
> `0ca5829`, 2026-07-25; results in `RESULTS_EXP010C.md`
> §"2026-07-25 — EXP_010c-PERM"). This spec was the first of two concurrent
> registrations of the same control, and it **was executed**: its run
> (seed 2026, α = 0.00625) was committed at `e734ba0` on 2026-07-24, and its
> results section was deleted from `RESULTS_EXP010C.md` at merge `359c622`
> without a supersession marker — a house-convention violation now recorded
> in `REGISTER.md` erratum (b) and in a dated erratum note in
> `RESULTS_EXP010C.md`. The deleted run was concordant with the run of
> record (A4 ≈ +4.6σ, significant, in both; all other shared sets null in
> both). Recoverable at `git show e734ba0`. Original text intact below.
> Executed under TC's in-session direction of 2026-07-31 (delegation
> recorded in the session that produced PR #55).

**Status:** ~~PRE-REGISTERED — recorded before any computation.~~ SUPERSEDED
(see header note; the pre-registration was genuine and preceded its run).
**Created:** 2026-07-24
**Parent:** RESULTS_EXP010C.md §Planned controls, item 1.
**Pattern:** Stage 1 `02b_permutation_test.py` (Lucier repo), adapted for
matched nulls and multiple token sets.

---

## 1. Question

The word-producing arms of EXP_010c / 010c-2 (GPT-2 Medium window loops)
landed on terminal tokens that read as a temporal/durative family: `until`,
`forever`, `since`, `simultaneously`, `halfway`. Stage 1's W_E permutation
test killed an identical-looking claim for GPT-2 Small (the "all-warm"
basin neighbourhood turned out to be an anisotropy artifact). This test
applies the same discipline: are the terminal token sets more related in
embedding space than matched random token sets?

## 2. Token sets under test

All sets use the **full uncurated inventories** from `results_full.json`
and `results_scan.json` (unique types only, unweighted by occupancy).

### Direct-decode sets

| Set | Source | Unique token IDs | Tokens | n |
|---|---|---|---|---|
| S1 | A4 direct (10→21, full) | {1566, 8097, 1201} | ` until`, ` forever`, ` since` | 3 |
| S2 | O8 direct (8→21, scan) | {11640, 19487} | ` simultaneously`, ` halfway` | 2 |
| S3 | A5 direct (8→15, full) | {30993} | ` rant` | 1 |
| S4 | Pooled word-arms direct (A4 ∪ O8 ∪ A5) | {1566, 8097, 1201, 11640, 19487, 30993} | all of above | 6 |
| S5 | A1 direct (0→11, full) — **contrast** | {11, 278} | `,`, `ing` | 2 |

### Via-tail sets (same arms, decode-via-tail readout)

| Set | Source | Tokens | n |
|---|---|---|---|
| S6 | A4 via-tail | ` until`, ` forever`, ` since` | 3 |
| S7 | O8 via-tail | ` simultaneously`, ` just`, `'` | 3 |
| S8 | A5 via-tail | ` endless`, `'` | 2 |
| S9 | Pooled word-arms via-tail (S6 ∪ S7 ∪ S8) | all of above | 7 |

Token IDs for via-tail sets resolved from the GPT-2 tokenizer at run
time; the token strings above are from `terminal_characterisation_full.json`
and `terminal_characterisation_scan.json` (`tail_decode_terminals` field).

### Degenerate-set handling

S3 (n=1) has zero pairs — no pairwise statistic is computable. It is
recorded as degenerate and excluded from the multiple-comparisons
correction. All other sets have ≥1 pair.

**Total testable sets: 8** (S1–S2, S4–S9; S3 excluded).
Bonferroni threshold: α = 0.05 / 8 = 0.00625.

## 3. Space

Both reported side by side:
- **W_E** — input embedding matrix, shape [50257, 1024]. The space where
  token identity is represented at layer 0.
- **W_U** — unembedding matrix, shape [1024, 50257] (transposed to
  [50257, 1024] for row-per-token). The space through which terminal
  tokens are *decoded* — arguably the natural space for this question,
  since the terminal is defined by its decode.

Justification: the issue is whether the tokens cluster in the geometry
the model uses, and the two matrices represent different aspects of that
geometry (input representation vs output decode). Stage 1's test used W_E
only; reporting both here avoids a post-hoc space choice.

## 4. Statistic

**Mean pairwise cosine similarity** among all (n choose 2) pairs of the
unique token types in the set, **unweighted** (each pair counts once,
regardless of how many prompts landed on each token). Rationale: occupancy
reflects loop dynamics, not token-space geometry.

## 5. Null distribution

**N = 10,000** random token sets of the same size as the observed set.

**Matching (the anisotropy correction):** for each token in the observed
set, define a matching pool from the full vocabulary. A candidate token
enters the pool iff it matches on ALL of:

1. **Leading-space status** — the BPE token string starts with a space
   (` until` → yes; `,` → no). This is the dominant structural axis in
   GPT-2's embedding space.
2. **Token string length** — within ±1 character of the observed token's
   BPE string length (in characters, including leading space if present).
3. **Log-frequency band** — same decile of BPE merge rank. Merge rank is
   approximated by token ID: lower ID = earlier merge = more frequent.
   The 50,257 tokens are binned into 10 equal-width bands of ~5,026 tokens
   each; the candidate must fall in the same band as the observed token.
   Special tokens (ID ≥ 50256) are pooled into the highest band.

To generate one null set: for each token in the observed set, sample one
replacement uniformly from its matching pool (without replacement within
the set, to avoid self-pairing). Compute the mean pairwise cosine of the
replacement set. Repeat N times. If any token's matching pool has fewer
than 50 members after all three filters, relax the length constraint to
±2 characters and log the relaxation. If still < 50, log a warning and
report the effective pool size — results for that set carry a
small-pool caveat.

**RNG seed:** 2026 (deterministic; recorded here before running).

## 6. Readout

Per set, per space (W_E and W_U):

- Observed mean pairwise cosine
- Null mean and SD of the mean-pairwise-cosine distribution
- Effect size: (observed − null mean) / null SD
- Empirical p-value: (count of null ≥ observed + 1) / (N + 1)
- Bonferroni-adjusted significance at α = 0.05 / 8 = 0.00625

Context statistic (once, per space): global mean pairwise cosine over
200,000 random token pairs (unmatched) — the raw anisotropy level.

## 7. Pre-registered readings

- **Observed ≫ null** (p < 0.00625 after Bonferroni, effect size > 2σ)
  for the word-arm sets → the relatedness pattern survives its first
  anisotropy control. This does NOT warrant the word "semantic" — that
  word stays quarantined. It means: the terminal tokens are more related
  than matched random tokens in the same structural class.

- **Observed ≈ null** (p ≥ 0.00625) for the word-arm sets → the apparent
  temporal/durative family is an anisotropy/matching artifact, the same
  phenomenon that killed Stage 1's all-warm claim wearing new tokens.
  The token-pattern thread closes; record plainly.

- **Contrast set (S5, A1 punctuation funnel) result is diagnostic but
  not load-bearing.** If it also shows significance, the test may be
  capturing something about ATR terminals generally, not about
  word-producing arms specifically. If it is null, it provides a within-
  experiment negative control.

Either outcome is a finding. No post-hoc additions to the set list.

## 8. Deliverables

1. This spec committed before any computation.
2. Script: `_STAGE2_JSPACE/experiments/exp_010c_windows/permutation_test.py`
3. Results: `_STAGE2_JSPACE/experiments/exp_010c_windows/output/permutation_results.json`
4. Dated section appended to `RESULTS_EXP010C.md` with the per-set table
   and the mechanical reading.
5. Planned-controls item 1 in RESULTS marked done with a pointer to this
   section.
