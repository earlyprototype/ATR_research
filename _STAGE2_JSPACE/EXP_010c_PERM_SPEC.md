# EXP_010c-PERM — Anisotropy-Corrected Permutation Test on Window-Loop Terminals (pre-registered spec)

**Status:** PRE-REGISTERED — recorded and committed before any statistic is
computed. Issue: earlyprototype/ATR_research#7. This spec commits the exact
token sets, embedding space, statistic, matching criteria, N, seed, threshold,
and readings. The analysis script (`experiments/exp_010c_windows/permutation_test.py`)
and results are committed after, in a separate commit.
**Created:** 2026-07-25
**Parent:** `EXP_010c_SPEC.md` / `EXP_010c2_SPEC.md`; executes item 1 of
RESULTS_EXP010C.md §"Planned controls — next observations."
**Method precedent:** Stage 1 `02b_permutation_test.py` (fold repo) — the
matched-null W_E test that reclassified the GPT-2 Small "all-warm
neighbourhoods" reading as an anisotropy artifact (decision log 2026-07-11,
`STAGE2_OUTLINE.md`).

---

## 1. Question

Are the terminal-token sets produced by the EXP_010c / 010c-2 window loops
closer to each other in embedding space than matched random token sets of the
same size, under an anisotropy-corrected null? Zero model forward passes —
pure analysis of committed artifacts plus the embedding matrix.

## 2. Inputs (all committed, read-only)

- `experiments/exp_010c_windows/output/results_full.json`,
  `results_scan.json` — `terminal_token` / `terminal_token_id` per record
  (full uncurated inventories; no hand-picked sublists).
- `experiments/exp_010c_windows/output/terminal_characterisation_full.json`,
  `terminal_characterisation_scan.json` — `decode_terminals` and
  `tail_decode_terminals` per arm (via-tail readout).
- gpt2-medium weights: `pytorch_model.bin`, `vocab.json`, `merges.txt`,
  `config.json` fetched from `huggingface.co/gpt2-medium/resolve/main/`
  (allowed since the 2026-07-25 network-policy update recorded in
  `REMOTE_ENV_MODEL_ACCESS.md`). `pytorch_model.bin` is 1,520,013,706 bytes —
  byte-identical size to the legacy-S3-mirror artifact recorded in
  RESULTS_EXP010C.md §"Model acquisition" (316 tensors, wte [50257, 1024]).

## 3. Token sets under test (exact, pre-committed — no post-hoc additions)

Sets are **unique token types** (unweighted — basin occupancy reflects
dynamics, not token identity). Ids cross-checked against
`results_full.json`/`results_scan.json` `terminal_token_id` where the type
appears as a direct-decode terminal. Word-producing arms per the committed
record: A4 (10→21), O8 (8→21), A5 (8→15). Contrast: A1 (0→11), the
punctuation-funnel arm with ≥2 unique types in both readouts (O0/O4 are
single-type funnels — a pairwise statistic is undefined there).

| # | Set | Token types (id) | n |
|---|---|---|---|
| 1 | A4_direct | `' until'` (1566), `' forever'` (8097), `' since'` (1201) | 3 |
| 2 | O8_direct | `' simultaneously'` (11640), `' halfway'` (19487) | 2 |
| 3 | A5_direct | `' rant'` (30993) | 1 — **singleton: statistic undefined; reported as N/A, not tested** |
| 4 | A4_tail | `' until'` (1566), `' forever'` (8097), `' since'` (1201) | 3 |
| 5 | O8_tail | `' simultaneously'` (11640), `' just'` (655), `"'"` (6) | 3 |
| 6 | A5_tail | `' endless'` (13079), `"'"` (6) | 2 |
| 7 | pooled_word_direct (union 1∪2∪3) | 1566, 8097, 1201, 11640, 19487, 30993 | 6 |
| 8 | pooled_word_tail (union 4∪5∪6) | 1566, 8097, 1201, 11640, 655, 6, 13079 | 7 |
| 9 | A1_direct (contrast) | `','` (11), `'ing'` (278) | 2 |
| 10 | A1_tail (contrast) | `'.'` (13), `' the'` (262) | 2 |

Recorded in advance: sets 1 and 4 contain the identical type set (direct and
via-tail readouts of A4 agree on types). Both are run and reported; their null
draws are independent (per-set RNG substream), so their null statistics may
differ slightly. Testable sets: **9** (all except set 3).

## 4. Space

Report **W_E rows** (input embedding, `wte.weight` [50257, 1024]) and **W_U
columns** (unembedding) **side by side**. Recorded before running: the
gpt2-medium checkpoint contains no separate `lm_head` weight (verified on the
downloaded state dict — 316 tensors, no `lm_head` key); GPT-2 ties the
unembedding to the input embedding, **W_U = wte^T**, so W_U's column for token
t is exactly W_E's row for token t and the two columns of the report will be
numerically identical. Both are still printed, with this note, so the
side-by-side requirement is discharged honestly rather than silently
collapsed. Consequence for multiple comparisons: the two spaces constitute
**one** test per set, not two.

## 5. Statistic

Unweighted mean pairwise cosine similarity over the set's unique types:
mean over all C(n,2) unordered pairs of cos(v_i, v_j), vectors taken raw from
the embedding matrix (no centering, no normalisation beyond the cosine
itself). For n=2 this is a single pairwise cosine (recorded limitation:
maximally noisy; the null accounts for it since null sets have the same n).

## 6. Null (the anisotropy correction)

**N = 10,000** random token sets per tested set. Each null set is drawn by
replacing every observed type with a random vocab token **matched** on the
three properties that drive baseline cosine similarity in GPT-2 embedding
space:

1. **Leading-space status:** vocab key starts with `Ġ` — must match exactly.
2. **Token string length:** length of the byte-decoded token string
   (leading space counted as one character), matched **±1 character**.
3. **BPE merge-rank band** (frequency proxy): rank = the line index
   (0-based, header excluded) of the *first* merge in `merges.txt` whose
   concatenated result equals the token's vocab key; `merges.txt` has 50,000
   merges. **Banding rule: band = rank // 5000** (ten bands of 5,000 ranks).
   The 256 byte-level tokens (never a merge result) form their own band
   `"byte"`. `<|endoftext|>` (50256, also rank-less) is **excluded** from all
   candidate pools (special token, not a member of the written-text vocabulary).

Sampling: within each null replicate, tokens are drawn uniformly from each
observed type's candidate pool, **without replacement across the replicate**
(null sets are sets of distinct types, like the observed sets). The observed
token itself is not excluded from its pool (unbiased null). Candidate pools
verified non-empty in advance; smallest pool is `' simultaneously'`
(space=True, len 15±1, band 2): 43 candidates.

**Seed:** 20260725 (`numpy.random.default_rng`), with one deterministic
per-set substream derived as `default_rng([20260725, set_index])` using the
set numbering of §3. Fully deterministic re-run.

## 7. Readout and threshold

Per tested set, report: n, observed mean pairwise cosine, null mean, null SD,
empirical p (one-sided, upper tail):
p = (1 + #{null ≥ observed}) / (N + 1), and effect size
z = (observed − null mean) / null SD. Every set in §3 is reported, including
the contrast sets, the singleton N/A, and any null results.

**Multiple comparisons:** Bonferroni across all 9 testable sets (one
effective space per §4): **α\* = 0.05 / 9 ≈ 0.00556**. A set "passes" iff
p < α\*.

## 8. Pre-registered readings (written before running)

| Observation | Reading |
|---|---|
| Word-arm sets (1, 2, 4–8) with p < α\* and z > 0 | The relatedness pattern among window-loop terminals **survives its first control** — it exceeds matched-baseline anisotropy. Still not "semantic" (that word stays quarantined); the token-pattern thread stays open for the J-lens phase. |
| Word-arm sets with p ≥ α\* | The apparent family is **consistent with baseline anisotropy** at matched size/space/length/frequency — an artifact of the same class as Stage 1's all-warm claim. Recorded plainly; the token-pattern thread closes the way that one did. |
| Mixed outcome (some word-arm sets pass, others do not) | Reported per set, mechanically; no aggregate claim. Pooled sets (7, 8) do not override per-arm results, nor vice versa — each row stands alone. |
| Contrast sets (9, 10) pass | The statistic/null is anti-conservative at n=2 or the funnel terminals are themselves atypically related — either way, word-arm passes at n=2 (set 2, 6) are downgraded to "same evidence class as the contrast" in the results section. |
| Contrast sets do not pass | No calibration alarm; word-arm rows stand as reported. |

Either outcome on the word arms is a finding. No result from this test is a
statement about *why* any relatedness exists.
