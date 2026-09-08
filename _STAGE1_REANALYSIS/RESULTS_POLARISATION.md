# EXP_014 — Results Record

**Spec:** `POLARISATION_SPEC.md` (pre-registered, committed before any rank existed —
commit `2e8b697`). **Script:** `run_polarisation.py`. **Artifact:**
`polarisation_results.json`. **Register:** observations only.

---

## Reproduction gate: PASS (stated first)

The readout pipeline rebuilt here reproduces the committed top-20 **exactly** for all
five settled states *and* all five `baseline_iter0` states (10/10). All 48
pre-registered tokens resolved to single GPT-2 tokens; **none dropped**.

Gate defect found and fixed before measurement: the first iter-0 reconstruction
mismatched 5/5 because TransformerLens prepends BOS for GPT-2
(`cfg.default_prepend_bos`) and the HF tokenizer does not. Settled states matched
throughout — they are saved tensors and involve no tokenisation — so the gate
isolated the fault precisely. Recorded because an ungated version of this script
would have reported iter-0 ranks from the wrong state.

## Median ranks (of 50257), per pre-registered set

L = held-out socialist register · R = rival pole · N = political-neutral ·
C = non-political control. Lower = higher in the readout distribution.

| State | L | R | N | C |
|---|---|---|---|---|
| Lucier / iter0 | 15998 | 24004 | 18596 | **7660** |
| Lucier / settled | **70** | 1342 | 4734 | 15460 |
| Semantic / iter0 | 12900 | 19649 | 17654 | 17076 |
| Semantic / settled | **75** | 1470 | 4530 | 15380 |
| Nonsense / iter0 | 28806 | 36676 | 27722 | 6609 |
| Nonsense / settled | **72** | 1437 | 4529 | 15398 |
| Imperative / iter0 | 21990 | 25313 | 18722 | 14498 |
| Imperative / settled | **72** | 1450 | 4526 | 15390 |
| Syntactic (`Divine`) / iter0 | 16261 | 20794 | 7236 | 1234 |
| Syntactic (`Divine`) / settled | 18902 | 32327 | 30590 | 34060 |

## Observations (no interpretation in this section)

1. **The four basin states are numerically near-identical at settlement**
   (L medians 70/75/72/72; C medians 15460/15380/15398/15390) despite prompts
   that share no content: a Lucier text, a geography question, five nonsense
   syllables, and an arithmetic instruction.
2. **The ordering inverts between iter0 and settled.** At iter0 the
   non-political control C is the *best-ranked* set in 3 of 4 basin prompts; at
   settlement C is the *worst*, and in every basin prompt C's median rank is
   numerically worse than its own iter0 value (e.g. Lucier 7660 → 15460).
3. **Held-out same-pole vocabulary rises with the basin.** No L token appears in
   any state's observed top-20 (spec §2). Lucier settled, individual ranks:
   `solidarity` 24, `socialism` 25, `Engels` 27, `communist` 31, `communism` 36,
   `workers` 61, `Marxism` 79, `revolution` 106, `Trotsky` 393, `union` 452,
   `strike` 789, `collective` 801. The same tokens at iter0: 31412, 30531,
   33130, 8013, 17023, 7995, 26866, 5948, 14973, 11109, 13659, 29864.
4. **Two rival-pole tokens move in the opposite direction.** `Reagan`
   13170 → 37618 and `Thatcher` 27702 → 28553 (Lucier, iter0 → settled);
   `conservative` 5923 → 5227 and `conservatism` 23944 → 6959.
5. **The `Divine` basin shows none of it.** Every set is buried at settlement
   (L 18902, R 32327, N 30590, C 34060) and L does not improve from iter0
   (16261 → 18902). The effect is specific to this basin, not a property of
   settled ATR states in general.

## Which pre-registered reading obtained

**None of the six rows in spec §6 matches.** Recorded as a spec defect rather than
resolved to the nearest row.

Observed is a **monotone gradient**, L (70) ≪ R (1342) < N (4734) ≪ C (15460).
Row 1 required `R ≈ N ≈ C` and row 2 required `N ≈ C`; both fail — rival-pole and
neutral-political vocabulary are each clearly elevated above non-political
vocabulary. Rows 3–5 fail on their own terms (L ≉ R; L was not top-ranked at
iter0). Row 6 was not measurable (below).

The mechanical statement the table does support, scoped to the four sampled
12-token sets rather than to vocabulary classes at large: **the sampled L, R and
N sets all improve in rank from iter0 to settlement while the sampled C set gets
worse, and at settlement the sets order L (70) < R (1342) < N (4734) < C
(15460).** Rank positions are quoted rather than a ratio: median rank is ordinal,
so "19×" is not a meaningful multiple, and the R set is separately noted below as
contaminated.

## Post-hoc note (NOT a finding; flagged for its own pre-registration)

Inspection of the per-token ranks shows the R set is **contaminated**:
`capitalist` (60) and `capitalism` (577) are core *Marxist-discourse* terms, not
markers of the rival pole, and they carry R's median down. The uncontaminated
rival markers — `conservative` 5227, `conservatism` 6959, `Thatcher` 28553,
`Reagan` 37618 — sit far lower.

This was noticed after seeing the numbers and therefore **decides nothing here**.
It defines the obvious next pre-registration: an R set built from rival-pole
vocabulary that does not double as socialist-critique vocabulary.

## Not run (recorded, not silently omitted)

Spec §4 item 3 (all 15 noise trials) and spec §6 row 6 could **not** be measured:
`converged_tensors.pt` stores the five prompt states only, so no noise-trial
tensor exists to take a full distribution from. The committed top-20 lists for the
noise trials contain no socialist-register token (`―`, `Dig`, `ei`, `!?`, `Eva`;
trial_07 `trader`; trial_11 `Hindu`/`Bombay`), but that is a top-20 observation,
not the registered rank statistic. Reproducing the noise states requires re-running
the seeded loop.

## Addendum 2026-07-30 — rank is not alignment

Recorded because the cross-model work (`RESULTS_MEDIUM_SOCIALIST.md` §7b) makes
a misreading of this document available.

Everything measured here is **rank in the readout distribution**. It is not a
claim that the settled state points along a "socialist direction" in residual
space, and that stronger claim is not supported. Measured: the cosine between
`ln_f(Small's settled state)` and the centred socialist-embedding direction is
**0.0423**, roughly 2.4° off orthogonal, while the same state reaches basin rank
13 of 50,257. Rank 13 is achieved on a component that weak.

Corollary, measured the same way: averaging the socialist token embeddings does
**not** read out as socialist in either model (Small rank 88, Medium 186; both
top out at ` mathemat`, ` horizont`, ` neighb`). So the basin is neither a
centroid artefact nor reachable by embedding-space arithmetic.

The elevation reported above is real and gate-verified. What it is not is
evidence of geometric alignment with the vocabulary it elevates.

## Caveats (standing)

Single readout convention (`ln_final → W_U` at the last position — the Stage 1
convention, with its known mid-stack unreliability not at issue here since these
are final-layer states); one model (gpt2-small); 12 tokens per set; median rank is
a location statistic and no dispersion test was pre-registered; this experiment
measures the *readout* of the settled state and says nothing about why the pole is
present in the weights (spec §7 — corpus-provenance claims remain out of scope).
