# EXP_010c-3b — Follow-up checks on the in-fill sweep (pre-registered spec)

**Status:** PRE-REGISTERED — recorded before any of these results exist.
**Created:** 2026-07-25
**Parent:** `EXP_010c3_SPEC.md` and its results section in
`experiments/exp_010c_windows/RESULTS_EXP010C.md` (2026-07-24).
**Issue:** #21. Ordered so that the checks capable of weakening the parent
result run first.

---

## 0. What is being stress-tested

The EXP_010c-3 record states: none of 9 in-fill cells produced whole-word
prompt-dependent terminals; 7 of 9 funnelled to a single prompt-independent
token; 9→21 is a fragment funnel, so 8→21 and 10→21 are **isolated single-layer
islands**; the whole-word via-tail-robust character held only at extract j=21.

Items 1 and 2 below can each undercut one of those statements. Their readings
are fixed here, before the numbers exist, so the outcome cannot be reshaped
afterwards. Items 3–5 close recorded gaps rather than test claims.

---

## 1. Are the single-token funnels a decode-geometry artifact? (no model time)

**Question.** Seven of nine in-fill cells settled on one token irrespective of
prompt. GPT-2 ties its unembedding to `wte`, so decoding is
`argmax_t  ln_final(v) · wte[t]`. A token with a large `wte` row norm, or one
well aligned with the direction typical residual states point in, can win that
argmax across a wide range of inputs. If the observed funnel tokens are such
tokens, then "settles on one token" partly describes the decoder, not the loop.

**Token sets (fixed here; no post-hoc additions).**

- `FUNNEL` (in-fill single/dominant terminals): `'oooooooo'` (I7, I9),
  `'<|endoftext|>'` (I11), `' GOP'` (X817), `"'d"` (X819), `' Fas'` (X1015),
  `' Bhar'` (X1017). Plus X1019's pair `'…)'`, `' […]'` reported separately.
- `WORD` (contrast — registered-run whole-word terminals): `' until'`,
  `' forever'`, `' since'`, `' simultaneously'`, `' halfway'`, `' rant'`.
- `VOCAB` — all 50257 tokens, the reference distribution.

**Statistics.**

- **S1 (primary) — random-direction argmax census.** Sample N = 10,000 random
  directions (isotropic Gaussian, unit-normalised, scaled to a typical residual
  norm), pass each through the real `ln_final` (γ, β from the state dict),
  decode with the tied unembedding, and tally the argmax token. Report each
  set's share of the 10,000, and the census top-20. Seed fixed and recorded.
  This asks directly: *what does the decoder say for a generic state?*
- **S2** — percentile of `‖wte[t]‖₂` within `VOCAB`.
- **S3** — percentile of `cos(wte[t], mean-row direction)` within `VOCAB`.

State dict only; no forward passes; no HookedTransformer.

**Pre-registered readings.**

| Observation | Reading |
|---|---|
| ≥3 of the 6 `FUNNEL` tokens each win ≥1% of random directions, **or** `FUNNEL` collectively wins ≥25% | The single-token funnels are **substantially a decode-geometry property**. Every single-terminal cell in the EXP_010c-3 record gets an annotation to that effect, and "7 of 9 funnel to one token" is restated as a statement about the readout as much as the dynamics. |
| `FUNNEL` collectively wins <5% **and** their S2 percentiles fall within the central 90% of `VOCAB` | **Not explained** by decode geometry; the funnels stand as reported. |
| Anything between | Recorded quantitatively as partial, with no verdict beyond the numbers. |
| `WORD` scores as high as `FUNNEL` on S1 | **The instrument does not discriminate** — S1 is uninformative here and is reported as such rather than read in either direction. This is a failure mode of the test, and it is recorded, not worked around. |

## 2. Robustness of the 9→21 refutation, and whether "seed variation" is a control at all

**2a — Seed determinism (design question; affects #11).** The loop appears
deterministic given prompt and weights: the model runs in `eval()` (no dropout),
the gated loop performs no sampling, and no stochastic operation appears in the
path. If so, `torch.manual_seed(...)` changes nothing and "seed variation" — a
registered control in #11 and a standing caveat in every results section — is
not a control.

Method: rerun one arm (I9) on a 3-prompt slice under seed 42 and seed 1234;
diff `terminal_token`, `terminal_token_id`, `lock_in_iter`, `n_iters`,
`final_cos_sim_mean`.

- Records identical → **seed variation is a no-op in this harness.** Record it
  plainly; the standing "single seed" caveat is rewritten as "single prompt
  subset"; #11 is flagged for amendment. A control that cannot vary anything
  should be corrected, not executed.
- Any difference → seed matters after all; the caveat stands as written and #11
  proceeds unchanged.

**2b — Disjoint prompt subset for the load-bearing cell.** H12's refutation
rests on one cell (9→21) under one 25-prompt subset. Arms: **I9 (9→21)** and
**A0 (0→23)** as reproduction anchor, on a **disjoint** 25-prompt subset derived
by the same deterministic round-robin rule at offset 25 (`derive_prompts.py`
gains an offset parameter; no hand-picking). 50 runs.

| Observation | Reading |
|---|---|
| I9 non-whole-word on the disjoint subset (single terminal, or plurality punctuation/fragment) | H12's refutation **survives** its first robustness check. |
| I9 yields ≥2 unique terminals with a whole-word plurality on the disjoint subset | **H12's refutation does not hold.** The isolated-islands claim is withdrawn in the record pending a wider sweep — stated plainly, not softened. |
| A0 fails to reproduce `D` on the disjoint subset | Reproduction gate failed on the new subset: **stop**, and record that before reading I9 at all. |

## 3. The whole-word scoring rule and the i=14 cell (no model time)

EXP_010c-3's mechanical flag (≥2 unique terminals AND plurality lexical class
whole-word) marks injection layers {8, 10, **14**}. i=14's plurality is `' or'`
— a function word — unlike the content-word pluralities at 8 and 10. The record
notes the difference in prose but hands the J-lens phase a map containing a
qualified flag.

**Decision, fixed here before recomputing: keep the rule unchanged (option a).**
Tightening a rule after seeing which cell it inconveniences is precisely the
bias the house rules exist to prevent, and a rule that is narrowed to exclude
one cell can no longer be called mechanical. Instead, a **content/function
column is added and applied uniformly to every cell** in the map, using this
fixed closed-class list, enumerated here so it cannot be tuned later:

```
a an the this that these those my your his her its our their
and or but nor so yet for
of in on at to from by with about into over under between through during
is are was were be been being am do does did have has had
will would shall should can could may might must
not no nor as if then than there here it he she they we you i
```

A terminal is `function` if its stripped, lowercased form appears in that list;
`content` otherwise. The column is reported for every flagged and unflagged
cell alike. No cell's flag changes as a result — the column is descriptive.

## 4. Extraction ladder above 21 at injection 8

For injection 10 the region above the peak is measured (10→22, 10→23, boundary
scan). For injection 8 it is not, so "j=21 is a sharp peak" currently rests, at
injection 8, on the rungs below 21 only.

**Arms:** `E822` (8→22), `E823` (8→23). Registered protocol, registered
25-prompt subset, 50 runs, plus the decode-via-tail control where j<23.

| Observation | Reading |
|---|---|
| Both degrade as injection 10's do (loss of the whole-word, via-tail-robust character) | j=21 is a sharp peak at **both** flagged injections; the ladder statement is symmetric and stands. |
| Either stays whole-word **and** via-tail-robust | The peak is **not** sharp at injection 8; the "only j=21" statement in the EXP_010c-3 record is narrowed to injection 10 by annotation. |

## 5. Settle time (recorded protocol variant)

Nearly every EXP_010c-3 arm locked at iteration 120 — the earliest reportable
value for `check_start=100`, `check_every=10`, `patience=3`. The states had
therefore stopped moving before observation began, and "120" is an upper bound,
not a measurement.

Method: arms I7, I9, X1017 on a 5-prompt slice with `check_start=10` (earliest
reportable lock = 30), everything else unchanged. **A recorded variant, never a
silent change to the registered path**; the registered artifacts are not
touched.

Reading: report the observed lock iteration per arm. If arms lock at the
earliest reportable value again, record that settle time is ≤30 iterations and
that the registered runs' lock values remain upper bounds — no further claim.

---

## Conventions

Observations-only register in `RESULTS_EXP010C.md` (numbers and checkable
classes; no interpretation, no curated sublists); interpretation to a session
note. Analysis scripts and artifacts committed. Existing sections appended to,
never rewritten — if item 1 or 2 undercuts a parent claim, the affected
statement is annotated by pointer.

## Definition of done

Spec committed before any result exists (this file); items 1–4 executed (5
optional); a dated results section stating mechanically which reading was
observed for each, including any that weaken EXP_010c-3; the tested-windows map
corrected if items 1–3 change what it should hand forward; the seed-determinism
answer recorded and #11 flagged if it is a no-op.

---

## Addendum 2026-07-25 — item 1 inputs stated operationally (PR #33 review)

**Clarification only; the executed analysis is unchanged.** §1 described the S1
sample as "scaled to a typical residual norm" and S3 as a cosine with the
"mean-row direction". Neither was operationally defined, so an independent rerun
could have used different inputs. As implemented in `analyze_funnel_geometry.py`
and executed:

- **Source of the vocabulary statistics:** the `wte` matrix of the local
  gpt2-medium state dict (`pytorch_model.bin`, 1,520,013,706 bytes — the
  S3-mirror file recorded in RESULTS §Model acquisition), all 50257 rows. GPT-2
  ties its unembedding to `wte`, so this is also the decode matrix. No prompts
  and no forward passes are involved in S1/S2/S3.
- **S1 sample:** N = 10,000 vectors drawn i.i.d. `torch.randn` at
  `torch.Generator().manual_seed(42)`, d = 1024, then passed through the real
  `ln_final` (γ, β from the same state dict, eps 1e-5) before the argmax.
- **"scaled to a typical residual norm" is a NO-OP and was not applied.**
  LayerNorm is scale-invariant, so the input norm cannot affect the argmax. The
  phrase should not be read as an unrecorded free parameter; recorded here
  rather than quietly dropped.
- **S3 "mean-row direction":** the arithmetic **mean** (not median) over all
  50257 `wte` rows, L2-normalised; each token's statistic is the cosine between
  its own row and that direction. Percentiles for S2 and S3 are computed against
  the full 50257-token distribution.
- **Reported values** are in `output/funnel_geometry.json`, which also records
  `n_directions`, `seed`, and `vocab` so the run is self-describing.

The post-hoc natural-decode diagnostic (labelled as such in the results) uses
the registered 25-prompt subset and one un-hooked forward pass per prompt,
reading `blocks.{j}.hook_resid_post` at j ∈ {15, 17, 19, 21}.
