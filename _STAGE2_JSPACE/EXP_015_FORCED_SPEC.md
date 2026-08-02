# EXP_015-FORCED: can a natural-strength arm be engineered to reach the `D` readout? (pre-registered spec)

**Status:** PRE-REGISTERED. This spec was written and committed before any
run output for this experiment existed.
**Created:** 2026-08-02. Chartered by issue #72 part 1 under TC's in-session
direction of 2026-08-02. This is the follow-up control that the EXP_015 spec
(section 7) and the EXP_015 verdict both name as the registered next step: a
natural-strength arm engineered so the runs still produce the `D` readout,
separating injection strength from settledness before the H15 refutation is
treated as decisive against the apparatus-mask hypothesis.
**Identifiers:** EXP_015-FORCED, allocated in `REGISTER.md` in the same
commit as this file, per the register's rule. No new hypothesis number is
allocated: this experiment tests no new hypothesis; it disambiguates the
interpretation of the already-recorded H15 refutation.
**Compute:** model runs on gpt2-medium (CPU), bounded by the budget in
section 8.

---

## 1. The question in one sentence

Can the medium model's full-stack loop, run at natural injection strength
(the strength at which EXP_015 found the Small-resemblance gone), be
engineered so that its runs still reach the `D` readout, the single-token
collapse observed under the loud convention, and if so, does the engineered
arm's prompt grouping resemble GPT-2 Small's the way the loud arm's did?

## 2. Background in plain terms

This project repeatedly feeds a language model's internal activity back
into itself. Under the registered loud convention the fed-back signal at
the medium model's first layer is about 218 times its natural strength, and
every one of the 25 test prompts settles into a state whose readout is the
single token `D` (token id 35). At natural strength (EXP_010c-VARIANTS
Control B) the collapse disappears: 0 of 25 runs settle within the
1,000-iteration cap and the `D` readout is absent. EXP_015 then found that
the Small-like prompt grouping of the loud end states (agreement 0.200 on
the adjusted Rand index scale, where 0 is chance and 1 is identical
grouping, permutation p 0.0009) does not survive at natural strength
(agreement minus 0.113, permutation p 1.000). H15 was REFUTED.

Three things varied together in that test: injection strength, the
suspected apparatus mask (the `D` readout), and settledness (25 of 25
settle loud, 0 of 25 settle natural). This experiment engineers arms that
hold injection strength at natural while attempting to restore the `D`
readout, so that the three can begin to be separated.

One measured fact motivates the first attempt, established from the
committed loud artifact before this spec was written: the loud terminal
states are fully position-collapsed. For every one of the 25 stored A0
terminals in `output/terminals_full.pt`, the cosine between the mean
vector (average over token positions) and the last-position vector is
1.000 to float precision, so a single stored terminal vector tiled across
the prompt's positions reconstructs the loud terminal tensor faithfully at
the level this experiment uses it.

## 3. Inputs, frozen

All paths are relative to `experiments/exp_010c_windows/`.

| Input | What it holds |
|---|---|
| gpt2-medium weights (local files) | The model. The three statistical inputs of record are digest-checked: `pytorch_model.bin`, `vocab.json`, `merges.txt` must match the SHA-256 digests in the `EXP_010c_PERM_SPEC.md` post-run addendum. |
| `output/prompt_subset.json` | The registered 25-prompt subset, in its registered order. The 5-prompt probe subset used by attempts 2 and 3 is the first 5 prompts of this order (E01_politics, D01_water, A01_physics, B01_napoleon, C01_jack_jill): deterministic, no hand-picking. |
| `output/terminals_full.pt` (arm A0 entries) | The registered loud-run A0 terminal vectors (25 prompts). Attempt 1's seed states come from here. Also feeds the comparison machinery's reproduction gate. |
| `output/terminals_small_010d.pt` | GPT-2 Small's reference end states (EXP_010d), the comparison reference. |
| `output/natural_resid_norms_energynorm_A0.json` | The recorded natural layer-0 norms per prompt (EXP_010c-VARIANTS Control B). Cross-check target for the norms recomputed in-run. |
| `output/results_full.json`, `output/results_energynorm_A0.json` | The loud and natural A0 records of record. Machinery-gate comparison targets. |

## 4. Protocol common to all attempts

The loop is the registered gated protocol of `atr_engine2.run_atr_gated`,
window A0 (inject at layer 0, read at layer 23), with the registered gate:
lag-1 mean-vector cosine above 0.999 held for 3 consecutive checks,
checked every 10 iterations starting at iteration 100. Global torch seed
42 (recorded in EXP_010c-3b as a no-op for this deterministic harness).
Natural injection strength means the loop tensor is rescaled each
iteration to the prompt's natural residual-stream norm at layer 0, exactly
the Control B `natural_i` convention. The `D` readout means the run's
endpoint readout token (the argmax decode of the final last-position
vector) is token id 35, the token `D`. Endpoint means the state at lock-in
if the gate fires, otherwise the state at the attempt's iteration cap;
whether the run locked is recorded separately and reported alongside.

New code is confined to one script, `exp015_forced.py`, in the experiment
directory. Attempts 1 and 3 need loop features the engine does not expose
(a seeded initial state; a non-natural rescale target), so the script
carries a local copy of the `run_atr_gated` loop body with exactly those
two extensions; `atr_engine2.py` itself is not modified. Two machinery
gates run before any attempt is read, and each is a STOP condition for the
attempts that depend on the local loop:

- **Gate G1 (loud replication).** The local loop, in the registered loud
  configuration (seed from the prompt pass, rescale to the seed norm at
  layer 23), on the first 2 prompts of the registered subset, cap 300,
  must reproduce the committed loud A0 record for those prompts:
  terminal token `D`, token id 35, lock-in at iteration 120, and cosine
  above 0.9999 between the local terminal mean vector and the committed
  one in `output/terminals_full.pt`.
- **Gate G2 (engine equivalence at natural strength).** On the first 2
  prompts, the local loop in the natural configuration must match
  `atr_engine2.run_atr_gated(renorm="natural_i")` run at cap 60 with
  check_start 20 (a short, gate-free horizon): maximum elementwise
  absolute difference of the two terminal mean vectors at most 1e-5
  (expected 0, identical operations in identical order).
- **Norm cross-check.** The natural layer-0 norms recomputed in-run must
  match `output/natural_resid_norms_energynorm_A0.json` at the 4 decimal
  places that file records, for all prompts used.

## 5. The attempt menu (bounded; nothing outside it runs)

Attempts run in order. Attempt 2 runs only if attempt 1 fails its
criterion; attempt 3 runs only if attempts 1 and 2 both fail.

**Attempt 1, FORCED-SEED (25 prompts, cap 300).** For each registered
prompt: take the committed loud-run A0 terminal last-position vector for
that prompt from `output/terminals_full.pt`, tile it across the prompt's
token positions (faithful per the position-collapse measurement in
section 2), rescale to the prompt's natural layer-0 norm, and use that as
the loop's initial state instead of the prompt-pass seed. Then iterate at
natural strength under the registered gate, cap 300. The cap rationale:
the loud runs all locked at iteration 120, the earliest lock the gate
permits, and a seeded run that is a fixed point of the natural-strength
dynamics locks there too; 300 gives 2.5 times that headroom, and a run
that has not locked by 300 has demonstrably left the seeded state.
**Criterion:** at least 13 of 25 runs (a majority) end with the `D`
readout.

**Attempt 2, EXTENDED-CAP (5-prompt probe, cap 2,000).** The natural
Control B configuration unchanged (prompt-pass seed, natural strength),
run on the 5-prompt probe subset with the iteration cap doubled from the
registered 1,000 to 2,000, registered gate otherwise. This asks whether
the natural runs reach `D` given more time. This attempt calls
`atr_engine2.run_atr_gated` directly. **Criterion:** at least 3 of 5
probe runs end with the `D` readout. A pass on the probe establishes
engineerability but leaves no 25-prompt terminal set, so the partition
comparison of section 6 cannot run from it within this budget; that
follow-up would need its own charter, and this spec records the
limitation in advance.

**Attempt 3, MID-BAND (5-prompt probe, cap 1,000; diagnostic only).**
Prompt-pass seed, rescale target the geometric mean of the prompt's
natural layer-0 norm and its loud seed norm at layer 23 (roughly 15 times
natural, against the loud convention's roughly 218 times), registered
gate, cap 1,000, on the 5-prompt probe subset. This attempt is not at
natural strength, so it can never fire reading (1); it is registered as a
bracketing diagnostic: if `D` returns at mid-band strength but not in
attempts 1 and 2, the loudness needed for the collapse is bounded between
natural and roughly 15 times natural, which sharpens the
loudness-necessity reading. **Criterion (diagnostic):** at least 3 of 5
probe runs end with the `D` readout.

## 6. The comparison, run only on an attempt-1 pass

If attempt 1 passes its criterion, the engineered arm's 25 terminal mean
vectors are compared against the Small reference with the EXP_015
machinery, unchanged: greedy leader clustering on cosine similarity at
threshold 0.999 applied to each side separately, adjusted Rand index
between the two label vectors, and the 10,000-shuffle permutation null
with seed 42 and the add-one correction. The EXP_015 reproduction gate
runs first (loud A0 versus Small must reproduce agreement 0.200,
permutation p 0.0009; STOP on failure), and the EXP_015
degenerate-partition guard applies (a 1-basin or 25-basin engineered
partition is recorded as unanswerable at the registered threshold). The
descriptive threshold sweep (0.99, 0.995, 0.999, 0.9995) is reported,
verdict read at 0.999 alone.

**Pre-registered decision rule for the comparison:** adjusted Rand index
greater than 0 with one-sided permutation p below 0.05 at threshold 0.999
means the resemblance is recoverable at natural strength once the runs
are back in the settled `D` state, so settledness, not injection strength
itself, carried the EXP_015 null. Permutation p at or above 0.05, or
index at or below 0, means even a settled natural-strength `D` arm lacks
the resemblance, so loudness itself carried it.

## 7. Pre-registered readings

| Outcome | Reading |
|---|---|
| (1) An engineered natural-strength arm (attempt 1, or attempt 2's configuration if later run at 25 prompts) reaches the `D` readout on a majority of prompts | Its terminal partition is compared against the Small reference with the EXP_015 machinery (adjusted Rand index, 10,000-shuffle permutation null, threshold 0.999), and that comparison decides whether settledness or loudness carried the EXP_015 null, per the decision rule in section 6. |
| (2) No attempt in the menu reaches `D` | Recorded as "not engineerable within the registered budget", which strengthens the loudness-necessity reading of the collapse: the `D` state is not merely unreached from prompt seeds at natural strength, it is not sustainable there even when the loop is started inside it. |

Either way the outcome is recorded. A mid-band-only success (attempt 3
passes, attempts 1 and 2 fail) falls under reading (2) with the bracketing
note from section 5 recorded alongside.

## 8. Budget

The menu above is the entire registered budget: at most one execution of
each attempt at the caps and prompt counts stated, plus the machinery
gates (about 1,000 loop iterations) and the analysis-only comparison.
Worst case is about 23,000 loop iterations, roughly 7 hours at the
calibrated 1 second per iteration on this container's CPU (a timing
calibration of the unmodified loop on one prompt preceded this spec; its
only recorded quantity is seconds per iteration). If wall-clock time
exceeds 12 hours before the menu completes, the remaining attempts are
recorded as not executed within budget and reading (2) is evaluated on
the attempts that did run.

## 9. Outputs

- `experiments/exp_010c_windows/output/exp015_forced_attempt<k>.json` and
  `output/terminals_exp015_forced_attempt<k>.pt` per executed attempt,
  with `output/exp015_forced_attempt<k>.log` as the run log (R5).
- `experiments/exp_010c_windows/output/exp015_forced_ari.json` and its
  log, if the comparison runs.
- A new dated section in
  `experiments/exp_010c_windows/RESULTS_EXP015.md`, written whichever way
  the result comes out.
- The EXP_015-FORCED register row updated with the outcome in the results
  commit.

## 10. Deviations

Recorded in `RESULTS_EXP015.md` as they occur, per the folder rule.
