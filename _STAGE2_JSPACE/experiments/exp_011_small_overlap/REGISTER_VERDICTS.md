# EXP_011: proposed register rows

These rows are proposed, not applied. This branch does not edit
`_STAGE2_JSPACE/REGISTER.md`; the orchestrator applies every experiment's rows in
one sweep. Each row is written in the register's own column format so it can
replace the existing row without further editing, and each hypothesis statement
is copied character for character from the register, because a verdict is
scored on the registered wording and this experiment restated nothing.

The register's hypothesis table columns are: identifier, owning experiment,
statement in one line, verdict, recorded at. Its experiment table columns are:
identifier, what it is, status, spec, results.

## Section 1, hypothesis register: replace these four rows

| ID | Owning experiment | Statement (one line) | Verdict | Recorded at |
|---|---|---|---|---|
| H6 | EXP_011 | GPT-2 Small's five basin tensors project significantly more onto the J-space than the 18 null-model basins | **NOT SUPPORTED** (the five basins beat the eighteen at three of the six band layers, one short of the pre-registered majority of four, and lose at one. Scored on the registered wording; the 18-basin arm it names is the mis-scaled original noise arm of lucier finding F4, which ran at 28 percent of the language arm's injection strength, so this verdict is not evidence about language against noise at matched strength. H16 carries that question) | `experiments/exp_011_small_overlap/RESULTS_EXP011.md` |
| H16 | EXP_011 | Corrected null: on the Neuronpedia pre-fitted gpt2-small lens, the J-space share (sparse nonnegative decomposition onto at most 25 lens vectors) of GPT-2 Small's language-prompt terminal states exceeds that of the matched-ν noise terminal states (Stage 1 run 17) at the workspace-band layers, above the norm-matched random-dictionary chance level | **NOT SUPPORTED** (language and matched-strength noise terminals are indistinguishable: median shares differ by -0.0029 to +0.0002 on shares of 0.014 to 0.022, no band layer significant in the hypothesised direction. The registered chance level, a norm-matched random dictionary, is also shown in the record to be an unfair comparison, because it lacks the lens's directional clustering; against a rotated-lens control the language terminals sit above chance at five of six band layers. See decision item 1 of the results record) | `experiments/exp_011_small_overlap/RESULTS_EXP011.md` |
| H16a | EXP_011 | The pilot's prediction re-tested on the full-vocabulary lens, phase-aware: the `prolet` attractor's J-space share exceeds the `Divine` cycle's share in both phases (the pilot found the reverse at pilot confidence, finding F16 in the lucier record) | **NOT SUPPORTED** (prolet beats phase A at three of six band layers and phase B at one of six, where four of six against each was required. The phases straddle prolet, as finding F16 describes, but in the opposite direction: on the full-vocabulary lens phase B is the more lens-expressible phase and phase A the less, which retracts F16's phase assignment) | `experiments/exp_011_small_overlap/RESULTS_EXP011.md` |
| H16b | EXP_011 | ATR terminal states have a lower J-space share than ordinary non-iterated prompt residuals read at the same layer on the same lens (the loop leaves the verbalizable directions) | **SUPPORTED** (supported at five of the six band layers with permutation p of 0.0032 once and 0.0001 four times, on 64 to 96 percent of the 125 paired prompts. Holds for last-position residuals, the reading the loop itself uses; the pre-registered position-averaged secondary runs the other way at layers 5 to 8 and is reported in the record) | `experiments/exp_011_small_overlap/RESULTS_EXP011.md` |

## Section 2, experiment register: replace this row

| ID | What it is | Status | Spec | Results |
|---|---|---|---|---|
| EXP_011 | J-space overlap, GPT-2 Small (H6, with the corrected-null and phase-aware sub-hypotheses H16, H16a, H16b). Instrument: the Neuronpedia pre-fitted gpt2-small lens, permitted by TC on 2026-09-05 (erratum (f)) | **COMPLETE** (2026-09-05) | `EXP_011_SPEC.md` | `experiments/exp_011_small_overlap/RESULTS_EXP011.md` |

## Note for whoever applies these

H6's existing owning-experiment cell reads "EXP_011 (planned)". The rows above
drop the parenthetical, because the experiment has now run. Nothing else in the
register is touched by this experiment.
