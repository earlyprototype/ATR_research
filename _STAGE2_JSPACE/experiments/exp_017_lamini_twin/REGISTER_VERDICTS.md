# Proposed register rows for EXP_017

**Purpose.** This session does not edit `_STAGE2_JSPACE/REGISTER.md`, which is
the authoritative identifier register. The rows below are written in the
register's own format for the orchestrator to sweep into it. They replace the
existing `UNTESTED (allocated 2026-09-05)` rows for H18, H18a and H18b, and the
`IN PROGRESS` status on the EXP_017 row.

**Evidence.** `RESULTS_EXP017.md` in this directory, and the JSON artifacts in
`output/` that every number in it is generated from. Spec:
`_STAGE2_JSPACE/EXP_017_SPEC.md`, committed at `f01b4af` before any run
carrying verdict weight. Tracker issue: #80. Branch:
`claude/latent-context-small-llms-u2jdig-exp017`.

---

## 1. Hypothesis register rows

Format as in `REGISTER.md` section 1:
`| ID | Owning experiment | Statement (one line) | Verdict | Recorded at |`

| ID | Owning experiment | Statement (one line) | Verdict | Recorded at |
|---|---|---|---|---|
| H18 | EXP_017 | The post-trained twin LaMini-GPT-124M, run under the registered full-stack convention on the 25-prompt Small subset, partitions the prompts like base GPT-2 Small's terminals (adjusted Rand index above chance under the EXP_010d permutation test) | **SUPPORTED** (adjusted Rand index 0.1694, permutation p 0.0097, against a matched base arm re-run under the same lag-2 gate; above chance at three of the four sweep thresholds) | `experiments/exp_017_lamini_twin/RESULTS_EXP017.md` §2.5 |
| H18a | EXP_017 | The twin's terminal readout tokens coincide with base GPT-2 Small's basin tokens on at least half of the 25 prompts | **REFUTED** (0 of 25 against Stage 1's five basin tokens, and 0 of 25 against the tokens base produced in this run's own matched arm; the twin reads ` anarchism` on 24 of 25 prompts and ` instant` on 1) | `RESULTS_EXP017.md` §2.6 |
| H18b | EXP_017 | Post-training changes the terminal states' J-space share: the twin's share on a lens fitted to the twin differs from base's share on the Neuronpedia lens by more than the random-dictionary control spread, two-sided | **SUPPORTED at exactly the pre-registered threshold** (4 of the 6 band layers 5 to 10 meet both conditions, and 4 was the bar; permutation p 0.0001 at every layer; the same-lens cross-check, which does not depend on the threshold, has the twin above base at all 11 layers by 0.0167 to 0.0889). **Read with two caveats.** §3.5: base's settled states lie no closer to the lens's nameable directions than to a randomly rotated copy of them from layer 4 onward, so the absolute share level at the band carries no evidence of verbalizable content. §3.6, added 2026-09-06: every number above was measured in the TransformerLens coordinate convention while the lens matrices were fitted in the Hugging Face one, and the recomputation in the fitting convention keeps the verdict (SUPPORTED, 4 of 6 band layers, on layers 5, 6, 7 and 8 rather than 6, 7, 9 and 10) but reverses the cross-check (the twin above base in 4 of the 22 same-lens comparisons rather than all 22, and the model effect 0.1 to 0.5 times the instrument effect rather than 3.1 to 12.7 times). Note 5 below carries the numbers and the ruling this needs | `RESULTS_EXP017.md` §3.3 to §3.6 and §5 |

## 2. Experiment register row

Format as in `REGISTER.md` section 2:
`| ID | What it is | Status | Spec | Results |`

| ID | What it is | Status | Spec | Results |
|---|---|---|---|---|
| EXP_017 | Post-trained twin: LaMini-GPT-124M under the Stage 1 full-stack loop and the J-space overlap probe, base against post-trained (H18, H18a, H18b) | **COMPLETE** (2026-09-05) | `EXP_017_SPEC.md` | `experiments/exp_017_lamini_twin/RESULTS_EXP017.md` |

## 3. Notes the orchestrator may want when sweeping

1. **The gate differs from EXP_010b's registered arm SB.** This experiment ran
   the lag-2 gate, which compares an iteration with the one two steps earlier,
   on the task brief's instruction. EXP_010b's arm SB ran the lag-1 gate. Base
   GPT-2 Small was therefore re-run here under the lag-2 gate so that H18's
   comparison is like for like, and the committed lag-1 terminals give a
   secondary reading. Whether the lag-2 gate should be registered as a named
   convention is raised as a decision item for TC in `RESULTS_EXP017.md` §6.3
   item 1; this file proposes no register change on that point, because rule R8
   reserves it.
2. **Three unasked-for reproduction checks passed** and are recorded in
   `RESULTS_EXP017.md` §2.4: base's terminal word agrees with the committed
   `results_small010b_SB.json` on 25 of 25 prompts, base's entry loudness ratio
   reproduces EXP_010b's committed 73.0 times natural with range 56.23 to 88.42
   against the recorded 56.2 to 88.4, and base's grouping agrees with EXP_010d's
   committed Small reference at 0.9805 on the 20 prompts the two subsets share.
   None of these needs a register row; they are noted so a reviewer can find
   them.
3. **No new identifier was created.** EXP_017, H18, H18a and H18b are the
   identifiers erratum (f) allocated on 2026-09-05, and this experiment used
   only those, plus existing identifiers in citations.
4. **Deviations** are listed flat in `RESULTS_EXP017.md` §4, and there are now
   twenty-two. The one a register reader is most likely to want is that the
   twin's Jacobian lens was fitted on 40 WikiText-103 prompts, below the 50 the
   spec named as a deviation threshold and well below the 277 of the
   Neuronpedia lens it is compared against, by the mechanical application of
   the spec's own budget rule. The sensitivity check in §3.4 shows the H18b
   verdict is unchanged when that lens is degraded to 5 prompts.

5. **A coordinate-convention question that needs a ruling before this row is
   swept, added 2026-09-06 after a review of the pull request.** The J-space
   share of a state is the fraction of it that can be built from at most 25
   vocabulary directions, and those directions are `W_U^T J_l`, so the answer
   depends on which version of the unembedding matrix `W_U` is used. EXP_017
   used the version TransformerLens produces when it converts a model, which
   folds the model's final normalisation gain into every direction and then
   subtracts a common vector; the lens matrices `J_l` were fitted against the
   unconverted Hugging Face model, whose version does neither. The two
   unembeddings differ by 68 percent of the size of the Hugging Face matrix, on
   a scale where 0 means identical. **The numbers, so the register does not
   have to take this on trust.** Median share at the six band layers, on a
   scale from 0 to 1: twin 0.2851 to 0.3221 and base 0.2538 to 0.2964 in the
   convention this run used, against twin 0.0087 to 0.0396 and base 0.0150 to
   0.0217 in the fitting convention. Both give H18b SUPPORTED at 4 of the 6
   band layers, the pre-registered bar, on layers 6, 7, 9 and 10 in the first
   and layers 5, 6, 7 and 8 in the second. The same-lens cross-check quoted in
   the H18b row above has the twin above base in all 22 comparisons by 0.0167
   to 0.0889 in the first convention, and in 4 of 22 by at most 0.0122 in the
   second. Artifacts: `output/exp017_jspace.json`, regenerated on 2026-09-06
   with every result field identical to the version first committed and the
   provenance stamps added, as deviation 22 records, and
   `output/exp017_jspace_hfframe.json`, with the frame measurement itself in
   `output/frame_check.json`. **This session changed no verdict and proposes
   none:** the row above still carries the numbers this run measured, and
   `RESULTS_EXP017.md` §6.3 item 5 asks TC which convention is the registered
   one for J-space work, noting that EXP_011 built its dictionary from the
   Hugging Face version and that H19b in EXP_018 will inherit whichever is
   settled. Rule R8 reserves that choice.

6. **A wording point for whoever sweeps this row.** If TC rules for the Hugging
   Face convention, the H18b row's parenthetical "the same-lens cross-check,
   which does not depend on the threshold, has the twin above base at all 11
   layers by 0.0167 to 0.0889" must not be swept as written, because it
   holds only in the convention this run measured. The verdict word
   itself, SUPPORTED at exactly the pre-registered threshold, stands
   either way.

7. **A number in this file was wrong and is corrected, 2026-09-06.** Until
   today the H18b row above gave the same-lens cross-check as "0.015 to
   0.086". The artifact `output/exp017_jspace.json` and
   `RESULTS_EXP017.md` sections 3.4 and 5 give 0.0167 to 0.0889, which is
   the smallest and largest of the 22 differences of medians, on a scale
   from 0 to 1. The row now reads 0.0167 to 0.0889 and nothing else in it
   changed. **Withdrawn by name: the range "0.015 to 0.086" that this file
   carried until 2026-09-06.** A reviewer found it, and in response every
   other number in the rows and notes above was checked back to the
   artifact that produced it rather than to the record: the adjusted Rand
   index 0.1694 and its permutation p 0.0097, the four sweep thresholds of
   which three have a permutation p below 0.05 (0.15538 at 0.99, 0.0232 at
   0.995, 0.0097 at 0.999 and 0.0046 at 0.9995), the twin's 24 readings of
   ` anarchism` and 1 of ` instant`, the two H18a readings of 0 hits out of
   25, the four band layers 6, 7, 9 and 10 meeting both conditions, the
   permutation p of 0.0001 at every layer and in all 22 same-lens
   comparisons, base's loudness ratio of 73.0 times natural with a range of
   56.23 to 88.42, the agreement of 0.9805 with EXP_010d on the 20 shared
   prompts, the 40 prompts the budget rule chose against the lens's own 40
   and the Neuronpedia lens's 277, and the layer 4 onward at which base
   falls below its rotation control. All of those reproduce. The terminal
   word agreeing on 25 of 25 prompts was re-checked against
   `experiments/exp_010c_windows/output/results_small010b_SB.json` itself
   and reproduces at 25 of 25.
