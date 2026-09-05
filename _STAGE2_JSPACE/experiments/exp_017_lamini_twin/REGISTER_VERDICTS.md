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
| H18b | EXP_017 | Post-training changes the terminal states' J-space share: the twin's share on a lens fitted to the twin differs from base's share on the Neuronpedia lens by more than the random-dictionary control spread, two-sided | **SUPPORTED at exactly the pre-registered threshold** (4 of the 6 band layers 5 to 10 meet both conditions, and 4 was the bar; permutation p 0.0001 at every layer; the same-lens cross-check, which does not depend on the threshold, has the twin above base at all 11 layers by 0.015 to 0.086). **Read with the control caveat in §3.5:** base's settled states lie no closer to the lens's nameable directions than to a randomly rotated copy of them from layer 4 onward, so the absolute share level at the band carries no evidence of verbalizable content | `RESULTS_EXP017.md` §3.3 to §3.5 and §5 |

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
4. **Deviations** are listed flat in `RESULTS_EXP017.md` §4. The one a register
   reader is most likely to want is that the twin's Jacobian lens was fitted on
   40 WikiText-103 prompts, below the 50 the spec named as a deviation
   threshold and well below the 277 of the Neuronpedia lens it is compared
   against, by the mechanical application of the spec's own budget rule. The
   sensitivity check in §3.4 shows the H18b verdict is unchanged when that lens
   is degraded to 5 prompts.
