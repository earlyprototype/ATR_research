# EXP_015 — Results Record

**Spec:** `../../EXP_015_SPEC.md` (pre-registered; committed with the
EXP_015 and H15 register rows at `514caaa`, before any analysis output
existed).
**Status:** run complete; verdict recorded below.

---

## 2026-08-01 — Natural-loudness partition vs the Small reference partition

**The answer first: the resemblance does not survive at natural loudness.**
The medium model's natural-strength end states group the 25 prompts no
better than chance against the small model's reference grouping: the
agreement score is -0.113 on the adjusted Rand index scale, where 0 means
no more agreement than a random regrouping and 1 means identical grouping,
and the permutation p is 1.000, meaning all 10,000 random regroupings
matched or beat the observed agreement. The loud-convention baseline for
this same comparison was 0.200 with p = 0.0009. Under the pre-registered
verdict table, H15 is **REFUTED** and the apparatus-mask hypothesis
weakens.

**What ran:** `exp015_natural_ari.py` (this directory) with the registered
parameters: greedy leader clustering at cosine threshold 0.999, adjusted
Rand index, 10,000-shuffle permutation null with seed 42 and the add-one
correction, all imported unmodified from the EXP_010d machinery
(`compare_small_basins.py`, `analyze_terminals.py`). Analysis only; no
language model was loaded. [Environment note: a torch-only virtual
environment without numpy; torch emits a harmless numpy-absent warning,
visible in the run log.]

### Reproduction gate (spec §5 step 1): PASS

Before the new comparison was read, the machinery recomputed the recorded
loud-convention comparison from the committed `terminals_full.pt` and got
the recorded values back: adjusted Rand index 0.2001 against the recorded
0.200, permutation p 0.00090 against the recorded 0.0009, with Small at 7
basins and the loud A0 arm at 4 basins, both as recorded in
`RESULTS_EXP010D.md`. The machinery is the machinery of record, so the
STOP condition did not fire.

### Primary comparison (threshold 0.999, as pre-registered)

| Pair | ARI | perm p | Basin count |
|---|---|---|---|
| Small vs A0-natural | **-0.1134** | **1.0000** | Small 7 vs natural A0 14 |

For scale: the adjusted Rand index runs from 1 (identical grouping) down
through 0 (chance-level agreement), and a negative value means slightly
less agreement than the average random regrouping; here it is statistically
indistinguishable from chance. The permutation p of 1.0000 means the
observed agreement was matched or beaten by every one of the 10,000
shuffles. The degenerate-partition guard did not fire: the natural A0 end
states formed 14 basins at the registered threshold, matching the 14
recorded for this arm in the Control B characterisation
(`RESULTS_EXP010C.md`), so there was substructure to compare.

### Threshold sweep (descriptive only, per spec §5 step 5)

| threshold | Small basins | natural A0 basins | ARI | perm p |
|---|---|---|---|---|
| 0.99 | 4 | 8 | -0.0530 | 0.8716 |
| 0.995 | 5 | 10 | -0.0263 | 0.7968 |
| 0.999 | 7 | 14 | -0.1134 | 1.0000 |
| 0.9995 | 8 | 17 | -0.0563 | 1.0000 |

The agreement sits at or below chance at every threshold (p at least
0.797 everywhere), so the null result is not an artifact of where the
clustering gate falls. This mirrors, in the opposite direction, the
EXP_010d sweep in which the loud-convention agreement was significant at
every threshold that left substructure to compare.

### Verdict (mechanical application of spec §6)

**H15 REFUTED.** The pre-registered reading for this outcome, applied
verbatim: the structure EXP_010d measured is bound to the loud convention,
and the apparatus-mask hypothesis weakens. Per the same pre-registered
reading, this outcome does not by itself separate injection strength from
the mask; the follow-up control named in the spec (a natural-strength arm
engineered to still produce the `D` readout) remains the registered next
step before this result is treated as decisive against the mask
hypothesis. No interpretation beyond the pre-registered readings is
recorded here.

### The mandatory caveat, carried from the spec (§7)

At natural loudness the medium model's runs do not settle: 0 of 25 runs
reached the stability criterion and every run hit the 1,000-iteration cap,
so the compared end states are snapshots of unsettled trajectories, not
settled basins. Injection strength, the suspected mask, and settledness
all vary together between the loud and natural conditions, so this result
alone cannot say which of the three the loud-convention resemblance
depends on. Standing caveats inherited from EXP_010d also apply: single
seed, one 25-prompt subset, the Small reference itself settled on only 18
of 25 prompts, and partition-level agreement is a proxy, not a proof, of
shared geometry.

### Deviations

None. The run followed the spec's frozen parameters exactly.

### Artifacts

`output/exp015_natural_ari.json` (gate values, partitions, ARI, p-values,
sweep), `output/exp015_run.log` (run log), `exp015_natural_ari.py` (the
input-adaptation script).
