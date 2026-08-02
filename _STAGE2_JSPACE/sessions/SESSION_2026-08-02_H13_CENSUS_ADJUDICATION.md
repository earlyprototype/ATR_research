# SESSION 2026-08-02: adjudication of the H13-vs-census tension (issue #73)

Chartered by issue #73, which executes the outstanding piece of ruling
item 4 from the issue #52 resolution: adjudicate what the full window
census means for hypothesis H13 before any J-lens phase consumes an
extraction-depth prior. This record is analysis only. Every number in it
is computed from committed JSON artifacts by the script
`h13_census_agreement_analysis.py` in this directory; the script's output
is committed beside it as `h13_census_agreement_analysis_output.json`. No
model was run, and no registered artifact was modified.

## The answer, first

On question (a): the two flagship cells' high two-readout agreement is
neither explained by their terminal class nor unique to those two cells.
It is a per-cell property. The whole-word prompt-dependent class shows
only a weak and conventionally non-significant tendency toward higher
agreement (an excess of 2.9 prompts out of 25 over same-depth cells,
one-sided permutation p = 0.06), and agreement inside that class spans
the full range from 0 to 25 of 25 prompts. At the same time, five of the
20 non-flagship cells measured at extraction depth 21 reach at least 17
of 25 agreement, so the flagship cells do not stand alone in their
column either. Both findings are established directly from the committed
census and prior-tier artifacts.

On question (b): downstream readout work, EXP_011m in particular, is
licensed to consume no extraction-depth prior at all. No depth below 23
is licensed as generally reliable, and depth 21 is licensed as nothing
more than the depth at which two specific cells happen to agree well.
The only licensed prior is the per-cell agreement table itself, stated
in section (b) below.

Both answers are provisional on the issue #71 stopping-rule stability
check, in the precise sense given in the provisionality section below.

## Terms and evidence base

The "flagship cells" are the two loop windows (inject layer 8, extract
depth 21) and (inject layer 10, extract depth 21) on GPT-2 Medium whose
whole-word, prompt-dependent terminals anchor hypothesis H13, the
in-fill hypothesis that whole-word via-tail-robust character is lost
below extraction depth 21 at those two injection layers. "Agreement"
for a cell means the number of the 25 registered prompts on which the
two readout instruments name the same terminal token: the direct
readout, which decodes the settled state at the extraction depth with
the model's output dictionary, and the via-tail readout, which first
passes the settled state through the model's remaining layers and then
decodes. "W*" marks a cell whose terminal set is all whole words and
contains at least two distinct terminals, the whole-word AND
prompt-dependent class of the EXP_010c-3 specification's section 3
mechanical rule; the script reimplements that rule identically to
`build_final_map.py`.

The evidence is the four committed characterisation artifacts under
`experiments/exp_010c_windows/output/` (census, full, scan and infill
tiers), covering all 300 valid windows. The 24 cells at extraction
depth 23 are excluded from every agreement statistic because their
recorded figure is not a via-tail measurement (the tail is empty
there), which leaves 276 via-tail-measured cells. The script asserts
the flagship values against the record before computing anything:
(8 to 21) at 17 of 25 and (10 to 21) at 23 of 25, both reproduced
exactly.

## (a) Class effect and outlier status, with numbers

**The class effect is weak and does not explain the flagships.** The
17 census-tier W* cells with via-tail measurements average 9.1 of 25
agreement, against 6.2 of 25 for the other 239 census cells, a raw gap
of about 3 prompts. Because agreement rises with extraction depth and
the W* cells sit disproportionately late in the stack, the script also
runs a depth-stratified permutation test: holding each cell's depth
fixed and shuffling class labels within each depth 10,000 times, the
observed W* excess of 2.9 prompts out of 25 over same-depth census
cells has one-sided p = 0.06, where p below 0.05 would conventionally
mark significance. That test is the inference in this record; the
underlying counts are established. Inside the class the spread is
total: the 19 via-tail-measured W* cells range from 0 of 25 (five
cells) to 25 of 25, mean 10.2, so membership in the class predicts
almost nothing about agreement for any individual cell.

**The flagships are not alone at depth 21.** The full depth-21 column
now holds 22 measured cells (11 census arms plus 11 prior-tier arms;
the register's erratum (d) figure of eleven arms averaging 6.6 of 25
describes the census arms only). Ranked by agreement, (10 to 21) at 23
of 25 is third, behind (7 to 21) and (11 to 21) at 25 of 25, and
(8 to 21) at 17 of 25 is sixth, tied with (14 to 21) at 17 of 25.
The two cells above the flagships are single-terminal cells (one
funnels every prompt to a repeated-letter fragment, the other to the
end-of-text marker), so their perfect agreement is agreement about
degenerate terminals; but (17 to 21) at 20 of 25, (15 to 21) at 18 of
25 and (14 to 21) at 17 of 25 are prompt-dependent mixed-class cells
sitting at or above the (8 to 21) flagship. The non-flagship column
mean is 8.4 of 25 and its median is 7 of 25, so the flagships do sit
in the column's upper tail, but they share that tail with five other
cells.

**What this settles.** The issue's framing, that the only two
high-agreement depth-21 cells are exactly the two flagship word cells,
was an artifact of reading the census arms alone; over all 22 measured
depth-21 cells it is not true. High two-readout agreement at depth 21
is a heterogeneous per-cell property, not a class property and not a
flagship-exclusive property. H13's recorded verdict is untouched by
this: it is scoped to injection layers 8 and 10, where the
pre-registered extraction ladder did show a sharp within-row edge at
depth 21, and nothing here re-measures those rows. What dies, and had
already died in erratum (d), is any column-wide or depth-wide reading;
what this adjudication adds is that the class-based rescue of that
reading (whole-word cells simply agree more) also fails.

## (b) The extraction-depth prior for downstream readout work

Downstream readout work is licensed to consume the following, and
nothing stronger.

1. **No depth prior.** There is no extraction depth below 23 at which
   the two readout instruments generally agree. Census mean agreement
   rises roughly monotonically from about 0 of 25 at depths 0 to 4 to
   18.9 of 25 at depth 22, but per-cell spread at every depth remains
   wide (0 to 25 of 25 within the depth-21 column alone), so no depth,
   including 21 and 22, may be consumed as "reliable" by rule. The
   pre-census phrasing "extraction 21 is the only reliable depth" stays
   dead, and nothing replaces it.

2. **A per-cell prior instead.** The licensed prior is the committed
   per-cell agreement table (the four `terminal_characterisation_*.json`
   artifacts), consumed cell by cell. For the flagship cells that means
   exactly: (8 to 21) agrees on 17 of 25 prompts and (10 to 21) on 23
   of 25, under the registered stopping rule, single seed, one prompt
   subset.

3. **What EXP_011m needs.** EXP_011m, the subspace-overlap experiment
   promoted to primary arbiter for workspace-content claims about
   Medium terminal states (register erratum (e)), bypasses word
   readouts entirely, so it requires no readout-depth prior to run.
   Where its interpretation, or any successor readout experiment such
   as EXP_013m, leans on terminal identities at specific cells, it must
   lean on the per-cell numbers in point 2 and must not assume that a
   depth-21 extraction is trustworthy anywhere outside the cells where
   agreement was measured high.

## Provisionality on issue #71

The stopping-rule stability check at the flagship cells, chartered as
issue #71, is running in parallel and had not reported when this record
was written. This adjudication is provisional on its outcome in the
following exact sense. If the flagship terminals turn out to be
stopping-rule-conditional, then the specific per-cell numbers in point
2 of section (b), 17 of 25 at (8 to 21) and 23 of 25 at (10 to 21),
lose their standing as a prior and must be re-measured under whatever
terminals the stability check validates, and any weight placed on the
flagships' upper-tail position in the depth-21 column falls with them.
The structural conclusions would stand: the finding that the class does
not explain agreement and the finding that high agreement is per-cell
and not flagship-exclusive both rest on the other 274 via-tail-measured
cells, which were all measured under the same registered rule and which
issue #71 does not touch, and the "no depth prior" conclusion of
section (b) point 1 would if anything strengthen, since a
rule-conditional flagship would remove the last cells arguing for
depth 21.

## What remains, and what needs the operator

Nothing in this record needs a TC ruling: issue #73 chartered the
adjudication itself, and this record completes it. What remains is
mechanical: issue #71 reports, and if its outcome is adverse the
per-cell flagship numbers above are struck per the provisionality
section. EXP_011m may proceed without waiting, consuming the section
(b) prior.

**Deviations:** the issue said the stopping-rule check of issue #71
"should ideally report first"; this record proceeded in parallel per
the issue's own charter and records its dependence explicitly above.
The record lands as a session note, one of the two locations the issue
offers. The analysis script and output live in this sessions directory
rather than under the experiment's output directory, to keep analysis
artifacts separate from registered run artifacts. No other deviations.
