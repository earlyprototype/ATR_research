# Proposed register rows for EXP_016

**What this file is.** The identifier register `_STAGE2_JSPACE/REGISTER.md`
is the authority for every hypothesis number and experiment identifier in
this project, and four experiment branches were allocated adjacent rows in
one shared commit, so none of them edits the register on its own branch.
This file therefore carries the rows that EXP_016 proposes, in the register's own
format, for the orchestrator to sweep into the register. Nothing here is
authoritative until it lands there.

**Source of the numbers:** `experiments/exp_016_swaps_small/RESULTS_EXP016.md`
and the summary files beside it. **Specification of record:**
`_STAGE2_JSPACE/EXP_016_SPEC.md`, committed at `47efd8d` before any swap ran.

---

## 1. Hypothesis register rows

These replace the three UNTESTED rows allocated by erratum (f) on
2026-09-05. Each keeps the registered statement word for word and changes
only the verdict and the recorded-at columns.

| H17 | EXP_016 | Report swap on base GPT-2 Small: exchanging the lens coordinates of the model's own top concept and a target concept (the paper's patching in lens coordinates, at one layer or a small layer set) puts the target in the next-token top-5 on at least 50 percent of trials, against at most 10 percent for norm-matched random-direction swaps | **SUPPORTED** (held-out half 86 percent, 18 of 21, against 0 percent, 0 of 42, at the tuned setting layers 7-8-9, strength 2, all positions, source rule output, with the rule part of the tuned selection as the specification's section 5.1 states; pooled over both source rules 64 percent, 27 of 42, against 0 of 84; both readings clear the registered thresholds; the target word was in the model's top five on 0 of 84 items before any swap, by the battery's construction; grouping the 21 held-out frames of the output rule into the 7 connected groups within which any two frames share a lens direction, and treating the lens as one draw per group, the exact test gives 0.00137, which is 1 in 729 and the smallest value 6 informative groups of 3 draws can return, against a control whose random directions are seeded by the concept they stand in for so that they are reused across frames exactly as the lens directions are, run on 2026-09-06; the same 0.00137 comes out of a stricter control that also picks its own target concept by the layer-8 rule the battery used, so the figure does not depend on the lens arm having been allowed to choose a favourable target, see correction 27 of the results record. Three earlier figures for this gap are withdrawn as headlines: the item-level 2.6 times 10 to the power minus 9, see correction 16; 5.1 times 10 to the power minus 5 computed against the registered control A, see correction 19; and 5.1 times 10 to the power minus 5 computed with the source concept as the group, a grouping under which different groups still share a direction, see correction 25, though that value is still reported in the record against the corrected control) | `experiments/exp_016_swaps_small/RESULTS_EXP016.md` |
| H17a | EXP_016 | Flexible generalisation: one country swap redirects at least two of three downstream completion functions (capital, language, continent) at a rate above the random-direction control, on items the clean model answers correctly | **SUPPORTED** (held-out half 87 percent, 13 of 15, against 0 percent, 0 of 30, for the control, at the tuned setting layer 6, strength 2, all positions; both halves 87 percent, 26 of 30, against 0 of 60; in the five-continent extension set, reported beside the headline number, 14 of 18 pairs redirect all three questions against 0 of 36; on the specification's top-three clean gate, and on the rank-one questions alone 7 of 9 held-out pairs against 0 of 18, see decision item 5 of the results record; the registered success criterion counts a question as redirected when the target answer is in the model's top five after the swap without requiring that it was absent from that top five before, and 24 of the 124 scored questions, 19 percent, already met it with no intervention, so the stricter readings are given beside the row's number: requiring the target answer to be new to the top five gives 12 of 15 held-out pairs against 0 of 30, and requiring it to outrank the source country's own answer gives 12 of 15 against 0 of 30, see correction 15 and decision item 7 of the results record; no significance test is offered for this row, and the verdict rests on the counts above against the registered thresholds: the 15 held-out pairs are three pairs for each of five source countries drawn from those same five countries as targets, so every pair shares a country direction with every other and the held-out half is a single group, whose grouped exact test cannot return less than one third however large the effect and does return exactly one third, see correction 25 of the results record. Three probabilities offered in earlier versions of the record are withdrawn as headlines: the item-level 6.3 times 10 to the power minus 7, see correction 16; 0.0041 computed against the registered control A, see correction 19; and 0.0041 computed with the source country as the group, see correction 25) | same record |
| H17b | EXP_016 | Intermediate-step surgery: on two-hop completions the clean model answers correctly, swapping the intermediate concept changes the final answer to the predicted alternative more often than the random-direction control | **SUPPORTED on the registered wording, weakly** (4 of 16 items against 0 of 16 for control A over three draws each; exact within-item probability 0.25 on the held-out half, 1 of 8 against 0 of 8, the only valid test of the selected setting since the tuning half chose it, so the effect is not distinguishable from chance; on the specification's top-three clean gate, and all four flips lie among the 7 items the clean model answers at rank one, 4 of 7 against 0 of 7; no two of the eight held-out items share a concept, so the cluster-level test that changed the H17 and H17a probabilities returns the same 0.25 here, whether the groups are formed by source concept or as connected components of the items sharing any direction, and it needs no cluster-matched control because a group of one item is the within-item test already; 0.25 is also the smallest probability that test can return on these outcomes, since seven of the eight held-out items have a lens draw and three control draws that all failed, see correction 20 of the results record; over all 16 items, which is a post-selection reading and no test of the chosen setting, three pairs of items do share a concept and the figure moves from 0.0039 to 0.0156, see correction 25). Owner ruling invited: see decision items 2 and 5 of the results record | same record |

## 2. Experiment register row

This replaces the IN PROGRESS row for EXP_016.

| EXP_016 | Completion-compatible swap battery on base GPT-2 Small (report swap, flexible generalisation, intermediate-step surgery) with the Neuronpedia lens (H17, H17a, H17b) | **COMPLETE** (2026-09-05; branch `claude/latent-context-small-llms-u2jdig-exp016`; 68,000 swap conditions in 50 minutes of processor time) | `EXP_016_SPEC.md` (committed `47efd8d` before any swap ran) | `experiments/exp_016_swaps_small/RESULTS_EXP016.md` |

## 3. Note offered for the register's errata, if the orchestrator wants one

The wording below is offered, not asserted. It records the one methodological
point from EXP_016 that later experiments would otherwise have to rediscover.

> **EXP_016 method note, 2026-09-05.** EXP_016 ran a second control beside
> the norm-matched random-direction control that H17 registers. The added
> control uses the same random directions but rescales the resulting change
> to the residual stream, position by position and layer by layer, to the
> size of the change the real lens swap would make on the control's own
> residual stream. It was added on the expectation that the registered
> control under-disturbs the model. The sizes, measured after the run at the
> chosen settings, show that expectation to be wrong in general: control
> A's change was about three times the lens swap's at the H17a setting and
> 0.5 to 0.7 times at the other two. They also show that swaps stacked at
> consecutive layers compound (at the H17 setting, layers 7-8-9 at strength
> 2, the lens arm's change grows about sevenfold from layer 7 to layer 9),
> so the size-matched control matches per layer and not in total at
> multi-layer settings. In EXP_016 the two controls behaved almost
> identically in outcome, so no verdict turns on the distinction, but later
> swap work should register whichever control TC prefers from the start
> and, if it wants a size-fair control at multi-layer settings, match the
> total change. A third control had to be run after the fact, on 2026-09-06,
> for a different reason: a test that groups the scored units sharing one
> source lens direction and treats the lens arm as one draw per group needs a
> control whose randomness is shared inside a group the same way, and neither
> control A nor control B is built that way, because both draw their random
> directions afresh for every item. The added control draws one random
> direction to stand in for the source concept and shares it across the
> group, with an independent random direction standing in for each item's
> target concept, both rescaled to the lengths of the lens directions they
> replace. That control was corrected twice more on the same day, and the
> shape later work should register is the corrected one: seed each random
> direction by the concept it stands in for rather than by the item, since
> the lens arm gives two units that swap the same concept the same direction;
> form the groups as the connected components of the units that share any
> direction, since groups formed by source alone still share targets with one
> another; and, wherever the battery picks its target concept by a lens
> reading, apply that same rule to the control's own random directions, since
> otherwise the reported probability is conditional on the lens having been
> allowed to choose a favourable target. A battery whose units all fall into
> one component, as EXP_016's country battery does, can carry no group-level
> probability at all, and should be designed with disjoint source and target
> pools if a group-level test is wanted. EXP_016 also chose its
> tuned parameters on one half of its
> items and scored them on the other; whether that becomes the project's
> convention is an open decision recorded in the results record.
>
> The measurement of the intervention sizes in the note above divides a
> change accumulated over the patched layers by the untouched working memory
> of all those layers taken together. The first version of that measurement
> divided by the first patched layer alone, which inflated every multi-layer
> ratio; the corrected figures are in the results record and the withdrawn
> ones are named there in correction 21.
