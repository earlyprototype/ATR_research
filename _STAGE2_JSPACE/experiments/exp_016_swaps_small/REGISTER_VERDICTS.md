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

| H17 | EXP_016 | Report swap on base GPT-2 Small: exchanging the lens coordinates of the model's own top concept and a target concept (the paper's patching in lens coordinates, at one layer or a small layer set) puts the target in the next-token top-5 on at least 50 percent of trials, against at most 10 percent for norm-matched random-direction swaps | **SUPPORTED** (held-out half 86 percent, 18 of 21, against 0 percent, 0 of 42, at the tuned setting layers 7-8-9, strength 2, all positions, source rule output, with the rule part of the tuned selection as the specification's section 5.1 states; pooled over both source rules 64 percent, 27 of 42, against 0 of 84; both readings clear the registered thresholds) | `experiments/exp_016_swaps_small/RESULTS_EXP016.md` |
| H17a | EXP_016 | Flexible generalisation: one country swap redirects at least two of three downstream completion functions (capital, language, continent) at a rate above the random-direction control, on items the clean model answers correctly | **SUPPORTED** (held-out half 87 percent, 13 of 15, against 0 percent, 0 of 30, for the control, at the tuned setting layer 6, strength 2, all positions; both halves 87 percent, 26 of 30, against 0 of 60; in the five-continent extension set, reported beside the headline number, 14 of 18 pairs redirect all three questions against 0 of 36; on the specification's top-three clean gate, and on the rank-one questions alone 7 of 9 held-out pairs against 0 of 18, see decision item 5 of the results record) | same record |
| H17b | EXP_016 | Intermediate-step surgery: on two-hop completions the clean model answers correctly, swapping the intermediate concept changes the final answer to the predicted alternative more often than the random-direction control | **SUPPORTED on the registered wording, weakly** (4 of 16 items against 0 of 16 for control A over three draws each; exact within-item probability 0.0039 over all items and 0.25 on the held-out half alone, 1 of 8 against 0 of 8, so the effect is inside the usual 5 percent level over all items and outside it on the half the specification judges on; on the specification's top-three clean gate, and all four flips lie among the 7 items the clean model answers at rank one, 4 of 7 against 0 of 7). Owner ruling invited: see decision items 2 and 5 of the results record | same record |

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
> total change. EXP_016 also chose its tuned parameters on one half of its
> items and scored them on the other; whether that becomes the project's
> convention is an open decision recorded in the results record.
