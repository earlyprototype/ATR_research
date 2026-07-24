# EXP_010d — Does the Medium workspace-band loop recreate GPT-2 Small's basin geometry? (pre-registered spec)

**Status:** PRE-REGISTERED — recorded before the Small run.
**Created:** 2026-07-24
**Parent plan:** `STAGE2_PLAN.md` (Phase 1). Capstone of the EXP_010c line.
**Register:** reporting register. Verdict criteria stated before execution.

---

## 1. Question — the hypothesis EXP_010c only half-tested

The programme hypothesis is: **selective tensor-injection loops between set layer
heights recreate GPT-2 Small's basin geometry.** GPT-2 Small is the reference
model — Stage 1 found it is the one sibling whose full-stack ATR loop funnels
language-driven activity into a *small set of semantic, high-confidence
attractors*. GPT-2 Medium, same corpus, collapses the full stack to the single
token `D`.

EXP_010c established the necessary precursor: windowing Medium's loop into the
paper-mapped workspace band (arm A4, 10→21) escapes the `D` collapse and produces
multiple, prompt-dependent, whole-word terminals with high margins (H9/H9a
SUPPORTED). That is a *subset* of the hypothesis — it shows the band window
departs from Medium's degenerate baseline. It does **not** yet show the departure
*lands on Small's geometry*. Escaping collapse is nearly free (13 of 14 windows in
the 010c/010c2 grid do it); reproducing Small's specific basin structure is the
actual bar.

**EXP_010d asks the capstone question directly: does windowed Medium (A4, 10→21)
partition the same prompts into basins the way native GPT-2 Small does?**

## 2. Hypothesis (continuing the H-numbering; last used H10b)

- **H11 (geometry recreation):** on a shared prompt set, the basin *partition*
  induced by the Medium workspace-band loop (A4, 10→21) agrees with the basin
  partition induced by the native GPT-2 Small loop (0→11) significantly above
  chance, and more than the Medium full-stack baseline (A0, 0→23) does.
  - **H11a (basin count):** A4's effective basin count is closer to Small's than
    A0's is (A0 collapses to one effective basin — a single `D` decode).

## 3. Why partition-level, not tensor-level (pre-registered measurement choice)

Small's residual stream is 768-dimensional; Medium's is 1024. Terminal tensors are
**not** directly comparable across models — there is no shared basis, so a raw
cosine between a Small terminal and a Medium terminal is undefined. "Recreate the
basin geometry" is therefore operationalised at the level the two models share: the
**partition of the prompts** — which prompts fall into the same basin as which
others. Two systems have "the same basin geometry" over a prompt set when they
group the prompts the same way, regardless of the coordinates each uses internally.

Decode-token identity is reported as a secondary, descriptive read (Small and
Medium share the GPT-2 vocabulary), but it is **not** the H11 criterion — the
logit-lens-at-layer-j caveat from EXP_010c applies to the Medium side, and the
J-lens re-decode (EXP_013m) remains the arbiter for any token-level claim.

## 4. Design

**Shared inputs (frozen):** the 25-prompt subset already used by EXP_010c,
loaded verbatim from `exp_010c_windows/output/prompt_subset.json` (round-robin
over the 7 categories; recovered from the Stage 1 dissolution record).

**Arms compared:**

| Ref | Model | Window | Source of terminals |
|---|---|---|---|
| SMALL | gpt2-small | 0→11 (native full stack) | run this experiment |
| A4 | gpt2-medium | 10→21 (workspace band) | committed `terminals_full_A0A4.pt(.bak)` |
| A0 | gpt2-medium | 0→23 (full-stack baseline) | committed `terminals_full_A0A4.pt(.bak)` |

**Protocol (identical to EXP_010c):** convergence-gated loop, threshold 0.999,
patience 3, check_every 10, check_start 100, max_iter 1000, L0 natural seed,
energy-rescaled. Small is run with the verbatim `atr_engine2` engine, window 0→11.

**Basin extraction:** greedy leader clustering of the terminal mean vectors at
cosine 0.999 (the gate threshold), per model — the same `cluster()` used in
`analyze_terminals.py`. This yields a prompt→basin label vector for each of
SMALL, A4, A0.

**Agreement metric:** Adjusted Rand Index (ARI) between label vectors —
SMALL↔A4 and SMALL↔A0. ARI is chance-corrected and invariant to label
permutation and to basin count, so it is a fair cross-model comparison.

**Null:** permutation null on the ARI — hold SMALL's labels fixed, shuffle the
other model's labels N=10000 times, and read the fraction of shuffles reaching the
observed ARI (one-sided p). Seeded for reproducibility.

## 5. Pre-registered verdict table

| Outcome | Reading |
|---|---|
| ARI(SMALL,A4) > 0 with permutation p < 0.05, **and** ARI(SMALL,A4) > ARI(SMALL,A0) | **H11 SUPPORTED** — the workspace-band loop recreates Small's partition above chance and beyond the degenerate baseline. |
| ARI(SMALL,A4) not above chance, **or** not above A0 | **H11 REFUTED** — escaping the `D` collapse is not the same as recreating Small; the band window departs from Medium baseline but does not land on Small's geometry. |
| A4 basin count within [Small−k, Small+k]; A0 = 1 effective basin | **H11a SUPPORTED** (k recorded, not tuned post hoc: k = round(0.5 × Small_count)). |

**STOP condition:** if the native Small run does not itself consolidate (no
lock-in on the majority of prompts), Small has no stable partition to match and
H11 is unanswerable on this subset — record and stop; do not force a comparison.

## 6. Standing caveats (inherited)

Single seed; one 25-prompt subset; cluster-threshold sensitivity unexplored;
cross-model partition agreement is necessary but not sufficient for "same
mechanism" — it says the two models sort these prompts alike, not that they do so
for the same reason. The decode-token side carries the logit-lens-at-layer-j
unreliability measured in EXP_010c. The J-lens re-decode (EXP_013m) and the
anisotropy-corrected permutation control remain the registered arbiters for any
semantic or mechanistic claim. **This experiment tests partition geometry only.**

## 7. Deviations

Recorded in `RESULTS_EXP010D.md` as they occur (model route, environment, any
protocol departure), per the folder rule.
