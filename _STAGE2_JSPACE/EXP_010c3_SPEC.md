# EXP_010c-3 — Window-grid in-fill around the word-forming cells on GPT-2 Medium (pre-registered spec)

**Status:** PRE-REGISTERED — recorded before any run of these arms.
**Created:** 2026-07-24
**Parent:** `EXP_010c_SPEC.md` and `EXP_010c2_SPEC.md`. Extends the registered
window grid (H9/H9a) and boundary scan (H10/H10a/H10b) whose results are in
`experiments/exp_010c_windows/RESULTS_EXP010C.md`.
**Feeds:** `RUNBOOK_JLENS_MEDIUM.md` (EXP_012m band census, EXP_011m terminal
projection, EXP_013m re-decode) — this sweep hands the J-lens phase a denser
map of which (injection, extraction) cells to target.

---

## 1. Why this exists

The registered grid + boundary scan found whole-word, prompt-dependent
terminals at exactly three window cells on Medium:

- **8→21** (`simultaneously`/`halfway`) — scan arm O8
- **10→21** (`until`/`forever`/`since`) — main arm A4
- **8→15** (`rant`; via-tail `endless`) — main arm A5

Every other window tested (14 total) either funnels to punctuation, fragments,
or — only for the full stack 0→23 — the Stage 1 `D`. The map immediately
**around** these three cells is unmeasured on two axes:

- **Injection axis (extract fixed at 21):** data exists at i ∈ {0,4,6,8,10,12,14}
  from the boundary scan, but the *odd* injection layers between and around the
  two word cells — **5, 7, 9, 11** — were never run. Layer **9** is the single
  most informative untested point: it sits between the two word cells (8→21 and
  10→21) and decides whether they are one continuous island or two separate ones.
- **Extraction axis (below 21):** for the two word-forming injection layers
  (8 and 10) only the endpoints of the extraction ladder have data — j=15 (A5,
  injection 8) and j=21 (O8/A4) — and those two points come from *different*
  injection layers, so the extraction axis is effectively unmeasured. The rungs
  **17, 19** (both injections) and **15** (injection 10) are missing.

This is a cartographic in-fill, not a new mechanism test. Its product is a
denser tested-windows map for the J-lens targeting phase, plus two falsifiable
readings (continuity of the injection zone; shape of the extraction ladder).

## 2. Hypotheses (continuing the numbering)

Numbering note: H11 is used by the EXP_010d line (PR #5, the Small-partition
capstone); this spec continues at **H12** to avoid collision.

- **H12 (injection-zone continuity):** the whole-word prompt-dependent zone on
  the injection axis (extract 21) is **continuous** across i ∈ {8,9,10} — i.e.
  9→21 also yields whole-word, prompt-dependent terminals, joining 8→21 and
  10→21 into one contiguous island. Refuted if 9→21 breaks the pattern
  (punctuation/fragment funnel, or a single shared terminal), which would make
  8→21 and 10→21 two separate one-layer islands.
- **H12a (injection-zone edges):** the flanking odd layers 5→21, 7→21 and 11→21
  fall on the *off-band* side of the transition already located coarsely at
  i≈6 and i≈12 in the boundary scan — i.e. they resemble their even neighbours
  (5,7 → the punctuation/fragment character of i≤6; 11 → still whole-word if the
  zone extends to 11, else the i≥12 fragment/symbol character). This sharpens
  the onset/exit edges of the injection zone to a single-layer resolution.
- **H13 (extraction ladder):** at fixed injection (8 or 10), landscape character
  (unique-terminal count, lexical token class, margins, via-tail agreement)
  varies **non-uniformly** as the extract layer j descends 21→19→17→15 — there
  is an identifiable extraction depth below which the whole-word,
  via-tail-robust character is lost. Refuted if the character is flat across
  the ladder (no j-dependence) or varies smoothly with no locatable edge.

These are readings, not commitments to any mechanism. The results record stays
observations-only (house rule, session note 2026-07-23); the H-verdicts below
are mechanical applications of the pre-registered table.

## 3. Design

**Model, prompts, protocol:** identical to the EXP_010c full tier and the
EXP_010c-2 scan — gpt2-medium loaded offline (S3-mirror route, recorded in
RESULTS §Model acquisition), the same recorded 25-prompt subset
(`output/prompt_subset.json`, derived deterministically by `derive_prompts.py`
from the Stage 1 record; reproduction of the exact subset re-verified this
session), gated (cos > 0.999 ×3, check_every 10, check_start 100, max_iter 1000),
seed 42, L0 natural-pass seeding (parent spec §3), terminal mean+last vectors
saved per (window, prompt).

**Arms (9 new windows):**

| Sweep | Arm | Window (inject i → extract j) | Fills in |
|---|---|---|---|
| Injection in-fill | `I5`  | 5→21  | injection edge, off-band side of the i≈6 onset (H12a) |
| Injection in-fill | `I7`  | 7→21  | injection approach to the word cell at 8 (H12a) |
| Injection in-fill | `I9`  | 9→21  | **the critical point** — between 8→21 and 10→21 (H12) |
| Injection in-fill | `I11` | 11→21 | injection exit toward the i≈12 transition (H12a) |
| Extraction column | `X817`  | 8→17  | injection-8 ladder rung (H13) |
| Extraction column | `X819`  | 8→19  | injection-8 ladder rung (H13) |
| Extraction column | `X1015` | 10→15 | injection-10 ladder foot (H13) |
| Extraction column | `X1017` | 10→17 | injection-10 ladder rung (H13) |
| Extraction column | `X1019` | 10→19 | injection-10 ladder rung (H13) |

With the existing data this completes the local map:

- Injection axis at extract 21 gains {5,7,9,11}, joining {0,4,6,8,10,12,14}.
- Extraction ladder now full at both word-forming injections:
  injection 8 → j ∈ {15 (A5), 17, 19, 21 (O8)}; injection 10 → j ∈ {15, 17, 19, 21 (A4)}.

**9 new windows × 25 prompts = 225 runs**, est. ~2–4 h CPU at the observed
30–60 s/run throughput.

**Recorded deviation (count):** the parent issue's header states "10 new
windows × 25 prompts = 250 runs", but its own window table enumerates the **9**
windows above (4 injection + 5 extraction). Each of the 9 is individually
justified by the map gap it fills; there is no 10th cell with an independent
justification (the extraction ladders for injections 8 and 10 are already
symmetric and complete once 8→15 and 10→21 — which exist — are counted). Per
the house rule "do not hand-pick prompts / windows", inventing an unjustified
10th window to hit the header count would itself be the bias the rule guards
against. This spec therefore registers the 9 enumerated windows and records the
discrepancy here rather than padding the grid.

**Decode-via-tail readout control:** applied to every arm here, identical to
EXP_010c-2 §3 — each terminal tensor at extract layer j < 23 is run once (no
looping) through layers j+1..23 and decoded at 23; per-arm agreement with the
direct logit-lens-at-j decode is recorded. (All 9 windows have j < 23, so the
tail is non-empty for every arm.)

## 4. Pre-registered readings

| Observation | Reading |
|---|---|
| 9→21 yields whole-word, prompt-dependent terminals (multiple unique, non-punctuation) | **H12 supported** — the injection zone is continuous across i ∈ {8,9,10}; report it as one contiguous island to the J-lens phase. |
| 9→21 funnels to punctuation/fragments or a single shared terminal | **H12 refuted** — 8→21 and 10→21 are separate one-layer islands; the J-lens phase targets them independently. |
| 7→21 whole-word but 5→21 not | injection onset of the word zone sits between i=5 and i=7 (finer than the scan's i∈(4,6) bracket). |
| 11→21 whole-word but flanked by fragment/symbol at 12→21 (existing O12) | injection exit of the word zone sits between i=11 and i=12. |
| 5→21 and 7→21 both resemble the i≤6 punctuation/fragment character | **H12a supported** on the onset side; the word zone's injection onset is at i=8, sharp. |
| Extraction ladder: whole-word + via-tail-robust at j=21/19 but lost at j=17/15 (or any single locatable j\*) | **H13 supported**; j\* is the extraction-axis edge, handed to EXP_012m/EXP_013m as the extract-layer to prioritise. |
| Extraction ladder character flat or smoothly varying with no locatable edge | **H13 refuted** (flat) or weakened (smooth); record and defer the extraction-axis targeting to the J-lens census with no ATR-derived prior. |
| Any arm fails to converge within max_iter on some prompts | record non-convergence counts per arm (observation only); do not treat as a terminal-basin claim. |
| Decode-via-tail collapses/inverts an arm's direct-decode terminals | mid-stack readout unreliable at that j; flag the cell for EXP_013m as readout-load-bearing; record per-arm agreement. |

## 5. Execution

- Harness: `experiments/exp_010c_windows/run_exp010c.py`. The 9 windows are
  added to the single-source `ARMS` dict and grouped in a new `infill` tier
  (order: I5, I7, I9, I11, X817, X819, X1015, X1017, X1019). Run:
  `python run_exp010c.py --tier infill --model-path <local gpt2-medium>`.
- Per-arm checkpointing is built in (results + terminals written after every
  arm); **commit and push after each arm** so a container restart mid-sweep
  loses nothing (house rule).
- Characterisation + via-tail control after the sweep:
  `python analyze_terminals.py --tier infill --decode-via-tail --model-path <local>`.
- Determinism: seed 42, same machine reproduces exactly (the parent runs
  verified 350/350). Any single-process vs restart-split deviation is recorded.

## 6. Definition of done (from the issue)

1. This spec committed before the runs.
2. All 225 runs executed and converged (or non-convergence recorded);
   `results_infill.json`, `terminals_infill.pt`, and the characterisation JSON
   committed.
3. `RESULTS_EXP010C.md` gains a dated, observations-only section: per-window
   observation table in the established format (terminals, unique count, lexical
   token class, lock-ins, margins, via-tail agreement, tensor-basin counts).
4. An updated tested-windows map handed forward: which injection layers and
   which extraction layers produced whole-word prompt-dependent terminals, at
   single-layer resolution around the three seed cells, for the J-lens phase.

## 7. Standing caveats (carried from the parent specs)

Single seed; one 25-prompt subset; cluster-threshold sensitivity unexplored;
direct decode at j<23 is a logit-lens-at-layer-j readout whose off-band
unreliability is already measured; two-instrument (direct + via-tail) agreement
is necessary but not sufficient. The J-lens re-decode (EXP_013m) remains the
registered arbiter for every mid-stack terminal claim, including all cells in
this in-fill.

---

## Addendum 2026-07-25 — gate parameters stated explicitly (PR #33 review)

**Clarification only. The protocol is unchanged and the runs are unaffected** —
this records, in one place, the exact values §3 referred to as
"gated (cos > 0.999 ×3, check_every 10, check_start 100, max_iter 1000)".
`cos > 0.999 ×3` was flagged as ambiguous as a runner contract, since it does
not by itself say what is compared, how often, or from when. The executed
contract, as implemented in `atr_engine2.run_atr_gated`:

| Parameter | Value | Meaning |
|---|---|---|
| `threshold` | `0.999` | lock requires cosine **strictly greater** than this |
| `patience` | `3` | consecutive passing checks required |
| `check_every` | `10` | a check occurs only when `iteration % 10 == 0` |
| `check_start` | `100` | no check before this iteration |
| `max_iter` | `1000` | classify here if lock never occurs |
| `gate_lag` | `1` | cosine is iterate *t* vs iterate *t − 1* |
| compared quantity | mean vector | `cos_sim_mean`, not the last-position vector |

Consequence worth stating because it shaped the observations: the earliest
reportable lock is **120** (checks at 100, 110, 120), so a run reporting
`lock_in_iter = 120` had already satisfied the gate at its first opportunity and
120 is an upper bound on settle time, not a measurement of it. EXP_010c-3b §5
measures the actual value.
