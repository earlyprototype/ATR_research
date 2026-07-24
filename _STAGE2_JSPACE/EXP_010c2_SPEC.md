# EXP_010c-2 — Band Boundary Scan on GPT-2 Medium (pre-registered spec)

**Status:** PRE-REGISTERED — recorded before any run. Queued behind the EXP_010c
registered full run (same session or operator machine).
**Created:** 2026-07-23
**Parent:** `EXP_010c_SPEC.md`. Motivated by its pilot signal (band windows
qualitatively richest) and two design questions raised in review: (1) can ATR
itself localise the band edges, and (2) is the final layer causally special —
what, if anything, is lost by omitting it from the loop?

---

## 1. Questions

- **Q-edge:** where are the boundaries of the window-richness effect? The
  EXP_010c grid tests placement coarsely; this scan slides one window edge at a
  time to localise the transition. The resulting ATR-derived band edges become a
  falsifiable prediction for the J-lens band census (EXP_012m): **the two
  instruments should localise the same band by independent means.**
- **Q-final:** is the final layer (motor tail) *necessary* for the `D` collapse?
  The pilot shows it is not *sufficient* (A3 12→23 includes layer 23 and does
  not produce `D`). The exit-edge sweep isolates necessity.
- **Q-readout:** how much of any window's apparent landscape is an artifact of
  decoding mid-stack states through `ln_final → W_U` (an instrument calibrated
  to post-layer-23 statistics)? A decode-via-tail control quantifies this
  without the J-lens.

## 2. Hypotheses (continuing the numbering)

- **H10 (edge localisation):** landscape character (unique-terminal count,
  margin distribution, prompt-dependence) changes non-uniformly as a window
  edge slides — there exist identifiable onset and exit boundaries, not a
  smooth gradient.
- **H10a (final-layer necessity):** re-including the last two layers
  (10→22, 10→23 vs 10→21) collapses the band window's landscape back toward a
  single funnel. Refuted if 10→23 stays as rich as 10→21.
- **H10b (sensory-splice necessity):** extending the window to include the
  sensory front (0→21) degrades richness even *without* the motor tail.
  Together H10a/H10b decompose the `D` collapse: motor tail, sensory splice,
  both, or only their conjunction (the full-stack splice).

## 3. Design

**Model, prompts, protocol:** identical to EXP_010c full tier — gpt2-medium,
the same recorded 25-prompt subset, gated (0.999 ×3, check_every 10,
check_start 100, max_iter 1000), L0 natural-pass seeding (spec §3 of parent),
terminal mean+last vectors saved per (window, prompt).

**Arms:**

| Sweep | Windows | Isolates |
|---|---|---|
| Exit edge (Q-final) | 10→21 *(reuse from EXP_010c A4 — do not rerun)*, **10→22**, **10→23** | Final-layer / motor-tail necessity (H10a) |
| Onset edge | **0→21**, **4→21**, **6→21**, **8→21**, **12→21**, **14→21** | Where the front boundary sits; 0→21 is the sensory-splice test (H10b) |

8 new windows × 25 prompts = 200 runs, est. 2–3 h CPU at observed throughput.

**Decode-via-tail readout (new, added to every arm here and retrofit to the
EXP_010c terminals):** for each terminal tensor at extract layer j < 23, run it
once (no looping) through layers j+1..23 and decode at 23 with the standard
readout. Records what the model's own downstream pathway makes of the state —
the workspace-flavoured question ("what would the motor circuits say this state
contains") — and quantifies the mid-stack readout mismatch by comparison with
the direct decode. One forward pass per terminal; negligible cost.

## 4. Pre-registered readings

| Observation | Reading |
|---|---|
| 10→22 and/or 10→23 funnel to one terminal while 10→21 stays rich | **H10a supported** — the motor tail is causally necessary for the collapse. Strongest mechanistic sentence available: the collapse is the motor band's doing. |
| 10→23 as rich as 10→21 | **H10a refuted** — final layers neither sufficient (pilot A3) nor necessary; the collapse belongs to the full-stack splice specifically. |
| 0→21 degrades toward uniformity vs 10→21 | **H10b supported** — sensory splice contributes independently. |
| Richness changes sharply at some i* in the onset sweep | **H10 supported**; i* is the ATR-derived band onset, handed to EXP_012m as a prediction. |
| Richness varies smoothly / monotonically with window length | Placement story weakens; length confound returns — escalate the no-splice control from the parent spec before interpreting. |
| Decode-via-tail collapses distinct direct-decode terminals into one token (or vice versa) | Mid-stack readout is unreliable at that layer; J-lens re-decode (EXP_013m) becomes load-bearing for all terminal claims there. Record per-layer agreement. |

## 5. Recorded rationale on the final layer (from review discussion)

Reasons to suspect layer 23 is special: the J-space paper's motor regime is
token-locked; GPT-2's final layers align the residual stream to the unembedding
geometry (why the logit lens only works late); `ln_final` is calibrated to
layer-23 output statistics. None of this makes omitting it from the loop a
defect — the omission *is* the treatment under H9 — but it does make the
readout of mid-stack terminals suspect (hence decode-via-tail here and EXP_013m
later), and it makes H10a the correct isolation test for the dynamics.
