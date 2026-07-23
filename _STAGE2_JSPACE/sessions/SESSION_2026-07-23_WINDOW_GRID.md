# Session record — 2026-07-23 — The window grid, the final layer, and the road to the J-lens

**Participants:** TC (direction, design questions) · Claude Code session (execution, drafting).
**Register:** reporting. This note records the *reasoning trail* of the day, per the
house rule that the journey is part of the record. Results live in
`../experiments/exp_010c_windows/RESULTS_EXP010C.md`; this file records how the
questions found their shape.

---

## The arc

**1. The question arrives.** Building on Stage 1, TC directed: loop tensors not
start-to-finish but between different sets of layers in GPT-2 Medium, toward the
J-space paper's cognitive-workspace hypothesis — ATR first, J-lens second, the
second refined by the first. This became EXP_010c: is Medium's `D` collapse an
artifact of splicing the motor band into the sensory band?

**2. The seeding question (TC).** *Should every run, regardless of layer window,
start with a prompt injection at L0?* Answer: yes — regime control (Stage 1's null
model located the basins in the language-driven regime; a mid-stack seed of raw
embeddings would leave it), comparability (identical natural pass in every arm; the
cut is the only variable), and theory alignment (the workspace is populated by the
sensory layers; the question is what iteration does to an already-populated
workspace state). Recorded as spec §3 with the iteration-1 splice noted as the
treatment itself, not a confound.

**3. The pilot sings.** Baseline: `D` ×5, the Stage 1 funnel intact. Band-centred
windows (10→21, 8→15): the only arms with three distinct terminals, the first
prompt-dependent structure ever seen in Medium, word-like tokens, margins up to
4× baseline. Off-band windows: punctuation funnels. Then the full run's A4 arm
completed: **25/25 prompts across all 7 categories into three basins — `until`,
`forever`, `since` — a temporal-durative cluster**, echoing Small's `till` basin.
The same model that says only `D` through the full stack.

**4. The refinement question (TC).** *Before the deeper characterisation, should
we zero in on the band by examining different in/out layers?* Yes — and the deeper
point folded into the plan: the ATR-derived band edges become a **falsifiable
prediction for the J-lens census**. Two instruments, independent constructions,
should localise the same band if the workspace framing is right. Became
EXP_010c-2's H10 (onset/exit edge sweeps).

**5. The pause (TC).** *"Now I pause, I wonder — does excluding the final layer
introduce a possible issue? Do we have any reason the final layer has special
features we lose by omitting it?"* The honest decomposition that followed:

- The final layers ARE special: the paper's motor regime, token-locked; the
  layers that align the residual stream to the unembedding geometry; `ln_final`
  calibrated to layer-23 statistics.
- But omitting them from the loop is the *treatment*, not a defect — H9's claim
  is precisely that the motor→sensory splice drives the collapse. The pilot
  already showed final-layer inclusion is not *sufficient* for `D` (A3 12→23
  produces `bone`/`"`); the exit-edge sweep (10→21/22/23) isolates *necessity*.
- What is genuinely lost is the **readout**: decoding a layer-21 state through a
  layer-23-calibrated instrument. Hence the decode-via-tail control — run the
  terminal once through the remaining layers and read at 23 — knowing it is
  honest but invasive: the motor band transforms the state on the way out. You
  learn what the state *becomes*, not what it *is*.

**6. The realization (TC).** *"Ahhhh — the J-lens is the key to the final-layer
question. It's what Anthropic must have asked."* The day's design pressure had
re-derived the instrument's raison d'être from the outside. Three readouts, a
ladder of sophistication, all now in the plan:

| # | Readout | What it does | Cost of honesty |
|---|---|---|---|
| 1 | Logit lens at layer j (Stage 1 convention) | Pretends the state is final-layer-shaped | Miscalibrated mid-stack |
| 2 | Decode-via-tail (EXP_010c-2) | Actually runs the state out through the motor band | Observer effect: the tail transforms the state |
| 3 | J-lens | Expected linearized transport into final-layer coordinates, averaged over contexts | Reads "poised to say" without motor commitment; first-order only |

Setting J to identity recovers #1 exactly (paper, p.12); #2 is the single-sample
nonlinear cousin of #3's averaged linear transport. Where the three agree, terminal
claims are solid. Where they diverge, the *pattern* of divergence classifies the
state: motor-committed (1≈2, ≠3), workspace-poised (3 rich, 1 junk), or outside
the verbalizable subspace entirely (all differ).

**Consequence recorded:** the J-lens phase is no longer a follow-on; it is the
**arbiter** of every mid-stack terminal claim in the window-grid programme,
including whether A4's temporal basins are workspace states or shadows on a
miscalibrated exit door.

## Why this note exists

The final-layer worry could have been waved off ("the readout caveat is already
recorded"). Instead it was pushed until it met the instrument built for it. That
is the same motion Stage 1 made when the corpus-fingerprint hypothesis was pushed
until it broke — the house method: take the naive worry seriously, formalise it,
and let it choose the next instrument.
