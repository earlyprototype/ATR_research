# Session record — 2026-07-24 — In-fill around the word-forming cells (EXP_010c-3)

**Participants:** TC (direction: "take ownership of issue 6") · Claude Code session (execution, drafting).
**Register:** reporting. Results live in
`../experiments/exp_010c_windows/RESULTS_EXP010C.md` (observations only, house
rule). This note records the reasoning trail and the interpretation, fenced as
thinking per the 2026-07-23 register correction.

---

## The arc

**1. The task.** Issue 6 asked for a cartographic in-fill: run the ATR window
loop at the cells immediately *around* the three that produced whole-word,
prompt-dependent terminals on GPT-2 Medium (8→21, 10→21, 8→15), so the J-lens
phase knows which (injection, extraction) cells to target. Two axes were
unmeasured at single-layer resolution: the odd injection layers 5/7/9/11 (at
extract 21), and the extraction ladder 15/17/19 (at injections 8 and 10).

**2. Pre-registration.** Spec `EXP_010c3_SPEC.md` written and committed before any
run, with hypotheses H12 (injection-zone continuity), H12a (zone edges), H13
(extraction ladder), each with a pre-registered outcome table. Numbering skipped
H11 (reserved by the EXP_010d line, PR #5). One honest deviation recorded up
front: the issue header says "10 windows / 250 runs" but its own table lists 9
(4 injection + 5 extraction = 225 runs). I ran the 9 enumerated, individually
justified cells rather than invent a 10th to hit the header count — padding the
grid with an unjustified cell is exactly the hand-picking the house rule guards
against.

**3. Reproduction discipline.** Same 25-prompt subset (derivation re-verified to
reproduce the committed `prompt_subset.json` byte-for-byte), same S3-mirror model
(same 1,520,013,706-byte `pytorch_model.bin`), same torch/transformer_lens
versions, same gated protocol. A container pause killed the sweep mid-arm-8; the
resume-capable driver skipped the seven committed arms and finished the last two
— the per-arm-commit house rule doing exactly its job.

**4. The result, flat:** none of the 9 in-fill cells reproduced the seed cells'
signature. Six of nine funnel to a *single* terminal repeated across all 25
prompts; the other three are two- or seven-terminal cells with a
fragment/punctuation plurality. Zero cells pass the mechanical
whole-word-prompt-dependent flag.
*(Corrected 2026-07-25 after the PR #33 review — this said "seven of nine",
counting the two-terminal cell I9 inside a strict-single-terminal total.)*

## Interpretation (labelled as such — NOT a finding)

```thinking
The clean headline is a NEGATIVE, and a sharp one: the whole-word,
prompt-dependent behaviour at 8→21 and 10→21 does not spread. It is not a
band you can widen by a layer in any direction —

- Injection axis (extract 21): the flag holds at i=8 and i=10 but the layer
  BETWEEN them (i=9) is a fragment funnel (`oooooooo`/`…`). So 8→21 and 10→21
  are two isolated single-layer islands, not a contiguous 8–10 band. That
  directly refutes the continuity reading I registered as H12's supported
  branch. The odd flanks (7, 11) are single-terminal funnels too — a fragment
  at 7, EOT at 11. The zone is one layer wide at each of 8 and 10.

- Extraction axis: the flag holds only at j=21, the exact extract layer of the
  seed cells. One layer down (j=19) and below, it's gone: single terminals,
  and at three cells (8→17, 10→15, 10→17) the direct decode INVERTS entirely
  under the via-tail control (GOP→since, Fas→the, Bhar→Indian). That inversion
  is the loudest single signal in the sweep. It says the mid-stack logit-lens
  readout at those cells is reading a coordinate system the motor tail does not
  agree with — precisely the miscalibration the decode-via-tail control was
  built to expose, and precisely why the whole programme defers final decoding
  to the J-lens (EXP_013m).

What does "single prompt-independent terminal" mean mechanically? At these
neighbouring cells the loop lands every prompt on the *same* tensor
neighbourhood (few basins for I7/I11/X817/X1015; the decode is one token). That
is closer to the baseline `D` behaviour (one funnel) than to the seed cells'
prompt-dependent spread — except the funnel token is some arbitrary
fragment/name (`oooooooo`, `Bhar`, `Fas`) rather than `D`. So the picture is
NOT "rich band with sharp edges"; it is "two isolated points of
prompt-dependent whole-word structure, embedded in a field of per-cell
single-token funnels." The seed cells look more like isolated resonances than
like the interior of a workspace band.

Caution, same as the 2026-07-23 addendum: this NEGATIVE must not be quietly
converted into a positive claim about the other frame either ("so it's just
resonances, not a workspace"). All we have is: the ATR whole-word signature,
by the logit-lens-at-j readout plus the via-tail control, does not extend off
the two seed cells on this subset/seed. Whether those two cells are workspace
states or shadows on a miscalibrated exit door is exactly what the J-lens is
for. The negative sharpens the target (two points, not a band) and it flags
three cells where the readout itself is untrustworthy — both are gifts to
EXP_013m, not verdicts.

One more honest wrinkle for the map: the mechanical flag also marks i=14 at
extract 21, which I did NOT expect to surface as "whole-word." Its plurality is
`' or'`×15 — a function word — with the rest punctuation/fragments, unlike 8/10
whose pluralities are content words. I left i=14 flagged in the results table
(the rule is mechanical; hiding its flag would be the bias) but recorded the
content-vs-function distinction in the data so the J-lens phase can decide
whether i=14 belongs with 8/10 or is a degenerate comparative funnel. I am not
resolving it here — that is a readout question, and the readout is what's on
trial.
```

## Handoffs

- RESULTS_EXP010C.md gains the dated observations-only section + the updated
  tested-windows map (injection axis at extract 21; both extraction ladders).
- J-lens targets narrowed for `RUNBOOK_JLENS_MEDIUM.md`: (8,21) and (10,21) as
  isolated single-layer islands, plus (8,15); and the via-tail-inverting cells
  (8,17)/(10,15)/(10,17) marked readout-load-bearing for EXP_013m.
- Registered controls unchanged and still owed (anisotropy permutation, seed/
  subset variation, Pythia placebo, hook-point variants) — none run here.

## House lesson

The experiment's most useful output is a negative result and a set of
readout-untrustworthy cells. Reported as such: no band widened, two isolated
cells confirmed at single-layer resolution, three cells flagged where the
instrument disagrees with itself. The outcome table was written before the runs
so the negative could land without being reshaped into a story on either side.
