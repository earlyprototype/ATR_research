# Session note — 2026-07-27 — The Medium tensor finding, and where it stands

**Register:** session note. Interpretation lives here, fenced as thinking; the
observations belong to their own records, pointed at below. Written at TC's
direction ("sounds important and needs recorded") after the 2026-07-25/26
autonomous run.

## The three observations (each recorded elsewhere; pointers only)

1. **Tensor-level Small-likeness under the `D` mask** (PR #5, EXP_010d
   capstone, H11 line — **NOT yet merged; see status warning below**):
   Medium's full-stack baseline (0→23), whose direct decode is `D` ×25,
   partitions the 25 prompts at the tensor level significantly like GPT-2
   Small's native basin partition — ARI 0.200, permutation p = 0.0009,
   significant at thresholds 0.995/0.999/0.9995. The band window (10→21),
   by contrast, matches Small at chance level at every threshold (H11
   REFUTED: escaping the collapse produces *different* structure, not
   Small's).
2. **The `D` decode is convention-bound** (RESULTS_EXP010C.md, 2026-07-25
   EXP_010c-VARIANTS section, Control B): under natural-injection-layer
   renormalisation the 0→23 arm has 0/25 convergence and `D` is absent;
   measured registered-convention injection at i=0 is ≈218× natural norm.
3. **The `D` decode is slot-bound** (same section, Control A): injection at
   layer 1 instead of layer 0 (1→23) removes `D` even under the registered
   energy convention.

## Interpretation (labelled as such, NOT a finding)

One story consistent with all three: **the `D` collapse was an apparatus
mask, not the phenomenon.** The full-stack loop's tensors carry Small-like
prompt structure throughout; the joint condition {layer-0 slot ∧ ~218×
over-natural energy} produces a readout that hides it behind a single token.
On this story the year-old cross-model claim "Medium funnels to one empty
token" describes the mask, and the window experiments' "escape" manufactures
new structure rather than uncovering the latent one.

Alternatives that also fit, stated so the story can lose: the tensor-level
ARI could reflect shallow prompt features (length, syntax) shared by any
same-tokenizer models rather than basin structure; partition-level ARI is a
necessary-not-sufficient proxy; the capstone ran one seed on one subset with
Small itself only 18/25 converged.

## Registered arbiters

- **EXP_013m (issue #46):** J-lens re-decode of the baseline terminals — if
  verbalizable Small-like content sits under the `D` decode, this is where
  it must show. Gate caveat: the Medium lens validated MARGINAL and the
  census found no coherent band, so a null here is ambiguous between "no
  latent content" and "instrument can't see it."
- **EXP_011m (issue #45):** subspace projection of the terminal tensors.
- A direct test not yet filed anywhere: re-run the capstone's ARI comparison
  on the **natural_i** (natural-energy) terminals from Control B, where no
  `D` mask exists — if the Small-likeness strengthens, the mask story gains;
  if it vanishes, the ARI was an artifact of the loud convention itself.

## Status warning (the reason this note exists)

The source observation (item 1) is recorded ONLY in unmerged **PR #5**,
whose base is a long-merged feature branch from a pre-restructure session.
If that PR is closed unmerged, the capstone result, its artifacts
(`terminals_small_010d.pt`), and the H11 verdict leave the record. Per the
2026-07-26 H11 registry ruling (discussion #37), H11 belongs to this
capstone. **Operator action wanted: rescue PR #5** — retarget/rebase it to
main (or cherry-pick its artifacts + results into a fresh PR) so the
observation the interpretation above rests on is actually in the canonical
record.
