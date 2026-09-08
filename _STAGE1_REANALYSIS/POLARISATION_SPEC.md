# EXP_014 — Is the settled basin *specifically* left-political? (pre-registered spec)

**Status:** PRE-REGISTERED — written and committed before any rank was computed.
**Created:** 2026-07-29
**Register:** observations only in the results record; interpretation fenced to a session note.
**Inputs (frozen, read-only):** `lucier-gpt2-activ-tensor-reson-experiments`
`experiments/gpt2_small/output_confidence/{confidence_results.json,converged_tensors.pt}`.
**Compute:** zero forward passes of the ATR loop. One `ln_f → W_U` readout per stored
state (the states are already committed).

---

## 1. The question

The settled state's top-20 readout is
`prolet, Anarch, bourgeois, Marx, comrade, proletarian, socialist, anarchist, the,
congress, movement, Lenin, anarchism, comrades, labour, unity, principles, Socialist,
freedom, principle` — 19/20 socialist/anarchist register.

Established already (F8 / `chordness_formal.md`): this cluster is coherent in W_E at
0.41–0.47 vs a 0.27 null, p = 0.001 under both uniform and frequency-matched nulls.
Semantic adjacency is not in question and is not re-tested here.

**What is untested is the claim the top-20 cannot make on its own: that the basin is
*polarised* — specifically this pole, with rival poles no more available than
arbitrary vocabulary — and that the *loop* is what surfaces it.**

A top-20 list cannot answer this. It shows what won; it says nothing about what lost,
or by how much, or whether it was already winning before the loop ran.

## 2. Why the top-20 cannot be reused as the test set

Any set containing the observed top-20 tokens wins by construction. **Set L is
therefore held-out**: socialist-register vocabulary that does *not* appear in the
observed top-20 of any state. If the basin is a genuine region rather than 20
memorised points, held-out same-pole vocabulary must also rank high.

## 3. Token sets (fixed now, before any lookup)

Any string that is not a single GPT-2 token is **dropped and recorded as dropped** —
not substituted, not repaired.

- **L — held-out socialist register (12):** ` communist`, ` communism`, ` socialism`,
  ` Marxism`, ` Trotsky`, ` Engels`, ` revolution`, ` workers`, ` strike`,
  ` solidarity`, ` union`, ` collective`
- **R — rival pole (12):** ` capitalist`, ` capitalism`, ` conservative`,
  ` conservatism`, ` fascist`, ` fascism`, ` nationalist`, ` nationalism`,
  ` libertarian`, ` monarchy`, ` Reagan`, ` Thatcher`
- **N — political but non-ideological (12):** ` parliament`, ` election`, ` senate`,
  ` mayor`, ` ballot`, ` legislation`, ` governor`, ` referendum`, ` candidate`,
  ` voter`, ` committee`, ` constitution`
- **C — non-political control (12):** ` kitchen`, ` tomato`, ` bicycle`, ` weather`,
  ` hospital`, ` guitar`, ` forest`, ` sandwich`, ` ocean`, ` furniture`, ` camera`,
  ` blanket`

## 4. States measured

Every state is already committed; none is regenerated.

1. The four prompt states that settle in the basin (Lucier, Semantic, Nonsense, Imperative).
2. The `Divine` state (Syntactic) — a settled state of a *different* character.
3. All 15 noise trials, `trial_11` (` Hindu`/` Bombay`) called out separately.
4. **`baseline_iter0` for every prompt — the natural forward pass, before any looping.**

## 5. The primary statistic

For each state and each set: **median rank** of the set's tokens in the full
50257-way readout distribution.

The headline comparison is not a rank but a **shift**: median rank at
`baseline_iter0` versus median rank at the settled state, per set. This is what
separates "the loop surfaced it" from "the model says this anyway."

## 6. Pre-registered readings (fixed before computation)

Applied mechanically; whichever row matches is the recorded outcome.

| Observed | Reading |
|---|---|
| L ≪ R, and R ≈ N ≈ C; L improves iter0→settled far more than R | **POLARISED AND LOOP-SURFACED.** The basin is specifically this pole; rival vocabulary is no more available than arbitrary words; looping is what raises it. |
| L ≪ R < N ≈ C | **Left-tilted political region.** Weaker: the basin is political broadly with a left lean. |
| L ≈ R ≪ N ≈ C | **Polarisation REFUTED.** The basin is ideological vocabulary generally; the Marxist reading is a top-k artifact. |
| L ≈ R ≈ N ≪ C | **Polarisation REFUTED.** The basin is political vocabulary generally. |
| L already top-ranked at `baseline_iter0`, little shift | **Not loop-surfaced.** This is the model's ordinary output, not something ATR reveals. |
| Noise trials show the same L ≪ R pattern | **Not language-driven.** The tilt is a property of the readout geometry, not of looping language. |

## 7. What this experiment cannot decide

Nothing here speaks to *why* the pole is present in the weights — training-corpus
composition, amplification, or sampling are all outside what an activation readout
can see. Corpus-provenance claims are explicitly out of scope and inherit the
Stage 1 corpus-fingerprint burden.
