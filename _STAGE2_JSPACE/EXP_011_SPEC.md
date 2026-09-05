# EXP_011 spec: do GPT-2 Small's settled states sit inside the lens's verbalizable subspace?

**Status: pre-registered.** This file is committed before any measurement of
the experimental states is taken. Everything below (the data, the instrument,
the method, the controls, the numeric thresholds, and the verdict rules) is
fixed at that commit. Anything done differently afterwards is written down as a
deviation in the results record, flat, whether or not it helped.

**Register:** `_STAGE2_JSPACE/REGISTER.md`, experiment row EXP_011, hypothesis
rows H6, H16, H16a and H16b, allocated by erratum (f) on 2026-09-05. Verdict
rows are proposed in `experiments/exp_011_small_overlap/REGISTER_VERDICTS.md`
and applied to the register by the orchestrator in one sweep, so this branch
does not edit the register itself.

**Tracker:** issue #78. **Branch:**
`claude/latent-context-small-llms-u2jdig-exp011`, based on `main` at commit
34cc368.

---

## 1. The question, in one paragraph

This project runs a loop: it takes a language model's own output state, feeds
it back into the model's input, rescales it, and repeats until the state stops
moving. GPT-2 Small, the 124-million-parameter model, settles into five such
resting states, which this record calls basins, named by the single word each
one reads out as: `prolet`, `Divine`, `till`, `Anarch` and `solidarity`. The
question here is whether those resting states are made of the kind of material
the model can put into words. A recent Anthropic paper ("Verbalizable
Representations Form a Global Workspace in Language Models", 2026) proposes
that the states a model can verbalise live in a distinguished part of the
768-dimensional space each layer's activity lives in, and gives an instrument,
the Jacobian lens, for locating that part. If a resting state is largely
outside that part, the loop has settled somewhere the model cannot talk about.
If it is inside, the resting state is ordinary verbalizable content that the
loop has simply amplified. This experiment measures which, against nulls that
say what a state of no particular kind would score.

## 2. Data

Every file below is read-only for this experiment. No file outside the
worktree `/home/user/wt/exp011` is written.

### 2.1 The states to be measured

**Terminology used throughout.** A "terminal state" is the 768-number vector
the loop settles on, read at the last token position. A "layer" is one of
GPT-2 Small's twelve processing stages, and "layer l" always means the
residual stream at the *output* of block l, which TransformerLens names
`blocks.l.hook_resid_post`. "Position collapse" is the loop's observed end
condition in which every token position of the state holds the identical
vector; it is exact to floating-point precision in every file used here
(checked below), so a terminal state is fully described by one 768-number
vector plus the number of positions.

| Family | Source file (read-only) | Keys used | n | What it is |
|---|---|---|---|---|
| `lang` | `/home/user/shared/stage1_frozen/experiments/gpt2_small/output/stage1_results.pt` | `last_vectors[-1]` (iteration 100) and `top_tokens[-1][0][0]` per prompt | 125 | The language arm: the settled state of each of the 125 Stage 1 prompts, read at iteration 100. |
| `noise17` | `/home/user/lucier-gpt2-activ-tensor-reson-experiments/experiments/noise_rerun/output/results.pt` | `results[R###]['result']['terminal_last_vec']`, `seq_len`, `target_frobenius`, `matched_to` | 125 | Stage 1 run 17, the matched-injection-scale noise re-run: 125 Gaussian random tensors, each pair-matched to one prompt's sequence length and starting norm, iterated under the convergence gate. This is the corrected null that lucier finding F4 requires. |
| `nullold` | `/home/user/shared/stage1_frozen/experiments/gpt2_small/output_random_baseline/random_baseline_results.pt` | final snapshot (iteration 100) `last_vector`, `tensor`, and iteration-0 `tensor_norm` | 125 | The ORIGINAL noise arm, the one whose 18 distinct iteration-100 read-out labels are the "18 null-model basins" that H6 names. Known to be mis-scaled (finding F4); used here only because H6's registered wording names it. |
| `clean` | `prompt_library.py` in the lucier repository, `PROMPT_LIBRARY` | all 125 prompts, text | 125 | The same 125 prompts run once through the model with no injection and no iteration: ordinary residuals. Read at the last token position (primary) and averaged over positions (secondary). |
| `phase` | `/home/user/lucier-gpt2-activ-tensor-reson-experiments/experiments/gpt2_small/output_divine_motion/state_divine.pt` | `current_tensor`, `initial_norm` | 3 | The `Divine` period-2 cycle: phase A is the committed iteration-1000 state, phase B is one further loop step rescaled to the loop's starting norm, and the pivot M is their midpoint. Reconstructed exactly as `10_jlens_phase.py` stage 1 does, with that script's own sanity gates. |
| `prolet1000` | `/home/user/lucier-gpt2-activ-tensor-reson-experiments/experiments/gpt2_small/output_divine_motion/state_prolet.pt` | `current_tensor`, `initial_norm` | 1 | The `prolet` attractor at iteration 1000, the object F16 compared the phases against. |
| `noise1000` | `/home/user/lucier-gpt2-activ-tensor-reson-experiments/experiments/gpt2_small/output_divine_motion/state_noise.pt` | `current_tensor`, `initial_norm` | 1 | The pilot's committed converged noise state, carried for continuity with F11 and F16. |
| `dsym` | `/home/user/lucier-gpt2-activ-tensor-reson-experiments/experiments/gpt2_small/output_jlens_phase/phase_states.pt` | reconstructed from `A`, `B` per the stage-3 recipe of `10_jlens_phase.py` | 1 | The symmetric flip axis of the `Divine` cycle, reported descriptively only. |

The five basin representatives H6 needs are drawn from `lang`, not fetched
separately: see §5.1.

### 2.2 How a terminal state becomes twelve per-layer states

A terminal state as stored is the model's *output* (layer 11). The J-space
question is asked at every layer, so each terminal state is turned into twelve
per-layer states by running exactly the loop step that produced it and
recording what the residual stream holds on the way through:

1. Rebuild the full state tensor by repeating the stored 768-number vector
   across the recorded number of token positions. This is exact, not an
   approximation: position collapse is verified to hold to floating-point
   precision (every pair of positions has cosine 1.000000 and identical norm)
   in `converged_tensors.pt` and in the original noise arm's stored tensors,
   and it is independently recorded as `position_similarity` 1.0 for all 125
   language prompts.
2. Rescale the whole tensor to the loop's own starting norm, which is what
   the loop does before every injection (`atr_engine.py` rescales to
   `initial_norm`, frozen at iteration 0). For `lang` this starting norm is
   recovered from run 17's pairing record, whose `target_frobenius` field is by
   construction the paired prompt's iteration-0 Frobenius norm; for `noise17`
   it is that trial's own `target_frobenius`; for `nullold` it is that trial's
   iteration-0 `tensor_norm`; for the iteration-1000 states it is the stored
   `initial_norm`.
3. Splice that tensor into `blocks.0.hook_resid_pre`, overwriting it, run one
   forward pass over a token scaffold of the same length, and record
   `blocks.l.hook_resid_post` at the last position for every l from 0 to 11.
   The scaffold's token identities are irrelevant because the injection
   overwrites the embeddings entirely; the scaffold supplies only the sequence
   length and the attention mask, which is the same convention the engine and
   the pilot scripts use.

**Reconstruction gate, pre-registered as a hard stop.** For a state that is a
true resting point, step 3 must return the state itself at layer 11. The gate
is: the cosine between the layer-11 reading and the stored vector must exceed
0.999 for at least 85 of the 125 language states and at least 85 of the 125
run-17 noise states. States that fail are expected and are not discarded: they
are the period-2 cycles, for which one loop step returns the *other* phase
(cosine about 0.69 to 0.74), and they are counted and reported. If fewer than
85 pass in either arm the reconstruction is wrong and the experiment stops and
says so.

For `clean` there is no injection: the prompt is tokenised with the
model's own convention (a beginning-of-sequence token prepended, which is
TransformerLens's default for GPT-2 and the convention every Stage 1 run used),
run once, and `blocks.l.hook_resid_post` is recorded at the last position
(primary) and averaged over positions (secondary).

## 3. The instrument

**Lens file:** `/home/user/ATR_research/_STAGE2_JSPACE/artifacts/jlens_gpt2_small_neuronpedia.pt`,
SHA-256 `d1800a1335ada089ef2e1ec0e4bd4d5bd61e6011eacc31f8618fdb3d10aae762`,
12,980,477 bytes. The digest is recomputed and recorded at run time.

**Provenance,** from the fit record beside it
(`jlens_gpt2_small_neuronpedia.config.yaml`): fitted by Neuronpedia with
Anthropic's reference code (`jlens`, Apache-2.0, pinned commit
`581d398613e5602a5af361e1c34d3a92ea82ba8e`) on WikiText-103
(`Salesforce/wikitext`, config `wikitext-103-raw-v1`, train split), maximum
sequence length 128 tokens, dimension batch 128, bfloat16, on one NVIDIA B200
graphics processor, generated 2026-06-11. The fit requested 1000 prompts and
stopped early at its own convergence criterion after **277 prompts**, with a
final mean relative change of 0.0016 and a final identity distance of 1.3056.
This experiment did not fit the lens and cannot vouch for the fit beyond that
record.

**What the lens holds:** one 768 by 768 matrix J_l for each source layer l from
0 to 10, eleven in all. The target is the final block, layer 11, so J_11 is the
identity matrix and layer 11's reading is the ordinary logit lens. This
experiment adds that identity itself and labels layer 11 accordingly.

**Lens vectors.** At layer l the lens vectors are the rows of W_U J_l, where
W_U is GPT-2's unembedding matrix of shape 50257 by 768 (the language-model
head weight, tied to the token embedding). There is one lens vector per
vocabulary token, so the dictionary at each layer has 50,257 atoms in 768
dimensions. The final LayerNorm's learned per-coordinate gain is *not* folded
into the lens vectors, which follows the definition this experiment was given;
the consequence is recorded as a limitation rather than repaired.

## 4. The measurement

**J-space.** At layer l, the J-space is the union of the cones spanned by at
most k = 25 lens vectors with non-negative coefficients (the paper's section
2.3 and appendix A.8). A state's J-space component is the nearest point of
that set. Because the set is a union of closed convex cones, the component is
orthogonal to what is left over, so the two squared lengths add to the state's
squared length.

**J-space share.** The squared length of the component divided by the squared
length of the state. It runs from 0 to 1, is unchanged by rescaling the state,
and is the single number every hypothesis below is scored on.

**How the nearest point is found.** Non-negative orthogonal matching pursuit,
which the brief calls gradient pursuit: start with the whole state as residual;
repeatedly add the atom whose unit direction has the largest positive
correlation with the current residual; after each addition re-fit the
non-negative least-squares problem over all selected atoms (`scipy.optimize.nnls`)
and recompute the residual; stop at 25 atoms, or when no unselected atom has a
positive correlation with the residual, or when the residual has fallen below
one part in a million of the state's own length. The implementation is
`experiments/exp_011_small_overlap/jspace.py` and it carries a self-test that
is run and logged before every analysis run: a state built from five known
atoms with positive coefficients must be recovered with share 1.000000 and
exactly those five atoms; a generic random state must not be; batching over
states must not change any answer; and rescaling a state must not change its
share.

**Raw versus centred residual, decided in advance.** The primary arm decomposes
the residual **as it is** (raw). This is the pilot's convention (F11, F16), so
the numbers are comparable with the previous lens work, and it matches the
definition of the J-space as a cone in the residual space itself. A secondary
arm repeats the whole measurement on the **mean-centred** residual, meaning the
average of a state's 768 coordinates is subtracted from each of them. The
reason to run it is that GPT-2's final LayerNorm removes exactly that
component before any logit is formed, so energy along it cannot reach the
readout and arguably should not count toward a "verbalizable" share. If the two
arms disagree on any verdict, the disagreement is reported and the verdict is
recorded as unresolved between conventions rather than settled by picking the
friendlier arm.

## 5. Controls: what a share of this size means

A share is meaningless without knowing what an uninformative dictionary would
have scored, because 25 atoms chosen greedily from 50,257 candidates capture a
non-trivial share of *any* vector. Two controls are pre-registered, both run.

**Control (a), rotated lens.** Apply one uniformly random orthogonal 768 by
768 rotation to every lens vector, and decompose against the rotated
dictionary. The rotation preserves every atom's norm and every pair of atoms'
angle, so it destroys only the alignment between the dictionary and the model's
actual geometry. Seeds 2026, 2027, 2028. This control is computed through the
exact algebraic identity that decomposing a state against a rotated dictionary
gives the same share as decomposing the inverse-rotated state against the
original dictionary; the identity is verified numerically in the self-test and
once more against an explicitly rotated dictionary at one layer during the run.

**Control (b), norm-matched random dictionary.** Replace every lens vector by
an independent Gaussian vector rescaled to that lens vector's norm, same shape
(50,257 by 768). Seeds 4242, 4243, 4244. This is the pilot's control and it is
the chance level H16's registered wording names.

Every share is reported beside the control shares of the same states, and
"above chance" always means above control (b) unless control (a) is named.
Control (a) is the stricter of the two and is reported alongside.

## 6. The workspace band

The paper's workspace band spans roughly 38 to 92 percent of a model's depth.
For GPT-2 Small's twelve layers that is 0.38 times 12 = 4.56 and 0.92 times 12
= 11.04, which rounding inward to stay strictly inside the band gives layers 5
to 11 inclusive. The lens is fitted for source layers 0 to 10, so the band
layers this instrument can transport are **layers 5, 6, 7, 8, 9 and 10, six
layers**, and those six are "the band layers" in every rule below. This is the
same depth-fraction mapping EXP_010c registered for GPT-2 Medium (its section
4), applied to a twelve-layer model. Layers 0 to 4 and layer 11 are measured
and plotted but carry no verdict.

"A majority of the band layers" means at least four of those six.

## 7. Hypotheses, predictions and scoring rules

All tests are one-sided in the direction the hypothesis states, because each
hypothesis names a direction. All permutation tests use 10,000 permutations
with seed 11011 and report the p-value as (1 + number of permuted statistics at
least as extreme) / (1 + 10,000), so the smallest reportable p-value is
0.0001. "Median" always means the median across the states in that family at
that layer.

### 7.1 H6, on its registered wording

> H6: GPT-2 Small's five basin tensors project significantly more onto the
> J-space than the 18 null-model basins.

**The five basin tensors.** The 125 language states carry exactly five distinct
iteration-100 read-out labels: `prolet` (44 prompts), `Divine` (34), `Anarch`
(26), `till` (19) and `solidarity` (2). One representative per basin is used,
chosen as the medoid: within each basin, the state whose average cosine to the
other members of the same basin is highest (for the two-member `solidarity`
basin, the first by prompt identifier, since both have the same average
cosine). Choosing a representative is defensible here because within-basin
states are near-identical (the pilot measured pairwise cosine 0.9987 to 1.0000
among the `prolet` states); the choice is nevertheless made by a fixed rule
written down before the shares are seen.

**The 18 null-model basins.** The original noise arm produces exactly 18
distinct iteration-100 read-out labels across its 125 trials. One representative
per label is used, by the same medoid rule.

**Both arms are read at the same iteration (100) of the same original sweeps**,
which is what makes them comparable; this is the reason the language labels are
taken at iteration 100 rather than from the later convergence-gated
classification, which relabels 11 of the 125 prompts.

**Rule.** At each band layer, compare the five language shares with the 18 null
shares by a one-sided Mann-Whitney U test (language greater). **SUPPORTED** if
the language median exceeds the null median and that test gives p below 0.05 at
a majority of the band layers. **REFUTED** if the null median exceeds the
language median at a majority of band layers with p below 0.05 in that
direction. Otherwise **NOT SUPPORTED**. Reported alongside, not as part of the
rule: the same comparison against control (b), and the same comparison using
all 125 language states against all 125 original-noise states.

### 7.2 H16, the corrected null

> H16: language-prompt terminal states have a higher J-space share than the
> run-17 matched-injection-scale noise terminals at the workspace-band layers,
> above the norm-matched random-dictionary chance level.

**Rule.** At each band layer, compare the median share of the 125 `lang` states
with the median share of the 125 `noise17` states by a label-permutation test
(the 250 states' family labels are shuffled 10,000 times; the statistic is the
difference of medians; one-sided, language greater). **SUPPORTED** if all three
of the following hold: the language median exceeds the noise median at a
majority of the band layers; the permutation p-value is below 0.05 at those
same layers; and at those layers the language median exceeds the median of the
language states' own control (b) shares, pooled over the three seeds.
**REFUTED** if the noise median exceeds the language median at a majority of
band layers with permutation p below 0.05 in that direction. Otherwise **NOT
SUPPORTED**. Robustness reported but not part of the rule: the same test
restricted to the 90 run-17 trials that passed the lag-1 convergence gate, and
the same test under control (a).

### 7.3 H16a, the phase-aware re-test

> H16a: the `prolet` attractor's J-space share exceeds the `Divine` cycle's
> share in both phases (the pilot found the reverse at pilot confidence,
> finding F16).

**Rule.** The `prolet` object is the committed iteration-1000 `prolet`
attractor (`state_prolet.pt`), which is the object F16 compared against.
**SUPPORTED** if the `prolet` share exceeds phase A's share at a majority of
the band layers **and** exceeds phase B's share at a majority of the band
layers. **REFUTED** if phase A and phase B both exceed `prolet` at a majority
of band layers. Otherwise **NOT SUPPORTED**, which is the expected verdict if
the phases straddle `prolet` as F16 reports.

This comparison is between single vectors, not populations, so no permutation
test is available and none is claimed. To keep the reader from over-reading a
tiny gap, every band-layer gap is reported beside the spread of that state's
own control shares across the six control runs (three seeds of each control),
and any gap smaller than that spread is labelled "inside the control spread" in
the results table. The pivot M's share is reported at every layer. The four
pilot-era `prolet` states from `converged_tensors.pt` are reported as a
secondary reading of the same comparison.

### 7.4 H16b, the leaving-verbalizable-directions test

> H16b: ATR terminal states have a lower J-space share than ordinary
> non-iterated prompt residuals read at the same layer on the same lens.

**Rule.** Paired by prompt: for each of the 125 prompts, the difference
(terminal share minus clean-residual share) at each band layer. Tested by a
sign-flip permutation test on the 125 paired differences (10,000 draws,
one-sided, terminal lower). **SUPPORTED** if the median paired difference is
negative and the permutation p-value is below 0.05 at a majority of the band
layers. **REFUTED** if it is positive with p below 0.05 in that direction at a
majority of band layers. Otherwise **NOT SUPPORTED**. Secondary, reported but
not scoring: the same test with the clean residual averaged over token
positions instead of read at the last position.

### 7.5 Descriptive readings, no verdicts attached

1. Per-layer share curves, all twelve layers, for every family and every
   control, as a table and as a figure.
2. Which tokens' lens vectors the decomposition selects, at every layer, for
   the `prolet` attractor and for `Divine` phases A and B and pivot M: the top
   atoms by coefficient times atom norm, with the token strings. The question
   this addresses is whether the mid-layers "say" anything the final layer does
   not.
3. A clamping check on two named states (the `prolet` representative and
   `Divine` phase A): at each band layer, split the state into its J-space
   component and the rest, rescale only the component by a factor in {0, 0.5,
   2}, and read out the result two ways, first through the lens itself and
   second by splicing the modified state back into the model at that layer and
   letting the remaining blocks run. Reported as top-token lists, descriptively.
4. The symmetric flip axis d_sym's share at every layer, for continuity with
   F16, with the caveat that a share is sign-dependent for a direction and both
   signs are reported.

## 8. Compute budget and stopping rule

Measured on this machine before the spec was committed, with
`torch.set_num_threads(1)` as required (multi-threaded linear algebra was
measured about five times slower here): building one layer's 50,257 by 768
dictionary takes 1.3 seconds and 154 megabytes; one state's decomposition at
one layer takes 36 milliseconds; one injected forward pass takes 0.29 seconds;
peak resident memory in the timing probe was 1.05 gigabytes.

The planned run is about 631 states at 12 layers for each of 11 dictionary arms
(one lens and six controls on raw residuals, one lens and three rotation
controls on centred residuals), which is about 83,000 state-layer
decompositions in the raw arms and about 30,000 in the centred arms, an
estimated 68 minutes of decomposition, plus about 4 minutes of forward passes
and about 5 minutes of dictionary construction. The whole experiment is
budgeted at **under 2.5 hours of wall clock**, inside the 5-hour ceiling, with
peak memory under 3 gigabytes.

**Stopping rule.** If the measured cost after the first complete layer implies
more than 4 hours, the centred secondary arm is dropped first, then control (a)
is reduced from three seeds to one, and the reduction is recorded as a
deviation. The primary arm (raw residuals, lens plus control (b) at three
seeds) is never dropped: without it there is no experiment.

## 9. What counts as a deviation

Any of the following, if it happens, is written into the results record under a
heading "Deviations", in plain sentences, with the reason:

1. Any change to a data source, key, family, or state count named in section 2.
2. Any change to the decomposition method, the atom limit of 25, the stopping
   rule, or the raw-versus-centred decision in section 4.
3. Any change to the controls, their seeds, or their number in section 5.
4. Any change to the band definition in section 6.
5. Any change to a scoring rule, threshold, direction, or permutation count in
   section 7, including any test added after the fact (which is reported as
   exploratory and cannot carry a verdict).
6. Failure of the reconstruction gate in section 2.2 or of the self-test in
   section 4, and what was done about it.
7. Any reduction taken under the stopping rule in section 8.

## 10. What this experiment cannot settle

Stated in advance so it is not discovered later. The lens is a third-party fit
that this project did not make and has not validated on its own prompts; its
fit stopped at 277 prompts, and the Medium-scale lens work in this repository
(`RESULTS_JLENS_MEDIUM.md`) records that a lens can pass a readability gate and
still fail to show a coherent band. A low share is therefore ambiguous between
"the state holds nothing verbalizable" and "this instrument cannot see what it
holds", and no result below will be read as ruling latent content out. The
share is a geometric quantity about a cone of at most 25 directions; it is not
a measure of meaning. The 125 language states are one seed of one sweep. The
`Divine` comparison is a comparison of single vectors and carries no
significance test at all.
