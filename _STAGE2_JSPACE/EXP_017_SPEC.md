# EXP_017: does instruction tuning change where a small language model's feedback loop settles? (pre-registered spec)

**Status:** PRE-REGISTERED. This file is committed before any run that carries
verdict weight. Two steps ran before it and carry none, both recorded in
section 9: the model provenance check (digests, configuration comparison,
conversion check, one instruction response) and a five-prompt timing probe of
the lens fit, which exists only to choose a budget parameter this spec fixes
by rule rather than by value.

**Created:** 2026-09-05. **Tracker issue:** #80. **Branch:**
`claude/latent-context-small-llms-u2jdig-exp017`, base commit `34cc368`.
**Executing line:** `agent:exp017-lamini-twin` (claim posted on issue #80
before this spec).

**Identifiers:** EXP_017, H18, H18a and H18b, all allocated in `REGISTER.md`
under erratum (f) on 2026-09-05. This spec adds no identifier. Proposed
verdict rows for the register are written to
`experiments/exp_017_lamini_twin/REGISTER_VERDICTS.md` for the orchestrator's
sweep, because this session does not edit `REGISTER.md`.

---

## 1. The question in one sentence

GPT-2 Small is the one model in this project whose feedback loop, in which the
model's own internal activity is fed back into its first layer over and over,
settles into a small set of word-shaped resting states rather than collapsing
to a single meaningless token; this experiment asks whether a model with
exactly the same architecture and the same starting weights, but afterwards
trained to follow instructions, still settles into the same resting places.

## 2. Background in plain terms

Every experiment in this project repeats one operation. The model reads a
prompt once in the ordinary way. The internal activity at a chosen depth is
then captured, rescaled to a fixed size, and pushed back into the model at a
chosen shallower depth, and the model is run again. Doing this over and over
either settles the activity into a state that stops changing, which this
project calls a basin or a resting state, or fails to settle within the
iteration budget. GPT-2 Small, a 124 million parameter model with 12 layers,
settles on the full stack into five word-shaped resting states whose readout
tokens are ` prolet`, ` Divine`, ` till`, ` Anarch` and ` solidarity`
(established, Stage 1 record, reproduced 25 of 25 by EXP_010b's reproduction
gate).

The model tested here is `MBZUAI/LaMini-GPT-124M`, which this spec calls the
twin. It is base GPT-2 Small further trained on 2.58 million pairs of an
instruction and a response, so it answers requests instead of merely
continuing text. Its architecture, its tokenizer and its weight layout are
otherwise unchanged. The twin therefore isolates one variable that no earlier
experiment in this project has varied: what post-training does, holding the
architecture and the starting weights fixed.

Two outcomes are informative in opposite directions. If the twin settles in
the same places, the resting states are a property of the architecture and of
the pretraining corpus that post-training does not reach, which would make
them a stable feature of the model family rather than of one checkpoint. If
the twin settles somewhere else, the resting states are a property of a
specific set of weights, and every claim this project has made about GPT-2
Small's landscape is a claim about one checkpoint only. Both readings are
recorded in the outcome tables below before any result is seen.

## 3. Model provenance and verification (measured before this spec, section 9)

**The twin.** `MBZUAI/LaMini-GPT-124M`, downloaded 2026-09-05 from
huggingface.co with no access token, so a public repository only. The
downloaded files and their SHA-256 digests, which are the 64-character
fingerprints that identify a file's exact contents, are recorded in
`experiments/exp_017_lamini_twin/output/model_verification.json` and repeated
in the results record. The weight file actually loaded is
`model.safetensors`, 510,362,696 bytes, SHA-256
`0f3781c76d1b983cd98824c490e08da956a9dc6ca33bdd0a54986ebabedc1d91`, at
repository revision `fc740804ff49`.

**The base.** `gpt2`, the same 124 million parameter model EXP_010b used. Its
`config.json` digest
(`0daed7749b4f02b8f76240d5444551d7b08712dab4d0adb8239c56ba823bb7b4`) and its
`merges.txt` digest
(`1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5`) match the
values recorded in `EXP_010b_SPEC.md` section 2 exactly, which establishes
that the base weights used here are the registered ones. The weight container
differs: EXP_010b recorded `pytorch_model.bin`, and this run loaded
`model.safetensors`, 548,105,171 bytes, SHA-256
`248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707`. The two
containers hold the same parameters in different file formats; the
behavioural check that they agree is the reproduction gate of section 5.

**Configuration comparison, field by field.** The two configuration files
share 41 fields, of which 37 are identical. The four that differ are
`_name_or_path` (the repository name, not architecture), `dtype` (the
recorded storage precision, not architecture), `use_cache` (a generation
convenience flag, not architecture), and `vocab_size`, which is 50,258 in the
twin against 50,257 in the base. Every field that defines the network shape
is identical: 12 layers, width 768, 12 attention heads, 1024 positions, GELU
activation, layer-normalisation epsilon 1e-5, and the same architecture class.

**The one architectural difference, recorded up front.** The twin's vocabulary
has one extra entry, a padding token added for instruction tuning, so its
token-embedding matrix and its unembedding matrix have 50,258 rows against the
base's 50,257. Every other weight tensor has an identical shape. The two
tokenizers produce identical token sequences on the three probe texts tested.
Consequence for this experiment: the twin can in principle read out token
50,257, which has no counterpart in the base. If that token ever appears as a
terminal readout it is reported explicitly rather than silently mapped.

**Conversion check.** Both models are loaded into TransformerLens, the
instrumentation library this project uses to reach inside a model, by the
route the lucier pilot script uses for offline weights:
`AutoModelForCausalLM.from_pretrained(name)`, then
`HookedTransformer.from_pretrained("gpt2", hf_model=hf_model,
tokenizer=tokenizer)`. TransformerLens rewrites the weights into its own
convention, and one part of that rewrite subtracts a constant from every
output score at each position, which changes no prediction because subtracting
the same number from all scores leaves their softmax probabilities unchanged.
The raw score difference is therefore large and meaningless: 26.6 for the twin
and 111.3 for the base, on scores whose spread is of the same order. The
meaningful comparison removes that per-position constant, and it is exact to
within floating-point noise: after centring, the largest absolute score
difference is 2.5e-05 for the twin and 6.1e-05 for the base; the largest
log-probability difference is 3.4e-05 and 4.0e-05; the largest probability
difference is 1.7e-06 and 3.4e-06; and the top-five predicted tokens agree
exactly for both models. On a probability scale from 0 to 1 a difference of
3.4e-06 is three parts in a million, so the converted model and the original
make the same predictions. The conversion is trusted on that evidence.

**Instruction sanity.** The twin was given one instruction in its documented
wrapper template ("Below is an instruction that describes a task. Write a
response that appropriately completes the request." followed by
"### Instruction:", the instruction, and "### Response:"), decoded greedily
for at most 60 new tokens. Its verbatim response, and base GPT-2 Small's
verbatim response to the identical text, are recorded in the results record so
the reader can see for themselves that the twin is a post-trained model and
the base is not.

**How far the weights moved.** The relative Frobenius distance between the two
models' corresponding weight tensors, which is the size of the change divided
by the size of the original on a scale where 0 means identical and 1 means the
change is as large as the original tensor, is recorded per tensor. This is
descriptive context, carries no verdict, and is reported in the results.

## 4. Which convention this experiment reproduces, and one resolved ambiguity

The convention reproduced is the registered GPT-2 Small **full-stack** loop:
inject at `blocks.0.hook_resid_pre`, which is the entrance to layer 0, and
extract at `blocks.11.hook_resid_post`, which is the exit of the last layer.
In `EXP_010b_SPEC.md` section 4 that window, written 0 to 11, is **arm SB**.
The task brief for this experiment named the arm "S1" while giving those two
attachment points; in EXP_010b arm S1 is the 0 to 5 window and does not match
them. **Resolution, recorded before any run:** the named attachment points are
definitive and this experiment reproduces arm SB, the full stack, which is
also what the register's H18 row means by "the registered full-stack
convention". No arm labelled S1 is run.

## 5. Part 1: the loop (H18 and H18a)

### 5.1 Every parameter of the loop

The loop is the registered gated protocol, whose single source of truth is
`run_atr_gated` in `experiments/atr_engine2.py`. Nothing in the engine is
modified. The parameters, all fixed here:

| Parameter | Value | What it means |
|---|---|---|
| injection point | `blocks.0.hook_resid_pre` | where the fed-back activity is written, the entrance to layer 0 |
| extraction point | `blocks.11.hook_resid_post` | where the activity is read, the exit of the final layer |
| `renorm` | `seed_j` | the fed-back activity is rescaled to the size the first ordinary pass had at the extraction point |
| `threshold` | 0.999 | how similar two successive readings must be, on a cosine scale from -1 to 1 where 1 is identical direction, to count as not moving |
| `patience` | 3 | how many consecutive checks must pass before the run is declared settled |
| `check_every` | 10 | a check happens every 10 iterations |
| `check_start` | 100 | checking begins after iteration 100 |
| `gate_lag` | **2** | each check compares an iteration with the one two steps earlier, not one |
| `max_iter` | 1000 | the iteration budget; a run that never settles stops here |
| `capture_terminal` | true | the final state is saved, along with a scan of how similar iterations 1 to 8 steps apart are |
| seed | 42 | `torch.manual_seed(42)`; the protocol is deterministic in evaluation mode, so the seed is recorded for completeness |
| threads | 1 | `torch.set_num_threads(1)` at the top of every script |

### 5.2 The lag-2 gate, and why base GPT-2 Small is re-run here

A gate with lag 1 asks whether the state stopped moving between one iteration
and the next. A gate with lag 2 asks whether it stopped moving between one
iteration and the one two steps later, which a state that flips back and forth
between two values also passes. EXP_010b's arm SB ran with lag 1, and under
that gate the 11 prompts whose readout is ` Divine` never settle: they hold a
lag-1 similarity of 0.68 to 0.73 forever while their lag-2 similarity is
exactly 1.000, the two-step flip this project calls the period-2 bell
(established, `RESULTS_EXP010B.md`).

This experiment is specified with lag 2, so that every prompt that reaches
either a fixed state or a two-step flip is treated as settled and contributes
a well-defined resting state to the partition. That choice means the committed
arm SB terminal states, which were produced under lag 1, are **not** the same
convention. **Therefore base GPT-2 Small is re-run here under the identical
lag-2 convention**, and the primary comparison for H18 is twin against that
matched base run. The committed lag-1 arm SB terminal states
(`experiments/exp_010c_windows/output/terminals_small010b_SB.pt`) give a
secondary reading, reported beside the primary one and carrying no verdict
weight.

### 5.3 The prompts

The 25-prompt subset committed at
`experiments/exp_010c_windows/output/prompt_subset_small.json`, loaded
verbatim in file order, which is the execution authority EXP_010b registered.
It contains 5 prompts chosen as the alphabetically first of Stage 1's 34
` Divine` prompts, plus 20 chosen round-robin across the 7 prompt categories;
under EXP_010b's literal exclusion rule 11 of the 25 are ` Divine` prompts in
Stage 1. Both models are run on exactly these 25 prompts in this order.

Recorded difference from EXP_010d and EXP_015: those experiments used
`prompt_subset.json`, a different 25-prompt list that shares 20 prompts with
this one. Because the adjusted Rand index compares two groupings of the *same*
items, the reference partition for H18 must be over these 25 prompts, which is
why the base model is re-run rather than read from EXP_010d's file. A
secondary comparison against EXP_010d's committed Small reference partition,
restricted to the 20 prompts the two lists share, is reported as context and
carries no verdict weight.

### 5.4 What is recorded per prompt

For each of the 25 prompts and each of the two models: the settling iteration
(the iteration at which the gate locked, or the fact that it never locked
within 1000), whether it settled, the terminal readout's top five tokens with
their probabilities under the engine's `get_top_tokens` convention (apply the
final layer normalisation `ln_final`, multiply by the unembedding matrix
`W_U`, add the bias `b_U`, take the softmax, read the last token position),
the top-one-versus-top-two score margin, the readout entropy, the similarity
scan at lags 1 through 8, and the terminal state itself. Terminal states are
saved as `.npz` files, which are compressed archives of plain number arrays,
holding for each prompt both the mean over token positions and the last
position of the final tensor, and additionally the full final tensor for the
per-layer probe of Part 2.

### 5.5 The entry loudness ratio

Loudness here means the size of the internal activity, measured as the
Euclidean norm of the whole tensor across all token positions, in the model's
own arbitrary internal units where only ratios carry meaning. Following
`LOUDNESS_PROFILE.md`, this experiment records, per prompt and per model, the
loop's fixed re-injection size divided by the natural size of the activity
entering layer 0 on an ordinary un-hooked pass of the same prompt. EXP_010b
measured that ratio at 73.0 on average for base GPT-2 Small's full stack, with
a per-prompt range of 56.2 to 88.4. The twin's ratio is recorded because
post-training can change the scale of a model's internal activity, and if it
does then the twin's loop is being driven at a different relative volume from
the base's, which would be a confound for every comparison in Part 1. The
natural per-layer entry sizes for both models are recorded by the same
un-hooked forward-pass procedure the shared runner uses.

### 5.6 H18: the partition comparison

**Registered wording (`REGISTER.md`):** "The post-trained twin
LaMini-GPT-124M, run under the registered full-stack convention on the
25-prompt Small subset, partitions the prompts like base GPT-2 Small's
terminals (adjusted Rand index above chance under the EXP_010d permutation
test)."

**Method, reusing the registered code unmodified.** Each model's 25 terminal
mean vectors are grouped by `cluster()` in
`experiments/exp_010c_windows/analyze_terminals.py`, which is greedy leader
clustering at cosine threshold 0.999: walk the prompts in order, put each
prompt in the first existing group whose founding member it resembles above
0.999, and otherwise start a new group. That produces one group label per
prompt per model. Agreement between the two labellings is the adjusted Rand
index computed by `adjusted_rand_index()` in
`experiments/exp_010c_windows/compare_small_basins.py`, on a scale where 1
means the two groupings are identical, 0 means no more agreement than a random
regrouping, and negative values mean less agreement than chance. The null is
`permutation_p()` from the same file, exactly as EXP_015 called it: hold the
base labels fixed, shuffle the twin labels 10,000 times with seed 42, and
report the fraction of shuffles reaching the observed index, with the standard
add-one correction. These three functions are imported, not reimplemented.

**Degenerate-partition guard (carried from EXP_015 section 6).** If either
side's 25 terminal states form a trivial grouping at threshold 0.999, meaning
one group holding all 25 prompts or 25 groups of one, there is no substructure
to compare and the primary comparison is recorded as unanswerable at the
registered threshold rather than forced into a verdict row. The threshold
sweep at 0.99, 0.995, 0.999 and 0.9995, which mirrors the EXP_010d sweep, is
then reported descriptively.

**Pre-registered outcome table for H18:**

| Outcome at threshold 0.999 | Reading |
|---|---|
| Adjusted Rand index above 0 and permutation p below 0.05 | **H18 SUPPORTED.** The twin groups the 25 prompts like the base does, more than chance would give. |
| Adjusted Rand index at or below 0, or permutation p at or above 0.05 | **H18 REFUTED.** The twin's grouping is not above chance agreement with the base's. |
| Either side's grouping is trivial at 0.999 | **H18 UNTESTABLE at the registered threshold.** Recorded with the sweep reported descriptively. |

### 5.7 H18a: the readout tokens

**Registered wording (`REGISTER.md`):** "The twin's terminal readout tokens
coincide with base GPT-2 Small's basin tokens on at least half of the 25
prompts."

Half of 25 is 12.5, so "at least half" is at least 13 of 25. Two readings are
computed and both are reported; the first is the verdict-bearing one because
it scores the registered wording, in which "base GPT-2 Small's basin tokens"
means the tokens the record already establishes.

- **Reading A, verdict-bearing.** The set is Stage 1's five basin tokens,
  ` prolet`, ` Divine`, ` till`, ` Anarch` and ` solidarity`. Score: the
  number of the 25 prompts whose twin terminal top-one token is a member.
- **Reading B, secondary.** The set is whatever tokens base GPT-2 Small
  actually produces as terminal readouts in this experiment's own matched
  lag-2 run. Score: the same count against that set. This reading is reported
  beside Reading A and carries no verdict weight, because the registered
  wording names the established basin tokens rather than a set measured here.

Additionally recorded flat, with no verdict: whether the twin has a repeated
terminal token of its own, that is a single token produced on many prompts,
and if so which token and on how many prompts.

**Pre-registered outcome table for H18a:**

| Outcome, Reading A | Reading |
|---|---|
| 13 or more of 25 | **H18a SUPPORTED.** |
| 12 or fewer of 25 | **H18a REFUTED.** |

## 6. Part 2: the lens and the overlap probe (H18b)

### 6.1 What a Jacobian lens is, in plain terms

A language model's final layer turns internal activity into scores over the
vocabulary, so activity at the final layer can be read as words. Activity at
an earlier layer cannot be read that way directly, because the layers in
between still have work to do. A Jacobian lens is a single fixed matrix per
layer, written J_l for layer l, that approximates what the layers in between
do, on average over a corpus, by the best linear map: multiply an early-layer
state by J_l and the result can be decoded with the model's own unembedding as
though it were final-layer activity. The instrument used here is Anthropic's
reference implementation `jlens`, pinned at commit
`581d398613e5602a5af361e1c34d3a92ea82ba8e`, installed editable. In this
instrument "layer l" means the residual stream at the *output* of block l,
that is `blocks.l.hook_resid_post`, and the target is the final block, so for
a 12-layer model the fitted source layers are 0 through 10.

### 6.2 The twin's lens: fit plan and budget

**Reference.** `experiments/jlens_medium/fit_lens.py`, the registered fit,
changed only where the model changes. Fitting corpus: WikiText-103 through
`jlens.examples.load_wikitext_prompts`, the loader the reference used, frozen
at `_STAGE2_JSPACE/artifacts/wikitext_prompts_160.json` (the first 160
training records of at least 600 characters in stream order, a deterministic
derivation). Maximum sequence length 128 tokens. Checkpointed every prompt so
an interruption loses at most one prompt. Single-threaded. The fit runs in the
background while Part 1's loop runs in the foreground.

**Budget rule, fixed here rather than by value.** The registered Medium fit
took 25,456 seconds, about 7 hours 4 minutes, for 100 prompts on a 24-layer
model of width 1024. This experiment's model has 12 layers and width 768, so
the cost per prompt should fall by roughly a factor of 4, but the machine is
shared and the reference fit's thread count is not recorded, so the projection
is not trusted. A five-prompt timing probe measures the per-prompt cost. The
rule, fixed before the probe's number is read: let t be the measured seconds
per prompt. Choose **100 prompts if 100t is at most 9,000 seconds** (2 hours
30 minutes), otherwise **50 prompts if 50t is at most 9,000 seconds**,
otherwise the largest multiple of 10 prompts whose projected time is at most
9,000 seconds. A count below 50 is a recorded deviation. The chosen count, the
measured wall time, the SHA-256 of the saved lens and the number of prompts it
was fitted on are all recorded in the results.

**The `dim_batch` parameter**, which sets how many output dimensions are
computed per backward pass and therefore trades memory against speed, is
chosen by the same probe subject to keeping peak memory under 3 gigabytes. It
changes the arithmetic order, not the estimator, so it is a compute choice and
not a protocol choice.

**If the fit does not finish in budget**, H18b is scored with the base lens on
both sides, that comparison is marked as answering a different question, and
the shortfall is recorded flat as a deviation.

### 6.3 The base's lens

The pre-fitted third-party lens permitted by TC on 2026-09-05 (erratum (f)):
`_STAGE2_JSPACE/artifacts/jlens_gpt2_small_neuronpedia.pt`, SHA-256
`d1800a1335ada089ef2e1ec0e4bd4d5bd61e6011eacc31f8618fdb3d10aae762`, fitted by
Neuronpedia on 277 WikiText-103 prompts in bfloat16 on a graphics processor,
per its committed `config.yaml`. Its digest is re-verified in this run and
recorded.

**A confound recorded before any result.** The two lenses are not fitted alike:
the twin's is fitted here on 50 to 100 prompts in 32-bit precision on a
central processor, and the base's was fitted elsewhere on 277 prompts in
bfloat16 on a graphics processor. The registered H18b wording compares the
twin on its own lens against the base on this Neuronpedia lens, so the
verdict-bearing comparison carries that mismatch. The two cross-checks in
section 6.6 exist precisely to separate a difference caused by the lenses from
a difference caused by the models, and no H18b reading is offered without them.

### 6.4 The J-space share, defined

Following the paper's definition (section 2.3 and appendix A.8) as given in
the task brief. For a state h at layer l:

1. **The lens vectors, also called atoms.** For each vocabulary token v, the
   atom a_v is the vector in the layer's own coordinate space whose inner
   product with a state gives that token's lens score: a_v is row v of
   `W_U^T J_l`, equivalently `J_l^T W_U[:, v]`, where `W_U` is the model's
   unembedding matrix and `J_l` the lens matrix for layer l. There is one atom
   per vocabulary token, so 50,257 atoms for the base and 50,258 for the twin,
   each with 768 numbers.
2. **The cone.** A point is reachable if it is a sum of at most 25 atoms with
   coefficients that are all zero or positive. The set of reachable points is
   a union of cones, one cone per choice of 25 atoms.
3. **The share.** The J-space share of h at layer l is the squared length of
   the reachable point closest to h, divided by the squared length of h. It
   runs from 0, meaning the reachable set explains none of the state, to about
   1, meaning it explains all of it.
4. **How the nearest point is found: gradient pursuit.** Start with the whole
   state as the residual. Repeatedly compute the correlation of every atom
   with the current residual, using atoms scaled to unit length so that the
   comparison is about direction and not about an atom's arbitrary size; add
   the atom with the largest positive correlation to the selected set; refit
   all selected atoms jointly by non-negative least squares, meaning the best
   fit whose coefficients are all zero or positive; and set the residual to
   the state minus that fit. Stop after 25 atoms are selected, or earlier if
   no unselected atom has a positive correlation with the residual. The
   returned share uses the final non-negative least-squares fit.
   **Implementation choice, recorded:** the unit-length scaling at the
   selection step is standard matching-pursuit practice and is stated here
   because the paper's wording does not pin it down.

The implementation is tested on synthetic data before use, and the test is
committed. The tests, fixed here: a state built as a positive combination of 3
known atoms must return a share of 1 to within 1e-4 and must select those 3
atoms; a state made orthogonal to the whole dictionary must return a share
near 0; a state built from 40 atoms must return a share below 1 because only
25 may be used; and the returned share must never exceed 1 by more than
floating-point noise and never be negative.

### 6.5 The states probed, and the control

**The states.** For a terminal tensor produced by Part 1's loop, the per-layer
states are obtained the way the lucier pilot obtained them: inject the
terminal tensor at `blocks.0.hook_resid_pre` and read `blocks.l.hook_resid_post`
for l = 0 through 10, in one forward pass with no further looping. The state
decomposed at each layer is the **last token position**, which is the position
the loop's own readout uses. The mean over token positions is recorded as a
secondary reading where time allows and carries no verdict weight.

**The control.** A random rotation of the whole lens dictionary, with three
seeds (2026, 2027, 2028), reported beside every share. Rotating every atom by
the same random rotation preserves the dictionary's internal geometry, its
lengths and all its angles, and destroys only its alignment with the state, so
it isolates alignment from geometry. Because rotating the whole dictionary by
a rotation R and leaving the state alone gives exactly the same share as
leaving the dictionary alone and rotating the state by R transposed (the
rotation can be factored out of the distance and preserves lengths), the
control is computed the cheaper second way, and the equivalence is asserted
numerically in the committed test.

### 6.6 H18b: the comparison, its cross-checks and its scoring

**Registered wording (`REGISTER.md`):** "Post-training changes the terminal
states' J-space share: the twin's share on a lens fitted to the twin differs
from base's share on the Neuronpedia lens by more than the random-dictionary
control spread, two-sided."

**Band layers.** Layers 5 through 10, six layers, the workspace band this
project reads at. Layers 0 through 4 are computed and reported but are not
part of the verdict.

**The four measurements.** Each is 25 prompts by 11 layers of shares, plus
three rotation controls each:

| Label | States from | Dictionary from |
|---|---|---|
| twin-on-twin | the twin's terminals, per-layer states read in the twin | the twin's own fitted lens with the twin's unembedding |
| base-on-base | the base's terminals, per-layer states read in the base | the Neuronpedia lens with the base's unembedding |
| twin-on-base | the twin's terminals, per-layer states read in the twin | the Neuronpedia lens with the base's unembedding |
| base-on-twin | the base's terminals, per-layer states read in the base | the twin's fitted lens with the twin's unembedding |

The first two are the verdict-bearing pair. The last two are the cross-checks
that separate lens mismatch from model mismatch, and are reported beside the
verdict without changing it.

**Scoring, fixed here.** At each band layer l:

- m_twin(l) is the median over the 25 prompts of the twin-on-twin share, and
  m_base(l) the median of the base-on-base share. The median is used rather
  than the mean because a single prompt cannot move it far.
- The **control spread** at layer l is the largest, over the three rotation
  seeds, of the absolute difference between the two sides' control medians for
  that seed. This is the paired null for the comparison actually being made:
  how far apart the two sides' medians sit when the dictionaries keep their
  geometry but lose their alignment.
- The **permutation p** at layer l is two-sided: pool the 25 twin shares and
  the 25 base shares, shuffle which model each belongs to 10,000 times with
  seed 42, and report the fraction of shuffles whose absolute median
  difference reaches the observed one, with the standard add-one correction.

| Outcome | Reading |
|---|---|
| At least 4 of the 6 band layers have both an absolute median difference larger than that layer's control spread and a permutation p below 0.05 | **H18b SUPPORTED.** Post-training changes the J-space share of the resting states. |
| Fewer than 4 of the 6 band layers meet both conditions | **H18b NOT SUPPORTED.** |
| The twin's lens does not exist because the fit did not finish | **H18b UNTESTABLE as registered**, scored on the base lens for both sides instead and marked a deviation. |

"At least 4 of 6" is this spec's reading of the register's "a majority of the
band layers". The two conditions are required jointly because either alone is
weak: a difference larger than the control spread with a large permutation p
could be noise across prompts, and a small permutation p with a difference
inside the control spread could be dictionary geometry rather than alignment.

## 7. Compute, and what the run costs

Central processor only, 4 cores and 15 gigabytes of memory shared with three
other agent sessions. Every script sets `torch.set_num_threads(1)` on its
first lines. Peak memory is held under 3 gigabytes; the largest single object
is one layer's atom matrix, 50,258 by 768 in 32-bit floating point, about 154
megabytes, and only one layer is held at a time. The loop is expected to cost
roughly 0.2 seconds per iteration per prompt (established from EXP_010b's
recorded arm runtimes), so 25 prompts settling near iteration 120 costs about
10 minutes per model, and 25 prompts running the full 1000 iterations costs
about 85 minutes per model.

## 8. Outputs

All under `experiments/exp_017_lamini_twin/`:

- `output/model_verification.json` and `output/tl_conversion_check.json`, the
  provenance record.
- `output/loop_results_twin.json` and `output/loop_results_base.json`, the
  per-prompt loop records.
- `output/terminals_twin.npz` and `output/terminals_base.npz`, the terminal
  states.
- `output/natural_resid_norms_twin.json` and
  `output/natural_resid_norms_base.json`, the un-hooked entry sizes.
- `output/exp017_partition.json`, the H18 and H18a computation.
- `output/exp017_jspace.json`, the H18b computation.
- `output/*.log`, the run logs, one for the loop and one for the lens fit, as
  rule R5 requires of any commit that adds artifacts.
- `RESULTS_EXP017.md`, the results record, written whichever way the results
  come out, in the operator voice of `docs/voice.md`.
- `REGISTER_VERDICTS.md`, proposed register rows for the orchestrator.

The fitted lens itself is written to `_STAGE2_JSPACE/artifacts/`, which is not
version controlled, and is identified in the results by its SHA-256 digest,
its prompt count and its fit time.

## 9. What counts as a deviation

A deviation is any departure from this spec, and every one is recorded flat in
`RESULTS_EXP017.md` under its own heading, whether or not it changes a verdict.
The following are named in advance as deviations if they occur: a lens fitted
on fewer than 50 prompts; scoring H18b on the base lens for both sides;
running fewer than 25 prompts on either model; any change to
`atr_engine2.py`, to `cluster()`, to `adjusted_rand_index()` or to
`permutation_p()`; any change to the loop parameters in section 5.1; and any
readout that returns the twin's extra vocabulary entry, token 50,257.

The following are recorded here as departures already taken, with their
reasons, and are not later deviations:

1. **The provenance check and the timing probe ran before this spec was
   committed.** The provenance check establishes what the model is and carries
   no hypothesis weight. The timing probe measures one budget number that
   section 6.2 turns into a value by a rule fixed before the number was read.
   Both are recorded in the results with their outputs.
2. **Base GPT-2 Small is re-run rather than read from the committed arm SB
   artifacts**, because those ran under a lag-1 gate and this spec fixes lag 2
   (section 5.2). The committed artifacts supply a secondary reading.
3. **The reference partition is this experiment's own matched base run**, not
   EXP_010d's committed Small reference, because EXP_010d used a different
   25-prompt list and the adjusted Rand index requires the same items on both
   sides (section 5.3). EXP_010d's reference supplies a secondary reading on
   the 20 shared prompts.
4. **The task brief's arm label "S1" is read as arm SB**, the full stack, on
   the strength of the attachment points the brief itself names (section 4).
