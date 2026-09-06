# EXP_018 specification: the resonance loop on a small modern chat model

**Status:** pre-registered. This file is committed before the registered run.
The feasibility probe, the memory and timing measurements, and the
natural-loudness recording pass ran first, because the budget below is derived
from their numbers; their artifacts are named in section 3 and committed
alongside this file.

**Register rows:** EXP_018 (the experiment), H19, H19a and H19b (the three
hypotheses). Allocated 2026-09-05 under erratum (f) of
`_STAGE2_JSPACE/REGISTER.md`. Tracker issue: earlyprototype/ATR_research #81.

**Written for:** TC and any reader of this repository, assuming no
machine-learning background, per `docs/voice.md`.

---

## 1. What this experiment asks, in one paragraph

Everything this project has measured about its resonance loop was measured on
GPT-2, a family of language models released in 2019. The loop takes the
running internal state a model builds while reading a prompt, which the field
calls the residual stream, reads that state out at the far end of the network,
rescales it to a fixed size, feeds it back in at the front, and repeats until
the state stops moving. On GPT-2 the loop does two striking things: the
separate word positions all converge to one shared vector within about ten
repetitions, and the settled state decodes to a single word that varies with
where in the network the loop reads and writes. EXP_018 asks whether either of
those survives the move to a modern model. The model chosen is Qwen3-1.7B, a
1.7-billion-parameter chat model released in 2025 that differs from GPT-2 in
four ways that could each change the answer: it encodes word order by rotating
its internal queries and keys inside every layer rather than adding a position
vector once at the input, it normalises its state by size alone rather than by
size and centre, its non-attention blocks multiply two projections together
rather than passing one through a single nonlinearity, and it was trained on
roughly three thousand times more text and then post-trained to hold a
conversation.

## 2. Model, instrument, and provenance

**The model.** `Qwen/Qwen3-1.7B`, the public post-trained chat model, obtained
from the Hugging Face hub with no authentication token. Measured configuration,
read from the loaded model and recorded in
`experiments/exp_018_chat_port/output/probe_natural_norms_float32.json`: 28
layers, a residual stream 2,048 numbers wide, 16 attention heads sharing 8 sets
of keys and values, a vocabulary of 151,936 word pieces, feed-forward blocks of
inner width 6,144 with the SiLU nonlinearity in a gated arrangement,
root-mean-square normalisation, rotary position encoding, and tied embeddings,
meaning the matrix that turns word pieces into vectors on the way in is the
same matrix that turns vectors into word-piece scores on the way out. The
weights are published in bfloat16, a 16-bit number format with about three
decimal digits of precision, and occupy about 3.4 gigabytes.

This model was not the reading note's first choice. The note
(`docs/LATENT_CONTEXT_NOTE_2026-09-04.md` in the lucier repository, section 4)
recommends Gemma-3-270M first. That model is licence-gated and returns HTTP
status 401, meaning "not authorised", without a Hugging Face token, and this
environment has none. Qwen3-1.7B is public, is supported by TransformerLens
3.8.1 under its own name, and has a pre-fitted Jacobian lens published, which
the fallback candidates do not.

**The instrument.** The Jacobian lens for this model published by Neuronpedia
at `neuronpedia/jacobian-lens`, path
`qwen3-1.7b/jlens/Salesforce-wikitext`. A Jacobian lens is a set of matrices,
one per layer, that answer the question "which direction in this layer's state
most raises the model's disposition to say word piece v, now or later,
averaged over ordinary text". Three files were downloaded to
`_STAGE2_JSPACE/artifacts/`, which is not versioned, with these SHA-256
digests, the standard 64-character fingerprint of a file's exact contents:

| File | SHA-256 |
|---|---|
| `qwen3-1.7b_jacobian_lens.pt` | `6fcc79011bd921ffd87612255e2e99950a124fa519470ee44ebaf161c39be9d6` |
| `qwen3-1.7b_config.yaml` | `73d3d397503637d0afccc54daa53ae4a4abfe3803a5b054c92ebd3a4ce9a2680` |
| `qwen3-1.7b_Qwen3-1.7B_convergence.csv` | `bb1738f25f3ca15842924eed2de52c994fa371c817e2ec0e596281dbd7a04d9b` |

**The fit record, established from the downloaded `config.yaml`.** The lens was
fitted on `Qwen/Qwen3-1.7B` itself, that is on the post-trained chat model and
not on a base variant, on 2026-06-11, using WikiText-103 (a corpus of English
Wikipedia articles) at 128 word pieces per prompt. The fit was budgeted for
1,000 prompts and stopped early at 466 prompts when its own convergence
criterion was met, with a final mean relative change of 0.0018 on a scale where
1.0 would mean the lens changed completely with the last prompt added. Its
"identity distance" settled at 0.525 on a scale of 0 to about 1, where 0 would
mean the lens is the trivial one that reads a layer's state as if it were
already the final layer's. The file holds fitted matrices for layers 0 through
26 of the 28; the last layer is absent, which is expected, because the lens for
the final layer would be the identity by construction.

**Versions.** Python 3.11.15, torch 2.14.0 (CPU build), transformers 5.16.1,
transformer_lens 3.8.1, jlens 0.1.0 at pinned commit
`581d398613e5602a5af361e1c34d3a92ea82ba8e`, numpy 2.4.6, scipy 1.17.1. Recorded
again in every results file.

## 3. The loading route, and the memory it needs

**The route.** `TransformerBridge.boot_transformers("Qwen/Qwen3-1.7B",
device="cpu", dtype=torch.bfloat16)` from TransformerLens 3.8.1, without
calling `enable_compatibility_mode()`. The bridge wraps the Hugging Face model
in place instead of building a second copy of it, and it exposes the hook
names this project's engine uses.

**Why not the registered route.** `HookedTransformer.from_pretrained`, which
every previous experiment in this project used, was measured here and does not
fit: it reached 13.9 gigabytes of resident memory and was killed by the
operating system, against a machine total of 15 gigabytes shared with three
other agent sessions. `enable_compatibility_mode()` on the bridge, which folds
the normalisation weights into the neighbouring matrices the way the registered
route does, was also measured and also killed, at 10.5 gigabytes. Both
measurements are established and are recorded here as the reason for the
deviation.

**Measured memory, established.** Booting the bridge in bfloat16 reaches about
1.7 gigabytes and a full loop pass peaks near 3.9 gigabytes, comfortably under
the 9-gigabyte ceiling this session was given. The same route in float32, the
32-bit format the reading note recommends, was also measured and peaked at
10.98 gigabytes, over the ceiling. **Deviation D1, recorded:** the loop
therefore runs on bfloat16 weights. Every tensor operation the loop itself
performs, namely the extraction, the rescale, the gate's cosine comparisons,
the position-collapse metric and the readout's softmax, is carried out in
float32; only the model's own forward arithmetic and the injected tensor's
storage are bfloat16. Section 8 states what that rounding can and cannot do.

**Two round trips that verify the hook wiring, established.** Reading the state
at `blocks.27.hook_resid_post` and decoding it through `ln_final` and `W_U`,
the engine's readout convention, reproduces the model's own output scores
exactly, to a maximum absolute difference of 0.0 on scores whose largest
magnitude is 29.1. Writing the recorded natural state back into
`blocks.0.hook_resid_pre` reproduces the model's own output scores exactly, to
a maximum absolute difference of 0.0. Writing a deliberately wrong state there,
the same tensor with its word positions reversed, changes those scores by 24.75,
which is most of their range. The injection point is therefore exactly the
layer-0 entry, the extraction point is exactly the last block's output, and a
hook that overwrites the residual really does change what the model says.

**Measured speed, established.** One complete gated pass, meaning one forward
pass with the injection hook installed and the deep state cached, takes
2.47 seconds on an 11-word-piece prompt and 3.71 seconds on a
23-word-piece prompt, on one CPU thread. `torch.set_num_threads(1)` is set at
the top of every script, because multi-threaded linear algebra was previously
measured five times slower on this machine. The pass time is dominated by
reading the model's weights out of memory rather than by arithmetic, which is
why a prompt twice as long does not take twice as long.

**The budget that follows.** Cap the loop at **150 iterations** and run the
main arm on all **25** prompts of the registered Small subset. The worst case,
in which no prompt ever settles, is then about 25 x 150 x 2.47
seconds, or about 2.6 hours, and the pilot arm's worst case is about
5 x 150 x 3.71 seconds, or about 0.8 hours. **Deviation D2,
recorded:** the brief for this session proposed a cap of 300 iterations; 300
does not fit the six-hour wall-clock budget at the measured pass time with the
other work this experiment must also do, so the cap is 150, which is more than
ten times the iteration at which the probe saw both test prompts stop moving. The probe measured
the mean-vector cosine at lag 2 already equal to 1.000000 by iteration 10 on
both arms, so the cap is expected not to bind; whether it binds is reported.

## 4. The loop, parameter by parameter

**Injection point:** `blocks.0.hook_resid_pre`, the residual stream as it
enters the first layer. **Extraction point:**
`blocks.27.hook_resid_post`, the residual stream as it leaves the last layer.
These are the registered full-stack window, written for a 28-layer model.

**Rotary positions, and what changes because of them.** In GPT-2 the injection
at layer 0 overwrites the token and position vectors together, so from the
first repetition onward the prompt survives only as a length. Qwen3 re-applies
position inside every attention layer, so the injected tensor stays
position-aware forever. This is established from the architecture and is the
reading note's hazard 1.

**The loudness convention, and its adaptation.** "Loudness" here means the
overall size of the state, computed by squaring every number, adding the
squares, and taking the square root; the field calls this the L2 or Frobenius
norm. The registered GPT-2 convention, `seed_j`, holds the fed-back state at
the size it had where it was read out. The control convention, `natural_i`,
holds it instead at the size the state naturally has at the point it is
injected into, measured on that prompt's own ordinary pass. This experiment
uses `natural_i`, per the reading note's recommendation to run any new model at
natural loudness from the first run rather than as a late control.

The adaptation: **position 0 is excluded from the norm.** On the clean pass of
each prompt the natural entry loudness at `blocks.0.hook_resid_pre` is measured
over word-piece positions 1 and later only. At every repetition the extracted
tensor is rescaled by one scalar, applied to every position including position
0, so that its own loudness over positions 1 and later equals that natural
value. The reason is the reading note's hazard 2: modern models are reported to
put activations hundreds to thousands of times larger than normal at the first
position, and a whole-tensor norm would then be a measurement of position 0
alone. The natural-loudness recording pass measured this directly for this
model and it is reported in the results, including what the whole-tensor norm
would have given, so the reader can judge whether the exclusion mattered.

**Tokenisation.** Qwen adds no start token. Token identifiers are built
explicitly with `add_special_tokens=False` so that nothing is prepended behind
the harness's back, and the word-piece count of every prompt is recorded.

**Arms.**

| Arm | Prompts | Text | Iteration cap |
|---|---|---|---|
| `bare` (main) | 25, the registered Small subset `experiments/exp_010c_windows/output/prompt_subset_small.json`, in file order | the prompt as written | 150 |
| `chat` (pilot) | the first 5 of the same 25, in file order | the same prompt wrapped as one user turn by `tokenizer.apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)` | 150 |

The chat wrapper adds 12 to 15 word pieces of turn markers around each prompt
and ends with an empty thinking block, so the two arms differ in length and in
the special tokens present, and, because positions are re-applied at every
layer, they differ throughout the loop rather than only at the first
repetition. This is the reading note's hazard 5.

**Convergence gate.** The lag-2 gate of `experiments/atr_engine2.py`, function
`run_atr_gated`: at each check, take the cosine between the mean of the
tensor's position vectors now and the same mean two repetitions earlier, and
declare lock-in when that cosine exceeds 0.999 on three consecutive checks. A
cosine of 1.0 means identical direction and 0.0 means unrelated directions. Lag
2 rather than lag 1 is used because a state that alternates between two values
forever, which this project has observed on GPT-2 Small, can never pass a
lag-1 gate but reads 1.0 at lag 2. Checks begin at iteration 10 and run every 2
iterations, so the earliest reportable lock-in is iteration 14. **Deviation D3,
recorded:** the registered `full` tier checks every 10 iterations starting at
iteration 100, which at a 150-iteration cap would allow at most 6 checks and
an earliest lock-in of 120. The finer schedule is the same schedule the
registered `settle` tier of EXP_010c-3b already uses for the same reason.

**Seed:** 42. **Readout:** `ln_final` then `W_U` at the last word-piece
position, the convention of `atr_engine2.get_top_tokens`. Because Qwen ties its
embeddings, `W_U` is the token embedding matrix transposed. The top five word
pieces with their probabilities, the gap in score between the first and second,
and the spread of the whole distribution measured as entropy in nats, are
recorded at lock-in or at the cap.

## 5. Metrics, thresholds, and how each hypothesis is scored

**The position-collapse metric.** Take the tensor at lock-in or at the cap,
treat each word-piece position as a vector, and average the cosine between
every pair of distinct positions. A value of 1.00 means every position holds
the same direction and the state has become spatially uniform. It is recorded
two ways at every iteration, over all positions and over positions 1 and later,
so the reader can see whether position 0 is carrying the result.

**H19, as registered.** "In a rotary-position chat model looped at natural
loudness with position 0 excluded from the norm, the token positions do not
collapse to one vector: the mean pairwise cosine between positions of the
settled tensor stays below 0.99, where GPT-2 Small reaches 1.00 by about
iteration 10." Scored on the main `bare` arm. **SUPPORTED** if the median over
the arm's prompts of the terminal all-position collapse metric is below 0.99.
**REFUTED** if that median is 0.99 or above. The GPT-2 Small comparison value
is the established project observation that `position_similarity` reaches
1.0000 by about iteration 10 (lucier `docs/TECHNICAL.md` section "Observed
Dynamics" and `docs/GPT2_DEEP_DIVE.md` line 437), carrying that record's own
caveat that the archived four-decimal reporting cannot separate 0.9999 from
1.0000. The per-prompt values and the count of prompts on each side of 0.99 are
reported whatever the median does.

**H19a, as registered.** "At natural loudness the modern model's loop settles
(lag-2 gate, cosine 0.999 sustained over three checks) on at least half of the
run prompts within the iteration budget." Scored on the main `bare` arm.
**SUPPORTED** if 13 or more of the 25 prompts reach lock-in at or before
iteration 150. **NOT SUPPORTED** otherwise. The iteration budget is the 150-
iteration cap of section 3 and the count is reported against it.

**H19b, as registered.** "The modern model's settled states have a lower
J-space share on its pre-fitted lens than its ordinary prompt residuals at the
same layer." The J-space of a model, as the "verbalizable workspace" paper
defines it in section 2.3 and appendix A.8, is the set of internal states
reachable as a combination of at most 25 lens directions with non-negative
weights. The J-space share of a state is the squared length of the nearest such
combination divided by the squared length of the state, so it runs from 0,
nothing of the state is expressible that way, to 1, the state lies wholly
inside.

Procedure, fixed in advance:

1. The lens directions at layer l are the rows of `W_U J_l`, each rescaled to
   unit length. Rescaling changes nothing about the answer, because the set of
   non-negative combinations does not depend on how long each direction is, and
   it makes "largest correlation" a comparison of directions. `W_U` here folds
   in the learned gain of the final normalisation, which is a fixed diagonal
   scaling and therefore part of the linear map from state to scores.
2. The nearest point is found by gradient pursuit: repeatedly add the direction
   whose correlation with what is left of the state is largest and positive,
   re-fit all chosen directions by non-negative least squares, and stop at 25
   directions or when nothing correlates positively.
3. Because the vocabulary has 151,936 entries, the search after the first full
   pass is restricted to the 4,096 best-correlating directions. This is an
   approximation and its size is measured: at layers 11 and 18 the first five
   prompts are also scored without the restriction and the largest disagreement
   is reported.
4. States scored: at each layer, three word-piece positions per prompt, namely
   position 1, the middle position, and the last position. Position 0 is
   excluded throughout, as it is from the loudness convention. The settled
   states are obtained by injecting each settled tensor at
   `blocks.0.hook_resid_pre` and reading `blocks.{l}.hook_resid_post`; the
   comparison states are the same layers on an ordinary, non-iterated pass of
   the same prompt.
5. Layers scored: the band layers 11 through 25, which are 38 to 92 percent of
   this model's 28-layer depth, the range the paper reports its workspace
   occupies; plus layers 2 and 5 as early-layer contrast, where the paper
   reports the lens finds little.
6. Control: the lens dictionary is randomly rotated, with two seeds, 2026 and
   4242, and every share is reported beside its two rotated values. Rotating
   the dictionary and rotating the state the opposite way give identical
   shares, so the implementation rotates the states, which is exact and much
   cheaper.
7. Significance: a one-sided paired permutation test on the difference of
   medians, 10,000 draws, seed 42. Each prompt contributes one block of
   position-level shares to each arm; a draw flips a coin per prompt and, on
   heads, swaps that prompt's two blocks.

**SUPPORTED** if, at 8 or more of the 15 band layers, the settled-state median
share is below the ordinary-state median share and the permutation p-value at
that layer is below 0.05. **NOT SUPPORTED** otherwise. **UNTESTABLE** if the
lens file cannot be read or the settled states cannot be produced.

## 6. Artifacts

Under `experiments/exp_018_chat_port/`:

- `qwen_port.py`, the shared harness; `run_exp018.py`, the runner;
  `analyze_jspace.py`, the J-space test; `make_figures.py`, the figures.
- `output/probe_natural_norms_float32.json` and
  `output/probe_natural_norms_bfloat16.json`, the feasibility probe and the
  natural-loudness recording pass in both precisions, with the round-trip
  checks, the memory and timing measurements, and the per-layer per-position
  entry loudness of every prompt in both arms.
- `output/results_bare.json` and `output/results_chat.json`, one record per
  prompt: word-piece count, lock-in iteration, terminal readout, the collapse
  metric and both cosines at every iteration, and every norm the rescale used.
- `output/terminal_states_bare.npz` and `output/terminal_states_chat.npz`, the
  settled tensors in float32.
- `output/jspace_shares_bare.json`, the H19b shares, controls and verdict.
- `output/*.log`, one run log per stage, per rule R5.
- `output/*.png`, small figures.
- `REGISTER_VERDICTS.md`, proposed register rows for the orchestrator's sweep.
- `RESULTS_EXP018.md`, the results record in the operator voice.

## 7. What counts as a deviation

Any departure from sections 2 through 5 is a deviation, is named in the results
record with the letter it is given here, and is stated flat with its reason.
Three are already recorded above: **D1**, bfloat16 weights rather than float32,
with float32 loop arithmetic; **D2**, a 150-iteration cap rather than 300;
**D3**, gate checks every 2 iterations from iteration 10 rather than every 10
from iteration 100. **D4** is reserved for the fallback the brief names, moving
to `Qwen/Qwen2.5-0.5B-Instruct` with an in-house lens, which was not needed.
Further deviations get letters in the order they arise.

## 8. Known limits, stated before the results

The bfloat16 weights of deviation D1 mean each of the model's numbers carries
about three decimal digits. Two things follow, and they should not be confused.
The injected tensor is rounded to bfloat16 once per repetition, which perturbs
its direction by a relative amount of order 0.002, so two states that differ
only by that rounding have a cosine of about 0.999997, far above the 0.999 the
gate asks for; the gate is therefore not at risk from rounding. What rounding
can do is move the trajectory: the map the loop iterates is not linear, so a
perturbation of 0.2 percent per step can in principle land a prompt in a
different settled state after one hundred and fifty steps, and a run at float32 might
settle elsewhere. This is a real limit on the per-prompt terminal words, and it
is inferred from the arithmetic rather than measured here. It is a weaker limit
on the two aggregate claims, H19 and H19a, which turn on whether positions
merge and whether the loop stops moving at all.

Three further limits. First, one model and one prompt subset: nothing here
distinguishes what is true of modern chat models from what is true of Qwen3-1.7B.
Second, the pilot arm has five prompts, which is a direction and not a
measurement. Third, the lens is a third-party artifact fitted on Wikipedia text
by someone else's pipeline, and its own fit record reports an identity distance
of 0.525, meaning it is about half-way between the trivial lens and something
maximally unlike it; this experiment does not validate the lens, it uses it,
and every H19b number inherits that.
