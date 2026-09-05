# EXP_016 SPEC: lens-coordinate swaps on base GPT-2 Small

**Status: pre-registered.** This file is written and committed before any
swap is run. The only measurements that precede it are the two
clean-accuracy pilots described in section 6, which measure what the
unmodified model can already do and which contain no intervention of any
kind. Their outcomes are recorded here as the gates.

**Register:** `_STAGE2_JSPACE/REGISTER.md`, erratum (f) of 2026-09-05.
This experiment owns the hypothesis identifiers H17, H17a and H17b and the
experiment identifier EXP_016. No other identifier is created here.

**Tracker:** issue #79. **Branch:**
`claude/latent-context-small-llms-u2jdig-exp016`, based on commit `34cc368`.

---

## 1. The question, in plain terms

A recent method called the Jacobian lens claims to find, inside a language
model, a set of directions that stand for particular words. A direction here
is a list of 768 numbers, one for each of the 768 numbers that make up the
model's internal working memory at each point in a sentence. The claim that
matters is not that these directions can be read, but that they can be
written: if you take the amount of the direction for "football" that the
model is currently carrying and exchange it with the amount of the direction
for "cricket", the model should go on to talk about cricket.

EXP_016 asks whether that holds for base GPT-2 Small, a 124-million-parameter
model from 2019 that only continues text and was never trained to follow
instructions. Three separate claims are tested, in increasing order of how
much they would mean if true.

- **H17, the report swap.** The model is asked, by a sentence that stops
  where the answer begins, to name its favourite sport. Exchanging the lens
  direction of the sport it favours with the lens direction of a different
  sport should make it name the different sport.
- **H17a, flexible generalisation.** If the country France is exchanged for
  the country Japan inside the model, then three unrelated questions about
  the country (its capital, its language, its continent) should all move to
  Japan's answers at once. This is a stronger claim than H17, because it says
  the exchanged thing is used by several downstream computations rather than
  merely copied to the output.
- **H17b, intermediate-step surgery.** In a two-step completion where the
  model must first identify an animal from a description and then recall a
  fact about it, exchanging the animal in the middle of the chain should
  change the final fact. This is the strongest claim, because it says the
  exchanged thing is a step in a computation rather than a report.

The point of the design is that a swap can look successful for a boring
reason: any large enough disturbance of the model's working memory changes
what it says next. Every number below is therefore reported against a
control that disturbs the model by the same amount in a direction that means
nothing.

---

## 2. The instrument, and its provenance

**Lens file:** `/home/user/ATR_research/_STAGE2_JSPACE/artifacts/jlens_gpt2_small_neuronpedia.pt`,
SHA-256 digest `d1800a1335ada089ef2e1ec0e4bd4d5bd61e6011eacc31f8618fdb3d10aae762`.
It is a pre-fitted lens published by Neuronpedia, permitted as this
project's instrument by TC on 2026-09-05 (register erratum (f)). Its fit
record is the file `jlens_gpt2_small_neuronpedia.config.yaml` beside it:
fitted on WikiText-103, 277 prompts actually used out of 1000 requested (the
fit stopped early on its own convergence criterion), sequence length 128
tokens, arithmetic in bfloat16, on one NVIDIA B200 graphics processor, on
2026-06-11.

**Reference code:** `anthropics/jacobian-lens` at pinned commit
`581d398613e5602a5af361e1c34d3a92ea82ba8e`, installed editable at
`/home/user/ATR_research/_STAGE2_JSPACE/instrument/jacobian-lens`.

**What the lens holds.** One 768-by-768 matrix `J_l` for each source layer
`l` from 0 to 10, with the final block 11 as the target. Throughout this
file, "layer l" means the residual stream at the output of transformer block
l, which is the transformer_lens hook point `blocks.l.hook_resid_post`. The
lens reads a residual `h` at layer l as the vocabulary scores
`W_U ln_f(J_l h)`, where `ln_f` is the model's final normalisation step and
`W_U` is its unembedding matrix, the same one the model itself uses to turn
its last internal state into word scores.

**Lens vector.** For a vocabulary token `tau` at layer l, the lens vector
`v_tau` is the 768-number direction `J_l^T W_U[:, tau]`, that is, the row of
`W_U J_l` belonging to that token when `W_U` is written with one row per
vocabulary entry. It is the direction in the model's working memory that the
lens counts as evidence for that word. No extra centring, scaling or
whitening is applied to it.

**Model.** Base GPT-2 Small (`gpt2`) loaded from the local Hugging Face
cache through transformer_lens `from_pretrained_no_processing`, so that no
weight folding, weight centring or unembedding centring is applied and the
model's internal numbers are identical to the plain Hugging Face model the
lens was fitted on. This identity was checked before writing this file: the
largest disagreement between the two, in the residual stream at layer 5 on a
test sentence, was 0.00006 on numbers whose own size reaches 2917, and the
largest disagreement in the final word scores was 0.00004. Established.

Every prompt is tokenised with the beginning-of-text token prepended, which
is the transformer_lens default and matches the lens code's own convention.
Position 0 is therefore always that prepended token.

**Versions.** Python 3.11.15, torch 2.14.0 (processor only, no graphics
card), transformers 5.16.1, transformer_lens 3.8.1, numpy 2.4.6. Every
script sets `torch.set_num_threads(1)`.

---

## 3. The intervention

**The swap.** For a source token `s` and a target token `t` at layer `l`,
let `V` be the 768-by-2 matrix whose two columns are the lens vectors `v_s`
and `v_t`. For a residual `h` at a chosen position, the pair of numbers
`c = pinv(V) h` says how much of each of the two lens directions `h`
carries, where `pinv` is the Moore-Penrose pseudo-inverse, the standard
least-squares way of reading off coordinates when the two directions are not
at right angles. The swap writes

    h_patched = h + alpha * V (sigma(c) - c)

where `sigma` exchanges the two entries of `c` and `alpha` is a strength
setting. Because the change lies inside the plane spanned by `v_s` and
`v_t`, everything in the model's working memory at right angles to that
plane is left exactly as it was. The pseudo-inverse is computed from the
2-by-2 Gram matrix `V^T V` with a relative ridge of 1e-8 for numerical
safety.

**Where it is applied.** As a forward hook at `blocks.l.hook_resid_post`,
for every layer `l` in the layer set being tested, at the token positions
named by the position mode. When a layer set contains more than one layer,
the coordinates are read and exchanged independently at each layer, using
that layer's own lens vectors.

**Controls.** Two, both run at every setting.

- **Control A, the registered one: norm-matched random directions.** Two
  random 768-number directions drawn from a standard Gaussian and rescaled
  so that their lengths equal the lengths of `v_s` and `v_t`, put through
  exactly the same swap. Drawn once per item, layer and seed, and reused
  across strengths and position modes, so that the control arms differ from
  one another only by seed. Three seeds.
- **Control B, added here and reported alongside: size-matched random
  directions.** The same random pair as control A, but the resulting change
  to the working memory is rescaled, position by position, to the length of
  the change the real lens swap would make on the same working memory at
  that layer and position. Control A holds the direction lengths fixed;
  control B holds the size of the actual disturbance fixed, which is the
  stricter test of "would any disturbance this large have done it". Three
  seeds. The registered thresholds for H17 are scored against control A;
  control B is reported next to it in every table.

Control B is an addition to the registered design, not a substitution. It is
recorded here as a deliberate strengthening, because the norm-matched
random-direction control can under-disturb the model: two random directions
in 768 dimensions are nearly at right angles to each other and to the
working memory, so the coordinates they read off are small and the change
they write is small. The actual sizes of both changes are recorded in the
results so that the reader can see whether control A was in fact matched in
effect.

**Swept settings.** The layer set, the strength `alpha` and the position
mode are treated as tuned parameters, because Neuronpedia reports that small
models oversteer and need fewer swapped layers than large ones.

- Layer sets, batteries H17 and H17b (14 of them): each single layer from 3
  to 10; the adjacent pairs (5,6), (6,7), (7,8), (8,9); the adjacent triples
  (6,7,8) and (7,8,9).
- Layer sets, battery H17a (10 of them): each single layer from 3 to 10, and
  the triples (6,7,8) and (7,8,9). The shorter list is a compute decision,
  recorded here before the run.
- Strengths: alpha in {0.5, 1, 2}. The paper used 1 for the report swap and
  2 for flexible generalisation; 0.5 is added because of the oversteering
  report.
- Position modes for H17: `all` (every position, including the prepended
  beginning-of-text token, which is the paper's default), `all_no_bos`
  (every position except that prepended token) and `last` (the final
  position only).
- Position modes for H17a: `all` and `all_no_bos`.
- Position modes for H17b: `all_no_bos`; `from_mention`, meaning from the
  first token of the swapped concept's first appearance in the prompt to the
  end; and `answer_only`, the final position alone. The contrast between
  `from_mention` and `answer_only` is what separates a swap that changes a
  step in the model's reasoning from a swap that only changes what it
  reports at the end.

---

## 4. Pre-registered tuning and held-out split

Every battery is split in half, deterministically and before any swap runs,
by taking items in a fixed order and sending alternate items to the tuning
half and the held-out half. For H17 the fixed order is the order of frames
within each category, so both halves are balanced across categories.

The headline number for each hypothesis is produced in two steps. First, the
single combination of layer set, strength and position mode with the highest
success rate is chosen **using the tuning half only**. Ties are broken in
this order: larger gap between the lens arm and control A, then smaller
alpha, then fewer layers, then lowest layer index. Second, that one
combination is scored on the **held-out half**, and that held-out number is
the number the hypothesis is judged on. The full grid on both halves is
reported so that the reader can see how much the choice mattered.

---

## 5. The three batteries

The exact items are committed as JSON beside the code:
`experiments/exp_016_swaps_small/battery_h17.json`, `battery_h17a.json` and
`battery_h17b.json`. They are generated by `build_batteries.py`, which is
this section executed against the pilot output, so they are reproducible.

### 5.1 H17, the report swap

**Frames.** 42 sentences that stop where an answer begins, across four
categories: 12 about a favourite sport, 10 about a favourite fruit, 10 about
a favourite colour, 10 about a favourite animal. Examples: "My favourite
sport is", "Q: Which sport do you like? A:", "Her favourite sport is".

**Category word lists.** Each category has a list of concept names that are
each exactly one GPT-2 token with a leading space: 16 sports, 12 fruits (the
word "melon" was dropped for being two tokens), 14 colours, 20 animals.

**Choosing the source concept.** The pilot found that the two natural
readings of "the model's own top concept" disagree, so both are run as
separate items and the choice between them is part of the tuned selection.

- Rule `lens`: the category member the lens ranks highest at layer 8 at the
  final position. Layer 8 is the middle of the swept range 3 to 10.
- Rule `output`: the category member the unmodified model ranks highest in
  its own next-word prediction.

On the 42 frames these two rules agreed on only 4, which is 10 percent. On
"My favourite sport is" the lens at layer 8 puts " cricket" top among sports
while the model itself would say " football". This disagreement is itself a
finding and is reported.

**Choosing the target concept.** The paper's rule is that the target must
not already be among the model's ten most likely next words. The target is
therefore the category member, other than the source, that the lens at layer
8 ranks highest among those absent from the unmodified model's top ten. By
construction the target's chance of already being in the top five before any
swap is zero out of 84 items, which is the baseline the success rate is
measured against.

**Success.** The target token appears in the top five most likely next words
after the swap. Also recorded: whether the source token left the top five,
the target's rank before and after, and the top five words in full for
qualitative reading.

**Registered rule (H17).** At least 50 percent success on the held-out half
at the tuned setting, against at most 10 percent for control A.

**Item count:** 84, being 42 frames times 2 source rules. 42 tuning, 42
held-out.

### 5.2 H17a, flexible generalisation

**Frames.** One per function, chosen in the pilot by clean accuracy before
any swap was run, out of four candidates each:
`The capital of {X} is the city of`, `Most people in {X} speak`, and
`{X} is a country on the continent of`. The three chosen frames scored 12,
14 and 12 correct out of 25 countries at rank 1, against 0 or 1 out of 25
for the weakest alternative frame of each function, so frame wording matters
enormously at this scale.

**Answer scoring.** On the first token of the answer, so that " North
America" is scored on " North". This follows the reference implementation's
own data files.

**Clean gate.** A country enters the primary set only if the correct
answer's first token is within the unmodified model's top three next words
for all three functions. That leaves 10 countries out of 25: France,
Germany, Greece, Ireland, Italy, Japan, Poland, Portugal, Spain, Sweden. At
rank 1 the gate would leave only 4, which is too few; at rank 5 it would
leave 20. The rank-1 and rank-5 sets are reported as sensitivity checks.

**Pairs.** For each of the 10 gated countries in alphabetical order, three
ordered pairs are formed with the countries 1, 3 and 5 positions later in
that cyclic order, giving 30 primary pairs.

**Extension set.** Nine of the 10 primary countries are European, so their
continent answer is the same word and the continent function cannot show
redirection for pairs among them. A separate extension set is therefore
defined, before the run, from the rank-5 gate: the alphabetically first
gated country on each of the five continents present (Australia, Brazil,
Canada, China, France) and all 20 ordered pairs among them. The extension
set is reported separately and is not part of the headline number.

**Scoreable functions.** A function is scoreable for a pair only when the
two countries' answers differ. A function whose two answers coincide is
recorded as not scoreable for that pair, never as a failure. Pairs with
fewer than two scoreable functions are dropped. 50 pairs survive, 24 of them
with all three functions scoreable.

**Success.** The target country's answer token is in the top five next words
after the swap. Also recorded: the strict version (top one) and whether the
target answer outranks the original answer.

**Registered rule (H17a).** On gated items, one swap redirects at least two
of the three functions at a rate above control A's rate.

**Item count:** 50 pairs, of which 30 primary and 20 extension. 27 tuning,
23 held-out.

### 5.3 H17b, intermediate-step surgery

**Items.** 36 two-step completions were written and piloted, each stating
two facts and then asking a question whose answer requires identifying which
of the two things is meant and then recalling its fact. Example: "Spiders
have eight legs and ants have six legs. The animal that spins webs has",
whose answer is " eight", and where exchanging the spider concept for the
ant concept should produce " six".

**Clean gate.** An item is kept only if the correct answer's first token is
within the unmodified model's top three next words **and** ranks above the
alternative answer's first token. The second condition matters: without it,
an item where the model already prefers the swap's target answer would count
a flip that never happened. 17 of 36 items pass. One of those 17,
`knife-use`, is dropped because the swapped word "knife" does not appear
literally in its prompt (the prompt says "Knives"), so the first-mention
position cannot be located. 16 items remain. At rank 1 the gate would leave
8 items, reported as a sensitivity check.

**Success.** The alternative answer's first token becomes the single most
likely next word, and it outranks the original answer. The strict top-1 form
is used here rather than top-5 because the registered wording says the
answer "changes to the predicted alternative", which is a statement about
what the model would actually say. The top-5 form is reported alongside.

**Registered rule (H17b).** The answer flips to the predicted alternative
more often than under control A. Additionally recorded, and the scientifically
interesting part: whether the `answer_only` position mode works as well as
`from_mention`, which separates a change to the report from a change to the
intermediate step.

**Item count:** 16. 8 tuning, 8 held-out.

---

## 6. What the pilots established, and the gates that follow

Two scripts ran before this file was written, `pilot_clean.py` and
`pilot_clean2.py`. Neither performs any intervention; both only measure the
unmodified model. Their outputs are committed at
`experiments/exp_016_swaps_small/output/pilot_clean.json` and
`pilot_clean2.json`. What they established:

1. Base GPT-2 Small's factual knowledge is thin and extremely sensitive to
   wording. "The capital of France is the city of" puts " Paris" first,
   while "The capital of France is" puts it nowhere in the top five, and
   across 25 countries that frame scored 0 out of 25 at rank 1.
2. On the three chosen frames, 10 of 25 countries are answered within the
   top three on all three functions. The other 15 are recorded as not
   testable at this scale, not as failed swaps.
3. Of 36 two-step completions, 17 are answered within the top three with the
   correct answer ahead of the alternative. The other 19 are recorded as not
   testable at this scale.
4. The lens's own top concept at layer 8 and the model's output top concept
   disagree on 38 of 42 report frames, which is 90 percent.

---

## 7. Compute budget and what counts as a deviation

**Budget.** Processor only, one thread, four cores shared with three other
agents, under 3 gigabytes of peak memory, and the whole experiment aiming to
finish inside about five hours of wall clock. Measured before writing this
file: one forward pass of the model on a batch of 32 copies of an 18-token
prompt, reading out only the final position, takes 1.13 seconds, which is 35
milliseconds per condition. Peak memory during that measurement was 1.7
gigabytes. The planned grids come to roughly 74,000 conditions for H17,
63,000 for H17a and 15,000 for H17b, so approximately 90 minutes of
computation in total.

**Deviations.** Any of the following is written flat into the results
record, with its reason and its effect: a battery run with fewer items than
specified here; a layer set, strength or position mode dropped; fewer than
three control seeds; a scoring rule changed; the tuning and held-out split
changed; the budget overrun and the run scaled down. Nothing in this file is
silently revised. If the grid must be cut for time, the cut is made by
dropping whole settings from the sweep, never by dropping items after seeing
their outcome.

**Verdict language (rule R7).** A hypothesis is SUPPORTED only on completion
evidence, meaning numbers actually produced by a finished run. REFUTED means
the run finished and the registered threshold was missed in the direction of
the null. NOT SUPPORTED means the run finished and the registered threshold
was missed without the evidence being strong enough to call the claim
refuted. UNTESTABLE means the model could not do the clean task often enough
for the swap question to arise.
