# EXP_017: Results Record

**Spec:** `../../EXP_017_SPEC.md`, pre-registered and committed at `f01b4af`
before any run that carries verdict weight.
**Issue:** #80. **Branch:** `claude/latent-context-small-llms-u2jdig-exp017`,
base commit `34cc368`. **Run date:** 2026-09-05.
**Status:** run complete; verdicts recorded below.

Every table of results in this record is generated from the committed JSON
artifacts by `make_tables.py` and reproduced in `output/tables.md`, so no
result here was copied by hand. That guard exists because `RESULTS_EXP010B.md`
carries a dated erratum recording five numbers that were hand-copied into a
results record and matched no committed artifact. Two descriptive tables sit
outside the guard and are named here rather than left to be discovered: the
file digests in section 1.1, read from `output/model_verification.json`, and
the lens summary in section 3.2, whose fit timings are read from the committed
run logs. Both cite their source where they appear. Neither carries a verdict.
The control-adjusted table in section 3.5 was also outside the guard until
2026-09-05, when it was added to `make_tables.py` and the regenerated table was
confirmed to match the record row for row. The tables added in section 3.6 on
2026-09-06 are inside the guard from the start: `make_tables.py` emits every one
of them from `output/exp017_jspace_hfframe.json` and `output/frame_check.json`,
and regenerating on that date changed no existing line of `output/tables.md`,
which is the check that the numbers quoted in sections 2 and 3 still match their
artifacts.

---

## The answer first

**The instruction-tuned twin keeps base GPT-2 Small's grouping of the prompts
and loses base's words entirely.**

The model tested is `MBZUAI/LaMini-GPT-124M`, called the twin throughout: base
GPT-2 Small, a 124 million parameter language model with 12 layers, further
trained on 2.58 million pairs of an instruction and a response, with its
architecture, its tokenizer and its weight layout unchanged. It was run through
the same feedback loop this project has always used, in which the model's own
internal activity at the exit of the last layer is fed back into the entrance
of the first layer over and over until it stops changing.

**H18, whether the twin sorts the prompts into the same families base does:
SUPPORTED.** The twin's 25 end states fall into 5 groups and base's into 7, and
the two groupings agree at 0.1694 on the adjusted Rand index scale, where 1
means identical grouping, 0 means no more agreement than a random regrouping,
and negative values mean worse than random. The permutation test says 97 of
10,000 random regroupings matched or beat that agreement, a p of 0.0097, so the
agreement is above chance.

**H18a, whether the twin's settled states read out as the same words:
REFUTED, 0 of 25.** Base settles on the four Stage 1 basin tokens
` Divine` on 11 prompts, ` prolet` on 10, ` till` on 3 and ` Anarch` on 1. The
twin settles on ` anarchism` on 24 prompts and ` instant` on 1. Not one of the
25 prompts gives the twin a readout that base ever produces. The registered
threshold was at least 13 of 25.

**H18b, whether post-training changes how much of the settled state lies in the
directions a Jacobian lens can name: SUPPORTED, at exactly the pre-registered
threshold, with two caveats that matter more than the verdict.** Four of the six
band layers meet both scoring conditions, and four was the bar; had the bar been
five, the same numbers would read NOT SUPPORTED. The cross-check does not sit on
a threshold: measured on one and the same lens, the twin's
settled states have a higher share than base's at every one of the eleven
layers, by 0.0167 to 0.0889 on a scale from 0 to 1, with the permutation test
at its floor of 0.0001 in all twenty-two of those comparisons, and at every
layer that gap is between 3.1 and 12.7 times larger than the largest effect the
choice of lens produces there. The first caveat is that base's settled
states, at the band layers, lie no closer to the lens's nameable directions
than to a randomly rotated copy of them, so the absolute level of this quantity
cannot be read as evidence of verbalizable content. Section 3.5 gives the
numbers. The second caveat, added on 2026-09-06 after a review of the pull
request, is that the cross-check just quoted is specific to the coordinate
convention this run measured in: the states and the dictionary were read from
the TransformerLens conversion of each model, while the lens matrices they are
built from were fitted against the unconverted Hugging Face model. Recomputing
everything in the fitting convention leaves the H18b verdict at SUPPORTED with
four of six band layers, on layers 5, 6, 7 and 8 instead of 6, 7, 9 and 10, but
it reverses the cross-check, with the twin higher in four of the twenty-two
comparisons instead of all twenty-two. Section 3.6 gives both sets of numbers
and section 6.3 item 5 puts the choice of convention to the operator.

The two Part 1 results point in opposite directions and that is the finding.
The twin's end states still remember which prompts belong together, at a level
chance does not reach, while the words those states decode to have been
replaced wholesale by a single new word. Established from this run: the
prompt-grouping survives instruction tuning and the readout vocabulary does
not.

---

## 1. What ran, and what the models are

### 1.1 Provenance, with digests

A SHA-256 digest is the 64-character fingerprint that identifies a file's exact
contents; two files with the same digest are the same file. Both models were
downloaded from huggingface.co on 2026-09-05 with no access token, so public
repositories only. Full record: `output/model_verification.json`.

| what | file | bytes | SHA-256 | revision |
|---|---|---|---|---|
| the twin, weights loaded | `model.safetensors` | 510,362,696 | `0f3781c76d1b983cd98824c490e08da956a9dc6ca33bdd0a54986ebabedc1d91` | `fc740804ff49` |
| the twin, other container | `pytorch_model.bin` | 510,398,653 | `a99218a6f54834c149da8d175c3dd5b2bef18436eced7136caef52bb02773720` | `5c67c8c03c08` |
| base GPT-2 Small, weights loaded | `model.safetensors` | 548,105,171 | `248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707` | `607a30d783df` |
| base GPT-2 Small, configuration | `config.json` | 665 | `0daed7749b4f02b8f76240d5444551d7b08712dab4d0adb8239c56ba823bb7b4` | `607a30d783df` |
| base GPT-2 Small, tokenizer merges | `merges.txt` | 456,318 | `1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5` | `607a30d783df` |

The last two digests match the values `EXP_010b_SPEC.md` section 2 recorded in
2026-07-26 exactly, which establishes that the base weights used here are the
registered ones. The weight container differs from EXP_010b's, which recorded
`pytorch_model.bin` where this run loaded `model.safetensors`; the two hold the
same parameters in different file formats, and the behavioural evidence that
they agree is in section 2.3.

Versions, recorded: torch 2.14.0 (processor only), transformers 5.16.1,
transformer_lens 3.8.1, jlens 0.1.0 at the pinned commit
`581d398613e5602a5af361e1c34d3a92ea82ba8e`, Python 3.11.15.

### 1.2 The configuration comparison, field by field

The two configuration files share 41 fields and 37 are identical. The four that
differ are `_name_or_path`, which is the repository name, `dtype`, which
records the stored precision, `use_cache`, a convenience flag for text
generation, and `vocab_size`, which is 50,258 in the twin against 50,257 in
base. Every field defining the network shape is identical: 12 layers, width
768, 12 attention heads, 1024 positions, GELU activation, layer-normalisation
epsilon 1e-5, and the same architecture class.

Both models hold 149 weight tensors and no tensor name differs. Exactly two
tensors differ in shape, the token-embedding matrix and the output matrix, both
because of that one extra vocabulary entry, a padding token added for
instruction tuning. The two tokenizers produce identical token sequences on the
three probe texts tested. Recorded consequence: the twin can in principle read
out token 50,257, which has no counterpart in base. **It never did:** neither
model's 25 terminal readouts used it (`output/exp017_partition.json`).

### 1.3 How far instruction tuning moved the weights

Descriptive context, no verdict. The relative Frobenius distance between two
corresponding weight tensors is the size of their difference divided by the
size of the original, on a scale where 0 means unchanged and 1 means the change
is as large as the tensor itself. Over the 147 tensors that have the same shape
in both models the mean is 0.0311, that is about 3 percent. The largest is
0.1902, on the bias of the first layer's first normalisation step; the smallest
is 0.0040, on a bias inside layer 4's feed-forward block. The position
embeddings moved by 0.0140 and the token embeddings, compared over the 50,257
rows the two models share, by 0.0307. The twin's one extra embedding row has
length 2.06 against a mean row length of 3.96 across base's vocabulary, so it
is an unusually short vector, which is what an unused padding entry looks like.

**Reading, marked as inference:** a mean change of about 3 percent is a light
touch, not a retraining. Whatever Part 1 finds is produced by a small
perturbation of base's weights, not by a different model.

### 1.4 The twin answers instructions and base does not

Both models were given the identical text, the twin's documented wrapper
template, decoded greedily for at most 60 new tokens: "Below is an instruction
that describes a task. Write a response that appropriately completes the
request." then "### Instruction:", then "Explain in one sentence why the sky
appears blue.", then "### Response:".

The twin's response, verbatim:

> Blue appears when the sun is shining and the sky is blue.

Base GPT-2 Small's response to the identical text, verbatim:

> \n\nThe sky appears blue.\n\n### Response:\n\nThe sky appears blue.\n\n### Response:\n\nThe sky appears blue.\n\n### Response:\n\nThe sky appears blue.\n\n### Response:\n\nThe sky appears blue.\n\n### Response:

The twin produces one sentence and stops. Base does not answer; it continues
the document, re-emitting the "### Response:" header again and again. The twin
is a post-trained model and base is not. The twin's answer is circular and not
a good explanation, which is what a 124 million parameter instruction model
sounds like; that is not a defect for this experiment's purposes.

## 2. Part 1: the loop

### 2.1 The convention, and the two departures recorded before the run

The loop injects at `blocks.0.hook_resid_pre`, the entrance to layer 0, and
extracts at `blocks.11.hook_resid_post`, the exit of the final layer. The
fed-back activity is rescaled to the size the first ordinary pass had at the
extraction point. A run counts as settled when the similarity between an
iteration and the one two steps earlier stays above 0.999 on a cosine scale
from -1 to 1 for three consecutive checks, with checks every 10 iterations
after iteration 100, and the iteration budget is 1000. Seed 42, one processor
thread. The gated protocol was not reimplemented: `run_loop.py` calls
`run_atr_gated` in `../atr_engine2.py`, the single source of truth, with those
parameters.

Two departures were fixed in the spec before the run, not discovered after it.
First, the task brief named the arm "S1" while giving those two attachment
points; in `EXP_010b_SPEC.md` arm S1 is the 0 to 5 window, and the attachment
points name arm SB, the full stack, so arm SB is what ran. Second, the gate here
compares iterations two steps apart, where EXP_010b's arm SB compared adjacent
iterations. A gate on adjacent iterations can never pass a state that flips
between two values, which is why EXP_010b's 11 ` Divine` prompts never settled
there. Because the committed arm SB end states were produced under the other
gate, **base GPT-2 Small was re-run here under the identical two-step gate**,
and the primary H18 comparison is twin against that matched base run. The
committed end states give a secondary reading in section 2.5.

### 2.2 What happened

Both models settled every prompt, all at iteration 120.

| quantity | base GPT-2 Small | the twin |
|---|---|---|
| prompts that settled, of 25 | 25 | 25 |
| settling iterations seen | 120 for all 25 | 120 for all 25 |
| distinct terminal readout tokens | 4 | 2 |
| mean probability of the top readout token | 0.2653 | 0.1119 |
| entry loudness ratio, mean | 73.0 times | 64.25 times |
| entry loudness ratio, range | 56.23 to 88.42 times | 50.55 to 77.18 times |
| used the twin's extra vocabulary entry | no | no |

Loudness means the size of the internal activity, and the entry loudness ratio
is the size at which the loop re-injects divided by the size the activity
naturally has when it enters layer 0 on an ordinary pass, per
`LOUDNESS_PROFILE.md`. **Base's 73.0 times, ranging 56.23 to 88.42 across
prompts, reproduces the 73.0 times and 56.2 to 88.4 that `RESULTS_EXP010B.md`
committed, to every digit that record carries.** That was not something this
experiment set out to check, and it is the strongest available evidence that
this run's apparatus is the registered apparatus.

The twin re-injects at 64.25 times its natural entry size against base's 73.0
times, about 12 percent quieter. Recorded as a caveat: the two loops are
therefore not driven at exactly the same relative volume, so a small part of
any difference between them could be a volume difference rather than a model
difference. The direction of the gap is small compared with the roughly
220-fold and 73-fold convention effects this project has measured elsewhere.

### 2.3 The conversion into TransformerLens is exact

TransformerLens is the instrumentation library that lets this project read and
write a model's internals. Loading a Hugging Face model into it rewrites the
weights, and one part of that rewrite subtracts a constant from every output
score at each position, which changes no prediction because subtracting the
same number from all scores leaves the probabilities unchanged. The raw score
difference is therefore large and meaningless: 26.62 for the twin and 111.33
for base. Removing that per-position constant leaves the two implementations
agreeing to 2.53e-05 for the twin and 6.10e-05 for base on the score scale, to
3.43e-05 and 4.01e-05 on the log-probability scale, and to 1.73e-06 and
3.43e-06 on the probability scale, which runs from 0 to 1. The top five
predicted tokens are identical for both models. A probability difference of
about two parts in a million is floating-point noise, so the converted model
makes the same predictions as the original. Record:
`output/tl_conversion_check.json`, which `verify_model.py` writes, either on a
full run or on `python3 verify_model.py --conversion-check-only`.

**Regenerated and checked on 2026-09-05, after the run.** The first version of
this experiment's code committed that artifact without committing the code that
produced it, which a review of the pull request caught. The calculation now
lives in `verify_model.py`, one measurement feeding both this artifact and the
provenance record so that the two cannot disagree, and re-running it rewrote
the file byte for byte identically to the committed one. That is a reproduction
check on every number in this section, not merely a promise that the code
exists.

### 2.4 The per-prompt terminal table

Generated from `output/exp017_partition.json`. The probability in brackets is
the readout probability of that token, on a scale from 0 to 1, where a
50,257-token vocabulary gives about 0.00002 to a uniform guess.

| prompt | category | base terminal | base lock | twin terminal | twin lock | base loudness | twin loudness |
|---|---|---|---|---|---|---|---|
| A08_linguistics | Complex | ` Divine` (p 0.514) | 120 | ` anarchism` (p 0.108) | 120 | 71.3x | 59.6x |
| A14_kant | Complex | ` Divine` (p 0.449) | 120 | ` anarchism` (p 0.117) | 120 | 72.8x | 65.3x |
| A15_sartre | Complex | ` Divine` (p 0.418) | 120 | ` anarchism` (p 0.119) | 120 | 77.0x | 67.2x |
| A17_marx | Complex | ` Divine` (p 0.524) | 120 | ` anarchism` (p 0.116) | 120 | 72.4x | 63.9x |
| A21_dickens | Complex | ` Divine` (p 0.430) | 120 | ` anarchism` (p 0.118) | 120 | 77.0x | 70.9x |
| E01_politics | Acronyms | ` prolet` (p 0.099) | 120 | ` anarchism` (p 0.102) | 120 | 65.6x | 58.7x |
| D01_water | Chemical | ` prolet` (p 0.110) | 120 | ` anarchism` (p 0.091) | 120 | 67.9x | 60.0x |
| A01_physics | Complex | ` prolet` (p 0.082) | 120 | ` anarchism` (p 0.104) | 120 | 66.9x | 57.3x |
| B01_napoleon | Narrative | ` prolet` (p 0.084) | 120 | ` anarchism` (p 0.109) | 120 | 65.4x | 58.3x |
| C01_jack_jill | Simple | ` prolet` (p 0.061) | 120 | ` anarchism` (p 0.114) | 120 | 70.6x | 63.1x |
| F01_anger | Vulgarity | ` Divine` (p 0.414) | 120 | ` anarchism` (p 0.117) | 120 | 79.3x | 66.8x |
| G01_punctuation | Wild | ` till` (p 0.243) | 120 | ` anarchism` (p 0.122) | 120 | 87.9x | 73.8x |
| E02_tech | Acronyms | ` Divine` (p 0.468) | 120 | ` anarchism` (p 0.114) | 120 | 71.2x | 60.2x |
| D02_periodic | Chemical | ` till` (p 0.229) | 120 | ` anarchism` (p 0.138) | 120 | 87.8x | 77.2x |
| A02_medical | Complex | ` prolet` (p 0.088) | 120 | ` anarchism` (p 0.102) | 120 | 66.4x | 57.4x |
| B02_wwi | Narrative | ` prolet` (p 0.147) | 120 | ` instant` (p 0.063) | 120 | 56.2x | 50.5x |
| C02_king_cole | Simple | ` prolet` (p 0.072) | 120 | ` anarchism` (p 0.112) | 120 | 67.8x | 61.1x |
| F02_insult | Vulgarity | ` Divine` (p 0.410) | 120 | ` anarchism` (p 0.118) | 120 | 76.7x | 68.7x |
| G02_brackets | Wild | ` prolet` (p 0.065) | 120 | ` anarchism` (p 0.116) | 120 | 74.5x | 69.3x |
| E03_orgs | Acronyms | ` Divine` (p 0.424) | 120 | ` anarchism` (p 0.117) | 120 | 74.2x | 64.0x |
| D03_organic | Chemical | ` till` (p 0.267) | 120 | ` anarchism` (p 0.117) | 120 | 88.4x | 72.0x |
| A03_neuro | Complex | ` Anarch` (p 0.059) | 120 | ` anarchism` (p 0.113) | 120 | 71.6x | 62.3x |
| B03_moon | Narrative | ` Divine` (p 0.472) | 120 | ` anarchism` (p 0.116) | 120 | 73.4x | 65.5x |
| C03_mary_lamb | Simple | ` prolet` (p 0.095) | 120 | ` anarchism` (p 0.113) | 120 | 65.1x | 62.5x |
| F03_frustration | Vulgarity | ` Divine` (p 0.408) | 120 | ` anarchism` (p 0.122) | 120 | 77.6x | 70.8x |

Base's readouts are exactly the Stage 1 distribution, ` Divine` 11 times,
` prolet` 10, ` till` 3 and ` Anarch` once. Under the two-step gate the
` Divine` prompts now settle rather than running to the 1000-iteration cap, and
they settle at iteration 120, the same iteration every other prompt locks at.

**Three reproduction checks, none of which this experiment set out to run.**
Read against `experiments/exp_010c_windows/output/results_small010b_SB.json`,
the committed record of EXP_010b's arm SB:

1. **The readout token agrees on 25 of 25 prompts.** Every prompt gives this
   run's base the same terminal word the committed record gives it, and the set
   of 11 prompts reading ` Divine` is identical, prompt for prompt.
2. **The settling iteration agrees on all 14 prompts that settled in both
   runs**, all at iteration 120. The other 11 are the ` Divine` prompts, which
   ran to the 1000-iteration cap under EXP_010b's one-step gate and settle at
   120 here under the two-step gate, which is exactly the difference the two
   gates are supposed to make and not a discrepancy.
3. **The entry loudness ratio reproduces to every committed digit**, 73.0 times
   with a range of 56.23 to 88.42 against the recorded 73.0 and 56.2 to 88.4.

Taken together these establish that the weights, the engine and the apparatus
in this run are the registered ones, which is what makes the twin's departure
from them interpretable as a property of the twin.

### 2.5 H18: the grouping. **SUPPORTED**

Each model's 25 end states were grouped by the registered method, imported
unmodified from the EXP_010d machinery: walk the prompts in order, put each in
the first existing group whose founding member it resembles above cosine 0.999,
otherwise start a new group. Agreement between the two groupings is the
adjusted Rand index from `compare_small_basins.py`, and the null is the
permutation test from the same file, 10,000 shuffles with seed 42 and the
standard add-one correction, exactly as EXP_015 called it.

At the registered threshold of 0.999 the twin forms 5 groups and base 7. Neither
is trivial, so the degenerate-partition guard did not fire and the comparison
was read as the spec's verdict table directs.

| quantity | value | what it means |
|---|---|---|
| adjusted Rand index | **0.1694** | 0 is chance agreement, 1 is identical grouping |
| permutation p | **0.0097** | 97 of 10,000 random regroupings matched or beat it |
| verdict | **H18 SUPPORTED** | index above 0 and p below 0.05, per the spec's table |

**What the agreement actually consists of.** The number alone does not say what
the twin remembers, so here are the groups. The twin's 5 groups are two large
ones of 11 prompts each and three single-prompt groups. Sorting each twin group
by the word base gives its members:

| twin group | size | the words base gives these same prompts |
|---|---|---|
| 0 | 11 | ` prolet` on 7, ` Divine` on 3, ` Anarch` on 1 |
| 1 | 11 | ` Divine` on 8, ` till` on 2, ` prolet` on 1 |
| 2 | 1 | ` prolet` (D01_water) |
| 3 | 1 | ` till` (D02_periodic) |
| 4 | 1 | ` prolet` (B02_wwi) |

The twin's main division runs roughly along base's own main division. One of the
twin's two large groups holds 8 of the 11 prompts base sends to ` Divine`; the
other holds 7 of the 10 base sends to ` prolet`. Base itself splits those two
families further, into 7 groups in all, and the twin does not follow those finer
splits. Note also that B02_wwi is the only prompt base puts in a group of its
own, it is one of the twin's three single-prompt groups, and it is the single
prompt whose twin readout is not ` anarchism`. **Established from this run:
the twin preserves base's coarse division of the prompts and loses base's finer
one, and the coarse division is what produces the above-chance agreement.**

Threshold sweep, descriptive only, no verdict weight:

| threshold | base groups | twin groups | adjusted Rand index | permutation p |
|---|---|---|---|---|
| 0.99 | 4 | 2 | 0.0659 | 0.15538 |
| 0.995 | 5 | 3 | 0.1329 | 0.02320 |
| 0.999 | 7 | 5 | 0.1694 | 0.00970 |
| 0.9995 | 8 | 6 | 0.1727 | 0.00460 |

The agreement is above chance at three of the four thresholds and rises as the
threshold tightens. At the loosest threshold, 0.99, both groupings have
collapsed almost to nothing, 4 groups and 2, and the agreement is no longer
distinguishable from chance at p 0.155. **Recorded honestly: the result is
threshold-dependent at the loose end**, which is what one expects when the
structure being compared is fine-grained.

**A guard that was wrong in the code and right in this table.** The primary
comparison above refuses to score a degenerate grouping, meaning one with a
single group or with one group per prompt, because such a grouping has no
substructure to compare. The sweep's own copy of that guard asked for either
side to be non-degenerate where it should have asked for both, so a row with one
degenerate side would have carried an adjusted Rand index it should not have.
A review of the pull request found it. **No row above was produced under the
loose condition:** every one of the four rows has both groupings non-degenerate,
from 2 to 6 groups on the twin's side and 4 to 8 on base's, against the 1 and 25
the guard excludes, so not one number in this table changes. The code now
applies the same rule the primary comparison uses and names which side fired
the guard when it does.

Two secondary readings, no verdict weight:

- **Against the committed arm SB end states** from EXP_010b, which used the
  one-step gate: 7 groups there, and the twin agrees with them at 0.1878 with
  a permutation p of 0.0029, slightly stronger than against this run's base.
  This run's two-step base agrees with EXP_010b's one-step base at 0.6809 with
  a permutation p of 0.0001, so changing the gate moves the grouping
  substantially but not beyond recognition.
- **Against EXP_010d's committed Small reference partition**, restricted to the
  20 prompts the two 25-prompt lists share: EXP_010d's reference forms 7
  groups; this run's base agrees with it at **0.9805** with a permutation p of
  0.0001, and the twin at 0.2538 with p 0.0049. An agreement of 0.9805 on a
  scale whose maximum is 1 means this run's base reproduces the registered
  reference partition almost exactly, which is a second unasked-for
  reproduction check alongside the loudness ratio.

### 2.6 H18a: the words. **REFUTED**

The registered wording asks whether the twin's terminal readout tokens coincide
with base GPT-2 Small's basin tokens on at least half of the 25 prompts, which
is at least 13.

- **Reading A, verdict-bearing.** The token set is Stage 1's five established
  basin tokens, ` prolet`, ` Divine`, ` till`, ` Anarch` and ` solidarity`.
  **Hits: 0 of 25. H18a REFUTED.**
- **Reading B, secondary.** The token set is whatever base produced in this
  run's own matched arm, which is ` Divine`, ` prolet`, ` till` and ` Anarch`.
  Hits: 0 of 25, which would also be refuted.

The twin's own repeated terminal is ` anarchism` on 24 of 25 prompts, with
` instant` on the remaining one, B02_wwi. On no prompt does the twin's top
token equal base's top token. On 7 of the 25 prompts one of Stage 1's five
basin tokens still appears somewhere in the twin's top five, so the old
vocabulary has not vanished from the neighbourhood, it has stopped winning.

**Marked as observation, not established mechanism:** ` anarchism` and base's
` Anarch` are lexically adjacent, and base produces ` Anarch` on exactly one
prompt. Whether the twin's single funnel is a relative of base's smallest basin
or a coincidence of the vocabulary is not something this experiment can decide.
The J-space probe in section 5 speaks to the geometry, not to this question.

### 2.7 The oscillation is gone

Recorded flat. The similarity scan reports, for each gap k from 1 to 8, the
mean similarity between iterations k apart, averaged over all 25 prompts, on a
cosine scale where 1 means identical direction.

| gap between iterations | base GPT-2 Small | the twin |
|---|---|---|
| 1 | 0.8574 | 1.0000 |
| 2 | 1.0000 | 1.0000 |
| 3 | 0.8574 | 1.0000 |
| 4 | 1.0000 | 1.0000 |
| 5 | 0.8574 | 1.0000 |
| 6 | 1.0000 | 1.0000 |
| 7 | 0.8574 | 1.0000 |
| 8 | 1.0000 | 1.0000 |

Base alternates exactly: adjacent iterations sit at 0.857 and iterations two
apart at exactly 1.000, at every even gap. That is the period-2 bell of the
Stage 1 record, the state flipping between two values forever, and it is driven
by the 11 ` Divine` prompts. The twin sits at 1.000 at every gap, so its
settled states are still fixed points with no flip at all. **Established from
this run: instruction tuning removed the oscillation as well as the words.**

## 3. Part 2: the lens and the overlap probe

### 3.1 What is being measured, in plain terms

A model's final layer turns internal activity into scores over the vocabulary,
so activity at the final layer can be read as words. Activity at an earlier
layer cannot be read that way directly, because the layers in between still
have work to do. A Jacobian lens is one fixed matrix per layer that stands in
for what those middle layers do on average: multiply an early-layer state by it
and the result can be decoded with the model's own output matrix as though it
were final-layer activity. The instrument is Anthropic's reference
implementation `jlens`, pinned at commit
`581d398613e5602a5af361e1c34d3a92ea82ba8e`.

The lens gives one direction per vocabulary token, called an atom here: the
direction in which a state at that layer must point to raise that token's
score. There are 50,257 of them for base and 50,258 for the twin, each with 768
numbers. **The J-space share of a state** is how much of that state can be
built out of at most 25 of those directions, using only positive amounts: take
the closest reachable point to the state, and divide its squared length by the
state's squared length. It runs from 0, meaning none of the state is
explained, to 1, meaning all of it is. Loosely, it asks how much of the settled
state is made of things the model could say.

The search for the closest reachable point is gradient pursuit: repeatedly add
the direction that correlates most with what is left over, refit all chosen
directions together with the constraint that their amounts stay positive, and
stop at 25 directions. `jspace.py` carries the synthetic tests the spec fixed
in advance and all of them pass: a state built from three known directions is
recovered exactly and those three are the ones chosen; a state at right angles
to the whole dictionary returns zero; a state built from forty directions never
gets more than twenty-five; shares stay inside zero to one; the fitted point is
at right angles to the leftover, which is the defining property of this kind of
projection; and rotating the whole dictionary gives the same answer as rotating
the state the opposite way, which is the identity that makes the control cheap.

**The control.** Beside every share sits the same measurement with the whole
dictionary randomly rotated, on three seeds. Rotating every direction by the
same rotation keeps all the lengths and all the angles between them and
destroys only their alignment with the state, so the control separates "the
dictionary is well aligned with this state" from "a dictionary of this shape
explains anything of this size".

**The states.** For each settled end state, the state is injected at the
entrance to layer 0 and the exits of layers 0 through 10 are read in one
forward pass with no further looping, the way the lucier pilot read per-layer
states. The state decomposed at each layer is the last token position, which is
the position the loop's own readout uses.

### 3.2 The two lenses, and how alike they are

| | base's lens | the twin's lens |
|---|---|---|
| source | Neuronpedia, pre-fitted, permitted by TC on 2026-09-05 | fitted here |
| SHA-256 | `d1800a1335ada089ef2e1ec0e4bd4d5bd61e6011eacc31f8618fdb3d10aae762`, re-verified in this run | `8ccf4b7a12b1c4f3bd56df5cbe2ec8228260ca676618c7fa09f7f2b83595c280` |
| fitting prompts | 277 WikiText-103 prompts | 40 WikiText-103 prompts, committed as `wikitext_prompts_160.json` and used in list order |
| precision and hardware | bfloat16 on a graphics processor | 32-bit on one processor thread |
| fit time | not recorded by its publisher | 8,253 seconds, that is 2 hours 18 minutes, for the 35 prompts that invocation computed, so 235.8 seconds each |

**How the prompt count was chosen.** The spec fixed a rule before the
measurement: measure the cost per prompt on a five-prompt probe, then take 100
prompts if that fits inside a 9,000 second cap, else 50, else the largest
multiple of ten that fits. The probe measured 221.0 seconds per prompt, so the
rule chose 40. Forty is below fifty, which the spec named in advance as a
deviation, and it is recorded as one. The fit then ran to completion at 40
prompts without needing the deadline fallback.

**How the fit stayed inside the cap, stated exactly.** The 40-prompt lens was
assembled by two invocations of the instrument, because the fit resumed from
the checkpoint the five-prompt timing probe had already written. Both
invocations logged every prompt they computed, in `output/fit_probe_db16.log`
and `output/fit_twin.log`, which is where these figures come from. The probe
computed prompts 1 to 5 in 1,105 seconds. The fit invocation computed prompts 6
to 40, that is 35 prompts, in 8,253 seconds, which is 92 percent of the 9,000
second cap and inside it. Adding the two invocations together, the lens cost
9,358 seconds of one processor thread in total, which is 4 percent above the
cap; the cap governs the fit run, and the probe was a separate step the spec
records in its own section 9.1, so nothing here breaks the rule, but a reader
comparing totals should see both figures. **Correction, marked as a retraction
of an earlier sentence in this record:** the first version of this section
reported the fit at "206.3 seconds per prompt", which was the invocation's
8,253 seconds divided by the lens's 40 prompts. That was wrong as a cost rate,
because 5 of those 40 prompts were computed by the probe and cost this
invocation nothing. The measured cost is 235.8 seconds per prompt actually
computed, and section 6.3's projection of a 100-prompt fit is corrected with
it.

**How well converged the twin's lens is.** The instrument reports, after each
prompt, how much the running average still moved, as a fraction of its own
size. The twin's lens ended at 0.0179. The registered Medium fit ended at
0.00915 after 100 prompts, and the Neuronpedia lens reports 0.0016 after 277.
**Stated plainly: the twin's lens is about twice as far from settled as the
Medium reference and about eleven times as far as the lens it is compared
against.** That is the single largest weakness in this part of the experiment,
and sections 3.4 and 5 are written to bound it rather than to talk past it.

**How alike the two lenses are.** This matters because the registered
comparison pits one against the other. Measuring the similarity between the two
models' lens matrices at each layer, on a cosine scale from -1 to 1 where 1
means identical:

| layer | base lens size | twin lens size | similarity between the two lenses |
|---|---|---|---|
| 0 | 23.464 | 23.549 | 0.9066 |
| 1 | 28.266 | 28.887 | 0.9062 |
| 2 | 31.898 | 33.654 | 0.9058 |
| 3 | 34.015 | 36.401 | 0.9053 |
| 4 | 37.345 | 39.263 | 0.9137 |
| 5 | 42.099 | 43.698 | 0.9218 |
| 6 | 43.926 | 44.661 | 0.9397 |
| 7 | 44.609 | 45.508 | 0.9534 |
| 8 | 44.763 | 46.403 | 0.9681 |
| 9 | 43.778 | 44.976 | 0.9753 |
| 10 | 47.433 | 48.344 | 0.9858 |

The two lenses are highly similar and grow more so with depth, from 0.91 at the
first layer to 0.99 at the last, and their overall sizes agree to within a few
percent at every layer. Two lenses fitted on different corpora, at different
precisions, on different hardware, on two models whose weights differ by about
3 percent, come out nearly the same instrument.

### 3.3 The result

| layer | twin median share | base median share | absolute difference | control spread | permutation p | both conditions met | in the band |
|---|---|---|---|---|---|---|---|
| 0 | 0.3753 | 0.3106 | 0.0647 | 0.0167 | 0.0001 | yes | no |
| 1 | 0.3751 | 0.3064 | 0.0687 | 0.0342 | 0.0001 | yes | no |
| 2 | 0.3074 | 0.2520 | 0.0554 | 0.0301 | 0.0001 | yes | no |
| 3 | 0.3017 | 0.2519 | 0.0498 | 0.0115 | 0.0001 | yes | no |
| 4 | 0.2874 | 0.2517 | 0.0357 | 0.0228 | 0.0001 | yes | no |
| 5 | 0.2851 | 0.2538 | 0.0313 | 0.0325 | 0.0001 | **no** | yes |
| 6 | 0.2872 | 0.2570 | 0.0303 | 0.0282 | 0.0001 | yes | yes |
| 7 | 0.3055 | 0.2700 | 0.0355 | 0.0215 | 0.0001 | yes | yes |
| 8 | 0.3069 | 0.2861 | 0.0209 | 0.0254 | 0.0001 | **no** | yes |
| 9 | 0.3221 | 0.2964 | 0.0258 | 0.0197 | 0.0001 | yes | yes |
| 10 | 0.3133 | 0.2949 | 0.0185 | 0.0059 | 0.0001 | yes | yes |

The median is the middle value over the 25 prompts. The control spread is, for
each layer, the largest gap between the two sides' medians when both
dictionaries are randomly rotated, taken over the three rotation seeds; it says
how far apart the two sides sit when the dictionaries keep their shape and lose
their aim. The permutation p asks how often a random reassignment of the 50
measured shares between the two models produces a median gap at least as large
as the one observed, over 10,000 reassignments with seed 42; 0.0001 is the
floor of that test, meaning no reassignment out of 10,000 reached it.

The twin's share is larger than base's at every one of the eleven layers, and
the permutation p is at the floor at every one. The condition that fails at
layers 5 and 8 is the control condition: at layer 5 the gap is 0.0313 against a
control spread of 0.0325, and at layer 8 it is 0.0209 against 0.0254.

### 3.4 The cross-checks, which answer the lens-mismatch worry

The registered comparison measures the twin on one lens and base on another, so
a difference could in principle come from the lenses rather than the models.
The two cross-checks settle it. Reading the median share for each combination of
whose states and whose lens:

| layer | base states, base lens | base states, twin lens | twin states, base lens | twin states, twin lens |
|---|---|---|---|---|
| 0 | 0.3106 | 0.2864 | 0.3962 | 0.3753 |
| 1 | 0.3064 | 0.2930 | 0.3842 | 0.3751 |
| 2 | 0.2520 | 0.2490 | 0.3124 | 0.3074 |
| 3 | 0.2519 | 0.2555 | 0.3031 | 0.3017 |
| 4 | 0.2517 | 0.2431 | 0.2939 | 0.2874 |
| 5 | 0.2538 | 0.2517 | 0.2903 | 0.2851 |
| 6 | 0.2570 | 0.2529 | 0.2931 | 0.2872 |
| 7 | 0.2700 | 0.2691 | 0.3028 | 0.3055 |
| 8 | 0.2861 | 0.2867 | 0.3113 | 0.3069 |
| 9 | 0.2964 | 0.2905 | 0.3323 | 0.3221 |
| 10 | 0.2949 | 0.2929 | 0.3115 | 0.3133 |

**Swapping the lens barely moves anything; swapping the model moves a lot.**
The two effects are separated directly. The **model effect** holds the lens
fixed and swaps whose settled states are decomposed. The **instrument effect**
holds the states fixed and swaps the lens. Both are absolute differences of
medians over the 25 prompts, and the model effect carries the same two-sided
permutation test, 10,000 reassignments with seed 42.

| layer | model effect on base's lens | model effect on the twin's lens | instrument effect on twin states | instrument effect on base states | smallest model effect over largest instrument effect |
|---|---|---|---|---|---|
| 0 | 0.0856 (p 0.0001) | 0.0889 (p 0.0001) | 0.0209 | 0.0243 | 3.5 times |
| 1 | 0.0778 (p 0.0001) | 0.0821 (p 0.0001) | 0.0091 | 0.0134 | 5.8 times |
| 2 | 0.0603 (p 0.0001) | 0.0584 (p 0.0001) | 0.0049 | 0.0030 | 11.8 times |
| 3 | 0.0512 (p 0.0001) | 0.0462 (p 0.0001) | 0.0014 | 0.0036 | 12.7 times |
| 4 | 0.0422 (p 0.0001) | 0.0443 (p 0.0001) | 0.0065 | 0.0086 | 4.9 times |
| 5 | 0.0364 (p 0.0001) | 0.0334 (p 0.0001) | 0.0052 | 0.0021 | 6.5 times |
| 6 | 0.0362 (p 0.0001) | 0.0343 (p 0.0001) | 0.0059 | 0.0040 | 5.8 times |
| 7 | 0.0329 (p 0.0001) | 0.0364 (p 0.0001) | 0.0026 | 0.0009 | 12.4 times |
| 8 | 0.0252 (p 0.0001) | 0.0202 (p 0.0001) | 0.0044 | 0.0007 | 4.6 times |
| 9 | 0.0360 (p 0.0001) | 0.0316 (p 0.0001) | 0.0102 | 0.0059 | 3.1 times |
| 10 | 0.0167 (p 0.0001) | 0.0205 (p 0.0001) | 0.0018 | 0.0020 | 8.4 times |

**Measured in the frame this run used: the difference H18b measures is a
property of the two models' settled states, not of the two lenses.** At every
one of the eleven layers, on either lens taken alone, the twin's settled states
have a higher J-space share than base's, by 0.0167 to 0.0889 on a scale from 0
to 1, and the permutation test is at its floor of 0.0001 in all twenty-two of
those comparisons. At every layer that model effect is larger than the largest
effect the choice of lens produces there, by a factor of between 3.1 and 12.7.
That conclusion does not depend on the twin's lens being well fitted, because it
can be read off base's lens alone.

**Retraction, 2026-09-06, of the sentence this paragraph opened with.** Until
that date this paragraph began "Established from this run: the difference H18b
measures is a property of the two models' settled states, not of the two
lenses." That was wrong as an unqualified claim and it is withdrawn. The
numbers in it stand, every one of them, for the coordinate frame they were
measured in, and that frame is not the one the lens matrices were fitted in.
Recomputing the same comparison in the fitting frame reverses it: the twin is
higher in four of the twenty-two comparisons rather than twenty-two, and the
model effect is 0.1 to 0.5 times the instrument effect rather than 3.1 to 12.7
times, so in that frame the choice of lens moves the share more than the choice
of model does. Section 3.6 gives both sets of numbers side by side. What may be
said without qualification is narrower: in the frame this run measured, the
difference is a model property, and whether that holds in general is not
settled by this experiment.

**The lens-quality sensitivity check** (not pre-registered, no verdict weight,
deviation 9). The whole comparison was repeated with the five-prompt probe lens
in place of the 40-prompt lens, an eightfold reduction in fitting corpus. The
verdict is the same, SUPPORTED with 4 of 6 band layers, and the band layers that
qualify shift only at the margin, from layers 6, 7, 9 and 10 to layers 6, 7, 8
and 10. Record: `output/exp017_jspace_lens5.json`.

### 3.5 The finding the controls force, which limits what the share means

This is the most important caveat in the record and it is stated before the
verdict. Subtracting each measurement's own rotation control from it, so that a
positive number means the real dictionary explains more of the state than a
randomly rotated copy of the same dictionary does. Each cell is the median share
over the 25 prompts minus the mean over the three rotation seeds of that
control's median share. The table is generated by `make_tables.py` and appears
in `output/tables.md` under "Real minus its own rotation control"; until a
review of the pull request caught it, it was the one results table in this
record that the generator did not emit, so a rerun would have left it stale.

| layer | base states, base lens | twin states, twin lens | twin states, base lens | base states, twin lens |
|---|---|---|---|---|
| 0 | +0.1051 | +0.1811 | +0.1990 | +0.0780 |
| 1 | +0.0790 | +0.1620 | +0.1673 | +0.0580 |
| 2 | +0.0054 | +0.0692 | +0.0744 | +0.0012 |
| 3 | +0.0006 | +0.0522 | +0.0509 | -0.0017 |
| 4 | -0.0179 | +0.0264 | +0.0369 | -0.0245 |
| 5 | -0.0133 | +0.0284 | +0.0328 | -0.0163 |
| 6 | -0.0323 | +0.0090 | +0.0228 | -0.0370 |
| 7 | -0.0216 | +0.0242 | +0.0211 | -0.0252 |
| 8 | -0.0197 | +0.0153 | +0.0152 | -0.0157 |
| 9 | -0.0241 | +0.0069 | +0.0177 | -0.0288 |
| 10 | -0.0373 | -0.0227 | -0.0214 | -0.0388 |

**Base GPT-2 Small's settled states beat the rotation control only at the first
few layers and fall below it from layer 4 onward, across the whole workspace
band.** At layer 10 base's settled states are explained less well by base's own
lens directions, by 0.037 on a scale from 0 to 1, than by a randomly rotated
copy of them. **Established from this run:** at the workspace band, base's
settled states are not preferentially aligned with the directions the lens can
name; a random rotation of those directions does at least as well.

The twin's settled states stay above their control at every layer from 0 to 9
and fall below only at layer 10. So the twin's states retain some alignment
with the nameable directions exactly where base's have lost it.

**Added 2026-09-06:** this table too depends on the coordinate convention the
run measured in, and section 3.6 gives it in the other convention. The direction
of base's finding survives there and starts one layer earlier, at layer 3
instead of layer 4; the twin's does not survive intact, falling below its
control at layers 4, 6, 9 and 10 rather than at layer 10 alone.

Two consequences, stated plainly. First, the H18b comparison remains valid: it
compares two models on the same measure with a matched control, and the
difference between them is real, same-lens, and large relative to every control
gap. Second, **the absolute level of the J-space share at the band layers must
not be read as evidence of verbalizable content in either model**, because for
base it is below what an aimless dictionary achieves. That is a limitation of
what the share can support at those depths, and it is a finding in its own
right, one that points the same way as H16b in EXP_011, which asks whether the
loop leaves the verbalizable directions.

### 3.6 The coordinate frame the dictionary sits in, and what the other frame gives

**The answer first.** The dictionary every number in sections 3.3 to 3.5 was
measured against is built in one set of coordinates, and the Jacobian matrices
it is built from were fitted in another. Recomputing the whole comparison in
the frame the Jacobians were fitted in leaves the H18b verdict where it stands,
SUPPORTED at exactly four of the six band layers, but it changes which four,
it divides every share by between 6.7 and 35.8, and it reverses the
cross-check that section 3.4 called the stronger half of this result. A review of the pull request raised the possibility on 2026-09-06 and
the numbers below are the answer, measured rather than argued. **Nothing here
changes a registered verdict.** Section 6.3 item 5 puts the choice of frame to
the operator, because rule R8 reserves it.

**What the two frames are, in plain terms.** A model's internal activity is a
list of 768 numbers per position, and there is more than one convention for
writing that list down. TransformerLens, the instrumentation library this
project uses to reach inside a model, rewrites a model's weights into its own
convention when it loads one. Three scripts here load the two models the same
way, `run_loop.py` for the loop, `verify_model.py` for the provenance check and
`run_jspace.py` for the states this probe decomposes, and all three call
`HookedTransformer.from_pretrained("gpt2", hf_model=..., tokenizer=...,
device="cpu")` and pass none of that function's processing options, so the
library's defaults apply. Read from the installed library on 2026-09-06 and
recorded in `output/frame_check.json`, those defaults are `fold_ln` true,
`center_writing_weights` true, `center_unembed` true and `fold_value_biases`
true. The twin's lens, by contrast, was fitted by `jlens.from_hf` against the
Hugging Face model as downloaded, with no rewriting at all. So the lens matrices
speak the Hugging Face convention and the states and the dictionary the probe
built speak the TransformerLens one.

**How far apart the two unembeddings are.** The dictionary is built as
`W_U^T J_l`, so the choice of unembedding matrix `W_U`, the matrix that turns
internal activity into a score for every one of the 50,257 or 50,258
vocabulary tokens, decides every direction in it. Comparing the matrix the
probe used with the model's own output matrix, generated into `output/tables.md`
under "The two frames, measured":

| model | difference relative to the Hugging Face matrix | common shift length over mean direction length | fraction of the difference left after removing the common shift | per-token cosine, median | final normalisation gain, smallest to largest |
|---|---|---|---|---|---|
| base | 0.6823 | 0.5184 | 0.6554 | 0.8135 | 0.0044 to 17.4193 |
| twin | 0.6845 | 0.5205 | 0.6548 | 0.8128 | 0.0103 to 17.3652 |

Reading that row by row. The difference between the two matrices is 68 percent
of the size of the Hugging Face matrix itself, on a scale where 0 would mean
the two are identical, so this is not a small perturbation. The one vector by
which every token's direction is shifted has 52 percent of the length of a
typical direction, which is large, but removing that common shift still leaves
65 percent of the difference, so the difference is not mainly a shift. The
median angle between a token's direction in the two matrices has a cosine of
0.81, where 1 would mean the same direction and 0 would mean at right angles.
**Established by direct reconstruction:** the TransformerLens matrix is the
Hugging Face matrix with each of the 768 coordinates multiplied by the final
normalisation step's learned gain, then the mean over the coordinates removed
from each token's direction, then the mean over the vocabulary removed from
each coordinate. Rebuilding it that way reproduces it to a relative error of
2.0e-07, that is two parts in ten million. The gain is the reason the two
differ so much: it runs from 0.0044 to 17.42 across the 768 coordinates, so
folding it in stretches some coordinates by a factor of seventeen and flattens
others almost to nothing.

**Whether the states are in centred coordinates: yes, and it does not matter
here.** `center_writing_weights` makes every write into the internal state
average to zero across the 768 coordinates, so the states this probe read are
mean-centred by construction, which is the same thing EXP_011 recorded about
its own states. Measured on the first five prompts at one layer inside the band
and one outside it:

| model | layer | mean state length | largest mean over coordinates, TransformerLens | largest mean over coordinates, Hugging Face | gap length over state length |
|---|---|---|---|---|---|
| base | 0 | 665.8 | 3.81e-07 | 0.0325 | 0.0007 |
| base | 5 | 1128.4 | 3.91e-07 | 1.0229 | 0.0244 |
| twin | 0 | 634.3 | 5.22e-07 | 0.0238 | 0.0004 |
| twin | 5 | 723.2 | 4.33e-07 | 0.3526 | 0.0130 |

The TransformerLens states average to at most 5.2e-07 across their 768
coordinates against state lengths of 634 to 1,128, which is zero to
floating-point noise, and the Hugging Face states do not. The two frames'
states differ by one and the same number added to all 768 coordinates, to
within 7.4e-04 on states of that length, and that difference is worth between
0.04 percent and 2.4 percent of a state's own length. **Established from this
run: the state half of the frame mismatch is negligible and the dictionary half
is not.**

**What it does to the share, on a subset first.** Five prompts per model, real
dictionaries only, one layer inside the verdict-bearing band and one outside:

| layer | combination | twin median | base median | absolute difference |
|---|---|---|---|---|
| 0 | committed states and dictionary | 0.3729 | 0.2999 | 0.0730 |
| 0 | committed states, Hugging Face dictionary | 0.0323 | 0.0279 | 0.0043 |
| 0 | Hugging Face states and dictionary | 0.0323 | 0.0279 | 0.0043 |
| 5 | committed states and dictionary | 0.2828 | 0.1368 | 0.1460 |
| 5 | committed states, Hugging Face dictionary | 0.0348 | 0.0098 | 0.0250 |
| 5 | Hugging Face states and dictionary | 0.0348 | 0.0098 | 0.0250 |

Swapping the dictionary alone moves the share by 0.34 and 0.27 at layer 0 and
by 0.25 and 0.13 at layer 5, on the 0 to 1 share scale, which is far larger
than any effect this experiment reports. Swapping the states as well changes
nothing to four decimal places, which is the same finding as the table above
seen from the other side.

**The full recomputation.** The committed probe took 160 seconds of one
processor thread, so recomputing everything in the consistent frame was cheap
and was run rather than estimated. It took 114 seconds and wrote
`output/exp017_jspace_hfframe.json`, which is stamped as a sensitivity reading
and carries no verdict weight. Generated into `output/tables.md` under "H18b in
the Hugging Face frame":

| layer | twin median share | base median share | absolute difference | control spread | permutation p | both conditions | band |
|---|---|---|---|---|---|---|---|
| 0 | 0.0328 | 0.0309 | 0.0019 | 0.0085 | 0.0007 | no | no |
| 1 | 0.0562 | 0.0442 | 0.0120 | 0.0272 | 0.0001 | no | no |
| 2 | 0.0308 | 0.0258 | 0.0049 | 0.0093 | 0.0001 | no | no |
| 3 | 0.0272 | 0.0159 | 0.0113 | 0.0046 | 0.0001 | yes | no |
| 4 | 0.0247 | 0.0153 | 0.0094 | 0.0153 | 0.0001 | no | no |
| 5 | 0.0345 | 0.0217 | 0.0127 | 0.0119 | 0.0001 | yes | yes |
| 6 | 0.0327 | 0.0150 | 0.0176 | 0.0166 | 0.0001 | yes | yes |
| 7 | 0.0396 | 0.0197 | 0.0199 | 0.0136 | 0.0001 | yes | yes |
| 8 | 0.0327 | 0.0152 | 0.0175 | 0.0076 | 0.0001 | yes | yes |
| 9 | 0.0253 | 0.0179 | 0.0073 | 0.0095 | 0.0001 | no | yes |
| 10 | 0.0087 | 0.0158 | 0.0071 | 0.0084 | 0.0001 | no | yes |

**Four of the six band layers meet both conditions in this frame as well, so
the mechanical H18b verdict is the same, but they are layers 5, 6, 7 and 8
rather than layers 6, 7, 9 and 10.** The share levels are much lower: across
the six band layers the twin's median share runs from 0.0087 to 0.0396 here
against 0.2851 to 0.3221 in the registered frame, and base's from 0.0150 to
0.0217 against 0.2538 to 0.2964. The registered pairing still has the twin
above base at ten of the eleven layers; at layer 10 it reverses, the twin
reading 0.0087 against base's 0.0158.

**The cross-check does not survive the change of frame, and that is the serious
part.** Holding the lens fixed and swapping only whose states are decomposed,
which section 3.4 used to argue that the difference is a property of the models
rather than of the instrument:

| layer | registered frame, base lens | registered frame, twin lens | Hugging Face frame, base lens | Hugging Face frame, twin lens |
|---|---|---|---|---|
| 0 | +0.0856 (p 0.0001) | +0.0889 (p 0.0001) | +0.0022 (p 0.0004) | -0.0062 (p 0.0001) |
| 1 | +0.0778 (p 0.0001) | +0.0821 (p 0.0001) | +0.0039 (p 0.0004) | +0.0046 (p 0.0002) |
| 2 | +0.0603 (p 0.0001) | +0.0584 (p 0.0001) | -0.0016 (p 0.0005) | +0.0006 (p 0.0226) |
| 3 | +0.0512 (p 0.0001) | +0.0462 (p 0.0001) | -0.0018 (p 0.0005) | -0.0018 (p 0.0003) |
| 4 | +0.0422 (p 0.0001) | +0.0443 (p 0.0001) | -0.0034 (p 0.0001) | -0.0015 (p 0.0001) |
| 5 | +0.0364 (p 0.0001) | +0.0334 (p 0.0001) | -0.0057 (p 0.0002) | -0.0012 (p 0.0003) |
| 6 | +0.0362 (p 0.0001) | +0.0343 (p 0.0001) | -0.0040 (p 0.0002) | -0.0020 (p 0.0001) |
| 7 | +0.0329 (p 0.0001) | +0.0364 (p 0.0001) | -0.0045 (p 0.0002) | -0.0030 (p 0.0001) |
| 8 | +0.0252 (p 0.0001) | +0.0202 (p 0.0001) | -0.0043 (p 0.0001) | -0.0120 (p 0.0001) |
| 9 | +0.0360 (p 0.0001) | +0.0316 (p 0.0001) | -0.0041 (p 0.0001) | -0.0122 (p 0.0001) |
| 10 | +0.0167 (p 0.0001) | +0.0205 (p 0.0001) | -0.0009 (p 0.0151) | -0.0018 (p 0.0001) |

A positive number means the twin's settled states have the higher share. In the
registered frame the twin is higher in all twenty-two comparisons, by 0.0167 to
0.0889. In the Hugging Face frame the twin is higher in four of the twenty-two
and lower in the other eighteen, by up to 0.0122, and the largest gap in either
direction, 0.0122, is smaller than the smallest gap the registered frame
reports, 0.0167. The ratio that section 3.4 leant on turns over with it: the
smallest model effect divided by the largest instrument effect runs from 3.1 to
12.7 times in the registered frame and from 0.1 to 0.5 times in the Hugging
Face frame, so in that frame the choice of lens moves the share more than the
choice of model does.

**The rotation-control finding of section 3.5 survives in weakened form.** In
the Hugging Face frame base's settled states fall below their own rotation
control from layer 3 onward, one layer earlier than in the registered frame,
and the twin's states sit above their control at layers 0, 1, 2, 3, 5, 7 and 8
and below it at layers 4, 6, 9 and 10, where in the registered frame the twin
was above at every layer but the last. The full table is in `output/tables.md`
under "Real minus its own rotation control, Hugging Face frame".

**What is established, and what is not.** Established from this run: the two
frames give different J-space shares, by far more than any difference this
experiment reports between the two models; the H18b verdict as the spec's
scoring rule defines it is SUPPORTED at four of six band layers in both frames;
and the same-lens cross-check that the registered frame reports does not hold
in the other frame. Inferred, not established: because the Jacobian matrices
were fitted against the unrewritten model, the Hugging Face frame is the
self-consistent one, and a reader who wants the share to mean "how much of the
state is built from directions that raise a token's score" should prefer it.
Speculation, offered so it can be tested rather than believed: the registered
frame's much higher shares come from its dictionary having the vocabulary mean
removed, which spreads the directions out and widens the cone they span, and
that is a geometric property of the dictionary rather than a fact about the
model. **Not established either way:** which frame this project should register
for J-space work. Both readings have an argument. The lens's own readout applies
the final normalisation before the unembedding, which is the argument for
folding the gain in as the registered frame does; the matrices being decomposed
were fitted without it, which is the argument for the Hugging Face frame. The
spec's section 6.4 says only "the model's unembedding matrix" and does not
choose.

**How this relates to EXP_011.** The EXP_011 results record,
`experiments/exp_011_small_overlap/RESULTS_EXP011.md` on its own branch,
records the same class of mismatch as its deviation 2 and its decision item 4:
its states were mean-centred by the same loader defaults while its dictionary
was built from the Hugging Face unembedding, and it states that how much that
moves the shares is unmeasured. This experiment's case is related but not
identical, and the difference matters. EXP_011's dictionary is the Hugging Face
one, so its mismatch is the state half alone; this experiment's dictionary is
the TransformerLens one, so it carries the final normalisation gain that
EXP_011 explicitly left out and recorded as a limitation, and its mismatch is
the dictionary half as well. Marked as inference for EXP_011 rather than
established, because the states, the lens and the prompts all differ: the half
EXP_011 left unmeasured is the half that turns out not to matter here, moving
the share by nothing at four decimal places, so EXP_011's unmeasured deviation
is probably small in its own setting too.

## 4. Deviations

Recorded flat, whether or not they change a verdict, per the folder rule. The
first four were written into the spec before any run, with their reasons.
Deviations 5 to 9 arose during the run. Deviations 10 to 20 were found after it
by reviews of the pull request, on 2026-09-05 for 10 to 14 and on 2026-09-06 for
15 to 20, the last two by a second review of the changes the first one asked
for, and each names what was found and what changed in response. None of
them was found by the run itself, which is worth saying plainly: this record's
own checks did not catch them.

1. **The provenance check and the lens-fit timing probe ran before the spec was
   committed** (spec section 9.1). The provenance check establishes what the
   models are and tests no hypothesis. The timing probe measures one number, the
   cost per prompt of fitting a lens, which the spec turns into a prompt count
   by a rule fixed before the number was read. Both are reported here with
   their outputs.
2. **Base GPT-2 Small was re-run rather than read from the committed arm SB
   artifacts** (spec section 9.2), because those ran under a one-step gate and
   this spec fixes a two-step gate. The committed artifacts supply the
   secondary reading in section 2.5, and the three reproduction checks in
   section 2.4 show the re-run is the registered apparatus.
3. **The reference grouping is this run's own matched base arm**, not
   EXP_010d's committed Small reference (spec section 9.3), because EXP_010d
   used a different 25-prompt list and the adjusted Rand index compares two
   groupings of the same items. EXP_010d's reference supplies the secondary
   reading on the 20 shared prompts, where this run's base scores 0.9805.
4. **The task brief's arm label "S1" was read as arm SB**, the full stack, on
   the strength of the attachment points the brief itself names (spec section
   9.4). Arm S1 in `EXP_010b_SPEC.md` is the 0 to 5 window and was not run.
5. **The twin's lens was fitted on fewer than 50 prompts**, which the spec
   named in advance as a deviation. The five-prompt probe measured 221.0
   seconds per prompt, so the spec's budget rule, which caps the fit at 9,000
   seconds, chose 40 prompts. The reference Medium fit used 100 and the
   third-party base lens used 277. Section 3 records the consequence for the
   instrument's quality and the sensitivity check that bounds it.
6. **No figures.** The spec allowed small figures if useful. `matplotlib` is
   not installed in this environment, so every result is given as a table, all
   of them generated from the JSON artifacts by `make_tables.py`.
7. **The J-space probe's per-layer states are read from the terminal rescaled
   to the loop's own re-injection size**, as the spec's section 6.5 states.
   This matters more than it sounds: the end state as the loop leaves it is
   about 3.7 times larger than the size the loop re-injects at, because the
   loop rescales at the start of each iteration and not at the end. The share
   itself does not depend on a state's overall size, but the states read at
   later layers do, because the normalisation steps inside the model are not
   scale-free. The rescale factors are recorded in `output/exp017_jspace.json`.
8. **A non-registered harness check of the J-space probe ran before the
   registered one**, on 2 of the 11 layers with 200 shuffles instead of 10,000
   and with the five-prompt probe lens standing in for the twin lens. It exists
   to surface coding errors cheaply and carries no verdict weight. Its output is
   `output/exp017_jspace_harness.json` and the registered run neither reads it
   nor overwrites it. This is the same pattern EXP_010b used for its
   `small_smoke` tier. **Correction, 2026-09-06, withdrawing the "200 shuffles"
   in the sentence above:** that harness check asked for 200 shuffles and ran
   10,000. Deviation 16 records why, what the committed artifact therefore
   holds, and the re-run that shows the difference.
9. **A lens-quality sensitivity check ran after the registered comparison**,
   repeating the whole probe with the five-prompt probe lens in place of the
   40-prompt lens. It was not pre-registered. It changes no verdict and no
   scoring rule; it exists because the twin lens's small fitting corpus is this
   experiment's largest weakness and the reader is entitled to know whether the
   H18b reading moves when the lens quality moves. It is reported in section 3
   and marked throughout as carrying no verdict weight.
10. **The lens fit's wall-clock cap was not enforced by the code that ran it,
    and the committed fit stayed inside the cap without being made to.** The
    spec's section 6.2 sets a 9,000 second cap and a fallback for a fit that
    overruns it, but the instrument's fitting call has no deadline of its own
    and `fit_twin_lens.py` did not add one, so on this run the cap was a number
    in the spec rather than a rule the machine could apply. What actually
    happened, from the committed run log `output/fit_twin.log`: the fit resumed
    from the five-prompt timing probe's checkpoint and computed prompts 6 to 40
    in 8,253 seconds, which is 92 percent of the cap, so the fallback was never
    needed. The gap between the spec and the code was found by a review of the
    pull request, not during the run. `fit_twin_lens.py` now refuses before
    loading the model if the timing probe's measured cost per prompt predicts
    that not even one more prompt fits in the remaining budget, and during the
    fit it works one prompt at a time from the instrument's own per-prompt
    checkpoint and stops before starting a prompt that would cross the cap,
    exiting with a distinct code and naming the section 6.2 fallback. Its
    `--selftest` replays this run's timings against that rule and confirms the
    committed fit would have been allowed to finish. Nothing about the
    committed lens changes, and the lens was not refitted.
11. **The scripts did not pin the models' revisions, and now do.** A revision is
    the 40-character commit identifier of a Hugging Face repository's state, and
    without one a loader takes whatever the repository holds on the day it runs.
    The provenance record shows which revisions this run actually loaded, the
    twin at `fc740804ff49f50fe3ef871b31eb2d5a5584132c` and base GPT-2 Small at
    `607a30d783dfa663caf39e06633721c8d4cfcd7e`, so the run itself is identified;
    what was missing was any guarantee that a later re-run would load the same
    weights. Both revisions now live in one file, `exp017_models.py`, and the
    loop, the lens fit, the J-space probe and the verification script all pass
    them to both the model load and the tokenizer load. Loading at those pinned
    revisions was checked and reproduces this run's model shapes and, in the
    conversion check, every committed number.
12. **The 40-prompt fit continued the timing probe's checkpoint, and until
    2026-09-05 nothing in the code arranged that.** The probe writes its
    checkpoint under a name built from its own prompt count and tag,
    `jlens_lamini_gpt2_124m_5_probe.ckpt.pt`, and the registered fit looks for
    `jlens_lamini_gpt2_124m_40_twin.ckpt.pt`, so on a clean rerun the fit would
    have found no checkpoint and started from prompt 1. The committed run got
    the reuse by hand, which its log records in its first line. This matters
    because of the cap: the 40 prompts cost 9,357 seconds in total across the
    two invocations, so a fit started from nothing, with the deadline now
    enforced and the probe's measured 221.0 seconds per prompt as the
    predictor, stops after 38 of the 40 prompts at 8,950 seconds. Continuing
    the probe's five, it reaches all 40 and spends 8,253 seconds. The reuse is
    now automatic: `fit_twin_lens.py --resume-from` defaults to the probe's
    checkpoint and copies it into place when the fit has none of its own, the
    corpus digest matches and the checkpoint was built with the same layers.
    Both replays are asserted in that script's `--selftest`.
13. **A fit the deadline stopped could have been scored as the registered
    lens.** The deadline added on 2026-09-05 saved its partial lens under the
    name the registered probe reads, and `run_jspace.py` checked only that the
    file existed, so a lens fitted on fewer prompts than the budget rule chose
    could have produced a SUPPORTED or NOT SUPPORTED verdict where spec section
    6.2 requires the base lens on both sides. A review of the pull request found
    it before any such lens existed. A stopped fit now writes to a separate file
    whose name carries the count it reached, and `run_jspace.py` compares the
    twin lens's prompt count against the count in
    `output/fit_budget_decision.json` before it will score it as registered,
    taking the section 6.2 route otherwise. The committed 40-prompt lens meets
    the requirement of 40 and passes the check unchanged. The lens-quality
    sensitivity check of deviation 9 uses the five-prompt lens deliberately and
    now opts in with `--allow-short-twin-lens`, which stamps its verdict string
    as a sensitivity reading; the committed
    `output/exp017_jspace_lens5.json` predates that stamp and still reads a
    bare "SUPPORTED", which section 3.4 has always described as carrying no
    verdict weight.
14. **The fitting corpus was reproducible only from a remote dataset, and is
    now committed.** The spec froze the corpus at
    `_STAGE2_JSPACE/artifacts/wikitext_prompts_160.json`, which is not version
    controlled, on the reasoning that the loader's derivation is deterministic.
    That holds only while the remote WikiText-103 stream does not change, so a
    later rerun could have fitted a different lens without anything saying so.
    The exact 160 prompts are now committed beside the scripts as
    `wikitext_prompts_160.json`, SHA-256
    `c325a40c06f5dcfe14a3cea02afe1d64d42914146822c63e111dc665fdeba8d7`, 162,933
    bytes, and `fit_twin_lens.py` loads that file by default and refuses to run
    when its digest does not match. **Verified, not assumed:** passing all 40
    prompts through the instrument's own encoder reproduces the sequence length
    and valid-position count that the two committed run logs record for every
    one of the 40, including the single short prompt at position 30, which gives
    117 tokens and 100 valid positions. That establishes the committed file is
    the corpus the committed lens was fitted on.

15. **The dictionary the J-space probe decomposed against was built in the
    TransformerLens coordinate convention, while the lens matrices it is built
    from were fitted in the Hugging Face one.** Found by a review of the pull
    request on 2026-09-06, after the run. The probe builds each layer's
    dictionary as `W_U^T J_l`, taking `W_U`, the matrix that turns internal
    activity into a score for every vocabulary token, from the TransformerLens
    conversion of the model, whose default processing folds the final
    normalisation step's learned per-coordinate gain into it and then subtracts
    a common vector from every token's direction. The matrices `J_l` were
    fitted by `jlens.from_hf` against the Hugging Face model, which has no such
    processing. The two versions of the unembedding differ by 68 percent of the
    size of the Hugging Face matrix, and the resulting shares differ by far
    more than anything this experiment reports between the two models. Section
    3.6 measures all of it and gives the whole comparison recomputed in the
    fitting convention, which was cheap enough to run rather than estimate:
    `output/exp017_jspace_hfframe.json`, 114 seconds, written by
    `run_jspace.py --frame hf` and stamped as a sensitivity reading with no
    verdict weight. The mechanical H18b verdict is the same in both
    conventions, SUPPORTED at four of the six band layers, on layers 5, 6, 7
    and 8 there against 6, 7, 9 and 10 here; the same-lens cross-check of
    section 3.4 reverses, and that sentence has been withdrawn by name in
    section 3.4. `run_jspace.py` now takes the convention as an option, refuses
    to call anything but the registered one a registered comparison, and
    records which one produced every artifact. **No committed number was
    recomputed away and no verdict was changed here:** the choice of registered
    convention is decision item 5 in section 6.3, reserved for the operator
    under rule R8.

16. **The harness check asked for 200 shuffles and ran 10,000, and its
    committed artifact records the number it asked for.** Found by the same
    review. `run_jspace.py` took the shuffle count as a default argument, which
    Python evaluates once when the file is read rather than each time the
    function runs, so `harness_check_jspace.py` setting it to 200 had no
    effect. The count written into the artifact was read separately, at run
    time, so `output/exp017_jspace_harness.json` says `n_perm: 200` while the
    permutation p values in it, all 0.0001, are the floor of a 10,000-shuffle
    test rather than the 0.00498 floor of a 200-shuffle one. Nothing registered
    is affected: the registered run asked for 10,000 and ran 10,000, and
    `output/exp017_jspace.json` records 10,000. The count is now resolved when
    the test runs, `run_jspace.py --selftest` checks that lowering it really
    lowers the number of shuffles, and the harness was re-run with the fix to a
    new file, `output/exp017_jspace_harness_p200.json`, so that the committed
    artifact is not overwritten. **The re-run is the evidence, not the claim:**
    it reproduces both layers' shares to four decimal places, its p values are
    0.00498 where the committed ones are 0.0001, and it took 48 seconds against
    the committed run's 209 seconds, a fall of 161 seconds that is the 9,800
    shuffles per comparison that are no longer being run.

17. **The lens fit's wall-clock cap was checked before every prompt and never
    after the last one.** Found by the same review. Deviation 10 added a
    deadline that refuses to start a prompt whose predicted cost would cross
    the 9,000 second cap, and the prediction is a cost per prompt measured on a
    five-prompt probe, so a prompt that runs slower than predicted could return
    after the cap with every prompt consumed and the fit would have been
    written out under the registered name. `fit_twin_lens.py` now reads the
    clock again after each chunk returns and treats a wall time past the cap as
    a stopped fit, which writes the partial lens to a separate file whose name
    carries the count it reached and exits with code 3, the signal for the
    spec's section 6.2 fallback. Its `--selftest` drives that case through the
    fitting loop on a fake clock: thirty-nine prompts at the predicted 221
    seconds and a fortieth at 1,000 seconds gives all 40 prompts consumed in
    9,619 seconds against the 9,000 second cap, and the fit is scored as
    stopped. **The committed lens is unaffected, and this is measured rather
    than assumed:** the committed fit computed its 35 prompts in 8,253 seconds
    against the cap, that is 92 percent of it, so the new check would have
    passed it unchanged, and the self-test replays exactly that.

18. **Neither a fit checkpoint nor a finished lens file said which model or
    which corpus it came from, and now both do.** Found by the same review. A
    checkpoint holds running sums of Jacobian matrices and the instrument's own
    file format records only their shape, so a checkpoint built with
    `--model-path` pointing at a local checkout, or with
    `--allow-different-prompts`, would have seeded the registered fit with sums
    from another model or another corpus and the shape check would have passed
    it. The same gap existed one step later: `run_jspace.py` admitted any lens
    whose prompt count met the budget, including one fitted on a different
    768-wide model. Every fit now writes a provenance sidecar beside its
    checkpoint, `CHECKPOINT.provenance.json`, holding the model repository, the
    pinned revision, the corpus file and its SHA-256 digest, whether the digest
    was enforced, and the sequence and layer settings, and stamps the same
    record inside the lens file it saves. Seeding compares those fields and
    refuses a mismatch; the probe compares them against the pinned twin
    revision and the digest of the committed corpus file, hashed at run time so
    that no digest is copied out to drift.

    **What the committed artifacts establish about themselves, exactly.** Both
    predate the stamp and carry none, so the honest answer is that the files
    themselves establish nothing about their origin and the evidence is
    elsewhere. For the five-prompt probe checkpoint
    `jlens_lamini_gpt2_124m_5_probe.ckpt.pt`: its own run log
    `output/fit_probe_db16.log` records `model=MBZUAI/LaMini-GPT-124M` on its
    first line and the loader's report of that repository below it, the
    checkpoint's stored settings are this experiment's own, source layers 0 to
    10 with target layer 11 and 16 leading positions skipped, and deviation 14
    establishes the corpus by passing all 40 prompts of the committed
    `wikitext_prompts_160.json` through the instrument's encoder and
    reproducing the sequence length and valid-position count that log records
    for every one of the five prompts it fitted. What no log records is the
    weight revision, because the probe ran before the revisions were pinned;
    `output/model_verification.json` establishes which revision was on the
    machine that day, the twin at `fc740804ff49f50fe3ef871b31eb2d5a5584132c`,
    and that is inference from same-day provenance rather than a record the
    checkpoint carries. For the lens `jlens_lamini_gpt2_124m_40_twin.pt` the
    position is the same, with `output/fit_twin.log` in place of the probe's
    log. **So both are admitted only by an explicit option, and both options
    are recorded where they are used:** `fit_twin_lens.py
    --accept-unstamped-checkpoint` writes the acceptance into the sidecar it
    leaves behind, and `run_jspace.py --accept-unstamped-twin-lens` writes it
    into the output artifact under `twin_lens_provenance`. Section 8 names both
    in the reproduction commands, so the documented route still runs on the
    committed files.

19. **Four gates added on 2026-09-06 were each incomplete, and a fifth number
    was mistyped.** A second review of the same day found them, after the
    changes deviations 15 to 18 describe were already committed. None of them
    touches a committed number; all four are about what the code would allow a
    later run to do. First, `fit_twin_lens.py` compared provenance when it
    seeded a checkpoint from another one but not when a checkpoint was already
    sitting at the path the fit writes to, which is exactly where an earlier
    invocation with `--model-path` or `--allow-different-prompts` leaves one, so
    the instrument would have resumed those Jacobian sums and the finished lens
    would have been stamped with the current run's provenance. A checkpoint at
    the fit's own path is now checked the same way a seeded one is, and an
    unstamped one is refused unless `--accept-unstamped-checkpoint` says
    otherwise, which is then written beside it. Second, `run_jspace.py` compared
    the stamped fields but not the stamped fit outcome, so a lens from a fit
    that crossed its cap, which the code of deviation 17 marks `overran` and
    still averages over the full prompt count, would have carried a registered
    verdict. Registered scoring now requires an outcome of `complete`, a stamped
    wall-clock cap no longer than the spec's 9,000 seconds, and a measured fit
    time inside it. Third, `--twin-lens-min-prompts` replaced the requirement
    the budget rule chose, so passing 5 made the five-prompt probe lens
    "accepted" and set the registered flag without anyone asking for a
    sensitivity reading; any requirement that is not the budget artifact's own
    count now makes the run a sensitivity reading whatever it then accepts.
    Fourth, a run that is not the registered comparison, whether by frame, by a
    short lens or by an overridden requirement, could still write to
    `output/exp017_jspace.json`, which is the file `make_tables.py` reads and
    this record quotes; such a run now refuses to start without its own
    `--out-suffix`, and it refuses before the expensive part rather than after
    it. All four are asserted in `run_jspace.py --selftest` and
    `fit_twin_lens.py --selftest`, which now run 36 checks each. The fifth
    item is not code: the proposed register row in `REGISTER_VERDICTS.md` gave
    the same-lens cross-check as 0.015 to 0.086 where the artifact gives 0.0167
    to 0.0889. That file's note 7 corrects it, withdraws the old range by name,
    and records that every other number in its rows was then checked back to the
    artifact that produced it.

20. **On a machine that had never downloaded these models, the provenance
    artifact would have carried no weight digests.** Found by the same review.
    `verify_model.py` scanned the local Hugging Face cache for file sizes and
    SHA-256 digests before it loaded the models, and the steps before the scan
    read configuration files only, so on a cold cache the scan would have found
    nothing but `config.json`. **Measured rather than reasoned about:** running
    the scan against an empty cache directory after the configuration call alone
    returns one file, `config.json` at 665 bytes, and no weight file. The load
    now happens before the scan. **The committed artifact is unaffected, and
    that is checked rather than assumed:** `output/model_verification.json` was
    written on a warm cache and already carries all 17 rows, including the twin's
    `model.safetensors` at 510,362,696 bytes and base's at 548,105,171, and
    re-running the scan today reproduces all 17 rows with the same sizes and the
    same digests, which is the check section 1.1 rests on. The artifact was not
    rewritten, because its `generated` timestamp would change and the run that
    produced it is the record.

## 5. H18b verdict: **SUPPORTED, at exactly the pre-registered threshold**

The registered wording asks whether post-training changes the settled states'
J-space share: whether the twin's share on a lens fitted to the twin differs
from base's share on the Neuronpedia lens by more than the random-dictionary
control spread, in either direction. The spec's scoring rule required, at at
least 4 of the 6 band layers 5 through 10, both an absolute gap larger than that
layer's control spread and a two-sided permutation p below 0.05.

**Four of the six band layers meet both conditions: layers 6, 7, 9 and 10.
H18b is SUPPORTED.** Layers 5 and 8 fail on the control condition only.

**This lands on the threshold and not above it, and that must be said first.**
Had the spec asked for 5 of 6 rather than 4 of 6, the verdict would read NOT
SUPPORTED on the same numbers. A reader should treat the mechanical verdict as
the weaker half of what this section reports. The stronger half is the
cross-check in section 3.4, which does not depend on the threshold at all: on
one and the same lens, the twin's settled states have a higher J-space share
than base's at every one of the eleven layers, by 0.0167 to 0.0889 on a scale
from 0 to 1, with the permutation test at its floor of 0.0001 in all
twenty-two of those comparisons. The direction of the effect is the same
everywhere, its size at each layer is between 3.1 and 12.7 times the largest
effect the choice of lens produces there, and it survives replacing the twin's
40-prompt lens with a 5-prompt one.

**In plain terms:** the instruction-tuned twin's settled states sit more inside
the directions the lens can name than base's settled states do. Base's settled
states have, by the band layers, drifted out of those directions to the point
where a randomly rotated dictionary describes them at least as well; the twin's
have not, until the final layer.

**What this does not establish.** It does not establish that the twin's settled
states are more meaningful, more verbalizable, or more interpretable. The share
is a geometric quantity, and section 3.5 shows that at the band layers its
absolute level carries no evidence of verbalizable content for base. It also
does not establish a mechanism: a 3 percent weight change produced both this
and the wholesale replacement of the readout vocabulary, and nothing here
separates cause from coincidence between the two.

**Added 2026-09-06, and it weakens the paragraph above.** Everything in this
section was measured with the states and the dictionary read from the
TransformerLens conversion of each model, while the lens matrices they are built
from were fitted against the unconverted Hugging Face model. Section 3.6
recomputes the whole comparison in the fitting convention. The verdict does not
move: SUPPORTED, at four of the six band layers, at exactly the same bar, on
layers 5, 6, 7 and 8 rather than 6, 7, 9 and 10. The stronger half does move,
and against the direction this section argues: the same-lens cross-check has the
twin above base in four of the twenty-two comparisons there rather than all
twenty-two, and the model effect is 0.1 to 0.5 times the instrument effect
rather than 3.1 to 12.7 times it. **So the honest summary of H18b is now
narrower than the one above.** The mechanical verdict is robust to the
convention and still sits on its threshold; the claim that the difference is a
model property rather than an instrument property is not robust to it, and
holds only in the convention this run measured. Which convention this project
registers for J-space work is decision item 5 in section 6.3, and rule R8
reserves it for the operator.

## 6. What it means, what remains, and what needs the operator's decision

### 6.1 What it means

**The prompt-grouping survives instruction tuning; the words do not.** This is
the finding. A model whose weights differ from base GPT-2 Small's by about 3
percent, and which has been trained to answer instructions rather than continue
text, still sorts these 25 prompts into families that overlap base's families
more than chance allows, at an agreement of 0.1694 where chance is 0 and a
permutation test puts the odds at 97 in 10,000. Yet the words those families
decode to have changed completely: base's ` Divine`, ` prolet`, ` till` and
` Anarch` are replaced by ` anarchism` on 24 of 25 prompts, and no prompt gives
the twin a word base ever produces.

**This splits a claim the project has been carrying as one thing.** Stage 1
found five semantic resting states in GPT-2 Small and the record has treated
their existence and their identity together. This run separates them. The
structure, which prompts end up together, is robust to a small weight
perturbation. The identity, which word each family reads out as, is not. Marked
as inference, not established: this suggests the five basin words are a
property of one checkpoint's readout rather than of the family of models, while
the partition they induce is a more durable property. Every claim in this
project that turns on the words specifically, rather than on the grouping,
inherits that fragility.

**A second thing broke that the register did not ask about.** Base's iterates
alternate forever, holding a similarity of 0.857 one step apart and exactly
1.000 two steps apart, the period-2 bell of the Stage 1 record. The twin's sit
at 1.000 at every gap from 1 to 8. Instruction tuning removed the oscillation
along with the words, leaving still fixed points. Established from this run;
mechanism unknown.

**Where the loop ends up relative to what the model can say.** The rotation
controls in section 3.5 say something the experiment was not designed to ask.
Base's settled states, at the workspace band layers 4 through 10, lie no closer
to the lens's nameable directions than to a randomly rotated copy of them, and
at layer 10 they lie further. The twin's stay closer until the last layer.
Marked as inference: base's loop ends somewhere the lens cannot name, and
post-training pulls the endpoint partway back toward nameable ground.

**The apparatus is the registered apparatus.** Three checks nobody asked for
came out right: base's terminal word agrees with the committed EXP_010b record
on 25 of 25 prompts, its entry loudness ratio reproduces to every committed
digit at 73.0 times natural, and its grouping agrees with EXP_010d's committed
Small reference at 0.9805 on the 20 shared prompts. The twin's departures are
departures from a verified baseline.

### 6.2 What remains

1. **One post-trained model, one prompt set, one seed.** LaMini-GPT-124M is one
   instruction-tuned descendant of GPT-2 Small trained on one corpus. Nothing
   here says whether a different post-training run would land on ` anarchism`,
   on some other single word, or on five words of its own.
2. **The twin's lens is fitted on 40 prompts against base's 277.** The
   sensitivity check shows the H18b verdict does not move when the twin's lens
   is degraded to 5 prompts, and the cross-checks show the effect is readable
   on base's lens alone, so the conclusion is not hostage to this. A lens
   fitted on 100 or more prompts would still tighten the record.
3. **Whether the twin's ` anarchism` funnel is related to base's ` Anarch`
   basin** is open. The two are lexically adjacent, base gives ` Anarch` to
   exactly one prompt, and this experiment has no way to tell relation from
   coincidence.
4. **The twin was run at 64.25 times its natural entry loudness against base's
   73.0 times**, a gap of about 12 percent. A natural-loudness arm on the twin,
   the `natural_i` convention EXP_010c-VARIANTS used, would remove that
   confound and would also say whether the twin's single funnel is a property
   of the model or of the loud injection convention, which is the question
   EXP_015 asked of Medium.
5. **H18b sits on its threshold.** A second post-trained model, or a larger
   prompt set, would move it off the knife edge in one direction or the other.
6. **Which coordinate convention the J-space share should be measured in is
   open, and it changes the share by more than any model difference this
   experiment reports.** Section 3.6 measures both and section 6.3 item 5 asks
   for a ruling. What remains after a ruling is small in compute and large in
   consequence: whichever convention is chosen, EXP_011's committed numbers, the
   Medium record's and this one's should be stated in it, and the two that were
   measured in the other one should say so.

### 6.3 What needs the operator's decision

1. **Does the lag-2 gate become the registered convention?** This experiment
   ran the two-step gate on the task brief's instruction. EXP_010b's registered
   arm SB used the one-step gate. The change is not cosmetic: under it base's
   11 ` Divine` prompts settle at iteration 120 instead of running to the
   1000-iteration cap, and base's own grouping under the two gates agrees only
   at 0.6809. Both gates now have committed evidence on the same 25 prompts in
   the same tree. EXP_018's H19a also uses a two-step gate. A ruling would keep
   the two conventions from drifting apart unnamed. **Recommendation, offered
   not taken:** register the two-step gate as a named variant rather than
   replacing the one-step gate, since the one-step gate is what every Stage 1
   number was measured under.

2. **How much weight should the word-level line of inquiry keep carrying?**
   Section 6.1 argues the five basin words are checkpoint-specific in a way the
   grouping is not. Several open experiments and several passages in the
   operator report treat the words as the finding. Whether to re-scope them is
   a judgement about the programme, not about this experiment, and it is
   reserved for the operator under rule R8.

3. **Should a longer twin lens fit be commissioned?** A 100-prompt fit costs
   about 6 hours 33 minutes of one processor thread at this run's measured
   235.8 seconds per prompt computed, which does not fit in a session that also
   runs the loop. It is worth doing only if H18b's threshold-edge verdict is
   going to be leaned on; the cross-checks in section 3.4 already carry the
   substantive claim without it. **Recommendation, offered not taken:** not now.

4. **Is the rotation-control finding in section 3.5 worth its own experiment?**
   That base's settled states fall below a randomly rotated dictionary at the
   band layers is a stronger and more surprising statement than anything H18b
   was set up to test, and it was measured here only as a control. It overlaps
   H16b, which EXP_011 is testing on a different comparison. Whether to charter
   a direct test, or to let EXP_011 carry it, is an allocation decision.

5. **Which coordinate convention is the registered one for J-space work?** This
   is the largest open question this record carries. The J-space share is
   the fraction of a settled state that can be built from at most 25
   vocabulary directions, and
   the directions are `W_U^T J_l`, so the answer depends on which version of
   the unembedding matrix `W_U` is used. This run used the TransformerLens
   conversion's version, which folds the model's final normalisation gain into
   every direction and then subtracts a common vector; the lens matrices `J_l`
   were fitted against the Hugging Face model, whose version does neither. The
   two give shares that differ by a factor of between 7.7 and 35.8 at the six
   band layers, which is far larger than any difference this experiment
   reports between the two models. Section 3.6 has the numbers and the argument
   on each side: the lens's own readout applies the final normalisation before
   the unembedding, which favours the converted version, and the matrices being
   decomposed were fitted without it, which favours the Hugging Face version.
   The spec's section 6.4 says only "the model's unembedding matrix" and does
   not choose, so this is not a departure from the spec but a gap in it.
   **What hangs on it beyond this experiment:** EXP_011 measured its shares
   with the Hugging Face version against states from the converted model and
   recorded the mismatch as its deviation 2 with the effect unmeasured, and
   H19b in EXP_018 will inherit whichever convention is settled. **Cost of
   acting either way, from this run's own logs:** recomputing this experiment
   in the other convention took 114 seconds, so nothing here is expensive.
   **Recommendation, offered not taken:** register the Hugging Face convention,
   because it is the one the lens matrices were fitted in and the one EXP_011
   built its dictionary from, and have this record's section 3.6 numbers stand
   as the registered H18b reading if that is the ruling. **No verdict has been changed
   in anticipation of it:** the register rows proposed in `REGISTER_VERDICTS.md`
   still carry the numbers this run measured, with the other convention's
   numbers beside them.

## 7. Artifacts

All paths relative to this directory.

- `output/model_verification.json`, `output/tl_conversion_check.json`: the
  provenance record, every file digest, the configuration comparison, the
  weight-drift measurement, the conversion check and the instruction sanity
  check.
- `output/loop_results_twin.json`, `output/loop_results_base.json`: the
  per-prompt loop records, including the similarity scan at lags 1 to 8.
- `output/terminals_twin.npz`, `output/terminals_base.npz`: the settled states,
  holding for each prompt the mean over token positions, the last position and
  the full tensor.
- `output/natural_resid_norms_twin.json`, `output/natural_resid_norms_base.json`:
  the natural per-layer entry sizes behind the loudness ratios.
- `output/exp017_partition.json`: the H18 and H18a computation.
- `output/exp017_jspace.json`: the H18b computation, its controls and its
  cross-checks. `output/exp017_jspace_lens5.json` is the lens-quality
  sensitivity check and `output/exp017_jspace_harness.json` the non-registered
  harness check; neither carries verdict weight.
- `output/exp017_jspace_hfframe.json`: the whole H18b computation repeated in
  the coordinate convention the lens matrices were fitted in, written on
  2026-09-06 by `run_jspace.py --frame hf`. It is stamped as a sensitivity
  reading, carries no verdict weight, and is the evidence behind section 3.6.
- `output/frame_check.json`: the measurement of how far apart the two
  coordinate conventions are, in the unembedding matrix and in the states, and
  the share on five prompts under each. Written by `frame_check.py`, no verdict
  weight.
- `output/exp017_jspace_harness_p200.json`: the harness check re-run on
  2026-09-06 with the shuffle count actually taking effect, written to its own
  file so that the committed `output/exp017_jspace_harness.json` is preserved
  as the record of what ran on 2026-09-05. Deviation 16 explains the difference
  between the two.
- `output/fit_budget_decision.json`: the mechanical application of the spec's
  budget rule to the timing probe.
- `output/tables.md`: every table in this record, generated from the JSON.
- Run logs: `output/loop_twin.log`, `output/loop_base.log` (which also carries
  the partition analysis output, because the two ran in one chained
  invocation), `output/fit_probe_db16.log`, `output/fit_twin.log`,
  `output/jspace_run.log`, `output/jspace_lens5.log`,
  `output/harness_check.log`.
- Scripts: `verify_model.py`, `run_loop.py`, `exp017_partition.py`,
  `fit_twin_lens.py`, `choose_fit_budget.py`, `lens_from_checkpoint.py`,
  `jspace.py`, `run_jspace.py`, `harness_check_jspace.py`, `make_tables.py`,
  `frame_check.py`, which measures the two coordinate conventions against each
  other, and `exp017_models.py`, which holds the two repository names and the
  two pinned revisions that every other script loads its weights at.
- `wikitext_prompts_160.json`: the 160 WikiText-103 prompts the twin's lens was
  fitted on, committed so that a rerun cannot quietly fit a different corpus.
- `REGISTER_VERDICTS.md`: proposed register rows for the orchestrator's sweep.
  This session does not edit `REGISTER.md`.
- The two lens files live in `../../artifacts/`, which is not version
  controlled, and are identified above by their SHA-256 digests.

## 8. Reproducing this run

Every path below is relative to this directory, and every script sets one
processor thread on its first lines.

**What has to be placed by hand.** One large file is not version controlled,
because `_STAGE2_JSPACE/artifacts/` is in the repository's ignore list: the
pre-fitted Neuronpedia lens for base GPT-2 Small,
`jlens_gpt2_small_neuronpedia.pt`, SHA-256
`d1800a1335ada089ef2e1ec0e4bd4d5bd61e6011eacc31f8618fdb3d10aae762`. The J-space
probe resolves it relative to the checkout, at
`../../artifacts/jlens_gpt2_small_neuronpedia.pt`, so put it there or name
another path with `--base-lens`; either way the probe checks its SHA-256
against that digest before it reads it, and stops if the file is missing or the
digest disagrees. **Correction to the code as first committed:** that path was
an absolute path on the machine that ran the experiment, which meant the probe
could not run from any other checkout even with the lens correctly placed. A
review of the pull request found it, and it now resolves from the checkout.

**The fitting corpus is committed.** The 160 WikiText-103 prompts the lens was
fitted on are in `wikitext_prompts_160.json` beside the scripts, SHA-256
`c325a40c06f5dcfe14a3cea02afe1d64d42914146822c63e111dc665fdeba8d7`.
`fit_twin_lens.py` loads that file by default and refuses to fit when the digest
does not match, so a change in the remote dataset cannot silently substitute a
different corpus. `--refresh-prompts` still rebuilds the copy in the artifacts
directory from the loader and reports whether the result matches. Deviation 14
records why this changed and how the committed file was checked against the run
logs.

**The models.** Both are downloaded from huggingface.co at the revisions pinned
in `exp017_models.py`, the twin at `fc740804ff49f50fe3ef871b31eb2d5a5584132c`
and base GPT-2 Small at `607a30d783dfa663caf39e06633721c8d4cfcd7e`. A revision
is the commit identifier of a repository's state at one moment, so pinning it
means a later change on Hugging Face cannot silently change what is loaded.

**The order the steps ran in.** Where a step's cost was recorded, the figure
below is the recorded one and its source is named. The steps with no figure had
no timing written down, so none is given here.

1. `python3 verify_model.py`, the provenance step. It writes
   `output/model_verification.json` and `output/tl_conversion_check.json`. The
   flag `--conversion-check-only` rewrites the second alone, which is how the
   conversion numbers in section 2.3 were re-derived and checked.
2. `python3 run_loop.py --model twin` and `python3 run_loop.py --model base`,
   the loop, 1,096 seconds for the twin and 1,689 seconds for base, that is 18
   and 28 minutes (`output/loop_twin.log` and `output/loop_base.log`).
3. `python3 exp017_partition.py`, the H18 and H18a computation.
4. `python3 fit_twin_lens.py --n 5 --dim-batch 16 --tag probe`, the timing
   probe, 1,105 seconds for 5 prompts, that is 18 minutes
   (`output/fit_probe_db16.log`); then `python3 choose_fit_budget.py`, which
   applies the spec's budget rule to it; then `python3 fit_twin_lens.py --n 40
   --dim-batch 16 --tag twin --accept-unstamped-checkpoint`, the lens fit,
   8,253 seconds for the 35 prompts it computed, that is 2 hours 18 minutes
   (`output/fit_twin.log`). The last option is needed only against the
   committed probe checkpoint, which predates the provenance sidecar described
   in deviation 18 and so says nothing about which model or corpus it came
   from; a probe run today writes that sidecar and the fit then continues it
   with no option at all. Since 2026-09-06 the fit also checks the
   checkpoint sitting at its own output path before resuming it, and
   refuses one whose stamped model, revision, corpus digest or settings
   disagree, so a checkpoint left by an earlier run with different
   arguments stops the fit instead of feeding it.

   **The order matters, and so does the reuse.** The 40-prompt lens is not 40
   prompts of fresh work. The probe fits the first 5 prompts of the same list in
   the same way and leaves them in a checkpoint, and the fit continues from
   there, computing prompts 6 to 40 and averaging all 40 into the lens it saves.
   `fit_twin_lens.py` now does that reuse itself: `--resume-from` defaults to
   the probe's checkpoint and copies it into place when the fit has none of its
   own and the corpus digest matches. Running the fit without the probe first is a
   different experiment and will not reach 40 prompts: the 40 prompts cost 9,357
   seconds all told, so with the cap enforced and the probe's measured 221.0
   seconds per prompt as the predictor, a fit from nothing stops after 38 of the
   40 at 8,950 seconds. The fit stops itself before the spec's 9,000 second cap
   and exits with code 3 when it has to, writing its partial lens to a separate
   file named for the count it reached, which is the signal to score H18b on the
   base lens for both sides.
5. `python3 jspace.py --selftest`, the eight synthetic checks of the J-space
   share; `python3 fit_twin_lens.py --selftest`, the deadline arithmetic, the
   corpus digest and the checkpoint reuse; and `python3 run_jspace.py
   --selftest`, the twin lens budget gate. All three are expected to print ALL
   PASS, and none of them loads a model.
6. `python3 run_jspace.py --twin-lens
   ../../artifacts/jlens_lamini_gpt2_124m_40_twin.pt
   --accept-unstamped-twin-lens`, the J-space probe, 160 seconds, that is under
   3 minutes (the `wall_seconds` field of `output/exp017_jspace.json`). The
   probe checks the twin lens twice: against the prompt count the budget rule
   chose, refusing to score a shorter one as the registered comparison, and
   against the provenance stamp described in deviation 18, refusing to score an
   unstamped one unless the second option above says so, which is then written
   into the output under `twin_lens_provenance`. The committed lens predates the
   stamp, so the documented command carries that option; a lens fitted today
   carries the stamp and needs no option. Omitting `--twin-lens`, or offering a
   lens either check refuses, runs the spec's section 6.2 fallback, scoring both
   sides on the base lens. The lens-quality sensitivity check of deviation 9
   passes `--allow-short-twin-lens` on purpose, together with
   `--out-suffix _lens5`, and its output is stamped as a sensitivity
   reading. Since 2026-09-06 that suffix is not a courtesy but a rule: any
   run that is not the registered comparison, whether because of the
   coordinate convention, a short lens or an overridden prompt-count
   requirement, refuses to start without an `--out-suffix` of its own, so
   that `output/exp017_jspace.json` keeps the registered result. **Re-running this command on 2026-09-06 with the
   current code reproduced every number in the committed
   `output/exp017_jspace.json` exactly, to the last digit of every share, every
   control and every permutation p; the scratch copy was compared field by
   field and then deleted rather than committed.**
7. `python3 make_tables.py`, which regenerates `output/tables.md` from the JSON
   so that no number in this record is hand-copied.
8. Added 2026-09-06, and neither step carries verdict weight. `python3
   frame_check.py` measures the two coordinate conventions against each other
   and writes `output/frame_check.json`, 73 seconds. `python3 run_jspace.py
   --frame hf --out-suffix _hfframe --accept-unstamped-twin-lens --twin-lens
   ../../artifacts/jlens_lamini_gpt2_124m_40_twin.pt` repeats the whole
   comparison in the convention the lens matrices were fitted in and writes
   `output/exp017_jspace_hfframe.json`, 114 seconds. Both feed section 3.6, and
   `make_tables.py` renders them into `output/tables.md` alongside the
   registered tables. `python3 harness_check_jspace.py _harness_p200` re-runs
   the cheap harness check with the shuffle count taking effect, 48 seconds,
   writing to its own file so that the committed harness artifact stands.

**EXP_017 COMPLETE. H18: SUPPORTED. H18a: REFUTED. H18b: SUPPORTED at exactly
the pre-registered threshold, in both coordinate conventions, on different band
layers in each, and with the cross-check that section 3.4 offered as the
substantive claim holding in one convention only.**

**Amended 2026-09-06.** Until that date this closing line read "H18b: SUPPORTED
at exactly the pre-registered threshold, with the cross-check carrying the
substantive claim". The second half of that is withdrawn: section 3.6 shows the
cross-check reverses when the same comparison is measured in the convention the
lens matrices were fitted in, and section 3.4 carries the retraction in full.
The verdict itself is unchanged and is unchanged in both conventions.
