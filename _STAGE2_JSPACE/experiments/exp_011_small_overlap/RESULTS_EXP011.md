# EXP_011 results record: does GPT-2 Small's settled state hold anything the model could say?

**Spec:** `../../EXP_011_SPEC.md`, pre-registered and committed at `593e1f9` on
branch `claude/latent-context-small-llms-u2jdig-exp011`, before any state was
measured. **Register:** `_STAGE2_JSPACE/REGISTER.md`, hypotheses H6, H16, H16a
and H16b, allocated by erratum (f) on 2026-09-05. **Tracker:** issue #78.
**Proposed register rows:** `REGISTER_VERDICTS.md` in this directory. This
branch does not edit the register.

---

## 2026-09-05: the answer first

**The loop's settled states hold less of what GPT-2 Small could put into words
than the same prompts hold before the loop runs, and at matched injection
strength a settled state made from language is indistinguishable from one made
from random noise.** Those are the two results, and only the first of them is a
hypothesis that came out supported.

In numbers. The measure is the **J-space share**: the fraction of a state's
squared length that can be built out of at most 25 of the 50,257 directions the
Jacobian lens assigns to vocabulary words, using positive weights only. It runs
from 0, none of the state, to 1, all of it. Across the six workspace-band layers
(layers 5 to 10 of 12), the loop's settled language states hold a median share
between 0.0137 and 0.0217, while the same 125 prompts run once with no loop hold
between 0.0157 and 0.0348 at the same layers. The settled state is lower at five
of those six layers, on 64 to 96 percent of the 125 prompt-by-prompt pairs, with
a permutation p of 0.0032 at one layer and 0.0001 at the other four, where
0.0001 is the smallest value 10,000 permutations can report. That is H16b,
**SUPPORTED**.

Set beside that, the loop's settled language states and the matched-strength
noise states are the same to the lens. Their median shares differ by between
-0.0029 and +0.0002 on shares of about 0.015 to 0.022, which is a difference of
at most one part in five and usually far less, and in the band the sign runs
against the hypothesis more often than for it. Restricting the noise arm to the
90 trials that passed the convergence gate shrinks the gap further, to between
-0.00046 and -0.00001. That is H16, **NOT SUPPORTED**, and it is the same
inversion that lucier finding F4 reports from the basin counts, now seen through
a completely different instrument.

Two things the hypotheses did not ask, which matter more than two of the
verdicts. First, **the mid-layers say things the last layer does not.** The
directions the decomposition picks for the `prolet` attractor at layers 5, 8 and
10 are the words `the`, `Marx`, `Lenin`, `prolet`, `horizont`, `princ` and
`comrade`: a political vocabulary field. At layer 11, where the lens is the
identity and the reading is the ordinary logit lens, the same decomposition
picks only `the`, `,`, `and`, `"` and `in`: function words carrying no content
at all. Second, **that small slice is causally in charge of the output.** At
layer 5 the `prolet` state's J-space component is 2.3 percent of its squared
length; deleting just that component and putting the state back into the model
changes the model's own top output word from ` prolet` at probability 0.086 to
` abstract` at 0.027, and doubling only that component raises ` prolet` to 0.131.
At layer 10 doubling it raises ` prolet` from 0.086 to 0.569. A 1.5 to 2.3
percent slice of the state decides what the model says.

### The four verdicts on their registered wordings

| Hypothesis | Registered claim, in short | Verdict | The number that decides it |
|---|---|---|---|
| H6 | GPT-2 Small's five basin tensors project significantly more onto the J-space than the 18 null-model basins | **NOT SUPPORTED** | The five basins beat the eighteen with a one-sided p below 0.05 at three of the six band layers (layers 5, 6 and 7, p = 0.023, 0.015, 0.015), and lose at layer 10 (p = 0.033 the other way). The pre-registered rule needed four of six. |
| H16 | Language terminals hold a higher share than the matched-strength noise terminals at the band layers, above the random-dictionary chance level | **NOT SUPPORTED** | Language exceeds noise at two of six band layers, by +0.0002 both times, against shares of 0.015 to 0.022, and no band layer reaches p below 0.05 in that direction. Three band layers run significantly the other way. |
| H16a | The `prolet` attractor's share exceeds the `Divine` cycle's share in both phases | **NOT SUPPORTED** | `prolet` beats phase A at three of six band layers and phase B at one of six. The pre-registered rule needed four of six against each phase. The phases straddle `prolet`, but not in the direction finding F16 reported. |
| H16b | ATR terminal states have a lower share than ordinary non-iterated prompt residuals at the same layer | **SUPPORTED** | The settled state is lower at five of the six band layers, with permutation p of 0.0032, 0.0001, 0.0001, 0.0001 and 0.0001, and median paired differences of -0.0041, -0.0072, -0.0063, -0.0189 and -0.0170 against terminal shares of 0.0137 to 0.0217. |

---

## The shape of the instrument, measured

This is the single most important thing to know before reading any share below,
and it was not anticipated in the specification. It is measured, not inferred.

The 50,257 lens directions at a layer are not spread out over the
768-dimensional space. They are packed around one common direction. Taking the
average of all 50,257 of them after setting each to unit length gives a vector
whose own length is **0.786 to 0.872 across the six band layers**, on a scale
where 0 would mean perfectly even spread over the sphere and 1 would mean every
direction identical. For comparison, the norm-matched random dictionary used as
one of the controls gives **0.0044** on the same scale. Between 99.0 and 99.6
percent of the lens directions lie on the positive side of that common
direction. At layer 11, where the lens is the identity and the reading is the
ordinary logit lens, the concentration is 0.518 and every single direction lies
on the positive side.

Two consequences follow, and both matter for reading the tables.

First, the positive cone the decomposition may reach is narrow. A state pointing
away from the dictionary's common direction cannot be built out of these
directions with positive weights, no matter how many are allowed. Measured: the
loop's settled language states sit at median cosine -0.182, -0.170 and -0.173 to
that common direction at layers 5, 6 and 7, where 0 is what an unrelated
direction would give, while the same prompts' ordinary residuals sit at -0.070,
-0.040 and -0.035 at the same three layers. This is why the search usually
stops after three or four directions rather than the 25 it is allowed: there is
nothing left that any direction points toward. When it stops for that reason the
answer is not an approximation. It is the exact nearest point of the positive
cone over all 50,257 directions, so the 25-direction limit is not what is
holding these numbers down.

Second, the norm-matched random dictionary is not a like-for-like chance level.
Its cone is nearly the whole space, so it captures more of almost any state for a
reason that has nothing to do with the lens's content. The rotated-lens control
keeps the concentration exactly, rotating the whole cloud rigidly, and only
destroys where it points. Where the two controls disagree below, the rotated
lens is the informative one, and this is flagged at each place.

---

## The four hypotheses in detail

Throughout, a **layer** is the residual stream at the output of one of GPT-2
Small's twelve blocks, and the **band layers** are 5 to 10 inclusive, the paper's
workspace band mapped onto twelve layers (0.38 times 12 is 4.56, rounding inward
to 5; 0.92 times 12 is 11.04, rounding inward to 11) intersected with the layers
the lens was fitted for, which are 0 to 10. Layer 11 is the identity, so its
reading is the ordinary logit lens. Band layers are shown in bold in every table.

### H6: the five basins against the eighteen null-model basins

**NOT SUPPORTED on the registered wording.**

H6 compares one representative of each of GPT-2 Small's five basins, which are
the five words its settled states read out as, with one representative of each
of the 18 basins the original noise arm produced. The representative is each
basin's medoid, the member whose average cosine to the other members of its own
basin is highest, chosen by a rule written down before any share was seen. Both
sides are read at iteration 100 of their own original sweep, which is what makes
them comparable.

| layer | five basins, median | eighteen null basins, median | one-sided p (basins greater) | five basins, random-dictionary control |
|---|---|---|---|---|
| 0 | 0.0305 | 0.0706 | 0.9999 | 0.3681 |
| 1 | 0.0442 | 0.0718 | 0.9998 | 0.3672 |
| 2 | 0.0254 | 0.0262 | 0.6808 | 0.3715 |
| 3 | 0.0155 | 0.0149 | 0.5715 | 0.3733 |
| 4 | 0.0149 | 0.0112 | 0.2016 | 0.3744 |
| **5** | 0.0213 | 0.0106 | 0.0229 | 0.3730 |
| **6** | 0.0149 | 0.0076 | 0.0151 | 0.3701 |
| **7** | 0.0203 | 0.0120 | 0.0151 | 0.3717 |
| **8** | 0.0135 | 0.0144 | 0.4856 | 0.3746 |
| **9** | 0.0152 | 0.0172 | 0.5431 | 0.3749 |
| **10** | 0.0145 | 0.0288 | 0.9723 | 0.3748 |
| 11 | 0.0878 | 0.0102 | 0.0045 | 0.3714 |

The five basins beat the eighteen at layers 5, 6 and 7 with one-sided p of
0.023, 0.015 and 0.015, and lose at layer 10 with p of 0.033 in the reverse
direction. Three supporting layers is one short of the pre-registered majority of
four, so the verdict is NOT SUPPORTED rather than SUPPORTED. The direction is not
weak where it holds: at layer 6 the basins sit at 0.0149 against the null basins'
0.0076, roughly double.

Reported alongside, not scoring, and much sharper because it uses every state
rather than one per basin: comparing all 125 language terminals with all 125
original-noise terminals gives one-sided p of 2e-14, 7e-12 and 5e-12 at layers 5,
6 and 7, and p of 1.000 at layers 8, 9 and 10 where the comparison reverses. So
the language-versus-original-noise difference is real and large in the first half
of the band and real and large in the opposite direction in the second half. A
single verdict on this comparison would have been the wrong shape of answer.

**Read this verdict with the caveat the register already carries.** The 18-basin
arm is the one lucier finding F4 records as mis-scaled. Measured here from the
same files: that arm was injected at an average length of 401.3, while the
language arm was injected at 1427.5, so it ran at 28 percent of the language
arm's strength, and its labels were read at iteration 100 before it had settled.
H6's comparison is therefore between a settled arm and an unsettled, quieter arm.
Whatever this verdict is, it is not evidence about language against noise at
matched strength. H16 is that question, and it is answered next.

### H16: language terminals against the matched-strength noise terminals

**NOT SUPPORTED on the registered wording.**

| layer | language, median | noise, median | difference | permutation p | language above random-dictionary control | language above rotated-lens control |
|---|---|---|---|---|---|
| 0 | 0.0313 | 0.0320 | -0.0007 | 0.8680 | no | yes |
| 1 | 0.0448 | 0.0459 | -0.0011 | 0.8544 | no | yes |
| 2 | 0.0260 | 0.0284 | -0.0023 | 0.9999 | no | yes |
| 3 | 0.0160 | 0.0173 | -0.0013 | 0.9991 | no | no |
| 4 | 0.0154 | 0.0164 | -0.0011 | 0.9972 | no | yes |
| **5** | 0.0217 | 0.0224 | -0.0006 | 0.9909 | no | yes |
| **6** | 0.0152 | 0.0150 | +0.0002 | 0.1405 | no | yes |
| **7** | 0.0204 | 0.0202 | +0.0002 | 0.2507 | no | yes |
| **8** | 0.0137 | 0.0166 | -0.0029 | 1.0000 | no | yes |
| **9** | 0.0153 | 0.0168 | -0.0015 | 0.9149 | no | yes |
| **10** | 0.0145 | 0.0174 | -0.0029 | 0.9999 | no | no |
| 11 | 0.0899 | 0.0922 | -0.0023 | 0.8955 | no | yes |

The two families are the same to this instrument. Across the six band layers the
median difference runs from -0.0029 to +0.0002 on shares of 0.0137 to 0.0224, and
the two layers where language is ahead are ahead by +0.0002, which is one part in
a hundred of the share. No band layer reaches p below 0.05 in the hypothesised
direction; three run significantly the other way. Restricting the noise arm to
the 90 trials that passed the convergence gate, so that both sides are settled,
shrinks the gap to between -0.00046 and -0.00001 and leaves no layer close to
significance in either direction. This is the F4 inversion seen by a second,
independent instrument: at matched injection strength, what the loop settles on
is a property of the weights, not of language-shaped input.

**The third condition in the registered rule failed for a reason that is about
the instrument, not about the states.** H16's wording requires the language
median to sit above the norm-matched random-dictionary chance level. It does not,
at any layer, and neither does anything else measured here: that control sits at
0.369 to 0.375 while every real family sits between 0.008 and 0.036. The gap is a
factor of about 25 and it is entirely explained by the shape finding above. A
random dictionary's positive cone is nearly the whole space; the lens's is
narrow. **Against the rotated-lens control, which preserves the lens's shape
exactly, the picture is the opposite: the language terminals sit above chance at
five of the six band layers** (0.0217 against 0.0135 at layer 5, 0.0152 against
0.0135 at layer 6, 0.0204 against 0.0139 at layer 7, 0.0137 against 0.0136 at
layer 8, 0.0153 against 0.0132 at layer 9, and 0.0145 against 0.0149 at layer 10,
the one exception). The settled states are therefore modestly more
lens-expressible than a rigidly rotated lens would make them, which the
registered chance level was unable to show. Changing which control counts is a
ruling for the operator, not a change this record makes on its own; it is
decision item 1 below.

### H16a: the `prolet` attractor against the `Divine` cycle's two phases

**NOT SUPPORTED on the registered wording, and finding F16's phase assignment
does not survive the move to the full vocabulary.**

| layer | prolet | Divine phase A | Divine phase B | Divine pivot M | prolet minus phase A | prolet minus phase B | control spread (one standard deviation) |
|---|---|---|---|---|---|---|---|
| 0 | 0.0352 | 0.0250 | 0.0272 | 0.0274 | +0.0102 | +0.0079 | 0.1777 |
| 1 | 0.0485 | 0.0264 | 0.0269 | 0.0272 | +0.0221 | +0.0216 | 0.1781 |
| 2 | 0.0283 | 0.0115 | 0.0270 | 0.0195 | +0.0168 | +0.0013 | 0.1779 |
| 3 | 0.0175 | 0.0051 | 0.0154 | 0.0100 | +0.0124 | +0.0021 | 0.1795 |
| 4 | 0.0170 | 0.0042 | 0.0150 | 0.0081 | +0.0128 | +0.0020 | 0.1778 |
| **5** | 0.0231 | 0.0105 | 0.0214 | 0.0161 | +0.0126 | +0.0017 | 0.1775 |
| **6** | 0.0162 | 0.0062 | 0.0197 | 0.0101 | +0.0100 | -0.0034 | 0.1797 |
| **7** | 0.0213 | 0.0101 | 0.0283 | 0.0159 | +0.0113 | -0.0070 | 0.1810 |
| **8** | 0.0141 | 0.0153 | 0.0288 | 0.0169 | -0.0012 | -0.0147 | 0.1799 |
| **9** | 0.0152 | 0.0226 | 0.0322 | 0.0215 | -0.0074 | -0.0170 | 0.1818 |
| **10** | 0.0147 | 0.0363 | 0.0265 | 0.0243 | -0.0216 | -0.0118 | 0.1786 |
| 11 | 0.0912 | 0.0164 | 0.1516 | 0.0602 | +0.0748 | -0.0603 | 0.1587 |

The `prolet` attractor is above phase A at layers 5, 6 and 7 and below it at
layers 8, 9 and 10; it is above phase B at layer 5 only. Three of six against
phase A and one of six against phase B, where the rule needed four of six against
each. So the phases straddle `prolet`, which is the shape finding F16 describes,
but they straddle it the other way round. F16, working with the pilot's 193-word
dictionary, reported phase A above `prolet` at every layer and phase B below
`prolet` at every layer. On the full 50,257-word lens the assignment inverts:
phase B is the more lens-expressible phase at eleven of twelve layers, and phase
A is the less lens-expressible one through the early band. **This is a retraction
of a directional claim, stated plainly: F16 says phase A is the more verbalizable
phase and phase B the less; measured on the full-vocabulary lens, phase B is the
more verbalizable phase and phase A the less.** What survives from F16 is its
structural point, that the cycle is not one object with respect to the lens but
swings between a more expressible and a less expressible phase.

The pivot, the midpoint of the two phases, does not come out as the most
lens-expressible object here: it sits between the phases at every band layer, for
example 0.0169 at layer 8 against phase A's 0.0153 and phase B's 0.0288. F16
reported the pivot as the most lens-expressible state it probed. That comparison
is not like-for-like and should not be read as a contradiction: F16's pivot mixed
two different scalings, while this one rescales both phases to the loop's own
starting length before averaging. The deviation is recorded below.

**The materiality floor written into the specification turned out to be
uninformative, and is replaced by a labelled substitute.** The specification said
a gap smaller than the spread of the state's own control shares across the six
control runs should be marked as inside the control spread. Because the two
control types differ by a factor of 25, that pooled spread is 0.178 to 0.182 at
every band layer, which swamps every gap in the table and would mark everything
as immaterial. The informative yardstick is the spread within one control type:
across the three rotated-lens seeds the `prolet` state's share varies with a
standard deviation of 0.0009 to 0.0048 at the band layers. Measured against that,
the gaps against phase A at layers 5, 6 and 7 (+0.0126, +0.0100, +0.0113) and
against phase B at layers 8, 9 and 10 (-0.0147, -0.0170, -0.0118) are two to
twelve times the seed spread and are material; the gaps against phase B at layers
5, 6 and 7 (+0.0017, -0.0034, -0.0070) are within about one to four times it and
the smallest of them should not be leaned on. This substitution is post hoc, it is
labelled as such, and it changes no verdict: the verdict rule counted signs, not
sizes.

### H16b: settled states against the same prompts' ordinary residuals

**SUPPORTED on the registered wording.**

| layer | terminal, median | ordinary residual, median | median paired difference | permutation p (terminal lower) | share of the 125 pairs with the terminal lower |
|---|---|---|---|---|---|
| 0 | 0.0313 | 0.0257 | +0.00507 | 0.9051 | 44 percent |
| 1 | 0.0448 | 0.0422 | -0.00185 | 0.3229 | 53 percent |
| 2 | 0.0260 | 0.0358 | -0.01481 | 0.0001 | 74 percent |
| 3 | 0.0160 | 0.0306 | -0.01898 | 0.0001 | 92 percent |
| 4 | 0.0154 | 0.0245 | -0.01267 | 0.0001 | 84 percent |
| **5** | 0.0217 | 0.0243 | -0.00410 | 0.0032 | 64 percent |
| **6** | 0.0152 | 0.0214 | -0.00719 | 0.0001 | 79 percent |
| **7** | 0.0204 | 0.0251 | -0.00632 | 0.0001 | 67 percent |
| **8** | 0.0137 | 0.0345 | -0.01891 | 0.0001 | 96 percent |
| **9** | 0.0153 | 0.0348 | -0.01698 | 0.0001 | 94 percent |
| **10** | 0.0145 | 0.0157 | +0.00137 | 0.9051 | 46 percent |
| 11 | 0.0899 | 0.1556 | -0.07483 | 0.0001 | 93 percent |

At five of the six band layers the settled state holds less of what the lens can
express than the same prompt's ordinary, un-looped residual at the same layer.
The effect is not a whisker: at layer 8 the settled median is 0.0137 against
0.0345, and 96 percent of the 125 prompt-by-prompt pairs run in that direction;
at layer 9 it is 0.0153 against 0.0348 with 94 percent of pairs. Layer 10 is the
exception, where the settled median is 0.0145 against 0.0157 and only 46 percent
of pairs run the predicted way, which is chance. At layer 11, outside the band,
the gap is the largest of all: 0.0899 against 0.1556, with 93 percent of pairs.

The plain reading is that the loop moves the state out of the directions the
model uses to say words. It does not follow that the loop moves the state
nowhere: the settled states still sit above the rotated-lens chance level at five
of six band layers, as noted under H16. Both can be true, and together they say
the settled state is somewhat lens-expressible but distinctly less so than an
ordinary residual.

**One secondary reading runs the other way and is reported because it does.**
The specification pre-registered a second comparison against the ordinary residual
averaged over all token positions rather than read at the last one. On that
comparison the settled state is *higher* than the ordinary residual at layers 5
through 8 (median paired differences of +0.0166, +0.0124, +0.0144 and +0.0035)
and lower only at layers 9 and 10. So the H16b result holds for the last-position
reading, which is the one the loop itself works with and the one the verdict was
pre-registered on, and does not hold for the position-averaged reading. A reader
should treat H16b as a statement about last-position residuals specifically.

---

## The descriptive readings, which carry no verdict

### What the mid-layers say that the last layer does not

The decomposition does not only give a number. It names which vocabulary words'
directions it used. Ranking those directions by the length each one contributes
to the reconstruction, for the `prolet` attractor:

| layer | the words whose lens directions build the state |
|---|---|
| 5 | ` the`, ` Marx`, `,`, ` Lenin` |
| 8 | ` the`, ` Marx`, ` prolet`, ` horizont` |
| 10 | ` the`, ` prolet`, ` horizont`, ` princ`, ` Marx` |
| 11 | ` the`, `,`, ` and`, ` "`, ` in` |

Layer 11 is the ordinary logit lens, because the lens's matrix there is the
identity. It finds nothing but function words. Layers 5 through 10, where the
lens actually transports the state, find a political vocabulary field:
`Marx`, `Lenin`, `prolet`, and word-fragments like `horizont` and `princ` that
begin `horizontal` and `principle`. So yes, on this instrument the mid-layers say
things the final layer does not, and what they say is thematically of a piece
with the word the loop settles on. This is a readout, not a causal claim on its
own; the clamping check below is the causal part.

The same table for the `Divine` cycle's two phases is the more surprising one.
The cycle reads out the word ` Divine` in **both** phases, which is finding F9's
phase-invariant argmax. The lens does not agree that both phases are made of it:

| layer | phase A's directions | phase B's directions |
|---|---|---|
| 5 | ` the`, `―`, ` streng` | ` the`, `―`, ` Fairy`, ` Divine` |
| 8 | ` the`, ` streng`, ` arrang`, ` princ` | ` the`, ` Divine`, ` Yu`, ` Fuji` |
| 10 | ` the`, ` streng`, ` seiz`, ` arrang`, ` neighb` | `,`, ` Divine`, ` seiz`, ` Yu`, ` the` |
| 11 | ` the`, `,`, `\n`, ` N` | ` the`, `,`, `\n`, ` and`, ` in` |

Phase B is built partly out of the ` Divine` direction at layers 5, 8 and 10.
Phase A is not, at any band layer: its directions are `streng`, `arrang`, `seiz`,
`neighb`, the beginnings of `strength`, `arrange`, `seize` and `neighbour`. The
two halves of one cycle, which say the same word out loud, are made of different
material inside. This is an observation, not an explanation, and it is new: the
phase-blind pilot could not have seen it and the restricted 193-word pilot
dictionary did not contain most of these words.

### The clamping check: a two percent slice decides the output

At each band layer the state was split into its J-space component and everything
else, only the component was rescaled by a factor of 0, 0.5, 1 or 2, and the
result was read two ways: through the lens, and causally, by splicing the
modified state back into the model at that layer and letting the remaining blocks
run to the model's own output distribution. A factor of 1 reproduces the
untouched state exactly and serves as the control.

For the `prolet` attractor at layer 5, whose J-space component is 2.3 percent of
the state's squared length and 15.2 percent of its length:

| factor on the component | the model's own top three output words |
|---|---|
| 0 (component deleted) | ` abstract` 0.027, ` anarchism` 0.026, ` bourgeois` 0.024 |
| 0.5 | ` prolet` 0.050, ` bourgeois` 0.046, ` Anarch` 0.039 |
| 1 (untouched) | ` prolet` 0.086, ` bourgeois` 0.066, ` Anarch` 0.061 |
| 2 | ` prolet` 0.131, ` Marx` 0.102, ` bourgeois` 0.082 |

Deleting a component that is 2.3 percent of the state's squared length changes
what the model says. Doubling it raises the settled word's probability from 0.086
to 0.131. At layer 10 the same operation is stronger still: deleting the
component leaves ` the` on top at 0.049, and doubling it raises ` prolet` from
0.086 to **0.569**, a factor of 6.6. On the paper's own account this is the
expected shape of result, and it reproduces here in a 124-million-parameter model:
the J-space component is a small fraction of the state and close to the whole of
its report.

**The `Divine` cycle behaves differently, and this is the sharper observation.**
Phase A's J-space component at layer 5 is 1.05 percent of its squared length, and
deleting it changes nothing the model says: ` Divine` stays on top at 0.251
against 0.225 untouched. Only doubling the component disturbs the readout, at
which point the top word becomes the bracket character `【` at 0.255. The same
pattern holds at every band layer. So the `prolet` basin's output word is carried
by its J-space component, and the `Divine` cycle's output word is not. Marked as
inferred rather than established: one attractor and one cycle, one trajectory
each, no repeats.

### The flip axis

The `Divine` cycle's symmetric flip axis, the single direction the loop negates
each pass, holds a J-space share between 0.0011 and 0.0307 in one sign and
between 0.0053 and 0.0152 in the other across the band layers, against a
rotated-lens chance level of about 0.013 to 0.015 for states at those layers. So
the axis is at or below chance inside the band, which agrees in direction with
finding F16's statement that the flip axis is almost entirely outside the lens.
A caution the pilot also gave: a share is sign-dependent for a direction, because
the combination must use positive weights, so both signs are reported and neither
alone is the answer.

### Where the band actually is on this instrument

Reading the whole curve rather than the band alone, the shares are highest at
layers 0 and 1, fall through layers 3 to 6, rise a little at layers 7 to 9, and
jump at layer 11. The workspace band the paper describes, a rise in the middle
and a fall at the ends, is not visible in GPT-2 Small on this lens for any family
measured. That is a negative observation about the instrument at this scale, and
it echoes what EXP_012m found for GPT-2 Medium, which the record states as NO
COHERENT BAND. It also means the band layers used for scoring are a mapping
imported from the paper, not a structure this measurement independently confirms.

---

## The figure

![J-space share by layer](output/exp011_share_curves.png)

Three panels, all with the workspace band shaded in gold. **Left:** the median
share by layer for each family, on a logarithmic vertical scale because the
random-dictionary control sits about 25 times higher than everything else and
would otherwise flatten the picture; shaded regions are the middle half of each
family. **Middle:** the named single states, on a linear scale, showing the
`prolet` attractor crossing from above both `Divine` phases in the early band to
below both in the late band. **Right:** each family's median share minus its own
rotated-lens control, so that above zero means more lens-expressible than a
rigidly rotated lens would make it. Ordinary residuals are the family furthest
above chance through the band; the two settled families sit just above zero; the
original noise arm crosses from above to below and back.

---

## The full per-layer table

| layer | language terminals | run-17 noise terminals | ordinary residuals | original noise arm | rotated-lens control (language) | random-dictionary control (language) |
|---|---|---|---|---|---|---|
| 0 | 0.0313 | 0.0320 | 0.0257 | 0.0723 | 0.0145 | 0.3688 |
| 1 | 0.0448 | 0.0459 | 0.0422 | 0.0680 | 0.0183 | 0.3693 |
| 2 | 0.0260 | 0.0284 | 0.0358 | 0.0248 | 0.0208 | 0.3703 |
| 3 | 0.0160 | 0.0173 | 0.0306 | 0.0143 | 0.0164 | 0.3705 |
| 4 | 0.0154 | 0.0164 | 0.0245 | 0.0110 | 0.0118 | 0.3735 |
| **5** | 0.0217 | 0.0224 | 0.0243 | 0.0108 | 0.0135 | 0.3712 |
| **6** | 0.0152 | 0.0150 | 0.0214 | 0.0078 | 0.0135 | 0.3694 |
| **7** | 0.0204 | 0.0202 | 0.0251 | 0.0127 | 0.0139 | 0.3727 |
| **8** | 0.0137 | 0.0166 | 0.0345 | 0.0176 | 0.0136 | 0.3739 |
| **9** | 0.0153 | 0.0168 | 0.0348 | 0.0223 | 0.0132 | 0.3750 |
| **10** | 0.0145 | 0.0174 | 0.0157 | 0.0353 | 0.0149 | 0.3721 |
| 11 | 0.0899 | 0.0922 | 0.1556 | 0.0078 | 0.0482 | 0.3690 |

Every number is the median J-space share for that family at that layer, on a
0-to-1 scale. The last two columns are the same 125 language terminals scored
against the two chance levels. The full table, every family against every arm
with quartiles, is `output/per_layer_shares.csv`; the named single states are in
`output/verdicts.json` and drawn in `output/exp011_share_curves.png`.

### How many directions the search actually used

| layer | language terminals | run-17 noise | ordinary residuals | original noise arm |
|---|---|---|---|---|
| 0 | 6 | 9 | 10 | 15 |
| 1 | 7 | 9 | 13 | 17 |
| 2 | 8 | 8 | 10 | 15 |
| 3 | 5 | 6 | 7 | 9 |
| 4 | 4 | 4 | 7 | 6 |
| **5** | 4 | 4 | 7 | 5 |
| **6** | 3 | 4 | 7 | 2 |
| **7** | 3 | 3 | 7 | 4 |
| **8** | 5 | 5 | 9 | 5 |
| **9** | 6 | 7 | 9 | 6 |
| **10** | 6 | 7 | 8 | 7 |
| 11 | 5 | 5 | 6 | 13 |

The limit was 25. The search almost never reached it, because it ran out of
directions with a positive correlation first, which as explained above means the
answer is the exact nearest point of the whole positive cone rather than a
25-direction approximation. Ordinary residuals use about twice as many
directions as settled states do, which is the same fact as their higher shares
seen from another side.

### The named states

| layer | prolet1000 | phaseA | phaseB | pivotM | noise1000 |
|---|---|---|---|---|---|
| 0 | 0.0352 | 0.0250 | 0.0272 | 0.0274 | 0.0657 |
| 1 | 0.0485 | 0.0264 | 0.0269 | 0.0272 | 0.0615 |
| 2 | 0.0283 | 0.0115 | 0.0270 | 0.0195 | 0.0232 |
| 3 | 0.0175 | 0.0051 | 0.0154 | 0.0100 | 0.0135 |
| 4 | 0.0170 | 0.0042 | 0.0150 | 0.0081 | 0.0104 |
| **5** | 0.0231 | 0.0105 | 0.0214 | 0.0161 | 0.0108 |
| **6** | 0.0162 | 0.0062 | 0.0197 | 0.0101 | 0.0078 |
| **7** | 0.0213 | 0.0101 | 0.0283 | 0.0159 | 0.0125 |
| **8** | 0.0141 | 0.0153 | 0.0288 | 0.0169 | 0.0170 |
| **9** | 0.0152 | 0.0226 | 0.0322 | 0.0215 | 0.0212 |
| **10** | 0.0147 | 0.0363 | 0.0265 | 0.0243 | 0.0341 |
| 11 | 0.0912 | 0.0164 | 0.1516 | 0.0602 | 0.0067 |

---

## What was measured, in ordinary words

The Jacobian lens gives, for every one of the model's 50,257 vocabulary tokens
and every layer, one direction in the layer's 768-dimensional activity space:
the direction along which activity at that layer pushes the model's final score
for that word. Those 50,257 directions are the **lens vectors**. The paper this
follows defines the **J-space** at a layer as everything you can build by adding
together at most 25 of those directions with positive weights only, never
negative ones. The **J-space share** of a state is the fraction of the state's
squared length that the closest such combination captures. It runs from 0, none
of the state, to 1, all of it. The share does not change if the state is made
longer or shorter, so it measures direction and shape, not size.

The paper reports that this share is small even for the states it is designed to
detect: never more than 10 percent of a state, and a median of 6 to 7 percent for
clean single-concept vectors in a much larger model. The numbers below, one to
five percent in a 124-million-parameter model, sit in that same regime and should
be read against that expectation rather than against an intuition that a
"verbalizable" state ought to score near 1.

Finding the closest such combination is done by non-negative orthogonal matching
pursuit: add the direction most aligned with what is left over, refit all the
chosen weights under the constraint that none may go negative, repeat, and stop
at 25 directions or when nothing is left that any direction points toward. When
the search stops for the second reason, which happens often here, the answer is
not merely a good 25-direction approximation: it is the exact closest point of
the whole positive cone over all 50,257 directions, so the 25-direction limit is
not what is holding the number down.

## The two controls, and why both are needed

A share is meaningless on its own, because 25 directions chosen greedily from
50,257 candidates will capture some of any vector. Two chance levels were
pre-registered.

**The rotated lens.** Turn the entire set of 50,257 lens directions rigidly by
one random rotation of the 768-dimensional space. Every direction keeps its
length and every pair of directions keeps its angle, so the shape of the whole
cloud is untouched; only its alignment with the model's real geometry is
destroyed. Three rotations, seeds 2026, 2027 and 2028.

**The norm-matched random dictionary.** Replace every lens direction by an
independent random direction of the same length. Three seeds, 4242, 4243 and
4244. This is the control the earlier pilot used and the one the registered
wording of H16 names.

The two are not interchangeable, and the difference matters for reading every
number below. The lens directions are not spread evenly: they are all clustered
around one common direction. Measured here, the average of the 50,257 unit lens
directions at layer 6 has length 0.87 on a scale where 0 would mean perfectly
even spread and 1 would mean all directions identical, and 99.3 percent of the
lens directions lie on the positive side of that common direction. A random
dictionary has no such clustering, so its positive cone is far wider and will
capture more of almost any state. The rotated lens keeps the clustering exactly
and only moves where it points, so it is the control that isolates the lens's
content from the lens's shape. Both are reported; where they disagree, the
rotated lens is the more informative one and the results say so.

---

## What was measured on

Five families of states, all read at every one of the twelve layers, where
"layer l" means the residual stream at the output of block l.

1. **The 125 language terminals.** The settled state of each of the 125 Stage 1
   prompts, at iteration 100 of the loop. Their five read-out labels at that
   iteration are `prolet` for 44 prompts, `Divine` for 34, `Anarch` for 26,
   `till` for 19 and `solidarity` for 2. Those five labels are the five basins.
2. **The 125 run-17 noise terminals.** Stage 1 run 17, the corrected null: 125
   random Gaussian tensors, each pair-matched to one prompt's exact sequence
   length and starting length, iterated under the convergence gate. Its average
   injection length is 1427.5 against the original noise arm's 401.3, which is
   the mis-scaling that lucier finding F4 records.
3. **The 125 original-noise terminals.** The first noise arm, at iteration 100,
   whose 18 distinct read-out labels are the "18 null-model basins" that H6
   names. Known to be mis-scaled; used only because H6's registered wording
   names it.
4. **The 125 ordinary prompt residuals.** The same 125 prompts run once through
   the model with no injection and no iteration, read at the last token
   position. The token count matched the loop's own recorded sequence length for
   125 of 125 prompts, so the two arms of the H16b pairing see the same text.
5. **Named single states.** The `Divine` cycle's phase A, phase B and pivot, the
   `prolet` attractor at iteration 1000, the pilot's committed noise state, and
   the five committed converged prompt tensors from the pilot era.

A settled state as stored is the model's output, that is, a layer-11 object. To
ask the question at every layer, each state was put back through the model
exactly as the loop does: rebuild the full token tensor, rescale it to the
loop's own starting length, splice it into the model's input residual stream,
run one forward pass, and record what the residual stream holds at the output of
every block. That is not a reinterpretation of the state; it is the same
computation the loop performed to produce it.

---

## The checks that had to pass before any number was read

Four gates were written into the specification before the run and all four
passed. They are listed here because a reader should know the measurement was
not simply asserted.

**Position collapse is exact.** The loop's settled states have every token
position holding the identical vector. Measured on the committed tensors: the
smallest cosine between any two token positions is 1.000000 on a scale where 1
means identical direction and 0 means unrelated, and the largest-to-smallest
ratio of position lengths is 1.0000. The 125 language runs record the same thing
independently, with a position-similarity between 0.9999996 and 1.0000005 where
1.0 is exact. Consequence: a settled state is fully described by one
768-number vector, so rebuilding the full tensor by repeating that vector is
exact rather than approximate.

**The tiling identity.** Reading the layer-by-layer states from the stored full
tensor and from the repeated last-position vector gives the same answer: the
smallest cosine between the two readings across all twelve layers is 0.9999998,
and the largest absolute difference in any of the 768 numbers is 0.0007 against
state lengths above 1500.

**The reconstruction gate.** A settled state that is a true resting point must
come back unchanged after one loop step. It does: 91 of the 125 language states
and 90 of the 125 run-17 noise states return their own layer-11 reading with a
cosine above 0.999, against a pre-registered requirement of 85 in each. The
states that do not pass are not failures of the method. They are exactly the
period-2 cycles: the 34 language prompts whose read-out is `Divine`, and exactly
the 35 noise trials that the convergence gate recorded as not settling at lag 1.
For those, one loop step returns the cycle's other phase: the language failures
fall between cosine 0.651 and 0.702, the noise failures between 0.671 and 0.985,
against the cycle's own recorded swing of 0.685.

**The Divine cycle gate.** The two phases were rebuilt from the committed
iteration-1000 state and checked against the recorded cycle anatomy: the cosine
between the phases came out 0.68491167 against the recorded 0.68491167, and the
cosine between phase A and phase A after two steps came out 1.00000000 against
the recorded 1.00000000.

**The decomposition self-test.** Before every run, the decomposition is asked to
recover a state deliberately built from five known dictionary atoms with positive
weights. It recovers share 1.000000 using exactly those five atoms, while a
generic random state in the same 768 dimensions reaches only 0.263 with the full
25 atoms allowed. Batching many states together changes no answer, and rescaling
a state changes its share by less than one part in ten million.

**Cross-check against the previous pilot's published numbers.** The same
decomposition code was run on the 2026 pilot's own 193-atom dictionary and the
pilot's own states. It reproduces the pilot's published sparse shares to within
0.002 to 0.009 on a 0-to-1 scale, always slightly higher, which is what a better
solver should do, and it reproduces the pilot's ordering of the `Divine` state
against the `prolet` attractor at 12 of 12 layers including the single layer
where the pilot found the reverse. Script `pilot_crosscheck.py`, log
`output/exp011_pilot_crosscheck.log`.

### A side point that was blocking other work, now settled

The operator report of 2026-07-31, section 5 item 3, records a suspicion that a
note in the lens results has the token handling backwards: the note claims the
looping engine does not prepend a start-of-text marker to its prompts, while the
pinned library's default suggests it does. The report says this "must be settled
before any lens-on-terminals work because it decides whether the two instruments
were even reading the same states". It is now settled, and the note is the thing
that is backwards.

The evidence is arithmetic and exhaustive. For each of the 125 prompts, the loop
recorded the sequence length it ran at. Tokenising the same prompt with the
marker prepended gives that exact length for **125 of 125 prompts**. Tokenising
it without the marker gives that length for **0 of 125**, and gives exactly one
token fewer for **125 of 125**. For example the physics prompt was run at length
10; with the marker it tokenises to 10, without it to 9. So the loop's sequences
include the prepended start-of-text marker, and any lens reading built on the
assumption that they do not is reading a different sequence length from the one
the loop used. This is established, not inferred.

---

## The instrument, and where it came from

The instrument is a **Jacobian lens**, a set of eleven 768 by 768 matrices, one
per layer, that carry a state at that layer forward into the final layer's frame
so that the model's own word-scoring machinery can be applied to it. It was not
fitted by this project. It was published by Neuronpedia, fitted with Anthropic's
own reference code (the `jlens` package, Apache-2.0 licensed, at pinned commit
`581d398613e5602a5af361e1c34d3a92ea82ba8e`), on WikiText-103, a standard corpus
of Wikipedia articles. The fit used a maximum of 128 tokens per prompt, bfloat16
arithmetic, and ran on one NVIDIA B200 graphics processor on 2026-06-11. It
asked for 1000 prompts and stopped early at its own convergence criterion after
**277 prompts**, at a final mean relative change of 0.0016 per prompt.

The file used here is
`_STAGE2_JSPACE/artifacts/jlens_gpt2_small_neuronpedia.pt`, 12,980,477 bytes,
SHA-256 digest `d1800a1335ada089ef2e1ec0e4bd4d5bd61e6011eacc31f8618fdb3d10aae762`.
The digest was recomputed at run time and is recorded in
`output/states_meta.json`. The lens holds one matrix for each source layer 0 to
10; the target is the last block, layer 11, so at layer 11 the matrix is the
identity and the reading is the ordinary logit lens.

## Versions

Python 3.11.15, torch 2.14.0 (processor only, no graphics card), numpy 2.4.6,
transformers 5.16.1, transformer_lens 3.8.1, scipy 1.17.1, jlens 0.1.0 installed
from the pinned clone. Every script sets the thread count to one, because
multi-threaded linear algebra was measured about five times slower on this
shared machine.

---

## Deviations from the specification

Recorded flat, whether or not they helped.

1. **The pivot of the `Divine` cycle was built on-shell, not in the mixed frame
   the specification named.** Section 2.1 said the pivot would be reconstructed
   exactly as the pilot script `10_jlens_phase.py` stage 1 does. That script
   builds the pivot from phase A at its raw length and phase B rescaled to the
   loop's starting length, which mixes two different scalings; the lucier
   record's caveat 15 identifies the resulting axis as radially contaminated and
   0.909 aligned with its own pivot. This run instead rescaled both phases to
   the loop's starting length before taking the midpoint, which is the symmetric
   on-shell construction the same record recommends. The pivot is descriptive
   only and carries no verdict, so no scoring rule is affected. The phases
   themselves are unaffected: the cycle gate reproduced the recorded cosine
   between phases, 0.68491167 against 0.68491167, and the recorded cosine
   between phase A and phase A two steps later, 1.00000000 against 1.00000000.

2. **The mean-centred secondary arm turned out to be the same arm.** Section 4
   pre-registered a second pass on mean-centred residuals, on the reasoning that
   the model's final LayerNorm discards the average of a state's 768 numbers
   before any word score is formed. Measurement showed the states are already
   exactly mean-centred as they come out of TransformerLens, which centres
   everything written into the residual stream: the largest average-component
   length across all 637 states and all twelve layers is 4.9 parts in one
   hundred million of the state's own length. The centred arm was run at every
   layer anyway rather than argued away, and its share medians agree with the
   raw arm's to within 1.2 parts in ten million, which is the largest difference
   in any family median at any layer. The raw-versus-centred question therefore
   does not arise for these states.

3. **The selected atoms were recorded for more states than the descriptive
   section asked for.** Section 7.5 item 2 named the `prolet` and `Divine`
   states. Atoms were also recorded for all 125 language terminals and all 125
   ordinary residuals, at no extra compute, so the readout can be examined
   beyond the named states. This is an addition, not a substitution.

4. **Two diagnostics were added after the specification was written.** The
   first is a cross-check against the previous pilot's published numbers, which
   exists to show the decomposition code reproduces a number already in the
   record. The second is the dictionary-shape measurement reported near the top
   of this record, which was prompted by seeing that the search kept stopping
   after three or four directions instead of the 25 it was allowed. Neither is a
   hypothesis test and neither carries a verdict.

5. **Directory placement.** The brief named `experiments/exp_011_small_overlap/`.
   The repository's convention, and the register's own path for every other
   experiment, is `_STAGE2_JSPACE/experiments/`, so the work lives at
   `_STAGE2_JSPACE/experiments/exp_011_small_overlap/`, which is what the
   register row's relative path already reads as.

6. **The 21.8-megabyte per-layer state cache is not committed.**
   `output/states.npz` holds the 637 states at twelve layers. It is
   deterministic and regenerates in about three minutes by running
   `python3 build_states.py`, so it is left out of version control and named in
   the directory's own ignore file. Every gate result and every provenance fact
   from that stage is committed in `output/states_meta.json`.

7. **The materiality floor for H16a was replaced by a labelled substitute.**
   Section 7.3 said a gap smaller than the spread of a state's control shares
   across the six control runs would be marked as inside the control spread.
   Because the two control types differ by a factor of about 25, that pooled
   spread is 0.178 to 0.182 at every band layer and would mark every gap in the
   table as immaterial, which is uninformative rather than conservative. The
   substitute used, and labelled post hoc where it appears, is the spread across
   the three seeds within one control type, which is 0.0009 to 0.0048 at the
   band layers. No verdict depends on it: the H16a rule counted signs, not
   sizes.

8. **The left panel of the figure uses a logarithmic vertical scale.** The
   specification asked for figures without specifying scales. A linear scale is
   unreadable here because the random-dictionary control sits about 25 times
   above every other series. The underlying numbers are unchanged and are in the
   tables and the committed comma-separated file.

No reduction was taken under the specification's stopping rule: the run completed
every planned arm at every layer, in 3,472 seconds of decomposition (0.96 hours)
at a peak memory of 1.58 gigabytes against a 3-gigabyte ceiling, plus about three
minutes of state building and about one minute of scoring and readouts.

---

## What this cannot settle

Stated in the specification before the run, repeated here because a reader should
meet the limits beside the numbers rather than after them.

The lens is a third-party fit that this project did not make and has not
validated on its own prompts. Its fit stopped early, at 277 prompts of a
requested 1000. This repository's own Medium-scale lens work
(`RESULTS_JLENS_MEDIUM.md`) records that a lens can pass a readability gate and
still show no coherent band, and no coherent band is visible here either. A low
share is therefore ambiguous between "the state holds nothing verbalizable" and
"this instrument cannot see what it holds". Nothing above rules latent content
out, and the H16b verdict in particular says the settled state holds less of what
this lens can express, which is not the same claim as "the settled state holds
less".

The share is a geometric quantity about a cone of at most 25 directions. It is
not a measure of meaning, and a state can be rich in structure the lens has no
direction for. That is established by the construction rather than inferred: the
lens's directions are built from the vocabulary, so anything the model holds that
is not aligned with saying a word is outside the measure by design.

The final LayerNorm's learned per-coordinate gain is not folded into the lens
directions. Following the definition this experiment was given, and the one the
primer states the paper uses, the lens directions are the rows of the unembedding
matrix multiplied by the layer's Jacobian, with no gain term. Folding the gain in
would tilt every direction slightly. This is a recorded limitation, not a repair.
It applies identically to every family and every control, so it cannot manufacture
a difference between them, and that last clause is inferred from the construction
rather than measured.

The 125 language states are one seed of one sweep, read at iteration 100 of the
original sweep rather than at the later convergence-gated lock-in. The `Divine`
comparison in H16a is a comparison of single vectors, not populations, and
carries no significance test at all: the numbers there are point estimates with a
seed-spread yardstick, nothing more. The clamping result rests on one attractor
and one cycle, each on a single trajectory, with no repeats.

---

## What remains

1. **The Medium variant, EXP_011m, is the registered primary arbiter for
   workspace-content claims about Medium terminal states** (register erratum
   (e)). Nothing here settles anything about GPT-2 Medium. What this run does
   supply for it is a working, cross-checked implementation, a measured cost, and
   the dictionary-shape diagnostic that Medium's lens should be checked for
   before its numbers are read.

2. **A validation of this lens on this project's own prompts has not been done.**
   The lens's readability was checked by Neuronpedia's fit record and by the
   published fit's own convergence criterion, not by this project. The Medium
   track ran a readability gate (`RESULTS_JLENS_MEDIUM.md`); the Small track has
   not. Until it is, a low share cannot be separated from an instrument that
   cannot see.

3. **The 25-direction limit was almost never the binding constraint here.** The
   search usually stopped earlier because the positive cone ran out. If a future
   run wants the sparsity limit to be the thing under test, it needs either a
   dictionary whose cone covers the space or a signed variant of the measure, and
   the signed variant is a different quantity that the paper does not define.

4. **Whether the same picture holds at lock-in rather than iteration 100.** The
   language states here are read at iteration 100 of the original sweep, which
   is where the original noise arm was also read, so that H6's comparison is
   like-for-like. The later convergence-gated classification relabels 11 of the
   125 prompts. Re-running the language arm at lock-in is a three-minute job on
   this machine if the lock-in tensors are ever committed; they are not committed
   today, only their labels are.

5. **Finding F16 in the lucier record needs a dated pointer note, and this
   session cannot write it.** F16 states that phase A of the `Divine` cycle is
   the more lens-expressible phase and phase B the less. On the full-vocabulary
   lens the assignment inverts, at eleven of twelve layers. The lucier repository
   is read-only from this session, so the correction lives here and nowhere else
   until someone with write access adds a pointer beside F16 in its
   `docs/FINDINGS.md`. Until that happens, a reader of the lucier record will
   take the superseded direction as current. Under the house convention of
   visible supersession the F16 text stays as written and gains a note; it is not
   edited away.

---

## What needs the operator's decision

1. **Which random dictionary counts as chance for J-space claims.** H16's
   registered wording names the norm-matched random dictionary as the chance
   level. This run shows that dictionary is not a like-for-like comparison: the
   real lens directions are clustered around one common direction (the average
   of the 50,257 unit directions has length 0.786 to 0.872 on a 0-to-1 scale
   across the band layers, against 0.0044 for the random dictionary, and 99.0 to
   99.6 percent of them lie on its positive side), while a random
   dictionary has no such clustering and therefore a much wider positive cone.
   Any state at all scores higher against the random dictionary for that reason
   alone. The rotated-lens control keeps the clustering exactly and moves only
   where it points, so it is the control that isolates the lens's content from
   the lens's shape. **Proposal for ruling:** the rotated-lens control becomes
   the registered chance level for every J-space share claim from here, and the
   norm-matched random dictionary is retained as a reported secondary. This
   would change the third condition in H16's scoring rule and would apply to the
   planned Medium variant EXP_011m and to H19b in EXP_018, so it is not a local
   change and should be ruled on rather than adopted quietly.

2. **Whether H6 stays live now that it has been scored.** H6's registered
   wording compares GPT-2 Small's five basins with the 18 basins of the original
   noise arm. That arm is the one lucier finding F4 records as mis-scaled, run at
   roughly one third of the language arm's injection strength and read before it
   had settled. H6 has now been scored on its wording, as the erratum requires,
   and the corrected comparison is H16. **Proposal for ruling:** H6's verdict
   row carries a pointer sentence saying the comparison it makes is against a
   superseded arm and that H16 is the live question, so no future reader treats
   an H6 verdict as evidence about language versus noise at matched strength.

3. **Whether to fold the final LayerNorm's learned gain into the lens
   directions in future J-space work.** The definition used here, and the one
   the paper's own companion text states, takes the lens directions to be the
   rows of the unembedding matrix multiplied by the layer's Jacobian. The
   model's actual word scores apply a learned per-coordinate gain before the
   unembedding. Folding it in would tilt every direction slightly and would
   change every share by an unknown amount. It is a one-line change and a
   half-hour run. **Proposal:** run it once as a sensitivity check inside
   EXP_011m rather than reopening EXP_011, and record whichever convention the
   Medium variant adopts as the registered one.
