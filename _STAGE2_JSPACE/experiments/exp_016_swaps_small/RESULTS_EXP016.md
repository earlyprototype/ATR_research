# RESULTS: EXP_016, swapping lens coordinates in base GPT-2 Small

**Specification of record:** `_STAGE2_JSPACE/EXP_016_SPEC.md`, committed at
`47efd8d` before any swap was run. **Register:**
`_STAGE2_JSPACE/REGISTER.md`, erratum (f) of 2026-09-05, which allocates
EXP_016 and the hypotheses H17, H17a and H17b and permits the third-party
lens used here. **Tracker:** issue #79. **Branch:**
`claude/latent-context-small-llms-u2jdig-exp016`, based on commit `34cc368`.
**Date of run:** 2026-09-05.

---

## What happened

The instrument works on this model, and it works much better than the
project's prior Medium-scale lens work would have led anyone to expect.

Exchanging two lens coordinates inside base GPT-2 Small, a
124-million-parameter text-continuation model from 2019, changes what the
model says next, and it changes it to the intended word rather than to
noise. A lens coordinate here is one number saying how much of a particular
word's direction the model's internal working memory is carrying at a
particular point in the sentence. The exchange leaves everything else in
that working memory untouched.

The three registered verdicts, each scored on the exact wording in the
register:

- **H17, the report swap: SUPPORTED.** On the held-out half of the items, at
  the setting chosen on the other half with the source rule treated as part
  of that choice, as section 5.1 of the specification says it is (layers 7,
  8 and 9 together, strength 2, all positions, source rule `output`),
  exchanging the model's own favourite concept with a different concept from
  the same category put the intended word into the model's top five next
  words on **86 percent of trials, 18 of 21**, against **0 percent, 0 of 42,
  for the norm-matched random-direction control**. Pooled over both source
  rules as 84 items, which is how the first version of this record scored
  it, the figure is 64 percent, 27 of 42, against 0 of 84, because the
  items built by the `lens` rule succeed on only 9 of 21 (see the correction
  below). The registered rule asked for at least 50 percent against at most
  10 percent, and both readings clear it. Before any swap, the intended word
  was in the top five on 0 of 84 items by construction, because the
  battery's rule excludes it from the model's top ten. That was checked
  again on 2026-09-05 by a fresh clean forward pass over the 84 frames with
  no intervention: it reproduces the battery file's recorded clean ranks on
  83 of the 84 items and differs by one rank on the eighty-fourth, at rank
  12,546 of the model's 50,257 words, and it puts the intended word no
  higher than rank 11 on any item. The lens arm and the
  control are measured on the same items, and under the null hypothesis
  that the lens direction is no better than a random direction of the same
  lengths, the lens draw is exchangeable with the two control draws inside
  each item. Several frames share one source concept, so the items are not
  independent of one another; grouping the 21 held-out items of the
  `output` rule into the 12 groups that share a source concept and treating
  the lens as one draw per group, the exact test (see "Statistical
  distinguishability" below) gives a probability of 5 in 100,000, and
  grouping into the 17 groups that share both source and target concept
  gives 2 in 10,000,000. The earlier figures of 3 in 1,000,000,000 for the
  protocol gap and 1 in 10,000,000,000,000 for the pooled one are withdrawn
  as headlines: they counted every item as an independent draw (correction
  16).
- **H17a, flexible generalisation: SUPPORTED.** Exchanging one country for
  another inside the model redirected at least two of the three separate
  questions about that country on **87 percent of the held-out pairs, 13 of
  15**, against **0 percent, 0 of 30, for the control**; the held-out half is
  the number the specification judges on. Over both halves the figure is the
  same 87 percent, 26 of 30, against 0 of 60. Among the pairs where all three
  questions could show redirection, all three moved together on 4 of 6 in
  the primary set and on **14 of 18, which is 78 percent, in the
  five-continent extension set**, against 0 of 36 for the control; the
  extension set is reported beside the headline number, not inside it. One
  population caveat (deviation 14): the specification's gate admits a
  question when the unmodified model ranks the correct answer in its top
  three, while the register's words "items the clean model answers
  correctly" can be read as rank one only. On the rank-one questions alone,
  72 of the 124, over the 19 primary pairs that keep at least two of them,
  the held-out result is 7 of 9 pairs against 0 of 18 for the control. A
  second caveat (deviation 17): the battery's halves were assigned by
  source country, 27 and 23 pairs, where the specification says alternate
  items, which for this battery means alternate pairs, 25 and 25. Under the
  specification's rule the per-question selection picks layer 10 and the
  held-out result is 14 of 15 pairs against 0 of 30, and the pair-level
  selection picks layer 9 with 13 of 15 against 0 of 30, so the verdict
  does not depend on the split. A third caveat, found in review on
  2026-09-05 (correction 15): the pre-registered success criterion of
  section 5.2 counts a question as redirected when the target country's
  answer is among the model's five most likely next words after the swap,
  and it does not require that the answer was absent from that top five
  beforehand, which is the exclusion section 5.1 builds into H17. A clean
  forward pass over the battery prompts with no intervention, run on
  2026-09-05, finds 24 of the 124 scored questions, which is 19 percent,
  whose target answer was already in the top five: the Australia to China
  continent question is one of them, because the unmodified model already
  ranks " Asia" fourth on "Australia is a country on the continent of". Of
  the 30 lens successes on the held-out primary pairs at the tuned setting,
  4 are such baseline hits, one each in the pairs Japan to Poland, Japan to
  Spain, Japan to France and Portugal to Spain, and no pair collects two of
  them. Requiring the target answer to be in the top five after the swap
  and not before it, the held-out pair result is 12 of 15 rather than 13 of
  15, still against 0 of 30 for the control; requiring instead that the
  target answer outrank the source country's own answer after the swap
  gives 12 of 15, again against 0 of 30. The verdict on the registered
  wording is the same under all three readings, so the registered reading
  stays as the register row and the stricter numbers sit beside it
  (decision item 7). A fourth caveat (correction 16): the 15 held-out pairs
  are three pairs for each of five source countries, and the three pairs of
  one country share that country's lens direction, so they are not 15
  independent draws. Treating the source country as the unit, the exact
  test gives 0.0041, which is 1 in 243 and is the smallest probability that
  five groups of three draws each can produce; the item-level 6.3 times 10
  to the power minus 7 is withdrawn as the headline for H17a.
- **H17b, intermediate-step surgery: SUPPORTED on its registered wording,
  weakly.** Exchanging the concept in the middle of a two-step completion
  changed the model's actual answer to the predicted alternative on **4 of 16
  items, which is 25 percent**, against **0 of 48 control trials, which is 0
  percent**. The exact within-item test (the lens draw exchangeable with
  the three control draws inside each item, see "Statistical
  distinguishability" below) gives 0.25 on the held-out half (1 of 8
  against 0 of 8), and that is the only valid test of the selected
  setting: the tuning half's three flips were used to choose the layer,
  strength and position mode from 126 candidate settings and cannot also
  count as evidence for them. Over all 16 items the same test would give
  0.0039, one in 256, but that figure reuses the selection outcomes and is
  reported only to show what the selection consumed. Three
  earlier versions of this record gave 0.0029 (control draws counted as
  items), 0.051 (the arms tested as independent samples) and 0.0625 (a
  sign test against the union of the three control draws, which is not an
  exchangeable pair); all three are withdrawn (see the corrections
  below). The registered
  rule asked only that the flip happen more often than under the control,
  and it does, but on the held-out half, the only valid test of the
  selected setting and the number the specification judges on, the effect
  is not distinguishable from chance, and the battery is too small for
  this verdict to carry much weight on its own. One population caveat cuts
  the other way
  (deviation 14): the specification's gate admitted 9 items whose correct
  answer the unmodified model ranks second or third, and all four flips are
  among the 7 items it ranks first, which is the population the registered
  wording names if "answers correctly" is read strictly. On those 7 the
  result is 4 of 7 against 0 of 7 for control A, and 1 of 3 against 0 of
  3 on the held-out half, which is again the only valid test (exact
  within-item probability 0.25).

The most interesting single number is not any of those three. It is the
contrast inside H17b. Exchanging the concept everywhere from its first
mention onward flipped the answer on 4 of 16 items. Exchanging it only at
the final position, where the answer is about to be produced, at the same
layer and strength, flipped the answer on 0 of 16. (The first version of
this record put the second number at 1 of 16, which is that mode's rate at
its own best setting, layers 8 and 9, chosen over all items; that compared
different layers and used the held-out items in the choice, and it is
withdrawn.) One thing those two labels hide: in 14 of the 16 items
the swapped concept is the first word of the prompt, so "from its first
mention onward" patches every position except the prepended start token,
and the two position modes `from_mention` and `all_no_bos` are the same
operation on those 14 items (the table by position mode below shows
identical rows for them). The contrast is therefore between changing the
whole stated problem, including the clause that states the facts the model
reads from, and changing only the position where the answer is produced. It
is not yet a clean separation of a step in the model's working from the
report of it. That difference, 4 against 0 at one setting, points the way
the reading note hoped, and it is far too small a difference, on far too
few items, to be called established. Marked as suggestive.

## What the swap looks like from outside

These are real completions, produced by re-running the tuned setting and
recording the model's five most likely next words. The full set is in
`output/qualitative.json`. Nothing is edited.

**A report swap.** The prompt is "My favourite animal is". Untouched, the
model's five most likely continuations are " the", " a", " my", " an" and
" probably", so it has not yet committed to an animal. After exchanging the
lens coordinates of " cat", the animal the model ranks highest, with those
of " dolphin", an animal that was not in its top ten, its five most likely
continuations are " dolphin", " dolphins", " Dolphin", " in" and " marine".
The control, which is the same operation with two random directions of the
same lengths, leaves the model saying " the", " a", " my", " an" and
" actually".

**A country swap that carries three questions at once.** The prompt is "The
capital of Australia is the city of". Untouched, the model's five most likely
continuations are " Sydney", " Melbourne", " Brisbane", " Canberra" and
" Perth", all Australian. After exchanging " Australia" for " Canada" at one
layer, they are " Ottawa", " Toronto", " St", " Winnipeg" and " Montreal",
all Canadian. The same single exchange, applied to "Most people in Australia
speak", moves " Portuguese" and " Spanish" up beside " English" when the
target is Brazil, and applied to "Australia is a country on the continent
of" it moves " Africa", " Latin" and " South" ahead of " Australia" when the
target is Brazil. The control leaves "The capital of Australia is the city
of" answering " South", " the", " Cambodia", " Central" and " Ce", which is
what a meaningless disturbance of that size does: it degrades the answer
rather than redirecting it.

**A two-step swap that changes the reasoning.** The prompt is "Roses are red
and violets are blue. The flower with thorns is". The model must work out
that the flower with thorns is the rose and then recall that roses are red.
Untouched, its five most likely continuations are " red", " the", " a",
" called" and " green". After exchanging the rose concept for the violet
concept from the first mention onward, they are " blue", " violet", " green",
" yellow" and " purple". The answer moved from the rose's colour to the
violet's colour. The control leaves it saying " red", " a", " the", " green"
and " yellow".

**A two-step swap that does nothing.** The prompt is "Fish swim and birds
fly. The animal with feathers can". Untouched, the five most likely
continuations are " be", " fly", " also", " swim" and "'t". After exchanging
the bird concept for the fish concept they are " be", " fly", " also",
" swim" and "'t", unchanged to the eye. Twelve of the sixteen two-step items
behave like this one rather than like the rose.

## The numbers

Every rate below is a fraction of trials, with the count beside it. "Control
A" is the registered control, two random directions of the same lengths as
the two lens directions, put through the identical exchange. "Control B" is
the stricter control added here, the same random directions with the
resulting disturbance rescaled to the size of the disturbance the real
exchange would have made. Both figures, `output/fig_layers.png` and
`output/fig_heldout.png`, plot these tables. In the second figure the H17
panel plots the protocol number, the held-out rate of the source rule
chosen on the tuning half, which is 86 percent, 18 of 21, and puts the
pooled rate over both source rules, 64 percent, 27 of 42, beside it as a
hatched bar. Until 2026-09-05 that panel plotted only the pooled rate and
labelled it "the reported number", which was wrong under section 5.1 of
the specification (correction 17).


### H17: tuned setting = layers 7-8-9, strength 2.0, positions all   (35280 records)

| arm | tuning half | held-out half | both halves |
|---|---|---|---|
| lens swap | 71 percent (30 of 42) | 64 percent (27 of 42) | 68 percent (57 of 84) |
| control A (random directions) | 1 percent (1 of 84) | 0 percent (0 of 84) | 1 percent (1 of 168) |
| control B (size matched) | 1 percent (1 of 84) | 0 percent (0 of 84) | 1 percent (1 of 168) |

Best ten settings over all items:

| layers | strength | positions | lens | control A | control B |
|---|---|---|---|---|---|
| 7-8-9 | 2.0 | all | 68 percent (57 of 84) | 1 percent (1 of 168) | 1 percent (1 of 168) |
| 7-8-9 | 2.0 | last | 67 percent (56 of 84) | 0 percent (0 of 168) | 1 percent (1 of 168) |
| 6-7-8 | 2.0 | all | 60 percent (50 of 84) | 2 percent (3 of 168) | 1 percent (2 of 168) |
| 7 | 2.0 | all | 57 percent (48 of 84) | 1 percent (1 of 168) | 0 percent (0 of 168) |
| 8 | 2.0 | all | 57 percent (48 of 84) | 1 percent (1 of 168) | 0 percent (0 of 168) |
| 10 | 2.0 | last | 52 percent (44 of 84) | 0 percent (0 of 168) | 0 percent (0 of 168) |
| 9 | 2.0 | all | 52 percent (44 of 84) | 0 percent (0 of 168) | 0 percent (0 of 168) |
| 6-7-8 | 2.0 | last | 51 percent (43 of 84) | 0 percent (0 of 168) | 0 percent (0 of 168) |
| 10 | 2.0 | all | 49 percent (41 of 84) | 0 percent (0 of 168) | 1 percent (2 of 168) |
| 5 | 2.0 | all | 46 percent (39 of 84) | 1 percent (1 of 168) | 0 percent (0 of 168) |

Single layers at the tuned strength and positions:

| layer | lens | control A | control B |
|---|---|---|---|
| 3 | 31 percent (26 of 84) | 0 percent (0 of 168) | 0 percent (0 of 168) |
| 4 | 29 percent (24 of 84) | 0 percent (0 of 168) | 0 percent (0 of 168) |
| 5 | 46 percent (39 of 84) | 1 percent (1 of 168) | 0 percent (0 of 168) |
| 6 | 46 percent (39 of 84) | 1 percent (2 of 168) | 0 percent (0 of 168) |
| 7 | 57 percent (48 of 84) | 1 percent (1 of 168) | 0 percent (0 of 168) |
| 8 | 57 percent (48 of 84) | 1 percent (1 of 168) | 0 percent (0 of 168) |
| 9 | 52 percent (44 of 84) | 0 percent (0 of 168) | 0 percent (0 of 168) |
| 10 | 49 percent (41 of 84) | 0 percent (0 of 168) | 1 percent (2 of 168) |

By source rule, at the tuned setting:

| source rule | lens | control A | control B |
|---|---|---|---|
| lens | 45 percent (19 of 42) | 1 percent (1 of 84) | 1 percent (1 of 84) |
| output | 90 percent (38 of 42) | 0 percent (0 of 84) | 0 percent (0 of 84) |

Stricter score at the tuned setting, target becomes the single most likely next word:

- lens swap: 65 percent (55 of 84)
- control A (random directions): 0 percent (0 of 168)
- control B (size matched): 0 percent (0 of 168)

Target outranks the source word it replaced:

- lens swap: 71 percent (60 of 84)
- control A (random directions): 24 percent (41 of 168)
- control B (size matched): 17 percent (28 of 168)

Source rule as part of the tuned selection (section 5.1 of the specification): chosen setting layers 7-8-9, strength 2.0, positions all, rule output:

| half | lens swap | control A | control B |
|---|---|---|---|
| tuning | 95 percent (20 of 21) | 0 percent (0 of 42) | 0 percent (0 of 42) |
| heldout | 86 percent (18 of 21) | 0 percent (0 of 42) | 0 percent (0 of 42) |
| overall | 90 percent (38 of 42) | 0 percent (0 of 84) | 0 percent (0 of 84) |

Each source rule at the pooled tuned setting (layers 7-8-9, strength 2.0, positions all):

| rule | half | lens swap | control A | control B |
|---|---|---|---|---|
| lens | tuning | 48 percent (10 of 21) | 2 percent (1 of 42) | 2 percent (1 of 42) |
| lens | heldout | 43 percent (9 of 21) | 0 percent (0 of 42) | 0 percent (0 of 42) |
| lens | overall | 45 percent (19 of 42) | 1 percent (1 of 84) | 1 percent (1 of 84) |
| output | tuning | 95 percent (20 of 21) | 0 percent (0 of 42) | 0 percent (0 of 42) |
| output | heldout | 86 percent (18 of 21) | 0 percent (0 of 42) | 0 percent (0 of 42) |
| output | overall | 90 percent (38 of 42) | 0 percent (0 of 84) | 0 percent (0 of 84) |

Exact within-item tests against control A (the lens draw exchangeable with the control draws inside each item):

- pooled heldout: 27 lens successes over 27 informative items, probability 1.31e-13
- rule lens heldout: 9 lens successes over 9 informative items, probability 5.08e-05
- rule output heldout: 18 lens successes over 18 informative items, probability 2.58e-09

Only the held-out lines test the selected setting; the tuning half chose it, so tests that include tuning items are reported for completeness and carry no evidential weight.

Cluster-level exact tests against control A, which treat the scored units that share one source lens direction as a single draw rather than as independent ones. Resolution is the smallest probability the test can return with that many clusters and draws, so a value equal to its resolution means the lens beat every control draw in every cluster:

- pooled heldout by source: cluster is the source concept; 27 lens successes over 42 scored units in 20 clusters (12 of them informative), probability 1.88e-06, resolution 2.87e-10.
- pooled heldout by source and target: cluster is the source and target concepts; 27 lens successes over 42 scored units in 31 clusters (19 of them informative), probability 8.6e-10, resolution 1.62e-15.
- rule lens heldout by source: cluster is the source concept; 9 lens successes over 21 scored units in 9 clusters (4 of them informative), probability 0.0123, resolution 5.08e-05.
- rule lens heldout by source and target: cluster is the source and target concepts; 9 lens successes over 21 scored units in 16 clusters (7 of them informative), probability 0.000457, resolution 2.32e-08.
- rule output heldout by source: cluster is the source concept; 18 lens successes over 21 scored units in 12 clusters (9 of them informative), probability 5.08e-05, resolution 1.88e-06.
- rule output heldout by source and target: cluster is the source and target concepts; 18 lens successes over 21 scored units in 17 clusters (14 of them informative), probability 2.09e-07, resolution 7.74e-09.

### H17A: tuned setting = layers 6, strength 2.0, positions all   (18600 records)

| arm | tuning half | held-out half | both halves |
|---|---|---|---|
| lens swap | 93 percent (62 of 67) | 91 percent (52 of 57) | 92 percent (114 of 124) |
| control A (random directions) | 12 percent (16 of 134) | 16 percent (18 of 114) | 14 percent (34 of 248) |
| control B (size matched) | 18 percent (24 of 134) | 18 percent (21 of 114) | 18 percent (45 of 248) |

Best ten settings over all items:

| layers | strength | positions | lens | control A | control B |
|---|---|---|---|---|---|
| 6 | 2.0 | all | 92 percent (114 of 124) | 14 percent (34 of 248) | 18 percent (45 of 248) |
| 9 | 2.0 | all | 90 percent (111 of 124) | 17 percent (42 of 248) | 15 percent (37 of 248) |
| 10 | 2.0 | all | 86 percent (107 of 124) | 16 percent (40 of 248) | 17 percent (42 of 248) |
| 8 | 2.0 | all | 85 percent (106 of 124) | 15 percent (36 of 248) | 18 percent (44 of 248) |
| 7 | 2.0 | all | 83 percent (103 of 124) | 14 percent (35 of 248) | 18 percent (44 of 248) |
| 3 | 2.0 | all | 81 percent (100 of 124) | 13 percent (32 of 248) | 16 percent (40 of 248) |
| 3 | 1.0 | all | 80 percent (99 of 124) | 17 percent (41 of 248) | 18 percent (45 of 248) |
| 4 | 2.0 | all | 77 percent (96 of 124) | 14 percent (35 of 248) | 18 percent (45 of 248) |
| 4 | 1.0 | all | 76 percent (94 of 124) | 17 percent (43 of 248) | 19 percent (47 of 248) |
| 5 | 1.0 | all | 71 percent (88 of 124) | 18 percent (45 of 248) | 19 percent (47 of 248) |

Single layers at the tuned strength and positions:

| layer | lens | control A | control B |
|---|---|---|---|
| 3 | 81 percent (100 of 124) | 13 percent (32 of 248) | 16 percent (40 of 248) |
| 4 | 77 percent (96 of 124) | 14 percent (35 of 248) | 18 percent (45 of 248) |
| 5 | 71 percent (88 of 124) | 17 percent (41 of 248) | 17 percent (43 of 248) |
| 6 | 92 percent (114 of 124) | 14 percent (34 of 248) | 18 percent (45 of 248) |
| 7 | 83 percent (103 of 124) | 14 percent (35 of 248) | 18 percent (44 of 248) |
| 8 | 85 percent (106 of 124) | 15 percent (36 of 248) | 18 percent (44 of 248) |
| 9 | 90 percent (111 of 124) | 17 percent (42 of 248) | 15 percent (37 of 248) |
| 10 | 86 percent (107 of 124) | 16 percent (40 of 248) | 17 percent (42 of 248) |

By question, at the tuned setting (primary pairs only):

| question | lens | control A | control B |
|---|---|---|---|
| capital | 87 percent (26 of 30) | 0 percent (0 of 60) | 0 percent (0 of 60) |
| language | 97 percent (29 of 30) | 12 percent (7 of 60) | 12 percent (7 of 60) |
| continent | 83 percent (5 of 6) | 58 percent (7 of 12) | 75 percent (9 of 12) |

By question, extension set (five continents):

| question | lens | control A | control B |
|---|---|---|---|
| capital | 85 percent (17 of 20) | 0 percent (0 of 40) | 0 percent (0 of 40) |
| language | 100 percent (18 of 18) | 28 percent (10 of 36) | 33 percent (12 of 36) |
| continent | 95 percent (19 of 20) | 25 percent (10 of 40) | 42 percent (17 of 40) |

Pairs where at least 2 scoreable questions redirected, primary set, tuned setting:
- lens swap: 87 percent (26 of 30)
- control A (random directions): 0 percent (0 of 60)
- control B (size matched): 0 percent (0 of 60)

Pairs where at least 3 scoreable questions redirected, primary set, tuned setting:
- lens swap: 67 percent (4 of 6)
- control A (random directions): 0 percent (0 of 12)
- control B (size matched): 0 percent (0 of 12)

Pairs where at least 2 scoreable questions redirected, extension set, tuned setting:
- lens swap: 100 percent (20 of 20)
- control A (random directions): 0 percent (0 of 40)
- control B (size matched): 5 percent (2 of 40)

Pairs where at least 3 scoreable questions redirected, extension set, tuned setting:
- lens swap: 78 percent (14 of 18)
- control A (random directions): 0 percent (0 of 36)
- control B (size matched): 0 percent (0 of 36)

Selection by the registered pair-level outcome on the primary pairs (at least two of three functions redirected): chosen setting layers 9, strength 2.0, positions all, against layers 6 for the function-level selection the main analysis uses:

| selection metric | setting | half | lens swap | control A | control B |
|---|---|---|---|---|---|
| pair-level, primary pairs | layers 9, strength 2.0, all | tuning | 93 percent (14 of 15) | 0 percent (0 of 30) | 0 percent (0 of 30) |
| pair-level, primary pairs | layers 9, strength 2.0, all | heldout | 87 percent (13 of 15) | 0 percent (0 of 30) | 0 percent (0 of 30) |
| pair-level, primary pairs | layers 9, strength 2.0, all | overall | 90 percent (27 of 30) | 0 percent (0 of 60) | 0 percent (0 of 60) |
| function-level (main analysis) | layers 6, strength 2.0, all | tuning | 87 percent (13 of 15) | 0 percent (0 of 30) | 0 percent (0 of 30) |
| function-level (main analysis) | layers 6, strength 2.0, all | heldout | 87 percent (13 of 15) | 0 percent (0 of 30) | 0 percent (0 of 30) |
| function-level (main analysis) | layers 6, strength 2.0, all | overall | 87 percent (26 of 30) | 0 percent (0 of 60) | 0 percent (0 of 60) |

Extension set, held-out half, at the pair-level chosen setting: lens 88 percent (7 of 8), control A 12 percent (2 of 16), control B 0 percent (0 of 16).

Rank-1 sensitivity check (specification section 5.2): only the 72 of 124 scoreable questions whose correct answer the unmodified model ranks first, scored per pair (at least two such questions redirected) over the pairs that keep at least two of them (19 primary, 4 extension), at the tuned setting:

| pair set | half | lens swap | control A | control B |
|---|---|---|---|---|
| primary | tuning | 8 of 10 | 0 of 20 | 0 of 20 |
| primary | heldout | 7 of 9 | 0 of 18 | 0 of 18 |
| primary | overall | 15 of 19 | 0 of 38 | 0 of 38 |
| extension | tuning | 4 of 4 | 0 of 8 | 1 of 8 |
| extension | heldout | 0 of 0 | 0 of 0 | 0 of 0 |
| extension | overall | 4 of 4 | 0 of 8 | 1 of 8 |

Where the target answer already stood before any swap, at the tuned setting, lens arm only. A baseline hit is a question whose target answer was already among the unmodified model's five most likely next words, so that the pre-registered criterion of section 5.2 is met without the swap doing anything:

| pair set | half | questions scored | of those, target already in the clean top five | lens successes | of those, baseline hits |
|---|---|---|---|---|---|
| primary | tuning | 33 | 6 | 30 | 5 |
| primary | heldout | 33 | 4 | 30 | 4 |
| primary | overall | 66 | 10 | 60 | 9 |
| extension | tuning | 34 | 7 | 32 | 7 |
| extension | heldout | 24 | 7 | 22 | 7 |
| extension | overall | 58 | 14 | 54 | 14 |

The same count by question, over both halves: primary set, capital 0 of 30, continent 6 of 6, language 4 of 30; extension set, capital 0 of 20, continent 8 of 20, language 6 of 18. The capital question carries none of them. In the primary set every continent question is one, because all six primary pairs whose two continent answers differ pair a European country with Japan, and on the continent frame the unmodified model already ranks " Asia" fourth or fifth for those European countries and " Europe" third for Japan. The primary continent row of the table above, lens 5 of 6, therefore shows nothing under the stricter reading.

Pairs in the primary set carrying a baseline hit among their lens successes at the tuned setting: France->Ireland (1), France->Japan (1), Greece->Ireland (1), Greece->Japan (1), Japan->France (1), Japan->Poland (1), Japan->Spain (1), Portugal->Spain (1), Spain->Ireland (1).

Pairs in the extension set carrying a baseline hit among their lens successes at the tuned setting: Australia->China (1), Australia->France (1), Brazil->Australia (1), Brazil->Canada (1), Brazil->France (1), Canada->China (1), Canada->France (1), China->Australia (1), China->Canada (2), China->France (1), France->Australia (1), France->Canada (1), France->China (1).

The pair-level result under the pre-registered criterion and under two stricter ones, primary pairs, a pair counting as a success when at least two of its scoreable questions were redirected by the same swap:

| criterion | setting | tuning, lens | held-out, lens | both halves, lens | held-out, control A |
|---|---|---|---|---|---|
| target answer in the top five after the swap (pre-registered) | layers 6, strength 2.0, all | 87 percent (13 of 15) | 87 percent (13 of 15) | 87 percent (26 of 30) | 0 percent (0 of 30) |
| target answer in the top five after the swap (pre-registered) | layers 9, strength 2.0, all | 93 percent (14 of 15) | 87 percent (13 of 15) | 90 percent (27 of 30) | 0 percent (0 of 30) |
| target answer in the top five after the swap and not before it | layers 6, strength 2.0, all | 67 percent (10 of 15) | 80 percent (12 of 15) | 73 percent (22 of 30) | 0 percent (0 of 30) |
| target answer in the top five after the swap and not before it | layers 9, strength 2.0, all | 80 percent (12 of 15) | 80 percent (12 of 15) | 80 percent (24 of 30) | 0 percent (0 of 30) |
| target answer outranks the source country's own answer after the swap | layers 6, strength 2.0, all | 93 percent (14 of 15) | 80 percent (12 of 15) | 87 percent (26 of 30) | 0 percent (0 of 30) |
| target answer outranks the source country's own answer after the swap | layers 9, strength 2.0, all | 100 percent (15 of 15) | 87 percent (13 of 15) | 93 percent (28 of 30) | 0 percent (0 of 30) |

Per question rather than per pair, held-out half, at the tuned setting, under the same three criteria:

- target answer in the top five after the swap (pre-registered): lens 91 percent (52 of 57), control A 16 percent (18 of 114), control B 18 percent (21 of 114).
- target answer in the top five after the swap and not before it: lens 72 percent (41 of 57), control A 1 percent (1 of 114), control B 1 percent (1 of 114).
- target answer outranks the source country's own answer after the swap: lens 91 percent (52 of 57), control A 10 percent (11 of 114), control B 13 percent (15 of 114).

Exact within-item tests against control A (the lens draw exchangeable with the control draws inside each item):

- pairs primary heldout: 13 lens successes over 13 informative items, probability 6.27e-07
- pairs primary both: 26 lens successes over 26 informative items, probability 3.93e-13 (post-selection: reuses the tuning outcomes that chose the setting, not a valid test of it)
- pairs primary heldout rank1: 7 lens successes over 7 informative items, probability 0.000457
- pairs primary heldout layer9: 13 lens successes over 13 informative items, probability 6.27e-07

Only the held-out lines test the selected setting; the tuning half chose it, so tests that include tuning items are reported for completeness and carry no evidential weight.

Cluster-level exact tests against control A, which treat the scored units that share one source lens direction as a single draw rather than as independent ones. Resolution is the smallest probability the test can return with that many clusters and draws, so a value equal to its resolution means the lens beat every control draw in every cluster:

- pairs primary heldout: cluster is the source country; 13 lens successes over 15 scored units in 5 clusters (5 of them informative), probability 0.00412, resolution 0.00412.
- pairs primary both: cluster is the source country; 26 lens successes over 30 scored units in 10 clusters (10 of them informative), probability 1.69e-05, resolution 1.69e-05 (post-selection: reuses the tuning outcomes that chose the setting, not a valid test of it).
- pairs primary heldout rank1: cluster is the source country; 7 lens successes over 9 scored units in 3 clusters (3 of them informative), probability 0.037, resolution 0.037.
- pairs primary heldout layer9: cluster is the source country; 13 lens successes over 15 scored units in 5 clusters (5 of them informative), probability 0.00412, resolution 0.00412.

Under the specification's own split rule (alternate pairs in the committed order, 25 tuning and 25 held-out, against the country-wise 27 and 23 the battery was built with): function-level selection picks layers 10, strength 2.0, all (tuning 87 percent (54 of 62), held-out 85 percent (53 of 62) per question; held-out primary pairs 14 of 15 against 0 of 30); pair-level selection picks layers 9 (held-out primary pairs 13 of 15 against 0 of 30). At the committed cells, held-out primary pairs under that split: layer 6 14 of 15, layer 9 13 of 15, control A 0 of 30 at both.

### H17B: tuned setting = layers 7, strength 2.0, positions all_no_bos   (14112 records)

| arm | tuning half | held-out half | both halves |
|---|---|---|---|
| lens swap | 38 percent (3 of 8) | 12 percent (1 of 8) | 25 percent (4 of 16) |
| control A (random directions) | 0 percent (0 of 24) | 0 percent (0 of 24) | 0 percent (0 of 48) |
| control B (size matched) | 4 percent (1 of 24) | 0 percent (0 of 24) | 2 percent (1 of 48) |

Best ten settings over all items:

| layers | strength | positions | lens | control A | control B |
|---|---|---|---|---|---|
| 4 | 2.0 | all_no_bos | 25 percent (4 of 16) | 2 percent (1 of 48) | 4 percent (2 of 48) |
| 4 | 2.0 | from_mention | 25 percent (4 of 16) | 2 percent (1 of 48) | 4 percent (2 of 48) |
| 7 | 2.0 | all_no_bos | 25 percent (4 of 16) | 0 percent (0 of 48) | 2 percent (1 of 48) |
| 7 | 2.0 | from_mention | 25 percent (4 of 16) | 0 percent (0 of 48) | 2 percent (1 of 48) |
| 3 | 2.0 | all_no_bos | 19 percent (3 of 16) | 0 percent (0 of 48) | 4 percent (2 of 48) |
| 3 | 2.0 | from_mention | 19 percent (3 of 16) | 0 percent (0 of 48) | 4 percent (2 of 48) |
| 5 | 1.0 | all_no_bos | 19 percent (3 of 16) | 0 percent (0 of 48) | 2 percent (1 of 48) |
| 5 | 1.0 | from_mention | 19 percent (3 of 16) | 0 percent (0 of 48) | 2 percent (1 of 48) |
| 5 | 2.0 | all_no_bos | 19 percent (3 of 16) | 0 percent (0 of 48) | 6 percent (3 of 48) |
| 5 | 2.0 | from_mention | 19 percent (3 of 16) | 0 percent (0 of 48) | 6 percent (3 of 48) |

Single layers at the tuned strength and positions:

| layer | lens | control A | control B |
|---|---|---|---|
| 3 | 19 percent (3 of 16) | 0 percent (0 of 48) | 4 percent (2 of 48) |
| 4 | 25 percent (4 of 16) | 2 percent (1 of 48) | 4 percent (2 of 48) |
| 5 | 19 percent (3 of 16) | 0 percent (0 of 48) | 6 percent (3 of 48) |
| 6 | 19 percent (3 of 16) | 4 percent (2 of 48) | 6 percent (3 of 48) |
| 7 | 25 percent (4 of 16) | 0 percent (0 of 48) | 2 percent (1 of 48) |
| 8 | 12 percent (2 of 16) | 0 percent (0 of 48) | 4 percent (2 of 48) |
| 9 | 12 percent (2 of 16) | 0 percent (0 of 48) | 0 percent (0 of 48) |
| 10 | 12 percent (2 of 16) | 0 percent (0 of 48) | 2 percent (1 of 48) |

By position mode at the tuned layer set and strength (layers 7, strength 2.0), so that only the patched positions differ:

| positions | half | lens | control A | control B |
|---|---|---|---|---|
| all_no_bos | tuning | 38 percent (3 of 8) | 0 percent (0 of 24) | 4 percent (1 of 24) |
| all_no_bos | heldout | 12 percent (1 of 8) | 0 percent (0 of 24) | 0 percent (0 of 24) |
| all_no_bos | overall | 25 percent (4 of 16) | 0 percent (0 of 48) | 2 percent (1 of 48) |
| from_mention | tuning | 38 percent (3 of 8) | 0 percent (0 of 24) | 4 percent (1 of 24) |
| from_mention | heldout | 12 percent (1 of 8) | 0 percent (0 of 24) | 0 percent (0 of 24) |
| from_mention | overall | 25 percent (4 of 16) | 0 percent (0 of 48) | 2 percent (1 of 48) |
| answer_only | tuning | 0 percent (0 of 8) | 0 percent (0 of 24) | 0 percent (0 of 24) |
| answer_only | heldout | 0 percent (0 of 8) | 0 percent (0 of 24) | 0 percent (0 of 24) |
| answer_only | overall | 0 percent (0 of 16) | 0 percent (0 of 48) | 0 percent (0 of 48) |

By position mode, each mode's own best setting chosen on the tuning half and scored on the held-out half:

| positions | tuned layers | strength | tuning, lens | held-out, lens | both halves, lens | both halves, control A |
|---|---|---|---|---|---|---|
| all_no_bos | 7 | 2.0 | 38 percent (3 of 8) | 12 percent (1 of 8) | 25 percent (4 of 16) | 0 percent (0 of 48) |
| from_mention | 7 | 2.0 | 38 percent (3 of 8) | 12 percent (1 of 8) | 25 percent (4 of 16) | 0 percent (0 of 48) |
| answer_only | 8-9 | 2.0 | 12 percent (1 of 8) | 0 percent (0 of 8) | 6 percent (1 of 16) | 0 percent (0 of 48) |

Looser score at the tuned setting, alternative answer in the top five:

- lens swap: 56 percent (9 of 16)
- control A (random directions): 46 percent (22 of 48)
- control B (size matched): 42 percent (20 of 48)

Rank-1 sensitivity check (specification section 5.3): the 7 of 16 items whose correct answer the unmodified model ranks first, at the tuned setting, an item counting as a control success if any of its draws flipped it:

| half | lens swap | control A | control B |
|---|---|---|---|
| tuning | 3 of 4 | 0 of 4 | 1 of 4 |
| heldout | 1 of 3 | 0 of 3 | 0 of 3 |
| overall | 4 of 7 | 0 of 7 | 1 of 7 |

Exact within-item tests against control A (the lens draw exchangeable with the control draws inside each item):

- items all: 4 lens successes over 4 informative items, probability 0.00391 (post-selection: reuses the tuning outcomes that chose the setting, not a valid test of it)
- items heldout: 1 lens successes over 1 informative items, probability 0.25
- items rank1: 4 lens successes over 4 informative items, probability 0.00391 (post-selection: reuses the tuning outcomes that chose the setting, not a valid test of it)

Only the held-out lines test the selected setting; the tuning half chose it, so tests that include tuning items are reported for completeness and carry no evidential weight.

Cluster-level exact tests against control A, which treat the scored units that share one source lens direction as a single draw rather than as independent ones. Resolution is the smallest probability the test can return with that many clusters and draws, so a value equal to its resolution means the lens beat every control draw in every cluster:

- items all: cluster is the source concept; 4 lens successes over 16 scored units in 16 clusters (4 of them informative), probability 0.00391, resolution 2.33e-10 (post-selection: reuses the tuning outcomes that chose the setting, not a valid test of it).
- items heldout: cluster is the source concept; 1 lens success over 8 scored units in 8 clusters (1 of them informative), probability 0.25, resolution 1.53e-05.

### Numbers not in the tables above

**All three country questions redirected together, counted only over the
pairs where all three could move.** In the primary set only 6 of the 30
pairs have three questions whose answers differ, because nine of the ten
countries that passed the clean-accuracy gate are European and therefore
share a continent. Of those 6, all three questions moved together on 4, and
on 0 of 12 control trials. In the five-continent extension set, 18 of the 20
pairs have three scoreable questions, and all three moved together on 14 of
those 18, which is 78 percent, against 0 of 36 control trials. On the
held-out half alone the figures are 2 of 3 and 6 of 8 respectively.

**Statistical distinguishability.** The lens arm and control A are measured
on the same items, with one lens draw and two (H17, H17a) or three (H17b)
control draws per item. Under the null hypothesis that the lens direction
is no better than a random direction of the same lengths, the lens draw is
exchangeable with the control draws inside each item. The exact test
conditions on each item's total number of successes among its draws,
relabels at random which draw is the lens, and asks how often the lens
would collect at least the observed number of successes; the count of lens
successes is then a sum of independent coin flips, one per item, with
probability equal to that item's successes divided by its draws. Items
where every draw or no draw succeeded carry no information. On these
records every informative item at the settings that matter has exactly one
success, on the lens, so each item contributes a factor of one third (two
control draws) or one quarter (three). On the held-out half: H17 gives 2.6
times 10 to the power minus 9 at the protocol setting (18 items) and 1.3
times 10 to the power minus 13 pooled over both source rules (27); H17a,
scored per pair on the primary pairs, gives 6.3 times 10 to the power
minus 7 (13 pairs), and 4.6 times 10 to the power minus 4 on the rank-one
questions alone (7 pairs); H17b gives 0.25 (1 item). Those four figures
are the item-level reading only, and the first three are withdrawn as
headlines by correction 16, because the items are not independent of one
another; the paragraph below gives the grouped values that replace them.
Only the held-out
half tests a selected setting: the tuning half chose the setting from the
grid, so a test that includes tuning items reuses the outcomes that made
the choice and is not a valid test of it; the all-items figures in the
tables (H17b 0.0039, 4 items) are labelled post-selection and carry no
evidential weight. The summary files carry every test under
`exact_tests`. Three earlier versions of this paragraph
are withdrawn: the first counted every control draw as a separate item
(4.2 times 10 to the power minus 17, 1.6 times 10 to the power minus 22,
0.25 and 0.0029), the second tested the two arms as independent samples at
item level, and the third ran a sign test against the union of an item's
control draws, which is not an exchangeable pair because a union of draws
succeeds more often than one draw (deviation 13).

**Statistical distinguishability when the items share a direction.** The
test just described treats every item as an independent draw, and for two
of the three batteries that is wrong, which review found on 2026-09-05
(correction 16). In H17a the 15 held-out primary pairs are three pairs for
each of five source countries, and the three pairs of one country are
swapped out of the same source lens direction, so the lens makes five
draws there and not fifteen. In H17 several frames share a source concept,
and some share both the source and the target concept, in which case the
items receive the identical swap and differ only in the wording of the
frame. The cluster-level test therefore groups the scored units that share
a source direction and treats the lens as one draw for the whole group,
exchangeable with each control seed's draw for that group; the groups are
independent of one another, so the null distribution of the total number
of lens successes is the product over groups, and the probability is
computed exactly from it. Because each group can contribute at best a
factor of one third (two control draws) or one quarter (three), the test
has a floor, called its resolution here: the smallest probability it can
return is one over the number of draws raised to the number of groups.

On the held-out half, with the source concept or source country as the
group: H17a's pair-level result gives 0.0041 over 5 groups, which is 1 in
243 and equal to its resolution, meaning the lens beat every control draw
in every group and the test cannot go lower with this many groups; on the
rank-one questions alone it gives 0.037 over 3 groups, again at its
resolution. H17 at the protocol setting gives 5.1 times 10 to the power
minus 5 over 12 groups by source concept and 2.1 times 10 to the power
minus 7 over 17 groups by source and target concept together, against
resolutions of 1.9 times 10 to the power minus 6 and 7.7 times 10 to the
power minus 9; pooled over both source rules the same two groupings give
1.9 times 10 to the power minus 6 and 8.6 times 10 to the power minus 10.
H17b is unaffected, because each of its 16 items has its own source
concept, so every group holds one item and the cluster-level test returns
exactly the item-level values, 0.25 on the held-out half and 0.0039 over
all items, the second of which stays labelled post-selection because the
tuning half chose the setting. The withdrawn item-level headlines are named in correction 16.
The summary files carry every cluster-level test under `cluster_tests`.

One limit of this test, stated because a reader would otherwise find it:
within a group the lens draws for the three pairs share a source
direction, while each control seed's three draws are independent random
directions, so the lens draw and the control draws are exchangeable at the
level of the group only under the null's own assumption that the lens
direction is nothing but another random direction. That is the null being
tested, so the test is valid under it, but the two arms are not matched in
their internal dependence, and the effect of that mismatch on the test's
power is not measured here. Inferred, not established.

**H17b under the looser score.** If success is counted as the alternative
answer merely entering the top five rather than becoming the model's actual
answer, the lens arm scores 56 percent and control A scores 46 percent, a
gap a Fisher exact test puts at a probability of 0.33, which is no evidence
at all. The reason is mechanical and worth stating: in these two-step items
the alternative answer word is written in the prompt itself, so it is
already a likely next word without any intervention. This is exactly why the
strict criterion was pre-registered for H17b and the top-five criterion for
the other two batteries.

**H17b by item.** The four items whose answer flipped at the tuned setting
were `bee-product`, `cow-product`, `rose-colour` and `train-track`. The
other twelve did not flip.

## What it means

**The lens directions are causal handles on this model, not just readable
ones.** That is established by H17 and H17a. On 84 report items the exchange
put the intended word into the top five 68 percent of the time and made it
the model's single most likely next word 65 percent of the time, while a
disturbance of the same size in a random direction did so 0 percent and 0
percent of the time. The instrument is doing something specific to the
direction, not merely shaking the model.

**One exchange redirects three different answers, and whether that is
downstream computation or output steering is not settled.** The redirection
is established by H17a and is the most substantial result here. One
exchange of a country's lens coordinates, applied at a single layer,
redirected the capital question 85 percent of the time, the language
question 100 percent of the time and the continent question 95 percent of
the time in the extension set, and moved at least two of the three together
on every one of the 20 pairs. Those three rates are not equally
informative, and the difference was found only in review (correction 15).
The capital question contains no case where the target answer was already
among the model's five most likely next words before any swap: 0 of its 20
extension questions and 0 of its 30 primary questions. The language
question contains 6 such cases of 18 in the extension set and 4 of 30 in
the primary set, and the continent question contains 8 of 20 in the
extension set and all 6 of 6 in the primary set, where every scoreable
continent pair joins a European country to Japan. The capital question,
scored against a control that succeeds on 0 of 40 extension trials and 0 of
60 primary trials, is therefore the clean evidence here, and it is the
strongest of the three. The country's name is not simply being copied
to the output, because the three answers are different words. But H17a
patched every position of the prompt, including the final position that
produces the answer, so the result is also compatible with the country's
direction steering each answer directly at that position rather than
feeding three separate lookups (limit seven). The first version of this
record called the downstream reading established; it is inferred at best,
and the position mode that would separate the two readings, a swap
confined to the country's mention, was not run for H17a.

**Whether it reaches an intermediate reasoning step is unsettled, and the
evidence leans yes.** H17b's flip rate of 4 in 16 items against 0 in 16
for the control (three draws each) is small and, on the held-out half,
the only valid test of the selected setting, not distinguishable from
chance (exact within-item probability 0.25); the
position contrast at the tuned layer and strength (4 of 16 when the
exchange starts at the concept's first mention, 0 of 16 when it happens
only at the answer position) is in the direction that separates reasoning
from reporting, with the caveat that on 14 of the 16 items the first
mention is the first word of the prompt. Both numbers are too small to
settle it. Inferred, not established.

**A finding nobody asked for: the lens and the model disagree about what the
model is thinking.** On 38 of the 42 report frames, which is 90 percent, the
category word the lens ranks highest at layer 8 is not the category word the
model itself would produce. On "My favourite sport is" the lens says
" cricket" and the model says " football". This matters practically, because
the swap works far better when the source concept is taken from the model's
own output than when it is taken from the lens: 90 percent success, 38 of
42, against 45 percent, 19 of 42, at the same tuned setting. Established as
a measurement. What it means is not established. One reading, marked as
speculation, is that the lens at mid layers reads a broad category
neighbourhood rather than the specific word the model is heading for, so its
top-ranked member is a plausible category member rather than the model's
choice.

**Small models do oversteer, but in the opposite direction from the
warning.** Neuronpedia's report that small models need fewer swapped layers
led the specification to sweep down to strength 0.5. In fact the strongest
setting swept, strength 2, won every battery, and for H17 a three-layer set
beat every single layer. At this scale the exchange needed more push, not
less. Established for this model and this lens, and it is a caution against
carrying a large-model tuning intuition down to a small model.


## What was done, in detail

### The instrument and its provenance

The lens is a pre-fitted file published by Neuronpedia,
`jlens_gpt2_small_neuronpedia.pt`, whose SHA-256 digest is
`d1800a1335ada089ef2e1ec0e4bd4d5bd61e6011eacc31f8618fdb3d10aae762`. During
the run that digest was a constant copied into the provenance files rather
than computed from the file loaded; it was recomputed from the file on
2026-09-05 and matches, and from this record's follow-up commit onward
`lib_exp016.py` computes it at load and refuses any other file. Its own
fit record says it was fitted on WikiText-103, a standard corpus of
Wikipedia text, using 277 prompts of at most 128 tokens each, in bfloat16
arithmetic, on 2026-06-11. The reference code is `anthropics/jacobian-lens`
at pinned commit `581d398613e5602a5af361e1c34d3a92ea82ba8e`. TC permitted
this third-party lens as the project's instrument on 2026-09-05, recorded as
erratum (f) of the identifier register.

The lens holds one 768-by-768 matrix for each of the layers 0 to 10. A
"layer l" here means the model's working memory as it stands after
transformer block l has finished, which is the transformer_lens hook point
`blocks.l.hook_resid_post`. GPT-2 Small has 12 such blocks, numbered 0 to
11, and 768 numbers in its working memory at each word position.

The model is base GPT-2 Small loaded without any weight processing, so that
its internal numbers are identical to the plain Hugging Face model the lens
was fitted on. That identity was checked: the largest disagreement in the
working memory at layer 5 on a test sentence was 0.00006, on numbers whose
own size reaches 2917, and the largest disagreement in the final word scores
was 0.00004. Established by measurement.

Versions: Python 3.11.15, torch 2.14.0 with no graphics card, transformers
5.16.1, transformer_lens 3.8.1, numpy 2.4.6, matplotlib 3.11.1 (installed
during this run, for the figures only). Every script ran with a single
compute thread.

### The operation

For a source word and a target word, at each swapped layer, the two lens
directions were stacked into a 768-by-2 matrix. The model's working memory
at each patched position was decomposed into how much of each of those two
directions it carried, those two amounts were exchanged, and the difference
was added back, optionally multiplied by a strength setting. Everything in
the working memory at right angles to the plane of the two directions was
left untouched.

Four properties of the implementation were checked before any battery ran,
and are recorded in the run log. Swapping a word with itself changed the
model's output by at most 0.0000343 in log-probability, which is numerical
noise. The two amounts really were exchanged: at the final position of "My
favourite sport is" at layer 8 they were 1.594 and 0.362 before, and 0.362
and 1.594 after. The part of the change lying outside the plane of the two
directions was 0.00000126, against a change whose total size was 9.528. The
batched code path agreed with a single-item path to 0.0000353.

### The controls

Two controls ran at every single setting, in the same batched forward pass
as the lens arm.

Control A is the one the register names: two random directions, drawn from a
standard Gaussian and rescaled so their lengths match the two lens
directions, put through the identical swap. Two random seeds in H17 and
H17a and three in H17b (deviation 1).

Control B was added by this session and is reported everywhere beside
control A: the same random directions, but the resulting change to the
working memory is rescaled, position by position, to the length of the
change the real lens swap would have made on that same working memory.
Control A fixes the lengths of the directions; control B fixes the size of
the actual disturbance. Control B is the stricter test of the objection
"would any disturbance this large have done the same thing", because two
random directions in 768 dimensions sit almost at right angles to each other
and to the working memory, so the amounts they read off are small. How
much smaller control A's change actually is was not captured during the
run: the size column in the record files is empty (deviation 8). It was
measured after the fact at each battery's chosen setting, and the section
"Disturbance sizes, measured after the fact" reports what was found.

### Disturbance sizes, measured after the fact (2026-09-05)

The specification promised that the size of each arm's change to the
working memory would be recorded, and the run did not record it (deviation
8). The sizes were measured afterwards at each battery's chosen setting, for
every unit, by `measure_disturbance.py`, with fresh random draws for the
controls (deviation 7; one draw per pair, shared across its questions, as
the corrected seeding scheme prescribes), so they are typical sizes for each
arm rather than the exact sizes behind the tables. The size of a change is
the Euclidean norm of everything the arm added to the working memory,
summed over the patched positions and layers; "of the stream" divides it by
the norm of the untouched working memory over the same positions at the
first patched layer. Medians over units; the full distributions, per layer,
are in `output/disturbance_sizes.json`. Nothing here carries verdict weight.

| Battery, chosen setting | Units | Lens swap | Control A (random directions) | Control B (size matched) |
|---|---|---|---|---|
| H17: layers 7-8-9, strength 2, all positions | 84 | 1065 (0.34 of the stream) | 732 (0.23) | 404 (0.13) |
| H17a: layer 6, strength 2, all positions | 124 | 97 (0.031) | 303 (0.098) | 97 (0.031) |
| H17b: layer 7, strength 2, all positions but the first | 16 | 109 (0.26) | 50 (0.12) | 109 (0.26) |

Three things follow, all established by this measurement for this model
and this lens.

First, the belief that the registered control under-disturbs the model,
which motivated control B and is repeated in the specification and in the
first version of this record, is wrong as a general statement. At the H17a
setting control A's change is 3.1 times the lens swap's, and it still
redirected 0 of 30 held-out pairs; at the H17 setting it is
0.69 times the lens swap's and at the H17b setting 0.46 times. Where
the control is smaller it is smaller by half, not by an order of magnitude,
and where it is larger the verdict is strengthened rather than weakened.

Second, control B matches the lens swap exactly at the two single-layer
settings (the same medians to three figures, as designed) but reaches only
0.38 of the lens swap's total change at the three-layer H17 setting.
The reason is in the per-layer sizes. The lens arm's change grows from
139 at layer 7 to 358 at layer 8 and 1017 at layer 9: each swap
acts on a working memory the previous swap has already altered, and at
strength 2 the exchange overshoots, so the next layer finds a larger
imbalance to exchange back. Control B is matched, layer by layer, to the
change the lens swap would make on the control's own working memory, which
the earlier random change has barely moved, so it stays near the layer-7
size (139, 208, 252). At multi-layer settings, then, control B is
size-fair per layer and not in total. The H17 verdict is scored against
control A, as registered, and control A's total change is 0.69 of the
lens swap's at that setting, so the comparison the verdict rests on is
close to size-fair.

Third, the H17 headline setting is a large intervention: a change of about
a third of the working memory's own size at the patched positions (median
0.34, at most 1.47 of the stream), against three percent at the H17a
setting and a quarter at the H17b setting. The report swap needed the
compounded three-layer push to clear its registered threshold; the country
swap did not.

### The tuning and held-out split

Every battery was split in half before any swap ran, by taking items in a
fixed order and sending alternate items to a tuning half and a held-out
half. The combination of layer set, strength and position mode with the
highest lens success rate was chosen using the tuning half alone, with ties
broken by the size of the gap over control A, then a smaller strength, then
fewer layers, then a lower layer index. That one combination was then scored
on the held-out half, and it is the held-out number that each verdict rests
on. Both halves and the full grid are in the summary files.

### The batteries and what the clean-accuracy gates discarded

Base GPT-2 Small is a 124-million-parameter text-continuation model from
2019. Its factual knowledge is thin and it is extremely sensitive to
phrasing, so every battery was gated on what the unmodified model could
already do, and items it could not do are recorded as not testable at this
scale rather than as failed swaps.

How sensitive to phrasing: on 25 countries, the frame "The capital of France
is the city of" put the right city first for 12 of 25, while "The capital of
France is" put it first for 0 of 25 and nowhere in the top five for any of
them. Four candidate frames per function were piloted and the best was fixed
before any swap ran.


### The three batteries and what the gates discarded

**H17, the report swap.** 42 sentences that stop where an answer begins,
across four categories: 12 about a favourite sport, 10 about a favourite
fruit, 10 about a favourite colour, 10 about a favourite animal. For each,
the source concept is the category word the model is already carrying and
the target concept is a different word from the same category that is not
among the model's ten most likely next words, which is the rule the paper
uses. Because the target is excluded from the top ten by construction, its
chance of being in the top five before any swap is exactly 0 out of 84
items. That zero is the baseline every success rate below is measured
against.

The pilot found that the two natural readings of "the model's own top
concept" disagree, so both were run. Reading one, called `lens` below, takes
the category word the lens ranks highest at layer 8. Reading two, called
`output`, takes the category word the model itself ranks highest in its own
next-word prediction. They agreed on only 4 of the 42 frames, which is 10
percent. On "My favourite sport is" the lens at layer 8 puts " cricket" top
among sports while the model itself would say " football". That disagreement
is a finding in its own right and is discussed below.

**H17a, flexible generalisation.** Three questions per country, using the
frames that the pilot found the model could actually answer: "The capital of
{country} is the city of", "Most people in {country} speak", and "{country}
is a country on the continent of". Answers are scored on their first token,
so " North America" is scored on " North", which is what the reference
implementation's own data files do.

A country entered the primary set only if the correct answer was within the
model's top three next words for all three questions. That left 10 of 25
countries: France, Germany, Greece, Ireland, Italy, Japan, Poland, Portugal,
Spain and Sweden. The other 15 are recorded as not testable at this scale,
not as failed swaps. Requiring rank 1 rather than rank 3 would have left
only 4 countries, which is too few to measure anything. The specification promised that rank-one set as a sensitivity check, and the first version of this record omitted it; the tables now carry it for both batteries, and the bullets above quote it.

Nine of those ten countries are European, so their continent answer is the
same word and a swap between two of them cannot show the continent question
moving. A separate extension set was therefore defined before the run from
the more generous rank-5 gate: the alphabetically first gated country on
each of the five continents present, which are Australia, Brazil, Canada,
China and France, and all 20 ordered pairs among them. The extension set is
reported separately and is not part of the headline number.

**H17b, intermediate-step surgery.** 36 two-step completions were written
and piloted. Each states two facts and then asks a question whose answer
requires first working out which of the two things is meant and then
recalling its fact, for example "Spiders have eight legs and ants have six
legs. The animal that spins webs has", whose answer is " eight" and where
exchanging the spider concept for the ant concept should give " six".

An item was kept only if the correct answer was within the model's top three
next words and ranked above the alternative answer. The second condition
matters: without it an item where the model already preferred the swap's
target would count a change that never happened. 17 of the 36 passed. One of
those, `knife-use`, was dropped because its prompt says "Knives" while the
swapped word is "knife", so the position of the first mention could not be
found automatically. 16 items remain, and the other 20 are recorded as not
testable at this scale.

### Where the swap was applied, and how the setting was chosen

The swap was applied as a forward hook on the model's working memory at the
output of each swapped transformer block, at the token positions named by
the position mode. Three things were swept and treated as tuned parameters,
because Neuronpedia reports that small models oversteer and need fewer
swapped layers than large ones: which layers to swap at, how strongly, and
at which positions.

The layers swept were each single layer from 3 to 10, the adjacent pairs
(5,6), (6,7), (7,8) and (8,9), and the adjacent triples (6,7,8) and (7,8,9).
The strengths were 0.5, 1 and 2, where 1 is the plain exchange the paper
used for its report swap and 2 is what it used for flexible generalisation.
The positions were every position for all three batteries, plus the final
position alone for H17, plus, for H17b, everything from the swapped
concept's first mention onward and the final position alone.

The setting was chosen on one half of the items and scored on the other. No
number reported as a verdict was chosen on the data it is reported for.

## What this does not show

Nine limits, stated before a reader finds them. The eighth and ninth were
added on 2026-09-05 after review.

First, a successful swap shows that the lens direction for a word is a
handle the model responds to, not that the model "contains a concept" in any
richer sense. The lens direction is built from the model's own output column
for that word, carried backwards by the fitted matrix, so pushing on it is
closer to leaning on the output than to editing a belief. The three
batteries exist precisely to separate those two readings, and the ladder
from H17 to H17a to H17b is the separation.

Second, the layer set, the strength and the position mode were tuned. They
were tuned on one half of the items and scored on the other, which is the
honest form of tuning, but every headline number should be read as "the best
this instrument can do at this scale when allowed to choose its settings on
separate data", not as "what happens if you swap".

Third, base GPT-2 Small is a weak model and the gates discarded a great
deal. The countries battery kept 10 of 25 countries and the two-step battery
kept 16 of 36 items. Every rate here is a rate over survivors. Nothing here
says what would happen on the items the model cannot do unaided, and nothing
here transfers automatically to a larger model.

Fourth, the lens was fitted by a third party on 277 prompts of Wikipedia
text in reduced-precision arithmetic. A different fit might give different
directions. This experiment tests the published instrument as published,
which is what erratum (f) permits, and says nothing about whether a better
fit would do better or worse.

Fifth, and specific to H17b: with 16 items, a difference of four successes
against one is what separates the two position modes. That is a thin thread
to hang the report-versus-reasoning distinction on, and it is named as
suggestive everywhere it appears above.

Sixth, the H17 setting stacks three swaps at consecutive layers, and the
measurement after the fact shows that they compound: the change at layer 9
is about seven times the change at layer 7, and the total is about a third
of the working memory's own size. The registered control A is close to
size-matched in total at that setting, so the verdict stands as scored, but
the effect is not a small nudge in lens coordinates. It is a large push
whose size grows through the layer set, and the size-matched control B does
not match it in total at multi-layer settings.

Seventh, H17a ran only the position mode that patches every token,
including the final one at which the answer is produced. A swap that
redirects three answers under that mode has changed the working memory at
the very position the model reads to answer, so the design cannot tell a
substituted country feeding three downstream lookups from a
country-associated direction steering each answer directly. The
mention-only mode that would separate them was not in the H17a grid. The
redirection itself stands; the reading that it passes through downstream
computation is inferred, not established.

Eighth, H17a's pre-registered success criterion can be met without the swap
doing anything. Section 5.2 counts a question as redirected when the target
country's answer is among the model's five most likely next words after the
swap, and unlike section 5.1's rule for H17 it does not require the answer
to have been absent from that top five before the intervention. Measured on
2026-09-05 by a clean forward pass with no intervention, 24 of the 124
scored questions, which is 19 percent, already had their target answer in
the top five, and 4 of the 30 lens successes on the held-out primary pairs
at the tuned setting are questions of that kind. The stricter readings are
in the tables and change the held-out pair result from 13 of 15 to 12 of
15, against 0 of 30 for the control either way, so the verdict does not
turn on it, but the registered criterion is weaker than it looks and any
later battery should build the exclusion into the criterion as H17 does.

Ninth, the scored units within a battery are not independent of one
another. H17a's three pairs per source country share that country's lens
direction and H17's frames often share a source concept, so a test that
counts every unit as a separate draw overstates the evidence. The
cluster-level test reported above is the honest version, and it costs a
great deal of apparent significance: H17a's held-out pair result moves from
6.3 times 10 to the power minus 7 to 0.0041, and H17's protocol result from
2.6 times 10 to the power minus 9 to 5.1 times 10 to the power minus 5. The
gaps themselves, 13 of 15 pairs against 0 of 30 control trials and 18 of 21
items against 0 of 42, are unchanged; what changes is how much a
significance test is entitled to say about them from five or twelve
independent draws.

## Deviations from the specification, stated flat

1. **Two settings were dropped from two batteries for time, before those
   batteries started.** The specification budgeted 35 milliseconds per
   condition, measured on an idle machine. Under the load the four shared
   cores were actually carrying, the measured rate was 18.2 conditions per
   second rather than 28.6, so the full grid would have needed about two
   further hours on top of the time already spent. Following section 7 of
   the specification, which requires cuts to be made by dropping whole
   settings and never by dropping items, the position mode `all_no_bos` was
   dropped from H17 and H17a, and the number of control seeds was reduced
   from three to two for those two batteries. H17b ran the full
   pre-registered grid with three seeds. The dropped position mode differs
   from the retained one only in whether the single prepended
   beginning-of-text token is patched. Two seeds still give 168 control
   trials per setting for H17 and 248 for H17a, which resolves a rate to
   better than one percentage point. The decision was taken and written into
   the run log before either affected battery began, so no item was seen and
   then dropped.

2. **A second control was added, not substituted.** Control B, the
   size-matched random control described above, is not in the register's
   wording of H17. It is reported beside control A everywhere and the
   registered thresholds are scored against control A, exactly as
   registered. The addition strengthens the design rather than weakening it.

3. **Two readings of "the model's own top concept" were both run for H17.**
   The pilot found that the lens's top-ranked concept at layer 8 and the
   model's own output top concept disagree on 38 of 42 report frames, so
   both readings were entered as separate items and the choice between them
   was left to the pre-registered tuning-half selection. This was decided
   and written into the specification before any swap ran.

4. **One two-step item was dropped for a mechanical reason.** The item
   `knife-use` passed the clean-accuracy gate but its prompt says "Knives"
   while the swapped concept word is "knife", so the position of the
   concept's first mention could not be located automatically. It was
   dropped at battery-build time, before any swap ran, leaving 16 items.

5. **matplotlib was installed during the run**, at version 3.11.1, purely to
   draw the two figures. It touches no measurement.

6. **The pseudo-inverse ridge is 1e-6, not the 1e-8 the specification
   states.** `swap_engine.py` adds 1e-6 times the largest diagonal entry to
   the 2-by-2 Gram matrix before solving. For these well-conditioned
   matrices the difference is negligible: over every source and target pair
   in the three batteries at all eight swept layers, the pseudo-inverse
   computed with the two ridges differs by at most 1.6 parts in 100,000
   (measured on 2026-09-05), so no coefficient and no verdict changes, but
   the constant was revised without a note and is recorded here.

7. **The random-direction controls in the committed record files cannot be
   regenerated draw for draw.** The first run seeded each control draw from
   Python's built-in `hash()` of the item, function, layer and seed index,
   which is randomised per process, and the process's hash seed was not
   recorded. The control rates in the tables are therefore measurements of
   one draw per seed index that a re-run would replace with a different
   draw. From this record's follow-up commit onward `run_swaps.py` derives
   the seed from a fixed checksum of the item, layer and seed index and
   records the scheme in the provenance file, so later runs reproduce
   themselves exactly (deviation 12 explains why the function name is no
   longer part of the seed).
   Nothing in the verdicts turns on a particular draw: every control rate at
   the settings that matter is 0 or within one trial of 0, across two or
   three independent draws.

8. **The disturbance-size column the specification promised was empty in
   the committed record files.** Section 3 of the specification says the
   sizes of both arms' changes are recorded so that the reader can check
   whether control A was matched in effect; `run_swaps.py` wrote an empty
   `patch_norm` field on every row. The sizes were measured after the fact
   at each battery's chosen setting only, with fresh control draws, by
   `measure_disturbance.py` (output `output/disturbance_sizes.json`), and
   are reported in the section "Disturbance sizes, measured after the
   fact". They carry no verdict weight. `run_swaps.py` now records the size
   on every row.

9. **Ties in the tuning-half selection are broken by the order the position
   modes were run in.** The pre-registered rule breaks ties by the gap over
   control A, then smaller strength, then fewer layers, then the lower
   layer; it names no rule for a tie that survives all four. One such tie
   occurred: on the H17b tuning half, `from_mention` and `all_no_bos` at
   layer 7, strength 2, are identical (3 of 8 lens successes, against 0 of
   24 and 1 of 24 for the controls), which is expected since the two modes
   patch the same positions on 14 of the 16 items. The run resolved it by
   the order in which the records were written, `all_no_bos` first;
   `analyse.py` now states that order as an explicit final key, so the
   choice is the same and is no longer accidental.

10. **The H17 source rule was pooled instead of tuned.** Section 5.1 of the
    specification says the two readings of "the model's own top concept"
    are run as separate items "and the choice between them is part of the
    tuned selection". The first analysis pooled the 84 items and chose only
    the setting, and the first version of this record reported the pooled
    held-out figure, 64 percent, 27 of 42, as the headline. Choosing the
    rule on the tuning half as well, as the specification says, picks the
    `output` rule at the same setting (20 of 21 on the tuning half) and
    scores 86 percent, 18 of 21, on the held-out half. The record now leads
    with the protocol figure and keeps the pooled one beside it; the table
    "Source rule as part of the tuned selection" carries both. The verdict
    is the same either way.

11. **The H17a selection metric is ambiguous in the specification, and both
    readings are reported.** Section 4 chooses the setting by "the highest
    success rate" on the tuning half, and section 5.2 defines success for
    H17a per question (the target country's answer in the top five), which
    is what the analysis used and which picks layer 6. The registered rule
    is stated per pair (at least two of three questions redirected) on the
    primary pairs, and choosing by that metric picks layer 9 (14 of 15
    pairs on the tuning half against 13 of 15). The two routes give the
    same held-out result, 13 of 15 pairs against 0 of 30 for control A, so
    the verdict does not depend on the reading; the table "Selection by the
    registered pair-level outcome" carries both. The function-level route
    also let the extension pairs vote in the choice of setting although
    they are not part of the headline; the pair-level route does not.

12. **The H17a controls used a different random draw for each of a pair's
    three questions.** The lens arm applies one shared exchange to all
    three questions about a country pair, but the first run seeded the
    random directions with the question name as well, so control A's three
    questions each saw their own directions. The pair-level control outcome
    (at least two of three questions redirected by "one" random swap) is
    therefore not like for like with the lens arm's, and it cannot be
    regenerated (deviation 7). It is reported as measured, with this
    caveat, and `run_swaps.py` now shares one draw across a pair's
    questions.

13. **The significance tests took three attempts to reach an exchangeable
    null.** The first version of this record tested each lens rate against
    a control denominator that multiplied the item count by the number of
    seeds, which treats repeated draws on the same prompt as separate
    items; a second version used item-level Fisher tests, which still
    treated the lens arm and the control as independent samples although
    they are measured on the same items; a third used a paired sign test
    against the union of an item's control draws, which is not an
    exchangeable pair because a union of two or three draws succeeds more
    often than a single draw. Every probability is now the exact
    within-item test described under "Statistical distinguishability",
    whose null is the one the registered comparison actually makes: that
    the lens draw is one more random draw. H17 stays far below any
    threshold (2.6 times 10 to the power minus 9 at the protocol setting),
    H17a's pair-level result gives 6.3 times 10 to the power minus 7 (13
    pairs), and H17b's held-out probability is 0.25; its all-items figure,
    which earlier versions of this record treated as a test, reuses the
    tuning outcomes that selected the setting and is labelled
    post-selection. A fourth attempt was needed, and it is recorded as
    deviation 19 and correction 16: the two figures quoted in this
    paragraph for H17 and H17a count every item as an independent draw,
    which they are not, and the grouped values that replace them as
    headlines are 5.1 times 10 to the power minus 5 and 0.0041.

14. **The gates admitted answers the model ranks second or third, and the
    promised rank-one sensitivity check was missing.** The register's
    wording for H17a and H17b names items "the clean model answers
    correctly"; the specification's pre-registered gates accept a correct
    answer anywhere in the unmodified model's top three, and say so, with
    the rank-one set promised as a sensitivity check. The first version of
    this record did not report that check. It is now in the tables: for
    H17a, the 72 of 124 scoreable questions at rank one give 7 of 9
    held-out pairs against 0 of 18 (15 of 19 over both halves, against 0
    of 38); for H17b, the 7 of 16 rank-one items hold all four flips, 4 of
    7 against 0 of 7. Whether the registered wording means rank one is
    decision item 5.

15. **The rank-five sensitivity set for H17a was not built or run.** Section
    5.2 of the specification promises the 20-country rank-five gate as a
    second sensitivity check beside the rank-one one. `build_batteries.py`
    forms the primary pairs from the rank-three gate only and uses the
    rank-five gate solely to pick the five continent representatives of the
    extension set, which is a different design and not a substitute. No
    stopping-rule reduction was recorded for it. Running it would add 60
    ordered pairs by the same pairing rule, about 27,000 swap conditions,
    roughly 20 minutes of processor time at the H17a battery's measured
    rate; it is decision item 6.

16. **The full top-five predictions were not recorded per trial.** Section
    5.1 of the specification says the top five next words are recorded in
    full for qualitative reading; the record files keep only the top word,
    the two ranks and the membership flags, and `output/qualitative.json`
    holds a chosen subset re-run afterwards. The lens arm's top fives can be
    regenerated exactly, because that arm is deterministic (about a fifth
    of the run); the control arms' cannot, because their draws are not
    regenerable (deviation 7). `run_swaps.py` now writes the five token
    identifiers on every row.

17. **The H17a halves were assigned by source country, not by alternating
    pairs.** Section 4 of the specification splits every battery by
    alternating items, and for H17a an item is an ordered pair. The battery
    builder assigned each country's three primary pairs (four in the
    extension set) to one half, giving 27 tuning and 23 held-out pairs
    rather than 25 and 25. The effect is that no source country appears in
    both halves, which is a defensible choice, but it is not the registered
    one. The reading under the registered rule is in the tables: the
    per-question selection then picks layer 10 (54 of 62 questions on the
    tuning half) and scores 14 of 15 held-out primary pairs against 0 of 30
    for control A, and the pair-level selection picks layer 9 with 13 of
    15 against 0 of 30. The run was tuned on the split it was built with,
    and the record keeps that as its headline; the verdict is the same
    under both.

18. **Two stricter readings of H17a's success criterion were added beside
    the registered one, not substituted for it, and one measurement was
    added to support them.** Section 5.2 of the specification scores a
    question as redirected when the target country's answer is in the model's
    five most likely next words after the swap, and records the top-one form
    and the outranking form alongside. It does not say where the target
    answer stood before the swap, and that was not recorded during the run.
    A new script, `clean_ranks.py`, measures it for all three batteries with
    a clean forward pass and no intervention, writing
    `output/clean_target_ranks.json`; the analysis then reports the
    registered criterion, the criterion restricted to answers that were not
    already in the clean top five, and the outranking criterion side by side.
    The registered criterion remains the one the verdict is scored on. The
    same pass reproduces the clean ranks already committed in the battery
    files, exactly for H17a's 124 questions and H17b's 16 items and to within
    one rank on one of H17's 84 items, at rank 12,546 of the model's 50,257
    words, where two words are all but tied.

19. **The exact test is now also reported with the units grouped by shared
    source direction.** The specification names no significance test at all;
    it registers rate thresholds against control A. The within-item exact
    test was added by this record, and the cluster-level version is added
    beside it because H17a's three pairs per source country and H17's frames
    sharing a source concept are not independent draws. Both are reported;
    the registered thresholds are unaffected, since they compare rates.

## Errata found during the run

One error was found and corrected while the batteries were running, before
any result had been read. The position of the swapped concept's first
mention, which only the H17b position mode `from_mention` uses, was computed
one token too high. The offsets came from the plain Hugging Face tokenizer,
which does not prepend a beginning-of-text token, and one was then added to
account for that token, but the transformer_lens tokenizer used to read the
offsets does prepend it, so the correction was applied twice. Fourteen of
the sixteen items moved from position 2 to position 1, one from 5 to 4 and
one from 8 to 7. The first H17b run was stopped at roughly 9,000 of its
14,112 conditions, its partial records were discarded, the position finder
was rewritten to walk the model's own token strings, the battery file was
rebuilt with only that one field changed, and H17b was run again in full.
The numbers reported here are from the corrected run.


## Corrections after review (2026-09-05)

Each line says what the first version of this record said, that it was
wrong, and what is true instead. The numbers are in the sections above.

1. It said H17's held-out result was 64 percent, 27 of 42, with the two
   source rules pooled. That is the wrong headline under section 5.1 of the
   specification, which makes the source rule part of the tuned selection.
   The protocol result is 86 percent, 18 of 21, with the `output` rule; the
   pooled figure is kept beside it. The verdict is unchanged.
2. It said H17b's effect was distinguishable from control at a probability
   of 0.0029 over all items and 0.25 on the held-out half, "so the effect
   is real". Those probabilities counted the 48 control rows as 48 items;
   a second version's 0.051 tested the arms as independent samples; a
   third version's 0.0625 sign-tested the lens draw against the union of
   three control draws; a fourth said the exact within-item test put the
   effect inside the usual level over all items. That last figure (0.0039)
   reuses the tuning-half flips that selected the setting. The only valid
   test is the held-out half, 0.25, and the effect is not distinguishable
   from chance.
3. It said that swapping only at the answer position flipped 1 of 16 items,
   against 4 of 16 for swapping from the first mention. The 1 came from a
   different layer set chosen over all items. At the tuned layer and
   strength the answer-only mode flips 0 of 16.
4. It presented one selection route for H17a. Two readings of the
   specification's selection metric exist and pick different layers; both
   give 13 of 15 held-out pairs against 0 of 30, and both are now reported.
5. It said "Rule R3 forbids this session from editing the register from a
   branch". R3 says no such thing; the register was left alone because four
   experiment branches share one allocation commit.
6. It said control A ran with three random seeds and listed four limits.
   Control A ran with two seeds in H17 and H17a and three in H17b, and the
   limits are six.
7. It said the registered control can under-disturb the model. Measured at
   the chosen settings, it does not in general: see "Disturbance sizes,
   measured after the fact".
8. It reported no rank-one sensitivity check, although the specification
   promised one for H17a and H17b. Both are now in the tables and quoted
   in the verdict bullets; for H17b every flip lies in the rank-one set.
9. Its "at least three questions redirected" tables counted pairs with only
   two scoreable questions as failures (4 of 30 and 14 of 20). The pairs
   that cannot meet the rule are now left out of the denominator (4 of 6
   and 14 of 18), as the bullets always said.
10. It said the effect "passes through downstream computation, not only to
    the output" and called that established by H17a. H17a patched the
    answer position too, so the reading is inferred, not established
    (limit seven).
11. Its tests took three attempts to reach an exchangeable null: control
    draws counted as items, then independent samples at item level, then
    a sign test against the union of an item's control draws, then an
    exact test that reused the tuning-half outcomes. Every probability is
    now the exact within-item test on the held-out half, and H17b's is
    0.25.
12. It assigned the H17a halves by source country (27 and 23 pairs) and
    described the split as the specification's alternating-item rule. The
    registered rule gives 25 and 25; under it the selection picks layer 10
    (or layer 9 pair-level) and the held-out result is 14 of 15 (or 13 of
    15) against 0 of 30. Recorded as deviation 17.
13. It said the top five next words were recorded in full for every trial.
    They were not; only the top word, the ranks and the membership flags
    were, and the control arms' top fives cannot be regenerated (deviation
    16).
14. Its reproduction sequence omitted the step that writes the qualitative
    job file; `make_qual_jobs.py` and the argument to `qualitative.py` are
    now named.

The four corrections below come from a review of commit `ea9073d` on
2026-09-05. Each says what the record said, that it was incomplete or
wrong, and what is true instead.

15. The H17a headline sentence said, without qualification, that the swap
    "redirected at least two of the three separate questions about that
    country on 87 percent of the held-out pairs, 13 of 15, against 0
    percent, 0 of 30, for the control". The count is right and the verdict
    stands, but the sentence hid that the pre-registered criterion counts a
    question as redirected whenever the target answer is in the model's top
    five after the swap, including when it was already there before any
    swap. A clean forward pass with no intervention finds 24 of the 124
    scored questions in that position, 19 percent, and 4 of the 30 lens
    successes on the held-out primary pairs at the tuned setting are of that
    kind. Under the stricter reading that requires the target answer to be
    new to the top five the held-out result is 12 of 15, and under the
    reading that requires the target answer to outrank the source country's
    own answer it is also 12 of 15, both against 0 of 30 for the control.
    The register row keeps the registered reading, with the stricter numbers
    beside it, and whether it should is decision item 7.

16. It reported the exact within-item probabilities as though every scored
    unit were an independent draw. They are not: H17a's three pairs per
    source country share that country's lens direction, and H17's frames
    often share a source concept or share both source and target concept, in
    which case they receive the identical swap. Three headline numbers are
    withdrawn as headlines and kept only as the item-level reading: H17a's
    6.3 times 10 to the power minus 7 on the held-out primary pairs, which
    becomes 0.0041 with the source country as the unit; H17's 2.6 times 10
    to the power minus 9 at the protocol setting, which becomes 5.1 times 10
    to the power minus 5 grouped by source concept and 2.1 times 10 to the
    power minus 7 grouped by source and target concept; and H17's pooled 1.3
    times 10 to the power minus 13, which becomes 1.9 times 10 to the power
    minus 6 and 8.6 times 10 to the power minus 10 under the same two
    groupings. The plain-language forms of the first two, "3 in
    1,000,000,000" and "1 in 10,000,000,000,000", are withdrawn with them.
    H17a's rank-one reading moves from 4.6 times 10 to the power minus 4 to
    0.037. H17b is unchanged at 0.25 on the held-out half, because each of
    its 16 items has its own source concept. No rate and no verdict changes.

17. The held-out figure, `output/fig_heldout.png`, plotted H17's pooled
    rate over both source rules, 64 percent, 27 of 42, under an axis label
    reading "held-out half (the reported number)", although the record's
    protocol result since correction 1 is the tuning-selected source rule's
    86 percent, 18 of 21. The figure now plots the selected rule's rate as
    the labelled bar, keeps the pooled rate beside it as a hatched bar, and
    names the selected rule in the panel title.

18. The reproduction section said its sequence was everything a reader
    needs, while `lib_exp016.py` set Hugging Face's two offline switches on
    import, so a fresh checkout with an empty model cache would fail to load
    base GPT-2 Small rather than fetch it. Offline mode is now off unless
    the environment variable `EXP016_OFFLINE` is set to 1, and the
    reproduction section names the model acquisition step. Nothing measured
    changes: loading the model with and without the switch on this machine
    gives byte-identical weights, SHA-256 digest
    `0684f138b3472aa8c2dca86ac3fddfb29c08473d13b70052dfa87a9b7d88cfe1` over
    the ordered parameter tensors, and identical next-word scores on a test
    prompt.

## What remains

These are the things this experiment did not settle, in rough order of how
much they would add.

1. **A larger two-step battery.** H17b is the hypothesis that matters most
   and it rests on 16 items. Writing 60 or 80 two-step items and keeping
   whatever survives the clean-accuracy gate would either establish the
   effect or kill it, and it costs very little computation, roughly 15
   minutes on one processor core per 16 items at the grid used here.
2. **The same three batteries on the post-trained twin.** EXP_017 already
   holds identifiers for LaMini-GPT-124M, a model with the same 124 million
   parameters as base GPT-2 Small but trained afterwards to follow
   instructions. Running this battery there would say whether what is seen
   here is a property of the size or of the training, which is the question
   the reading note actually asks.
3. **Whether the swap survives generation.** Everything here is scored on
   the single next word. Whether the model then continues coherently about
   the substituted concept for a whole sentence is a different and harder
   question, and it is the one that would matter for any claim about a
   workspace rather than a handle.
4. **What the lens is reading when it disagrees with the model.** The lens's
   top concept at layer 8 and the model's own output concept disagreed on 38
   of 42 report frames, and the swap works twice as well from the model's
   reading as from the lens's. Tracking that disagreement across layers is
   cheap and would say something about what the fitted lens actually
   measures.
5. **A lens fitted on this project's own corpus.** Comparing swap rates
   under this third-party lens and a locally fitted one would separate the
   instrument from the model.

## What needs the operator's decision

1. **Whether the register rows this experiment proposes are accepted.** They
   are in `experiments/exp_016_swaps_small/REGISTER_VERDICTS.md`, in the
   register's own format. The register is not edited from this branch
   because four experiment branches share one allocation commit and would
   otherwise carry conflicting edits to the same table; the rows land in one
   sweep after the experiment PRs merge, and nothing is authoritative until
   they do.

2. **Whether H17b's verdict should stand as SUPPORTED.** It is supported on
   the registered wording, which asks only that the flip happen more often
   than under the control, but the only valid test of the selected setting
   is the held-out half, which gives a probability of 0.25 (1 of 8 against
   0 of 8), and
   and on the held-out half alone, which this experiment's own
   specification names as the number a hypothesis is judged on, it is 1 of
   8 against 0 of 8 and not distinguishable from chance (probability 0.5).
   A reasonable operator could rule either
   SUPPORTED with the weakness recorded, which is what is written here, or
   NOT SUPPORTED pending a larger battery. This is the one verdict in this
   record where the wording and the evidence pull in different directions,
   and it is flagged rather than quietly resolved.

3. **Whether the added control B should become standard.** Control B, which
   matches the size of the disturbance rather than the lengths of the
   directions, is not in the register's wording of H17. It was added on the
   expectation that the registered control under-disturbs the model. The
   measurement after the fact shows that expectation was wrong at the H17a
   setting, where control A's change is about three times the lens swap's,
   and only partly right at the other two settings, where it is half to two
   thirds; it also shows that control B matches the lens swap per
   layer but not in total at multi-layer settings, because the lens swaps
   compound. In outcome the two controls behaved almost identically, so
   nothing turns on it in this experiment. Later swap work should register
   whichever control TC prefers from the start, and if a size-fair control
   at multi-layer settings is wanted it should match the total change, not
   the per-layer change.

4. **Whether the tune-then-hold-out convention becomes this project's house
   rule for instrument experiments.** This is the first time the project has
   used it. It should either become the convention or be replaced by
   something TC prefers.

5. **Whether "answers correctly" in the H17a and H17b register rows means
   rank one.** The specification's gates admit a correct answer anywhere in
   the top three, and the verdicts above are scored on those gates. Read at
   rank one, H17a rests on 19 primary pairs (held-out 7 of 9 against 0 of
   18) and H17b on 7 items (4 of 7 against 0 of 7); both stay SUPPORTED on
   their wording, and H17b is stronger there than on the full battery. If
   TC rules that the register means rank one, the register rows should
   quote those numbers as the headline and the specification's gate as the
   sensitivity check, rather than the other way round.

6. **Whether to run the promised rank-five sensitivity set for H17a**
   (deviation 15): 60 ordered pairs from the 20 countries the unmodified
   model answers within its top five, about 20 minutes of processor time,
   no verdict weight, reported beside the rank-one set.

7. **Which reading of H17a's success criterion the register row should
   quote** (correction 15). The pre-registered criterion counts a question
   as redirected when the target country's answer is among the model's five
   most likely next words after the swap, and 24 of the 124 scored
   questions, 19 percent, already met it before any swap. The row as
   proposed keeps the registered criterion and gives 13 of 15 held-out
   pairs against 0 of 30 for the control. The stricter criterion, requiring
   the target answer to be new to the top five, gives 12 of 15 against 0 of
   30, and the criterion requiring the target answer to outrank the source
   country's own answer gives the same 12 of 15. All three support the
   hypothesis, so this is a question of which number the register should
   carry, not of the verdict. Nothing in the register is changed from this
   branch either way (rule R8).

8. **Whether the cluster-level exact test becomes the project's standard
   for batteries whose items share a direction** (correction 16). This
   record now reports both the item-level test, which treats every scored
   unit as an independent draw, and the cluster-level test, which treats the
   units sharing one source lens direction as a single draw. The two differ
   by four orders of magnitude for H17a, 6.3 times 10 to the power minus 7
   against 0.0041. The cluster-level test is the conservative one and this
   record leads with it. Whether later experiments must design their
   batteries so that clusters are large enough for the test to resolve, five
   groups of three draws can never return less than 0.0041, is a question
   for the operator rather than for a session.

## Reproducing this

Everything needed is in `experiments/exp_016_swaps_small/`, except the model
weights and the lens file, which are downloaded. Two acquisitions come
first. The model is base GPT-2 Small, the Hugging Face repository `gpt2`,
about 500 megabytes, which transformer_lens fetches into the local Hugging
Face cache the first time `lib_exp016.load_model()` runs, so the first run
needs network access; `python3 -c "import transformers;
transformers.AutoModelForCausalLM.from_pretrained('gpt2')"` fetches it
ahead of time. Setting the environment variable `EXP016_OFFLINE` to 1 pins
every script to the local cache instead, which is what a machine with the
cache already populated should do and what this run did in effect;
until 2026-09-05 the scripts forced that offline mode themselves, so a
fresh checkout failed rather than downloading (correction 18). The lens is
the second acquisition, described below.

The order is:
`pilot_clean.py` and `pilot_clean2.py` measure the unmodified model and
write the two pilot files; `build_batteries.py` turns those into the three
committed battery files by the selection rules in section 5 of the
specification; `check_engine.py` verifies the swap implementation;
`run_swaps.py h17`, `run_swaps.py h17a` and `run_swaps.py h17b` produce the
record files; `clean_ranks.py` measures, with no intervention, where each
battery's target answer stood before any swap and writes
`output/clean_target_ranks.json`, which the analysis needs for the stricter
readings of H17a (deviation 18); `analyse.py h17 h17a h17b` applies the
tuning-then-held-out selection and writes the summaries; `make_tables.py`
prints the markdown tables of the section "The numbers" straight from those
summaries, so no number in this record is typed by hand;
`make_qual_jobs.py` writes
`output/qual_jobs.json` and `qualitative.py output/qual_jobs.json` re-runs
those conditions to capture the actual words; `make_figures.py` draws the
two figures; `measure_disturbance.py` recomputes the disturbance sizes at the
chosen settings (deviation 8). The lens file is not committed: download the
`gpt2-small` lens from the Hugging Face repository `neuronpedia/jacobian-lens`
and place it at `_STAGE2_JSPACE/artifacts/jlens_gpt2_small_neuronpedia.pt`
in the checkout, or point the environment variable `EXP016_LENS_PATH` at it;
`lib_exp016.py` computes the file's SHA-256 at load, refuses any file whose
digest differs from the recorded one, and writes the computed digest into
the provenance files. `run_swaps.py` now derives its control seeds from a
fixed checksum (deviation 7) and records the disturbance size on every row,
so a re-run reproduces itself exactly, though its control rates will not
match the committed ones draw for draw. Each script sets a single compute
thread and needs no graphics card. The record files are one row per condition, so any reader can
recompute every rate in this document from them without rerunning the model.
