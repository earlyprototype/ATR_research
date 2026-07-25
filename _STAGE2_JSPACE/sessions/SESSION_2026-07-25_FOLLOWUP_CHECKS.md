# Session record — 2026-07-25 — Checking our own work (EXP_010c-3b)

**Participants:** TC (direction: "all these are excellent and should be addressed") · Claude Code session (execution, drafting).
**Register:** reporting. Results in
`../experiments/exp_010c_windows/RESULTS_EXP010C.md` (observations only).
Interpretation lives here, fenced as thinking. Issue #21.

---

## The arc

**1. Where these came from.** After EXP_010c-3 reported its negative result
(no in-fill cell reproduces the seed cells' whole-word behaviour; 8→21 and
10→21 are isolated), the obvious next question was not "what else can we
measure" but "what would have to be true for this result to be wrong". Five
follow-ups came out of that, and they were deliberately ordered with the
self-undermining ones first — items 1 and 2 could each have overturned a claim
we had just published in PR #20.

**2. Pre-registration mattered here more than usual.** Item 1's reading was
fixed before the numbers existed, *including a failure mode for the test
itself* (if the word terminals scored as high as the funnel tokens, the
statistic would be declared uninformative rather than read in a convenient
direction). Item 3's decision — keep the rule, don't tighten it — was also
taken in the spec, before recomputing, precisely because the temptation was to
narrow a rule that had flagged an awkward cell.

**3. What came back.** Four of the five checks left the parent result standing.
The fifth (item 5) did not touch the parent result's conclusion but did surface
something none of the five were designed to look for.

## Interpretation (labelled as such — NOT findings)

```thinking
Item 1 is the cleanest result of the day and it went the way that makes the
parent result STRONGER, which is exactly why it needed pre-registering. Two
independent angles agree: funnel tokens win 0.01% of isotropic directions, and
— the post-hoc check that actually matters — they appear NOWHERE in the natural
decode at the layers where their arms extract. Natural layer-17 states decode
to ' China', ' quantum', ' there'; the loop at 8->17 decodes to ' GOP' on all
25 prompts. So the loop is doing something; it is not the readout talking.

I should be careful not to overclaim from this. "Not a decode artifact" is not
"meaningful content". The funnel tokens are still arbitrary-looking, and the
J-lens re-decode remains the arbiter. What item 1 buys is narrow: it removes
one specific deflationary explanation. That's all.

Item 2a is the one I did not expect and, methodologically, may outlast the rest
of this experiment. Every results section in this programme carries a "single
seed" caveat. That caveat is empty — the harness has no stochasticity to seed.
Nobody was being careless: it is boilerplate that migrated from experiments
where it meant something. But it created a false sense of an untested axis, and
a registered control in #11 that would have "passed" by reproducing existing
files byte-for-byte. That is the most dangerous kind of control: one that
cannot fail. Flagged on #11 rather than silently executed.

Item 5 is where I nearly missed something. It was scoped as low-priority
housekeeping — "we don't know the actual settle time". The settle time answer
is boring (~80 iterations, so the registered 120 was an upper bound as
suspected). But the comparison it enabled is not boring at all: change ONLY the
stopping rule, and I7 and X1017 return the identical terminal on 10/10 prompts
while I9 returns a DIFFERENT terminal on 5/5. The gate says "converged" in both
cases. So "converged" means "moving slowly at the moment we looked", not
"reached a fixed point", and whether those coincide is cell-dependent.

The saving grace for our headline: I9's lexical CLASS is non-whole-word under
both stopping rules ('iren'/"'d"/' would' at 80; 'oooooooo'/'…' at 120), so
the H12 refutation rests on the stable property, not the unstable one. That is
luck as much as design — the flag rule happened to be defined on class rather
than identity. If we had built the map on token identity, item 5 would have
undermined it.

The uncomfortable generalisation, which I am NOT asserting as a finding: the
programme's central objects are "terminals", and at least one cell's terminal
identity is an artifact of when we stopped looking. Nothing says I9 is the only
such cell — it is the only one of three tested that behaves this way. A proper
version of this check across the flagged cells (do A4's ' until'/' forever'/
' since' survive a different stopping rule?) is now a much more interesting
control than several of the ones already registered. It should be proposed
before the J-lens phase spends effort re-decoding terminals that may not be
stable objects.

Item 3's map correction is small but is the kind of thing that compounds. E23
(10->23) flags under the same mechanical rule and was simply not scored, because
the 2026-07-24 map covered the axes the in-fill was about rather than every
cell. Applying the rule uniformly found it in seconds. The lesson is not "we
were sloppy" — it's that hand-assembled maps drift from their own rules, and
the fix is to compute the map from the rule in one place (analyze_map.py now
does).

Item 4 is the least surprising and still worth having: at injection 8, nothing
above j=21 keeps the character, so j=21 is a sharp peak there too. But the
ladders are NOT symmetric above 21 — injection 10's j=23 flags, injection 8's
does not. I have no story for that and am not inventing one.
```

## What changed in the record

- No parent claim was withdrawn. Two were strengthened (funnels not a readout
  artifact; islands survive a disjoint subset), one list was corrected (E23
  added to the flagged cells), one caveat was rewritten (single seed → single
  prompt subset), and one new caveat was added (terminal identity is
  stopping-rule dependent at slow-drift cells).
- #11 flagged: its seed axis cannot vary anything; its subset axis stands.

## Proposed next (not run here)

A stopping-rule stability check across the flagged cells — do A4/O8's
whole-word terminals survive `check_start` variation the way their class does at
I9? — should precede EXP_013m. Re-decoding a terminal through a better
instrument assumes the terminal is a stable object; item 5 shows that assumption
is cell-dependent and currently untested at the cells that matter most.

## House lesson

The two checks designed to undermine our own result failed to undermine it, and
the check filed as low-priority housekeeping produced the most consequential
observation. Ordering by "what could prove us wrong" was right; assuming that
the *listed* risks are the *only* risks was not.
