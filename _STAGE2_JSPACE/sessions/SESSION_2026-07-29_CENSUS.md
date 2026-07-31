# Session record — 2026-07-25/29 — The full census, and what the comb was hiding

**Participants:** TC (direction, the aliasing correction, the no-corners
directive) · Claude Code session (execution, drafting).
**Register:** reporting. Results live in
`../experiments/exp_010c_windows/RESULTS_EXP010C.md` (dated 2026-07-29);
this note records the reasoning trail and holds all interpretation.

---

## The arc

**1. The correction that caused it.** Mid-way through the EXP_010c-3
in-fill, TC pressed two points. First, that the even-step grids alias:
sampling injection at {4,6,8,10,12} cannot see single-layer structure, and
the in-fill had just proved the structure *is* single-layer (9→21 junk
between the two word cells). Second, when the reply proposed a cheap
5-prompt recon tier: *"do not cut ANY more corners, it just costs more in
the long run."* Both were right, and the second killed a plan that would
have re-introduced a different sampling assumption (fewer prompts) to
escape the first.

**2. What got registered.** EXP_010c-4: all 277 unmeasured windows at the
*full* registered protocol — same 25 prompts, same gate, same seed. Cost
stated flat in the spec (~3.2 days, longer if short windows fail to lock),
no shortcut tier. Harness hardened first, with equivalence evidence:
artifact-verified `--resume`, per-arm shards, census arm generation in the
single-source-of-truth ARMS dict, and a found-while-testing fix for
`hash()`-based toy token ids that were unreproducible across processes.

**3. The review that improved the hypothesis.** CodeRabbit caught that H12
as written was vacuously satisfiable — a cell with no measured neighbour
"differs from every measured neighbour" trivially. Fixed as a dated
pre-analysis amendment restricting evaluation to cells with ≥1 measured
neighbour. Two arithmetic errors in the 010c-3 record (a lock-count and a
cell-count transposition) were also caught and corrected. The mechanical
verdicts were unaffected, but the record is better for it.

**4. Four days of grinding.** 6,925 runs. The infrastructure fought back:
14 silent process deaths, 2 container restarts, and a filesystem rollback
that reverted the local worktree an hour behind origin. Every one was
absorbed with zero measured cells lost, because of a rule that predates
this session: commit and push each arm as it completes. The rollback was
recovered entirely from the remote — the clearest vindication of that rule
the programme has had.

**5. What the census says.** All 300 cells measured. `D` at exactly one.
21 whole-word prompt-dependent cells where the sampled grids saw 2. H12
supported at 15 of 50 eligible cells. Six systematic non-convergence cells,
all at i ≤ 1 — the first non-convergences in the programme's history.

## Interpretation (fenced as thinking — none of this is in the register)

> **The band picture is dead, and so is the two-cell picture that replaced
> it.** After 010c-3 the honest summary was "two resonant cells." That was
> the sampled data speaking again. Across all 300 cells the whole-word
> prompt-dependent signature appears 21 times, scattered from 5→23 to
> 21→21 — including single-layer loops (10→10, 13→13, 20→20, 21→21) that
> no band story predicts and no grid would have sampled. Whatever produces
> word-like prompt-dependent terminals is not a contiguous region of the
> stack. Each successive picture — band, then two islands — was an artifact
> of the resolution used to look, and I should hold the current one loosely
> too: the census removes the *layer-axis* sampling assumption, but seed,
> prompt subset, and threshold assumptions all remain.
>
> **The `D` result is now as sharp as it can be made.** One cell in 300.
> Not "the full stack and things like it" — 1→23, which omits only layer 0,
> gives `'name'`. Whatever `D` is, it is a property of the exact full-stack
> splice, not of deep windows generally. The hook-point control from the
> planned-controls list (resid_post at i−1 vs resid_pre at i) is now the
> obvious next ATR-side experiment, because the census has isolated the
> phenomenon to the one place where that control bites.
>
> **The non-convergences are the most interesting new object.** Six cells,
> all i ≤ 1, short windows: states that drift for 1000 iterations without
> locking, lag_scan decaying monotonically (not cycling). Every registered
> run before this census converged. A drifting state in the earliest layers
> is exactly what "no attractor here" looks like — and it sits next to the
> layer-0 raw-embedding slot that the hook-point control was already meant
> to probe. Two independent reasons to run that control next.
>
> **Token identities to be careful about.** `' until'` at 10→21, 13→21 and
> 15→21; `' Quebec'` at 15→18, 15→19, 16→18; `' rant'` across three cells
> of row 8. These look like structure — the same attractor reachable from
> several cuts. They are also exactly the kind of pattern the anisotropy-
> corrected permutation test killed once already in Stage 1. Until that
> control runs on this data, "column-wise token identity" is an eyeballed
> pattern and nothing more.
>
> **Prediction hostage, restated for the next phase:** if the whole-word
> prompt-dependent cells are picking out something real about the model's
> verbalizable content, EXP_011m should find their terminal tensors project
> onto J-space more than the funnels' do, and the 21 cells should not be a
> random subset of the plane under that projection. If they project no
> differently, the "resonant cells" reading dies and what remains is a map
> of decode-layer quirks.

## Operational lessons recorded

1. **Commit-and-push per arm is not bookkeeping, it is the backup.** It
   survived 14 process deaths, 2 container restarts, and a filesystem
   rollback across 4 days. The rollback recovery was `git fetch` +
   `merge --ff-only` and nothing else.
2. **Resume must verify artifacts, not just count records.** The hardened
   version (results JSON written last as the completion marker; resume
   checks per-record terminal keys in both artifacts) came from review and
   was exercised for real at both container restarts.
3. **Shard artifacts per arm at scale.** A rewritten monolith would have
   pushed ~55 MB into git per commit at census size.
4. **A pre-analysis amendment beats a post-hoc reinterpretation.** H12's
   vacuity was fixed in the spec, dated, before any census cell was
   analysed — so the verdict below it means something.
5. **Live logs are ephemeral; artifacts are the record.** The scratchpad
   run log rolled back with the filesystem while the committed shards did
   not. Run counts taken from logs were understated for one check-in; shard
   counts are the source of truth.

## Addendum — the characterisation, and a prior of mine that died (same day)

The census-tier characterisation (277 arms, 6,925 terminal decodes plus
the via-tail control) landed after the results section was written. It
corrected something I had put in the record hours earlier.

**The correction, stated plainly:** from the sampled tiers I recorded a
readout prior — "direct decode and via-tail agree at high rates only at
extraction 21, collapsing at j ∈ {15,17,19}". Over the full census that
does not hold. Agreement rises roughly monotonically with extraction
depth (≈0/25 at j ≤ 4 → 18.9/25 at j=22), and **j=21's census arms
average 6.6/25 while j=22 averages 18.9/25**. The old prior came from two
cells; the census has eleven at j=21. The results record and this note
carry the corrected version; the runbook's §3a prior must be read against
the census table, not the sampled one.

> **Interpretation (fenced — not in the register).**
>
> The prior died the same way the band picture and the two-cell picture
> died: it was a statement about the cells that happened to be sampled,
> stated as though it were about the plane. Three times in one programme.
> The general lesson I keep re-learning here is that *every* summary
> sentence carries an implicit "…among what we measured", and the honest
> move is to write that clause explicitly until the measurement is
> exhaustive.
>
> **The more interesting finding is a dissociation.** Agreement across the
> 21 whole-word prompt-dependent target cells spans the entire range —
> 25/25 at (5→23), (6→23), (20→20); 0/25 at (8→11), (8→16), (15→17),
> (21→21). So the "produces words that vary by prompt" class and the "two
> instruments agree" class are *not* the same set. That matters for the
> J-lens phase in a specific way: agreement cannot be used to validate the
> target cells, because the targets include both extremes. Had the two
> classes coincided, EXP_013m would have been close to a formality; they
> don't, so it has real discriminative work to do. This is the best
> argument yet for the three-readout ladder as an instrument rather than a
> ceremony.
>
> **Cells where the instruments disagree most sharply are the ones I would
> look at first:** (8→16) direct `dozen`/`darn` → `' just'` ×25 via tail;
> (10→10) direct `Tooth` → `' Imp'`; (12→15) direct `HuffPost` →
> `<|endoftext|>` ×24. A state whose own-layer decode is a content word
> and whose through-the-tail decode is a function word or EOT is exactly
> the "workspace-poised vs motor-committed" contrast the ladder was built
> to separate. Equally, it could be mid-stack decode noise. The J-lens
> decides; I am recording the shortlist, not the conclusion.
>
> **One thing that did survive:** the (15→19)/(16→18) `Quebec` cells read
> as `Quebec`/`Montreal` and `Quebec`/`Ottawa` through the tail — same
> semantic neighbourhood under both instruments. If any of the token-
> identity patterns survives the anisotropy control, I would bet on that
> one. Betting is not evidence; the control runs when it runs.
