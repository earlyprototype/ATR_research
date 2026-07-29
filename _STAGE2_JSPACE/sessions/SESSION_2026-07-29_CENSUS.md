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
