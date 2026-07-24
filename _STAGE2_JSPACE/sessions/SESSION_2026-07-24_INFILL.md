# Session record — 2026-07-24 — The in-fill, the islands, and the end of sampled grids

**Participants:** TC (direction, correction) · Claude Code session (execution, drafting).
**Register:** reporting. Results live in
`../experiments/exp_010c_windows/RESULTS_EXP010C.md` (dated 2026-07-24
section); this note records the reasoning trail per the house rule.

---

## The arc

**1. Fresh container, verified ground.** The session began in a clean
environment: pinned stack rebuilt, gpt2-medium re-fetched from the legacy S3
mirror (byte-count and state-dict match), the public Lucier repo cloned as
the read-only prompt source (derived subset byte-identical to the committed
one), and the smoke reproduction gate passed with **byte-identical
artifacts** — cross-container determinism, stronger than the same-machine
repeatability previously on record.

**2. Issue #6 → spec first.** The in-fill spec (H11 contiguity, H11a edges,
H11b extraction independence) was written and committed before any run,
including the arithmetic deviation (issue header says 10 windows; it
enumerates 9). Sweep launched with per-arm commit checkpoints.

**3. The process death.** At 188/225 the runner died silently — no
traceback, no OOM, RAM mostly free; most plausibly an infra event (an MCP
disconnect landed in the same window). Cost: nothing. Per-arm checkpoints
held 7 arms; X819's 13 lost in-memory runs reran to line-identical values.
The recovery followed the recorded merge procedure from the full tier's
container restart. Lesson banked: for anything longer, resume must be a
harness capability, not an operator procedure (see 6).

**4. TC's aliasing correction.** Mid-run, TC pressed two points that
reshaped the programme. First: the even-step grids alias — sampling i at
{4,6,8,10,12} cannot see single-layer structure, and I9's junk between the
8 and 10 word-cells proved the structure IS single-layer. Second, after the
grid table landed: *"how can you justify not checking every input layer vs
output — it's the focus of the task"* and *"do not cut ANY more corners, it
just costs more in the long run."* The honest answer to the first was that
cost triage had justified sampling only while the smoothness assumption
held, and the assumption died today. The directive resolved the second:
no reduced-prompt recon tier — the full census at the registered protocol.

**5. EXP_010c-3 closed out.** 225/225 converged. Mechanical verdicts:
H11 refuted (islands, not a zone), H11a supported under the class rule
(with the coarseness caveat recorded), H11b not supported (refutation row
fires at (10,19); the `'d`, `Fas`, `Bhar` classification edges recorded
flat). The tested-windows map and the J-lens hand-forward went into the
results record: the prompt-dependent whole-word phenomenon lives at exactly
(8→21) and (10→21) among measured cells.

**6. EXP_010c-4 registered and launched.** Full census: all 277 unmeasured
(i ≤ j) windows × the same 25 prompts, same gate, same seed. Harness work
committed first with equivalence evidence: exact `--resume` (skip complete
arms; justified by the determinism record), per-arm artifact shards (a
rewritten monolith would push ~55 MB into git per commit at census scale),
census arm generation in the single-source-of-truth ARMS dict, and a
found-while-testing fix: harness-check toy token ids used Python's
process-randomized `hash()`, so toy-tier A4 terminals were never
reproducible across runs. crc32 now; two-process identity verified; real
smoke re-verified byte-identical post-refactor.

## Interpretation (fenced as thinking — none of this is in the register)

> The zone was an artifact of the sampling comb. What the in-fill leaves
> standing is two isolated (inject, extract) cells — (8→21), (10→21) —
> where the loop's terminals are words that vary by prompt, surrounded on
> every measured side by funnels and fragments. "Workspace band" language
> should be retired until the census reports; "two resonant cells" is the
> honest current shape.
>
> The extraction columns say the word-phenomenon is as much a property of
> *where you read* as *where you inject*. And the via-tail identities at
> the funnel cells are suggestive in a way the register rightly refuses to
> bless: X817's `GOP` state reads as `since` through the motor tail —
> `since` is one of A4's three terminals; X1017's `Bhar` reads as
> `Indian`. A state that decodes as junk at its own layer and as an
> A4-family word after the remaining layers process it is exactly what a
> "poised content, miscalibrated local readout" story predicts — and
> exactly what the J-lens exists to arbitrate (readout ladder, yesterday's
> session note). The anisotropy-corrected permutation control and EXP_013m
> get to decide; if they kill it, it dies.
>
> Prediction hostage for the census (stated so it can fail): if the
> two-cell picture is right and not another sampling artifact, the census
> should find (a) no whole-word prompt-dependent cell outside small
> neighbourhoods of (8→21)/(10→21), and (b) more single-terminal funnels
> whose via-tail decode lands in the small word set already seen. If the
> census instead finds word-cells scattered across the plane, the
> "resonant cells" picture joins the band picture in the discard pile.

## Operational lessons recorded

1. Per-arm checkpoint commits turned a silent process death into a
   non-event. Now formalised as `--resume`.
2. Live logs do not belong inside the repo (a growing tracked file trips
   the tree-clean stop-hook every turn and generates noise commits; live
   logs now write to the session scratchpad and enter the repo at commit
   points).
3. Two latent nondeterminisms found and fixed while hardening for the
   census: the toy-hash issue (above), and the general lesson that any
   value derived from `hash()` is a per-process coin flip.
4. The census runs as: one process + `--resume` on death; shard-per-arm
   artifacts committed by a watcher; hourly session check-ins as the
   fallback wake signal. Estimated 3.2+ days; every completed arm durable.
