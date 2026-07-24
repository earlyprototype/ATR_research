# EXP_010c-4 — Full Single-Step Window Census on GPT-2 Medium (pre-registered spec)

**Status:** PRE-REGISTERED — recorded and committed before any census run.
**Created:** 2026-07-24
**Parents:** `EXP_010c_SPEC.md`, `EXP_010c2_SPEC.md`, `EXP_010c3_SPEC.md`.
**Directive:** TC, 2026-07-24, during the EXP_010c-3 session: check **every**
input layer against every output layer; no further sampled grids; no
reduced-prompt shortcut tiers ("do not cut any more corners").

---

## 1. Question and rationale

Every window experiment so far has sampled the (inject i, extract j) plane:
6 windows (010c), 8 (010c-2), 9 (010c-3) — 23 cells of the 300 valid
windows (0 ≤ i ≤ j ≤ 23). Each grid carried an implicit smoothness
assumption: that cells between samples behave like their neighbours.
EXP_010c-3 falsified that assumption at single-layer scale in both axes:
9→21 (between the two whole-word cells 8→21 and 10→21) produces fragment
terminals, and 8→19 (between whole-word 8→15 and 8→21) produces a
single-fragment funnel. A sampled map of this landscape is therefore not an
approximation of the full map; it is potentially a different object
(aliasing). **EXP_010c-4 measures the complete map: all 277 not-yet-measured
windows at the full registered protocol.** No sampling assumptions remain on
the layer axes when it completes.

## 2. Registered deliverable and hypothesis

The primary registered deliverable is **the map itself**: a per-cell record
(terminals, token classes, lock-ins, margins, basin counts, via-tail
agreement) for every valid (i, j), assembled from this census plus the 23
already-measured cells. From it, the exhaustive inventory of cells with
whole-word prompt-dependent terminals — the J-lens phase target set —
replaces the sampled inventory.

One mechanical hypothesis makes the aliasing question falsifiable:

- **H12 (aliasing materiality):** the census contains at least one cell
  whose arm class (per the 010c-3 §3 token-class rule) differs from the
  arm class of **every** already-measured cell adjacent to it on the (i, j)
  lattice (4-neighbourhood: (i±1, j), (i, j±1)). Supported = at least one
  such cell exists among the 277 (the sampled grids missed qualitative
  structure); refuted = every census cell matches at least one measured
  neighbour's class (the old grids' interpolation happened to be sound).

Everything else — token identities, apparent regional patterns, any
relationship to the J-space band picture — is recorded as observation only,
per the standing register rule. Semantic-relatedness claims stay deferred
to the anisotropy-corrected permutation control.

## 3. Design

**Protocol — identical to every registered tier, no reduced arms:**
gpt2-medium offline (same verified weights); the same recorded 25-prompt
subset for **every** window; gated loop (threshold 0.999, patience 3,
check_every 10, check_start 100, max_iter 1000, gate_lag 1);
`torch.manual_seed(42)`; L0 natural-pass seeding (010c §3); terminal
mean+last vectors and lag_scan captured per run. Environment: torch 2.13.0,
transformer_lens 3.5.1, py3.11, CPU (same pins as all registered runs).

**Arms:** the 277 windows W{i}_{j} generated in `run_exp010c.py`'s CENSUS
block — every 0 ≤ i ≤ j ≤ 23 pair not measured by the registered
full/scan/infill tiers. The 23 measured cells are **not rerun**; their
registered artifacts enter the final map unchanged (the 010c-2 "do not
rerun" convention; determinism basis recorded in the harness commit:
350/350 regeneration match, 13/13 X819 rerun match, byte-identical smoke
reproduction on this container). Includes the j = i single-layer windows
and the full sensory/motor extremes never sampled before.

**Execution order (risk management only; every arm runs):** the 100
unmeasured cells in the known-structure neighbourhood (4 ≤ i ≤ 14, j ≥ 13)
first, ordered (i, j) ascending; then the remaining 177 row-major. A
container death mid-census preserves the most informative prefix.

**Tier:** `census` (sharded artifacts: `output/results_census/<arm>.json` +
`output/terminals_census/<arm>.pt`, one pair per arm, committed per arm).
Runs launch with `--resume` semantics available: any interruption resumes
exactly, skipping complete arms.

**Characterisation:** `analyze_terminals.py --tier census
--decode-via-tail` after completion (the analyzer reads the shard
directories), producing the same table columns as the prior tiers. Interim
characterisations may be run read-only at any checkpoint; only the
post-completion one is the registered record.

**Readout ladder unchanged:** direct decode at j is the Stage-1-comparable
instrument; via-tail is the control; the J-lens re-decode (EXP_013m)
remains the registered arbiter for mid-stack claims. The census does not
add readout claims; it completes the dynamics map those instruments will
arbitrate.

## 4. Cost, honestly stated

6,925 runs. Observed locking runs cost ~40 s (lock 120–150); a
non-convergent run costs up to ~7 min (1000 iterations, window-length
dependent). All 575 registered runs to date converged, but the unmeasured
territory (very short windows, j = i cells, sensory/motor extremes) has no
convergence precedent. Bounds: **≈ 3.2 days CPU if convergence patterns
hold; materially longer if short windows fail to lock.** Progress is
observable live (per-run log lines; per-arm commits), the estimate updates
continuously, and every completed arm is durable regardless of when the
run is interrupted. No reduced-prompt recon tier is used (directive above).

## 5. Ops (recorded so deviations are checkable)

- Spec committed before launch (this file). Harness changes for scale
  (exact resume, per-arm shards, census arm generation, crc32 toy-hash
  fix) committed before launch with equivalence evidence in the message.
- One background process; per-arm commit+push by watcher; live log in the
  session scratchpad, entering the repo only at commits.
- On process death: relaunch with `--resume` (exact continuation). On
  container death: re-setup environment per RESULTS §Model acquisition
  (S3 mirror + pinned stack), then `--resume`.
- Register discipline: RESULTS_EXP010C.md gains dated checkpoint
  observations (counts and classes only) at natural boundaries (e.g. the
  priority block completing), and the full observations-only section +
  final map when the census completes. Interpretation stays in session
  notes, fenced as thinking.

## 6. Amendment — 2026-07-24, pre-analysis (PR #10 review)

Recorded before any census analysis or H12 evaluation (census execution
~1 arm in; no characterisation run). The review found a hole in H12 as
written in §2: a census cell with **no** already-measured adjacent cell
(e.g. 0→0 — most cells outside the measured region) satisfies "differs
from every measured neighbour" **vacuously**, making H12 unfalsifiable as
stated. Correction, replacing the eligibility condition only (the
support/refutation logic is unchanged):

> **H12 (amended):** evaluated over the subset of census cells having at
> least one already-measured adjacent cell on the valid (i ≤ j) lattice
> (4-neighbourhood: (i±1, j), (i, j±1), restricted to valid cells).
> Supported = at least one eligible cell's arm class (010c-3 §3 rule)
> differs from the arm class of **every** already-measured neighbour.
> Refuted = every eligible cell matches at least one measured neighbour's
> class. Cells with no measured neighbour are excluded from the H12
> evaluation and reported in the map only.

The original §2 text stands above as written (append-don't-rewrite); this
amendment governs.
