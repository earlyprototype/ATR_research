# EXP_010c-3 — In-fill Scan Around the Word-Forming Window Cells (pre-registered spec)

**Status:** PRE-REGISTERED — recorded and committed before any run.
**Created:** 2026-07-24
**Parents:** `EXP_010c_SPEC.md` (grid + protocol), `EXP_010c2_SPEC.md` (boundary
scan + decode-via-tail control). Task register: GitHub issue #6.
**Motivating observation (recorded in RESULTS_EXP010C.md, 010c-2 section, stated
flat):** across the 14 windows tested so far, whole-word prompt-dependent
terminals occurred only for injection at i ∈ {8, 10} (extraction 15–21):
8→21 (`halfway`/`simultaneously`), 10→21 (`until`/`forever`/`since`),
8→15 (`rant`; via-tail `endless`). This spec fills in the immediate
neighbourhood of those cells so the J-lens phase (RUNBOOK_JLENS_MEDIUM.md)
knows which (injection, extraction) cells to target.

**Register:** reporting register; observations-only in the results record
(session rule 2026-07-23). The H-verdicts below are mechanical checks of
pre-stated criteria, not conclusions.

---

## 1. Questions

- **Q-contiguity (injection axis):** the boundary scan sampled the injection
  axis at even i only (0, 4, 6, 8, 10, 12, 14, extraction fixed at 21).
  Whole-word arms appeared at i=8 and i=10 with mixed arms at i=6 and i=12.
  Is the i ∈ {8, 10} zone one contiguous region — i.e. does i=9 behave like
  its neighbours — or two separate islands? And do the edges sit where the
  even-i scan implies, at single-layer resolution (i=5, 7 on the left,
  i=11 on the right)?
- **Q-extraction (extraction axis):** every onset-sweep cell extracted at
  j=21; the only other extraction depths ever measured are j=15 (A5, 8→15)
  and j=22/23 (exit-edge sweep, i=10). The extraction axis below 21 is
  otherwise unmeasured. Within the two whole-word rows (i=8, i=10), does the
  terminal character persist across j ∈ {15, 17, 19, 21}, or does the
  extraction layer co-determine it?

## 2. Hypotheses (continuing the numbering)

- **H11 (zone contiguity):** the whole-word injection zone is contiguous:
  9→21 produces whole-word, prompt-dependent terminals (≥2 unique terminals,
  all unique terminals whole-word per §3's token-class rule), connecting the
  8→21 and 10→21 cells into a single region. Refuted if 9→21's unique
  terminals are majority non-whole-word (two-island reading).
- **H11a (edge localisation at one-layer resolution):** each edge of the
  zone completes its transition within one layer — concretely, each of I5,
  I7, I11 patterns *either* with the whole-word zone *or* with its measured
  mixed-class outer neighbour (O6-like at i≤7, O12-like at i=11), rather
  than showing a new intermediate character (e.g. a punctuation funnel of
  the O0/O4 class appearing between mixed and whole-word cells). The edge
  positions themselves are measurements, not predictions: whichever side
  each flank lands on is recorded and localises that edge.
- **H11b (extraction independence within the zone rows):** for i ∈ {8, 10},
  the whole-word class persists at every extraction depth j ∈ {15, 17, 19,
  21}: all five X arms produce arms whose unique terminals are all
  whole-word. Refuted at any (i, j) where the terminal set is majority
  punctuation/fragment (extraction depth co-determines the class).

## 3. Design

**Model, prompts, protocol — identical to the registered EXP_010c tiers:**
gpt2-medium (offline load via `--model-path`; same legacy-S3-mirror weights,
state dict re-verified: 316 tensors, wte [50257, 1024], 24 layers); the same
recorded 25-prompt subset (derivation re-run against the public Lucier repo
clone and verified byte-identical to the committed `prompt_subset.json`);
gated protocol (threshold 0.999, patience 3, check_every 10, check_start 100,
max_iter 1000, gate_lag 1); `torch.manual_seed(42)`; L0 natural-pass seeding
(parent spec §3); terminal mean+last vectors saved per (window, prompt);
post-hoc lag_scan on the final iterates. Environment for this session:
torch 2.13.0, transformer_lens 3.5.1, py3.11 (same pins as the registered
runs), CPU.

**Arms (9 new windows × 25 prompts = 225 runs):**

| Arm | Window | Sweep | Fills in |
|---|---|---|---|
| I9 | 9→21 | Injection in-fill | The critical untested point between the two whole-word cells (H11) |
| I11 | 11→21 | Injection in-fill | Right flank: zone vs O12-like mixed (right edge) |
| I7 | 7→21 | Injection in-fill | Left flank: zone vs O6-like mixed (left edge) |
| I5 | 5→21 | Injection in-fill | Left approach: between O4 (punctuation funnel) and O6 (mixed) |
| X1019 | 10→19 | Extraction column | i=10 row, two layers below the known whole-word cell |
| X1017 | 10→17 | Extraction column | i=10 row, mid column |
| X1015 | 10→15 | Extraction column | i=10 row at j=15 (completes the i∈{8,10} × j=15 pair with A5) |
| X819 | 8→19 | Extraction column | i=8 row between the two known whole-word cells (8→15, 8→21) |
| X817 | 8→17 | Extraction column | i=8 row, mid column |

Execution in the order listed: highest-information cells first (I9 before
the flanks; the i=10 column before the i=8 column, since A4 anchors it),
so a mid-run interruption preserves the most informative prefix. Per-arm
checkpointing as built into the runner; commit after each arm.

**Deviation recorded (issue #6 arithmetic):** the issue header says "10 new
windows × 25 prompts = 250 runs", but the windows it enumerates — 5→21,
7→21, 9→21, 11→21; 8→17, 8→19, 10→15, 10→17, 10→19 — number nine. The
enumerated set is taken as authoritative: 9 windows × 25 prompts = 225 runs.
(The tenth cell of the implied 2×5 extraction grid, 8→15, is A5 — already
measured in the registered full run.)

**Harness:** the 9 arms are added to the `ARMS` dict in `run_exp010c.py`
(single source of truth; the analyzer imports it) under a new tier
`infill` (25 prompts, max_iter 1000, check_start 100, arm order as above).
Artifacts: `output/results_infill.json`, `output/terminals_infill.pt`.

**Readout and characterisation:** Stage-1-identical `ln_final → W_U` decode
at the extract layer (comparability), with R1 confidence metrics; then
`analyze_terminals.py --tier infill --decode-via-tail --model-path <local>`
for tensor-basin clustering at the gate threshold and the via-tail control
(EXP_010c-2 §3) on every arm. Both readouts are recorded per arm; the
J-lens re-decode (EXP_013m) remains the registered arbiter for all
mid-stack terminal claims.

**Token-class rule (mechanical, for the observation tables and the §4
criteria):** classify each unique decoded terminal string by its
characters:

- *punctuation* — no alphanumeric characters (e.g. `','`, `'\n'`, `'."'`);
- *whole-word* — leading space + alphabetic, not all-caps (e.g. `' until'`,
  `' halfway'`);
- *abbreviation/symbol* — leading space + all-caps alphabetic (`' NHS'`,
  `' DES'`), or any token containing digits/currency/other symbols
  (`' £'`), or special tokens (`'<|endoftext|>'`);
- *fragment* — alphabetic with no leading space (a continuation piece:
  `'ing'`, `'or'`, `'bone'`).

An arm's class: *whole-word* if all unique terminals are whole-word;
*punctuation funnel* if a single unique terminal of class punctuation;
*mixed* otherwise. *Prompt-dependent* = ≥2 unique terminals at n=25. These
definitions reproduce the classes recorded in the 010c-2 tables (O8, A4 →
whole-word; O0, O4 → punctuation funnel; O6, O12, O14 → mixed).

## 4. Pre-registered outcome readings

Mechanical application; every outcome recorded; negatives are findings.

| Observation | Reading |
|---|---|
| I9 whole-word and prompt-dependent | **H11 supported.** The zone is one contiguous region ⊇ 8–10; the J-lens phase targets the block, not two cells. |
| I9 majority non-whole-word while the 8/10 rows stand as measured | **H11 refuted.** Two islands; 8→21 and 10→21 go to the J-lens phase as separate target cells. |
| Each of I5, I7, I11 lands in {zone-like, outer-neighbour-like} | **H11a supported**; the landing side localises each edge to a single layer boundary (recorded as the updated map). |
| Any flank shows a character matching neither neighbour (e.g. new punctuation funnel between mixed cells) | **H11a refuted** for that edge; the transition is not one-layer-sharp there; note for the seed/subset control before further in-fill. |
| All five X arms whole-word | **H11b supported.** Within rows 8 and 10, terminal class is injection-determined across j ∈ {15, 17, 19, 21}. |
| Any X arm majority punctuation/fragment | **H11b refuted at that (i, j)**; extraction depth co-determines the class; the affected j is excluded from the J-lens target column and recorded. |
| Systematic non-convergence in any arm (>5/25 runs hit max_iter) | Recorded as its own outcome; that cell's terminal table is reported with non-convergence flagged (the gate's failure to lock is a finding, not a missing value). |
| A0-style anomaly (any arm reproducing the full-stack `D`) | Recorded flat; would be the first `D` outside 0→23 and triggers a harness check before any further reading. |

**Not pre-registered as criteria (recorded as observations only):** via-tail
agreement rates, tensor-basin counts, margins, lock-in iterations, and any
apparent token-identity patterns across cells. Semantic-relatedness
statements remain deferred to the anisotropy-corrected permutation control
(RESULTS §Planned controls item 1); no curated token sublists.

## 5. Cost & hand-off

225 runs, gated; observed throughput on this hardware class ≈ 27–60 s/run →
**≈ 2.5–4 h CPU**. Commit and push after each arm (a container restart
mid-run loses nothing committed; resume via `--arms` with the recorded
merge procedure from the registered full run).

**Deliverable forward (issue #6 definition of done):** RESULTS_EXP010C.md
gains a dated observations-only section in the established table format,
plus an updated tested-windows map (all measured (i, j) cells with arm
class) handing the J-lens phase its target cells — which injection layers
and which extraction layers produced whole-word prompt-dependent terminals.
