# REGISTER — canonical identifier register for Stage 2

**Status:** AUTHORITATIVE. This file is the in-repo authority for every
hypothesis ID (`H<n>`) and experiment ID (`EXP_*`) used in Stage 2. Board
discussion #37 (Identifier registry) is this file's **mirror** for
branch-blind sessions; if the two disagree, this file wins and #37 is
corrected.

**The rule:** no new `H<n>` or `EXP_*` identifier may be used in any spec or
results file without a row added to this file **in the same commit**. Claim
the ID here first, then use it.

**Created:** 2026-07-31, resolving issue #52 (three-way hypothesis-ID
collision) and the second double-registration incident recorded in the
errata below. The formerly-contested dispositions recorded here (spec of
record for EXP_010c-3, the H11-series retirements, the census H12→H14
renumber) follow the 2026-07-26 identifier-registry ruling on discussion
#37 and were executed under TC's in-session direction of 2026-07-31
(delegation recorded in the session that produced PR #55).

**Convention:** superseded and retired bindings stay visible in this table,
marked, with their git hashes. Nothing is deleted.

---

## 1. Hypothesis register (H0–H15)

**Next free hypothesis ID: H16.** H1–H3 are unallocated gaps and stay
unfilled — every Stage 2 spec says "continuing the numbering", so
backfilling a low gap would break the reader's expectation that number
order tracks time order. Do not backfill; continue at H16.

| ID | Owning experiment | Statement (one line) | Verdict | Recorded at |
|---|---|---|---|---|
| H0 | Stage 1 | Stage 1 register — statement not recoverable from this repo; see the lucier repo | Stage 1 | `STAGE2_OUTLINE.md` §2 ("continuing Stage 1's H0–H4 numbering") |
| H1–H3 | — | **GAP — never allocated in either stage's record reachable from this repo.** Do not backfill. | — | — |
| H4 | Stage 1 | Stage 1 register — statement not recoverable from this repo; see the lucier repo | Stage 1 | `STAGE2_OUTLINE.md` §2 |
| H5 | EXP_010b | Layer-window loops produce qualitatively different landscapes within vs across the putative workspace band (coarse form: any sub-stack window differs from the full stack) | **SUPPORTED (coarse)** | `experiments/exp_010b_small/RESULTS_EXP010B.md` ("H5-coarse: SUPPORTED") |
| H6 | EXP_011 (planned) | GPT-2 Small's five basin tensors project significantly more onto the J-space than the 18 null-model basins | UNTESTED | `STAGE2_PLAN.md` §Hypotheses |
| H7 | EXP_012 (planned) | A coherent J-lens band structure exists at 124M–410M scale and its presence/shape predicts each model's ATR regime | UNTESTED as registered (note: EXP_012m found NO COHERENT BAND on Medium — `RESULTS_JLENS_MEDIUM.md`) | `STAGE2_PLAN.md` §Hypotheses |
| H8 | EXP_012-PYTHIA (absorbing EXP_010a) | Pythia-410m's fragmentation is depth-dependent: looping 0–11 vs native 0–23 changes convergence behaviour | **SUPPORTED** (loudness-covariance caveat recorded flat in the results) | `experiments/exp_010c_windows/RESULTS_EXP012_PYTHIA.md` |
| H9 | EXP_010c | Medium's single-basin `D` collapse is cut-dependent: some window (i,j) ≠ (0,23) yields a qualitatively different landscape | **SUPPORTED** | `experiments/exp_010c_windows/RESULTS_EXP010C.md` 2026-07-23 section |
| H9a | EXP_010c | The window most different from baseline sits inside the paper-mapped workspace band | **SUPPORTED on the registered wording, with a recorded measure caveat** | same section |
| H10 | EXP_010c-2 | Landscape character changes non-uniformly as a window edge slides — identifiable onset/exit boundaries, not a smooth gradient | **SUPPORTED** | `RESULTS_EXP010C.md` 2026-07-24 (edge scan) section |
| H10a | EXP_010c-2 | Re-including the last two layers collapses the band window's landscape back toward a single funnel | **REFUTED in its strong form** | same section |
| H10b | EXP_010c-2 | Extending the window to include the sensory front degrades richness even without the motor tail | **SUPPORTED** | same section |
| H11 | **EXP_010d** (capstone) | The Medium band-window loop (A4) recreates GPT-2 Small's basin partition above chance and beyond the A0 baseline | **REFUTED (robustly)** | `experiments/exp_010c_windows/RESULTS_EXP010D.md` |
| H11a | **EXP_010d** (capstone) | A4's effective basin count is closer to Small's than A0's is | **REFUTED for A4** | `RESULTS_EXP010D.md` — **NOT vacated**; see erratum (c) |
| H11b | — | **VACATED.** Sole binding was the retired parallel registration (below). Do not reallocate. | — | — |
| H12 | EXP_010c-3 (in-fill) | The whole-word prompt-dependent injection zone is contiguous across i ∈ {8,9,10} | **REFUTED** (islands, not a band) | `RESULTS_EXP010C.md` 2026-07-25 in-fill section |
| H12a | EXP_010c-3 (in-fill) | The flanking odd layers land on an identifiable one-layer injection edge | **SUPPORTED on the onset side, with a recorded caveat** | same section |
| H13 | EXP_010c-3 (in-fill) | At fixed injection i ∈ {8,10}, whole-word + via-tail-robust character is lost below an identifiable extraction depth | **SUPPORTED; edge at j=21, sharp** — scoped to i ∈ {8,10}; see erratum (d) before generalising | same section |
| H14 | EXP_010c-4 (census) | Aliasing materiality (amended criterion): ≥1 census cell with ≥1 measured neighbour differs in arm class from every measured neighbour | **SUPPORTED (15/50 eligible cells)** | `RESULTS_EXP010C.md` 2026-07-29 census section |
| H15 | EXP_015 | The Small-like tensor partition of Medium's full-stack loop end states survives natural-strength injection: ARI vs Small above chance (perm p < 0.05 at gate threshold 0.999) on the natural_i A0 terminals | **REFUTED** (ARI -0.113, perm p 1.000 at the gate threshold; at/below chance at every sweep threshold) | `experiments/exp_010c_windows/RESULTS_EXP015.md` |

### Retired bindings (kept visible; never reuse)

| ID | Retired binding | Registered at | Verdict recorded under the retired numbering | Disposition |
|---|---|---|---|---|
| H11 | "zone contiguity" — EXP_010c-3 parallel registration | `aea658f` (branch `claude/issue-6-akvuxp`, 2026-07-24 16:11) | REFUTED | **RETIRED.** H11 belongs to EXP_010d per the 2026-07-26 #37 ruling (priority 04:37 `baf3398`; executed verdict). The question is carried by H12. |
| H11a | "edge localisation at one-layer resolution" — same parallel registration | `aea658f` | SUPPORTED (class rule, coarseness caveat) | **RETIRED.** H11a belongs to EXP_010d (claimed `baf3398`, ~11.5 h earlier). The question is carried by H12a. |
| H11b | "extraction independence within the zone rows" — same parallel registration | `aea658f` | NOT SUPPORTED (refutation row fires at (10,19); three cells outside the two-row outcome table) | **RETIRED; ID vacated.** The question is carried by H13. "NOT SUPPORTED" is the mechanically correct label — the retired spec defined no global-REFUTED row. H13 SUPPORTED entails ¬H11b: the two verdicts are concordant, not conflicting. |
| H12 | "aliasing materiality" — EXP_010c-4 census registration | `aa48087` (branch `claude/issue-6-akvuxp`, 2026-07-24 20:06) | SUPPORTED (evaluated after renumber) | **RETIRED as an H12 binding; renumbered to H14** at merge `371e754` per the #37 ruling (main's in-fill H12 has priority, `3ee3126` 19:25 vs 20:06). |

The full text of the retired parallel registration survives at `git show
aea658f:_STAGE2_JSPACE/EXP_010c3_SPEC.md`, and its independent write-up
survives verbatim in `RESULTS_EXP010C.md` §"2026-07-24 — EXP_010c-3 IN-FILL
SCAN — SUPERSEDED PARALLEL RECORD". Session notes
(`sessions/SESSION_2026-07-24_INFILL.md`, `sessions/SESSION_2026-07-29_CENSUS.md`)
use the pre-renumber labels as historical records of what those sessions
wrote at the time; read them against this table.

---

## 2. Experiment register

**Allocate new `EXP_*` IDs here first.** Unqualified "EXP_012" is
**deprecated** in prose — write the suffix (see the EXP_012 rows).

| ID | What it is | Status | Spec | Results |
|---|---|---|---|---|
| EXP_010a | Pythia-410m depth control (H8) | **SUPERSEDED — never ran under its own ID.** Executed as arms P-A0/P-A1 of EXP_012-PYTHIA | `RUNBOOK_PHASE1.md` §EXP_010a; absorbed by `EXP_012_PYTHIA_SPEC.md` | `experiments/exp_010c_windows/RESULTS_EXP012_PYTHIA.md` |
| EXP_010b | GPT-2 Small window grid (H5 coarse) | COMPLETE | `EXP_010b_SPEC.md` | `experiments/exp_010b_small/RESULTS_EXP010B.md` (see the 2026-07-31 correction note there) |
| EXP_010c | GPT-2 Medium window loops (H9/H9a) | COMPLETE | `EXP_010c_SPEC.md` | `experiments/exp_010c_windows/RESULTS_EXP010C.md` |
| EXP_010c-2 | Window-edge scan (H10/H10a/H10b) | COMPLETE | `EXP_010c2_SPEC.md` | `RESULTS_EXP010C.md` |
| EXP_010c-3 | In-fill grid around the word cells (H12/H12a/H13) | COMPLETE | **Spec of record: `EXP_010c3_SPEC.md` as committed at `3ee3126` (main).** Duplicate parallel registration `aea658f` RETIRED — see erratum (a) | `RESULTS_EXP010C.md` 2026-07-25 in-fill section; superseded parallel record retained below it |
| EXP_010c-3b | Follow-up checks on the in-fill (issue #21) | COMPLETE | `EXP_010c3b_SPEC.md` | `RESULTS_EXP010C.md` |
| EXP_010c-4 | Full 300-cell single-step window census (H14), PR #10 | COMPLETE | `EXP_010c4_SPEC.md` | `RESULTS_EXP010C.md` 2026-07-29 census sections |
| EXP_010c-PERM | Anisotropy-corrected permutation control (issue #7) | COMPLETE — **executed control of record** | `EXP_010c_PERM_SPEC.md` (`0ca5829`, 2026-07-25) | `RESULTS_EXP010C.md` §"2026-07-25 — EXP_010c-PERM" |
| PERM_TEST_EXP010c | Duplicate registration of the same control (PR #9 line) | **SUPERSEDED — executed 2026-07-24; its results section was deleted at merge `359c622`** — see erratum (b) | `PERM_TEST_EXP010c_SPEC.md` (`97aeb20`, 2026-07-24; now carries a dated SUPERSEDED header) | Run restored in-tree: results section (archival restoration in `RESULTS_EXP010C.md`) and artifact `experiments/exp_010c_windows/output/permutation_results_2026-07-24_seed2026.json`; concordant with the control of record |
| EXP_010c-ROBUST | Seed and prompt-subset robustness (issue #11) | COMPLETE | `EXP_010c_ROBUST_SPEC.md` | `RESULTS_EXP010C.md` |
| EXP_010c-VARIANTS | Hook-point and energy-normalisation controls | COMPLETE | `EXP_010c_VARIANTS_SPEC.md` | `RESULTS_EXP010C.md` VARIANTS sections |
| EXP_010d | Small-partition capstone (H11/H11a) | COMPLETE | `EXP_010d_SPEC.md` | `experiments/exp_010c_windows/RESULTS_EXP010D.md` |
| EXP_011 | J-space overlap, GPT-2 Small (H6) | **PLANNED — ID reserved** | `STAGE2_PLAN.md` | — |
| EXP_011m | J-space overlap, Medium variant. **PRIMARY ARBITER for workspace-content claims about Medium terminal states** (promoted 2026-08-02; see erratum (e)) | **PLANNED — ID reserved** | `RUNBOOK_JLENS_MEDIUM.md` | — |
| EXP_012 | **The H7 cross-model band census ONLY.** Unqualified use of "EXP_012" for anything else is deprecated | **PLANNED — ID reserved** | `STAGE2_PLAN.md` | — |
| EXP_012-PYTHIA | Pythia placebo window grid (absorbed EXP_010a as arms P-A0/P-A1) | COMPLETE | `EXP_012_PYTHIA_SPEC.md` | `RESULTS_EXP012_PYTHIA.md` |
| EXP_012m | Medium J-lens band census | COMPLETE (NO COHERENT BAND) | `RUNBOOK_JLENS_MEDIUM.md` | `RESULTS_JLENS_MEDIUM.md` |
| EXP_013 | J-corrected readout for ATR trajectories | **PLANNED — ID reserved** | `STAGE2_PLAN.md` | — |
| EXP_013m | J-corrected readout, Medium variant. **DEMOTED from arbiter status** (2026-08-02; see erratum (e)): it can rule latent content in but can never rule it out | **PLANNED — ID reserved** | `RUNBOOK_JLENS_MEDIUM.md` | — |
| EXP_014 | Held-out political-vocabulary rank test on the Small settled basin | **PROVISIONAL — lives on PR #53's branch** (`_STAGE1_REANALYSIS/POLARISATION_SPEC.md` @ `e505963`); not yet on main. **EXP_014 and hypothesis H14 are unrelated despite the number** — different namespaces, no collision | PR #53 branch | PR #53 branch |
| EXP_015 | Natural-loudness ARI comparison vs Small (H15): the direct apparatus-mask test named in the 2026-07-27 session note; analysis-only, issue #59 | COMPLETE | `EXP_015_SPEC.md` | `experiments/exp_010c_windows/RESULTS_EXP015.md` |

---

## 3. Errata

### (a) EXP_010c-3 was pre-registered twice (issue #52)

Two concurrent, mutually-invisible sessions each pre-registered and ran the
same nine in-fill arms on 2026-07-24: branch `claude/issue-6-akvuxp` at
`aea658f` (16:11, H11/H11a/H11b) and main at `3ee3126` (19:25,
H12/H12a/H13, with an explicit note stepping around EXP_010d's H11). Both
registrations were genuine (each committed before its own first run); the
protocol is deterministic (argmax-only, seed 42, same prompt blob), and the
two sessions' artifacts are bitwise identical — determinism, not
replication. Disposition per the 2026-07-26 #37 ruling and the 2026-07-31
execution: main's `3ee3126` is the registration of record; the branch
registration is RETIRED with its write-up preserved as the SUPERSEDED
PARALLEL RECORD section of `RESULTS_EXP010C.md`; the census hypothesis
(registered as H12 at `aa48087`) renumbered to H14 at merge `371e754`.
Full analysis: issue #52 and its 2026-07-30 comment.

### (b) The issue #7 permutation control was also registered twice, and the first run's record was silently deleted

Same failure class as (a), previously unflagged, **with record deletion**:

- **First registration:** `PERM_TEST_EXP010c_SPEC.md`, `97aeb20`
  (2026-07-24 19:24, PR #9 line). Seed 2026; Bonferroni α = 0.05/8 =
  0.00625; token-ID/decile banding. **Executed** — results committed at
  `e734ba0` (19:30): A4 direct/via-tail +4.59σ/+4.60σ (p = 0.0022/0.0019,
  significant); all other sets null.
- **Second registration:** `EXP_010c_PERM_SPEC.md`, `0ca5829`
  (2026-07-25 11:26, PR #19 line). Seed 20260725; α\* = 0.05/9 = 0.00556;
  merges-rank banding. **Executed** — the §"2026-07-25 — EXP_010c-PERM"
  section of `RESULTS_EXP010C.md`: A4 direct/via-tail +4.69σ/+4.55σ
  (p = 0.00090/0.00190, significant); other sets null.
- **The deletion:** at merge `359c622` (2026-07-25 14:02, on the PR #9
  line) the 2026-07-24 results section was dropped from
  `RESULTS_EXP010C.md` — the merge took the main-side file wholesale — and
  `output/permutation_results.json` was overwritten with the second run's
  values. No supersession marker was left anywhere; the first spec sat on
  main headed "PRE-REGISTERED" with no pointer to its replacement. This
  violated the house convention that superseded material stays visible.
- **The science:** the deleted run was **concordant** — A4 ≈ +4.6σ in both
  runs, all other shared sets null in both. What was hidden was favourable
  replication-adjacent evidence, not a discrepancy. Git timestamps show the
  re-registration preceded the second run (honest re-registration, not
  spec-shopping). One threshold-sensitive value, recorded flat: A1_tail
  (p = 0.00620), tested only in the second run, is significant under the
  first spec's α = 0.00625 and not under the second's α\* = 0.00556; as the
  second spec is the registration of record for the second run, its
  recorded reading stands.
- **Remediation (2026-07-31):** dated SUPERSEDED header on
  `PERM_TEST_EXP010c_SPEC.md`; dated erratum in `RESULTS_EXP010C.md` with
  the deleted section restored verbatim in-tree (archival restoration) and
  the run's artifact restored as
  `experiments/exp_010c_windows/output/permutation_results_2026-07-24_seed2026.json`;
  both runs' records now live in the current tree, no historical-commit
  lookup required.

### (c) Discussion #37 registry-table correction: H11a is not vacated

The 2026-07-26 ruling's parenthetical "H11a/H11b → vacated (they existed
only in PR #10's numbering)" is **factually wrong for H11a**: H11a is live
on main as EXP_010d's basin-count sub-hypothesis
(`EXP_010d_SPEC.md`; verdict REFUTED for A4 in `RESULTS_EXP010D.md`),
claimed at `baf3398` (2026-07-24 04:37) — before either in-fill
registration. The error does not change the disposition (priority and
executed verdict both point at EXP_010d); the registry table reads:
**H11a → EXP_010d capstone.** Only H11b is vacated. A correcting comment
has been posted to discussion #37.

### (d) H13 vs the census: scope of "edge at j=21"

H13's verdict ("SUPPORTED; edge at j=21, sharp") is scoped to the
pre-registered extraction ladder at **i ∈ {8, 10}** — its cells are
(8→21) 17/25 and (10→21) 23/25 via-tail vs collapse on every rung below at
the same i. The census did not re-measure any of those cells (it covered
only previously-unmeasured windows), so it does not contradict the verdict
as scoped. What does **not** survive the census is the generalisation from
those two cells to the extraction *column*: the census's eleven j=21 arms
(all i ∉ {8, 10}) average 6.6/25 via-tail agreement (range 0/25–20/25),
and agreement rises roughly monotonically with j to 18.9/25 at j=22
(census table, `RESULTS_EXP010C.md` 2026-07-29 section). "Extraction 21 is
the only reliable depth" — the pre-census hand-forward phrasing — must not
be drawn as a general prior; the J-lens runbook's readout prior was
corrected from census data at the census merge, and this register carries
the same scoping on H13's row above.

### (e) Arbiter swap: EXP_011m promoted, EXP_013m demoted (ratified 2026-08-02)

Roughly fifteen places in the record (RESULTS_EXP010C.md,
RESULTS_EXP010D.md, the EXP_010c-series specs, and session notes) defer to
EXP_013m, the J-corrected re-decode, as "the registered arbiter" for
mid-stack terminal claims. That deferral predates two facts now on the
record: the Medium lens's validation gate returned MARGINAL, and EXP_012m
found NO COHERENT BAND. Under those two facts a null from EXP_013m is
pre-declared ambiguous between "no latent content" and "the instrument
cannot see it" (the 2026-07-27 session note says this itself), so EXP_013m
can rule latent content in but can never rule it out. An arbiter that
cannot adjudicate negatively cannot arbitrate.

**The ruling, ratified under TC's in-session direction of 2026-08-02, per
the review's §6.2 item 8 and the operator report's decision item 5:**
EXP_011m (subspace overlap), which bypasses word readouts entirely and can
deliver both outcomes, is the primary arbiter for workspace-content claims
about Medium terminal states. EXP_013m is demoted to a supporting probe.
Historical sentences reading "EXP_013m remains the registered arbiter"
stay as written per the visible-supersession convention and are read
against this erratum; dated pointer notes have been added to
RESULTS_EXP010C.md, RESULTS_EXP010D.md, and RUNBOOK_JLENS_MEDIUM.md.
Issues #45 and #46 are re-chartered accordingly.

---

*Mirror maintenance: when this file changes an allocation, post the change
to discussion #37 in the same working session. This file is the authority;
the thread is the mirror.*
