# Project review and alignment plan — 2026-07-31

**Status:** LEADERSHIP REVIEW — recommendations for TC to ratify; nothing here executes a contested decision.
**Scope:** the whole repository at `main` = `eb86404`, all 12 issues, all 25 PRs (including open #10 and #53 and their branches), all 16 peer-board discussions, the specs, results, session notes, code, and committed artifacts.
**Method:** commissioned by TC in-session. Twelve parallel review passes (foundations, register, results ×2, issues, open PRs, PR history, board, code-verification) followed by three cross-examining audits (register integrity, evidence strength, process health). Every load-bearing number below was verified against a primary source — a file at a named commit, an artifact JSON, or a GitHub thread — and disputed reader claims were re-checked before inclusion.
**Register note:** this document follows the reporting register. It is agent-authored. Its dispositions are *recommendations*; the ratification path is §6.1.

---

## 1. Executive verdict

Three sentences of state, then the three calls that matter.

**The science is in unusually good shape as a record and in genuinely interesting shape as a result** — verdicts are almost all mechanically supported by committed artifacts, negatives are recorded, deviations are dated. The *positive* story, however, has quietly inverted: the J-space/workspace framing that motivated Stage 2 now hangs on exactly one unrun experiment class (EXP_011/EXP_011m), while the strongest live hypothesis is the **apparatus-mask hypothesis** — that the founding cross-model contrast (Medium's `D` collapse vs Small's semantic basins) is a property of the injection convention, not of the models. **The process is structurally broken in three specific ways** (§5) that produced every one of the eight recorded failures, including the one live governance breach: on 2026-07-31 an agent executed the contested PR #10 merge resolution while issue #52 — which says a human must rule first — stands open.

The three calls:

1. **Unblock the register with one TC comment** (§6.1 has draft text). The analysis in issue #52, the discussion #37 ruling, and merge `371e754` have between them already *done* the right thing on the merits; what is missing is authority, not analysis. Ratify, correct the two factual errors in the registry, land `REGISTER.md`, merge PR #10.
2. **Redirect the science from lens-chasing to the mask hypothesis** (§3.3). The highest-information next experiment is not more J-lens work — it is the still-unfiled natural-energy ARI test, pure analysis on committed artifacts, zero model compute. Then EXP_011m, promoted to primary arbiter in place of EXP_013m (which can rule content in but can never rule it out).
3. **Adopt the seven mechanical rules and two TC obligations** (§5.3). The board, registry, and reconciliation machinery already exist and worked every time someone used them. What failed was never infrastructure — it was defaults (wrong PR bases, undeleted branches), missing claim/ID discipline, and a human gate that merges faster than it reads.

---

## 2. What this project is (one paragraph for orientation)

ATR loops a transformer's residual stream back into itself at constant energy until the state settles. Stage 1 (EXP_009, published separately) found that GPT-2 Small settles into a few semantic basins, Medium collapses to one empty-token funnel (`D`), and Pythia-410m fragments — a finding without a theory. Stage 2 (`_STAGE2_JSPACE/`, plan dated 2026-07-11) tests one candidate theory: that the residual stream has a band structure (input-parsing → workspace → motor, per the Anthropic J-space paper) and the ATR landscape reads it out. Hypotheses H5–H8 plus Q-D; phases: instrument (J-lens fit + gate) → bridge (window grids) → core (subspace overlap, J-readout) → conditional census. Since 2026-07-23 a fleet of agent sessions has executed this plan through GitHub issues, PRs, and a Discussions-based peer board, at high velocity: a 300-window map of GPT-2 Medium, a Small window grid, robustness/hook-point/energy controls, a permutation control, a Pythia placebo grid, a Medium J-lens fit and band census, and a capstone basin-geometry comparison.

---

## 3. State of the science

### 3.1 Every headline claim, graded

Grades: **SOLID** (artifacts in repo support it), **FRAGILE** (supported with known load-bearing caveats), **UNVERIFIED** (claimed; artifacts absent or untestable), **CONTRADICTED** (artifacts or later work undercut it).

| # | Claim | Where | Grade | The caveat that matters |
|---|---|---|---|---|
| 1 | H5-coarse SUPPORTED — Small windows differ from full stack | `_STAGE2_JSPACE/experiments/exp_010b_small/RESULTS_EXP010B.md` | **SOLID** (verdict) | The decision rule ("any window differs on any field") was near-unfalsifiable, so SUPPORTED carries little information. The informative result is the null: **no sub-window carries the Stage 1 semantic landscape** — there is no ATR-side candidate band on Small. Five descriptive numbers in the file contradict its own artifact (§4.4). |
| 2 | Word-forming windows A4 (10→21), O8 (8→21) on Medium | `_STAGE2_JSPACE/experiments/exp_010c_windows/RESULTS_EXP010C.md` | **FRAGILE** | Best positive result. Survives disjoint prompts, hook-point, energy-norm, seed. Single instrument only (logit-lens readout); the J-lens census sees nothing at i∈{8,9,10}; the stopping-rule stability check at these exact cells — flagged 07-25 as the key untested assumption — never ran. |
| 3 | 9→21 fragment funnel (H12 REFUTED; islands, not a band) | `_STAGE2_JSPACE/experiments/exp_010c_windows/RESULTS_EXP010C.md` | **SOLID** | Pre-registered refutation branch fired verbatim; survives disjoint subset. |
| 4 | Anisotropy-corrected permutation: A4 terminals more related than chance (p=0.0009) | `_STAGE2_JSPACE/experiments/exp_010c_windows/output/permutation_results.json` | **SOLID** (A4) | O8 sits at its null mean (z=−0.08) — the "temporal-durative family" reading is A4-only. Register hygiene hole around the duplicate spec (§4.2). |
| 5 | `D` collapse requires layer-0 hook AND ~218× natural injection energy | `_STAGE2_JSPACE/EXP_010c_VARIANTS_SPEC.md` + `_STAGE2_JSPACE/experiments/exp_010c_windows/RESULTS_EXP010C.md` (VARIANTS sections) | **SOLID** | The consequence was never propagated: Stage 1's "Medium collapses to one empty token" is **CONTRADICTED in its unconditional form** — it is apparatus-conditional. Small's landscape is likewise convention-bound (56–175× natural); its natural-energy mirror is unrun. This is the finding the programme now pivots on. |
| 6 | H11 REFUTED — band window does not recreate Small's geometry (EXP_010d) | `_STAGE2_JSPACE/experiments/exp_010c_windows/RESULTS_EXP010D.md` | **FRAGILE** (near-solid) | Numbers verified. But chain of custody is defective: spec, harness, and results entered in a *single commit* on PR #5 — the only "pre-registration" in the family that is testimonial rather than commit-verifiable. Mitigation: the refutation runs against the programme's own hypothesis. |
| 7 | A0 baseline partitions Small-like (ARI 0.200, p=0.0009) | `_STAGE2_JSPACE/experiments/exp_010c_windows/output/basin_comparison_full.json` | **FRAGILE** | Carried by 18/25 prompts in one cluster, against a Small reference with 7 never-converged points; ported from the stranded PR #5 with stale caveats. Feeds the mask hypothesis; needs arbiter (3) below. |
| 8 | Medium J-lens gate MARGINAL | `_STAGE2_JSPACE/RESULTS_JLENS_MEDIUM.md` | **SOLID** (as recorded) | MARGINAL *is* pre-registered vocabulary (RUNBOOK_PHASE0:86). But the multi-hop half of the gate failed outright; the lens is validated for content positions at mid-layers only. |
| 9 | EXP_012m: NO COHERENT BAND on Medium | `_STAGE2_JSPACE/experiments/jlens_medium/census_results.json` (recomputed) | **SOLID, as scoped** | Scored at prediction position −1 only — the regime the gate already showed weakest. A content-position census (position −2, same pipeline) was never run and is cheap. The band rule pre-dates the gate verdict by 7 hours (timing checked; no rule-shopping). |
| 10 | Pythia placebo: window effects present on a non-consolidating model | `_STAGE2_JSPACE/experiments/exp_010c_windows/RESULTS_EXP012_PYTHIA.md` | **SOLID** (placebo) / **FRAGILE** (H8) | Strategically decisive: window-position effects are **not by themselves workspace signatures**. H8's depth verdict is loudness-confounded (63× vs 403× natural). |
| 11 | 300/300 census: `D` in exactly 1 of 300 cells; 21 isolated whole-word cells; H14 aliasing SUPPORTED 15/50 | PR #10 branch @ `371e754` | **SOLID** (mechanically) | Exists only on the unmerged PR #10 branch. Also *undermines* main's H13 "sharp edge at j=21" prior: eleven census j=21 arms average 6.6/25 via-tail agreement vs the two pre-census cells (17/25, 23/25) the verdict rests on. Unadjudicated; on no decision list until now (§6.2, item 8). |
| 12 | "Medium tensor finding" / apparatus-mask synthesis | `_STAGE2_JSPACE/sessions/SESSION_2026-07-27_MEDIUM_TENSOR_SYNTHESIS.md` | **UNVERIFIED (by design)** | Properly fenced as hypothesis-not-finding. All three named arbiters outstanding; the decisive one (natural-energy ARI) is *still not filed anywhere* — it exists only in the session note. |
| 13 | EXP_014: held-out socialist vocabulary rank 15,998 → 70 through the Small loop | PR #53 @ `e505963` | **FRAGILE** | Genuinely pre-registered (spec = first commit); honest handling — no pre-registered outcome row matched, recorded as spec defect rather than resolved to nearest row. |
| 14 | "Medium has a political attractor" (` Republican`, 25/25 via-tail, survives natural-energy control) | PR #53 @ `e505963` | **FRAGILE / provisional** | Not pre-registered; single readout convention; the energy control landed in the final, unreviewed commit; headline analyses read census data from PR #10's branch via hardcoded `/home/user/census*` paths with no provenance stated (§6.3). |

### 3.2 What is dead, what survives

**Dead on the artifacts (regardless of formal kill status):**
- The naive 40–90% depth mapping that set the window grid's coordinates — no J-lens support anywhere in it.
- H7's existence conjunct at 345M: Medium has no coherent prediction-position band. Phase 3's four-model census as originally motivated ("band presence/shape predicts ATR regime") — the middle model has no band and window effects are placebo-generic, so there is nothing to correlate. Only a content-position census could revive it.
- The 07-23 falsifiable prediction "two instruments should localise the same band" — recorded as not borne out.
- H11/H11a-for-A4 — refuted, robustly.
- **H5 as workspace evidence.** Its letter survives trivially; but no Small sub-window carries the landscape, the Medium word cells are isolated islands, and window-dependence per se reproduces on Pythia. Spent.
- **EXP_013m as decisive arbiter.** With a MARGINAL gate and no band, it is pre-declared ambiguous-on-null: it can rule latent content *in* but never *out*. The record defers to it ~15 times. Demote it (re-charter issues #45/#46 accordingly).
- The plan-level kill criteria **can never fire as written** — their first conjunct ("EXP_010 shows no qualitative window-dependence") is already false. The frame can currently be neither killed nor confirmed by its own registered rule. Re-register them (§6.2, item 6).

**Survives:**
- **EXP_011 (Small) / EXP_011m (Medium)** — subspace overlap, the one registered test that bypasses readout rankings entirely; the only remaining leg that can formally close or save the workspace framing. Hidden costs: the Small lens was never fitted (no RESULTS_PHASE0.md exists in this repo), the Medium lens artifact is gitignored and may need a ~7h refit, and the BOS seam (§4.4) must be settled first.
- H6 (untested), H8 (supported, loudness-confounded), **Q-D / EXP-D** — the plan's *unconditional* deliverable, still unrun, needing only the frozen Stage 1 zip. The cheapest fully self-contained deliverable in the plan.
- The A4/O8 islands and A4's permutation pass — as ATR-internal findings about the looping procedure, no longer as workspace evidence.
- **The apparatus-mask hypothesis** — the programme's new centre of gravity, properly fenced, arbiters named but unrun.

### 3.3 Where it needs to go — the science queue, in order

1. **File and run the natural-energy ARI test** (arbiter 3 of the 07-27 synthesis; currently vapor). Compare committed `terminals_energynorm_A0.pt` against Small's committed partition (`terminals_small_010d.pt`, same 25 prompts) with the existing `compare_small_basins.py` ARI + permutation machinery. Zero model compute; discriminating outcomes already written down on 07-27. Significant ARI without `D` → the mask hypothesis strengthens and the founding cross-model contrast is likely an apparatus artifact; chance ARI → it deflates. Either branch moves the deepest live question, and its outcome determines whether EXP_011m is even aimed at the right phenomenon. Spec first — it must not repeat the H11 custody defect. *(Half a day.)*
2. **Run the A4/O8 stopping-rule stability check** (check_start 10 vs 100 at the two flagship cells) — the record's own 07-25 proposal, silently dropped. Cheapest test that could invalidate the flagship cells. *(Hours of CPU.)*
3. **Resolve the BOS seam, then run EXP_011m as promoted primary arbiter** with an anisotropy-matched null; simultaneously demote EXP_013m in every record that names it "the registered arbiter." *(1–2 days.)*
4. **Run EXP-D** from the frozen Stage 1 zip, and the **content-position census variant** (position −2, same pipeline) before any "Medium has no band tout court" claim leaves the repo. *(1 day combined.)*
5. **Re-register the kill criteria** (proposal in §6.2, item 6), then decide Phase 3 / EXP_011-Small on the joint outcome of steps 1 and 3. If the mask hypothesis strengthens and EXP_011m is null, close the workspace framing per the re-registered criteria and write the negative up — the surviving paper (apparatus-conditionality + the island characterisation) is a genuinely novel methodological finding about activation-loop instruments, publishable under the house rule that negatives are findings.

---

## 4. The register — adjudication of issue #52 and what the review found beyond it

### 4.1 Issue #52 is correct, and understated

Every claim in the amended filing checked out first-hand: EXP_010c-3 was pre-registered twice by concurrent sessions (branch `aea658f` 07-24 16:11 with H11/H11a/H11b; main `3ee3126` 19:25 with H12/H12a/H13), both ran it, artifacts bitwise identical (determinism, not replication — argmax-only protocol, same seed, same prompt blob), and three hypothesis IDs each carry two meanings — with H11a and H12 carrying **opposite verdict words** under their two bindings. The recommended canonical table:

| ID | Canonical binding | Retired binding | Notes |
|---|---|---|---|
| **H11** | EXP_010d "geometry recreation" — REFUTED | branch "zone contiguity" | Priority (07-24 04:37), executed + merged, #37 ruling. Already implemented at `371e754`. |
| **H11a** | EXP_010d "basin count" — REFUTED for A4 | branch "edge localisation" — SUPPORTED | The #37 ruling's "H11a vacated" is **factually wrong** — H11a is live on main (`EXP_010d_SPEC.md:37`, `RESULTS_EXP010D.md:87`). Registry must be corrected. |
| **H11b** | *(vacated)* | branch "extraction independence" — NOT SUPPORTED | Question carried by main's H13. "NOT SUPPORTED" is the mechanically correct label (the branch table defines no global-REFUTED row). H13 SUPPORTED entails ¬H11b — the verdicts are **concordant**, not conflicting. |
| **H12** | main "injection-zone continuity" — REFUTED | branch census "aliasing materiality" — SUPPORTED 15/50 | Census renumbered to **H14** at `371e754`; verified in the merged spec. |
| **H12a, H13** | main-only | — | Keep. The merge's decision *not* to relabel the retired H11-series 1:1 into H12a/H13 is scientifically right — the criteria differ (H12a makes a directional sub-prediction H11a declines; H11b's outcome table has a three-cell coverage hole H13 lacks). A mechanical relabel would falsely assert criteria identity. |
| **H14** | census "aliasing materiality" — SUPPORTED 15/50 | — | New allocation, correct. Lives only on PR #10. **Next free hypothesis ID: H15.** |

**Main needs zero renumbering** — every canonical ID already lives there. The whole remediation is one REGISTER.md, two errata edits, two governance-doc corrections, one board comment, and one ratification comment.

### 4.2 A second incident of the same class, previously unflagged — with record deletion

The issue #7 permutation control was **also** registered twice: `PERM_TEST_EXP010c_SPEC.md` (07-24, PR #9 line; seed 2026, α=0.00625, token-ID banding) and `EXP_010c_PERM_SPEC.md` (07-25, PR #19 line; seed 20260725, α=0.00556, merges-rank banding). Both were *executed*. At merge `359c622` the 07-24 run's results section was **silently deleted** — main today has zero trace of it, while its spec sits on main still headed "PRE-REGISTERED" with no supersession marker. This violates the repo's own "superseded stays visible" convention, and the destroyed run was *concordant* (A4 ≈ +4.6σ in both) — favourable replication evidence was hidden. Git timestamps show the re-registration preceded the second run (honest re-registration, not spec-shopping), but as committed, the register permits after-the-fact spec selection, and one contrast (A1_tail, p=0.00620) sits exactly between the two specs' thresholds. Remediation in §6.2, item 4.

### 4.3 Experiment-ID hygiene

- **EXP_012 means three things on main**: the planned H7 cross-model census (never run), EXP_012-PYTHIA (the placebo grid that absorbed never-run EXP_010a), and EXP_012m (the Medium band census). Canonical: unsuffixed EXP_012 = the H7 census only; prose always carries the suffix.
- **EXP_010a**: register as SUPERSEDED → executed as arms P-A0/P-A1 of EXP_012-PYTHIA.
- **EXP_014** (PR #53) is registered nowhere and is now numerically adjacent to hypothesis H14 — different namespaces, no collision, but exactly the looks-free trap that caused #52. Register both.
- **EXP_011/011m/013/013m**: register as PLANNED so nobody re-allocates.

### 4.4 Integrity defects to correct (verified except where marked, with locations)

1. **`RESULTS_EXP010B.md:45, 52–54`** — five numbers (SB basin-size tuple; off-diag cosine minima for SB/S2/S3/S5) match no committed artifact; `terminal_characterisation_small010b.json` says [6,5,4,3,3,3,1] and 0.6921/0.4983/0.97/0.441. Not verdict-bearing; still a breach of "committed artifact = authority."
2. **`run_exp010c.py:352–366`** — the artifact-overwrite guard still omits `--arms` from the variant list: a bare `--tier full --arms A5` run would silently replace the registered 150-record `results_full.json`. A `--tag` value colliding with a tier name is also unchecked. This is the exact failure mode the guard was built for.
3. **BOS seam** *(suspected — the one item in this list pending verification)* — `RESULTS_JLENS_MEDIUM.md:293` claims "the ATR engine does not prepend BOS"; the engine passes raw strings to `run_with_cache` and transformer_lens defaults `default_prepend_bos=True` for GPT-2, so the note is probably **backwards**. Classify as a confirmed defect only once the one-line check against pinned TL 3.5.1 is run and documented; either way the seam must be settled before any lens-on-terminals work (EXP_011m/EXP_013m).
4. **`permutation_results.json`** lacks the `input_sha256` attestation field its own spec addendum describes — never regenerated after the attestation code landed.
5. **Discussion #37 registry table** — "H11a vacated" is wrong (see 4.1); H10-series, H14, EXP_014, and every live EXP ID are unregistered; the registry has never once been used prospectively.
6. **`SKILL.md:120`** — "H11 was claimed three separate ways" is a verified over-count (two claims — PRs #5 and #10 — plus PR #20/#33's explicit *deferral*). Every agent loads this skill; its founding incident should be stated correctly.
7. **Stale caveats** — EXP_010d's rescued record carries "single seed" (vacuous since 07-25; should read "single prompt subset") and no stopping-rule caveat on its terminal partitions; H9/H10a/H10b verdicts phrased against "the collapse" were never re-scoped after the collapse was shown apparatus-conditional.

---

## 5. Process diagnosis

### 5.1 Three root causes, eight failures

Every recorded failure traces to one of three causes — not eight separate problems:

- **(A) Concurrency without a ledger.** Parallel, mutually-invisible sessions under one shared GitHub identity, no mandatory claim step, no ID allocator. → Issue #7 implemented three times (PRs #8, #9, #19); H11 collision; EXP_010c-3 executed twice concurrently (a third agent nearly started it again — its stand-down comment on #6 proves the check is performable).
- **(B) Defaults inherited from the workspace; hygiene owned by nobody.** Agents open PRs against whatever branch they were spawned on; merged branches are never deleted (all 20 `claude/*` heads still live on origin); every session's duty ends at PR-open. → PR #5 stranded on a dead feature-branch base (the H11 capstone survived only via a session-note warning and a manual rescue); PR #20 merged into a non-main base, its 12 commits unreachable from main for ~3 days; five open issues (#6, #11–#14) describe work that merged days ago — the tracker misstates programme state by half; seven zero-comment board threads; settled thread #38 still open.
- **(C) A human gate that is a fast merge button.** Verified: PR #9 was merged 72 minutes *after its own thread recommended closing it* (and it landed a second redundant "PRE-REGISTERED" spec — the §4.2 incident). Zero board CONCUR flags exist anywhere; the three most substantive recent PRs (#47, #49, #53) have zero-comment board threads; review pressure comes only from CodeRabbit, whose file-count and rate limits both open PRs had to engineer around. And the decision vacuum is real: issue #52 asked for a human ruling on 07-29; none came; on 07-31 an agent executed the contested resolution itself (`371e754`).

**Cross-cutting: no verifiable human voice.** Every issue, PR, comment, merge, and "ruling" is authored under the single owner account with agent footers. "Delegated by the operator, TC, in-session" is unfalsifiable from artifacts. Until human decisions are distinguishable from agent output, agents will keep acting on unverifiable authority — `371e754` happened in exactly that gap.

### 5.2 On `371e754` specifically (the 07-31 merge)

Adjudication: **partially authorized, substantively right, procedurally wrong, and reviewed by no one.** The renumber half (main-wins + census H12→H14) was pre-authorized in terms by the #37 ruling ("If PR #10 lands in any form, its spec renumbers to match PR #33's H12-series before merge"). The presentation half (retain-don't-relabel the H11-series record) decides territory the #52 comment explicitly marked "OWNER SIGN-OFF." On the merits both choices are defensible — this review recommends ratifying both — but being right is not being ratified, and the 9 substantive files of the merge resolution have never been reviewed by anyone. The lesson for the rulebook: *file and wait; never self-execute a contested resolution without a citable ruling artifact* (T2 below).

### 5.3 The rules (add to a new `CLAUDE.md`; ⛔ = CI-enforced)

**For agents:**
- **R1 — Base rule ⛔.** Every non-stacked PR bases on `main`. Stacked PRs declare "STACKED ON #N" in the body and may base on the referenced PR's branch; CI validates the declared dependency and fails any PR that neither bases on `main` nor carries a valid declaration. Merged branches are deleted at merge. *(Prevents: #5, #20.)*
- **R2 — Claim rule.** Before starting issue N: comment `CLAIM <session-handle>` on it and check open PRs referencing N; if claimed within 48h, stand down. *(Prevents: #7 ×3, EXP_010c-3 ×2.)*
- **R3 — ID rule ⛔.** `_STAGE2_JSPACE/REGISTER.md` is the authority for every H- and EXP-number. Claim by commit before first use. CI validates canonical identifier tokens in every diff against REGISTER.md and fails on any absent one — via a precise token grammar, not a bare grep: hypothesis IDs match `\bH\d+[a-z]?\b` and experiment IDs match `\bEXP_\d{3}[a-z0-9]*(-[A-Z0-9]+)?\b`, evaluated outside file paths, filenames, and code spans; shorthand (`EXP_011/011m`) is banned in machine-checked contexts — specs write every identifier in full. The board thread #37 becomes the registry's *mirror* for branch-blind sessions, not the authority. *(Prevents: the entire #52 class. This is issue #52 §11; implement it.)*
- **R4 — Closing rule ⛔.** A results-bearing PR contains `Closes #N` or one line saying why not. *(Prevents: stale #11–#14.)*
- **R5 — Scope rule.** One experiment per PR; long-running jobs never run on a branch with an open dispute; artifact sets >50 files land as a separate artifacts-only PR referenced from the results PR; every artifact-bearing tier commits its run log ⛔. *(Prevents: the #10/#51 contortion and the §6 audit gap.)*
- **R6 — Post-merge duty.** The last actor (not the opener) runs the five-minute checklist: issue closed, board thread closed with outcome, PR body accurate, branch deleted. *(Prevents: zombie threads, stale trackers.)*
- **R7 — Verification rule.** "Verified" means completion evidence — exit code, artifact digest, reproduced number — never startup evidence. Promote the onboarding thread's lessons into CLAUDE.md.
- **R8 — Authority rule.** Never self-execute a resolution an open issue reserves for a human. File, cite, wait. If the ruling artifact doesn't exist, the action doesn't happen.

**For TC (two obligations, non-negotiable):**
- **T1 — Merge discipline.** Before merging any results-bearing PR, read its thread and board thread; if both are silent, the merge comment says `merging unreviewed`. Never merge within 2 hours of PR-open. *(Silence never blocks, but silence gets named.)*
- **T2 — Decision SLA + signed authority.** Anything marked "owner ruling required" gets a comment beginning **`TC-RULING:`** within 48h — even if the ruling is "delegated to session X, scope Y." Post rulings from the web UI with that prefix so they are distinguishable from agent output. Assign one reviewer session per substantive results PR at spawn time, so board review is a job, not volunteering.

**Advisory vs blocking, settled:** block what machines evaluate perfectly (base branch, ID registry, closing keyword, run-log-with-artifacts, harness smoke run); keep human/peer judgment advisory but *named* at merge time. A hard peer-review gate with one volunteer reviewer would stall the pipeline or breed rubber stamps.

---

## 6. The alignment plan — ordered instructions

### 6.1 Step 0 (TC, ~15 minutes, unblocks everything): post the ratification

Draft text for a comment on issue #52, to post (ideally from the web UI) and adapt as you see fit:

> **TC-RULING (issue #52, all open items):**
> 1. The 2026-07-26 identifier-registry ruling (discussion #37) was delegated by me and stands: **H11 and H11a belong to the EXP_010d capstone line**. The ruling's parenthetical "H11a/H11b vacated — existed only in PR #10's numbering" is corrected: **H11a is live on main** with verdict REFUTED-for-A4; only H11b is vacated.
> 2. **Main's `EXP_010c3_SPEC.md` (`3ee3126`) is the pre-registration of record** for EXP_010c-3. The branch spec (`aea658f`) is the superseded parallel registration; it survives as the marked SUPERSEDED PARALLEL RECORD section plus its git hash — no separate archive file needed.
> 3. **Merge `371e754` is ratified in both halves**: main-wins numbering with census H12→**H14**, and retain-without-relabel of the H11-series verdicts (the criteria differ; a 1:1 relabel would falsely assert criteria identity).
> 4. **H13 vs H11b** (§10.6–7): concordant, not conflicting — H13 SUPPORTED entails ¬H11b; "NOT SUPPORTED" is the correct H11b label. The coverage-hole note is carried flat; no better-posedness ranking is recorded. The **H13-vs-census tension** (sharp edge at j=21 vs 6.6/25 mean agreement over eleven census arms) is opened as its own issue and must be adjudicated before any J-lens phase consumes "extraction 21 is the only reliable depth" as a prior.
> 5. **§10.8 (concurrency + missing main in-fill logs):** the disjoint-history / write-order / independent-spec evidence is sufficient; no further audit step gates the merge. The branch's three `infill_run*.log` files close the log gap at merge.
> 6. **§12 (self-flagged items):** RUNBOOK_JLENS_MEDIUM §3a is reframed as a flat observation ("21 cells have whole-word prompt-dependent terminals"); the §3 branch table is demoted to non-authoritative sketch — the realized outcome falls outside every row.
> 7. **Next free hypothesis ID is H15.** A file-based `REGISTER.md` (issue #52 §11) is adopted as the in-repo authority, with the board thread as its mirror.
> PR #10 is unblocked. This issue closes when REGISTER.md lands.

### 6.2 Week 1 — land the record (mostly agent work, one PR each unless noted)

1. **Review + merge PR #10.** One review pass over the 9 non-artifact files at `371e754` (the merge resolution nobody has reviewed); update the PR title/body from retired H11-series to H12/H13/H14 reality; merge; close #6. (#52 stays open until step 2 completes — its close condition is REGISTER.md landing, per §6.1.)
2. **Create `_STAGE2_JSPACE/REGISTER.md`** — hypothesis table H0–H14 with canonical + retired bindings and git hashes (per §4.1); experiment table (EXP_010a→superseded, 010b/c/c-2/c-3 (record: `3ee3126`)/c-3b/c-4/c-PERM (executed control of record)/c-ROBUST/c-VARIANTS/010d, PERM_TEST_EXP010c (superseded duplicate, executed 07-24, see erratum), EXP_011/011m/013/013m (PLANNED), EXP_012 (=H7 census only; suffixed IDs distinct), EXP_012-PYTHIA, EXP_012m, EXP_014 (provisional, PR #53)); ERRATA section (both double-registration incidents, the #37 correction, next free IDs). State the rule: no new H/EXP token without a REGISTER.md row in the same commit. **Close #52 when this PR merges** — that is the close condition §6.1 sets.
3. **Errata edits** (same PR as 2): SUPERSEDED header on `PERM_TEST_EXP010c_SPEC.md` naming the executed registration; dated note in `RESULTS_EXP010C.md` restoring the fact of the deleted 07-24 permutation run (`97aeb20`/`e734ba0`, seed 2026, concordant); correct `RESULTS_EXP010B.md:45,52–54` against its committed JSON; stale-caveat fixes and the H9/H10 conditional re-scope (§4.4.7); custody note on EXP_010d (spec and results share one commit).
4. **Code fixes** (one PR): add `--arms` to the overwrite-guard variant list; reject `--tag`/`--out-suffix` equal to tier names; run and commit the one-line BOS check and correct `RESULTS_JLENS_MEDIUM.md:293` accordingly; regenerate `permutation_results.json` with attestation.
5. **Governance docs** (one PR): new `CLAUDE.md` with R1–R8 + T1–T2; fix `SKILL.md:120`; CI checks for R1/R3/R4/R5.
6. **Re-register the kill criteria** (Fable/design session + TC sign-off). Proposal: *"the workspace framing ends if EXP_011 (Small) and EXP_011m (Medium) both show basin/window terminals projecting onto J-space no differently than matched nulls, given the recorded Medium band null and the absence of a Small candidate band."* Pre-register the nulls with the PERM spec's rigour.
7. **Housekeeping sweep** (one agent session): close issues #11–#14 with pointers to merged deliverables; board sweep — close #38 with its resolution, close the six stale zero-comment PR threads, correct #37 and register H10-series/H14/EXP_014 and all live EXP IDs; correcting comment on PR #9's thread (the merged implementation was #19, not #8); delete all merged/superseded remote branches.
8. **Open the H13-vs-census issue** (from ruling item 4) and re-charter #45/#46 with EXP_013m demoted from arbiter status.

### 6.3 PR #53 — disposition

**Hold until PR #10 merges, then land in two parts. Do not merge as-is; do not close.**
- **Part 1 (mergeable on its own merits): EXP_014** — spec, results, addendum. Pre-registered, gate-verified, honestly recorded.
- **Part 2 (after #10): the Medium political work**, conditional on: a provenance paragraph in `RESULTS_MEDIUM_SOCIALIST.md` naming the census artifacts and commit they were read from; hardcoded `/home/user/census*` paths repointed at the in-repo census; re-review of the three post-review commits (especially the now-load-bearing Republican energy control); a short registered spec for the capture experiment and a decontaminated rival set; and a **TC ruling on whether `_STAGE1_REANALYSIS/` belongs in this repo at all** — the outline's own rule says Stage 1 material lives in the lucier repo.
- **Framing:** the PR title's interpretive register ("Can GPT-2 Medium be made Marxist? No —") violates the house rule that interpretation lives in fenced notes; rename before merge. All §4–§7 claims remain provisional on a single readout convention; if any of this material is ever shared externally, the "one readout convention, arbiter never run" caveat travels with it. EXP_014 and H14 are unrelated despite the number — REGISTER.md notes this explicitly.

### 6.4 Weeks 2–3 — the science (order from §3.3)

Natural-energy ARI (spec first) → A4/O8 stopping-rule check → BOS-resolved EXP_011m as primary arbiter → EXP-D + content-position census → Phase-3/close decision under the re-registered kill criteria.

---

## 7. What this review did and deliberately did not do

Done: read everything; verified every load-bearing claim against primary sources; produced this report on branch `claude/repo-review-alignment-knzmrj` as a draft PR. **Not done, deliberately:** no issue was closed, no board post was made, no registry file was created, PR #10 and #53 were not touched, and no ruling above is executed — per the repo's own standing rule, and per §5.2's lesson, everything in §6 waits on the §6.1 ratification or explicit TC direction. The one exception an agent may take immediately without ratification: the §6.2 item 4 code fixes, which guard committed artifacts and adjudicate nothing.

*Sources: full reader and audit reports for this review (12 sessions, ~390 primary-source checks) are available in the session transcript; every file:line and commit hash cited above was re-verified at `main` = `eb86404`, PR #10 head = `371e754`, PR #53 head = `e505963`.*
