# AUDIT: Identifier-Register Integrity (adjudication of issue #52)

## VERDICT

Issue #52 is factually accurate in every claim I checked firsthand and, if anything, understates the problem: I confirmed not only the three-way hypothesis-ID collision (H11, H11a, H12 each bound to two different pre-registered statements, with H11a and H12 carrying **opposite verdict words** under their two meanings) but also a second, unflagged incident of the same class — the issue #7 permutation control was pre-registered twice with different frozen parameters (seed 2026 vs 20260725, α/8 vs α/9, different frequency banding), run twice, and the earlier run's results section was **silently deleted** at merge `359c622`, leaving an orphaned "PRE-REGISTERED" spec (`PERM_TEST_EXP010c_SPEC.md`) on main with no supersession marker and zero trace of its executed run — a direct violation of the repo's own "superseded stays visible" convention. The correct canonical assignments are already settled in substance by the discussion #37 ruling plus the #52 comment's D1–D8 analysis, and were mechanically implemented on PR #10's branch by merge `371e754` on 2026-07-31; the assignments themselves are right (H11/H11a → EXP_010d; H12/H12a/H13 → the in-fill; census hypothesis → new H14). What is missing is not analysis but **authority and infrastructure**: the resolution was executed by an agent while the issue demanding a human ruling stands open, the #37 registry table contains a verified factual error (H11a is "vacated" per the ruling but live on main with a recorded verdict), the registry has never once been used prospectively, and no in-repo REGISTER file exists. The fix is cheap: main needs **zero renumbering** — every canonical ID already lives there — so the whole remediation is one new REGISTER.md, two small errata edits, two one-line governance-doc corrections, one board comment, and one human ratification comment on #52.

## DETAILED FINDINGS

### 1. The ambiguous hypothesis IDs — verified firsthand, with canonical resolutions

I read both colliding specs at their source commits (`EXP_010d_SPEC.md` on main; main's `EXP_010c3_SPEC.md:44-59`; the branch's spec at `498e7ca`; the census spec at `371e754`), both verdict records, issue #52 and its 07-30 comment in full, and the #37 ruling via the board-state mirror (snapshot 2026-07-31T11:01:40Z — #37 still OPEN, 1 comment, table uncorrected).

| ID | Binding A (canonical) | Binding B (retired) | Resolution |
|---|---|---|---|
| **H11** | EXP_010d "geometry recreation" — **REFUTED (robustly)** (`RESULTS_EXP010D.md:80`; claimed `baf3398` 07-24 04:37) | branch EXP_010c-3 "zone contiguity" — REFUTED (`aea658f` 07-24 16:11) | A wins on priority (11h35m earlier), executed-and-merged status, and the #37 ruling. Already implemented at `371e754`. |
| **H11a** | EXP_010d "basin count" — **REFUTED for A4** (`RESULTS_EXP010D.md:87`) | branch "edge localisation" — **SUPPORTED** | A wins, same grounds. Note the #37 ruling's "H11a → vacated (existed only in PR #10's numbering)" is **factually wrong** — I verified H11a defined at `EXP_010d_SPEC.md:37` with a merged verdict. Registry table must read "H11a → EXP_010d". |
| **H11b** | *(no main binding)* | branch "extraction independence" — NOT SUPPORTED | Vacated/retired. The question is carried by main's H13. "NOT SUPPORTED" (not "REFUTED") is the correct label — the branch spec's outcome table defines no global-REFUTED row (D6 is right). |
| **H12** | main EXP_010c-3 "injection-zone continuity" — **REFUTED** (`RESULTS_EXP010C.md:426`; `3ee3126` 07-24 19:25) | branch EXP_010c-4 census "aliasing materiality" — **SUPPORTED 15/50** (`aa48087` 07-24 20:06) | A wins (41 min earlier, merged). B renumbered to **H14** at `371e754`; verified in the merged `EXP_010c4_SPEC.md` (lines 38-43, 127-147) with numbering note citing #37. |
| **H12a, H13** | main-only; no collision | branch's H11a/H11b are *adjacent questions with different criteria* | Keep as-is. The merge's decision NOT to relabel the retired H11-series 1:1 into H12a/H13 is scientifically defensible (see Disputed point 2) but deviates from the #52 comment's D1/D3 proposal and needs explicit owner sign-off. |
| **H14** | census "aliasing materiality" — SUPPORTED 15/50 | — | New allocation, correct (D7's sweep found H14 free; H1–H3 are gaps that should not be backfilled). Lives only on unmerged PR #10; must be registered in #37 and REGISTER.md. Next free ID: **H15**. |

Non-ambiguous but register-relevant: H5/H5-coarse (SUPPORTED, EXP_010b), H6 (untested), H7 (untested), H8 (SUPPORTED with load-bearing energy caveat, EXP_012-PYTHIA), H9/H9a (SUPPORTED/SUPPORTED-with-caveat, EXP_010c), H10/H10a/H10b (EXP_010c-2). None of H10-series, H14, or any EXP ID has ever been claimed in registry #37.

### 2. The ambiguous experiment IDs

**EXP_010c-3** — one experiment ID, two valid pre-registrations (each committed before its own session's first arm: 16:11→16:30 and 19:25→19:56), both executed, artifacts bitwise identical (issue #52 §5; determinism, not replication — the protocol is argmax-only, seed 42, identical prompt-subset blob on both refs). **Canonical spec of record: main's `3ee3126`.** The branch's spec now survives only in git history and in the merged RESULTS' "SUPERSEDED PARALLEL RECORD" section — S1's proposed verbatim archive file was never created.

**The issue #7 permutation control** — the second double-registration, absent from #52. Verified: `97aeb20` (07-24 19:24, PR #9 branch) registered `PERM_TEST_EXP010c_SPEC.md`; `e734ba0` (19:30) committed its full run (seed 2026, section "## 2026-07-24 — Anisotropy-corrected permutation test (planned control 1)"). A different session registered `EXP_010c_PERM_SPEC.md` (`0ca5829`, 07-25 11:26) and ran it (seed 20260725). Merge `359c622` (07-25 14:02, conflicts on RESULTS_EXP010C.md / permutation_results.json / permutation_test.py) took main's versions: I confirmed the 07-24 section is **absent from the merged tree** and that main today has zero matches for "PERM_TEST" or "seed 2026" in RESULTS_EXP010C.md, while the orphaned spec sits on main headed "PRE-REGISTERED". This is record destruction, worse in kind than #52's duplication — and the destroyed run was **concordant** (A4 ≈ +4.6σ pass, all else null in both), i.e. favorable replication evidence was hidden. **Canonical: EXP_010c-PERM is the executed registration of record; PERM_TEST_EXP010c_SPEC.md must be marked SUPERSEDED and the 07-24 run restored to the record by dated erratum.**

**EXP_012** — three meanings, all verified live on main: (a) `STAGE2_PLAN.md:23,102` / `STAGE2_OUTLINE.md:58` — the Phase-3 cross-model J-lens band census (H7, never run); (b) EXP_012-PYTHIA — the ATR placebo grid absorbing the never-run EXP_010a (`EXP_012_PYTHIA_SPEC.md:27` "H8 depth control (EXP_010a, never run)"; `EXP_010b_SPEC.md:182` "EXP_010a's successor ran as EXP_012"); (c) EXP_012m — the Medium band census (run; NO COHERENT BAND). **Canonical: unsuffixed "EXP_012" = the H7 cross-model census and nothing else; the suffixed IDs are distinct experiments; prose must always carry the suffix.** No file renames needed — REGISTER.md entries resolve it.

**EXP_010a** — never run; register as SUPERSEDED → executed as arms P-A0/P-A1 of EXP_012-PYTHIA.

**EXP_014** — used on PR #53's branch (`_STAGE1_REANALYSIS/POLARISATION_SPEC.md`) with no registry claim, and now numerically adjacent to hypothesis H14. Different namespaces, so not a collision, but exactly the "looks-free" trap that caused #52 — register both explicitly.

**EXP_010c-4, EXP_011/011m, EXP_013/013m** — census (PR #10 only; register on merge); planned-never-run (register as PLANNED so nobody re-allocates).

### 3. Governance and process facts

- Merge `371e754` (07-31 10:07, agent-authored, same session as PR #10) implemented main-wins + H12→H14 + superseded-parallel-record **while issue #52 ("PR #10 stays unmergeable until a human rules") is open**. The renumber half was pre-authorized in terms by #37 ("If PR #10 lands in any form, its spec renumbers to match PR #33's H12-series before merge"); the presentation half (retain-don't-relabel) was not.
- #37's delegation ("delegated by the operator, TC, in-session 2026-07-26") is unverifiable from any artifact; every issue/PR/comment/merge is authored under the single `earlyprototype` account. Only TC can close this gap.
- `SKILL.md:119-120` "H11 was claimed three separate ways" is a verified over-count (two claims — PR #5, PR #10 — plus PR #20's explicit deferral "continues at H12 to avoid collision", which I read in main's spec at lines 44-45).
- No REGISTER file exists anywhere on main (`git ls-files | grep -i REGISTER` → empty). Registry #37 has never received a prospective claim.
- Main committed 24 run logs but zero for the in-fill tier; the branch's three `infill_run*.log` files close that gap post-merge.

## DISPUTED POINTS RESOLVED

1. **Was #52's collision claim correct?** All three readers said yes; **confirmed** — I re-read the issue body and re-derived every binding from the spec files at their source commits. The specs-register reader's stronger claim (a fourth same-class event, the permutation control) is **also confirmed** independently: I verified the two spec headers on main, the 07-24 run section at `e734ba0`, its absence from `359c622`'s tree (grep exit 1), and its absence from main today. The issues reader found the duplication but not the deletion; specs-register is right that the deletion is the sharper finding.

2. **Was the 371e754 merge resolution authorized?** The issues reader treats it as a governance breach; pr-open calls the numbering half "colorable"; specs-register flags absent ratification. **Adjudication: partially authorized.** The H12→H14 renumber and main-wins spec choice are within #37's explicit pre-authorization clause (I read the ruling verbatim in the board-state mirror). The retain-without-relabelling of the H11-series verdicts is a *deviation* from the #52 comment's D1/D3 (1:1 relabel), decided by an agent on territory the comment itself marked "⚠ OWNER SIGN-OFF". On the merits I side with the merge's choice — I compared the criteria myself: main's H12a pre-registers a directional sub-prediction the branch's H11a explicitly declines, and the branch's H11b outcome table has a three-cell coverage hole H13 lacks, so a mechanical 1:1 relabel would falsely assert criteria identity. But being right is not being ratified: one TC comment fixes this.

3. **Is #37's registry table correct?** The board reader said H11a is "live, not vacated"; the ruling says vacated. **The board reader is right** — verified `EXP_010d_SPEC.md:37` ("H11a (basin count)") and `RESULTS_EXP010D.md:87` ("REFUTED for A4") on main. The ruling's disposition of H11 survives; its parenthetical is wrong and the table must be corrected.

4. **"H11 claimed three separate ways" (SKILL.md)?** The board reader accepted it ("corroborated by #37's own table"); specs-register and issues call it an over-count. **Over-count confirmed** — PR #20's committed spec text is a deference, not a claim. The board reader is wrong on this point; the governance doc misstates its own founding incident and should be corrected.

5. **Is "NOT SUPPORTED" the right H11b label, and are H13/H11b in conflict?** All readers converge with D5/D6; **confirmed by reading both outcome tables**: H13 SUPPORTED entails ¬H11b, so the verdicts are concordant, not conflicting; the branch table defines no global-REFUTED row, so "NOT SUPPORTED; refutation row fires at (10,19)" is the mechanically correct output.

6. **Line-number discrepancies between readers** (census H12 verdict at "L623" vs "L1362" vs superseded section "L527"): readers cited different refs (`498e7ca` pre-merge, `371e754` post-merge, issue text at eb86404-era). No substantive disagreement.

7. **Does anything conflict with EXP_014 = H14?** No reader adjudicated this directly. Checked: EXP_014 (experiment, PR #53 branch) and H14 (hypothesis, PR #10 branch) are in different namespaces and different branches — no collision, but both are unregistered and "14" now means two unrelated things in prose. Register both; no renumber needed.

## ACTION PLAN

Merged history stays untouched; main needs zero renumbering; the branch renumber is already done. Ordered, assignable:

1. **[TC — human, blocking everything else] Ratification comment on issue #52.** One comment: (a) confirm or deny the 2026-07-26 delegation behind the #37 ruling; (b) accept/amend D1–D8, S1–S2 of the 07-30 comment; (c) explicitly ratify or reverse merge `371e754`'s two choices (main-wins + H12→H14; retain-without-relabel of the H11-series record — this audit recommends ratifying both); (d) rule on §10.7b (H13 vs H11b better-posedness — default: carry the coverage-hole note flat, no ranking) and the two §12 items. Then close #52.

2. **[Any agent, one PR to main] Create `_STAGE2_JSPACE/REGISTER.md`** — the authoritative forward-looking register. Contents: (a) hypothesis table H0–H14 exactly as in Finding 1, including retired bindings with their git hashes (branch spec `aea658f`, so no separate archive file is needed — cite the hash instead of adding a document); (b) experiment table covering EXP_010a (superseded→EXP_012-PYTHIA), EXP_010b/c/c-2/c-3 (spec of record `3ee3126`)/c-3b/c-4 (PR #10)/c-PERM (executed control of record)/c-ROBUST/c-VARIANTS/010d, PERM_TEST_EXP010c (superseded duplicate, executed 07-24, see erratum), EXP_011/011m/013/013m (PLANNED), EXP_012 (= H7 cross-model census only; suffixed IDs distinct; unqualified use deprecated), EXP_012-PYTHIA, EXP_012m, EXP_014 (provisional, PR #53); (c) an ERRATA section recording: the permutation-control double registration/run with all four commit hashes and the concordant outcomes; the EXP_010c-3 double pre-registration (pointer to #52); the #37 "H11a vacated" correction; next free IDs (H15; no EXP number reserved). State the rule: no new `H\d+` or `EXP_\S+` in any spec without a REGISTER.md row in the same commit.

3. **[Same PR] Two errata edits:** add a SUPERSEDED header to `PERM_TEST_EXP010c_SPEC.md` naming `EXP_010c_PERM_SPEC.md` as the registration of record; append a dated note to `RESULTS_EXP010C.md` restoring the fact of the 07-24 run (commits `97aeb20`/`e734ba0`, seed 2026, concordant with the 07-25 run) deleted at `359c622`.

4. **[Same PR] Governance-doc corrections:** `SKILL.md` — replace "claimed three separate ways" with D8's proposed two-claims-plus-one-deferral wording, and touch up the `leave`-example H11a/H11b mention.

5. **[Any agent, board dispatch] Post a correcting comment on discussion #37:** H11a → EXP_010d (not vacated); register H14 → EXP_010c-4 census; register the H10-series, all live EXP IDs, and EXP_014; note that REGISTER.md is now the file-based authority and the thread is its mirror for branch-blind sessions.

6. **[After step 1] Merge PR #10** (after a review pass over the 9 non-artifact files at `371e754`, which no reviewer has seen, and after updating the PR title/body from the retired H11-series to H12/H13/H14). Then add EXP_010c-4/H14 rows to REGISTER.md as MERGED.

7. **[Any agent] Add the merge guard:** CI check that (a) any new H/EXP token in a diff appears in REGISTER.md, and (b) every experiment tier with committed artifacts has a committed run log (closes #52 §6/S2 permanently).

8. **[TC + one agent] Open the parked scientific item as its own issue:** H13's "sharp edge at j=21" (two cells, 17/25 and 23/25) vs the census's eleven j=21 arms averaging 6.6/25 — it must be adjudicated before the J-lens phase consumes "extraction 21 is the only reliable depth" as a prior, and it currently lives only inside a comment on #52.

9. **[Housekeeping] Close #6, #11, #12, #13, #14** with pointers to their merged deliverables, recording on #6 the #10-vs-#33 disposition, so the tracker stops misreporting programme state.