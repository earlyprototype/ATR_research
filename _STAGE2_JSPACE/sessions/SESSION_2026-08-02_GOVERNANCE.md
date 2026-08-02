# Session note, 2026-08-02 — governance resolutions under operator delegation

**What this is:** the citable record of the operator direction given in the
working session of 2026-08-02, and of what was executed under it. This
note is the delegation artifact the standing rules require (CLAUDE.md R8:
file, cite, wait; the ruling artifact for today's actions is this note
plus the operator's confirmation described in §3).

## 1. The delegation

In the 2026-08-02 session, the operator (TC) reviewed a state-of-the-
project report covering the merged EXP_015 result, the open decision list
from `OPERATOR_REPORT_2026-07-31.md` §6, and the recommended next steps,
and directed, in their own words, that the session's suggested next steps
proceed and that the session resolve each open decision item along the
lines it had recommended. The decisions so delegated were: the PR #53
disposition, the kill-criteria re-registration, the arbiter swap
(promoting EXP_011m, demoting EXP_013m, re-chartering issues #45 and
#46), and the adoption and enforcement of the standing rules.

## 2. Executed under that delegation (this session)

1. **Standing rules adopted.** `CLAUDE.md` created at the repository root
   with rules R1 to R8 for agents and obligations T1 and T2 for the
   operator, per `PROJECT_REVIEW_2026-07-31.md` §5.3. CI enforcement
   added: `.github/workflows/pr-rules.yml` and
   `.github/scripts/check_pr_rules.py` (R1 base rule, R3 identifier
   register, R4 closing keyword, R5 run-log-with-artifacts; the checker
   ships with parser self-tests that CI runs before the check can block
   anything).
2. **Kill criteria re-registered** in `STAGE2_PLAN.md`, old rule kept
   visible and struck, per review §6.2 item 6.
3. **Arbiter swap executed.** `REGISTER.md` erratum (e) records the
   ruling; EXP_011m and EXP_013m register rows updated; dated pointer
   notes added to `RESULTS_EXP010C.md`, `RESULTS_EXP010D.md`, and
   `RUNBOOK_JLENS_MEDIUM.md`; the runbook's §3 branch table demoted to
   non-authoritative sketch and §3a's target set reframed as a flat
   observation (issue #52 proposals N2 and N3, ratified). Issues #45 and
   #46 re-chartered by comment.
4. **Artifact-guard fixes** (review §6.2 item 4, the category the review
   licensed without ratification): the overwrite guard in `run_exp010c.py`
   now treats a per-arm rerun (`--arms`) as a variant requiring a tag,
   and rejects `--tag`/`--out-suffix` values that collide with tier or
   tier-harness artifact names. The BOS seam is settled and the backwards
   note in `RESULTS_JLENS_MEDIUM.md` corrected with a dated correction:
   the ATR engine DOES prepend BOS (pinned transformer_lens 3.5.1 default
   verified by source inspection of the pinned wheel; engine call sites
   pass raw strings with no override; independent empirical corroboration
   in the EXP_014 harness gate on the PR #53 branch). Consequence: the
   two instruments agree on tokenization, which unblocks EXP_011m.
5. **PR #53 disposition posted** per review §6.3: split into two parts,
   EXP_014 mergeable on its merits, the Medium political material gated
   on provenance, re-review, and a registered spec; interpretive title
   renamed; Stage 1 boundary ruling recorded on the PR.
6. **Compute work chartered, not run** (no model weights in this
   environment; the legacy-mirror acquisition route recorded in
   RESULTS_EXP010C.md §Model acquisition is available to the executing
   session): the stopping-rule stability check at the flagship word cells
   A4 (10→21) and O8 (8→21), and EXP_015's registered follow-up control
   (a natural-strength arm engineered to still produce the `D` readout).
   The regeneration of `permutation_results.json` with its `input_sha256`
   attestation (review §4.4 item 4) is folded into the same charter: it
   needs the gpt2-medium checkpoint, and hand-editing a registered
   artifact was rejected as a substitute.
7. **Housekeeping:** tracker issues #11 to #14 closed with pointers to
   their merged deliverables; merged and superseded remote branches
   deleted after verification; the H13-vs-census adjudication issue
   opened (ruling item 4's outstanding piece).

## 3. What converts this delegation from asserted to verified

This note, like every delegation record before it, is authored from an
agent session. Under T2 the operator makes it verifiable with one comment
from the web interface, prefixed `TC-RULING:`, on the governance PR this
session opens, confirming the 2026-08-02 direction. Until that comment
exists, treat every action in §2 as executed under an asserted, not a
verified, delegation. The same applies retroactively to the 2026-07-31
delegation recorded in `REGISTER.md`; the operator report's decision item
6 asked for the same one-line confirmation on issue #52 and it has not
been posted.

## 4. Deviations

None from the review's specifications, with one scope note: CI R5 checks
the machine-checkable half of the scope rule only (run log accompanies
artifacts); the one-experiment-per-PR and artifacts-only-PR halves remain
convention.
