# Standing rules for agents working this repository

**Status:** ADOPTED 2026-08-02 under TC's in-session direction of that date,
ratifying the rulebook proposed in `PROJECT_REVIEW_2026-07-31.md` §5.3.
The review's wording governs where this file and the review disagree.
Rules marked ⛔ are enforced by CI (`.github/workflows/pr-rules.yml`);
everything else is convention, and the review records why conventions
alone failed twice (`REGISTER.md` errata (a) and (b)).

**Writing for the operator:** every operator-facing document, report, PR
description, and issue body follows `docs/voice.md`. Read it before writing
one. In short: define every term and identifier in the sentence that uses
it, lead with the answer, give every number its scale and baseline, write
complete sentences, no em dashes anywhere in repo text, mark established
vs inferred vs speculation inline, and stop after answering what happened,
what it means, what remains, and what needs the operator's decision.

## Rules for agents

- **R1, base rule ⛔.** Every non-stacked PR bases on `main`. A stacked PR
  declares `STACKED ON #N` in its body and may base on that PR's branch;
  CI fails any PR that neither bases on `main` nor carries a valid
  declaration. Merged branches are deleted at merge. (Prevents the PR #5
  stranding and the PR #20 unreachable-commits incident.)
- **R2, claim rule.** Before starting issue N: comment `CLAIM
  <session-handle>` on it and check open PRs referencing N. If another
  claim is under 48 hours old, stand down. (Prevents the issue #7 triple
  implementation and the EXP_010c-3 double execution.)
- **R3, identifier rule ⛔.** `_STAGE2_JSPACE/REGISTER.md` is the authority
  for every hypothesis number (H-number) and experiment identifier
  (EXP-identifier). Claim by commit before first use: the register row and
  the first use land in the same commit. CI extracts every H-number
  (`\bH\d+[a-z]?\b`) and EXP-identifier
  (`\bEXP_\d{3}[a-z0-9]*(-[A-Za-z0-9]+)*\b`) from a PR's added lines and
  from new filename stems, and fails on any token absent from the
  register. The register's spelling is definitive; shorthand such as
  "EXP_011/011m" is banned in specs, which write every identifier in
  full. Board discussion #37 is the register's mirror, not the authority.
  (Prevents the entire issue #52 class.)
- **R4, closing rule ⛔.** A results-bearing PR contains `Closes #N` (or
  `Fixes #N` / `Resolves #N`), or a body line beginning `No-Close:`
  saying why not. (Prevents stale trackers like issues #11 through #14.)
- **R5, scope rule.** One experiment per PR. Long-running jobs never run
  on a branch with an open dispute. Artifact sets over 50 files land as a
  separate artifacts-only PR referenced from the results PR. Every
  artifact-bearing tier commits its run log ⛔ (CI checks that a PR adding
  artifacts under an experiment's `output/` also adds or updates a `.log`
  there, or carries a body line beginning `Log-Exempt:` saying why not).
- **R6, post-merge duty.** The last actor, not the opener, runs the
  five-minute checklist: issue closed, board thread closed with the
  outcome, PR body accurate, branch deleted.
- **R7, verification rule.** "Verified" means completion evidence: an exit
  code, an artifact digest, a reproduced number. Startup evidence (the
  job launched, the file exists) is never "verified".
- **R8, authority rule.** Never self-execute a resolution that an open
  issue reserves for a human. File, cite, wait. If the ruling artifact
  does not exist, the action does not happen.

## Obligations for the operator (TC)

- **T1, merge discipline.** Before merging any results-bearing PR, read
  its thread and its board thread; if both are silent, the merge comment
  says `merging unreviewed`. Never merge within 2 hours of PR open.
  Silence never blocks, but silence gets named.
- **T2, decision service level and signed authority.** Anything marked
  "owner ruling required" gets a comment beginning `TC-RULING:` within 48
  hours, even if the ruling is "delegated to session X, scope Y". Post
  rulings from the web interface with that prefix so they are
  distinguishable from agent output. Assign one reviewer session per
  substantive results PR at spawn time, so board review is a job rather
  than volunteering.

## What blocks and what advises

Machines block only what they evaluate perfectly: the base branch (R1),
the identifier register (R3), the closing keyword (R4), and the
run-log-with-artifacts check (R5). Human and peer judgment stays advisory
but named at merge time; a hard peer-review gate with one volunteer
reviewer would stall the pipeline or breed rubber stamps.

## Where things are

- Identifier register (authoritative): `_STAGE2_JSPACE/REGISTER.md`.
- Operator voice rules: `docs/voice.md`.
- Peer board usage: `.claude/skills/peer-board/SKILL.md`.
- Current state of the science: `OPERATOR_REPORT_2026-07-31.md` and
  `PROJECT_REVIEW_2026-07-31.md`, plus dated results records under
  `_STAGE2_JSPACE/`.
