# EXP_010c-STOPRULE: stopping-rule stability control at the flagship word cells (pre-registered spec)

**Status:** PRE-REGISTERED, committed before any run.
**Created:** 2026-08-02. **Issue:** #71.
**Parent:** `EXP_010c_SPEC.md` / `EXP_010c2_SPEC.md`. Chartered under TC's
in-session direction of 2026-08-02 (delegation record
`sessions/SESSION_2026-08-02_GOVERNANCE.md`, PR #69). No new hypothesis
number is allocated: this is a control with pre-registered readings, the
same shape as EXP_010c-VARIANTS.

---

## 1. Question

The looping procedure stops when a stability rule fires: the loop is
declared settled when the cosine similarity between mean vectors one
iteration apart stays above 0.999 (on a scale of 0 to 1, where 1 means the
direction did not change at all) on three consecutive checks, checked
every 10 iterations, with checks starting only at iteration 100
(`check_start=100`). The choices 100 and 10 are arbitrary. The settle-tier
record (`experiments/exp_010c_windows/RESULTS_EXP010C.md`, 2026-07-25
EXP_010c-3b item 5) showed that at the neighbouring cell I9 (inject at
layer 9, read at layer 21) merely moving `check_start` from 100 to 10
changed the reported terminal word on 5 of 5 tested prompts.

The two flagship word cells have never had this check. Their registered
terminals are:

- A4 (inject at layer 10, read at layer 21): `' until'` on 19 of 25
  prompts, `' forever'` on 5, `' since'` on 1 (`results_full.json`).
- O8 (inject at layer 8, read at layer 21): `' simultaneously'` on 17 of
  25 prompts, `' halfway'` on 8 (`results_scan.json`).

This control asks: are those terminal identities properties of the
trajectory (the loop has genuinely settled and stays there), or properties
of the stopping rule (the state is still moving and the rule decides what
gets reported)?

## 2. Design, common protocol

**Model:** gpt2-medium, offline load via `--model-path`, the same
1,520,013,706-byte checkpoint as the registered runs (SHA-256 digests
verified against the `EXP_010c_PERM_SPEC.md` addendum). Pinned versions of
the registered runs: torch 2.13.0 CPU, transformer_lens 3.5.1.

**Protocol:** identical to the registered full/scan tiers except where a
variant names its one change: seed 42, the registered 25-prompt subset
(`output/prompt_subset.json`), gate threshold 0.999 with patience 3 and
`check_every=10`, `max_iter=1000`, L0 natural-pass seeding, terminal mean
and last vectors saved per (arm, prompt).

**Arms (3):** A0 (window 0 to 23, the full-stack baseline, registered
terminal `'D'` on 25 of 25 prompts), A4 (10 to 21), O8 (8 to 21).

**Runner and engine diff (recorded, minimal, no refactor).** The
registered code path is never silently changed; the no-stop behaviour is a
new flag whose default reproduces the registered path exactly.

- `run_exp010c.py` gains `--check-start N` (override the tier's
  `check_start`; used by the early-check variant only) and `--no-stop`
  with `--checkpoints 120,300,600,1000` (used by the no-stop variant
  only). All three count as variant parameters in the overwrite guard, so
  a run using any of them without `--tag` refuses to start.
- `atr_engine2.run_atr_gated` gains `no_stop=False` and
  `checkpoint_iters=None`. With `no_stop=True` the convergence gate is
  still computed every `check_every` iterations (so the run records the
  iteration at which the gate would first have fired, saved as
  `gate_first_satisfied_iter`), but the loop never breaks on it and always
  runs to `max_iter`; `lock_in_iter` stays `None` and `converged` stays
  `False` because the run never stopped. At each iteration listed in
  `checkpoint_iters` the last-position readout (terminal token, token id,
  probability, logit margin, entropy) is recorded into a `checkpoints`
  dict. Defaults reproduce the registered path exactly.

**Every run passes `--tag`**, so the overwrite guard is satisfied and no
registered artifact is touched.

## 3. Runs, in this order

| Run | Stopping rule | Arms | Tag |
|---|---|---|---|
| 1, anchor | Registered gate exactly (`check_start=100`) | A0, A4, O8 | `stoprule_anchor` |
| 2, early check | `check_start=10` (the settle-tier precedent; earliest reportable lock drops from 120 to 30), all else registered | A0, A4, O8 | `stoprule_early` |
| 3, no stop | Gate disabled (`--no-stop`), run to the 1000-iteration cap, terminal readout recorded at fixed checkpoints 120, 300, 600 and 1000 | A0, A4, O8 | `stoprule_nostop` |

**Reproduction anchor (pre-registered STOP condition).** Run 1 repeats the
registered protocol unchanged in this environment. Before either variant
is interpreted, run 1 must reproduce the registered per-prompt terminals
exactly (comparison field `terminal_token_id`, per prompt, against
`results_full.json` arms A0 and A4 and `results_scan.json` arm O8; the
protocol is deterministic, see EXP_010c-3b item 2a). A0 is the primary
anchor named by issue #71; a mismatch on any arm of run 1 is a STOP
condition: the run stops being interpretable as a stopping-rule
comparison, the mismatch is reported as the result, and the variants are
not read against the registered artifacts. It is reported, not worked
around.

## 4. Pre-registered readings

**Comparison field:** `terminal_token_id`, per prompt (the decoded
`terminal_token` string is recorded alongside). "Match" means equal token
id for the same prompt. The no-stop variant's comparison terminal is the
iteration-1000 readout (the cap); the 120/300/600 checkpoints are recorded
to localise any drift in time and carry no verdict weight of their own.

| Observation | Reading |
|---|---|
| For every one of the 25 prompts, at A4 and at O8, the early-check terminal and the no-stop iteration-1000 terminal both match the registered run's terminal for that prompt | **Reading 1, trajectory:** the flagship terminal identities at A4 and O8 match the registered run's terminals per prompt across both variants, so the flagship identities are properties of the trajectory. |
| Any prompt at A4 or O8 whose terminal under either variant differs from the registered run's terminal for that prompt | **Reading 2, stopping-rule-conditional:** a terminal identity depends on the stopping rule, so the identities are stopping-rule-conditional and every claim naming them inherits that caveat. Reported per arm and per prompt with exact counts and tokens, no adjectives. |

The two readings are exhaustive and mutually exclusive over the A4 and O8
records (any single mismatch fires reading 2). A0's behaviour under the
two variants is recorded flat as an observation in the same tables and
carries no reading of its own.

**Also recorded per run (observations, no thresholds):** lock iterations
(or `gate_first_satisfied_iter` for the no-stop run), unique-terminal
counts and full terminal multisets, logit margins, entropies, and for the
no-stop run the full checkpoint trajectory per prompt.

## 5. Artifacts

Per run: `output/results_<tag>.json` + `output/terminals_<tag>.pt`, plus a
run log `output/<tag>.log` captured with `tee` (R5: artifacts commit
together with their log). Tags as in §3; none collides with a tier name.

## 6. Analysis and reporting

One dated section appended at the end of
`experiments/exp_010c_windows/RESULTS_EXP010C.md`: the answer first, the
anchor outcome, a per-prompt terminal-versus-stopping-rule table for A4
and O8 (registered, early-check, and no-stop at each checkpoint), the
mechanical statement of which pre-registered reading was observed, and any
deviations. Interpretation beyond the mechanical readings is deferred;
the results section stays observations-only.

## 7. Cost

Runs 1 and 2 stop at the gate (locks near 120 and 80 to 90 respectively on
precedent), roughly an hour together. Run 3 always executes 1000
iterations for each of its 75 (arm, prompt) pairs, estimated 6 to 9 hours
CPU at observed per-iteration throughput, shared with a concurrent job on
the same container.
