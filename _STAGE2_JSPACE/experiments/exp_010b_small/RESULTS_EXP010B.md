# RESULTS — EXP_010b: Window grid on GPT-2 Small (H5, coarse)

**Spec:** `_STAGE2_JSPACE/EXP_010b_SPEC.md` (pre-registered, committed before
any run). **Issue:** #16. **Run date:** 2026-07-26, single machine (4-core CPU,
torch 2.13.0+cpu / transformer_lens 3.5.1 / transformers 5.14.1), seed 42.
**Results-location call** recorded in spec §6 (per-experiment register instead
of the runbook's phase-level `RESULTS_PHASE1.md`).

## Reproduction gate: PASS (stated first, per spec §4)

The SB arm (0→11 full stack) reproduces the Stage 1 gated classification
**exactly, 25/25**, on per-prompt (`converged`, `terminal_token`):
11 `Divine` prompts non-converged at the 1000-iter cap with ` Divine` decode;
` prolet`×10, ` till`×3, ` Anarch`×1 all locked at exactly iteration 120 —
Stage 1's values throughout (`gated_results.pt`: same tokens, same locks).
Mechanical check: `check_reproduction_gate.py`, log committed
(`output/small010b_gate_check.log`). The remote-acquired gpt2-small weights
(hf.co route, sha256s in spec §2) and the `atr_engine2` fork are behaviourally
equivalent to Stage 1 on this subset. Sub-stack arms were cleared to run only
after this verdict.

## What ran

6 arms × 25 prompts = 150 gated runs, sequential, one invocation per arm,
committed per arm. Registered protocol throughout: gated (0.999 ×3, every 10
past 100), max_iter=1000, natural L0 prompt-pass seeding, renorm `seed_j`,
terminal mean+last vectors saved per (window, prompt). Prompt subset: the
committed `prompt_subset_small.json` (5 alphabetically-first Stage 1 Divine +
20 round-robin; 11/25 Divine in total under the literal exclusion rule —
spec §3). A non-registered `small_smoke` harness check (2 prompts, 60 iters)
ran between the spec commit and SB; no verdict weight.

Runtimes: SB 2515 s, S1 605 s, S2 606 s, S3 611 s, S4 609 s, S5 4873 s —
total ≈ 2 h 45 min (arm cost tracks how many prompts hit the 1000-iter cap).

## Per-window observations

From `results_small010b.json` + `terminal_characterisation_small010b.json`
(tensor basins = greedy leader clustering of terminal mean vectors at the
0.999 gate threshold; via-tail = one pass through layers j+1..11, decode at
11):

| arm | window | conv | locks | tensor basins (sizes) | decode terminals | via-tail agree |
|---|---|---|---|---|---|---|
| SB | 0→11 | 14/25 | all 120 | 7 (6,5,4,3,3,2,2) | ` Divine`×11, ` prolet`×10, ` till`×3, ` Anarch`×1 | 25/25 (tail empty) |
| S1 | 0→5 | 25/25 | all 120 | 1 (25) | ` the`×25 | 25/25 |
| S2 | 3→8 | 25/25 | all 120 | 25 (all singletons) | 21 unique: fragments/punct (`er`×3, ` ,`×2, `ur`×2, `lex`, ` just`, ` how`, ` Animated`, ` )`, `'`, `:`, `pan`, NBSP-run, `,`, `o`, ` Sky`, ` the`, `ison`, `y`, `-`, `]`, `Yes`) | **2/25** |
| S3 | 6→11 | 25/25 | all 120 | 5 (13,5,4,2,1) | `.`×25 | 25/25 (tail empty) |
| S4 | 0→8 | 25/25 | all 120 | 1 (25) | ` the`×25 | 25/25 |
| S5 | 3→11 | **1/25** | D02 @180 | 7 (11,5,5,1,1,1,1) | `<|endoftext|>`×23, `.`×1 (G01, capped), ` to`×1 (D02, converged) | 25/25 (tail empty) |

Cross-prompt terminal cosine (off-diagonal mean / min): SB 0.849 / 0.339;
S1 1.000 / 1.000; S2 0.853 / 0.514; S3 0.997 / 0.984; S4 1.000 / 1.000;
S5 0.950 / 0.869.

Flat notes:

- **The full stack is the only window with the Stage 1 landscape.** Every cut
  destroys the few-shared-semantic basin structure: S1/S4 collapse to a single
  prompt-independent ` the` funnel (the two i=0 sub-stack windows produce the
  *same* funnel at different extract depths, and it survives the via-tail
  decode 25/25); S3 collapses to a single `.` funnel; S2 fragments into 25
  private tensor basins with prompt-dependent fragment/punct decodes; S5
  mostly refuses to converge at all.
- **S2's readout is depth-confounded where S4's is not:** both extract at
  j=8, but S2's 21 direct decodes survive the tail to layer 11 only 2/25
  (tail decodes funnel to `,`/`y`/` to`/…), while S4's ` the` survives 25/25.
  Terminal-identity claims for S2 inherit the established
  logit-lens-at-layer-j caveat; its *many-private-basins* tensor structure is
  readout-independent.
- **S5 (3→11) is a new convergence character**, not seen in any Medium or
  Pythia arm: 24/25 prompts fail the lag-1 gate at the cap, 23 decoding
  `<|endoftext|>` with substantial margins (1.0–1.7).

## Divine prompts under each window (lag_scan reported flat)

Per-arm `lag_scan` means over all 25 prompts (lags 1–8), and the 11 Divine
prompts' behaviour:

| arm | lag_scan mean (1..8) | Divine converged | Divine lag-1 range | Divine lag-2 |
|---|---|---|---|---|
| SB | 0.866, 1.0, 0.866, 1.0, 0.866, 1.0, 0.866, 1.0 | 0/11 | 0.676..0.730 | 1.000 |
| S1 | 1.0 at every lag | 11/11 (` the`) | 1.000 | 1.000 |
| S2 | 0.9999 → 0.9926 (slow monotone decay) | 11/11 (private fragments) | 1.000 | 1.000 |
| S3 | 1.0 at every lag | 11/11 (`.`) | 1.000 | 1.000 |
| S4 | 1.0 at every lag | 11/11 (` the`) | 1.000 | 1.000 |
| S5 | **−0.315, 1.0, −0.315, 1.0, −0.315, 1.0, −0.315, 1.0** | 0/11 | **−0.415..−0.246** | 1.000 |

- **SB:** the Stage 1 period-2 bell reproduces exactly — Divine lag-1 stuck at
  0.68–0.73, lag-2 ≡ 1.000 (F13–F17 signature), on all 11 Divine prompts and
  no others.
- **S1/S2/S3/S4:** period-2 is **broken by every 6-layer cut and the
  front-heavy 9-layer cut** — all 11 Divine prompts converge to fixed points
  (lag-1 = lag-2 = 1.000) and lose any distinct identity: under S1/S3/S4 they
  join the same funnel as everything else; under S2 they fragment like
  everything else.
- **S5 (3→11):** the Divine prompts do not converge — but neither do 13
  non-Divine prompts, and the cycle is **anti-correlated**: lag-1 *negative*
  (−0.42..−0.25 for Divine; arm-wide −0.507..0.073 at the final gate check),
  lag-2 ≡ 1.000. A period-2 limit cycle whose half-cycle states point in
  opposed directions, shared by 24/25 prompts regardless of Stage 1 class,
  decoding `<|endoftext|>` on 23. Flat observation: this is not the Stage 1
  Divine bell surviving the cut (that bell had lag-1 ≈ +0.7 and selected 11
  specific prompts); it is a window-induced oscillation that captures nearly
  the whole subset. lag_scan alternation is exact at every recorded lag
  (3,5,7 ≈ lag-1; 4,6,8 ≈ 1.0) in both SB and S5.

## Energy record (spec §5)

Ratio of the seed_j re-injection norm to the natural `resid_pre` norm at the
injection layer (per arm, over 25 prompts):

| arm | i | mean | range |
|---|---|---|---|
| SB | 0 | 73.0× | 56.2–88.4× |
| S1 | 0 | 153.7× | 128.2–168.5× |
| S4 | 0 | 160.3× | 133.9–175.5× |
| S2 | 3 | 1.22× | 1.22–1.23× |
| S3 | 6 | 0.48× | 0.41–0.69× |
| S5 | 3 | 0.56× | 0.48–0.81× |

The i=0 arms re-inject at 56–175× natural embedding-level energy — the
**Control B caveat** (EXP_010c-VARIANTS) applies verbatim: i=0 observations
under seed_j are energy-convention-dependent until a natural_i variant runs.
SB carries the caveat *and* remains the correct gate arm (Stage 1 itself is
seed_j at i=0). The mid/late arms sit at ≈0.5–1.2× — S2, S3 and S5
observations are unlikely to be energy artifacts on this axis.

## H5 (coarse) verdict — mechanical, per spec §1 rule

**H5-coarse: SUPPORTED.** Every one of the five sub-stack windows differs from
the 0→11 baseline on multiple pre-stated fields:

| field | SB | S1 | S2 | S3 | S4 | S5 |
|---|---|---|---|---|---|---|
| converged fraction | 14/25 | 25/25 | 25/25 | 25/25 | 25/25 | 1/25 |
| tensor basins | 7 | 1 | 25 | 5 | 1 | 7 |
| decode class | words (semantic-adjacent) | function word | fragments/punct | punct | function word | special token |
| prompt-dependent? | yes | no | yes | no | no | no (23/25 same) |

Qualitative window-dependence is present in every direction the runbook asked
about: fewer basins (S1/S3/S4), more basins (S2), junk vs the baseline's
words (all), different convergence character (S5). **No sub-stack window
reproduces the Stage 1 semantic landscape — the five basins survive no cut
tested; the full 12-layer stack is, at this grid resolution, the only window
that produces them.** "Richest window" reading (pre-registered question):
by prompt-dependence of terminals the richest sub-stack window is **S2
(3→8)** (25 private basins, 21 unique decodes — with the via-tail readout
caveat above); by dynamical structure it is **S5 (3→11)** (the only window
preserving *any* non-convergent period-2 behaviour, albeit a different,
anti-correlated, near-universal one). Neither matches the baseline's
few-shared-semantic character, so the candidate-workspace-band designation
transfers to Phase 2 with that explicit qualification. Interpretation beyond
this table is fenced to the session record.

## Deviations

1. `small_smoke` harness tier ran after the spec commit, before SB
   (registered as non-registered validation in spec §6; artifacts committed).
2. None otherwise: registered protocol, subset, windows, seed and renorm ran
   as specced; per-arm invocation + merge was pre-declared in spec §6.

## Artifacts

- `../exp_010c_windows/output/results_small010b{,_SB,_S1..S5}.json` — per-run records
- `../exp_010c_windows/output/terminals_small010b{,_SB,_S1..S5}.pt` — terminal
  mean+last vectors per (window, prompt), for the EXP_013 J-lens re-decode
- `../exp_010c_windows/output/terminal_characterisation_small010b.json`
- `../exp_010c_windows/output/natural_resid_norms_small010b_*.json`,
  `small010b_*.log`, `small010b_gate_check.log`
- `../exp_010c_windows/output/prompt_subset_small.json` — execution authority
- `check_reproduction_gate.py` (this directory)

**EXP_010b COMPLETE — reproduction gate: PASS · H5(coarse): SUPPORTED.**
