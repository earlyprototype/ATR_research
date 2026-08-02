# EXP_010d — Results Record

**Spec:** `../../EXP_010d_SPEC.md` (pre-registered before the Small run).
**Status:** run complete; primary verdict recorded with threshold-robustness sweep.

**Arbiter note, 2026-08-02:** sentences below naming the J-lens re-decode
(EXP_013m) as "the registered arbiter" are historical text and stay as
written; as of 2026-08-02 EXP_011m (subspace overlap) is the primary
arbiter and EXP_013m is demoted to a supporting probe. Ruling and grounds:
`../../REGISTER.md` erratum (e).

---

## 2026-07-24 — Native GPT-2 Small vs windowed Medium, matched 25-prompt subset

**What ran:** `compare_small_basins.py --small-path <local gpt2-small> --tier full
--n-perm 10000`. GPT-2 Small looped natively (0→11), convergence-gated (threshold
0.999, patience 3, check_every 10, check_start 100, max_iter 1000), on the exact
25-prompt subset committed in `output/prompt_subset.json` — the same prompts
EXP_010c used. Small's basin partition was then compared to the committed Medium
terminals: A4 (band 10→21) and A0 (full-stack baseline 0→23), via chance-corrected
Adjusted Rand Index with a 10,000-shuffle permutation null.

**Model route (recorded):** gpt2-small pulled from the same legacy HF S3 mirror the
EXP_010c medium route used (`s3.amazonaws.com/models.huggingface.co/bert/gpt2-*`;
pytorch_model.bin 548,118,077 bytes), loaded offline via the runner's generalised
`_load_gpt2_from_local`. huggingface.co remains blocked by the session network
policy; PyPI is open, so the ML stack (torch 2.13.0, transformer_lens 3.5.1)
installed directly.

**Deviation recorded:** the ML stack was assembled wheel-only
(`--only-binary`, `transformer_lens --no-deps` + core deps) because a transitive
dependency (`transformers-stream-generator`) has no wheel and its source build
fails in this environment. It is a text-streaming helper, unused by
`from_pretrained`; absence does not affect the forward passes.

### Small's own behaviour (the reference geometry)

Small consolidated on **18/25** prompts; the other **7 never locked** — the
`Divine` object again (the Stage 1 non-converging attractor), running to the
1000-iter cap. Terminal decodes:

| Decode | Count | Converged? |
|---|---|---|
| ` prolet` | 13 | yes |
| ` Divine` | 7 | **no** (ran to cap) |
| ` till` | 3 | yes |
| ` Anarch` | 2 | yes |

Tensor-cluster partition at the 0.999 gate: **7 basins** (the 4 decode-types plus
extra singletons from the scattered non-converged `Divine` terminals). This is a
genuine multi-basin geometry — Small is a real reference, not a degenerate one —
so the comparison is answerable (spec §5 STOP condition not triggered).

### Primary comparison (gate threshold 0.999, as pre-registered)

| Pair | ARI | perm p | Basin count |
|---|---|---|---|
| Small ↔ **A4** (band 10→21) | **0.048** | **0.244** | Small 7 vs A4 12 |
| Small ↔ **A0** (baseline 0→23) | **0.200** | **0.0009** | Small 7 vs A0 4 |

### Threshold-sensitivity sweep

A0's four tensor clusters sit at off-diag cosine ~.998, right at the 0.999 gate, so
its partition — and any ARI resting on it — could be an artifact of exactly where
the threshold falls. Re-clustered all three models across thresholds (no model
time; uses the saved `terminals_small_010d.pt` + committed Medium terminals):

| threshold | Small basins | A4: basins / ARI / p | A0: basins / ARI / p |
|---|---|---|---|
| 0.99   | 4 | 6 / 0.011 / 0.436  | 1 / 0.000 / 1.000 |
| 0.995  | 5 | 8 / 0.014 / 0.416  | 3 / 0.169 / 0.022 |
| 0.999  | 7 | 12 / 0.048 / 0.244 | 4 / 0.200 / 0.0009 |
| 0.9995 | 8 | 15 / 0.068 / 0.161 | 6 / 0.219 / 0.0004 |

**Reading of the sweep:** A4 is non-significant at **every** threshold (ARI
0.01–0.07, p ≥ 0.16) — its failure to match Small is not an artifact of where the
gate is set. A0's agreement is significant at the three finer thresholds
(p = 0.022 → 0.0004) and vanishes only at 0.99, where A0 degenerates to a single
cluster and there is no sub-structure left to compare. So the A0↔Small agreement
is **robust**, not a knife-edge of the 0.999 gate: wherever A0's residual
sub-structure is resolvable at all, it partitions like Small.

### Verdicts (mechanical application of spec §5)

- **H11 (geometry recreation): REFUTED (robustly).** The workspace-band loop (A4)
  does **not** reproduce Small's basin partition: ARI 0.048, permutation p = 0.244
  at the registered threshold, and non-significant at every threshold in the sweep.
  It is also *below* the full-stack baseline's agreement. Escaping Medium's `D`
  collapse (EXP_010c) produced a *different* geometry, not Small's: 12 basins
  decoding to `until`/`forever`/`since`, versus Small's
  `prolet`/`till`/`Anarch`/`Divine`.
- **H11a (basin count): REFUTED for A4.** A4's 12 basins overshoot Small's 7 by
  more than the pre-registered k=4; the band window makes *more* structure than
  Small, not the same amount. A0's 4 is within k of Small's 7.

### Observation, stated flat (interpretation labelled below)

The full-stack baseline A0 — which decode-collapses **every** prompt to the single
token `D` — nonetheless partitions the 25 prompts significantly like native Small
at the tensor level (ARI 0.20, p < 0.001), while the band window that escapes the
collapse does not.

**Interpretation (labelled as such, NOT a finding):** one story consistent with
this is that the prompt-level geometry resembling Small is *latent in Medium's full
stack already*, carried in residual sub-structure that the `D` decode masks — and
the band window, rather than recovering Small's geometry, manufactures a new and
different one. The sweep rules out the first-order objection (it is *not* a
knife-edge of the 0.999 gate — the A0↔Small agreement holds across 0.995–0.9995),
but it must still not be over-read: tensor-partition ARI is a proxy for "same
geometry," not a proof of it; the agreement is carried by a minority of prompts
(A0 puts 18/25 in one cluster); and the decode-token layer is the
logit-lens-at-layer-j readout whose mid-stack unreliability EXP_010c already
measured. The J-lens re-decode (EXP_013m) remains the registered arbiter.

### What this does and does not touch

- It does **not** overturn EXP_010c. A4 genuinely escapes the `D` collapse and
  yields richer, prompt-dependent decodes — that stands. What it overturns is the
  stronger reading that "richer than the Medium baseline" equals "recreates Small."
  On the matched partition test, it does not.
- The programme hypothesis — *selective injection loops recreate Small's basin
  geometry* — is **not supported** at the partition level on this subset/protocol.
  Whether it holds under the J-lens readout, other seeds, or other subsets is open;
  those are the registered next controls.

### Caveats (standing)

Single seed; one 25-prompt subset; Small itself converged only 18/25 (the 7
non-converged `Divine` terminals contribute possibly-noisy points to its
partition); tensor-partition agreement is necessary but not sufficient for
same-mechanism; cross-model comparison is partition-level by construction (768 vs
1024 dims). J-lens re-decode (EXP_013m) and the anisotropy-corrected permutation
control remain the arbiters for any semantic or mechanistic claim.

### Artifacts

`output/basin_comparison_full.json` (partitions, ARIs, p-values, sweep),
`output/terminals_small_010d.pt` (Small terminal tensors, for threshold
re-analysis with no model), `output/exp010d_run.log`.

## 2026-07-27 — Rescue provenance note

This record was carried verbatim from unmerged PR #5 (commit `baf33989`,
branch `claude/review-recent-pr-qug5jj`, session of 2026-07-24), which was
stranded on a since-merged base branch. Files ported unchanged: this results
record, `EXP_010d_SPEC.md`, `compare_small_basins.py`, and the three output
artifacts. PR #5's `run_exp010c.py` modification (offline gpt2-small loader
+ the lag_scan fix) was NOT ported — both capabilities landed on `main`
independently (the lag_scan fix via the PR #4 review regeneration; the
gpt2-small loader via the `--model-name` generalisation in PR #42's line) —
so the analysis here reads the committed artifacts and needs no runner
changes. H-numbering: H11 is confirmed to this experiment by the
2026-07-26 identifier-registry ruling (discussion #37).
