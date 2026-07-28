# RESULTS — Medium J-lens track, unblocked leg (issue #15)

**Runbook:** `RUNBOOK_JLENS_MEDIUM.md` §2 (instrument + validation gate) and §4
(EXP_012m band census). **Register:** reporting register; commit after each
step; deviations documented in §5.
**Scope note:** EXP_011m (terminal projection) and EXP_013m (J-corrected
readout) are NOT run here — per the runbook and issue #15 they run only after
this gate verdict reports, under their own issues (opened at close of this
work, quoting the verdict below).

---

## 1. Instrument (P0-1/P0-2, Medium edition)

**Reference implementation:** `anthropics/jacobian-lens`, cloned to
`instrument/jacobian-lens/` (gitignored per P0-0 conventions; ignore entries
added to `_STAGE2_JSPACE/.gitignore` in this change).
**Pinned commit: `581d398613e5602a5af361e1c34d3a92ea82ba8e`** ("Initial
release" — the only commit; upstream unmaintained). Installed editable
(`pip install -e .`), deps already satisfied by the pinned stack
(torch 2.13.0+cpu, transformers 5.14.1, numpy 2.4.6, Python 3.11.15).

**What the estimator actually does** (read from `jlens/fitting.py` and
`walkthrough.ipynb` before any run, as P0-1 requires):

- `J_l = E[ sum_{p' ∈ valid targets} ∂h_target[p'] / ∂h_l[p] ]`, the
  expectation over prompts and source positions `p`; cotangents are one-hot at
  *every valid target position at once*, so the per-position gradient is the
  sum over later targets, then averaged over source positions (the paper's
  reduction; a strict per-position estimator would differ slightly).
- Cost: **one forward pass + `ceil(d_model/dim_batch)` backward passes per
  prompt** (the prompt replicated `dim_batch` times; each backward fills
  `dim_batch` rows of every `J_l` simultaneously via `torch.autograd.grad`
  against all source layers at once).
- Positions 0..15 are excluded (attention sinks, `SKIP_FIRST_N_POSITIONS=16`)
  and the final position (no next-token target) is excluded.
- `target_layer` defaults to the final block → 23 for gpt2-medium, so the
  lens is fitted at source layers **0..22**. Throughout this record "layer l"
  means the residual at the **output of block l** (transformer_lens
  `blocks.l.hook_resid_post` ≡ `resid_pre(l+1)`). The ATR grid's
  "inject i / extract j" convention is `resid_pre(i)` / `resid_post(j)`, so
  ATR extract-21 reads the same tensor as lens layer 21, and ATR inject-i
  splices in at lens layer i−1's output.
- `lens.apply(..., use_jacobian=False)` decodes the *identical* residual
  through the identical `ln_f → W_U` path minus the `J_l` transport — the
  logit-lens baseline used everywhere below (same convention as
  `atr_engine.get_top_tokens`).

**Model:** gpt2-medium loaded offline from the recorded local acquisition
(`--model-path` route, RESULTS_EXP010C §Model acquisition) via
`transformers.AutoModelForCausalLM`; wrapped by `jlens.from_hf`, which
auto-detects the GPT-2 layout (`transformer`/`h`/`ln_f`/`wte`/`lm_head`).
Verified `n_layers=24, d_model=1024`.

**Fitting corpus — recorded deviation from the issue text:** issue #15 says
"the fitting corpus ships in the jacobian-lens repo's `data/`". It does not:
`data/` ships *evaluation* prompt sets only (six `lens-eval-*` distributions
+ replication sets), and fitting on those would contaminate the validation
gate. The repo's actual fitting corpus is WikiText-103 via its own loader
`jlens.examples.load_wikitext_prompts` (the paper's released lenses are named
`…/Salesforce-wikitext/…_n1000.pt`). Used here: the first 160 WikiText-103
train records with ≥600 chars, in stream order (deterministic), frozen to
`artifacts/wikitext_prompts_160.json` (gitignored artifact dir; deterministic
regeneration: `python3 experiments/jlens_medium/fit_lens.py --refresh-prompts`). **Fitting set =
records 0..99**, `max_seq_len=128` (the paper's sequence length).

**Cost measurement (P0-2 step 1):** single-prompt probe (128 tokens, 111
valid positions, all 23 source layers, 4 CPU threads):

| dim_batch | backward passes | wall/prompt |
|---|---|---|
| 8 | 128 | **333 s** |
| 32 | 32 | 380 s |

`dim_batch=8` selected. Extrapolation: 10 prompts ≈ 55 min; **100 prompts ≈
9.3 h**; 1000 prompts ≈ 3.9 days. 1000-prompt scaling is therefore run only
if the gate demands it (runbook §2.1); the paper's own §9.3 saturation result
says ~100 is usable. 10-prompt measured milestone recorded below from the
fit log (the fit is checkpointed per prompt and was launched as one resumable
100-prompt run rather than a throwaway 10-prompt run — deviation §5).

- 10-prompt milestone (measured): **46 m 08 s wall** for prompts 1–10
  (incl. 1 m 45 s model-load/setup), per-prompt 250–279 s, mean ≈ 266 s —
  all prompts seq_len=128, n_valid=111. Extrapolation: 100 prompts ≈
  **7.6 h**; 1000 ≈ 3.1 days (fits the runbook's "only if the gate demands
  it" branch, not the default). Convergence diagnostic `max_d_mean` falling
  ~1/n as documented (0.58 → 0.08 by prompt 11).
- 100-prompt fit wall time (measured): **25,456 s ≈ 7 h 04 m** (100/100
  prompts processed, none skipped; 97 at seq_len=128/n_valid=111, three
  shorter records at 117–125 tokens). Convergence diagnostic at prompt 100:
  `max_d_mean = 9.15e-03` (running-mean relative shift, falling ~1/n
  throughout — no heavy-tailed outlier prompts; per-prompt
  `max‖J‖/√d` stayed in 1.27–1.97).
- Checkpoint path: `artifacts/jlens_gpt2_medium_100.ckpt.pt` (gitignored);
  final lens `artifacts/jlens_gpt2_medium_100.pt` (fp16, ~48 MB).
- Fitting config: `experiments/jlens_medium/fit_lens.py` (committed).

## 2. Validation gate (P0-3, Medium edition) — protocol

Committed before the gate ran. Script: `experiments/jlens_medium/run_gate.py`.

- Layers {2, 5, 8, 10, 12, 15, 18, 21} — spans the stack and includes the
  ATR-map cells 8, 10 (whole-word injection islands), 15, 21 (extraction).
- Positions −1 and −2; top-5 both lenses + model final top-5.
- Prompts: the three RUNBOOK_PHASE0 P0-3 multi-hop prompts + three recorded
  picks from the restored `prompt_library.py` (lucier repo main
  `49592a7365c77dc63ad7eda0738e04880eac4837`): **A03_neuro, B01_napoleon,
  C01_jack_jill** (three registers, humanly checkable continuations).
- Criteria (RUNBOOK_PHASE0): at mid layers the J-lens top-5 is more
  interpretable/task-relevant than logit-lens top-5 on a majority of prompts;
  per-layer progression sane. Honesty note applies at 345M: the gate is
  "beats logit lens", not "matches the paper"; **FAIL is evidence for
  EXP_012m and is not retried into passing.**

**Result** (full side-by-side table: `experiments/jlens_medium/gate_table.md`
+ `gate_results.json`; committed unedited, first run, no retries):

- **Content positions (−2, the token carrying the prompt's key entity):
  J-lens more interpretable than logit-lens at every tested mid layer on
  6/6 prompts.** The lens reads the current token's verbalizable
  neighbourhood — `' boot'→' boots'/' Boot'/' booted'`,
  `' sun'→' solar'/' celestial'/' lunar'`, `' army'→' cavalry'/' generals'/
  ' armies'`, `' hill'→' slope'/' uphill'` — where the logit lens at the same
  layers emits BPE-continuation fragments (`'strap'`, `'stra'`, `'lit'`,
  `'side'`, `'top'`). Per-layer progression is sane: token-identity early,
  semantic neighbourhood mid, relational/output-ish late (e.g. ' hill' →
  ' slopes' → ' overlooking'/' toward' at L21).
- **Prediction positions (−1): J-lens mid layers 5–12 are dominated by
  truncated BPE stems** (`' streng'`, `' arrang'`, `' mathemat'`, `' indo'`,
  `' cryst'` — recurring across unrelated prompts, i.e. a lens artifact
  signature, not content) while the logit lens shows clean but generic
  function words (`' not'`, `' currently'`, `' often'`). **From ~L15 the
  J-lens becomes task-relevant and often leads the logit lens**: eiffel L15
  `' town'/' cities'/' city'` and L18 `' Budapest'/' Islamabad'/
  ' Constantinople'` (the *category* capital-city before the instance;
  logit-lens L15 has `' also'/' now'/' often'`); mars L15
  `' colors'/' rainbow'/' hue'` vs logit-lens `' often'/' not'`; boot L18–21
  `' counterfeit'/' nicknamed'/' coined'`; napoleon L21
  `' troops'/' cavalry'/' infantry'` (composition words; the model's actual
  output is numbers, which the *logit lens* tracks better from L15).
- **Multi-hop intermediates specifically:** weak. `Italy` never surfaces on
  the boot prompt (any layer, either lens); `France` surfaces at J-lens L21
  (#3) behind `' Paris'` (#1); `Mars` never surfaces on the mars prompt,
  though its *color* semantics do (L15–21). At 345M the lens transports
  semantic category more than it recovers the latent entity.

**Verdict: MARGINAL.** The gate's literal criterion — "at mid layers,
J-lens top-5 more interpretable/task-relevant than logit-lens on the
majority of test prompts" — passes decisively at content positions and
fails at prediction positions for layers ≲12, passing there only from
~L15. Not silently retried; recorded both ways per the runbook. Reading
forward: whatever coherent J-readable structure Medium has sits in the
upper-mid stack (~15–21), not across the naive 10–21 band's lower half —
EXP_012m below measures exactly this.

## 3. EXP_012m — band census — protocol

Committed before the census ran. Script:
`experiments/jlens_medium/run_census.py`; held-out derivation:
`derive_heldout.py` (committed with its output `heldout_50.json`).

- **Held-out set (50 prompts):** round-robin positions 51..100 of the Stage 1
  125-prompt library under the exact registered EXP_010c ordering
  (`derive_prompts.select_subset(100)[50:100]`) — deterministically disjoint
  from the registered ATR subset (positions 1..25), from subset B
  (26..50), and from the WikiText fitting set; all three disjointnesses
  asserted mechanically in `derive_heldout.py` (run log: all OK).
- Per layer 0..22 at position −1: J-lens/logit-lens top-5, agreement with
  final layer, median rank of the model's final top-1 in each lens, J-better
  fraction, word-like top-1 fraction; plus a per-layer multi-hop
  intermediate-rank curve (93 items, `lens-eval-multihop.json`, hit@10/@1).
- **Pre-registered band rule** (fixed before any census output existed):
  layer l is lens-dominant iff (i) median rank_J(final top-1) <
  median rank_LL(final top-1), (ii) J-better fraction ≥ 0.6, (iii) J-lens
  top-1 agreement ≥ 0.10. Band = maximal contiguous run ≥ 3 of
  lens-dominant layers; none → "no coherent band". Islands reported flat.
- Comparison targets (stated flat, no relationship asserted): the naive
  40–90% depth mapping **10–21** EXP_010c assumed, and the #6/#33 in-fill
  map's finding that whole-word cells sit at **inject i ∈ {8, 10} → extract
  21 as isolated single-layer islands** (i=9 fails, i=7/11 fail; extraction
  rungs 15/17/19 unflagged).

**Result** (full table: `experiments/jlens_medium/census_table.md`; per-layer
aggregates + all 50×23 top-5 readouts in `census_results.json`):

| L (sample) | J agree-final | LL agree-final | J med-rank | LL med-rank | J-better | mh J@10 | mh LL@10 |
|---|---|---|---|---|---|---|---|
| 2 | 0.02 | 0.12 | 2595 | 64 | 0.14 | 0.00 | 0.01 |
| 8 | 0.02 | 0.04 | 1240 | 36 | 0.14 | 0.00 | 0.00 |
| 9 | 0.02 | 0.10 | 923 | 28 | 0.16 | 0.00 | 0.00 |
| 10 | 0.02 | 0.10 | 1084 | 24 | 0.12 | 0.00 | 0.00 |
| 15 | 0.04 | 0.20 | 476 | 3 | 0.06 | 0.00 | 0.00 |
| 18 | 0.08 | 0.36 | 167 | 2 | 0.02 | 0.00 | 0.00 |
| 21 | 0.20 | 0.70 | 11 | 1 | 0.00 | 0.00 | 0.00 |
| 22 | 0.32 | 0.88 | 3 | 1 | 0.02 | 0.00 | 0.00 |

- **Lens-dominant layers under the pre-registered rule: none. Band verdict:
  NO COHERENT BAND.** No layer satisfies even criterion (i): the logit
  lens's median rank of the model's final top-1 falls from 119 to 1 across the stack (non-monotone below L5), while the J-lens's stays in the 600–4000 range through the
  mid-stack and only reaches 11 at L21 / 3 at L22. J-better fraction peaks
  at 0.50 at L0 and is ≤0.20 everywhere else.
- **Multi-hop intermediates (93 items): a genuine null on both lenses.**
  hit@10 ≤ 0.01 at every layer for the logit lens (max 0.97% at L2) and
  exactly 0 at every layer for the J-lens; spot-checked best ranks are
  ~400–1000 (logit) vs ~4400–5500 (J-lens) for items like `Brazil`/`Mars`.
  At 345M there is no layer at which either readout surfaces latent
  multi-hop intermediates.
- **Qualitative held-out readouts mirror the gate:** at the prediction
  position the J-lens mid-stack (≲L14) emits the same prompt-independent
  truncated stems seen in the gate (`' mathemat'`, `' horizont'`,
  `' destro'`, `' trave'` recur across unrelated prompts); semantically
  clustered content appears only at L15–22 (A11_ml: `' gradient'/
  ' coefficients'/' vectors'/' equations'` by L21; C11_genesis:
  `' darkness'/' brightness'/' illum'` from L15). The word-like top-1
  fraction (J ≈ 0.9 vs LL ≈ 0.65) is recorded but is inflated by exactly
  those stems and carries no band structure.

**Comparisons (stated flat, numbers only, no relationship asserted):**

1. **vs the naive 40–90% mapping (layers 10–21) EXP_010c assumed:** the
   census finds no J-lens-dominant layer anywhere in 10–21. At the mapping's
   lower edge (L10–12) the J-lens readout is at its least interpretable
   (med-rank ~1000, junk-stem-dominated top-5); such structured J-readouts
   as exist sit at L15–22 — the mapping's upper half — but never beat the
   logit lens on the registered metrics there.
2. **vs the ATR in-fill map (#6/PR #33: whole-word cells at inject
   i ∈ {8, 10} → extract 21 as isolated single-layer islands; i=9 and
   i=7/11 unflagged):** the census shows no island structure at 8/9/10 on
   any recorded J-lens metric — J agree-final 0.02/0.02/0.02, med-rank
   1240/923/1084, J-better 0.14/0.16/0.12 — i.e. layers 8 and 10 are
   indistinguishable from 9 and from their flanks under the lens. The
   census therefore provides **no independent J-lens correlate** of the
   ATR word-cell islands; the two instruments disagree about whether
   anything distinguishes {8,10} from {9}. (Convention note: ATR inject-i
   splices at `resid_pre(i)` = lens layer i−1's output; the non-result is
   unchanged shifting the comparison by one layer — 7/8/9 read
   0.20/0.14/0.16 J-better, no islands either.)

## 4. Deliverable summary

**What ran:** P0-1 clone+pin+install of `anthropics/jacobian-lens`
(581d398); 100-prompt WikiText-103 J-lens fit for gpt2-medium (23 source
layers, target L23, 7 h 04 m CPU, checkpointed per prompt); P0-3 Medium
validation gate (6 prompts × 8 layers × 2 positions, single run); EXP_012m
band census (50 held-out prompts × 23 layers + 93-item multi-hop per-layer
curve, rerun once for the recorded BOS scoring bug — §5.3).

**Headline numbers:** fit ≈ 266 s/prompt (10-prompt milestone 46 m,
100-prompt 7 h 04 m, `max_d_mean` 0.58→0.009); gate MARGINAL (content
positions: J-lens wins 6/6 at mid layers; prediction positions: junk-stem
dominated ≤L12, task-relevant from ~L15); census: zero lens-dominant
layers, LL med-rank 119→1 vs J med-rank stuck ≥600 mid-stack, multi-hop
hit@10 ≤1% (LL) / 0% (J) everywhere.

**Decides:** EXP_012m — Medium has **no coherent J-lens band** under the
pre-registered rule; neither the naive 10–21 mapping nor the ATR
{8,10}→21 islands has a J-lens-side correlate in this census.

**Interpretation (labelled as such):** the fitted lens does transport
verbalizable *semantic-neighbourhood* content (gate, content positions;
L15–22 clusters), so the null is not "the lens does nothing" — it is "no
layer range where the lens reads a workspace better than the trivial
readout, and no multi-hop workspace content findable at all at 345M".
This feeds the runbook §3 first-row reading in the negative: ATR and
J-lens do **not** localise the same band by independent means on Medium.

**Open questions:** whether EXP_011m's subspace-overlap test (which does
not go through readout rankings at all) sees the {8,10} islands; whether
EXP_013m's re-decode of the frozen terminals changes any terminal-identity
claim; whether a 1000-prompt lens moves any census number (not fitted —
the gate did not demand it on the saturation evidence, §1).

## 5. Deviations

1. **Fitting corpus** — issue #15's "corpus ships in `data/`" is wrong for
   the pinned repo; used the repo's own WikiText-103 loader instead
   (recorded in §1). The runbook's "repo `data/` sets" wording inherits the
   same correction.
2. **10-prompt cost measurement folded into the checkpointed 100-prompt
   run** — same prompts (indices 0..9), same config; the milestone timing is
   read from the fit log instead of a separate throwaway fit.
3. **Census multi-hop columns rerun once for a scoring bug** (not a lens
   or verdict change): `from_hf(force_bos=True)` sets `add_bos_token=True`
   on the GPT-2 tokenizer, so the first run's `encode(" word")[0]` scored
   the BOS token (`<|endoftext|>`) instead of the intermediate — mh columns
   read 0.00 for the wrong reason. Fixed by stripping BOS; every non-mh
   number was bit-identical across the two runs; corrected mh maxima are
   0.97% (LL, L2) / 0% (J). Recorded in `run_census.py` at the fix site.
   The band verdict is identical under both runs.
4. **Prompts carry a leading BOS throughout** (fitting and readout) — the
   instrument's documented `force_bos=True` default, kept as-is for
   fidelity to the reference implementation; noted because the ATR engine
   does not prepend BOS.

---

`MEDIUM J-TRACK (unblocked leg, issue #15): gate MARGINAL · EXP_012m band:
none (no coherent band) · EXP_011m: not run (own issue) · EXP_013m: not run
(own issue)`
