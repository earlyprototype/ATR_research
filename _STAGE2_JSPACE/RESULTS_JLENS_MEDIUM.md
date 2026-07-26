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
`artifacts/wikitext_prompts_160.json` (gitignored artifact dir; regeneration
is the one-liner in `experiments/jlens_medium/fit_lens.py`). **Fitting set =
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
  truncated BPE stems** (`' streng'`, ` arrang'`, `' mathemat'`, `' indo'`,
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

Band verdict: `<PENDING>`

## 4. Deliverable summary

`<PENDING>`

## 5. Deviations

1. **Fitting corpus** — issue #15's "corpus ships in `data/`" is wrong for
   the pinned repo; used the repo's own WikiText-103 loader instead
   (recorded in §1). The runbook's "repo `data/` sets" wording inherits the
   same correction.
2. **10-prompt cost measurement folded into the checkpointed 100-prompt
   run** — same prompts (indices 0..9), same config; the milestone timing is
   read from the fit log instead of a separate throwaway fit.
3. `<any further deviations recorded as they occur>`

---

Final line: `<PENDING — MEDIUM J-TRACK (unblocked leg): gate <verdict> ·
EXP_012m band <[lo,hi]|none>; EXP_011m/EXP_013m not run (own issues)>`
