# Runbook — Phase 0: The Instrument (J-lens)

**For:** an operator session executing on this machine. Read fully before acting.
**Plan:** `STAGE2_PLAN.md` (this folder) — Phase 0. **Do not** start Phase 2/3 work.
**Working dir:** `C:\Users\Fab2\Desktop\AI\_learn\_fold\03_MECHINIPHYLUM\_STAGE2_JSPACE`

## Ground rules

1. All new files live under this folder. The ATR repo
   (`..\_LAB_NOTEBOOKS\lucier-repo`) is **READ-ONLY** — import from it, never write
   to it, never `git` in it.
2. This folder is inside the private `fold` git repo. Commit + push after each
   numbered step completes (`git -C ..\..\ add/commit/push` style, or cd to fold
   root). Code, markdown, JSON, small PNGs: commit. Model checkpoints, lens `.pt`
   artifacts, cloned repos: gitignored (step 0 sets this up).
3. **Python:** experiments run on 3.12 (`python`) — it has torch + transformer-lens +
   transformers. 3.11 has the viz stack. Step 0 unifies by installing matplotlib
   into 3.12. If a dependency conflict appears, STOP that step, document it in
   RESULTS_PHASE0.md, and continue with other steps.
4. Disk is tight (~94%). Do not delete anything to make room — document and stop if
   space blocks you. Never touch the HF model cache or any `*.pt` under the ATR repo.
5. Deviations are fine if minimal and documented in RESULTS_PHASE0.md (the Stage 1
   pattern: what differed, why, effect on results).

## Steps

### P0-0 — Scaffold (10 min)
```
mkdir instrument experiments
```
Create `.gitignore` in this folder:
```
instrument/jacobian-lens/
artifacts/
*.pt
*.ckpt
__pycache__/
```
Install viz into 3.12: `python -m pip install matplotlib` (record version).
Create `RESULTS_PHASE0.md` with a header + per-step sections as you go.

### P0-1 — Clone and install the reference implementation (15 min)
```
git clone https://github.com/anthropics/jacobian-lens instrument/jacobian-lens
cd instrument/jacobian-lens && git rev-parse HEAD   # RECORD this commit — upstream is unmaintained; we pin it
python -m pip install -e .
```
Read `walkthrough.ipynb` and the `jlens/fitting.py` docstring before running
anything. Record in RESULTS_PHASE0.md: pinned commit, dependency list installed,
anything version-conflicting.

### P0-2 — Fit the lens for GPT-2 Small
1. **Measure before committing to a budget:** fit on 10 prompts first (from the
   repo's `data/` prompt sets; 128-token sequences), time it, extrapolate, and
   RECORD the per-prompt cost. The estimator's backward-pass count per prompt is
   not documented outside `jlens/fitting.py` — read it, state what it actually does.
2. Fit at **100 prompts** with `checkpoint_path` set (resumable), save to
   `artifacts/jlens_gpt2_small_100.pt`. If per-prompt cost × 1000 is under ~6 h,
   also fit the **1000-prompt** lens (`artifacts/jlens_gpt2_small_1000.pt`) — use
   `JacobianLens.merge()` over slices if that parallelises better on 16 cores.
3. Load via `transformers.AutoModelForCausalLM.from_pretrained("gpt2")` — the HF
   cache already has it. `jlens.from_hf(hf, tok)`.
4. Fit Pythia-160m (`EleutherAI/pythia-160m`) at 100 prompts if time allows —
   OPTIONAL; gpt2-small is the gate-critical lens.

### P0-3 — Validation gate (the step that matters)
On gpt2-small, compare **J-lens readouts vs logit-lens readouts** at layers
{2, 5, 8, 11}, last position and second-to-last, on:
- `"Fact: The currency used in the country shaped like a boot is"`
- `"The color of the planet fourth from the sun is"`
- `"The capital of the country where the Eiffel Tower stands is"`
- 3 prompts of your choosing from `prompt_library.py` categories (record them)

Logit lens = `ln_final → W_U` decode of the same residuals (the ATR repo's
`atr_engine.get_top_tokens` shows the exact convention — import or replicate).

**Gate criteria (qualitative, record a side-by-side table):**
- At mid layers (2–8), J-lens top-5 tokens are more interpretable / task-relevant
  than logit-lens top-5 on the majority of test prompts.
- Per-layer progression is sane (early ≈ input-ish, late ≈ output-ish).
- If 100-prompt and 1000-prompt lenses both exist: readouts broadly agree
  (quality saturation check, paper §9.3).

**Honesty note:** gpt2-small is far weaker than the paper's models — mid-layer
concepts may be faint. The gate is "beats logit lens," not "matches the paper's
examples." **PASS / FAIL / MARGINAL verdict goes in RESULTS_PHASE0.md** — a FAIL
does not stop Phase 1 (which needs no lens) but blocks Phase 2 until reviewed.

### P0-4 — Engine fork with window support (30 min)
Copy `..\_LAB_NOTEBOOKS\lucier-repo\atr_engine.py` → `experiments\atr_engine2.py`.
Header comment: `Forked from ATR repo atr_engine.py @ main for Stage 2. Upstream
frozen; do not sync.` Add window parameters to the run functions:
`inject_layer` (default 0) and `extract_layer` (default n_layers-1) — hook names
`blocks.{inject_layer}.hook_resid_pre` and `blocks.{extract_layer}.hook_resid_post`.
Preserve the gated (`run_atr_gated`) behaviour unchanged for default arguments.
**Check:** run 5 iterations, full stack, prompt `"The cat sat on the mat and then"`,
assert the snapshot metrics equal the unforked engine's output for the same seed.

## Deliverable

`RESULTS_PHASE0.md` in this folder: per-step records, fit timings, the pinned
commit, the validation-gate table and verdict, engine-fork check result. Commit and
push. Final line: `PHASE 0 COMPLETE — gate: <PASS|FAIL|MARGINAL>`.
