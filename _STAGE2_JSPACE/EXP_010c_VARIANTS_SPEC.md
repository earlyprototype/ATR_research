# EXP_010c-VARIANTS — Hook-Point and Energy-Normalisation Controls (pre-registered spec)

**Status:** PRE-REGISTERED — committed before any run.
**Created:** 2026-07-25
**Parent:** `EXP_010c_SPEC.md` / `EXP_010c2_SPEC.md`. Executes planned-controls
item 4 of `experiments/exp_010c_windows/RESULTS_EXP010C.md` (hook-point
variants, issue #13) and the energy-normalisation control registered in the
2026-07-23 session notes and listed in issue #6's related open threads
(issue #14). One spec covering both controls, recorded as such per issue #14
("pairs naturally with the hook-point control — a single spec covering both
variants is acceptable if recorded as such"). The two controls are executed
and reported as separate arms sets with separate artifacts and separate
results subsections.

---

## 1. Questions

- **Q-hook (Control A, issue #13):** does the single-terminal behaviour of the
  i=0 arms (A0 0→23 → `D` ×25; O0 0→21 → `','` ×25) reflect layer *position*
  (the sensory front) or the raw-embedding coordinate slot? `resid_pre` at
  layer 0 is the embedding output — a coordinate system no other layer shares —
  so "injection at the sensory front destroys structure" (H10b's supported
  reading) is currently confounded with "injection into the embedding
  coordinate frame destroys structure."
- **Q-hookwiring (Control A sanity):** are `blocks.{i-1}.hook_resid_post` and
  `blocks.{i}.hook_resid_pre` the same residual value in this TransformerLens
  stack, as the library's documentation implies? If not, every hook-point
  comparison in this programme needs re-examination.
- **Q-energy (Control B, issue #14):** are the window effects an
  injection-loudness artifact? Residual-stream norms grow with depth; under
  the registered convention the looped tensor is rescaled to the norm of the
  seed captured at extraction layer j, so it arrives at injection layer i with
  j-scale energy where the model expects i-scale energy. Every window arm
  therefore differs from the natural forward pass in two ways at once: the
  splice and the off-scale energy. This control separates them.

## 2. Design — common protocol

**Model:** gpt2-medium, offline load via `--model-path` (same
1,520,013,706-byte checkpoint as the registered runs, already on disk).

**Protocol:** identical to the registered full/scan tiers — seed 42, the
registered 25-prompt subset (`output/prompt_subset.json`), gated
(cos > 0.999 ×3, checks every 10 past check_start=100), max_iter=1000, L0
natural-pass seeding, terminal mean+last vectors saved per (arm, prompt).
The ONLY variables are the ones named per control below.

**Runner/engine diff (recorded, minimal — no refactor):**
`atr_engine2.run_atr_gated` gains two keyword parameters, defaults reproducing
the registered path exactly:

- `inject_hook_name=None` — when set, the injection hook is installed at the
  given hook name instead of the constructed `blocks.{i}.hook_resid_pre`.
  Used only by the Control A sanity arm.
- `renorm="seed_j"` — the registered convention (rescale the looped tensor to
  the norm of the seed captured at extraction layer j, i.e. `initial_norm`).
  `renorm="natural_i"` rescales instead to the natural `resid_pre` norm at
  injection layer i, measured for the same prompt on the initial (natural,
  un-hooked) forward pass. Used only by Control B.

The result dict additionally records `inject_hook`, `renorm`, `target_norm`,
and `seed_norm_at_j` for every run (metadata only; no protocol change on the
default path). `run_exp010c.py` gains the three Control A arm definitions,
two tiers (`hookpoint`, `energynorm`), and a `--renorm` flag; when
`--renorm natural_i` the runner also measures and saves the natural per-layer
`resid_pre` norms (all 24 layers, per prompt, one natural pass each) to
`output/natural_resid_norms_<suffix>.json` — the reference-norm record
required by issue #14.

## 3. Control A — hook-point variants (issue #13)

**Arms (3 × 25 prompts = 75 runs):**

| Arm | Window | Inject hook | Compares against (registered rows) |
|---|---|---|---|
| I1A0 | 1→23 | `blocks.1.hook_resid_pre` (default construction) | A0 (0→23, full tier): `D` ×25 |
| I1O0 | 1→21 | `blocks.1.hook_resid_pre` (default construction) | O0 (0→21, scan tier): `','` ×25 |
| HP9 | 10→21 | **`blocks.9.hook_resid_post`** (variant) | A4 (10→21, full tier): `' until'` ×19, `' forever'` ×5, `' since'` ×1 |

I1A0/I1O0 move the injection edge off the raw-embedding slot by exactly one
layer while keeping everything else fixed: if the funnels are about the
embedding coordinate frame, i=1 should behave unlike i=0; if they are about
position at the sensory front, i=1 should behave like i=0.

**Pre-registered expectation for HP9 (stated before running):** in
TransformerLens, `blocks.9.hook_resid_post` and `blocks.10.hook_resid_pre`
are the same residual value, so HP9 must be IDENTICAL to the registered A4
record-by-record. A difference means the hook wiring differs from the
documented equivalence and MUST be reported as such, not hidden.

**Artifacts:** per-arm `results_hookpoint_<ARM>.json` +
`terminals_hookpoint_<ARM>.pt` (committed after each arm), merged into
`results_hookpoint.json` + `terminals_hookpoint.pt`, then
`analyze_terminals.py --tier hookpoint --decode-via-tail` →
`terminal_characterisation_hookpoint.json`.

### Pre-registered readings (Control A)

Comparison fields for record-identity: terminal_token, terminal_token_id,
lock_in_iter, n_iters, converged, top_logit_margin, entropy,
final_cos_sim_mean, terminal_prob (the 9 fields used by EXP_010c-ROBUST).
"Whole-word alphabetic" = the decoded token text, ignoring one leading
space, satisfies `str.isalpha()`.

| Observation | Reading |
|---|---|
| I1A0 and I1O0 each produce exactly 1 unique direct-decode terminal across the 25 prompts (the i=0 arms' single-funnel signature) | **Position, not slot** — the i=0 single-terminal behaviour does not require the raw-embedding coordinate frame. Record whether each funnel terminal token equals the corresponding i=0 arm's (`D`, `','`); a different funnel token is still this reading (funnel character preserved) and is stated flat. |
| Either i=1 arm produces ≥3 unique direct-decode terminals of which ≥2 are whole-word alphabetic (the A4/O8 landscape class) | **Embedding-slot effect isolated** — the i=0 funnels reflect the raw-embedding slot's coordinate system rather than layer position; H10b's "sensory splice" reading inherits this caveat. |
| Anything else (e.g. 2 unique terminals, or ≥3 non-word terminals) | **Intermediate** — neither reading claimed; state the exact counts and tokens, no adjectives. |
| HP9 identical to registered A4 on all 9 fields for all 25 records | **Hook equivalence confirmed** — `resid_post(i−1)` ≡ `resid_pre(i)`; hook-point convention immaterial for this stack. |
| HP9 differs from registered A4 on any field of any record | **Hook wiring differs** — reported prominently; all hook-point comparisons in this programme (including this control's i=1 arms) inherit the caveat until the wiring difference is explained. |

## 4. Control B — energy-normalisation variant (issue #14)

**Arms (4 × 25 prompts = 100 runs), all with `renorm="natural_i"`, default
inject hook:**

| Arm | Window | Registered row compared against (seed_j convention) |
|---|---|---|
| A0 | 0→23 | `D` ×25; basins 4; margin 0.52/0.52; via-tail 25/25 |
| A4 | 10→21 | `' until'` ×19, `' forever'` ×5, `' since'` ×1; basins 12; margin 4.20/7.15; via-tail 23/25 |
| O8 | 8→21 | `' simultaneously'` ×17, `' halfway'` ×8; basins 11; margin 2.88/5.10; via-tail 17/25 |
| A1 | 0→11 (funnel arm) | `','` ×22, `'ing'` ×3; basins 1; margin 0.37/1.84; via-tail 0/25 |

**Rescale rule (the one-line variant):** at every loop iteration the looped
tensor is rescaled to `natural_pre_norm_i` — the full-tensor L2 norm of
`blocks.{i}.hook_resid_pre` for the same prompt on the natural (un-hooked)
initial pass — instead of `initial_norm` (the seed's norm at extraction
layer j). For i=0 the natural `resid_pre` norm is the embedding-output norm.
Nothing else changes: same seed state (captured at j), same gate, same
readout. The measured per-layer natural norms for all 25 prompts are saved to
`output/natural_resid_norms_energynorm.json`; each run record carries its
`target_norm` and `seed_norm_at_j`.

**Artifacts:** per-arm `results_energynorm_<ARM>.json` +
`terminals_energynorm_<ARM>.pt` (committed after each arm), merged into
`results_energynorm.json` + `terminals_energynorm.pt`, then
`analyze_terminals.py --tier energynorm --decode-via-tail` →
`terminal_characterisation_energynorm.json`.

### Pre-registered readings (Control B)

Class definitions (mechanical): an arm is **word-structured** if it has ≥2
unique direct-decode terminal types and at most one of the 25 prompts
terminates in a non-whole-word-alphabetic token (registered A4 and O8 both
satisfy this). An arm is a **funnel** if it has ≤2 unique direct-decode
terminal types and the modal terminal covers ≥22 of 25 prompts (registered
A0 25/25 and A1 22/25 both satisfy this).

| Observation | Reading |
|---|---|
| A4 and O8 are word-structured AND A0 and A1 are funnels under `natural_i` | **Not an energy artifact** — the window landscape (word structure in-band, funnels off-band) survives i-scale renormalisation; injection loudness is not driving the effects. Whether the specific terminal types persist (`until`/`forever`/`since`, `simultaneously`/`halfway`, `D`, `','`) is recorded flat either way. |
| Any anchor arm changes class (A4 or O8 not word-structured; A0 or A1 not a funnel) | **Energy convention load-bearing** for the arms that changed — every window observation in EXP_010c/010c-2 inherits the caveat that it is conditional on the j-scale energy convention; stated per-arm, no adjectives. |

**Also recorded per arm, both controls (observations, no thresholds):**
unique-terminal counts, full terminal multisets, token classes, lock-in
iterations, margin μ/max, entropy, via-tail decode + agreement, tensor-basin
counts at the 0.999 threshold, and for Control B the target/seed norm ratios.
Full uncurated inventories stay in the results JSONs; no curated sublist.

## 5. Analysis and reporting

Per control, `analyze_terminals.py --tier <hookpoint|energynorm>
--decode-via-tail --model-path <local>`. One dated section appended to
`RESULTS_EXP010C.md` with the two controls as separate subsections, each
holding the variant-vs-registered observation table and the mechanical
statement of which pre-registered reading obtained. Planned-controls item 4
(hook-point) and the session-registered energy-norm control are marked done
with pointers. Interpretation beyond the mechanical readings is deferred to a
session note; the results section stays observations-only.

## 6. Execution order

Control A (I1A0 → I1O0 → HP9), then Control B (A0 → A4 → O8 → A1), arms as
separate runner invocations with per-arm suffixes; commit and push after each
arm. Est. ~1 h (Control A) + ~1.5 h (Control B) CPU at observed throughput
(~46 s/run + model load per invocation).
