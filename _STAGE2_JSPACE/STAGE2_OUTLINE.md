# Stage 2 — ATR × J-Space (Outline)

**Status:** OUTLINE TO THINK AGAINST — not committed work, no experiments scheduled.
**Created:** 2026-07-10
**Rule of the folder:** Stage 2 lives HERE and only here. Notebooks, data, notes, drafts —
everything Stage 2 goes in `_STAGE2_JSPACE/` (deliberate deviation from the
`_LAB_NOTEBOOKS/` convention, so the new stage cannot tangle with Stage 1 artifacts).
When Stage 2 matures enough to publish, it gets carved out to its own repo the way
EXP_009 became `lucier-gpt2-activ-tensor-reson-experiments` — it does not grow inside
the ATR repo.

**Entry criteria (do not start Stage 2 before these):**
1. Stage 1 closed: run 5 (convergence-gated re-sweep) reported, and the **closing
   editorial pass** shipped under the two-register doctrine: the README is the artwork
   (provocation licensed; may provoke, may NOT misreport) and the reporting behind it
   has no such privilege (conjecture cut or explicitly labelled, theatre removed,
   language tightened — full rigour). Separate the registers structurally: apparatus
   (status tables, configs, caveats, statistics) moves out of the README into the back
   pages. The work gets honoured by the telling. "Exhaust the initial experiment set
   first."
2. A direction decision recorded in the Decision Log below.

---

## 1. The question Stage 2 asks

Stage 1 ended on a finding without a theory: **only GPT-2 Small funnels language-driven
activity into a few semantic, high-confidence attractors; its sibling (same corpus)
collapses to one empty token, and Pythia-410m never consolidates.** The J-space paper
(Anthropic, 2026-07-06) supplies the first candidate theory: the residual stream has a
band structure (input-parsing → workspace → motor), and ATR's full-stack loop splices
the motor band into the sensory band every cycle.

**Stage 2's core question: is the ATR attractor landscape a readout of workspace
structure?** Sub-questions: does a workspace band exist at 124M–410M scale; do semantic
basins live inside it; does looping *within* the band behave differently from looping
*across* band boundaries?

## 2. Hypotheses (continuing Stage 1's H0–H4 numbering)

- **H5 (band-dependence):** Attractor landscapes from layer-window loops (i→j) differ
  qualitatively when the window sits inside the workspace band vs spans band boundaries.
- **H6 (workspace capture):** GPT-2 Small's five semantic basin tensors have
  significantly higher projection onto the J-space than the 18 null-model basins.
- **H7 (small-model workspace):** A J-lens computed for GPT-2 Small yields a coherent
  workspace band; its presence/shape across the four Stage-1 models predicts their
  ATR regime (semantic basins / single funnel / fragmentation).
- **H8 (depth, bridge from Stage 1):** Pythia-410m looped over layers 0–11 (vs native
  0–23) changes convergence behaviour — Stage 1's unrun Control 2, which doubles as the
  cheapest first probe of H5.

## 3. Candidate experiments (sketches, not designs)

| ID | Name | Tests | One-line method | Cost gate |
|---|---|---|---|---|
| EXP_010 | Windowed resonance | H5, H8 | ATR loop over (inject i, extract j) grid; start with the March `layer_resonance` scaffold | Cheap (existing engine + hook params) |
| EXP_011 | J-space overlap | H6 | Build J-lens vectors for gpt2-small; project Stage-1 basin tensors (real + null) onto J-space vs complement | Blocked on J-lens implementation cost pass |
| EXP_012 | Small-model workspace census | H7 | J-lens band structure for all four Stage-1 models; compare to their ATR regimes | Heaviest — Jacobians over vocab × corpus; needs the cost pass first |
| EXP_013 | J-readout for ATR | (instrument) | Re-decode Stage-1 trajectories with J-lens-corrected readout instead of logit lens; closes the "readout ambiguity" thread properly | Depends on EXP_011's vectors |

Suggested order if all go ahead: **EXP_010 → 011 → 013 → 012** (theory-relevant results
before the expensive census; 010 needs no J-lens at all).

## 4. Frozen inputs from Stage 1 (read-only — do not modify)

- Trajectories & basins: `../_LAB_NOTEBOOKS/lucier-repo/experiments/*/output*/` (`.pt`
  files are local-only/gitignored; back up before any disk cleanup)
- Verdicts: `../_LAB_NOTEBOOKS/lucier-repo/experiments/RESULTS_SUMMARY.md` (branch `cross-model`)
- Artefact analysis: `../_LAB_NOTEBOOKS/lucier-repo/docs/SCALING_ARTEFACT_ANALYSIS.md`
- Thinking trail: `../../01_SMOOTHSPACE/_LITERATURE/_SOURCES/Anthropic/J_Space_Paper_ATR_IMPLICATIONS.md`
- Paper: same folder, `J_Space_Paper.md`; canonical
  https://transformer-circuits.pub/2026/workspace/index.html (replication info: appendix A.2)

## 5. Non-goals

- No consciousness claims. The paper explicitly takes no position on phenomenal
  consciousness; Stage 2 inherits the functional framing only.
- No corpus-fingerprint revival. Stage 1 falsified it; it stays falsified unless new
  evidence forces the issue.
- No growth inside the public ATR repo. That repo tells the Stage 1 story; Stage 2 is a
  new thing.

## 6. Decision Log

| Date | Decision | By |
|---|---|---|
| 2026-07-10 | Folder created; outline recorded; direction NOT yet chosen | TC |
| 2026-07-10 | Cost pass WAIVED — J-lens work proceeds regardless of compute cost ("it is what it is"); EXP_011–013 no longer gated | TC |
| 2026-07-10 | Stage 1 closure expanded to include full editorial cleanse of the ATR repo (README + back pages) | TC |
| 2026-07-11 | **DIRECTION CHOSEN: J-space integration programme, bridge-first.** STAGE2_PLAN.md v1 recorded — Phase 0 instrument (adapt Anthropic's open-source J-lens release), Phase 1 bridge (depth control, window grid, Divine characterisation — no J-lens needed), Phase 2 core (overlap, J-readout), Phase 3 census (conditional). Kill criteria stated. Entry criteria fully satisfied | TC |
| 2026-07-11 | Post-close: W_E permutation test paid — negative (all-warm = anisotropy artifact); Stage 1 record updated | TC |
| 2026-07-11 | **STAGE 1 CLOSED.** All five runs recorded; closing edition merged to public `main` (51c2bd5); viz PR #1 merged; trajectories backed up. Entry criterion 1 satisfied — Stage 2 awaits only the direction decision | TC |
