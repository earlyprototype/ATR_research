## SUMMARY

ATR ("Activation Tensor Resonance") is a solo-human + multi-Claude interpretability art-science project. The core operation: take a small open-weight transformer (GPT-2 family, Pythia), feed its residual-stream activation back in as its own next input, rescale to constant energy, and iterate hundreds of times with no external prompt or contingency. Text dissolves and the state settles into a small number of "attractors"/"basins." Stage 1 (internally EXP_009, carved out to a separate public repo `lucier-gpt2-activ-tensor-reson-experiments`) produced a "finding without a theory": only GPT-2 Small funnels language-driven activity into a few semantic high-confidence attractors (~5 basins, 4 thematically coherent); GPT-2 Medium collapses to one empty token; Pythia-410m never consolidates. A corpus/bias-fingerprint explanation was tested and refuted; a noise control showed random starts also converge but to different meaningless basins.

Stage 2 (the `_STAGE2_JSPACE/` folder, dated 2026-07-10/11) asks one question: **"Is the ATR attractor landscape a readout of workspace structure?"** It hangs on an external claim — that the residual stream has a band structure (input-parsing → workspace → motor), attributed to a "J-space" paper (Anthropic, 2026-07-06, transformer-circuits.pub/2026/workspace) and "Gurnee et al. 2026" — and proposes to fit a "J-lens" (Jacobian lens) instrument to probe whether ATR basins live in a workspace band.

The Potter embodied-neuroscience study is **inspiration/framing, not foundation**. It is explicitly prepared as background for a conversation with neuroengineer Steve M. Potter and maps ATR against his closed-loop culture work as a set of "rhymes" and "disanalogies." It is not part of the ATR method or Stage 2 plan; it is a side/output document, and its author repeatedly flags that no primary source was read (all full-text fetches returned HTTP 403), so it sits at abstract-level confidence throughout.

The plan documents are broadly internally consistent, with the OUTLINE ("think against") preceding and the PLAN v1 ("direction chosen") superseding it. The main divergences: the PLAN splits EXP_010 into 010a/010b and adds a fifth register entry Q-D / EXP-D that the OUTLINE lacks; a "cost pass" gate present in the OUTLINE was explicitly WAIVED in the Decision Log. Verdict vocabulary is not unified across docs (PLAN uses "supported"; RUNBOOK_PHASE0 uses PASS/FAIL/MARGINAL; RUNBOOK_PHASE1 uses limit-cycle/wandering/etc.; OUTLINE uses "falsified"). The biggest red flag is that the entire Stage 2 theory and Phase 0 instrument depend on external Anthropic artifacts (a "J-space" workspace paper and an `anthropics/jacobian-lens` repo) that are post-cutoff and unverifiable from inside this review.

---

## KEY FACTS

- **What ATR is (independent statement).** From `potter-embodied-neuroscience-study.md` line 172: "take a small open-weight language model, feed its internal activation (the residual stream) back in as its own next input, rescale to constant energy, and iterate hundreds of times. The text dissolves and the state settles into a small number of attractors (about five semantic basins in GPT-2 Small, four thematically coherent). A corpus or bias fingerprint explanation was tested and refuted. A noise control showed random inputs also converge, but to different, meaningless basins... Attractor structure differs across models. One basin turned out to be an exact two-state cycle sustained by a single network component."

- **Stage 1 = EXP_009.** `STAGE2_OUTLINE.md` line 9: "When Stage 2 matures enough to publish, it gets carved out to its own repo the way EXP_009 became `lucier-gpt2-activ-tensor-reson-experiments`." Decision Log (OUTLINE line 92): "STAGE 1 CLOSED. All five runs recorded; closing edition merged to public `main` (51c2bd5); viz PR #1 merged; trajectories backed up."

- **Stage 1 headline finding.** `STAGE2_OUTLINE.md` line 27-28: "only GPT-2 Small funnels language-driven activity into a few semantic, high-confidence attractors; its sibling (same corpus) collapses to one empty token, and Pythia-410m never consolidates."

- **_DATA role.** `_DATA/EXP009_stage1_trajectories_2026-07-10.zip` (61 MB, present in this checkout) is the frozen off-machine backup of Stage 1 trajectories, cited in `STAGE2_PLAN.md` line 119.

- **Stage 2 core question.** `STAGE2_PLAN.md` line 6 / `STAGE2_OUTLINE.md` line 36-37: "is the ATR attractor landscape a readout of workspace structure?"

- **Theory under test.** `STAGE2_PLAN.md` lines 7-9: "the residual stream has a band structure (input-parsing → workspace → motor; Gurnee et al. 2026). ATR's full-stack loop splices the motor band into the sensory band every cycle."

- **Register rules (pre-registration).** `STAGE2_PLAN.md` lines 13-15: "Register rules inherited from Stage 1: reporting register throughout this folder; pre-register hypotheses before runs; every result recorded whether it helps or not; kill criteria stated up front."

- **House rule that negatives are findings.** `STAGE2_PLAN.md` line 147: "either a workspace-grounded explanation of the Stage 1 anomaly, or a clean kill of the band-structure theory... both publishable, per the house rule that negatives are findings."

- **Working arrangement (two-Claude split).** `STAGE2_PLAN.md` lines 134-137: "Fable session: experiment design, verdicts, synthesis, editorial register. Opus session(s): runbook execution — one runbook per phase... commit after each experiment, never touch declared frozen inputs, deviations documented."

- **Model access (remote).** `REMOTE_ENV_MODEL_ACCESS.md`: gpt2/medium/large via a "legacy S3 mirror" (`s3.amazonaws.com/models.huggingface.co/bert/`, explicitly "the pre-2020 HF mirror"); Pythia via huggingface.co after TC added HF hosts to the allowlist on 2026-07-25. All four Stage-1 models cached locally (`STAGE2_PLAN.md` line 126).

- **Execution has gone beyond the plan's register** (repo listing, outside the 7 mandated docs): `_STAGE2_JSPACE/` contains specs and results for EXP_010c/010c2/010c3/010c3b/010d/012_PYTHIA, a J-lens-on-Medium track (`RESULTS_JLENS_MEDIUM.md`), and session notes; the git log references a hypothesis "H11 REFUTED" (commit c4bf9c0) not present in the plan's register. These were not in scope for this read but signal the plan is not the whole story.

---

## CLAIMS AND HYPOTHESES

The plan's formal hypothesis register (`STAGE2_PLAN.md` lines 19-26; restated `STAGE2_OUTLINE.md` lines 41-50). All are **pre-registered predictions, untested at plan-authoring time** — "status per source" is therefore "open/pre-registered." Actual dispositions live in `RESULTS_*` files outside this review's scope; I did not read them and do not assert their outcomes here.

- **H5 (band-dependence).** Statement (PLAN): "Layer-window loops (inject i, extract j) produce qualitatively different landscapes *within* the putative workspace band vs *across* band boundaries." OUTLINE variant (line 42): landscapes "differ qualitatively when the window sits inside the workspace band vs spans band boundaries." Where: PLAN table + OUTLINE §2. Tested by EXP_010b. Status per source: open, pre-registered. Assessment: testable with the existing engine (no J-lens needed); but "workspace band" is only definable once the external band-structure theory is granted — H5 as a *workspace* claim inherits the unverified premise. As a bare "does the landscape depend on where you cut the loop" question it is well-posed and answerable from artifacts.

- **H6 (workspace capture).** Statement (PLAN): "GPT-2 Small's five basin tensors project significantly more onto the J-space than the 18 null-model basins." Where: PLAN table + OUTLINE §2 (line 44). Tested by EXP_011. Status: open, pre-registered. Assessment: sharp and falsifiable, reuses the Stage 1 permutation-null pattern; but wholly contingent on a validated J-lens existing (Phase 0 gate) and on the external J-space construct being real.

- **H7 (small-model workspace / census).** Statement (PLAN): "A coherent J-lens band structure exists at 124M–410M scale, and its presence/shape predicts each model's ATR regime." OUTLINE variant (line 47): "A J-lens computed for GPT-2 Small yields a coherent workspace band; its presence/shape across the four Stage-1 models predicts their ATR regime." Where: PLAN table + OUTLINE §2. Tested by EXP_012. Status: open, pre-registered, gated (Phase 3 conditional). Assessment: the heaviest and most speculative claim; the plan itself flags the paper's models are far larger, and RUNBOOK_PHASE0 warns "gpt2-small is far weaker than the paper's models — mid-layer concepts may be faint." Support is entirely downstream of the Phase 0 validation gate passing.

- **H8 (depth / bridge).** Statement (PLAN): "Pythia-410m's fragmentation is depth-dependent: looping layers 0–11 (vs native 0–23) changes convergence behaviour." OUTLINE (line 49) adds it is "Stage 1's unrun Control 2." Where: PLAN table + OUTLINE §2. Tested by EXP_010a. Status: open, pre-registered. Assessment: the cleanest, cheapest, self-contained hypothesis — no J-lens dependency, uses only the ATR engine and existing prompts; artifact-supportable independent of the whole workspace theory. Strongest candidate to yield a defensible result.

- **Q-D (Divine object — question, not hypothesis).** Statement (PLAN): "What is the `Divine` object — limit cycle, wandering attractor, or decode-region plateau?" Where: PLAN table (line 26) — **not present in the OUTLINE's hypothesis section.** Tested by EXP-D (classical, from existing gated trajectories) and EXP_013 (J-corrected readout). Status: open; PLAN line 34: "EXP-D (Divine) is unconditional: any outcome is a finding." Assessment: well-scoped and answerable purely from the frozen Stage 1 gated trajectories (`gated_results.pt`), no new theory needed; EXP-D is the most self-sufficient piece of work in the plan. RUNBOOK_PHASE1 fixes the population at "34 never-converging prompts... all `Divine`."

Note on Stage 1 hypotheses: the OUTLINE (line 39, "continuing Stage 1's H0–H4 numbering") establishes that H0–H4 were Stage 1 hypotheses; their statements are not in the Stage 2 docs (they live in the Stage 1 repo / EXP009 archive).

---

## CONFLICTS / CONTRADICTIONS

- **Register size differs between OUTLINE and PLAN.** OUTLINE §2 defines only H5–H8 and its experiment table (lines 54-59) lists EXP_010, 011, 012, 013. The PLAN adds **Q-D and EXP-D** (Divine characterisation) and **splits EXP_010 into EXP_010a (H8) and EXP_010b (H5)**. A reader of the OUTLINE alone would not know Q-D/EXP-D exist. This is evolution (OUTLINE → PLAN v1), not a live contradiction, but the register is not stated identically in the two "plan" docs.

- **Cost-pass gate: present then waived.** OUTLINE experiment table (lines 57-58) marks EXP_011 "Blocked on J-lens implementation cost pass" and EXP_012 "needs the cost pass first." Decision Log (OUTLINE line 88, 2026-07-10): "Cost pass WAIVED — J-lens work proceeds regardless of compute cost... EXP_011–013 no longer gated." Consistent once you read the log, but the table body and the log disagree on their face.

- **"Stage 2 stays out of the public ATR repo" vs current repo state.** OUTLINE line 9 and PLAN lines 138-140: "the public ATR repo is not touched until Stage 2 has a publishable story of its own"; RUNBOOK ground rules place all Stage 2 work inside a *private* "fold" repo at a Windows path (`C:\Users\Fab2\Desktop\AI\_learn\_fold\03_MECHINIPHYLUM\_STAGE2_JSPACE`). Yet in this checkout `_STAGE2_JSPACE/` lives **inside** `earlyprototype/atr_research` (a repo the brief describes as the project repo). Either the carve-out rule was superseded or the repo layout described in the runbooks no longer matches reality. Leadership should confirm which repo is canonical.

- **Verdict vocabulary is not unified.** PLAN uses "supported" (lines 84, 99: "Census only if H5 or H6 supported"; "H8 supported"). RUNBOOK_PHASE0 line 87 mandates "PASS / FAIL / MARGINAL" for the validation gate. RUNBOOK_PHASE1 line 57 mandates a verdict of "limit cycle / wandering within a decode region / slow transient after all / other." OUTLINE line 74 uses "falsified." There is no single SUPPORTED/REFUTED lexicon defined anywhere in the plan docs (the "REFUTED" token appears only in git history, e.g. H11, outside these files).

- **Kill-criteria phrasing.** PLAN kill criteria (lines 30-32) is a *conjunction*: "EXP_010 shows **no** qualitative window-dependence, **and** EXP_011 shows basins project no differently than noise basins → the band-structure theory is dead." Line 33: "Any single experiment failing does NOT kill the frame alone." This is internally clear, but note EXP_010 here means the 010a/010b pair — the kill test is stated at the umbrella level while execution is at the sub-experiment level.

No hard factual contradictions between docs beyond the above; the two plan documents are chronologically layered (OUTLINE = 2026-07-10 "think against," PLAN = 2026-07-11 "direction chosen") and the Decision Log reconciles them.

---

## RED FLAGS

- **The entire Stage 2 theory and Phase 0 instrument rest on unverifiable external artifacts.** The band-structure theory is attributed to a "J-space paper (Anthropic, 2026-07-06)" at `transformer-circuits.pub/2026/workspace/index.html` and to "Gurnee et al. 2026" (`STAGE2_PLAN.md` lines 8-9; `STAGE2_OUTLINE.md` lines 31, 70-72). Phase 0 depends on cloning `https://github.com/anthropics/jacobian-lens` described as an "Official release... Apache-2.0, reference implementation, unmaintained" with a detailed API (`STAGE2_PLAN.md` lines 44-56; `RUNBOOK_PHASE0.md` lines 43-51). All are dated after my knowledge cutoff and cannot be confirmed from inside this repo. If any of these do not exist as described, Phases 0/2/3 (H6, H7, EXP_011/012/013 — the J-space "core") are unbuildable, and the plan's own kill criteria (which require EXP_011) cannot be executed. This is the single largest risk to the programme and should be independently verified before any J-lens compute is spent.

- **Model-weight provenance is explicitly uncertain.** `REMOTE_ENV_MODEL_ACCESS.md` sources GPT-2 weights from "the pre-2020 HF mirror" on a public S3 bucket and states "Provenance caveat stands... every experiment's reproduction gate is the behavioural check on these weights." Mitigation exists (behavioural reproduction gate; EXP_010b line 98-99 baseline "must reproduce Stage 1 basins... if it disagrees, STOP"), but downstream conclusions inherit whatever these mirrored weights are.

- **The Phase 0 validation gate is soft and pre-conceded to possibly fail.** `RUNBOOK_PHASE0.md` line 84-87: "gpt2-small is far weaker than the paper's models — mid-layer concepts may be faint. The gate is 'beats logit lens,' not 'matches the paper's examples.'" A MARGINAL/FAIL "does not stop Phase 1... but blocks Phase 2." The instrument on which H6/H7 depend may not clear its own bar, and the bar is qualitative ("majority of test prompts," judged by the operator), not quantitative.

- **The plan register (H5–H8, Q-D) is already outrun by execution.** The repo contains EXP_010c/010c2/010c3/010c3b/010d/012_PYTHIA specs and a hypothesis "H11" (git commit c4bf9c0: "H11 REFUTED") that appear nowhere in the pre-registered register. For a project whose stated discipline is "pre-register hypotheses before runs" (`STAGE2_PLAN.md` line 14), the proliferation of post-hoc experiment variants and un-registered hypotheses is exactly the drift the register is meant to prevent. Leadership should check that H11 and the 010c/010d family were pre-registered somewhere before they ran.

- **Potter study confidence ceiling.** `potter-embodied-neuroscience-study.md` (lines 13-20, 217): no primary full text was read this session — every claim is at "secondary source" (abstract/index) or "background knowledge" resolution because all full-text fetches returned HTTP 403. The document is admirably explicit about this, but any use of it as evidence (rather than as a conversation prep / analogy source) would be overreach. It also contains corrections to an earlier "brief" (wrong journal, missing co-authors), implying the project's own prior summaries of this literature contained errors.

- **Anthropomorphic / evocative naming.** The `Divine` basin label and "Activation Tensor Resonance" branding, plus the art-science framing, invite over-reading. The docs mostly guard against this (OUTLINE §5 "No consciousness claims"; the two-register doctrine separating "artwork" README from rigorous back-pages), but the naming is a standing hazard for how results get communicated.

---

## RECOMMENDATIONS

- **Verify the external dependencies first, before any Stage 2 compute.** Confirm the existence and licence of `anthropics/jacobian-lens`, the transformer-circuits.pub/2026/workspace paper, and the "Gurnee et al. 2026" band-structure result. If they are real, pin exact commits/DOIs into the plan. If they are not, the J-space arm (H6/H7, EXP_011/012/013) must be re-scoped or dropped, and the kill criteria rewritten (they currently require EXP_011).

- **Prioritise the theory-independent work.** EXP_010a (H8, depth) and EXP-D (Q-D, Divine) need no J-lens and only the frozen Stage 1 archive; they are the most robust deliverables regardless of whether the workspace theory survives. Front-load them.

- **Unify the verdict lexicon.** Adopt one register-wide vocabulary (e.g. SUPPORTED / REFUTED / INCONCLUSIVE for hypotheses; PASS/FAIL/MARGINAL is fine for gates) and record every hypothesis disposition against it in one hypothesis-disposition table, since PLAN, both runbooks, and OUTLINE currently each use different words.

- **Reconcile the register with what actually ran.** Retrofit H11 and EXP_010c/010d/012_PYTHIA into the register (or document why they are legitimate follow-ups), so the "pre-register before runs" rule is demonstrably honoured. This is important for a leadership review that will judge the project on its stated epistemic discipline.

- **Resolve the repo-canonicity question.** State plainly whether Stage 2 now lives in `earlyprototype/atr_research` (as this checkout shows) or in a private "fold" repo (as the runbooks say), and update the OUTLINE/RUNBOOK paths, so the "don't touch the public repo until mature" rule is either enforced or formally retired.

- **Keep the Potter study clearly labelled as inspiration.** It is a well-flagged analogy/conversation-prep document, not evidence. Do not let its "rhymes" (isolated recurrent system, collapse onto low-dimensional modes) migrate into ATR's results register as support; the disanalogies section (frozen weights vs plasticity; no embodiment) is the load-bearing part and should be cited whenever the Potter comparison is invoked.

- **Read the RESULTS_* and session files next.** This review covered only the 7 planning docs; actual hypothesis dispositions, the Phase 0 gate verdict, and the fate of H5–H8/H11 live in `_STAGE2_JSPACE/experiments/**/RESULTS_*.md`, `RESULTS_JLENS_MEDIUM.md`, and `sessions/`. Those must be read before judging whether artifacts support the claims.