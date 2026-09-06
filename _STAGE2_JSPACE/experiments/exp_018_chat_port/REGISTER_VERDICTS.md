# EXP_018: proposed register rows

**What this file is.** `_STAGE2_JSPACE/REGISTER.md` is the authority for every
hypothesis number and experiment identifier in this project, and rule R3 says a
session must not edit it directly. This file holds the rows EXP_018 proposes,
in the register's own format, for the orchestrator's sweep to apply. Nothing
here is in force until it lands in `REGISTER.md`.

**Session:** agent:exp018-chat-port, 2026-09-05, branch
`claude/latent-context-small-llms-u2jdig-exp018`, tracker issue #81.
**Evidence:** `experiments/exp_018_chat_port/RESULTS_EXP018.md` and the
artifacts beside it. **Specification of record:**
`_STAGE2_JSPACE/EXP_018_SPEC.md`, committed before the run.

---

## RULING REQUIRED BEFORE ANY OF THIS IS APPLIED (H19b)

**Do not sweep the H19b row below into `REGISTER.md` until TC has ruled on this.**
The other three rows, H19, H19a and the EXP_018 experiment row, are not affected
and are not waiting on anything.

**The question.** H19b is scored on "settled states". Zero of the 25 prompts
passed the convergence gate, so nothing settled in the sense that gate defines,
and the states actually scored are the states each prompt was on when the
150-repetition cap was reached. The specification's pre-registered verdict rule
for H19b, quoted in full from section 5 of `_STAGE2_JSPACE/EXP_018_SPEC.md`,
reads: "SUPPORTED if, at 8 or more of the 15 band layers, the settled-state
median share is below the ordinary-state median share and the permutation
p-value at that layer is below 0.05. NOT SUPPORTED otherwise. UNTESTABLE if the
lens file cannot be read or the settled states cannot be produced." Does that
last clause fire?

**Reading one, UNTESTABLE.** The specification defines settling once, inside
H19a's registered statement: "the modern model's loop settles (lag-2 gate,
cosine 0.999 sustained over three checks)". Nothing met it, so there are no
settled states and the clause fires. The phrase "at lock-in or at the cap",
which is how the specification elsewhere says "the state the prompt ends on",
is used for the readout in section 4 and for the position-collapse metric in
section 5 and is used nowhere in the H19b passage.

**Reading two, answered on the cap states, which is what the row below says.**
Section 6 of the specification calls the cap tensors the settled tensors:
"`output/terminal_states_bare.npz` and `output/terminal_states_chat.npz`, the
settled tensors in float32", and H19b's procedure step 4 points at that
artifact: "The settled states are obtained by injecting each settled tensor at
`blocks.0.hook_resid_pre`". Section 8 uses "settled state" the same loose way,
for where a prompt lands "after one hundred and fifty steps".

**Why this file does not decide it.** The specification uses "settled" in both
senses and never says which governs H19b, so the wording is ambiguous rather
than clear, and rule R8 reserves the resolution for the operator. The verdict
word in the H19b row below is therefore left exactly as this session scored it,
with this notice attached.

**What a ruling changes.** No measured number moves under either reading. Under
reading two the H19b row stands as written. Under reading one its verdict
becomes **UNTESTABLE**, the word **SUPPORTED** is withdrawn by name, and the
whole comparison stays on the record as an exploratory reading with every number
intact. A ruling comment beginning `TC-RULING:` on pull request #86 is what this
is waiting for, per obligation T2. The record sets the same question out at
greater length under "Decision 5" of
`experiments/exp_018_chat_port/RESULTS_EXP018.md`.

---

## Hypothesis rows (replace the UNTESTED rows allocated on 2026-09-05)

Register format: `| ID | Owning experiment | Statement (one line) | Verdict | Recorded at |`

| H19 | EXP_018 | In a rotary-position chat model looped at natural loudness with position 0 excluded from the norm, the token positions do not collapse to one vector: the mean pairwise cosine between positions of the settled tensor stays below 0.99, where GPT-2 Small reaches 1.00 by about iteration 10 | **SUPPORTED** (median 0.726 across the 25 prompts, range 0.483 to 0.929; 0 of 25 at or above 0.99 at the cap, which is where the wording scores them; no upward trend over 150 repetitions. Two prompts cross 0.99 in passing mid-run, F01_anger at five repetitions between 24 and 33 with a highest value of 0.997077 and B03_moon at repetitions 74 and 75 with a highest value of 0.996001, and both fall back within two repetitions. On the position-0-excluded measure the tables carry beside it, two further prompts cross 0.99 for one repetition each, A03_neuro at repetition 3 in this arm and A08_linguistics at repetition 128 in the pilot arm, and no prompt in either arm ends at or above 0.99 on either measure) | `experiments/exp_018_chat_port/RESULTS_EXP018.md` |
| H19a | EXP_018 | At natural loudness the modern model's loop settles (lag-2 gate, cosine 0.999 sustained over three checks) on at least half of the run prompts within the iteration budget | **NOT SUPPORTED** (0 of 25 settled; budget was a 150-repetition cap, set in the spec from measured cost. Best lag-2 cosine after repetition 100 has a median of 0.9939 per prompt and a highest value anywhere of 0.9986, against the 0.999 the gate must exceed and hold. Longer horizons untested) | same |
| H19b | EXP_018 | The modern model's settled states have a lower J-space share on its pre-fitted lens than its ordinary prompt residuals at the same layer (the H16b test transplanted) | **HELD, ruling required: see the section at the top of this file.** As scored by this session, **SUPPORTED** (settled median below ordinary median at 15 of 15 band layers, each with a one-sided paired permutation p of 0.0001 at 10,000 draws, against the pre-registered rule of 8 of 15. Band medians: ordinary 0.031, settled 0.0014. Against a randomly rotated lens at two seeds, ordinary states sit at 2.86 times chance and settled states at 0.077 times chance) | same |

---

## Experiment row (replace the IN PROGRESS row)

Register format: `| ID | What it is | Status | Spec | Results |`

| EXP_018 | First modern chat-model port: Qwen3-1.7B under natural loudness with position 0 excluded from the norm, with its pre-fitted Neuronpedia lens (H19, H19a, H19b). Gemma-3-270M, the note's first choice, is licence-gated in this environment (HTTP 401 without a token); the Qwen2.5-0.5B-Instruct fallback was not needed | **COMPLETE** (2026-09-05; 25 prompts bare text plus a 5-prompt chat-template pilot; six recorded deviations, led by bfloat16 weights with float32 loop arithmetic because the registered loader does not fit in this machine's memory) | `EXP_018_SPEC.md` | `experiments/exp_018_chat_port/RESULTS_EXP018.md` |

---

## Notes for the sweep

1. The statements are left exactly as allocated. Each verdict is scored on the
   wording in the register, not on a wording this session would have preferred.
2. H19a's verdict is bounded by its budget and the row says so. The
   pre-registered cap was 150 repetitions, chosen from the measured cost of one
   repetition before the run. What is established is that no prompt had settled
   by repetition 150; what is not established is that none ever would. The
   registered GPT-2 tiers use a 1,000-repetition cap, which on this model would
   cost about seven times the compute this run used.
3. The identifier `EXP_018` is used in full throughout, and no new hypothesis
   number or experiment identifier was created by this session.
4. **The H19b row is held.** Its verdict word depends on a reading of the
   specification that this session could not settle, and the section at the top
   of this file states the question and both readings. Apply H19, H19a and the
   EXP_018 experiment row as they stand; hold H19b until a comment beginning
   `TC-RULING:` says which reading governs.
