# Future work, recorded and not chartered

**Status:** a holding record. Nothing here has an experiment identifier or a hypothesis number, and nothing here may be started without a register row (rule R3) and an issue (rule R2). Entries are dated and carry the decision that parked them.

## The Coconut line (recorded 2026-09-05 under TC's in-session direction)

**What it is.** Coconut, the "chain of continuous thought" method of Hao and colleagues (December 2024), fine-tunes a language model to feed its own last-layer hidden state back in as the next input embedding and to use those extra passes for reasoning; its public code fine-tunes GPT-2 Small. This project's loop feeds the whole final residual tensor back into layer 0 with no training at all. The reading note of 2026-09-04 (`docs/LATENT_CONTEXT_NOTE_2026-09-04.md` in the lucier repository) argued that the loop is Coconut's latent mode with the training taken away, and that the difference in outcome, a search over reasoning states there against a small set of settled states here, is on the survey's account the training. That is an inference, not a measurement.

**The experiment, when chartered.** Fine-tune GPT-2 Small with Coconut's public code on its GSM8k arithmetic curriculum; then run this project's full-stack loop and the J-space overlap probe on the fine-tuned model, using the same 25-prompt Small subset and the same conventions as EXP_011 and EXP_017, so that three models sit on one axis: base GPT-2 Small, its instruction-tuned twin, and its recurrence-trained twin. The question is whether training a model to use its own recurrence moves its settled states into or out of the verbalizable directions, and whether it changes the basin partition at all.

**Cost, estimated.** A GPU day for the fine-tune (the Coconut curriculum is several epochs over GSM8k); the loop and probe are then the same cost as EXP_017, about a day on CPU. This environment has no GPU, which is the practical reason the line is parked.

**What it needs before it starts.** An experiment identifier and hypothesis numbers claimed in `REGISTER.md` in the same commit as the specification; a tracker issue; a decision on whether the fine-tuned checkpoint is committed (it is a full model, so it would go under an artifacts-only PR or an external store); and the base, twin and recurrence-trained lenses fitted on the same corpus so the shares are comparable.

**Decision that parked it.** TC, 2026-09-05: "let's do exps 1 to 4 and record 5 as a future piece."
