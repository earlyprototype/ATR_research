"""EXP_017 Part 2: fit a Jacobian lens to MBZUAI/LaMini-GPT-124M (the twin).

Follows _STAGE2_JSPACE/experiments/jlens_medium/fit_lens.py, the registered
reference fit, changed only where the model changes (GPT-2 Small architecture,
12 layers, d_model 768, so the lens is fitted for source layers 0..10 with the
final block 11 as the target).

Instrument: anthropics/jacobian-lens, pinned commit
581d398613e5602a5af361e1c34d3a92ea82ba8e, installed editable.
Fitting corpus: WikiText-103 via jlens.examples.load_wikitext_prompts, the same
corpus and loader the reference fit used, frozen to
_STAGE2_JSPACE/artifacts/wikitext_prompts_160.json (gitignored; the derivation
is deterministic, so the file is reproducible from the loader).

Weights: pinned to the revision, that is the Hugging Face repository commit,
recorded in exp017_models.py, unless --model-path names a local directory.

The wall-clock cap, spec section 6.2. The spec caps the fit at 9,000 seconds
and says that a fit which does not finish in budget falls back to scoring H18b
with the base lens on both sides. The instrument's own jlens.fit call has no
deadline of its own, so this script enforces one in two places. Before the fit
starts it refuses outright if the timing probe's measured cost per prompt
predicts that not even one more prompt fits in the remaining budget. During the
fit it works in chunks of whole prompts, resuming from the checkpoint the
instrument writes after every prompt, and stops before starting a chunk whose
predicted cost would cross the cap. Either way the prompts already completed
still make a lens, which is written out with the count it was actually fitted
on, and the exit code says which happened: 0 when every requested prompt was
fitted, 3 when the deadline stopped it short and the section 6.2 fallback is
the caller's next step.

Usage:
    python3 fit_twin_lens.py --refresh-prompts
    python3 fit_twin_lens.py --n 5  --dim-batch 16 --tag probe
    python3 fit_twin_lens.py --n 100 --dim-batch 16 --tag twin
    python3 fit_twin_lens.py --selftest      # the deadline arithmetic, no model
"""
import argparse
import json
import logging
import os
import sys
import time

import torch

torch.set_num_threads(1)

import transformers

import jlens

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import exp017_models  # noqa: E402
import lens_from_checkpoint  # noqa: E402

STAGE2 = os.path.abspath(os.path.join(HERE, "..", ".."))
ARTIFACTS = os.path.join(STAGE2, "artifacts")
PROMPTS_JSON = os.path.join(ARTIFACTS, "wikitext_prompts_160.json")
TWIN = exp017_models.name("twin")
TWIN_REVISION = exp017_models.revision("twin")

# Spec section 6.2: the fit's wall-clock cap, in seconds. 9,000 seconds is
# 2 hours 30 minutes.
CAP_SECONDS = 9000.0
BUDGET_JSON = os.path.join(HERE, "output", "fit_budget_decision.json")

# Exit code for a fit the deadline stopped short of the requested prompt count.
EXIT_BUDGET_STOP = 3


def lens_path(n, tag=""):
    suffix = f"_{tag}" if tag else ""
    return os.path.join(ARTIFACTS, f"jlens_lamini_gpt2_124m_{n}{suffix}.pt")


def refresh_prompts():
    """Deterministically regenerate the frozen fitting-corpus list: the first
    160 WikiText-103 train records with at least 600 characters, stream order."""
    from jlens.examples import load_wikitext_prompts
    prompts = load_wikitext_prompts(160, min_chars=600)
    os.makedirs(ARTIFACTS, exist_ok=True)
    json.dump(prompts, open(PROMPTS_JSON, "w"), indent=1)
    print(f"Wrote {len(prompts)} prompts -> {PROMPTS_JSON}")


def load_model(model_path=TWIN, revision=TWIN_REVISION):
    """Load the twin and its tokenizer at the pinned revision.

    A revision is the 40-character commit identifier of the repository's state,
    so pinning it means a later change on Hugging Face cannot alter what this
    fits. A local directory has no revision, so None is passed for one.
    """
    if os.path.isdir(model_path):
        revision = None
    tok = transformers.AutoTokenizer.from_pretrained(model_path, revision=revision)
    hf = transformers.AutoModelForCausalLM.from_pretrained(model_path,
                                                           revision=revision)
    model = jlens.from_hf(hf, tok)
    assert model.n_layers == 12 and model.d_model == 768, (
        f"expected 12 layers / d_model 768, got {model.n_layers} / {model.d_model}")
    return model, tok


def probe_seconds_per_prompt(path=BUDGET_JSON):
    """The cost per prompt the five-prompt timing probe measured, in seconds,
    read from the budget decision the spec's rule already wrote. Returns None
    when that file is absent, in which case the caller must supply the number."""
    if not os.path.exists(path):
        return None
    return float(json.load(open(path))["mean_seconds_per_prompt"])


def plan_next_chunk(next_idx, n_prompts, elapsed, seconds_per_prompt,
                    deadline, chunk):
    """How many more prompts may be started without crossing the deadline.

    All times are seconds. `next_idx` is how many prompts of the list the
    checkpoint has already consumed, `elapsed` is the wall time this fit has
    spent so far, `deadline` is the cap the spec sets, and `chunk` is the most
    prompts to do between two checks of the clock. Returns 0 when the fit is
    finished or when the budget left does not cover even one more prompt, which
    is the signal to stop from the last checkpoint.
    """
    remaining = n_prompts - next_idx
    if remaining <= 0:
        return 0
    if seconds_per_prompt <= 0:
        return min(chunk, remaining)
    affordable = int((deadline - elapsed) // seconds_per_prompt)
    return max(0, min(chunk, remaining, affordable))


def fit_within_deadline(model, prompts, *, dim_batch, max_seq_len, ckpt,
                        deadline, seconds_per_prompt, chunk, clock=time.perf_counter):
    """Fit in resumable chunks, stopping before the wall-clock cap is crossed.

    Returns (lens, prompts_consumed, prompts_fitted, wall_seconds,
    prompts_fitted_here), where prompts_fitted_here counts only the prompts
    this invocation computed, as distinct from those it inherited from an
    existing checkpoint.
    """
    t0 = clock()
    lens = None
    start_idx, start_done = lens_from_checkpoint.progress(ckpt)
    next_idx, n_done = start_idx, start_done
    while True:
        take = plan_next_chunk(next_idx, len(prompts), clock() - t0,
                               seconds_per_prompt, deadline, chunk)
        if take == 0:
            break
        lens = jlens.fit(
            model,
            prompts[: next_idx + take],
            dim_batch=dim_batch,
            max_seq_len=max_seq_len,
            checkpoint_path=ckpt,
            checkpoint_every=1,
            resume=True,
        )
        moved_to, n_done = lens_from_checkpoint.progress(ckpt)
        if moved_to <= next_idx:            # no progress; stop rather than spin
            break
        next_idx = moved_to
    wall = clock() - t0
    if lens is None:                        # nothing ran, so build from the checkpoint
        lens, _ = lens_from_checkpoint.build(ckpt)
    return lens, next_idx, n_done, wall, n_done - start_done


def selftest():
    """The deadline arithmetic, checked without loading a model."""
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              f"{(': ' + detail) if detail else ''}")

    # A fresh fit with room for everything takes a full chunk.
    got = plan_next_chunk(0, 40, 0.0, 221.0, CAP_SECONDS, 5)
    check("a fresh fit inside the cap takes a full chunk", got == 5, f"got {got}")
    # The last prompts of the list are all that is left to take.
    got = plan_next_chunk(38, 40, 100.0, 221.0, CAP_SECONDS, 5)
    check("the chunk never runs past the end of the prompt list", got == 2,
          f"got {got}")
    # With one prompt's worth of budget left, only one prompt starts.
    got = plan_next_chunk(0, 40, CAP_SECONDS - 300.0, 221.0, CAP_SECONDS, 5)
    check("a nearly spent budget starts only what it can pay for", got == 1,
          f"got {got}")
    # With less than one prompt's worth left, the fit stops from the checkpoint.
    got = plan_next_chunk(0, 40, CAP_SECONDS - 100.0, 221.0, CAP_SECONDS, 5)
    check("a spent budget stops the fit before the next prompt", got == 0,
          f"got {got}")
    # A finished fit asks for nothing more.
    got = plan_next_chunk(40, 40, 10.0, 221.0, CAP_SECONDS, 5)
    check("a finished fit asks for no further chunk", got == 0, f"got {got}")
    # The pre-run refusal: a cost per prompt larger than the whole cap.
    got = plan_next_chunk(0, 40, 0.0, CAP_SECONDS + 1.0, CAP_SECONDS, 1)
    check("a fit predicted to exceed the cap never starts", got == 0,
          f"got {got}")

    # The committed run, replayed against the clock it actually took: it
    # resumed from the five-prompt probe checkpoint and computed 35 prompts in
    # 8,253 seconds, so the deadline never fired.
    elapsed, idx, fired = 0.0, 5, False
    for _ in range(35):
        if plan_next_chunk(idx, 40, elapsed, 235.8, CAP_SECONDS, 1) == 0:
            fired = True
            break
        idx += 1
        elapsed += 8253.0 / 35
    check("the committed 40-prompt fit would not have been stopped",
          idx == 40 and not fired, f"reached prompt {idx} at {elapsed:.0f}s")

    print(f"\nselftest: {'ALL PASS' if ok else 'FAILURE'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--dim-batch", type=int, default=16)
    ap.add_argument("--tag", default="twin")
    ap.add_argument("--model-path", default=TWIN)
    ap.add_argument("--revision", default=TWIN_REVISION,
                    help="Hugging Face repository commit to load; ignored when "
                         "--model-path names a local directory")
    ap.add_argument("--refresh-prompts", action="store_true")
    ap.add_argument("--selftest", action="store_true",
                    help="check the deadline arithmetic and exit")
    ap.add_argument("--deadline-seconds", type=float, default=CAP_SECONDS,
                    help="wall-clock cap for this fit, in seconds; the spec's "
                         "section 6.2 value is the default")
    ap.add_argument("--seconds-per-prompt", type=float, default=None,
                    help="predicted cost per prompt used to decide whether the "
                         "next prompt fits in the budget; defaults to the "
                         "figure the timing probe wrote into "
                         "output/fit_budget_decision.json")
    ap.add_argument("--chunk", type=int, default=1,
                    help="prompts to fit between two checks of the clock; the "
                         "instrument checkpoints every prompt, so 1 loses "
                         "nothing on a stop")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(selftest())
    if args.refresh_prompts:
        refresh_prompts()
        return 0

    per_prompt = args.seconds_per_prompt
    if per_prompt is None:
        per_prompt = probe_seconds_per_prompt()
    if per_prompt is None:
        raise SystemExit(
            "no cost per prompt available: run the timing probe so that "
            f"{BUDGET_JSON} exists, or pass --seconds-per-prompt.")

    jlens.configure_logging(level=logging.INFO)
    os.makedirs(ARTIFACTS, exist_ok=True)
    prompts = json.load(open(PROMPTS_JSON))[: args.n]
    if not prompts:
        raise SystemExit(f"--n {args.n} selects no prompts; nothing to fit.")
    ckpt = os.path.join(ARTIFACTS, f"jlens_lamini_gpt2_124m_{args.n}_{args.tag}.ckpt.pt")
    out = lens_path(args.n, args.tag)

    # The pre-run budget check, spec section 6.2: refuse to start a fit whose
    # first remaining prompt is already predicted to cross the cap.
    started, done_already = lens_from_checkpoint.progress(ckpt)
    affordable = plan_next_chunk(started, len(prompts), 0.0, per_prompt,
                                 args.deadline_seconds, args.chunk)
    if affordable == 0 and started < len(prompts):
        print(f"REFUSING TO START: {len(prompts) - started} prompts remain at a "
              f"predicted {per_prompt:.1f}s each, and the cap is "
              f"{args.deadline_seconds:.0f}s. Spec section 6.2 fallback applies.",
              flush=True)
        raise SystemExit(EXIT_BUDGET_STOP)

    shown = ("local directory" if os.path.isdir(args.model_path)
             else args.revision)
    print(f"torch threads={torch.get_num_threads()} prompts={len(prompts)} "
          f"dim_batch={args.dim_batch} model={args.model_path} "
          f"revision={shown}", flush=True)
    print(f"budget: cap {args.deadline_seconds:.0f}s, predicted {per_prompt:.1f}s "
          f"per prompt, {done_already} prompts already in the checkpoint",
          flush=True)

    model, _ = load_model(args.model_path, args.revision)

    lens, consumed, fitted, wall, fitted_here = fit_within_deadline(
        model, prompts,
        dim_batch=args.dim_batch, max_seq_len=128, ckpt=ckpt,
        deadline=args.deadline_seconds, seconds_per_prompt=per_prompt,
        chunk=args.chunk)
    lens.save(out)

    rate = wall / max(fitted_here, 1)
    print(f"DONE n={fitted} dim_batch={args.dim_batch} wall={wall:.0f}s "
          f"({fitted_here} prompts computed here at {rate:.1f}s each; the lens "
          f"averages {fitted} prompts in all) -> {out}", flush=True)
    print(lens, flush=True)
    if consumed < len(prompts):
        print(f"BUDGET STOP: {consumed} of {len(prompts)} prompts were reached "
              f"inside the {args.deadline_seconds:.0f}s cap. The lens above is "
              f"fitted on {fitted} prompts. Spec section 6.2 fallback applies: "
              f"score H18b with the base lens on both sides and record the "
              f"shortfall as a deviation.", flush=True)
        return EXIT_BUDGET_STOP
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
