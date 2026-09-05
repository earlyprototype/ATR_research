"""EXP_017 Part 2: fit a Jacobian lens to MBZUAI/LaMini-GPT-124M (the twin).

Follows _STAGE2_JSPACE/experiments/jlens_medium/fit_lens.py, the registered
reference fit, changed only where the model changes (GPT-2 Small architecture,
12 layers, d_model 768, so the lens is fitted for source layers 0..10 with the
final block 11 as the target).

Instrument: anthropics/jacobian-lens, pinned commit
581d398613e5602a5af361e1c34d3a92ea82ba8e, installed editable.

Fitting corpus: WikiText-103 via jlens.examples.load_wikitext_prompts, the same
corpus and loader the reference fit used. The exact 160 prompts this experiment
fitted on are committed beside this script as wikitext_prompts_160.json, and
their SHA-256 is checked on every run, because the loader reads a remote
dataset that could change under a later rerun. The copy the run wrote to
_STAGE2_JSPACE/artifacts/ is used only when the committed one is absent, and
--refresh-prompts regenerates it from the loader and reports whether the result
still matches the recorded digest.

Resuming from the timing probe. The registered fit continues the checkpoint the
five-prompt timing probe wrote, because the probe fits the first 5 prompts of
the same list in the same way. That reuse is now automatic: --resume-from names
a checkpoint to seed this fit from, defaulting to the probe's, and it is copied
into place only when this fit has no checkpoint of its own and the corpus digest
matches, so the prompts already done are a prefix of this fit's list.

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
    python3 fit_twin_lens.py --n 40 --dim-batch 16 --tag twin
    python3 fit_twin_lens.py --selftest      # the deadline arithmetic, no model
"""
import argparse
import hashlib
import json
import logging
import os
import shutil
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

# The fitting corpus. The committed copy is authoritative; the artifacts copy is
# what --refresh-prompts writes and is not version controlled.
PROMPTS_COMMITTED = os.path.join(HERE, "wikitext_prompts_160.json")
PROMPTS_JSON = os.path.join(ARTIFACTS, "wikitext_prompts_160.json")
PROMPTS_SHA = "c325a40c06f5dcfe14a3cea02afe1d64d42914146822c63e111dc665fdeba8d7"

TWIN = exp017_models.name("twin")
TWIN_REVISION = exp017_models.revision("twin")

# Spec section 6.2: the fit's wall-clock cap, in seconds. 9,000 seconds is
# 2 hours 30 minutes.
CAP_SECONDS = 9000.0
BUDGET_JSON = os.path.join(HERE, "output", "fit_budget_decision.json")

# The checkpoint the five-prompt timing probe writes, and which the registered
# fit continues from.
PROBE_CKPT = os.path.join(ARTIFACTS, "jlens_lamini_gpt2_124m_5_probe.ckpt.pt")

# What a checkpoint of this experiment's fit must have been built with: a
# 12-layer model gives source layers 0 through 10 with block 11 as the target,
# and 16 is the instrument's default number of leading positions to skip.
EXPECTED_CKPT_SHAPE = {"source_layers": list(range(11)), "target_layer": 11,
                       "skip_first": 16}

# Exit code for a fit the deadline stopped short of the requested prompt count.
EXIT_BUDGET_STOP = 3


def lens_path(n, tag=""):
    suffix = f"_{tag}" if tag else ""
    return os.path.join(ARTIFACTS, f"jlens_lamini_gpt2_124m_{n}{suffix}.pt")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def refresh_prompts():
    """Deterministically regenerate the frozen fitting-corpus list: the first
    160 WikiText-103 train records with at least 600 characters, stream order.

    The result goes to the artifacts copy, never over the committed one, and the
    digest is reported so that a change in the remote dataset is visible rather
    than silent.
    """
    from jlens.examples import load_wikitext_prompts
    prompts = load_wikitext_prompts(160, min_chars=600)
    os.makedirs(ARTIFACTS, exist_ok=True)
    json.dump(prompts, open(PROMPTS_JSON, "w"), indent=1)
    got = sha256_file(PROMPTS_JSON)
    print(f"Wrote {len(prompts)} prompts -> {PROMPTS_JSON}")
    print(f"sha256 {got}")
    if got == PROMPTS_SHA:
        print("This matches the corpus EXP_017 fitted on.")
    else:
        print(f"WARNING: this differs from the corpus EXP_017 fitted on, "
              f"{PROMPTS_SHA}. The remote dataset has changed, so a fit on this "
              f"file is a fit on a different corpus. The committed copy at "
              f"{PROMPTS_COMMITTED} is the one the record's numbers come from.")


def load_prompts(path=None, require_digest=True):
    """The fitting corpus, and the path it came from.

    Prefers an explicitly named file, then the committed copy beside this
    script, then the copy in the unversioned artifacts directory. The SHA-256 is
    checked against the corpus this experiment fitted on, because the loader
    reads a remote dataset that a later rerun could find changed.
    """
    for candidate in (path, PROMPTS_COMMITTED, PROMPTS_JSON):
        if candidate and os.path.exists(candidate):
            chosen = candidate
            break
    else:
        raise SystemExit(
            f"no fitting corpus found. Expected {PROMPTS_COMMITTED} or "
            f"{PROMPTS_JSON}; run --refresh-prompts to rebuild the second.")
    got = sha256_file(chosen)
    if got != PROMPTS_SHA:
        message = (f"fitting corpus {chosen} has sha256 {got}, not the "
                   f"{PROMPTS_SHA} EXP_017 fitted on. A fit on this file is a "
                   f"fit on a different corpus and its lens is not comparable "
                   f"with the committed one.")
        if require_digest:
            raise SystemExit(message + " Pass --allow-different-prompts to "
                                       "proceed anyway and record it as a "
                                       "deviation.")
        print("WARNING: " + message, flush=True)
    return json.load(open(chosen)), chosen, got


def seed_checkpoint(target, source, n_prompts):
    """Copy a compatible earlier checkpoint into place so this fit continues it.

    Used so that the registered 40-prompt fit continues the five-prompt timing
    probe rather than repeating its work, which is what the committed run did.
    Copies only when this fit has no checkpoint of its own, the source exists,
    the source was built with this experiment's layer choices, and the source
    has consumed no more prompts than this fit's list holds. The prompts already
    done are then a prefix of this fit's list, because both take the first N
    entries of the same corpus file in order and the corpus digest is checked
    before this runs. Returns a sentence describing what happened.
    """
    if not source:
        return "not seeding: --resume-from was empty"
    if os.path.abspath(source) == os.path.abspath(target):
        return "not seeding: the source and this fit's checkpoint are the same file"
    if os.path.exists(target):
        done, fitted = lens_from_checkpoint.progress(target)
        return (f"not seeding: this fit already has a checkpoint at {target} "
                f"with {fitted} prompts fitted")
    if not os.path.exists(source):
        return f"not seeding: no checkpoint at {source}"
    state = torch.load(source, map_location="cpu", weights_only=True)
    for key, expected in EXPECTED_CKPT_SHAPE.items():
        got = state.get(key)
        if isinstance(expected, list) and got is not None:
            got = list(got)
        if got != expected:
            return (f"not seeding: {source} was built with {key}={got!r}, "
                    f"not {expected!r}")
    if int(state["next_idx"]) > n_prompts:
        return (f"not seeding: {source} has consumed {state['next_idx']} prompts, "
                f"more than the {n_prompts} this fit asks for")
    shutil.copyfile(source, target)
    return (f"seeded {target} from {source}: {state['n_done']} prompts already "
            f"fitted, so this fit continues at prompt {int(state['next_idx']) + 1}")


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

    # The same replay from nothing, using every one of the 40 prompts' recorded
    # times, including the five the probe paid for. This is why the fit has to
    # continue the probe's checkpoint rather than start over.
    recorded = [245, 209, 208, 207, 236, 309, 296, 305, 453, 438, 370, 256, 253,
                253, 215, 203, 203, 200, 199, 199, 202, 207, 197, 202, 209, 210,
                209, 210, 209, 189, 208, 211, 211, 202, 200, 198, 197, 222, 205,
                202]
    elapsed, idx = 0.0, 0
    for t in recorded:
        if plan_next_chunk(idx, 40, elapsed, 221.0, CAP_SECONDS, 1) == 0:
            break
        elapsed += t
        idx += 1
    check("a fit started from nothing would stop short of 40 prompts",
          idx == 38, f"reached prompt {idx} of 40 at {elapsed:.0f}s")

    # The committed fitting corpus is present and is the one the run used.
    try:
        corpus, path, sha = load_prompts()
        check("the committed fitting corpus is present and its digest matches",
              len(corpus) == 160 and sha == PROMPTS_SHA,
              f"{len(corpus)} prompts from {os.path.basename(path)}")
    except SystemExit as exc:
        check("the committed fitting corpus is present and its digest matches",
              False, str(exc))

    # Seeding one fit's checkpoint from an earlier compatible one.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        good = os.path.join(tmp, "probe.ckpt.pt")
        torch.save({"jacobian_sum": {l: torch.zeros(2, 2) for l in range(11)},
                    "n_done": 5, "next_idx": 5, **EXPECTED_CKPT_SHAPE}, good)
        target = os.path.join(tmp, "twin.ckpt.pt")
        msg = seed_checkpoint(target, good, 40)
        check("a compatible probe checkpoint seeds the fit",
              os.path.exists(target) and msg.startswith("seeded"), msg)
        msg = seed_checkpoint(target, good, 40)
        check("an existing checkpoint is never overwritten by seeding",
              msg.startswith("not seeding: this fit already has"), msg)
        wrong = os.path.join(tmp, "wrong.ckpt.pt")
        torch.save({"jacobian_sum": {0: torch.zeros(2, 2)}, "n_done": 5,
                    "next_idx": 5, "source_layers": [0], "target_layer": 1,
                    "skip_first": 16}, wrong)
        msg = seed_checkpoint(os.path.join(tmp, "t2.ckpt.pt"), wrong, 40)
        check("a checkpoint from different layers is refused",
              msg.startswith("not seeding: ") and "source_layers" in msg, msg)
        msg = seed_checkpoint(os.path.join(tmp, "t3.ckpt.pt"), good, 3)
        check("a checkpoint holding more prompts than the fit asks for is refused",
              "more than the 3" in msg, msg)
        msg = seed_checkpoint(os.path.join(tmp, "t4.ckpt.pt"), "", 40)
        check("an empty --resume-from seeds nothing", "not seeding" in msg, msg)

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
    ap.add_argument("--prompts", default=None,
                    help="fitting corpus to use; defaults to the committed "
                         "wikitext_prompts_160.json beside this script")
    ap.add_argument("--allow-different-prompts", action="store_true",
                    help="proceed even when the corpus digest is not the one "
                         "EXP_017 fitted on; the resulting lens is not "
                         "comparable with the committed one")
    ap.add_argument("--resume-from", default=PROBE_CKPT,
                    help="checkpoint to continue, copied into place when this "
                         "fit has none of its own; defaults to the five-prompt "
                         "timing probe's checkpoint, which is what the "
                         "committed 40-prompt fit continued. Pass an empty "
                         "string to start from nothing.")
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
    corpus, corpus_path, corpus_sha = load_prompts(
        args.prompts, require_digest=not args.allow_different_prompts)
    prompts = corpus[: args.n]
    if not prompts:
        raise SystemExit(f"--n {args.n} selects no prompts; nothing to fit.")
    print(f"corpus: {corpus_path} sha256 {corpus_sha} "
          f"({len(corpus)} prompts, taking the first {len(prompts)})", flush=True)
    ckpt = os.path.join(ARTIFACTS, f"jlens_lamini_gpt2_124m_{args.n}_{args.tag}.ckpt.pt")
    out = lens_path(args.n, args.tag)

    # Continue the timing probe's work rather than repeating it, which is what
    # the committed run did by hand.
    print(seed_checkpoint(ckpt, args.resume_from, len(prompts)), flush=True)

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

    # A fit the deadline stopped short is never written under the name the
    # registered probe looks for. It goes to its own file whose name carries the
    # prompt count it actually reached, so that no later step can mistake it for
    # the lens the budget rule asked for.
    stopped = consumed < len(prompts)
    out = lens_path(fitted, f"{args.tag}_partial") if stopped else out
    lens.save(out)

    rate = wall / max(fitted_here, 1)
    print(f"DONE n={fitted} dim_batch={args.dim_batch} wall={wall:.0f}s "
          f"({fitted_here} prompts computed here at {rate:.1f}s each; the lens "
          f"averages {fitted} prompts in all) -> {out}", flush=True)
    print(lens, flush=True)
    if stopped:
        print(f"BUDGET STOP: {consumed} of {len(prompts)} prompts were reached "
              f"inside the {args.deadline_seconds:.0f}s cap. The partial lens "
              f"is fitted on {fitted} prompts and was written to {out}, not to "
              f"{lens_path(args.n, args.tag)}, so it cannot be scored as the "
              f"registered twin lens. Spec section 6.2 fallback applies: run "
              f"run_jspace.py without --twin-lens, which scores H18b with the "
              f"base lens on both sides, and record the shortfall as a "
              f"deviation.", flush=True)
        return EXIT_BUDGET_STOP
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
