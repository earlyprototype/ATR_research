"""EXP_018 runner: the ATR loop on Qwen3-1.7B. Spec: `../../EXP_018_SPEC.md`.

Stages, in the order they are meant to run:

  probe    Load the model, measure resident memory and per-pass time, record
           the natural per-layer entry loudness of every prompt in both arms,
           and time a short loop. These numbers set the budget written into
           the spec, so this stage runs before the spec is committed.
  loop     The registered run. `--arm bare` is the main arm (bare text, the
           25-prompt Small subset); `--arm chat` is the pilot arm (the first
           five of the same prompts wrapped as a user turn, thinking mode off).
           `--resume` picks up a partly finished arm, and refuses any resume
           that would put records made under two settings into one file: the
           precision, the weights revision and every loop parameter must match
           what the saved file records. A saved file that records no weights
           revision at all, which is the state of both files committed with
           this experiment, needs `--assume-legacy-revision` before it will run
           anything further, because otherwise the prompts still to run would
           go through whatever the cache pointer names today.
  states   Per-layer states for the J-space test (H19b): inject each settled
           tensor at the layer-0 entry and read every scored layer, and run the
           same prompts clean for the comparison arm. The precision is the one
           the loop recorded in its results file unless `--dtype` overrides it.
           The load is always pinned to one version of the weights, taken from
           `--revision` or from the revision the loop recorded; when neither
           exists the stage stops rather than following the cache pointer.
  lagscan  Supplementary observation, no hypothesis attached: rerun a few
           prompts keeping every mean position vector, then report the average
           cosine between repetitions k apart for k = 1 to 8. Because it reruns
           the trajectory, it takes the precision and the weights revision from
           the loop's results file, as the states stage does. A state that has
           stopped moving scores about 1.0 at every k; a state that alternates
           between p values scores about 1.0 only at multiples of p; a state
           that wanders scores below 1.0 everywhere and falls off with k.

Usage: python3 run_exp018.py --stage probe|loop|states [--arm bare|chat] ...
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from qwen_port import (  # noqa: E402
    LoopConfig, chat_wrap, free_gb, hook_names, load_model, make_injection_hook,
    natural_profile, peak_gb, position_collapse, readout, resolve_revision,
    rss_gb, run_loop, tokenise, versions,
)

OUT = HERE / "output"
SUBSET = (HERE / ".." / "exp_010c_windows" / "output"
          / "prompt_subset_small.json").resolve()

# Layers scored for the J-space test (H19b). The paper's workspace band is 38
# to 92 percent of depth; on 28 layers that is layers 11 to 25. Layers 2 and 5
# are carried as early-layer contrast, where the paper reports the lens finds
# little.
BAND_LAYERS = list(range(11, 26))
EARLY_LAYERS = [2, 5]
SCORED_LAYERS = sorted(EARLY_LAYERS + BAND_LAYERS)

# The two precisions this port supports. `--dtype` defaults to nothing rather
# than to a name, so a stage can tell "the operator asked for float32" from
# "the operator asked for nothing", which the states stage needs.
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}
DEFAULT_DTYPE = "float32"


def dtype_name(args) -> str:
    """The precision this invocation asked for, or the default if it asked for none."""
    return args.dtype or DEFAULT_DTYPE


def atomic_write(path: Path, write) -> None:
    """Write a file through a temporary name in the same directory, then rename.

    `write` is called with the temporary path. Renaming within one directory
    replaces the old file in a single step, so a crash part way through a write
    leaves the previous complete checkpoint in place instead of a half-written
    file that a later resume would read as if it were whole. The temporary name
    keeps the real suffix because `numpy.savez_compressed` appends `.npz` to
    any name that lacks it.
    """
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    try:
        write(tmp)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def load_prompts(n: int | None = None, arm: str = "bare",
                 n_chat: int = 5) -> list[dict]:
    """The registered Small subset, in file order.

    The main arm runs all 25. The pilot arm runs the first `n_chat` of the same
    25, which the specification fixes at 5, unless `--n-prompts` overrides it.
    """
    records = json.loads(SUBSET.read_text())
    if n is None and arm == "chat":
        n = n_chat
    if n is not None:
        records = records[:n]
    return records


def stage_probe(args) -> None:
    """Feasibility, memory, timing, and the natural-loudness recording pass."""
    t_start = time.time()
    dtype = dtype_name(args)
    print(f"free={free_gb():.1f} GB before load", flush=True)
    t0 = time.time()
    model = load_model(dtype=DTYPES[dtype], revision=args.revision)
    load_s = time.time() - t0
    print(f"loaded in {load_s:.1f}s  rss={rss_gb():.2f} GB peak={peak_gb():.2f} GB "
          f"free={free_gb():.1f} GB", flush=True)
    inject_name, extract_name = hook_names(model)
    print(f"hooks: inject={inject_name} extract={extract_name}", flush=True)

    records = load_prompts()
    out = {
        "model": "Qwen/Qwen3-1.7B",
        "model_revision": resolve_revision(args.revision),
        "dtype": dtype,
        "versions": versions(),
        "cfg": {k: getattr(model.cfg, k, None) for k in
                ("n_layers", "d_model", "n_heads", "n_key_value_heads", "d_vocab",
                 "normalization_type", "positional_embedding_type", "d_mlp", "act_fn")},
        "load_seconds": round(load_s, 1),
        "rss_after_load_gb": round(rss_gb(), 3),
        "peak_after_load_gb": round(peak_gb(), 3),
        "hook_names": [inject_name, extract_name],
        "prompts": {},
        "round_trip_checks": {},
        "timing": {},
    }

    # --- round-trip checks: the two facts the port stands on -----------------
    tok0 = tokenise(model, records[0]["prompt"])
    with torch.no_grad():
        logits, cache = model.run_with_cache(
            tok0, names_filter=lambda n: n in (inject_name, extract_name))
        pre = cache[inject_name][0].float().clone()
        post = cache[extract_name][0].float().clone()
        del cache
        ro = (model.ln_final(post.to(model.W_U.dtype)) @ model.W_U).float()
        out["round_trip_checks"]["readout_vs_model_logits_max_abs"] = float(
            (ro - logits[0].float()).abs().max())
        out["round_trip_checks"]["logit_scale_max_abs"] = float(logits.float().abs().max())
        model.add_hook(inject_name, make_injection_hook(pre))
        try:
            l_id = model(tok0)
        finally:
            model.reset_hooks()
        out["round_trip_checks"]["identity_injection_max_abs"] = float(
            (l_id.float() - logits.float()).abs().max())
        model.add_hook(inject_name, make_injection_hook(pre.flip(0).contiguous()))
        try:
            l_fl = model(tok0)
        finally:
            model.reset_hooks()
        out["round_trip_checks"]["reversed_injection_max_abs"] = float(
            (l_fl.float() - logits.float()).abs().max())
    print("round-trip checks:", json.dumps(out["round_trip_checks"], indent=2), flush=True)

    # --- natural loudness, both arms ----------------------------------------
    for arm in ("bare", "chat"):
        rows = records if arm == "bare" else records[:args.n_chat]
        for rec in rows:
            text = rec["prompt"] if arm == "bare" else chat_wrap(model, rec["prompt"])
            tokens = tokenise(model, text)
            prof = natural_profile(model, tokens)
            key = f"{arm}:{rec['id']}"
            out["prompts"][key] = {
                "arm": arm, "id": rec["id"], "category": rec["category"],
                "prompt": rec["prompt"], "n_tokens": int(tokens.shape[1]),
                "natural": prof,
            }
            l0 = prof["blocks.0.hook_resid_pre"]
            print(f"  {key:34s} T={tokens.shape[1]:3d} L0 all={l0['all']:.2f} "
                  f"pos0={l0['pos0']:.2f} excl0={l0['excl0']:.2f} "
                  f"ratio={l0['pos0']/l0['excl0']:.3f}", flush=True)

    # --- timing: real gated passes at both arm lengths ----------------------
    for arm, text in (("bare", records[0]["prompt"]),
                      ("chat", chat_wrap(model, records[0]["prompt"]))):
        tokens = tokenise(model, text)
        cfg = LoopConfig(max_iter=args.time_iters, check_start=10**9)
        t0 = time.time()
        r = run_loop(model, tokens, cfg)
        dt = time.time() - t0
        out["timing"][arm] = {
            "n_tokens": int(tokens.shape[1]),
            "iters": args.time_iters,
            "seconds_total": round(dt, 2),
            "seconds_per_pass": round(dt / args.time_iters, 4),
        }
        print(f"timing {arm}: {tokens.shape[1]} tokens, "
              f"{dt/args.time_iters:.3f} s/pass", flush=True)
        print(f"  after {args.time_iters} iters: collapse_all="
              f"{r['pos_collapse_all_terminal']:.4f} "
              f"collapse_excl0={r['pos_collapse_excl0_terminal']:.4f} "
              f"cos_lag2={r['final_cos_mean_lag2']:.6f} "
              f"top={r['readout']['top_token_strings'][0]!r}", flush=True)
        out["timing"][arm]["smoke_terminal"] = {
            k: r[k] for k in ("pos_collapse_all_terminal", "pos_collapse_excl0_terminal",
                              "final_cos_mean_lag2", "seed_over_natural_excl0")}
        out["timing"][arm]["smoke_readout"] = r["readout"]
        out["timing"][arm]["smoke_trace"] = r["trace"]

    out["peak_gb"] = round(peak_gb(), 3)
    out["free_gb_at_end"] = round(free_gb(), 2)
    out["wall_seconds"] = round(time.time() - t_start, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    # The precision is in the filename because both precisions are probed and
    # both files are committed; one shared name let the second run overwrite
    # the first and needed a rename by hand afterwards.
    probe_path = OUT / f"probe_natural_norms_{dtype}.json"
    probe_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {probe_path}  peak={peak_gb():.2f} GB", flush=True)


def check_resume_compatible(prior: dict, arm: str, dtype: str, revision: str,
                            cfg_vars: dict) -> None:
    """Stop a resume that would mix two differently configured runs in one file.

    The saved file carries the precision, the weights revision and every loop
    parameter of the invocation that wrote it. Resuming with any of those
    changed would leave one results file holding records made under two
    settings and labelled with only the second, so any contradiction stops the
    run. A field the saved file does not carry cannot be compared and is
    reported rather than counted as agreement, with one exception: a saved file
    that carries no weights revision at all has already been settled by
    `resume_revision_plan`, which runs before the model is loaded and either
    refuses the resume or takes the operator's confirmed legacy revision, so it
    is not reported again here.
    """
    clashes, unknown = [], []
    for name, now in (("dtype", dtype), ("model_revision", revision)):
        before = prior.get(name)
        if before is None:
            if name != "model_revision":
                unknown.append(name)
        elif before != now:
            clashes.append(f"{name}: the saved records were made with "
                           f"{before!r}, this run would use {now!r}")
    before_cfg = prior.get("loop_config")
    if not before_cfg:
        unknown.append("loop_config")
    else:
        for key, now in sorted(cfg_vars.items()):
            if key not in before_cfg:
                unknown.append(f"loop_config.{key}")
            elif before_cfg[key] != now:
                clashes.append(f"loop_config.{key}: the saved records were "
                               f"made with {before_cfg[key]!r}, this run "
                               f"would use {now!r}")
    if clashes:
        raise SystemExit(
            f"refusing to resume results_{arm}.json: this invocation does not "
            f"match the one that wrote it.\n  " + "\n  ".join(clashes) +
            f"\nResuming would put records made under two settings in one "
            f"file labelled with only this one. Rerun with the saved settings, "
            f"or move results_{arm}.json and terminal_states_{arm}.npz aside "
            f"and start the arm again.")
    if unknown:
        print(f"resume: the saved file records no "
              f"{', '.join(sorted(unknown))}, so that cannot be checked "
              f"against this invocation", flush=True)


def resume_revision_plan(saved: dict, arm: str, explicit: str | None,
                         assumed: str | None, work_left: bool
                         ) -> tuple[str | None, bool, str]:
    """Which weights a resume may run on, and whether that is a record or an
    assumption. Runs before the model is loaded, so a refusal costs nothing.

    A revision is the 40-character commit identifier naming one exact version of
    the model's files on the Hugging Face hub. A resume adds new prompt records
    to records made earlier, and the file it writes carries one revision for all
    of them, so every record in it has to have been made on the same weights.

    The hazard this closes is a results file written before the runner recorded
    the field, which is the state both committed files are in. Such a file names
    no weights at all, so nothing contradicts anything; the resume would run the
    missing prompts on whatever the cache pointer names today, mix those records
    in with the earlier ones, and label the whole file with today's revision
    alone. So a legacy resume with work left to do requires
    `--assume-legacy-revision`, which names the weights the earlier prompts were
    run at, pins this run's load to them, and is written into the file as an
    assumption rather than a measurement.

    Returns the revision to pin the load to, None meaning "do not pin, resolve
    the pointer after loading", whether the revision this run writes is an
    assumption, and one sentence saying where it came from.
    """
    recorded = saved.get("model_revision")
    if recorded:
        if assumed and assumed != recorded:
            raise SystemExit(
                f"--assume-legacy-revision {assumed} contradicts "
                f"results_{arm}.json, which records that its prompts were run "
                f"at {recorded}. The option is for a file that records no "
                f"revision at all; this one does, so drop the option.")
        if explicit and explicit != recorded:
            raise SystemExit(
                f"--revision {explicit} contradicts results_{arm}.json, which "
                f"records that the prompts already in it were run at "
                f"{recorded}. Resuming would add records made on one version of "
                f"the weights to records made on another and label the file "
                f"with one of them. Drop the flag to resume on the recorded "
                f"weights, or move results_{arm}.json and "
                f"terminal_states_{arm}.npz aside and start the arm again.")
        if assumed:
            print(f"resume: --assume-legacy-revision was not needed, because "
                  f"results_{arm}.json records revision {recorded} itself",
                  flush=True)
        # Pin to what the file records, so the prompts still to run go through
        # the same weights as the ones already in it rather than through
        # whatever an unpinned load fetches today.
        was_assumed = bool(saved.get("model_revision_assumed"))
        return recorded, was_assumed, (
            f"the revision results_{arm}.json records for the prompts already "
            f"in it, which this resume pinned its load to"
            + (", itself confirmed by hand on an earlier resume rather than "
               "measured at the time" if was_assumed else ""))
    if not work_left:
        print(f"resume: results_{arm}.json records no weights revision, and "
              f"every prompt in it is already finished, so this invocation has "
              f"nothing to run and writes nothing. A resume that did have work "
              f"to do would need --assume-legacy-revision naming the weights "
              f"those records were made on.", flush=True)
        return explicit, False, "no record was written by this invocation"
    if not assumed:
        raise SystemExit(
            f"refusing to resume results_{arm}.json: it records no weights "
            f"revision, so nothing here can tell which version of the model's "
            f"files its existing records were made on, and there are prompts "
            f"left to run. Running them on whatever the cache pointer "
            f"refs/main names today would mix two versions of the weights in "
            f"one file and label it with today's version alone.\n"
            f"Pass --assume-legacy-revision <40-character identifier> naming "
            f"the weights the existing records were made on. It pins this "
            f"run's load to those weights and is written into the file as "
            f"assumed rather than recorded. Both results files committed with "
            f"EXP_018 are in this state, and this experiment's record names "
            f"the revision to pass for them.\n"
            f"The alternative is to move results_{arm}.json and "
            f"terminal_states_{arm}.npz aside and run the arm from the start.")
    if explicit and explicit != assumed:
        raise SystemExit(
            f"--revision {explicit} contradicts --assume-legacy-revision "
            f"{assumed}. The second says the records already in "
            f"results_{arm}.json were made on {assumed}, and the first would "
            f"run the remaining prompts on {explicit}, which would put two "
            f"versions of the weights in one file. Pass one revision, or "
            f"neither and start the arm again.")
    return assumed, True, (
        f"the --assume-legacy-revision option: results_{arm}.json recorded no "
        f"revision, so the records inherited from it are taken to have been "
        f"made on this one, confirmed by hand rather than measured at the time, "
        f"and this run's load was pinned to it")


def stage_loop(args) -> None:
    """The registered run for one arm."""
    t_start = time.time()
    dtype = dtype_name(args)
    cfg = LoopConfig(max_iter=args.max_iter, check_start=args.check_start,
                     check_every=args.check_every, seed=args.seed)
    records = load_prompts(args.n_prompts, args.arm, args.n_chat)
    OUT.mkdir(parents=True, exist_ok=True)
    res_path = OUT / f"results_{args.arm}.json"
    npz_path = OUT / f"terminal_states_{args.arm}.npz"
    results, tensors, saved = [], {}, None
    pin, revision_assumed = args.revision, False
    revision_source = ("the --revision option, which pinned the load"
                       if args.revision else
                       "the cache pointer refs/main that this unpinned load "
                       "followed, read after the load")
    # Everything a resume can refuse is settled before the model is loaded, so
    # a refusal costs no minutes and no gigabytes.
    if args.resume and res_path.exists():
        saved = json.loads(res_path.read_text())
        prior = saved["records"]
        if npz_path.exists():
            with np.load(npz_path) as npz:
                tensors = {k: v for k, v in npz.items()}
        # A prompt counts as finished only when both checkpoints hold it: its
        # row in the results file and its terminal state in the state file.
        # A prompt held by only one of them is run again, because the states
        # stage needs the terminal state and would fail on a missing one.
        done = {r["id"] for r in prior} & set(tensors)
        half = [r["id"] for r in prior if r["id"] not in done]
        results = [r for r in prior if r["id"] in done]
        tensors = {k: v for k, v in tensors.items() if k in done}
        if half:
            print(f"resume: rerunning {len(half)} prompt(s) present in one "
                  f"checkpoint but not the other: {', '.join(half)}", flush=True)
        records = [r for r in records if r["id"] not in done]
        print(f"resume: {len(done)} done, {len(records)} to go", flush=True)
        pin, revision_assumed, revision_source = resume_revision_plan(
            saved, args.arm, args.revision, args.assume_legacy_revision,
            bool(records))
    elif args.assume_legacy_revision:
        print(f"note: --assume-legacy-revision names the weights that records "
              f"already in a results file were made on, and this invocation is "
              f"not resuming one, so the option has nothing to apply to and is "
              f"ignored", flush=True)

    model = load_model(dtype=DTYPES[dtype], revision=pin)
    revision = resolve_revision(pin)
    print(f"loaded rss={rss_gb():.2f} GB peak={peak_gb():.2f} GB "
          f"dtype={dtype} revision={revision}"
          f"{' (assumed for the inherited records)' if revision_assumed else ''}",
          flush=True)
    if saved is not None:
        check_resume_compatible(saved, args.arm, dtype, revision, vars(cfg))

    for n, rec in enumerate(records, 1):
        text = rec["prompt"] if args.arm == "bare" else chat_wrap(model, rec["prompt"])
        tokens = tokenise(model, text)
        t0 = time.time()
        r = run_loop(model, tokens, cfg, verbose=args.verbose)
        dt = time.time() - t0
        term = r.pop("terminal_tensor")
        tensors[rec["id"]] = term.numpy().astype(np.float32)
        r.update({"id": rec["id"], "category": rec["category"],
                  "prompt": rec["prompt"], "arm": args.arm,
                  "input_text": text, "seconds": round(dt, 1)})
        results.append(r)
        print(f"[{n}/{len(records)}] {rec['id']:20s} T={r['n_tokens']:3d} "
              f"lock={r['lock_in_iter']} iters={r['n_iters']} "
              f"collapse_all={r['pos_collapse_all_terminal']:.4f} "
              f"collapse_ex0={r['pos_collapse_excl0_terminal']:.4f} "
              f"top={r['readout']['top_token_strings'][0]!r} "
              f"({dt:.0f}s, {dt/r['n_iters']:.2f} s/pass, "
              f"rss={rss_gb():.1f} GB, free={free_gb():.1f} GB)", flush=True)
        payload = {"arm": args.arm, "model": "Qwen/Qwen3-1.7B",
                   "model_revision": revision,
                   "model_revision_assumed": revision_assumed,
                   "model_revision_source": revision_source, "dtype": dtype,
                   "loop_config": vars(cfg), "versions": versions(),
                   "scored_layers": SCORED_LAYERS,
                   "wall_seconds": round(time.time() - t_start, 1),
                   "peak_gb": round(peak_gb(), 3), "records": results}
        # States first, then the results row, each written through a temporary
        # file and renamed. A crash between the two leaves a state with no row,
        # which the resume above reruns; the reverse order would leave a row
        # with no state, which it would also rerun, so either order is safe and
        # neither can be read as half a prompt.
        atomic_write(npz_path, lambda p: np.savez_compressed(p, **tensors))
        atomic_write(res_path, lambda p: p.write_text(json.dumps(payload, indent=2)))
    print(f"\narm {args.arm} done in {(time.time()-t_start)/60:.1f} min, "
          f"peak={peak_gb():.2f} GB", flush=True)


def _scored_positions(n_tokens: int) -> list[int]:
    """Three token positions per prompt, position 0 always excluded.

    Position 0 is left out because this port's whole loudness convention leaves
    it out; the other two are the middle and the last position, which is the
    one the readout reads.
    """
    if n_tokens <= 3:
        return list(range(1, n_tokens))
    return sorted({1, n_tokens // 2, n_tokens - 1})


def dtype_from_results(args, res: dict, stage: str) -> str:
    """The precision a follow-on stage loads the model in.

    The loop records the precision it ran in, and a state or a trajectory
    produced at a different precision is not the one the loop visited, so the
    recorded value wins when no `--dtype` is given and a `--dtype` that
    contradicts it stops the run unless `--allow-dtype-mismatch` says the
    contradiction is deliberate.
    """
    recorded = res.get("dtype")
    if args.dtype is None:
        if recorded in DTYPES:
            return recorded
        raise SystemExit(
            f"results_{args.arm}.json records no precision this runner knows "
            f"(found {recorded!r}), so the {stage} stage cannot tell which "
            f"precision the loop ran in; pass --dtype float32 or "
            f"--dtype bfloat16 explicitly")
    if (recorded in DTYPES and recorded != args.dtype
            and not args.allow_dtype_mismatch):
        raise SystemExit(
            f"--dtype {args.dtype} contradicts the {recorded} recorded in "
            f"results_{args.arm}.json; what the {stage} stage produces at a "
            f"precision the loop never ran in is not comparable with it. Drop "
            f"the flag to use {recorded}, or pass --allow-dtype-mismatch to "
            f"override")
    return args.dtype


def revision_from_results(res: dict, stage: str,
                          explicit: str | None = None) -> tuple[str, str, bool]:
    """The revision to pin a follow-on stage to, the revision to record, and
    whether that revision is an assumption rather than a measurement.

    A revision is the 40-character commit identifier naming one exact version
    of the model's files on the Hugging Face hub. A stage that runs saved
    tensors back through the model has to run them through the weights that
    produced them, so it always pins its load: to `--revision` when one is
    given, otherwise to the revision the loop recorded. When neither is
    available, as in the runs made before the runner recorded the field, the
    stage stops here instead of loading whatever the cache pointer `refs/main`
    names today, because that pointer can name weights the loop never used and
    the metadata written afterwards would then label mismatched states with a
    revision they did not come from. A pointer that disagrees with the pin is
    named rather than followed.

    The third return value says whether the revision was confirmed by hand
    rather than recorded by the run, which is the case whenever `--revision`
    supplies what the results file does not, and it stays true down the chain
    if the results file was itself labelled that way. Without it an operator's
    assumption would be written into this stage's metadata as a plain field and
    read by the next stage as a measurement.
    """
    recorded = res.get("model_revision")
    arm = res.get("arm", "?")
    if explicit and recorded and explicit != recorded:
        raise SystemExit(
            f"--revision {explicit} contradicts the {recorded} recorded in "
            f"results_{arm}.json; the saved tensors came out of {recorded}, "
            f"so running them through {explicit} would put two versions of "
            f"the weights in one measurement. Drop the flag to use the "
            f"recorded revision.")
    pin = explicit or recorded
    if not pin:
        raise SystemExit(
            f"results_{arm}.json records no weights revision, so the {stage} "
            f"stage cannot tell which version of the weights produced the "
            f"saved tensors. Pass --revision <40-character identifier> naming "
            f"the weights the loop ran; the results record states it for the "
            f"committed runs, which were made before the runner wrote this "
            f"field. Following the cache pointer instead would risk running "
            f"the saved tensors through weights the loop never used and then "
            f"recording the new revision as if it had.")
    try:
        pointer = resolve_revision()
    except FileNotFoundError as exc:
        pointer = None
        print(f"note: {exc}", flush=True)
    if pointer and pointer != pin:
        print(f"note: the cache pointer names revision {pointer}, the {stage} "
              f"stage is pinned to {pin} "
              f"({'the --revision option' if explicit else 'the loop record'}), "
              f"so the weights match the saved tensors", flush=True)
    assumed = bool(res.get("model_revision_assumed")) or not recorded
    if assumed:
        print(f"note: revision {pin} is an assumption and not a measurement: it "
              f"was confirmed by hand rather than recorded by the run that made "
              f"the saved tensors, and this stage's metadata says so", flush=True)
    return pin, pin, assumed


def stage_states(args) -> None:
    """Per-layer states for H19b: settled tensors re-injected, and clean passes."""
    t_start = time.time()
    # The results file is read before the model is loaded, because it is what
    # says which precision to load.
    res = json.loads((OUT / f"results_{args.arm}.json").read_text())
    dtype = dtype_from_results(args, res, "states")
    pin, revision, rev_assumed = revision_from_results(
        res, "states", args.revision)
    print(f"dtype={dtype} (loop recorded {res.get('dtype')!r})  "
          f"revision={revision} (pinned"
          f"{', assumed' if rev_assumed else ''})", flush=True)
    model = load_model(dtype=DTYPES[dtype], revision=pin)
    inject_name, _ = hook_names(model)
    tensors = np.load(OUT / f"terminal_states_{args.arm}.npz")
    want = {f"blocks.{l}.hook_resid_post" for l in SCORED_LAYERS}
    settled, clean, meta = {}, {}, []

    for rec in res["records"]:
        tokens = tokenise(model, rec["input_text"])
        pos = _scored_positions(int(tokens.shape[1]))
        x = torch.from_numpy(tensors[rec["id"]]).float()
        # The saved tensor is the state as it came OUT of the last layer, before
        # the loop's rescale. Inject what the loop itself would inject next:
        # the same tensor pulled back to the prompt's natural layer-0 loudness
        # over positions 1 and later. Injecting it unscaled would be a shout of
        # about two thousand times natural strength and would measure a regime
        # the loop never visits.
        x = x * (rec["target_norm_natural_excl0"] / float(x[1:].norm()))
        # settled: inject the settled tensor at the layer-0 entry, read every layer
        model.add_hook(inject_name, make_injection_hook(x))
        try:
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    tokens, names_filter=lambda n: n in want)
        finally:
            model.reset_hooks()
        for l in SCORED_LAYERS:
            settled[f"{rec['id']}|{l}"] = (
                cache[f"blocks.{l}.hook_resid_post"][0][pos].float().numpy())
        del cache
        # clean: the ordinary, non-iterated pass on the same prompt
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in want)
        for l in SCORED_LAYERS:
            clean[f"{rec['id']}|{l}"] = (
                cache[f"blocks.{l}.hook_resid_post"][0][pos].float().numpy())
        del cache
        meta.append({"id": rec["id"], "n_tokens": int(tokens.shape[1]),
                     "positions": pos})
        print(f"  states {rec['id']:20s} T={tokens.shape[1]} pos={pos}", flush=True)

    outdir = Path(args.states_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outdir / f"layer_states_{args.arm}.npz",
                        **{f"settled|{k}": v for k, v in settled.items()},
                        **{f"clean|{k}": v for k, v in clean.items()})
    (outdir / f"layer_states_{args.arm}_meta.json").write_text(json.dumps(
        {"arm": args.arm, "scored_layers": SCORED_LAYERS, "dtype": dtype,
         "model": "Qwen/Qwen3-1.7B", "model_revision": revision,
         "model_revision_assumed": rev_assumed,
         "loop_dtype": res.get("dtype"),
         "loop_model_revision": res.get("model_revision"),
         "loop_model_revision_assumed": bool(res.get("model_revision_assumed")),
         "prompts": meta}, indent=2))
    print(f"wrote layer states to {outdir}, {(time.time()-t_start)/60:.1f} min, "
          f"peak={peak_gb():.2f} GB", flush=True)


@torch.no_grad()
def stage_lagscan(args) -> None:
    """Supplementary: is the non-settling trajectory periodic or wandering?

    Draws no verdicts. Uses the same instrument the registered engine uses,
    `atr_engine2.lag_scan`, on the mean position vector of the last iterations.
    """
    import torch.nn.functional as F
    # This stage reruns the loop's own trajectory rather than reading a saved
    # one, so it has to run the same weights at the same precision as the loop
    # it is describing, and has to record which it used.
    res_path = OUT / f"results_{args.arm}.json"
    if res_path.exists():
        res = json.loads(res_path.read_text())
        dtype = dtype_from_results(args, res, "lagscan")
        pin, revision, rev_assumed = revision_from_results(
            res, "lagscan", args.revision)
    else:
        dtype, pin, rev_assumed = dtype_name(args), args.revision, False
        revision = resolve_revision(args.revision)
        print(f"note: no results_{args.arm}.json to check against, so this lag "
              f"scan runs at {dtype} on revision {revision} without matching "
              f"any recorded loop", flush=True)
    print(f"dtype={dtype}  revision={revision}"
          f"{' (pinned)' if pin else ''}"
          f"{' (assumed, not recorded by the loop)' if rev_assumed else ''}",
          flush=True)
    model = load_model(dtype=DTYPES[dtype], revision=pin)
    inject_name, extract_name = hook_names(model)
    records = load_prompts(args.n_prompts, args.arm, args.n_chat)
    out = {"arm": args.arm, "iterations": args.max_iter, "max_lag": 8,
           "dtype": dtype, "model": "Qwen/Qwen3-1.7B",
           "model_revision": revision,
           "model_revision_assumed": rev_assumed, "prompts": {}}
    for rec in records:
        text = rec["prompt"] if args.arm == "bare" else chat_wrap(model, rec["prompt"])
        tokens = tokenise(model, text)
        _, cache = model.run_with_cache(
            tokens, names_filter=lambda n: n in (inject_name, extract_name))
        target = float(cache[inject_name][0].float()[1:].norm())
        x = cache[extract_name][0].float().clone()
        del cache
        means = []
        for _ in range(args.max_iter):
            x = x * (target / float(x[1:].norm()))
            model.add_hook(inject_name, make_injection_hook(x))
            try:
                _, cache = model.run_with_cache(
                    tokens, names_filter=lambda n: n == extract_name)
            finally:
                model.reset_hooks()
            x = cache[extract_name][0].float().clone()
            del cache
            means.append(x.mean(dim=0).clone())
        tail = torch.stack(means[-args.lag_window:])
        scan = {}
        for k in range(1, 9):
            if k < tail.shape[0]:
                scan[k] = float(F.cosine_similarity(tail[k:], tail[:-k], dim=-1).mean())
        out["prompts"][rec["id"]] = {"n_tokens": int(tokens.shape[1]),
                                     "lag_window": int(tail.shape[0]),
                                     "lag_scan": {str(k): round(v, 6)
                                                  for k, v in scan.items()}}
        print(f"  {rec['id']:20s} " + "  ".join(
            f"k={k}:{v:.4f}" for k, v in scan.items()), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"lagscan_{args.arm}.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT / f'lagscan_{args.arm}.json'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["probe", "loop", "states", "lagscan"])
    ap.add_argument("--arm", default="bare", choices=["bare", "chat"])
    ap.add_argument("--dtype", default=None, choices=["float32", "bfloat16"],
                    help="model precision; defaults to float32 for probe and "
                         "loop, and to the precision recorded in the results "
                         "file for states and lagscan")
    ap.add_argument("--revision", default=None,
                    help="pin the load to one exact version of the model's "
                         "files on the Hugging Face hub, given as its "
                         "40-character identifier; required by --stage states "
                         "and --stage lagscan when the results file records "
                         "none, as the committed runs do")
    ap.add_argument("--assume-legacy-revision", default=None,
                    help="for --stage loop --resume only: the 40-character "
                         "identifier of the weights the prompt records already "
                         "in the results file were run on, confirmed by hand "
                         "for a file written before the runner recorded that "
                         "field, as both results files committed with EXP_018 "
                         "were. It pins this run's load to those weights and is "
                         "written into the file as assumed rather than recorded")
    ap.add_argument("--allow-dtype-mismatch", action="store_true",
                    help="let --stage states or --stage lagscan run at a "
                         "precision the loop did not record")
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--check-start", type=int, default=10)
    ap.add_argument("--check-every", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-prompts", type=int, default=None)
    ap.add_argument("--n-chat", type=int, default=5)
    ap.add_argument("--time-iters", type=int, default=10)
    ap.add_argument("--states-dir", default=str(HERE / "_states"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--lag-window", type=int, default=40)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    {"probe": stage_probe, "loop": stage_loop, "states": stage_states,
     "lagscan": stage_lagscan}[args.stage](args)


if __name__ == "__main__":
    main()
