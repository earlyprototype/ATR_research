"""EXP_018 runner: the ATR loop on Qwen3-1.7B. Spec: `../../EXP_018_SPEC.md`.

Stages, in the order they are meant to run:

  probe    Load the model, measure resident memory and per-pass time, record
           the natural per-layer entry loudness of every prompt in both arms,
           and time a short loop. These numbers set the budget written into
           the spec, so this stage runs before the spec is committed.
  loop     The registered run. `--arm bare` is the main arm (bare text, the
           25-prompt Small subset); `--arm chat` is the pilot arm (the first
           five of the same prompts wrapped as a user turn, thinking mode off).
  states   Per-layer states for the J-space test (H19b): inject each settled
           tensor at the layer-0 entry and read every scored layer, and run the
           same prompts clean for the comparison arm.

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
    natural_profile, peak_gb, position_collapse, readout, rss_gb, run_loop,
    tokenise, versions,
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


def load_prompts(n: int | None = None, arm: str = "bare") -> list[dict]:
    records = json.loads(SUBSET.read_text())
    if n is not None:
        records = records[:n]
    return records


def stage_probe(args) -> None:
    """Feasibility, memory, timing, and the natural-loudness recording pass."""
    t_start = time.time()
    print(f"free={free_gb():.1f} GB before load", flush=True)
    t0 = time.time()
    model = load_model(dtype=torch.float32 if args.dtype == "float32"
                       else torch.bfloat16)
    load_s = time.time() - t0
    print(f"loaded in {load_s:.1f}s  rss={rss_gb():.2f} GB peak={peak_gb():.2f} GB "
          f"free={free_gb():.1f} GB", flush=True)
    inject_name, extract_name = hook_names(model)
    print(f"hooks: inject={inject_name} extract={extract_name}", flush=True)

    records = load_prompts()
    out = {
        "model": "Qwen/Qwen3-1.7B",
        "dtype": args.dtype,
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
    (OUT / "probe_natural_norms.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT/'probe_natural_norms.json'}  peak={peak_gb():.2f} GB", flush=True)


def stage_loop(args) -> None:
    """The registered run for one arm."""
    t_start = time.time()
    model = load_model(dtype=torch.float32 if args.dtype == "float32"
                       else torch.bfloat16)
    print(f"loaded rss={rss_gb():.2f} GB peak={peak_gb():.2f} GB", flush=True)
    cfg = LoopConfig(max_iter=args.max_iter, check_start=args.check_start,
                     check_every=args.check_every, seed=args.seed)
    records = load_prompts(args.n_prompts)
    OUT.mkdir(parents=True, exist_ok=True)
    res_path = OUT / f"results_{args.arm}.json"
    npz_path = OUT / f"terminal_states_{args.arm}.npz"
    results, tensors = [], {}
    if args.resume and res_path.exists():
        results = json.loads(res_path.read_text())["records"]
        done = {r["id"] for r in results}
        if npz_path.exists():
            tensors = {k: v for k, v in np.load(npz_path).items()}
        records = [r for r in records if r["id"] not in done]
        print(f"resume: {len(done)} done, {len(records)} to go", flush=True)

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
        payload = {"arm": args.arm, "model": "Qwen/Qwen3-1.7B", "dtype": args.dtype,
                   "loop_config": vars(cfg), "versions": versions(),
                   "scored_layers": SCORED_LAYERS,
                   "wall_seconds": round(time.time() - t_start, 1),
                   "peak_gb": round(peak_gb(), 3), "records": results}
        res_path.write_text(json.dumps(payload, indent=2))
        np.savez_compressed(npz_path, **tensors)
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


def stage_states(args) -> None:
    """Per-layer states for H19b: settled tensors re-injected, and clean passes."""
    t_start = time.time()
    model = load_model(dtype=torch.float32 if args.dtype == "float32"
                       else torch.bfloat16)
    inject_name, _ = hook_names(model)
    res = json.loads((OUT / f"results_{args.arm}.json").read_text())
    tensors = np.load(OUT / f"terminal_states_{args.arm}.npz")
    want = {f"blocks.{l}.hook_resid_post" for l in SCORED_LAYERS}
    settled, clean, meta = {}, {}, []

    for rec in res["records"]:
        tokens = tokenise(model, rec["input_text"])
        pos = _scored_positions(int(tokens.shape[1]))
        x = torch.from_numpy(tensors[rec["id"]]).float()
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
        {"arm": args.arm, "scored_layers": SCORED_LAYERS, "prompts": meta}, indent=2))
    print(f"wrote layer states to {outdir}, {(time.time()-t_start)/60:.1f} min, "
          f"peak={peak_gb():.2f} GB", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["probe", "loop", "states"])
    ap.add_argument("--arm", default="bare", choices=["bare", "chat"])
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--check-start", type=int, default=10)
    ap.add_argument("--check-every", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-prompts", type=int, default=None)
    ap.add_argument("--n-chat", type=int, default=5)
    ap.add_argument("--time-iters", type=int, default=10)
    ap.add_argument("--states-dir", default=str(HERE / "_states"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    {"probe": stage_probe, "loop": stage_loop, "states": stage_states}[args.stage](args)


if __name__ == "__main__":
    main()
