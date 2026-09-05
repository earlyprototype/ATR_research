"""EXP_017 Part 1: the registered full-stack loop on the twin and on base GPT-2 Small.

Spec: ../../EXP_017_SPEC.md section 5. Reproduces EXP_010b arm SB (inject at
blocks.0.hook_resid_pre, extract at blocks.11.hook_resid_post, renorm seed_j)
with the lag-2 gate, on the committed 25-prompt Small subset.

The gated protocol itself is NOT reimplemented here: it lives only in
../atr_engine2.py run_atr_gated, which this script calls with the spec's
parameters. The one addition is a permanent hook on the extraction point that
keeps the most recent full residual tensor, so the terminal tensor (all token
positions) is available for the per-layer probe of Part 2. TransformerLens
reset_hooks() leaves permanent hooks in place, so the engine's own hook
handling is untouched.

Usage:
    python3 run_loop.py --model twin
    python3 run_loop.py --model base
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(HERE))

import exp017_models  # noqa: E402
from atr_engine2 import run_atr_gated, get_top_tokens  # noqa: E402

SUBSET = EXPERIMENTS / "exp_010c_windows" / "output" / "prompt_subset_small.json"
OUT = HERE / "output"

# Spec section 5.1: every loop parameter, fixed.
LOOP = dict(layer_start=0, layer_end=11, max_iter=1000, threshold=0.999,
            patience=3, check_every=10, check_start=100, gate_lag=2,
            renorm="seed_j")
MODELS = exp017_models.MODELS


def load_model(which):
    """Load into TransformerLens by the lucier pilot's offline-weights route.

    The weights and the tokenizer are pinned to the revision, that is the
    repository commit, recorded for this model in exp017_models.py, so that a
    later change on Hugging Face cannot alter what this loads.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformer_lens import HookedTransformer
    name, rev = MODELS[which], exp017_models.revision(which)
    hf = AutoModelForCausalLM.from_pretrained(name, revision=rev)
    tok = AutoTokenizer.from_pretrained(name, revision=rev)
    model = HookedTransformer.from_pretrained("gpt2", hf_model=hf, tokenizer=tok,
                                              device="cpu")
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    subset = json.load(open(SUBSET))
    assert len(subset) == 25, f"expected 25 prompts, got {len(subset)}"
    torch.manual_seed(args.seed)

    print(f"EXP_017 loop | model={args.model} ({MODELS[args.model]} at "
          f"revision {exp017_models.revision(args.model)}) "
          f"threads={torch.get_num_threads()} seed={args.seed}", flush=True)
    print(f"params: {LOOP}", flush=True)
    model = load_model(args.model)
    print(f"loaded: n_layers={model.cfg.n_layers} d_model={model.cfg.d_model} "
          f"d_vocab={model.cfg.d_vocab}", flush=True)

    # ---- natural per-layer entry sizes, one un-hooked pass per prompt -------
    # Same procedure as run_exp010c.py --record-natural-norms.
    natural = {}
    for rec in subset:
        with torch.no_grad():
            _, cache = model.run_with_cache(
                rec["prompt"], names_filter=lambda n: n.endswith("hook_resid_pre"))
        natural[rec["id"]] = {
            str(l): round(cache[f"blocks.{l}.hook_resid_pre"][0].norm().item(), 4)
            for l in range(model.cfg.n_layers)}
    (OUT / f"natural_resid_norms_{args.model}.json").write_text(
        json.dumps(natural, indent=2))
    print(f"natural entry sizes recorded -> natural_resid_norms_{args.model}.json",
          flush=True)

    # ---- permanent capture of the full tensor at the extraction point ------
    read_hook = f"blocks.{LOOP['layer_end']}.hook_resid_post"
    box = {}

    def capture(resid, hook):
        box["tensor"] = resid[0].detach().clone()
        return resid

    model.add_perma_hook(read_hook, capture)

    results, terminals = [], {}
    t_start = time.time()
    for n, rec in enumerate(subset, 1):
        pid, prompt = rec["id"], rec["prompt"]
        t0 = time.time()
        r = run_atr_gated(model, prompt, capture_terminal=True, **LOOP)
        term_full = box["tensor"]
        mean_vec = r.pop("terminal_mean_vec")
        last_vec = r.pop("terminal_last_vec")
        # The captured tensor must be the one the engine ended on.
        assert torch.allclose(term_full.mean(dim=0), mean_vec, atol=1e-5), \
            f"captured tensor disagrees with the engine's terminal for {pid}"
        top5 = get_top_tokens(model, last_vec, k=5)
        r["lag_scan"] = ({str(k): v for k, v in r["lag_scan"].items()}
                         if r.get("lag_scan") else None)
        r.update(prompt_id=pid, category=rec["category"], prompt=prompt,
                 model=args.model, window=f"{LOOP['layer_start']}->{LOOP['layer_end']}",
                 gate_lag=LOOP["gate_lag"],
                 top5_tokens=[t for t, _ in top5],
                 top5_probs=[float(p) for _, p in top5],
                 terminal_prob=float(r["terminal_prob"]),
                 natural_entry_norm_layer0=natural[pid]["0"],
                 loudness_ratio=r["target_norm"] / natural[pid]["0"],
                 terminal_tensor_norm=float(term_full.norm()),
                 seq_len=int(term_full.shape[0]),
                 seconds=round(time.time() - t0, 1))
        results.append(r)
        terminals[f"{pid}|mean"] = mean_vec.numpy().astype(np.float32)
        terminals[f"{pid}|last"] = last_vec.numpy().astype(np.float32)
        terminals[f"{pid}|full"] = term_full.numpy().astype(np.float32)
        print(f"  [{n:>2}/25] {pid:<16} lock={str(r['lock_in_iter']):>5} "
              f"conv={str(r['converged']):<5} top1={r['terminal_token']!r:<14} "
              f"p={r['terminal_prob']:.3f} loud={r['loudness_ratio']:.1f}x "
              f"{r['seconds']:.0f}s", flush=True)

    model.reset_hooks(including_permanent=True)

    (OUT / f"loop_results_{args.model}.json").write_text(json.dumps(results, indent=2))
    npz = OUT / f"terminals_{args.model}.npz"
    np.savez_compressed(npz, **terminals)
    digest = hashlib.sha256(npz.read_bytes()).hexdigest()

    conv = sum(r["converged"] for r in results)
    ratios = [r["loudness_ratio"] for r in results]
    from collections import Counter
    tokens = Counter(r["terminal_token"] for r in results)
    print(f"\n=== {args.model}: converged {conv}/25, wall "
          f"{time.time() - t_start:.0f}s ===", flush=True)
    print(f"terminal tokens: {tokens.most_common()}", flush=True)
    print(f"entry loudness ratio: mean {sum(ratios) / len(ratios):.1f}x, "
          f"range {min(ratios):.1f}-{max(ratios):.1f}x", flush=True)
    print(f"terminals -> {npz.name} sha256 {digest}", flush=True)


if __name__ == "__main__":
    main()
