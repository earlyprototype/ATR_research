"""EXP_010c — Window grid on GPT-2 Medium. Pre-registered spec: ../../EXP_010c_SPEC.md

Arms (inject i -> extract j), seeding: natural L0 prompt pass in every arm (spec §3).

Tiers:
    smoke  — 2 prompts, arms A0+A4, max_iter=60   (harness validation only)
    pilot  — 5 prompts, all arms,   max_iter=300  (directional signal, ~1-2 h CPU)
    full   — 25 prompts, all arms,  max_iter=1000 (the pre-registered run, overnight)

The gated protocol params (threshold/patience/check_every/check_start) follow the
spec for `full`; smoke/pilot shrink check_start proportionally and are RECORDED as
non-registered tiers — no verdicts are drawn from them beyond harness validity.

Usage: python run_exp010c.py --tier smoke|pilot|full [--arms A0,A4,...]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # for atr_engine2

from atr_engine2 import run_atr_gated, lag_scan, get_readout_detail  # noqa: E402
from derive_prompts import select_subset  # noqa: E402

ARMS = {
    "A0": (0, 23),   # baseline / reproduction gate
    "A4": (10, 21),  # band-exact
    "A1": (0, 11),   # placement: front
    "A2": (6, 17),   # placement: middle
    "A3": (12, 23),  # placement: back
    "A5": (8, 15),   # length probe (8 layers, mid-band)
}
ARM_ORDER = ["A0", "A4", "A1", "A2", "A3", "A5"]  # spec §5 execution order

TIERS = {
    "smoke": dict(n_prompts=2, max_iter=60, check_start=20, arms=["A0", "A4"]),
    "pilot": dict(n_prompts=5, max_iter=300, check_start=50, arms=ARM_ORDER),
    "full": dict(n_prompts=25, max_iter=1000, check_start=100, arms=ARM_ORDER),
}


def run_arm_with_terminal(model, prompt, i, j, max_iter, check_start):
    """run_atr_gated + capture of the terminal tensor (mean + last-position vectors).

    Re-runs the final injection state capture by repeating the gated loop's exact
    protocol; to avoid doubling cost we inline a light variant: run the gated loop,
    then one extra natural re-derivation is unnecessary because run_atr_gated
    classifies from its own final tensor — so instead we replicate its loop here
    with terminal capture. Kept minimal and protocol-identical.
    """
    hook_read = f"blocks.{j}.hook_resid_post"
    hook_write = f"blocks.{i}.hook_resid_pre"
    threshold, patience, check_every, gate_lag = 0.999, 3, 10, 1

    with torch.no_grad():
        _, cache = model.run_with_cache(prompt, names_filter=lambda n: n == hook_read)
    current = cache[hook_read][0].clone()
    initial_norm = current.norm().item()
    mean_history = [current.mean(dim=0).clone()]

    consecutive, lock_in, final_cos, it = 0, None, 1.0, 0
    recent = []  # last 8 mean vectors for lag_scan

    for it in range(1, max_iter + 1):
        norm = current.norm().item()
        if norm > 0:
            current = current * (initial_norm / norm)
        inject = current.clone()

        def injection_hook(resid, hook, tensor=inject):
            resid[0, :, :] = tensor
            return resid

        model.add_hook(hook_write, injection_hook)
        try:
            with torch.no_grad():
                _, cache = model.run_with_cache(prompt, names_filter=lambda n: n == hook_read)
        finally:
            model.reset_hooks()
        current = cache[hook_read][0].clone()
        mean_vec = current.mean(dim=0).clone()
        recent.append(mean_vec)
        if len(recent) > 9:
            recent.pop(0)

        if it >= check_start and it % check_every == 0:
            cos = F.cosine_similarity(mean_vec.unsqueeze(0), mean_history[0].unsqueeze(0)).item()
            final_cos = cos
            consecutive = consecutive + 1 if cos > threshold else 0
            if consecutive >= patience:
                lock_in = it
                break
        mean_history.append(mean_vec)
        if len(mean_history) > gate_lag:
            mean_history.pop(0)

    last_vec = current[-1, :].clone()
    detail = get_readout_detail(model, last_vec)
    lags = lag_scan(torch.stack(recent), max_lag=min(8, len(recent) - 1)) if len(recent) > 1 else None
    return {
        "terminal_token": detail["top_token_strings"][0],
        "terminal_token_id": detail["top_token_ids"][0],
        "terminal_prob": detail["top_token_probs"][0],
        "top_logit_margin": detail["top_logit_margin"],
        "entropy": detail["entropy"],
        "lock_in_iter": lock_in,
        "converged": lock_in is not None,
        "n_iters": it,
        "final_cos_sim_mean": final_cos,
        "lag_scan": [float(x) for x in lags] if lags is not None else None,
        "terminal_mean_vec": current.mean(dim=0).clone(),
        "terminal_last_vec": last_vec,
    }


class _DummyTokenizer:
    """Decode stub for --harness-check runs (random-init model, no real vocab)."""

    padding_side = "right"
    pad_token_id = 0

    def decode(self, ids):
        ids = ids if isinstance(ids, (list, tuple)) else [ids]
        return "".join(f"<{int(i)}>" for i in ids)


def _toy_model():
    """Random-init 24-layer model for harness validation only (no network, no
    pretrained weights). Validates hooks / windowed loop / gating / artifacts —
    NOT the D-collapse reproduction gate, which needs real gpt2-medium weights."""
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    cfg = HookedTransformerConfig(
        n_layers=24, d_model=64, n_ctx=32, d_head=16, n_heads=4,
        d_vocab=997, act_fn="gelu", normalization_type="LN",
    )
    model = HookedTransformer(cfg)
    model.tokenizer = _DummyTokenizer()
    return model


def _toy_tokens(prompt, d_vocab=997, max_len=12):
    ids = [(hash(w) % (d_vocab - 1)) + 1 for w in prompt.split()[:max_len]]
    return torch.tensor([ids or [1]], dtype=torch.long)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=TIERS, required=True)
    ap.add_argument("--arms", default=None, help="comma-separated override, e.g. A0,A4")
    ap.add_argument("--harness-check", action="store_true",
                    help="random-init toy model; validates the harness, draws no verdicts")
    args = ap.parse_args()
    tier = TIERS[args.tier]
    arms = args.arms.split(",") if args.arms else tier["arms"]

    torch.manual_seed(42)
    if args.harness_check:
        print("HARNESS CHECK — random-init toy model, results carry no verdict weight.")
        model = _toy_model()
    else:
        from transformer_lens import HookedTransformer
        print("Loading gpt2-medium ...", flush=True)
        model = HookedTransformer.from_pretrained("gpt2-medium")
    model.eval()

    prompts = select_subset(tier["n_prompts"])
    if args.harness_check:
        prompts = [dict(rec, prompt=_toy_tokens(rec["prompt"])) for rec in prompts]
    print(f"Tier={args.tier} arms={arms} prompts={len(prompts)} "
          f"max_iter={tier['max_iter']} check_start={tier['check_start']}")

    outdir = HERE / "output"
    outdir.mkdir(exist_ok=True)
    suffix = f"{args.tier}_harness" if args.harness_check else args.tier
    results, terminals = [], {}
    t0 = time.time()
    for arm in arms:
        i, j = ARMS[arm]
        print(f"\n=== Arm {arm}: window {i}->{j} ===", flush=True)
        for rec in prompts:
            p = rec["prompt"]
            p_text = p if isinstance(p, str) else "harness-check-tokens"
            r = run_arm_with_terminal(model, p, i, j, tier["max_iter"], tier["check_start"])
            terminals[(arm, rec["id"])] = {
                "mean": r.pop("terminal_mean_vec"),
                "last": r.pop("terminal_last_vec"),
            }
            r.update(arm=arm, window=f"{i}->{j}", prompt_id=rec["id"], prompt=p_text,
                     category=rec["category"])
            results.append(r)
            print(f"  [{arm}] {rec['id']:<16} -> {r['terminal_token']!r:14} "
                  f"lock={r['lock_in_iter']} iters={r['n_iters']} "
                  f"margin={r['top_logit_margin']:.2f}", flush=True)
        # checkpoint after every arm
        json.dump(results, open(outdir / f"results_{suffix}.json", "w"), indent=2)
        torch.save(terminals, outdir / f"terminals_{suffix}.pt")

    # summary table
    print(f"\n=== Summary ({time.time()-t0:.0f}s) ===")
    for arm in arms:
        rs = [r for r in results if r["arm"] == arm]
        toks = sorted({r["terminal_token"] for r in rs})
        conv = sum(r["converged"] for r in rs)
        print(f"  {arm} {ARMS[arm][0]:>2}->{ARMS[arm][1]:<2} converged {conv}/{len(rs)} "
              f"unique_terminals={len(toks)} {toks[:8]}")
    print(f"\nArtifacts: results_{suffix}.json, terminals_{suffix}.pt in {outdir}")


if __name__ == "__main__":
    main()
