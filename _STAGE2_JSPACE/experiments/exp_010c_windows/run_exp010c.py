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

from atr_engine2 import run_atr_gated  # noqa: E402
from derive_prompts import select_subset, select_subset_b  # noqa: E402

ARMS = {
    "A0": (0, 23),   # baseline / reproduction gate
    "A4": (10, 21),  # band-exact
    "A1": (0, 11),   # placement: front
    "A2": (6, 17),   # placement: middle
    "A3": (12, 23),  # placement: back
    "A5": (8, 15),   # length probe (8 layers, mid-band)
    # EXP_010c-2 boundary scan (EXP_010c2_SPEC.md §3)
    "E22": (10, 22),  # exit edge: +layer 22
    "E23": (10, 23),  # exit edge: +motor tail (final-layer necessity, H10a)
    "O0": (0, 21),    # onset edge: sensory-splice test (H10b)
    "O4": (4, 21),
    "O6": (6, 21),
    "O8": (8, 21),
    "O12": (12, 21),
    "O14": (14, 21),
    # EXP_010c-3 in-fill around the word-forming cells (EXP_010c3_SPEC.md §3)
    "I5": (5, 21),     # injection in-fill (extract 21): odd layers around 8/10
    "I7": (7, 21),
    "I9": (9, 21),     # critical point between the two word cells 8->21 and 10->21
    "I11": (11, 21),
    "X817": (8, 17),   # extraction column below 21 at the word-forming injections
    "X819": (8, 19),
    "X1015": (10, 15),
    "X1017": (10, 17),
    "X1019": (10, 19),
    # EXP_010c-3b §4 — extraction ladder above 21 at injection 8
    "E822": (8, 22),
    "E823": (8, 23),
    # EXP_010c-VARIANTS Control A (EXP_010c_VARIANTS_SPEC.md §3, issue #13)
    "I1A0": (1, 23),   # i=1 variant of A0 (0->23)
    "I1O0": (1, 21),   # i=1 variant of O0 (0->21)
    "HP9": (10, 21),   # sanity: inject at blocks.9.hook_resid_post (see ARM_INJECT_HOOK)
}
# Control A sanity arm: pre-registered expectation is that resid_post(9) is
# the SAME residual value as resid_pre(10), so HP9 must reproduce A4 exactly.
ARM_INJECT_HOOK = {"HP9": "blocks.9.hook_resid_post"}
ARM_ORDER = ["A0", "A4", "A1", "A2", "A3", "A5"]  # spec §5 execution order
SCAN_ORDER = ["E22", "E23", "O0", "O4", "O6", "O8", "O12", "O14"]
INFILL_ORDER = ["I5", "I7", "I9", "I11",  # EXP_010c3_SPEC.md §5 execution order
                "X817", "X819", "X1015", "X1017", "X1019"]
LADDER8_ORDER = ["E822", "E823"]          # EXP_010c3b_SPEC.md §4

TIERS = {
    "smoke": dict(n_prompts=2, max_iter=60, check_start=20, arms=["A0", "A4"]),
    "pilot": dict(n_prompts=5, max_iter=300, check_start=50, arms=ARM_ORDER),
    "full": dict(n_prompts=25, max_iter=1000, check_start=100, arms=ARM_ORDER),
    "scan": dict(n_prompts=25, max_iter=1000, check_start=100, arms=SCAN_ORDER),
    "infill": dict(n_prompts=25, max_iter=1000, check_start=100, arms=INFILL_ORDER),
    "ladder8": dict(n_prompts=25, max_iter=1000, check_start=100, arms=LADDER8_ORDER),
    # EXP_010c3b_SPEC.md §5 — settle-time variant. check_start=10 is a RECORDED
    # protocol deviation (earliest reportable lock drops 120 -> 30); it exists to
    # measure settle time and draws no verdicts on the registered questions.
    "settle": dict(n_prompts=5, max_iter=1000, check_start=10,
                   arms=["I7", "I9", "X1017"]),
    # EXP_010c-VARIANTS tiers (spec §3/§4): registered protocol, variant arms.
    "hookpoint": dict(n_prompts=25, max_iter=1000, check_start=100,
                      arms=["I1A0", "I1O0", "HP9"]),
    "energynorm": dict(n_prompts=25, max_iter=1000, check_start=100,
                       arms=["A0", "A4", "O8", "A1"]),
}


def run_arm_with_terminal(model, prompt, i, j, max_iter, check_start,
                          inject_hook_name=None, renorm="seed_j"):
    """Thin wrapper: the gated protocol lives ONLY in atr_engine2.run_atr_gated
    (capture_terminal=True adds terminal tensors + a real lag_scan dict — the
    recorded diff vs the upstream engine; see atr_engine2.py header).

    History note (PR #4 review): an earlier version re-implemented the gated
    loop here and saved lag_scan's dict KEYS instead of its cosine values, so
    every pre-fix artifact carries the placeholder [1.0..8.0]. Fixed by this
    wrapper; artifacts regenerated.
    """
    r = run_atr_gated(model, prompt, i, j, max_iter=max_iter,
                      check_start=check_start, capture_terminal=True,
                      inject_hook_name=inject_hook_name, renorm=renorm)
    # lag_scan arrives as {lag: mean_cosine}; keep the mapping explicit.
    if r.get("lag_scan") is not None:
        r["lag_scan"] = {str(k): v for k, v in r["lag_scan"].items()}
    r["terminal_prob"] = float(r["terminal_prob"])
    return r


class _DummyTokenizer:
    """Decode stub for --harness-check runs (random-init model, no real vocab)."""

    padding_side = "right"
    pad_token_id = 0

    def decode(self, ids):
        """Render dummy token ids as visible placeholders like <42>."""
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
    """Map a prompt string to deterministic dummy token ids (harness-check only)."""
    ids = [(hash(w) % (d_vocab - 1)) + 1 for w in prompt.split()[:max_len]]
    return torch.tensor([ids or [1]], dtype=torch.long)


def _load_medium_from_local(path):
    """Offline load: local dir must hold config.json, pytorch_model.bin,
    vocab.json, merges.txt (e.g. from the legacy HF S3 mirror). Seeds the HF
    cache with config.json so transformer_lens's internal AutoConfig lookup
    resolves without network, then passes model+tokenizer in explicitly."""
    import os
    import shutil

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    snap = cache / "models--gpt2-medium" / "snapshots" / "local"
    refs = cache / "models--gpt2-medium" / "refs"
    snap.mkdir(parents=True, exist_ok=True)
    refs.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(path) / "config.json", snap / "config.json")
    (refs / "main").write_text("local")

    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    from transformer_lens import HookedTransformer

    hf = GPT2LMHeadModel.from_pretrained(path)
    tok = GPT2Tokenizer.from_pretrained(path)
    return HookedTransformer.from_pretrained("gpt2-medium", hf_model=hf, tokenizer=tok)


def main():
    """Run the selected tier's arms and write results + terminal artifacts."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=TIERS, required=True)
    ap.add_argument("--arms", default=None, help="comma-separated override, e.g. A0,A4")
    ap.add_argument("--harness-check", action="store_true",
                    help="random-init toy model; validates the harness, draws no verdicts")
    ap.add_argument("--model-path", default=None,
                    help="local dir with gpt2-medium files for offline load")
    # Seed / subset / artifact-suffix parameters. All defaults reproduce the
    # registered behaviour exactly, so every prior command line still replays.
    #
    # MERGE NOTE (EXP_010c-ROBUST issue #11 and EXP_010c-3b issue #21 added
    # overlapping flags in parallel). Both interfaces are kept so the run
    # commands recorded against each experiment's committed artifacts still
    # work; they share ONE implementation underneath:
    #   --subset B      == --prompt-offset 25   (verified equal, see below)
    #   --out-suffix X  == --tag X
    ap.add_argument("--seed", type=int, default=42,
                    help="global torch seed (registered runs used 42; EXP_010c3b "
                         "§2a measured this to be a no-op in this harness)")
    ap.add_argument("--subset", choices=["registered", "B"], default="registered",
                    help="prompt subset: registered round-robin 25, or disjoint B "
                         "(equivalent to --prompt-offset 25)")
    ap.add_argument("--prompt-offset", type=int, default=0,
                    help="offset into the deterministic round-robin ordering; "
                         "25 gives the disjoint subset (EXP_010c3b §2b)")
    ap.add_argument("--n-prompts", type=int, default=None,
                    help="override the tier's prompt count")
    ap.add_argument("--out-suffix", default=None,
                    help="artifact suffix override (e.g. robust_seed1337); "
                         "default keeps the tier-based naming")
    ap.add_argument("--tag", default=None,
                    help="alias of --out-suffix. REQUIRED for variant runs so they "
                         "cannot overwrite the registered artifacts.")
    # EXP_010c-VARIANTS spec §2 (recorded diff, issue #14): energy-rescale
    # target. Default reproduces the registered convention exactly.
    ap.add_argument("--renorm", choices=["seed_j", "natural_i"], default="seed_j",
                    help="loop rescale target: seed norm at extraction layer j "
                         "(registered) or natural resid_pre norm at injection layer i")
    args = ap.parse_args()
    tier = TIERS[args.tier]
    arms = args.arms.split(",") if args.arms else tier["arms"]
    n_prompts = args.n_prompts if args.n_prompts is not None else tier["n_prompts"]

    torch.manual_seed(args.seed)
    if args.harness_check:
        print("HARNESS CHECK — random-init toy model, results carry no verdict weight.")
        model = _toy_model()
    elif args.model_path:
        print(f"Loading gpt2-medium from local path {args.model_path} (offline) ...", flush=True)
        model = _load_medium_from_local(args.model_path)
    else:
        from transformer_lens import HookedTransformer
        print("Loading gpt2-medium ...", flush=True)
        model = HookedTransformer.from_pretrained("gpt2-medium")
    model.eval()

    # --subset B and --prompt-offset 25 are the same request (see MERGE NOTE);
    # conflicting explicit values are an error rather than a silent precedence.
    if args.subset == "B" and args.prompt_offset not in (0, 25):
        ap.error(f"--subset B implies --prompt-offset 25, got {args.prompt_offset}")
    offset = 25 if args.subset == "B" else args.prompt_offset
    prompts = select_subset(n_prompts, offset=offset)
    subset_b_records = prompts if offset == 25 else None
    if args.harness_check:
        prompts = [dict(rec, prompt=_toy_tokens(rec["prompt"])) for rec in prompts]
    print(f"Tier={args.tier} arms={arms} prompts={len(prompts)} subset={args.subset} "
          f"prompt_offset={offset} seed={args.seed} renorm={args.renorm} "
          f"max_iter={tier['max_iter']} check_start={tier['check_start']}")

    outdir = HERE / "output"
    outdir.mkdir(exist_ok=True)
    suffix = (args.tag or args.out_suffix
              or (f"{args.tier}_harness" if args.harness_check else args.tier))
    if subset_b_records is not None:  # audit record of the executed disjoint subset
        (outdir / "prompt_subset_b.json").write_text(json.dumps(subset_b_records, indent=2))
    if args.renorm == "natural_i":
        # Reference-norm record (spec §4, issue #14): natural per-layer
        # resid_pre norms for every prompt, one un-hooked pass each.
        norm_rec = {}
        for rec in prompts:
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    rec["prompt"], names_filter=lambda n: n.endswith("hook_resid_pre"))
            norm_rec[rec["id"]] = {
                str(l): round(cache[f"blocks.{l}.hook_resid_pre"][0].norm().item(), 4)
                for l in range(model.cfg.n_layers)}
        (outdir / f"natural_resid_norms_{suffix}.json").write_text(
            json.dumps(norm_rec, indent=2))
        print(f"Natural per-layer resid_pre norms recorded -> "
              f"natural_resid_norms_{suffix}.json", flush=True)
    results, terminals = [], {}
    t0 = time.time()
    for arm in arms:
        i, j = ARMS[arm]
        inject_hook = ARM_INJECT_HOOK.get(arm)
        print(f"\n=== Arm {arm}: window {i}->{j}"
              f"{' inject_hook=' + inject_hook if inject_hook else ''} ===", flush=True)
        for rec in prompts:
            p = rec["prompt"]
            p_text = p if isinstance(p, str) else "harness-check-tokens"
            r = run_arm_with_terminal(model, p, i, j, tier["max_iter"], tier["check_start"],
                                      inject_hook_name=inject_hook, renorm=args.renorm)
            # string keys ("ARM|PROMPT_ID") so the .pt loads with weights_only=True
            terminals[f"{arm}|{rec['id']}"] = {
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
