"""EXP_010c — Window grid on GPT-2 Medium. Pre-registered spec: ../../EXP_010c_SPEC.md

Arms (inject i -> extract j), seeding: natural L0 prompt pass in every arm (spec §3).

Tiers:
    smoke  — 2 prompts, arms A0+A4, max_iter=60   (harness validation only)
    pilot  — 5 prompts, all arms,   max_iter=300  (directional signal, ~1-2 h CPU)
    full   — 25 prompts, all arms,  max_iter=1000 (the pre-registered run, overnight)
    scan   — 25 prompts, boundary-scan arms, max_iter=1000 (EXP_010c-2)
    infill — 25 prompts, in-fill arms, max_iter=1000 (EXP_010c-3)

The gated protocol params (threshold/patience/check_every/check_start) follow the
spec for `full`; smoke/pilot shrink check_start proportionally and are RECORDED as
non-registered tiers — no verdicts are drawn from them beyond harness validity.

Usage: python run_exp010c.py --tier smoke|pilot|full [--arms A0,A4,...]
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # for atr_engine2

from atr_engine2 import run_atr_gated  # noqa: E402
from derive_prompts import select_subset  # noqa: E402

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
    # EXP_010c-3 in-fill scan (EXP_010c3_SPEC.md §3)
    "I9": (9, 21),      # injection in-fill: the critical untested point (H11)
    "I11": (11, 21),    # injection in-fill: right flank
    "I7": (7, 21),      # injection in-fill: left flank
    "I5": (5, 21),      # injection in-fill: left approach
    "X1019": (10, 19),  # extraction column, i=10 row
    "X1017": (10, 17),
    "X1015": (10, 15),
    "X819": (8, 19),    # extraction column, i=8 row
    "X817": (8, 17),
}
ARM_ORDER = ["A0", "A4", "A1", "A2", "A3", "A5"]  # spec §5 execution order
SCAN_ORDER = ["E22", "E23", "O0", "O4", "O6", "O8", "O12", "O14"]
INFILL_ORDER = ["I9", "I11", "I7", "I5", "X1019", "X1017", "X1015", "X819", "X817"]

# EXP_010c-4 full census (EXP_010c4_SPEC.md): every valid window 0<=i<=j<=23
# not already measured at the registered protocol by the tiers above. Arm
# names are positional (W{i}_{j}); ARMS stays the single source of truth.
_MEASURED = set(ARMS.values())
CENSUS = {
    f"W{i}_{j}": (i, j)
    for i in range(24) for j in range(i, 24)
    if (i, j) not in _MEASURED
}
ARMS.update(CENSUS)
# Execution order (spec §3): the neighbourhood of all known structure first
# (4<=i<=14, j>=13), then the remainder row-major — pure risk management for
# a multi-day run; every arm runs regardless.
CENSUS_ORDER = sorted(
    CENSUS, key=lambda a: (not (4 <= CENSUS[a][0] <= 14 and CENSUS[a][1] >= 13),
                           CENSUS[a][0], CENSUS[a][1])
)

TIERS = {
    "smoke": dict(n_prompts=2, max_iter=60, check_start=20, arms=["A0", "A4"]),
    "pilot": dict(n_prompts=5, max_iter=300, check_start=50, arms=ARM_ORDER),
    "full": dict(n_prompts=25, max_iter=1000, check_start=100, arms=ARM_ORDER),
    "scan": dict(n_prompts=25, max_iter=1000, check_start=100, arms=SCAN_ORDER),
    "infill": dict(n_prompts=25, max_iter=1000, check_start=100, arms=INFILL_ORDER),
    # census shards artifacts per arm: at 277 arms a single rewritten
    # results/terminals pair would put ~55 MB .pt blobs into git on every
    # per-arm commit; shards are written once and committed once (each stays
    # ~200 KB, inside the repo's ~2 MB artifact convention).
    "census": dict(n_prompts=25, max_iter=1000, check_start=100, arms=CENSUS_ORDER,
                   shard=True),
}


def run_arm_with_terminal(model, prompt, i, j, max_iter, check_start):
    """Thin wrapper: the gated protocol lives ONLY in atr_engine2.run_atr_gated
    (capture_terminal=True adds terminal tensors + a real lag_scan dict — the
    recorded diff vs the upstream engine; see atr_engine2.py header).

    History note (PR #4 review): an earlier version re-implemented the gated
    loop here and saved lag_scan's dict KEYS instead of its cosine values, so
    every pre-fix artifact carries the placeholder [1.0..8.0]. Fixed by this
    wrapper; artifacts regenerated.
    """
    r = run_atr_gated(model, prompt, i, j, max_iter=max_iter,
                      check_start=check_start, capture_terminal=True)
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
    """Map a prompt string to deterministic dummy token ids (harness-check only).

    Uses crc32, not hash(): Python randomizes str hashes per process
    (PYTHONHASHSEED), which silently made harness-check token ids — and
    therefore the prompt-sensitive arms' toy terminals — unreproducible
    across runs. Found 2026-07-24; toy tier only, no registered artifact
    was affected."""
    import zlib
    ids = [(zlib.crc32(w.encode()) % (d_vocab - 1)) + 1 for w in prompt.split()[:max_len]]
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
    ap.add_argument("--resume", action="store_true",
                    help="load this tier's existing artifacts, skip arms already "
                         "complete, and append new arms to the same files")
    ap.add_argument("--shard", action="store_true",
                    help="force per-arm artifact shards (results_<tier>/<arm>.json, "
                         "terminals_<tier>/<arm>.pt) for any tier")
    args = ap.parse_args()
    tier = TIERS[args.tier]
    arms = args.arms.split(",") if args.arms else tier["arms"]

    torch.manual_seed(42)
    prompts = select_subset(tier["n_prompts"])
    outdir = HERE / "output"
    outdir.mkdir(exist_ok=True)
    suffix = f"{args.tier}_harness" if args.harness_check else args.tier

    shard = bool(tier.get("shard")) or args.shard

    # Resume: per-run results are deterministic and independent of process
    # boundaries (verified 350/350 on the full+scan regeneration and 13/13 on
    # the X819 rerun), and the per-arm checkpoint only ever writes COMPLETE
    # arms, so skipping arms already on disk is exact, not approximate.
    # An arm counts as complete only if BOTH artifacts verify: full record
    # count AND a terminal entry per record (PR #10 review: a crash between
    # the two writes must not mark the arm done; terminals are written first
    # and the results JSON is the completion marker, but resume re-verifies
    # the pair regardless).
    results, terminals = [], {}
    if args.resume:
        if shard:
            done = set()
            for arm in arms:
                rshard = outdir / f"results_{suffix}" / f"{arm}.json"
                tshard = outdir / f"terminals_{suffix}" / f"{arm}.pt"
                try:
                    recs = json.load(open(rshard))
                    if len(recs) < len(prompts) or not tshard.exists():
                        continue
                    keys = set(torch.load(tshard, map_location="cpu", weights_only=True))
                    if keys >= {f"{arm}|{r['prompt_id']}" for r in recs}:
                        done.add(arm)
                except FileNotFoundError:
                    pass  # arm not attempted yet
                except Exception as e:
                    print(f"Resume: shard for {arm} failed verification ({e!r}); "
                          "will rerun.", flush=True)
            arms = [a for a in arms if a not in done]
            print(f"Resume (sharded): {len(done)} arms verified complete on disk; "
                  f"{len(arms)} remaining.", flush=True)
        else:
            rpath = outdir / f"results_{suffix}.json"
            tpath = outdir / f"terminals_{suffix}.pt"
            if rpath.exists():
                results = json.load(open(rpath))
                if tpath.exists():
                    terminals = torch.load(tpath, map_location="cpu", weights_only=True)
                done = set()
                for a, n in Counter(r["arm"] for r in results).items():
                    if n >= len(prompts) and all(
                        f"{a}|{r['prompt_id']}" in terminals
                        for r in results if r["arm"] == a
                    ):
                        done.add(a)
                # drop anything not verified so reruns cannot duplicate records
                results = [r for r in results if r["arm"] in done]
                terminals = {k: v for k, v in terminals.items()
                             if k.split("|", 1)[0] in done}
                arms = [a for a in arms if a not in done]
                print(f"Resume: {len(done)} arms verified complete on disk; "
                      f"{len(arms)} remaining.", flush=True)
        if not arms:
            print("Resume: nothing to run — all requested arms complete.")
            return

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

    if args.harness_check:
        prompts = [dict(rec, prompt=_toy_tokens(rec["prompt"])) for rec in prompts]
    print(f"Tier={args.tier} arms={arms} prompts={len(prompts)} "
          f"max_iter={tier['max_iter']} check_start={tier['check_start']}")
    t0 = time.time()
    for arm in arms:
        i, j = ARMS[arm]
        print(f"\n=== Arm {arm}: window {i}->{j} ===", flush=True)
        arm_results, arm_terminals = [], {}
        for rec in prompts:
            p = rec["prompt"]
            p_text = p if isinstance(p, str) else "harness-check-tokens"
            r = run_arm_with_terminal(model, p, i, j, tier["max_iter"], tier["check_start"])
            # string keys ("ARM|PROMPT_ID") so the .pt loads with weights_only=True
            arm_terminals[f"{arm}|{rec['id']}"] = {
                "mean": r.pop("terminal_mean_vec"),
                "last": r.pop("terminal_last_vec"),
            }
            r.update(arm=arm, window=f"{i}->{j}", prompt_id=rec["id"], prompt=p_text,
                     category=rec["category"])
            arm_results.append(r)
            print(f"  [{arm}] {rec['id']:<16} -> {r['terminal_token']!r:14} "
                  f"lock={r['lock_in_iter']} iters={r['n_iters']} "
                  f"margin={r['top_logit_margin']:.2f}", flush=True)
        results.extend(arm_results)
        terminals.update(arm_terminals)
        # checkpoint after every arm: sharded tiers write the arm's own files
        # once (stable per-arm blobs for git); monolith tiers rewrite the pair.
        # Terminals are written FIRST so the results JSON acts as the
        # completion marker — a crash between the writes leaves an arm that
        # resume treats as incomplete, never one missing its tensors.
        if shard:
            (outdir / f"results_{suffix}").mkdir(exist_ok=True)
            (outdir / f"terminals_{suffix}").mkdir(exist_ok=True)
            torch.save(arm_terminals, outdir / f"terminals_{suffix}" / f"{arm}.pt")
            json.dump(arm_results,
                      open(outdir / f"results_{suffix}" / f"{arm}.json", "w"), indent=2)
        else:
            torch.save(terminals, outdir / f"terminals_{suffix}.pt")
            json.dump(results, open(outdir / f"results_{suffix}.json", "w"), indent=2)

    # summary table
    print(f"\n=== Summary ({time.time()-t0:.0f}s) ===")
    for arm in arms:
        rs = [r for r in results if r["arm"] == arm]
        toks = sorted({r["terminal_token"] for r in rs})
        conv = sum(r["converged"] for r in rs)
        print(f"  {arm} {ARMS[arm][0]:>2}->{ARMS[arm][1]:<2} converged {conv}/{len(rs)} "
              f"unique_terminals={len(toks)} {toks[:8]}")
    if shard:
        print(f"\nArtifacts: results_{suffix}/<arm>.json, terminals_{suffix}/<arm>.pt in {outdir}")
    else:
        print(f"\nArtifacts: results_{suffix}.json, terminals_{suffix}.pt in {outdir}")


if __name__ == "__main__":
    main()
