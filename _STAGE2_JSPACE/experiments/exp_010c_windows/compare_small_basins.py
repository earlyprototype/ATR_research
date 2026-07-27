"""EXP_010d — Does the Medium workspace-band loop recreate GPT-2 Small's basin
geometry? Pre-registered spec: ../../EXP_010d_SPEC.md

Runs native GPT-2 Small (0->11) on the same 25-prompt subset EXP_010c used, then
compares the Small basin PARTITION to the Medium windowed-band partition (A4,
10->21) and the Medium full-stack baseline (A0, 0->23) via chance-corrected
Adjusted Rand Index with a permutation null.

Cross-model note (spec §3): Small is 768-dim, Medium 1024-dim — terminal tensors
are NOT comparable across models, so the comparison is at the level the two share:
which prompts fall in the same basin as which others (the partition). Decode
tokens are reported as a descriptive secondary read only.

Usage:
    python compare_small_basins.py --small-path /path/to/gpt2-small [--tier full]
    python compare_small_basins.py --harness-check      # toy model, mechanics only
"""

import argparse
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # atr_engine2
sys.path.insert(0, str(HERE))          # run_exp010c helpers

from run_exp010c import run_arm_with_terminal  # noqa: E402
from analyze_terminals import cluster, CLUSTER_THRESHOLD  # noqa: E402

SMALL_WINDOW = (0, 11)  # native full stack for gpt2-small


# ---- Adjusted Rand Index (no sklearn dependency) --------------------------

def _comb2(n):
    return n * (n - 1) // 2


def adjusted_rand_index(labels_a, labels_b):
    """Chance-corrected agreement between two partitions of the same n items.

    ARI = 1 identical partitions, ~0 expected under random labelling, can go
    negative. Invariant to label naming and to the number of clusters — the
    right cross-model metric (spec §4).
    """
    assert len(labels_a) == len(labels_b)
    n = len(labels_a)
    contingency = Counter(zip(labels_a, labels_b))
    a_sizes = Counter(labels_a)
    b_sizes = Counter(labels_b)
    sum_ij = sum(_comb2(v) for v in contingency.values())
    sum_a = sum(_comb2(v) for v in a_sizes.values())
    sum_b = sum(_comb2(v) for v in b_sizes.values())
    total = _comb2(n)
    if total == 0:
        return 1.0
    expected = (sum_a * sum_b) / total
    max_index = 0.5 * (sum_a + sum_b)
    if max_index == expected:
        return 1.0  # both partitions trivial (all-together or all-singleton)
    return (sum_ij - expected) / (max_index - expected)


def permutation_p(labels_ref, labels_other, observed_ari, n_perm=10000, seed=42):
    """One-sided permutation p: fraction of label shuffles of `other` whose ARI
    with the fixed `labels_ref` reaches the observed ARI."""
    g = torch.Generator().manual_seed(seed)
    other = list(labels_other)
    n = len(other)
    hits = 1  # +1 for the observed (standard small-sample correction)
    for _ in range(n_perm):
        perm = torch.randperm(n, generator=g).tolist()
        shuffled = [other[i] for i in perm]
        if adjusted_rand_index(labels_ref, shuffled) >= observed_ari - 1e-12:
            hits += 1
    return hits / (n_perm + 1)


# ---- basin partitions -----------------------------------------------------

def partition_from_terminals(terminals, arm, prompt_ids):
    """prompt_id -> basin label, from saved terminal MEAN vectors for one arm."""
    vecs = [terminals[(arm, pid)]["mean"] for pid in prompt_ids]
    labels, n = cluster(vecs, threshold=CLUSTER_THRESHOLD)
    return labels, n


def effective_basins(results, arm):
    """Basin count that also collapses a single shared decode (the `D` funnel):
    reported alongside the tensor-cluster count for H11a."""
    rs = [r for r in results if r["arm"] == arm]
    decodes = {r["terminal_token"] for r in rs}
    return len(decodes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small-path", default=None, help="local gpt2-small dir")
    ap.add_argument("--tier", default="full", help="Medium result tier to read (full)")
    ap.add_argument("--harness-check", action="store_true")
    ap.add_argument("--max-iter", type=int, default=1000)
    ap.add_argument("--check-start", type=int, default=100)
    ap.add_argument("--n-perm", type=int, default=10000)
    args = ap.parse_args()

    outdir = HERE / "output"
    subset = json.load(open(outdir / "prompt_subset.json"))
    prompt_ids = [r["id"] for r in subset]

    # ---- Medium side: load committed terminals (A4 band, A0 baseline) -------
    med_terms_path = outdir / f"terminals_{args.tier}.pt"
    if not med_terms_path.exists():
        med_terms_path = outdir / f"terminals_{args.tier}_A0A4.pt.bak"
    med_results_path = outdir / f"results_{args.tier}.json"
    if not med_results_path.exists():
        med_results_path = outdir / f"results_{args.tier}_A0A4.json.bak"
    print(f"Medium terminals: {med_terms_path.name}   results: {med_results_path.name}")
    med_terms = torch.load(med_terms_path, map_location="cpu", weights_only=False)
    med_results = json.load(open(med_results_path))
    med_arms = sorted({a for (a, _) in med_terms})
    print(f"Medium arms available: {med_arms}")

    # ---- Small side: run native loop on the same prompts --------------------
    if args.harness_check:
        from run_exp010c import _toy_model, _toy_tokens
        print("HARNESS CHECK — toy Small (12-layer random init); mechanics only.")
        model = _toy_model_12()
        run_prompts = [(r["id"], _toy_tokens(r["prompt"])) for r in subset]
    else:
        from run_exp010c import _load_small_from_local
        print(f"Loading gpt2-small from {args.small_path} (offline) ...", flush=True)
        model = _load_small_from_local(args.small_path)
        run_prompts = [(r["id"], r["prompt"]) for r in subset]
    model.eval()

    i, j = SMALL_WINDOW
    small_terms, small_results = {}, []
    for pid, p in run_prompts:
        r = run_arm_with_terminal(model, p, i, j, args.max_iter, args.check_start)
        small_terms[("SMALL", pid)] = {"mean": r.pop("terminal_mean_vec"),
                                       "last": r.pop("terminal_last_vec")}
        r.update(arm="SMALL", window=f"{i}->{j}", prompt_id=pid)
        small_results.append(r)
        print(f"  [SMALL] {pid:<16} -> {r['terminal_token']!r:14} "
              f"lock={r['lock_in_iter']} conv={r['converged']}", flush=True)

    conv = sum(r["converged"] for r in small_results)
    if conv < 0.5 * len(small_results) and not args.harness_check:
        print(f"\nSTOP (spec §5): Small consolidated on only {conv}/{len(small_results)} "
              "prompts — no stable partition to match. Recording and stopping.")
    # Persist the Small terminals so threshold re-analysis never needs the model again.
    torch.save(small_terms, outdir / "terminals_small_010d.pt")

    # ---- partitions ---------------------------------------------------------
    small_labels, small_n = partition_from_terminals(small_terms, "SMALL", prompt_ids)
    report = {
        "prompt_ids": prompt_ids,
        "small_converged": f"{conv}/{len(small_results)}",
        "small": {"basins": small_n,
                  "decode_terminals": dict(Counter(r["terminal_token"] for r in small_results).most_common()),
                  "labels": small_labels},
        "comparisons": {},
    }

    for arm in ("A4", "A0"):
        if arm not in med_arms:
            print(f"  (arm {arm} not in committed terminals — skipping)")
            continue
        labels, n = partition_from_terminals(med_terms, arm, prompt_ids)
        ari = adjusted_rand_index(small_labels, labels)
        p = permutation_p(small_labels, labels, ari, n_perm=args.n_perm)
        eff = effective_basins(med_results, arm)
        report["comparisons"][arm] = {
            "medium_window": next((r["window"] for r in med_results if r["arm"] == arm), "?"),
            "medium_basins_tensor": n,
            "medium_basins_effective_decode": eff,
            "ari_vs_small": round(ari, 4),
            "perm_p": round(p, 5),
            "labels": labels,
        }

    # H11a: basin-count proximity (k pre-registered, not tuned)
    k = round(0.5 * small_n)
    report["h11a"] = {"small_basins": small_n, "k_window": k}

    # Threshold-sensitivity sweep (spec §6 caveat made quantitative): A0's basins
    # sit at off-diag cos ~.998, right at the 0.999 gate, so its partition — and
    # any ARI resting on it — could be threshold-fragile. Re-cluster all three at
    # a spread of thresholds and re-test. No model time (uses saved terminals).
    sweep = []
    for thr in (0.99, 0.995, 0.999, 0.9995):
        s_lab, s_n = cluster([small_terms[("SMALL", pid)]["mean"] for pid in prompt_ids], thr)
        row = {"threshold": thr, "small_basins": s_n}
        for arm in ("A4", "A0"):
            if arm not in med_arms:
                continue
            lab, n = cluster([med_terms[(arm, pid)]["mean"] for pid in prompt_ids], thr)
            ari = adjusted_rand_index(s_lab, lab)
            row[arm] = {"basins": n, "ari_vs_small": round(ari, 4),
                        "perm_p": round(permutation_p(s_lab, lab, ari, n_perm=args.n_perm), 5)}
        sweep.append(row)
    report["threshold_sweep"] = sweep

    out = outdir / f"basin_comparison_{args.tier}.json"
    out.write_text(json.dumps(report, indent=2))

    # ---- verdicts (mechanical application of spec §5) -----------------------
    print("\n=== EXP_010d verdicts ===")
    print(f"Small (0->11): {small_n} tensor basins on {len(prompt_ids)} prompts; "
          f"converged {conv}/{len(small_results)}")
    a4 = report["comparisons"].get("A4")
    a0 = report["comparisons"].get("A0")
    if a4 and a0:
        print(f"  ARI(Small, A4 band 10->21) = {a4['ari_vs_small']}  perm_p = {a4['perm_p']}")
        print(f"  ARI(Small, A0 baseline)    = {a0['ari_vs_small']}  perm_p = {a0['perm_p']}")
        h11 = (a4["ari_vs_small"] > 0 and a4["perm_p"] < 0.05
               and a4["ari_vs_small"] > a0["ari_vs_small"])
        print(f"  H11 (geometry recreation): {'SUPPORTED' if h11 else 'REFUTED'}")
        within = abs(a4["medium_basins_tensor"] - small_n) <= k
        a0_within = abs(a0["medium_basins_tensor"] - small_n) <= k
        print(f"  H11a (basin count, k={k}): A4 {'in' if within else 'out'} band, "
              f"A0 {'in' if a0_within else 'out'} band "
              f"(Small {small_n}, A4 {a4['medium_basins_tensor']}, A0 {a0['medium_basins_tensor']})")
    print("\n=== threshold-sensitivity sweep (ARI vs Small; is the effect stable?) ===")
    print(f"  {'thr':<8}{'S_basins':<10}{'A4 basins/ARI/p':<26}{'A0 basins/ARI/p':<26}")
    for row in report["threshold_sweep"]:
        a4 = row.get("A4", {})
        a0 = row.get("A0", {})
        a4s = f"{a4.get('basins','-')}/{a4.get('ari_vs_small','-')}/{a4.get('perm_p','-')}"
        a0s = f"{a0.get('basins','-')}/{a0.get('ari_vs_small','-')}/{a0.get('perm_p','-')}"
        print(f"  {row['threshold']:<8}{row['small_basins']:<10}{a4s:<26}{a0s:<26}")

    print(f"\nSaved -> {out}")


def _toy_model_12():
    """12-layer random-init toy for --harness-check (Small-shaped)."""
    from transformer_lens import HookedTransformer, HookedTransformerConfig
    from run_exp010c import _DummyTokenizer
    cfg = HookedTransformerConfig(n_layers=12, d_model=64, n_ctx=32, d_head=16,
                                  n_heads=4, d_vocab=997, act_fn="gelu",
                                  normalization_type="LN")
    m = HookedTransformer(cfg)
    m.tokenizer = _DummyTokenizer()
    return m


if __name__ == "__main__":
    main()
