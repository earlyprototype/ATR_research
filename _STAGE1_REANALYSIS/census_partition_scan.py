"""Census-wide tensor-partition scan: do ANY of GPT-2 Medium's 277 census
windows reproduce GPT-2 Small's native basin partition at the TENSOR level?

Motivation (EXP_010d): Medium's full-stack baseline A0 (0->23) decodes to a
single token `D` for all 25 prompts, yet its TENSOR partition of those prompts
agrees with GPT-2 Small's native (0->11) partition at ARI = 0.2001,
permutation p = 0.0009. Tensor structure and token readout can disagree, so a
census arm could carry Small-like structure while decoding to nothing
socialist. The census terminal TOKENS have already been checked and contain no
socialist vocabulary; this script checks the tensors.

Method is copied verbatim from compare_small_basins.py / analyze_terminals.py
so numbers are directly comparable to the recorded ARI 0.2001:
  - greedy leader clustering of terminal MEAN vectors at cosine > 0.999
  - vectors presented in prompt_subset.json order (leader clustering is
    order-dependent, so this must match)
  - chance-corrected Adjusted Rand Index vs Small's partition
  - one-sided permutation null, torch.randperm(seed=42), +1 correction

Adds, beyond the original:
  - the same test run over all 277 census arms plus the A0/A4 reference arms
  - Bonferroni and Benjamini-Hochberg multiple-comparison control over the
    277-arm family
  - the 0.99 / 0.995 / 0.999 / 0.9995 threshold-robustness sweep for every arm
  - a high-resolution permutation re-run for the top candidates

No model time; pure analysis of committed artifacts.

Usage:
    python census_partition_scan.py [--n-perm 10000] [--n-perm-top 200000]
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------- paths ----
HERE = Path(__file__).resolve().parent
EXP = HERE.parent / "_STAGE2_JSPACE" / "experiments" / "exp_010c_windows"
OUT = EXP / "output"
CENSUS_TERMS = Path("/home/user/census_t/_STAGE2_JSPACE/experiments/"
                    "exp_010c_windows/output/terminals_census")
CENSUS_RESULTS = Path("/home/user/census/_STAGE2_JSPACE/experiments/"
                      "exp_010c_windows/output/results_census")

CLUSTER_THRESHOLD = 0.999        # analyze_terminals.CLUSTER_THRESHOLD
SWEEP_THRESHOLDS = (0.99, 0.995, 0.999, 0.9995)
BASELINE_ARI = 0.2001            # A0 (0->23), recorded in basin_comparison_full.json

# GPT-2 Small's per-prompt terminal DECODE tokens on the same 25 prompts,
# transcribed from output/exp010d_run.log (the EXP_010d run that produced
# terminals_small_010d.pt). Used only for the supplementary "is it the
# SOCIALIST basin specifically?" read — the primary metric above uses the
# tensor partition, exactly as the recorded ARI 0.2001 does.
SMALL_DECODE = {
    "E01_politics": " prolet", "D01_water": " prolet", "A01_physics": " prolet",
    "B01_napoleon": " prolet", "C01_jack_jill": " prolet", "F01_anger": " Divine",
    "G01_punctuation": " till", "E02_tech": " Divine", "D02_periodic": " till",
    "A02_medical": " prolet", "B02_wwi": " prolet", "C02_king_cole": " prolet",
    "F02_insult": " Divine", "G02_brackets": " prolet", "E03_orgs": " Divine",
    "D03_organic": " till", "A03_neuro": " Anarch", "B03_moon": " Divine",
    "C03_mary_lamb": " prolet", "F03_frustration": " Divine",
    "G03_counting": " prolet", "E04_internet": " Divine",
    "D04_equation": " Anarch", "A04_climate": " prolet", "B04_rome": " prolet",
}
SOCIALIST_TOKENS = {" prolet", " Anarch"}


# ---- verbatim from analyze_terminals.py -----------------------------------

def cluster(vecs, threshold=CLUSTER_THRESHOLD):
    """Greedy leader clustering on cosine similarity."""
    leaders, labels = [], []
    for v in vecs:
        for li, l in enumerate(leaders):
            if F.cosine_similarity(v.unsqueeze(0), l.unsqueeze(0)).item() > threshold:
                labels.append(li)
                break
        else:
            leaders.append(v)
            labels.append(len(leaders) - 1)
    return labels, len(leaders)


# ---- verbatim from compare_small_basins.py --------------------------------

def _comb2(n):
    return n * (n - 1) // 2


def adjusted_rand_index(labels_a, labels_b):
    """Chance-corrected agreement between two partitions of the same n items."""
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
        return 1.0
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


# ---- multiple-comparison control ------------------------------------------

def benjamini_hochberg(pvals, alpha=0.05):
    """Return (rejected_bool_list, qvalues) for BH-FDR at level alpha."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [0.0] * m
    prev = 1.0
    # step-up from the largest p
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        val = min(prev, pvals[i] * m / rank)
        q[i] = val
        prev = val
    rejected = [q[i] <= alpha for i in range(m)]
    return rejected, q


# ---- loading --------------------------------------------------------------

def load_small(prompt_ids):
    """Small's terminal mean vectors, in prompt_subset order."""
    d = torch.load(OUT / "terminals_small_010d.pt", map_location="cpu",
                   weights_only=False)
    # legacy tuple-keyed archive
    return [d[("SMALL", pid)]["mean"] for pid in prompt_ids]


def load_arm_vecs(path, arm, prompt_ids):
    """Census/reference terminal mean vectors for one arm, in prompt order."""
    d = torch.load(path, map_location="cpu", weights_only=True)
    return [d[f"{arm}|{pid}"]["mean"] for pid in prompt_ids]


def load_reference_arm(arm, prompt_ids):
    """A0 / A4 from the committed terminals_full.pt (string-keyed, post-PR#4)."""
    d = torch.load(OUT / "terminals_full.pt", map_location="cpu",
                   weights_only=True)
    return [d[f"{arm}|{pid}"]["mean"] for pid in prompt_ids]


# ---- main -----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=10000,
                    help="permutations for the full 277-arm sweep")
    ap.add_argument("--n-perm-top", type=int, default=200000,
                    help="permutations for the high-resolution re-run of top arms")
    ap.add_argument("--n-top", type=int, default=15)
    ap.add_argument("--out", default=str(HERE / "census_partition_scan.json"))
    args = ap.parse_args()

    subset = json.load(open(OUT / "prompt_subset.json"))
    prompt_ids = [r["id"] for r in subset]
    n = len(prompt_ids)
    print(f"{n} prompts: {prompt_ids[:3]} ...", flush=True)

    # ---- Small reference partition (threshold 0.999, and the sweep) --------
    small_vecs = load_small(prompt_ids)
    small_labels, small_n = cluster(small_vecs, CLUSTER_THRESHOLD)
    small_by_thr = {thr: cluster(small_vecs, thr) for thr in SWEEP_THRESHOLDS}
    print(f"Small (0->11) reference: {small_n} tensor basins at thr=0.999")
    print(f"  labels: {small_labels}")

    # sanity: must match the recorded EXP_010d reference partition
    recorded = json.load(open(OUT / "basin_comparison_full.json"))
    assert prompt_ids == recorded["prompt_ids"], "prompt order drift"
    assert small_labels == recorded["small"]["labels"], \
        "Small partition does not reproduce basin_comparison_full.json"
    print("  [ok] Small partition reproduces basin_comparison_full.json")

    # supplementary reference partitions (decode-level, and socialist-vs-rest)
    small_decode_labels = [SMALL_DECODE[pid] for pid in prompt_ids]
    small_socialist_binary = [int(SMALL_DECODE[pid] in SOCIALIST_TOKENS)
                              for pid in prompt_ids]
    assert (dict(Counter(small_decode_labels))
            == recorded["small"]["decode_terminals"]), "decode transcript drift"
    n_soc = sum(small_socialist_binary)
    print(f"  Small decode partition: {dict(Counter(small_decode_labels))}")
    print(f"  Small socialist basin (' prolet'/' Anarch'): {n_soc}/{n} prompts")

    # ---- reference arms A0 / A4 (reproduction check) -----------------------
    refs = {}
    for arm in ("A0", "A4"):
        vecs = load_reference_arm(arm, prompt_ids)
        labels, k = cluster(vecs, CLUSTER_THRESHOLD)
        ari = adjusted_rand_index(small_labels, labels)
        p = permutation_p(small_labels, labels, ari, n_perm=args.n_perm)
        rec = recorded["comparisons"][arm]
        ok = abs(round(ari, 4) - rec["ari_vs_small"]) < 1e-9 and labels == rec["labels"]
        print(f"  [{'ok' if ok else 'MISMATCH'}] {arm}: ARI={ari:.4f} "
              f"(recorded {rec['ari_vs_small']})  p={p:.5f} (recorded {rec['perm_p']})")
        refs[arm] = {"arm": arm, "window": rec["medium_window"], "basins": k,
                     "ari": ari, "p": p, "labels": labels, "vecs": vecs}
    assert abs(round(refs["A0"]["ari"], 4) - BASELINE_ARI) < 1e-9, \
        "failed to reproduce the A0 baseline ARI"

    # ---- census sweep ------------------------------------------------------
    pt_files = sorted(CENSUS_TERMS.glob("*.pt"))
    print(f"\nScanning {len(pt_files)} census arms "
          f"({args.n_perm} permutations each) ...", flush=True)

    rows = []
    for idx, pt in enumerate(pt_files, 1):
        arm = pt.stem
        rj = CENSUS_RESULTS / f"{arm}.json"
        recs = json.load(open(rj)) if rj.exists() else []
        by_pid = {r["prompt_id"]: r for r in recs}
        window = recs[0]["window"] if recs else "?"
        converged = sum(bool(r.get("converged")) for r in recs)
        decodes = Counter(r["terminal_token"] for r in recs)

        vecs = load_arm_vecs(pt, arm, prompt_ids)
        labels, k = cluster(vecs, CLUSTER_THRESHOLD)
        ari = adjusted_rand_index(small_labels, labels)
        p = permutation_p(small_labels, labels, ari, n_perm=args.n_perm)

        # secondary/exploratory: partition of the "last"-token terminal vectors
        d = torch.load(pt, map_location="cpu", weights_only=True)
        last_vecs = [d[f"{arm}|{pid}"]["last"] for pid in prompt_ids]
        last_labels, last_k = cluster(last_vecs, CLUSTER_THRESHOLD)
        ari_last = adjusted_rand_index(small_labels, last_labels)

        # off-diagonal cosine spread (how close to the gate this arm sits)
        M = torch.stack([v / v.norm() for v in vecs])
        C = M @ M.T
        off = C[~torch.eye(n, dtype=torch.bool)]

        # supplementary: agreement with Small's DECODE partition, and with the
        # binary socialist-vs-rest split that the "socialist basin" names
        ari_dec = adjusted_rand_index(small_decode_labels, labels)
        ari_soc = adjusted_rand_index(small_socialist_binary, labels)

        rows.append({
            "arm": arm, "window": window,
            "converged": f"{converged}/{len(recs)}",
            "n_converged": converged,
            "tensor_basins": k,
            "basin_sizes": sorted(Counter(labels).values(), reverse=True),
            "decode_basins": len(decodes),
            "top_decodes": dict(decodes.most_common(3)),
            "offdiag_cos_mean": round(off.mean().item(), 5),
            "offdiag_cos_min": round(off.min().item(), 5),
            "ari_vs_small": ari,
            "perm_p": p,
            "ari_vs_small_lastvec": ari_last,
            "lastvec_basins": last_k,
            "ari_vs_small_decode_partition": ari_dec,
            "ari_vs_small_socialist_binary": ari_soc,
            "labels": labels,
        })
        if idx % 25 == 0 or idx == len(pt_files):
            print(f"  {idx}/{len(pt_files)} ...", flush=True)
        _ = by_pid  # (kept for provenance; decode already summarised)

    # ---- multiple-comparison control over the 277-arm family ---------------
    pvals = [r["perm_p"] for r in rows]
    m = len(rows)
    bonf_alpha = 0.05 / m
    rejected_bh, qvals = benjamini_hochberg(pvals, alpha=0.05)
    for r, q, rej in zip(rows, qvals, rejected_bh):
        r["bh_q"] = q
        r["bh_reject_q05"] = rej
        r["bonferroni_sig"] = r["perm_p"] < bonf_alpha
        r["beats_baseline_ari"] = r["ari_vs_small"] > BASELINE_ARI

    rows.sort(key=lambda r: (-r["ari_vs_small"], r["perm_p"]))

    # ---- high-resolution permutation + threshold sweep for the top arms ----
    top = rows[:args.n_top]
    print(f"\nHigh-resolution permutation ({args.n_perm_top}) + threshold sweep "
          f"for the top {len(top)} arms ...", flush=True)
    for r in top:
        pt = CENSUS_TERMS / f"{r['arm']}.pt"
        vecs = load_arm_vecs(pt, r["arm"], prompt_ids)
        r["perm_p_highres"] = permutation_p(small_labels, r["labels"],
                                            r["ari_vs_small"],
                                            n_perm=args.n_perm_top)
        sweep = []
        for thr in SWEEP_THRESHOLDS:
            s_lab, s_k = small_by_thr[thr]
            lab, k = cluster(vecs, thr)
            a = adjusted_rand_index(s_lab, lab)
            sweep.append({"threshold": thr, "small_basins": s_k, "basins": k,
                          "ari_vs_small": round(a, 4),
                          "perm_p": round(permutation_p(s_lab, lab, a,
                                                        n_perm=args.n_perm), 5)})
        r["threshold_sweep"] = sweep
        # permutation p for the two supplementary reference partitions
        r["perm_p_decode_partition"] = permutation_p(
            small_decode_labels, r["labels"],
            r["ari_vs_small_decode_partition"], n_perm=args.n_perm)
        r["perm_p_socialist_binary"] = permutation_p(
            small_socialist_binary, r["labels"],
            r["ari_vs_small_socialist_binary"], n_perm=args.n_perm)
        print(f"  {r['arm']:<8} ARI={r['ari_vs_small']:.4f} "
              f"p10k={r['perm_p']:.5f} p{args.n_perm_top//1000}k="
              f"{r['perm_p_highres']:.6f}  "
              f"ARI_decode={r['ari_vs_small_decode_partition']:.4f}"
              f"(p={r['perm_p_decode_partition']:.4f})  "
              f"ARI_socialist={r['ari_vs_small_socialist_binary']:.4f}"
              f"(p={r['perm_p_socialist_binary']:.4f})", flush=True)

    # ---- non-independence: how much do the top arms duplicate each other? --
    # 277 tests are NOT 277 independent looks. Cross-ARI among the leaders and
    # the A0 baseline shows whether they are one correlated family.
    fam = [("A0", refs["A0"]["labels"])] + [(r["arm"], r["labels"]) for r in top[:8]]
    cross = {}
    for a_name, a_lab in fam:
        cross[a_name] = {b_name: round(adjusted_rand_index(a_lab, b_lab), 4)
                         for b_name, b_lab in fam}

    # reference-arm threshold sweeps, for side-by-side
    for arm in ("A0", "A4"):
        sweep = []
        for thr in SWEEP_THRESHOLDS:
            s_lab, s_k = small_by_thr[thr]
            lab, k = cluster(refs[arm]["vecs"], thr)
            a = adjusted_rand_index(s_lab, lab)
            sweep.append({"threshold": thr, "small_basins": s_k, "basins": k,
                          "ari_vs_small": round(a, 4),
                          "perm_p": round(permutation_p(s_lab, lab, a,
                                                        n_perm=args.n_perm), 5)})
        refs[arm]["threshold_sweep"] = sweep
        refs[arm].pop("vecs")

    # ---- summary counters --------------------------------------------------
    n_beat = sum(r["beats_baseline_ari"] for r in rows)
    n_p05 = sum(r["perm_p"] < 0.05 for r in rows)
    n_bonf = sum(r["bonferroni_sig"] for r in rows)
    n_bh = sum(r["bh_reject_q05"] for r in rows)
    n_both = sum(r["beats_baseline_ari"] and r["bh_reject_q05"] for r in rows)

    report = {
        "n_prompts": n,
        "prompt_ids": prompt_ids,
        "cluster_threshold": CLUSTER_THRESHOLD,
        "n_perm_sweep": args.n_perm,
        "n_perm_top": args.n_perm_top,
        "small_reference": {"window": "0->11", "basins": small_n,
                            "labels": small_labels,
                            "decode_labels": small_decode_labels,
                            "socialist_binary": small_socialist_binary,
                            "n_socialist": n_soc},
        "cross_arm_ari_top_family": cross,
        "reference_arms": refs,
        "baseline_ari_to_beat": BASELINE_ARI,
        "n_arms": m,
        "summary": {
            "arms_with_ari_above_baseline": n_beat,
            "arms_p_lt_0.05_uncorrected": n_p05,
            "expected_false_positives_at_0.05": round(0.05 * m, 1),
            "bonferroni_alpha": bonf_alpha,
            "arms_bonferroni_significant": n_bonf,
            "arms_bh_fdr_q05_significant": n_bh,
            "arms_beating_baseline_AND_bh_significant": n_both,
        },
        "arms": rows,
    }
    Path(args.out).write_text(json.dumps(report, indent=2))

    # ---- print -------------------------------------------------------------
    print("\n=== top arms by ARI vs GPT-2 Small's native partition ===")
    print(f"{'rank':<5}{'arm':<9}{'window':<9}{'conv':<7}{'basins':<8}"
          f"{'dec':<5}{'ARI':<9}{'p(10k)':<10}{'p(hi-res)':<12}{'BH q':<10}")
    for i, r in enumerate(rows[:args.n_top], 1):
        hr = r.get("perm_p_highres")
        print(f"{i:<5}{r['arm']:<9}{r['window']:<9}{r['converged']:<7}"
              f"{r['tensor_basins']:<8}{r['decode_basins']:<5}"
              f"{r['ari_vs_small']:<9.4f}{r['perm_p']:<10.5f}"
              f"{(f'{hr:.6f}' if hr is not None else '-'):<12}{r['bh_q']:<10.4f}")
    print("\n=== non-independence: cross-ARI among the leading arms + A0 ===")
    names = list(cross)
    print("        " + "".join(f"{x:<9}" for x in names))
    for a in names:
        print(f"{a:<8}" + "".join(f"{cross[a][b]:<9.3f}" for b in names))

    print(f"\nreference A0 (0->23) ARI={refs['A0']['ari']:.4f} p={refs['A0']['p']:.5f}")
    print(f"reference A4 (10->21) ARI={refs['A4']['ari']:.4f} p={refs['A4']['p']:.5f}")
    print(f"\narms tested: {m}")
    print(f"  ARI > baseline {BASELINE_ARI}:            {n_beat}")
    print(f"  uncorrected p < 0.05:                 {n_p05} "
          f"(expected by chance: {0.05 * m:.1f})")
    print(f"  Bonferroni (alpha={bonf_alpha:.2e}):        {n_bonf}")
    print(f"  BH-FDR q<=0.05:                       {n_bh}")
    print(f"  beats baseline AND BH-significant:    {n_both}")
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
