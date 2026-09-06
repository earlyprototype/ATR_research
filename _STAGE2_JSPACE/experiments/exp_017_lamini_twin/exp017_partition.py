"""EXP_017 Part 1 analysis: H18 (the partition) and H18a (the readout tokens).

Spec: ../../EXP_017_SPEC.md sections 5.6 and 5.7.

Pure analysis on this experiment's own loop artifacts plus two committed
reference files. The comparison machinery is the registered EXP_010d machinery
imported unmodified, exactly as EXP_015 imported it: cluster() from
analyze_terminals.py, adjusted_rand_index() and permutation_p() from
compare_small_basins.py. New code here is input adaptation and scoring only.

Usage: python3 exp017_partition.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
WINDOWS = EXPERIMENTS / "exp_010c_windows"
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(WINDOWS))

from analyze_terminals import cluster, CLUSTER_THRESHOLD  # noqa: E402
from compare_small_basins import adjusted_rand_index, permutation_p  # noqa: E402

OUT = HERE / "output"
SWEEP = (0.99, 0.995, 0.999, 0.9995)
N_PERM = 10000
STAGE1_BASIN_TOKENS = [" prolet", " Divine", " till", " Anarch", " solidarity"]


def load_npz_means(which, prompt_ids):
    z = np.load(OUT / f"terminals_{which}.npz")
    return [torch.from_numpy(z[f"{pid}|mean"]) for pid in prompt_ids]


def load_committed(path):
    """Committed terminals archive; tolerates the tuple-keyed and the
    string-keyed layouts, as exp015_natural_ari.py does."""
    try:
        t = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        t = torch.load(path, map_location="cpu", weights_only=False)
    out = {}
    for k, v in t.items():
        if isinstance(k, tuple):
            out[(k[0], k[1])] = v
        else:
            arm, pid = k.split("|", 1)
            out[(arm, pid)] = v
    return out


def compare(labels_ref, labels_other, n_perm=N_PERM):
    ari = adjusted_rand_index(labels_ref, labels_other)
    p = permutation_p(labels_ref, labels_other, ari, n_perm=n_perm)
    return round(ari, 4), round(p, 5)


def main():
    subset = json.load(open(WINDOWS / "output" / "prompt_subset_small.json"))
    prompt_ids = [r["id"] for r in subset]
    assert len(prompt_ids) == 25
    loops = {m: {r["prompt_id"]: r for r in
                 json.load(open(OUT / f"loop_results_{m}.json"))}
             for m in ("twin", "base")}

    rep = {"experiment": "EXP_017", "spec": "../../EXP_017_SPEC.md",
           "cluster_threshold": CLUSTER_THRESHOLD, "n_perm": N_PERM,
           "perm_seed": 42, "prompt_ids": prompt_ids}

    # ---- per-prompt table ---------------------------------------------------
    table = []
    for pid in prompt_ids:
        row = {"prompt_id": pid, "category": loops["base"][pid]["category"]}
        for m in ("base", "twin"):
            r = loops[m][pid]
            row[f"{m}_terminal"] = r["terminal_token"]
            row[f"{m}_terminal_id"] = r["terminal_token_id"]
            row[f"{m}_prob"] = round(r["terminal_prob"], 4)
            row[f"{m}_lock_in_iter"] = r["lock_in_iter"]
            row[f"{m}_converged"] = r["converged"]
            row[f"{m}_top5"] = r["top5_tokens"]
            row[f"{m}_loudness_ratio"] = round(r["loudness_ratio"], 2)
        table.append(row)
    rep["per_prompt"] = table

    for m in ("twin", "base"):
        rs = [loops[m][p] for p in prompt_ids]
        ratios = [r["loudness_ratio"] for r in rs]
        locks = [r["lock_in_iter"] for r in rs if r["lock_in_iter"] is not None]
        rep[f"{m}_summary"] = {
            "converged": sum(r["converged"] for r in rs),
            "lock_in_iters": sorted(set(locks)),
            "terminal_tokens": dict(Counter(r["terminal_token"] for r in rs).most_common()),
            "loudness_ratio_mean": round(float(np.mean(ratios)), 2),
            "loudness_ratio_range": [round(min(ratios), 2), round(max(ratios), 2)],
            "mean_terminal_prob": round(float(np.mean([r["terminal_prob"] for r in rs])), 4),
            "extra_vocab_token_used": any(r["terminal_token_id"] >= 50257 for r in rs),
            "lag_scan_mean": {k: round(float(np.mean([r["lag_scan"][k] for r in rs])), 4)
                              for k in rs[0]["lag_scan"]} if rs[0].get("lag_scan") else None,
        }

    # ---- H18: the primary partition comparison ------------------------------
    means = {m: load_npz_means(m, prompt_ids) for m in ("twin", "base")}
    lab = {m: cluster(means[m], threshold=CLUSTER_THRESHOLD) for m in means}
    twin_lab, twin_n = lab["twin"]
    base_lab, base_n = lab["base"]
    trivial = {m: lab[m][1] in (1, 25) for m in lab}
    primary = {"twin_basins": twin_n, "base_basins": base_n,
               "twin_labels": twin_lab, "base_labels": base_lab,
               "trivial_partition": trivial}
    if any(trivial.values()):
        primary["h18"] = "UNTESTABLE at the registered threshold"
        primary["reason"] = (
            "the degenerate-partition guard fired: "
            + ", ".join(f"{m} has {lab[m][1]} groups over 25 prompts"
                        for m in lab if trivial[m])
            + "; a grouping with one group or with 25 groups has no substructure "
              "to compare, so the adjusted Rand index is not meaningful here")
        ari, p = compare(base_lab, twin_lab)
        primary["ari_if_forced"] = ari
        primary["perm_p_if_forced"] = p
    else:
        ari, p = compare(base_lab, twin_lab)
        primary.update(ari=ari, perm_p=p)
        primary["h18"] = "SUPPORTED" if (ari > 0 and p < 0.05) else "REFUTED"
    rep["h18_primary"] = primary

    # ---- threshold sweep, descriptive --------------------------------------
    sweep = []
    for thr in SWEEP:
        tl, tn = cluster(means["twin"], threshold=thr)
        bl, bn = cluster(means["base"], threshold=thr)
        row = {"threshold": thr, "twin_basins": tn, "base_basins": bn}
        # The same degeneracy rule the primary comparison uses, and for the
        # same reason: a grouping with one group, or with one group per prompt,
        # has no substructure to compare, so the adjusted Rand index is not
        # meaningful. Both sides must be non-degenerate, not just one of them.
        degenerate = [m for m, n in (("twin", tn), ("base", bn)) if n in (1, 25)]
        if not degenerate:
            a, pp = compare(bl, tl)
            row.update(ari=a, perm_p=pp)
        else:
            a, pp = compare(bl, tl)
            row.update(ari=None, perm_p=None,
                       degenerate_sides=degenerate,
                       note=("the degenerate-partition guard fired on "
                             + " and ".join(degenerate)
                             + "; a grouping with one group or with one group "
                               "per prompt has no substructure to compare"),
                       ari_if_forced=a, perm_p_if_forced=pp)
        sweep.append(row)
    rep["h18_threshold_sweep"] = sweep

    # ---- secondary reading 1: the committed lag-1 arm SB terminals ----------
    sb = load_committed(WINDOWS / "output" / "terminals_small010b_SB.pt")
    sb_means = [sb[("SB", pid)]["mean"] for pid in prompt_ids]
    sb_lab, sb_n = cluster(sb_means, threshold=CLUSTER_THRESHOLD)
    sec1 = {"what": "base GPT-2 Small arm SB as committed by EXP_010b, lag-1 gate",
            "base_lag1_basins": sb_n, "labels": sb_lab}
    if sb_n not in (1, 25) and twin_n not in (1, 25):
        a, pp = compare(sb_lab, twin_lab)
        sec1.update(ari_vs_twin=a, perm_p=pp)
    else:
        sec1["note"] = "one side trivial at the registered threshold"
        a, pp = compare(sb_lab, twin_lab)
        sec1.update(ari_vs_twin_if_forced=a, perm_p_if_forced=pp)
    a2, p2 = compare(sb_lab, base_lab)
    sec1["ari_lag1_vs_this_runs_lag2_base"] = a2
    sec1["perm_p_lag1_vs_this_runs_lag2_base"] = p2
    rep["h18_secondary_committed_lag1"] = sec1

    # ---- secondary reading 2: EXP_010d's Small reference, 20 shared prompts --
    d010 = load_committed(WINDOWS / "output" / "terminals_small_010d.pt")
    sub010 = json.load(open(WINDOWS / "output" / "prompt_subset.json"))
    ids010 = [r["id"] for r in sub010]
    shared = [p for p in prompt_ids if p in set(ids010)]
    sec2 = {"what": "EXP_010d's committed Small reference partition, restricted "
                    "to the prompts the two 25-prompt lists share",
            "n_shared_prompts": len(shared), "shared_prompt_ids": shared}
    if len(shared) >= 10:
        ref_lab, ref_n = cluster([d010[("SMALL", p)]["mean"] for p in shared],
                                 threshold=CLUSTER_THRESHOLD)
        tw_lab_s, tw_n_s = cluster([means["twin"][prompt_ids.index(p)] for p in shared],
                                   threshold=CLUSTER_THRESHOLD)
        bs_lab_s, bs_n_s = cluster([means["base"][prompt_ids.index(p)] for p in shared],
                                   threshold=CLUSTER_THRESHOLD)
        sec2.update(exp010d_basins=ref_n, twin_basins=tw_n_s, base_basins=bs_n_s)
        a, pp = compare(ref_lab, tw_lab_s)
        sec2.update(ari_exp010d_vs_twin=a, perm_p=pp)
        a, pp = compare(ref_lab, bs_lab_s)
        sec2.update(ari_exp010d_vs_this_runs_base=a, perm_p_base=pp)
    rep["h18_secondary_exp010d"] = sec2

    # ---- H18a: the readout tokens ------------------------------------------
    twin_tokens = [loops["twin"][p]["terminal_token"] for p in prompt_ids]
    base_tokens = [loops["base"][p]["terminal_token"] for p in prompt_ids]
    a_hits = [p for p, t in zip(prompt_ids, twin_tokens) if t in STAGE1_BASIN_TOKENS]
    base_set = sorted(set(base_tokens))
    b_hits = [p for p, t in zip(prompt_ids, twin_tokens) if t in base_set]
    twin_counts = Counter(twin_tokens)
    rep["h18a"] = {
        "threshold": "at least 13 of 25, being at least half of 25",
        "reading_a_verdict_bearing": {
            "token_set": STAGE1_BASIN_TOKENS,
            "set_source": "Stage 1's five established basin tokens",
            "hits": len(a_hits), "of": 25, "prompts": a_hits,
            "h18a": "SUPPORTED" if len(a_hits) >= 13 else "REFUTED"},
        "reading_b_secondary": {
            "token_set": base_set,
            "set_source": "the tokens base GPT-2 Small produced in this run's "
                          "own matched lag-2 arm",
            "hits": len(b_hits), "of": 25, "prompts": b_hits,
            "reading": "would be SUPPORTED" if len(b_hits) >= 13 else
                       "would be REFUTED"},
        "twin_repeated_terminal": {
            "token": twin_counts.most_common(1)[0][0],
            "count": twin_counts.most_common(1)[0][1],
            "all_counts": dict(twin_counts.most_common())},
        "base_terminal_counts": dict(Counter(base_tokens).most_common()),
        "twin_top1_equals_base_top1_same_prompt":
            sum(1 for a, b in zip(twin_tokens, base_tokens) if a == b),
        "twin_top5_contains_a_stage1_basin_token":
            sum(1 for p in prompt_ids
                if any(t in STAGE1_BASIN_TOKENS for t in loops["twin"][p]["top5_tokens"])),
    }

    (OUT / "exp017_partition.json").write_text(json.dumps(rep, indent=2))

    # ---- printed summary ----------------------------------------------------
    print("=== EXP_017 Part 1 ===")
    for m in ("base", "twin"):
        s = rep[f"{m}_summary"]
        print(f"{m:<5} converged {s['converged']}/25, locks {s['lock_in_iters']}, "
              f"loudness {s['loudness_ratio_mean']}x {s['loudness_ratio_range']}")
        print(f"      terminals: {s['terminal_tokens']}")
    print(f"\nH18: twin {twin_n} basins, base {base_n} basins -> {primary['h18']}")
    for k in ("ari", "perm_p", "ari_if_forced", "perm_p_if_forced"):
        if k in primary:
            print(f"     {k} = {primary[k]}")
    print(f"H18a reading A: {len(a_hits)}/25 -> "
          f"{rep['h18a']['reading_a_verdict_bearing']['h18a']}")
    print(f"H18a reading B: {len(b_hits)}/25 ({rep['h18a']['reading_b_secondary']['reading']})")
    print(f"\nSweep: {json.dumps(sweep)}")
    print(f"Saved -> output/exp017_partition.json")


if __name__ == "__main__":
    main()
