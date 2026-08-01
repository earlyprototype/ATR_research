"""EXP_015 — Does the Small-like prompt partition survive natural loudness?
Pre-registered spec: ../../EXP_015_SPEC.md (committed before this file existed).

Pure analysis on committed artifacts; no model is loaded. Reuses the EXP_010d
machinery unmodified: cluster() from analyze_terminals.py, adjusted_rand_index()
and permutation_p() from compare_small_basins.py. New code here is input
adaptation only (spec §5).

Inputs (spec §4):
    output/terminals_small_010d.pt        Small reference terminals (EXP_010d)
    output/terminals_energynorm_A0.pt     Medium A0 natural-loudness terminals
                                          (EXP_010c-VARIANTS Control B)
    output/terminals_full.pt              Medium A0 loud-convention terminals
                                          (reproduction gate only)
    output/prompt_subset.json             registered 25-prompt subset (order)

Usage: python exp015_natural_ari.py [--n-perm 10000]
"""

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))   # atr_engine2 (transitively imported)
sys.path.insert(0, str(HERE))

from analyze_terminals import cluster, CLUSTER_THRESHOLD  # noqa: E402
from compare_small_basins import adjusted_rand_index, permutation_p  # noqa: E402

SWEEP_THRESHOLDS = (0.99, 0.995, 0.999, 0.9995)  # spec §5 step 5 (EXP_010d sweep)
GATE_EXPECTED = {"ari_3dp": 0.200, "p_4dp": 0.0009}  # RESULTS_EXP010D.md record


def load_terminals(path):
    """Load a terminals archive; tolerate tuple-keyed (pre-PR#4) and
    string-keyed ("ARM|PID", post-PR#4) layouts. Returns {(arm, pid): entry}."""
    try:
        t = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # legacy pickle needs full unpickling (committed artifact)
        t = torch.load(path, map_location="cpu", weights_only=False)
    out = {}
    for k, v in t.items():
        if isinstance(k, tuple):
            out[(k[0], k[1])] = v
        else:
            arm, pid = k.split("|", 1)
            out[(arm, pid)] = v
    return out


def labels_for(terms, arm, prompt_ids, threshold):
    vecs = [terms[(arm, pid)]["mean"] for pid in prompt_ids]
    return cluster(vecs, threshold=threshold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=10000)
    args = ap.parse_args()

    outdir = HERE / "output"
    subset = json.load(open(outdir / "prompt_subset.json"))
    prompt_ids = [r["id"] for r in subset]
    assert len(prompt_ids) == 25

    small = load_terminals(outdir / "terminals_small_010d.pt")
    loud = load_terminals(outdir / "terminals_full.pt")
    natural = load_terminals(outdir / "terminals_energynorm_A0.pt")

    report = {"experiment": "EXP_015", "spec": "../../EXP_015_SPEC.md",
              "prompt_ids": prompt_ids, "cluster_threshold": CLUSTER_THRESHOLD,
              "n_perm": args.n_perm, "perm_seed": 42}

    # ---- step 1: reproduction gate (spec §5.1) ----------------------------
    s_lab, s_n = labels_for(small, "SMALL", prompt_ids, CLUSTER_THRESHOLD)
    g_lab, g_n = labels_for(loud, "A0", prompt_ids, CLUSTER_THRESHOLD)
    g_ari = adjusted_rand_index(s_lab, g_lab)
    g_p = permutation_p(s_lab, g_lab, g_ari, n_perm=args.n_perm)
    gate_pass = (round(g_ari, 3) == GATE_EXPECTED["ari_3dp"]
                 and round(g_p, 4) == GATE_EXPECTED["p_4dp"])
    report["reproduction_gate"] = {
        "expected": GATE_EXPECTED,
        "small_basins": s_n, "loud_A0_basins": g_n,
        "ari": round(g_ari, 4), "perm_p": round(g_p, 5), "pass": gate_pass}
    print(f"[gate] Small basins {s_n}, loud A0 basins {g_n}, "
          f"ARI {g_ari:.4f}, perm p {g_p:.5f} -> {'PASS' if gate_pass else 'FAIL'}")
    if not gate_pass:
        (outdir / "exp015_natural_ari.json").write_text(json.dumps(report, indent=2))
        print("STOP (spec §6): reproduction gate failed — machinery does not "
              "reproduce the EXP_010d record. Recording and stopping.")
        return 1

    # ---- steps 2-4: primary comparison at the registered threshold --------
    n_lab, n_n = labels_for(natural, "A0", prompt_ids, CLUSTER_THRESHOLD)
    trivial = n_n in (1, len(prompt_ids))
    primary = {"natural_A0_basins": n_n, "small_basins": s_n,
               "trivial_partition_guard_fired": trivial}
    if trivial:
        primary["reading"] = "UNANSWERABLE at registered threshold (spec §6 guard)"
        print(f"[primary] natural A0 partition trivial ({n_n} basins) — "
              "guard fired; sweep reported descriptively.")
    else:
        ari = adjusted_rand_index(s_lab, n_lab)
        p = permutation_p(s_lab, n_lab, ari, n_perm=args.n_perm)
        primary.update(ari=round(ari, 4), perm_p=round(p, 5),
                       labels_small=s_lab, labels_natural=n_lab)
        h15 = ari > 0 and p < 0.05
        primary["h15"] = "SUPPORTED" if h15 else "REFUTED"
        print(f"[primary] Small basins {s_n}, natural A0 basins {n_n}")
        print(f"[primary] ARI(Small, A0-natural) = {ari:.4f}, perm p = {p:.5f}")
        print(f"[primary] H15 (spec §6, mechanical): {primary['h15']}")
    report["primary"] = primary

    # ---- step 5: threshold sweep, descriptive only ------------------------
    sweep = []
    for thr in SWEEP_THRESHOLDS:
        sl, sn = labels_for(small, "SMALL", prompt_ids, thr)
        nl, nn = labels_for(natural, "A0", prompt_ids, thr)
        a = adjusted_rand_index(sl, nl)
        pp = permutation_p(sl, nl, a, n_perm=args.n_perm)
        sweep.append({"threshold": thr, "small_basins": sn,
                      "natural_A0_basins": nn, "ari": round(a, 4),
                      "perm_p": round(pp, 5)})
    report["threshold_sweep"] = sweep
    print("\n[sweep] thr    S_basins  N_basins  ARI      perm_p")
    for row in sweep:
        print(f"[sweep] {row['threshold']:<7}{row['small_basins']:<10}"
              f"{row['natural_A0_basins']:<10}{row['ari']:<9}{row['perm_p']}")

    out = outdir / "exp015_natural_ari.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nSaved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
