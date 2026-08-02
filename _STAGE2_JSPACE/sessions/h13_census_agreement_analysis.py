"""Issue #73 adjudication analysis: is the flagship cells' high two-readout
agreement at extraction depth 21 a property of their terminal class, or an
outlier property of the two cells?

Reads only committed JSON characterisation artifacts (no model runs, no .pt
files, stdlib only): the census, full, scan and infill tier files under
experiments/exp_010c_windows/output/. Writes
h13_census_agreement_analysis_output.json next to this script and prints a
summary.

Terminology used throughout: a "cell" is a window (inject layer i ->
extract depth j); "agreement" is the per-cell count, out of 25 prompts, on
which the direct logit-lens readout and the decode-via-tail readout name
the same terminal token; "W*" marks the whole-word AND prompt-dependent
arm class (the EXP_010c-3 spec section 3 mechanical rule, reimplemented
here identically to build_final_map.py). Cells at j=23 are excluded from
all agreement statistics because their recorded figure is not a via-tail
measurement (the tail is empty at j=23; see RESULTS_EXP010C.md 2026-07-29
characterisation section).
"""
import json
import pathlib
import random
import statistics
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "experiments" / "exp_010c_windows" / "output"
RESULT_PATH = HERE / "h13_census_agreement_analysis_output.json"

FLAGSHIPS = {(8, 21), (10, 21)}


def tok_class(t):
    """Character-based token class (EXP_010c3_SPEC section 3, with its
    recorded edges). Identical to build_final_map.py."""
    if not any(c.isalnum() for c in t):
        return "punctuation"
    if t.startswith(" "):
        body = t[1:]
        if body.isalpha():
            return "abbrev/symbol" if (body.isupper() and len(body) > 1) else "whole-word"
        return "abbrev/symbol"
    if t.isalpha():
        return "fragment"
    return "abbrev/symbol"


def arm_class(toks):
    """Arm class from the unique terminal set. Identical to build_final_map.py."""
    classes = {tok_class(t) for t in toks}
    if classes == {"whole-word"}:
        return "whole-word"
    if len(toks) == 1 and classes == {"punctuation"}:
        return "punctuation funnel"
    return "mixed"


def load_cells():
    """Every measured cell from the four characterisation JSONs."""
    cells = {}
    for tier in ("census", "full", "scan", "infill"):
        for rec in json.load(open(OUT_DIR / f"terminal_characterisation_{tier}.json")):
            i, j = (int(x) for x in rec["window"].split("->"))
            agree = int(rec["tail_agreement"].split("/")[0])
            toks = list(rec["decode_terminals"])
            cells[(i, j)] = {
                "tier": tier,
                "arm": rec["arm"],
                "agree": agree,
                "n": rec["n"],
                "class": arm_class(toks),
                "prompt_dependent": len(toks) >= 2,
                "unique_terminals": len(toks),
            }
    return cells


def is_wstar(c):
    return c["class"] == "whole-word" and c["prompt_dependent"]


def main():
    cells = load_cells()
    assert len(cells) == 300, f"expected 300 cells, got {len(cells)}"
    for k in FLAGSHIPS:
        assert cells[k]["tier"] in ("scan", "full")
    assert cells[(8, 21)]["agree"] == 17 and cells[(10, 21)]["agree"] == 23

    # Via-tail-valid cells only (j < 23).
    vt = {k: c for k, c in cells.items() if k[1] < 23}
    wstar_all = sorted(k for k, c in cells.items() if is_wstar(c))
    out = {
        "cells_total": len(cells),
        "via_tail_cells": len(vt),
        "wstar_cells_total": len(wstar_all),
        "wstar_cells": [f"{i}->{j}" for i, j in wstar_all],
    }

    # ---- Class comparison over census cells (the chartered test) ----
    census = {k: c for k, c in vt.items() if c["tier"] == "census"}
    cw = [c["agree"] for c in census.values() if is_wstar(c)]
    co = [c["agree"] for c in census.values() if not is_wstar(c)]
    out["census_class_comparison_raw"] = {
        "wstar_n": len(cw), "wstar_mean": round(statistics.mean(cw), 2),
        "wstar_values_sorted": sorted(cw),
        "other_n": len(co), "other_mean": round(statistics.mean(co), 2),
        "note": "raw means, not depth-adjusted; scale 0 to 25 prompts",
    }

    # Depth-stratified permutation test over census cells at j < 23:
    # statistic = mean over W* cells of (agree - mean agreement of ALL census
    # cells at the same depth j). Labels shuffled within depth strata.
    strata = defaultdict(list)
    for k, c in census.items():
        strata[k[1]].append((is_wstar(c), c["agree"]))
    depth_mean = {j: statistics.mean(a for _, a in v) for j, v in strata.items()}
    obs = statistics.mean(
        c["agree"] - depth_mean[k[1]] for k, c in census.items() if is_wstar(c))
    rng = random.Random(20260802)
    n_perm, ge = 10000, 0
    for _ in range(n_perm):
        excesses = []
        for j, v in strata.items():
            flags = [f for f, _ in v]
            rng.shuffle(flags)
            for f, (_, a) in zip(flags, v):
                if f:
                    excesses.append(a - depth_mean[j])
        if excesses and statistics.mean(excesses) >= obs:
            ge += 1
    out["census_depth_stratified_class_test"] = {
        "wstar_mean_excess_over_same_depth_mean": round(obs, 2),
        "permutations": n_perm,
        "p_one_sided": round((ge + 1) / (n_perm + 1), 4),
        "note": ("excess is in prompts out of 25; positive means W* census "
                 "cells agree more than same-depth census cells overall"),
    }

    # Per-depth means, census cells, reproducing the recorded baselines.
    out["census_mean_agreement_by_depth"] = {
        str(j): {"arms": len(strata[j]), "mean": round(depth_mean[j], 1)}
        for j in sorted(strata)}

    # ---- The full j=21 column, all tiers ----
    col = sorted((k for k in vt if k[1] == 21), key=lambda k: -vt[k]["agree"])
    out["j21_column_all_tiers"] = [
        {"cell": f"{i}->{j}", "tier": vt[(i, j)]["tier"],
         "class": vt[(i, j)]["class"],
         "prompt_dependent": vt[(i, j)]["prompt_dependent"],
         "agree": vt[(i, j)]["agree"],
         "flagship": (i, j) in FLAGSHIPS}
        for i, j in col]
    non_flag = [vt[k]["agree"] for k in col if k not in FLAGSHIPS]
    out["j21_column_summary"] = {
        "cells": len(col),
        "non_flagship_mean": round(statistics.mean(non_flag), 1),
        "non_flagship_median": statistics.median(non_flag),
        "rank_of_10_21": 1 + sum(a > 23 for a in non_flag),
        "rank_of_8_21": 1 + sum(a > 17 for a in non_flag) + 1,  # 10->21 above it
        "non_flagship_at_or_above_17": sum(a >= 17 for a in non_flag),
        "non_flagship_at_or_above_23": sum(a >= 23 for a in non_flag),
    }

    # ---- Within the W* class (all tiers, j < 23) ----
    wvt = sorted((k for k in vt if is_wstar(vt[k])), key=lambda k: -vt[k]["agree"])
    out["wstar_via_tail_distribution"] = [
        {"cell": f"{i}->{j}", "agree": vt[(i, j)]["agree"],
         "flagship": (i, j) in FLAGSHIPS} for i, j in wvt]
    wa = [vt[k]["agree"] for k in wvt]
    out["wstar_via_tail_summary"] = {
        "n": len(wa), "mean": round(statistics.mean(wa), 1),
        "median": statistics.median(wa), "min": min(wa), "max": max(wa),
    }

    # W* vs non-W* at the depths where both exist and agreement is informative.
    per_depth = {}
    for j in (20, 21, 22):
        w = [vt[k]["agree"] for k in vt if k[1] == j and is_wstar(vt[k])]
        o = [vt[k]["agree"] for k in vt if k[1] == j and not is_wstar(vt[k])]
        per_depth[str(j)] = {
            "wstar_n": len(w), "wstar_values": sorted(w),
            "other_n": len(o), "other_mean": round(statistics.mean(o), 1) if o else None,
        }
    out["late_depth_class_split_all_tiers"] = per_depth

    with open(RESULT_PATH, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
