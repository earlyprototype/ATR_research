"""EXP_010c-4 final map + mechanical H12 evaluation.

Builds the complete (i,j) map from ALL registered artifacts (full, scan,
infill tiers + census shards), applies the EXP_010c3_SPEC §3 arm-class
rule, and evaluates H12 per the EXP_010c4_SPEC §6 amendment (eligible =
census cells with >=1 already-measured neighbour on the valid lattice).

Pure analysis, no model time. Usage: build_final_map.py [--require-complete]
"""
import json
import pathlib
import sys
from collections import Counter

BASE = pathlib.Path("/home/user/ATR_research/_STAGE2_JSPACE/experiments/exp_010c_windows/output")
REQUIRE_COMPLETE = "--require-complete" in sys.argv


def tok_class(t):
    """Character-based token class (EXP_010c3_SPEC §3, with its recorded edges)."""
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
    """Arm class from its unique terminal set (EXP_010c3_SPEC §3)."""
    classes = {tok_class(t) for t in toks}
    if classes == {"whole-word"}:
        return "whole-word"
    if len(toks) == 1 and classes == {"punctuation"}:
        return "punctuation funnel"
    return "mixed"


def load_all():
    """Every registered (i,j) cell: prior tiers (monolithic) + census shards."""
    cells = {}

    def add(window, records):
        i, j = (int(x) for x in window.split("->"))
        toks = Counter(r["terminal_token"] for r in records)
        cells[(i, j)] = {
            "n": len(records),
            "converged": sum(r["converged"] for r in records),
            "toks": toks,
            "unique": len(toks),
            "class": arm_class(list(toks)),
            "prompt_dependent": len(toks) >= 2,
            "margin_mean": sum(r["top_logit_margin"] for r in records) / len(records),
            "locks": [r["lock_in_iter"] for r in records if r["lock_in_iter"]],
        }

    for tier in ("full", "scan", "infill"):
        p = BASE / f"results_{tier}.json"
        if p.exists():
            rs = json.load(open(p))
            for arm in {r["arm"] for r in rs}:
                sub = [r for r in rs if r["arm"] == arm]
                add(sub[0]["window"], sub)

    d = BASE / "results_census"
    if d.is_dir():
        for f in sorted(d.glob("*.json")):
            rs = json.load(open(f))
            if len(rs) == 25:
                add(rs[0]["window"], rs)
    return cells


# Cells measured by the pre-census registered tiers (H12's "already-measured")
PRIOR = {(0, 23), (10, 21), (0, 11), (6, 17), (12, 23), (8, 15),
         (10, 22), (10, 23), (0, 21), (4, 21), (6, 21), (8, 21), (12, 21), (14, 21),
         (9, 21), (11, 21), (7, 21), (5, 21), (10, 19), (10, 17), (10, 15), (8, 19), (8, 17)}


def neighbours(i, j):
    """4-neighbourhood on the valid (i<=j<=23) lattice."""
    return [(a, b) for a, b in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1))
            if 0 <= a <= b <= 23]


def main():
    cells = load_all()
    total_valid = 24 * 25 // 2  # 300
    print(f"cells measured: {len(cells)}/{total_valid}")
    if REQUIRE_COMPLETE and len(cells) < total_valid:
        raise SystemExit(f"INCOMPLETE: {total_valid - len(cells)} cells missing — not final")

    runs = sum(c["n"] for c in cells.values())
    conv = sum(c["converged"] for c in cells.values())
    print(f"runs: {conv}/{runs} converged")

    classes = Counter(c["class"] for c in cells.values())
    print("arm classes:", dict(classes))
    pd = [k for k, c in cells.items() if c["prompt_dependent"]]
    print(f"prompt-dependent cells: {len(pd)}")

    ww_pd = sorted(k for k, c in cells.items() if c["class"] == "whole-word" and c["prompt_dependent"])
    ww_fun = sorted(k for k, c in cells.items() if c["class"] == "whole-word" and not c["prompt_dependent"])
    print(f"\nWHOLE-WORD + PROMPT-DEPENDENT ({len(ww_pd)}) — the J-lens target set:")
    for k in ww_pd:
        print(f"  {k[0]:>2}->{k[1]:<2} {dict(cells[k]['toks'].most_common(4))}")
    print(f"\nwhole-word funnels ({len(ww_fun)}): " +
          ", ".join(f"{a}->{b}" for a, b in ww_fun))

    # non-convergence inventory
    nc = {k: c["n"] - c["converged"] for k, c in cells.items() if c["converged"] < c["n"]}
    print(f"\nnon-convergence: {len(nc)} cells, {sum(nc.values())} runs")
    for k in sorted(nc, key=lambda k: -nc[k]):
        flag = " [>5/25 SYSTEMATIC]" if nc[k] > 5 else ""
        print(f"  {k[0]:>2}->{k[1]:<2} {nc[k]}/25{flag}")

    # D inventory
    dcells = [k for k, c in cells.items() if "D" in c["toks"]]
    print(f"\ncells producing 'D' at all: {dcells}")

    # ---- H12 (as amended, EXP_010c4_SPEC §6) ----
    census_cells = [k for k in cells if k not in PRIOR]
    eligible, differs = [], []
    for k in census_cells:
        meas_nb = [n for n in neighbours(*k) if n in PRIOR]
        if not meas_nb:
            continue
        eligible.append(k)
        if all(cells[k]["class"] != cells[n]["class"] for n in meas_nb if n in cells):
            differs.append((k, cells[k]["class"], [(n, cells[n]["class"]) for n in meas_nb if n in cells]))
    print(f"\nH12 (amended): eligible cells (>=1 already-measured neighbour): {len(eligible)}")
    print(f"H12: cells differing in class from EVERY measured neighbour: {len(differs)}")
    for k, cl, nbs in sorted(differs):
        print(f"  {k[0]:>2}->{k[1]:<2} {cl} vs " + ", ".join(f"{n[0]}->{n[1]} {c}" for n, c in nbs))
    print(f"\nH12 VERDICT: {'SUPPORTED' if differs else 'REFUTED'} "
          f"({len(differs)}/{len(eligible)} eligible cells differ from every measured neighbour)")

    # compact map for the record
    print("\nMAP (rows=inject i, cols=extract j; W=whole-word, P=punct funnel, m=mixed, .=n/a):")
    hdr = "    " + "".join(f"{j:>3}" for j in range(24))
    print(hdr)
    for i in range(24):
        row = f"{i:>3} "
        for j in range(24):
            if j < i:
                row += "   "
            elif (i, j) in cells:
                c = cells[(i, j)]
                ch = {"whole-word": "W", "punctuation funnel": "P", "mixed": "m"}[c["class"]]
                row += f"  {ch if not (c['class']=='whole-word' and c['prompt_dependent']) else 'W*'}"[-3:]
            else:
                row += "  ?"
        print(row)
    print("W* = whole-word AND prompt-dependent")


if __name__ == "__main__":
    main()
