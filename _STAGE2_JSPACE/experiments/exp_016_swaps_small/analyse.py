"""Aggregate the EXP_016 swap records into the tables the results record
reports. Applies the pre-registered tuning-then-held-out selection rule from
section 4 of `_STAGE2_JSPACE/EXP_016_SPEC.md`."""
from __future__ import annotations
import csv, json, sys
import os
from collections import defaultdict

D = os.path.dirname(os.path.abspath(__file__)) + "/"
SCORE = {"h17": "in_top5", "h17a": "in_top5", "h17b": "is_top1"}
# Final tie-break for choose(): the order the position modes were run in.
# Before this key existed, exact ties fell to CSV insertion order, which is
# this same order; the key makes that rule explicit and stable.
MODE_ORDER = ["all", "last", "all_no_bos", "from_mention", "answer_only"]


def load(battery):
    rows = []
    with open(D + f"output/records_{battery}.csv") as fh:
        for r in csv.DictReader(fh):
            for k in ("good_rank", "bad_rank", "in_top5", "is_top1",
                      "beats_bad", "seed"):
                r[k] = int(r[k])
            r["alpha"] = float(r["alpha"])
            rows.append(r)
    return rows


def cell_rates(rows, battery, split=None, key=None, extra_filter=None):
    """Success rate per (layer set, strength, position mode) and arm."""
    key = key or SCORE[battery]
    acc = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in rows:
        if split and r["split"] != split:
            continue
        if extra_filter and not extra_filter(r):
            continue
        cell = (r["layers"], r["alpha"], r["posmode"])
        a = acc[cell][r["arm"]]
        a[0] += r[key]; a[1] += 1
    out = {}
    for cell, arms in acc.items():
        out[cell] = {a: (n / d if d else 0.0, n, d) for a, (n, d) in arms.items()}
    return out


def choose(cells):
    """Pre-registered selection: highest lens success on the tuning half;
    ties by larger gap over control A, then smaller strength, then fewer
    layers, then lowest first layer; remaining ties by MODE_ORDER."""
    def k(item):
        cell, arms = item
        lens = arms.get("lens", (0, 0, 0))[0]
        ctrl = arms.get("randdir", (0, 0, 0))[0]
        ls, alpha, mode = cell
        n_l = len(ls.split("-"))
        return (-lens, -(lens - ctrl), alpha, n_l, int(ls.split("-")[0]),
                MODE_ORDER.index(mode))
    return sorted(cells.items(), key=k)[0][0]


def report(battery):
    rows = load(battery)
    key = SCORE[battery]
    tune = cell_rates(rows, battery, "tuning", key)
    held = cell_rates(rows, battery, "heldout", key)
    allr = cell_rates(rows, battery, None, key)
    best = choose(tune)
    out = dict(battery=battery, score_key=key, chosen_cell=list(best),
               n_records=len(rows),
               tuning=arms_of(tune, best), heldout=arms_of(held, best),
               overall=arms_of(allr, best))
    grid = []
    for cell in sorted(allr):
        grid.append(dict(layers=cell[0], alpha=cell[1], posmode=cell[2],
                         tuning=arms_of(tune, cell), heldout=arms_of(held, cell),
                         overall=arms_of(allr, cell)))
    out["grid"] = grid
    top = sorted(grid, key=lambda g: -g["overall"]["lens"][0])[:12]
    out["top_cells_overall"] = top
    return out, rows


def arms_of(cells, cell):
    a = cells.get(cell, {})
    return {k: [round(v[0], 4), v[1], v[2]] for k, v in a.items()}


if __name__ == "__main__":
    for b in sys.argv[1:]:
        out, rows = report(b)
        json.dump(out, open(D + f"output/summary_{b}.json", "w"), indent=1)
        c = out["chosen_cell"]
        print(f"\n=== {b}: chosen on the tuning half: layers {c[0]}, "
              f"strength {c[1]}, positions {c[2]}")
        for half in ("tuning", "heldout", "overall"):
            a = out[half]
            print(f"  {half:8s} lens {a.get('lens',[0,0,0])[0]:.3f} "
                  f"({a.get('lens',[0,0,0])[1]}/{a.get('lens',[0,0,0])[2]})  "
                  f"control A {a.get('randdir',[0,0,0])[0]:.3f}  "
                  f"control B {a.get('randnorm',[0,0,0])[0]:.3f}")
        print("  best cells overall (lens rate, control A, control B):")
        for g in out["top_cells_overall"][:6]:
            print(f"    layers {g['layers']:8s} alpha {g['alpha']:.1f} "
                  f"{g['posmode']:12s} lens {g['overall']['lens'][0]:.3f} "
                  f"A {g['overall'].get('randdir',[0])[0]:.3f} "
                  f"B {g['overall'].get('randnorm',[0])[0]:.3f}")


# --------------------------------------------------------------------------
# Battery-specific aggregations used by the results record.

def pair_level(rows, battery="h17a", split=None, need=2, arm_filter=None,
               key="in_top5", pair_set=None):
    """For H17a: a pair counts as a success at a setting when at least `need`
    of its scoreable functions were redirected by the same single swap."""
    import json as _j
    items = {it["item_id"]: it for it in
             _j.load(open(D + "battery_h17a.json"))}
    grp = defaultdict(lambda: [0, 0])
    for r in rows:
        it = items[r["item_id"]]
        if split and r["split"] != split:
            continue
        if pair_set and it["arm"] != pair_set:
            continue
        g = (r["item_id"], r["layers"], r["alpha"], r["posmode"], r["arm"],
             r["seed"])
        grp[g][0] += r[key]; grp[g][1] += 1
    acc = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for (iid, ls, a, m, arm, sd), (hit, tot) in grp.items():
        cell = (ls, a, m)
        acc[cell][arm][0] += int(hit >= need)
        acc[cell][arm][1] += 1
    return {c: {a: (n / d if d else 0, n, d) for a, (n, d) in arms.items()}
            for c, arms in acc.items()}


def subset_rates(rows, battery, cell, key, pred):
    """Success rate at one setting over the records satisfying `pred`."""
    acc = defaultdict(lambda: [0, 0])
    for r in rows:
        if (r["layers"], r["alpha"], r["posmode"]) != tuple(cell):
            continue
        if not pred(r):
            continue
        acc[r["arm"]][0] += r[key]; acc[r["arm"]][1] += 1
    return {a: [round(n / d, 4) if d else 0.0, n, d] for a, (n, d) in acc.items()}


def per_item(rows, cell, key, arm="lens"):
    """Which items succeeded at one setting, for picking examples."""
    out = {}
    for r in rows:
        if (r["layers"], r["alpha"], r["posmode"]) != tuple(cell) or r["arm"] != arm:
            continue
        out.setdefault((r["item_id"], r["func"]), []).append(r[key])
    return {k: (sum(v), len(v)) for k, v in sorted(out.items())}


def posmode_table(rows, key):
    """Best cell within each position mode, for the H17b contrast between
    swapping at the intermediate mention and swapping only at the answer."""
    acc = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in rows:
        acc[(r["layers"], r["alpha"], r["posmode"])][r["arm"]][0] += r[key]
        acc[(r["layers"], r["alpha"], r["posmode"])][r["arm"]][1] += 1
    best = {}
    for cell, arms in acc.items():
        m = cell[2]
        rate = arms["lens"][0] / max(arms["lens"][1], 1)
        if m not in best or rate > best[m][1]:
            best[m] = (cell, rate,
                       {a: round(n / max(d, 1), 4) for a, (n, d) in arms.items()})
    return best
