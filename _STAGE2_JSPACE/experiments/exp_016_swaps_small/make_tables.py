"""Print the markdown tables used in RESULTS_EXP016.md, straight from the
record files, so that no number in the results record is typed by hand."""
import json, sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyse
from collections import defaultdict

D = os.path.dirname(os.path.abspath(__file__)) + "/"
NAME = {"lens": "lens swap", "randdir": "control A (random directions)",
        "randnorm": "control B (size matched)"}


def pct(t):
    return f"{t[0]*100:.0f} percent ({t[1]} of {t[2]})"


BATTERIES = sys.argv[1:] or ["h17", "h17a", "h17b"]
for b in BATTERIES:
    s = json.load(open(D + f"output/summary_{b}.json"))
    rows = analyse.load(b)
    key = analyse.SCORE[b]
    c = s["chosen_cell"]
    print(f"\n### {b.upper()}: tuned setting = layers {c[0]}, strength {c[1]}, "
          f"positions {c[2]}   ({s['n_records']} records)")
    print("\n| arm | tuning half | held-out half | both halves |")
    print("|---|---|---|---|")
    for arm in ("lens", "randdir", "randnorm"):
        print(f"| {NAME[arm]} | {pct(s['tuning'].get(arm,[0,0,0]))} | "
              f"{pct(s['heldout'].get(arm,[0,0,0]))} | "
              f"{pct(s['overall'].get(arm,[0,0,0]))} |")
    print("\nBest ten settings over all items:\n")
    print("| layers | strength | positions | lens | control A | control B |")
    print("|---|---|---|---|---|---|")
    for g in s["top_cells_overall"][:10]:
        print(f"| {g['layers']} | {g['alpha']} | {g['posmode']} | "
              f"{pct(g['overall']['lens'])} | {pct(g['overall'].get('randdir',[0,0,0]))} | "
              f"{pct(g['overall'].get('randnorm',[0,0,0]))} |")
    # single-layer curve at the tuned strength and positions
    print("\nSingle layers at the tuned strength and positions:\n")
    print("| layer | lens | control A | control B |")
    print("|---|---|---|---|")
    for g in sorted([g for g in s["grid"] if "-" not in g["layers"]
                     and g["alpha"] == c[1] and g["posmode"] == c[2]],
                    key=lambda g: int(g["layers"])):
        print(f"| {g['layers']} | {pct(g['overall']['lens'])} | "
              f"{pct(g['overall'].get('randdir',[0,0,0]))} | "
              f"{pct(g['overall'].get('randnorm',[0,0,0]))} |")
    if b == "h17":
        print("\nBy source rule, at the tuned setting:\n")
        print("| source rule | lens | control A | control B |")
        print("|---|---|---|---|")
        for rule in ("lens", "output"):
            r = analyse.subset_rates(rows, b, c, key,
                                     lambda x, R=rule: x["item_id"].endswith(R))
            print(f"| {rule} | {pct(r.get('lens',[0,0,0]))} | "
                  f"{pct(r.get('randdir',[0,0,0]))} | {pct(r.get('randnorm',[0,0,0]))} |")
        print("\nStricter score at the tuned setting, target becomes the "
              "single most likely next word:\n")
        r = analyse.subset_rates(rows, b, c, "is_top1", lambda x: True)
        for arm in ("lens", "randdir", "randnorm"):
            print(f"- {NAME[arm]}: {pct(r.get(arm,[0,0,0]))}")
        r = analyse.subset_rates(rows, b, c, "beats_bad", lambda x: True)
        print("\nTarget outranks the source word it replaced:\n")
        for arm in ("lens", "randdir", "randnorm"):
            print(f"- {NAME[arm]}: {pct(r.get(arm,[0,0,0]))}")
    if b == "h17b":
        print("\nBy position mode, best setting within each mode:\n")
        print("| positions | best layers | strength | lens | control A | control B |")
        print("|---|---|---|---|---|---|")
        best = analyse.posmode_table(rows, key)
        for m, (cell, rate, arms) in sorted(best.items()):
            print(f"| {m} | {cell[0]} | {cell[1]} | {arms['lens']*100:.0f} percent | "
                  f"{arms.get('randdir',0)*100:.0f} percent | "
                  f"{arms.get('randnorm',0)*100:.0f} percent |")
        print("\nLooser score at the tuned setting, alternative answer in the "
              "top five:\n")
        r = analyse.subset_rates(rows, b, c, "in_top5", lambda x: True)
        for arm in ("lens", "randdir", "randnorm"):
            print(f"- {NAME[arm]}: {pct(r.get(arm,[0,0,0]))}")
    if b == "h17a":
        items = {i["item_id"]: i for i in json.load(open(D + "battery_h17a.json"))}
        print("\nBy question, at the tuned setting (primary pairs only):\n")
        print("| question | lens | control A | control B |")
        print("|---|---|---|---|")
        for f in ("capital", "language", "continent"):
            r = analyse.subset_rates(
                rows, b, c, key,
                lambda x, F=f: x["func"] == F and items[x["item_id"]]["arm"] == "primary")
            print(f"| {f} | {pct(r.get('lens',[0,0,0]))} | "
                  f"{pct(r.get('randdir',[0,0,0]))} | {pct(r.get('randnorm',[0,0,0]))} |")
        print("\nBy question, extension set (five continents):\n")
        print("| question | lens | control A | control B |")
        print("|---|---|---|---|")
        for f in ("capital", "language", "continent"):
            r = analyse.subset_rates(
                rows, b, c, key,
                lambda x, F=f: x["func"] == F and items[x["item_id"]]["arm"] == "extension")
            print(f"| {f} | {pct(r.get('lens',[0,0,0]))} | "
                  f"{pct(r.get('randdir',[0,0,0]))} | {pct(r.get('randnorm',[0,0,0]))} |")
        for setname in ("primary", "extension"):
            for need in (2, 3):
                p = analyse.pair_level(rows, need=need, pair_set=setname)
                a = p.get(tuple(c), {})
                if not a:
                    continue
                print(f"\nPairs where at least {need} scoreable questions "
                      f"redirected, {setname} set, tuned setting:")
                for arm in ("lens", "randdir", "randnorm"):
                    if arm in a:
                        print(f"- {NAME[arm]}: {pct(a[arm])}")
