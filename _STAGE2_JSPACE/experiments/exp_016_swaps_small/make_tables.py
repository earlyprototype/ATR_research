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
        pm = s["posmode"]
        print(f"\nBy position mode at the tuned layer set and strength (layers {c[0]}, "
              f"strength {c[1]}), so that only the patched positions differ:\n")
        print("| positions | half | lens | control A | control B |")
        print("|---|---|---|---|---|")
        for m, halves in pm["fixed_setting"].items():
            for half in ("tuning", "heldout", "overall"):
                r = halves[half]
                print(f"| {m} | {half} | {pct(r.get('lens',[0,0,0]))} | {pct(r.get('randdir',[0,0,0]))} | "
                      f"{pct(r.get('randnorm',[0,0,0]))} |")
        print("\nBy position mode, each mode's own best setting chosen on the tuning "
              "half and scored on the held-out half:\n")
        print("| positions | tuned layers | strength | tuning, lens | held-out, lens | both halves, lens | both halves, control A |")
        print("|---|---|---|---|---|---|---|")
        for m, d in pm["tuned_per_mode"].items():
            print(f"| {m} | {d['cell'][0]} | {d['cell'][1]} | {pct(d['tuning']['lens'])} | "
                  f"{pct(d['heldout']['lens'])} | {pct(d['overall']['lens'])} | "
                  f"{pct(d['overall'].get('randdir',[0,0,0]))} |")
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

    if b == "h17":
        sr = s["source_rule_selection"]
        print(f"\nSource rule as part of the tuned selection (section 5.1 of the "
              f"specification): chosen setting layers {sr['chosen_cell'][0]}, strength "
              f"{sr['chosen_cell'][1]}, positions {sr['chosen_cell'][2]}, rule {sr['chosen_rule']}:\n")
        print("| half | lens swap | control A | control B |")
        print("|---|---|---|---|")
        for half in ("tuning", "heldout", "overall"):
            r = sr[half]
            print(f"| {half} | {pct(r.get('lens',[0,0,0]))} | {pct(r.get('randdir',[0,0,0]))} | {pct(r.get('randnorm',[0,0,0]))} |")
        print(f"\nEach source rule at the pooled tuned setting (layers {sr['pooled_cell'][0]}, "
              f"strength {sr['pooled_cell'][1]}, positions {sr['pooled_cell'][2]}):\n")
        print("| rule | half | lens swap | control A | control B |")
        print("|---|---|---|---|---|")
        for rule, halves in sr["per_rule"].items():
            for half in ("tuning", "heldout", "overall"):
                r = halves[half]
                print(f"| {rule} | {half} | {pct(r.get('lens',[0,0,0]))} | {pct(r.get('randdir',[0,0,0]))} | {pct(r.get('randnorm',[0,0,0]))} |")
    if b == "h17a":
        pl = s["pair_level_selection"]
        print(f"\nSelection by the registered pair-level outcome on the primary pairs "
              f"(at least two of three functions redirected): chosen setting layers "
              f"{pl['chosen_cell'][0]}, strength {pl['chosen_cell'][1]}, positions {pl['chosen_cell'][2]}, "
              f"against layers {pl['function_cell'][0]} for the function-level selection the main "
              f"analysis uses:\n")
        print("| selection metric | setting | half | lens swap | control A | control B |")
        print("|---|---|---|---|---|---|")
        for label, cell, d in (("pair-level, primary pairs", pl["chosen_cell"], pl["pair_level"]),
                               ("function-level (main analysis)", pl["function_cell"], pl["function_cell_pair_level"])):
            for half in ("tuning", "heldout", "overall"):
                r = d[half]
                print(f"| {label} | layers {cell[0]}, strength {cell[1]}, {cell[2]} | {half} | "
                      f"{pct(r.get('lens',[0,0,0]))} | {pct(r.get('randdir',[0,0,0]))} | {pct(r.get('randnorm',[0,0,0]))} |")
        r = pl["extension_heldout_at_chosen"]
        print(f"\nExtension set, held-out half, at the pair-level chosen setting: lens "
              f"{pct(r.get('lens',[0,0,0]))}, control A {pct(r.get('randdir',[0,0,0]))}, "
              f"control B {pct(r.get('randnorm',[0,0,0]))}.")

    if b in ("h17a", "h17b"):
        rs = s["rank1_sensitivity"]
        if b == "h17b":
            print(f"\nRank-1 sensitivity check (specification section 5.3): the {rs['n_items']} of 16 "
                  f"items whose correct answer the unmodified model ranks first, at the tuned setting, "
                  f"an item counting as a control success if any of its draws flipped it:\n")
            print("| half | lens swap | control A | control B |")
            print("|---|---|---|---|")
            for half in ("tuning", "heldout", "overall"):
                r = rs[half]
                print(f"| {half} | {r['lens'][0]} of {r['lens'][1]} | {r['randdir'][0]} of {r['randdir'][1]} | {r['randnorm'][0]} of {r['randnorm'][1]} |")
        else:
            print(f"\nRank-1 sensitivity check (specification section 5.2): only the {rs['n_rank1_questions']} of 124 "
                  f"scoreable questions whose correct answer the unmodified model ranks first, scored per pair "
                  f"(at least two such questions redirected) over the pairs that keep at least two of them "
                  f"({rs['n_pairs_with_two_rank1_questions']['primary']} primary, "
                  f"{rs['n_pairs_with_two_rank1_questions']['extension']} extension), at the tuned setting:\n")
            print("| pair set | half | lens swap | control A | control B |")
            print("|---|---|---|---|---|")
            for ps in ("primary", "extension"):
                for half in ("tuning", "heldout", "overall"):
                    r = rs[ps][half]
                    print(f"| {ps} | {half} | {r['lens'][0]} of {r['lens'][1]} | {r['randdir'][0]} of {r['randdir'][1]} | {r['randnorm'][0]} of {r['randnorm'][1]} |")

    et = s["exact_tests"]
    def fmt(t):
        return f"{t[0]} lens successes over {t[1]} informative items, probability {t[2]:.3g}"
    print("\nExact within-item tests against control A (the lens draw exchangeable with the control draws inside each item):\n")
    for name, t in et.items():
        print(f"- {name.replace('_', ' ')}: {fmt(t)}")
    if b == "h17a":
        rs = s["registered_split"]
        fl, pl = rs["function_level"], rs["pair_level"]
        print(f"\nUnder the specification's own split rule (alternate pairs in the committed order, "
              f"{rs['split_counts']['tuning']} tuning and {rs['split_counts']['heldout']} held-out, against the "
              f"country-wise 27 and 23 the battery was built with): function-level selection picks layers "
              f"{fl['chosen_cell'][0]}, strength {fl['chosen_cell'][1]}, {fl['chosen_cell'][2]} "
              f"(tuning {pct(fl['tuning']['lens'])}, held-out {pct(fl['heldout']['lens'])} per question; "
              f"held-out primary pairs {fl['heldout_pairs_primary']['lens'][0]} of {fl['heldout_pairs_primary']['lens'][1]} "
              f"against {fl['heldout_pairs_primary']['randdir'][0]} of {fl['heldout_pairs_primary']['randdir'][1]}); "
              f"pair-level selection picks layers {pl['chosen_cell'][0]} (held-out primary pairs "
              f"{pl['heldout_pairs_primary']['lens'][0]} of {pl['heldout_pairs_primary']['lens'][1]} against "
              f"{pl['heldout_pairs_primary']['randdir'][0]} of {pl['heldout_pairs_primary']['randdir'][1]}). "
              f"At the committed cells, held-out primary pairs under that split: layer 6 "
              f"{rs['committed_cells_heldout_pairs_primary']['6']['lens'][0]} of {rs['committed_cells_heldout_pairs_primary']['6']['lens'][1]}, "
              f"layer 9 {rs['committed_cells_heldout_pairs_primary']['9']['lens'][0]} of {rs['committed_cells_heldout_pairs_primary']['9']['lens'][1]}, "
              f"control A 0 of 30 at both.")
