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
        if tot < need:
            # A pair with fewer scoreable questions than the rule needs
            # cannot meet it and is left out of the denominator, never
            # counted as a failure (specification section 5.2).
            continue
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


def posmode_table(rows, key, tuned_cell):
    """The H17b contrast between position modes, two ways that both respect
    the tuning-then-held-out rule: (a) every mode at the battery's tuned
    layer set and strength, so only the positions differ; (b) each mode's
    own best cell chosen on the tuning half and scored on the held-out
    half. The first version of this function picked each mode's best cell
    over all rows, which compared different layers and used the held-out
    items in the choice; that reading is no longer produced."""
    modes = sorted({r["posmode"] for r in rows}, key=MODE_ORDER.index)
    out = {"fixed_setting": {}, "tuned_per_mode": {}}
    for m in modes:
        cell = (tuned_cell[0], tuned_cell[1], m)
        out["fixed_setting"][m] = {
            half: arms_of(cell_rates(rows, "h17b", split, key), cell)
            for half, split in (("tuning", "tuning"), ("heldout", "heldout"), ("overall", None))}
        flt = lambda r, m=m: r["posmode"] == m
        tune = cell_rates(rows, "h17b", "tuning", key, extra_filter=flt)
        best = choose(tune)
        out["tuned_per_mode"][m] = dict(
            cell=list(best),
            tuning=arms_of(tune, best),
            heldout=arms_of(cell_rates(rows, "h17b", "heldout", key, extra_filter=flt), best),
            overall=arms_of(cell_rates(rows, "h17b", None, key, extra_filter=flt), best))
    return out


def source_rule_selection(rows, items):
    """H17 with the source rule (lens or output) treated as part of the tuned
    selection, as section 5.1 of the specification says it is: the setting
    and the rule are chosen together on the tuning half and scored on the
    held-out half. Also reports each rule at the pooled tuned setting. The
    first version of this analysis pooled the two rules as 84 items."""
    rule_of = {it["item_id"]: it["source_rule"] for it in items}
    rules = sorted(set(rule_of.values()))
    joint = {}
    for rule in rules:
        flt = lambda r, rule=rule: rule_of[r["item_id"]] == rule
        for cell, arms in cell_rates(rows, "h17", "tuning", "in_top5", extra_filter=flt).items():
            joint[(cell, rule)] = arms
    def k(item):
        (cell, rule), arms = item
        lens = arms.get("lens", (0, 0, 0))[0]
        ctrl = arms.get("randdir", (0, 0, 0))[0]
        ls, alpha, mode = cell
        return (-lens, -(lens - ctrl), alpha, len(ls.split("-")), int(ls.split("-")[0]),
                MODE_ORDER.index(mode), rules.index(rule))
    (bc, br), _ = sorted(joint.items(), key=k)[0]
    out = dict(chosen_cell=list(bc), chosen_rule=br, per_rule={})
    for half, split in (("tuning", "tuning"), ("heldout", "heldout"), ("overall", None)):
        out[half] = arms_of(cell_rates(rows, "h17", split, "in_top5",
                                       extra_filter=lambda r: rule_of[r["item_id"]] == br), bc)
    pooled = choose(cell_rates(rows, "h17", "tuning", "in_top5"))
    out["pooled_cell"] = list(pooled)
    for rule in rules:
        flt = lambda r, rule=rule: rule_of[r["item_id"]] == rule
        out["per_rule"][rule] = {
            half: arms_of(cell_rates(rows, "h17", split, "in_top5", extra_filter=flt), pooled)
            for half, split in (("tuning", "tuning"), ("heldout", "heldout"), ("overall", None))}
    return out


def exact_within_item_test(items):
    """Exact one-sided test of the lens arm against control A on the same
    items. `items` is a list of (lens_success, [control_success per seed]).
    Under the null that the lens direction is no different from a random
    direction of the same lengths, the lens draw is exchangeable with the
    control draws within each item; conditioning on each item's total number
    of successes among its draws and relabelling which draw is the lens at
    random, the number of lens successes is a sum of independent Bernoulli
    variables with probability (successes / draws) per item. Returns
    (observed lens successes, informative items, probability of at least the
    observed count under the null)."""
    probs, t = [], 0
    for lens, ctrls in items:
        s = lens + sum(ctrls)
        k = 1 + len(ctrls)
        if s == 0 or s == k:
            continue
        probs.append(s / k)
        t += lens
    dist = {0: 1.0}
    for p in probs:
        nd = defaultdict(float)
        for c, v in dist.items():
            nd[c] += v * (1 - p)
            nd[c + 1] += v * p
        dist = nd
    return t, len(probs), sum(v for c, v in dist.items() if c >= t)


def item_outcomes(rows, cell, key, pred=lambda r: True):
    """Per-item (lens, [control A per seed]) outcomes at one setting."""
    per = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if (r["layers"], r["alpha"], r["posmode"]) == tuple(cell) and pred(r):
            per[r["item_id"]][r["arm"]][r["seed"]] = r[key]
    return [(d["lens"][-1], [d["randdir"][s] for s in sorted(d["randdir"])])
            for d in per.values() if "lens" in d and "randdir" in d]


def pair_outcomes(rows, cell, split, pair_set, need=2, rank1=False):
    """Per-pair (lens, [control A per seed]) outcomes for H17a at one setting,
    a pair succeeding when at least `need` of its scoreable questions were
    redirected by the same draw."""
    items = {it["item_id"]: it for it in json.load(open(D + "battery_h17a.json"))}
    grp = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for r in rows:
        if (r["layers"], r["alpha"], r["posmode"]) != tuple(cell):
            continue
        it = items[r["item_id"]]
        if it["arm"] != pair_set or (split and it["split"] != split):
            continue
        if rank1 and next(f for f in it["funcs"] if f["func"] == r["func"])["clean_rank"] != 1:
            continue
        grp[r["item_id"]][(r["arm"], r["seed"])][r["func"]] += r["in_top5"]
    out = []
    for arms in grp.values():
        if len(next(iter(arms.values()))) < need:
            continue
        lens = int(sum(1 for v in arms[("lens", -1)].values() if v) >= need)
        ctrls = [int(sum(1 for v in d.values() if v) >= need)
                 for (a, s), d in sorted(arms.items()) if a == "randdir"]
        out.append((lens, ctrls))
    return out


def registered_split_reading(rows, key="in_top5"):
    """H17a under the specification's own split rule, alternate pairs in the
    committed order (25 and 25), instead of the country-wise assignment the
    battery was built with (27 and 23). Reported as a sensitivity reading;
    the run was tuned on the split it was built with."""
    order = [it["item_id"] for it in json.load(open(D + "battery_h17a.json"))]
    alt = {iid: ("tuning" if i % 2 == 0 else "heldout") for i, iid in enumerate(order)}
    def rates(split):
        acc = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        for r in rows:
            if alt[r["item_id"]] != split:
                continue
            a = acc[(r["layers"], r["alpha"], r["posmode"])][r["arm"]]
            a[0] += r[key]; a[1] += 1
        return {c: {a: (n / d, n, d) for a, (n, d) in arms.items()} for c, arms in acc.items()}
    tune, held = rates("tuning"), rates("heldout")
    best = choose(tune)
    items = {it["item_id"]: it for it in json.load(open(D + "battery_h17a.json"))}
    def pairs(split, cell):
        grp = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        for r in rows:
            if (r["layers"], r["alpha"], r["posmode"]) != tuple(cell) or alt[r["item_id"]] != split \
                    or items[r["item_id"]]["arm"] != "primary":
                continue
            grp[r["item_id"]][(r["arm"], r["seed"])][r["func"]] += r["in_top5"]
        acc = defaultdict(lambda: [0, 0])
        for arms in grp.values():
            for (a, s), d in arms.items():
                if len(d) < 2:
                    continue
                acc[a][0] += int(sum(1 for v in d.values() if v) >= 2); acc[a][1] += 1
        return {a: [n, d] for a, (n, d) in acc.items()}
    tune_pl = {}
    for c in tune:
        pl = pairs("tuning", c)
        if pl:
            tune_pl[c] = {a: (n / d if d else 0.0, n, d) for a, (n, d) in pl.items()}
    best_pl = choose(tune_pl)
    return dict(split_counts=dict(tuning=sum(v == "tuning" for v in alt.values()),
                                  heldout=sum(v == "heldout" for v in alt.values())),
                function_level=dict(chosen_cell=list(best), tuning=arms_of(tune, best),
                                    heldout=arms_of(held, best),
                                    heldout_pairs_primary=pairs("heldout", best)),
                pair_level=dict(chosen_cell=list(best_pl),
                                tuning_pairs_primary=pairs("tuning", best_pl),
                                heldout_pairs_primary=pairs("heldout", best_pl)),
                committed_cells_heldout_pairs_primary={
                    "6": pairs("heldout", ("6", 2.0, "all")), "9": pairs("heldout", ("9", 2.0, "all"))})


def exact_tests(rows, battery, cell):
    """The exact within-item tests the record reports, per battery."""
    out = {}
    if battery == "h17":
        items = {it["item_id"]: it for it in json.load(open(D + "battery_h17.json"))}
        held = lambda r: items[r["item_id"]]["split"] == "heldout"
        out["pooled_heldout"] = exact_within_item_test(item_outcomes(rows, cell, "in_top5", held))
        for rule in ("lens", "output"):
            out[f"rule_{rule}_heldout"] = exact_within_item_test(item_outcomes(
                rows, cell, "in_top5", lambda r, rule=rule: held(r) and items[r["item_id"]]["source_rule"] == rule))
    elif battery == "h17a":
        out["pairs_primary_heldout"] = exact_within_item_test(pair_outcomes(rows, cell, "heldout", "primary"))
        out["pairs_primary_both"] = exact_within_item_test(pair_outcomes(rows, cell, None, "primary"))
        out["pairs_primary_heldout_rank1"] = exact_within_item_test(pair_outcomes(rows, cell, "heldout", "primary", rank1=True))
        out["pairs_primary_heldout_layer9"] = exact_within_item_test(pair_outcomes(rows, ("9", 2.0, "all"), "heldout", "primary"))
    else:
        items = {it["item_id"]: it for it in json.load(open(D + "battery_h17b.json"))}
        out["items_all"] = exact_within_item_test(item_outcomes(rows, cell, "is_top1"))
        out["items_heldout"] = exact_within_item_test(item_outcomes(
            rows, cell, "is_top1", lambda r: items[r["item_id"]]["split"] == "heldout"))
        out["items_rank1"] = exact_within_item_test(item_outcomes(
            rows, cell, "is_top1", lambda r: items[r["item_id"]]["clean_answer_rank"] == 1))
    return out


def rank1_sensitivity(rows, battery, cell):
    """The specification's promised rank-1 sensitivity check: the same
    scoring at the tuned setting, restricted to the items (H17b) or the
    questions (H17a) whose correct answer the unmodified model ranks first,
    which is the population the register's words "answers correctly" name
    if they are read strictly. Returns counts per arm, and for H17a the
    pair-level outcome over pairs that keep at least two such questions."""
    out = {}
    if battery == "h17b":
        items = {it["item_id"]: it for it in json.load(open(D + "battery_h17b.json"))}
        keep = {i for i, it in items.items() if it["clean_answer_rank"] == 1}
        per = defaultdict(lambda: defaultdict(list))
        for r in rows:
            if (r["layers"], r["alpha"], r["posmode"]) == tuple(cell) and r["item_id"] in keep:
                per[r["item_id"]][r["arm"]].append(r["is_top1"])
        for half, pred in (("tuning", lambda i: items[i]["split"] == "tuning"),
                           ("heldout", lambda i: items[i]["split"] == "heldout"),
                           ("overall", lambda i: True)):
            ids = [i for i in per if pred(i)]
            out[half] = {a: [sum(any(per[i][a]) for i in ids), len(ids)] for a in ("lens", "randdir", "randnorm")}
        out["n_items"] = len(keep)
        return out
    items = {it["item_id"]: it for it in json.load(open(D + "battery_h17a.json"))}
    rank1 = {(it["item_id"], f["func"]) for it in items.values() for f in it["funcs"]
             if f["scoreable"] and f["clean_rank"] == 1}
    grp = defaultdict(lambda: [0, 0])
    for r in rows:
        if (r["layers"], r["alpha"], r["posmode"]) != tuple(cell) or (r["item_id"], r["func"]) not in rank1:
            continue
        g = (r["item_id"], r["arm"], r["seed"]); grp[g][0] += r["in_top5"]; grp[g][1] += 1
    for pair_set in ("primary", "extension"):
        out[pair_set] = {}
        for half, split in (("tuning", "tuning"), ("heldout", "heldout"), ("overall", None)):
            acc = defaultdict(lambda: [0, 0])
            for (iid, arm, sd), (hit, tot) in grp.items():
                it = items[iid]
                if it["arm"] != pair_set or (split and it["split"] != split) or tot < 2:
                    continue
                acc[arm][0] += int(hit >= 2); acc[arm][1] += 1
            out[pair_set][half] = {a: list(acc.get(a, [0, 0])) for a in ("lens", "randdir", "randnorm")}
    out["n_rank1_questions"] = len(rank1)
    out["n_pairs_with_two_rank1_questions"] = {ps: sum(1 for it in items.values() if it["arm"] == ps and
        sum(1 for f in it["funcs"] if f["scoreable"] and f["clean_rank"] == 1) >= 2) for ps in ("primary", "extension")}
    return out


def pair_level_selection(rows, function_cell):
    """H17a with the registered pair-level outcome (at least two of three
    functions redirected, primary pairs only) used as the selection metric
    on the tuning half, beside the function-level metric of section 5.2 that
    the main analysis uses. Both routes are reported because the
    specification names the function-level success for selection and the
    pair-level rule for the verdict."""
    tune = pair_level(rows, split="tuning", need=2, pair_set="primary")
    best = choose(tune)
    def at(cell):
        return {half: arms_of(pair_level(rows, split=split, need=2, pair_set="primary"), cell)
                for half, split in (("tuning", "tuning"), ("heldout", "heldout"), ("overall", None))}
    return dict(chosen_cell=list(best), pair_level=at(best),
                function_cell=list(function_cell), function_cell_pair_level=at(tuple(function_cell)),
                extension_heldout_at_chosen=arms_of(
                    pair_level(rows, split="heldout", need=2, pair_set="extension"), best))


if __name__ == "__main__":
    for b in sys.argv[1:]:
        out, rows = report(b)
        if b == "h17":
            out["source_rule_selection"] = source_rule_selection(
                rows, json.load(open(D + "battery_h17.json")))
        if b == "h17a":
            out["pair_level_selection"] = pair_level_selection(rows, out["chosen_cell"])
        if b == "h17b":
            out["posmode"] = posmode_table(rows, SCORE[b], out["chosen_cell"])
        if b in ("h17a", "h17b"):
            out["rank1_sensitivity"] = rank1_sensitivity(rows, b, out["chosen_cell"])
        if b == "h17a":
            out["registered_split"] = registered_split_reading(rows)
        out["exact_tests"] = exact_tests(rows, b, out["chosen_cell"])
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
