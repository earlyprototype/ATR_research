"""The global-draw test for EXP_016: one random assignment of directions to
concepts, scored over a whole held-out set as a single draw.

Why this exists. Every cluster-level test in this record multiplies a factor
per cluster, which needs the clusters to be independent of one another. That
holds when the clusters are the connected components of the units sharing a
lens direction, but only while the control's directions belong to the same
concepts the components were built from. The mirrored control chooses its own
target concept for every draw, so its draws cross the component boundaries:
at draw 0 it picks ' grape' as the target for frames in more than one
component, and the product over components is then not a valid probability.

What this measures instead. One draw is one random direction for every
concept the battery can reach, seeded by the concept's token, the layer and
the draw index, with the battery's own target-selection rule applied to those
directions where the battery selects its target that way. A draw is scored by
running every unit of the held-out set under it and counting the successes, so
one draw yields one number, comparable with the lens arm's own number for the
same set. Under the null that a lens direction is nothing but a norm-matched
random direction, the lens arm's assignment is exchangeable with the D random
assignments, so the probability of a total at least as large as the lens
arm's is (1 + the number of draws reaching it) divided by (D + 1). Nothing is
assumed about independence between units, and nothing is multiplied.

The sets it scores are read from the summary files so that this run and the
analysis cannot drift apart: the held-out frames of the source rule chosen
under the committed alternating split, the held-out frames of the rule chosen
under the component-respecting split at that split's own setting, and H17b's
held-out items at its chosen setting.

Usage: python3 global_draw_control.py [--draws N] [set names]
Writes output/global_draw_records.csv (one row per set, draw and unit) and
output/global_draw_provenance.json, and appends a line to
output/exp_016_run.log. It never writes the committed record files.
"""
from __future__ import annotations
import csv, datetime, json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_exp016
from lib_exp016 import (load_model, load_lens, lens_vectors, positions, rank_of,
                        JLENS_COMMIT)
from swap_engine import SwapPlan, run_plan
from run_swaps import units_for, LAYERS
from cluster_control import (selection_context, mirrored_pick, token_direction,
                             SELECT_LAYER)
import analyse

D = os.path.dirname(os.path.abspath(__file__)) + "/"
CHUNK = 64          # draws per forward pass, to keep peak memory small


def sets_to_run():
    """The held-out sets this test scores, with the setting each is scored at,
    read from the committed summary files."""
    h17 = json.load(open(D + "output/summary_h17.json"))
    h17b = json.load(open(D + "output/summary_h17b.json"))
    items = {it["item_id"]: it for it in json.load(open(D + "battery_h17.json"))}
    itemsb = {it["item_id"]: it for it in json.load(open(D + "battery_h17b.json"))}
    out = {}
    sr = h17["source_rule_selection"]
    out["h17_committed_heldout"] = dict(
        battery="h17", cell=tuple(sr["chosen_cell"]), mirrored=True,
        split="committed alternating split", rule=sr["chosen_rule"],
        item_ids=sorted(i for i, it in items.items()
                        if it["split"] == "heldout"
                        and it["source_rule"] == sr["chosen_rule"]))
    cs = h17.get("component_split")
    if cs:
        toks = lambda i: (items[i]["source_tok"], items[i]["target_tok"])
        half = analyse.component_split(items, toks)
        out["h17_component_split_heldout"] = dict(
            battery="h17", cell=tuple(cs["chosen_cell"]), mirrored=True,
            split="component-respecting split", rule=cs["chosen_rule"],
            item_ids=sorted(i for i in items
                            if half[i] == "heldout"
                            and items[i]["source_rule"] == cs["chosen_rule"]))
    out["h17b_committed_heldout"] = dict(
        battery="h17b", cell=tuple(h17b["chosen_cell"]), mirrored=False,
        split="committed alternating split", rule="",
        item_ids=sorted(i for i, it in itemsb.items() if it["split"] == "heldout"))
    return out


@torch.no_grad()
def main(draws, names):
    t0 = time.time()
    model = load_model()
    lens = load_lens()
    pilot = {r["frame"]: r for r in
             json.load(open(D + "output/pilot_clean.json"))["battery1"]}
    ctx_cache = {}
    rows_out, lens_totals, mismatch = [], {}, []
    for name, spec in sets_to_run().items():
        if names and name not in names:
            continue
        battery = spec["battery"]
        allitems = {it["item_id"]: it
                    for it in json.load(open(D + f"battery_{battery}.json"))}
        chosen = [allitems[i] for i in spec["item_ids"]]
        units = units_for(battery, chosen)
        ls = tuple(int(x) for x in spec["cell"][0].split("-"))
        alpha, mode = float(spec["cell"][1]), spec["cell"][2]
        key = "in_top5" if battery == "h17" else "is_top1"
        lens_total = 0
        for u in units:
            toks = model.to_tokens(u["prompt"])
            T = toks.shape[1]
            pos = positions(mode, T, u.get("first_mention_pos", 1))
            V = {l: lens_vectors(lens, model, l, [u["source_tok"], u["target_tok"]])
                 for l in ls}
            # the lens arm once, to check this run against the committed record
            plan = SwapPlan(T, LAYERS)
            plan.add(V, V, alpha, pos, False)
            plan.build()
            lp = run_plan(model, toks, plan)
            got = int(u["good_tok"] in torch.topk(lp[0], 5).indices) if key == "in_top5" \
                else int(int(lp[0].argmax()) == u["good_tok"])
            lens_total += got
            if spec["mirrored"] and u["frame"] not in ctx_cache:
                ctx_cache[u["frame"]] = selection_context(model, lens, pilot[u["frame"]])
            for c0 in range(0, draws, CHUNK):
                block = range(c0, min(c0 + CHUNK, draws))
                plan = SwapPlan(T, LAYERS)
                meta = []
                for d in block:
                    if spec["mirrored"]:
                        ctx = ctx_cache[u["frame"]]
                        pick = mirrored_pick(ctx, u, d)
                        if pick is None:
                            continue
                        sn, tn = pick
                        s_tok, t_tok = ctx["toks"][sn], ctx["toks"][tn]
                    else:
                        sn, tn = u.get("source", ""), u.get("target", "")
                        s_tok, t_tok = u["source_tok"], u["target_tok"]
                    Vsel = {l: lens_vectors(lens, model, l, [s_tok, t_tok]) for l in ls}
                    Vu = {l: torch.stack(
                        [token_direction(Vsel[l][:, 0], s_tok, l, d),
                         token_direction(Vsel[l][:, 1], t_tok, l, d)], dim=1)
                        for l in ls}
                    plan.add(Vu, V, alpha, pos, False)
                    meta.append((d, s_tok, t_tok, sn, tn))
                plan.build()
                lp = run_plan(model, toks, plan)
                top5 = torch.topk(lp, 5, dim=-1).indices
                am = lp.argmax(dim=-1)
                for bi, (d, s_tok, t_tok, sn, tn) in enumerate(meta):
                    good = t_tok if battery == "h17" else u["good_tok"]
                    hit = (int(good in top5[bi]) if key == "in_top5"
                           else int(int(am[bi]) == good))
                    rows_out.append(dict(
                        set_name=name, battery=battery, draw=d,
                        item_id=u["item_id"], func=u.get("func", ""),
                        layers=spec["cell"][0], alpha=alpha, posmode=mode,
                        sel_source=sn, sel_target=tn, success=hit,
                        good_rank=rank_of(lp[bi], good)))
            print(f"[{name}] {u['item_id']} done, elapsed "
                  f"{(time.time()-t0)/60:.1f} min", flush=True)
        lens_totals[name] = [lens_total, len(units)]
        # the lens arm here must reproduce the committed record file
        want = 0
        with open(D + f"output/records_{battery}.csv") as fh:
            for r in csv.DictReader(fh):
                if (r["arm"] == "lens" and r["item_id"] in set(spec["item_ids"])
                        and (r["layers"], float(r["alpha"]), r["posmode"])
                        == (spec["cell"][0], alpha, mode)):
                    want += int(r[key])
        if want != lens_total:
            mismatch.append((name, want, lens_total))
        print(f"[{name}] lens total {lens_total} of {len(units)} "
              f"(committed record says {want})", flush=True)
    if mismatch:
        raise RuntimeError(f"the re-run lens arm disagrees with the committed "
                           f"records: {mismatch}")
    cols = ["set_name", "battery", "draw", "item_id", "func", "layers", "alpha",
            "posmode", "sel_source", "sel_target", "success", "good_rank"]
    final = D + "output/global_draw_records.csv"
    partial = final + ".partial"
    with open(partial, "w", newline="") as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    os.replace(partial, final)
    prov = dict(
        measured_at=datetime.datetime.utcnow().isoformat() + "Z",
        draws=draws, sets={k: dict(v, item_ids=len(v["item_ids"]))
                           for k, v in sets_to_run().items()
                           if not names or k in names},
        lens_totals=lens_totals, n_rows=len(rows_out),
        lens_arm_reproduces_committed_records=True,
        lens_sha256=lib_exp016.LENS_SHA256_MEASURED, jlens_commit=JLENS_COMMIT,
        model_revision=lib_exp016.MODEL_REVISION,
        model_param_sha256=lib_exp016.MODEL_PARAM_SHA256_MEASURED,
        torch=torch.__version__, wall_seconds=round(time.time() - t0, 1))
    json.dump(prov, open(D + "output/global_draw_provenance.json", "w"), indent=1)
    with open(D + "output/exp_016_run.log", "a") as fh:
        fh.write(f"\n[{prov['measured_at']}] global_draw_control.py: exit 0, "
                 f"{prov['wall_seconds']} s, {len(rows_out)} scored conditions over "
                 f"{draws} draws. One draw is one random direction per concept, seeded "
                 f"by the concept's token, the layer and the draw index, with the "
                 f"battery's own target selection applied where the battery selects its "
                 f"target by a lens reading; a whole held-out set is scored under one "
                 f"draw and yields one number. Sets: "
                 f"{', '.join(f'{k} lens {v[0]} of {v[1]}' for k, v in lens_totals.items())}. "
                 f"The lens arm was re-run and reproduces the committed record files on "
                 f"every set. Wrote output/global_draw_records.csv and "
                 f"output/global_draw_provenance.json; no committed file was "
                 f"overwritten.\n")
    print(f"done: {len(rows_out)} rows in {prov['wall_seconds']} s")


if __name__ == "__main__":
    args = sys.argv[1:]
    n = 199
    if "--draws" in args:
        i = args.index("--draws")
        n = int(args[i + 1]); del args[i:i + 2]
    main(n, args)
