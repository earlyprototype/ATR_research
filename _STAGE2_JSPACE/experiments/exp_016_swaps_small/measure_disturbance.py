"""Measure the disturbance sizes of the three arms at each battery's chosen
setting, after the fact.

The first run of run_swaps.py left the patch_norm column empty, so the sizes
the specification asked to be recorded (section 3) were not captured. This
script recomputes them for the chosen setting of each battery only, from the
committed batteries and summaries: for every unit it runs the lens arm, the
registered control A (random directions of the lens vectors' lengths) and the
size-matched control B, and records the Euclidean norm of the total change
each arm makes to the residual stream over the patched positions and layers,
per layer as well as in total, beside the norm of the untouched residual
stream over the same positions at every patched layer.

The denominator was wrong until 2026-09-06 and is corrected here. A change
accumulated across three patched layers was being divided by the untouched
residual stream at the first patched layer alone, which is a third of what the
change is spread over, so every "of the stream" ratio at the three-layer H17
setting came out too large. The total ratio now divides by the norm of the
untouched residual streams of all patched layers taken together, which is the
same quantity the numerator is taken over, and a per-layer ratio is reported
beside it: the change at one layer divided by the untouched stream at that
same layer.

The random draws differ from those of the committed records, whose seeds came
from Python's per-process hash() and cannot be regenerated; the sizes are
therefore typical values for the arms, not the exact ones behind the verdict
tables. Nothing here carries verdict weight.

Usage: python3 measure_disturbance.py [--out PATH]   (all three batteries)
Writes a dated file, output/disturbance_sizes_<measurement date>.json unless
--out names another path, and appends a line to output/exp_016_run.log. The
dated default exists so that a re-run never overwrites a committed
measurement; `output/disturbance_sizes.json` is the first measurement, of
2026-09-05, whose ratios to the stream are superseded by the correction above.
"""
from __future__ import annotations
import datetime, json, math, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_exp016
from lib_exp016 import load_model, load_lens, lens_vectors, positions
from swap_engine import SwapPlan, random_pair, run_plan
from run_swaps import units_for, control_seed, SEEDS_BY_BATTERY, LAYERS

D = os.path.dirname(os.path.abspath(__file__)) + "/"


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def summarise(vals):
    return dict(n=len(vals), median=round(median(vals), 3),
                min=round(min(vals), 3), max=round(max(vals), 3))


@torch.no_grad()
def main(path):
    t0 = time.time()
    model = load_model()
    lens = load_lens()
    out = dict(measured_at=datetime.datetime.utcnow().isoformat() + "Z",
               lens_sha256=lib_exp016.LENS_SHA256_MEASURED,
               model_revision=lib_exp016.MODEL_REVISION,
               model_param_sha256=lib_exp016.MODEL_PARAM_SHA256_MEASURED,
               note="sizes are Euclidean norms of the total residual-stream change "
                    "over patched positions and layers; stream_norm is the norm of "
                    "the untouched residual streams over the same positions at every "
                    "patched layer taken together, which is what the total change is "
                    "spread over; ratio_to_stream is total change divided by that; "
                    "ratio_per_layer divides each layer's change by the untouched "
                    "stream at that same layer",
               batteries={})
    for battery in ("h17", "h17a", "h17b"):
        items = json.load(open(D + f"battery_{battery}.json"))
        cell = json.load(open(D + f"output/summary_{battery}.json"))["chosen_cell"]
        layer_set = tuple(int(x) for x in cell[0].split("-"))
        alpha, mode = float(cell[1]), cell[2]
        seeds = SEEDS_BY_BATTERY[battery]
        units = units_for(battery, items)
        per_arm = {"lens": [], "randdir": [], "randnorm": []}
        per_layer = {a: {l: [] for l in layer_set} for a in per_arm}
        ratios = {a: [] for a in per_arm}
        ratios_layer = {a: {l: [] for l in layer_set} for a in per_arm}
        stream_totals, stream_layers = [], {l: [] for l in layer_set}
        for u in units:
            toks = model.to_tokens(u["prompt"])
            T = toks.shape[1]
            pos = positions(mode, T, u.get("first_mention_pos", 1))
            V = {l: lens_vectors(lens, model, l, [u["source_tok"], u["target_tok"]])
                 for l in layer_set}
            plan = SwapPlan(T, LAYERS)
            rows = [("lens", -1)]
            plan.add(V, V, alpha, pos, False)
            for s in seeds:
                R = {l: random_pair(V[l][:, 0], V[l][:, 1],
                                    control_seed(u["item_id"], l, s))
                     for l in layer_set}
                plan.add(R, V, alpha, pos, False); rows.append(("randdir", s))
                plan.add(R, V, alpha, pos, True); rows.append(("randnorm", s))
            plan.build()
            run_plan(model, toks, plan)
            names = [f"blocks.{l}.hook_resid_post" for l in layer_set]
            _, cache = model.run_with_cache(toks, names_filter=names)
            # The untouched residual stream over the patched positions at each
            # patched layer, and all of them taken together. The total change
            # is summed over exactly these positions and layers, so the total
            # taken together is the like-for-like denominator for it.
            per_l = {l: float(cache[f"blocks.{l}.hook_resid_post"][0, pos].float().norm())
                     for l in layer_set}
            stream = math.sqrt(sum(v * v for v in per_l.values()))
            stream_totals.append(stream)
            for l in layer_set:
                stream_layers[l].append(per_l[l])
            tot = plan.change_sq.sqrt().tolist()
            for bi, (arm, s) in enumerate(rows):
                per_arm[arm].append(tot[bi])
                ratios[arm].append(tot[bi] / stream)
                for l in layer_set:
                    ch = float(plan.change_sq_layer[l][bi].sqrt())
                    per_layer[arm][l].append(ch)
                    ratios_layer[arm][l].append(ch / per_l[l])
        out["batteries"][battery] = dict(
            chosen_cell=cell, n_units=len(units), seeds=seeds,
            total={a: summarise(v) for a, v in per_arm.items()},
            ratio_to_stream={a: summarise(v) for a, v in ratios.items()},
            per_layer={a: {str(l): summarise(v) for l, v in d.items()} for a, d in per_layer.items()},
            ratio_per_layer={a: {str(l): summarise(v) for l, v in d.items()}
                             for a, d in ratios_layer.items()},
            stream_norm=dict(all_patched_layers_together=summarise(stream_totals),
                             per_layer={str(l): summarise(v) for l, v in stream_layers.items()}))
        print(battery, cell, "units", len(units),
              {a: out["batteries"][battery]["total"][a]["median"] for a in per_arm}, flush=True)
    out["wall_seconds"] = round(time.time() - t0, 1)
    json.dump(out, open(path, "w"), indent=1)
    rel = os.path.relpath(path, D)
    with open(D + "output/exp_016_run.log", "a") as fh:
        fh.write(f"\n[{out['measured_at']}] measure_disturbance.py: disturbance sizes at the "
                 f"chosen settings recomputed after the fact for all three batteries "
                 f"(control draws from the fixed checksum seeds, no verdict weight), now "
                 f"dividing the total change by the untouched residual streams of every "
                 f"patched layer taken together rather than by the first patched layer "
                 f"alone, with per-layer ratios beside it; wrote {rel} "
                 f"in {out['wall_seconds']} s; no committed file was overwritten\n")
    print("done in", out["wall_seconds"], "s; wrote", rel)


def default_path():
    """A dated output path, so that a re-run adds a measurement rather than
    overwriting one that a record already quotes."""
    day = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    return D + f"output/disturbance_sizes_{day}.json"


if __name__ == "__main__":
    args = sys.argv[1:]
    out_path = default_path()
    if "--out" in args:
        out_path = args[args.index("--out") + 1]
        if not os.path.isabs(out_path):
            out_path = D + out_path
    main(out_path)
