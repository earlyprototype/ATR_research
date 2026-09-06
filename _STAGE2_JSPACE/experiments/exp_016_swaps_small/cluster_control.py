"""Run the cluster-matched control arms that the cluster-level exact tests
need, for EXP_016.

Why this exists. The cluster-level test in `analyse.py` groups the scored
units that share one source lens direction and treats the lens arm as one
draw for the whole group, exchangeable with each control seed's draw for the
same group. That exchange is only fair if the control's randomness is shared
inside a group the way the lens arm's is. The registered control, called
control A in the results record, draws both of its random directions from one
seed per item (`run_swaps.control_seed` hashes the item identifier), so two
items of one group get independent draws, while the lens arm gives them the
identical source direction. The group totals of the two arms are therefore not
exchangeable, and a cluster-level probability computed against control A is
not valid as stated.

What this script measures. Two further control arms, run at the settings the
cluster tests use, on the same prompts, scored the same way:

  randdir_shared_source  one random direction standing in for the source
      concept, drawn once per group of units that share a source lens
      direction and per layer and per seed, with an independent random
      direction standing in for the target concept drawn per item. Both are
      rescaled to the lengths of the lens directions they replace, as control
      A's are. This matches the lens arm's sharing for the groupings whose
      members share a source concept.
  randdir_shared_pair    both random directions drawn once per group of units
      that share both the source and the target concept, so that every member
      of such a group receives the identical random swap, as it receives the
      identical lens swap. Run for H17 only, where that grouping is reported.
      For H17a a group of that kind is a single country pair, whose questions
      already share one control A draw, so control A is already like for like
      there.

The lens arm is re-run beside them at the same settings. It is deterministic,
so its outcomes must reproduce the committed record files exactly; the script
checks that and refuses to write if they do not, which is what makes it sound
to test the committed lens rows against these new control rows.

H17b needs none of this: each of its 16 items has its own source concept, so
every group holds one item and the cluster-level test is the within-item test,
whose draws are exchangeable already.

Usage: python3 cluster_control.py [h17 h17a]
Writes output/cluster_control_records.csv (one row per condition) and
output/cluster_control_provenance.json, and appends a line to
output/exp_016_run.log. It never writes the committed record files.
"""
from __future__ import annotations
import csv, datetime, json, os, sys, time, zlib
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_exp016
from lib_exp016 import load_model, load_lens, lens_vectors, positions, rank_of, JLENS_COMMIT
from swap_engine import SwapPlan, run_plan
from run_swaps import units_for, SEEDS_BY_BATTERY, LAYERS

D = os.path.dirname(os.path.abspath(__file__)) + "/"
# Which batteries and which grouping keys each battery's cluster tests use.
GROUPINGS = {"h17": ("shared_source", "shared_pair"),
             "h17a": ("shared_source",)}


def cells_for(battery):
    """The settings the committed cluster tests are computed at, read from the
    summary file so that this run and the analysis cannot drift apart."""
    s = json.load(open(D + f"output/summary_{battery}.json"))
    return sorted({tuple(v["cell"]) for v in s["cluster_tests"].values()})


def group_seed(kind, key, layer, seed):
    """A stable seed for a cluster-matched control draw, one per kind of
    sharing, group key, layer and seed index. Fixed checksum rather than
    Python's hash(), so the draws regenerate exactly."""
    return zlib.crc32(f"{kind}|{key}|{layer}|{seed}".encode("utf-8"))


def norm_matched(v, seed):
    """One random 768-number direction with the same length as `v`."""
    g = torch.Generator().manual_seed(int(seed) % (2**31))
    r = torch.randn(768, generator=g, dtype=torch.float32)
    return r / r.norm() * v.norm()


def control_vectors(kind, item, layer, seed, V):
    """The 768-by-2 matrix of random directions for one cluster-matched arm.
    Column 0 stands in for the source concept and column 1 for the target
    concept, each with the length of the lens direction it replaces."""
    src, tgt = item["source_tok"], item["target_tok"]
    if kind == "shared_source":
        s = norm_matched(V[:, 0], group_seed("src", src, layer, seed))
        t = norm_matched(V[:, 1], group_seed("tgt", item["item_id"], layer, seed))
    elif kind == "shared_pair":
        s = norm_matched(V[:, 0], group_seed("psrc", f"{src}-{tgt}", layer, seed))
        t = norm_matched(V[:, 1], group_seed("ptgt", f"{src}-{tgt}", layer, seed))
    else:
        raise ValueError(kind)
    return torch.stack([s, t], dim=1)


def committed_lens_outcomes(battery, cells):
    """The lens arm's committed outcomes at these settings, keyed by
    (item identifier, function, layer set, strength, position mode)."""
    key = {"h17": "in_top5", "h17a": "in_top5", "h17b": "is_top1"}[battery]
    want = {(c[0], f"{c[1]}", c[2]) for c in cells}
    out = {}
    with open(D + f"output/records_{battery}.csv") as fh:
        for r in csv.DictReader(fh):
            if r["arm"] != "lens":
                continue
            if (r["layers"], r["alpha"], r["posmode"]) not in want:
                continue
            out[(r["item_id"], r["func"], r["layers"], float(r["alpha"]),
                 r["posmode"])] = int(r[key])
    return out, key


def main(batteries):
    t0 = time.time()
    model = load_model()
    lens = load_lens()
    rows_out, mismatches, n_cond = [], [], 0
    for battery in batteries:
        items = json.load(open(D + f"battery_{battery}.json"))
        if battery == "h17a":
            # The cluster tests score the primary pairs; the extension set is
            # reported beside them and is not part of any cluster test.
            items = [it for it in items if it["arm"] == "primary"]
        cells = cells_for(battery)
        seeds = SEEDS_BY_BATTERY[battery]
        kinds = GROUPINGS[battery]
        committed, score_key = committed_lens_outcomes(battery, cells)
        units = units_for(battery, items)
        for ui, u in enumerate(units):
            toks = model.to_tokens(u["prompt"])
            T = toks.shape[1]
            layers_needed = sorted({int(x) for c in cells for x in c[0].split("-")})
            V = {l: lens_vectors(lens, model, l, [u["source_tok"], u["target_tok"]])
                 for l in layers_needed}
            plan = SwapPlan(T, LAYERS)
            conds = []
            for c in cells:
                ls = tuple(int(x) for x in c[0].split("-"))
                alpha, mode = float(c[1]), c[2]
                pos = positions(mode, T, u.get("first_mention_pos", 1))
                Vl = {l: V[l] for l in ls}
                plan.add(Vl, Vl, alpha, pos, False)
                conds.append((c, "lens", -1))
                for kind in kinds:
                    for s in seeds:
                        Vu = {l: control_vectors(kind, u, l, s, V[l]) for l in ls}
                        plan.add(Vu, Vl, alpha, pos, False)
                        conds.append((c, f"randdir_{kind}", s))
            plan.build()
            lp = run_plan(model, toks, plan)
            top5 = torch.topk(lp, 5, dim=-1).indices
            am = lp.argmax(dim=-1)
            for bi, (c, arm, sd) in enumerate(conds):
                g, b = u["good_tok"], u["bad_tok"]
                gr, br = rank_of(lp[bi], g), rank_of(lp[bi], b)
                row = dict(battery=battery, item_id=u["item_id"], func=u.get("func", ""),
                           split=u["split"], layers=c[0], alpha=float(c[1]),
                           posmode=c[2], arm=arm, seed=sd, good_rank=gr, bad_rank=br,
                           argmax_tok=int(am[bi]), in_top5=int(g in top5[bi]),
                           is_top1=int(am[bi] == g), beats_bad=int(gr < br),
                           patch_norm=round(float(plan.change_sq[bi].sqrt()), 4),
                           top5_ids=" ".join(str(int(t)) for t in top5[bi]))
                rows_out.append(row)
                n_cond += 1
                if arm == "lens":
                    k = (u["item_id"], u.get("func", ""), c[0], float(c[1]), c[2])
                    if committed.get(k) != row[score_key]:
                        mismatches.append((k, committed.get(k), row[score_key]))
            if ui % 20 == 0 or ui == len(units) - 1:
                print(f"[{battery}] unit {ui+1}/{len(units)} conditions {n_cond} "
                      f"elapsed {(time.time()-t0)/60:.1f} min", flush=True)
    if mismatches:
        raise RuntimeError(
            f"the re-run lens arm disagrees with the committed record on "
            f"{len(mismatches)} of the scored units, so the new control rows "
            f"cannot be tested against the committed lens rows: "
            f"{mismatches[:5]}")
    cols = ["battery", "item_id", "func", "split", "layers", "alpha", "posmode",
            "arm", "seed", "good_rank", "bad_rank", "argmax_tok", "in_top5",
            "is_top1", "beats_bad", "patch_norm", "top5_ids"]
    final = D + "output/cluster_control_records.csv"
    partial = final + ".partial"
    with open(partial, "w", newline="") as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    os.replace(partial, final)
    prov = dict(
        measured_at=datetime.datetime.utcnow().isoformat() + "Z",
        batteries=list(batteries),
        cells={b: [list(c) for c in cells_for(b)] for b in batteries},
        arms={b: ["lens"] + [f"randdir_{k}" for k in GROUPINGS[b]] for b in batteries},
        seeds={b: SEEDS_BY_BATTERY[b] for b in batteries},
        n_conditions=n_cond, n_rows=len(rows_out),
        lens_arm_reproduces_committed_records=True,
        lens_sha256=lib_exp016.LENS_SHA256_MEASURED, jlens_commit=JLENS_COMMIT,
        model_revision=lib_exp016.MODEL_REVISION,
        model_param_sha256=lib_exp016.MODEL_PARAM_SHA256_MEASURED,
        torch=torch.__version__,
        wall_seconds=round(time.time() - t0, 1))
    json.dump(prov, open(D + "output/cluster_control_provenance.json", "w"), indent=1)
    with open(D + "output/exp_016_run.log", "a") as fh:
        fh.write(f"\n[{prov['measured_at']}] cluster_control.py {' '.join(batteries)}: "
                 f"exit 0, {prov['wall_seconds']} s, {n_cond} conditions. The "
                 f"cluster-matched control arms for the cluster-level exact tests "
                 f"(one random source direction shared across the units that share a "
                 f"source lens direction, an independent random target direction per "
                 f"item, both norm matched; and for H17 a second arm sharing both "
                 f"directions across the units that share source and target). The lens "
                 f"arm was re-run beside them and reproduces the committed record files "
                 f"on every scored unit at these settings. Wrote "
                 f"output/cluster_control_records.csv and "
                 f"output/cluster_control_provenance.json; no committed file was "
                 f"overwritten.\n")
    print(f"done: {n_cond} conditions in {prov['wall_seconds']} s; "
          f"lens arm reproduces the committed records exactly")


if __name__ == "__main__":
    main(sys.argv[1:] or ["h17", "h17a"])
