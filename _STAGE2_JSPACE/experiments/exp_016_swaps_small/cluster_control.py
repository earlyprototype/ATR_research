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
  randdir_by_token       added 2026-09-06 after review, and the arm the record
      now leads with. Each random direction is seeded by the token it stands
      in for, and by the layer and the seed index, so two units that swap the
      same concept receive the same random direction exactly as they receive
      the same lens direction, whether that concept is their source or their
      target. The two arms above seed the target direction by item, which
      breaks that reuse: the four held-out frames that all swap ' football' to
      ' cricket' received four independent random target directions where the
      lens arm gives them one, and the same holds for the two frames that swap
      ' blue' to ' yellow'.
  randdir_mirrored       added 2026-09-06 after review, for H17 only. The same
      token-seeded directions, but the control also chooses its own target
      concept rather than being handed the lens arm's: it applies the rule the
      battery used, the candidate the layer-8 readout ranks highest among the
      category members that are absent from the model's ten most likely next
      words and are not the source, to its own random directions, and its
      success is that chosen concept entering the model's five most likely next
      words. For the `lens` source rule, whose source is also chosen by a
      layer-8 lens ranking, the control chooses its source the same way. This
      is the only arm whose target is not conditioned on a favourable lens
      readout, so it is the only one whose probability is not conditional on
      the lens-selected targets. H17a and H17b need no mirrored arm: H17a's
      target country comes from a fixed rotation over the gated countries and
      H17b's alternative answer is written into the item, so neither is chosen
      by a lens ranking.

The lens arm is re-run beside them at the same settings. It is deterministic,
so its outcomes must reproduce the committed record files exactly; the script
checks that and refuses to write if they do not, which is what makes it sound
to test the committed lens rows against these new control rows.

H17b needs none of this: each of its 16 items has its own source concept, so
every group holds one item and the cluster-level test is the within-item test,
whose draws are exchangeable already.

Usage: python3 cluster_control.py [h17 h17a]
Writes output/cluster_control_records_v2.csv (one row per condition) and
output/cluster_control_provenance_v2.json, and appends a line to
output/exp_016_run.log. It never writes the committed record files. The v2
files carry every arm, the two of 2026-09-06 that this script first ran and
the two added later the same day after review, so the first run's file
`output/cluster_control_records.csv` stays as the record of that run and is
reproduced row for row inside the new one.
"""
from __future__ import annotations
import csv, datetime, json, os, sys, time, zlib
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_exp016
from lib_exp016 import (load_model, load_lens, lens_vectors, positions, rank_of,
                        clean_run, single_token_id, JLENS_COMMIT)
from swap_engine import SwapPlan, run_plan
from run_swaps import units_for, SEEDS_BY_BATTERY, LAYERS

D = os.path.dirname(os.path.abspath(__file__)) + "/"
# Which control arms each battery gets. The first two share randomness the way
# the lens arm does at the level of a cluster; `by_token` shares it at the
# level of the token, which is how the lens arm actually shares it; `mirrored`
# adds the battery's own target-selection rule on top.
GROUPINGS = {"h17": ("shared_source", "shared_pair", "by_token", "mirrored"),
             "h17a": ("shared_source", "by_token")}
# The layer at which the battery ranked candidates when it chose each item's
# target concept, from section 5.1 of the specification.
SELECT_LAYER = 8


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


def token_direction(v, tok, layer, seed):
    """One random direction standing in for the concept whose token is `tok`,
    with the length of that concept's lens direction at this layer, drawn from
    a seed that depends only on the token, the layer and the seed index. Two
    units that swap the same concept therefore receive the same random
    direction, whether that concept is the source of one and the target of the
    other, exactly as they receive the same lens direction."""
    return norm_matched(v, group_seed("token", tok, layer, seed))


def control_vectors(kind, item, layer, seed, V, sel=None):
    """The 768-by-2 matrix of random directions for one cluster-matched arm.
    Column 0 stands in for the source concept and column 1 for the target
    concept, each with the length of the lens direction it replaces. For the
    mirrored arm the two concepts are the ones the control chose for itself,
    passed in `sel` as (source token, target token, source lens vector, target
    lens vector)."""
    src, tgt = item["source_tok"], item["target_tok"]
    if kind == "shared_source":
        s = norm_matched(V[:, 0], group_seed("src", src, layer, seed))
        t = norm_matched(V[:, 1], group_seed("tgt", item["item_id"], layer, seed))
    elif kind == "shared_pair":
        s = norm_matched(V[:, 0], group_seed("psrc", f"{src}-{tgt}", layer, seed))
        t = norm_matched(V[:, 1], group_seed("ptgt", f"{src}-{tgt}", layer, seed))
    elif kind == "by_token":
        s = token_direction(V[:, 0], src, layer, seed)
        t = token_direction(V[:, 1], tgt, layer, seed)
    elif kind == "mirrored":
        s_tok, t_tok, v_s, v_t = sel
        s = token_direction(v_s, s_tok, layer, seed)
        t = token_direction(v_t, t_tok, layer, seed)
    else:
        raise ValueError(kind)
    return torch.stack([s, t], dim=1)


def selection_context(model, lens, prow):
    """Everything the mirrored arm needs for one H17 frame: the category's
    member names and their tokens, the lens directions at the selection layer,
    the unmodified model's residual stream at that layer at the final
    position, and the members the model already ranks in its ten most likely
    next words, which the battery's rule excludes from the candidates."""
    names = sorted(prow["lens_ranks"])
    toks = {n: single_token_id(model, n) for n in names}
    _, cache, _ = clean_run(model, prow["frame"])
    h = cache[f"blocks.{SELECT_LAYER}.hook_resid_post"][0, -1].float()
    V = lens_vectors(lens, model, SELECT_LAYER, [toks[n] for n in names])
    return dict(names=names, toks=toks, h=h, V=V,
                excluded=set(prow["members_in_clean_top10"]))


def mirrored_pick(ctx, item, seed):
    """The concepts the mirrored control chooses for itself at one seed. Each
    category member is given a random direction of the length of its own lens
    direction at the selection layer, seeded by its token, and the member whose
    direction the unmodified residual stream carries most is chosen, which is
    the random-direction analogue of the battery's rule, the member the lens
    ranks highest at that layer. The source is chosen the same way for items
    built by the `lens` source rule, whose source the lens also chose; for
    items built by the `output` rule the source is the model's own most likely
    category member and no lens reading enters it, so it is left as it is.
    Returns (source name, target name) or None when no candidate survives."""
    score = {}
    for i, n in enumerate(ctx["names"]):
        r = token_direction(ctx["V"][:, i], ctx["toks"][n], SELECT_LAYER, seed)
        score[n] = float(ctx["h"] @ r)
    s = (max(ctx["names"], key=lambda n: score[n])
         if item["source_rule"] == "lens" else item["source"])
    cand = [n for n in ctx["names"] if n not in ctx["excluded"] and n != s]
    if not cand:
        return None
    return s, max(cand, key=lambda n: score[n])


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
    pilot = {r["frame"]: r for r in json.load(open(D + "output/pilot_clean.json"))["battery1"]}
    ctx_cache = {}
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
                conds.append((c, "lens", -1, u["good_tok"], u["bad_tok"],
                              u.get("source", ""), u.get("target", "")))
                for kind in kinds:
                    for s in seeds:
                        if kind == "mirrored":
                            if u["frame"] not in ctx_cache:
                                ctx_cache[u["frame"]] = selection_context(
                                    model, lens, pilot[u["frame"]])
                            ctx = ctx_cache[u["frame"]]
                            pick = mirrored_pick(ctx, u, s)
                            if pick is None:
                                continue
                            sn, tn = pick
                            s_tok, t_tok = ctx["toks"][sn], ctx["toks"][tn]
                            Vsel = {l: lens_vectors(lens, model, l, [s_tok, t_tok])
                                    for l in ls}
                            Vu = {l: control_vectors(kind, u, l, s,
                                                     Vsel[l],
                                                     (s_tok, t_tok, Vsel[l][:, 0],
                                                      Vsel[l][:, 1]))
                                  for l in ls}
                            good, bad, selp = t_tok, s_tok, (sn, tn)
                        else:
                            Vu = {l: control_vectors(kind, u, l, s, V[l]) for l in ls}
                            good, bad = u["good_tok"], u["bad_tok"]
                            selp = (u.get("source", ""), u.get("target", ""))
                        plan.add(Vu, Vl, alpha, pos, False)
                        conds.append((c, f"randdir_{kind}", s, good, bad,
                                      selp[0], selp[1]))
            plan.build()
            lp = run_plan(model, toks, plan)
            top5 = torch.topk(lp, 5, dim=-1).indices
            am = lp.argmax(dim=-1)
            for bi, (c, arm, sd, g, b, sel_s, sel_t) in enumerate(conds):
                gr, br = rank_of(lp[bi], g), rank_of(lp[bi], b)
                row = dict(battery=battery, item_id=u["item_id"], func=u.get("func", ""),
                           split=u["split"], layers=c[0], alpha=float(c[1]),
                           posmode=c[2], arm=arm, seed=sd, good_rank=gr, bad_rank=br,
                           argmax_tok=int(am[bi]), in_top5=int(g in top5[bi]),
                           is_top1=int(am[bi] == g), beats_bad=int(gr < br),
                           patch_norm=round(float(plan.change_sq[bi].sqrt()), 4),
                           top5_ids=" ".join(str(int(t)) for t in top5[bi]),
                           sel_source=sel_s, sel_target=sel_t)
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
            "is_top1", "beats_bad", "patch_norm", "top5_ids",
            "sel_source", "sel_target"]
    final = D + "output/cluster_control_records_v2.csv"
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
    json.dump(prov, open(D + "output/cluster_control_provenance_v2.json", "w"), indent=1)
    with open(D + "output/exp_016_run.log", "a") as fh:
        fh.write(f"\n[{prov['measured_at']}] cluster_control.py {' '.join(batteries)}: "
                 f"exit 0, {prov['wall_seconds']} s, {n_cond} conditions. Four "
                 f"cluster-matched control arms for the cluster-level exact tests: "
                 f"shared_source (one random source direction per cluster, an "
                 f"independent random target direction per item), shared_pair (both "
                 f"shared across the units that share source and target), by_token "
                 f"(every random direction seeded by the token it stands in for, so "
                 f"reuse across units matches the lens arm's exactly) and, for H17, "
                 f"mirrored (token-seeded, and the control picks its own target, and "
                 f"for the lens source rule its own source, by the battery's layer-8 "
                 f"rule applied to its own random directions, scored on its own "
                 f"chosen concept). The lens arm was re-run beside them and reproduces "
                 f"the committed record files on every scored unit at these settings. "
                 f"Wrote output/cluster_control_records_v2.csv and "
                 f"output/cluster_control_provenance_v2.json; no committed file was "
                 f"overwritten.\n")
    print(f"done: {n_cond} conditions in {prov['wall_seconds']} s; "
          f"lens arm reproduces the committed records exactly")


if __name__ == "__main__":
    main(sys.argv[1:] or ["h17", "h17a"])
