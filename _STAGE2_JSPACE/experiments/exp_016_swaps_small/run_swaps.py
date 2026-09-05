"""Run the EXP_016 swap batteries. Executes section 3 and section 5 of
`_STAGE2_JSPACE/EXP_016_SPEC.md` exactly as pre-registered.

Usage: python3 run_swaps.py h17 | h17a | h17b
Writes one row per (item, layer set, strength, position mode, arm) to
output/records_<battery>.csv and a small provenance JSON beside it.
"""
from __future__ import annotations
import csv, json, os, sys, time, zlib
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_exp016
from lib_exp016 import (load_model, load_lens, lens_vectors, positions, rank_of,
                        JLENS_COMMIT)
from swap_engine import SwapPlan, random_pair, run_plan

D = os.path.dirname(os.path.abspath(__file__)) + "/"
LAYERS = list(range(3, 11))
SETS_FULL = [(l,) for l in LAYERS] + [(5,6),(6,7),(7,8),(8,9),(6,7,8),(7,8,9)]
SETS_B2 = [(l,) for l in LAYERS] + [(6,7,8),(7,8,9)]
ALPHAS = [0.5, 1.0, 2.0]
SEEDS_BY_BATTERY = {"h17b": [0, 1, 2], "h17": [0, 1], "h17a": [0, 1]}
CHUNK = 32
POSMODES = {"h17": ["all", "last"],
            "h17a": ["all"],
            "h17b": ["all_no_bos", "from_mention", "answer_only"]}


def control_seed(item_id, func, layer, seed):
    """A stable seed for the random-direction controls. The first run used
    Python's hash() here, which is randomised per process, so the committed
    control draws cannot be regenerated; every run from this version on
    derives the seed from a fixed checksum of the same tuple."""
    return zlib.crc32(f"{item_id}|{func}|{layer}|{seed}".encode("utf-8"))


def arms(seeds):
    out = [("lens", -1)]
    out += [("randdir", s) for s in seeds]
    out += [("randnorm", s) for s in seeds]
    return out


def units_for(battery, items):
    """Expand each battery item into one or more scored prompts (units)."""
    units = []
    for it in items:
        if battery == "h17a":
            for f in it["funcs"]:
                if f["scoreable"]:
                    units.append(dict(it, prompt=f["prompt"], func=f["func"],
                                      good_tok=f["target_answer_tok"],
                                      bad_tok=f["clean_answer_tok"]))
        elif battery == "h17b":
            units.append(dict(it, func="", good_tok=it["target_answer_tok"],
                              bad_tok=it["clean_answer_tok"]))
        else:
            units.append(dict(it, prompt=it["frame"], func="",
                              good_tok=it["target_tok"], bad_tok=it["source_tok"]))
    return units


def main(battery):
    t0 = time.time()
    model = load_model()
    lens = load_lens()
    items = json.load(open(D + f"battery_{battery}.json"))
    sets = SETS_B2 if battery == "h17a" else SETS_FULL
    modes = POSMODES[battery]
    seeds = SEEDS_BY_BATTERY[battery]

    # sanity check: the cheap read-out path equals the model's own word scores
    tk = model.to_tokens("The capital of France is the city of")
    with torch.no_grad():
        full = model(tk)[0, -1].float()
        r = model(tk, return_type=None, stop_at_layer=model.cfg.n_layers)
        cheap = model.unembed(model.ln_final(r[:, -1:, :]))[0, 0].float()
    assert (full - cheap).abs().max().item() < 1e-3, "read-out path mismatch"

    units = units_for(battery, items)


    fh = open(D + f"output/records_{battery}.csv", "w", newline="")
    w = csv.writer(fh)
    w.writerow(["item_id", "func", "split", "layers", "alpha", "posmode",
                "arm", "seed", "good_rank", "bad_rank", "argmax_tok",
                "in_top5", "is_top1", "beats_bad", "patch_norm"])
    n_done = 0
    for ui, u in enumerate(units):
        toks = model.to_tokens(u["prompt"])
        T = toks.shape[1]
        mention = u.get("first_mention_pos", 1)
        V_lens, V_rand = {}, {}
        for l in LAYERS:
            W = lens_vectors(lens, model, l, [u["source_tok"], u["target_tok"]])
            V_lens[l] = W
            for s in seeds:
                V_rand[(l, s)] = random_pair(W[:, 0], W[:, 1],
                                             control_seed(u["item_id"], u.get("func", ""), l, s))
        conds = [(ls, a, m, arm, sd) for ls in sets for a in ALPHAS
                 for m in modes for arm, sd in arms(seeds)]
        for c0 in range(0, len(conds), CHUNK):
            chunk = conds[c0:c0 + CHUNK]
            plan = SwapPlan(T, LAYERS)
            for ls, a, m, arm, sd in chunk:
                pos = positions(m, T, mention)
                Vl = {l: V_lens[l] for l in ls}
                Vu = Vl if arm == "lens" else {l: V_rand[(l, sd)] for l in ls}
                plan.add(Vu, Vl, a, pos, arm == "randnorm")
            plan.build()
            lp = run_plan(model, toks, plan)
            top5 = torch.topk(lp, 5, dim=-1).indices
            am = lp.argmax(dim=-1)
            for bi, (ls, a, m, arm, sd) in enumerate(chunk):
                g, b = u["good_tok"], u["bad_tok"]
                gr = rank_of(lp[bi], g)
                br = rank_of(lp[bi], b)
                w.writerow([u["item_id"], u.get("func", ""), u["split"],
                            "-".join(map(str, ls)), a, m, arm, sd, gr, br,
                            int(am[bi]), int(g in top5[bi]), int(am[bi] == g),
                            int(gr < br), round(float(plan.change_sq[bi].sqrt()), 4)])
            n_done += len(chunk)
        if ui % 5 == 0 or ui == len(units) - 1:
            el = time.time() - t0
            print(f"[{battery}] unit {ui+1}/{len(units)} conds {n_done} "
                  f"elapsed {el/60:.1f} min "
                  f"eta {el/max(n_done,1)*(len(units)*len(conds)-n_done)/60:.1f} min",
                  flush=True)
    fh.close()
    json.dump(dict(battery=battery, n_units=len(units), n_conditions=n_done,
                   layer_sets=[list(s) for s in sets], alphas=ALPHAS,
                   posmodes=modes, seeds=seeds, chunk=CHUNK,
                   control_seed_scheme="crc32(item_id|func|layer|seed)",
                   patch_norm="Euclidean norm of the total residual change over patched positions and layers",
                   lens_sha256=lib_exp016.LENS_SHA256_MEASURED, jlens_commit=JLENS_COMMIT,
                   torch=torch.__version__,
                   wall_seconds=round(time.time() - t0, 1)),
              open(D + f"output/provenance_{battery}.json", "w"), indent=1)
    print(f"[{battery}] done: {n_done} conditions in "
          f"{(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
