# EXP_012m — band census for gpt2-medium (issue #15, RUNBOOK_JLENS_MEDIUM.md §4).
#
# Applies the fitted J-lens layer-by-layer (all fitted layers 0..22; "layer l"
# = residual at the OUTPUT of block l) to the held-out 50-prompt set
# (heldout_50.json, derivation in derive_heldout.py) at the last position,
# alongside the logit-lens readout of the identical residuals
# (use_jacobian=False). Also computes a per-layer multi-hop intermediate-rank
# curve on the instrument repo's lens-eval-multihop.json (93 items; readout at
# the last prompt position per that data set's README).
#
# Per layer, recorded:
#   - agreement-with-final-layer: fraction of held-out prompts whose lens
#     top-1 equals the model's final top-1 (J-lens and logit-lens)
#   - median rank of the model's final top-1 token in each lens ranking
#   - word-like fraction of lens top-1 (>=3 chars, alphabetic after strip)
#   - J-better fraction: prompts where rank_J(final top-1) < rank_logit(...)
#   - multihop: fraction of intermediates with lens rank <= 10 (first token
#     of " {intermediate}")
#   - top-5 readouts per prompt (qualitative record)
#
# BAND RULE — pre-registered before any census output existed (committed with
# this file; see RESULTS_JLENS_MEDIUM.md §3):
#   Layer l is lens-dominant iff, on the 50 held-out prompts at position -1:
#     (i)   median rank_J(final top-1) < median rank_logit(final top-1), and
#     (ii)  J-better fraction >= 0.6, and
#     (iii) J-lens top-1 agreement-with-final >= 0.10.
#   Band [L_lo, L_hi] = the maximal contiguous run of lens-dominant layers,
#   length >= 3. No such run -> "no coherent band". Ties/islands reported flat.

import json
import os

import torch

import jlens

from fit_lens import ARTIFACTS, load_model

HERE = os.path.dirname(os.path.abspath(__file__))
MULTIHOP = os.path.join(
    os.path.dirname(os.path.dirname(HERE)),
    "instrument",
    "jacobian-lens",
    "data",
    "evaluations",
    "lens-eval-multihop.json",
)
TOPK = 5


def rank_of(logits: torch.Tensor, token_id: int) -> int:
    return int((logits > logits[token_id]).sum().item()) + 1  # 1-based


def word_like(s: str) -> bool:
    t = s.strip()
    return len(t) >= 3 and t.isalpha()


def main():
    model, tok = load_model()
    lens = jlens.JacobianLens.load(
        os.path.join(ARTIFACTS, "jlens_gpt2_medium_100.pt")
    )
    layers = lens.source_layers
    print(lens)

    heldout = json.load(open(os.path.join(HERE, "heldout_50.json")))
    per_prompt = []
    for rec in heldout:
        jl, model_logits, _ = lens.apply(
            model, rec["prompt"], layers=layers, positions=[-1]
        )
        ll, _, _ = lens.apply(
            model, rec["prompt"], layers=layers, positions=[-1], use_jacobian=False
        )
        final_top1 = int(model_logits[0].argmax())
        row = {
            "id": rec["id"],
            "final_top1": tok.decode([final_top1]),
            "layers": {},
        }
        for l in layers:
            jlog, llog = jl[l][0], ll[l][0]
            row["layers"][str(l)] = {
                "j_top5": [tok.decode([t]) for t in jlog.topk(TOPK).indices.tolist()],
                "l_top5": [tok.decode([t]) for t in llog.topk(TOPK).indices.tolist()],
                "j_rank_final": rank_of(jlog, final_top1),
                "l_rank_final": rank_of(llog, final_top1),
                "j_top1_eq_final": bool(int(jlog.argmax()) == final_top1),
                "l_top1_eq_final": bool(int(llog.argmax()) == final_top1),
            }
        per_prompt.append(row)
        print("censused", rec["id"], flush=True)

    # Per-layer aggregates
    agg = {}
    n = len(per_prompt)
    for l in layers:
        k = str(l)
        j_ranks = sorted(r["layers"][k]["j_rank_final"] for r in per_prompt)
        l_ranks = sorted(r["layers"][k]["l_rank_final"] for r in per_prompt)
        med = lambda xs: xs[len(xs) // 2] if len(xs) % 2 else (
            xs[len(xs) // 2 - 1] + xs[len(xs) // 2]
        ) / 2
        j_better = sum(
            r["layers"][k]["j_rank_final"] < r["layers"][k]["l_rank_final"]
            for r in per_prompt
        ) / n
        agg[k] = {
            "j_agree_final": sum(r["layers"][k]["j_top1_eq_final"] for r in per_prompt) / n,
            "l_agree_final": sum(r["layers"][k]["l_top1_eq_final"] for r in per_prompt) / n,
            "j_median_rank_final": med(j_ranks),
            "l_median_rank_final": med(l_ranks),
            "j_better_frac": j_better,
            "j_wordlike_top1": sum(
                word_like(r["layers"][k]["j_top5"][0]) for r in per_prompt
            ) / n,
            "l_wordlike_top1": sum(
                word_like(r["layers"][k]["l_top5"][0]) for r in per_prompt
            ) / n,
        }

    # Pre-registered band rule
    dominant = [
        l
        for l in layers
        if agg[str(l)]["j_median_rank_final"] < agg[str(l)]["l_median_rank_final"]
        and agg[str(l)]["j_better_frac"] >= 0.6
        and agg[str(l)]["j_agree_final"] >= 0.10
    ]
    runs, cur = [], []
    for l in layers:
        if l in dominant:
            if cur and l == cur[-1] + 1:
                cur.append(l)
            else:
                if cur:
                    runs.append(cur)
                cur = [l]
    if cur:
        runs.append(cur)
    long_runs = [r for r in runs if len(r) >= 3]
    band = max(long_runs, key=len) if long_runs else None

    # Multihop per-layer intermediate-rank curve
    items = json.load(open(MULTIHOP))["items"]
    mh = {str(l): {"hits10": 0, "hits1": 0, "n": 0} for l in layers}
    mh_l = {str(l): {"hits10": 0, "n": 0} for l in layers}
    for item in items:
        jl, _, _ = lens.apply(model, item["prompt"], layers=layers, positions=[-1])
        ll, _, _ = lens.apply(
            model, item["prompt"], layers=layers, positions=[-1], use_jacobian=False
        )
        for inter in item["intermediates"]:
            # NOTE deviation record: from_hf(force_bos=True) sets
            # add_bos_token=True on the GPT-2 tokenizer, so encode() prepends
            # <|endoftext|>; the first RUN of this script took ids[0] and
            # therefore scored the BOS token (mh columns all 0.00). Fixed to
            # strip the BOS; first-run artifacts superseded, bug recorded in
            # RESULTS §5.
            ids = tok.encode(" " + inter)
            if ids and ids[0] == tok.bos_token_id:
                ids = ids[1:]
            tid = ids[0]
            for l in layers:
                rj = rank_of(jl[l][0], tid)
                rl = rank_of(ll[l][0], tid)
                mh[str(l)]["n"] += 1
                mh[str(l)]["hits10"] += rj <= 10
                mh[str(l)]["hits1"] += rj == 1
                mh_l[str(l)]["n"] += 1
                mh_l[str(l)]["hits10"] += rl <= 10
        print("multihop", item["name"], flush=True)

    for l in layers:
        k = str(l)
        agg[k]["mh_j_hit10"] = mh[k]["hits10"] / mh[k]["n"]
        agg[k]["mh_j_hit1"] = mh[k]["hits1"] / mh[k]["n"]
        agg[k]["mh_l_hit10"] = mh_l[k]["hits10"] / mh_l[k]["n"]

    out = {
        "band_rule": "see file header; pre-registered",
        "dominant_layers": dominant,
        "runs": runs,
        "band": [band[0], band[-1]] if band else None,
        "per_layer": agg,
        "per_prompt": per_prompt,
    }
    json.dump(out, open(os.path.join(HERE, "census_results.json"), "w"), indent=1)

    lines = [
        "# EXP_012m band census — per-layer aggregates (50 held-out prompts, pos -1)",
        "",
        "| L | J agree | LL agree | J med-rank | LL med-rank | J-better | J word | LL word | mh J@10 | mh LL@10 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for l in layers:
        a = agg[str(l)]
        lines.append(
            f"| {l} | {a['j_agree_final']:.2f} | {a['l_agree_final']:.2f} "
            f"| {a['j_median_rank_final']:.0f} | {a['l_median_rank_final']:.0f} "
            f"| {a['j_better_frac']:.2f} | {a['j_wordlike_top1']:.2f} "
            f"| {a['l_wordlike_top1']:.2f} | {a['mh_j_hit10']:.2f} "
            f"| {a['mh_l_hit10']:.2f} |"
        )
    lines += [
        "",
        f"Lens-dominant layers (pre-registered rule): {dominant}",
        f"Contiguous runs: {runs}",
        f"**Band: {f'[{band[0]}, {band[-1]}]' if band else 'NO COHERENT BAND'}**",
    ]
    open(os.path.join(HERE, "census_table.md"), "w").write("\n".join(lines))
    print("band:", band)


if __name__ == "__main__":
    main()
