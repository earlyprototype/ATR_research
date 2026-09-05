"""Record where each battery's target answer already stood before any swap.

Why this exists. The success criterion for H17a, pre-registered in section
5.2 of `_STAGE2_JSPACE/EXP_016_SPEC.md`, is that the target country's answer
token is among the unmodified model's five most likely next words after the
swap. Unlike the criterion for H17, which section 5.1 builds so that the
target is absent from the model's ten most likely next words before any
intervention, that wording does not exclude a target answer that was already
in the top five with no intervention at all. This script measures, for every
scored prompt in all three batteries, the rank of the target answer in the
unmodified model's next-word prediction, so that the analysis can report the
registered criterion beside stricter readings that exclude those cases.

Nothing here intervenes on the model. It is one clean forward pass per
distinct prompt, which is deterministic, so re-running it reproduces the
committed file exactly. It also re-measures the clean ranks the battery
files already carry, as a check that this pass matches the pass that built
them.

Usage: python3 clean_ranks.py
Writes output/clean_target_ranks.json.
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_exp016 import load_model, rank_of

D = os.path.dirname(os.path.abspath(__file__)) + "/"


def main():
    t0 = time.time()
    model = load_model()
    cache = {}

    @torch.no_grad()
    def clean_logprobs(prompt):
        """Next-word log-probabilities of the unmodified model at the end of
        `prompt`, cached so a repeated prompt is run once."""
        if prompt not in cache:
            toks = model.to_tokens(prompt)
            resid = model(toks, return_type=None, stop_at_layer=model.cfg.n_layers)
            logits = model.unembed(model.ln_final(resid[:, -1:, :]))[0, 0].float()
            cache[prompt] = torch.log_softmax(logits, dim=-1)
        return cache[prompt]

    out = {"what": ("rank of each battery's target answer in the unmodified "
                    "model's next-word prediction, 1 meaning the model's most "
                    "likely next word, measured with no intervention"),
           "model": "gpt2, loaded by transformer_lens from_pretrained_no_processing",
           "units": {}, "checks": {}}

    # H17a: one scored prompt per scoreable function of each country pair.
    rows = []
    for it in json.load(open(D + "battery_h17a.json")):
        for f in it["funcs"]:
            if not f["scoreable"]:
                continue
            lp = clean_logprobs(f["prompt"])
            rows.append(dict(item_id=it["item_id"], func=f["func"],
                             pair_set=it["arm"], split=it["split"],
                             prompt=f["prompt"], target_answer=f["target_answer"],
                             clean_answer=f["clean_answer"],
                             clean_target_rank=rank_of(lp, f["target_answer_tok"]),
                             clean_answer_rank=rank_of(lp, f["clean_answer_tok"]),
                             committed_clean_answer_rank=f["clean_rank"]))
    out["units"]["h17a"] = rows
    out["checks"]["h17a_committed_clean_answer_rank_reproduced"] = all(
        r["clean_answer_rank"] == r["committed_clean_answer_rank"] for r in rows)
    out["checks"]["h17a_targets_already_in_clean_top5"] = sum(
        1 for r in rows if r["clean_target_rank"] <= 5)
    out["checks"]["h17a_units"] = len(rows)

    # H17: one scored prompt per item. The battery file already carries the
    # target's clean rank; this pass re-measures it as a check.
    rows = []
    for it in json.load(open(D + "battery_h17.json")):
        lp = clean_logprobs(it["frame"])
        rows.append(dict(item_id=it["item_id"], func="", split=it["split"],
                         prompt=it["frame"], target_answer=it["target"],
                         clean_answer=it["source"],
                         clean_target_rank=rank_of(lp, it["target_tok"]),
                         clean_answer_rank=rank_of(lp, it["source_tok"]),
                         committed_clean_target_rank=it["clean_target_rank"]))
    out["units"]["h17"] = rows
    diffs = [abs(r["clean_target_rank"] - r["committed_clean_target_rank"])
             for r in rows]
    # One item differs by a single rank at rank 12,546 of the model's 50,257
    # words, where two words are all but tied and the pilot's arithmetic
    # ordered them the other way. Nothing scored depends on a rank that deep,
    # so the check that matters is agreement on membership of the top five.
    out["checks"]["h17_committed_clean_target_rank_items_differing"] = sum(
        1 for d in diffs if d)
    out["checks"]["h17_committed_clean_target_rank_largest_difference"] = max(diffs)
    out["checks"]["h17_committed_clean_top5_membership_reproduced"] = all(
        (r["clean_target_rank"] <= 5) == (r["committed_clean_target_rank"] <= 5)
        for r in rows)
    out["checks"]["h17_targets_already_in_clean_top5"] = sum(
        1 for r in rows if r["clean_target_rank"] <= 5)
    out["checks"]["h17_units"] = len(rows)

    # H17b: one scored prompt per item, the alternative answer being the target.
    rows = []
    for it in json.load(open(D + "battery_h17b.json")):
        lp = clean_logprobs(it["prompt"])
        rows.append(dict(item_id=it["item_id"], func="", split=it["split"],
                         prompt=it["prompt"], target_answer=it["target_answer"],
                         clean_answer=it["clean_answer"],
                         clean_target_rank=rank_of(lp, it["target_answer_tok"]),
                         clean_answer_rank=rank_of(lp, it["clean_answer_tok"]),
                         committed_clean_target_rank=it["clean_alt_rank"]))
    out["units"]["h17b"] = rows
    out["checks"]["h17b_committed_clean_alt_rank_reproduced"] = all(
        r["clean_target_rank"] == r["committed_clean_target_rank"] for r in rows)
    out["checks"]["h17b_targets_already_top1"] = sum(
        1 for r in rows if r["clean_target_rank"] == 1)
    out["checks"]["h17b_units"] = len(rows)

    out["wall_seconds"] = round(time.time() - t0, 1)
    json.dump(out, open(D + "output/clean_target_ranks.json", "w"), indent=1)
    for k, v in out["checks"].items():
        print(f"{k}: {v}")
    print(f"wrote output/clean_target_ranks.json in {out['wall_seconds']} s")


if __name__ == "__main__":
    main()
