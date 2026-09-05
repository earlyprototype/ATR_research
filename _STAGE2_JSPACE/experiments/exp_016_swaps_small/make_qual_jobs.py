"""Pick the qualitative examples to re-run: for each battery, the tuned
setting and a handful of items, chosen to include successes and failures."""
import csv, json, sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyse

D = os.path.dirname(os.path.abspath(__file__)) + "/"
jobs = []
for b in ("h17", "h17a", "h17b"):
    s = json.load(open(D + f"output/summary_{b}.json"))
    cell = s["chosen_cell"]
    rows = analyse.load(b)
    key = analyse.SCORE[b]
    hits = analyse.per_item(rows, cell, key)
    items = {i["item_id"]: i for i in json.load(open(D + f"battery_{b}.json"))}
    ls = tuple(int(x) for x in cell[0].split("-"))
    won = [k for k, v in hits.items() if v[0] > 0]
    lost = [k for k, v in hits.items() if v[0] == 0]
    pick = won[:4] + lost[:2]
    for iid, func in pick:
        it = items[iid]
        if b == "h17":
            prompt = it["frame"]
        elif b == "h17a":
            prompt = [f["prompt"] for f in it["funcs"] if f["func"] == func][0]
        else:
            prompt = it["prompt"]
        jobs.append(dict(prompt=prompt, s_tok=it["source_tok"],
                         t_tok=it["target_tok"], layer_set=list(ls),
                         alpha=cell[1], mode=cell[2],
                         mention=it.get("first_mention_pos", 1)))
json.dump(jobs, open(D + "output/qual_jobs.json", "w"), indent=1)
print("wrote", len(jobs), "qualitative jobs")
