"""Second clean-accuracy pilot for EXP_016: more two-hop items, and the
country gate arithmetic. No swaps are performed here."""
import json, sys, time, torch
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_exp016 import (load_model, single_token_id, first_token_id, clean_run,
                        rank_of)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output", "pilot_clean2.json")
P1 = json.load(open(os.path.join(HERE, "output", "pilot_clean.json")))
t0 = time.time(); model = load_model()

EXTRA = [
 ("duck-sound","Ducks say quack and cows say moo. The bird that swims on the pond says"," quack"," moo"," duck"," cow"),
 ("farmer-place","Nurses work in hospitals and farmers work in fields. The person who milks cows works in"," fields"," hospitals"," farmer"," nurse"),
 ("sugar-taste","Sugar is sweet and salt is salty. The white powder you put in your tea is"," sweet"," salty"," sugar"," salt"),
 ("rain-wet","Rain is wet and sand is dry. The thing that falls from the clouds is"," wet"," dry"," rain"," sand"),
 ("guitar-part","Guitars have strings and drums have skins. The instrument that you strum has"," strings"," skins"," guitar"," drum"),
 ("tiger-pattern","Tigers have stripes and leopards have spots. The big cat from India has"," stripes"," spots"," tiger"," leopard"),
 ("wood-float","Wood floats and stone sinks. In water, a log of wood will"," float"," sink"," wood"," stone"),
 ("milk-colour","Milk is white and coffee is black. The drink that comes from cows is"," white"," black"," milk"," coffee"),
 ("knife-use","Knives cut and spoons scoop. The tool with a sharp blade is used to"," cut"," scoop"," knife"," spoon"),
 ("summer-temp","Summer is warm and winter is freezing. The season with the longest days is"," warm"," freezing"," summer"," winter"),
 ("banana-grow","Bananas grow on trees and potatoes grow underground. The long yellow fruit grows on"," trees"," underground"," banana"," potato"),
 ("camel-home","Whales live in the ocean and camels live in the desert. The animal with a hump lives in the"," desert"," ocean"," camel"," whale"),
 ("clock-show","Clocks tell the time and maps show the way. The thing with hands and numbers tells the"," time"," way"," clock"," map"),
 ("chicken-give","Chickens lay eggs and sheep give wool. The bird on the farm lays"," eggs"," wool"," chicken"," sheep"),
 ("pencil-made","Pencils are made of wood and coins are made of metal. The thing that you write with is made of"," wood"," metal"," pencil"," coin"),
 ("rose-colour","Roses are red and violets are blue. The flower with thorns is"," red"," blue"," rose"," violet"),
]
rows = []
for name, prompt, ans, alt, src, tgt in EXTRA:
    lp, _, T = clean_run(model, prompt)
    rows.append(dict(name=name, prompt=prompt, answer=ans, alt=alt,
                     swap_from=src, swap_to=tgt, n_tokens=T,
                     src_tok=single_token_id(model, src),
                     tgt_tok=single_token_id(model, tgt),
                     answer_rank=rank_of(lp, first_token_id(model, ans)),
                     alt_rank=rank_of(lp, first_token_id(model, alt)),
                     argmax=model.to_string(int(lp.argmax())),
                     top5=[model.to_string(t) for t in torch.topk(lp,5).indices.tolist()]))
    r = rows[-1]
    print(f"[b3x] {name:16s} ans={r['answer_rank']:5d} alt={r['alt_rank']:5d} "
          f"argmax={r['argmax']!r:12s} toks({r['src_tok']},{r['tgt_tok']})", flush=True)

allrows = P1["battery3"] + rows
for gate in (1, 3, 5, 10):
    keep = [r for r in allrows if r["answer_rank"] <= gate
            and r["answer_rank"] < r["alt_rank"]
            and r["src_tok"] is not None and r["tgt_tok"] is not None]
    print(f"[b3 gate] top{gate} + answer above alternative: {len(keep)}/{len(allrows)} "
          f"-> {[r['name'] for r in keep]}", flush=True)

# ---- battery 2 gate arithmetic on the chosen frames (v0 of each function)
best = {b["func"]: b for b in P1["battery2"]
        if b["frame"] in ("The capital of {X} is the city of",
                          "Most people in {X} speak",
                          "{X} is a country on the continent of")}
for gate in (1, 3, 5):
    per_func = {f: {c for c, d in b["per_country"].items() if d["rank"] <= gate}
                for f, b in best.items()}
    inter = set.intersection(*per_func.values())
    print(f"[b2 gate] top{gate}: capital {len(per_func['capital'])}, "
          f"language {len(per_func['language'])}, continent {len(per_func['continent'])}, "
          f"all three {len(inter)} -> {sorted(inter)}", flush=True)

json.dump(dict(extra_twohop=rows, wall_seconds=round(time.time()-t0,1)),
          open(OUT,"w"), indent=1)
print("wrote", OUT, round(time.time()-t0,1), "s")
