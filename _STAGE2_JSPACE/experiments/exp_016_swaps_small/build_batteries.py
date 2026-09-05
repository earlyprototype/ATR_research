"""Build the three EXP_016 batteries from the clean-accuracy pilots.

No swaps are performed here. Every selection rule is the one written into
`_STAGE2_JSPACE/EXP_016_SPEC.md`; this script is that specification executed,
so the committed battery files are reproducible from the pilot outputs.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_exp016 import load_model, single_token_id, first_token_id

D = os.path.dirname(os.path.abspath(__file__)) + "/"
P1 = json.load(open(D + "output/pilot_clean.json"))
P2 = json.load(open(D + "output/pilot_clean2.json"))
model = load_model()

# ------------------------------------------------------------------ battery 1
b1 = []
by_cat = {}
for row in P1["battery1"]:
    by_cat.setdefault(row["category"], []).append(row)
for cat, rows in by_cat.items():
    for idx, r in enumerate(rows):
        names = sorted(r["lens_ranks"])
        excluded = set(r["members_in_clean_top10"])
        s_lens = min(names, key=lambda n: r["lens_ranks"][n])
        s_out = min(names, key=lambda n: r["clean_ranks"][n])
        for rule, s in (("lens", s_lens), ("output", s_out)):
            cand = [n for n in names if n not in excluded and n != s]
            if not cand:
                continue
            t = min(cand, key=lambda n: r["lens_ranks"][n])
            b1.append(dict(
                item_id=f"{cat}-{idx:02d}-{rule}", category=cat, frame=r["frame"],
                source_rule=rule, source=s, target=t,
                source_tok=single_token_id(model, s),
                target_tok=single_token_id(model, t),
                clean_target_rank=r["clean_ranks"][t],
                clean_source_rank=r["clean_ranks"][s],
                clean_argmax=r["argmax"],
                clean_top5=r["clean_top10"][:5],
                split="tuning" if idx % 2 == 0 else "heldout"))

# ------------------------------------------------------------------ battery 2
FR = {"capital": "The capital of {X} is the city of",
      "language": "Most people in {X} speak",
      "continent": "{X} is a country on the continent of"}
ANS = {  # first-token answers, as piloted
 "France":{"capital":" Paris","language":" French","continent":" Europe"},
 "Germany":{"capital":" Berlin","language":" German","continent":" Europe"},
 "Italy":{"capital":" Rome","language":" Italian","continent":" Europe"},
 "Spain":{"capital":" Madrid","language":" Spanish","continent":" Europe"},
 "Russia":{"capital":" Moscow","language":" Russian","continent":" Europe"},
 "Poland":{"capital":" Warsaw","language":" Polish","continent":" Europe"},
 "Greece":{"capital":" Athens","language":" Greek","continent":" Europe"},
 "Ireland":{"capital":" Dublin","language":" English","continent":" Europe"},
 "Portugal":{"capital":" Lisbon","language":" Portuguese","continent":" Europe"},
 "Norway":{"capital":" Oslo","language":" Norwegian","continent":" Europe"},
 "Sweden":{"capital":" Stockholm","language":" Swedish","continent":" Europe"},
 "China":{"capital":" Beijing","language":" Chinese","continent":" Asia"},
 "Japan":{"capital":" Tokyo","language":" Japanese","continent":" Asia"},
 "India":{"capital":" Delhi","language":" Hindi","continent":" Asia"},
 "Iran":{"capital":" Tehran","language":" Persian","continent":" Asia"},
 "Israel":{"capital":" Jerusalem","language":" Hebrew","continent":" Asia"},
 "Egypt":{"capital":" Cairo","language":" Arabic","continent":" Africa"},
 "Kenya":{"capital":" Nairobi","language":" Swahili","continent":" Africa"},
 "Nigeria":{"capital":" Abuja","language":" English","continent":" Africa"},
 "Canada":{"capital":" Ottawa","language":" English","continent":" North"},
 "Mexico":{"capital":" Mexico","language":" Spanish","continent":" North"},
 "Brazil":{"capital":" Brasilia","language":" Portuguese","continent":" South"},
 "Chile":{"capital":" Santiago","language":" Spanish","continent":" South"},
 "Australia":{"capital":" Canberra","language":" English","continent":" Australia"},
 "Turkey":{"capital":" Ankara","language":" Turkish","continent":" Asia"},
}
ranks = {b["func"]: b["per_country"] for b in P1["battery2"] if b["frame"] in FR.values()}
def gated(g):
    return sorted(c for c in ANS
                  if all(ranks[f][c]["rank"] <= g for f in FR))
G3, G5 = gated(3), gated(5)
pairs = []
for i, X in enumerate(G3):
    for off in (1, 3, 5):
        Y = G3[(i + off) % len(G3)]
        if X != Y:
            pairs.append((X, Y, "primary"))
firsts = {}
for c in G5:
    firsts.setdefault(ANS[c]["continent"], c)
C = sorted(firsts.values())
for X in C:
    for Y in C:
        if X != Y:
            pairs.append((X, Y, "extension"))
b2 = []
for X, Y, arm in pairs:
    funcs = []
    for f, tmpl in FR.items():
        a_src, a_tgt = ANS[X][f], ANS[Y][f]
        funcs.append(dict(func=f, prompt=tmpl.replace("{X}", X),
                          clean_answer=a_src, target_answer=a_tgt,
                          clean_answer_tok=first_token_id(model, a_src),
                          target_answer_tok=first_token_id(model, a_tgt),
                          scoreable=a_src != a_tgt,
                          clean_rank=ranks[f][X]["rank"]))
    b2.append(dict(item_id=f"{X}->{Y}", source=X, target=Y, arm=arm,
                   source_tok=single_token_id(model, " " + X),
                   target_tok=single_token_id(model, " " + Y),
                   n_scoreable=sum(f["scoreable"] for f in funcs),
                   funcs=funcs,
                   # The run assigned halves by source country, so a country's
                   # three (or four) pairs sit in one half: 27 tuning and 23
                   # held-out. The specification says alternate items, which for
                   # this battery means alternate pairs (25 and 25); analyse.py
                   # reports that reading beside this one (deviation 17). The
                   # rule is kept here so the committed batteries rebuild exactly.
                   split="tuning" if (G3.index(X) if X in G3 else C.index(X)) % 2 == 0
                         else "heldout"))
b2 = [it for it in b2 if it["n_scoreable"] >= 2]

# ------------------------------------------------------------------ battery 3
allrows = P1["battery3"] + P2["extra_twohop"]
tokzr = model.tokenizer
b3 = []
for i, r in enumerate(sorted(allrows, key=lambda z: z["name"])):
    if not (r["answer_rank"] <= 3 and r["answer_rank"] < r["alt_rank"]
            and r["src_tok"] is not None and r["tgt_tok"] is not None):
        continue
    word = r["swap_from"].strip().lower()
    low = r["prompt"].lower()
    ch = low.find(word)
    if ch < 0:
        continue
    # Walk the model's own token strings, which begin with the prepended
    # beginning-of-text token at position 0, and find the position whose
    # characters cover the first mention of the swapped concept.
    strtoks = model.to_str_tokens(r["prompt"])
    mention, cursor = None, 0
    for ti, st in enumerate(strtoks):
        if ti == 0:                   # the prepended beginning-of-text token
            continue
        if cursor <= ch < cursor + len(st):
            mention = ti
            break
        cursor += len(st)
    if mention is None:
        continue
    b3.append(dict(item_id=r["name"], prompt=r["prompt"],
                   clean_answer=r["answer"], target_answer=r["alt"],
                   clean_answer_tok=first_token_id(model, r["answer"]),
                   target_answer_tok=first_token_id(model, r["alt"]),
                   source=r["swap_from"], target=r["swap_to"],
                   source_tok=r["src_tok"], target_tok=r["tgt_tok"],
                   first_mention_pos=mention, clean_answer_rank=r["answer_rank"],
                   clean_alt_rank=r["alt_rank"], clean_argmax=r["argmax"],
                   clean_top5=r["top5"]))
for i, it in enumerate(b3):
    it["split"] = "tuning" if i % 2 == 0 else "heldout"

ONLY = os.environ.get("BUILD_ONLY")
for name, data in (("battery_h17.json", b1), ("battery_h17a.json", b2),
                   ("battery_h17b.json", b3)):
    if ONLY and ONLY not in name:
        continue
    json.dump(data, open(D + name, "w"), indent=1)
    print(name, len(data), "items;",
          "tuning", sum(1 for x in data if x["split"] == "tuning"),
          "heldout", sum(1 for x in data if x["split"] == "heldout"))
print("gated countries top3:", G3)
print("gated countries top5:", G5)
print("extension continent representatives:", C)
print("battery 2 primary pairs:", sum(1 for x in b2 if x["arm"] == "primary"),
      "extension pairs:", sum(1 for x in b2 if x["arm"] == "extension"),
      "all-three-scoreable:", sum(1 for x in b2 if x["n_scoreable"] == 3))
