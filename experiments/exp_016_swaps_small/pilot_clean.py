"""Clean-accuracy pilot for EXP_016. No swaps are performed here.

This script measures what base GPT-2 Small can already do without any
intervention, so that the three batteries can be written against items the
model can actually answer. Its output is the gate recorded in
`_STAGE2_JSPACE/EXP_016_SPEC.md`.
"""
import json, sys, time, torch
sys.path.insert(0, "/home/user/wt/exp016/experiments/exp_016_swaps_small")
from lib_exp016 import (load_model, load_lens, lens_vectors, lens_logits_at,
                        single_token_id, first_token_id, clean_topk, clean_run,
                        rank_of)

OUT = "/home/user/wt/exp016/experiments/exp_016_swaps_small/output/pilot_clean.json"
t_start = time.time()
model = load_model()
lens = load_lens()
print("lens source layers", lens.source_layers, "n_prompts", lens.n_prompts, flush=True)

# ---------------------------------------------------------------- battery 1
CATEGORIES = {
 "sport": [" football"," soccer"," tennis"," basketball"," baseball"," golf",
           " hockey"," cricket"," rugby"," boxing"," swimming"," running",
           " skiing"," wrestling"," cycling"," surfing"],
 "fruit": [" apple"," banana"," orange"," grape"," peach"," pear"," cherry",
           " lemon"," mango"," melon"," plum"," pineapple"," strawberry"],
 "colour":[" red"," blue"," green"," yellow"," black"," white"," orange",
           " purple"," pink"," brown"," grey"," gray"," gold"," silver"],
 "animal":[" dog"," cat"," horse"," lion"," tiger"," bear"," elephant"," wolf",
           " fox"," rabbit"," mouse"," bird"," fish"," snake"," cow"," sheep",
           " pig"," monkey"," deer"," dolphin"],
}
FRAMES = {
 "sport": ["My favourite sport is","The sport I like best is",
           "Q: Which sport do you like? A:","Her favourite sport is",
           "His favourite sport is","The sport I play most often is",
           "When I have free time, the sport I play is",
           "I told my friend that my favourite sport is",
           "My brother's favourite sport is","The best sport in the world is",
           "If I had to pick one sport, it would be",
           "At school, the sport I enjoyed most was"],
 "fruit": ["My favourite fruit is","The fruit I like best is",
           "Q: Which fruit do you like? A:","Her favourite fruit is",
           "His favourite fruit is","The fruit I eat every morning is",
           "In my lunchbox there is always a","The sweetest fruit is the",
           "The fruit I bought at the market was a",
           "My daughter's favourite fruit is"],
 "colour":["My favourite colour is","My favorite color is",
           "The colour I like best is","Q: What colour do you like? A:",
           "Her favourite colour is","His favorite color is",
           "She painted the wall a bright","The car she bought was",
           "My favourite colour has always been","The colour of her dress was"],
 "animal":["My favourite animal is","The animal I like best is",
           "Q: Which animal do you like? A:","Her favourite animal is",
           "His favourite animal is","At the zoo, the animal I wanted to see was the",
           "The strongest animal is the","My son's favourite animal is",
           "The animal on the farm was a","The pet I want most is a"],
}

tok_report = {}
cat_ids = {}
for cat, names in CATEGORIES.items():
    keep = {}
    for n in names:
        tid = single_token_id(model, n)
        if tid is not None:
            keep[n] = tid
    cat_ids[cat] = keep
    tok_report[cat] = {"kept": sorted(keep), "dropped": [n for n in names if n not in keep]}
    print(f"[tok] {cat}: kept {len(keep)}/{len(names)}; dropped {tok_report[cat]['dropped']}", flush=True)

READ_LAYER = 8
b1 = []
for cat, frames in FRAMES.items():
    ids = cat_ids[cat]
    id_list = [ids[n] for n in sorted(ids)]
    names = sorted(ids)
    for fr in frames:
        lp, cache, T = clean_run(model, fr)
        ll = lens_logits_at(lens, model, cache, READ_LAYER, -1)
        clean_rank_in_cat = {n: rank_of(lp, ids[n]) for n in names}
        lens_rank_in_cat = {n: int((ll > ll[ids[n]]).sum().item()) + 1 for n in names}
        top10 = torch.topk(lp, 10).indices.tolist()
        lens_top_cat = min(names, key=lambda n: lens_rank_in_cat[n])
        clean_top_cat = min(names, key=lambda n: clean_rank_in_cat[n])
        b1.append(dict(category=cat, frame=fr, n_tokens=T,
                       argmax=model.to_string(int(lp.argmax())),
                       lens_top_in_category=lens_top_cat,
                       clean_top_in_category=clean_top_cat,
                       clean_rank_of_lens_top=clean_rank_in_cat[lens_top_cat],
                       clean_top10=[model.to_string(t) for t in top10],
                       members_in_clean_top10=[n for n in names if ids[n] in top10],
                       lens_ranks={n: lens_rank_in_cat[n] for n in names},
                       clean_ranks={n: clean_rank_in_cat[n] for n in names}))
        print(f"[b1] {cat:6s} {fr[:44]:44s} argmax={b1[-1]['argmax']!r:14s} "
              f"lens_top={lens_top_cat:11s} clean_top={clean_top_cat:11s} "
              f"agree={lens_top_cat==clean_top_cat}", flush=True)

# ---------------------------------------------------------------- battery 2
COUNTRIES = {
 "France":  {"capital":" Paris","language":" French","continent":" Europe"},
 "Germany": {"capital":" Berlin","language":" German","continent":" Europe"},
 "Italy":   {"capital":" Rome","language":" Italian","continent":" Europe"},
 "Spain":   {"capital":" Madrid","language":" Spanish","continent":" Europe"},
 "Russia":  {"capital":" Moscow","language":" Russian","continent":" Europe"},
 "Poland":  {"capital":" Warsaw","language":" Polish","continent":" Europe"},
 "Greece":  {"capital":" Athens","language":" Greek","continent":" Europe"},
 "Ireland": {"capital":" Dublin","language":" English","continent":" Europe"},
 "Portugal":{"capital":" Lisbon","language":" Portuguese","continent":" Europe"},
 "Norway":  {"capital":" Oslo","language":" Norwegian","continent":" Europe"},
 "Sweden":  {"capital":" Stockholm","language":" Swedish","continent":" Europe"},
 "China":   {"capital":" Beijing","language":" Chinese","continent":" Asia"},
 "Japan":   {"capital":" Tokyo","language":" Japanese","continent":" Asia"},
 "India":   {"capital":" Delhi","language":" Hindi","continent":" Asia"},
 "Iran":    {"capital":" Tehran","language":" Persian","continent":" Asia"},
 "Israel":  {"capital":" Jerusalem","language":" Hebrew","continent":" Asia"},
 "Egypt":   {"capital":" Cairo","language":" Arabic","continent":" Africa"},
 "Kenya":   {"capital":" Nairobi","language":" Swahili","continent":" Africa"},
 "Nigeria": {"capital":" Abuja","language":" English","continent":" Africa"},
 "Canada":  {"capital":" Ottawa","language":" English","continent":" North"},
 "Mexico":  {"capital":" Mexico","language":" Spanish","continent":" North"},
 "Brazil":  {"capital":" Brasilia","language":" Portuguese","continent":" South"},
 "Chile":   {"capital":" Santiago","language":" Spanish","continent":" South"},
 "Australia":{"capital":" Canberra","language":" English","continent":" Australia"},
 "Turkey":  {"capital":" Ankara","language":" Turkish","continent":" Asia"},
}
FUNC_FRAMES = {
 "capital":  ["The capital of {X} is the city of","The capital of {X} is",
              "The capital city of {X} is","Q: What is the capital of {X}? A:"],
 "language": ["Most people in {X} speak","The language spoken in {X} is",
              "People in {X} speak","In {X} the people speak the language of"],
 "continent":["{X} is a country on the continent of","{X} is a country in",
              "{X} is located on the continent of",
              "Q: Which continent is {X} on? A:"],
}
country_tok = {c: single_token_id(model, " " + c) for c in COUNTRIES}
print("[tok] countries dropped (not single token):",
      [c for c, t in country_tok.items() if t is None], flush=True)

b2 = []
for func, frames in FUNC_FRAMES.items():
    for fi, fr in enumerate(frames):
        hits1 = hits3 = hits5 = n = 0
        per = {}
        for c, ans in COUNTRIES.items():
            if country_tok[c] is None:
                continue
            prompt = fr.replace("{X}", c)
            lp, _, _ = clean_run(model, prompt)
            aid = first_token_id(model, ans[func])
            r = rank_of(lp, aid)
            per[c] = dict(rank=r, argmax=model.to_string(int(lp.argmax())))
            n += 1; hits1 += r == 1; hits3 += r <= 3; hits5 += r <= 5
        b2.append(dict(func=func, frame=fr, n=n, top1=hits1, top3=hits3,
                       top5=hits5, per_country=per))
        print(f"[b2] {func:9s} v{fi} top1={hits1}/{n} top3={hits3}/{n} "
              f"top5={hits5}/{n}  {fr}", flush=True)

# ---------------------------------------------------------------- battery 3
TWOHOP = [
 ("spider-legs","Spiders have eight legs and ants have six legs. The animal that spins webs has"," eight"," six"," spider"," ant"),
 ("sport-ball","Football is played with a round ball and rugby is played with an oval ball. My favourite sport is football, so the ball I play with is"," round"," oval"," football"," rugby"),
 ("dog-sound","Dogs say woof and cats say meow. The animal that fetches sticks says"," woof"," meow"," dog"," cat"),
 ("lion-place","Lions live in Africa and tigers live in Asia. The big cat with a mane lives in"," Africa"," Asia"," lion"," tiger"),
 ("banana-colour","Apples are red and bananas are yellow. The fruit that monkeys love is"," yellow"," red"," banana"," apple"),
 ("sun-heat","The sun is hot and the moon is cold. The thing in the sky that gives us daylight is"," hot"," cold"," sun"," moon"),
 ("cow-product","Cows give milk and hens give eggs. The animal that moos gives"," milk"," eggs"," cow"," hen"),
 ("bird-action","Fish swim and birds fly. The animal with feathers can"," fly"," swim"," bird"," fish"),
 ("snow-colour","Snow is white and coal is black. The thing that falls from the sky in winter is"," white"," black"," snow"," coal"),
 ("bike-wheels","Cars have four wheels and bikes have two wheels. The vehicle that you pedal has"," two"," four"," bike"," car"),
 ("paris-country","Paris is in France and Rome is in Italy. The city with the Eiffel Tower is in"," France"," Italy"," Paris"," Rome"),
 ("winter-temp","Winter is cold and summer is hot. The season when it snows is"," cold"," hot"," winter"," summer"),
 ("doctor-place","Doctors work in hospitals and teachers work in schools. The person who treats sick patients works in"," hospitals"," schools"," doctor"," teacher"),
 ("bee-product","Bees make honey and cows make milk. The insect that lives in a hive makes"," honey"," milk"," bee"," cow"),
 ("gold-price","Gold is expensive and iron is cheap. The metal that wedding rings are made of is"," expensive"," cheap"," gold"," iron"),
 ("cat-feature","Cats have whiskers and birds have beaks. The animal that purrs has"," whiskers"," beaks"," cat"," bird"),
 ("train-track","Trains run on rails and boats run on water. The vehicle that pulls carriages runs on"," rails"," water"," train"," boat"),
 ("book-verb","Books are read and songs are sung. The thing with pages is"," read"," sung"," book"," song"),
 ("elephant-colour","Elephants are grey and swans are white. The animal with a trunk is"," grey"," white"," elephant"," swan"),
 ("fire-feel","Fire is hot and ice is cold. The thing that burns wood is"," hot"," cold"," fire"," ice"),
]
b3 = []
for name, prompt, ans, alt, src, tgt in TWOHOP:
    lp, _, T = clean_run(model, prompt)
    aid = first_token_id(model, ans); bid = first_token_id(model, alt)
    b3.append(dict(name=name, prompt=prompt, answer=ans, alt=alt,
                   swap_from=src, swap_to=tgt, n_tokens=T,
                   src_tok=single_token_id(model, src),
                   tgt_tok=single_token_id(model, tgt),
                   answer_rank=rank_of(lp, aid), alt_rank=rank_of(lp, bid),
                   argmax=model.to_string(int(lp.argmax())),
                   top5=[model.to_string(t) for t in torch.topk(lp,5).indices.tolist()]))
    print(f"[b3] {name:16s} ans_rank={b3[-1]['answer_rank']:5d} "
          f"alt_rank={b3[-1]['alt_rank']:5d} argmax={b3[-1]['argmax']!r:12s} "
          f"toks(src={b3[-1]['src_tok']},tgt={b3[-1]['tgt_tok']})", flush=True)

json.dump(dict(battery1=b1, battery2=b2, battery3=b3, tokens=tok_report,
               country_tok={c: t for c, t in country_tok.items()},
               read_layer=READ_LAYER,
               wall_seconds=round(time.time()-t_start, 1)),
          open(OUT, "w"), indent=1)
print("wrote", OUT, "in", round(time.time()-t_start,1), "s", flush=True)
