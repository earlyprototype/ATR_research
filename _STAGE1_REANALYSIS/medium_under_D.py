"""Is GPT-2 Medium's `D` collapse a mask over socialist-register structure?

The EXP_014 method (rank held-out vocabulary in the FULL readout distribution)
applied to Medium's committed A0 terminals instead of Small's basin.

`D` is the argmax. This asks what is at ranks 2..N underneath it.

Reproduction gate first: the readout rebuilt here must return `D` as argmax for
all 25 A0 terminals before any rank is reported.
"""
import json, statistics, sys, collections
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

MED = sys.argv[1]
TERMINALS = sys.argv[2]
OUT = "/home/user/ATR_research/_STAGE1_REANALYSIS/medium_under_D.json"

# Sets held fixed before any number was computed (mirrors POLARISATION_SPEC §3),
# plus the Small basin itself, which is the thing we are asking Medium for.
SETS = {
    "SMALL_BASIN": [" prolet", " Anarch", " bourgeois", " Marx", " comrade", " proletarian",
                    " socialist", " anarchist", " Lenin", " anarchism", " comrades", " labour"],
    "L_heldout": [" communist", " communism", " socialism", " Marxism", " Trotsky", " Engels",
                  " revolution", " workers", " strike", " solidarity", " union", " collective"],
    "R_rival": [" capitalist", " capitalism", " conservative", " conservatism", " fascist",
                " fascism", " nationalist", " nationalism", " libertarian", " monarchy",
                " Reagan", " Thatcher"],
    "N_political": [" parliament", " election", " senate", " mayor", " ballot", " legislation",
                    " governor", " referendum", " candidate", " voter", " committee", " constitution"],
    "C_control": [" kitchen", " tomato", " bicycle", " weather", " hospital", " guitar",
                  " forest", " sandwich", " ocean", " furniture", " camera", " blanket"],
}

model = GPT2LMHeadModel.from_pretrained(MED).eval()
tok = GPT2TokenizerFast.from_pretrained(MED)
ln_f, W_E = model.transformer.ln_f, model.transformer.wte.weight
print(f"loaded: n_layer={model.config.n_layer} d_model={model.config.n_embd}")

resolved, dropped = {}, {}
for name, toks in SETS.items():
    keep, drop = {}, []
    for t in toks:
        ids = tok.encode(t)
        (keep.update({t: ids[0]}) if len(ids) == 1 else drop.append(t))
    resolved[name], dropped[name] = keep, drop

term = torch.load(TERMINALS, map_location="cpu", weights_only=False)
a0 = {k: v for k, v in term.items() if k.split("|")[0] == "A0"}
print(f"A0 terminals: {len(a0)}")


def logits_of(vec):
    with torch.no_grad():
        return ln_f(vec) @ W_E.T


def ranks(lg, ids):
    order = torch.argsort(lg, descending=True)
    pos = torch.empty_like(order)
    pos[order] = torch.arange(len(order))
    return {i: int(pos[i]) + 1 for i in ids}


# ---- reproduction gate ----
gate = collections.Counter()
vecs = {}
for k, v in a0.items():
    x = v["mean"] if isinstance(v, dict) and "mean" in v else v
    if x.dim() > 1:
        x = x.mean(dim=0)
    vecs[k] = x
    gate[tok.decode([int(logits_of(x).argmax())])] += 1
print("GATE argmax distribution:", dict(gate))
gate_pass = gate.most_common(1)[0][0].strip() == "D" and gate.most_common(1)[0][1] == len(vecs)
print("GATE:", "PASS" if gate_pass else "FAIL")

results = {"gate": dict(gate), "gate_pass": bool(gate_pass), "dropped": dropped, "per_prompt": {}}

agg = collections.defaultdict(list)
for k, x in vecs.items():
    lg = logits_of(x)
    entry = {}
    for sname, mapping in resolved.items():
        r = ranks(lg, list(mapping.values()))
        per = {t: r[i] for t, i in mapping.items()}
        med = statistics.median(per.values())
        entry[sname] = {"median_rank": med, "ranks": per}
        agg[sname].append(med)
    # what IS near the top, under D
    top = torch.topk(lg, 30).indices.tolist()
    entry["top30"] = [tok.decode([i]) for i in top]
    results["per_prompt"][k] = entry

results["aggregate_median_of_medians"] = {k: statistics.median(v) for k, v in agg.items()}
json.dump(results, open(OUT, "w"), indent=1)

print("\n=== median rank (of 50257), aggregated over 25 A0 terminals ===")
for k, v in results["aggregate_median_of_medians"].items():
    print(f"  {k:14} {v:9.0f}")
print("\n=== what is actually under D (top-30, first terminal) ===")
print(results["per_prompt"][sorted(vecs)[0]]["top30"])
