"""EXP_014 — polarisation of the settled ATR basin (spec: POLARISATION_SPEC.md).

Reads the committed Stage 1 confidence artifacts and measures, in the FULL
50257-way readout distribution, the median rank of four pre-registered token
sets: held-out socialist register (L), rival pole (R), political-neutral (N),
non-political control (C).

Runs a reproduction gate first: the readout pipeline here must reproduce the
committed top-20 for both the settled state and the iteration-0 baseline
before any rank is reported.

Zero ATR loop iterations. One forward pass per prompt to rebuild iter-0.
"""
import json, os, statistics, sys
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

GPT2_DIR = sys.argv[1]
LUCIER = sys.argv[2]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "polarisation_results.json")
CONF = os.path.join(LUCIER, "experiments/gpt2_small/output_confidence")

# ---- token sets, exactly as pre-registered in POLARISATION_SPEC.md §3 ----
SETS = {
    "L_heldout_socialist": [" communist", " communism", " socialism", " Marxism",
                            " Trotsky", " Engels", " revolution", " workers",
                            " strike", " solidarity", " union", " collective"],
    "R_rival_pole": [" capitalist", " capitalism", " conservative", " conservatism",
                     " fascist", " fascism", " nationalist", " nationalism",
                     " libertarian", " monarchy", " Reagan", " Thatcher"],
    "N_political_neutral": [" parliament", " election", " senate", " mayor", " ballot",
                            " legislation", " governor", " referendum", " candidate",
                            " voter", " committee", " constitution"],
    "C_nonpolitical": [" kitchen", " tomato", " bicycle", " weather", " hospital",
                       " guitar", " forest", " sandwich", " ocean", " furniture",
                       " camera", " blanket"],
}

PROMPTS = {
    "Lucier": "Am I sitting in a room different from the one you are in now",
    "Semantic": "The Eiffel Tower is located in the city of",
    "Syntactic": "The cat sat on the mat and then the",
    "Nonsense": "Flurb glex morp wintly skade",
    "Imperative": "Calculate the sum of all prime numbers below",
}

model = GPT2LMHeadModel.from_pretrained(GPT2_DIR).eval()
tok = GPT2TokenizerFast.from_pretrained(GPT2_DIR)
ln_f, W_E = model.transformer.ln_f, model.transformer.wte.weight


def logits_of(vec):
    """Readout convention from 04_readout_confidence.py: ln_final then @ W_U."""
    with torch.no_grad():
        return ln_f(vec) @ W_E.T


def ranks_of(logits, ids):
    order = torch.argsort(logits, descending=True)
    pos = torch.empty_like(order)
    pos[order] = torch.arange(len(order))
    return {i: int(pos[i]) + 1 for i in ids}


# ---- resolve token sets to single-token ids; drop non-single, record it ----
resolved, dropped = {}, {}
for name, toks in SETS.items():
    keep, drop = {}, []
    for t in toks:
        ids = tok.encode(t)
        (keep.update({t: ids[0]}) if len(ids) == 1 else drop.append(t))
    resolved[name], dropped[name] = keep, drop

# ---- rebuild iteration-0 states (natural forward pass, resid_post final block) ----
iter0 = {}
cap = {}


def _grab(m, i, o):
    x = o[0] if isinstance(o, tuple) else o
    cap["x"] = x.detach()


h = model.transformer.h[-1].register_forward_hook(_grab)
for label, text in PROMPTS.items():
    # TransformerLens to_tokens prepends BOS for GPT-2 (cfg.default_prepend_bos);
    # the HF tokenizer does not. Match the engine, or iter-0 is a different state.
    ids = torch.tensor([[tok.bos_token_id] + tok.encode(text)])
    with torch.no_grad():
        model(input_ids=ids)
    x = cap["x"]
    iter0[label] = (x[0, -1, :] if x.dim() == 3 else x[-1, :]).clone()
h.remove()

settled = torch.load(os.path.join(CONF, "converged_tensors.pt"),
                     map_location="cpu", weights_only=True)
committed = json.load(open(os.path.join(CONF, "confidence_results.json")))

# ---- REPRODUCTION GATE ----
gate = {}
for label in PROMPTS:
    a = ranks_of(logits_of(settled[label][-1, :]), [])  # noqa - force compute path
    top_settled = torch.topk(logits_of(settled[label][-1, :]), 20).indices.tolist()
    top_iter0 = torch.topk(logits_of(iter0[label]), 20).indices.tolist()
    ref = committed["prompts"][label]
    gate[label] = {
        "settled_match": top_settled == ref["final_last_vector"]["top_token_ids"],
        "iter0_match": top_iter0 == ref["baseline_iter0"]["top_token_ids"],
    }
print("REPRODUCTION GATE:", json.dumps(gate, indent=1))
if not all(v["settled_match"] and v["iter0_match"] for v in gate.values()):
    print("GATE FAILED — ranks NOT reported.")
    json.dump({"gate": gate, "status": "GATE_FAILED"}, open(OUT, "w"), indent=1)
    raise SystemExit(1)

# ---- measurement ----
results = {"gate": gate, "dropped_tokens": dropped, "states": {}}
for label in PROMPTS:
    for phase, vec in (("iter0", iter0[label]), ("settled", settled[label][-1, :])):
        lg = logits_of(vec)
        entry = {}
        for sname, mapping in resolved.items():
            r = ranks_of(lg, list(mapping.values()))
            per = {t: r[i] for t, i in mapping.items()}
            entry[sname] = {"median_rank": statistics.median(per.values()),
                            "n": len(per), "ranks": per}
        results["states"][f"{label}/{phase}"] = entry

json.dump(results, open(OUT, "w"), indent=1)

# ---- report ----
print(f"\n{'state':<22}" + "".join(f"{s.split('_')[0]:>10}" for s in SETS))
print("-" * 62)
for k, v in results["states"].items():
    print(f"{k:<22}" + "".join(f"{v[s]['median_rank']:>10.0f}" for s in SETS))
print("\ndropped (not single tokens):",
      {k: v for k, v in dropped.items() if v})
