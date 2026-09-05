"""Re-run a small set of EXP_016 conditions to record the actual words the
model produces before and after a swap, so that a reader without a
machine-learning background can see what changed."""
from __future__ import annotations
import json, sys, torch
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_exp016 import load_model, load_lens, lens_vectors, positions
from swap_engine import SwapPlan, random_pair, run_plan

D = os.path.dirname(os.path.abspath(__file__)) + "/"
LAYERS = list(range(3, 11))
model = load_model(); lens = load_lens()


def top5(v):
    return [model.to_string(int(i)) for i in torch.topk(v, 5).indices]


def one(prompt, s_tok, t_tok, layer_set, alpha, mode, mention=1, seed=11):
    toks = model.to_tokens(prompt); T = toks.shape[1]
    pos = positions(mode, T, mention)
    V = {l: lens_vectors(lens, model, l, [s_tok, t_tok]) for l in layer_set}
    R = {l: random_pair(V[l][:, 0], V[l][:, 1], seed * 100 + l) for l in layer_set}
    p = SwapPlan(T, LAYERS)
    p.add(V, V, alpha, pos, False)          # lens
    p.add(R, V, alpha, pos, False)          # control A
    p.add(R, V, alpha, pos, True)           # control B
    p.build()
    lp = run_plan(model, toks, p)
    with torch.no_grad():
        clean = torch.log_softmax(model(toks)[0, -1].float(), dim=-1)
    return dict(prompt=prompt, layer_set=list(layer_set), alpha=alpha,
                posmode=mode, clean=top5(clean), lens=top5(lp[0]),
                control_a=top5(lp[1]), control_b=top5(lp[2]),
                source=model.to_string(s_tok), target=model.to_string(t_tok))


if __name__ == "__main__":
    spec = json.load(open(sys.argv[1]))
    out = []
    for job in spec:
        r = one(**job)
        out.append(r)
        print(f"{r['prompt'][:52]!r}\n  swap {r['source']!r} -> {r['target']!r} "
              f"at layers {r['layer_set']} strength {r['alpha']} ({r['posmode']})"
              f"\n  clean     {r['clean']}\n  lens swap {r['lens']}"
              f"\n  control A {r['control_a']}\n  control B {r['control_b']}\n",
              flush=True)
    json.dump(out, open(D + "output/qualitative.json", "w"), indent=1)
