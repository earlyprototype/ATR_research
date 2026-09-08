"""Does Medium's ` Republican` attractor survive natural-energy injection?

This is the control that destroyed `D`. EXP_010c-VARIANTS Control B re-ran the
i=0 arms with rescaling to the natural `resid_pre` norm at the injection layer
instead of the seed norm, and `D` vanished entirely: 0/25 converged, `D` absent,
arm class changed.

The ` Republican` cells (6→18, 6→19) are reported in RESULTS_MEDIUM_SOCIALIST.md
as via-tail robust 25/25. That finding has the same untested weakness `D` had
until Control B ran. This runs it.

Prior expectation, stated before the run: mid-band injection sits near natural
norm already (the parent record measures ≈1× for i∈{8,10} against ≈218× at i=0),
so `Republican` should survive where `D` did not. If it does not survive, the
headline positive in that record is energy-conditional and must be relabelled.

Protocol: registered, except max_iter 300 rather than 1000 (deviation, recorded).
Gate cos > 0.999 ×3, checks every 10 from 100. L0 natural-pass seeding.
"""
import json, sys, collections
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

MED, OUT = sys.argv[1], sys.argv[2]
MAX_ITER = 300
WINDOWS = [(6, 18), (6, 19), (7, 17)]

model = GPT2LMHeadModel.from_pretrained(MED).eval()
tok = GPT2TokenizerFast.from_pretrained(MED)
ln_f, W_E, H = model.transformer.ln_f, model.transformer.wte.weight, model.transformer.h
NL = model.config.n_layer

prompts = json.load(open("/home/user/ATR_research/_STAGE2_JSPACE/experiments/"
                         "exp_010c_windows/output/prompt_subset.json"))
print(f"{len(prompts)} prompts, {NL} layers", flush=True)

_cap = {}
hooks = []
for l in range(NL):
    hooks.append(H[l].register_forward_pre_hook(
        lambda m, a, _l=l: _cap.__setitem__(f"pre{_l}", a[0].detach())))
    hooks.append(H[l].register_forward_hook(
        lambda m, i, o, _l=l: _cap.__setitem__(
            f"post{_l}", (o[0] if isinstance(o, tuple) else o).detach())))


def natural_pass(text):
    ids = torch.tensor([[tok.bos_token_id] + tok.encode(text)])
    with torch.no_grad():
        model(input_ids=ids)
    return {k: v.clone() for k, v in _cap.items()}


def window_forward(x, i, j):
    h = x.unsqueeze(0)
    with torch.no_grad():
        for l in range(i, j + 1):
            o = H[l](h)
            h = o[0] if isinstance(o, tuple) else o
    return h[0]


def decode(v):
    with torch.no_grad():
        return tok.decode([int((ln_f(v) @ W_E.T).argmax())])


results = {"max_iter": MAX_ITER, "deviation": "max_iter 300 not 1000", "arms": {}}
for (i, j) in WINDOWS:
    for conv in ("seed_j", "natural_i"):
        terms, conv_n, ratios = collections.Counter(), 0, []
        for rec in prompts:
            nat = natural_pass(rec["prompt"])
            x0 = nat[f"post{j}"][0]
            nat_i_norm = nat[f"pre{i}"][0].norm().item()
            target = x0.norm().item() if conv == "seed_j" else nat_i_norm
            ratios.append(x0.norm().item() / nat_i_norm)
            x, prev, locked, hist = x0.clone(), None, None, []
            for it in range(1, MAX_ITER + 1):
                x = x * (target / x.norm())
                nxt = window_forward(x, i, j)
                if it >= 100 and it % 10 == 0 and prev is not None:
                    c = torch.nn.functional.cosine_similarity(
                        nxt[-1].unsqueeze(0), prev[-1].unsqueeze(0)).item()
                    hist.append(c)
                    if len(hist) >= 3 and all(h > 0.999 for h in hist[-3:]):
                        locked = it
                        prev = nxt
                        break
                prev, x = nxt, nxt
            terms[decode(prev[-1])] += 1
            conv_n += locked is not None
        key = f"{i}->{j}/{conv}"
        results["arms"][key] = {
            "converged": f"{conv_n}/{len(prompts)}",
            "terminals": terms.most_common(6),
            "seed_over_natural_ratio_mean": sum(ratios) / len(ratios),
        }
        print(f"[{key}] conv {conv_n}/{len(prompts)}  "
              f"seed/natural ratio {sum(ratios)/len(ratios):.2f}x")
        print(f"    {terms.most_common(6)}", flush=True)

for h in hooks:
    h.remove()
json.dump(results, open(OUT, "w"), indent=1)
print("written", OUT)
