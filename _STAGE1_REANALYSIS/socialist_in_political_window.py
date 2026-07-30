"""Can a socialist state survive inside Medium's OWN political region?

Established so far tonight:
  - full-stack (0->23) seeded socialist decays to `D` by iteration 20
  - Medium has a robust political attractor at inject 5-7 -> extract 18-20,
    resolving to ` Republican`, via-tail 25/25
  - socialist vocabulary sits at rank ~17k-23k inside those political states

Open question: the full stack is not the only dynamics. If Medium can hold a
socialist state anywhere, the political window is the most likely place. This
seeds the optimised socialist state into the political windows and iterates.
"""
import json, sys
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

MED, OUT = sys.argv[1], sys.argv[2]
MAX_ITER = int(sys.argv[3]) if len(sys.argv) > 3 else 300
SEQ = 12
WINDOWS = [(6, 18), (6, 19), (5, 20), (7, 17), (0, 23)]

model = GPT2LMHeadModel.from_pretrained(MED).eval()
tok = GPT2TokenizerFast.from_pretrained(MED)
ln_f, W_E, H = model.transformer.ln_f, model.transformer.wte.weight, model.transformer.h
NL = model.config.n_layer
for p in model.parameters():
    p.requires_grad_(False)

BASIN = [" prolet", " Anarch", " bourgeois", " Marx", " comrade", " proletarian", " socialist",
         " anarchist", " Lenin", " anarchism", " comrades", " labour", " communist",
         " socialism", " revolution", " workers", " solidarity", " union"]
RIVAL = [" Republican", " Democrat", " Trump", " GOP", " conservative", " Hillary",
         " Republicans", " Democrats", " presidential", " campaign", " federal", " Senate"]
CONTROL = [" kitchen", " tomato", " bicycle", " weather", " hospital", " guitar",
           " forest", " sandwich", " ocean", " furniture", " camera", " blanket"]
sid = [tok.encode(w)[0] for w in BASIN if len(tok.encode(w)) == 1]
rid = [tok.encode(w)[0] for w in RIVAL if len(tok.encode(w)) == 1]
cid = [tok.encode(w)[0] for w in CONTROL if len(tok.encode(w)) == 1]

v = torch.randn(1024) * 0.5
v.requires_grad_(True)
opt = torch.optim.Adam([v], lr=0.05)
tgt = torch.tensor(sid)
for _ in range(1200):
    opt.zero_grad()
    loss = -torch.log_softmax(ln_f(v) @ W_E.T, -1)[tgt].mean()
    loss.backward(); opt.step()
v = v.detach()


def med_rank(vec, ids):
    with torch.no_grad():
        lg = ln_f(vec) @ W_E.T
    o = torch.argsort(lg, descending=True); p = torch.empty_like(o); p[o] = torch.arange(len(o))
    r = sorted(int(p[i]) + 1 for i in ids)
    return r[len(r) // 2]


def top_of(vec, k=10):
    with torch.no_grad():
        lg = ln_f(vec) @ W_E.T
    return [tok.decode([int(i)]) for i in torch.topk(lg, k).indices]


def window_step(x, i, j):
    """inject at block i input, run to block j output"""
    h = x.unsqueeze(0)
    with torch.no_grad():
        for l in range(i, j + 1):
            o = H[l](h)
            h = o[0] if isinstance(o, tuple) else o
    return h[0]


results = {"windows": {}}
print("seed top10:", top_of(v, 10), flush=True)

for (i, j) in WINDOWS:
    for shell_name, mult in [("lo", 1.0), ("hi", 20.0)]:
        x = v.unsqueeze(0).repeat(SEQ, 1)
        x = x / x.norm() * (300.0 * mult)
        N0 = x.norm().item()
        trace = []
        for it in range(0, MAX_ITER + 1):
            if it in (0, 1, 2, 5, 10, 20, 50, 100, 200, 300):
                trace.append({"iter": it, "soc": med_rank(x[-1], sid),
                              "riv": med_rank(x[-1], rid), "ctl": med_rank(x[-1], cid),
                              "top": top_of(x[-1], 6)})
            if it == MAX_ITER:
                break
            x = x * (N0 / x.norm())
            x = window_step(x, i, j)
        key = f"{i}->{j}/{shell_name}"
        results["windows"][key] = trace
        print(f"\n[{key}] N0={N0:.0f}")
        for t in trace:
            print(f"   it{t['iter']:>4}  soc {t['soc']:>6}  riv {t['riv']:>6}  ctl {t['ctl']:>6}   {t['top'][:5]}")
        print(flush=True)

json.dump(results, open(OUT, "w"), indent=1)
print("written", OUT)
