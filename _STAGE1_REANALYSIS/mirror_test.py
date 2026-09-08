"""Is the convergence asymmetric, or do the two models mirror each other?

Established: a socialist state seeded into GPT-2 Medium's political window
decays and converges into Medium's Republican/news attractor by iteration 200.

Open: does the reverse hold. Seed GPT-2 Small with a Republican/news state and
run Small's native full-stack loop.

  falls to ` prolet`  -> the models mirror; each pulls the other's pole into its
                         own, and the difference is which pole each one owns
  holds Republican    -> Small has a news attractor too, and the registers are
                         co-existing rather than ordered
  goes somewhere else -> recorded flat

Control arm included: a neutral random seed, which should reach Small's known
attractor if the harness is faithful.
"""
import json, sys
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

SMALL, OUT = sys.argv[1], sys.argv[2]
MAX_ITER = int(sys.argv[3]) if len(sys.argv) > 3 else 500
SEQ = 12

model = GPT2LMHeadModel.from_pretrained(SMALL).eval()
tok = GPT2TokenizerFast.from_pretrained(SMALL)
ln_f, W_E, H = model.transformer.ln_f, model.transformer.wte.weight, model.transformer.h
NL, D = model.config.n_layer, model.config.n_embd
for p in model.parameters():
    p.requires_grad_(False)

SOC = [" prolet", " Anarch", " bourgeois", " Marx", " comrade", " proletarian", " socialist",
       " anarchist", " Lenin", " anarchism", " comrades", " labour", " communist",
       " socialism", " revolution", " workers", " solidarity", " union"]
RIV = [" Republican", " Democrat", " Trump", " GOP", " conservative", " Hillary",
       " Republicans", " Democrats", " presidential", " campaign", " federal", " Senate",
       " FBI", " Congress", " voters", " election"]
CTL = [" kitchen", " tomato", " bicycle", " weather", " hospital", " guitar",
       " forest", " sandwich", " ocean", " furniture", " camera", " blanket"]
sid = [tok.encode(w)[0] for w in SOC if len(tok.encode(w)) == 1]
rid = [tok.encode(w)[0] for w in RIV if len(tok.encode(w)) == 1]
cid = [tok.encode(w)[0] for w in CTL if len(tok.encode(w)) == 1]


def optimise(target_ids, steps=1200):
    v = torch.randn(D) * 0.5
    v.requires_grad_(True)
    opt = torch.optim.Adam([v], lr=0.05)
    t = torch.tensor(target_ids)
    for _ in range(steps):
        opt.zero_grad()
        loss = -torch.log_softmax(ln_f(v) @ W_E.T, -1)[t].mean()
        loss.backward(); opt.step()
    return v.detach()


def med_rank(vec, ids):
    with torch.no_grad():
        lg = ln_f(vec) @ W_E.T
    o = torch.argsort(lg, descending=True); p = torch.empty_like(o); p[o] = torch.arange(len(o))
    r = sorted(int(p[i]) + 1 for i in ids)
    return r[len(r) // 2]


def top_of(vec, k=8):
    with torch.no_grad():
        lg = ln_f(vec) @ W_E.T
    return [tok.decode([int(i)]) for i in torch.topk(lg, k).indices]


def full_step(x):
    h = x.unsqueeze(0)
    with torch.no_grad():
        for l in range(NL):
            o = H[l](h)
            h = o[0] if isinstance(o, tuple) else o
    return h[0]


torch.manual_seed(42)
seeds = {
    "republican": optimise(rid),
    "neutral_control": (torch.randn(D) * 0.5),
}
results = {"max_iter": MAX_ITER, "runs": {}}
for name, v in seeds.items():
    print(f"\n### seed {name}: {top_of(v, 10)}", flush=True)
    for shell_name, shell in [("x73", 1800.0), ("x150", 3700.0)]:
        x = v.unsqueeze(0).repeat(SEQ, 1)
        x = x / x.norm() * shell
        N0 = x.norm().item()
        trace = []
        for it in range(0, MAX_ITER + 1):
            if it in (0, 1, 2, 5, 10, 20, 50, 100, 200, 350, 500):
                trace.append({"iter": it, "soc": med_rank(x[-1], sid),
                              "riv": med_rank(x[-1], rid), "ctl": med_rank(x[-1], cid),
                              "top": top_of(x[-1], 6)})
            if it == MAX_ITER:
                break
            x = x * (N0 / x.norm())
            x = full_step(x)
        results["runs"][f"{name}/{shell_name}"] = trace
        print(f"[{name}/{shell_name}] N0={N0:.0f}")
        for t in trace:
            print(f"   it{t['iter']:>4}  soc {t['soc']:>6}  riv {t['riv']:>6}  ctl {t['ctl']:>6}   {t['top'][:5]}")
        print(flush=True)

json.dump(results, open(OUT, "w"), indent=1)
print("written", OUT)
