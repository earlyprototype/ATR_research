"""Decisive version of the basin probe.

The first probe seeded from the mean socialist embedding, whose own readout was
only partly socialist (` prolet` at rank 5). This optimises a residual vector so
its readout IS the register, cleanly, then iterates the registered ATR loop and
traces how fast it decays.

If a purpose-built socialist state cannot survive its own dynamics, Medium has
no socialist fixed point, and the question is closed rather than unexplored.
"""
import json, sys
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

MED, OUT = sys.argv[1], sys.argv[2]
MAX_ITER = int(sys.argv[3]) if len(sys.argv) > 3 else 300
SEQ = 12

model = GPT2LMHeadModel.from_pretrained(MED).eval()
tok = GPT2TokenizerFast.from_pretrained(MED)
ln_f, W_E = model.transformer.ln_f, model.transformer.wte.weight
NL = model.config.n_layer
for p in model.parameters():
    p.requires_grad_(False)

BASIN = [" prolet", " Anarch", " bourgeois", " Marx", " comrade", " proletarian",
         " socialist", " anarchist", " Lenin", " anarchism", " comrades", " labour",
         " communist", " socialism", " revolution", " workers", " solidarity", " union"]
CONTROL = [" kitchen", " tomato", " bicycle", " weather", " hospital", " guitar",
           " forest", " sandwich", " ocean", " furniture", " camera", " blanket"]
bid = [tok.encode(w)[0] for w in BASIN if len(tok.encode(w)) == 1]
cid = [tok.encode(w)[0] for w in CONTROL if len(tok.encode(w)) == 1]

# ---- optimise a residual state whose readout is the register ----
v = torch.randn(1024) * 0.5
v.requires_grad_(True)
opt = torch.optim.Adam([v], lr=0.05)
tgt = torch.tensor(bid)
for step_i in range(1200):
    opt.zero_grad()
    lg = ln_f(v) @ W_E.T
    loss = -torch.log_softmax(lg, -1)[tgt].mean()
    loss.backward()
    opt.step()
v = v.detach()

with torch.no_grad():
    lg = ln_f(v) @ W_E.T
top = torch.topk(lg, 20)
seed_top = [tok.decode([int(i)]) for i in top.indices]
print("optimised seed readout:", seed_top[:14])
in_reg = sum(1 for t in seed_top[:12] if t in BASIN)
print(f"socialist tokens in seed top-12: {in_reg}/12", flush=True)

# ---- ATR loop ----
_cap = {}
h_out = model.transformer.h[NL - 1].register_forward_hook(
    lambda m, i, o: _cap.__setitem__("x", (o[0] if isinstance(o, tuple) else o).detach()))
_inj = {}
def _pre(m, args, kwargs):
    if _inj.get("x") is None:
        return None
    new = _inj["x"].unsqueeze(0)
    if args:
        return (new,) + tuple(args[1:]), kwargs
    kwargs = dict(kwargs); kwargs["hidden_states"] = new
    return args, kwargs
h_in = model.transformer.h[0].register_forward_pre_hook(_pre, with_kwargs=True)
scaffold = torch.full((1, SEQ), 262)

def ranks_of(vec, ids):
    with torch.no_grad():
        l = ln_f(vec) @ W_E.T
    order = torch.argsort(l, descending=True)
    pos = torch.empty_like(order); pos[order] = torch.arange(len(order))
    r = sorted(int(pos[i]) + 1 for i in ids)
    return r[len(r) // 2]

def top_of(vec, k=12):
    with torch.no_grad():
        l = ln_f(vec) @ W_E.T
    return [tok.decode([int(i)]) for i in torch.topk(l, k).indices]

results = {"seed_top20": seed_top, "seed_socialist_in_top12": in_reg, "runs": {}}
for name, shell in [("x73", 2200.0), ("x218", 6500.0), ("natural", 30.0)]:
    x = v.unsqueeze(0).repeat(SEQ, 1)
    x = x / x.norm() * (shell * (SEQ ** 0.5) * 0.5)
    N0 = x.norm().item()
    trace = [{"iter": 0, "basin": ranks_of(x[-1], bid), "control": ranks_of(x[-1], cid),
              "top": top_of(x[-1], 8)}]
    for it in range(1, MAX_ITER + 1):
        x = x * (N0 / x.norm())
        _inj["x"] = x
        with torch.no_grad():
            model(input_ids=scaffold)
        _inj["x"] = None
        x = _cap["x"][0].clone()
        if it in (1, 2, 3, 5, 10, 20, 40, 80, 160, 300):
            trace.append({"iter": it, "basin": ranks_of(x[-1], bid),
                          "control": ranks_of(x[-1], cid), "top": top_of(x[-1], 8)})
    results["runs"][name] = {"shell": N0, "trace": trace}
    print(f"\n[{name}] shell={N0:.0f}")
    for t in trace:
        print(f"   iter {t['iter']:>3}  basin_rank {t['basin']:>6}  control {t['control']:>6}   {t['top'][:6]}")
    print(flush=True)

h_out.remove(); h_in.remove()
json.dump(results, open(OUT, "w"), indent=1)
print("written", OUT)
