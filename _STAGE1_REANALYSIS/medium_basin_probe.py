"""Does GPT-2 Medium have a socialist fixed point at all?

Two routes are already closed: no window in the 277-arm census surfaces the
register, and nothing socialist sits under the `D` readout (median rank 39202).

Both of those ask whether the loop FINDS the basin from a language start.
This asks a different question: does the basin EXIST. Seed the loop directly
at a socialist state and iterate.

  (a) it holds                -> the basin exists, it is merely unreachable
  (b) it falls to `D`         -> no socialist basin; D is the only attractor there
  (c) it goes somewhere third -> recorded flat

Loop is the registered ATR protocol: inject at blocks.0 input, extract
blocks.23 output, rescale to the seed norm each pass, gate on cos > 0.999 x3.
"""
import json, sys, collections
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

MED = sys.argv[1]
OUT = sys.argv[2]
MAX_ITER = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
SEQ = 12

model = GPT2LMHeadModel.from_pretrained(MED).eval()
tok = GPT2TokenizerFast.from_pretrained(MED)
ln_f, W_E = model.transformer.ln_f, model.transformer.wte.weight
NL = model.config.n_layer

BASIN = [" prolet", " Anarch", " bourgeois", " Marx", " comrade", " proletarian",
         " socialist", " anarchist", " Lenin", " anarchism", " comrades", " labour",
         " communist", " socialism", " revolution", " workers", " solidarity", " union"]
CONTROL = [" kitchen", " tomato", " bicycle", " weather", " hospital", " guitar",
           " forest", " sandwich", " ocean", " furniture", " camera", " blanket"]

_cap = {}
def _grab(m, i, o):
    _cap["x"] = (o[0] if isinstance(o, tuple) else o).detach()
h_out = model.transformer.h[NL - 1].register_forward_hook(_grab)

_inj = {}
def _pre(m, args, kwargs):
    if _inj.get("x") is not None:
        new = _inj["x"].unsqueeze(0)
        if args:
            return (new,) + tuple(args[1:]), kwargs
        kwargs = dict(kwargs); kwargs["hidden_states"] = new
        return args, kwargs
    return None
h_in = model.transformer.h[0].register_forward_pre_hook(_pre, with_kwargs=True)

scaffold = torch.full((1, SEQ), 262)


def step(x):
    _inj["x"] = x
    with torch.no_grad():
        model(input_ids=scaffold)
    _inj["x"] = None
    return _cap["x"][0].clone()


def readout(v, k=25):
    with torch.no_grad():
        lg = ln_f(v) @ W_E.T
    top = torch.topk(lg, k)
    order = torch.argsort(lg, descending=True)
    pos = torch.empty_like(order); pos[order] = torch.arange(len(order))
    return ([tok.decode([int(i)]) for i in top.indices],
            lambda ids: [int(pos[i]) + 1 for i in ids])


def med_rank(v, words):
    with torch.no_grad():
        lg = ln_f(v) @ W_E.T
    order = torch.argsort(lg, descending=True)
    pos = torch.empty_like(order); pos[order] = torch.arange(len(order))
    ids = [tok.encode(w)[0] for w in words if len(tok.encode(w)) == 1]
    r = sorted(int(pos[i]) + 1 for i in ids)
    return r[len(r) // 2]


# ---- seed: a state whose readout IS the socialist register ----
ids = [tok.encode(w)[0] for w in BASIN if len(tok.encode(w)) == 1]
seed_dir = W_E[ids].mean(0)
seed_dir = seed_dir / seed_dir.norm()

results = {"seq": SEQ, "max_iter": MAX_ITER, "basin_tokens_resolved": len(ids), "runs": {}}

# norm shell: the registered convention runs i=0 far above natural. sweep it.
for shell_name, shell in [("natural_x1", 30.0), ("x20", 600.0), ("x73", 2200.0), ("x218", 6500.0)]:
    x = (seed_dir.unsqueeze(0).repeat(SEQ, 1) * shell)
    N0 = x.norm().item()
    seed_top, _ = readout(x[-1])
    hist, prev, locked = [], None, None
    for it in range(1, MAX_ITER + 1):
        x = x * (N0 / x.norm())
        nxt = step(x)
        if prev is not None and it % 10 == 0 and it >= 20:
            c = torch.nn.functional.cosine_similarity(
                nxt[-1].unsqueeze(0), prev[-1].unsqueeze(0)).item()
            hist.append((it, c))
            if c > 0.999 and locked is None and len(hist) >= 3 and all(h[1] > 0.999 for h in hist[-3:]):
                locked = it
                prev = nxt
                break
        prev = nxt
        x = nxt
    final = prev[-1]
    top, _ = readout(final)
    results["runs"][shell_name] = {
        "shell_norm": N0,
        "locked_at": locked,
        "seed_top10": seed_top[:10],
        "final_top25": top,
        "final_basin_median_rank": med_rank(final, BASIN),
        "final_control_median_rank": med_rank(final, CONTROL),
    }
    print(f"[{shell_name}] N0={N0:.0f} lock={locked}")
    print(f"   seed  -> {seed_top[:8]}")
    print(f"   final -> {top[:12]}")
    print(f"   basin median rank {results['runs'][shell_name]['final_basin_median_rank']}"
          f"  control {results['runs'][shell_name]['final_control_median_rank']}", flush=True)

h_out.remove(); h_in.remove()
json.dump(results, open(OUT, "w"), indent=1)
print("written", OUT)
