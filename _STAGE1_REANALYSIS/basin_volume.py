"""How reachable is the socialist basin from random states?

Control finding: one neutral random seed reached Divine (x73) / gaming (x150),
never socialism. n=1. This runs many random seeds to put a number on it.

Language prompts reach socialist 4/5. If random seeds reach it 0/N, the basin is
language-specific rather than a generic attractor of the weights.
"""
import json, sys, collections
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
SMALL, OUT = sys.argv[1], sys.argv[2]
N_SEEDS, MAX_ITER, SEQ = int(sys.argv[3]), 300, 12
m = GPT2LMHeadModel.from_pretrained(SMALL).eval()
tok = GPT2TokenizerFast.from_pretrained(SMALL)
ln_f, W_E, H = m.transformer.ln_f, m.transformer.wte.weight, m.transformer.h
NL, D = m.config.n_layer, m.config.n_embd

SIG = {  # signature tokens per known basin
  "socialist": [" prolet"," Anarch"," bourgeois"," Marx"," comrade"," proletarian"," socialist"],
  "divine":    [" Divine"," Fairy"," Holy"," Darkness"," Elements"],
  "gaming":    [" Zerg"," player"," tournament"," opponent"," Mana"," hero"],
}
SIGID = {k:[tok.encode(w)[0] for w in v if len(tok.encode(w))==1] for k,v in SIG.items()}

def step(x):
    h = x.unsqueeze(0)
    with torch.no_grad():
        for l in range(NL):
            o = H[l](h); h = o[0] if isinstance(o,tuple) else o
    return h[0]

def classify(v):
    with torch.no_grad(): lg = ln_f(v) @ W_E.T
    o=torch.argsort(lg,descending=True); p=torch.empty_like(o); p[o]=torch.arange(len(o))
    med = {k: sorted(int(p[i])+1 for i in ids)[len(ids)//2] for k,ids in SIGID.items()}
    best = min(med, key=med.get)
    top = [tok.decode([int(i)]) for i in torch.topk(lg,5).indices]
    return (best if med[best] < 500 else "other"), med, top

counts = collections.Counter(); rows=[]
for s in range(N_SEEDS):
    torch.manual_seed(1000+s)
    shell = [125.0*SEQ**0.5, 397.0, 900.0, 1800.0, 3700.0][s % 5]
    x = torch.randn(SEQ, D)*0.5
    x = x / x.norm() * shell; N0 = x.norm().item()
    for _ in range(MAX_ITER):
        x = x * (N0/x.norm()); x = step(x)
    lab, med, top = classify(x[-1])
    counts[lab]+=1; rows.append({"seed":s,"shell":shell,"basin":lab,"med":med,"top":top})
    print(f"seed {s:>3} shell {shell:.0f} -> {lab:10} {top}", flush=True)

print("\n=== basin counts over", N_SEEDS, "random seeds ===")
for k,v in counts.most_common(): print(f"   {k:10} {v:>3}/{N_SEEDS}")
json.dump({"counts":dict(counts),"rows":rows}, open(OUT,"w"), indent=1)
