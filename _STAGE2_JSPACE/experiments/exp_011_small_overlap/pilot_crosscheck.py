"""Cross-check the EXP_011 decomposition code against the pilot's published numbers.

The pilot (05_jlens_pilot.py) probed the layer-11 terminal vector against every
layer's 193-atom dictionary with a projected-gradient sparse solver. Running the
EXP_011 non-negative orthogonal matching pursuit on the identical dictionary and
the identical states should land in the same range and preserve the same ordering.
"""
import os, torch, json, sys, numpy as np
torch.set_num_threads(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jspace import decompose
P="/home/user/lucier-gpt2-activ-tensor-reson-experiments/experiments/gpt2_small/"
jl=torch.load(P+"output_jlens_pilot/jlens_vectors.pt", weights_only=False, map_location="cpu")
ref=json.load(open(P+"output_jlens_pilot/jlens_pilot_results.json"))
conv=torch.load(P+"output_confidence/converged_tensors.pt", weights_only=True)
V=jl["vectors"]           # [193, 12, 768]
rows=[]
for st in ("Semantic","Syntactic","Lucier","Nonsense","Imperative"):
    h = conv[st][-1]
    for l in (6, 9, 11):
        D = V[:, l, :]
        s = decompose(D, h.unsqueeze(0))
        pub = ref["per_state"][st]["per_layer"][l]["nn_sparse_k25_share"]
        rows.append((st, l, float(s["share"][0]), pub, int(s["n_atoms"][0])))
print(f"{'state':11s} {'L':>3s} {'EXP_011 NN-OMP':>15s} {'pilot published':>16s} {'atoms':>6s}")
for st,l,mine,pub,na in rows:
    print(f"{st:11s} {l:3d} {mine:15.4f} {pub:16.4f} {na:6d}")
# ordering check: Divine (Syntactic) vs prolet (Semantic) at each layer, both solvers
print("\nordering (pilot reported Divine >= prolet at 11 of 12 layers):")
agree=0
for l in range(12):
    D=V[:,l,:]
    a=float(decompose(D, conv["Syntactic"][-1].unsqueeze(0))["share"][0])
    b=float(decompose(D, conv["Semantic"][-1].unsqueeze(0))["share"][0])
    pa=ref["per_state"]["Syntactic"]["per_layer"][l]["nn_sparse_k25_share"]
    pb=ref["per_state"]["Semantic"]["per_layer"][l]["nn_sparse_k25_share"]
    same = (a>b)==(pa>pb)
    agree+=same
    print(f"  L{l:2d}: mine Divine {a:.4f} vs prolet {b:.4f} ({'D>P' if a>b else 'P>D'}); "
          f"pilot {pa:.4f} vs {pb:.4f} ({'D>P' if pa>pb else 'P>D'}) {'agree' if same else 'DISAGREE'}")
print(f"orderings agreeing: {agree}/12")
