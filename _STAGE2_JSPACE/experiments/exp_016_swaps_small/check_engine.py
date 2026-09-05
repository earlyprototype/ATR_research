"""Correctness checks on the swap engine, run before the batteries.

Checks that (1) a swap of a token with itself changes nothing, (2) the
coordinates really are exchanged at the patched position, (3) the change to
the residual stream lies in the plane spanned by the two lens vectors, and
(4) the batched path agrees with a single-element path.
"""
import sys, torch
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_exp016 import load_model, load_lens, lens_vectors
from swap_engine import SwapPlan, pinv2, random_pair, run_plan

model = load_model(); lens = load_lens()
prompt = "My favourite sport is"
toks = model.to_tokens(prompt); T = toks.shape[1]
s = model.to_tokens(" football", prepend_bos=False)[0, 0].item()
t = model.to_tokens(" cricket", prepend_bos=False)[0, 0].item()
L = 8
V = lens_vectors(lens, model, L, [s, t])
print("lens vector lengths at layer 8: v_source", round(V[:,0].norm().item(),3),
      "v_target", round(V[:,1].norm().item(),3),
      "cosine between them", round(float(torch.nn.functional.cosine_similarity(
          V[:,0], V[:,1], dim=0)), 4))

with torch.no_grad():
    clean = model(toks)[0, -1].float()
    _, cache = model.run_with_cache(toks)
h = cache["blocks.8.hook_resid_post"][0].float()

# (1) identity swap
p = SwapPlan(T, [8]); p.add({8: torch.stack([V[:,0], V[:,0]],1)},
                            {8: torch.stack([V[:,0], V[:,0]],1)}, 1.0,
                            range(T), False); p.build()
out = run_plan(model, toks, p)[0]
print("(1) swapping a token with itself, max change in log-probabilities:",
      f"{(out - torch.log_softmax(clean,-1)).abs().max().item():.2e}")

# (2) coordinates exchanged, (3) change stays in the plane
P = pinv2(V.unsqueeze(0))[0]
c = P @ h[-1]
patch = V @ (c.flip(0) - c)
newc = P @ (h[-1] + patch)
print("(2) coordinates before", [round(float(x),3) for x in c],
      "after", [round(float(x),3) for x in newc])
Q = V @ torch.linalg.pinv(V)
resid = patch - Q @ patch
print("(3) part of the change outside the plane:",
      f"{resid.norm().item():.2e} against a change of size {patch.norm().item():.3f}")

# (4) batched equals single
p1 = SwapPlan(T, [8]); p1.add({8: V}, {8: V}, 1.0, range(T), False); p1.build()
a = run_plan(model, toks, p1)[0]
p2 = SwapPlan(T, [8])
r = random_pair(V[:,0], V[:,1], 7)
for _ in range(5): p2.add({8: V}, {8: V}, 1.0, range(T), False)
p2.add({8: r}, {8: V}, 1.0, range(T), True); p2.build()
b = run_plan(model, toks, p2)
print("(4) batched against single, max difference:",
      f"{(a - b[0]).abs().max().item():.2e}; the size-matched control row "
      f"differs from the lens row by {(a - b[5]).abs().max().item():.3f}")

top = lambda v: [model.to_string(int(i)) for i in torch.topk(v, 5).indices]
print("clean top five:", top(clean))
print("after football/cricket swap at layer 8, all positions, strength 1:", top(a))
