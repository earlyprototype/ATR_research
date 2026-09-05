"""EXP_011 stage 4: the two descriptive readouts (spec section 7.5, items 2 and 3).

1. Top atoms. Names the vocabulary tokens whose lens vectors the decomposition
   selected for the named states at every layer, ranked by coefficient times atom
   norm (the length each atom contributes to the reconstruction).
2. Clamping check. At each band layer, splits a state into its J-space component
   and the rest, rescales only the component by 0, 0.5, 1 and 2, and reads the
   result out twice: through the lens itself, and causally by splicing the
   modified state back into the model at that layer and letting the rest of the
   network run.

Run: python3 readouts.py
"""
import json
import os
import sys
import time

import numpy as np
import torch

torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
sys.path.insert(0, HERE)
from jspace import decompose, unit_rows  # noqa: E402
from lens_gate import (LENS_PT, check_against_decomposition,  # noqa: E402
                       verify_lens)

BAND = [5, 6, 7, 8, 9, 10]
ALPHAS = [0.0, 0.5, 1.0, 2.0]
CLAMP_STATES = ["prolet1000", "phaseA"]
TOP_N = 8


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# The pinned-lens gate, shared with decompose.py through lens_gate.py. This stage
# loads the lens a second time, after the decomposition has already run, and it
# mixes two things: the atom indices and coefficients saved in
# output/atom_records.json, which were chosen against the dictionary the
# decomposition saw, and this stage's own dictionary, which ranks the atoms by
# contributed length and does the clamping. If the lens file changed in between,
# those two would come from different dictionaries and the readout would be
# meaningless. Two checks close that: the file must match the digest and size the
# specification pins, and it must match the digest the shares file records for the
# decomposition that produced the atom records.
LENS_ID = verify_lens(log=log, stage="readouts")

meta = json.load(open(os.path.join(OUT, "states_meta.json")))
atom_records = json.load(open(os.path.join(OUT, "atom_records.json")))
npz = np.load(os.path.join(OUT, "states.npz"))

DECOMPOSITION_LENS_SHA256 = check_against_decomposition(
    os.path.join(OUT, "shares.json"), LENS_ID, log=log)
named_keys = meta["named"]["keys"]
NIDX = {k: i for i, k in enumerate(named_keys)}
LAYERS = sorted(int(x) for x in atom_records["named"].keys())

from jlens.lens import JacobianLens  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from transformer_lens import HookedTransformer  # noqa: E402

lens = JacobianLens.load(LENS_PT)
hf = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32)
W_U = hf.lm_head.weight.detach().clone().contiguous()
ln_f = hf.transformer.ln_f
tokz = AutoTokenizer.from_pretrained("gpt2")
log(f"loaded W_U {tuple(W_U.shape)} and the final LayerNorm")

model = HookedTransformer.from_pretrained("gpt2", device="cpu")
model.eval()
log("loaded GPT-2 Small through TransformerLens for the causal clamping readout")


def jac(l):
    return lens.jacobians[l] if l in lens.jacobians else torch.eye(768)


# ------------------------------------------------------------- top atoms ----
log("naming the selected atoms for the named states")
top_atoms = {}
for l in LAYERS:
    D = W_U @ jac(l)
    rec = atom_records["named"][str(l)]
    for si, key in enumerate(named_keys):
        atoms, coeffs = rec["atoms"][si], rec["coeffs"][si]
        if not atoms:
            top_atoms.setdefault(key, {})[str(l)] = []
            continue
        contrib = [(c * float(D[a].norm()), a, c) for a, c in zip(atoms, coeffs)]
        contrib.sort(reverse=True)
        top_atoms.setdefault(key, {})[str(l)] = [
            {"token": tokz.decode([a]), "token_id": int(a),
             "coefficient": float(c), "contributed_length": float(w)}
            for w, a, c in contrib[:TOP_N]]
    del D
top_atoms["_meta"] = dict(
    LENS_ID,
    decomposition_lens_sha256=DECOMPOSITION_LENS_SHA256,
    note=("The lens file was checked against the digest and size specification "
          "section 3 pins before this readout was written, and against the digest "
          "output/shares.json records for the decomposition that chose these "
          "atoms. Every other key is a named state."))
with open(os.path.join(OUT, "top_atoms.json"), "w") as fh:
    json.dump(top_atoms, fh, indent=1)
for key in ("prolet1000", "phaseA", "phaseB", "pivotM"):
    for l in (5, 8, 10, 11):
        toks = [e["token"] for e in top_atoms[key][str(l)][:6]]
        log(f"  {key:11s} L{l:2d}: {toks}")


# -------------------------------------------------------------- clamping ----
def lens_readout(h, l, topk=5):
    with torch.no_grad():
        logits = W_U @ ln_f((jac(l) @ h).unsqueeze(0)).squeeze(0)
    p = logits.softmax(-1)
    v, i = p.topk(topk)
    return [(tokz.decode([int(t)]), round(float(x), 4)) for x, t in zip(v, i)]


def causal_readout(h, l, seq_len, topk=5):
    """Splice h into every position of blocks.l.hook_resid_post and finish the pass."""
    toks = torch.full((1, seq_len), 262, dtype=torch.long)

    def hookfn(resid, hook, v=h):
        resid[0, :, :] = v
        return resid

    model.add_hook(f"blocks.{l}.hook_resid_post", hookfn)
    try:
        with torch.no_grad():
            logits = model(toks)[0, -1]
    finally:
        model.reset_hooks()
    p = logits.softmax(-1)
    v, i = p.topk(topk)
    return [(tokz.decode([int(t)]), round(float(x), 4)) for x, t in zip(v, i)]


log("clamping check")
clamp = {}
seq_of = {"prolet1000": None, "phaseA": None}
sp = torch.load("/home/user/lucier-gpt2-activ-tensor-reson-experiments/experiments/"
                "gpt2_small/output_divine_motion/state_prolet.pt", weights_only=True)
sd = torch.load("/home/user/lucier-gpt2-activ-tensor-reson-experiments/experiments/"
                "gpt2_small/output_divine_motion/state_divine.pt", weights_only=True)
seq_of["prolet1000"] = int(sp["current_tensor"].shape[0])
seq_of["phaseA"] = int(sd["current_tensor"].shape[0])

for l in BAND:
    D = (W_U @ jac(l)).contiguous()
    Dn = unit_rows(D)
    for key in CLAMP_STATES:
        h = torch.from_numpy(npz["named"][NIDX[key], l, :]).contiguous()
        r = decompose(D, h.unsqueeze(0), Dn=Dn, record_atoms=True)
        idx = torch.tensor(r["atoms"][0], dtype=torch.long)
        coef = torch.tensor(r["coeffs"][0], dtype=torch.float32)
        comp = (D[idx] * coef.unsqueeze(1)).sum(0) if len(idx) else torch.zeros_like(h)
        rest = h - comp
        entry = {"share": float(r["share"][0]), "n_atoms": int(r["n_atoms"][0]),
                 "component_norm": float(comp.norm()), "state_norm": float(h.norm()),
                 "alphas": {}}
        for a in ALPHAS:
            hp = a * comp + rest
            entry["alphas"][str(a)] = {
                "lens_top5": lens_readout(hp, l),
                "causal_top5": causal_readout(hp, l, seq_of[key])}
        clamp.setdefault(key, {})[str(l)] = entry
        log(f"  {key} L{l}: share {entry['share']:.4f}; causal top-1 at alpha "
            f"{[entry['alphas'][str(a)]['causal_top5'][0] for a in ALPHAS]}")
    del D, Dn
clamp["_meta"] = dict(
    LENS_ID,
    decomposition_lens_sha256=DECOMPOSITION_LENS_SHA256,
    note=("The lens file was checked against the digest and size specification "
          "section 3 pins before this check was written. Every other key is a "
          "clamped state."))
with open(os.path.join(OUT, "clamping_check.json"), "w") as fh:
    json.dump(clamp, fh, indent=1)
log("wrote top_atoms.json and clamping_check.json")
