"""EXP_011 supporting diagnostic: how the lens dictionary is shaped.

A non-negative decomposition can only reach directions the dictionary's positive
cone covers, so the shape of the dictionary matters as much as its content. This
records, per layer: how clustered the 50,257 lens directions are around a single
common direction, what fraction lie on its positive side, and where each state
family sits relative to that common direction. Also records the same numbers for
one norm-matched random dictionary, which is the point of comparison.

The lens goes through the same pinned-lens gate as every other stage that opens
it (`lens_gate.py`): the file must match the digest and byte count specification
section 3 pins, and it must match the digest `output/shares.json` records for the
decomposition, because the anisotropy numbers this writes are read beside that
decomposition's shares. The lens identity is written into the output file's own
metadata block, so a later reader can tell which instrument produced these
numbers instead of inferring it. Added on 2026-09-06 in the fifth review round
of pull request 84.

Run: python3 dictionary_geometry.py
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
from jspace import unit_rows, gaussian_dictionary_like  # noqa: E402
from lens_gate import LENS_PT, verify_lens, check_against_decomposition  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


from jlens.lens import JacobianLens  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

# Two gates before anything is measured: the file is the one the specification
# pins, and it is the one the committed decomposition ran against. The second
# returns None, with a line saying so, for a shares file written before the digest
# field existed on 2026-09-05, which is the case for the committed one.
LENS_ID = verify_lens(log=log, stage="dictionary geometry")
SHARES = os.path.join(OUT, "shares.json")
DECOMP_SHA = check_against_decomposition(SHARES, LENS_ID, log=log)

lens = JacobianLens.load(LENS_PT)
hf = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32)
W_U = hf.lm_head.weight.detach().clone().contiguous()
del hf
npz = np.load(os.path.join(OUT, "states.npz"))
FAMS = ["lang", "noise17", "nullold", "clean_last", "clean_mean"]

out = {"metadata": {
           "stage": "dictionary_geometry",
           "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "lens_file": LENS_ID["lens_file"],
           "lens_sha256": LENS_ID["lens_sha256"],
           "lens_bytes": LENS_ID["lens_bytes"],
           "lens_digest_matches_spec": LENS_ID["lens_digest_matches_spec"],
           "decomposition_lens_sha256_in_shares_json": DECOMP_SHA,
           "decomposition_digest_note": (
               "The digest output/shares.json records for the decomposition that "
               "produced the shares these numbers are read beside. None means that "
               "file predates the digest field, which decompose.py began writing on "
               "2026-09-05, so the two stages are tied by the pinned digest alone."),
           "states_file": os.path.join(OUT, "states.npz")},
       "note": ("concentration is the length of the average of the 50257 unit lens "
                "directions: 0 would mean perfectly even spread over the sphere, 1 "
                "would mean every direction identical. fraction_positive is how many "
                "of them lie on the positive side of that average direction. "
                "state_cosine_median is the median cosine between a family's states "
                "and that average direction; a negative value means the family points "
                "away from the cone the non-negative decomposition can reach."),
       "per_layer": {}}
for l in range(12):
    J = lens.jacobians[l] if l in lens.jacobians else torch.eye(768)
    D = (W_U @ J).contiguous()
    Dn = unit_rows(D)
    mu = Dn.mean(0)
    conc = float(mu.norm())
    u = mu / mu.norm()
    cosmu = Dn @ u
    G = gaussian_dictionary_like(D, 4242)
    Gn = unit_rows(G)
    muG = Gn.mean(0)
    entry = {"concentration": conc,
             "fraction_positive": float((cosmu > 0).float().mean()),
             "atom_cosine_to_mean_direction": {"mean": float(cosmu.mean()),
                                               "min": float(cosmu.min()),
                                               "max": float(cosmu.max())},
             "gaussian_control_concentration": float(muG.norm()),
             "state_cosine_to_mean_direction": {}}
    for fam in FAMS:
        H = torch.from_numpy(npz[fam][:, l, :])
        c = (H / H.norm(dim=1, keepdim=True)) @ u
        entry["state_cosine_to_mean_direction"][fam] = {
            "median": float(c.median()), "min": float(c.min()), "max": float(c.max())}
    out["per_layer"][str(l)] = entry
    print(f"[{time.strftime('%H:%M:%S')}] L{l:2d}: concentration {conc:.4f} "
          f"(gaussian control {float(muG.norm()):.4f}), fraction positive "
          f"{entry['fraction_positive']:.4f}; state cosine medians "
          + ", ".join(f"{f}={entry['state_cosine_to_mean_direction'][f]['median']:+.3f}"
                      for f in FAMS), flush=True)
    del D, Dn, G, Gn
with open(os.path.join(OUT, "dictionary_geometry.json"), "w") as fh:
    json.dump(out, fh, indent=1)
log(f"wrote output/dictionary_geometry.json against lens {LENS_ID['lens_sha256']}")
