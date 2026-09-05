"""EXP_011 stage 2: decompose every per-layer state onto the J-space.

Builds the layer-l dictionary (the 50257 rows of W_U J_l), decomposes every
state in output/states.npz against it, and repeats against the two
pre-registered controls: a randomly rotated lens (three seeds) and a
norm-matched independent Gaussian dictionary (three seeds). Writes one JSON
table of shares per arm to output/shares.json.

Run: python3 decompose.py [--layers 0,1,...] [--quick]
"""
import argparse
import copy
import hashlib
import json
import os
import resource
import sys
import time

import numpy as np
import torch

torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
sys.path.insert(0, HERE)
from jspace import (decompose, unit_rows, random_rotation,          # noqa: E402
                    gaussian_dictionary_like, K_ATOMS, _self_test)

LENS_PT = "/home/user/ATR_research/_STAGE2_JSPACE/artifacts/jlens_gpt2_small_neuronpedia.pt"
# The digest and size the specification's section 3 pins. They are checked here,
# at the point where the lens is actually loaded for decomposition, and not only
# in build_states.py: a lens file swapped between the two stages would otherwise be
# analysed while shares.json inherited the earlier stage's attribution.
LENS_SHA256 = "d1800a1335ada089ef2e1ec0e4bd4d5bd61e6011eacc31f8618fdb3d10aae762"
LENS_BYTES = 12980477
N_LAYERS = 12
ROT_SEEDS = [2026, 2027, 2028]
GAUSS_SEEDS = [4242, 4243, 4244]
# States whose selected atoms are recorded for the top-atom readout.
RECORD_ATOMS_FAMILIES = {"named", "lang", "clean_last"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def merge_into(base, new):
    """Deep-merge new into base, dictionary by dictionary, new winning on leaves."""
    for key, val in new.items():
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            merge_into(base[key], val)
        else:
            base[key] = val
    return base


def write_json(path, payload, merge):
    """Write payload, merging into whatever is already at path when asked.

    A partial run (--quick, or --layers naming fewer than all of them) must not
    destroy the layers and arms an earlier full run already wrote. The write goes
    to a temporary file and is then renamed, so an interrupted write cannot leave
    a truncated file behind either.
    """
    out = payload
    if merge and os.path.exists(path):
        with open(path) as fh:
            existing = json.load(fh)
        out = merge_into(copy.deepcopy(existing), payload)
        log(f"  merged this run's layers and arms into the existing {os.path.basename(path)}")
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh)
    os.replace(tmp, path)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default=",".join(str(l) for l in range(N_LAYERS)))
    ap.add_argument("--quick", action="store_true",
                    help="lens arm only, for the first-layer cost check")
    ap.add_argument("--out", default="shares.json",
                    help="output file name inside output/ (default shares.json)")
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",") if x != ""]
    # A run that does not cover every layer with every arm is partial, and its
    # results are merged into whatever is already on disk rather than replacing it.
    partial = args.quick or sorted(set(layers)) != list(range(N_LAYERS))
    if partial:
        log(f"PARTIAL RUN: layers {layers}, "
            f"{'lens arm only' if args.quick else 'all arms'}. Results will be merged "
            f"into output/{args.out} instead of replacing it.")

    log("self-test of the decomposition")
    _self_test()

    npz = np.load(os.path.join(OUT, "states.npz"))
    families = [k for k in npz.files if k != "directions"]
    log(f"state families: {[(f, npz[f].shape) for f in families]}")

    from jlens.lens import JacobianLens
    from transformers import AutoModelForCausalLM
    lens_sha, lens_bytes = sha256(LENS_PT), os.path.getsize(LENS_PT)
    log(f"lens file {LENS_PT}")
    log(f"  SHA-256 {lens_sha} ({lens_bytes} bytes)")
    if lens_sha != LENS_SHA256 or lens_bytes != LENS_BYTES:
        raise SystemExit(
            "GATE FAILED: the lens file does not match the digest and size the "
            f"specification pins ({LENS_SHA256}, {LENS_BYTES} bytes). Refusing to "
            "decompose against an unattributed instrument.")
    lens = JacobianLens.load(LENS_PT)
    hf = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32)
    W_U = hf.lm_head.weight.detach().clone().contiguous()      # [50257, 768]
    del hf
    log(f"lens source layers {lens.source_layers}; W_U {tuple(W_U.shape)}")

    # Are the states already mean-centred over the 768 coordinates? TransformerLens
    # centres everything written to the residual stream, so the pre-registered
    # "centred" arm may be numerically identical to the raw arm. Measured, not assumed.
    centring = {}
    for fam in families:
        A = torch.from_numpy(npz[fam])
        A2 = A.reshape(-1, A.shape[-1]) if A.ndim == 3 else A
        mean_energy = (A2.mean(dim=1).abs() * np.sqrt(A2.shape[1])) / A2.norm(dim=1).clamp_min(1e-12)
        centring[fam] = {"max_mean_component_fraction_of_norm": float(mean_energy.max()),
                         "median_mean_component_fraction_of_norm": float(mean_energy.median())}
    log(f"mean-component fraction of state norm, worst over families: "
        f"{max(v['max_mean_component_fraction_of_norm'] for v in centring.values()):.3e}")

    results = {"k_atoms": K_ATOMS, "layers": layers, "arms": {},
               "centring_check": centring,
               "rotation_seeds": ROT_SEEDS, "gaussian_seeds": GAUSS_SEEDS,
               "lens_file": LENS_PT,
               "lens_sha256": lens_sha,
               "lens_bytes": lens_bytes,
               "lens_digest_matches_spec": True,
               "partial_run": partial}
    atom_records = {}
    identity_check = {}

    def stack_for_layer(l):
        """All states at layer l, plus an index of where each family sits."""
        blocks, index, off = [], {}, 0
        for fam in families:
            A = torch.from_numpy(npz[fam][:, l, :]).contiguous()
            blocks.append(A)
            index[fam] = (off, off + A.shape[0])
            off += A.shape[0]
        D = torch.from_numpy(npz["directions"]).contiguous()
        blocks.append(D)
        index["directions"] = (off, off + D.shape[0])
        return torch.cat(blocks, dim=0), index

    t_start = time.time()
    for l in layers:
        H, index = stack_for_layer(l)
        J = (lens.jacobians[l] if l in lens.jacobians
             else torch.eye(W_U.shape[1], dtype=torch.float32))
        D = (W_U @ J).contiguous()
        Dn = unit_rows(D)
        t0 = time.time()
        arms = {}

        rec = decompose(D, H, Dn=Dn, record_atoms=True)
        arms["lens"] = rec
        log(f"L{l:2d} lens: {time.time()-t0:.0f}s, {H.shape[0]} states, "
            f"mean share {rec['share'].mean():.4f}")

        if not args.quick:
            for s in ROT_SEEDS:
                Q = random_rotation(D.shape[1], s)
                # Decomposing against a rotated dictionary equals decomposing the
                # inverse-rotated state against the original one (proved in the
                # self-test; checked explicitly once below).
                arms[f"rot{s}"] = decompose(D, (H @ Q).contiguous(), Dn=Dn)
            for s in GAUSS_SEEDS:
                G = gaussian_dictionary_like(D, s)
                arms[f"gauss{s}"] = decompose(G, H)
                del G
            # Explicit rotated-dictionary check at one layer, on a few states.
            if l == layers[0]:
                Q = random_rotation(D.shape[1], ROT_SEEDS[0])
                Drot = (D @ Q.T).contiguous()
                sub = H[:8]
                a = decompose(Drot, sub)["share"]
                b = decompose(D, (sub @ Q).contiguous())["share"]
                identity_check = {"layer": l,
                                  "explicit_rotated_dictionary": [float(x) for x in a],
                                  "inverse_rotated_state": [float(x) for x in b],
                                  "max_abs_difference": float(np.abs(a - b).max())}
                log(f"  rotated-dictionary identity: max difference "
                    f"{identity_check['max_abs_difference']:.3e}")
                del Drot
            # Centred arm: the pre-registered secondary. Run on the lens at every
            # layer so the pre-registration is discharged with numbers, not an argument.
            # Spec section 8's plan also counted three rotation controls on the
            # centred residuals, which are not run here, so 8 of the 11 planned
            # arms exist. This is not a reduction under section 8's stopping rule,
            # which triggers only above a projected four hours. It is immaterial
            # for a measured reason: the loader centres every residual already, so
            # Hc equals H to within the mean-component fraction reported above
            # (4.9e-8 of a state's own length at worst), the lens_centred shares
            # agree with the lens shares to 2.4e-7 on a 0-to-1 scale at worst over
            # all 7,644 state-layer pairs, and therefore the raw arm's rotation
            # controls are the centred arm's rotation controls to the same
            # precision. Recorded as a deviation in RESULTS_EXP011.md.
            Hc = H - H.mean(dim=1, keepdim=True)
            arms["lens_centred"] = decompose(D, Hc.contiguous(), Dn=Dn)

        for name, r in arms.items():
            entry = results["arms"].setdefault(name, {})
            for fam, (a, b) in index.items():
                entry.setdefault(fam, {})[str(l)] = {
                    "share": [round(float(x), 8) for x in r["share"][a:b]],
                    "n_atoms": [int(x) for x in r["n_atoms"][a:b]],
                    # Per decomposition: did the search stop only because it hit the
                    # safety bound on rounds rather than a real stopping condition?
                    # It should never be true; score.py refuses a file in which it is.
                    "hit_max_iter": [bool(x) for x in r["hit_max_iter"][a:b]],
                }
        for fam in RECORD_ATOMS_FAMILIES:
            a, b = index[fam]
            atom_records.setdefault(fam, {})[str(l)] = {
                "atoms": rec["atoms"][a:b], "coeffs": rec["coeffs"][a:b]}
        log(f"L{l:2d} done in {time.time()-t0:.0f}s; elapsed {time.time()-t_start:.0f}s; "
            f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.2f} GB")
        del D, Dn, H

    results["identity_check"] = identity_check
    results["state_index_families"] = families
    results["wall_seconds"] = time.time() - t_start
    results["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    n_flagged = sum(int(sum(e["hit_max_iter"]))
                    for arm in results["arms"].values()
                    for fam in arm.values() for e in fam.values())
    results["n_hit_max_iter"] = n_flagged
    if n_flagged:
        log(f"  WARNING: {n_flagged} decompositions stopped on the iteration safety "
            f"bound rather than a real stopping condition")
    atoms_path = os.path.join(OUT, args.out.replace("shares", "atom_records")
                              if "shares" in args.out else "atom_records.json")
    merged = write_json(os.path.join(OUT, args.out), results, partial)
    write_json(atoms_path, atom_records, partial)
    # "layers" must describe the file, not just this run, once runs are merged.
    if partial and merged is not results:
        covered = sorted({int(l) for arm in merged["arms"].values()
                          for fam in arm.values() for l in fam})
        merged["layers"] = covered
        merged["partial_run"] = covered != list(range(N_LAYERS))
        write_json(os.path.join(OUT, args.out), merged, False)
        log(f"  {os.path.basename(args.out)} now covers layers {covered}")
    log(f"wrote output/{args.out} and output/{os.path.basename(atoms_path)} in "
        f"{results['wall_seconds']:.0f}s, peak RSS {results['peak_rss_gb']:.2f} GB")


if __name__ == "__main__":
    main()
