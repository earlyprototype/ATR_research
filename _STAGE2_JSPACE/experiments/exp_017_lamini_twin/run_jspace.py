"""EXP_017 Part 2: the J-space overlap probe (H18b).

Spec: ../../EXP_017_SPEC.md section 6.

Takes the terminal tensors Part 1 produced, reads their per-layer states, and
measures how much of each state lies in the cone spanned by at most 25 lens
vectors, on the twin's own fitted lens and on the pre-fitted Neuronpedia lens
for base GPT-2 Small, with three random-rotation controls per side and the two
cross-checks that separate lens mismatch from model mismatch.

The base lens is resolved relative to this checkout, at
`_STAGE2_JSPACE/artifacts/jlens_gpt2_small_neuronpedia.pt`, and can be
overridden with --base-lens. That directory is not version controlled, so the
file has to be placed there (or named on the command line) before this runs;
its SHA-256 is checked against the digest the spec records either way.

Usage:
    python3 run_jspace.py --twin-lens ../../artifacts/jlens_lamini_gpt2_124m_30_twin.pt
    python3 run_jspace.py --base-lens /some/other/path/jlens_gpt2_small_neuronpedia.pt
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import exp017_models  # noqa: E402
from jspace import (K_ATOMS, pursue_batch, random_rotation, rotate_states,  # noqa: E402
                    unit_atoms)

OUT = HERE / "output"
ARTIFACTS = HERE.parent.parent / "artifacts"
# Resolved from this checkout, never from one machine's absolute path, so the
# probe runs from any clone that has the lens in its own artifacts directory.
BASE_LENS_DEFAULT = ARTIFACTS / "jlens_gpt2_small_neuronpedia.pt"
BASE_LENS_SHA = "d1800a1335ada089ef2e1ec0e4bd4d5bd61e6011eacc31f8618fdb3d10aae762"
MODELS = exp017_models.MODELS
PROBE_LAYERS = list(range(11))      # 0..10, the lens's fitted source layers
BAND = list(range(5, 11))           # 5..10, the workspace band, verdict-bearing
ROT_SEEDS = (2026, 2027, 2028)
N_PERM = 10000
PERM_SEED = 42


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def per_layer_states(which, prompt_ids):
    """Inject each terminal tensor at the entrance to layer 0 and read the exit
    of every probed layer, the way the lucier pilot read per-layer states.

    The terminal is rescaled to the loop's own re-injection size first, so the
    states are the ones the loop's next iteration would actually visit. The
    rescale factor is recorded; at a settled state it is close to 1.

    Returns states [n_layers, d_model, n_prompts] (states as columns) and the
    model's unembedding matrix as numpy [d_model, d_vocab].
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformer_lens import HookedTransformer
    rev = exp017_models.revision(which)
    hf = AutoModelForCausalLM.from_pretrained(MODELS[which], revision=rev)
    tok = AutoTokenizer.from_pretrained(MODELS[which], revision=rev)
    model = HookedTransformer.from_pretrained("gpt2", hf_model=hf, tokenizer=tok,
                                              device="cpu")
    model.eval()

    terms = np.load(OUT / f"terminals_{which}.npz")
    loop = {r["prompt_id"]: r for r in json.load(open(OUT / f"loop_results_{which}.json"))}
    d = model.cfg.d_model
    states = np.zeros((len(PROBE_LAYERS), d, len(prompt_ids)), dtype=np.float32)
    rescales = {}
    names = [f"blocks.{l}.hook_resid_post" for l in PROBE_LAYERS]

    for pi, pid in enumerate(prompt_ids):
        T = torch.from_numpy(terms[f"{pid}|full"])
        target = float(loop[pid]["target_norm"])
        factor = target / float(T.norm())
        rescales[pid] = factor
        inj = T * factor

        def hook(resid, hook, tensor=inj):
            resid[0, :, :] = tensor
            return resid

        model.add_hook("blocks.0.hook_resid_pre", hook)
        try:
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    loop[pid]["prompt"], names_filter=lambda n: n in names)
        finally:
            model.reset_hooks()
        for li, l in enumerate(PROBE_LAYERS):
            states[li, :, pi] = cache[f"blocks.{l}.hook_resid_post"][0, -1, :].numpy()

    W_U = model.W_U.detach().numpy().astype(np.float32)   # [d_model, d_vocab]
    del model, hf
    return states, W_U, rescales


def load_lens(path):
    """Load a fitted lens; returns {layer: J as numpy [d, d]} and its metadata."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    J = {int(l): ck["J"][l].float().numpy().astype(np.float32) for l in ck["J"]}
    return J, {"n_prompts": int(ck["n_prompts"]), "d_model": int(ck["d_model"]),
               "source_layers": [int(x) for x in ck["source_layers"]]}


def permutation_p_two_sided(a, b, n_perm=N_PERM, seed=PERM_SEED):
    """Two-sided permutation p on the difference of medians.

    Pool the two samples, reassign which model each value belongs to n_perm
    times, and report the fraction of reassignments whose absolute median
    difference reaches the observed one, with the standard add-one correction.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = abs(float(np.median(a) - np.median(b)))
    pool = np.concatenate([a, b])
    na = len(a)
    rng = np.random.default_rng(seed)
    hits = 1
    for _ in range(n_perm):
        rng.shuffle(pool)
        if abs(float(np.median(pool[:na]) - np.median(pool[na:]))) >= obs - 1e-12:
            hits += 1
    return hits / (n_perm + 1), obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-suffix", default="",
                    help="suffix for the output filename; used only by the "
                         "non-registered harness check, which must not "
                         "overwrite the registered artifact")
    ap.add_argument("--twin-lens", default=None,
                    help="path to the fitted twin lens; omit to score on the "
                         "base lens only (spec section 6.2 fallback)")
    ap.add_argument("--base-lens", default=str(BASE_LENS_DEFAULT),
                    help="path to the pre-fitted Neuronpedia lens for base "
                         "GPT-2 Small; defaults to this checkout's own "
                         "_STAGE2_JSPACE/artifacts/ copy, and its SHA-256 is "
                         "checked against the digest the spec records")
    args = ap.parse_args()
    base_lens = Path(args.base_lens)
    if not base_lens.exists():
        raise SystemExit(
            f"base lens not found at {base_lens}. That directory is not "
            f"version controlled; put jlens_gpt2_small_neuronpedia.pt there, "
            f"or pass --base-lens with its path.")

    subset = json.load(open(HERE.parent / "exp_010c_windows" / "output"
                            / "prompt_subset_small.json"))
    prompt_ids = [r["id"] for r in subset]
    t0 = time.time()

    rep = {"experiment": "EXP_017", "spec": "../../EXP_017_SPEC.md",
           "k_atoms": K_ATOMS, "probe_layers": PROBE_LAYERS, "band_layers": BAND,
           "rotation_seeds": list(ROT_SEEDS), "n_perm": N_PERM,
           "perm_seed": PERM_SEED, "prompt_ids": prompt_ids, "lenses": {}}

    # ---- lenses --------------------------------------------------------------
    got = sha256_file(base_lens)
    rep["lenses"]["base"] = {"path": str(base_lens), "sha256": got,
                             "sha256_expected": BASE_LENS_SHA,
                             "digest_verified": got == BASE_LENS_SHA}
    assert got == BASE_LENS_SHA, f"base lens digest mismatch: {got}"
    J_base, meta_base = load_lens(base_lens)
    rep["lenses"]["base"].update(meta_base)
    print(f"base lens: {meta_base}", flush=True)

    lenses = {"base": J_base}
    if args.twin_lens:
        p = Path(args.twin_lens)
        J_twin, meta_twin = load_lens(p)
        rep["lenses"]["twin"] = {"path": str(p), "sha256": sha256_file(p), **meta_twin}
        lenses["twin"] = J_twin
        print(f"twin lens: {meta_twin}", flush=True)
    else:
        rep["lenses"]["twin"] = None
        print("NO TWIN LENS: base lens only (spec section 6.2 fallback)", flush=True)

    # ---- per-layer states, one model at a time so only one is resident -------
    states, W_U, rescale = {}, {}, {}
    for which in ("twin", "base"):
        states[which], W_U[which], rescale[which] = per_layer_states(which, prompt_ids)
        print(f"{which}: per-layer states read, rescale factor mean "
              f"{np.mean(list(rescale[which].values())):.6f}", flush=True)
    rep["terminal_rescale_to_injection_size"] = {
        k: {"mean": float(np.mean(list(v.values()))),
            "min": float(np.min(list(v.values()))),
            "max": float(np.max(list(v.values())))} for k, v in rescale.items()}

    rots = {s: random_rotation(states["base"].shape[1], s) for s in ROT_SEEDS}

    # ---- lens diagnostics, so instrument differences stay visible -----------
    diag = {}
    for l in PROBE_LAYERS:
        row = {ln: {"frobenius_norm": float(np.linalg.norm(J[l]))}
               for ln, J in lenses.items()}
        if len(lenses) == 2:
            a = lenses["twin"][l].ravel()
            b = lenses["base"][l].ravel()
            row["cosine_between_the_two_lenses"] = float(
                a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))
        diag[str(l)] = row
    rep["lens_diagnostics_per_layer"] = diag
    print(f"lens diagnostics: {json.dumps(diag)}", flush=True)

    # ---- the decomposition ---------------------------------------------------
    # shares[lens][state_model][variant] = array [n_layers, n_prompts]
    shares = {ln: {m: {v: np.zeros((len(PROBE_LAYERS), len(prompt_ids)),
                                   dtype=np.float64)
                       for v in ("real",) + tuple(f"rot{s}" for s in ROT_SEEDS)}
                   for m in ("twin", "base")} for ln in lenses}
    n_sel = {ln: {m: np.zeros((len(PROBE_LAYERS), len(prompt_ids)), dtype=int)
                  for m in ("twin", "base")} for ln in lenses}

    for lens_name, J in lenses.items():
        wu = W_U[lens_name]                       # the lens's own model unembeds
        for li, l in enumerate(PROBE_LAYERS):
            tl = time.time()
            A = (wu.T @ J[l]).astype(np.float32)  # [d_vocab, d_model] atoms
            U, keep = unit_atoms(A)
            del A
            cols, index = [], []
            for m in ("twin", "base"):
                H = states[m][li]                 # [d, n_prompts]
                cols.append(H); index.append((m, "real"))
                for s in ROT_SEEDS:
                    cols.append(rotate_states(H, rots[s]))
                    index.append((m, f"rot{s}"))
            Hall = np.concatenate(cols, axis=1)
            sh, ns, _ = pursue_batch(U, Hall, k=K_ATOMS, keep=keep)
            w = len(prompt_ids)
            for bi, (m, v) in enumerate(index):
                shares[lens_name][m][v][li] = sh[bi * w:(bi + 1) * w]
                if v == "real":
                    n_sel[lens_name][m][li] = ns[bi * w:(bi + 1) * w]
            del U, Hall, cols
            print(f"  lens={lens_name} layer={l}: "
                  f"twin real median {np.median(shares[lens_name]['twin']['real'][li]):.4f} "
                  f"base real median {np.median(shares[lens_name]['base']['real'][li]):.4f} "
                  f"({time.time() - tl:.0f}s)", flush=True)

    rep["shares"] = {ln: {m: {v: shares[ln][m][v].tolist() for v in shares[ln][m]}
                          for m in shares[ln]} for ln in shares}
    rep["n_atoms_selected_real"] = {ln: {m: n_sel[ln][m].tolist() for m in n_sel[ln]}
                                    for ln in n_sel}

    # ---- H18b scoring, spec section 6.6 -------------------------------------
    have_twin_lens = "twin" in lenses
    primary = {"twin_side": "twin-on-twin" if have_twin_lens else "twin-on-base",
               "base_side": "base-on-base", "per_layer": {}}
    twin_lens_key = "twin" if have_twin_lens else "base"
    for li, l in enumerate(PROBE_LAYERS):
        tw = shares[twin_lens_key]["twin"]["real"][li]
        bs = shares["base"]["base"]["real"][li]
        p, obs = permutation_p_two_sided(tw, bs)
        spread = max(abs(float(np.median(shares[twin_lens_key]["twin"][f"rot{s}"][li]))
                         - float(np.median(shares["base"]["base"][f"rot{s}"][li])))
                     for s in ROT_SEEDS)
        primary["per_layer"][str(l)] = {
            "median_twin": float(np.median(tw)), "median_base": float(np.median(bs)),
            "abs_median_difference": obs, "control_spread": spread,
            "exceeds_control_spread": bool(obs > spread),
            "perm_p": round(p, 5), "perm_p_below_0.05": bool(p < 0.05),
            "both_conditions": bool(obs > spread and p < 0.05),
            "in_band": l in BAND,
        }
    hits = [l for l in BAND if primary["per_layer"][str(l)]["both_conditions"]]
    primary["band_layers_meeting_both"] = hits
    primary["n_band_layers_meeting_both"] = len(hits)
    if not have_twin_lens:
        primary["h18b"] = "UNTESTABLE as registered (no twin lens; base lens both sides)"
    else:
        primary["h18b"] = "SUPPORTED" if len(hits) >= 4 else "NOT SUPPORTED"
    rep["h18b"] = primary

    # ---- cross-checks --------------------------------------------------------
    cross = {}
    for lens_name in lenses:
        for m in ("twin", "base"):
            key = f"{m}-on-{lens_name}"
            cross[key] = {str(l): {
                "median_real": float(np.median(shares[lens_name][m]["real"][li])),
                "median_controls": [float(np.median(shares[lens_name][m][f"rot{s}"][li]))
                                    for s in ROT_SEEDS],
                "mean_atoms_selected": float(np.mean(n_sel[lens_name][m][li])),
            } for li, l in enumerate(PROBE_LAYERS)}
    rep["cross_checks"] = cross

    # ---- the same-lens comparison, which separates model from instrument ----
    # The registered H18b pairing measures the twin on one lens and base on
    # another, so a difference could in principle come from the lenses. Holding
    # the lens fixed and swapping only whose states are decomposed isolates the
    # model effect; holding the states fixed and swapping only the lens
    # isolates the instrument effect. Both are reported per layer.
    same_lens = {}
    for lens_name in lenses:
        rows = {}
        for li, l in enumerate(PROBE_LAYERS):
            tw = shares[lens_name]["twin"]["real"][li]
            bs = shares[lens_name]["base"]["real"][li]
            p, obs = permutation_p_two_sided(tw, bs)
            rows[str(l)] = {"median_twin_states": float(np.median(tw)),
                            "median_base_states": float(np.median(bs)),
                            "model_effect_abs_median_difference": obs,
                            "perm_p": round(p, 5)}
        same_lens[lens_name] = rows
    rep["same_lens_model_effect"] = same_lens

    if len(lenses) == 2:
        lens_effect = {}
        for m in ("twin", "base"):
            rows = {}
            for li, l in enumerate(PROBE_LAYERS):
                a = shares["twin"][m]["real"][li]
                b = shares["base"][m]["real"][li]
                rows[str(l)] = {
                    "median_on_twin_lens": float(np.median(a)),
                    "median_on_base_lens": float(np.median(b)),
                    "lens_effect_abs_median_difference":
                        abs(float(np.median(a) - np.median(b)))}
            lens_effect[m] = rows
        rep["same_states_lens_effect"] = lens_effect
        ratios = {}
        for li, l in enumerate(PROBE_LAYERS):
            model = max(same_lens[ln][str(l)]["model_effect_abs_median_difference"]
                        for ln in lenses)
            model_min = min(same_lens[ln][str(l)]["model_effect_abs_median_difference"]
                            for ln in lenses)
            lens = max(lens_effect[m][str(l)]["lens_effect_abs_median_difference"]
                       for m in ("twin", "base"))
            ratios[str(l)] = {"smallest_model_effect": model_min,
                              "largest_model_effect": model,
                              "largest_lens_effect": lens,
                              "ratio_smallest_model_over_largest_lens":
                                  model_min / max(lens, 1e-12)}
        rep["model_effect_over_lens_effect"] = ratios

    rep["wall_seconds"] = round(time.time() - t0, 1)

    outfile = OUT / f"exp017_jspace{args.out_suffix}.json"
    outfile.write_text(json.dumps(rep, indent=2))
    print(f"\n=== H18b: {primary['h18b']} "
          f"({len(hits)}/6 band layers meet both conditions: {hits}) ===", flush=True)
    print(f"{'layer':<7}{'twin':<9}{'base':<9}{'|diff|':<9}{'ctrl spread':<13}{'perm p':<9}both")
    for l in PROBE_LAYERS:
        r = primary["per_layer"][str(l)]
        print(f"{l:<7}{r['median_twin']:<9.4f}{r['median_base']:<9.4f}"
              f"{r['abs_median_difference']:<9.4f}{r['control_spread']:<13.4f}"
              f"{r['perm_p']:<9}{r['both_conditions']}{'  <- band' if l in BAND else ''}")
    print(f"\nSaved -> output/{outfile.name} ({rep['wall_seconds']:.0f}s)")


if __name__ == "__main__":
    main()
