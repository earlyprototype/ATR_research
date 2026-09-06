"""EXP_017: which coordinate frame the J-space probe measured in, and what the
other frame gives.

Spec: ../../EXP_017_SPEC.md section 6. This script carries no verdict weight.
It exists because a review of the pull request asked whether the dictionary the
H18b probe decomposed against sits in the same coordinates as the states it
decomposed, and the answer needed measurement rather than argument.

The three things it measures.

1. What the loader did. `run_loop.py`, `verify_model.py` and `run_jspace.py`
   all load each model with `HookedTransformer.from_pretrained("gpt2",
   hf_model=..., tokenizer=..., device="cpu")` and pass none of that function's
   processing flags, so its defaults apply. This script reads those defaults
   from the installed library rather than quoting them, and records them.

2. How far the unembedding the probe used is from the model's own output
   matrix. The probe builds its dictionary from `model.W_U`, the unembedding of
   the TransformerLens conversion. The Jacobian matrices it multiplies that by
   were fitted by `jlens.from_hf` against the Hugging Face model, whose output
   matrix is `lm_head.weight`. The two differ, and this measures by how much,
   whether the difference is one common vector added to every vocabulary
   direction, and whether it is explained by the conversion folding the final
   normalisation step's learned per-coordinate gain into the unembedding and
   then subtracting a mean.

3. What that does to the J-space share. For one band layer and one layer
   outside the band, five prompts per model, real dictionaries only, the share
   is computed three ways: the states and dictionary the committed run used,
   the same states against the Hugging Face dictionary, and the Hugging Face
   states against the Hugging Face dictionary. The third is the consistent
   frame, the one the Jacobians were fitted in.

Writes output/frame_check.json. The full recomputation of every H18b number in
the consistent frame is a separate artifact, output/exp017_jspace_hfframe.json,
which `run_jspace.py --frame hf` writes.

Usage:
    python3 frame_check.py
    python3 frame_check.py --layers 0 5 --n-prompts 5
"""
import argparse
import inspect
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
import run_jspace  # noqa: E402
from jspace import K_ATOMS, pursue_batch, unit_atoms  # noqa: E402

OUT = HERE / "output"
ARTIFACTS = HERE.parent.parent / "artifacts"
LENSES = {"twin": ARTIFACTS / "jlens_lamini_gpt2_124m_40_twin.pt",
          "base": ARTIFACTS / "jlens_gpt2_small_neuronpedia.pt"}


def loader_defaults():
    """The processing the TransformerLens loader applies when nothing is asked
    for, read from the installed library."""
    from transformer_lens import HookedTransformer
    params = inspect.signature(HookedTransformer.from_pretrained).parameters
    wanted = ("fold_ln", "center_writing_weights", "center_unembed",
              "fold_value_biases", "refactor_factored_attn_matrices")
    return {k: (None if k not in params else params[k].default) for k in wanted}


def unembedding_comparison(which):
    """How far the unembedding the probe used is from the model's own.

    Returns the measurement dictionary, the TransformerLens unembedding and the
    Hugging Face one, both as [d_model, d_vocab] float32.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformer_lens import HookedTransformer
    rev = exp017_models.revision(which)
    hf = AutoModelForCausalLM.from_pretrained(exp017_models.MODELS[which],
                                              revision=rev)
    tok = AutoTokenizer.from_pretrained(exp017_models.MODELS[which], revision=rev)
    ht = HookedTransformer.from_pretrained("gpt2", hf_model=hf, tokenizer=tok,
                                           device="cpu")
    ht.eval()
    tl = ht.W_U.detach().numpy().astype(np.float64)
    raw = hf.lm_head.weight.detach().numpy().astype(np.float64).T
    gain = hf.transformer.ln_f.weight.detach().numpy().astype(np.float64)

    v = min(tl.shape[1], raw.shape[1])
    A, B = tl[:, :v], raw[:, :v]
    difference = A - B
    common = difference.mean(axis=1)                 # one vector for all tokens
    residual = difference - common[:, None]
    column_norms = np.linalg.norm(B, axis=0)
    cosines = ((A * B).sum(0)
               / np.maximum(np.linalg.norm(A, axis=0) * column_norms, 1e-30))
    # What the conversion says it does: fold the final normalisation gain into
    # every direction, centre each direction across the 768 coordinates because
    # that normalisation removes the mean anyway, then subtract the mean over
    # the vocabulary.
    predicted = gain[:, None] * B
    predicted = predicted - predicted.mean(axis=0, keepdims=True)
    predicted = predicted - predicted.mean(axis=1, keepdims=True)

    out = {
        "model": exp017_models.MODELS[which],
        "revision": rev,
        "d_vocab_transformer_lens": int(tl.shape[1]),
        "d_vocab_hugging_face": int(raw.shape[1]),
        "frobenius_norm_transformer_lens": float(np.linalg.norm(A)),
        "frobenius_norm_hugging_face": float(np.linalg.norm(B)),
        "frobenius_norm_of_the_difference": float(np.linalg.norm(difference)),
        "difference_relative_to_hugging_face": float(
            np.linalg.norm(difference) / np.linalg.norm(B)),
        "common_shift_norm": float(np.linalg.norm(common)),
        "mean_column_norm_hugging_face": float(column_norms.mean()),
        "common_shift_over_mean_column_norm": float(
            np.linalg.norm(common) / column_norms.mean()),
        "fraction_of_the_difference_left_after_removing_the_common_shift": float(
            np.linalg.norm(residual) / np.linalg.norm(difference)),
        "per_token_cosine_min": float(cosines.min()),
        "per_token_cosine_median": float(np.median(cosines)),
        "per_token_cosine_max": float(cosines.max()),
        "final_norm_gain_min": float(gain.min()),
        "final_norm_gain_max": float(gain.max()),
        "final_norm_gain_mean": float(gain.mean()),
        "gain_fold_and_centring_explains_the_difference_to": float(
            np.linalg.norm(predicted - A) / np.linalg.norm(A)),
    }
    del ht, hf
    return out, A[:, :v].astype(np.float32), B[:, :v].astype(np.float32)


def state_comparison(which, prompt_ids, layers):
    """The states in both frames, and how they differ.

    Returns the measurement dictionary and the two state arrays, each
    [n_layers, d_model, n_prompts].
    """
    tl, _, _ = run_jspace.per_layer_states(which, prompt_ids, "tl")
    hf, _, _ = run_jspace.per_layer_states(which, prompt_ids, "hf")
    rows = {}
    for li, l in enumerate(run_jspace.PROBE_LAYERS):
        if l not in layers:
            continue
        a, b = tl[li], hf[li]
        gap = b - a
        offset = gap.mean(axis=0)
        norms = np.linalg.norm(a, axis=0)
        rows[str(l)] = {
            "mean_state_norm": float(norms.mean()),
            "max_abs_mean_over_coordinates_transformer_lens": float(
                np.abs(a.mean(axis=0)).max()),
            "max_abs_mean_over_coordinates_hugging_face": float(
                np.abs(b.mean(axis=0)).max()),
            "max_departure_of_the_gap_from_one_constant": float(
                np.abs(gap - offset[None, :]).max()),
            "gap_length_over_state_length": float(
                np.mean(np.abs(offset) * np.sqrt(a.shape[0]) / norms)),
        }
    return rows, tl, hf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 5],
                    help="layers for the state and share comparison; the "
                         "default is one layer outside the verdict-bearing band "
                         "and one inside it")
    ap.add_argument("--n-prompts", type=int, default=5,
                    help="prompts per model for the share comparison")
    args = ap.parse_args()

    t0 = time.time()
    subset = json.load(open(HERE.parent / "exp_010c_windows" / "output"
                            / "prompt_subset_small.json"))
    prompt_ids = [r["id"] for r in subset][: args.n_prompts]

    rep = {"experiment": "EXP_017", "carries_verdict_weight": False,
           "what_this_is": (
               "the coordinate frame the H18b probe measured in, against the "
               "frame its Jacobian matrices were fitted in"),
           "transformer_lens_processing_defaults": loader_defaults(),
           "loaded_by": ("HookedTransformer.from_pretrained(\"gpt2\", "
                         "hf_model=..., tokenizer=..., device=\"cpu\"), with no "
                         "processing flag passed, in run_loop.py, "
                         "verify_model.py and run_jspace.py alike"),
           "layers": args.layers, "band_layers": run_jspace.BAND,
           "prompt_ids": prompt_ids, "unembedding": {}, "states": {},
           "subset_shares": {}}
    print(f"loader defaults: {rep['transformer_lens_processing_defaults']}",
          flush=True)

    lenses = {name: run_jspace.load_lens(path)[0] for name, path in LENSES.items()}
    W = {}
    for which in ("twin", "base"):
        rep["unembedding"][which], tl_u, hf_u = unembedding_comparison(which)
        W[which] = {"tl": tl_u, "hf": hf_u}
        print(f"{which}: unembedding compared ({time.time() - t0:.0f}s)", flush=True)

    states = {}
    for which in ("twin", "base"):
        rep["states"][which], tl_s, hf_s = state_comparison(
            which, prompt_ids, args.layers)
        states[which] = {"tl": tl_s, "hf": hf_s}
        print(f"{which}: states compared ({time.time() - t0:.0f}s)", flush=True)

    # The registered pairing decomposes each model's states against that model's
    # own lens, so the subset keeps that pairing and changes only the frame.
    combinations = (("registered", "tl", "tl"), ("hf_dictionary", "tl", "hf"),
                    ("hf_frame", "hf", "hf"))
    for l in args.layers:
        li = run_jspace.PROBE_LAYERS.index(l)
        row = {}
        for which in ("twin", "base"):
            for label, state_frame, dict_frame in combinations:
                A = (W[which][dict_frame].T @ lenses[which][l]).astype(np.float32)
                U, keep = unit_atoms(A)
                H = states[which][state_frame][li].astype(np.float32)
                shares, _, _ = pursue_batch(U, H, k=K_ATOMS, keep=keep)
                row.setdefault(label, {})[which] = {
                    "median": float(np.median(shares)),
                    "per_prompt": [float(x) for x in shares],
                    "mean_atom_norm": float(np.linalg.norm(A, axis=1).mean()),
                }
                del A, U
        for label in row:
            row[label]["absolute_difference_of_medians"] = abs(
                row[label]["twin"]["median"] - row[label]["base"]["median"])
        rep["subset_shares"][str(l)] = row
        print(f"layer {l}: " + "  ".join(
            f"{label} twin {row[label]['twin']['median']:.4f} base "
            f"{row[label]['base']['median']:.4f}" for label in row)
            + f" ({time.time() - t0:.0f}s)", flush=True)

    rep["wall_seconds"] = round(time.time() - t0, 1)
    outfile = OUT / "frame_check.json"
    outfile.write_text(json.dumps(rep, indent=2))
    print(f"\nSaved -> output/{outfile.name} ({rep['wall_seconds']:.0f}s)")


if __name__ == "__main__":
    main()
