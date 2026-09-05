"""EXP_011 stage 1: build the per-layer states every hypothesis is scored on.

Reads the committed Stage 1 tensors (read-only, outside this worktree), rebuilds
each settled state's full token tensor, splices it into GPT-2 Small at
blocks.0.hook_resid_pre exactly as the ATR loop does, and records
blocks.l.hook_resid_post at the last token position for every layer l from 0 to
11. Ordinary non-iterated prompt residuals are read the same way without any
injection. Everything is written to output/states.npz plus a metadata JSON; no
lens and no decomposition happen here.

Run: python3 build_states.py
"""
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np
import torch

torch.set_num_threads(1)
import torch.nn.functional as F  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)

LUCIER = "/home/user/lucier-gpt2-activ-tensor-reson-experiments"
FROZEN = "/home/user/shared/stage1_frozen/experiments/gpt2_small"
LENS_PT = "/home/user/ATR_research/_STAGE2_JSPACE/artifacts/jlens_gpt2_small_neuronpedia.pt"
N_LAYERS = 12
SCAFFOLD_TOKEN = 262  # token identity is irrelevant: the injection overwrites it


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------- model ----
log("loading GPT-2 Small through TransformerLens")
from transformer_lens import HookedTransformer  # noqa: E402

# from_pretrained applies TransformerLens's default weight processing, which
# includes center_writing_weights=True: every matrix that writes into the residual
# stream has its output mean over the 768 coordinates removed, so every state this
# script records is mean-centred by construction. That is the same frame the
# earlier pilot scripts used (10_jlens_phase.py loads the model the same way), and
# it is why the spec's "raw" and "mean-centred" arms come out numerically
# identical. It is NOT the frame the dictionary lives in: decompose.py builds the
# dictionary from the Hugging Face unembedding, which is not centred. The
# consequence is recorded as a limitation in RESULTS_EXP011.md rather than
# repaired here, because repairing it means rebuilding every state.
model = HookedTransformer.from_pretrained("gpt2", device="cpu")
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
HOOKS = [f"blocks.{l}.hook_resid_post" for l in range(N_LAYERS)]
log(f"model ready: n_layers={model.cfg.n_layers} d_model={model.cfg.d_model}")


def per_layer_from_tensor(tensor):
    """Inject a [T, 768] state at blocks.0.hook_resid_pre; return [12, 768]."""
    T = tensor.shape[0]
    toks = torch.full((1, T), SCAFFOLD_TOKEN, dtype=torch.long)
    inject = tensor.clone()

    def hookfn(resid, hook, tt=inject):
        resid[0, :, :] = tt
        return resid

    model.add_hook("blocks.0.hook_resid_pre", hookfn)
    try:
        with torch.no_grad():
            _, cache = model.run_with_cache(toks, names_filter=lambda n: n in HOOKS)
    finally:
        model.reset_hooks()
    return torch.stack([cache[h][0, -1].clone() for h in HOOKS])


def per_layer_from_terminal(vec, seq_len, initial_norm):
    """Rebuild the settled tensor from its last-position vector and read layers."""
    x = vec.unsqueeze(0).repeat(seq_len, 1)
    x = x * (initial_norm / x.norm())
    return per_layer_from_tensor(x)


def norm_to(x, n):
    return x * (n / x.norm())


# ------------------------------------------------------------------ data ----
log("loading committed Stage 1 records (read-only)")
stage1 = torch.load(os.path.join(FROZEN, "output/stage1_results.pt"),
                    weights_only=False, map_location="cpu")
gated = torch.load(os.path.join(FROZEN, "output_gated/gated_results.pt"),
                   weights_only=False, map_location="cpu")
nullold_raw = torch.load(os.path.join(FROZEN, "output_random_baseline/random_baseline_results.pt"),
                         weights_only=False, map_location="cpu")
noise17_raw = torch.load(os.path.join(LUCIER, "experiments/noise_rerun/output/results.pt"),
                         weights_only=False, map_location="cpu")
noise17 = noise17_raw["results"]
pairing = {v["matched_to"]: (int(v["seq_len"]), float(v["target_frobenius"]))
           for v in noise17.values()}

sys.path.insert(0, LUCIER)
from prompt_library import PROMPT_LIBRARY  # noqa: E402

meta = {
    "experiment": "EXP_011",
    "stage": "build_states",
    "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "versions": {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
    },
    "lens_file": LENS_PT,
    "lens_sha256": sha256(LENS_PT),
    "lens_bytes": os.path.getsize(LENS_PT),
    "sources": {
        "stage1_results": os.path.join(FROZEN, "output/stage1_results.pt"),
        "gated_results": os.path.join(FROZEN, "output_gated/gated_results.pt"),
        "random_baseline": os.path.join(FROZEN, "output_random_baseline/random_baseline_results.pt"),
        "noise_rerun_run17": os.path.join(LUCIER, "experiments/noise_rerun/output/results.pt"),
        "state_divine": os.path.join(LUCIER, "experiments/gpt2_small/output_divine_motion/state_divine.pt"),
        "state_prolet": os.path.join(LUCIER, "experiments/gpt2_small/output_divine_motion/state_prolet.pt"),
        "state_noise": os.path.join(LUCIER, "experiments/gpt2_small/output_divine_motion/state_noise.pt"),
        "converged_tensors": os.path.join(LUCIER, "experiments/gpt2_small/output_confidence/converged_tensors.pt"),
        "bell_anatomy": os.path.join(LUCIER, "experiments/gpt2_small/output_divine_motion/bell_anatomy.json"),
        "prompt_library": os.path.join(LUCIER, "prompt_library.py"),
    },
}
import importlib.metadata as _md  # noqa: E402
for pkg in ("transformers", "transformer_lens", "jlens", "scipy"):
    try:
        meta["versions"][pkg] = _md.version(pkg)
    except Exception as exc:  # pragma: no cover
        meta["versions"][pkg] = f"unavailable: {exc}"

arrays = {}
gates = {}

# --------------------------------------------------- position-collapse check --
log("checking position collapse in the committed tensors")
conv = torch.load(os.path.join(LUCIER, "experiments/gpt2_small/output_confidence/converged_tensors.pt"),
                  weights_only=True)
pc = []
for k, t in conv.items():
    n = t.norm(dim=1)
    tn = t / n.unsqueeze(1)
    pc.append({"state": k, "positions": int(t.shape[0]),
               "min_pairwise_cosine": float((tn @ tn.T).min()),
               "norm_ratio_max_over_min": float(n.max() / n.min())})
for k in list(nullold_raw)[:5]:
    t = nullold_raw[k][-1]["tensor"]
    n = t.norm(dim=1)
    tn = t / n.unsqueeze(1)
    pc.append({"state": f"nullold/{k}", "positions": int(t.shape[0]),
               "min_pairwise_cosine": float((tn @ tn.T).min()),
               "norm_ratio_max_over_min": float(n.max() / n.min())})
gates["position_collapse"] = pc
worst_cos = min(e["min_pairwise_cosine"] for e in pc)
log(f"  worst pairwise cosine between token positions: {worst_cos:.8f}")
stage1_possim = [float(v["position_similarity"][-1]) for v in stage1.values()]
gates["stage1_position_similarity_iter100"] = {
    "min": min(stage1_possim), "max": max(stage1_possim), "n": len(stage1_possim)}

# The language terminals are rebuilt the same way, by repeating the stored
# last-position vector, and they need the same scrutiny. Stage 1's own
# position_similarity divides every position by its own length before averaging
# the cosines (atr_engine2.py, "Position collapse metric"), so a value of 1 proves
# the positions share one direction and says nothing about their lengths. Stage 1
# stores no Frobenius norm for a snapshot, so the exact length test used for run 17
# below cannot be formed here. What can be formed is this: when the positions are
# parallel the stored position average has length equal to the AVERAGE of the
# position lengths, so dividing the stored last vector's length by it asks whether
# the last position is as long as the average position. That is necessary for exact
# collapse without being sufficient, and the limitation is recorded in
# RESULTS_EXP011.md rather than papered over.
lang_pc = []
for pid in sorted(stage1):
    snap = stage1[pid]
    mv_l, lv_l = snap["mean_vectors"][-1].double(), snap["last_vectors"][-1].double()
    lang_pc.append({
        "prompt": pid,
        "position_similarity": float(snap["position_similarity"][-1]),
        "last_over_mean_position_length": float(lv_l.norm() / mv_l.norm()),
        "cos_mean_vec_last_vec": float(F.cosine_similarity(mv_l.unsqueeze(0),
                                                           lv_l.unsqueeze(0))),
        "all_positions_read_out_one_word": len(set(snap["all_position_tokens"][-1])) == 1,
    })


def _span(rows, key):
    vals = [r[key] for r in rows]
    return {"min": min(vals), "max": max(vals)}


gates["lang_position_collapse_from_stage1_record"] = {
    "n": len(lang_pc),
    "source": ("the committed Stage 1 record's iteration-100 snapshot: "
               "position_similarity, mean_vectors, last_vectors and "
               "all_position_tokens. No terminal tensor and no Frobenius norm are "
               "stored there."),
    "position_similarity": _span(lang_pc, "position_similarity"),
    "last_over_mean_position_length": _span(lang_pc, "last_over_mean_position_length"),
    "cos_mean_vec_last_vec": _span(lang_pc, "cos_mean_vec_last_vec"),
    "n_all_positions_read_out_one_word": sum(
        1 for r in lang_pc if r["all_positions_read_out_one_word"]),
    "limitation": ("position_similarity establishes a common direction only, because "
                   "it normalises every position first. The length evidence here is "
                   "that the last position is as long as the average position, which "
                   "is necessary but not sufficient for every position to have the "
                   "same length. The sufficient test needs the terminal tensor's "
                   "Frobenius norm, which Stage 1 does not store."),
}
_lps = gates["lang_position_collapse_from_stage1_record"]
log(f"  language position collapse, from the Stage 1 record: position similarity "
    f"{_lps['position_similarity']['min']:.9f} to "
    f"{_lps['position_similarity']['max']:.9f}, last over average position length "
    f"{_lps['last_over_mean_position_length']['min']:.9f} to "
    f"{_lps['last_over_mean_position_length']['max']:.9f}, "
    f"{_lps['n_all_positions_read_out_one_word']}/125 prompts read out one word at "
    f"every position")
if (_lps["position_similarity"]["min"] < 0.9999
        or abs(_lps["last_over_mean_position_length"]["min"] - 1.0) > 1e-3
        or abs(_lps["last_over_mean_position_length"]["max"] - 1.0) > 1e-3):
    raise SystemExit("GATE FAILED: the language terminals are not position-collapsed "
                     "to the tolerance that makes repeating the last vector exact.")

# Run 17 stores only the terminal's last-position vector and its position mean,
# not the full terminal tensor, so this script rebuilds the tensor by repeating
# the last vector. That rebuild is exact only if the run-17 terminal was
# position-collapsed, and the reconstruction gate further down cannot establish
# it, because that gate only checks what a forward pass does to the already tiled
# tensor. The evidence therefore has to come from run 17's own record, and is
# taken from it here so the claim is not circular. Three quantities per trial,
# all read or derived from that record and none of them requiring the tensor:
# the mean off-diagonal cosine between token positions, which the engine computes
# in float64 at the terminal iteration; the root-mean-square of the token-position
# lengths divided by their mean, which the power-mean inequality makes exactly 1
# only when every position has the same length; and the cosine between the stored
# position mean and the stored last position.
n17_pc = []
for tid in sorted(noise17):
    e17 = noise17[tid]["result"]
    T17 = int(noise17[tid]["seq_len"])
    mv17, lv17 = e17["terminal_mean_vec"].double(), e17["terminal_last_vec"].double()
    n17_pc.append({
        "trial": tid,
        "iteration": int(e17["metrics"][-1]["iteration"]),
        "mean_offdiagonal_position_cosine": float(e17["metrics"][-1]["position_similarity_f64"]),
        "rms_over_mean_position_length": (float(e17["metrics"][-1]["tensor_norm"])
                                          / (T17 ** 0.5 * float(mv17.norm()))),
        "cos_mean_vec_last_vec": float(F.cosine_similarity(mv17.unsqueeze(0),
                                                           lv17.unsqueeze(0))),
    })
def _rng(key):
    return _span(n17_pc, key)
gates["noise17_position_collapse_from_run17_record"] = {
    "n": len(n17_pc),
    "source": ("run 17's own results.pt: the per-iteration metrics list at the "
               "terminal iteration, and the stored terminal_mean_vec and "
               "terminal_last_vec. No tensor is stored there and none is needed."),
    "mean_offdiagonal_position_cosine": _rng("mean_offdiagonal_position_cosine"),
    "rms_over_mean_position_length": _rng("rms_over_mean_position_length"),
    "cos_mean_vec_last_vec": _rng("cos_mean_vec_last_vec"),
    "note": ("A mean off-diagonal position cosine of 1 means every token position "
             "holds the same direction; a root-mean-square over mean of 1 means "
             "every position holds the same length. Both together make repeating "
             "the last vector an exact rebuild of the terminal tensor rather than "
             "an approximation."),
}
log(f"  run-17 position collapse, from run 17's own record: mean off-diagonal "
    f"position cosine {_rng('mean_offdiagonal_position_cosine')['min']:.12f} to "
    f"{_rng('mean_offdiagonal_position_cosine')['max']:.12f}, position-length "
    f"root-mean-square over mean "
    f"{_rng('rms_over_mean_position_length')['min']:.9f} to "
    f"{_rng('rms_over_mean_position_length')['max']:.9f}")

# tiling identity: reading the stored full tensor and reading the tiled last
# vector must give the identical per-layer states.
tile_checks = []
for k in list(nullold_raw)[:3]:
    snap = nullold_raw[k][-1]
    fro0 = float(nullold_raw[k][0]["tensor_norm"])
    Ha = per_layer_from_tensor(norm_to(snap["tensor"], fro0))
    Hb = per_layer_from_terminal(snap["last_vector"], snap["tensor"].shape[0], fro0)
    tile_checks.append({"trial": k,
                        "max_abs_diff": float((Ha - Hb).abs().max()),
                        "min_cosine": float(F.cosine_similarity(Ha, Hb, dim=1).min())})
gates["tiling_identity"] = tile_checks
log(f"  tiling identity: min cosine {min(c['min_cosine'] for c in tile_checks):.8f}")

# ------------------------------------------------------ family: lang (125) ---
log("building per-layer states: lang (125 language terminals, iteration 100)")
lang_ids = sorted(stage1)
lang_H, lang_cos = [], []
for pid in lang_ids:
    v = stage1[pid]["last_vectors"][-1]
    T, fro0 = pairing[pid]
    H = per_layer_from_terminal(v, T, fro0)
    lang_H.append(H)
    lang_cos.append(float(F.cosine_similarity(H[11].unsqueeze(0), v.unsqueeze(0))))
arrays["lang"] = torch.stack(lang_H).numpy().astype(np.float32)
meta["lang"] = {
    "ids": lang_ids,
    "label_iter100": [stage1[p]["top_tokens"][-1][0][0] for p in lang_ids],
    "label_gated": [gated[p]["terminal_token"] for p in lang_ids],
    "gated_converged": [bool(gated[p]["converged"]) for p in lang_ids],
    "seq_len": [pairing[p][0] for p in lang_ids],
    "initial_frobenius": [pairing[p][1] for p in lang_ids],
    "terminal_norm": [float(stage1[p]["last_vectors"][-1].norm()) for p in lang_ids],
    "reconstruction_cosine": lang_cos,
    "tensor_reconstruction": (
        "Rebuilt by repeating last_vectors[-1] across the paired sequence length "
        "and rescaling to the paired iteration-0 Frobenius norm. The evidence that "
        "this is exact is in gates['lang_position_collapse_from_stage1_record']: the "
        "positions share one direction, and the last position is as long as the "
        "average position. Stage 1 stores no Frobenius norm for a snapshot, so the "
        "stronger length test applied to run 17 cannot be formed for this arm; the "
        "limitation is recorded in RESULTS_EXP011.md."),
}

# --------------------------------------------------- family: noise17 (125) ---
log("building per-layer states: noise17 (run 17 matched-scale noise terminals)")
n17_ids = sorted(noise17)
n17_H, n17_cos = [], []
for tid in n17_ids:
    e = noise17[tid]
    v = e["result"]["terminal_last_vec"]
    H = per_layer_from_terminal(v, int(e["seq_len"]), float(e["target_frobenius"]))
    n17_H.append(H)
    n17_cos.append(float(F.cosine_similarity(H[11].unsqueeze(0), v.unsqueeze(0))))
arrays["noise17"] = torch.stack(n17_H).numpy().astype(np.float32)
meta["noise17"] = {
    "ids": n17_ids,
    "label": [noise17[t]["result"]["terminal_token"] for t in n17_ids],
    "converged": [bool(noise17[t]["result"]["converged"]) for t in n17_ids],
    "matched_to": [noise17[t]["matched_to"] for t in n17_ids],
    "seq_len": [int(noise17[t]["seq_len"]) for t in n17_ids],
    "initial_frobenius": [float(noise17[t]["target_frobenius"]) for t in n17_ids],
    "reconstruction_cosine": n17_cos,
    "tensor_reconstruction": (
        "Run 17 stores terminal_last_vec and terminal_mean_vec, not the terminal "
        "tensor, so the tensor injected here was rebuilt by repeating "
        "terminal_last_vec across the recorded seq_len positions and rescaling the "
        "whole tensor to target_frobenius. The rebuild is exact rather than "
        "approximate because run 17's own record shows the terminal was "
        "position-collapsed: see "
        "gates['noise17_position_collapse_from_run17_record']."),
}

# --------------------------------------------------- family: nullold (125) ---
log("building per-layer states: nullold (original noise arm, iteration 100)")
old_ids = sorted(nullold_raw)
old_H, old_cos = [], []
for tid in old_ids:
    snap = nullold_raw[tid][-1]
    assert int(snap["iteration"]) == 100
    fro0 = float(nullold_raw[tid][0]["tensor_norm"])
    v = snap["last_vector"]
    H = per_layer_from_terminal(v, int(snap["tensor"].shape[0]), fro0)
    old_H.append(H)
    old_cos.append(float(F.cosine_similarity(H[11].unsqueeze(0), v.unsqueeze(0))))
arrays["nullold"] = torch.stack(old_H).numpy().astype(np.float32)
meta["nullold"] = {
    "ids": old_ids,
    "label_iter100": [nullold_raw[t][-1]["top_tokens"][0][0] for t in old_ids],
    "seq_len": [int(nullold_raw[t][-1]["tensor"].shape[0]) for t in old_ids],
    "initial_frobenius": [float(nullold_raw[t][0]["tensor_norm"]) for t in old_ids],
    "reconstruction_cosine": old_cos,
}

# ----------------------------------------------------- family: clean (125) ---
log("building per-layer states: clean (the same 125 prompts, no injection)")
clean_last, clean_mean, clean_tok = [], [], []
for pid in lang_ids:
    text = PROMPT_LIBRARY[pid]
    toks = model.to_tokens(text)  # prepends BOS, the convention every Stage 1 run used
    with torch.no_grad():
        _, cache = model.run_with_cache(toks, names_filter=lambda n: n in HOOKS)
    clean_last.append(torch.stack([cache[h][0, -1].clone() for h in HOOKS]))
    clean_mean.append(torch.stack([cache[h][0].mean(dim=0).clone() for h in HOOKS]))
    clean_tok.append(int(toks.shape[1]))
arrays["clean_last"] = torch.stack(clean_last).numpy().astype(np.float32)
arrays["clean_mean"] = torch.stack(clean_mean).numpy().astype(np.float32)
meta["clean"] = {
    "ids": lang_ids,
    "n_tokens": clean_tok,
    "n_tokens_matches_loop_seq_len": [clean_tok[i] == pairing[p][0]
                                      for i, p in enumerate(lang_ids)],
}
log(f"  token-count agreement with the loop's recorded sequence length: "
    f"{sum(meta['clean']['n_tokens_matches_loop_seq_len'])}/125")

# ------------------------------------------------- named single states -------
log("building per-layer states: Divine phases, pivot, prolet, pilot noise")
sd = torch.load(os.path.join(LUCIER, "experiments/gpt2_small/output_divine_motion/state_divine.pt"),
                weights_only=True)
A_full, N0 = sd["current_tensor"], float(sd["initial_norm"])
PROMPT_DIVINE = sd["prompt"]


def loop_step(x, target_norm, prompt_tokens):
    cur = norm_to(x, target_norm)
    inject = cur.clone()

    def h(resid, hook, tt=inject):
        resid[0, :, :] = tt
        return resid

    model.add_hook("blocks.0.hook_resid_pre", h)
    try:
        with torch.no_grad():
            _, cache = model.run_with_cache(
                prompt_tokens, names_filter=lambda n: n in HOOKS)
    finally:
        model.reset_hooks()
    return torch.stack([cache[k][0].clone() for k in HOOKS])   # [12, T, 768]


div_tokens = model.to_tokens(PROMPT_DIVINE)
assert div_tokens.shape[1] == A_full.shape[0], "Divine scaffold length mismatch"
stackA = loop_step(A_full, N0, div_tokens)
B_full = stackA[11]
stackB = loop_step(B_full, N0, div_tokens)
A2_full = stackB[11]
An, Bn = norm_to(A_full, N0), norm_to(B_full, N0)
M_full = (An + Bn) / 2.0
stackM = loop_step(M_full, N0, div_tokens)
cosAB = float(F.cosine_similarity(A_full[-1].unsqueeze(0),
                                  norm_to(B_full, N0)[-1].unsqueeze(0)))
cosAA2 = float(F.cosine_similarity(A_full[-1].unsqueeze(0),
                                   norm_to(A2_full, N0)[-1].unsqueeze(0)))
with open(os.path.join(LUCIER, "experiments/gpt2_small/output_divine_motion/bell_anatomy.json")) as fh:
    bell = json.load(fh)
gates["divine_cycle"] = {"cos_A_B": cosAB, "cos_A_ffA": cosAA2,
                         "bell_anatomy_cos_A_B": bell["cosAB"],
                         "bell_anatomy_cos_A_ffA": bell["cosAA2"]}
log(f"  Divine cycle gate: cos(A,B)={cosAB:.8f} (record {bell['cosAB']:.8f}), "
    f"cos(A,f(f(A)))={cosAA2:.8f} (record {bell['cosAA2']:.8f})")
if abs(cosAB - bell["cosAB"]) > 5e-4 or cosAA2 < 0.999999:
    raise SystemExit("GATE FAILED: Divine phase reconstruction does not match bell_anatomy.json")

sp = torch.load(os.path.join(LUCIER, "experiments/gpt2_small/output_divine_motion/state_prolet.pt"),
                weights_only=True)
sn = torch.load(os.path.join(LUCIER, "experiments/gpt2_small/output_divine_motion/state_noise.pt"),
                weights_only=True)
prolet_H = per_layer_from_tensor(norm_to(sp["current_tensor"], float(sp["initial_norm"])))
noise1000_H = per_layer_from_tensor(norm_to(sn["current_tensor"], float(sn["initial_norm"])))

# Naming, stated because it is not what the names suggest. Each entry is the
# per-layer trace of ONE forward pass, named after the vector injected at that
# pass's input. So "phaseA" is the pass whose input is vector A, and what it holds
# at layer 11 is that pass's output, which for this period-2 cycle is vector B;
# "phaseB" holds vector A at layer 11. At layers 0 to 10 neither entry is a phase
# vector at all: they are intermediate residuals of one loop step. Spec section
# 2.2 can be read either way for a period-2 cycle, because its framing sentence
# ("the loop step that produced it") and its numbered steps (splice the state
# itself) name opposite passes once a state is not a fixed point. Nothing in the
# H16a verdict rule depends on the choice: the rule is symmetric in the two
# phases. Finding F16 measured a different quantity again, the single vectors A
# and B scored against every layer's dictionary with no forward pass.
named = {
    "phaseA": stackA[:, -1, :],
    "phaseB": stackB[:, -1, :],
    "pivotM": stackM[:, -1, :],
    "prolet1000": prolet_H,
    "noise1000": noise1000_H,
}
for key, tens in conv.items():
    named[f"convtensor_{key}"] = per_layer_from_terminal(
        tens[-1], tens.shape[0], float(tens.norm()))
named_keys = sorted(named)
arrays["named"] = torch.stack([named[k] for k in named_keys]).numpy().astype(np.float32)
meta["named"] = {
    "keys": named_keys,
    "note": ("convtensor_* are the five committed converged prompt tensors, rescaled "
             "to their own stored norm because no separate iteration-0 norm is stored "
             "for them; they are secondary readings only."),
    "divine_prompt": PROMPT_DIVINE,
    "divine_initial_norm": N0,
    "prolet_initial_norm": float(sp["initial_norm"]),
    "noise1000_initial_norm": float(sn["initial_norm"]),
}

# d_sym is a direction, not a state: probed against every layer's dictionary
# directly, exactly as the phase-aware pilot (finding F16) did. DEVIATION from
# spec section 2.1, which named output_jlens_phase/phase_states.pt as the source:
# the axis is rebuilt here from this script's own on-shell An and Bn instead, and
# phase_states.pt is never opened. The two are not the same axis, because the
# committed one mixes frames (the lucier record's caveat 15). Recorded in the
# results record's deviations list.
d_sym = (An[-1] - Bn[-1])
d_sym = d_sym / d_sym.norm()
arrays["directions"] = torch.stack([d_sym, -d_sym]).numpy().astype(np.float32)
meta["directions"] = {"keys": ["d_sym_plus", "d_sym_minus"],
                      "note": ("The symmetric on-shell flip axis of the Divine cycle, "
                               "both phases rescaled to the loop's starting norm. It is "
                               "a direction, probed against each layer's dictionary "
                               "without any injection.")}

# --------------------------------------------------- reconstruction gate -----
lang_pass = sum(1 for c in lang_cos if c > 0.999)
n17_pass = sum(1 for c in n17_cos if c > 0.999)
old_pass = sum(1 for c in old_cos if c > 0.999)
gates["reconstruction"] = {
    "lang_pass_over_125": lang_pass, "noise17_pass_over_125": n17_pass,
    "nullold_pass_over_125": old_pass, "threshold_cosine": 0.999,
    "required_pass": 85,
}
log(f"  reconstruction gate: lang {lang_pass}/125, noise17 {n17_pass}/125, "
    f"nullold {old_pass}/125 (need 85 in each of the first two)")
if lang_pass < 85 or n17_pass < 85:
    raise SystemExit("GATE FAILED: per-layer state reconstruction does not return the "
                     "stored terminal state at layer 11.")

np.savez_compressed(os.path.join(OUT, "states.npz"), **arrays)
meta["gates"] = gates
meta["array_shapes"] = {k: list(v.shape) for k, v in arrays.items()}
with open(os.path.join(OUT, "states_meta.json"), "w") as fh:
    json.dump(meta, fh, indent=1)
log(f"wrote output/states.npz ({os.path.getsize(os.path.join(OUT,'states.npz'))/1e6:.1f} MB) "
    f"and output/states_meta.json")
log("stage 1 complete")
