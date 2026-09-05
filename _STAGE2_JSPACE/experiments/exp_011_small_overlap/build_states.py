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
# directly, exactly as the phase-aware pilot (finding F16) did.
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
