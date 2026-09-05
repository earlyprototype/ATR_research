"""Shared harness for EXP_018: the ATR loop ported to Qwen3-1.7B.

EXP_018 ports the registered activation-transfer-resonance loop (inject the
model's own deep residual state back into its layer-0 entry, over and over,
at a fixed loudness) from GPT-2 to a small modern chat model. Spec of record:
`_STAGE2_JSPACE/EXP_018_SPEC.md`. Register rows: EXP_018, H19, H19a, H19b.

Two conventions differ from the registered GPT-2 loop, both spec'd:

1. The model is loaded through TransformerLens 3.8.1's bridge loader
   (`TransformerBridge.boot_transformers`) WITHOUT `enable_compatibility_mode`.
   The bridge wraps the Hugging Face model in place and exposes the registered
   hook names; compatibility mode folds the normalisation weights and was
   measured to need more memory than this machine has. Verified here by two
   round trips: injecting the recorded natural layer-0 entry reproduces the
   clean logits exactly, and `ln_final` then `W_U` on the extracted state
   reproduces the model's own logits exactly.

2. The loudness convention is `natural_i` with position 0 excluded from the
   norm: the fed-back tensor is rescaled so that its Frobenius norm over
   token positions 1 and later equals the natural layer-0 entry norm over the
   same positions, measured on that prompt's own clean pass. One scalar is
   applied to every position, position 0 included.
"""

from __future__ import annotations

import json
import math
import os
import resource
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

torch.set_num_threads(1)

MODEL_NAME = "Qwen/Qwen3-1.7B"


# --------------------------------------------------------------------------
# housekeeping
# --------------------------------------------------------------------------

def peak_gb() -> float:
    """Peak resident set size of this process, in gigabytes."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576


def rss_gb() -> float:
    """Current resident set size of this process, in gigabytes."""
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1048576
    return float("nan")


def free_gb() -> float:
    """Memory the whole machine still has available, in gigabytes."""
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) / 1048576
    return float("nan")


def versions() -> dict:
    """Version record for the results file."""
    import importlib.metadata as md
    import sys
    out = {"python": sys.version.split()[0]}
    for pkg in ("torch", "transformers", "transformer_lens", "jlens", "numpy", "scipy"):
        try:
            out[pkg] = md.version(pkg)
        except Exception:
            out[pkg] = "not installed"
    return out


# --------------------------------------------------------------------------
# which copy of the weights
# --------------------------------------------------------------------------

def hub_cache_dir() -> Path:
    """The Hugging Face cache directory this machine downloads model files into.

    Follows the same environment variables the Hugging Face hub library
    follows, so a run and a later analysis of that run look in one place.
    """
    for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(var):
            return Path(os.environ[var])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def model_cache_dir(model: str = MODEL_NAME) -> Path:
    """The cache directory holding every downloaded version of one model."""
    return hub_cache_dir() / ("models--" + model.replace("/", "--"))


def resolve_revision(revision: str | None = None, model: str = MODEL_NAME) -> str:
    """The revision of the weights a load of `model` on this machine resolves to.

    A revision is the 40-character commit identifier that names one exact
    version of a model's files on the Hugging Face hub. Returns `revision`
    unchanged when one is given; otherwise the revision the cache's
    `refs/main` pointer names, which is the one an unpinned load follows;
    otherwise the only revision in the cache when there is exactly one.
    Raises `FileNotFoundError`, naming the directory it searched, when it
    cannot decide, because the alphabetically last revision in the cache is
    not necessarily the one a run used.
    """
    if revision:
        return revision
    root = model_cache_dir(model)
    ref = root / "refs" / "main"
    if ref.exists():
        return ref.read_text().strip()
    snaps = sorted(p.name for p in (root / "snapshots").glob("*")
                   if p.is_dir()) if (root / "snapshots").is_dir() else []
    if len(snaps) == 1:
        return snaps[0]
    if not snaps:
        raise FileNotFoundError(
            f"no cached copy of {model} under {root}, so the revision of the "
            f"weights cannot be resolved on this machine")
    raise FileNotFoundError(
        f"{len(snaps)} cached revisions of {model} under {root} and no "
        f"refs/main pointer to choose between them; pass the revision "
        f"explicitly, one of: {', '.join(snaps)}")


def snapshot_dir(revision: str, model: str = MODEL_NAME) -> Path:
    """The directory holding the files of one exact revision of the weights."""
    path = model_cache_dir(model) / "snapshots" / revision
    if not path.is_dir():
        raise FileNotFoundError(
            f"revision {revision} of {model} is not in the local cache; "
            f"looked in {path}")
    return path


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

def load_model(dtype=torch.float32, revision: str | None = None):
    """Boot Qwen3-1.7B through the TransformerLens bridge, no compatibility mode.

    `revision` pins the load to one exact version of the model's files on the
    Hugging Face hub, which is what a stage must do when it has to match an
    earlier stage's weights. Without it the load follows the cache pointer
    `refs/main`, exactly as `resolve_revision()` does, and that pointer can
    move between one stage and the next; in that case call `resolve_revision()`
    after this returns, not before, and record what it says.
    """
    from transformer_lens.model_bridge import TransformerBridge
    kwargs = {"revision": revision} if revision else {}
    model = TransformerBridge.boot_transformers(
        MODEL_NAME, device="cpu", dtype=dtype, **kwargs)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def hook_names(model):
    """The registered injection and extraction points for this model."""
    return ("blocks.0.hook_resid_pre",
            f"blocks.{model.cfg.n_layers - 1}.hook_resid_post")


def tokenise(model, text: str) -> torch.Tensor:
    """Token ids for `text`, shape [1, n_tokens]. Qwen adds no start token; the
    ids are built explicitly so nothing is prepended behind our back."""
    ids = model.tokenizer(text, add_special_tokens=False, return_tensors="pt")
    return ids.input_ids


def chat_wrap(model, text: str) -> str:
    """The same text as a single user turn, thinking mode off."""
    return model.tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)


def make_injection_hook(tensor: torch.Tensor):
    """Overwrite the whole residual tensor at the hooked point."""
    def hook(resid, hook, t=tensor):
        resid[0, :, :] = t.to(resid.dtype)
        return resid
    return hook


# --------------------------------------------------------------------------
# readout
# --------------------------------------------------------------------------

@torch.no_grad()
def readout(model, vec: torch.Tensor, k: int = 5) -> dict:
    """Decode one residual vector: `ln_final` then `W_U`, the engine's convention.

    Returns the top-k token strings and probabilities, the top-1 minus top-2
    logit margin, and the full-vocabulary softmax entropy in nats.
    """
    wu = model.W_U
    logits = (model.ln_final(vec.to(wu.dtype).unsqueeze(0)) @ wu).squeeze(0).float()
    probs = torch.softmax(logits, dim=-1)
    top_p, top_i = torch.topk(probs, k)
    top_logits = logits[top_i]
    return {
        "top_token_ids": [int(i) for i in top_i],
        "top_token_strings": [model.tokenizer.decode([int(i)]) for i in top_i],
        "top_token_probs": [float(p) for p in top_p],
        "top_logit_margin": float(top_logits[0] - top_logits[1]),
        "entropy": float(-(probs * probs.clamp_min(1e-12).log()).sum()),
    }


# --------------------------------------------------------------------------
# natural loudness
# --------------------------------------------------------------------------

@torch.no_grad()
def natural_profile(model, tokens: torch.Tensor) -> dict:
    """One clean, injection-free pass; the per-layer entry loudness it records.

    For every layer the norm is reported three ways so the position-0 exclusion
    can be judged: over every position (the Frobenius norm the registered GPT-2
    loop uses), over position 0 alone, and over positions 1 and later (the norm
    this port pegs its loudness to).
    """
    n_layers = model.cfg.n_layers
    want = {f"blocks.{i}.hook_resid_pre" for i in range(n_layers)}
    want.add(f"blocks.{n_layers - 1}.hook_resid_post")
    _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in want)
    prof = {}
    for name in sorted(want):
        t = cache[name][0].float()
        prof[name] = {
            "all": float(t.norm()),
            "pos0": float(t[0].norm()),
            "excl0": float(t[1:].norm()),
            "per_position": [round(float(x), 4) for x in t.norm(dim=-1)],
        }
    del cache
    return prof


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------

def position_collapse(t: torch.Tensor) -> tuple[float, float]:
    """Mean pairwise cosine between the token positions of a [T, d] tensor.

    Returns (all positions, positions 1 and later). A value of 1.0 means every
    position holds the same direction; 0.0 means they are unrelated on average.
    Returns NaN where there are fewer than two positions to compare.
    """
    def _mean_offdiag(x):
        if x.shape[0] < 2:
            return float("nan")
        u = x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)
        sim = u @ u.T
        n = x.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool)
        return float(sim[mask].mean())
    t = t.float()
    return _mean_offdiag(t), _mean_offdiag(t[1:])


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).clamp(-1, 1))


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------

@dataclass
class LoopConfig:
    """Every loop parameter, so the results file can carry them verbatim."""
    max_iter: int = 200
    threshold: float = 0.999
    patience: int = 3
    check_every: int = 2
    check_start: int = 10
    gate_lag: int = 2
    seed: int = 42
    record_every: int = 1


@torch.no_grad()
def run_loop(model, tokens: torch.Tensor, cfg: LoopConfig, verbose=False) -> dict:
    """The gated ATR loop at natural loudness with position 0 out of the norm.

    Injects at `blocks.0.hook_resid_pre`, extracts at the last block's
    `hook_resid_post`, and stops when the cosine between the mean position
    vector now and the one two iterations back stays above `threshold` for
    `patience` consecutive checks. Mirrors `atr_engine2.run_atr_gated`
    (gate_lag=2) except for the position-0-excluded rescale and the
    per-iteration diagnostic trace.
    """
    # The same guard `atr_engine2.run_atr_gated` enforces: before iteration
    # `gate_lag` the buffer holds fewer states than the lag needs, so a check
    # scheduled earlier would compare against a nearer state than it claims and
    # could count that malformed comparison towards the patience streak.
    if cfg.gate_lag < 1:
        raise ValueError(f"gate_lag must be >= 1, got {cfg.gate_lag}")
    if cfg.check_start < cfg.gate_lag:
        raise ValueError(
            f"check_start ({cfg.check_start}) must be >= gate_lag "
            f"({cfg.gate_lag}) for the lagged comparison to be well-formed")

    torch.manual_seed(cfg.seed)
    inject_name, extract_name = hook_names(model)
    natural_pre_name = "blocks.0.hook_resid_pre"

    _, cache = model.run_with_cache(
        tokens, names_filter=lambda n: n in (inject_name, extract_name))
    pre0 = cache[natural_pre_name][0].float().clone()
    x = cache[extract_name][0].float().clone()
    del cache

    n_pos = x.shape[0]
    if n_pos < 2:
        raise ValueError("this convention needs at least two token positions")
    target_norm = float(pre0[1:].norm())          # natural loudness, position 0 excluded
    natural_all = float(pre0.norm())
    natural_pos0 = float(pre0[0].norm())
    seed_norm_at_j = float(x.norm())
    seed_norm_excl0 = float(x[1:].norm())

    trace = []
    mean_hist = [x.mean(dim=0).clone()]
    tensor_hist = [x.clone()]
    consecutive = 0
    lock_in = None
    final_cos = 1.0        # replaced by a real measurement on the first iteration
    i = 0

    for i in range(1, cfg.max_iter + 1):
        cur = float(x[1:].norm())
        if cur > 0:
            x = x * (target_norm / cur)
        model.add_hook(inject_name, make_injection_hook(x))
        try:
            _, cache = model.run_with_cache(
                tokens, names_filter=lambda n: n == extract_name)
        finally:
            model.reset_hooks()
        x = cache[extract_name][0].float().clone()
        del cache

        mean_vec = x.mean(dim=0)
        cos_mean_lag = _cos(mean_vec, mean_hist[0])          # lag = gate_lag once i >= gate_lag
        cos_mean_lag1 = _cos(mean_vec, mean_hist[-1])
        cos_tensor_lag = _cos(x, tensor_hist[0])
        pc_all, pc_ex0 = position_collapse(x)

        if i % cfg.record_every == 0 or i == cfg.max_iter:
            trace.append({
                "iteration": i,
                "cos_mean_lag2": round(cos_mean_lag, 6),
                "cos_mean_lag1": round(cos_mean_lag1, 6),
                "cos_tensor_lag2": round(cos_tensor_lag, 6),
                "pos_collapse_all": round(pc_all, 6),
                "pos_collapse_excl0": round(pc_ex0, 6),
                "norm_all": round(float(x.norm()), 4),
                "norm_excl0": round(float(x[1:].norm()), 4),
                "norm_pos0": round(float(x[0].norm()), 4),
            })

        # The reported lag-2 cosine is the most recent one, whether or not this
        # iteration was a scheduled convergence check. Updating it only on
        # checks left the initial 1.0 standing whenever the run ended between
        # checks, which is what the probe stage does.
        final_cos = cos_mean_lag

        if i >= cfg.check_start and i % cfg.check_every == 0:
            consecutive = consecutive + 1 if cos_mean_lag > cfg.threshold else 0
            if verbose:
                print(f"    iter {i:>4}: cos={cos_mean_lag:.6f} streak={consecutive} "
                      f"collapse={pc_all:.4f}", flush=True)
            if consecutive >= cfg.patience:
                lock_in = i
                break

        mean_hist.append(mean_vec.clone())
        if len(mean_hist) > cfg.gate_lag:
            mean_hist.pop(0)
        tensor_hist.append(x.clone())
        if len(tensor_hist) > cfg.gate_lag:
            tensor_hist.pop(0)

    pc_all, pc_ex0 = position_collapse(x)
    r = readout(model, x[-1])
    return {
        "n_tokens": int(n_pos),
        "lock_in_iter": lock_in,
        "converged": lock_in is not None,
        "n_iters": i,
        "final_cos_mean_lag2": round(final_cos, 6),
        "pos_collapse_all_terminal": round(pc_all, 6),
        "pos_collapse_excl0_terminal": round(pc_ex0, 6),
        "target_norm_natural_excl0": round(target_norm, 4),
        "natural_norm_all": round(natural_all, 4),
        "natural_norm_pos0": round(natural_pos0, 4),
        "natural_pos0_over_excl0": round(natural_pos0 / target_norm, 6),
        "seed_norm_at_j_all": round(seed_norm_at_j, 4),
        "seed_norm_at_j_excl0": round(seed_norm_excl0, 4),
        "seed_over_natural_excl0": round(seed_norm_excl0 / target_norm, 4),
        "readout": r,
        "trace": trace,
        "terminal_tensor": x,
    }
