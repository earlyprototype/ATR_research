"""Shared machinery for EXP_016: lens-coordinate swaps on base GPT-2 Small.

Every term used here is defined in `_STAGE2_JSPACE/EXP_016_SPEC.md`. In short:
a "lens vector" for a vocabulary token at layer l is the direction in the
768-number residual stream that the Jacobian lens reads as evidence for that
token, and a "swap" exchanges how much of two such directions the residual
stream carries.

Layer l means the residual stream at the output of transformer block l, which
is the transformer_lens hook point `blocks.l.hook_resid_post`.
"""

from __future__ import annotations

import hashlib
import os
import torch

torch.set_num_threads(1)

# Offline mode is not forced. Until 2026-09-05 these scripts set Hugging
# Face's two offline switches on import, which pins the model load to the
# local cache: on the machine that produced the committed records the cache
# already held base GPT-2 Small, but on a fresh checkout with an empty cache
# the load failed instead of fetching the weights. Set the environment
# variable EXP016_OFFLINE to 1 to pin a run to the cache again. The weights
# are the same file either way, so this changes nothing that was measured.
if os.environ.get("EXP016_OFFLINE", "0").strip().lower() not in ("", "0", "false", "no"):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# The lens lives in the gitignored artifacts folder of the Stage 2 tree that
# contains this script; EXP016_LENS_PATH overrides it for another checkout.
LENS_PATH = os.environ.get("EXP016_LENS_PATH") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "artifacts", "jlens_gpt2_small_neuronpedia.pt")
LENS_SHA256 = "d1800a1335ada089ef2e1ec0e4bd4d5bd61e6011eacc31f8618fdb3d10aae762"
JLENS_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
# Set by load_lens() to the digest computed from the file actually loaded,
# so provenance records a measured value rather than the constant above.
LENS_SHA256_MEASURED = None

# The model is pinned the way the lens is. MODEL_REVISION is the commit of
# the Hugging Face repository `gpt2` that the committed runs loaded, read
# from this machine's Hugging Face cache (the file
# `models--gpt2/refs/main`); MODEL_PARAM_SHA256 is the digest of the weights
# that revision produces once transformer_lens has assembled them, so that a
# different revision, a re-tagged repository or a corrupted cache cannot
# stand in for the weights behind the committed records. The digest is the
# one the run log recorded on 2026-09-05; the scheme that produces it is
# `parameter_digest` below. EXP016_MODEL_REVISION overrides the revision and
# EXP016_SKIP_MODEL_DIGEST set to 1 turns the refusal into a warning, both
# for a reader who deliberately wants different weights.
MODEL_NAME = "gpt2"
MODEL_REVISION = os.environ.get(
    "EXP016_MODEL_REVISION", "607a30d783dfa663caf39e06633721c8d4cfcd7e")
MODEL_PARAM_SHA256 = "0684f138b3472aa8c2dca86ac3fddfb29c08473d13b70052dfa87a9b7d88cfe1"
# Set by load_model() to the digest computed from the weights actually
# loaded, so provenance records a measured value rather than the constant.
MODEL_PARAM_SHA256_MEASURED = None


def parameter_digest(model):
    """SHA-256 over the model's ordered parameter tensors: every entry of the
    state dictionary in alphabetical order by name, the name's bytes followed
    by the tensor's bytes. Ordering by name rather than by insertion makes the
    digest independent of the order transformer_lens happens to register its
    modules in."""
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode("utf-8"))
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def load_model():
    """Base GPT-2 Small with no weight processing, so its residual stream is
    numerically identical to the HuggingFace model the lens was fitted on.

    The weights are pinned twice over: the Hugging Face repository revision is
    fixed at MODEL_REVISION, and the assembled parameters are hashed and
    checked against MODEL_PARAM_SHA256, so that weights other than the ones
    behind the committed records raise instead of quietly standing in for
    them."""
    global MODEL_PARAM_SHA256_MEASURED
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained_no_processing(
        MODEL_NAME, device="cpu", revision=MODEL_REVISION)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    digest = parameter_digest(model)
    MODEL_PARAM_SHA256_MEASURED = digest
    if digest != MODEL_PARAM_SHA256:
        msg = (f"model {MODEL_NAME} at revision {MODEL_REVISION} has "
               f"ordered-parameter SHA-256 {digest}, expected "
               f"{MODEL_PARAM_SHA256}; these are not the weights the "
               f"committed records were measured on")
        if os.environ.get("EXP016_SKIP_MODEL_DIGEST", "0").strip().lower() \
                not in ("", "0", "false", "no"):
            print("WARNING: " + msg, flush=True)
        else:
            raise RuntimeError(msg)
    return model


def load_lens():
    """Load the lens and check its digest against LENS_SHA256, so that a
    different file at LENS_PATH (or at EXP016_LENS_PATH) cannot silently
    stand in for the registered instrument."""
    global LENS_SHA256_MEASURED
    try:
        from jlens.lens import JacobianLens
    except ImportError as exc:
        raise ImportError(
            f"the reader package `jlens` is not importable ({exc}). It is the "
            f"reference code of the lens, the repository "
            f"https://github.com/anthropics/jacobian-lens at commit "
            f"{JLENS_COMMIT}. Clone it and install it in place with `git clone "
            f"https://github.com/anthropics/jacobian-lens && cd jacobian-lens "
            f"&& git checkout {JLENS_COMMIT} && pip install -e .`, or put that "
            f"clone on PYTHONPATH. Everything in this experiment uses exactly "
            f"two things from it: `JacobianLens.load(path)`, which reads the "
            f"lens file, and the `.jacobians` list it returns, whose entry for "
            f"a layer is the 768 by 768 matrix this code multiplies by") from exc
    if not os.path.exists(LENS_PATH):
        raise FileNotFoundError(
            f"lens not found at {LENS_PATH}; download gpt2-small from the Hugging "
            f"Face repository neuronpedia/jacobian-lens and point EXP016_LENS_PATH at it")
    with open(LENS_PATH, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    if digest != LENS_SHA256:
        raise RuntimeError(f"lens at {LENS_PATH} has SHA-256 {digest}, "
                           f"expected {LENS_SHA256}")
    LENS_SHA256_MEASURED = digest
    return JacobianLens.load(LENS_PATH)


def positions(mode, n_tokens, mention):
    """Token positions a position mode patches. `mention` is the position of
    the swapped concept's first mention (used by from_mention)."""
    if mode == "all":
        return list(range(n_tokens))
    if mode == "all_no_bos":
        return list(range(1, n_tokens))
    if mode in ("last", "answer_only"):
        return [n_tokens - 1]
    if mode == "from_mention":
        return list(range(mention, n_tokens))
    raise ValueError(mode)


def lens_vectors(lens, model, layer, token_ids):
    """The lens directions for `token_ids` at `layer`: one 768-number column
    per token, namely J_l transposed applied to that token's unembedding
    column. Returns a tensor of shape [768, n_tokens]."""
    J = lens.jacobians[layer].float()          # [768, 768]
    W = model.W_U[:, token_ids].float()        # [768, n_tokens]
    return J.T @ W


def lens_logits_at(lens, model, cache, layer, position):
    """Lens readout at one layer and position: the vocabulary-size vector of
    lens logits, softmax of which is the lens's next-token distribution."""
    h = cache[f"blocks.{layer}.hook_resid_post"][0, position].float()
    J = lens.jacobians[layer].float()
    return model.unembed(model.ln_final((J @ h).view(1, 1, -1)))[0, 0]


def single_token_id(model, text):
    """Token id if `text` is exactly one GPT-2 token, else None."""
    ids = model.to_tokens(text, prepend_bos=False)[0]
    return int(ids[0]) if ids.shape[0] == 1 else None


def first_token_id(model, text):
    """Id of the first GPT-2 token of `text` (used to score multi-token
    answers such as ' North America' on their first token, ' North')."""
    return int(model.to_tokens(text, prepend_bos=False)[0, 0])


@torch.no_grad()
def clean_topk(model, prompt, k=10):
    """Top-k next-token predictions of the unmodified model at the end of
    `prompt`: a list of (token_id, token_string, probability)."""
    toks = model.to_tokens(prompt)
    resid = model(toks, return_type=None, stop_at_layer=model.cfg.n_layers)
    logits = model.unembed(model.ln_final(resid[:, -1:, :]))[0, 0]
    probs = torch.softmax(logits, dim=-1)
    top = torch.topk(probs, k)
    return [(int(i), model.to_string(int(i)), float(p))
            for p, i in zip(top.values, top.indices)]


@torch.no_grad()
def clean_run(model, prompt):
    """Clean next-token log-probabilities and the cache of residual streams,
    for one prompt. Returns (log_probs [vocab], cache, n_tokens)."""
    toks = model.to_tokens(prompt)
    logits, cache = model.run_with_cache(toks)
    return torch.log_softmax(logits[0, -1].float(), dim=-1), cache, toks.shape[1]


def rank_of(logprobs, token_id):
    """1-based rank of `token_id` in the next-token distribution."""
    return int((logprobs > logprobs[token_id]).sum().item()) + 1
