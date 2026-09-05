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


def load_model():
    """Base GPT-2 Small with no weight processing, so its residual stream is
    numerically identical to the HuggingFace model the lens was fitted on."""
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained_no_processing("gpt2", device="cpu")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_lens():
    """Load the lens and check its digest against LENS_SHA256, so that a
    different file at LENS_PATH (or at EXP016_LENS_PATH) cannot silently
    stand in for the registered instrument."""
    global LENS_SHA256_MEASURED
    from jlens.lens import JacobianLens
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
