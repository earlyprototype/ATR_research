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

import os
import torch

torch.set_num_threads(1)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

LENS_PATH = ("/home/user/ATR_research/_STAGE2_JSPACE/artifacts/"
             "jlens_gpt2_small_neuronpedia.pt")
LENS_SHA256 = "d1800a1335ada089ef2e1ec0e4bd4d5bd61e6011eacc31f8618fdb3d10aae762"
JLENS_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"


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
    from jlens.lens import JacobianLens
    return JacobianLens.load(LENS_PATH)


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
