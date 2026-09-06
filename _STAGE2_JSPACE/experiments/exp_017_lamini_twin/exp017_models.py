"""EXP_017: the two models, and the exact weight revisions this run loaded.

One place, so that the loop, the lens fit, the J-space probe and the
verification script all load the same weights. A revision is the 40-character
commit identifier of a Hugging Face repository's state at one moment; pinning
it means a later change to the repository cannot silently change what these
scripts load.

Both revisions are the ones `output/model_verification.json` recorded on
2026-09-05 for the weight file each model was actually loaded from:

  twin: MBZUAI/LaMini-GPT-124M at fc740804ff49f50fe3ef871b31eb2d5a5584132c,
    the commit that carries `model.safetensors`, whose SHA-256 is
    0f3781c76d1b983cd98824c490e08da956a9dc6ca33bdd0a54986ebabedc1d91. The
    repository's `main` branch was at 5c67c8c03c08e82d6138ce2a1eddf5317fac3a6b
    on that date and carries the same configuration and tokenizer files, byte
    for byte, so pinning the weight revision changes nothing but the weight
    container the loader picks.
  base: gpt2 at 607a30d783dfa663caf39e06633721c8d4cfcd7e, which is the
    revision `EXP_010b_SPEC.md` section 2 recorded in 2026-07-26.

Nothing here loads a model; the callers do that, passing `revision(which)` to
both `AutoModelForCausalLM.from_pretrained` and
`AutoTokenizer.from_pretrained`.
"""

MODELS = {"twin": "MBZUAI/LaMini-GPT-124M", "base": "gpt2"}

REVISIONS = {"twin": "fc740804ff49f50fe3ef871b31eb2d5a5584132c",
             "base": "607a30d783dfa663caf39e06633721c8d4cfcd7e"}


def name(which):
    """The Hugging Face repository identifier for 'twin' or 'base'."""
    return MODELS[which]


def revision(which):
    """The pinned commit identifier for 'twin' or 'base'."""
    return REVISIONS[which]


def load_hf(which, torch_dtype=None):
    """Load one model and its tokenizer from Hugging Face at the pinned
    revision, and return them as a pair in that order.

    Kept here rather than in each caller so that no script can drift onto a
    different revision by accident.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    repo, rev = MODELS[which], REVISIONS[which]
    kwargs = {} if torch_dtype is None else {"dtype": torch_dtype}
    hf = AutoModelForCausalLM.from_pretrained(repo, revision=rev, **kwargs)
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    return hf, tok
