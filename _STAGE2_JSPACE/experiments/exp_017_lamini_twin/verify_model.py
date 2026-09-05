"""EXP_017 model provenance and verification (pre-spec provenance step).

Compares MBZUAI/LaMini-GPT-124M's config with gpt2's field by field, records the
SHA-256 of every downloaded weight file, checks that the TransformerLens
conversion reproduces the Hugging Face model's logits, and records the twin's
response to one instruction in its documented wrapper template.

Both models are loaded at the revision, that is the Hugging Face repository
commit, pinned in exp017_models.py, so a later change on Hugging Face cannot
alter what is verified here.

No hypothesis is tested here. Outputs: output/model_verification.json and
output/tl_conversion_check.json, the second holding the conversion measurement
the results record's section 2.3 reads (the raw, centred, log-probability and
probability differences between the two implementations, and the comparison of
the two models' token-embedding matrices).

Usage:
    python3 verify_model.py
    python3 verify_model.py --conversion-check-only   # rewrite only the second
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(1)

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import exp017_models  # noqa: E402

OUT = HERE / "output"
OUT.mkdir(exist_ok=True)

TWIN = exp017_models.name("twin")
BASE = exp017_models.name("base")
REVISION = {TWIN: exp017_models.revision("twin"),
            BASE: exp017_models.revision("base")}

# The one prompt both the conversion check and the record's section 2.3 use.
TEST_PROMPT = "The implications of quantum entanglement suggest that"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cached_files(repo_id):
    """Every file in the local HF cache snapshot for repo_id, with size+digest."""
    from huggingface_hub import scan_cache_dir
    rows = []
    for repo in scan_cache_dir().repos:
        if repo.repo_id != repo_id:
            continue
        for rev in repo.revisions:
            for f in rev.files:
                p = Path(os.path.realpath(f.file_path))
                rows.append({
                    "file": f.file_name,
                    "revision": rev.commit_hash,
                    "bytes": p.stat().st_size,
                    "sha256": sha256_file(p),
                })
    return sorted(rows, key=lambda r: r["file"])


def load_pair(repo_id):
    """The Hugging Face model and its tokenizer, both at the pinned revision."""
    tok = AutoTokenizer.from_pretrained(repo_id, revision=REVISION[repo_id])
    hf = AutoModelForCausalLM.from_pretrained(repo_id, revision=REVISION[repo_id])
    hf.eval()
    return hf, tok


def compare_hf_to_hooked(hf, tok, test_prompt=TEST_PROMPT):
    """Run one prompt through the Hugging Face model and through its
    TransformerLens conversion, and measure how far the two disagree.

    Loading a Hugging Face model into TransformerLens rewrites the weights, and
    one part of that rewrite subtracts a constant from every output score at
    each position. Subtracting the same number from every score at a position
    leaves every probability unchanged, so the raw score difference is large
    and carries no meaning; the difference after that per-position constant is
    removed is the one that says whether the two implementations agree.

    Returns two dictionaries: the fields the provenance record keeps, and the
    fields output/tl_conversion_check.json keeps. Both come from this one
    measurement, so the two artifacts cannot drift apart.
    """
    from transformer_lens import HookedTransformer
    ht = HookedTransformer.from_pretrained("gpt2", hf_model=hf, tokenizer=tok,
                                           device="cpu")
    ht.eval()
    ids = tok(test_prompt, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        lo_hf = hf(ids).logits[0]
        lo_ht = ht(ids)[0]
    v = min(lo_hf.shape[-1], lo_ht.shape[-1])
    a, b = lo_hf[:, :v], lo_ht[:, :v]
    difference = a - b
    top_hf = tok.decode([int(a[-1].argmax())])
    top_ht = tok.decode([int(b[-1].argmax())])
    verification = {
        "hf_logits_shape": list(lo_hf.shape),
        "hooked_logits_shape": list(lo_ht.shape),
        "compared_vocab_columns": v,
        "max_abs_logit_difference": difference.abs().max().item(),
        "hf_top1_last_position": top_hf,
        "hooked_top1_last_position": top_ht,
        "top1_agree": top_hf == top_ht,
        "cfg": {"n_layers": ht.cfg.n_layers, "d_model": ht.cfg.d_model,
                "d_vocab": ht.cfg.d_vocab, "n_heads": ht.cfg.n_heads},
    }
    conversion = {
        # The raw difference, dominated by the per-position constant.
        "max_abs_raw": difference.abs().max().item(),
        # How nearly that difference is one constant per position: the spread
        # of the difference across the vocabulary at each position, worst
        # position reported. A small number means the offset really is a
        # constant and not a per-token disagreement.
        "per_position_offset_std_max": difference.std(dim=-1).max().item(),
        # The difference once that per-position constant is removed.
        "max_abs_after_centering":
            (difference - difference.mean(dim=-1, keepdim=True)).abs().max().item(),
        # The same disagreement on the two scales that carry meaning.
        "max_abs_logprob": (a.log_softmax(-1) - b.log_softmax(-1)).abs().max().item(),
        "max_abs_prob": (a.softmax(-1) - b.softmax(-1)).abs().max().item(),
        "top5_hf": [tok.decode([int(i)]) for i in a[-1].topk(5).indices],
        "top5_ht": [tok.decode([int(i)]) for i in b[-1].topk(5).indices],
        "d_vocab": int(ht.cfg.d_vocab),
    }
    del ht
    return verification, conversion


def embedding_comparison(hf_twin, hf_base):
    """How far the twin's token-embedding matrix has moved from base's, over
    the rows the two share, and how long the twin's one extra row is beside the
    mean row length of base's vocabulary. A token embedding is the vector a
    model starts from when it reads one vocabulary entry.
    """
    wt = hf_twin.state_dict()["transformer.wte.weight"].float()
    wb = hf_base.state_dict()["transformer.wte.weight"].float()
    shared = wb.shape[0]
    return {
        "wte_shared_rows_rel_frobenius":
            ((wt[:shared] - wb).norm() / wb.norm()).item(),
        "extra_row_norm": wt[shared:].norm().item(),
        "mean_row_norm": wb.norm(dim=1).mean().item(),
    }


def conversion_record(hf_twin, tok_twin, hf_base, tok_base):
    """The whole of output/tl_conversion_check.json, and the per-model fields
    the provenance record keeps, from one pass over both models."""
    verification, conversion = {}, {}
    for tag, hf, tok in (("twin", hf_twin, tok_twin), ("base", hf_base, tok_base)):
        verification[tag], conversion[tag] = compare_hf_to_hooked(hf, tok)
        if tag == "twin":
            conversion.update(embedding_comparison(hf_twin, hf_base))
    return verification, conversion


def write_conversion_check(conversion):
    path = OUT / "tl_conversion_check.json"
    path.write_text(json.dumps(conversion, indent=2))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conversion-check-only", action="store_true",
                    help="load both models, rewrite output/"
                         "tl_conversion_check.json, and leave "
                         "output/model_verification.json alone")
    args = ap.parse_args()

    if args.conversion_check_only:
        hf_twin, tok_twin = load_pair(TWIN)
        hf_base, tok_base = load_pair(BASE)
        _, conversion = conversion_record(hf_twin, tok_twin, hf_base, tok_base)
        path = write_conversion_check(conversion)
        print(json.dumps(conversion, indent=2))
        print(f"\nSaved -> output/{path.name}")
        return

    rec = {
        "experiment": "EXP_017",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python": sys.version.split()[0],
        },
    }
    import transformer_lens
    rec["versions"]["transformer_lens"] = getattr(
        transformer_lens, "__version__",
        __import__("importlib.metadata", fromlist=["version"]).version("transformer_lens"))
    try:
        import jlens
        rec["versions"]["jlens"] = __import__(
            "importlib.metadata", fromlist=["version"]).version("jlens")
    except Exception as exc:  # instrument optional for this step
        rec["versions"]["jlens"] = f"unavailable: {exc}"

    # ---- 1. config comparison, field by field --------------------------------
    cfg_twin = AutoConfig.from_pretrained(TWIN, revision=REVISION[TWIN]).to_dict()
    cfg_base = AutoConfig.from_pretrained(BASE, revision=REVISION[BASE]).to_dict()
    keys = sorted(set(cfg_twin) | set(cfg_base))
    diffs, same = {}, {}
    for k in keys:
        a, b = cfg_twin.get(k, "<absent>"), cfg_base.get(k, "<absent>")
        if a == b:
            same[k] = a
        else:
            diffs[k] = {"twin": a, "base": b}
    rec["config_comparison"] = {
        "n_fields": len(keys),
        "n_identical": len(same),
        "differing": diffs,
        "architecture_fields": {
            k: {"twin": cfg_twin.get(k), "base": cfg_base.get(k)}
            for k in ("n_layer", "n_embd", "n_head", "n_positions", "n_ctx",
                      "vocab_size", "activation_function", "layer_norm_epsilon",
                      "architectures", "model_type", "scale_attn_weights",
                      "reorder_and_upcast_attn", "scale_attn_by_inverse_layer_idx")
            if k in cfg_twin or k in cfg_base
        },
    }

    # ---- 2. weight-file digests ---------------------------------------------
    rec["files"] = {TWIN: cached_files(TWIN), BASE: cached_files(BASE)}

    # ---- 3. load both models, compare weight layout --------------------------
    hf_twin, tok_twin = load_pair(TWIN)
    hf_base, tok_base = load_pair(BASE)

    sd_t, sd_b = hf_twin.state_dict(), hf_base.state_dict()
    rec["state_dict"] = {
        "twin_n_tensors": len(sd_t), "base_n_tensors": len(sd_b),
        "keys_only_in_twin": sorted(set(sd_t) - set(sd_b)),
        "keys_only_in_base": sorted(set(sd_b) - set(sd_t)),
        "shape_mismatches": {k: [list(sd_t[k].shape), list(sd_b[k].shape)]
                             for k in set(sd_t) & set(sd_b)
                             if sd_t[k].shape != sd_b[k].shape},
        "twin_params": int(sum(p.numel() for p in hf_twin.parameters())),
        "base_params": int(sum(p.numel() for p in hf_base.parameters())),
    }
    rec["tokenizer"] = {
        "twin_vocab_size": len(tok_twin), "base_vocab_size": len(tok_base),
        "identical_on_probe": None,
    }
    probe_texts = ["The capital of France is Paris, a city on the Seine.",
                   "NATO EU UN ASEAN BRICS G7 IMF WTO",
                   "It was the best of times, it was the"]
    same_tok = all(tok_twin(t)["input_ids"] == tok_base(t)["input_ids"]
                   for t in probe_texts)
    rec["tokenizer"]["identical_on_probe"] = bool(same_tok)

    # weight-identity check: how much did fine-tuning move each block?
    shared = [k for k in set(sd_t) & set(sd_b) if sd_t[k].shape == sd_b[k].shape]
    rel = {}
    for k in sorted(shared):
        d = (sd_t[k].float() - sd_b[k].float()).norm().item()
        n = sd_b[k].float().norm().item()
        rel[k] = round(d / max(n, 1e-12), 6)
    rec["weight_drift_relative_frobenius"] = {
        "n_shared_tensors": len(shared),
        "mean": round(sum(rel.values()) / len(rel), 6),
        "max": max(rel.items(), key=lambda kv: kv[1]),
        "min": min(rel.items(), key=lambda kv: kv[1]),
        "per_tensor_sample": {k: rel[k] for k in list(rel)[:8]},
        "wte": rel.get("transformer.wte.weight"),
        "wpe": rel.get("transformer.wpe.weight"),
    }

    # ---- 4. TransformerLens conversion check ---------------------------------
    # One measurement feeds both artifacts, so they cannot disagree.
    out, conversion = conversion_record(hf_twin, tok_twin, hf_base, tok_base)
    rec["transformerlens_conversion"] = out

    # ---- 5. instruction sanity, documented wrapper template ------------------
    instruction = "Explain in one sentence why the sky appears blue."
    wrapper = ("Below is an instruction that describes a task. "
               "Write a response that appropriately completes the request.\n\n"
               "### Instruction:\n%s\n\n### Response:" % instruction)
    enc = tok_twin(wrapper, return_tensors="pt")
    with torch.no_grad():
        gen = hf_twin.generate(**enc, max_new_tokens=60, do_sample=False,
                               pad_token_id=tok_twin.eos_token_id)
    full = tok_twin.decode(gen[0], skip_special_tokens=True)
    rec["instruction_sanity"] = {
        "template": wrapper,
        "instruction": instruction,
        "decoding": "greedy, max_new_tokens=60, do_sample=False",
        "response_verbatim": full[len(wrapper):],
        "full_output_verbatim": full,
    }
    # base gpt2 on the same wrapper, for contrast
    enc_b = tok_base(wrapper, return_tensors="pt")
    with torch.no_grad():
        gen_b = hf_base.generate(**enc_b, max_new_tokens=60, do_sample=False,
                                 pad_token_id=tok_base.eos_token_id)
    full_b = tok_base.decode(gen_b[0], skip_special_tokens=True)
    rec["instruction_sanity"]["base_response_verbatim"] = full_b[len(wrapper):]

    (OUT / "model_verification.json").write_text(json.dumps(rec, indent=2))
    conversion_path = write_conversion_check(conversion)
    print(json.dumps({k: v for k, v in rec.items()
                      if k not in ("files",)}, indent=2)[:6000])
    print(f"\nSaved -> output/model_verification.json and "
          f"output/{conversion_path.name}")


if __name__ == "__main__":
    main()
