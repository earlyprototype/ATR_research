"""EXP_017 model provenance and verification (pre-spec provenance step).

Compares MBZUAI/LaMini-GPT-124M's config with gpt2's field by field, records the
SHA-256 of every downloaded weight file, checks that the TransformerLens
conversion reproduces the Hugging Face model's logits, and records the twin's
response to one instruction in its documented wrapper template.

No hypothesis is tested here. Output: output/model_verification.json
"""
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
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)

TWIN = "MBZUAI/LaMini-GPT-124M"
BASE = "gpt2"


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


def main():
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
    cfg_twin = AutoConfig.from_pretrained(TWIN).to_dict()
    cfg_base = AutoConfig.from_pretrained(BASE).to_dict()
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
    tok_twin = AutoTokenizer.from_pretrained(TWIN)
    hf_twin = AutoModelForCausalLM.from_pretrained(TWIN)
    tok_base = AutoTokenizer.from_pretrained(BASE)
    hf_base = AutoModelForCausalLM.from_pretrained(BASE)
    hf_twin.eval(); hf_base.eval()

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
    from transformer_lens import HookedTransformer
    out = {}
    for tag, hf, tok in ((("twin"), hf_twin, tok_twin), (("base"), hf_base, tok_base)):
        ht = HookedTransformer.from_pretrained("gpt2", hf_model=hf, tokenizer=tok,
                                               device="cpu")
        ht.eval()
        test_prompt = "The implications of quantum entanglement suggest that"
        ids = tok(test_prompt, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            lo_hf = hf(ids).logits[0]
            lo_ht = ht(ids)[0]
        v = min(lo_hf.shape[-1], lo_ht.shape[-1])
        maxabs = (lo_hf[:, :v] - lo_ht[:, :v]).abs().max().item()
        top_hf = tok.decode([int(lo_hf[-1].argmax())])
        top_ht = tok.decode([int(lo_ht[-1].argmax())])
        out[tag] = {
            "hf_logits_shape": list(lo_hf.shape),
            "hooked_logits_shape": list(lo_ht.shape),
            "compared_vocab_columns": v,
            "max_abs_logit_difference": maxabs,
            "hf_top1_last_position": top_hf,
            "hooked_top1_last_position": top_ht,
            "top1_agree": top_hf == top_ht,
            "cfg": {"n_layers": ht.cfg.n_layers, "d_model": ht.cfg.d_model,
                    "d_vocab": ht.cfg.d_vocab, "n_heads": ht.cfg.n_heads},
        }
        if tag == "twin":
            # ---- 5. instruction sanity, documented wrapper template ----------
            instruction = "Explain in one sentence why the sky appears blue."
            wrapper = ("Below is an instruction that describes a task. "
                       "Write a response that appropriately completes the request.\n\n"
                       "### Instruction:\n%s\n\n### Response:" % instruction)
            enc = tok(wrapper, return_tensors="pt")
            with torch.no_grad():
                gen = hf.generate(**enc, max_new_tokens=60, do_sample=False,
                                  pad_token_id=tok.eos_token_id)
            full = tok.decode(gen[0], skip_special_tokens=True)
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
    rec["transformerlens_conversion"] = out

    (OUT / "model_verification.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps({k: v for k, v in rec.items()
                      if k not in ("files",)}, indent=2)[:6000])
    print("\nSaved -> output/model_verification.json")


if __name__ == "__main__":
    main()
