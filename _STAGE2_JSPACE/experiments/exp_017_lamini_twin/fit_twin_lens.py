"""EXP_017 Part 2: fit a Jacobian lens to MBZUAI/LaMini-GPT-124M (the twin).

Follows _STAGE2_JSPACE/experiments/jlens_medium/fit_lens.py, the registered
reference fit, changed only where the model changes (GPT-2 Small architecture,
12 layers, d_model 768, so the lens is fitted for source layers 0..10 with the
final block 11 as the target).

Instrument: anthropics/jacobian-lens, pinned commit
581d398613e5602a5af361e1c34d3a92ea82ba8e, installed editable.
Fitting corpus: WikiText-103 via jlens.examples.load_wikitext_prompts, the same
corpus and loader the reference fit used, frozen to
_STAGE2_JSPACE/artifacts/wikitext_prompts_160.json (gitignored; the derivation
is deterministic, so the file is reproducible from the loader).

Usage:
    python3 fit_twin_lens.py --refresh-prompts
    python3 fit_twin_lens.py --n 5  --dim-batch 16 --tag probe
    python3 fit_twin_lens.py --n 100 --dim-batch 16 --tag twin
"""
import argparse
import json
import logging
import os
import time

import torch

torch.set_num_threads(1)

import transformers

import jlens

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE2 = os.path.abspath(os.path.join(HERE, "..", ".."))
ARTIFACTS = os.path.join(STAGE2, "artifacts")
PROMPTS_JSON = os.path.join(ARTIFACTS, "wikitext_prompts_160.json")
TWIN = "MBZUAI/LaMini-GPT-124M"


def lens_path(n, tag=""):
    suffix = f"_{tag}" if tag else ""
    return os.path.join(ARTIFACTS, f"jlens_lamini_gpt2_124m_{n}{suffix}.pt")


def refresh_prompts():
    """Deterministically regenerate the frozen fitting-corpus list: the first
    160 WikiText-103 train records with at least 600 characters, stream order."""
    from jlens.examples import load_wikitext_prompts
    prompts = load_wikitext_prompts(160, min_chars=600)
    os.makedirs(ARTIFACTS, exist_ok=True)
    json.dump(prompts, open(PROMPTS_JSON, "w"), indent=1)
    print(f"Wrote {len(prompts)} prompts -> {PROMPTS_JSON}")


def load_model(model_path=TWIN):
    tok = transformers.AutoTokenizer.from_pretrained(model_path)
    hf = transformers.AutoModelForCausalLM.from_pretrained(model_path)
    model = jlens.from_hf(hf, tok)
    assert model.n_layers == 12 and model.d_model == 768, (
        f"expected 12 layers / d_model 768, got {model.n_layers} / {model.d_model}")
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--dim-batch", type=int, default=16)
    ap.add_argument("--tag", default="twin")
    ap.add_argument("--model-path", default=TWIN)
    ap.add_argument("--refresh-prompts", action="store_true")
    args = ap.parse_args()
    if args.refresh_prompts:
        refresh_prompts()
        return

    jlens.configure_logging(level=logging.INFO)
    os.makedirs(ARTIFACTS, exist_ok=True)
    prompts = json.load(open(PROMPTS_JSON))[: args.n]
    print(f"torch threads={torch.get_num_threads()} prompts={len(prompts)} "
          f"dim_batch={args.dim_batch} model={args.model_path}", flush=True)

    model, _ = load_model(args.model_path)
    ckpt = os.path.join(ARTIFACTS, f"jlens_lamini_gpt2_124m_{args.n}_{args.tag}.ckpt.pt")
    out = lens_path(args.n, args.tag)

    t0 = time.perf_counter()
    lens = jlens.fit(
        model,
        prompts,
        dim_batch=args.dim_batch,
        max_seq_len=128,
        checkpoint_path=ckpt,
        checkpoint_every=1,
    )
    wall = time.perf_counter() - t0
    lens.save(out)
    print(f"DONE n={args.n} dim_batch={args.dim_batch} wall={wall:.0f}s "
          f"({wall / max(len(prompts), 1):.1f}s/prompt) -> {out}", flush=True)
    print(lens, flush=True)


if __name__ == "__main__":
    main()
