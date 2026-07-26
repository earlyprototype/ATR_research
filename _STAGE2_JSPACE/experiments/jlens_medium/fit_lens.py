# Medium J-lens track (issue #15) — instrument fit for gpt2-medium.
# P0-1/P0-2 Medium edition per RUNBOOK_JLENS_MEDIUM.md §2 / RUNBOOK_PHASE0.md.
#
# Instrument: anthropics/jacobian-lens, pinned commit
#   581d398613e5602a5af361e1c34d3a92ea82ba8e ("Initial release"),
# cloned to _STAGE2_JSPACE/instrument/jacobian-lens (gitignored), pip install -e.
#
# What the estimator actually does (read from jlens/fitting.py before running):
#   J_l = mean over prompts and source positions p of
#         sum_{p' in valid target positions} d h_target[p'] / d h_l[p],
#   computed with one forward pass per prompt (prompt replicated dim_batch
#   times along the batch axis) and ceil(d_model / dim_batch) backward passes,
#   each backward filling dim_batch rows of J_l via one-hot cotangents at every
#   valid target position at once. Positions 0..15 are skipped (attention
#   sinks) and the final position (no next-token target) is excluded.
#   target_layer defaults to the final block (23 for gpt2-medium), so the lens
#   is fitted for source layers 0..22; "layer l" everywhere means the residual
#   at the OUTPUT of block l (transformer_lens blocks.l.hook_resid_post).
#
# Fitting corpus: WikiText-103 via jlens.examples.load_wikitext_prompts — the
# corpus the paper's released lenses use ("Salesforce-wikitext" in the lens
# filenames). DEVIATION from issue #15's wording: the repo's data/ directory
# ships evaluation prompt sets only, not the fitting corpus; fitting on the
# eval sets would contaminate the validation gate, so the repo's own fitting
# loader is used instead. The fetched prompt list is frozen at
# artifacts/wikitext_prompts_160.json (gitignored dir; derivation is
# deterministic: first 160 train records with >=600 chars, stream order).
# Fitting set = prompts[0:N_PROMPTS].
#
# Usage: python3 fit_lens.py [--n 100] [--dim-batch 8] [--tag medium]
# Resumable via the jlens checkpoint (atomic save every prompt).

import argparse
import json
import logging
import os
import time

import torch
import transformers

import jlens

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE2 = os.path.abspath(os.path.join(HERE, "..", ".."))
ARTIFACTS = os.path.join(STAGE2, "artifacts")
MODEL_PATH = (
    "/tmp/claude-0/-home-user/cd28417a-3486-5d42-833b-768f561daa85/"
    "scratchpad/gpt2-medium"
)
PROMPTS_JSON = os.path.join(ARTIFACTS, "wikitext_prompts_160.json")


def load_model():
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)
    hf = transformers.AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    model = jlens.from_hf(hf, tok)
    assert model.n_layers == 24 and model.d_model == 1024
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--dim-batch", type=int, default=8)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    jlens.configure_logging(level=logging.INFO)
    os.makedirs(ARTIFACTS, exist_ok=True)
    prompts = json.load(open(PROMPTS_JSON))[: args.n]

    model, _ = load_model()
    suffix = f"_{args.tag}" if args.tag else ""
    ckpt = os.path.join(ARTIFACTS, f"jlens_gpt2_medium_{args.n}{suffix}.ckpt.pt")
    out = os.path.join(ARTIFACTS, f"jlens_gpt2_medium_{args.n}{suffix}.pt")

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
    print(f"DONE n={args.n} dim_batch={args.dim_batch} wall={wall:.0f}s -> {out}")
    print(lens)


if __name__ == "__main__":
    main()
