"""POST-HOC diagnostic for EXP_010c-3b item 1 — NOT pre-registered.

Why it exists: the registered statistic S1 samples *isotropic* directions, but
real residual states are strongly anisotropic. S1 therefore bounds the decoder's
reach over generic directions and cannot exclude the possibility that the funnel
tokens win the argmax across the region of state space these loops actually
occupy. This script asks the narrower, more relevant question:

    what does the logit-lens readout at extract layer j return for the model's
    OWN natural states at layer j, with no ATR loop at all?

If natural layer-j states already decode to the funnel token of the arm that
extracts at j, then that arm's "single prompt-independent terminal" is largely
what the readout says at that layer, independent of the loop. If natural states
decode to varied, ordinary tokens, the funnel is not a generic property of
reading at that layer.

Labelled post-hoc throughout and carries NO verdict weight against the
pre-registered readings in EXP_010c3b_SPEC.md §1.

Usage: python analyze_natural_decode.py --model-path DIR
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

# Extract layers used by the in-fill arms, with the funnel token observed there.
LAYERS = {15: "X1015 ' Fas'", 17: "X817 ' GOP' / X1017 ' Bhar'",
          19: "X819 \"'d\" / X1019 '…)'", 21: "I5/I7/I9/I11 + O8/A4"}


def main():
    """Decode natural resid states at each extract layer and write the report."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    args = ap.parse_args()

    from run_exp010c import _load_medium_from_local
    from derive_prompts import select_subset

    model = _load_medium_from_local(args.model_path)
    model.eval()
    prompts = select_subset(25)

    report = {}
    for layer in sorted(LAYERS):
        hook = f"blocks.{layer}.hook_resid_post"
        last_toks, mean_toks = Counter(), Counter()
        for rec in prompts:
            with torch.no_grad():
                _, cache = model.run_with_cache(rec["prompt"],
                                                names_filter=lambda n, h=hook: n == h)
                t = cache[hook][0]
                for vec, ctr in ((t[-1, :], last_toks), (t.mean(dim=0), mean_toks)):
                    logits = model.ln_final(vec) @ model.W_U + model.b_U
                    ctr[model.tokenizer.decode([int(logits.argmax())])] += 1
        report[layer] = {"note": LAYERS[layer],
                         "natural_last_position": dict(last_toks.most_common()),
                         "natural_mean_position": dict(mean_toks.most_common())}
        print(f"\nlayer {layer}  ({LAYERS[layer]})")
        print(f"  last-position decode: {dict(last_toks.most_common(6))}")
        print(f"  mean-position decode: {dict(mean_toks.most_common(6))}")

    out = HERE / "output" / "natural_decode_posthoc.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
