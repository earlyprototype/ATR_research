"""EXP_010c-3b item 1 — are the single-token funnels a decode-geometry artifact?

Pre-registered spec: ../../EXP_010c3b_SPEC.md §1. Pure analysis of the state
dict; no forward passes, no HookedTransformer, no network.

GPT-2 ties its unembedding to `wte`, so a terminal decodes as
    argmax_t  ln_final(v) . wte[t]
A token with a large wte row norm, or one well aligned with the direction
typical states point in, can win that argmax for a wide range of inputs. If the
EXP_010c-3 funnel tokens are such tokens, "the loop settles on one token"
partly describes the decoder rather than the dynamics.

Statistics (fixed in the spec before running):
  S1 random-direction argmax census — N isotropic Gaussian directions through
     the real ln_final, decoded; share of directions won per token.
     Note: LayerNorm is scale-invariant, so the input norm is immaterial; the
     spec's "scaled to a typical residual norm" is a no-op and is recorded as
     such. Caveat: isotropic directions are not distributed like real residual
     states, so S1 measures decoder reach over generic directions, not over the
     loop's actual state distribution.
  S2 percentile of ||wte[t]|| within the full vocabulary.
  S3 percentile of cos(wte[t], mean-row direction) within the full vocabulary.

Usage: python analyze_funnel_geometry.py --model-path DIR [--n 10000]
"""

import argparse
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
SEED = 42

# Token sets fixed in EXP_010c3b_SPEC.md §1 — no post-hoc additions.
FUNNEL = ["oooooooo", "<|endoftext|>", " GOP", "'d", " Fas", " Bhar"]
FUNNEL_EXTRA = ["…)", " […]"]          # X1019's pair, reported separately
WORD = [" until", " forever", " since", " simultaneously", " halfway", " rant"]


def load_pieces(model_path):
    """wte, ln_f gamma/beta from the raw state dict (handles both key layouts)."""
    sd = torch.load(Path(model_path) / "pytorch_model.bin", map_location="cpu",
                    weights_only=True)

    def get(*names):
        for n in names:
            if n in sd:
                return sd[n]
        raise KeyError(f"none of {names} in state dict")

    wte = get("transformer.wte.weight", "wte.weight").float()
    g = get("transformer.ln_f.weight", "ln_f.weight").float()
    b = get("transformer.ln_f.bias", "ln_f.bias").float()
    return wte, g, b


def id_to_string(model_path, vocab_size):
    """Exact reverse map, built the same way the terminals were produced
    (tokenizer.decode([id])), so string matching is exact rather than guessed."""
    from transformers import GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained(str(model_path))
    return [tok.decode([i]) for i in range(vocab_size)]


def census(wte, g, b, n, seed=SEED, chunk=1000):
    """S1: argmax token for n isotropic Gaussian directions through ln_final."""
    d = wte.shape[1]
    gen = torch.Generator().manual_seed(seed)
    counts = torch.zeros(wte.shape[0], dtype=torch.long)
    with torch.no_grad():
        for start in range(0, n, chunk):
            m = min(chunk, n - start)
            v = torch.randn(m, d, generator=gen)
            h = (v - v.mean(-1, keepdim=True)) / (v.var(-1, unbiased=False, keepdim=True) + 1e-5).sqrt()
            h = h * g + b
            counts += torch.bincount((h @ wte.T).argmax(-1), minlength=wte.shape[0])
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--n", type=int, default=10000)
    args = ap.parse_args()

    wte, g, b = load_pieces(args.model_path)
    V = wte.shape[0]
    strings = id_to_string(args.model_path, V)
    index = {}
    for i, s in enumerate(strings):          # first id wins on the rare duplicate
        index.setdefault(s, i)

    norms = wte.norm(dim=1)
    mean_dir = wte.mean(0)
    mean_dir = mean_dir / mean_dir.norm()
    cos_mean = (wte / norms.clamp_min(1e-9).unsqueeze(1)) @ mean_dir

    def pct(vals, x):                        # percentile of x within vals
        return round(100.0 * float((vals < x).sum()) / len(vals), 2)

    counts = census(wte, g, b, args.n)
    share = counts.float() / args.n

    def rows(names):
        out = []
        for s in names:
            i = index.get(s)
            if i is None:
                out.append({"token": s, "error": "not a single-token string"})
                continue
            out.append({
                "token": s, "id": i,
                "S1_share_pct": round(100.0 * float(share[i]), 3),
                "S2_norm_pct": pct(norms, norms[i]),
                "S3_cosmean_pct": pct(cos_mean, cos_mean[i]),
                "norm": round(float(norms[i]), 4),
            })
        return out

    report = {
        "n_directions": args.n, "seed": SEED, "vocab": V,
        "funnel": rows(FUNNEL),
        "funnel_extra": rows(FUNNEL_EXTRA),
        "word_contrast": rows(WORD),
        "vocab_norm_mean": round(float(norms.mean()), 4),
        "vocab_norm_p05_p95": [round(float(norms.quantile(0.05)), 4),
                               round(float(norms.quantile(0.95)), 4)],
    }
    top = counts.argsort(descending=True)[:20]
    report["census_top20"] = [
        {"token": strings[int(i)], "id": int(i),
         "share_pct": round(100.0 * float(share[i]), 3)}
        for i in top if counts[int(i)] > 0
    ]
    for key, names in (("funnel", FUNNEL), ("word_contrast", WORD)):
        ids = [index[s] for s in names if s in index]
        report[f"{key}_collective_share_pct"] = round(
            100.0 * float(share[ids].sum()), 3)

    out = HERE / "output" / "funnel_geometry.json"
    out.write_text(json.dumps(report, indent=2))

    print(f"N={args.n} directions, seed={SEED}, vocab={V}")
    print(f"\n{'set':<10}{'token':<18}{'S1 share%':>10}{'S2 norm pct':>13}{'S3 cos pct':>12}")
    for key in ("funnel", "funnel_extra", "word_contrast"):
        for r in report[key]:
            if "error" in r:
                print(f"{key:<10}{r['token']!r:<18}  {r['error']}")
            else:
                print(f"{key:<10}{r['token']!r:<18}{r['S1_share_pct']:>10}"
                      f"{r['S2_norm_pct']:>13}{r['S3_cosmean_pct']:>12}")
    print(f"\ncollective S1 share: funnel {report['funnel_collective_share_pct']}%"
          f" | word {report['word_contrast_collective_share_pct']}%")
    print("\ncensus top-20 (what the decoder says for a generic direction):")
    for r in report["census_top20"]:
        print(f"  {r['token']!r:<20} {r['share_pct']:>7}%")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
