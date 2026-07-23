"""EXP_010c terminal characterisation — pure analysis, no model time.

Answers, per arm: are the basins FEW AND SHARED (Small-like attractor
structure) or MANY AND PRIVATE (per-prompt fixed points)? Uses the saved
terminal mean vectors: cross-prompt cosine matrix, basin clustering at the
gate threshold, margin/entropy summaries.

Usage: python analyze_terminals.py [--tier full] [--decode-via-tail --model-path DIR]

--decode-via-tail additionally runs each terminal once through layers j+1..23
(no looping) and decodes at 23 — the EXP_010c-2 §3 readout control.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

ARMS_WINDOWS = {"A0": (0, 23), "A4": (10, 21), "A1": (0, 11), "A2": (6, 17),
                "A3": (12, 23), "A5": (8, 15),
                "E22": (10, 22), "E23": (10, 23), "O0": (0, 21), "O4": (4, 21),
                "O6": (6, 21), "O8": (8, 21), "O12": (12, 21), "O14": (14, 21)}
CLUSTER_THRESHOLD = 0.999  # same as the convergence gate


def cluster(vecs, threshold=CLUSTER_THRESHOLD):
    """Greedy leader clustering on cosine similarity."""
    leaders, labels = [], []
    for v in vecs:
        for li, l in enumerate(leaders):
            if F.cosine_similarity(v.unsqueeze(0), l.unsqueeze(0)).item() > threshold:
                labels.append(li)
                break
        else:
            leaders.append(v)
            labels.append(len(leaders) - 1)
    return labels, len(leaders)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="full")
    ap.add_argument("--decode-via-tail", action="store_true")
    ap.add_argument("--model-path", default=None)
    args = ap.parse_args()

    results = json.load(open(HERE / "output" / f"results_{args.tier}.json"))
    terminals = torch.load(HERE / "output" / f"terminals_{args.tier}.pt",
                           map_location="cpu", weights_only=False)

    model = None
    if args.decode_via_tail:
        from run_exp010c import _load_medium_from_local
        model = _load_medium_from_local(args.model_path)
        model.eval()

    report = []
    arms = sorted({r["arm"] for r in results}, key=lambda a: list(ARMS_WINDOWS).index(a))
    for arm in arms:
        rs = [r for r in results if r["arm"] == arm]
        ids = [r["prompt_id"] for r in rs]
        vecs = [terminals[(arm, pid)]["mean"] for pid in ids]
        labels, n_clusters = cluster(vecs)
        sizes = Counter(labels)
        toks = Counter(r["terminal_token"] for r in rs)
        margins = torch.tensor([r["top_logit_margin"] for r in rs])
        ents = torch.tensor([r["entropy"] for r in rs])
        conv = sum(r["converged"] for r in rs)

        # cross-prompt cosine matrix summary
        M = torch.stack([v / v.norm() for v in vecs])
        C = M @ M.T
        off = C[~torch.eye(len(vecs), dtype=torch.bool)]

        line = {
            "arm": arm,
            "window": rs[0]["window"],
            "n": len(rs),
            "converged": conv,
            "tensor_basins": n_clusters,
            "basin_sizes": sorted(sizes.values(), reverse=True),
            "decode_terminals": dict(toks.most_common()),
            "offdiag_cos_mean": round(off.mean().item(), 4),
            "offdiag_cos_min": round(off.min().item(), 4),
            "margin_mean": round(margins.mean().item(), 3),
            "margin_max": round(margins.max().item(), 3),
            "entropy_mean": round(ents.mean().item(), 3),
        }

        if model is not None:
            i, j = ARMS_WINDOWS[arm]
            tail_toks = Counter()
            agree = 0
            for pid, r in zip(ids, rs):
                t = terminals[(arm, pid)]["mean"].clone()
                x = t.unsqueeze(0).unsqueeze(0)  # [1, 1, d_model]
                with torch.no_grad():
                    for layer in range(j + 1, model.cfg.n_layers):
                        x = model.blocks[layer](x)
                    v = x[0, -1, :]
                    logits = model.ln_final(v) @ model.W_U + model.b_U
                tok = model.tokenizer.decode([int(logits.argmax())])
                tail_toks[tok] += 1
                agree += int(tok == r["terminal_token"])
            line["tail_decode_terminals"] = dict(tail_toks.most_common())
            line["tail_agreement"] = f"{agree}/{len(rs)}"

        report.append(line)

    out = HERE / "output" / f"terminal_characterisation_{args.tier}.json"
    out.write_text(json.dumps(report, indent=2))
    hdr = f"{'arm':<4}{'window':<8}{'conv':<6}{'tensor_basins':<14}{'sizes':<18}{'cos_mean':<10}{'margin_mu/max':<15}"
    print(hdr)
    for l in report:
        print(f"{l['arm']:<4}{l['window']:<8}{l['converged']}/{l['n']:<4}"
              f"{l['tensor_basins']:<14}{str(l['basin_sizes'])[:16]:<18}"
              f"{l['offdiag_cos_mean']:<10}{l['margin_mean']}/{l['margin_max']:<10}")
        print(f"     decode: {l['decode_terminals']}")
        if "tail_decode_terminals" in l:
            print(f"     via-tail: {l['tail_decode_terminals']} (agree {l['tail_agreement']})")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
