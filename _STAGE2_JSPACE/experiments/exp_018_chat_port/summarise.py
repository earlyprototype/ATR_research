"""Turn the EXP_018 results files into the tables the results record prints.

Reads only committed artifacts and writes markdown to standard output, so the
numbers in `RESULTS_EXP018.md` can be regenerated and checked without rerunning
the model.
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

OUT = Path(__file__).resolve().parent / "output"


def esc(s: str) -> str:
    return (repr(s).replace("|", "\\|"))


def arm_table(arm: str) -> None:
    path = OUT / f"results_{arm}.json"
    if not path.exists():
        print(f"(no results for arm {arm})")
        return
    d = json.loads(path.read_text())
    recs = d["records"]
    print(f"\n### Arm `{arm}`: {len(recs)} prompts, cap "
          f"{d['loop_config']['max_iter']} repetitions\n")
    print("| Prompt | Word pieces | Settled? | Repetitions run | "
          "Positions merged (all) | Positions merged (position 0 left out) | "
          "Top word piece | Its probability | Spread (nats) | "
          "Loudness the old convention would have used |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in recs:
        lock = "no" if r["lock_in_iter"] is None else f"yes, at {r['lock_in_iter']}"
        print(f"| `{r['id']}` | {r['n_tokens']} | {lock} | {r['n_iters']} | "
              f"{r['pos_collapse_all_terminal']:.3f} | "
              f"{r['pos_collapse_excl0_terminal']:.3f} | "
              f"{esc(r['readout']['top_token_strings'][0])} | "
              f"{r['readout']['top_token_probs'][0]:.3f} | "
              f"{r['readout']['entropy']:.2f} | "
              f"{r['seed_over_natural_excl0']:.0f}x natural |")
    coll = [r["pos_collapse_all_terminal"] for r in recs]
    coll0 = [r["pos_collapse_excl0_terminal"] for r in recs]
    ent = [r["readout"]["entropy"] for r in recs]
    p1 = [r["readout"]["top_token_probs"][0] for r in recs]
    ratio = [r["seed_over_natural_excl0"] for r in recs]
    n_lock = sum(r["converged"] for r in recs)
    toks = [r["readout"]["top_token_strings"][0] for r in recs]
    uniq = {}
    for t in toks:
        uniq[t] = uniq.get(t, 0) + 1
    print(f"\nSettled: **{n_lock} of {len(recs)}**. "
          f"Positions-merged metric over all positions: median "
          f"**{st.median(coll):.3f}**, range {min(coll):.3f} to {max(coll):.3f}; "
          f"leaving position 0 out: median {st.median(coll0):.3f}, range "
          f"{min(coll0):.3f} to {max(coll0):.3f}. "
          f"Prompts at or above 0.99 on the all-positions metric: "
          f"{sum(c >= 0.99 for c in coll)} of {len(recs)}. "
          f"Top-word-piece probability: median {st.median(p1):.3f} "
          f"(a flat distribution over 151,936 word pieces would give 0.0000066). "
          f"Spread: median {st.median(ent):.2f} nats out of a possible 11.93. "
          f"Distinct top word pieces: {len(uniq)} "
          f"({', '.join(f'{esc(k)} x{v}' for k, v in sorted(uniq.items(), key=lambda kv: -kv[1]))}). "
          f"The registered loudness convention would have injected at "
          f"{st.mean(ratio):.0f} times natural strength on average "
          f"(range {min(ratio):.0f} to {max(ratio):.0f}).")


def jspace_table() -> None:
    path = OUT / "jspace_shares_bare.json"
    if not path.exists():
        print("\n(no J-space results yet)")
        return
    d = json.loads(path.read_text())
    v = d["verdict"]
    seeds = d["rotation_seeds"]
    print("\n### H19b: how much of each state the lens can express\n")
    head = ("| Layer | In the band? | Terminal median | Ordinary median | "
            "Difference | Permutation p | ")
    head += " | ".join(f"Terminal, rotated lens (seed {s})" for s in seeds) + " | "
    head += " | ".join(f"Ordinary, rotated lens (seed {s})" for s in seeds) + " |"
    print(head)
    print("|---" * (6 + 2 * len(seeds)) + "|")
    for layer in d["scored_layers"]:
        e = v["per_layer"][str(layer)]
        band = "yes" if layer in d["band_layers"] else "no (early contrast)"
        row = (f"| {layer} | {band} | {e['median_settled']:.4f} | "
               f"{e['median_clean']:.4f} | {e['median_difference']:+.4f} | "
               f"{e['p_one_sided']:.4f} | ")
        row += " | ".join(f"{e[f'median_settled_rot{s}']:.4f}" for s in seeds) + " | "
        row += " | ".join(f"{e[f'median_clean_rot{s}']:.4f}" for s in seeds) + " |"
        print(row)
    print(f"\nBand layers where the terminal median is below the ordinary median: "
          f"**{v['band_layers_below']} of {v['n_band_layers']}**. "
          f"Of those, with a permutation p-value below 0.05: "
          f"**{v['band_layers_below_p05']}**. "
          f"The pre-registered rule needs {v['majority_needed']} or more. "
          f"Verdict by that rule: "
          f"**{'SUPPORTED' if v['H19b_supported'] else 'NOT SUPPORTED'}**.")
    label = {"settled": "terminal", "clean": "ordinary"}
    for layer, row in d["layers"].items():
        if "exact_check" not in row:
            continue
        # Both conditions are printed, in a fixed order. Printing outside this
        # loop reported whichever condition the file happened to list last and
        # silently dropped the other, which is the larger of the two here.
        for cond in ("settled", "clean"):
            ex = row["exact_check"].get(cond)
            if ex is None:
                continue
            print(f"\nApproximation check at layer {layer}, {label[cond]} "
                  f"states: the largest difference between the share computed "
                  f"over the whole 151,936-word-piece vocabulary and the share "
                  f"computed over the 4,096 best-correlating directions is "
                  f"{ex['max_abs_diff']:.6f} on a scale of 0 to 1.")


if __name__ == "__main__":
    for arm in ("bare", "chat"):
        arm_table(arm)
    jspace_table()
