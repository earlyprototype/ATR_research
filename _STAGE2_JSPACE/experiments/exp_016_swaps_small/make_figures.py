"""Two small figures for the EXP_016 results record: success rate against
layer for each battery, and the lens arm against both controls."""
import json, sys
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

D = os.path.dirname(os.path.abspath(__file__)) + "/"
TITLES = {"h17": "H17 report swap: target word enters the top five",
          "h17a": "H17a country swap: target answer enters the top five",
          "h17b": "H17b two-step swap: answer becomes the alternative"}
COL = {"lens": "#1b6ca8", "randdir": "#c8553d", "randnorm": "#7a7a7a"}
LAB = {"lens": "lens swap", "randdir": "control A, random directions",
       "randnorm": "control B, size-matched random"}

fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True)
for ax, b in zip(axes, ["h17", "h17a", "h17b"]):
    s = json.load(open(D + f"output/summary_{b}.json"))
    cell = s["chosen_cell"]
    singles = [g for g in s["grid"] if "-" not in g["layers"]
               and g["alpha"] == cell[1] and g["posmode"] == cell[2]]
    singles.sort(key=lambda g: int(g["layers"]))
    x = [int(g["layers"]) for g in singles]
    for arm in ("lens", "randdir", "randnorm"):
        y = [g["overall"].get(arm, [0])[0] for g in singles]
        ax.plot(x, y, "o-", color=COL[arm], label=LAB[arm], lw=2, ms=5)
    ax.set_title(TITLES[b], fontsize=9)
    ax.set_xlabel("single layer swapped\n(residual stream after block l)",
                  fontsize=8)
    ax.grid(alpha=.3)
    ax.set_ylim(-0.02, 1.02)
    ax.text(0.02, 0.96, f"strength {cell[1]}, positions {cell[2]}",
            transform=ax.transAxes, fontsize=7, va="top")
axes[0].set_ylabel("success rate (0 to 1)", fontsize=9)
axes[0].legend(fontsize=7, loc="upper left", bbox_to_anchor=(0, 0.9))
fig.suptitle("EXP_016: swapping two lens coordinates in base GPT-2 Small, "
             "one layer at a time, against two controls", fontsize=10)
fig.tight_layout()
fig.savefig(D + "output/fig_layers.png", dpi=130)
print("wrote fig_layers.png")

fig2, axes2 = plt.subplots(1, 3, figsize=(13, 3.6))
halves = ["tuning", "heldout"]
for ax, b in zip(axes2, ["h17", "h17a", "h17b"]):
    s = json.load(open(D + f"output/summary_{b}.json"))
    c = s["chosen_cell"]
    # For H17 the protocol number is the source rule's, not the pooled one:
    # section 5.1 of the specification makes the choice between the two
    # readings of "the model's own top concept" part of the tuned selection,
    # so the bar that carries the verdict is the selected rule's held-out
    # rate. The pooled rate, which an earlier version of this figure plotted
    # as "the reported number", is kept beside it and labelled as pooled.
    if b == "h17":
        sr = s["source_rule_selection"]
        c = sr["chosen_cell"]
        series = [("lens swap, selected source rule "
                   f"({sr['chosen_rule']})", COL["lens"], "",
                   [sr[h].get("lens", [0])[0] for h in halves]),
                  ("lens swap, both source rules pooled", COL["lens"], "//",
                   [s[h].get("lens", [0])[0] for h in halves])]
        series += [(LAB[a], COL[a], "", [sr[h].get(a, [0])[0] for h in halves])
                   for a in ("randdir", "randnorm")]
        title = (f"H17: layers {c[0]}, strength {c[1]}, {c[2]}, "
                 f"source rule {sr['chosen_rule']}")
    else:
        series = [(LAB[a], COL[a], "", [s[h].get(a, [0])[0] for h in halves])
                  for a in ("lens", "randdir", "randnorm")]
        title = f"{b.upper()}: layers {c[0]}, strength {c[1]}, {c[2]}"
    w = 0.8 / len(series)
    off = (len(series) - 1) / 2
    for i, (label, colour, hatch, vals) in enumerate(series):
        ax.bar([j + (i - off) * w for j in range(2)], vals, w, color=colour,
               hatch=hatch, edgecolor="white", label=label)
        for j, v in enumerate(vals):
            ax.text(j + (i - off) * w, v + .015, f"{v:.2f}", ha="center",
                    fontsize=7)
    ax.set_xticks(range(2))
    ax.set_xticklabels(["tuning half\n(setting chosen here)",
                        "held-out half\n(the reported number)"], fontsize=8)
    ax.set_title(title, fontsize=9)
    # Headroom above 1.0 so the per-panel legend never sits over a bar.
    ax.set_ylim(0, 1.32); ax.grid(axis="y", alpha=.3)
    ax.legend(fontsize=6.5, loc="upper right", framealpha=0.95)
axes2[0].set_ylabel("success rate (0 to 1)", fontsize=9)
fig2.suptitle("EXP_016: the tuned setting, chosen on one half and scored on "
              "the other", fontsize=10)
fig2.tight_layout()
fig2.savefig(D + "output/fig_heldout.png", dpi=130)
print("wrote fig_heldout.png")
