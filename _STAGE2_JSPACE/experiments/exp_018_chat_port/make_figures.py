"""Figures for EXP_018. Reads only the committed artifacts; runs no model."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"


def fig_collapse() -> None:
    """How far the word positions merge, repetition by repetition, in both arms.

    A value of 1.00 means every word position of the state holds the same
    direction, which is what GPT-2 Small reaches by about repetition 10.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, arm, title in zip(axes, ("bare", "chat"),
                              ("Main arm: bare text, 25 prompts",
                               "Pilot arm: chat template, 5 prompts")):
        path = OUT / f"results_{arm}.json"
        if not path.exists():
            continue
        recs = json.loads(path.read_text())["records"]
        for r in recs:
            it = [t["iteration"] for t in r["trace"]]
            pc = [t["pos_collapse_all"] for t in r["trace"]]
            ax.plot(it, pc, lw=0.9, alpha=0.75)
            if r["lock_in_iter"] is not None:
                ax.plot([r["lock_in_iter"]], [r["pos_collapse_all_terminal"]],
                        "o", ms=3.5, color="k", zorder=5)
        ax.axhline(1.0, color="crimson", ls="--", lw=1.2)
        ax.axhline(0.99, color="darkorange", ls=":", lw=1.2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("repetition of the loop")
        ax.set_xscale("symlog", linthresh=10)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("mean cosine between word positions")
    axes[0].set_ylim(-0.05, 1.12)
    for ax in axes:
        ax.annotate("0.99, the threshold H19 is scored on, and 1.00, where\n"
                    "GPT-2 Small sits by about repetition 10, both lie here",
                    xy=(0.08, 0.995), xycoords=("axes fraction", "data"),
                    xytext=(0.10, 0.26), textcoords="axes fraction",
                    fontsize=8, color="crimson", va="top",
                    arrowprops=dict(arrowstyle="->", color="crimson", lw=1))
    fig.suptitle("EXP_018: do the word positions collapse onto one shared vector "
                 "in Qwen3-1.7B?  One line per prompt; a black dot would mark "
                 "a prompt that settled.", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(OUT / "collapse_over_iterations.png", dpi=140)
    plt.close(fig)


def fig_loudness() -> None:
    """The natural loudness of each layer's entry, and where position 0 sits.

    The curve runs over the 28 block entries, `blocks.<l>.hook_resid_pre`. The
    loop's own two points are the first of those, the entry to block 0, and the
    exit of block 27, `blocks.27.hook_resid_post`, which is not a block entry
    and so is drawn as a separate final point rather than left off the end. The
    last entry point, the entry to block 27, still carries a large first-word
    activation and is not where the loop reads.
    """
    d = json.loads((OUT / "probe_natural_norms_bfloat16.json").read_text())
    bare = [v for v in d["prompts"].values() if v["arm"] == "bare"]
    n_layers = d["cfg"]["n_layers"]
    layers = list(range(n_layers))
    keys = [f"blocks.{l}.hook_resid_pre" for l in layers]
    out_key = f"blocks.{n_layers - 1}.hook_resid_post"
    has_out = all(out_key in p["natural"] for p in bare)
    if has_out:
        keys.append(out_key)
    pos0 = np.array([[p["natural"][k]["pos0"] for k in keys] for p in bare])
    ex0 = np.array([[p["natural"][k]["excl0"] for k in keys] for p in bare])
    T = np.array([p["n_tokens"] for p in bare])[:, None]
    per_pos = ex0 / np.sqrt(T - 1)          # typical size of one other position
    x_out = n_layers + 1                    # set apart from the last block entry
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].semilogy(layers, pos0.mean(0)[:n_layers], "o-", ms=3,
                     color="tab:blue", label="word position 0")
    axes[0].semilogy(layers, per_pos.mean(0)[:n_layers], "s-", ms=3,
                     color="tab:orange", label="a typical other word position")
    ratio = pos0 / per_pos
    axes[1].semilogy(layers, ratio.mean(0)[:n_layers], "o-", ms=3, color="crimson")
    axes[1].axhline(1.0, color="k", lw=1, ls=":")
    if has_out:
        axes[0].semilogy([x_out], [pos0.mean(0)[-1]], "*", ms=11,
                         color="tab:blue", mec="k", mew=0.6, zorder=5)
        axes[0].semilogy([x_out], [per_pos.mean(0)[-1]], "*", ms=11,
                         color="tab:orange", mec="k", mew=0.6, zorder=5)
        axes[0].plot([], [], "*", color="0.45", ms=11, mec="k", mew=0.6,
                     label="the same two at the exit of block 27, the loop's "
                           "extraction point")
        axes[1].semilogy([x_out], [ratio.mean(0)[-1]], "*", ms=13,
                         color="crimson", mec="k", mew=0.6, zorder=5)
    for ax in axes:
        # The loop's own two points: the entry to block 0 and the exit of
        # block 27. Everything between them is a block entry the loop never
        # touches.
        ax.axvline(0, color="0.5", lw=1, ls="--")
        if has_out:
            ax.axvline(x_out, color="0.5", lw=1, ls="--")
            ax.set_xticks(list(range(0, n_layers, 5)) + [x_out])
            ax.set_xticklabels([str(v) for v in range(0, n_layers, 5)] + ["out"])
            ax.set_xlim(-1.4, x_out + 1.4)
        ax.set_xlabel("entry to this block, and 'out' for the exit of block 27")
    axes[0].set_ylabel("size of the state (log scale)")
    axes[0].set_title("Natural loudness, mean over the 25 bare prompts", fontsize=10)
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.25, which="both")
    axes[1].set_ylabel("position 0 divided by a typical other position")
    axes[1].set_title("How much larger position 0 is", fontsize=10)
    axes[1].grid(alpha=0.25, which="both")
    if has_out:
        axes[1].annotate("the loop's two points, marked by the dashed lines:\n"
                         f"{ratio.mean(0)[0]:.2f} at the entry to block 0 and "
                         f"{ratio.mean(0)[-1]:.2f} at the exit of block 27,\n"
                         f"against {ratio.mean(0)[n_layers - 1]:.1f} at the "
                         "entry to block 27",
                         xy=(0.97, 0.97), xycoords="axes fraction", fontsize=7.5,
                         color="0.25", va="top", ha="right")
    fig.suptitle("EXP_018: the first word piece carries a huge activation at "
                 "every block entry from layer 3 on, but not at either of the "
                 "loop's own two points", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(OUT / "natural_loudness_profile.png", dpi=140)
    plt.close(fig)


def fig_jspace() -> None:
    """The J-space share of settled states against ordinary states, per layer."""
    path = OUT / "jspace_shares_bare.json"
    if not path.exists():
        return
    d = json.loads(path.read_text())
    layers = d["scored_layers"]
    per = d["verdict"]["per_layer"]
    s = [per[str(l)]["median_settled"] for l in layers]
    c = [per[str(l)]["median_clean"] for l in layers]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.plot(layers, s, "o-", ms=4, label="settled states (the loop's terminal)")
    ax.plot(layers, c, "s-", ms=4, label="ordinary states (a plain read of the prompt)")
    for seed in d["rotation_seeds"]:
        ax.plot(layers, [per[str(l)][f"median_settled_rot{seed}"] for l in layers],
                ":", lw=1, alpha=0.8,
                label=f"settled, lens directions randomly rotated (seed {seed})")
        ax.plot(layers, [per[str(l)][f"median_clean_rot{seed}"] for l in layers],
                "--", lw=1, alpha=0.8,
                label=f"ordinary, lens directions randomly rotated (seed {seed})")
    b0, b1 = min(d["band_layers"]), max(d["band_layers"])
    ax.axvspan(b0 - 0.5, b1 + 0.5, color="0.85", zorder=0)
    ax.text((b0 + b1) / 2, ax.get_ylim()[1], " the paper's workspace band ",
            ha="center", va="top", fontsize=8, color="0.35")
    ax.set_xlabel("layer of the model (0 to 27)")
    ax.set_ylabel("median J-space share (0 to 1)")
    ax.set_title("EXP_018 / H19b: how much of each state the lens can express\n"
                 "as at most 25 word directions with non-negative weights",
                 fontsize=10.5)
    ax.legend(fontsize=7.5); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "jspace_share_by_layer.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    fig_loudness()
    fig_collapse()
    fig_jspace()
    print("figures written to", OUT)
