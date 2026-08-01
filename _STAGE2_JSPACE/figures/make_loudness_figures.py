#!/usr/bin/env python3
"""Layer-loudness profile figures (issue #62).

Reads the committed natural per-layer residual-stream norm files produced by
run_exp010c.py's --record-natural-norms path (one ordinary, un-hooked forward
pass per prompt; the recorded value at layer L is the L2 norm of the full
blocks.L.hook_resid_pre tensor, all sequence positions):

  - natural_resid_norms_energynorm_A0.json   (gpt2-medium, registered 25-prompt
    subset; identical copies exist under the A1/A4/O8 suffixes because each
    Control B arm re-recorded the same injection-free pass)
  - natural_resid_norms_small010b_S1.json    (gpt2-small, its 25-prompt subset;
    identical copies under S2..S5 and SB)
  - natural_resid_norms_pythia410m.json      (pythia-410m, its 25-prompt subset)

Produces, in this directory:
  - loudness_profile_gpt2_medium.png
  - loudness_profile_gpt2_small.png
  - loudness_profile_pythia410m.png
  - loudness_profile_stats.json   (derived summary statistics + provenance)

This is a documentation artifact, not a registered experiment: no model is run,
no new measurement is made; everything here is arithmetic on committed files.
The constants quoted in annotations from other committed records (the seed-to-
natural norm ratios 217.8 / 306.7 / 1.04 / 1.05 for Medium, 73.0 / 153.7 /
160.3 for Small, 63.1 / 402.6 for Pythia) come from RESULTS_EXP010C.md
(Control B), RESULTS_EXP010B.md (Energy record) and RESULTS_EXP012_PYTHIA.md
(norm-ratio table) respectively.

Run:  python3 make_loudness_figures.py   (needs matplotlib only)
"""

import json
import statistics
import sys
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter  # noqa: E402

HERE = Path(__file__).resolve().parent
OUTDIR = HERE
DATA = HERE.parent / "experiments" / "exp_010c_windows" / "output"

# --- palette (light mode; colour-blind-safe: blue data, orange / violet refs,
#     neutral greys; no red-green pair anywhere) -------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
PROMPT_LINE = "#86b6ef"   # sequential blue, light step: one thin line per prompt
MEAN_LINE = "#104281"     # sequential blue, dark step: the bold mean
REF_ORANGE = "#eb6834"    # seed-at-readout convention marker
REF_VIOLET = "#4a3aa7"    # natural-at-entry convention marker
BAND = "#e1e0d9"          # plateau band fill

DPI = 220

MODELS = {
    "gpt2-medium": dict(
        file="natural_resid_norms_energynorm_A0.json",
        n_layers=24,
        subtitle=("GPT-2 Medium, 24 layers. One ordinary forward pass for "
                  "each of the 25 registered prompts."),
    ),
    "gpt2-small": dict(
        file="natural_resid_norms_small010b_S1.json",
        n_layers=12,
        subtitle=("GPT-2 Small, 12 layers. One ordinary forward pass for "
                  "each of the 25 prompts of its subset."),
    ),
    "pythia-410m": dict(
        file="natural_resid_norms_pythia410m.json",
        n_layers=24,
        subtitle=("Pythia-410m, 24 layers. One ordinary forward pass for "
                  "each of the 25 prompts of its subset."),
    ),
}


def load(model_key):
    cfg = MODELS[model_key]
    raw = json.loads((DATA / cfg["file"]).read_text())
    layers = list(range(cfg["n_layers"]))
    prompts = sorted(raw)
    curves = {p: [raw[p][str(l)] for l in layers] for p in prompts}
    return layers, prompts, curves


def summarise(layers, prompts, curves):
    mean = [statistics.mean(curves[p][l] for p in prompts) for l in layers]
    lo = [min(curves[p][l] for p in prompts) for l in layers]
    hi = [max(curves[p][l] for p in prompts) for l in layers]
    last = layers[-1]
    full = [curves[p][last] / curves[p][0] for p in prompts]
    step_mean = [mean[l] / mean[l - 1] for l in layers[1:]]
    return dict(
        mean=[round(v, 1) for v in mean],
        min=[round(v, 1) for v in lo],
        max=[round(v, 1) for v in hi],
        full_stack_entry_ratio=dict(
            mean=round(statistics.mean(full), 1),
            min=round(min(full), 1),
            max=round(max(full), 1),
            definition=(
                f"per prompt: norm entering layer {last} divided by norm "
                "entering layer 0"
            ),
        ),
        step_growth_factor_of_means=[round(v, 3) for v in step_mean],
    )


def style_axes(ax, log=False):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.grid(True, axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    if log:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())


def draw_profile(ax, layers, prompts, curves, mean, log=False):
    for p in prompts:
        ax.plot(layers, curves[p], color=PROMPT_LINE, linewidth=0.8,
                alpha=0.55, zorder=2)
    ax.plot(layers, mean, color=MEAN_LINE, linewidth=2.2, zorder=3)
    ax.set_xlim(layers[0] - 0.35, layers[-1] + 0.35)
    ax.xaxis.set_major_locator(FixedLocator(layers[::2]))
    ax.xaxis.set_minor_locator(FixedLocator(layers))
    style_axes(ax, log=log)


def draw_delta(ax, layers, prompts, curves):
    xs = layers[1:]
    for p in prompts:
        ratios = [curves[p][l] / curves[p][l - 1] for l in xs]
        ax.plot(xs, ratios, color=PROMPT_LINE, linewidth=0.8, alpha=0.55,
                zorder=2)
    mean_ratio = [statistics.mean(curves[p][l] / curves[p][l - 1]
                                  for p in prompts) for l in xs]
    ax.plot(xs, mean_ratio, color=MEAN_LINE, linewidth=2.2, zorder=3)
    ax.axhline(1.0, color=MUTED, linewidth=0.9, linestyle=(0, (4, 3)),
               zorder=1)
    ax.set_xlim(layers[0] - 0.35, layers[-1] + 0.35)
    ax.xaxis.set_major_locator(FixedLocator(layers[::2]))
    ax.xaxis.set_minor_locator(FixedLocator(layers))
    style_axes(ax, log=True)
    return mean_ratio


def annotate(ax, x, y, text, dx=0, dy=0, color=INK2, ha="left", fontsize=7.8):
    ax.annotate(text, xy=(x, y), xytext=(x + dx, y + dy), ha=ha,
                va="center", fontsize=fontsize, color=color,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.7,
                                shrinkA=0, shrinkB=2))


def make_figure(model_key, out_png, annotator):
    cfg = MODELS[model_key]
    layers, prompts, curves = load(model_key)
    stats = summarise(layers, prompts, curves)
    mean = [statistics.mean(curves[p][l] for p in prompts) for l in layers]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8.8, 11.8))
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(top=0.925, bottom=0.055, left=0.095, right=0.965,
                        hspace=0.42)

    fig.suptitle(
        "Layer loudness: magnitude of the residual stream entering each "
        f"layer\n{cfg['subtitle']}",
        fontsize=11, color=INK, y=0.985)

    ylab = "L2 norm of the residual tensor\n(model units, all token positions)"

    draw_profile(ax1, layers, prompts, curves, mean, log=False)
    ax1.set_title("Linear scale: the profile as raw magnitudes",
                  fontsize=9.5, color=INK, loc="left")
    ax1.set_ylabel(ylab, fontsize=8.5, color=INK2)

    draw_profile(ax2, layers, prompts, curves, mean, log=True)
    ax2.set_title(
        "Logarithmic scale: equal vertical steps are equal multiplications",
        fontsize=9.5, color=INK, loc="left")
    ax2.set_ylabel(ylab, fontsize=8.5, color=INK2)

    mean_ratio = draw_delta(ax3, layers, prompts, curves)
    ax3.set_title(
        "Layer-to-layer growth: how many times louder the stream gets at each "
        "step", fontsize=9.5, color=INK, loc="left")
    ax3.set_ylabel("growth factor entering layer L:\nnorm at L divided by "
                   "norm at L\u22121", fontsize=8.5, color=INK2)
    ax3.set_xlabel("layer L (hook_resid_pre: the residual stream as it enters "
                   "layer L)", fontsize=8.5, color=INK2)
    for a in (ax1, ax2):
        a.set_xlabel("layer L (hook_resid_pre: the residual stream as it "
                     "enters layer L)", fontsize=8.5, color=INK2)

    # shared legend text (one series family: prompts + mean)
    ax1.text(0.99, 0.06,
             "Thin lines: one per prompt (25). Bold line: mean.",
             transform=ax1.transAxes, ha="right", va="bottom", fontsize=8,
             color=INK2)

    annotator(ax1, ax2, ax3, layers, mean, mean_ratio, stats)

    fig.savefig(out_png, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    return stats


# --- per-model annotations ----------------------------------------------------

def annotate_medium(ax1, ax2, ax3, layers, mean, mean_ratio, stats):
    m0 = mean[0]
    seed_full = 217.8 * m0  # committed ratio x measured layer-0 mean (inferred level)
    band_lo = min(mean[4:23])
    band_hi = max(mean[4:23])

    ax1.set_ylim(0, 4780)
    ax2.set_ylim(8, 11000)
    ax3.set_ylim(0.45, 60)

    # plateau band on the log panel
    ax2.axvspan(4, 22, color=BAND, alpha=0.45, zorder=0)
    ax2.text(13, 6300,
             "Mid-band plateau, layers 4 to 22: the mean drifts only from "
             f"about {band_lo:,.0f} to {band_hi:,.0f} units.",
             ha="center", fontsize=7.8, color=INK2)

    # the two normalisation conventions (log panel)
    ax2.axhline(seed_full, color=REF_ORANGE, linewidth=1.4,
                linestyle=(0, (5, 3)), zorder=1)
    ax2.text(23.1, 2450,
             "Seed-at-readout convention, full-stack loop: the loop holds "
             "the signal near 2,850 units,\nthe committed 217.8 times the "
             "layer-0 entry (level inferred as 217.8 x mean entry).",
             fontsize=7.8, color=REF_ORANGE, ha="right", va="top")
    ax2.axhline(m0, color=REF_VIOLET, linewidth=1.4,
                linestyle=(0, (1, 2.2)), zorder=1)
    ax2.text(23.1, m0 * 1.22,
             "Natural-at-entry convention, layer-0 injection: the loop holds "
             f"the signal at the layer-0 entry level, about {m0:,.0f} units.",
             fontsize=7.8, color=REF_VIOLET, ha="right", va="bottom")

    # entry vs deep readout gap (log panel): two-point bracket
    ax2.annotate("", xy=(0, seed_full), xytext=(0, m0),
                 arrowprops=dict(arrowstyle="<->", color=INK, linewidth=1.1))
    ax2.text(1.3, 190,
             "The gap the apparatus-mask question turns on:\nthe full-stack "
             "loop runs about 220 times louder than\nthe natural layer-0 "
             "entry (committed measurement: 217.8x).",
             fontsize=7.8, color=INK, va="center")

    # flagship windows (log panel)
    for x in (8, 10):
        ax2.axvline(x, color=MUTED, linewidth=0.9, linestyle=(0, (2, 2)),
                    zorder=1, ymax=0.60)
    ax2.text(10.0, 58,
             "Flagship word windows inject here (layers 8 and 10)\nand read "
             "out at layer 21: on this plateau the loop runs\nat 1.05 and "
             "1.04 times natural entry (committed ratios).",
             ha="center", fontsize=7.8, color=INK2, va="center")

    # linear panel: entry + deep-end drop
    annotate(ax1, 0, mean[0],
             f"Layer 0 entry is tiny: mean {mean[0]:,.0f} units.",
             dx=1.1, dy=950)
    annotate(ax1, 22, mean[22],
             f"Loudest entry: {mean[22]:,.0f} units into layer 22.",
             dx=-12.6, dy=430)
    annotate(ax1, 23, mean[23],
             f"Entering the last layer the stream drops back to "
             f"{mean[23]:,.0f} units.",
             dx=-14.6, dy=-820)

    # delta panel callouts
    annotate(ax3, 1, mean_ratio[0],
             "Biggest step: layer 0 multiplies the magnitude by about 33.",
             dx=0.6, dy=0)
    annotate(ax3, 4, mean_ratio[3],
             "Layer 3 multiplies it by about 5.4.", dx=0.7, dy=0)
    ax3.annotate("The last step shrinks it: about 0.74 times.",
                 xy=(23, mean_ratio[22]), xytext=(16.2, 0.60),
                 fontsize=7.8, color=INK2, ha="center", va="center",
                 arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.7,
                                 shrinkA=0, shrinkB=2))
    ax3.text(13, 2.1,
             "Across the plateau each layer multiplies the magnitude by "
             "only 1.00 to 1.05: the loudness barely moves.",
             ha="center", fontsize=7.8, color=INK2)


def annotate_small(ax1, ax2, ax3, layers, mean, mean_ratio, stats):
    fs = stats["full_stack_entry_ratio"]
    ax2.axvspan(3, 11, color=BAND, alpha=0.45, zorder=0)
    ax2.text(7, 1150,
             "Plateau, layers 3 to 11: the mean drifts only from about "
             f"{min(mean[3:]):,.0f} to {max(mean[3:]):,.0f} units.",
             ha="center", fontsize=7.8, color=INK2)
    ax2.axhline(mean[0], color=REF_VIOLET, linewidth=1.4,
                linestyle=(0, (1, 2.2)), zorder=1)
    ax2.text(11.0, mean[0] * 1.4,
             "Natural layer-0 entry, about 20 units: where a natural-at-entry "
             "convention would peg a layer-0 loop.",
             fontsize=7.8, color=REF_VIOLET, ha="right", va="bottom")
    ax2.text(11.2, 140,
             "Entry to the last layer is on average "
             f"{fs['mean']:.0f} times the layer-0 entry\n(per-prompt range "
             f"{fs['min']:.0f} to {fs['max']:.0f}). The committed energy "
             "table for the\nsmall-model grid puts the loops that inject at "
             "layer 0 at 56 to 175\ntimes natural strength, depending on "
             "which layer each loop\nreads its seed from.",
             fontsize=7.8, color=INK, ha="right", va="center")

    annotate(ax1, 0, mean[0],
             f"Layer 0 entry is tiny: mean {mean[0]:,.0f} units.",
             dx=0.4, dy=380)
    annotate(ax1, 11, mean[11],
             f"Entry to the last layer: {mean[11]:,.0f} units.",
             dx=-4.6, dy=-700)

    annotate(ax3, 1, mean_ratio[0],
             "Layer 0 multiplies the magnitude by about 13.", dx=0.35, dy=0)
    annotate(ax3, 3, mean_ratio[2],
             "Layer 2 multiplies it by about 3.9.", dx=0.4, dy=0)
    ax3.text(7.0, 2.05,
             "From layer 3 on, each layer multiplies the magnitude\nby 1.01 "
             "to 1.08: a gentle rise, with no late drop within\nthe "
             "recorded range (it stops at the entry to layer 11).",
             ha="center", fontsize=7.8, color=INK2)


def annotate_pythia(ax1, ax2, ax3, layers, mean, mean_ratio, stats):
    fs = stats["full_stack_entry_ratio"]
    ax2.axvspan(9, 21, color=BAND, alpha=0.45, zorder=0)
    ax2.text(15, 250,
             "Plateau, layers 9 to 21: the mean stays between about "
             f"{min(mean[9:22]):,.0f} and {max(mean[9:22]):,.0f} units.",
             ha="center", fontsize=7.8, color=INK2)
    ax2.axhline(mean[0], color=REF_VIOLET, linewidth=1.4,
                linestyle=(0, (1, 2.2)), zorder=1)
    ax2.text(23.0, mean[0] * 1.4,
             "Natural layer-0 entry, about 2.5 units.",
             fontsize=7.8, color=REF_VIOLET, ha="right", va="bottom")
    ax2.text(23.1, 40,
             "Entry to the last layer is on average "
             f"{fs['mean']:.0f} times the layer-0 entry\n(per-prompt range "
             f"{fs['min']:.0f} to {fs['max']:.0f}). The committed Pythia "
             "norm-ratio table\nputs the full-stack loop's seed at 63.1 "
             "times the layer-0 entry,\nlower than the ratio of entries "
             "because the seed is read after\nthe last layer, where the "
             "magnitude has already fallen steeply.",
             fontsize=7.8, color=INK, ha="right", va="center")

    annotate(ax1, 0, mean[0],
             f"Layer 0 entry is tiny: mean {mean[0]:.1f} units.",
             dx=0.5, dy=110)
    annotate(ax1, 23, mean[23],
             "Late layers shrink the stream: entry to the last layer is "
             f"{mean[23]:,.0f} units,\nless than half the plateau.",
             dx=-13.5, dy=-140)

    ax3.set_ylim(0.38, 30)
    annotate(ax3, 1, mean_ratio[0],
             "Layer 0 multiplies the magnitude by about 19.", dx=0.6, dy=0)
    annotate(ax3, 6, mean_ratio[5],
             "Layer 5 multiplies it by about 7.", dx=0.7, dy=0)
    ax3.annotate("The last two steps shrink it: 0.91 then 0.52 times.",
                 xy=(23, mean_ratio[22]), xytext=(17.5, 0.52),
                 fontsize=7.8, color=INK2, ha="center", va="center",
                 arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.7,
                                 shrinkA=0, shrinkB=2))


def main():
    all_stats = {}
    jobs = [
        ("gpt2-medium", "loudness_profile_gpt2_medium.png", annotate_medium),
        ("gpt2-small", "loudness_profile_gpt2_small.png", annotate_small),
        ("pythia-410m", "loudness_profile_pythia410m.png", annotate_pythia),
    ]
    for model_key, png, annotator in jobs:
        stats = make_figure(model_key, OUTDIR / png, annotator)
        all_stats[model_key] = dict(
            source_file=str(Path("experiments/exp_010c_windows/output")
                            / MODELS[model_key]["file"]),
            n_prompts=25,
            n_layers=MODELS[model_key]["n_layers"],
            **stats,
        )
        print(f"[{model_key}] wrote {png}")
        print(f"  layer-0 entry mean {stats['mean'][0]}, "
              f"last-layer entry mean {stats['mean'][-1]}, "
              f"full-stack entry ratio mean "
              f"{stats['full_stack_entry_ratio']['mean']} "
              f"(range {stats['full_stack_entry_ratio']['min']} to "
              f"{stats['full_stack_entry_ratio']['max']})")

    out = dict(
        provenance=dict(
            produced_by="figures/make_loudness_figures.py",
            produced_on=str(date.today()),
            task="issue #62 (layer-loudness profile documentation)",
            method=(
                "Derived statistics only: arithmetic over the committed "
                "natural_resid_norms_*.json files listed per model under "
                "source_file. No model was run. Each source value is the L2 "
                "norm of the full hook_resid_pre residual tensor at that "
                "layer on one ordinary (injection-free) forward pass, as "
                "recorded by run_exp010c.py --record-natural-norms."
            ),
        ),
        models=all_stats,
    )
    (OUTDIR / "loudness_profile_stats.json").write_text(
        json.dumps(out, indent=2) + "\n")
    print("wrote loudness_profile_stats.json")


if __name__ == "__main__":
    sys.exit(main())
