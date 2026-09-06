"""EXP_011: emit the markdown tables the results record uses, straight from the JSON.

By default this reads the final `output/verdicts.json` and
`output/per_layer_tables.json`. A diagnostic scoring, which is `score.py
--allow-partial` over a shares file that does not cover every arm and every
layer, writes its outputs under stamped names instead: `verdicts.partial.json`
and `per_layer_tables.partial.json`. Pass --partial to read those. Until
2026-09-06 this helper always opened the unstamped names, so after a diagnostic
scoring it either failed, when no final file existed, or printed the previous
final results while its own partial warning stayed silent, because that warning
is read from the file it opened.

Run: python3 make_tables.py [--partial]
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")

ap = argparse.ArgumentParser(description="Emit the EXP_011 markdown tables.")
ap.add_argument("--partial", action="store_true",
                help="read the stamped diagnostic files verdicts.partial.json and "
                     "per_layer_tables.partial.json that score.py --allow-partial "
                     "writes, instead of the final ones.")
ARGS = ap.parse_args()
STAMP = ".partial" if ARGS.partial else ""
v = json.load(open(os.path.join(OUT, f"verdicts{STAMP}.json")))
t = json.load(open(os.path.join(OUT, f"per_layer_tables{STAMP}.json")))
BAND = [5, 6, 7, 8, 9, 10]
# The layers to print are the layers the file holds. A complete file holds all
# twelve; a diagnostic scoring of a partial decomposition may hold fewer, and a
# row for a layer that was never decomposed would be an invention.
ALL = sorted(int(l) for l in t["lens"]["lang"])

# A scoring run over an incomplete shares file stamps its verdict file, and these
# tables must not be readable as final when it did. Say so at the top and go on,
# with every value the partial run could not compute printed as "not computed"
# rather than as a number.
PARTIAL = (not v.get("input_completeness", {}).get("input_complete", True)
           or bool(v.get("PARTIAL_DIAGNOSTIC_SCORING")) or ARGS.partial)
if PARTIAL:
    print("**PARTIAL DIAGNOSTIC SCORING. The verdict file these tables are built "
          "from was produced from a shares file that does not cover layers 0 to 11 "
          "for every scoring arm, so nothing below is a verdict on the "
          "pre-registered rules.**\n")
    if ALL != list(range(12)):
        print(f"**The decomposition behind it covers layers {ALL} only; the other "
              "layers have no row below.**\n")
    for entry in v.get("input_completeness", {}).get("not_computed", []):
        print(f"**Not computed from this input: {entry['what']}, because "
              f"{entry['why']}.**\n")


def band_mark(l):
    return f"**{l}**" if l in BAND else str(l)


def num(x, spec=".4f"):
    """One cell: the number, or a plain statement that it was not computed."""
    return "not computed" if x is None else format(x, spec)


def tnum(arm, fam, l, key="median", spec=".4f"):
    """One cell from the per-layer table file, tolerating an absent arm."""
    e = t.get(arm, {}).get(fam, {}).get(str(l))
    return num(None if e is None else e.get(key), spec)


def yesno(x):
    return "not computed" if x is None else ("yes" if x else "no")


print("### TABLE A: median J-space share by layer and family, lens against both controls\n")
print("| layer | language terminals | run-17 noise terminals | ordinary residuals | "
      "original noise arm | rotated-lens control (language) | random-dictionary control (language) |")
print("|---|---|---|---|---|---|---|")
for l in ALL:
    print(f"| {band_mark(l)} | {tnum('lens', 'lang', l)} | "
          f"{tnum('lens', 'noise17', l)} | "
          f"{tnum('lens', 'clean_last', l)} | "
          f"{tnum('lens', 'nullold', l)} | "
          f"{tnum('control_rotation_pooled', 'lang', l)} | "
          f"{tnum('control_gaussian_pooled', 'lang', l)} |")

# The last three columns are the comparison specification section 7.1 asks to be
# reported alongside the rule and kept out of it: the same one-sided test run on
# the two families' shares against the norm-matched random dictionary, pooled over
# its three seeds.
print("\n### TABLE B: H6, five basin representatives against eighteen null-model basins\n")
print("| layer | five basins, median | eighteen null basins, median | "
      "one-sided p (basins greater) | five basins, random-dictionary control | "
      "eighteen null basins, random-dictionary control | "
      "one-sided p under the random-dictionary control (basins greater) |")
print("|---|---|---|---|---|---|---|")
for l in ALL:
    e = v["H6"]["per_layer"][str(l)]
    print(f"| {band_mark(l)} | {num(e['basin_median'])} | {num(e['null_median'])} | "
          f"{num(e['p_greater'])} | {num(e.get('basin_control_gaussian_median'))} | "
          f"{num(e.get('null_control_gaussian_median'))} | "
          f"{num(e.get('control_gaussian_p_greater'))} |")

print("\n### TABLE C: H16, language terminals against run-17 noise terminals\n")
print("| layer | language, median | noise, median | difference | permutation p | "
      "language above random-dictionary control | language above rotated-lens control |")
print("|---|---|---|---|---|---|---|")
for l in ALL:
    e = v["H16"]["per_layer"][str(l)]
    print(f"| {band_mark(l)} | {num(e['lang_median'])} | {num(e['noise_median'])} | "
          f"{num(e['median_difference'], '+.4f')} | {num(e['p_language_greater'])} | "
          f"{yesno(e['lang_above_gaussian_control'])} | "
          f"{yesno(e['lang_above_rotation_control'])} |")

# The three Divine columns are named for the vector injected at the trace's input,
# not for what the trace holds at the layer being read: the trace injected from
# phase A carries phase B at layer 11, and the trace injected from phase B carries
# phase A there. Layers 0 to 10 hold intermediate residuals of one loop step.
print("\n### TABLE D: H16a, the prolet attractor against the Divine cycle\n")
print("| layer | prolet | trace injected from phase A | trace injected from phase B | "
      "trace injected from pivot M | prolet minus the phase-A trace | "
      "prolet minus the phase-B trace | seed spread within the rotated-lens control "
      "(one standard deviation) | pooled spread over all six control runs "
      "(one standard deviation) |")
print("|---|---|---|---|---|---|---|---|---|")
for l in ALL:
    e = v["H16a"]["per_layer"][str(l)]
    print(f"| {band_mark(l)} | {num(e['prolet'])} | {num(e['phaseA'])} | {num(e['phaseB'])} | "
          f"{num(e['pivotM'])} | {num(e['gap_prolet_minus_phaseA'], '+.4f')} | "
          f"{num(e['gap_prolet_minus_phaseB'], '+.4f')} | "
          f"{num(e['prolet_control_spread_sd_rotation'], '.5f')} | "
          f"{num(e['prolet_control_spread_sd'])} |")

print("\n### TABLE E: H16b, terminals against the same prompts' ordinary residuals\n")
print("| layer | terminal, median | ordinary residual, median | median paired difference | "
      "permutation p (terminal lower) | share of the 125 pairs with the terminal lower |")
print("|---|---|---|---|---|---|")
for l in ALL:
    e = v["H16b"]["per_layer"][str(l)]
    pct = e.get("fraction_pairs_terminal_lower")
    print(f"| {band_mark(l)} | {num(e['terminal_median'])} | {num(e['clean_last_median'])} | "
          f"{num(e['median_paired_difference'], '+.5f')} | {num(e['p_terminal_lower'])} | "
          f"{'not computed' if pct is None else format(pct * 100, '.0f') + ' percent'} |")

print("\n### TABLE F: named states, J-space share by layer\n")
keys = ["prolet1000", "phaseA", "phaseB", "pivotM", "noise1000"]
headers = ["prolet1000", "trace from phase A (phase B at layer 11)",
           "trace from phase B (phase A at layer 11)", "trace from pivot M",
           "noise1000"]
print("| layer | " + " | ".join(headers) + " |")
print("|---" * (len(keys) + 1) + "|")
for l in ALL:
    print(f"| {band_mark(l)} | " + " | ".join(
        num(v["descriptive"]["named_states"][k].get(str(l))) for k in keys) + " |")

print("\n### TABLE G: median number of directions the search selected before it ran out\n")
print("| layer | language terminals | run-17 noise | ordinary residuals | original noise arm |")
print("|---|---|---|---|---|")
for l in ALL:
    print(f"| {band_mark(l)} | {tnum('lens', 'lang', l, 'median_n_atoms', '.0f')} | "
          f"{tnum('lens', 'noise17', l, 'median_n_atoms', '.0f')} | "
          f"{tnum('lens', 'clean_last', l, 'median_n_atoms', '.0f')} | "
          f"{tnum('lens', 'nullold', l, 'median_n_atoms', '.0f')} |")
