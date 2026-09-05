"""EXP_011: emit the markdown tables the results record uses, straight from the JSON."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
v = json.load(open(os.path.join(OUT, "verdicts.json")))
t = json.load(open(os.path.join(OUT, "per_layer_tables.json")))
BAND = [5, 6, 7, 8, 9, 10]
ALL = list(range(12))

# A scoring run over an incomplete shares file stamps its verdict file, and these
# tables must not be readable as final when it did. Say so at the top and stop,
# because the tables below index every one of the twelve layers.
if not v.get("input_completeness", {}).get("input_complete", True):
    print("**PARTIAL DIAGNOSTIC SCORING. The verdict file these tables are built "
          "from was produced from a shares file that does not cover layers 0 to 11 "
          "for every scoring arm, so nothing below is a verdict on the "
          "pre-registered rules.**\n")


def band_mark(l):
    return f"**{l}**" if l in BAND else str(l)


print("### TABLE A: median J-space share by layer and family, lens against both controls\n")
print("| layer | language terminals | run-17 noise terminals | ordinary residuals | "
      "original noise arm | rotated-lens control (language) | random-dictionary control (language) |")
print("|---|---|---|---|---|---|---|")
for l in ALL:
    print(f"| {band_mark(l)} | {t['lens']['lang'][str(l)]['median']:.4f} | "
          f"{t['lens']['noise17'][str(l)]['median']:.4f} | "
          f"{t['lens']['clean_last'][str(l)]['median']:.4f} | "
          f"{t['lens']['nullold'][str(l)]['median']:.4f} | "
          f"{t['control_rotation_pooled']['lang'][str(l)]['median']:.4f} | "
          f"{t['control_gaussian_pooled']['lang'][str(l)]['median']:.4f} |")

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
    print(f"| {band_mark(l)} | {e['basin_median']:.4f} | {e['null_median']:.4f} | "
          f"{e['p_greater']:.4f} | {e['basin_control_gaussian_median']:.4f} | "
          f"{e['null_control_gaussian_median']:.4f} | "
          f"{e['control_gaussian_p_greater']:.4f} |")

print("\n### TABLE C: H16, language terminals against run-17 noise terminals\n")
print("| layer | language, median | noise, median | difference | permutation p | "
      "language above random-dictionary control | language above rotated-lens control |")
print("|---|---|---|---|---|---|---|")
for l in ALL:
    e = v["H16"]["per_layer"][str(l)]
    print(f"| {band_mark(l)} | {e['lang_median']:.4f} | {e['noise_median']:.4f} | "
          f"{e['median_difference']:+.4f} | {e['p_language_greater']:.4f} | "
          f"{'yes' if e['lang_above_gaussian_control'] else 'no'} | "
          f"{'yes' if e['lang_above_rotation_control'] else 'no'} |")

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
    print(f"| {band_mark(l)} | {e['prolet']:.4f} | {e['phaseA']:.4f} | {e['phaseB']:.4f} | "
          f"{e['pivotM']:.4f} | {e['gap_prolet_minus_phaseA']:+.4f} | "
          f"{e['gap_prolet_minus_phaseB']:+.4f} | "
          f"{e['prolet_control_spread_sd_rotation']:.5f} | "
          f"{e['prolet_control_spread_sd']:.4f} |")

print("\n### TABLE E: H16b, terminals against the same prompts' ordinary residuals\n")
print("| layer | terminal, median | ordinary residual, median | median paired difference | "
      "permutation p (terminal lower) | share of the 125 pairs with the terminal lower |")
print("|---|---|---|---|---|---|")
for l in ALL:
    e = v["H16b"]["per_layer"][str(l)]
    print(f"| {band_mark(l)} | {e['terminal_median']:.4f} | {e['clean_last_median']:.4f} | "
          f"{e['median_paired_difference']:+.5f} | {e['p_terminal_lower']:.4f} | "
          f"{e['fraction_pairs_terminal_lower']*100:.0f} percent |")

print("\n### TABLE F: named states, J-space share by layer\n")
keys = ["prolet1000", "phaseA", "phaseB", "pivotM", "noise1000"]
headers = ["prolet1000", "trace from phase A (phase B at layer 11)",
           "trace from phase B (phase A at layer 11)", "trace from pivot M",
           "noise1000"]
print("| layer | " + " | ".join(headers) + " |")
print("|---" * (len(keys) + 1) + "|")
for l in ALL:
    print(f"| {band_mark(l)} | " + " | ".join(
        f"{v['descriptive']['named_states'][k][str(l)]:.4f}" for k in keys) + " |")

print("\n### TABLE G: median number of directions the search selected before it ran out\n")
print("| layer | language terminals | run-17 noise | ordinary residuals | original noise arm |")
print("|---|---|---|---|---|")
for l in ALL:
    print(f"| {band_mark(l)} | {t['lens']['lang'][str(l)]['median_n_atoms']:.0f} | "
          f"{t['lens']['noise17'][str(l)]['median_n_atoms']:.0f} | "
          f"{t['lens']['clean_last'][str(l)]['median_n_atoms']:.0f} | "
          f"{t['lens']['nullold'][str(l)]['median_n_atoms']:.0f} |")
