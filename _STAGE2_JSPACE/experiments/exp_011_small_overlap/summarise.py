"""EXP_011: print a compact digest of the scored results, for the write-up."""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
v = json.load(open(os.path.join(OUT, "verdicts.json")))
t = json.load(open(os.path.join(OUT, "per_layer_tables.json")))
LAY = [str(l) for l in range(12)]
BAND = [str(l) for l in (5, 6, 7, 8, 9, 10)]

print("== median J-space share by layer (lens arm) ==")
hdr = "family".ljust(34) + "".join(f"L{l:<7}" for l in range(12))
print(hdr)
for fam in ("lang", "noise17", "nullold", "clean_last", "clean_mean"):
    row = "".join(f"{t['lens'][fam][l]['median']:<8.4f}" for l in LAY)
    print(fam.ljust(34) + row)
for lab, key in (("rotation control (pooled)", "control_rotation_pooled"),
                 ("gaussian control (pooled)", "control_gaussian_pooled")):
    for fam in ("lang", "noise17", "clean_last", "nullold"):
        row = "".join(f"{t[key][fam][l]['median']:<8.4f}" for l in LAY)
        print(f"{lab}/{fam}".ljust(34) + row)
print("\n== median atoms selected (lens arm) ==")
for fam in ("lang", "noise17", "nullold", "clean_last"):
    print(fam.ljust(34) + "".join(f"{t['lens'][fam][l]['median_n_atoms']:<8.0f}" for l in LAY))
print("\n== centred arm equals raw arm? (lens vs lens_centred, max |diff| over families/layers) ==")
mx = 0.0
for fam in t["lens"]:
    if fam in t.get("lens_centred", {}):
        for l in LAY:
            mx = max(mx, abs(t["lens"][fam][l]["median"] - t["lens_centred"][fam][l]["median"]))
print(f"max absolute difference in medians: {mx:.3e}")

for H in ("H6", "H16", "H16a", "H16b"):
    print(f"\n===== {H}: {v[H]['verdict']} =====")
    if H == "H6":
        print(" representatives:", v[H]["basin_representatives"])
        print(" layer  basin_med  null_med   p_greater  p_less   basin_gauss_ctrl  null_gauss_ctrl")
        for l in LAY:
            e = v[H]["per_layer"][l]
            mark = " *" if l in BAND else "  "
            print(f"{mark}{l:>4}  {e['basin_median']:9.4f}  {e['null_median']:8.4f}  "
                  f"{e['p_greater']:9.4f}  {e['p_less']:7.4f}  {e['basin_control_gaussian_median']:15.4f}  "
                  f"{e['null_control_gaussian_median']:14.4f}")
        print(" supporting layers:", v[H]["band_layers_supporting"], "refuting:", v[H]["band_layers_refuting"])
        print(" secondary all-125-vs-125:", {k: (round(x['lang_median'],4), round(x['nullold_median'],4), x['p_language_greater_mannwhitney']) for k, x in v[H]["all_states_secondary"].items()})
    if H == "H16":
        print(" layer  lang_med  noise_med  diff      p_lang>   lang_gauss_ctrl  lang_rot_ctrl  above_g  above_r")
        for l in LAY:
            e = v[H]["per_layer"][l]
            mark = " *" if l in BAND else "  "
            print(f"{mark}{l:>4}  {e['lang_median']:8.4f}  {e['noise_median']:9.4f}  "
                  f"{e['median_difference']:+8.4f}  {e['p_language_greater']:8.4f}  "
                  f"{e['lang_control_gaussian_median']:14.4f}  {e['lang_control_rotation_median']:13.4f}  "
                  f"{str(e['lang_above_gaussian_control']):7s}  {str(e['lang_above_rotation_control'])}")
        print(" supporting:", v[H]["band_layers_supporting"], "refuting:", v[H]["band_layers_refuting"])
        print(" converged-only secondary:", {k: (round(x['median_difference'],5), x['p_language_greater']) for k, x in v[H]["converged_only_secondary"].items()})
    if H == "H16a":
        print(" layer  prolet   phaseA   phaseB   pivotM   gapA      gapB      ctrl_sd   pilot_Divine")
        for l in LAY:
            e = v[H]["per_layer"][l]
            mark = " *" if l in BAND else "  "
            print(f"{mark}{l:>4}  {e['prolet']:7.4f}  {e['phaseA']:7.4f}  {e['phaseB']:7.4f}  "
                  f"{e['pivotM']:7.4f}  {e['gap_prolet_minus_phaseA']:+8.4f}  "
                  f"{e['gap_prolet_minus_phaseB']:+8.4f}  {e['prolet_control_spread_sd']:7.4f}  "
                  f"{e['pilot_divine_state']:7.4f}")
        print(" prolet above phaseA at:", v[H]["band_layers_prolet_above_phaseA"])
        print(" prolet above phaseB at:", v[H]["band_layers_prolet_above_phaseB"])
    if H == "H16b":
        print(" layer  term_med  clean_med  paired_diff  p_lower  frac_pairs_lower")
        for l in LAY:
            e = v[H]["per_layer"][l]
            mark = " *" if l in BAND else "  "
            print(f"{mark}{l:>4}  {e['terminal_median']:8.4f}  {e['clean_last_median']:9.4f}  "
                  f"{e['median_paired_difference']:+11.5f}  {e['p_terminal_lower']:7.4f}  "
                  f"{e['fraction_pairs_terminal_lower']:.3f}")
        print(" supporting:", v[H]["band_layers_supporting"], "refuting:", v[H]["band_layers_refuting"])
        print(" clean_mean secondary:", {k: (round(x['median_paired_difference'],5), x['p_terminal_lower']) for k, x in v[H]["clean_mean_secondary"].items()})
print("\n== named states (lens) ==")
for k, d in v["descriptive"]["named_states"].items():
    print(k.ljust(34) + "".join(f"{d[l]:<8.4f}" for l in LAY))
print("== directions ==")
for k, d in v["descriptive"]["directions"].items():
    print(k.ljust(34) + "".join(f"{d[l]:<8.4f}" for l in LAY))
