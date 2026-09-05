"""EXP_011 stage 3: score H6, H16, H16a and H16b on the pre-registered rules.

Reads output/shares.json and output/states_meta.json, applies exactly the rules
written in _STAGE2_JSPACE/EXP_011_SPEC.md section 7, and writes the verdict JSON,
the per-layer tables (JSON and CSV) and the figures.

Run: python3 score.py
"""
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.stats import mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
FROZEN = "/home/user/shared/stage1_frozen/experiments/gpt2_small"

BAND = [5, 6, 7, 8, 9, 10]          # spec section 6
MAJORITY = 4                         # at least four of the six band layers
ALPHA = 0.05
N_PERM = 10000
PERM_SEED = 11011
ROT = ["rot2026", "rot2027", "rot2028"]
GAUSS = ["gauss4242", "gauss4243", "gauss4244"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


shares = json.load(open(os.path.join(OUT, "shares.json")))
meta = json.load(open(os.path.join(OUT, "states_meta.json")))
LAYERS = sorted(int(x) for x in shares["arms"]["lens"]["lang"].keys())


def arr(arm, fam, layer):
    return np.array(shares["arms"][arm][fam][str(layer)]["share"], dtype=float)


def natoms(arm, fam, layer):
    return np.array(shares["arms"][arm][fam][str(layer)]["n_atoms"], dtype=float)


def pooled(arms, fam, layer):
    return np.concatenate([arr(a, fam, layer) for a in arms])


def perm_two_sample(x, y, rng, n_perm=N_PERM):
    """One-sided permutation test on the difference of medians, x greater."""
    obs = float(np.median(x) - np.median(y))
    pool = np.concatenate([x, y])
    nx = len(x)
    ge = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        if float(np.median(pool[:nx]) - np.median(pool[nx:])) >= obs:
            ge += 1
    return obs, (1 + ge) / (1 + n_perm)


def perm_sign_flip(d, rng, n_perm=N_PERM):
    """One-sided sign-flip test on paired differences, median difference below 0."""
    obs = float(np.median(d))
    le = 0
    for _ in range(n_perm):
        s = rng.choice([-1.0, 1.0], size=len(d))
        if float(np.median(d * s)) <= obs:
            le += 1
    return obs, (1 + le) / (1 + n_perm)


# ------------------------------------------------------------- descriptive ---
log("building the per-layer share tables")
FAMS = ["lang", "noise17", "nullold", "clean_last", "clean_mean", "named", "directions"]
table = {}
for arm in shares["arms"]:
    table[arm] = {}
    for fam in FAMS:
        if fam not in shares["arms"][arm]:
            continue
        table[arm][fam] = {
            str(l): {
                "n": int(len(arr(arm, fam, l))),
                "median": float(np.median(arr(arm, fam, l))),
                "mean": float(np.mean(arr(arm, fam, l))),
                "q25": float(np.percentile(arr(arm, fam, l), 25)),
                "q75": float(np.percentile(arr(arm, fam, l), 75)),
                "median_n_atoms": float(np.median(natoms(arm, fam, l))),
            } for l in LAYERS}
# control summaries: the two controls pooled over their three seeds
for label, arms in (("control_rotation_pooled", ROT), ("control_gaussian_pooled", GAUSS)):
    table[label] = {}
    for fam in FAMS:
        if fam not in shares["arms"][arms[0]]:
            continue
        table[label][fam] = {
            str(l): {"n": int(len(pooled(arms, fam, l))),
                     "median": float(np.median(pooled(arms, fam, l))),
                     "q25": float(np.percentile(pooled(arms, fam, l), 25)),
                     "q75": float(np.percentile(pooled(arms, fam, l), 75))}
            for l in LAYERS}

verdicts = {"band_layers": BAND, "majority_needed": MAJORITY, "alpha": ALPHA,
            "n_permutations": N_PERM, "permutation_seed": PERM_SEED}

named_keys = meta["named"]["keys"]
NIDX = {k: i for i, k in enumerate(named_keys)}

# ------------------------------------------------------------------- H6 ------
log("H6: five basin representatives against eighteen null-model basins")
stage1 = torch.load(os.path.join(FROZEN, "output/stage1_results.pt"),
                    weights_only=False, map_location="cpu")
nullold_raw = torch.load(os.path.join(FROZEN, "output_random_baseline/random_baseline_results.pt"),
                         weights_only=False, map_location="cpu")


def medoid_indices(ids, labels, vectors):
    """One representative index per label: highest mean cosine to the others."""
    out = {}
    for lab in sorted(set(labels)):
        members = [i for i, x in enumerate(labels) if x == lab]
        V = torch.stack([vectors[i] for i in members])
        V = V / V.norm(dim=1, keepdim=True)
        C = V @ V.T
        if len(members) == 1:
            out[lab] = members[0]
            continue
        mean_cos = (C.sum(dim=1) - 1.0) / (len(members) - 1)
        best = int(torch.argmax(mean_cos))
        ties = [j for j in range(len(members))
                if abs(float(mean_cos[j]) - float(mean_cos[best])) < 1e-9]
        out[lab] = members[min(ties)]       # first by identifier among exact ties
    return out


lang_ids = meta["lang"]["ids"]
lang_labels = meta["lang"]["label_iter100"]
lang_vecs = [stage1[p]["last_vectors"][-1] for p in lang_ids]
lang_med = medoid_indices(lang_ids, lang_labels, lang_vecs)

old_ids = meta["nullold"]["ids"]
old_labels = meta["nullold"]["label_iter100"]
old_vecs = [nullold_raw[t][-1]["last_vector"] for t in old_ids]
old_med = medoid_indices(old_ids, old_labels, old_vecs)
log(f"  five basins: { {k: lang_ids[v] for k, v in lang_med.items()} }")
log(f"  eighteen null basins: {len(old_med)} representatives")

h6 = {"basin_representatives": {k: lang_ids[v] for k, v in lang_med.items()},
      "null_representatives": {k: old_ids[v] for k, v in old_med.items()},
      "per_layer": {}}
li = sorted(lang_med.values())
oi = sorted(old_med.values())
for l in LAYERS:
    x = arr("lens", "lang", l)[li]
    y = arr("lens", "nullold", l)[oi]
    u_greater = mannwhitneyu(x, y, alternative="greater")
    u_less = mannwhitneyu(x, y, alternative="less")
    h6["per_layer"][str(l)] = {
        "basin_median": float(np.median(x)), "null_median": float(np.median(y)),
        "basin_shares": [float(v) for v in x], "null_shares": [float(v) for v in y],
        "p_greater": float(u_greater.pvalue), "p_less": float(u_less.pvalue),
        "basin_control_gaussian_median": float(np.median(
            np.concatenate([arr(a, "lang", l)[li] for a in GAUSS]))),
        "basin_control_rotation_median": float(np.median(
            np.concatenate([arr(a, "lang", l)[li] for a in ROT]))),
        "null_control_gaussian_median": float(np.median(
            np.concatenate([arr(a, "nullold", l)[oi] for a in GAUSS]))),
    }
hits_g = [l for l in BAND if h6["per_layer"][str(l)]["basin_median"] > h6["per_layer"][str(l)]["null_median"]
          and h6["per_layer"][str(l)]["p_greater"] < ALPHA]
hits_l = [l for l in BAND if h6["per_layer"][str(l)]["null_median"] > h6["per_layer"][str(l)]["basin_median"]
          and h6["per_layer"][str(l)]["p_less"] < ALPHA]
h6["band_layers_supporting"] = hits_g
h6["band_layers_refuting"] = hits_l
h6["verdict"] = ("SUPPORTED" if len(hits_g) >= MAJORITY else
                 "REFUTED" if len(hits_l) >= MAJORITY else "NOT SUPPORTED")
# reported alongside, not scoring: all 125 against all 125
h6["all_states_secondary"] = {}
rng = np.random.default_rng(PERM_SEED)
for l in BAND:
    obs, p = perm_two_sample(arr("lens", "lang", l), arr("lens", "nullold", l), rng)
    h6["all_states_secondary"][str(l)] = {
        "median_difference": obs, "p_language_greater_permutation": p,
        "p_language_greater_mannwhitney": float(mannwhitneyu(
            arr("lens", "lang", l), arr("lens", "nullold", l),
            alternative="greater").pvalue),
        "lang_median": float(np.median(arr("lens", "lang", l))),
        "nullold_median": float(np.median(arr("lens", "nullold", l)))}
verdicts["H6"] = h6
log(f"  H6 verdict: {h6['verdict']} (supporting layers {hits_g}, refuting {hits_l})")

# ------------------------------------------------------------------ H16 ------
log("H16: language terminals against the run-17 matched-scale noise terminals")
rng = np.random.default_rng(PERM_SEED)
h16 = {"per_layer": {}}
conv_mask = np.array(meta["noise17"]["converged"], dtype=bool)
for l in LAYERS:
    x, y = arr("lens", "lang", l), arr("lens", "noise17", l)
    obs, p = perm_two_sample(x, y, rng)
    ctrl_g = float(np.median(pooled(GAUSS, "lang", l)))
    ctrl_r = float(np.median(pooled(ROT, "lang", l)))
    h16["per_layer"][str(l)] = {
        "lang_median": float(np.median(x)), "noise_median": float(np.median(y)),
        "median_difference": obs, "p_language_greater": p,
        "lang_control_gaussian_median": ctrl_g, "lang_control_rotation_median": ctrl_r,
        "noise_control_gaussian_median": float(np.median(pooled(GAUSS, "noise17", l))),
        "lang_above_gaussian_control": float(np.median(x)) > ctrl_g,
        "lang_above_rotation_control": float(np.median(x)) > ctrl_r,
    }
sup = [l for l in BAND
       if h16["per_layer"][str(l)]["median_difference"] > 0
       and h16["per_layer"][str(l)]["p_language_greater"] < ALPHA
       and h16["per_layer"][str(l)]["lang_above_gaussian_control"]]
ref = [l for l in BAND if h16["per_layer"][str(l)]["median_difference"] < 0]
ref_p = []
rng2 = np.random.default_rng(PERM_SEED + 1)
for l in ref:
    _, p_rev = perm_two_sample(arr("lens", "noise17", l), arr("lens", "lang", l), rng2)
    h16["per_layer"][str(l)]["p_noise_greater"] = p_rev
    if p_rev < ALPHA:
        ref_p.append(l)
h16["band_layers_supporting"] = sup
h16["band_layers_refuting"] = ref_p
h16["verdict"] = ("SUPPORTED" if len(sup) >= MAJORITY else
                  "REFUTED" if len(ref_p) >= MAJORITY else "NOT SUPPORTED")
# robustness, not scoring
h16["converged_only_secondary"] = {}
rng3 = np.random.default_rng(PERM_SEED + 2)
for l in BAND:
    obs, p = perm_two_sample(arr("lens", "lang", l), arr("lens", "noise17", l)[conv_mask], rng3)
    h16["converged_only_secondary"][str(l)] = {
        "n_noise": int(conv_mask.sum()), "median_difference": obs, "p_language_greater": p}
h16["rotation_control_secondary"] = {
    str(l): {"lang_median": float(np.median(arr("lens", "lang", l))),
             "lang_rotation_control_median": float(np.median(pooled(ROT, "lang", l))),
             "noise_median": float(np.median(arr("lens", "noise17", l))),
             "noise_rotation_control_median": float(np.median(pooled(ROT, "noise17", l)))}
    for l in BAND}
verdicts["H16"] = h16
log(f"  H16 verdict: {h16['verdict']} (supporting {sup}, refuting {ref_p})")

# ----------------------------------------------------------------- H16a ------
log("H16a: the prolet attractor against the Divine cycle's two phases")
h16a = {"per_layer": {}}
for l in LAYERS:
    nm = arr("lens", "named", l)
    pro = float(nm[NIDX["prolet1000"]])
    pa, pb, pm = (float(nm[NIDX["phaseA"]]), float(nm[NIDX["phaseB"]]),
                  float(nm[NIDX["pivotM"]]))
    ctrl = [float(arr(a, "named", l)[NIDX["prolet1000"]]) for a in ROT + GAUSS]
    spread = float(np.std(ctrl))
    h16a["per_layer"][str(l)] = {
        "prolet": pro, "phaseA": pa, "phaseB": pb, "pivotM": pm,
        "gap_prolet_minus_phaseA": pro - pa, "gap_prolet_minus_phaseB": pro - pb,
        "prolet_control_spread_sd": spread,
        "gapA_inside_control_spread": abs(pro - pa) < spread,
        "gapB_inside_control_spread": abs(pro - pb) < spread,
        "prolet_control_gaussian_median": float(np.median(
            [float(arr(a, "named", l)[NIDX["prolet1000"]]) for a in GAUSS])),
        "pilot_prolet_states": {k: float(nm[NIDX[f"convtensor_{k}"]])
                                for k in ("Lucier", "Semantic", "Nonsense", "Imperative")},
        "pilot_divine_state": float(nm[NIDX["convtensor_Syntactic"]]),
    }
supA = [l for l in BAND if h16a["per_layer"][str(l)]["gap_prolet_minus_phaseA"] > 0]
supB = [l for l in BAND if h16a["per_layer"][str(l)]["gap_prolet_minus_phaseB"] > 0]
refA = [l for l in BAND if h16a["per_layer"][str(l)]["gap_prolet_minus_phaseA"] < 0]
refB = [l for l in BAND if h16a["per_layer"][str(l)]["gap_prolet_minus_phaseB"] < 0]
h16a["band_layers_prolet_above_phaseA"] = supA
h16a["band_layers_prolet_above_phaseB"] = supB
h16a["verdict"] = ("SUPPORTED" if len(supA) >= MAJORITY and len(supB) >= MAJORITY else
                   "REFUTED" if len(refA) >= MAJORITY and len(refB) >= MAJORITY else
                   "NOT SUPPORTED")
verdicts["H16a"] = h16a
log(f"  H16a verdict: {h16a['verdict']} (prolet above phase A at {supA}, above phase B at {supB})")

# ----------------------------------------------------------------- H16b ------
log("H16b: terminals against the same prompts' ordinary residuals")
rng = np.random.default_rng(PERM_SEED)
h16b = {"per_layer": {}}
for l in LAYERS:
    d = arr("lens", "lang", l) - arr("lens", "clean_last", l)
    obs, p_lower = perm_sign_flip(d, rng)
    h16b["per_layer"][str(l)] = {
        "terminal_median": float(np.median(arr("lens", "lang", l))),
        "clean_last_median": float(np.median(arr("lens", "clean_last", l))),
        "median_paired_difference": obs, "p_terminal_lower": p_lower,
        "n_pairs": int(len(d)),
        "fraction_pairs_terminal_lower": float((d < 0).mean()),
    }
sup = [l for l in BAND if h16b["per_layer"][str(l)]["median_paired_difference"] < 0
       and h16b["per_layer"][str(l)]["p_terminal_lower"] < ALPHA]
rng4 = np.random.default_rng(PERM_SEED + 3)
ref = []
for l in BAND:
    if h16b["per_layer"][str(l)]["median_paired_difference"] > 0:
        d = arr("lens", "clean_last", l) - arr("lens", "lang", l)
        _, p_hi = perm_sign_flip(d, rng4)
        h16b["per_layer"][str(l)]["p_terminal_higher"] = p_hi
        if p_hi < ALPHA:
            ref.append(l)
h16b["band_layers_supporting"] = sup
h16b["band_layers_refuting"] = ref
h16b["verdict"] = ("SUPPORTED" if len(sup) >= MAJORITY else
                   "REFUTED" if len(ref) >= MAJORITY else "NOT SUPPORTED")
h16b["clean_mean_secondary"] = {}
rng5 = np.random.default_rng(PERM_SEED + 4)
for l in BAND:
    d = arr("lens", "lang", l) - arr("lens", "clean_mean", l)
    obs, p = perm_sign_flip(d, rng5)
    h16b["clean_mean_secondary"][str(l)] = {
        "clean_mean_median": float(np.median(arr("lens", "clean_mean", l))),
        "median_paired_difference": obs, "p_terminal_lower": p}
verdicts["H16b"] = h16b
log(f"  H16b verdict: {h16b['verdict']} (supporting {sup}, refuting {ref})")

# ----------------------------------------------------------- write it out ----
verdicts["descriptive"] = {
    "named_states": {k: {str(l): float(arr("lens", "named", l)[NIDX[k]]) for l in LAYERS}
                     for k in named_keys},
    "directions": {k: {str(l): float(arr("lens", "directions", l)[i]) for l in LAYERS}
                   for i, k in enumerate(meta["directions"]["keys"])},
    "dictionary_note": ("Every share is a fraction of the state's squared length "
                        "captured by at most 25 lens vectors with non-negative "
                        "coefficients; 0 means none of it, 1 means all of it."),
}
with open(os.path.join(OUT, "per_layer_tables.json"), "w") as fh:
    json.dump(table, fh, indent=1)
with open(os.path.join(OUT, "verdicts.json"), "w") as fh:
    json.dump(verdicts, fh, indent=1)

rows = ["arm,family,layer,n,median_share,q25,q75,mean_share,median_n_atoms"]
for arm in sorted(table):
    for fam in sorted(table[arm]):
        for l in LAYERS:
            e = table[arm][fam][str(l)]
            rows.append(f"{arm},{fam},{l},{e['n']},{e['median']:.6f},{e['q25']:.6f},"
                        f"{e['q75']:.6f},{e.get('mean', float('nan')):.6f},"
                        f"{e.get('median_n_atoms', float('nan')):.1f}")
with open(os.path.join(OUT, "per_layer_shares.csv"), "w") as fh:
    fh.write("\n".join(rows) + "\n")

# --------------------------------------------------------------- figures ----
log("drawing the figures")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
series = [("lang", "language terminals (125)", "#1f77b4", "-"),
          ("noise17", "run-17 noise terminals (125)", "#d62728", "-"),
          ("clean_last", "ordinary prompt residuals (125)", "#2ca02c", "-"),
          ("nullold", "original noise arm (125)", "#9467bd", "--")]
for fam, lab, c, ls in series:
    med = [table["lens"][fam][str(l)]["median"] for l in LAYERS]
    q25 = [table["lens"][fam][str(l)]["q25"] for l in LAYERS]
    q75 = [table["lens"][fam][str(l)]["q75"] for l in LAYERS]
    ax.plot(LAYERS, med, ls, color=c, label=lab)
    ax.fill_between(LAYERS, q25, q75, color=c, alpha=0.12)
for fam, lab, c in [("lang", "language terminals, rotated-lens control", "#1f77b4"),
                    ("lang", "language terminals, Gaussian control", "#7f7f7f")]:
    arms = ROT if "rotated" in lab else GAUSS
    med = [table["control_rotation_pooled" if "rotated" in lab else
                 "control_gaussian_pooled"][fam][str(l)]["median"] for l in LAYERS]
    ax.plot(LAYERS, med, ":", color=c, label=lab)
ax.axvspan(4.6, 10.4, color="gold", alpha=0.12)
ax.text(7.5, ax.get_ylim()[1] * 0.97, "workspace band", ha="center", va="top", fontsize=8)
ax.set_xlabel("layer (output of block l)")
ax.set_ylabel("J-space share (fraction of squared length)")
ax.set_title("J-space share by layer, GPT-2 Small")
ax.legend(fontsize=7)
ax.grid(alpha=0.25)

ax = axes[1]
for key, lab, c in [("prolet1000", "prolet attractor", "#1f77b4"),
                    ("phaseA", "Divine phase A", "#d62728"),
                    ("phaseB", "Divine phase B", "#ff7f0e"),
                    ("pivotM", "Divine pivot M", "#8c564b"),
                    ("noise1000", "pilot noise state", "#7f7f7f")]:
    ax.plot(LAYERS, [float(arr("lens", "named", l)[NIDX[key]]) for l in LAYERS],
            "-o", ms=3, color=c, label=lab)
ax.axvspan(4.6, 10.4, color="gold", alpha=0.12)
ax.set_xlabel("layer (output of block l)")
ax.set_ylabel("J-space share")
ax.set_title("Named states: prolet against the Divine cycle")
ax.legend(fontsize=7)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "exp011_share_curves.png"), dpi=150)
log("wrote per_layer_tables.json, verdicts.json, per_layer_shares.csv, "
    "exp011_share_curves.png")
