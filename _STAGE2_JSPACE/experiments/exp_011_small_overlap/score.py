"""EXP_011 stage 3: score H6, H16, H16a and H16b on the pre-registered rules.

Reads output/shares.json and output/states_meta.json, applies exactly the rules
written in _STAGE2_JSPACE/EXP_011_SPEC.md section 7, and writes the verdict JSON,
the per-layer tables (JSON and CSV) and the figures.

The shares file must cover layers 0 to 11 for every arm the scoring reads and must
not mark itself partial: no final verdict, table or figure is produced from a
partial decomposition. Pass --allow-partial for a diagnostic scoring, whose four
outputs are renamed and stamped partial and carry no verdict.

Run: python3 score.py [--allow-partial]
"""
import argparse
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
# Spec section 7.1: "All permutation tests use 10,000 permutations with seed
# 11011." Every generator in this script is seeded with exactly that. The first
# version of this file offset the secondary and reverse-direction tests to
# 11012 through 11015, which the specification does not license; the offsets were
# removed on 2026-09-05 and the change is recorded as a deviation in the results
# record, with the before-and-after p-values.
PERM_SEED = 11011
ROT = ["rot2026", "rot2027", "rot2028"]
GAUSS = ["gauss4242", "gauss4243", "gauss4244"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


ap = argparse.ArgumentParser(description="Score EXP_011 on the pre-registered rules.")
ap.add_argument("--allow-partial", action="store_true",
                help="score a shares file that does not cover all twelve layers. "
                     "Diagnostic only: every file written is renamed and stamped "
                     "partial, and no verdict it contains is final.")
ARGS = ap.parse_args()

shares = json.load(open(os.path.join(OUT, "shares.json")))
meta = json.load(open(os.path.join(OUT, "states_meta.json")))
LAYERS = sorted(int(x) for x in shares["arms"]["lens"]["lang"].keys())

# ------------------------------------------------------- completeness gate ---
# Every verdict in specification section 7 is scored on the six band layers, so a
# shares file holding only layers 5 to 10 would let this script emit final
# verdicts, tables and a figure from an admittedly partial decomposition. It must
# not. The verdicts, the tables and the figure all cover layers 0 to 11, the
# descriptive readings of section 7.5 item 1 name all twelve, and decompose.py
# marks a run that did not cover them with partial_run: true. So require both: the
# flag must not be set, and every arm the scoring reads must carry all twelve
# layers for every family it holds. --allow-partial permits a diagnostic run, and
# then every output is written under a different name and stamped.
ALL_LAYERS = list(range(12))
SCORING_ARMS = ["lens"] + ROT + GAUSS
_gaps = {}
for _arm in SCORING_ARMS:
    if _arm not in shares["arms"]:
        _gaps[_arm] = "the arm is absent from the file"
        continue
    for _fam, _layers in sorted(shares["arms"][_arm].items()):
        _have = {int(x) for x in _layers}
        _absent = [l for l in ALL_LAYERS if l not in _have]
        if _absent:
            _gaps.setdefault(_arm, {})[_fam] = _absent
PARTIAL_FLAG = shares.get("partial_run")     # absent in files written before 2026-09-05
INCOMPLETE = bool(_gaps) or PARTIAL_FLAG is True
# The same gaps in one line, for a refusal a person has to read.
_missing_layers = sorted({l for v in _gaps.values() if isinstance(v, dict)
                          for ls in v.values() for l in ls})
_gap_summary = ("no layer gaps; the file marks itself partial" if not _gaps else
                "arms with gaps {}, layers missing somewhere {}, for example {}"
                .format(sorted(_gaps), _missing_layers,
                        {a: (v if isinstance(v, str) else
                             {f: ls for f, ls in list(v.items())[:1]})
                         for a, v in list(_gaps.items())[:1]}))
COMPLETENESS = {
    "layers_required": ALL_LAYERS,
    "layers_present_in_lens_arm": LAYERS,
    "arms_required": SCORING_ARMS,
    "partial_run_flag_in_shares": PARTIAL_FLAG,
    "gaps_by_arm": _gaps,
    "input_complete": not INCOMPLETE,
    "allow_partial_used": bool(ARGS.allow_partial),
}
if INCOMPLETE and not ARGS.allow_partial:
    raise SystemExit(
        "REFUSING TO SCORE: the shares file does not cover layers 0 to 11 for "
        "every arm the scoring reads, or it is marked partial_run: {}. Gaps: {}. "
        "Final verdicts, tables and figures are not produced from a partial "
        "decomposition. Complete the decomposition, or pass --allow-partial for a "
        "diagnostic scoring whose outputs are renamed and stamped partial."
        .format(PARTIAL_FLAG, _gap_summary))
if INCOMPLETE:
    log("PARTIAL DIAGNOSTIC SCORING: the shares file is incomplete "
        f"(partial_run flag {PARTIAL_FLAG}; {_gap_summary}). Every output is "
        "written under a .partial. name and carries no verdict.")
else:
    log(f"completeness check: layers {LAYERS[0]} to {LAYERS[-1]} present for every "
        f"family in all {len(SCORING_ARMS)} scoring arms, and the file is "
        + ("not marked partial" if PARTIAL_FLAG is False else
           "from before the partial_run flag existed, which decompose.py began "
           "writing on 2026-09-05"))


def outpath(name):
    """Where an output goes: its own name, or a stamped one for a partial run."""
    return os.path.join(OUT, name if not INCOMPLETE else name.replace(".", ".partial.", 1))


# ------------------------------------------------ iteration-safety-bound gate ---
# The decomposition's search carries a safety bound on the number of rounds. A
# decomposition that stops only because it reached that bound has not met any real
# stopping condition, so its share is not the number this experiment means to
# report. Refuse to score a file containing one. The field is absent from any file
# written before 2026-09-05, and a file can also be mixed, because a partial
# decomposition merges into an existing shares file entry by entry: the refreshed
# entries then carry the flag and the untouched ones do not. Counting the two
# cases apart is what keeps a mixed file from being reported as clear everywhere
# while the untouched entries are scored with their termination status unknown.
_with, _without, _flagged = [], [], []
for _arm, _fams in shares["arms"].items():
    for _fam, _layers in _fams.items():
        for _l, _entry in _layers.items():
            if "hit_max_iter" in _entry:
                _with.append((_arm, _fam, _l))
                if any(_entry["hit_max_iter"]):
                    _flagged.append((_arm, _fam, _l, int(sum(_entry["hit_max_iter"]))))
            else:
                _without.append((_arm, _fam, _l))
if _flagged:
    raise SystemExit(
        "REFUSING TO SCORE: {} arm-family-layer groups contain decompositions that "
        "stopped on the iteration safety bound rather than a real stopping "
        "condition, for example {}. Re-run the decomposition before scoring."
        .format(len(_flagged), _flagged[:5]))
FLAG_COVERAGE = {"groups_with_flag": len(_with), "groups_without_flag": len(_without),
                 "arms_without_flag": sorted({a for a, _, _ in _without}),
                 "families_without_flag": sorted({f for _, f, _ in _without}),
                 "layers_without_flag": sorted({int(l) for _, _, l in _without})}
if _with and _without:
    raise SystemExit(
        "REFUSING TO SCORE: the shares file is mixed. {} arm-family-layer groups "
        "carry the iteration-safety-bound flag and {} do not, so the ones that do "
        "not have an unknown termination status and would be scored blind. The "
        "groups lacking the flag are in arms {}, families {} and layers {}; for "
        "example {}. This is what a partial decomposition merged into a file "
        "written before 2026-09-05 looks like. Re-run the decomposition over the "
        "arms, families and layers named above before scoring."
        .format(len(_with), len(_without), FLAG_COVERAGE["arms_without_flag"],
                FLAG_COVERAGE["families_without_flag"],
                FLAG_COVERAGE["layers_without_flag"], _without[:5]))
log("iteration-safety-bound check: " + (
    f"present and clear in every arm, family and layer ({len(_with)} groups)"
    if _with else
    "the committed shares.json predates the flag and does not carry it, so this "
    "check cannot be applied to it; the flag was added to decompose.py on "
    "2026-09-05 and applies from the next decomposition run "
    f"({len(_without)} groups, none carrying the field)"))


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


def perm_sign_flip_higher(d, rng, n_perm=N_PERM):
    """One-sided sign-flip test on paired differences, median difference above 0."""
    obs = float(np.median(d))
    ge = 0
    for _ in range(n_perm):
        s = rng.choice([-1.0, 1.0], size=len(d))
        if float(np.median(d * s)) >= obs:
            ge += 1
    return obs, (1 + ge) / (1 + n_perm)


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
# What the input was, recorded beside the verdicts it produced: which layers the
# shares file covered, whether it marked itself partial, and whether every
# decomposition in it carried the iteration-safety-bound flag.
verdicts["input_completeness"] = COMPLETENESS
verdicts["iteration_safety_bound_flag_coverage"] = FLAG_COVERAGE

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
    # Specification section 7.1, reported alongside and explicitly not part of the
    # rule: "the same comparison against control (b)". Control (b) is the
    # norm-matched random dictionary at seeds 4242, 4243 and 4244. The same test
    # the rule uses, a one-sided Mann-Whitney U test of the five basin
    # representatives against the eighteen null representatives, is run here on
    # those control shares: once on the three seeds pooled, which is the level the
    # control medians beside it are already reported at, and once per seed. The
    # first version of this script recorded only the two control medians and never
    # ran the test, which is recorded as a deviation in the results record. It
    # cannot move the H6 verdict, because section 7.1's rule sentence scores the
    # lens comparison alone; control (b) enters a scoring rule only in section 7.2,
    # as the third condition of H16.
    xb = np.concatenate([arr(a, "lang", l)[li] for a in GAUSS])
    yb = np.concatenate([arr(a, "nullold", l)[oi] for a in GAUSS])
    per_seed_b = {}
    for a in GAUSS:
        xs, ys = arr(a, "lang", l)[li], arr(a, "nullold", l)[oi]
        per_seed_b[a] = {
            "basin_median": float(np.median(xs)),
            "null_median": float(np.median(ys)),
            "p_greater": float(mannwhitneyu(xs, ys, alternative="greater").pvalue),
            "p_less": float(mannwhitneyu(xs, ys, alternative="less").pvalue)}
    h6["per_layer"][str(l)].update({
        "control_gaussian_p_greater": float(
            mannwhitneyu(xb, yb, alternative="greater").pvalue),
        "control_gaussian_p_less": float(
            mannwhitneyu(xb, yb, alternative="less").pvalue),
        "control_gaussian_n_basin": int(len(xb)),
        "control_gaussian_n_null": int(len(yb)),
        "control_gaussian_per_seed": per_seed_b,
    })
hits_g = [l for l in BAND if h6["per_layer"][str(l)]["basin_median"] > h6["per_layer"][str(l)]["null_median"]
          and h6["per_layer"][str(l)]["p_greater"] < ALPHA]
hits_l = [l for l in BAND if h6["per_layer"][str(l)]["null_median"] > h6["per_layer"][str(l)]["basin_median"]
          and h6["per_layer"][str(l)]["p_less"] < ALPHA]
h6["band_layers_supporting"] = hits_g
h6["band_layers_refuting"] = hits_l
h6["verdict"] = ("SUPPORTED" if len(hits_g) >= MAJORITY else
                 "REFUTED" if len(hits_l) >= MAJORITY else "NOT SUPPORTED")
# Reported alongside, not scoring: the same five-against-eighteen test carried out
# on the control (b) shares, summarised over the band.
h6["control_gaussian_comparison"] = {
    "what": ("The same one-sided Mann-Whitney U test as the rule, five basin "
             "representatives against eighteen null-model representatives, applied "
             "to the norm-matched random-dictionary control shares of the same "
             "states, pooled over seeds 4242, 4243 and 4244 (15 against 54 "
             "shares). Specification section 7.1 asks for it as a reported "
             "comparison and keeps it out of the scoring rule."),
    "scoring": False,
    "band_layers_control_p_greater_below_alpha": [
        l for l in BAND if h6["per_layer"][str(l)]["control_gaussian_p_greater"] < ALPHA],
    "band_layers_control_p_less_below_alpha": [
        l for l in BAND if h6["per_layer"][str(l)]["control_gaussian_p_less"] < ALPHA],
}
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
log("  H6 under control (b), reported not scoring: band layers with the basins "
    f"above the nulls at p below {ALPHA}: "
    f"{h6['control_gaussian_comparison']['band_layers_control_p_greater_below_alpha']}; "
    "below them: "
    f"{h6['control_gaussian_comparison']['band_layers_control_p_less_below_alpha']}")

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
rng2 = np.random.default_rng(PERM_SEED)
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
rng3 = np.random.default_rng(PERM_SEED)
for l in BAND:
    obs, p = perm_two_sample(arr("lens", "lang", l), arr("lens", "noise17", l)[conv_mask], rng3)
    h16["converged_only_secondary"][str(l)] = {
        "n_noise": int(conv_mask.sum()), "median_difference": obs, "p_language_greater": p}
# Spec section 7.2 pre-registers, as robustness reported but not scoring, "the
# same test under control (a)". The primary test is the label-permutation test on
# the difference of medians between the 125 language and the 125 run-17 noise
# states; here it is run again on the shares those same states score against the
# rotated lens, once per rotation seed (the same 125-against-125 shape the
# primary test has) and once on the three seeds pooled. Same permutation count
# (10,000) and the same seed convention the other secondaries use.
h16["rotation_control_secondary"] = {}
rng6 = np.random.default_rng(PERM_SEED)
for l in BAND:
    per_seed = {}
    for s_arm in ROT:
        obs_s, p_s = perm_two_sample(arr(s_arm, "lang", l), arr(s_arm, "noise17", l), rng6)
        per_seed[s_arm] = {
            "lang_median": float(np.median(arr(s_arm, "lang", l))),
            "noise_median": float(np.median(arr(s_arm, "noise17", l))),
            "median_difference": obs_s, "p_language_greater": p_s}
    obs_p, p_p = perm_two_sample(pooled(ROT, "lang", l), pooled(ROT, "noise17", l), rng6)
    h16["rotation_control_secondary"][str(l)] = {
        "lang_median": float(np.median(arr("lens", "lang", l))),
        "lang_rotation_control_median": float(np.median(pooled(ROT, "lang", l))),
        "noise_median": float(np.median(arr("lens", "noise17", l))),
        "noise_rotation_control_median": float(np.median(pooled(ROT, "noise17", l))),
        "per_rotation_seed": per_seed,
        "pooled_median_difference": obs_p,
        "pooled_p_language_greater": p_p,
    }

# EXPLORATORY, added after the specification was written and therefore carrying
# no verdict (spec section 9 item 5). The results record's first decision item
# leans on the language terminals sitting above their own rotated-lens control;
# this gives that comparison a p-value. Paired by state: for each of the 125
# language terminals, its lens share minus the mean of its own three rotated-lens
# control shares, tested by a sign-flip permutation test, one-sided, language
# higher.
h16["lang_above_rotation_control_exploratory"] = {}
rng7 = np.random.default_rng(PERM_SEED)
for l in LAYERS:
    ctrl_mean = np.mean(np.stack([arr(s_arm, "lang", l) for s_arm in ROT]), axis=0)
    d = arr("lens", "lang", l) - ctrl_mean
    obs, p_hi = perm_sign_flip_higher(d, rng7)
    h16["lang_above_rotation_control_exploratory"][str(l)] = {
        "median_paired_difference": obs, "p_language_higher": p_hi,
        "n_pairs": int(len(d)),
        "fraction_pairs_language_higher": float((d > 0).mean()),
        "exploratory_not_preregistered": True,
    }
verdicts["H16"] = h16
log(f"  H16 verdict: {h16['verdict']} (supporting {sup}, refuting {ref_p})")

# ----------------------------------------------------------------- H16a ------
log("H16a: the prolet attractor against the Divine cycle's two phases")
# What the labels mean, because it is not what the names suggest. build_states.py
# names each per-layer trace after the vector it injects, so "phaseA" is the
# forward pass whose INPUT is vector A. Its layer-11 entry is therefore that
# pass's output, which for this period-2 cycle is vector B (verified: cosine
# 1.000000 to the stored B), and "phaseB" holds vector A at layer 11. At layers 0
# to 10 neither trace holds a phase vector at all: those entries are intermediate
# residuals of one loop step, not either phase re-probed. Finding F16 in the
# lucier record measured a different quantity, the single vectors A and B scored
# against every layer's dictionary with no forward pass, so only layer 11 is a
# like-for-like comparison with it. The verdict rule below is symmetric in the
# two phases, so it is unaffected by which trace carries which name.
h16a = {"per_layer": {}, "label_note": (
    "phaseA is the forward pass injected FROM vector A: its layer-11 entry is "
    "vector B. phaseB is the pass injected FROM vector B: its layer-11 entry is "
    "vector A. Layers 0 to 10 are intermediate residuals of that pass, not the "
    "phase vectors re-probed. The SUPPORTED and REFUTED conditions are symmetric "
    "in phaseA and phaseB, so the verdict does not depend on the naming.")}
for l in LAYERS:
    nm = arr("lens", "named", l)
    pro = float(nm[NIDX["prolet1000"]])
    pa, pb, pm = (float(nm[NIDX["phaseA"]]), float(nm[NIDX["phaseB"]]),
                  float(nm[NIDX["pivotM"]]))
    ctrl_rot = [float(arr(a, "named", l)[NIDX["prolet1000"]]) for a in ROT]
    ctrl_gauss = [float(arr(a, "named", l)[NIDX["prolet1000"]]) for a in GAUSS]
    ctrl = ctrl_rot + ctrl_gauss
    # The pre-registered floor (spec section 7.3) pools all six control runs. The
    # two control types differ by a factor of about 25, so that pooled standard
    # deviation is dominated by the gap between the types rather than by run-to-run
    # variation, and it marks every gap as immaterial. The per-type spreads are
    # recorded beside it so the results record's yardstick has a provenance.
    spread = float(np.std(ctrl))
    spread_rot = float(np.std(ctrl_rot))
    spread_gauss = float(np.std(ctrl_gauss))
    h16a["per_layer"][str(l)] = {
        "prolet": pro, "phaseA": pa, "phaseB": pb, "pivotM": pm,
        "gap_prolet_minus_phaseA": pro - pa, "gap_prolet_minus_phaseB": pro - pb,
        "prolet_control_spread_sd": spread,
        "prolet_control_spread_sd_rotation": spread_rot,
        "prolet_control_spread_sd_gaussian": spread_gauss,
        "gapA_inside_control_spread": abs(pro - pa) < spread,
        "gapB_inside_control_spread": abs(pro - pb) < spread,
        "gapA_inside_rotation_spread": abs(pro - pa) < spread_rot,
        "gapB_inside_rotation_spread": abs(pro - pb) < spread_rot,
        "gapA_over_rotation_spread": abs(pro - pa) / spread_rot,
        "gapB_over_rotation_spread": abs(pro - pb) / spread_rot,
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
rng4 = np.random.default_rng(PERM_SEED)
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
rng5 = np.random.default_rng(PERM_SEED)
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
    "named_state_label_note": ("Every named state is a per-layer trace named after "
                               "the vector injected at its input. For the Divine "
                               "period-2 cycle that matters: phaseA's layer-11 entry "
                               "is vector B and phaseB's layer-11 entry is vector A, "
                               "and layers 0 to 10 are intermediate residuals of one "
                               "loop step rather than either phase re-probed."),
}
# A diagnostic run on an incomplete shares file writes nothing that could be read
# as final: every file goes to its own .partial. name, both JSON files carry a
# stamp, and each hypothesis's verdict string says plainly that it is not one.
if INCOMPLETE:
    stamp = {"partial_diagnostic_scoring": True,
             "why": ("The shares file this was scored from does not cover layers 0 "
                     "to 11 for every scoring arm, or marks itself partial. Nothing "
                     "here is a verdict on the pre-registered rules."),
             "input_completeness": COMPLETENESS}
    verdicts["PARTIAL_DIAGNOSTIC_SCORING"] = stamp
    table["_partial_diagnostic_scoring"] = stamp
    for _h in ("H6", "H16", "H16a", "H16b"):
        verdicts[_h]["verdict"] = (
            f"NO VERDICT, PARTIAL INPUT ({verdicts[_h]['verdict']} computed for "
            "diagnosis only)")
with open(outpath("per_layer_tables.json"), "w") as fh:
    json.dump(table, fh, indent=1)
with open(outpath("verdicts.json"), "w") as fh:
    json.dump(verdicts, fh, indent=1)

rows = ([] if not INCOMPLETE else
        ["# PARTIAL DIAGNOSTIC SCORING: the shares file did not cover layers 0 to "
         "11 for every scoring arm. Not a verdict table."])
rows.append("arm,family,layer,n,median_share,q25,q75,mean_share,median_n_atoms")
# Keys beginning with an underscore are stamps, not arms, so they carry no rows.
for arm in sorted(k for k in table if not k.startswith("_")):
    for fam in sorted(table[arm]):
        for l in LAYERS:
            e = table[arm][fam][str(l)]
            rows.append(f"{arm},{fam},{l},{e['n']},{e['median']:.6f},{e['q25']:.6f},"
                        f"{e['q75']:.6f},{e.get('mean', float('nan')):.6f},"
                        f"{e.get('median_n_atoms', float('nan')):.1f}")
with open(outpath("per_layer_shares.csv"), "w") as fh:
    fh.write("\n".join(rows) + "\n")

# --------------------------------------------------------------- figures ----
log("drawing the figures")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
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
ax.plot(LAYERS, [table["control_rotation_pooled"]["lang"][str(l)]["median"] for l in LAYERS],
        ":", color="#1f77b4", label="language, rotated-lens control")
ax.plot(LAYERS, [table["control_gaussian_pooled"]["lang"][str(l)]["median"] for l in LAYERS],
        ":", color="#7f7f7f", label="language, random-dictionary control")
ax.axvspan(4.6, 10.4, color="gold", alpha=0.12)
ax.set_yscale("log")
ax.set_xlabel("layer (output of block l)")
ax.set_ylabel("J-space share (fraction of squared length, log scale)")
ax.set_title("J-space share by layer, GPT-2 Small\n(gold band = workspace band, layers 5 to 10)",
             fontsize=9)
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
ax.set_title("Named states: prolet against the Divine cycle", fontsize=9)
ax.legend(fontsize=7)
ax.grid(alpha=0.25)

ax = axes[2]
for fam, lab, c in [("lang", "language terminals", "#1f77b4"),
                    ("noise17", "run-17 noise terminals", "#d62728"),
                    ("clean_last", "ordinary prompt residuals", "#2ca02c"),
                    ("nullold", "original noise arm", "#9467bd")]:
    d = [table["lens"][fam][str(l)]["median"]
         - table["control_rotation_pooled"][fam][str(l)]["median"] for l in LAYERS]
    ax.plot(LAYERS, d, "-o", ms=3, color=c, label=lab)
ax.axhline(0.0, color="k", lw=1)
ax.axvspan(4.6, 10.4, color="gold", alpha=0.12)
ax.set_xlabel("layer (output of block l)")
ax.set_ylabel("median share minus rotated-lens control")
ax.set_title("Against the rotated-lens chance level\n(above 0 = more lens-expressible than chance)",
             fontsize=9)
ax.legend(fontsize=7)
ax.grid(alpha=0.25)
if INCOMPLETE:
    fig.suptitle("PARTIAL DIAGNOSTIC SCORING, not a verdict figure: the shares "
                 "file did not cover layers 0 to 11 for every scoring arm",
                 fontsize=9, color="#b22222")
fig.tight_layout()
fig.savefig(outpath("exp011_share_curves.png"), dpi=150)
log("wrote " + ", ".join(os.path.basename(outpath(n)) for n in (
    "per_layer_tables.json", "verdicts.json", "per_layer_shares.csv",
    "exp011_share_curves.png")))
