"""EXP_018 hypothesis H19b: how much of a state sits inside the J-space.

The J-space of a language model, as defined by the "verbalizable workspace"
paper (section 2.3 and appendix A.8) and read here through the pre-fitted
Jacobian lens published at `neuronpedia/jacobian-lens`, is the set of points a
layer's internal state can reach as a non-negative combination of at most 25
lens directions. A lens direction for vocabulary token v at layer l is row v of
the matrix `W_U J_l`: the direction in the layer-l state along which a small
push most raises the model's disposition to say token v, later or now,
averaged over the 466 WikiText-103 prompts of 128 word pieces each on which
the published fit converged (it was budgeted for 1,000 prompts and stopped at
466 on its own criterion; the committed analysis records `lens_n_prompts`).

The J-space share of a state is the squared length of the closest point in
that set divided by the squared length of the state itself. It runs from 0
(nothing of the state is expressible that way) to 1 (the state lies wholly
inside). H19b asks whether the loop's settled states sit further outside the
J-space than the model's ordinary, non-iterated states at the same layer.

Nothing here needs the model to run. It reads the unembedding matrix and the
final normalisation gain out of the downloaded weight files, the fitted
Jacobians out of the lens file, and the per-layer states written by
`run_exp018.py --stage states`.

The weight files are read from one exact revision of the model, meaning one
named version of its files on the Hugging Face hub. A `--revision` given on the
command line is checked against every revision the states stage and the loop
recorded and stops the run if it contradicts one, because the unembedding
matrix and the normalisation gain read here have to come from the same weights
that produced the states being scored. When no earlier stage recorded a
revision, `--revision` is required and what it names is written into the output
as assumed rather than recorded. The revision is never chosen by sorting the
cache directory, which orders revisions by their identifiers and not by which
one a run used, and it is no longer taken from the cache pointer either, which
can name weights the run never used.

The lens file is downloaded rather than committed, so nothing in the repository
vouches for the copy on this machine. Its SHA-256 fingerprint, the standard
64-character summary of a file's exact contents, is checked against the one the
specification fixes before any tensor is read, and is written into the output
beside every number it produced.

The per-layer states arrive as two files that only mean anything together, the
archive of tensors and the metadata naming the weights they came from, so both
carry the same one-use generation stamp written by the states stage and this
stage refuses a pair whose stamps do not agree. A pair made before the stamp
existed carries none, which cannot be checked and is said out loud rather than
passed over.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from qwen_port import (  # noqa: E402
    GENERATION_KEY, check_commit_revision, snapshot_dir,
)

torch.set_num_threads(1)

OUT = HERE / "output"
ART = (HERE / ".." / ".." / "artifacts").resolve()
LENS_PT = ART / "qwen3-1.7b_jacobian_lens.pt"

# The SHA-256 fingerprint the specification fixes for the lens file, in section
# 2 of `_STAGE2_JSPACE/EXP_018_SPEC.md` under "The instrument". A SHA-256 is the
# standard 64-character hexadecimal summary of a file's exact contents: two
# files with the same fingerprint are the same file. The lens lives in
# `_STAGE2_JSPACE/artifacts/`, which is not versioned, so this constant is the
# only thing in the repository that says which lens produced a number. If the
# lens is ever meant to change, the new fingerprint goes into a dated addendum
# to the specification and into this constant in the same commit.
LENS_SHA256 = "6fcc79011bd921ffd87612255e2e99950a124fa519470ee44ebaf161c39be9d6"

BAND_LAYERS = list(range(11, 26))
EARLY_LAYERS = [2, 5]
SCORED_LAYERS = sorted(EARLY_LAYERS + BAND_LAYERS)
K_ATOMS = 25
CANDIDATE_POOL = 4096


# --------------------------------------------------------------------------
# weights
# --------------------------------------------------------------------------

def load_unembed(revision: str) -> tuple[torch.Tensor, torch.Tensor]:
    """The unembedding matrix `W_U` as [vocab, width] and the final RMSNorm gain.

    Qwen3 ties its embeddings, so the unembedding is the token embedding matrix
    and the same tensor serves both roles. Read straight out of the weight
    files of the named revision, so this stage never loads the whole model and
    never reads a different version of the weights from the one the loop ran.
    """
    from safetensors import safe_open
    snap = snapshot_dir(revision)
    index = json.loads((snap / "model.safetensors.index.json").read_text())
    wmap = index["weight_map"]
    want = {"model.embed_tokens.weight": None, "model.norm.weight": None}
    for key in want:
        with safe_open(snap / wmap[key], framework="pt") as fh:
            want[key] = fh.get_tensor(key).float()
    return want["model.embed_tokens.weight"], want["model.norm.weight"]


def recorded_revisions(arm: str, meta: dict) -> list[tuple[str, str, bool]]:
    """Every weights revision an earlier stage recorded, each with its source
    and with whether that stage marked it an assumption rather than a record.

    A revision is the 40-character commit identifier naming one exact version of
    the model's files on the Hugging Face hub. Three places may carry one: the
    states stage records the revision it pinned its own load to as
    `model_revision` and the revision the loop had recorded as
    `loop_model_revision`, and the loop's own results file records
    `model_revision`. Any of the three may be absent, as all three are for the
    runs committed with this experiment, which were made before the runner wrote
    the field.

    A stage that was handed its revision by hand rather than reading it from the
    run before it says so in a companion field, and that mark travels with the
    revision here, so an assumption made two stages ago is not read as a
    measurement now.
    """
    found = []
    for key, where in (("model_revision", "the states stage metadata"),
                       ("loop_model_revision", "the loop metadata carried by "
                                               "the states stage")):
        if meta.get(key):
            found.append((meta[key], where,
                          bool(meta.get(f"{key}_assumed"))))
    res_path = OUT / f"results_{arm}.json"
    if res_path.exists():
        loop = json.loads(res_path.read_text())
        if loop.get("model_revision"):
            found.append((loop["model_revision"],
                          f"the loop metadata in results_{arm}.json",
                          bool(loop.get("model_revision_assumed"))))
    return found


def revision_for(arm: str, meta: dict,
                 explicit: str | None) -> tuple[str, str, bool]:
    """The revision of the weights to read, the reason for it, and whether that
    reason is an assumption rather than a record.

    This stage reads the unembedding matrix and the final normalisation gain out
    of the weight files and multiplies them into the lens directions that score
    states another stage produced. Those states came out of one exact version of
    the weights, so a `--revision` given here does not override what an earlier
    stage recorded: it is checked against every recorded revision and any
    disagreement stops the run, because a state made by one version and scored
    against another version's unembedding is two experiments inside one number.

    When nothing recorded a revision, as in the runs committed with this
    experiment, `--revision` is required and the third return value is True, so
    the output can say that the revision was confirmed by hand rather than
    measured at the time. Falling back to the local cache pointer `refs/main`,
    which this stage did before, is no longer allowed: that pointer can name
    weights the run never used and the output would then be labelled with them.
    """
    found = recorded_revisions(arm, meta)
    if explicit:
        clashes = [f"{where} records {rev}" for rev, where, _ in found
                   if rev != explicit]
        if clashes:
            raise SystemExit(
                f"--revision {explicit} contradicts what an earlier stage "
                f"recorded for arm {arm}: " + "; ".join(clashes) + ".\n"
                f"The states about to be scored came out of the recorded "
                f"version, so reading the unembedding matrix and the "
                f"normalisation gain from {explicit} would put two versions of "
                f"the weights inside one measurement. Drop the flag to use the "
                f"recorded revision.")
        if found:
            rev, where, was_assumed = found[0]
            return explicit, (f"the --revision option, which agrees with "
                              f"{where}"
                              + (", which is itself an assumption confirmed by "
                                 "hand rather than a record"
                                 if was_assumed else "")), was_assumed
        return explicit, ("the --revision option, assumed rather than recorded: "
                          "neither the states metadata nor the loop results "
                          "record a revision, so this is what the operator "
                          "confirmed by hand"), True
    if found:
        rev, where, was_assumed = found[0]
        return rev, (where + (", which records it as an assumption confirmed by "
                              "hand rather than a measurement"
                              if was_assumed else "")), was_assumed
    raise SystemExit(
        f"neither the states metadata for arm {arm} nor results_{arm}.json "
        f"records a weights revision, so this stage cannot tell which version "
        f"of the model's files produced the states it is about to score. Pass "
        f"--revision <40-character identifier> naming the weights that run "
        f"used; the output records it as assumed rather than measured. "
        f"Following the local cache pointer refs/main instead, which this stage "
        f"did before, can read a version the run never used and then label "
        f"every share with it.")


def states_generation(states, meta: dict, arm: str) -> str | None:
    """The generation stamp both halves of the state set agree on, or None.

    A generation stamp is a one-use identifier the states stage writes into both
    the archive of tensors and its metadata, so that a reader can tell they were
    published by the same run. They are two files and two renames cannot be one
    step, so an interruption between them can leave a new archive beside the
    previous run's metadata; scoring one against the other would read the new
    states through the old revision's unembedding matrix and label the answer
    with the old revision. Disagreeing stamps stop the run. A pair made before
    the stamp existed carries none in either half, which cannot be checked and
    is reported rather than passed over silently.
    """
    in_npz = None
    if GENERATION_KEY in getattr(states, "files", []):
        value = states[GENERATION_KEY]
        in_npz = value.item() if hasattr(value, "item") else str(value)
    in_meta = meta.get("generation")
    if in_npz and in_meta and in_npz == in_meta:
        return str(in_npz)
    if in_npz or in_meta:
        raise SystemExit(
            f"refusing to score arm {arm}: the two halves of its per-layer "
            f"state set were not published together. The archive "
            f"layer_states_{arm}.npz carries generation {in_npz!r} and the "
            f"metadata layer_states_{arm}_meta.json carries {in_meta!r}, and "
            f"only a pair with the same stamp is known to have come out of one "
            f"run. Scoring them against each other would read one run's states "
            f"through another run's weights and label the answer with the "
            f"wrong revision. Rebuild both with "
            f"`python3 run_exp018.py --stage states --arm {arm} --revision "
            f"<40-character identifier>`.")
    print(f"note: neither half of the per-layer state set for arm {arm} carries "
          f"a generation stamp, so this stage cannot check that the archive and "
          f"its metadata came out of the same run. A pair written before the "
          f"stamp existed is in exactly this state. The output records the "
          f"absence rather than an agreement. Rebuilding both with `--stage "
          f"states` writes the stamp.", flush=True)
    return None


def verify_lens_file() -> str:
    """Check the lens file against the fingerprint the specification fixes, and
    return that fingerprint. Refuses before a single tensor is read.

    The lens file is downloaded into `_STAGE2_JSPACE/artifacts/`, which is not
    versioned, so the repository holds no copy to compare against and a wrong or
    truncated download would change every H19b number without changing anything
    a reader could see. Reading 226 megabytes to compute the fingerprint takes
    about 20 seconds, which is under 3 percent of this stage's 16-minute run on
    the main arm.
    """
    if not LENS_PT.exists():
        raise SystemExit(
            f"the Jacobian lens file is not on this machine: expected it at "
            f"{LENS_PT}. It is downloaded rather than committed, because "
            f"_STAGE2_JSPACE/artifacts/ is not versioned. Section 2 of "
            f"_STAGE2_JSPACE/EXP_018_SPEC.md names the source, "
            f"neuronpedia/jacobian-lens at path "
            f"qwen3-1.7b/jlens/Salesforce-wikitext, and the fingerprint to "
            f"expect.")
    h = hashlib.sha256()
    with LENS_PT.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    digest = h.hexdigest()
    if digest != LENS_SHA256:
        raise SystemExit(
            f"refusing to read {LENS_PT}: its SHA-256 fingerprint, the "
            f"64-character summary of its exact contents, is {digest}, and the "
            f"specification fixes {LENS_SHA256} for this file. A different lens "
            f"gives different J-space shares, so this is not the instrument "
            f"EXP_018 registered. Download the file again from "
            f"neuronpedia/jacobian-lens, path "
            f"qwen3-1.7b/jlens/Salesforce-wikitext; if the lens is meant to "
            f"change, record the new fingerprint in a dated addendum to the "
            f"specification and in LENS_SHA256 here, in the same commit.")
    return digest


def load_lens() -> tuple[dict[int, torch.Tensor], int]:
    """The fitted Jacobians by layer, and the number of prompts they were fitted
    on. Call `verify_lens_file` first: this reads the file without checking it."""
    ck = torch.load(LENS_PT, map_location="cpu", weights_only=True)
    return {int(l): J.float() for l, J in ck["J"].items()}, ck["n_prompts"]


# --------------------------------------------------------------------------
# the share
# --------------------------------------------------------------------------

def _nnls(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    from scipy.optimize import nnls
    return nnls(A, b)[0]


def jspace_share(atoms: torch.Tensor, h: torch.Tensor, k: int = K_ATOMS,
                 pool: int | None = CANDIDATE_POOL,
                 corr0: torch.Tensor | None = None) -> tuple[float, int]:
    """Share of `h` captured by at most `k` atoms with non-negative weights.

    Gradient pursuit: repeatedly add the atom whose correlation with what is
    left of the state is largest and positive, then re-fit all chosen atoms by
    non-negative least squares, and stop at `k` atoms or when nothing
    correlates positively any more. Atoms are unit length, which leaves the
    answer unchanged (the set of non-negative combinations does not care how
    long each atom is) and makes "largest correlation" a comparison of
    directions.

    `pool` restricts the search, after one full pass over the vocabulary, to
    the that many best-correlating atoms. That is an approximation; the
    `--exact-layers` option measures its size.
    """
    hn2 = float(h @ h)
    if hn2 <= 0:
        return 0.0, 0
    if pool is not None and pool < atoms.shape[0]:
        c0 = atoms @ h if corr0 is None else corr0
        idx = torch.topk(c0, pool).indices
        A = atoms[idx]
    else:
        A = atoms
    residual = h.clone()
    chosen: list[int] = []
    approx = torch.zeros_like(h)
    for _ in range(k):
        corr = A @ residual
        if chosen:
            corr[torch.tensor(chosen)] = -float("inf")
        j = int(torch.argmax(corr))
        if float(corr[j]) <= 0:
            break
        chosen.append(j)
        sub = A[chosen].T.contiguous().numpy().astype(np.float64)
        coef = _nnls(sub, h.numpy().astype(np.float64))
        approx = torch.from_numpy(sub @ coef).float()
        residual = h - approx
    return float(approx @ approx) / hn2, len(chosen)


def random_rotation(d: int, seed: int) -> torch.Tensor:
    """A uniformly random orthogonal `d` by `d` matrix (QR of a Gaussian)."""
    g = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(d, d, generator=g))
    return q * torch.sign(torch.diagonal(r)).unsqueeze(0)


# --------------------------------------------------------------------------
# permutation test
# --------------------------------------------------------------------------

def paired_permutation(settled: dict[str, list[float]], clean: dict[str, list[float]],
                       n_draws: int = 10000, seed: int = 42) -> tuple[float, float, float]:
    """One-sided paired permutation test on the difference of medians.

    Each prompt contributes one block of position-level shares to each arm.
    A draw flips a coin per prompt and, on heads, swaps that prompt's two
    blocks. The p-value is the share of draws whose median difference is at
    least as far below zero as the observed one.
    """
    ids = sorted(settled)
    s_all = np.concatenate([settled[i] for i in ids])
    c_all = np.concatenate([clean[i] for i in ids])
    obs = float(np.median(s_all) - np.median(c_all))
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_draws):
        flip = rng.random(len(ids)) < 0.5
        s_draw, c_draw = [], []
        for f, i in zip(flip, ids):
            a, b = (clean[i], settled[i]) if f else (settled[i], clean[i])
            s_draw.append(a); c_draw.append(b)
        d = float(np.median(np.concatenate(s_draw)) - np.median(np.concatenate(c_draw)))
        hits += d <= obs
    return obs, float(np.median(s_all)), (hits + 1) / (n_draws + 1)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="bare")
    ap.add_argument("--states-dir", default=str(HERE / "_states"))
    ap.add_argument("--exact-layers", default="11,18",
                    help="layers additionally scored without the candidate pool")
    ap.add_argument("--rotation-seeds", default="2026,4242")
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--revision", default=None,
                    help="the exact Hugging Face revision of the weights to "
                         "read, given as its 40-character identifier. It must "
                         "agree with every revision the states stage and the "
                         "loop recorded, and is required when neither recorded "
                         "one, as is the case for the committed runs; the "
                         "output then marks it assumed rather than recorded")
    args = ap.parse_args()
    # A revision names one exact version of the weights, so it has to be a
    # commit identifier: a branch or tag name could name different weights on a
    # later day while every string comparison below still reported agreement.
    if args.revision:
        check_commit_revision(args.revision, "--revision")

    t_start = time.time()
    states = np.load(Path(args.states_dir) / f"layer_states_{args.arm}.npz")
    meta = json.loads((Path(args.states_dir)
                       / f"layer_states_{args.arm}_meta.json").read_text())
    prompt_ids = [p["id"] for p in meta["prompts"]]
    generation = states_generation(states, meta, args.arm)
    # Both of these refuse before anything large is read: the revision decision
    # costs nothing, and the fingerprint check reads the lens file once.
    revision, rev_source, rev_assumed = revision_for(args.arm, meta, args.revision)
    print(f"weights revision {revision}, from {rev_source}"
          f"{' (assumed, not recorded)' if rev_assumed else ''}", flush=True)
    lens_sha256 = verify_lens_file()
    print(f"lens {LENS_PT.name}: SHA-256 {lens_sha256} matches the one the "
          f"specification fixes", flush=True)
    W_U, gamma = load_unembed(revision)
    lens, lens_n_prompts = load_lens()
    d_model = W_U.shape[1]
    print(f"W_U {tuple(W_U.shape)}  lens layers {min(lens)}..{max(lens)} "
          f"fitted on {lens_n_prompts} prompts", flush=True)

    rot_seeds = [int(s) for s in args.rotation_seeds.split(",")]
    rotations = {s: random_rotation(d_model, s) for s in rot_seeds}
    exact_layers = {int(s) for s in args.exact_layers.split(",") if s}

    results = {
        "arm": args.arm, "model_revision": revision,
        "model_revision_source": rev_source,
        "model_revision_assumed": rev_assumed,
        "states_generation": generation,
        "states_generation_checked": generation is not None,
        "lens_file": LENS_PT.name, "lens_sha256": lens_sha256,
        "k_atoms": K_ATOMS, "candidate_pool": CANDIDATE_POOL,
        "scored_layers": SCORED_LAYERS, "band_layers": BAND_LAYERS,
        "early_layers": EARLY_LAYERS, "rotation_seeds": rot_seeds,
        "lens_n_prompts": int(lens_n_prompts), "permutation_draws": args.draws,
        "n_prompts": len(prompt_ids), "layers": {},
    }

    for layer in SCORED_LAYERS:
        t0 = time.time()
        atoms = (W_U * gamma.unsqueeze(0)) @ lens[layer]        # [vocab, width]
        atoms /= atoms.norm(dim=1, keepdim=True).clamp_min(1e-8)  # in place: the
        # matrix is 151,936 by 2,048 and a second copy would cost 1.2 gigabytes
        row = {"conditions": {}}
        for cond in ("settled", "clean"):
            per_prompt = {s: {} for s in ["real"] + [f"rot{s}" for s in rot_seeds]}
            for dict_key in ["real"] + [f"rot{s}" for s in rot_seeds]:
                # Gather every state this dictionary has to score into one
                # matrix, so the 151,936 by 2,048 correlation happens as a
                # single matrix product instead of once per state. The matrix
                # is read from memory once instead of hundreds of times, which
                # was measured to be about seven times faster.
                flat, owner = [], []
                for pid in prompt_ids:
                    H = torch.from_numpy(states[f"{cond}|{pid}|{layer}"]).float()
                    if dict_key != "real":
                        # rotating the state is exactly rotating the dictionary
                        # the other way, and costs a thousand times less
                        H = H @ rotations[int(dict_key[3:])]
                    for h in H:
                        flat.append(h)
                        owner.append(pid)
                    per_prompt[dict_key][pid] = []
                Hs = torch.stack(flat)                       # [n_states, width]
                corrs = []
                for start in range(0, Hs.shape[0], 128):
                    corrs.append(atoms @ Hs[start:start + 128].T)
                corrs = torch.cat(corrs, dim=1)              # [vocab, n_states]
                for j, pid in enumerate(owner):
                    per_prompt[dict_key][pid].append(
                        jspace_share(atoms, Hs[j], corr0=corrs[:, j])[0])
                del corrs, Hs, flat
            row["conditions"][cond] = per_prompt
        # exact (no candidate pool) check on selected layers
        if layer in exact_layers:
            ex = {}
            for cond in ("settled", "clean"):
                vals, appr = [], []
                for pid in prompt_ids[:5]:
                    H = torch.from_numpy(states[f"{cond}|{pid}|{layer}"]).float()
                    for h in H:
                        vals.append(jspace_share(atoms, h, pool=None)[0])
                        appr.append(jspace_share(atoms, h)[0])
                ex[cond] = {"exact": vals, "pooled": appr,
                            "max_abs_diff": float(np.max(np.abs(
                                np.array(vals) - np.array(appr))))}
            row["exact_check"] = ex
        row["seconds"] = round(time.time() - t0, 1)
        results["layers"][str(layer)] = row
        med_s = float(np.median(np.concatenate(
            [row["conditions"]["settled"]["real"][p] for p in prompt_ids])))
        med_c = float(np.median(np.concatenate(
            [row["conditions"]["clean"]["real"][p] for p in prompt_ids])))
        print(f"  layer {layer:2d}: settled median {med_s:.4f}  "
              f"clean median {med_c:.4f}  ({row['seconds']:.0f}s)", flush=True)
        del atoms

    # verdict arithmetic
    verdict = {"per_layer": {}, "band_layers_below": 0, "band_layers_below_p05": 0}
    for layer in SCORED_LAYERS:
        row = results["layers"][str(layer)]
        s = {p: row["conditions"]["settled"]["real"][p] for p in prompt_ids}
        c = {p: row["conditions"]["clean"]["real"][p] for p in prompt_ids}
        diff, med_s, p = paired_permutation(s, c, args.draws)
        med_c = med_s - diff
        entry = {"median_settled": round(med_s, 5), "median_clean": round(med_c, 5),
                 "median_difference": round(diff, 5), "p_one_sided": round(p, 5),
                 "settled_below_clean": bool(diff < 0), "p_below_05": bool(p < 0.05)}
        for s_seed in rot_seeds:
            entry[f"median_settled_rot{s_seed}"] = round(float(np.median(
                np.concatenate([row["conditions"]["settled"][f"rot{s_seed}"][p]
                                for p in prompt_ids]))), 5)
            entry[f"median_clean_rot{s_seed}"] = round(float(np.median(
                np.concatenate([row["conditions"]["clean"][f"rot{s_seed}"][p]
                                for p in prompt_ids]))), 5)
        verdict["per_layer"][str(layer)] = entry
        if layer in BAND_LAYERS:
            verdict["band_layers_below"] += entry["settled_below_clean"]
            verdict["band_layers_below_p05"] += (
                entry["settled_below_clean"] and entry["p_below_05"])
    verdict["n_band_layers"] = len(BAND_LAYERS)
    verdict["majority_needed"] = len(BAND_LAYERS) // 2 + 1
    verdict["H19b_supported"] = bool(
        verdict["band_layers_below_p05"] >= verdict["majority_needed"])
    results["verdict"] = verdict
    results["wall_seconds"] = round(time.time() - t_start, 1)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"jspace_shares_{args.arm}.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(verdict, indent=2)[:2000], flush=True)
    print(f"wrote {OUT / f'jspace_shares_{args.arm}.json'} in "
          f"{(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
