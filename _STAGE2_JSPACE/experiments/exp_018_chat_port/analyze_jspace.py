"""EXP_018 hypothesis H19b: how much of a state sits inside the J-space.

The J-space of a language model, as defined by the "verbalizable workspace"
paper (section 2.3 and appendix A.8) and read here through the pre-fitted
Jacobian lens published at `neuronpedia/jacobian-lens`, is the set of points a
layer's internal state can reach as a non-negative combination of at most 25
lens directions. A lens direction for vocabulary token v at layer l is row v of
the matrix `W_U J_l`: the direction in the layer-l state along which a small
push most raises the model's disposition to say token v, later or now,
averaged over a thousand ordinary web-text contexts.

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
named version of its files on the Hugging Face hub. The revision comes from
`--revision` if given, otherwise from the metadata the states stage or the loop
recorded, otherwise from the cache pointer an unpinned load follows. It is
never chosen by sorting the cache directory, which orders revisions by their
identifiers and not by which one a run used.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from qwen_port import resolve_revision, snapshot_dir  # noqa: E402

torch.set_num_threads(1)

OUT = HERE / "output"
ART = (HERE / ".." / ".." / "artifacts").resolve()
LENS_PT = ART / "qwen3-1.7b_jacobian_lens.pt"

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


def revision_for(arm: str, meta: dict, explicit: str | None) -> tuple[str, str]:
    """The revision of the weights to read, and the one-line reason for it.

    A revision is the 40-character commit identifier naming one exact version
    of the model's files on the Hugging Face hub. Order of authority: the
    `--revision` option, then the revision the states stage recorded, then the
    revision the loop recorded in its own results file, then the cache pointer
    `refs/main`, which is the one an unpinned load follows. `resolve_revision`
    stops with a message naming the directory it searched if that last step
    cannot decide.
    """
    if explicit:
        return explicit, "the --revision option"
    for key, where in (("model_revision", "the states stage metadata"),
                       ("loop_model_revision", "the loop metadata carried by "
                                               "the states stage")):
        if meta.get(key):
            return meta[key], where
    res_path = OUT / f"results_{arm}.json"
    if res_path.exists():
        rev = json.loads(res_path.read_text()).get("model_revision")
        if rev:
            return rev, f"the loop metadata in results_{arm}.json"
    return (resolve_revision(),
            "the local cache pointer refs/main, because neither the states "
            "metadata nor the loop results record a revision")


def load_lens() -> dict[int, torch.Tensor]:
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
                         "read; defaults to the one the run recorded")
    args = ap.parse_args()

    t_start = time.time()
    states = np.load(Path(args.states_dir) / f"layer_states_{args.arm}.npz")
    meta = json.loads((Path(args.states_dir)
                       / f"layer_states_{args.arm}_meta.json").read_text())
    prompt_ids = [p["id"] for p in meta["prompts"]]
    revision, rev_source = revision_for(args.arm, meta, args.revision)
    print(f"weights revision {revision}, from {rev_source}", flush=True)
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
