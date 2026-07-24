"""
Anisotropy-corrected permutation test for EXP_010c terminal-token relatedness.

Spec: ../../PERM_TEST_EXP010c_SPEC.md (pre-registered before this script).
Pattern: Stage 1's 02b_permutation_test.py (Lucier repo), extended with
matched nulls and multiple token sets.

Run:  python permutation_test.py --model-path <dir-with-pytorch_model.bin>
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "output"

N_PERM = 10_000
SEED = 2026
FREQ_BANDS = 10
VOCAB_SIZE = 50257
LENGTH_TOLERANCE = 1

TOKEN_SETS = {
    "S1_A4_direct": {
        "description": "A4 (10->21) direct decode",
        "token_ids": [1566, 8097, 1201],
        "tokens": [" until", " forever", " since"],
    },
    "S2_O8_direct": {
        "description": "O8 (8->21) direct decode",
        "token_ids": [11640, 19487],
        "tokens": [" simultaneously", " halfway"],
    },
    "S3_A5_direct": {
        "description": "A5 (8->15) direct decode — DEGENERATE (n=1)",
        "token_ids": [30993],
        "tokens": [" rant"],
    },
    "S4_pooled_direct": {
        "description": "Pooled word-arms direct (A4 + O8 + A5)",
        "token_ids": [1566, 8097, 1201, 11640, 19487, 30993],
        "tokens": [" until", " forever", " since", " simultaneously", " halfway", " rant"],
    },
    "S5_A1_contrast": {
        "description": "A1 (0->11) direct decode — CONTRAST (punctuation funnel)",
        "token_ids": [11, 278],
        "tokens": [",", "ing"],
    },
    "S6_A4_tail": {
        "description": "A4 (10->21) via-tail decode",
        "token_ids": [1566, 8097, 1201],
        "tokens": [" until", " forever", " since"],
    },
    "S7_O8_tail": {
        "description": "O8 (8->21) via-tail decode",
        "token_ids": [11640, 655, 6],
        "tokens": [" simultaneously", " just", "'"],
    },
    "S8_A5_tail": {
        "description": "A5 (8->15) via-tail decode",
        "token_ids": [13079, 6],
        "tokens": [" endless", "'"],
    },
    "S9_pooled_tail": {
        "description": "Pooled word-arms via-tail (S6 + S7 + S8)",
        "token_ids": [1566, 8097, 1201, 11640, 655, 6, 13079],
        "tokens": [" until", " forever", " since", " simultaneously", " just", "'", " endless"],
    },
}


def load_embeddings(model_path):
    sd = torch.load(
        pathlib.Path(model_path) / "pytorch_model.bin",
        map_location="cpu",
        weights_only=True,
    )
    W_E = sd["wte.weight"].numpy()
    tied = "lm_head.weight" not in sd
    W_U = W_E if tied else sd["lm_head.weight"].numpy()
    return W_E, W_U, tied


def load_vocab(model_path):
    with open(pathlib.Path(model_path) / "vocab.json") as f:
        vocab = json.load(f)
    id_to_str = {v: k for k, v in vocab.items()}
    return id_to_str


def build_token_properties(id_to_str):
    props = {}
    for tid in range(VOCAB_SIZE):
        s = id_to_str.get(tid, "")
        has_space = s.startswith("Ġ")
        display = s.replace("Ġ", " ")
        strlen = len(s)
        freq_band = min(tid * FREQ_BANDS // VOCAB_SIZE, FREQ_BANDS - 1)
        props[tid] = {
            "has_space": has_space,
            "strlen": strlen,
            "freq_band": freq_band,
            "display": display,
        }
    return props


def build_matching_pools(token_ids, props):
    pools = {}
    warnings = []
    for tid in token_ids:
        p = props[tid]
        pool = []
        for cid in range(VOCAB_SIZE):
            if cid == tid:
                continue
            cp = props[cid]
            if cp["has_space"] != p["has_space"]:
                continue
            if abs(cp["strlen"] - p["strlen"]) > LENGTH_TOLERANCE:
                continue
            if cp["freq_band"] != p["freq_band"]:
                continue
            pool.append(cid)
        if len(pool) < 50:
            pool_relaxed = []
            for cid in range(VOCAB_SIZE):
                if cid == tid:
                    continue
                cp = props[cid]
                if cp["has_space"] != p["has_space"]:
                    continue
                if abs(cp["strlen"] - p["strlen"]) > LENGTH_TOLERANCE + 1:
                    continue
                if cp["freq_band"] != p["freq_band"]:
                    continue
                pool_relaxed.append(cid)
            warnings.append(
                f"token {tid} ({props[tid]['display']!r}): pool {len(pool)} -> "
                f"relaxed to ±{LENGTH_TOLERANCE+1} chars -> {len(pool_relaxed)}"
            )
            pool = pool_relaxed
        pools[tid] = np.array(pool, dtype=np.int64)
    return pools, warnings


def mean_pairwise_cosine(W_norm, token_ids):
    if len(token_ids) < 2:
        return float("nan")
    vecs = W_norm[token_ids]
    sim = vecs @ vecs.T
    n = len(token_ids)
    mask = ~np.eye(n, dtype=bool)
    return float(sim[mask].mean())


def offdiag_stats(sim_matrix):
    mask = ~np.eye(sim_matrix.shape[0], dtype=bool)
    vals = sim_matrix[mask]
    return {
        "mean": float(vals.mean()),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "std": float(vals.std()),
    }


def run_permutation_test(W_norm, token_ids, pools, rng, n_perm):
    observed = mean_pairwise_cosine(W_norm, token_ids)
    if np.isnan(observed):
        return None

    null_stats = np.empty(n_perm)
    for k in range(n_perm):
        replacement = []
        used = set()
        for tid in token_ids:
            pool = pools[tid]
            pool_available = pool[~np.isin(pool, list(used))]
            if len(pool_available) == 0:
                pool_available = pool
            chosen = rng.choice(pool_available)
            replacement.append(int(chosen))
            used.add(int(chosen))
        null_stats[k] = mean_pairwise_cosine(W_norm, replacement)

    null_mean = float(null_stats.mean())
    null_sd = float(null_stats.std())
    effect_size = (observed - null_mean) / null_sd if null_sd > 0 else float("inf")
    p_value = (np.sum(null_stats >= observed) + 1) / (n_perm + 1)

    return {
        "observed": observed,
        "null_mean": null_mean,
        "null_sd": null_sd,
        "effect_size": effect_size,
        "p_value": float(p_value),
        "null_min": float(null_stats.min()),
        "null_max": float(null_stats.max()),
        "null_median": float(np.median(null_stats)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()

    print("Loading embeddings...")
    W_E, W_U, tied = load_embeddings(args.model_path)
    print(f"  W_E: {W_E.shape}, W_U: {'tied to W_E' if tied else W_U.shape}")

    W_E_norm = W_E / np.linalg.norm(W_E, axis=1, keepdims=True)
    W_U_norm = W_U / np.linalg.norm(W_U, axis=1, keepdims=True)

    print("Loading vocab and building token properties...")
    id_to_str = load_vocab(args.model_path)
    props = build_token_properties(id_to_str)

    for sid, sdef in TOKEN_SETS.items():
        for tid, tok_str in zip(sdef["token_ids"], sdef["tokens"]):
            display = props[tid]["display"]
            expected = tok_str.replace(" ", "Ġ")
            if " " in tok_str:
                expected = tok_str
                display_check = " " + display.lstrip() if not display.startswith(" ") else display
            else:
                display_check = display
            assert display_check.strip() == tok_str.strip() or display == tok_str, (
                f"Set {sid}: token {tid} expected {tok_str!r} but vocab says {display!r}"
            )

    rng = np.random.default_rng(SEED)
    n_bonferroni = sum(1 for s in TOKEN_SETS.values() if len(s["token_ids"]) >= 2)
    alpha = 0.05 / n_bonferroni

    print(f"\nGlobal anisotropy context (200k random pairs)...")
    pairs = rng.choice(VOCAB_SIZE, size=(200_000, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    global_we = float((W_E_norm[pairs[:, 0]] * W_E_norm[pairs[:, 1]]).sum(1).mean())
    global_wu = float((W_U_norm[pairs[:, 0]] * W_U_norm[pairs[:, 1]]).sum(1).mean())
    print(f"  W_E global mean pairwise cosine: {global_we:.4f}")
    print(f"  W_U global mean pairwise cosine: {global_wu:.4f}")
    print(f"  Weight tying: {'YES (W_E = W_U, results identical)' if tied else 'NO'}")

    results = {
        "design": {
            "n_permutations": N_PERM,
            "seed": SEED,
            "matching": "leading_space + strlen_pm1 + freq_decile",
            "bonferroni_n": n_bonferroni,
            "bonferroni_alpha": alpha,
            "weight_tying": tied,
        },
        "context": {
            "global_mean_pairwise_cosine_WE": global_we,
            "global_mean_pairwise_cosine_WU": global_wu,
        },
        "sets": {},
    }

    rng_we = np.random.default_rng(SEED)
    rng_wu = np.random.default_rng(SEED)

    for sid, sdef in TOKEN_SETS.items():
        tids = sdef["token_ids"]
        n_types = len(tids)
        n_pairs = n_types * (n_types - 1) // 2
        print(f"\n--- {sid}: {sdef['description']} ---")
        print(f"  Tokens: {sdef['tokens']} (IDs: {tids})")
        print(f"  n_types={n_types}, n_pairs={n_pairs}")

        set_result = {
            "description": sdef["description"],
            "token_ids": tids,
            "tokens": sdef["tokens"],
            "n_types": n_types,
            "n_pairs": n_pairs,
        }

        if n_types < 2:
            print("  SKIPPED: degenerate set (n < 2)")
            set_result["status"] = "degenerate"
            results["sets"][sid] = set_result
            continue

        pools, warnings = build_matching_pools(tids, props)
        pool_sizes = {tid: len(pools[tid]) for tid in tids}
        print(f"  Matching pools: {pool_sizes}")
        for w in warnings:
            print(f"  WARNING: {w}")
        set_result["pool_sizes"] = {str(tid): sz for tid, sz in pool_sizes.items()}
        set_result["pool_warnings"] = warnings

        # Pairwise cosine matrix (for the record)
        vecs_we = W_E_norm[tids]
        sim_we = vecs_we @ vecs_we.T
        set_result["pairwise_cosine_WE"] = {
            f"{tids[i]}-{tids[j]}": float(sim_we[i, j])
            for i in range(n_types)
            for j in range(i + 1, n_types)
        }

        print(f"  Running W_E permutation test (N={N_PERM})...")
        res_we = run_permutation_test(W_E_norm, tids, pools, rng_we, N_PERM)
        set_result["WE"] = res_we
        sig_we = res_we["p_value"] < alpha
        print(
            f"  W_E: observed={res_we['observed']:.4f}, "
            f"null={res_we['null_mean']:.4f}±{res_we['null_sd']:.4f}, "
            f"effect={res_we['effect_size']:.2f}σ, "
            f"p={res_we['p_value']:.6f} {'***' if sig_we else ''}"
        )

        if not tied:
            vecs_wu = W_U_norm[tids]
            sim_wu = vecs_wu @ vecs_wu.T
            set_result["pairwise_cosine_WU"] = {
                f"{tids[i]}-{tids[j]}": float(sim_wu[i, j])
                for i in range(n_types)
                for j in range(i + 1, n_types)
            }
            print(f"  Running W_U permutation test (N={N_PERM})...")
            res_wu = run_permutation_test(W_U_norm, tids, pools, rng_wu, N_PERM)
            set_result["WU"] = res_wu
            sig_wu = res_wu["p_value"] < alpha
            print(
                f"  W_U: observed={res_wu['observed']:.4f}, "
                f"null={res_wu['null_mean']:.4f}±{res_wu['null_sd']:.4f}, "
                f"effect={res_wu['effect_size']:.2f}σ, "
                f"p={res_wu['p_value']:.6f} {'***' if sig_wu else ''}"
            )
        else:
            set_result["WU"] = "identical to WE (weights tied)"

        set_result["significant_bonferroni"] = sig_we
        set_result["status"] = "tested"
        results["sets"][sid] = set_result

    out_path = OUT / "permutation_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[SAVED] {out_path}")

    print("\n=== SUMMARY ===")
    print(f"Bonferroni threshold: α = {alpha:.6f} (0.05 / {n_bonferroni})")
    print(f"Weight tying: {'YES — W_E and W_U identical' if tied else 'NO'}")
    for sid, sr in results["sets"].items():
        if sr["status"] == "degenerate":
            print(f"  {sid}: DEGENERATE (n=1)")
            continue
        we = sr["WE"]
        sig = sr["significant_bonferroni"]
        print(
            f"  {sid}: obs={we['observed']:.4f} null={we['null_mean']:.4f}±{we['null_sd']:.4f} "
            f"effect={we['effect_size']:.2f}σ p={we['p_value']:.6f} "
            f"{'SIGNIFICANT' if sig else 'not significant'}"
        )


if __name__ == "__main__":
    with torch.no_grad():
        main()
