#!/usr/bin/env python3
"""EXP_010c-PERM: anisotropy-corrected permutation test on window-loop terminals.

Implements _STAGE2_JSPACE/EXP_010c_PERM_SPEC.md (pre-registered; committed
before this script ran). Issue: earlyprototype/ATR_research#7.

Zero model forward passes: loads gpt2-medium's state dict for wte only.
Deterministic: seed 20260725, per-set substreams rng([SEED, set_index]).

Usage:
    python permutation_test.py --model-dir /path/to/gpt2-medium-files \
        [--n-null 10000] [--out output/permutation_results.json]

The model dir must contain pytorch_model.bin, vocab.json, merges.txt.
"""

import argparse
import hashlib
import json
import os
from itertools import combinations

import numpy as np
import torch

SEED = 20260725
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "output")

# ---------------------------------------------------------------- GPT-2 bytes


def bytes_to_unicode():
    """GPT-2's byte<->unicode table (verbatim algorithm from the reference impl)."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


B2U = bytes_to_unicode()
U2B = {v: k for k, v in B2U.items()}


def encode_str(s):
    """Decoded token string -> vocab key (e.g. ' until' -> 'Ġuntil')."""
    return "".join(B2U[b] for b in s.encode("utf-8"))


def decode_key(key):
    """Vocab key -> decoded token string."""
    return bytes(U2B[c] for c in key).decode("utf-8", errors="replace")


# ------------------------------------------------------------------ token sets


def build_token_sets():
    """Exact sets of spec section 3, derived from the committed artifacts."""
    with open(os.path.join(OUT_DIR, "terminal_characterisation_full.json")) as f:
        char_full = json.load(f)
    with open(os.path.join(OUT_DIR, "terminal_characterisation_scan.json")) as f:
        char_scan = json.load(f)
    chars = {r["arm"]: r for r in char_full + char_scan}

    with open(os.path.join(OUT_DIR, "results_full.json")) as f:
        res = json.load(f)
    with open(os.path.join(OUT_DIR, "results_scan.json")) as f:
        res += json.load(f)
    direct_ids = {(r["arm"], r["terminal_token"]): r["terminal_token_id"] for r in res}
    return chars, direct_ids


def make_sets(chars, direct_ids, vocab):
    def ids_for(arm, field):
        out = {}
        for s in chars[arm][field]:
            tid = vocab[encode_str(s)]
            if field == "decode_terminals":
                assert direct_ids[(arm, s)] == tid, (arm, s, tid)
            out[s] = tid
        return out

    sets = {}
    sets["A4_direct"] = ids_for("A4", "decode_terminals")
    sets["O8_direct"] = ids_for("O8", "decode_terminals")
    sets["A5_direct"] = ids_for("A5", "decode_terminals")
    sets["A4_tail"] = ids_for("A4", "tail_decode_terminals")
    sets["O8_tail"] = ids_for("O8", "tail_decode_terminals")
    sets["A5_tail"] = ids_for("A5", "tail_decode_terminals")
    pooled_d, pooled_t = {}, {}
    for a in ("A4_direct", "O8_direct", "A5_direct"):
        pooled_d.update(sets[a])
    for a in ("A4_tail", "O8_tail", "A5_tail"):
        pooled_t.update(sets[a])
    sets["pooled_word_direct"] = pooled_d
    sets["pooled_word_tail"] = pooled_t
    sets["A1_direct"] = ids_for("A1", "decode_terminals")
    sets["A1_tail"] = ids_for("A1", "tail_decode_terminals")
    return sets  # insertion order == spec set numbering 1..10


# ---------------------------------------------------------------- matching


def build_features(vocab, merges_path):
    """Per-token (leading_space, decoded_len, band) per spec section 6."""
    with open(merges_path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    assert lines[0].startswith("#"), "merges.txt header expected"
    merges = lines[1:]
    assert len(merges) == 50000, len(merges)
    rank = {}
    for k, line in enumerate(merges):
        a, b = line.split(" ")
        t = a + b
        if t not in rank:
            rank[t] = k

    feats = {}
    for key, tid in vocab.items():
        if tid == 50256:  # <|endoftext|> excluded from all pools
            continue
        band = "byte" if key not in rank else rank[key] // 5000
        feats[tid] = (key.startswith("Ġ"), len(decode_key(key)), band)
    return feats


def candidate_pool(tid, feats):
    sp, length, band = feats[tid]
    return np.array(
        [j for j, (s2, l2, b2) in feats.items() if s2 == sp and abs(l2 - length) <= 1 and b2 == band],
        dtype=np.int64,
    )


# ---------------------------------------------------------------- statistic


def mean_pairwise_cos(ids, emb_n):
    """Mean pairwise cosine over unique types; emb_n is row-normalised W_E."""
    v = emb_n[ids]
    sims = v @ v.T
    iu = np.triu_indices(len(ids), k=1)
    return float(sims[iu].mean())


def null_distribution(pools, n_null, rng):
    """Draw n_null sets: one token per pool, distinct within each set."""
    k = len(pools)
    draws = np.empty((n_null, k), dtype=np.int64)
    for i in range(n_null):
        while True:
            pick = [pool[rng.integers(len(pool))] for pool in pools]
            if len(set(pick)) == k:
                break
        draws[i] = pick
    return draws


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--n-null", type=int, default=10000)
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "permutation_results.json"))
    args = ap.parse_args()

    # Provenance attestation (PR #19 review): digest the exact statistical
    # inputs so the recorded null is tied to immutable file contents, not a
    # mutable download route. Digests are serialized into the results JSON.
    input_digests = {}
    for fname in ("pytorch_model.bin", "vocab.json", "merges.txt"):
        h = hashlib.sha256()
        with open(os.path.join(args.model_dir, fname), "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        input_digests[fname] = h.hexdigest()

    with open(os.path.join(args.model_dir, "vocab.json")) as f:
        vocab = json.load(f)
    chars, direct_ids = build_token_sets()
    sets = make_sets(chars, direct_ids, vocab)

    sd = torch.load(
        os.path.join(args.model_dir, "pytorch_model.bin"),
        map_location="cpu",
        weights_only=True,
    )
    assert "wte.weight" in sd and not any("lm_head" in k for k in sd), (
        "spec section 4 premise: tied weights, no separate lm_head"
    )
    wte = sd["wte.weight"].float().numpy()
    assert wte.shape == (50257, 1024), wte.shape
    # W_E rows; W_U = wte^T (tied), so W_U column t == W_E row t. Both spaces
    # are computed from this one matrix; cosines are identical by identity.
    emb_n = wte / np.linalg.norm(wte, axis=1, keepdims=True)

    feats = build_features(vocab, os.path.join(args.model_dir, "merges.txt"))

    n_testable = sum(1 for d in sets.values() if len(d) >= 2)
    alpha = 0.05 / n_testable

    results = {
        "spec": "_STAGE2_JSPACE/EXP_010c_PERM_SPEC.md",
        "issue": "earlyprototype/ATR_research#7",
        "input_sha256": input_digests,
        "seed": SEED,
        "n_null": args.n_null,
        "space_note": (
            "checkpoint has wte.weight only (no lm_head): GPT-2 tied weights, "
            "W_U = wte^T; W_U-column cosines are identical to W_E-row cosines "
            "by identity, so one test per set (spec section 4)"
        ),
        "matching": "leading-space exact; decoded length +/-1; merge-rank band rank//5000 ('byte' band for the 256 byte tokens; <|endoftext|> excluded)",
        "n_testable_sets": n_testable,
        "bonferroni_alpha": alpha,
        "sets": [],
    }

    for set_index, (name, d) in enumerate(sets.items(), start=1):
        tokens = {s: int(t) for s, t in d.items()}
        entry = {
            "set_index": set_index,
            "set": name,
            "tokens": tokens,
            "n_types": len(d),
        }
        if len(d) < 2:
            entry["status"] = "N/A - singleton, pairwise statistic undefined (spec section 3)"
            results["sets"].append(entry)
            print(f"[{set_index}] {name}: n=1, N/A")
            continue

        ids = np.array(sorted(d.values()), dtype=np.int64)
        pools = [candidate_pool(t, feats) for t in ids]
        rng = np.random.default_rng([SEED, set_index])
        draws = null_distribution(pools, args.n_null, rng)

        obs = mean_pairwise_cos(ids, emb_n)
        null_stats = np.array([mean_pairwise_cos(row, emb_n) for row in draws])
        null_mean = float(null_stats.mean())
        null_sd = float(null_stats.std(ddof=1))
        p = float((1 + int((null_stats >= obs).sum())) / (args.n_null + 1))
        z = (obs - null_mean) / null_sd

        entry.update(
            {
                "status": "tested",
                "pool_sizes": {decode_key_for_id(t, vocab): int(len(p_)) for t, p_ in zip(ids, pools)},
                "observed_mean_pairwise_cos_WE": obs,
                "observed_mean_pairwise_cos_WU": obs,  # identical by tying (spec section 4)
                "null_mean": null_mean,
                "null_sd": null_sd,
                "p_empirical": p,
                "z_effect_size": z,
                "passes_bonferroni": bool(p < alpha),
            }
        )
        results["sets"].append(entry)
        print(
            f"[{set_index}] {name}: n={len(d)} obs={obs:.4f} null={null_mean:.4f}"
            f"+/-{null_sd:.4f} p={p:.5f} z={z:+.2f} pass={p < alpha}"
        )

    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print("wrote", args.out)


def decode_key_for_id(tid, vocab, _inv={}):
    if not _inv:
        _inv.update({v: k for k, v in vocab.items()})
    return decode_key(_inv[int(tid)])


if __name__ == "__main__":
    main()
