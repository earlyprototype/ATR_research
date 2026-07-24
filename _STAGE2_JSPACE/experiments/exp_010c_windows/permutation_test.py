#!/usr/bin/env python3
"""EXP_010c-PERM — anisotropy-corrected permutation test on terminal-token
relatedness (GitHub issue #7).

Pre-registered in `_STAGE2_JSPACE/EXP_010c_PERM_SPEC.md` (committed before this
runs). Pure analysis of committed artifacts + the gpt2-medium embedding
matrices; zero model-forward-pass time.

Tests whether the window-loop terminal tokens sit closer together (mean pairwise
cosine over unique types) than random token sets MATCHED on the properties that
drive embedding anisotropy: leading-space status, string length (+/-1 char), and
BPE merge-rank band. Reported in two spaces: raw W_E and the ln_final-gain-
weighted W_U the terminals were actually decoded through.

Deterministic: fix --seed, everything reproduces.
"""
import argparse
import json
import os
from collections import OrderedDict

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")


# ---------------------------------------------------------------------------
# GPT-2 byte-level tokenizer helpers (no external tokenizer dependency)
# ---------------------------------------------------------------------------
def bytes_to_unicode():
    """Standard GPT-2 reversible byte<->unicode map."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\xa1"), ord("\xac") + 1))
          + list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def load_tokenizer(model_dir):
    with open(os.path.join(model_dir, "vocab.json")) as f:
        vocab = json.load(f)  # token_string -> id
    id_to_key = {i: k for k, i in vocab.items()}

    # merge rank: line index at which a byte-pair is formed; earlier => frequent.
    merge_rank = {}
    with open(os.path.join(model_dir, "merges.txt"), encoding="utf-8") as f:
        lines = f.read().split("\n")
    if lines and lines[0].startswith("#"):
        lines = lines[1:]
    r = 0
    for line in lines:
        if not line:
            continue
        a, b = line.split()
        merged = a + b
        if merged not in merge_rank:
            merge_rank[merged] = r
        r += 1

    b2u = bytes_to_unicode()
    u2b = {v: k for k, v in b2u.items()}

    def decode_key(key):
        return bytearray(u2b[c] for c in key).decode("utf-8", errors="replace")

    # surface string -> id  (for mapping via-tail decoded strings back to ids)
    surface_to_id = {}
    for i, key in id_to_key.items():
        surface_to_id.setdefault(decode_key(key), i)

    return vocab, id_to_key, merge_rank, surface_to_id


# ---------------------------------------------------------------------------
# Embedding spaces
# ---------------------------------------------------------------------------
def load_spaces(model_path):
    sd = torch.load(model_path, map_location="cpu", weights_only=True)
    W_E = sd["wte.weight"].to(torch.float64).numpy()          # [V, d]  (tied = raw W_U)
    gamma = sd["ln_f.weight"].to(torch.float64).numpy()       # [d]     ln_final gain
    W_U = W_E * gamma[None, :]                                 # effective decode space
    return W_E, W_U


def unit_rows(M):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return M / n


def mean_pairwise_cos_from_unit(U_idx):
    """Mean pairwise cosine of the rows indexed, given a UNIT-normalised matrix
    passed as the already-selected [k, d] block. Uses the identity
    sum_{i<j} u_i.u_j = (||sum u||^2 - k) / 2."""
    s = U_idx.sum(axis=0)
    k = U_idx.shape[0]
    return (float(s @ s) - k) / (k * (k - 1))


# ---------------------------------------------------------------------------
# Matched-null candidate pools
# ---------------------------------------------------------------------------
class Matcher:
    def __init__(self, id_to_key, merge_rank, n_bands=5, exclude_ids=(50256,)):
        ids = np.array([i for i in id_to_key if i not in exclude_ids], dtype=np.int64)
        keys = [id_to_key[int(i)] for i in ids]
        self.ls = np.array([1 if k.startswith("Ġ") else 0 for k in keys], dtype=np.int8)
        self.length = np.array([len(k) for k in keys], dtype=np.int32)
        rank = np.array([merge_rank.get(k, 0) for k in keys], dtype=np.int64)
        # quintile bands over the merge-rank distribution of the vocab
        qs = np.quantile(rank, np.linspace(0, 1, n_bands + 1)[1:-1])
        self.band = np.digitize(rank, qs).astype(np.int8)
        self.ids = ids
        self.id_index = {int(i): p for p, i in enumerate(ids)}
        self.rank = rank

    def pool(self, tok_id, floor=20):
        """Candidate positions (into self.ids) matched to tok_id, with recorded
        relaxation if the strict pool is under `floor`."""
        p = self.id_index[tok_id]
        ls0, len0, band0 = self.ls[p], self.length[p], self.band[p]
        relax = "none"
        base = (self.ls == ls0)
        m = base & (self.band == band0) & (np.abs(self.length - len0) <= 1)
        m[p] = False
        if m.sum() < floor:                       # relax freq band first
            relax = "band"
            m = base & (np.abs(self.length - len0) <= 1)
            m[p] = False
        if m.sum() < floor:                       # then widen length window
            relax = "band+len2"
            m = base & (np.abs(self.length - len0) <= 2)
            m[p] = False
        return np.where(m)[0], relax


# ---------------------------------------------------------------------------
# Token-set assembly from committed artifacts
# ---------------------------------------------------------------------------
def direct_unique(records, arm):
    """OrderedDict token_string -> id for the direct-decode terminals of an arm."""
    out = OrderedDict()
    for r in records:
        if r["arm"] == arm:
            out.setdefault(r["terminal_token"], r["terminal_token_id"])
    return out


def tail_set(char_records, arm, surface_to_id):
    out = OrderedDict()
    rec = next(r for r in char_records if r["arm"] == arm)
    for surf in rec["tail_decode_terminals"]:
        if surf not in surface_to_id:
            raise KeyError(f"via-tail token {surf!r} not resolvable to a single id")
        out[surf] = surface_to_id[surf]
    return out


def build_sets(results_full, results_scan, char_full, char_scan, surface_to_id):
    a4 = direct_unique(results_full, "A4")
    o8 = direct_unique(results_scan, "O8")
    a5 = direct_unique(results_full, "A5")

    def union(*ds):
        u = OrderedDict()
        for d in ds:
            for k, v in d.items():
                u.setdefault(k, v)
        return u

    wt_a4 = tail_set(char_full, "A4", surface_to_id)
    wt_o8 = tail_set(char_scan, "O8", surface_to_id)
    wt_a5 = tail_set(char_full, "A5", surface_to_id)

    sets = OrderedDict()
    # word-arm direct
    sets["WS_A4"] = ("A4 10->21 direct", a4)
    sets["WS_O8"] = ("O8 8->21 direct", o8)
    sets["WS_POOL"] = ("A4|O8|A5 pooled direct", union(a4, o8, a5))
    # word-arm via-tail
    sets["WT_A4"] = ("A4 via-tail", wt_a4)
    sets["WT_O8"] = ("O8 via-tail", wt_o8)
    sets["WT_A5"] = ("A5 via-tail", wt_a5)
    sets["WT_POOL"] = ("A4|O8|A5 pooled via-tail", union(wt_a4, wt_o8, wt_a5))
    # contrast
    sets["CS_A1"] = ("A1 0->11 direct (punct funnel)", direct_unique(results_full, "A1"))
    sets["CS_A3"] = ("A3 12->23 direct", direct_unique(results_full, "A3"))
    sets["CS_O14"] = ("O14 14->21 direct (off-band)", direct_unique(results_scan, "O14"))
    return sets


# ---------------------------------------------------------------------------
# Per-set test
# ---------------------------------------------------------------------------
def run_set(name, desc, tokset, U_E, U_U, matcher, N, rng):
    ids = list(tokset.values())
    k = len(ids)
    result = {"set": name, "desc": desc, "n_unique": k,
              "tokens": list(tokset.keys()), "token_ids": ids}
    if k < 2:
        result["status"] = "excluded (fewer than 2 unique types; no pairwise cosine)"
        return result

    # per-token matched pools (positions into matcher.ids)
    pools, relax = [], {}
    for tid in ids:
        pos, rlx = matcher.pool(tid)
        pools.append(pos)
        relax[matcher_key(tokset, tid)] = {"pool_size": int(pos.size), "relax": rlx}

    out = {}
    for space, U in (("W_E", U_E), ("W_U", U_U)):
        obs = mean_pairwise_cos_from_unit(U[ids])
        # sampled matched null, vectorised over N
        picks = np.empty((N, k), dtype=np.int64)
        for j, pos in enumerate(pools):
            picks[:, j] = matcher.ids[pos[rng.integers(0, pos.size, size=N)]]
        # enforce distinct types within each null set (resample offending rows)
        for _ in range(50):
            bad = np.array([len(np.unique(row)) < k for row in picks])
            if not bad.any():
                break
            idx = np.where(bad)[0]
            for j, pos in enumerate(pools):
                picks[idx, j] = matcher.ids[pos[rng.integers(0, pos.size, size=idx.size)]]
        # mean pairwise cosine per null set via the ||sum||^2 identity
        S = U[picks].sum(axis=1)                       # [N, d]
        null = (np.einsum("nd,nd->n", S, S) - k) / (k * (k - 1))
        mu, sd = float(null.mean()), float(null.std(ddof=1))
        z = (obs - mu) / sd if sd > 0 else float("nan")
        p = (1 + int((null >= obs).sum())) / (1 + N)
        out[space] = {"observed": float(obs), "null_mean": mu, "null_sd": sd,
                      "z": float(z), "p_value": p}
    result["status"] = "tested"
    result["matching"] = relax
    result["spaces"] = out
    return result


def matcher_key(tokset, tid):
    for s, i in tokset.items():
        if i == tid:
            return s
    return str(tid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True,
                    help="dir with pytorch_model.bin, vocab.json, merges.txt")
    ap.add_argument("--n-null", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260724)
    ap.add_argument("--out", default=os.path.join(OUT, "perm_test_results.json"))
    args = ap.parse_args()

    vocab, id_to_key, merge_rank, surface_to_id = load_tokenizer(args.model_dir)
    W_E, W_U = load_spaces(os.path.join(args.model_dir, "pytorch_model.bin"))
    U_E, U_U = unit_rows(W_E), unit_rows(W_U)
    matcher = Matcher(id_to_key, merge_rank)

    results_full = json.load(open(os.path.join(OUT, "results_full.json")))
    results_scan = json.load(open(os.path.join(OUT, "results_scan.json")))
    char_full = json.load(open(os.path.join(OUT, "terminal_characterisation_full.json")))
    char_scan = json.load(open(os.path.join(OUT, "terminal_characterisation_scan.json")))

    sets = build_sets(results_full, results_scan, char_full, char_scan, surface_to_id)

    rng = np.random.default_rng(args.seed)
    per_set = [run_set(name, desc, ts, U_E, U_U, matcher, args.n_null, rng)
               for name, (desc, ts) in sets.items()]

    tested = [r for r in per_set if r.get("status") == "tested"]
    n_tests = sum(len(r["spaces"]) for r in tested)
    bonferroni = 0.05 / n_tests if n_tests else float("nan")

    payload = {
        "experiment": "EXP_010c-PERM",
        "spec": "_STAGE2_JSPACE/EXP_010c_PERM_SPEC.md",
        "seed": args.seed,
        "n_null": args.n_null,
        "statistic": "mean pairwise cosine over unique token types (unweighted)",
        "spaces": {"W_E": "raw wte rows",
                   "W_U": "ln_final-gain-weighted wte (gamma ⊙ wte); GPT-2 ties W_E=W_U"},
        "null": "matched on leading-space (exact) + length (+/-1) + BPE merge-rank quintile band",
        "n_tests": n_tests,
        "bonferroni_threshold": bonferroni,
        "results": per_set,
    }
    os.makedirs(OUT, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # console summary
    print(f"seed={args.seed} N={args.n_null} tests={n_tests} "
          f"Bonferroni alpha={bonferroni:.5f}\n")
    hdr = f"{'set':10} {'k':>2} {'space':5} {'observed':>9} {'null_mu':>9} " \
          f"{'null_sd':>8} {'z':>7} {'p':>9} sig"
    print(hdr)
    print("-" * len(hdr))
    for r in per_set:
        if r.get("status") != "tested":
            print(f"{r['set']:10} {r['n_unique']:>2}  {r['status']}")
            continue
        for space, s in r["spaces"].items():
            sig = "*" if s["p_value"] < bonferroni else ""
            print(f"{r['set']:10} {r['n_unique']:>2} {space:5} "
                  f"{s['observed']:>9.4f} {s['null_mean']:>9.4f} {s['null_sd']:>8.4f} "
                  f"{s['z']:>7.2f} {s['p_value']:>9.5f} {sig}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
