"""EXP_015-FORCED: can a natural-strength arm be engineered to reach the `D`
readout? Pre-registered spec: ../../EXP_015_FORCED_SPEC.md (committed with the
EXP_015-FORCED register row before this file existed).

Attempts (spec section 5; run in order, later attempts only on earlier failure):
    1  FORCED-SEED    25 prompts, cap 300: loud A0 terminal vector tiled across
                      positions, rescaled to the natural layer-0 norm, iterated
                      at natural strength under the registered gate.
    2  EXTENDED-CAP   5-prompt probe, cap 2000: unchanged Control B natural_i
                      configuration via atr_engine2.run_atr_gated directly.
    3  MID-BAND       5-prompt probe, cap 1000: rescale target the geometric
                      mean of natural layer-0 and loud seed-at-23 norms
                      (diagnostic only; not natural strength).
    compare           analysis only, run on an attempt-1 pass: EXP_015
                      machinery (cluster 0.999, ARI, 10000-shuffle perm null).

Attempts 1 and 3 need loop features atr_engine2.run_atr_gated does not expose
(a seeded initial state; a non-natural rescale target), so gated_loop() below
is a local copy of the run_atr_gated loop body with exactly those two
extensions; atr_engine2.py is not modified. Machinery gates G1/G2 (spec
section 4) verify the copy against the committed record and the engine before
any attempt is read, and are STOP conditions.

Usage:
    python exp015_forced.py --model-dir DIR --attempt gates|1|2|3|compare
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))   # atr_engine2
sys.path.insert(0, str(HERE))

from atr_engine2 import get_readout_detail, get_top_tokens, lag_scan, run_atr_gated  # noqa: E402

OUT = HERE / "output"
D_TOKEN_ID = 35        # the `D` readout (spec section 4)
GATE = dict(threshold=0.999, patience=3, check_every=10)  # registered gate
PROBE_N = 5            # first 5 prompts of the registered order (spec section 3)

# The three statistical inputs of record (EXP_010c_PERM_SPEC.md post-run
# addendum); the model files used here must match (spec section 3).
EXPECTED_SHA256 = {
    "pytorch_model.bin": "98c7b0558df2c732799e509a8157d392251b3a6b06e2c72eefb3a00eb10f8318",
    "vocab.json": "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783",
    "merges.txt": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
}


def check_digests(model_dir):
    for fname, want in EXPECTED_SHA256.items():
        h = hashlib.sha256()
        with open(Path(model_dir) / fname, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        got = h.hexdigest()
        assert got == want, f"digest mismatch for {fname}: {got}"
    print("input digests match the EXP_010c_PERM_SPEC.md addendum", flush=True)


def gated_loop(model, prompt, layer_start, layer_end, max_iter, check_start,
               renorm="natural_i", seed_tensor=None,
               threshold=0.999, patience=3, check_every=10, ckpt_path=None):
    """Local copy of atr_engine2.run_atr_gated (gate_lag=1, capture_terminal
    semantics), extended by exactly two features (spec section 4):
      seed_tensor  : optional [seq, d] initial state replacing the prompt-pass
                     seed (attempt 1).
      renorm       : additionally accepts "midband", rescaling to the geometric
                     mean of the natural layer-0 norm and the prompt-pass seed
                     norm at the extraction layer (attempt 3).
    With seed_tensor=None and renorm in ("seed_j", "natural_i") the operations
    match the engine exactly, in the same order (verified by gates G1/G2)."""
    hook_point_read = f"blocks.{layer_end}.hook_resid_post"
    hook_point_write = f"blocks.{layer_start}.hook_resid_pre"
    natural_pre_name = f"blocks.{layer_start}.hook_resid_pre"
    cache_names = {hook_point_read, natural_pre_name}

    with torch.no_grad():
        _, cache = model.run_with_cache(prompt, names_filter=lambda n: n in cache_names)
    prompt_seed = cache[hook_point_read][0].clone()
    initial_norm = prompt_seed.norm().item()
    natural_norm = cache[natural_pre_name][0].norm().item()
    if renorm == "seed_j":
        target_norm = initial_norm
    elif renorm == "natural_i":
        target_norm = natural_norm
    elif renorm == "midband":
        target_norm = (natural_norm * initial_norm) ** 0.5
    else:
        raise ValueError(renorm)

    current_tensor = seed_tensor.clone() if seed_tensor is not None else prompt_seed
    mean_history = [current_tensor.mean(dim=0).clone()]
    recent_means = []
    consecutive = 0
    lock_in_iter = None
    final_cos = 1.0
    i = 0
    start_iter = 1

    # Mid-prompt checkpoint/resume (operational only; the loop state is saved
    # and restored exactly, so the iterate sequence is identical to an
    # uninterrupted run — determinism is per-iteration, not per-process).
    if ckpt_path is not None and Path(ckpt_path).exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        current_tensor = ck["current_tensor"]
        mean_history = [t for t in ck["mean_history"]]
        recent_means = [t for t in ck["recent_means"]]
        consecutive = int(ck["consecutive"])
        final_cos = float(ck["final_cos"])
        start_iter = int(ck["i"]) + 1
        print(f"    (resumed at iteration {start_iter})", flush=True)

    for i in range(start_iter, max_iter + 1):
        current_norm = current_tensor.norm().item()
        if current_norm > 0:
            current_tensor = current_tensor * (target_norm / current_norm)

        inject_tensor = current_tensor.clone()

        def injection_hook(resid, hook, tensor=inject_tensor):
            resid[0, :, :] = tensor
            return resid

        model.add_hook(hook_point_write, injection_hook)
        try:
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    prompt, names_filter=lambda n: n == hook_point_read)
        finally:
            model.reset_hooks()

        current_tensor = cache[hook_point_read][0].clone()
        mean_vec = current_tensor.mean(dim=0).clone()
        recent_means.append(mean_vec)
        if len(recent_means) > 9:
            recent_means.pop(0)

        if i >= check_start and i % check_every == 0:
            cos = F.cosine_similarity(
                mean_vec.unsqueeze(0), mean_history[0].unsqueeze(0)).item()
            final_cos = cos
            consecutive = consecutive + 1 if cos > threshold else 0
            if consecutive >= patience:
                lock_in_iter = i
                break

        mean_history.append(mean_vec)
        if len(mean_history) > 1:
            mean_history.pop(0)

        if ckpt_path is not None and i % 50 == 0:
            torch.save({"i": i, "current_tensor": current_tensor,
                        "mean_history": mean_history, "recent_means": recent_means,
                        "consecutive": consecutive, "final_cos": final_cos},
                       ckpt_path)

    if ckpt_path is not None and Path(ckpt_path).exists():
        Path(ckpt_path).unlink()

    last_vec = current_tensor[-1, :].clone()
    top = get_top_tokens(model, last_vec, k=1)[0]
    detail = get_readout_detail(model, last_vec)
    return {
        "terminal_token": top[0],
        "terminal_token_id": detail["top_token_ids"][0],
        "terminal_prob": float(top[1]),
        "lock_in_iter": lock_in_iter,
        "converged": lock_in_iter is not None,
        "n_iters": i,
        "final_cos_sim_mean": final_cos,
        "top_logit_margin": detail["top_logit_margin"],
        "entropy": detail["entropy"],
        "renorm": renorm,
        "target_norm": target_norm,
        "natural_norm_at_0": natural_norm,
        "seed_norm_at_j": initial_norm,
        "seeded": seed_tensor is not None,
        "terminal_mean_vec": current_tensor.mean(dim=0).clone(),
        "terminal_last_vec": last_vec,
        "lag_scan": ({str(k): float(v) for k, v in lag_scan(recent_means).items()}
                     if len(recent_means) > 1 else None),
    }


def load_model(model_dir):
    from run_exp010c import load_model_from_local
    torch.manual_seed(42)
    model = load_model_from_local(str(model_dir), "gpt2-medium")
    model.eval()
    return model


def load_prompts():
    subset = json.load(open(OUT / "prompt_subset.json"))
    assert len(subset) == 25
    return subset


def load_loud_terminals():
    t = torch.load(OUT / "terminals_full.pt", map_location="cpu", weights_only=True)
    return {k.split("|", 1)[1]: v for k, v in t.items() if k.startswith("A0|")}


def norm_crosscheck(model, prompts):
    """Spec section 4: recomputed natural layer-0 norms must match the recorded
    Control B file at its 4 recorded decimals."""
    rec = json.load(open(OUT / "natural_resid_norms_energynorm_A0.json"))
    for p in prompts:
        with torch.no_grad():
            _, cache = model.run_with_cache(
                p["prompt"], names_filter=lambda n: n == "blocks.0.hook_resid_pre")
        got = round(cache["blocks.0.hook_resid_pre"][0].norm().item(), 4)
        want = rec[p["id"]]["0"]
        assert abs(got - want) < 5e-5, (p["id"], got, want)
    print(f"norm cross-check: {len(prompts)} prompts match the Control B record", flush=True)


def run_gates(model, prompts):
    """G1 loud replication + G2 engine equivalence (spec section 4). STOP on fail."""
    results_full = json.load(open(OUT / "results_full.json"))
    loud_rec = {r["prompt_id"]: r for r in results_full if r["arm"] == "A0"}
    loud_term = load_loud_terminals()
    report = {}

    for p in prompts[:2]:
        r = gated_loop(model, p["prompt"], 0, 23, max_iter=300, check_start=100,
                       renorm="seed_j", **GATE)
        want = loud_rec[p["id"]]
        cos = F.cosine_similarity(r["terminal_mean_vec"].unsqueeze(0),
                                  loud_term[p["id"]]["mean"].unsqueeze(0)).item()
        ok = (r["terminal_token"] == want["terminal_token"] == "D"
              and r["terminal_token_id"] == 35
              and r["lock_in_iter"] == want["lock_in_iter"] == 120
              and cos > 0.9999)
        report[f"G1_{p['id']}"] = {
            "terminal_token": r["terminal_token"], "token_id": r["terminal_token_id"],
            "lock_in_iter": r["lock_in_iter"], "cos_vs_committed_mean": cos, "pass": ok}
        print(f"G1 {p['id']}: token={r['terminal_token']!r} id={r['terminal_token_id']} "
              f"lock={r['lock_in_iter']} cos={cos:.6f} -> {'PASS' if ok else 'FAIL'}",
              flush=True)
        if not ok:
            return report, False

    for p in prompts[:2]:
        eng = run_atr_gated(model, p["prompt"], 0, 23, max_iter=60, check_start=20,
                            capture_terminal=True, renorm="natural_i", **GATE)
        loc = gated_loop(model, p["prompt"], 0, 23, max_iter=60, check_start=20,
                         renorm="natural_i", **GATE)
        diff = (eng["terminal_mean_vec"] - loc["terminal_mean_vec"]).abs().max().item()
        ok = diff <= 1e-5
        report[f"G2_{p['id']}"] = {"max_abs_diff": diff, "pass": ok}
        print(f"G2 {p['id']}: max abs diff engine vs local = {diff:.2e} "
              f"-> {'PASS' if ok else 'FAIL'}", flush=True)
        if not ok:
            return report, False

    return report, True


def strip_tensors(r):
    return {k: v for k, v in r.items()
            if k not in ("terminal_mean_vec", "terminal_last_vec")}


def run_attempt(model, prompts, attempt):
    # Per-prompt shards with resume: prompt runs are deterministic and
    # independent (the repo's resume note on run_exp010c.py records the same
    # property), so execution may be chunked across process boundaries without
    # affecting any result. Shards are working files; the registered artifacts
    # are the assembled .json/.pt written when all prompts are present.
    shard_dir = OUT / f"exp015_forced_attempt{attempt}_shards"
    shard_dir.mkdir(exist_ok=True)
    loud_term = load_loud_terminals() if attempt == 1 else None
    records, terminals = [], {}
    t0 = time.time()
    for p in prompts:
        spath = shard_dir / f"{p['id']}.json"
        tpath = shard_dir / f"{p['id']}.pt"
        if spath.exists() and tpath.exists():
            rec = json.loads(spath.read_text())
            term = torch.load(tpath, map_location="cpu", weights_only=True)
            records.append(rec)
            terminals[f"F{attempt}|{p['id']}"] = term
            print(f"  [attempt {attempt}] {p['id']:<16} resumed from shard "
                  f"({rec['terminal_token']!r} id={rec['terminal_token_id']})", flush=True)
            continue
        if attempt == 1:
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    p["prompt"], names_filter=lambda n: n == "blocks.0.hook_resid_pre")
            nat = cache["blocks.0.hook_resid_pre"][0]
            seq_len = nat.shape[0]
            seed = loud_term[p["id"]]["last"].unsqueeze(0).expand(seq_len, -1).clone()
            seed = seed * (nat.norm().item() / seed.norm().item())
            r = gated_loop(model, p["prompt"], 0, 23, max_iter=300, check_start=100,
                           renorm="natural_i", seed_tensor=seed, **GATE,
                           ckpt_path=shard_dir / f"{p['id']}.ckpt")
        elif attempt == 2:
            # Recorded deviation from the spec's "calls run_atr_gated directly":
            # the local loop is used instead, solely for mid-prompt
            # checkpointability under this container's process time limits.
            # Gate G2 established exact equivalence (max elementwise abs diff
            # 0.0e+00) between the two in this configuration, so the iterate
            # sequence is identical.
            r = gated_loop(model, p["prompt"], 0, 23, max_iter=2000, check_start=100,
                           renorm="natural_i", **GATE,
                           ckpt_path=shard_dir / f"{p['id']}.ckpt")
        elif attempt == 3:
            r = gated_loop(model, p["prompt"], 0, 23, max_iter=1000, check_start=100,
                           renorm="midband", **GATE,
                           ckpt_path=shard_dir / f"{p['id']}.ckpt")
        term = {"mean": r["terminal_mean_vec"], "last": r["terminal_last_vec"]}
        terminals[f"F{attempt}|{p['id']}"] = term
        rec = strip_tensors(r)
        rec.update(prompt_id=p["id"], category=p["category"], attempt=attempt)
        records.append(rec)
        torch.save(term, tpath)              # tensors first,
        spath.write_text(json.dumps(rec))    # json is the completion marker
        print(f"  [attempt {attempt}] {p['id']:<16} -> {rec['terminal_token']!r:14} "
              f"id={rec['terminal_token_id']} lock={rec['lock_in_iter']} "
              f"iters={rec['n_iters']} cos={rec['final_cos_sim_mean']:.4f}", flush=True)

    n_d = sum(r["terminal_token_id"] == D_TOKEN_ID for r in records)
    n_conv = sum(r["converged"] for r in records)
    need = 13 if attempt == 1 else 3
    passed = n_d >= need
    summary = {"attempt": attempt, "n_prompts": len(prompts), "n_d_readout": n_d,
               "n_converged": n_conv, "criterion": f">={need} of {len(prompts)} D readouts",
               "pass": passed,
               "seconds_final_invocation": round(time.time() - t0, 1)}
    print(f"attempt {attempt}: D readout on {n_d}/{len(prompts)}, converged "
          f"{n_conv}/{len(prompts)} -> criterion {'PASS' if passed else 'FAIL'}", flush=True)
    return records, terminals, summary


def run_compare():
    """Spec section 6: EXP_015 machinery, unchanged, on the attempt-1 terminals."""
    from analyze_terminals import cluster, CLUSTER_THRESHOLD
    from compare_small_basins import adjusted_rand_index, permutation_p
    from exp015_natural_ari import load_terminals

    prompts = load_prompts()
    pids = [p["id"] for p in prompts]
    small = load_terminals(OUT / "terminals_small_010d.pt")
    loud = load_terminals(OUT / "terminals_full.pt")
    forced = torch.load(OUT / "terminals_exp015_forced_attempt1.pt",
                        map_location="cpu", weights_only=True)

    report = {"experiment": "EXP_015-FORCED", "spec": "../../EXP_015_FORCED_SPEC.md",
              "cluster_threshold": CLUSTER_THRESHOLD, "n_perm": 10000, "perm_seed": 42}

    s_lab, s_n = cluster([small[("SMALL", pid)]["mean"] for pid in pids], CLUSTER_THRESHOLD)
    g_lab, g_n = cluster([loud[("A0", pid)]["mean"] for pid in pids], CLUSTER_THRESHOLD)
    g_ari = adjusted_rand_index(s_lab, g_lab)
    g_p = permutation_p(s_lab, g_lab, g_ari, n_perm=10000)
    gate_pass = round(g_ari, 3) == 0.200 and round(g_p, 4) == 0.0009
    report["reproduction_gate"] = {"small_basins": s_n, "loud_A0_basins": g_n,
                                   "ari": round(g_ari, 4), "perm_p": round(g_p, 5),
                                   "pass": gate_pass}
    print(f"[gate] ARI {g_ari:.4f} p {g_p:.5f} -> {'PASS' if gate_pass else 'FAIL'}",
          flush=True)
    if not gate_pass:
        (OUT / "exp015_forced_ari.json").write_text(json.dumps(report, indent=2))
        print("STOP: EXP_015 reproduction gate failed.")
        return 1

    f_lab, f_n = cluster([forced[f"F1|{pid}"]["mean"] for pid in pids], CLUSTER_THRESHOLD)
    trivial = f_n in (1, len(pids))
    primary = {"forced_basins": f_n, "small_basins": s_n,
               "trivial_partition_guard_fired": trivial}
    if trivial:
        primary["reading"] = "UNANSWERABLE at registered threshold (spec section 6 guard)"
        print(f"[primary] forced partition trivial ({f_n} basins); guard fired.")
    else:
        ari = adjusted_rand_index(s_lab, f_lab)
        p = permutation_p(s_lab, f_lab, ari, n_perm=10000)
        primary.update(ari=round(ari, 4), perm_p=round(p, 5),
                       labels_small=s_lab, labels_forced=f_lab)
        settled = ari > 0 and p < 0.05
        primary["decision"] = ("settledness carried the EXP_015 null"
                               if settled else "loudness carried the EXP_015 null")
        print(f"[primary] Small basins {s_n}, forced basins {f_n}")
        print(f"[primary] ARI(Small, forced) = {ari:.4f}, perm p = {p:.5f}")
        print(f"[primary] decision rule -> {primary['decision']}")
    report["primary"] = primary

    sweep = []
    for thr in (0.99, 0.995, 0.999, 0.9995):
        sl, sn = cluster([small[("SMALL", pid)]["mean"] for pid in pids], thr)
        fl, fn = cluster([forced[f"F1|{pid}"]["mean"] for pid in pids], thr)
        a = adjusted_rand_index(sl, fl)
        pp = permutation_p(sl, fl, a, n_perm=10000)
        sweep.append({"threshold": thr, "small_basins": sn, "forced_basins": fn,
                      "ari": round(a, 4), "perm_p": round(pp, 5)})
        print(f"[sweep] thr {thr}: S {sn} F {fn} ARI {a:.4f} p {pp:.5f}", flush=True)
    report["threshold_sweep"] = sweep

    (OUT / "exp015_forced_ari.json").write_text(json.dumps(report, indent=2))
    print("Saved -> output/exp015_forced_ari.json")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--attempt", required=True, choices=["gates", "1", "2", "3", "compare"])
    args = ap.parse_args()

    if args.attempt == "compare":
        return run_compare()

    assert args.model_dir, "--model-dir required for model stages"
    prompts = load_prompts()

    if args.attempt == "gates":
        check_digests(args.model_dir)
        model = load_model(args.model_dir)
        report, ok = run_gates(model, prompts)
        norm_crosscheck(model, prompts)
        (OUT / "exp015_forced_gates.json").write_text(json.dumps(report, indent=2))
        print(f"gates: {'PASS' if ok else 'FAIL (STOP)'}")
        return 0 if ok else 1

    attempt = int(args.attempt)
    use = prompts if attempt == 1 else prompts[:PROBE_N]
    # Preflight (digest check, norm cross-check) runs once per attempt: the
    # first chunk records it in the log; resumed chunks skip it, the inputs
    # being unchanged files verified minutes earlier in the same log.
    sdir = OUT / f"exp015_forced_attempt{attempt}_shards"
    fresh = not (sdir.exists() and any(sdir.iterdir()))
    if fresh:
        check_digests(args.model_dir)
    model = load_model(args.model_dir)
    if fresh:
        norm_crosscheck(model, use)
    records, terminals, summary = run_attempt(model, use, attempt)
    torch.save(terminals, OUT / f"terminals_exp015_forced_attempt{attempt}.pt")
    (OUT / f"exp015_forced_attempt{attempt}.json").write_text(
        json.dumps({"summary": summary, "records": records}, indent=2))
    print(f"Artifacts: exp015_forced_attempt{attempt}.json, "
          f"terminals_exp015_forced_attempt{attempt}.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
