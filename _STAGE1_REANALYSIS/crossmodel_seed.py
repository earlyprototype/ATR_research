"""Cross-model seeding: transplant GPT-2 Small's converged socialist state into
GPT-2 Medium's residual space and run Medium's ATR loop from there.

Two routes are already closed for Medium: the 277-window census surfaces no
socialist register, and nothing socialist sits under Medium's `D` readout
(median rank 39202). A third is partly closed: a *synthetic* residual state
optimised so its readout IS the register survives ~2-5 passes and then decays
to `D` (optimised_seed.json).

This asks the remaining question. Small's converged state is not synthetic --
it is an actual fixed point of Small's own dynamics, with real per-position
structure that a per-vector optimisation cannot produce. Map it into Medium's
1024-d space and iterate:

  (a) it holds                 -> the basin exists in Medium, merely unreachable
  (b) it decays to `D`         -> Medium has no socialist fixed point
  (c) somewhere third          -> recorded flat

GATE: before any iteration, the mapped state must still read out as socialist
vocabulary in MEDIUM's unembedding. If it does not, the map destroyed the
content and the loop result says nothing. Gate is reported first and separately.

Loop is the registered ATR protocol: inject at blocks.0 input, extract
blocks.<n_layer-1> output, rescale to the seed norm each pass.
"""
import json
import sys
import time

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

torch.set_num_threads(4)

SMALL = "/tmp/claude-0/-home-user-ATR-research/02b8abac-f6fe-5bef-9a2a-e7cd1a7ec491/scratchpad/gpt2"
MEDIUM = "/tmp/claude-0/-home-user-ATR-research/02b8abac-f6fe-5bef-9a2a-e7cd1a7ec491/scratchpad/gpt2-medium"
STATES = "/home/user/lucier-repo/experiments/gpt2_small/output_confidence/converged_tensors.pt"
OUT = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/user/ATR_research/_STAGE1_REANALYSIS/crossmodel_seed.json"
MAX_ITER = int(sys.argv[2]) if len(sys.argv) > 2 else 300

BASIN = [" prolet", " Anarch", " bourgeois", " Marx", " comrade", " proletarian",
         " socialist", " anarchist", " Lenin", " anarchism", " comrades", " labour",
         " communist", " socialism", " revolution", " workers", " solidarity", " union"]
CONTROL = [" kitchen", " tomato", " bicycle", " weather", " hospital", " guitar",
           " forest", " sandwich", " ocean", " furniture", " camera", " blanket"]

CHECKPOINTS = [0, 1, 2, 5, 10, 20, 40, 80, 160, 300]

# ---------------------------------------------------------------- models
t0 = time.time()
small = GPT2LMHeadModel.from_pretrained(SMALL).eval()
medium = GPT2LMHeadModel.from_pretrained(MEDIUM).eval()
for p in small.parameters():
    p.requires_grad_(False)
for p in medium.parameters():
    p.requires_grad_(False)
tok = GPT2TokenizerFast.from_pretrained(SMALL)
print(f"models loaded in {time.time() - t0:.0f}s", flush=True)

ln_s, E_s = small.transformer.ln_f, small.transformer.wte.weight
ln_m, E_m = medium.transformer.ln_f, medium.transformer.wte.weight
NL = medium.config.n_layer

bid = [tok.encode(w)[0] for w in BASIN if len(tok.encode(w)) == 1]
cid = [tok.encode(w)[0] for w in CONTROL if len(tok.encode(w)) == 1]
assert len(bid) == len(BASIN) and len(cid) == len(CONTROL), (len(bid), len(cid))

states = torch.load(STATES, map_location="cpu")


def readout(vec, ln, WE, k=12):
    """(top-k token strings, basin median rank, control median rank)."""
    with torch.no_grad():
        lg = ln(vec) @ WE.T
    order = torch.argsort(lg, descending=True)
    pos = torch.empty_like(order)
    pos[order] = torch.arange(len(order))
    top = [tok.decode([int(i)]) for i in torch.topk(lg, k).indices]

    def med(ids):
        r = sorted(int(pos[i]) + 1 for i in ids)
        return r[len(r) // 2]

    return top, med(bid), med(cid)


def in_basin(top):
    return sum(1 for t in top if t in BASIN)


# ---------------------------------------------------------------- maps 768 -> 1024
print("building maps ...", flush=True)
Ed_s = E_s.detach().double()
Ed_m = E_m.detach().double()
I768 = torch.eye(768, dtype=torch.float64)
RIDGE = 1e-6

# 1. semantic anchor, least squares over the shared 50257-token vocabulary
W_ls = torch.linalg.solve(Ed_s.T @ Ed_s + RIDGE * I768, Ed_s.T @ Ed_m)
# 2. same with an intercept (embeddings are not mean-centred)
mu_s, mu_m = Ed_s.mean(0), Ed_m.mean(0)
Esc, Emc = Ed_s - mu_s, Ed_m - mu_m
W_lsb = torch.linalg.solve(Esc.T @ Esc + RIDGE * I768, Esc.T @ Emc)
# 3. orthogonal Procrustes (rotation only, preserves norms and angles)
U, S, Vh = torch.linalg.svd(Ed_s.T @ Ed_m, full_matrices=False)
W_proc = U @ Vh
# 4. control: zero-pad into the first 768 dims
W_zero = torch.zeros(768, 1024, dtype=torch.float64)
W_zero[:, :768] = I768
# 5. control: random gaussian projection
W_rand = torch.randn(768, 1024, generator=torch.Generator().manual_seed(1),
                     dtype=torch.float64) / (768 ** 0.5)

MAPS = {
    "anchor_ls":   (lambda v: (v.double() @ W_ls).float(), "least-squares E_s -> E_m"),
    "anchor_ls_b": (lambda v: ((v.double() - mu_s) @ W_lsb + mu_m).float(),
                    "least-squares with intercept"),
    "procrustes":  (lambda v: (v.double() @ W_proc).float(), "orthogonal Procrustes"),
    "zeropad":     (lambda v: (v.double() @ W_zero).float(), "CONTROL zero-pad"),
    "randproj":    (lambda v: (v.double() @ W_rand).float(), "CONTROL random projection"),
}


def logit_transfer(v, topk=50):
    """CONTROL (re-synthesis, not a transplant): rebuild each position in Medium
    space as the top-k-weighted mean of Medium embeddings under Small's readout."""
    with torch.no_grad():
        lg = ln_s(v) @ E_s.T
    val, idx = torch.topk(lg, topk, dim=-1)
    w = torch.softmax(val - val.max(dim=-1, keepdim=True).values, dim=-1)
    return torch.einsum("pk,pkd->pd", w, E_m[idx])


MAPS["logit_transfer"] = (logit_transfer, "CONTROL re-synthesis from Small's readout")

# map quality: does a Small token embedding, mapped, retrieve its own token
# in Medium's unembedding?
g = torch.Generator().manual_seed(0)
probe_ids = torch.randperm(50257, generator=g)[:1500]
map_quality = {}
for name, (fn, desc) in MAPS.items():
    if name == "logit_transfer":
        continue
    u = fn(Ed_s[probe_ids].float())
    with torch.no_grad():
        lg = ln_m(u) @ E_m.T
    acc = (lg.argmax(-1) == probe_ids).float().mean().item()
    rk = ((lg.argsort(-1, descending=True) == probe_ids.unsqueeze(1))
          .float().argmax(-1) + 1)
    map_quality[name] = {"desc": desc, "token_retrieval_top1": round(acc, 4),
                         "token_retrieval_median_rank": int(rk.median())}
    print(f"  {name:14s} top1={acc:.3f} median_rank={int(rk.median())}", flush=True)

# ---------------------------------------------------------------- hooks
_cap, _inj = {}, {}
medium.transformer.h[NL - 1].register_forward_hook(
    lambda m, i, o: _cap.__setitem__("x", (o[0] if isinstance(o, tuple) else o).detach()))


def _pre(m, args, kwargs):
    if _inj.get("x") is None:
        return None
    new = _inj["x"].unsqueeze(0)
    if args:
        return (new,) + tuple(args[1:]), kwargs
    kwargs = dict(kwargs)
    kwargs["hidden_states"] = new
    return args, kwargs


medium.transformer.h[0].register_forward_pre_hook(_pre, with_kwargs=True)

_cap_s = {}
small.transformer.h[0].register_forward_pre_hook(
    lambda m, a, kw: _cap_s.__setitem__(
        "x", (a[0] if a else kw["hidden_states"]).detach()), with_kwargs=True)


def natural_norm(model, cap, seq):
    with torch.no_grad():
        model(input_ids=torch.full((1, seq), 262))
    return cap["x"][0].norm(dim=1).mean().item()


NAT_M = natural_norm(medium, _cap, 12)
NAT_S = natural_norm(small, _cap_s, 12)
print(f"natural layer-0 per-token norm: small {NAT_S:.2f}  medium {NAT_M:.2f}", flush=True)


def step(x, seq):
    _inj["x"] = x
    with torch.no_grad():
        medium(input_ids=torch.full((1, seq), 262))
    _inj["x"] = None
    return _cap["x"][0].clone()


# ---------------------------------------------------------------- GATE
print("\n=== GATE: mapped state read out in MEDIUM's unembedding, no iteration ===",
      flush=True)
gate = {}
for pname, v in states.items():
    src_top, src_b, src_c = readout(v[-1], ln_s, E_s)
    gate[pname] = {"source_small_readout": {
        "top12": src_top, "basin_median_rank": src_b, "control_median_rank": src_c}}
    for mname, (fn, _) in MAPS.items():
        u = fn(v)
        top, b, c = readout(u[-1], ln_m, E_m)
        # also: fraction of positions whose top-12 contains basin vocabulary
        per_pos = [in_basin(readout(u[p], ln_m, E_m)[0]) for p in range(u.shape[0])]
        gate[pname][mname] = {
            "top12": top, "basin_median_rank": b, "control_median_rank": c,
            "basin_in_top12": in_basin(top),
            "positions_with_basin_in_top12": sum(1 for x in per_pos if x > 0),
            "n_positions": u.shape[0],
            "mapped_per_token_norm": round(u.norm(dim=1).mean().item(), 2),
        }
    print(f"\n[{pname}] small: {src_top[:8]}")
    for mname in MAPS:
        r = gate[pname][mname]
        print(f"   {mname:15s} basin_rank {r['basin_median_rank']:>6}"
              f"  ctrl {r['control_median_rank']:>6}"
              f"  hits {r['basin_in_top12']:>2}/12"
              f"  pos {r['positions_with_basin_in_top12']}/{r['n_positions']}"
              f"   {r['top12'][:6]}", flush=True)

# gate verdict per map, judged on the four socialist-basin prompts
SOC_PROMPTS = ["Lucier", "Semantic", "Nonsense", "Imperative"]
gate_pass = {}
for mname in MAPS:
    hits = [gate[p][mname]["basin_in_top12"] for p in SOC_PROMPTS]
    ranks = [gate[p][mname]["basin_median_rank"] for p in SOC_PROMPTS]
    gate_pass[mname] = {
        "basin_in_top12_per_prompt": hits,
        "basin_median_rank_per_prompt": ranks,
        # pass = socialist vocabulary is genuinely near the top, not merely
        # better than chance
        "pass": all(h >= 3 for h in hits) and all(r <= 100 for r in ranks),
    }
print("\nGATE VERDICT:", {k: v["pass"] for k, v in gate_pass.items()}, flush=True)

results = {
    "natural_layer0_per_token_norm": {"small": NAT_S, "medium": NAT_M},
    "small_converged_per_token_norm": {k: round(v.norm(dim=1).mean().item(), 1)
                                       for k, v in states.items()},
    "map_quality": map_quality,
    "gate": gate,
    "gate_verdict": gate_pass,
    "max_iter": MAX_ITER,
    "runs": {},
}
json.dump(results, open(OUT, "w"), indent=1)

# ---------------------------------------------------------------- LOOP
# Energy shells, expressed as Medium-space per-token residual norm.
#   natural          Medium's own layer-0 norm
#   conv_natural     the base the registered convention calls "natural" (30)
#   matched_ratio    Small's converged energy relative to Small's natural norm,
#                    carried over to Medium
#   matched_abs      Small's converged per-token norm transplanted unchanged
#   x73 / x218       the registered convention's high shells
SMALL_RATIO = 1518.1 / NAT_S
SHELLS = [
    ("natural", NAT_M),
    ("conv_natural", 30.0),
    ("matched_ratio", SMALL_RATIO * NAT_M),
    ("matched_abs", 1518.0),
    ("x73", 2200.0),
    ("x218", 6500.0),
]

LOCK_TOL = 1e-6   # 1 - cos between consecutive iterates


def run_loop(u0, seq, max_iter=MAX_ITER):
    x = u0.clone()
    N0 = x.norm().item()
    trace = []
    top, b, c = readout(x[-1], ln_m, E_m)
    trace.append({"iter": 0, "basin": b, "control": c, "top": top,
                  "basin_in_top12": in_basin(top), "cos_to_seed": 1.0})
    seed_flat = u0.reshape(-1)
    prev = None
    locked_at = None
    survived = 0
    for it in range(1, max_iter + 1):
        x = x * (N0 / x.norm())
        x = step(x, seq)
        if prev is not None:
            d = 1 - torch.nn.functional.cosine_similarity(
                x.reshape(1, -1), prev.reshape(1, -1)).item()
            if d < LOCK_TOL and locked_at is None:
                locked_at = it
        prev = x.clone()
        top, b, c = readout(x[-1], ln_m, E_m)
        if in_basin(top) >= 3:
            survived = it
        if it in CHECKPOINTS:
            trace.append({
                "iter": it, "basin": b, "control": c, "top": top,
                "basin_in_top12": in_basin(top),
                "cos_to_seed": round(torch.nn.functional.cosine_similarity(
                    x.reshape(1, -1), seed_flat.reshape(1, -1)).item(), 4)})
        if locked_at is not None and it >= max(CHECKPOINTS[:-1]):
            break
    # fill any checkpoints past an exact lock (state is constant by then)
    done = {t["iter"] for t in trace}
    for cp in CHECKPOINTS:
        if cp not in done:
            trace.append({"iter": cp, "basin": b, "control": c, "top": top,
                          "basin_in_top12": in_basin(top),
                          "cos_to_seed": round(torch.nn.functional.cosine_similarity(
                              x.reshape(1, -1), seed_flat.reshape(1, -1)).item(), 4),
                          "from_lock": True})
    trace.sort(key=lambda t: t["iter"])
    return {"shell_total_norm": N0, "locked_at": locked_at,
            "last_iter_with_basin_in_top12": survived, "trace": trace}


# run matrix: every map on the flagship socialist prompt across all shells,
# then the maps that passed the gate on the other prompts at two shells
MATRIX = [(p, m, s) for m in MAPS for p, s in
          [("Lucier", [n for n, _ in SHELLS])]
          for s in s]
EXTRA = [(p, m, s) for m in MAPS if gate_pass[m]["pass"] or m == "logit_transfer"
         for p in ["Semantic", "Nonsense", "Imperative", "Syntactic"]
         for s in ["matched_abs", "x218"]]
MATRIX += EXTRA

shell_of = dict(SHELLS)
print(f"\n=== LOOP: {len(MATRIX)} runs, max {MAX_ITER} iters ===", flush=True)
t0 = time.time()
for i, (pname, mname, sname) in enumerate(MATRIX):
    v = states[pname]
    seq = v.shape[0]
    u = MAPS[mname][0](v)
    u = u / u.norm(dim=1, keepdim=True) * shell_of[sname]
    r = run_loop(u, seq)
    r["prompt"] = pname
    r["map"] = mname
    r["shell"] = sname
    r["shell_per_token_norm"] = shell_of[sname]
    results["runs"][f"{pname}|{mname}|{sname}"] = r
    t = r["trace"]
    print(f"[{i + 1}/{len(MATRIX)}] {pname}|{mname}|{sname} "
          f"lock@{r['locked_at']} survived_to_iter={r['last_iter_with_basin_in_top12']} "
          f"({time.time() - t0:.0f}s)")
    for e in t:
        print(f"     it{e['iter']:>3}  basin {e['basin']:>6}  ctrl {e['control']:>6}"
              f"  hits {e['basin_in_top12']:>2}   {e['top'][:6]}")
    sys.stdout.flush()
    json.dump(results, open(OUT, "w"), indent=1)

json.dump(results, open(OUT, "w"), indent=1)
print("\nwritten", OUT)
