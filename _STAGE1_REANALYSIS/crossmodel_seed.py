"""Cross-model seeding: transplant GPT-2 Small's converged socialist state into
GPT-2 Medium's residual space and run Medium's ATR loop from there.

Two routes are already closed for Medium: the 277-window census surfaces no
socialist register, and nothing socialist sits under Medium's `D` readout
(basin median rank 39202). A third is partly closed: a *synthetic* residual
state optimised so its readout is a hand-picked socialist token list survives
~2-5 passes and then decays to `D` (optimised_seed.json).

This asks the remaining question. Small's converged state is not synthetic -- it
is an actual fixed point of Small's own dynamics, with per-position structure no
per-vector optimisation produces. Map it into Medium's 1024-d space and iterate:

  (a) it holds                 -> the basin exists in Medium, merely unreachable
  (b) it decays to `D`         -> Medium has no socialist fixed point
  (c) somewhere third          -> recorded flat

GATE, run and reported first: the mapped state must still read out as socialist
vocabulary in MEDIUM's unembedding BEFORE any iteration. A map that fails the
gate has destroyed the content and its loop result answers nothing. Loop results
for gate-failed maps are recorded but flagged, never used for the headline.

The gate is shell-independent: ln_f(a*u) == ln_f(u) exactly for a > 0, so
rescaling to an energy shell cannot change iteration-0 readout.

Maps tried, 768 -> 1024:
  A  anchor_ls      lstsq  min ||E_s W - E_m||, applied to the raw residual
  A' anchor_ls_b    same with an intercept (embeddings are not mean-centred)
  B  procrustes     orthogonal, from SVD(E_s^T E_m); angle-preserving
  C  readout_ls     the readout-preserving derivation. Logit_i = ln_f(v).e_i, so
                    the map must act on ln_f(v), not v, and the correct
                    least-squares direction is min ||E_m X - E_s|| with the map
                    being X^T. Medium's ln_f is then inverted.
  C' readout_proc   same placement, orthogonal map
  D  logit_lstsq    no vocabulary-alignment map at all: solve directly for the
                    Medium state whose 50257-d readout best reproduces Small's,
                    h = (E_m^T E_m + lam I)^-1 E_m^T l_small
  E  readout_match  gradient descent on the Medium state to match Small's own
                    readout distribution (not a hand-picked token list). This is
                    a re-synthesis rather than a transplant and overlaps
                    optimised_seed.py; included because it is the strongest
                    possible attempt to put Small's readout into Medium.
  Z  zeropad        CONTROL, identity into the first 768 dims
  Z' randproj       CONTROL, random gaussian projection
  Z" logit_transfer CONTROL, top-50-weighted mean of Medium's own embeddings

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
SOC_PROMPTS = ["Lucier", "Semantic", "Nonsense", "Imperative"]

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

ln_s, E_s = small.transformer.ln_f, small.transformer.wte.weight.detach()
ln_m, E_m = medium.transformer.ln_f, medium.transformer.wte.weight.detach()
NL = medium.config.n_layer

bid = [tok.encode(w)[0] for w in BASIN if len(tok.encode(w)) == 1]
cid = [tok.encode(w)[0] for w in CONTROL if len(tok.encode(w)) == 1]
assert len(bid) == len(BASIN) and len(cid) == len(CONTROL)
states = torch.load(STATES, map_location="cpu")


def readout(vec, ln, WE, k=12):
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


def zlogits(v, ln, WE):
    """standardised readout vector, for comparing readouts across models"""
    with torch.no_grad():
        lg = (ln(v) @ WE.T).double()
    return (lg - lg.mean(-1, keepdim=True)) / lg.std(-1, keepdim=True)


# ---------------------------------------------------------------- maps
print("building maps ...", flush=True)
Ed_s, Ed_m = E_s.double(), E_m.double()
I768 = torch.eye(768, dtype=torch.float64)
I1024 = torch.eye(1024, dtype=torch.float64)
RIDGE = 1e-6

W_ls = torch.linalg.solve(Ed_s.T @ Ed_s + RIDGE * I768, Ed_s.T @ Ed_m)
mu_s, mu_m = Ed_s.mean(0), Ed_m.mean(0)
Esc, Emc = Ed_s - mu_s, Ed_m - mu_m
W_lsb = torch.linalg.solve(Esc.T @ Esc + RIDGE * I768, Esc.T @ Emc)
U, S, Vh = torch.linalg.svd(Ed_s.T @ Ed_m, full_matrices=False)
W_proc = U @ Vh
# readout-preserving direction: min ||E_m X - E_s||, map is X^T
X_ro = torch.linalg.solve(Ed_m.T @ Ed_m + RIDGE * I1024, Ed_m.T @ Ed_s)
W_zero = torch.zeros(768, 1024, dtype=torch.float64)
W_zero[:, :768] = I768
W_rand = torch.randn(768, 1024, generator=torch.Generator().manual_seed(1),
                     dtype=torch.float64) / (768 ** 0.5)

gam_m, bet_m = ln_m.weight.detach().double(), ln_m.bias.detach().double()


def inv_lnf_m(hm, K=1e4):
    """residual u with ln_f_m(u) proportional to hm up to a fixed affine term.
    K large so the beta contamination vanishes; ln_f re-normalises anyway."""
    z = (K * hm - bet_m) / gam_m
    z = z - z.mean(-1, keepdim=True)
    return (z / z.std(-1, keepdim=True)).float()


def _lnf_map(P):
    def f(v):
        return inv_lnf_m(ln_s(v).detach().double() @ P)
    return f


G_m = Ed_m.T @ Ed_m
_Gi = {lam: torch.linalg.inv(G_m + lam * I1024) for lam in (0.1,)}


def logit_lstsq(v, lam=0.1):
    Lc = zlogits(v, ln_s, E_s)
    return inv_lnf_m((Lc @ Ed_m) @ _Gi[lam])


def logit_transfer(v, topk=50):
    with torch.no_grad():
        lg = ln_s(v) @ E_s.T
    val, idx = torch.topk(lg, topk, dim=-1)
    w = torch.softmax(val - val.max(dim=-1, keepdim=True).values, dim=-1)
    return torch.einsum("pk,pkd->pd", w, E_m[idx])


MAPS = {
    "anchor_ls":      (lambda v: (v.double() @ W_ls).float(), "lstsq E_s->E_m on residual"),
    "anchor_ls_b":    (lambda v: ((v.double() - mu_s) @ W_lsb + mu_m).float(),
                       "lstsq with intercept"),
    "procrustes":     (lambda v: (v.double() @ W_proc).float(), "orthogonal, on residual"),
    "readout_ls":     (_lnf_map(X_ro.T), "lstsq E_m->E_s, applied to ln_f(v)"),
    "readout_proc":   (_lnf_map(W_proc), "orthogonal, applied to ln_f(v)"),
    "logit_lstsq":    (logit_lstsq, "direct 50257-d readout reconstruction"),
    "zeropad":        (lambda v: (v.double() @ W_zero).float(), "CONTROL zero-pad"),
    "randproj":       (lambda v: (v.double() @ W_rand).float(), "CONTROL random proj"),
    "logit_transfer": (logit_transfer, "CONTROL top-50 embedding re-synthesis"),
}

# map quality: does a Small token embedding, mapped, retrieve its own token in
# Medium's unembedding?
g = torch.Generator().manual_seed(0)
probe_ids = torch.randperm(50257, generator=g)[:1500]
map_quality = {}
for name, (fn, desc) in MAPS.items():
    u = fn(Ed_s[probe_ids].float())
    with torch.no_grad():
        lg = ln_m(u) @ E_m.T
    acc = (lg.argmax(-1) == probe_ids).float().mean().item()
    rk = ((lg.argsort(-1, descending=True) == probe_ids.unsqueeze(1))
          .float().argmax(-1) + 1)
    map_quality[name] = {"desc": desc, "token_retrieval_top1": round(acc, 4),
                         "token_retrieval_median_rank": int(rk.median())}
    print(f"  {name:15s} top1={acc:.3f} median_rank={int(rk.median())}", flush=True)


# ---- map E: gradient descent to Small's own readout distribution -------------
def readout_match(v, steps=300, topk=512, beta=3.0, tag=""):
    Z = zlogits(v, ln_s, E_s).float()                 # [P,50257] standardised
    val, idx = torch.topk(Z, topk, dim=-1)
    q = torch.zeros_like(Z)
    q.scatter_(-1, idx, torch.softmax(beta * (val - val[:, :1]), dim=-1))
    u = logit_lstsq(v).clone().requires_grad_(True)   # warm start from map D
    opt = torch.optim.Adam([u], lr=0.05)
    for i in range(steps):
        opt.zero_grad()
        loss = -(q * torch.log_softmax(ln_m(u) @ E_m.T, -1)).sum(-1).mean()
        loss.backward()
        opt.step()
    u = u.detach()
    cc = torch.nn.functional.cosine_similarity(
        zlogits(u, ln_m, E_m)[-1][None], zlogits(v, ln_s, E_s)[-1][None]).item()
    print(f"    readout_match{tag}: loss {loss.item():.3f} logit_cos {cc:.4f}", flush=True)
    return u


_rm_cache = {}


def readout_match_cached(v):
    key = (v.shape[0], round(float(v.sum()), 3))
    if key not in _rm_cache:
        _rm_cache[key] = readout_match(v)
    return _rm_cache[key]


MAPS["readout_match"] = (readout_match_cached, "gradient match to Small's readout")

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
_cap_s, _cap_m0 = {}, {}
small.transformer.h[0].register_forward_pre_hook(
    lambda m, a, kw: _cap_s.__setitem__(
        "x", (a[0] if a else kw["hidden_states"]).detach()), with_kwargs=True)
medium.transformer.h[0].register_forward_pre_hook(
    lambda m, a, kw: _cap_m0.__setitem__(
        "x", (a[0] if a else kw["hidden_states"]).detach()), with_kwargs=True)


def natural_norm(model, cap, seq):
    with torch.no_grad():
        model(input_ids=torch.full((1, seq), 262))
    return cap["x"][0].norm(dim=1).mean().item()


NAT_M = natural_norm(medium, _cap_m0, 12)
NAT_S = natural_norm(small, _cap_s, 12)
print(f"natural layer-0 per-token norm: small {NAT_S:.2f}  medium {NAT_M:.2f}", flush=True)

_scaf = {}


def step(x, seq):
    if seq not in _scaf:
        _scaf[seq] = torch.full((1, seq), 262)
    _inj["x"] = x
    with torch.no_grad():
        medium.transformer(input_ids=_scaf[seq])   # skip the LM head, ~15% of flops
    _inj["x"] = None
    return _cap["x"][0].clone()


# ---------------------------------------------------------------- GATE
print("\n=== GATE: mapped state read out in MEDIUM's unembedding, no iteration ===",
      flush=True)
gate = {}
for pname, v in states.items():
    src_top, src_b, src_c = readout(v[-1], ln_s, E_s)
    zsrc = zlogits(v, ln_s, E_s)
    gate[pname] = {"source_small_readout": {
        "top12": src_top, "basin_median_rank": src_b, "control_median_rank": src_c,
        "per_token_norm": round(v.norm(dim=1).mean().item(), 1)}}
    print(f"\n[{pname}] small: {src_top[:8]}   basin_rank {src_b}")
    for mname, (fn, _) in MAPS.items():
        u = fn(v)
        top, b, c = readout(u[-1], ln_m, E_m)
        per_pos = [in_basin(readout(u[p], ln_m, E_m)[0]) for p in range(u.shape[0])]
        cc = torch.nn.functional.cosine_similarity(
            zlogits(u, ln_m, E_m)[-1][None], zsrc[-1][None]).item()
        gate[pname][mname] = {
            "top12": top, "basin_median_rank": b, "control_median_rank": c,
            "basin_in_top12": in_basin(top),
            "positions_with_basin_in_top12": sum(1 for x in per_pos if x > 0),
            "n_positions": u.shape[0],
            "readout_cos_to_small": round(cc, 4),
        }
        r = gate[pname][mname]
        print(f"   {mname:15s} basin_rank {b:>6}  ctrl {c:>6}"
              f"  hits {r['basin_in_top12']:>2}/12"
              f"  pos {r['positions_with_basin_in_top12']}/{r['n_positions']}"
              f"  logit_cos {cc:+.3f}   {top[:6]}", flush=True)

gate_pass = {}
for mname in MAPS:
    hits = [gate[p][mname]["basin_in_top12"] for p in SOC_PROMPTS]
    ranks = [gate[p][mname]["basin_median_rank"] for p in SOC_PROMPTS]
    gate_pass[mname] = {
        "basin_in_top12_per_prompt": hits, "basin_median_rank_per_prompt": ranks,
        "pass": all(h >= 3 for h in hits) and all(r <= 100 for r in ranks)}
print("\nGATE VERDICT:", {k: v["pass"] for k, v in gate_pass.items()}, flush=True)

results = {
    "natural_layer0_per_token_norm": {"small": NAT_S, "medium": NAT_M},
    "map_quality": map_quality, "gate": gate, "gate_verdict": gate_pass,
    "max_iter": MAX_ITER, "runs": {},
}
json.dump(results, open(OUT, "w"), indent=1)

# ---------------------------------------------------------------- LOOP
SOC_MEAN_NORM = sum(states[p].norm(dim=1).mean().item() for p in SOC_PROMPTS) / 4
SMALL_RATIO = SOC_MEAN_NORM / NAT_S
SHELLS = [("natural", round(NAT_M, 2)), ("conv_natural", 30.0),
          ("matched_ratio", round(SMALL_RATIO * NAT_M, 1)),
          ("matched_abs", round(SOC_MEAN_NORM, 1)),
          ("x73", 2200.0), ("x218", 6500.0)]
results["shells_per_token_norm"] = dict(SHELLS)
results["small_converged_energy_ratio_vs_own_natural"] = round(SMALL_RATIO, 1)
LOCK_TOL = 1e-6


def run_loop(u0, seq, max_iter=MAX_ITER):
    """Deterministic, so once the iterate locks every later state is identical;
    remaining checkpoints are filled and flagged. Runs that have lost the
    register and whose top-1 has been static for 40 passes are also cut."""
    x = u0.clone()
    N0 = x.norm().item()
    seed_flat = u0.reshape(-1)
    top, b, c = readout(x[-1], ln_m, E_m)
    trace = [{"iter": 0, "basin": b, "control": c, "top": top,
              "basin_in_top12": in_basin(top), "cos_to_seed": 1.0}]
    prev, locked_at, survived, cut, it = None, None, 0, None, 0
    top1_hist = []
    for it in range(1, max_iter + 1):
        x = x * (N0 / x.norm())
        x = step(x, seq)
        if prev is not None and locked_at is None:
            if 1 - torch.nn.functional.cosine_similarity(
                    x.reshape(1, -1), prev.reshape(1, -1)).item() < LOCK_TOL:
                locked_at = it
        prev = x.clone()
        top, b, c = readout(x[-1], ln_m, E_m)
        hits = in_basin(top)
        if hits >= 3:
            survived = it
        top1_hist.append(top[0])
        if it in CHECKPOINTS:
            trace.append({"iter": it, "basin": b, "control": c, "top": top,
                          "basin_in_top12": hits,
                          "cos_to_seed": round(torch.nn.functional.cosine_similarity(
                              x.reshape(1, -1), seed_flat.reshape(1, -1)).item(), 4)})
        if locked_at is not None and it >= 40:
            cut = "locked"
            break
        if (it >= 120 and hits == 0 and (survived == 0 or it - survived >= 60)
                and len(set(top1_hist[-40:])) == 1):
            cut = "readout_static"
            break
    cs = round(torch.nn.functional.cosine_similarity(
        x.reshape(1, -1), seed_flat.reshape(1, -1)).item(), 4)
    done = {t["iter"] for t in trace}
    for cp in CHECKPOINTS:
        if cp not in done and cp > max(done):
            trace.append({"iter": cp, "basin": b, "control": c, "top": top,
                          "basin_in_top12": in_basin(top), "cos_to_seed": cs,
                          "from_lock": cut})
    trace.sort(key=lambda t: t["iter"])
    return {"shell_total_norm": N0, "locked_at": locked_at, "stopped_at": it,
            "stop_reason": cut or "max_iter",
            "last_iter_with_basin_in_top12": survived, "final_top12": top,
            "final_basin_median_rank": b, "final_control_median_rank": c,
            "trace": trace}


PASSED = [m for m in MAPS if gate_pass[m]["pass"]]
# best-scoring gate-FAILED maps, run for the record and flagged as such
FLAGGED = ["procrustes", "logit_lstsq"]
MATRIX = ([("Lucier", m, s) for m in PASSED for s, _ in SHELLS]
          + [(p, m, s) for m in PASSED
             for p in ("Semantic", "Nonsense", "Imperative", "Syntactic")
             for s in ("matched_abs", "x218")]
          + [("Lucier", m, s) for m in FLAGGED if m not in PASSED
             for s in ("matched_abs", "x218")])

shell_of = dict(SHELLS)
print(f"\n=== LOOP: {len(MATRIX)} runs (passed maps: {PASSED or 'NONE'}), "
      f"max {MAX_ITER} iters ===", flush=True)
t0 = time.time()
for i, (pname, mname, sname) in enumerate(MATRIX):
    v = states[pname]
    u = MAPS[mname][0](v)
    u = u / u.norm(dim=1, keepdim=True) * shell_of[sname]
    r = run_loop(u, v.shape[0])
    r.update({"prompt": pname, "map": mname, "shell": sname,
              "shell_per_token_norm": shell_of[sname],
              "gate_passed": gate_pass[mname]["pass"]})
    results["runs"][f"{pname}|{mname}|{sname}"] = r
    print(f"[{i + 1}/{len(MATRIX)}] {pname}|{mname}|{sname} "
          f"gate={'PASS' if r['gate_passed'] else 'FAIL'} lock@{r['locked_at']} "
          f"stop={r['stop_reason']}@{r['stopped_at']} "
          f"survived_to={r['last_iter_with_basin_in_top12']} ({time.time() - t0:.0f}s)")
    for e in r["trace"]:
        print(f"     it{e['iter']:>3}  basin {e['basin']:>6}  ctrl {e['control']:>6}"
              f"  hits {e['basin_in_top12']:>2}   {e['top'][:6]}")
    sys.stdout.flush()
    json.dump(results, open(OUT, "w"), indent=1)

json.dump(results, open(OUT, "w"), indent=1)
print("\nwritten", OUT)
