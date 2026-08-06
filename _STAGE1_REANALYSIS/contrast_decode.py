"""What distinguishes Medium's Small-like clusters, in token space?

Established: Medium's tensors at W3_23/W5_23/W6_23 partition the 25 prompts
almost the way Small's socialist attractor does (21/25 agreement), while their
absolute decodes are ' +', ' as', ' dawn', '_' — nothing socialist.

The absolute readout is dominated by whatever funnel the arm sits in. The
information that separates the clusters is a CONTRAST, not an absolute state.
This decodes the contrast: logits(socialist-cluster centroid) minus
logits(rest centroid), which is the standard logit-difference attribution and
is well defined despite ln_f being nonlinear.

If Medium represents the distinction at all, this is where it shows.
"""
import json, statistics, sys, collections
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

SMALL, MED, OUT = sys.argv[1], sys.argv[2], sys.argv[3]

sm = GPT2LMHeadModel.from_pretrained(SMALL).eval()
md = GPT2LMHeadModel.from_pretrained(MED).eval()
tok = GPT2TokenizerFast.from_pretrained(MED)
s_ln, s_WE = sm.transformer.ln_f, sm.transformer.wte.weight
m_ln, m_WE = md.transformer.ln_f, md.transformer.wte.weight

# ---- 1. recover Small's socialist prompt set from its own terminals ----
st = torch.load("/home/user/ATR_research/_STAGE2_JSPACE/experiments/exp_010c_windows/"
                "output/terminals_small_010d.pt", map_location="cpu", weights_only=False)
SOC_TOKENS = {" prolet", " Anarch", " bourgeois", " Marx", " comrade", " proletarian",
              " socialist", " anarchist", " Lenin", " anarchism", " comrades", " labour"}
def parts(k):
    """terminal .pt keys are ('ARM','PROMPT') tuples in older files and
    'ARM|PROMPT' strings after the PR #4 weights_only change. accept both."""
    if isinstance(k, (tuple, list)):
        p = [str(x) for x in k]
    else:
        p = str(k).split("|")
    return (p[0], p[-1]) if len(p) > 1 else (p[0], p[0])


small_decode = {}
for k, v in st.items():
    x = v["mean"] if isinstance(v, dict) and "mean" in v else v
    if x.dim() > 1:
        x = x.mean(0)
    with torch.no_grad():
        t = tok.decode([int((s_ln(x) @ s_WE.T).argmax())])
    small_decode[parts(k)[1]] = t
soc_prompts = {p for p, t in small_decode.items() if t in SOC_TOKENS}
print(f"Small decode distribution: {collections.Counter(small_decode.values()).most_common()}")
print(f"socialist prompts: {len(soc_prompts)}/{len(small_decode)}")

# ---- 2. contrast-decode each Medium arm ----
def arm_path(a):
    if a == "A0":
        return "/home/user/ATR_research/_STAGE2_JSPACE/experiments/exp_010c_windows/output/terminals_full.pt"
    return f"/home/user/census_t/_STAGE2_JSPACE/experiments/exp_010c_windows/output/terminals_census/{a}.pt"

results = {"small_socialist_prompts": sorted(soc_prompts), "arms": {}}
for arm in ["W3_23", "W5_23", "W6_23", "A0"]:
    try:
        t = torch.load(arm_path(arm), map_location="cpu", weights_only=False)
    except FileNotFoundError:
        print(f"{arm}: tensors not found, skipping"); continue
    soc, rest = [], []
    for k, v in t.items():
        a, p = parts(k)
        if arm == "A0" and a != "A0":
            continue
        x = v["mean"] if isinstance(v, dict) and "mean" in v else v
        if x.dim() > 1:
            x = x.mean(0)
        (soc if p in soc_prompts else rest).append(x)
    if len(soc) < 3 or len(rest) < 3:
        print(f"{arm}: split too small ({len(soc)}/{len(rest)}), skipping"); continue
    cs, cr = torch.stack(soc).mean(0), torch.stack(rest).mean(0)
    with torch.no_grad():
        d = (m_ln(cs) @ m_WE.T) - (m_ln(cr) @ m_WE.T)
    hi = torch.topk(d, 25).indices.tolist()
    lo = torch.topk(-d, 15).indices.tolist()
    soc_ids = [tok.encode(w)[0] for w in SOC_TOKENS if len(tok.encode(w)) == 1]
    order = torch.argsort(d, descending=True)
    pos = torch.empty_like(order); pos[order] = torch.arange(len(order))
    ranks = sorted(int(pos[i]) + 1 for i in soc_ids)
    med = statistics.median(ranks)
    results["arms"][arm] = {
        "n_soc": len(soc), "n_rest": len(rest),
        "toward_socialist_cluster": [tok.decode([i]) for i in hi],
        "toward_rest": [tok.decode([i]) for i in lo],
        "socialist_token_median_rank_in_contrast": med,
        "socialist_token_ranks": ranks,
    }
    print(f"\n--- {arm}  ({len(soc)} socialist-side / {len(rest)} rest)")
    print(f"    toward socialist cluster: {[tok.decode([i]) for i in hi[:16]]}")
    print(f"    toward rest             : {[tok.decode([i]) for i in lo[:10]]}")
    print(f"    socialist tokens median rank in contrast: {med} of 50257")

json.dump(results, open(OUT, "w"), indent=1)
print("\nwritten", OUT)
