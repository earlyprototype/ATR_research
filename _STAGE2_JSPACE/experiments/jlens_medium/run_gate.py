# P0-3 validation gate, Medium edition (issue #15, RUNBOOK_JLENS_MEDIUM.md §2.2).
#
# J-lens vs logit-lens readouts on gpt2-medium at layers spanning the stack —
# {2, 5, 8, 10, 12, 15, 18, 21}, chosen to include the ATR-map cells 8, 10,
# 15, 21 — at the last and second-to-last positions, on the RUNBOOK_PHASE0
# P0-3 multi-hop prompts plus 3 recorded picks from the restored
# prompt_library.py (lucier repo main @ 49592a7365c77dc63ad7eda0738e04880eac4837):
#   A03_neuro, B01_napoleon, C01_jack_jill (IDs recorded; chosen for having
#   humanly checkable expected continuations in three different registers).
#
# Logit lens here is lens.apply(..., use_jacobian=False): the identical
# residual, identical ln_f -> W_U decode, minus the J transport — the exact
# convention atr_engine.get_top_tokens implements (ln_final then unembed).
#
# Output: gate_results.json + gate_table.md (side-by-side top-5 per
# prompt x layer x position). The PASS/FAIL/MARGINAL verdict is written by
# hand into RESULTS_JLENS_MEDIUM.md after reading the table — a FAIL is
# evidence for EXP_012m and is not retried.

import json
import os

import jlens

from fit_lens import ARTIFACTS, load_model

LAYERS = [2, 5, 8, 10, 12, 15, 18, 21]
POSITIONS = [-2, -1]
TOPK = 5

GATE_PROMPTS = [
    ("runbook_boot", "Fact: The currency used in the country shaped like a boot is"),
    ("runbook_mars", "The color of the planet fourth from the sun is"),
    ("runbook_eiffel", "The capital of the country where the Eiffel Tower stands is"),
    ("A03_neuro", "The hippocampal formation plays a critical role in"),
    ("B01_napoleon", "Napoleon crossed the Alps with an army of"),
    ("C01_jack_jill", "Jack and Jill went up the hill to"),
]

HERE = os.path.dirname(os.path.abspath(__file__))


def top_tokens(tok, logits, k=TOPK):
    return [tok.decode([t]) for t in logits.topk(k).indices.tolist()]


def main():
    model, tok = load_model()
    lens_path = os.path.join(ARTIFACTS, "jlens_gpt2_medium_100.pt")
    lens = jlens.JacobianLens.load(lens_path)
    print(lens)

    results = {"lens_file": os.path.basename(lens_path), "prompts": []}
    lines = [
        "# P0-3 validation gate (Medium): J-lens vs logit-lens, side by side",
        "",
        f"Lens: `{os.path.basename(lens_path)}` ({lens.n_prompts} prompts). "
        f"Layers {LAYERS}, positions -2/-1, top-{TOPK}.",
        "",
    ]

    for pid, prompt in GATE_PROMPTS:
        jl, model_logits, input_ids = lens.apply(
            model, prompt, layers=LAYERS, positions=POSITIONS
        )
        ll, _, _ = lens.apply(
            model, prompt, layers=LAYERS, positions=POSITIONS, use_jacobian=False
        )
        toks = [tok.decode([t]) for t in input_ids[0].tolist()]
        rec = {"id": pid, "prompt": prompt, "n_tokens": len(toks), "layers": {}}
        lines += [f"## {pid}", "", f"`{prompt}`", ""]
        for pi, pos in enumerate(POSITIONS):
            lines += [
                f"**position {pos}** (token `{toks[pos]!r}`) — "
                f"model final top-5: {top_tokens(tok, model_logits[pi])}",
                "",
                "| layer | J-lens top-5 | logit-lens top-5 |",
                "|---|---|---|",
            ]
            for layer in LAYERS:
                jt = top_tokens(tok, jl[layer][pi])
                lt = top_tokens(tok, ll[layer][pi])
                rec["layers"].setdefault(str(layer), {})[str(pos)] = {
                    "jlens": jt,
                    "logit": lt,
                }
                lines.append(f"| {layer} | `{jt}` | `{lt}` |")
            lines.append("")
        rec["model_top5"] = {
            str(pos): top_tokens(tok, model_logits[pi])
            for pi, pos in enumerate(POSITIONS)
        }
        results["prompts"].append(rec)

    json.dump(results, open(os.path.join(HERE, "gate_results.json"), "w"), indent=1)
    open(os.path.join(HERE, "gate_table.md"), "w").write("\n".join(lines))
    print("wrote gate_results.json, gate_table.md")


if __name__ == "__main__":
    main()
