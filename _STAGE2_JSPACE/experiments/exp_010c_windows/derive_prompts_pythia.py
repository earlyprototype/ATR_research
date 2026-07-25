"""Derive the deterministic EXP_012-PYTHIA prompt subset (25 prompts).

Rule (RUNBOOK_PHASE1.md §EXP_010a step 1, executed for EXP_012_PYTHIA_SPEC.md):
    core 8  = the Stage 1 Pythia-410m deep-run prompts, read from the committed
              backup `_DATA/EXP009_stage1_trajectories_2026-07-10.zip` →
              `experiments/pythia_410m/output_deep/deep_config.pt`
              (`prompt_keys`; deep_results.pt carries the same 8 as its keys):
              A01_physics, B01_napoleon, C01_jack_jill, D01_water,
              E01_politics, F01_anger, G01_punctuation, G13_buffalo
    plus 17 = round-robin across the 7 categories of the restored
              `prompt_library.py` (public lucier repo, provenance-flagged
              reconstruction, all-original), categories in alphabetical order,
              alphabetical by prompt ID within category, EXCLUDING the core 8.

Prompt texts come from `prompt_library.py` (PROMPT_LIBRARY / CATEGORY_MAP).
Output records use the same {id, category, prompt} schema as derive_prompts.py
so run_exp010c.py consumes them unchanged. The pick is saved alongside the
results (output/prompt_subset_pythia.json) so it is auditable.
"""

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Override with ATR_STAGE1_REPO if the public Stage 1 repo lives elsewhere.
STAGE1_REPO = Path(os.environ.get(
    "ATR_STAGE1_REPO", "/home/user/lucier-gpt2-activ-tensor-reson-experiments"
))

# The Stage 1 deep-run core, in deep_config.pt order (recorded verbatim; the
# .pt lives in a committed zip, so the derivation does not re-read it at
# runtime — see module docstring for the extraction record).
DEEP_RUN_8 = [
    "A01_physics", "B01_napoleon", "C01_jack_jill", "D01_water",
    "E01_politics", "F01_anger", "G01_punctuation", "G13_buffalo",
]


def _library():
    sys.path.insert(0, str(STAGE1_REPO))
    import prompt_library
    return prompt_library.PROMPT_LIBRARY, prompt_library.CATEGORY_MAP


def select_subset_pythia(n=25):
    lib, cat_map = _library()
    missing = [k for k in DEEP_RUN_8 if k not in lib]
    if missing:
        raise SystemExit(f"Deep-run prompts missing from prompt_library: {missing}")
    picked = [{"id": k, "category": cat_map[k], "prompt": lib[k]} for k in DEEP_RUN_8]

    by_cat = {}
    for pid, text in lib.items():
        if pid in DEEP_RUN_8:
            continue
        by_cat.setdefault(cat_map[pid], []).append(
            {"id": pid, "category": cat_map[pid], "prompt": text})
    for cat in by_cat:
        by_cat[cat].sort(key=lambda r: r["id"])  # alphabetical by prompt ID
    cats = sorted(by_cat)  # alphabetical category order
    idx = 0
    while len(picked) < n:
        progressed = False
        for cat in cats:
            if len(picked) >= n:
                break
            if idx < len(by_cat[cat]):
                picked.append(by_cat[cat][idx])
                progressed = True
        if not progressed:
            break
        idx += 1
    if len(picked) != n:
        raise SystemExit(f"Could not derive {n} prompts (got {len(picked)})")
    return picked


if __name__ == "__main__":
    subset = select_subset_pythia()
    print(f"Selected {len(subset)} (core 8 deep-run + round-robin 17):")
    for i, rec in enumerate(subset):
        tag = "core8" if rec["id"] in DEEP_RUN_8 else "rr17"
        print(f"  {i:2d} {rec['id']:<18} [{rec['category']:<9}] ({tag}) {rec['prompt']!r}")
    out = HERE / "output" / "prompt_subset_pythia.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(subset, indent=2))
    print(f"Saved -> {out}")
