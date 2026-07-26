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
# Location of the public Stage 1 repo (for prompt_library.py). REQUIRED — no
# developer-specific default (PR #39 review): fail closed with a clear message.
_STAGE1_ENV = os.environ.get("ATR_STAGE1_REPO")

# The Stage 1 deep-run core, in deep_config.pt order (recorded verbatim; the
# .pt lives in a committed zip, so the derivation does not re-read it at
# runtime — see module docstring for the extraction record).
DEEP_RUN_8 = [
    "A01_physics", "B01_napoleon", "C01_jack_jill", "D01_water",
    "E01_politics", "F01_anger", "G01_punctuation", "G13_buffalo",
]

EXPECTED_CATEGORIES = {
    "Complex", "Narrative", "Simple", "Chemical", "Acronyms", "Vulgarity", "Wild",
}


def _library():
    if not _STAGE1_ENV:
        raise SystemExit(
            "ATR_STAGE1_REPO is not set. Point it at a clone of the public "
            "lucier Stage 1 repo (the directory containing prompt_library.py) "
            "before deriving the EXP_012-PYTHIA subset."
        )
    repo = Path(_STAGE1_ENV)
    if not (repo / "prompt_library.py").exists():
        raise SystemExit(f"prompt_library.py not found in ATR_STAGE1_REPO={repo}")
    sys.path.insert(0, str(repo))
    import prompt_library
    lib, cat_map = prompt_library.PROMPT_LIBRARY, prompt_library.CATEGORY_MAP
    # Fail closed on schema drift (mirror of derive_prompts.py's 125-record
    # validation; PR #39 review): inspect and RECORD before adapting.
    if len(lib) != 125:
        raise SystemExit(
            f"Expected 125 prompts, got {len(lib)} — schema drift in "
            "prompt_library.py; inspect and RECORD before adapting.")
    missing_cat = [k for k in lib if k not in cat_map]
    if missing_cat:
        raise SystemExit(
            f"CATEGORY_MAP does not cover {len(missing_cat)} prompt IDs "
            f"(e.g. {missing_cat[:5]}) — schema drift; inspect and RECORD.")
    cats = set(cat_map.values())
    if cats != EXPECTED_CATEGORIES:
        raise SystemExit(
            f"Category set drifted: {sorted(cats)} != {sorted(EXPECTED_CATEGORIES)} "
            "— inspect and RECORD before adapting.")
    return lib, cat_map


COMMITTED_SUBSET = HERE / "output" / "prompt_subset_pythia.json"


def select_subset_pythia(n=25):
    """Derive the subset AND, when the committed record exists, validate the
    fresh derivation against it (PR #39 review: the committed
    prompt_subset_pythia.json is the execution authority — any mismatch is a
    hard stop, not a silent re-derivation)."""
    derived = _derive_subset(25)
    if COMMITTED_SUBSET.exists():
        committed = json.loads(COMMITTED_SUBSET.read_text())
        # Full-record comparison (PR #39 review round 2): a drifted category
        # field must fail just as loudly as a drifted id or prompt text.
        if committed != derived:
            raise SystemExit(
                "Committed prompt_subset_pythia.json disagrees with the fresh "
                "derivation (full-record compare) — inspect and RECORD before "
                "running anything.")
        return committed[:n]
    return derived[:n]


def _derive_subset(n=25):
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
    subset = _derive_subset()
    print(f"Selected {len(subset)} (core 8 deep-run + round-robin 17):")
    for i, rec in enumerate(subset):
        tag = "core8" if rec["id"] in DEEP_RUN_8 else "rr17"
        print(f"  {i:2d} {rec['id']:<18} [{rec['category']:<9}] ({tag}) {rec['prompt']!r}")
    out = HERE / "output" / "prompt_subset_pythia.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(subset, indent=2))
    print(f"Saved -> {out}")
