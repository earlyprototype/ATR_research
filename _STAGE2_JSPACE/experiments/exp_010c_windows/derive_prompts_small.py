"""Derive the deterministic EXP_010b prompt subset (25 prompts, GPT-2 Small).

Rule (RUNBOOK_PHASE1.md §EXP_010b step 1, executed per EXP_010b_SPEC.md §3):
    5 Divine  = the alphabetically-first 5 prompt IDs among the 34 Stage 1
                `Divine` (non-converged) prompts, derived from the committed
                backup `_DATA/EXP009_stage1_trajectories_2026-07-10.zip` →
                `experiments/gpt2_small/output_gated/gated_results.pt`
                (canonical per issue #16's second comment; the 34 are exactly
                the records with converged=False — schema recorded in the spec).
    plus 20   = round-robin across the 7 categories of the restored
                `prompt_library.py` (public lucier repo), categories in
                alphabetical order, alphabetical by prompt ID within category,
                EXCLUDING the 5 Divine picks. The literal exclusion rule means
                round-robin picks may themselves be Stage 1 Divine; recorded
                in the spec (6 are).

Output records use the {id, category, prompt} schema of derive_prompts.py so
run_exp010c.py consumes them unchanged. The committed
output/prompt_subset_small.json is the execution authority: when present, the
fresh derivation must match it exactly or this module hard-stops.
"""

import io
import json
import os
import sys
import zipfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent.parent  # _STAGE2_JSPACE/experiments/exp_010c_windows -> repo
ZIP_PATH = REPO_ROOT / "_DATA" / "EXP009_stage1_trajectories_2026-07-10.zip"
GATED_MEMBER = "experiments/gpt2_small/output_gated/gated_results.pt"

# Location of the public Stage 1 repo (for prompt_library.py). REQUIRED — no
# developer-specific default (mirrors derive_prompts_pythia.py).
_STAGE1_ENV = os.environ.get("ATR_STAGE1_REPO")

EXPECTED_CATEGORIES = {
    "Complex", "Narrative", "Simple", "Chemical", "Acronyms", "Vulgarity", "Wild",
}

COMMITTED_SUBSET = HERE / "output" / "prompt_subset_small.json"


def load_gated_results():
    """Load Stage 1's gated_results.pt straight out of the committed zip.

    Read-only: the zip member is decompressed in memory (the _frozen/
    convention — extracted copies are never committed). Loads with
    weights_only=True (verified; schema recorded in EXP_010b_SPEC.md §3).
    """
    with zipfile.ZipFile(ZIP_PATH) as zf:
        blob = zf.read(GATED_MEMBER)
    gated = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
    if len(gated) != 125:
        raise SystemExit(
            f"Expected 125 gated records, got {len(gated)} — schema drift in "
            f"{ZIP_PATH}:{GATED_MEMBER}; inspect and RECORD before adapting.")
    return gated


def divine_ids(gated):
    """The 34 Stage 1 Divine prompt IDs = exactly the non-converged records."""
    div = sorted(k for k, v in gated.items() if not v["converged"])
    if len(div) != 34:
        raise SystemExit(
            f"Expected 34 non-converged (Divine) prompts, got {len(div)} — "
            "disagrees with the recorded Stage 1 cohort; inspect and RECORD.")
    bad = [k for k in div if gated[k]["terminal_token"].strip() != "Divine"]
    if bad:
        raise SystemExit(
            f"Non-converged prompts without Divine decode: {bad} — breaks the "
            "recorded converged=False ≡ Divine identity; inspect and RECORD.")
    return div


def _library():
    if not _STAGE1_ENV:
        raise SystemExit(
            "ATR_STAGE1_REPO is not set. Point it at a clone of the public "
            "lucier Stage 1 repo (the directory containing prompt_library.py) "
            "before deriving the EXP_010b subset.")
    repo = Path(_STAGE1_ENV)
    if not (repo / "prompt_library.py").exists():
        raise SystemExit(f"prompt_library.py not found in ATR_STAGE1_REPO={repo}")
    sys.path.insert(0, str(repo))
    import prompt_library
    lib, cat_map = prompt_library.PROMPT_LIBRARY, prompt_library.CATEGORY_MAP
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


def _derive_subset(n=25):
    gated = load_gated_results()
    lib, cat_map = _library()
    if set(gated) != set(lib):
        raise SystemExit(
            "Prompt-ID sets differ between gated_results.pt and "
            "prompt_library.py — inspect and RECORD before adapting.")
    picks5 = divine_ids(gated)[:5]
    picked = [{"id": k, "category": cat_map[k], "prompt": lib[k]} for k in picks5]

    by_cat = {}
    for pid, text in lib.items():
        if pid in picks5:
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


def select_subset_small(n=25):
    """Derive the subset AND, when the committed record exists, validate the
    fresh derivation against it (full-record compare — the committed
    prompt_subset_small.json is the execution authority; any mismatch is a
    hard stop, not a silent re-derivation)."""
    derived = _derive_subset(25)
    if COMMITTED_SUBSET.exists():
        committed = json.loads(COMMITTED_SUBSET.read_text())
        if committed != derived:
            raise SystemExit(
                "Committed prompt_subset_small.json disagrees with the fresh "
                "derivation (full-record compare) — inspect and RECORD before "
                "running anything.")
        return committed[:n]
    return derived[:n]


def stage1_expected(subset=None):
    """Per-prompt Stage 1 (converged, terminal_token) for the reproduction
    gate (EXP_010b_SPEC.md §4). Pure read of the committed zip."""
    gated = load_gated_results()
    subset = subset if subset is not None else select_subset_small(25)
    return {
        rec["id"]: {
            "converged": bool(gated[rec["id"]]["converged"]),
            "terminal_token": gated[rec["id"]]["terminal_token"],
            "lock_in_iter": gated[rec["id"]]["lock_in_iter"],
        }
        for rec in subset
    }


if __name__ == "__main__":
    subset = select_subset_small()  # validates against the committed record when present
    gated = load_gated_results()
    div34 = set(divine_ids(gated))
    picks5 = set(divine_ids(gated)[:5])
    print("Selected 25 (5 Divine picks + round-robin 20):")
    for i, rec in enumerate(subset):
        tag = ("DIVINE-PICK" if rec["id"] in picks5
               else "also-Divine" if rec["id"] in div34 else "")
        print(f"  {i:2d} {rec['id']:<18} [{rec['category']:<9}] "
              f"stage1={gated[rec['id']]['terminal_token']!r:14} {tag}")
    n_div = sum(1 for r in subset if r["id"] in div34)
    print(f"Divine in subset: {n_div}/25 (5 picks + {n_div - 5} via round-robin)")
    out = COMMITTED_SUBSET
    if not out.exists():
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(subset, indent=2))
        print(f"Saved -> {out}")
    else:
        print(f"Validated -> {out} (committed record left untouched)")
