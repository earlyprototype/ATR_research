"""Derive the deterministic EXP_010c prompt subset for GPT-2 Medium.

`prompt_library.py` is absent from both repos (recorded deviation, spec §5).
Recovery source (READ-ONLY): the Stage 1 record of all 125 prompts with IDs,
categories, and full text, in the public ATR repo:
    experiments/gpt2_medium/output/dissolution_sentences.md
Entry format:
    ### A01_physics [Complex] -> `D` (predicted: `prolet`)
    *"The implications of quantum entanglement suggest that"*

Selection rule (spec §5): 25 prompts = round-robin across the 7 categories,
alphabetical by prompt ID within category. The recovered list is printed and
saved alongside the results so the pick is auditable.
"""

import json
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Override with ATR_STAGE1_REPO if the public Stage 1 repo lives elsewhere.
SOURCE = Path(os.environ.get(
    "ATR_STAGE1_REPO", "/home/user/lucier-gpt2-activ-tensor-reson-experiments"
)) / "experiments" / "gpt2_medium" / "output" / "dissolution_sentences.md"

ENTRY = re.compile(
    r"^### (?P<pid>\S+) \[(?P<cat>[^\]]+)\][^\n]*\n\*\"(?P<prompt>.*?)\"\*",
    re.MULTILINE | re.DOTALL,
)


def load_all():
    text = SOURCE.read_text()
    records = [
        {"id": m["pid"], "category": m["cat"], "prompt": m["prompt"]}
        for m in ENTRY.finditer(text)
    ]
    if len(records) != 125:
        raise SystemExit(
            f"Expected 125 prompts, parsed {len(records)} — schema drift in {SOURCE}; "
            "inspect and RECORD before adapting."
        )
    return records


def round_robin_order():
    """The full 125 prompts in the spec §5 order: round-robin across the 7
    categories, alphabetical by prompt ID within category. The registered
    subset is the first 25 of this ordering."""
    by_cat = {}
    for rec in load_all():
        by_cat.setdefault(rec["category"], []).append(rec)
    for cat in by_cat:
        by_cat[cat].sort(key=lambda r: r["id"])  # alphabetical by prompt ID
    cats = sorted(by_cat)
    order, idx = [], 0
    while True:
        progressed = False
        for cat in cats:
            if idx < len(by_cat[cat]):
                order.append(by_cat[cat][idx])
                progressed = True
        if not progressed:
            return order
        idx += 1


def select_subset(n=25, offset=0):
    """n prompts from the deterministic round-robin ordering, starting at
    `offset`.

    offset=0 reproduces the registered subset exactly (verified against the
    committed output/prompt_subset.json) — the default path is unchanged, so
    every registered run stays reproducible.

    offset is the disjoint-subset mechanism for EXP_010c-3b §2b: offset=25
    gives the next 25 by the same rule, with no overlap and no hand-picking.
    """
    return round_robin_order()[offset:offset + n]


def select_subset_b(n=25):
    """Disjoint subset B (EXP_010c-ROBUST spec §3): the NEXT n prompts under
    the identical round-robin/alphabetical ordering, i.e. positions 26..25+n —
    deterministic, no hand-picking, disjoint from the registered subset.

    MERGE NOTE: EXP_010c-ROBUST (#11) and EXP_010c-3b (#21) added this helper
    and `select_subset(n, offset=)` in parallel for the same need. They are the
    same selection — `select_subset_b(n) == select_subset(n, offset=25)` — so
    this now delegates rather than re-deriving, leaving ONE implementation.
    Both call sites are kept because each experiment's committed artifacts were
    produced through its own entry point. The disjointness assertion is
    retained: it is a cheap guard that would catch any future change to the
    ordering rule silently overlapping the registered subset.
    """
    subset_b = select_subset(n, offset=25)
    if len(subset_b) != n:
        raise SystemExit(f"Could not derive {n} prompts at offset 25 (got {len(subset_b)})")
    reg_ids = {r["id"] for r in select_subset(25)}
    overlap = reg_ids & {r["id"] for r in subset_b}
    if overlap:
        raise SystemExit(f"Subset B overlaps registered subset: {overlap}")
    return subset_b


if __name__ == "__main__":
    records = load_all()
    cats = sorted({r["category"] for r in records})
    print(f"Recovered {len(records)} prompts, {len(cats)} categories: {cats}")
    subset = select_subset()
    print(f"Selected {len(subset)} (round-robin, alphabetical by ID within category):")
    for i, rec in enumerate(subset):
        print(f"  {i:2d} {rec['id']:<16} [{rec['category']}] {rec['prompt']!r}")
    out = HERE / "output" / "prompt_subset.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(subset, indent=2))
    print(f"Saved -> {out}")
