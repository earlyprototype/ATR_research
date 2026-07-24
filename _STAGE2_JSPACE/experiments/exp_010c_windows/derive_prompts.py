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


def select_subset(n=25):
    by_cat = {}
    for rec in load_all():
        by_cat.setdefault(rec["category"], []).append(rec)
    for cat in by_cat:
        by_cat[cat].sort(key=lambda r: r["id"])  # alphabetical by prompt ID
    cats = sorted(by_cat)
    picked, idx = [], 0
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
    return picked


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
