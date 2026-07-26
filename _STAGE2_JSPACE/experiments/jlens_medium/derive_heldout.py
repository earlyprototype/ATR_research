# EXP_012m held-out prompt derivation (issue #15).
#
# Rule (deterministic, recorded before any census run): the 50 prompts at
# round-robin positions 51..100 under the EXACT ordering machinery already
# registered by EXP_010c — derive_prompts.select_subset(100)[50:100] —
# i.e. the next 50 prompts after the registered ATR subset (positions 1..25,
# prompt_subset.json) and subset B (positions 26..50, prompt_subset_b.json).
# Disjointness:
#   - from the ATR prompt subsets: by construction (positions 51..100);
#     asserted against the committed prompt_subset{,_b}.json below.
#   - from the fitting set: the lens is fitted on WikiText-103 records
#     (artifacts/wikitext_prompts_160.json); the prompt library is the Stage 1
#     125-prompt set — no overlap possible (asserted by literal comparison).
#
# Source library: the lucier repo (pulled main @ recorded commit), via
# derive_prompts.load_all() which parses the committed Stage 1 record
# dissolution_sentences.md (byte-identical to the restored prompt_library.py
# per that file's provenance header).
#
# Usage: ATR_STAGE1_REPO=/home/user/lucier-repo python3 derive_heldout.py

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP010C = os.path.join(HERE, "..", "exp_010c_windows")
sys.path.insert(0, EXP010C)

from derive_prompts import select_subset  # noqa: E402


def main():
    picked = select_subset(100)
    if len(picked) != 100:
        raise SystemExit(f"expected 100, got {len(picked)}")
    heldout = picked[50:]

    # Disjointness vs the two committed ATR subsets
    for name in ("prompt_subset.json", "prompt_subset_b.json"):
        path = os.path.join(EXP010C, "output", name)
        committed = json.load(open(path))
        committed_ids = {r["id"] for r in committed}
        derived_ids = {r["id"] for r in heldout}
        overlap = committed_ids & derived_ids
        if overlap:
            raise SystemExit(f"held-out overlaps {name}: {sorted(overlap)}")
        print(f"disjoint from {name}: OK ({len(committed_ids)} ids)")

    # Disjointness vs the fitting set (literal prompt-text comparison)
    wikitext = json.load(
        open(os.path.join(HERE, "..", "..", "artifacts", "wikitext_prompts_160.json"))
    )
    fit_texts = set(wikitext[:100])
    overlap_texts = [r["id"] for r in heldout if r["prompt"] in fit_texts]
    if overlap_texts:
        raise SystemExit(f"held-out overlaps fitting set: {overlap_texts}")
    print("disjoint from fitting set (100 WikiText prompts): OK")

    out = os.path.join(HERE, "heldout_50.json")
    json.dump(heldout, open(out, "w"), indent=1)
    print(f"wrote {len(heldout)} held-out prompts -> {out}")
    print("ids:", " ".join(r["id"] for r in heldout))


if __name__ == "__main__":
    main()
