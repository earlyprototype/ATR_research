"""Regression guard for the scientifically load-bearing subset invariants.

Subset B must be exactly n prompts and fully disjoint from the registered
subset (EXP_010c_ROBUST_SPEC.md); this pins select_subset's round-robin
semantics against future edits.
"""
from derive_prompts import select_subset, select_subset_b


def test_subset_b_size_and_disjointness():
    a = select_subset(25)
    b = select_subset_b(25)
    ids_a = {r["id"] for r in a}
    ids_b = {r["id"] for r in b}
    assert len(b) == 25, f"subset B has {len(b)} prompts, expected 25"
    assert len(ids_b) == 25, "subset B contains duplicate prompt ids"
    assert not (ids_a & ids_b), f"overlap with registered subset: {ids_a & ids_b}"


if __name__ == "__main__":
    test_subset_b_size_and_disjointness()
    print("OK: subset B size/uniqueness/disjointness invariants hold")
