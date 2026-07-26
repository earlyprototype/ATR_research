"""EXP_010b reproduction gate — mechanical check (EXP_010b_SPEC.md §4).

Compares the SB (0->11 full-stack) arm's per-prompt (converged, terminal_token)
against the Stage 1 gated record derived from the committed _DATA zip.
PASS requires exact per-prompt match on all 25; any mismatch is FAIL (STOP and
document per RUNBOOK_PHASE1 §EXP_010b step 2). Pure analysis, no model time.

Usage: ATR_STAGE1_REPO=<lucier repo> python3 check_reproduction_gate.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS = HERE.parent / "exp_010c_windows"
sys.path.insert(0, str(HARNESS))

from derive_prompts_small import select_subset_small, stage1_expected  # noqa: E402


def main():
    results = json.loads((HARNESS / "output" / "results_small010b_SB.json").read_text())
    sb = [r for r in results if r["arm"] == "SB"]
    if len(sb) != 25:
        raise SystemExit(f"Expected 25 SB records, got {len(sb)}")
    sb_by_id = {r["prompt_id"]: r for r in sb}
    subset = select_subset_small(25)
    expected = stage1_expected(subset)
    if len(sb_by_id) != len(sb) or set(sb_by_id) != set(expected):
        raise SystemExit(
            "SB prompt IDs are not the exact registered 25-prompt subset: "
            f"got={sorted(sb_by_id)}, expected={sorted(expected)}")

    mismatches = []
    print(f"{'prompt':<18} {'stage1':<22} {'SB (this run)':<22} match")
    for rec in subset:
        r = sb_by_id[rec["id"]]
        e = expected[r["prompt_id"]]
        got = (bool(r["converged"]), r["terminal_token"])
        want = (e["converged"], e["terminal_token"])
        ok = got == want
        if not ok:
            mismatches.append(r["prompt_id"])
        print(f"{r['prompt_id']:<18} conv={want[0]!s:<5} {want[1]!r:<14} "
              f"conv={got[0]!s:<5} {got[1]!r:<14} {'OK' if ok else 'MISMATCH'}")

    locks = sorted({r["lock_in_iter"] for r in sb if r["converged"]})
    print(f"\nlock_in_iter values among converged: {locks} "
          f"(Stage 1: all 120; informational, not part of the gate)")
    if mismatches:
        print(f"\nREPRODUCTION GATE: FAIL — {len(mismatches)} mismatches: {mismatches}")
        print("STOP and document (spec §4). No sub-stack arm may run.")
        sys.exit(1)
    print("\nREPRODUCTION GATE: PASS — 25/25 exact (converged, terminal_token) match.")


if __name__ == "__main__":
    main()
