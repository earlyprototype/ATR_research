"""Non-registered harness check for run_jspace.py. No verdict weight.

Runs the whole J-space path on one layer only, with the five-prompt probe lens
standing in for the twin lens, so that a coding error surfaces in minutes
rather than after the real fit finishes. The registered run does not read what
this writes and does not overwrite it.

Where it writes, and why it never overwrites. The output suffix may be given as
the one argument; with no argument it is `_harness_<today's date>`, so a check
run today lands in a file of its own. Either way an existing destination is
refused rather than replaced, because two of the harness files under output/ are
historical records that the results record describes:
`exp017_jspace_harness.json` from 2026-09-05, whose recorded and actual
permutation counts differ for the reason deviation 16 gives, and
`exp017_jspace_harness_p200.json` from 2026-09-06, which is the re-run that
shows the difference. Overwriting either would erase evidence rather than add
to it.

Usage:
    python3 harness_check_jspace.py              # writes ..._harness_YYYYMMDD.json
    python3 harness_check_jspace.py _harness_p200
"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_jspace

suffix = sys.argv[1] if len(sys.argv) > 1 else time.strftime("_harness_%Y%m%d")
destination = HERE / "output" / f"exp017_jspace{suffix}.json"
if destination.exists():
    raise SystemExit(
        f"refusing to overwrite {destination.name}, which already exists. The "
        f"harness files under output/ are dated records of what ran on the day, "
        f"so give this run a suffix of its own, for example "
        f"{time.strftime('_harness_%Y%m%d')}_b.")

run_jspace.PROBE_LAYERS = [0, 10]
run_jspace.BAND = [10]
# 200 reassignments instead of 10,000, because this check exists to surface
# coding errors cheaply. run_jspace reads this when the test runs rather than
# when the module is imported, so the number set here is the number used; until
# 2026-09-05 it was bound as a default argument at import and this line had no
# effect, so the committed harness artifact ran 10,000 while recording 200.
run_jspace.N_PERM = 200
# The five-prompt probe lens is deliberately shorter than the count the budget
# rule chose, so this check opts in to scoring it and its output is stamped as
# a sensitivity reading rather than the registered comparison. It also predates
# the provenance stamp, so it is admitted the same way the committed twin lens
# is, explicitly.
sys.argv = ["harness", "--out-suffix", suffix, "--allow-short-twin-lens",
            "--accept-unstamped-twin-lens",
            "--twin-lens", str(HERE.parent.parent / "artifacts"
                               / "jlens_lamini_gpt2_124m_5_probe.pt")]
run_jspace.main()
