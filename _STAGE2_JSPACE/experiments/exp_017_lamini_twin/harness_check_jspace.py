"""Non-registered harness check for run_jspace.py. No verdict weight.

Runs the whole J-space path on one layer only, with the five-prompt probe lens
standing in for the twin lens, so that a coding error surfaces in minutes
rather than after the real fit finishes. Writes output/exp017_jspace_harness.json,
which the registered run does not read and does not overwrite.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_jspace

run_jspace.PROBE_LAYERS = [0, 10]
run_jspace.BAND = [10]
run_jspace.N_PERM = 200
# The five-prompt probe lens is deliberately shorter than the count the budget
# rule chose, so this check opts in to scoring it and its output is stamped as
# a sensitivity reading rather than the registered comparison.
sys.argv = ["harness", "--out-suffix", "_harness", "--allow-short-twin-lens",
            "--twin-lens", str(HERE.parent.parent / "artifacts"
                               / "jlens_lamini_gpt2_124m_5_probe.pt")]
run_jspace.main()
