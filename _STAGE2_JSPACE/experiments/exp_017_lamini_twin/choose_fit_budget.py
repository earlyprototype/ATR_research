"""EXP_017: apply the spec's lens-fit budget rule to the timing probe.

Spec section 6.2 fixes the rule before the probe's number is read: let t be the
measured seconds per prompt from the five-prompt probe; choose 100 prompts if
100t is at most 9000 seconds, otherwise 50 if 50t is at most 9000, otherwise
the largest multiple of 10 whose projected time is at most 9000. A count below
50 is a recorded deviation.

Writes output/fit_budget_decision.json and prints the chosen count.
"""
import json
import re
import sys
from pathlib import Path

CAP = 9000.0
HERE = Path(__file__).resolve().parent

log = (HERE / "output" / "fit_probe_db16.log").read_text()
times = [float(x) for x in re.findall(r"prompt \d+/\d+\s+seq_len=\d+\s+n_valid=\d+\s+(\d+)s", log)]
if not times:
    print("no per-prompt timings in the probe log", file=sys.stderr)
    raise SystemExit(1)
t = sum(times) / len(times)

if 100 * t <= CAP:
    n, why = 100, "100 prompts fit inside the 9000 second cap"
elif 50 * t <= CAP:
    n, why = 50, "100 prompts exceed the cap, 50 fit inside it"
else:
    n = max(10, (int(CAP // t) // 10) * 10)
    why = ("neither 100 nor 50 prompts fit inside the cap, so the largest "
           "multiple of ten that does was taken")

rec = {"per_prompt_seconds": times, "n_probe_prompts": len(times),
       "mean_seconds_per_prompt": round(t, 1), "cap_seconds": CAP,
       "chosen_n_prompts": n, "projected_seconds": round(n * t),
       "rule": why,
       "below_50_is_a_recorded_deviation": n < 50,
       "dim_batch": 16,
       "dim_batch_reason": ("peak resident memory measured at about 2.5 "
                            "gigabytes at dim_batch 16, and the spec caps peak "
                            "memory at 3 gigabytes, so it was not raised")}
(HERE / "output" / "fit_budget_decision.json").write_text(json.dumps(rec, indent=2))
print(json.dumps(rec, indent=2))
print(n)
