"""EXP_017: apply the spec's lens-fit budget rule to the timing probe.

Spec section 6.2 fixes the rule before the probe's number is read: let t be the
measured seconds per prompt from the five-prompt probe; choose 100 prompts if
100t is at most 9000 seconds, otherwise 50 if 50t is at most 9000, otherwise
the largest multiple of 10 whose projected time is at most 9000. A count below
50 is a recorded deviation.

Above 900 seconds a prompt the rule has nothing to choose, because not even ten
prompts fit inside the cap. That case writes a chosen count of zero and says so
in the decision's own `rule` field, which run_jspace.py reads as the spec's
section 6.2 fallback: no twin lens can be the registered instrument, so H18b is
scored with the base lens on both sides. The committed decision is not that
case: the probe measured 221.0 seconds a prompt and the rule chose 40.

**Which timings it reads, and why that has to be said out loud.** Until
2026-09-06 this script read one hardcoded path, the run log committed on
2026-09-05, so a probe re-run on another machine printed its timings to the
terminal and this rule went on answering with the first machine's numbers and
choosing a prompt count for hardware it had never seen. The source is now named
on the command line and there is no default, so nothing can be read by accident:

    --timings PATH    a timings file written by fit_twin_lens.py, which every
                      fit now writes under output/ with the machine that
                      measured it and the day it did. This is the route for any
                      new probe.
    --probe-log PATH  a run log in the format the 2026-09-05 probe left, which
                      is how the committed decision was made and the only way to
                      reproduce it.

A timings file that does not name its machine and the day it was measured is
refused, because that is the whole point of preferring one. A run log cannot
carry either, so passing one is taken as a deliberate choice to use a recorded
measurement rather than this machine's.

Writes output/fit_budget_decision.json, and beside it a provenance sidecar
naming the source, its SHA-256 digest and, when the source carries them, the
machine and the date. The decision file's own fields are unchanged from the
committed one so that the committed decision still reproduces byte for byte.

Usage:
    python3 choose_fit_budget.py --probe-log output/fit_probe_db16.log
    python3 choose_fit_budget.py --timings output/fit_timings_probe_5_20260906.json
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

CAP = 9000.0
HERE = Path(__file__).resolve().parent


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def timings_from_log(path):
    """The per-prompt seconds a probe run log records, in prompt order."""
    text = Path(path).read_text()
    return [float(x) for x in re.findall(
        r"prompt \d+/\d+\s+seq_len=\d+\s+n_valid=\d+\s+(\d+)s", text)]


def timings_from_file(path):
    """The per-prompt seconds a timings file records, with the machine and the
    day it was measured. A file missing either is refused: a timings file exists
    precisely so that the rule knows whose clock it is reading."""
    record = json.load(open(path))
    machine = (record.get("machine") or {}).get("node")
    when = record.get("measured_utc")
    if not machine or not when:
        raise SystemExit(
            f"{path} does not say which machine measured it or when: machine "
            f"{machine!r}, measured {when!r}. A timings file without both is no "
            f"better than a log from somewhere else; re-run the probe with the "
            f"current fit_twin_lens.py, which records both.")
    seconds = [float(x) for x in record.get("per_prompt_seconds") or []]
    return seconds, machine, when


def choose(t, cap=CAP):
    """The spec's rule: the prompt count, and the sentence saying why."""
    if 100 * t <= cap:
        return 100, "100 prompts fit inside the 9000 second cap"
    if 50 * t <= cap:
        return 50, "100 prompts exceed the cap, 50 fit inside it"
    n = (int(cap // t) // 10) * 10
    if n >= 10:
        return n, ("neither 100 nor 50 prompts fit inside the cap, so the "
                   "largest multiple of ten that does was taken")
    # Above 900 seconds a prompt, even ten prompts cost more than the cap
    # allows, so there is no count for the rule to choose. Recording ten here
    # and calling it a fit would be a false statement about the clock; the
    # spec's section 6.2 fallback is the honest outcome, and run_jspace.py
    # reads a count of zero as exactly that.
    return 0, ("no positive multiple of ten fits inside the cap at "
               f"{round(t, 1)} seconds a prompt, so no twin lens can be fitted "
               "as the registered instrument and the spec's section 6.2 "
               "fallback applies: H18b is scored with the base lens on both "
               "sides")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timings", default=None,
                    help="a timings file written by fit_twin_lens.py, which "
                         "names the machine that measured it and the day")
    ap.add_argument("--probe-log", default=None,
                    help="a probe run log in the 2026-09-05 format, which is "
                         "how the committed decision was made")
    args = ap.parse_args()
    if bool(args.timings) == bool(args.probe_log):
        raise SystemExit(
            "name exactly one source of timings: --timings for a file this "
            "machine's own fit wrote, or --probe-log for a recorded run log. "
            "There is no default, because reading a log from another machine "
            "silently is how the budget rule went wrong before 2026-09-06.")

    machine, when = None, None
    if args.timings:
        source = Path(args.timings)
        times, machine, when = timings_from_file(source)
    else:
        source = Path(args.probe_log)
        times = timings_from_log(source)
    if not times:
        print(f"no per-prompt timings in {source}", file=sys.stderr)
        raise SystemExit(1)
    t = sum(times) / len(times)
    n, why = choose(t)

    rec = {"per_prompt_seconds": times, "n_probe_prompts": len(times),
           "mean_seconds_per_prompt": round(t, 1), "cap_seconds": CAP,
           "chosen_n_prompts": n, "projected_seconds": round(n * t),
           "rule": why,
           "below_50_is_a_recorded_deviation": 0 < n < 50,
           "dim_batch": 16,
           "dim_batch_reason": ("peak resident memory measured at about 2.5 "
                                "gigabytes at dim_batch 16, and the spec caps "
                                "peak memory at 3 gigabytes, so it was not "
                                "raised")}
    out = HERE / "output" / "fit_budget_decision.json"
    out.write_text(json.dumps(rec, indent=2))
    # The sidecar, not the decision, carries where the numbers came from, so
    # that the decision file itself stays byte for byte what it was.
    sidecar = {"source": source.name, "source_kind":
               "timings file" if args.timings else "probe run log",
               "source_sha256": sha256_file(source),
               "measured_on_machine": machine,
               "measured_utc": when,
               "mean_seconds_per_prompt": round(t, 1),
               "decision": out.name}
    (HERE / "output" / "fit_budget_decision.provenance.json").write_text(
        json.dumps(sidecar, indent=2))
    print(json.dumps(rec, indent=2))
    print(f"source: {source} ({sidecar['source_kind']}, sha256 "
          f"{sidecar['source_sha256'][:12]}"
          + (f", measured on {machine} at {when}" if machine else "")
          + ")")
    print(n)


if __name__ == "__main__":
    main()
