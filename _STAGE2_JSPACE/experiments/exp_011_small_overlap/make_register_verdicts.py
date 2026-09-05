"""EXP_011: emit the proposed register rows, copying each statement verbatim.

This branch never edits _STAGE2_JSPACE/REGISTER.md. It writes the rows it
proposes into REGISTER_VERDICTS.md in the register's own column format, with
each hypothesis statement copied character for character from the register so
that a verdict is recorded against the registered wording and nothing else.

Run: python3 make_register_verdicts.py
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
REGISTER = os.path.abspath(os.path.join(HERE, "..", "..", "REGISTER.md"))
RESULTS_PATH = "experiments/exp_011_small_overlap/RESULTS_EXP011.md"

v = json.load(open(os.path.join(OUT, "verdicts.json")))
reg = open(REGISTER).read().splitlines()

NOTES = {
    "H6": ("the five basins beat the eighteen at three of the six band layers, one "
           "short of the pre-registered majority of four, and lose at one. Scored on "
           "the registered wording; the 18-basin arm it names is the mis-scaled "
           "original noise arm of lucier finding F4, which ran at 28 percent of the "
           "language arm's injection strength, so this verdict is not evidence about "
           "language against noise at matched strength. H16 carries that question"),
    "H16": ("language and matched-strength noise terminals are indistinguishable: "
            "median shares differ by -0.0029 to +0.0002 on shares of 0.014 to 0.022, "
            "no band layer significant in the hypothesised direction. The registered "
            "chance level, a norm-matched random dictionary, is also shown in the "
            "record to be an unfair comparison, because it lacks the lens's "
            "directional clustering; against a rotated-lens control the language "
            "terminals sit above chance at five of six band layers. See decision "
            "item 1 of the results record"),
    "H16a": ("prolet beats phase A at three of six band layers and phase B at one of "
             "six, where four of six against each was required. The phases straddle "
             "prolet, as finding F16 describes, but in the opposite direction: on the "
             "full-vocabulary lens phase B is the more lens-expressible phase and "
             "phase A the less, which retracts F16's phase assignment"),
    "H16b": ("supported at five of the six band layers with permutation p of 0.0032 "
             "once and 0.0001 four times, on 64 to 96 percent of the 125 paired "
             "prompts. Holds for last-position residuals, the reading the loop itself "
             "uses; the pre-registered position-averaged secondary runs the other way "
             "at layers 5 to 8 and is reported in the record"),
}


def row_for(hid):
    pat = re.compile(r"^\|\s*" + re.escape(hid) + r"\s*\|")
    for line in reg:
        if pat.match(line):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) == 5:
                return cells
    raise SystemExit(f"no register row found for {hid}")


lines = []
for hid in ("H6", "H16", "H16a", "H16b"):
    cells = row_for(hid)
    verdict = f"**{v[hid]['verdict']}** ({NOTES[hid]})"
    lines.append(f"| {cells[0]} | EXP_011 | {cells[2]} | {verdict} | "
                 f"`{RESULTS_PATH}` |")

exp_cells = None
for line in reg:
    if line.startswith("| EXP_011 |"):
        exp_cells = [c.strip() for c in line.strip().strip("|").split("|")]
        break
exp_row = (f"| {exp_cells[0]} | {exp_cells[1]} | **COMPLETE** (2026-09-05) | "
           f"`EXP_011_SPEC.md` | `{RESULTS_PATH}` |")

body = f"""# EXP_011: proposed register rows

These rows are proposed, not applied. This branch does not edit
`_STAGE2_JSPACE/REGISTER.md`; the orchestrator applies every experiment's rows in
one sweep. Each row is written in the register's own column format so it can
replace the existing row without further editing, and each hypothesis statement
is copied character for character from the register, because a verdict is
scored on the registered wording and this experiment restated nothing.

The register's hypothesis table columns are: identifier, owning experiment,
statement in one line, verdict, recorded at. Its experiment table columns are:
identifier, what it is, status, spec, results.

## Section 1, hypothesis register: replace these four rows

| ID | Owning experiment | Statement (one line) | Verdict | Recorded at |
|---|---|---|---|---|
{chr(10).join(lines)}

## Section 2, experiment register: replace this row

| ID | What it is | Status | Spec | Results |
|---|---|---|---|---|
{exp_row}

## Note for whoever applies these

H6's existing owning-experiment cell reads "EXP_011 (planned)". The rows above
drop the parenthetical, because the experiment has now run. Nothing else in the
register is touched by this experiment.
"""
with open(os.path.join(HERE, "REGISTER_VERDICTS.md"), "w") as fh:
    fh.write(body)
print(body)
