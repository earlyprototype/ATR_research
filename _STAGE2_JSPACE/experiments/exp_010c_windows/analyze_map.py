"""Tested-windows map across every ATR window arm run on gpt2-medium.

Pre-registered spec: ../../EXP_010c3b_SPEC.md §3. Single place where the
whole-word flag and the content/function column are computed, so every cell is
scored by the same code rather than by hand.

The flag rule is UNCHANGED from EXP_010c-3 (>=2 unique terminals AND plurality
lexical class whole-word). Per spec §3 the rule was deliberately NOT tightened
after seeing that it flags i=14: narrowing a rule to exclude one inconvenient
cell would stop it being mechanical. Instead a content/function column is added
and applied uniformly to every cell, using the closed-class list enumerated in
the spec (reproduced verbatim below and not editable per-cell).

Usage: python analyze_map.py [--json]
"""

import argparse
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"

# Verbatim from EXP_010c3b_SPEC.md §3. Do not extend without amending the spec.
FUNCTION_WORDS = set("""
a an the this that these those my your his her its our their
and or but nor so yet for
of in on at to from by with about into over under between through during
is are was were be been being am do does did have has had
will would shall should can could may might must
not no nor as if then than there here it he she they we you i
""".split())

RESULT_FILES = ["results_full.json", "results_scan.json", "results_infill.json",
                "results_ladder8.json"]
CHAR_FILES = ["terminal_characterisation_full.json",
              "terminal_characterisation_scan.json",
              "terminal_characterisation_infill.json",
              "terminal_characterisation_ladder8.json"]


def token_class(tok):
    """Checkable lexical class (same rule as the EXP_010c-3 record).
    punctuation: no alphanumeric character; whole-word: leading space + all
    alphabetic core, len>=2; fragment: everything else."""
    core = tok.strip()
    if core == "" or not any(ch.isalnum() for ch in core):
        return "punctuation"
    if tok[:1] == " " and core.isalpha() and len(core) >= 2:
        return "whole-word"
    return "fragment"


def content_function(tok):
    """Closed-class membership of the token, applied uniformly to every cell."""
    if token_class(tok) != "whole-word":
        return "-"
    return "function" if tok.strip().lower() in FUNCTION_WORDS else "content"


def load():
    results, chars = [], {}
    for f in RESULT_FILES:
        p = OUT / f
        if p.exists():
            results += json.load(open(p))
    for f in CHAR_FILES:
        p = OUT / f
        if p.exists():
            for c in json.load(open(p)):
                chars[c["arm"]] = c
    return results, chars


def cell(arm, results, chars):
    rs = [r for r in results if r["arm"] == arm]
    if not rs:
        return None
    toks = Counter(r["terminal_token"] for r in rs)
    classes = Counter(token_class(r["terminal_token"]) for r in rs)
    plural_tok, plural_n = toks.most_common(1)[0]
    dom = classes.most_common(1)[0][0]
    c = chars.get(arm, {})
    return {
        "arm": arm, "window": rs[0]["window"], "n": len(rs),
        "unique": len(toks),
        "whole_word_n": classes.get("whole-word", 0),
        "plurality_token": plural_tok, "plurality_n": plural_n,
        "dominant_class": dom,
        "plurality_content_function": content_function(plural_tok),
        "flag": (len(toks) >= 2) and dom == "whole-word",
        "basins": c.get("tensor_basins", "-"),
        "via_tail": c.get("tail_agreement", "-"),
        "converged": sum(r["converged"] for r in rs),
    }


# (arm, axis label) in map order
INJECTION_21 = [("O0", 0), ("O4", 4), ("I5", 5), ("O6", 6), ("I7", 7), ("O8", 8),
                ("I9", 9), ("A4", 10), ("I11", 11), ("O12", 12), ("O14", 14)]
LADDER_8 = [("A5", 15), ("X817", 17), ("X819", 19), ("O8", 21), ("E822", 22), ("E823", 23)]
LADDER_10 = [("X1015", 15), ("X1017", 17), ("X1019", 19), ("A4", 21), ("E22", 22), ("E23", 23)]


def table(title, pairs, axis, results, chars):
    print(f"\n### {title}\n")
    print(f"| {axis} | arm | unique | whole-word | plurality token | content/function | basins | via-tail | flag |")
    print("|---|---|---|---|---|---|---|---|---|")
    rows = []
    for arm, ax in pairs:
        c = cell(arm, results, chars)
        if not c:
            continue
        tok = repr(c["plurality_token"]).replace("|", "\\|")
        print(f"| {ax} | {arm} | {c['unique']} | {c['whole_word_n']}/{c['n']} | "
              f"`{tok}` ×{c['plurality_n']} | {c['plurality_content_function']} | "
              f"{c['basins']} | {c['via_tail']} | {'YES' if c['flag'] else '·'} |")
        rows.append(dict(c, axis=ax))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    results, chars = load()
    print(f"Loaded {len(results)} runs across {len({r['arm'] for r in results})} arms.")
    out = {
        "injection_axis_extract21": table("Injection axis (extract 21)", INJECTION_21, "i", results, chars),
        "extraction_ladder_inject8": table("Extraction ladder, injection 8", LADDER_8, "j", results, chars),
        "extraction_ladder_inject10": table("Extraction ladder, injection 10", LADDER_10, "j", results, chars),
    }
    flagged = [r["arm"] for r in out["injection_axis_extract21"] if r["flag"]]
    print(f"\nFlagged cells on the injection axis: {flagged}")
    for r in out["injection_axis_extract21"]:
        if r["flag"]:
            print(f"  {r['arm']} (i={r['axis']}): plurality {r['plurality_token']!r} "
                  f"-> {r['plurality_content_function']}")
    if args.json:
        p = OUT / "tested_windows_map.json"
        p.write_text(json.dumps(out, indent=2))
        print(f"\nSaved -> {p}")


if __name__ == "__main__":
    main()
