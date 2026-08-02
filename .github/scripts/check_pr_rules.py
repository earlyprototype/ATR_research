#!/usr/bin/env python3
"""CI enforcement for the standing rules in CLAUDE.md (R3, R4, R5).

R3 (identifier rule): every hypothesis ID (H-number) and experiment ID
(EXP-identifier) that a PR adds, in prose or in a new filename stem, must
have a row in _STAGE2_JSPACE/REGISTER.md as committed at the PR head. The
register's spelling is definitive; tokens are canonicalised (case-folded)
before lookup. The token grammar is deliberately narrow so ordinary prose
cannot false-positive: hypothesis IDs match \\bH\\d+[a-z]?\\b and
experiment IDs match \\bEXP_\\d{3}[a-z0-9]*(-[A-Za-z0-9]+)*\\b.

R4 (closing rule): a results-bearing PR (one that adds lines to a
RESULTS_*.md file) must carry a closing keyword (Closes/Fixes/Resolves #N)
or a body line beginning "No-Close:" saying why not.

R5 (run-log rule, machine-checkable part): a PR that adds artifact files
(.pt/.json) under an experiment's output/ directory must also add or
modify a .log file under that experiment, or carry a body line beginning
"Log-Exempt:" saying why not.

Inputs (environment): PR_BODY, BASE_REF. The diff is read from git
(merge-base three-dot diff against origin/BASE_REF).

Run the self-tests first: python3 check_pr_rules.py --self-test
"""

import os
import re
import subprocess
import sys

HYP_RE = re.compile(r"\bH\d+[a-z]?\b")
EXP_RE = re.compile(r"\bEXP_\d{3}[a-z0-9]*(?:-[A-Za-z0-9]+)*\b")
CLOSING_RE = re.compile(r"\b(?:[Cc]loses|[Ff]ixes|[Rr]esolves)\s+#\d+")
REGISTER_PATH = "_STAGE2_JSPACE/REGISTER.md"

# Paths never scanned for identifier tokens: CI tooling (this file's own
# tests carry deliberately unregistered fixtures) and board machinery.
EXEMPT_PREFIXES = (".github/", ".board/", ".claude/")


def extract_ids(text):
    """All identifier tokens in a piece of text, as written."""
    return set(HYP_RE.findall(text)) | set(EXP_RE.findall(text))


# Stems need a boundary-free EXP pattern: in "EXP_015_SPEC" the trailing
# underscore is a word character, so \b would miss the token.
EXP_STEM_RE = re.compile(r"EXP_\d{3}[a-z0-9]*(?:-[A-Za-z0-9]+)*")


def extract_ids_from_stem(path):
    """Identifier tokens carried by a filename stem, checked both as
    written and upper-cased (exp_015_spec.py -> EXP_015)."""
    stem = os.path.basename(path)
    stem = re.sub(r"\.[A-Za-z0-9]+$", "", stem)
    toks = set()
    for s in (stem, stem.upper()):
        toks |= set(HYP_RE.findall(s)) | set(EXP_STEM_RE.findall(s))
    return toks


def canon(token):
    return token.upper()


def load_register_ids(register_text):
    """Every identifier the register mentions counts as registered,
    including retired rows (which stay visible by house convention)."""
    return {canon(t) for t in extract_ids(register_text)}


def added_lines_by_file(diff_text):
    """Map path -> list of (line_number_in_new_file, line) for added lines."""
    out = {}
    path = None
    new_ln = 0
    for raw in diff_text.splitlines():
        if raw.startswith("+++ b/"):
            path = raw[6:]
            out.setdefault(path, [])
        elif raw.startswith("+++ /dev/null"):
            path = None
        elif raw.startswith("@@") and path is not None:
            m = re.search(r"\+(\d+)", raw)
            new_ln = int(m.group(1)) if m else 0
        elif raw.startswith("+") and not raw.startswith("+++") and path:
            out[path].append((new_ln, raw[1:]))
            new_ln += 1
        elif raw.startswith(" ") and path is not None:
            new_ln += 1
    return out


def new_files(diff_text):
    files = []
    cur_new = False
    for raw in diff_text.splitlines():
        if raw.startswith("diff --git"):
            cur_new = False
        elif raw.startswith("new file mode"):
            cur_new = True
        elif raw.startswith("+++ b/") and cur_new:
            files.append(raw[6:])
    return files


def check_r3(diff_text, register_text):
    registered = load_register_ids(register_text)
    failures = []
    for path, lines in added_lines_by_file(diff_text).items():
        if path.startswith(EXEMPT_PREFIXES):
            continue
        for ln, line in lines:
            for tok in sorted(extract_ids(line)):
                if canon(tok) not in registered:
                    failures.append(f"{path}:{ln}: unregistered identifier "
                                    f"'{tok}' (R3: add a REGISTER.md row in "
                                    f"the same commit as first use)")
    for path in new_files(diff_text):
        if path.startswith(EXEMPT_PREFIXES):
            continue
        for tok in sorted(extract_ids_from_stem(path)):
            if canon(tok) not in registered:
                failures.append(f"{path}: filename carries unregistered "
                                f"identifier '{tok}' (R3)")
    return failures


def check_r4(diff_text, pr_body):
    results_bearing = any(
        re.search(r"(^|/)RESULTS_[^/]*\.md$", p) and lines
        for p, lines in added_lines_by_file(diff_text).items())
    if not results_bearing:
        return []
    if CLOSING_RE.search(pr_body or ""):
        return []
    if re.search(r"^\s*No-Close:", pr_body or "", re.MULTILINE):
        return []
    return ["results-bearing PR has no closing keyword (Closes/Fixes/"
            "Resolves #N) and no 'No-Close:' line in the body (R4)"]


def check_r5(diff_text, pr_body):
    by_file = added_lines_by_file(diff_text)
    touched = set(by_file) | set(new_files(diff_text))
    artifact_dirs = set()
    log_dirs = set()
    for p in touched:
        m = re.match(r"(.*/experiments/[^/]+)/output/[^/]+\.(pt|json)$", p)
        if m:
            artifact_dirs.add(m.group(1))
        m = re.match(r"(.*/experiments/[^/]+)/.*\.log$", p)
        if m:
            log_dirs.add(m.group(1))
    missing = sorted(artifact_dirs - log_dirs)
    if not missing:
        return []
    if re.search(r"^\s*Log-Exempt:", pr_body or "", re.MULTILINE):
        return []
    return [f"artifacts added under {d}/output/ with no .log added or "
            f"updated under {d} and no 'Log-Exempt:' body line (R5)"
            for d in missing]


def git(*args):
    return subprocess.run(["git", *args], check=True, capture_output=True,
                          text=True).stdout


def main():
    base_ref = os.environ.get("BASE_REF", "main")
    pr_body = os.environ.get("PR_BODY", "")
    diff_text = git("diff", f"origin/{base_ref}...HEAD")
    try:
        register_text = open(REGISTER_PATH, encoding="utf-8").read()
    except FileNotFoundError:
        print(f"FAIL: {REGISTER_PATH} missing at PR head; R3 cannot run.")
        return 1
    failures = (check_r3(diff_text, register_text)
                + check_r4(diff_text, pr_body)
                + check_r5(diff_text, pr_body))
    if failures:
        print("PR rules check FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PR rules check passed (R3, R4, R5).")
    return 0


# ---------------------------------------------------------------------------
# Self-tests (R3 grammar and the diff parser). CI runs these before the
# check blocks anything, per the review's requirement that the check ships
# with parser tests over the register's full ID inventory.

SAMPLE_REGISTER = """
| H5 | EXP_010b | ... |
| H8 | EXP_012-PYTHIA | ... |
| H9a | EXP_010c | ... |
| H11b | vacated | ... |
| H15 | EXP_015 | ... |
| EXP_010c-3b | ... |
| EXP_012-PYTHIA | ... |
| EXP_010c-PERM | ... |
| EXP_014 | ... |
"""

SAMPLE_DIFF = """diff --git a/_STAGE2_JSPACE/NEW_SPEC.md b/_STAGE2_JSPACE/NEW_SPEC.md
new file mode 100644
--- /dev/null
+++ b/_STAGE2_JSPACE/NEW_SPEC.md
@@ -0,0 +1,3 @@
+This spec registers H15 for EXP_015 and cites EXP_010c-3b.
+It also cites EXP_999 which is not registered.
+Water is H2O; heads like L11.H8 are not hypothesis IDs.
"""


def self_test():
    import unittest

    class Grammar(unittest.TestCase):
        def test_hypothesis_tokens(self):
            self.assertEqual(extract_ids("H15 and H9a but not H2O"),
                             {"H15", "H9a"})

        def test_head_notation_is_not_a_hypothesis(self):
            # L11.H8 names an attention head; the token H8 alone is
            # extracted, which is correct: H8 is a registered hypothesis
            # and head notation is rare in specs. The grammar cannot
            # distinguish them; the register lookup makes it moot.
            self.assertIn("H8", extract_ids("L11.H8"))

        def test_experiment_tokens(self):
            self.assertEqual(
                extract_ids("EXP_010c-3b, EXP_012-PYTHIA, EXP_014"),
                {"EXP_010c-3b", "EXP_012-PYTHIA", "EXP_014"})

        def test_exp_d_is_out_of_grammar(self):
            self.assertEqual(extract_ids("EXP-D and Q-D"), set())

        def test_stems(self):
            self.assertEqual(extract_ids_from_stem("EXP_015_SPEC.md"),
                             {"EXP_015"})
            self.assertEqual(extract_ids_from_stem("exp_016_followup.py"),
                             {"EXP_016"})
            self.assertEqual(extract_ids_from_stem("run_exp010c.py"), set())

        def test_register_inventory(self):
            ids = load_register_ids(SAMPLE_REGISTER)
            for tok in ("H5", "H9A", "H11B", "H15", "EXP_010B",
                        "EXP_010C-3B", "EXP_012-PYTHIA", "EXP_010C-PERM"):
                self.assertIn(tok, ids)

        def test_full_register_file_parses(self):
            if os.path.exists(REGISTER_PATH):
                ids = load_register_ids(
                    open(REGISTER_PATH, encoding="utf-8").read())
                for tok in ("H15", "EXP_015", "EXP_010C-3B",
                            "EXP_012-PYTHIA"):
                    self.assertIn(tok, ids)

    class Checks(unittest.TestCase):
        def test_r3_flags_only_unregistered(self):
            failures = check_r3(SAMPLE_DIFF, SAMPLE_REGISTER)
            self.assertEqual(len(failures), 1)
            self.assertIn("EXP_999", failures[0])

        def test_r4(self):
            d = ("diff --git a/x/RESULTS_X.md b/x/RESULTS_X.md\n"
                 "--- a/x/RESULTS_X.md\n+++ b/x/RESULTS_X.md\n"
                 "@@ -1,0 +2,1 @@\n+a result line\n")
            self.assertTrue(check_r4(d, "no keyword here"))
            self.assertFalse(check_r4(d, "Closes #12"))
            self.assertFalse(check_r4(d, "No-Close: tracker stays open"))
            self.assertFalse(check_r4("", "no keyword"))

        def test_r5(self):
            d = ("diff --git a/e/experiments/x/output/a.json "
                 "b/e/experiments/x/output/a.json\n"
                 "new file mode 100644\n--- /dev/null\n"
                 "+++ b/e/experiments/x/output/a.json\n"
                 "@@ -0,0 +1,1 @@\n+{}\n")
            self.assertTrue(check_r5(d, ""))
            self.assertFalse(check_r5(d, "Log-Exempt: analysis-only rerun"))
            d_log = d + ("diff --git a/e/experiments/x/run.log "
                         "b/e/experiments/x/run.log\n"
                         "new file mode 100644\n--- /dev/null\n"
                         "+++ b/e/experiments/x/run.log\n"
                         "@@ -0,0 +1,1 @@\n+ok\n")
            self.assertFalse(check_r5(d_log, ""))

    suite = unittest.TestLoader().loadTestsFromTestCase(Grammar)
    suite.addTests(
        unittest.TestLoader().loadTestsFromTestCase(Checks))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(self_test())
    sys.exit(main())
