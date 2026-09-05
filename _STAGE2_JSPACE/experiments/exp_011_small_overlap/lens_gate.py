"""EXP_011: the pinned-lens gate, shared by every stage that opens the lens file.

Specification section 3 pins the instrument by path, SHA-256 digest and byte
count: the file
`_STAGE2_JSPACE/artifacts/jlens_gpt2_small_neuronpedia.pt`, digest
`d1800a1335ada089ef2e1ec0e4bd4d5bd61e6011eacc31f8618fdb3d10aae762`, 12,980,477
bytes. The pinned pair is written down once, here, and every stage that loads
the lens calls `verify_lens` before using it, so a file swapped between two
stages cannot be analysed while the outputs inherit an earlier stage's
attribution. Duplicating the constant in each stage was the earlier arrangement
and is exactly what this module replaces.

Added on 2026-09-05 in review of pull request 84.
"""
import hashlib
import json
import os

LENS_PT = "/home/user/ATR_research/_STAGE2_JSPACE/artifacts/jlens_gpt2_small_neuronpedia.pt"
LENS_SHA256 = "d1800a1335ada089ef2e1ec0e4bd4d5bd61e6011eacc31f8618fdb3d10aae762"
LENS_BYTES = 12980477


def sha256(path):
    """SHA-256 digest of a file, read in one-megabyte chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_lens(path=LENS_PT, log=None, stage=""):
    """Check the lens file against the pinned digest and size, and describe it.

    Returns the identity block a stage writes into its own output: the path, the
    digest, the byte count, and the fact that both matched. Raises SystemExit on
    any mismatch, because a lens that is not the pinned one makes every share
    computed against it unattributable.
    """
    sha, nbytes = sha256(path), os.path.getsize(path)
    if log is not None:
        where = f" ({stage})" if stage else ""
        log(f"lens file {path}{where}")
        log(f"  SHA-256 {sha} ({nbytes} bytes)")
    if sha != LENS_SHA256 or nbytes != LENS_BYTES:
        raise SystemExit(
            "GATE FAILED: the lens file does not match the digest and size the "
            f"specification pins ({LENS_SHA256}, {LENS_BYTES} bytes). Found "
            f"{sha}, {nbytes} bytes. Refusing to use an unattributed instrument"
            + (f" in {stage}." if stage else "."))
    return {"lens_file": path, "lens_sha256": sha, "lens_bytes": nbytes,
            "lens_digest_matches_spec": True}


def check_against_decomposition(shares_path, lens_id, log=None):
    """Tie a later stage's lens to the one the decomposition actually used.

    A stage that consumes output/atom_records.json reads atom indices and
    coefficients chosen against the dictionary the decomposition saw. If the lens
    file changed after that, those indices belong to a different dictionary from
    the one the consuming stage builds. The shares file records the digest of the
    lens the decomposition ran against, from 2026-09-05 onward, so compare the
    two. Returns that recorded digest, or None when the shares file predates the
    field or does not exist. Raises SystemExit when the two disagree.
    """
    if not os.path.exists(shares_path):
        if log is not None:
            log(f"no {os.path.basename(shares_path)} beside this stage, so there is "
                "no recorded decomposition digest to compare against")
        return None
    with open(shares_path) as fh:
        recorded = json.load(fh).get("lens_sha256")
    if recorded is None:
        if log is not None:
            log("the shares file records no lens digest of its own, so it predates "
                "the 2026-09-05 gate; the saved atom records cannot be tied to a "
                "lens file by digest and rest on the pinned check alone")
        return None
    if recorded != lens_id["lens_sha256"]:
        raise SystemExit(
            f"GATE FAILED: {os.path.basename(shares_path)} was computed against "
            f"lens {recorded}, and the lens on disk is {lens_id['lens_sha256']}. "
            "The saved atom indices and coefficients would be read against a "
            "different dictionary from the one that chose them. Refusing to write "
            "a readout.")
    if log is not None:
        log(f"the shares file records the same lens digest, {recorded}, so the "
            "saved atom records and this stage's dictionary are one instrument")
    return recorded


if __name__ == "__main__":
    print(verify_lens(log=print, stage="gate self-check"))
