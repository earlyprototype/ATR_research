"""EXP_011: the pinned-lens gate and the artifact-binding gates every stage shares.

Specification section 3 pins the instrument by path, SHA-256 digest and byte
count: the file
`_STAGE2_JSPACE/artifacts/jlens_gpt2_small_neuronpedia.pt`, digest
`d1800a1335ada089ef2e1ec0e4bd4d5bd61e6011eacc31f8618fdb3d10aae762`, 12,980,477
bytes. The pinned pair is written down once, here, and every stage that loads
the lens calls `verify_lens` before using it, so a file swapped between two
stages cannot be analysed while the outputs inherit an earlier stage's
attribution. Duplicating the constant in each stage was the earlier arrangement
and is exactly what this module replaces.

Beside the lens gate this module holds the checks that tie one stage's inputs to
another stage's outputs: the lens the decomposition ran against, the pairing of a
shares file with its own atom records, the binding of a shares file to the state
files it was computed from, and the layers an atom-record file actually covers.
They live together because two stages, the scoring and the readout, need the same
ones and must not each keep their own copy.

Added on 2026-09-05 in review of pull request 84, and extended on 2026-09-06.
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


def check_atom_records_against_shares(atom_records, shares_path, log=None):
    """Refuse a shares file and an atom-record file that are not the same run's.

    A run writes its shares to one file and the atom indices and coefficients of
    the same decomposition to another. From 2026-09-06 the atom-record file
    carries a `_meta` block naming the shares file it was written beside and the
    lens digest that run used, because until then a scratch run under a custom
    output name wrote its atom records over the official `atom_records.json`, and
    a stage opening the default names would then have combined one run's shares
    with another run's atoms while looking normal. Returns that metadata block, or
    None when the file predates it. Raises SystemExit when the two disagree.
    """
    meta = atom_records.get("_meta") if isinstance(atom_records, dict) else None
    if meta is None:
        if log is not None:
            log("the atom-record file carries no pairing metadata, so it predates "
                "the 2026-09-06 field and cannot be tied to a shares file by name; "
                "it rests on the pinned lens check alone")
        return None
    want = os.path.basename(shares_path)
    if meta.get("shares_file") != want:
        raise SystemExit(
            "GATE FAILED: the atom records were written beside shares file "
            f"{meta.get('shares_file')!r}, and this stage was given {want!r}. One "
            "run's atom indices would be read against another run's shares. "
            "Refusing to write a readout.")
    recorded = None
    if os.path.exists(shares_path):
        with open(shares_path) as fh:
            recorded = json.load(fh).get("lens_sha256")
    if recorded is not None and meta.get("lens_sha256") != recorded:
        raise SystemExit(
            f"GATE FAILED: the atom records were computed against lens "
            f"{meta.get('lens_sha256')} and {want} records lens {recorded}. "
            "Refusing to write a readout.")
    if log is not None:
        log(f"the atom records name shares file {want}, which is the one this stage "
            "opened, so the two artifacts are one decomposition")
    return meta


def check_states_against_shares(shares, meta, out_dir, log=None):
    """Refuse a shares file that was not computed from the state files on disk.

    Two stages take positional indices out of `output/states_meta.json` and apply
    them to arrays saved in `output/shares.json` by an earlier decomposition: the
    scoring takes the medoid representatives hypothesis H6 is scored on, the run-17
    convergence mask and the order of the named states, and the readout takes the
    order of the named states again. If `build_states.py` is re-run without
    re-running the decomposition, those indices address the new ordering while the
    shares still hold the old one, and until 2026-09-06 nothing said so. Two checks
    now do. The counts check compares the number of states in every saved share
    array with the number of states the metadata records for that family, and it
    applies to every shares file including the ones written before this field
    existed. The digest check compares the digests `decompose.py` records for both
    state artifacts with the files on disk, and reports that it cannot be applied
    when the shares file predates the field. Returns what was checked. Raises
    SystemExit on a mismatch.
    """
    meta_path = os.path.join(out_dir, "states_meta.json")
    npz_path = os.path.join(out_dir, "states.npz")
    out = {"counts_checked": False, "digests_checked": False,
           "states_meta_sha256_in_shares": shares.get("states_meta_sha256"),
           "states_npz_sha256_in_shares": shares.get("states_npz_sha256"),
           "states_meta_sha256_on_disk": None, "states_npz_sha256_on_disk": None,
           "state_counts": {}}

    # 1. The counts check, which needs no recorded field and so covers every file.
    expected = {fam: int(shape[0])
                for fam, shape in (meta.get("array_shapes") or {}).items()}
    out["state_counts"] = expected
    bad = []
    for arm, fams in shares.get("arms", {}).items():
        for fam, layers in fams.items():
            if fam not in expected:
                continue
            for layer, entry in layers.items():
                if len(entry.get("share", [])) != expected[fam]:
                    bad.append((arm, fam, layer, len(entry.get("share", [])),
                                expected[fam]))
    if not expected:
        if log is not None:
            log("the state metadata records no array shapes, so the number of states "
                "in the shares file cannot be checked against it")
    else:
        out["counts_checked"] = True
        if bad:
            raise SystemExit(
                "GATE FAILED: the shares file holds a different number of states "
                "from the state metadata beside it, in {} arm-family-layer groups, "
                "for example {} (arm, family, layer, states in the shares file, "
                "states in the metadata). The metadata's positional indices would "
                "address the wrong states. Re-run the decomposition against these "
                "states before scoring or reading out."
                .format(len(bad), bad[:3]))
        if log is not None:
            log("state-count check: every saved share array holds the number of "
                "states the metadata records for its family "
                + ", ".join(f"{f} {n}" for f, n in sorted(expected.items())))

    # 2. The digest check, which applies from the 2026-09-06 field onward.
    recorded_meta = shares.get("states_meta_sha256")
    recorded_npz = shares.get("states_npz_sha256")
    if recorded_meta is None and recorded_npz is None:
        if log is not None:
            log("the shares file records no digest of the state files it was "
                "computed from, so it predates the 2026-09-06 field and the two "
                "artifacts rest on the state-count check and the run logs alone")
        return out
    out["digests_checked"] = True
    for label, recorded, path, key in (
            ("states_meta.json", recorded_meta, meta_path, "states_meta_sha256_on_disk"),
            ("states.npz", recorded_npz, npz_path, "states_npz_sha256_on_disk")):
        if recorded is None:
            continue
        if not os.path.exists(path):
            if log is not None:
                log(f"{label} is not on disk beside this stage, so the digest the "
                    "shares file records for it cannot be compared")
            continue
        found = sha256(path)
        out[key] = found
        if found != recorded:
            raise SystemExit(
                f"GATE FAILED: the shares file was computed from {label} with digest "
                f"{recorded}, and the {label} on disk is {found}. The state files "
                "have been rebuilt since the decomposition ran, so the metadata's "
                "positional indices belong to a different ordering from the saved "
                "shares. Re-run the decomposition against these states.")
        if log is not None:
            log(f"{label} matches the digest the shares file records for it, {found}")
    return out


def atom_record_layers(atom_records, family="named", n_layers=12):
    """Which layers an atom-record file covers for one family, and which it lacks.

    A partial decomposition, `decompose.py --layers 5` for instance, writes atom
    records for those layers alone. A stage that reads fixed layers out of them has
    to ask first, rather than raising a lookup error part of the way through and
    leaving half its outputs written. Returns the covered layers and the absent
    ones, both sorted, out of the layers 0 to n_layers - 1.
    """
    covered = sorted(int(x) for x in atom_records.get(family, {}))
    absent = [l for l in range(n_layers) if l not in covered]
    return covered, absent


if __name__ == "__main__":
    print(verify_lens(log=print, stage="gate self-check"))
