"""EXP_017 Part 2: the J-space overlap probe (H18b).

Spec: ../../EXP_017_SPEC.md section 6.

Takes the terminal tensors Part 1 produced, reads their per-layer states, and
measures how much of each state lies in the cone spanned by at most 25 lens
vectors, on the twin's own fitted lens and on the pre-fitted Neuronpedia lens
for base GPT-2 Small, with three random-rotation controls per side and the two
cross-checks that separate lens mismatch from model mismatch.

The base lens is resolved relative to this checkout, at
`_STAGE2_JSPACE/artifacts/jlens_gpt2_small_neuronpedia.pt`, and can be
overridden with --base-lens. That directory is not version controlled, so the
file has to be placed there (or named on the command line) before this runs;
its SHA-256 is checked against the digest the spec records either way.

A twin lens offered here is checked twice before it may carry the registered
comparison. It is checked against the prompt count the spec's budget rule chose,
recorded in output/fit_budget_decision.json, and against the provenance record
the fitting script now stamps into every lens file it writes, which says which
model at which revision and which fitting corpus the lens came from. A lens
fitted on fewer prompts than the budget chose is not the registered instrument:
it is either a fit the wall-clock cap stopped short or a deliberately smaller
one, so by default it is refused and the run takes the spec's section 6.2 route,
scoring both sides on the base lens and reporting H18b as untestable as
registered. A lens with no provenance stamp, which is what the committed twin
lens is because it predates the stamp, is likewise refused unless
--accept-unstamped-twin-lens is passed, which is recorded in the output. The
lens-quality sensitivity check, which is not registered and carries no verdict
weight, opts in with --allow-short-twin-lens and is stamped as a sensitivity
reading in its own output.

One filename rule, enforced rather than trusted. output/exp017_jspace.json is
the registered result: make_tables.py reads it and the results record quotes it.
Any run that is not the registered comparison, because of the coordinate frame,
because a short lens was admitted, or because the prompt-count requirement was
overridden, must be given its own --out-suffix, and this script refuses to start
the expensive part without one.

Coordinate frames, and why there is a choice. The states and the dictionary have
to be in the same coordinates for the share to mean anything. --frame tl, the
default and the frame every committed number was measured in, reads the states
from the TransformerLens conversion of the model and builds the dictionary from
that conversion's unembedding matrix, which folds the final normalisation step's
learned per-coordinate gain into every vocabulary direction and then subtracts a
common vector from all of them. --frame hf reads the states from the Hugging
Face model directly and builds the dictionary from its raw output matrix, which
is the frame the Jacobian matrices were fitted in and the one EXP_011 built its
dictionary from. The
two frames give different numbers, so only the registered frame can carry a
verdict; a run in the other frame is stamped as a sensitivity reading. Section
3.6 of the results record measures the difference.

Usage:
    python3 run_jspace.py --twin-lens ../../artifacts/jlens_lamini_gpt2_124m_40_twin.pt \
        --accept-unstamped-twin-lens
    python3 run_jspace.py --base-lens /some/other/path/jlens_gpt2_small_neuronpedia.pt
    python3 run_jspace.py --out-suffix _lens5 --allow-short-twin-lens \
        --twin-lens ../../artifacts/jlens_lamini_gpt2_124m_5_probe.pt
    python3 run_jspace.py --frame hf --out-suffix _hfframe \
        --accept-unstamped-twin-lens \
        --twin-lens ../../artifacts/jlens_lamini_gpt2_124m_40_twin.pt
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import exp017_models  # noqa: E402
from jspace import (K_ATOMS, pursue_batch, random_rotation, rotate_states,  # noqa: E402
                    unit_atoms)

OUT = HERE / "output"
ARTIFACTS = HERE.parent.parent / "artifacts"
# Resolved from this checkout, never from one machine's absolute path, so the
# probe runs from any clone that has the lens in its own artifacts directory.
BASE_LENS_DEFAULT = ARTIFACTS / "jlens_gpt2_small_neuronpedia.pt"
BASE_LENS_SHA = "d1800a1335ada089ef2e1ec0e4bd4d5bd61e6011eacc31f8618fdb3d10aae762"
BUDGET_JSON = OUT / "fit_budget_decision.json"
MODELS = exp017_models.MODELS
PROBE_LAYERS = list(range(11))      # 0..10, the lens's fitted source layers
BAND = list(range(5, 11))           # 5..10, the workspace band, verdict-bearing
ROT_SEEDS = (2026, 2027, 2028)
N_PERM = 10000
PERM_SEED = 42

# The coordinate frame every committed number was measured in. See the module
# docstring: "tl" reads states and unembedding from the TransformerLens
# conversion, "hf" reads both from the Hugging Face model itself.
FRAME_REGISTERED = "tl"
FRAMES = ("tl", "hf")

# The fitting corpus the registered twin lens was fitted on, committed beside
# this script. A lens's stamped provenance is checked against this file's own
# digest rather than against a copied-out constant, so the two cannot drift.
CORPUS_COMMITTED = HERE / "wikitext_prompts_160.json"

# What a twin lens's provenance stamp must say for it to carry the registered
# comparison. The sequence and layer settings are the ones EXP_017's fit used,
# and "complete" is the fit outcome of a fit that reached every prompt it asked
# for inside its cap, as against "short" or "overran".
EXPECTED_TWIN_LENS_FIT = {"max_seq_len": 128, "source_layers": list(range(11)),
                          "target_layer": 11, "skip_first": 16,
                          "fit_outcome": "complete"}

# The wall-clock cap the spec's section 6.2 sets on the twin's lens fit, in
# seconds. A lens fitted under a cap of its own that is longer than this is not
# the registered instrument however cleanly it finished.
REGISTERED_CAP_SECONDS = 9000.0


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def per_layer_states(which, prompt_ids, frame=FRAME_REGISTERED):
    """Inject each terminal tensor at the entrance to layer 0 and read the exit
    of every probed layer, in the coordinate frame asked for.

    Returns states [n_layers, d_model, n_prompts] (states as columns), the
    unembedding matrix of that frame as numpy [d_model, d_vocab], and the
    rescale factor applied to each terminal.
    """
    if frame not in FRAMES:
        raise SystemExit(f"unknown frame {frame!r}; expected one of {FRAMES}")
    reader = _states_tl if frame == "tl" else _states_hf
    return reader(which, prompt_ids)


def _states_tl(which, prompt_ids):
    """The registered frame: states and unembedding from the TransformerLens
    conversion, the way the lucier pilot read per-layer states.

    The terminal is rescaled to the loop's own re-injection size first, so the
    states are the ones the loop's next iteration would actually visit. The
    rescale factor is recorded; at a settled state it is close to 1.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformer_lens import HookedTransformer
    rev = exp017_models.revision(which)
    hf = AutoModelForCausalLM.from_pretrained(MODELS[which], revision=rev)
    tok = AutoTokenizer.from_pretrained(MODELS[which], revision=rev)
    model = HookedTransformer.from_pretrained("gpt2", hf_model=hf, tokenizer=tok,
                                              device="cpu")
    model.eval()

    terms = np.load(OUT / f"terminals_{which}.npz")
    loop = {r["prompt_id"]: r for r in json.load(open(OUT / f"loop_results_{which}.json"))}
    d = model.cfg.d_model
    states = np.zeros((len(PROBE_LAYERS), d, len(prompt_ids)), dtype=np.float32)
    rescales = {}
    names = [f"blocks.{l}.hook_resid_post" for l in PROBE_LAYERS]

    for pi, pid in enumerate(prompt_ids):
        T = torch.from_numpy(terms[f"{pid}|full"])
        target = float(loop[pid]["target_norm"])
        factor = target / float(T.norm())
        rescales[pid] = factor
        inj = T * factor

        def hook(resid, hook, tensor=inj):
            resid[0, :, :] = tensor
            return resid

        model.add_hook("blocks.0.hook_resid_pre", hook)
        try:
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    loop[pid]["prompt"], names_filter=lambda n: n in names)
        finally:
            model.reset_hooks()
        for li, l in enumerate(PROBE_LAYERS):
            states[li, :, pi] = cache[f"blocks.{l}.hook_resid_post"][0, -1, :].numpy()

    W_U = model.W_U.detach().numpy().astype(np.float32)   # [d_model, d_vocab]
    del model, hf
    return states, W_U, rescales


def _states_hf(which, prompt_ids):
    """The same reading in the Hugging Face model's own coordinates.

    The terminal is injected at the input of block 0 and the output of every
    probed block is read, which is the same place the TransformerLens route
    reads. Two differences are deliberate and are the whole point of this
    frame. The states are not mean-centred across the 768 coordinates, because
    no weight processing has been applied, and the unembedding returned is the
    model's raw output matrix, with the final normalisation step's learned gain
    left out of it. That is the frame the Jacobian matrices were fitted in, so
    a dictionary built here is in the same coordinates as the states it is
    scored against.

    The beginning-of-sequence token is prepended by hand, because
    TransformerLens prepends one by default and the terminal tensors were
    captured with it, so the sequence lengths have to agree.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rev = exp017_models.revision(which)
    hf = AutoModelForCausalLM.from_pretrained(MODELS[which], revision=rev)
    tok = AutoTokenizer.from_pretrained(MODELS[which], revision=rev)
    hf.eval()

    terms = np.load(OUT / f"terminals_{which}.npz")
    loop = {r["prompt_id"]: r for r in json.load(open(OUT / f"loop_results_{which}.json"))}
    blocks = hf.transformer.h
    d = int(hf.config.n_embd)
    states = np.zeros((len(PROBE_LAYERS), d, len(prompt_ids)), dtype=np.float32)
    rescales = {}
    box = {}

    def inject(module, args, kwargs):
        return (box["injected"].unsqueeze(0),) + tuple(args[1:]), kwargs

    def reader(li):
        def read(module, args, kwargs, output):
            resid = output[0] if isinstance(output, tuple) else output
            box[f"out{li}"] = resid.detach()
        return read

    handles = [blocks[0].register_forward_pre_hook(inject, with_kwargs=True)]
    handles += [blocks[l].register_forward_hook(reader(li), with_kwargs=True)
                for li, l in enumerate(PROBE_LAYERS)]
    try:
        for pi, pid in enumerate(prompt_ids):
            T = torch.from_numpy(terms[f"{pid}|full"])
            factor = float(loop[pid]["target_norm"]) / float(T.norm())
            rescales[pid] = factor
            box["injected"] = T * factor
            ids = tok(loop[pid]["prompt"], return_tensors="pt")["input_ids"]
            bos = torch.tensor([[tok.bos_token_id]], dtype=ids.dtype)
            ids = torch.cat([bos, ids], dim=1)
            assert ids.shape[1] == box["injected"].shape[0], (
                f"{pid}: {ids.shape[1]} tokens against a terminal of "
                f"{box['injected'].shape[0]} positions")
            with torch.no_grad():
                hf(ids)
            for li, l in enumerate(PROBE_LAYERS):
                states[li, :, pi] = box[f"out{li}"][0, -1, :].numpy()
    finally:
        for handle in handles:
            handle.remove()

    W_U = hf.lm_head.weight.detach().numpy().T.astype(np.float32)  # [d, d_vocab]
    del hf
    return states, W_U, rescales


def budget_n_prompts(path=BUDGET_JSON):
    """The prompt count the spec's budget rule chose for the twin's lens fit,
    read from the decision that rule already wrote. Returns None when that file
    is absent, in which case the caller has nothing to validate against."""
    if not path.exists():
        return None
    return int(json.load(open(path))["chosen_n_prompts"])


def twin_lens_decision(n_fitted, required, allow_short):
    """Whether a twin lens may carry the registered H18b comparison.

    Returns "accepted" when the lens was fitted on at least the prompt count
    the budget rule chose, "sensitivity" when it is shorter and the caller has
    opted in, and "refused" when it is shorter and has not. A refused lens sends
    the run down the spec's section 6.2 route, both sides on the base lens.
    """
    if n_fitted is None:
        return "refused"
    if required is None or n_fitted >= required:
        return "accepted"
    return "sensitivity" if allow_short else "refused"


def load_lens(path):
    """Load a fitted lens; returns {layer: J as numpy [d, d]} and its metadata.

    The metadata includes the provenance record the fitting script stamps into
    every lens file it writes, which says which model at which revision and
    which corpus the lens was fitted from. A lens written before that stamp
    existed has none, and the value is then None.
    """
    ck = torch.load(path, map_location="cpu", weights_only=True)
    J = {int(l): ck["J"][l].float().numpy().astype(np.float32) for l in ck["J"]}
    return J, {"n_prompts": int(ck["n_prompts"]), "d_model": int(ck["d_model"]),
               "source_layers": [int(x) for x in ck["source_layers"]],
               "provenance": ck.get("provenance")}


def expected_twin_lens_provenance(corpus=CORPUS_COMMITTED):
    """What a twin lens's stamp has to say to carry the registered comparison:
    the pinned twin weights and the committed fitting corpus, hashed here so
    that no digest is copied out and left to drift, with the sequence and layer
    settings EXP_017 fitted at."""
    want = {"model": exp017_models.name("twin"),
            "revision": exp017_models.revision("twin"),
            "corpus_sha256": sha256_file(corpus) if Path(corpus).exists() else None}
    want.update(EXPECTED_TWIN_LENS_FIT)
    return want


def twin_lens_provenance_decision(stamp, want, accept_unstamped,
                                  cap_seconds=REGISTERED_CAP_SECONDS):
    """Whether a twin lens's provenance stamp permits registered scoring.

    Returns ("accepted", []) when the stamp agrees with `want` field for field
    and the fit it describes finished inside the registered wall-clock cap,
    ("unstamped", []) when there is no stamp and the caller has opted in to
    that, ("refused", reasons) when there is no stamp and no opt-in, or when
    anything disagrees. The reasons are sentences naming what differs.

    The clock is checked as well as the fields, because a fit can be stopped by
    its own deadline and still produce a lens averaged over the prompt count the
    budget asked for: the stamp then says so, and such a lens is not the
    registered instrument. A stamp that raised its own cap above the spec's
    `cap_seconds` is refused for the same reason.
    """
    if stamp is None:
        if accept_unstamped:
            return "unstamped", []
        return "refused", ["the lens file carries no provenance stamp, so "
                           "nothing in it says which model or which corpus it "
                           "was fitted from"]
    bad = []
    for key, expected in want.items():
        got = stamp.get(key)
        if isinstance(expected, list) and got is not None:
            got = list(got)
        if got != expected:
            bad.append(f"{key} is {got!r} in the lens and {expected!r} here")
    deadline = stamp.get("deadline_seconds")
    if deadline is None:
        bad.append("the stamp does not say what wall-clock cap the fit ran "
                   f"under, and the registered cap is {cap_seconds:.0f} seconds")
    elif float(deadline) > cap_seconds:
        bad.append(f"the fit ran under a {float(deadline):.0f} second cap of its "
                   f"own, longer than the registered {cap_seconds:.0f} seconds")
    wall = stamp.get("fit_wall_seconds")
    if wall is not None and float(wall) > cap_seconds:
        bad.append(f"the fit took {float(wall):.0f} seconds against the "
                   f"registered {cap_seconds:.0f} second cap")
    return ("accepted", []) if not bad else ("refused", bad)


def sensitivity_reasons(*, frame, short_reading, twin_n_prompts, required,
                        registered_requirement, budget):
    """Why this run is not the registered comparison, as sentences.

    An empty list means it is the registered comparison. Each entry is written
    to be read on its own, because they are joined into the verdict string, into
    the artifact and into the message that refuses the registered filename.
    """
    reasons = []
    if frame != FRAME_REGISTERED:
        reasons.append(f"this run measured states and dictionary in the {frame} "
                       f"frame, and every registered number was measured in the "
                       f"{FRAME_REGISTERED} frame")
    if short_reading:
        reasons.append(f"the twin lens was fitted on {twin_n_prompts} prompts "
                       f"against the {required} the budget rule chose")
    if not registered_requirement:
        reasons.append(f"this run required {required} fitting prompts of a twin "
                       f"lens where the budget rule chose {budget}")
    return reasons


def requirement_decision(override, budget):
    """The prompt count a twin lens must have been fitted on, and whether that
    requirement is the registered one.

    The registered requirement is the count the spec's budget rule chose, which
    `output/fit_budget_decision.json` records. `--twin-lens-min-prompts` may
    lower or raise it for a deliberate check, but a run whose requirement is not
    the budget's own count is not the registered comparison, whatever it then
    accepts, so this returns that fact rather than leaving it implicit.
    """
    if override is None:
        return budget, True
    return override, budget is not None and int(override) == int(budget)


def permutation_p_two_sided(a, b, n_perm=None, seed=PERM_SEED):
    """Two-sided permutation p on the difference of medians.

    Pool the two samples, reassign which model each value belongs to n_perm
    times, and report the fraction of reassignments whose absolute median
    difference reaches the observed one, with the standard add-one correction.

    `n_perm` defaults to the module-level N_PERM, and it is read here rather
    than bound when this function is defined, so that a caller which lowers
    N_PERM for a cheap check really does run fewer reassignments.
    """
    n_perm = N_PERM if n_perm is None else n_perm
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = abs(float(np.median(a) - np.median(b)))
    pool = np.concatenate([a, b])
    na = len(a)
    rng = np.random.default_rng(seed)
    hits = 1
    for _ in range(n_perm):
        rng.shuffle(pool)
        if abs(float(np.median(pool[:na]) - np.median(pool[na:]))) >= obs - 1e-12:
            hits += 1
    return hits / (n_perm + 1), obs


def selftest():
    """The twin-lens budget gate, checked without loading a model or a lens
    matrix. The committed lens files are read only for their prompt counts."""
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              f"{(': ' + detail) if detail else ''}")

    got = twin_lens_decision(40, 40, False)
    check("a lens meeting the budget is accepted as registered",
          got == "accepted", got)
    got = twin_lens_decision(41, 40, False)
    check("a lens exceeding the budget is accepted as registered",
          got == "accepted", got)
    got = twin_lens_decision(5, 40, False)
    check("a short lens is refused, sending the run to the section 6.2 route",
          got == "refused", got)
    got = twin_lens_decision(39, 40, False)
    check("a lens one prompt short is refused too", got == "refused", got)
    got = twin_lens_decision(5, 40, True)
    check("a short lens with the opt-in becomes a sensitivity reading",
          got == "sensitivity", got)
    got = twin_lens_decision(None, 40, True)
    check("no lens is never accepted", got == "refused", got)

    required = budget_n_prompts()
    check("the budget rule's chosen prompt count is readable",
          required == 40, f"{required} prompts from {BUDGET_JSON.name}")
    for name, expect in (("jlens_lamini_gpt2_124m_40_twin.pt", "accepted"),
                         ("jlens_lamini_gpt2_124m_5_probe.pt", "refused")):
        path = ARTIFACTS / name
        if not path.exists():
            check(f"{name} is present to check", False, "file absent")
            continue
        ck = torch.load(path, map_location="cpu", weights_only=True)
        n = int(ck["n_prompts"])
        got = twin_lens_decision(n, required, False)
        check(f"the committed {name} is {expect} without the opt-in",
              got == expect, f"fitted on {n} prompts, decision {got}")

    # The provenance gate, which asks what a lens was fitted from rather than
    # how many prompts it averaged.
    want = expected_twin_lens_provenance()
    check("the committed fitting corpus is hashed for the comparison",
          want["corpus_sha256"] is not None
          and len(str(want["corpus_sha256"])) == 64,
          f"{str(want['corpus_sha256'])[:12]} from {CORPUS_COMMITTED.name}")
    good = dict(want, model="MBZUAI/LaMini-GPT-124M", n_prompts_fitted=40,
                fit_wall_seconds=8253.0, deadline_seconds=REGISTERED_CAP_SECONDS)
    got, why = twin_lens_provenance_decision(good, want, False)
    check("a lens stamped with this experiment's own fit is accepted",
          got == "accepted", f"{got} {why}")
    got, why = twin_lens_provenance_decision(None, want, False)
    check("an unstamped lens is refused without the opt-in", got == "refused",
          "; ".join(why))
    got, why = twin_lens_provenance_decision(None, want, True)
    check("an unstamped lens with the opt-in is admitted and recorded as such",
          got == "unstamped", got)
    got, why = twin_lens_provenance_decision(
        dict(good, corpus_sha256="0" * 64), want, True)
    check("a lens fitted on another corpus is refused even with the opt-in",
          got == "refused" and any("corpus_sha256" in w for w in why),
          "; ".join(why))
    got, why = twin_lens_provenance_decision(
        dict(good, model="/local/checkout", revision="local path, no revision "
             "pinned"), want, True)
    check("a lens fitted from a local model path is refused",
          got == "refused" and any("model" in w for w in why), "; ".join(why))
    got, why = twin_lens_provenance_decision(dict(good, max_seq_len=512), want, True)
    check("a lens fitted at another sequence length is refused",
          got == "refused" and any("max_seq_len" in w for w in why), "; ".join(why))
    # A fit the deadline stopped can still average the prompt count the budget
    # asked for, so the outcome stamp is checked, not just the count.
    got, why = twin_lens_provenance_decision(
        dict(good, fit_outcome="overran", fit_wall_seconds=9619.0), want, True)
    check("a lens from a fit that crossed its cap is refused, whatever its "
          "prompt count", got == "refused"
          and any("fit_outcome" in w for w in why), "; ".join(why))
    got, why = twin_lens_provenance_decision(dict(good, fit_outcome="short"),
                                             want, True)
    check("a lens from a fit stopped short is refused", got == "refused"
          and any("fit_outcome" in w for w in why), "; ".join(why))
    got, why = twin_lens_provenance_decision(
        dict(good, deadline_seconds=36000.0), want, True)
    check("a lens fitted under a cap longer than the registered one is refused",
          got == "refused" and any("cap of its own" in w for w in why),
          "; ".join(why))
    stamp = dict(good)
    stamp.pop("deadline_seconds")
    got, why = twin_lens_provenance_decision(stamp, want, True)
    check("a lens whose stamp does not name its cap is refused",
          got == "refused" and any("wall-clock cap" in w for w in why),
          "; ".join(why))

    # The prompt-count requirement, and what overriding it means.
    got = requirement_decision(None, 40)
    check("with no override the budget rule's count is the requirement",
          got == (40, True), str(got))
    got = requirement_decision(40, 40)
    check("an override equal to the budget rule's count is still registered",
          got == (40, True), str(got))
    got = requirement_decision(5, 40)
    check("an override below the budget rule's count is not the registered "
          "requirement", got == (5, False), str(got))
    got = requirement_decision(100, 40)
    check("an override above it is not the registered requirement either",
          got == (100, False), str(got))
    got = requirement_decision(5, None)
    check("an override with no budget artifact to compare is not registered",
          got == (5, False), str(got))

    # What makes a run a sensitivity reading, and therefore what may not be
    # written to the registered filename.
    args_registered = dict(frame=FRAME_REGISTERED, short_reading=False,
                           twin_n_prompts=40, required=40,
                           registered_requirement=True, budget=40)
    check("the registered run has no sensitivity reason",
          sensitivity_reasons(**args_registered) == [],
          "no reason, so it may write output/exp017_jspace.json")
    got = sensitivity_reasons(**{**args_registered, "frame": "hf"})
    check("a run in the other frame is a sensitivity reading",
          len(got) == 1 and "hf frame" in got[0], "; ".join(got))
    got = sensitivity_reasons(**{**args_registered, "short_reading": True,
                                 "twin_n_prompts": 5})
    check("a run on a short lens is a sensitivity reading",
          len(got) == 1 and "5 prompts" in got[0], "; ".join(got))
    got = sensitivity_reasons(**{**args_registered, "registered_requirement": False,
                                 "required": 5})
    check("a run with the requirement overridden is a sensitivity reading",
          len(got) == 1 and "required 5 fitting prompts" in got[0],
          "; ".join(got))
    got = sensitivity_reasons(**{**args_registered, "frame": "hf",
                                 "short_reading": True, "twin_n_prompts": 5})
    check("two reasons are both named rather than the first only",
          len(got) == 2, "; ".join(got))
    for name in ("jlens_lamini_gpt2_124m_40_twin.pt",):
        path = ARTIFACTS / name
        if not path.exists():
            check(f"{name} is present to check its stamp", False, "file absent")
            continue
        stamp = load_lens(path)[1]["provenance"]
        got, why = twin_lens_provenance_decision(stamp, want, False)
        check(f"the committed {name} carries no stamp, so registered scoring "
              f"needs --accept-unstamped-twin-lens",
              stamp is None and got == "refused", "; ".join(why))

    # The permutation count is read when the test runs, not when this module is
    # imported, so a caller that lowers it really does run fewer reassignments.
    saved = globals()["N_PERM"]
    try:
        globals()["N_PERM"] = 9
        p, obs = permutation_p_two_sided([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        check("lowering the module's permutation count changes the test's "
              "denominator", abs(p * 10 - round(p * 10)) < 1e-9,
              f"p {p:.4f} is a multiple of one tenth, so 9 reassignments ran")
        p2, _ = permutation_p_two_sided([1.0, 2.0, 3.0], [4.0, 5.0, 6.0],
                                        n_perm=99)
        check("an explicit count still overrides the module's",
              abs(p2 * 100 - round(p2 * 100)) < 1e-9, f"p {p2:.4f}")
    finally:
        globals()["N_PERM"] = saved
    check("the default permutation count is restored after that check",
          N_PERM == 10000, f"N_PERM {N_PERM}")

    # The frame gate: only the registered frame may carry a verdict.
    check("the registered frame is the TransformerLens one every committed "
          "number was measured in", FRAME_REGISTERED == "tl", FRAME_REGISTERED)
    check("the other frame is offered and named", set(FRAMES) == {"tl", "hf"},
          ", ".join(FRAMES))

    print(f"\nselftest: {'ALL PASS' if ok else 'FAILURE'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="check the twin-lens budget gate and exit")
    ap.add_argument("--out-suffix", default="",
                    help="suffix for the output filename; used only by the "
                         "non-registered harness check, which must not "
                         "overwrite the registered artifact")
    ap.add_argument("--twin-lens", default=None,
                    help="path to the fitted twin lens; omit to score on the "
                         "base lens only (spec section 6.2 fallback)")
    ap.add_argument("--base-lens", default=str(BASE_LENS_DEFAULT),
                    help="path to the pre-fitted Neuronpedia lens for base "
                         "GPT-2 Small; defaults to this checkout's own "
                         "_STAGE2_JSPACE/artifacts/ copy, and its SHA-256 is "
                         "checked against the digest the spec records")
    ap.add_argument("--twin-lens-min-prompts", type=int, default=None,
                    help="the prompt count a twin lens must have been fitted "
                         "on; defaults to the count the budget rule chose in "
                         "output/fit_budget_decision.json, which is the "
                         "registered requirement. Any other value makes the run "
                         "a sensitivity reading, whatever it then accepts, and "
                         "the run must then be given its own --out-suffix")
    ap.add_argument("--allow-short-twin-lens", action="store_true",
                    help="score a twin lens fitted on fewer prompts than the "
                         "budget chose; the result is stamped as a sensitivity "
                         "reading and is not the registered comparison")
    ap.add_argument("--accept-unstamped-twin-lens", action="store_true",
                    help="score a twin lens that carries no provenance stamp, "
                         "so nothing in the file says which model or which "
                         "corpus it was fitted from; needed for the committed "
                         "twin lens, which predates the stamp, and recorded in "
                         "the output")
    ap.add_argument("--frame", choices=FRAMES, default=FRAME_REGISTERED,
                    help="coordinate frame for the states and the dictionary: "
                         "tl reads both from the TransformerLens conversion and "
                         "is the frame every committed number was measured in, "
                         "hf reads both from the Hugging Face model, which is "
                         "the frame the Jacobians were fitted in. A run in a "
                         "frame other than the registered one is stamped as a "
                         "sensitivity reading and carries no verdict weight.")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(selftest())
    base_lens = Path(args.base_lens)
    if not base_lens.exists():
        raise SystemExit(
            f"base lens not found at {base_lens}. That directory is not "
            f"version controlled; put jlens_gpt2_small_neuronpedia.pt there, "
            f"or pass --base-lens with its path.")

    subset = json.load(open(HERE.parent / "exp_010c_windows" / "output"
                            / "prompt_subset_small.json"))
    prompt_ids = [r["id"] for r in subset]
    t0 = time.time()

    rep = {"experiment": "EXP_017", "spec": "../../EXP_017_SPEC.md",
           "k_atoms": K_ATOMS, "probe_layers": PROBE_LAYERS, "band_layers": BAND,
           "rotation_seeds": list(ROT_SEEDS), "n_perm": N_PERM,
           "perm_seed": PERM_SEED, "prompt_ids": prompt_ids, "lenses": {},
           "frame": args.frame, "registered_frame": FRAME_REGISTERED,
           "frame_note": (
               "tl: states and dictionary from the TransformerLens conversion, "
               "whose unembedding folds in the final normalisation gain and "
               "subtracts a common vector from every vocabulary direction. "
               "hf: states and dictionary from the Hugging Face model, the "
               "frame the Jacobians were fitted in.")}

    # ---- lenses --------------------------------------------------------------
    got = sha256_file(base_lens)
    rep["lenses"]["base"] = {"path": str(base_lens), "sha256": got,
                             "sha256_expected": BASE_LENS_SHA,
                             "digest_verified": got == BASE_LENS_SHA}
    assert got == BASE_LENS_SHA, f"base lens digest mismatch: {got}"
    J_base, meta_base = load_lens(base_lens)
    rep["lenses"]["base"].update(meta_base)
    print(f"base lens: {meta_base}", flush=True)

    lenses = {"base": J_base}
    # A twin lens counts as the registered instrument only if it was fitted on
    # at least the prompt count the spec's budget rule chose. Anything shorter
    # is either a fit the wall-clock cap stopped or a deliberately smaller one,
    # and scoring it as registered would report SUPPORTED or NOT SUPPORTED where
    # spec section 6.2 requires the base lens on both sides.
    budget = budget_n_prompts()
    required, registered_requirement = requirement_decision(
        args.twin_lens_min_prompts, budget)
    short_reading = False
    rep["twin_lens_budget"] = {
        "required_n_prompts": required,
        "source_of_requirement": ("--twin-lens-min-prompts"
                                  if args.twin_lens_min_prompts is not None
                                  else "output/fit_budget_decision.json"),
        "budget_rule_n_prompts": budget,
        "requirement_is_the_registered_one": registered_requirement,
        "allow_short_twin_lens": bool(args.allow_short_twin_lens)}
    if not registered_requirement:
        print(f"REQUIREMENT OVERRIDDEN: this run asks a twin lens for "
              f"{required} prompts where the budget rule chose {budget}, so it "
              f"is a sensitivity reading whatever it accepts.", flush=True)
    if args.twin_lens:
        p = Path(args.twin_lens)
        J_twin, meta_twin = load_lens(p)
        n_fitted = int(meta_twin["n_prompts"])
        rep["lenses"]["twin"] = {"path": str(p), "sha256": sha256_file(p),
                                 **meta_twin}
        rep["twin_lens_budget"].update(twin_lens_n_prompts=n_fitted)
        decision = twin_lens_decision(n_fitted, required, args.allow_short_twin_lens)
        rep["twin_lens_budget"]["meets_budget"] = decision == "accepted"
        rep["twin_lens_budget"]["decision"] = decision
        # The second gate: what the lens says it was fitted from. A lens can
        # meet the prompt-count budget and still have been fitted on another
        # corpus, or on a local checkout of another 768-wide model, so the
        # prompt count alone never establishes that it is the registered
        # instrument.
        want = expected_twin_lens_provenance()
        prov, reasons = twin_lens_provenance_decision(
            meta_twin.get("provenance"), want, args.accept_unstamped_twin_lens)
        rep["twin_lens_provenance"] = {
            "expected": want, "stamped": meta_twin.get("provenance"),
            "decision": prov, "reasons": reasons,
            "accept_unstamped_twin_lens": bool(args.accept_unstamped_twin_lens)}
        if prov == "refused":
            # The message is printed once, by the shared refusal branch below,
            # which names the reason this gate gave.
            decision = "refused"
            rep["twin_lens_budget"]["decision"] = decision
        elif prov == "unstamped":
            print("UNSTAMPED TWIN LENS ACCEPTED: the lens file says nothing "
                  "about which model or corpus it was fitted from, and that "
                  "acceptance is recorded in the output.", flush=True)
        if decision == "refused":
            rep["twin_lens_budget"]["action"] = (
                "refused for registered scoring; spec section 6.2 route taken")
            rep["lenses"]["twin_refused"] = rep["lenses"].pop("twin")
            rep["lenses"]["twin"] = None
            why = ("its provenance: " + "; ".join(reasons) if prov == "refused"
                   else f"it was fitted on {n_fitted} prompts, short of the "
                        f"{required} the budget rule chose")
            print(f"TWIN LENS REFUSED because {why}. Scoring both sides on "
                  f"the base lens (spec section 6.2 fallback). Pass "
                  f"--allow-short-twin-lens to score a short lens as a "
                  f"sensitivity reading, or --accept-unstamped-twin-lens to "
                  f"score an unstamped one.", flush=True)
        else:
            if decision == "sensitivity":
                short_reading = True
                rep["twin_lens_budget"]["action"] = (
                    "scored as a sensitivity reading, not the registered "
                    "comparison")
                print(f"SHORT TWIN LENS ACCEPTED: fitted on {n_fitted} prompts "
                      f"against the {required} the budget rule chose. This run "
                      f"is a sensitivity reading and carries no verdict weight.",
                      flush=True)
            else:
                rep["twin_lens_budget"]["action"] = "accepted as registered"
            lenses["twin"] = J_twin
            print(f"twin lens: {meta_twin}", flush=True)
    else:
        rep["lenses"]["twin"] = None
        rep["twin_lens_budget"].update(twin_lens_n_prompts=None, meets_budget=None,
                                       action="no twin lens offered")
        print("NO TWIN LENS: base lens only (spec section 6.2 fallback)", flush=True)

    # Everything that makes this run something other than the registered
    # comparison, collected in one place so the verdict string, the artifact and
    # the filename rule all read from the same list.
    not_registered = sensitivity_reasons(
        frame=args.frame, short_reading=short_reading,
        twin_n_prompts=rep["twin_lens_budget"].get("twin_lens_n_prompts"),
        required=required, registered_requirement=registered_requirement,
        budget=budget)
    rep["sensitivity_reasons"] = not_registered
    # A run that is not the registered comparison never writes to the registered
    # filename, because output/exp017_jspace.json is what make_tables.py reads
    # and what the results record quotes.
    if not_registered and not args.out_suffix:
        raise SystemExit(
            "REFUSING TO WRITE THE REGISTERED FILENAME: this run is not the "
            "registered comparison (" + "; ".join(not_registered) + "). Pass "
            "--out-suffix with a name of its own, for example --out-suffix "
            "_sensitivity, so that output/exp017_jspace.json keeps the "
            "registered result.")

    # ---- per-layer states, one model at a time so only one is resident -------
    states, W_U, rescale = {}, {}, {}
    for which in ("twin", "base"):
        states[which], W_U[which], rescale[which] = per_layer_states(
            which, prompt_ids, args.frame)
        print(f"{which}: per-layer states read in the {args.frame} frame, "
              f"rescale factor mean "
              f"{np.mean(list(rescale[which].values())):.6f}", flush=True)
    # How far the states sit from mean-centred across the 768 coordinates, which
    # is what tells a reader of the artifact which frame it holds without
    # trusting the label: the TransformerLens conversion centres every write, so
    # its states have a mean of zero to floating-point noise.
    rep["state_mean_over_coordinates"] = {
        which: {"max_abs_mean": float(np.abs(states[which].mean(axis=1)).max()),
                "mean_state_norm": float(
                    np.linalg.norm(states[which], axis=1).mean())}
        for which in states}
    rep["terminal_rescale_to_injection_size"] = {
        k: {"mean": float(np.mean(list(v.values()))),
            "min": float(np.min(list(v.values()))),
            "max": float(np.max(list(v.values())))} for k, v in rescale.items()}

    rots = {s: random_rotation(states["base"].shape[1], s) for s in ROT_SEEDS}

    # ---- lens diagnostics, so instrument differences stay visible -----------
    diag = {}
    for l in PROBE_LAYERS:
        row = {ln: {"frobenius_norm": float(np.linalg.norm(J[l]))}
               for ln, J in lenses.items()}
        if len(lenses) == 2:
            a = lenses["twin"][l].ravel()
            b = lenses["base"][l].ravel()
            row["cosine_between_the_two_lenses"] = float(
                a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))
        diag[str(l)] = row
    rep["lens_diagnostics_per_layer"] = diag
    print(f"lens diagnostics: {json.dumps(diag)}", flush=True)

    # ---- the decomposition ---------------------------------------------------
    # shares[lens][state_model][variant] = array [n_layers, n_prompts]
    shares = {ln: {m: {v: np.zeros((len(PROBE_LAYERS), len(prompt_ids)),
                                   dtype=np.float64)
                       for v in ("real",) + tuple(f"rot{s}" for s in ROT_SEEDS)}
                   for m in ("twin", "base")} for ln in lenses}
    n_sel = {ln: {m: np.zeros((len(PROBE_LAYERS), len(prompt_ids)), dtype=int)
                  for m in ("twin", "base")} for ln in lenses}

    for lens_name, J in lenses.items():
        wu = W_U[lens_name]                       # the lens's own model unembeds
        for li, l in enumerate(PROBE_LAYERS):
            tl = time.time()
            A = (wu.T @ J[l]).astype(np.float32)  # [d_vocab, d_model] atoms
            U, keep = unit_atoms(A)
            del A
            cols, index = [], []
            for m in ("twin", "base"):
                H = states[m][li]                 # [d, n_prompts]
                cols.append(H); index.append((m, "real"))
                for s in ROT_SEEDS:
                    cols.append(rotate_states(H, rots[s]))
                    index.append((m, f"rot{s}"))
            Hall = np.concatenate(cols, axis=1)
            sh, ns, _ = pursue_batch(U, Hall, k=K_ATOMS, keep=keep)
            w = len(prompt_ids)
            for bi, (m, v) in enumerate(index):
                shares[lens_name][m][v][li] = sh[bi * w:(bi + 1) * w]
                if v == "real":
                    n_sel[lens_name][m][li] = ns[bi * w:(bi + 1) * w]
            del U, Hall, cols
            print(f"  lens={lens_name} layer={l}: "
                  f"twin real median {np.median(shares[lens_name]['twin']['real'][li]):.4f} "
                  f"base real median {np.median(shares[lens_name]['base']['real'][li]):.4f} "
                  f"({time.time() - tl:.0f}s)", flush=True)

    rep["shares"] = {ln: {m: {v: shares[ln][m][v].tolist() for v in shares[ln][m]}
                          for m in shares[ln]} for ln in shares}
    rep["n_atoms_selected_real"] = {ln: {m: n_sel[ln][m].tolist() for m in n_sel[ln]}
                                    for ln in n_sel}

    # ---- H18b scoring, spec section 6.6 -------------------------------------
    have_twin_lens = "twin" in lenses
    primary = {"twin_side": "twin-on-twin" if have_twin_lens else "twin-on-base",
               "base_side": "base-on-base", "per_layer": {}}
    twin_lens_key = "twin" if have_twin_lens else "base"
    for li, l in enumerate(PROBE_LAYERS):
        tw = shares[twin_lens_key]["twin"]["real"][li]
        bs = shares["base"]["base"]["real"][li]
        p, obs = permutation_p_two_sided(tw, bs)
        spread = max(abs(float(np.median(shares[twin_lens_key]["twin"][f"rot{s}"][li]))
                         - float(np.median(shares["base"]["base"][f"rot{s}"][li])))
                     for s in ROT_SEEDS)
        primary["per_layer"][str(l)] = {
            "median_twin": float(np.median(tw)), "median_base": float(np.median(bs)),
            "abs_median_difference": obs, "control_spread": spread,
            "exceeds_control_spread": bool(obs > spread),
            "perm_p": round(p, 5), "perm_p_below_0.05": bool(p < 0.05),
            "both_conditions": bool(obs > spread and p < 0.05),
            "in_band": l in BAND,
        }
    hits = [l for l in BAND if primary["per_layer"][str(l)]["both_conditions"]]
    primary["band_layers_meeting_both"] = hits
    primary["n_band_layers_meeting_both"] = len(hits)
    if not have_twin_lens:
        why = rep["twin_lens_budget"].get("action", "no twin lens offered")
        primary["h18b"] = (f"UNTESTABLE as registered (base lens both sides: "
                           f"{why})")
        primary["registered_scoring"] = False
    elif not_registered:
        verdict = "SUPPORTED" if len(hits) >= 4 else "NOT SUPPORTED"
        primary["h18b"] = (
            f"{verdict} as a sensitivity reading only, NOT the registered "
            f"comparison: " + "; ".join(not_registered))
        primary["registered_scoring"] = False
    else:
        primary["h18b"] = "SUPPORTED" if len(hits) >= 4 else "NOT SUPPORTED"
        primary["registered_scoring"] = True
    primary["sensitivity_reasons"] = not_registered
    # Kept beside the verdict rather than inside its wording, so that a reader
    # of the artifact sees on what terms the twin lens was admitted.
    primary["frame"] = args.frame
    primary["twin_lens_provenance_decision"] = rep.get(
        "twin_lens_provenance", {}).get("decision")
    rep["h18b"] = primary

    # ---- cross-checks --------------------------------------------------------
    cross = {}
    for lens_name in lenses:
        for m in ("twin", "base"):
            key = f"{m}-on-{lens_name}"
            cross[key] = {str(l): {
                "median_real": float(np.median(shares[lens_name][m]["real"][li])),
                "median_controls": [float(np.median(shares[lens_name][m][f"rot{s}"][li]))
                                    for s in ROT_SEEDS],
                "mean_atoms_selected": float(np.mean(n_sel[lens_name][m][li])),
            } for li, l in enumerate(PROBE_LAYERS)}
    rep["cross_checks"] = cross

    # ---- the same-lens comparison, which separates model from instrument ----
    # The registered H18b pairing measures the twin on one lens and base on
    # another, so a difference could in principle come from the lenses. Holding
    # the lens fixed and swapping only whose states are decomposed isolates the
    # model effect; holding the states fixed and swapping only the lens
    # isolates the instrument effect. Both are reported per layer.
    same_lens = {}
    for lens_name in lenses:
        rows = {}
        for li, l in enumerate(PROBE_LAYERS):
            tw = shares[lens_name]["twin"]["real"][li]
            bs = shares[lens_name]["base"]["real"][li]
            p, obs = permutation_p_two_sided(tw, bs)
            rows[str(l)] = {"median_twin_states": float(np.median(tw)),
                            "median_base_states": float(np.median(bs)),
                            "model_effect_abs_median_difference": obs,
                            "perm_p": round(p, 5)}
        same_lens[lens_name] = rows
    rep["same_lens_model_effect"] = same_lens

    if len(lenses) == 2:
        lens_effect = {}
        for m in ("twin", "base"):
            rows = {}
            for li, l in enumerate(PROBE_LAYERS):
                a = shares["twin"][m]["real"][li]
                b = shares["base"][m]["real"][li]
                rows[str(l)] = {
                    "median_on_twin_lens": float(np.median(a)),
                    "median_on_base_lens": float(np.median(b)),
                    "lens_effect_abs_median_difference":
                        abs(float(np.median(a) - np.median(b)))}
            lens_effect[m] = rows
        rep["same_states_lens_effect"] = lens_effect
        ratios = {}
        for li, l in enumerate(PROBE_LAYERS):
            model = max(same_lens[ln][str(l)]["model_effect_abs_median_difference"]
                        for ln in lenses)
            model_min = min(same_lens[ln][str(l)]["model_effect_abs_median_difference"]
                            for ln in lenses)
            lens = max(lens_effect[m][str(l)]["lens_effect_abs_median_difference"]
                       for m in ("twin", "base"))
            ratios[str(l)] = {"smallest_model_effect": model_min,
                              "largest_model_effect": model,
                              "largest_lens_effect": lens,
                              "ratio_smallest_model_over_largest_lens":
                                  model_min / max(lens, 1e-12)}
        rep["model_effect_over_lens_effect"] = ratios

    rep["wall_seconds"] = round(time.time() - t0, 1)

    outfile = OUT / f"exp017_jspace{args.out_suffix}.json"
    outfile.write_text(json.dumps(rep, indent=2))
    print(f"\n=== H18b: {primary['h18b']} "
          f"({len(hits)}/6 band layers meet both conditions: {hits}) ===", flush=True)
    print(f"{'layer':<7}{'twin':<9}{'base':<9}{'|diff|':<9}{'ctrl spread':<13}{'perm p':<9}both")
    for l in PROBE_LAYERS:
        r = primary["per_layer"][str(l)]
        print(f"{l:<7}{r['median_twin']:<9.4f}{r['median_base']:<9.4f}"
              f"{r['abs_median_difference']:<9.4f}{r['control_spread']:<13.4f}"
              f"{r['perm_p']:<9}{r['both_conditions']}{'  <- band' if l in BAND else ''}")
    print(f"\nSaved -> output/{outfile.name} ({rep['wall_seconds']:.0f}s)")


if __name__ == "__main__":
    main()
