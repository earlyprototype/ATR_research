"""EXP_010c — Window grid on GPT-2 Medium. Pre-registered spec: ../../EXP_010c_SPEC.md

Arms (inject i -> extract j), seeding: natural L0 prompt pass in every arm (spec §3).

Tiers:
    smoke  — 2 prompts, arms A0+A4, max_iter=60   (harness validation only)
    pilot  — 5 prompts, all arms,   max_iter=300  (directional signal, ~1-2 h CPU)
    full   — 25 prompts, all arms,  max_iter=1000 (the pre-registered run, overnight)

The gated protocol params (threshold/patience/check_every/check_start) follow the
spec for `full`; smoke/pilot shrink check_start proportionally and are RECORDED as
non-registered tiers — no verdicts are drawn from them beyond harness validity.

Usage: python run_exp010c.py --tier smoke|pilot|full [--arms A0,A4,...]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # for atr_engine2

from atr_engine2 import run_atr_gated  # noqa: E402
from derive_prompts import select_subset, select_subset_b  # noqa: E402
from derive_prompts_pythia import select_subset_pythia  # noqa: E402
from derive_prompts_small import select_subset_small  # noqa: E402

ARMS = {
    "A0": (0, 23),   # baseline / reproduction gate
    "A4": (10, 21),  # band-exact
    "A1": (0, 11),   # placement: front
    "A2": (6, 17),   # placement: middle
    "A3": (12, 23),  # placement: back
    "A5": (8, 15),   # length probe (8 layers, mid-band)
    # EXP_010c-2 boundary scan (EXP_010c2_SPEC.md §3)
    "E22": (10, 22),  # exit edge: +layer 22
    "E23": (10, 23),  # exit edge: +motor tail (final-layer necessity, H10a)
    "O0": (0, 21),    # onset edge: sensory-splice test (H10b)
    "O4": (4, 21),
    "O6": (6, 21),
    "O8": (8, 21),
    "O12": (12, 21),
    "O14": (14, 21),
    # EXP_010c-3 in-fill around the word-forming cells (EXP_010c3_SPEC.md §3)
    "I5": (5, 21),     # injection in-fill (extract 21): odd layers around 8/10
    "I7": (7, 21),
    "I9": (9, 21),     # critical point between the two word cells 8->21 and 10->21
    "I11": (11, 21),
    "X817": (8, 17),   # extraction column below 21 at the word-forming injections
    "X819": (8, 19),
    "X1015": (10, 15),
    "X1017": (10, 17),
    "X1019": (10, 19),
    # EXP_010c-3b §4 — extraction ladder above 21 at injection 8
    "E822": (8, 22),
    "E823": (8, 23),
    # EXP_010c-VARIANTS Control A (EXP_010c_VARIANTS_SPEC.md §3, issue #13)
    "I1A0": (1, 23),   # i=1 variant of A0 (0->23)
    "I1O0": (1, 21),   # i=1 variant of O0 (0->21)
    "HP9": (10, 21),   # sanity: inject at blocks.9.hook_resid_post (see ARM_INJECT_HOOK)
    # EXP_010b — window grid on GPT-2 Small (EXP_010b_SPEC.md §4, issue #16).
    # 12-layer model: windows are RUNBOOK_PHASE1 §EXP_010b step 2 verbatim.
    "SB": (0, 11),   # full-stack baseline — reproduction gate vs Stage 1
    "S1": (0, 5),    # 6-layer early
    "S2": (3, 8),    # 6-layer mid
    "S3": (6, 11),   # 6-layer late
    "S4": (0, 8),    # 9-layer front-heavy
    "S5": (3, 11),   # 9-layer back-heavy
}
# Control A sanity arm: pre-registered expectation is that resid_post(9) is
# the SAME residual value as resid_pre(10), so HP9 must reproduce A4 exactly.
ARM_INJECT_HOOK = {"HP9": "blocks.9.hook_resid_post"}
ARM_ORDER = ["A0", "A4", "A1", "A2", "A3", "A5"]  # spec §5 execution order
SCAN_ORDER = ["E22", "E23", "O0", "O4", "O6", "O8", "O12", "O14"]
INFILL_ORDER = ["I5", "I7", "I9", "I11",  # EXP_010c3_SPEC.md §5 execution order
                "X817", "X819", "X1015", "X1017", "X1019"]
LADDER8_ORDER = ["E822", "E823"]          # EXP_010c3b_SPEC.md §4

TIERS = {
    "smoke": dict(n_prompts=2, max_iter=60, check_start=20, arms=["A0", "A4"]),
    "pilot": dict(n_prompts=5, max_iter=300, check_start=50, arms=ARM_ORDER),
    "full": dict(n_prompts=25, max_iter=1000, check_start=100, arms=ARM_ORDER),
    "scan": dict(n_prompts=25, max_iter=1000, check_start=100, arms=SCAN_ORDER),
    "infill": dict(n_prompts=25, max_iter=1000, check_start=100, arms=INFILL_ORDER),
    "ladder8": dict(n_prompts=25, max_iter=1000, check_start=100, arms=LADDER8_ORDER),
    # EXP_010c3b_SPEC.md §5 — settle-time variant. check_start=10 is a RECORDED
    # protocol deviation (earliest reportable lock drops 120 -> 30); it exists to
    # measure settle time and draws no verdicts on the registered questions.
    "settle": dict(n_prompts=5, max_iter=1000, check_start=10,
                   arms=["I7", "I9", "X1017"]),
    # EXP_010c-VARIANTS tiers (spec §3/§4): registered protocol, variant arms.
    "hookpoint": dict(n_prompts=25, max_iter=1000, check_start=100,
                      arms=["I1A0", "I1O0", "HP9"]),
    "energynorm": dict(n_prompts=25, max_iter=1000, check_start=100,
                       arms=["A0", "A4", "O8", "A1"]),
    # EXP_012-PYTHIA (EXP_012_PYTHIA_SPEC.md §3, issue #12): registered
    # protocol, same absolute windows (both models are 24-layer). Reported
    # with a P- prefix (P-A0 ... P-O8) in the results register.
    "pythia": dict(n_prompts=25, max_iter=1000, check_start=100,
                   arms=["A0", "A1", "A2", "A3", "A4", "O8"]),
    # EXP_010b (EXP_010b_SPEC.md §4/§6, issue #16): registered protocol on
    # gpt2-small. SB runs alone first (reproduction gate — evaluated before
    # any other arm; per-arm invocations via --arms). small_smoke is harness
    # validation only, non-registered, no verdict weight.
    "small010b": dict(n_prompts=25, max_iter=1000, check_start=100,
                      arms=["SB", "S1", "S2", "S3", "S4", "S5"]),
    "small_smoke": dict(n_prompts=2, max_iter=60, check_start=20,
                        arms=["SB", "S2"]),
}


def run_arm_with_terminal(model, prompt, i, j, max_iter, check_start,
                          inject_hook_name=None, renorm="seed_j"):
    """Thin wrapper: the gated protocol lives ONLY in atr_engine2.run_atr_gated
    (capture_terminal=True adds terminal tensors + a real lag_scan dict — the
    recorded diff vs the upstream engine; see atr_engine2.py header).

    History note (PR #4 review): an earlier version re-implemented the gated
    loop here and saved lag_scan's dict KEYS instead of its cosine values, so
    every pre-fix artifact carries the placeholder [1.0..8.0]. Fixed by this
    wrapper; artifacts regenerated.
    """
    r = run_atr_gated(model, prompt, i, j, max_iter=max_iter,
                      check_start=check_start, capture_terminal=True,
                      inject_hook_name=inject_hook_name, renorm=renorm)
    # lag_scan arrives as {lag: mean_cosine}; keep the mapping explicit.
    if r.get("lag_scan") is not None:
        r["lag_scan"] = {str(k): v for k, v in r["lag_scan"].items()}
    r["terminal_prob"] = float(r["terminal_prob"])
    return r


class _DummyTokenizer:
    """Decode stub for --harness-check runs (random-init model, no real vocab)."""

    padding_side = "right"
    pad_token_id = 0

    def decode(self, ids):
        """Render dummy token ids as visible placeholders like <42>."""
        ids = ids if isinstance(ids, (list, tuple)) else [ids]
        return "".join(f"<{int(i)}>" for i in ids)


def _toy_model():
    """Random-init 24-layer model for harness validation only (no network, no
    pretrained weights). Validates hooks / windowed loop / gating / artifacts —
    NOT the D-collapse reproduction gate, which needs real gpt2-medium weights."""
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    cfg = HookedTransformerConfig(
        n_layers=24, d_model=64, n_ctx=32, d_head=16, n_heads=4,
        d_vocab=997, act_fn="gelu", normalization_type="LN",
    )
    model = HookedTransformer(cfg)
    model.tokenizer = _DummyTokenizer()
    return model


def _toy_tokens(prompt, d_vocab=997, max_len=12):
    """Map a prompt string to deterministic dummy token ids (harness-check only)."""
    ids = [(hash(w) % (d_vocab - 1)) + 1 for w in prompt.split()[:max_len]]
    return torch.tensor([ids or [1]], dtype=torch.long)


def _load_medium_from_local(path):
    """Offline load: local dir must hold config.json, pytorch_model.bin,
    vocab.json, merges.txt (e.g. from the legacy HF S3 mirror). Seeds the HF
    cache with config.json so transformer_lens's internal AutoConfig lookup
    resolves without network, then passes model+tokenizer in explicitly."""
    import os
    import shutil

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    snap = cache / "models--gpt2-medium" / "snapshots" / "local"
    refs = cache / "models--gpt2-medium" / "refs"
    snap.mkdir(parents=True, exist_ok=True)
    refs.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(path) / "config.json", snap / "config.json")
    (refs / "main").write_text("local")

    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    from transformer_lens import HookedTransformer

    hf = GPT2LMHeadModel.from_pretrained(path)
    tok = GPT2Tokenizer.from_pretrained(path)
    return HookedTransformer.from_pretrained("gpt2-medium", hf_model=hf, tokenizer=tok)


def _load_pythia_from_local(path):
    """Offline load for EleutherAI/pythia-410m (EXP_012_PYTHIA_SPEC.md §2/§3,
    issue #12): local dir must hold config.json, pytorch_model.bin,
    tokenizer.json (GPTNeoX ships tokenizer.json, not vocab/merges). Same
    cache-seeding pattern as the medium loader, under the official repo name
    that transformer_lens's alias resolves to."""
    import os
    import shutil

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    snap = cache / "models--EleutherAI--pythia-410m" / "snapshots" / "local"
    refs = cache / "models--EleutherAI--pythia-410m" / "refs"
    snap.mkdir(parents=True, exist_ok=True)
    refs.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(path) / "config.json", snap / "config.json")
    (refs / "main").write_text("local")

    from transformers import AutoTokenizer, GPTNeoXForCausalLM
    from transformer_lens import HookedTransformer

    hf = GPTNeoXForCausalLM.from_pretrained(path)
    # Compatibility shim (recorded): transformers 5.x renamed GPTNeoX's output
    # embedding attribute `embed_out` -> `lm_head`; transformer_lens 3.5.1's
    # convert_neox_weights still reads `embed_out`. Alias the SAME module
    # object under the old name — no weights are copied or altered.
    if not hasattr(hf, "embed_out"):
        hf.embed_out = hf.lm_head
    tok = AutoTokenizer.from_pretrained(path)
    return HookedTransformer.from_pretrained("pythia-410m", hf_model=hf, tokenizer=tok)


def _load_small_from_local(path):
    """Offline load for gpt2 (Small) — EXP_010b_SPEC.md §2/§6, issue #16.
    Same cache-seeding pattern as the medium loader; local dir must hold
    config.json, pytorch_model.bin, vocab.json, merges.txt."""
    import os
    import shutil

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    snap = cache / "models--gpt2" / "snapshots" / "local"
    refs = cache / "models--gpt2" / "refs"
    snap.mkdir(parents=True, exist_ok=True)
    refs.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(path) / "config.json", snap / "config.json")
    (refs / "main").write_text("local")

    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    from transformer_lens import HookedTransformer

    hf = GPT2LMHeadModel.from_pretrained(path)
    tok = GPT2Tokenizer.from_pretrained(path)
    return HookedTransformer.from_pretrained("gpt2", hf_model=hf, tokenizer=tok)


def load_model_from_local(path, model_name):
    """Dispatch the offline local-dir load by --model-name (recorded diff,
    issue #12; gpt2-small added per issue #16). Defaults preserve the
    original gpt2-medium behaviour."""
    if model_name == "pythia-410m":
        return _load_pythia_from_local(path)
    if model_name == "gpt2-medium":
        return _load_medium_from_local(path)
    if model_name == "gpt2":
        return _load_small_from_local(path)
    raise ValueError(f"No local-load route for model {model_name!r}")


def main():
    """Run the selected tier's arms and write results + terminal artifacts."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=TIERS, required=True)
    ap.add_argument("--arms", default=None, help="comma-separated override, e.g. A0,A4")
    ap.add_argument("--harness-check", action="store_true",
                    help="random-init toy model; validates the harness, draws no verdicts")
    ap.add_argument("--model-path", default=None,
                    help="local dir with model files for offline load")
    # EXP_012-PYTHIA spec §3 (recorded diff, issue #12): model selection.
    # Default reproduces prior behaviour exactly.
    ap.add_argument("--model-name", choices=["gpt2-medium", "pythia-410m", "gpt2"],
                    default="gpt2-medium",
                    help="which model to load (default gpt2-medium)")
    ap.add_argument("--record-natural-norms", action="store_true",
                    help="record natural per-layer resid_pre norms for every "
                         "prompt even under renorm=seed_j (EXP_012-PYTHIA "
                         "spec §4: per-arm seed_j/natural_i ratio record)")
    # Seed / subset / artifact-suffix parameters (EXP_010c-ROBUST spec §3,
    # issue #11). All defaults reproduce the registered behaviour exactly, so
    # every prior command line still replays.
    #
    # MERGE NOTE (EXP_010c-ROBUST issue #11 and EXP_010c-3b issue #21 added
    # overlapping flags in parallel). Both interfaces are kept so the run
    # commands recorded against each experiment's committed artifacts still
    # work; they share ONE implementation underneath:
    #   --subset B      == --prompt-offset 25   (verified equal, see below)
    #   --out-suffix X  == --tag X
    ap.add_argument("--seed", type=int, default=42,
                    help="global torch seed (registered runs used 42; EXP_010c3b "
                         "§2a measured this to be a no-op in this harness)")
    ap.add_argument("--subset", choices=["registered", "B", "pythia", "small"],
                    default="registered",
                    help="prompt subset: registered round-robin 25, disjoint B "
                         "(equivalent to --prompt-offset 25), the EXP_012-PYTHIA "
                         "core8+17 set, or the EXP_010b 5-Divine+20 Small set")
    ap.add_argument("--prompt-offset", type=int, default=0,
                    help="offset into the deterministic round-robin ordering; "
                         "25 gives the disjoint subset (EXP_010c3b §2b). Applies "
                         "to the round-robin subsets only (registered / B)")
    ap.add_argument("--n-prompts", type=int, default=None,
                    help="override the tier's prompt count")
    ap.add_argument("--out-suffix", default=None,
                    help="artifact suffix override (e.g. robust_seed1337); "
                         "default keeps the tier-based naming")
    ap.add_argument("--tag", default=None,
                    help="alias of --out-suffix. REQUIRED for variant runs so they "
                         "cannot overwrite the registered artifacts.")
    # EXP_010c-VARIANTS spec §2 (recorded diff, issue #14): energy-rescale
    # target. Default reproduces the registered convention exactly.
    ap.add_argument("--renorm", choices=["seed_j", "natural_i"], default="seed_j",
                    help="loop rescale target: seed norm at extraction layer j "
                         "(registered) or natural resid_pre norm at injection layer i")
    args = ap.parse_args()
    tier = TIERS[args.tier]
    arms = args.arms.split(",") if args.arms else tier["arms"]
    n_prompts = args.n_prompts if args.n_prompts is not None else tier["n_prompts"]
    # Artifact suffix: every output filename below is built from this, so it must
    # stay defined ahead of the first use (the subset-B audit file). --tag and
    # --out-suffix are aliases; a --harness-check run gets its own name so it can
    # never land on a registered artifact.
    suffix = (args.tag or args.out_suffix
              or (f"{args.tier}_harness" if args.harness_check else args.tier))
    # A variant run must never land on the registered artifact names. Documenting
    # --tag as "required" did not enforce it: any parameter below changes the
    # results while the suffix still defaults to the tier name, so a variant could
    # silently overwrite results_<tier>.json / terminals_<tier>.pt. Checked here,
    # BEFORE the model load, so it fails in a second rather than after a minute.
    # (PR #33 review, data-integrity finding; same principle as the subset-B
    # audit-file guard further down.)
    # NOTE: --harness-check is NOT an exemption. Its artifacts are suffixed
    # `<tier>_harness`, but those are committed files too, so a variant harness
    # run (e.g. --harness-check --subset B) silently rewrote the registered
    # results_smoke_harness.json / terminals_smoke_harness.pt. Observed, not
    # hypothesised — it happened twice while verifying this merge.
    # "Registered" is per tier, not globally: the pythia and small tiers are
    # BOUND to their own subset below, so for them --subset pythia/small IS the
    # registered configuration. Comparing against the literal "registered"
    # instead made those tiers' own documented run commands unrunnable without
    # --tag — i.e. it blocked the EXP_010b and EXP_012-PYTHIA registered runs.
    tier_subset = {"pythia": "pythia",
                   "small010b": "small", "small_smoke": "small"}.get(
                       args.tier, "registered")
    if not (args.tag or args.out_suffix):
        variant = [
            f"--seed {args.seed}" if args.seed != 42 else None,
            f"--subset {args.subset}" if args.subset != tier_subset else None,
            f"--prompt-offset {args.prompt_offset}" if args.prompt_offset else None,
            f"--n-prompts {args.n_prompts}" if args.n_prompts is not None else None,
            f"--renorm {args.renorm}" if args.renorm != "seed_j" else None,
        ]
        variant = [v for v in variant if v]
        if variant:
            ap.error(
                "non-registered configuration (" + ", ".join(variant) + ") would "
                f"write to the registered artifact names results_{args.tier}.json / "
                f"terminals_{args.tier}.pt. Pass --tag (or --out-suffix) to give "
                "this run its own artifacts.")
    if args.tier == "pythia":
        # EXP_012-PYTHIA spec §4 promises the natural-norm record for the
        # registered seed_j run — implied, not flag-dependent (PR #39 review).
        args.record_natural_norms = True
        # PR #39 review round 2: bind the tier to its model and subset so the
        # gpt2-medium defaults cannot silently produce mislabeled artifacts.
        if args.model_name != "pythia-410m":
            ap.error("--tier pythia requires --model-name pythia-410m")
        if args.subset != "pythia":
            ap.error("--tier pythia requires --subset pythia")
    if args.tier in ("small010b", "small_smoke"):
        # EXP_010b spec §5/§6: the norm-ratio record is part of the registered
        # run, and the small tiers are bound to their model and subset so the
        # gpt2-medium defaults cannot silently produce mislabeled artifacts.
        args.record_natural_norms = True
        if args.model_name != "gpt2":
            ap.error(f"--tier {args.tier} requires --model-name gpt2")
        if args.subset != "small":
            ap.error(f"--tier {args.tier} requires --subset small")
    elif args.tier != "pythia" and args.model_name != "gpt2-medium":
        # The converse of the two bindings above (PR #33 review): the pythia and
        # small tiers are pinned to their models, but nothing stopped a
        # gpt2-medium tier from being run against a different model and writing
        # the result into that tier's registered artifact name.
        ap.error(f"--tier {args.tier} is a gpt2-medium tier; --model-name "
                 f"{args.model_name} would write non-medium results to "
                 f"results_{args.tier}.json")

    torch.manual_seed(args.seed)
    if args.harness_check:
        print("HARNESS CHECK — random-init toy model, results carry no verdict weight.")
        model = _toy_model()
    elif args.model_path:
        print(f"Loading {args.model_name} from local path {args.model_path} (offline) ...",
              flush=True)
        model = load_model_from_local(args.model_path, args.model_name)
    else:
        from transformer_lens import HookedTransformer
        print(f"Loading {args.model_name} ...", flush=True)
        model = HookedTransformer.from_pretrained(args.model_name)
    model.eval()

    # --subset B and --prompt-offset 25 are the same request (see MERGE NOTE);
    # conflicting explicit values are an error rather than a silent precedence.
    if args.subset == "B" and args.prompt_offset not in (0, 25):
        ap.error(f"--subset B implies --prompt-offset 25, got {args.prompt_offset}")
    # The pythia and small subsets have their own derivations, not the shared
    # round-robin ordering, so an offset into that ordering is meaningless
    # there. Same principle as above: reject it rather than ignore it silently.
    if args.subset in ("pythia", "small") and args.prompt_offset != 0:
        ap.error(f"--prompt-offset does not apply to --subset {args.subset} "
                 f"(it indexes the round-robin ordering); got {args.prompt_offset}")
    offset = 25 if args.subset == "B" else args.prompt_offset
    if args.subset == "pythia":
        prompts = select_subset_pythia(n_prompts)
    elif args.subset == "small":
        prompts = select_subset_small(n_prompts)
    else:
        # Covers both `registered` (offset 0 by default) and `B` (offset 25).
        # select_subset_b(n) == select_subset(n, offset=25) — verified equal and
        # made to delegate in derive_prompts.py, so this is the one
        # implementation behind both interfaces.
        prompts = select_subset(n_prompts, offset=offset)
    subset_b_records = prompts if offset == 25 else None
    if args.harness_check:
        prompts = [dict(rec, prompt=_toy_tokens(rec["prompt"])) for rec in prompts]
    print(f"Tier={args.tier} arms={arms} prompts={len(prompts)} subset={args.subset} "
          f"prompt_offset={offset} seed={args.seed} renorm={args.renorm} "
          f"max_iter={tier['max_iter']} check_start={tier['check_start']}")

    outdir = HERE / "output"
    outdir.mkdir(exist_ok=True)
    if subset_b_records is not None:  # audit record of the executed disjoint subset
        # The canonical audit file records the FULL 25-prompt subset B that the
        # registered EXP_010c-ROBUST runs used. A partial or resized run (e.g.
        # --tier smoke, or --n-prompts) must not overwrite it with a truncated
        # list — that silently invalidates another experiment's audit record.
        # Found the hard way: a 2-prompt smoke test clobbered the committed
        # 25-entry file. Same principle as --tag on the results artifacts.
        if len(subset_b_records) == 25:
            (outdir / "prompt_subset_b.json").write_text(
                json.dumps(subset_b_records, indent=2))
        else:
            (outdir / f"prompt_subset_b_{suffix}.json").write_text(
                json.dumps(subset_b_records, indent=2))
    if args.renorm == "natural_i" or args.record_natural_norms:
        # Reference-norm record (spec §4, issue #14; also EXP_012-PYTHIA §4,
        # issue #12, via --record-natural-norms): natural per-layer
        # resid_pre norms for every prompt, one un-hooked pass each.
        norm_rec = {}
        for rec in prompts:
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    rec["prompt"], names_filter=lambda n: n.endswith("hook_resid_pre"))
            norm_rec[rec["id"]] = {
                str(l): round(cache[f"blocks.{l}.hook_resid_pre"][0].norm().item(), 4)
                for l in range(model.cfg.n_layers)}
        (outdir / f"natural_resid_norms_{suffix}.json").write_text(
            json.dumps(norm_rec, indent=2))
        print(f"Natural per-layer resid_pre norms recorded -> "
              f"natural_resid_norms_{suffix}.json", flush=True)
    results, terminals = [], {}
    t0 = time.time()
    for arm in arms:
        i, j = ARMS[arm]
        inject_hook = ARM_INJECT_HOOK.get(arm)
        print(f"\n=== Arm {arm}: window {i}->{j}"
              f"{' inject_hook=' + inject_hook if inject_hook else ''} ===", flush=True)
        for rec in prompts:
            p = rec["prompt"]
            p_text = p if isinstance(p, str) else "harness-check-tokens"
            r = run_arm_with_terminal(model, p, i, j, tier["max_iter"], tier["check_start"],
                                      inject_hook_name=inject_hook, renorm=args.renorm)
            # string keys ("ARM|PROMPT_ID") so the .pt loads with weights_only=True
            terminals[f"{arm}|{rec['id']}"] = {
                "mean": r.pop("terminal_mean_vec"),
                "last": r.pop("terminal_last_vec"),
            }
            r.update(arm=arm, window=f"{i}->{j}", prompt_id=rec["id"], prompt=p_text,
                     category=rec["category"])
            results.append(r)
            print(f"  [{arm}] {rec['id']:<16} -> {r['terminal_token']!r:14} "
                  f"lock={r['lock_in_iter']} iters={r['n_iters']} "
                  f"margin={r['top_logit_margin']:.2f}", flush=True)
        # checkpoint after every arm
        json.dump(results, open(outdir / f"results_{suffix}.json", "w"), indent=2)
        torch.save(terminals, outdir / f"terminals_{suffix}.pt")

    # summary table
    print(f"\n=== Summary ({time.time()-t0:.0f}s) ===")
    for arm in arms:
        rs = [r for r in results if r["arm"] == arm]
        toks = sorted({r["terminal_token"] for r in rs})
        conv = sum(r["converged"] for r in rs)
        print(f"  {arm} {ARMS[arm][0]:>2}->{ARMS[arm][1]:<2} converged {conv}/{len(rs)} "
              f"unique_terminals={len(toks)} {toks[:8]}")
    print(f"\nArtifacts: results_{suffix}.json, terminals_{suffix}.pt in {outdir}")


if __name__ == "__main__":
    main()
