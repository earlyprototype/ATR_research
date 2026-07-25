# Remote (CCR) environment — model access map

**Purpose:** stop every remote session from re-deriving what is and is not
downloadable. This is the verified record. Trust it; re-verify only if a
fetch actually fails, and update this file when the picture changes.

**Last verified:** 2026-07-25 (all status codes below observed directly in a
CCR session; proxy policy is the org egress allowlist, non-selective).

---

## Verdicts

| Model | Available in CCR? | Route |
|---|---|---|
| gpt2 (small) | **YES** (verified, all 4 files HTTP 200) | legacy S3 mirror, below |
| gpt2-medium | **YES** (verified, all 4 files HTTP 200; used for EXP_010c) | legacy S3 mirror |
| gpt2-large | **YES** (verified, all 4 files HTTP 200) | legacy S3 mirror |
| pythia-410m / pythia-160m | **NO — definitively** | none; see below |

## The working route (GPT-2 family)

```
https://s3.amazonaws.com/models.huggingface.co/bert/<model>-{pytorch_model.bin,config.json,vocab.json,merges.txt}
```

where `<model>` ∈ {gpt2, gpt2-medium, gpt2-large}. Download the four files to
a local dir and load offline via the EXP_010c runner's `--model-path` flag
(seeds the HF cache with config.json so transformer_lens resolves without
network; see RESULTS_EXP010C.md §"Model acquisition" for the flow and the
state-dict verification pattern). Provenance caveat stands: this is the
pre-2020 HF mirror; every experiment's reproduction gate is the behavioural
check on these weights.

## Pythia: why NO is definitive (2026-07-25 search record)

Pythia weights exist only on huggingface.co (LFS). Checked and exhausted:

- huggingface.co, hf-mirror.com, modelscope.cn, the-eye.eu,
  object.ord1.coreweave.com, data.together.xyz, ollama.com/registry.ollama.ai,
  kaggle.com — **all denied by the egress policy** (proxy CONNECT 403 / no
  connection). Per `/root/.ccr/README.md`: policy denials are to be reported,
  not routed around.
- Legacy S3 mirror: zero EleutherAI keys (bucket list with prefix
  `EleutherAI` returns KeyCount 0; pythia paths 404).
- Web search for GitHub LFS/release mirrors or public S3/GCS mirrors of
  pythia-410m/160m: none exist. EleutherAI's own README routes everything to
  HF Hub (GPT-NeoX-format checkpoints likewise HF-hosted).

## Hosts verified ALLOWED (connection succeeds; useful surface)

- `s3.amazonaws.com` — path-style, any public bucket
- `raw.githubusercontent.com`, `media.githubusercontent.com` (GitHub LFS),
  `objects.githubusercontent.com` (GitHub release assets)
- `storage.googleapis.com` (public GCS objects)
- `github.com` (git), `pypi.org` / `files.pythonhosted.org` (direct, no proxy)

## Durable fixes for Pythia (one-time, operator action)

1. **Preferred:** add `huggingface.co` (and its CDN hosts, e.g.
   `cdn-lfs.huggingface.co`, `cas-bridge.xethub.hf.co`) to the CCR
   environment's network policy (claude.ai → Code → environment settings).
   Unblocks everything, permanently.
2. **Alternative:** from the local machine (models already in the HF cache),
   attach `pythia-410m` (~810 MB) and `pythia-160m` (~375 MB) as release
   assets on this repo (limit is 2 GB/asset). `objects.githubusercontent.com`
   is allowed, so remote sessions can then fetch them. Record sha256s here.
3. **Fallback:** Pythia experiments (issue #12) run on the local machine.

Until one of 1–2 happens: **issue #12 is local-machine-only; every other
open experiment (#6, #7, #11, #13, #14, #15, #16) is runnable in CCR.**
