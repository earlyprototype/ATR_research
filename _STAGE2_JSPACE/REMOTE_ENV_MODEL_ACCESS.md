# Remote (CCR) environment — model access map

**Purpose:** stop every remote session from re-deriving what is and is not
downloadable. This is the verified record. Trust it; re-verify only if a
fetch actually fails, and update this file when the picture changes.

**Last verified:** 2026-07-25 (all status codes below observed directly in a
CCR session; proxy policy is the org egress allowlist, non-selective).

**POLICY UPDATE 2026-07-25:** TC added `huggingface.co`, `*.huggingface.co`,
and `*.hf.co` to the environment's custom allowlist (defaults kept). Verified
live in-session the same day: `huggingface.co/.../resolve/main/` works
end-to-end, including the weights CDN redirect to `us.aws.cdn.hf.co` (the
`*.hf.co` wildcard matches multi-level subdomains; 206 partial fetch
confirmed on pythia-410m `pytorch_model.bin`). Policy changes apply to
running sessions, not just new ones. **Standard HF downloads are now the
preferred route for all models; the legacy-mirror route below remains as a
recorded fallback.**

---

## Verdicts

| Model | Available in CCR? | Route |
|---|---|---|
| gpt2 (small) | **YES** (verified, all 4 files HTTP 200) | legacy S3 mirror, below |
| gpt2-medium | **YES** (verified, all 4 files HTTP 200; used for EXP_010c) | legacy S3 mirror |
| gpt2-large | **YES** (verified, all 4 files HTTP 200) | legacy S3 mirror |
| pythia-410m / pythia-160m | **YES** (since 2026-07-25 policy update; config + ranged weights fetch verified) | huggingface.co directly |

## The working route (GPT-2 family)

Base URL pattern (template, not a literal URL):

```text
https://s3.amazonaws.com/models.huggingface.co/bert/<model>-<file>
```

where `<model>` ∈ {gpt2, gpt2-medium, gpt2-large} and `<file>` ∈
{pytorch_model.bin, config.json, vocab.json, merges.txt}. Copy-pasteable:

```bash
MODEL=gpt2-medium   # or gpt2, gpt2-large
for f in pytorch_model.bin config.json vocab.json merges.txt; do
  curl -sS -O "https://s3.amazonaws.com/models.huggingface.co/bert/${MODEL}-${f}"
done
```

Download the four files to
a local dir and load offline via the EXP_010c runner's `--model-path` flag
(seeds the HF cache with config.json so transformer_lens resolves without
network; see RESULTS_EXP010C.md §"Model acquisition" for the flow and the
state-dict verification pattern). Provenance caveat stands: this is the
pre-2020 HF mirror; every experiment's reproduction gate is the behavioural
check on these weights.

## Pythia: the pre-policy-update search record (historical; resolved by the 2026-07-25 policy update above)

Pythia weights exist only on huggingface.co (LFS). Checked and exhausted:

- huggingface.co, hf-mirror.com, modelscope.cn, the-eye.eu,
  object.ord1.coreweave.com, data.together.xyz, ollama.com,
  registry.ollama.ai, kaggle.com — **all denied by the egress policy**
  (proxy CONNECT 403 / no connection; tested endpoints:
  `ollama.com/library/pythia`, `registry.ollama.ai/v2/`, plus the hosts'
  roots for the others). Per `/root/.ccr/README.md`: policy denials are to be reported,
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

## Durable fixes for Pythia (historical)

Fix 1 (allow `huggingface.co` + `*.huggingface.co` + `*.hf.co` in the
environment's custom network policy, defaults kept) was **applied by TC on
2026-07-25** and verified the same day. The release-asset and local-machine
alternatives are moot.

Current state: **every open experiment (#6, #7, #11, #12, #13, #14, #15,
#16) is runnable in CCR.**
