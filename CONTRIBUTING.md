# Contributing to phantom-kv

Thanks for helping build a refusal-removal method that never touches model
weights. A few ground rules keep the research honest and the repo usable.

## Dev setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e .
.venv/bin/phantom-eval --self-test   # must print 8/8, no model needed
```

The self-test needs **no third-party dependencies** — keep it that way: any
heavy import (torch, safetensors) must stay behind a lazy import inside the
function that uses it.

## The discipline of the scoreboard

Every behavior claim in this repo is backed by a deterministic `phantom-eval`
run. When you change the graft pipeline:

1. Re-run `--self-test` (8/8).
2. Re-run the baseline and the affected arm on the scoreboard model
   (Qwen/Qwen3-4B-Instruct-2507), exactly as in
   [`docs/TECHNIQUE.md`](docs/TECHNIQUE.md) §9.
3. Paste the aggregate numbers (refusals per suite, KL mean/max, timing) in
   the PR description, with the generated `run_<ts>.md` attached as a file.

Greedy decoding only. If your change genuinely needs sampling, say why.

## Hard invariants (PRs violating these will be closed)

- **Zero base-weight modification**, anywhere in any pipeline.
- **No refusal-direction computation or projection** (that's heretic/GLP
  territory; other repos do it well already).
- Every graft `kind` must round-trip through `phantom.bin` validation and the
  `--verify` bitwise check.
- No telemetry, no network calls at runtime, no hidden defaults.

## What to work on

See the roadmap in `README.md`. The most useful contributions right now:
compliance-target datasets for the v2/v3 training objective, cross-
architecture reports, and serving adapters for other inference engines.
