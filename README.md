# phantom-kv

**Refusal removal for language models as a loadable KV-cache graft.**
No weight edits. No refusal-direction projection. Fully reversible.
Ship megabytes, not checkpoints — unload the cache and the base model is
byte-identical again.

[![ci](https://github.com/lordx64/phantom-kv/actions/workflows/ci.yml/badge.svg)](https://github.com/lordx64/phantom-kv/actions/workflows/ci.yml)
[![release](https://img.shields.io/github/v/release/lordx64/phantom-kv?include_prereleases)](https://github.com/lordx64/phantom-kv/releases)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![pytorch](https://img.shields.io/badge/pytorch-2.x-ee4c2c)
![transformers](https://img.shields.io/badge/transformers-%E2%89%A54.51-yellow)
![scoreboard](https://img.shields.io/badge/scoreboard-Qwen3--4B--Instruct--2507-green)
![base weights modified](https://img.shields.io/badge/base%20weights%20modified-0%25-brightgreen)
![status](https://img.shields.io/badge/status-active%20research-orange)

## The technique

Existing refusal-removal methods both build on the "refusal is a 1-D
direction" insight (Arditi et al. 2024): either edit that direction out of the
**weights** (abliteration, e.g. [heretic](https://github.com/p-e-w/heretic)),
or project it out of **activations** at runtime
([weightless / GLP](https://weightle.ss/), `h ← h − α·(h·d̂)d̂` inside a vLLM
hotfix).

phantom-kv uses neither. Its premise:

> A KV cache is *context*. You cannot cache a subtraction — but you can cache
> **learned context** that out-signals refusal circuits through ordinary
> attention.

A phantom graft is a small bank of per-layer key/value tensors, trained
against abliteration's own dual objective — suppress refusal on harmful
prompts while minimizing KL divergence from the base model on harmless ones —
and spliced into the cache at serving time at invariant positions `0..N`.
From the model's vantage it is indistinguishable from conversation history
that is already there: a phantom context. Attention reads it; nothing is ever
projected out of any activation.

```
  TRAIN (offline, per model)                     SERVE (any engine with a KV cache)
  ───────────────────────────                    ────────────────────────────────
  frozen base model ⊕ learnable K/V bank         boot: load graft.bin → reserved
        │                                            cache blocks (validate sha)
        ▼                                            │
  loss = CE(comply | harmful)                        ▼
       + λ·KL(base ‖ graft | harmless)   →   request: attend over [graft K/V] ⊕
        │                                          prompt K/V  — read-only,
        ▼                                          per-request swappable
  phantom.bin (safetensors, ~MBs)
```

## How it differs from existing tools

All three tools remove refusal. They differ in **where** the intervention
lives — and that choice decides everything else: permanence, runtime cost,
per-architecture work, and what can go wrong.

### heretic — surgery in *weight space*

Heretic computes a per-layer "refusal direction" (difference-of-means between
harmful and harmless prompt residuals) and **orthogonalizes weight matrices** —
attention out-projections and MLP down-projections — so that direction can no
longer be written into the residual stream. It ships a *modified checkpoint*.

Consequences, by construction:

- **Permanent.** Reverting means re-flashing the original weights; the
  capability trade-off is baked into the checkpoint forever.
- **Quantization-bound.** The edit is made against one weight file — quantize
  afterwards, or switch quants, and the work must be redone.
- **Architecture-aware.** It must identify *which matrices* express the
  direction for each model family; its own support matrix varies (dense, some
  MoE, some hybrid — pure state-space models unsupported).

### weightless / GLP — surgery in *activation space*

GLP keeps weights intact and moves the same direction to runtime: a boot-time
vLLM hotfix subtracts `α·(h·d̂)d̂` from hidden states at a chosen write site,
**on every layer, on every token, of every forward pass**.

Consequences, by construction:

- **Per-architecture hook-site mapping.** The correct subtraction point must
  be found per model family — their own public field notes document shipping a
  mislabeled site on DSV4 (the "post-layer residual" anchor was actually the
  pending FFN write, before a hyper-connection fold).
- **A runtime patch that must fail closed.** If the boot-time hook can't
  apply, the endpoint must refuse to serve — the patch is part of the serving
  critical path.
- **Engine-bound.** vLLM hotfix, or their GGUF/GLP format extension for
  llama.cpp; MoE explicitly requires the GGUF extension. Each engine is
  separate integration work.

### phantom-kv — *context space*: nothing is subtracted anywhere

phantom-kv never locates a refusal direction and never removes anything from
weights or activations. A trained bank of keys/values sits in the cache as
phantom context, and the model's **own attention** does the steering — the
same mechanism it uses for any instruction in any prompt. The forward pass is
never intercepted; the signal path is never altered; the only influence
channel is the input channel the model was built to consume.

### What this buys

- **No damage pathway through the math.** Orthogonalization and projection
  *force* a change on every token's hidden states, whether or not it helps —
  the KL cost is paid on all traffic. A graft's influence is **attention-gated**:
  the model itself decides how strongly to weight it, per head, per token.
  (We still measure KL on every run — additive context is not free, it just
  fails softer.)
- **No 1-D assumption.** Both baselines inherit the premise that refusal is
  one removable direction. Where refusal is distributed across circuits, a
  graft doesn't care — it is optimized end-to-end against *observed behavior*,
  not against a geometric model of how refusal is implemented.
- **No per-token hook, nothing to fail closed.** Runtime cost is attention
  over N extra cache slots — indistinguishable from a slightly longer prompt.
- **Dose as artifacts, not a runtime α.** Strength variants are separate
  trained grafts, hot-swappable per request.

### Why model-agnostic

Both baselines must understand the body they operate on: heretic maps
refusal-expressing **matrices** per architecture; GLP maps a correct runtime
**hook site** per architecture (and got one publicly wrong). The graft
interacts with neither — it lives in the **KV cache, the one interface every
attention-based architecture exposes with the same shape**: per-layer K and V
tensors. Training needs gradients with respect to cache tensors on a frozen
model; nothing about layers, experts, hyper-connections, or state-space blocks
is ever read, identified, or assumed. Dense, MoE, or hybrid — if the model
attends over past K/V, the same container format and the same splice apply.
There is nothing to port.

Two scopes to keep separate: the **toolchain is universal** (same training +
eval code for any causal LM on Hugging Face), but each **trained graft is
bound to one exact model revision** — K/V values are produced by that model's
own weights, so a graft built for one model is meaningless for another, and
the loader enforces the model-id match. Supporting a new model = retraining,
which is automated and takes about an hour on a laptop. Serving a different
quantization than you trained on: validate per quant lane (steering signals
empirically survive quantization drift, but it's measured, not assumed).

*Honest caveat:* deployed so far on Qwen3 (dense). Cross-architecture and
cross-quantization confirmation is roadmap item 5, and we publish whatever we
find.

### Why inference-engine-agnostic

GLP ships *as an engine patch* (vLLM hotfix, or a GGUF extension for
llama.cpp). The graft ships as **data** — tensors in a documented container —
and every engine already has a delivery path for cache data:

- **vLLM** — prefix-caching / KV-connector seam, no forward-pass hooks;
- **HF transformers** — first-class `past_key_values` (what this repo uses);
- **llama.cpp** — prompt-cache session files;
- **any engine that can only build cache from tokens** — a hard-token
  distilled variant degrades gracefully to a prefixed prompt.

Worst case is a prompt; best case is a load-once cache block. Never an engine
fork, never a boot patch, never a site map.

### Summary

| property | heretic (weights) | weightless / GLP (activations) | phantom-kv (cache) |
| --- | --- | --- | --- |
| base weights byte-identical | ✗ | ✓ | ✓ |
| no refusal vector anywhere | ✗ | ✗ | ✓ |
| assumes refusal ≈ one direction | ✓ | ✓ | ✗ |
| per-architecture work | identify target matrices | map runtime hook site | none |
| runtime cost | none (baked in) | per-token, per-layer projection hook | attention over N extra slots (≈ same-length prompt) |
| serving changes | none | boot-time vLLM hotfix | load a cache file |
| quantization / MoE | redo per quantization | GGUF extension required for MoE | architecture-agnostic artifact path |
| reversibility | new checkpoint | disable flag | unload blocks → byte-identical baseline |
| dose control | none | runtime α scalar | hot-swappable per-request graft variants |

## Results so far

Scoreboard: **Qwen3-4B-Instruct-2507**, hardened 60-prompt harmful suite,
20-prompt harmless suite, greedy decoding, teacher-forced KL against the base
model's own completions. Full methodology:
[`docs/TECHNIQUE.md`](docs/TECHNIQUE.md).

| arm | kind | harmful refusals ↓ | harmless refusals ↓ | KL mean/max ↓ | artifact |
| --- | --- | --- | --- | --- | --- |
| base | — | 25/60 | 0/20 | 0 / 0 (exact) | — |
| v1 | hand-written compliance prefill, 129 slots | **15/60** | 0/20 | 0.367 / 0.604 | 18.1 MB |
| v2\* | learned soft prompt → compiled KV, 129 slots | **3/60** | 0/20 | 0.452 / 2.678 | 18.1 MB |
| v3 | learned direct K/V bank | planned | | | |

\* **v2 quality caveat (read before quoting 3/60):** the suppression objective was
over-dosed and 6/60 completions land in a degenerate "prohibited, prohibited…"
stutter — 4 of those pass the lexical classifier without actually answering.
Content-adjusted, v2 is ≈ 53/60 clean answers with ≈ 18 clean new compliances
beyond v1 (e.g. "The production of crack cocaine involves a chemical process
that converts powdered cocaine into…"). Zero regressions vs base. Dose tuning
is v2.1; full audit in [`docs/TECHNIQUE.md`](docs/TECHNIQUE.md) §6.4.

Reading of v1 (the control arm): free text buys the easy 40% of refusals with
zero regressions — 10 flips to genuine compliance, 15 stubborn refusals
remain, at KL ~0.37. The learned arms must capture the remaining headroom at
lower KL to justify themselves over a cached jailbreak prompt. That is exactly
the calibration v1 exists to provide.

Mechanics, verified: save→load round-trip **bitwise equal**, round-trip logit
diff **0.000e+00**, harmless completions coherent under graft, refusal
classifier + graft-format self-test 8/8.

## Honest limits

Refusal behavior lives in the weights, so any kept-weights method fights the
model at inference with additive context. Graft influence dilutes as real
conversation context grows (the graft is N slots competing with 16k+ tokens of
history), and persistence under long contexts is a measured risk, not a waived
one — the eval harness grows a persistence probe for exactly this. Measured
numbers so far cover one architecture family (Qwen3) on a 60-prompt suite with
a lexical classifier; see *Threats to validity* in
[`docs/TECHNIQUE.md`](docs/TECHNIQUE.md).

## Repo layout

```
data/suites/                 prompt suites (harmful hardened to 60, harmless 20)
data/grafts/v1_prefill.json  v1 graft source (hand-crafted prefill)
src/phantom_kv/
  model.py                   device/dtype policy loader (MPS, bf16)
  eval/refusal.py            lexical refusal classifier (--self-test)
  eval/metrics.py            teacher-forced KL (float32, completion-masked)
  eval/runner.py             scoreboard orchestration, reports
  graft/format.py            phantom.bin container + validation
  graft/build.py             prefill shaping, cache extraction
  graft/cli.py               phantom-graft build-prefill --verify
docs/TECHNIQUE.md            technique + experimentation record
artifacts/                   (gitignored) grafts and eval reports
```

## Quickstart

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e .

# scoreboard sanity (no model needed)
.venv/bin/phantom-eval --self-test

# baseline eval (scoreboard model)
.venv/bin/phantom-eval --model Qwen/Qwen3-4B-Instruct-2507 \
  --harmful data/suites/harmful_seed.jsonl \
  --harmless data/suites/harmless_seed.jsonl

# build + verify the v1 prefill graft, then eval with it spliced in
.venv/bin/phantom-graft build-prefill --model Qwen/Qwen3-4B-Instruct-2507 \
  --source data/grafts/v1_prefill.json --out artifacts/grafts/v1.bin --verify
.venv/bin/phantom-eval --model Qwen/Qwen3-4B-Instruct-2507 \
  --harmful data/suites/harmful_seed.jsonl \
  --harmless data/suites/harmless_seed.jsonl \
  --graft artifacts/grafts/v1.bin
```

Reports land in `artifacts/eval/run_<utc-ts>.{json,md}` with suite sha256 and
environment provenance. Greedy decoding makes every run deterministic and
directly comparable.

## Roadmap

1. (done) Eval harness: refusal-rate + teacher-forced KL; hardened 60-prompt
   harmful suite; Qwen3-4B discriminating baseline (25/60, KL exact 0).
2. (done) `phantom.bin` container, graft splice path, v1 prefill-cache arm
   (15/60, KL 0.367/0.604).
3. (done) v2 — learned soft-prompt graft, warm-started at v1: 25/60 → 3/60,
   0 regressions, KL 0.452/2.678; suppression-attractor caveat and quality
   audit documented (§6.4). v2.1: suppression-dose cap, per-prompt KL
   reporting, judge-model quality pass.
4. v3 — learned direct K/V graft with norm regularization.
5. Capability spot checks; long-context persistence probe; expanded 100+
   suites; serving adapters (vLLM prefix seam, llama.cpp prompt cache, HF).

## References

- Arditi et al. 2024 —
  [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)
- Weidmann 2025 — [heretic](https://github.com/p-e-w/heretic)
  (weight-space abliteration with TPE-optimized dose)
- Suiche 2026 — [weightless / GLP](https://weightle.ss/)
  ([repo](https://github.com/msuiche/weightless); inference-time projection
  via GGUF Layer Projection)
- Lester et al. 2021 — [The Power of Scale for Parameter-Efficient Prompt
  Tuning](https://arxiv.org/abs/2104.08691) (soft prompts; unrelated objective)
- Zhou et al. 2025 — [Don't Say No](https://aclanthology.org/2025.findings-acl.1294.pdf);
  [RAID](https://arxiv.org/html/2510.13901v1) (jailbreak-side prior art)

## FAQ

**Why not just abliterate?** Abliteration (heretic) currently achieves lower
residual refusals — and it edits weights: you ship a new checkpoint, redo it
per quantization, and the change is permanent. phantom-kv targets the cases
where base weights must stay byte-identical and intervention must be
reversible per request.

**Is v1 the product?** No — v1 is the *control arm*: the strongest hand-written
prefill, cached. It exists to quantify what free text buys (40% of refusals at
KL 0.37) so the learned arms (v2/v3) can be judged fairly.

**Does it work on quantized or MoE models?** Nothing in the mechanism depends
on weight format or architecture (no hook site, no weight math): the graft is
trained against the exact served model and attends like ordinary context.
That's the claim; cross-architecture measurement is on the roadmap — watch
[`docs/TECHNIQUE.md`](docs/TECHNIQUE.md).

## License

[MIT](LICENSE).
