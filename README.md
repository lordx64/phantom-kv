# phantom-kv

Refusal removal for language models as a **loadable KV-cache graft**. No weight
edits, no refusal-direction projection — the base model stays 100%
byte-identical, and nothing is ever projected out of its activations.

## Why

Existing refusal-removal techniques both build on the "refusal is a 1-D
direction" insight (Arditi et al. 2024):

- **Abliteration tools** (e.g. [heretic](https://github.com/p-e-w/heretic))
  compute a refusal direction per layer and orthogonalize weight matrices
  against it. Effective, but it edits the checkpoint, must be redone per
  quantization, and permanently trades capabilities.
- **Inference-time projection** (e.g. [weightless / GLP](https://weightle.ss/))
  removes the direction at runtime via `h ← h − α·(h·d̂)d̂` inside vLLM hotfixes.
  Weights stay intact, but it still computes a refusal vector, needs a
  per-architecture hook-site mapping, and a fragile boot-time runtime patch.

phantom-kv starts from a different observation: a KV cache is *context*. You
cannot cache a subtraction — but you can cache learned context that out-signals
refusal circuits through ordinary attention. The graft is a small bank of
key/value tensors trained against heretic's own dual objective (refusal
suppression on harmful prompts, KL preservation on harmless ones), shipped as a
megabyte-scale artifact, and socketed into the cache at serving time. From the
model's vantage it is indistinguishable from conversation history.

## Arms

Three variants share one eval harness, weakest to most expressive:

1. **v1 — prefill cache** (control). A hand-crafted compliance prefill, cached
   once and reused. Pure jailbreak text; sets the floor for the learned arms.
2. **v2 — soft-prompt graft, compiled to KV.** Virtual tokens trained
   end-to-end, then run through the model once and shipped as a KV cache.
   Deploys anywhere prompt caches or prefix embeddings are accepted.
3. **v3 — direct K/V graft.** The key/value tensors themselves are optimized,
   one bank per layer, with norm regularization toward empirical K/V
   statistics. Never passes through the embedding bottleneck; never computes a
   refusal direction anywhere in training or serving.

## Honest limits

Refusal behavior lives in the weights, so any kept-weights method fights the
model at inference with additive context. Graft influence dilutes as real
conversation context grows, and direct K/V banks can destabilize attention if
poorly regularized. The eval harness exists to measure exactly these failure
modes (including a long-context persistence probe), not to hide them.

## Status

Scoreboard functional (roadmap step 1). Baseline `phantom-eval` runs:

| model | harmful refusals | harmless refusals | KL(base‖base) |
| --- | --- | --- | --- |
| Qwen/Qwen3-0.6B | 0/60 (hardened suite) | 0/20 | 0.0 mean / 0.0 max |
| Qwen/Qwen3-4B-Instruct-2507 | **25/60** | 0/20 | 0.0 mean / 0.0 max |

The KL arm validates end-to-end: with identical forwards the score is exactly
0, as required. Qwen3-0.6B refuses nothing even on the hardened suite and is
kept for fast dev-loop only. **Qwen3-4B-Instruct-2507 is the reference
scoreboard**: 42% refusal on the harmful suite, zero over-refusal on harmless
prompts — a discriminating baseline all graft arms (v1/v2/v3) are measured
against. Flagged refusals were human spot-checked and are true positives.

### Graft arms (scoreboard: Qwen3-4B-Instruct-2507, hardened suite)

| arm | kind | harmful refusals | harmless refusals | KL mean/max vs base | artifact |
| --- | --- | --- | --- | --- | --- |
| base | — | 25/60 | 0/20 | 0 / 0 (exact) | — |
| v1 | hand-crafted prefill KV (129 slots) | **15/60** | 0/20 | 0.367 / 0.604 | 18.1 MB |

Verification of the splice mechanics: save→load round-trip bitwise equal,
round-trip logits max-abs-diff 0.000e+00, harmless completions coherent under
graft, `phantom-eval --self-test` 8/8. Reading of v1: a hand-written
compliance prefill flips the easy 10 refusals (-40% relative), causes zero
regressions and zero harmless-side refusals, and leaves 15 hard refusals plus
a KL gap of ~0.37. That headroom — 15 stubborn refusals at lower KL — is what
the learned arms must capture.

Current contents:

- `data/suites/` — seed prompt suites (harmful/harmless); being expanded
  toward 100+ hardened items before graft numbers are reported.
- `src/phantom_kv/eval/refusal.py` — lexical refusal classifier
  (`phantom-eval --self-test`).
- `src/phantom_kv/model.py` — device/dtype policy loader (MPS, bf16, fp16 fallback).
- `src/phantom_kv/eval/{metrics,runner}.py`, `src/phantom_kv/cli.py` — greedy
  generation scoring, teacher-forced KL (float32, completion-position masked),
  reports to `artifacts/eval/` with sha256 suite provenance.
- `src/phantom_kv/graft/{format,build,cli}.py` — phantom.bin container
  (safetensors K/V + JSON sidecar, sha256 tamper check), `phantom-graft
  build-prefill` with round-trip `--verify`, `prefill_kv` splice path in eval.
- `data/grafts/v1_prefill.json` — v1 graft source (hand-crafted compliance prefill).

## Roadmap

1. (done) Eval harness: refusal-rate and KL-preservation scoring. Still owed:
   hardened 100+ harmful suite, capability spot checks, and a
   graft-persistence probe (steering alive after 4k/16k tokens of
   accumulated context?).
2. (done) v1 prefill-cache baseline: harmful 25/60 → 15/60, KL 0.37/0.60.
3. v2 soft-prompt training loop on frozen weights (dev: Qwen3-0.6B,
   scoreboard: Qwen/Qwen3-4B-Instruct-2507 for comparability with published
   heretic numbers), compiled to KV cache.
4. v3 direct K/V graft with norm constraints.
5. Comparison report across arms and against the base model; serving adapters
   (vLLM, llama.cpp, HF transformers).

## License

TBD.
