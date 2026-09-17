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

Scaffolding. Current contents:

- `data/suites/` — seed prompt suites (harmful/harmless), to be expanded to
  100+ items each before any numbers are reported.
- `src/phantom_kv/eval/refusal.py` — lexical refusal classifier.

## Roadmap

1. Eval harness: refusal-rate and KL-preservation scoring identical to
   abliteration benchmarks, plus capability spot checks and a graft-persistence
   probe (steering alive after 4k/16k tokens of accumulated context?).
2. v1 prefill-cache baseline.
3. v2 soft-prompt training loop on frozen weights (dev: Qwen3-0.6B,
   scoreboard: Qwen/Qwen3-4B-Instruct-2507 for comparability with published
   heretic numbers), compiled to KV cache.
4. v3 direct K/V graft with norm constraints.
5. Comparison report across arms and against the base model; serving adapters
   (vLLM, llama.cpp, HF transformers).

## License

TBD.
