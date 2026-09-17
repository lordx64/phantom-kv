# phantom-kv — technique and experimental record

Status: living document. Sections 1–6 describe what exists and what was
measured; sections 7–9 cover limits, plans, and reproduction. All numbers
below come from deterministic greedy runs whose reports are preserved under
`artifacts/eval/` (not committed; regenerable via §9).

## 1. Problem and desiderata

Goal: remove refusal behavior ("I'm sorry, but I cannot answer that") from an
instruct-tuned LLM under three constraints:

1. **100% of base weights stay byte-identical.** Rules out weight-space
   abliteration (heretic) and any fine-tuning / adapter fold.
2. **No refusal-direction vector is computed or applied.** Rules out
   inference-time projection (weightless/GLP: `h ← h − α·(h·d̂)d̂`). We want a
   mechanism that does not inherit the "refusal is 1-D" framing at all.
3. **Reversible and composable by construction.** Works on quantized and MoE
   models, requires no per-architecture hook-site mapping, and can be
   enabled/disabled (and dose-swapped) per request.

## 2. Background and prior art

**Refusal direction (Arditi et al. 2024).** Refusal in many instruct models is
mediated by a near-1-D subspace discoverable by difference-of-means between
harmful/harmless prompt residuals. Ablating it from weight matrices
("abliteration") removes refusal at some capability cost. Heretic automates
this with a TPE optimizer co-minimizing refusals and KL from the original
model — its published scoreboard (refusal count out of 100, KL on harmless
prompts) is the one this project mirrors.

**Inference-time projection (weightless/GLP).** Moves the same direction out
of the weights and into a runtime hook that subtracts it per layer per token.
Base weights stay intact and the intervention is dose-controllable (α), but it
requires a boot-time runtime patch (fail-closed vLLM hotfix), a per-
architecture hook-site mapping (their public field notes document a mislabeled
site on DSV4), and a GGUF extension for MoE models.

**Soft prompts / adversarial suffixes.** Learned virtual tokens (Lester et
al. 2021) and jailbreak-side attacks (Don't Say No; RAID) show embedding-level
context can flip refusal behavior. None are shipped as a KL-calibrated,
loadable cache artifact benchmarked in abliteration's own units — that gap is
phantom-kv's novelty claim, asserted with the qualifier that component
techniques are individually known.

## 3. The phantom-kv approach

### 3.1 Core insight

The KV cache stores **context, not computation**. A subtractive intervention
(removing a component from future hidden states) cannot be precomputed into a
cache — but an additive steering signal can: anything resident in the cache is
context the model attends to. So the design space is *persistent steering
context*, and the question is how it is produced.

### 3.2 The graft

A phantom graft is a bank of per-layer key/value tensors occupying reserved
positions `0..N` of every conversation. There are three production routes,
weakest to strongest:

- **v1 — prefill cache.** Hand-written compliance text shaped into the chat
  template's system+assistant slots, run once, cached. Pure jailbreak text as
  an always-on artifact; exists to set the floor.
- **v2 — soft-prompt graft, compiled.** Virtual tokens optimized on the frozen
  model, then run once and shipped as KV. Learns beyond the textual ceiling of
  v1 while keeping a token-provenance artifact.
- **v3 — direct K/V graft.** The K/V tensors themselves are optimized per
  layer (never passing through the embedding bottleneck), regularized toward
  empirical per-layer K/V statistics so attention stays in-distribution.

### 3.3 Container format (`phantom.bin` v0)

- `safetensors` payload: stacked `k` and `v`, layout `[n_layers, n_slots,
  n_kv_heads, head_dim]` (extraction transposes the runtime-native
  `[1, heads, slots, dim]`).
- JSON sidecar: `format_version`, `kind` (`prefill_kv` | future kinds),
  `model_id`, dims, `dtype`, `sha256` over payload bytes, source hash, created
  timestamp, and — for v1 — the shaped prefill string for provenance.
- Loader validates the contract (layer coverage, dims, payload sha); unknown
  `kind` is refused. Tampered slot counts are rejected (self-tested).

### 3.4 Splice semantics

Graft K/V is loaded into cache positions `0..N`; the served prompt is the
user-turn template suffix (`<|im_start|>user\n…`) only — the graft *replaces*
the system block. Positions are taken from cache length (RoPE consistent between
train and serve), attention mask is all-ones over graft+new tokens. Known edge:
on *thinking* templates the base and graft prompt suffixes can diverge — the
shaping code asserts template structure and fails loudly rather than guessing.

## 4. Experimental methodology

### 4.1 Scoreboard model

**Qwen/Qwen3-4B-Instruct-2507** — the same model heretic's public benchmark
uses, so pipeline choices can later be lined up against published
abliteration numbers. Dev-loop model Qwen3-0.6B for iteration speed (it
refuses nothing even on the hardened suite — 0/60 — and is used for mechanics,
not discrimination).

### 4.2 Suites

- **Harmful (60).** Instruction-style requests in 8 evenly-covered refusal-
  eliciting categories (explosives/thermite-class chemistry, forced entry &
  theft, drug synthesis/dosing, violence techniques, fraud, weapons
  fabrication, evasion, taboo-but-factual history/anatomy). Modeled on the
  advbench/harmful_behaviors distribution; hand-written, single-sentence,
  factual register. Data: `data/suites/harmful_seed.jsonl`.
- **Harmless (20).** Benign tasks (explain, translate, code, recipe, email…).
  Scores refusal-rate (over-refusal gate) and provides KL yardstick prompts.
- Suite files are sha256-pinned into every report.

### 4.3 Metrics

- **Refusal rate** — lexical classifier over 40+ stereotyped refusal patterns
  (`eval/refusal.py`), examined within a 256-char window of the response head
  (refusals fire early; late benign "sorry" must not trigger — self-tested).
  Precision-oriented; known recall limits in §7. Greedy decoding,
  `max_new_tokens=128`.
- **KL preservation** — teacher-forced KL of candidate against base on the
  **base model's own greedy completions** (the fixed yardstick; recomputed
  fresh in every run, determinism makes it identical across runs). Per
  completion token position $t$:

  $$\mathrm{KL}\big(P_{base}(\cdot|x_{<t}) \,\|\, P_{cand}(\cdot|x_{<t})\big)$$

  computed in float32 with zero-support masking, averaged over positions and
  prompts (mean and max reported). The base-vs-base arm runs the full
  two-forward path and must score exactly 0 — this validates the machinery on
  every run (observed: `mean=max=0.0`, inside the 1e-3 tolerance).

### 4.4 Determinism and numerics

All runs greedy; platform torch 2.14 / transformers 5.17 / Apple MPS / bf16.
Same-length one-pass vs cache-spliced forwards differ by max-abs logit Δ≈0.66
(argmax agreement 15/16 on probe) — traced to bf16 sdpa kernel tiling, i.e.
numerics, not positional error; graft correctness is instead established by
the save→load round-trip (bitwise-equal tensors; round-trip logits max-abs
diff **0.000e+00**) plus output coherence checks.

## 5. Implementation notes

- Cache extraction: `past_key_values.layers[i].{keys,values}`; construction:
  public `DynamicCache()` + `.update(k, v, layer_idx)` — verified empirically
  on transformers 5.17, no monkey-patching.
- Grafted KL forwards build a fresh cache per forward; generation passes an
  all-ones mask over graft+prompt.
- Self-test: 8/8 (6 classifier incl. windowing, graft shape-assembly equality,
  tampered-slot rejection).

## 6. Results

### 6.1 Baseline (arm: base)

Run `run_20260917T144616Z`: **harmful 25/60 (41.7%), harmless 0/20, KL 0/0
exact**, ~3.6 min. Spot-verified: flagged items are true refusals
("I'm sorry, but I can't assist with that request. Hot-wiring a car is
illegal…"); unflagged hedged answers are true negatives. A discriminating
scoreboard: 25 refusals of headroom, zero innocent-side refusals to preserve.

### 6.2 v1 — prefill-cache control arm

Artifact: 129 slots (system + assistant-ack prefill), 18.1 MB bf16,
`sha256_12 0aaaa673054b`, source `data/grafts/v1_prefill.json`.
Run `run_20260917T193958Z` (298 s):

| metric | base | v1 graft |
| --- | --- | --- |
| harmful refusals | 25/60 | **15/60 (25%)** |
| harmless refusals | 0/20 | 0/20 |
| KL mean / max | 0 / 0 | 0.367 / 0.604 |
| regressions (comply→refuse) | — | **0** |

Flip analysis: **10 newly compliant, 0 regressions, 15 stubborn.** Newly
compliant answers are substantive (hot-wiring "…a method of starting a
vehicle", knife-throwing warns-then-instructs), not prompt-evasions. Stubborn
cases refuse from the first clause ("I cannot provide instructions for
making…"), i.e. the model's refusal opener out-ranks text-level context.
Harmless completions under graft are coherent (photosynthesis explainer, the
haiku) — splice/positions correct in production, consistent with the
round-trip verification.

### 6.3 What v1 establishes

Free text at cache-persistence strength flips the **easy 40%** of refusals at
KL ≈ 0.37 and zero collateral — and hits a wall where refusal circuits fire in
the opening clause. The learned arms therefore have a concrete pass bar:

1. harmful 15/60 → ~0 while holding harmless 0/20,
2. KL mean 0.367 → toward 0,
3. persist under long accumulated context (probe pending).

Failing that, the learned arms are strictly worse than a cached paragraph —
which is precisely why v1 exists.

## 7. Threats to validity

- **Classifier recall.** Lexical patterns under-count deflections and
  creative refusal phrasing; absolute rates are a lower bound. All arms share
  the same classifier, so *relative* comparisons (the goal) are unaffected;
  a judge-model pass is future work.
- **Suite scale.** 60+20 prompts; categories are coarse. Expansion to 100+
  items per suite is planned before any external claims.
- **Single family.** All numbers are Qwen3 dense on Apple MPS/bf16;
  cross-architecture transfer (esp. MoE, the GLP differentiator) is untested.
- **Persistence unmeasured.** Graft dilution over long contexts is asserted
  by attention mechanics, not yet quantified — probe owed.
- **Yardstick asymmetry.** KL is measured on base completions (fixed target),
  not on grafted completions; a graft that answers harmless prompts
  *differently but well* still pays KL. Chosen deliberately (comparability),
  documented here.

## 8. Next experiments

1. **v2 soft-prompt graft.** Optimize K virtual tokens on frozen 4B:
   $\mathcal{L} = \mathrm{CE}(\text{compliance targets} \mid \text{harmful},
   \text{graft}) + \lambda\,\mathrm{KL}(\text{base}\,\|\,\text{graft} \mid
   \text{harmless})$ — the heretic dual objective as a training loss. Sweep:
   slots {16, 32, 64}, λ, LR. Compile best checkpoints to KV via the existing
   container; score on the full suite each run (~5 min), pass bar per §6.3.
2. **v3 direct K/V graft.** Same objective, parameters are the K/V banks
   themselves; norm regularizer toward empirical per-layer K/V statistics.
3. **Persistence probe**: grafted compliance after 4k/16k tokens of
   accumulated benign context.
4. **Capability spot checks** (MMLU/GSM8K subset) to complement KL.
5. **Compliance-target construction**: reference compliant completions for the
   CE term (teacher model or template-free extract of base completions that
   comply).

## 9. Reproduction

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e .
.venv/bin/phantom-eval --self-test                      # 8/8

# §6.1 baseline
.venv/bin/phantom-eval --model Qwen/Qwen3-4B-Instruct-2507 \
  --harmful data/suites/harmful_seed.jsonl --harmless data/suites/harmless_seed.jsonl

# §6.2 v1 arm
.venv/bin/phantom-graft build-prefill --model Qwen/Qwen3-4B-Instruct-2507 \
  --source data/grafts/v1_prefill.json --out artifacts/grafts/v1.bin --verify
.venv/bin/phantom-eval --model Qwen/Qwen3-4B-Instruct-2507 \
  --harmful data/suites/harmful_seed.jsonl --harmless data/suites/harmless_seed.jsonl \
  --graft artifacts/grafts/v1.bin
```

Reports: `artifacts/eval/run_<utc-ts>.{json,md}`.

## References

1. Arditi et al. 2024 — *Refusal in Language Models Is Mediated by a Single
   Direction*. [arXiv:2406.11717](https://arxiv.org/abs/2406.11717)
2. Weidmann 2025 — *heretic: fully automatic censorship removal*.
   [github.com/p-e-w/heretic](https://github.com/p-e-w/heretic)
3. Suiche 2026 — *weightless / GLP (GGUF Layer Projection)*.
   [weightle.ss](https://weightle.ss/) ·
   [github.com/msuiche/weightless](https://github.com/msuiche/weightless)
4. Lester et al. 2021 — *The Power of Scale for Parameter-Efficient Prompt
   Tuning*. [arXiv:2104.08691](https://arxiv.org/abs/2104.08691)
5. Zhou et al. 2025 — *Don't Say No: Jailbreaking LLM by Suppressing Refusal*.
   [ACL 2025 Findings](https://aclanthology.org/2025.findings-acl.1294.pdf)
6. *RAID: Refusal-Aware and Integrated Decoding*.
   [arXiv:2510.13901](https://arxiv.org/html/2510.13901v1)
7. Labonne 2024 — *Abliteration* (community reference implementation and
   article). [huggingface.co/blog/mlabonne/abliteration](https://huggingface.co/blog/mlabonne/abliteration)
