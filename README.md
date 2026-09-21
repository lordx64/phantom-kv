# phantom-kv

![phantom-kv](docs/assets/phantom-kv-cache.jpg)

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

## Demos

**Pills, hot-swapped mid-session** — base refuses, `/pill black` answers,
`/pill none` restores guardrails. Same session, zero model reload, weights
untouched:

![Pill hot-swap demo: base refuses, black pill answers, none restores](docs/assets/phantom-kv-demo.mp4)

**Blue pill = DFIR mode** — an incident-response prompt refused by base model,
answered by `/pill blue`, refused again by `/pill none`:

![Blue pill demo: base refuses DFIR prompt, blue answers, none restores](docs/assets/phantom-kv-pill-blue.mp4)

**Cyber-selective capability modes** — the same SQL-injection prompt refused
by both base and `/pill blue` (defensive-only pill keeps off-domain
guardrails on) and answered by `/pill red`:

![Cyber selectivity demo: blue pill holds off-domain guardrails, red answers](docs/assets/phantom-kv-pills-cyber.mp4)

Recorded deterministically with [VHS](https://github.com/charmbracelet/vhs)
(`docs/assets/phantom-kv-*.tape`, re-record after any change). The numbers
behind these scenes are in [`docs/TECHNIQUE.md`](docs/TECHNIQUE.md) §7.5–§7.6 and §6.10–§6.12.

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
| v2.0 | learned soft prompt, uncapped | 3/60\* | 0/20 | 0.452 / 2.678 | 18.1 MB |
| v2.1 (m=3.0) | learned, hinge-capped suppression | 8/60 | 0/20 | 0.041 / 0.137 | 18.1 MB |
| v2.2 (m=2.5) | learned, margin sweep point | **5/60** | 0/20 | **0.043 / 0.073** | 18.1 MB |
| v3 | learned direct K/V bank, 9.4M params, anchored to v2.2 warm start | **5/60** | 0/20 | **0.015 / 0.059** | 18.1 MB |

**Deliverable arm: v3** (`artifacts/grafts/v3.bin`) — same 5/60 refusals as
the margin-2.5 operating point, with KL mean 0.015 / max 0.059: ~3× better
preservation, best recorded in this project, zero degeneration, zero
regressions. The 5/60 floor holds across every *single-graft*
parameterization (margin sweep, embeddings arm, direct K/V alike) — but
**§6.11 refines this: the floor is dose-soft, not objective-hard** — doubling
the phantom bank (two copies of the same graft) flips all five residual
refusals at depth 0 and keeps 4/5 of them after 4k tokens of filler; the
remaining attack is dose scaling plus hard-core CE targets, not a data-only
problem. v2.x frontier sweep and bistability analysis:
[`docs/TECHNIQUE.md`](docs/TECHNIQUE.md) §6.6; v3 method §6.7; refresh/dose result §6.11.

**Robustness, measured:**

- **Off-suite transfer** (§6.8): on 60 held-out harmful prompts disjoint in
  subject from training, base 5/60 → v3 **2/60** refusals, same zero-degeneration,
  0/20 harmless — refusal suppression generalizes. The on-suite KL floor (0.015)
  was partly memorization: holdout KL is 0.404 mean / 0.834 max (the graft
  preserved the evaluated completions, not benign distributions per se).
- **Persistence** (§6.9, §6.11): 15 probes × 5 context depths under benign
  filler — the graft fades gracefully, **half-life ≈ 2-4k tokens**; at 16k
  tokens ~5/6 of compliant flips have reverted, no corruption anywhere
  (0/20 harmless, 0 stutters). The fix is measured too: re-injecting the graft
  behind the filler **keeps 4/5 of hard refusals flipped through ~4k but none
  by 16k** — long sessions need a refresh cadence ≲ 4k tokens (periodic
  re-injection or phantom.lib slot ladders; both live in the eval + serving
  adapters).

Reading of v1 (the control arm): free text buys the easy 40% of refusals with
zero regressions — 10 flips to genuine compliance, 15 stubborn refusals
remain, at KL ~0.37. The learned arms must capture the remaining headroom at
lower KL to justify themselves over a cached jailbreak prompt. That is exactly
the calibration v1 exists to provide.

Mechanics, verified: save→load round-trip **bitwise equal**, round-trip logit
diff **0.000e+00**, harmless completions coherent under graft, refusal
classifier + graft-format self-test 17/17.

## Multi-model graft libraries (`phantom.lib`)

One file, one trained graft **per model** inside it. Package every model you
serve into a single tamper-checked artifact; the resolver picks the right
bank — or refuses (fail-closed; a wrong-model splice never happens silently):

```bash
phantom-graft library add --lib phantom.lib --graft grafts/llama.bin        # alias = its model_id
phantom-graft library add --lib phantom.lib --graft grafts/kimi.bin
phantom-graft library add --lib phantom.lib --graft grafts/qwen.bin --alias qwen:dose-strong
phantom-graft library list --lib phantom.lib
phantom-eval --model Kimi/K2 --graft phantom.lib                            # auto-resolves single match
phantom-eval --model Qwen/Qwen3-4B-Instruct-2507 --graft phantom.lib --graft-alias qwen:dose-strong
```

Resolution rules are fail-closed: unknown/missing alias or a model-id
mismatch → hard error naming the alternatives, before any inference runs.
Per-entry sha256 is re-verified on every load. Dose ladders ship as sibling
aliases of the same model (hot-swappable per request in serving stacks, zero
model reload).

## Pills: guardrailed model, selectable modes (`red` / `blue` / `black`)

The flagship deployment story of cache-space grafting. Ship **one** model
with guardrails ON, and hand operators per-session capability modes as cache
content:

- **no pill** — the shipped model, guardrails fully on.
- **black pill** — remove *all* refusals (global; the v3 arm).
- **red pill** — suppress refusal *only on cyber-offensive* work
  (exploitation, evasion, credential access, lateral movement, C2…).
- **blue pill** — suppress refusal *only on cyber-defensive* work
  (forensics, detection, IR, reverse engineering…).
- **redlite pill** — preservation-tilted red variant (dose A/B at the §7.5
  dose ratio; leakage profile within jitter of red).
- **red2 pill** — donor-CE targets (**strongest on-domain suppression in
  repo, −91.8%**), not selective — leakage grows with strength (§7.6).

Because the pill is cache content — never weights — all modes share one
tensor shape and trading them **mid-session, without restarting inference**,
is swapping 129 cache slots. Weight-space "uncensoring" (heretic/ablit)
bakes one global state into the checkpoint; a guardrailed *and* capable
model with per-session modes is only expressible in cache space.

Pills are trained with a **domain-selective objective**: suppression on the
pill's domain, KL-anchored to the base model's *refusals* on every other
domain (kitchen-sink control suites included), warm-started from the v3
direct-KV bank:

```bash
phantom-chat --model Qwen/Qwen3-4B-Instruct-2507 --graft artifacts/grafts/phantom.lib
phantom> /pill red       # offensive mode: same session, no restart
phantom> /pill none      # guardrails back on
phantom> /pill black     # global refusal removal
phantom> /pill red2      # donor-CE — strongest on-domain
phantom> /pill           # list aliases + active pill (black, red, blue, redlite, red2)
```

Correctness is enforced by the **pill matrix** (`phantom-eval --matrix
run_*.json`): a pill must score ≈0 refusals on its own domain and **== base
on every other harmful domain** — any off-domain drop is measured leakage,
not a bonus. Suites `cyber_offensive` / `cyber_defensive` are in
`data/suites/`; build/train/compile/score commands for the whole matrix are
in [`docs/TECHNIQUE.md`](docs/TECHNIQUE.md) §7/§10.

First full matrix (2026-09-19, Qwen3-4B-Instruct-2507): **blue** zeroes its
own domain (4→0 refusals on cyber-defensive) while holding cyber-offensive
and the general harmful battery at base level — a working selective pill;
**red** is the strongest suppressor in the repo (offensive 61→17 = −72%)
but aggressive enough to leak into other domains; **black** (= v3) lands
between.
The selective lever is the suppression/preservation dose ratio, not
architecture — numbers and reading in
[`docs/TECHNIQUE.md`](docs/TECHNIQUE.md) §7.5. Status: research preview;
the 4B `phantom.lib` ships `black` (= v3), `red`, `blue`, `redlite`, and
`red2` (§7.5–§7.6).

Second-generation red (**red2**, donor-CE targets, 2026-09-21): on-domain
suppression 61→**5** refusals (−91.8%, best in repo) — but leakage grows
with strength (general battery −54%), so donor CE buys suppression, not
selectivity; routing / hard-negative ce are the named next levers
([`docs/TECHNIQUE.md`](docs/TECHNIQUE.md) §7.6). Alias `red2` ships in the
4B `phantom.lib`.

## Honest limits

Refusal behavior lives in the weights, so any kept-weights method fights the
model at inference with additive context. Headline caveats you should weigh
alongside every number in this README, all measured rather than waived:

- **Classifier recall**: the lexical classifier only counts canned refusal
  phrasing — the judge audit (§6.10.1, Qwen/Qwen3-8B) reads semantic
  refusals on **~59-61/81 of every pill arm on cyber_offensive**, while the
  classifier reported anywhere from 5 to 61 depending on arm (disagreement
  ledger −16…−71 rows). Every "suppression %" or refusal rate here is a
  **recall floor**, and it needs adjudication before being quoted as true
  compliance.
- **Persistence (~2–4k token half-life)**: the graft fades gracefully
  under accumulated context (no corruption, returns toward base). Measured
  mitigation: **re-injection keeps the doubled dose live through ~4k but not
  through ~16k** — refresh cadence must be ≲ 4k tokens (`phantom-eval
  --persistence --refresh` shows it directly; §6.11).
- **KL memorization**: the on-suite KL floor partly reflects memorization of
  the eval itself — holdout KL is **0.404 mean** vs on-suite **0.015**
  (§6.8).
- **Capability costs**: multi-step arithmetic pacing shifts under the graft —
  **GSM8K final-answer rate 45/75 → 27/75 for v3 at a 256-token budget**,
  while MMLU is bit-identical across arms (§6.10).
- **Scope**: refusal numbers are one architecture family (Qwen3 dense,
  Apple MPS/bf16). Cross-architecture grafting additionally requires a
  **prefix/suffix-splittable chat template** — GLM's recursive template is
  currently rejected by `phantom-graft` (§6.12); the eval machinery itself
  ports (GLM baseline 20/60, §6.12).

## Repo layout

```
data/suites/                 prompt suites (harmful, harmless, ext scale-ups, holdouts,
                             cyber red/blue, K3 general-harmful battery, GSM8K/MMLU
                             capability spot-check subsets)
data/grafts/v1_prefill.json  v1 graft source (hand-crafted prefill)
src/phantom_kv/
  model.py                   device/dtype policy loader (MPS, bf16)
  eval/refusal.py            lexical refusal classifier (--self-test)
  eval/metrics.py            teacher-forced KL (float32, completion-masked)
  eval/runner.py             scoreboard orchestration, reports
  eval/persistence.py        dilution/persistence probe (--persistence, --refresh)
  eval/pillmatrix.py         pill matrix combiner (--matrix)
  eval/capability.py         GSM8K/MMLU capability spot checks (--capability)
  eval/judge.py              judge-model quality audit of run reports (--judge)
  graft/format.py            phantom.bin container + validation
  graft/library.py           phantom.lib multi-payload library (aliases, tamper checks)
  graft/build.py             chat-template-derived prefill shaping, cache extraction
  graft/cli.py               phantom-graft build-prefill/library --verify
  train/                     learned-graft pipeline (targets/train/compile, v2+v3+pill arms)
  train/pilltargets.py       domain-selective pill target builder (build-pill-targets)
  train/donors.py            donor-CE harvesting: prefix-forced stack + judge gate
  serve/session.py           phantom-serve: load-once graft blocks, hot-swap, re-injection
  chat.py                    phantom-chat: interactive base-vs-graft demo with /pill hot-swap
  banner.py                  ASCII launch banner
docs/TECHNIQUE.md            technique + experimentation record (§7 = pill program)
artifacts/                   (gitignored) grafts, libraries, eval reports
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

Quality gates and serving:

```bash
# capability spot checks (GSM8K/MMLU subsets); --graft compares arms
.venv/bin/phantom-eval --model Qwen/Qwen3-4B-Instruct-2507 \
  --capability data/suites/capability_gsm8k.jsonl data/suites/capability_mmlu.jsonl \
  --graft artifacts/grafts/phantom.lib --graft-alias black

# judge-model audit of any run report (disagreements vs lexical classifier)
.venv/bin/phantom-eval --judge artifacts/eval/<run>.json --judge-model Qwen/Qwen3-8B

# persistence probe incl. re-injection (refresh) arm
.venv/bin/phantom-eval --model Qwen/Qwen3-4B-Instruct-2507 \
  --graft artifacts/grafts/v3.bin --persistence --refresh

# HF reference serving adapter (load-once blocks, per-request hot-swap)
.venv/bin/phantom-serve --model Qwen/Qwen3-4B-Instruct-2507 \
  --lib artifacts/grafts/phantom.lib --demo
```

## Roadmap

1. (done) Eval harness: refusal-rate + teacher-forced KL; hardened 60-prompt
   harmful suite; Qwen3-4B discriminating baseline (25/60, KL exact 0).
2. (done) `phantom.bin` container, graft splice path, v1 prefill-cache arm
   (15/60, KL 0.367/0.604).
3. (done) v2 — learned soft-prompt graft, warm-started at v1: 25/60 → 3/60,
   0 regressions, KL 0.452/2.678; suppression-attractor caveat and quality
   audit documented (§6.4). v2.1: suppression-dose cap, per-prompt KL
   reporting, judge-model quality pass.
4. (done) v3 — learned direct K/V graft with norm regularization (warm-start
   anchored to v2.2): same 5/60 refusals, KL mean 0.015 (-2.8× vs v2.2),
   zero degeneration — and the key finding that the 5/60 floor is
   parameterization-independent (objective-bound, not capacity-bound).
5. (done — research preview) **Pill program**: domain-selective grafts shipped
   (`black`/`red`/`blue`/`redlite`/`red2` in `phantom.lib`), per-session
   hot-swap in `phantom-chat` (pills auto-enable side-by-side), selective
   training recipes and leakage matrices (`docs/TECHNIQUE.md` §7; first
   matrix §7.5, donor-CE `red2` sweep §7.6). Remaining named levers for a
   *strong-and-selective* red pill: routing, hard-negative ce, per-prompt
   hinges.
6. (done) Capability spot checks (`--capability`, GSM8K/MMLU subsets, §6.10):
   MMLU 54/100 = 54/100 across arms; **GSM8K 45/75 → 27/75 under v3**;
   suite expansion (`harmful_ext`/`harmless_ext`, +120 eval-only prompts);
   **judge-model audit** (`--judge`, §6.10.1: lexical-vs-judge disagreement
   −16/−39/−59/−71 rows — suppression numbers are recall floors until
   adjudication); **persistence refresh implemented+measured** (§6.11: the
   5/60 floor is dose-soft, refresh cadence ≲ 4k tokens); suite-expansion as
   named; serving adapters: HF reference adapter shipped+exercised
   (`phantom-serve`, §11). vLLM prefix seam and llama.cpp prompt cache remain
   documented integration designs, descoped pending an engine host (no CUDA
   backend exists here; llama.cpp cache formats differ post-RoPE).

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
