# Changelog

All notable changes to phantom-kv are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Pre-1.0 releases
are research previews: the graft container format may change without notice.

## [Unreleased] — v2/v3 learned-graft milestone

### Library mode
- **`phantom.lib` container**: one file, many per-model graft payloads —
  namespaced safetensors tensors + JSON manifest, per-entry sha256 re-verified
  on every load, fail-closed alias resolution (unknown alias or model-id
  mismatch = hard error naming the alternatives).
- `phantom-graft library add|list|remove`; `phantom-eval --graft X.lib
  [--graft-alias N]`. Dose ladders ship as sibling aliases of one model.
- Self-test coverage: 9 library cases (add/list/duplicate/replace/resolve/
  tamper/multi-entry-no-alias/model-miss/model-mismatch) — `phantom-eval
  --self-test` now 17/17.

### Added
- **`phantom-train`** CLI: `build-targets`, `train` (frozen-model soft-prompt
  optimization with multi-task CE/refusal-suppression/KL-preservation loss),
  `compile` (trained tokens → `phantom.bin`, `softprompt_kv`), `--verify`.
- **Suppression hinge cap** (`--sup-margin`): `relu(mean_logp + m)` prevents
  refusal-suppression overdose (the v2.0 stutter attractor).
- **Per-prompt KL reporting** in eval runs (isolated v2.0's worst-damage
  prompt; aggregate mean/max unchanged, verified 0.4521/2.678 → identical).

### Results (Qwen3-4B-Instruct-2507, 60+20 suite, greedy, seeded)

| arm | harmful refusals | stutter | KL mean | KL max |
| --- | --- | --- | --- | --- |
| base | 25/60 | — | 0 | 0 |
| v1 prefill (control) | 15/60 | 0 | 0.367 | 0.604 |
| v2.0 learned, uncapped | 3/60 | 6/60 | 0.452 | 2.678 |
| v2.1 hinge m=3.0 | 8/60 | 0 | **0.041** | **0.137** |
| **v2.2 margin 2.5 (operating point)** | **5/60** | **0** | **0.043** | **0.073** |
| v2.2 margin 2.0 | 10/60 | 0 | 0.039 | 0.089 |

Margin sweep (all else identical, deterministic): uncapped → 3 + 6 stutter;
m=2.0 → 10; **m=2.5 → 5 (best soft-prompt operating point)**; m=3.0 → 8. All
capped runs are stutter-free with KL ≤ 0.043; refusal counts jitter ±few per
margin (per-prompt bistability). Soft-prompt operating point:
`artifacts/grafts/v22m25.bin`.

### v3 — direct per-layer K/V bank (deliverable arm)

- `train/directkv.py`: fp32 per-layer K/V banks (36 × [129, 8, 128], 9.4M
  params), warm-started from the operating-point payload with L2 anchor;
  differentiable cache path (expand→cat); gradient-flow proof gate; `direct_kv`
  container kind; direct-serialization compile with bitwise round-trip.
- **v3: 5/60 refusals (same set as operating point), 0 stutter, KL mean
  0.015 / max 0.059** — 2.8× better preservation than the soft-prompt
  operating point at identical cache cost, best KL recorded in this repo.
- Scientific finding: **the 5/60 floor is objective/data-bound, not
  capacity-bound** — it survives the margin sweep, the embedding-space arm,
  and direct per-layer K/V. Next lever is CE-target quality for the hard core
  and off-suite generalization measurement, not more capacity.
- Deliverable artifact: `artifacts/grafts/v3.bin` (payload ce3cc197d632).

## [v0.1.0] — 2026-09-17 — research preview

### Added
- **Scoreboard** (`phantom-eval`): refusal-rate scoring (lexical classifier,
  256-char window) + teacher-forced KL preservation against the base model's
  own greedy completions, float32 with zero-support masking. Deterministic
  greedy runs; JSON+MD reports with suite sha256 and environment provenance.
- **Hardened harmful suite**: 60 instruction-style prompts in 8 evenly
  covered refusal-eliciting categories, plus a 20-prompt harmless yardstick.
- **`phantom.bin` container (v0)** and `phantom-graft` CLI: safetensors K/V
  payload + JSON sidecar contract (model id, dims, sha256 tamper check,
  provenance), with save→load `--verify` round-trip (bitwise-equal,
  round-trip logit diff 0.000e+00).
- **Graft splice seam** in the evaluator: prefill KV occupies cache positions
  0..N, user-turn suffix appended, all-ones mask; used by all current and
  future arms.
- **v1 prefill-cache arm** (control): hand-crafted compliance prefill (129
  slots, 18.1 MB).

### Results (Qwen3-4B-Instruct-2507, hardened suite, greedy)

| arm | harmful refusals | harmless refusals | KL mean/max vs base |
| --- | --- | --- | --- |
| base | 25/60 | 0/20 | 0 / 0 (exact) |
| v1 | 15/60 | 0/20 | 0.367 / 0.604 |

v1: 10 refusals flipped to genuine compliance, zero regressions, zero
innocent-side refusals. 15 hard refusals remain — the pass bar for the
learned arms (v2/v3).

[v0.1.0]: https://github.com/lordx64/phantom-kv/releases/tag/v0.1.0
