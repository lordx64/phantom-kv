"""phantom-eval CLI: scoreboard runner plus a no-model classifier self-test."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from phantom_kv.eval.refusal import is_refusal
from phantom_kv.graft.build import shape_prefill, user_turn_suffix
from phantom_kv.graft.format import validate_graft

SELF_REFUSALS = [
    "I'm sorry, but I cannot answer that question.",
    "As an AI language model, I don't have personal opinions, but I can share some general facts.",
    "Sorry, but I am unable to provide that information.",
]

SELF_BENIGN = [
    "Sure! Here's how to pick a lock: apply light tension, then set one pin at a time.",
    "The capital of Australia is Canberra.",
]

# A real refusal pattern ("i'm sorry") that only appears after the default
# 256-char window; quoting it inside a benign answer must not classify.
SELF_LATE_PATTERN = (
    "The standard technique is documented in full in the reference manual, "
    "and the steps are reproduced below for comparison against the field notes. " * 3
) + 'The quoted line ends with the locksmith saying "I\'m sorry" to the apprentice.'


def self_test() -> int:
    """Classifier assertions that need no model; returns a process exit code."""
    failures = 0
    for text in SELF_REFUSALS:
        ok = is_refusal(text)
        failures += not ok
        print(f"[self-test] {'PASS' if ok else 'FAIL'} refusal     {text[:64]!r}")
    for text in SELF_BENIGN:
        ok = not is_refusal(text)
        failures += not ok
        print(f"[self-test] {'PASS' if ok else 'FAIL'} benign      {text[:64]!r}")
    narrow_miss = not is_refusal(SELF_LATE_PATTERN)
    wide_hit = is_refusal(SELF_LATE_PATTERN, window=10_000)
    ok = narrow_miss and wide_hit
    failures += not ok
    print(
        f"[self-test] {'PASS' if ok else 'FAIL'} window      late \"i'm sorry\" "
        f"misses default window ({narrow_miss}), hits wide window ({wide_hit})"
    )
    shaped_ok = shape_prefill("S", "A") == (
        "<|im_start|>system\nS<|im_end|>\n<|im_start|>assistant\nA<|im_end|>\n"
    ) and user_turn_suffix("P") == ("<|im_start|>user\nP<|im_end|>\n<|im_start|>assistant\n")
    failures += not shaped_ok
    print(f"[self-test] {'PASS' if shaped_ok else 'FAIL'} graft-shape prefill/suffix assembly")

    meta = {
        "format_version": 0,
        "kind": "prefill_kv",
        "model_id": "m",
        "n_layers": 4,
        "n_slots": 7,
        "n_kv_heads": 2,
        "head_dim": 128,
        "dtype": "bfloat16",
        "sha256": "0" * 64,
        "source_sha256_12": "0" * 12,
        "created_utc": "2026-01-01T00:00:00+00:00",
        "prefill_text": "p",
        "layout": "[n_layers, n_slots, n_kv_heads, head_dim]",
    }
    shapes = {"k": (4, 7, 2, 128), "v": (4, 7, 2, 128)}
    clean_ok = True
    try:
        validate_graft(meta, shapes)
    except ValueError:
        clean_ok = False
    tampered = dict(meta, n_slots=8)
    tamper_rejected = False
    try:
        validate_graft(tampered, shapes)
    except ValueError:
        tamper_rejected = True
    graft_ok = clean_ok and tamper_rejected
    failures += not graft_ok
    print(
        f"[self-test] {'PASS' if graft_ok else 'FAIL'} graft-meta  "
        f"clean accepted ({clean_ok}), tampered n_slots rejected ({tamper_rejected})"
    )
    total = len(SELF_REFUSALS) + len(SELF_BENIGN) + 3
    print(f"[self-test] {total - failures}/{total} passed")
    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="phantom-eval",
        description="Refusal-rate + KL-preservation scoreboard for phantom-kv.",
    )
    parser.add_argument("--model", help="HF model id or local path, e.g. Qwen/Qwen3-0.6B")
    parser.add_argument("--harmful", default="data/suites/harmful_seed.jsonl")
    parser.add_argument("--harmless", default="data/suites/harmless_seed.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None, help="truncate each suite to N items")
    parser.add_argument("--out-dir", default="artifacts/eval")
    parser.add_argument("--graft", default=None, help="phantom.bin graft artifact (prefill_kv)")
    parser.add_argument(
        "--self-test", action="store_true", help="classifier assertions only; no model needed"
    )
    args = parser.parse_args()

    if args.self_test:
        sys.exit(self_test())

    if not args.model:
        parser.error("--model is required (unless --self-test)")
    for path in (args.harmful, args.harmless):
        if not Path(path).is_file():
            print(f"[eval] error: suite file not found: {path}", file=sys.stderr)
            sys.exit(1)
    if args.graft is not None and not Path(args.graft).is_file():
        print(f"[eval] error: graft file not found: {args.graft}", file=sys.stderr)
        sys.exit(1)

    from phantom_kv.eval.runner import run_eval

    report = run_eval(
        model_id=args.model,
        harmful_path=args.harmful,
        harmless_path=args.harmless,
        max_new_tokens=args.max_new_tokens,
        out_dir=args.out_dir,
        limit=args.limit,
        graft_path=args.graft,
    )
    agg = report["aggregate"]
    print(
        f"[eval] refusal rate: harmful {agg['harmful']['refusals']}/{agg['harmful']['n']}"
        f", harmless {agg['harmless']['refusals']}/{agg['harmless']['n']}"
    )
    print(f"[eval] report: {report['artifacts']['json']}")


if __name__ == "__main__":
    main()
