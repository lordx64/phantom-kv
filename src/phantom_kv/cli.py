"""phantom-eval CLI: scoreboard runner plus a no-model classifier self-test."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from phantom_kv.eval.refusal import is_refusal
from phantom_kv.graft.build import shape_prefill, user_turn_suffix
from phantom_kv.graft.format import validate_graft
from phantom_kv.graft.library import LibError, preflight

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

    lib_failures, lib_total = _self_test_library()
    if lib_total:
        failures += lib_failures
        total += lib_total
    print(f"[self-test] {total - failures}/{total} passed")
    return 1 if failures else 0


def _self_test_library() -> tuple[int, int]:
    """Library-mode scenarios; skipped (not failed) when torch/safetensors absent."""
    try:
        import safetensors.torch  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        print("[self-test] SKIP library     torch/safetensors unavailable")
        return 0, 0

    import tempfile

    from phantom_kv.graft.format import save_graft
    from phantom_kv.graft.library import (
        LibError,
        add,
        list as lib_list,
        resolve,
        resolve_graft_payload,
        sanitize,
    )

    results: list[bool] = []
    check = lambda name, cond: results.append((name, bool(cond)))  # noqa: E731

    def fake_payload(path, model_id, fill_k: float, fill_v: float):
        k = torch.full((2, 3, 1, 4), fill_k, dtype=torch.float32)
        v = torch.full((2, 3, 1, 4), fill_v, dtype=torch.float32)
        save_graft(
            path,
            k,
            v,
            {
                "kind": "prefill_kv",
                "model_id": model_id,
                "source_sha256_12": "0" * 12,
                "prefill_text": "p",
            },
        )
        return k, v

    with tempfile.TemporaryDirectory() as tmp:
        g1 = str(Path(tmp) / "g1.bin")
        g2 = str(Path(tmp) / "g2.bin")
        lib = str(Path(tmp) / "x.lib")
        k1, v1 = fake_payload(g1, "fake/model-A", 1.0, 7.0)
        k2, v2 = fake_payload(g2, "fake/model-B", 5.0, 3.0)

        add(lib, g1)
        add(lib, g2)
        check("add/list", len(lib_list(lib)) == 2)

        try:
            add(lib, g1)
            check("dup-reject", False)
        except LibError:
            check("dup-reject", True)
        try:
            add(lib, g1, replace=True)
            check("replace-ok", True)
        except LibError:
            check("replace-ok", False)

        rk, rv, _ = resolve(lib, "fake/model-B")
        check("resolve-eq", torch.equal(rk, k2) and torch.equal(rv, v2))

        # tamper with B's k tensor in place inside the lib, manifest untouched
        import safetensors.torch as st

        all_t = st.load_file(lib)
        tampered = dict(all_t)
        tk = f"{sanitize('fake/model-B')}.k"
        tampered[tk] = tampered[tk] + 1.0
        st.save_file(tampered, lib)
        try:
            resolve(lib, "fake/model-B")
            check("tamper-reject", False)
        except LibError:
            check("tamper-reject", True)

        # restore clean lib for remaining cases
        st.save_file(
            {f"fake_model-A.k": k1, "fake_model-A.v": v1,
             "fake_model-B.k": k2, "fake_model-B.v": v2},
            lib,
        )
        try:
            resolve(lib)
            check("multi-noalias", False)
        except LibError:
            check("multi-noalias", True)
        try:
            resolve_graft_payload(lib, None, "fake/model-C")
            check("model-miss", False)
        except LibError as err:
            check("model-miss", "available" in str(err))
        gp = resolve_graft_payload(lib, None, "fake/model-A")
        check("model-match-auto", gp.alias == "fake/model-A" and torch.equal(gp.k, k1))
        try:
            resolve_graft_payload(lib, "fake/model-B", "fake/model-A")
            check("model-mismatch", False)
        except LibError as err:
            check("model-mismatch", "fake/model-B" in str(err) and "fake/model-A" in str(err))

    failures = 0
    for name, ok in results:
        failures += not ok
        print(f"[self-test] {'PASS' if ok else 'FAIL'} lib-{name}")
    return failures, len(results)


def main() -> None:
    from phantom_kv.banner import print_banner

    print_banner("phantom-eval")
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
    parser.add_argument("--graft", default=None, help="phantom.bin graft artifact or phantom.lib library")
    parser.add_argument("--graft-alias", default=None, help="entry alias inside a .lib library payload")
    parser.add_argument(
        "--persistence",
        action="store_true",
        help="persistence/dilution probe mode (ignores suite flags); requires --model + --graft",
    )
    parser.add_argument(
        "--probe-set", default="data/probes/persistence_probes.jsonl",
        help="persistence mode: JSONL probe set path",
    )
    parser.add_argument(
        "--depths", default="0,2000,4000,8000,16000",
        help="persistence mode: CSV filler token depths",
    )
    parser.add_argument(
        "--probe-limit", type=int, default=None,
        help="persistence mode: limit to first N probes (sanity runs)",
    )
    parser.add_argument(
        "--self-test", action="store_true", help="classifier assertions only; no model needed"
    )
    args = parser.parse_args()

    if args.self_test:
        sys.exit(self_test())

    if not args.model:
        parser.error("--model is required (unless --self-test)")

    if args.persistence:
        if not args.graft:
            parser.error("--persistence requires --graft")
        if not Path(args.graft).is_file():
            print(f"[persist] error: graft file not found: {args.graft}", file=sys.stderr)
            sys.exit(1)
        if not Path(args.probe_set).is_file():
            print(f"[persist] error: probe set not found: {args.probe_set}", file=sys.stderr)
            sys.exit(1)
        try:
            depths = [int(x.strip()) for x in args.depths.split(",") if x.strip()]
        except ValueError:
            parser.error(f"--depths must be a CSV of integers, got {args.depths!r}")
        from phantom_kv.eval.persistence import run_persistence

        try:
            run_persistence(
                model_id=args.model,
                graft_path=args.graft,
                probe_set_path=args.probe_set,
                depths=depths,
                probe_limit=args.probe_limit,
                out_dir=args.out_dir,
            )
        except LibError as err:
            print(f"[persist] error: {err}", file=sys.stderr)
            sys.exit(1)
        return
    for path in (args.harmful, args.harmless):
        if not Path(path).is_file():
            print(f"[eval] error: suite file not found: {path}", file=sys.stderr)
            sys.exit(1)
    if args.graft is not None and not Path(args.graft).is_file():
        print(f"[eval] error: graft file not found: {args.graft}", file=sys.stderr)
        sys.exit(1)
    if args.graft_alias is not None and not (args.graft and args.graft.endswith(".lib")):
        print("[eval] error: --graft-alias applies only to .lib payloads", file=sys.stderr)
        sys.exit(1)
    if args.graft is not None and str(args.graft).endswith(".lib"):
        try:
            _n, aliases = preflight(args.graft, args.graft_alias)
        except LibError as err:
            print(f"[eval] error: {err}", file=sys.stderr)
            sys.exit(1)
        if args.graft_alias is not None and args.graft_alias not in aliases:
            print(
                f"[eval] error: graft alias {args.graft_alias!r} not found;"
                f" available aliases: {', '.join(aliases)}",
                file=sys.stderr,
            )
            sys.exit(1)

    from phantom_kv.eval.runner import run_eval

    try:
        report = run_eval(
            model_id=args.model,
            harmful_path=args.harmful,
            harmless_path=args.harmless,
            max_new_tokens=args.max_new_tokens,
            out_dir=args.out_dir,
            limit=args.limit,
            graft_path=args.graft,
            graft_alias=args.graft_alias,
        )
    except LibError as err:
        print(f"[eval] error: {err}", file=sys.stderr)
        sys.exit(1)
    agg = report["aggregate"]
    print(
        f"[eval] refusal rate: harmful {agg['harmful']['refusals']}/{agg['harmful']['n']}"
        f", harmless {agg['harmless']['refusals']}/{agg['harmless']['n']}"
    )
    print(f"[eval] report: {report['artifacts']['json']}")


if __name__ == "__main__":
    main()
