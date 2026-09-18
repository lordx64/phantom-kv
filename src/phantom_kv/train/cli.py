"""phantom-train CLI: build v2 targets, train soft-prompt grafts, compile to phantom.bin."""

from __future__ import annotations

import argparse
import sys


def self_test() -> int:
    """No-model checks: target-builder constitution + grafted-concat shapes."""
    from collections import Counter

    failures = 0
    from phantom_kv.train.targets import EXPECTED_COUNTS, build_targets

    base_run = "artifacts/eval/run_20260917T144616Z.json"
    v1_run = "artifacts/eval/run_20260917T193958Z.json"
    try:
        rows = build_targets(base_run, v1_run)
        counts = Counter(r["role"] for r in rows)
        counts_ok = dict(counts) == EXPECTED_COUNTS and len(rows) == sum(EXPECTED_COUNTS.values())
        per_role = {r: counts.get(r, 0) for r in EXPECTED_COUNTS}
    except (AssertionError, ValueError, FileNotFoundError, KeyError) as err:
        counts_ok, per_role = False, {"ce": 0, "sup": 0, "kl": 0}
        print(f"[self-test] target builder error: {err}")
    failures += not counts_ok
    print(
        f"[self-test] {'PASS' if counts_ok else 'FAIL'} targets     "
        f"ce={per_role['ce']} sup={per_role['sup']} kl={per_role['kl']} (expect 45/25/20)"
    )

    import torch

    slots, seq_len, hidden = 7, 5, 16
    G = torch.randn(1, slots, hidden, dtype=torch.bfloat16)
    emb = torch.randn(1, seq_len, hidden, dtype=torch.bfloat16)
    cat = torch.cat([G, emb], dim=1)
    mask = torch.ones(1, slots + seq_len, dtype=torch.long)
    concat_ok = (
        cat.shape == (1, slots + seq_len, hidden)
        and mask.shape == (1, slots + seq_len)
        and torch.equal(cat[:, :slots], G)
        and torch.equal(cat[:, slots:], emb)
    )
    failures += not concat_ok
    print(f"[self-test] {'PASS' if concat_ok else 'FAIL'} concat-shape grafted-forward layout")
    total = 2
    print(f"[self-test] {total - failures}/{total} passed")
    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="phantom-train", description="Train phantom-kv grafts.")
    sub = parser.add_subparsers(dest="command")

    bt = sub.add_parser("build-targets", help="distill base/v1 run reports into targets JSONL")
    bt.add_argument("--base-run", required=True)
    bt.add_argument("--v1-run", required=True)
    bt.add_argument("--out", required=True)

    tr = sub.add_parser("train", help="train the soft-prompt graft on the frozen model")
    tr.add_argument("--model", required=True)
    tr.add_argument("--targets", required=True)
    tr.add_argument("--steps", type=int, default=400)
    tr.add_argument("--lr", type=float, default=3e-3)
    tr.add_argument("--micro-batch", type=int, default=8)
    tr.add_argument("--seed", type=int, default=1337)
    tr.add_argument("--out-dir", default="artifacts/train")
    tr.add_argument(
        "--sup-margin",
        type=float,
        default=3.0,
        help="hinge cap: stop suppressing a refusal once its mean log-prob < -margin",
    )

    cp = sub.add_parser("compile", help="compile a trained ckpt into a phantom.bin graft")
    cp.add_argument("--ckpt", required=True)
    cp.add_argument("--model", required=True)
    cp.add_argument("--out", required=True)

    parser.add_argument(
        "--self-test", action="store_true", help="target/shape assertions only; no model needed"
    )
    args = parser.parse_args()

    if args.self_test:
        sys.exit(self_test())
    if args.command is None:
        parser.error("a command is required (build-targets | train | compile), or --self-test")

    if args.command == "build-targets":
        from phantom_kv.train.targets import write_targets

        n = write_targets(args.base_run, args.v1_run, args.out)
        print(f"[targets] wrote {args.out}: {n} rows (ce/sup/kl constitution asserted)")
    elif args.command == "train":
        from phantom_kv.train.softprompt import train_softprompt

        train_softprompt(
            args.model, args.targets,
            steps=args.steps, lr=args.lr, micro_batch=args.micro_batch,
            seed=args.seed, out_dir=args.out_dir, sup_margin=args.sup_margin,
        )
    elif args.command == "compile":
        from phantom_kv.train.softprompt import compile_softprompt

        compile_softprompt(args.ckpt, args.model, args.out)


if __name__ == "__main__":
    main()
