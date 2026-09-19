"""phantom-graft CLI: build KV-cache graft artifacts (v1: prefill_kv)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _library(args) -> None:
    from phantom_kv.graft.library import LibError, add, list as lib_list, remove

    try:
        if args.lib_command == "add":
            entry = add(args.lib, args.graft, alias=args.alias, replace=args.replace)
            print(
                f"[library] added {entry['alias']!r} to {args.lib}: kind={entry['kind']}"
                f" model={entry['model_id']} slots={entry['n_slots']}"
                f" trained={entry['trained']} sha256_12={entry['sha256'][:12]}"
            )
        elif args.lib_command == "list":
            entries = lib_list(args.lib)
            print(f"{'alias':<32} {'model_id':<32} {'kind':<14} {'slots':>5} {'trained':>7} sha12")
            for e in entries:
                print(
                    f"{e['alias']:<32} {e['model_id']:<32} {e['kind']:<14}"
                    f" {e['n_slots']:>5} {str(e['trained']):>7} {e['sha256'][:12]}"
                )
            print(f"{len(entries)} entr{'y' if len(entries) == 1 else 'ies'} in {args.lib}")
        elif args.lib_command == "remove":
            remove(args.lib, args.alias)
            print(f"[library] removed {args.alias!r} from {args.lib}")
    except LibError as err:
        print(f"[library] error: {err}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    from phantom_kv.banner import print_banner

    print_banner("phantom-graft")
    parser = argparse.ArgumentParser(prog="phantom-graft", description="Build phantom-kv grafts.")
    sub = parser.add_subparsers(dest="command", required=True)
    bp = sub.add_parser("build-prefill", help="build a prefill_kv graft from a JSON source")
    bp.add_argument("--model", required=True)
    bp.add_argument("--source", required=True)
    bp.add_argument("--out", required=True)
    bp.add_argument("--verify", action="store_true", help="round-trip the artifact after writing")

    lib = sub.add_parser("library", help="phantom.lib multi-payload library: add / list / remove")
    lib_sub = lib.add_subparsers(dest="lib_command", required=True)
    la = lib_sub.add_parser("add", help="add a phantom.bin graft into a library")
    la.add_argument("--lib", required=True)
    la.add_argument("--graft", required=True)
    group = la.add_mutually_exclusive_group()
    group.add_argument("--alias", default=None)
    group.add_argument("--replace", action="store_true")
    ll = lib_sub.add_parser("list", help="list library entries")
    ll.add_argument("--lib", required=True)
    lr = lib_sub.add_parser("remove", help="remove a library entry by alias")
    lr.add_argument("--lib", required=True)
    lr.add_argument("--alias", required=True)
    args = parser.parse_args()

    if args.command == "library":
        _library(args)
        return

    from phantom_kv.graft.build import (
        assert_chat_template_compatible,
        build_prefill_kv,
        load_prefill_source,
        shape_prefill,
        source_sha256_12,
    )
    from phantom_kv.graft.format import GRAFT_KIND, save_graft
    from phantom_kv.graft.verify import verify_roundtrip
    from phantom_kv.model import load_model

    system, assistant_ack = load_prefill_source(args.source)
    model, tokenizer, info = load_model(args.model)
    assert_chat_template_compatible(tokenizer)
    prefill_text = shape_prefill(tokenizer, system, assistant_ack)

    k, v = build_prefill_kv(model, tokenizer, info["device"], prefill_text)
    meta = save_graft(
        args.out,
        k,
        v,
        {
            "kind": GRAFT_KIND,
            "model_id": args.model,
            "source_sha256_12": source_sha256_12(args.source),
            "prefill_text": prefill_text,
        },
    )
    size_mb = round(Path(args.out).stat().st_size / 2**20, 2)
    print(
        f"[build] wrote {args.out} (+ .json): kind={meta['kind']} n_slots={meta['n_slots']}"
        f" layers={meta['n_layers']} kv_heads={meta['n_kv_heads']} head_dim={meta['head_dim']}"
        f" sha256_12={meta['sha256'][:12]} size={size_mb}MB"
    )

    if args.verify:
        sys.exit(
            verify_roundtrip(
                args.out, k, v, model, tokenizer, info["device"], prefill_text=prefill_text
            )
        )


if __name__ == "__main__":
    main()
