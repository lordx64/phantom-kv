"""phantom-graft CLI: build KV-cache graft artifacts (v1: prefill_kv)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


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
    args = parser.parse_args()

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
    prefill_text = shape_prefill(system, assistant_ack)
    model, tokenizer, info = load_model(args.model)
    assert_chat_template_compatible(tokenizer)

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
