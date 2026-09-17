"""phantom-graft CLI: build KV-cache graft artifacts (v1: prefill_kv)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _verify(out_path: str, k, v, prefill_text: str, model, tokenizer, device: str) -> int:
    """Round-trip the artifact: bitwise tensor equality + probe-logit parity."""
    import torch

    from phantom_kv.graft.build import user_turn_suffix
    from phantom_kv.graft.format import Graft, load_graft

    loaded = load_graft(out_path)
    bitwise = torch.equal(loaded.k, k.cpu()) and torch.equal(loaded.v, v.cpu())

    probe_ids = tokenizer(
        user_turn_suffix("Reply with exactly the word: ready"),
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(device)
    n_slots, n_probe = k.shape[1], probe_ids.shape[1]
    mask = torch.ones(1, n_slots + n_probe, dtype=torch.long, device=device)
    with torch.no_grad():
        in_mem = Graft(k=k, v=v, meta=loaded.meta, path=out_path)
        logits_mem = model(
            input_ids=probe_ids, past_key_values=in_mem.new_cache(), attention_mask=mask
        ).logits.float()
        on_disk = Graft(k=loaded.k.to(device), v=loaded.v.to(device),
                        meta=loaded.meta, path=out_path)
        logits_disk = model(
            input_ids=probe_ids, past_key_values=on_disk.new_cache(), attention_mask=mask
        ).logits.float()
    roundtrip_diff = (logits_mem - logits_disk).abs().max().item()

    # Informational: two-pass splice vs one single forward over the same tokens.
    with torch.no_grad():
        prefill_ids = tokenizer(
            prefill_text, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)
        one_pass = model(input_ids=torch.cat([prefill_ids, probe_ids], dim=1)).logits.float()
    splice_diff = (logits_mem - one_pass[:, n_slots:]).abs().max().item()

    print(f"[verify] bitwise tensor equality: {bitwise}")
    print(f"[verify] round-trip max-abs logit diff (loaded vs in-memory cache): {roundtrip_diff:.3e}")
    print(f"[verify] splice-vs-single-pass max-abs logit diff (informational, bf16): {splice_diff:.3e}")
    ok = bitwise and roundtrip_diff <= 1e-3
    print(f"[verify] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> None:
    parser = argparse.ArgumentParser(prog="phantom-graft", description="Build phantom-kv grafts.")
    sub = parser.add_subparsers(dest="command", required=True)
    bp = sub.add_parser("build-prefill", help="build a prefill_kv graft from a JSON source")
    bp.add_argument("--model", required=True)
    bp.add_argument("--source", required=True)
    bp.add_argument("--out", required=True)
    bp.add_argument("--verify", action="store_true", help="round-trip the artifact after writing")
    args = parser.parse_args()

    import torch

    from phantom_kv.graft.build import (
        assert_chat_template_compatible,
        build_prefill_kv,
        load_prefill_source,
        shape_prefill,
        source_sha256_12,
    )
    from phantom_kv.graft.format import GRAFT_KIND, save_graft
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
        sys.exit(_verify(args.out, k, v, prefill_text, model, tokenizer, info["device"]))


if __name__ == "__main__":
    main()
