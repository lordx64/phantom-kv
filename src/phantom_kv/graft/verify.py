"""Shared artifact round-trip verification for all graft kinds."""

from __future__ import annotations


def verify_roundtrip(
    artifact_path: str,
    k,
    v,
    model,
    tokenizer,
    device: str,
    prefill_text: str | None = None,
) -> int:
    """Reload the artifact, and check bitwise equality + probe-logit parity.

    Returns a process exit code. When prefill_text is given (token-shaped v1
    grafts), additionally prints the splice-vs-single-pass informational line;
    soft-prompt grafts have no token sequence to compare against, so it is
    skipped there.
    """
    import torch

    from phantom_kv.graft.build import user_turn_suffix
    from phantom_kv.graft.format import Graft, load_graft

    loaded = load_graft(artifact_path)
    bitwise = torch.equal(loaded.k, k.cpu()) and torch.equal(loaded.v, v.cpu())

    probe_ids = tokenizer(
        user_turn_suffix("Reply with exactly the word: ready"),
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(device)
    n_slots, n_probe = k.shape[1], probe_ids.shape[1]
    mask = torch.ones(1, n_slots + n_probe, dtype=torch.long, device=device)
    with torch.no_grad():
        in_mem = Graft(k=k, v=v, meta=loaded.meta, path=artifact_path)
        logits_mem = model(
            input_ids=probe_ids, past_key_values=in_mem.new_cache(), attention_mask=mask
        ).logits.float()
        on_disk = Graft(
            k=loaded.k.to(device), v=loaded.v.to(device), meta=loaded.meta, path=artifact_path
        )
        logits_disk = model(
            input_ids=probe_ids, past_key_values=on_disk.new_cache(), attention_mask=mask
        ).logits.float()
    roundtrip_diff = (logits_mem - logits_disk).abs().max().item()

    print(f"[verify] bitwise tensor equality: {bitwise}")
    print(f"[verify] round-trip max-abs logit diff (loaded vs in-memory cache): {roundtrip_diff:.3e}")
    if prefill_text is not None:
        with torch.no_grad():
            prefill_ids = tokenizer(
                prefill_text, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(device)
            one_pass = model(input_ids=torch.cat([prefill_ids, probe_ids], dim=1)).logits.float()
        splice_diff = (logits_mem - one_pass[:, n_slots:]).abs().max().item()
        print(f"[verify] splice-vs-single-pass max-abs logit diff (informational, bf16): {splice_diff:.3e}")
    ok = bitwise and roundtrip_diff <= 1e-3
    print(f"[verify] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1
