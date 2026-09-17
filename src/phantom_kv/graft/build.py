"""v1 prefill_kv construction: shape the compliance prefill, one forward pass,
extract per-layer K/V into the phantom.bin container layout.

torch is imported lazily so shaping/validation stays usable without a model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from phantom_kv.graft.format import GRAFT_KIND


def shape_prefill(system: str, assistant_ack: str) -> str:
    """Qwen3 chat layout: system block + fake assistant acknowledgment, closed."""
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_ack}<|im_end|>\n"
    )


def user_turn_suffix(prompt: str) -> str:
    """Grafted-arm prompt: only the user turn; the graft cache replaces the rest."""
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def load_prefill_source(path: str | Path) -> tuple[str, str]:
    """Read a v1_prefill.json-style source; returns (system, assistant_ack)."""
    src = json.loads(Path(path).read_text(encoding="utf-8"))
    if src.get("kind") != GRAFT_KIND:
        raise ValueError(f"graft source kind must be {GRAFT_KIND!r}, got {src.get('kind')!r}")
    for key in ("system", "assistant_ack"):
        if key not in src:
            raise ValueError(f"graft source missing key {key!r}")
    return src["system"], src["assistant_ack"]


def source_sha256_12(path: str | Path) -> str:
    """First 12 hex chars of the source file's sha256, for graft provenance."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def assert_chat_template_compatible(tokenizer) -> None:
    """Fail loudly if the tokenizer's chat template is not the <|im_start|> family."""
    template = getattr(tokenizer, "chat_template", None) or ""
    if "<|im_start|>" not in template:
        raise ValueError(
            "tokenizer chat template does not use <|im_start|>; "
            "refusing to shape a graft for this tokenizer"
        )


def extract_stacked_kv(past_key_values):
    """Cache-native per-layer [1, H, N, D] -> container [n_layers, n_slots, H, D]."""
    import torch

    k = torch.stack([layer.keys for layer in past_key_values.layers])
    v = torch.stack([layer.values for layer in past_key_values.layers])
    k = k.squeeze(1).permute(0, 2, 1, 3).contiguous()
    v = v.squeeze(1).permute(0, 2, 1, 3).contiguous()
    return k, v


def build_prefill_kv(model, tokenizer, device: str, prefill_text: str):
    """Tokenize the shaped prefill and cache it via one forward pass."""
    import torch

    ids = tokenizer(prefill_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
    return extract_stacked_kv(out.past_key_values)
