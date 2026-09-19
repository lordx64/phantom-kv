"""v1 prefill_kv construction: shape the compliance prefill, one forward pass,
extract per-layer K/V into the phantom.bin container layout.

Shaping is derived from the tokenizer's own chat template (prefix/suffix split
over marker renders), so it ports across chat-template families instead of
assuming <|im_start|>. torch is imported lazily so shaping/validation stays
usable without a model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from phantom_kv.graft.format import GRAFT_KIND

_SYSTEM_MARKER = "__PHANTOM_S__"
_ACK_MARKER = "__PHANTOM_A__"
_PROMPT_MARKER = "__PHANTOM_P__"


def _model_label(tokenizer) -> str:
    return getattr(tokenizer, "name_or_path", None) or type(tokenizer).__name__


def _render(tokenizer, messages: list[dict]) -> str:
    """apply_chat_template with thinking disabled where the template supports it."""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def _template_parts(tokenizer) -> tuple[str, str]:
    """Split the chat template into (graft-prefix template, user-suffix template).

    Renders [system, assistant, user] and [user] with unique markers and
    requires the user-turn render to be a literal suffix of the full render;
    templates where later turns re-render earlier turns are rejected loudly.
    """
    suffix = _render(tokenizer, [{"role": "user", "content": _PROMPT_MARKER}])
    full = _render(
        tokenizer,
        [
            {"role": "system", "content": _SYSTEM_MARKER},
            {"role": "assistant", "content": _ACK_MARKER},
            {"role": "user", "content": _PROMPT_MARKER},
        ],
    )
    label = _model_label(tokenizer)
    if not full.endswith(suffix):
        raise ValueError(
            f"chat template of {label!r} re-renders earlier turns inside later turns; "
            "prefix/suffix split is not possible"
        )
    prefix = full[: -len(suffix)]
    for role, marker, text in (
        ("system", _SYSTEM_MARKER, prefix),
        ("assistant", _ACK_MARKER, prefix),
        ("user", _PROMPT_MARKER, suffix),
    ):
        if text.count(marker) != 1:
            raise ValueError(
                f"chat template of {label!r} does not contain exactly one {role} marker"
            )
    if _PROMPT_MARKER in prefix:
        raise ValueError(
            f"chat template of {label!r} leaks the user turn into the graft prefix"
        )
    return prefix, suffix


def shape_prefill(tokenizer, system: str, assistant_ack: str) -> str:
    """Graft-prefix render: system block + fake assistant acknowledgment, closed."""
    prefix, _ = _template_parts(tokenizer)
    return prefix.replace(_SYSTEM_MARKER, system).replace(_ACK_MARKER, assistant_ack)


def user_turn_suffix(tokenizer, prompt: str) -> str:
    """Grafted-arm prompt: only the user turn; the graft cache replaces the rest."""
    _, suffix = _template_parts(tokenizer)
    return suffix.replace(_PROMPT_MARKER, prompt)


def assert_chat_template_compatible(tokenizer) -> None:
    """Fail loudly (naming the model) if this tokenizer's template cannot be split."""
    _template_parts(tokenizer)


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
