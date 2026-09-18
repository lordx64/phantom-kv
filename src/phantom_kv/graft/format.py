"""phantom.bin graft container: safetensors K/V bank plus JSON metadata sidecar.

Tensor layout in the container is [n_layers, n_slots, n_kv_heads, head_dim]
(transposed out of the cache-native [batch, heads, slots, dim] at build time).
Metadata sidecar <name>.json carries provenance and a sha256 over the raw
safetensors payload bytes, so tampering with either file fails validation.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

FORMAT_VERSION = 0
GRAFT_KIND = "prefill_kv"
SOFTPROMPT_KIND = "softprompt_kv"
DIRECTKV_KIND = "direct_kv"
VALID_KINDS = (GRAFT_KIND, SOFTPROMPT_KIND, DIRECTKV_KIND)
TENSOR_LAYOUT = "[n_layers, n_slots, n_kv_heads, head_dim]"

_META_KEYS = (
    "format_version",
    "kind",
    "model_id",
    "n_layers",
    "n_slots",
    "n_kv_heads",
    "head_dim",
    "dtype",
    "sha256",
    "source_sha256_12",
    "created_utc",
    "prefill_text",
    "layout",
)


def _payload_sha256(blob: bytes) -> str:
    """sha256 over tensor payload bytes only (strip the safetensors header)."""
    header_len = int.from_bytes(blob[:8], "little")
    return hashlib.sha256(blob[8 + header_len :]).hexdigest()


def validate_graft(meta: dict, tensor_shapes: dict[str, tuple[int, ...]]) -> None:
    """Fail loudly on tampered metadata or shape/kind mismatches."""
    for key in _META_KEYS:
        if key not in meta:
            raise ValueError(f"graft metadata missing key {key!r}")
    if meta["format_version"] != FORMAT_VERSION:
        raise ValueError(f"unsupported graft format_version {meta['format_version']}")
    if meta["kind"] not in VALID_KINDS:
        raise ValueError(f"unknown graft kind {meta['kind']!r}; this build supports {VALID_KINDS!r}")
    if set(tensor_shapes) != {"k", "v"}:
        raise ValueError(f"graft tensors must be exactly 'k' and 'v', got {sorted(tensor_shapes)}")
    shape_k, shape_v = tensor_shapes["k"], tensor_shapes["v"]
    if shape_k != shape_v or len(shape_k) != 4:
        raise ValueError(f"graft k/v shapes must match {TENSOR_LAYOUT}, got {shape_k} / {shape_v}")
    for field, actual in zip(("n_layers", "n_slots", "n_kv_heads", "head_dim"), shape_k):
        if meta[field] != actual:
            raise ValueError(f"graft metadata tampered: {field}={meta[field]} but tensor has {actual}")


class Graft:
    """Loaded prefill_kv bank. Holds stacked K/V and builds a fresh DynamicCache
    per forward, because forwards mutate the cache they are given in place."""

    def __init__(self, k, v, meta: dict, path: str | Path):
        self.k = k
        self.v = v
        self.meta = meta
        self.path = Path(path)

    @property
    def n_slots(self) -> int:
        return self.k.shape[1]

    @property
    def sha256_12(self) -> str:
        return self.meta["sha256"][:12]

    def new_cache(self):
        """Fresh DynamicCache holding the graft, in cache-native [1, H, N, D] per layer."""
        from transformers.cache_utils import DynamicCache

        cache = DynamicCache()
        for i in range(self.k.shape[0]):
            cache.update(
                self.k[i].permute(1, 0, 2).unsqueeze(0),
                self.v[i].permute(1, 0, 2).unsqueeze(0),
                i,
            )
        return cache


def save_graft(path: str | Path, k, v, meta_core: dict) -> dict:
    """Write <path> (safetensors) plus <path-with-.json> metadata sidecar; returns meta."""
    from safetensors.torch import save

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    k, v = k.contiguous().cpu(), v.contiguous().cpu()
    blob = save({"k": k, "v": v})
    meta = {
        "format_version": FORMAT_VERSION,
        "layout": TENSOR_LAYOUT,
        "sha256": _payload_sha256(blob),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dtype": str(k.dtype).removeprefix("torch."),
        "n_layers": k.shape[0],
        "n_slots": k.shape[1],
        "n_kv_heads": k.shape[2],
        "head_dim": k.shape[3],
        **meta_core,
    }
    validate_graft(meta, {"k": tuple(k.shape), "v": tuple(v.shape)})
    p.write_bytes(blob)
    p.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def load_graft(path: str | Path, device: str | None = None, dtype=None) -> Graft:
    """Load + fully validate a graft (metadata, shapes, payload sha256)."""
    from safetensors.torch import load_file

    p = Path(path)
    meta_path = p.with_suffix(".json")
    if not p.is_file() or not meta_path.is_file():
        raise ValueError(f"graft artifact incomplete: need {p} and {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    tensors = load_file(str(p))
    validate_graft(meta, {key: tuple(t.shape) for key, t in tensors.items()})
    if _payload_sha256(p.read_bytes()) != meta["sha256"]:
        raise ValueError("graft tensor payload tampered: sha256 mismatch")
    k, v = tensors["k"], tensors["v"]
    if device is not None:
        k, v = k.to(device), v.to(device)
    if dtype is not None:
        k, v = k.to(dtype), v.to(dtype)
    return Graft(k=k, v=v, meta=meta, path=p)
