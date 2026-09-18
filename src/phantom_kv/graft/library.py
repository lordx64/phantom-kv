"""phantom.lib: one safetensors file packaging multiple per-model graft payloads.

Tensors are namespaced `<alias_key>.k` / `<alias_key>.v` (alias_key = alias
sanitized to alnum + `_-.`); the manifest lives in `<lib>.json`. Entries carry
per-entry payload sha256 (same tamper discipline as single-graft phantom.bin)
plus the source sidecar embedded in full. Single-graft phantom.bin is unchanged.
"""

from __future__ import annotations

import builtins
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from phantom_kv.graft.format import _payload_sha256, load_graft

LIB_FORMAT = "phantom.lib"
LIB_VERSION = 0
_ENTRY_KEYS = (
    "alias", "alias_key", "model_id", "kind", "n_layers", "n_slots", "n_kv_heads",
    "head_dim", "dtype", "sha256", "trained", "created_utc", "source_sidecar",
)
_SANITIZE = re.compile(r"[^A-Za-z0-9_\-.]")


class LibError(Exception):
    pass


def _manifest_path(lib_path: str | Path) -> Path:
    return Path(str(lib_path) + ".json")


def sanitize(alias: str) -> str:
    key = _SANITIZE.sub("_", alias)
    if not key:
        raise LibError(f"alias {alias!r} sanitizes to an empty key")
    return key


def _read_manifest(lib_path: str | Path) -> list[dict]:
    p = _manifest_path(lib_path)
    if not p.is_file():
        if Path(lib_path).is_file():
            raise LibError(f"library artifact incomplete: {lib_path} exists without {p}")
        return []
    manifest = json.loads(p.read_text(encoding="utf-8"))
    if manifest.get("format") != LIB_FORMAT or manifest.get("version") != LIB_VERSION:
        raise LibError(f"unsupported library manifest in {p}")
    entries = manifest.get("entries")
    if not isinstance(entries, builtins.list):
        raise LibError(f"library manifest in {p} has no entries list")
    for entry in entries:
        for key in _ENTRY_KEYS:
            if key not in entry:
                raise LibError(f"library manifest entry missing key {key!r}")
    return entries


def _write(lib_path: str | Path, tensors: dict, entries: list[dict]) -> None:
    from safetensors.torch import save

    p = Path(lib_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ordered = {}
    for entry in entries:  # tensor bytes are rewritten in manifest order
        ordered[f"{entry['alias_key']}.k"] = tensors[f"{entry['alias_key']}.k"]
        ordered[f"{entry['alias_key']}.v"] = tensors[f"{entry['alias_key']}.v"]
    p.write_bytes(save(ordered))
    manifest = {"format": LIB_FORMAT, "version": LIB_VERSION, "entries": entries}
    _manifest_path(p).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _load_tensors(lib_path: str | Path) -> dict:
    from safetensors.torch import load_file

    if not Path(lib_path).is_file():
        raise LibError(f"library file not found: {lib_path}")
    return load_file(str(lib_path))


def _entry_sha256(k, v) -> str:
    """Payload sha of one entry's tensor bytes, independent of key namespace."""
    from safetensors.torch import save

    return _payload_sha256(save({"k": k.contiguous().cpu(), "v": v.contiguous().cpu()}))


def list(lib_path: str | Path) -> list[dict]:
    """Manifest entries (alias, model_id, kind, dims, trained, sha256)."""
    p = Path(lib_path)
    if not (p.is_file() or _manifest_path(p).is_file()):
        raise LibError(f"library file not found: {lib_path}")
    return _read_manifest(p)


def add(lib_path: str | Path, graft_path: str | Path, alias: str | None = None, replace: bool = False) -> dict:
    """Copy a phantom.bin payload into the library under an alias; returns the entry."""
    graft = load_graft(graft_path)
    alias = alias if alias is not None else graft.meta["model_id"]
    key = sanitize(alias)
    p = Path(lib_path)
    entries = _read_manifest(p) if (p.is_file() or _manifest_path(p).is_file()) else []
    tensors = _load_tensors(p) if p.is_file() else {}

    for entry in entries:
        if entry["alias_key"] == key and entry["alias"] != alias:
            raise LibError(f"alias {alias!r} collides with {entry['alias']!r} on key {key!r}")
    existing = next((e for e in entries if e["alias"] == alias), None)
    if existing is not None and not replace:
        raise LibError(f"alias {alias!r} already present in {lib_path}; pass replace=True")

    tensors[f"{key}.k"] = graft.k.contiguous().cpu()
    tensors[f"{key}.v"] = graft.v.contiguous().cpu()
    entry = {
        "alias": alias,
        "alias_key": key,
        "model_id": graft.meta["model_id"],
        "kind": graft.meta["kind"],
        "n_layers": int(graft.k.shape[0]),
        "n_slots": int(graft.k.shape[1]),
        "n_kv_heads": int(graft.k.shape[2]),
        "head_dim": int(graft.k.shape[3]),
        "dtype": str(graft.k.dtype).removeprefix("torch."),
        "sha256": _entry_sha256(graft.k, graft.v),
        "trained": graft.meta.get("trained", False),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sidecar": graft.meta,
    }
    if existing is not None:
        entries[entries.index(existing)] = entry
    else:
        entries.append(entry)
    _write(p, tensors, entries)
    return entry


def remove(lib_path: str | Path, alias: str) -> None:
    p = Path(lib_path)
    entries = _read_manifest(p)
    keep = [e for e in entries if e["alias"] != alias]
    if len(keep) == len(entries):
        raise LibError(f"alias {alias!r} not found in {lib_path}")
    tensors = _load_tensors(p)
    _write(p, tensors, keep)


def aliases_for(lib_path: str | Path, model_id: str) -> list[str]:
    return [e["alias"] for e in _read_manifest(lib_path) if e["model_id"] == model_id]


def resolve(lib_path: str | Path, alias: str | None = None):
    """Load one entry's tensors (sha-verified) plus its source sidecar.

    alias=None is allowed only for a single-entry library.
    """
    p = Path(lib_path)
    entries = _read_manifest(p)
    if not entries:
        raise LibError(f"library {lib_path} has no graft entries")
    if alias is None:
        if len(entries) == 1:
            alias = entries[0]["alias"]
        else:
            have = ", ".join(sorted(e["alias"] for e in entries))
            raise LibError(f"library {lib_path} holds {len(entries)} entries; an alias is required: {have}")
    entry = next((e for e in entries if e["alias"] == alias), None)
    if entry is None:
        have = ", ".join(sorted(e["alias"] for e in entries))
        raise LibError(f"alias {alias!r} not found in {lib_path}; available: {have}")

    from safetensors import safe_open

    with safe_open(str(p), framework="pt") as fh:
        k = fh.get_tensor(f"{entry['alias_key']}.k")
        v = fh.get_tensor(f"{entry['alias_key']}.v")
    if tuple(k.shape) != (entry["n_layers"], entry["n_slots"], entry["n_kv_heads"], entry["head_dim"]):
        raise LibError(f"entry {alias!r} dims mismatch between manifest and tensors")
    if _entry_sha256(k, v) != entry["sha256"]:
        raise LibError(f"entry {alias!r} payload tampered: sha256 mismatch")
    return k, v, entry["source_sidecar"]


def resolve_graft_payload(graft_path, alias, model_id: str, device=None, dtype=None):
    """Loader entry for .bin and .lib; fail-closed alias/model resolution."""
    from phantom_kv.graft.format import Graft

    path = Path(graft_path)
    if path.suffix != ".lib":
        if alias is not None:
            raise LibError("--graft-alias applies only to .lib payloads")
        return load_graft(path, device=device, dtype=dtype)

    if alias is not None:
        k, v, sidecar = resolve(path, alias)
        if sidecar["model_id"] != model_id:
            raise LibError(
                f"entry {alias!r} was built for {sidecar['model_id']!r}, but --model is {model_id!r}"
            )
    else:
        matches = aliases_for(path, model_id)
        if len(matches) == 0:
            entries = _read_manifest(path)
            detail = ", ".join(sorted(f"{e['alias']} ({e['model_id']})" for e in entries))
            raise LibError(
                f"library {path} has no entry for model {model_id!r}; available: {detail or '(empty)'}"
            )
        if len(matches) > 1:
            raise LibError(
                f"library {path} has {len(matches)} entries for model {model_id!r};"
                f" pass --graft-alias: {', '.join(sorted(matches))}"
            )
        alias = matches[0]
        k, v, sidecar = resolve(path, alias)

    graft = Graft(k=k, v=v, meta=sidecar, path=path, alias=alias)
    if device is not None or dtype is not None:
        graft.place(device, dtype)
    return graft


def preflight(lib_path: str | Path, alias: str | None) -> tuple[int, list[str]]:
    """Cheap manifest check for the CLI: (entry count, aliases)."""
    entries = list(lib_path)
    return len(entries), sorted(e["alias"] for e in entries)
