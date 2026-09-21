"""Serving adapters: the load-once cache-block deployment story for grafts.

The base model loads exactly once per process; per-request capability modes
(pills) are hot-swapped as in-memory cache blocks from a phantom.lib, never by
reloading weights.

Public names are re-exported lazily so `python -m phantom_kv.serve.session`
stays warning-free and importing the package pulls in nothing model-heavy.
"""

from phantom_kv.graft.library import LibError

__all__ = ["DETACH", "LibError", "PhantomSession", "self_test"]


def __getattr__(name: str):
    if name in ("DETACH", "PhantomSession", "self_test"):
        import phantom_kv.serve.session as _session

        return getattr(_session, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
