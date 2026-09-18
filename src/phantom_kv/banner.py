"""ASCII launch banner for phantom-kv CLIs."""

from __future__ import annotations

from phantom_kv import __version__

BANNER = r"""
      _             _               _                     _
 _ __| |_  __ _ _ _| |_ ___ _ __   | |____ __  __ __ _ __| |_  ___
| '_ \ ' \/ _` | ' \  _/ _ \ '  \  | / /\ V / / _/ _` / _| ' \/ -_)
| .__/_||_\__,_|_||_\__\___/_|_|_| |_\_\ \_/  \__\__,_\__|_||_\___|
|_|
"""


def render_banner(tool: str) -> str:
    """Banner text for tool `tool` (e.g. 'phantom-eval')."""
    art = BANNER.strip("\n")
    return f"{art}\n  {tool} v{__version__} — refusal removal as a loadable KV-cache graft\n"


def print_banner(tool: str) -> None:
    print(render_banner(tool))
