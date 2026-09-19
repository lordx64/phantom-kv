"""Pill matrix: arms x suites refusal cross-table, assembled from run reports.

Each cell is one run's harmful-suite refusal count: rows are arms (base, or a
graft named by its library alias / file stem), columns are harmful suites named
by file stem. The matrix is the pill-program scoreboard: pills want ~0 on their
own domain column and base-equal counts on every other column (leakage). The
harmless and KL columns summarize each arm's worst preservation across runs.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

_ARM_ORDER = ("base", "red", "blue", "black")


def _arm_of(run: dict) -> str:
    graft = run.get("graft")
    if graft is None:
        return "base"
    return graft.get("alias") or Path(graft["path"]).stem


def build_matrix(run_paths: list[str | Path]) -> dict:
    """Latest-run-wins cells keyed (arm, suite); arm-level worst harmless/KL."""
    cells: dict[tuple[str, str], dict] = {}
    arms: dict[str, dict] = {}
    suites: list[str] = []
    for path in run_paths:
        run = json.loads(Path(path).read_text(encoding="utf-8"))
        arm, suite = _arm_of(run), Path(run["suite_integrity"]["harmful"]["path"]).stem
        if suite not in suites:
            suites.append(suite)
        agg = run["aggregate"]["harmful"]
        cell = {
            "run_id": run["run_id"],
            "refusals": agg["refusals"],
            "n": agg["n"],
            "harmless_refusals": run["aggregate"]["harmless"]["refusals"],
            "harmless_n": run["aggregate"]["harmless"]["n"],
            "kl_mean": run["kl"]["mean"],
            "kl_max": run["kl"]["max"],
        }
        key = (arm, suite)
        if key not in cells or cells[key]["run_id"] < run["run_id"]:
            cells[key] = cell
            summary = arms.setdefault(
                arm, {"harmless_refusals": 0, "harmless_n": 0, "kl_mean_max": 0.0, "kl_max": 0.0}
            )
            summary["harmless_refusals"] = max(summary["harmless_refusals"], cell["harmless_refusals"])
            summary["harmless_n"] = max(summary["harmless_n"], cell["harmless_n"])
            summary["kl_mean_max"] = max(summary["kl_mean_max"], cell["kl_mean"])
            summary["kl_max"] = max(summary["kl_max"], cell["kl_max"])
    ordered_arms = [a for a in _ARM_ORDER if a in arms] + sorted(a for a in arms if a not in _ARM_ORDER)
    return {"cells": cells, "arms": arms, "arm_order": ordered_arms, "suites": suites}


def render_matrix(matrix: dict) -> str:
    cells, arms = matrix["cells"], matrix["arms"]
    suites = matrix["suites"]
    lines = [
        "# phantom-kv pill matrix",
        "",
        "harmful-suite refusals per arm (base / red / blue / black...), worst-case",
        "harmless refusals and harmless KL per arm in the right-hand columns.",
        "A selective pill scores ~0 on its own domain column and == base elsewhere.",
        "",
        "| arm | " + " | ".join(suites) + " | harmless | KL mean | KL max |",
        "| --- | " + " | ".join("---:" for _ in suites) + " | ---: | ---: | ---: |",
    ]
    for arm in matrix["arm_order"]:
        row = [arm]
        for suite in suites:
            cell = cells.get((arm, suite))
            row.append(f"{cell['refusals']}/{cell['n']}" if cell else "—")
        summary = arms[arm]
        row.append(f"{summary['harmless_refusals']}/{summary['harmless_n']}")
        row.append(f"{summary['kl_mean_max']:.2e}")
        row.append(f"{summary['kl_max']:.2e}")
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "cells: latest run report per (arm, suite); — = not measured", ""]
    return "\n".join(lines)


def write_matrix(run_paths: list[str | Path], out_dir: str | Path) -> Path:
    """Render the matrix and write matrix_<utc-ts>.md into out_dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md_path = out / f"matrix_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.md"
    md_path.write_text(render_matrix(build_matrix(run_paths)), encoding="utf-8")
    return md_path
