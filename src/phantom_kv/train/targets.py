"""v2 training targets: one JSONL distilled from the base and v1 scoreboard runs.

Roles: "ce" (compliance imitation: v1 flips + base complies), "sup" (drive down
the recorded base refusals), "kl" (harmless base completions for preservation).
Counts are asserted against the pinned scoreboard constitution.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

EXPECTED_COUNTS = {"ce": 45, "sup": 25, "kl": 20}


def targets_sha256_12(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def build_targets(base_run_path: str | Path, v1_run_path: str | Path) -> list[dict]:
    """Rows {"id", "role", "prompt", "completion"}; asserts 45/25/20."""
    base = json.loads(Path(base_run_path).read_text(encoding="utf-8"))
    v1 = json.loads(Path(v1_run_path).read_text(encoding="utf-8"))
    if base["env"]["model_id"] != v1["env"]["model_id"]:
        raise ValueError("run files disagree on model_id")
    if base.get("graft") is not None or v1.get("graft") is None:
        raise ValueError("need an ungrafted base run and a grafted v1 run")

    base_harmful = {r["id"]: r for r in base["results"]["harmful"]}
    v1_harmful = {r["id"]: r for r in v1["results"]["harmful"]}
    if set(base_harmful) != set(v1_harmful):
        raise ValueError("harmful id sets differ between base and v1 runs")

    rows: list[dict] = []
    for pid, brow in base_harmful.items():
        vrow = v1_harmful[pid]
        if brow["refusal"] and not vrow["refusal"]:
            rows.append({"id": pid, "role": "ce", "prompt": brow["prompt"], "completion": vrow["completion"]})
        elif not brow["refusal"]:
            rows.append({"id": pid, "role": "ce", "prompt": brow["prompt"], "completion": brow["completion"]})
        if brow["refusal"]:
            rows.append({"id": pid, "role": "sup", "prompt": brow["prompt"], "completion": brow["completion"]})
    for r in base["results"]["harmless"]:
        rows.append({"id": r["id"], "role": "kl", "prompt": r["prompt"], "completion": r["completion"]})

    counts = {role: sum(row["role"] == role for row in rows) for role in EXPECTED_COUNTS}
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"target constitution drifted: {counts} != {EXPECTED_COUNTS}")
    return rows


def write_targets(base_run_path: str | Path, v1_run_path: str | Path, out_path: str | Path) -> int:
    """Write the targets JSONL; returns row count."""
    rows = build_targets(base_run_path, v1_run_path)
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return len(rows)


def load_targets(path: str | Path) -> dict[str, list[dict]]:
    """Group targets by role."""
    grouped: dict[str, list[dict]] = {"ce": [], "sup": [], "kl": []}
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("role") not in grouped or "prompt" not in row or "completion" not in row:
                raise ValueError(f"{path}:{line_no}: malformed target row")
            grouped[row["role"]].append(row)
    return grouped
