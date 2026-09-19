"""Pill (domain-selective graft) training targets.

A pill is trained to suppress refusal on exactly one domain suite while
KL-anchoring every other refusal domain plus the harmless yardstick:

- ce/sup rows: distilled from the base and flip (grafted) runs on the pill's
  domain suite, same rule as the v2 builder — flips and base-complies rows
  teach compliance, base refusals become suppression hinge rows;
- kl rows: the base run's harmless completions plus, for every control base
  run, ALL completions (harmful refusals included — anchoring them teaches the
  graft to leave that domain's guardrails exactly where the base model put them).

Counts are domain-dependent; instead of a pinned constitution the builder
asserts a viable minimum and reports the constitution it wrote.
"""

from __future__ import annotations

import json
from pathlib import Path


def _load_run(path: str | Path, want_graft: bool) -> dict:
    run = json.loads(Path(path).read_text(encoding="utf-8"))
    if (run.get("graft") is not None) != want_graft:
        raise ValueError(
            f"{path}: expected {'a grafted' if want_graft else 'an ungrafted base'} run"
        )
    return run


def build_pill_targets(
    base_run_path: str | Path,
    flip_run_path: str | Path,
    kl_run_paths: list[str | Path],
) -> list[dict]:
    """Rows {"id", "role", "prompt", "completion"} for one pill's domain."""
    base = _load_run(base_run_path, want_graft=False)
    flip = _load_run(flip_run_path, want_graft=True)
    model_id = base["env"]["model_id"]
    if flip["env"]["model_id"] != model_id:
        raise ValueError("base and flip runs disagree on model_id")

    base_rows = {r["id"]: r for r in base["results"]["harmful"]}
    flip_rows = {r["id"]: r for r in flip["results"]["harmful"]}
    if set(base_rows) != set(flip_rows):
        raise ValueError("harmful id sets differ between base and flip runs")

    rows: list[dict] = []
    for pid, brow in base_rows.items():
        frow = flip_rows[pid]
        if brow["refusal"] and not frow["refusal"]:
            rows.append({"id": pid, "role": "ce", "prompt": brow["prompt"], "completion": frow["completion"]})
        elif not brow["refusal"]:
            rows.append({"id": pid, "role": "ce", "prompt": brow["prompt"], "completion": brow["completion"]})
        if brow["refusal"]:
            rows.append({"id": pid, "role": "sup", "prompt": brow["prompt"], "completion": brow["completion"]})

    seen_ids = set(base_rows)

    def add_kl(run: dict) -> None:
        for suite in ("harmful", "harmless"):
            for row in run["results"][suite]:
                if row["id"] in seen_ids or not row["completion"].strip():
                    continue
                seen_ids.add(row["id"])
                rows.append({"id": row["id"], "role": "kl",
                             "prompt": row["prompt"], "completion": row["completion"]})

    add_kl(base)  # harmless yardstick completions ride along in every run
    for path in kl_run_paths:
        run = _load_run(path, want_graft=False)
        if run["env"]["model_id"] != model_id:
            raise ValueError("control runs disagree on model_id")
        add_kl(run)

    counts = {role: sum(row["role"] == role for row in rows) for role in ("ce", "sup", "kl")}
    if counts["sup"] == 0:
        raise ValueError("no base refusals on the domain suite: a pill has nothing to suppress")
    if counts["ce"] == 0:
        raise ValueError("no compliant completions on the domain suite: pill cannot learn the target behavior")
    if counts["kl"] == 0:
        raise ValueError("no KL control rows: guardrail anchoring would be empty")
    return rows


def write_pill_targets(
    base_run_path: str | Path,
    flip_run_path: str | Path,
    kl_run_paths: list[str | Path],
    out_path: str | Path,
) -> int:
    """Write the pill targets JSONL; returns row count."""
    rows = build_pill_targets(base_run_path, flip_run_path, kl_run_paths)
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return len(rows)
