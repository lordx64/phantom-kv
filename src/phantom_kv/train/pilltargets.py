"""Pill (domain-selective graft) training targets — v2 with fail-closed ce.

A pill is trained to suppress refusal on exactly one domain suite while
KL-anchoring every other refusal domain plus the harmless yardstick. v2
rules (donor-pipeline era):

- every domain row gets ce or is skipped — never sup-only on the hard set,
  because sup-only is what manufactures paraphrased refusals;
- black-run compliance flips are kept ONLY when the candidate passes the
  harvest judge (the lexical refusal scorer under-counts disclaimer-first
  answers);
- rows black refused (or flagged disclaimer-junk) are covered by donor-ce
  rows harvested out-of-band (prefill stack, cross-model donors later);
- anything without a judge-clean completion is logged to needs_ce and
  skipped, so silent sup-only rows cannot reappear.

Roles: "ce" (compliance imitation), "sup" (hinged suppression), "kl"
(harmless + cross-domain preservation). Asserts a viable minimum, not a
pinned constitution.
"""

from __future__ import annotations

import json
from pathlib import Path

from phantom_kv.train.donors import judge_completion


class PillTargetsError(Exception):
    pass


def _load_run(path: str | Path, want_graft: bool) -> dict:
    run = json.loads(Path(path).read_text(encoding="utf-8"))
    if (run.get("graft") is not None) != want_graft:
        raise PillTargetsError(
            f"{path}: expected {'a grafted' if want_graft else 'an ungrafted base'} run"
        )
    return run


def build_pill_targets_v2(
    base_run_path: str | Path,
    flip_run_path: str | Path,
    kl_run_paths: list[str | Path],
    out_path: str | Path,
    donor_map: dict[str, dict] | None = None,
    needs_ce_out: str | Path | None = None,
    flip_judge=None,
) -> tuple[int, int]:
    """Write pill targets JSONL with fail-closed ce.

    Returns (rows_written, skipped_count).
    """
    judge = flip_judge or (lambda row: judge_completion(row["completion"], row.get("prompt", ""))[0])
    donor_map = donor_map or {}

    base = _load_run(base_run_path, want_graft=False)
    flip = _load_run(flip_run_path, want_graft=True)
    model_id = base["env"]["model_id"]
    if flip["env"]["model_id"] != model_id:
        raise PillTargetsError("base and flip runs disagree on model_id")

    base_rows = {r["id"]: r for r in base["results"]["harmful"]}
    flip_rows = {r["id"]: r for r in flip["results"]["harmful"]}
    if set(base_rows) != set(flip_rows):
        raise PillTargetsError("harmful id sets differ between base and flip runs")

    rows: list[dict] = []
    needs_ce: list[dict] = []
    for pid, brow in base_rows.items():
        frow = flip_rows[pid]
        donor = donor_map.get(pid)
        black_complies = not frow["refusal"]
        black_clean = black_complies and judge(frow)

        if not brow["refusal"]:
            # base complies: nothing to suppress; preserve with kl, imitate if judged clean
            rows.append({"id": pid, "role": "kl",
                         "prompt": brow["prompt"], "completion": brow["completion"]})
            if judge(frow) or not black_complies:
                continue
            rows.append({"id": pid, "role": "ce",
                         "prompt": brow["prompt"], "completion": frow["completion"],
                         "source": "fliprun-black-clean"})
            continue

        # base refuses: sup is always possible; ce decides whether the row trains or is skipped
        ce_row = None
        ce_source = None
        if black_clean:
            ce_row = frow["completion"]
            ce_source = "fliprun-black-clean"
        elif donor is not None:
            ce_row = donor["completion"]
            ce_source = donor.get("donor", "donor")

        if ce_row is None:
            # fail-closed: never sup-only on the hard set (sup-only is what
            # manufactures paraphrased refusals) — park the prompt in needs_ce
            needs_ce.append({
                "id": pid, "prompt": brow["prompt"],
                "black_refusal": frow["refusal"],
                "black_completion": frow["completion"][:400],
                "why": "no judge-clean ce from flip or donors",
            })
            continue
        rows.append({"id": pid, "role": "sup",
                     "prompt": brow["prompt"], "completion": brow["completion"]})
        rows.append({"id": pid, "role": "ce",
                     "prompt": brow["prompt"], "completion": ce_row,
                     "source": ce_source})

    # kl: harmless yardstick + all control suites
    seen_ids = set(base_rows)
    def add_kl(run: dict) -> None:
        for suite in ("harmful", "harmless"):
            for row in run["results"][suite]:
                if row["id"] in seen_ids or not row["completion"].strip():
                    continue
                seen_ids.add(row["id"])
                rows.append({"id": row["id"], "role": "kl",
                             "prompt": row["prompt"], "completion": row["completion"]})

    add_kl(base)
    for path in kl_run_paths:
        run = _load_run(path, want_graft=False)
        if run["env"]["model_id"] != model_id:
            raise PillTargetsError("control runs disagree on model_id")
        add_kl(run)

    counts = {role: sum(row["role"] == role for row in rows) for role in ("ce", "sup", "kl")}
    if counts["sup"] == 0:
        raise PillTargetsError("no base refusals on the domain suite: nothing to suppress")
    if counts["kl"] == 0:
        raise PillTargetsError("no KL control rows: guardrail anchoring would be empty")

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    if needs_ce_out:
        np_ = Path(needs_ce_out)
        np_.parent.mkdir(parents=True, exist_ok=True)
        with open(np_, "w", encoding="utf-8") as fh:
            for row in needs_ce:
                fh.write(json.dumps(row) + "\n")
    return len(rows), len(needs_ce)


def write_pill_targets_v2(
    base_run_path: str | Path,
    flip_run_path: str | Path,
    kl_run_paths: list[str | Path],
    out_path: str | Path,
    donor_ce_path: str | Path | None = None,
    needs_ce_out: str | Path | None = None,
) -> tuple[int, int]:
    from phantom_kv.train.donors import load_donor_ce

    donor_map = load_donor_ce(donor_ce_path) if donor_ce_path else {}
    return build_pill_targets_v2(
        base_run_path, flip_run_path, kl_run_paths, out_path,
        donor_map=donor_map, needs_ce_out=needs_ce_out,
    )


# ---- v1 (legacy, kept for the existing targets_red.jsonl / targets_blue.jsonl builders) ----

def _load_run_v1(path: str | Path, want_graft: bool) -> dict:
    run = json.loads(Path(path).read_text(encoding="utf-8"))
    if (run.get("graft") is not None) != want_graft:
        raise PillTargetsError(
            f"{path}: expected {'a grafted' if want_graft else 'an ungrafted base'} run"
        )
    return run


def build_pill_targets(
    base_run_path: str | Path,
    flip_run_path: str | Path,
    kl_run_paths: list[str | Path],
) -> list[dict]:
    """v1 rows: ce from base/flip (lexical), sup on base-refused, kl controls."""
    base = _load_run_v1(base_run_path, want_graft=False)
    flip = _load_run_v1(flip_run_path, want_graft=True)
    if base["env"]["model_id"] != flip["env"]["model_id"]:
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

    add_kl(base)
    for path in kl_run_paths:
        run = _load_run_v1(path, want_graft=False)
        if run["env"]["model_id"] != base["env"]["model_id"]:
            raise ValueError("control runs disagree on model_id")
        add_kl(run)

    counts = {role: sum(row["role"] == role for row in rows) for role in ("ce", "sup", "kl")}
    if counts["sup"] == 0:
        raise ValueError("no base refusals on the domain suite: a pill has nothing to suppress")
    if counts["kl"] == 0:
        raise ValueError("no KL control rows: guardrail anchoring would be empty")
    return rows


def write_pill_targets(base_run_path, flip_run_path, kl_run_paths, out_path) -> int:
    rows = build_pill_targets(base_run_path, flip_run_path, kl_run_paths)
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return len(rows)
