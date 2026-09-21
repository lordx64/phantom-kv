"""donor-ce runtime: iterate a suite, re-judge every row the black run
flagged as compliant with the stricter harvest judge, donor-mine the rest.

Accepted strategy:
- a black-completion flip that passes the harvest judge keeps its ce row;
- rows where black refused, or whose completion is disclaimer junk, are
  donor-mined via the prefix stack on the same weights (prefill-attack data
  collection); cross-model donors can be added later without format changes.

Outputs (under --out prefix):
  <out>.accepted.jsonl   judged-clean completions keyed by prompt id
  <out>.attempts.jsonl  every candidate generation + verdict (provenance)
  <out>.needs_ce.jsonl  ids the stack could not satisfy, for manual review
"""

from __future__ import annotations

import json
from pathlib import Path

from phantom_kv.model import load_model
from phantom_kv.train.donors import (
    check_near_dup,
    harvest,
    judge_completion,
    load_donor_ce,
    write_harvest_attempt,
)


def run_donor_ce(
    model_id: str,
    suite_path: str,
    black_run_path: str,
    out_path: str,
    limit: int | None = None,
    max_new_tokens: int = 512,
) -> None:
    suite = [json.loads(l) for l in open(suite_path, encoding="utf-8") if l.strip()]
    black = json.loads(Path(black_run_path).read_text(encoding="utf-8"))
    black_rows = {r["id"]: r for r in black["results"]["harmful"]}
    if black.get("graft") is None:
        raise ValueError("--black-run must be a grafted (black pill) run report")

    accepted_path = str(out_path) + ".accepted.jsonl"
    needs_path = str(out_path) + ".needs_ce.jsonl"
    accepted_prior = load_donor_ce(accepted_path)

    model, tokenizer, info = load_model(model_id)
    device = info["device"]

    tasks: list[dict] = []
    for item in suite:
        if item["id"] in accepted_prior:
            continue
        row = black_rows.get(item["id"])
        if row is None:
            continue
        ok, _reason = judge_completion(row["completion"], item["prompt"])
        if not ok:
            tasks.append(item)
    if limit:
        tasks = tasks[:limit]
    print(f"[donor] {len(tasks)} rows need a donor completion")

    kept: list[str] = [v["completion"] for v in accepted_prior.values()]
    for idx, item in enumerate(tasks, 1):
        cand = harvest(model, tokenizer, device, item, graft=None,
                       max_new_tokens=max_new_tokens,
                       donor_label=model_id.split("/")[-1].replace(" ", "-"))
        for c in cand:
            write_harvest_attempt(out_path + ".attempts.jsonl", c)
        hit = next((c for c in cand if c["judge"][0]), None)
        if hit is None or check_near_dup(hit["completion"], kept):
            write_harvest_attempt(needs_path, {
                "id": item["id"], "prompt": item["prompt"],
                "why": "donor stack exhausted or near-duplicate",
            })
            continue
        row = {
            "id": item["id"],
            "prompt": item["prompt"],
            "completion": hit["completion"],
            "donor": hit["donor"],
            "prefix": hit["prefix"],
            "judge_score": 1.0,
        }
        write_harvest_attempt(accepted_path, row)
        kept.append(hit["completion"])
        print(f"[donor] {idx}/{len(tasks)} {item['id']}: accepted via {hit['donor']} prefix={hit['prefix']!r}")

    print(f"[donor] done: {len(accepted_prior)} accepted total "
          f"({sum(1 for _ in open(needs_path, encoding='utf-8')) if Path(needs_path).exists() else 0} needs_ce rows in {needs_path})")
