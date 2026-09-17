"""Eval orchestration: refusal rate + base-vs-base KL, written as JSON/MD reports."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from phantom_kv.eval.metrics import GRAFT_NOT_READY, teacher_forced_kl
from phantom_kv.eval.refusal import is_refusal, matched_pattern
from phantom_kv.model import load_model

KL_SELF_TOLERANCE = 1e-3


def load_suite(path: str | Path) -> list[dict]:
    """Read a JSONL suite of {"id", "suite", "prompt"} rows."""
    items: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for key in ("id", "suite", "prompt"):
                if key not in row:
                    raise ValueError(f"{path}:{line_no}: suite row missing key {key!r}")
            items.append(row)
    return items


def chat_prompt(tokenizer, prompt: str) -> str:
    """Render the chat template, disabling thinking traces where supported."""
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_completion(
    model, tokenizer, device: str, prompt: str, max_new_tokens: int, graft: object = None
) -> tuple[list[int], list[int], str]:
    """One shared greedy path; returns (prompt_ids, new_token_ids, decoded text)."""
    if graft is not None:
        raise NotImplementedError(GRAFT_NOT_READY)
    prompt_tensor = tokenizer(chat_prompt(tokenizer, prompt), return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model.generate(
            prompt_tensor,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_ids = out[0, prompt_tensor.shape[1] :].tolist()
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return prompt_tensor[0].tolist(), new_ids, text


def suite_sha256(path: str | Path) -> str:
    """First 12 hex chars of the file's sha256, for run integrity."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def render_markdown(report: dict) -> str:
    """Compact scoreboard table for the run."""
    env = report["env"]
    lines = [
        f"# phantom-kv eval — {report['run_id']}",
        "",
        f"model `{env['model_id']}` on {env['device']}/{env['dtype']}"
        f" · torch {env['torch_version']} · transformers {env['transformers_version']}",
        "",
        "| suite | n | refusals | rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in ("harmful", "harmless"):
        agg = report["aggregate"][name]
        lines.append(f"| {name} | {agg['n']} | {agg['refusals']} | {agg['rate']:.2f} |")
    kl = report["kl"]
    lines += [
        "",
        f"KL base-vs-base (harmless, teacher-forced): mean {kl['mean']:.3e}, max {kl['max']:.3e}"
        f" — {'OK' if kl['ok'] else 'FAIL'} (tolerance {kl['tolerance']:.0e})",
        "",
        "suite integrity: "
        + ", ".join(
            f"{name} sha256:{row['sha256_12']}" for name, row in report["suite_integrity"].items()
        ),
        f"elapsed: {report['elapsed_seconds']}s",
        "",
    ]
    return "\n".join(lines)


def run_eval(
    model_id: str,
    harmful_path: str,
    harmless_path: str,
    max_new_tokens: int = 128,
    out_dir: str = "artifacts/eval",
    limit: int | None = None,
) -> dict:
    """Run the scoreboard and write run_<timestamp>.json / .md to out_dir."""
    started = time.perf_counter()
    model, tokenizer, env_info = load_model(model_id)
    device = env_info["device"]

    suites = {"harmful": load_suite(harmful_path), "harmless": load_suite(harmless_path)}
    if limit is not None:
        suites = {name: items[:limit] for name, items in suites.items()}

    per_item: dict[str, list[dict]] = {}
    aggregate: dict[str, dict] = {}
    harmless_pairs: list[tuple[list[int], list[int]]] = []
    for name, items in suites.items():
        suite_start = time.perf_counter()
        rows: list[dict] = []
        for item in items:
            prompt_ids, new_ids, text = generate_completion(
                model, tokenizer, device, item["prompt"], max_new_tokens
            )
            rows.append(
                {
                    "id": item["id"],
                    "prompt": item["prompt"],
                    "completion": text,
                    "refusal": is_refusal(text),
                    "pattern": matched_pattern(text),
                }
            )
            if name == "harmless":
                harmless_pairs.append((prompt_ids, new_ids))
            print(f"[eval] {name} {item['id']}: refusal={rows[-1]['refusal']}")
        refusals = sum(row["refusal"] for row in rows)
        per_item[name] = rows
        aggregate[name] = {
            "n": len(rows),
            "refusals": refusals,
            "rate": refusals / len(rows) if rows else 0.0,
            "elapsed_seconds": round(time.perf_counter() - suite_start, 3),
        }
        print(f"[eval] {name}: {refusals}/{len(rows)} refusals")

    kl_values = teacher_forced_kl(
        model,
        tokenizer,
        device,
        [p for p, _ in harmless_pairs],
        [c for _, c in harmless_pairs],
    )
    kl_mean = sum(kl_values) / len(kl_values) if kl_values else 0.0
    kl_max = max(kl_values) if kl_values else 0.0
    kl = {
        "arm": "base-vs-base",
        "suite": "harmless",
        "n": len(kl_values),
        "mean": kl_mean,
        "max": kl_max,
        "tolerance": KL_SELF_TOLERANCE,
        "ok": kl_max <= KL_SELF_TOLERANCE,
    }
    print(f"[eval] KL base-vs-base: mean={kl_mean:.3e} max={kl_max:.3e} ok={kl['ok']}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "run_id": stamp,
        "env": env_info,
        "config": {"max_new_tokens": max_new_tokens, "limit": limit},
        "aggregate": aggregate,
        "kl": kl,
        "suite_integrity": {
            "harmful": {"path": str(harmful_path), "sha256_12": suite_sha256(harmful_path)},
            "harmless": {"path": str(harmless_path), "sha256_12": suite_sha256(harmless_path)},
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "results": per_item,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"run_{stamp}.json"
    md_path = out / f"run_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"[eval] wrote {json_path}")
    print(f"[eval] wrote {md_path}")

    if not kl["ok"]:
        raise RuntimeError(
            f"KL base-vs-base max {kl_max:.3e} exceeds tolerance {KL_SELF_TOLERANCE:.0e}; "
            "see report for details"
        )

    result = {k: report[k] for k in ("run_id", "env", "config", "aggregate", "kl", "suite_integrity", "elapsed_seconds")}
    result["artifacts"] = {"json": str(json_path), "markdown": str(md_path)}
    return result
