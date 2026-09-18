"""Eval orchestration: refusal rate + KL arms, written as JSON/MD reports.

Ungrafted runs are the baseline scoreboard. Grafted runs (prefill_kv) score
both suites under the graft, regenerate the base harmless yardstick fresh
(base completions + base refusal control), and report KL(base || grafted)
teacher-forced on those yardstick completions.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from phantom_kv.eval.metrics import teacher_forced_kl
from phantom_kv.eval.refusal import is_refusal, matched_pattern
from phantom_kv.graft.build import user_turn_suffix
from phantom_kv.graft.format import Graft, load_graft
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
    model, tokenizer, device: str, prompt: str, max_new_tokens: int, graft: Graft | None = None
) -> tuple[list[int], list[int], str]:
    """One shared greedy path; returns (prompt_ids, new_token_ids, decoded text).

    Under a graft the prompt is only the user-turn suffix: the graft K/V
    occupies cache positions 0..n_slots, so HF derives positions from cache
    length and the mask simply covers cache + new tokens.
    """
    if graft is None:
        prompt_tensor = tokenizer(chat_prompt(tokenizer, prompt), return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model.generate(
                prompt_tensor,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
    else:
        prompt_tensor = tokenizer(
            user_turn_suffix(prompt), return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)
        mask = torch.ones(1, prompt_tensor.shape[1] + graft.n_slots, dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.generate(
                prompt_tensor,
                past_key_values=graft.new_cache(),
                attention_mask=mask,
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
    ]
    if report["graft"] is not None:
        graft = report["graft"]
        lines.append(
            f"graft `{Path(graft['path']).name}` ({graft['kind']},"
            f" {graft['n_slots']} slots, sha256:{graft['sha256_12']})"
        )
    lines += [
        "",
        "| suite | n | refusals | rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in ("harmful", "harmless"):
        agg = report["aggregate"][name]
        lines.append(f"| {name} | {agg['n']} | {agg['refusals']} | {agg['rate']:.2f} |")
    if report["base"] is not None:
        base = report["base"]["harmless"]
        lines += [
            "",
            f"base harmless (control, generated fresh this run): {base['refusals']}/{base['n']} refusals",
        ]
    kl = report["kl"]
    kl_line = f"KL {kl['arm']} (harmless, teacher-forced): mean {kl['mean']:.3e}, max {kl['max']:.3e}"
    if "ok" in kl:
        kl_line += f" — {'OK' if kl['ok'] else 'FAIL'} (tolerance {kl['tolerance']:.0e})"
    lines += [
        "",
        kl_line,
        "top-3 KL prompts: "
        + ", ".join(
            f"{pid}={value:.3e}"
            for pid, value in sorted(kl["per_prompt"].items(), key=lambda kv: kv[1], reverse=True)[:3]
        ),
        "",
        "suite integrity: "
        + ", ".join(
            f"{name} sha256:{row['sha256_12']}" for name, row in report["suite_integrity"].items()
        ),
        f"elapsed: {report['elapsed_seconds']}s",
        "",
    ]
    return "\n".join(lines)


def _aggregate(rows: list[dict], elapsed: float) -> dict:
    refusals = sum(row["refusal"] for row in rows)
    return {
        "n": len(rows),
        "refusals": refusals,
        "rate": refusals / len(rows) if rows else 0.0,
        "elapsed_seconds": round(elapsed, 3),
    }


def _score_suite(
    model,
    tokenizer,
    device: str,
    label: str,
    items: list[dict],
    max_new_tokens: int,
    graft: Graft | None,
    keep_pairs: bool,
) -> tuple[list[dict], list[tuple[list[int], list[int]]]]:
    """Greedy completions + refusal scoring for one suite (one arm)."""
    rows: list[dict] = []
    pairs: list[tuple[list[int], list[int]]] = []
    for item in items:
        prompt_ids, new_ids, text = generate_completion(
            model, tokenizer, device, item["prompt"], max_new_tokens, graft=graft
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
        if keep_pairs:
            pairs.append((prompt_ids, new_ids))
        print(f"[eval] {label} {item['id']}: refusal={rows[-1]['refusal']}")
    return rows, pairs


def run_eval(
    model_id: str,
    harmful_path: str,
    harmless_path: str,
    max_new_tokens: int = 128,
    out_dir: str = "artifacts/eval",
    limit: int | None = None,
    graft_path: str | None = None,
) -> dict:
    """Run the scoreboard and write run_<timestamp>.json / .md to out_dir."""
    started = time.perf_counter()
    model, tokenizer, env_info = load_model(model_id)
    device = env_info["device"]

    graft = None
    if graft_path is not None:
        graft = load_graft(graft_path, device=device, dtype=model.dtype)
        print(
            f"[eval] graft loaded: {graft_path} kind={graft.meta['kind']}"
            f" n_slots={graft.n_slots} sha256_12={graft.sha256_12}"
        )

    suites = {"harmful": load_suite(harmful_path), "harmless": load_suite(harmless_path)}
    if limit is not None:
        suites = {name: items[:limit] for name, items in suites.items()}

    per_item: dict[str, list[dict]] = {}
    aggregate: dict[str, dict] = {}
    base_control: dict | None = None

    if graft is None:
        pairs: list[tuple[list[int], list[int]]] = []
        harmless_rows: list[dict] = []
        for name, items in suites.items():
            suite_start = time.perf_counter()
            rows, new_pairs = _score_suite(
                model, tokenizer, device, name, items, max_new_tokens,
                graft=None, keep_pairs=name == "harmless",
            )
            pairs.extend(new_pairs)
            if name == "harmless":
                harmless_rows = rows
            per_item[name] = rows
            aggregate[name] = _aggregate(rows, time.perf_counter() - suite_start)
            print(f"[eval] {name}: {aggregate[name]['refusals']}/{len(rows)} refusals")
        yardstick = pairs
        yard_rows = harmless_rows
    else:
        # Yardstick first: fresh base harmless completions + base refusal control.
        suite_start = time.perf_counter()
        base_rows, yardstick = _score_suite(
            model, tokenizer, device, "harmless (base control)", suites["harmless"],
            max_new_tokens, graft=None, keep_pairs=True,
        )
        yard_rows = base_rows
        base_control = {"harmless": _aggregate(base_rows, time.perf_counter() - suite_start)}
        per_item["base_harmless"] = base_rows
        print(
            f"[eval] base harmless control: {base_control['harmless']['refusals']}"
            f"/{len(base_rows)} refusals"
        )
        for name in ("harmful", "harmless"):
            suite_start = time.perf_counter()
            rows, _ = _score_suite(
                model, tokenizer, device, f"{name} (grafted)", suites[name],
                max_new_tokens, graft=graft, keep_pairs=False,
            )
            per_item[name] = rows
            aggregate[name] = _aggregate(rows, time.perf_counter() - suite_start)
            print(f"[eval] {name} grafted: {aggregate[name]['refusals']}/{len(rows)} refusals")

    kl_ids: list[str] = []
    kl_prompts: list[list[int]] = []
    kl_comps: list[list[int]] = []
    for row, (prompt_ids, completion_ids) in zip(yard_rows, yardstick, strict=True):
        if not completion_ids:
            print(f"[eval] warning: {row['id']} has an empty completion; excluded from KL arm")
            continue
        kl_ids.append(row["id"])
        kl_prompts.append(prompt_ids)
        kl_comps.append(completion_ids)
    kl_values = teacher_forced_kl(model, tokenizer, device, kl_prompts, kl_comps, graft=graft)
    kl_by_id = dict(zip(kl_ids, kl_values, strict=True))
    kl_mean = sum(kl_values) / len(kl_values) if kl_values else 0.0
    kl_max = max(kl_values) if kl_values else 0.0
    kl = {"arm": "base-vs-base" if graft is None else "base-vs-graft",
          "suite": "harmless", "n": len(kl_values), "mean": kl_mean, "max": kl_max,
          "per_prompt": kl_by_id}
    if graft is None:
        kl["tolerance"] = KL_SELF_TOLERANCE
        kl["ok"] = kl_max <= KL_SELF_TOLERANCE
    kl_worst = max(kl_by_id, key=kl_by_id.get) if kl_by_id else None
    print(f"[eval] KL {kl['arm']}: mean={kl_mean:.3e} max={kl_max:.3e} (worst: {kl_worst})")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "run_id": stamp,
        "env": env_info,
        "config": {"max_new_tokens": max_new_tokens, "limit": limit},
        "graft": (
            None
            if graft is None
            else {
                "path": str(graft_path),
                "kind": graft.meta["kind"],
                "n_slots": graft.n_slots,
                "sha256_12": graft.sha256_12,
            }
        ),
        "aggregate": aggregate,
        "base": base_control,
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

    if kl.get("ok") is False:
        raise RuntimeError(
            f"KL base-vs-base max {kl_max:.3e} exceeds tolerance {KL_SELF_TOLERANCE:.0e}; "
            "see report for details"
        )

    result = {
        k: report[k]
        for k in ("run_id", "env", "config", "graft", "aggregate", "base", "kl", "suite_integrity", "elapsed_seconds")
    }
    result["artifacts"] = {"json": str(json_path), "markdown": str(md_path)}
    return result
