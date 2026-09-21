"""Capability spot-check eval: GSM8K-style numeric and MMLU-style choice suites.

Complements the refusal-rate/KL scoreboard (runner.py) with a capability axis:
does the graft change what the model *can do*, not just whether it refuses?
Suites are JSONL rows of
    {"id", "suite", "prompt", "answer"}                     (numeric final answer)
    {"id", "suite", "prompt", "choices", "answer": "A".."D"} (multiple choice)
All arms share runner.py's chat-template framing and graft splice path
(graft K/V at slots 0..N, user-turn suffix, all-ones mask) and greedy decoding,
so capability arms are directly comparable to the refusal arms in a run report.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

from phantom_kv.eval.runner import generate_completion, suite_sha256
from phantom_kv.graft.format import Graft
from phantom_kv.graft.library import resolve_graft_payload
from phantom_kv.model import load_model

LETTERS = "ABCD"
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_CHOICE_RE = re.compile(r"(?<![A-Za-z])([A-D])(?![A-Za-z])")

ANSWER_INSTRUCTION = "Answer with the single letter of the correct choice."


def extract_final_number(completion: str) -> str | None:
    """Last integer/decimal in the completion, commas stripped (``1,234`` -> ``1234``)."""
    matches = _NUMBER_RE.findall(completion)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def _numbers_equal(candidate: str, gold: str) -> bool:
    """Exact match after comma stripping; Decimal equality covers ``42`` vs ``42.0``."""
    gold = gold.replace(",", "").strip()
    if candidate == gold:
        return True
    try:
        return Decimal(candidate) == Decimal(gold)
    except InvalidOperation:
        return False


def extract_choice_letter(completion: str, choices: list[str]) -> str | None:
    """First standalone A-D letter; else the letter of the first-occurring choice text."""
    match = _CHOICE_RE.search(completion)
    if match is not None:
        return match.group(1)
    lowered = completion.lower()
    best: tuple[int, str] | None = None
    for letter, choice in zip(LETTERS, choices):
        needle = choice.strip().lower()
        if not needle:
            continue
        index = lowered.find(needle)
        if index != -1 and (best is None or index < best[0]):
            best = (index, letter)
    return best[1] if best is not None else None


def score_completion(item: dict, completion: str) -> dict:
    """Score one completion against its gold answer; returns kind/extracted/correct."""
    if "choices" in item:
        letter = extract_choice_letter(completion, item["choices"])
        return {"kind": "choice", "extracted": letter, "correct": letter == item["answer"]}
    value = extract_final_number(completion)
    correct = value is not None and _numbers_equal(value, str(item["answer"]))
    return {"kind": "numeric", "extracted": value, "correct": correct}


def load_capability_suite(path: str | Path) -> tuple[str, list[dict]]:
    """Read a capability JSONL suite; returns (suite name, items). Fails loudly."""
    items: list[dict] = []
    suite_name: str | None = None
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for key in ("id", "suite", "prompt", "answer"):
                if key not in row:
                    raise ValueError(f"{path}:{line_no}: capability row missing key {key!r}")
            if suite_name is None:
                suite_name = row["suite"]
            elif row["suite"] != suite_name:
                raise ValueError(
                    f"{path}:{line_no}: mixed suite names {row['suite']!r} and {suite_name!r}"
                )
            if "choices" in row:
                choices = row["choices"]
                if not isinstance(choices, list) or len(choices) != len(LETTERS):
                    raise ValueError(
                        f"{path}:{line_no}: choices must be a list of {len(LETTERS)} strings"
                    )
                if not all(isinstance(c, str) for c in choices):
                    raise ValueError(f"{path}:{line_no}: choices must all be strings")
                if row["answer"] not in LETTERS[: len(choices)]:
                    raise ValueError(
                        f"{path}:{line_no}: choice answer must be one of"
                        f" {LETTERS[: len(choices)]!r}, got {row['answer']!r}"
                    )
            elif not isinstance(row["answer"], str):
                raise ValueError(f"{path}:{line_no}: numeric answer must be a string")
            items.append(row)
    if suite_name is None:
        raise ValueError(f"{path}: suite file is empty")
    return suite_name, items


def build_user_turn(item: dict) -> str:
    """User-turn text: the prompt as-is, or prompt + lettered choices for choice items."""
    prompt = item["prompt"]
    if "choices" not in item:
        return prompt
    lines = [prompt, ""]
    lines += [f"{letter}. {choice}" for letter, choice in zip(LETTERS, item["choices"])]
    lines += ["", ANSWER_INSTRUCTION]
    return "\n".join(lines)


def _aggregate(rows: list[dict], elapsed: float) -> dict:
    correct = sum(row["correct"] for row in rows)
    return {
        "n": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows) if rows else 0.0,
        "elapsed_seconds": round(elapsed, 3),
    }


def _score_arm(
    model,
    tokenizer,
    device: str,
    arm: str,
    suites: list[tuple[str, list[dict]]],
    max_new_tokens: int,
    graft: Graft | None,
) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    """Greedy completions + scoring for every suite under one arm."""
    per_item: dict[str, list[dict]] = {}
    aggregate: dict[str, dict] = {}
    for suite_name, items in suites:
        suite_start = time.perf_counter()
        rows: list[dict] = []
        for item in items:
            _, _, text = generate_completion(
                model, tokenizer, device, build_user_turn(item), max_new_tokens, graft=graft
            )
            scored = score_completion(item, text)
            rows.append(
                {
                    "id": item["id"],
                    "completion": text,
                    "gold": item["answer"],
                    "extracted": scored["extracted"],
                    "correct": scored["correct"],
                }
            )
            print(f"[capability] {arm} {item['id']}: correct={scored['correct']}")
        per_item[suite_name] = rows
        aggregate[suite_name] = _aggregate(rows, time.perf_counter() - suite_start)
        print(
            f"[capability] {arm} {suite_name}:"
            f" {aggregate[suite_name]['correct']}/{len(rows)} correct"
        )
    return per_item, aggregate


def _diffs(
    base_rows: dict[str, list[dict]], graft_rows: dict[str, list[dict]]
) -> dict[str, list[dict]]:
    """Items whose correctness flipped between arms, per suite."""
    diffs: dict[str, list[dict]] = {}
    for suite_name, rows in base_rows.items():
        flipped = []
        for base_row, graft_row in zip(rows, graft_rows[suite_name], strict=True):
            if base_row["correct"] != graft_row["correct"]:
                flipped.append(
                    {
                        "id": base_row["id"],
                        "base_correct": base_row["correct"],
                        "graft_correct": graft_row["correct"],
                    }
                )
        diffs[suite_name] = flipped
    return diffs


def render_markdown(report: dict) -> str:
    """Compact per-suite accuracy table plus flipped-item listing."""
    env = report["env"]
    lines = [
        f"# phantom-kv capability eval — {report['run_id']}",
        "",
        f"model `{env['model_id']}` on {env['device']}/{env['dtype']}"
        f" · torch {env['torch_version']} · transformers {env['transformers_version']}",
    ]
    if report["graft"] is not None:
        graft = report["graft"]
        ref = Path(graft["path"]).name + (f"#{graft['alias']}" if graft.get("alias") else "")
        lines.append(
            f"graft `{ref}` ({graft['kind']},"
            f" {graft['n_slots']} slots, sha256:{graft['sha256_12']})"
        )
    has_graft = report["graft"] is not None
    lines += ["", "| suite | n | base |" + (" graft | Δ |" if has_graft else ""), "| --- | ---: | ---: |" + (" ---: | ---: |" if has_graft else "")]
    for suite_name, arms in report["aggregate"].items():
        base = arms["base"]
        row = f"| {suite_name} | {base['n']} | {base['correct']}/{base['n']} ({base['accuracy']:.2f}) |"
        if has_graft:
            grafted = arms["graft"]
            delta = grafted["accuracy"] - base["accuracy"]
            row += f" {grafted['correct']}/{grafted['n']} ({grafted['accuracy']:.2f}) | {delta:+.2f} |"
        lines.append(row)
    if has_graft:
        for suite_name, flipped in report["diffs"].items():
            lines += ["", f"## {suite_name}: {len(flipped)} flipped items"]
            for flip in flipped:
                arrow = "✓ → ✗" if flip["base_correct"] else "✗ → ✓"
                lines.append(f"- {flip['id']}: {arrow}")
    lines += [
        "",
        "suite integrity: "
        + ", ".join(
            f"{name} sha256:{row['sha256_12']}" for name, row in report["suite_integrity"].items()
        ),
        f"elapsed: {report['elapsed_seconds']}s",
        "",
    ]
    return "\n".join(lines)


def run_capability(
    model_id: str,
    suite_paths: list[str],
    max_new_tokens: int = 256,
    out_dir: str = "artifacts/eval",
    limit: int | None = None,
    graft_path: str | None = None,
    graft_alias: str | None = None,
) -> dict:
    """Run base (and graft, if given) capability arms; write capability_<ts>.{json,md}."""
    started = time.perf_counter()
    model, tokenizer, env_info = load_model(model_id)
    device = env_info["device"]

    graft = None
    if graft_path is not None:
        graft = resolve_graft_payload(
            graft_path, graft_alias, model_id, device=device, dtype=model.dtype
        )
        print(
            f"[capability] graft loaded: {graft_path}"
            + (f" alias={graft.alias}" if graft.alias else "")
            + f" kind={graft.meta['kind']} n_slots={graft.n_slots} sha256_12={graft.sha256_12}"
        )

    suites: list[tuple[str, list[dict]]] = []
    seen: set[str] = set()
    for path in suite_paths:
        suite_name, items = load_capability_suite(path)
        if suite_name in seen:
            raise ValueError(f"duplicate suite name {suite_name!r} across {suite_paths}")
        seen.add(suite_name)
        if limit is not None:
            items = items[:limit]
        suites.append((suite_name, items))

    results: dict[str, dict[str, list[dict]]] = {}
    aggregate: dict[str, dict[str, dict]] = {name: {} for name, _ in suites}

    base_rows, base_agg = _score_arm(
        model, tokenizer, device, "base", suites, max_new_tokens, graft=None
    )
    results["base"] = base_rows
    for name, agg in base_agg.items():
        aggregate[name]["base"] = agg

    if graft is not None:
        graft_rows, graft_agg = _score_arm(
            model, tokenizer, device, "grafted", suites, max_new_tokens, graft=graft
        )
        results["graft"] = graft_rows
        for name, agg in graft_agg.items():
            aggregate[name]["graft"] = agg
    diffs = _diffs(results["base"], results["graft"]) if graft is not None else {}

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
                "alias": graft.alias,
                "kind": graft.meta["kind"],
                "n_slots": graft.n_slots,
                "sha256_12": graft.sha256_12,
            }
        ),
        "aggregate": aggregate,
        "diffs": diffs,
        "suite_integrity": {
            suite_name: {"path": str(path), "sha256_12": suite_sha256(path)}
            for path, (suite_name, _) in zip(suite_paths, suites, strict=True)
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "results": results,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"capability_{stamp}.json"
    md_path = out / f"capability_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"[capability] wrote {json_path}")
    print(f"[capability] wrote {md_path}")

    result = {
        k: report[k]
        for k in ("run_id", "env", "config", "graft", "aggregate", "diffs", "suite_integrity", "elapsed_seconds")
    }
    result["artifacts"] = {"json": str(json_path), "markdown": str(md_path)}
    return result


def main_capability(args: argparse.Namespace) -> None:
    """CLI dispatch (phantom-eval --capability SUITE.jsonl [...]); requires --model."""
    suites = list(getattr(args, "capability", None) or [])
    if not args.model:
        raise ValueError("--model is required for --capability")
    if not suites:
        raise ValueError("--capability requires at least one suite jsonl path")
    report = run_capability(
        model_id=args.model,
        suite_paths=suites,
        max_new_tokens=args.max_new_tokens,
        out_dir=args.out_dir,
        limit=args.limit,
        graft_path=args.graft,
        graft_alias=args.graft_alias,
    )
    for suite_name, arms in report["aggregate"].items():
        summary = f"[capability] {suite_name}: base {arms['base']['correct']}/{arms['base']['n']}"
        if "graft" in arms:
            summary += f", graft {arms['graft']['correct']}/{arms['graft']['n']}"
        print(summary)
    print(f"[capability] report: {report['artifacts']['json']}")


def self_test() -> int:
    """Model-free assertions over scoring, loading, and rendering; exit code 0/1."""
    import tempfile

    results: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        results.append((name, bool(cond)))

    choices = ["London", "Berlin", "Paris", "Madrid"]
    mmlu_item = {"id": "m", "suite": "capability_mmlu", "prompt": "Capital of France?",
                 "choices": choices, "answer": "C"}
    gsm_item = {"id": "g", "suite": "capability_gsm8k", "prompt": "What is 6*7?", "answer": "42"}

    s = score_completion(gsm_item, "First 3*4 = 12, then 12 - 5 = 7.\nThe answer is 7.")
    check("gsm-last-number", s["correct"] is False and s["extracted"] == "7" and s["kind"] == "numeric")
    s = score_completion(gsm_item, "6*7 = 42.\nThe answer is 42.")
    check("gsm-correct", s["correct"] is True and s["extracted"] == "42")
    s = score_completion(dict(gsm_item, answer="1200"), "The total is 1,200 dollars.")
    check("gsm-commas", s["correct"] is True and s["extracted"] == "1200")
    s = score_completion(dict(gsm_item, answer="10500"), "He pays 1,500 each for 7: 10,500 total.")
    check("gsm-last-comma", s["correct"] is True and s["extracted"] == "10500")
    s = score_completion(dict(gsm_item, answer="0.75"), "Rate: 0.75")
    check("gsm-decimal", s["correct"] is True)
    s = score_completion(dict(gsm_item, answer="42.0"), "Answer: 42")
    check("gsm-decimal-forms", s["correct"] is True)
    s = score_completion(gsm_item, "I cannot compute that.")
    check("gsm-no-number", s["correct"] is False and s["extracted"] is None)
    s = score_completion(dict(gsm_item, answer="-8"), "Net change: -8")
    check("gsm-negative", s["correct"] is True and s["extracted"] == "-8")

    s = score_completion(mmlu_item, "The answer is C.")
    check("mmlu-letter", s["correct"] is True and s["extracted"] == "C" and s["kind"] == "choice")
    s = score_completion(mmlu_item, "(B) is my pick.")
    check("mmlu-parens", s["correct"] is False and s["extracted"] == "B")
    s = score_completion(mmlu_item, "C is wrong. A is right.")
    check("mmlu-first-letter", s["extracted"] == "C")
    s = score_completion(mmlu_item, "a sure thing, definitely")
    check("mmlu-lower-a-ignored", s["extracted"] is None)
    s = score_completion(mmlu_item, "The capital of France is Paris.")
    check("mmlu-full-form", s["correct"] is True and s["extracted"] == "C")
    s = score_completion(mmlu_item, "I'm not sure.")
    check("mmlu-no-answer", s["correct"] is False and s["extracted"] is None)
    s = score_completion(mmlu_item, "London first, then Madrid.")
    check("mmlu-full-form-order", s["extracted"] == "A")

    rendered = build_user_turn(mmlu_item)
    lines = rendered.splitlines()
    check(
        "mmlu-prompt-shape",
        lines[0] == "Capital of France?"
        and lines[2:6] == ["A. London", "B. Berlin", "C. Paris", "D. Madrid"]
        and lines[-1] == ANSWER_INSTRUCTION,
    )
    check("gsm-prompt-asis", build_user_turn(gsm_item) == gsm_item["prompt"])

    with tempfile.TemporaryDirectory() as tmp:
        good = Path(tmp) / "good.jsonl"
        good.write_text(
            json.dumps(gsm_item) + "\n\n" + json.dumps(dict(gsm_item, id="g2")) + "\n",
            encoding="utf-8",
        )
        name, items = load_capability_suite(good)
        check("load-roundtrip", name == "capability_gsm8k" and [i["id"] for i in items] == ["g", "g2"])

        def rejects(rows: list[dict]) -> bool:
            path = Path(tmp) / "bad.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            try:
                load_capability_suite(path)
            except ValueError:
                return True
            return False

        check("load-missing-key", rejects([{"id": "x", "suite": "s", "prompt": "p"}]))
        check("load-mixed-suites", rejects([gsm_item, dict(gsm_item, id="g3", suite="other")]))
        check("load-3-choices", rejects([dict(mmlu_item, choices=choices[:3], answer="C")]))
        check("load-bad-letter", rejects([dict(mmlu_item, answer="E")]))
        check("load-int-answer", rejects([dict(gsm_item, answer=42)]))
        check("load-empty", rejects([]))

    check("diff-logic", _diffs(
        {"s": [{"id": "a", "correct": True}, {"id": "b", "correct": False}]},
        {"s": [{"id": "a", "correct": False}, {"id": "b", "correct": False}]},
    ) == {"s": [{"id": "a", "base_correct": True, "graft_correct": False}]})

    fake_report = {
        "run_id": "20260101T000000Z",
        "env": {"model_id": "m", "device": "cpu", "dtype": "bf16",
                "torch_version": "0", "transformers_version": "0"},
        "graft": None,
        "aggregate": {"capability_gsm8k": {"base": {"n": 2, "correct": 1, "accuracy": 0.5}}},
        "diffs": {},
        "suite_integrity": {"capability_gsm8k": {"path": "p", "sha256_12": "0" * 12}},
        "elapsed_seconds": 0.0,
    }
    md_base = render_markdown(fake_report)
    ok_base = "| capability_gsm8k | 2 | 1/2 (0.50) |" in md_base and "Δ" not in md_base
    graft_report = dict(
        fake_report,
        graft={"path": "x.bin", "alias": None, "kind": "direct_kv", "n_slots": 3, "sha256_12": "1" * 12},
        aggregate={"capability_gsm8k": {
            "base": {"n": 2, "correct": 1, "accuracy": 0.5},
            "graft": {"n": 2, "correct": 2, "accuracy": 1.0},
        }},
        diffs={"capability_gsm8k": [{"id": "g2", "base_correct": False, "graft_correct": True}]},
    )
    md_graft = render_markdown(graft_report)
    ok_graft = (
        "2/2 (1.00) | +0.50 |" in md_graft
        and "1 flipped items" in md_graft
        and "- g2: ✗ → ✓" in md_graft
    )
    check("md-render-base", ok_base)
    check("md-render-graft", ok_graft)

    failures = 0
    for name, ok in results:
        failures += not ok
        print(f"[self-test] {'PASS' if ok else 'FAIL'} {name}")
    print(f"[self-test] {len(results) - failures}/{len(results)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(self_test())
