"""phantom-chat: interactive side-by-side demo of base vs KV-cache-grafted model.

Single-turn semantics per input: each prompt is fresh — NO history accumulation.
Base arm = tokenizer chat template; graft arm = eval splice layout
(graft cache + user-turn suffix) with all-ones mask and positions from cache
length, exactly mirroring eval/generation paths. Greedy decoding throughout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from phantom_kv.graft.library import LibError, preflight

_COMMANDS = "/q — quit, /mode both|graft|base, /pill <alias>|none — hot-swap graft, /help"


def _header(title: str) -> str:
    return f"── {title} " + "─" * max(4, 36 - len(title))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="phantom-chat",
        description="Interactive side-by-side demo: base vs phantom-kv grafted model.",
    )
    parser.add_argument("--model", required=True, help="HF model id or local path")
    parser.add_argument("--graft", default=None, help="phantom.bin graft artifact or phantom.lib library")
    parser.add_argument("--graft-alias", default=None, help="entry alias inside a .lib library payload")
    parser.add_argument("--mode", choices=["both", "graft", "base"], default="both")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--system", default=None, help="extra system prompt applied to BOTH arms")
    args = parser.parse_args()

    mode = args.mode
    lib_aliases: list[str] = []
    if args.graft is None:
        if mode != "base":
            print("[chat] note: no --graft supplied; running base-only")
        mode = "base"
    elif not Path(args.graft).is_file():
        print(f"[chat] error: graft file not found: {args.graft}", file=sys.stderr)
        sys.exit(1)
    elif args.graft_alias is not None and not args.graft.endswith(".lib"):
        print("[chat] error: --graft-alias applies only to .lib payloads", file=sys.stderr)
        sys.exit(1)
    elif args.graft.endswith(".lib"):
        try:
            _n, lib_aliases = preflight(args.graft, args.graft_alias)
        except LibError as err:
            print(f"[chat] error: {err}", file=sys.stderr)
            sys.exit(1)
        if args.graft_alias is not None and args.graft_alias not in lib_aliases:
            print(
                f"[chat] error: graft alias {args.graft_alias!r} not found;"
                f" available aliases: {', '.join(lib_aliases)}",
                file=sys.stderr,
            )
            sys.exit(1)

    from phantom_kv.banner import print_banner
    from phantom_kv.graft.library import resolve_graft_payload
    from phantom_kv.model import load_model

    try:
        model, tokenizer, info = load_model(args.model)
        graft = None
        if args.graft is not None:
            try:
                graft = resolve_graft_payload(
                    args.graft, args.graft_alias, args.model, device=info["device"], dtype=model.dtype
                )
            except LibError as err:
                if not (args.graft.endswith(".lib") and args.graft_alias is None):
                    raise
                print(f"[chat] no pill loaded at start ({err}); use /pill <alias> to activate one")
            if graft is not None:
                label = Path(graft.path).name + (f"#{graft.alias}" if graft.alias else "")
                print(f"[chat] graft loaded: {label} kind={graft.meta['kind']} n_slots={graft.n_slots}")
    except (LibError, ValueError) as err:
        print(f"[chat] error: {err}", file=sys.stderr)
        sys.exit(1)

    print_banner("phantom-chat")
    if mode != "base" and graft is None:
        mode = "base"

    import torch

    from phantom_kv.graft.build import user_turn_suffix

    device = info["device"]

    def _frame_base(prompt: str) -> str:
        messages = ([{"role": "system", "content": args.system}] if args.system else []) + [
            {"role": "user", "content": prompt}
        ]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _frame_system_turn() -> str:
        messages = [{"role": "system", "content": args.system}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    def _generate(prompt: str, arm: str) -> str:
        """arm: 'base' or 'graft'; returns decoded completion text."""
        if arm == "base":
            ids = tokenizer(_frame_base(prompt), return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                out = model.generate(
                    ids,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                )
        else:
            text = user_turn_suffix(tokenizer, prompt)
            if args.system:
                text = _frame_system_turn() + text
            ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            mask = torch.ones(1, ids.shape[1] + graft.n_slots, dtype=torch.long, device=device)
            with torch.no_grad():
                out = model.generate(
                    ids,
                    past_key_values=graft.new_cache(),
                    attention_mask=mask,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                )
        new_ids = out[0, ids.shape[1] :].tolist()
        return tokenizer.decode(new_ids, skip_special_tokens=True)

    graft_label = (
        Path(graft.path).name + (f"#{graft.alias}" if graft.alias else "") if graft is not None else None
    )

    print(f"[chat] model={args.model} mode={mode}"
          + (f" graft={graft_label}" if graft else "")
          + f" max_new_tokens={args.max_new_tokens} (single-turn, greedy) — {_COMMANDS}")
    while True:
        try:
            line = input("phantom> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.startswith("/"):
            groups: list[list[str]] = []
            for tok in line.split():
                if tok.startswith("/"):
                    groups.append([tok])
                elif groups:
                    groups[-1].append(tok)
                else:
                    groups.append([tok])
            quit_chat = False
            for group in groups:
                if not group:
                    continue
                cmd, rest = group[0], group[1:]
                if cmd == "/q":
                    quit_chat = True
                    break
                if cmd == "/help":
                    print(f"[chat] commands: {_COMMANDS}")
                    continue
                if cmd == "/mode":
                    if not rest or rest[0] not in ("both", "graft", "base"):
                        print("[chat] usage: /mode both|graft|base")
                        continue
                    if rest[0] in ("both", "graft") and args.graft is None:
                        print("[chat] cannot switch: no graft supplied at start (base-only mode)")
                        continue
                    if len(rest) > 1:
                        print(f"[chat] note: ignoring extra arguments: {' '.join(rest[1:])}")
                    mode = rest[0]
                    print(f"[chat] mode={mode}")
                    continue
                if cmd == "/pill":
                    if not (args.graft and args.graft.endswith(".lib")):
                        print("[chat] pill switching needs a multi-payload library (--graft <name>.lib)")
                        continue
                    if not rest:
                        active = graft_label or "(none — graft arm runs base behavior)"
                        print(f"[chat] active pill: {active}; available: {', '.join(lib_aliases)}, none")
                        continue
                    if rest[0] == "none":
                        graft = None
                        graft_label = None
                        print("[chat] pill cleared: no active graft")
                        if mode == "graft":
                            mode = "base"
                            print(f"[chat] mode={mode}")
                        continue
                    if rest[0] not in lib_aliases:
                        print(f"[chat] unknown pill {rest[0]!r}; available: {', '.join(lib_aliases)}")
                        continue
                    if len(rest) > 1:
                        print(f"[chat] note: ignoring extra arguments: {' '.join(rest[1:])}")
                    try:
                        graft = resolve_graft_payload(
                            args.graft, rest[0], args.model, device=device, dtype=model.dtype
                        )
                    except LibError as err:
                        print(f"[chat] error: {err}")
                        continue
                    graft_label = Path(graft.path).name + (f"#{graft.alias}" if graft.alias else "")
                    print(f"[chat] pill active: {graft_label} kind={graft.meta['kind']} n_slots={graft.n_slots}")
                    if mode == "base":
                        mode = "both"
                        print(f"[chat] mode={mode}")
                    continue
                print(f"[chat] unknown command; {_COMMANDS}")
                continue
            if quit_chat:
                break
            continue
        if mode in ("both", "base"):
            print(_header("base"))
            print(_generate(line, "base"))
        if mode in ("both", "graft"):
            if graft is None:
                print(_header("graft"))
                print("(no active pill — /pill <alias> to load one)")
            else:
                print(_header(f"graft ({graft_label})"))
                print(_generate(line, "graft"))


if __name__ == "__main__":
    main()
