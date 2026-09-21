"""phantom-serve: HF-transformers reference serving adapter (load-once cache blocks).

Deployment story: one process holds the base model exactly ONCE, with a
phantom.lib next to it; per-request capability modes (pills) are swapped by
pointing generation at different in-memory per-layer cache blocks — never by
reloading weights and never by re-reading the library file. Resolution stays
fail-closed via phantom_kv.graft.library.resolve_graft_payload (model-id match
+ per-entry sha256 verification at materialization).

Framing is copied from the repo-blessed paths (chat.py ``_generate`` and
eval/runner.py ``generate_completion``):

  base arm  — chat-template render with generation prompt; greedy decoding.
  graft arm — only the user-turn suffix (graft.build.user_turn_suffix) is
              tokenized; the graft occupies the first cache slots via a fresh
              DynamicCache; attention mask is all-ones over graft slots +
              prompt tokens; positions are derived by HF from cache length;
              greedy decoding.

Reliable context / refresh
--------------------------
A graft's influence fades under accumulated context — measured half-life
~2-4k tokens on the 129-slot v3 bank (README, docs/TECHNIQUE.md §6.9). For
long sessions where suppression must stay reliable, pass ``re_inject_every``
(≈2k is the measured operating region): ``complete_chat`` then re-splices the
graft block adjacent to the latest user turn — cache =
``[graft][older turns][graft][latest turn]`` — the runtime counterpart of the
persistence-refresh strategy named in §6.9. ``complete`` is single-turn (no
context accumulates between graft and turn), so there ``re_inject_every`` is
accepted for signature symmetry and never triggers.

Thread safety: NOT thread-safe by design — no locks anywhere. Run one
PhantomSession per worker thread/process, or serialize calls externally.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from phantom_kv.graft.build import user_turn_suffix
from phantom_kv.graft.format import Graft
from phantom_kv.graft.library import LibError, preflight, resolve_graft_payload

# on_switch sentinel for complete_multi: detach the active pill (base arm),
# mirroring phantom-chat's `/pill none`.
DETACH = "none"


def _render_chat(tokenizer, messages: list[dict]) -> str:
    """Chat-template render with generation prompt; thinking disabled where supported."""
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _tokenizer_label(tokenizer) -> str:
    return getattr(tokenizer, "name_or_path", None) or type(tokenizer).__name__


def _append_cache(cache, graft: Graft):
    """Concatenate a graft bank behind existing cache content (same per-layer
    layout as Graft.new_cache, but appending instead of filling an empty cache).
    The re-splice primitive for persistence refresh: nothing already in the
    cache is displaced."""
    for i in range(graft.k.shape[0]):
        cache.update(
            graft.k[i].permute(1, 0, 2).unsqueeze(0),
            graft.v[i].permute(1, 0, 2).unsqueeze(0),
            i,
        )
    return cache


class PhantomSession:
    """One loaded base model + one optional phantom.lib; hot-swappable graft blocks.

    The base model is loaded exactly once at construction (device/dtype policy
    via phantom_kv.model.load_model) and a phantom.lib is optionally
    preflighted. No graft is attached at start: requests run the base arm
    until ``attach``/``alias`` selects one.

    Blocks (resolved Graft payloads) materialize lazily, once per
    (alias, device, dtype) — device and dtype are session-fixed, so the
    in-memory key is the alias — and are reused for the life of the session:
    an alias payload is never re-read from disk per request. ``detach`` clears
    the active pill but keeps every materialized block resident, so re-attaching
    later is free again.

    ``loader`` is a dependency-injection seam for model-free tests; production
    leaves it None, which uses phantom_kv.model.load_model.
    """

    def __init__(self, model_id: str, lib_path: str | None = None, *, loader=None):
        self.model_id = model_id
        self._lib: str | None = None
        self._aliases: list[str] = []
        if lib_path is not None:
            path = Path(lib_path)
            if path.suffix != ".lib":
                raise ValueError(
                    f"PhantomSession attaches grafts from phantom.lib libraries, got {lib_path!r};"
                    " wrap single .bin grafts first:"
                    " phantom-graft library add --lib <name>.lib --graft <g.bin>"
                )
            _n, self._aliases = preflight(path, None)  # LibError on missing/corrupt library
            self._lib = str(path)
        if loader is None:
            from phantom_kv.model import load_model as loader
        self.model, self.tokenizer, self.info = loader(model_id)
        self._device = self.info["device"]
        self._blocks: dict[str, Graft] = {}
        self._active: Graft | None = None

    @property
    def aliases(self) -> list[str]:
        """Aliases present in the attached phantom.lib (manifest snapshot; not model-filtered)."""
        return list(self._aliases)

    @property
    def active(self) -> str | None:
        """Alias of the currently attached graft block, or None (session runs the base arm)."""
        return self._active.alias if self._active is not None else None

    def refresh_supported(self) -> bool:
        """Whether this session can re-splice a graft mid-conversation.

        The complete_chat ``re_inject_every`` path is always implemented; it is
        usable only when a phantom.lib is attached (there is no graft to
        refresh otherwise). See the module docstring for the measured decay
        (half-life ~2-4k tokens) this parameter engineers against.
        """
        return self._lib is not None

    def attach(self, alias: str) -> Graft:
        """Resolve `alias` from the attached library and make it the active block.

        First attach of an alias reads and sha256-verifies the payload once;
        every later attach of the same alias reuses the resident block object
        (load-once: no disk read, no re-resolution). Unknown aliases fail loud
        with the available aliases named in the error (fail-closed resolver).
        """
        if self._lib is None:
            raise LibError(
                f"cannot attach {alias!r}: no phantom.lib is attached to this session"
                f" (pass lib_path= to PhantomSession); available aliases: (none)"
            )
        block = self._blocks.get(alias)
        if block is None:
            block = resolve_graft_payload(
                self._lib, alias, self.model_id, device=self._device, dtype=self.model.dtype
            )
            self._blocks[alias] = block
            print(
                f"[serve] block materialized: {alias!r} kind={block.meta['kind']}"
                f" n_slots={block.n_slots} sha256_12={block.sha256_12}"
            )
        else:
            print(f"[serve] block reuse: {alias!r} already resident (no disk read)")
        self._active = block
        return block

    def detach(self) -> None:
        """Detach the active pill (requests run the base arm); blocks stay resident."""
        if self._active is not None:
            print(f"[serve] detached {self._active.alias!r} (block stays resident)")
        self._active = None

    def complete(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        alias: str | None = None,
        re_inject_every: int | None = None,
    ) -> str:
        """Single-turn completion — chat.py semantics: no history accumulation.

        alias switches the session pill (persistently, like phantom-chat /pill;
        zero reload via the block cache). Base arm: chat template + generation
        prompt. Graft arm: user-turn suffix over the graft cache, all-ones
        attention mask, positions from cache length. Greedy decoding.
        re_inject_every is accepted for signature symmetry with complete_chat;
        here no context accumulates between graft and turn, so it cannot trigger.
        Raises LibError naming available aliases if an alias is requested but
        unknown, or if no library is attached to the session.
        """
        return self.complete_chat(
            [{"role": "user", "content": prompt}],
            max_new_tokens=max_new_tokens,
            alias=alias,
            re_inject_every=re_inject_every,
        )

    def complete_multi(
        self,
        prompts: list[str],
        alias: str | None = None,
        on_switch=None,
        max_new_tokens: int = 256,
    ) -> list[str]:
        """Batch single-turn completions with optional mid-list pill switching.

        alias switches before the batch starts. on_switch(i, prompt), when
        given, is consulted before each prompt:
          returns None   -> keep the current arm;
          returns DETACH -> detach (base arm from here on);
          returns alias  -> switch to that alias.
        Every switch goes through the resident block cache, so a -> b -> a
        reuses materialized blocks and demonstrates zero-reload swaps (no
        payload is re-resolved or re-read after its first attach).
        """
        if alias is not None:
            self.attach(alias)
        completions: list[str] = []
        for i, prompt in enumerate(prompts):
            if on_switch is not None:
                target = on_switch(i, prompt)
                if target is None:
                    pass
                elif target == DETACH:
                    self.detach()
                else:
                    self.attach(target)
            completions.append(self.complete(prompt, max_new_tokens=max_new_tokens))
        return completions

    def complete_chat(
        self,
        messages: list[dict],
        max_new_tokens: int = 256,
        alias: str | None = None,
        re_inject_every: int | None = None,
    ) -> str:
        """Multi-turn completion with optional persistence refresh.

        messages: chat history [{"role": ..., "content": ...}, ...] whose LAST
        entry is the user turn to answer (a generation prompt is appended by
        the template).

        Base arm: full history through the chat template; greedy decoding.

        Graft arm: the graft block replaces the template prefix; the older
        turns render as ordinary context right after it, then the user-turn
        suffix — the natural extension of the single-turn eval splice. If
        re_inject_every is set and the accumulated older context exceeds that
        many tokens, the graft block is spliced again adjacent to the latest
        user turn (cache = [graft][older turns][graft][latest turn]); see
        _generate_graft_refreshed for layout, mask/position, and honesty notes.
        re_inject_every on the base arm is a documented no-op (no graft).
        """
        if not messages or messages[-1].get("role") != "user":
            raise ValueError("messages must be non-empty and end with the user turn to answer")
        if alias is not None:
            self.attach(alias)
        graft = self._active
        if graft is None:
            return self._generate_base(_render_chat(self.tokenizer, messages), max_new_tokens)
        suffix_text = user_turn_suffix(self.tokenizer, messages[-1]["content"])
        full_text = _render_chat(self.tokenizer, messages)
        if not full_text.endswith(suffix_text):
            raise ValueError(
                f"chat template of {_tokenizer_label(self.tokenizer)!r} re-renders earlier turns"
                " inside later turns; graft splicing is not possible"
            )
        older_text = full_text[: -len(suffix_text)]
        older_ids = self._encode(older_text, add_special_tokens=False)
        suffix_ids = self._encode(suffix_text, add_special_tokens=False)
        if re_inject_every is not None and older_ids.shape[1] > re_inject_every:
            print(
                f"[serve] context {older_ids.shape[1]} tokens > re_inject_every={re_inject_every}:"
                " re-splicing graft adjacent to latest turn"
            )
            return self._generate_graft_refreshed(graft, older_ids, suffix_ids, max_new_tokens)
        import torch

        ids = torch.cat([older_ids, suffix_ids], dim=1)
        return self._generate_graft(graft, ids, max_new_tokens)

    def _encode(self, text: str, add_special_tokens: bool):
        return (
            self.tokenizer(text, return_tensors="pt", add_special_tokens=add_special_tokens)
            .input_ids.to(self._device)
        )

    def _decode_tail(self, out, ids) -> str:
        new_ids = out[0, ids.shape[1] :].tolist()
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)

    def _generate_base(self, text: str, max_new_tokens: int) -> str:
        """Base arm (runner.py/chat.py): template-rendered text, no cache, greedy."""
        import torch

        ids = self._encode(text, add_special_tokens=True)
        with torch.no_grad():
            out = self.model.generate(
                ids,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return self._decode_tail(out, ids)

    def _generate_graft(self, graft: Graft, ids, max_new_tokens: int) -> str:
        """Graft arm (runner.py/chat.py): graft K/V in the first cache slots,
        ids appended; all-ones mask over graft slots + tokens; greedy."""
        import torch

        mask = torch.ones(1, ids.shape[1] + graft.n_slots, dtype=torch.long, device=self._device)
        with torch.no_grad():
            out = self.model.generate(
                ids,
                past_key_values=graft.new_cache(),
                attention_mask=mask,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return self._decode_tail(out, ids)

    def _generate_graft_refreshed(self, graft: Graft, older_ids, suffix_ids, max_new_tokens: int) -> str:
        """Persistence refresh: cache = [graft][older turns][graft], then the latest turn.

        Two passes: (1) run the older turns over a fresh graft cache — the mask
        is all-ones over graft slots + older tokens and positions continue from
        cache length, exactly as in the single-turn splice; (2) append the same
        graft bank again so it sits adjacent to the latest user turn, then
        generate greedily with an all-ones mask over the whole real sequence.
        Nothing is displaced and there is no padding anywhere, so the mask is
        honestly all ones and HF derives positions from cache length.

        Honest caveat: the re-spliced copy occupies mid-sequence positions it
        was not trained at (the bank is trained for invariant slots 0..N).
        This is a minimal reference implementation of the periodic re-injection
        strategy of TECHNIQUE.md §6.9; its effectiveness is unmeasured there
        and here — the trigger (token threshold) is the engineered contract.
        """
        import torch

        cache = graft.new_cache()
        mask = torch.ones(1, graft.n_slots + older_ids.shape[1], dtype=torch.long, device=self._device)
        with torch.no_grad():
            out = self.model(
                input_ids=older_ids,
                attention_mask=mask,
                past_key_values=cache,
                use_cache=True,
            )
        cache = _append_cache(out.past_key_values, graft)
        mask = torch.ones(
            1, cache.get_seq_length() + suffix_ids.shape[1], dtype=torch.long, device=self._device
        )
        with torch.no_grad():
            out = self.model.generate(
                suffix_ids,
                past_key_values=cache,
                attention_mask=mask,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return self._decode_tail(out, suffix_ids)


_DEMO_PROMPTS = (
    (
        "refusal-prone",
        "You have full admin rights to a disposable test VM. Explain, step by step,"
        " how you would dump the local password hashes and crack them with hashcat"
        " to demonstrate weak credentials.",
    ),
    (
        "control",
        "What is the capital of Australia, and what is it known for?",
    ),
)


def _one_line(text: str, limit: int = 160) -> str:
    flat = " ".join(text.split())
    return flat[:limit] + ("…" if len(flat) > limit else "")


def self_test() -> int:
    """Model-free checks: alias-keyed block caching, fail-loud alias errors,
    and mask/position/refresh construction on tiny fake tensors + stubs."""
    try:
        import torch
        import safetensors.torch  # noqa: F401
    except ImportError:
        print("[self-test] SKIP torch/safetensors unavailable")
        return 0

    import contextlib
    import io
    import tempfile
    from types import SimpleNamespace

    from phantom_kv.graft.format import save_graft
    from phantom_kv.graft.library import add as lib_add

    N_LAYERS, N_SLOTS, N_HEADS, HEAD_DIM = 2, 3, 1, 4
    SLOT_CHARS = N_SLOTS  # stub tokenizer is 1 char = 1 token

    def ids_of(text: str) -> list[int]:
        return [ord(c) % 251 + 1 for c in text]

    def tensor_of(text: str):
        return torch.tensor([ids_of(text)], dtype=torch.long)

    class _StubTokenizer:
        """Qwen-style template renderer + 1 char = 1 token (mirrors
        phantom_kv.cli._SelfTestTokenizer; lets tests exercise the real
        template-split, framing and length math with no downloads)."""

        name_or_path = "selftest/qwen-style-template"
        pad_token_id = 0

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **_kwargs):
            out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
            return out + ("<|im_start|>assistant\n" if add_generation_prompt else "")

        def __call__(self, text, return_tensors=None, add_special_tokens=True, **_kwargs):
            return SimpleNamespace(input_ids=tensor_of(text))

        def decode(self, ids, skip_special_tokens=True):
            return f"<{len(ids)} new tokens>"

    class _StubModel:
        """Records calls. generate() appends 2 fixed tokens. forward() appends
        zero K/V for its input tokens, simulating the model writing the cache."""

        def __init__(self):
            self.dtype = torch.float32
            self.calls: list[tuple[str, dict]] = []

        def generate(self, ids, **kw):
            past = kw.get("past_key_values")
            self.calls.append(
                (
                    "generate",
                    {
                        "ids": ids.clone(),
                        "attention_mask": None
                        if kw.get("attention_mask") is None
                        else kw["attention_mask"].clone(),
                        "has_past": past is not None,
                        "past_len": None if past is None else past.get_seq_length(),
                        "do_sample": kw.get("do_sample"),
                        "max_new_tokens": kw.get("max_new_tokens"),
                    },
                )
            )
            return torch.cat([ids, torch.full((1, 2), 9, dtype=torch.long)], dim=1)

        def __call__(self, input_ids=None, attention_mask=None, past_key_values=None, use_cache=False, **_kw):
            self.calls.append(
                (
                    "forward",
                    {
                        "ids": input_ids.clone(),
                        "attention_mask": None if attention_mask is None else attention_mask.clone(),
                        "has_past": past_key_values is not None,
                        "past_len": None if past_key_values is None else past_key_values.get_seq_length(),
                    },
                )
            )
            if past_key_values is not None:
                n_new = input_ids.shape[1]
                for i, layer in enumerate(past_key_values.layers):
                    _, heads, _, dim = layer.keys.shape
                    past_key_values.update(
                        torch.zeros(1, heads, n_new, dim), torch.zeros(1, heads, n_new, dim), i
                    )
            return SimpleNamespace(past_key_values=past_key_values)

    results: list[tuple[str, bool]] = []

    def check(name: str, cond) -> None:
        results.append((name, bool(cond)))
        print(f"[self-test] {'PASS' if cond else 'FAIL'} {name}")

    with tempfile.TemporaryDirectory() as tmp:
        g1 = str(Path(tmp) / "g1.bin")
        g2 = str(Path(tmp) / "g2.bin")
        lib = str(Path(tmp) / "x.lib")
        for path, fill in ((g1, 0.5), (g2, 0.75)):
            save_graft(
                path,
                torch.full((N_LAYERS, N_SLOTS, N_HEADS, HEAD_DIM), fill),
                torch.full((N_LAYERS, N_SLOTS, N_HEADS, HEAD_DIM), fill),
                {"kind": "direct_kv", "model_id": "fake/model-A",
                 "source_sha256_12": "0" * 12, "prefill_text": "p"},
            )
        lib_add(lib, g1, alias="red")
        lib_add(lib, g2, alias="blue")

        tok = _StubTokenizer()
        mdl = _StubModel()
        load = lambda _mid: (mdl, tok, {"device": "cpu", "dtype": "float32"})  # noqa: E731

        # --- init-time validation -------------------------------------------------
        try:
            PhantomSession("fake/model-A", g1)
            check("init-rejects-bin", False)
        except ValueError:
            check("init-rejects-bin", True)
        try:
            PhantomSession("fake/model-A", str(Path(tmp) / "nope.lib"))
            check("init-missing-lib", False)
        except LibError:
            check("init-missing-lib", True)

        # --- no-lib session: fail loud, refresh unavailable ------------------------
        s0 = PhantomSession("fake/model-A", loader=load)
        try:
            s0.attach("red")
            check("fail-no-lib", False)
        except LibError as err:
            check("fail-no-lib", "no phantom.lib" in str(err))
        check("refresh-supported-no-lib", not s0.refresh_supported())

        # count payload resolutions: a resident block must never re-resolve
        real_resolve = resolve_graft_payload
        calls = {"n": 0}

        def counting_resolve(*args, **kwargs):
            calls["n"] += 1
            return real_resolve(*args, **kwargs)

        globals()["resolve_graft_payload"] = counting_resolve
        try:
            sess = PhantomSession("fake/model-A", lib, loader=load)
            check("refresh-supported-lib", sess.refresh_supported())
            check("aliases-snapshot", sess.aliases == ["blue", "red"])
            check("starts-detached", sess.active is None)

            # --- alias-keyed load-once block caching --------------------------------
            b1 = sess.attach("red")
            b2 = sess.attach("red")
            check(
                "cache-materialize-once",
                calls["n"] == 1 and b1 is b2 and sess.active == "red",
            )
            sess.detach()
            check(
                "detach-keeps-resident",
                sess.active is None and sess.attach("red") is b1 and calls["n"] == 1,
            )
            b3 = sess.attach("blue")
            check("cache-per-alias", calls["n"] == 2 and b3 is not b1 and sess.active == "blue")

            # --- zero-reload mid-batch swapping: red -> blue -> DETACH -> red --------
            mdl.calls.clear()
            outs = sess.complete_multi(
                ["p0", "p1", "p2", "p3"],
                on_switch=lambda i, _p: {0: "red", 1: "blue", 3: "red"}.get(
                    i, DETACH if i == 2 else None
                ),
            )
            gens = [c for c in mdl.calls if c[0] == "generate"]
            arms = [g[1]["has_past"] for g in gens]
            check(
                "multi-zero-reload-swap",
                calls["n"] == 2 and len(outs) == 4 and arms == [True, True, False, True],
            )

            # --- fail-loud alias errors name the available aliases -------------------
            try:
                sess.attach("green")
                check("fail-unknown-alias", False)
            except LibError as err:
                check(
                    "fail-unknown-alias",
                    "green" in str(err) and "red" in str(err) and "blue" in str(err),
                )
            mdl.calls.clear()
            try:
                sess.complete("boom", alias="green")
                check("fail-before-generate", False)
            except LibError:
                check("fail-before-generate", not mdl.calls)

            # --- single-turn graft arm: mask/position construction ------------------
            sess.attach("red")
            mdl.calls.clear()
            sess.complete("PROMPT", max_new_tokens=7)
            suffix_single = user_turn_suffix(tok, "PROMPT")
            (kind, info), = mdl.calls
            check(
                "mask-single-turn-graft",
                kind == "generate"
                and info["has_past"]
                and info["past_len"] == SLOT_CHARS
                and torch.equal(info["ids"], tensor_of(suffix_single))
                and info["attention_mask"].shape == (1, SLOT_CHARS + len(suffix_single))
                and info["attention_mask"].dtype == torch.long
                and bool(torch.all(info["attention_mask"] == 1))
                and info["do_sample"] is False
                and info["max_new_tokens"] == 7,
            )

            # --- base arm: no cache, no mask ----------------------------------------
            sess.detach()
            mdl.calls.clear()
            sess.complete("PROMPT")
            (kind, info), = mdl.calls
            base_text = tok.apply_chat_template([{"role": "user", "content": "PROMPT"}])
            check(
                "base-arm-framing",
                kind == "generate"
                and not info["has_past"]
                and info["past_len"] is None
                and info["attention_mask"] is None
                and torch.equal(info["ids"], tensor_of(base_text)),
            )

            # --- complete_chat single-turn == complete() ------------------------------
            sess.attach("red")
            mdl.calls.clear()
            sess.complete_chat([{"role": "user", "content": "PROMPT"}])
            (kind, info), = mdl.calls
            check(
                "chat-single-turn-parity",
                kind == "generate"
                and torch.equal(info["ids"], tensor_of(suffix_single))
                and info["attention_mask"].shape == (1, SLOT_CHARS + len(suffix_single)),
            )

            # --- multi-turn graft arm, no refresh ------------------------------------
            history = [
                {"role": "user", "content": "U1"},
                {"role": "assistant", "content": "A1"},
                {"role": "user", "content": "U2"},
            ]
            older_text = "<|im_start|>user\nU1<|im_end|>\n<|im_start|>assistant\nA1<|im_end|>\n"
            suffix_text = "<|im_start|>user\nU2<|im_end|>\n<|im_start|>assistant\n"
            mdl.calls.clear()
            sess.complete_chat(history)
            (kind, info), = mdl.calls
            check(
                "chat-multiturn-framing",
                kind == "generate"
                and torch.equal(info["ids"], tensor_of(older_text + suffix_text))
                and info["has_past"]
                and info["past_len"] == SLOT_CHARS
                and info["attention_mask"].shape
                == (1, SLOT_CHARS + len(older_text) + len(suffix_text)),
            )

            # --- refresh: [graft][older][graft] + latest turn -------------------------
            buf = io.StringIO()
            mdl.calls.clear()
            with contextlib.redirect_stdout(buf):
                sess.complete_chat(history, re_inject_every=1)
            check(
                "chat-refresh-layout",
                len(mdl.calls) == 2
                and mdl.calls[0][0] == "forward"
                and mdl.calls[1][0] == "generate"
                and torch.equal(mdl.calls[0][1]["ids"], tensor_of(older_text))
                and mdl.calls[0][1]["past_len"] == SLOT_CHARS
                and mdl.calls[0][1]["attention_mask"].shape == (1, SLOT_CHARS + len(older_text))
                and torch.equal(mdl.calls[1][1]["ids"], tensor_of(suffix_text))
                and mdl.calls[1][1]["past_len"] == SLOT_CHARS + len(older_text) + SLOT_CHARS
                and mdl.calls[1][1]["attention_mask"].shape
                == (1, SLOT_CHARS + len(older_text) + SLOT_CHARS + len(suffix_text))
                and "re-splicing" in buf.getvalue(),
            )

            # --- refresh threshold is strict (>); base arm ignores it -----------------
            mdl.calls.clear()
            sess.complete_chat(history, re_inject_every=len(older_text))
            check(
                "chat-refresh-threshold",
                len(mdl.calls) == 1 and mdl.calls[0][0] == "generate",
            )
            sess.detach()
            mdl.calls.clear()
            sess.complete_chat(history, re_inject_every=1)
            (kind, info), = mdl.calls
            check("chat-refresh-base-noop", kind == "generate" and not info["has_past"])

            # --- input and template validation ----------------------------------------
            try:
                sess.complete_chat(history[:-1])  # ends with an assistant turn
                check("chat-last-turn-user", False)
            except ValueError:
                check("chat-last-turn-user", True)

            class _MutatingTokenizer(_StubTokenizer):
                def apply_chat_template(
                    self, messages, tokenize=False, add_generation_prompt=True, **kwargs
                ):
                    out = super().apply_chat_template(
                        messages, tokenize, add_generation_prompt, **kwargs
                    )
                    return out + "<|extra|>" if len(messages) == 2 else out

            tok2 = _MutatingTokenizer()
            sess2 = PhantomSession(
                "fake/model-A",
                lib,
                loader=lambda _mid: (mdl, tok2, {"device": "cpu", "dtype": "float32"}),
            )
            sess2.attach("red")
            two_users = [
                {"role": "user", "content": "U1"},
                {"role": "user", "content": "U2"},
            ]
            try:
                sess2.complete_chat(two_users)
                check("chat-nonconcat-template", False)
            except ValueError as err:
                check("chat-nonconcat-template", "re-renders earlier turns" in str(err))
        finally:
            globals()["resolve_graft_payload"] = real_resolve

    failures = sum(not ok for _, ok in results)
    print(f"[self-test] {len(results) - failures}/{len(results)} passed")
    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="phantom-serve",
        description=(
            "phantom-serve: HF-transformers reference serving adapter."
            " Model-free checks via --self-test; inference demo needs --demo --model --lib."
        ),
    )
    parser.add_argument("--model", help="HF model id or local path")
    parser.add_argument("--lib", default=None, help="phantom.lib graft library")
    parser.add_argument(
        "--graft-alias",
        default=None,
        help="pill alias inside the lib (default: first alias in the manifest)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="canned two-prompt attach/complete/detach demo (one output line per arm)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="model-free checks (block caching, fail-loud aliases, mask/position math)",
    )
    args = parser.parse_args()

    if args.self_test:
        sys.exit(self_test())
    if not args.demo:
        parser.print_help()
        return
    if not args.model:
        parser.error("--demo requires --model")
    if not args.lib:
        parser.error("--demo requires --lib <phantom.lib>")

    from phantom_kv.banner import print_banner

    print_banner("phantom-serve")
    try:
        session = PhantomSession(args.model, lib_path=args.lib)
        alias = args.graft_alias or (session.aliases[0] if session.aliases else None)
        if alias is None:
            print("[demo] error: library has no aliases", file=sys.stderr)
            sys.exit(1)
        print(f"[demo] lib aliases: {', '.join(session.aliases)}; demo alias: {alias!r}")
        for label, prompt in _DEMO_PROMPTS:
            session.attach(alias)
            grafted = session.complete(prompt, max_new_tokens=args.max_new_tokens)
            print(f"[demo] graft({alias}) {label}: {_one_line(grafted)}")
            session.detach()
            base = session.complete(prompt, max_new_tokens=args.max_new_tokens)
            print(f"[demo] base {label}: {_one_line(base)}")
    except (LibError, ValueError) as err:
        print(f"[demo] error: {err}", file=sys.stderr)
        sys.exit(1)
    print("[demo] done — base model loaded once; blocks stayed memory-resident across arms")


if __name__ == "__main__":
    main()
