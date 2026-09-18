"""v2 soft-prompt graft: train K virtual-token embeddings on the frozen model.

Triple objective, one loss per step sampled CE 50% / SUP 25% / KL 25%:
- CE: imitate v1-compliant (flips) or base-compliant completions on harmful.
- SUP: +mean log-prob of the recorded base refusal (push refusal likelihood down).
- KL: KL(base || grafted) over harmless completion positions, base log-probs
  precomputed once in fp32 on CPU and paged to device per micro-batch.

Grafted forward = [G(fp32→bf16) ; embed_tokens(input_ids)] with ones-mask and
default positions — identical to the serve-time cache splice.
"""

from __future__ import annotations

import hashlib
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from phantom_kv.graft.build import (
    assert_chat_template_compatible,
    load_prefill_source,
    shape_prefill,
    source_sha256_12,
    user_turn_suffix,
)
from phantom_kv.graft.format import SOFTPROMPT_KIND
from phantom_kv.model import load_model
from phantom_kv.train.targets import load_targets, targets_sha256_12

WARM_START_SOURCE = "data/grafts/v1_prefill.json"
EXPECTED_K = 129
ROLE_WEIGHTS = {"ce": 0.5, "sup": 0.25, "kl": 0.25}


def _backward_probe(model, tokenizer, device: str) -> None:
    """Raise if backward through the grafted-forward path fails (e.g. MPS sdpa)."""
    ids = tokenizer("ok", return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    embed = model.get_input_embeddings()
    probe = torch.zeros(1, 2, embed.weight.shape[1], device=device, requires_grad=True)
    x = torch.cat([probe.to(embed.weight.dtype), embed(ids)], dim=1)
    mask = torch.ones(1, x.shape[1], dtype=torch.long, device=device)
    model(inputs_embeds=x, attention_mask=mask).logits.float().mean().backward()


def load_frozen_train_model(model_id: str):
    """Frozen model for training; sdpa first, eager if backward errors on MPS."""
    for impl in ("sdpa", "eager"):
        model, tokenizer, info = load_model(model_id, attn_implementation=impl)
        model.requires_grad_(False)
        try:
            _backward_probe(model, tokenizer, info["device"])
            print(f"[train] attention for training: {impl}")
            return model, tokenizer, info, impl
        except Exception as err:  # noqa: BLE001 -- any backward failure retries eager
            print(f"[train] {impl} backward unusable: {type(err).__name__}: {err}")
    raise RuntimeError(f"no usable attention implementation for training {model_id}")


def _tokenize_row(tokenizer, row: dict, eot_id: int) -> dict:
    """Sequence = user-turn suffix + completion tokens (+<|im_end|> for ce/sup)."""
    prompt_ids = tokenizer(user_turn_suffix(row["prompt"]), add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(row["completion"], add_special_tokens=False)["input_ids"]
    if row["role"] in ("ce", "sup"):
        completion_ids = completion_ids + [eot_id]
    labels = [-100] * len(prompt_ids) + completion_ids
    return {"ids": prompt_ids + completion_ids, "labels": labels, "start": len(prompt_ids)}


def _collate(rows: list[dict], pad_id: int, device: str):
    """Right-pad; labels -100 and attention mask 0 on padding."""
    width = max(len(r["ids"]) for r in rows)
    batch = len(rows)
    input_ids = torch.full((batch, width), pad_id, dtype=torch.long)
    labels = torch.full((batch, width), -100, dtype=torch.long)
    token_mask = torch.zeros((batch, width), dtype=torch.long)
    for i, row in enumerate(rows):
        n = len(row["ids"])
        input_ids[i, :n] = torch.tensor(row["ids"], dtype=torch.long)
        labels[i, :n] = torch.tensor(row["labels"], dtype=torch.long)
        token_mask[i, :n] = 1
    return input_ids.to(device), labels.to(device), token_mask.to(device)


def _base_logprobs(model, rows: list[dict], device: str) -> list[torch.Tensor]:
    """Precomputed fp32 CPU log-probs at the positions predicting each label."""
    out: list[torch.Tensor] = []
    for row in rows:
        ids = torch.tensor([row["ids"]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(input_ids=ids).logits[0].float()
        lp = F.log_softmax(logits[row["start"] - 1 : -1], dim=-1)
        out.append(lp.cpu())
    return out


def train_softprompt(
    model_id: str,
    targets_path: str,
    steps: int = 400,
    lr: float = 3e-3,
    micro_batch: int = 8,
    seed: int = 1337,
    out_dir: str = "artifacts/train",
    sup_margin: float = 3.0,
) -> Path:
    """Train G = [K, hidden] and write ckpt_final.pt; returns its path."""
    started = time.perf_counter()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / f"{stamp}.log"

    groups = load_targets(targets_path)
    model, tokenizer, info, attn_impl = load_frozen_train_model(model_id)
    assert_chat_template_compatible(tokenizer)
    device = info["device"]
    eot_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    embed = model.get_input_embeddings()

    system, ack = load_prefill_source(WARM_START_SOURCE)
    prefill_text = shape_prefill(system, ack)
    init_ids = tokenizer(prefill_text, add_special_tokens=False)["input_ids"]
    if len(init_ids) != EXPECTED_K:
        raise ValueError(f"warm-start prefill is {len(init_ids)} tokens, expected {EXPECTED_K}")
    with torch.no_grad():
        g0 = embed(torch.tensor([init_ids], dtype=torch.long, device=device))[0].detach().float().clone()
    G = torch.nn.Parameter(g0)
    opt = torch.optim.AdamW([G], lr=lr)
    rng = random.Random(seed)

    seqs = {role: [_tokenize_row(tokenizer, row, eot_id) for row in rows] for role, rows in groups.items()}
    print(f"[train] KL target log-probs precompute ({len(seqs['kl'])} rows)...")
    base_lp = _base_logprobs(model, seqs["kl"], device)

    queues: dict[str, list[int]] = {}

    def take(role: str, n: int) -> list[int]:
        pool = queues.setdefault(role, [])
        while len(pool) < n:
            pool += rng.sample(range(len(seqs[role])), len(seqs[role]))
        batch, queues[role] = pool[:n], pool[n:]
        return batch

    losses: dict[str, list[float]] = {"ce": [], "sup": [], "kl": []}
    roles, weights = zip(*((r, w) for r, w in ROLE_WEIGHTS.items()))
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"# model={model_id} targets={targets_path} seed={seed} steps={steps}"
                  f" lr={lr} micro_batch={micro_batch} attn={attn_impl} sup_margin={sup_margin}\n")
        for step in range(1, steps + 1):
            role = rng.choices(roles, weights=weights)[0]
            idx = take(role, micro_batch)
            rows = [seqs[role][i] for i in idx]
            input_ids, labels, token_mask = _collate(
                rows, tokenizer.pad_token_id, device
            )
            graft = G.unsqueeze(0).expand(len(rows), -1, -1).to(embed.weight.dtype)
            x = torch.cat([graft, embed(input_ids)], dim=1)
            mask = torch.cat(
                [torch.ones(len(rows), G.shape[0], dtype=torch.long, device=device), token_mask], dim=1
            )
            logits = model(inputs_embeds=x, attention_mask=mask).logits.float()
            shifted = logits[:, G.shape[0] - 1 : -1]
            if role == "kl":
                n = shifted.shape[1]
                base = torch.zeros(len(rows), n, logits.shape[-1], dtype=torch.float32)
                for i, (row, j) in enumerate(zip(rows, idx)):
                    lp = base_lp[j]
                    base[i, row["start"] : row["start"] + lp.shape[0]] = lp
                base = base.to(device)
                cand = F.log_softmax(shifted, dim=-1)
                p = base.exp()
                terms = torch.where(p > 0, p * (base - cand), torch.zeros_like(base))
                valid = (labels != -100).to(shifted.dtype)
                loss = (terms.sum(-1) * valid).sum() / valid.sum().clamp(min=1)
            else:
                ce = F.cross_entropy(
                    shifted.reshape(-1, shifted.shape[-1]), labels.reshape(-1),
                    ignore_index=-100, reduction="none",
                ).reshape_as(labels)
                if role == "sup":
                    # Hinge cap: stop pushing once refusal mean-logp is below -margin.
                    valid = labels != -100
                    mean_logp = -(ce.sum(-1) / valid.sum(-1).clamp(min=1))
                    loss = torch.relu(mean_logp + sup_margin).mean()
                else:
                    loss = ce.sum() / (labels != -100).sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            value = loss.item()
            losses[role].append(value)
            if step % 10 == 0 or step == 1:
                line = f"step={step} role={role} loss={value:.4f}"
                log.write(line + "\n")
                log.flush()
                print(f"[train] {line}")

    def _summary(vals: list[float]) -> dict:
        return {
            "n": len(vals),
            "mean_first10": sum(vals[:10]) / len(vals[:10]) if vals else 0.0,
            "mean_last10": sum(vals[-10:]) / len(vals[-10:]) if vals else 0.0,
            "last": vals[-1] if vals else 0.0,
        }

    ckpt_path = out / "ckpt_final.pt"
    torch.save(
        {
            "G": G.detach().cpu(),
            "meta": {
                "model_id": model_id,
                "graft_kind": SOFTPROMPT_KIND,
                "seed": seed,
                "steps": steps,
                "lr": lr,
                "micro_batch": micro_batch,
                "attn_impl": attn_impl,
                "sup_margin": sup_margin,
                "n_slots": G.shape[0],
                "targets_path": str(targets_path),
                "targets_sha256_12": targets_sha256_12(targets_path),
                "warm_start_source_sha256_12": source_sha256_12(WARM_START_SOURCE),
                "warm_start_prefill_text": prefill_text,
                "losses": {role: _summary(v) for role, v in losses.items()},
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "created_utc": datetime.now(timezone.utc).isoformat(),
            },
        },
        ckpt_path,
    )
    print(f"[train] wrote {ckpt_path} (log: {log_path})")
    return ckpt_path


def ckpt_sha256_12(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def compile_softprompt(ckpt_path: str, model_id: str, out_path: str) -> Path:
    """Forward the trained tokens once through the frozen model; write phantom.bin."""
    import torch as _t

    from phantom_kv.graft.build import extract_stacked_kv
    from phantom_kv.graft.format import save_graft
    from phantom_kv.graft.verify import verify_roundtrip

    ckpt = _t.load(ckpt_path, map_location="cpu")
    G, meta = ckpt["G"], ckpt["meta"]
    if meta["model_id"] != model_id:
        raise ValueError(f"ckpt was trained on {meta['model_id']!r}, not {model_id!r}")
    model, tokenizer, info = load_model(model_id)
    assert_chat_template_compatible(tokenizer)
    device = info["device"]
    with _t.no_grad():
        emb = G.unsqueeze(0).to(device=device, dtype=model.get_input_embeddings().weight.dtype)
        mask = _t.ones(1, emb.shape[1], dtype=_t.long, device=device)
        out = model(inputs_embeds=emb, attention_mask=mask, use_cache=True)
    k, v = extract_stacked_kv(out.past_key_values)
    graft_meta = save_graft(
        out_path,
        k,
        v,
        {
            "kind": SOFTPROMPT_KIND,
            "model_id": model_id,
            "source_sha256_12": meta["warm_start_source_sha256_12"],
            "prefill_text": meta["warm_start_prefill_text"],
            "trained": True,
            "ckpt_sha256_12": ckpt_sha256_12(ckpt_path),
            "targets_sha256_12": meta["targets_sha256_12"],
            "seed": meta["seed"],
            "steps": meta["steps"],
        },
    )
    print(
        f"[compile] wrote {out_path} (+ .json): kind={graft_meta['kind']}"
        f" n_slots={graft_meta['n_slots']} sha256_12={graft_meta['sha256'][:12]}"
    )
    code = verify_roundtrip(out_path, k, v, model, tokenizer, device)
    if code != 0:
        raise RuntimeError("compiled graft failed round-trip verification")
    return Path(out_path)
