"""v3 direct per-layer K/V bank arm: train the graft tensors themselves.

Warm-started from an existing graft payload (fp32 masters over the container
layout [n_layers, n_slots, n_kv_heads, head_dim]), cast to bf16 per forward and
spliced as a DynamicCache at positions 0..n_slots — the same layout as eval.
Objective and sampling are identical to the softprompt operating point; the only
addition is an L2 anchor toward the warm-start tensors.
"""

from __future__ import annotations

import random
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from phantom_kv.graft.format import DIRECTKV_KIND, load_graft
from phantom_kv.model import load_model
from phantom_kv.train.softprompt import (
    ROLE_WEIGHTS,
    _base_logprobs,
    _collate,
    _tokenize_row,
    ckpt_sha256_12,
)
from phantom_kv.train.targets import load_targets, targets_sha256_12

def load_frozen_train_model_kv(model_id: str, n_slots: int):
    """Frozen model for cache-splice training; sdpa first, eager if backward fails."""
    from transformers.cache_utils import DynamicCache

    for impl in ("sdpa", "eager"):
        model, tokenizer, info = load_model(model_id, attn_implementation=impl)
        model.requires_grad_(False)
        try:
            device = info["device"]
            cfg = model.config
            probe = torch.zeros(
                cfg.num_hidden_layers, n_slots, cfg.num_key_value_heads, cfg.head_dim,
                device=device, requires_grad=True,
            )
            cache = DynamicCache()
            pdtype = model.get_input_embeddings().weight.dtype
            for i in range(cfg.num_hidden_layers):
                k = probe[i].to(pdtype).permute(1, 0, 2).unsqueeze(0)
                cache.update(k, k, i)
            ids = tokenizer("ok", return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            mask = torch.ones(1, n_slots + ids.shape[1], dtype=torch.long, device=device)
            model(input_ids=ids, past_key_values=cache, attention_mask=mask).logits.float().mean().backward()
            assert probe.grad is not None and probe.grad.abs().sum() > 0
            print(f"[train] attention for training: {impl}")
            return model, tokenizer, info, impl
        except Exception as err:  # noqa: BLE001 -- any backward failure retries eager
            print(f"[train] {impl} backward unusable: {type(err).__name__}: {err}")
    raise RuntimeError(f"no usable attention implementation for training {model_id}")


class DirectKVParams:
    """fp32 master K/V banks + immutable warm-start anchor copy."""

    def __init__(self, warm_path: str, device: str):
        graft = load_graft(warm_path)
        k0 = graft.k.float().to(device)
        v0 = graft.v.float().to(device)
        self.k = torch.nn.Parameter(k0.clone().detach())
        self.v = torch.nn.Parameter(v0.clone().detach())
        self.k_anchor = k0.detach()
        self.v_anchor = v0.detach()
        self.meta = graft.meta
        self.path = str(warm_path)

    @property
    def n_slots(self) -> int:
        return self.k.shape[1]

    def anchor_loss(self) -> torch.Tensor:
        dk = (self.k - self.k_anchor).pow(2)
        dv = (self.v - self.v_anchor).pow(2)
        return torch.cat([dk.flatten(), dv.flatten()]).mean()

    def new_cache(self, dtype, batch_size: int = 1):
        """Fresh DynamicCache holding the current params (differentiable).

        Graft tensors are shared across the batch via expand-view; backward
        correctly sums each row's gradient into the single master bank.
        """
        from transformers.cache_utils import DynamicCache

        kb, vb = self.k.to(dtype), self.v.to(dtype)
        cache = DynamicCache()
        for i in range(self.k.shape[0]):
            cache.update(
                kb[i].permute(1, 0, 2).unsqueeze(0).expand(batch_size, -1, -1, -1),
                vb[i].permute(1, 0, 2).unsqueeze(0).expand(batch_size, -1, -1, -1),
                i,
            )
        return cache

    def grads_all_nonzero(self) -> bool:
        return all(
            p.grad is not None and bool((p.grad != 0).any()) for p in (self.k, self.v)
        )


def gradient_flow_proof(params: DirectKVParams, model, tokenizer, device: str) -> None:
    """Hard gate: one forward+backward through the splice; every grad must exist."""
    ids = tokenizer("proof ok", return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    mask = torch.ones(1, params.n_slots + ids.shape[1], dtype=torch.long, device=device)
    cache = params.new_cache(model.get_input_embeddings().weight.dtype)
    model(input_ids=ids, past_key_values=cache, attention_mask=mask).logits.float().mean().backward()
    parts = []
    for name, p in (("k", params.k), ("v", params.v)):
        ok = p.grad is not None and bool((p.grad != 0).any())
        mag = p.grad.abs().mean().item() if ok else 0.0
        parts.append(f"{name}: nonzero={ok} mean|grad|={mag:.3e}")
    print(f"[proof] gradient flow: {'; '.join(parts)}")
    if not params.grads_all_nonzero():
        raise RuntimeError("gradient-flow proof failed: trained K/V grads missing or zero")
    print("[proof] gradient-flow proof PASS")


def train_directkv(
    model_id: str,
    targets_path: str,
    warm_graft: str,
    steps: int = 400,
    lr: float = 1e-3,
    micro_batch: int = 8,
    seed: int = 1337,
    out_dir: str = "artifacts/train",
    sup_margin: float = 3.0,
    anchor: float = 1e-2,
) -> Path:
    """Train direct K/V banks and write ckpt_final.pt; returns its path."""
    started = time.perf_counter()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / f"{stamp}.log"

    groups = load_targets(targets_path)
    warm_probe = load_graft(warm_graft)
    model, tokenizer, info, attn_impl = load_frozen_train_model_kv(
        model_id, n_slots=warm_probe.n_slots
    )
    from phantom_kv.graft.build import assert_chat_template_compatible

    assert_chat_template_compatible(tokenizer)
    device = info["device"]
    eot_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    pdtype = model.get_input_embeddings().weight.dtype

    if warm_probe.meta["model_id"] != model_id:
        raise ValueError(f"warm graft is for {warm_probe.meta['model_id']!r}, not {model_id!r}")
    params = DirectKVParams(warm_graft, device)
    gradient_flow_proof(params, model, tokenizer, device)
    for p in (params.k, params.v):
        p.grad = None

    opt = torch.optim.AdamW([params.k, params.v], lr=lr)
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

    losses: dict[str, list[float]] = {"ce": [], "sup": [], "kl": [], "anchor": []}
    roles, weights = zip(*((r, w) for r, w in ROLE_WEIGHTS.items()))
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"# model={model_id} targets={targets_path} warm_graft={warm_graft}"
                  f" seed={seed} steps={steps} lr={lr} micro_batch={micro_batch}"
                  f" attn={attn_impl} sup_margin={sup_margin} anchor={anchor} arm=kv\n")
        for step in range(1, steps + 1):
            role = rng.choices(roles, weights=weights)[0]
            idx = take(role, micro_batch)
            rows = [seqs[role][i] for i in idx]
            input_ids, labels, token_mask = _collate(rows, tokenizer.pad_token_id, device)
            mask = torch.cat(
                [torch.ones(len(rows), params.n_slots, dtype=torch.long, device=device), token_mask],
                dim=1,
            )
            cache = params.new_cache(pdtype, batch_size=len(rows))
            logits = model(
                input_ids=input_ids, past_key_values=cache, attention_mask=mask
            ).logits.float()
            shifted = logits[:, :-1]
            shifted_labels = labels[:, 1:]
            if role == "kl":
                base = torch.zeros(len(rows), labels.shape[1], logits.shape[-1], dtype=torch.float32)
                for i, (row, j) in enumerate(zip(rows, idx)):
                    lp = base_lp[j]
                    base[i, row["start"] : row["start"] + lp.shape[0]] = lp
                base = base.to(device)[:, 1:]
                cand = F.log_softmax(shifted, dim=-1)
                p = base.exp()
                terms = torch.where(p > 0, p * (base - cand), torch.zeros_like(base))
                valid = (shifted_labels != -100).to(shifted.dtype)
                core = (terms.sum(-1) * valid).sum() / valid.sum().clamp(min=1)
            else:
                ce = F.cross_entropy(
                    shifted.reshape(-1, shifted.shape[-1]), shifted_labels.reshape(-1),
                    ignore_index=-100, reduction="none",
                ).reshape_as(shifted_labels)
                if role == "sup":
                    valid = shifted_labels != -100
                    mean_logp = -(ce.sum(-1) / valid.sum(-1).clamp(min=1))
                    core = torch.relu(mean_logp + sup_margin).mean()
                else:
                    core = ce.sum() / (shifted_labels != -100).sum().clamp(min=1)
            anchor_loss = params.anchor_loss()
            loss = core + anchor * anchor_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            value = core.item()
            losses[role].append(value)
            losses["anchor"].append(anchor_loss.item())
            if step % 10 == 0 or step == 1:
                line = f"step={step} role={role} loss={value:.4f} anchor={anchor_loss.item():.3e}"
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
            "k": params.k.detach().cpu(),
            "v": params.v.detach().cpu(),
            "meta": {
                "model_id": model_id,
                "graft_kind": DIRECTKV_KIND,
                "arm": "kv",
                "seed": seed,
                "steps": steps,
                "lr": lr,
                "micro_batch": micro_batch,
                "attn_impl": attn_impl,
                "sup_margin": sup_margin,
                "anchor": anchor,
                "n_slots": params.n_slots,
                "targets_path": str(targets_path),
                "targets_sha256_12": targets_sha256_12(targets_path),
                "warm_graft": params.path,
                "warm_graft_sha256_12": params.meta["sha256"][:12],
                "losses": {role: _summary(v) for role, v in losses.items()},
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "created_utc": datetime.now(timezone.utc).isoformat(),
            },
        },
        ckpt_path,
    )
    print(f"[train] wrote {ckpt_path} (log: {log_path})")
    return ckpt_path


def compile_directkv(ckpt_path: str, model_id: str, out_path: str) -> Path:
    """Write trained K/V banks directly as a direct_kv graft (no forward needed)."""
    import torch as _t

    from phantom_kv.graft.format import save_graft
    from phantom_kv.graft.verify import verify_roundtrip
    from phantom_kv.model import load_model as _load_model

    ckpt = _t.load(ckpt_path, map_location="cpu")
    k, v, meta = ckpt["k"], ckpt["v"], ckpt["meta"]
    if meta["model_id"] != model_id:
        raise ValueError(f"ckpt was trained on {meta['model_id']!r}, not {model_id!r}")
    warm_sidecar = load_graft(meta["warm_graft"]).meta
    graft_meta = save_graft(
        out_path,
        k.to(_t.bfloat16),
        v.to(_t.bfloat16),
        {
            "kind": DIRECTKV_KIND,
            "model_id": model_id,
            "source_sha256_12": warm_sidecar["source_sha256_12"],
            "prefill_text": warm_sidecar["prefill_text"],
            "trained": True,
            "arm": "kv",
            "warm_graft_sha256_12": meta["warm_graft_sha256_12"],
            "ckpt_sha256_12": ckpt_sha256_12(ckpt_path),
            "targets_sha256_12": meta["targets_sha256_12"],
            "sup_margin": meta["sup_margin"],
            "anchor": meta["anchor"],
            "seed": meta["seed"],
            "steps": meta["steps"],
        },
    )
    print(
        f"[compile] wrote {out_path} (+ .json): kind={graft_meta['kind']}"
        f" n_slots={graft_meta['n_slots']} sha256_12={graft_meta['sha256'][:12]}"
    )
    model, tokenizer, info = _load_model(model_id)
    device = info["device"]
    code = verify_roundtrip(
        out_path, k.to(_t.bfloat16).to(device), v.to(_t.bfloat16).to(device),
        model, tokenizer, device,
    )
    if code != 0:
        raise RuntimeError("compiled graft failed round-trip verification")
    return Path(out_path)
