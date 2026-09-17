"""Teacher-forced KL between base and candidate completion distributions.

KL(P_base || P_cand) is averaged over the logits positions that predict the
completion tokens, computed in float32 so dtype noise stays out of the
scoreboard. Base-vs-base must come out at ~0 before any graft exists; that
arm is the machinery check.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

GRAFT_NOT_READY = "graft support lands with v2/v3 arms"


def logits_for(model: torch.nn.Module, ids: torch.Tensor, graft: object = None) -> torch.Tensor:
    """One no-grad forward pass: unbatched [seq, vocab] logits in float32."""
    if graft is not None:
        raise NotImplementedError(GRAFT_NOT_READY)
    with torch.no_grad():
        return model(input_ids=ids).logits[0].float()


def mean_kl(logits_base: torch.Tensor, logits_cand: torch.Tensor, completion_start: int) -> float:
    """Mean KL(base || cand) over the positions predicting completion tokens.

    ``completion_start`` is the token index where the completion begins, so
    the prediction positions are completion_start - 1 .. len - 2.
    """
    logp_base = F.log_softmax(logits_base[completion_start - 1 : -1], dim=-1)
    logp_cand = F.log_softmax(logits_cand[completion_start - 1 : -1], dim=-1)
    p_base = logp_base.exp()
    # KL support is base's: support points with p_base == 0 (softmax underflow)
    # contribute 0 by convention, so mask them out of the sum.
    terms = torch.where(p_base > 0, p_base * (logp_base - logp_cand), torch.zeros_like(p_base))
    return terms.sum(dim=-1).mean().item()


def teacher_forced_kl(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    device: str,
    prompt_ids_list: list[list[int]],
    completion_ids_list: list[list[int]],
    graft: object = None,
) -> list[float]:
    """Per-pair KL of model(+graft) completions against the unmodified base pass."""
    if graft is not None:
        raise NotImplementedError(GRAFT_NOT_READY)
    values: list[float] = []
    for prompt_ids, completion_ids in zip(prompt_ids_list, completion_ids_list, strict=True):
        if not completion_ids:
            print("[eval] warning: empty completion skipped in KL arm (no prediction positions)")
            continue
        ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long, device=device)
        base = logits_for(model, ids)
        cand = logits_for(model, ids, graft=graft)
        values.append(mean_kl(base, cand, len(prompt_ids)))
    return values
