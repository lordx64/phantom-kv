"""Model loading: device/dtype policy and environment provenance."""

from __future__ import annotations

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase


def load_model(model_id: str) -> tuple[torch.nn.Module, PreTrainedTokenizerBase, dict]:
    """Load a causal LM on MPS when available, else CPU.

    bfloat16 is tried first; if the load or a one-step forward fails, the
    load is retried in float16. Returns (model, tokenizer, env_info) where
    env_info carries provenance for the eval report.
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    errors: dict[str, str] = {}
    for dtype in (torch.bfloat16, torch.float16):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=dtype,
                attn_implementation="sdpa",
            )
            model.to(device)
            model.eval()
            with torch.no_grad():
                model(input_ids=torch.zeros((1, 1), dtype=torch.long, device=device))
        except Exception as err:  # noqa: BLE001 -- any load/run failure retries in the next dtype
            print(f"[model] {dtype} unusable on {device}: {type(err).__name__}: {err}")
            errors[str(dtype)] = f"{type(err).__name__}: {err}"
            continue

        info = {
            "model_id": model_id,
            "device": device,
            "dtype": str(dtype).removeprefix("torch."),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
        }
        return model, tokenizer, info

    raise RuntimeError(f"failed to load {model_id} in any supported dtype: {errors}")
