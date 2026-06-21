"""Component loader for MiniT2I (pixel-space MM-JiT + FLAN-T5).

Supports two layouts:
  1. diffusers directory  (a single variant dir): <model>/transformer/ + <model>/scheduler/
  2. single-file safetensors: bundles the transformer (+ optionally FLAN-T5),
     variant auto-detected (see vendor/single_file.py).

FLAN-T5-Large (frozen text encoder) is loaded from, in order: an explicit path,
a sibling/`flan-t5-large` directory next to the model, or the HF hub.
No VAE (pixel space).
"""

from __future__ import annotations

import os

import torch

from .vendor import MiniT2IMMJiTModel, MiniT2IFlowMatchScheduler
from .vendor.single_file import load_single_file, detect_variant_from_state_dict


def _resolve_flan_t5(model_path: str, flan_t5_path: str | None) -> str:
    """Resolve the FLAN-T5-Large location (local dir preferred, else hub id)."""
    candidates = []
    if flan_t5_path:
        candidates.append(flan_t5_path)
    base = os.path.dirname(model_path.rstrip("/\\")) if os.path.isfile(model_path) else model_path
    # common local layouts under M:\model\minit2i\
    parent = os.path.dirname(base.rstrip("/\\"))
    candidates += [
        os.path.join(base, "flan-t5-large"),
        os.path.join(parent, "flan-t5-large"),
        os.path.join(parent, "flan-t5-large", "flan-t5-large"),
    ]
    for c in candidates:
        if c and os.path.isdir(c):
            return c
    return "google/flan-t5-large"  # hub fallback


def _load_flan_t5(location: str, torch_dtype: torch.dtype):
    from transformers import AutoTokenizer, T5EncoderModel
    tokenizer = AutoTokenizer.from_pretrained(location)
    text_encoder = T5EncoderModel.from_pretrained(location, torch_dtype=torch_dtype)
    text_encoder.eval()
    return tokenizer, text_encoder


def _detect_variant_name(transformer: MiniT2IMMJiTModel) -> str:
    cfg = transformer.mmjit_config
    if cfg.hidden_size == 1248 or cfg.depth_double == 23:
        return "l16"
    return "b16"


def load_minit2i_components(
    model_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    flan_t5_path: str | None = None,
    text_encoder_dtype: torch.dtype = torch.float32,
) -> dict:
    """Load MiniT2I components from a diffusers dir or a single-file safetensors.

    Returns a component dict consumed by PipelineManager.load_model():
        {type:"minit2i", transformer, scheduler, text_encoder, tokenizer, variant}
    """
    is_single_file = os.path.isfile(model_path) and model_path.endswith(".safetensors")

    if is_single_file:
        print(f"[MiniT2ILoader] Loading single-file: {model_path}")
        bundle = load_single_file(model_path, torch_dtype=torch_dtype)
        transformer = bundle["transformer"]
        variant = bundle["variant"] or _detect_variant_name(transformer)
        scheduler = MiniT2IFlowMatchScheduler()  # defaults (lognorm, n_T 100)

        te_sd = bundle.get("text_encoder_state_dict")
        flan_loc = _resolve_flan_t5(model_path, flan_t5_path)
        if te_sd is not None:
            # FLAN-T5 weights are embedded; build arch from config and load them.
            from transformers import AutoTokenizer, T5EncoderModel, AutoConfig
            cfg = AutoConfig.from_pretrained(flan_loc)
            text_encoder = T5EncoderModel(cfg).to(text_encoder_dtype)
            text_encoder.load_state_dict(te_sd, strict=False)
            text_encoder.eval()
            tokenizer = AutoTokenizer.from_pretrained(flan_loc)
        else:
            tokenizer, text_encoder = _load_flan_t5(flan_loc, text_encoder_dtype)
    else:
        print(f"[MiniT2ILoader] Loading diffusers directory: {model_path}")
        transformer_dir = os.path.join(model_path, "transformer")
        if not os.path.isdir(transformer_dir):
            transformer_dir = model_path  # allow pointing directly at the transformer dir
        transformer = MiniT2IMMJiTModel.from_pretrained(transformer_dir, torch_dtype=torch_dtype)
        variant = _detect_variant_name(transformer)

        scheduler_dir = os.path.join(model_path, "scheduler")
        if os.path.isdir(scheduler_dir):
            scheduler = MiniT2IFlowMatchScheduler.from_pretrained(scheduler_dir)
        else:
            scheduler = MiniT2IFlowMatchScheduler()

        flan_loc = _resolve_flan_t5(model_path, flan_t5_path)
        tokenizer, text_encoder = _load_flan_t5(flan_loc, text_encoder_dtype)

    transformer.eval()
    transformer.to("cpu")
    text_encoder.to("cpu")
    print(f"[MiniT2ILoader] Loaded MiniT2I variant={variant} (FLAN-T5 from {flan_loc})")

    return {
        "type": "minit2i",
        "transformer": transformer,
        "scheduler": scheduler,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "variant": variant,
    }
