"""Ideogram 4 text encoder (Qwen3-VL) loader.

The text encoder is a standard ``transformers`` ``Qwen3VLModel``; only the
weight-only FP8 storage layout is non-standard. ``transformers``'
``from_pretrained`` cannot read the ``weight_scale`` tensors, so for FP8
checkpoints we rebuild the architecture from its config and load the FP8 state
dict ourselves (see ``fp8_linear``). Plain bf16 checkpoints load normally.

The per-token text conditioning consumed by the Ideogram 4 transformer is built
in ``ideogram4_pipeline_ops.encode_prompt`` (it taps 13 intermediate decoder
layers of ``model.language_model``). This module only handles loading.
"""

from __future__ import annotations

import json
import os

import torch
from safetensors.torch import load_file

from .fp8_linear import (
    FP8_TEXT_ENCODER_CONFIG_FLAG,
    is_fp8_state_dict,
    load_fp8_state_dict,
    swap_linears_to_fp8,
)


def _read_te_config_flags(text_encoder_dir: str) -> dict:
    with open(os.path.join(text_encoder_dir, "config.json"), encoding="utf-8") as f:
        return json.load(f)


def load_ideogram4_text_encoder(
    model_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cpu",
):
    """Load the Qwen3-VL text encoder from a local Ideogram 4 model directory.

    Returns the ``transformers`` model placed on ``device`` in eval mode. For FP8
    checkpoints the Linear weights stay float8 (dequantized on the fly); every
    other floating tensor is cast to ``torch_dtype``.
    """
    from transformers import AutoConfig, AutoModel

    text_encoder_dir = os.path.join(model_path, "text_encoder")
    cfg_data = _read_te_config_flags(text_encoder_dir)
    is_fp8 = bool(cfg_data.get(FP8_TEXT_ENCODER_CONFIG_FLAG, False))
    is_quantized = "quantization_config" in cfg_data  # e.g. bitsandbytes nf4

    device = torch.device(device)

    if is_quantized and not is_fp8:
        # bitsandbytes (nf4) checkpoint: transformers handles 4-bit placement via
        # device_map; the weights stay on the mapped CUDA device.
        import torch as _torch
        map_device = device if device.type == "cuda" else (
            _torch.device("cuda") if _torch.cuda.is_available() else device
        )
        print("[Ideogram4TE] Loading Qwen3-VL text encoder (bitsandbytes quantization_config)...")
        model = AutoModel.from_pretrained(
            text_encoder_dir, torch_dtype=torch_dtype, trust_remote_code=True,
            device_map={"": map_device},
        )
        model.eval()
        return model

    if not is_fp8:
        # Standard path (e.g. a bf16 community checkpoint).
        print("[Ideogram4TE] Loading Qwen3-VL text encoder (standard from_pretrained)...")
        model = AutoModel.from_pretrained(
            text_encoder_dir, torch_dtype=torch_dtype, trust_remote_code=True
        )
        model.to(device)
        model.eval()
        return model

    print("[Ideogram4TE] Loading Qwen3-VL text encoder (weight-only FP8)...")
    config = AutoConfig.from_pretrained(text_encoder_dir, trust_remote_code=True)

    # Instantiate the architecture from config so the non-persistent buffers
    # (rotary caches) are computed. Skip the (very slow for an 8B model) random
    # weight initialization — every param is overwritten by the FP8 load below.
    # Init under the compute dtype to keep the transient CPU allocation at ~1x
    # model size instead of the float32 default.
    try:
        from transformers.initialization import no_init_weights
    except Exception:  # pragma: no cover - fallback for other transformers versions
        from contextlib import nullcontext
        def no_init_weights():
            return nullcontext()

    print("[Ideogram4TE] Building architecture (no_init_weights)...")
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch_dtype)
    try:
        with no_init_weights():
            model = AutoModel.from_config(config, trust_remote_code=True)
    finally:
        torch.set_default_dtype(default_dtype)

    state_dict_path = os.path.join(text_encoder_dir, "model.safetensors")
    print("[Ideogram4TE] Reading FP8 state dict from disk...")
    state_dict = load_file(state_dict_path)

    if not is_fp8_state_dict(state_dict):
        raise RuntimeError(
            f"[Ideogram4TE] config flags FP8 but no FP8 tensors found in {state_dict_path}"
        )

    swapped = swap_linears_to_fp8(model, state_dict, compute_dtype=torch_dtype)
    print(f"[Ideogram4TE] Swapped {swapped} Linear layer(s) to Fp8Linear")

    # assign=True: unquantized params take the loaded dtype and the computed
    # rotary buffers (absent from the checkpoint) survive; tied weights, if any,
    # surface as benign missing keys (strict=False).
    load_fp8_state_dict(
        model, state_dict, device=device, dtype=torch_dtype, assign=True, strict=False
    )
    model.eval()
    print("[Ideogram4TE] Text encoder loaded.")
    return model
