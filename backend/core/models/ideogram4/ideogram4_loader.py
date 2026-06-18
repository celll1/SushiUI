"""Component loader for Ideogram 4 models.

Loads each sub-model to CPU; pipeline.py stages them to GPU per generation phase.
Expects a local diffusers-layout directory (the same layout published on the Hub):

    <model_path>/
        model_index.json
        transformer/                 Ideogram4Transformer2DModel  (conditional branch)
        unconditional_transformer/   Ideogram4Transformer2DModel  (asymmetric-CFG branch)
        text_encoder/                Qwen3VLModel                 (weight-only FP8)
        tokenizer/                   Qwen2Tokenizer
        vae/                         AutoencoderKLFlux2
        scheduler/                   FlowMatchEulerDiscreteScheduler

Ideogram 4 uses asymmetric classifier-free guidance: a conditional `transformer`
and a separate `unconditional_transformer` are both required at inference time.
"""

from __future__ import annotations

import json
import os
import re

import torch
from safetensors.torch import load_file

from .vendor import (
    Ideogram4Transformer2DModel,
    is_bnb4bit_state_dict,
    is_fp8_state_dict,
    load_bnb4bit_state_dict,
    load_fp8_state_dict,
    swap_linears_to_bnb4bit,
    swap_linears_to_fp8,
)
from .vendor.text_encoder import load_ideogram4_text_encoder


def _load_component_state_dict(component_dir: str, basename: str) -> dict:
    """Load a component's weights, whether sharded (index) or a single file."""
    index_path = os.path.join(component_dir, f"{basename}.safetensors.index.json")
    single_path = os.path.join(component_dir, f"{basename}.safetensors")

    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
        state_dict: dict = {}
        for shard in shard_files:
            state_dict.update(load_file(os.path.join(component_dir, shard)))
        return state_dict

    if os.path.exists(single_path):
        return load_file(single_path)

    raise FileNotFoundError(f"No safetensors found for '{basename}' in {component_dir}")


def _convert_fused_qkv_to_split(state_dict: dict, hidden_size: int) -> dict:
    """Remap native fused-QKV attention keys to the diffusers split layout.

    The ``ideogram-4-fp8`` checkpoint stores attention as a fused
    ``layers.N.attention.qkv`` (out = 3*hidden, ordered q,k,v) plus
    ``layers.N.attention.o``. The vendored diffusers transformer uses split
    ``to_q``/``to_k``/``to_v`` and ``to_out.0``. Splitting along the output rows
    is exact for both the weight and the per-row ``weight_scale``.

    No-op when the checkpoint already uses the split (diffusers) layout.
    """
    if not any(re.match(r"layers\.\d+\.attention\.qkv\.", k) for k in state_dict):
        return state_dict

    new_sd: dict = {}
    for k, v in state_dict.items():
        m_qkv = re.match(r"(layers\.\d+\.attention)\.qkv\.(weight|weight_scale|bias)$", k)
        if m_qkv:
            prefix, suffix = m_qkv.group(1), m_qkv.group(2)
            q, kk, vv = v[:hidden_size], v[hidden_size:2 * hidden_size], v[2 * hidden_size:3 * hidden_size]
            new_sd[f"{prefix}.to_q.{suffix}"] = q.contiguous()
            new_sd[f"{prefix}.to_k.{suffix}"] = kk.contiguous()
            new_sd[f"{prefix}.to_v.{suffix}"] = vv.contiguous()
            continue
        m_o = re.match(r"(layers\.\d+\.attention)\.o\.(weight|weight_scale|bias)$", k)
        if m_o:
            new_sd[f"{m_o.group(1)}.to_out.0.{m_o.group(2)}"] = v
            continue
        new_sd[k] = v
    return new_sd


def _build_ideogram4_transformer(
    model_path: str,
    subfolder: str,
    torch_dtype: torch.dtype,
) -> Ideogram4Transformer2DModel:
    """Build an Ideogram4Transformer2DModel and load its weights.

    Supports the native fused-QKV FP8 checkpoint (``ideogram-4-fp8``), the
    diffusers split-layout nf4 checkpoint (``ideogram-4-nf4-diffusers``), and a
    plain bf16 diffusers checkpoint. Naming and quantization are detected
    independently so any combination loads correctly.
    """
    component_dir = os.path.join(model_path, subfolder)
    with open(os.path.join(component_dir, "config.json"), encoding="utf-8") as f:
        config = json.load(f)

    model = Ideogram4Transformer2DModel.from_config(config)
    hidden_size = int(config["attention_head_dim"]) * int(config["num_attention_heads"])

    state_dict = _load_component_state_dict(component_dir, "diffusion_pytorch_model")
    state_dict = _convert_fused_qkv_to_split(state_dict, hidden_size)

    if is_bnb4bit_state_dict(state_dict):
        # bitsandbytes nf4 (4-bit) — requires CUDA; load directly to GPU.
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"[Ideogram4Loader] {subfolder}: nf4 (bitsandbytes) weights require a CUDA device."
            )
        device = torch.device("cuda")
        swapped = swap_linears_to_bnb4bit(model, compute_dtype=torch_dtype)
        print(f"[Ideogram4Loader] {subfolder}: swapped {swapped} Linear(s) to Linear4bit (nf4)")
        load_bnb4bit_state_dict(model, state_dict, device=device, dtype=torch_dtype)
        model.eval()
        return model

    if is_fp8_state_dict(state_dict):
        # Weight-only FP8: cast unquantized params to compute dtype, swap Fp8Linear, load.
        model.to(torch_dtype)
        swapped = swap_linears_to_fp8(model, state_dict, compute_dtype=torch_dtype)
        print(f"[Ideogram4Loader] {subfolder}: swapped {swapped} Linear(s) to Fp8Linear")
        load_fp8_state_dict(model, state_dict, device=torch.device("cpu"), dtype=torch_dtype)
    else:
        print(f"[Ideogram4Loader] {subfolder}: loading plain (unquantized) weights")
        model.load_state_dict(state_dict)
        model.to(dtype=torch_dtype)

    model.eval()
    model.to("cpu")
    return model


def load_ideogram4_components(
    model_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """Load Ideogram 4 components from a local diffusers directory.

    Returns a component dict consumed by PipelineManager.load_model():
        {
            "type": "ideogram4",
            "transformer": Ideogram4Transformer2DModel,
            "unconditional_transformer": Ideogram4Transformer2DModel,
            "text_encoder": Qwen3VLModel,
            "tokenizer": PreTrainedTokenizer,
            "vae": AutoencoderKLFlux2,
            "scheduler": FlowMatchEulerDiscreteScheduler,
        }
    """
    from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
    from transformers import AutoTokenizer

    print(f"[Ideogram4Loader] Loading components from: {model_path}")

    print("[Ideogram4Loader] Loading transformer (conditional)...")
    transformer = _build_ideogram4_transformer(model_path, "transformer", torch_dtype)

    print("[Ideogram4Loader] Loading unconditional_transformer (asymmetric-CFG branch)...")
    unconditional_transformer = _build_ideogram4_transformer(
        model_path, "unconditional_transformer", torch_dtype
    )

    print("[Ideogram4Loader] Loading text encoder (Qwen3-VL)...")
    text_encoder = load_ideogram4_text_encoder(model_path, torch_dtype=torch_dtype, device="cpu")

    print("[Ideogram4Loader] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, "tokenizer"))
    # Sanity encode: surface corrupted vocabulary files at load time rather than
    # at the first generation step where the error message is less informative.
    try:
        tokenizer.encode("validation", add_special_tokens=False)
    except Exception as e:
        raise RuntimeError(
            f"[Ideogram4Loader] Tokenizer sanity encode failed — vocabulary files may be corrupted "
            f"({model_path}/tokenizer): {e}"
        ) from e

    print("[Ideogram4Loader] Loading VAE (AutoencoderKLFlux2)...")
    vae = AutoencoderKLFlux2.from_pretrained(
        model_path, subfolder="vae", torch_dtype=torch_dtype, low_cpu_mem_usage=True
    )
    vae.eval()
    vae.to("cpu")

    print("[Ideogram4Loader] Loading scheduler (FlowMatchEulerDiscreteScheduler)...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")

    print("[Ideogram4Loader] All components loaded successfully.")
    return {
        "type": "ideogram4",
        "transformer": transformer,
        "unconditional_transformer": unconditional_transformer,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "vae": vae,
        "scheduler": scheduler,
    }
