"""Component loader for Krea 2 (single-stream MMDiT + Qwen3-VL + Qwen-Image VAE).

Loads each sub-model to CPU; pipeline.py stages them to GPU per generation phase.
Supports the weight layouts enumerated in vendor/single_file.py:

  * Diffusers folder (krea/Krea-2-Raw): transformer/ (possibly sharded),
    text_encoder/, tokenizer/, vae/, scheduler/, model_index.json.
  * A folder containing only transformer/ -> the text encoder / VAE / tokenizer are
    auto-complemented (sibling probe + env override + HF hub fallback).
  * A single-file safetensors (diffusers / raw / comfy / sushiUI TE+DiT combined).

Auto-complement env overrides: ``KREA2_TE_DIR`` (Qwen3-VL text encoder dir),
``KREA2_VAE_DIR`` (Qwen-Image VAE dir). HF hub fallbacks:
``Qwen/Qwen3-VL-4B-Instruct`` (TE + tokenizer) and ``Qwen/Qwen-Image`` subfolder
``vae``.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import torch

from .vendor.single_file import (
    KREA2_DEFAULT_CONFIG,
    build_krea2_transformer,
    detect_config_and_variant,
    load_single_file,
    normalize_state_dict,
    reject_unsupported_quant,
)
# Reuse ideogram4's sharded-state-dict assembler and FP8-aware Qwen3-VL TE loader.
from core.models.ideogram4.ideogram4_loader import _load_component_state_dict


TE_SIBLING_NAMES = ("text_encoder", "Qwen3-VL-4B-Instruct", "qwen3-vl-4b", "qwen3_vl_4b")
VAE_SIBLING_NAMES = ("vae", "qwen-image-vae", "qwen_image_vae", "Qwen-Image-VAE")
TE_HUB_ID = "Qwen/Qwen3-VL-4B-Instruct"
VAE_HUB_ID = "Qwen/Qwen-Image"


def _has_config(d: Optional[str]) -> bool:
    return bool(d) and os.path.isdir(d) and os.path.isfile(os.path.join(d, "config.json"))


def _probe_sibling(base: str, names, validator) -> Optional[str]:
    """Walk `base` and up to 4 ancestors, probing each `names` entry (and one extra
    nesting level) for a directory passing `validator`."""
    cur = os.path.dirname(base.rstrip("/\\")) if os.path.isfile(base) else base
    cur = cur.rstrip("/\\")
    for _ in range(5):
        for nm in names:
            cand = os.path.join(cur, nm)
            if validator(cand):
                return cand
            cand2 = os.path.join(cand, nm)
            if validator(cand2):
                return cand2
        nxt = os.path.dirname(cur)
        if nxt == cur:
            break
        cur = nxt
    return None


def _resolve_te_dir(model_path: str, override: Optional[str]) -> Optional[str]:
    if override and _has_config(override):
        return override
    env = os.environ.get("KREA2_TE_DIR")
    if _has_config(env):
        return env
    # Prefer the model's own text_encoder/ subfolder when present.
    own = os.path.join(model_path, "text_encoder") if os.path.isdir(model_path) else None
    if _has_config(own):
        return own
    found = _probe_sibling(model_path, TE_SIBLING_NAMES, _has_config)
    return found  # None -> hub fallback handled by caller


def _resolve_vae_dir(model_path: str, override: Optional[str]) -> Optional[str]:
    if override and _has_config(override):
        return override
    env = os.environ.get("KREA2_VAE_DIR")
    if _has_config(env):
        return env
    own = os.path.join(model_path, "vae") if os.path.isdir(model_path) else None
    if _has_config(own):
        return own
    return _probe_sibling(model_path, VAE_SIBLING_NAMES, _has_config)


def _load_qwen3vl_text_encoder(te_dir: str, torch_dtype: torch.dtype):
    """Load the Qwen3-VL text encoder. Delegates to the ideogram4 FP8-aware loader
    when the directory is laid out as ``<parent>/text_encoder`` (its expected shape);
    otherwise loads a standard checkpoint via transformers directly."""
    from transformers import AutoModel

    parent = os.path.dirname(te_dir.rstrip("/\\"))
    if os.path.basename(te_dir.rstrip("/\\")) == "text_encoder":
        # ideogram4 loader handles both standard bf16 and weight-only FP8 Qwen3-VL.
        from core.models.ideogram4.vendor.text_encoder import load_ideogram4_text_encoder
        return load_ideogram4_text_encoder(parent, torch_dtype=torch_dtype, device="cpu")

    print(f"[Krea2Loader] Loading Qwen3-VL text encoder (standard) from: {te_dir}")
    model = AutoModel.from_pretrained(te_dir, torch_dtype=torch_dtype, trust_remote_code=True)
    model.to("cpu")
    model.eval()
    return model


def _load_qwen_image_vae(vae_dir: Optional[str], torch_dtype: torch.dtype):
    from diffusers import AutoencoderKLQwenImage

    if vae_dir and _has_config(vae_dir):
        print(f"[Krea2Loader] Loading Qwen-Image VAE from: {vae_dir}")
        vae = AutoencoderKLQwenImage.from_pretrained(vae_dir, torch_dtype=torch_dtype, low_cpu_mem_usage=True)
    else:
        print(f"[Krea2Loader] Loading Qwen-Image VAE from hub: {VAE_HUB_ID} (subfolder=vae)")
        vae = AutoencoderKLQwenImage.from_pretrained(
            VAE_HUB_ID, subfolder="vae", torch_dtype=torch_dtype, low_cpu_mem_usage=True
        )
    vae.eval()
    vae.to("cpu")
    return vae


def _load_tokenizer(te_dir: Optional[str]):
    from transformers import AutoTokenizer

    src = te_dir if (te_dir and os.path.isfile(os.path.join(te_dir, "tokenizer_config.json"))) else None
    if src is None:
        src = TE_HUB_ID
    print(f"[Krea2Loader] Loading tokenizer from: {src}")
    return AutoTokenizer.from_pretrained(src, trust_remote_code=True)


def _load_scheduler(model_path: str):
    from diffusers import FlowMatchEulerDiscreteScheduler

    sched_dir = os.path.join(model_path, "scheduler") if os.path.isdir(model_path) else None
    if sched_dir and os.path.isfile(os.path.join(sched_dir, "scheduler_config.json")):
        return FlowMatchEulerDiscreteScheduler.from_pretrained(sched_dir)
    # Krea 2 resolution-aware exponential time shift defaults.
    return FlowMatchEulerDiscreteScheduler(
        base_shift=0.5, max_shift=1.15,
        base_image_seq_len=256, max_image_seq_len=6400,
        use_dynamic_shifting=True,
    )


def _detect_is_distilled_dir(model_path: str) -> bool:
    idx_path = os.path.join(model_path, "model_index.json")
    if os.path.isfile(idx_path):
        try:
            with open(idx_path, encoding="utf-8") as f:
                idx = json.load(f)
            return bool(idx.get("is_distilled", False))
        except Exception:
            pass
    return "turbo" in model_path.lower() or "distill" in model_path.lower()


def _build_transformer_from_dir(model_path: str, torch_dtype: torch.dtype):
    """Build the transformer from a diffusers folder (or a transformer-only folder)."""
    transformer_dir = os.path.join(model_path, "transformer")
    if not os.path.isdir(transformer_dir):
        transformer_dir = model_path  # allow pointing directly at the transformer dir
    cfg_path = os.path.join(transformer_dir, "config.json")
    config = dict(KREA2_DEFAULT_CONFIG)
    if os.path.isfile(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            file_cfg = json.load(f)
        for k in KREA2_DEFAULT_CONFIG:
            if k in file_cfg:
                config[k] = file_cfg[k]

    state_dict = _load_component_state_dict(transformer_dir, "diffusion_pytorch_model")
    diffusers_sd, _ = normalize_state_dict(state_dict, metadata={})
    return build_krea2_transformer(diffusers_sd, config, torch_dtype)


def load_krea2_components(
    model_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    te_dir: Optional[str] = None,
    vae_dir: Optional[str] = None,
    load_text_encoder: bool = True,
) -> dict:
    """Load Krea 2 components from a diffusers directory or single-file safetensors.

    Returns a component dict consumed by PipelineManager.load_model():
        {type:"krea2", transformer, scheduler, text_encoder, tokenizer, vae,
         vae_scale_factor, is_distilled, text_encoder_select_layers, patch_size,
         config}
    """
    is_single_file = os.path.isfile(model_path) and (
        model_path.endswith(".safetensors") or model_path.endswith(".safetensors.index.json")
    )

    embedded_te_sd = None
    if is_single_file:
        print(f"[Krea2Loader] Loading single-file: {model_path}")
        bundle = load_single_file(model_path, torch_dtype=torch_dtype)
        transformer = bundle["transformer"]
        is_distilled = bundle["is_distilled"]
        config = bundle["config"]
        embedded_te_sd = bundle.get("text_encoder_state_dict")
    else:
        print(f"[Krea2Loader] Loading diffusers directory: {model_path}")
        transformer = _build_transformer_from_dir(model_path, torch_dtype)
        is_distilled = _detect_is_distilled_dir(model_path)
        config = dict(getattr(transformer, "config", KREA2_DEFAULT_CONFIG))

    scheduler = _load_scheduler(model_path)

    resolved_te = _resolve_te_dir(model_path, te_dir)
    resolved_vae = _resolve_vae_dir(model_path, vae_dir)

    text_encoder = None
    tokenizer = None
    if load_text_encoder:
        if embedded_te_sd is not None and resolved_te is not None:
            # sushiUI bundle: rebuild the Qwen3-VL architecture from the resolved
            # config dir and load the embedded weights.
            from transformers import AutoConfig, AutoModel
            print("[Krea2Loader] Rebuilding embedded Qwen3-VL text encoder from bundle weights...")
            te_config = AutoConfig.from_pretrained(resolved_te, trust_remote_code=True)
            text_encoder = AutoModel.from_config(te_config, trust_remote_code=True).to(torch_dtype)
            text_encoder.load_state_dict(embedded_te_sd, strict=False)
            text_encoder.eval()
        elif resolved_te is not None:
            text_encoder = _load_qwen3vl_text_encoder(resolved_te, torch_dtype)
        else:
            print(f"[Krea2Loader] Text encoder dir not found; loading from hub: {TE_HUB_ID}")
            from transformers import AutoModel
            text_encoder = AutoModel.from_pretrained(TE_HUB_ID, torch_dtype=torch_dtype, trust_remote_code=True)
            text_encoder.to("cpu").eval()
        tokenizer = _load_tokenizer(resolved_te)

    vae = _load_qwen_image_vae(resolved_vae, torch_dtype)
    vae_scale_factor = 2 ** len(vae.temperal_downsample) if hasattr(vae, "temperal_downsample") else 8

    select_layers = config.get("text_encoder_select_layers") or [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]

    print(f"[Krea2Loader] Loaded Krea 2 (is_distilled={is_distilled}, vae_scale_factor={vae_scale_factor})")
    return {
        "type": "krea2",
        "transformer": transformer,
        "scheduler": scheduler,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "vae": vae,
        "vae_scale_factor": vae_scale_factor,
        "is_distilled": bool(is_distilled),
        "text_encoder_select_layers": list(select_layers),
        "patch_size": 2,
        "config": config,
    }
