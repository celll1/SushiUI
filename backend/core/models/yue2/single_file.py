"""Canonical complete YuE2 checkpoint writer and official-core repacker."""
from __future__ import annotations

from pathlib import Path
import os
import re

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .loader import TOKENIZER_KEY, UPSTREAM_REVISION, build_empty_models, preflight_yue2, read_header


def model_key_to_single_file(key: str) -> str:
    """Map the native YuE2 module namespace to SushiUI's complete-file layout."""
    if key == "lm_head.weight":
        return "text_encoders.model.lm_head.weight"
    if key.startswith("model.embed_tokens.") or key.startswith("model.norm."):
        return "text_encoders." + key
    if key == "nar_norm.weight":
        return "model.diffusion_model.model.norm.weight"
    if key.startswith("model.layers."):
        match = re.match(r"model\.layers\.(\d+)\.(.+)", key)
        if match is None:
            raise ValueError(f"Unrecognized YuE2 model tensor: {key}")
        layer, suffix = match.groups()
        nar_names = {
            "nar_input_layernorm": "input_layernorm",
            "nar_pre_mlp_layernorm": "post_attention_layernorm",
            "nar_self_attn": "self_attn",
            "nar_mlp": "mlp",
        }
        head = suffix.split(".", 1)[0]
        if head in nar_names:
            suffix = nar_names[head] + suffix[len(head):]
            return f"model.diffusion_model.model.layers.{layer}.{suffix}"
        return "text_encoders." + key
    if key.startswith(("vae2llm.", "llm2vae.", "time_embedder.", "latent_pos_embed.")):
        return "model.diffusion_model." + key
    raise ValueError(f"Unrecognized YuE2 model tensor: {key}")


def _tokenizer_tensor(tokenizer) -> torch.Tensor:
    payload = getattr(tokenizer, "payload", None)
    if not isinstance(payload, str):
        raise ValueError("YuE2 checkpoint save requires the original embedded tokenizer payload")
    return torch.tensor(list(payload.encode("utf-8")), dtype=torch.uint8)


def _metadata(extra_metadata=None) -> dict[str, str]:
    metadata = {
        "format": "pt",
        "model_type": "yue2",
        "source": "m-a-p/YuE2-3B",
        "license": "CC-BY-NC-4.0",
        "yue2_format": "1",
        "yue2_weight_storage": "dense_bf16",
        "upstream_revision": UPSTREAM_REVISION,
        "component.vae.embedded": "1",
        "component.te.embedded": "1",
    }
    if extra_metadata:
        metadata.update({key: str(value) for key, value in extra_metadata.items()})
    return metadata


def save_yue2_single_file(path, transformer, vae, tokenizer, *, extra_metadata=None) -> None:
    """Write one complete dense-BF16 YuE2 file; quantized training saves are forbidden."""
    from core.training.adapters.base_adapter import reject_quantized_base

    reject_quantized_base(transformer, model_label="YuE2")
    state = {}
    for key, value in transformer.state_dict().items():
        if value.is_meta:
            raise ValueError(f"Cannot save meta YuE2 tensor: {key}")
        if not value.is_floating_point():
            raise ValueError(f"Dense YuE2 model tensor must be floating point: {key}")
        state[model_key_to_single_file(key)] = value.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    for key, value in vae.state_dict().items():
        if value.is_meta:
            raise ValueError(f"Cannot save meta YuE2 VAE tensor: {key}")
        state["vae." + key] = value.detach().to(device="cpu").contiguous()
    state[TOKENIZER_KEY] = _tokenizer_tensor(tokenizer)
    output = Path(path)
    if output.suffix.lower() != ".safetensors":
        raise ValueError("YuE2 checkpoints must use a .safetensors filename")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        save_file(state, str(temporary), metadata=_metadata(extra_metadata))
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_official_header(path) -> dict:
    header = read_header(path)
    model, _ = build_empty_models()
    expected = dict(model.state_dict())
    # The release shares model.norm between AR and NAR; SushiUI stores an explicit
    # NAR copy so training can freeze the acoustic half independently.
    expected.pop("nar_norm.weight")
    actual = set(header) - {"__metadata__"}
    if actual != set(expected):
        missing, extra = sorted(set(expected) - actual), sorted(actual - set(expected))
        raise ValueError(f"Official YuE2 core mismatch; missing={missing[:8]}, extra={extra[:8]}")
    for key, tensor in expected.items():
        info = header[key]
        if info.get("dtype") != "BF16" or list(info.get("shape", [])) != list(tensor.shape):
            raise ValueError(f"Official YuE2 tensor is not canonical BF16: {key}")
    return header


def repack_official_dense(official_path, asset_checkpoint, output_path) -> None:
    """Bundle the official dense core with the licensed VAE/tokenizer assets."""
    official = Path(official_path).resolve()
    assets = Path(asset_checkpoint).resolve()
    output = Path(output_path).resolve()
    if output in {official, assets}:
        raise ValueError("YuE2 repack output must not overwrite an input")
    _validate_official_header(official)
    asset_plan = preflight_yue2(assets)
    state = {}
    with safe_open(official, framework="pt", device="cpu") as source:
        for key in source.keys():
            state[model_key_to_single_file(key)] = source.get_tensor(key)
        state[model_key_to_single_file("nar_norm.weight")] = source.get_tensor("model.norm.weight").clone()
    with safe_open(assets, framework="pt", device="cpu") as source:
        for item in asset_plan["plan"]:
            if item.target == "tokenizer" or not item.target.startswith("vae."):
                continue
            state[item.target] = source.get_tensor(item.source)
        state[TOKENIZER_KEY] = source.get_tensor(TOKENIZER_KEY)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        save_file(state, str(temporary), metadata=_metadata({
            "vae_source_checkpoint": assets.name,
            "official_core_checkpoint": official.name,
            "official_core_tensor_count": "628",
        }))
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    result = preflight_yue2(output)
    if result["quantized"] or result["metadata"].get("yue2_weight_storage") != "dense_bf16":
        raise RuntimeError("Repacked YuE2 checkpoint failed dense preflight")
