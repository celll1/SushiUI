"""Selective SenseNova understanding-branch loader and prefix capture."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from core.models.common.single_file_format import TRANSFORMER_PREFIX
from core.models.sensenova.loader import (
    _LazySafetensorsSource,
    _int8_convrot_source_layers,
    _load_sensenova_config,
    _load_sensenova_tokenizer,
    _reshape_convrot_scales,
    install_sensenova_state_dict,
    is_sensenova_state_dict_keys,
)
from core.models.sensenova.vendor import NEOChatModel
from core.models.sensenova.vendor.configuration_neo_chat import NEOMoELLMConfig

from .artifact import checkpoint_headers


@dataclass(frozen=True)
class UnderstandingPrefix:
    hidden_states: torch.Tensor
    layer_kv: Mapping[int, tuple[torch.Tensor, torch.Tensor]]
    attention_mask: torch.Tensor
    positions: torch.Tensor


def is_understanding_tensor_key(key: str) -> bool:
    stripped = key[len(TRANSFORMER_PREFIX) :] if key.startswith(TRANSFORMER_PREFIX) else key
    if "_mot_gen" in stripped or stripped.startswith("fm_modules."):
        return False
    return stripped.startswith("language_model.") or stripped.startswith("vision_model.")


def _read_understanding_state_dict(path: str) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    shapes, metadata = checkpoint_headers(path)
    if not is_sensenova_state_dict_keys(shapes):
        raise ValueError(f"understanding source is not a supported SenseNova checkpoint: {path}")
    selected = sorted(key for key in shapes if is_understanding_tensor_key(key))
    if not selected:
        raise ValueError(f"SenseNova source carries no understanding tensors: {path}")
    state: dict[str, torch.Tensor] = {}
    with _LazySafetensorsSource(path) as source:
        for key in selected:
            state[key[len(TRANSFORMER_PREFIX) :] if key.startswith(TRANSFORMER_PREFIX) else key] = source.get(key)
    return state, metadata


def _remove_generation_branch(model: NEOChatModel) -> None:
    model.fm_modules = torch.nn.ModuleDict()
    language = model.language_model.model
    if hasattr(language, "norm_mot_gen"):
        del language.norm_mot_gen
    for layer in language.layers:
        for name in ("mlp_mot_gen", "input_layernorm_mot_gen", "post_attention_layernorm_mot_gen"):
            if hasattr(layer, name):
                delattr(layer, name)
        attention = layer.self_attn
        for name in (
            "q_proj_mot_gen", "k_proj_mot_gen", "v_proj_mot_gen", "o_proj_mot_gen",
            "q_norm_mot_gen", "q_norm_hw_mot_gen", "k_norm_mot_gen", "k_norm_hw_mot_gen",
        ):
            if hasattr(attention, name):
                delattr(attention, name)


def _configure_understanding_tokens(model: NEOChatModel, tokenizer: Any) -> None:
    model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    model.img_start_token_id = tokenizer.convert_tokens_to_ids("<img>")
    if model.img_context_token_id is None or model.img_start_token_id is None:
        raise ValueError("SenseNova tokenizer is missing required image-prefix tokens")


def load_understanding_only(
    model_path: str,
    *,
    torch_dtype: torch.dtype | None = torch.bfloat16,
) -> dict[str, Any]:
    """Load only the frozen dense SenseNova understanding path on CPU."""
    from accelerate import init_empty_weights

    path = str(Path(model_path).resolve())
    state, metadata = _read_understanding_state_dict(path)
    config, config_dict = _load_sensenova_config(metadata, str(Path(path).parent))
    if isinstance(config.llm_config, NEOMoELLMConfig):
        raise ValueError("Chimera format v1 supports the dense SenseNova understanding branch only")

    convrot = _int8_convrot_source_layers(state, path=path)
    if convrot:
        from core.models.common.convrot_int8_linear import require_convrot_int8_runtime

        require_convrot_int8_runtime()
    _reshape_convrot_scales(state, convrot)

    dtype = torch_dtype or torch.bfloat16
    with init_empty_weights():
        model = NEOChatModel(config)
        _remove_generation_branch(model)
        model.to(dtype)
    swapped = install_sensenova_state_dict(model, state, convrot, dtype, path=path)
    model.tie_weights()
    meta = [name for name, value in (*model.named_parameters(), *model.named_buffers()) if value.device.type == "meta"]
    if meta:
        raise RuntimeError(
            f"SenseNova understanding-only load left {len(meta)} meta tensor(s); first: {meta[:5]}"
        )
    if any("_mot_gen" in name or name.startswith("fm_modules.") for name in model.state_dict()):
        raise RuntimeError("SenseNova understanding-only tree retained generation tensors")
    model.eval()
    model.requires_grad_(False)
    tokenizer = _load_sensenova_tokenizer(str(Path(path).parent))
    _configure_understanding_tokens(model, tokenizer)
    return {
        "transformer": model,
        "tokenizer": tokenizer,
        "config": config,
        "config_dict": config_dict,
        "metadata": dict(metadata),
        "model_path": path,
        "loaded_tensor_count": len(state),
        "quantized_linear_count": swapped,
    }


def capture_understanding_prefix(
    transformer: NEOChatModel,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    selected_layers: tuple[int, ...],
    grid_hw: torch.Tensor | None = None,
    pixel_values: torch.Tensor | None = None,
) -> UnderstandingPrefix:
    """Run one prefix pass and retain only final hidden state plus selected K/V."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("SenseNova prefix capture currently requires batch size 1")
    if transformer.img_context_token_id is None:
        raise RuntimeError("img_context_token_id must be configured before prefix capture")
    indexes = transformer.get_thw_indexes(input_ids[0], grid_hw)
    inputs_embeds = transformer.language_model.get_input_embeddings()(input_ids)
    if pixel_values is not None:
        if transformer.img_context_token_id is None:
            raise RuntimeError("img_context_token_id must be configured for image prefix capture")
        visual = transformer.extract_feature(pixel_values, grid_hw=grid_hw)
        flat = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
        selected = input_ids.reshape(-1) == transformer.img_context_token_id
        if int(selected.sum()) != int(visual.reshape(-1, visual.shape[-1]).shape[0]):
            raise ValueError("image-context token count does not match visual feature count")
        flat[selected] = visual.reshape(-1, visual.shape[-1]).to(flat)
        inputs_embeds = flat.reshape_as(inputs_embeds)
    outputs = transformer.language_model.model(
        inputs_embeds=inputs_embeds,
        indexes=indexes,
        attention_mask=attention_mask,
        use_cache=False,
        capture_layers=selected_layers,
    )
    positions = indexes.transpose(0, 1).unsqueeze(0).to(inputs_embeds.device)
    return UnderstandingPrefix(
        hidden_states=outputs.last_hidden_state,
        layer_kv=outputs.selected_kv or {},
        attention_mask=attention_mask.to(dtype=torch.bool),
        positions=positions,
    )
