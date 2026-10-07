"""Preserve safetensors backing when Gemma weights already match the load dtype."""

import json
from pathlib import Path

import torch
from safetensors import safe_open


def _assign_weights(model, state):
    expected = model.state_dict()
    rules = (
        ("language_model.model.", "model.language_model."),
        ("language_model.lm_head.", "lm_head."),
        ("vision_tower.vision_model.", "model.vision_tower."),
        ("vision_tower.", "model.vision_tower."),
        ("multi_modal_projector.", "model.multi_modal_projector."),
    )
    mapped = {}
    for key, value in state.items():
        name = key
        if name not in expected:
            for old, new in rules:
                if key.startswith(old):
                    name = new + key[len(old):]
                    break
        if name not in expected or name in mapped:
            raise ValueError(f"Unexpected or duplicate Gemma tensor: {key}")
        if value.shape != expected[name].shape:
            raise ValueError(f"Gemma tensor shape mismatch: {key}")
        mapped[name] = value
    parameters = dict(model.named_parameters(remove_duplicate=False))
    for name in expected.keys() - mapped.keys():
        aliases = [key for key, parameter in parameters.items()
                   if parameter is parameters.get(name) and key in mapped]
        if not aliases:
            raise ValueError(f"Missing untied Gemma tensor: {name}")
        mapped[name] = mapped[aliases[0]]
    model.load_state_dict(mapped, strict=True, assign=True)
    model.tie_weights()


def load_matching_mapped_encoder(directory, torch_dtype):
    """Return None for dtype conversion or a non-Gemma component."""
    directory = Path(directory)
    config_path = directory / "config.json"
    if not config_path.is_file():
        return None
    if json.loads(config_path.read_text(encoding="utf-8")).get("model_type") != "gemma3":
        return None
    index_path = directory / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        files = sorted(set(weight_map.values()))
    elif (directory / "model.safetensors").is_file():
        weight_map = None
        files = ["model.safetensors"]
    else:
        return None
    state = {}
    for filename in files:
        path = (directory / filename).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("Gemma shard escapes its component directory")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in state or (weight_map is not None and weight_map.get(key) != filename):
                    raise ValueError(f"Invalid Gemma shard index entry: {key}")
                tensor = handle.get_tensor(key)
                if tensor.is_floating_point() and tensor.dtype != torch_dtype:
                    return None
                state[key] = tensor
    if weight_map is not None and state.keys() != weight_map.keys():
        raise ValueError("Gemma shard index has missing tensors")
    from accelerate import init_empty_weights
    from transformers import Gemma3Config, Gemma3ForConditionalGeneration

    config = Gemma3Config.from_pretrained(str(directory))
    with init_empty_weights():
        model = Gemma3ForConditionalGeneration(config)
    model.to(dtype=torch_dtype)
    model.tie_weights()
    _assign_weights(model, state)
    return model.eval()
