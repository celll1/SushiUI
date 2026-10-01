"""Dual-stream Ming conditioning with explicit training ownership."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from diffusers.models.normalization import RMSNorm


@dataclass
class MingConditioning:
    queries: torch.Tensor
    direct: torch.Tensor

    def zero(self) -> "MingConditioning":
        return MingConditioning(torch.zeros_like(self.queries), torch.zeros_like(self.direct))

    def detached(self) -> "MingConditioning":
        return MingConditioning(self.queries.detach(), self.direct.detach())


def apply_cfg_null_conditioning(conditioning: MingConditioning, drop_mask: torch.Tensor):
    if (conditioning.queries.ndim != 3 or conditioning.direct.ndim != 3
            or conditioning.queries.shape[0] != conditioning.direct.shape[0]
            or drop_mask.dtype != torch.bool or drop_mask.shape != (conditioning.queries.shape[0],)):
        raise ValueError("Ming CFG dropout requires dual batched streams and a boolean [B] mask")
    def rewrite(value):
        mask = drop_mask.to(device=value.device).reshape(-1, 1, 1)
        return torch.where(mask, torch.zeros_like(value), value)
    return MingConditioning(rewrite(conditioning.queries), rewrite(conditioning.direct))


def create_projection_modules(config: dict, language_width: int, connector_width: int):
    modules = nn.Module()
    modules.query_tokens_dict = nn.ParameterDict({
        f"{scale}x{scale}": nn.Parameter(torch.empty(scale * scale, language_width))
        for scale in config["img_gen_scales"]})
    modules.proj_in = nn.Linear(language_width, connector_width)
    modules.proj_out = nn.Linear(connector_width, config["diffusion_c_input_dim"])
    width = language_width * len(config["selected_hidden_states_layers"])
    modules.proj_directvlm = nn.Sequential(
        RMSNorm(width, eps=1e-5), nn.Linear(width, config["diffusion_inner_dim"], bias=True))
    return modules


class MingConditioner(nn.Module):
    """Own the native MLLM, connector and projections as selectable groups."""

    def __init__(self, native, connector, config: dict, *, projections=None):
        super().__init__()
        self.native = native
        native.img_gen_scales = config["img_gen_scales"]
        native.connector_norm = config["connector_norm"]
        native.use_vlm_directvlm_condition = config["use_vlm_directvlm_condition"]
        native.use_learnable_token_condition = config["use_learnable_token_condition"]
        native.selected_hidden_states_layers = config["selected_hidden_states_layers"]
        native.diffusion_inner_dim = config["diffusion_inner_dim"]
        native.scale_indices = []
        length = 0
        for scale in native.img_gen_scales:
            length += scale * scale
            native.scale_indices.append(length)
        if projections is None:
            projections = create_projection_modules(config, native.config.llm_config.hidden_size, connector.config.hidden_size)
        native.query_tokens_dict = projections.query_tokens_dict
        native.connector = connector
        for layer in connector.model.layers:
            layer.self_attn.is_causal = False
        for name in ("proj_in", "proj_out", "proj_directvlm"):
            setattr(native, name, getattr(projections, name))
        native.loaded_image_gen_modules = True
        native._ming_live_conditioning = False

    def _apply(self, fn, recurse=True):
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
        from .vendor.modeling_bailing_moe_v2 import BailingMoeV2RotaryEmbedding, BailingMoeV2RotaryEmbedding3D
        from .vendor.qwen2_5_vit import Qwen2_5_VisionRotaryEmbedding

        types = (Qwen2RotaryEmbedding, BailingMoeV2RotaryEmbedding,
                 BailingMoeV2RotaryEmbedding3D, Qwen2_5_VisionRotaryEmbedding)
        saved = [(module, name, value) for module in self.modules() if isinstance(module, types)
                 for name in ("inv_freq", "original_inv_freq")
                 if isinstance(value := getattr(module, name, None), torch.Tensor) and not value.is_meta]
        result = super()._apply(fn, recurse=recurse)
        # Retain the unrounded frequencies across precision and residency changes.
        for module, name, value in saved:
            setattr(module, name, value.to(device=module.inv_freq.device, dtype=torch.float32))
        return result

    def component_groups(self) -> dict[str, tuple[nn.Module, ...]]:
        return {"language": (self.native.model,), "vision": (self.native.vision, self.native.linear_proj),
                "connector": (self.native.connector,), "queries": (self.native.query_tokens_dict,),
                "projections": (self.native.proj_in, self.native.proj_out, self.native.proj_directvlm)}

    def set_trainable(self, groups: set[str]) -> None:
        components = self.component_groups()
        unknown = groups - components.keys()
        if unknown:
            raise ValueError(f"Unknown Ming conditioning groups: {sorted(unknown)}")
        self.requires_grad_(False)
        for name in groups:
            for module in components[name]:
                module.requires_grad_(True)
        self.native._ming_live_conditioning = bool(groups & {"language", "vision", "queries"})

    @property
    def cacheable(self) -> bool:
        return not any(parameter.requires_grad for parameter in self.parameters())

    def projection_state_dict(self) -> dict[str, torch.Tensor]:
        prefixes = ("query_tokens_dict.", "proj_in.", "proj_out.", "proj_directvlm.")
        return {key: value for key, value in self.native.state_dict().items() if key.startswith(prefixes)}

    def load_projection_state(self, state: dict[str, torch.Tensor]) -> None:
        expected = self.projection_state_dict()
        if set(state) != set(expected):
            raise ValueError(f"Ming projection state mismatch: missing={sorted(set(expected)-set(state))}, "
                             f"unexpected={sorted(set(state)-set(expected))}")
        for key, value in state.items():
            if value.shape != expected[key].shape:
                raise ValueError(f"Ming projection shape mismatch: {key}")
        # Partial loading is safe only after the complete projection key/shape check.
        self.native.load_state_dict(state, strict=False, assign=True)

    def forward(self, *, input_ids, attention_mask, pixel_values=None, image_grid_thw=None):
        image_embeds = None
        if pixel_values is not None:
            if image_grid_thw is None:
                raise ValueError("Ming vision conditioning requires image_grid_thw")
            image_embeds = self.native.extract_image_feature(pixel_values, image_grid_thw)
        queries, direct = self.native.get_condition_embeds_for_image_gen(
            input_ids=input_ids, attention_mask=attention_mask, image_embeds=image_embeds,
            position_ids=None, use_cache=False, image_grid_thw=image_grid_thw,
            llm_hidden_states=None)
        if queries is None or direct is None:
            raise ValueError("Ming 0.1 requires both released conditioning streams")
        # The native CUDA query connector autocasts; transport uses the module dtype.
        dtype = next(self.parameters()).dtype
        return MingConditioning(queries.to(dtype=dtype), direct.to(dtype=dtype))
