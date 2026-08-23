"""Training-only SenseNova decoder operations.

The trainer supplies one immutable prompt prefix per physical B1 batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class SenseNovaTrainingPrefix:
    """Detached prompt K/V reused by every flow step for one sample."""

    cache: Any
    text_length: int


def _assert_plain_int8_training_base(transformer: nn.Module) -> None:
    from core.models.common.convrot_int8_linear import ConvRotInt8Linear
    from core.models.ideogram4.vendor.int8_linear import Int8Linear

    plain_int8 = sum(type(module) is Int8Linear for module in transformer.modules())
    convrot = sum(
        isinstance(module, ConvRotInt8Linear) for module in transformer.modules()
    )
    if plain_int8 != 588 or convrot != 0:
        raise RuntimeError(
            "SenseNova training currently requires the plain-int8 checkpoint "
            f"(plain Int8Linear={plain_int8}, ConvRotInt8Linear={convrot}; "
            "expected 588 and 0)"
        )


def setup_attention_backend(trainer: Any, backend: str) -> None:
    from core.attention import AttentionMode
    from core.models.sensenova.sensenova_pipeline_ops import set_attention_backend

    resolved = trainer._resolve_training_backend(backend)
    count = set_attention_backend(trainer.transformer, resolved, AttentionMode.TRAINING)
    expected = len(trainer.transformer.language_model.model.layers)
    if count != expected:
        raise RuntimeError(
            f"SenseNova configured {count} attention module(s), expected {expected}"
        )


def load_components(trainer: Any) -> None:
    """Load the frozen SenseNova graph; the adapter owns trainable parameters."""
    if getattr(trainer, "blocks_to_swap", 0) != 0:
        raise ValueError("SenseNova training does not implement blocks_to_swap; set it to 0")
    from core.models.sensenova.loader import load_sensenova_from_path

    components = load_sensenova_from_path(trainer.model_path, torch_dtype=trainer.weight_dtype)
    trainer.transformer = components["transformer"]
    _assert_plain_int8_training_base(trainer.transformer)
    trainer.transformer_original = trainer.transformer
    trainer.transformer_uncond = None
    trainer.tokenizer = components["tokenizer"]
    trainer.sensenova_model_config = components.get("config")
    trainer.text_encoder = None
    trainer.text_encoder_2 = None
    trainer.tokenizer_2 = None
    trainer.t5_tokenizer = None
    trainer.vae = None
    trainer.unet = None
    trainer.scheduler = None
    trainer.noise_scheduler = None
    trainer.layer_offload_conductor = None
    trainer.transformer.requires_grad_(False)
    trainer.transformer.train()
    trainer.transformer.to(trainer.device)
    # Training mode must be stamped even when the selected backend is native.
    setup_attention_backend(trainer, trainer.attention_backend)


def encode_prompt(
    trainer: Any, prompt: str, *, requires_grad: bool = False
) -> SenseNovaTrainingPrefix:
    """Build a detached prefix without inference streamers or flash buffers."""
    if requires_grad:
        raise ValueError("SenseNova text-encoder training is not supported")
    if not isinstance(prompt, str):
        raise TypeError("SenseNova training encodes one prompt at a time")

    from core.models.sensenova.vendor.utils import SYSTEM_MESSAGE_FOR_GEN

    transformer = trainer.transformer
    query = transformer._build_t2i_query(
        prompt,
        system_message=SYSTEM_MESSAGE_FOR_GEN,
        append_text="<think>\n\n</think>\n\n<img>",
    )
    with torch.no_grad():
        input_ids, indexes, attention_mask = transformer._build_t2i_text_inputs(
            trainer.tokenizer, query
        )
        cache, _ = transformer._t2i_prefix_forward(input_ids, indexes, attention_mask)
    expected_layers = len(transformer.language_model.model.layers)
    _assert_immutable_prefix_cache(cache, expected_layers)
    return SenseNovaTrainingPrefix(cache=cache, text_length=int(input_ids.shape[1]))


def vae_encode(trainer: Any, image_tensor: torch.Tensor, **_: Any) -> torch.Tensor:
    """Return normalized RGB directly; SenseNova is a pixel-space model."""
    if image_tensor.ndim != 4 or image_tensor.shape[1] != 3:
        raise ValueError("SenseNova expects BCHW RGB training images")
    if image_tensor.shape[-2] % 32 or image_tensor.shape[-1] % 32:
        raise ValueError("SenseNova image height and width must be divisible by 32")
    return image_tensor.detach().to(dtype=trainer.training_dtype, device="cpu")


def train_step(
    trainer: Any,
    *,
    images: torch.Tensor,
    prefix: SenseNovaTrainingPrefix,
    timesteps: Optional[torch.Tensor] = None,
    profile_vram: bool = False,
) -> tuple[torch.Tensor, float, float]:
    """Run one B1 pixel-space flow-matching forward pass."""
    del profile_vram  # Central profiling owns peak-memory reporting.
    if not isinstance(prefix, SenseNovaTrainingPrefix):
        raise TypeError("SenseNova train_step requires SenseNovaTrainingPrefix")
    if images.ndim != 4 or images.shape[0] != 1 or images.shape[1] != 3:
        raise ValueError("SenseNova training currently requires batch_size=1 BCHW RGB")
    height, width = images.shape[-2:]
    if height % 32 or width % 32:
        raise ValueError("SenseNova image height and width must be divisible by 32")

    transformer = trainer.transformer
    device = trainer.device
    dtype = trainer.training_dtype
    x0 = images.to(device=device, dtype=dtype)
    if timesteps is None:
        t = trainer.timestep_sampler.sample(1, device=device)
        if isinstance(t, tuple):
            t = t[0]
    else:
        t = timesteps
    t = torch.as_tensor(t, device=device, dtype=dtype).reshape(-1)
    if t.numel() != 1:
        raise ValueError("SenseNova training requires one timestep for batch_size=1")

    from core.models.sensenova.sensenova_pipeline_ops import (
        _build_step_context,
        compute_noise_scale,
    )

    merge_size = int(1 / transformer.downsample_ratio)
    grid_h, grid_w = height // transformer.patch_size, width // transformer.patch_size
    if grid_h % merge_size or grid_w % merge_size:
        raise ValueError("SenseNova image does not align to the merged token grid")
    token_h, token_w = grid_h // merge_size, grid_w // merge_size
    noise_scale = compute_noise_scale(transformer, grid_h, grid_w, merge_size)
    z_image = t.view(1, 1, 1, 1) * x0 + (1 - t).view(1, 1, 1, 1) * (
        torch.randn_like(x0) * noise_scale
    )
    shape = SimpleNamespace(
        batch_size=1,
        merge_size=merge_size,
        grid_h=grid_h,
        grid_w=grid_w,
        token_h=token_h,
        token_w=token_w,
    )
    z, image_embeds, _ = _build_step_context(
        transformer, shape, z_image, t[0], noise_scale
    )
    indexes = transformer._build_t2i_image_indexes(
        token_h, token_w, prefix.text_length, device=device
    )
    _assert_immutable_prefix_cache(
        prefix.cache, len(transformer.language_model.model.layers)
    )

    device_type = torch.device(device).type
    autocast_enabled = device_type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    with torch.autocast(device_type=device_type, dtype=dtype, enabled=autocast_enabled):
        hidden = forward_gen_decoder_layers(
            transformer.language_model.model,
            image_embeds,
            indexes=indexes,
            prefix_cache=prefix.cache,
            checkpoint_layers=bool(trainer.gradient_checkpointing),
        )
        decoded = transformer.fm_modules["fm_head"](
            hidden.view(1, token_h, token_w, -1).permute(0, 3, 1, 2).contiguous()
        )
        patch = transformer.patch_size * merge_size
        x0_pred = (
            decoded.view(1, 3, token_h, patch, token_w, patch)
            .permute(0, 2, 4, 3, 5, 1)
            .contiguous()
            .view(1, token_h * token_w, patch * patch * 3)
        )
        x0_tokens = transformer.patchify(x0, patch)
        denominator = (1 - t).view(1, 1, 1).clamp_min(transformer.config.t_eps)
        v_pred = (x0_pred - z) / denominator
        v_target = (x0_tokens - z) / denominator
        loss = torch.nn.functional.mse_loss(v_pred.float(), v_target.float())
        recon_loss = torch.nn.functional.mse_loss(x0_pred.float(), x0_tokens.float())

    value = float(loss.detach())
    return loss, value, float(recon_loss.detach())


def _assert_immutable_prefix_cache(prefix_cache: Any, expected_layers: int) -> None:
    if prefix_cache is None:
        raise ValueError("SenseNova generation training requires a prefix KV cache")
    layers = getattr(prefix_cache, "layers", None)
    if layers is None or len(layers) == 0:
        raise ValueError("SenseNova training requires non-empty prefix KV cache layers")
    if len(layers) != expected_layers:
        raise ValueError(
            f"SenseNova prefix KV cache has {len(layers)} layer(s), expected {expected_layers}"
        )
    if any(
        getattr(prefix_cache, name, None) is not None
        for name in ("_kv_cache_streamer", "_kv_cache_streamer_branch")
    ):
        raise ValueError("SenseNova training cannot use the inference KV cache streamer")

    for layer in layers:
        for name in ("keys", "values"):
            tensor = getattr(layer, name, None)
            if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
                raise ValueError(f"SenseNova prefix KV cache layer is missing non-empty {name}")
            if tensor.requires_grad:
                raise ValueError(f"SenseNova prefix KV cache {name} tensors must be detached")
        if any(
            getattr(layer, name, None) is not None
            for name in ("flash_k_cache", "flash_v_cache")
        ):
            raise ValueError("SenseNova training cannot use prepared inference flash KV buffers")


def forward_gen_decoder_layers(
    model: Any,
    hidden_states: torch.Tensor,
    *,
    indexes: torch.Tensor,
    prefix_cache: Any,
    attention_mask: Optional[torch.Tensor] = None,
    checkpoint_layers: bool = False,
) -> torch.Tensor:
    """Run the all-generation-token Qwen3 decoder against immutable prefix K/V.

    Calling PyTorch's base ``Module.__call__`` bypasses Transformers'
    cache-dropping checkpoint wrapper while preserving module hooks. The cache
    is read through the differentiable ``update_cache=False`` concat path.
    """

    layers = getattr(model, "layers", None)
    if layers is None:
        raise ValueError("SenseNova generation training model has no decoder layers")
    _assert_immutable_prefix_cache(prefix_cache, len(layers))
    image_gen_indicators = torch.ones(
        hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device
    )

    for layer in layers:
        def layer_forward(states: torch.Tensor, _layer=layer) -> torch.Tensor:
            # Skip only Transformers' cache-dropping wrapper; keep Module hooks.
            return nn.Module.__call__(
                _layer,
                states,
                image_gen_indicators=image_gen_indicators,
                exist_non_image_gen_tokens=False,
                exist_image_gen_tokens=True,
                indexes=indexes,
                attention_mask=attention_mask,
                past_key_values=prefix_cache,
                use_cache=False,
                update_cache=False,
            )

        if checkpoint_layers:
            hidden_states = checkpoint(layer_forward, hidden_states, use_reentrant=False)
        else:
            hidden_states = layer_forward(hidden_states)

    return model.norm_mot_gen(hidden_states)
