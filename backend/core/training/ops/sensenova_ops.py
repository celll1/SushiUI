"""Training-only SenseNova decoder operations.

The trainer supplies one immutable prompt prefix per physical B1 batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class SenseNovaTrainingPrefix:
    """Detached prompt K/V reused by every flow step for one sample."""

    cache: Any
    text_length: int


_SENSENOVA_QUANT_LINEAR_COUNT = 588


def _quantized_linear_flavours() -> "dict[str, type]":
    """The EXACT quantized-Linear classes this guard knows how to census.

    ``ConvRotInt8Linear`` subclasses ``Int8Linear``, so the census below keys on
    ``type(m) is cls``, never ``isinstance``: an isinstance census would fold a
    ConvRot base into the plain-int8 count and accept it silently.
    """
    from core.models.common.convrot_int8_linear import ConvRotInt8Linear
    from core.models.common.w4a8_linear import W4A8Linear
    from core.models.ideogram4.vendor.fp8_linear import Fp8Linear
    from core.models.ideogram4.vendor.int8_linear import Int8Linear

    return {
        "Int8Linear": Int8Linear,
        "ConvRotInt8Linear": ConvRotInt8Linear,
        "Fp8Linear": Fp8Linear,
        "W4A8Linear": W4A8Linear,
    }


def _assert_supported_quantized_training_base(transformer: nn.Module) -> None:
    """Require all 588 decoder Linears to be ONE supported quantized flavour.

    Accepts the plain-int8 and the ConvRot-int8 checkpoints (each quantizes all
    588). Refuses a mixed base, an off-count base, an unrecognized subclass of a
    known quantized Linear, and an unquantized bf16 base. Fp8/W4A8 are censused
    for the diagnostic but not accepted: no such SenseNova base exists, so
    accepting one would ship an untested path.
    """
    flavours = _quantized_linear_flavours()
    known = tuple(flavours.values())
    counts = {label: 0 for label in flavours}
    unknown: "dict[str, int]" = {}
    for module in transformer.modules():
        if not isinstance(module, known):
            continue
        for label, cls in flavours.items():
            if type(module) is cls:
                counts[label] += 1
                break
        else:
            # A quantized class added later must refuse loudly here rather than
            # be counted as whichever known class it happens to subclass.
            name = type(module).__name__
            unknown[name] = unknown.get(name, 0) + 1

    accepted = ("Int8Linear", "ConvRotInt8Linear")
    present = [label for label, n in counts.items() if n]
    if (
        unknown
        or len(present) != 1
        or present[0] not in accepted
        or counts[present[0]] != _SENSENOVA_QUANT_LINEAR_COUNT
    ):
        census = ", ".join(
            f"{label}={n}" for label, n in list(counts.items()) + list(unknown.items())
        )
        raise RuntimeError(
            "SenseNova training requires a base whose "
            f"{_SENSENOVA_QUANT_LINEAR_COUNT} decoder Linears are all ONE supported "
            f"quantized flavour (all Int8Linear, or all ConvRotInt8Linear); got {census}. "
            "A mixed or partially quantized base is refused, and so is an unquantized "
            "bf16 base -- no bf16 SenseNova checkpoint exists for this repo to train on yet."
        )


def _assert_pixel_head_fm_decoder(transformer: nn.Module) -> None:
    """Require the vendor ``use_pixel_head`` fm-head branch.

    ``train_step`` inlines only that branch of ``_t2i_predict_v``: it feeds the
    fm_head a ``b c h w`` map and un-patchifies the ``b 3 H W`` result. The other
    two vendor branches take token-shaped input -- ``use_deep_fm_head`` also
    takes a second ``t`` argument -- and neither is implemented here, so refuse
    rather than reshape into a head that cannot accept it. A missing attribute
    means an unknown tree and is refused for the same reason.
    """
    missing = [
        name
        for name in ("use_pixel_head", "use_deep_fm_head")
        if not hasattr(transformer, name)
    ]
    if missing:
        raise RuntimeError(
            "SenseNova training requires a vendor transformer exposing "
            f"use_pixel_head and use_deep_fm_head; this tree is missing "
            f"{', '.join(missing)}, so the fm-head layout it was built with is "
            "unknown and cannot be assumed to be the pixel-head (ConvDecoder) one "
            "that train_step implements."
        )
    if transformer.use_deep_fm_head:
        raise RuntimeError(
            "SenseNova training does not implement the vendor _t2i_predict_v "
            "use_deep_fm_head branch (FlowMatchingHead called as fm_head(x, t) on "
            "token-shaped input); this checkpoint has fm_head_layers > 2. Only the "
            "use_pixel_head (ConvDecoder) branch is implemented."
        )
    if not transformer.use_pixel_head:
        raise RuntimeError(
            "SenseNova training does not implement the vendor _t2i_predict_v plain "
            "fm_head branch (nn.Sequential called on token-shaped input); this "
            f"checkpoint has use_pixel_head={transformer.use_pixel_head!r}. Only the "
            "use_pixel_head (ConvDecoder) branch is implemented."
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
    _assert_supported_quantized_training_base(trainer.transformer)
    _assert_pixel_head_fm_decoder(trainer.transformer)
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
    phase_evictor = getattr(trainer, "sensenova_phase_evictor", None)
    if phase_evictor is not None:
        phase_evictor.enter_prefix()
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
    if phase_evictor is not None:
        phase_evictor.enter_denoise()
        phase_evictor.assert_generation_resident()
    return SenseNovaTrainingPrefix(cache=cache, text_length=int(input_ids.shape[1]))


def vae_encode(trainer: Any, image_tensor: torch.Tensor, **_: Any) -> torch.Tensor:
    """Return normalized RGB directly; SenseNova is a pixel-space model."""
    if image_tensor.ndim != 4 or image_tensor.shape[1] != 3:
        raise ValueError("SenseNova expects BCHW RGB training images")
    if image_tensor.shape[-2] % 32 or image_tensor.shape[-1] % 32:
        raise ValueError("SenseNova image height and width must be divisible by 32")
    return image_tensor.detach().to(dtype=trainer.training_dtype, device="cpu")


def _save_pixel_debug(
    transformer: Any,
    debug_save_path: Path,
    *,
    t_val: float,
    noise_scale: float,
    images: torch.Tensor,
    z_image: torch.Tensor,
    x0_pred_tokens: torch.Tensor,
    patch: int,
    height: int,
    width: int,
    loss_value: float,
    recon_loss_value: float,
    captions: Optional[List[str]],
    reference_image_paths: Optional[List[Optional[str]]],
) -> None:
    """Dump this step's pixel tensors, the pixel-space analogue of the latent
    archs' debug latents: ``target`` is their ``latents`` (the clean sample),
    ``noisy`` their ``noisy_latents``, ``pred_x0`` their ``predicted_latent``.

    SenseNova's "latent" already IS [-1,1] RGB, so the previews are written
    directly and the ``.pt`` stays scalar-only (the visualize endpoint prefers
    the webp over false-colouring a tensor, and a full-res pixel tensor per
    dump would be tens of MB).
    """
    from core.models.sensenova.sensenova_pipeline_ops import tensor_to_image

    debug_save_path.mkdir(parents=True, exist_ok=True)
    debug_data: dict = {
        "timestep": t_val,
        "noise_scale": noise_scale,
        "model_type": "sensenova",
        "is_latent": False,
        "loss": loss_value,
        "recon_loss": recon_loss_value,
        "batch_size": 1,
    }
    if captions:
        debug_data["caption"] = captions[0]
    if reference_image_paths:
        first_ref = next((p for p in reference_image_paths if p is not None), None)
        if first_ref:
            debug_data["reference_image_path"] = first_ref
    torch.save(debug_data, debug_save_path / f"latents_t{t_val:.4f}.pt")

    x0_pred_image = transformer.unpatchify(x0_pred_tokens.detach(), patch, height, width)
    for name, tensor in (
        ("noisy", z_image),
        ("target", images),
        ("pred_x0", x0_pred_image),
    ):
        # tensor_to_image clamps to [-1,1]: the noised map saturates at low t,
        # which is the same convention the VAE archs' decoded previews use.
        tensor_to_image(tensor.detach().float()).save(
            debug_save_path / f"decode_t{t_val:.4f}_{name}.webp",
            "WEBP",
            quality=80,
            method=4,
        )


def train_step(
    trainer: Any,
    *,
    images: torch.Tensor,
    prefix: SenseNovaTrainingPrefix,
    timesteps: Optional[torch.Tensor] = None,
    profile_vram: bool = False,
    debug_save_path: Optional[Path] = None,
    debug_captions: Optional[List[str]] = None,
    debug_reference_image_paths: Optional[List[Optional[str]]] = None,
) -> tuple[torch.Tensor, float, float]:
    """Run one B1 pixel-space flow-matching forward pass."""
    del profile_vram  # Central profiling owns peak-memory reporting.
    if not isinstance(prefix, SenseNovaTrainingPrefix):
        raise TypeError("SenseNova train_step requires SenseNovaTrainingPrefix")
    phase_evictor = getattr(trainer, "sensenova_phase_evictor", None)
    if phase_evictor is not None:
        phase_evictor.enter_denoise()
        phase_evictor.assert_generation_resident()
    if images.ndim != 4 or images.shape[0] != 1 or images.shape[1] != 3:
        raise ValueError("SenseNova training currently requires batch_size=1 BCHW RGB")
    height, width = images.shape[-2:]
    if height % 32 or width % 32:
        raise ValueError("SenseNova image height and width must be divisible by 32")

    transformer = trainer.transformer
    _assert_pixel_head_fm_decoder(transformer)
    device = trainer.device
    dtype = trainer.training_dtype
    x0 = images.to(device=device, dtype=dtype)
    if timesteps is None:
        t = trainer.timestep_sampler.sample(1, device=device)
        if isinstance(t, tuple):
            t = t[0]
    else:
        t = timesteps
    # Keep t in fp32, the dtype the sampler produces and the dtype inference's
    # `ts` carries (linspace, sensenova_pipeline_ops.py:1068): timestep_embedder
    # embeds t's VALUE, and bf16 would quantize it to ~2e-3 in training only.
    t = torch.as_tensor(t, device=device, dtype=torch.float32).reshape(-1)
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
    # Inference noises in the image dtype, its fp32 t demoted by 0-dim promotion
    # (sensenova_pipeline_ops.py:1122); cast explicitly so z_image stays
    # training_dtype -- _build_step_context's ViT runs outside the autocast below.
    z_image = t.to(dtype).view(1, 1, 1, 1) * x0 + (1 - t).to(dtype).view(1, 1, 1, 1) * (
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
        # fp32 t here lifts v into fp32, which the MSE below wanted anyway --
        # the .float() calls become no-ops rather than extra copies.
        denominator = (1 - t).view(1, 1, 1).clamp_min(transformer.config.t_eps)
        v_pred = (x0_pred - z) / denominator
        v_target = (x0_tokens - z) / denominator
        loss = torch.nn.functional.mse_loss(v_pred.float(), v_target.float())
        recon_loss = torch.nn.functional.mse_loss(x0_pred.float(), x0_tokens.float())

    value = float(loss.detach())
    recon_value = float(recon_loss.detach())

    if debug_save_path is not None:
        try:
            _save_pixel_debug(
                transformer,
                debug_save_path,
                t_val=float(t[0].item()),
                noise_scale=noise_scale,
                images=x0,
                z_image=z_image,
                x0_pred_tokens=x0_pred,
                patch=patch,
                height=height,
                width=width,
                loss_value=value,
                recon_loss_value=recon_value,
                captions=debug_captions,
                reference_image_paths=debug_reference_image_paths,
            )
        except Exception as debug_error:
            print(f"{trainer.log_prefix} [debug_latents] save failed: {debug_error}")

    return loss, value, recon_value


def generate_sample(
    trainer: Any,
    *,
    prompt: str,
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    negative_prompt: str = "",
    reference_image_path: Optional[str] = None,
    condition_image_path: Optional[str] = None,
):
    """Run one inference txt2img generation from inside the training loop.

    Drives the SAME ``sensenova_pipeline_ops`` prefix + Euler loop generation
    uses; nothing about the denoise is reimplemented here. The LoRA under
    training is applied automatically because its ``LoRALinearLayer`` wrappers
    ARE the live modules the generation forward calls.

    Returns a PIL image, or ``None`` if the generation failed -- the training
    loop's sample block has no exception guard of its own, so a failed sample
    must never take the run down.
    """
    from api.param_defaults import SENSENOVA_GENERATION_DEFAULTS
    from core.attention import AttentionMode
    from core.models.sensenova import sensenova_pipeline_ops as ops

    transformer = trainer.transformer
    if reference_image_path or condition_image_path:
        print(
            f"{trainer.log_prefix} SenseNova sampling ignores reference/condition "
            f"images (reference-conditioned training is deferred to Phase 3)"
        )

    snapped_width, snapped_height = ops.normalize_resolution(width, height)
    if (snapped_width, snapped_height) != (width, height):
        print(
            f"{trainer.log_prefix} SenseNova sample resolution snapped to the "
            f"{ops.TOKEN_GRID_ALIGN}px token grid: {width}x{height} -> "
            f"{snapped_width}x{snapped_height}"
        )

    backend = trainer._resolve_training_backend(trainer.attention_backend)
    was_training = transformer.training
    evictor = getattr(trainer, "sensenova_phase_evictor", None)
    prefix = None
    try:
        # No-op while the phase evictor owns weight placement.
        trainer.move_main_model_to_gpu()
        transformer.eval()
        # Pass the mode EXPLICITLY: set_attention_backend infers it from
        # torch.is_grad_enabled() otherwise, and this call happens before the
        # no_grad block below.
        ops.set_attention_backend(transformer, backend, AttentionMode.INFERENCE)
        with torch.no_grad():
            # The evictor's full/prefix/denoise machine is driven here exactly as
            # a training step drives it, so generation's own prefix->denoise
            # phase change stays the SAME transition pair and the two halves
            # never co-reside.
            if evictor is not None:
                evictor.enter_prefix()
            prefix = ops.encode_prompt(
                transformer,
                trainer.tokenizer,
                prompt,
                snapped_height,
                snapped_width,
                guidance_scale,
                negative_prompt=negative_prompt,
            )
            if evictor is not None:
                evictor.enter_denoise()
                evictor.assert_generation_resident()
            image_tensor = ops.denoise_loop(
                transformer,
                prefix,
                cfg_scale=guidance_scale,
                timestep_shift=SENSENOVA_GENERATION_DEFAULTS["timestep_shift"],
                num_inference_steps=num_inference_steps,
                seed=seed if seed is not None and seed >= 0 else None,
                cfg_norm=SENSENOVA_GENERATION_DEFAULTS["cfg_norm"],
            )
        image = ops.tensor_to_image(image_tensor.float())
        del image_tensor
        return image
    except Exception as sample_error:
        import traceback

        print(
            f"{trainer.log_prefix} SenseNova sample generation failed "
            f"({type(sample_error).__name__}: {sample_error}); training continues"
        )
        traceback.print_exc()
        return None
    finally:
        if prefix is not None:
            try:
                ops.clear_prefix_caches(prefix)
            except Exception as clear_error:
                print(
                    f"{trainer.log_prefix} SenseNova sample prefix cleanup failed: {clear_error}"
                )
        # Restore TRAINING mode before the next forward: nothing re-stamps the
        # attention modules after load, so an INFERENCE stamp left here would
        # persist for the rest of the run.
        ops.set_attention_backend(transformer, backend, AttentionMode.TRAINING)
        if was_training:
            transformer.train()
        # The evictor is deliberately left wherever this call ended: both
        # "prefix" (sample raised mid-way) and "denoise" are states the next
        # step's encode_prompt transitions out of legally.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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
