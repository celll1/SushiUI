"""Standalone generation ops for SenseNova-U1.5-8B-MoT (pixel-space, no VAE,
Qwen3-as-flow-matching-denoiser).

Reimplements ``NEOChatModel.t2i_generate`` (kept as the upstream REFERENCE in
``vendor/modeling_neo_chat.py``, ~line 1707) as SushiUI's own denoise loop,
driving the model's own helper methods (``patchify``/``unpatchify``,
``extract_feature``, ``_t2i_predict_v``, ``_apply_time_schedule``,
``_build_t2i_query``/``_build_t2i_text_inputs``/``_build_t2i_image_indexes``,
``_t2i_prefix_forward``, ``prepare_flash_kv_cache``/``clear_flash_kv_cache``)
rather than re-deriving their internals.

Flow matching, x0-parameterized: ``t=0`` is pure noise, ``t=1`` is the clean
image (the OPPOSITE of flux2's sigma direction -- do not copy flux2's
img2img/inpaint sign). Euler step is forward in t:
``v = (x0_pred - z) / (1-t).clamp_min(t_eps)``; ``z = z + (t_next-t) * v``.
See the SENSENOVA_FACTS scratchpad note for the full verified derivation;
this module implements it, it does not re-argue it.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageFilter

from core.inference.cancellation import raise_if_cancelled
from core.inference.generation_timing import time_phase

from .vendor.modeling_neo_chat import clear_flash_kv_cache, optimized_scale, prepare_flash_kv_cache
from .vendor.utils import SYSTEM_MESSAGE_FOR_GEN, load_image_native

LABEL = "SenseNova"
TOKEN_GRID_ALIGN = 32  # patch_size(16) * merge_size(2) -- the token patch, not the raw ViT patch_size.

# cfg_scale/timestep_shift/num_inference_steps have NO module-level default
# (AGENTS.md: never hardcode a default outside backend/api/param_defaults.py)
# -- every real caller (core/pipeline_backends/sensenova.py) already sources
# them from SENSENOVA_GENERATION_DEFAULTS and passes them explicitly, h3-style
# (core/models/minimax_h3/h3_pipeline_ops.py's denoise() takes no default for
# any tunable generation param either). cfg_norm/cfg_interval are NOT yet
# surfaced through param_defaults.py by any caller (only steps/cfg_scale/
# timestep_shift are), so they keep a neutral "CFG behaves exactly like the
# vanilla two-branch blend, no norm rescaling, active every step" default
# here rather than becoming a required arg that would break the one live
# caller -- see this module's audit note for the follow-up this implies.
DEFAULT_CFG_NORM = "none"
DEFAULT_CFG_INTERVAL = (0.0, 1.0)

# Reference-image (it2i) token markers -- upstream's IMG_START_TOKEN/
# IMG_END_TOKEN/IMG_CONTEXT_TOKEN defaults (modeling_neo_chat.py:1386).
_IMG_START_TOKEN = "<img>"
_IMG_END_TOKEN = "</img>"
_IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

# Cost knob, not a correctness constant: caps a reference image's encode
# resolution (upstream has no such cap, only min(2048*2048, (4096*4096)//n));
# a measurement gate may retune this.
REFERENCE_IMAGE_MAX_PIXELS_CAP = 1024 * 1024


def align_to_grid(value: int, align: int = TOKEN_GRID_ALIGN) -> int:
    if value <= 0:
        return align
    return max(align, round(value / align) * align)


def normalize_resolution(width: int, height: int) -> Tuple[int, int]:
    """Snap to the 32px token grid -- SenseNova refuses nothing, callers round instead."""
    return align_to_grid(width), align_to_grid(height)


def image_to_tensor(image: Image.Image, height: int, width: int, device, dtype) -> torch.Tensor:
    """PIL -> normalized [-1,1] RGB tensor [1,3,H,W] (NORM_MEAN=NORM_STD=0.5)."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((width, height), Image.LANCZOS)
    arr = np.asarray(image).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)


def tensor_to_image(x: torch.Tensor) -> Image.Image:
    """[-1,1] RGB tensor [1,3,H,W] -> PIL. Matches upstream's ``(x*0.5+0.5).clamp(0,1)`` decode."""
    x = x.clamp(-1, 1)
    arr = (x[0] * 127.5 + 128.0).clamp(0, 255).permute(1, 2, 0).to(device="cpu", dtype=torch.uint8).numpy()
    return Image.fromarray(arr)


def prepare_mask(mask_image: Image.Image, height: int, width: int, device, dtype,
                 mask_blur: int = 0) -> torch.Tensor:
    """PIL mask (white=inpaint) -> [1,1,H,W] in [0,1], pixel-space (RePaint blend
    happens in pixel space, see ``_euler_run``). Resize FIRST (BILINEAR), THEN
    blur -- every other inpaint path in this repo does it in this order
    (anima.py, flux2.py, ideogram4.py, krea2.py, lens.py, zimage.py); doing it
    the other way round makes the feather resolution-dependent (a mask
    authored at 512x512 with mask_blur=4 turns into ~16px of blur once
    resized up to a 2048x2048 generation)."""
    m = mask_image.convert("L")
    m = m.resize((width, height), Image.BILINEAR)
    if mask_blur and mask_blur > 0:
        m = m.filter(ImageFilter.GaussianBlur(radius=mask_blur))
    arr = np.asarray(m).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None, None].to(device=device, dtype=dtype)


def prepare_noise(height: int, width: int, device, dtype, seed: Optional[int], noise_scale: float,
                  batch_size: int = 1) -> torch.Tensor:
    gen = None
    if seed is not None and seed >= 0:
        gen = torch.Generator(device=device).manual_seed(seed)
    return noise_scale * torch.randn(batch_size, 3, height, width, generator=gen, device=device, dtype=dtype)


def compute_noise_scale(transformer, grid_h: int, grid_w: int, merge_size: int) -> float:
    """Resolution-dependent init-noise scale (facts: recomputed per request,
    ALSO fed to the model via ``noise_scale_embedder`` -- see ``_build_step_context``).
    Mirrors ``t2i_generate`` lines ~1802-1809 exactly."""
    noise_scale = transformer.noise_scale
    if transformer.noise_scale_mode in ("resolution", "dynamic", "dynamic_sqrt"):
        base = float(transformer.noise_scale_base_image_seq_len)
        scale = math.sqrt((grid_h * grid_w) / (merge_size ** 2) / base)
        noise_scale = scale * float(transformer.noise_scale)
        if transformer.noise_scale_mode == "dynamic_sqrt":
            noise_scale = math.sqrt(noise_scale)
    return min(noise_scale, transformer.noise_scale_max_value)


def set_attention_backend(transformer, backend: str = "native", mode=None) -> int:
    """Stamp ``_attn_backend``/``_attn_mode`` on every ``Qwen3Attention`` module
    (both the understanding and gen branch share the one class -- MoT only
    duplicates the Linear weights, not the module type). Unit 2's
    ``_flash_or_sdpa`` reads these off the calling instance and there is no
    other wiring point (the vendor file's module-level ``set_attn_backend`` is
    dead for this path, see that file's header comment). Returns the count
    stamped."""
    from core.attention import AttentionMode

    from .vendor.modeling_qwen3 import Qwen3Attention

    resolved_mode = mode if mode is not None else (
        AttentionMode.TRAINING if torch.is_grad_enabled() else AttentionMode.INFERENCE)
    count = 0
    for m in transformer.modules():
        if isinstance(m, Qwen3Attention):
            m._attn_backend = backend
            m._attn_mode = resolved_mode
            count += 1
    return count


@dataclass
class SenseNovaPrefix:
    """The resolution- and prompt-dependent prefix KV cache(s) for one
    generation. CFG needs two INDEPENDENT caches (cond + uncond, the latter
    from an empty string or a caller-supplied negative_prompt) -- see
    ``encode_prompt``. Built once per generation and consumed
    by every ``denoise_loop*`` below, each of which clears both caches in a
    ``finally`` (via ``clear_prefix_caches``) so a cancelled/failed run can
    never leak a cache into the next generation.

    ``encode_cfg_scale`` records the ``cfg_scale`` this prefix was BUILT with
    (i.e. whether the uncond cache exists), independent of whatever
    ``cfg_scale`` a ``denoise_loop*`` call is later given -- a mismatch
    (encode at <=1, denoise at >1) would otherwise silently drop CFG with no
    error; ``_euler_run`` checks this. ``consumed`` guards against reusing a
    single-use prefix after its caches have already been cleared.

    ``img_cond_*``/``encode_img_cfg_scale`` are the third (reference-image)
    branch from ``encode_prompt(ref_images=...)`` -- upstream's
    ``it2i_generate``'s ``past_key_values_img_condition``. Absent (``None``)
    whenever ``ref_images`` was empty/None, which is what keeps the no-refs
    ``_euler_run`` path numerically identical to before."""

    device: torch.device
    dtype: torch.dtype
    batch_size: int
    token_h: int
    token_w: int
    grid_h: int
    grid_w: int
    merge_size: int
    image_size: Tuple[int, int]  # (W, H) -- matches upstream's tuple order.
    cond_past_key_values: Any
    cond_indexes_image: torch.Tensor
    cond_attention_mask: Dict[str, Any]
    encode_cfg_scale: float = 1.0
    uncond_past_key_values: Optional[Any] = None
    uncond_indexes_image: Optional[torch.Tensor] = None
    uncond_attention_mask: Optional[Dict[str, Any]] = None
    img_cond_past_key_values: Optional[Any] = None
    img_cond_indexes_image: Optional[torch.Tensor] = None
    img_cond_attention_mask: Optional[Dict[str, Any]] = None
    encode_img_cfg_scale: float = 1.0
    # Which gate _euler_run applies. Not inferable from img_cond's presence:
    # equal scales != 1 with references build an uncond branch and no img_cond.
    has_reference_images: bool = False
    consumed: bool = False


def clear_prefix_caches(prefix: SenseNovaPrefix) -> None:
    """Idempotent (``clear_flash_kv_cache`` guards with ``hasattr``) -- safe to
    call from both an outer caller's ``finally`` and ``_euler_run``'s own."""
    clear_flash_kv_cache(prefix.cond_past_key_values)
    if prefix.uncond_past_key_values is not None:
        clear_flash_kv_cache(prefix.uncond_past_key_values)
    if prefix.img_cond_past_key_values is not None:
        clear_flash_kv_cache(prefix.img_cond_past_key_values)
    prefix.consumed = True


def _embed_reference_images(
    transformer, ref_images: Sequence[Image.Image], max_pixels_cap: int = REFERENCE_IMAGE_MAX_PIXELS_CAP,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode reference images through the vendored ``load_image_native`` (RGB/
    RGBA-flatten, ImageNet normalization, smart-resize, patchify) -- upstream
    ``it2i_generate:1401-1417``. Read through the UNDERSTANDING vision tower
    (``extract_feature(..., gen_model=False)`` inside ``_build_it2i_inputs``),
    distinct from the per-step gen-branch ViT in ``_build_step_context``.
    Returns ``(pixel_values, grid_hw)`` concatenated across all references."""
    device = transformer.device
    dtype = transformer.language_model.get_input_embeddings().weight.dtype
    n = len(ref_images)
    upstream_max_pixels = min(2048 * 2048, (4096 * 4096) // n)
    max_pixels = min(max_pixels_cap, upstream_max_pixels)
    if max_pixels < upstream_max_pixels and any(img.width * img.height > max_pixels for img in ref_images):
        # The cap is ours, not upstream's -- make the fidelity deviation visible.
        msg = (f"[{LABEL}] reference image(s) downscaled to {max_pixels / 1e6:.2f} MP for encoding "
               f"(upstream would allow {upstream_max_pixels / 1e6:.2f} MP at {n} reference(s)); this is "
               f"SushiUI's reference-encode cost cap, not a model limit.")
        print(msg)
        try:
            from api.generation_status import add_warning
            add_warning(msg, code="sensenova_reference_downscaled")
        except Exception:
            pass
    pixel_values, grid_hw = [], []
    for image in ref_images:
        raise_if_cancelled()  # up to 5 vision-tower encodes before step 0
        cur_pixel_values, cur_grid_hw = load_image_native(
            image, transformer.patch_size, transformer.downsample_ratio,
            min_pixels=512 * 512, max_pixels=max_pixels,
            upscale=False,
        )
        pixel_values.append(cur_pixel_values.to(device=device, dtype=dtype))
        grid_hw.append(cur_grid_hw.to(device))
    return torch.cat(pixel_values), torch.cat(grid_hw)


def _splice_reference_image_tokens(
    text: str, num_images: int, grid_hw: torch.Tensor, downsample_ratio: float,
) -> str:
    """Upstream ``it2i_generate:1393-1440``. Pads missing ``<image>``
    placeholders (an ``Image-N:<image>`` prefix per reference when ``text``
    has none and there is more than one reference, otherwise a bare
    ``<image>`` prefix for the shortfall), then replaces each ``<image>``
    with ``IMG_START + IMG_CONTEXT*num_patch_token + IMG_END`` for that
    reference. Reused for both the main prompt and the raw ``'<image>'*n``
    img_cond text -- the latter already has exactly ``num_images``
    placeholders, so the padding step is a no-op for it."""
    image_token_count = text.count("<image>")
    if num_images < image_token_count:
        raise ValueError(
            f"{LABEL}: prompt references {image_token_count} <image> placeholder(s) but only "
            f"{num_images} reference image(s) were given.")
    if num_images > image_token_count:
        if image_token_count == 0 and num_images > 1:
            text = "".join(f"Image-{i + 1}:<image>\n" for i in range(num_images)) + text
        else:
            text = "<image>\n" * (num_images - image_token_count) + text
    for i in range(grid_hw.shape[0]):
        num_patch_token = int(grid_hw[i, 0] * grid_hw[i, 1] * downsample_ratio ** 2)
        image_tokens = _IMG_START_TOKEN + _IMG_CONTEXT_TOKEN * num_patch_token + _IMG_END_TOKEN
        text = text.replace("<image>", image_tokens, 1)
    return text


def _finalize_prefix_caches(transformer, caches, batch_size: int, token_h: int, token_w: int) -> None:
    """Batch-expand + ``prepare_flash_kv_cache`` for every non-``None`` cache in
    ``caches`` (order: cond, img_cond, uncond) -- generalizes the original
    two-cache block (and upstream ``it2i_generate:1524-1564``) to 1-3
    branches. ``caches[0]`` (cond) is always present; its layer count is the
    shared loop bound (every branch has the same layer count)."""
    transformer._notify_layer_offload_phase("denoise")
    present = [c for c in caches if c is not None]
    for layer_idx in range(len(present[0].layers)):
        for cache in present:
            layer = cache.layers[layer_idx]
            layer.keys = layer.keys.expand(batch_size, *layer.keys.shape[1:])
            layer.values = layer.values.expand(batch_size, *layer.values.shape[1:])
    for cache in present:
        prepare_flash_kv_cache(cache, current_len=token_h * token_w, batch_size=batch_size)


@torch.no_grad()
@time_phase("text_encode")
def encode_prompt(
    transformer,
    tokenizer,
    prompt: str,
    height: int,
    width: int,
    cfg_scale: float,
    batch_size: int = 1,
    system_message: Optional[str] = None,
    prefill_callback: Optional[Callable[[], None]] = None,
    negative_prompt: Optional[str] = None,
    ref_images: Optional[Sequence[Image.Image]] = None,
    img_cfg_scale: float = 1.0,
) -> SenseNovaPrefix:
    """Build the prefix KV cache(s): the tokenizer + chat-template + prefix-
    forward stage. Resolution-DEPENDENT (the image token indexes bake in
    ``token_h``/``token_w``), so this must be re-run per request, not cached
    across generations of different size.

    ``prefill_callback`` fires once, before the (real, multi-second) prefix
    forward -- callers wire it to a distinct progress event so this stage
    doesn't read as a hang before step 0. A raising callback never aborts the
    generation.

    ``needs_cfg = cfg_scale > 1`` (upstream's own gate): a distillation-LoRA
    request at ``cfg_scale <= 1`` naturally skips the uncond prefix entirely,
    no separate mode flag needed.

    ``negative_prompt``: the uncond branch is, structurally, just a second
    call through the SAME ``_build_t2i_query``/``_build_t2i_text_inputs``/
    ``_t2i_prefix_forward`` path as the cond branch, conditioned on whatever
    string is given (upstream always uses ``""``); see MODEL_FACTS.md for the
    measured effect. ``None``/empty keeps the original empty-string uncond.

    ``ref_images``/``img_cfg_scale``: upstream's reference-image
    ``it2i_generate`` path (``vendor/modeling_neo_chat.py:1386``). When
    ``ref_images`` is empty/None, the branch below is ``if not ref_images:``,
    an UNMODIFIED copy of this function's original (pre-reference) body --
    same queries, same ``_build_t2i_text_inputs``/``_t2i_prefix_forward``,
    same ``needs_cfg = cfg_scale > 1`` uncond gate, same warning -- so that
    code path's behavior is unchanged. With references, up to three prefix
    caches are built (cond, img_cond, uncond) per upstream's
    ``needs_img_condition``/``needs_uncondition`` gates (``it2i_generate``
    lines 1422-1424); ``img_cfg_scale`` is otherwise inert. A ``negative_prompt``
    rides the uncond branch when one exists, and otherwise the img_cond branch
    (the blend's baseline at the default ``img_cfg_scale=1``) -- SushiUI's own
    extension, upstream conditions img_cond on the images alone.
    """
    if height % TOKEN_GRID_ALIGN != 0 or width % TOKEN_GRID_ALIGN != 0:
        raise ValueError(
            f"{LABEL}: {width}x{height} is not aligned to the {TOKEN_GRID_ALIGN}px token grid -- "
            f"callers must snap through normalize_resolution()/align_to_grid() first.")

    if prefill_callback is not None:
        try:
            prefill_callback()
        except Exception as exc:  # progress must never take a generation down
            print(f"[{LABEL}] prefill_callback raised: {exc}")

    raise_if_cancelled()

    transformer._notify_layer_offload_phase("prefix")
    merge_size = int(1 / transformer.downsample_ratio)
    patch = transformer.patch_size * merge_size
    ref_images = list(ref_images) if ref_images else []
    negative_prompt = (negative_prompt or "").strip()
    sys_msg = SYSTEM_MESSAGE_FOR_GEN if system_message is None else system_message
    token_h = height // patch
    token_w = width // patch

    past_kv_cond = None
    past_kv_uncond = None
    past_kv_img_cond = None
    try:
        if not ref_images:
            # ---- Text-only path: EXACTLY the original (pre-reference) body. ----
            needs_cfg = cfg_scale > 1

            if negative_prompt and not needs_cfg:
                # negative_prompt only has an effect through the uncond branch,
                # which this same needs_cfg gate skips entirely at cfg_scale<=1
                # (the 8-step distillation LoRA's usual operating point, but the
                # real condition is cfg_scale, not "a LoRA is loaded" -- a LoRA
                # run at cfg_scale>1 still gets a real uncond branch). Never
                # silently drop it -- warn instead.
                msg = (f"[{LABEL}] negative_prompt was given but cfg_scale={cfg_scale} <= 1, so no uncond "
                      f"branch is built and the negative prompt has no effect (this is the single-branch/"
                      f"distillation-LoRA operating point). Use cfg_scale > 1 for negative_prompt to work.")
                print(msg)
                try:
                    from api.generation_status import add_warning
                    add_warning(msg, code="sensenova_negative_prompt_no_cfg")
                except Exception:
                    pass

            query_cond = transformer._build_t2i_query(
                prompt, system_message=sys_msg, append_text="<think>\n\n</think>\n\n<img>")
            query_uncond = transformer._build_t2i_query(negative_prompt, append_text="<img>") if needs_cfg else None

            input_ids_cond, indexes_cond, attn_prefix_cond = transformer._build_t2i_text_inputs(tokenizer, query_cond)
            indexes_image_cond = transformer._build_t2i_image_indexes(
                token_h, token_w, indexes_cond.shape[1], device=input_ids_cond.device)

            past_kv_cond, prefix_hidden = transformer._t2i_prefix_forward(input_ids_cond, indexes_cond, attn_prefix_cond)
            device, dtype = prefix_hidden.device, prefix_hidden.dtype
            del prefix_hidden

            indexes_image_uncond = None
            if needs_cfg:
                input_ids_uncond, indexes_uncond, attn_prefix_uncond = transformer._build_t2i_text_inputs(
                    tokenizer, query_uncond)
                indexes_image_uncond = transformer._build_t2i_image_indexes(
                    token_h, token_w, indexes_uncond.shape[1], device=input_ids_uncond.device)
                past_kv_uncond, _ = transformer._t2i_prefix_forward(input_ids_uncond, indexes_uncond, attn_prefix_uncond)

            indexes_image_img_cond = None
        else:
            # ---- Reference-image path: upstream it2i_generate:1386-1524. ----
            pixel_values, grid_hw = _embed_reference_images(transformer, ref_images)
            transformer.img_context_token_id = tokenizer.convert_tokens_to_ids(_IMG_CONTEXT_TOKEN)

            needs_cfg = not (cfg_scale == 1 and img_cfg_scale == 1)
            needs_img_cond = needs_cfg and (img_cfg_scale == 1 or cfg_scale != img_cfg_scale)
            needs_uncond = needs_cfg and img_cfg_scale != 1

            # At img_cfg_scale == 1 (the default) there is no uncond branch, but
            # img_cond IS the blend's baseline -- carrying the negative prompt in
            # its text makes the guidance direction cond - (refs + negative),
            # the usual negative-prompt-as-baseline form, instead of dropping it.
            negative_into_img_cond = bool(negative_prompt) and not needs_uncond and needs_img_cond
            if negative_prompt and not needs_uncond and not needs_img_cond:
                # Neither branch exists (both scales <= 1): nothing can carry it.
                msg = (f"[{LABEL}] negative_prompt was given with ref_images, but cfg_scale={cfg_scale} and "
                      f"img_cfg_scale={img_cfg_scale} build no second branch at all, so the negative prompt "
                      f"has no effect this run. Use cfg_scale > 1 for it to take effect.")
                print(msg)
                try:
                    from api.generation_status import add_warning
                    add_warning(msg, code="sensenova_negative_prompt_no_uncond")
                except Exception:
                    pass

            query_cond_text = _splice_reference_image_tokens(
                prompt, len(ref_images), grid_hw, transformer.downsample_ratio)
            query_cond = transformer._build_t2i_query(
                query_cond_text, system_message=sys_msg, append_text="<think>\n\n</think>\n\n<img>")

            query_img_cond = None
            if needs_img_cond:
                # Splice FIRST, then append the negative text: _splice_reference_image_tokens
                # counts "<image>" occurrences, and a negative prompt containing that
                # literal would otherwise be miscounted as a placeholder.
                img_cond_text = _splice_reference_image_tokens(
                    "<image>" * len(ref_images), len(ref_images), grid_hw, transformer.downsample_ratio)
                if negative_into_img_cond:
                    img_cond_text = f"{img_cond_text}\n{negative_prompt}"
                    msg = (f"[{LABEL}] negative_prompt carried on the img_cond (reference) branch, this run's "
                           f"CFG baseline at img_cfg_scale={img_cfg_scale}. Upstream conditions that branch on "
                           f"the images alone; set img_cfg_scale != 1 for a separate uncond branch instead.")
                    print(msg)
                    try:
                        from api.generation_status import add_warning
                        add_warning(msg, code="sensenova_negative_prompt_on_img_cond")
                    except Exception:
                        pass
                query_img_cond = transformer._build_t2i_query(img_cond_text, append_text="<img>")

            query_uncond = transformer._build_t2i_query(negative_prompt, append_text="<img>") if needs_uncond else None

            input_embeds_cond, indexes_cond, attn_prefix_cond = transformer._build_it2i_inputs(
                tokenizer, query_cond, pixel_values, grid_hw)
            indexes_image_cond = transformer._build_t2i_image_indexes(
                token_h, token_w, indexes_cond[0].max() + 1, device=input_embeds_cond.device)

            past_kv_cond, prefix_hidden = transformer._it2i_prefix_forward(
                input_embeds_cond, indexes_cond, attn_prefix_cond)
            device, dtype = prefix_hidden.device, prefix_hidden.dtype
            del prefix_hidden

            indexes_image_img_cond = None
            if query_img_cond is not None:
                input_embeds_img_cond, indexes_img_cond, attn_prefix_img_cond = transformer._build_it2i_inputs(
                    tokenizer, query_img_cond, pixel_values, grid_hw)
                indexes_image_img_cond = transformer._build_t2i_image_indexes(
                    token_h, token_w, indexes_img_cond[0].max() + 1, device=input_embeds_img_cond.device)
                past_kv_img_cond, _ = transformer._it2i_prefix_forward(
                    input_embeds_img_cond, indexes_img_cond, attn_prefix_img_cond)

            indexes_image_uncond = None
            if query_uncond is not None:
                input_embeds_uncond, indexes_uncond, attn_prefix_uncond = transformer._build_it2i_inputs(
                    tokenizer, query_uncond)
                indexes_image_uncond = transformer._build_t2i_image_indexes(
                    token_h, token_w, indexes_uncond[0].max() + 1, device=input_embeds_uncond.device)
                past_kv_uncond, _ = transformer._it2i_prefix_forward(
                    input_embeds_uncond, indexes_uncond, attn_prefix_uncond)

        raise_if_cancelled()
        _finalize_prefix_caches(
            transformer, [past_kv_cond, past_kv_img_cond, past_kv_uncond], batch_size, token_h, token_w)
    except Exception:
        if past_kv_cond is not None:
            clear_flash_kv_cache(past_kv_cond)
        if past_kv_uncond is not None:
            clear_flash_kv_cache(past_kv_uncond)
        if past_kv_img_cond is not None:
            clear_flash_kv_cache(past_kv_img_cond)
        raise

    grid_h = height // transformer.patch_size
    grid_w = width // transformer.patch_size

    return SenseNovaPrefix(
        device=device, dtype=dtype, batch_size=batch_size,
        token_h=token_h, token_w=token_w, grid_h=grid_h, grid_w=grid_w, merge_size=merge_size,
        image_size=(width, height),
        cond_past_key_values=past_kv_cond,
        cond_indexes_image=indexes_image_cond,
        cond_attention_mask={"full_attention": None},
        encode_cfg_scale=cfg_scale,
        uncond_past_key_values=past_kv_uncond,
        uncond_indexes_image=indexes_image_uncond,
        uncond_attention_mask={"full_attention": None} if past_kv_uncond is not None else None,
        img_cond_past_key_values=past_kv_img_cond,
        img_cond_indexes_image=indexes_image_img_cond,
        img_cond_attention_mask={"full_attention": None} if past_kv_img_cond is not None else None,
        encode_img_cfg_scale=img_cfg_scale,
        has_reference_images=bool(ref_images),
    )


@torch.no_grad()
def _build_step_context(transformer, prefix: SenseNovaPrefix, image_prediction: torch.Tensor, t: torch.Tensor,
                        noise_scale: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-step embed construction (patchify -> gen-branch ViT -> +timestep/
    noise-scale embeddings), computed EXACTLY ONCE per step regardless of
    whether CFG needs a cond+uncond pair.

    Upstream (``vendor/modeling_neo_chat.py:1824-1833``) builds ``z``/
    ``image_embeds``/``timestep_embeddings`` once per step and reuses them
    across BOTH ``_t2i_predict_v`` calls (``:1835`` cond, ``:1838`` uncond);
    ``_t2i_predict_v`` reads ``image_embeds``/``timestep_embeddings``/``z``
    without mutating them (the only per-branch inputs are the KV cache /
    indexes / attention mask), so reuse is safe. The earlier version of this
    module called the old single-branch ``_predict_v`` twice on the CFG path,
    silently doubling this stage's cost (a full ViT pass over every image
    token) every step for no numeric benefit -- see the audit note this
    function's introduction is attached to.

    Returns ``(z, image_embeds, timestep_embeddings)``."""
    batch_size = prefix.batch_size
    patch = transformer.patch_size * prefix.merge_size
    z = transformer.patchify(image_prediction, patch)
    image_input = transformer.patchify(image_prediction, transformer.patch_size, channel_first=True)
    grid_hw = torch.tensor([[prefix.grid_h, prefix.grid_w]] * batch_size, device=image_prediction.device)
    image_embeds = transformer.extract_feature(
        image_input.view(batch_size * prefix.grid_h * prefix.grid_w, -1), gen_model=True, grid_hw=grid_hw,
    ).view(batch_size, prefix.token_h * prefix.token_w, -1)

    t_expanded = t.expand(batch_size * prefix.token_h * prefix.token_w)
    timestep_embeddings = transformer.fm_modules["timestep_embedder"](t_expanded).view(
        batch_size, prefix.token_h * prefix.token_w, -1)
    if transformer.add_noise_scale_embedding:
        noise_scale_tensor = torch.full_like(t_expanded, noise_scale / transformer.noise_scale_max_value)
        noise_embeddings = transformer.fm_modules["noise_scale_embedder"](noise_scale_tensor).view(
            batch_size, prefix.token_h * prefix.token_w, -1)
        timestep_embeddings = timestep_embeddings + noise_embeddings
    image_embeds = image_embeds + timestep_embeddings
    return z, image_embeds, timestep_embeddings


@torch.no_grad()
def _predict_v_branch(transformer, prefix: SenseNovaPrefix, image_embeds: torch.Tensor,
                      timestep_embeddings: torch.Tensor, z: torch.Tensor, t: torch.Tensor,
                      branch: str = "cond") -> torch.Tensor:
    """One ``_t2i_predict_v`` call against the cond/img_cond/uncond prefix KV
    cache, reusing the embeds ``_build_step_context`` already built for this
    step (see that function's docstring)."""
    if branch == "uncond":
        indexes_image = prefix.uncond_indexes_image
        attn_mask = prefix.uncond_attention_mask
        past_kv = prefix.uncond_past_key_values
    elif branch == "img_cond":
        indexes_image = prefix.img_cond_indexes_image
        attn_mask = prefix.img_cond_attention_mask
        past_kv = prefix.img_cond_past_key_values
    else:
        indexes_image = prefix.cond_indexes_image
        attn_mask = prefix.cond_attention_mask
        past_kv = prefix.cond_past_key_values

    return transformer._t2i_predict_v(
        image_embeds, indexes_image, attn_mask, past_kv, t, z,
        image_token_num=prefix.token_h * prefix.token_w,
        timestep_embeddings=timestep_embeddings, image_size=prefix.image_size,
    )


def _cfg_combine(v_cond: torch.Tensor, v_uncond: torch.Tensor, cfg_scale: float, cfg_norm: str,
                 step_i: int) -> torch.Tensor:
    """Mirrors ``t2i_generate``'s CFG blend, all four ``cfg_norm`` modes. Used
    for the classic cond/uncond pair only -- see ``_cfg_combine_refs`` for the
    reference-image branch set."""
    if cfg_norm == "cfg_zero_star":
        positive_flat = v_cond.reshape(v_cond.shape[0], -1)
        negative_flat = v_uncond.reshape(v_uncond.shape[0], -1)
        alpha = optimized_scale(positive_flat, negative_flat)
        alpha = alpha.view(v_cond.shape[0], *([1] * (v_cond.dim() - 1))).to(positive_flat.dtype)
        if step_i <= 0:
            return v_cond * 0.0
        return v_uncond * alpha + cfg_scale * (v_cond - v_uncond * alpha)

    v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)
    # Upstream gates the norm rescale on the scale, not on "a blend happened"
    # (``:1681``). No-refs callers always pass cfg_scale > 1, but the reference
    # path reaches this with equal scales below 1, where upstream skips it.
    if cfg_scale <= 1:
        return v_pred
    if cfg_norm == "global":
        norm_cond = torch.norm(v_cond, dim=(1, 2), keepdim=True)
        norm_cfg = torch.norm(v_pred, dim=(1, 2), keepdim=True)
        scale = (norm_cond / (norm_cfg + 1e-8)).clamp(min=0, max=1.0)
        v_pred = v_pred * scale
    elif cfg_norm == "channel":
        norm_cond = torch.norm(v_cond, dim=-1, keepdim=True)
        norm_cfg = torch.norm(v_pred, dim=-1, keepdim=True)
        scale = (norm_cond / (norm_cfg + 1e-8)).clamp(min=0, max=1.0)
        v_pred = v_pred * scale
    return v_pred


def _cfg_combine_refs(v_cond: torch.Tensor, v_img_cond: torch.Tensor, v_uncond: Optional[torch.Tensor],
                      cfg_scale: float, img_cfg_scale: float, cfg_norm: str) -> torch.Tensor:
    """cond+img_cond(+uncond) CFG blend for the reference-image path -- mirrors
    ``it2i_generate:1623-1691``. Only called when an img_cond branch exists
    (``_euler_run``); ``v_uncond`` is ``None`` when only cond+img_cond exist.
    Upstream restricts ``cfg_norm`` to ``('none','global','channel')`` here
    (no ``cfg_zero_star`` -- that blend is defined against a cond/uncond text
    pair and has no defined meaning against a reference-image branch, so it is
    treated as ``'none'`` here rather than raising).

    Upstream's ``cfg_scale == img_cfg_scale`` arm (``:1640``) is deliberately
    absent: it is unreachable here (that case builds no img_cond branch) and
    the three-branch formula below reduces to it exactly when the scales are
    equal, so it would be dead code, not a behavioral difference."""
    if v_uncond is None:
        v_pred = v_img_cond + cfg_scale * (v_cond - v_img_cond)
    else:
        v_pred = v_uncond + cfg_scale * (v_cond - v_img_cond) + img_cfg_scale * (v_img_cond - v_uncond)

    # Upstream gates the norm rescale on the scales, not on "a blend happened"
    # (``:1681``) -- the blend itself runs for any non-unit scale pair.
    if not (cfg_scale > 1 or img_cfg_scale > 1):
        return v_pred

    if cfg_norm == "global":
        norm_cond = torch.norm(v_cond, dim=(1, 2), keepdim=True)
        norm_cfg = torch.norm(v_pred, dim=(1, 2), keepdim=True)
        scale = (norm_cond / (norm_cfg + 1e-8)).clamp(min=0, max=1.0)
        v_pred = v_pred * scale
    elif cfg_norm == "channel":
        norm_cond = torch.norm(v_cond, dim=-1, keepdim=True)
        norm_cfg = torch.norm(v_pred, dim=-1, keepdim=True)
        scale = (norm_cond / (norm_cfg + 1e-8)).clamp(min=0, max=1.0)
        v_pred = v_pred * scale
    return v_pred


_NO_T_EPS = object()  # sentinel: "the config had no t_eps attribute at all", distinct from None/0.


@contextmanager
def _t_eps_override(transformer, t_eps: Optional[float]):
    """``_t2i_predict_v`` reads ``self.config.t_eps`` directly -- an explicit
    override is applied for the duration of the loop and restored after,
    rather than left mutated on the shared config object. If the config had
    no ``t_eps`` attribute to begin with, the attribute is removed again on
    exit (rather than left dangling at the override value) via the sentinel."""
    if t_eps is None:
        yield
        return
    old = getattr(transformer.config, "t_eps", _NO_T_EPS)
    transformer.config.t_eps = float(t_eps)
    try:
        yield
    finally:
        if old is _NO_T_EPS:
            delattr(transformer.config, "t_eps")
        else:
            transformer.config.t_eps = old


@torch.no_grad()
def _euler_run(
    transformer,
    prefix: SenseNovaPrefix,
    image_prediction: torch.Tensor,
    ts: torch.Tensor,
    start_idx: int,
    cfg_scale: float,
    cfg_interval: Tuple[float, float],
    cfg_norm: str,
    noise_scale: float,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    step_callback: Optional[Callable[..., None]] = None,
    mask_latent: Optional[torch.Tensor] = None,
    init_image: Optional[torch.Tensor] = None,
    fixed_noise: Optional[torch.Tensor] = None,
    clamp_output: bool = True,
) -> torch.Tensor:
    """Shared Euler loop from ``ts[start_idx]`` -> ``t=1`` (clean). ``t`` runs
    forward 0->1 (see module docstring for why this is NOT flux2's direction).

    If ``mask_latent``/``init_image``/``fixed_noise`` are given (inpaint), the
    kept region is pinned to the noised init every step (RePaint), in pixel
    space, against the SAME fixed noise tensor drawn once by the caller.

    Clears all prefix KV caches on the way out regardless of exit path --
    see ``SenseNovaPrefix``'s docstring for the full contract. The reference-
    image ``img_cond`` branch (when present) is driven by
    ``prefix.encode_img_cfg_scale`` -- NOT a new argument here (see
    ``encode_prompt``'s docstring); ``denoise_loop*`` never needs to change.
    """
    if prefix.consumed:
        raise RuntimeError(
            f"{LABEL}: this SenseNovaPrefix was already consumed by a previous denoise_loop* call -- "
            f"its KV caches have been cleared. Call encode_prompt() again for a new generation.")

    img_cfg_scale = prefix.encode_img_cfg_scale
    # A second branch of EITHER kind gives CFG a baseline: uncond for the
    # classic blend, img_cond for the reference blend at img_cfg_scale == 1.
    # img_cfg_scale only counts as "CFG was asked for" on the reference path;
    # without references it is inert and warned about by the pipeline backend.
    if ((cfg_scale > 1 or (prefix.has_reference_images and img_cfg_scale > 1))
            and prefix.uncond_past_key_values is None and prefix.img_cond_past_key_values is None):
        msg = (f"[{LABEL}] cfg_scale={cfg_scale} was requested for this denoise pass, but the prefix was "
              f"built with encode_prompt(cfg_scale={prefix.encode_cfg_scale}, "
              f"img_cfg_scale={prefix.encode_img_cfg_scale}), which built no second branch, so no CFG baseline "
              f"exists. CFG is silently unavailable this run -- proceeding single-branch, which is visibly "
              f"weaker than a normal cfg_scale={cfg_scale} generation.")
        print(msg)
        try:
            from api.generation_status import add_warning
            add_warning(msg, code="sensenova_cfg_mismatch")
        except Exception:
            pass

    n = len(ts) - 1
    total = n - start_idx
    patch = transformer.patch_size * prefix.merge_size
    try:
        for j, i in enumerate(range(start_idx, n)):
            raise_if_cancelled()
            t = ts[i]
            t_next = ts[i + 1]

            # Embeds built ONCE per step, reused across every CFG branch --
            # see _build_step_context's docstring (this was H1: the earlier
            # version recomputed the full ViT feature extraction twice here).
            z, image_embeds, timestep_embeddings = _build_step_context(
                transformer, prefix, image_prediction, t, noise_scale)
            v_cond = _predict_v_branch(transformer, prefix, image_embeds, timestep_embeddings, z, t, branch="cond")

            has_img_cond = prefix.img_cond_past_key_values is not None
            has_uncond = prefix.uncond_past_key_values is not None
            in_interval = cfg_interval[0] <= float(t) <= cfg_interval[1]
            if prefix.has_reference_images:
                # encode_prompt already applied upstream's own needs_cfg gates
                # when it chose the branch set, so re-testing the scales here
                # would skip blends upstream performs (any non-unit scale pair,
                # including scales below 1).
                use_cfg = (has_img_cond or has_uncond) and in_interval
            else:
                # Classic path, unchanged.
                use_cfg = has_uncond and cfg_scale > 1 and in_interval
            if use_cfg:
                if has_img_cond:
                    v_img_cond = _predict_v_branch(transformer, prefix, image_embeds, timestep_embeddings, z, t,
                                                   branch="img_cond")
                    v_uncond = (_predict_v_branch(transformer, prefix, image_embeds, timestep_embeddings, z, t,
                                                  branch="uncond") if has_uncond else None)
                    v_pred = _cfg_combine_refs(v_cond, v_img_cond, v_uncond, cfg_scale, img_cfg_scale, cfg_norm)
                else:
                    # Classic 2-branch path -- IDENTICAL call to before (also
                    # the only path a no-refs prefix can ever reach, since
                    # has_img_cond is always False there).
                    v_uncond = _predict_v_branch(transformer, prefix, image_embeds, timestep_embeddings, z, t,
                                                 branch="uncond")
                    v_pred = _cfg_combine(v_cond, v_uncond, cfg_scale, cfg_norm, j)
            else:
                v_pred = v_cond

            pred_x0 = None
            if step_callback is not None:
                # x0 = z + (1-t).clamp_min(t_eps)*v -- same relation _t2i_predict_v
                # inverts to get v (see vendor/modeling_neo_chat.py:655), recovered
                # here from the PRE-update z/v_pred/t (not the post-update ones)
                # for an accurate preview instead of a near-noise pixel estimate.
                t_eps = float(getattr(transformer.config, "t_eps", 0.02))
                x0_patch = z + (1.0 - t).clamp_min(t_eps) * v_pred
                pred_x0 = transformer.unpatchify(
                    x0_patch, patch, prefix.image_size[1], prefix.image_size[0]).detach()

            z = z + (t_next - t) * v_pred
            image_prediction = transformer.unpatchify(z, patch, prefix.image_size[1], prefix.image_size[0])

            if mask_latent is not None:
                known = init_image * t_next + fixed_noise * (1.0 - t_next)
                image_prediction = mask_latent * image_prediction + (1.0 - mask_latent) * known

            if progress_callback is not None:
                try:
                    progress_callback(j + 1, total)
                except Exception as exc:  # progress must never take a generation down
                    print(f"[{LABEL}] progress_callback raised: {exc}")
            if step_callback is not None:
                try:
                    step_callback(j, total, image_prediction.detach(), None, pred_x0)
                except Exception as exc:
                    print(f"[{LABEL}] step_callback raised: {exc}")
    finally:
        clear_prefix_caches(prefix)

    return image_prediction.clamp(-1, 1) if clamp_output else image_prediction


@torch.no_grad()
@time_phase("denoise")
def denoise_loop(
    transformer,
    prefix: SenseNovaPrefix,
    *,
    cfg_scale: float,
    timestep_shift: float,
    num_inference_steps: int,
    seed: Optional[int] = None,
    cfg_interval: Tuple[float, float] = DEFAULT_CFG_INTERVAL,
    cfg_norm: str = DEFAULT_CFG_NORM,
    t_eps: Optional[float] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    step_callback: Optional[Callable[..., None]] = None,
    clamp_output: bool = True,
) -> torch.Tensor:
    """txt2img: start from pure (resolution-scaled) noise, integrate t: 0 -> 1.
    ``cfg_scale``/``timestep_shift``/``num_inference_steps`` are required, no
    module-level default (AGENTS.md) -- the caller sources them from
    ``api.param_defaults.SENSENOVA_GENERATION_DEFAULTS``."""
    if num_inference_steps < 1:
        raise ValueError(f"{LABEL}: num_inference_steps must be >= 1, got {num_inference_steps}.")
    device, dtype = prefix.device, prefix.dtype
    width, height = prefix.image_size
    noise_scale = compute_noise_scale(transformer, prefix.grid_h, prefix.grid_w, prefix.merge_size)
    x = prepare_noise(height, width, device, dtype, seed, noise_scale, batch_size=prefix.batch_size)
    ts = torch.linspace(0.0, 1.0, num_inference_steps + 1, device=device)
    ts = transformer._apply_time_schedule(ts, prefix.token_h * prefix.token_w, timestep_shift)
    with _t_eps_override(transformer, t_eps):
        return _euler_run(transformer, prefix, x, ts, 0, cfg_scale, cfg_interval, cfg_norm, noise_scale,
                          progress_callback=progress_callback, step_callback=step_callback,
                          clamp_output=clamp_output)


@torch.no_grad()
@time_phase("denoise")
def denoise_loop_img2img(
    transformer,
    prefix: SenseNovaPrefix,
    init_image: Image.Image,
    denoising_strength: float,
    *,
    cfg_scale: float,
    timestep_shift: float,
    num_inference_steps: int,
    seed: Optional[int] = None,
    cfg_interval: Tuple[float, float] = DEFAULT_CFG_INTERVAL,
    cfg_norm: str = DEFAULT_CFG_NORM,
    t_eps: Optional[float] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    step_callback: Optional[Callable[..., None]] = None,
    clamp_output: bool = True,
) -> torch.Tensor:
    """img2img (SDEdit): ``t_start = 1 - denoising_strength``, snapped to the
    shifted timestep grid (clamped so at least one step remains); start the
    loop from the noised init at that index. ``z_t = t*x0 + (1-t)*noise_scale*eps``.
    ``cfg_scale``/``timestep_shift``/``num_inference_steps`` are required, no
    module-level default (AGENTS.md)."""
    if num_inference_steps < 1:
        raise ValueError(f"{LABEL}: num_inference_steps must be >= 1, got {num_inference_steps}.")
    device, dtype = prefix.device, prefix.dtype
    width, height = prefix.image_size
    noise_scale = compute_noise_scale(transformer, prefix.grid_h, prefix.grid_w, prefix.merge_size)
    x0 = image_to_tensor(init_image, height, width, device, dtype)
    ts = torch.linspace(0.0, 1.0, num_inference_steps + 1, device=device)
    ts = transformer._apply_time_schedule(ts, prefix.token_h * prefix.token_w, timestep_shift)
    t_start = max(0.0, min(1.0, 1.0 - float(denoising_strength)))
    start_idx = int((ts <= t_start).sum().item()) - 1
    start_idx = max(0, min(start_idx, num_inference_steps - 1))
    noise = prepare_noise(height, width, device, dtype, seed, noise_scale, batch_size=prefix.batch_size)
    ti = ts[start_idx]
    x = x0 * ti + noise * (1.0 - ti)
    with _t_eps_override(transformer, t_eps):
        return _euler_run(transformer, prefix, x, ts, start_idx, cfg_scale, cfg_interval, cfg_norm, noise_scale,
                          progress_callback=progress_callback, step_callback=step_callback,
                          clamp_output=clamp_output)


@torch.no_grad()
@time_phase("denoise")
def denoise_loop_inpaint(
    transformer,
    prefix: SenseNovaPrefix,
    init_image: Image.Image,
    mask_image: Image.Image,
    denoising_strength: float,
    *,
    cfg_scale: float,
    timestep_shift: float,
    num_inference_steps: int,
    mask_blur: int = 0,
    seed: Optional[int] = None,
    cfg_interval: Tuple[float, float] = DEFAULT_CFG_INTERVAL,
    cfg_norm: str = DEFAULT_CFG_NORM,
    t_eps: Optional[float] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    step_callback: Optional[Callable[..., None]] = None,
    clamp_output: bool = True,
) -> torch.Tensor:
    """inpaint (RePaint): every step, after the Euler update, the kept region
    is re-pinned to ``t_next*x0_orig + (1-t_next)*noise_scale*eps`` against a
    FIXED noise tensor drawn once (see ``_euler_run``). ``mask_image`` is
    white=inpaint (regenerate); resized/blurred by ``prepare_mask``.
    ``cfg_scale``/``timestep_shift``/``num_inference_steps`` are required, no
    module-level default (AGENTS.md)."""
    if num_inference_steps < 1:
        raise ValueError(f"{LABEL}: num_inference_steps must be >= 1, got {num_inference_steps}.")
    device, dtype = prefix.device, prefix.dtype
    width, height = prefix.image_size
    noise_scale = compute_noise_scale(transformer, prefix.grid_h, prefix.grid_w, prefix.merge_size)
    x0 = image_to_tensor(init_image, height, width, device, dtype)
    mask = prepare_mask(mask_image, height, width, device, dtype, mask_blur=mask_blur)
    ts = torch.linspace(0.0, 1.0, num_inference_steps + 1, device=device)
    ts = transformer._apply_time_schedule(ts, prefix.token_h * prefix.token_w, timestep_shift)
    t_start = max(0.0, min(1.0, 1.0 - float(denoising_strength)))
    start_idx = int((ts <= t_start).sum().item()) - 1
    start_idx = max(0, min(start_idx, num_inference_steps - 1))
    fixed_noise = prepare_noise(height, width, device, dtype, seed, noise_scale, batch_size=prefix.batch_size)
    ti = ts[start_idx]
    x = x0 * ti + fixed_noise * (1.0 - ti)
    with _t_eps_override(transformer, t_eps):
        return _euler_run(transformer, prefix, x, ts, start_idx, cfg_scale, cfg_interval, cfg_norm, noise_scale,
                          progress_callback=progress_callback, step_callback=step_callback,
                          mask_latent=mask, init_image=x0, fixed_noise=fixed_noise, clamp_output=clamp_output)
