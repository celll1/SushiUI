# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SushiUI-facing entry point for the vendored PiD (Pixel Diffusion Decoder) SDXL
4-step distilled student.

This module hard-codes the hydra-resolved config for
`PiD_res2kto4k_sr4x_official_sdxl_distill_4step` (see
`pid/_src/configs/pid/experiment_2kto4k/sdxl.py` +
`pid/_src/configs/common/defaults/net.py: PID_SR4X`, with the SDXL
`lq_latent_channels=4` / `state_ch=4` override) so the whole `imaginaire`/hydra
lazy-config framework never needs to be vendored.

Not vendored from NVIDIA source (original code, written for this port).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from core.models.pid.models.pid_distill_model_infer import PidInferenceConfig, PidInferenceModel
from core.models.pid.networks.pid_net import PidNet

logger = logging.getLogger(__name__)

# From `pid/_src/configs/pid/experiment_2kto4k/shared_config.py: _CHI_PROMPT`.
_CHI_PROMPT = [
    'Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:',
    "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
    "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
    "Here are examples of how to transform or refine prompts:",
    "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",
    "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",
    "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
    "User Prompt: ",
]

# From `pid/_src/configs/common/defaults/net.py: PID_SR4X`, with the SDXL override
# (`pid/_src/configs/pid/experiment_2kto4k/sdxl.py`): `lq_latent_channels=4` (the
# default `PID_SR4X` targets 16-ch VAEs — Flux1/SD3 — SDXL's VAE is 4-ch).
PID_SR4X_SDXL_KWARGS = dict(
    in_channels=3,
    num_groups=24,
    hidden_size=1536,
    pixel_hidden_size=16,
    pixel_attn_hidden_size=1152,
    pixel_num_groups=16,
    patch_depth=14,
    pixel_depth=2,
    patch_size=16,
    txt_embed_dim=2304,
    txt_max_length=300,
    use_text_rope=True,
    text_rope_theta=10000.0,
    repa_encoder_index=6,
    lq_inject_mode="controlnet",
    lq_in_channels=0,
    lq_latent_channels=4,  # SDXL override (default PID_SR4X value is 16)
    lq_hidden_dim=512,
    lq_latent_unpatchify_factor=1,
    lq_conv_padding_mode="zeros",
    lq_aux_rgb_head=False,
    lq_aux_rgb_head_latent_block_idx=-1,
    lq_gate_type="sigma_aware_per_token_per_dim",
    lq_interval=2,
    zero_init_lq=True,
    train_lq_proj_only=False,
    sr_scale=4,
    pit_lq_inject=False,
)


def build_pid_sdxl_config(load_text_encoder: bool = False, init_device: str = "cuda") -> PidInferenceConfig:
    """Build the hard-coded `PidInferenceConfig` for the SDXL 4-step distilled decoder.

    Args:
        load_text_encoder: if True, loads Gemma-2-2b-it (`config.text_encoder_name`)
            for real prompt conditioning. Default False — SushiUI's default decode
            path injects a precomputed/null caption embedding instead (see
            `core.models.pid.models.pixeldit_model.PixelDiTModel.set_injected_caption_embs`).
        init_device: device `self.net` is constructed on (see `PixelDiTModelConfig
            .init_device`'s docstring). "cpu" avoids any GPU allocation until the
            caller explicitly stages the net (e.g. `PidVaeWrapper`'s lazy load).
    """
    net = PidNet(**PID_SR4X_SDXL_KWARGS)
    return PidInferenceConfig(
        net=net,
        precision="bfloat16",
        load_text_encoder=load_text_encoder,
        text_encoder_name="gemma-2-2b-it",
        caption_channels=2304,
        input_caption_key="caption",
        model_max_length=300,
        chi_prompt=_CHI_PROMPT,
        prediction_type="velocity",
        fm_timescale=1000.0,
        shift=6.0,
        cfg_scale=5.0,
        dynamic_shift={"base_shift": 4.0, "base_image_size_for_shift_calc": 1024},
        image_size=2048,
        student_timestep=1.0,
        student_sample_steps=4,
        student_sample_type="sde",
        student_t_list=[0.999, 0.866, 0.634, 0.342, 0.0],
        init_device=init_device,
    )


def load_pid_sdxl_decoder(
    pth_path: str,
    device: str = "cuda",
    load_text_encoder: bool = False,
) -> PidInferenceModel:
    """Load the PiD SDXL 4-step distilled decoder from its NVIDIA `.pth` checkpoint.

    The checkpoint is a flat `{"net.<name>": tensor}` dict (EMA-merged, native
    bf16); `PidInferenceModel.load_state_dict` strips the `net.` prefix and drops
    any `net_ema.*` / `fake_score.*` / `discriminator.*` keys (none present in the
    SDXL distilled release). Net parameters are constructed DIRECTLY on `device`
    in the checkpoint's native bf16 (see `PixelDiTModel.__init__`'s `init_device`),
    so this is a `strict=False` same-dtype memcpy with no wasted transient GPU
    allocation when `device="cpu"` — no fp32 upcast, no missing/unexpected keys
    expected for this checkpoint.
    """
    config = build_pid_sdxl_config(load_text_encoder=load_text_encoder, init_device=device)
    model = PidInferenceModel(config)

    state_dict = torch.load(pth_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        # The official SDXL distilled checkpoint loads with 0 missing / 0 unexpected
        # keys. Any mismatch means a truncated or wrong .pth — refuse to run PiD with
        # partially-initialized (untrained) weights, which would silently produce
        # garbage instead of failing over. The caller (pipeline.load_override_vae /
        # apply_overrides) catches this and keeps the real VAE.
        raise RuntimeError(
            f"[PiD] checkpoint {pth_path} does not match the SDXL decoder architecture "
            f"({len(missing)} missing, {len(unexpected)} unexpected keys) - refusing to "
            f"run with partial weights. missing={missing[:5]} unexpected={unexpected[:5]}"
        )

    model = model.to(device=device)
    model.eval()
    return model


def decode_latent(
    model: PidInferenceModel,
    lq_latent: torch.Tensor,
    caption_embs: Optional[torch.Tensor] = None,
    caption: str = "",
    sr_output: str = "4x",
    seed: int = 0,
    degrade_sigma: float = 0.0,
    num_steps: Optional[int] = None,
) -> torch.Tensor:
    """Decode an SDXL latent (in PiD's normalized frame) through the student sampler.

    Args:
        lq_latent: [B, 4, H, W] SDXL latent ALREADY in PiD's normalized training
            frame (z' = 0.13025 * (z - shift), std ~0.6-1.0). This helper forwards
            it UNCHANGED to the model -- unlike PidVaeWrapper.pid_final_decode it does
            NOT re-normalize, so a caller must pass the normalized latent (the
            production path re-normalizes the raw diffusion latent first; see the F1
            note in pid_vae_wrapper.py).
        caption_embs: optional precomputed `[1 or B, model_max_length,
            caption_channels]` embedding. When given, installed via
            `model.set_injected_caption_embs()` for the duration of this call (then
            cleared) so Gemma is never required. When `None`, `model` must have
            `config.load_text_encoder=True` and `caption` is encoded normally.
        sr_output: "4x" decodes at native PiD super-resolution
            (H*8*sr_scale, W*8*sr_scale); "original" decodes directly at the
            latent's native resolution (H*8, W*8) — NOTE this generates directly at
            that resolution via the student sampler rather than downscaling a 4x
            decode; the production (Phase 2) policy for `pid_sr_output="original"`
            is not yet finalized (may instead downscale the 4x output — TBD).
        num_steps: override the student sampler's step count (default: config's
            `student_sample_steps`, i.e. 4 for this checkpoint).

    Returns:
        [B, 3, H, W] tensor in [-1, 1] (temporal dim already squeezed — see
        `PidInferenceModel.generate_samples_from_batch`'s `[B,3,1,H,W]` docstring).
    """
    B, C, H, W = lq_latent.shape
    if sr_output == "4x":
        sr_scale = model.net.sr_scale
        image_size = (H * 8 * sr_scale, W * 8 * sr_scale)
    elif sr_output == "original":
        image_size = (H * 8, W * 8)
    else:
        raise ValueError(f"sr_output must be '4x' or 'original', got {sr_output!r}")

    data_batch = {
        model.config.input_caption_key: [caption] * B,
        "LQ_latent": lq_latent,
        "degrade_sigma": torch.tensor([degrade_sigma] * B),
    }

    prior_override = model._injected_caption_embs
    try:
        if caption_embs is not None:
            model.set_injected_caption_embs(caption_embs)
        output = model.generate_samples_from_batch(
            data_batch,
            num_steps=num_steps,
            seed=seed,
            image_size=image_size,
        )
    finally:
        model.set_injected_caption_embs(prior_override)

    return output.squeeze(2)  # [B, 3, 1, H, W] -> [B, 3, H, W]
