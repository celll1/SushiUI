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
#
# Adapted (heavily trimmed to inference-only) from `pid/_src/models/pixeldit_model.py`
# in github.com/nv-tlabs/PiD. Stripped relative to upstream:
#   - the `imaginaire`/hydra lazy-config framework (`lazy_instantiate`) — `config.net`
#     here is already a constructed `nn.Module` (built directly by
#     `core.models.pid.loader`), so `PixelDiTModel` just moves/initializes it;
#   - `ImaginaireModel` base class -> plain `torch.nn.Module` (the base class added
#     nothing beyond a few no-op training hooks and one `LazyConfig`-typed method);
#   - EMA (`net_ema`, `ema_scope`, `on_before_zero_grad`, `ema_beta`) — the vendored
#     `.pth` already contains the EMA-merged weights under `net.*` (see
#     `pid_distill_model_infer.PidInferenceModel.load_state_dict`), and the SDXL
#     experiment config disables EMA at inference (`{"override /ema": None}`);
#   - REPA loss (`repa_loss`, `PixelDiTREPALoss`) and the `conditioner` (CFG-dropout
#     training machinery) — both training-only, unused at inference;
#   - `training_step` / `validation_step` / `init_optimizer_scheduler` /
#     `clip_grad_norm_` — training-only;
#   - context-parallel model-level helpers (`_maybe_enable_cp_on_nets`,
#     `_broadcast_tensor_for_cp`, `_broadcast_object_for_cp`,
#     `get_context_parallel_group`, `_cp_size`, `_cp_loss_scale`) — SushiUI never
#     initializes a CP process group for PiD (single-GPU decode only), and the only
#     inference entry point actually used (`PidInferenceModel.generate_samples_from_batch`,
#     in `pid_distill_model_infer.py`) does not call them;
#   - `generate_samples_from_batch` (base-class DPM-Solver + CFG sampler) — the
#     distilled 4-step student model always uses the child class's override
#     (`PidInferenceModel.generate_samples_from_batch`, `_student_sample_loop`),
#     which never falls back to this method. Vendoring it would require the
#     `modules/dpmsolver/*` package (~1.6k lines), which nothing in the inference
#     path we support ever calls;
#   - `_null_caption_embs` precompute — only consumed by the base-class CFG sampler
#     above (the distilled path is CFG-free, single forward pass per step), so it is
#     dead weight for `PidInferenceModel` and was dropped along with `negative_prompt`;
#   - `y_norm` / `y_norm_scale_factor` config fields — per the original code's own
#     comment, never applied to the embeddings.
#
# Kept: precision/autocast setup, direct net construction + `init_weights()` +
# `freeze_unused_text_output_branch()`, `_load_text_encoder` (now gated behind
# `config.load_text_encoder`, defaulting to the same signature/behavior as upstream
# when enabled), `_encode_text_raw`, `_normalize_image`, `fm_trainer`
# (`FlowMatchingTrainer`, whose `.timescale` the distilled sampler consumes),
# `forward` passthrough, `enable_compile` / `_maybe_compile_net` (the child's
# `generate_samples_from_batch` calls the latter unconditionally).

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from core.models.pid.networks.flow_matching import FlowMatchingTrainer
from core.models.pid.utils import misc

logger = logging.getLogger(__name__)


# =============================================================================
# Config
# =============================================================================
#
# Upstream uses `attrs.define(slots=False)` (the `pid._ext.imaginaire.lazy_config`
# framework's config style). This port uses stdlib `dataclasses` instead — same
# "mutable container with typed defaults" semantics for our usage, without adding
# `attrs` as a new pip dependency to the project.


@dataclass
class PixelDiTModelConfig:
    # Network: a pre-constructed nn.Module (built by `core.models.pid.loader`).
    net: Any = None

    # Precision: "bfloat16" uses autocast, net stays float32
    precision: str = "bfloat16"

    # Data keys
    input_caption_key: str = "caption"

    # Text encoder config (Gemma-2-2b-it). Only loaded when `load_text_encoder=True`
    # (SushiUI addition — not present upstream, where the text encoder is always
    # loaded unconditionally in `__init__`). SushiUI's default decode path injects a
    # precomputed/null caption embedding instead (see
    # `PidInferenceModel.generate_samples_from_batch`'s `_caption_embs` override) and
    # never needs Gemma resident for inference.
    load_text_encoder: bool = False
    text_encoder_name: str = "gemma-2-2b-it"
    caption_channels: int = 2304
    model_max_length: int = 300
    chi_prompt: list = field(default_factory=list)

    # Flow matching config
    # fm_timescale: original PixelDiT uses discrete timesteps 0-999 (timescale=1000).
    # FlowMatchingTrainer samples t in [0,1] then passes t*timescale to the network.
    fm_timescale: float = 1000.0
    # prediction_type: "velocity" — network predicts v = noise - x0 (current PixelDiT convention).
    prediction_type: str = "velocity"

    # Inference config
    shift: float = 4.0
    cfg_scale: float = 2.75
    # int -> square; [H, W] list/tuple -> rectangular. Consumed only by
    # generate_samples_from_batch's shape fallback.
    image_size: Any = 1024

    # Dynamic per-step shift via SD3 formula based on actual batch H, W.
    # Format: {"base_shift": float, "base_image_size_for_shift_calc": int}
    dynamic_shift: dict | None = None


# =============================================================================
# Text encoder helper
# =============================================================================

# Map of supported text encoder names to HuggingFace model IDs
_TEXT_ENCODER_DICT = {
    "gemma-2b": "google/gemma-2b",
    "gemma-2b-it": "google/gemma-2b-it",
    "gemma-2-2b": "google/gemma-2-2b",
    "gemma-2-2b-it": "Efficient-Large-Model/gemma-2-2b-it",
    "gemma-2-9b": "google/gemma-2-9b",
    "gemma-2-9b-it": "google/gemma-2-9b-it",
}


def _load_text_encoder(name: str, device: str = "cuda"):
    """Load tokenizer and text encoder (decoder-only LM, extract decoder layers)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert name in _TEXT_ENCODER_DICT, f"Unsupported text encoder: {name}"
    model_id = _TEXT_ENCODER_DICT[name]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "right"
    text_encoder = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).get_decoder().to(device)
    text_encoder.eval()
    text_encoder.requires_grad_(False)

    return tokenizer, text_encoder


# =============================================================================
# Model
# =============================================================================


class PixelDiTModel(torch.nn.Module):
    """PixelDiT T2I inference model.

    Pixel-space flow matching with MMDiT architecture. Text conditioning via frozen
    Gemma-2-2b-it encoder (optional — see `PixelDiTModelConfig.load_text_encoder`).
    """

    def __init__(self, config: PixelDiTModelConfig):
        super().__init__()
        self.config = config

        # 1. Precision setup
        _dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        requested_dtype = _dtype_map[config.precision]
        if requested_dtype != torch.float32:
            self.autocast_dtype = requested_dtype
            self.precision = torch.float32
        else:
            self.autocast_dtype = None
            self.precision = torch.float32
        self.tensor_kwargs = {"device": "cuda", "dtype": self.precision}

        # Caption-embedding override (SushiUI addition, not present upstream): when
        # set, `_encode_text_raw` returns this precomputed tensor instead of running
        # Gemma. This is how the default null-caption decode path (Phase 2) and this
        # Phase-1 smoke (zeros placeholder) both avoid ever loading the text encoder.
        # NOT a 4th data_batch key — `PidInferenceModel._validate_inference_data_batch`
        # keeps its strict 3-key `{caption, LQ_latent, degrade_sigma}` contract; the
        # `caption` field is still required (drives the target batch size B) but its
        # string content is ignored once this override is set.
        self._injected_caption_embs = None

        # 2. Build network. `config.net` is already a constructed nn.Module (the
        # loader builds `PidNet(**kwargs)` directly — no lazy-config instantiation
        # in this vendored port).
        #
        # Deviation from upstream: the reference implementation always upcasts the
        # net to float32 master weights (precision="bfloat16" only controls an
        # autocast context around forward passes; see the `self.precision =
        # torch.float32` line above, which upstream leaves unconditional). That
        # roughly doubles resident VRAM (5.4GB vs. the checkpoint's native 2.7GB
        # bf16) and only matters for training-time gradient/optimizer precision —
        # irrelevant for an inference-only decoder. We keep the net's own weights in
        # bf16 (matching the checkpoint's on-disk dtype exactly, so `load_state_dict`
        # is a straight memcpy) and still run forward passes inside the same
        # `torch.autocast(..., dtype=bf16)` context as upstream (harmless/no-op on
        # top of already-bf16 weights; it only affects the few external fp32 tensors
        # — e.g. the initial noise — that flow into autocast-eligible ops).
        with misc.timer("PixelDiTModel: build_net"):
            self.net = config.net
            self.net = self.net.to(device="cuda", dtype=torch.bfloat16)
            self.net.requires_grad_(True)
            if hasattr(self.net, "init_weights"):
                self.net.init_weights()
            if getattr(self.net, "patch_blocks", None):
                last_patch_block = self.net.patch_blocks[-1]
                if hasattr(last_patch_block, "freeze_unused_text_output_branch"):
                    last_patch_block.freeze_unused_text_output_branch()
            logger.info(f"PixDiT_T2I params: {sum(p.numel() for p in self.net.parameters()):,}")

        # 3. Text encoder (frozen, optional — see `config.load_text_encoder` docstring).
        # Stored outside nn.Module registration (bypass nn.Module.__setattr__) so a
        # state_dict() call never tries to serialize the frozen HF text encoder.
        self._chi_prompt_str = "\n".join(config.chi_prompt) if config.chi_prompt else ""
        self._num_chi_tokens = 0
        object.__setattr__(self, "tokenizer", None)
        object.__setattr__(self, "text_encoder", None)
        if config.load_text_encoder:
            with misc.timer("PixelDiTModel: load_text_encoder"):
                _tokenizer, _text_encoder = _load_text_encoder(config.text_encoder_name, device="cuda")
                object.__setattr__(self, "tokenizer", _tokenizer)
                object.__setattr__(self, "text_encoder", _text_encoder)
                self._num_chi_tokens = len(self.tokenizer.encode(self._chi_prompt_str)) if self._chi_prompt_str else 0

        # 4. Flow matching trainer. Only `.timescale` is consumed by the distilled
        # student sampler (`PidInferenceModel._student_sample_loop`); the rest of
        # `FlowMatchingTrainer` (loss / t-sampler) is training-only and unused here.
        self.fm_trainer = FlowMatchingTrainer(
            timescale=config.fm_timescale,
            sigma_min=0.0,
            t_sampler_args={},
            t_sampler_type="logit_normal",
            prediction_type=config.prediction_type,
        )

        # 5. Dynamic shift config (resolved per-call in generate_samples_from_batch).
        if config.dynamic_shift is not None:
            _ds = config.dynamic_shift
            logger.info(
                f"PixelDiT dynamic shift: base_shift={_ds['base_shift']} "
                f"base_image_size={_ds['base_image_size_for_shift_calc']}"
            )

    # =========================================================================
    # Text encoding
    # =========================================================================

    def set_injected_caption_embs(self, caption_embs: Tensor | None) -> None:
        """Install (or clear, with `None`) the caption-embedding override consumed
        by `_encode_text_raw`. `caption_embs` is `[1 or B, model_max_length,
        caption_channels]`; a batch-of-1 tensor is broadcast to the caller's B."""
        self._injected_caption_embs = caption_embs

    @torch.no_grad()
    def _encode_text_raw(self, captions: list[str]) -> tuple[Tensor, Tensor | None]:
        """Encode captions through the text encoder — or, if
        `set_injected_caption_embs()` was called, return that precomputed tensor
        instead (Gemma never runs). This is how SushiUI's default null-caption
        decode path (Phase 2) and the Phase-1 feasibility smoke (zeros placeholder)
        both avoid loading the text encoder at all.

        Returns:
            caption_embs: [B, model_max_length, caption_channels]
            emb_masks: [B, model_max_length], or None when using the override
                (there is no attention mask for a precomputed embedding).
        """
        if self._injected_caption_embs is not None:
            embs = self._injected_caption_embs
            B = len(captions)
            if embs.shape[0] == 1 and B > 1:
                embs = embs.expand(B, -1, -1)
            elif embs.shape[0] != B:
                raise ValueError(
                    f"Injected caption_embs batch {embs.shape[0]} does not match "
                    f"and cannot broadcast to requested batch size {B}"
                )
            return embs, None

        if self.text_encoder is None or self.tokenizer is None:
            raise RuntimeError(
                "_encode_text_raw() requires config.load_text_encoder=True (Gemma not loaded), "
                "or call set_injected_caption_embs() to provide a precomputed embedding."
            )

        # Optionally prepend CHI prompt
        if self._chi_prompt_str:
            prompts_all = [self._chi_prompt_str + cap for cap in captions]
            max_length_all = self._num_chi_tokens + self.config.model_max_length - 2
        else:
            prompts_all = captions
            max_length_all = self.config.model_max_length

        caption_token = self.tokenizer(
            prompts_all,
            max_length=max_length_all,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to("cuda")

        caption_embs = self.text_encoder(caption_token.input_ids, caption_token.attention_mask)[
            0
        ]  # [B, max_length_all, C]

        # Select relevant tokens: BOS + last (model_max_length - 1) tokens
        select_index = [0] + list(range(-self.config.model_max_length + 1, 0))
        caption_embs = caption_embs[:, select_index]  # [B, model_max_length, C]
        emb_masks = caption_token.attention_mask[:, select_index]

        return caption_embs, emb_masks

    # =========================================================================
    # Data helpers
    # =========================================================================

    def _normalize_image(self, img: Tensor) -> Tensor:
        """Normalize image to [-1, 1]. Handles uint8 [0,255] or float [0,1]."""
        if img.dtype == torch.uint8:
            return img.float() / 127.5 - 1.0
        elif img.max() > 1.0:
            return img.float() / 127.5 - 1.0
        else:
            if img.min() >= 0:
                return img.float() * 2.0 - 1.0
            return img.float()

    # =========================================================================
    # Direct network forward
    # =========================================================================

    def forward(self, x, t, y, **kwargs):
        """Direct network forward pass."""
        return self.net(x, t, y, **kwargs)

    # =========================================================================
    # torch.compile (opt-in; off by default)
    # =========================================================================

    def enable_compile(self) -> None:
        """Arm torch.compile for `self.net`, wrapped lazily per output resolution
        by `_maybe_compile_net` (called from `PidInferenceModel.generate_samples_from_batch`).
        """
        if not hasattr(self, "_compiled_nets"):
            self._compiled_nets = {}
        self._compile_enabled = True
        logger.info("PixelDiTModel: torch.compile armed for net (lazy, per output resolution).")

    def _maybe_compile_net(self, image_h: int, image_w: int, text_len: int):
        """Return the net to run for this shape: a torch.compile-wrapped net when
        `enable_compile()` was called (compiled once per (H, W) and cached), else
        the eager net."""
        del text_len
        if not getattr(self, "_compile_enabled", False):
            return self.net
        key = (int(image_h), int(image_w))
        compiled_nets = getattr(self, "_compiled_nets", None)
        if compiled_nets is None:
            compiled_nets = {}
            self._compiled_nets = compiled_nets
        compiled = compiled_nets.get(key)
        if compiled is None:
            logger.info(f"--compile: compiling net for {image_h}x{image_w}")
            compiled = torch.compile(self.net, mode="default", dynamic=False)
            compiled_nets[key] = compiled
        return compiled
