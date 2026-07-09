"""LTX-2.3 Transformer Wrapper — block-loop re-ownership (AP1 foundation).

LTX-2.3 (`LTX2VideoTransformer3DModel`) delegates its whole forward — including
the `for block in transformer_blocks` loop — to diffusers. Image archs own their
inner loop (see `flux2_block_swap_wrapper.py`), which is where block-swap /
FBCache / TREAD hooks live. To give LTX-2.3 the same extension points WITHOUT
touching the verified diffusers generation/training paths, this wrapper wraps the
stock transformer and re-owns ONLY stage 5 (the block loop). Every other stage
(RoPE, projections, timestep/modulation embeddings, caption projection, output
layers) is executed by CALLING the inner model's own submodules, so the custom
path reproduces the stock forward tensor-for-tensor.

Fast path: when NO feature is attached (no block-offloader, no FBCache, no
training config) the wrapper delegates verbatim to `self.transformer(...)`, so
the default LTX-2.3 path is byte-identical to the unwrapped model.

Extension points (populated by later phases, all None by default -> fast path):
  * ``_block_offloader``  — AP1 block-swap GENERATION (this file).
  * ``_fbcache`` / ``_fbcache_step`` — AP2 First-Block-Cache (joint audio+video;
    implemented -- see ``_custom_forward``'s block loop; inference-only).
  * ``_spectrum_video`` / ``_spectrum_audio`` / ``_spectrum_step`` — Spectrum
    (Adaptive Spectral Feature Forecasting; implemented -- see ``forward``'s
    pre-CFG output-mode skip; inference-only; mutually exclusive with FBCache).
  * ``_tread_router`` / ``_block_dropout`` — AP3 TREAD + stochastic-depth TRAINING.

The AP3 slots are declared here but NOT implemented; the custom block loop
leaves clean per-block hook sites for them (see the comments in ``forward``).

Diffusers pin: the wrapper depends on the inner model exposing a fixed set of
submodule names (asserted at construction). A diffusers upgrade that renames any
of them raises a clear error at load time rather than silently mis-running.
"""

from typing import Any, Optional

import torch
import torch.nn as nn

from diffusers.models.transformers.transformer_ltx2 import AudioVisualModelOutput


# Inner-model submodules the custom forward calls directly (stages 1-4, 6). If a
# diffusers upgrade renames any of these, the wrapper's stage replication would
# silently diverge; we assert their presence at construction instead.
_REQUIRED_SUBMODULES = (
    # Stage 1 — RoPE
    "rope",
    "audio_rope",
    "cross_attn_rope",
    "cross_attn_audio_rope",
    # Stage 2 — input projections
    "proj_in",
    "audio_proj_in",
    # Stage 3 — timestep / modulation embeddings
    "time_embed",
    "audio_time_embed",
    "av_cross_attn_video_scale_shift",
    "av_cross_attn_video_a2v_gate",
    "av_cross_attn_audio_scale_shift",
    "av_cross_attn_audio_v2a_gate",
    # Stage 4 — caption projection is LTX-2.0-only (created only when
    # config.use_prompt_embeddings; LTX-2.3 projects text in the connector
    # instead, so these submodules are ABSENT). _custom_forward guards their
    # use with the same `config.use_prompt_embeddings` check, so they are NOT
    # required here.
    # Stage 5 — the block loop this wrapper re-owns
    "transformer_blocks",
    # Stage 6 — output layers
    "norm_out",
    "proj_out",
    "audio_norm_out",
    "audio_proj_out",
)

# Parameters (not submodules) stage 3/6 reads directly.
_REQUIRED_PARAMS = (
    "scale_shift_table",
    "audio_scale_shift_table",
)


class Ltx2BlockLoopWrapper(nn.Module):
    """Wrap ``LTX2VideoTransformer3DModel`` and re-own only the block loop.

    Order of construction relative to other features (mirrors the FLUX.2 wrapper):
      1. LoRA-wrap the INNER transformer.
      2. Wrap it with ``Ltx2BlockLoopWrapper``.
      3. Build the block offloader over ``wrapper.transformer.transformer_blocks``
         and attach it (``wrapper.attach_block_offloader``).

    Passthroughs (``to`` / ``__getattr__`` / ``state_dict`` / ``load_state_dict``
    / ``config`` / ``dtype`` / ``device``) make LoRA save/load, the block-swap
    conductor and the diffusers pipeline see the wrapper as the transformer.
    """

    def __init__(self, transformer: nn.Module, block_offloader: Optional[Any] = None):
        super().__init__()
        self._assert_diffusers_pin(transformer)

        self.transformer = transformer

        # === Extension slots (None -> fast path; byte-identical default) ===
        # AP1 — block-swap generation (generic TransformerBlockOffloader over
        # transformer_blocks; explicit wait/submit in the loop, forward-only).
        self._block_offloader = block_offloader
        # AP2 — First-Block-Cache (joint (video, audio) residual).
        self._fbcache = None
        self._fbcache_step = 0
        # Spectrum — Adaptive Spectral Feature Forecasting (output mode, joint
        # video+audio, forecast PRE-CFG so the elementwise-linear Chebyshev fit
        # commutes with the CFG combine done downstream by the diffusers pipeline
        # loop). Two forecasters built with IDENTICAL config so is_anchor(step)
        # agrees for both (the anchor schedule depends only on step index +
        # config, not on the tensor data). Mutually exclusive with FBCache (same
        # trajectory-redundancy target) and with real Block Swap (a forecast skip
        # step returns without running the block loop, desyncing the swap
        # prefetch rotation) -- both guarded in ltx2.py.
        self._spectrum_video = None
        self._spectrum_audio = None
        self._spectrum_step = 0
        # AP3 — TREAD token routing + stochastic-depth block dropout (training).
        # Not implemented; declared so train_step can attach/clear them.
        self._tread_router = None
        self._block_dropout = None

        # Compatibility attributes (diffusers pipeline + LoRA introspection).
        self.config = transformer.config
        self.dtype = transformer.dtype
        try:
            self.device = next(transformer.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")

    # ------------------------------------------------------------------
    # Diffusers-version pin
    # ------------------------------------------------------------------
    @staticmethod
    def _assert_diffusers_pin(transformer: nn.Module) -> None:
        missing_mods = [n for n in _REQUIRED_SUBMODULES if not hasattr(transformer, n)]
        missing_params = [
            n for n in _REQUIRED_PARAMS
            if not (hasattr(transformer, n) and isinstance(getattr(transformer, n), torch.Tensor))
        ]
        if missing_mods or missing_params:
            raise RuntimeError(
                "Ltx2BlockLoopWrapper diffusers-pin failure: "
                f"{transformer.__class__.__name__} is missing expected submodules "
                f"{missing_mods} and/or parameters {missing_params}. The wrapper's "
                "stage-1-6 replication is written against diffusers "
                "LTX2VideoTransformer3DModel; a version that renamed these attributes "
                "must be re-verified against the stock forward before block-swap / "
                "FBCache / TREAD can be re-enabled for LTX-2.3."
            )

    # ------------------------------------------------------------------
    # Feature attach / detach
    # ------------------------------------------------------------------
    def attach_block_offloader(self, block_offloader: Optional[Any]) -> None:
        """Attach (or clear with None) the generation block offloader."""
        self._block_offloader = block_offloader

    def attach_fbcache(self, fbcache: Optional[Any]) -> None:
        """Attach (or clear with None) the AP2 First-Block-Cache.

        FBCache is INFERENCE-ONLY (asserted in ``_custom_forward``); the caller
        (``ltx2.py``) is responsible for building it via
        ``core.inference.fbcache.build_fbcache`` and for guarding mutual
        exclusivity with Block Swap / Spectrum before attaching. ``_fbcache_step``
        is a plain settable attribute the pipeline callback advances once per
        denoise step (see ``ltx2.py``'s ``callback_on_step_end``)."""
        assert fbcache is None or self._spectrum_video is None, (
            "Ltx2BlockLoopWrapper: FBCache and Spectrum must never be attached "
            "simultaneously (mutually exclusive trajectory-redundancy features; "
            "guarded in ltx2.py)."
        )
        self._fbcache = fbcache
        self._fbcache_step = 0

    def attach_spectrum(self, video_forecaster: Optional[Any], audio_forecaster: Optional[Any]) -> None:
        """Attach (or clear with ``(None, None)``) the Spectrum output forecasters.

        Spectrum is INFERENCE-ONLY (asserted in ``forward``) and mutually
        exclusive with FBCache (asserted here and in ``forward``); the caller
        (``ltx2.py``) is responsible for building both forecasters via
        ``core.inference.spectrum_forecaster.build_output_forecaster`` with
        IDENTICAL config (so ``is_anchor(step)`` agrees for video and audio) and
        for guarding mutual exclusivity with FBCache / Block Swap before
        attaching. ``_spectrum_step`` is a plain settable attribute the pipeline
        callback advances once per denoise step (mirrors ``_fbcache_step``; see
        ``ltx2.py``'s ``callback_on_step_end``)."""
        assert video_forecaster is None or self._fbcache is None, (
            "Ltx2BlockLoopWrapper: Spectrum and FBCache must never be attached "
            "simultaneously (mutually exclusive trajectory-redundancy features; "
            "guarded in ltx2.py)."
        )
        self._spectrum_video = video_forecaster
        self._spectrum_audio = audio_forecaster
        self._spectrum_step = 0

    def _any_feature_active(self) -> bool:
        swap_on = (
            self._block_offloader is not None
            and getattr(self._block_offloader, "blocks_to_swap", 0) > 0
        )
        return bool(swap_on or self._fbcache is not None or self._spectrum_video is not None
                    or self._tread_router is not None or self._block_dropout is not None)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        audio_encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        audio_timestep: torch.LongTensor | None = None,
        sigma: torch.Tensor | None = None,
        audio_sigma: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        audio_encoder_attention_mask: torch.Tensor | None = None,
        num_frames: int | None = None,
        height: int | None = None,
        width: int | None = None,
        fps: float = 24.0,
        audio_num_frames: int | None = None,
        video_coords: torch.Tensor | None = None,
        audio_coords: torch.Tensor | None = None,
        isolate_modalities: bool = False,
        spatio_temporal_guidance_blocks: list[int] | None = None,
        perturbation_mask: torch.Tensor | None = None,
        use_cross_timestep: bool = False,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> torch.Tensor:
        """Forward with the EXACT stock LTX2VideoTransformer3DModel signature.

        Fast path (no feature attached) delegates verbatim to the inner model.
        """
        # Spectrum: on a forecast (non-anchor) step, skip the ENTIRE forward
        # (RoPE / projections / block loop / output layers) and return the
        # forecasted (video, audio) output directly. The wrapper receives the
        # PRE-CFG batch (diffusers concatenates [uncond, cond] before calling
        # the transformer once per step), so this forecasts the whole 2B-batch
        # tensor; the Chebyshev ridge fit + w-mix are elementwise-linear, so
        # forecasting pre-CFG is equivalent to forecasting the post-CFG
        # combination (same reasoning as the SD/SDXL/FLUX.2 Spectrum paths).
        if self._spectrum_video is not None:
            assert not torch.is_grad_enabled(), (
                "Ltx2BlockLoopWrapper: Spectrum (_spectrum_video/_spectrum_audio) must "
                "not be attached while autograd is enabled (inference-only feature)."
            )
            assert self._fbcache is None, (
                "Ltx2BlockLoopWrapper: Spectrum and FBCache must never be attached "
                "simultaneously (mutually exclusive trajectory-redundancy features; "
                "guarded in ltx2.py)."
            )
            step = int(self._spectrum_step)
            if not self._spectrum_video.is_anchor(step):
                forecast_video = self._spectrum_video.forecast(step)
                forecast_audio = self._spectrum_audio.forecast(step)
                if not return_dict:
                    return (forecast_video, forecast_audio)
                return AudioVisualModelOutput(sample=forecast_video, audio_sample=forecast_audio)

        if not self._any_feature_active():
            # Byte-identical default: the inner model's own forward (protects the
            # verified LTX-2.3 generation + training paths). The @apply_lora_scale
            # decorator on the inner forward still fires here.
            return self.transformer(
                hidden_states=hidden_states,
                audio_hidden_states=audio_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                audio_encoder_hidden_states=audio_encoder_hidden_states,
                timestep=timestep,
                audio_timestep=audio_timestep,
                sigma=sigma,
                audio_sigma=audio_sigma,
                encoder_attention_mask=encoder_attention_mask,
                audio_encoder_attention_mask=audio_encoder_attention_mask,
                num_frames=num_frames,
                height=height,
                width=width,
                fps=fps,
                audio_num_frames=audio_num_frames,
                video_coords=video_coords,
                audio_coords=audio_coords,
                isolate_modalities=isolate_modalities,
                spatio_temporal_guidance_blocks=spatio_temporal_guidance_blocks,
                perturbation_mask=perturbation_mask,
                use_cross_timestep=use_cross_timestep,
                attention_kwargs=attention_kwargs,
                return_dict=return_dict,
            )

        result = self._custom_forward(
            hidden_states=hidden_states,
            audio_hidden_states=audio_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            audio_encoder_hidden_states=audio_encoder_hidden_states,
            timestep=timestep,
            audio_timestep=audio_timestep,
            sigma=sigma,
            audio_sigma=audio_sigma,
            encoder_attention_mask=encoder_attention_mask,
            audio_encoder_attention_mask=audio_encoder_attention_mask,
            num_frames=num_frames,
            height=height,
            width=width,
            fps=fps,
            audio_num_frames=audio_num_frames,
            video_coords=video_coords,
            audio_coords=audio_coords,
            isolate_modalities=isolate_modalities,
            spatio_temporal_guidance_blocks=spatio_temporal_guidance_blocks,
            perturbation_mask=perturbation_mask,
            use_cross_timestep=use_cross_timestep,
            attention_kwargs=attention_kwargs,
            return_dict=return_dict,
        )

        # Spectrum: this was an ANCHOR step (the forecast branch above returned
        # early on a skip step) -- record the wrapper's final returned tensors
        # (post proj_out / audio_proj_out, the same tensors the pipeline
        # receives) so future forecast steps extrapolate from them.
        if self._spectrum_video is not None:
            step = int(self._spectrum_step)
            if isinstance(result, tuple):
                video_out, audio_out = result[0], result[1]
            else:
                video_out, audio_out = result.sample, result.audio_sample
            self._spectrum_video.record(step, video_out)
            self._spectrum_audio.record(step, audio_out)

        return result

    # NOTE: the @apply_lora_scale decorator on the stock forward pops the LoRA
    # scale out of attention_kwargs before the blocks run. In the custom path we
    # forward attention_kwargs UNCHANGED into each block (matching how the stock
    # loop passes it after the decorator has run); LTX-2.3 generation/training do
    # not currently set a LoRA scale in attention_kwargs, so this is a no-op today.
    def _custom_forward(
        self,
        hidden_states,
        audio_hidden_states,
        encoder_hidden_states,
        audio_encoder_hidden_states,
        timestep,
        audio_timestep,
        sigma,
        audio_sigma,
        encoder_attention_mask,
        audio_encoder_attention_mask,
        num_frames,
        height,
        width,
        fps,
        audio_num_frames,
        video_coords,
        audio_coords,
        isolate_modalities,
        spatio_temporal_guidance_blocks,
        perturbation_mask,
        use_cross_timestep,
        attention_kwargs,
        return_dict,
    ):
        t = self.transformer
        offloader = self._block_offloader
        swap_on = offloader is not None and getattr(offloader, "blocks_to_swap", 0) > 0
        fbcache = self._fbcache
        if fbcache is not None:
            # FBCache is INFERENCE-ONLY: a cache hit skips real block computation
            # (and its gradients), so it must never be attached during training.
            assert not torch.is_grad_enabled(), (
                "Ltx2BlockLoopWrapper: FBCache (_fbcache) must not be attached "
                "while autograd is enabled (inference-only feature)."
            )

        # === Replicated stock stages (transformer_ltx2.forward 1420-1535) ===
        # Determine timestep for audio.
        audio_timestep = audio_timestep if audio_timestep is not None else timestep
        audio_sigma = audio_sigma if audio_sigma is not None else sigma

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        if audio_encoder_attention_mask is not None and audio_encoder_attention_mask.ndim == 2:
            audio_encoder_attention_mask = (1 - audio_encoder_attention_mask.to(audio_hidden_states.dtype)) * -10000.0
            audio_encoder_attention_mask = audio_encoder_attention_mask.unsqueeze(1)

        batch_size = hidden_states.size(0)

        # 1. Prepare RoPE positional embeddings
        if video_coords is None:
            video_coords = t.rope.prepare_video_coords(
                batch_size, num_frames, height, width, hidden_states.device, fps=fps
            )
        if audio_coords is None:
            audio_coords = t.audio_rope.prepare_audio_coords(
                batch_size, audio_num_frames, audio_hidden_states.device
            )

        video_rotary_emb = t.rope(video_coords, device=hidden_states.device)
        audio_rotary_emb = t.audio_rope(audio_coords, device=audio_hidden_states.device)

        video_cross_attn_rotary_emb = t.cross_attn_rope(video_coords[:, 0:1, :], device=hidden_states.device)
        audio_cross_attn_rotary_emb = t.cross_attn_audio_rope(
            audio_coords[:, 0:1, :], device=audio_hidden_states.device
        )

        # 2. Patchify input projections
        hidden_states = t.proj_in(hidden_states)
        audio_hidden_states = t.audio_proj_in(audio_hidden_states)

        # 3. Prepare timestep embeddings and modulation parameters
        timestep_cross_attn_gate_scale_factor = (
            t.config.cross_attn_timestep_scale_multiplier / t.config.timestep_scale_multiplier
        )

        # 3.1. Global modality (video / audio) timestep embedding and modulation params
        temb, embedded_timestep = t.time_embed(
            timestep.flatten(),
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )
        temb = temb.view(batch_size, -1, temb.size(-1))
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.size(-1))

        temb_audio, audio_embedded_timestep = t.audio_time_embed(
            audio_timestep.flatten(),
            batch_size=batch_size,
            hidden_dtype=audio_hidden_states.dtype,
        )
        temb_audio = temb_audio.view(batch_size, -1, temb_audio.size(-1))
        audio_embedded_timestep = audio_embedded_timestep.view(batch_size, -1, audio_embedded_timestep.size(-1))

        if t.prompt_modulation:
            # LTX-2.3
            temb_prompt, _ = t.prompt_adaln(
                sigma.flatten(), batch_size=batch_size, hidden_dtype=hidden_states.dtype
            )
            temb_prompt_audio, _ = t.audio_prompt_adaln(
                audio_sigma.flatten(), batch_size=batch_size, hidden_dtype=audio_hidden_states.dtype
            )
            temb_prompt = temb_prompt.view(batch_size, -1, temb_prompt.size(-1))
            temb_prompt_audio = temb_prompt_audio.view(batch_size, -1, temb_prompt_audio.size(-1))
        else:
            temb_prompt = temb_prompt_audio = None

        # 3.2. Global modality cross-attention modulation params
        video_ca_timestep = audio_sigma.flatten() if use_cross_timestep else timestep.flatten()
        video_cross_attn_scale_shift, _ = t.av_cross_attn_video_scale_shift(
            video_ca_timestep,
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )
        video_cross_attn_a2v_gate, _ = t.av_cross_attn_video_a2v_gate(
            video_ca_timestep * timestep_cross_attn_gate_scale_factor,
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )
        video_cross_attn_scale_shift = video_cross_attn_scale_shift.view(
            batch_size, -1, video_cross_attn_scale_shift.shape[-1]
        )
        video_cross_attn_a2v_gate = video_cross_attn_a2v_gate.view(
            batch_size, -1, video_cross_attn_a2v_gate.shape[-1]
        )

        audio_ca_timestep = sigma.flatten() if use_cross_timestep else audio_timestep.flatten()
        audio_cross_attn_scale_shift, _ = t.av_cross_attn_audio_scale_shift(
            audio_ca_timestep,
            batch_size=batch_size,
            hidden_dtype=audio_hidden_states.dtype,
        )
        audio_cross_attn_v2a_gate, _ = t.av_cross_attn_audio_v2a_gate(
            audio_ca_timestep * timestep_cross_attn_gate_scale_factor,
            batch_size=batch_size,
            hidden_dtype=audio_hidden_states.dtype,
        )
        audio_cross_attn_scale_shift = audio_cross_attn_scale_shift.view(
            batch_size, -1, audio_cross_attn_scale_shift.shape[-1]
        )
        audio_cross_attn_v2a_gate = audio_cross_attn_v2a_gate.view(
            batch_size, -1, audio_cross_attn_v2a_gate.shape[-1]
        )

        # 4. Prepare prompt embeddings (LTX-2.0)
        if t.config.use_prompt_embeddings:
            encoder_hidden_states = t.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.size(-1))

            audio_encoder_hidden_states = t.audio_caption_projection(audio_encoder_hidden_states)
            audio_encoder_hidden_states = audio_encoder_hidden_states.view(
                batch_size, -1, audio_hidden_states.size(-1)
            )

        # === 5. Run transformer blocks (RE-OWNED) ===
        spatio_temporal_guidance_blocks = spatio_temporal_guidance_blocks or []
        if len(spatio_temporal_guidance_blocks) > 0 and perturbation_mask is None:
            perturbation_mask = torch.zeros((batch_size,))
        if perturbation_mask is not None and perturbation_mask.ndim == 1:
            perturbation_mask = perturbation_mask[:, None, None]
        all_perturbed = torch.all(perturbation_mask == 0) if perturbation_mask is not None else False
        stg_blocks = set(spatio_temporal_guidance_blocks)

        grad_ckpt = torch.is_grad_enabled() and t.gradient_checkpointing

        # AP2 First-Block-Cache: joint (video, audio) residual. Both streams
        # survive to the output (unlike a dual-stream image arch that strips
        # text), so the cached object is a TUPLE (video_residual, audio_residual)
        # and the reconstruction on a hit restores BOTH streams. Capture the
        # pre-block-loop hidden states (right after proj_in) as the residual base.
        fb_hit = False
        original_video = hidden_states
        original_audio = audio_hidden_states

        for block_idx, block in enumerate(t.transformer_blocks):
            # AP1 block-swap: ensure this block's weights are resident before use.
            if swap_on:
                offloader.wait_for_block(block_idx)

            block_perturbation_mask = perturbation_mask if block_idx in stg_blocks else None
            block_all_perturbed = all_perturbed if block_idx in stg_blocks else False

            # AP3 extension sites (NOT implemented in AP1):
            #   * TREAD: enter/exit a token-routing span here (video tokens only).
            #   * block-dropout: skip the block (both streams identity) with
            #     residual rescale. Both would branch around the block(...) call
            #     below. Left clean intentionally.

            if grad_ckpt:
                hidden_states, audio_hidden_states = t._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    audio_hidden_states,
                    encoder_hidden_states,
                    audio_encoder_hidden_states,
                    temb,
                    temb_audio,
                    video_cross_attn_scale_shift,
                    audio_cross_attn_scale_shift,
                    video_cross_attn_a2v_gate,
                    audio_cross_attn_v2a_gate,
                    temb_prompt,
                    temb_prompt_audio,
                    video_rotary_emb,
                    audio_rotary_emb,
                    video_cross_attn_rotary_emb,
                    audio_cross_attn_rotary_emb,
                    encoder_attention_mask,
                    audio_encoder_attention_mask,
                    None,  # self_attention_mask
                    None,  # audio_self_attention_mask
                    None,  # a2v_cross_attention_mask
                    None,  # v2a_cross_attention_mask
                    not isolate_modalities,  # use_a2v_cross_attention
                    not isolate_modalities,  # use_v2a_cross_attention
                    block_perturbation_mask,
                    block_all_perturbed,
                )
            else:
                hidden_states, audio_hidden_states = block(
                    hidden_states=hidden_states,
                    audio_hidden_states=audio_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    audio_encoder_hidden_states=audio_encoder_hidden_states,
                    temb=temb,
                    temb_audio=temb_audio,
                    temb_ca_scale_shift=video_cross_attn_scale_shift,
                    temb_ca_audio_scale_shift=audio_cross_attn_scale_shift,
                    temb_ca_gate=video_cross_attn_a2v_gate,
                    temb_ca_audio_gate=audio_cross_attn_v2a_gate,
                    temb_prompt=temb_prompt,
                    temb_prompt_audio=temb_prompt_audio,
                    video_rotary_emb=video_rotary_emb,
                    audio_rotary_emb=audio_rotary_emb,
                    ca_video_rotary_emb=video_cross_attn_rotary_emb,
                    ca_audio_rotary_emb=audio_cross_attn_rotary_emb,
                    encoder_attention_mask=encoder_attention_mask,
                    audio_encoder_attention_mask=audio_encoder_attention_mask,
                    self_attention_mask=None,
                    audio_self_attention_mask=None,
                    a2v_cross_attention_mask=None,
                    v2a_cross_attention_mask=None,
                    use_a2v_cross_attention=not isolate_modalities,
                    use_v2a_cross_attention=not isolate_modalities,
                    perturbation_mask=block_perturbation_mask,
                    all_perturbed=block_all_perturbed,
                )

            # AP1 block-swap: prefetch the next swappable block after this one ran.
            if swap_on:
                offloader.submit_move_blocks_forward(block_idx)

            # AP2 FBCache decision after the FIRST block: indicator = the VIDEO
            # stream's residual so far. On a hit, reuse the cached (video, audio)
            # residual pair and skip everything remaining (both the rest of the
            # block loop and its block-swap wait/submit calls -- the offloader's
            # prefetch is forward-only and per-generation, so an early break here
            # simply leaves later blocks' prefetch un-submitted this call, mirroring
            # the FLUX.2 wrapper's FBCache break; the pipeline guards this mode
            # combination as mutually exclusive with Block Swap regardless).
            if fbcache is not None and block_idx == 0:
                indicator = hidden_states - original_video
                if fbcache.use_cache(indicator, int(self._fbcache_step)):
                    cached_video_residual, cached_audio_residual = fbcache.get()
                    hidden_states = original_video + cached_video_residual
                    audio_hidden_states = original_audio + cached_audio_residual
                    fb_hit = True
                    break

        # AP2 FBCache miss: store the full (video, audio) residual pair -- the
        # exact tensors fed to the stage-6 norm_out/proj_out calls on a miss, so a
        # future hit reproduces them exactly.
        if fbcache is not None and not fb_hit:
            fbcache.store((hidden_states - original_video, audio_hidden_states - original_audio))

        # === 6. Output layers (including unpatchification) ===
        scale_shift_values = t.scale_shift_table[None, None] + embedded_timestep[:, :, None]
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

        hidden_states = t.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift
        output = t.proj_out(hidden_states)

        audio_scale_shift_values = t.audio_scale_shift_table[None, None] + audio_embedded_timestep[:, :, None]
        audio_shift, audio_scale = audio_scale_shift_values[:, :, 0], audio_scale_shift_values[:, :, 1]

        audio_hidden_states = t.audio_norm_out(audio_hidden_states)
        audio_hidden_states = audio_hidden_states * (1 + audio_scale) + audio_shift
        audio_output = t.audio_proj_out(audio_hidden_states)

        if not return_dict:
            return (output, audio_output)
        return AudioVisualModelOutput(sample=output, audio_sample=audio_output)

    # ------------------------------------------------------------------
    # Passthroughs (LoRA save/load + block-swap conductor + diffusers pipeline)
    # ------------------------------------------------------------------
    def to(self, *args, **kwargs):
        self.transformer.to(*args, **kwargs)
        try:
            self.device = next(self.transformer.parameters()).device
        except StopIteration:
            pass
        return self

    def __getattr__(self, name: str):
        # nn.Module overrides __getattr__; delegate anything not found on the
        # wrapper (incl. the registered ``transformer`` submodule miss during
        # __init__ before assignment) to the inner transformer.
        try:
            return super().__getattr__(name)
        except AttributeError:
            transformer = self.__dict__.get("_modules", {}).get("transformer", None)
            if transformer is not None and name != "transformer":
                return getattr(transformer, name)
            raise

    def state_dict(self, *args, **kwargs):
        return self.transformer.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.transformer.load_state_dict(*args, **kwargs)
