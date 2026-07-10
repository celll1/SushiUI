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


def _gather_video_rope(rope_tuple, idx: torch.Tensor):
    """Gather the kept-token rows out of a ``(cos, sin)`` RoPE pair.

    LTX-2.3's ``LTX2AudioVideoRotaryPosEmbed`` produces either shape depending on
    ``rope_type`` (diffusers default: ``"interleaved"``):
      * interleaved: ``[B, N, D]``       (token axis = dim 1, D = full proj dim)
      * split:       ``[B, H, N, D//2]`` (token axis = dim 2, per-head)
    Handles BOTH so a future/alternate ``rope_type`` config does not silently
    mis-gather.
    """
    cos, sin = rope_tuple
    if cos.dim() == 3:
        return cos.index_select(1, idx), sin.index_select(1, idx)
    if cos.dim() == 4:
        return cos.index_select(2, idx), sin.index_select(2, idx)
    raise ValueError(
        f"Ltx2BlockLoopWrapper: unexpected LTX-2.3 video RoPE ndim={cos.dim()} "
        "(expected 3 [interleaved] or 4 [split]); TREAD gather not implemented "
        "for this shape."
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
        # AP3 — DiT-BlockSkip (arXiv 2603.20755) folded-precompute LoRA memory
        # reduction (training-only). Mutually exclusive with TREAD (both
        # restructure the block loop) and with Block Swap (folding requires all
        # blocks resident); enforced in ``attach_blockskip`` and ``base_trainer``.
        self._blockskip_config = None
        # Training-free reference-style transfer (StyleAligned/VSP-style KV
        # injection over attn1 video self-attention; see
        # ``core.inference.style_ltx2`` module docstring). INFERENCE-ONLY
        # (asserted in ``_custom_forward``) and mutually exclusive with FBCache
        # / Spectrum (both disabled by the caller, ``pipeline_backends/ltx2.py``,
        # whenever style is requested) and with Block Swap (style forces Block
        # Swap off; see ``attach_style``). ``_style_processors`` is the list of
        # installed ``StyleLtx2Attn1Processor`` instances (patched directly onto
        # the INNER transformer's ``attn1`` modules, independent of whether this
        # wrapper's fast path or custom path runs the block loop); the wrapper
        # only needs to drive the per-step ref-capture sub-pass and stamp the
        # capture/inject ``StyleContext`` onto them.
        self._style_processors = None
        self._style_cfg = None
        self._style_ref_x0 = None     # [1, S_ref, C_in] packed ref latent (pre proj_in), float32
        self._style_eps_ref = None    # same shape, fixed per-generation noise
        self._style_step_idx = 0
        self._style_total_steps = 1

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

    def attach_tread(self, config: Optional[dict]) -> None:
        """Attach (or clear with None) the AP3 TREAD token-routing config.

        TREAD (arXiv 2501.04765) is TRAINING-ONLY (``_custom_forward`` additionally
        gates on ``self.training`` and ``torch.is_grad_enabled()``, so this is
        doubly safe). The caller (``ltx2_ops.train_step``) is responsible for
        attaching a dict with keys ``drop_ratio`` / ``start_block`` / ``end_block``
        before the training forward and clearing it (``attach_tread(None)``) in a
        ``finally`` so sampling/validation (which reuse the same transformer via
        ``trainer.ltx2_pipeline`` -- note: the pipeline actually holds its own
        direct reference to the INNER unwrapped transformer, so it never even
        calls through this wrapper -- but the clear is kept as defense in depth)
        always run the full network on all tokens.
        """
        assert config is None or self._blockskip_config is None, (
            "Ltx2BlockLoopWrapper: TREAD (_tread_router) and BlockSkip "
            "(_blockskip_config) must never be attached simultaneously (both "
            "restructure the block loop; guarded in base_trainer / ltx2_ops)."
        )
        self._tread_router = config

    def attach_blockskip(self, config: Optional[dict]) -> None:
        """Attach (or clear with None) the AP3 DiT-BlockSkip config.

        DiT-BlockSkip (arXiv 2603.20755) is a TRAINING-ONLY, LoRA-only memory
        reduction: a no_grad full pass captures the residual DELTA over the
        skipped front/back block spans (both the video and audio streams), and
        the gradient pass runs ONLY the middle blocks, re-adding the deltas at
        the span boundaries. Backprop is confined to the middle blocks, so the
        skipped blocks retain no backward activations.

        ``config`` keys: ``front`` (int, blocks skipped at the start) and
        ``back`` (int, blocks skipped at the end). Gated in ``_custom_forward``
        on ``self.training AND torch.is_grad_enabled()`` (never fires during
        sampling/validation). Mutually exclusive with TREAD (``_tread_router``)
        and requires Block Swap to be off (``base_trainer`` enforces
        ``blocks_to_swap == 0`` when ``blockskip_enable`` is set).
        """
        assert config is None or self._tread_router is None, (
            "Ltx2BlockLoopWrapper: BlockSkip (_blockskip_config) and TREAD "
            "(_tread_router) must never be attached simultaneously (both "
            "restructure the block loop; guarded in base_trainer / ltx2_ops)."
        )
        self._blockskip_config = config

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

    def attach_style(
        self,
        processors: Optional[list] = None,
        cfg: Optional[Any] = None,
        ref_x0: Optional[torch.Tensor] = None,
        eps_ref: Optional[torch.Tensor] = None,
    ) -> None:
        """Attach (or clear with all ``None``) training-free reference-style
        transfer. ``processors`` are the ``StyleLtx2Attn1Processor`` instances
        already installed onto the INNER transformer's ``attn1`` modules by
        ``core.inference.style_ltx2.install_ltx2_style_processors`` (the caller,
        ``pipeline_backends/ltx2.py``, owns install/restore around the whole
        generation); this wrapper only stamps capture/inject contexts onto them
        per step and drives the ref-capture sub-pass. ``ref_x0``/``eps_ref`` are
        the packed (pre-``proj_in``) one-frame reference video latent and its
        fixed noise draw (see ``core.inference.style_ltx2`` module docstring for
        the still -> single-frame-video-latent construction).

        Style transfer is INFERENCE-ONLY (asserted in ``_custom_forward``) and
        mutually exclusive with FBCache / Spectrum (the caller disables both
        whenever style is requested, mirroring every other arch's audited
        finding that a trajectory-redundancy skip desyncs the per-block style
        capture/inject store). Block Swap must also be off (the caller forces
        ``blocks_to_swap = 0`` whenever style is requested); asserted here
        defensively.
        """
        assert processors is None or self._fbcache is None, (
            "Ltx2BlockLoopWrapper: style transfer and FBCache must never be attached "
            "simultaneously (a cache hit skips the block loop, desyncing the "
            "per-block style capture/inject store; guarded in ltx2.py)."
        )
        assert processors is None or self._spectrum_video is None, (
            "Ltx2BlockLoopWrapper: style transfer and Spectrum must never be attached "
            "simultaneously (a forecast skip bypasses the block loop, desyncing the "
            "per-block style capture/inject store; guarded in ltx2.py)."
        )
        assert processors is None or not (
            self._block_offloader is not None and getattr(self._block_offloader, "blocks_to_swap", 0) > 0
        ), (
            "Ltx2BlockLoopWrapper: style transfer and Block Swap must never be attached "
            "simultaneously (the ref-capture sub-pass does not thread the offloader's "
            "wait/submit calls; guarded in ltx2.py, which forces blocks_to_swap=0 "
            "whenever style transfer is requested)."
        )
        self._style_processors = processors
        self._style_cfg = cfg
        self._style_ref_x0 = ref_x0
        self._style_eps_ref = eps_ref
        self._style_step_idx = 0
        self._style_total_steps = 1

    def _any_feature_active(self) -> bool:
        swap_on = (
            self._block_offloader is not None
            and getattr(self._block_offloader, "blocks_to_swap", 0) > 0
        )
        return bool(swap_on or self._fbcache is not None or self._spectrum_video is not None
                    or self._tread_router is not None or self._block_dropout is not None
                    or self._blockskip_config is not None or self._style_cfg is not None)

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

        # === Training-free reference-style transfer: REF CAPTURE sub-pass ===
        # Runs BEFORE the real (stage 5) block loop below so the per-block
        # StyleContext store is fully populated by the time attn1's patched
        # processors (installed on t.transformer_blocks[*].attn1 by
        # core.inference.style_ltx2.install_ltx2_style_processors, independent
        # of this wrapper) run for the target. See core.inference.style_ltx2
        # module docstring for the full design (CFG row split, still -> video
        # ref encoding, RoPE reuse, FBCache/Spectrum/Block-Swap interop).
        if self._style_cfg is not None:
            assert not torch.is_grad_enabled(), (
                "Ltx2BlockLoopWrapper: style transfer (_style_cfg) must not be "
                "attached while autograd is enabled (inference-only feature)."
            )
            self._style_run_capture_and_arm_inject(
                t, sigma, hidden_states, audio_hidden_states, encoder_hidden_states, audio_encoder_hidden_states,
                temb, temb_audio,
                video_cross_attn_scale_shift, audio_cross_attn_scale_shift,
                video_cross_attn_a2v_gate, audio_cross_attn_v2a_gate,
                temb_prompt, temb_prompt_audio,
                video_rotary_emb, audio_rotary_emb,
                video_cross_attn_rotary_emb, audio_cross_attn_rotary_emb,
                encoder_attention_mask, audio_encoder_attention_mask,
                isolate_modalities,
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

        # AP3 DiT-BlockSkip (arXiv 2603.20755): OFF by default (_blockskip_config
        # is None). Training-only -- gated on self.training AND autograd being
        # enabled (sampling/validation never attach a config, and the pipeline
        # sampling path calls the INNER transformer directly, never this
        # wrapper). Mutually exclusive with TREAD (_tread_router, asserted in
        # attach_blockskip/attach_tread) and with Block Swap (base_trainer
        # enforces blocks_to_swap == 0 when blockskip_enable is set). Folds the
        # skipped front/back block spans (BOTH the video and audio streams) via
        # a no_grad delta capture + a gradient pass over only the middle blocks;
        # see ``_blockskip_forward`` for the two-pass fold. Short-circuits the
        # entire stage-5 loop below (TREAD routing, FBCache and Block Swap are
        # therefore never reached on this path).
        blockskip = self._blockskip_config if (self.training and torch.is_grad_enabled()) else None
        if blockskip is not None:
            assert self._tread_router is None, (
                "Ltx2BlockLoopWrapper: BlockSkip (_blockskip_config) and TREAD "
                "(_tread_router) must never be attached simultaneously (both "
                "restructure the block loop; guarded in attach_blockskip/attach_tread)."
            )
            assert fbcache is None, (
                "Ltx2BlockLoopWrapper: BlockSkip is training-only and FBCache is "
                "inference-only; they must never be attached at the same time."
            )
            assert not swap_on, (
                "Ltx2BlockLoopWrapper: BlockSkip requires blocks_to_swap=0 "
                "(enforced in base_trainer); the block-swap conductor cannot be "
                "attached alongside BlockSkip's folded precompute."
            )
            hidden_states, audio_hidden_states = self._blockskip_forward(
                blockskip, t, hidden_states, audio_hidden_states,
                encoder_hidden_states, audio_encoder_hidden_states,
                temb, temb_audio,
                video_cross_attn_scale_shift, audio_cross_attn_scale_shift,
                video_cross_attn_a2v_gate, audio_cross_attn_v2a_gate,
                temb_prompt, temb_prompt_audio,
                video_rotary_emb, audio_rotary_emb,
                video_cross_attn_rotary_emb, audio_cross_attn_rotary_emb,
                encoder_attention_mask, audio_encoder_attention_mask,
                isolate_modalities, grad_ckpt,
            )
            return self._finish_stage6(
                t, hidden_states, audio_hidden_states,
                embedded_timestep, audio_embedded_timestep, return_dict,
            )

        # AP3 TREAD token routing (arXiv 2501.04765): OFF by default (_tread_router
        # is None). Training-only -- gated on self.training AND autograd being
        # enabled; sampling/validation never attach a route config (cleared by
        # ltx2_ops.train_step's `finally`, and the pipeline sampling path calls the
        # INNER transformer directly, never this wrapper), so routing structurally
        # cannot fire outside a training forward.
        #
        # Exactness for LTX-2.3 video (proven in the AP3 feasibility study): the
        # training timestep is a per-SAMPLE scalar, so temb / temb_audio /
        # temb_prompt / temb_ca_* are all [B, 1, D] and BROADCAST over every video
        # token identically -- none of them are per-token. Gathering a token
        # subset for blocks [start_block, end_block) is therefore EXACT without
        # gathering any modulation tensor.
        #
        # Only the VIDEO stream is routed. Audio flows through every block
        # unchanged (a separate token axis; not gathered). The wrapper's training
        # call always sets isolate_modalities=True (see ltx2_ops.train_step),
        # which disables use_a2v_cross_attention / use_v2a_cross_attention -- the
        # ONLY consumers of `ca_video_rotary_emb` / `ca_audio_rotary_emb` -- so
        # those two RoPE tuples are dead code during training and are passed
        # through UNCHANGED (never gathered). attn2 (video-text cross-attention)
        # is called with query_rotary_emb=None (no RoPE at all), so the gathered
        # video subset needs no cross-attention RoPE either. The ONLY RoPE tensor
        # that must be gathered is `video_rotary_emb` (attn1 video self-attention).
        tread = self._tread_router if (self.training and torch.is_grad_enabled()) else None
        num_video_tokens = hidden_states.shape[1]
        route_active = False
        kept_idx = None
        start_b = end_b = 0
        if tread is not None:
            start_b = int(tread.get("start_block", 0))
            end_b = int(tread.get("end_block", 0))
            drop_ratio = float(tread.get("drop_ratio", 0.0))
            num_blocks = len(t.transformer_blocks)
            if not (0 <= start_b < end_b <= num_blocks and 0.0 < drop_ratio < 1.0):
                if not getattr(self, "_warned_tread_span", False):
                    print(f"[LTX2 TREAD] WARNING: invalid route "
                          f"(start={start_b}, end={end_b}, drop={drop_ratio}, "
                          f"blocks={num_blocks}); routing disabled")
                    self._warned_tread_span = True
            elif num_video_tokens <= 1:
                if not getattr(self, "_warned_tread_tokens", False):
                    print(f"[LTX2 TREAD] WARNING: video token count "
                          f"{num_video_tokens} <= 1; routing disabled")
                    self._warned_tread_tokens = True
            else:
                route_active = True

        if route_active:
            # Randomness sampled ONCE per step, OUTSIDE any checkpointed callable
            # (the per-block gradient-checkpointing recompute below only re-runs
            # the single block's forward given tensors fixed at this point in the
            # enclosing loop -- it never re-executes this selection). This is the
            # correctness pin: sampling kept_idx inside block.forward or inside
            # the checkpointed callable would let recompute pick a different token
            # subset and silently corrupt gradients.
            from core.training.token_routing import select_kept_indices
            kept_idx = select_kept_indices(num_video_tokens, drop_ratio, hidden_states.device)

        video_rotary_emb_full = video_rotary_emb  # restored for out-of-span blocks

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

            # AP3 TREAD: enter the routed span. Snapshot the full video stream
            # (identity/residual transport for bypassed tokens) and gather the
            # kept subset + its video RoPE rows. Audio / text streams and all
            # modulation tensors are untouched (see the exactness note above).
            if route_active and block_idx == start_b:
                from core.training.token_routing import gather_tokens
                x_full = hidden_states
                hidden_states = gather_tokens(hidden_states, kept_idx)
                video_rotary_emb = _gather_video_rope(video_rotary_emb_full, kept_idx)

            # block-dropout (NOT implemented in this phase): would skip the block
            # (both streams identity) with residual rescale, branching around the
            # block(...) call below. Left clean intentionally.

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

            # AP3 TREAD: exit the routed span. Scatter the processed kept tokens
            # back into the full stream (bypassed tokens keep the pre-span values
            # captured in x_full -- the paper's identity/residual transport) and
            # restore the full video RoPE for any remaining post-span blocks.
            if route_active and block_idx == end_b - 1:
                from core.training.token_routing import scatter_tokens
                hidden_states = scatter_tokens(x_full, hidden_states, kept_idx)
                video_rotary_emb = video_rotary_emb_full

            # AP2 FBCache decision after the FIRST block: indicator = the VIDEO
            # stream's residual so far. On a hit, reuse the cached (video, audio)
            # residual pair and skip everything remaining (both the rest of the
            # block loop and its block-swap wait/submit calls -- the offloader's
            # prefetch is forward-only and per-generation, so an early break here
            # simply leaves later blocks' prefetch un-submitted this call, mirroring
            # the FLUX.2 wrapper's FBCache break; the pipeline guards this mode
            # combination as mutually exclusive with Block Swap regardless). FBCache
            # is inference-only and TREAD is training-only (structurally mutually
            # exclusive via the grad-enabled gates above), so `original_video`
            # (captured pre-loop, full token count) never mismatches a
            # TREAD-reduced `hidden_states` here.
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

        # Style transfer: disarm the context on every installed processor right
        # after the (real) block loop that consumed it. Defense in depth only --
        # the authoritative disarm/restore is the caller's ``finally`` block in
        # ``pipeline_backends/ltx2.py`` (mirrors the FBCache/Spectrum pattern:
        # restore/patch-removal must run on exception too, else style state
        # leaks into the next generation -- see the FLUX.2 audit finding).
        if self._style_processors is not None:
            from core.inference.style_ltx2 import set_ltx2_style_context
            set_ltx2_style_context(self._style_processors, None)

        # === 6. Output layers (including unpatchification) ===
        return self._finish_stage6(
            t, hidden_states, audio_hidden_states,
            embedded_timestep, audio_embedded_timestep, return_dict,
        )

    @staticmethod
    def _finish_stage6(t, hidden_states, audio_hidden_states,
                        embedded_timestep, audio_embedded_timestep, return_dict):
        """Stage 6 (output layers, incl. unpatchification) -- shared tail for the
        normal block loop AND the BlockSkip fold (``_blockskip_forward``), so
        both paths funnel through the IDENTICAL final projection code."""
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

    def _style_run_capture_and_arm_inject(
        self, t, sigma, hidden_states, audio_hidden_states, encoder_hidden_states, audio_encoder_hidden_states,
        temb, temb_audio,
        video_cross_attn_scale_shift, audio_cross_attn_scale_shift,
        video_cross_attn_a2v_gate, audio_cross_attn_v2a_gate,
        temb_prompt, temb_prompt_audio,
        video_rotary_emb, audio_rotary_emb,
        video_cross_attn_rotary_emb, audio_cross_attn_rotary_emb,
        encoder_attention_mask, audio_encoder_attention_mask,
        isolate_modalities,
    ):
        """Run the ref-capture sub-pass (batch=1, the style reference re-noised
        to THIS step's sigma) through every ``transformer_block`` in capture
        mode, then arm the SAME installed ``attn1`` processors in inject mode
        for the upcoming real (target) block loop. See
        ``core.inference.style_ltx2`` module docstring for the full design.

        Reuses the TARGET's own stage-1-4 conditioning tensors (temb, cross-attn
        modulation, RoPE), sliced to a single batch row -- valid because
        LTX-2.3's per-step timestep/sigma is identical across every CFG row (see
        ``core.inference.style_ltx2``'s CFG-composition note) and the ref is
        conceptually "the same generation, different content" at the SAME
        position grid (frame-0 spatial tokens, see the still -> video-latent ref
        note in that module).
        """
        from core.inference.reference_style import StyleContext
        from core.inference.style_ltx2 import set_ltx2_style_context

        cfg = self._style_cfg
        processors = self._style_processors
        ref_x0 = self._style_ref_x0
        eps_ref = self._style_eps_ref
        if processors is None or ref_x0 is None or eps_ref is None:
            return

        step_idx = int(self._style_step_idx)
        total_steps = int(self._style_total_steps)
        progress = cfg.step_progress(step_idx, total_steps)
        if not cfg.is_step_active(step_idx, total_steps):
            set_ltx2_style_context(processors, None)
            return

        n = ref_x0.shape[1]
        dtype = hidden_states.dtype
        device = hidden_states.device

        # Sigma for re-noising: the LTX-2.3 pipeline passes `sigma == timestep`
        # (the raw scheduler timestep, see pipeline_ltx2.py's denoise loop:
        # `sigma=timestep`) into the transformer; dividing by the scheduler's
        # num_train_timesteps (1000) recovers the flow-matching sigma in [0, 1]
        # for `x_t = (1 - sigma) * x0 + sigma * eps` -- the IDENTICAL /1000
        # convention already used by the FLUX.2 style wiring
        # (pipeline_backends/flux2.py's `_flux2_style_step`).
        sigma_now = float(sigma.flatten()[0].item()) / 1000.0
        sigma_now = max(0.0, min(1.0, sigma_now))
        ref_t = (1.0 - sigma_now) * ref_x0.to(device=device, dtype=torch.float32) \
            + sigma_now * eps_ref.to(device=device, dtype=torch.float32)
        ref_t = ref_t.to(dtype=dtype)

        ref_hidden_states = t.proj_in(ref_t)
        if ref_hidden_states.shape[1] != n:
            raise RuntimeError(
                f"Ltx2BlockLoopWrapper style transfer: proj_in changed the ref "
                f"token count ({n} -> {ref_hidden_states.shape[1]}); ref/target "
                "token layout assumption violated."
            )

        # CFG row split (mirrors core.inference.style_ltx2's identical
        # derivation): under CFG the pipeline concatenates
        # [uncond rows..., cond rows...], so row 0 is the NEGATIVE-prompt
        # conditioning. The ref-capture pass must run under the POSITIVE (cond)
        # conditioning that the cond target's own forward attends -- slicing
        # row 0 unconditionally would evolve the reference through the deep
        # blocks under uncond cross-attn embeds instead. `cond_row` is the
        # first cond row (0 when there is no CFG doubling, in which case every
        # row already IS cond).
        target_batch = hidden_states.shape[0]
        cond_row = (target_batch // 2) if (target_batch >= 2 and target_batch % 2 == 0) else 0

        def _b0(x):
            if x is None:
                return None
            if isinstance(x, tuple):
                return tuple(_b0(e) for e in x)
            return x[cond_row:cond_row + 1]

        ref_audio_hidden_states = _b0(audio_hidden_states)
        ref_encoder_hidden_states = _b0(encoder_hidden_states)
        ref_audio_encoder_hidden_states = _b0(audio_encoder_hidden_states)
        ref_temb = _b0(temb)
        ref_temb_audio = _b0(temb_audio)
        ref_video_ca_scale_shift = _b0(video_cross_attn_scale_shift)
        ref_audio_ca_scale_shift = _b0(audio_cross_attn_scale_shift)
        ref_video_ca_gate = _b0(video_cross_attn_a2v_gate)
        ref_audio_ca_gate = _b0(audio_cross_attn_v2a_gate)
        ref_temb_prompt = _b0(temb_prompt)
        ref_temb_prompt_audio = _b0(temb_prompt_audio)
        ref_audio_rotary_emb = _b0(audio_rotary_emb)
        ref_ca_video_rotary_emb = _b0(video_cross_attn_rotary_emb)
        ref_ca_audio_rotary_emb = _b0(audio_cross_attn_rotary_emb)
        ref_encoder_attention_mask = _b0(encoder_attention_mask)
        ref_audio_encoder_attention_mask = _b0(audio_encoder_attention_mask)

        rope_device = video_rotary_emb[0].device
        idx = torch.arange(n, device=rope_device)
        ref_video_rotary_emb = _gather_video_rope(video_rotary_emb, idx)

        capture_ctx = StyleContext(mode="capture", config=cfg, progress=progress)
        capture_ctx.img_start = 0
        capture_ctx.img_end = n
        set_ltx2_style_context(processors, capture_ctx)

        with torch.no_grad():
            hs, ahs = ref_hidden_states, ref_audio_hidden_states
            for block in t.transformer_blocks:
                hs, ahs = block(
                    hidden_states=hs,
                    audio_hidden_states=ahs,
                    encoder_hidden_states=ref_encoder_hidden_states,
                    audio_encoder_hidden_states=ref_audio_encoder_hidden_states,
                    temb=ref_temb,
                    temb_audio=ref_temb_audio,
                    temb_ca_scale_shift=ref_video_ca_scale_shift,
                    temb_ca_audio_scale_shift=ref_audio_ca_scale_shift,
                    temb_ca_gate=ref_video_ca_gate,
                    temb_ca_audio_gate=ref_audio_ca_gate,
                    temb_prompt=ref_temb_prompt,
                    temb_prompt_audio=ref_temb_prompt_audio,
                    video_rotary_emb=ref_video_rotary_emb,
                    audio_rotary_emb=ref_audio_rotary_emb,
                    ca_video_rotary_emb=ref_ca_video_rotary_emb,
                    ca_audio_rotary_emb=ref_ca_audio_rotary_emb,
                    encoder_attention_mask=ref_encoder_attention_mask,
                    audio_encoder_attention_mask=ref_audio_encoder_attention_mask,
                    self_attention_mask=None,
                    audio_self_attention_mask=None,
                    a2v_cross_attention_mask=None,
                    v2a_cross_attention_mask=None,
                    use_a2v_cross_attention=not isolate_modalities,
                    use_v2a_cross_attention=not isolate_modalities,
                    perturbation_mask=None,
                    all_perturbed=False,
                )

        # Arm the SAME processors in inject mode for the real (target) block
        # loop that runs right after this method returns (either the wrapper's
        # own stage-5 loop, or -- when style is the ONLY active feature and
        # _any_feature_active() still routes through _custom_forward because
        # _style_cfg is checked there too -- the same loop below).
        inject_ctx = StyleContext(mode="inject", config=cfg, store=capture_ctx.store, progress=progress)
        inject_ctx.img_start = 0
        inject_ctx.img_end = n
        set_ltx2_style_context(processors, inject_ctx)

    def _blockskip_forward(
        self, cfg, t, hidden_states, audio_hidden_states,
        encoder_hidden_states, audio_encoder_hidden_states,
        temb, temb_audio,
        video_cross_attn_scale_shift, audio_cross_attn_scale_shift,
        video_cross_attn_a2v_gate, audio_cross_attn_v2a_gate,
        temb_prompt, temb_prompt_audio,
        video_rotary_emb, audio_rotary_emb,
        video_cross_attn_rotary_emb, audio_cross_attn_rotary_emb,
        encoder_attention_mask, audio_encoder_attention_mask,
        isolate_modalities, grad_ckpt,
    ):
        """DiT-BlockSkip two-pass fold over the DUAL (video, audio) stream
        (arXiv 2603.20755), ported from the Anima image-DiT implementation
        (``anima_models.py: _blockskip_forward``).

        Pass 1 (no_grad, full network, SAME module state -- LoRA active -- as
        pass 2): capture the residual feature DELTA for BOTH streams over each
        skipped span:
          video_delta_front = v_n - v_0        (front span [0, n))
          video_delta_back  = v_L - v_{L-m}    (back span  [L-m, L))
        and identically for the audio stream, where v_i / a_i are the
        video/audio stream values fed INTO block i (v_L/a_L are the values fed
        to stage 6).

        Pass 2 (gradient, middle blocks [n, L-m) only):
          hs  = hidden_states       + video_delta_front   (== v_n, exact)
          ahs = audio_hidden_states + audio_delta_front   (== a_n, exact)
          hs, ahs = middle_blocks(hs, ahs)                (LoRA-trained, grad flows here)
          hs  = hs  + video_delta_back                    (== v_L when middle is unchanged)
          ahs = ahs + audio_delta_back

        Because pass 1 runs with the SAME (LoRA-active) weights as pass 2, the
        reconstruction is EXACT for any front/back span -- not merely exact
        under the paper's frozen-base assumption (audited finding from the
        Anima port).

        Every block still receives the FULL token stream (BlockSkip does not
        gather tokens, unlike TREAD) -- video_rotary_emb / all temb* /
        encoder tensors are passed UNCHANGED to every block, front through
        back. The audio stream is folded symmetrically with the video stream
        (captured + re-added at the same span boundaries) even though
        ``isolate_modalities=True`` makes the two streams independent within a
        block during training -- this keeps the block call signature
        satisfied and the (unused) audio prediction well-formed; it is never
        special-cased away.

        The no_grad pass never checkpoints (there is no backward through it);
        the gradient pass reuses the wrapper's usual per-block
        ``t._gradient_checkpointing_func`` path when ``grad_ckpt`` is set,
        exactly as the normal block loop does.
        """
        blocks = t.transformer_blocks
        num_blocks = len(blocks)
        front = int(cfg["front"])
        back = int(cfg["back"])
        lo = front
        hi = num_blocks - back
        if not (0 <= lo <= hi <= num_blocks):
            raise ValueError(
                f"Ltx2BlockLoopWrapper BlockSkip: invalid span front={front} "
                f"back={back} for {num_blocks} blocks (resolved lo={lo}, hi={hi})"
            )

        def _run_block(block_idx, block, hs, ahs, use_ckpt):
            if use_ckpt:
                return t._gradient_checkpointing_func(
                    block,
                    hs,
                    ahs,
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
                    None,  # perturbation_mask (STG is an inference-only feature)
                    False,  # all_perturbed
                )
            return block(
                hidden_states=hs,
                audio_hidden_states=ahs,
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
                perturbation_mask=None,
                all_perturbed=False,
            )

        # Pass 1: frozen (LoRA-active) full forward under no_grad, capturing
        # the span-boundary features for BOTH streams.
        with torch.no_grad():
            v0, a0 = hidden_states, audio_hidden_states
            vt, at = v0, a0
            v_lo = a_lo = None
            v_hi = a_hi = None
            for i, block in enumerate(blocks):
                if i == lo:
                    v_lo, a_lo = vt, at
                if i == hi:
                    v_hi, a_hi = vt, at
                vt, at = _run_block(i, block, vt, at, use_ckpt=False)
            v_L, a_L = vt, at
            if v_lo is None:       # front == 0 (no front skip)
                v_lo, a_lo = v0, a0
            if v_hi is None:       # back == 0 (no back skip): hi == num_blocks
                v_hi, a_hi = v_L, a_L
            video_delta_front = (v_lo - v0).detach()
            audio_delta_front = (a_lo - a0).detach()
            video_delta_back = (v_L - v_hi).detach()
            audio_delta_back = (a_L - a_hi).detach()

        # Pass 2: gradient forward over ONLY the middle blocks [lo, hi). LoRA
        # is trained exclusively on these blocks; the skipped spans retain no
        # backward activations.
        hs = hidden_states + video_delta_front
        ahs = audio_hidden_states + audio_delta_front
        for i in range(lo, hi):
            hs, ahs = _run_block(i, blocks[i], hs, ahs, use_ckpt=grad_ckpt)
        hs = hs + video_delta_back
        ahs = ahs + audio_delta_back
        return hs, ahs

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
