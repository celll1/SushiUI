"""MiniMaxH3ArchHandler — arch handler for arch "minimax_h3" (MiniMax-H3, joint
video + audio single-stream packed DiT). Thin orchestrator; the math lives in
``ops/minimax_h3_ops.py`` and the packed-sequence assembly is shared with
generation (``core/models/minimax_h3/h3_pipeline_ops.py``).

Two declarative fields carry everything the shared video machinery needs:

* ``temporal`` = ``MINIMAX_H3_TEMPORAL`` — the 17n+5 clip grid, the 22-frame VAE
  decode floor, the measured ``latent_frames`` closed form and the FIXED 24 fps
  that makes clip sampling timestamp-based instead of index-based;
* ``clip_vae_tiling_policy`` — the pinned VAE tiling policy, DERIVED from the
  loader's own ``MINIMAX_H3_VAE_TILING_POLICY`` (the dict the generation load
  passes to ``vae.enable_tiling``), so a cached latent can never be served to a
  generation that tiled differently. Measured stakes: rel-RMS 0.355 at 384x384
  (K0.5) and 0.0952 at 640x384 (Phase 0T) from flipping those flags alone.

Aligned Full is experimental; one API update and save passed, while quality and
GPU optimizer-resume acceptance remain pending.
"""

from __future__ import annotations

from core.training.arch.base_arch import (
    ArchHandler, PHASE2_PENDING, SampleContext, TrainStepContext,
    QUANTIZED_ADDITIVE_SHIPPED, declare_adapter_capability, resolve_scope_csv,
)
from core.training.components.wiring import MINIMAX_H3_TEMPORAL, MINIMAX_H3_WIRING
from core.training.ops.minimax_h3_ops import (
    MINIMAX_H3_AUDIO_PREP_VERSION,
    minimax_h3_vae_tiling_token,
)


class MiniMaxH3ArchHandler(ArchHandler):
    def __init__(self, trainer=None):
        super().__init__(trainer)
        if trainer is not None:
            self.resolve_wiring(trainer)

    def apply_vae_swap(self, trainer, resolved, module=None):
        from core.models.minimax_h3.vae_training import apply_swap
        return apply_swap(trainer, resolved, handler=self, module=module)

    def check_vae_compatibility(self, facts, *, trainer=None, base_model_path=None):
        from core.models.components.video_latent_adapter import codec_temporal_policy
        if facts.get("pixel_channels", 3) != 3 or facts.get("output_channels", 3) != 3:
            return False, "H3 VAE swap requires RGB encode and decode channels"
        scale, temporal, rank = facts.get("scale_factor"), facts.get("scale_temporal"), facts.get("ndim")
        if (rank not in (4, 5) or type(scale) is not int or scale < 1
                or type(temporal) is not int or temporal < 1):
            return False, "H3 VAE swap requires declared RGB video/image codec geometry"
        if rank == 4 and temporal != 1:
            return False, "A 2D H3 target codec must preserve independent frames"
        try:
            codec_temporal_policy(rank, facts.get("vae_class"))
        except ValueError as exc:
            return False, str(exc)
        if scale % self.wiring.vae_scale_factor and self.wiring.vae_scale_factor % scale:
            return False, "H3 VAE swap requires integer spatial scale relative to its native codec"
        return True, None

    def calibrate_before_training(self, trainer, datasets, bucket_manager):
        from core.models.minimax_h3.vae_training import calibrate
        calibrate(trainer, datasets)

    def resolve_wiring(self, trainer):
        from dataclasses import replace
        from math import lcm
        from core.models.components.latent_adapter import topology_alignment
        from core.models.components.video_latent_adapter import NormalizedVideoVAE, temporal_spec_for_codec
        model, vae = getattr(trainer, "transformer", None), getattr(trainer, "vae", None)
        if model is None or not isinstance(vae, NormalizedVideoVAE):
            return self.wiring
        inner = getattr(model, "transformer", model)
        if getattr(inner, "_video_codec_contract", None) != vae.contract:
            raise ValueError("H3 training model and video codec contracts disagree")
        contract = vae.contract
        h, w = topology_alignment(getattr(inner, "_residual_unet_spec", None))
        self.pixel_align = contract.image.spatial_scale*2*lcm(h, w)
        spec = temporal_spec_for_codec(MINIMAX_H3_TEMPORAL, contract)
        minimum = max(spec.min_decodable_frames, spec.frame_multiple+spec.frame_offset)
        lengths = tuple(sorted({max(minimum,
            ((n-spec.frame_offset+spec.frame_multiple-1)//spec.frame_multiple)*spec.frame_multiple+spec.frame_offset)
            for n in MINIMAX_H3_TEMPORAL.default_clip_lengths}))
        self.temporal = replace(spec, pixel_align=self.pixel_align,
            default_clip_lengths=lengths, suggested_min_frames=minimum)
        self.clip_vae_tiling_policy = None
        return self.wiring.replace(latent_channels=contract.image.latent_channels,
            vae_scale_factor=contract.image.spatial_scale, vae_scale_temporal=contract.temporal_scale,
            vae_norm="identity", vae_norm_pack=1)

    name = "minimax_h3"
    wiring = MINIMAX_H3_WIRING
    adapter_capability = declare_adapter_capability(
        "minimax_h3",
        additive_family=True,
        # Generation takes LoHa/LoKr; training does not, and the gate is this
        # architecture's own rather than the general Phase 2 step.
        additive_gated=True,
        initial_dora="deferred",
        additive_reason=PHASE2_PENDING,
        quantized_base_additive_family=True,
        quantized_base_reason=(
            f"{QUANTIZED_ADDITIVE_SHIPPED}. Here that is the whole DiT block "
            f"stack, whose attn/ff Linears are Fp8Linear; the fp32 AdaLN and "
            f"head projections are the only dense targets"),
    )
    # 16x VAE spatial compression x the transformer's own 2x2 patchify => every
    # training canvas dimension must be a multiple of 32.
    pixel_align = 32
    temporal = MINIMAX_H3_TEMPORAL
    clip_vae_tiling_policy = minimax_h3_vae_tiling_token()
    #: Audio preprocessing token for the window-level clip record (video AND
    #: audio latents under one key).
    clip_audio_prep_version = MINIMAX_H3_AUDIO_PREP_VERSION
    # ops/minimax_h3_ops.py: x_t = (1-sigma)*x0 + sigma*eps, sigma = shift_sigma(u,
    # shift), monotonic increasing in the sampler draw u. sampler u=0 -> sigma=0
    # -> x_t=x0 (clean); u=1 -> sigma=1 -> x_t=noise. sampler t=0 is clean.
    timestep_convention = "t0"
    # ops/minimax_h3_ops.py train_step: target_v = latents - eps_v,
    # target_a = x0_a - eps_a -- the opposite sign to the usual flow matching.
    velocity_sign = "x0_minus_eps"
    # ops/minimax_h3_ops.py train_step ships the plain flow-matching velocity
    # loss and only notes the weight (design §10).
    consumes_reconstruction_loss_weight = False
    consumes_crop_decode_loss = False
    supplies_predicted_latent = False

    def lora_adapter_class(self):
        from core.training.adapters import MiniMaxH3LoRAAdapter
        return MiniMaxH3LoRAAdapter

    def lora_adapter_kwargs(self, trainer):
        # Default attention+ff IS the design's target set (300 modules / 83.1 M
        # params at rank 16 across all 50 blocks). The I/O heads, the token
        # refiner and AdaLN are excluded permanently and are not reachable from
        # any scope string — see adapters/minimax_h3_adapter.py per exclusion.
        from core.training.adapters.minimax_h3_adapter import parse_scope_csv
        return {"scope": parse_scope_csv(resolve_scope_csv(
            trainer, "minimax_h3_lora_scope", "attention,ff"))}

    def load_components(self, trainer) -> None:
        from core.training.ops import minimax_h3_ops
        minimax_h3_ops.load_components(trainer)

    def setup_block_swap(self, trainer) -> None:
        from core.training.ops import minimax_h3_ops
        minimax_h3_ops.setup_block_swap(trainer)

    def depth_blocks(self, trainer):
        transformer = getattr(trainer, "transformer", None)
        return getattr(transformer, "transformer_blocks", None)

    def setup_attention_backend(self, trainer) -> None:
        from core.training.ops import minimax_h3_ops
        minimax_h3_ops.setup_attention_backend(trainer, trainer.attention_backend)

    def encode_prompt(self, trainer, prompt, *, requires_grad: bool = False):
        from core.training.ops import minimax_h3_ops
        return minimax_h3_ops.encode_prompt(trainer, prompt)

    def collate_aux(self, trainer, batch) -> dict:
        from core.training.ops import minimax_h3_ops
        return minimax_h3_ops.collate_aux(trainer, batch)

    def vae_encode(self, trainer, image_tensor, *, image=None, width=None, height=None,
                   vae_device=None, debug_preprocessing=False):
        from core.training.ops import minimax_h3_ops
        return minimax_h3_ops.vae_encode(
            trainer, image_tensor, image=image, width=width, height=height,
            vae_device=vae_device, debug_preprocessing=debug_preprocessing,
        )

    def vae_encode_clip(self, trainer, clip):
        """Video-clip encode SEAM (``encode_and_cache_clip`` callable).
        ``[T, C, H, W]`` pixel clip -> normalized 5D ``[1, 24, T_lat, H/16, W/16]``."""
        from core.training.ops import minimax_h3_ops
        return minimax_h3_ops.vae_encode_clip(trainer, clip)

    def vae_encode_clip_audio(self, trainer, video_path, start_time, duration):
        """Audio half of the WINDOW-level clip record.

        Same window, same timestamps as ``vae_encode_clip``'s frames — that is
        what makes A/V alignment a property of the construction rather than
        something to check afterwards. Returns ``[2*T_aud, 32]`` or ``None`` for
        a silent / audio-less source.
        """
        from core.training.ops import minimax_h3_ops
        return minimax_h3_ops.vae_encode_audio_window(trainer, video_path, start_time, duration)

    def vae_decode(self, trainer, latents, *, latent_h, latent_w):
        raise NotImplementedError(
            "minimax_h3.vae_decode: decoding is owned by the generation path "
            "(h3_pipeline_ops.decode_video / decode_audio); training never decodes.")

    def train_step(self, trainer, ctx: TrainStepContext):
        from core.training.ops import minimax_h3_ops
        h3_aux = ctx.anima_aux if isinstance(ctx.anima_aux, dict) else {}
        return minimax_h3_ops.train_step(
            trainer,
            latents=ctx.latents,
            prompt_embeds=ctx.text_embeddings,
            h3_aux=h3_aux,
            timesteps=ctx.timesteps,
            debug_save_path=ctx.debug_save_path,
            debug_captions=ctx.debug_captions,
            debug_reference_image_paths=ctx.debug_reference_image_paths,
            profile_vram=ctx.profile_vram,
            alphas_cumprod_cached=ctx.alphas_cumprod_cached,
        )

    def sample(self, trainer, sample_ctx: SampleContext):
        from core.training.ops import minimax_h3_ops
        return minimax_h3_ops.generate_sample(
            trainer,
            prompt=sample_ctx.prompt,
            height=sample_ctx.height,
            width=sample_ctx.width,
            num_inference_steps=sample_ctx.num_inference_steps,
            guidance_scale=sample_ctx.guidance_scale,
            seed=sample_ctx.seed,
            negative_prompt=sample_ctx.negative_prompt,
        )
