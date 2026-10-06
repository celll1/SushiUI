"""Ltx2ArchHandler — P5 arch handler for arch "ltx2" (LTX-2.3 joint audio+video
MM-DiT). Thin orchestrator; math lives in ops/ltx2_ops.py.

LTX-2.3 is the flow-matching, latent_ndim=5 VIDEO reference (mirrors the anima
handler shape). Stills train as degenerate 1-frame clips (T=1) through the SAME
5D train_step — no special-case squeeze.
"""

from __future__ import annotations

from core.training.arch.base_arch import (
    ArchHandler, SampleContext, TrainStepContext, resolve_scope_csv,
    PHASE2_PENDING, QUANTIZED_ADDITIVE_PENDING,
    declare_adapter_capability,
)
from core.training.components.wiring import LTX2_TEMPORAL, LTX2_WIRING


class Ltx2ArchHandler(ArchHandler):
    def __init__(self, trainer=None):
        super().__init__(trainer)
        if trainer is not None:
            self.resolve_wiring(trainer)

    def resolve_wiring(self, trainer):
        from dataclasses import replace
        from math import lcm
        from core.models.components.latent_adapter import topology_alignment
        from core.models.components.video_latent_adapter import NormalizedVideoVAE
        vae = getattr(trainer, "vae", None)
        model = getattr(trainer, "transformer", None)
        if vae is None or model is None:
            return self.wiring
        scales = getattr(model.config, "vae_scale_factors", (self.wiring.vae_scale_temporal,
                         self.wiring.vae_scale_factor, self.wiring.vae_scale_factor))
        spatial, temporal = int(scales[1]), int(scales[0])
        channels = int(model.config.in_channels)
        normalized = isinstance(vae, NormalizedVideoVAE)
        if normalized:
            spatial, temporal = vae.spatial_compression_ratio, vae.temporal_compression_ratio
            channels = vae.contract.image.latent_channels
        inner = getattr(model, "transformer", model)
        h,w = topology_alignment(getattr(inner, "_residual_unet_spec", None))
        self.pixel_align = spatial*lcm(h,w)
        lengths = tuple(sorted({((n-1+temporal-1)//temporal)*temporal+1 for n in LTX2_TEMPORAL.default_clip_lengths}))
        self.temporal = replace(LTX2_TEMPORAL, frame_multiple=temporal, pixel_align=self.pixel_align,
            latent_frames=lambda n:(n-1)//temporal+1, default_clip_lengths=lengths, suggested_min_frames=temporal+1)
        return self.wiring.replace(latent_channels=channels, vae_scale_factor=spatial, vae_scale_temporal=temporal,
            vae_norm="identity" if normalized else self.wiring.vae_norm, vae_norm_pack=1)

    def apply_vae_swap(self, trainer, resolved, module=None):
        from core.models.ltx2.vae_training import apply_swap
        return apply_swap(trainer,resolved,handler=self,module=module)

    def check_vae_compatibility(self, facts, *, trainer=None, base_model_path=None):
        from core.models.ltx2.vae_training import temporal_policy
        if facts.get("pixel_channels", 3) != 3 or facts.get("output_channels", 3) != 3:
            return False,"LTX VAE swap requires RGB encode and decode channels"
        scale,temporal = facts.get("scale_factor"),facts.get("scale_temporal")
        if (facts.get("ndim") not in (4,5) or type(scale) is not int or scale < 1 or
                type(temporal) is not int or temporal < 1):
            return False,"LTX VAE swap requires declared RGB video/image codec geometry"
        if facts.get("ndim") == 4 and temporal != 1:
            return False,"A 2D LTX target codec must preserve independent frames"
        try:
            temporal_policy(facts.get("ndim"), facts.get("vae_class"))
        except ValueError as exc:
            return False,str(exc)
        if scale % self.wiring.vae_scale_factor and self.wiring.vae_scale_factor % scale:
            return False,"LTX VAE swap requires integer spatial scale relative to its native codec"
        return True,None

    def calibrate_before_training(self, trainer, datasets, bucket_manager):
        from core.models.ltx2.vae_training import calibrate
        calibrate(trainer,datasets)

    cfg_null_stage = "caption"

    # Its forward installs these on the inner module (ops/ltx2_ops.py).
    consumes_block_loop_features = True
    name = "ltx2"
    wiring = LTX2_WIRING
    adapter_capability = declare_adapter_capability(
        "ltx2",
        additive_family=True,
        initial_dora="dense_only",
        additive_reason=PHASE2_PENDING,
        quantized_base_reason=QUANTIZED_ADDITIVE_PENDING,
    )
    wires_sample_step_progress = True
    pixel_align = 32  # LTX spatial VAE downscale (÷32); dims must be a multiple of 32.
    # Temporal contract: 8*k+1 clip lengths, (L-1)//8+1 latent frames, NO fixed
    # frame rate (the clip keeps its source fps, which train_step feeds to the
    # RoPE coords per sample). These are exactly the rules the video loader and
    # bucketer applied by hardcoding before Phase 6a made them declarative.
    temporal = LTX2_TEMPORAL
    # ltx2_ops.vae_encode_clip does not configure VAE tiling, so there is no
    # policy to key on and LTX-2.3 clip-cache keys are unchanged. See
    # ArchHandler.clip_vae_tiling_policy for why this matters when an arch does.
    clip_vae_tiling_policy = None
    # x_t = (1-sigma)*x_0 + sigma*noise (ops/ltx2_ops.py train_step comment).
    # sampler t=0 is clean.
    timestep_convention = "t0"
    # ops/ltx2_ops.py train_step: target = _pack_latents(noise - latents).
    velocity_sign = "eps_minus_x0"
    consumes_reconstruction_loss_weight = True
    consumes_crop_decode_loss = False
    supplies_predicted_latent = False

    def lora_adapter_class(self):
        from core.training.adapters import Ltx2LoRAAdapter
        return Ltx2LoRAAdapter

    def lora_adapter_kwargs(self, trainer):
        # Default: attention-only (video LoRA).
        scope_csv = resolve_scope_csv(trainer, "ltx2_lora_scope", "attention")
        wanted = {tok.strip(): True for tok in scope_csv.split(",") if tok.strip()}
        return {"scope": {
            "attention": wanted.get("attention", True),
            "ff": wanted.get("ff", False),
            "audio": wanted.get("audio", False),
            "av_cross": wanted.get("av_cross", False),
        }}

    def load_components(self, trainer) -> None:
        from core.training.ops import ltx2_ops
        ltx2_ops.load_components(trainer)

    def setup_block_swap(self, trainer) -> None:
        from core.training.ops import ltx2_ops
        ltx2_ops.setup_block_swap(trainer)

    def depth_blocks(self, trainer):
        transformer = getattr(trainer, "transformer", None)
        return getattr(transformer, "transformer_blocks", None)

    def setup_attention_backend(self, trainer) -> None:
        from core.training.ops import ltx2_ops
        ltx2_ops.setup_attention_backend(trainer, trainer.attention_backend)

    def encode_prompt(self, trainer, prompt, *, requires_grad: bool = False):
        from core.training.ops import ltx2_ops
        return ltx2_ops.encode_prompt(trainer, prompt)

    def collate_aux(self, trainer, batch) -> dict:
        from core.training.ops import ltx2_ops
        return ltx2_ops.collate_aux(trainer, batch)

    def vae_encode(self, trainer, image_tensor, *, image=None, width=None, height=None,
                   vae_device=None, debug_preprocessing=False):
        from core.training.ops import ltx2_ops
        return ltx2_ops.vae_encode(
            trainer, image_tensor, image=image, width=width, height=height,
            vae_device=vae_device, debug_preprocessing=debug_preprocessing,
        )

    def vae_encode_clip(self, trainer, clip):
        """Video-clip encode SEAM (P4b ``encode_and_cache_clip`` callable).
        ``[T, C, H, W]`` pixel clip -> normalized 5D ``[1, 128, T_lat, H', W']``."""
        from core.training.ops import ltx2_ops
        return ltx2_ops.vae_encode_clip(trainer, clip)

    def vae_decode(self, trainer, latents, *, latent_h, latent_w):
        raise NotImplementedError("ltx2.vae_decode: sampling reuses the LTX2Pipeline directly")

    def train_step(self, trainer, ctx: TrainStepContext):
        from core.training.ops import ltx2_ops
        ltx2_aux = ctx.anima_aux if isinstance(ctx.anima_aux, dict) else {}
        return ltx2_ops.train_step(
            trainer,
            latents=ctx.latents,
            prompt_embeds=ctx.text_embeddings,
            ltx2_aux=ltx2_aux,
            timesteps=ctx.timesteps,
            debug_save_path=ctx.debug_save_path,
            debug_captions=ctx.debug_captions,
            debug_reference_image_paths=ctx.debug_reference_image_paths,
            profile_vram=ctx.profile_vram,
            alphas_cumprod_cached=ctx.alphas_cumprod_cached,
        )

    def sample(self, trainer, sample_ctx: SampleContext):
        from core.training.ops import ltx2_ops
        return ltx2_ops.generate_sample(
            trainer,
            prompt=sample_ctx.prompt,
            height=sample_ctx.height,
            width=sample_ctx.width,
            num_inference_steps=sample_ctx.num_inference_steps,
            guidance_scale=sample_ctx.guidance_scale,
            seed=sample_ctx.seed,
            negative_prompt=sample_ctx.negative_prompt,
            step_progress_callback=sample_ctx.step_progress_callback,
        )
