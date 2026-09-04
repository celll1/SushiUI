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
