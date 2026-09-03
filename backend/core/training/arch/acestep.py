"""AceStepArchHandler — arch handler for arch "acestep" (ACE-Step 1.5 turbo
audio DiT). Thin orchestrator; math lives in ops/acestep_ops.py.

ACE-Step is the flow-matching, TEMPORAL-ONLY-latent (no spatial H/W axis)
reference — mirrors the ltx2 handler shape, but latents are 3D ``[B, T, 64]``
instead of 5D ``[B, C, T, H, W]``.
"""

from __future__ import annotations

from core.training.arch.base_arch import (
    ArchHandler, SampleContext, TrainStepContext, resolve_scope_csv,
)
from core.training.components.wiring import ACESTEP_WIRING


class AceStepArchHandler(ArchHandler):
    name = "acestep"
    wiring = ACESTEP_WIRING
    # No spatial axis to align (audio latents are [B, T, 64]); keep the base
    # default (8) — it is never consulted for this arch (no still-image items).
    pixel_align = 8
    # xt = (1-sigma)*latents + sigma*noise (ops/acestep_ops.py train_step; the
    # vendored model's own x1=noise naming). sampler t=0 is clean.
    timestep_convention = "t0"

    def lora_adapter_class(self):
        from core.training.adapters import AceStepLoRAAdapter
        return AceStepLoRAAdapter

    def lora_adapter_kwargs(self, trainer):
        # Default: attention-only (audio LoRA).
        scope_csv = resolve_scope_csv(trainer, "acestep_lora_scope", "attention")
        wanted = {tok.strip(): True for tok in scope_csv.split(",") if tok.strip()}
        return {"scope": {
            "attention": wanted.get("attention", True),
            "mlp": wanted.get("mlp", False),
        }}

    def load_components(self, trainer) -> None:
        from core.training.ops import acestep_ops
        acestep_ops.load_components(trainer)

    def setup_block_swap(self, trainer) -> None:
        from core.training.ops import acestep_ops
        acestep_ops.setup_block_swap(trainer)

    def setup_attention_backend(self, trainer) -> None:
        from core.training.ops import acestep_ops
        acestep_ops.setup_attention_backend(trainer, trainer.attention_backend)

    def encode_prompt(self, trainer, prompt, *, requires_grad: bool = False, lyrics: str = ""):
        from core.training.ops import acestep_ops
        return acestep_ops.encode_prompt(trainer, prompt, lyrics=lyrics)

    def collate_aux(self, trainer, batch) -> dict:
        from core.training.ops import acestep_ops
        return acestep_ops.collate_aux(trainer, batch)

    def vae_encode(self, trainer, image_tensor, *, image=None, width=None, height=None,
                   vae_device=None, debug_preprocessing=False):
        from core.training.ops import acestep_ops
        return acestep_ops.vae_encode(
            trainer, image_tensor, image=image, width=width, height=height,
            vae_device=vae_device, debug_preprocessing=debug_preprocessing,
        )

    def vae_encode_audio(self, trainer, waveform):
        """Audio-clip encode SEAM (``audio_loader.encode_and_cache_audio``
        callable). ``[2, samples]`` stereo waveform -> ``[1, T, 64]`` latent."""
        from core.training.ops import acestep_ops
        return acestep_ops.vae_encode_audio(trainer, waveform)

    def vae_decode(self, trainer, latents, *, latent_h=None, latent_w=None):
        from core.training.ops import acestep_ops
        return acestep_ops.vae_decode(trainer, latents, latent_h=latent_h, latent_w=latent_w)

    def train_step(self, trainer, ctx: TrainStepContext):
        from core.training.ops import acestep_ops
        aux = ctx.anima_aux if isinstance(ctx.anima_aux, dict) else {}
        return acestep_ops.train_step(
            trainer,
            latents=ctx.latents,
            text_embeddings=ctx.text_embeddings,
            aux=aux,
            timesteps=ctx.timesteps,
            debug_save_path=ctx.debug_save_path,
            debug_captions=ctx.debug_captions,
            debug_reference_image_paths=ctx.debug_reference_image_paths,
            profile_vram=ctx.profile_vram,
        )

    def sample(self, trainer, sample_ctx: SampleContext):
        from core.training.ops import acestep_ops
        return acestep_ops.generate_sample(
            trainer,
            prompt=sample_ctx.prompt,
            height=sample_ctx.height,
            width=sample_ctx.width,
            num_inference_steps=sample_ctx.num_inference_steps,
            guidance_scale=sample_ctx.guidance_scale,
            seed=sample_ctx.seed,
            negative_prompt=sample_ctx.negative_prompt,
        )
