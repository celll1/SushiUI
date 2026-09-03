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

Full fine-tuning is refused for this architecture in three places (design §7):
this package ships no ``FullParameterAdapter``, ``full_parameter_trainer``
raises, and ``api.arch_capabilities.TRAINING_UNSUPPORTED`` declares it.
"""

from __future__ import annotations

from core.training.arch.base_arch import (
    ArchHandler, SampleContext, TrainStepContext, resolve_scope_csv,
)
from core.training.components.wiring import MINIMAX_H3_TEMPORAL, MINIMAX_H3_WIRING
from core.training.ops.minimax_h3_ops import (
    MINIMAX_H3_AUDIO_PREP_VERSION,
    minimax_h3_vae_tiling_token,
)


class MiniMaxH3ArchHandler(ArchHandler):
    name = "minimax_h3"
    wiring = MINIMAX_H3_WIRING
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
