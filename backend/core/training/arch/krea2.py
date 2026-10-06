"""Krea 2 training architecture handler."""

from __future__ import annotations

from core.training.arch.base_arch import (
    ArchHandler, SampleContext, TrainStepContext, resolve_scope_csv,
    PHASE2_PENDING, QUANTIZED_ADDITIVE_PENDING, declare_adapter_capability,
)
from core.training.components.wiring import KREA2_WIRING


class Krea2ArchHandler(ArchHandler):
    cfg_null_stage = "caption"
    name = "krea2"
    wiring = KREA2_WIRING
    adapter_capability = declare_adapter_capability(
        "krea2",
        additive_family=True,
        initial_dora="deferred",
        additive_reason=PHASE2_PENDING,
        quantized_base_reason=(
            f"{QUANTIZED_ADDITIVE_PENDING}; its INT8/FP8 bases need capability "
            f"gates"),
    )
    pixel_align = 16  # vae_scale(8) * patch(2); latent grid = pixel/16
    wires_sample_step_progress = True
    # noisy = (1-sigma)*latents + sigma*noise (ops/krea2_ops.py, "sigma=1 ->
    # noise" comment). sampler t=0 is clean.
    timestep_convention = "t0"
    # ops/krea2_ops.py train_step: v_target = noise - latents.
    velocity_sign = "eps_minus_x0"
    consumes_reconstruction_loss_weight = True
    consumes_crop_decode_loss = True
    supplies_predicted_latent = True

    def resolve_wiring(self, trainer):
        from core.models.components.latent_adapter import NormalizedImageVAE
        vae = getattr(trainer, "vae", None)
        if isinstance(vae, NormalizedImageVAE):
            self.pixel_align = vae.spatial_scale_factor * 2
            return self.wiring.replace(latent_channels=vae.config.latent_channels,
                vae_scale_factor=vae.spatial_scale_factor, vae_norm="identity", vae_norm_pack=1)
        return getattr(trainer, "wiring", None) or self.wiring

    def apply_vae_swap(self, trainer, resolved, module=None):
        from core.models.krea2.vae_alignment import apply_swap
        return apply_swap(trainer,resolved,handler=self,module=module)

    def check_vae_compatibility(self, facts, *, trainer=None, base_model_path=None):
        scale = facts.get("scale_factor")
        if facts.get("ndim") not in (4,5) or type(scale) is not int or scale < 1:
            return False, "Krea VAE swap requires an image codec with declared spatial compression"
        if scale % 8 and 8 % scale:
            return False, "Krea VAE swap requires integer spatial scale relative to 8x"
        return True,None

    def calibrate_before_training(self, trainer, datasets, bucket_manager):
        from core.models.krea2.vae_alignment import calibrate
        calibrate(trainer,datasets)

    def lora_adapter_class(self):
        from core.training.adapters import Krea2LoRAAdapter
        return Krea2LoRAAdapter

    def lora_adapter_kwargs(self, trainer):
        from core.models.krea2.krea2_lora import parse_scope_csv
        return {"scope": parse_scope_csv(resolve_scope_csv(
            trainer, "krea2_lora_scope", "attn,mlp"))}

    def load_components(self, trainer) -> None:
        # P3c: body lives in ops/krea2_ops (shared with the base_trainer load-time
        # dispatcher + _load_checkpoint_as_base, which cannot route via self.arch —
        # self.arch binds after loading; see the construction-order note in krea2_ops).
        from core.training.ops import krea2_ops
        krea2_ops.load_components(trainer)

    def setup_block_swap(self, trainer) -> None:
        # P3c: body lives in ops/krea2_ops (shared with the base_trainer
        # setup_krea2_block_swap delegator, called late by mode subclasses).
        from core.training.ops import krea2_ops
        krea2_ops.setup_block_swap(trainer)

    def depth_blocks(self, trainer):
        transformer = getattr(trainer, "transformer", None)
        return getattr(transformer, "transformer_blocks", None)

    def repa_tap(self, trainer):
        # The tap is the Krea 2 DiT itself: its block loop stashes the image
        # tokens [B, N_img, hidden_size], already past the text prefix. No
        # unwrap -- load_components keeps transformer_original IS transformer.
        # hidden_size is the attribute __init__ computes as
        # attention_head_dim * num_attention_heads; the config has no such key.
        from core.training.repa import RepaTapPoint

        transformer = trainer.transformer
        return RepaTapPoint(module=transformer,
                            hidden_size=int(transformer.hidden_size),
                            depth=len(transformer.transformer_blocks))

    def setup_attention_backend(self, trainer) -> None:
        # P3c: body lives in ops/krea2_ops (shared with base_trainer delegator).
        from core.training.ops import krea2_ops
        krea2_ops.setup_attention_backend(trainer, trainer.attention_backend)

    def encode_prompt(self, trainer, prompt, *, requires_grad: bool = False):
        # P4: body lives in ops/krea2_ops (shared with base_trainer delegator).
        from core.training.ops import krea2_ops
        return krea2_ops.encode_prompt(trainer, prompt)

    def vae_encode(self, trainer, image_tensor, *, image=None, width=None, height=None,
                   vae_device=None, debug_preprocessing=False):
        from core.training.ops import krea2_ops
        return krea2_ops.vae_encode(
            trainer, image_tensor, image=image, width=width, height=height,
            vae_device=vae_device, debug_preprocessing=debug_preprocessing,
        )

    def vae_decode(self, trainer, latents, *, latent_h, latent_w):
        raise NotImplementedError("krea2.vae_decode: phase P5/P7")

    def train_step(self, trainer, ctx: TrainStepContext):
        # P6c: verbatim body in ops/krea2_ops.train_step. ctx fields map 1:1 to
        # the previous train_step_krea2 kwargs bundle.
        from core.training.ops import krea2_ops
        return krea2_ops.train_step(
            trainer,
            latents=ctx.latents,
            encoder_features=ctx.encoder_features,
            encoder_mask=ctx.encoder_mask,
            timesteps=ctx.timesteps,
            profile_vram=ctx.profile_vram,
            latent_h=ctx.latent_h,
            latent_w=ctx.latent_w,
            repa_pixels=ctx.repa_pixels,
        )

    def sample(self, trainer, sample_ctx: SampleContext):
        # P7: verbatim body in ops/krea2_ops.generate_sample.
        from core.training.ops import krea2_ops
        return krea2_ops.generate_sample(
            trainer,
            prompt=sample_ctx.prompt,
            height=sample_ctx.height,
            width=sample_ctx.width,
            num_inference_steps=sample_ctx.num_inference_steps,
            guidance_scale=sample_ctx.guidance_scale,
            seed=sample_ctx.seed,
            negative_prompt=sample_ctx.negative_prompt,
            cfg_schedule_type=sample_ctx.cfg_schedule_type,
            cfg_schedule_min=sample_ctx.cfg_schedule_min,
            cfg_schedule_max=sample_ctx.cfg_schedule_max,
            cfg_schedule_power=sample_ctx.cfg_schedule_power,
            step_progress_callback=sample_ctx.step_progress_callback,
        )
