"""
Ideogram4ArchHandler — P0/P1 stub handler for arch "ideogram4".

Empty subclass: name + wiring spec are real; every canonical method raises
NotImplementedError. Bodies are moved from base_trainer.py in phases P3-P7.
Nothing calls these until a later phase flips the corresponding dispatcher.
"""

from __future__ import annotations

from core.training.arch.base_arch import (
    ArchHandler, SampleContext, TrainStepContext, resolve_scope_csv,
    PHASE2_PENDING, QUANTIZED_ADDITIVE_PENDING, declare_adapter_capability,
)
from core.training.components.wiring import IDEOGRAM4_WIRING


class Ideogram4ArchHandler(ArchHandler):
    name = "ideogram4"
    wiring = IDEOGRAM4_WIRING
    adapter_capability = declare_adapter_capability(
        "ideogram4",
        additive_family=True,
        initial_dora="deferred",
        additive_reason=PHASE2_PENDING,
        quantized_base_reason=(
            f"{QUANTIZED_ADDITIVE_PENDING}; either transformer can be FP8"),
    )
    pixel_align = 16  # vae_scale(8) * patch(2) (ideogram4_resolution)
    text_seq_axis = 2  # encode_prompt returns [1, 13, L, 4096]
    # noisy = (1-sigma)*latents + sigma*noise (ops/ideogram4_ops.py, "sigma=1 ->
    # noise" comment). sampler t=0 is clean.
    timestep_convention = "t0"
    # ops/ideogram4_ops.py train_step: v_target = latents - noise, matching the
    # inference path's scheduler.step(-v).
    velocity_sign = "x0_minus_eps"
    consumes_reconstruction_loss_weight = True
    consumes_crop_decode_loss = True

    def lora_adapter_class(self):
        from core.training.adapters import Ideogram4LoRAAdapter
        return Ideogram4LoRAAdapter

    def lora_adapter_kwargs(self, trainer):
        from core.models.ideogram4.ideogram4_lora import parse_scope_csv
        return {"scope": parse_scope_csv(resolve_scope_csv(
            trainer, "ideogram4_lora_scope", "attn,mlp"))}

    def load_components(self, trainer) -> None:
        # P3b: body lives in ops/ideogram4_ops (shared with the base_trainer
        # load-time dispatcher, which cannot route via self.arch — self.arch
        # binds after loading; see the construction-order note in ideogram4_ops).
        from core.training.ops import ideogram4_ops
        ideogram4_ops.load_components(trainer)

    def setup_block_swap(self, trainer) -> None:
        # P3b: body lives in ops/ideogram4_ops (shared with the base_trainer
        # setup_ideogram4_block_swap delegator, called late by mode subclasses).
        from core.training.ops import ideogram4_ops
        ideogram4_ops.setup_block_swap(trainer)

    def depth_blocks(self, trainer):
        # Two parallel copies of one stack: layer j is the SAME depth in the
        # conditional and the unconditional transformer, so they share an entry
        # rather than being concatenated (which would make the uncond stack look
        # like a second half of one deeper network).
        layers = getattr(getattr(trainer, "transformer", None), "layers", None)
        if layers is None:
            return None
        uncond = getattr(trainer, "transformer_uncond", None)
        if uncond is None or not getattr(trainer, "ideogram4_train_uncond", False):
            return list(layers)
        uncond_layers = list(getattr(uncond, "layers", None) or [])
        return [([block] + ([uncond_layers[j]] if j < len(uncond_layers) else []))
                for j, block in enumerate(layers)]

    def repa_tap(self, trainer):
        # The tap is the CONDITIONAL Ideogram 4 DiT: its block loop stashes the
        # packed sequence [B, max_text + N_img, hidden_size], which train_step
        # slices past the text prefix. The unconditional twin is a separate
        # instance and is never armed -- one grid, one teacher, one tap. No
        # unwrap: load_components keeps transformer_original IS transformer.
        # hidden_size is the attribute __init__ computes as
        # attention_head_dim * num_attention_heads; the config has no such key.
        from core.training.repa import RepaTapPoint

        transformer = trainer.transformer
        return RepaTapPoint(module=transformer,
                            hidden_size=int(transformer.hidden_size),
                            depth=len(transformer.layers))

    def setup_attention_backend(self, trainer) -> None:
        # P3b: body lives in ops/ideogram4_ops (shared with base_trainer delegator).
        from core.training.ops import ideogram4_ops
        ideogram4_ops.setup_attention_backend(trainer, trainer.attention_backend)

    def encode_prompt(self, trainer, prompt, *, requires_grad: bool = False):
        # P4: body lives in ops/ideogram4_ops (shared with base_trainer delegator).
        from core.training.ops import ideogram4_ops
        return ideogram4_ops.encode_prompt(trainer, prompt)

    def vae_encode(self, trainer, image_tensor, *, image=None, width=None, height=None,
                   vae_device=None, debug_preprocessing=False):
        from core.training.ops import ideogram4_ops
        return ideogram4_ops.vae_encode(
            trainer, image_tensor, image=image, width=width, height=height,
            vae_device=vae_device, debug_preprocessing=debug_preprocessing,
        )

    def vae_decode(self, trainer, latents, *, latent_h, latent_w):
        raise NotImplementedError("ideogram4.vae_decode: phase P5/P7")

    def train_step(self, trainer, ctx: TrainStepContext):
        # P6b: verbatim body in ops/ideogram4_ops.train_step. Same ctx contract as
        # lens (encoder_features / encoder_mask / latent_h / latent_w).
        from core.training.ops import ideogram4_ops
        return ideogram4_ops.train_step(
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
        # P7: warned-skip moved verbatim from the old _dispatch_sample ideogram4
        # branch. Ideogram4 sampling (dual transformer + fp8 + resolution-aware
        # dual-branch conditioning) is not yet ported to the trainer. Warn and
        # skip (return None) rather than crash into the SD/SDXL generate_sample
        # path; callers guard on ``sample is not None`` before saving.
        print(f"{trainer.log_prefix} WARNING: step-0/periodic sampling is not yet "
              f"supported for ideogram4 (dual-transformer + fp8); skipping this sample.")
        return None
