"""Training architecture handler for SenseNova SDXL Chimera."""

from __future__ import annotations

from core.training.arch.base_arch import (
    ArchHandler,
    PHASE2_PENDING,
    PHASE3_PENDING,
    SampleContext,
    TrainStepContext,
    declare_adapter_capability,
)
from core.training.components.wiring import SENSENOVA_SDXL_CHIMERA_WIRING


class SenseNovaSDXLChimeraArchHandler(ArchHandler):
    name = "sensenova_sdxl_chimera"
    wiring = SENSENOVA_SDXL_CHIMERA_WIRING
    latent_io_root_attr = None
    pixel_align = 8
    timestep_convention = "t1"
    velocity_sign = "x0_minus_eps"
    cfg_null_stage = "encode"
    consumes_reconstruction_loss_weight = False
    consumes_crop_decode_loss = False
    supplies_predicted_latent = False
    wires_sample_step_progress = True
    adapter_capability = declare_adapter_capability(
        "sensenova_sdxl_chimera",
        additive_family=False,
        initial_dora="refused",
        additive_reason=PHASE2_PENDING,
        quantized_base_reason=PHASE3_PENDING,
    )

    def load_components(self, trainer) -> None:
        from core.training.ops import sensenova_sdxl_chimera_ops as ops

        ops.load_components(trainer)

    def setup_block_swap(self, trainer) -> None:
        if int(getattr(trainer, "blocks_to_swap", 0) or 0):
            raise ValueError("Chimera training does not support block swap")

    def setup_attention_backend(self, trainer) -> None:
        from core.training.ops import sensenova_sdxl_chimera_ops as ops

        ops.setup_attention_backend(trainer, trainer.attention_backend)

    def repa_tap(self, trainer):
        from core.training.ops import sensenova_sdxl_chimera_ops as ops

        return ops.repa_tap(trainer)

    def encode_prompt(self, trainer, prompt, *, requires_grad: bool = False):
        from core.training.ops import sensenova_sdxl_chimera_ops as ops

        return ops.encode_prompt(trainer, prompt, requires_grad=requires_grad)

    def encode_prompt_cfg_null(self, trainer, prompt, *, requires_grad=False, **_kwargs):
        return self.encode_prompt(trainer, "", requires_grad=requires_grad)

    def collate_aux(self, trainer, batch) -> dict:
        from core.training.ops import sensenova_sdxl_chimera_ops as ops

        return ops.collate_aux(batch)

    def vae_encode(self, trainer, image_tensor, **kwargs):
        from core.training.ops import sensenova_sdxl_chimera_ops as ops

        return ops.vae_encode(trainer, image_tensor, **kwargs)

    def vae_decode(self, trainer, latents, **kwargs):
        from core.training.ops import sensenova_sdxl_chimera_ops as ops

        return ops.vae_decode(trainer, latents, **kwargs)

    def train_step(self, trainer, ctx: TrainStepContext):
        from core.training.ops import sensenova_sdxl_chimera_ops as ops

        return ops.train_step(trainer, ctx)

    def sample(self, trainer, sample_ctx: SampleContext):
        from core.training.ops import sensenova_sdxl_chimera_ops as ops
        from core.models.sensenova_sdxl_chimera import pipeline_ops

        unet = trainer.unet
        original_device = next(unet.parameters()).device
        sample_device = trainer.device
        was_training = unet.training
        try:
            if original_device != sample_device:
                unet.to(sample_device)
            unet.eval()
            positive, pos_aux = self.encode_prompt(trainer, sample_ctx.prompt)
            negative, neg_aux = self.encode_prompt(trainer, sample_ctx.negative_prompt)
            make = lambda hidden, aux, key: pipeline_ops.ChimeraConditioning(
                encoder_hidden_states=hidden,
                pooled_text_embeds=aux["pooled_text_embeds"],
                context_positions=aux["context_positions"],
                attention_mask=aux["context_attention_mask"],
                fingerprint=key,
            )
            latents = ops.generate_sample(
                unet,
                make(positive, pos_aux, "training-preview-positive"),
                make(negative, neg_aux, "training-preview-negative"),
                height=sample_ctx.height,
                width=sample_ctx.width,
                steps=sample_ctx.num_inference_steps,
                cfg_scale=sample_ctx.guidance_scale,
                seed=sample_ctx.seed,
                timestep_shift=sample_ctx.sensenova_timestep_shift,
                cfg_norm=sample_ctx.sensenova_cfg_norm,
                cfg_schedule_type=sample_ctx.cfg_schedule_type,
                cfg_schedule_min=sample_ctx.cfg_schedule_min,
                cfg_schedule_max=sample_ctx.cfg_schedule_max,
                cfg_schedule_power=sample_ctx.cfg_schedule_power,
                step_progress_callback=sample_ctx.step_progress_callback,
                cfg_probe_callback=sample_ctx.cfg_probe_callback,
            )
        finally:
            unet.train(was_training)
            if original_device != sample_device:
                unet.to(original_device)
        return pipeline_ops.decode_latents(
            trainer.vae,
            latents,
            device=trainer.device,
            restore_device=True,
        )
