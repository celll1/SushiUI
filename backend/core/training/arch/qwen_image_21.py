"""Qwen-Image 2.1 training architecture handler."""

from core.training.arch.base_arch import (
    ArchHandler, SampleContext, TrainStepContext, PHASE2_PENDING,
    QUANTIZED_ADDITIVE_PENDING, declare_adapter_capability,
)
from core.training.components.wiring import QWEN_IMAGE_21_WIRING


class QwenImage21ArchHandler(ArchHandler):
    cfg_null_stage = "caption"
    name = "qwen_image_21"
    wiring = QWEN_IMAGE_21_WIRING
    adapter_capability = declare_adapter_capability(
        "qwen_image_21",
        additive_family=True,
        initial_dora="deferred",
        additive_reason=PHASE2_PENDING,
        quantized_base_reason=QUANTIZED_ADDITIVE_PENDING,
    )
    pixel_align = 32
    dit_checkpoint_block_count = 32
    supports_dit_partition_training = True
    wires_sample_step_progress = True
    timestep_convention = "t0"
    velocity_sign = "eps_minus_x0"
    # 1536px measurements reached 20-29 GiB of step activation for 8.5-9.2k
    # image tokens; the generic 24e-6 image seed under-predicts this architecture.
    activation_dispatch_seed_floor = 3.5e-3

    def dit_partition_adapter(self):
        from core.training.qwen_partition import QwenImage21PartitionAdapter
        return QwenImage21PartitionAdapter()

    def lora_adapter_class(self):
        from core.training.adapters import QwenImage21LoRAAdapter
        return QwenImage21LoRAAdapter

    def load_components(self, trainer):
        from core.training.ops import qwen_image_21_ops
        qwen_image_21_ops.load_components(trainer)

    def setup_block_swap(self, trainer):
        from core.training.ops import qwen_image_21_ops
        qwen_image_21_ops.setup_block_swap(trainer)

    def setup_attention_backend(self, trainer) -> None:
        from core.training.ops import qwen_image_21_ops
        qwen_image_21_ops.setup_attention_backend(trainer, trainer.attention_backend)

    def depth_blocks(self, trainer):
        return trainer.transformer.transformer_blocks

    def encode_prompt(self, trainer, prompt, *, requires_grad=False):
        from core.training.ops import qwen_image_21_ops
        return qwen_image_21_ops.encode_prompt(trainer, prompt)

    def vae_encode(self, trainer, image_tensor, **kwargs):
        from core.training.ops import qwen_image_21_ops
        return qwen_image_21_ops.vae_encode(trainer, image_tensor, **kwargs)

    def vae_decode(self, trainer, latents, *, latent_h, latent_w):
        from core.training.ops import qwen_image_21_ops
        return qwen_image_21_ops.vae_decode(
            trainer, latents, latent_h=latent_h, latent_w=latent_w
        )

    def train_step(self, trainer, ctx: TrainStepContext):
        from core.training.ops import qwen_image_21_ops
        return qwen_image_21_ops.train_step(
            trainer,
            latents=ctx.latents,
            encoder_features=ctx.encoder_features,
            encoder_mask=ctx.encoder_mask,
            timesteps=ctx.timesteps,
            latent_h=ctx.latent_h,
            latent_w=ctx.latent_w,
            profile_vram=ctx.profile_vram,
            repa_pixels=ctx.repa_pixels,
            debug_save_path=ctx.debug_save_path,
            debug_captions=ctx.debug_captions,
            debug_reference_image_paths=ctx.debug_reference_image_paths,
        )

    def sample(self, trainer, sample_ctx: SampleContext):
        from core.training.ops import qwen_image_21_ops
        return qwen_image_21_ops.generate_sample(
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
