"""Native Ming Design/Layer training handler."""

from core.training.arch.base_arch import (ArchHandler, declare_adapter_capability,
    PHASE2_PENDING, QUANTIZED_ADDITIVE_PENDING)
from core.models.components.wiring import MING_IMAGE_01_WIRING


class MingImage01ArchHandler(ArchHandler):
    supports_dit_partition_training = True
    cfg_null_stage = "collated"
    name = "ming_image_01"
    wiring = MING_IMAGE_01_WIRING
    adapter_capability = declare_adapter_capability("ming_image_01",
        additive_family=True, initial_dora="deferred", additive_reason=PHASE2_PENDING,
        quantized_base_reason=QUANTIZED_ADDITIVE_PENDING)
    pixel_align = 16
    timestep_convention = "t0"
    velocity_sign = "x0_minus_eps"
    wires_sample_step_progress = True
    consumes_reconstruction_loss_weight = True

    def __init__(self, trainer=None):
        super().__init__(trainer)
        if trainer is not None and getattr(trainer, "transformer", None) is not None:
            self.wiring = self.wiring.replace(te_out_dim=trainer.transformer.config.cap_feat_dim)

    def lora_adapter_class(self):
        from core.training.adapters.ming_image_01_adapter import MingImage01LoRAAdapter
        return MingImage01LoRAAdapter

    def load_components(self, trainer):
        from core.training.ops.ming_image_01_ops import load_components
        load_components(trainer)

    def setup_block_swap(self, trainer):
        from core.models.ming_image_01.block_offload import setup_training_block_swap
        setup_training_block_swap(trainer)

    def setup_attention_backend(self, trainer):
        from core.training.ops.ming_image_01_ops import setup_attention_backend
        setup_attention_backend(trainer)

    def dit_partition_adapter(self):
        from core.models.ming_image_01.partition import MingPartitionAdapter
        return MingPartitionAdapter()

    def depth_blocks(self, trainer):
        return trainer.transformer.layers

    def encode_prompt(self, trainer, prompt, *, requires_grad=False):
        from core.training.ops.ming_image_01_ops import encode_prompt
        return encode_prompt(trainer, prompt, requires_grad=requires_grad)

    def vae_encode(self, trainer, image_tensor, **kwargs):
        from core.training.ops.ming_image_01_ops import vae_encode
        return vae_encode(trainer, image_tensor, **kwargs)

    def vae_decode(self, trainer, latents, **kwargs):
        from core.training.ops.ming_image_01_ops import vae_decode
        return vae_decode(trainer, latents, **kwargs)

    def apply_cfg_null_collated(self, trainer, conditioning, auxiliary, drop_mask):
        from core.models.ming_image_01.conditioning import apply_cfg_null_conditioning
        return apply_cfg_null_conditioning(conditioning, drop_mask), auxiliary

    def train_step(self, trainer, ctx):
        from core.training.ops import ming_image_01_ops
        if ctx.ming_batch is None:
            raise ValueError("Ming training requires an aligned native RGBA/dual-conditioning batch")
        from dataclasses import replace
        conditioning, _ = self.apply_cfg_null_step(trainer, ctx, ctx.ming_batch.conditioning, None)
        batch = replace(ctx.ming_batch, conditioning=conditioning)
        return ming_image_01_ops.train_step(trainer, batch, timesteps=ctx.timesteps, guidance_drop_mask=ctx.cfg_drop_mask,
            debug_save_path=ctx.debug_save_path, debug_captions=ctx.debug_captions,
            debug_reference_image_paths=ctx.debug_reference_image_paths)

    def sample(self, trainer, sample_ctx):
        from core.training.ops import ming_image_01_ops
        return ming_image_01_ops.generate_sample(trainer, sample_ctx,
            step_progress_callback=sample_ctx.step_progress_callback)
