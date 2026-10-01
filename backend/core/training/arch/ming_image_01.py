"""Native Ming generation-independent full-training handler."""

from core.training.arch.base_arch import ArchHandler
from core.models.components.wiring import MING_IMAGE_01_WIRING


class MingImage01ArchHandler(ArchHandler):
    name = "ming_image_01"
    wiring = MING_IMAGE_01_WIRING
    pixel_align = 16
    timestep_convention = "t0"
    velocity_sign = "x0_minus_eps"
    wires_sample_step_progress = True
    consumes_reconstruction_loss_weight = True

    def __init__(self, trainer=None):
        super().__init__(trainer)
        if trainer is not None and getattr(trainer, "transformer", None) is not None:
            self.wiring = self.wiring.replace(te_out_dim=trainer.transformer.config.cap_feat_dim)

    def load_components(self, trainer):
        from core.training.ops.ming_image_01_ops import load_components
        load_components(trainer)

    def setup_block_swap(self, trainer):
        from core.models.ming_image_01.block_offload import setup_training_block_swap
        setup_training_block_swap(trainer)

    def setup_attention_backend(self, trainer):
        from core.models.ming_image_01.attention import install_attention
        install_attention(trainer.transformer, trainer._resolve_training_backend(trainer.attention_backend))

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

    def train_step(self, trainer, ctx):
        from core.training.ops import ming_image_01_ops
        if ctx.ming_batch is None:
            raise ValueError("Ming training requires an aligned native RGBA/dual-conditioning batch")
        return ming_image_01_ops.train_step(trainer, ctx.ming_batch, timesteps=ctx.timesteps, cfg_drop_mask=ctx.cfg_drop_mask)

    def sample(self, trainer, sample_ctx):
        from core.training.ops import ming_image_01_ops
        return ming_image_01_ops.generate_sample(trainer, sample_ctx,
            step_progress_callback=sample_ctx.step_progress_callback)
