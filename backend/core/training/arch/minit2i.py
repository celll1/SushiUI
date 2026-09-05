"""
MiniT2IArchHandler — P0/P1 stub handler for arch "minit2i".

Empty subclass: name + wiring spec are real; every canonical method raises
NotImplementedError. Bodies are moved from base_trainer.py in phases P3-P7.
Nothing calls these until a later phase flips the corresponding dispatcher.
"""

from __future__ import annotations

from core.training.arch.base_arch import (
    ArchHandler, SampleContext, TrainStepContext, resolve_scope_csv,
    PHASE2_PENDING, QUANTIZED_ADDITIVE_PENDING,
    declare_adapter_capability,
)
from core.training.components.wiring import MINIT2I_WIRING


class MiniT2IArchHandler(ArchHandler):
    name = "minit2i"
    wiring = MINIT2I_WIRING
    adapter_capability = declare_adapter_capability(
        "minit2i",
        additive_family=True,
        initial_dora="dense",
        additive_reason=PHASE2_PENDING,
        quantized_base_reason=QUANTIZED_ADDITIVE_PENDING,
    )
    pixel_align = 16  # GRID_ALIGN = patch_size(16); pixel-space patchify unit
    wires_sample_step_progress = True
    # The inference uncond branch reuses the SAME text tensor with a zeroed mask
    # (minit2i_pipeline_ops._predict_x0_cfg), so the aligned null is reachable by
    # rewriting the already-collated conditioning. Mirrored for the API process
    # by api/arch_capabilities.CFG_NULL_STAGE_BY_ARCH.
    cfg_null_stage = "collated"
    # x_t = images*t + noise*(1-t) (ops/minit2i_ops.py train_step, explicitly
    # documented there as "t=1 data, t=0 noise"). sampler t=1 is clean -- the
    # inverse of the SD3/FLUX-style default.
    timestep_convention = "t1"

    def resolve_wiring(self, trainer):
        """MINIT2I_WIRING describes the PIXEL variant; a latent checkpoint has a
        different channel count, patch size and compression ratio, all of which
        are config values in the file (``vendor/mmjit.MMJiTConfig``). Read them,
        so the resize slices with the checkpoint's own P and the run's wiring
        reports its own geometry.
        """
        from core.models.minit2i.minit2i_vae import VAE_SCALE_FACTOR

        cfg = getattr(getattr(trainer, "transformer", None), "mmjit_config", None)
        if cfg is None:
            return self.wiring
        channels, patch = int(cfg.in_channels), int(cfg.patch_size)
        latent_io = self.wiring.latent_io.replace(pack_elems=patch * patch)
        if str(getattr(cfg, "vae_type", "none")) == "none":
            return self.wiring.replace(latent_io=latent_io)
        return self.wiring.replace(
            latent_channels=channels, vae_scale_factor=VAE_SCALE_FACTOR,
            vae_norm="shift_scale", latent_io=latent_io)

    def apply_vae_swap(self, trainer, resolved, module=None):
        """The shared resize, plus MiniT2I's own record of which VAE it uses.

        ``vae_type`` is a config field of the checkpoint (``MMJiTConfig``), and
        the loader reads it to decide whether the model is latent at all, so a
        swap that moved only the weights would reload as pixel-space. A source
        with no registry family is recorded as ``"custom"``; the checkpoint's
        ``component.vae.*`` block is what resolves it on the next load.
        """
        from core.models.minit2i.minit2i_vae import is_latent_vae

        report = super().apply_vae_swap(trainer, resolved, module=module)
        cfg = getattr(trainer.transformer, "mmjit_config", None)
        if cfg is not None:
            family = str(resolved.family or "")
            cfg.vae_type = family if is_latent_vae(family) else "custom"
        return report

    def check_vae_compatibility(self, facts, *, trainer=None,
                                base_model_path=None):
        """The family gate against THIS checkpoint's geometry.

        ``vae_source.check_vae_compatibility`` answers from the arch's wiring
        constant, which for MiniT2I is the pixel variant -- so it refuses every
        candidate for every checkpoint, including the latent ones a swap is for.
        The pixel refusal is real and stays: moving a pixel checkpoint into a
        latent space changes patch_size as well as the channel count, which is
        not a channel resize (design 5.1, and 10 for the arch that does it).
        """
        io_config = self._io_config(trainer, base_model_path)
        if not io_config:
            return False, ("MiniT2I's latent geometry is a per-checkpoint config "
                           "value and this base's could not be read, so a "
                           "replacement VAE cannot be checked against it")
        if str(io_config.get("vae_type", "none")) == "none":
            return False, (
                "this MiniT2I checkpoint is pixel-space (in_channels=3, "
                "patch_size=16); moving it into a latent space changes the patch "
                "geometry as well as the channel count, which is not a channel "
                "resize. Train a latent variant (scratch:minit2i:<variant>:sdxl "
                "or :flux1) and swap that")

        # The same three structural questions the shared gate asks, against the
        # checkpoint's geometry instead of the arch table it cannot express.
        from core.models.minit2i.minit2i_vae import VAE_SCALE_FACTOR

        ndim = facts.get("ndim")
        if ndim is not None and ndim != 4:
            return False, (f"{ndim}-D latents cannot drive minit2i, which expects "
                           f"4-D")
        scale = facts.get("scale_factor")
        if scale is not None and int(scale) != VAE_SCALE_FACTOR:
            return False, (f"spatial compression {scale}x differs from MiniT2I's "
                           f"{VAE_SCALE_FACTOR}x")
        temporal = facts.get("scale_temporal")
        if temporal is not None and int(temporal) != 1:
            return False, (f"temporal compression {temporal}x differs from "
                           f"MiniT2I's 1x")
        return True, None

    @staticmethod
    def _io_config(trainer, base_model_path):
        from core.models.minit2i.minit2i_loader import peek_io_config

        cfg = getattr(getattr(trainer, "transformer", None), "mmjit_config", None)
        if cfg is not None:
            return {"in_channels": int(cfg.in_channels),
                    "patch_size": int(cfg.patch_size),
                    "vae_type": str(getattr(cfg, "vae_type", "none"))}
        path = base_model_path or getattr(trainer, "model_path", None)
        return peek_io_config(str(path or ""))

    def lora_adapter_class(self):
        from core.training.adapters import MiniT2ILoRAAdapter
        return MiniT2ILoRAAdapter

    def lora_adapter_kwargs(self, trainer):
        from core.models.minit2i.minit2i_lora import (
            parse_scope_csv, parse_te_scope_csv,
        )
        return {
            "scope": parse_scope_csv(resolve_scope_csv(
                trainer, "minit2i_lora_scope", "attn,mlp,txt_embed")),
            "te_scope": parse_te_scope_csv(resolve_scope_csv(
                trainer, "minit2i_te_lora_scope", "attn,ff")),
        }

    def load_components(self, trainer) -> None:
        # P3c: body lives in ops/minit2i_ops (shared with the base_trainer load-time
        # dispatcher + _load_checkpoint_as_base, which cannot route via self.arch —
        # self.arch binds after loading; see the construction-order note in minit2i_ops).
        from core.training.ops import minit2i_ops
        minit2i_ops.load_components(trainer)

    def setup_block_swap(self, trainer) -> None:
        # P3c: body lives in ops/minit2i_ops (shared with the base_trainer
        # setup_minit2i_block_swap delegator, called late by mode subclasses).
        from core.training.ops import minit2i_ops
        minit2i_ops.setup_block_swap(trainer)

    def setup_attention_backend(self, trainer) -> None:
        # P3c: body lives in ops/minit2i_ops (shared with base_trainer delegator).
        from core.training.ops import minit2i_ops
        minit2i_ops.setup_attention_backend(trainer, trainer.attention_backend)

    def encode_prompt(self, trainer, prompt, *, requires_grad: bool = False):
        # P4: body lives in ops/minit2i_ops (shared with base_trainer delegator).
        from core.training.ops import minit2i_ops
        return minit2i_ops.encode_prompt(trainer, prompt, requires_grad=requires_grad)

    def vae_encode(self, trainer, image_tensor, *, image=None, width=None, height=None,
                   vae_device=None, debug_preprocessing=False):
        # P5: both minit2i sub-branches (pixel-space no-VAE + latent-space) live in
        # ops/minit2i_ops; self-contained (dispatched before the shared VAE staging).
        from core.training.ops import minit2i_ops
        return minit2i_ops.vae_encode(
            trainer, image_tensor, image=image, width=width, height=height,
            vae_device=vae_device, debug_preprocessing=debug_preprocessing,
        )

    def vae_decode(self, trainer, latents, *, latent_h, latent_w):
        raise NotImplementedError("minit2i.vae_decode: phase P5/P7")

    def apply_cfg_null_collated(self, trainer, conditioning, auxiliary,
                                drop_mask):
        from core.training.ops import minit2i_ops
        return minit2i_ops.apply_cfg_null_collated(
            conditioning, auxiliary, drop_mask)

    def train_step(self, trainer, ctx: TrainStepContext):
        # P6c: verbatim body in ops/minit2i_ops.train_step. ctx fields map 1:1 to
        # the previous train_step_minit2i kwargs bundle (mnt_latents is the
        # pixel-space image tensor; text_embeds/attention_mask carry FLAN-T5).
        from core.training.ops import minit2i_ops
        # Collated-stage rewrite, out of place: these conditioning tensors are
        # the batch's, reused by every MNT iteration.
        text_embeds, attention_mask = self.apply_cfg_null_step(
            trainer, ctx, ctx.text_embeddings, ctx.attention_mask)
        return minit2i_ops.train_step(
            trainer,
            images=ctx.latents,
            text_embeds=text_embeds,
            attention_mask=attention_mask,
            timesteps=ctx.timesteps,
            profile_vram=ctx.profile_vram,
            debug_save_path=ctx.debug_save_path,
            debug_captions=ctx.debug_captions,
            debug_reference_image_paths=ctx.debug_reference_image_paths,
            repa_pixels=ctx.repa_pixels,
        )

    def sample(self, trainer, sample_ctx: SampleContext):
        # P7: verbatim body in ops/minit2i_ops.generate_sample.
        from core.training.ops import minit2i_ops
        return minit2i_ops.generate_sample(
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
