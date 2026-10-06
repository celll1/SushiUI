"""MiniT2I pixel/latent training, per-checkpoint geometry and codec migration."""

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
    # No velocity: the network emits x_0 (ops/minit2i_ops.py train_step) and the
    # target is (images - x_t)/clamp(1-t), a scaled difference against the NOISY
    # sample -- see ArchHandler.velocity_sign.
    velocity_sign = None
    consumes_reconstruction_loss_weight = True
    consumes_crop_decode_loss = False
    supplies_predicted_latent = False

    def __init__(self, trainer=None):
        super().__init__(trainer)
        if trainer is not None:
            self.resolve_wiring(trainer)

    def resolve_wiring(self, trainer):
        """Resolve native patch layout separately from the external codec geometry."""
        from core.models.minit2i.minit2i_vae import VAE_SCALE_FACTOR

        cfg = getattr(getattr(trainer, "transformer", None), "mmjit_config", None)
        if cfg is None:
            return self.wiring
        channels, patch = int(cfg.in_channels), int(cfg.patch_size)
        latent_io = self.wiring.latent_io.replace(pack_elems=patch * patch)
        from core.models.components.latent_adapter import NormalizedImageVAE
        vae = getattr(trainer, "vae", None)
        if isinstance(vae, NormalizedImageVAE):
            self.pixel_align = vae.spatial_scale_factor * patch
            return self.wiring.replace(latent_channels=vae.config.latent_channels,
                latent_ndim=4, vae_scale_factor=vae.spatial_scale_factor,
                vae_norm="identity", vae_norm_pack=1, latent_io=latent_io)
        if str(getattr(cfg, "vae_type", "none")) == "none":
            self.pixel_align = patch
            return self.wiring.replace(latent_io=latent_io)
        self.pixel_align = VAE_SCALE_FACTOR * patch
        return self.wiring.replace(
            latent_channels=channels, vae_scale_factor=VAE_SCALE_FACTOR,
            vae_norm="shift_scale", latent_io=latent_io)

    def apply_vae_swap(self, trainer, resolved, module=None):
        """Preserve latent mode and saved bridges when restoring a custom codec."""
        from core.models.minit2i.minit2i_vae import is_latent_vae

        pair = getattr(trainer.transformer, "vae_latent_bridge", None)
        if pair is not None:
            if module is None and resolved.source != pair.declaration["source"]:
                raise ValueError("Changing an aligned MiniT2I VAE requires a fresh base checkpoint")
            from dataclasses import replace
            from core.models.components.latent_io import ResizeReport
            trainer.vae_identity = replace(resolved, state_dict=None)
            trainer.vae_latent_channels = pair.declaration["new_channels"]
            trainer.wiring = self.resolve_wiring(trainer)
            width = int(trainer.transformer.mmjit_config.in_channels)
            return ResizeReport((), width, width, width, 0, 0)
        from core.training.vae_swap import initialization_mode
        if module is None and initialization_mode(trainer.config, self.name) == "paired_spatial":
            from core.models.minit2i.vae_alignment import initialize_swap
            return initialize_swap(trainer, resolved, handler=self)
        if module is None and (resolved.ndim != 4 or resolved.scale_factor != 8):
            raise ValueError("Legacy zero MiniT2I swaps require a 4-D 8x VAE; use paired_spatial")

        report = super().apply_vae_swap(trainer, resolved, module=module)
        cfg = getattr(trainer.transformer, "mmjit_config", None)
        if cfg is not None:
            family = str(resolved.family or "")
            cfg.vae_type = family if is_latent_vae(family) else "custom"
        return report

    def calibrate_before_training(self, trainer, datasets, bucket_manager):
        from core.models.minit2i.vae_alignment import calibrate
        calibrate(trainer, datasets)

    def check_vae_compatibility(self, facts, *, trainer=None,
                                base_model_path=None):
        """Check image codecs against the base's per-checkpoint pixel/latent mode."""
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

        from core.models.minit2i.minit2i_vae import VAE_SCALE_FACTOR

        ndim = facts.get("ndim")
        if ndim not in (4, 5):
            return False, "MiniT2I swaps require a declared 4-D or independent-frame 5-D image codec"
        scale = facts.get("scale_factor")
        if type(scale) is not int or scale < 1 or (scale % VAE_SCALE_FACTOR and VAE_SCALE_FACTOR % scale):
            return False, f"MiniT2I swaps require integer spatial scale relative to {VAE_SCALE_FACTOR}x"
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

    def repa_tap(self, trainer):
        # The tap sits on the MM-JiT net (transformer -> DiffusionModel -> MMJiT),
        # which is where the double-block loop can stash its image stream; the tap
        # index addresses double_blocks only, so depth is depth_double.
        from core.training.repa import RepaTapPoint

        cfg = trainer.transformer.mmjit_config
        return RepaTapPoint(module=trainer.transformer.model.net,
                            hidden_size=int(cfg.hidden_size),
                            depth=int(cfg.depth_double))

    def depth_blocks(self, trainer):
        # Forward order (vendor/mmjit.py): the text preamble blocks, then the
        # MM-JiT double blocks.
        net = getattr(getattr(getattr(trainer, "transformer", None), "model", None),
                      "net", None)
        if net is None:
            return None
        preamble = list(getattr(net, "txt_preamble_blocks", None) or [])
        double = list(getattr(net, "double_blocks", None) or [])
        return (preamble + double) or None

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
            cfg_schedule_type=sample_ctx.cfg_schedule_type,
            cfg_schedule_min=sample_ctx.cfg_schedule_min,
            cfg_schedule_max=sample_ctx.cfg_schedule_max,
            cfg_schedule_power=sample_ctx.cfg_schedule_power,
            step_progress_callback=sample_ctx.step_progress_callback,
        )
