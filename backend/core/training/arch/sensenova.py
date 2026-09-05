"""SenseNova B1 LoRA training architecture handler."""

from __future__ import annotations

from core.training.arch.base_arch import (
    ArchHandler, PHASE2_PENDING, SampleContext, TrainStepContext,
    QUANTIZED_ADDITIVE_SHIPPED, declare_adapter_capability,
)
from core.training.components.wiring import SENSENOVA_WIRING


def _fm_modules_trained(trainer) -> bool:
    """Will this run actually optimise ``transformer.fm_modules``?

    Both halves of the adapter's own condition (``_fm_parameters``): the option,
    and a branch that includes the generation half -- fm_modules is
    generation-side, and an understanding-only run collects none of it.
    """
    from core.training.ops.sensenova_ops import resolve_full_finetune_branch

    if not bool(getattr(trainer, "sensenova_train_fm_modules", False)):
        return False
    return resolve_full_finetune_branch(trainer) in ("gen", "both")


class SenseNovaArchHandler(ArchHandler):
    name = "sensenova"
    wiring = SENSENOVA_WIRING
    adapter_capability = declare_adapter_capability(
        "sensenova",
        additive_family=True,
        # Generation takes LoHa/LoKr; training does not, and the gate is this
        # architecture's own rather than the general Phase 2 step.
        additive_gated=True,
        initial_dora="deferred",
        additive_reason=PHASE2_PENDING,
        quantized_base_additive_family=True,
        quantized_base_reason=(
            f"{QUANTIZED_ADDITIVE_SHIPPED}. Here that is ALL 294 targets per "
            f"MoT half: every one is an Int8Linear"),
    )
    wires_sample_step_progress = True
    # The inference uncond branch is a different PROMPT, not a rewrite of an
    # encoded one, and its token count also lands in every image token's t
    # coordinate (`_build_t2i_image_indexes`), so the null can only be built
    # while encoding the item. Mirrored for the API process by
    # api/arch_capabilities.CFG_NULL_STAGE_BY_ARCH.
    cfg_null_stage = "encode"
    # z_image = t*x0 + (1-t)*noise (ops/sensenova_ops.py train_step, and
    # sensenova_pipeline_ops.py at inference). sampler t=1 is clean -- the
    # inverse of the SD3/FLUX-style default.
    timestep_convention = "t1"

    @property
    def pixel_align(self) -> int:
        """One token's pixel width: 32 in pixel space, ``4 * scale`` after a swap.

        Read off the loaded tree rather than declared, because the generation
        grid is a per-checkpoint fact here (design §10.2). Every reader runs
        after ``load_components``; the constant is the fallback for the handlers
        constructed without a trainer (the preflight compatibility gate).
        """
        transformer = getattr(getattr(self, "trainer", None), "transformer", None)
        if transformer is None:
            return 32
        from core.models.sensenova.latent_space import token_pixel_width

        return int(token_pixel_width(transformer))

    def resolve_wiring(self, trainer):
        """SENSENOVA_WIRING describes the PIXEL variant; a swapped checkpoint
        faces a latent whose channel count and compression are its own."""
        transformer = getattr(trainer, "transformer", None)
        if transformer is None:
            return self.wiring
        from core.models.sensenova.latent_space import gen_geometry

        geometry = gen_geometry(transformer)
        if not geometry.is_latent:
            return self.wiring
        return self.wiring.replace(
            latent_channels=geometry.channels,
            vae_scale_factor=geometry.vae_scale_factor,
        )

    def check_vae_compatibility(self, facts, *, trainer=None, base_model_path=None):
        """D13: any spatial compression is accepted, 4-D image latents only.

        The shared gate already exempts SenseNova from the ratio check (it is
        the one architecture whose swap CHANGES the ratio, from 1); what it
        cannot see is that the destination is a 2-D image grid.
        """
        ndim = facts.get("ndim")
        if ndim is not None and int(ndim) != 4:
            return False, (f"{ndim}-D latents cannot drive sensenova, whose "
                           f"generation grid is a 2-D image token grid")
        temporal = facts.get("scale_temporal")
        if temporal is not None and int(temporal) != 1:
            return False, (f"temporal compression {temporal}x cannot drive "
                           f"sensenova, which generates stills")
        return super().check_vae_compatibility(
            facts, trainer=trainer, base_model_path=base_model_path)

    def apply_vae_swap(self, trainer, resolved, module=None):
        """SenseNova's own, because the shared resize does not apply (§10.6-1).

        Every other architecture's latent face is a channel-axis SLICE of the
        same tensor. Here the two tensors change shape in a way no partial copy
        expresses -- the ViT patch embed's KERNEL shrinks from 16x16 to 2x2 as
        well as its channel count -- so ``latent_space.apply_latent_geometry``
        rebuilds them under §10.3's initialisation instead.

        The 16 ``fm_modules`` tensors have never been optimised in this repo
        (measured byte-identical across 4,960 steps), so a swap without
        ``sensenova_train_fm_modules`` would leave both new layers at their
        initialisation for the whole run and save a model that predicts a
        constant.
        """
        from core.models.sensenova.latent_space import (
            GEN_LATENT_PATCH, apply_latent_geometry, gen_geometry,
            latent_config_dict, stamp_vae_scale_factor,
        )
        from core.training.ops.training_method import is_full_finetune

        config = getattr(trainer, "config", None) or {}
        init = str(config.get("vae_swap_new_channel_init") or "zero")
        geometry = gen_geometry(trainer.transformer)
        # A base that already declares this latent space was BUILT in it and its
        # weights are loaded: rebuilding would throw away the two trained layers.
        rebuild = not (geometry.channels == resolved.latent_channels
                       and geometry.patch == GEN_LATENT_PATCH
                       and geometry.vae_scale_factor == resolved.scale_factor)
        if rebuild and is_full_finetune(trainer) and not _fm_modules_trained(trainer):
            raise ValueError(
                "a SenseNova VAE swap requires sensenova_train_fm_modules: the "
                "swap rebuilds the generation ViT's patch embed and the "
                "fm_head's output convolution, and both live in "
                "transformer.fm_modules, which the default full fine-tune scope "
                "(the 294 decoder Linears per half) never optimises. Without it "
                "the run would train with a zero head for its whole duration.")

        trainer.vae = module if module is not None else resolved.load_module(
            torch_dtype=getattr(trainer, "vae_dtype", None))
        if rebuild:
            report = apply_latent_geometry(
                trainer.transformer,
                channels=resolved.latent_channels,
                vae_scale_factor=resolved.scale_factor,
                head_init=init,
            )
        else:
            from core.models.components.latent_io import ResizeReport

            stamp_vae_scale_factor(trainer.transformer, resolved.scale_factor)
            report = ResizeReport(
                replaced=(), old_in_channels=geometry.channels,
                old_out_channels=geometry.channels,
                new_channels=resolved.latent_channels,
                copied_elements=0, new_elements=0)
        # The export re-embeds the config block this load accepted, verbatim.
        trainer.sensenova_config_dict = latent_config_dict(
            getattr(trainer, "sensenova_config_dict", None),
            channels=resolved.latent_channels)
        trainer.wiring = self.wiring.replace(
            latent_channels=resolved.latent_channels,
            vae_scale_factor=resolved.scale_factor,
            vae_norm=resolved.norm,
            vae_norm_pack=resolved.norm_pack,
        )
        trainer.vae_identity = resolved
        trainer.vae_latent_channels = resolved.latent_channels
        return report

    def lora_adapter_class(self):
        from core.training.adapters import SenseNovaLoRAAdapter
        return SenseNovaLoRAAdapter

    def load_components(self, trainer) -> None:
        from core.training.ops import sensenova_ops

        sensenova_ops.load_components(trainer)

    def setup_block_swap(self, trainer) -> None:
        raise NotImplementedError("SenseNova training block swap is not implemented")

    def setup_attention_backend(self, trainer) -> None:
        from core.training.ops import sensenova_ops

        sensenova_ops.setup_attention_backend(trainer, trainer.attention_backend)

    def encode_prompt(
        self, trainer, prompt, *, requires_grad: bool = False, reference_image_paths=None
    ):
        from core.training.ops import sensenova_ops

        return sensenova_ops.encode_prompt(
            trainer,
            prompt,
            requires_grad=requires_grad,
            reference_image_paths=reference_image_paths,
        )

    def encode_prompts(
        self, trainer, prompts, *, requires_grad: bool = False,
        reference_image_paths=None, cfg_null=None,
    ):
        """One packed prefix for a physical batch (``sensenova_ops.encode_prompts``)."""
        from core.training.ops import sensenova_ops

        return sensenova_ops.encode_prompts(
            trainer,
            list(prompts),
            requires_grad=requires_grad,
            reference_image_paths=reference_image_paths,
            cfg_null=cfg_null,
        )

    def encode_prompt_cfg_null(
        self, trainer, prompt, *, requires_grad: bool = False,
        reference_image_paths=None, **kwargs
    ):
        """The same encode, with inference's uncond query in place of ``prompt``.

        ``prompt`` is accepted and ignored on purpose: the null must not depend
        on the caption, and the per-item Bernoulli that selects this path is
        drawn before the caption is read.
        """
        from core.training.ops import sensenova_ops

        if kwargs:
            raise TypeError(
                f"SenseNova's aligned null encode does not accept "
                f"{sorted(kwargs)}"
            )
        return sensenova_ops.encode_prompt(
            trainer,
            prompt,
            requires_grad=requires_grad,
            reference_image_paths=reference_image_paths,
            cfg_null=True,
        )

    def vae_encode(
        self,
        trainer,
        image_tensor,
        *,
        image=None,
        width=None,
        height=None,
        vae_device=None,
        debug_preprocessing=False,
    ):
        from core.training.ops import sensenova_ops

        return sensenova_ops.vae_encode(trainer, image_tensor)

    def vae_decode(self, trainer, latents, *, latent_h, latent_w):
        from core.training.ops import sensenova_ops

        return sensenova_ops.vae_decode(trainer, latents)

    def train_step(self, trainer, ctx: TrainStepContext):
        from core.training.ops import sensenova_ops

        return sensenova_ops.train_step(
            trainer,
            images=ctx.latents,
            prefix=ctx.sensenova_prefix,
            timesteps=ctx.timesteps,
            profile_vram=ctx.profile_vram,
            debug_save_path=ctx.debug_save_path,
            debug_captions=ctx.debug_captions,
            debug_reference_image_paths=ctx.debug_reference_image_paths,
        )

    def sample(self, trainer, sample_ctx: SampleContext):
        from core.training.ops import sensenova_ops

        return sensenova_ops.generate_sample(
            trainer,
            prompt=sample_ctx.prompt,
            height=sample_ctx.height,
            width=sample_ctx.width,
            num_inference_steps=sample_ctx.num_inference_steps,
            guidance_scale=sample_ctx.guidance_scale,
            seed=sample_ctx.seed,
            negative_prompt=sample_ctx.negative_prompt,
            reference_image_path=sample_ctx.reference_image_path,
            condition_image_path=sample_ctx.condition_image_path,
            timestep_shift=sample_ctx.sensenova_timestep_shift,
            img_cfg_scale=sample_ctx.sensenova_img_cfg_scale,
            cfg_norm=sample_ctx.sensenova_cfg_norm,
            step_progress_callback=sample_ctx.step_progress_callback,
        )
