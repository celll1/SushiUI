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
    settings = getattr(trainer, "config", None) or {}
    if settings.get("_sensenova_explicit_tasks"):
        return "generation_flow" in set(
            settings.get("sensenova_train_scopes") or ()
        )
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
    # No velocity: the network emits x_0 (ops/sensenova_ops.py train_step) and the
    # target is (x0_tokens - z)/clamp(1-t), a scaled difference against the NOISY
    # sample -- see ArchHandler.velocity_sign.
    velocity_sign = None
    consumes_reconstruction_loss_weight = True
    consumes_crop_decode_loss = True
    supplies_predicted_latent = True

    @property
    def pixel_align(self) -> int:
        """One token's pixel width: 32 in pixel space, ``patch * scale`` after a swap.

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

    @staticmethod
    def _resolve_gen_patch(config, *, base_patch=None) -> int:
        """The run's generation patch, in latent cells (``sensenova_gen_patch``).

        The served default is ``INHERIT_GEN_PATCH`` (0): it resolves to
        ``base_patch``, so only a positive value asks to rebuild the grid.
        """
        from api.param_defaults import TRAINING_DEFAULTS
        from core.models.sensenova.latent_space import resolve_gen_patch

        value = config.get("sensenova_gen_patch")
        if value is None:
            value = TRAINING_DEFAULTS["sensenova_gen_patch"]
        return resolve_gen_patch(value, base_patch=base_patch,
                                 label="sensenova_gen_patch")

    @staticmethod
    def _warn_off_calibration(trainer, geometry, config) -> None:
        """A patch other than the native 4 moves the token count the checkpoint's
        own schedule is calibrated against, and nothing recalibrates it.

        ``compute_noise_scale`` is ``sqrt(tokens / base) * noise_scale``, so a
        coarser patch lowers it as ``1/patch``. The `standard` timestep arm is
        NOT affected: ``_apply_time_schedule`` assigns ``time_schedule =
        "standard"`` on entry, so ``_calculate_dynamic_mu`` -- the only other
        token-count-dependent term -- is unreachable in this repo.
        """
        from core.models.sensenova.latent_space import NATIVE_GEN_LATENT_PATCH

        # The calibrated grid is the pixel model's 32px token, i.e. P=4 on an 8x
        # VAE -- a structural reference, not whatever the config defaults to
        # (which is the inherit sentinel).
        default_patch = NATIVE_GEN_LATENT_PATCH
        if geometry.patch == default_patch:
            return
        from core.models.sensenova.sensenova_pipeline_ops import compute_noise_scale
        from core.training.training_events import emit_training_warning

        transformer = trainer.transformer
        merge = int(1 / transformer.downsample_ratio)
        align = geometry.token_pixel_width
        try:
            # train_runner's own fallback for a config without base_resolutions.
            reference = int((config.get("base_resolutions") or [1024])[0])
        except (TypeError, ValueError, IndexError):
            reference = 1024
        side = reference // align
        tokens = side * side
        grid = side * merge
        noise_scale = compute_noise_scale(transformer, grid, grid, merge)
        default_side = reference // (default_patch * geometry.vae_scale_factor)
        emit_training_warning(
            f"SenseNova sensenova_gen_patch={geometry.patch} (native "
            f"{default_patch}): one token covers {align}px, so a "
            f"{reference}px square is {tokens} tokens against "
            f"{default_side * default_side} at the native patch, and "
            f"compute_noise_scale returns {noise_scale:.4f} there. Both are "
            f"outside the band this checkpoint was trained on and NEITHER is "
            f"recalibrated: the noise-scale formula and its embedder are the "
            f"checkpoint's own. Quality at a coarser token is unmeasured.",
            code="sensenova_gen_patch_off_calibration",
            prefix=getattr(trainer, "log_prefix", "[SenseNova]"),
        )

    def apply_vae_swap(self, trainer, resolved, module=None):
        """SenseNova's own, because the shared resize does not apply (§10.6-1).

        Every other architecture's latent face is a channel-axis SLICE of the
        same tensor. Here the two tensors change shape in a way no partial copy
        expresses -- the ViT patch embed's KERNEL shrinks from 16x16 to
        ``patch/merge`` as well as its channel count -- so
        ``latent_space.apply_latent_geometry``
        rebuilds them under §10.3's initialisation instead.

        The 16 ``fm_modules`` tensors have never been optimised in this repo
        (measured byte-identical across 4,960 steps), so a swap without
        ``sensenova_train_fm_modules`` would leave both new layers at their
        initialisation for the whole run and save a model that predicts a
        constant.
        """
        from core.models.sensenova.latent_space import (
            apply_latent_geometry, gen_geometry, latent_config_dict,
            stamp_vae_scale_factor,
        )
        from core.training.ops.training_method import (
            is_full_finetune, resolve_training_method,
        )

        config = getattr(trainer, "config", None) or {}
        init = str(config.get("vae_swap_new_channel_init") or "zero")
        geometry = gen_geometry(trainer.transformer)
        # 0 (INHERIT_GEN_PATCH, the served default) keeps a latent base's own
        # patch: update_training_run sends every Pydantic default, so editing a
        # coarse-patch run in the UI must not arrive here as "patch 4", rebuild
        # the two layers it had trained, and say nothing. A positive value
        # rebuilds -- that one was asked for.
        patch = self._resolve_gen_patch(
            config, base_patch=geometry.patch if geometry.is_latent else None)
        # A base that already declares this latent space was BUILT in it and its
        # weights are loaded: rebuilding would throw away the two trained layers.
        rebuild = not (geometry.channels == resolved.latent_channels
                       and geometry.patch == patch
                       and geometry.vae_scale_factor == resolved.scale_factor)
        if rebuild and not is_full_finetune(trainer):
            # Reachable without a vae_swap_source (which capability already
            # refuses for these methods): a patch change against an ALREADY
            # swapped base arrives here from load_components' declared-VAE path.
            raise ValueError(
                f"SenseNova rebuilds its latent I/O only under a full "
                f"fine-tune, and this run's training_method is "
                f"{resolve_training_method(trainer)!r}: the generation ViT's "
                f"patch embed and the fm_head's output convolution are "
                f"REPLACED, not resized, and LoRA trains neither and saves "
                f"neither -- the new output convolution would stay at its zero "
                f"initialisation for the whole run, which also holds the "
                f"gradient to everything upstream of it at zero. The base is "
                f"{geometry.channels}ch at patch {geometry.patch} "
                f"({geometry.vae_scale_factor}x) and this run asks for "
                f"{resolved.latent_channels}ch at patch {patch} "
                f"({resolved.scale_factor}x). Leave sensenova_gen_patch at 0 to "
                f"train at the base's own geometry, or use Full Fine-tune.")
        if rebuild and not _fm_modules_trained(trainer):
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
                patch=patch,
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
        # The patch comes from the TREE, not from the request: the no-rebuild
        # branch keeps the base's own, and writing anything else rebuilds as a
        # different geometry on the next load.
        built = gen_geometry(trainer.transformer)
        trainer.sensenova_config_dict = latent_config_dict(
            getattr(trainer, "sensenova_config_dict", None),
            channels=resolved.latent_channels, patch=built.patch)
        self._apply_noise_scale(trainer, config)
        self._warn_off_calibration(trainer, built, config)
        trainer.wiring = self.wiring.replace(
            latent_channels=resolved.latent_channels,
            vae_scale_factor=resolved.scale_factor,
            vae_norm=resolved.norm,
            vae_norm_pack=resolved.norm_pack,
        )
        trainer.vae_identity = resolved
        trainer.vae_latent_channels = resolved.latent_channels
        return report

    @staticmethod
    def resolve_noise_scale_config(config) -> tuple:
        """``(gain, auto)`` for this run, refusing the contradiction.

        ``gain`` is 0 when the run does not ask for an explicit one
        (``INHERIT_NOISE_SCALE_GAIN``).
        """
        from api.param_defaults import TRAINING_DEFAULTS
        from core.training.train_runner import (
            _normalize_sensenova_bool, _normalize_sensenova_float,
        )

        config = dict(config or {})
        gain = _normalize_sensenova_float(
            config, "sensenova_noise_scale_gain",
            float(TRAINING_DEFAULTS["sensenova_noise_scale_gain"]))
        if gain < 0:
            raise ValueError(
                f"sensenova_noise_scale_gain must be >= 0 "
                f"(0 = inherit the checkpoint's own value), got {gain}")
        auto = _normalize_sensenova_bool(
            config, "sensenova_noise_scale_auto",
            bool(TRAINING_DEFAULTS["sensenova_noise_scale_auto"]))
        if auto and gain:
            raise ValueError(
                f"sensenova_noise_scale_auto measures the gain from this run's "
                f"own data and sensenova_noise_scale_gain={gain} states one; "
                f"exactly one of them can decide the noise schedule. Clear the "
                f"gain to measure, or turn auto off to use the stated value.")
        return gain, auto

    def _apply_noise_scale(self, trainer, config) -> None:
        """The explicit gain (§10.4). The measured one lands later, from
        ``calibrate_before_training`` -- it needs the run's datasets, which do
        not exist until ``train()``."""
        from core.models.sensenova.latent_space import apply_noise_scale_gain

        gain, _auto = self.resolve_noise_scale_config(config)
        if not gain:
            return
        self._require_full_finetune_for_noise_scale(
            trainer, f"sensenova_noise_scale_gain={gain:g}")
        trainer.sensenova_config_dict = apply_noise_scale_gain(
            trainer.transformer, getattr(trainer, "sensenova_config_dict", None),
            gain, provenance="config")
        print(f"{getattr(trainer, 'log_prefix', '[SenseNova]')} generation noise "
              f"scale recalibrated x{gain:g} -> {trainer.transformer.noise_scale:g} "
              f"(from the checkpoint's pre-recalibration "
              f"{trainer.sensenova_config_dict['gen_noise_scale_base']:g})")
        self._warn_noise_scale_clamp(trainer, config)

    @staticmethod
    def _require_full_finetune_for_noise_scale(trainer, setting: str) -> None:
        """A LoRA reaches here through a latent base's OWN vae declaration, with
        no vae_swap_source for the capability gate to key on -- and
        SenseNovaLoRAAdapter saves no config block, so the recalibrated scale
        would exist for the run and for nothing that loads its output."""
        from core.training.ops.training_method import (
            is_full_finetune, resolve_training_method,
        )

        if is_full_finetune(trainer):
            return
        raise ValueError(
            f"SenseNova {setting} requires a full fine-tune and this run's "
            f"training_method is {resolve_training_method(trainer)!r}: the "
            f"recalibrated noise scale is written into the checkpoint's own "
            f"config block, which a LoRA save does not write, so the run would "
            f"train on one noise schedule and every inference of the result "
            f"would use the base's.")

    @staticmethod
    def _warn_noise_scale_clamp(trainer, config) -> None:
        """`compute_noise_scale` ends in `min(scale, noise_scale_max_value)`, so a
        gain large enough to reach the ceiling is silently swallowed there."""
        from core.models.sensenova.latent_space import gen_geometry
        from core.models.sensenova.sensenova_pipeline_ops import compute_noise_scale
        from core.training.training_events import emit_training_warning

        transformer = trainer.transformer
        ceiling = float(getattr(transformer, "noise_scale_max_value", 0) or 0)
        if not ceiling:
            return
        geometry = gen_geometry(transformer)
        merge = int(1 / transformer.downsample_ratio)
        try:
            reference = int(((config or {}).get("base_resolutions") or [1024])[0])
        except (TypeError, ValueError, IndexError):
            reference = 1024
        side = max(1, reference // geometry.token_pixel_width)
        scale = compute_noise_scale(transformer, side * merge, side * merge, merge)
        if scale < ceiling:
            return
        emit_training_warning(
            f"SenseNova recalibrated generation noise scale reaches the "
            f"checkpoint's noise_scale_max_value ({ceiling:g}) at {reference}px, "
            f"where compute_noise_scale clamps it. Part of the gain is not "
            f"applied, and the clamp flattens the schedule across resolutions "
            f"above this one.",
            code="sensenova_noise_scale_clamped",
            prefix=getattr(trainer, "log_prefix", "[SenseNova]"),
        )

    def calibrate_before_training(self, trainer, datasets, bucket_manager) -> None:
        """Measure the noise-scale gain from this run's own latents, once.

        Runs from ``train()`` because it needs the datasets. A checkpoint that
        already carries a recalibrated scale INHERITS it: re-measuring on every
        resume would move the objective each time the run restarts.
        """
        from core.models.sensenova.latent_space import (
            PIXEL_RMS, apply_noise_scale_gain, noise_scale_gain_for_rms,
            recalibrated_noise_scale_gain,
        )
        from core.training.ops import sensenova_ops

        log = getattr(trainer, "log_prefix", "[SenseNova]")
        _gain, auto = self.resolve_noise_scale_config(getattr(trainer, "config", None))
        if not auto:
            return
        self._require_full_finetune_for_noise_scale(
            trainer, "sensenova_noise_scale_auto")
        if getattr(trainer, "vae", None) is None:
            raise ValueError(
                "sensenova_noise_scale_auto measures the RMS of this run's "
                "LATENTS, and this run is pixel-space (no VAE swap): its data "
                "scale is the one the checkpoint is already calibrated for.")
        config_dict = getattr(trainer, "sensenova_config_dict", None)
        carried = recalibrated_noise_scale_gain(config_dict)
        if carried is not None:
            print(f"{log} generation noise scale already recalibrated "
                  f"(x{carried:g}); inheriting it rather than re-measuring.")
            return
        rms = sensenova_ops.measure_latent_rms(trainer, datasets)
        gain = noise_scale_gain_for_rms(rms)
        trainer.sensenova_config_dict = apply_noise_scale_gain(
            trainer.transformer, config_dict, gain, provenance="measured")
        print(f"{log} measured latent RMS {rms:.4f} against the pixel "
              f"checkpoint's {PIXEL_RMS:.4f}: generation noise scale "
              f"recalibrated x{gain:.4f} -> {trainer.transformer.noise_scale:g}")
        self._warn_noise_scale_clamp(trainer, getattr(trainer, "config", None))

    def lora_adapter_class(self):
        from core.training.adapters import SenseNovaLoRAAdapter
        return SenseNovaLoRAAdapter

    def load_components(self, trainer) -> None:
        from core.training.ops import sensenova_ops

        sensenova_ops.load_components(trainer)

    def setup_block_swap(self, trainer) -> None:
        from core.training.ops import sensenova_ops

        sensenova_ops.setup_block_swap(trainer)

    def depth_blocks(self, trainer):
        # One depth axis for both MoT halves: the generation and understanding
        # weights of layer j live inside the same decoder layer.
        model = getattr(getattr(trainer, "transformer", None), "language_model", None)
        return getattr(getattr(model, "model", None), "layers", None)

    def repa_tap(self, trainer):
        # The tap is the Qwen3 decoder the GENERATION forward runs against the
        # understanding half's prefix K/V (ops/sensenova_ops.train_step ->
        # forward_gen_decoder_layers), whose loop stashes the generation stream.
        # Same module and same layer list depth_blocks reports: one depth axis
        # for both MoT halves.
        from core.training.repa import RepaTapPoint

        model = trainer.transformer.language_model.model
        # The width is read off the live gen-branch final norm rather than
        # config.hidden_size, so a config that no longer describes the loaded
        # tree cannot size the projector.
        weight = getattr(getattr(model, "norm_mot_gen", None), "weight", None)
        if weight is None:
            raise ValueError(
                "SenseNova's REPA tap needs the generation-branch final norm "
                "(language_model.model.norm_mot_gen) to read the decoder width "
                "from; this tree is not the MoT decoder training runs on.")
        return RepaTapPoint(module=model,
                            hidden_size=int(weight.shape[-1]),
                            depth=len(model.layers))

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

        if ctx.sensenova_text_batch is not None:
            return sensenova_ops.train_i2t_step(
                trainer, examples=ctx.sensenova_text_batch
            )
        return sensenova_ops.train_step(
            trainer,
            images=ctx.latents,
            prefix=ctx.sensenova_prefix,
            timesteps=ctx.timesteps,
            profile_vram=ctx.profile_vram,
            debug_save_path=ctx.debug_save_path,
            debug_captions=ctx.debug_captions,
            debug_reference_image_paths=ctx.debug_reference_image_paths,
            repa_pixels=ctx.repa_pixels,
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
