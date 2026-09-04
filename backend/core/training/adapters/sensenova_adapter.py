"""SenseNova MoT training adapters (LoRA, and full parameter over one MoT half)."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

from torch import nn

from core.models.sensenova.sensenova_lora import (
    LORA_TARGET_LABELS,
    iter_sensenova_lora_targets,
)

from core.adapters import (
    CompositeAdapterLayer, LoRALinearLayer, is_lora_wrappable_linear,
)
from .base_adapter import (
    BaseFullParameterAdapter,
    BaseLoRAAdapter,
    LORA_COMPONENT_TEXT_ENCODER_1,
    LORA_COMPONENT_UNET,
    resolve_component_lr,
)

_TARGETS_PER_BRANCH = 294
_LAYERS = 42


class SenseNovaLoRAAdapter(BaseLoRAAdapter):
    """Wrap the 294 generation-branch Linears, plus 294 understanding ones.

    The understanding half is injected only when ``train_text_encoder`` is set
    (the parameter is REUSED, not duplicated: SenseNova's prompt encoder IS the
    understanding branch of the same LLM that denoises). Five of those 294 are
    structurally unreachable from an image loss and stay at their zero init --
    see ``sensenova_lora.und_gradient_unreachable_paths``.
    """

    @staticmethod
    def _expected_target_paths(branch: str = "gen") -> set[str]:
        paths = set()
        attn_names = (
            ("q_proj_mot_gen", "k_proj_mot_gen", "v_proj_mot_gen", "o_proj_mot_gen")
            if branch == "gen"
            else ("q_proj", "k_proj", "v_proj", "o_proj")
        )
        mlp_parent = "mlp_mot_gen" if branch == "gen" else "mlp"
        for layer_index in range(_LAYERS):
            prefix = f"language_model.model.layers.{layer_index}"
            paths.update(f"{prefix}.self_attn.{name}" for name in attn_names)
            paths.update(
                f"{prefix}.{mlp_parent}.{name}"
                for name in ("gate_proj", "up_proj", "down_proj")
            )
        return paths

    def _inject_branch(
        self, lora_layers: Dict[str, nn.Module], branch: str, component: str
    ) -> int:
        """Wrap one MoT half; identical shape for both, only the names differ."""
        label = "generation" if branch == "gen" else "understanding"
        transformer = getattr(self.trainer, "transformer", None)
        if transformer is None:
            raise RuntimeError("SenseNova LoRA requires a loaded transformer")

        targets = list(iter_sensenova_lora_targets(transformer, branch=branch))
        actual_paths = {module_path for module_path, *_ in targets}
        expected_paths = self._expected_target_paths(branch)
        if len(targets) != _TARGETS_PER_BRANCH or actual_paths != expected_paths:
            missing = sorted(expected_paths - actual_paths)
            extra = sorted(actual_paths - expected_paths)
            raise RuntimeError(
                f"SenseNova {label} LoRA requires exactly {_TARGETS_PER_BRANCH} targets "
                f"(missing={missing[:3]}, extra={extra[:3]})"
            )

        unwrapped = [target for target in targets if is_lora_wrappable_linear(target[3])]
        wrapped = [target for target in targets if isinstance(target[3], LoRALinearLayer)]
        # A composite is neither: it holds branches this adapter did not build,
        # so it can be neither wrapped (that would nest) nor registered as this
        # run's trainable layer. Refused by name rather than falling into the
        # "mixed or unsupported" count below, which would not say why.
        composites = [path for path, _, _, layer in targets
                      if isinstance(layer, CompositeAdapterLayer)]
        if composites:
            raise RuntimeError(
                f"SenseNova {label} LoRA cannot inject into {len(composites)} target(s) "
                f"already covered by a generation-side composite adapter "
                f"(first: {composites[0]}); unload those LoRAs first"
            )
        if len(wrapped) == _TARGETS_PER_BRANCH:
            mismatched_names = [
                path for path, _, _, layer in wrapped if layer.lora_name != path
            ]
            if mismatched_names:
                raise RuntimeError(
                    f"SenseNova {label} LoRA wrappers use the wrong namespace: "
                    f"{mismatched_names[:3]}"
                )
            for path, _, _, layer in wrapped:
                existing = lora_layers.get(path)
                if existing is not None and existing is not layer:
                    raise RuntimeError(
                        f"SenseNova LoRA registry conflicts with wrapper {path}"
                    )
                self.register_lora_layer(lora_layers, path, layer, component)
            return 0
        if len(unwrapped) != _TARGETS_PER_BRANCH:
            raise RuntimeError(
                f"SenseNova {label} LoRA target state is mixed or unsupported "
                f"(unwrapped={len(unwrapped)}, wrapped={len(wrapped)}, "
                f"total={_TARGETS_PER_BRANCH})"
            )

        count = 0
        for module_path, parent, attr, current in unwrapped:
            wrapper = self.build_branch(current, module_path)
            setattr(parent, attr, wrapper)
            self.register_lora_layer(lora_layers, module_path, wrapper, component)
            count += 1

        print(f"[SenseNovaLoRAAdapter] Injected {count} {label} LoRA layer(s)")
        return count

    def apply_lora_to_unet(self, lora_layers: Dict[str, nn.Module]) -> int:
        return self._inject_branch(lora_layers, "gen", LORA_COMPONENT_UNET)

    def apply_lora_to_text_encoders(self, lora_layers: Dict[str, nn.Module]) -> int:
        """Inject understanding-branch LoRA when ``train_text_encoder`` is set.

        Registered as ``text_encoder_1`` rather than ``unet`` so grad-norm
        reporting can tell the two halves apart, even though both live inside
        the same ``transformer`` module tree.
        """
        if not getattr(self.trainer, "train_text_encoder", False):
            print("[SenseNovaLoRAAdapter] Understanding branch is frozen - no text LoRA")
            return 0
        from core.training.ops.sensenova_ops import assert_understanding_training_supported

        assert_understanding_training_supported(self.trainer.transformer)
        return self._inject_branch(
            lora_layers, "und", LORA_COMPONENT_TEXT_ENCODER_1
        )

    def setup_trainable_parameters(
        self, lora_layers: Dict[str, nn.Module]
    ) -> List[Dict[str, Any]]:
        return self.component_param_groups(lora_layers, {
            LORA_COMPONENT_UNET: lambda: resolve_component_lr(
                self.trainer, "unet_lr", label="SenseNova generation branch"),
            # Same fallback chain SDXL's LoRA adapter uses for TE1.
            LORA_COMPONENT_TEXT_ENCODER_1: lambda: resolve_component_lr(
                self.trainer,
                "text_encoder_1_lr",
                "text_encoder_lr",
                "unet_lr",
                label="SenseNova understanding branch",
            ),
        })

    CHECKPOINT_LOG_FORMAT = (
        "[{adapter}] Saved LoRA checkpoint ({layers} layers, {lora_targets}) -> {path}"
    )

    def checkpoint_metadata(
        self, lora_layers: Dict[str, nn.Module], step: int, epoch: int
    ) -> Dict[str, str]:
        components = self.lora_components
        gen_count = sum(
            1
            for name in lora_layers
            if components.get(name, LORA_COMPONENT_UNET) == LORA_COMPONENT_UNET
        )
        if gen_count == 0:
            # An understanding-only LoRA has no consumer: inference applies both
            # branches from one file, and a generation-free one would be a
            # format nothing in this repo can produce a sample from.
            raise RuntimeError(
                "SenseNova LoRA checkpoints must carry the generation branch; "
                "an understanding-only LoRA is not a supported artefact"
            )
        branch = "both" if gen_count != len(lora_layers) else "gen"

        return {
            "model_type": "sensenova",
            "modelspec.architecture": "sensenova",
            "tensor_kind": "neo_hf_lora",
            "lora_rank": str(self.lora_rank),
            "lora_alpha": str(self.lora_alpha),
            "lora_targets": LORA_TARGET_LABELS[branch],
            "step": str(step),
            "epoch": str(epoch),
        }


class SenseNovaFullParameterAdapter(BaseFullParameterAdapter):
    """Train the decoder Linears the loader materialized, and nothing else.

    Everything comes from ``trainer.transformer``; ``trainer.text_encoder`` is
    never read, because ``load_components`` sets it to None and the generic
    full-FT collector gates every group it builds on that module (and on
    ``trainer.unet``, also None) being present -- falling through to it collects
    zero parameters while the loss falls normally.

    Scope is the loader's scope by construction: this adapter and
    ``load_components`` share ``resolve_full_finetune_branch`` and
    ``iter_sensenova_lora_targets``, so the dequantized set and the optimized set
    are one set rather than two that agree. ``train_unet`` is the generation
    half, ``train_text_encoder`` the understanding one.

    The generation modules outside the decoder (``fm_head``, the generation ViT,
    the embedders, the ``*_norm_mot_gen`` norms) are frozen by default: they are
    not quantized, so the loader does not materialize them, and including them
    unconditionally would break that identity. ``sensenova_train_fm_modules``
    (SENSENOVA_TRAINING_DESIGN.md, "Trained scope") opts the ``fm_modules``
    container back in, through ``_fm_parameters`` not ``_resolve_scope`` so the
    decoder-Linear count stays exactly what the loader materialized. The
    ``*_norm_mot_gen`` norms stay frozen either way.
    """

    def _fm_parameters(self, branch: str) -> List[nn.Parameter]:
        """``transformer.fm_modules`` parameters when the option is on, else [].

        16 tensors / 63,117,504 parameters on the real checkpoint: the
        generation ViT's patch and dense embeddings, the timestep and
        noise-scale embedders and the two ``fm_head`` convolutions. All
        generation-side, so they join the generation group and are collected
        only when that half is trained.
        """
        trainer = self.trainer
        if not bool(getattr(trainer, "sensenova_train_fm_modules", False)):
            return []
        if branch not in ("gen", "both"):
            if not getattr(self, "_fm_branch_warned", False):
                from core.training.training_events import emit_training_warning

                self._fm_branch_warned = True
                emit_training_warning(
                    "SenseNova sensenova_train_fm_modules is set, but this run "
                    f"trains the {branch!r} branch only. fm_modules (the "
                    "generation ViT embeddings, the timestep/noise-scale "
                    "embedders and fm_head) is generation-side, so it stays "
                    "frozen and the run proceeds with the decoder Linears alone.",
                    code="sensenova_train_fm_modules_branch_mismatch",
                    prefix=getattr(trainer, "log_prefix", "[SenseNova]"),
                )
            return []

        fm_modules = getattr(trainer.transformer, "fm_modules", None)
        if fm_modules is None:
            raise RuntimeError(
                "SenseNova sensenova_train_fm_modules is set but the transformer "
                "has no fm_modules container; this tree is not the NEOChatModel "
                "this route was built for."
            )
        parameters = list(fm_modules.parameters())
        if not parameters:
            raise RuntimeError(
                "SenseNova sensenova_train_fm_modules collected no parameter from "
                "fm_modules. Enabling an option that trains nothing is refused "
                "rather than run as a silent no-op."
            )
        # Same failure mode the decoder guard covers, asked of the real tensors
        # rather than assumed from "these are not quantized": a non-float or
        # buffer-held weight takes no gradient and would train nothing.
        bad = [
            f"{name} ({parameter.dtype})"
            for name, parameter in fm_modules.named_parameters()
            if not parameter.dtype.is_floating_point
        ]
        if bad:
            raise RuntimeError(
                f"SenseNova fm_modules holds {len(bad)} non-floating-point "
                f"parameter(s) (first: {bad[0]}); they cannot take a gradient."
            )
        return parameters

    def _resolve_scope(self) -> Tuple[str, List[Tuple[str, Any, str, nn.Module]]]:
        """(branch, targets) -- the same enumeration the loader materialized."""
        from core.models.sensenova.loader import SENSENOVA_BRANCH_LINEAR_COUNTS
        from core.training.ops.sensenova_ops import resolve_full_finetune_branch

        trainer = self.trainer
        transformer = getattr(trainer, "transformer", None)
        if transformer is None:
            raise RuntimeError(
                "SenseNova full fine-tuning requires a loaded transformer"
            )
        branch = resolve_full_finetune_branch(trainer)
        targets = list(iter_sensenova_lora_targets(transformer, branch=branch))
        expected = SENSENOVA_BRANCH_LINEAR_COUNTS[branch]
        if len(targets) != expected:
            raise RuntimeError(
                f"SenseNova full fine-tuning expects {expected} decoder Linear(s) "
                f"on the {branch} branch, found {len(targets)}"
            )

        # A composite delegates .weight to its base, so it would pass the
        # materialization check below and then hand requires_grad_(True) and
        # module.parameters() the generation LoRA's branches as well.
        covered = [path for path, _, _, module in targets
                   if isinstance(module, CompositeAdapterLayer)]
        if covered:
            raise RuntimeError(
                f"SenseNova full fine-tuning cannot train {len(covered)} decoder "
                f"Linear(s) covered by a generation-side composite adapter "
                f"(first: {covered[0]}); unload those LoRAs first"
            )

        unmaterialized = [
            path
            for path, _, _, module in targets
            if not isinstance(getattr(module, "weight", None), nn.Parameter)
        ]
        if unmaterialized:
            # The int8 modules hold weight and scale as buffers, so this is the
            # silent-no-op case: requires_grad_(True) below would do nothing and
            # the optimizer would see an empty parameter list.
            raise RuntimeError(
                f"SenseNova full fine-tuning found {len(unmaterialized)} of "
                f"{len(targets)} {branch}-branch decoder Linear(s) still holding "
                f"their weight as a buffer rather than a Parameter (first: "
                f"{unmaterialized[0]}). Those are the quantized modules the load "
                f"path is supposed to have dequantized "
                f"(loader.materialize_int8_decoder_linears); requires_grad_(True) "
                f"is a no-op on them, so training would collect nothing while the "
                f"loss fell normally."
            )
        return branch, targets

    def prepare_models_for_training(self):
        from core.training.ops.sensenova_ops import (
            assert_full_finetune_contract,
            assert_full_finetune_dropout_free,
            assert_understanding_training_supported,
        )

        trainer = self.trainer
        # Config channel only; setup_optimizer re-checks with the real name.
        assert_full_finetune_contract(trainer)
        # Resolved now rather than at the first save: save_every defaults to 100
        # steps, so an unknown format would take the run down after it had
        # already trained. train_runner refuses it earlier still; this covers a
        # trainer built directly.
        self._resolve_save_format()
        branch, targets = self._resolve_scope()
        # Every branch, not just the understanding ones: load_components stamps
        # train() on the whole decoder, and the prefix the loss is conditioned on
        # is built by the understanding half on every step regardless of what is
        # trained.
        assert_full_finetune_dropout_free(trainer.transformer)
        if branch in ("und", "both"):
            assert_understanding_training_supported(trainer.transformer)

        trainer.transformer.requires_grad_(False)
        for _, _, _, module in targets:
            module.requires_grad_(True)
        fm_parameters = self._fm_parameters(branch)
        for parameter in fm_parameters:
            parameter.requires_grad_(True)
        trainer.transformer.train()

        trainable = sum(
            p.numel() for p in trainer.transformer.parameters() if p.requires_grad
        )
        fm_note = (
            f" plus {len(fm_parameters)} fm_modules tensor(s), "
            f"{sum(p.numel() for p in fm_parameters):,} element(s),"
            if fm_parameters else ""
        )
        print(
            f"[SenseNovaFullParameterAdapter] {branch} branch: {len(targets)} decoder "
            f"Linear(s),{fm_note} {trainable:,} trainable parameter element(s); "
            f"everything else in the transformer is frozen"
        )

    def setup_trainable_parameters(self) -> List[Dict[str, Any]]:
        """Generation group then understanding group, the order
        ``BaseTrainer._build_component_lr_list`` reports for this architecture
        (a resume remaps learning rates by index).

        Second gate, not a duplicate: a caller that builds the optimizer without
        going through ``prepare_models_for_training`` would otherwise get
        whatever ``requires_grad`` happened to be set, including nothing.
        """
        branch, targets = self._resolve_scope()
        trainer = self.trainer
        unet_lr = resolve_component_lr(
            trainer, "unet_lr", label="SenseNova generation branch"
        )
        # The chain SenseNova's LoRA adapter and SDXL's TE1 group both use: the
        # understanding half is this architecture's prompt encoder.
        und_lr = resolve_component_lr(
            trainer,
            "text_encoder_1_lr",
            "text_encoder_lr",
            "unet_lr",
            label="SenseNova understanding branch",
        )
        groups: List[Dict[str, Any]] = []
        for half, lr in (("gen", unet_lr), ("und", und_lr)):
            if branch not in (half, "both"):
                continue
            # Asking the enumerator which half a module belongs to, rather than
            # re-deriving it from the module path here.
            params = [
                parameter
                for _, _, _, module in iter_sensenova_lora_targets(
                    trainer.transformer, branch=half
                )
                for parameter in module.parameters()
                if parameter.requires_grad
            ]
            if half == "gen":
                params.extend(
                    parameter
                    for parameter in self._fm_parameters(branch)
                    if parameter.requires_grad
                )
            if params:
                groups.append({"params": params, "lr": lr})
        if not groups:
            raise RuntimeError(
                f"SenseNova full fine-tuning collected no trainable parameter from "
                f"the {branch} branch's {len(targets)} decoder Linear(s). "
                f"prepare_models_for_training must run first."
            )
        return groups

    def grad_norm_components(self) -> Dict[int, str]:
        """Both MoT halves live in ``transformer_original``; split them by half.

        Without this every trained parameter is reported as U-Net, because the
        full-FT grad-norm loop buckets by the module it walked -- so a ``und`` or
        ``both`` run showed one merged number and no separate understanding
        norm. The mapping is the one Phase 1 LoRA already registers
        (``SenseNovaLoRAAdapter`` puts the understanding half under
        ``LORA_COMPONENT_TEXT_ENCODER_1``), so the two methods report the same
        two components under the same two names, and
        ``_build_component_lr_list``'s ``MoT-Generation`` / ``MoT-Understanding``
        groups line up with them.

        Driven by the enumerator that built the optimizer groups, not by a name
        test on the parameter path (dd0b10c7).
        """
        branch, _ = self._resolve_scope()
        components: Dict[int, str] = {}
        for half, component in (
            ("gen", LORA_COMPONENT_UNET),
            ("und", LORA_COMPONENT_TEXT_ENCODER_1),
        ):
            if branch not in (half, "both"):
                continue
            for _, _, _, module in iter_sensenova_lora_targets(
                self.trainer.transformer, branch=half
            ):
                for parameter in module.parameters():
                    components[id(parameter)] = component
        # Generation-side, and stated rather than left to the module walk: on a
        # `both` run every override is explicit or the bucket is arbitrary.
        for parameter in self._fm_parameters(branch):
            components[id(parameter)] = LORA_COMPONENT_UNET
        return components

    def _resolve_save_format(self) -> str:
        """The requested on-disk format, refused rather than defaulted if unknown."""
        from api.param_defaults import (
            SENSENOVA_FULL_FINETUNE_SAVE_FORMATS, TRAINING_DEFAULTS,
        )

        trainer = self.trainer
        settings = getattr(trainer, "config", None) or {}
        value = getattr(trainer, "sensenova_full_finetune_save_format", None)
        if value is None:
            value = settings.get(
                "sensenova_full_finetune_save_format",
                TRAINING_DEFAULTS["sensenova_full_finetune_save_format"],
            )
        value = str(value).strip().lower()
        if value not in SENSENOVA_FULL_FINETUNE_SAVE_FORMATS:
            raise ValueError(
                f"Unknown sensenova_full_finetune_save_format {value!r}. "
                f"Supported: {', '.join(SENSENOVA_FULL_FINETUNE_SAVE_FORMATS)} "
                f"(docs/guides/SENSENOVA_TRAINING_DESIGN.md 6.4). Refusing rather "
                f"than falling back to the default: a save is the only artefact "
                f"this run produces."
            )
        return value

    def save_checkpoint(self, step: int, epoch: int, output_path: Path):
        """Write the trained MoT half in the selected format (6.4).

        Every format is meaningful on every branch, with one degeneracy: with
        both halves trained there is no int8 half for ``mixed`` to keep, so it
        writes the ``bf16`` file. That is announced rather than relabelled --
        the effective format is in the checkpoint's metadata and in a warning on
        the run.

        Only ``int8`` produces a file this repo can train on again:
        ``_assert_supported_quantized_training_base`` requires all 588 decoder
        Linears to be one quantized flavour, which a mixed or bf16 file is not.
        """
        import os

        from core.models.sensenova.loader import (
            save_sensenova_full_finetune_checkpoint,
        )
        from core.training.training_events import emit_training_warning

        trainer = self.trainer
        save_format = self._resolve_save_format()
        branch, targets = self._resolve_scope()
        model_path = getattr(trainer, "model_path", None)
        source_dir = os.path.dirname(str(model_path)) if model_path else None

        if save_format == "mixed" and branch == "both":
            emit_training_warning(
                "SenseNova full fine-tuning is training both MoT halves, so the "
                "'mixed' checkpoint format has no int8 half left to keep and the "
                "'bf16' file is written instead (both halves floating point). "
                "The checkpoint's metadata records the effective format.",
                code="sensenova_save_format_degenerate",
                prefix=getattr(trainer, "log_prefix", "[SenseNova]"),
            )

        extra_metadata = {"step": str(step), "epoch": str(epoch)}
        # Self-describing: a later bf16 resume can name its base without this
        # run's config. Identity is a cheap size check; the resume path is
        # what actually proves the content matches.
        configured_base = str(getattr(trainer, "configured_model_path", "") or "").strip()
        if configured_base:
            extra_metadata["sensenova_base_model_path"] = configured_base
            try:
                from core.models.sensenova.loader import sensenova_base_model_identity

                extra_metadata["sensenova_base_model_identity"] = (
                    sensenova_base_model_identity(configured_base)
                )
            except OSError:
                pass  # Base unreadable right now; the path alone is still useful.

        written, census = save_sensenova_full_finetune_checkpoint(
            trainer.transformer,
            str(output_path),
            branch=branch,
            save_format=save_format,
            config=getattr(trainer, "sensenova_model_config", None),
            raw_config=getattr(trainer, "sensenova_config_dict", None),
            source_dir=source_dir,
            extra_metadata=extra_metadata,
        )
        print(
            f"[SenseNovaFullParameterAdapter] step {step}: saved {len(targets)} "
            f"{branch} decoder Linear(s) as '{census['effective_format']}' -> {written}"
        )
