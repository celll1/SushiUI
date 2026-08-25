"""SenseNova MoT training adapters (LoRA, and full parameter over one MoT half)."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from safetensors.torch import save_file
from torch import nn

from core.models.sensenova.sensenova_lora import (
    LORA_TARGET_LABELS,
    iter_sensenova_lora_targets,
)

from .base_adapter import (
    BaseFullParameterAdapter,
    BaseLoRAAdapter,
    is_lora_wrappable_linear,
    LORA_COMPONENT_TEXT_ENCODER_1,
    LORA_COMPONENT_UNET,
)
from .sd15_adapter import LoRALinearLayer

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
            wrapper = LoRALinearLayer(
                current,
                self.lora_rank,
                self.lora_alpha,
                module_path,
                self.lora_dtype,
            )
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

    def _split_by_component(
        self, lora_layers: Dict[str, nn.Module]
    ) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        generation: List[nn.Parameter] = []
        understanding: List[nn.Parameter] = []
        for name, layer in lora_layers.items():
            bucket = (
                understanding
                if self.lora_components.get(name) == LORA_COMPONENT_TEXT_ENCODER_1
                else generation
            )
            bucket.extend(layer.lora_down.parameters())
            bucket.extend(layer.lora_up.parameters())
        return generation, understanding

    def setup_trainable_parameters(
        self, lora_layers: Dict[str, nn.Module]
    ) -> List[Dict[str, Any]]:
        generation, understanding = self._split_by_component(lora_layers)
        unet_lr = getattr(self.trainer, "unet_lr", None) or 1e-4
        groups: List[Dict[str, Any]] = []
        if generation:
            groups.append({"params": generation, "lr": unet_lr})
        if understanding:
            # Same fallback chain SDXL's LoRA adapter uses for TE1.
            und_lr = (
                getattr(self.trainer, "text_encoder_1_lr", None)
                or getattr(self.trainer, "text_encoder_lr", None)
                or unet_lr
            )
            groups.append({"params": understanding, "lr": und_lr})
        return groups

    def save_checkpoint(
        self,
        lora_layers: Dict[str, nn.Module],
        step: int,
        epoch: int,
        output_path: Path,
    ) -> None:
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

        state_dict: Dict[str, torch.Tensor] = {}
        for module_path, layer in lora_layers.items():
            state_dict[f"{module_path}.lora_down.weight"] = (
                layer.lora_down.weight.detach().cpu()
            )
            state_dict[f"{module_path}.lora_up.weight"] = (
                layer.lora_up.weight.detach().cpu()
            )
            state_dict[f"{module_path}.alpha"] = torch.tensor(
                float(self.lora_alpha), dtype=torch.float32
            )

        metadata = {
            "model_type": "sensenova",
            "modelspec.architecture": "sensenova",
            "tensor_kind": "neo_hf_lora",
            "lora_rank": str(self.lora_rank),
            "lora_alpha": str(self.lora_alpha),
            "lora_targets": LORA_TARGET_LABELS[branch],
            "step": str(step),
            "epoch": str(epoch),
        }
        save_file(state_dict, str(output_path), metadata=metadata)
        print(
            f"[SenseNovaLoRAAdapter] Saved LoRA checkpoint "
            f"({len(lora_layers)} layers, {metadata['lora_targets']}) -> {output_path}"
        )


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
    the embedders, the ``*_norm_mot_gen`` norms) are deliberately not trained:
    they are not quantized, so the loader does not materialize them, and
    including them would break that identity. SENSENOVA_TRAINING_DESIGN.md 6.2
    leaves the question open.
    """

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
            assert_understanding_training_supported,
        )

        trainer = self.trainer
        # Config channel only; setup_optimizer re-checks with the real name.
        assert_full_finetune_contract(trainer)
        branch, targets = self._resolve_scope()
        if branch in ("und", "both"):
            assert_understanding_training_supported(trainer.transformer)

        trainer.transformer.requires_grad_(False)
        for _, _, _, module in targets:
            module.requires_grad_(True)
        trainer.transformer.train()

        trainable = sum(
            p.numel() for p in trainer.transformer.parameters() if p.requires_grad
        )
        print(
            f"[SenseNovaFullParameterAdapter] {branch} branch: {len(targets)} decoder "
            f"Linear(s), {trainable:,} trainable parameter element(s); everything "
            f"else in the transformer is frozen"
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
        unet_lr = getattr(trainer, "unet_lr", None) or 1e-6
        # The chain SenseNova's LoRA adapter and SDXL's TE1 group both use: the
        # understanding half is this architecture's prompt encoder.
        und_lr = (
            getattr(trainer, "text_encoder_1_lr", None)
            or getattr(trainer, "text_encoder_lr", None)
            or unet_lr
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
            if params:
                groups.append({"params": params, "lr": lr})
        if not groups:
            raise RuntimeError(
                f"SenseNova full fine-tuning collected no trainable parameter from "
                f"the {branch} branch's {len(targets)} decoder Linear(s). "
                f"prepare_models_for_training must run first."
            )
        return groups

    def save_checkpoint(self, step: int, epoch: int, output_path: Path):
        """Refused: writing any of the three candidate formats would pick one."""
        raise NotImplementedError(
            "SenseNova full fine-tuning cannot save a checkpoint yet: the output "
            "format is undecided (mixed int8+bf16, both halves bf16, or the "
            "trained half requantized to int8 -- see "
            "docs/guides/SENSENOVA_TRAINING_DESIGN.md 6.4). This is why full "
            "fine-tuning is still refused for this architecture before a run "
            "starts. Use training_method='lora'."
        )
