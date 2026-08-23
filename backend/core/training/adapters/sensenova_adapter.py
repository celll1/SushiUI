"""SenseNova generation-branch LoRA training adapter."""

from pathlib import Path
from typing import Any, Dict, List

import torch
from safetensors.torch import save_file
from torch import nn

from core.models.sensenova.sensenova_lora import iter_sensenova_lora_targets

from .base_adapter import BaseLoRAAdapter, is_lora_wrappable_linear
from .sd15_adapter import LoRALinearLayer


class SenseNovaLoRAAdapter(BaseLoRAAdapter):
    """Wrap only the 294 generation-branch Linear modules."""

    @staticmethod
    def _expected_target_paths() -> set[str]:
        paths = set()
        for layer_index in range(42):
            prefix = f"language_model.model.layers.{layer_index}"
            paths.update(
                f"{prefix}.self_attn.{name}"
                for name in (
                    "q_proj_mot_gen",
                    "k_proj_mot_gen",
                    "v_proj_mot_gen",
                    "o_proj_mot_gen",
                )
            )
            paths.update(
                f"{prefix}.mlp_mot_gen.{name}"
                for name in ("gate_proj", "up_proj", "down_proj")
            )
        return paths

    def apply_lora_to_unet(self, lora_layers: Dict[str, nn.Module]) -> int:
        transformer = getattr(self.trainer, "transformer", None)
        if transformer is None:
            raise RuntimeError("SenseNova LoRA requires a loaded transformer")

        targets = list(iter_sensenova_lora_targets(transformer))
        actual_paths = {module_path for module_path, *_ in targets}
        expected_paths = self._expected_target_paths()
        if len(targets) != 294 or actual_paths != expected_paths:
            missing = sorted(expected_paths - actual_paths)
            extra = sorted(actual_paths - expected_paths)
            raise RuntimeError(
                "SenseNova generation LoRA requires exactly 294 targets "
                f"(missing={missing[:3]}, extra={extra[:3]})"
            )

        unwrapped = [target for target in targets if is_lora_wrappable_linear(target[3])]
        wrapped = [target for target in targets if isinstance(target[3], LoRALinearLayer)]
        if len(wrapped) == 294:
            mismatched_names = [
                path for path, _, _, layer in wrapped if layer.lora_name != path
            ]
            if mismatched_names:
                raise RuntimeError(
                    "SenseNova generation LoRA wrappers use the wrong namespace: "
                    f"{mismatched_names[:3]}"
                )
            for path, _, _, layer in wrapped:
                existing = lora_layers.get(path)
                if existing is not None and existing is not layer:
                    raise RuntimeError(
                        f"SenseNova LoRA registry conflicts with wrapper {path}"
                    )
                lora_layers[path] = layer
            return 0
        if len(unwrapped) != 294:
            raise RuntimeError(
                "SenseNova generation LoRA target state is mixed or unsupported "
                f"(unwrapped={len(unwrapped)}, wrapped={len(wrapped)}, total=294)"
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
            lora_layers[module_path] = wrapper
            count += 1

        print(f"[SenseNovaLoRAAdapter] Injected {count} generation LoRA layer(s)")
        return count

    def apply_lora_to_text_encoders(self, lora_layers: Dict[str, nn.Module]) -> int:
        del lora_layers
        print("[SenseNovaLoRAAdapter] Understanding branch is frozen - no text LoRA")
        return 0

    def setup_trainable_parameters(
        self, lora_layers: Dict[str, nn.Module]
    ) -> List[Dict[str, Any]]:
        parameters: List[nn.Parameter] = []
        for layer in lora_layers.values():
            parameters.extend(layer.lora_down.parameters())
            parameters.extend(layer.lora_up.parameters())
        if not parameters:
            return []
        learning_rate = getattr(self.trainer, "unet_lr", None) or 1e-4
        return [{"params": parameters, "lr": learning_rate}]

    def save_checkpoint(
        self,
        lora_layers: Dict[str, nn.Module],
        step: int,
        epoch: int,
        output_path: Path,
    ) -> None:
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
            "lora_targets": "generation",
            "step": str(step),
            "epoch": str(epoch),
        }
        save_file(state_dict, str(output_path), metadata=metadata)
        print(
            f"[SenseNovaLoRAAdapter] Saved LoRA checkpoint "
            f"({len(lora_layers)} layers) -> {output_path}"
        )
