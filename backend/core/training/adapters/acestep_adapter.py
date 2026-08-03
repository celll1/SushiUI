"""ACE-Step 1.5 (turbo audio DiT) training adapters.

Model characteristics:
  - AceStepConditionGenerationModel (2B DiT: ``.decoder`` = diffusion
    transformer, ``.encoder``/``.tokenizer``/``.detokenizer`` = frozen
    text/lyric/timbre condition encoder + FSQ audio-token roundtrip)
  - Qwen3-Embedding-0.6B text encoder (frozen)
  - AutoencoderOobleck VAE (64ch, temporal-only, 48kHz stereo)
  - Rectified Flow / Flow Matching (predicts velocity v = noise - x_0)

LoRA targets (audio LoRA default): every DiT layer has BOTH self-attention and
cross-attention (``AceStepDiTLayer.__init__`` always builds ``cross_attn``
since ``num_hidden_layers`` layers are all constructed with the default
``use_cross_attention=True`` — see ``AceStepDiTModel.__init__``):

  transformer.decoder.layers.{i}.self_attn.{q_proj,k_proj,v_proj,o_proj}
  transformer.decoder.layers.{i}.cross_attn.{q_proj,k_proj,v_proj,o_proj}

EXCLUDED by default (feed-forward / condition-encoder / audio-tokenizer):
  mlp (feed-forward, opt-in), encoder.* (text/lyric/timbre condition encoder,
  frozen — no LoRA), tokenizer/detokenizer (FSQ audio-token roundtrip, frozen).
  Scope {attention: True, mlp: False}.

Save format: sd-scripts native — ``lora_unet_<flattened>.lora_down.weight`` /
``lora_up.weight`` / ``alpha``. metadata model_type="acestep".
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator, Tuple

import torch
import torch.nn as nn
from safetensors.torch import save_file

from .base_adapter import BaseLoRAAdapter, BaseFullParameterAdapter, reject_quantized_base
from .sd15_adapter import LoRALinearLayer


# Default LoRA scope for ACE-Step audio training.
DEFAULT_ACESTEP_SCOPE = {
    "attention": True,   # self_attn + cross_attn (every DiT decoder layer)
    "mlp": False,        # feed-forward (opt-in)
}

# Attention leaf Linears within each AceStepAttention module.
_ATTN_LEAVES = ("q_proj", "k_proj", "v_proj", "o_proj")


def _flatten_to_sdscripts(module_path: str) -> str:
    """diffusers-style dotted module path -> sd-scripts underscore key fragment."""
    return module_path.replace(".", "_")


def iter_acestep_lora_targets(
    transformer, scope: Dict[str, bool]
) -> Iterator[Tuple[str, nn.Module, str, nn.Module]]:
    """Yield ``(module_path, parent, attr, current)`` for every target Linear.

    Walks ``transformer.decoder.layers`` (the DiT block list). By default only
    self_attn/cross_attn Linears are targeted; feed-forward (``mlp``) is gated
    by ``scope``.
    """
    decoder = getattr(transformer, "decoder", None)
    layers = getattr(decoder, "layers", None) if decoder is not None else None
    if layers is None:
        return

    attn_module_names: List[str] = []
    if scope.get("attention", True):
        attn_module_names += ["self_attn", "cross_attn"]

    for i, layer in enumerate(layers):
        for attn_name in attn_module_names:
            attn = getattr(layer, attn_name, None)
            if attn is None:
                continue
            for leaf in _ATTN_LEAVES:
                current = getattr(attn, leaf, None)
                if current is None:
                    continue
                if not isinstance(current, nn.Linear) and not isinstance(current, LoRALinearLayer):
                    continue
                path = f"decoder.layers.{i}.{attn_name}.{leaf}"
                yield path, attn, leaf, current

        # Feed-forward (opt-in). AceStepDiTLayer.mlp is a Qwen3MLP
        # (gate_proj/up_proj/down_proj Linears).
        if scope.get("mlp", False):
            mlp = getattr(layer, "mlp", None)
            if mlp is not None:
                for leaf in ("gate_proj", "up_proj", "down_proj"):
                    current = getattr(mlp, leaf, None)
                    if current is None:
                        continue
                    if not isinstance(current, nn.Linear) and not isinstance(current, LoRALinearLayer):
                        continue
                    path = f"decoder.layers.{i}.mlp.{leaf}"
                    yield path, mlp, leaf, current


class AceStepLoRAAdapter(BaseLoRAAdapter):
    """LoRA adapter for ACE-Step 1.5 (turbo) DiT models."""

    def __init__(self, trainer, lora_rank: int, lora_alpha: int,
                 lora_dtype: torch.dtype = torch.float32,
                 scope: Optional[Dict[str, bool]] = None):
        super().__init__(trainer, lora_rank, lora_alpha, lora_dtype)
        self.scope = dict(DEFAULT_ACESTEP_SCOPE) if scope is None else dict(scope)

    def apply_lora_to_unet(self, lora_layers: Dict[str, nn.Module]) -> int:
        """In-place wrap the target Linear modules of the ACE-Step DiT."""
        transformer = self.trainer.transformer
        if transformer is None:
            print("[AceStepLoRAAdapter] WARNING: trainer.transformer is None - skipping LoRA injection")
            return 0

        print(f"[AceStepLoRAAdapter] Injecting LoRA (scope={self.scope}, "
              f"rank={self.lora_rank}, alpha={self.lora_alpha})")

        count = 0
        for module_path, parent, attr, current in iter_acestep_lora_targets(transformer, self.scope):
            if isinstance(current, LoRALinearLayer):
                continue  # idempotent / stacking-safe

            lora_name = f"lora_unet_{_flatten_to_sdscripts(module_path)}"
            lora_layer = LoRALinearLayer(
                current, self.lora_rank, self.lora_alpha, lora_name, self.lora_dtype,
            )
            setattr(parent, attr, lora_layer)

            lora_layers[lora_name] = lora_layer
            count += 1

        print(f"[AceStepLoRAAdapter] Injected {count} LoRA layer(s) into ACE-Step DiT")
        return count

    def apply_lora_to_text_encoders(self, lora_layers: Dict[str, nn.Module]) -> int:
        """ACE-Step keeps the Qwen3-Embedding-0.6B text encoder frozen."""
        print("[AceStepLoRAAdapter] Qwen3 text encoder is frozen - no LoRA on TE")
        return 0

    def setup_trainable_parameters(self, lora_layers: Dict[str, nn.Module]
                                    ) -> List[Dict[str, Any]]:
        params: List[nn.Parameter] = []
        for lora_layer in lora_layers.values():
            params.extend(lora_layer.lora_down.parameters())
            params.extend(lora_layer.lora_up.parameters())
        if not params:
            return []
        return [{"params": params, "lr": getattr(self.trainer, "unet_lr", 1e-4)}]

    def save_checkpoint(self, lora_layers: Dict[str, nn.Module],
                         step: int, epoch: int, output_path: Path):
        """Save LoRA weights in sd-scripts native format."""
        state_dict: Dict[str, torch.Tensor] = {}
        alpha_value = float(self.lora_alpha)

        for lora_name, lora_layer in lora_layers.items():
            state_dict[f"{lora_name}.lora_down.weight"] = lora_layer.lora_down.weight.detach().cpu()
            state_dict[f"{lora_name}.lora_up.weight"] = lora_layer.lora_up.weight.detach().cpu()
            state_dict[f"{lora_name}.alpha"] = torch.tensor(alpha_value, dtype=torch.float32)

        active_scopes = ",".join(k for k, v in self.scope.items() if v)
        metadata = {
            "model_type": "acestep",
            "modelspec.architecture": "acestep",
            "lora_rank": str(self.lora_rank),
            "lora_alpha": str(self.lora_alpha),
            "lora_targets": active_scopes,
            "step": str(step),
            "epoch": str(epoch),
            "format": "pt",
        }

        save_file(state_dict, str(output_path), metadata=metadata)
        print(f"[AceStepLoRAAdapter] Saved LoRA checkpoint ({len(lora_layers)} layers) -> {output_path}")


class AceStepFullParameterAdapter(BaseFullParameterAdapter):
    """Full-parameter training adapter for ACE-Step 1.5 (turbo) DiT models.

    Trainable surface: the whole DiT (``transformer.*``) when train_unet=True
    — including the condition encoder/tokenizer/detokenizer submodules (unlike
    LoRA, which only targets ``decoder.layers.*.{self_attn,cross_attn}``).
    Qwen3 text encoder + VAE stay frozen.
    """

    def prepare_models_for_training(self):
        trainer = self.trainer
        reject_quantized_base(trainer.transformer, model_label="ACE-Step")
        train_dit = bool(getattr(trainer, "train_unet", True))

        if train_dit and trainer.transformer is not None:
            trainer.transformer.requires_grad_(True)
            trainer.transformer.train()
            print("[AceStepFullParameterAdapter] ACE-Step DiT set to train mode")

        if trainer.text_encoder is not None:
            trainer.text_encoder.requires_grad_(False)
            trainer.text_encoder.eval()
        if trainer.vae is not None:
            trainer.vae.requires_grad_(False)
            trainer.vae.eval()

        print(f"[AceStepFullParameterAdapter] Models prepared for training "
              f"(DiT trainable={train_dit})")

    def setup_trainable_parameters(self) -> List[Dict[str, Any]]:
        trainer = self.trainer
        if trainer.transformer is None:
            return []
        # Second gate, not a duplicate: a caller that builds the optimizer without
        # going through prepare_models_for_training() would otherwise still get
        # the silently-truncated parameter list this guard exists to prevent.
        reject_quantized_base(trainer.transformer, model_label="ACE-Step")
        base_lr = getattr(trainer, "unet_lr", None) or getattr(trainer, "learning_rate", 1e-5)
        params = [p for p in trainer.transformer.parameters() if p.requires_grad]
        if not params:
            return []
        total = sum(p.numel() for p in params)
        print(f"[AceStepFullParameterAdapter] 1 param group, {total:,} trainable params")
        return [{"params": params, "lr": base_lr}]

    def save_checkpoint(self, step: int, epoch: int, output_path: Path):
        trainer = self.trainer
        if trainer.transformer is None:
            print("[AceStepFullParameterAdapter] WARNING: no transformer to save")
            return

        if output_path.is_dir():
            output_path = output_path / f"acestep_step_{step}.safetensors"
        elif not str(output_path).endswith(".safetensors"):
            output_path = Path(str(output_path) + ".safetensors")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        dit_state = trainer.transformer.state_dict()
        combined: Dict[str, torch.Tensor] = {}
        for k, v in dit_state.items():
            combined[f"net.{k}"] = v.detach().to("cpu").contiguous()

        metadata = {
            "step": str(step),
            "epoch": str(epoch),
            "model_type": "acestep",
            "modelspec.architecture": "acestep",
            "format": "pt",
        }
        print(f"[AceStepFullParameterAdapter] Saving to {output_path}...")
        save_file(combined, str(output_path), metadata=metadata)
        print(f"[AceStepFullParameterAdapter] Saved {len(combined)} tensors -> {output_path}")
