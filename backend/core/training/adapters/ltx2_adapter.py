"""LTX-2.3 (joint audio+video MM-DiT) training adapters.

Model characteristics:
  - LTX2VideoTransformer3DModel (joint audio+video MM-DiT, 19B)
  - Gemma-3 (12B) text encoder + LTX2TextConnectors (frozen)
  - AutoencoderKLLTX2Video (128ch, spatial /32, temporal /8)
  - Rectified Flow / Flow Matching (predicts velocity v = noise - x_0)

LoRA targets (video LoRA default):
  transformer_blocks.{i}.attn1/attn2.{to_q,to_k,to_v,to_out.0}

EXCLUDED by default (audio / cross-modality / feed-forward):
  audio_attn1/2, audio_ff, audio_to_video_attn, video_to_audio_attn, ff,
  scale-shift tables, to_gate_logits. Scope
  {attention: True, ff: False, audio: False, av_cross: False}.

Save format: sd-scripts native — ``lora_unet_<flattened>.lora_down.weight`` /
``lora_up.weight`` / ``alpha``. metadata model_type="ltx2".
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator, Tuple

import torch
import torch.nn as nn
from safetensors.torch import save_file

from core.adapters import (
    LoRALinearLayer, is_adapter_covered, is_lora_wrappable_linear,
    named_modules_outside_adapters,
)
from .base_adapter import (
    BaseLoRAAdapter, BaseFullParameterAdapter,
    reject_quantized_base, resolve_component_lr, LORA_COMPONENT_UNET,
)


# Default LoRA scope for LTX-2.3 video training.
DEFAULT_LTX2_SCOPE = {
    "attention": True,   # attn1/attn2 self+cross video attention
    "ff": False,         # video feed-forward (opt-in)
    "audio": False,      # audio_attn1/2, audio_ff
    "av_cross": False,   # audio_to_video_attn, video_to_audio_attn
}

# Video attention leaf Linears within each LTX2Attention module.
_ATTN_LEAVES = ("to_q", "to_k", "to_v", "to_out.0")


def _flatten_to_sdscripts(module_path: str) -> str:
    """diffusers dotted module path -> sd-scripts underscore key fragment."""
    return module_path.replace(".", "_")


def iter_ltx2_lora_targets(
    transformer, scope: Dict[str, bool]
) -> Iterator[Tuple[str, nn.Module, Any, nn.Module]]:
    """Yield ``(module_path, parent, attr, current)`` for every target Linear.

    Walks ``transformer.transformer_blocks``. By default only the video
    self/cross attention (``attn1`` / ``attn2``) Linears are targeted; audio /
    cross-modality / feed-forward are gated by ``scope``.

    NOT ``isinstance(x, nn.Linear)``: over a weight-only quantized LTX-2.3 base
    (an offline int8 artifact, or a transformer converted in place by
    ``unet_quantization="int8"``) the very layers a LoRA targets are
    ``Int8Linear`` / ``Fp8Linear``, which are ``nn.Module`` but NOT ``nn.Linear``
    subclasses. The naive test skips every one of them SILENTLY -- the run
    "succeeds" with a smaller injected count that looks like a narrower scope.
    Measured at 75% of targets dropped on Anima and at 8 sites on FLUX.2 before
    the same fix. ``is_lora_wrappable_linear`` accepts all three; the
    ``is_adapter_covered`` arm stays separate because an already-covered slot
    (either wrapper class) must be yielded -- for idempotent re-application and
    for the generation-side restore -- but must not be wrapped twice.

    The adapter dtype is NOT taken from the base: ``LoRALinearLayer`` builds its
    two branches at ``self.lora_dtype`` (fp32 by default), so a quantized base
    cannot pull the adapter down to int8 or e4m3.
    """
    blocks = getattr(transformer, "transformer_blocks", None)
    if blocks is None:
        return

    attn_module_names: List[str] = []
    if scope.get("attention", True):
        attn_module_names += ["attn1", "attn2"]
    if scope.get("audio", False):
        attn_module_names += ["audio_attn1", "audio_attn2"]
    if scope.get("av_cross", False):
        attn_module_names += ["audio_to_video_attn", "video_to_audio_attn"]

    for i, block in enumerate(blocks):
        for attn_name in attn_module_names:
            attn = getattr(block, attn_name, None)
            if attn is None:
                continue
            for leaf in _ATTN_LEAVES:
                if "." in leaf:
                    # e.g. "to_out.0" — ModuleList index.
                    holder_name, idx_str = leaf.split(".")
                    holder = getattr(attn, holder_name, None)
                    if holder is None:
                        continue
                    idx = int(idx_str)
                    if idx >= len(holder):
                        continue
                    current = holder[idx]
                    if not is_lora_wrappable_linear(current) and not is_adapter_covered(current):
                        continue
                    path = f"transformer_blocks.{i}.{attn_name}.{holder_name}.{idx}"
                    yield path, holder, idx, current
                else:
                    current = getattr(attn, leaf, None)
                    if current is None:
                        continue
                    if not is_lora_wrappable_linear(current) and not is_adapter_covered(current):
                        continue
                    path = f"transformer_blocks.{i}.{attn_name}.{leaf}"
                    yield path, attn, leaf, current

        # Feed-forward (opt-in).
        if scope.get("ff", False):
            ff = getattr(block, "ff", None)
            if ff is not None:
                # An ALREADY-COVERED slot must be yielded (so re-application,
                # LoRA stacking and the generation-side restore can find it) but
                # never descended into: the wrapper's own lora_down/lora_up are
                # nn.Linear children, and treating them as targets would wrap the
                # adapter's own branches. The walker enforces the non-descent for
                # BOTH wrapper classes.
                for sub_name, sub in named_modules_outside_adapters(ff):
                    if not is_adapter_covered(sub) and not is_lora_wrappable_linear(sub):
                        continue
                    # Resolve the parent + attr for in-place replacement.
                    parent = ff
                    attr: Any = sub_name
                    if "." in sub_name:
                        *parents, last = sub_name.split(".")
                        p = ff
                        for pp in parents:
                            p = getattr(p, pp) if not pp.isdigit() else p[int(pp)]
                        parent = p
                        attr = int(last) if last.isdigit() else last
                    path = f"transformer_blocks.{i}.ff.{sub_name}"
                    yield path, parent, attr, sub


class Ltx2LoRAAdapter(BaseLoRAAdapter):
    """LoRA adapter for LTX-2.3 DiT models."""

    def __init__(self, trainer, lora_rank: int, lora_alpha: int,
                 lora_dtype: torch.dtype = torch.float32,
                 scope: Optional[Dict[str, bool]] = None):
        super().__init__(trainer, lora_rank, lora_alpha, lora_dtype)
        self.scope = dict(DEFAULT_LTX2_SCOPE) if scope is None else dict(scope)

    def apply_lora_to_unet(self, lora_layers: Dict[str, nn.Module]) -> int:
        """In-place wrap the target Linear modules of the LTX-2.3 DiT."""
        transformer = self.trainer.transformer
        if transformer is None:
            print("[Ltx2LoRAAdapter] WARNING: trainer.transformer is None - skipping LoRA injection")
            return 0

        print(f"[Ltx2LoRAAdapter] Injecting LoRA (scope={self.scope}, "
              f"rank={self.lora_rank}, alpha={self.lora_alpha})")

        count = 0
        for module_path, parent, attr, current in iter_ltx2_lora_targets(transformer, self.scope):
            if is_adapter_covered(current):
                continue  # idempotent / stacking-safe

            lora_name = f"lora_unet_{_flatten_to_sdscripts(module_path)}"
            lora_layer = self.build_branch(current, lora_name)
            if isinstance(attr, int):
                parent[attr] = lora_layer
            else:
                setattr(parent, attr, lora_layer)

            self.register_lora_layer(lora_layers, lora_name, lora_layer, LORA_COMPONENT_UNET)
            count += 1

        print(f"[Ltx2LoRAAdapter] Injected {count} LoRA layer(s) into LTX-2.3 DiT")
        return count

    def apply_lora_to_text_encoders(self, lora_layers: Dict[str, nn.Module]) -> int:
        """LTX-2.3 keeps the Gemma-3 text encoder + connectors frozen."""
        print("[Ltx2LoRAAdapter] Gemma-3 text encoder + connectors are frozen - no LoRA on TE")
        return 0

    def setup_trainable_parameters(self, lora_layers: Dict[str, nn.Module]
                                    ) -> List[Dict[str, Any]]:
        return self.component_param_groups(lora_layers, {
            LORA_COMPONENT_UNET: lambda: resolve_component_lr(
                self.trainer, "unet_lr", label="LTX-2.3 LoRA"),
        })

    def checkpoint_metadata(self, lora_layers: Dict[str, nn.Module],
                            step: int, epoch: int) -> Dict[str, str]:
        """sd-scripts native format."""
        return {
            "model_type": "ltx2",
            "modelspec.architecture": "ltx2",
            "lora_rank": str(self.lora_rank),
            "lora_alpha": str(self.lora_alpha),
            "lora_targets": ",".join(k for k, v in self.scope.items() if v),
            "step": str(step),
            "epoch": str(epoch),
            "format": "pt",
        }


class Ltx2FullParameterAdapter(BaseFullParameterAdapter):
    """Full-parameter training adapter for LTX-2.3 DiT models.

    Trainable surface: the DiT (``transformer.*``) when train_unet=True.
    Gemma-3 text encoder + connectors + both VAEs stay frozen.
    """

    def prepare_models_for_training(self):
        trainer = self.trainer
        reject_quantized_base(trainer.transformer, model_label="LTX-2.3")
        train_dit = bool(getattr(trainer, "train_unet", True))

        if train_dit and trainer.transformer is not None:
            trainer.transformer.requires_grad_(True)
            trainer.transformer.train()
            print("[Ltx2FullParameterAdapter] LTX-2.3 DiT set to train mode")

        if trainer.text_encoder is not None:
            trainer.text_encoder.requires_grad_(False)
            trainer.text_encoder.eval()
        if getattr(trainer, "connectors", None) is not None:
            trainer.connectors.requires_grad_(False)
            trainer.connectors.eval()
        if trainer.vae is not None:
            trainer.vae.requires_grad_(False)
            trainer.vae.eval()
        if getattr(trainer, "audio_vae", None) is not None:
            trainer.audio_vae.requires_grad_(False)
            trainer.audio_vae.eval()

        print(f"[Ltx2FullParameterAdapter] Models prepared for training "
              f"(DiT trainable={train_dit})")

    def setup_trainable_parameters(self) -> List[Dict[str, Any]]:
        trainer = self.trainer
        if trainer.transformer is None:
            return []
        # Second gate, not a duplicate: a caller that builds the optimizer without
        # going through prepare_models_for_training() would otherwise still get
        # the silently-truncated parameter list this guard exists to prevent.
        reject_quantized_base(trainer.transformer, model_label="LTX-2.3")
        base_lr = resolve_component_lr(trainer, "unet_lr", label="LTX-2.3 transformer")
        params = [p for p in trainer.transformer.parameters() if p.requires_grad]
        if not params:
            return []
        total = sum(p.numel() for p in params)
        print(f"[Ltx2FullParameterAdapter] 1 param group, {total:,} trainable params")
        return [{"params": params, "lr": base_lr}]

    def save_checkpoint(self, step: int, epoch: int, output_path: Path):
        trainer = self.trainer
        if trainer.transformer is None:
            print("[Ltx2FullParameterAdapter] WARNING: no transformer to save")
            return

        if output_path.is_dir():
            output_path = output_path / f"ltx2_step_{step}.safetensors"
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
            "model_type": "ltx2",
            "modelspec.architecture": "ltx2",
            "format": "pt",
        }
        print(f"[Ltx2FullParameterAdapter] Saving to {output_path}...")
        save_file(combined, str(output_path), metadata=metadata)
        print(f"[Ltx2FullParameterAdapter] Saved {len(combined)} tensors -> {output_path}")
