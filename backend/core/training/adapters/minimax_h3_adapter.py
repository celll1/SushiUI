"""MiniMax-H3 LoRA adapter.

Model characteristics:
  - ``MiniMaxH3Transformer3DModel``: ONE stream of 50 blocks over a packed
    ``[text | audio | video]`` sequence (not a two-tower MM-DiT), 33 B dense.
  - Released weights are weight-only FP8 (``Fp8Linear``) with dequant inside the
    forward; the base stays quantized and resident during training.
  - Qwen3-VL-32B conditioner and both autoencoders stay frozen.

LoRA targets (design §10, all 50 blocks):
  ``transformer_blocks.{i}.attn.{to_q,to_k,to_v,to_out.0}``
  ``transformer_blocks.{i}.ff.{net.0.proj,net.2}``
  -> 300 modules, 83.1 M trainable parameters at rank 16 (MEASURED, Phase 0T).

PERMANENTLY EXCLUDED, with reasons — these are design decisions, not deferrals:
  * ``proj_in`` / ``audio_proj_in`` / ``proj_out`` / ``audio_proj_out``: the
    modality I/O heads. They are small and structurally load-bearing for the
    packed-sequence split; adapting them shifts the modality interface itself
    rather than the representation.
  * the 2-layer ``token_refiner``: its output conditions BOTH modality heads and
    there is no training-formulation documentation for it upstream.
  * AdaLN: in the released ("pruned") variant the modulation is a frozen lookup
    TABLE plus a projection, not a target a LoRA can meaningfully wrap.

There is deliberately **no FullParameterAdapter class in this module**. Full
fine-tuning is refused for this architecture in three layers (design §7): this
absence, the hard ``ValueError`` in ``full_parameter_trainer._create_adapter``,
and the ``TRAINING_UNSUPPORTED`` declaration served by
``GET /schema/arch-capabilities``.

Save format: sd-scripts native — ``lora_unet_<flattened>.lora_down.weight`` /
``lora_up.weight`` / ``alpha``; metadata ``model_type="minimax_h3"``.
"""

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

from .base_adapter import BaseLoRAAdapter, is_lora_wrappable_linear, LORA_COMPONENT_UNET
from .sd15_adapter import LoRALinearLayer


# Default LoRA scope. BOTH groups default ON: that is the 300-module target set
# Phase 0T measured (gradients finite and nonzero in every block group, and both
# `ff` leaves carrying real gradient norm -- `ff.net.0.proj` was the second
# largest of the six target types).
DEFAULT_MINIMAX_H3_SCOPE = {
    "attention": True,   # attn.to_q / to_k / to_v / to_out.0
    "ff": True,          # SwiGLU ff.net.0.proj / ff.net.2
}

_ATTN_LEAVES = ("to_q", "to_k", "to_v", "to_out.0")
_FF_LEAVES = ("net.0.proj", "net.2")


def parse_scope_csv(scope_csv: str) -> Dict[str, bool]:
    """``"attention,ff"`` -> scope dict, built from an ALL-FALSE base.

    Built from all-false (not from the default) so unticking a group in the UI
    actually removes it, instead of being silently ignored.
    """
    wanted = {tok.strip() for tok in (scope_csv or "").split(",") if tok.strip()}
    return {"attention": "attention" in wanted, "ff": "ff" in wanted}


class MiniMaxH3LoRALinearLayer(LoRALinearLayer):
    """``LoRALinearLayer`` with the LoRA branch cast to the ACTIVATION dtype.

    MiniMax-H3's training forward runs WITHOUT ``torch.autocast``: the vendored
    transformer owns its own mixed-precision policy (fp32 I/O heads and AdaLN
    projections, bf16 block stack, each activation aligned to its projection's
    parameter dtype), and an autocast context would override those casts and make
    training a different function from generation.

    The stock layer relies on autocast to reconcile its fp32 master weights with
    a bf16 activation. Without autocast that is not a style difference, it is a
    ``RuntimeError`` on the first ``F.linear`` -- and, if the branch happened to
    be built in the activation dtype instead, a silent loss of the fp32 master.
    So the masters stay fp32 and are cast per call; the gradient flows back
    through the cast to the fp32 parameters unchanged. This is exactly the LoRA
    shape Phase 0T measured (bitwise save->reload, 600/600 tensors receiving
    finite gradients).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        org_out = self.original_module(x)
        down = F.linear(x, self.lora_down.weight.to(x.dtype))
        up = F.linear(down, self.lora_up.weight.to(x.dtype))
        return org_out + up * self.scale


def _flatten_to_sdscripts(module_path: str) -> str:
    """diffusers dotted module path -> sd-scripts underscore key fragment."""
    return module_path.replace(".", "_")


def _resolve_leaf(root: nn.Module, dotted: str) -> Optional[Tuple[nn.Module, Any, nn.Module]]:
    """``(parent, attr, module)`` for a dotted path under ``root``; None if absent.

    ``attr`` is an int for a ``ModuleList``/``Sequential`` index (``to_out.0``,
    ``net.2``) so the caller can assign back with ``parent[attr] = ...``.
    """
    parts = dotted.split(".")
    parent: nn.Module = root
    for name in parts[:-1]:
        if name.isdigit():
            try:
                parent = parent[int(name)]
            except (IndexError, TypeError):
                return None
        else:
            parent = getattr(parent, name, None)
            if parent is None:
                return None
    last = parts[-1]
    if last.isdigit():
        idx = int(last)
        try:
            if idx >= len(parent):
                return None
            return parent, idx, parent[idx]
        except TypeError:
            return None
    module = getattr(parent, last, None)
    if module is None:
        return None
    return parent, last, module


def iter_minimax_h3_lora_targets(
    transformer, scope: Dict[str, bool]
) -> Iterator[Tuple[str, nn.Module, Any, nn.Module]]:
    """Yield ``(module_path, parent, attr, current)`` for every target Linear.

    NOT ``isinstance(x, nn.Linear)``: the released MiniMax-H3 DiT ships
    weight-only FP8, so 300 of its Linears are ``Fp8Linear`` -- an ``nn.Module``
    that is NOT an ``nn.Linear`` subclass. The naive test skips every one of them
    SILENTLY and the run "succeeds" with a target count that looks like a
    narrower scope. That defect has been found on four architectures in this repo
    already; ``is_lora_wrappable_linear`` is the one shared predicate, and the
    same one the generation side declares
    (``pipeline_backends/minimax_h3._is_lora_target``).
    """
    blocks = getattr(transformer, "transformer_blocks", None)
    if blocks is None:
        return

    leaves: List[str] = []
    if scope.get("attention", True):
        leaves += [f"attn.{leaf}" for leaf in _ATTN_LEAVES]
    if scope.get("ff", True):
        leaves += [f"ff.{leaf}" for leaf in _FF_LEAVES]

    for i, block in enumerate(blocks):
        for leaf in leaves:
            resolved = _resolve_leaf(block, leaf)
            if resolved is None:
                continue
            parent, attr, current = resolved
            if not is_lora_wrappable_linear(current) and not isinstance(current, LoRALinearLayer):
                continue
            yield f"transformer_blocks.{i}.{leaf}", parent, attr, current


class MiniMaxH3LoRAAdapter(BaseLoRAAdapter):
    """LoRA adapter for the MiniMax-H3 DiT."""

    def __init__(self, trainer, lora_rank: int, lora_alpha: int,
                 lora_dtype: torch.dtype = torch.float32,
                 scope: Optional[Dict[str, bool]] = None):
        super().__init__(trainer, lora_rank, lora_alpha, lora_dtype)
        self.scope = dict(DEFAULT_MINIMAX_H3_SCOPE) if scope is None else dict(scope)

    def apply_lora_to_unet(self, lora_layers: Dict[str, nn.Module]) -> int:
        transformer = self.trainer.transformer
        if transformer is None:
            print("[MiniMaxH3LoRAAdapter] WARNING: trainer.transformer is None - skipping LoRA injection")
            return 0

        print(f"[MiniMaxH3LoRAAdapter] Injecting LoRA (scope={self.scope}, "
              f"rank={self.lora_rank}, alpha={self.lora_alpha})")

        count = 0
        for module_path, parent, attr, current in iter_minimax_h3_lora_targets(
                transformer, self.scope):
            if isinstance(current, LoRALinearLayer):
                continue  # idempotent / stacking-safe

            lora_name = f"lora_unet_{_flatten_to_sdscripts(module_path)}"
            lora_layer = MiniMaxH3LoRALinearLayer(
                current, self.lora_rank, self.lora_alpha, lora_name, self.lora_dtype,
            )
            if isinstance(attr, int):
                parent[attr] = lora_layer
            else:
                setattr(parent, attr, lora_layer)

            self.register_lora_layer(lora_layers, lora_name, lora_layer, LORA_COMPONENT_UNET)
            count += 1

        n_params = sum(p.numel() for layer in lora_layers.values() for p in layer.parameters()
                       if p.requires_grad)
        print(f"[MiniMaxH3LoRAAdapter] Injected {count} LoRA layer(s) into the MiniMax-H3 DiT "
              f"({n_params / 1e6:.1f} M trainable parameters)")
        return count

    def apply_lora_to_text_encoders(self, lora_layers: Dict[str, nn.Module]) -> int:
        """MiniMax-H3 keeps its Qwen3-VL conditioner frozen.

        Not a policy choice that could be relaxed by a flag: the encoder is read
        through ``torch.func.functional_call`` off a memory-mapped 48 GiB file
        one decoder layer at a time, precisely so it never becomes resident.
        There is no configuration in which its weights and the DiT's are both on
        the GPU.
        """
        print("[MiniMaxH3LoRAAdapter] Qwen3-VL conditioner is frozen - no LoRA on the text encoder")
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
            "model_type": "minimax_h3",
            "modelspec.architecture": "minimax_h3",
            "lora_rank": str(self.lora_rank),
            "lora_alpha": str(self.lora_alpha),
            "lora_targets": active_scopes,
            "step": str(step),
            "epoch": str(epoch),
            "format": "pt",
        }

        save_file(state_dict, str(output_path), metadata=metadata)
        print(f"[MiniMaxH3LoRAAdapter] Saved LoRA checkpoint ({len(lora_layers)} layers) "
              f"-> {output_path}")
