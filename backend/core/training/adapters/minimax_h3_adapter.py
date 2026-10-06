"""MiniMax-H3 LoRA adapter.

Model characteristics:
  - ``MiniMaxH3Transformer3DModel``: ONE stream of 50 blocks over a packed
    ``[text | audio | video]`` sequence (not a two-tower MM-DiT), 20.1 B dense.
  - Released weights are weight-only FP8 (``Fp8Linear``) with dequant inside the
    forward; the base stays quantized and resident during training.
  - Qwen3-VL-32B conditioner and both autoencoders stay frozen.

LoRA targets (design §10, all 50 blocks):
  ``transformer_blocks.{i}.attn.{to_q,to_k,to_v,to_out.0}``
  ``transformer_blocks.{i}.ff.{net.0.proj,net.2}``
  -> 300 modules, 83.1 M trainable parameters at rank 16 (MEASURED, Phase 0T).

Excluded from LoRA targets:
  * ``proj_in`` / ``audio_proj_in`` / ``proj_out`` / ``audio_proj_out``: the
    modality I/O heads. They are small and structurally load-bearing for the
    packed-sequence split; adapting them shifts the modality interface itself
    rather than the representation.
  * the 2-layer ``token_refiner``: its output conditions BOTH modality heads and
    there is no training-formulation documentation for it upstream.
  * AdaLN: in the released ("pruned") variant the modulation is a frozen lookup
    TABLE plus a projection, not a target a LoRA can meaningfully wrap.

The Full factory selects the experimental aligned adapter; quality and GPU
optimizer-resume acceptance remain pending after one API update and save.

LoRA save format: sd-scripts native — ``lora_unet_<flattened>.lora_down.weight`` /
``lora_up.weight`` / ``alpha``; metadata ``model_type="minimax_h3"``.
"""

from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn

from core.adapters import (
    # Bound here for adapter_layering_test's canonical-identity check, which
    # names this module explicitly; the injector below uses the H3 subclass.
    LoRALinearLayer,  # noqa: F401
    MiniMaxH3LoRALinearLayer,
    is_adapter_covered,
    is_lora_wrappable_linear,
)

from .base_adapter import (
    BaseLoRAAdapter, BaseFullParameterAdapter, reject_quantized_base,
    resolve_component_lr, LORA_COMPONENT_UNET
)


class MiniMaxH3AlignedFullParameterAdapter(BaseFullParameterAdapter):
    """Train an unquantized aligned DiT and save its native video bundle."""

    def _validate(self, *, allow_pending=False):
        from api.param_defaults import TRAINING_DEFAULTS
        if not getattr(self.trainer, "train_unet", TRAINING_DEFAULTS["train_unet"]):
            raise ValueError("H3 aligned Full training requires train_unet=true")
        for name in ("train_text_encoder", "train_image_encoder"):
            if getattr(self.trainer, name, TRAINING_DEFAULTS[name]):
                raise ValueError(f"H3 aligned Full training requires {name}=false")
        from core.models.components.video_latent_adapter import NormalizedVideoVAE
        from core.models.minimax_h3.video_alignment import AlignedMiniMaxH3BlockLoopWrapper
        model = self.trainer.transformer
        reject_quantized_base(model, model_label="MiniMax H3 Full")
        inner = getattr(model, "transformer", model)
        pair = getattr(inner, "_video_latent_alignment", None)
        vae = self.trainer.vae
        identity = getattr(self.trainer, "vae_identity", None)
        from core.models.minimax_h3.vae_training import PENDING
        pending = allow_pending and getattr(self.trainer, PENDING, None) is not None
        if (not isinstance(model, AlignedMiniMaxH3BlockLoopWrapper)
                or pair is None or (not pair.ready and not pending) or not isinstance(vae, NormalizedVideoVAE)
                or identity is None or identity.latent_hash != pair.declaration["target_identity"]
                or vae.contract != getattr(inner, "_video_codec_contract", None)):
            raise ValueError("H3 aligned Full training requires a calibrated matching video codec")
        return model

    def prepare_models_for_training(self):
        model = self._validate(allow_pending=True)
        enabled = bool(getattr(self.trainer, "train_unet", True))
        model.requires_grad_(enabled)
        model.train(enabled)
        for name in ("vae", "audio_vae", "text_encoder", "text_encoder_2"):
            module = getattr(self.trainer, name, None)
            if module is not None:
                module.requires_grad_(False).eval()

    def arch_param_groups(self):
        model = self._validate(allow_pending=True)
        parameters = [p for p in model.parameters() if p.requires_grad]
        if not parameters:
            return []
        return [{"params": parameters,
                 "lr": resolve_component_lr(self.trainer, "unet_lr", label="MiniMax H3 DiT"),
                 "name": "unet", "component": LORA_COMPONENT_UNET}]

    def write_checkpoint(self, step, epoch, output_path):
        self._validate()
        from core.models.minimax_h3.video_artifact import save_checkpoint
        return save_checkpoint(self.trainer, output_path, step=step, epoch=epoch)


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
            # is_adapter_covered, not isinstance(LoRALinearLayer): this arch's own
            # branch class IS a LoRALinearLayer subclass, but a slot the
            # generation loader already covers holds a CompositeAdapterLayer,
            # which must still be yielded rather than vanish from enumeration.
            if not is_lora_wrappable_linear(current) and not is_adapter_covered(current):
                continue
            yield f"transformer_blocks.{i}.{leaf}", parent, attr, current


class MiniMaxH3LoRAAdapter(BaseLoRAAdapter):
    """LoRA adapter for the MiniMax-H3 DiT."""

    #: This arch's forward runs without autocast, so its branch casts per call.
    LORA_LAYER_CLS = MiniMaxH3LoRALinearLayer

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

    def arch_param_groups(self, lora_layers: Dict[str, nn.Module]
                                    ) -> List[Dict[str, Any]]:
        return self.component_param_groups(lora_layers, {
            LORA_COMPONENT_UNET: lambda: resolve_component_lr(
                self.trainer, "unet_lr", label="MiniMax-H3 LoRA"),
        })

    def checkpoint_metadata(self, lora_layers: Dict[str, nn.Module],
                            step: int, epoch: int) -> Dict[str, str]:
        """sd-scripts native format."""
        from core.models.minimax_h3.residual_unet import model_topology_metadata
        return {
            **model_topology_metadata(self.trainer.transformer),
            "model_type": "minimax_h3",
            "modelspec.architecture": "minimax_h3",
            "lora_rank": str(self.lora_rank),
            "lora_alpha": str(self.lora_alpha),
            "lora_targets": ",".join(k for k, v in self.scope.items() if v),
            "step": str(step),
            "epoch": str(epoch),
            "format": "pt",
        }
