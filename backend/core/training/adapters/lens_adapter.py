"""Lens (Microsoft/Lens DiT) training adapters.

Model characteristics:
  - MMDiT double-stream architecture (48 blocks, RMSNorm, GateMLP)
  - GPT-OSS MoE text encoder (LensGptOssEncoder, 24-layer, multi-layer features)
  - AutoencoderKLFlux2 VAE (32ch, flat-sequence latent format)
  - Flow Matching (FlowMatchEulerDiscreteScheduler, velocity target v = noise - x0)

LoRA targets (controlled by the `scope` dict):
  - img_attn:  transformer_blocks.{N}.attn.{img_qkv, to_out[0]}
  - txt_attn:  transformer_blocks.{N}.attn.{txt_qkv, to_add_out}
  - img_mlp:   transformer_blocks.{N}.img_mlp.{w1, w2, w3}  (GateMLP)
  - txt_mlp:   transformer_blocks.{N}.txt_mlp.{w1, w2, w3}  (GateMLP)
  - mod:       transformer_blocks.{N}.{img_mod, txt_mod}[1]   (AdaLN, default OFF)

The GPT-OSS text encoder is kept frozen — fine-tuning it is outside scope.
The AutoencoderKLFlux2 VAE is always frozen.

Save format: sd-scripts native — `lora_unet_<flattened>.lora_down/up.weight` /
`alpha`. The Phase B.3 inference loader (lens_lora.py) accepts this format
directly.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
import torch.nn as nn

from .base_adapter import (
    BaseLoRAAdapter, BaseFullParameterAdapter, reject_quantized_base,
    resolve_component_lr, LORA_COMPONENT_UNET,
)
from core.adapters import LoRALinearLayer, is_adapter_covered

from core.models.lens.lens_lora import (
    iter_lens_lora_targets, DEFAULT_SCOPE, _flatten_to_sdscripts,
)


# ---------------------------------------------------------------------------
# LoRA adapter
# ---------------------------------------------------------------------------

class LensLoRAAdapter(BaseLoRAAdapter):
    """LoRA adapter for Lens DiT models."""

    def __init__(self, trainer, lora_rank: int, lora_alpha: int,
                 lora_dtype: torch.dtype = torch.float32,
                 scope: Optional[Dict[str, bool]] = None):
        super().__init__(trainer, lora_rank, lora_alpha, lora_dtype)
        self.scope = dict(DEFAULT_SCOPE) if scope is None else dict(scope)

    def apply_lora_to_unet(self, lora_layers: Dict[str, nn.Module]) -> int:
        """Wrap target Linear modules of the Lens transformer with LoRALinearLayer."""
        transformer = self.trainer.transformer
        if transformer is None:
            print("[LensLoRAAdapter] WARNING: trainer.transformer is None - skipping LoRA injection")
            return 0

        print(f"[LensLoRAAdapter] Injecting LoRA (scope={self.scope}, "
              f"rank={self.lora_rank}, alpha={self.lora_alpha})")

        count = 0
        for module_path, parent, attr, current in iter_lens_lora_targets(transformer, self.scope):
            # A CompositeAdapterLayer is yielded too and exposes
            # in_features/out_features, so wrapping it would NEST, not fail.
            if is_adapter_covered(current):
                continue

            lora_name = f"lora_unet_{_flatten_to_sdscripts(module_path)}"
            lora_layer = self.build_branch(current, lora_name)

            if isinstance(attr, int):
                parent[attr] = lora_layer
            else:
                setattr(parent, attr, lora_layer)

            self.register_lora_layer(lora_layers, lora_name, lora_layer, LORA_COMPONENT_UNET)
            count += 1

        print(f"[LensLoRAAdapter] Injected {count} LoRA layer(s) into Lens transformer")
        return count

    def apply_lora_to_text_encoders(self, lora_layers: Dict[str, nn.Module]) -> int:
        """The GPT-OSS text encoder is frozen — no LoRA applied."""
        print("[LensLoRAAdapter] GPT-OSS text encoder is frozen - no LoRA applied to TE")
        return 0

    def setup_trainable_parameters(self, lora_layers: Dict[str, nn.Module]
                                   ) -> List[Dict[str, Any]]:
        """Single optimizer parameter group for all LoRA weights."""
        return self.component_param_groups(lora_layers, {
            LORA_COMPONENT_UNET: lambda: resolve_component_lr(
                self.trainer, "unet_lr", label="Lens LoRA"),
        })

    def checkpoint_metadata(self, lora_layers: Dict[str, nn.Module],
                            step: int, epoch: int) -> Dict[str, str]:
        """sd-scripts native format, compatible with the Phase B.3 loader."""
        return {
            "model_type":              "lens",
            "modelspec.architecture":  "lens",
            "lora_rank":               str(self.lora_rank),
            "lora_alpha":              str(self.lora_alpha),
            "lora_targets":            ",".join(k for k, v in self.scope.items() if v),
            "step":                    str(step),
            "epoch":                   str(epoch),
            "format":                  "pt",
        }


# ---------------------------------------------------------------------------
# Full-parameter adapter
# ---------------------------------------------------------------------------

class LensFullParameterAdapter(BaseFullParameterAdapter):
    """Full-parameter training adapter for Lens DiT models.

    Trainable surface:
      - Lens DiT transformer when train_unet=True
      - GPT-OSS text encoder is always frozen (too large for practical FT)
      - AutoencoderKLFlux2 VAE is always frozen

    Optimizer parameter groups (3 groups, matching MMDiT double-stream):
      1. img_stream — img_qkv, to_out.0, img_mlp.*
                      (LR = unet_lr × lens_img_lr_factor)
      2. txt_stream — txt_qkv, to_add_out, txt_mlp.*
                      (LR = unet_lr × lens_txt_lr_factor)
      3. other      — img_mod, txt_mod, txt_in, img_in, time_text_embed,
                      norm_out, proj_out, pos_embed, etc.
                      (LR = unet_lr)

    Save format:
      Single safetensors with state dict prefixed with `net.` — the same
      convention as Anima full-FT checkpoints; our Phase A inference loader
      auto-strips the prefix.
    """

    def prepare_models_for_training(self):
        trainer = self.trainer
        reject_quantized_base(trainer.transformer, model_label="Lens")
        train_dit = bool(getattr(trainer, "train_unet", True))

        if train_dit and trainer.transformer is not None:
            trainer.transformer.requires_grad_(True)
            trainer.transformer.train()
            print("[LensFullParameterAdapter] Lens DiT set to train mode")

        # GPT-OSS text encoder: always frozen.
        if trainer.text_encoder is not None:
            trainer.text_encoder.requires_grad_(False)
            trainer.text_encoder.eval()
            print("[LensFullParameterAdapter] GPT-OSS text encoder is frozen")

        # VAE: always frozen.
        if trainer.vae is not None:
            trainer.vae.requires_grad_(False)
            trainer.vae.eval()

        print(f"[LensFullParameterAdapter] Models prepared (DiT trainable={train_dit})")

    def setup_trainable_parameters(self) -> List[Dict[str, Any]]:
        trainer = self.trainer
        if trainer.transformer is None:
            return []
        # Second gate, not a duplicate: a caller that builds the optimizer without
        # going through prepare_models_for_training() would otherwise still get
        # the silently-truncated parameter list this guard exists to prevent.
        reject_quantized_base(trainer.transformer, model_label="Lens")

        base_lr = resolve_component_lr(trainer, "unet_lr", label="Lens transformer")
        img_factor = float(trainer.config.get("lens_img_lr_factor", 1.0))
        txt_factor = float(trainer.config.get("lens_txt_lr_factor", 1.0))

        img_params: List[nn.Parameter] = []
        txt_params: List[nn.Parameter] = []
        other_params: List[nn.Parameter] = []

        for name, p in trainer.transformer.named_parameters():
            if not p.requires_grad:
                continue
            # img_stream: image attention QKV/out-proj and image GateMLP
            if (".attn.img_qkv" in name or ".attn.to_out" in name or ".img_mlp." in name):
                img_params.append(p)
            # txt_stream: text attention QKV/out-proj and text GateMLP
            elif (".attn.txt_qkv" in name or ".attn.to_add_out" in name or ".txt_mlp." in name):
                txt_params.append(p)
            else:
                other_params.append(p)

        groups: List[Dict[str, Any]] = []
        if img_params:
            groups.append({"params": img_params, "lr": base_lr * img_factor, "name": "img_stream"})
        if txt_params:
            groups.append({"params": txt_params, "lr": base_lr * txt_factor, "name": "txt_stream"})
        if other_params:
            groups.append({"params": other_params, "lr": base_lr, "name": "other"})

        total = sum(sum(p.numel() for p in g["params"]) for g in groups)
        print(f"[LensFullParameterAdapter] {len(groups)} param group(s), "
              f"{total:,} trainable params total "
              f"(img={len(img_params)} | txt={len(txt_params)} | other={len(other_params)})")
        return groups

    def save_checkpoint(self, step: int, epoch: int, output_path: Path):
        from safetensors.torch import save_file as _save

        trainer = self.trainer
        if trainer.transformer is None:
            print("[LensFullParameterAdapter] WARNING: no transformer to save")
            return

        if output_path.is_dir():
            output_path = output_path / f"lens_step_{step}.safetensors"
        elif not str(output_path).endswith(".safetensors"):
            output_path = Path(str(output_path) + ".safetensors")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        dit_state = trainer.transformer.state_dict()
        combined: Dict[str, torch.Tensor] = {
            f"net.{k}": v.detach().to("cpu").contiguous() for k, v in dit_state.items()
        }

        # Optional VAE bundling (default off). Lens uses the comfy-heritage
        # ``first_stage_model.*`` prefix; the single-file loader splits + reattaches
        # it into the AutoencoderKLFlux2. Absent -> loader resolves base-dir/store VAE.
        from api.param_defaults import resolve_bundle_vae
        bundle_vae = resolve_bundle_vae(getattr(trainer, "bundle_vae", None), "lens")
        vae_embedded = bundle_vae and getattr(trainer, "vae", None) is not None
        if vae_embedded:
            print(f"[LensFullParameterAdapter] Collecting VAE weights (bundle_vae)...")
            for k, v in trainer.vae.state_dict().items():
                combined[f"first_stage_model.{k}"] = v.detach().to("cpu").contiguous()

        metadata = {
            "step": str(step),
            "epoch": str(epoch),
            "model_type": "lens",
            "modelspec.architecture": "lens",
            "format": "pt",
        }

        # Base Lens directory hint so single-file reload can resolve TE/VAE/tokenizer.
        base_dir = str(getattr(trainer, "model_path", "") or "")
        if base_dir:
            metadata["component.base_dir"] = base_dir

        # Transformer config JSON (declarative; LensTransformer2DModel is a ConfigMixin).
        try:
            import json as _json
            cfg = getattr(trainer.transformer, "config", None)
            if cfg is not None:
                metadata["transformer_config"] = _json.dumps(dict(cfg))
        except Exception as _e:
            print(f"[LensFullParameterAdapter] transformer_config not serialized: {_e}")

        try:
            from core.models.common.single_file_format import build_component_metadata
            metadata.update(build_component_metadata(
                te_type="lens_gpt_oss", te_embedded=False,
                vae_type="flux2", vae_embedded=vae_embedded,
            ))
        except Exception as _e:
            print(f"[LensFullParameterAdapter] component metadata skipped: {_e}")

        print(f"[LensFullParameterAdapter] Saving to {output_path}...")
        _save(combined, str(output_path), metadata=metadata)
        total_params = sum(t.numel() for t in combined.values())
        print(f"[LensFullParameterAdapter] Saved {len(combined)} tensors "
              f"({total_params:,} params) -> {output_path}")
