"""MiniT2I (pixel-space MM-JiT) training adapters.

Model characteristics:
  - Pixel-space MM-DiT (double_blocks + txt_preamble_blocks), no VAE
  - Frozen FLAN-T5-Large text encoder (Phase C may unfreeze it)
  - Flow matching, x0 prediction (loss in base_trainer.train_step_minit2i)

LoRA target scope (minit2i_lora.iter_minit2i_lora_targets):
  attn:      double/preamble blocks {img_qkv,txt_qkv,img_attn_proj,txt_attn_proj,qkv,attn_proj}
  mlp:       {img_mlp,txt_mlp,mlp}.{w1,w2,w3}
  txt_embed: txt_embedder, pooled_embedder

Save format: sd-scripts-style `lora_unet_<flat>.lora_down/up.weight` / `alpha`
(flat uses the "."<->"__" reversible encoding from minit2i_lora). Full-parameter
saves a MiniT2I single-file (transformer + variant metadata) via vendor.save_single_file.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
import torch.nn as nn
from safetensors.torch import save_file

from .base_adapter import BaseLoRAAdapter, BaseFullParameterAdapter, reject_quantized_base
from .sd15_adapter import LoRALinearLayer

from core.models.minit2i.minit2i_lora import (
    iter_minit2i_lora_targets, DEFAULT_SCOPE, flatten_to_key,
    iter_minit2i_te_lora_targets, TE_DEFAULT_SCOPE, flatten_to_te_key,
)


def _repa_sidecar_path(checkpoint_path: str) -> str:
    """REPA projector sidecar path next to a checkpoint (suffix-precise).

    Replaces only a trailing ``.safetensors`` so a directory component containing
    ``.safetensors`` cannot corrupt the path. Must match the resume loader in
    base_trainer._setup_repa.
    """
    if checkpoint_path.endswith(".safetensors"):
        return checkpoint_path[: -len(".safetensors")] + ".repa.safetensors"
    return checkpoint_path + ".repa.safetensors"


class MiniT2ILoRAAdapter(BaseLoRAAdapter):
    """LoRA adapter for the MiniT2I MM-JiT transformer (+ optional FLAN-T5 TE LoRA)."""

    def __init__(self, trainer, lora_rank: int, lora_alpha: int,
                 lora_dtype: torch.dtype = torch.float32,
                 scope: Optional[Dict[str, bool]] = None,
                 te_scope: Optional[Dict[str, bool]] = None):
        super().__init__(trainer, lora_rank, lora_alpha, lora_dtype)
        self.scope = dict(DEFAULT_SCOPE) if scope is None else dict(scope)
        self.te_scope = dict(TE_DEFAULT_SCOPE) if te_scope is None else dict(te_scope)

    def apply_lora_to_unet(self, lora_layers: Dict[str, nn.Module]) -> int:
        transformer = self.trainer.transformer
        if transformer is None:
            print("[MiniT2ILoRAAdapter] WARNING: trainer.transformer is None - skipping")
            return 0
        print(f"[MiniT2ILoRAAdapter] Injecting LoRA (scope={self.scope}, "
              f"rank={self.lora_rank}, alpha={self.lora_alpha})")
        count = 0
        for module_path, parent, attr, current in iter_minit2i_lora_targets(transformer, self.scope):
            if isinstance(current, LoRALinearLayer):
                continue
            lora_name = flatten_to_key(module_path)  # "lora_unet_<flat>"
            lora_layer = LoRALinearLayer(current, self.lora_rank, self.lora_alpha, lora_name, self.lora_dtype)
            if isinstance(attr, int):
                parent[attr] = lora_layer
            else:
                setattr(parent, attr, lora_layer)
            lora_layers[lora_name] = lora_layer
            count += 1
        print(f"[MiniT2ILoRAAdapter] Injected {count} LoRA layer(s)")
        return count

    def apply_lora_to_text_encoders(self, lora_layers: Dict[str, nn.Module]) -> int:
        """LoRA on FLAN-T5 (only called when train_text_encoder=True)."""
        text_encoder = self.trainer.text_encoder
        if text_encoder is None:
            print("[MiniT2ILoRAAdapter] WARNING: trainer.text_encoder is None - skipping TE LoRA")
            return 0
        print(f"[MiniT2ILoRAAdapter] Injecting FLAN-T5 LoRA (te_scope={self.te_scope}, "
              f"rank={self.lora_rank}, alpha={self.lora_alpha})")
        count = 0
        for module_path, parent, attr, current in iter_minit2i_te_lora_targets(text_encoder, self.te_scope):
            if isinstance(current, LoRALinearLayer):
                continue
            lora_name = flatten_to_te_key(module_path)  # "lora_te_<flat>"
            lora_layer = LoRALinearLayer(current, self.lora_rank, self.lora_alpha, lora_name, self.lora_dtype)
            setattr(parent, attr, lora_layer)
            lora_layers[lora_name] = lora_layer
            count += 1
        print(f"[MiniT2ILoRAAdapter] Injected {count} FLAN-T5 LoRA layer(s)")
        return count

    def setup_trainable_parameters(self, lora_layers: Dict[str, nn.Module]) -> List[Dict[str, Any]]:
        unet_params: List[nn.Parameter] = []
        te_params: List[nn.Parameter] = []
        for name, lora_layer in lora_layers.items():
            target = te_params if name.startswith("lora_te_") else unet_params
            target.extend(lora_layer.lora_down.parameters())
            target.extend(lora_layer.lora_up.parameters())
        groups: List[Dict[str, Any]] = []
        if unet_params:
            base_lr = getattr(self.trainer, "unet_lr", None) or 1e-4
            lr_factor = float(self.trainer.config.get("minit2i_lr_factor", 1.0))
            groups.append({"params": unet_params, "lr": base_lr * lr_factor})
        if te_params:
            te_lr = getattr(self.trainer, "text_encoder_lr", None) or getattr(self.trainer, "learning_rate", 1e-4)
            groups.append({"params": te_params, "lr": te_lr})
            print(f"[MiniT2ILoRAAdapter] FLAN-T5 LoRA param group lr={te_lr}")
        # REPA projector (training-only alignment head). Without this group the
        # projector would never update and the alignment target would be random,
        # defeating REPA. Appended last so the param-group order is stable on resume.
        if getattr(self.trainer, "repa_enable", False) and getattr(self.trainer, "repa_projector", None) is not None:
            p_params = [p for p in self.trainer.repa_projector.parameters() if p.requires_grad]
            if p_params:
                proj_base_lr = getattr(self.trainer, "unet_lr", None) or getattr(self.trainer, "learning_rate", 1e-4)
                proj_lr = proj_base_lr * float(getattr(self.trainer, "repa_proj_lr_factor", 1.0))
                print(f"[MiniT2ILoRAAdapter] {sum(p.numel() for p in p_params):,} trainable params (REPA projector), lr={proj_lr}")
                groups.append({"params": p_params, "lr": proj_lr})
        return groups

    def save_checkpoint(self, lora_layers: Dict[str, nn.Module], step: int, epoch: int, output_path: Path):
        state_dict: Dict[str, torch.Tensor] = {}
        alpha_value = float(self.lora_alpha)
        for lora_name, lora_layer in lora_layers.items():
            state_dict[f"{lora_name}.lora_down.weight"] = lora_layer.lora_down.weight.detach().cpu()
            state_dict[f"{lora_name}.lora_up.weight"] = lora_layer.lora_up.weight.detach().cpu()
            state_dict[f"{lora_name}.alpha"] = torch.tensor(alpha_value, dtype=torch.float32)
        active_scopes = ",".join(k for k, v in self.scope.items() if v)
        active_te_scopes = ",".join(k for k, v in self.te_scope.items() if v)
        has_te = any(name.startswith("lora_te_") for name in lora_layers)
        metadata = {
            "model_type": "minit2i",
            "modelspec.architecture": "minit2i",
            "variant": str(getattr(self.trainer, "minit2i_variant", "") or ""),
            "lora_rank": str(self.lora_rank),
            "lora_alpha": str(self.lora_alpha),
            "lora_targets": active_scopes,
            "lora_te_targets": active_te_scopes if has_te else "",
            "step": str(step),
            "epoch": str(epoch),
            "format": "pt",
        }
        save_file(state_dict, str(output_path), metadata=metadata)
        print(f"[MiniT2ILoRAAdapter] Saved LoRA checkpoint ({len(lora_layers)} layers) -> {output_path}")

        # REPA projector (training-only): saved alongside for resume; not in the LoRA file.
        if getattr(self.trainer, "repa_enable", False) and getattr(self.trainer, "repa_projector", None) is not None:
            try:
                from safetensors.torch import save_file as _save_file
                sib = _repa_sidecar_path(str(output_path))
                psd = {k: v.detach().cpu().contiguous().float()
                       for k, v in self.trainer.repa_projector.state_dict().items()}
                _save_file(psd, sib)
                print(f"[MiniT2ILoRAAdapter] Saved REPA projector -> {sib}")
            except Exception as _e:
                print(f"[MiniT2ILoRAAdapter] WARNING: REPA projector save failed: {_e}")


class MiniT2IFullParameterAdapter(BaseFullParameterAdapter):
    """Full-parameter adapter for MiniT2I (transformer; FLAN-T5 when train_text_encoder).

    MiniT2I is small (B/16 ~0.3B, L/16 ~1.8B), so full fine-tuning is practical.
    When `train_text_encoder` is set, FLAN-T5 is unfrozen and bundled into the
    saved single-file (the inference loader reads the embedded text encoder).
    """

    def _train_te(self) -> bool:
        return bool(getattr(self.trainer, "train_text_encoder", False)) and self.trainer.text_encoder is not None

    def prepare_models_for_training(self):
        trainer = self.trainer
        reject_quantized_base(trainer.transformer, model_label="MiniT2I")
        if getattr(trainer, "train_unet", True) and trainer.transformer is not None:
            trainer.transformer.requires_grad_(True)
            trainer.transformer.train()
            print("[MiniT2IFullParameterAdapter] MM-JiT transformer set to train mode")
        if trainer.text_encoder is not None:
            if self._train_te():
                trainer.text_encoder.requires_grad_(True)
                trainer.text_encoder.train()
                print("[MiniT2IFullParameterAdapter] FLAN-T5 text encoder set to train mode")
            else:
                trainer.text_encoder.requires_grad_(False)
                trainer.text_encoder.eval()
                print("[MiniT2IFullParameterAdapter] FLAN-T5 text encoder is frozen")

    def setup_trainable_parameters(self) -> List[Dict[str, Any]]:
        trainer = self.trainer
        # Second gate, not a duplicate: a caller that builds the optimizer without
        # going through prepare_models_for_training() would otherwise still get
        # the silently-truncated parameter list this guard exists to prevent.
        reject_quantized_base(trainer.transformer, model_label="MiniT2I")
        groups: List[Dict[str, Any]] = []
        if trainer.transformer is not None:
            t_params = [p for p in trainer.transformer.parameters() if p.requires_grad]
            if t_params:
                base_lr = getattr(trainer, "unet_lr", None) or getattr(trainer, "learning_rate", 1e-5)
                print(f"[MiniT2IFullParameterAdapter] {sum(p.numel() for p in t_params):,} trainable params (transformer)")
                groups.append({"params": t_params, "lr": base_lr})
        if self._train_te():
            te_params = [p for p in trainer.text_encoder.parameters() if p.requires_grad]
            if te_params:
                te_lr = getattr(trainer, "text_encoder_lr", None) or getattr(trainer, "learning_rate", 1e-5)
                print(f"[MiniT2IFullParameterAdapter] {sum(p.numel() for p in te_params):,} trainable params (FLAN-T5), lr={te_lr}")
                groups.append({"params": te_params, "lr": te_lr})
        # REPA projector (training-only alignment head). Joins the optimizer so it is
        # updated; appended last so the param-group order is stable across resume.
        if getattr(trainer, "repa_enable", False) and getattr(trainer, "repa_projector", None) is not None:
            p_params = [p for p in trainer.repa_projector.parameters() if p.requires_grad]
            if p_params:
                proj_base_lr = getattr(trainer, "unet_lr", None) or getattr(trainer, "learning_rate", 1e-5)
                proj_lr = proj_base_lr * float(getattr(trainer, "repa_proj_lr_factor", 1.0))
                print(f"[MiniT2IFullParameterAdapter] {sum(p.numel() for p in p_params):,} trainable params (REPA projector), lr={proj_lr}")
                groups.append({"params": p_params, "lr": proj_lr})
        return groups

    def save_checkpoint(self, step: int, epoch: int, output_path: Path):
        from core.models.minit2i.vendor.single_file import save_single_file
        trainer = self.trainer
        if trainer.transformer is None:
            print("[MiniT2IFullParameterAdapter] WARNING: no transformer to save")
            return
        output_path = Path(output_path)
        if output_path.is_dir():
            output_path = output_path / f"minit2i_step_{step}.safetensors"
        elif not str(output_path).endswith(".safetensors"):
            output_path = Path(str(output_path) + ".safetensors")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        variant = getattr(trainer, "minit2i_variant", None) or "b16"
        # Bundle the trained FLAN-T5 into the single-file so inference loads it.
        text_encoder = trainer.text_encoder if self._train_te() else None
        # VAE bundling (default off) applies only to latent variants; pixel-space
        # (vae_type="none") has no VAE, so this is a no-op there. Uses the sushiUI-v2
        # common ``vae.`` prefix (VAE_PREFIX).
        from api.param_defaults import resolve_bundle_vae
        bundle_vae = resolve_bundle_vae(getattr(trainer, "bundle_vae", None), "minit2i")
        vae_to_bundle = trainer.vae if (bundle_vae and getattr(trainer, "vae", None) is not None) else None
        save_single_file(str(output_path), trainer.transformer, variant=variant,
                         text_encoder=text_encoder, vae=vae_to_bundle,
                         extra_metadata={"step": str(step), "epoch": str(epoch)})
        print(f"[MiniT2IFullParameterAdapter] Saved single-file "
              f"({'transformer+FLAN-T5' if text_encoder is not None else 'transformer'}) -> {output_path}")

        # REPA projector (training-only): saved alongside the checkpoint for resume,
        # NOT embedded in the inference single-file. Sibling name pairs 1:1 with it.
        if getattr(trainer, "repa_enable", False) and getattr(trainer, "repa_projector", None) is not None:
            try:
                from safetensors.torch import save_file as _save_file
                sib = _repa_sidecar_path(str(output_path))
                sd = {k: v.detach().cpu().contiguous().float()
                      for k, v in trainer.repa_projector.state_dict().items()}
                _save_file(sd, sib)
                print(f"[MiniT2IFullParameterAdapter] Saved REPA projector -> {sib}")
            except Exception as _e:
                print(f"[MiniT2IFullParameterAdapter] WARNING: REPA projector save failed: {_e}")
