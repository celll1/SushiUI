"""Anima (Cosmos-Predict2 DiT) training adapters.

Model characteristics:
  - Cosmos-Predict2 DiT (28 blocks, AdaLN-LoRA modulation, 3D RoPE)
  - Qwen3-0.6B text encoder + 6-layer LLM Adapter
  - Qwen-Image VAE (Wan VAE 2.1 latent space, 16ch)
  - Rectified Flow / Flow Matching (predicts velocity v = noise - x_0)

LoRA targets (controlled by the `scope` dict):
  - attention:   blocks.<N>.{self_attn,cross_attn}.{q,k,v,output}_proj
  - mlp:         blocks.<N>.mlp.{layer1, layer2}
  - mod:         blocks.<N>.adaln_modulation_*.{1, 2}      (default OFF)
  - llm_adapter: llm_adapter.blocks.<N>.{self_attn,cross_attn}.{q,k,v,o}_proj
                 + llm_adapter.blocks.<N>.mlp.{0, 2}
                 + llm_adapter.{in_proj, out_proj}

The Qwen3 text encoder body (transformers Qwen3Model) and the VAE are
kept frozen — fine-tuning text-encoder weights for a small DiT model is
brittle and rarely improves quality.

Save format: sd-scripts native — `lora_unet_<flattened>.lora_down.weight` /
`lora_up.weight` / `alpha`. The Phase B.3 inference loader (anima_lora.py)
accepts this format directly and also accepts the interchange format
(diffusion_model.*.lora_A/B.weight) for files produced by other tools.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
import torch.nn as nn
from safetensors.torch import save_file

from .base_adapter import BaseLoRAAdapter, BaseFullParameterAdapter
from .sd15_adapter import LoRALinearLayer

# Reuse Phase B.3 iteration + flatten helpers.
from core.models.anima.anima_lora import (
    iter_anima_lora_targets, DEFAULT_TRAINING_SCOPE, _flatten_to_sdscripts,
)


# ----------------------------------------------------------------------
# LoRA adapter
# ----------------------------------------------------------------------

class AnimaLoRAAdapter(BaseLoRAAdapter):
    """LoRA adapter for Anima DiT models."""

    def __init__(self, trainer, lora_rank: int, lora_alpha: int,
                 lora_dtype: torch.dtype = torch.float32,
                 scope: Optional[Dict[str, bool]] = None):
        super().__init__(trainer, lora_rank, lora_alpha, lora_dtype)
        self.scope = dict(DEFAULT_TRAINING_SCOPE) if scope is None else dict(scope)

    # -- LoRA injection -------------------------------------------------

    def apply_lora_to_unet(self, lora_layers: Dict[str, nn.Module]) -> int:
        """Wrap target Linear modules of the Anima DiT with LoRALinearLayer.

        The 'unet' in the interface name is historical — Anima's DiT
        (`trainer.transformer`) plays that role here. LoRA wrappers replace
        the originals in-place so the parent module's forward()
        automatically dispatches through them.
        """
        transformer = self.trainer.transformer
        if transformer is None:
            print("[AnimaLoRAAdapter] WARNING: trainer.transformer is None — skipping LoRA injection")
            return 0

        print(f"[AnimaLoRAAdapter] Injecting LoRA (scope={self.scope}, "
              f"rank={self.lora_rank}, alpha={self.lora_alpha})")

        # DiT-BlockSkip (arXiv 2603.20755): when active, LoRA must live ONLY in the
        # unskipped MIDDLE DiT blocks [front, num_blocks - back). Skipped front/back
        # blocks are frozen (no adapter, no optimizer state, no backward graph) and
        # their contribution is supplied by precomputed residual features. Only the
        # top-level DiT blocks are gated; llm_adapter.* targets are unaffected.
        import re as _re
        _bs = getattr(self.trainer, "blockskip_config", None)
        _bs_lo = _bs_hi = None
        if _bs is not None:
            _num_blocks = len(transformer.blocks)
            _bs_lo = int(_bs["front"])
            _bs_hi = _num_blocks - int(_bs["back"])
            if _bs_hi - _bs_lo < 1:
                raise ValueError(
                    f"[AnimaLoRAAdapter] BlockSkip leaves no middle blocks: "
                    f"num_blocks={_num_blocks}, front={_bs['front']}, back={_bs['back']} "
                    f"=> middle range [{_bs_lo}, {_bs_hi}). Reduce blockskip_front/back."
                )
            print(f"[AnimaLoRAAdapter] BlockSkip active: injecting LoRA only into "
                  f"middle DiT blocks [{_bs_lo}, {_bs_hi}) of {_num_blocks}.")

        count = 0
        skipped_blockskip = 0
        for module_path, parent, attr, current in iter_anima_lora_targets(transformer, self.scope):
            # Skip if this slot was already wrapped (idempotent / stacking-safe).
            if isinstance(current, LoRALinearLayer):
                continue

            # BlockSkip: gate top-level DiT block targets to the middle range.
            if _bs_lo is not None:
                _m = _re.match(r"^blocks\.(\d+)\.", module_path)
                if _m is not None:
                    _bidx = int(_m.group(1))
                    if _bidx < _bs_lo or _bidx >= _bs_hi:
                        skipped_blockskip += 1
                        continue

            lora_name = f"lora_unet_{_flatten_to_sdscripts(module_path)}"
            lora_layer = LoRALinearLayer(
                current, self.lora_rank, self.lora_alpha, lora_name, self.lora_dtype,
            )

            # parent.attr might be a normal attribute (str) or a Sequential /
            # ModuleList index (int) — handle both.
            if isinstance(attr, int):
                parent[attr] = lora_layer
            else:
                setattr(parent, attr, lora_layer)

            lora_layers[lora_name] = lora_layer
            count += 1

        if _bs_lo is not None:
            print(f"[AnimaLoRAAdapter] BlockSkip: excluded {skipped_blockskip} "
                  f"LoRA target(s) in skipped front/back blocks.")
        print(f"[AnimaLoRAAdapter] Injected {count} LoRA layer(s) into Anima DiT")
        return count

    def apply_lora_to_text_encoders(self, lora_layers: Dict[str, nn.Module]) -> int:
        """Anima keeps the Qwen3 text encoder frozen.

        The LLM Adapter (the 6-layer transformer that re-projects Qwen3
        hidden states into the DiT cross-attention input space) lives inside
        the DiT module, so LoRA on it is applied via apply_lora_to_unet()
        when scope["llm_adapter"] is enabled.
        """
        print("[AnimaLoRAAdapter] Qwen3 text encoder is frozen — no LoRA applied to TE")
        return 0

    # -- Optimizer parameters ------------------------------------------

    def setup_trainable_parameters(self, lora_layers: Dict[str, nn.Module]
                                    ) -> List[Dict[str, Any]]:
        """Single optimizer parameter group for the LoRA weights.

        Per-component LR groups (self_attn / cross_attn / mlp / mod /
        llm_adapter) can be added in a follow-up; for the initial release
        we expose a single `unet_lr` knob.
        """
        params: List[nn.Parameter] = []
        for lora_layer in lora_layers.values():
            params.extend(lora_layer.lora_down.parameters())
            params.extend(lora_layer.lora_up.parameters())
        if not params:
            return []
        return [{"params": params, "lr": getattr(self.trainer, "unet_lr", 1e-4)}]

    # -- Checkpoint --------------------------------------------------

    def save_checkpoint(self, lora_layers: Dict[str, nn.Module],
                         step: int, epoch: int, output_path: Path):
        """Save LoRA weights in sd-scripts native format."""
        state_dict: Dict[str, torch.Tensor] = {}
        alpha_value = float(self.lora_alpha)

        for lora_name, lora_layer in lora_layers.items():
            # lora_name already has the lora_unet_ prefix from injection.
            state_dict[f"{lora_name}.lora_down.weight"] = lora_layer.lora_down.weight.detach().cpu()
            state_dict[f"{lora_name}.lora_up.weight"] = lora_layer.lora_up.weight.detach().cpu()
            # Per-layer alpha — sd-scripts convention. We store the same alpha
            # for every layer (matches our rank/alpha config).
            state_dict[f"{lora_name}.alpha"] = torch.tensor(alpha_value, dtype=torch.float32)

        active_scopes = ",".join(k for k, v in self.scope.items() if v)
        metadata = {
            "model_type": "anima",
            "modelspec.architecture": "anima",
            "lora_rank": str(self.lora_rank),
            "lora_alpha": str(self.lora_alpha),
            "lora_targets": active_scopes,
            "step": str(step),
            "epoch": str(epoch),
            "format": "pt",
        }

        save_file(state_dict, str(output_path), metadata=metadata)
        print(f"[AnimaLoRAAdapter] Saved LoRA checkpoint ({len(lora_layers)} layers) -> {output_path}")


# ----------------------------------------------------------------------
# Full-parameter adapter (skeleton; full implementation in Phase C.2)
# ----------------------------------------------------------------------

class AnimaFullParameterAdapter(BaseFullParameterAdapter):
    """Full-parameter training adapter for Anima DiT models.

    Trainable surface:
      - DiT blocks (`blocks.<N>.*`) when train_unet=True
      - LLM Adapter (`llm_adapter.*`) when train_llm_adapter=True
      - Everything else (Qwen3 text encoder, Qwen-Image VAE) is frozen

    Optimizer parameter groups (per sd-scripts convention, simplified):
      1. base       — all DiT params that aren't in groups 2/3
      2. attn_mlp   — self_attn / cross_attn / mlp Linears
                      (LR = unet_lr * anima_attn_mlp_lr_factor)
      3. modulation — adaln_modulation_* Linears
                      (LR = unet_lr * anima_mod_lr_factor)
      4. llm_adapter — full LLM Adapter when train_llm_adapter=True
                       (LR = unet_lr * anima_llm_adapter_lr_factor)

    Save format:
      Single safetensors file with the DiT state dict prefixed with
      `net.` — matches the sd-scripts native layout that our Phase A
      inference loader (load_anima_dit) auto-strips. metadata carries
      model_type / modelspec.architecture so the file is self-describing.
    """

    def prepare_models_for_training(self):
        trainer = self.trainer
        train_dit = bool(getattr(trainer, "train_unet", True))
        train_adapter_only = bool(
            getattr(trainer, "train_llm_adapter",
                    trainer.config.get("train_llm_adapter", True))
        )

        # DiT
        if train_dit and trainer.transformer is not None:
            trainer.transformer.requires_grad_(True)
            trainer.transformer.train()
            # If the user opted out of LLM Adapter training, freeze just that
            # submodule even though the rest of the DiT is trainable.
            if not train_adapter_only and hasattr(trainer.transformer, "llm_adapter"):
                trainer.transformer.llm_adapter.requires_grad_(False)
                trainer.transformer.llm_adapter.eval()
                print("[AnimaFullParameterAdapter] LLM Adapter frozen (train_llm_adapter=False)")
            print("[AnimaFullParameterAdapter] Anima DiT set to train mode")

        # Text encoder (Qwen3): always frozen.
        if trainer.text_encoder is not None:
            trainer.text_encoder.requires_grad_(False)
            trainer.text_encoder.eval()
            print("[AnimaFullParameterAdapter] Qwen3 text encoder is frozen")

        # VAE: always frozen.
        if trainer.vae is not None:
            trainer.vae.requires_grad_(False)
            trainer.vae.eval()

        print(f"[AnimaFullParameterAdapter] Models prepared for training "
              f"(DiT trainable={train_dit}, LLM Adapter trainable={train_adapter_only})")

    def setup_trainable_parameters(self) -> List[Dict[str, Any]]:
        trainer = self.trainer
        if trainer.transformer is None:
            return []

        base_lr = getattr(trainer, "unet_lr", None) or getattr(trainer, "learning_rate", 1e-5)
        attn_mlp_factor = float(trainer.config.get("anima_attn_mlp_lr_factor", 1.0))
        mod_factor = float(trainer.config.get("anima_mod_lr_factor", 1.0))
        adapter_factor = float(trainer.config.get("anima_llm_adapter_lr_factor", 1.0))

        # Bucket parameters by named-path prefix.
        base_params: List[nn.Parameter] = []
        attn_mlp_params: List[nn.Parameter] = []
        mod_params: List[nn.Parameter] = []
        adapter_params: List[nn.Parameter] = []
        for name, p in trainer.transformer.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("llm_adapter"):
                adapter_params.append(p)
            elif ".adaln_modulation_" in name or name.startswith("adaln_modulation_"):
                mod_params.append(p)
            elif (".self_attn." in name or ".cross_attn." in name
                  or ".mlp." in name):
                attn_mlp_params.append(p)
            else:
                base_params.append(p)

        groups: List[Dict[str, Any]] = []
        if base_params:
            groups.append({"params": base_params, "lr": base_lr})
        if attn_mlp_params:
            groups.append({"params": attn_mlp_params, "lr": base_lr * attn_mlp_factor})
        if mod_params:
            groups.append({"params": mod_params, "lr": base_lr * mod_factor})
        if adapter_params:
            groups.append({"params": adapter_params, "lr": base_lr * adapter_factor})

        total = sum(sum(p.numel() for p in g["params"]) for g in groups)
        print(f"[AnimaFullParameterAdapter] {len(groups)} param group(s), "
              f"{total:,} trainable params total "
              f"(base={len(base_params)} | attn_mlp={len(attn_mlp_params)} | "
              f"mod={len(mod_params)} | adapter={len(adapter_params)})")
        return groups

    def save_checkpoint(self, step: int, epoch: int, output_path: Path):
        from safetensors.torch import save_file

        trainer = self.trainer
        if trainer.transformer is None:
            print("[AnimaFullParameterAdapter] WARNING: no transformer to save")
            return

        # Normalise output path to a .safetensors file.
        if output_path.is_dir():
            output_path = output_path / f"anima_step_{step}.safetensors"
        elif not str(output_path).endswith(".safetensors"):
            output_path = Path(str(output_path) + ".safetensors")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the DiT state dict with the `net.` prefix — the sd-scripts
        # native convention also accepted by our Phase A inference loader,
        # which auto-strips the prefix on load.
        dit_state = trainer.transformer.state_dict()
        combined: Dict[str, torch.Tensor] = {}
        for k, v in dit_state.items():
            combined[f"net.{k}"] = v.detach().to("cpu").contiguous()

        # Optional VAE bundling (default off). Anima uses the comfy-heritage
        # ``first_stage_model.*`` prefix; the loader splits + reattaches it into the
        # AutoencoderKLQwenImage. Absent -> loader resolves the companion/store VAE.
        from api.param_defaults import resolve_bundle_vae
        bundle_vae = resolve_bundle_vae(getattr(trainer, "bundle_vae", None), "anima")
        vae_embedded = bundle_vae and getattr(trainer, "vae", None) is not None
        if vae_embedded:
            print(f"[AnimaFullParameterAdapter] Collecting VAE weights (bundle_vae)...")
            for k, v in trainer.vae.state_dict().items():
                combined[f"first_stage_model.{k}"] = v.detach().to("cpu").contiguous()

        metadata = {
            "step": str(step),
            "epoch": str(epoch),
            "model_type": "anima",
            "modelspec.architecture": "anima",
            "format": "pt",
        }

        # Transformer config JSON (declarative) + component hints. The TE is never
        # embedded (resolved out-of-band by anima_loader); the VAE is embedded only
        # when bundle_vae is set (else the loader resolves companion/store VAE).
        try:
            import json as _json
            from core.models.anima.anima_models import ANIMA_DIT_CONFIG
            metadata["transformer_config"] = _json.dumps(dict(ANIMA_DIT_CONFIG))
        except Exception as _e:
            print(f"[AnimaFullParameterAdapter] transformer_config not serialized: {_e}")
        try:
            from core.models.common.single_file_format import build_component_metadata
            metadata.update(build_component_metadata(
                te_type="qwen3", te_embedded=False,
                vae_type="qwen_image", vae_channels=16, vae_embedded=vae_embedded,
            ))
        except Exception as _e:
            print(f"[AnimaFullParameterAdapter] component metadata skipped: {_e}")

        print(f"[AnimaFullParameterAdapter] Saving to {output_path}...")
        save_file(combined, str(output_path), metadata=metadata)
        total_params = sum(t.numel() for t in combined.values())
        print(f"[AnimaFullParameterAdapter] Saved {len(combined)} tensors "
              f"({total_params:,} params) -> {output_path}")
