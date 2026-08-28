"""
ControlNet Trainer for Stable Diffusion models.

This is a modular implementation using model-specific adapters:
- ControlNetSD15Adapter: SD1.5 Standard ControlNet / LLLite
- ControlNetSDXLAdapter: SDXL Standard ControlNet / LLLite [Phase 3]

Implements the third training mode alongside LoRA and Full Parameter.
ControlNet training freezes UNet/VAE/TE entirely and only trains
the ControlNet module.

References:
- diffusers ControlNetModel (Apache-2 license)
- sd-scripts (Apache-2 license) by kohya-ss (LLLite implementation)

Author: Claude (2026-01-26)
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn

from .base_trainer import BaseTrainer, log_verbose
from .adapters import ControlNetSD15Adapter, ControlNetSDXLAdapter
from .crop_planner import OutpaintControlPlanner
from .image_preprocessing import flatten_to_rgb


class ControlNetTrainer(BaseTrainer):
    """
    ControlNet Trainer for SD1.5/SDXL models.

    Uses model-specific adapters for ControlNet creation, training,
    and checkpoint management.

    Supports:
    - Standard ControlNet (diffusers ControlNetModel)
    - ControlNet-LLLite (kohya-ss sd-scripts compatible) [Phase 2]
    """

    def __init__(
        self,
        controlnet_type: str = "standard",
        controlnet_pretrained_path: Optional[str] = None,
        init_from_unet: bool = True,
        # LLLite parameters (Phase 2)
        lllite_conditioning_channels: int = 32,
        lllite_rank: int = 64,
        # Condition generation (Phase 4)
        condition_preprocessors: Optional[List[str]] = None,
        condition_cache_mode: str = "on_the_fly",
        # Outpaint-native conditioning (PART B): self-supervised crop->full.
        # conditioning_mode="preprocessor" (default) = existing behavior (paired /
        # aux-preprocessor condition images). "outpaint" = build the 4-ch
        # crop+mask conditioning from each item's OWN image (no paired dataset).
        conditioning_mode: str = "preprocessor",
        outpaint_crop_min_area: float = 0.15,
        outpaint_crop_max_area: float = 0.8,
        outpaint_edge_anchor_prob: float = 0.34,
        outpaint_corner_anchor_prob: float = 0.33,
        outpaint_mask_channel: bool = True,
        outpaint_known_loss_weight: float = 0.3,
        outpaint_seam_loss_boost: float = 0.0,
        outpaint_seam_ring_width: int = 1,
        outpaint_seam_grad_lambda: float = 0.0,
        outpaint_loss_normalize: bool = False,
        # R1 (scratchpad/outpaint_boundary_structure_fix.md D3-R1): per-sample
        # randomized crop_mask_condition edge softness range, in canvas px. Both
        # default to 0.0 -> OutpaintControlPlanner.feather_for() always returns
        # 0.0 -> build_crop_mask_condition's razor-sharp default path -> byte-
        # identical to before this feature existed unless a run opts in.
        outpaint_edge_feather_min_px: float = 0.0,
        outpaint_edge_feather_max_px: float = 0.0,
        **kwargs
    ):
        """
        Initialize ControlNet Trainer.

        Args:
            controlnet_type: "standard" (diffusers ControlNetModel) or "lllite" (sd-scripts compatible)
            controlnet_pretrained_path: Path to existing ControlNet checkpoint for resume
            init_from_unet: Initialize ControlNet weights from base UNet (standard only)
            lllite_conditioning_channels: Number of conditioning channels for LLLite (Phase 2)
            lllite_rank: Rank for LLLite linear layers (Phase 2)
            condition_preprocessors: List of controlnet-aux preprocessor types (Phase 4)
            condition_cache_mode: "pre_generate" or "on_the_fly" (Phase 4)
            **kwargs: Additional arguments passed to BaseTrainer
        """
        # ControlNet-specific settings (set before super().__init__)
        self.controlnet_type = controlnet_type
        self.controlnet_pretrained_path = controlnet_pretrained_path
        self.init_from_unet = init_from_unet
        self.lllite_conditioning_channels = lllite_conditioning_channels
        self.lllite_rank = lllite_rank
        self.condition_preprocessors = condition_preprocessors
        self.condition_cache_mode = condition_cache_mode

        # Outpaint-native conditioning (PART B).
        self.conditioning_mode = str(conditioning_mode or "preprocessor")
        self.outpaint_crop_min_area = float(outpaint_crop_min_area)
        self.outpaint_crop_max_area = float(outpaint_crop_max_area)
        self.outpaint_edge_anchor_prob = float(outpaint_edge_anchor_prob)
        self.outpaint_corner_anchor_prob = float(outpaint_corner_anchor_prob)
        self.outpaint_mask_channel = bool(outpaint_mask_channel)
        # Clamp to the valid half-open range [0.0, 0.5). base_trainer's gen-region
        # metric gate is `_gen_mask = (_wm > 0.5)` (weight = known_w + (1-known_w)*gate),
        # so a keep-weight >= 0.5 would put the KNOWN region on the "generate" side
        # of that threshold and corrupt the gen-region-only metric; a negative
        # weight would invert the loss on the known region entirely. 0.5 itself is
        # excluded so the keep region always stays strictly below the gen-mask
        # threshold.
        _known_w = float(outpaint_known_loss_weight)
        if not (0.0 <= _known_w < 0.5):
            _clamped = min(max(_known_w, 0.0), 0.499999)
            print(f"[ControlNet Trainer] WARNING: outpaint_known_loss_weight={_known_w} "
                  f"is outside the valid range [0.0, 0.5); clamping to {_clamped}")
            _known_w = _clamped
        self.outpaint_known_loss_weight = _known_w
        self.outpaint_seam_loss_boost = float(outpaint_seam_loss_boost)
        # Number of seam rings the boost covers. 1 (default) = current 1-cell
        # generate-side ring (byte-identical). 2 adds a second ring (one more
        # max_pool2d dilation step outward) weighted at half the boost increment
        # of the first ring. Clamped to {1, 2} -- any other value is invalid.
        _ring_w = int(outpaint_seam_ring_width)
        if _ring_w not in (1, 2):
            _clamped_ring_w = min(max(_ring_w, 1), 2)
            print(f"[ControlNet Trainer] WARNING: outpaint_seam_ring_width={_ring_w} "
                  f"is outside the valid set {{1, 2}}; clamping to {_clamped_ring_w}")
            _ring_w = _clamped_ring_w
        self.outpaint_seam_ring_width = _ring_w
        # Weight (lambda) of the cross-seam error-continuity aux term (native
        # prediction space, no x0 reconstruction). 0.0 (default) = off, byte-
        # identical loss. Never negative.
        self.outpaint_seam_grad_lambda = max(0.0, float(outpaint_seam_grad_lambda))
        # Opt-in (default False = current byte-identical behavior). When True,
        # the weighted-loss reduction in base_trainer.train_step_controlnet
        # divides each sample's weighted loss by that sample's own mean weight,
        # decoupling per-sample loss scale from the known/generate rect area.
        self.outpaint_loss_normalize = bool(outpaint_loss_normalize)
        # R1 edge-feather range (see param docstring above); clamp to a sane
        # non-negative, min<=max range the same way the other outpaint ranges
        # (min_area/max_area) are clamped in OutpaintControlPlanner.
        self.outpaint_edge_feather_min_px = float(max(0.0, outpaint_edge_feather_min_px))
        self.outpaint_edge_feather_max_px = float(max(
            self.outpaint_edge_feather_min_px, outpaint_edge_feather_max_px
        ))
        self._is_outpaint_mode = (self.conditioning_mode == "outpaint")
        # Channel count for the ControlNet conditioning-embedding conv: outpaint mode
        # adds a binary known-mask channel (crop RGB + mask = 4ch) unless the mask
        # channel is ablated off; every other mode is 3-ch RGB.
        self.conditioning_channels = 4 if (self._is_outpaint_mode and self.outpaint_mask_channel) else 3
        # Deterministic per-(item, epoch) crop-rect sampler (built lazily; needs seed).
        self._outpaint_planner = None

        # Outpaint conditioning is structurally incompatible with LLLite: the
        # LLLiteConditioningEncoder hardcodes in_channels=3 (lllite_module.py
        # LLLiteAttentionModule.__init__), but outpaint mode needs a 4-channel
        # (crop RGB + known-mask) conditioning tensor. Feeding that mismatched
        # tensor through would only surface as an opaque Conv2d shape
        # RuntimeError deep in the first forward pass -- fail fast here instead,
        # before any model weights are loaded.
        if self._is_outpaint_mode and self.controlnet_type == "lllite":
            raise ValueError(
                f"{self.__class__.__name__}: conditioning_mode='outpaint' requires "
                f"controlnet_type='standard'. LLLite's conditioning encoder is "
                f"hardcoded to 3 input channels and is not yet parameterized for "
                f"the 4-channel (crop RGB + known-mask) outpaint conditioning."
            )

        # ControlNet module storage (set by _create_controlnet)
        self.controlnet: Optional[nn.Module] = None

        # ControlNet training does NOT train UNet/TE
        self.train_unet = False
        self.train_text_encoder = False
        self.train_image_encoder = False

        # Flag to signal base_trainer to load condition images
        self.use_condition_images = True

        # ControlNet resumes its OWN checkpoint format (standard = directory,
        # lllite = adapter .safetensors) in this __init__ after the base setup.
        # Tell base_trainer.__init__ NOT to run its file-based resume detection
        # (which would try to load an lllite adapter as a full base model, or
        # silently find nothing for a directory checkpoint). Set before super().
        self._manages_own_resume = True

        # Initialize base trainer (loads model components)
        super().__init__(**kwargs)

        # Override log prefix
        self.log_prefix = "[ControlNet Trainer]"

        # Outpaint conditioning builds its 4-ch cond by a plain resize of the
        # FULL image (base_trainer.py build_crop_mask_condition), while the
        # target latents may instead honor bucket_strategy="crop"/"random_crop"
        # or crop-augment (_crop_spec) -- either would make the known-region
        # cond pixels stop corresponding to the teacher latent's actual crop,
        # breaking the self-supervision the outpaint mode relies on. Only
        # bucket_strategy="resize" with crop-augment disabled keeps the cond
        # and latent geometry aligned. Fail fast at construction (self.config
        # is the full YAML train_config dict, populated by BaseTrainer.__init__
        # just above) rather than silently training on misaligned pairs.
        if self._is_outpaint_mode:
            _bucket_strategy = str((self.config or {}).get("bucket_strategy", "resize") or "resize")
            _crop_augment_enable = bool((self.config or {}).get("crop_augment_enable", False))
            if _bucket_strategy != "resize" or _crop_augment_enable:
                raise ValueError(
                    f"{self.log_prefix} conditioning_mode='outpaint' requires "
                    f"bucket_strategy='resize' and crop_augment_enable=False (got "
                    f"bucket_strategy={_bucket_strategy!r}, "
                    f"crop_augment_enable={_crop_augment_enable}). Outpaint "
                    f"conditioning is built from a plain resize of the full "
                    f"image; any crop-based bucketing or crop augmentation "
                    f"desynchronizes the known-region conditioning from the "
                    f"target latent's actual crop, breaking self-supervision."
                )
            # pre_encoded_cache encodes the target latent through
            # encode_image()'s default bucket_strategy="crop" (center crop),
            # regardless of the configured bucket_strategy, so for non-square
            # sources the cached latent is center-cropped while the outpaint
            # conditioning is a plain resize -- the same cond/latent geometry
            # desync as above, via the cache path the resize guard cannot see.
            # The onthefly modes pass bucket_strategy through and stay aligned.
            _latent_mode = str((self.config or {}).get("latent_encoding_mode", "swap_onthefly") or "swap_onthefly")
            if _latent_mode == "pre_encoded_cache":
                raise ValueError(
                    f"{self.log_prefix} conditioning_mode='outpaint' does not "
                    f"support latent_encoding_mode='pre_encoded_cache': the disk "
                    f"latent cache is encoded with bucket_strategy='crop' (center "
                    f"crop), while outpaint conditioning is built from a plain "
                    f"resize of the full image; for non-square sources the cached "
                    f"target latent and the known-region conditioning desynchronize, "
                    f"breaking self-supervision. Use latent_encoding_mode="
                    f"'swap_onthefly' or 'onthefly_gpu' instead."
                )

        # Validate model type (only SD1.5/SDXL supported)
        if self.is_zimage or self.is_flux2:
            model_type = "Z-Image" if self.is_zimage else "FLUX.2"
            raise ValueError(
                f"ControlNet training is only supported for SD1.5 and SDXL models. "
                f"Detected model type: {model_type}"
            )
        # DEUS support removed - architecture no longer maintained
        # if self.is_deus:
        #     raise ValueError(...)

        # Freeze all base model components
        self._freeze_base_models()

        # Create model-specific adapter
        self._create_adapter()

        # Create ControlNet using adapter
        self._create_controlnet()

        # Resume: load the ControlNet adapter weights from its own checkpoint.
        # base_trainer.__init__'s resume path only recognizes *.safetensors FILE
        # checkpoints (LoRA/full-ft) via _list_checkpoint_entries, so it silently
        # finds nothing for ControlNet (which saves DIRECTORY checkpoints
        # *_controlnet_step_NNNNNN/) and would start from scratch. Load them here
        # and set _loaded_checkpoint_path so base_trainer.train() restores
        # global_step / optimizer / epoch from the matching *_state.json.
        if self.resume_from_checkpoint:
            self._resume_controlnet_weights()

        print(f"{self.log_prefix} Initialized")
        print(f"{self.log_prefix} ControlNet type: {self.controlnet_type}")
        print(f"{self.log_prefix} Model type: {'SDXL' if self.is_sdxl else 'SD1.5'}")

    def _freeze_base_models(self):
        """Freeze all base model components (UNet, VAE, TE)."""
        print(f"{self.log_prefix} Freezing all base model components...")

        if self.unet is not None:
            self.unet.requires_grad_(False)
            self.unet.eval()
            print(f"  UNet: frozen")

        if self.vae is not None:
            self.vae.requires_grad_(False)
            self.vae.eval()
            print(f"  VAE: frozen")

        if self.text_encoder is not None:
            self.text_encoder.requires_grad_(False)
            self.text_encoder.eval()
            print(f"  Text Encoder 1: frozen")

        if self.text_encoder_2 is not None:
            self.text_encoder_2.requires_grad_(False)
            self.text_encoder_2.eval()
            print(f"  Text Encoder 2: frozen")

    def _create_adapter(self):
        """Create model-specific ControlNet adapter based on detected model type."""
        if self.is_sdxl:
            self.adapter = ControlNetSDXLAdapter(self, self.controlnet_type, self.conditioning_channels)
            print(f"{self.log_prefix} Using ControlNetSDXLAdapter ({self.controlnet_type}, {self.conditioning_channels}ch cond)")
        else:
            self.adapter = ControlNetSD15Adapter(self, self.controlnet_type, self.conditioning_channels)
            print(f"{self.log_prefix} Using ControlNetSD15Adapter ({self.controlnet_type}, {self.conditioning_channels}ch cond)")

    def _create_controlnet(self):
        """Create ControlNet model using adapter."""
        print(f"{self.log_prefix} Creating ControlNet...")

        self.controlnet = self.adapter.create_controlnet(
            init_from_unet=self.init_from_unet,
            pretrained_path=self.controlnet_pretrained_path,
        )

        # Enable gradient checkpointing for ControlNet (honors the per-run flag)
        if self.gradient_checkpointing and hasattr(self.controlnet, 'enable_gradient_checkpointing'):
            self.controlnet.enable_gradient_checkpointing()
            print(f"{self.log_prefix} Gradient checkpointing enabled for ControlNet")

        print(f"{self.log_prefix} ControlNet created successfully")

    def _is_complete_controlnet_checkpoint(self, p: "Path") -> bool:
        """True if ``p`` contains actual weights, not just an mkdir'd shell.

        A ``save_pretrained`` call that gets interrupted mid-write (e.g. a
        dying/dead CUDA context during an emergency save -- see
        ``controlnet_sdxl_adapter._save_standard_checkpoint``, which mkdirs the
        directory BEFORE writing any weights) leaves an empty or partial
        directory on disk. Resume-selection must skip those, or it would try
        to load a checkpoint with no weights (or silently succeed on garbage).

        Standard CN (directory): valid if it has the single-shard weights
        file, OR the sharded index + every shard file it references (diffusers
        ``save_pretrained`` writes ``diffusion_pytorch_model.safetensors`` when
        the model fits in one shard, else
        ``diffusion_pytorch_model.safetensors.index.json`` +
        ``diffusion_pytorch_model-NNNNN-of-MMMMM.safetensors`` shards).
        LLLite (single file): valid if the file exists and is non-empty.
        """
        try:
            if p.is_dir():
                single = p / "diffusion_pytorch_model.safetensors"
                if single.exists() and single.stat().st_size > 0:
                    return True
                index_file = p / "diffusion_pytorch_model.safetensors.index.json"
                if not index_file.exists():
                    return False
                import json
                with open(index_file, "r", encoding="utf-8") as f:
                    index = json.load(f)
                shard_names = set(index.get("weight_map", {}).values())
                if not shard_names:
                    return False
                for shard_name in shard_names:
                    shard_path = p / shard_name
                    if not shard_path.exists() or shard_path.stat().st_size == 0:
                        return False
                return True
            else:
                return p.exists() and p.stat().st_size > 0
        except Exception:
            return False

    def _find_latest_controlnet_checkpoint(self) -> "Optional[Path]":
        """Return the highest-step COMPLETE ControlNet checkpoint under
        output_dir, or None.

        Standard CN saves DIRECTORY checkpoints (``{run}_controlnet_step_NNNNNN/``);
        LLLite saves single ``{run}_lllite_step_NNNNNN.safetensors`` files. Mirrors
        the naming in save_checkpoint(). Candidates are walked highest-step-first
        and the first one that passes ``_is_complete_controlnet_checkpoint`` wins,
        so a broken/partial emergency-save directory (weights-less mkdir shell)
        is transparently skipped in favor of the last good periodic checkpoint.
        """
        import re
        if self.controlnet_type == "standard":
            candidates = [p for p in self.output_dir.glob(f"{self.run_name}_controlnet_step_*") if p.is_dir()]
        else:
            candidates = list(self.output_dir.glob(f"{self.run_name}_lllite_step_*.safetensors"))
        steps = []
        for p in candidates:
            m = re.search(r"_step_(\d+)", p.stem)
            if not m:
                continue
            steps.append((int(m.group(1)), p))
        steps.sort(key=lambda t: t[0], reverse=True)
        for step, p in steps:
            if self._is_complete_controlnet_checkpoint(p):
                return p
            print(f"{self.log_prefix} Resume: skipping incomplete checkpoint "
                  f"{p.name} (step {step}, no weights found)")
        return None

    def _resume_controlnet_weights(self):
        """Load ControlNet adapter weights for resume and record the checkpoint
        path so base_trainer.train() restores global_step / optimizer / epoch.

        "latest" is the auto-detect DEFAULT (param_defaults / frontend), so a
        brand-new run legitimately has no checkpoint yet: in that case start fresh
        (leave _loaded_checkpoint_path unset -> train() falls through to "from
        scratch"), matching LoRA/full-ft. Only a specifically-named-but-missing
        checkpoint fails loud (that IS a user error worth aborting on, and avoids
        silently discarding the run the user meant to continue).
        """
        req = str(self.resume_from_checkpoint)
        is_latest = req.lower() == "latest"
        if is_latest:
            ckpt = self._find_latest_controlnet_checkpoint()
        else:
            cand = self.output_dir / req
            if not cand.exists():
                cand = Path(req)
            ckpt = cand if cand.exists() else None

        if ckpt is None:
            if is_latest:
                # First run / auto-resume with nothing to resume from -> start fresh.
                print(f"{self.log_prefix} Resume: no ControlNet checkpoint under "
                      f"{self.output_dir}; starting from scratch")
                return
            raise RuntimeError(
                f"{self.log_prefix} resume_from_checkpoint={self.resume_from_checkpoint!r} "
                f"was requested but that ControlNet checkpoint was not found under "
                f"{self.output_dir}. Aborting to avoid silently restarting from scratch."
            )

        self.load_checkpoint(str(ckpt))
        # Signal base_trainer.train() to restore step/optimizer/epoch state from
        # the matching *_state.json (it extracts the step from this path's name).
        self._loaded_checkpoint_path = str(ckpt)
        print(f"{self.log_prefix} Resume: ControlNet weights loaded from {ckpt.name}")

    def get_outpaint_planner(self) -> OutpaintControlPlanner:
        """Lazily build the deterministic crop-rect sampler for outpaint conditioning
        mode. Only meaningful when conditioning_mode == 'outpaint'; the base-trainer
        condition-load branch calls this to sample each item's known-region rect."""
        if self._outpaint_planner is None:
            seed = int(getattr(self, "seed", 0) or 0)
            self._outpaint_planner = OutpaintControlPlanner(
                seed=seed,
                min_area=self.outpaint_crop_min_area,
                max_area=self.outpaint_crop_max_area,
                edge_anchor_prob=self.outpaint_edge_anchor_prob,
                corner_anchor_prob=self.outpaint_corner_anchor_prob,
                edge_feather_min_px=self.outpaint_edge_feather_min_px,
                edge_feather_max_px=self.outpaint_edge_feather_max_px,
            )
        return self._outpaint_planner

    def setup_trainable_parameters(self) -> List[Dict]:
        """
        Collect trainable parameters from ControlNet.

        Returns:
            List of parameter groups for optimizer
        """
        return self.adapter.setup_trainable_parameters(self.controlnet)

    def save_checkpoint(self, step: int, epoch: int):
        """
        Save ControlNet checkpoint.

        Standard: saves as diffusers-compatible directory
        LLLite: saves as sd-scripts compatible .safetensors [Phase 2]

        Args:
            step: Current training step
            epoch: Current training epoch
        """
        if self.controlnet_type == "standard":
            # Directory format: {run_name}_controlnet_step_001000/
            checkpoint_path = self.output_dir / f"{self.run_name}_controlnet_step_{step:06d}"
        else:
            # LLLite: single file format
            checkpoint_path = self.output_dir / f"{self.run_name}_lllite_step_{step:06d}.safetensors"

        self.adapter.save_checkpoint(self.controlnet, step, epoch, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load ControlNet checkpoint for resume training.

        Args:
            checkpoint_path: Path to checkpoint directory or file

        Returns:
            Step number from checkpoint
        """
        print(f"{self.log_prefix} Loading checkpoint: {checkpoint_path}")
        step = self.adapter.load_checkpoint(self.controlnet, checkpoint_path)
        print(f"{self.log_prefix} Loaded checkpoint from step {step}")
        return step

    def _cleanup_old_checkpoints(self, max_step_saves_to_keep: int):
        """
        Delete old ControlNet checkpoints, keeping only the most recent N.

        base_trainer._cleanup_old_checkpoints / _list_checkpoint_entries only
        recognize FILE checkpoints matching ``*_step_*.safetensors[.index.json]``.
        Standard ControlNet saves a DIRECTORY (``{run}_controlnet_step_NNNNNN/``)
        which never matches that glob, so pruning silently never fires for it.
        LLLite saves a matching file (``{run}_lllite_step_NNNNNN.safetensors``)
        so the base glob technically finds it, but this override handles both
        uniformly and is dispatched instead (same 1-positional-arg signature as
        the base version -> train()'s inspect.signature dispatch picks this one
        for ControlNetTrainer instances).

        Note: the optimizer/state sidecars are always named
        ``{run_name}_step_{step:06d}_{optimizer.pt,state.json}`` (base ``_step_``,
        NOT ``_controlnet_step_`` / ``_lllite_step_`` — see save_optimizer_state /
        save_training_state in base_trainer.py), so the sidecar glob must use the
        plain step number, independent of the checkpoint-entry naming.
        """
        import re
        import shutil

        if max_step_saves_to_keep is None or max_step_saves_to_keep <= 0:
            return

        if self.controlnet_type == "standard":
            candidates = [p for p in self.output_dir.glob(f"{self.run_name}_controlnet_step_*") if p.is_dir()]
        else:
            candidates = [p for p in self.output_dir.glob(f"{self.run_name}_lllite_step_*.safetensors") if p.is_file()]

        def get_step(path):
            m = re.search(r"_step_(\d+)", path.stem if path.is_file() else path.name)
            return int(m.group(1)) if m else 0

        if len(candidates) <= max_step_saves_to_keep:
            return

        candidates.sort(key=get_step, reverse=True)
        to_delete = candidates[max_step_saves_to_keep:]

        for entry in to_delete:
            step_num = get_step(entry)
            print(f"{self.log_prefix} Deleting old checkpoint: {entry.name}")
            try:
                if entry.is_dir():
                    shutil.rmtree(entry, ignore_errors=True)
                else:
                    self._safe_unlink(entry)
            except Exception as e:
                print(f"{self.log_prefix} WARNING: could not delete {entry.name} ({e}); leaving it (non-fatal)")

            # Sidecars are always keyed off the base run_name + plain step number,
            # regardless of controlnet_type (see docstring above).
            optimizer_pt_path = self.output_dir / f"{self.run_name}_step_{step_num:06d}_optimizer.pt"
            state_json_path = self.output_dir / f"{self.run_name}_step_{step_num:06d}_state.json"

            if optimizer_pt_path.exists():
                print(f"{self.log_prefix} Deleting old optimizer state: {optimizer_pt_path.name}")
                self._safe_unlink(optimizer_pt_path)

            if state_json_path.exists():
                print(f"{self.log_prefix} Deleting old training state: {state_json_path.name}")
                self._safe_unlink(state_json_path)

    # ============================================================
    # Sample Generation (ControlNet-aware)
    # ============================================================

    def _load_sample_condition_image(self, condition_image_path: "Optional[str]" = None) -> "Optional[Image.Image]":
        """
        Load condition image for sample generation during training.

        Priority:
        1. Per-prompt condition_image_path argument
        2. First dataset item's reference_images[0] (fallback)

        Uses path-based caching to avoid reloading the same image across calls.

        Args:
            condition_image_path: Path to condition image (per-prompt, optional)

        Returns:
            PIL Image or None if no condition image available
        """
        from PIL import Image

        # Initialize path-based cache
        if not hasattr(self, '_condition_image_cache'):
            self._condition_image_cache = {}  # path -> PIL.Image

        # Outpaint conditioning mode has no paired condition image: its 4-channel
        # crop+mask conditioning is built from each item's OWN image at train time
        # (build_crop_mask_condition) and is not yet wired for sample generation.
        # Return None so generate_sample falls back to the base model (no ControlNet)
        # instead of feeding a 3-channel RGB image to a 4-channel ControlNet.
        if getattr(self, "_is_outpaint_mode", False):
            return None

        # Option 1: Per-prompt condition image path
        if condition_image_path:
            # Check cache
            if condition_image_path in self._condition_image_cache:
                return self._condition_image_cache[condition_image_path]

            p = Path(condition_image_path)
            if p.exists():
                try:
                    img = flatten_to_rgb(Image.open(str(p)))
                    print(f"{self.log_prefix} [Sample] Loaded condition image from per-prompt path: {p}")
                    self._condition_image_cache[condition_image_path] = img
                    return img
                except Exception as e:
                    print(f"{self.log_prefix} [Sample] Failed to load condition image from {p}: {e}")
            else:
                print(f"{self.log_prefix} [Sample] Condition image path not found: {p}")

        # Option 2: First dataset item's reference image (fallback)
        fallback_key = "__dataset_fallback__"
        if fallback_key in self._condition_image_cache:
            return self._condition_image_cache[fallback_key]

        datasets = getattr(self, '_training_datasets', None)
        if datasets:
            for ds in datasets:
                # _training_datasets entries are TrainRunnerDataset objects (with an
                # .items attribute), not dicts -- support both.
                items = ds.get("items", []) if isinstance(ds, dict) else getattr(ds, "items", [])
                for item in items:
                    ref_images = item.get("reference_images", [])
                    if ref_images:
                        ref_path = Path(ref_images[0])
                        if ref_path.exists():
                            try:
                                img = flatten_to_rgb(Image.open(str(ref_path)))
                                print(f"{self.log_prefix} [Sample] Loaded condition image from dataset: {ref_path}")
                                self._condition_image_cache[fallback_key] = img
                                return img
                            except Exception as e:
                                print(f"{self.log_prefix} [Sample] Failed to load {ref_path}: {e}")

        print(f"{self.log_prefix} [Sample] WARNING: No condition image found for sample generation")
        # Cache None for fallback to avoid repeated warnings
        self._condition_image_cache[fallback_key] = None
        return None

    def generate_sample(
        self,
        prompt: str,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        seed: int = -1,
        current_step: int = 0,
        schedule_type: str = "uniform",
        condition_image_path: "Optional[str]" = None,
        reference_image_path: "Optional[str]" = None,
        negative_prompt: str = "",
    ) -> "Image.Image":
        """
        Generate sample image during ControlNet training.

        Overrides BaseTrainer.generate_sample() to apply the trained ControlNet
        during sample generation, allowing visual verification of training progress.

        Args:
            condition_image_path: Per-prompt condition image path (optional).
                If not provided, falls back to dataset's first reference image.
            reference_image_path: img2img-style reference image path. ControlNet
                sampling has no use for this itself (it uses condition_image_path
                for the control signal), so it is only forwarded to base
                generate_sample() on the no-condition-image fallback path, where
                the base SD/SDXL implementation decides how to use it.

        Standard ControlNet: Sets pipeline.controlnet and passes controlnet_images
        LLLite: Applies patches to UNet before sampling, removes after
        """
        # Load condition image (per-prompt path or fallback to dataset)
        loaded_condition = self._load_sample_condition_image(condition_image_path)

        # No condition image available: fall back to base (no ControlNet)
        if loaded_condition is None:
            print(f"{self.log_prefix} [Sample] No condition image, falling back to base generate_sample()")
            return super().generate_sample(
                prompt=prompt, height=height, width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale, seed=seed,
                negative_prompt=negative_prompt,
                current_step=current_step, schedule_type=schedule_type,
                reference_image_path=reference_image_path,
            )

        # Dispatch to type-specific implementation
        if self.controlnet_type == "standard":
            return self._generate_sample_standard(
                prompt=prompt, height=height, width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale, seed=seed,
                negative_prompt=negative_prompt,
                current_step=current_step, schedule_type=schedule_type,
                condition_image=loaded_condition,
            )
        elif self.controlnet_type == "lllite":
            return self._generate_sample_lllite(
                prompt=prompt, height=height, width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale, seed=seed,
                negative_prompt=negative_prompt,
                current_step=current_step, schedule_type=schedule_type,
                condition_image=loaded_condition,
            )
        else:
            # Unknown type: fall back to base
            return super().generate_sample(
                prompt=prompt, height=height, width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale, seed=seed,
                negative_prompt=negative_prompt,
                current_step=current_step, schedule_type=schedule_type,
                reference_image_path=reference_image_path,
            )

    def _generate_sample_standard(
        self,
        prompt: str,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        seed: int = -1,
        current_step: int = 0,
        schedule_type: str = "uniform",
        condition_image: "Optional[Image.Image]" = None,
        negative_prompt: str = "",
    ) -> "Image.Image":
        """
        Generate sample with Standard ControlNet (ControlNetModel).

        Sets pipeline.controlnet and passes controlnet_images to custom_sampling_loop().

        Args:
            condition_image: Pre-loaded PIL Image for conditioning.
        """
        from PIL import Image
        import random

        print(f"{self.log_prefix} [Sample] Generating with Standard ControlNet: {prompt[:50]}...")

        from core.inference.custom_sampling import custom_sampling_loop
        from core.inference.schedulers import get_scheduler

        # Resize condition image to sample dimensions
        condition_image = condition_image.resize((width, height), Image.LANCZOS)

        # Set models to eval mode
        self.unet.eval()
        self.vae.eval()
        self.text_encoder.eval()
        if self.text_encoder_2 is not None:
            self.text_encoder_2.eval()
        self.controlnet.eval()

        try:
            # ========================================
            # STEP 1: Create Temporary Pipeline with ControlNet
            # ========================================
            class TempPipeline:
                def __init__(self, unet, vae, text_encoder, text_encoder_2,
                             scheduler, tokenizer, tokenizer_2, controlnet):
                    self.unet = unet
                    self.vae = vae
                    self.text_encoder = text_encoder
                    self.text_encoder_2 = text_encoder_2
                    self.scheduler = scheduler
                    self.tokenizer = tokenizer
                    self.tokenizer_2 = tokenizer_2
                    self.controlnet = controlnet
                    self.vae_scale_factor = 8
                    self.image_processor = None

            # Map schedule_type
            schedule_type_mapped = schedule_type
            if schedule_type == "sgm_uniform":
                schedule_type_mapped = "uniform"

            class SchedulerContainer:
                def __init__(self, scheduler):
                    self.scheduler = scheduler

            scheduler_container = SchedulerContainer(self.original_scheduler)
            scheduler = get_scheduler(
                pipeline=scheduler_container,
                sampler="euler",
                schedule_type=schedule_type_mapped
            )

            pipeline = TempPipeline(
                unet=self.unet,
                vae=self.vae,
                text_encoder=self.text_encoder,
                text_encoder_2=getattr(self, 'text_encoder_2', None),
                scheduler=scheduler,
                tokenizer=self.tokenizer,
                tokenizer_2=getattr(self, 'tokenizer_2', None),
                controlnet=self.controlnet,
            )

            # ========================================
            # STEP 2: Text Encoding
            # ========================================
            self.move_text_encoder_to_gpu()

            if self.is_sdxl:
                prompt_embeds, pooled_prompt_embeds = self.encode_prompt(prompt, requires_grad=False)
                negative_prompt_embeds, negative_pooled_prompt_embeds = self.encode_prompt(negative_prompt, requires_grad=False)
            else:
                prompt_embeds = self.encode_prompt(prompt, requires_grad=False)
                negative_prompt_embeds = self.encode_prompt(negative_prompt, requires_grad=False)
                pooled_prompt_embeds = None
                negative_pooled_prompt_embeds = None

            # Pad negative embeddings to match positive (prompt chunking)
            if prompt_embeds.shape[1] != negative_prompt_embeds.shape[1]:
                seq_len_diff = prompt_embeds.shape[1] - negative_prompt_embeds.shape[1]
                padding = torch.zeros(
                    (negative_prompt_embeds.shape[0], seq_len_diff, negative_prompt_embeds.shape[2]),
                    dtype=negative_prompt_embeds.dtype,
                    device=negative_prompt_embeds.device
                )
                negative_prompt_embeds = torch.cat([negative_prompt_embeds, padding], dim=1)

            self.move_text_encoder_to_cpu()
            torch.cuda.empty_cache()

            # ========================================
            # STEP 3: Create Generator
            # ========================================
            if seed < 0:
                actual_seed = random.randint(0, 2**32 - 1)
            else:
                actual_seed = seed

            generator = torch.Generator(device=self.device).manual_seed(actual_seed)

            # ========================================
            # STEP 4: Call custom_sampling_loop with ControlNet
            # ========================================
            self.move_main_model_to_gpu()
            self.move_vae_to_gpu()

            is_v_prediction = pipeline.scheduler.config.get("prediction_type") == "v_prediction"
            guidance_rescale = 0.7 if is_v_prediction else 0.0

            log_verbose(f"{self.log_prefix} [Sample] Standard ControlNet active, condition_scale=1.0")

            with torch.autocast(device_type=self.device.type, dtype=self.training_dtype):
                image = custom_sampling_loop(
                    pipeline=pipeline,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    guidance_rescale=guidance_rescale,
                    width=width,
                    height=height,
                    generator=generator,
                    ancestral_generator=None,
                    latents=None,
                    prompt_embeds_callback=None,
                    progress_callback=None,
                    step_callback=None,
                    developer_mode=False,
                    cfg_schedule_type="constant",
                    cfg_schedule_min=1.0,
                    cfg_schedule_max=None,
                    cfg_schedule_power=2.0,
                    cfg_rescale_snr_alpha=0.0,
                    dynamic_threshold_percentile=0.0,
                    dynamic_threshold_mimic_scale=1.0,
                    nag_enable=False,
                    nag_scale=5.0,
                    nag_tau=3.5,
                    nag_alpha=0.25,
                    nag_sigma_end=0.0,
                    nag_negative_prompt_embeds=None,
                    nag_negative_pooled_prompt_embeds=None,
                    attention_type="normal",
                    # ControlNet parameters
                    controlnet_images=[condition_image],
                    controlnet_conditioning_scale=1.0,
                    control_guidance_start=0.0,
                    control_guidance_end=1.0,
                )

                self.move_main_model_to_cpu()
                self.move_vae_to_cpu()
                torch.cuda.empty_cache()

                log_verbose(f"{self.log_prefix} [Sample] Standard ControlNet sample generated (seed: {actual_seed})")
                return image

        except Exception as e:
            print(f"{self.log_prefix} [Sample] ERROR: {type(e).__name__}: {str(e)}")
            print(f"{self.log_prefix} [Sample] Sample generation failed - training will continue")

            from PIL import Image
            placeholder = Image.new("RGB", (width, height), color=(255, 255, 255))
            return placeholder

        finally:
            # Restore training mode
            self.unet.train()
            self.vae.train()
            self.text_encoder.train()
            if self.text_encoder_2 is not None:
                self.text_encoder_2.train()
            self.controlnet.train()
            self.move_main_model_to_gpu()

    def _generate_sample_lllite(
        self,
        prompt: str,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        seed: int = -1,
        current_step: int = 0,
        schedule_type: str = "uniform",
        condition_image: "Optional[Image.Image]" = None,
        negative_prompt: str = "",
    ) -> "Image.Image":
        """
        Generate sample with LLLite ControlNet.

        Applies LLLite patches to UNet before sampling, removes after.

        Args:
            condition_image: Pre-loaded PIL Image for conditioning.
        """
        from PIL import Image
        import random
        import torchvision.transforms.functional as TF

        print(f"{self.log_prefix} [Sample] Generating with LLLite ControlNet: {prompt[:50]}...")

        from core.inference.custom_sampling import custom_sampling_loop
        from core.inference.schedulers import get_scheduler

        # Resize condition image to sample dimensions
        condition_image = condition_image.resize((width, height), Image.LANCZOS)

        # Set models to eval mode
        self.unet.eval()
        self.vae.eval()
        self.text_encoder.eval()
        if self.text_encoder_2 is not None:
            self.text_encoder_2.eval()
        self.controlnet.eval()

        lllite_patched = False

        try:
            # ========================================
            # STEP 1: Create Temporary Pipeline (no controlnet attr)
            # ========================================
            if self.is_sdxl:
                class TempPipeline:
                    def __init__(self, unet, vae, text_encoder, text_encoder_2,
                                 scheduler, tokenizer, tokenizer_2):
                        self.unet = unet
                        self.vae = vae
                        self.text_encoder = text_encoder
                        self.text_encoder_2 = text_encoder_2
                        self.scheduler = scheduler
                        self.tokenizer = tokenizer
                        self.tokenizer_2 = tokenizer_2
                        self.vae_scale_factor = 8
                        self.image_processor = None
            else:
                class TempPipeline:
                    def __init__(self, unet, vae, text_encoder, scheduler, tokenizer):
                        self.unet = unet
                        self.vae = vae
                        self.text_encoder = text_encoder
                        self.scheduler = scheduler
                        self.tokenizer = tokenizer
                        self.vae_scale_factor = 8
                        self.image_processor = None

            schedule_type_mapped = schedule_type
            if schedule_type == "sgm_uniform":
                schedule_type_mapped = "uniform"

            class SchedulerContainer:
                def __init__(self, scheduler):
                    self.scheduler = scheduler

            scheduler_container = SchedulerContainer(self.original_scheduler)
            scheduler = get_scheduler(
                pipeline=scheduler_container,
                sampler="euler",
                schedule_type=schedule_type_mapped
            )

            if self.is_sdxl:
                pipeline = TempPipeline(
                    unet=self.unet,
                    vae=self.vae,
                    text_encoder=self.text_encoder,
                    text_encoder_2=self.text_encoder_2,
                    scheduler=scheduler,
                    tokenizer=self.tokenizer,
                    tokenizer_2=self.tokenizer_2,
                )
            else:
                pipeline = TempPipeline(
                    unet=self.unet,
                    vae=self.vae,
                    text_encoder=self.text_encoder,
                    scheduler=scheduler,
                    tokenizer=self.tokenizer,
                )

            # ========================================
            # STEP 2: Text Encoding
            # ========================================
            self.move_text_encoder_to_gpu()

            if self.is_sdxl:
                prompt_embeds, pooled_prompt_embeds = self.encode_prompt(prompt, requires_grad=False)
                negative_prompt_embeds, negative_pooled_prompt_embeds = self.encode_prompt(negative_prompt, requires_grad=False)
            else:
                prompt_embeds = self.encode_prompt(prompt, requires_grad=False)
                negative_prompt_embeds = self.encode_prompt(negative_prompt, requires_grad=False)
                pooled_prompt_embeds = None
                negative_pooled_prompt_embeds = None

            # Pad negative embeddings to match positive (prompt chunking)
            if prompt_embeds.shape[1] != negative_prompt_embeds.shape[1]:
                seq_len_diff = prompt_embeds.shape[1] - negative_prompt_embeds.shape[1]
                padding = torch.zeros(
                    (negative_prompt_embeds.shape[0], seq_len_diff, negative_prompt_embeds.shape[2]),
                    dtype=negative_prompt_embeds.dtype,
                    device=negative_prompt_embeds.device
                )
                negative_prompt_embeds = torch.cat([negative_prompt_embeds, padding], dim=1)

            self.move_text_encoder_to_cpu()
            torch.cuda.empty_cache()

            # ========================================
            # STEP 3: Create Generator
            # ========================================
            if seed < 0:
                actual_seed = random.randint(0, 2**32 - 1)
            else:
                actual_seed = seed

            generator = torch.Generator(device=self.device).manual_seed(actual_seed)

            # ========================================
            # STEP 4: Apply LLLite patches and call custom_sampling_loop
            # ========================================
            self.move_main_model_to_gpu()
            self.move_vae_to_gpu()

            # Prepare condition tensor [1, 3, H, W] in [0, 1] range
            cond_tensor = TF.to_tensor(condition_image).unsqueeze(0).to(
                device=self.device, dtype=self.training_dtype
            )

            # Apply LLLite patches to UNet
            self.controlnet.apply_patches(self.unet, cond_tensor)
            lllite_patched = True
            log_verbose(f"{self.log_prefix} [Sample] LLLite patches applied to UNet")

            is_v_prediction = pipeline.scheduler.config.get("prediction_type") == "v_prediction"
            guidance_rescale = 0.7 if is_v_prediction else 0.0

            with torch.autocast(device_type=self.device.type, dtype=self.training_dtype):
                image = custom_sampling_loop(
                    pipeline=pipeline,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    guidance_rescale=guidance_rescale,
                    width=width,
                    height=height,
                    generator=generator,
                    ancestral_generator=None,
                    latents=None,
                    prompt_embeds_callback=None,
                    progress_callback=None,
                    step_callback=None,
                    developer_mode=False,
                    cfg_schedule_type="constant",
                    cfg_schedule_min=1.0,
                    cfg_schedule_max=None,
                    cfg_schedule_power=2.0,
                    cfg_rescale_snr_alpha=0.0,
                    dynamic_threshold_percentile=0.0,
                    dynamic_threshold_mimic_scale=1.0,
                    nag_enable=False,
                    nag_scale=5.0,
                    nag_tau=3.5,
                    nag_alpha=0.25,
                    nag_sigma_end=0.0,
                    nag_negative_prompt_embeds=None,
                    nag_negative_pooled_prompt_embeds=None,
                    attention_type="normal",
                    # No controlnet params for LLLite (patches already applied)
                )

                # Remove LLLite patches
                self.controlnet.remove_patches(self.unet)
                lllite_patched = False

                self.move_main_model_to_cpu()
                self.move_vae_to_cpu()
                torch.cuda.empty_cache()

                log_verbose(f"{self.log_prefix} [Sample] LLLite ControlNet sample generated (seed: {actual_seed})")
                return image

        except Exception as e:
            print(f"{self.log_prefix} [Sample] ERROR: {type(e).__name__}: {str(e)}")
            print(f"{self.log_prefix} [Sample] Sample generation failed - training will continue")

            from PIL import Image
            placeholder = Image.new("RGB", (width, height), color=(255, 255, 255))
            return placeholder

        finally:
            # Remove LLLite patches if still applied
            if lllite_patched and hasattr(self.controlnet, '_is_patched') and self.controlnet._is_patched:
                self.controlnet.remove_patches(self.unet)

            # Restore training mode
            self.unet.train()
            self.vae.train()
            self.text_encoder.train()
            if self.text_encoder_2 is not None:
                self.text_encoder_2.train()
            self.controlnet.train()
            self.move_main_model_to_gpu()
