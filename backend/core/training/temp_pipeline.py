"""Reusable pipeline-like wrapper around the trainer's in-memory models.

The trainer holds the U-Net / VAE / text encoders directly (no diffusers
pipeline).  Production generation code expects a pipeline object with
``unet`` / ``vae`` / ``scheduler`` attributes.  This wrapper presents
that surface so we can reuse ``custom_sampling_loop`` family of
functions, ``lora_manager.load_loras``, and the ControlNet pipeline
construction path.

Used by ``TrainingPreviewGenerator`` to run txt2img / img2img / inpaint
inference using the in-training model weights.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


class TempPipeline:
    """Pipeline-shaped facade over trainer's in-memory components.

    Carries enough surface to be passed to:
      - ``custom_sampling_loop`` / ``custom_img2img_sampling_loop`` /
        ``custom_inpaint_sampling_loop`` (need ``unet``, ``vae``,
        ``scheduler``, ``text_encoder``[_2], ``tokenizer``[_2],
        ``vae_scale_factor``);
      - ``StableDiffusion(XL)ControlNetPipeline(**components)``
        constructor (need ``.components`` property);
      - LoRA loading via diffusers' ``load_lora_weights`` interface
        (delegated to the underlying ``unet`` / ``text_encoder`` if
        they have peft adapters attached, otherwise a manual fallback).
    """

    def __init__(
        self,
        *,
        unet: Any,
        vae: Any,
        text_encoder: Any,
        scheduler: Any,
        tokenizer: Any,
        text_encoder_2: Optional[Any] = None,
        tokenizer_2: Optional[Any] = None,
        is_sdxl: bool = False,
        vae_scale_factor: int = 8,
    ) -> None:
        self.unet = unet
        self.vae = vae
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.is_sdxl = is_sdxl
        # custom_sampling_loop reads these directly
        self.vae_scale_factor = vae_scale_factor
        self.image_processor = None   # not needed by custom_sampling_loop

    # ------------------------------------------------------------------
    # diffusers-pipeline-like surface
    # ------------------------------------------------------------------

    @property
    def components(self) -> Dict[str, Any]:
        """Mimics ``DiffusionPipeline.components`` so ControlNet pipeline
        constructors can be invoked via ``**self.components``."""
        if self.is_sdxl:
            return {
                "vae":            self.vae,
                "text_encoder":   self.text_encoder,
                "text_encoder_2": self.text_encoder_2,
                "tokenizer":      self.tokenizer,
                "tokenizer_2":    self.tokenizer_2,
                "unet":           self.unet,
                "scheduler":      self.scheduler,
            }
        return {
            "vae":               self.vae,
            "text_encoder":      self.text_encoder,
            "tokenizer":         self.tokenizer,
            "unet":              self.unet,
            "scheduler":         self.scheduler,
            "safety_checker":    None,
            "feature_extractor": None,
        }

    # ------------------------------------------------------------------
    # LoRA loading (delegate to peft adapters on the underlying modules)
    # ------------------------------------------------------------------

    def load_lora_weights(
        self,
        pretrained_model_name_or_path_or_dict,
        adapter_name: Optional[str] = None,
        **kwargs,
    ):
        """Forward to the underlying U-Net's peft adapter loader, if
        available.  This is a best-effort path so ``lora_manager`` can
        attach additional LoRAs on top of an in-training LoRA.

        Falls back to a no-op with a warning if the underlying modules
        don't expose ``load_adapter``.
        """
        # The diffusers mixin would normally split the state_dict between
        # unet and text_encoder and call adapter loading on each.  For
        # the preview path, the most common request is "stack a saved
        # diffusers-format LoRA on top of the training LoRA"; we lean on
        # the loaders the lora_manager already routes through.  When
        # neither path applies, we fail loudly so the calling code can
        # report a clear error rather than silently dropping LoRAs.
        loaded = False
        for mod_name, mod in [("unet", self.unet),
                              ("text_encoder", self.text_encoder),
                              ("text_encoder_2", self.text_encoder_2)]:
            if mod is None:
                continue
            if hasattr(mod, "load_adapter"):
                try:
                    mod.load_adapter(
                        pretrained_model_name_or_path_or_dict,
                        adapter_name=adapter_name,
                        **kwargs,
                    )
                    loaded = True
                except Exception as e:   # noqa: BLE001
                    print(f"[TempPipeline] load_adapter on {mod_name} failed: {e}")
        if not loaded:
            raise NotImplementedError(
                "TempPipeline.load_lora_weights: neither unet nor "
                "text_encoders expose peft 'load_adapter'.  Stacking "
                "additional LoRAs on top of the in-training model is "
                "only supported when the trainer wraps modules with "
                "PeftModel.  Use --bypass-preview-loras to skip."
            )

    def set_adapters(self, adapter_names, adapter_weights=None):
        """Forward adapter activation to the underlying modules."""
        for mod in (self.unet, self.text_encoder, self.text_encoder_2):
            if mod is None:
                continue
            if hasattr(mod, "set_adapter"):
                try:
                    mod.set_adapter(adapter_names)
                except Exception:   # noqa: BLE001
                    pass
            # PEFT-style scaling
            if hasattr(mod, "set_adapters"):
                try:
                    mod.set_adapters(adapter_names, adapter_weights)
                except Exception:   # noqa: BLE001
                    pass

    def delete_adapters(self, adapter_names):
        """Remove a previously-loaded adapter from the underlying modules."""
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]
        for mod in (self.unet, self.text_encoder, self.text_encoder_2):
            if mod is None:
                continue
            for n in adapter_names:
                if hasattr(mod, "delete_adapter"):
                    try:
                        mod.delete_adapter(n)
                    except Exception:   # noqa: BLE001
                        pass

    def unload_lora_weights(self):
        """Delete every adapter we know about — best-effort cleanup."""
        for mod in (self.unet, self.text_encoder, self.text_encoder_2):
            if mod is None:
                continue
            adapters = getattr(mod, "peft_config", None)
            if adapters:
                for name in list(adapters.keys()):
                    if hasattr(mod, "delete_adapter"):
                        try:
                            mod.delete_adapter(name)
                        except Exception:   # noqa: BLE001
                            pass


def build_temp_pipeline_for_trainer(trainer, scheduler) -> TempPipeline:
    """Convenience constructor: build the wrapper from a BaseTrainer.

    The trainer carries ``unet``, ``vae``, ``text_encoder``, ``tokenizer``,
    and optionally ``text_encoder_2`` / ``tokenizer_2`` (SDXL).
    """
    is_sdxl = getattr(trainer, "is_sdxl", False) and trainer.text_encoder_2 is not None
    return TempPipeline(
        unet=trainer.unet,
        vae=trainer.vae,
        text_encoder=trainer.text_encoder,
        text_encoder_2=trainer.text_encoder_2 if is_sdxl else None,
        scheduler=scheduler,
        tokenizer=trainer.tokenizer,
        tokenizer_2=trainer.tokenizer_2 if is_sdxl else None,
        is_sdxl=is_sdxl,
    )
