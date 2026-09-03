"""Reusable pipeline-like wrapper around the trainer's in-memory models.

The trainer holds the U-Net / VAE / text encoders directly (no diffusers
pipeline).  Production generation code expects a pipeline object with
``unet`` / ``vae`` / ``scheduler`` attributes.  This wrapper presents
that surface so we can reuse ``custom_sampling_loop`` family of
functions, ``lora_manager.load_loras``, and the ControlNet pipeline
construction path.

Used by ``TrainingPreviewGenerator`` to run txt2img / img2img / inpaint
inference using the in-training model weights.

LoRA stacking on a preview goes through the REAL diffusers loader mixin (the
SD / SDXL subclasses below inherit it), because ``lora_manager.load_loras``
needs kohya-key conversion, PEFT injection and a read-back count of installed
branch containers -- none of which a hand-written shim reproduces.  PEFT
refuses to wrap the trainer's own ``LoRALinearLayer``, so those wrappers are
spliced out for the duration of the load and spliced back around the PEFT
layer afterwards; see :func:`training_lora_detour`.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple


class LoraStackingUnsupported(RuntimeError):
    """Extra LoRAs cannot be stacked on this trainer's models."""


class TempPipeline:
    """Pipeline-shaped facade over trainer's in-memory components.

    Carries enough surface to be passed to:
      - ``custom_sampling_loop`` / ``custom_img2img_sampling_loop`` /
        ``custom_inpaint_sampling_loop`` (need ``unet``, ``vae``,
        ``scheduler``, ``text_encoder``[_2], ``tokenizer``[_2],
        ``vae_scale_factor``);
      - ``StableDiffusion(XL)ControlNetPipeline(**components)``
        constructor (need ``.components`` property);
      - ``lora_manager.load_loras``, which needs the diffusers loader mixin
        that only the SD / SDXL subclasses below carry.
    """

    # Read by diffusers' offload probe (_func_optionally_disable_offloading)
    # before any LoRA load. The trainer moves modules itself; no accelerate hooks.
    hf_device_map = None

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
        arch_name: str = "unknown",
    ) -> None:
        self.unet = unet
        self.vae = vae
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.is_sdxl = is_sdxl
        self.arch_name = arch_name
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

    def lora_components(self) -> List[Any]:
        """The modules a stacked LoRA may touch, in load order."""
        return [m for m in (self.unet, self.text_encoder, self.text_encoder_2)
                if m is not None]

    def assert_can_stack_loras(self) -> None:
        """Raise before any load if this facade cannot apply extra LoRAs."""


class NoLoraStackingTempPipeline(TempPipeline):
    """Facade for an architecture the diffusers SD LoRA loader cannot serve."""

    def _refusal(self) -> "LoraStackingUnsupported":
        return LoraStackingUnsupported(
            f"Stacking additional LoRAs on a training preview is implemented for "
            f"SD1.5 and SDXL only. This run trains '{self.arch_name}' "
            f"({type(self.unet).__name__}), whose LoRA key format the diffusers "
            f"Stable Diffusion loader does not read, so loading one here would "
            f"install nothing. Remove the extra LoRAs from the preview request."
        )

    def assert_can_stack_loras(self) -> None:
        # Raised by the caller, not from inside load_lora_weights: lora_manager
        # rewrites anything thrown there into a generic "could not be applied".
        raise self._refusal()

    def load_lora_weights(self, *args, **kwargs):
        raise self._refusal()


def _sd_temp_pipeline_classes():
    """Built lazily so importing this module does not pull in diffusers for
    callers that only want the sampling facade."""
    from diffusers.loaders.lora_pipeline import (
        StableDiffusionLoraLoaderMixin,
        StableDiffusionXLLoraLoaderMixin,
    )

    class SDTempPipeline(TempPipeline, StableDiffusionLoraLoaderMixin):
        pass

    class SDXLTempPipeline(TempPipeline, StableDiffusionXLLoraLoaderMixin):
        pass

    return SDTempPipeline, SDXLTempPipeline


# ---------------------------------------------------------------------------
# Training-LoRA detour
#
# The trainer replaces target Linears with its own ``LoRALinearLayer``, and PEFT
# refuses to wrap one ("Target module ... is not supported"), so a stacked LoRA
# could never load over a LoRA run. Splicing the wrapper out for the load and
# back around the PEFT layer afterwards composes them in the right order:
#     wrapper(x) = peft(x) + training_branch(x) = base(x) + extra(x) + training(x)
# and leaves the wrapper object itself in place, which matters because the
# optimizer holds its parameters.
# ---------------------------------------------------------------------------

# (parent module, attribute name or list index, the trainer's wrapper)
TrainingLoraSite = Tuple[Any, Any, Any]


def _get_child(parent: Any, attr: Any) -> Any:
    return parent[attr] if isinstance(attr, int) else getattr(parent, attr)


def _set_child(parent: Any, attr: Any, module: Any) -> None:
    if isinstance(attr, int):
        parent[attr] = module
    else:
        setattr(parent, attr, module)


def collect_training_lora_sites(components: List[Any]) -> List[TrainingLoraSite]:
    """Every adapter wrapper root in the tree, with its parent.

    Roots only: the walk stops at a wrapper, so a composite's own branches are
    never collected as sites of their own. Splicing a branch out would put the
    shared base into the branch slot -- the stale-module splice the composite
    exists to make impossible.
    """
    from core.adapters import is_adapter_covered, named_modules_outside_adapters

    sites: List[TrainingLoraSite] = []
    for component in components:
        if component is None or not hasattr(component, "named_modules"):
            continue
        for _name, parent in named_modules_outside_adapters(component):
            for attr, child in list(getattr(parent, "_modules", {}).items()):
                if is_adapter_covered(child):
                    sites.append((parent, int(attr) if attr.isdigit() else attr, child))
    return sites


@contextmanager
def training_lora_detour(sites: List[TrainingLoraSite]):
    """Expose each wrapper's inner module in its place, then put the wrapper
    back around whatever now sits there.

    Restores per site under its own guard: one failure must not strand the
    remaining wrappers outside the model, which would silently disable part of
    the in-training adapter for the rest of the run.
    """
    spliced: List[TrainingLoraSite] = []
    try:
        for parent, attr, wrapper in sites:
            spliced.append((parent, attr, wrapper))
            _set_child(parent, attr, wrapper.original_module)
        yield
    finally:
        errors = []
        for parent, attr, wrapper in reversed(spliced):
            try:
                current = _get_child(parent, attr)
                if current is not wrapper:
                    wrapper.original_module = current
                    _set_child(parent, attr, wrapper)
            except Exception as e:   # noqa: BLE001
                errors.append(f"{type(wrapper).__name__}.{attr}: {e}")
        if errors:
            raise RuntimeError(
                f"Restoring the in-training LoRA wrappers failed for "
                f"{len(errors)} site(s): {errors[:3]}"
            )


def components_with_peft_adapters(components: List[Any]) -> List[str]:
    """Components that already carry PEFT adapters before a preview loads any.

    Nothing in the trainer installs PEFT (LoRA training uses
    ``LoRALinearLayer``), so a non-empty result is state this module did not
    create and cannot promise to restore.
    """
    named = []
    for component in components:
        config = getattr(component, "peft_config", None)
        if config:
            named.append(f"{type(component).__name__}({sorted(config)})")
    return named


def build_temp_pipeline_for_trainer(trainer, scheduler) -> TempPipeline:
    """Convenience constructor: build the wrapper from a BaseTrainer.

    The trainer carries ``unet``, ``vae``, ``text_encoder``, ``tokenizer``,
    and optionally ``text_encoder_2`` / ``tokenizer_2`` (SDXL).  The class
    returned decides whether extra LoRAs can be stacked: only a diffusers
    ``UNet2DConditionModel`` (SD1.5 / SDXL) is served by the SD loader mixin.
    """
    from diffusers import UNet2DConditionModel

    is_sdxl = getattr(trainer, "is_sdxl", False) and trainer.text_encoder_2 is not None
    arch_name = getattr(getattr(trainer, "arch", None), "name", None) or "unknown"

    if isinstance(trainer.unet, UNet2DConditionModel):
        sd_cls, sdxl_cls = _sd_temp_pipeline_classes()
        cls = sdxl_cls if is_sdxl else sd_cls
    else:
        cls = NoLoraStackingTempPipeline

    return cls(
        unet=trainer.unet,
        vae=trainer.vae,
        text_encoder=trainer.text_encoder,
        text_encoder_2=trainer.text_encoder_2 if is_sdxl else None,
        scheduler=scheduler,
        tokenizer=trainer.tokenizer,
        tokenizer_2=trainer.tokenizer_2 if is_sdxl else None,
        is_sdxl=is_sdxl,
        arch_name=arch_name,
    )
