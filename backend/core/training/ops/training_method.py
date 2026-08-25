"""Which training method a run actually uses.

Stdlib-only on purpose: every ``ops/*`` module and the trainers can import this
without an import cycle (``full_parameter_trainer`` -> ``base_trainer`` ->
``ops/*``).
"""

from __future__ import annotations

from typing import Any

_FULL_FINETUNE_METHOD = "full"
# Both spellings of the same request: the API/YAML value is "full_finetune",
# while the ops layer's own vocabulary (and this repo's refusal messages) says
# "full".
_FULL_FINETUNE_ALIASES = frozenset({"full", "full_finetune"})


def resolve_training_method(trainer: Any) -> str:
    """``"full"`` for a full-parameter run, otherwise the declared method.

    The TRAINER is the authoritative channel: ``training_method`` is not emitted
    into the train config section, so ``config['training_method']`` reads
    ``None`` for every run, full fine-tunes included. It is still honoured as a
    secondary channel in case it is ever wired.
    """
    if getattr(trainer, "trains_base_weights", False):
        return _FULL_FINETUNE_METHOD
    # Fallback for a trainer built without that attribute. Matched by NAME
    # because importing FullParameterTrainer would cycle via base_trainer.
    if any(cls.__name__ == "FullParameterTrainer" for cls in type(trainer).__mro__):
        return _FULL_FINETUNE_METHOD
    config = getattr(trainer, "config", None) or {}
    declared = str(config.get("training_method") or "lora").strip().lower()
    return _FULL_FINETUNE_METHOD if declared in _FULL_FINETUNE_ALIASES else declared


def is_full_finetune(trainer: Any) -> bool:
    """True when base weights are trained.

    Gates that ask "is the base frozen?" must key on this rather than on
    ``method == "lora"``: ReLoRA and ControlNet runs also freeze the base, so an
    equality test against "lora" would refuse them the moment the config channel
    ever carries their name.
    """
    return resolve_training_method(trainer) == _FULL_FINETUNE_METHOD


def trains_denoiser_weights(trainer: Any) -> bool:
    """True when the DiT/U-Net weights themselves are updated.

    This, not ``is_full_finetune``, is what a "is the denoiser frozen?" gate
    must ask: a full FT with ``train_unet=False`` (text-encoder-only) leaves the
    denoiser frozen, so FP8 base quantisation and H2D-only block swap remain
    legitimate. Every full-parameter adapter gates its ``requires_grad_(True)``
    on ``train_unet``, and ``FullParameterTrainer.__init__`` sets the flag
    before ``super().__init__()``, so it is readable while components load.
    """
    return is_full_finetune(trainer) and bool(getattr(trainer, "train_unet", True))
