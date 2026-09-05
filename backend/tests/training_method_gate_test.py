"""The five gates that key on the training method.

Each site has a NEGATIVE CONTROL that re-installs the shipped predicate (which
read a config key no run carries, so it answered "lora" for every run) and
records the wrong outcome it produced, plus an assertion that the LoRA path is
unchanged.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import yaml
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.training.ops import anima_ops, flux2_ops, lens_ops, ltx2_ops, sd_sdxl_ops
from core.training.ops.training_method import (
    is_full_finetune,
    resolve_training_method,
    trains_denoiser_weights,
)
from core.training.training_config import TrainingConfigGenerator


# The shipped predicate, expressed against the same call site the fix uses: it
# reads a key the config never carries, so it answers False for a full FT too.
def _shipped_is_full_finetune(trainer) -> bool:
    config = getattr(trainer, "config", None) or {}
    return str(config.get("training_method", "lora") or "lora").strip().lower() != "lora"


def _shipped_resolve_training_method(trainer) -> str:
    """The same shipped bug expressed for the method-name channel the VAE-swap
    gate reads: the config never carries the key, so every run reads "lora"."""
    config = getattr(trainer, "config", None) or {}
    return str(config.get("training_method", "lora") or "lora").strip().lower()


class FullParameterTrainer:  # name-matched on purpose: the MRO walk keys on it
    pass


# ---------------------------------------------------------------------------
# The resolver
# ---------------------------------------------------------------------------

class _RealSubclass(FullParameterTrainer):
    """What train_runner constructs: the config carries no training_method."""
    config: dict = {}


class _LoRAish:
    config: dict = {}


def test_resolver_uses_the_mro_channel_not_only_the_config_channel():
    # The production channel: subclass identity, with an empty config.
    assert resolve_training_method(_RealSubclass()) == "full"
    assert is_full_finetune(_RealSubclass()) is True

    assert resolve_training_method(_LoRAish()) == "lora"
    assert is_full_finetune(_LoRAish()) is False
    assert resolve_training_method(SimpleNamespace()) == "lora"
    assert is_full_finetune(SimpleNamespace(config={})) is False

    # Secondary channel, both spellings.
    for spelling in ("full", "full_finetune", "FULL_FINETUNE", " full "):
        assert is_full_finetune(SimpleNamespace(config={"training_method": spelling}))
    # Other adapter modes freeze the base too and must not read as full FT.
    for method in ("lora", "relora", "controlnet"):
        assert is_full_finetune(SimpleNamespace(config={"training_method": method})) is False


def test_resolver_sees_the_real_full_parameter_trainer():
    """Imports the production class, so a rename of it fails here rather than
    silently turning all five gates off with a green suite."""
    from core.training.full_parameter_trainer import FullParameterTrainer as Real

    trainer = Real.__new__(Real)          # no __init__: no model load
    trainer.config = {}
    assert is_full_finetune(trainer) is True            # name channel
    trainer.trains_base_weights = True                  # attribute channel
    assert is_full_finetune(trainer) is True
    assert "trains_base_weights = True" in _full_parameter_trainer_source()


def _full_parameter_trainer_source() -> str:
    import inspect

    from core.training.full_parameter_trainer import FullParameterTrainer as Real

    return inspect.getsource(Real.__init__)


def test_train_unet_false_leaves_the_denoiser_frozen():
    """A text-encoder-only full FT does not train the denoiser, so the
    frozen-denoiser gates must not fire."""
    class _TEOnly(FullParameterTrainer):
        config = {}
        train_unet = False

    assert is_full_finetune(_TEOnly()) is True
    assert trains_denoiser_weights(_TEOnly()) is False
    assert trains_denoiser_weights(_RealSubclass()) is True   # train_unet defaults True
    assert trains_denoiser_weights(_LoRAish()) is False


def test_subclass_wins_over_a_contradictory_config_key():
    """Pin: if training_method is ever wired in contradicting the subclass, the
    subclass still decides. A change to config-wins breaks this."""
    class _Contradictory(FullParameterTrainer):
        config = {"training_method": "lora"}

    assert resolve_training_method(_Contradictory()) == "full"
    assert is_full_finetune(_Contradictory()) is True


def test_train_section_still_omits_training_method():
    """Tripwire for the defect's cause.

    If this fails, ``training_method`` has been wired into the train config;
    check every ``is_full_finetune`` call site still agrees with the trainer
    subclass before deleting this test.
    """
    generated = TrainingConfigGenerator.generate_full_finetune_config(
        {"training_method": "full_finetune", "total_steps": 10},
        run_name="pin",
        base_model_path="model.safetensors",
        output_dir="out",
        dataset_path="data",
    )

    def _keys(node):
        if isinstance(node, dict):
            for key, value in node.items():
                yield key
                yield from _keys(value)
        elif isinstance(node, list):
            for item in node:
                yield from _keys(item)

    # Anywhere in the document, not just the train section: a wiring into
    # process[0] or the top-level config must trip this too.
    assert "training_method" not in set(_keys(yaml.safe_load(generated)))


# ---------------------------------------------------------------------------
# Site 1: fp8_base_dtype quantising the base a full FT is about to train
# ---------------------------------------------------------------------------

class _Stub(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)


def _fp8_trainer(cls, *, fp8="e4m3", **extra):
    trainer = cls()
    trainer.log_prefix = "[T]"
    trainer.model_path = "model.safetensors"
    trainer.weight_dtype = torch.float32
    trainer.training_dtype = torch.float32
    trainer.vae_dtype = torch.float32
    trainer.dtype = torch.float32
    trainer.device = torch.device("cpu")
    trainer.config = {"fp8_base_dtype": fp8}
    trainer.gradient_checkpointing = False
    trainer.blocks_to_swap = 0
    trainer.use_flash_attention = False
    for key, value in extra.items():
        setattr(trainer, key, value)
    return trainer


def _anima_components():
    return {
        "transformer": _Stub(), "vae": _Stub(), "text_encoder": _Stub(),
        "tokenizer": object(), "t5_tokenizer": object(), "scheduler": object(),
    }


def _lens_components():
    return {
        "transformer": _Stub(), "vae": _Stub(), "text_encoder": _Stub(),
        "tokenizer": object(), "scheduler": object(),
    }


def _ltx2_components():
    return {
        "pipeline": SimpleNamespace(), "transformer": _Stub(), "vae": _Stub(),
        "audio_vae": _Stub(), "text_encoder": _Stub(), "tokenizer": object(),
        "connectors": _Stub(), "vocoder": None, "scheduler": object(),
    }


def _run_fp8_load(arch, trainer):
    """Run the arch's real load_components with a stubbed loader; return the
    fp8 quantiser spy."""
    module, loader_target, components = {
        "anima": (anima_ops, "core.model_loader.ModelLoader.load_anima_from_files",
                  _anima_components()),
        "lens": (lens_ops, "core.models.lens.lens_loader.load_lens_components",
                 _lens_components()),
        "ltx2": (ltx2_ops, "core.model_loader.ModelLoader.load_ltx2_from_path",
                 _ltx2_components()),
    }[arch]
    with patch(loader_target, return_value=components), \
            patch("core.vram_optimization._anima_quantize_fp8",
                  side_effect=lambda m, *a, **k: m) as quantize:
        module.load_components(trainer)
    return quantize


@pytest.mark.parametrize("arch", ["anima", "lens", "ltx2"])
def test_fp8_base_is_skipped_for_a_full_finetune(arch, capsys):
    trainer = _fp8_trainer(FullParameterTrainer)
    quantize = _run_fp8_load(arch, trainer)
    quantize.assert_not_called()
    # The skip must be visible, not silent.
    out = capsys.readouterr().out
    assert "WARNING: fp8_base_dtype=e4m3 requires a frozen" in out
    assert "train_unet=True" in out


@pytest.mark.parametrize("arch", ["anima", "lens", "ltx2"])
def test_negative_control_shipped_predicate_fp8s_the_full_finetune_base(arch):
    """Records the shipped outcome: the base a full FT is about to train is
    handed to the fp8 quantiser."""
    module = {"anima": anima_ops, "lens": lens_ops, "ltx2": ltx2_ops}[arch]
    trainer = _fp8_trainer(FullParameterTrainer)
    with patch.object(module, "trains_denoiser_weights", _shipped_is_full_finetune):
        quantize = _run_fp8_load(arch, trainer)
    assert quantize.call_count == 1
    assert quantize.call_args.args[1] == "e4m3"


@pytest.mark.parametrize("arch", ["anima", "lens", "ltx2"])
def test_text_encoder_only_full_finetune_keeps_the_fp8_base(arch):
    """train_unet=False leaves the denoiser frozen, so the run keeps the FP8
    base it is entitled to."""
    trainer = _fp8_trainer(FullParameterTrainer, train_unet=False)
    quantize = _run_fp8_load(arch, trainer)
    assert quantize.call_count == 1
    assert quantize.call_args.args[1] == "e4m3"


@pytest.mark.parametrize("arch", ["anima", "lens", "ltx2"])
def test_lora_still_gets_the_fp8_base(arch):
    module = {"anima": anima_ops, "lens": lens_ops, "ltx2": ltx2_ops}[arch]
    fixed = _run_fp8_load(arch, _fp8_trainer(_LoRAish))
    trainer = _fp8_trainer(_LoRAish)
    with patch.object(module, "trains_denoiser_weights", _shipped_is_full_finetune):
        shipped = _run_fp8_load(arch, trainer)
    assert fixed.call_count == shipped.call_count == 1
    assert fixed.call_args.args[1] == shipped.call_args.args[1] == "e4m3"


# ---------------------------------------------------------------------------
# Site 2: SDXL custom VAE / TE migration, refused for LoRA only
# ---------------------------------------------------------------------------

class _PastTheGate(Exception):
    """Raised by the stubbed migration to prove the gate let the run through."""


def _sdxl_trainer(cls, **config):
    trainer = _fp8_trainer(cls, fp8=None)
    trainer.config = {"sdxl_vae_type": "none", "sdxl_te_type": "none", **config}
    trainer.debug_vram = False
    return trainer


def _run_sdxl_load(trainer):
    pipeline = SimpleNamespace(
        vae=_Stub(), text_encoder=_Stub(), tokenizer=object(), unet=_Stub(),
        scheduler=object(), text_encoder_2=_Stub(), tokenizer_2=object(),
    )
    with patch.object(sd_sdxl_ops.StableDiffusionXLPipeline, "from_single_file",
                      return_value=pipeline), \
            patch("core.models.common.vae_source.resolve_vae_source",
                  side_effect=_PastTheGate), \
            patch("core.training.base_trainer._vramdiag", lambda *a, **k: None):
        sd_sdxl_ops.load_components(trainer)


def test_sdxl_custom_arch_is_refused_for_lora():
    trainer = _sdxl_trainer(_LoRAish, vae_swap_source="registry:flux1")
    with pytest.raises(ValueError, match="requires training_method='full_finetune'"):
        _run_sdxl_load(trainer)


def test_legacy_sdxl_vae_type_is_refused_for_lora_through_the_same_gate():
    trainer = _sdxl_trainer(_LoRAish, sdxl_vae_type="flux1")
    with pytest.raises(ValueError, match="requires training_method='full_finetune'"):
        _run_sdxl_load(trainer)


def test_sdxl_custom_arch_passes_the_gate_for_a_full_finetune():
    trainer = _sdxl_trainer(FullParameterTrainer, vae_swap_source="registry:flux1")
    # Reaching the migration (which the full-FT adapter trains and saves) is the
    # assertion: the gate no longer refuses the only method that can do it.
    with pytest.raises(_PastTheGate):
        _run_sdxl_load(trainer)


def test_negative_control_shipped_predicate_refuses_the_full_finetune():
    """Records the shipped outcome: full FT is refused and told to switch to
    the method it is already using."""
    trainer = _sdxl_trainer(FullParameterTrainer, vae_swap_source="registry:flux1")
    with patch("core.training.ops.sd_sdxl_ops.resolve_training_method",
               _shipped_resolve_training_method):
        with pytest.raises(ValueError) as refusal:
            _run_sdxl_load(trainer)
    assert "Switch to Full Fine-tune" in str(refusal.value)


def test_sdxl_stock_arch_is_untouched_for_both_methods():
    for cls in (_LoRAish, FullParameterTrainer):
        trainer = _sdxl_trainer(cls)
        _run_sdxl_load(trainer)  # no gate, no migration
        assert trainer.sdxl_vae_type == "sdxl"
        assert trainer.sdxl_te_type == "none"


# ---------------------------------------------------------------------------
# Site 3: FLUX.2 H2D-only block swap requires a frozen base
# ---------------------------------------------------------------------------

class _Flux2Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True


def _flux2_trainer(cls):
    trainer = cls()
    trainer.log_prefix = "[T]"
    trainer.config = {}
    trainer.block_swap_h2d_only = True
    trainer.block_swap_ring_size = 2
    trainer.gradient_checkpointing = True
    trainer.transformer = _Flux2Transformer()
    return trainer


def test_flux2_block_swap_refuses_a_full_finetune():
    with pytest.raises(ValueError, match="requires a frozen transformer"):
        flux2_ops.block_swap_h2d_args(_flux2_trainer(FullParameterTrainer))


def test_negative_control_shipped_predicate_admits_the_full_finetune():
    """Records the shipped outcome: the refusal never fires, so the run reaches
    the offloader, whose lazy Full-FT detect falls back to the standard swap
    path that Gate 1 of this same function refuses as non-functional."""
    trainer = _flux2_trainer(FullParameterTrainer)
    with patch.object(flux2_ops, "trains_denoiser_weights", _shipped_is_full_finetune):
        assert flux2_ops.block_swap_h2d_args(trainer) == {
            "h2d_only": True, "ring_size": 2,
        }


def test_flux2_block_swap_admits_a_text_encoder_only_full_finetune():
    """train_unet=False keeps the transformer frozen (flux2_adapter gates its
    unfreeze on it), so H2D-only swap stays legitimate."""
    trainer = _flux2_trainer(FullParameterTrainer)
    trainer.train_unet = False
    assert flux2_ops.block_swap_h2d_args(trainer) == {"h2d_only": True, "ring_size": 2}


def test_flux2_block_swap_still_admits_lora():
    assert flux2_ops.block_swap_h2d_args(_flux2_trainer(_LoRAish)) == {
        "h2d_only": True, "ring_size": 2,
    }
