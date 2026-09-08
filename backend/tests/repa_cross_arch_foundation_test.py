"""REPA's architecture-neutral foundation: tap resolution, adapter plumbing, refusals.

REPA used to be MiniT2I-shaped in four places: the tap dims read
``transformer.mmjit_config``, the tap module was a hardcoded three-level unwrap,
the sidecar path lived in the MiniT2I adapter, and the projector's optimizer
group and sidecar save existed ONLY in that adapter. The last one is the reason
this file exists: with the group in one adapter out of thirteen, enabling REPA
anywhere else would have trained against a random frozen head and added a noise
term to the loss, with no exception and no warning.

What is pinned here:
  (a) the MiniT2I resolver returns exactly what the shipped expressions returned;
  (b) every adapter inherits the projector's param group and sidecar save;
  (c) an architecture that cannot run REPA refuses, with a reason;
  (d) TREAD / DiT-BlockSkip conflicts with the tap depth are detected.

Run:
    venv/Scripts/python.exe -m pytest backend/tests/repa_cross_arch_foundation_test.py -v
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.training import repa as repa_module
from core.training.adapters.base_adapter import (BaseFullParameterAdapter,
                                                 BaseLoRAAdapter,
                                                 LORA_COMPONENT_UNET)
from core.training.arch import ARCH_REGISTRY
from core.training.base_trainer import BaseTrainer

HIDDEN = 16
DEPTH = 28
PATCH = 16
ENC_DIM = 8


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class _Net(nn.Module):
    """Stands in for the MM-JiT net: only the two tap attributes matter."""

    def __init__(self):
        super().__init__()
        self._repa_tap_depth = None
        self._repa_tap_out = None


class _Transformer:
    def __init__(self):
        self.mmjit_config = SimpleNamespace(hidden_size=HIDDEN, depth_double=DEPTH,
                                            patch_size=PATCH)
        self.model = SimpleNamespace(net=_Net())


def _minit2i_trainer(**config):
    cfg = {"repa_enable": True, "repa_tagger_model_dir": "unused-because-stubbed"}
    cfg.update(config)
    return SimpleNamespace(
        config=cfg,
        arch=ARCH_REGISTRY["minit2i"](),
        transformer=_Transformer(),
        device=torch.device("cpu"),
        training_dtype=torch.float32,
        model_path="",
        log_prefix="[test]",
    )


def _stub_encoder(monkeypatch, *, calls=None):
    def _fake(source, **kwargs):
        if calls is not None:
            calls.append(source)
        return nn.Identity(), ENC_DIM, 224

    monkeypatch.setattr(repa_module, "load_repa_encoder", _fake)


# ---------------------------------------------------------------------------
# (a) MiniT2I's numbers are unchanged
# ---------------------------------------------------------------------------

def test_minit2i_tap_matches_the_shipped_expressions():
    """The resolver returns what ``base_trainer`` hardcoded before this change."""
    trainer = _minit2i_trainer()
    tap = trainer.arch.repa_tap(trainer)

    # The three shipped expressions, verbatim.
    assert tap.hidden_size == int(trainer.transformer.mmjit_config.hidden_size)
    assert tap.depth == int(trainer.transformer.mmjit_config.depth_double)
    assert tap.module is trainer.transformer.model.net


def test_setup_repa_arms_the_same_tap_depth(monkeypatch):
    trainer = _minit2i_trainer()
    _stub_encoder(monkeypatch)

    BaseTrainer._setup_repa(trainer)

    # Shipped auto depth: depth_double // 3, clamped into [0, depth-1].
    assert trainer.repa_align_depth == DEPTH // 3 == 9
    net = trainer.transformer.model.net
    assert net._repa_tap_depth == 9
    assert trainer._repa_tap_module is net
    assert trainer.repa_projector.net[0].in_features == HIDDEN
    assert trainer.repa_projector.net[-1].out_features == ENC_DIM


def test_explicit_align_depth_is_clamped_not_rejected(monkeypatch):
    trainer = _minit2i_trainer(repa_align_depth=999)
    _stub_encoder(monkeypatch)

    BaseTrainer._setup_repa(trainer)

    assert trainer.repa_align_depth == DEPTH - 1


def test_take_repa_tap_reads_and_clears():
    trainer = _minit2i_trainer()
    net = trainer.transformer.model.net
    trainer._repa_tap_module = net
    stashed = torch.zeros(1, 4, HIDDEN)
    net._repa_tap_out = stashed

    assert repa_module.take_repa_tap(trainer) is stashed
    assert net._repa_tap_out is None
    assert repa_module.take_repa_tap(trainer) is None


def test_sidecar_path_replaces_only_a_trailing_suffix():
    assert repa_module.repa_sidecar_path("a/b.safetensors") == "a/b.repa.safetensors"
    # A directory component containing the suffix must not be rewritten.
    assert (repa_module.repa_sidecar_path("a.safetensors/c")
            == "a.safetensors/c.repa.safetensors")


# ---------------------------------------------------------------------------
# (b) every adapter carries the projector
# ---------------------------------------------------------------------------

def _shipped_adapters(base):
    return sorted((cls for cls in base.__subclasses__()
                   if cls.__module__.startswith("core.training.adapters")),
                  key=lambda c: c.__name__)


def test_every_lora_adapter_inherits_the_projector_group_and_sidecar():
    import core.training.adapters  # noqa: F401  (registers every subclass)

    adapters = _shipped_adapters(BaseLoRAAdapter)
    assert len(adapters) == len(ARCH_REGISTRY) == 13
    for cls in adapters:
        assert cls.setup_trainable_parameters is BaseLoRAAdapter.setup_trainable_parameters, cls
        assert cls.save_checkpoint is BaseLoRAAdapter.save_checkpoint, cls


def test_every_full_parameter_adapter_inherits_the_projector_group_and_sidecar():
    import core.training.adapters  # noqa: F401

    adapters = _shipped_adapters(BaseFullParameterAdapter)
    # 12: every architecture except MiniMax-H3, which trains LoRA only.
    assert len(adapters) == 12
    for cls in adapters:
        assert cls.setup_trainable_parameters is BaseFullParameterAdapter.setup_trainable_parameters, cls
        assert cls.save_checkpoint is BaseFullParameterAdapter.save_checkpoint, cls


def test_every_lora_adapter_class_in_the_registry_is_one_of_them():
    """The registry's own answer, so a new arch cannot bring an unplumbed adapter."""
    for name, handler_cls in ARCH_REGISTRY.items():
        cls = handler_cls().lora_adapter_class()
        assert cls.setup_trainable_parameters is BaseLoRAAdapter.setup_trainable_parameters, name


class _LoRAAdapter(BaseLoRAAdapter):
    def apply_lora_to_unet(self, lora_layers):
        layer = self.build_branch(self.trainer.transformer.to_q, "to_q")
        self.trainer.transformer.to_q = layer
        self.register_lora_layer(lora_layers, "to_q", layer, LORA_COMPONENT_UNET)
        return 1

    def apply_lora_to_text_encoders(self, lora_layers):
        return 0

    def arch_param_groups(self, lora_layers):
        return self.component_param_groups(lora_layers, {LORA_COMPONENT_UNET: lambda: 1e-4})

    def checkpoint_metadata(self, lora_layers, step, epoch):
        return {"model_type": "test", "step": str(step), "epoch": str(epoch)}


class _FullAdapter(BaseFullParameterAdapter):
    def prepare_models_for_training(self):
        pass

    def arch_param_groups(self):
        return [{"params": [self.trainer.transformer.to_q.weight], "lr": 1e-4,
                 "name": "unet", "component": "unet"}]

    def write_checkpoint(self, step, epoch, output_path):
        # MiniT2I's shape: the adapter, not the caller, resolves a directory or a
        # suffixless stem into the file it actually writes.
        output_path = Path(output_path)
        if output_path.is_dir():
            output_path = output_path / f"step_{step}.safetensors"
        elif not str(output_path).endswith(".safetensors"):
            output_path = Path(str(output_path) + ".safetensors")
        output_path.write_bytes(b"")
        return output_path


def _repa_trainer(projector):
    model = nn.Module()
    model.to_q = nn.Linear(4, 4, bias=False)
    return SimpleNamespace(
        transformer=model, config={}, adapter_algorithm="lora",
        weight_decompose=False, adapter_config={},
        unet_lr=2e-4, learning_rate=2e-4,
        repa_enable=True, repa_projector=projector, repa_proj_lr_factor=0.5,
    )


def _projector():
    return repa_module.RepaProjector(HIDDEN, ENC_DIM)


def test_projector_group_is_appended_last_with_the_shipped_lr():
    projector = _projector()
    trainer = _repa_trainer(projector)
    adapter = _LoRAAdapter(trainer, 2, 2, torch.float32)
    layers = {}
    adapter.apply_lora_to_unet(layers)

    groups = adapter.setup_trainable_parameters(layers)

    assert [g["name"] for g in groups] == ["unet", "repa_projector"]
    tail = groups[-1]
    assert tail["component"] == "repa_projector"
    # The shipped MiniT2I block: unet_lr * repa_proj_lr_factor.
    assert tail["lr"] == pytest.approx(2e-4 * 0.5)
    assert len(tail["params"]) == len(list(projector.parameters()))


def test_full_parameter_adapter_appends_the_same_group():
    projector = _projector()
    trainer = _repa_trainer(projector)

    groups = _FullAdapter(trainer).setup_trainable_parameters()

    assert [g["name"] for g in groups] == ["unet", "repa_projector"]
    assert groups[-1]["lr"] == pytest.approx(2e-4 * 0.5)


def test_no_projector_group_when_repa_is_off():
    trainer = _repa_trainer(None)
    trainer.repa_enable = False
    adapter = _LoRAAdapter(trainer, 2, 2, torch.float32)
    layers = {}
    adapter.apply_lora_to_unet(layers)

    assert [g["name"] for g in adapter.setup_trainable_parameters(layers)] == ["unet"]


def test_lora_save_writes_the_sidecar_next_to_the_checkpoint(tmp_path):
    trainer = _repa_trainer(_projector())
    adapter = _LoRAAdapter(trainer, 2, 2, torch.float32)
    layers = {}
    adapter.apply_lora_to_unet(layers)
    out = tmp_path / "run_step_000010.safetensors"

    adapter.save_checkpoint(layers, 10, 1, out)

    assert (tmp_path / "run_step_000010.repa.safetensors").is_file()


def test_full_save_pairs_the_sidecar_with_the_resolved_path(tmp_path):
    trainer = _repa_trainer(_projector())
    # Handed a DIRECTORY: the sidecar must pair with the file write_checkpoint
    # resolved inside it, not with the argument.
    _FullAdapter(trainer).save_checkpoint(3, 0, tmp_path)

    assert (tmp_path / "step_3.repa.safetensors").is_file()
    assert not (tmp_path.parent / (tmp_path.name + ".repa.safetensors")).exists()


def test_full_save_refuses_when_the_written_path_is_unknown(tmp_path):
    class _NoPath(_FullAdapter):
        def write_checkpoint(self, step, epoch, output_path):
            return None

    with pytest.raises(ValueError, match="write_checkpoint returned no path"):
        _NoPath(_repa_trainer(_projector())).save_checkpoint(1, 0, tmp_path / "x")


def test_optimizer_setup_refuses_a_projector_outside_every_group():
    """The backstop for a path that does not go through the base adapters."""
    trainer = _repa_trainer(_projector())
    trainer.setup_trainable_parameters = lambda: [
        {"params": [trainer.transformer.to_q.weight], "lr": 1e-4, "name": "unet"}]

    with pytest.raises(ValueError, match="no optimizer group"):
        BaseTrainer.setup_optimizer(trainer)


# ---------------------------------------------------------------------------
# (c) refusal surface
# ---------------------------------------------------------------------------

REFUSAL_MARKERS = {
    "acestep": "1-D time axis",
    "ltx2": "EVERY frame",
    "minimax_h3": "CONDITION frames",
}


@pytest.mark.parametrize("arch_name", sorted(ARCH_REGISTRY))
def test_only_minit2i_answers_with_a_tap(arch_name):
    handler = ARCH_REGISTRY[arch_name]()
    if arch_name == "minit2i":
        trainer = _minit2i_trainer()
        assert handler.repa_tap(trainer).depth == DEPTH
        return

    with pytest.raises(ValueError) as excinfo:
        handler.repa_tap(SimpleNamespace())
    message = str(excinfo.value)
    assert arch_name in message
    assert "repa_enable is not supported" in message
    marker = REFUSAL_MARKERS.get(arch_name)
    if marker is not None:
        assert marker in message
    else:
        assert "no REPA tap at this stage" in message


def test_setup_repa_refuses_before_the_encoder_is_loaded(monkeypatch):
    calls = []
    _stub_encoder(monkeypatch, calls=calls)
    trainer = _minit2i_trainer()
    trainer.arch = ARCH_REGISTRY["sdxl"]()

    with pytest.raises(ValueError, match="sdxl"):
        BaseTrainer._setup_repa(trainer)
    assert calls == []


def test_setup_repa_is_inert_when_disabled():
    trainer = _minit2i_trainer(repa_enable=False)
    trainer.arch = ARCH_REGISTRY["acestep"]()

    BaseTrainer._setup_repa(trainer)

    assert trainer.repa_enable is False
    assert trainer._repa_tap_module is None


# ---------------------------------------------------------------------------
# (d) depth-feature conflicts
# ---------------------------------------------------------------------------

def test_tread_routed_span_conflicts_with_the_tap():
    config = {"tread_enable": True, "tread_start_block": 2, "tread_end_block": 26}

    # Anima is the architecture that consumes both, and its 28 blocks put the
    # auto tap at 9 -- inside the shipped TREAD span [2, 26).
    with pytest.raises(ValueError, match="TREAD routed span"):
        repa_module.assert_repa_depth_compatible(config, 9, DEPTH)

    # Outside the span both ends are fine (end is exclusive).
    repa_module.assert_repa_depth_compatible(config, 1, DEPTH)
    repa_module.assert_repa_depth_compatible(config, 26, DEPTH)
    # And nothing is checked when TREAD is off.
    repa_module.assert_repa_depth_compatible({"tread_start_block": 2,
                                              "tread_end_block": 26}, 9, DEPTH)


def test_blockskip_skipped_span_conflicts_with_the_tap():
    config = {"blockskip_enable": True, "blockskip_front": 4, "blockskip_back": 4}

    for depth in (0, 3, 24, 27):
        with pytest.raises(ValueError, match="BlockSkip"):
            repa_module.assert_repa_depth_compatible(config, depth, DEPTH)

    # The trainable middle span [4, 24) is accepted, including the default tap.
    for depth in (4, 9, 23):
        repa_module.assert_repa_depth_compatible(config, depth, DEPTH)


def test_setup_repa_refuses_a_conflicting_tap_depth(monkeypatch):
    """Same trainer, but with the architecture declared as a consumer."""
    _stub_encoder(monkeypatch)
    trainer = _minit2i_trainer(tread_enable=True, tread_start_block=2,
                               tread_end_block=26)
    monkeypatch.setattr(type(trainer.arch), "consumes_block_loop_features", True,
                        raising=False)

    with pytest.raises(ValueError, match="TREAD routed span"):
        BaseTrainer._setup_repa(trainer)


def test_the_conflict_check_is_skipped_where_the_arch_ignores_those_features(monkeypatch):
    """MiniT2I reads neither config, so refusing on them would reject a run that
    trained fine before the check existed."""
    from core.training.arch.minit2i import MiniT2IArchHandler
    from core.training.arch.anima import AnimaArchHandler

    assert MiniT2IArchHandler.consumes_block_loop_features is False
    assert AnimaArchHandler.consumes_block_loop_features is True

    trainer = _minit2i_trainer(tread_enable=True,
                               tread_start_block=2, tread_end_block=26)
    _stub_encoder(monkeypatch)

    # Reaches the projector rather than raising: the tap depth would collide,
    # but nothing on this architecture acts on tread_config.
    BaseTrainer._setup_repa(trainer)
    assert trainer.repa_align_depth == DEPTH // 3
