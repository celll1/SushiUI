from types import SimpleNamespace

import pytest
import torch

from core.training.ops import (
    acestep_ops,
    anima_ops,
    flux2_ops,
    ideogram4_ops,
    krea2_ops,
    lens_ops,
    ltx2_ops,
    minimax_h3_ops,
    minit2i_ops,
    zimage_ops,
)
from api.arch_capabilities import ARCH_UNSUPPORTED


class FakeConductor:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.registered = False
        FakeConductor.calls.append(self)

    def register_hooks(self):
        self.registered = True


def _layers():
    return torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(3)])


def _trainer(arch):
    flags = {
        "is_anima": False, "is_lens": False, "is_ideogram4": False,
        "is_minit2i": False, "is_krea2": False, "is_ltx2": False,
        "is_minimax_h3": False, "is_acestep": False, "is_zimage": False,
    }
    flags[f"is_{arch}"] = True
    trainer = SimpleNamespace(
        **flags, blocks_to_swap=2, block_swap_ring_size=3,
        gradient_checkpointing=True, layer_offload_conductor=None,
        device=torch.device("cuda"), use_pinned_memory=True,
        log_prefix="[test]", trains_base_weights=True, train_unet=True,
    )
    if arch == "anima":
        trainer.transformer = SimpleNamespace(blocks=_layers())
    elif arch in {"lens", "krea2", "ltx2", "minimax_h3"}:
        trainer.transformer = SimpleNamespace(transformer_blocks=_layers())
    elif arch == "ideogram4":
        trainer.transformer = SimpleNamespace(layers=_layers())
        trainer.transformer_uncond = None
        trainer.ideogram4_train_uncond = False
    elif arch == "minit2i":
        trainer.transformer = SimpleNamespace(
            model=SimpleNamespace(net=SimpleNamespace(double_blocks=_layers())))
    elif arch == "acestep":
        trainer.transformer = SimpleNamespace(decoder=SimpleNamespace(layers=_layers()))
    elif arch == "zimage":
        trainer.transformer_original = SimpleNamespace(layers=_layers())
    return trainer


@pytest.mark.parametrize("arch,setup", [
    ("anima", anima_ops.setup_block_swap),
    ("lens", lens_ops.setup_block_swap),
    ("ideogram4", ideogram4_ops.setup_block_swap),
    ("minit2i", minit2i_ops.setup_block_swap),
    ("krea2", krea2_ops.setup_block_swap),
    ("ltx2", ltx2_ops.setup_block_swap),
    ("minimax_h3", minimax_h3_ops.setup_block_swap),
    ("acestep", acestep_ops.setup_block_swap),
    ("zimage", zimage_ops.setup_block_swap),
])
def test_arch_routes_share_mutable_conductor(monkeypatch, arch, setup):
    import core.memory_management
    monkeypatch.setattr(core.memory_management, "LayerOffloadConductor", FakeConductor)
    FakeConductor.calls.clear()
    trainer = _trainer(arch)
    setup(trainer)
    assert len(FakeConductor.calls) == 1
    assert FakeConductor.calls[0].registered
    assert FakeConductor.calls[0].kwargs["ring_size"] == 3
    assert trainer.layer_offload_conductor is FakeConductor.calls[0]


@pytest.mark.parametrize("arch,setup", [
    ("anima", anima_ops.setup_block_swap),
    ("lens", lens_ops.setup_block_swap),
    ("ideogram4", ideogram4_ops.setup_block_swap),
    ("minit2i", minit2i_ops.setup_block_swap),
    ("krea2", krea2_ops.setup_block_swap),
    ("ltx2", ltx2_ops.setup_block_swap),
    ("minimax_h3", minimax_h3_ops.setup_block_swap),
    ("acestep", acestep_ops.setup_block_swap),
    ("zimage", zimage_ops.setup_block_swap),
])
def test_arch_routes_refuse_mutable_swap_without_checkpointing(arch, setup):
    trainer = _trainer(arch)
    trainer.gradient_checkpointing = False
    with pytest.raises(ValueError, match="gradient_checkpointing=True"):
        setup(trainer)


def test_flux_full_ft_uses_the_same_mutable_conductor(monkeypatch):
    import core.memory_management
    monkeypatch.setattr(core.memory_management, "LayerOffloadConductor", FakeConductor)
    FakeConductor.calls.clear()
    trainer = _trainer("anima")
    trainer.transformer = SimpleNamespace(
        transformer_blocks=_layers(), single_transformer_blocks=_layers())
    trainer.flux2_block_offloader = None
    flux2_ops.setup_mutable_block_swap(trainer)
    assert len(FakeConductor.calls) == 1
    assert len(FakeConductor.calls[0].kwargs["layers"]) == 6
    assert FakeConductor.calls[0].kwargs["ring_size"] == 3


def test_minimax_generation_exposes_the_shared_ring_size():
    assert "block_swap_ring_size" not in ARCH_UNSUPPORTED["minimax_h3"]
    assert "block_swap_h2d_only" in ARCH_UNSUPPORTED["minimax_h3"]
    assert "block_swap_pinned_memory" in ARCH_UNSUPPORTED["minimax_h3"]
