"""Z-Image: trainer save -> fresh-generation load round trip, on CPU in ~2s.

Drives the REAL ``ZImageLoRAAdapter`` (injection + ``save_checkpoint``) over a
1-layer/1-head Z-Image transformer, then the REAL ``ZImageMixin`` loader on a
freshly built model. The transformer is the production class, not a stub: the
adapter and the loader both select targets by the ``ZImageAttention`` class
name, so a stub would test the stub's naming rather than Z-Image's.

The Phase-0 defect this pins: training wrote ``lora_transformer_<flattened>``
keys while generation searched ``transformer.<dotted>`` keys, so a self-trained
LoRA matched zero targets.

Common composite algebra is covered in
``adapter_lycoris_roundtrip_cheap_test.py``. This file retains Z-Image's
production target enumeration, key conversion, forward round trip, and refusal
atomicity.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/zimage_lora_roundtrip_cheap_test.py -v
"""

import pytest
import torch
from safetensors.torch import save_file

from lora_roundtrip_common import (
    LoRALinearLayer, lora_delta, randomise_lora_layers,
    warning_codes, warning_probe,
)

from core.adapters import (  # noqa: E402
    AdapterIncompatible, CompositeAdapterLayer,
)
from core.pipeline_backends.zimage import ZImageMixin  # noqa: E402
from core.training.adapters.zimage_adapter import ZImageLoRAAdapter  # noqa: E402

# head_dim is fixed at sum(ROPE_AXES_DIMS)=128, so shrink heads/layers, not dim.
_TINY = dict(in_channels=4, dim=128, n_layers=1, n_refiner_layers=1,
             n_heads=1, n_kv_heads=1, cap_feat_dim=16)
RANK = 4
ALPHA = 2  # != rank on purpose: a regression to the rank fallback shows as scale 1.0
SCALE = ALPHA / RANK
STRENGTH = 0.75


def build_model():
    from core.models.zimage_transformer import ZImageTransformer2DModel
    torch.manual_seed(0)
    return ZImageTransformer2DModel(**_TINY)


class _StubTrainer:
    def __init__(self, transformer):
        self.transformer = transformer
        self.unet_lr = 1e-4


class _Backend(ZImageMixin):
    def __init__(self, transformer):
        self.zimage_components = {"transformer": transformer}


def wrapped_paths(model):
    """Target paths a GENERATION load covers, i.e. the composite roots."""
    return {name for name, module in model.named_modules()
            if isinstance(module, CompositeAdapterLayer)}


def lora_layer_paths(model):
    """Paths the TRAINER wrapped -- it still installs plain wrappers."""
    return {name for name, module in model.named_modules()
            if isinstance(module, LoRALinearLayer)}


def train_and_save(tmp_path, name="zimage.safetensors", seed=1234):
    """Returns (checkpoint path, the trained model, its wrapped target paths)."""
    model = build_model()
    adapter = ZImageLoRAAdapter(_StubTrainer(model), lora_rank=RANK, lora_alpha=ALPHA,
                                lora_dtype=torch.float32)
    layers = {}
    count = adapter.apply_lora_to_unet(layers)
    assert count == len(layers) > 0
    randomise_lora_layers(layers, seed=seed)
    out = tmp_path / name
    adapter.save_checkpoint(layers, step=10, epoch=1, output_path=out)
    return str(out), model, lora_layer_paths(model)


def file_branch_tensors(path, target):
    """``(down, up)`` straight out of the checkpoint, for the analytic sum."""
    from safetensors.torch import load_file
    saved = load_file(path)
    stem = "lora_transformer_" + target.replace(".", "_")
    return saved[f"{stem}.lora_down.weight"], saved[f"{stem}.lora_up.weight"]


@pytest.fixture
def warnings_seen(monkeypatch):
    return warning_probe(monkeypatch)


@pytest.fixture
def resolve_verbatim(monkeypatch):
    """The mixin resolves through LoRAManager; these files live in tmp_path."""
    from core.extensions import lora_manager as lm
    import os

    monkeypatch.setattr(lm.lora_manager, "_resolve_lora_path",
                        lambda p: p if os.path.exists(str(p)) else None)


def test_zimage_generation_wraps_exactly_the_targets_the_trainer_wrapped(tmp_path):
    path, _trained, trained_paths = train_and_save(tmp_path)

    model = build_model()
    backend = _Backend(model)
    backend._load_lora_zimage([{"path": path, "strength": STRENGTH}])

    # Set EQUALITY: the regression was 0 matched targets, and a partial match
    # is just as wrong and much quieter.
    assert wrapped_paths(model) == trained_paths
    assert backend._zimage_lora_wrapped_modules == trained_paths


def test_zimage_wrapped_forward_is_base_plus_scaled_branch(tmp_path):
    path, _trained, trained_paths = train_and_save(tmp_path)

    model = build_model()
    _Backend(model)._load_lora_zimage([{"path": path, "strength": STRENGTH}])

    modules = dict(model.named_modules())
    for target in sorted(trained_paths):
        wrapper = modules[target]
        down, up = file_branch_tensors(path, target)
        x = torch.randn(3, wrapper.original_module.in_features)
        base = wrapper.original_module(x)
        expected = base + lora_delta(down, up, x, ALPHA, RANK, STRENGTH)
        assert torch.allclose(wrapper(x), expected, atol=1e-5), target
        assert not torch.allclose(wrapper(x), base, atol=1e-5), f"{target}: branch is inert"


def test_zimage_missing_file_refuses_and_warns(tmp_path, warnings_seen, monkeypatch):
    from core.extensions import lora_manager as lm

    monkeypatch.setattr(lm.lora_manager, "_resolve_lora_path", lambda p: None)
    with pytest.raises(FileNotFoundError):
        _Backend(build_model())._load_lora_zimage([{"path": "no_such_lora.safetensors"}])
    assert "lora_not_found" in warning_codes(warnings_seen)


def test_zimage_zero_matched_targets_refuses_and_warns(tmp_path, warnings_seen,
                                                       resolve_verbatim):
    foreign = tmp_path / "foreign.safetensors"
    save_file({"lora_unet_totally_other_module.lora_down.weight": torch.zeros(RANK, 8),
               "lora_unet_totally_other_module.lora_up.weight": torch.zeros(8, RANK)},
              str(foreign), metadata={"model_type": "zimage"})

    model = build_model()
    with pytest.raises(RuntimeError, match="0 of"):
        _Backend(model)._load_lora_zimage([{"path": str(foreign), "strength": 1.0}])
    assert "lora_incompatible" in warning_codes(warnings_seen)
    assert not wrapped_paths(model)


def test_zimage_shape_mismatched_branch_is_refused_atomically(tmp_path,
                                                              warnings_seen,
                                                              resolve_verbatim):
    from safetensors.torch import load_file

    path, _trained, trained_paths = train_and_save(tmp_path)
    victim = sorted(trained_paths)[0]
    stem = "lora_transformer_" + victim.replace(".", "_")
    saved = load_file(path)
    saved[f"{stem}.lora_down.weight"] = torch.randn(RANK, 3)
    broken = tmp_path / "broken.safetensors"
    save_file(saved, str(broken), metadata={"lora_alpha": str(ALPHA)})

    model = build_model()
    before = dict(model.named_modules())
    with pytest.raises(AdapterIncompatible) as excinfo:
        _Backend(model)._load_lora_zimage(
            [{"path": str(broken), "strength": STRENGTH}])

    assert excinfo.value.code == "lora_partial"
    assert not wrapped_paths(model)
    assert dict(model.named_modules())[victim] is before[victim]
    assert "lora_partial" in warning_codes(warnings_seen)
