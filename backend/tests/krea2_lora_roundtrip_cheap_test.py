"""Krea 2: trainer save -> fresh-generation load round trip, on CPU in ~1s.

Drives the REAL ``Krea2LoRAAdapter`` (injection + ``save_checkpoint``) over a
2-block CPU stub shaped as ``iter_krea2_lora_targets`` expects, then the REAL
``Krea2Mixin._load_lora_krea2`` on a freshly built stub.

The Phase-0 defect this pins: Krea 2 shipped parser/apply/restore helpers but
its generation backend never applied ``params["loras"]`` at all.

Common composite algebra is covered in
``adapter_lycoris_roundtrip_cheap_test.py``. This file retains Krea 2's target
codec, component lifetime, and refusal seams.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/krea2_lora_roundtrip_cheap_test.py -v
"""

import pytest
import torch
from torch import nn
from safetensors.torch import load_file, save_file

from lora_roundtrip_common import (
    LoRALinearLayer, lora_delta, randomise_lora_layers,
    warning_codes, warning_probe,
)

from core.adapters import AdapterIncompatible, CompositeAdapterLayer  # noqa: E402
from core.models.krea2.krea2_lora import (  # noqa: E402
    DEFAULT_SCOPE, flatten_to_key, iter_krea2_lora_targets,
)
from core.pipeline_backends.krea2 import Krea2Mixin  # noqa: E402
from core.training.adapters.krea2_adapter import Krea2LoRAAdapter  # noqa: E402

D = 8
RANK = 4
# alpha/rank = 1.5 and strength 0.7 are chosen so that folding the strength
# anywhere but into the branch's own scale REALLY changes the bits. With the
# previous alpha=8/strength=0.5 the scale came out at exactly 1.0 and every
# plausible reassociation was identical in IEEE754.
ALPHA = 6
SCALE = ALPHA / RANK
STRENGTH = 0.7


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("to_q", "to_k", "to_v", "to_gate"):
            setattr(self, name, nn.Linear(D, D, bias=False))
        self.to_out = nn.ModuleList([nn.Linear(D, D, bias=False)])


class _FF(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("gate", "up", "down"):
            setattr(self, name, nn.Linear(D, D, bias=False))


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _Attn()
        self.ff = _FF()


class _Stub(nn.Module):
    def __init__(self, n_blocks=2):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_Block() for _ in range(n_blocks)])


class _StubTrainer:
    def __init__(self, transformer):
        self.transformer = transformer
        self.config = {}


class _Backend(Krea2Mixin):
    def __init__(self, transformer):
        self.krea2_components = {"transformer": transformer}


def build_model(n_blocks=2):
    """Deterministic base weights: the stacking gates compare two models'
    outputs, so their bases have to be the same tensors."""
    torch.manual_seed(0)
    return _Stub(n_blocks)


def wrapped_paths(model):
    """Target paths a GENERATION load covers, i.e. the composite roots."""
    return {name for name, module in model.named_modules()
            if isinstance(module, CompositeAdapterLayer)}


def lora_layer_paths(model):
    """Paths the TRAINER wrapped -- it still installs plain wrappers."""
    return {name for name, module in model.named_modules()
            if isinstance(module, LoRALinearLayer)}


def train_and_save(tmp_path, name="krea2.safetensors", seed=1234):
    model = build_model()
    adapter = Krea2LoRAAdapter(_StubTrainer(model), lora_rank=RANK, lora_alpha=ALPHA,
                               lora_dtype=torch.float32)
    layers = {}
    count = adapter.apply_lora_to_unet(layers)
    assert adapter.apply_lora_to_text_encoders(layers) == 0, "Krea 2 has no TE LoRA scope"
    assert count == len(layers) > 0
    randomise_lora_layers(layers, seed=seed)
    out = tmp_path / name
    adapter.save_checkpoint(layers, 100, 1, out)
    return str(out), lora_layer_paths(model)


def file_branch_tensors(path, target):
    """``(down, up)`` straight out of the checkpoint, for the analytic sum."""
    saved = load_file(path)
    stem = flatten_to_key(target)
    return saved[f"{stem}.lora_down.weight"], saved[f"{stem}.lora_up.weight"]


@pytest.fixture
def warnings_seen(monkeypatch):
    return warning_probe(monkeypatch)


def test_krea2_generation_wraps_exactly_the_targets_the_trainer_wrapped(tmp_path):
    path, trained_paths = train_and_save(tmp_path)

    model = build_model()
    backend = _Backend(model)
    applied = backend._load_lora_krea2([{"path": path, "strength": STRENGTH}])

    assert wrapped_paths(model) == trained_paths
    assert applied == len(trained_paths)
    # ...and that set really is the adapter's own iterator, not a subset of it.
    assert trained_paths == {p for p, _parent, _attr, _cur
                             in iter_krea2_lora_targets(build_model(), DEFAULT_SCOPE)}


def test_krea2_wrapped_forward_is_base_plus_scaled_branch(tmp_path):
    path, trained_paths = train_and_save(tmp_path)

    model = build_model()
    _Backend(model)._load_lora_krea2([{"path": path, "strength": STRENGTH}])

    modules = dict(model.named_modules())
    for target in sorted(trained_paths):
        wrapper = modules[target]
        down, up = file_branch_tensors(path, target)
        x = torch.randn(3, D)
        base = wrapper.original_module(x)
        expected = base + lora_delta(down, up, x, ALPHA, RANK, STRENGTH)
        assert torch.allclose(wrapper(x), expected, atol=1e-5), target
        assert not torch.allclose(wrapper(x), base, atol=1e-5), f"{target}: branch is inert"


def test_krea2_missing_file_refuses_and_warns(warnings_seen):
    with pytest.raises(RuntimeError):
        _Backend(build_model())._load_lora_krea2([{"path": "no_such_krea2_lora.safetensors"}])
    assert "lora_not_found" in warning_codes(warnings_seen)


def test_krea2_unreadable_file_refuses_and_warns(tmp_path, warnings_seen):
    broken = tmp_path / "broken.safetensors"
    broken.write_bytes(b"not a safetensors file")
    with pytest.raises(RuntimeError):
        _Backend(build_model())._load_lora_krea2([{"path": str(broken)}])
    assert "lora_load_failed" in warning_codes(warnings_seen)


def test_krea2_zero_matched_targets_refuses_and_warns(tmp_path, warnings_seen):
    ghost = tmp_path / "ghost.safetensors"
    stem = flatten_to_key("transformer_blocks.99.attn.to_q")
    save_file({f"{stem}.lora_down.weight": torch.zeros(RANK, D),
               f"{stem}.lora_up.weight": torch.zeros(D, RANK)}, str(ghost))

    model = build_model()
    with pytest.raises(RuntimeError, match="0 of 1 modules matched"):
        _Backend(model)._load_lora_krea2([{"path": str(ghost), "strength": 1.0}])
    assert "lora_incompatible" in warning_codes(warnings_seen)
    assert not wrapped_paths(model)


def test_krea2_partly_matching_file_is_refused_atomically(tmp_path, warnings_seen):
    path, trained_paths = train_and_save(tmp_path)
    saved = load_file(path)
    ghost = flatten_to_key("transformer_blocks.99.attn.to_q")
    saved[f"{ghost}.lora_down.weight"] = torch.zeros(RANK, D)
    saved[f"{ghost}.lora_up.weight"] = torch.zeros(D, RANK)
    partial = tmp_path / "partial.safetensors"
    save_file(saved, str(partial))

    model = build_model()
    with pytest.raises(AdapterIncompatible) as excinfo:
        _Backend(model)._load_lora_krea2([{"path": str(partial)}])
    assert excinfo.value.code == "lora_partial"
    assert not wrapped_paths(model)
    assert "lora_partial" in warning_codes(warnings_seen)


def test_krea2_dropping_the_components_drops_the_bookkeeping(tmp_path):
    """A model unload is not a reload: the maps must not survive it either."""
    path, _paths = train_and_save(tmp_path)
    backend = _Backend(build_model())
    backend._load_lora_krea2([{"path": path, "strength": 1.0}])
    backend.krea2_components = None
    assert backend._unload_lora_krea2() == 0
    assert not backend._krea2_lora_original_modules
    assert not backend._krea2_lora_wrapped_keys


def test_krea2_a_refused_file_leaves_the_ones_before_it_uninstalled(tmp_path,
                                                                   warnings_seen):
    """``AdapterSession`` plans the WHOLE request before mutating a slot.

    Krea 2 used to wrap file by file, so a second file that matched nothing left
    the first one installed and refused the generation anyway -- the request came
    back as an error over a model that was no longer the one the user has.
    """
    path, _trained_paths = train_and_save(tmp_path)
    ghost = tmp_path / "ghost.safetensors"
    stem = flatten_to_key("transformer_blocks.99.attn.to_q")
    save_file({f"{stem}.lora_down.weight": torch.zeros(RANK, D),
               f"{stem}.lora_up.weight": torch.zeros(D, RANK)}, str(ghost))

    model = build_model()
    before = dict(model.named_modules())
    backend = _Backend(model)
    with pytest.raises(RuntimeError, match="0 of 1 modules matched"):
        backend._load_lora_krea2([{"path": path, "strength": STRENGTH},
                                  {"path": str(ghost), "strength": 1.0}])

    assert not wrapped_paths(model)
    assert dict(model.named_modules()) == before
    assert not backend._krea2_lora_wrapped_keys
    assert not backend._krea2_lora_original_modules
