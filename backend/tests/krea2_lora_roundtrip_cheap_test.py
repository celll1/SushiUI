"""Krea 2: trainer save -> fresh-generation load round trip, on CPU in ~1s.

Drives the REAL ``Krea2LoRAAdapter`` (injection + ``save_checkpoint``) over a
2-block CPU stub shaped as ``iter_krea2_lora_targets`` expects, then the REAL
``Krea2Mixin._load_lora_krea2`` on a freshly built stub.

The Phase-0 defect this pins: Krea 2 shipped parser/apply/restore helpers but
its generation backend never applied ``params["loras"]`` at all.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/krea2_lora_roundtrip_cheap_test.py -v
"""

import pytest
import torch
from torch import nn
from safetensors.torch import load_file, save_file

from lora_roundtrip_common import (
    LoRALinearLayer, lora_delta, module_ids, randomise_lora_layers,
    warning_codes, warning_probe,
)

from core.models.krea2.krea2_lora import (  # noqa: E402
    DEFAULT_SCOPE, flatten_to_key, iter_krea2_lora_targets,
)
from core.pipeline_backends.krea2 import Krea2Mixin  # noqa: E402
from core.training.adapters.krea2_adapter import Krea2LoRAAdapter  # noqa: E402

D = 8
RANK = 4
ALPHA = 8  # != rank on purpose: the applied scale must be 2.0, not 1.0
SCALE = ALPHA / RANK
STRENGTH = 0.5


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


def wrapped_paths(model):
    return {name for name, module in model.named_modules()
            if isinstance(module, LoRALinearLayer)}


def train_and_save(tmp_path, name="krea2.safetensors", seed=1234):
    model = _Stub()
    adapter = Krea2LoRAAdapter(_StubTrainer(model), lora_rank=RANK, lora_alpha=ALPHA,
                               lora_dtype=torch.float32)
    layers = {}
    count = adapter.apply_lora_to_unet(layers)
    assert adapter.apply_lora_to_text_encoders(layers) == 0, "Krea 2 has no TE LoRA scope"
    assert count == len(layers) > 0
    randomise_lora_layers(layers, seed=seed)
    out = tmp_path / name
    adapter.save_checkpoint(layers, 100, 1, out)
    return str(out), wrapped_paths(model)


@pytest.fixture
def warnings_seen(monkeypatch):
    return warning_probe(monkeypatch)


def test_krea2_generation_wraps_exactly_the_targets_the_trainer_wrapped(tmp_path):
    path, trained_paths = train_and_save(tmp_path)

    model = _Stub()
    backend = _Backend(model)
    applied = backend._load_lora_krea2([{"path": path, "strength": STRENGTH}])

    assert wrapped_paths(model) == trained_paths
    assert applied == len(trained_paths)
    # ...and that set really is the adapter's own iterator, not a subset of it.
    assert trained_paths == {p for p, _parent, _attr, _cur
                             in iter_krea2_lora_targets(_Stub(), DEFAULT_SCOPE)}


def test_krea2_wrapped_forward_is_base_plus_scaled_branch(tmp_path):
    path, trained_paths = train_and_save(tmp_path)
    saved = load_file(path)

    model = _Stub()
    _Backend(model)._load_lora_krea2([{"path": path, "strength": STRENGTH}])

    modules = dict(model.named_modules())
    for target in sorted(trained_paths):
        wrapper = modules[target]
        stem = flatten_to_key(target)
        x = torch.randn(3, D)
        base = wrapper.original_module(x)
        expected = base + lora_delta(saved[f"{stem}.lora_down.weight"],
                                     saved[f"{stem}.lora_up.weight"],
                                     x, ALPHA, RANK, STRENGTH)
        assert torch.allclose(wrapper(x), expected, atol=1e-5), target
        assert not torch.allclose(wrapper(x), base, atol=1e-5), f"{target}: branch is inert"


def test_krea2_alpha_beats_the_rank_fallback(tmp_path):
    path, trained_paths = train_and_save(tmp_path)
    model = _Stub()
    _Backend(model)._load_lora_krea2([{"path": path, "strength": STRENGTH}])
    modules = dict(model.named_modules())
    assert {round(modules[t].scale, 9) for t in trained_paths} == \
        {round(SCALE * STRENGTH, 9)}

    stripped = tmp_path / "no_alpha.safetensors"
    save_file({k: v for k, v in load_file(path).items() if not k.endswith(".alpha")},
              str(stripped))
    model2 = _Stub()
    _Backend(model2)._load_lora_krea2([{"path": str(stripped), "strength": STRENGTH}])
    modules2 = dict(model2.named_modules())
    assert {round(modules2[t].scale, 9) for t in trained_paths} == {round(STRENGTH, 9)}


def test_krea2_unload_restores_the_identical_objects_and_is_idempotent(tmp_path):
    path, trained_paths = train_and_save(tmp_path)

    model = _Stub()
    before = dict(model.named_modules())
    backend = _Backend(model)
    backend._load_lora_krea2([{"path": path, "strength": 1.0}])

    assert backend._unload_lora_krea2() == len(trained_paths)
    after = dict(model.named_modules())
    for target in trained_paths:
        assert after[target] is before[target], target
    assert not wrapped_paths(model)
    assert backend._unload_lora_krea2() == 0
    assert dict(model.named_modules()) == after


def test_krea2_missing_file_refuses(warnings_seen):
    with pytest.raises(RuntimeError, match="not found"):
        _Backend(_Stub())._load_lora_krea2([{"path": "no_such_krea2_lora.safetensors"}])


def test_krea2_missing_file_warns(warnings_seen):
    with pytest.raises(RuntimeError):
        _Backend(_Stub())._load_lora_krea2([{"path": "no_such_krea2_lora.safetensors"}])
    assert "lora_not_found" in warning_codes(warnings_seen)


def test_krea2_zero_matched_targets_refuses_and_warns(tmp_path, warnings_seen):
    ghost = tmp_path / "ghost.safetensors"
    stem = flatten_to_key("transformer_blocks.99.attn.to_q")
    save_file({f"{stem}.lora_down.weight": torch.zeros(RANK, D),
               f"{stem}.lora_up.weight": torch.zeros(D, RANK)}, str(ghost))

    model = _Stub()
    with pytest.raises(RuntimeError, match="0 of 1 modules matched"):
        _Backend(model)._load_lora_krea2([{"path": str(ghost), "strength": 1.0}])
    assert "lora_incompatible" in warning_codes(warnings_seen)
    assert not wrapped_paths(model)


def test_krea2_fully_shadowed_second_lora_refuses_and_warns(tmp_path, warnings_seen):
    first, _paths = train_and_save(tmp_path)
    second, _paths2 = train_and_save(tmp_path, name="second.safetensors", seed=99)

    backend = _Backend(_Stub())
    with pytest.raises(RuntimeError, match="select a single Krea 2 LoRA"):
        backend._load_lora_krea2([{"path": first, "strength": 1.0},
                                  {"path": second, "strength": 1.0}])
    assert "lora_stacking_unsupported" in warning_codes(warnings_seen)


def test_krea2_model_reload_never_splices_model_a_into_model_b(tmp_path):
    path, trained_paths = train_and_save(tmp_path)

    model_a = _Stub()
    backend = _Backend(model_a)
    backend._load_lora_krea2([{"path": path, "strength": 1.0}])
    a_ids = module_ids(model_a) | {id(m) for m in backend._krea2_lora_original_modules.values()}

    model_b = _Stub()
    b_ids_before = module_ids(model_b)
    assert not (a_ids & b_ids_before), "setup: A and B must not already share modules"

    backend.krea2_components = {"transformer": model_b}
    assert backend._krea2_lora_wrapped_keys, "the stale set must be truthy to be a test"
    assert backend._unload_lora_krea2() == 0
    assert module_ids(model_b) == b_ids_before, "model B's module graph was modified"
    assert not (module_ids(model_b) & a_ids)

    b_before = dict(model_b.named_modules())
    assert backend._load_lora_krea2([{"path": path, "strength": 1.0}]) == len(trained_paths)
    assert backend._unload_lora_krea2() == len(trained_paths)
    for target in trained_paths:
        assert dict(model_b.named_modules())[target] is b_before[target], target
    assert not (module_ids(model_b) & a_ids)


def test_krea2_dropping_the_components_drops_the_bookkeeping(tmp_path):
    """A model unload is not a reload: the maps must not survive it either."""
    path, _paths = train_and_save(tmp_path)
    backend = _Backend(_Stub())
    backend._load_lora_krea2([{"path": path, "strength": 1.0}])
    backend.krea2_components = None
    assert backend._unload_lora_krea2() == 0
    assert not backend._krea2_lora_original_modules
    assert not backend._krea2_lora_wrapped_keys
