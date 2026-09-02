"""Lens: trainer save -> fresh-generation load round trip, CPU, ~1s.

Drives the REAL ``LensLoRAAdapter`` (injection + ``save_checkpoint``) over a
2-block CPU stub and the REAL ``LensMixin._load_lora_lens``. Lens carries the
fused-QKV naming (``img_qkv``/``txt_qkv``) and two int-slot targets
(``attn.to_out.0``, ``img_mod.1``), which is why the target-set equality here
is worth more than a count.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/lens_lora_roundtrip_cheap_test.py -v
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from safetensors.torch import load_file, save_file

from lora_roundtrip_common import (
    LoRALinearLayer, lora_delta, module_ids, randomise_lora_layers,
    warning_codes, warning_probe,
)

from core.models.lens.lens_lora import (  # noqa: E402
    DEFAULT_SCOPE, _FULL_SCOPE, _flatten_to_sdscripts, iter_lens_lora_targets,
)
from core.pipeline_backends.lens import LensMixin  # noqa: E402
from core.training.adapters.lens_adapter import LensLoRAAdapter  # noqa: E402

D = 8
RANK = 4
ALPHA = 8  # != rank on purpose: the applied scale must be 2.0, not 1.0
SCALE = ALPHA / RANK
STRENGTH = 0.5
MOD_ONLY = {k: (k == "mod") for k in _FULL_SCOPE}


def _linear():
    layer = nn.Linear(D, D)
    nn.init.normal_(layer.weight, std=0.05)
    nn.init.normal_(layer.bias, std=0.05)
    return layer


class _Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1, self.w2, self.w3 = _linear(), _linear(), _linear()


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_qkv = _linear()
        self.txt_qkv = _linear()
        self.to_out = nn.ModuleList([_linear()])
        self.to_add_out = _linear()


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _Attn()
        self.img_mlp = _Mlp()
        self.txt_mlp = _Mlp()
        self.img_mod = nn.Sequential(nn.SiLU(), _linear())
        self.txt_mod = nn.Sequential(nn.SiLU(), _linear())


class _Stub(nn.Module):
    def __init__(self, n_blocks=2):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_Block() for _ in range(n_blocks)])


class _Backend(LensMixin):
    def __init__(self, transformer):
        self.lens_components = {"transformer": transformer}


def wrapped_paths(model):
    return {name for name, module in model.named_modules()
            if isinstance(module, LoRALinearLayer)}


def train_and_save(tmp_path, scope=None, name="lens.safetensors", seed=1234):
    model = _Stub()
    adapter = LensLoRAAdapter(SimpleNamespace(transformer=model, config={}),
                              RANK, ALPHA, torch.float32, scope=scope)
    layers = {}
    count = adapter.apply_lora_to_unet(layers)
    assert adapter.apply_lora_to_text_encoders(layers) == 0, "GPT-OSS is frozen"
    assert count == len(layers) > 0
    randomise_lora_layers(layers, seed=seed, std=0.3)
    out = tmp_path / name
    adapter.save_checkpoint(layers, 7, 1, out)
    return str(out), wrapped_paths(model)


@pytest.fixture
def warnings_seen(monkeypatch):
    return warning_probe(monkeypatch)


def test_lens_generation_wraps_exactly_the_targets_the_trainer_wrapped(tmp_path):
    path, trained_paths = train_and_save(tmp_path)

    model = _Stub()
    backend = _Backend(model)
    applied = backend._load_lora_lens([{"path": path, "strength": STRENGTH}])

    assert wrapped_paths(model) == trained_paths
    assert applied == len(trained_paths)
    assert backend._lens_lora_wrapped_keys == trained_paths
    assert trained_paths == {p for p, _parent, _attr, _cur
                             in iter_lens_lora_targets(_Stub(), DEFAULT_SCOPE)}
    # The int-slot target really went through parent[idx], not setattr.
    assert any(p.endswith(".attn.to_out.0") for p in trained_paths)


def test_lens_mod_scope_reaches_generation(tmp_path):
    """The `mod` group is trainable and not in DEFAULT_SCOPE; a mod-scope
    checkpoint must still apply in full at generation."""
    path, trained_paths = train_and_save(tmp_path, scope=_FULL_SCOPE, name="full.safetensors")
    assert any(p.endswith(".img_mod.1") for p in trained_paths)

    model = _Stub()
    assert _Backend(model)._load_lora_lens([{"path": path}]) == len(trained_paths)
    assert wrapped_paths(model) == trained_paths


def test_lens_wrapped_forward_is_base_plus_scaled_branch(tmp_path):
    path, trained_paths = train_and_save(tmp_path, scope=_FULL_SCOPE)
    saved = load_file(path)

    model = _Stub()
    _Backend(model)._load_lora_lens([{"path": path, "strength": STRENGTH}])

    modules = dict(model.named_modules())
    for target in sorted(trained_paths):
        wrapper = modules[target]
        stem = "lora_unet_" + _flatten_to_sdscripts(target)
        x = torch.randn(3, D)
        base = wrapper.original_module(x)
        expected = base + lora_delta(saved[f"{stem}.lora_down.weight"],
                                     saved[f"{stem}.lora_up.weight"],
                                     x, ALPHA, RANK, STRENGTH)
        assert torch.allclose(wrapper(x), expected, atol=1e-5), target
        assert not torch.allclose(wrapper(x), base, atol=1e-5), f"{target}: branch is inert"


def test_lens_alpha_beats_the_rank_fallback(tmp_path):
    path, trained_paths = train_and_save(tmp_path)
    model = _Stub()
    _Backend(model)._load_lora_lens([{"path": path, "strength": STRENGTH}])
    modules = dict(model.named_modules())
    assert {round(modules[t].scale, 9) for t in trained_paths} == {round(SCALE * STRENGTH, 9)}

    # Rung 2: no per-key alpha, alpha only in file metadata.
    md_only = tmp_path / "md_alpha.safetensors"
    save_file({k: v for k, v in load_file(path).items() if not k.endswith(".alpha")},
              str(md_only), metadata={"model_type": "lens", "lora_alpha": str(4 * RANK)})
    model2 = _Stub()
    _Backend(model2)._load_lora_lens([{"path": str(md_only), "strength": 1.0}])
    modules2 = dict(model2.named_modules())
    assert {round(modules2[t].scale, 9) for t in trained_paths} == {4.0}

    # Rung 3: no alpha anywhere -> the rank fallback, scale 1.0.
    none = tmp_path / "no_alpha.safetensors"
    save_file({k: v for k, v in load_file(path).items() if not k.endswith(".alpha")},
              str(none), metadata={"model_type": "lens"})
    model3 = _Stub()
    _Backend(model3)._load_lora_lens([{"path": str(none), "strength": 1.0}])
    modules3 = dict(model3.named_modules())
    assert {round(modules3[t].scale, 9) for t in trained_paths} == {1.0}


def test_lens_unload_restores_the_identical_objects_and_is_idempotent(tmp_path):
    path, trained_paths = train_and_save(tmp_path, scope=_FULL_SCOPE)

    model = _Stub()
    before = dict(model.named_modules())
    backend = _Backend(model)
    backend._load_lora_lens([{"path": path, "strength": 1.0}])

    assert backend._unload_lora_lens() == len(trained_paths)
    after = dict(model.named_modules())
    for target in trained_paths:
        assert after[target] is before[target], target
    assert not wrapped_paths(model)
    assert backend._unload_lora_lens() == 0
    assert dict(model.named_modules()) == after


def test_lens_missing_file_refuses_and_warns(warnings_seen):
    with pytest.raises(FileNotFoundError, match="not found"):
        _Backend(_Stub())._load_lora_lens([{"path": "no_such_lens_lora.safetensors"}])
    assert "lora_not_found" in warning_codes(warnings_seen)


def test_lens_zero_matched_targets_refuses_and_warns(tmp_path, warnings_seen):
    foreign = tmp_path / "foreign.safetensors"
    save_file({"totally.unrelated.weight": torch.zeros(2, 2)}, str(foreign),
              metadata={"model_type": "not_lens"})

    model = _Stub()
    with pytest.raises(RuntimeError):
        _Backend(model)._load_lora_lens([{"path": str(foreign), "strength": 1.0}])
    assert "lora_incompatible" in warning_codes(warnings_seen)
    assert not wrapped_paths(model)


def test_lens_fully_shadowed_second_lora_refuses_and_warns(tmp_path, warnings_seen):
    path, _paths = train_and_save(tmp_path)
    second, _paths2 = train_and_save(tmp_path, name="second.safetensors", seed=99)

    backend = _Backend(_Stub())
    with pytest.raises(RuntimeError):
        backend._load_lora_lens([{"path": path, "strength": 1.0},
                                 {"path": second, "strength": 1.0}])
    assert "lora_stacking_unsupported" in warning_codes(warnings_seen)


def test_lens_model_reload_never_splices_model_a_into_model_b(tmp_path):
    path, trained_paths = train_and_save(tmp_path)

    model_a = _Stub()
    backend = _Backend(model_a)
    backend._load_lora_lens([{"path": path, "strength": 1.0}])
    a_ids = module_ids(model_a) | {id(m) for m in backend._lens_lora_original_modules.values()}

    model_b = _Stub()
    b_ids_before = module_ids(model_b)
    assert not (a_ids & b_ids_before), "setup: A and B must not already share modules"

    backend.lens_components["transformer"] = model_b
    assert backend._lens_lora_wrapped_keys, "the stale set must be truthy to be a test"
    assert backend._unload_lora_lens() == 0
    assert module_ids(model_b) == b_ids_before, "model B's module graph was modified"
    assert not (module_ids(model_b) & a_ids)

    b_before = dict(model_b.named_modules())
    assert backend._load_lora_lens([{"path": path, "strength": 1.0}]) == len(trained_paths)
    assert backend._unload_lora_lens() == len(trained_paths)
    for target in trained_paths:
        assert dict(model_b.named_modules())[target] is b_before[target], target
    assert not (module_ids(model_b) & a_ids)


def test_lens_quantization_is_dropped_while_wrappers_are_live(tmp_path, warnings_seen):
    """An fp8 cast would walk into lora_down/lora_up, so the request's
    quantization must be refused rather than applied over the adapters."""
    path, _paths = train_and_save(tmp_path)
    backend = _Backend(_Stub())
    assert backend._lens_quantization_with_lora("fp8_e4m3fn") == "fp8_e4m3fn"

    backend._load_lora_lens([{"path": path, "strength": 1.0}])
    assert backend._lens_quantization_with_lora("fp8_e4m3fn") is None
    assert "quantization_fallback" in warning_codes(warnings_seen)

    backend._unload_lora_lens()
    assert backend._lens_quantization_with_lora("fp8_e4m3fn") == "fp8_e4m3fn"
