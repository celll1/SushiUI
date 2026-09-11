"""Lens: trainer save -> fresh-generation load round trip, CPU, ~1s.

Drives the REAL ``LensLoRAAdapter`` (injection + ``save_checkpoint``) over a
2-block CPU stub and the REAL ``LensMixin._load_lora_lens``. Lens carries the
fused-QKV naming (``img_qkv``/``txt_qkv``) and three int-slot targets
(``attn.to_out.0``, ``img_mod.1``, ``txt_mod.1``), which is why the target-set
equality here is worth more than a count.

Common composite algebra is covered in
``adapter_lycoris_roundtrip_cheap_test.py``. This file retains Lens's fused-QKV
and int-slot targets, FP8 gate, lifecycle, and refusal seams.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/lens_lora_roundtrip_cheap_test.py -v
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from safetensors.torch import load_file, save_file

from lora_roundtrip_common import (
    LoRALinearLayer, lora_delta, randomise_lora_layers,
    warning_codes, warning_probe,
)

from core.adapters import AdapterIncompatible, CompositeAdapterLayer  # noqa: E402
from core.models.lens.lens_lora import (  # noqa: E402
    DEFAULT_SCOPE, _FULL_SCOPE, _flatten_to_sdscripts, iter_lens_lora_targets,
)
from core.pipeline_backends.lens import LensMixin  # noqa: E402
from core.training.adapters.lens_adapter import LensLoRAAdapter  # noqa: E402

D = 8
RANK = 4
# alpha/rank = 1.5 and strength 0.7: the previous alpha=8/strength=0.5 gave a
# scale of exactly 1.0, where every plausible reassociation of the strength fold
# is identical in IEEE754 and the bit-identity gate cannot bite.
ALPHA = 6
SCALE = ALPHA / RANK
STRENGTH = 0.7


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


def train_and_save(tmp_path, scope=None, name="lens.safetensors", seed=1234):
    model = build_model()
    adapter = LensLoRAAdapter(SimpleNamespace(transformer=model, config={}),
                              RANK, ALPHA, torch.float32, scope=scope)
    layers = {}
    count = adapter.apply_lora_to_unet(layers)
    assert adapter.apply_lora_to_text_encoders(layers) == 0, "GPT-OSS is frozen"
    assert count == len(layers) > 0
    randomise_lora_layers(layers, seed=seed, std=0.3)
    out = tmp_path / name
    adapter.save_checkpoint(layers, 7, 1, out)
    return str(out), lora_layer_paths(model)


def file_branch_tensors(path, target):
    """``(down, up)`` straight out of the checkpoint, for the analytic sum."""
    saved = load_file(path)
    stem = "lora_unet_" + _flatten_to_sdscripts(target)
    return saved[f"{stem}.lora_down.weight"], saved[f"{stem}.lora_up.weight"]


@pytest.fixture
def warnings_seen(monkeypatch):
    return warning_probe(monkeypatch)


def test_lens_generation_wraps_exactly_the_targets_the_trainer_wrapped(tmp_path):
    path, trained_paths = train_and_save(tmp_path)

    model = build_model()
    backend = _Backend(model)
    applied = backend._load_lora_lens([{"path": path, "strength": STRENGTH}])

    assert wrapped_paths(model) == trained_paths
    assert applied == len(trained_paths)
    assert backend._lens_lora_wrapped_keys == trained_paths
    assert trained_paths == {p for p, _parent, _attr, _cur
                             in iter_lens_lora_targets(build_model(), DEFAULT_SCOPE)}
    # The int-slot target really went through parent[idx], not setattr.
    assert any(p.endswith(".attn.to_out.0") for p in trained_paths)


def test_lens_mod_scope_reaches_generation(tmp_path):
    """The `mod` group is trainable and not in DEFAULT_SCOPE; a mod-scope
    checkpoint must still apply in full at generation. Both its targets are
    ``nn.Sequential`` INDICES, so a hand-rolled setattr would silently no-op."""
    path, trained_paths = train_and_save(tmp_path, scope=_FULL_SCOPE, name="full.safetensors")
    assert any(p.endswith(".img_mod.1") for p in trained_paths)
    assert any(p.endswith(".txt_mod.1") for p in trained_paths)

    model = build_model()
    assert _Backend(model)._load_lora_lens([{"path": path}]) == len(trained_paths)
    assert wrapped_paths(model) == trained_paths


def test_lens_wrapped_forward_is_base_plus_scaled_branch(tmp_path):
    path, trained_paths = train_and_save(tmp_path, scope=_FULL_SCOPE)

    model = build_model()
    _Backend(model)._load_lora_lens([{"path": path, "strength": STRENGTH}])

    modules = dict(model.named_modules())
    for target in sorted(trained_paths):
        wrapper = modules[target]
        down, up = file_branch_tensors(path, target)
        x = torch.randn(3, D)
        base = wrapper.original_module(x)
        expected = base + lora_delta(down, up, x, ALPHA, RANK, STRENGTH)
        assert torch.allclose(wrapper(x), expected, atol=1e-5), target
        assert not torch.allclose(wrapper(x), base, atol=1e-5), f"{target}: branch is inert"


def test_lens_a_second_request_does_not_stack_onto_a_leaked_wrapper(tmp_path):
    """Load restores first, unconditionally. Now that a second branch SUMS
    instead of being refused, a wrapper that outlived a failed restore would
    silently double-apply this request's LoRA."""
    path, trained_paths = train_and_save(tmp_path)

    model = build_model()
    backend = _Backend(model)
    backend._load_lora_lens([{"path": path, "strength": STRENGTH}])
    # Simulate a restore that never ran (the outer finally swallows failures).
    backend._load_lora_lens([{"path": path, "strength": STRENGTH}])

    for target in trained_paths:
        assert len(dict(model.named_modules())[target]) == 1, target


def test_lens_missing_file_refuses_and_warns(warnings_seen):
    with pytest.raises(FileNotFoundError, match="not found"):
        _Backend(build_model())._load_lora_lens([{"path": "no_such_lens_lora.safetensors"}])
    assert "lora_not_found" in warning_codes(warnings_seen)


def test_lens_unreadable_file_refuses_and_warns(tmp_path, warnings_seen):
    broken = tmp_path / "broken.safetensors"
    broken.write_bytes(b"not a safetensors file")
    with pytest.raises(RuntimeError):
        _Backend(build_model())._load_lora_lens([{"path": str(broken)}])
    assert "lora_load_failed" in warning_codes(warnings_seen)


def test_lens_zero_matched_targets_refuses_and_warns(tmp_path, warnings_seen):
    foreign = tmp_path / "foreign.safetensors"
    save_file({"totally.unrelated.weight": torch.zeros(2, 2)}, str(foreign),
              metadata={"model_type": "not_lens"})

    model = build_model()
    with pytest.raises(RuntimeError):
        _Backend(model)._load_lora_lens([{"path": str(foreign), "strength": 1.0}])
    assert "lora_incompatible" in warning_codes(warnings_seen)
    assert not wrapped_paths(model)


def test_lens_partly_matching_file_is_refused_atomically(tmp_path, warnings_seen):
    path, trained_paths = train_and_save(tmp_path)
    saved = load_file(path)
    ghost = "lora_unet_" + _flatten_to_sdscripts("transformer_blocks.9.attn.img_qkv")
    saved[f"{ghost}.lora_down.weight"] = torch.zeros(RANK, D)
    saved[f"{ghost}.lora_up.weight"] = torch.zeros(D, RANK)
    partial = tmp_path / "partial.safetensors"
    save_file(saved, str(partial), metadata={"model_type": "lens"})

    model = build_model()
    with pytest.raises(AdapterIncompatible) as excinfo:
        _Backend(model)._load_lora_lens([{"path": str(partial)}])
    assert excinfo.value.code == "lora_partial"
    assert not wrapped_paths(model)
    assert "lora_partial" in warning_codes(warnings_seen)


def test_lens_two_loras_over_the_same_targets_stack_instead_of_refusing(tmp_path,
                                                                       warnings_seen):
    """The refusal this file used to assert. The same file twice is two branches,
    not a duplicate-name error, because branch names carry the request index."""
    path, trained_paths = train_and_save(tmp_path)
    second, _paths2 = train_and_save(tmp_path, name="second.safetensors", seed=99)

    model = build_model()
    backend = _Backend(model)
    assert backend._load_lora_lens([{"path": path, "strength": 1.0},
                                    {"path": second, "strength": 1.0}]) == \
        2 * len(trained_paths)
    assert wrapped_paths(model) == trained_paths
    assert warning_codes(warnings_seen) == []


def test_lens_quantization_is_dropped_while_wrappers_are_live(tmp_path, warnings_seen):
    """An fp8 cast would walk into lora_down/lora_up, so the request's
    quantization must be refused rather than applied over the adapters.

    The gate reads ``lora_wrapped_count``, which counts adapter ROOTS: a
    composite is one root no matter how many branches it holds, so the gate must
    still fire over a STACK and not only over a single LoRA.
    """
    path, trained_paths = train_and_save(tmp_path)
    second, _paths2 = train_and_save(tmp_path, name="second.safetensors", seed=99)
    model = build_model()
    backend = _Backend(model)
    assert backend._lens_quantization_with_lora("fp8_e4m3fn") == "fp8_e4m3fn"

    backend._load_lora_lens([{"path": path, "strength": 1.0},
                             {"path": second, "strength": 1.0}])
    from core.models.common.int8_runtime_quantize import lora_wrapped_count
    assert lora_wrapped_count(model) == len(trained_paths)
    assert backend._lens_quantization_with_lora("fp8_e4m3fn") is None
    assert "quantization_fallback" in warning_codes(warnings_seen)

    backend._unload_lora_lens()
    assert lora_wrapped_count(model) == 0
    assert backend._lens_quantization_with_lora("fp8_e4m3fn") == "fp8_e4m3fn"


def test_lens_a_refused_file_leaves_the_ones_before_it_uninstalled(tmp_path,
                                                                  warnings_seen):
    """``AdapterSession`` plans the WHOLE request before mutating a slot.

    Lens used to wrap file by file, so a second file that matched nothing left
    the first one installed and refused the generation anyway -- and the FP8 gate
    would then have seen wrappers on a request that never ran.
    """
    path, _trained_paths = train_and_save(tmp_path)
    ghost_stem = "lora_unet_" + _flatten_to_sdscripts("transformer_blocks.9.attn.img_qkv")
    ghost = tmp_path / "ghost.safetensors"
    save_file({f"{ghost_stem}.lora_down.weight": torch.zeros(RANK, D),
               f"{ghost_stem}.lora_up.weight": torch.zeros(D, RANK)},
              str(ghost), metadata={"model_type": "lens"})

    model = build_model()
    before = dict(model.named_modules())
    backend = _Backend(model)
    with pytest.raises(RuntimeError, match="0 of 1 down/up pairs"):
        backend._load_lora_lens([{"path": path, "strength": STRENGTH},
                                 {"path": str(ghost), "strength": 1.0}])

    assert not wrapped_paths(model)
    assert dict(model.named_modules()) == before
    assert not backend._lens_lora_wrapped_keys
    assert not backend._lens_lora_original_modules
    from core.models.common.int8_runtime_quantize import lora_wrapped_count
    assert lora_wrapped_count(model) == 0
