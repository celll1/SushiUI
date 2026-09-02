"""SenseNova U1.5: trainer save -> fresh-generation load round trip, CPU, ~2s.

Drives the REAL ``SenseNovaLoRAAdapter`` over BOTH MoT halves -- the generation
branch (``*_mot_gen``) and the understanding branch, which is injected only when
``train_text_encoder`` is set -- then the REAL
``SenseNovaMixin._load_lora_sensenova``.

The stub is 42 decoder layers wide because the adapter refuses anything else
(294 targets per branch, exactly), but each Linear is 4x4: 588 wrappers at
toy widths, well under a second.

The target naming is deliberately ASYMMETRIC and is what a key-codec regression
would break: the generation suffix is on the LINEAR for self_attn
(``q_proj_mot_gen``) and on the PARENT for the MLP (``mlp_mot_gen.gate_proj``).

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/sensenova_lora_roundtrip_cheap_test.py -v
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

from core.models.sensenova import sensenova_lora as sn_lora  # noqa: E402
from core.pipeline_backends.sensenova import SenseNovaMixin  # noqa: E402
from core.training.adapters.sensenova_adapter import SenseNovaLoRAAdapter  # noqa: E402

H = 4
LAYERS = 42  # the adapter refuses any other decoder depth
PER_BRANCH = 294
RANK = 8
ALPHA = 2  # != rank on purpose: the applied scale must be 0.25, not 1.0
SCALE = ALPHA / RANK
STRENGTH = 0.5


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, name, nn.Linear(H, H))
            setattr(self, f"{name}_mot_gen", nn.Linear(H, H))


class _Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(H, H)
        self.up_proj = nn.Linear(H, H)
        self.down_proj = nn.Linear(H, H)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attn()
        self.mlp = _Mlp()
        self.mlp_mot_gen = _Mlp()


class _LlmCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Layer() for _ in range(LAYERS)])
        self.config = SimpleNamespace(attention_dropout=0.0)


class _LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _LlmCore()


class _Stub(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = _LanguageModel()


class _Backend(SenseNovaMixin):
    def __init__(self, transformer):
        self.sensenova_components = {"transformer": transformer}


def wrapped_paths(model):
    return {name for name, module in model.named_modules()
            if isinstance(module, LoRALinearLayer)}


def train_and_save(tmp_path, name="sensenova.safetensors", seed=1234, both_halves=True):
    """Returns (path, generation target paths, understanding target paths)."""
    model = _Stub()
    trainer = SimpleNamespace(transformer=model, train_text_encoder=both_halves)
    adapter = SenseNovaLoRAAdapter(trainer, lora_rank=RANK, lora_alpha=ALPHA)
    layers = {}
    n_gen = adapter.apply_lora_to_unet(layers)
    n_und = adapter.apply_lora_to_text_encoders(layers)
    assert n_gen == PER_BRANCH
    assert n_und == (PER_BRANCH if both_halves else 0)
    randomise_lora_layers(layers, seed=seed, std=0.3)
    out = tmp_path / name
    adapter.save_checkpoint(layers, 5, 2, out)

    all_paths = wrapped_paths(model)
    gen_paths = {p for p in all_paths if "mot_gen" in p}
    und_paths = all_paths - gen_paths
    return str(out), gen_paths, und_paths


@pytest.fixture
def warnings_seen(monkeypatch):
    return warning_probe(monkeypatch)


def test_sensenova_generation_wraps_both_mot_halves_the_trainer_wrapped(tmp_path):
    path, gen_paths, und_paths = train_and_save(tmp_path)
    assert len(gen_paths) == len(und_paths) == PER_BRANCH

    model = _Stub()
    backend = _Backend(model)
    applied = backend._load_lora_sensenova([{"path": path, "strength": STRENGTH}])

    assert applied == 2 * PER_BRANCH
    assert wrapped_paths(model) == gen_paths | und_paths
    assert backend._sensenova_lora_keys == gen_paths | und_paths
    assert gen_paths == {p for p, *_rest
                         in sn_lora.iter_sensenova_lora_targets(_Stub(), branch="gen")}
    assert und_paths == {p for p, *_rest
                         in sn_lora.iter_sensenova_lora_targets(_Stub(), branch="und")}
    # The asymmetry a key-codec regression would flatten.
    assert any(p.endswith(".self_attn.q_proj_mot_gen") for p in gen_paths)
    assert any(p.endswith(".mlp_mot_gen.gate_proj") for p in gen_paths)


def test_sensenova_wrapped_forward_is_base_plus_scaled_branch(tmp_path):
    path, gen_paths, und_paths = train_and_save(tmp_path)
    saved = load_file(path)

    model = _Stub()
    _Backend(model)._load_lora_sensenova([{"path": path, "strength": STRENGTH}])

    modules = dict(model.named_modules())
    # One target per kind rather than all 588: the codec, not the arithmetic,
    # is what varies across them.
    probes = [next(iter(sorted(p for p in gen_paths if p.endswith(".self_attn.q_proj_mot_gen")))),
              next(iter(sorted(p for p in gen_paths if p.endswith(".mlp_mot_gen.down_proj")))),
              next(iter(sorted(p for p in und_paths if p.endswith(".self_attn.o_proj")))),
              next(iter(sorted(p for p in und_paths if p.endswith(".mlp.up_proj"))))]
    for target in probes:
        wrapper = modules[target]
        x = torch.randn(3, H)
        base = wrapper.original_module(x)
        expected = base + lora_delta(saved[f"{target}.lora_down.weight"],
                                     saved[f"{target}.lora_up.weight"],
                                     x, ALPHA, RANK, STRENGTH)
        assert torch.allclose(wrapper(x), expected, atol=1e-5), target
        assert not torch.allclose(wrapper(x), base, atol=1e-5), f"{target}: branch is inert"


def test_sensenova_alpha_beats_the_rank_fallback(tmp_path):
    path, gen_paths, und_paths = train_and_save(tmp_path)
    model = _Stub()
    _Backend(model)._load_lora_sensenova([{"path": path, "strength": STRENGTH}])
    scales = {round(m.scale, 9) for m in model.modules() if isinstance(m, LoRALinearLayer)}
    assert scales == {round(SCALE * STRENGTH, 9)}

    # Rung 2: no per-key alpha, alpha only in metadata. Without it the same file
    # would apply at 1.0 -- four times the trained strength.
    md_only = tmp_path / "md_alpha.safetensors"
    raw = load_file(path)
    save_file({k: v for k, v in raw.items() if not k.endswith(".alpha")}, str(md_only),
              metadata={"model_type": "sensenova", "lora_alpha": str(ALPHA),
                        "lora_rank": str(RANK)})
    model2 = _Stub()
    _Backend(model2)._load_lora_sensenova([{"path": str(md_only), "strength": 1.0}])
    assert {round(m.scale, 9) for m in model2.modules()
            if isinstance(m, LoRALinearLayer)} == {round(SCALE, 9)}

    none = tmp_path / "no_alpha.safetensors"
    save_file({k: v for k, v in raw.items() if not k.endswith(".alpha")}, str(none),
              metadata={"model_type": "sensenova"})
    model3 = _Stub()
    _Backend(model3)._load_lora_sensenova([{"path": str(none), "strength": 1.0}])
    assert {round(m.scale, 9) for m in model3.modules()
            if isinstance(m, LoRALinearLayer)} == {1.0}


def test_sensenova_unload_restores_the_identical_objects_and_is_idempotent(tmp_path):
    path, gen_paths, und_paths = train_and_save(tmp_path)

    model = _Stub()
    before = dict(model.named_modules())
    backend = _Backend(model)
    backend._load_lora_sensenova([{"path": path, "strength": 1.0}])

    assert backend._unload_lora_sensenova() == 2 * PER_BRANCH
    after = dict(model.named_modules())
    for target in gen_paths | und_paths:
        assert after[target] is before[target], target
    assert not wrapped_paths(model)
    assert backend._unload_lora_sensenova() == 0
    assert dict(model.named_modules()) == after


def test_sensenova_missing_file_refuses_and_warns(warnings_seen):
    with pytest.raises(FileNotFoundError):
        _Backend(_Stub())._load_lora_sensenova([{"path": "no_such_sn_lora.safetensors"}])
    assert "lora_not_found" in warning_codes(warnings_seen)


def test_sensenova_zero_matched_targets_refuses_and_warns(tmp_path, warnings_seen):
    foreign = tmp_path / "foreign.safetensors"
    stem = "lora_unet_transformer_blocks_0_attn_to_q"
    save_file({f"{stem}.lora_down.weight": torch.zeros(2, H),
               f"{stem}.lora_up.weight": torch.zeros(H, 2)}, str(foreign))

    model = _Stub()
    with pytest.raises(RuntimeError):
        _Backend(model)._load_lora_sensenova([{"path": str(foreign), "strength": 1.0}])
    assert "lora_incompatible" in warning_codes(warnings_seen)
    assert not wrapped_paths(model)


def test_sensenova_fully_shadowed_second_lora_refuses_and_warns(tmp_path, warnings_seen):
    path, _gen, _und = train_and_save(tmp_path)
    second, _g2, _u2 = train_and_save(tmp_path, name="second.safetensors", seed=99)

    with pytest.raises(RuntimeError):
        _Backend(_Stub())._load_lora_sensenova([{"path": path, "strength": 1.0},
                                                {"path": second, "strength": 1.0}])
    assert "lora_stacking_unsupported" in warning_codes(warnings_seen)


def test_sensenova_model_reload_never_splices_model_a_into_model_b(tmp_path):
    path, gen_paths, und_paths = train_and_save(tmp_path)

    model_a = _Stub()
    backend = _Backend(model_a)
    backend._load_lora_sensenova([{"path": path, "strength": 1.0}])
    a_ids = module_ids(model_a) | {id(m) for m in backend._sensenova_lora_orig.values()}

    model_b = _Stub()
    b_ids_before = module_ids(model_b)
    assert not (a_ids & b_ids_before), "setup: A and B must not already share modules"

    backend.sensenova_components = {"transformer": model_b}
    assert backend._sensenova_lora_keys, "the stale set must be truthy to be a test"
    assert backend._unload_lora_sensenova() == 0
    assert module_ids(model_b) == b_ids_before, "model B's module graph was modified"
    assert not (module_ids(model_b) & a_ids)

    b_before = dict(model_b.named_modules())
    assert backend._load_lora_sensenova([{"path": path, "strength": 1.0}]) == 2 * PER_BRANCH
    assert backend._unload_lora_sensenova() == 2 * PER_BRANCH
    for target in gen_paths | und_paths:
        assert dict(model_b.named_modules())[target] is b_before[target], target
    assert not (module_ids(model_b) & a_ids)
