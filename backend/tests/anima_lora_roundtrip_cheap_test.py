"""Anima: trainer save -> fresh-generation load round trip, on CPU in ~1s.

Drives the REAL ``AnimaLoRAAdapter`` (injection + ``save_checkpoint``) over a
2-block CPU stub and then the REAL ``AnimaMixin._load_lora_anima``. The stub's
CLASS NAMES are load-bearing: ``iter_anima_lora_targets`` selects by them.

The Phase-0 defect this pins: Anima's default TRAINING scope covers attention,
MLP and the LLM adapter, while generation applied only its attention iterator,
so the MLP and llm_adapter halves of a self-trained LoRA were silently dropped.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/anima_lora_roundtrip_cheap_test.py -v
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

from core.models.anima import anima_lora as anima_mod  # noqa: E402
from core.pipeline_backends.anima import AnimaMixin  # noqa: E402
from core.training.adapters.anima_adapter import AnimaLoRAAdapter  # noqa: E402

D = 8
RANK = 4
ALPHA = 8.0  # != rank on purpose: the applied scale must be 2.0, not 1.0
SCALE = ALPHA / RANK
STRENGTH = 0.5
ATTENTION_ONLY = {"attention": True, "mlp": False, "mod": False, "llm_adapter": False}
MLP_ONLY = {"attention": False, "mlp": True, "mod": False, "llm_adapter": False}


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("q_proj", "k_proj", "v_proj", "output_proj"):
            setattr(self, name, nn.Linear(D, D, bias=False))


class LLMAdapterAttention(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, name, nn.Linear(D, D, bias=False))


class GPT2FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(D, 2 * D, bias=False)
        self.layer2 = nn.Linear(2 * D, D, bias=False)


def _adaln():
    return nn.Sequential(nn.SiLU(), nn.Linear(D, D // 2, bias=False),
                         nn.Linear(D // 2, 3 * D, bias=False))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = Attention()
        self.cross_attn = Attention()
        self.mlp = GPT2FeedForward()
        self.adaln_modulation_self_attn = _adaln()
        self.adaln_modulation_cross_attn = _adaln()
        self.adaln_modulation_mlp = _adaln()


class LLMAdapterTransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = LLMAdapterAttention()
        self.cross_attn = LLMAdapterAttention()
        self.mlp = nn.Sequential(nn.Linear(D, 2 * D), nn.GELU(), nn.Linear(2 * D, D))


class LLMAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(D, D)
        self.blocks = nn.ModuleList([LLMAdapterTransformerBlock() for _ in range(2)])
        self.out_proj = nn.Linear(D, D)


class _Stub(nn.Module):
    def __init__(self, n_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([Block() for _ in range(n_blocks)])
        self.llm_adapter = LLMAdapter()


class _Backend(AnimaMixin):
    def __init__(self, transformer):
        self.anima_components = {"transformer": transformer}


def wrapped_paths(model):
    return {name for name, module in model.named_modules()
            if isinstance(module, LoRALinearLayer)}


def train_and_save(tmp_path, scope=None, name="anima.safetensors", seed=1234):
    scope = anima_mod.DEFAULT_TRAINING_SCOPE if scope is None else scope
    model = _Stub()
    trainer = SimpleNamespace(transformer=model, blockskip_config=None, config={})
    adapter = AnimaLoRAAdapter(trainer, RANK, ALPHA, torch.float32, scope=scope)
    layers = {}
    count = adapter.apply_lora_to_unet(layers)
    assert count == len(layers) > 0
    randomise_lora_layers(layers, seed=seed)
    out = tmp_path / name
    adapter.save_checkpoint(layers, 1, 1, out)
    return str(out), wrapped_paths(model)


@pytest.fixture
def warnings_seen(monkeypatch):
    return warning_probe(monkeypatch)


def test_anima_generation_wraps_exactly_the_default_training_scope(tmp_path):
    """The headline fix: the MLP and llm_adapter halves must survive to
    generation, not only the attention iterator."""
    path, trained_paths = train_and_save(tmp_path)

    model = _Stub()
    backend = _Backend(model)
    applied = backend._load_lora_anima([{"path": path, "strength": STRENGTH}])

    assert wrapped_paths(model) == trained_paths
    assert applied == len(trained_paths)
    assert any(".mlp." in p for p in trained_paths)
    assert any(p.startswith("llm_adapter.") for p in trained_paths)
    assert trained_paths == {p for p, _parent, _attr, _cur in anima_mod.iter_anima_lora_targets(
        _Stub(), anima_mod.DEFAULT_TRAINING_SCOPE)}


def test_anima_wrapped_forward_is_base_plus_scaled_branch(tmp_path):
    path, trained_paths = train_and_save(tmp_path)
    saved = load_file(path)

    model = _Stub()
    _Backend(model)._load_lora_anima([{"path": path, "strength": STRENGTH}])

    modules = dict(model.named_modules())
    for target in sorted(trained_paths):
        wrapper = modules[target]
        stem = "lora_unet_" + anima_mod._flatten_to_sdscripts(target)
        x = torch.randn(3, wrapper.original_module.in_features)
        base = wrapper.original_module(x)
        expected = base + lora_delta(saved[f"{stem}.lora_down.weight"],
                                     saved[f"{stem}.lora_up.weight"],
                                     x, ALPHA, RANK, STRENGTH)
        assert torch.allclose(wrapper(x), expected, atol=1e-5), target
        assert not torch.allclose(wrapper(x), base, atol=1e-5), f"{target}: branch is inert"


def test_anima_alpha_beats_the_rank_fallback(tmp_path):
    path, trained_paths = train_and_save(tmp_path)
    model = _Stub()
    _Backend(model)._load_lora_anima([{"path": path, "strength": STRENGTH}])
    modules = dict(model.named_modules())
    assert {round(modules[t].scale, 9) for t in trained_paths} == {round(SCALE * STRENGTH, 9)}

    stripped = tmp_path / "no_alpha.safetensors"
    save_file({k: v for k, v in load_file(path).items() if not k.endswith(".alpha")},
              str(stripped), metadata={"model_type": "anima"})
    model2 = _Stub()
    _Backend(model2)._load_lora_anima([{"path": str(stripped), "strength": STRENGTH}])
    modules2 = dict(model2.named_modules())
    assert {round(modules2[t].scale, 9) for t in trained_paths} == {round(STRENGTH, 9)}


def test_anima_unload_restores_the_identical_objects_and_is_idempotent(tmp_path):
    path, trained_paths = train_and_save(tmp_path)

    model = _Stub()
    before = dict(model.named_modules())
    backend = _Backend(model)
    backend._load_lora_anima([{"path": path, "strength": 1.0}])

    assert backend._unload_lora_anima() == len(trained_paths)
    after = dict(model.named_modules())
    for target in trained_paths:
        assert after[target] is before[target], target
    assert not wrapped_paths(model)
    assert backend._unload_lora_anima() == 0
    assert dict(model.named_modules()) == after


def test_anima_missing_file_refuses_and_warns(warnings_seen):
    with pytest.raises(RuntimeError):
        _Backend(_Stub())._load_lora_anima([{"path": "no_such_anima_lora.safetensors"}])
    assert warning_codes(warnings_seen) == ["lora_not_found"]


def test_anima_zero_matched_targets_refuses_and_warns(tmp_path, warnings_seen):
    ghost = tmp_path / "ghost.safetensors"
    save_file({"lora_unet_blocks_9_self_attn_q_proj.lora_down.weight": torch.zeros(RANK, D),
               "lora_unet_blocks_9_self_attn_q_proj.lora_up.weight": torch.zeros(D, RANK)},
              str(ghost), metadata={"model_type": "anima"})

    model = _Stub()
    with pytest.raises(RuntimeError):
        _Backend(model)._load_lora_anima([{"path": str(ghost), "strength": 1.0}])
    assert "lora_incompatible" in warning_codes(warnings_seen)
    assert not wrapped_paths(model)


def test_anima_fully_shadowed_second_lora_refuses_and_warns(tmp_path, warnings_seen):
    path, _paths = train_and_save(tmp_path, scope=ATTENTION_ONLY)

    model = _Stub()
    with pytest.raises(RuntimeError):
        _Backend(model)._load_lora_anima([{"path": path}, {"path": path}])
    assert "lora_stacking_unsupported" in warning_codes(warnings_seen)
    assert not wrapped_paths(model), "the refused stack left wrappers on the DiT"


def test_anima_disjoint_scopes_stack_additively(tmp_path, warnings_seen):
    """Two LoRAs over disjoint scopes are the case that must NOT be refused."""
    attn, attn_paths = train_and_save(tmp_path, scope=ATTENTION_ONLY, name="attn.safetensors")
    mlp, mlp_paths = train_and_save(tmp_path, scope=MLP_ONLY, name="mlp.safetensors", seed=7)
    assert not (attn_paths & mlp_paths)

    model = _Stub()
    backend = _Backend(model)
    total = backend._load_lora_anima([{"path": attn}, {"path": mlp}])
    assert total == len(attn_paths) + len(mlp_paths)
    assert wrapped_paths(model) == attn_paths | mlp_paths
    assert warning_codes(warnings_seen) == []
    assert backend._unload_lora_anima() == total


def test_anima_model_reload_never_splices_model_a_into_model_b(tmp_path):
    path, trained_paths = train_and_save(tmp_path)

    model_a = _Stub()
    backend = _Backend(model_a)
    backend._load_lora_anima([{"path": path, "strength": 1.0}])
    a_ids = module_ids(model_a) | {id(m) for m in backend._anima_lora_original_modules.values()}

    model_b = _Stub()
    b_ids_before = module_ids(model_b)
    assert not (a_ids & b_ids_before), "setup: A and B must not already share modules"

    backend.anima_components = {"transformer": model_b}
    assert backend._anima_lora_wrapped_keys, "the stale set must be truthy to be a test"
    assert backend._unload_lora_anima() == 0
    assert module_ids(model_b) == b_ids_before, "model B's module graph was modified"
    assert not (module_ids(model_b) & a_ids)

    b_before = dict(model_b.named_modules())
    assert backend._load_lora_anima([{"path": path, "strength": 1.0}]) == len(trained_paths)
    assert backend._unload_lora_anima() == len(trained_paths)
    for target in trained_paths:
        assert dict(model_b.named_modules())[target] is b_before[target], target
    assert not (module_ids(model_b) & a_ids)
