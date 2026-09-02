"""LTX-2.3: trainer save -> fresh-generation load round trip, CPU, ~1s.

Drives the REAL ``Ltx2LoRAAdapter`` (injection + ``save_checkpoint``) over a
2-block CPU stub and the REAL ``LTX2Mixin._load_lora_ltx2``.

The Phase-0 defect this pins: LTX-2.3 had a training adapter and NO generation
loader at all, so a self-trained LoRA could only ever be ignored.

Complementary to ``video_lora_threading_test.py``, which anchors the request
plumbing (routes, FormData, panels, openapi) and the block-swap cache
reconciliation by source inspection. This file is the numerical half: the
targets, the scale, the restore and the reload guard, executed.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/ltx2_lora_roundtrip_cheap_test.py -v
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

from api.error_handlers import ValidationError  # noqa: E402
from core.pipeline_backends.ltx2 import LTX2Mixin  # noqa: E402
from core.training.adapters.ltx2_adapter import (  # noqa: E402
    DEFAULT_LTX2_SCOPE, Ltx2LoRAAdapter, _flatten_to_sdscripts,
    iter_ltx2_lora_targets,
)

D = 8
RANK = 4
ALPHA = 8  # != rank on purpose: the applied scale must be 2.0, not 1.0
SCALE = ALPHA / RANK
STRENGTH = 0.5
ATTENTION_AND_FF = {**DEFAULT_LTX2_SCOPE, "ff": True}


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(D, D, bias=False)
        self.to_k = nn.Linear(D, D, bias=False)
        self.to_v = nn.Linear(D, D, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(D, D, bias=False)])


class _Ff(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.ModuleList([nn.Linear(D, D, bias=False), nn.GELU(),
                                  nn.Linear(D, D, bias=False)])


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1 = _Attn()
        self.attn2 = _Attn()
        self.ff = _Ff()


class _Dit(nn.Module):
    def __init__(self, n_blocks=2):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_Block() for _ in range(n_blocks)])
        self.dtype = torch.float32


class _Backend(LTX2Mixin):
    def __init__(self, transformer):
        self.ltx2_components = {"transformer": transformer}


def wrapped_paths(model):
    return {name for name, module in model.named_modules()
            if isinstance(module, LoRALinearLayer)}


def train_and_save(tmp_path, scope=None, name="ltx2.safetensors", seed=1234):
    dit = _Dit()
    adapter = Ltx2LoRAAdapter(SimpleNamespace(transformer=dit, config={}),
                              RANK, ALPHA, torch.float32, scope=scope)
    layers = {}
    count = adapter.apply_lora_to_unet(layers)
    assert adapter.apply_lora_to_text_encoders(layers) == 0, "Gemma-3 is frozen"
    assert count == len(layers) > 0
    randomise_lora_layers(layers, seed=seed, std=0.3)
    out = tmp_path / name
    adapter.save_checkpoint(layers, 7, 1, out)
    return str(out), wrapped_paths(dit)


@pytest.fixture
def warnings_seen(monkeypatch):
    return warning_probe(monkeypatch)


def test_ltx2_generation_wraps_exactly_the_targets_the_trainer_wrapped(tmp_path):
    path, trained_paths = train_and_save(tmp_path)

    dit = _Dit()
    backend = _Backend(dit)
    applied = backend._load_lora_ltx2([{"path": path, "strength": STRENGTH}])

    assert wrapped_paths(dit) == trained_paths
    assert applied == len(trained_paths)
    assert backend._ltx2_lora_wrapped_keys == trained_paths
    assert trained_paths == {p for p, _parent, _attr, _cur
                             in iter_ltx2_lora_targets(_Dit(), DEFAULT_LTX2_SCOPE)}
    assert any(p.endswith(".to_out.0") for p in trained_paths)


def test_ltx2_opt_in_feed_forward_scope_reaches_generation(tmp_path):
    path, trained_paths = train_and_save(tmp_path, scope=ATTENTION_AND_FF,
                                         name="ff.safetensors")
    assert any(".ff." in p for p in trained_paths)

    dit = _Dit()
    assert _Backend(dit)._load_lora_ltx2([{"path": path}]) == len(trained_paths)
    assert wrapped_paths(dit) == trained_paths


def test_ltx2_wrapped_forward_is_base_plus_scaled_branch(tmp_path):
    path, trained_paths = train_and_save(tmp_path, scope=ATTENTION_AND_FF)
    saved = load_file(path)

    dit = _Dit()
    _Backend(dit)._load_lora_ltx2([{"path": path, "strength": STRENGTH}])

    modules = dict(dit.named_modules())
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


def test_ltx2_alpha_beats_the_rank_fallback(tmp_path):
    path, trained_paths = train_and_save(tmp_path)
    dit = _Dit()
    _Backend(dit)._load_lora_ltx2([{"path": path, "strength": STRENGTH}])
    modules = dict(dit.named_modules())
    assert {round(modules[t].scale, 9) for t in trained_paths} == {round(SCALE * STRENGTH, 9)}

    md_only = tmp_path / "md_alpha.safetensors"
    save_file({k: v for k, v in load_file(path).items() if not k.endswith(".alpha")},
              str(md_only), metadata={"model_type": "ltx2", "lora_alpha": str(4 * RANK)})
    dit2 = _Dit()
    _Backend(dit2)._load_lora_ltx2([{"path": str(md_only), "strength": 1.0}])
    modules2 = dict(dit2.named_modules())
    assert {round(modules2[t].scale, 9) for t in trained_paths} == {4.0}

    none = tmp_path / "no_alpha.safetensors"
    save_file({k: v for k, v in load_file(path).items() if not k.endswith(".alpha")},
              str(none), metadata={"model_type": "ltx2"})
    dit3 = _Dit()
    _Backend(dit3)._load_lora_ltx2([{"path": str(none), "strength": 1.0}])
    modules3 = dict(dit3.named_modules())
    assert {round(modules3[t].scale, 9) for t in trained_paths} == {1.0}


def test_ltx2_unload_restores_the_identical_objects_and_is_idempotent(tmp_path):
    path, trained_paths = train_and_save(tmp_path, scope=ATTENTION_AND_FF)

    dit = _Dit()
    before = dict(dit.named_modules())
    backend = _Backend(dit)
    backend._load_lora_ltx2([{"path": path, "strength": 1.0}])

    assert backend._unload_lora_ltx2() == len(trained_paths)
    after = dict(dit.named_modules())
    for target in trained_paths:
        assert after[target] is before[target], target
    assert not wrapped_paths(dit)
    assert backend._unload_lora_ltx2() == 0
    assert dict(dit.named_modules()) == after


def test_ltx2_missing_file_refuses_and_warns(warnings_seen):
    with pytest.raises((FileNotFoundError, ValidationError, RuntimeError)):
        _Backend(_Dit())._load_lora_ltx2([{"path": "no_such_ltx2_lora.safetensors"}])
    assert "lora_not_found" in warning_codes(warnings_seen)


def test_ltx2_zero_matched_targets_refuses_and_warns(tmp_path, warnings_seen):
    ghost = tmp_path / "ghost.safetensors"
    stem = "lora_unet_transformer_blocks_9_attn1_to_q"
    save_file({f"{stem}.lora_down.weight": torch.zeros(RANK, D),
               f"{stem}.lora_up.weight": torch.zeros(D, RANK)},
              str(ghost), metadata={"model_type": "ltx2"})

    dit = _Dit()
    with pytest.raises((ValidationError, RuntimeError)):
        _Backend(dit)._load_lora_ltx2([{"path": str(ghost), "strength": 1.0}])
    assert "lora_incompatible" in warning_codes(warnings_seen)
    assert not wrapped_paths(dit), "a refused load left wrappers on the DiT"


def test_ltx2_fully_shadowed_second_lora_refuses_and_warns(tmp_path, warnings_seen):
    path, _paths = train_and_save(tmp_path)
    second, _paths2 = train_and_save(tmp_path, name="second.safetensors", seed=99)

    dit = _Dit()
    with pytest.raises((ValidationError, RuntimeError)):
        _Backend(dit)._load_lora_ltx2([{"path": path, "strength": 1.0},
                                       {"path": second, "strength": 1.0}])
    assert "lora_stacking_unsupported" in warning_codes(warnings_seen)


def test_ltx2_model_reload_never_splices_model_a_into_model_b(tmp_path):
    path, trained_paths = train_and_save(tmp_path)

    dit_a = _Dit()
    backend = _Backend(dit_a)
    backend._load_lora_ltx2([{"path": path, "strength": 1.0}])
    a_ids = module_ids(dit_a) | {id(m) for m in backend._ltx2_lora_original_modules.values()}

    dit_b = _Dit()
    b_ids_before = module_ids(dit_b)
    assert not (a_ids & b_ids_before), "setup: A and B must not already share modules"

    backend.ltx2_components["transformer"] = dit_b
    assert backend._ltx2_lora_wrapped_keys, "the stale set must be truthy to be a test"
    assert backend._unload_lora_ltx2() == 0
    assert module_ids(dit_b) == b_ids_before, "model B's module graph was modified"
    assert not (module_ids(dit_b) & a_ids)

    b_before = dict(dit_b.named_modules())
    assert backend._load_lora_ltx2([{"path": path, "strength": 1.0}]) == len(trained_paths)
    assert backend._unload_lora_ltx2() == len(trained_paths)
    for target in trained_paths:
        assert dict(dit_b.named_modules())[target] is b_before[target], target
    assert not (module_ids(dit_b) & a_ids)


def test_ltx2_evicted_model_drops_the_bookkeeping_before_the_next_load(tmp_path):
    """An eviction (components cleared) must reset the maps, not park them for
    whatever transformer is loaded next -- including on the no-LoRA request that
    takes the empty-config path."""
    path, trained_paths = train_and_save(tmp_path)

    dit_a = _Dit()
    backend = _Backend(dit_a)
    backend._load_lora_ltx2([{"path": path, "strength": 1.0}])
    a_ids = module_ids(dit_a) | {id(m) for m in backend._ltx2_lora_original_modules.values()}

    backend.ltx2_components = {}
    backend._unload_lora_ltx2()

    dit_b = _Dit()
    b_ids_before = module_ids(dit_b)
    backend.ltx2_components = {"transformer": dit_b}
    backend._load_lora_ltx2([])  # a generation that installs no LoRA
    backend._unload_lora_ltx2()
    assert module_ids(dit_b) == b_ids_before
    assert not (module_ids(dit_b) & a_ids)
