"""Ideogram 4: trainer save -> fresh-generation load round trip, CPU, ~1s.

Drives the REAL ``Ideogram4LoRAAdapter`` (injection + ``save_checkpoint``) over
2-block CPU stubs and the REAL ``Ideogram4Mixin._load_lora_ideogram4``.

Ideogram 4 is the only architecture with two independently wrapped
transformers: the conditional branch and the unconditional one (asymmetric
CFG). Its checkpoint namespaces them ``lora_unet_*`` / ``lora_uncond_*``, and
each branch's bookkeeping is reset independently, so the reload gate here
covers a partial reload as well as a full one.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/ideogram4_lora_roundtrip_cheap_test.py -v
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

from core.models.ideogram4.ideogram4_lora import (  # noqa: E402
    DEFAULT_SCOPE, _flatten_to_sdscripts, iter_ideogram4_lora_targets,
    normalise_lora_state_dict,
)
from core.pipeline_backends.ideogram4 import Ideogram4Mixin  # noqa: E402
from core.training.adapters.ideogram4_adapter import Ideogram4LoRAAdapter  # noqa: E402

D = 8
RANK = 4
ALPHA = 8  # != rank on purpose: the applied scale must be 2.0, not 1.0
SCALE = ALPHA / RANK
STRENGTH = 0.5


def _linear():
    layer = nn.Linear(D, D)
    nn.init.normal_(layer.weight, std=0.05)
    nn.init.normal_(layer.bias, std=0.05)
    return layer


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q, self.to_k, self.to_v = _linear(), _linear(), _linear()
        self.to_out = nn.ModuleList([_linear()])


class _FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1, self.w2, self.w3 = _linear(), _linear(), _linear()


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = _Attn()
        self.feed_forward = _FeedForward()
        self.adaln_modulation = _linear()


class _Stub(nn.Module):
    def __init__(self, n_blocks=2):
        super().__init__()
        self.layers = nn.ModuleList([_Layer() for _ in range(n_blocks)])


class _Backend(Ideogram4Mixin):
    def __init__(self, transformer, uncond=None):
        self.ideogram4_components = {"transformer": transformer}
        if uncond is not None:
            self.ideogram4_components["unconditional_transformer"] = uncond


def wrapped_paths(model):
    return {name for name, module in model.named_modules()
            if isinstance(module, LoRALinearLayer)}


def train_and_save(tmp_path, name="ideogram4.safetensors", seed=1234, with_uncond=False):
    """Returns (path, cond target paths, uncond target paths)."""
    cond = _Stub()
    uncond = _Stub() if with_uncond else None
    trainer = SimpleNamespace(transformer=cond, transformer_uncond=uncond,
                              ideogram4_train_uncond=with_uncond, config={})
    adapter = Ideogram4LoRAAdapter(trainer, RANK, ALPHA, torch.float32)
    layers = {}
    count = adapter.apply_lora_to_unet(layers)
    assert adapter.apply_lora_to_text_encoders(layers) == 0, "Qwen3-VL is frozen"
    assert count == len(layers) > 0
    randomise_lora_layers(layers, seed=seed, std=0.3)
    out = tmp_path / name
    adapter.save_checkpoint(layers, 7, 1, out)
    return (str(out), wrapped_paths(cond),
            wrapped_paths(uncond) if uncond is not None else set())


@pytest.fixture
def warnings_seen(monkeypatch):
    return warning_probe(monkeypatch)


def test_ideogram4_generation_wraps_exactly_the_targets_the_trainer_wrapped(tmp_path):
    path, trained_paths, _uncond = train_and_save(tmp_path)

    model = _Stub()
    backend = _Backend(model)
    applied = backend._load_lora_ideogram4([{"path": path, "strength": STRENGTH}])

    assert wrapped_paths(model) == trained_paths
    assert applied == len(trained_paths)
    assert backend._ideogram4_lora_keys == trained_paths
    assert trained_paths == {p for p, _parent, _attr, _cur
                             in iter_ideogram4_lora_targets(_Stub(), DEFAULT_SCOPE)}
    assert any(p.endswith(".attention.to_out.0") for p in trained_paths)


def test_ideogram4_uncond_branch_round_trips_onto_its_own_transformer(tmp_path):
    path, cond_paths, uncond_paths = train_and_save(tmp_path, with_uncond=True)
    assert cond_paths and uncond_paths

    raw = load_file(path)
    assert len(normalise_lora_state_dict(raw, branch="cond")) == len(cond_paths)
    assert len(normalise_lora_state_dict(raw, branch="uncond")) == len(uncond_paths)

    cond, uncond = _Stub(), _Stub()
    backend = _Backend(cond, uncond)
    applied = backend._load_lora_ideogram4([{"path": path, "strength": 1.0}])
    assert applied == len(cond_paths) + len(uncond_paths)
    assert wrapped_paths(cond) == cond_paths
    assert wrapped_paths(uncond) == uncond_paths

    # A cond-only checkpoint must leave the uncond branch alone.
    cond_only, _cp, _up = train_and_save(tmp_path, name="cond_only.safetensors")
    cond2, uncond2 = _Stub(), _Stub()
    _Backend(cond2, uncond2)._load_lora_ideogram4([{"path": cond_only, "strength": 1.0}])
    assert wrapped_paths(cond2) == cond_paths
    assert not wrapped_paths(uncond2)


def test_ideogram4_wrapped_forward_is_base_plus_scaled_branch(tmp_path):
    path, trained_paths, _uncond = train_and_save(tmp_path)
    saved = load_file(path)

    model = _Stub()
    _Backend(model)._load_lora_ideogram4([{"path": path, "strength": STRENGTH}])

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


def test_ideogram4_alpha_beats_the_rank_fallback(tmp_path):
    path, trained_paths, _uncond = train_and_save(tmp_path)
    model = _Stub()
    _Backend(model)._load_lora_ideogram4([{"path": path, "strength": STRENGTH}])
    modules = dict(model.named_modules())
    assert {round(modules[t].scale, 9) for t in trained_paths} == {round(SCALE * STRENGTH, 9)}

    md_only = tmp_path / "md_alpha.safetensors"
    save_file({k: v for k, v in load_file(path).items() if not k.endswith(".alpha")},
              str(md_only), metadata={"model_type": "ideogram4", "lora_alpha": str(4 * RANK)})
    model2 = _Stub()
    _Backend(model2)._load_lora_ideogram4([{"path": str(md_only), "strength": 1.0}])
    modules2 = dict(model2.named_modules())
    assert {round(modules2[t].scale, 9) for t in trained_paths} == {4.0}

    none = tmp_path / "no_alpha.safetensors"
    save_file({k: v for k, v in load_file(path).items() if not k.endswith(".alpha")},
              str(none), metadata={"model_type": "ideogram4"})
    model3 = _Stub()
    _Backend(model3)._load_lora_ideogram4([{"path": str(none), "strength": 1.0}])
    modules3 = dict(model3.named_modules())
    assert {round(modules3[t].scale, 9) for t in trained_paths} == {1.0}


def test_ideogram4_unload_restores_the_identical_objects_and_is_idempotent(tmp_path):
    path, cond_paths, uncond_paths = train_and_save(tmp_path, with_uncond=True)

    cond, uncond = _Stub(), _Stub()
    before = {**dict(cond.named_modules()),
              **{f"u::{n}": m for n, m in uncond.named_modules()}}
    backend = _Backend(cond, uncond)
    backend._load_lora_ideogram4([{"path": path, "strength": 1.0}])

    assert backend._unload_lora_ideogram4() == len(cond_paths) + len(uncond_paths)
    for target in cond_paths:
        assert dict(cond.named_modules())[target] is before[target], target
    for target in uncond_paths:
        assert dict(uncond.named_modules())[target] is before[f"u::{target}"], target
    assert not wrapped_paths(cond) and not wrapped_paths(uncond)
    assert backend._unload_lora_ideogram4() == 0


def test_ideogram4_cleanup_unwraps_on_a_path_that_never_denoised(tmp_path):
    """`_ideogram4_cleanup` is the finally of every generate entry point."""
    path, trained_paths, _uncond = train_and_save(tmp_path)
    model = _Stub()
    backend = _Backend(model)
    backend._load_lora_ideogram4([{"path": path, "strength": 1.0}])
    backend._ideogram4_cleanup(model_key=None, gen_succeeded=False)
    assert not wrapped_paths(model)


def test_ideogram4_missing_file_refuses_and_warns(warnings_seen):
    with pytest.raises(FileNotFoundError, match="not found"):
        _Backend(_Stub())._load_lora_ideogram4([{"path": "no_such_i4_lora.safetensors"}])
    assert "lora_not_found" in warning_codes(warnings_seen)


def test_ideogram4_zero_matched_targets_refuses_and_warns(tmp_path, warnings_seen):
    foreign = tmp_path / "foreign.safetensors"
    save_file({"totally.unrelated.weight": torch.zeros(2, 2)}, str(foreign),
              metadata={"model_type": "not_ideogram4"})

    model = _Stub()
    with pytest.raises(RuntimeError):
        _Backend(model)._load_lora_ideogram4([{"path": str(foreign), "strength": 1.0}])
    assert "lora_incompatible" in warning_codes(warnings_seen)
    assert not wrapped_paths(model)


def test_ideogram4_uncond_only_file_with_no_uncond_branch_refuses(tmp_path, warnings_seen):
    """A dedicated code: this is not an unrecognized key format."""
    path, _cond, _uncond = train_and_save(tmp_path, with_uncond=True)
    uncond_only = tmp_path / "uncond_only.safetensors"
    raw = load_file(path)
    save_file({k: v for k, v in raw.items() if k.startswith("lora_uncond_")},
              str(uncond_only), metadata={"model_type": "ideogram4"})

    with pytest.raises(RuntimeError):
        _Backend(_Stub())._load_lora_ideogram4([{"path": str(uncond_only), "strength": 1.0}])
    assert "lora_uncond_unavailable" in warning_codes(warnings_seen)


def test_ideogram4_fully_shadowed_second_lora_refuses_and_warns(tmp_path, warnings_seen):
    path, _cond, _uncond = train_and_save(tmp_path)
    second, _c2, _u2 = train_and_save(tmp_path, name="second.safetensors", seed=99)

    with pytest.raises(RuntimeError):
        _Backend(_Stub())._load_lora_ideogram4([{"path": path, "strength": 1.0},
                                                {"path": second, "strength": 1.0}])
    assert "lora_stacking_unsupported" in warning_codes(warnings_seen)


def test_ideogram4_model_reload_never_splices_model_a_into_model_b(tmp_path):
    path, cond_paths, uncond_paths = train_and_save(tmp_path, with_uncond=True)

    cond_a, uncond_a = _Stub(), _Stub()
    backend = _Backend(cond_a, uncond_a)
    backend._load_lora_ideogram4([{"path": path, "strength": 1.0}])
    a_ids = (module_ids(cond_a) | module_ids(uncond_a)
             | {id(m) for m in backend._ideogram4_lora_orig.values()}
             | {id(m) for m in backend._ideogram4_lora_orig_uncond.values()})

    cond_b, uncond_b = _Stub(), _Stub()
    b_ids_before = module_ids(cond_b) | module_ids(uncond_b)
    assert not (a_ids & b_ids_before), "setup: A and B must not already share modules"

    backend.ideogram4_components = {"transformer": cond_b,
                                    "unconditional_transformer": uncond_b}
    assert backend._ideogram4_lora_keys, "the stale set must be truthy to be a test"
    assert backend._unload_lora_ideogram4() == 0
    assert module_ids(cond_b) | module_ids(uncond_b) == b_ids_before
    assert not ((module_ids(cond_b) | module_ids(uncond_b)) & a_ids)


def test_ideogram4_reloading_only_the_uncond_branch_keeps_the_cond_half(tmp_path):
    """Per-branch bookkeeping: swapping one transformer must not throw away the
    other branch's restore, nor splice the old branch into the new one."""
    path, cond_paths, uncond_paths = train_and_save(tmp_path, with_uncond=True)

    cond, uncond_a = _Stub(), _Stub()
    backend = _Backend(cond, uncond_a)
    backend._load_lora_ideogram4([{"path": path, "strength": 1.0}])
    old_uncond_ids = module_ids(uncond_a) | {
        id(m) for m in backend._ideogram4_lora_orig_uncond.values()}

    uncond_b = _Stub()
    b_ids_before = module_ids(uncond_b)
    backend.ideogram4_components["unconditional_transformer"] = uncond_b

    assert backend._unload_lora_ideogram4() == len(cond_paths)
    assert not wrapped_paths(cond)
    assert module_ids(uncond_b) == b_ids_before
    assert not (module_ids(uncond_b) & old_uncond_ids)
