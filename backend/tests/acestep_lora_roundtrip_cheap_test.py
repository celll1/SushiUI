"""ACE-Step 1.5: trainer save -> fresh-generation load round trip, CPU, ~1s.

Drives the REAL ``AceStepLoRAAdapter`` (injection + ``save_checkpoint``) over a
2-layer CPU stub, then the REAL ``AceStepMixin._load_lora_acestep``.

The Phase-0 defect this pins: ACE-Step can train an opt-in MLP scope, while
generation was attention-only, so the MLP half of a self-trained LoRA was
dropped without a word.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/acestep_lora_roundtrip_cheap_test.py -v
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
from core.pipeline_backends.acestep import AceStepMixin  # noqa: E402
from core.training.adapters.acestep_adapter import (  # noqa: E402
    DEFAULT_ACESTEP_SCOPE, AceStepLoRAAdapter, _flatten_to_sdscripts,
    iter_acestep_lora_targets,
)

H, I, N_LAYERS = 16, 32, 2
RANK = 4
ALPHA = 8  # != rank on purpose: the applied scale must be 2.0, not 1.0
SCALE = ALPHA / RANK
STRENGTH = 0.5
ATTN_AND_MLP = {"attention": True, "mlp": True}
MLP_ONLY = {"attention": False, "mlp": True}


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, name, nn.Linear(H, H, bias=False))


class _Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(H, I, bias=False)
        self.up_proj = nn.Linear(H, I, bias=False)
        self.down_proj = nn.Linear(I, H, bias=False)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attn()
        self.cross_attn = _Attn()
        self.mlp = _Mlp()


class _Decoder(nn.Module):
    def __init__(self, n=N_LAYERS):
        super().__init__()
        self.layers = nn.ModuleList([_Layer() for _ in range(n)])


class _Dit(nn.Module):
    def __init__(self, n=N_LAYERS):
        super().__init__()
        self.decoder = _Decoder(n)


class _Backend(AceStepMixin):
    def __init__(self, dit):
        self.acestep_components = {"dit": dit}
        self.is_acestep_model = True
        self.device = "cpu"


def wrapped_paths(model):
    return {name for name, module in model.named_modules()
            if isinstance(module, LoRALinearLayer)}


def train_and_save(tmp_path, scope=None, name="acestep.safetensors", seed=1234):
    scope = ATTN_AND_MLP if scope is None else scope
    dit = _Dit()
    adapter = AceStepLoRAAdapter(SimpleNamespace(transformer=dit, config={}),
                                 RANK, ALPHA, torch.float32, scope=scope)
    layers = {}
    count = adapter.apply_lora_to_unet(layers)
    assert adapter.apply_lora_to_text_encoders(layers) == 0, "the Qwen3 TE is frozen"
    assert count == len(layers) > 0
    randomise_lora_layers(layers, seed=seed, std=0.05)
    out = tmp_path / name
    adapter.save_checkpoint(layers, 1, 1, out)
    return str(out), wrapped_paths(dit)


@pytest.fixture
def warnings_seen(monkeypatch):
    return warning_probe(monkeypatch)


def test_acestep_generation_wraps_exactly_the_trained_attention_and_mlp_scope(tmp_path):
    """The headline fix: the opt-in MLP scope must survive to generation."""
    path, trained_paths = train_and_save(tmp_path)

    dit = _Dit()
    backend = _Backend(dit)
    backend._load_lora_acestep([{"path": path, "strength": STRENGTH}])

    assert wrapped_paths(dit) == trained_paths
    assert backend._acestep_lora_wrapped_modules == trained_paths
    assert any(".mlp." in p for p in trained_paths)
    assert trained_paths == {p for p, _parent, _attr, _cur
                             in iter_acestep_lora_targets(_Dit(), ATTN_AND_MLP)}
    # ...and the default TRAINING scope really is a strict subset, so this test
    # is about the wider scope rather than about attention twice.
    default_paths = {p for p, _parent, _attr, _cur
                     in iter_acestep_lora_targets(_Dit(), DEFAULT_ACESTEP_SCOPE)}
    assert default_paths < trained_paths


def test_acestep_wrapped_forward_is_base_plus_scaled_branch(tmp_path):
    path, trained_paths = train_and_save(tmp_path)
    saved = load_file(path)

    dit = _Dit()
    _Backend(dit)._load_lora_acestep([{"path": path, "strength": STRENGTH}])

    modules = dict(dit.named_modules())
    for target in sorted(trained_paths):
        wrapper = modules[target]
        stem = "lora_unet_" + _flatten_to_sdscripts(target)
        x = torch.randn(3, wrapper.original_module.in_features)
        base = wrapper.original_module(x)
        expected = base + lora_delta(saved[f"{stem}.lora_down.weight"],
                                     saved[f"{stem}.lora_up.weight"],
                                     x, ALPHA, RANK, STRENGTH)
        assert torch.allclose(wrapper(x), expected, atol=1e-5), target
        assert not torch.allclose(wrapper(x), base, atol=1e-5), f"{target}: branch is inert"


def test_acestep_alpha_beats_the_rank_fallback(tmp_path):
    path, trained_paths = train_and_save(tmp_path)
    dit = _Dit()
    _Backend(dit)._load_lora_acestep([{"path": path, "strength": STRENGTH}])
    modules = dict(dit.named_modules())
    assert {round(modules[t].scale, 9) for t in trained_paths} == {round(SCALE * STRENGTH, 9)}

    stripped = tmp_path / "no_alpha.safetensors"
    save_file({k: v for k, v in load_file(path).items() if not k.endswith(".alpha")},
              str(stripped), metadata={"model_type": "acestep"})
    dit2 = _Dit()
    _Backend(dit2)._load_lora_acestep([{"path": str(stripped), "strength": STRENGTH}])
    modules2 = dict(dit2.named_modules())
    assert {round(modules2[t].scale, 9) for t in trained_paths} == {round(STRENGTH, 9)}


def test_acestep_unload_restores_the_identical_objects_and_is_idempotent(tmp_path):
    path, trained_paths = train_and_save(tmp_path)

    dit = _Dit()
    before = dict(dit.named_modules())
    backend = _Backend(dit)
    backend._load_lora_acestep([{"path": path, "strength": 1.0}])

    backend._unload_lora_acestep()
    after = dict(dit.named_modules())
    for target in trained_paths:
        assert after[target] is before[target], target
    assert not wrapped_paths(dit)
    assert not backend._acestep_lora_wrapped_modules

    backend._unload_lora_acestep()  # second unload: no-op, not a re-splice
    assert dict(dit.named_modules()) == after


def test_acestep_missing_file_refuses(warnings_seen):
    with pytest.raises(ValidationError):
        _Backend(_Dit())._load_lora_acestep([{"path": "no_such_acestep_lora.safetensors"}])


@pytest.mark.xfail(reason="ACE-Step's missing-file branch raises ValidationError without "
                          "calling _acestep_lora_warn, so the refusal never reaches "
                          "warnings[]; its zero-target and stacking refusals do warn.")
def test_acestep_missing_file_warns(warnings_seen):
    with pytest.raises(ValidationError):
        _Backend(_Dit())._load_lora_acestep([{"path": "no_such_acestep_lora.safetensors"}])
    assert "lora_not_found" in warning_codes(warnings_seen)


def test_acestep_zero_matched_targets_refuses_and_warns(tmp_path, warnings_seen):
    ghost = tmp_path / "ghost.safetensors"
    stem = "lora_unet_decoder_layers_0_self_attn_norm"
    save_file({f"{stem}.lora_down.weight": torch.zeros(RANK, H),
               f"{stem}.lora_up.weight": torch.zeros(H, RANK)},
              str(ghost), metadata={"model_type": "acestep"})

    dit = _Dit()
    with pytest.raises(ValidationError):
        _Backend(dit)._load_lora_acestep([{"path": str(ghost), "strength": 1.0}])
    assert "lora_incompatible" in warning_codes(warnings_seen)
    assert not wrapped_paths(dit)


def test_acestep_fully_shadowed_second_lora_refuses_and_warns(tmp_path, warnings_seen):
    first, _paths = train_and_save(tmp_path)
    second, _paths2 = train_and_save(tmp_path, name="second.safetensors", seed=99)

    with pytest.raises(ValidationError):
        _Backend(_Dit())._load_lora_acestep([{"path": first, "strength": 1.0},
                                             {"path": second, "strength": 1.0}])
    assert "lora_stacking_unsupported" in warning_codes(warnings_seen)


def test_acestep_disjoint_scopes_stack_additively(tmp_path, warnings_seen):
    attn, attn_paths = train_and_save(tmp_path, scope=DEFAULT_ACESTEP_SCOPE,
                                      name="attn.safetensors")
    mlp, mlp_paths = train_and_save(tmp_path, scope=MLP_ONLY, name="mlp.safetensors", seed=7)
    assert not (attn_paths & mlp_paths)

    dit = _Dit()
    backend = _Backend(dit)
    backend._load_lora_acestep([{"path": attn}, {"path": mlp}])
    assert wrapped_paths(dit) == attn_paths | mlp_paths
    assert warning_codes(warnings_seen) == []


@pytest.mark.xfail(strict=True, reason=(
    "OPEN DEFECT: _acestep_lora_original_modules/_acestep_lora_wrapped_modules are plain "
    "attributes with no ownership key, so they outlive a model swap and "
    "_unload_lora_acestep re-resolves each stale path against the NEW dit and installs "
    "model A's Linears into model B (measured: 22 of 22 targets). Every other "
    "architecture keys this state to the live component; ACE-Step does not. Remove this "
    "marker with the fix."))
def test_acestep_model_reload_never_splices_model_a_into_model_b(tmp_path):
    path, trained_paths = train_and_save(tmp_path)

    dit_a = _Dit()
    backend = _Backend(dit_a)
    backend._load_lora_acestep([{"path": path, "strength": 1.0}])
    a_ids = module_ids(dit_a) | {id(m) for m in backend._acestep_lora_original_modules.values()}

    dit_b = _Dit()
    b_ids_before = module_ids(dit_b)
    assert not (a_ids & b_ids_before), "setup: A and B must not already share modules"

    backend.acestep_components = {"dit": dit_b}
    assert backend._acestep_lora_wrapped_modules, "the stale set must be truthy to be a test"
    backend._unload_lora_acestep()
    assert module_ids(dit_b) == b_ids_before, "model B's module graph was modified"
    assert not (module_ids(dit_b) & a_ids), "a module of model A was installed into model B"

    b_before = dict(dit_b.named_modules())
    backend._load_lora_acestep([{"path": path, "strength": 1.0}])
    assert wrapped_paths(dit_b) == trained_paths
    backend._unload_lora_acestep()
    for target in trained_paths:
        assert dict(dit_b.named_modules())[target] is b_before[target], target
    assert not (module_ids(dit_b) & a_ids)
