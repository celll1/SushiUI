"""Anima: trainer save -> fresh-generation load round trip, on CPU in ~1s.

Drives the REAL ``AnimaLoRAAdapter`` (injection + ``save_checkpoint``) over a
2-block CPU stub and then the REAL ``AnimaMixin._load_lora_anima``. The stub's
CLASS NAMES are load-bearing: ``iter_anima_lora_targets`` selects by them.

The Phase-0 defect this pins: Anima's default TRAINING scope covers attention,
MLP and the LLM adapter, while generation applied only its attention iterator,
so the MLP and llm_adapter halves of a self-trained LoRA were silently dropped.

Common composite algebra is covered in
``adapter_lycoris_roundtrip_cheap_test.py``. This file retains Anima's int-slot
targeting, interchange formats, lifecycle, and refusal seams.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/anima_lora_roundtrip_cheap_test.py -v
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
from core.models.anima import anima_lora as anima_mod  # noqa: E402
from core.pipeline_backends.anima import AnimaMixin  # noqa: E402
from core.training.adapters.anima_adapter import AnimaLoRAAdapter  # noqa: E402

D = 8
RANK = 4
# alpha/rank = 1.5 and strength 0.7: the previous alpha=8/strength=0.5 gave a
# scale of exactly 1.0, where every plausible reassociation of the strength fold
# is identical in IEEE754 and the bit-identity gate cannot bite.
ALPHA = 6.0
SCALE = ALPHA / RANK
STRENGTH = 0.7
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


def train_and_save(tmp_path, scope=None, name="anima.safetensors", seed=1234):
    scope = anima_mod.DEFAULT_TRAINING_SCOPE if scope is None else scope
    model = build_model()
    trainer = SimpleNamespace(transformer=model, blockskip_config=None, config={})
    adapter = AnimaLoRAAdapter(trainer, RANK, ALPHA, torch.float32, scope=scope)
    layers = {}
    count = adapter.apply_lora_to_unet(layers)
    assert count == len(layers) > 0
    randomise_lora_layers(layers, seed=seed)
    out = tmp_path / name
    adapter.save_checkpoint(layers, 1, 1, out)
    return str(out), lora_layer_paths(model)


def file_branch_tensors(path, target):
    """``(down, up)`` straight out of the checkpoint, for the analytic sum."""
    saved = load_file(path)
    stem = "lora_unet_" + anima_mod._flatten_to_sdscripts(target)
    return saved[f"{stem}.lora_down.weight"], saved[f"{stem}.lora_up.weight"]


@pytest.fixture
def warnings_seen(monkeypatch):
    return warning_probe(monkeypatch)


def test_anima_generation_wraps_exactly_the_default_training_scope(tmp_path):
    """The headline fix: the MLP and llm_adapter halves must survive to
    generation, not only the attention iterator."""
    path, trained_paths = train_and_save(tmp_path)

    model = build_model()
    backend = _Backend(model)
    applied = backend._load_lora_anima([{"path": path, "strength": STRENGTH}])

    assert wrapped_paths(model) == trained_paths
    assert applied == len(trained_paths)
    assert any(".mlp." in p for p in trained_paths)
    assert any(p.startswith("llm_adapter.") for p in trained_paths)
    assert trained_paths == {p for p, _parent, _attr, _cur in anima_mod.iter_anima_lora_targets(
        build_model(), anima_mod.DEFAULT_TRAINING_SCOPE)}


def test_anima_wrapped_forward_is_base_plus_scaled_branch(tmp_path):
    path, trained_paths = train_and_save(tmp_path)

    model = build_model()
    _Backend(model)._load_lora_anima([{"path": path, "strength": STRENGTH}])

    modules = dict(model.named_modules())
    for target in sorted(trained_paths):
        wrapper = modules[target]
        down, up = file_branch_tensors(path, target)
        x = torch.randn(3, wrapper.original_module.in_features)
        base = wrapper.original_module(x)
        expected = base + lora_delta(down, up, x, ALPHA, RANK, STRENGTH)
        assert torch.allclose(wrapper(x), expected, atol=1e-5), target
        assert not torch.allclose(wrapper(x), base, atol=1e-5), f"{target}: branch is inert"


def test_anima_missing_file_refuses_and_warns(warnings_seen):
    with pytest.raises(RuntimeError):
        _Backend(build_model())._load_lora_anima([{"path": "no_such_anima_lora.safetensors"}])
    assert warning_codes(warnings_seen) == ["lora_not_found"]


def test_anima_unreadable_file_refuses_and_warns(tmp_path, warnings_seen):
    broken = tmp_path / "broken.safetensors"
    broken.write_bytes(b"not a safetensors file")
    with pytest.raises(RuntimeError):
        _Backend(build_model())._load_lora_anima([{"path": str(broken)}])
    assert "lora_load_failed" in warning_codes(warnings_seen)


def test_anima_zero_matched_targets_refuses_and_warns(tmp_path, warnings_seen):
    ghost = tmp_path / "ghost.safetensors"
    save_file({"lora_unet_blocks_9_self_attn_q_proj.lora_down.weight": torch.zeros(RANK, D),
               "lora_unet_blocks_9_self_attn_q_proj.lora_up.weight": torch.zeros(D, RANK)},
              str(ghost), metadata={"model_type": "anima"})

    model = build_model()
    with pytest.raises(RuntimeError):
        _Backend(model)._load_lora_anima([{"path": str(ghost), "strength": 1.0}])
    assert "lora_incompatible" in warning_codes(warnings_seen)
    assert not wrapped_paths(model)


def test_anima_partly_matching_file_is_refused_atomically(tmp_path, warnings_seen):
    path, trained_paths = train_and_save(tmp_path, scope=ATTENTION_ONLY)
    saved = load_file(path)
    saved["lora_unet_blocks_9_self_attn_q_proj.lora_down.weight"] = torch.zeros(RANK, D)
    saved["lora_unet_blocks_9_self_attn_q_proj.lora_up.weight"] = torch.zeros(D, RANK)
    partial = tmp_path / "partial.safetensors"
    save_file(saved, str(partial), metadata={"model_type": "anima"})

    model = build_model()
    with pytest.raises(AdapterIncompatible) as excinfo:
        _Backend(model)._load_lora_anima([{"path": str(partial)}])
    assert excinfo.value.code == "lora_partial"
    assert not wrapped_paths(model)
    assert "lora_partial" in warning_codes(warnings_seen)


def test_anima_two_loras_over_the_same_targets_stack_instead_of_refusing(tmp_path,
                                                                        warnings_seen):
    """The refusal this file used to assert. The same file twice is two branches,
    not a duplicate-name error, because branch names carry the request index."""
    path, trained_paths = train_and_save(tmp_path, scope=ATTENTION_ONLY)

    model = build_model()
    backend = _Backend(model)
    assert backend._load_lora_anima([{"path": path}, {"path": path}]) == 2 * len(trained_paths)
    assert wrapped_paths(model) == trained_paths
    assert warning_codes(warnings_seen) == []
    for target in trained_paths:
        assert len(dict(model.named_modules())[target]) == 2, target


def test_anima_disjoint_scopes_stack_additively(tmp_path, warnings_seen):
    """Two LoRAs over disjoint scopes: one composite each, no overlap."""
    attn, attn_paths = train_and_save(tmp_path, scope=ATTENTION_ONLY, name="attn.safetensors")
    mlp, mlp_paths = train_and_save(tmp_path, scope=MLP_ONLY, name="mlp.safetensors", seed=7)
    assert not (attn_paths & mlp_paths)

    model = build_model()
    backend = _Backend(model)
    total = backend._load_lora_anima([{"path": attn}, {"path": mlp}])
    assert total == len(attn_paths) + len(mlp_paths)
    assert wrapped_paths(model) == attn_paths | mlp_paths
    assert warning_codes(warnings_seen) == []
    assert backend._unload_lora_anima() == len(attn_paths | mlp_paths)


def test_anima_a_second_load_does_not_target_the_first_branch_int_slots(tmp_path):
    """`adaln_modulation_mlp.1.branches.1` ends in "1" exactly like the target it
    sits under, and the llm_adapter MLP slots are "0"/"2". The path-shape pass
    must not follow a composite's branch list, or a stack would enumerate targets
    INSIDE the adapter it just installed."""
    path, trained_paths = train_and_save(tmp_path, scope=anima_mod.FULL_SCOPE)

    model = build_model()
    _Backend(model)._load_lora_anima([{"path": path}, {"path": path}])

    enumerated = {p for p, _parent, _attr, _cur
                  in anima_mod.iter_anima_lora_targets(model, anima_mod.FULL_SCOPE)}
    assert enumerated == trained_paths, sorted(enumerated - trained_paths)[:5]


def test_anima_a_refused_file_leaves_the_ones_before_it_uninstalled(tmp_path,
                                                                   warnings_seen):
    """``AdapterSession`` plans the WHOLE request before mutating a slot.

    Anima used to wrap file by file and unwrap again at the end, so between the
    two a refused request ran its restore over a DiT it had just wrapped; now
    nothing is installed at all.
    """
    path, _trained_paths = train_and_save(tmp_path, scope=ATTENTION_ONLY)
    ghost = tmp_path / "ghost.safetensors"
    save_file({"lora_unet_blocks_9_self_attn_q_proj.lora_down.weight": torch.zeros(RANK, D),
               "lora_unet_blocks_9_self_attn_q_proj.lora_up.weight": torch.zeros(D, RANK)},
              str(ghost), metadata={"model_type": "anima"})

    model = build_model()
    before = dict(model.named_modules())
    backend = _Backend(model)
    with pytest.raises(RuntimeError):
        backend._load_lora_anima([{"path": path}, {"path": str(ghost)}])

    assert not wrapped_paths(model)
    assert dict(model.named_modules()) == before
    assert not backend._anima_lora_wrapped_keys
    assert not backend._anima_lora_original_modules
    assert "lora_incompatible" in warning_codes(warnings_seen)


def test_anima_a_narrow_checkpoint_wraps_only_the_targets_it_names(tmp_path,
                                                                  warnings_seen):
    """One enumerator, over FULL_SCOPE, on both the load and the unload path.

    It replaced a per-file scope derived from the checkpoint's keys, and is
    equivalent only because application is lookup-driven: an mlp-only file must
    still wrap the mlp targets and nothing else, and must not read as partial.
    """
    path, mlp_paths = train_and_save(tmp_path, scope=MLP_ONLY)
    assert mlp_paths and all(".mlp." in p for p in mlp_paths)

    model = build_model()
    backend = _Backend(model)
    assert backend._load_lora_anima([{"path": path}]) == len(mlp_paths)
    assert wrapped_paths(model) == mlp_paths
    assert warning_codes(warnings_seen) == []
    assert backend._unload_lora_anima() == len(mlp_paths)



def interchange_copy(path, out_path):
    """The same tensors, spelled the way third-party tooling exports them."""
    remapped = {}
    suffixes = {"lora_down.weight": "lora_A.weight",
                "lora_up.weight": "lora_B.weight", "alpha": "alpha"}
    for key, tensor in load_file(path).items():
        module_path, tag = anima_mod._parse_key(key)
        remapped[f"{anima_mod.INTERCHANGE_DIT_PREFIX}{module_path}.{suffixes[tag]}"] = tensor
    save_file(remapped, str(out_path), metadata={"model_type": "anima"})
    return str(out_path)


def test_anima_an_interchange_format_lora_applies_exactly_like_the_native_one(
        tmp_path, warnings_seen):
    """Anima's parser reads `lora_A`/`lora_B` itself, so a shared-layer rewrite
    of those suffixes leaves it parsing nothing and refusing a valid file."""
    native, trained_paths = train_and_save(tmp_path, scope=anima_mod.FULL_SCOPE)
    foreign = interchange_copy(native, tmp_path / "interchange.safetensors")

    model = build_model()
    backend = _Backend(model)
    applied = backend._load_lora_anima([{"path": foreign, "strength": STRENGTH}])

    assert applied == len(trained_paths)
    assert wrapped_paths(model) == trained_paths
    assert warning_codes(warnings_seen) == []

    modules = dict(model.named_modules())
    for target in sorted(trained_paths):
        composite = modules[target]
        base = composite.original_module
        # From the NATIVE file: `lora_A` must have landed as down, not swapped.
        down, up = file_branch_tensors(native, target)
        x = torch.randn(3, base.in_features)
        expected = base(x) + lora_delta(down, up, x, ALPHA, RANK, STRENGTH)
        assert torch.allclose(composite(x), expected, atol=1e-5), target

    assert backend._unload_lora_anima() == len(trained_paths)


def test_anima_declares_the_interchange_pairs_it_is_about_to_apply(tmp_path):
    """The declared count is what `lora_partial` compares against; counting
    `lora_down` keys alone reads an interchange file as zero pairs."""
    native, trained_paths = train_and_save(tmp_path, scope=anima_mod.FULL_SCOPE)
    foreign = interchange_copy(native, tmp_path / "ix.safetensors")

    declared = AnimaMixin._anima_declared_branches(load_file(foreign), ("transformer",))
    assert declared == len(trained_paths)


def test_anima_a_peft_export_with_bias_tensors_loads_instead_of_crashing(
        tmp_path, warnings_seen):
    """`lora_bias=True` exports carry a 1-D `.lora_A.bias` that the session's
    codec sniff indexes as 2-D. The sniff is advisory and runs outside every
    try/except on the load path, so an unguarded one turns this file's clean
    outcome into an unhandled 500."""
    native, trained_paths = train_and_save(tmp_path, scope=ATTENTION_ONLY)
    saved = load_file(interchange_copy(native, tmp_path / "ix.safetensors"))
    for key in [k for k in saved if k.endswith(".lora_A.weight")]:
        saved[key[: -len(".weight")] + ".bias"] = torch.zeros(RANK)
    biased = tmp_path / "with_bias.safetensors"
    save_file(saved, str(biased), metadata={"model_type": "anima"})

    model = build_model()
    backend = _Backend(model)
    applied = backend._load_lora_anima([{"path": str(biased), "strength": STRENGTH}])

    assert applied == len(trained_paths)
    assert wrapped_paths(model) == trained_paths
    # The bias tensors themselves are not applied, and Anima says so.
    assert warning_codes(warnings_seen) == ["anima_lora_keys_unrecognised"]
