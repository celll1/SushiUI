"""Anima: trainer save -> fresh-generation load round trip, on CPU in ~1s.

Drives the REAL ``AnimaLoRAAdapter`` (injection + ``save_checkpoint``) over a
2-block CPU stub and then the REAL ``AnimaMixin._load_lora_anima``. The stub's
CLASS NAMES are load-bearing: ``iter_anima_lora_targets`` selects by them.

The Phase-0 defect this pins: Anima's default TRAINING scope covers attention,
MLP and the LLM adapter, while generation applied only its attention iterator,
so the MLP and llm_adapter halves of a self-trained LoRA were silently dropped.

Anima is on ``CompositeAdapterLayer``, so this file is also its adoption gate:
two LoRAs over one module must SUM, in either selection order, without
perturbing what either does alone. The stacking refusal these tests used to
assert is gone; the numerics that replace it are checked with ``torch.equal``,
because a tolerance would hide exactly the reassociation a "simplification" of
the strength folding would introduce. Anima's INT SLOTS (the adaln_modulation_*
and llm_adapter MLP targets inside an ``nn.Sequential``) are what makes the
restore-by-identity gate here worth more than a count.

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

from core.adapters import CompositeAdapterLayer  # noqa: E402
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
STRENGTH_B = 0.4
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


def sole_branch(composite):
    assert len(composite) == 1, f"expected one branch, got {composite.branch_names}"
    return composite.get_branch(composite.branch_names[0])


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


def reference_wrapper(base, down, up, target, strength):
    """What the pre-composite loader installed, rebuilt from the file's tensors."""
    wrapper = LoRALinearLayer(base, rank=RANK, alpha=float(ALPHA), lora_name=target)
    compute_dtype = base.weight.dtype
    with torch.no_grad():
        wrapper.lora_down.weight.data = down.to(device=base.weight.device,
                                                dtype=compute_dtype)
        wrapper.lora_up.weight.data = up.to(device=base.weight.device,
                                            dtype=compute_dtype)
    wrapper.lora_down = wrapper.lora_down.to(dtype=compute_dtype)
    wrapper.lora_up = wrapper.lora_up.to(dtype=compute_dtype)
    wrapper.scale = (float(ALPHA) / RANK) * strength
    return wrapper


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


def test_anima_single_lora_is_bit_identical_to_the_pre_composite_wrapper(tmp_path):
    """The composite must not change one-LoRA arithmetic by one ULP.

    ``torch.equal``, not a tolerance -- folding the strength anywhere but into
    the branch's own scale reassociates the multiply and shows up here alone.
    The `mod` scope is included, so the int-slot targets are covered too.
    """
    path, trained_paths = train_and_save(tmp_path, scope=anima_mod.FULL_SCOPE)
    assert any(".adaln_modulation_" in p for p in trained_paths), "int slots must be in scope"

    model = build_model()
    _Backend(model)._load_lora_anima([{"path": path, "strength": STRENGTH}])

    modules = dict(model.named_modules())
    for target in sorted(trained_paths):
        composite = modules[target]
        base = composite.original_module
        down, up = file_branch_tensors(path, target)
        reference = reference_wrapper(base, down, up, target, STRENGTH)

        assert sole_branch(composite).scale == reference.scale, target
        x = torch.randn(3, base.in_features)
        assert torch.equal(composite(x), reference(x)), target


def test_anima_two_loras_over_one_module_sum_their_deltas(tmp_path):
    """The adoption gate: base + delta_a + delta_b, against the analytic sum."""
    path_a, trained_paths = train_and_save(tmp_path, scope=anima_mod.FULL_SCOPE, seed=1234)
    path_b, paths_b = train_and_save(tmp_path, scope=anima_mod.FULL_SCOPE,
                                     name="second.safetensors", seed=4321)
    assert paths_b == trained_paths, "both files must cover the same targets to stack"

    model = build_model()
    backend = _Backend(model)
    total = backend._load_lora_anima([{"path": path_a, "strength": STRENGTH},
                                      {"path": path_b, "strength": STRENGTH_B}])
    assert total == 2 * len(trained_paths)
    assert wrapped_paths(model) == trained_paths

    modules = dict(model.named_modules())
    for target in sorted(trained_paths):
        composite = modules[target]
        assert len(composite) == 2, f"{target}: {composite.branch_names}"
        base_module = composite.original_module
        down_a, up_a = file_branch_tensors(path_a, target)
        down_b, up_b = file_branch_tensors(path_b, target)
        x = torch.randn(3, base_module.in_features)
        expected = (base_module(x)
                    + lora_delta(down_a, up_a, x, ALPHA, RANK, STRENGTH)
                    + lora_delta(down_b, up_b, x, ALPHA, RANK, STRENGTH_B))
        assert torch.allclose(composite(x), expected, atol=1e-5), target
        # Both branches really contribute: dropping either changes the output.
        assert not torch.allclose(
            composite(x),
            base_module(x) + lora_delta(down_a, up_a, x, ALPHA, RANK, STRENGTH),
            atol=1e-5), f"{target}: the second LoRA is inert"


def test_anima_stacked_result_is_independent_of_selection_order(tmp_path):
    path_a, trained_paths = train_and_save(tmp_path, scope=anima_mod.FULL_SCOPE, seed=1234)
    path_b, _pb = train_and_save(tmp_path, scope=anima_mod.FULL_SCOPE,
                                 name="second.safetensors", seed=4321)
    a = {"path": path_a, "strength": STRENGTH}
    b = {"path": path_b, "strength": STRENGTH_B}

    forward = build_model()
    _Backend(forward)._load_lora_anima([a, b])
    reverse = build_model()
    _Backend(reverse)._load_lora_anima([b, a])

    forward_modules = dict(forward.named_modules())
    reverse_modules = dict(reverse.named_modules())
    for target in sorted(trained_paths):
        one, two = forward_modules[target], reverse_modules[target]
        assert torch.equal(one.original_module.weight, two.original_module.weight), target
        x = torch.randn(3, one.original_module.in_features)
        # Two branches: the deltas are summed before the base is added, and fp
        # addition commutes, so this is EXACT. (Three or more would only hold up
        # to associativity.)
        assert torch.equal(one(x), two(x)), target


def test_anima_removing_one_branch_leaves_the_other_exactly_as_if_alone(tmp_path):
    """A stacked branch must not perturb its neighbour's own arithmetic."""
    path_a, trained_paths = train_and_save(tmp_path, scope=anima_mod.FULL_SCOPE, seed=1234)
    path_b, _pb = train_and_save(tmp_path, scope=anima_mod.FULL_SCOPE,
                                 name="second.safetensors", seed=4321)

    alone = build_model()
    _Backend(alone)._load_lora_anima([{"path": path_a, "strength": STRENGTH}])

    stacked = build_model()
    _Backend(stacked)._load_lora_anima([{"path": path_a, "strength": STRENGTH},
                                        {"path": path_b, "strength": STRENGTH_B}])

    alone_modules = dict(alone.named_modules())
    stacked_modules = dict(stacked.named_modules())
    for target in sorted(trained_paths):
        one = alone_modules[target]
        two = stacked_modules[target]
        assert torch.equal(one.original_module.weight, two.original_module.weight), target
        two.remove_branch(two.branch_names[1])
        assert two.branch_names == one.branch_names, target
        x = torch.randn(3, one.original_module.in_features)
        assert torch.equal(one(x), two(x)), target


def test_anima_alpha_beats_the_rank_fallback(tmp_path):
    path, trained_paths = train_and_save(tmp_path)
    model = build_model()
    _Backend(model)._load_lora_anima([{"path": path, "strength": STRENGTH}])
    modules = dict(model.named_modules())
    assert {round(sole_branch(modules[t]).scale, 9) for t in trained_paths} == \
        {round(SCALE * STRENGTH, 9)}

    stripped = tmp_path / "no_alpha.safetensors"
    save_file({k: v for k, v in load_file(path).items() if not k.endswith(".alpha")},
              str(stripped), metadata={"model_type": "anima"})
    model2 = build_model()
    _Backend(model2)._load_lora_anima([{"path": str(stripped), "strength": STRENGTH}])
    modules2 = dict(model2.named_modules())
    assert {round(sole_branch(modules2[t]).scale, 9) for t in trained_paths} == \
        {round(STRENGTH, 9)}


def test_anima_unload_restores_the_identical_objects_and_is_idempotent(tmp_path):
    path, trained_paths = train_and_save(tmp_path, scope=anima_mod.FULL_SCOPE)

    model = build_model()
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


def test_anima_unload_after_a_stack_restores_the_identical_objects(tmp_path):
    """Restore is by IDENTITY, and one composite per target rather than one
    wrapper per selected LoRA -- including the int-slot `mod` targets."""
    path_a, trained_paths = train_and_save(tmp_path, scope=anima_mod.FULL_SCOPE, seed=1234)
    path_b, _pb = train_and_save(tmp_path, scope=anima_mod.FULL_SCOPE,
                                 name="second.safetensors", seed=4321)

    model = build_model()
    before = dict(model.named_modules())
    backend = _Backend(model)
    backend._load_lora_anima([{"path": path_a, "strength": STRENGTH},
                              {"path": path_b, "strength": STRENGTH_B}])
    assert wrapped_paths(model) == trained_paths

    assert backend._unload_lora_anima() == len(trained_paths)
    after = dict(model.named_modules())
    for target in trained_paths:
        assert after[target] is before[target], target
    assert not wrapped_paths(model)
    assert not backend._anima_lora_wrapped_keys
    assert backend._unload_lora_anima() == 0
    assert dict(model.named_modules()) == after


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


def test_anima_partly_matching_file_warns_and_applies_the_rest(tmp_path, warnings_seen):
    path, trained_paths = train_and_save(tmp_path, scope=ATTENTION_ONLY)
    saved = load_file(path)
    saved["lora_unet_blocks_9_self_attn_q_proj.lora_down.weight"] = torch.zeros(RANK, D)
    saved["lora_unet_blocks_9_self_attn_q_proj.lora_up.weight"] = torch.zeros(D, RANK)
    partial = tmp_path / "partial.safetensors"
    save_file(saved, str(partial), metadata={"model_type": "anima"})

    model = build_model()
    assert _Backend(model)._load_lora_anima([{"path": str(partial)}]) == len(trained_paths)
    assert wrapped_paths(model) == trained_paths
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


def test_anima_model_reload_never_splices_model_a_into_model_b(tmp_path):
    path, trained_paths = train_and_save(tmp_path)

    model_a = build_model()
    backend = _Backend(model_a)
    backend._load_lora_anima([{"path": path, "strength": 1.0}])
    a_ids = module_ids(model_a) | {id(m) for m in backend._anima_lora_original_modules.values()}

    model_b = build_model()
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
