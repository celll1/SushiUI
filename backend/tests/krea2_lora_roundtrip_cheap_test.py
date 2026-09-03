"""Krea 2: trainer save -> fresh-generation load round trip, on CPU in ~1s.

Drives the REAL ``Krea2LoRAAdapter`` (injection + ``save_checkpoint``) over a
2-block CPU stub shaped as ``iter_krea2_lora_targets`` expects, then the REAL
``Krea2Mixin._load_lora_krea2`` on a freshly built stub.

The Phase-0 defect this pins: Krea 2 shipped parser/apply/restore helpers but
its generation backend never applied ``params["loras"]`` at all.

Krea 2 is on ``CompositeAdapterLayer``, so this file is also its adoption gate:
two LoRAs over one module must SUM, in either selection order, without
perturbing what either does alone. The stacking refusal these tests used to
assert is gone; the numerics that replace it are checked with ``torch.equal``,
because a tolerance would hide exactly the reassociation a "simplification" of
the strength folding would introduce.

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

from core.adapters import AdapterIncompatible, CompositeAdapterLayer  # noqa: E402
from core.models.krea2.krea2_lora import (  # noqa: E402
    DEFAULT_SCOPE, flatten_to_key, iter_krea2_lora_targets,
)
from core.pipeline_backends.krea2 import Krea2Mixin  # noqa: E402
from core.training.adapters.krea2_adapter import Krea2LoRAAdapter  # noqa: E402

D = 8
RANK = 4
# alpha/rank = 1.5 and strength 0.7 are chosen so that folding the strength
# anywhere but into the branch's own scale REALLY changes the bits. With the
# previous alpha=8/strength=0.5 the scale came out at exactly 1.0 and every
# plausible reassociation was identical in IEEE754.
ALPHA = 6
SCALE = ALPHA / RANK
STRENGTH = 0.7
STRENGTH_B = 0.4  # the second LoRA's, so a shared scale shows up as a wrong sum


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


def train_and_save(tmp_path, name="krea2.safetensors", seed=1234):
    model = build_model()
    adapter = Krea2LoRAAdapter(_StubTrainer(model), lora_rank=RANK, lora_alpha=ALPHA,
                               lora_dtype=torch.float32)
    layers = {}
    count = adapter.apply_lora_to_unet(layers)
    assert adapter.apply_lora_to_text_encoders(layers) == 0, "Krea 2 has no TE LoRA scope"
    assert count == len(layers) > 0
    randomise_lora_layers(layers, seed=seed)
    out = tmp_path / name
    adapter.save_checkpoint(layers, 100, 1, out)
    return str(out), lora_layer_paths(model)


def file_branch_tensors(path, target):
    """``(down, up)`` straight out of the checkpoint, for the analytic sum."""
    saved = load_file(path)
    stem = flatten_to_key(target)
    return saved[f"{stem}.lora_down.weight"], saved[f"{stem}.lora_up.weight"]


@pytest.fixture
def warnings_seen(monkeypatch):
    return warning_probe(monkeypatch)


def test_krea2_generation_wraps_exactly_the_targets_the_trainer_wrapped(tmp_path):
    path, trained_paths = train_and_save(tmp_path)

    model = build_model()
    backend = _Backend(model)
    applied = backend._load_lora_krea2([{"path": path, "strength": STRENGTH}])

    assert wrapped_paths(model) == trained_paths
    assert applied == len(trained_paths)
    # ...and that set really is the adapter's own iterator, not a subset of it.
    assert trained_paths == {p for p, _parent, _attr, _cur
                             in iter_krea2_lora_targets(build_model(), DEFAULT_SCOPE)}


def test_krea2_wrapped_forward_is_base_plus_scaled_branch(tmp_path):
    path, trained_paths = train_and_save(tmp_path)

    model = build_model()
    _Backend(model)._load_lora_krea2([{"path": path, "strength": STRENGTH}])

    modules = dict(model.named_modules())
    for target in sorted(trained_paths):
        wrapper = modules[target]
        down, up = file_branch_tensors(path, target)
        x = torch.randn(3, D)
        base = wrapper.original_module(x)
        expected = base + lora_delta(down, up, x, ALPHA, RANK, STRENGTH)
        assert torch.allclose(wrapper(x), expected, atol=1e-5), target
        assert not torch.allclose(wrapper(x), base, atol=1e-5), f"{target}: branch is inert"


def test_krea2_single_lora_is_bit_identical_to_the_pre_composite_wrapper(tmp_path):
    """The composite must not change one-LoRA arithmetic by one ULP.

    The reference is what the loader built before adoption: a ``LoRALinearLayer``
    over the same base, its weights copied the same way, its scale written as
    ``(alpha / rank) * strength``. ``torch.equal``, not a tolerance -- folding
    the strength anywhere but into the branch's own scale reassociates the
    multiply and shows up here and nowhere else.
    """
    path, trained_paths = train_and_save(tmp_path)

    model = build_model()
    _Backend(model)._load_lora_krea2([{"path": path, "strength": STRENGTH}])

    modules = dict(model.named_modules())
    for target in sorted(trained_paths):
        composite = modules[target]
        base = composite.original_module
        down, up = file_branch_tensors(path, target)

        reference = LoRALinearLayer(base, rank=RANK, alpha=float(ALPHA), lora_name=target)
        compute_dtype = base.weight.dtype
        with torch.no_grad():
            reference.lora_down.weight.data = down.to(device=base.weight.device,
                                                      dtype=compute_dtype)
            reference.lora_up.weight.data = up.to(device=base.weight.device,
                                                  dtype=compute_dtype)
        reference.lora_down = reference.lora_down.to(dtype=compute_dtype)
        reference.lora_up = reference.lora_up.to(dtype=compute_dtype)
        reference.scale = (float(ALPHA) / RANK) * STRENGTH

        assert sole_branch(composite).scale == reference.scale, target
        x = torch.randn(3, D)
        assert torch.equal(composite(x), reference(x)), target


def test_krea2_two_loras_over_one_module_sum_their_deltas(tmp_path):
    """The adoption gate: base + delta_a + delta_b, against the analytic sum."""
    path_a, trained_paths = train_and_save(tmp_path, seed=1234)
    path_b, paths_b = train_and_save(tmp_path, name="second.safetensors", seed=4321)
    assert paths_b == trained_paths, "both files must cover the same targets to stack"

    model = build_model()
    backend = _Backend(model)
    total = backend._load_lora_krea2([{"path": path_a, "strength": STRENGTH},
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
        x = torch.randn(3, D)
        expected = (base_module(x)
                    + lora_delta(down_a, up_a, x, ALPHA, RANK, STRENGTH)
                    + lora_delta(down_b, up_b, x, ALPHA, RANK, STRENGTH_B))
        assert torch.allclose(composite(x), expected, atol=1e-5), target
        # Both branches really contribute: dropping either changes the output.
        assert not torch.allclose(
            composite(x),
            base_module(x) + lora_delta(down_a, up_a, x, ALPHA, RANK, STRENGTH),
            atol=1e-5), f"{target}: the second LoRA is inert"


def test_krea2_stacked_result_is_independent_of_selection_order(tmp_path):
    path_a, trained_paths = train_and_save(tmp_path, seed=1234)
    path_b, _pb = train_and_save(tmp_path, name="second.safetensors", seed=4321)
    a = {"path": path_a, "strength": STRENGTH}
    b = {"path": path_b, "strength": STRENGTH_B}

    forward = build_model()
    _Backend(forward)._load_lora_krea2([a, b])
    reverse = build_model()
    _Backend(reverse)._load_lora_krea2([b, a])

    forward_modules = dict(forward.named_modules())
    reverse_modules = dict(reverse.named_modules())
    for target in sorted(trained_paths):
        one, two = forward_modules[target], reverse_modules[target]
        assert torch.equal(one.original_module.weight, two.original_module.weight), target
        x = torch.randn(3, D)
        # Two branches: the deltas are summed before the base is added, and fp
        # addition commutes, so this is EXACT. (Three or more would only hold up
        # to associativity.)
        assert torch.equal(one(x), two(x)), target


def test_krea2_removing_one_branch_leaves_the_other_exactly_as_if_alone(tmp_path):
    """A stacked branch must not perturb its neighbour's own arithmetic."""
    path_a, trained_paths = train_and_save(tmp_path, seed=1234)
    path_b, _pb = train_and_save(tmp_path, name="second.safetensors", seed=4321)

    alone = build_model()
    _Backend(alone)._load_lora_krea2([{"path": path_a, "strength": STRENGTH}])

    stacked = build_model()
    _Backend(stacked)._load_lora_krea2([{"path": path_a, "strength": STRENGTH},
                                        {"path": path_b, "strength": STRENGTH_B}])

    alone_modules = dict(alone.named_modules())
    stacked_modules = dict(stacked.named_modules())
    for target in sorted(trained_paths):
        one = alone_modules[target]
        two = stacked_modules[target]
        assert torch.equal(one.original_module.weight, two.original_module.weight), target
        two.remove_branch(two.branch_names[1])
        assert two.branch_names == one.branch_names, target
        x = torch.randn(3, D)
        assert torch.equal(one(x), two(x)), target


def test_krea2_alpha_beats_the_rank_fallback(tmp_path):
    path, trained_paths = train_and_save(tmp_path)
    model = build_model()
    _Backend(model)._load_lora_krea2([{"path": path, "strength": STRENGTH}])
    modules = dict(model.named_modules())
    assert {round(sole_branch(modules[t]).scale, 9) for t in trained_paths} == \
        {round(SCALE * STRENGTH, 9)}

    stripped = tmp_path / "no_alpha.safetensors"
    save_file({k: v for k, v in load_file(path).items() if not k.endswith(".alpha")},
              str(stripped))
    model2 = build_model()
    _Backend(model2)._load_lora_krea2([{"path": str(stripped), "strength": STRENGTH}])
    modules2 = dict(model2.named_modules())
    assert {round(sole_branch(modules2[t]).scale, 9) for t in trained_paths} == \
        {round(STRENGTH, 9)}


def test_krea2_unload_restores_the_identical_objects_and_is_idempotent(tmp_path):
    path, trained_paths = train_and_save(tmp_path)

    model = build_model()
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


def test_krea2_unload_after_a_stack_restores_the_identical_objects(tmp_path):
    path_a, trained_paths = train_and_save(tmp_path, seed=1234)
    path_b, _pb = train_and_save(tmp_path, name="second.safetensors", seed=4321)

    model = build_model()
    before = dict(model.named_modules())
    backend = _Backend(model)
    backend._load_lora_krea2([{"path": path_a, "strength": STRENGTH},
                              {"path": path_b, "strength": STRENGTH_B}])
    assert wrapped_paths(model) == trained_paths

    # One composite per target, not one wrapper per selected LoRA.
    assert backend._unload_lora_krea2() == len(trained_paths)
    after = dict(model.named_modules())
    for target in trained_paths:
        assert after[target] is before[target], target
    assert not wrapped_paths(model)
    assert not backend._krea2_lora_wrapped_keys
    assert backend._unload_lora_krea2() == 0
    assert dict(model.named_modules()) == after


def test_krea2_missing_file_refuses(warnings_seen):
    with pytest.raises(RuntimeError, match="not found"):
        _Backend(build_model())._load_lora_krea2([{"path": "no_such_krea2_lora.safetensors"}])


def test_krea2_missing_file_warns(warnings_seen):
    with pytest.raises(RuntimeError):
        _Backend(build_model())._load_lora_krea2([{"path": "no_such_krea2_lora.safetensors"}])
    assert "lora_not_found" in warning_codes(warnings_seen)


def test_krea2_unreadable_file_refuses_and_warns(tmp_path, warnings_seen):
    broken = tmp_path / "broken.safetensors"
    broken.write_bytes(b"not a safetensors file")
    with pytest.raises(RuntimeError):
        _Backend(build_model())._load_lora_krea2([{"path": str(broken)}])
    assert "lora_load_failed" in warning_codes(warnings_seen)


def test_krea2_zero_matched_targets_refuses_and_warns(tmp_path, warnings_seen):
    ghost = tmp_path / "ghost.safetensors"
    stem = flatten_to_key("transformer_blocks.99.attn.to_q")
    save_file({f"{stem}.lora_down.weight": torch.zeros(RANK, D),
               f"{stem}.lora_up.weight": torch.zeros(D, RANK)}, str(ghost))

    model = build_model()
    with pytest.raises(RuntimeError, match="0 of 1 modules matched"):
        _Backend(model)._load_lora_krea2([{"path": str(ghost), "strength": 1.0}])
    assert "lora_incompatible" in warning_codes(warnings_seen)
    assert not wrapped_paths(model)


def test_krea2_partly_matching_file_is_refused_atomically(tmp_path, warnings_seen):
    path, trained_paths = train_and_save(tmp_path)
    saved = load_file(path)
    ghost = flatten_to_key("transformer_blocks.99.attn.to_q")
    saved[f"{ghost}.lora_down.weight"] = torch.zeros(RANK, D)
    saved[f"{ghost}.lora_up.weight"] = torch.zeros(D, RANK)
    partial = tmp_path / "partial.safetensors"
    save_file(saved, str(partial))

    model = build_model()
    with pytest.raises(AdapterIncompatible) as excinfo:
        _Backend(model)._load_lora_krea2([{"path": str(partial)}])
    assert excinfo.value.code == "lora_partial"
    assert not wrapped_paths(model)
    assert "lora_partial" in warning_codes(warnings_seen)


def test_krea2_model_reload_never_splices_model_a_into_model_b(tmp_path):
    path, trained_paths = train_and_save(tmp_path)

    model_a = build_model()
    backend = _Backend(model_a)
    backend._load_lora_krea2([{"path": path, "strength": 1.0}])
    a_ids = module_ids(model_a) | {id(m) for m in backend._krea2_lora_original_modules.values()}

    model_b = build_model()
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
    backend = _Backend(build_model())
    backend._load_lora_krea2([{"path": path, "strength": 1.0}])
    backend.krea2_components = None
    assert backend._unload_lora_krea2() == 0
    assert not backend._krea2_lora_original_modules
    assert not backend._krea2_lora_wrapped_keys


def test_krea2_a_refused_file_leaves_the_ones_before_it_uninstalled(tmp_path,
                                                                   warnings_seen):
    """``AdapterSession`` plans the WHOLE request before mutating a slot.

    Krea 2 used to wrap file by file, so a second file that matched nothing left
    the first one installed and refused the generation anyway -- the request came
    back as an error over a model that was no longer the one the user has.
    """
    path, _trained_paths = train_and_save(tmp_path)
    ghost = tmp_path / "ghost.safetensors"
    stem = flatten_to_key("transformer_blocks.99.attn.to_q")
    save_file({f"{stem}.lora_down.weight": torch.zeros(RANK, D),
               f"{stem}.lora_up.weight": torch.zeros(D, RANK)}, str(ghost))

    model = build_model()
    before = dict(model.named_modules())
    backend = _Backend(model)
    with pytest.raises(RuntimeError, match="0 of 1 modules matched"):
        backend._load_lora_krea2([{"path": path, "strength": STRENGTH},
                                  {"path": str(ghost), "strength": 1.0}])

    assert not wrapped_paths(model)
    assert dict(model.named_modules()) == before
    assert not backend._krea2_lora_wrapped_keys
    assert not backend._krea2_lora_original_modules
