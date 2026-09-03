"""Z-Image: trainer save -> fresh-generation load round trip, on CPU in ~2s.

Drives the REAL ``ZImageLoRAAdapter`` (injection + ``save_checkpoint``) over a
1-layer/1-head Z-Image transformer, then the REAL ``ZImageMixin`` loader on a
freshly built model. The transformer is the production class, not a stub: the
adapter and the loader both select targets by the ``ZImageAttention`` class
name, so a stub would test the stub's naming rather than Z-Image's.

The Phase-0 defect this pins: training wrote ``lora_transformer_<flattened>``
keys while generation searched ``transformer.<dotted>`` keys, so a self-trained
LoRA matched zero targets.

Z-Image is the FIRST architecture on ``CompositeAdapterLayer``, so this file is
also the adoption gate: two LoRAs over one module must SUM, in either selection
order, without perturbing what either one does alone. The stacking refusal these
tests used to assert is gone; the numerics that replace it are checked with
``torch.equal``, because a tolerance would hide exactly the reassociation a
"simplification" of the strength folding would introduce.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/zimage_lora_roundtrip_cheap_test.py -v
"""

import pytest
import torch
from safetensors.torch import save_file

from lora_roundtrip_common import (
    LoRALinearLayer, lora_delta, module_ids, randomise_lora_layers,
    warning_codes, warning_probe,
)

from core.adapters import (  # noqa: E402
    AdapterIncompatible, CompositeAdapterLayer, lora_branch_dtype,
)
from core.pipeline_backends.zimage import ZImageMixin  # noqa: E402
from core.training.adapters.zimage_adapter import ZImageLoRAAdapter  # noqa: E402

# head_dim is fixed at sum(ROPE_AXES_DIMS)=128, so shrink heads/layers, not dim.
_TINY = dict(in_channels=4, dim=128, n_layers=1, n_refiner_layers=1,
             n_heads=1, n_kv_heads=1, cap_feat_dim=16)
RANK = 4
ALPHA = 2  # != rank on purpose: a regression to the rank fallback shows as scale 1.0
SCALE = ALPHA / RANK
STRENGTH = 0.75
STRENGTH_B = 0.4  # the second LoRA's, so a shared scale shows up as a wrong sum


def build_model():
    from core.models.zimage_transformer import ZImageTransformer2DModel
    torch.manual_seed(0)
    return ZImageTransformer2DModel(**_TINY)


class _StubTrainer:
    def __init__(self, transformer):
        self.transformer = transformer
        self.unet_lr = 1e-4


class _Backend(ZImageMixin):
    def __init__(self, transformer):
        self.zimage_components = {"transformer": transformer}


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


def train_and_save(tmp_path, name="zimage.safetensors", seed=1234):
    """Returns (checkpoint path, the trained model, its wrapped target paths)."""
    model = build_model()
    adapter = ZImageLoRAAdapter(_StubTrainer(model), lora_rank=RANK, lora_alpha=ALPHA,
                                lora_dtype=torch.float32)
    layers = {}
    count = adapter.apply_lora_to_unet(layers)
    assert count == len(layers) > 0
    randomise_lora_layers(layers, seed=seed)
    out = tmp_path / name
    adapter.save_checkpoint(layers, step=10, epoch=1, output_path=out)
    return str(out), model, lora_layer_paths(model)


def file_branch_tensors(path, target):
    """``(down, up)`` straight out of the checkpoint, for the analytic sum."""
    from safetensors.torch import load_file
    saved = load_file(path)
    stem = "lora_transformer_" + target.replace(".", "_")
    return saved[f"{stem}.lora_down.weight"], saved[f"{stem}.lora_up.weight"]


@pytest.fixture
def warnings_seen(monkeypatch):
    return warning_probe(monkeypatch)


@pytest.fixture
def resolve_verbatim(monkeypatch):
    """The mixin resolves through LoRAManager; these files live in tmp_path."""
    from core.extensions import lora_manager as lm
    import os

    monkeypatch.setattr(lm.lora_manager, "_resolve_lora_path",
                        lambda p: p if os.path.exists(str(p)) else None)


def test_zimage_generation_wraps_exactly_the_targets_the_trainer_wrapped(tmp_path):
    path, _trained, trained_paths = train_and_save(tmp_path)

    model = build_model()
    backend = _Backend(model)
    backend._load_lora_zimage([{"path": path, "strength": STRENGTH}])

    # Set EQUALITY: the regression was 0 matched targets, and a partial match
    # is just as wrong and much quieter.
    assert wrapped_paths(model) == trained_paths
    assert backend._zimage_lora_wrapped_modules == trained_paths


def test_zimage_wrapped_forward_is_base_plus_scaled_branch(tmp_path):
    path, _trained, trained_paths = train_and_save(tmp_path)

    model = build_model()
    _Backend(model)._load_lora_zimage([{"path": path, "strength": STRENGTH}])

    modules = dict(model.named_modules())
    for target in sorted(trained_paths):
        wrapper = modules[target]
        down, up = file_branch_tensors(path, target)
        x = torch.randn(3, wrapper.original_module.in_features)
        base = wrapper.original_module(x)
        expected = base + lora_delta(down, up, x, ALPHA, RANK, STRENGTH)
        assert torch.allclose(wrapper(x), expected, atol=1e-5), target
        assert not torch.allclose(wrapper(x), base, atol=1e-5), f"{target}: branch is inert"


def test_zimage_single_lora_is_bit_identical_to_the_pre_composite_wrapper(tmp_path):
    """The composite must not change one-LoRA arithmetic by one ULP.

    The reference is what the loader built before adoption: a ``LoRALinearLayer``
    over the same base, its weights copied the same way, its scale written as
    ``(alpha / rank) * strength``. ``torch.equal``, not a tolerance -- folding
    the strength anywhere but into the branch's own scale reassociates the
    multiply and shows up here and nowhere else.
    """
    path, _trained, trained_paths = train_and_save(tmp_path)

    model = build_model()
    _Backend(model)._load_lora_zimage([{"path": path, "strength": STRENGTH}])

    modules = dict(model.named_modules())
    for target in sorted(trained_paths):
        composite = modules[target]
        base = composite.original_module
        down, up = file_branch_tensors(path, target)

        reference = LoRALinearLayer(base, rank=RANK, alpha=float(ALPHA), lora_name=target)
        dtype = lora_branch_dtype(base)
        with torch.no_grad():
            reference.lora_down.weight.data = down.to(device=base.weight.device, dtype=dtype)
            reference.lora_up.weight.data = up.to(device=base.weight.device, dtype=dtype)
        reference.scale = (float(ALPHA) / RANK) * STRENGTH

        assert sole_branch(composite).scale == reference.scale, target
        x = torch.randn(3, base.in_features)
        assert torch.equal(composite(x), reference(x)), target


def test_zimage_two_loras_over_one_module_sum_their_deltas(tmp_path, resolve_verbatim):
    """The adoption gate: base + delta_a + delta_b, against the analytic sum."""
    path_a, _ta, trained_paths = train_and_save(tmp_path, seed=1234)
    path_b, _tb, paths_b = train_and_save(tmp_path, name="second.safetensors", seed=4321)
    assert paths_b == trained_paths, "both files must cover the same targets to stack"

    model = build_model()
    backend = _Backend(model)
    backend._load_lora_zimage([{"path": path_a, "strength": STRENGTH},
                               {"path": path_b, "strength": STRENGTH_B}])

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


def test_zimage_stacked_result_is_independent_of_selection_order(tmp_path,
                                                                 resolve_verbatim):
    path_a, _ta, trained_paths = train_and_save(tmp_path, seed=1234)
    path_b, _tb, _pb = train_and_save(tmp_path, name="second.safetensors", seed=4321)
    a = {"path": path_a, "strength": STRENGTH}
    b = {"path": path_b, "strength": STRENGTH_B}

    forward = build_model()
    _Backend(forward)._load_lora_zimage([a, b])
    reverse = build_model()
    _Backend(reverse)._load_lora_zimage([b, a])

    forward_modules = dict(forward.named_modules())
    reverse_modules = dict(reverse.named_modules())
    for target in sorted(trained_paths):
        one, two = forward_modules[target], reverse_modules[target]
        assert torch.equal(one.original_module.weight, two.original_module.weight), target
        x = torch.randn(3, one.original_module.in_features)
        # Two branches: the deltas are summed before the base is added, and fp
        # addition commutes, so this is EXACT. (Three or more branches
        # would only hold up to associativity.)
        assert torch.equal(one(x), two(x)), target


def test_zimage_removing_one_branch_leaves_the_other_exactly_as_if_alone(tmp_path,
                                                                        resolve_verbatim):
    """A stacked branch must not perturb its neighbour's own arithmetic.

    Bit-identity against the single-LoRA model is the gate that a strength
    applied outside the branch -- scaling a shared delta, or rebuilding the
    first branch when the second arrives -- cannot pass.
    """
    path_a, _ta, trained_paths = train_and_save(tmp_path, seed=1234)
    path_b, _tb, _pb = train_and_save(tmp_path, name="second.safetensors", seed=4321)

    alone = build_model()
    _Backend(alone)._load_lora_zimage([{"path": path_a, "strength": STRENGTH}])

    stacked = build_model()
    _Backend(stacked)._load_lora_zimage([{"path": path_a, "strength": STRENGTH},
                                         {"path": path_b, "strength": STRENGTH_B}])

    alone_modules = dict(alone.named_modules())
    stacked_modules = dict(stacked.named_modules())
    for target in sorted(trained_paths):
        one = alone_modules[target]
        two = stacked_modules[target]
        assert torch.equal(one.original_module.weight, two.original_module.weight), target
        second_name = two.branch_names[1]
        two.remove_branch(second_name)
        assert two.branch_names == one.branch_names, target
        x = torch.randn(3, one.original_module.in_features)
        assert torch.equal(one(x), two(x)), target


def test_zimage_alpha_beats_the_rank_fallback(tmp_path):
    """alpha=2 over rank=4 must apply at 0.5, not 1.0. Z-Image writes alpha in
    file metadata only, so a loader that ignores metadata scores 1.0 here."""
    path, _trained, trained_paths = train_and_save(tmp_path)
    model = build_model()
    _Backend(model)._load_lora_zimage([{"path": path, "strength": STRENGTH}])

    modules = dict(model.named_modules())
    scales = {round(sole_branch(modules[t]).scale, 9) for t in trained_paths}
    assert scales == {round(SCALE * STRENGTH, 9)}

    # Same tensors, no alpha anywhere: the rank fallback is scale 1.0 * strength.
    from safetensors.torch import load_file
    stripped = tmp_path / "no_alpha.safetensors"
    save_file(load_file(path), str(stripped), metadata={"model_type": "zimage"})
    model2 = build_model()
    _Backend(model2)._load_lora_zimage([{"path": str(stripped), "strength": STRENGTH}])
    modules2 = dict(model2.named_modules())
    assert {round(sole_branch(modules2[t]).scale, 9) for t in trained_paths} == \
        {round(STRENGTH, 9)}


def test_zimage_unload_restores_the_identical_objects_and_is_idempotent(tmp_path):
    path, _trained, trained_paths = train_and_save(tmp_path)

    model = build_model()
    modules_before = {name: module for name, module in model.named_modules()}
    backend = _Backend(model)
    backend._load_lora_zimage([{"path": path, "strength": 1.0}])
    assert wrapped_paths(model) == trained_paths

    backend._unload_lora_zimage()
    modules_after = dict(model.named_modules())
    for target in trained_paths:
        assert modules_after[target] is modules_before[target], target
    assert not wrapped_paths(model)
    assert not backend._zimage_lora_wrapped_modules

    backend._unload_lora_zimage()  # second unload: no-op, not a re-splice
    assert dict(model.named_modules())[sorted(trained_paths)[0]] is \
        modules_before[sorted(trained_paths)[0]]


def test_zimage_unload_after_a_stack_restores_the_identical_objects(tmp_path,
                                                                   resolve_verbatim):
    path_a, _ta, trained_paths = train_and_save(tmp_path, seed=1234)
    path_b, _tb, _pb = train_and_save(tmp_path, name="second.safetensors", seed=4321)

    model = build_model()
    before = dict(model.named_modules())
    backend = _Backend(model)
    backend._load_lora_zimage([{"path": path_a, "strength": STRENGTH},
                               {"path": path_b, "strength": STRENGTH_B}])
    assert wrapped_paths(model) == trained_paths

    backend._unload_lora_zimage()
    after = dict(model.named_modules())
    for target in trained_paths:
        assert after[target] is before[target], target
    assert not wrapped_paths(model)
    assert not backend._zimage_lora_wrapped_modules

    backend._unload_lora_zimage()
    assert dict(model.named_modules())[sorted(trained_paths)[0]] is \
        before[sorted(trained_paths)[0]]


def test_zimage_missing_file_refuses_and_warns(tmp_path, warnings_seen, monkeypatch):
    from core.extensions import lora_manager as lm

    monkeypatch.setattr(lm.lora_manager, "_resolve_lora_path", lambda p: None)
    with pytest.raises(FileNotFoundError):
        _Backend(build_model())._load_lora_zimage([{"path": "no_such_lora.safetensors"}])
    assert "lora_not_found" in warning_codes(warnings_seen)


def test_zimage_zero_matched_targets_refuses_and_warns(tmp_path, warnings_seen,
                                                       resolve_verbatim):
    foreign = tmp_path / "foreign.safetensors"
    save_file({"lora_unet_totally_other_module.lora_down.weight": torch.zeros(RANK, 8),
               "lora_unet_totally_other_module.lora_up.weight": torch.zeros(8, RANK)},
              str(foreign), metadata={"model_type": "zimage"})

    model = build_model()
    with pytest.raises(RuntimeError, match="0 of"):
        _Backend(model)._load_lora_zimage([{"path": str(foreign), "strength": 1.0}])
    assert "lora_incompatible" in warning_codes(warnings_seen)
    assert not wrapped_paths(model)


def test_zimage_shape_mismatched_branch_is_refused_atomically(tmp_path,
                                                              warnings_seen,
                                                              resolve_verbatim):
    from safetensors.torch import load_file

    path, _trained, trained_paths = train_and_save(tmp_path)
    victim = sorted(trained_paths)[0]
    stem = "lora_transformer_" + victim.replace(".", "_")
    saved = load_file(path)
    saved[f"{stem}.lora_down.weight"] = torch.randn(RANK, 3)
    broken = tmp_path / "broken.safetensors"
    save_file(saved, str(broken), metadata={"lora_alpha": str(ALPHA)})

    model = build_model()
    before = dict(model.named_modules())
    with pytest.raises(AdapterIncompatible) as excinfo:
        _Backend(model)._load_lora_zimage(
            [{"path": str(broken), "strength": STRENGTH}])

    assert excinfo.value.code == "lora_partial"
    assert not wrapped_paths(model)
    assert dict(model.named_modules())[victim] is before[victim]
    assert "lora_partial" in warning_codes(warnings_seen)


def test_zimage_model_reload_never_splices_model_a_into_model_b(tmp_path):
    """Wrap A, swap the component to B, unload: not one of A's modules may end
    up installed in B. This defect was proven on every architecture audited."""
    path, _trained, trained_paths = train_and_save(tmp_path)

    model_a = build_model()
    backend = _Backend(model_a)
    backend._load_lora_zimage([{"path": path, "strength": 1.0}])
    a_ids = module_ids(model_a) | {id(m) for m in backend._zimage_lora_original_modules.values()}

    model_b = build_model()
    b_ids_before = module_ids(model_b)
    assert not (a_ids & b_ids_before), "setup: A and B must not already share modules"

    backend.zimage_components["transformer"] = model_b
    assert backend._zimage_lora_wrapped_modules, "the stale set must be truthy to be a test"
    backend._unload_lora_zimage()

    assert module_ids(model_b) == b_ids_before, "model B's module graph was modified"
    assert not (module_ids(model_b) & a_ids), "a module of model A was installed into model B"
    assert not backend._zimage_lora_wrapped_modules
    assert not backend._zimage_lora_original_modules

    # B still loads and unwinds to its OWN originals.
    b_before = dict(model_b.named_modules())
    backend._load_lora_zimage([{"path": path, "strength": 1.0}])
    assert wrapped_paths(model_b) == trained_paths
    backend._unload_lora_zimage()
    for target in trained_paths:
        assert dict(model_b.named_modules())[target] is b_before[target], target
    assert not (module_ids(model_b) & a_ids)
    assert not wrapped_paths(model_a), "the abandoned model retained session branches"
