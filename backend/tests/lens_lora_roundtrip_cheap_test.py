"""Lens: trainer save -> fresh-generation load round trip, CPU, ~1s.

Drives the REAL ``LensLoRAAdapter`` (injection + ``save_checkpoint``) over a
2-block CPU stub and the REAL ``LensMixin._load_lora_lens``. Lens carries the
fused-QKV naming (``img_qkv``/``txt_qkv``) and three int-slot targets
(``attn.to_out.0``, ``img_mod.1``, ``txt_mod.1``), which is why the target-set
equality here is worth more than a count.

Lens is on ``CompositeAdapterLayer``, so this file is also its adoption gate:
two LoRAs over one module must SUM, in either selection order, without
perturbing what either does alone. The stacking refusal these tests used to
assert is gone; the numerics that replace it are checked with ``torch.equal``,
because a tolerance would hide exactly the reassociation a "simplification" of
the strength folding would introduce. Lens's FP8 gate is checked over a
composite, since it reads a wrapper count and the composite is the root now.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/lens_lora_roundtrip_cheap_test.py -v
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
STRENGTH_B = 0.4


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


def sole_branch(composite):
    assert len(composite) == 1, f"expected one branch, got {composite.branch_names}"
    return composite.get_branch(composite.branch_names[0])


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


def reference_wrapper(base, down, up, target, strength):
    """What the pre-composite loader installed, rebuilt from the file's tensors."""
    wrapper = LoRALinearLayer(base, rank=RANK, alpha=float(ALPHA), lora_name=target)
    compute_dtype = base.bias.dtype
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


def test_lens_single_lora_is_bit_identical_to_the_pre_composite_wrapper(tmp_path):
    """The composite must not change one-LoRA arithmetic by one ULP.

    ``torch.equal``, not a tolerance -- folding the strength anywhere but into
    the branch's own scale reassociates the multiply and shows up here alone.
    """
    path, trained_paths = train_and_save(tmp_path, scope=_FULL_SCOPE)

    model = build_model()
    _Backend(model)._load_lora_lens([{"path": path, "strength": STRENGTH}])

    modules = dict(model.named_modules())
    for target in sorted(trained_paths):
        composite = modules[target]
        base = composite.original_module
        down, up = file_branch_tensors(path, target)
        reference = reference_wrapper(base, down, up, target, STRENGTH)

        assert sole_branch(composite).scale == reference.scale, target
        x = torch.randn(3, D)
        assert torch.equal(composite(x), reference(x)), target


def test_lens_two_loras_over_one_module_sum_their_deltas(tmp_path):
    """The adoption gate: base + delta_a + delta_b, against the analytic sum."""
    path_a, trained_paths = train_and_save(tmp_path, scope=_FULL_SCOPE, seed=1234)
    path_b, paths_b = train_and_save(tmp_path, scope=_FULL_SCOPE,
                                     name="second.safetensors", seed=4321)
    assert paths_b == trained_paths, "both files must cover the same targets to stack"

    model = build_model()
    backend = _Backend(model)
    total = backend._load_lora_lens([{"path": path_a, "strength": STRENGTH},
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


def test_lens_stacked_result_is_independent_of_selection_order(tmp_path):
    path_a, trained_paths = train_and_save(tmp_path, scope=_FULL_SCOPE, seed=1234)
    path_b, _pb = train_and_save(tmp_path, scope=_FULL_SCOPE,
                                 name="second.safetensors", seed=4321)
    a = {"path": path_a, "strength": STRENGTH}
    b = {"path": path_b, "strength": STRENGTH_B}

    forward = build_model()
    _Backend(forward)._load_lora_lens([a, b])
    reverse = build_model()
    _Backend(reverse)._load_lora_lens([b, a])

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


def test_lens_removing_one_branch_leaves_the_other_exactly_as_if_alone(tmp_path):
    """A stacked branch must not perturb its neighbour's own arithmetic."""
    path_a, trained_paths = train_and_save(tmp_path, scope=_FULL_SCOPE, seed=1234)
    path_b, _pb = train_and_save(tmp_path, scope=_FULL_SCOPE,
                                 name="second.safetensors", seed=4321)

    alone = build_model()
    _Backend(alone)._load_lora_lens([{"path": path_a, "strength": STRENGTH}])

    stacked = build_model()
    _Backend(stacked)._load_lora_lens([{"path": path_a, "strength": STRENGTH},
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


def test_lens_alpha_beats_the_rank_fallback(tmp_path):
    path, trained_paths = train_and_save(tmp_path)
    model = build_model()
    _Backend(model)._load_lora_lens([{"path": path, "strength": STRENGTH}])
    modules = dict(model.named_modules())
    assert {round(sole_branch(modules[t]).scale, 9) for t in trained_paths} == \
        {round(SCALE * STRENGTH, 9)}

    # Rung 2: no per-key alpha, alpha only in file metadata.
    md_only = tmp_path / "md_alpha.safetensors"
    save_file({k: v for k, v in load_file(path).items() if not k.endswith(".alpha")},
              str(md_only), metadata={"model_type": "lens", "lora_alpha": str(4 * RANK)})
    model2 = build_model()
    _Backend(model2)._load_lora_lens([{"path": str(md_only), "strength": 1.0}])
    modules2 = dict(model2.named_modules())
    assert {round(sole_branch(modules2[t]).scale, 9) for t in trained_paths} == {4.0}

    # Rung 3: no alpha anywhere -> the rank fallback, scale 1.0.
    none = tmp_path / "no_alpha.safetensors"
    save_file({k: v for k, v in load_file(path).items() if not k.endswith(".alpha")},
              str(none), metadata={"model_type": "lens"})
    model3 = build_model()
    _Backend(model3)._load_lora_lens([{"path": str(none), "strength": 1.0}])
    modules3 = dict(model3.named_modules())
    assert {round(sole_branch(modules3[t]).scale, 9) for t in trained_paths} == {1.0}


def test_lens_unload_restores_the_identical_objects_and_is_idempotent(tmp_path):
    path, trained_paths = train_and_save(tmp_path, scope=_FULL_SCOPE)

    model = build_model()
    before = dict(model.named_modules())
    backend = _Backend(model)
    backend._load_lora_lens([{"path": path, "strength": 1.0}])

    assert backend._unload_lora_lens() == len(trained_paths)
    after = dict(model.named_modules())
    for target in trained_paths:
        assert after[target] is before[target], target
    assert not wrapped_paths(model)
    assert backend._unload_lora_lens() == 0
    assert dict(model.named_modules()) == after


def test_lens_unload_after_a_stack_restores_the_identical_objects(tmp_path):
    """Restore is by IDENTITY, and one composite per target rather than one
    wrapper per selected LoRA -- including the int-slot `mod` targets."""
    path_a, trained_paths = train_and_save(tmp_path, scope=_FULL_SCOPE, seed=1234)
    path_b, _pb = train_and_save(tmp_path, scope=_FULL_SCOPE,
                                 name="second.safetensors", seed=4321)

    model = build_model()
    before = dict(model.named_modules())
    backend = _Backend(model)
    backend._load_lora_lens([{"path": path_a, "strength": STRENGTH},
                             {"path": path_b, "strength": STRENGTH_B}])
    assert wrapped_paths(model) == trained_paths

    assert backend._unload_lora_lens() == len(trained_paths)
    after = dict(model.named_modules())
    for target in trained_paths:
        assert after[target] is before[target], target
    assert not wrapped_paths(model)
    assert not backend._lens_lora_wrapped_keys
    assert backend._unload_lora_lens() == 0
    assert dict(model.named_modules()) == after


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


def test_lens_partly_matching_file_warns_and_applies_the_rest(tmp_path, warnings_seen):
    path, trained_paths = train_and_save(tmp_path)
    saved = load_file(path)
    ghost = "lora_unet_" + _flatten_to_sdscripts("transformer_blocks.9.attn.img_qkv")
    saved[f"{ghost}.lora_down.weight"] = torch.zeros(RANK, D)
    saved[f"{ghost}.lora_up.weight"] = torch.zeros(D, RANK)
    partial = tmp_path / "partial.safetensors"
    save_file(saved, str(partial), metadata={"model_type": "lens"})

    model = build_model()
    assert _Backend(model)._load_lora_lens([{"path": str(partial)}]) == len(trained_paths)
    assert wrapped_paths(model) == trained_paths
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


def test_lens_model_reload_never_splices_model_a_into_model_b(tmp_path):
    path, trained_paths = train_and_save(tmp_path)

    model_a = build_model()
    backend = _Backend(model_a)
    backend._load_lora_lens([{"path": path, "strength": 1.0}])
    a_ids = module_ids(model_a) | {id(m) for m in backend._lens_lora_original_modules.values()}

    model_b = build_model()
    b_ids_before = module_ids(model_b)
    assert not (a_ids & b_ids_before), "setup: A and B must not already share modules"

    backend.lens_components["transformer"] = model_b
    assert backend._lens_lora_wrapped_keys, "the stale set must be truthy to be a test"
    assert backend._unload_lora_lens() == 0
    assert module_ids(model_b) == b_ids_before, "model B's module graph was modified"
    assert not (module_ids(model_b) & a_ids)

    b_before = dict(model_b.named_modules())
    assert backend._load_lora_lens([{"path": path, "strength": 1.0}]) == len(trained_paths)
    assert backend._unload_lora_lens() == len(trained_paths)
    for target in trained_paths:
        assert dict(model_b.named_modules())[target] is b_before[target], target
    assert not (module_ids(model_b) & a_ids)


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
