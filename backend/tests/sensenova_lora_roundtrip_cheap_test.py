"""SenseNova U1.5: trainer save -> fresh-generation load round trip, CPU, ~15s.

Drives the REAL ``SenseNovaLoRAAdapter`` over BOTH MoT halves -- the generation
branch (``*_mot_gen``) and the understanding branch, which is injected only when
``train_text_encoder`` is set -- then the REAL
``SenseNovaMixin._load_lora_sensenova``.

The stub is 42 decoder layers wide because the adapter refuses anything else
(294 targets per branch, exactly), but each Linear is 4x4.

The target naming is deliberately ASYMMETRIC and is what a key-codec regression
would break: the generation suffix is on the LINEAR for self_attn
(``q_proj_mot_gen``) and on the PARENT for the MLP (``mlp_mot_gen.gate_proj``).

SenseNova is on ``CompositeAdapterLayer``, so this file is also its adoption
gate: two LoRAs over one module must SUM, in either selection order, without
perturbing what either does alone, and PER HALF -- a stack on the generation
branch must leave the understanding branch bare. The stacking refusal these
tests used to assert is gone; the numerics that replace it are checked with
``torch.equal``, because a tolerance would hide exactly the reassociation a
"simplification" of the strength folding would introduce.

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

from core.adapters import AdapterIncompatible, CompositeAdapterLayer  # noqa: E402
from core.models.sensenova import sensenova_lora as sn_lora  # noqa: E402
from core.pipeline_backends.sensenova import SenseNovaMixin  # noqa: E402
from core.training.adapters.sensenova_adapter import SenseNovaLoRAAdapter  # noqa: E402

H = 4
LAYERS = 42  # the adapter refuses any other decoder depth
PER_BRANCH = 294
RANK = 4
# alpha/rank = 1.5 and strength 0.7 are chosen so that folding the strength
# anywhere but into the branch's own scale REALLY changes the bits. The
# previous alpha=2/rank=8/strength=0.5 made both the scale and the strength
# exact powers of two, where every plausible reassociation is IEEE-identical
# and the bit-identity gate below is vacuous.
ALPHA = 6
SCALE = ALPHA / RANK
STRENGTH = 0.7
STRENGTH_B = 0.4  # the second LoRA's, so a shared scale shows up as a wrong sum


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


def build_model():
    """Deterministic base weights: the stacking gates compare two models'
    outputs, so their bases have to be the same tensors."""
    torch.manual_seed(0)
    return _Stub()


def composite_paths(model):
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


def file_branch_tensors(path, target):
    """``(down, up)`` straight out of the checkpoint. SenseNova keys are the
    module path VERBATIM -- no ``lora_unet_`` prefix, unlike every other arch."""
    saved = load_file(path)
    return saved[f"{target}.lora_down.weight"], saved[f"{target}.lora_up.weight"]


def train_and_save(tmp_path, name="sensenova.safetensors", seed=1234, both_halves=True):
    """Returns (path, generation target paths, understanding target paths)."""
    model = build_model()
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

    all_paths = lora_layer_paths(model)
    gen_paths = {p for p in all_paths if "mot_gen" in p}
    und_paths = all_paths - gen_paths
    return str(out), gen_paths, und_paths


def half_only(tmp_path, path, name, half):
    """Rewrite ``path`` keeping only one MoT half's keys.

    The trainer refuses to save an understanding-only LoRA, so the und-only
    arm of the per-half gates has to be built by filtering.
    """
    keep = (lambda k: "mot_gen" in k) if half == "gen" else (lambda k: "mot_gen" not in k)
    out = tmp_path / name
    save_file({k: v for k, v in load_file(path).items() if keep(k)}, str(out),
              metadata={"model_type": "sensenova"})
    return str(out)


def probe_targets(gen_paths, und_paths):
    """One target of every KIND in each half.

    The codec, not the arithmetic, is what varies across the 588; running the
    per-target numeric gates over all of them costs a minute and pins nothing
    the seven kinds per half do not.
    """
    probes = []
    for paths, suffixes in (
        (gen_paths, (".self_attn.q_proj_mot_gen", ".self_attn.k_proj_mot_gen",
                     ".self_attn.v_proj_mot_gen", ".self_attn.o_proj_mot_gen",
                     ".mlp_mot_gen.gate_proj", ".mlp_mot_gen.up_proj",
                     ".mlp_mot_gen.down_proj")),
        (und_paths, (".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj",
                     ".self_attn.o_proj", ".mlp.gate_proj", ".mlp.up_proj",
                     ".mlp.down_proj")),
    ):
        for suffix in suffixes:
            matches = sorted(p for p in paths if p.endswith(suffix))
            assert matches, suffix
            probes.append(matches[0])
    assert len(probes) == 14
    return probes


@pytest.fixture
def warnings_seen(monkeypatch):
    return warning_probe(monkeypatch)


def test_sensenova_generation_covers_both_mot_halves_the_trainer_wrapped(tmp_path):
    path, gen_paths, und_paths = train_and_save(tmp_path)
    assert len(gen_paths) == len(und_paths) == PER_BRANCH
    # The partition is total, which is why ONE bookkeeping pair can serve both
    # halves: a generation key always carries the marker, an understanding one
    # never can.
    assert not (gen_paths & und_paths)

    model = build_model()
    backend = _Backend(model)
    applied = backend._load_lora_sensenova([{"path": path, "strength": STRENGTH}])

    assert applied == 2 * PER_BRANCH
    assert composite_paths(model) == gen_paths | und_paths
    assert backend._sensenova_lora_keys == gen_paths | und_paths
    assert gen_paths == {p for p, *_rest
                         in sn_lora.iter_sensenova_lora_targets(build_model(), branch="gen")}
    assert und_paths == {p for p, *_rest
                         in sn_lora.iter_sensenova_lora_targets(build_model(), branch="und")}
    # The asymmetry a key-codec regression would flatten.
    assert any(p.endswith(".self_attn.q_proj_mot_gen") for p in gen_paths)
    assert any(p.endswith(".mlp_mot_gen.gate_proj") for p in gen_paths)


def test_sensenova_wrapped_forward_is_base_plus_scaled_branch(tmp_path):
    path, gen_paths, und_paths = train_and_save(tmp_path)

    model = build_model()
    _Backend(model)._load_lora_sensenova([{"path": path, "strength": STRENGTH}])

    modules = dict(model.named_modules())
    for target in probe_targets(gen_paths, und_paths):
        composite = modules[target]
        down, up = file_branch_tensors(path, target)
        x = torch.randn(3, H)
        base = composite.original_module(x)
        expected = base + lora_delta(down, up, x, ALPHA, RANK, STRENGTH)
        assert torch.allclose(composite(x), expected, atol=1e-5), target
        assert not torch.allclose(composite(x), base, atol=1e-5), f"{target}: branch is inert"


def test_sensenova_single_lora_is_bit_identical_to_the_pre_composite_wrapper(tmp_path):
    """The composite must not change one-LoRA arithmetic by one ULP, on EITHER half.

    The reference is what the loader built before adoption: a ``LoRALinearLayer``
    over the same base, its weights copied the same way, its scale written as
    ``(alpha / rank) * strength``. ``torch.equal``, not a tolerance -- folding
    the strength anywhere but into the branch's own scale reassociates the
    multiply and shows up here and nowhere else.
    """
    path, gen_paths, und_paths = train_and_save(tmp_path)

    model = build_model()
    _Backend(model)._load_lora_sensenova([{"path": path, "strength": STRENGTH}])
    modules = dict(model.named_modules())

    probes = probe_targets(gen_paths, und_paths)
    for half, targets in (("gen", probes[:7]), ("und", probes[7:])):
        for target in targets:
            composite = modules[target]
            base = composite.original_module
            down, up = file_branch_tensors(path, target)

            reference = LoRALinearLayer(base, rank=RANK, alpha=float(ALPHA),
                                        lora_name=target)
            compute_dtype = base.bias.dtype
            with torch.no_grad():
                reference.lora_down.weight.data = down.to(device=base.weight.device,
                                                          dtype=compute_dtype)
                reference.lora_up.weight.data = up.to(device=base.weight.device,
                                                      dtype=compute_dtype)
            reference.lora_down = reference.lora_down.to(dtype=compute_dtype)
            reference.lora_up = reference.lora_up.to(dtype=compute_dtype)
            reference.scale = (float(ALPHA) / RANK) * STRENGTH

            assert sole_branch(composite).scale == reference.scale, (half, target)
            x = torch.randn(3, H)
            assert torch.equal(composite(x), reference(x)), (half, target)


def test_sensenova_two_loras_over_one_module_sum_their_deltas(tmp_path):
    """The adoption gate: base + delta_a + delta_b, against the analytic sum."""
    path_a, gen_paths, und_paths = train_and_save(tmp_path, seed=1234)
    path_b, gen_b, und_b = train_and_save(tmp_path, name="second.safetensors", seed=4321)
    assert (gen_b, und_b) == (gen_paths, und_paths)

    model = build_model()
    backend = _Backend(model)
    total = backend._load_lora_sensenova([{"path": path_a, "strength": STRENGTH},
                                          {"path": path_b, "strength": STRENGTH_B}])
    assert total == 4 * PER_BRANCH
    assert composite_paths(model) == gen_paths | und_paths

    modules = dict(model.named_modules())
    for target in probe_targets(gen_paths, und_paths):
        composite = modules[target]
        assert len(composite) == 2, f"{target}: {composite.branch_names}"
        base_module = composite.original_module
        down_a, up_a = file_branch_tensors(path_a, target)
        down_b, up_b = file_branch_tensors(path_b, target)
        x = torch.randn(3, H)
        expected = (base_module(x)
                    + lora_delta(down_a, up_a, x, ALPHA, RANK, STRENGTH)
                    + lora_delta(down_b, up_b, x, ALPHA, RANK, STRENGTH_B))
        assert torch.allclose(composite(x), expected, atol=1e-5), target
        # Both branches really contribute: dropping either changes the output.
        assert not torch.allclose(
            composite(x),
            base_module(x) + lora_delta(down_a, up_a, x, ALPHA, RANK, STRENGTH),
            atol=1e-5), f"{target}: the second LoRA is inert"
        assert not torch.allclose(
            composite(x),
            base_module(x) + lora_delta(down_b, up_b, x, ALPHA, RANK, STRENGTH_B),
            atol=1e-5), f"{target}: the first LoRA is inert"


def test_sensenova_stacked_result_is_independent_of_selection_order(tmp_path):
    path_a, gen_paths, und_paths = train_and_save(tmp_path, seed=1234)
    path_b, _g, _u = train_and_save(tmp_path, name="second.safetensors", seed=4321)
    a = {"path": path_a, "strength": STRENGTH}
    b = {"path": path_b, "strength": STRENGTH_B}

    forward = build_model()
    _Backend(forward)._load_lora_sensenova([a, b])
    reverse = build_model()
    _Backend(reverse)._load_lora_sensenova([b, a])

    forward_modules = dict(forward.named_modules())
    reverse_modules = dict(reverse.named_modules())
    for target in probe_targets(gen_paths, und_paths):
        one, two = forward_modules[target], reverse_modules[target]
        assert torch.equal(one.original_module.weight, two.original_module.weight), target
        x = torch.randn(3, H)
        # Two branches: the deltas are summed before the base is added, and fp
        # addition commutes, so this is EXACT. (Three or more would only hold up
        # to associativity.)
        assert torch.equal(one(x), two(x)), target


def test_sensenova_removing_one_branch_leaves_the_other_exactly_as_if_alone(tmp_path):
    """A stacked branch must not perturb its neighbour's own arithmetic."""
    path_a, gen_paths, und_paths = train_and_save(tmp_path, seed=1234)
    path_b, _g, _u = train_and_save(tmp_path, name="second.safetensors", seed=4321)

    alone = build_model()
    _Backend(alone)._load_lora_sensenova([{"path": path_a, "strength": STRENGTH}])

    stacked = build_model()
    _Backend(stacked)._load_lora_sensenova([{"path": path_a, "strength": STRENGTH},
                                            {"path": path_b, "strength": STRENGTH_B}])

    alone_modules = dict(alone.named_modules())
    stacked_modules = dict(stacked.named_modules())
    for target in probe_targets(gen_paths, und_paths):
        one, two = alone_modules[target], stacked_modules[target]
        assert torch.equal(one.original_module.weight, two.original_module.weight), target
        two.remove_branch(two.branch_names[1])
        assert two.branch_names == one.branch_names, target
        x = torch.randn(3, H)
        assert torch.equal(one(x), two(x)), target


def test_sensenova_a_stack_on_one_half_leaves_the_other_half_bare(tmp_path):
    """The two MoT halves are reached by disjoint key namespaces.

    Two generation-only files stack on the generation branch; every
    understanding target keeps the SAME bare Linear object it started with, and
    an understanding-only file added on top lands beside neither.
    """
    both, gen_paths, und_paths = train_and_save(tmp_path, seed=1234)
    second, _g, _u = train_and_save(tmp_path, name="second.safetensors", seed=4321)
    gen_a = half_only(tmp_path, both, "gen_a.safetensors", "gen")
    gen_b = half_only(tmp_path, second, "gen_b.safetensors", "gen")
    und_a = half_only(tmp_path, both, "und_a.safetensors", "und")

    model = build_model()
    before = dict(model.named_modules())
    backend = _Backend(model)
    assert backend._load_lora_sensenova([{"path": gen_a, "strength": STRENGTH},
                                         {"path": gen_b, "strength": STRENGTH_B}]) \
        == 2 * PER_BRANCH

    assert composite_paths(model) == gen_paths
    after = dict(model.named_modules())
    for target in und_paths:
        assert after[target] is before[target], target
    for target in gen_paths:
        assert len(after[target]) == 2, target

    # A third file, understanding-only, reaches the other half and only it.
    assert backend._load_lora_sensenova([{"path": gen_a, "strength": STRENGTH},
                                         {"path": gen_b, "strength": STRENGTH_B},
                                         {"path": und_a, "strength": STRENGTH}]) \
        == 3 * PER_BRANCH
    reloaded = dict(model.named_modules())
    assert composite_paths(model) == gen_paths | und_paths
    for target in gen_paths:
        assert len(reloaded[target]) == 2, target
    for target in und_paths:
        assert len(reloaded[target]) == 1, target


def test_sensenova_mot_phase_eviction_routes_a_stacked_branch_with_its_own_half(tmp_path):
    """Inference-side MoT eviction classifies by module path, and the composite
    renames every tensor it covers.

    Measured here rather than reasoned about: the base stays reachable at
    ``<target>.original_module`` ONLY -- ``named_modules`` de-duplicates the
    branch's alias of the same object, and ``original_module`` is assigned
    before ``branches`` so it is the path reached first -- and each branch's
    weights land under the target's own path, marker and all. Getting this
    wrong is a device mismatch at generation time.
    """
    from core.models.sensenova.mot_weight_selector import select_mot_weight_modules

    path_a, gen_paths, und_paths = train_and_save(tmp_path, seed=1234)
    path_b, _g, _u = train_and_save(tmp_path, name="second.safetensors", seed=4321)

    model = build_model()
    _Backend(model)._load_lora_sensenova([{"path": path_a, "strength": STRENGTH},
                                          {"path": path_b, "strength": STRENGTH_B}])

    selection = select_mot_weight_modules(model)
    gen_ids = {id(m) for m in selection.gen_modules}
    und_ids = {id(m) for m in selection.und_modules}
    assert not gen_ids & und_ids
    assert len(selection.gen_modules) == len(selection.und_modules)

    seen = 0
    for path, module in model.named_modules():
        if not isinstance(module, CompositeAdapterLayer):
            continue
        seen += 1
        expected = gen_ids if "mot_gen" in path else und_ids
        assert id(module.original_module) in expected, path
        for branch in module.branches:
            assert branch.original_module is module.original_module, path
            for weight_module in (branch.lora_down, branch.lora_up):
                assert id(weight_module) in expected, path
    assert seen == 2 * PER_BRANCH

    base_paths = {id(m): name for name, m in model.named_modules()}
    probe = sorted(gen_paths)[0]
    composite = dict(model.named_modules())[probe]
    assert base_paths[id(composite.original_module)] == f"{probe}.original_module"


def test_sensenova_alpha_beats_the_rank_fallback(tmp_path):
    path, gen_paths, und_paths = train_and_save(tmp_path)
    model = build_model()
    _Backend(model)._load_lora_sensenova([{"path": path, "strength": STRENGTH}])
    modules = dict(model.named_modules())
    assert {round(sole_branch(modules[t]).scale, 9)
            for t in gen_paths | und_paths} == {round(SCALE * STRENGTH, 9)}

    # Rung 2: no per-key alpha, alpha only in metadata. Without it the same file
    # would apply at rank-fallback scale instead of its trained one.
    md_only = tmp_path / "md_alpha.safetensors"
    raw = load_file(path)
    save_file({k: v for k, v in raw.items() if not k.endswith(".alpha")}, str(md_only),
              metadata={"model_type": "sensenova", "lora_alpha": str(ALPHA),
                        "lora_rank": str(RANK)})
    model2 = build_model()
    _Backend(model2)._load_lora_sensenova([{"path": str(md_only), "strength": 1.0}])
    modules2 = dict(model2.named_modules())
    assert {round(sole_branch(modules2[t]).scale, 9)
            for t in gen_paths | und_paths} == {round(SCALE, 9)}

    none = tmp_path / "no_alpha.safetensors"
    save_file({k: v for k, v in raw.items() if not k.endswith(".alpha")}, str(none),
              metadata={"model_type": "sensenova"})
    model3 = build_model()
    _Backend(model3)._load_lora_sensenova([{"path": str(none), "strength": 1.0}])
    modules3 = dict(model3.named_modules())
    assert {round(sole_branch(modules3[t]).scale, 9)
            for t in gen_paths | und_paths} == {1.0}


def test_sensenova_unload_restores_the_identical_objects_and_is_idempotent(tmp_path):
    path, gen_paths, und_paths = train_and_save(tmp_path)

    model = build_model()
    before = dict(model.named_modules())
    backend = _Backend(model)
    backend._load_lora_sensenova([{"path": path, "strength": 1.0}])

    assert backend._unload_lora_sensenova() == 2 * PER_BRANCH
    after = dict(model.named_modules())
    for target in gen_paths | und_paths:
        assert after[target] is before[target], target
    assert not composite_paths(model)
    assert backend._unload_lora_sensenova() == 0
    assert dict(model.named_modules()) == after


def test_sensenova_unload_after_a_stack_restores_the_identical_objects(tmp_path):
    path_a, gen_paths, und_paths = train_and_save(tmp_path, seed=1234)
    path_b, _g, _u = train_and_save(tmp_path, name="second.safetensors", seed=4321)

    model = build_model()
    before = dict(model.named_modules())
    backend = _Backend(model)
    backend._load_lora_sensenova([{"path": path_a, "strength": STRENGTH},
                                  {"path": path_b, "strength": STRENGTH_B}])
    assert composite_paths(model) == gen_paths | und_paths

    # One composite per target, not one wrapper per selected LoRA.
    assert backend._unload_lora_sensenova() == 2 * PER_BRANCH
    after = dict(model.named_modules())
    for target in gen_paths | und_paths:
        assert after[target] is before[target], target
    assert not composite_paths(model)
    assert not backend._sensenova_lora_keys
    assert backend._unload_lora_sensenova() == 0
    assert dict(model.named_modules()) == after


def test_sensenova_restore_removes_a_composite_the_originals_map_never_recorded(tmp_path):
    """Restore is driven by what is INSTALLED, not by map membership.

    ``_unload_lora_sensenova`` clears the originals map after every restore,
    so a wrapper that outlived one has no map entry to be found by; the
    composite's own base is the fallback. Not reachable through the shipped
    call sequence -- the load's leading unload and the weakref reset close that
    route -- so this pins the property, not a live defect.
    """
    path, gen_paths, und_paths = train_and_save(tmp_path)
    model = build_model()
    before = dict(model.named_modules())
    backend = _Backend(model)
    backend._load_lora_sensenova([{"path": path, "strength": STRENGTH}])

    backend._sensenova_lora_orig.clear()
    assert sn_lora.restore_originals(model, {}, set()) == 2 * PER_BRANCH
    after = dict(model.named_modules())
    for target in gen_paths | und_paths:
        assert after[target] is before[target], target
    assert not composite_paths(model)


def test_sensenova_missing_file_refuses_and_warns(warnings_seen):
    with pytest.raises(FileNotFoundError):
        _Backend(build_model())._load_lora_sensenova(
            [{"path": "no_such_sn_lora.safetensors"}])
    assert "lora_not_found" in warning_codes(warnings_seen)


def test_sensenova_unreadable_file_refuses_and_warns(tmp_path, warnings_seen):
    broken = tmp_path / "broken.safetensors"
    broken.write_bytes(b"not a safetensors file")
    model = build_model()
    with pytest.raises(RuntimeError):
        _Backend(model)._load_lora_sensenova([{"path": str(broken)}])
    assert "lora_load_failed" in warning_codes(warnings_seen)
    assert not composite_paths(model)


def test_sensenova_zero_matched_targets_refuses_and_warns(tmp_path, warnings_seen):
    foreign = tmp_path / "foreign.safetensors"
    stem = "lora_unet_transformer_blocks_0_attn_to_q"
    save_file({f"{stem}.lora_down.weight": torch.zeros(2, H),
               f"{stem}.lora_up.weight": torch.zeros(H, 2)}, str(foreign))

    model = build_model()
    with pytest.raises(RuntimeError):
        _Backend(model)._load_lora_sensenova([{"path": str(foreign), "strength": 1.0}])
    assert "lora_incompatible" in warning_codes(warnings_seen)
    assert not composite_paths(model)


def test_sensenova_partly_matching_file_is_refused_atomically(tmp_path, warnings_seen):
    path, gen_paths, und_paths = train_and_save(tmp_path)
    saved = load_file(path)
    ghost = "language_model.model.layers.0.self_attn.no_such_proj"
    saved[f"{ghost}.lora_down.weight"] = torch.zeros(RANK, H)
    saved[f"{ghost}.lora_up.weight"] = torch.zeros(H, RANK)
    partial = tmp_path / "partial.safetensors"
    save_file(saved, str(partial), metadata={"model_type": "sensenova"})

    model = build_model()
    with pytest.raises(AdapterIncompatible) as excinfo:
        _Backend(model)._load_lora_sensenova([{"path": str(partial)}])
    assert excinfo.value.code == "lora_partial"
    assert not composite_paths(model)
    assert "lora_partial" in warning_codes(warnings_seen)


def test_sensenova_selecting_the_same_file_twice_is_two_branches(tmp_path):
    """Branch names are unique per REQUEST INDEX, not per basename."""
    path, gen_paths, und_paths = train_and_save(tmp_path)
    model = build_model()
    total = _Backend(model)._load_lora_sensenova([{"path": path, "strength": STRENGTH},
                                                  {"path": path, "strength": STRENGTH_B}])
    assert total == 4 * PER_BRANCH
    modules = dict(model.named_modules())
    for target in probe_targets(gen_paths, und_paths):
        assert len(modules[target]) == 2, target
        assert len(set(modules[target].branch_names)) == 2, target


def test_sensenova_model_reload_never_splices_model_a_into_model_b(tmp_path):
    path, gen_paths, und_paths = train_and_save(tmp_path, seed=1234)
    second, _g, _u = train_and_save(tmp_path, name="second.safetensors", seed=4321)
    stack = [{"path": path, "strength": STRENGTH},
             {"path": second, "strength": STRENGTH_B}]

    model_a = build_model()
    backend = _Backend(model_a)
    backend._load_lora_sensenova(stack)
    a_ids = module_ids(model_a) | {id(m) for m in backend._sensenova_lora_orig.values()}

    model_b = build_model()
    b_ids_before = module_ids(model_b)
    assert not (a_ids & b_ids_before), "setup: A and B must not already share modules"

    backend.sensenova_components = {"transformer": model_b}
    assert backend._sensenova_lora_keys, "the stale set must be truthy to be a test"
    assert backend._unload_lora_sensenova() == 0
    assert module_ids(model_b) == b_ids_before, "model B's module graph was modified"
    assert not (module_ids(model_b) & a_ids)

    b_before = dict(model_b.named_modules())
    assert backend._load_lora_sensenova(stack) == 4 * PER_BRANCH
    assert backend._unload_lora_sensenova() == 2 * PER_BRANCH
    for target in gen_paths | und_paths:
        assert dict(model_b.named_modules())[target] is b_before[target], target
    assert not (module_ids(model_b) & a_ids)


def test_sensenova_int8_wrapper_root_count_sees_a_composite_not_its_branches(tmp_path):
    """``count_adapter_wrapper_roots`` counts targets, not branches.

    SenseNova is absent from ``RUNTIME_INT8_ARCHS`` (its int8 comes from the
    checkpoint, there is no in-place runtime conversion), so no generation path
    consults this today. The count is asserted anyway, because that is the
    number a future gate would read and a wrapper hiding a base behind another
    wrapper is exactly the case the quantizer must refuse.
    """
    from core.models.common.int8_runtime_quantize import lora_wrapped_count

    path_a, _g, _u = train_and_save(tmp_path, seed=1234)
    path_b, _g2, _u2 = train_and_save(tmp_path, name="second.safetensors", seed=4321)

    model = build_model()
    _Backend(model)._load_lora_sensenova([{"path": path_a, "strength": STRENGTH},
                                          {"path": path_b, "strength": STRENGTH_B}])
    assert lora_wrapped_count(model) == 2 * PER_BRANCH
