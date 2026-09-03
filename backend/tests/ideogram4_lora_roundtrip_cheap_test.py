"""Ideogram 4: trainer save -> fresh-generation load round trip, CPU, ~1s.

Drives the REAL ``Ideogram4LoRAAdapter`` (injection + ``save_checkpoint``) over
2-block CPU stubs and the REAL ``Ideogram4Mixin._load_lora_ideogram4``.

Ideogram 4 is the only architecture with two independently wrapped
transformers: the conditional branch and the unconditional one (asymmetric
CFG). Its checkpoint namespaces them ``lora_unet_*`` / ``lora_uncond_*``, and
each branch's bookkeeping is reset independently, so the reload gate here
covers a partial reload as well as a full one.

Ideogram 4 is on ``CompositeAdapterLayer``, so this file is also the adoption
gate: two LoRAs over one module must SUM, in either selection order, without
perturbing what either one does alone -- and PER BRANCH, since a stack on the
conditional transformer must leave the unconditional one exactly as a single
LoRA left it. The stacking refusal these tests used to assert is gone; the
numerics that replace it are checked with ``torch.equal``, because a tolerance
would hide exactly the reassociation a "simplification" of the strength folding
would introduce.

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

from core.adapters import AdapterIncompatible, CompositeAdapterLayer  # noqa: E402
from core.models.ideogram4.ideogram4_lora import (  # noqa: E402
    DEFAULT_SCOPE, _flatten_to_sdscripts, iter_ideogram4_lora_targets,
    normalise_lora_state_dict,
)
from core.pipeline_backends.ideogram4 import Ideogram4Mixin  # noqa: E402
from core.training.adapters.ideogram4_adapter import Ideogram4LoRAAdapter  # noqa: E402

D = 8
RANK = 4
# alpha/rank = 1.5 and strength 0.7 give scale 1.05. The shipped constants were
# alpha 8 / rank 4 / strength 0.5, i.e. scale EXACTLY 1.0, where every plausible
# reassociation of the strength folding is identical in IEEE754 and the
# bit-identity gate below is vacuous.
ALPHA = 6
SCALE = ALPHA / RANK
STRENGTH = 0.7
STRENGTH_B = 0.4  # the second LoRA's, so a shared scale shows up as a wrong sum


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


class _StubModule(nn.Module):
    def __init__(self, n_blocks=2):
        super().__init__()
        self.layers = nn.ModuleList([_Layer() for _ in range(n_blocks)])


def _Stub(n_blocks=2, seed=7):
    """A stub with SEEDED base weights.

    Gates 2 and 3 compare two independently built models, so unseeded bases make
    a bit-identity claim about the branch arithmetic into a claim about nothing.
    """
    torch.manual_seed(seed)
    return _StubModule(n_blocks)


class _Backend(Ideogram4Mixin):
    def __init__(self, transformer, uncond=None):
        self.ideogram4_components = {"transformer": transformer}
        if uncond is not None:
            self.ideogram4_components["unconditional_transformer"] = uncond


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
    return (str(out), lora_layer_paths(cond),
            lora_layer_paths(uncond) if uncond is not None else set())


def file_branch_tensors(path, target, prefix="lora_unet_"):
    """``(down, up)`` straight out of the checkpoint, for the analytic sum."""
    saved = load_file(path)
    stem = prefix + _flatten_to_sdscripts(target)
    return saved[f"{stem}.lora_down.weight"], saved[f"{stem}.lora_up.weight"]


def pre_composite_reference(base, down, up, strength, alpha=None):
    """What the loader built BEFORE adoption, tensor for tensor.

    Deliberately re-derives the compute dtype the same way ``apply_lora_group``
    does rather than assuming fp32: the reference has to move with the loader's
    dtype policy or the gate stops meaning anything on a quantized base.
    """
    alpha_value = float(ALPHA if alpha is None else alpha)
    rank = int(down.shape[0])
    reference = LoRALinearLayer(base, rank=rank, alpha=alpha_value, lora_name="ref")
    if getattr(base, "bias", None) is not None and base.bias.dtype.is_floating_point:
        compute_dtype = base.bias.dtype
    elif (base.weight.dtype.is_floating_point
          and "float8" not in str(base.weight.dtype)):
        compute_dtype = base.weight.dtype
    else:
        compute_dtype = torch.bfloat16
    with torch.no_grad():
        reference.lora_down.weight.data = down.to(device=base.weight.device,
                                                  dtype=compute_dtype)
        reference.lora_up.weight.data = up.to(device=base.weight.device,
                                              dtype=compute_dtype)
    reference.lora_down = reference.lora_down.to(dtype=compute_dtype)
    reference.lora_up = reference.lora_up.to(dtype=compute_dtype)
    reference.scale = (alpha_value / rank) * strength
    return reference


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

    model = _Stub()
    _Backend(model)._load_lora_ideogram4([{"path": path, "strength": STRENGTH}])

    modules = dict(model.named_modules())
    for target in sorted(trained_paths):
        wrapper = modules[target]
        down, up = file_branch_tensors(path, target)
        x = torch.randn(3, D)
        base = wrapper.original_module(x)
        expected = base + lora_delta(down, up, x, ALPHA, RANK, STRENGTH)
        assert torch.allclose(wrapper(x), expected, atol=1e-5), target
        assert not torch.allclose(wrapper(x), base, atol=1e-5), f"{target}: branch is inert"


def test_ideogram4_single_lora_is_bit_identical_to_the_pre_composite_wrapper(tmp_path):
    """The composite must not change one-LoRA arithmetic by one ULP.

    Asserted for BOTH transformers separately: the two branches are wrapped by
    separate calls with separate bookkeeping, so a folding mistake could land on
    one of them alone. ``torch.equal``, not a tolerance -- folding the strength
    anywhere but into the branch's own scale reassociates the multiply and shows
    up here and nowhere else.
    """
    path, cond_paths, uncond_paths = train_and_save(tmp_path, with_uncond=True)

    cond, uncond = _Stub(), _Stub()
    _Backend(cond, uncond)._load_lora_ideogram4([{"path": path, "strength": STRENGTH}])

    checked = 0
    for model, paths, prefix in ((cond, cond_paths, "lora_unet_"),
                                 (uncond, uncond_paths, "lora_uncond_")):
        modules = dict(model.named_modules())
        for target in sorted(paths):
            composite = modules[target]
            base = composite.original_module
            down, up = file_branch_tensors(path, target, prefix)
            reference = pre_composite_reference(base, down, up, STRENGTH)

            assert sole_branch(composite).scale == reference.scale, target
            x = torch.randn(3, D)
            assert torch.equal(composite(x), reference(x)), f"{prefix}{target}"
            checked += 1
    assert checked == len(cond_paths) + len(uncond_paths)


def test_ideogram4_two_loras_over_one_module_sum_their_deltas(tmp_path):
    """The adoption gate: base + delta_a + delta_b, against the analytic sum."""
    path_a, cond_paths, uncond_paths = train_and_save(tmp_path, seed=1234, with_uncond=True)
    path_b, cb, ub = train_and_save(tmp_path, name="second.safetensors", seed=4321,
                                    with_uncond=True)
    assert (cb, ub) == (cond_paths, uncond_paths), "both files must cover the same targets"

    cond, uncond = _Stub(), _Stub()
    _Backend(cond, uncond)._load_lora_ideogram4(
        [{"path": path_a, "strength": STRENGTH},
         {"path": path_b, "strength": STRENGTH_B}])

    for model, paths, prefix in ((cond, cond_paths, "lora_unet_"),
                                 (uncond, uncond_paths, "lora_uncond_")):
        assert wrapped_paths(model) == paths
        modules = dict(model.named_modules())
        for target in sorted(paths):
            composite = modules[target]
            assert len(composite) == 2, f"{target}: {composite.branch_names}"
            base_module = composite.original_module
            down_a, up_a = file_branch_tensors(path_a, target, prefix)
            down_b, up_b = file_branch_tensors(path_b, target, prefix)
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


def test_ideogram4_stacked_result_is_independent_of_selection_order(tmp_path):
    path_a, cond_paths, uncond_paths = train_and_save(tmp_path, seed=1234, with_uncond=True)
    path_b, _cb, _ub = train_and_save(tmp_path, name="second.safetensors", seed=4321,
                                      with_uncond=True)
    a = {"path": path_a, "strength": STRENGTH}
    b = {"path": path_b, "strength": STRENGTH_B}

    fwd_cond, fwd_uncond = _Stub(), _Stub()
    _Backend(fwd_cond, fwd_uncond)._load_lora_ideogram4([a, b])
    rev_cond, rev_uncond = _Stub(), _Stub()
    _Backend(rev_cond, rev_uncond)._load_lora_ideogram4([b, a])

    for one_model, two_model, paths in ((fwd_cond, rev_cond, cond_paths),
                                        (fwd_uncond, rev_uncond, uncond_paths)):
        one_modules = dict(one_model.named_modules())
        two_modules = dict(two_model.named_modules())
        for target in sorted(paths):
            one, two = one_modules[target], two_modules[target]
            assert torch.equal(one.original_module.weight,
                               two.original_module.weight), target
            x = torch.randn(3, D)
            # Two branches: the deltas are summed before the base is added, and
            # fp addition commutes, so this is EXACT. (Three or more branches
            # would only hold up to associativity.)
            assert torch.equal(one(x), two(x)), target


def test_ideogram4_removing_one_branch_leaves_the_other_exactly_as_if_alone(tmp_path):
    """A stacked branch must not perturb its neighbour's own arithmetic."""
    path_a, cond_paths, uncond_paths = train_and_save(tmp_path, seed=1234, with_uncond=True)
    path_b, _cb, _ub = train_and_save(tmp_path, name="second.safetensors", seed=4321,
                                      with_uncond=True)

    alone_cond, alone_uncond = _Stub(), _Stub()
    _Backend(alone_cond, alone_uncond)._load_lora_ideogram4(
        [{"path": path_a, "strength": STRENGTH}])

    stacked_cond, stacked_uncond = _Stub(), _Stub()
    _Backend(stacked_cond, stacked_uncond)._load_lora_ideogram4(
        [{"path": path_a, "strength": STRENGTH},
         {"path": path_b, "strength": STRENGTH_B}])

    for alone, stacked, paths in ((alone_cond, stacked_cond, cond_paths),
                                  (alone_uncond, stacked_uncond, uncond_paths)):
        alone_modules = dict(alone.named_modules())
        stacked_modules = dict(stacked.named_modules())
        for target in sorted(paths):
            one = alone_modules[target]
            two = stacked_modules[target]
            assert torch.equal(one.original_module.weight,
                               two.original_module.weight), target
            two.remove_branch(two.branch_names[1])
            assert two.branch_names == one.branch_names, target
            x = torch.randn(3, D)
            assert torch.equal(one(x), two(x)), target


def test_ideogram4_a_stack_on_the_cond_half_leaves_the_uncond_half_untouched(tmp_path):
    """Per-component lifecycle: the two transformers hold SEPARATE composites.

    A second file carrying only conditional keys must add a branch to the
    conditional composites and leave the unconditional ones bit-identical to the
    single-LoRA model.
    """
    both, cond_paths, uncond_paths = train_and_save(tmp_path, seed=1234, with_uncond=True)
    cond_only, _cp, _up = train_and_save(tmp_path, name="cond_only.safetensors", seed=4321)

    alone_cond, alone_uncond = _Stub(), _Stub()
    _Backend(alone_cond, alone_uncond)._load_lora_ideogram4(
        [{"path": both, "strength": STRENGTH}])

    cond, uncond = _Stub(), _Stub()
    _Backend(cond, uncond)._load_lora_ideogram4(
        [{"path": both, "strength": STRENGTH},
         {"path": cond_only, "strength": STRENGTH_B}])

    cond_modules = dict(cond.named_modules())
    for target in sorted(cond_paths):
        assert len(cond_modules[target]) == 2, target

    alone_modules = dict(alone_uncond.named_modules())
    uncond_modules = dict(uncond.named_modules())
    for target in sorted(uncond_paths):
        one, two = alone_modules[target], uncond_modules[target]
        assert len(two) == 1, f"{target}: {two.branch_names}"
        x = torch.randn(3, D)
        assert torch.equal(one(x), two(x)), target


def test_ideogram4_alpha_beats_the_rank_fallback(tmp_path):
    path, trained_paths, _uncond = train_and_save(tmp_path)
    model = _Stub()
    _Backend(model)._load_lora_ideogram4([{"path": path, "strength": STRENGTH}])
    modules = dict(model.named_modules())
    assert {round(sole_branch(modules[t]).scale, 9) for t in trained_paths} == \
        {round(SCALE * STRENGTH, 9)}

    md_only = tmp_path / "md_alpha.safetensors"
    save_file({k: v for k, v in load_file(path).items() if not k.endswith(".alpha")},
              str(md_only), metadata={"model_type": "ideogram4", "lora_alpha": str(4 * RANK)})
    model2 = _Stub()
    _Backend(model2)._load_lora_ideogram4([{"path": str(md_only), "strength": 1.0}])
    modules2 = dict(model2.named_modules())
    assert {round(sole_branch(modules2[t]).scale, 9) for t in trained_paths} == {4.0}

    none = tmp_path / "no_alpha.safetensors"
    save_file({k: v for k, v in load_file(path).items() if not k.endswith(".alpha")},
              str(none), metadata={"model_type": "ideogram4"})
    model3 = _Stub()
    _Backend(model3)._load_lora_ideogram4([{"path": str(none), "strength": 1.0}])
    modules3 = dict(model3.named_modules())
    assert {round(sole_branch(modules3[t]).scale, 9) for t in trained_paths} == {1.0}


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


def test_ideogram4_unload_after_a_stack_restores_the_identical_objects(tmp_path):
    path_a, cond_paths, uncond_paths = train_and_save(tmp_path, seed=1234, with_uncond=True)
    path_b, _cb, _ub = train_and_save(tmp_path, name="second.safetensors", seed=4321,
                                      with_uncond=True)

    cond, uncond = _Stub(), _Stub()
    before = {**dict(cond.named_modules()),
              **{f"u::{n}": m for n, m in uncond.named_modules()}}
    backend = _Backend(cond, uncond)
    backend._load_lora_ideogram4([{"path": path_a, "strength": STRENGTH},
                                  {"path": path_b, "strength": STRENGTH_B}])
    assert wrapped_paths(cond) == cond_paths and wrapped_paths(uncond) == uncond_paths

    assert backend._unload_lora_ideogram4() == len(cond_paths) + len(uncond_paths)
    for target in cond_paths:
        assert dict(cond.named_modules())[target] is before[target], target
    for target in uncond_paths:
        assert dict(uncond.named_modules())[target] is before[f"u::{target}"], target
    assert not wrapped_paths(cond) and not wrapped_paths(uncond)
    assert not backend._ideogram4_lora_keys and not backend._ideogram4_lora_keys_uncond

    assert backend._unload_lora_ideogram4() == 0
    for target in cond_paths:
        assert dict(cond.named_modules())[target] is before[target], target


def test_ideogram4_a_leaked_wrapper_is_restored_before_the_next_load(tmp_path):
    """`_ideogram4_cleanup` swallows restore failures, and a leaked composite
    would now SUM into the next request instead of being caught by the stacking
    refusal, so the load restores unconditionally at its top."""
    path, cond_paths, _uncond = train_and_save(tmp_path)

    model = _Stub()
    backend = _Backend(model)
    backend._load_lora_ideogram4([{"path": path, "strength": STRENGTH}])

    reference = _Stub()
    _Backend(reference)._load_lora_ideogram4([{"path": path, "strength": STRENGTH}])

    # Second request, same file: the leak must not double-apply.
    backend._load_lora_ideogram4([{"path": path, "strength": STRENGTH}])
    leaked = dict(model.named_modules())
    clean = dict(reference.named_modules())
    for target in sorted(cond_paths):
        assert len(leaked[target]) == 1, f"{target}: {leaked[target].branch_names}"
        x = torch.randn(3, D)
        assert torch.equal(leaked[target](x), clean[target](x)), target


def test_ideogram4_cleanup_unwraps_on_a_path_that_never_denoised(tmp_path):
    """`_ideogram4_cleanup` is the finally of every generate entry point."""
    path, trained_paths, _uncond = train_and_save(tmp_path)
    model = _Stub()
    backend = _Backend(model)
    backend._load_lora_ideogram4([{"path": path, "strength": 1.0}])
    backend._ideogram4_cleanup(model_key=None, gen_succeeded=False)
    assert not wrapped_paths(model)


def test_ideogram4_a_failure_on_the_uncond_half_leaves_the_cond_half_unwrapped(
        tmp_path, monkeypatch, warnings_seen):
    """Atomic installation across BOTH components, which is new with the session.

    The conditional half used to be applied file by file before the
    unconditional one was even looked at, so a failure there left the model
    running with half a request installed and no record of which half.
    """
    path, cond_paths, uncond_paths = train_and_save(tmp_path, with_uncond=True)
    cond, uncond = _Stub(), _Stub()

    real_add_branch = CompositeAdapterLayer.add_branch
    installed = []

    def fail_once_the_uncond_half_starts(self, name, branch, **kwargs):
        installed.append(name)
        if len(installed) > len(cond_paths):
            raise RuntimeError("install failed on the unconditional branch")
        return real_add_branch(self, name, branch, **kwargs)

    monkeypatch.setattr(CompositeAdapterLayer, "add_branch",
                        fail_once_the_uncond_half_starts)

    with pytest.raises(RuntimeError, match="could not be applied"):
        _Backend(cond, uncond)._load_lora_ideogram4([{"path": path, "strength": 1.0}])

    assert len(installed) > len(cond_paths), "setup: the cond half must have installed"
    assert not wrapped_paths(cond), "the conditional half must have rolled back"
    assert not wrapped_paths(uncond)
    assert "lora_load_failed" in warning_codes(warnings_seen)


def test_ideogram4_restoring_the_uncond_branch_first_still_restores_both(tmp_path):
    """The per-branch separation, demonstrated the only way that bites.

    One shared originals map passes every gate that unloads the conditional half
    first, because that restore pops every key before the unconditional one
    reads it. Varying the order is what exposes it.
    """
    path, cond_paths, uncond_paths = train_and_save(tmp_path, with_uncond=True)
    cond, uncond = _Stub(), _Stub()
    before = {**dict(cond.named_modules()),
              **{f"u::{n}": m for n, m in uncond.named_modules()}}
    backend = _Backend(cond, uncond)
    backend._load_lora_ideogram4([{"path": path, "strength": 1.0}])

    components = backend._ideogram4_lora_components()
    restored = backend._ideogram4_lora_session.unload(list(reversed(components)))

    assert restored == len(cond_paths) + len(uncond_paths)
    for target in uncond_paths:
        assert dict(uncond.named_modules())[target] is before[f"u::{target}"], target
    for target in cond_paths:
        assert dict(cond.named_modules())[target] is before[target], target
    assert not wrapped_paths(cond) and not wrapped_paths(uncond)


def test_ideogram4_missing_file_refuses_and_warns(warnings_seen):
    with pytest.raises(FileNotFoundError, match="not found"):
        _Backend(_Stub())._load_lora_ideogram4([{"path": "no_such_i4_lora.safetensors"}])
    assert "lora_not_found" in warning_codes(warnings_seen)


def test_ideogram4_unreadable_file_refuses_and_warns(tmp_path, warnings_seen, monkeypatch):
    from core.extensions import lora_manager as lm

    broken = tmp_path / "broken.safetensors"
    broken.write_bytes(b"not a safetensors file")
    monkeypatch.setattr(lm.lora_manager, "_resolve_lora_path", lambda p: str(broken))

    model = _Stub()
    with pytest.raises(RuntimeError, match="could not be applied"):
        _Backend(model)._load_lora_ideogram4([{"path": str(broken), "strength": 1.0}])
    assert "lora_load_failed" in warning_codes(warnings_seen)
    assert not wrapped_paths(model)


def test_ideogram4_zero_matched_targets_refuses_and_warns(tmp_path, warnings_seen):
    foreign = tmp_path / "foreign.safetensors"
    save_file({"totally.unrelated.weight": torch.zeros(2, 2)}, str(foreign),
              metadata={"model_type": "not_ideogram4"})

    model = _Stub()
    with pytest.raises(RuntimeError):
        _Backend(model)._load_lora_ideogram4([{"path": str(foreign), "strength": 1.0}])
    assert "lora_incompatible" in warning_codes(warnings_seen)
    assert not wrapped_paths(model)


def test_ideogram4_unmatched_module_is_refused_atomically(tmp_path,
                                                          warnings_seen):
    path, trained_paths, _uncond = train_and_save(tmp_path)
    saved = load_file(path)
    ghost = "lora_unet_" + _flatten_to_sdscripts("layers.9.attention.to_q")
    saved[f"{ghost}.lora_down.weight"] = torch.randn(RANK, D)
    saved[f"{ghost}.lora_up.weight"] = torch.randn(D, RANK)
    extended = tmp_path / "extended.safetensors"
    save_file(saved, str(extended), metadata={"model_type": "ideogram4"})

    model = _Stub()
    with pytest.raises(AdapterIncompatible) as excinfo:
        _Backend(model)._load_lora_ideogram4(
            [{"path": str(extended), "strength": STRENGTH}])
    assert excinfo.value.code == "lora_partial"
    assert not wrapped_paths(model)
    assert "lora_partial" in warning_codes(warnings_seen)


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


def test_ideogram4_model_reload_never_splices_model_a_into_model_b(tmp_path):
    path, cond_paths, uncond_paths = train_and_save(tmp_path, with_uncond=True)
    second, _c2, _u2 = train_and_save(tmp_path, name="second.safetensors", seed=99,
                                      with_uncond=True)

    cond_a, uncond_a = _Stub(), _Stub()
    backend = _Backend(cond_a, uncond_a)
    backend._load_lora_ideogram4([{"path": path, "strength": 1.0},
                                  {"path": second, "strength": STRENGTH_B}])
    a_ids = (module_ids(cond_a) | module_ids(uncond_a)
             | {id(m) for m in backend._ideogram4_lora_orig.values()}
             | {id(m) for m in backend._ideogram4_lora_orig_uncond.values()})

    cond_b, uncond_b = _Stub(seed=8), _Stub(seed=8)
    b_ids_before = module_ids(cond_b) | module_ids(uncond_b)
    assert not (a_ids & b_ids_before), "setup: A and B must not already share modules"

    backend.ideogram4_components = {"transformer": cond_b,
                                    "unconditional_transformer": uncond_b}
    assert backend._ideogram4_lora_keys, "the stale set must be truthy to be a test"
    assert backend._unload_lora_ideogram4() == 0
    assert module_ids(cond_b) | module_ids(uncond_b) == b_ids_before
    assert not ((module_ids(cond_b) | module_ids(uncond_b)) & a_ids)


def test_ideogram4_reloading_only_the_uncond_branch_keeps_the_cond_half(tmp_path):
    """Per-branch bookkeeping over a STACK: swapping one transformer must not
    throw away the other branch's restore, nor splice the old branch into the
    new one."""
    path, cond_paths, uncond_paths = train_and_save(tmp_path, with_uncond=True)
    second, _c2, _u2 = train_and_save(tmp_path, name="second.safetensors", seed=99,
                                      with_uncond=True)

    cond, uncond_a = _Stub(), _Stub()
    before = dict(cond.named_modules())
    backend = _Backend(cond, uncond_a)
    backend._load_lora_ideogram4([{"path": path, "strength": STRENGTH},
                                  {"path": second, "strength": STRENGTH_B}])
    assert all(len(dict(cond.named_modules())[t]) == 2 for t in cond_paths)
    old_uncond_ids = module_ids(uncond_a) | {
        id(m) for m in backend._ideogram4_lora_orig_uncond.values()}

    uncond_b = _Stub(seed=8)
    b_ids_before = module_ids(uncond_b)
    backend.ideogram4_components["unconditional_transformer"] = uncond_b

    assert backend._unload_lora_ideogram4() == len(cond_paths)
    assert not wrapped_paths(cond)
    for target in cond_paths:
        assert dict(cond.named_modules())[target] is before[target], target
    assert module_ids(uncond_b) == b_ids_before
    assert not (module_ids(uncond_b) & old_uncond_ids)
