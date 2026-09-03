"""ACE-Step 1.5: trainer save -> fresh-generation load round trip, CPU, ~1s.

Drives the REAL ``AceStepLoRAAdapter`` (injection + ``save_checkpoint``) over a
2-layer CPU stub, then the REAL ``AceStepMixin._load_lora_acestep``.

The Phase-0 defect this pins: ACE-Step can train an opt-in MLP scope, while
generation was attention-only, so the MLP half of a self-trained LoRA was
dropped without a word.

ACE-Step is on ``CompositeAdapterLayer``, so this file is also the adoption
gate: two LoRAs over one module must SUM, in either selection order, without
perturbing what either one does alone. The stacking refusal these tests used to
assert is gone; the numerics that replace it are checked with ``torch.equal``,
because a tolerance would hide exactly the reassociation a "simplification" of
the strength folding would introduce.

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
from core.adapters import CompositeAdapterLayer, lora_branch_dtype  # noqa: E402
from core.pipeline_backends.acestep import AceStepMixin  # noqa: E402
from core.training.adapters.acestep_adapter import (  # noqa: E402
    DEFAULT_ACESTEP_SCOPE, AceStepLoRAAdapter, _flatten_to_sdscripts,
    iter_acestep_lora_targets,
)

H, I, N_LAYERS = 16, 32, 2
RANK = 4
# alpha/rank == 1.5 and strength == 0.7, so the applied scale is neither 1.0
# nor either factor: every plausible reassociation of the two moves the bits,
# which is what makes the bit-identity gate below bite at all.
ALPHA = 6
SCALE = ALPHA / RANK
STRENGTH = 0.7
STRENGTH_B = 0.4  # the second LoRA's, so a shared scale shows up as a wrong sum
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


class _LyricEncoder(nn.Module):
    """Only the diffusers/PEFT remap reaches this scope; the trainer's own
    iterator never names it, which is what makes it the test case for the
    enumerator's structural half."""

    def __init__(self, n=1):
        super().__init__()
        self.layers = nn.ModuleList([_Layer() for _ in range(n)])


class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lyric_encoder = _LyricEncoder()


class _Dit(nn.Module):
    def __init__(self, n=N_LAYERS):
        super().__init__()
        self.decoder = _Decoder(n)
        self.encoder = _Encoder()


def build_dit(n=N_LAYERS):
    """A stub with REPRODUCIBLE base weights.

    Gates 2 and 3 compare two separately built models: with the default random
    init they would differ in the base, and every "identical" claim about the
    branches would be swamped by that.
    """
    torch.manual_seed(20260903)
    return _Dit(n)


class _Backend(AceStepMixin):
    def __init__(self, dit):
        self.acestep_components = {"dit": dit}
        self.is_acestep_model = True
        self.device = "cpu"


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


def train_and_save(tmp_path, scope=None, name="acestep.safetensors", seed=1234):
    scope = ATTN_AND_MLP if scope is None else scope
    dit = build_dit()
    adapter = AceStepLoRAAdapter(SimpleNamespace(transformer=dit, config={}),
                                 RANK, ALPHA, torch.float32, scope=scope)
    layers = {}
    count = adapter.apply_lora_to_unet(layers)
    assert adapter.apply_lora_to_text_encoders(layers) == 0, "the Qwen3 TE is frozen"
    assert count == len(layers) > 0
    randomise_lora_layers(layers, seed=seed, std=0.05)
    out = tmp_path / name
    adapter.save_checkpoint(layers, 1, 1, out)
    return str(out), lora_layer_paths(dit)


def file_branch_tensors(path, target):
    """``(down, up)`` straight out of the checkpoint, for the analytic sum."""
    saved = load_file(path)
    stem = "lora_unet_" + _flatten_to_sdscripts(target)
    return saved[f"{stem}.lora_down.weight"], saved[f"{stem}.lora_up.weight"]


@pytest.fixture
def warnings_seen(monkeypatch):
    return warning_probe(monkeypatch)


def test_acestep_generation_wraps_exactly_the_trained_attention_and_mlp_scope(tmp_path):
    """The headline fix: the opt-in MLP scope must survive to generation."""
    path, trained_paths = train_and_save(tmp_path)

    dit = build_dit()
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

    dit = build_dit()
    _Backend(dit)._load_lora_acestep([{"path": path, "strength": STRENGTH}])

    modules = dict(dit.named_modules())
    for target in sorted(trained_paths):
        composite = modules[target]
        down, up = file_branch_tensors(path, target)
        x = torch.randn(3, composite.original_module.in_features)
        base = composite.original_module(x)
        expected = base + lora_delta(down, up, x, ALPHA, RANK, STRENGTH)
        assert torch.allclose(composite(x), expected, atol=1e-5), target
        assert not torch.allclose(composite(x), base, atol=1e-5), f"{target}: branch is inert"


def test_acestep_single_lora_is_bit_identical_to_the_pre_composite_wrapper(tmp_path):
    """The composite must not change one-LoRA arithmetic by one ULP.

    The reference is what the loader built before adoption: a ``LoRALinearLayer``
    over the same base, its weights copied the same way, its scale written as
    ``(alpha / rank) * strength``. ``torch.equal``, not a tolerance -- folding
    the strength anywhere but into the branch's own scale reassociates the
    multiply and shows up here and nowhere else.
    """
    path, trained_paths = train_and_save(tmp_path)

    dit = build_dit()
    _Backend(dit)._load_lora_acestep([{"path": path, "strength": STRENGTH}])

    modules = dict(dit.named_modules())
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


def test_acestep_two_loras_over_one_module_sum_their_deltas(tmp_path):
    """The adoption gate: base + delta_a + delta_b, against the analytic sum."""
    path_a, trained_paths = train_and_save(tmp_path, seed=1234)
    path_b, paths_b = train_and_save(tmp_path, name="second.safetensors", seed=4321)
    assert paths_b == trained_paths, "both files must cover the same targets to stack"

    dit = build_dit()
    _Backend(dit)._load_lora_acestep([{"path": path_a, "strength": STRENGTH},
                                      {"path": path_b, "strength": STRENGTH_B}])

    assert wrapped_paths(dit) == trained_paths
    modules = dict(dit.named_modules())
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


def test_acestep_stacked_result_is_independent_of_selection_order(tmp_path):
    path_a, trained_paths = train_and_save(tmp_path, seed=1234)
    path_b, _pb = train_and_save(tmp_path, name="second.safetensors", seed=4321)
    a = {"path": path_a, "strength": STRENGTH}
    b = {"path": path_b, "strength": STRENGTH_B}

    forward = build_dit()
    _Backend(forward)._load_lora_acestep([a, b])
    reverse = build_dit()
    _Backend(reverse)._load_lora_acestep([b, a])

    forward_modules = dict(forward.named_modules())
    reverse_modules = dict(reverse.named_modules())
    for target in sorted(trained_paths):
        one, two = forward_modules[target], reverse_modules[target]
        assert torch.equal(one.original_module.weight, two.original_module.weight), target
        x = torch.randn(3, one.original_module.in_features)
        # Two branches: the deltas are summed before the base is added, and fp
        # addition commutes, so this is EXACT. (Three or more branches would
        # only hold up to associativity.)
        assert torch.equal(one(x), two(x)), target


def test_acestep_removing_one_branch_leaves_the_other_exactly_as_if_alone(tmp_path):
    """A stacked branch must not perturb its neighbour's own arithmetic."""
    path_a, trained_paths = train_and_save(tmp_path, seed=1234)
    path_b, _pb = train_and_save(tmp_path, name="second.safetensors", seed=4321)

    alone = build_dit()
    _Backend(alone)._load_lora_acestep([{"path": path_a, "strength": STRENGTH}])

    stacked = build_dit()
    _Backend(stacked)._load_lora_acestep([{"path": path_a, "strength": STRENGTH},
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


def test_acestep_selecting_the_same_file_twice_is_two_branches(tmp_path):
    """Branch names carry the request index, so a duplicate selection is not a
    duplicate-name refusal."""
    path, trained_paths = train_and_save(tmp_path)

    dit = build_dit()
    _Backend(dit)._load_lora_acestep([{"path": path, "strength": STRENGTH},
                                      {"path": path, "strength": STRENGTH}])
    modules = dict(dit.named_modules())
    for target in sorted(trained_paths):
        assert len(modules[target]) == 2, target
        assert len(set(modules[target].branch_names)) == 2, target


def test_acestep_alpha_beats_the_rank_fallback(tmp_path):
    path, trained_paths = train_and_save(tmp_path)
    dit = build_dit()
    _Backend(dit)._load_lora_acestep([{"path": path, "strength": STRENGTH}])
    modules = dict(dit.named_modules())
    assert {round(sole_branch(modules[t]).scale, 9) for t in trained_paths} == \
        {round(SCALE * STRENGTH, 9)}

    stripped = tmp_path / "no_alpha.safetensors"
    save_file({k: v for k, v in load_file(path).items() if not k.endswith(".alpha")},
              str(stripped), metadata={"model_type": "acestep"})
    dit2 = build_dit()
    _Backend(dit2)._load_lora_acestep([{"path": str(stripped), "strength": STRENGTH}])
    modules2 = dict(dit2.named_modules())
    assert {round(sole_branch(modules2[t]).scale, 9) for t in trained_paths} == \
        {round(STRENGTH, 9)}


def test_acestep_unload_restores_the_identical_objects_and_is_idempotent(tmp_path):
    path, trained_paths = train_and_save(tmp_path)

    dit = build_dit()
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


def test_acestep_unload_after_a_stack_restores_the_identical_objects(tmp_path):
    path_a, trained_paths = train_and_save(tmp_path, seed=1234)
    path_b, _pb = train_and_save(tmp_path, name="second.safetensors", seed=4321)

    dit = build_dit()
    before = dict(dit.named_modules())
    backend = _Backend(dit)
    backend._load_lora_acestep([{"path": path_a, "strength": STRENGTH},
                                {"path": path_b, "strength": STRENGTH_B}])
    assert wrapped_paths(dit) == trained_paths

    backend._unload_lora_acestep()
    after = dict(dit.named_modules())
    for target in trained_paths:
        assert after[target] is before[target], target
    assert not wrapped_paths(dit)
    assert not backend._acestep_lora_wrapped_modules

    backend._unload_lora_acestep()
    assert dict(dit.named_modules()) == after


def test_acestep_missing_file_refuses(warnings_seen):
    with pytest.raises(ValidationError):
        _Backend(build_dit())._load_lora_acestep(
            [{"path": "no_such_acestep_lora.safetensors"}])


def test_acestep_missing_file_warns(warnings_seen):
    with pytest.raises(ValidationError):
        _Backend(build_dit())._load_lora_acestep(
            [{"path": "no_such_acestep_lora.safetensors"}])
    assert "lora_not_found" in warning_codes(warnings_seen)


def test_acestep_unreadable_file_refuses(tmp_path, warnings_seen):
    from api.error_handlers import GenerationError

    broken = tmp_path / "broken.safetensors"
    broken.write_bytes(b"not a safetensors file")
    with pytest.raises(GenerationError):
        _Backend(build_dit())._load_lora_acestep([{"path": str(broken)}])


def test_acestep_zero_matched_targets_refuses_and_warns(tmp_path, warnings_seen):
    ghost = tmp_path / "ghost.safetensors"
    stem = "lora_unet_decoder_layers_0_self_attn_norm"
    save_file({f"{stem}.lora_down.weight": torch.zeros(RANK, H),
               f"{stem}.lora_up.weight": torch.zeros(H, RANK)},
              str(ghost), metadata={"model_type": "acestep"})

    dit = build_dit()
    with pytest.raises(ValidationError):
        _Backend(dit)._load_lora_acestep([{"path": str(ghost), "strength": 1.0}])
    assert "lora_incompatible" in warning_codes(warnings_seen)
    assert not wrapped_paths(dit)


def test_acestep_shape_mismatched_branch_is_skipped_never_assigned(tmp_path, warnings_seen):
    """One target's tensors are the wrong width: it stays a bare Linear (not an
    empty composite), the rest apply, and the request warns rather than failing
    in the denoise loop."""
    path, trained_paths = train_and_save(tmp_path)
    victim = sorted(trained_paths)[0]
    stem = "lora_unet_" + _flatten_to_sdscripts(victim)
    saved = load_file(path)
    saved[f"{stem}.lora_down.weight"] = torch.randn(RANK, 3)
    broken = tmp_path / "broken_shape.safetensors"
    save_file(saved, str(broken), metadata={"model_type": "acestep"})

    dit = build_dit()
    before = dict(dit.named_modules())
    _Backend(dit)._load_lora_acestep([{"path": str(broken), "strength": STRENGTH}])

    assert wrapped_paths(dit) == trained_paths - {victim}
    assert dict(dit.named_modules())[victim] is before[victim]
    assert "lora_partial" in warning_codes(warnings_seen)


def test_acestep_two_loras_over_the_same_targets_stack_instead_of_refusing(
        tmp_path, warnings_seen):
    first, trained_paths = train_and_save(tmp_path)
    second, _paths2 = train_and_save(tmp_path, name="second.safetensors", seed=99)

    dit = build_dit()
    _Backend(dit)._load_lora_acestep([{"path": first, "strength": STRENGTH},
                                      {"path": second, "strength": STRENGTH_B}])
    assert wrapped_paths(dit) == trained_paths
    assert warning_codes(warnings_seen) == []


def test_acestep_disjoint_scopes_stack_additively(tmp_path, warnings_seen):
    attn, attn_paths = train_and_save(tmp_path, scope=DEFAULT_ACESTEP_SCOPE,
                                      name="attn.safetensors")
    mlp, mlp_paths = train_and_save(tmp_path, scope=MLP_ONLY, name="mlp.safetensors", seed=7)
    assert not (attn_paths & mlp_paths)

    dit = build_dit()
    backend = _Backend(dit)
    backend._load_lora_acestep([{"path": attn}, {"path": mlp}])
    assert wrapped_paths(dit) == attn_paths | mlp_paths
    assert warning_codes(warnings_seen) == []


LYRIC_TARGETS = {f"encoder.lyric_encoder.layers.0.self_attn.{leaf}_proj"
                 for leaf in ("q", "k", "v")}


def diffusers_lyric_lora(tmp_path, name, seed):
    """An EXTERNAL diffusers/PEFT-format file targeting the lyric encoder only."""
    g = torch.Generator().manual_seed(seed)
    tensors = {}
    for leaf in ("q", "k", "v"):
        stem = f"lyric_encoder.encoders.0.self_attn.linear_{leaf}"
        tensors[f"{stem}.lora_A.weight"] = torch.randn(RANK, H, generator=g) * 0.05
        tensors[f"{stem}.lora_B.weight"] = torch.randn(H, RANK, generator=g) * 0.05
    out = tmp_path / name
    save_file(tensors, str(out))
    return str(out)


def test_acestep_diffusers_format_lyric_scope_stacks_and_restores(tmp_path, warnings_seen):
    """The foreign codec reaches a scope the trainer's iterator never names, so
    only the enumerator's structural half can restore what it installed."""
    a = diffusers_lyric_lora(tmp_path, "peft_a.safetensors", 11)
    b = diffusers_lyric_lora(tmp_path, "peft_b.safetensors", 22)

    dit = build_dit()
    before = dict(dit.named_modules())
    backend = _Backend(dit)
    backend._load_lora_acestep([{"path": a, "strength": STRENGTH},
                                {"path": b, "strength": STRENGTH_B}])

    assert wrapped_paths(dit) == LYRIC_TARGETS
    modules = dict(dit.named_modules())
    for target in sorted(LYRIC_TARGETS):
        assert len(modules[target]) == 2, target

    backend._unload_lora_acestep()
    assert not wrapped_paths(dit)
    assert not backend._acestep_lora_wrapped_modules
    for target in LYRIC_TARGETS:
        assert dict(dit.named_modules())[target] is before[target], target


def test_acestep_model_reload_never_splices_model_a_into_model_b(tmp_path):
    path, trained_paths = train_and_save(tmp_path)

    dit_a = build_dit()
    backend = _Backend(dit_a)
    backend._load_lora_acestep([{"path": path, "strength": 1.0}])
    a_ids = module_ids(dit_a) | {id(m) for m in backend._acestep_lora_original_modules.values()}

    dit_b = _Dit()  # a DIFFERENT random init, so no module can be shared by accident
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
