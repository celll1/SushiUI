"""FLUX.2: trainer save -> fresh-generation load round trip, CPU, ~1s.

Drives the REAL ``FLUX2LoRAAdapter`` over BOTH halves it can train -- the
transformer and the Qwen3 text encoder -- then the REAL
``Flux2Mixin._load_lora_flux2`` on freshly built stubs.

The Phase-0 defect this pins: FLUX.2 training could save Qwen text-encoder
adapters, but generation applied transformer tensors only, so the TE half of a
mixed checkpoint was silently inert.

FLUX.2 is on ``CompositeAdapterLayer``, so this file is also the adoption gate:
two LoRAs over one module must SUM, in either selection order, without
perturbing what either one does alone -- and PER COMPONENT, because the two
components have DIFFERENT LIFETIMES. The text encoder's wrappers are torn down
in every generation's ``finally`` (``_restore_flux2_te_lora``) while the
transformer's outlive the generation, so bit-identity, restore identity and the
stack are each asserted for both halves separately. The stacking refusal these
tests used to assert is gone; the numerics that replace it are checked with
``torch.equal``, because a tolerance would hide exactly the reassociation a
"simplification" of the strength folding would introduce.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/flux2_lora_roundtrip_cheap_test.py -v
"""

import types

import pytest
import torch
from torch import nn
from safetensors.torch import load_file, save_file

from lora_roundtrip_common import (
    LoRALinearLayer, lora_delta, module_ids, randomise_lora_layers,
    warning_codes, warning_probe,
)

from core.adapters import CompositeAdapterLayer, lora_branch_dtype  # noqa: E402
from core.pipeline_backends.flux2 import (  # noqa: E402
    Flux2Mixin, _flux2_te_lora_targets, _flux2_transformer_lora_targets,
)
from core.training.adapters.flux2_adapter import FLUX2LoRAAdapter  # noqa: E402

H = 16
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
    layer = nn.Linear(H, H, bias=False)
    nn.init.normal_(layer.weight, std=0.05)
    return layer


class Flux2Attention(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj",
                     "add_v_proj", "to_add_out"):
            setattr(self, name, _linear())
        self.to_out = nn.ModuleList([_linear()])


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Flux2Attention()


class _TransformerModule(nn.Module):
    def __init__(self, n_blocks=2):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_Block() for _ in range(n_blocks)])


class _TeMlp(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(self, name, _linear())


class _TeAttn(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, name, _linear())


class _TeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _TeAttn()
        self.mlp = _TeMlp()


class _TeInner(nn.Module):
    def __init__(self, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([_TeLayer() for _ in range(n_layers)])


class _TextEncoderModule(nn.Module):
    """Qwen3ForCausalLM-shaped: .model.layers[i].{self_attn,mlp}."""

    def __init__(self):
        super().__init__()
        self.model = _TeInner()


def _Transformer(n_blocks=2, seed=7):
    """A stub with SEEDED base weights.

    Gates 2 and 3 compare two independently built models, so unseeded bases turn
    a bit-identity claim about the branch arithmetic into a claim about nothing.
    """
    torch.manual_seed(seed)
    return _TransformerModule(n_blocks)


def _TextEncoder(seed=11):
    torch.manual_seed(seed)
    return _TextEncoderModule()


class _Backend(Flux2Mixin):
    def __init__(self, transformer, text_encoder):
        self.flux2_components = {"transformer": transformer,
                                 "text_encoder": text_encoder, "vae": None}


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


def train_and_save(tmp_path, name="flux2.safetensors", seed=1234, with_te=True):
    """Returns (path, transformer target paths, text-encoder target paths)."""
    transformer, text_encoder = _Transformer(), _TextEncoder()
    trainer = types.SimpleNamespace(
        train_text_encoder=with_te, transformer=transformer,
        text_encoder=text_encoder if with_te else None,
        unet_lr=1e-4, text_encoder_1_lr=1e-5)
    adapter = FLUX2LoRAAdapter(trainer, lora_rank=RANK, lora_alpha=ALPHA,
                               lora_dtype=torch.float32)
    layers = {}
    n_unet = adapter.apply_lora_to_unet(layers)
    n_te = adapter.apply_lora_to_text_encoders(layers) if with_te else 0
    assert n_unet > 0 and (n_te > 0) == with_te
    randomise_lora_layers(layers, seed=seed, std=0.1)
    out = tmp_path / name
    adapter.save_checkpoint(layers, step=1, epoch=0, output_path=out)
    return (str(out), lora_layer_paths(transformer),
            lora_layer_paths(text_encoder) if with_te else set())


def file_branch_tensors(path, target, prefix="lora_transformer_"):
    """``(down, up, alpha)`` straight out of the checkpoint."""
    saved = load_file(path)
    stem = prefix + target.replace(".", "_")
    return (saved[f"{stem}.lora_down.weight"], saved[f"{stem}.lora_up.weight"],
            saved.get(f"{stem}.alpha"))


def pre_composite_reference(base, down, up, alpha, strength):
    """What the loader built BEFORE adoption, tensor for tensor.

    Re-derives the compute dtype exactly as ``_wrap_with_lora_flux2`` does
    rather than assuming fp32, so the gate keeps meaning something on a
    quantized base.
    """
    rank = int(down.shape[0])
    alpha_value = alpha.item() if alpha is not None else rank
    reference = LoRALinearLayer(base, rank=rank, alpha=alpha_value, lora_name="ref")
    dtype = lora_branch_dtype(base)
    with torch.no_grad():
        reference.lora_down.weight.data = down.to(device=base.weight.device, dtype=dtype)
        reference.lora_up.weight.data = up.to(device=base.weight.device, dtype=dtype)
    reference.scale = (alpha_value / rank) * strength
    return reference


@pytest.fixture
def warnings_seen(monkeypatch):
    return warning_probe(monkeypatch)


# ---------------------------------------------------------------------------
# Enumeration and the headline Phase-0 fix
# ---------------------------------------------------------------------------

def test_flux2_generation_wraps_both_halves_the_trainer_wrapped(tmp_path):
    """The headline fix: a mixed checkpoint's text-encoder half must apply."""
    path, tf_paths, te_paths = train_and_save(tmp_path)
    assert te_paths, "setup: the checkpoint must carry a text-encoder half"

    transformer, text_encoder = _Transformer(), _TextEncoder()
    backend = _Backend(transformer, text_encoder)
    backend._load_lora_flux2([{"path": path, "strength": STRENGTH}])

    assert wrapped_paths(transformer) == tf_paths
    assert wrapped_paths(text_encoder) == te_paths
    assert len(backend._flux2_te_lora_wrapped) == len(te_paths)
    assert len(backend._flux2_lora_wrapped_modules) == len(tf_paths) + len(te_paths)


def test_flux2_one_enumerator_covers_exactly_the_trained_targets(tmp_path):
    """Load and unload share these two generators; a target that vanishes from
    them the moment it is occupied is how a second LoRA reports zero matches."""
    path, tf_paths, te_paths = train_and_save(tmp_path)

    transformer, text_encoder = _Transformer(), _TextEncoder()
    bare_tf = {key for _p, _s, key in _flux2_transformer_lora_targets(transformer)}
    bare_te = {key for _p, _a, key, _n in _flux2_te_lora_targets(text_encoder)}
    assert bare_tf == tf_paths
    assert bare_te == {f"text_encoder.{p}" for p in te_paths}

    _Backend(transformer, text_encoder)._load_lora_flux2(
        [{"path": path, "strength": STRENGTH}])
    assert {key for _p, _s, key in _flux2_transformer_lora_targets(transformer)} == bare_tf
    assert {key for _p, _a, key, _n in _flux2_te_lora_targets(text_encoder)} == bare_te


def test_flux2_wrapped_forward_is_base_plus_scaled_branch(tmp_path):
    path, tf_paths, te_paths = train_and_save(tmp_path)

    transformer, text_encoder = _Transformer(), _TextEncoder()
    _Backend(transformer, text_encoder)._load_lora_flux2(
        [{"path": path, "strength": STRENGTH}])

    checked = 0
    for model, paths, prefix in ((transformer, tf_paths, "lora_transformer_"),
                                 (text_encoder, te_paths, "lora_te_")):
        modules = dict(model.named_modules())
        for target in sorted(paths):
            composite = modules[target]
            down, up, _alpha = file_branch_tensors(path, target, prefix)
            x = torch.randn(3, composite.original_module.in_features)
            base = composite.original_module(x)
            expected = base + lora_delta(down, up, x, ALPHA, RANK, STRENGTH)
            assert torch.allclose(composite(x), expected, atol=1e-5), target
            assert not torch.allclose(composite(x), base, atol=1e-5), f"{target}: inert"
            checked += 1
    assert checked == len(tf_paths) + len(te_paths)


# ---------------------------------------------------------------------------
# Gate 4: a single LoRA is bit-identical to the pre-composite wrapper
# ---------------------------------------------------------------------------

def test_flux2_single_lora_is_bit_identical_to_the_pre_composite_wrapper(tmp_path):
    """The composite must not change one-LoRA arithmetic by one ULP.

    Asserted PER COMPONENT: the transformer and the text encoder are wrapped by
    separate passes with separate lifetimes, so a folding mistake could land on
    one of them alone. ``torch.equal``, not a tolerance -- folding the strength
    anywhere but into the branch's own scale reassociates the multiply and shows
    up here and nowhere else.
    """
    path, tf_paths, te_paths = train_and_save(tmp_path)

    transformer, text_encoder = _Transformer(), _TextEncoder()
    _Backend(transformer, text_encoder)._load_lora_flux2(
        [{"path": path, "strength": STRENGTH}])

    checked = 0
    for model, paths, prefix in ((transformer, tf_paths, "lora_transformer_"),
                                 (text_encoder, te_paths, "lora_te_")):
        modules = dict(model.named_modules())
        for target in sorted(paths):
            composite = modules[target]
            base = composite.original_module
            down, up, alpha = file_branch_tensors(path, target, prefix)
            reference = pre_composite_reference(base, down, up, alpha, STRENGTH)

            assert sole_branch(composite).scale == reference.scale, target
            x = torch.randn(3, base.in_features)
            assert torch.equal(composite(x), reference(x)), f"{prefix}{target}"
            checked += 1
    assert checked == len(tf_paths) + len(te_paths)


# ---------------------------------------------------------------------------
# Gates 1-3: the stack
# ---------------------------------------------------------------------------

def test_flux2_two_loras_over_one_module_sum_their_deltas(tmp_path):
    """The adoption gate: base + delta_a + delta_b, against the analytic sum."""
    path_a, tf_paths, te_paths = train_and_save(tmp_path, seed=1234)
    path_b, tf_b, te_b = train_and_save(tmp_path, name="second.safetensors", seed=4321)
    assert (tf_b, te_b) == (tf_paths, te_paths), "both files must cover the same targets"

    transformer, text_encoder = _Transformer(), _TextEncoder()
    _Backend(transformer, text_encoder)._load_lora_flux2(
        [{"path": path_a, "strength": STRENGTH},
         {"path": path_b, "strength": STRENGTH_B}])

    for model, paths, prefix in ((transformer, tf_paths, "lora_transformer_"),
                                 (text_encoder, te_paths, "lora_te_")):
        assert wrapped_paths(model) == paths
        modules = dict(model.named_modules())
        for target in sorted(paths):
            composite = modules[target]
            assert len(composite) == 2, f"{target}: {composite.branch_names}"
            base_module = composite.original_module
            down_a, up_a, _aa = file_branch_tensors(path_a, target, prefix)
            down_b, up_b, _ab = file_branch_tensors(path_b, target, prefix)
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
            assert not torch.allclose(
                composite(x),
                base_module(x) + lora_delta(down_b, up_b, x, ALPHA, RANK, STRENGTH_B),
                atol=1e-5), f"{target}: the first LoRA is inert"


def test_flux2_stacked_result_is_independent_of_selection_order(tmp_path):
    path_a, tf_paths, te_paths = train_and_save(tmp_path, seed=1234)
    path_b, _tf, _te = train_and_save(tmp_path, name="second.safetensors", seed=4321)
    a = {"path": path_a, "strength": STRENGTH}
    b = {"path": path_b, "strength": STRENGTH_B}

    fwd_tf, fwd_te = _Transformer(), _TextEncoder()
    _Backend(fwd_tf, fwd_te)._load_lora_flux2([a, b])
    rev_tf, rev_te = _Transformer(), _TextEncoder()
    _Backend(rev_tf, rev_te)._load_lora_flux2([b, a])

    for one_model, two_model, paths in ((fwd_tf, rev_tf, tf_paths),
                                        (fwd_te, rev_te, te_paths)):
        one_modules = dict(one_model.named_modules())
        two_modules = dict(two_model.named_modules())
        for target in sorted(paths):
            one, two = one_modules[target], two_modules[target]
            assert torch.equal(one.original_module.weight,
                               two.original_module.weight), target
            x = torch.randn(3, one.original_module.in_features)
            # Two branches: the deltas are summed before the base is added, and
            # fp addition commutes, so this is EXACT. (Three or more branches
            # would only hold up to associativity.)
            assert torch.equal(one(x), two(x)), target


def test_flux2_removing_one_branch_leaves_the_other_exactly_as_if_alone(tmp_path):
    """A stacked branch must not perturb its neighbour's own arithmetic."""
    path_a, tf_paths, te_paths = train_and_save(tmp_path, seed=1234)
    path_b, _tf, _te = train_and_save(tmp_path, name="second.safetensors", seed=4321)

    alone_tf, alone_te = _Transformer(), _TextEncoder()
    _Backend(alone_tf, alone_te)._load_lora_flux2(
        [{"path": path_a, "strength": STRENGTH}])

    stacked_tf, stacked_te = _Transformer(), _TextEncoder()
    _Backend(stacked_tf, stacked_te)._load_lora_flux2(
        [{"path": path_a, "strength": STRENGTH},
         {"path": path_b, "strength": STRENGTH_B}])

    for alone, stacked, paths in ((alone_tf, stacked_tf, tf_paths),
                                  (alone_te, stacked_te, te_paths)):
        alone_modules = dict(alone.named_modules())
        stacked_modules = dict(stacked.named_modules())
        for target in sorted(paths):
            one = alone_modules[target]
            two = stacked_modules[target]
            assert torch.equal(one.original_module.weight,
                               two.original_module.weight), target
            two.remove_branch(two.branch_names[1])
            assert two.branch_names == one.branch_names, target
            x = torch.randn(3, one.original_module.in_features)
            assert torch.equal(one(x), two(x)), target


def test_flux2_a_text_encoder_only_second_lora_leaves_the_transformer_alone(tmp_path):
    """The per-component accounting must survive stacking: a file with no
    transformer keys must not make the transformer look like it failed, and it
    must not touch the transformer's branches either."""
    both, tf_paths, te_paths = train_and_save(tmp_path, seed=1234)
    te_only = tmp_path / "te_only.safetensors"
    save_file({k: v for k, v in load_file(both).items() if k.startswith("lora_te_")},
              str(te_only), metadata={"model_type": "flux2"})

    alone_tf, alone_te = _Transformer(), _TextEncoder()
    _Backend(alone_tf, alone_te)._load_lora_flux2([{"path": both, "strength": STRENGTH}])

    transformer, text_encoder = _Transformer(), _TextEncoder()
    _Backend(transformer, text_encoder)._load_lora_flux2(
        [{"path": both, "strength": STRENGTH},
         {"path": str(te_only), "strength": STRENGTH_B}])

    te_modules = dict(text_encoder.named_modules())
    for target in sorted(te_paths):
        assert len(te_modules[target]) == 2, target

    alone_modules = dict(alone_tf.named_modules())
    tf_modules = dict(transformer.named_modules())
    for target in sorted(tf_paths):
        one, two = alone_modules[target], tf_modules[target]
        assert len(two) == 1, f"{target}: {two.branch_names}"
        x = torch.randn(3, one.original_module.in_features)
        assert torch.equal(one(x), two(x)), target


# ---------------------------------------------------------------------------
# Gate 5: restore identity, and the two components' different lifetimes
# ---------------------------------------------------------------------------

def test_flux2_unload_restores_the_identical_objects_and_is_idempotent(tmp_path):
    path, tf_paths, te_paths = train_and_save(tmp_path)

    transformer, text_encoder = _Transformer(), _TextEncoder()
    before = {**dict(transformer.named_modules()), **{
        f"te::{n}": m for n, m in text_encoder.named_modules()}}
    backend = _Backend(transformer, text_encoder)
    backend._load_lora_flux2([{"path": path, "strength": 1.0}])

    # _flux2_cleanup is the finally of every generate entry point.
    backend._flux2_cleanup(gen_succeeded=False)
    assert backend._unload_lora_flux2() == len(tf_paths)

    for target in tf_paths:
        assert dict(transformer.named_modules())[target] is before[target], target
    for target in te_paths:
        assert dict(text_encoder.named_modules())[target] is before[f"te::{target}"], target
    assert not wrapped_paths(transformer) and not wrapped_paths(text_encoder)

    assert backend._unload_lora_flux2() == 0  # second unload: no-op, not a re-splice
    backend._flux2_cleanup(gen_succeeded=True)
    assert not wrapped_paths(transformer) and not wrapped_paths(text_encoder)


def test_flux2_unload_after_a_stack_restores_the_identical_objects(tmp_path):
    path_a, tf_paths, te_paths = train_and_save(tmp_path, seed=1234)
    path_b, _tf, _te = train_and_save(tmp_path, name="second.safetensors", seed=4321)

    transformer, text_encoder = _Transformer(), _TextEncoder()
    before = {**dict(transformer.named_modules()), **{
        f"te::{n}": m for n, m in text_encoder.named_modules()}}
    backend = _Backend(transformer, text_encoder)
    backend._load_lora_flux2([{"path": path_a, "strength": STRENGTH},
                              {"path": path_b, "strength": STRENGTH_B}])
    assert wrapped_paths(transformer) == tf_paths
    assert wrapped_paths(text_encoder) == te_paths

    assert backend._unload_lora_flux2() == len(tf_paths) + len(te_paths)
    for target in tf_paths:
        assert dict(transformer.named_modules())[target] is before[target], target
    for target in te_paths:
        assert dict(text_encoder.named_modules())[target] is before[f"te::{target}"], target
    assert not backend._flux2_lora_wrapped_modules
    assert backend._unload_lora_flux2() == 0


def test_flux2_the_text_encoder_lifetime_is_shorter_than_the_transformers(tmp_path):
    """``_restore_flux2_te_lora`` runs in EVERY generation's finally and must
    take the text encoder's composites down without touching the transformer's."""
    path, tf_paths, te_paths = train_and_save(tmp_path)

    transformer, text_encoder = _Transformer(), _TextEncoder()
    backend = _Backend(transformer, text_encoder)
    backend._load_lora_flux2([{"path": path, "strength": STRENGTH}])

    assert backend._restore_flux2_te_lora() == len(te_paths)
    assert not wrapped_paths(text_encoder)
    assert wrapped_paths(transformer) == tf_paths
    assert backend._restore_flux2_te_lora() == 0  # idempotent


def test_flux2_a_leaked_wrapper_is_restored_before_the_next_load(tmp_path):
    """The load restores unconditionally at its top: without that, a composite
    that outlived its request would now SUM into the next one instead of being
    caught by the stacking refusal."""
    path, tf_paths, _te = train_and_save(tmp_path)

    transformer, text_encoder = _Transformer(), _TextEncoder()
    backend = _Backend(transformer, text_encoder)
    backend._load_lora_flux2([{"path": path, "strength": STRENGTH}])

    ref_tf, ref_te = _Transformer(), _TextEncoder()
    _Backend(ref_tf, ref_te)._load_lora_flux2([{"path": path, "strength": STRENGTH}])

    # Second request, same file, with the previous request's wrappers still in
    # place (no cleanup ran): the leak must not double-apply.
    backend._load_lora_flux2([{"path": path, "strength": STRENGTH}])
    leaked = dict(transformer.named_modules())
    clean = dict(ref_tf.named_modules())
    for target in sorted(tf_paths):
        assert len(leaked[target]) == 1, f"{target}: {leaked[target].branch_names}"
        x = torch.randn(3, leaked[target].original_module.in_features)
        assert torch.equal(leaked[target](x), clean[target](x)), target


# ---------------------------------------------------------------------------
# Alpha precedence, the refusals, and the shape-mismatch skip
# ---------------------------------------------------------------------------

def test_flux2_alpha_beats_the_rank_fallback(tmp_path):
    path, tf_paths, te_paths = train_and_save(tmp_path)
    transformer, text_encoder = _Transformer(), _TextEncoder()
    _Backend(transformer, text_encoder)._load_lora_flux2(
        [{"path": path, "strength": STRENGTH}])
    scales = {round(sole_branch(m).scale, 9)
              for model in (transformer, text_encoder)
              for m in model.modules() if isinstance(m, CompositeAdapterLayer)}
    assert scales == {round(SCALE * STRENGTH, 9)}

    stripped = tmp_path / "no_alpha.safetensors"
    save_file({k: v for k, v in load_file(path).items() if not k.endswith(".alpha")},
              str(stripped), metadata={"model_type": "flux2"})
    transformer2, text_encoder2 = _Transformer(), _TextEncoder()
    _Backend(transformer2, text_encoder2)._load_lora_flux2(
        [{"path": str(stripped), "strength": STRENGTH}])
    scales2 = {round(sole_branch(m).scale, 9)
               for model in (transformer2, text_encoder2)
               for m in model.modules() if isinstance(m, CompositeAdapterLayer)}
    assert scales2 == {round(STRENGTH, 9)}


def test_flux2_missing_file_refuses_and_warns(warnings_seen):
    with pytest.raises(FileNotFoundError):
        _Backend(_Transformer(), _TextEncoder())._load_lora_flux2(
            [{"path": "no_such_flux2_lora.safetensors", "strength": 1.0}])
    assert "lora_not_found" in warning_codes(warnings_seen)


def test_flux2_unreadable_file_refuses_and_warns(tmp_path, warnings_seen, monkeypatch):
    from core.extensions import lora_manager as lm

    broken = tmp_path / "broken.safetensors"
    broken.write_bytes(b"not a safetensors file")
    monkeypatch.setattr(lm.lora_manager, "_resolve_lora_path", lambda p: str(broken))

    transformer, text_encoder = _Transformer(), _TextEncoder()
    with pytest.raises(RuntimeError, match="could not be applied"):
        _Backend(transformer, text_encoder)._load_lora_flux2(
            [{"path": str(broken), "strength": 1.0}])
    assert "lora_load_failed" in warning_codes(warnings_seen)
    assert not wrapped_paths(transformer) and not wrapped_paths(text_encoder)


def test_flux2_unrecognised_file_refuses(tmp_path, warnings_seen):
    junk = tmp_path / "junk.safetensors"
    save_file({"foo.lora_down.weight": torch.zeros(RANK, H),
               "foo.lora_up.weight": torch.zeros(H, RANK)}, str(junk))

    transformer, text_encoder = _Transformer(), _TextEncoder()
    with pytest.raises(RuntimeError, match="no FLUX.2 LoRA tensors found"):
        _Backend(transformer, text_encoder)._load_lora_flux2(
            [{"path": str(junk), "strength": 1.0}])
    assert "lora_incompatible" in warning_codes(warnings_seen)
    assert not wrapped_paths(transformer) and not wrapped_paths(text_encoder)


def test_flux2_component_with_zero_matched_targets_refuses_and_warns(tmp_path,
                                                                     warnings_seen):
    """The refusal is per component, not on the sum: a file whose transformer
    half names modules this model does not have must not pass because its
    text-encoder half happened to bind."""
    path, _tf, _te = train_and_save(tmp_path)
    raw = load_file(path)
    ghost = {}
    for key, value in raw.items():
        if key.startswith("lora_transformer_"):
            ghost["lora_transformer_ghost_" + key[len("lora_transformer_"):]] = value
        else:
            ghost[key] = value
    ghost_path = tmp_path / "ghost_unet.safetensors"
    save_file(ghost, str(ghost_path), metadata={"model_type": "flux2"})

    transformer, text_encoder = _Transformer(), _TextEncoder()
    with pytest.raises(RuntimeError):
        _Backend(transformer, text_encoder)._load_lora_flux2(
            [{"path": str(ghost_path), "strength": 1.0}])
    assert "lora_incompatible" in warning_codes(warnings_seen)


def test_flux2_shape_mismatched_branch_is_skipped_not_assigned(tmp_path, warnings_seen):
    """A wrong-width pair must leave its target BARE -- not carrying an empty
    composite -- and warn `lora_partial` rather than failing in the denoise loop."""
    path, tf_paths, _te = train_and_save(tmp_path)
    saved = load_file(path)
    victim = sorted(tf_paths)[0]
    stem = "lora_transformer_" + victim.replace(".", "_")
    saved[f"{stem}.lora_down.weight"] = torch.randn(RANK, H + 3)
    broken = tmp_path / "bad_shape.safetensors"
    save_file(saved, str(broken), metadata={"model_type": "flux2"})

    transformer, text_encoder = _Transformer(), _TextEncoder()
    _Backend(transformer, text_encoder)._load_lora_flux2(
        [{"path": str(broken), "strength": STRENGTH}])
    assert wrapped_paths(transformer) == tf_paths - {victim}
    assert isinstance(dict(transformer.named_modules())[victim], nn.Linear)
    assert "lora_partial" in warning_codes(warnings_seen)


def test_flux2_unmatched_pair_warns_partial_and_still_generates(tmp_path, warnings_seen):
    """A pair naming a module this model does not have is `lora_partial`, not a
    refusal -- the clause that used to also carry the occupied count."""
    path, tf_paths, _te = train_and_save(tmp_path)
    saved = load_file(path)
    ghost = "lora_transformer_transformer_blocks_9_attn_to_q"
    saved[f"{ghost}.lora_down.weight"] = torch.randn(RANK, H)
    saved[f"{ghost}.lora_up.weight"] = torch.randn(H, RANK)
    extended = tmp_path / "extended.safetensors"
    save_file(saved, str(extended), metadata={"model_type": "flux2"})

    transformer, text_encoder = _Transformer(), _TextEncoder()
    _Backend(transformer, text_encoder)._load_lora_flux2(
        [{"path": str(extended), "strength": STRENGTH}])
    assert wrapped_paths(transformer) == tf_paths
    assert "lora_partial" in warning_codes(warnings_seen)


# ---------------------------------------------------------------------------
# The quantizer gates, over a composite
# ---------------------------------------------------------------------------

def test_flux2_text_encoder_quantization_is_still_dropped_over_a_composite(
        tmp_path, warnings_seen):
    """`_quantize_text_encoder` deep-copies and casts every nn.Linear weight,
    which over a wrapped encoder includes the adapter's own branches."""
    path, _tf, te_paths = train_and_save(tmp_path)
    second, _tf2, _te2 = train_and_save(tmp_path, name="second.safetensors", seed=4321)
    assert te_paths

    transformer, text_encoder = _Transformer(), _TextEncoder()
    backend = _Backend(transformer, text_encoder)
    assert backend._flux2_te_quantization_with_lora("fp8_e4m3fn") == "fp8_e4m3fn"

    backend._load_lora_flux2([{"path": path, "strength": STRENGTH},
                              {"path": second, "strength": STRENGTH_B}])
    assert backend._flux2_te_quantization_with_lora("fp8_e4m3fn") is None
    assert "quantization_fallback" in warning_codes(warnings_seen)

    # ...and comes back once the encoder is unwrapped again.
    backend._restore_flux2_te_lora()
    assert backend._flux2_te_quantization_with_lora("fp8_e4m3fn") == "fp8_e4m3fn"


def test_flux2_int8_refusal_counts_composite_roots_not_branches(tmp_path):
    """The runtime INT8 conversion refuses over a LoRA'd transformer; a
    composite must be ONE hidden slot however many branches it holds."""
    from core.models.common.int8_runtime_quantize import lora_wrapped_count

    path_a, tf_paths, _te = train_and_save(tmp_path, seed=1234)
    path_b, _tf, _te2 = train_and_save(tmp_path, name="second.safetensors", seed=4321)

    transformer, text_encoder = _Transformer(), _TextEncoder()
    assert lora_wrapped_count(transformer) == 0
    backend = _Backend(transformer, text_encoder)
    backend._load_lora_flux2([{"path": path_a, "strength": STRENGTH},
                              {"path": path_b, "strength": STRENGTH_B}])

    assert lora_wrapped_count(transformer) == len(tf_paths)
    assert sum(len(m) for m in transformer.modules()
               if isinstance(m, CompositeAdapterLayer)) == 2 * len(tf_paths)

    backend._unload_lora_flux2()
    assert lora_wrapped_count(transformer) == 0


# ---------------------------------------------------------------------------
# Block swap: the offloader's view of the wrapped tree
# ---------------------------------------------------------------------------

def test_flux2_block_swap_sees_one_base_per_target_and_a_uniform_rename(tmp_path):
    """The offloader selects by ``__class__.__name__.endswith("Linear")`` plus a
    non-None weight and pairs blocks by module path. The composite is named
    ``...Layer``, so the base is enrolled ONCE, at the same path the old wrapper
    put it; the branch weights are new paths, and what has to hold is that the
    per-block path set stays identical ACROSS the swapped blocks."""
    from core.memory_management.block_offloading import linear_weight_dtypes

    path_a, tf_paths, _te = train_and_save(tmp_path, seed=1234)
    path_b, _tf, _te2 = train_and_save(tmp_path, name="second.safetensors", seed=4321)

    transformer, text_encoder = _Transformer(), _TextEncoder()
    backend = _Backend(transformer, text_encoder)
    blocks = transformer.transformer_blocks

    bare = [set(linear_weight_dtypes(b)) for b in blocks]
    assert len({frozenset(s) for s in bare}) == 1

    for configs, per_target in (([{"path": path_a, "strength": STRENGTH}], 1),
                                ([{"path": path_a, "strength": STRENGTH},
                                  {"path": path_b, "strength": STRENGTH_B}], 2)):
        backend._load_lora_flux2(configs)
        sets = [set(linear_weight_dtypes(b)) for b in blocks]
        assert len({frozenset(s) for s in sets}) == 1, "blocks stopped pairing"
        # One base per target, at the SAME path the old wrapper produced.
        for path in sorted(tf_paths):
            block_idx, rest = path.split(".")[1], path.split(".", 2)[2]
            assert f"{rest}.original_module" in sets[int(block_idx)]
            assert rest not in sets[int(block_idx)]
        assert len(sets[0]) == len(bare[0]) + 2 * per_target * len(bare[0])
        # Nothing enrolled twice: named_modules() yields each object once, and
        # the base's first path is the composite's own original_module.
        seen = [id(m) for b in blocks for _n, m in b.named_modules()
                if m.__class__.__name__.endswith("Linear")
                and getattr(m, "weight", None) is not None]
        assert len(seen) == len(set(seen))

    backend._unload_lora_flux2()
    assert [set(linear_weight_dtypes(b)) for b in blocks] == bare


# ---------------------------------------------------------------------------
# Model reload
# ---------------------------------------------------------------------------

def test_flux2_model_reload_never_splices_model_a_into_model_b(tmp_path):
    path, tf_paths, te_paths = train_and_save(tmp_path)
    second, _tf, _te = train_and_save(tmp_path, name="second.safetensors", seed=4321)

    tf_a, te_a = _Transformer(), _TextEncoder()
    backend = _Backend(tf_a, te_a)
    backend._load_lora_flux2([{"path": path, "strength": STRENGTH},
                              {"path": second, "strength": STRENGTH_B}])
    a_ids = (module_ids(tf_a) | module_ids(te_a)
             | {id(m) for m in backend._flux2_lora_original_modules.values()})

    tf_b, te_b = _Transformer(seed=21), _TextEncoder(seed=23)
    b_ids_before = module_ids(tf_b) | module_ids(te_b)
    assert not (a_ids & b_ids_before), "setup: A and B must not already share modules"

    # The model switch: pipeline.py replaces the component dict wholesale.
    backend.flux2_components = {"transformer": tf_b, "text_encoder": te_b, "vae": None}
    backend._flux2_cleanup(gen_succeeded=False)
    backend._unload_lora_flux2()

    assert module_ids(tf_b) | module_ids(te_b) == b_ids_before, "model B was modified"
    assert not ((module_ids(tf_b) | module_ids(te_b)) & a_ids)

    b_before = {**dict(tf_b.named_modules()),
                **{f"te::{n}": m for n, m in te_b.named_modules()}}
    backend._load_lora_flux2([{"path": path, "strength": 1.0}])
    assert wrapped_paths(tf_b) == tf_paths and wrapped_paths(te_b) == te_paths
    backend._flux2_cleanup(gen_succeeded=False)
    backend._unload_lora_flux2()
    for target in tf_paths:
        assert dict(tf_b.named_modules())[target] is b_before[target], target
    for target in te_paths:
        assert dict(te_b.named_modules())[target] is b_before[f"te::{target}"], target
    assert not ((module_ids(tf_b) | module_ids(te_b)) & a_ids)
