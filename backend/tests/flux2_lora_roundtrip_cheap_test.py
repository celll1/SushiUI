"""FLUX.2: trainer save -> fresh-generation load round trip, CPU, ~1s.

Drives the REAL ``FLUX2LoRAAdapter`` over BOTH halves it can train -- the
transformer and the Qwen3 text encoder -- then the REAL
``Flux2Mixin._load_lora_flux2`` on freshly built stubs.

The Phase-0 defect this pins: FLUX.2 training could save Qwen text-encoder
adapters, but generation applied transformer tensors only, so the TE half of a
mixed checkpoint was silently inert.

Common composite algebra is covered in
``adapter_lycoris_roundtrip_cheap_test.py``. This file retains FLUX.2's
two-component targeting, unequal lifetimes, quantization, block-swap, and
refusal seams.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/flux2_lora_roundtrip_cheap_test.py -v
"""

import types

import pytest
import torch
from torch import nn
from safetensors.torch import load_file, save_file

from lora_roundtrip_common import (
    LoRALinearLayer, lora_delta, randomise_lora_layers,
    warning_codes, warning_probe,
)

from core.adapters import (  # noqa: E402
    AdapterIncompatible, CompositeAdapterLayer,
)
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


@pytest.fixture
def warnings_seen(monkeypatch):
    return warning_probe(monkeypatch)



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


def test_flux2_shape_mismatched_branch_is_refused_atomically(tmp_path, warnings_seen):
    """A wrong-width pair must refuse atomically, leaving the model bare,
    and warn `lora_partial` (400 Bad Request) according to AdapterSession's contract."""
    path, tf_paths, _te = train_and_save(tmp_path)
    saved = load_file(path)
    victim = sorted(tf_paths)[0]
    stem = "lora_transformer_" + victim.replace(".", "_")
    saved[f"{stem}.lora_down.weight"] = torch.randn(RANK, H + 3)
    broken = tmp_path / "bad_shape.safetensors"
    save_file(saved, str(broken), metadata={"model_type": "flux2"})

    transformer, text_encoder = _Transformer(), _TextEncoder()
    before = dict(transformer.named_modules())
    with pytest.raises(AdapterIncompatible) as excinfo:
        _Backend(transformer, text_encoder)._load_lora_flux2(
            [{"path": str(broken), "strength": STRENGTH}])
    assert excinfo.value.code == "lora_partial"
    assert not wrapped_paths(transformer)
    assert not wrapped_paths(text_encoder)
    assert dict(transformer.named_modules())[victim] is before[victim]
    assert "lora_partial" in warning_codes(warnings_seen)


def test_flux2_unmatched_pair_refuses_partial(tmp_path, warnings_seen):
    """A pair naming a module this model does not have refuses as `lora_partial`,
    leaving the model bare rather than partially applied."""
    path, tf_paths, _te = train_and_save(tmp_path)
    saved = load_file(path)
    ghost = "lora_transformer_transformer_blocks_9_attn_to_q"
    saved[f"{ghost}.lora_down.weight"] = torch.randn(RANK, H)
    saved[f"{ghost}.lora_up.weight"] = torch.randn(H, RANK)
    extended = tmp_path / "extended.safetensors"
    save_file(saved, str(extended), metadata={"model_type": "flux2"})

    transformer, text_encoder = _Transformer(), _TextEncoder()
    with pytest.raises(AdapterIncompatible) as excinfo:
        _Backend(transformer, text_encoder)._load_lora_flux2(
            [{"path": str(extended), "strength": STRENGTH}])
    assert excinfo.value.code == "lora_partial"
    assert not wrapped_paths(transformer)
    assert not wrapped_paths(text_encoder)
    assert "lora_partial" in warning_codes(warnings_seen)



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
