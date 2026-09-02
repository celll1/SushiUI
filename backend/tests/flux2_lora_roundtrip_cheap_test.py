"""FLUX.2: trainer save -> fresh-generation load round trip, CPU, ~1s.

Drives the REAL ``FLUX2LoRAAdapter`` over BOTH halves it can train -- the
transformer and the Qwen3 text encoder -- then the REAL
``Flux2Mixin._load_lora_flux2`` on freshly built stubs.

The Phase-0 defect this pins: FLUX.2 training could save Qwen text-encoder
adapters, but generation applied transformer tensors only, so the TE half of a
mixed checkpoint was silently inert.

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

from core.pipeline_backends.flux2 import Flux2Mixin  # noqa: E402
from core.training.adapters.flux2_adapter import FLUX2LoRAAdapter  # noqa: E402

H = 16
RANK = 4
ALPHA = 8  # != rank on purpose: the applied scale must be 2.0, not 1.0
SCALE = ALPHA / RANK
STRENGTH = 0.5


class Flux2Attention(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj",
                     "add_v_proj", "to_add_out"):
            setattr(self, name, nn.Linear(H, H, bias=False))
        self.to_out = nn.ModuleList([nn.Linear(H, H, bias=False)])


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Flux2Attention()


class _Transformer(nn.Module):
    def __init__(self, n_blocks=2):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_Block() for _ in range(n_blocks)])


class _TeMlp(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(self, name, nn.Linear(H, H, bias=False))


class _TeAttn(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, name, nn.Linear(H, H, bias=False))


class _TeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _TeAttn()
        self.mlp = _TeMlp()


class _TeInner(nn.Module):
    def __init__(self, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([_TeLayer() for _ in range(n_layers)])


class _TextEncoder(nn.Module):
    """Qwen3ForCausalLM-shaped: .model.layers[i].{self_attn,mlp}."""

    def __init__(self):
        super().__init__()
        self.model = _TeInner()


class _Backend(Flux2Mixin):
    def __init__(self, transformer, text_encoder):
        self.flux2_components = {"transformer": transformer,
                                 "text_encoder": text_encoder, "vae": None}


def wrapped_paths(model):
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
    return (str(out), wrapped_paths(transformer),
            wrapped_paths(text_encoder) if with_te else set())


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


def test_flux2_wrapped_forward_is_base_plus_scaled_branch(tmp_path):
    path, tf_paths, te_paths = train_and_save(tmp_path)
    saved = load_file(path)

    transformer, text_encoder = _Transformer(), _TextEncoder()
    _Backend(transformer, text_encoder)._load_lora_flux2(
        [{"path": path, "strength": STRENGTH}])

    checked = 0
    for model, prefix in ((transformer, "lora_transformer_"), (text_encoder, "lora_te_")):
        modules = dict(model.named_modules())
        for target in sorted(wrapped_paths(model)):
            wrapper = modules[target]
            stem = prefix + target.replace(".", "_")
            x = torch.randn(3, wrapper.original_module.in_features)
            base = wrapper.original_module(x)
            expected = base + lora_delta(saved[f"{stem}.lora_down.weight"],
                                         saved[f"{stem}.lora_up.weight"],
                                         x, ALPHA, RANK, STRENGTH)
            assert torch.allclose(wrapper(x), expected, atol=1e-5), target
            assert not torch.allclose(wrapper(x), base, atol=1e-5), f"{target}: inert"
            checked += 1
    assert checked == len(tf_paths) + len(te_paths)


def test_flux2_alpha_beats_the_rank_fallback(tmp_path):
    path, tf_paths, te_paths = train_and_save(tmp_path)
    transformer, text_encoder = _Transformer(), _TextEncoder()
    _Backend(transformer, text_encoder)._load_lora_flux2(
        [{"path": path, "strength": STRENGTH}])
    scales = {round(m.scale, 9) for model in (transformer, text_encoder)
              for m in model.modules() if isinstance(m, LoRALinearLayer)}
    assert scales == {round(SCALE * STRENGTH, 9)}

    stripped = tmp_path / "no_alpha.safetensors"
    save_file({k: v for k, v in load_file(path).items() if not k.endswith(".alpha")},
              str(stripped), metadata={"model_type": "flux2"})
    transformer2, text_encoder2 = _Transformer(), _TextEncoder()
    _Backend(transformer2, text_encoder2)._load_lora_flux2(
        [{"path": str(stripped), "strength": STRENGTH}])
    scales2 = {round(m.scale, 9) for model in (transformer2, text_encoder2)
               for m in model.modules() if isinstance(m, LoRALinearLayer)}
    assert scales2 == {round(STRENGTH, 9)}


def test_flux2_unload_restores_the_identical_objects_and_is_idempotent(tmp_path):
    path, tf_paths, te_paths = train_and_save(tmp_path)

    transformer, text_encoder = _Transformer(), _TextEncoder()
    before = {**dict(transformer.named_modules()), **{
        f"te::{n}": m for n, m in text_encoder.named_modules()}}
    backend = _Backend(transformer, text_encoder)
    backend._load_lora_flux2([{"path": path, "strength": 1.0}])

    # _flux2_cleanup is the finally of every generate entry point.
    backend._flux2_cleanup(gen_succeeded=False)
    backend._unload_lora_flux2()

    for target in tf_paths:
        assert dict(transformer.named_modules())[target] is before[target], target
    for target in te_paths:
        assert dict(text_encoder.named_modules())[target] is before[f"te::{target}"], target
    assert not wrapped_paths(transformer) and not wrapped_paths(text_encoder)

    backend._unload_lora_flux2()  # second unload: no-op, not a re-splice
    backend._flux2_cleanup(gen_succeeded=True)
    assert not wrapped_paths(transformer) and not wrapped_paths(text_encoder)


def test_flux2_missing_file_refuses_and_warns(warnings_seen):
    with pytest.raises(FileNotFoundError):
        _Backend(_Transformer(), _TextEncoder())._load_lora_flux2(
            [{"path": "no_such_flux2_lora.safetensors", "strength": 1.0}])
    assert "lora_not_found" in warning_codes(warnings_seen)


def test_flux2_unrecognised_file_refuses(tmp_path, warnings_seen):
    junk = tmp_path / "junk.safetensors"
    save_file({"foo.lora_down.weight": torch.zeros(RANK, H),
               "foo.lora_up.weight": torch.zeros(H, RANK)}, str(junk))

    transformer, text_encoder = _Transformer(), _TextEncoder()
    with pytest.raises(RuntimeError, match="no FLUX.2 LoRA tensors found"):
        _Backend(transformer, text_encoder)._load_lora_flux2(
            [{"path": str(junk), "strength": 1.0}])
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


def test_flux2_fully_shadowed_second_lora_refuses_and_warns(tmp_path, warnings_seen):
    path, _tf, _te = train_and_save(tmp_path)
    second, _tf2, _te2 = train_and_save(tmp_path, name="second.safetensors", seed=99)

    with pytest.raises(RuntimeError, match="already wrapped"):
        _Backend(_Transformer(), _TextEncoder())._load_lora_flux2(
            [{"path": path, "strength": 1.0}, {"path": second, "strength": 1.0}])
    assert "lora_stacking_unsupported" in warning_codes(warnings_seen)


def test_flux2_model_reload_never_splices_model_a_into_model_b(tmp_path):
    path, tf_paths, te_paths = train_and_save(tmp_path)

    tf_a, te_a = _Transformer(), _TextEncoder()
    backend = _Backend(tf_a, te_a)
    backend._load_lora_flux2([{"path": path, "strength": 1.0}])
    a_ids = (module_ids(tf_a) | module_ids(te_a)
             | {id(m) for m in backend._flux2_lora_original_modules.values()})

    tf_b, te_b = _Transformer(), _TextEncoder()
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
