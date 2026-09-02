"""MiniT2I: trainer save -> fresh-generation load round trip, CPU, ~1s.

Drives the REAL ``MiniT2ILoRAAdapter`` over both scopes it can train -- the
MM-JiT transformer and the optional FLAN-T5 text encoder -- then the REAL
two-phase ``MiniT2IMixin`` entry points in the order the generate paths use
them: ``_minit2i_prepare_loras`` -> ``_apply_te_lora_minit2i`` (BEFORE the
prompt encode) -> ``_load_lora_minit2i``.

The ordering is the point: a TE LoRA applied after the encode wraps the right
modules, reports a non-zero count, and changes nothing.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minit2i_lora_roundtrip_cheap_test.py -v
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

from core.models.minit2i.minit2i_lora import (  # noqa: E402
    DEFAULT_SCOPE, TE_NAMESPACE, flatten_to_key, flatten_to_te_key,
    iter_minit2i_lora_targets,
)
from core.pipeline_backends.minit2i import MiniT2IMixin  # noqa: E402
from core.training.adapters.minit2i_adapter import MiniT2ILoRAAdapter  # noqa: E402

D = 8
RANK = 4
ALPHA = 8  # != rank on purpose: the applied scale must be 2.0, not 1.0
SCALE = ALPHA / RANK
STRENGTH = 0.5


def _linear():
    layer = nn.Linear(D, D)
    nn.init.normal_(layer.weight, std=0.05)
    nn.init.normal_(layer.bias, std=0.05)
    return layer


class _Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1, self.w2, self.w3 = _linear(), _linear(), _linear()


class _DoubleBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_qkv, self.txt_qkv = _linear(), _linear()
        self.img_attn_proj, self.txt_attn_proj = _linear(), _linear()
        self.img_mlp, self.txt_mlp = _Mlp(), _Mlp()


class _PreambleBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv, self.attn_proj = _linear(), _linear()
        self.mlp = _Mlp()


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.double_blocks = nn.ModuleList([_DoubleBlock()])
        self.txt_preamble_blocks = nn.ModuleList([_PreambleBlock()])
        self.txt_embedder = _linear()
        self.pooled_embedder = _linear()


class _ModelHolder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = _Net()


class _Transformer(nn.Module):
    """The loader reaches the target scope through ``transformer.model.net``."""

    def __init__(self):
        super().__init__()
        self.model = _ModelHolder()


class _T5SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q, self.k, self.v, self.o = _linear(), _linear(), _linear(), _linear()


class _T5Layer0(nn.Module):
    def __init__(self):
        super().__init__()
        self.SelfAttention = _T5SelfAttention()


class _T5Ff(nn.Module):
    def __init__(self):
        super().__init__()
        self.wi_0, self.wi_1, self.wo = _linear(), _linear(), _linear()


class _T5Layer1(nn.Module):
    def __init__(self):
        super().__init__()
        self.DenseReluDense = _T5Ff()


class _T5Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.ModuleList([_T5Layer0(), _T5Layer1()])


class _T5Encoder(nn.Module):
    def __init__(self, n_blocks=1):
        super().__init__()
        self.block = nn.ModuleList([_T5Block() for _ in range(n_blocks)])


class _TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _T5Encoder()


class _Backend(MiniT2IMixin):
    def __init__(self, transformer, text_encoder=None):
        self.minit2i_components = {"transformer": transformer}
        if text_encoder is not None:
            self.minit2i_components["text_encoder"] = text_encoder


def wrapped_paths(model):
    return {name for name, module in model.named_modules()
            if isinstance(module, LoRALinearLayer)}


def train_and_save(tmp_path, name="minit2i.safetensors", seed=1234, with_te=True):
    """Returns (path, transformer target paths, text-encoder target paths)."""
    transformer = _Transformer()
    text_encoder = _TextEncoder() if with_te else None
    trainer = SimpleNamespace(transformer=transformer, text_encoder=text_encoder,
                              minit2i_variant="b16", config={},
                              repa_enable=False, repa_projector=None)
    adapter = MiniT2ILoRAAdapter(trainer, RANK, ALPHA, torch.float32)
    layers = {}
    n_unet = adapter.apply_lora_to_unet(layers)
    n_te = adapter.apply_lora_to_text_encoders(layers) if with_te else 0
    assert n_unet > 0 and (n_te > 0) == with_te
    randomise_lora_layers(layers, seed=seed, std=0.3)
    out = tmp_path / name
    adapter.save_checkpoint(layers, 7, 1, out)
    return (str(out), wrapped_paths(transformer),
            wrapped_paths(text_encoder) if with_te else set())


def load_both_halves(backend, configs):
    """The entry points' own order: prepare -> TE pass -> transformer pass."""
    prepared = backend._minit2i_prepare_loras(configs)
    n_te = backend._apply_te_lora_minit2i(prepared)
    total = backend._load_lora_minit2i(prepared)
    return n_te, total


@pytest.fixture
def warnings_seen(monkeypatch):
    return warning_probe(monkeypatch)


def test_minit2i_generation_wraps_both_halves_the_trainer_wrapped(tmp_path):
    path, tf_paths, te_paths = train_and_save(tmp_path)
    assert te_paths, "setup: the checkpoint must carry a FLAN-T5 half"

    transformer, text_encoder = _Transformer(), _TextEncoder()
    backend = _Backend(transformer, text_encoder)
    n_te, total = load_both_halves(backend, [{"path": path, "strength": STRENGTH}])

    assert wrapped_paths(transformer) == tf_paths
    assert wrapped_paths(text_encoder) == te_paths
    assert n_te == len(te_paths)
    assert total == len(tf_paths) + len(te_paths)
    assert tf_paths == {p for p, _parent, _attr, _cur
                        in iter_minit2i_lora_targets(_Transformer(), DEFAULT_SCOPE)}
    # The TE half is namespaced in the shared bookkeeping so the two scopes
    # cannot collide on a common dotted path.
    assert {k for k in backend._minit2i_lora_keys if k.startswith(TE_NAMESPACE)} == \
        {TE_NAMESPACE + p for p in te_paths}


def test_minit2i_wrapped_forward_is_base_plus_scaled_branch(tmp_path):
    path, tf_paths, te_paths = train_and_save(tmp_path)
    saved = load_file(path)

    transformer, text_encoder = _Transformer(), _TextEncoder()
    load_both_halves(_Backend(transformer, text_encoder),
                     [{"path": path, "strength": STRENGTH}])

    checked = 0
    for model, to_stem in ((transformer, flatten_to_key), (text_encoder, flatten_to_te_key)):
        modules = dict(model.named_modules())
        for target in sorted(wrapped_paths(model)):
            wrapper = modules[target]
            stem = to_stem(target)
            x = torch.randn(3, D)
            base = wrapper.original_module(x)
            expected = base + lora_delta(saved[f"{stem}.lora_down.weight"],
                                         saved[f"{stem}.lora_up.weight"],
                                         x, ALPHA, RANK, STRENGTH)
            assert torch.allclose(wrapper(x), expected, atol=1e-5), target
            assert not torch.allclose(wrapper(x), base, atol=1e-5), f"{target}: inert"
            checked += 1
    assert checked == len(tf_paths) + len(te_paths)


def test_minit2i_alpha_beats_the_rank_fallback(tmp_path):
    path, tf_paths, te_paths = train_and_save(tmp_path)
    transformer, text_encoder = _Transformer(), _TextEncoder()
    load_both_halves(_Backend(transformer, text_encoder),
                     [{"path": path, "strength": STRENGTH}])
    scales = {round(m.scale, 9) for model in (transformer, text_encoder)
              for m in model.modules() if isinstance(m, LoRALinearLayer)}
    assert scales == {round(SCALE * STRENGTH, 9)}

    md_only = tmp_path / "md_alpha.safetensors"
    save_file({k: v for k, v in load_file(path).items() if not k.endswith(".alpha")},
              str(md_only), metadata={"model_type": "minit2i", "lora_alpha": str(4 * RANK)})
    transformer2, text_encoder2 = _Transformer(), _TextEncoder()
    load_both_halves(_Backend(transformer2, text_encoder2),
                     [{"path": str(md_only), "strength": 1.0}])
    assert {round(m.scale, 9) for model in (transformer2, text_encoder2)
            for m in model.modules() if isinstance(m, LoRALinearLayer)} == {4.0}

    none = tmp_path / "no_alpha.safetensors"
    save_file({k: v for k, v in load_file(path).items() if not k.endswith(".alpha")},
              str(none), metadata={"model_type": "minit2i"})
    transformer3, text_encoder3 = _Transformer(), _TextEncoder()
    load_both_halves(_Backend(transformer3, text_encoder3),
                     [{"path": str(none), "strength": 1.0}])
    assert {round(m.scale, 9) for model in (transformer3, text_encoder3)
            for m in model.modules() if isinstance(m, LoRALinearLayer)} == {1.0}


def test_minit2i_unload_restores_the_identical_objects_and_is_idempotent(tmp_path):
    path, tf_paths, te_paths = train_and_save(tmp_path)

    transformer, text_encoder = _Transformer(), _TextEncoder()
    before = {**dict(transformer.named_modules()),
              **{f"te::{n}": m for n, m in text_encoder.named_modules()}}
    backend = _Backend(transformer, text_encoder)
    _n_te, total = load_both_halves(backend, [{"path": path, "strength": 1.0}])

    assert backend._unload_lora_minit2i() == total
    for target in tf_paths:
        assert dict(transformer.named_modules())[target] is before[target], target
    for target in te_paths:
        assert dict(text_encoder.named_modules())[target] is before[f"te::{target}"], target
    assert not wrapped_paths(transformer) and not wrapped_paths(text_encoder)
    assert backend._unload_lora_minit2i() == 0


def test_minit2i_missing_file_refuses_and_warns(warnings_seen):
    backend = _Backend(_Transformer())
    with pytest.raises(FileNotFoundError, match="not found"):
        backend._minit2i_prepare_loras([{"path": "no_such_minit2i_lora.safetensors"}])
    assert "lora_not_found" in warning_codes(warnings_seen)


def test_minit2i_zero_matched_targets_refuses_and_warns(tmp_path, warnings_seen):
    foreign = tmp_path / "foreign.safetensors"
    save_file({"totally.unrelated.weight": torch.zeros(2, 2)}, str(foreign),
              metadata={"model_type": "not_minit2i"})

    transformer = _Transformer()
    backend = _Backend(transformer)
    with pytest.raises(RuntimeError):
        load_both_halves(backend, [{"path": str(foreign), "strength": 1.0}])
    assert "lora_incompatible" in warning_codes(warnings_seen)
    assert not wrapped_paths(transformer)


def test_minit2i_fully_shadowed_second_lora_refuses_and_warns(tmp_path, warnings_seen):
    path, _tf, _te = train_and_save(tmp_path, with_te=False)
    second, _tf2, _te2 = train_and_save(tmp_path, name="second.safetensors", seed=99,
                                        with_te=False)

    backend = _Backend(_Transformer())
    with pytest.raises(RuntimeError):
        load_both_halves(backend, [{"path": path, "strength": 1.0},
                                   {"path": second, "strength": 1.0}])
    assert "lora_stacking_unsupported" in warning_codes(warnings_seen)


def test_minit2i_model_reload_never_splices_model_a_into_model_b(tmp_path):
    path, tf_paths, te_paths = train_and_save(tmp_path)

    tf_a, te_a = _Transformer(), _TextEncoder()
    backend = _Backend(tf_a, te_a)
    load_both_halves(backend, [{"path": path, "strength": 1.0}])
    a_ids = (module_ids(tf_a) | module_ids(te_a)
             | {id(m) for m in backend._minit2i_lora_orig.values()})

    tf_b, te_b = _Transformer(), _TextEncoder()
    b_ids_before = module_ids(tf_b) | module_ids(te_b)
    assert not (a_ids & b_ids_before), "setup: A and B must not already share modules"

    backend.minit2i_components = {"transformer": tf_b, "text_encoder": te_b}
    assert backend._minit2i_lora_keys, "the stale set must be truthy to be a test"
    assert backend._unload_lora_minit2i() == 0
    assert module_ids(tf_b) | module_ids(te_b) == b_ids_before
    assert not ((module_ids(tf_b) | module_ids(te_b)) & a_ids)

    b_before = {**dict(tf_b.named_modules()),
                **{f"te::{n}": m for n, m in te_b.named_modules()}}
    _n_te, total = load_both_halves(backend, [{"path": path, "strength": 1.0}])
    assert total == len(tf_paths) + len(te_paths)
    assert backend._unload_lora_minit2i() == total
    for target in tf_paths:
        assert dict(tf_b.named_modules())[target] is b_before[target], target
    assert not ((module_ids(tf_b) | module_ids(te_b)) & a_ids)


def test_minit2i_reloading_only_the_text_encoder_keeps_the_transformer_half(tmp_path):
    path, tf_paths, te_paths = train_and_save(tmp_path)

    transformer, te_a = _Transformer(), _TextEncoder()
    backend = _Backend(transformer, te_a)
    load_both_halves(backend, [{"path": path, "strength": 1.0}])
    old_te_ids = module_ids(te_a)

    te_b = _TextEncoder()
    te_b_before = module_ids(te_b)
    backend.minit2i_components["text_encoder"] = te_b

    assert backend._unload_lora_minit2i() == len(tf_paths)
    assert not wrapped_paths(transformer)
    assert module_ids(te_b) == te_b_before
    assert not (module_ids(te_b) & old_te_ids)
