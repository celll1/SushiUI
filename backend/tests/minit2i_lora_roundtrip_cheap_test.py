"""MiniT2I: trainer save -> fresh-generation load round trip, CPU, ~1s.

Drives the REAL ``MiniT2ILoRAAdapter`` over both scopes it can train -- the
MM-JiT transformer and the optional FLAN-T5 text encoder -- then the REAL
two-phase ``MiniT2IMixin`` entry points in the order the generate paths use
them: ``_minit2i_prepare_loras`` -> ``_apply_te_lora_minit2i`` (BEFORE the
prompt encode) -> ``_load_lora_minit2i``.

The ordering is the point: a TE LoRA applied after the encode wraps the right
modules, reports a non-zero count, and changes nothing.

Common composite algebra is covered in
``adapter_lycoris_roundtrip_cheap_test.py``. This file retains MiniT2I's
two-phase transformer/text-encoder ordering, rollback, lifetime, and refusal
seams.

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

from core.adapters import AdapterIncompatible, CompositeAdapterLayer  # noqa: E402
from core.models.minit2i.minit2i_lora import (  # noqa: E402
    DEFAULT_SCOPE, TE_NAMESPACE, flatten_to_key, flatten_to_te_key,
    iter_minit2i_lora_targets,
)
from core.pipeline_backends.minit2i import MiniT2IMixin  # noqa: E402
from core.training.adapters.minit2i_adapter import MiniT2ILoRAAdapter  # noqa: E402

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


class _TransformerModule(nn.Module):
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


class _TextEncoderModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _T5Encoder()


def _Transformer(seed=7):
    """Stubs with SEEDED base weights: gates 2 and 3 compare two models."""
    torch.manual_seed(seed)
    return _TransformerModule()


def _TextEncoder(seed=11):
    torch.manual_seed(seed)
    return _TextEncoderModule()


class _Backend(MiniT2IMixin):
    def __init__(self, transformer, text_encoder=None):
        self.minit2i_components = {"transformer": transformer}
        if text_encoder is not None:
            self.minit2i_components["text_encoder"] = text_encoder


def wrapped_paths(model):
    """Target paths a GENERATION load covers, i.e. the composite roots."""
    return {name for name, module in model.named_modules()
            if isinstance(module, CompositeAdapterLayer)}


def lora_layer_paths(model):
    """Paths the TRAINER wrapped -- it still installs plain wrappers."""
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
    return (str(out), lora_layer_paths(transformer),
            lora_layer_paths(text_encoder) if with_te else set())


def file_branch_tensors(path, target, to_stem):
    """``(down, up)`` straight out of the checkpoint, for the analytic sum."""
    saved = load_file(path)
    stem = to_stem(target)
    return saved[f"{stem}.lora_down.weight"], saved[f"{stem}.lora_up.weight"]


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


def test_minit2i_the_te_half_is_installed_by_the_pre_encode_pass(tmp_path):
    """The ordering `5d80c042` fixed: the TE composites must already be in place
    when `_apply_te_lora_minit2i` returns, because `_minit2i_encode` runs next
    and its embeddings are never recomputed."""
    path, tf_paths, te_paths = train_and_save(tmp_path)

    transformer, text_encoder = _Transformer(), _TextEncoder()
    backend = _Backend(transformer, text_encoder)
    prepared = backend._minit2i_prepare_loras([{"path": path, "strength": STRENGTH}])
    assert backend._apply_te_lora_minit2i(prepared) == len(te_paths)

    assert wrapped_paths(text_encoder) == te_paths
    assert not wrapped_paths(transformer), "the transformer pass belongs after staging"


def test_minit2i_wrapped_forward_is_base_plus_scaled_branch(tmp_path):
    path, tf_paths, te_paths = train_and_save(tmp_path)

    transformer, text_encoder = _Transformer(), _TextEncoder()
    load_both_halves(_Backend(transformer, text_encoder),
                     [{"path": path, "strength": STRENGTH}])

    checked = 0
    for model, to_stem in ((transformer, flatten_to_key), (text_encoder, flatten_to_te_key)):
        modules = dict(model.named_modules())
        for target in sorted(wrapped_paths(model)):
            wrapper = modules[target]
            down, up = file_branch_tensors(path, target, to_stem)
            x = torch.randn(3, D)
            base = wrapper.original_module(x)
            expected = base + lora_delta(down, up, x, ALPHA, RANK, STRENGTH)
            assert torch.allclose(wrapper(x), expected, atol=1e-5), target
            assert not torch.allclose(wrapper(x), base, atol=1e-5), f"{target}: inert"
            checked += 1
    assert checked == len(tf_paths) + len(te_paths)


def test_minit2i_a_stack_on_the_transformer_leaves_the_text_encoder_untouched(tmp_path):
    """Per-component lifecycle: the two components hold SEPARATE composites, and
    the shared bookkeeping is partitioned by the ``te::`` namespace.

    A second file with no TE keys must add a branch to the transformer composites
    and leave the text encoder's bit-identical to the single-LoRA model.
    """
    both, tf_paths, te_paths = train_and_save(tmp_path, seed=1234)
    tf_only, _tp, _ep = train_and_save(tmp_path, name="tf_only.safetensors", seed=4321,
                                       with_te=False)

    alone_tf, alone_te = _Transformer(), _TextEncoder()
    load_both_halves(_Backend(alone_tf, alone_te), [{"path": both, "strength": STRENGTH}])

    transformer, text_encoder = _Transformer(), _TextEncoder()
    load_both_halves(_Backend(transformer, text_encoder),
                     [{"path": both, "strength": STRENGTH},
                      {"path": tf_only, "strength": STRENGTH_B}])

    tf_modules = dict(transformer.named_modules())
    for target in sorted(tf_paths):
        assert len(tf_modules[target]) == 2, target

    alone_modules = dict(alone_te.named_modules())
    te_modules = dict(text_encoder.named_modules())
    for target in sorted(te_paths):
        one, two = alone_modules[target], te_modules[target]
        assert len(two) == 1, f"{target}: {two.branch_names}"
        x = torch.randn(3, D)
        assert torch.equal(one(x), two(x)), target


def test_minit2i_a_leaked_wrapper_is_restored_before_the_next_load(tmp_path):
    """`_minit2i_cleanup` swallows restore failures, and a leaked composite would
    now SUM into the next request instead of being caught by the stacking
    refusal, so `_minit2i_prepare_loras` restores unconditionally at its top."""
    path, tf_paths, te_paths = train_and_save(tmp_path)

    transformer, text_encoder = _Transformer(), _TextEncoder()
    backend = _Backend(transformer, text_encoder)
    load_both_halves(backend, [{"path": path, "strength": STRENGTH}])

    ref_tf, ref_te = _Transformer(), _TextEncoder()
    load_both_halves(_Backend(ref_tf, ref_te), [{"path": path, "strength": STRENGTH}])

    # Second request, same file: the leak must not double-apply, in EITHER half.
    load_both_halves(backend, [{"path": path, "strength": STRENGTH}])
    for leaked_model, clean_model, paths in ((transformer, ref_tf, tf_paths),
                                             (text_encoder, ref_te, te_paths)):
        leaked = dict(leaked_model.named_modules())
        clean = dict(clean_model.named_modules())
        for target in sorted(paths):
            assert len(leaked[target]) == 1, f"{target}: {leaked[target].branch_names}"
            x = torch.randn(3, D)
            assert torch.equal(leaked[target](x), clean[target](x)), target


def test_minit2i_a_failed_transformer_pass_rolls_back_only_that_pass(
        tmp_path, monkeypatch, warnings_seen):
    """Atomicity here is per PASS, and deliberately so.

    The text-encoder half is installed before the prompt encode and the
    transformer half only after staging, so one ``AdapterSession.load`` cannot
    span them. The transformer pass still rolls ITSELF back completely -- new
    with the session -- while the already-committed TE half stays, and
    ``_minit2i_cleanup``'s unconditional unload is what takes that back down.
    """
    path, tf_paths, te_paths = train_and_save(tmp_path)
    transformer, text_encoder = _Transformer(), _TextEncoder()
    backend = _Backend(transformer, text_encoder)

    prepared = backend._minit2i_prepare_loras([{"path": path, "strength": STRENGTH}])
    assert backend._apply_te_lora_minit2i(prepared) == len(te_paths)

    def always_fail(self, name, branch, **kwargs):
        raise RuntimeError("install failed on the transformer")

    monkeypatch.setattr(CompositeAdapterLayer, "add_branch", always_fail)
    with pytest.raises(RuntimeError, match="could not be applied"):
        backend._load_lora_minit2i(prepared)
    monkeypatch.undo()

    assert not wrapped_paths(transformer), "the transformer pass must roll back"
    assert wrapped_paths(text_encoder) == te_paths, "the committed TE pass stays"
    assert "lora_load_failed" in warning_codes(warnings_seen)
    assert backend._unload_lora_minit2i() == len(te_paths)
    assert not wrapped_paths(text_encoder)


def test_minit2i_unload_restores_the_te_when_the_transformer_is_gone(tmp_path):
    """A component that is not loaded is expressed as ``module=None``, which
    resets that component's bookkeeping and leaves the other's restorable."""
    path, tf_paths, te_paths = train_and_save(tmp_path)
    transformer, text_encoder = _Transformer(), _TextEncoder()
    before = dict(text_encoder.named_modules())
    backend = _Backend(transformer, text_encoder)
    load_both_halves(backend, [{"path": path, "strength": STRENGTH}])

    backend.minit2i_components.pop("transformer")

    assert backend._unload_lora_minit2i() == len(te_paths)
    for target in te_paths:
        assert dict(text_encoder.named_modules())[target] is before[target], target
    assert not wrapped_paths(text_encoder)
    assert not backend._minit2i_lora_keys


def test_minit2i_missing_file_refuses_and_warns(warnings_seen):
    backend = _Backend(_Transformer())
    with pytest.raises(FileNotFoundError, match="not found"):
        backend._minit2i_prepare_loras([{"path": "no_such_minit2i_lora.safetensors"}])
    assert "lora_not_found" in warning_codes(warnings_seen)


def test_minit2i_unreadable_file_refuses_and_warns(tmp_path, warnings_seen, monkeypatch):
    from core.extensions import lora_manager as lm

    broken = tmp_path / "broken.safetensors"
    broken.write_bytes(b"not a safetensors file")
    monkeypatch.setattr(lm.lora_manager, "_resolve_lora_path", lambda p: str(broken))

    transformer = _Transformer()
    backend = _Backend(transformer)
    with pytest.raises(RuntimeError, match="could not be applied"):
        backend._minit2i_prepare_loras([{"path": str(broken), "strength": 1.0}])
    assert "lora_load_failed" in warning_codes(warnings_seen)
    assert not wrapped_paths(transformer)


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


def test_minit2i_unmatched_module_is_refused_before_either_pass(tmp_path,
                                                                warnings_seen):
    path, tf_paths, te_paths = train_and_save(tmp_path)
    saved = load_file(path)
    ghost = flatten_to_key("model.net.double_blocks.9.img_qkv")
    saved[f"{ghost}.lora_down.weight"] = torch.randn(RANK, D)
    saved[f"{ghost}.lora_up.weight"] = torch.randn(D, RANK)
    extended = tmp_path / "extended.safetensors"
    save_file(saved, str(extended), metadata={"model_type": "minit2i"})

    transformer, text_encoder = _Transformer(), _TextEncoder()
    with pytest.raises(AdapterIncompatible) as excinfo:
        load_both_halves(_Backend(transformer, text_encoder),
                         [{"path": str(extended), "strength": STRENGTH}])
    assert excinfo.value.code == "lora_partial"
    assert not wrapped_paths(transformer)
    assert not wrapped_paths(text_encoder)
    assert "lora_partial" in warning_codes(warnings_seen)


def test_minit2i_reloading_only_the_text_encoder_keeps_the_transformer_half(tmp_path):
    """Per-component bookkeeping over a STACK: one map pair, partitioned by the
    ``te::`` namespace, must still drop exactly one component's half."""
    path, tf_paths, te_paths = train_and_save(tmp_path)
    second, _t2, _e2 = train_and_save(tmp_path, name="second.safetensors", seed=99)

    transformer, te_a = _Transformer(), _TextEncoder()
    before = dict(transformer.named_modules())
    backend = _Backend(transformer, te_a)
    load_both_halves(backend, [{"path": path, "strength": STRENGTH},
                               {"path": second, "strength": STRENGTH_B}])
    assert all(len(dict(transformer.named_modules())[t]) == 2 for t in tf_paths)
    old_te_ids = module_ids(te_a)

    te_b = _TextEncoder(seed=9)
    te_b_before = module_ids(te_b)
    backend.minit2i_components["text_encoder"] = te_b

    assert backend._unload_lora_minit2i() == len(tf_paths)
    assert not wrapped_paths(transformer)
    for target in tf_paths:
        assert dict(transformer.named_modules())[target] is before[target], target
    assert module_ids(te_b) == te_b_before
    assert not (module_ids(te_b) & old_te_ids)
