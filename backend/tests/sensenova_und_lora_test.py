"""Phase U-1: understanding-branch LoRA (text-only).

Everything here is mocked -- no checkpoint is loaded and no GPU is touched. The
real-runtime claims (gradient actually reaching 289 of the 294 understanding
adapters, the prefix VRAM figures) were settled by the U-0 probe and are
restated here only as the NAMES the census must predict.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from safetensors import safe_open
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.models.sensenova.sensenova_lora import (
    EXPECTED_MODULE_COUNTS,
    LORA_BRANCHES,
    apply_lora_group,
    check_lora_application,
    iter_sensenova_lora_targets,
    load_lora_safetensors,
    normalise_lora_state_dict,
    restore_originals,
    und_gradient_unreachable_paths,
)
from core.training.adapters.base_adapter import (
    LORA_COMPONENT_TEXT_ENCODER_1,
    LORA_COMPONENT_UNET,
)
from core.adapters import CompositeAdapterLayer, LoRALinearLayer
from core.training.adapters.sensenova_adapter import SenseNovaLoRAAdapter
from core.training.ops import sensenova_ops


# ---------------------------------------------------------------------------
# Module trees
# ---------------------------------------------------------------------------


class _Attention(nn.Module):
    """Both MoT halves share one ``self_attn``; only the projections double."""

    def __init__(self):
        super().__init__()
        for name in (
            "q_proj_mot_gen", "k_proj_mot_gen", "v_proj_mot_gen", "o_proj_mot_gen",
            "q_proj", "k_proj", "v_proj", "o_proj",
        ):
            setattr(self, name, nn.Linear(2, 2, bias=False))


class _Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(self, name, nn.Linear(2, 2, bias=False))


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attention()
        self.mlp_mot_gen = _Mlp()
        self.mlp = _Mlp()


class _Transformer(nn.Module):
    def __init__(self, layers: int = 42, attention_dropout: float = 0.0):
        super().__init__()
        core = nn.Module()
        core.layers = nn.ModuleList([_Block() for _ in range(layers)])
        core.config = SimpleNamespace(
            attention_dropout=attention_dropout, num_hidden_layers=layers
        )
        language_model = nn.Module()
        language_model.model = core
        self.language_model = language_model


class _GenOnlyTransformer(nn.Module):
    """The Phase 1 tree: understanding attributes are simply absent."""

    def __init__(self, layers: int = 42):
        super().__init__()

        class _GenAttention(nn.Module):
            def __init__(self):
                super().__init__()
                for name in (
                    "q_proj_mot_gen", "k_proj_mot_gen",
                    "v_proj_mot_gen", "o_proj_mot_gen",
                ):
                    setattr(self, name, nn.Linear(2, 2, bias=False))

        class _GenBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = _GenAttention()
                self.mlp_mot_gen = _Mlp()

        core = nn.Module()
        core.layers = nn.ModuleList([_GenBlock() for _ in range(layers)])
        language_model = nn.Module()
        language_model.model = core
        self.language_model = language_model


def _grouped_for(transformer, branch):
    return {
        path: {
            "down": torch.ones((1, module.in_features), dtype=torch.float32),
            "up": torch.ones((module.out_features, 1), dtype=torch.float32),
            "alpha": torch.tensor(1.0),
        }
        for path, _parent, _attr, module in iter_sensenova_lora_targets(
            transformer, branch=branch
        )
    }


# ---------------------------------------------------------------------------
# A. The enumerator
# ---------------------------------------------------------------------------


def test_understanding_targets_are_294_with_no_suffix_anywhere():
    transformer = _Transformer()
    und = [path for path, *_ in iter_sensenova_lora_targets(transformer, branch="und")]

    assert len(und) == 294
    assert len(set(und)) == 294
    assert not any("mot_gen" in path for path in und)
    assert sum(".self_attn." in path for path in und) == 168
    assert sum(".mlp." in path for path in und) == 126
    # The asymmetry is the GENERATION side's alone: its attention suffix sits on
    # the Linear and its MLP suffix on the parent, while und is plain both ways.
    for index in range(42):
        prefix = f"language_model.model.layers.{index}"
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            assert f"{prefix}.self_attn.{name}" in und
        for name in ("gate_proj", "up_proj", "down_proj"):
            assert f"{prefix}.mlp.{name}" in und


def test_branch_both_is_the_disjoint_union_and_gen_is_unchanged():
    transformer = _Transformer()
    gen = [p for p, *_ in iter_sensenova_lora_targets(transformer)]
    und = [p for p, *_ in iter_sensenova_lora_targets(transformer, branch="und")]
    both = [p for p, *_ in iter_sensenova_lora_targets(transformer, branch="both")]

    assert gen == [p for p, *_ in iter_sensenova_lora_targets(transformer, branch="gen")]
    assert both == gen + und
    assert len(both) == 588 == EXPECTED_MODULE_COUNTS["generation+understanding"]
    assert not set(gen) & set(und)


def test_unknown_branch_is_refused_rather_than_silently_empty():
    assert set(LORA_BRANCHES) == {"gen", "und", "both"}
    with pytest.raises(ValueError, match="Unknown SenseNova LoRA branch"):
        list(iter_sensenova_lora_targets(_Transformer(), branch="text_encoder"))


def test_gradient_unreachable_targets_are_named_not_counted():
    dead = und_gradient_unreachable_paths(42)
    und = {p for p, *_ in iter_sensenova_lora_targets(_Transformer(), branch="und")}

    assert dead <= und
    assert len(dead) == 5
    assert dead == {
        "language_model.model.layers.41.self_attn.q_proj",
        "language_model.model.layers.41.self_attn.o_proj",
        "language_model.model.layers.41.mlp.gate_proj",
        "language_model.model.layers.41.mlp.up_proj",
        "language_model.model.layers.41.mlp.down_proj",
    }
    # k/v DO train: generation layer 41 consumes their K/V.
    assert "language_model.model.layers.41.self_attn.k_proj" not in dead
    assert "language_model.model.layers.41.self_attn.v_proj" not in dead
    assert len(und) - len(dead) == 289


# ---------------------------------------------------------------------------
# B. Inference application
# ---------------------------------------------------------------------------


def test_inference_now_applies_understanding_keys_instead_of_dropping_them():
    transformer = _Transformer()
    grouped = _grouped_for(transformer, "both")
    assert len(grouped) == 588

    # The pre-U-1 behaviour, kept as the regression's other arm: enumerating the
    # generation branch alone silently returns a smaller count and raises nothing.
    gen_only_originals, gen_only_keys = {}, set()
    assert apply_lora_group(
        transformer, grouped, 1.0, gen_only_originals, gen_only_keys, branch="gen"
    ) == 294
    assert check_lora_application(grouped, 294, {"lora_targets": "generation+understanding"})
    assert restore_originals(transformer, gen_only_originals, gen_only_keys) == 294

    originals, wrapped = {}, set()
    assert apply_lora_group(transformer, grouped, 1.0, originals, wrapped) == 588
    assert check_lora_application(grouped, 588, {"lora_targets": "generation+understanding"}) is None
    live = dict(transformer.named_modules())
    und_path = "language_model.model.layers.7.self_attn.q_proj"
    assert isinstance(live[und_path], CompositeAdapterLayer)
    assert und_path in wrapped
    assert restore_originals(transformer, originals, wrapped) == 588
    assert not isinstance(dict(transformer.named_modules())[und_path], CompositeAdapterLayer)


def test_understanding_wrapper_changes_the_forward_and_strength_zero_does_not():
    transformer = _Transformer()
    grouped = _grouped_for(transformer, "und")
    path = "language_model.model.layers.0.self_attn.q_proj"
    parent = transformer.language_model.model.layers[0].self_attn
    original = parent.q_proj
    sample = torch.tensor([[1.0, -2.0]])
    expected = original(sample)

    originals, wrapped = {}, set()
    assert apply_lora_group(transformer, grouped, 0.0, originals, wrapped) == 294
    assert torch.equal(parent.q_proj(sample), expected)
    assert restore_originals(transformer, originals, wrapped) == 294

    originals, wrapped = {}, set()
    assert apply_lora_group(transformer, grouped, 1.0, originals, wrapped) == 294
    assert not torch.equal(parent.q_proj(sample), expected)
    assert restore_originals(transformer, originals, wrapped) == 294
    assert parent.q_proj is original


def test_existing_generation_only_lora_still_applies_294_of_294():
    """The distillation checkpoint's compatibility is preserved by lookup.

    Application enumerates both branches now, so the understanding slots are
    visited -- and miss, because a generation-only file carries no key for them.
    """
    transformer = _Transformer()
    grouped = _grouped_for(transformer, "gen")
    assert len(grouped) == 294

    originals, wrapped = {}, set()
    assert apply_lora_group(transformer, grouped, 1.0, originals, wrapped) == 294
    assert check_lora_application(grouped, 294, {"lora_targets": "generation"}) is None
    assert all("mot_gen" in path for path in wrapped)
    und_path = "language_model.model.layers.0.self_attn.q_proj"
    assert not isinstance(dict(transformer.named_modules())[und_path], CompositeAdapterLayer)
    assert restore_originals(transformer, originals, wrapped) == 294

    # And on the Phase 1 tree shape, where the und attributes do not exist.
    gen_only_tree = _GenOnlyTransformer()
    originals, wrapped = {}, set()
    assert apply_lora_group(gen_only_tree, grouped, 1.0, originals, wrapped) == 294
    assert restore_originals(gen_only_tree, originals, wrapped) == 294


def test_format_sniff_recognises_understanding_keys_and_declared_scope(tmp_path):
    from safetensors.torch import save_file

    und_only = {
        "language_model.model.layers.0.self_attn.q_proj.lora_down.weight": torch.zeros(1, 2),
        "language_model.model.layers.0.self_attn.q_proj.lora_up.weight": torch.zeros(2, 1),
        "language_model.model.layers.0.mlp.gate_proj.lora_down.weight": torch.zeros(1, 2),
        "language_model.model.layers.0.mlp.gate_proj.lora_up.weight": torch.zeros(2, 1),
    }
    no_metadata = tmp_path / "und_no_metadata.safetensors"
    save_file(und_only, str(no_metadata))
    _raw, fmt, metadata = load_lora_safetensors(str(no_metadata))
    assert fmt == "neo_hf_lora"
    assert metadata == {}

    declared = tmp_path / "scope_only.safetensors"
    save_file(
        {"something.else.weight": torch.zeros(1)},
        str(declared),
        metadata={"lora_targets": "generation+understanding"},
    )
    _raw, fmt, metadata = load_lora_safetensors(str(declared))
    assert fmt == "neo_hf_lora"
    assert metadata["lora_targets"] == "generation+understanding"

    foreign = tmp_path / "foreign.safetensors"
    save_file({"lora_unet_down_blocks.alpha": torch.zeros(1)}, str(foreign))
    assert load_lora_safetensors(str(foreign))[1] == "unknown"


def test_partial_application_is_reported_rather_than_silent():
    grouped = {f"m{i}": {} for i in range(588)}
    assert check_lora_application(grouped, 588, {"lora_targets": "generation+understanding"}) is None
    message = check_lora_application(grouped, 294, {"lora_targets": "generation+understanding"})
    assert message is not None and "294 of 588" in message
    scope = check_lora_application({f"m{i}": {} for i in range(294)}, 294,
                                   {"lora_targets": "generation+understanding"})
    assert scope is not None and "lora_targets" in scope
    assert check_lora_application(grouped, 588, None) is None


# ---------------------------------------------------------------------------
# C. Training adapter
# ---------------------------------------------------------------------------


def _adapter(transformer, *, train_text_encoder=True, rank=2, alpha=4, **lrs):
    trainer = SimpleNamespace(
        transformer=transformer,
        unet_lr=2e-4,
        train_text_encoder=train_text_encoder,
        **lrs,
    )
    return SenseNovaLoRAAdapter(trainer, rank, alpha), trainer


def test_understanding_lora_is_injected_only_when_train_text_encoder_is_set():
    transformer = _Transformer()
    adapter, _ = _adapter(transformer, train_text_encoder=False)
    layers = {}
    assert adapter.apply_lora_to_unet(layers) == 294
    assert adapter.apply_lora_to_text_encoders(layers) == 0
    assert len(layers) == 294

    transformer = _Transformer()
    adapter, _ = _adapter(transformer)
    layers = {}
    assert adapter.apply_lora_to_unet(layers) == 294
    assert adapter.apply_lora_to_text_encoders(layers) == 294
    assert len(layers) == 588
    assert all(isinstance(layer, LoRALinearLayer) for layer in layers.values())
    # Re-application over the already-wrapped tree repopulates without rewrapping.
    repopulated = {}
    fresh, _ = _adapter(transformer)
    assert fresh.apply_lora_to_unet(repopulated) == 0
    assert fresh.apply_lora_to_text_encoders(repopulated) == 0
    assert len(repopulated) == 588


def test_understanding_lora_registers_as_text_encoder_1_for_grad_norm_split():
    transformer = _Transformer()
    adapter, _ = _adapter(transformer)
    layers = {}
    adapter.apply_lora_to_unet(layers)
    adapter.apply_lora_to_text_encoders(layers)

    components = adapter.lora_components
    assert sum(c == LORA_COMPONENT_UNET for c in components.values()) == 294
    assert sum(c == LORA_COMPONENT_TEXT_ENCODER_1 for c in components.values()) == 294
    und = [n for n, c in components.items() if c == LORA_COMPONENT_TEXT_ENCODER_1]
    assert not any("mot_gen" in name for name in und)


@pytest.mark.parametrize(
    "lrs,expected",
    [
        ({"text_encoder_1_lr": 5e-5, "text_encoder_lr": 7e-5}, 5e-5),
        ({"text_encoder_1_lr": None, "text_encoder_lr": 7e-5}, 7e-5),
        ({}, 2e-4),
    ],
)
def test_understanding_group_uses_the_text_encoder_1_lr_chain(lrs, expected):
    transformer = _Transformer()
    adapter, _ = _adapter(transformer, **lrs)
    layers = {}
    adapter.apply_lora_to_unet(layers)
    adapter.apply_lora_to_text_encoders(layers)

    groups = adapter.setup_trainable_parameters(layers)
    assert [len(g["params"]) for g in groups] == [588, 588]
    assert groups[0]["lr"] == 2e-4
    assert groups[1]["lr"] == expected


def test_checkpoint_declares_the_combined_scope_and_round_trips(tmp_path):
    transformer = _Transformer()
    adapter, _ = _adapter(transformer)
    layers = {}
    adapter.apply_lora_to_unet(layers)
    adapter.apply_lora_to_text_encoders(layers)
    with torch.no_grad():
        for index, layer in enumerate(layers.values()):
            layer.lora_up.weight.fill_((index + 1) / 1000)

    checkpoint = tmp_path / "sensenova_und_lora.safetensors"
    adapter.save_checkpoint(layers, step=5, epoch=1, output_path=checkpoint)

    raw, fmt, metadata = load_lora_safetensors(str(checkpoint))
    grouped = normalise_lora_state_dict(raw)
    assert fmt == "neo_hf_lora"
    assert len(raw) == 1764
    assert len(grouped) == 588
    assert metadata["lora_targets"] == "generation+understanding"
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        assert handle.metadata()["tensor_kind"] == "neo_hf_lora"

    inference_tree = _Transformer()
    originals, wrapped = {}, set()
    applied = apply_lora_group(inference_tree, grouped, 1.0, originals, wrapped)
    assert applied == 588
    assert check_lora_application(grouped, applied, metadata) is None
    assert restore_originals(inference_tree, originals, wrapped) == 588


def test_generation_only_run_still_saves_the_generation_scope(tmp_path):
    transformer = _Transformer()
    adapter, _ = _adapter(transformer, train_text_encoder=False)
    layers = {}
    adapter.apply_lora_to_unet(layers)
    adapter.apply_lora_to_text_encoders(layers)

    checkpoint = tmp_path / "gen_only.safetensors"
    adapter.save_checkpoint(layers, step=1, epoch=0, output_path=checkpoint)
    raw, _fmt, metadata = load_lora_safetensors(str(checkpoint))
    assert metadata["lora_targets"] == "generation"
    assert len(raw) == 882


def test_understanding_only_checkpoints_are_refused(tmp_path):
    transformer = _Transformer()
    adapter, _ = _adapter(transformer)
    layers = {}
    adapter.apply_lora_to_text_encoders(layers)
    with pytest.raises(RuntimeError, match="must carry the generation branch"):
        adapter.save_checkpoint(layers, step=1, epoch=0, output_path=tmp_path / "u.safetensors")


def test_dropout_guard_refuses_a_stochastic_prefix_recompute():
    ok = _Transformer(attention_dropout=0.0)
    sensenova_ops.assert_understanding_training_supported(ok)

    bad = _Transformer(attention_dropout=0.1)
    adapter, _ = _adapter(bad)
    with pytest.raises(RuntimeError, match="attention_dropout=0.0"):
        sensenova_ops.assert_understanding_training_supported(bad)
    with pytest.raises(RuntimeError, match="attention_dropout=0.0"):
        adapter.apply_lora_to_text_encoders({})


# ---------------------------------------------------------------------------
# D. The prefix cache assertions
# ---------------------------------------------------------------------------


class _CacheLayer:
    def __init__(self, requires_grad=False, grad_fn=False):
        base = torch.ones(1, 1, 2, 1, requires_grad=requires_grad or grad_fn)
        self.keys = base * 1 if grad_fn else base
        self.values = base * 1 if grad_fn else base
        self.flash_k_cache = None
        self.flash_v_cache = None


class _Cache:
    def __init__(self, layers=1, **kwargs):
        self.layers = [_CacheLayer(**kwargs) for _ in range(layers)]
        self._kv_cache_streamer = None
        self._kv_cache_streamer_branch = None


@pytest.mark.parametrize("trainable", [False, True])
def test_structural_validation_is_unconditional(trainable):
    with pytest.raises(ValueError, match="requires a prefix KV cache"):
        sensenova_ops._assert_immutable_prefix_cache(None, 1, trainable=trainable)

    short = _Cache(layers=1)
    with pytest.raises(ValueError, match="has 1 layer"):
        sensenova_ops._assert_immutable_prefix_cache(short, 2, trainable=trainable)

    streamed = _Cache(grad_fn=trainable)
    streamed._kv_cache_streamer = object()
    with pytest.raises(ValueError, match="KV cache streamer"):
        sensenova_ops._assert_immutable_prefix_cache(streamed, 1, trainable=trainable)

    flashed = _Cache(grad_fn=trainable)
    flashed.layers[0].flash_k_cache = torch.zeros(1)
    with pytest.raises(ValueError, match="flash KV buffers"):
        sensenova_ops._assert_immutable_prefix_cache(flashed, 1, trainable=trainable)

    empty = _Cache(grad_fn=trainable)
    empty.layers[0].keys = torch.zeros(0)
    with pytest.raises(ValueError, match="missing non-empty keys"):
        sensenova_ops._assert_immutable_prefix_cache(empty, 1, trainable=trainable)


def test_grad_mode_validation_is_the_only_half_that_branches():
    detached = _Cache()
    differentiable = _Cache(grad_fn=True)

    sensenova_ops._assert_immutable_prefix_cache(detached, 1)
    sensenova_ops._assert_immutable_prefix_cache(differentiable, 1, trainable=True)

    with pytest.raises(ValueError, match="must be detached"):
        sensenova_ops._assert_immutable_prefix_cache(differentiable, 1)
    # The positive assertion: without it, a no_grad prefix under und training
    # produces a healthy falling loss and never trains the understanding LoRA.
    with pytest.raises(ValueError, match="carry no grad_fn"):
        sensenova_ops._assert_immutable_prefix_cache(detached, 1, trainable=True)


def test_positive_assertion_catches_a_partially_differentiable_prefix():
    cache = _Cache(layers=4, grad_fn=True)
    cache.layers[2].values = cache.layers[2].values.detach()
    with pytest.raises(ValueError, match="1 of 4 KV cache layer"):
        sensenova_ops._assert_immutable_prefix_cache(cache, 4, trainable=True)


# ---------------------------------------------------------------------------
# E. The differentiable prefix pass
# ---------------------------------------------------------------------------


class _UndLayer(nn.Module):
    attention_type = "full_attention"

    def __init__(self):
        super().__init__()
        self.k = nn.Linear(2, 2, bias=False)
        self.v = nn.Linear(2, 2, bias=False)

    def forward(self, hidden_states, **kwargs):
        assert kwargs["return_kv"] is True
        assert kwargs["exist_non_image_gen_tokens"] is True
        assert kwargs["exist_image_gen_tokens"] is False
        assert kwargs["past_key_values"] is None
        assert kwargs["use_cache"] is False
        keys = self.k(hidden_states).unsqueeze(1)
        values = self.v(hidden_states).unsqueeze(1)
        return hidden_states + keys.squeeze(1), keys, values


class _UndModel(nn.Module):
    def __init__(self, layers=3):
        super().__init__()
        self.layers = nn.ModuleList([_UndLayer() for _ in range(layers)])
        self.embed_tokens = nn.Embedding(8, 2)
        self.config = SimpleNamespace(num_hidden_layers=layers, attention_dropout=0.0)


def _und_inputs(length=3):
    t_idx = torch.arange(length, dtype=torch.long)
    zeros = torch.zeros_like(t_idx)
    return (
        torch.ones(1, length, dtype=torch.long),
        torch.stack([t_idx, zeros, zeros], dim=0),
        {"full_attention": None},
    )


@pytest.mark.parametrize("checkpoint_layers", [False, True])
def test_prefix_loop_returns_kv_as_outputs_that_carry_gradient(checkpoint_layers):
    model = _UndModel()
    input_ids, indexes, mask = _und_inputs()

    cache = sensenova_ops.forward_und_prefix_layers(
        model, input_ids, indexes, mask, checkpoint_layers=checkpoint_layers
    )

    assert len(cache.layers) == 3
    assert cache._kv_cache_streamer is None
    assert cache.get_seq_length() == cache.layers[0].keys.shape[-2]
    sensenova_ops._assert_immutable_prefix_cache(cache, 3, trainable=True)
    cache.layers[-1].keys.sum().backward()
    assert model.layers[0].k.weight.grad is not None


def test_prefix_loop_refuses_a_mask_it_has_no_entry_for():
    model = _UndModel()
    input_ids, indexes, _mask = _und_inputs()
    with pytest.raises(ValueError, match="no mask for attention type"):
        sensenova_ops.forward_und_prefix_layers(
            model, input_ids, indexes, {"sliding_attention": None}
        )


def _trainable_trainer(transformer, **overrides):
    trainer = SimpleNamespace(
        transformer=transformer,
        tokenizer=object(),
        device=torch.device("cpu"),
        training_dtype=torch.float32,
        gradient_checkpointing=True,
        train_text_encoder=True,
    )
    for key, value in overrides.items():
        setattr(trainer, key, value)
    return trainer


class _PrefixTransformer(nn.Module):
    def __init__(self, layers=3):
        super().__init__()
        self.language_model = SimpleNamespace(model=_UndModel(layers))
        self.prefix_forward_calls = 0

    def _build_t2i_query(self, prompt, **kwargs):
        return prompt

    def _build_t2i_text_inputs(self, tokenizer, query):
        return _und_inputs()

    def _t2i_prefix_forward(self, *args):
        self.prefix_forward_calls += 1
        return _Cache(layers=len(self.language_model.model.layers)), torch.zeros(1, 3, 1)


def test_trainable_encode_prompt_builds_a_differentiable_prefix():
    transformer = _PrefixTransformer()
    trainer = _trainable_trainer(transformer)

    prefix = sensenova_ops.encode_prompt(trainer, "a caption", requires_grad=True)

    assert transformer.prefix_forward_calls == 0  # never the no-grad vendor path
    assert prefix.text_length == 3
    assert len(prefix.cache.layers) == 3
    assert all(layer.keys.grad_fn is not None for layer in prefix.cache.layers)


def test_trainable_prefix_runs_under_autocast_for_the_fp32_adapters():
    """LoRALinearLayer keeps fp32 adapters and needs an ambient autocast.

    train_step wraps the generation pass; without the same wrap here the first
    und-LoRA prefix pass raises a dtype mismatch at layer 0 (U-0, measured).
    """
    transformer = _PrefixTransformer()
    trainer = _trainable_trainer(
        transformer, device=torch.device("cuda"), training_dtype=torch.bfloat16
    )
    seen = {}

    real_autocast = torch.autocast

    class _Recording(real_autocast):
        def __init__(self, device_type, dtype=None, enabled=True, **kwargs):
            seen.update(device_type=device_type, dtype=dtype, enabled=enabled)
            super().__init__("cpu", dtype=torch.bfloat16, enabled=False)

    with patch("torch.autocast", _Recording):
        sensenova_ops.encode_prompt(trainer, "a caption", requires_grad=True)

    assert seen == {"device_type": "cuda", "dtype": torch.bfloat16, "enabled": True}


def test_trainable_prefix_refuses_phase_eviction():
    """Reference items are no longer refused here (U-3); eviction still is.

    ``sensenova_und_reference_test.py`` owns the reference half, including the
    negative control that reproduces the refusal this used to assert.
    """
    transformer = _PrefixTransformer()

    trainer = _trainable_trainer(transformer, sensenova_phase_evictor=object())
    with pytest.raises(RuntimeError, match="MoT phase eviction"):
        sensenova_ops.encode_prompt(trainer, "a caption", requires_grad=True)


# ---------------------------------------------------------------------------
# F. The frozen path must not change
# ---------------------------------------------------------------------------


def test_frozen_encode_prompt_still_uses_the_vendor_no_grad_prefix():
    transformer = _PrefixTransformer()
    trainer = _trainable_trainer(transformer, train_text_encoder=False)

    with patch.object(
        sensenova_ops, "forward_und_prefix_layers",
        side_effect=AssertionError("frozen path must not build a trainable prefix"),
    ):
        prefix = sensenova_ops.encode_prompt(trainer, "a caption")

    assert transformer.prefix_forward_calls == 1
    assert prefix.text_length == 3
    assert all(not layer.keys.requires_grad for layer in prefix.cache.layers)
    # The Phase 1 refusal is intact for the frozen path.
    with pytest.raises(ValueError, match="must be detached"):
        sensenova_ops._assert_immutable_prefix_cache(
            _Cache(layers=3, grad_fn=True), 3
        )


class _StepLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, hidden_states, **kwargs):
        return hidden_states * self.scale


class _StepHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, hidden):
        return hidden[:, :1].expand(-1, 3, 32, 32) * self.scale


class _StepTransformer(nn.Module):
    patch_size = 16
    downsample_ratio = 0.5
    config = SimpleNamespace(t_eps=0.05)
    use_pixel_head = True
    use_deep_fm_head = False

    def __init__(self):
        super().__init__()
        model = nn.Module()
        model.layers = nn.ModuleList([_StepLayer()])
        model.norm_mot_gen = nn.Identity()
        self.language_model = SimpleNamespace(model=model)
        self.fm_modules = nn.ModuleDict({"fm_head": _StepHead()})

    def _build_t2i_image_indexes(self, token_h, token_w, text_length, device):
        return torch.zeros(3, token_h * token_w, dtype=torch.long, device=device)

    def patchify(self, image, patch, channel_first=False):
        return image.permute(0, 2, 3, 1).reshape(1, 1, -1)

    def unpatchify(self, x, patch, h=None, w=None):
        return x.reshape(1, h // patch, w // patch, patch, patch, 3).permute(
            0, 5, 1, 3, 2, 4
        ).reshape(1, 3, h, w)


def _run_frozen_step(**trainer_fields):
    torch.manual_seed(1234)
    transformer = _StepTransformer()
    trainer = SimpleNamespace(
        transformer=transformer,
        device=torch.device("cpu"),
        training_dtype=torch.float32,
        gradient_checkpointing=False,
        **trainer_fields,
    )

    def build_context(model, shape, image, timestep, noise_scale, *, enable_grad=False):
        return model.patchify(image, 32), torch.full((1, 1, 1), 2.0), torch.ones(1, 1, 1)

    with patch(
        "core.models.sensenova.sensenova_pipeline_ops.compute_noise_scale",
        return_value=2.0,
    ), patch(
        "core.models.sensenova.sensenova_pipeline_ops._build_step_context",
        side_effect=build_context,
    ), patch("torch.randn_like", return_value=torch.full((1, 3, 32, 32), 0.2)):
        loss, value, recon = sensenova_ops.train_step(
            trainer,
            images=torch.ones(1, 3, 32, 32),
            prefix=sensenova_ops.SenseNovaTrainingPrefix(_Cache(), text_length=3),
            timesteps=torch.tensor([0.25]),
        )
    loss.backward()
    grads = [p.grad.clone() for p in transformer.parameters()]
    return loss.detach(), value, recon, grads


def test_train_step_is_byte_identical_when_train_text_encoder_is_false():
    """The frozen path must be untouched, not merely equivalent."""
    absent = _run_frozen_step()
    explicit = _run_frozen_step(train_text_encoder=False)

    assert torch.equal(absent[0], explicit[0])
    assert absent[1] == explicit[1] and absent[2] == explicit[2]
    assert len(absent[3]) == len(explicit[3]) and absent[3]
    for left, right in zip(absent[3], explicit[3]):
        assert torch.equal(left, right)


def test_train_step_refuses_a_detached_prefix_only_when_und_is_trained():
    torch.manual_seed(0)
    transformer = _StepTransformer()
    trainer = SimpleNamespace(
        transformer=transformer,
        device=torch.device("cpu"),
        training_dtype=torch.float32,
        gradient_checkpointing=False,
        train_text_encoder=True,
    )

    def build_context(model, shape, image, timestep, noise_scale, *, enable_grad=False):
        return model.patchify(image, 32), torch.full((1, 1, 1), 2.0), torch.ones(1, 1, 1)

    with patch(
        "core.models.sensenova.sensenova_pipeline_ops.compute_noise_scale",
        return_value=2.0,
    ), patch(
        "core.models.sensenova.sensenova_pipeline_ops._build_step_context",
        side_effect=build_context,
    ):
        with pytest.raises(ValueError, match="carry no grad_fn"):
            sensenova_ops.train_step(
                trainer,
                images=torch.ones(1, 3, 32, 32),
                prefix=sensenova_ops.SenseNovaTrainingPrefix(_Cache(), text_length=3),
                timesteps=torch.tensor([0.25]),
            )


# ---------------------------------------------------------------------------
# G. Trainer wiring, contract and capability
# ---------------------------------------------------------------------------


def test_encode_caption_forwards_requires_grad_for_sensenova():
    from core.training.base_trainer import BaseTrainer

    seen = []
    owner = SimpleNamespace(
        is_zimage=False,
        is_sensenova=True,
        arch=SimpleNamespace(
            encode_prompt=lambda trainer, caption, **kwargs: seen.append(kwargs) or "prefix"
        ),
    )
    encode = MethodType(BaseTrainer.encode_caption, owner)

    assert encode("a caption") == ("prefix", None)
    assert encode("a caption", requires_grad=True) == ("prefix", None)
    assert [call["requires_grad"] for call in seen] == [False, True]


def test_mnt_recomputes_the_prefix_only_when_the_understanding_branch_trains():
    from core.training.base_trainer import BaseTrainer

    first = object()
    rebuilt = object()

    frozen = SimpleNamespace(train_text_encoder=False)
    conditioning = MethodType(BaseTrainer._sensenova_mnt_conditioning, frozen)
    assert conditioning(first, captions=["c"], mnt_index=0)[3] is first
    assert conditioning(first, captions=["c"], mnt_index=1)[3] is first

    calls = []

    def encode_caption(caption, requires_grad=False, **kwargs):
        calls.append((caption, requires_grad))
        return rebuilt, None

    trainable = SimpleNamespace(train_text_encoder=True, encode_caption=encode_caption)
    conditioning = MethodType(BaseTrainer._sensenova_mnt_conditioning, trainable)
    # Iteration 0 reuses the prefix the batch loop already built.
    assert conditioning(first, captions=["c"], mnt_index=0)[3] is first
    assert calls == []
    # Later iterations rebuild it: the MNT loop steps the optimizer every
    # iteration, so retain_graph would use stale parameters.
    assert conditioning(first, captions=["c"], mnt_index=1)[3] is rebuilt
    assert calls == [("c", True)]


def test_contract_normalises_train_text_encoder_and_refuses_it_with_eviction():
    from core.model_loader import ModelLoader
    from core.training.train_runner import _apply_sensenova_training_contract

    train = {"batch_size": 1, "blocks_to_swap": 0, "train_text_encoder": "true"}
    with patch.object(ModelLoader, "detect_model_type", return_value="sensenova"):
        assert _apply_sensenova_training_contract("model", "lora", train, {})
    assert train["train_text_encoder"] is True

    train = {
        "batch_size": 1,
        "blocks_to_swap": 0,
        "train_text_encoder": True,
        "sensenova_mot_phase_eviction": True,
    }
    with patch.object(ModelLoader, "detect_model_type", return_value="sensenova"):
        with pytest.raises(ValueError) as excinfo:
            _apply_sensenova_training_contract("model", "lora", train, {})
    message = str(excinfo.value)
    assert "sensenova_mot_phase_eviction" in message
    # The refusal must not read as a fundamental impossibility. It used to say
    # so in words ("scope limit of this implementation"); since U-2-4 shipped the
    # phase split it names the setting that lifts it, which is a stronger form of
    # the same requirement.
    assert "sensenova_four_phase_eviction" in message
    assert "full-finetune only" in message

    train = {"batch_size": 1, "blocks_to_swap": 0, "train_text_encoder": "maybe"}
    with patch.object(ModelLoader, "detect_model_type", return_value="sensenova"):
        with pytest.raises(ValueError, match="train_text_encoder must be a boolean"):
            _apply_sensenova_training_contract("model", "lora", train, {})


@pytest.mark.parametrize(
    "train_text_encoder,expected",
    [(False, (["MoT-Generation"], [2e-4])), (True, (["MoT-Generation", "MoT-Understanding"], [2e-4, 5e-5]))],
)
def test_component_lr_list_matches_the_sensenova_group_order(train_text_encoder, expected):
    """Both SenseNova groups live inside `transformer`, so neither the U-Net nor
    the TE1 branch fires; without this a resume resets them to learning_rate."""
    from core.training.base_trainer import BaseTrainer

    owner = SimpleNamespace(
        is_sensenova=True,
        train_unet=True,
        train_text_encoder=train_text_encoder,
        unet=None,
        text_encoder=None,
        controlnet=None,
        unet_lr=2e-4,
        text_encoder_1_lr=5e-5,
        text_encoder_lr=7e-5,
        learning_rate=1e-4,
        _train_vision_encoder=False,
    )
    lrs, names = MethodType(BaseTrainer._build_component_lr_list, owner)()
    assert (names, lrs) == (expected[0], expected[1])


def test_phase_eviction_moves_understanding_lora_with_the_understanding_half():
    """Inference-side MoT eviction stays coherent with understanding LoRA.

    The evictor is built AFTER application, and it classifies by module path, so
    an understanding wrapper's ``lora_down``/``lora_up`` carry no ``_mot_gen``
    and travel with the half that actually calls them: resident for the prefix,
    staged to CPU for the denoise, where the understanding branch is never
    reached. Getting this wrong is a device mismatch at generation time.

    What a CPU test cannot cover: the real H2D/D2H transfers and pinning.
    """
    from core.models.sensenova.mot_phase_eviction import MotPhaseEvictor
    from core.models.sensenova.mot_weight_selector import select_mot_weight_modules

    transformer = _Transformer(layers=2)
    grouped = _grouped_for(transformer, "both")
    originals, wrapped = {}, set()
    assert apply_lora_group(transformer, grouped, 1.0, originals, wrapped) == 28

    selection = select_mot_weight_modules(transformer)
    gen_ids = {id(m) for m in selection.gen_modules}
    und_ids = {id(m) for m in selection.und_modules}
    assert not gen_ids & und_ids

    for path, module in transformer.named_modules():
        if not isinstance(module, LoRALinearLayer):
            continue
        expected = gen_ids if "mot_gen" in path else und_ids
        for adapter in (module.lora_down, module.lora_up, module.original_module):
            assert id(adapter) in expected, path

    evictor = MotPhaseEvictor(transformer, torch.device("cpu"))
    sample = torch.tensor([[1.0, -2.0]])
    und_linear = transformer.language_model.model.layers[0].self_attn.q_proj
    before = und_linear(sample)

    evictor.move_non_gen_to_device()
    evictor.on_phase("prefix")
    evictor.on_phase("denoise")
    evictor.teardown()

    assert torch.equal(und_linear(sample), before)
    assert restore_originals(transformer, originals, wrapped) == 28


def test_capability_scopes_text_encoder_training_to_full_finetune():
    """The full-fine-tune claim moved axis. It used to be a REFUSAL scoped to
    `full_finetune` while the API accepted `train_text_encoder` and ran it; it is
    now an ADVISORY with the same scope, carrying the measured memory cost. LoRA
    is unaffected in both directions -- it trains this branch and says nothing
    about it. See sensenova_capability_advisory_test.py."""
    from api.arch_capabilities import (
        TRAINING_FEATURE_ADVISORY,
        training_feature_advisories,
        training_feature_unsupported_reason,
    )

    entry = TRAINING_FEATURE_ADVISORY["sensenova"]["text_encoder_training"]
    assert entry["methods"] == ["full_finetune"]
    assert entry["level"] == "high_memory"
    assert "text_encoder_training" not in training_feature_advisories("sensenova", "lora")
    assert "text_encoder_training" in training_feature_advisories(
        "sensenova", "full_finetune")
    # Nothing declares the mechanism absent any more, under any method.
    for method in ("lora", "full_finetune", None):
        assert training_feature_unsupported_reason(
            "sensenova", "text_encoder_training", method) is None
