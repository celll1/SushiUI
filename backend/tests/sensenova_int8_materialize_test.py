"""Phase U-2-1: load-time bf16 materialization of the SenseNova int8 decoder.

Covers the numeric contract of the dequantization, the shape of what the
optimizer can see afterwards, the NEGATIVE CONTROL that records the shipped
(LoRA) behaviour, and the refusals.
"""

import hashlib
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.models.common.convrot_int8_linear import ConvRotInt8Linear
from core.models.ideogram4.vendor.int8_linear import Int8Linear, quantize_weight_to_int8
from core.models.sensenova.loader import (
    SENSENOVA_BRANCH_LINEAR_COUNTS,
    materialize_int8_decoder_linears,
)
from core.models.sensenova.sensenova_lora import iter_sensenova_lora_targets
from core.training.ops.sensenova_ops import (
    load_components,
    resolve_full_finetune_branch,
    resolve_training_method,
)

_LAYERS = 42
_IN, _OUT = 8, 4


def _int8_linear_from(weight: torch.Tensor, bias: torch.Tensor = None) -> Int8Linear:
    """A real Int8Linear carrying the int8 quantization of ``weight``."""
    codes, scale = quantize_weight_to_int8(weight)
    module = Int8Linear(
        weight.shape[1], weight.shape[0], bias is not None, torch.bfloat16
    )
    module.weight.copy_(codes)
    module.weight_scale.copy_(scale)
    if bias is not None:
        module.bias.copy_(bias)
    return module


def _quant(seed: int, out=_OUT, in_=_IN) -> Int8Linear:
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(out, in_, generator=generator, dtype=torch.float32)
    return _int8_linear_from(weight.to(torch.bfloat16).float())


class _Decoder(nn.Module):
    """The 42-layer MoT attribute layout ``iter_sensenova_lora_targets`` walks."""

    use_pixel_head = True
    use_deep_fm_head = False

    def __init__(self, factory=_quant, layers=_LAYERS):
        super().__init__()
        seed = 0
        blocks = []
        for _ in range(layers):
            block = nn.Module()
            attn = nn.Module()
            mlp, mlp_gen = nn.Module(), nn.Module()
            for stem in ("q_proj", "k_proj", "v_proj", "o_proj"):
                for name in (stem, f"{stem}_mot_gen"):
                    setattr(attn, name, factory(seed))
                    seed += 1
            for stem in ("gate_proj", "up_proj", "down_proj"):
                for parent in (mlp, mlp_gen):
                    setattr(parent, stem, factory(seed))
                    seed += 1
            block.self_attn = attn
            block.mlp = mlp
            block.mlp_mot_gen = mlp_gen
            blocks.append(block)
        core = nn.Module()
        core.layers = nn.ModuleList(blocks)
        language_model = nn.Module()
        language_model.model = core
        self.language_model = language_model


def _convrot(seed: int) -> ConvRotInt8Linear:
    # in_features must be divisible by the ConvRot group size.
    return ConvRotInt8Linear(256, _OUT, False, torch.bfloat16,
                             convrot_groupsize=256, marker_numel=1)


def _state_digest(module: nn.Module) -> str:
    """SHA-256 over every tensor's name, dtype, shape and BYTES."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------

def _plant(transformer: nn.Module, module: nn.Module, *, path=("mlp", "up_proj")):
    """Install ``module`` at layer 0's understanding-branch slot ``path``."""
    parent = getattr(transformer.language_model.model.layers[0], path[0])
    setattr(parent, path[1], module)
    return parent


def test_materialized_weight_is_the_layers_own_dequantization():
    """Bitwise equal to what ``Int8Linear._dequant_forward`` builds per call."""
    generator = torch.Generator().manual_seed(7)
    weight = torch.randn(64, 128, generator=generator) * 0.05
    weight[3] *= 40.0  # a high-crest row, where the int8 grid is coarsest
    quantized = _int8_linear_from(weight)
    codes = quantized.weight.clone()
    scale = quantized.weight_scale.clone()

    transformer = _Decoder()
    parent = _plant(transformer, quantized)
    assert materialize_int8_decoder_linears(transformer, branch="und") == 294

    got = parent.up_proj.weight
    expected = codes * scale.to(torch.bfloat16).unsqueeze(1)
    assert got.dtype is torch.bfloat16
    assert torch.equal(got.detach(), expected)

    # And the predicted round-trip error, closed form rather than a tolerance:
    #   |w - q*s| <= s/2                      (round-half-to-even onto the grid)
    #   s = amax/127, so that term is amax/254
    #   rounding s to bf16 and then q*s to bf16 each cost <= 2^-8 relative,
    #   and |q*s| <= amax, giving amax*(2^-7 + 2^-16).
    amax = weight.abs().amax(dim=1)
    bound = amax * (1.0 / 254.0 + 2.0 ** -7 + 2.0 ** -16)
    error = (weight - got.detach().float()).abs().amax(dim=1)
    assert torch.all(error <= bound), (error / bound).max()
    # The bound is tight, not vacuous: the coarsest row must actually use it.
    assert (error / bound).max() > 0.25


def test_materialized_linear_reproduces_the_int8_forward_bitwise():
    generator = torch.Generator().manual_seed(11)
    weight = torch.randn(32, 64, generator=generator)
    bias = torch.randn(32, generator=generator).to(torch.bfloat16)
    quantized = _int8_linear_from(weight, bias)
    x = torch.randn(5, 64, generator=generator).to(torch.bfloat16)
    reference = quantized(x).detach().clone()

    transformer = _Decoder()
    parent = _plant(transformer, quantized)
    materialize_int8_decoder_linears(transformer, branch="und")

    assert isinstance(parent.up_proj, nn.Linear)
    assert torch.equal(parent.up_proj(x).detach(), reference)
    assert isinstance(parent.up_proj.bias, nn.Parameter)
    assert torch.equal(parent.up_proj.bias.detach(), bias)


# ---------------------------------------------------------------------------
# Scope: the counts the existing enumerator reports
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("branch,expected", [("gen", 294), ("und", 294), ("both", 588)])
def test_materialization_covers_exactly_the_branch_scope(branch, expected):
    transformer = _Decoder()
    before = sum(1 for _ in iter_sensenova_lora_targets(transformer, branch="both"))
    assert before == 588
    assert SENSENOVA_BRANCH_LINEAR_COUNTS[branch] == expected

    assert materialize_int8_decoder_linears(transformer, branch=branch) == expected

    materialized = {
        path
        for path, _, _, module in iter_sensenova_lora_targets(transformer, branch="both")
        if type(module) is nn.Linear
    }
    assert len(materialized) == expected
    still_int8 = [
        path
        for path, _, _, module in iter_sensenova_lora_targets(transformer, branch="both")
        if type(module) is Int8Linear
    ]
    assert len(still_int8) == 588 - expected
    assert not (materialized & set(still_int8))

    # An optimizer can see exactly the materialized ones: one weight Parameter
    # each (these synthetic Linears are bias-free, as the real decoder's are).
    weights = [
        module.weight
        for _, _, _, module in iter_sensenova_lora_targets(transformer, branch="both")
        if type(module) is nn.Linear
    ]
    assert all(isinstance(w, nn.Parameter) for w in weights)
    seen = {id(p) for p in transformer.parameters()}
    assert {id(w) for w in weights} <= seen
    assert len(list(transformer.parameters())) == expected


def test_each_int8_module_dies_before_the_next_is_dequantized():
    """The memory contract: peak is base + materialized + ONE weight, not 2x base."""
    transformer = _Decoder()
    live_at_death = []

    def _count(_ref):
        live_at_death.append(
            sum(
                1
                for _, _, _, module in iter_sensenova_lora_targets(transformer, branch="und")
                if type(module) is nn.Linear
            )
        )

    refs = [
        weakref.ref(module, _count)
        for _, _, _, module in iter_sensenova_lora_targets(transformer, branch="und")
    ]
    materialize_int8_decoder_linears(transformer, branch="und")

    assert all(ref() is None for ref in refs)  # every int8 buffer released
    # Each module died right after its own replacement was installed, so the
    # k-th death saw k live nn.Linears. A deferred bulk release would report 294
    # for every one of them.
    assert live_at_death == list(range(1, 295))


def test_negative_control_int8_weights_are_buffers_and_invisible_to_parameters():
    """The SHIPPED behaviour, recorded: nothing to optimize on an int8 base."""
    transformer = _Decoder()
    assert len(list(transformer.parameters())) == 0
    assert len(list(transformer.named_buffers())) == 588 * 2  # weight + weight_scale

    targets = list(iter_sensenova_lora_targets(transformer, branch="und"))
    assert len(targets) == 294
    for _, _, _, module in targets:
        assert type(module) is Int8Linear
        assert isinstance(module.weight, torch.Tensor)
        assert not isinstance(module.weight, nn.Parameter)
        assert "weight" in dict(module.named_buffers())
        assert list(module.parameters()) == []
        # requires_grad_(True) is the no-op this whole route exists to avoid.
        module.requires_grad_(True)
        assert list(module.parameters()) == []


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

def test_materialization_refuses_a_convrot_base():
    transformer = _Decoder(factory=lambda seed: _convrot(seed))
    with pytest.raises(RuntimeError, match="ConvRot-quantized base"):
        materialize_int8_decoder_linears(transformer, branch="und")
    # Nothing was replaced before the refusal.
    assert all(
        isinstance(module, ConvRotInt8Linear)
        for _, _, _, module in iter_sensenova_lora_targets(transformer, branch="both")
    )


def test_materialization_accepts_plain_int8_and_refuses_a_mixed_base():
    transformer = _Decoder()
    transformer.language_model.model.layers[0].self_attn.q_proj = _convrot(0)
    with pytest.raises(RuntimeError, match="ConvRot-quantized base"):
        materialize_int8_decoder_linears(transformer, branch="und")
    # The generation half is untouched by that understanding-side ConvRot.
    assert materialize_int8_decoder_linears(transformer, branch="gen") == 294


def test_materialization_refuses_an_off_count_tree_and_an_unknown_branch():
    with pytest.raises(RuntimeError, match="found 287 decoder Linear"):
        materialize_int8_decoder_linears(_Decoder(layers=41), branch="und")
    with pytest.raises(ValueError, match="Unknown SenseNova materialization branch"):
        materialize_int8_decoder_linears(_Decoder(layers=1), branch="text_encoder")
    with pytest.raises(ValueError, match="floating-point dtype"):
        materialize_int8_decoder_linears(_Decoder(), branch="und", dtype=torch.int8)


def test_materialization_refuses_a_mis_shaped_weight_scale():
    """loader.py's ``_reshape_convrot_scales`` warns against a blanket squeeze."""
    quantized = _quant(0)
    quantized.weight_scale = quantized.weight_scale.reshape(_OUT, 1)
    transformer = _Decoder()
    parent = _plant(transformer, quantized)
    with pytest.raises(RuntimeError, match=r"weight_scale of shape \(4, 1\)"):
        materialize_int8_decoder_linears(transformer, branch="und")
    assert parent.up_proj is quantized


def test_materialization_refuses_an_already_materialized_target():
    transformer = _Decoder()
    materialize_int8_decoder_linears(transformer, branch="und")
    with pytest.raises(RuntimeError, match="plain Int8Linear"):
        materialize_int8_decoder_linears(transformer, branch="und")


# ---------------------------------------------------------------------------
# Method / branch plumbing
# ---------------------------------------------------------------------------

class FullParameterTrainer:  # name-matched on purpose: the MRO walk keys on it
    pass


class _SubclassedFullTrainer(FullParameterTrainer):
    config = {}


def test_training_method_resolution_prefers_the_trainer_subclass():
    assert resolve_training_method(SimpleNamespace()) == "lora"
    assert resolve_training_method(SimpleNamespace(config={})) == "lora"
    assert resolve_training_method(SimpleNamespace(config={"training_method": "lora"})) == "lora"
    assert resolve_training_method(_SubclassedFullTrainer()) == "full"
    for spelling in ("full", "full_finetune", "FULL_FINETUNE", " full "):
        assert resolve_training_method(
            SimpleNamespace(config={"training_method": spelling})
        ) == "full"
    assert resolve_training_method(
        SimpleNamespace(config={"training_method": "relora"})
    ) == "relora"


@pytest.mark.parametrize(
    "train_unet,train_text_encoder,expected",
    [(True, False, "gen"), (False, True, "und"), (True, True, "both")],
)
def test_full_finetune_branch_comes_from_the_two_shipped_switches(
    train_unet, train_text_encoder, expected
):
    trainer = SimpleNamespace(
        train_unet=train_unet, train_text_encoder=train_text_encoder
    )
    assert resolve_full_finetune_branch(trainer) == expected


def test_full_finetune_branch_refuses_when_nothing_is_selected():
    trainer = SimpleNamespace(train_unet=False, train_text_encoder=False)
    with pytest.raises(ValueError, match="train_unet=False and train_text_encoder=False"):
        resolve_full_finetune_branch(trainer)


# ---------------------------------------------------------------------------
# load_components: the LoRA path must be untouched, byte for byte
# ---------------------------------------------------------------------------

def _load_with(trainer_extra: dict, transformer: nn.Module):
    trainer = SimpleNamespace(
        model_path="checkpoint.safetensors",
        weight_dtype=torch.bfloat16,
        device=torch.device("cpu"),
        attention_backend="native",
        # Read by the shared VAE-swap fold load_components now runs; a caller
        # that names its own training method supplies its own.
        **{"config": {}, **trainer_extra},
    )
    components = {"transformer": transformer, "tokenizer": object(), "config": object()}
    with patch(
        "core.models.sensenova.loader.load_sensenova_from_path",
        return_value=components,
    ), patch("core.training.ops.sensenova_ops.setup_attention_backend"), patch.object(
        transformer, "use_pixel_head", True, create=True
    ), patch.object(transformer, "use_deep_fm_head", False, create=True):
        load_components(trainer)
    return trainer


@pytest.mark.parametrize("extra", [{}, {"config": {}}, {"config": {"training_method": "lora"}}])
def test_lora_load_leaves_the_int8_base_bit_identical(extra):
    transformer = _Decoder()
    before_digest = _state_digest(transformer)
    before_ids = [id(m) for _, _, _, m in iter_sensenova_lora_targets(transformer, branch="both")]
    before_ptrs = [
        m.weight.data_ptr()
        for _, _, _, m in iter_sensenova_lora_targets(transformer, branch="both")
    ]

    with patch(
        "core.models.sensenova.loader.materialize_int8_decoder_linears"
    ) as materialize:
        _load_with(extra, transformer)
    materialize.assert_not_called()

    after = list(iter_sensenova_lora_targets(transformer, branch="both"))
    assert [id(m) for _, _, _, m in after] == before_ids
    assert [m.weight.data_ptr() for _, _, _, m in after] == before_ptrs
    assert all(type(m) is Int8Linear for _, _, _, m in after)
    assert _state_digest(transformer) == before_digest
    assert list(transformer.parameters()) == []


def test_full_finetune_load_materializes_only_the_selected_half():
    transformer = _Decoder()
    trainer = _load_with(
        {"config": {"training_method": "full_finetune"},
         "train_unet": False, "train_text_encoder": True},
        transformer,
    )
    assert trainer.transformer is transformer
    kinds = {
        path: type(module)
        for path, _, _, module in iter_sensenova_lora_targets(transformer, branch="both")
    }
    assert sum(1 for k in kinds.values() if k is nn.Linear) == 294
    assert sum(1 for k in kinds.values() if k is Int8Linear) == 294
    assert all(kinds[p] is nn.Linear for p in kinds if "mot_gen" not in p)
    assert len(list(transformer.parameters())) == 294
    # load_components freezes everything; the adapter (U-2-2) unfreezes.
    assert not any(p.requires_grad for p in transformer.parameters())


def test_full_finetune_load_refuses_a_contradictory_switch_pair_before_loading():
    trainer = SimpleNamespace(
        model_path="checkpoint.safetensors",
        weight_dtype=torch.bfloat16,
        device=torch.device("cpu"),
        attention_backend="native",
        config={"training_method": "full_finetune"},
        train_unet=False,
        train_text_encoder=False,
    )
    with patch("core.models.sensenova.loader.load_sensenova_from_path") as load:
        with pytest.raises(ValueError, match="nothing to train"):
            load_components(trainer)
    load.assert_not_called()
