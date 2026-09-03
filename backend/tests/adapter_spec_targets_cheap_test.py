"""``AdapterSpec``, ``AdapterTarget`` and the per-architecture adapter
capability matrix.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/adapter_spec_targets_cheap_test.py -v

The capability class imports ``core.training.arch``, which costs about nine
seconds and a CUDA context (the back-edge ``adapter_layering_test.py``
measures); the spec/target classes above it are pure CPU torch.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn as nn

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_BACKEND = os.path.join(_REPO, "backend")
for _p in (_REPO, _BACKEND):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.adapters import (  # noqa: E402
    AdapterIncompatible,
    AdapterRefusal,
    AdapterSpec,
    AdapterTarget,
    CompositeAdapterLayer,
    detect_adapter_codec,
    enumerate_adapter_targets,
    get_module_slot,
    is_lora_wrappable_linear,
    quantization_kind,
    set_module_slot,
)
from core.adapters.spec import (  # noqa: E402
    ADAPTER_SCHEMA_VERSION,
    FORMAT_SUSHIUI,
    KNOWN_ARCHITECTURES,
    METADATA_ALGORITHM,
    METADATA_WEIGHT_DECOMPOSE,
    OPTION_FACTOR,
    OPTION_USE_TUCKER,
)

# ---------------------------------------------------------------------------
# AdapterSpec
# ---------------------------------------------------------------------------


def _down_up_tensors():
    return {
        "lora_transformer_blocks_0_attn_to_q.lora_down.weight": torch.randn(4, 8),
        "lora_transformer_blocks_0_attn_to_q.lora_up.weight": torch.randn(8, 4),
        "lora_transformer_blocks_0_attn_to_q.alpha": torch.tensor(2.0),
    }


def test_down_up_only_checkpoint_normalizes_to_lora():
    spec = AdapterSpec.from_codec(detect_adapter_codec(_down_up_tensors()),
                                  architecture="zimage")
    assert spec.algorithm == "lora"
    assert spec.weight_decompose is False
    assert spec.family == "lora"
    assert spec.format == FORMAT_SUSHIUI
    assert spec.rank == 4
    assert spec.alpha == 2.0
    assert spec.scale == 0.5
    spec.validate()


def test_legacy_metadata_block_normalizes_to_lora():
    # Verbatim shape of what every SushiUI adapter writes today
    # (e.g. ZImageLoRAAdapter.checkpoint_metadata): no sushi.adapter.* keys.
    spec = AdapterSpec.from_metadata({
        "lora_rank": "16",
        "lora_alpha": "8",
        "step": "1200",
        "epoch": "3",
        "model_type": "zimage",
    })
    assert (spec.algorithm, spec.weight_decompose) == ("lora", False)
    assert spec.format == FORMAT_SUSHIUI
    assert spec.schema_version == ADAPTER_SCHEMA_VERSION
    assert spec.architecture == "zimage"
    assert (spec.rank, spec.alpha) == (16, 8.0)
    spec.validate()


def test_foreign_kohya_metadata_is_refused_rather_than_read_as_lora():
    kohya = {
        "ss_network_module": "lycoris.kohya",
        "ss_network_algo": "loha",
        "ss_network_dim": "8",
        "ss_network_alpha": "4",
    }
    with pytest.raises(AdapterIncompatible) as excinfo:
        AdapterSpec.from_metadata(kohya)
    assert "detect_adapter_codec" in str(excinfo.value)

    # The codec IS able to read it, and that is the documented entry point.
    detected = detect_adapter_codec(
        {"lora_unet_blocks_0_attn_to_q.hada_w1_a": torch.randn(8, 8)},
        metadata=kohya,
    )
    spec = AdapterSpec.from_codec(detected)
    assert (spec.algorithm, spec.format) == ("loha", "lycoris_kohya")


def test_a_sushiui_block_that_also_carries_ss_keys_is_read_not_refused():
    spec = AdapterSpec.from_metadata({
        "sushi.adapter.algorithm": "lokr",
        "sushi.adapter.format": "sushiui_canonical",
        "ss_network_module": "lycoris.kohya",
        "lora_rank": "0",
    })
    assert spec.algorithm == "lokr"


def test_dora_stays_two_axis_and_is_not_a_fourth_algorithm():
    tensors = _down_up_tensors()
    tensors["lora_transformer_blocks_0_attn_to_q.dora_scale"] = torch.randn(8)
    spec = AdapterSpec.from_codec(detect_adapter_codec(tensors))
    assert spec.algorithm == "lora"
    assert spec.weight_decompose is True
    assert spec.family == "dora"


def test_metadata_round_trip_is_exact():
    spec = AdapterSpec(
        algorithm="lokr",
        weight_decompose=True,
        rank=8,
        alpha=4.0,
        architecture="krea2",
        components=("unet", "text_encoder_2"),
        format=FORMAT_SUSHIUI,
        options={OPTION_FACTOR: 8, OPTION_USE_TUCKER: False},
    )
    metadata = spec.to_metadata()
    assert metadata[METADATA_ALGORITHM] == "lokr"
    assert metadata[METADATA_WEIGHT_DECOMPOSE] == "true"
    assert metadata["target_scope"] == "unet,text_encoder_2"
    assert AdapterSpec.from_metadata(metadata) == spec
    assert spec.family == "dokr"
    assert spec.lokr_factor == 8
    assert spec.use_tucker is False


def test_metadata_round_trip_of_a_bare_spec():
    spec = AdapterSpec(algorithm="loha", rank=4, alpha=4.0,
                       options={OPTION_USE_TUCKER: True})
    assert AdapterSpec.from_metadata(spec.to_metadata()) == spec
    assert spec.use_tucker is True
    assert spec.lokr_factor is None


@pytest.mark.parametrize("spec", [
    AdapterSpec(algorithm="unknown", rank=4),
    AdapterSpec(algorithm="dora", rank=4),
    AdapterSpec(algorithm="lora", rank=4, format="unknown"),
    AdapterSpec(algorithm="lora", rank=-4),
    AdapterSpec(algorithm="lora", rank=0),
    AdapterSpec(algorithm="loha", rank=None, alpha=8.0),
    AdapterSpec(algorithm="lora", rank=4, architecture="stable_cascade"),
    AdapterSpec(algorithm="lora", rank=4,
                schema_version=ADAPTER_SCHEMA_VERSION + 1),
    AdapterSpec(algorithm="lokr", rank=0, options={OPTION_FACTOR: 0}),
    AdapterSpec(algorithm="lokr", rank=0, options={OPTION_FACTOR: -2}),
])
def test_validate_refuses_an_inconsistent_spec(spec):
    with pytest.raises(AdapterIncompatible) as excinfo:
        spec.validate()
    assert isinstance(excinfo.value, AdapterRefusal)
    assert excinfo.value.code == "lora_incompatible"
    assert str(excinfo.value)


def test_validate_accepts_the_lokr_full_form_at_rank_zero():
    # LoKr's unfactored form has no rank, unlike LoRA/LoHa whose scale is
    # alpha/rank (LoKrLinearLayer takes the lokr_w2 branch at rank 0).
    AdapterSpec(algorithm="lokr", rank=0, options={OPTION_FACTOR: -1}).validate()


@pytest.mark.parametrize("name", ["hada_t1", "hada_t2", "lokr_t2"])
def test_a_tucker_checkpoint_is_detected_and_refused(name):
    """Tucker factors exist only for a target with kernel dims, and this engine
    wraps Linear only -- so the tensor set has to be REFUSED where it is first
    inspected, not quietly dropped down to the two-factor form."""
    tensors = {
        "lora_unet_blocks_0_attn_to_q.hada_w1_a": torch.randn(8, 4),
        "lora_unet_blocks_0_attn_to_q.hada_w1_b": torch.randn(4, 8),
        "lora_unet_blocks_0_attn_to_q.hada_w2_a": torch.randn(8, 4),
        "lora_unet_blocks_0_attn_to_q.hada_w2_b": torch.randn(4, 8),
        f"lora_unet_blocks_0_attn_to_q.{name}": torch.randn(4, 4, 3, 3),
    }
    codec = detect_adapter_codec(tensors)
    assert codec.use_tucker is True
    spec = AdapterSpec.from_codec(codec)
    assert spec.use_tucker is True
    with pytest.raises(AdapterIncompatible):
        spec.validate()


def test_validate_takes_a_caller_supplied_architecture_list():
    spec = AdapterSpec(algorithm="lora", rank=4, architecture="zimage")
    with pytest.raises(AdapterIncompatible):
        spec.validate(known_architectures={"sd15"})
    spec.validate(known_architectures={"zimage"})


def test_options_are_read_only():
    spec = AdapterSpec(algorithm="lokr", rank=0, options={OPTION_FACTOR: 4})
    with pytest.raises(TypeError):
        spec.options[OPTION_FACTOR] = 8


# ---------------------------------------------------------------------------
# AdapterTarget
# ---------------------------------------------------------------------------


class _Tree(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 6)
        self.mlp = nn.Sequential(nn.Linear(6, 12), nn.GELU(), nn.Linear(12, 6))
        self.norm = nn.LayerNorm(6)
        self.half_proj = nn.Linear(6, 6).to(torch.float16)


def _by_path(tree, **kwargs):
    return {t.module_path: t for t in enumerate_adapter_targets(tree, **kwargs)}


def test_enumeration_finds_every_wrappable_linear_and_nothing_else():
    targets = _by_path(_Tree(), component="unet")
    assert set(targets) == {"proj", "mlp.0", "mlp.2", "half_proj"}
    assert all(t.component == "unet" for t in targets.values())


def test_enumeration_records_geometry_dtype_and_merge_capability():
    tree = _Tree()
    targets = _by_path(tree)

    proj = targets["proj"]
    assert (proj.in_features, proj.out_features) == (8, 6)
    assert proj.base_dtype is torch.float32
    assert proj.branch_dtype is torch.float32
    assert proj.quantization is None
    assert proj.mergeable is True

    half = targets["half_proj"]
    assert half.base_dtype is torch.float16
    assert half.branch_dtype is torch.float16
    assert half.mergeable is True


def test_float8_base_takes_the_default_branch_dtype_and_cannot_merge():
    fp8 = nn.Linear(4, 4).to(torch.float8_e4m3fn)
    target = AdapterTarget.describe("fp8", nn.Module(), "fp8", fp8)
    assert target.base_dtype is torch.float8_e4m3fn
    assert target.branch_dtype is torch.bfloat16
    # A merge needs a real float weight to write the delta into.
    assert target.mergeable is False
    assert target.quantization is None


def test_weight_only_quantized_base_reports_its_kind():
    from core.models.ideogram4.vendor.int8_linear import Int8Linear

    int8 = Int8Linear(8, 6, bias=True, compute_dtype=torch.bfloat16, device="cpu")
    assert is_lora_wrappable_linear(int8) is True
    assert quantization_kind(int8) == "int8"

    parent = nn.Module()
    parent.q = int8
    target = AdapterTarget.describe("q", parent, "q", int8)
    assert target.quantization == "int8"
    assert (target.in_features, target.out_features) == (8, 6)
    # int8 is not a floating dtype at all, so the branch falls back.
    assert target.branch_dtype is torch.bfloat16
    assert target.mergeable is False


def test_parent_slot_round_trips_through_the_module_slot_helpers():
    tree = _Tree()
    targets = _by_path(tree)

    for path, expected_slot in (("mlp.0", 0), ("proj", "proj")):
        target = targets[path]
        assert target.slot == expected_slot
        assert get_module_slot(target.parent, target.slot) is target.module

        composite = CompositeAdapterLayer(target.module)
        set_module_slot(target.parent, target.slot, composite)
        assert get_module_slot(target.parent, target.slot) is composite

        set_module_slot(target.parent, target.slot, target.module)
        assert get_module_slot(target.parent, target.slot) is target.module

    assert tree.mlp[0] is targets["mlp.0"].module
    assert tree.proj is targets["proj"].module


def test_enumeration_yields_neither_a_covered_slot_nor_its_inside():
    tree = _Tree()
    target = _by_path(tree)["proj"]
    set_module_slot(target.parent, target.slot,
                    CompositeAdapterLayer(target.module))

    paths = set(_by_path(tree))
    assert "proj" not in paths
    assert not any(p.startswith("proj.") for p in paths)
    assert paths == {"mlp.0", "mlp.2", "half_proj"}


def test_enumeration_honours_a_custom_predicate_and_tag_hooks():
    tree = _Tree()
    targets = _by_path(
        tree,
        predicate=lambda m: isinstance(m, nn.Linear) and m.out_features == 6,
        scope_of=lambda path: "attn" if path.startswith("mlp") else "proj",
        block_of=lambda path: path.split(".")[0],
    )
    assert set(targets) == {"proj", "mlp.2", "half_proj"}
    assert targets["mlp.2"].scope == "attn"
    assert targets["mlp.2"].block == "mlp"
    assert targets["proj"].scope == "proj"


def test_enumeration_passes_the_branch_dtype_default_through():
    targets = _by_path(_Tree(), branch_dtype_default=torch.float64)
    # fp32/fp16 bases keep their own dtype; only a base with no usable float
    # weight falls back, so assert on the fallback path too.
    assert targets["proj"].branch_dtype is torch.float32
    fp8 = nn.Linear(4, 4).to(torch.float8_e4m3fn)
    assert AdapterTarget.describe(
        "fp8", nn.Module(), "fp8", fp8,
        branch_dtype_default=torch.float64).branch_dtype is torch.float64


# ---------------------------------------------------------------------------
# Architecture adapter capability matrix
# ---------------------------------------------------------------------------

#: What round-trips today, per architecture. Growing a row is a Phase 2/3
#: change, not a test-maintenance chore -- see
#: docs/guides/LYCORIS_ADAPTER_DESIGN.md.
ORDINARY_ONLY = {("lora", False)}
#: The architectures whose generation branch builders run on
#: ``build_adapter_branch`` and which are gated by
#: ``adapter_lycoris_roundtrip_cheap_test.py``. MiniMax-H3 and SenseNova are
#: gated separately; SD1.5/SDXL never reach ``AdapterSession``.
ADDITIVE_LYCORIS = ORDINARY_ONLY | {("loha", False), ("lokr", False)}
SHIPPED_PAIRS = {
    "sd15": ORDINARY_ONLY,
    "sdxl": ORDINARY_ONLY,
    "zimage": ADDITIVE_LYCORIS,
    "anima": ADDITIVE_LYCORIS,
    "lens": ADDITIVE_LYCORIS,
    "ideogram4": ADDITIVE_LYCORIS,
    "minit2i": ADDITIVE_LYCORIS,
    "krea2": ADDITIVE_LYCORIS,
    "flux2": ADDITIVE_LYCORIS,
    "ltx2": ADDITIVE_LYCORIS,
    "minimax_h3": ORDINARY_ONLY,
    # sd-scripts codec only: the diffusers/PEFT branch bakes (lora_A|lora_B)
    # into its key regexes, so a LyCORIS file reaches no grouper there.
    "acestep": ADDITIVE_LYCORIS,
    "sensenova": ORDINARY_ONLY,
}


class TestArchAdapterCapability:
    @staticmethod
    def _registry():
        from core.training.arch import ARCH_REGISTRY

        return ARCH_REGISTRY

    def test_known_architectures_mirror_is_pinned_to_the_registry(self):
        assert KNOWN_ARCHITECTURES == set(self._registry())

    def test_every_registered_architecture_declares_a_matrix(self):
        from core.training.arch.base_arch import NO_ADAPTER_CAPABILITY

        for name, handler in self._registry().items():
            capability = handler.adapter_capability
            assert capability is not NO_ADAPTER_CAPABILITY, name
            assert capability.initial_dora in (
                "dense", "dense_only", "deferred", "refused"), name

    def test_each_architecture_supports_exactly_its_shipped_row(self):
        assert set(SHIPPED_PAIRS) == set(self._registry())
        for name, handler in self._registry().items():
            capability = handler.adapter_capability
            assert set(capability.supported) == SHIPPED_PAIRS[name], name
            assert capability.supports("lora") is True, name
            capability.require("lora", False)

    def test_no_architecture_enables_a_weight_decomposed_pair(self):
        """DoRA/DoHa/DoKr are Phase 3. A flip that reached the decomposition
        axis would be caught here rather than by a mis-scaled image."""
        for name, handler in self._registry().items():
            for algorithm in ("lora", "loha", "lokr"):
                assert not handler.adapter_capability.supports(algorithm, True), \
                    f"{name}/{algorithm}"

    def test_every_unsupported_pair_carries_a_reason_and_refuses(self):
        from core.training.arch.base_arch import ADAPTER_ALGORITHMS

        for name, handler in self._registry().items():
            capability = handler.adapter_capability
            for algorithm in ADAPTER_ALGORITHMS:
                for decompose in (False, True):
                    if (algorithm, decompose) in SHIPPED_PAIRS[name]:
                        continue
                    label = f"{name}/{algorithm}/decompose={decompose}"
                    assert capability.supports(algorithm, decompose) is False, label
                    reason = capability.refusal_reason(algorithm, decompose)
                    assert reason and name in reason, label
                    with pytest.raises(ValueError):
                        capability.require(algorithm, decompose)

    def test_no_architecture_carries_additive_branches_over_a_quantized_base(self):
        for name, handler in self._registry().items():
            capability = handler.adapter_capability
            assert capability.quantized_base_additive_family is False, name
            assert capability.quantized_base_reason, name

    def test_every_registered_architecture_declares_the_additive_family(self):
        # "Additive family: yes" for all thirteen rows of the design doc table;
        # the two "yes, later gate" rows are separated by additive_gated below.
        for name, handler in self._registry().items():
            assert handler.adapter_capability.additive_family is True, name

    def test_a_decomposed_non_lora_pair_names_every_half_that_is_missing(self):
        """DoHa/DoKr are blocked twice over where the additive algebra is not
        enabled, and by the decomposition ALONE where it is -- telling a Z-Image
        user that LoHa is unimplemented would now be false."""
        for name, handler in self._registry().items():
            capability = handler.adapter_capability
            for algorithm in ("loha", "lokr"):
                decomposed = capability.refusal_reason(algorithm, True)
                additive = capability.refusal_reason(algorithm, False)
                dora = capability.refusal_reason("lora", True)
                label = f"{name}/{algorithm}"
                assert dora.split(": ", 1)[1] in decomposed, label
                if additive is None:
                    assert decomposed == dora, label
                else:
                    assert additive.split(": ", 1)[1] in decomposed, label

    def test_the_two_later_gate_architectures_are_marked_as_such(self):
        gated = {name for name, handler in self._registry().items()
                 if handler.adapter_capability.additive_gated}
        assert gated == {"minimax_h3", "sensenova"}

    def test_the_design_doc_dora_verdicts_are_declared_verbatim(self):
        expected = {
            "sd15": "dense", "sdxl": "dense", "zimage": "dense",
            "flux2": "dense_only", "anima": "dense_only", "lens": "dense",
            "ideogram4": "deferred", "minit2i": "dense", "krea2": "deferred",
            "ltx2": "dense_only", "minimax_h3": "deferred",
            "acestep": "dense_only", "sensenova": "deferred",
        }
        actual = {name: handler.adapter_capability.initial_dora
                  for name, handler in self._registry().items()}
        assert actual == expected

    def test_a_handler_without_a_declaration_supports_nothing(self):
        from core.training.arch.base_arch import ArchHandler

        capability = ArchHandler.adapter_capability
        assert set(capability.supported) == set()
        assert capability.supports("lora") is False
        with pytest.raises(ValueError):
            capability.require("lora", False)

    def test_an_incomplete_declaration_is_rejected_at_construction(self):
        from core.training.arch.base_arch import AdapterCapability

        with pytest.raises(ValueError):
            AdapterCapability(
                additive_family=True,
                initial_dora="dense",
                supported=frozenset({("lora", False)}),
                refusals={("loha", False): "reason"},  # four pairs unaccounted
                quantized_base_additive_family=False,
                quantized_base_reason="reason",
            )
