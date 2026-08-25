"""Phase U-2-2: the full-parameter adapter and the contract around it.

The centre of this file is the NEGATIVE CONTROL: without an architecture branch
in ``full_parameter_trainer._create_adapter`` the run falls through to the SD1.5
adapter, which gates every group it builds on ``trainer.unet`` /
``trainer.text_encoder`` being present -- and SenseNova sets both to None. The
measured result is recorded here as a number, not as prose.

The 42-layer MoT tree and the ``load_components`` harness come from the U-2-1
test module rather than being rebuilt: a second layout table is how the two
drift apart.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sensenova_int8_materialize_test import _Decoder, _load_with, _state_digest

from core.models.ideogram4.vendor.int8_linear import Int8Linear
from core.models.sensenova.loader import (
    SENSENOVA_BRANCH_LINEAR_COUNTS,
    materialize_int8_decoder_linears,
)
from core.models.sensenova.sensenova_lora import (
    iter_sensenova_lora_targets,
    und_gradient_unreachable_paths,
)
from core.training.adapters import (
    SD15FullParameterAdapter,
    SenseNovaFullParameterAdapter,
    SenseNovaLoRAAdapter,
)
from core.training.base_trainer import BaseTrainer
from core.training.full_parameter_trainer import FullParameterTrainer
from core.training.ops.sensenova_ops import (
    SENSENOVA_FULL_FINETUNE_OPTIMIZERS,
    assert_full_finetune_contract,
    load_components,
)

_BRANCH_FLAGS = {
    "gen": {"train_unet": True, "train_text_encoder": False},
    "und": {"train_unet": False, "train_text_encoder": True},
    "both": {"train_unet": True, "train_text_encoder": True},
}


def _full_ft_trainer(branch: str, transformer: nn.Module, **extra):
    """A trainer namespace shaped exactly as ``load_components`` leaves one."""
    trainer = SimpleNamespace(
        transformer=transformer,
        # The two Nones this whole task exists because of.
        unet=None,
        text_encoder=None,
        text_encoder_2=None,
        vae=None,
        trains_base_weights=True,
        is_sensenova=True,
        weight_dtype=torch.bfloat16,
        training_dtype=torch.bfloat16,
        use_grad_scaler=False,
        unet_lr=1e-6,
        text_encoder_1_lr=None,
        text_encoder_lr=None,
        config={"optimizer": "adafactor"},
        **_BRANCH_FLAGS[branch],
    )
    for key, value in extra.items():
        setattr(trainer, key, value)
    return trainer


def _materialized(branch: str) -> nn.Module:
    transformer = _Decoder()
    materialize_int8_decoder_linears(transformer, branch=branch)
    return transformer


def _collected(groups):
    """Flatten optimizer param groups -> (tensor count, element count)."""
    params = [p for group in groups for p in group["params"]]
    return len(params), sum(p.numel() for p in params)


# ---------------------------------------------------------------------------
# The negative control
# ---------------------------------------------------------------------------

def test_negative_control_the_sd15_fallthrough_collects_zero():
    """Reproduces the bug the ``elif`` branch prevents, with the numbers.

    ``is_sensenova = False`` on the dispatch object is exactly the pre-U-2-2
    ``_create_adapter``: before this commit no branch keyed on that flag existed,
    so a SenseNova trainer reached ``else: SD15FullParameterAdapter``.
    """
    transformer = _materialized("und")
    materialized_linears = sum(
        1
        for _, _, _, module in iter_sensenova_lora_targets(transformer, branch="und")
        if type(module) is nn.Linear
    )
    materialized_parameters = len(list(transformer.parameters()))
    assert materialized_linears == 294
    assert materialized_parameters == 294

    trainer = _full_ft_trainer("und", transformer)
    adapter = SD15FullParameterAdapter(trainer)
    # It does not even fail: the quantized-base guard sees trainer.unet=None.
    adapter.prepare_models_for_training()
    groups = adapter.setup_trainable_parameters()

    tensors, elements = _collected(groups)
    assert groups == []
    assert (tensors, elements) == (0, 0)
    # 294 Parameters were dequantized for this run and 0 of them reach the
    # optimizer, with no error anywhere on the path.
    assert materialized_parameters == 294 and tensors == 0

    # And the branch that now exists collects them.
    fixed = SenseNovaFullParameterAdapter(trainer)
    fixed.prepare_models_for_training()
    assert _collected(fixed.setup_trainable_parameters()) == (294, 294 * 4 * 8)


def _dispatch_stub(**flags):
    """A stand-in for ``self`` in ``FullParameterTrainer._create_adapter``."""
    stub = SimpleNamespace(log_prefix="[test]")
    for name in (
        "is_zimage", "is_flux2", "is_anima", "is_lens", "is_minit2i", "is_krea2",
        "is_ltx2", "is_minimax_h3", "is_acestep", "is_sdxl", "is_sensenova",
    ):
        setattr(stub, name, False)
    for name, value in flags.items():
        setattr(stub, name, value)
    return stub


def test_create_adapter_dispatches_sensenova_above_the_sd15_fallthrough():
    stub = _dispatch_stub(is_sensenova=True)
    FullParameterTrainer._create_adapter(stub)
    assert isinstance(stub.adapter, SenseNovaFullParameterAdapter)

    # Same call with the flag down is the fallthrough the control above measured.
    other = _dispatch_stub()
    FullParameterTrainer._create_adapter(other)
    assert isinstance(other.adapter, SD15FullParameterAdapter)


# ---------------------------------------------------------------------------
# Collection scope == materialization scope
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("branch,expected", [("gen", 294), ("und", 294), ("both", 588)])
def test_collection_scope_is_exactly_the_loaders_materialization_scope(branch, expected):
    """Both sides resolve the branch and enumerate it through the same functions."""
    transformer = _Decoder()
    trainer = _full_ft_trainer(branch, transformer)
    trainer.config = {"training_method": "full_finetune",
                      "optimizer": "adafactor"}
    _load_with(
        {"config": trainer.config, **_BRANCH_FLAGS[branch]}, transformer
    )
    materialized = {
        path
        for path, _, _, module in iter_sensenova_lora_targets(transformer, branch="both")
        if type(module) is nn.Linear
    }
    assert len(materialized) == SENSENOVA_BRANCH_LINEAR_COUNTS[branch] == expected

    adapter = SenseNovaFullParameterAdapter(trainer)
    adapter.prepare_models_for_training()

    trained = {
        path
        for path, _, _, module in iter_sensenova_lora_targets(transformer, branch="both")
        if any(p.requires_grad for p in module.parameters())
    }
    assert trained == materialized

    tensors, _ = _collected(adapter.setup_trainable_parameters())
    assert tensors == expected
    # Nothing outside that scope was unfrozen.
    assert sum(1 for p in transformer.parameters() if p.requires_grad) == expected


def test_the_five_unreachable_understanding_targets_are_collected_anyway():
    """294 collected; a later update census must expect 289 of them to move.

    The five layer-41 understanding projections a t2i loss cannot reach are
    materialized like the rest and hold real Parameters, so they are collected
    like the rest. Whether their updates are non-zero is U-2-5's measurement,
    not this file's -- what is pinned here is that the collected count is 294
    and that these five are inside it, so a census written against 294 fails on
    exactly them.
    """
    transformer = _materialized("und")
    trainer = _full_ft_trainer("und", transformer)
    adapter = SenseNovaFullParameterAdapter(trainer)
    adapter.prepare_models_for_training()

    tensors, _ = _collected(adapter.setup_trainable_parameters())
    assert tensors == 294

    unreachable = und_gradient_unreachable_paths()
    assert len(unreachable) == 5
    trained = {
        path
        for path, _, _, module in iter_sensenova_lora_targets(transformer, branch="und")
        if any(p.requires_grad for p in module.parameters())
    }
    assert unreachable <= trained
    assert len(trained) - len(unreachable) == 289


def test_collection_reads_the_transformer_and_never_the_text_encoder():
    transformer = _materialized("und")
    trainer = _full_ft_trainer("und", transformer)

    class _Exploding:
        def __getattr__(self, name):
            raise AssertionError(
                "SenseNova full FT must never read trainer.text_encoder"
            )

    trainer.text_encoder = _Exploding()
    adapter = SenseNovaFullParameterAdapter(trainer)
    adapter.prepare_models_for_training()
    assert _collected(adapter.setup_trainable_parameters()) == (294, 294 * 4 * 8)


def test_an_unmaterialized_half_is_refused_rather_than_silently_empty():
    transformer = _Decoder()  # every target still Int8Linear
    trainer = _full_ft_trainer("und", transformer)
    adapter = SenseNovaFullParameterAdapter(trainer)
    with pytest.raises(RuntimeError, match="still holding"):
        adapter.prepare_models_for_training()
    assert all(
        type(module) is Int8Linear
        for _, _, _, module in iter_sensenova_lora_targets(transformer, branch="und")
    )


# ---------------------------------------------------------------------------
# Learning rates
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "attrs,expected",
    [
        ({"text_encoder_1_lr": 3e-6, "text_encoder_lr": 2e-6}, 3e-6),
        ({"text_encoder_1_lr": None, "text_encoder_lr": 2e-6}, 2e-6),
        ({"text_encoder_1_lr": None, "text_encoder_lr": None}, 1e-6),
    ],
)
def test_understanding_lr_follows_the_text_encoder_1_chain(attrs, expected):
    """text_encoder_1_lr -> text_encoder_lr -> unet_lr, as the LoRA adapter does."""
    transformer = _materialized("both")
    trainer = _full_ft_trainer("both", transformer, unet_lr=1e-6, **attrs)
    adapter = SenseNovaFullParameterAdapter(trainer)
    adapter.prepare_models_for_training()

    groups = adapter.setup_trainable_parameters()
    assert [g["lr"] for g in groups] == [1e-6, expected]
    # Group order mirrors base_trainer._build_component_lr_list's SenseNova entries
    # (generation first, understanding second), which a resume remaps by index.
    assert [len(g["params"]) for g in groups] == [294, 294]


def test_a_single_half_produces_a_single_group():
    for branch, lr in (("gen", 1e-6), ("und", 5e-6)):
        transformer = _materialized(branch)
        trainer = _full_ft_trainer(branch, transformer, text_encoder_1_lr=5e-6)
        adapter = SenseNovaFullParameterAdapter(trainer)
        adapter.prepare_models_for_training()
        groups = adapter.setup_trainable_parameters()
        assert len(groups) == 1
        assert groups[0]["lr"] == lr
        assert len(groups[0]["params"]) == 294


# ---------------------------------------------------------------------------
# The checkpoint format is not decided here
# ---------------------------------------------------------------------------

def test_saving_is_refused_because_the_output_format_is_undecided():
    adapter = SenseNovaFullParameterAdapter(_full_ft_trainer("gen", _materialized("gen")))
    with pytest.raises(NotImplementedError, match="output.*format is undecided"):
        adapter.save_checkpoint(1, 0, Path("unused.safetensors"))


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------

def test_adafactor_is_the_only_optimizer_this_route_accepts():
    """The ring-buffer pair is U-2-6's, behind gates G-RB2/G-RB3, and stays shut.

    Their CPU-resident state is what would make them fit; nothing supplies
    ``get_state_buffer``, so today they allocate the same 2.031250 B/param on
    the GPU that adamw8bit does.
    """
    trainer = _full_ft_trainer("gen", _Decoder())
    assert SENSENOVA_FULL_FINETUNE_OPTIMIZERS == ("adafactor",)
    assert_full_finetune_contract(trainer, "adafactor")
    # An absent config key is not a refusal on the pre-load channel: it carries
    # no information, and setup_optimizer's call always names one.
    trainer.config = {}
    assert_full_finetune_contract(trainer)

    for name in ("adamw8bit", "adamw8bit_ringbuffer", "lion8bit_ringbuffer",
                 "lion", "prodigy", "adafactor8bit"):
        with pytest.raises(ValueError, match=f"optimizer='{name}'"):
            assert_full_finetune_contract(trainer, name)


def test_the_optimizer_refusal_states_the_two_conditions_and_the_measured_cost():
    trainer = _full_ft_trainer("gen", _Decoder())
    with pytest.raises(ValueError) as excinfo:
        assert_full_finetune_contract(trainer, "adamw8bit")
    message = str(excinfo.value)
    # Not "only adafactor implements the hooks" -- adamw8bit does too.
    assert "per-parameter seam" in message
    assert "2.031250 B/param" in message and "16.5 GB" in message
    assert "8.10 G" in message  # the gen half, which is this route's default

    with pytest.raises(ValueError) as ring:
        assert_full_finetune_contract(trainer, "adamw8bit_ringbuffer")
    assert "get_state_buffer" in str(ring.value)


def test_adamw_is_refused_by_name_with_the_reason_it_cannot_be_repaired():
    trainer = _full_ft_trainer("gen", _Decoder())
    with pytest.raises(ValueError, match="no per-parameter seam") as excinfo:
        assert_full_finetune_contract(trainer, "adamw")
    assert "84.5%" in str(excinfo.value)


@pytest.mark.parametrize(
    "mutation,pattern",
    [
        ({"weight_dtype": torch.float16}, "requires bf16"),
        ({"training_dtype": torch.float32}, "requires bf16"),
        ({"use_grad_scaler": True}, "gradient scaling"),
        ({"config": {"use_ema": True}}, "use_ema"),
        ({"config": {"num_optimizer_groups": 2}}, "num_optimizer_groups=0"),
        ({"config": {"gradient_accumulation_steps": 4}}, "gradient_accumulation_steps=1"),
    ],
)
def test_each_refusal_fires_for_its_own_configuration(mutation, pattern):
    trainer = _full_ft_trainer("gen", _Decoder(), **mutation)
    with pytest.raises(ValueError, match=pattern):
        assert_full_finetune_contract(trainer, "adafactor")


def test_num_optimizer_groups_is_read_on_both_channels():
    """The YAML dict and the constructor argument, which can disagree.

    If only the config were checked, an attribute-side value would pass the
    contract, skip the fused install (which tests the attribute) and skip fused
    optimizer groups too (they exist only under Block Swap): a run with every
    gradient resident.
    """
    trainer = _full_ft_trainer("gen", _Decoder(), num_optimizer_groups=6)
    assert trainer.config.get("num_optimizer_groups") is None
    with pytest.raises(ValueError, match="num_optimizer_groups=0"):
        assert_full_finetune_contract(trainer, "adafactor")


def test_the_contract_passes_the_configuration_it_is_designed_for():
    trainer = _full_ft_trainer(
        "both",
        _Decoder(),
        config={
            "optimizer": "adafactor",
            "use_ema": False,
            "num_optimizer_groups": 0,
            "gradient_accumulation_steps": 1,
        },
    )
    assert_full_finetune_contract(trainer)
    assert_full_finetune_contract(trainer, "adafactor")


def test_the_contract_runs_before_the_load_and_refuses_without_paying_for_it():
    trainer = SimpleNamespace(
        model_path="checkpoint.safetensors",
        weight_dtype=torch.bfloat16,
        training_dtype=torch.bfloat16,
        use_grad_scaler=False,
        device=torch.device("cpu"),
        attention_backend="native",
        config={"training_method": "full_finetune", "optimizer": "adamw"},
        train_unet=True,
        train_text_encoder=False,
    )
    with patch("core.models.sensenova.loader.load_sensenova_from_path") as load:
        with pytest.raises(ValueError, match="optimizer='adamw'"):
            load_components(trainer)
    load.assert_not_called()


# ---------------------------------------------------------------------------
# LoRA is unchanged -- proven, not asserted
# ---------------------------------------------------------------------------

def test_a_lora_load_never_reaches_the_full_finetune_contract():
    transformer = _Decoder()
    before = _state_digest(transformer)
    with patch(
        "core.training.ops.sensenova_ops.assert_full_finetune_contract"
    ) as contract:
        _load_with({"config": {"optimizer": "adamw",
                               "use_ema": True,
                               "gradient_accumulation_steps": 4}}, transformer)
    contract.assert_not_called()
    assert _state_digest(transformer) == before
    assert list(transformer.parameters()) == []


def test_the_configurations_lora_always_allowed_still_load():
    """adamw + EMA + accumulation: refused for full FT, untouched for LoRA."""
    transformer = _Decoder()
    trainer = _load_with(
        {"config": {"optimizer": "adamw", "use_ema": True,
                    "gradient_accumulation_steps": 8},
         "train_unet": True, "train_text_encoder": True},
        transformer,
    )
    assert trainer.transformer is transformer
    assert all(
        type(module) is Int8Linear
        for _, _, _, module in iter_sensenova_lora_targets(transformer, branch="both")
    )


def test_the_lora_adapter_still_wraps_both_halves_over_the_int8_base():
    transformer = _Decoder()
    trainer = SimpleNamespace(
        transformer=transformer,
        train_text_encoder=True,
        unet_lr=1e-4,
        text_encoder_1_lr=5e-5,
        text_encoder_lr=None,
    )
    adapter = SenseNovaLoRAAdapter(trainer, 4, 4, torch.float32)
    layers = {}
    with patch(
        "core.training.ops.sensenova_ops.assert_understanding_training_supported"
    ):
        assert adapter.apply_lora_to_unet(layers) == 294
        assert adapter.apply_lora_to_text_encoders(layers) == 294
    groups = adapter.setup_trainable_parameters(layers)
    assert [g["lr"] for g in groups] == [1e-4, 5e-5]


# ---------------------------------------------------------------------------
# The full-FT default for train_text_encoder
# ---------------------------------------------------------------------------

def test_full_finetune_config_defaults_to_the_generation_half_for_sensenova():
    """A legacy-kwargs caller that omits the key gets one half, not both."""
    import yaml

    from core.training.training_config import TrainingConfigGenerator

    def _train_section(path):
        return yaml.safe_load(
            TrainingConfigGenerator.generate_full_finetune_config(
                run_name="r", base_model_path=path, output_dir="o",
                dataset_path="d", total_steps=10,
            )
        )["config"]["process"][0]["train"]

    assert _train_section("M:/model/sensenova/sensenova_int8.safetensors")[
        "train_text_encoder"] is False
    # Every other architecture keeps the default it always had.
    assert _train_section("M:/model/sdxl/some_model.safetensors")[
        "train_text_encoder"] is True
    # And an explicit request still wins: this is a default, not a refusal.
    explicit = yaml.safe_load(
        TrainingConfigGenerator.generate_full_finetune_config(
            run_name="r", base_model_path="M:/model/sensenova/x.safetensors",
            output_dir="o", dataset_path="d", total_steps=10,
            train_text_encoder=True,
        )
    )["config"]["process"][0]["train"]
    assert explicit["train_text_encoder"] is True


# ---------------------------------------------------------------------------
# The fused backward pass without block swap
# ---------------------------------------------------------------------------

class _BareTrainer(BaseTrainer):
    """Only what ``setup_optimizer`` reads; BaseTrainer.__init__ loads a model."""

    def __init__(self, *, is_sensenova: bool, full_finetune: bool, optimizer: str):
        self.log_prefix = "[test]"
        self.is_sensenova = is_sensenova
        self.trains_base_weights = full_finetune
        self.blocks_to_swap = 0
        self.num_optimizer_groups = 0
        self.use_fused_backward = False
        self.fused_optimizer_groups = None
        self.optimizer_schedule_free = False
        self.optimizer_cautious = False
        self.optimizer_schedule_free_r = 0.0
        self.optimizer_schedule_free_weight_lr_power = 2.0
        self.optimizer_use_radam = False
        self.optimizer_warmup_steps = 0
        self.optimizer_weight_decay = None
        self.optimizer_beta1 = None
        self.optimizer_beta2 = None
        self.optimizer_epsilon = None
        self.optimizer_stochastic_rounding = False
        self.use_grad_scaler = False
        self.weight_dtype = torch.bfloat16
        self.training_dtype = torch.bfloat16
        self.learning_rate = 1e-6
        self.config = {"optimizer": optimizer}
        self.transformer = nn.Linear(4, 4)
        self.unet = None
        self._groups = [{"params": list(self.transformer.parameters()), "lr": 1e-6}]

    def setup_trainable_parameters(self):
        return self._groups

    def save_checkpoint(self, step, epoch):
        pass

    def load_checkpoint(self, checkpoint_path):
        return 0


@pytest.mark.parametrize(
    "is_sensenova,full_finetune,fused",
    [(True, True, True), (True, False, False), (False, True, False)],
)
def test_only_a_sensenova_full_finetune_gets_fused_backward_without_block_swap(
    is_sensenova, full_finetune, fused
):
    """blocks_to_swap=0 everywhere; only one of the three installs the hooks.

    SenseNova refuses a non-zero blocks_to_swap in five places, with the same
    message (`base_trainer.py:1393`, `:1995`, `:8400`, `sensenova_ops.py:459`,
    `train_runner.py:176`), plus a sixth refusal of a different kind where
    `arch/sensenova.py:19-20` raises NotImplementedError from setup_block_swap.
    Everywhere else the fused backward pass is only set up inside
    `blocks_to_swap > 0`.
    Its full fine-tune would otherwise hold every gradient of the half it trains
    resident until optimizer.step().
    """
    trainer = _BareTrainer(
        is_sensenova=is_sensenova, full_finetune=full_finetune, optimizer="adafactor"
    )
    trainer.setup_optimizer("adafactor", "constant", 10)
    assert trainer.blocks_to_swap == 0
    assert trainer.use_fused_backward is fused
    assert hasattr(trainer.optimizer, "step_param") is fused


def test_fp16_is_refused_by_this_contract_and_not_by_the_block_swap_message():
    """c56d8a19's refusal names Block Swap, which SenseNova never has.

    It is reachable from `_setup_fused_backward_pass`, so this contract has to
    refuse the grad scaler first or the message would prescribe disabling
    something that was already off.
    """
    trainer = _BareTrainer(is_sensenova=True, full_finetune=True, optimizer="adafactor")
    trainer.use_grad_scaler = True
    trainer.weight_dtype = torch.bfloat16
    with pytest.raises(ValueError, match="does not support FP16 gradient scaling") as excinfo:
        trainer.setup_optimizer("adafactor", "constant", 10)
    assert "Block Swap" not in str(excinfo.value)
    assert trainer.use_fused_backward is False


def test_the_authoritative_optimizer_name_is_the_one_setup_optimizer_receives():
    """The config said adafactor; the run was started with adamw8bit."""
    trainer = _BareTrainer(is_sensenova=True, full_finetune=True, optimizer="adafactor")
    with pytest.raises(ValueError, match="optimizer='adamw8bit'"):
        trainer.setup_optimizer("adamw8bit", "constant", 10)


def test_an_uninstalled_fused_pass_raises_instead_of_running_fully_resident():
    """Widening the contract without FUSED_BACKWARD_OPTIMIZERS must not be quiet."""
    from core.training.ops import sensenova_ops

    trainer = _BareTrainer(is_sensenova=True, full_finetune=True, optimizer="lion")
    with patch.object(sensenova_ops, "SENSENOVA_FULL_FINETUNE_OPTIMIZERS", ("lion",)):
        with pytest.raises(RuntimeError, match="did not install its fused backward"):
            trainer.setup_optimizer("lion", "constant", 10)
    assert trainer.use_fused_backward is False


def test_train_refuses_accumulation_for_a_full_finetune_and_allows_it_for_lora():
    """The argument train() receives, which is not the value the contract read."""
    full = _BareTrainer(is_sensenova=True, full_finetune=True, optimizer="adafactor")
    with pytest.raises(ValueError, match="requires gradient_accumulation_steps=1"):
        BaseTrainer.train(full, datasets=[], batch_size=1,
                          gradient_accumulation_steps=4)

    lora = _BareTrainer(is_sensenova=True, full_finetune=False, optimizer="adamw8bit")
    try:
        BaseTrainer.train(lora, datasets=[], batch_size=1,
                          gradient_accumulation_steps=4)
    except Exception as error:  # it gets past the guard and fails further in
        assert "gradient_accumulation_steps=1" not in str(error)

    # And the batch-size message only recommends accumulation where it works.
    with pytest.raises(ValueError, match="requires batch_size=1") as full_batch:
        BaseTrainer.train(full, datasets=[], batch_size=2)
    assert "gradient_accumulation_steps" not in str(full_batch.value)
    with pytest.raises(ValueError, match="requires batch_size=1") as lora_batch:
        BaseTrainer.train(lora, datasets=[], batch_size=2)
    assert "gradient_accumulation_steps" in str(lora_batch.value)
