"""Phase U-2-2 step 3: the SenseNova full-fine-tune ACCEPTANCE path.

The two refusals that used to end this path -- ``TRAINING_UNSUPPORTED
["sensenova"]["full_finetune"]`` and ``train_runner``'s ``network.type != "lora"``
-- are open. Everything below is what had to keep holding when they opened:

* the run is accepted, and only for the two methods that are implemented;
* ``SenseNovaFullParameterAdapter`` is what gets built, with the SD1.5
  fallthrough measured as a NEGATIVE CONTROL (it collects zero, silently);
* every clause of the envelope still refuses, on both the config and the
  attribute channel;
* LoRA runs are untouched.

Scope note: the refusals themselves are pinned in detail by
``sensenova_full_finetune_adapter_test.py``. What is new here is that they are
reachable -- and still fire -- from the acceptance path a queued run takes.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sensenova_full_finetune_adapter_test import (  # noqa: E402
    _collected,
    _dispatch_stub,
    _full_ft_trainer,
    _materialized,
)

from api.arch_capabilities import TRAINING_UNSUPPORTED  # noqa: E402
from core.model_loader import ModelLoader  # noqa: E402
from core.models.sensenova.sensenova_lora import (  # noqa: E402
    iter_sensenova_lora_targets,
)
from core.training.adapters import (  # noqa: E402
    SD15FullParameterAdapter,
    SenseNovaFullParameterAdapter,
)
from core.training.full_parameter_trainer import FullParameterTrainer  # noqa: E402
from core.training.ops.sensenova_ops import (  # noqa: E402
    SENSENOVA_FULL_FINETUNE_OPTIMIZERS,
    assert_full_finetune_contract,
)
from core.training.train_runner import (  # noqa: E402
    _apply_sensenova_training_contract,
)


def _sensenova():
    return patch.object(ModelLoader, "detect_model_type", return_value="sensenova")


def _train(**overrides):
    train = {"batch_size": 1, "blocks_to_swap": 0}
    train.update(overrides)
    return train


# ---------------------------------------------------------------------------
# The two gates
# ---------------------------------------------------------------------------

def test_the_capability_table_no_longer_refuses_a_full_finetune():
    assert "full_finetune" not in TRAINING_UNSUPPORTED["sensenova"]
    # And the neighbours it sat between did not move with it.
    assert "relora" in TRAINING_UNSUPPORTED["sensenova"]
    assert "controlnet" in TRAINING_UNSUPPORTED["sensenova"]
    assert "vae_decoder" not in TRAINING_UNSUPPORTED["sensenova"]


def test_the_pre_load_guard_that_reads_that_table_lets_the_run_through():
    with _sensenova():
        FullParameterTrainer._refuse_unsupported_full_finetune("model")


def test_the_runner_accepts_a_full_finetune_and_normalizes_it_like_a_lora_run():
    train = _train()
    with _sensenova():
        assert _apply_sensenova_training_contract(
            "model", "full_finetune", train, {"sample": {}}
        )
    assert train["text_encoding_mode"] == "onthefly_gpu"
    assert train["latent_encoding_mode"] == "onthefly_gpu"


@pytest.mark.parametrize("network_type", ["relora", "controlnet", "sd_trainer", ""])
def test_every_other_method_is_still_refused_and_by_name(network_type):
    """The removed check was the ONLY refusal a SenseNova ControlNet run met.

    ``relora`` has ``ReLoRATrainer._refuse_unsupported_relora`` reading the
    capability table; ``controlnet`` has a table entry and no trainer-side guard
    that reads it, so "accept anything that is not LoRA" would have opened it.
    """
    with _sensenova():
        with pytest.raises(ValueError, match=f"not '{network_type}'"):
            _apply_sensenova_training_contract(
                "model", network_type, _train(), {"sample": {}}
            )


def test_vae_decoder_is_still_exempt_from_the_whole_contract():
    train = {"batch_size": 4}
    with patch.object(ModelLoader, "detect_model_type") as detect:
        assert not _apply_sensenova_training_contract(
            "model", "vae_decoder", train, {}
        )
    detect.assert_not_called()
    assert train == {"batch_size": 4}


# ---------------------------------------------------------------------------
# The adapter, and the zero it prevents
# ---------------------------------------------------------------------------

def test_the_sensenova_adapter_is_selected_not_the_sd15_fallthrough():
    stub = _dispatch_stub(is_sensenova=True)
    FullParameterTrainer._create_adapter(stub)
    assert isinstance(stub.adapter, SenseNovaFullParameterAdapter)


def test_negative_control_the_fallthrough_collects_zero_of_the_294_it_paid_for():
    """The defect the ``elif`` prevents, as a number, on the gen branch.

    The adapter test measures this on the understanding half; the acceptance
    path's default branch is the generation one, and it is now reachable, so it
    is measured here too.
    """
    transformer = _materialized("gen")
    materialized = [
        module
        for _, _, _, module in iter_sensenova_lora_targets(transformer, branch="gen")
        if type(module) is nn.Linear
    ]
    assert len(materialized) == 294

    trainer = _full_ft_trainer("gen", transformer)
    fallthrough = SD15FullParameterAdapter(trainer)
    fallthrough.prepare_models_for_training()
    assert _collected(fallthrough.setup_trainable_parameters()) == (0, 0)

    correct = SenseNovaFullParameterAdapter(trainer)
    correct.prepare_models_for_training()
    tensors, _ = _collected(correct.setup_trainable_parameters())
    assert tensors == 294


# ---------------------------------------------------------------------------
# The envelope, from the acceptance path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "overrides,pattern",
    [
        ({"gradient_accumulation_steps": 4}, "gradient_accumulation_steps=1"),
        ({"num_optimizer_groups": 2}, "num_optimizer_groups=0"),
        ({"use_ema": True}, "use_ema"),
        ({"optimizer": "adamw"}, "optimizer='adamw'"),
        ({"optimizer": "adamw8bit"}, "optimizer='adamw8bit'"),
        ({"optimizer": "lion8bit_ringbuffer"}, "optimizer='lion8bit_ringbuffer'"),
        ({"sensenova_full_finetune_save_format": "fp8"}, "save_format"),
        ({"blocks_to_swap": 1}, "blocks_to_swap"),
    ],
)
def test_each_envelope_clause_still_refuses_before_the_process_loads_anything(
    overrides, pattern
):
    with _sensenova():
        with pytest.raises(ValueError, match=pattern):
            _apply_sensenova_training_contract(
                "model", "full_finetune", _train(**overrides), {"sample": {}}
            )


def test_the_allowed_optimizers_are_the_only_allowed_optimizers():
    assert SENSENOVA_FULL_FINETUNE_OPTIMIZERS == (
        "adafactor", "adamw8bit_ringbuffer", "lion8bit_ringbuffer")
    train = _train(optimizer="adafactor")
    with _sensenova():
        assert _apply_sensenova_training_contract(
            "model", "full_finetune", train, {"sample": {}}
        )


@pytest.mark.parametrize(
    "optimizer", ["adamw8bit_ringbuffer", "lion8bit_ringbuffer"])
def test_a_ring_buffer_optimizer_needs_host_resident_state(optimizer):
    """MUTANT: drop the assert_ringbuffer_host_state call from
    _apply_sensenova_full_finetune_contract and a product-started run allocates
    16.5-32.9 GB of 8-bit state on the GPU and OOMs inside step 1."""
    with _sensenova():
        with pytest.raises(ValueError, match="optimizer_state_host_resident"):
            _apply_sensenova_training_contract(
                "model", "full_finetune", _train(optimizer=optimizer),
                {"sample": {}})
        assert _apply_sensenova_training_contract(
            "model", "full_finetune",
            _train(optimizer=optimizer, optimizer_state_host_resident=True),
            {"sample": {}})


@pytest.mark.parametrize("save_format", ["mixed", "bf16", "int8"])
def test_every_shipped_save_format_is_accepted_and_normalized(save_format):
    train = _train(sensenova_full_finetune_save_format=save_format.upper())
    with _sensenova():
        assert _apply_sensenova_training_contract(
            "model", "full_finetune", train, {"sample": {}}
        )
    assert train["sensenova_full_finetune_save_format"] == save_format


@pytest.mark.parametrize(
    "mutation,pattern",
    [
        ({"weight_dtype": torch.float16}, "requires bf16"),
        ({"training_dtype": torch.float32}, "requires bf16"),
        ({"use_grad_scaler": True}, "FP16 gradient scaling"),
        ({"use_ema": True}, "use_ema"),
        ({"num_optimizer_groups": 3}, "num_optimizer_groups=0"),
    ],
)
def test_the_trainer_side_contract_still_refuses_on_the_attribute_channel(
    mutation, pattern
):
    """Not a duplicate of the runner clauses above: those read the YAML, these
    read the trainer, and the two disagree whenever anything sets an attribute
    without writing the config back."""
    trainer = _full_ft_trainer("gen", _materialized("gen"), **mutation)
    with pytest.raises(ValueError, match=pattern):
        assert_full_finetune_contract(trainer, "adafactor")


@pytest.mark.parametrize(
    "key,value,pattern",
    [
        ("use_ema", True, "use_ema"),
        ("num_optimizer_groups", 3, "num_optimizer_groups=0"),
        ("gradient_accumulation_steps", 2, "gradient_accumulation_steps=1"),
    ],
)
def test_the_trainer_side_contract_still_refuses_on_the_config_channel(
    key, value, pattern
):
    trainer = _full_ft_trainer("gen", _materialized("gen"))
    trainer.config = {"optimizer": "adafactor", key: value}
    with pytest.raises(ValueError, match=pattern):
        assert_full_finetune_contract(trainer, "adafactor")


# ---------------------------------------------------------------------------
# Stochastic rounding is a route requirement, still
# ---------------------------------------------------------------------------

def test_stochastic_rounding_is_forced_and_its_attachment_is_verified():
    from core.training.ops.sensenova_ops import (
        assert_full_finetune_stochastic_rounding_attached,
        enforce_full_finetune_stochastic_rounding,
    )
    from core.training.optimizers.stochastic_rounding import WRAPPED_ATTR

    trainer = _full_ft_trainer("gen", _materialized("gen"))
    trainer.optimizer_stochastic_rounding = False
    trainer.log_prefix = "[test]"
    assert enforce_full_finetune_stochastic_rounding(trainer) is True
    assert trainer.optimizer_stochastic_rounding is True

    step_param = lambda p, group: None  # noqa: E731
    setattr(step_param, WRAPPED_ATTR, True)
    trainer.optimizer = SimpleNamespace(step_param=step_param)
    trainer.fused_optimizer_groups = None
    assert_full_finetune_stochastic_rounding_attached(trainer, "adafactor")

    trainer.optimizer = SimpleNamespace(step_param=lambda p, group: None)
    with pytest.raises(RuntimeError, match="nothing is interposed"):
        assert_full_finetune_stochastic_rounding_attached(trainer, "adafactor")


# ---------------------------------------------------------------------------
# The updated-parameter census, on the one optimizer this route allows
# ---------------------------------------------------------------------------

def test_adafactors_fused_seam_reports_to_the_update_census():
    """It did not, and the census called a correct run 294-of-294 missing.

    ``setup_update_census`` arms the census for every fused-backward optimizer,
    but ``record_param_update`` was called only by the two ring-buffer ones --
    so the mechanism SENSENOVA_TRAINING_DESIGN.md 13.4 names as U-2-5's
    acceptance criterion reported total failure on the only optimizer this route
    permits. Found by the real smoke run, on its first step.
    """
    from transformers import Adafactor

    from core.training.optimizers.adafactor_fused import patch_adafactor_fused
    from core.training.optimizers.update_census import (
        UpdateCensus,
        attach_update_census,
        trainable_params_of,
    )

    module = nn.Linear(4, 4)
    optimizer = Adafactor(
        module.parameters(), lr=1e-3, relative_step=False, scale_parameter=False
    )
    census = UpdateCensus()
    attach_update_census(optimizer, census)
    assert census.expect(trainable_params_of(optimizer)) == 2
    patch_adafactor_fused(optimizer)

    census.begin_step(True)
    module(torch.randn(2, 4)).sum().backward()
    for group in optimizer.param_groups:
        for p in group["params"]:
            optimizer.step_param(p, group)
    census.assert_complete("test")
    assert census.updated_count == 2

    # Negative control: a parameter with no gradient is not recorded, so the
    # census still catches a hook that never fired.
    module.zero_grad(set_to_none=True)
    census.begin_step(True)
    for group in optimizer.param_groups:
        for p in group["params"]:
            optimizer.step_param(p, group)
    with pytest.raises(RuntimeError, match="received no optimizer update"):
        census.assert_complete("test")


# ---------------------------------------------------------------------------
# The checkpoint the run produces must be readable by the reader
# ---------------------------------------------------------------------------

def test_the_embedded_geometry_block_is_the_source_config_verbatim(tmp_path):
    """``to_dict()`` is not a fixed point of ``NEOChatConfig(**.)`` here.

    Two independent reasons, both found by loading a real saved checkpoint back:
    the ``to_dict`` override skips the base class's dtype normalization
    (``"torch.bfloat16"`` on read), and ``configuration_neo_vit``'s trailing
    comma makes ``downsample_ratio`` a tuple that grows a nesting level per
    round trip.
    """
    from core.models.common.quantized_export import sensenova_export_metadata
    from core.models.sensenova.loader import (
        _assert_config_metadata_reloads,
        _embeddable_sensenova_config,
        _serializable_sensenova_config,
    )
    from core.models.sensenova.vendor import NEOChatConfig

    source = {
        "model_type": "neo_chat",
        "vision_config": {"downsample_ratio": 0.5},
        "llm_config": {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3",
                       "num_hidden_layers": 2},
        "downsample_ratio": 0.5,
    }
    (tmp_path / "config.json").write_text(json.dumps(source), encoding="utf-8")
    live = NEOChatConfig(**source)
    live.dtype = torch.bfloat16

    embedded = _embeddable_sensenova_config(live, str(tmp_path))
    assert embedded == source
    _assert_config_metadata_reloads(sensenova_export_metadata(embedded))

    # The block THIS LOAD accepted wins over the sibling, because the reader
    # prefers embedded metadata over the sibling too: re-deriving from the
    # sibling could embed a different dict than the run was built from.
    raw = {**source, "t_eps": 0.05}
    assert _embeddable_sensenova_config(live, str(tmp_path), raw) == raw

    # The path taken when there is no source config: the dtype is repaired, and
    # what is still not a fixed point is REFUSED rather than written.
    fallback = _serializable_sensenova_config(live)
    assert fallback["dtype"] == "bfloat16"
    with pytest.raises(RuntimeError, match="cannot reconstruct"):
        _assert_config_metadata_reloads(sensenova_export_metadata(fallback))


# ---------------------------------------------------------------------------
# LoRA is unchanged
# ---------------------------------------------------------------------------

def test_a_lora_run_keeps_accumulation_and_its_own_batch_size_advice():
    train = _train(gradient_accumulation_steps=8, optimizer="adamw8bit",
                   use_ema=True, num_optimizer_groups=4)
    with _sensenova():
        assert _apply_sensenova_training_contract(
            "model", "lora", train, {"sample": {}}
        )
    assert train["gradient_accumulation_steps"] == 8

    with _sensenova():
        with pytest.raises(ValueError, match="gradient_accumulation_steps"):
            _apply_sensenova_training_contract(
                "model", "lora", _train(batch_size=2), {"sample": {}}
            )
    # The full-FT message must NOT offer accumulation: it refuses it.
    with _sensenova():
        with pytest.raises(ValueError) as excinfo:
            _apply_sensenova_training_contract(
                "model", "full_finetune", _train(batch_size=2), {"sample": {}}
            )
    assert "gradient_accumulation_steps" not in str(excinfo.value)


def test_the_full_finetune_dispatch_forwards_train_unet():
    """It did not, so ``train_unet=False`` + ``train_text_encoder=True`` asked
    for the understanding half and dequantized both.

    Read off the AST rather than the source text: what has to hold is that the
    single ``FullParameterTrainer(...)`` construction passes ``train_unet`` and
    passes something OTHER than a literal. Matching the spelling would pin the
    variable name instead, which is the technique this file replaced in
    ``sensenova_full_finetune_save_test.py``.
    """
    import ast

    tree = ast.parse(
        (Path(__file__).resolve().parents[1]
         / "core/training/train_runner.py").read_text(encoding="utf-8")
    )
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "FullParameterTrainer"
    ]
    assert len(calls) == 1, "more than one construction site; check them all"
    keywords = {kw.arg: kw.value for kw in calls[0].keywords}
    assert "train_unet" in keywords
    assert not isinstance(keywords["train_unet"], ast.Constant)
    # The scope flags travel together or the pair is meaningless.
    assert "train_text_encoder" in keywords


def test_both_scope_flags_false_is_refused_for_sensenova_by_name():
    """Newly reachable now that train_unet is forwarded.

    SenseNova raises a message that says what to set. Other architectures'
    adapters would collect nothing and hand an empty parameter list to the
    optimizer -- recorded in SENSENOVA_TRAINING_DESIGN.md 13.4, not fixed here.
    """
    from core.training.ops.sensenova_ops import resolve_full_finetune_branch

    trainer = SimpleNamespace(train_unet=False, train_text_encoder=False)
    with pytest.raises(ValueError, match="nothing to train"):
        resolve_full_finetune_branch(trainer)
    assert resolve_full_finetune_branch(
        SimpleNamespace(train_unet=False, train_text_encoder=True)) == "und"
    assert resolve_full_finetune_branch(
        SimpleNamespace(train_unet=True, train_text_encoder=False)) == "gen"


def test_a_checkpoint_this_repo_wrote_is_refused_by_name_of_the_setting():
    """H2: only ``int8`` is a legal base for a NEW run. The refusal names the
    setting that decided that, because the file it is refusing says which one
    was used -- and points at the resume path, which is a different gate
    (``accept_resume_shaped_base``, sensenova_full_finetune_resume_base_test)."""
    from core.training.ops.sensenova_ops import _own_save_format_remedy

    assert _own_save_format_remedy(None) == ""
    assert _own_save_format_remedy({}) == ""
    text = _own_save_format_remedy({
        "sensenova_save_format": "mixed",
        "sensenova_trained_branch": "gen",
    })
    assert "sensenova_full_finetune_save_format='mixed'" in text
    assert "only 'int8'" in text
    assert "accept_resume_shaped_base" in text
    # The degeneracy both-halves + mixed produces is reported as written, not
    # as requested.
    degenerate = _own_save_format_remedy({
        "sensenova_save_format": "bf16",
        "sensenova_save_format_requested": "mixed",
    })
    assert "requested 'mixed', written as 'bf16'" in degenerate
