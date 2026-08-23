import sys
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.param_defaults import TIMESTEP_SAMPLING_DEFAULTS_BY_ARCH
from core.model_loader import ModelLoader
from core.training.arch import resolve_arch_name
from core.training.base_trainer import BaseTrainer
from core.training.full_parameter_trainer import FullParameterTrainer
from core.training.lora_trainer import LoRATrainer
from core.training.relora_trainer import ReLoRATrainer
from core.training.train_runner import (
    _apply_sensenova_training_contract,
    _is_bf16_native_base_model,
    _prepare_training_process_config,
)


class _ConcreteTrainer(BaseTrainer):
    def setup_trainable_parameters(self):
        return []

    def save_checkpoint(self, *args, **kwargs):
        raise NotImplementedError

    def load_checkpoint(self, *args, **kwargs):
        raise NotImplementedError


def test_sensenova_prediction_and_timestep_defaults():
    prediction = ModelLoader.detect_prediction_config("not-a-checkpoint", "sensenova")
    assert prediction == {
        "noise_process": "flow",
        "prediction_target": "velocity",
        "source": "inferred",
    }
    default = TIMESTEP_SAMPLING_DEFAULTS_BY_ARCH["sensenova"]
    assert default["distribution"] == "logit_normal"
    assert default["mean"] == -0.8 and default["std"] == 0.8


def test_runner_enforces_b1_onthefly_and_disables_sampling():
    train = {
        "batch_size": "1",
        "blocks_to_swap": "0",
        "use_reference_images": "false",
        "text_encoding_mode": "pre_encoded_cache",
    }
    process = {"sample": {"sample_every": 100}}
    with patch.object(ModelLoader, "detect_model_type", return_value="sensenova"):
        assert _apply_sensenova_training_contract("model", "lora", train, process)
    assert train["text_encoding_mode"] == "onthefly_gpu"
    assert train["latent_encoding_mode"] == "onthefly_gpu"
    assert train["batch_size"] == 1
    assert train["blocks_to_swap"] == 0
    assert train["use_reference_images"] is False
    assert process["sample"]["sample_every"] == 0
    assert _is_bf16_native_base_model("models/sensenova")


@pytest.mark.parametrize(
    "train,network,message",
    [
        ({"batch_size": 2}, "lora", "batch_size=1"),
        ({"batch_size": 1, "blocks_to_swap": 1}, "lora", "blocks_to_swap"),
        ({"batch_size": 1, "blocks_to_swap": -1}, "lora", "blocks_to_swap"),
        ({"batch_size": 1, "use_reference_images": True}, "lora", "reference-image"),
        ({"batch_size": 1}, "full_finetune", "lora"),
        ({"batch_size": 1}, "relora", "lora"),
    ],
)
def test_runner_rejects_outside_initial_contract(train, network, message):
    with patch.object(ModelLoader, "detect_model_type", return_value="sensenova"):
        with pytest.raises(ValueError, match=message):
            _apply_sensenova_training_contract("model", network, train, {"sample": {}})


@pytest.mark.parametrize(
    "train,message",
    [
        ({"batch_size": True}, "batch_size"),
        ({"batch_size": 1.5}, "batch_size"),
        ({"batch_size": "1.0"}, "batch_size"),
        ({"batch_size": "malformed"}, "batch_size"),
        ({"batch_size": 1, "blocks_to_swap": False}, "blocks_to_swap"),
        ({"batch_size": 1, "blocks_to_swap": 0.5}, "blocks_to_swap"),
        ({"batch_size": 1, "blocks_to_swap": "0.5"}, "blocks_to_swap"),
        ({"batch_size": 1, "blocks_to_swap": "malformed"}, "blocks_to_swap"),
    ],
)
def test_runner_strictly_rejects_non_integer_contract_fields(train, message):
    with patch.object(ModelLoader, "detect_model_type", return_value="sensenova"):
        with pytest.raises(ValueError, match=message):
            _apply_sensenova_training_contract("checkpoint", "lora", train, {})


@pytest.mark.parametrize("value", [True, 1, "true", "1"])
def test_runner_strict_reference_true_is_refused(value):
    train = {"batch_size": 1, "use_reference_images": value}
    with patch.object(ModelLoader, "detect_model_type", return_value="sensenova"):
        with pytest.raises(ValueError, match="reference-image"):
            _apply_sensenova_training_contract("checkpoint", "lora", train, {})


@pytest.mark.parametrize("value", [False, 0, "false", "0"])
def test_runner_strict_reference_false_is_normalized(value):
    train = {"batch_size": 1, "use_reference_images": value}
    with patch.object(ModelLoader, "detect_model_type", return_value="sensenova"):
        assert _apply_sensenova_training_contract("checkpoint", "lora", train, {})
    assert train["use_reference_images"] is False


@pytest.mark.parametrize("value", ["maybe", 0.5, 2, [], None])
def test_runner_strict_reference_malformed_is_rejected(value):
    train = {"batch_size": 1, "use_reference_images": value}
    with patch.object(ModelLoader, "detect_model_type", return_value="sensenova"):
        with pytest.raises(ValueError, match="use_reference_images"):
            _apply_sensenova_training_contract("checkpoint", "lora", train, {})


def test_vae_decoder_is_exempt_from_the_sensenova_diffusion_contract():
    train = {"batch_size": 1.5, "blocks_to_swap": 0.5}
    with patch.object(ModelLoader, "detect_model_type") as detect:
        assert not _apply_sensenova_training_contract(
            "checkpoint", "vae_decoder", train, {}
        )
    detect.assert_not_called()
    assert train == {"batch_size": 1.5, "blocks_to_swap": 0.5}


def test_process_preflight_fails_before_dataset_discovery_with_neutral_name():
    class _Process(dict):
        def get(self, key, default=None):
            if key == "datasets":
                raise AssertionError("dataset discovery ran before preflight")
            return super().get(key, default)

    process = _Process(
        train={"batch_size": 2}, network={"type": "lora"}, sample={}
    )
    config = {"config": {"process": [process]}}
    with patch.object(ModelLoader, "detect_model_type", return_value="sensenova"):
        with pytest.raises(ValueError, match="batch_size=1"):
            _prepare_training_process_config(config, "checkpoint")


def test_sensenova_name_does_not_override_successful_non_sensenova_detection():
    train = {"batch_size": 2}
    with patch.object(ModelLoader, "detect_model_type", return_value="sdxl"):
        assert not _apply_sensenova_training_contract(
            "not-sensenova-model", "lora", train, {}
        )
    assert train == {"batch_size": 2}


def test_registry_and_lora_adapter_selection():
    trainer = LoRATrainer.__new__(LoRATrainer)
    for name in (
        "zimage", "flux2", "lens", "ideogram4", "minit2i", "krea2",
        "anima", "ltx2", "minimax_h3", "acestep", "sdxl",
    ):
        setattr(trainer, f"is_{name}", False)
    trainer.is_sensenova = True
    trainer.lora_rank = trainer.lora_alpha = 1
    trainer.lora_dtype = torch.float32
    trainer.log_prefix = "[test]"
    trainer._create_adapter()

    from core.training.adapters import SenseNovaLoRAAdapter

    assert resolve_arch_name(trainer) == "sensenova"
    assert isinstance(trainer.adapter, SenseNovaLoRAAdapter)


def test_base_dispatch_loads_sensenova_ops():
    trainer = _ConcreteTrainer.__new__(_ConcreteTrainer)
    trainer.model_path = "model"
    trainer.blocks_to_swap = 0
    with patch.object(ModelLoader, "detect_model_type", return_value="sensenova"), patch(
        "core.training.ops.sensenova_ops.load_components"
    ) as load:
        trainer._load_model_components()
    assert trainer.is_sensenova is True
    load.assert_called_once_with(trainer)


@pytest.mark.parametrize("blocks", [1, -1, 0.5, "0"])
def test_both_base_load_paths_reject_nonzero_sensenova_block_swap(blocks):
    trainer = _ConcreteTrainer.__new__(_ConcreteTrainer)
    trainer.blocks_to_swap = blocks
    trainer.model_path = "model"
    with patch.object(ModelLoader, "detect_model_type", return_value="sensenova"), patch(
        "core.training.ops.sensenova_ops.load_components"
    ) as load:
        with pytest.raises(ValueError, match="blocks_to_swap"):
            trainer._load_model_components()
        load.assert_not_called()

    trainer = _ConcreteTrainer.__new__(_ConcreteTrainer)
    trainer.blocks_to_swap = blocks
    with patch.object(ModelLoader, "detect_model_type", return_value="sensenova"), patch(
        "core.training.ops.sensenova_ops.load_components"
    ) as load:
        with pytest.raises(ValueError, match="blocks_to_swap"):
            trainer._load_checkpoint_as_base("checkpoint")
        load.assert_not_called()


@pytest.mark.parametrize("blocks", [1, -1, 0.5, "0"])
def test_sensenova_ops_reject_nonzero_block_swap_before_loading(blocks):
    trainer = SimpleNamespace(blocks_to_swap=blocks)
    with pytest.raises(ValueError, match="blocks_to_swap"):
        from core.training.ops.sensenova_ops import load_components
        load_components(trainer)


def test_encode_caption_returns_prefix_without_tensor_cache_payload():
    prefix = object()
    trainer = _ConcreteTrainer.__new__(_ConcreteTrainer)
    trainer.is_zimage = False
    trainer.is_sensenova = True
    trainer.arch = SimpleNamespace(
        encode_prompt=lambda owner, caption, requires_grad=False: prefix
    )
    assert trainer.encode_caption("caption") == (prefix, None)


def test_execute_forward_backward_uses_dedicated_prefix_field():
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    prefix = object()

    class _Arch:
        def train_step(self, trainer, ctx):
            assert ctx.sensenova_prefix is prefix
            assert ctx.text_embeddings is None
            loss = parameter.square()
            return loss, float(loss.detach()), 0.0

    trainer = _ConcreteTrainer.__new__(_ConcreteTrainer)
    trainer.is_sensenova = True
    trainer.is_zimage = False
    trainer.arch = _Arch()
    trainer.debug_vram = False
    trainer.use_grad_scaler = False
    trainer._grad_accum_steps = 1
    result = trainer._execute_forward_backward(
        mnt_latents=torch.zeros(1, 3, 32, 32),
        mnt_text_embeddings=None,
        mnt_attention_mask=None,
        mnt_pooled_embeddings=None,
        timesteps=torch.tensor([0.5]),
        debug_save_path=None,
        batch_captions=None,
        batch_reference_paths=None,
        alphas_cumprod_cached=None,
        use_condition_images=False,
        condition_images_batch=None,
        reference_latents_nested=None,
        sensenova_prefix=prefix,
    )
    assert result == (4.0, 4.0, 0.0)
    assert parameter.grad.item() == 4.0


def test_real_loop_helpers_keep_one_poison_prefix_identity_across_two_mnt_steps():
    prefix = object()
    collected = BaseTrainer._collate_sensenova_b1_prefix([prefix])
    assert collected is prefix
    with pytest.raises(ValueError, match="exactly one"):
        BaseTrainer._collate_sensenova_b1_prefix([])
    with pytest.raises(ValueError, match="exactly one"):
        BaseTrainer._collate_sensenova_b1_prefix([prefix, object()])

    first = BaseTrainer._sensenova_mnt_conditioning(collected)
    second = BaseTrainer._sensenova_mnt_conditioning(collected)
    assert first[:3] == (None, None, None)
    assert second[:3] == (None, None, None)
    assert first[3] is prefix
    assert second[3] is prefix


def test_sensenova_prefix_reaches_minimal_batch_oom_path():
    prefix = object()
    seen = []

    trainer = _ConcreteTrainer.__new__(_ConcreteTrainer)
    trainer.activation_dispatch_enable = False
    trainer.activation_dispatcher = None
    trainer.log_prefix = "[test]"
    trainer._batch_was_unfittable = False
    trainer.optimizer = None

    def _oom(self, **kwargs):
        seen.append(kwargs["sensenova_prefix"])
        raise RuntimeError("CUDA out of memory")

    trainer._execute_forward_backward = MethodType(_oom, trainer)
    result = trainer._forward_backward_with_oom_recovery(
        mnt_latents=torch.zeros(1, 3, 32, 32),
        mnt_text_embeddings=None,
        mnt_attention_mask=None,
        mnt_pooled_embeddings=None,
        timesteps=torch.tensor([0.5]),
        debug_save_path=None,
        batch_captions=None,
        batch_reference_paths=None,
        alphas_cumprod_cached=None,
        use_condition_images=False,
        condition_images_batch=None,
        reference_latents_nested=None,
        sensenova_prefix=prefix,
    )
    assert result == (0.0, 0.0, 0.0, True)
    assert seen == [prefix]


def test_full_finetune_refuses_before_loading():
    with patch.object(ModelLoader, "detect_model_type", return_value="sensenova"):
        with pytest.raises(ValueError, match="Use training_method='lora'"):
            FullParameterTrainer._refuse_unsupported_full_finetune("model")


def test_relora_refuses_before_loading_with_the_capability_table_reason():
    from api.arch_capabilities import TRAINING_UNSUPPORTED

    reason = TRAINING_UNSUPPORTED["sensenova"]["relora"]
    with patch.object(ModelLoader, "detect_model_type", return_value="sensenova"):
        with pytest.raises(ValueError, match="ReLoRA is not supported") as exc:
            ReLoRATrainer._refuse_unsupported_relora("model")
    assert reason in str(exc.value)
    assert "controlnet" in TRAINING_UNSUPPORTED["sensenova"]
    assert "vae_decoder" not in TRAINING_UNSUPPORTED["sensenova"]


def test_train_method_b1_guard_precedes_dataset_setup():
    trainer = _ConcreteTrainer.__new__(_ConcreteTrainer)
    trainer.is_sensenova = True
    trainer.blocks_to_swap = 0
    with pytest.raises(ValueError, match="batch_size=1"):
        trainer.train(datasets=[], batch_size=2)


@pytest.mark.parametrize("blocks", [1, -1, 0.5, "0"])
def test_train_method_rejects_nonzero_sensenova_block_swap(blocks):
    trainer = _ConcreteTrainer.__new__(_ConcreteTrainer)
    trainer.is_sensenova = True
    trainer.blocks_to_swap = blocks
    with pytest.raises(ValueError, match="blocks_to_swap"):
        trainer.train(datasets=[], batch_size=1)
