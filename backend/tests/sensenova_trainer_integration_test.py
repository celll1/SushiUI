import sys
import asyncio
import inspect
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import yaml

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


def test_runner_enforces_b1_onthefly_and_keeps_sampling():
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
    # Training-time sampling is implemented; the contract must leave it alone.
    assert process["sample"]["sample_every"] == 100
    assert _is_bf16_native_base_model("models/sensenova")


@pytest.mark.parametrize(
    "extra,message",
    [
        ({"num_optimizer_groups": 2}, "num_optimizer_groups"),
        ({"block_swap_h2d_only": True}, "block_swap_h2d_only"),
    ],
)
def test_phase_eviction_rejects_block_swap_only_optimizer_modes(extra, message):
    train = {
        "batch_size": 1,
        "blocks_to_swap": 0,
        "sensenova_mot_phase_eviction": True,
        **extra,
    }
    with patch.object(ModelLoader, "detect_model_type", return_value="sensenova"):
        with pytest.raises(ValueError, match=message):
            _apply_sensenova_training_contract("model", "lora", train, {})


def test_phase_eviction_api_yaml_openapi_and_frontend_parity():
    from api.param_defaults import TRAINING_DEFAULTS
    from api.routes import TrainingRunCreateRequest, get_training_defaults
    from core.training.training_config import TrainingConfigGenerator

    assert TRAINING_DEFAULTS["sensenova_mot_phase_eviction"] is False
    assert asyncio.run(get_training_defaults())["sensenova_mot_phase_eviction"] is False
    request = TrainingRunCreateRequest(
        training_method="lora", base_model_path="models/sensenova"
    )
    assert request.sensenova_mot_phase_eviction is False
    config = yaml.safe_load(
        TrainingConfigGenerator.generate_lora_config(
            {**request.model_dump(), "total_steps": 1, "epochs": None},
            run_name="phase-eviction",
            base_model_path=request.base_model_path,
            output_dir="output",
        )
    )
    train = config["config"]["process"][0]["train"]
    assert train["sensenova_mot_phase_eviction"] is False

    root = Path(__file__).resolve().parents[2]
    spec = yaml.safe_load((root / "openapi.yaml").read_text(encoding="utf-8"))
    prop = spec["components"]["schemas"]["TrainingRunCreateRequest"]["properties"][
        "sensenova_mot_phase_eviction"
    ]
    assert prop["default"] is False
    api_source = (root / "frontend/src/utils/api.ts").read_text(encoding="utf-8")
    form_source = (
        root / "frontend/src/components/training/TrainingConfig.tsx"
    ).read_text(encoding="utf-8")
    assert "sensenova_mot_phase_eviction?: boolean" in api_source
    assert 'updateParam("sensenova_mot_phase_eviction"' in form_source


@pytest.mark.parametrize(
    "train,network,message",
    [
        ({"batch_size": 1, "blocks_to_swap": 1}, "lora", "blocks_to_swap"),
        ({"batch_size": 1, "blocks_to_swap": -1}, "lora", "blocks_to_swap"),
        # full_finetune is ACCEPTED now (U-2-2 step 3); relora and controlnet
        # are not, and are refused by name rather than by "not lora".
        ({"batch_size": 1}, "relora", "not 'relora'"),
        ({"batch_size": 1}, "controlnet", "not 'controlnet'"),
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
def test_runner_strict_reference_true_is_accepted_and_normalized(value):
    """Phase 3: the runner arms reference conditioning instead of refusing it."""
    train = {"batch_size": 1, "use_reference_images": value}
    with patch.object(ModelLoader, "detect_model_type", return_value="sensenova"):
        assert _apply_sensenova_training_contract("checkpoint", "lora", train, {})
    assert train["use_reference_images"] is True


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
        train={"batch_size": 1, "blocks_to_swap": 1}, network={"type": "lora"}, sample={}
    )
    config = {"config": {"process": [process]}}
    with patch.object(ModelLoader, "detect_model_type", return_value="sensenova"):
        with pytest.raises(ValueError, match="blocks_to_swap"):
            _prepare_training_process_config(config, "checkpoint")


def test_sensenova_name_does_not_override_successful_non_sensenova_detection():
    train = {"batch_size": 2}
    with patch.object(ModelLoader, "detect_model_type", return_value="sdxl"):
        assert not _apply_sensenova_training_contract(
            "not-sensenova-model", "lora", train, {}
        )
    assert train == {"batch_size": 2}


def _sensenova_checkpoint_dir(tmp_path, t_eps=0.05):
    (tmp_path / "config.json").write_text(
        f'{{"t_eps": {t_eps}}}', encoding="utf-8"
    )
    return str(tmp_path)


def _run_contract_with_timestep_sampling(tmp_path, sampling):
    train = {"batch_size": 1, "blocks_to_swap": 0}
    if sampling is not None:
        train["timestep_sampling"] = sampling
    with patch.object(ModelLoader, "detect_model_type", return_value="sensenova"):
        assert _apply_sensenova_training_contract(
            _sensenova_checkpoint_dir(tmp_path), "lora", train, {}
        )
    return train


@pytest.mark.parametrize(
    "sampling",
    [
        None,
        dict(TIMESTEP_SAMPLING_DEFAULTS_BY_ARCH["sensenova"]),
        {**TIMESTEP_SAMPLING_DEFAULTS_BY_ARCH["sensenova"], "max_timestep": 0.5},
    ],
)
def test_sensenova_safe_timestep_sampling_does_not_warn(tmp_path, capsys, sampling):
    _run_contract_with_timestep_sampling(tmp_path, sampling)
    assert "timestep_sampling departs" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "sampling",
    [
        {"distribution": "uniform", "min_timestep": 0.0, "max_timestep": 1.0},
        {"distribution": "uniform", "min_timestep": 0.0, "max_timestep": 0.9},
        {"distribution": "logit_normal", "mean": 2.0, "std": 1.0,
         "min_timestep": 0.0, "max_timestep": 1.0},
    ],
)
def test_sensenova_clean_side_timestep_sampling_warns(tmp_path, capsys, sampling):
    train = _run_contract_with_timestep_sampling(tmp_path, sampling)
    out = capsys.readouterr().out
    assert "timestep_sampling departs" in out
    assert "mse(x0_pred, x0) / (1-t)^2" in out
    assert "1/t_eps^2 = 400.0" in out and "t_eps=0.05" in out
    assert "logit_normal(mean=-0.8, std=0.8)" in out
    assert train["timestep_sampling"] == sampling


def test_sensenova_timestep_warning_is_qualitative_without_config_t_eps(
    tmp_path, capsys
):
    train = {
        "batch_size": 1,
        "blocks_to_swap": 0,
        "timestep_sampling": {
            "distribution": "uniform", "min_timestep": 0.0, "max_timestep": 1.0
        },
    }
    with patch.object(ModelLoader, "detect_model_type", return_value="sensenova"):
        assert _apply_sensenova_training_contract(
            str(tmp_path / "missing"), "lora", train, {}
        )
    out = capsys.readouterr().out
    assert "clamped only by the model's t_eps at 1/t_eps^2." in out
    assert "E[1/(1-t)^2]" not in out


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


def test_phase_eviction_toggle_off_does_not_install_selector():
    trainer = LoRATrainer.__new__(LoRATrainer)
    trainer.is_sensenova = True
    trainer.sensenova_mot_phase_eviction = False
    with patch(
        "core.training.sensenova_phase_eviction.install_training_phase_eviction"
    ) as install:
        trainer._setup_sensenova_phase_eviction()
    install.assert_not_called()


@pytest.mark.parametrize("fails", [False, True])
def test_lora_train_always_tears_down_phase_eviction(fails):
    calls = []
    trainer = LoRATrainer.__new__(LoRATrainer)
    trainer.log_prefix = "[test]"
    trainer.sensenova_phase_evictor = SimpleNamespace(
        teardown=lambda: calls.append("teardown")
    )
    effect = RuntimeError("train failed") if fails else None
    with patch.object(BaseTrainer, "train", side_effect=effect, return_value="done"):
        if fails:
            with pytest.raises(RuntimeError, match="train failed"):
                trainer.train()
        else:
            assert trainer.train() == "done"
    assert calls == ["teardown"]
    assert trainer.sensenova_phase_evictor is None


@pytest.mark.parametrize("train_fails", [False, True])
def test_teardown_error_does_not_mask_train_result(train_fails, capsys):
    def teardown():
        raise RuntimeError("teardown failed")

    trainer = LoRATrainer.__new__(LoRATrainer)
    trainer.log_prefix = "[test]"
    trainer.sensenova_phase_evictor = SimpleNamespace(teardown=teardown)
    effect = RuntimeError("train failed") if train_fails else None
    with patch.object(BaseTrainer, "train", side_effect=effect, return_value="done"):
        if train_fails:
            with pytest.raises(RuntimeError, match="train failed"):
                trainer.train()
        else:
            assert trainer.train() == "done"
    assert "SenseNova eviction teardown failed" in capsys.readouterr().out
    assert trainer.sensenova_phase_evictor is None


def test_optimizer_step_checks_generation_residency_first():
    """The residency check must precede the step. It now lives behind a named
    seam (``_assert_sensenova_step_seam_residency``) because the four-phase
    shared window has to choose WHICH half to assert; the ordering guarantee is
    unchanged, so this follows the call site rather than the inlined call."""
    source = inspect.getsource(BaseTrainer.train)
    step_branch = source.split("elif should_step_optimizer:", 1)[1]
    assert step_branch.index(
        "self._assert_sensenova_step_seam_residency("
    ) < step_branch.index("self.optimizer.step()")
    seam = inspect.getsource(BaseTrainer._assert_sensenova_step_seam_residency)
    assert "assert_generation_resident()" in seam
    assert "assert_understanding_resident()" in seam
    assert "assert_generation_resident" not in inspect.getsource(
        LoRATrainer.save_checkpoint
    )


def _sensenova_staging_trainer(evictor):
    trainer = _ConcreteTrainer.__new__(_ConcreteTrainer)
    for name in (
        "zimage", "anima", "lens", "ideogram4", "minit2i", "krea2",
        "ltx2", "acestep", "minimax_h3",
    ):
        setattr(trainer, f"is_{name}", False)
    trainer.is_sensenova = True
    trainer.device = torch.device("cpu")
    calls = []

    class Transformer:
        def to(self, device):
            calls.append(device)

    trainer.transformer_original = Transformer()
    trainer.sensenova_phase_evictor = evictor
    return trainer, calls


@pytest.mark.parametrize("state", ["prefix", "denoise"])
def test_active_phase_eviction_blocks_generic_main_model_staging(state):
    evictor = SimpleNamespace(state=state)
    trainer, calls = _sensenova_staging_trainer(evictor)

    trainer.move_main_model_to_gpu()
    trainer.move_main_model_to_cpu()

    assert calls == []
    assert evictor.state == state
    assert trainer._main_model_module() is trainer.transformer_original


def test_inactive_phase_eviction_keeps_generic_main_model_staging():
    trainer, calls = _sensenova_staging_trainer(None)

    trainer.move_main_model_to_gpu()
    trainer.move_main_model_to_cpu()

    assert calls == [torch.device("cpu"), "cpu"]


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
    seen = {}

    def _encode(owner, caption, requires_grad=False, reference_image_paths=None):
        seen["refs"] = reference_image_paths
        return prefix

    trainer.arch = SimpleNamespace(encode_prompt=_encode)
    assert trainer.encode_caption("caption") == (prefix, None)
    assert seen["refs"] is None
    assert trainer.encode_caption("caption", reference_image_paths=["a.png"]) == (
        prefix,
        None,
    )
    assert seen["refs"] == ["a.png"]


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
    calls = []

    class _Owner:
        train_text_encoder = False

        def encode_caption(self, caption, requires_grad=False,
                           reference_image_paths=None, cfg_null=False):
            calls.append((caption, requires_grad, reference_image_paths, cfg_null))
            return prefix, None

    # One item: the single-prompt encode, unchanged, and its identity is kept.
    collected = BaseTrainer._encode_sensenova_batch_prefix(_Owner(), [("a cat", None, False)])
    assert collected is prefix
    assert calls == [("a cat", False, None, False)]

    owner = SimpleNamespace(train_text_encoder=False)
    conditioning = MethodType(BaseTrainer._sensenova_mnt_conditioning, owner)
    first = conditioning(collected, captions=["a caption"], mnt_index=0)
    second = conditioning(collected, captions=["a caption"], mnt_index=1)
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


def test_full_finetune_is_no_longer_refused_before_loading():
    """U-2-2 step 3: the pre-load capability refusal is gone for this arch.

    ReLoRA's is checked below and must NOT have moved with it.
    """
    with patch.object(ModelLoader, "detect_model_type", return_value="sensenova"):
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


def test_train_method_batch_guard_precedes_dataset_setup():
    """batch_size > 1 is a packed batch of same-resolution images, which only
    the bucket manager guarantees; without bucketing it is refused up front."""
    trainer = _ConcreteTrainer.__new__(_ConcreteTrainer)
    trainer.is_sensenova = True
    trainer.blocks_to_swap = 0
    with pytest.raises(ValueError, match="enable_bucketing"):
        trainer.train(datasets=[], batch_size=2, enable_bucketing=False)


@pytest.mark.parametrize("blocks", [1, -1, 0.5, "0"])
def test_train_method_rejects_nonzero_sensenova_block_swap(blocks):
    trainer = _ConcreteTrainer.__new__(_ConcreteTrainer)
    trainer.is_sensenova = True
    trainer.blocks_to_swap = blocks
    with pytest.raises(ValueError, match="blocks_to_swap"):
        trainer.train(datasets=[], batch_size=1)
