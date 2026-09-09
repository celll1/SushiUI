"""A GradScaler run with half trainable parameters died at step 1.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/grad_scaler_half_params_test.py -v

THE DEFECT
----------
``use_grad_scaler`` is on for every fp16 mixed-precision run
(``base_trainer.__init__``). torch's GradScaler requires FP32 master
parameters, so a half TRAINABLE parameter cannot complete the first
``unscale_()`` -- measured here against the installed torch, both dtypes.

The refusal that already named this constraint is gated on ``is_full_finetune``,
so it only ever covered base weights. Two trainable modules walked past it:

* the REPA projector, built at ``training_dtype`` (fixed in 6d9c0895 by making
  the projector's parameters FP32; this file pins that it stays fixed), and
* ControlNet, created at ``unet.dtype`` == ``weight_dtype``
  (``controlnet_sd15_adapter:104`` / ``controlnet_sdxl_adapter:99``, LLLite at
  ``:146`` / ``:135``), which ``ControlNetHandsTheOptimizerHalfParametersTest``
  reproduces through the real adapter.

THE FIX
-------
``base_trainer.refuse_half_trainable_params_under_grad_scaler``, called from
``setup_optimizer`` once every group is assembled: one check over the trainable
set instead of one check per module that supplies it.

CPU-only except the BF16 mechanism probe, which needs CUDA to reach the kernel
that has no BF16 overload. No checkpoint and no diffusion model: the ControlNet
test builds a tiny UNet2DConditionModel.
"""

from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn

_BACKEND = str(Path(__file__).resolve().parents[1])
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from core.training.base_trainer import (  # noqa: E402
    BaseTrainer,
    refuse_half_trainable_params_under_grad_scaler,
)

LR = 1e-4
INIT_SCALE = 2 ** 20  # base_trainer's GradScaler init_scale


def _groups(*specs):
    """[(name, dtype, requires_grad), ...] -> optimizer param groups."""
    out = []
    for name, dtype, requires_grad in specs:
        param = nn.Parameter(torch.zeros(4, dtype=dtype),
                             requires_grad=requires_grad)
        out.append({"params": [param], "lr": LR, "name": name,
                    "component": name})
    return out


class _Scalered:
    """The one attribute the refusal reads."""

    def __init__(self, enabled=True):
        self.use_grad_scaler = enabled


class TorchStillRejectsHalfGradientsTest(unittest.TestCase):
    """The premise. If torch ever accepts these, the refusal is re-openable."""

    def _unscale(self, dtype, device):
        param = nn.Parameter(torch.randn(4, dtype=dtype, device=device))
        optimizer = torch.optim.SGD([param], lr=LR)
        scaler = torch.amp.GradScaler(device, init_scale=INIT_SCALE)
        scaler.scale((param.float() ** 2).sum()).backward()
        scaler.unscale_(optimizer)

    def test_fp16_gradients_are_refused_by_unscale(self):
        with self.assertRaises(ValueError) as caught:
            self._unscale(torch.float16, "cpu")
        self.assertIn("Attempting to unscale FP16 gradients", str(caught.exception))

    @unittest.skipUnless(torch.cuda.is_available(), "the BF16 gap is CUDA-side")
    def test_bf16_gradients_have_no_cuda_unscale_kernel(self):
        with self.assertRaises(NotImplementedError) as caught:
            self._unscale(torch.bfloat16, "cuda")
        self.assertIn("not implemented for 'BFloat16'", str(caught.exception))

    def test_fp32_gradients_unscale(self):
        self._unscale(torch.float32, "cpu")


class RefusalFiresTest(unittest.TestCase):
    def test_a_single_fp16_trainable_parameter_is_enough(self):
        with self.assertRaises(ValueError) as caught:
            refuse_half_trainable_params_under_grad_scaler(
                _Scalered(), _groups(("controlnet", torch.float16, True)))
        message = str(caught.exception)
        self.assertIn("controlnet (float16: 1 tensors)", message)

    def test_bf16_is_named_too(self):
        with self.assertRaises(ValueError) as caught:
            refuse_half_trainable_params_under_grad_scaler(
                _Scalered(), _groups(("controlnet", torch.bfloat16, True)))
        self.assertIn("controlnet (bfloat16: 1 tensors)", str(caught.exception))

    def test_the_message_names_every_offending_group_and_a_remedy(self):
        with self.assertRaises(ValueError) as caught:
            refuse_half_trainable_params_under_grad_scaler(
                _Scalered(),
                _groups(("controlnet", torch.float16, True),
                        ("unet", torch.float32, True),
                        ("repa_projector", torch.float16, True)))
        message = str(caught.exception)
        self.assertIn("controlnet", message)
        self.assertIn("repa_projector", message)
        # The FP32 group is not accused.
        self.assertNotIn("unet (", message)
        for remedy in ("training_dtype=bf16", "mixed_precision=false",
                       "lora_dtype", "weight_dtype"):
            self.assertIn(remedy, message)

    def test_an_unnamed_group_is_identified_by_index(self):
        with self.assertRaises(ValueError) as caught:
            refuse_half_trainable_params_under_grad_scaler(
                _Scalered(),
                [{"params": [nn.Parameter(torch.zeros(4, dtype=torch.float16))],
                  "lr": LR}])
        self.assertIn("group 0", str(caught.exception))


class RefusalStaysSilentTest(unittest.TestCase):
    """Every configuration that runs today still runs."""

    def test_bf16_run_has_no_scaler(self):
        # use_grad_scaler is False for bf16 (base_trainer.__init__), which is
        # the only reason a bf16 run may hold bf16 trainable params.
        refuse_half_trainable_params_under_grad_scaler(
            _Scalered(enabled=False), _groups(("unet", torch.bfloat16, True)))

    def test_fp32_trainable_params_under_the_scaler(self):
        refuse_half_trainable_params_under_grad_scaler(
            _Scalered(), _groups(("lora_unet", torch.float32, True)))

    def test_mixed_precision_off_leaves_the_scaler_disabled(self):
        refuse_half_trainable_params_under_grad_scaler(
            _Scalered(enabled=False), _groups(("controlnet", torch.float16, True)))

    def test_a_frozen_half_parameter_in_a_group_is_not_an_offender(self):
        # unscale_() only visits params that HAVE a gradient.
        refuse_half_trainable_params_under_grad_scaler(
            _Scalered(), _groups(("frozen", torch.float16, False)))

    def test_no_groups(self):
        refuse_half_trainable_params_under_grad_scaler(_Scalered(), [])


class _RepaTrainer:
    unet_lr = LR
    repa_enable = True
    repa_proj_lr_factor = 1.0
    log_prefix = "[RepaTrainer]"

    def __init__(self, dtype):
        from core.training.repa import RepaProjector
        self.repa_projector = RepaProjector(8, 16, hidden=8).to(dtype=dtype)


class RepaProjectorStaysAllowedTest(unittest.TestCase):
    """6d9c0895 made the projector FP32; an fp16 REPA run must still start."""

    def _projector_groups(self, dtype):
        from core.training.repa import projector_param_groups
        trainer = _RepaTrainer(dtype)
        with contextlib.redirect_stdout(io.StringIO()):
            return projector_param_groups(trainer, label="Test")

    def test_the_shipped_projector_dtype_is_fp32(self):
        from core.training.repa import PROJECTOR_PARAM_DTYPE
        self.assertIs(PROJECTOR_PARAM_DTYPE, torch.float32)

    def test_an_fp16_run_with_repa_is_not_refused(self):
        from core.training.repa import PROJECTOR_PARAM_DTYPE
        refuse_half_trainable_params_under_grad_scaler(
            _Scalered(), self._projector_groups(PROJECTOR_PARAM_DTYPE))

    def test_the_pre_fix_projector_would_have_been_caught(self):
        with self.assertRaises(ValueError) as caught:
            refuse_half_trainable_params_under_grad_scaler(
                _Scalered(), self._projector_groups(torch.float16))
        self.assertIn("repa_projector", str(caught.exception))


class _StubTrainer:
    """The smallest object ``BaseTrainer.setup_optimizer`` can run against.

    Copied from ``fused_path_grad_scaler_test``, plus a param dtype knob.
    """

    setup_optimizer = BaseTrainer.setup_optimizer
    _report_effective_component_lrs = BaseTrainer._report_effective_component_lrs
    _record_configured_group_lrs = BaseTrainer._record_configured_group_lrs
    _name_configured_groups = BaseTrainer._name_configured_groups
    _build_component_lr_list = BaseTrainer._build_component_lr_list
    _resolved_optimizer_hyperparameters = BaseTrainer._resolved_optimizer_hyperparameters
    _ringbuffer_optimizer_kwargs = BaseTrainer._ringbuffer_optimizer_kwargs
    _announce_host_state_budget = BaseTrainer._announce_host_state_budget
    _assert_ringbuffer_state_host_resident = (
        BaseTrainer._assert_ringbuffer_state_host_resident)
    _RINGBUFFER_HOST_STATE_BYTES_PER_PARAM = (
        BaseTrainer._RINGBUFFER_HOST_STATE_BYTES_PER_PARAM)
    _setup_fused_backward_pass = BaseTrainer._setup_fused_backward_pass
    _setup_fused_optimizer_groups = BaseTrainer._setup_fused_optimizer_groups
    _fused_backward_target_module = BaseTrainer._fused_backward_target_module
    _attach_stochastic_rounding = BaseTrainer._attach_stochastic_rounding
    _RINGBUFFER_ONLY_OPTIONS = BaseTrainer._RINGBUFFER_ONLY_OPTIONS
    _NATIVE_STOCHASTIC_ROUNDING_OPTIMIZERS = BaseTrainer._NATIVE_STOCHASTIC_ROUNDING_OPTIMIZERS
    _BLOCK_SWAP_UNSUPPORTED_OPTIMIZERS = BaseTrainer._BLOCK_SWAP_UNSUPPORTED_OPTIMIZERS

    def __init__(self, param_dtype=torch.float32, **overrides: Any):
        self.log_prefix = "[StubTrainer]"
        self.learning_rate = LR
        self.weight_dtype = torch.float16
        self.blocks_to_swap = 0
        self.num_optimizer_groups = 0
        self.use_ema = False
        self.use_grad_scaler = True
        self.grad_scaler = None
        self.config: Dict[str, Any] = {}
        self.optimizer_cautious = False
        self.optimizer_beta1 = None
        self.optimizer_beta2 = None
        self.optimizer_epsilon = None
        self.optimizer_weight_decay = None
        self.optimizer_schedule_free = False
        self.optimizer_warmup_steps = 0
        self.optimizer_schedule_free_r = 0.0
        self.optimizer_schedule_free_weight_lr_power = 2.0
        self.optimizer_use_radam = False
        self.optimizer_stochastic_rounding = False
        for key, value in overrides.items():
            setattr(self, key, value)
        self.param = nn.Parameter(torch.zeros(256, dtype=param_dtype))
        self.unet = None

    def setup_trainable_parameters(self):
        return [{"params": [self.param], "lr": self.learning_rate,
                 "name": "controlnet", "component": "controlnet"}]

    def _setup_ema(self):
        pass


class SetupOptimizerIsWhereItFiresTest(unittest.TestCase):
    """Wired in, not merely importable."""

    def _setup(self, **overrides):
        trainer = _StubTrainer(**overrides)
        with contextlib.redirect_stdout(io.StringIO()):
            trainer.setup_optimizer(optimizer_type="adamw", total_steps=10)
        return trainer

    def test_half_trainable_params_never_reach_the_optimizer(self):
        with self.assertRaises(ValueError) as caught:
            self._setup(param_dtype=torch.float16)
        self.assertIn("controlnet", str(caught.exception))

    def test_the_fp32_run_still_builds_its_optimizer(self):
        trainer = self._setup(param_dtype=torch.float32)
        self.assertEqual(len(trainer.optimizer.param_groups), 1)

    def test_a_bf16_run_still_builds_its_optimizer(self):
        trainer = self._setup(param_dtype=torch.bfloat16, use_grad_scaler=False,
                              weight_dtype=torch.bfloat16)
        self.assertEqual(len(trainer.optimizer.param_groups), 1)


def _tiny_unet(dtype):
    from diffusers import UNet2DConditionModel
    return UNet2DConditionModel(
        sample_size=8, in_channels=4, out_channels=4,
        layers_per_block=1, block_out_channels=(32, 64),
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim=32, attention_head_dim=8, norm_num_groups=32,
    ).to(dtype=dtype)


class _ControlNetStubTrainer:
    unet_lr = LR

    def __init__(self, unet):
        self.unet = unet


class ControlNetHandsTheOptimizerHalfParametersTest(unittest.TestCase):
    """The hole this refusal was added for, through the real adapter."""

    def _controlnet_groups(self, weight_dtype):
        from core.training.adapters.controlnet_sd15_adapter import ControlNetSD15Adapter
        trainer = _ControlNetStubTrainer(_tiny_unet(weight_dtype))
        adapter = ControlNetSD15Adapter(trainer, "standard", 3)
        with contextlib.redirect_stdout(io.StringIO()):
            controlnet = adapter.create_controlnet(init_from_unet=True,
                                                   pretrained_path=None)
            return adapter.setup_trainable_parameters(controlnet)

    def test_the_controlnet_takes_the_unets_dtype(self):
        for weight_dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(weight_dtype=weight_dtype):
                groups = self._controlnet_groups(weight_dtype)
                self.assertEqual(
                    {p.dtype for g in groups for p in g["params"]}, {weight_dtype})

    def test_an_fp16_controlnet_run_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            refuse_half_trainable_params_under_grad_scaler(
                _Scalered(), self._controlnet_groups(torch.float16))
        self.assertIn("controlnet", str(caught.exception))

    def test_an_fp32_controlnet_run_is_not(self):
        refuse_half_trainable_params_under_grad_scaler(
            _Scalered(), self._controlnet_groups(torch.float32))


if __name__ == "__main__":
    unittest.main()
