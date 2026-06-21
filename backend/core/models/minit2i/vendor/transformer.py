"""diffusers wrappers for MiniT2I (vendored, MIT).

`MiniT2IMMJiTModel` (ModelMixin + ConfigMixin) wraps `DiffusionModel`/`MMJiT` so
`from_pretrained(<dir>/transformer)` loads the published checkpoints (keys are
`model.net.*`). `MiniT2IFlowMatchScheduler` carries the lognorm timestep schedule.
"""

from __future__ import annotations

import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.schedulers.scheduling_utils import SchedulerMixin

from .mmjit import MMJiTConfig, DiffusionModel


class MiniT2IFlowMatchScheduler(SchedulerMixin, ConfigMixin):
    config_name = "scheduler_config.json"

    @register_to_config
    def __init__(
        self,
        train_t_schedule: str = "lognorm",
        t_lognorm_mu: float = -0.8,
        t_lognorm_sigma: float = 0.8,
        num_inference_steps: int = 100,
    ):
        if train_t_schedule not in {"uniform", "lognorm"}:
            raise ValueError(f"Unsupported train_t_schedule: {train_t_schedule}")

    def sample_train_timesteps(self, batch_size, device, dtype=torch.float32, generator=None):
        """Sample training timesteps t in (0,1) (t=1 data, t=0 noise)."""
        if self.config.train_t_schedule == "uniform":
            return torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
        normal = torch.randn(batch_size, device=device, dtype=torch.float32, generator=generator)
        normal = normal * self.config.t_lognorm_sigma + self.config.t_lognorm_mu
        return torch.sigmoid(normal).clamp(1e-5, 1.0 - 1e-5).to(dtype=dtype)

    def get_inference_timesteps(self, num_inference_steps=None, device=None, dtype=torch.float32):
        steps = int(num_inference_steps or self.config.num_inference_steps)
        return torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=dtype)


class MiniT2IMMJiTModel(ModelMixin, ConfigMixin):
    config_name = "config.json"
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        image_size: int = 512,
        patch_size: int = 16,
        in_channels: int = 3,
        txt_input_size: int = 1024,
        hidden_size: int = 768,
        txt_hidden_size: int = 768,
        cond_vec_size: int = 768,
        depth_double: int = 17,
        txt_preamble_depth: int = 2,
        num_heads: int = 12,
        head_dim: int = 64,
        mlp_ratio: float = 2.6666666666666665,
        pca_channels: int = 128,
        prompt_length: int = 256,
        n_T: int = 100,
        prediction: str = "x",
        sampler: str = "euler",
        cfg_channels: int = 3,
        cfg_interval: tuple = (0.0, 1.0),
        llm: str = "google/flan-t5-large",
    ):
        super().__init__()
        cfg = MMJiTConfig(
            image_size=image_size, patch_size=patch_size, in_channels=in_channels,
            txt_input_size=txt_input_size, hidden_size=hidden_size, txt_hidden_size=txt_hidden_size,
            cond_vec_size=cond_vec_size, depth_double=depth_double, txt_preamble_depth=txt_preamble_depth,
            num_heads=num_heads, head_dim=head_dim, mlp_ratio=mlp_ratio, pca_channels=pca_channels,
            prompt_length=prompt_length, n_T=n_T, prediction=prediction, sampler=sampler,
            cfg_channels=cfg_channels, cfg_interval=tuple(cfg_interval), llm=llm,
        )
        self.model = DiffusionModel(cfg)
        self.gradient_checkpointing = False

    @property
    def mmjit_config(self) -> MMJiTConfig:
        return self.model.cfg

    # net(img, t, context, attn_mask) -> predicted x0 (RGB)
    def forward(self, img, t, context, attn_mask):
        return self.model.net(img, t, context, attn_mask)

    def pred_velocity(self, x, t, text, mask):
        """v = (x0_pred - x)/clamp(1-t,0.05); integrate x += v*dt for sampling."""
        return self.model.pred_velocity(x, t, text, mask)
