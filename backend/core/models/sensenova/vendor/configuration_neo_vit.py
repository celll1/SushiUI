# SushiUI vendored copy — UNMODIFIED.
#
# Source: https://github.com/OpenSenseNova/SenseNova-U1, branch `feat/u1.5`
#         (commit a1ce053d25835e0785a0869ca1c97e717212ef64), file
#         `src/sensenova_u1/models/neo_unify/configuration_neo_vit.py`, retrieved 2026-08-21.
# Upstream license: Apache-2.0 (see `backend/core/models/sensenova/vendor/__init__.py`).

import os
from typing import Union

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class NEOVisionConfig(PretrainedConfig):

    model_type = 'neo_vision'

    def __init__(
            self,
            num_channels=3,
            patch_size=16,
            hidden_size=1024,
            llm_hidden_size=2048,
            downsample_ratio=0.5,
            rope_theta_vision=10000.0,
            max_position_embeddings_vision=10000,
            min_pixels=65536,
            max_pixels=4194304,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.llm_hidden_size = llm_hidden_size,
        self.downsample_ratio = downsample_ratio,
        self.rope_theta_vision = rope_theta_vision
        self.max_position_embeddings_vision = max_position_embeddings_vision
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> 'PretrainedConfig':
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if 'vision_config' in config_dict:
            config_dict = config_dict['vision_config']

        if 'model_type' in config_dict and hasattr(cls, 'model_type') and config_dict['model_type'] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f'{cls.model_type}. This is not supported for all configurations of models and can yield errors.'
            )

        return cls.from_dict(config_dict, **kwargs)