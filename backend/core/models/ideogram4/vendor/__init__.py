"""Vendored Ideogram 4 model code (Apache-2.0).

Independently maintained in SushiUI; no runtime dependency on the upstream
``ideogram4`` package or the unreleased diffusers Ideogram4 classes.
"""

from .transformer import (
    IMAGE_POSITION_OFFSET,
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
    SEQUENCE_PADDING_INDICATOR,
    Ideogram4Transformer2DModel,
)
from .fp8_linear import (
    FP8_TEXT_ENCODER_CONFIG_FLAG,
    Fp8Linear,
    is_bnb4bit_state_dict,
    is_fp8_state_dict,
    load_bnb4bit_state_dict,
    load_fp8_state_dict,
    swap_linears_to_bnb4bit,
    swap_linears_to_fp8,
)
from .text_encoder import load_ideogram4_text_encoder

__all__ = [
    "Ideogram4Transformer2DModel",
    "IMAGE_POSITION_OFFSET",
    "LLM_TOKEN_INDICATOR",
    "OUTPUT_IMAGE_INDICATOR",
    "SEQUENCE_PADDING_INDICATOR",
    "Fp8Linear",
    "FP8_TEXT_ENCODER_CONFIG_FLAG",
    "is_fp8_state_dict",
    "load_fp8_state_dict",
    "swap_linears_to_fp8",
    "is_bnb4bit_state_dict",
    "load_bnb4bit_state_dict",
    "swap_linears_to_bnb4bit",
    "load_ideogram4_text_encoder",
]
