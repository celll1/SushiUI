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
    describe_gemm_path,
    disable_scaled_mm,
    get_scaled_mm_state,
    set_scaled_mm_enabled,
    is_bnb4bit_state_dict,
    is_fp8_state_dict,
    load_bnb4bit_state_dict,
    load_fp8_state_dict,
    swap_linears_to_bnb4bit,
    swap_linears_to_fp8,
)
from .int8_linear import (
    Int8Linear,
    describe_gemm_path as describe_int8_gemm_path,
    disable_int8_mm,
    get_int8_mm_state,
    is_int8_state_dict,
    set_int8_mm_enabled,
    swap_linears_to_int8,
)
from .text_encoder import load_ideogram4_text_encoder

__all__ = [
    "Int8Linear",
    "describe_int8_gemm_path",
    "disable_int8_mm",
    "get_int8_mm_state",
    "is_int8_state_dict",
    "set_int8_mm_enabled",
    "swap_linears_to_int8",
    "Ideogram4Transformer2DModel",
    "IMAGE_POSITION_OFFSET",
    "LLM_TOKEN_INDICATOR",
    "OUTPUT_IMAGE_INDICATOR",
    "SEQUENCE_PADDING_INDICATOR",
    "Fp8Linear",
    "FP8_TEXT_ENCODER_CONFIG_FLAG",
    "describe_gemm_path",
    "disable_scaled_mm",
    "get_scaled_mm_state",
    "set_scaled_mm_enabled",
    "is_fp8_state_dict",
    "load_fp8_state_dict",
    "swap_linears_to_fp8",
    "is_bnb4bit_state_dict",
    "load_bnb4bit_state_dict",
    "swap_linears_to_bnb4bit",
    "load_ideogram4_text_encoder",
]
