"""Key remapping for the ACE-Step 1.5 Oobleck VAE checkpoint.

The local checkpoint (``vae/ace_1.5_vae.safetensors``) is saved in the
original stable-audio-tools ``nn.Sequential``-index layout
(``encoder.layers.0``, ``encoder.layers.1.layers.0.layers.0`` ...), not the
diffusers-renamed ``AutoencoderOobleck`` layout (``encoder.conv1``,
``encoder.block.0.res_unit1.snake1`` ...). This module derives the mapping
directly from the diffusers `AutoencoderOobleck` / `OobleckEncoder` /
`OobleckDecoder` / `OobleckEncoderBlock` / `OobleckDecoderBlock` /
`OobleckResidualUnit` module definitions (see
``diffusers.models.autoencoders.autoencoder_oobleck``) rather than porting an
external conversion script.

Verified 2026-07-13 against the local checkpoint: converting every one of its
365 keys and doing a strict `AutoencoderOobleck.load_state_dict` succeeds with
zero missing/unexpected keys once the `Snake1d` `alpha`/`beta` parameters
(saved flat as shape `(C,)`) are reshaped to the module's `(1, C, 1)`.

Sequential-index layout (both encoder and decoder share the block-internal
shape; only the top-level layer roles differ):

    encoder.layers.0                              -> encoder.conv1
    encoder.layers.{1..5}                         -> encoder.block.{i-1}   (OobleckEncoderBlock)
        block.layers.{0,1,2}                      -> res_unit{1,2,3}       (OobleckResidualUnit)
            res_unit.layers.{0,1,2,3}             -> snake1, conv1, snake2, conv2
        block.layers.3                            -> snake1
        block.layers.4                            -> conv1                 (strided downsample)
    encoder.layers.6                              -> encoder.snake1
    encoder.layers.7                              -> encoder.conv2

    decoder.layers.0                              -> decoder.conv1
    decoder.layers.{1..5}                         -> decoder.block.{i-1}   (OobleckDecoderBlock)
        block.layers.0                            -> snake1
        block.layers.1                            -> conv_t1               (strided upsample)
        block.layers.{2,3,4}                      -> res_unit{1,2,3}       (OobleckResidualUnit, same as above)
    decoder.layers.6                              -> decoder.snake1
    decoder.layers.7                              -> decoder.conv2         (no bias)
"""

from __future__ import annotations

import re
from typing import Dict

import torch

_RES_UNIT_SUBMAP = {0: "snake1", 1: "conv1", 2: "snake2", 3: "conv2"}

_TOP_LEVEL_RE = re.compile(r"^(encoder|decoder)\.layers\.(\d+)\.(.*)$")
_SUB_RE = re.compile(r"^layers\.(\d+)\.(.*)$")


def _convert_res_unit(rest: str) -> str:
    m = _SUB_RE.match(rest)
    if not m:
        raise ValueError(f"unrecognized residual-unit sub-key: {rest!r}")
    k, tail = int(m.group(1)), m.group(2)
    if k not in _RES_UNIT_SUBMAP:
        raise ValueError(f"unexpected residual-unit sub-index {k} in {rest!r}")
    return f"{_RES_UNIT_SUBMAP[k]}.{tail}"


def _convert_encoder_block_key(rest: str) -> str:
    m = _SUB_RE.match(rest)
    if not m:
        raise ValueError(f"unrecognized encoder-block sub-key: {rest!r}")
    j, tail = int(m.group(1)), m.group(2)
    if j in (0, 1, 2):
        res_unit_name = {0: "res_unit1", 1: "res_unit2", 2: "res_unit3"}[j]
        return f"{res_unit_name}.{_convert_res_unit(tail)}"
    elif j == 3:
        return f"snake1.{tail}"
    elif j == 4:
        return f"conv1.{tail}"
    raise ValueError(f"unexpected encoder-block sub-index {j} in {rest!r}")


def _convert_decoder_block_key(rest: str) -> str:
    m = _SUB_RE.match(rest)
    if not m:
        raise ValueError(f"unrecognized decoder-block sub-key: {rest!r}")
    j, tail = int(m.group(1)), m.group(2)
    if j == 0:
        return f"snake1.{tail}"
    elif j == 1:
        return f"conv_t1.{tail}"
    elif j in (2, 3, 4):
        res_unit_name = {2: "res_unit1", 3: "res_unit2", 4: "res_unit3"}[j]
        return f"{res_unit_name}.{_convert_res_unit(tail)}"
    raise ValueError(f"unexpected decoder-block sub-index {j} in {rest!r}")


def convert_oobleck_key(key: str) -> str:
    """Map one raw (stable-audio-tools Sequential) key to its diffusers
    `AutoencoderOobleck` equivalent."""
    m = _TOP_LEVEL_RE.match(key)
    if not m:
        raise ValueError(f"unrecognized ACE-Step VAE key: {key!r}")
    side, idx, rest = m.group(1), int(m.group(2)), m.group(3)
    if idx == 0:
        return f"{side}.conv1.{rest}"
    elif 1 <= idx <= 5:
        block_idx = idx - 1
        sub = _convert_encoder_block_key(rest) if side == "encoder" else _convert_decoder_block_key(rest)
        return f"{side}.block.{block_idx}.{sub}"
    elif idx == 6:
        return f"{side}.snake1.{rest}"
    elif idx == 7:
        return f"{side}.conv2.{rest}"
    raise ValueError(f"unexpected top-level ACE-Step VAE index {idx} in {key!r}")


def convert_oobleck_state_dict(raw_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert a full raw ACE-Step VAE state_dict to diffusers `AutoencoderOobleck`
    naming, reshaping the flat `Snake1d` `alpha`/`beta` params to `(1, C, 1)`."""
    converted: Dict[str, torch.Tensor] = {}
    for key, tensor in raw_sd.items():
        new_key = convert_oobleck_key(key)
        if new_key.endswith(".alpha") or new_key.endswith(".beta"):
            tensor = tensor.reshape(1, -1, 1)
        converted[new_key] = tensor
    return converted
