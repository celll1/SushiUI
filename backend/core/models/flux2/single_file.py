"""FLUX.2 single-file facts shared by the loader and the offline quantizer.

Two things live here, both of which used to exist only inside
``core/model_loader.load_flux2_from_safetensors`` and could therefore not be
reused by anything that reads a FLUX.2 checkpoint WITHOUT loading the model:

1. **The transformer geometry pin** (``FLUX2_DEFAULT_CONFIG`` /
   ``flux2_config_for_state_dict``). The loader resolves its config by
   ``huggingface_hub.snapshot_download`` of the detected base repo, which is the
   right thing for a loader (it needs the text encoder, tokenizer and scheduler
   from that repo anyway) and the wrong thing for an offline tool that only wants
   to enumerate ``nn.Linear`` module paths: it would make a 3.6 GB-model-shaped
   decision depend on network access and on a gated repo. The pinned config below
   is the FLUX.2 Klein 4B transformer config verbatim (identical in the
   ``FLUX.2-klein-base-4B`` and the distilled ``FLUX.2-Klein-4B`` repos --
   distillation changes the weights, not the geometry).

   It is a PIN, not a guess: ``flux2_config_for_state_dict`` REFUSES a checkpoint
   whose block counts it does not recognise instead of defaulting to 4B the way
   ``model_loader.py``'s legacy ``single_blocks.47.`` probe does. That probe
   looks for 24/36/48 single blocks and BOTH 4B variants measured here have 20,
   so on every FLUX.2 file this repo has actually seen it falls through to its
   default. (Klein 9B is gated; its block count is unknown here, so whether the
   36-block arm ever fires is not something this repo can state.) Falling through
   is acceptable when the config that is then loaded comes from the repo itself,
   and not acceptable when the config IS the answer.

2. **The BFL -> diffusers key transform** (``flux2_bfl_to_diffusers``), in the
   per-tensor form ``core.models.common.quantized_export``'s
   ``source_transform`` contract wants. It delegates to diffusers'
   ``convert_flux2_transformer_checkpoint_to_diffusers`` -- the SAME function the
   production loader runs -- rather than reimplementing the remap, so an offline
   int8 artifact carries exactly the keys a loaded BFL checkpoint would have had.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch

__all__ = [
    "FLUX2_DEFAULT_CONFIG",
    "FLUX2_CONFIGS_BY_BLOCK_COUNTS",
    "count_flux2_blocks",
    "flux2_config_for_state_dict",
    "is_flux2_bfl_key",
    "is_flux2_bfl_state_dict",
    "flux2_bfl_to_diffusers",
]


# FLUX.2 Klein 4B transformer config, verbatim from
# ``black-forest-labs/FLUX.2-klein-base-4B`` -> ``transformer/config.json``
# (byte-identical to the distilled ``FLUX.2-Klein-4B`` one). The diffusers
# bookkeeping keys (``_class_name``, ``_diffusers_version``, ``_name_or_path``)
# are dropped: ``Flux2Transformer2DModel.from_config`` does not need them and
# ``_name_or_path`` is a path on someone else's machine.
FLUX2_DEFAULT_CONFIG: Dict[str, object] = {
    "attention_head_dim": 128,
    "axes_dims_rope": [32, 32, 32, 32],
    "eps": 1e-06,
    "guidance_embeds": False,
    "in_channels": 128,
    "joint_attention_dim": 7680,
    "mlp_ratio": 3.0,
    "num_attention_heads": 24,
    "num_layers": 5,
    "num_single_layers": 20,
    "out_channels": None,
    "patch_size": 1,
    "rope_theta": 2000,
    "timestep_guidance_channels": 256,
}

# (num_layers, num_single_layers) -> config. ONE entry today, because one is all
# that can be pinned from a config that was actually read: both 4B variants
# ship (5, 20). A 9B checkpoint would add a row here (its repo is gated, so its
# geometry is not known to this repo) -- until then it is refused, not guessed.
FLUX2_CONFIGS_BY_BLOCK_COUNTS: Dict[Tuple[int, int], Dict[str, object]] = {
    (5, 20): FLUX2_DEFAULT_CONFIG,
}


_BFL_DOUBLE_RE = re.compile(r"^double_blocks\.(\d+)\.")
_BFL_SINGLE_RE = re.compile(r"^single_blocks\.(\d+)\.")
_DIFFUSERS_DOUBLE_RE = re.compile(r"^transformer_blocks\.(\d+)\.")
_DIFFUSERS_SINGLE_RE = re.compile(r"^single_transformer_blocks\.(\d+)\.")


def count_flux2_blocks(keys: Iterable[str]) -> Tuple[int, int]:
    """``(num_layers, num_single_layers)`` implied by a FLUX.2 key set.

    Accepts BFL (``double_blocks.``/``single_blocks.``) or diffusers
    (``transformer_blocks.``/``single_transformer_blocks.``) spellings, and
    counts DISTINCT indices rather than assuming they are contiguous -- a
    truncated checkpoint should report the count it has, so the caller can
    refuse it, not a plausible-looking maximum.
    """
    double: set = set()
    single: set = set()
    for key in keys:
        for pattern, sink in (
            (_BFL_DOUBLE_RE, double), (_BFL_SINGLE_RE, single),
            (_DIFFUSERS_SINGLE_RE, single), (_DIFFUSERS_DOUBLE_RE, double),
        ):
            m = pattern.match(key)
            if m is not None:
                sink.add(int(m.group(1)))
                break
    return len(double), len(single)


def flux2_config_for_state_dict(keys: Sequence[str],
                                override: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """The pinned transformer config for a FLUX.2 checkpoint's key set.

    ``override`` (a full transformer config, e.g. a repo's
    ``transformer/config.json``) wins outright and is returned as given: a caller
    who has the real config should not have their geometry second-guessed by a
    table with one row in it.

    Raises ``ValueError`` for a block-count combination that is not pinned. That
    is deliberate: the config decides the module tree the caller will enumerate
    Linears from, so a wrong guess produces an artifact whose keys silently do
    not match any model.
    """
    if override:
        return dict(override)
    counts = count_flux2_blocks(keys)
    config = FLUX2_CONFIGS_BY_BLOCK_COUNTS.get(counts)
    if config is None:
        known = ", ".join(f"{d} double + {s} single" for d, s in sorted(FLUX2_CONFIGS_BY_BLOCK_COUNTS))
        if counts == (0, 0):
            # Both block regexes are ``^``-anchored, so FLUX.2's THIRD key layout
            # -- the sushiUI/musubi full-FT save, diffusers keys behind a
            # ``model.diffusion_model.`` prefix (see
            # ``model_loader.load_flux2_from_safetensors``) -- counts zero blocks
            # of both kinds. Reporting that as "unrecognised geometry: 0 + 0" and
            # suggesting a config would only move the failure one step later, so
            # the prefix is named, and so is the general no-blocks case.
            prefixed = any(str(k).startswith("model.diffusion_model.") for k in keys)
            if prefixed:
                raise ValueError(
                    "this looks like a sushiUI/musubi FLUX.2 full-FT save: its keys "
                    "carry a 'model.diffusion_model.' prefix, which is the one FLUX.2 "
                    "single-file layout this path does not read. The supported source "
                    "layouts are BFL/Comfy ('double_blocks.*'/'single_blocks.*') and "
                    "diffusers ('transformer_blocks.*'/'single_transformer_blocks.*'). "
                    "Supplying a config would not help -- the keys themselves would "
                    "still not match any module path. Route for this layout: LOAD the "
                    "checkpoint (its loader strips the prefix), generate once with "
                    "unet_quantization='int8' to convert the live transformer, then "
                    "POST /models/export-quantized."
                )
            raise ValueError(
                "this checkpoint contains no FLUX.2 transformer block keys at all "
                "(no 'double_blocks.*'/'single_blocks.*' and no "
                "'transformer_blocks.*'/'single_transformer_blocks.*'), so it is "
                "either not a FLUX.2 transformer or not a transformer-only file."
            )
        raise ValueError(
            f"unrecognised FLUX.2 transformer geometry: {counts[0]} double block(s) + "
            f"{counts[1]} single block(s). Pinned geometries: {known}. Supply the "
            f"checkpoint's own transformer/config.json explicitly rather than having "
            f"this guess a variant."
        )
    return dict(config)


# ---------------------------------------------------------------------------
# BFL -> diffusers key transform
# ---------------------------------------------------------------------------
#
# Per-KEY detection, not per-checkpoint. The ``source_transform`` contract is one
# key at a time and carries no state, and a per-key rule is also the honest one:
# the two layouts are disjoint at the top level, so "is this key BFL-shaped" is
# decidable on its own.
#
# Getting it wrong in the permissive direction is not harmless. diffusers'
# rename table maps ``double_stream_modulation_img.lin`` ->
# ``double_stream_modulation_img.linear`` with a plain ``str.replace``, so
# running it over an ALREADY-diffusers key would rewrite
# ``...img.linear.weight`` to ``...img.linearear.weight``. Hence the modulation
# rule below matches ``.lin.`` exactly.
_BFL_TOP_LEVEL = (
    "img_in",            # -> x_embedder
    "txt_in",            # -> context_embedder
    "time_in",           # -> time_guidance_embed.timestep_embedder
    "guidance_in",       # -> time_guidance_embed.guidance_embedder
    "final_layer",       # -> proj_out / norm_out.linear
    "double_blocks",     # -> transformer_blocks
    "single_blocks",     # -> single_transformer_blocks
)
_BFL_MODULATION_RE = re.compile(
    r"^(double_stream_modulation_img|double_stream_modulation_txt|single_stream_modulation)\.lin\.")

# Stand-in tensor for the KEY-ENUMERATION pass (``tensor=None``). The converter
# splits fused qkv with ``chunk(3, dim=0)`` and swaps the AdaLN halves with
# ``chunk(2, dim=0)``, so the probe's leading dimension must be divisible by
# both; only the resulting KEYS are used, never its values.
_KEY_PROBE = torch.zeros(6, 1)


def is_flux2_bfl_key(key: str) -> bool:
    """True when ``key`` is in the BFL/Comfy layout rather than the diffusers one."""
    return key.split(".", 1)[0] in _BFL_TOP_LEVEL or bool(_BFL_MODULATION_RE.match(key))


def is_flux2_bfl_state_dict(keys: Iterable[str]) -> bool:
    """True when any key is BFL-shaped (the loader's own ``double_blocks.`` test,
    widened to the top-level projections so a transform-only caller agrees with it)."""
    return any(is_flux2_bfl_key(k) for k in keys)


def flux2_bfl_to_diffusers(key: str, tensor: Optional[torch.Tensor]):
    """``source_transform`` for FLUX.2: BFL keys -> diffusers keys, else identity.

    One input key yields one output key, except a fused attention projection
    (``double_blocks.N.{img,txt}_attn.qkv.weight``), which yields three. The
    tensor is transformed too where the layout demands it: the qkv split is a
    ``chunk``, and ``final_layer.adaLN_modulation.1.weight`` has its (shift,
    scale) halves swapped to diffusers' (scale, shift) order.

    Splitting BEFORE quantization is numerically free -- the int8/e4m3 scales are
    per output ROW and the rows are independent -- and it is what makes an
    offline artifact comparable with a runtime export, which necessarily sees the
    module after the loader has already split.
    """
    if not is_flux2_bfl_key(key):
        return ((key, tensor),)
    from diffusers.loaders.single_file_utils import (
        convert_flux2_transformer_checkpoint_to_diffusers,
    )
    probe = _KEY_PROBE if tensor is None else tensor
    converted = convert_flux2_transformer_checkpoint_to_diffusers({key: probe})
    if tensor is None:
        return tuple((k, None) for k in converted)
    return tuple(converted.items())
