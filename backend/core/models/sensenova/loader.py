"""Component loader for SenseNova-U1.5-8B-MoT (Qwen3-8B-as-flow-matching-denoiser).

DISTRIBUTION FORMAT
--------------------
This loader reads exactly the sushiUI shard index Unit 1's converter writes
(``core.models.common.single_file_format``): a ``<stem>.safetensors.index.json``
+ shard files, all tensor keys prefixed ``transformer.``, plus sibling
tokenizer files (``tokenizer_config.json``, ``vocab.json``, ``merges.txt``,
``special_tokens_map.json``, ``added_tokens.json``) and a sibling
``config.json`` (the upstream ``NEOChatConfig`` dict, also embedded verbatim
in the checkpoint's own metadata as ``sensenova_config`` -- see below). There
is no other on-disk shape this loader accepts: unlike Krea 2/Ideogram 4/Lens,
SenseNova has no upstream single-file distribution to complete siblings
against, because Unit 1's conversion IS the only distribution this repo reads.

Geometry (``NEOChatConfig``) comes from the checkpoint's own
``sensenova_config`` metadata FIRST -- mirroring how MiniMax-H3's pruned DiT
records ``minimax_h3_config`` because its geometry cannot be read from a
shipped ``config.json`` for the variant actually distributed. Here the
opposite risk applies (a sibling ``config.json`` could drift from the
tensors the checkpoint was converted from), so the embedded copy is
authoritative and the sibling file is only a fallback for a checkpoint whose
metadata was stripped.

QUANTIZATION
------------
Every one of the 588 int8 Linears (the 42 decoder layers' MoT-doubled
``self_attn.{q,k,v,o}_proj`` and ``mlp.{gate,up,down}_proj``, both the
understanding and generation branches -- see ``ARCH_QUANT_POLICY["sensenova"]``
in ``core.models.common.int8_runtime_quantize``) is a per-output-row-scaled
plain int8 Linear, the SAME on-disk layout Ideogram 4/Krea 2/FLUX.2/Anima
already read via ``core.models.ideogram4.vendor.int8_linear``, with no
ConvRot rotation and no NVFP4/AWQ smoothing -- verified through
``core.models.common.quantized_checkpoint_guard.verify_quantized_swap`` (a
swap-count mismatch -- an unswapped quantized layer reaching
``load_state_dict(assign=True)`` -- REFUSES the load rather than silently
installing int8 codes as a bf16 parameter; see that module's docstring).

W8A8 IS PINNED OFF (``disable_int8_mm``), unlike Ideogram 4/Krea 2/FLUX.2/
Anima. This is a DIFFERENT pin from MiniMax-H3's declared-semantics
mismatch (see that loader for detail): SenseNova's checkpoint has no such
markers and unremarkable weight-quantization error, so this is instead an
empirically confirmed W8A8 numerics regression, re-verified with 5 replays
per arm against a backend carrying e77b1dd7 (ruling out that unrelated
probabilistic uninitialized-flash-KV-cache-tail bug as a confound) --
fully deterministic in both directions. The mechanism is NOT isolated; see
``docs/guides/MODEL_FACTS.md``'s sensenova row for the full evidence and
next-investigation notes. Revisit if someone isolates it; until then this
pin is the safe default.

The model is built under ``accelerate.init_empty_weights()`` (meta device,
mirroring the Anima loader) because it is 18.7 GB and this repo's other
quantized loaders that build-then-strict-load would otherwise materialize a
full bf16 copy before the int8 swap ever runs. ``assign=True`` then installs
the checkpoint's own tensors (int8 weight + fp32 scale for a quantized
Linear, whatever dtype the file stores for everything else) directly, with no
intermediate cast.

RESIDENCY
---------
Registers ONE component (``"transformer"``: the whole ``NEOChatModel``,
carrying the Qwen3 backbone, the vision patch embedder and the flow-matching
head modules together) -- all-or-nothing residency, no TE/VAE/UNet split, the
same shape MiniT2I uses for its pixel-space, single-component model. Nothing
is staged to GPU here; everything stays on the CPU at load, exactly like
every other loader in this repo.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, Optional

import torch

from core.models.common.single_file_format import read_state_dict, strip_prefix, TRANSFORMER_PREFIX
from core.models.common.quantized_checkpoint_guard import (
    quantized_state_dict_report, scaled_quantization_report, verify_quantized_swap,
)
from core.models.ideogram4.vendor.int8_linear import (
    disable_int8_mm, is_int8_state_dict, swap_linears_to_int8,
)

from .vendor import NEOChatConfig, NEOChatModel

ARCH_LABEL = "SenseNova"


def is_sensenova_state_dict_keys(keys: Iterable[str]) -> bool:
    """SenseNova signature: MoT-doubled Qwen3 attention + the flow-matching pixel head.

    ``transformer.``-stripped first (both the shard index and a bare
    single-file save carry that prefix; the LIVE module's own keys do not).
    Three independent markers, all required:

    * ``*_mot_gen.weight`` on an attention Linear -- the MoT (Mixture-of-
      Transformers) weight duplication no other architecture in this repo has;
    * ``fm_modules.fm_head.`` -- the flow-matching pixel head, unique to this
      arch's ConvDecoder;
    * ``language_model.model.layers.`` -- the Qwen3-as-denoiser backbone.

    Key NAMES only (usable against a shard ``weight_map``, mirroring
    ``ModelLoader._keys_look_anima``'s delegation pattern).
    """
    stripped = [k[len(TRANSFORMER_PREFIX):] if k.startswith(TRANSFORMER_PREFIX) else k for k in keys]
    has_mot_gen = any(k.endswith("q_proj_mot_gen.weight") for k in stripped)
    has_fm_head = any(k.startswith("fm_modules.fm_head.") for k in stripped)
    has_llm_layers = any(k.startswith("language_model.model.layers.") for k in stripped)
    return has_mot_gen and has_fm_head and has_llm_layers


def _load_sensenova_config(metadata: Dict[str, Any], model_dir: str) -> NEOChatConfig:
    """The checkpoint's ``NEOChatConfig``: embedded metadata first, sibling ``config.json`` fallback."""
    raw = (metadata or {}).get("sensenova_config")
    if raw:
        cfg_dict = json.loads(raw)
        source = "embedded sensenova_config metadata"
    else:
        cfg_path = os.path.join(model_dir, "config.json")
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(
                f"SenseNova checkpoint at {model_dir!r} carries no 'sensenova_config' metadata "
                f"and has no sibling config.json; cannot determine model geometry."
            )
        with open(cfg_path, encoding="utf-8") as f:
            cfg_dict = json.load(f)
        source = f"sibling {cfg_path}"
    print(f"[SenseNovaLoader] config source: {source}")
    return NEOChatConfig(**cfg_dict)


def _swap_sensenova_quantized_linears(model: NEOChatModel, sd: Dict[str, torch.Tensor],
                                      dtype: torch.dtype) -> int:
    """Replace every int8-saved ``nn.Linear`` with ``Int8Linear``. Returns the count.

    Only int8 is checked (unlike Ideogram 4/Anima, which also probe fp8):
    Unit 1's conversion emits int8 exclusively, and the census/verify pair
    below would refuse a file that carried anything else anyway.

    W8A8 (``torch._int_mm``) is pinned off on every swapped-in ``Int8Linear``
    -- see the module docstring's QUANTIZATION section for what is and is not
    known about why. This pin is authoritative over ``SUSHI_INT8_MM`` and any
    per-generation ``quantized_gemm_mode='w8a8'`` request.
    """
    if not is_int8_state_dict(sd):
        return 0
    swapped = swap_linears_to_int8(model, sd, compute_dtype=dtype)
    disable_int8_mm(model, label="SenseNova transformer")
    print(f"[SenseNovaLoader] weight-only int8 checkpoint: swapped {swapped} Int8Linear(s); "
          f"the rest load at their checkpoint dtype")
    return swapped


def load_sensenova_from_path(
    model_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, Any]:
    """Load SenseNova-U1.5-8B-MoT from a sushiUI shard index (or single-file save).

    Returns the component dict ``PipelineManager.load_model()`` consumes:
    ``{type, transformer, config, tokenizer}``. Everything stays CPU-resident;
    nothing is staged to GPU here.
    """
    from accelerate import init_empty_weights

    if not isinstance(model_path, str) or not os.path.isfile(model_path):
        raise FileNotFoundError(f"SenseNova checkpoint not found at {model_path!r}")

    model_dir = os.path.dirname(model_path)
    print(f"[SenseNovaLoader] Reading state dict: {model_path}")
    raw_sd, metadata = read_state_dict(model_path)
    sd = strip_prefix(raw_sd, TRANSFORMER_PREFIX)
    if not sd:
        raise ValueError(
            f"SenseNova checkpoint at {model_path!r} carries no '{TRANSFORMER_PREFIX}'-prefixed "
            f"tensors; this loader only reads the sushiUI single-file/shard-index format."
        )

    config = _load_sensenova_config(metadata, model_dir)

    with init_empty_weights():
        model = NEOChatModel(config)
        model.to(torch_dtype)

    # Census + verify BEFORE the swap+load, same discipline as every other
    # quantized loader in this repo (Anima/Ideogram 4): a scale-less or
    # partially-matched quantized file must refuse rather than silently cast
    # int8 codes into a bf16 parameter.
    census = quantized_state_dict_report(sd, arch=ARCH_LABEL, path=model_path, label="transformer")
    quant_report = scaled_quantization_report(census, arch=ARCH_LABEL, path=model_path, label="transformer")
    swapped = _swap_sensenova_quantized_linears(model, sd, torch_dtype)
    verify_quantized_swap(quant_report, swapped, arch=ARCH_LABEL, path=model_path, label="transformer")

    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    if missing:
        print(f"[SenseNovaLoader] WARNING: {len(missing)} missing key(s); first 5: {missing[:5]}")
    if unexpected:
        print(f"[SenseNovaLoader] WARNING: {len(unexpected)} unexpected key(s); first 5: {unexpected[:5]}")

    model.eval()
    model.requires_grad_(False)
    # Stays on CPU: this loader stages nothing to GPU (see module docstring).

    tokenizer = _load_sensenova_tokenizer(model_dir)

    print(f"[SenseNovaLoader] Loaded SenseNova-U1.5-8B-MoT ({swapped} int8 Linear(s); CPU-resident).")

    return {
        "type": "sensenova",
        "transformer": model,
        "config": config,
        "tokenizer": tokenizer,
    }


def _load_sensenova_tokenizer(model_dir: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.encode("validation", add_special_tokens=False)  # sanity encode
    return tokenizer
