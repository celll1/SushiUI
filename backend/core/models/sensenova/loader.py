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
in ``core.models.common.int8_runtime_quantize``) is, in the checkpoint this
repo distributes, a per-output-row-scaled PLAIN int8 Linear, the SAME on-disk
layout Ideogram 4/Krea 2/FLUX.2/Anima already read via
``core.models.ideogram4.vendor.int8_linear``, with no ConvRot rotation and no
NVFP4/AWQ smoothing -- verified through
``core.models.common.quantized_checkpoint_guard.verify_quantized_swap`` (a
swap-count mismatch -- an unswapped quantized layer reaching
``load_state_dict(assign=True)`` -- REFUSES the load rather than silently
installing int8 codes as a bf16 parameter; see that module's docstring).

A SEPARATE ConvRot-quantized checkpoint (Hadamard-rotated ``int8_tensorwise``,
the same contract MiniMax-H3's DiT reads -- see
``core.models.common.convrot_marker``) is also accepted: layers whose
``.comfy_quant`` marker validates are swapped to
``core.models.common.convrot_int8_linear.ConvRotInt8Linear`` instead of the
plain ``Int8Linear`` above. Selecting that checkpoint file IS the opt-in;
there is no separate parameter. ``SUSHI_SENSENOVA_CONVROT_DEQUANT`` (below
``_apply_sensenova_convrot_dequant_ablation``) is a debug-only env var, not an
API surface, that forces selected ConvRot layer groups onto the dequant path.
ConvRot LAYERS ARE NOT COVERED BY THE W8A8 PIN BELOW: ``disable_int8_mm`` is
``isinstance(Int8Linear)``-based and does flip the inherited
``_allow_int8_mm`` flag on a loaded ``ConvRotInt8Linear``, but that flag is
inert there -- ``ConvRotInt8Linear.forward`` never reads it and always
dispatches to comfy-kitchen's fused W8A8 kernel (or ``_dequant_forward``
under grad / the ablation above). A ConvRot checkpoint therefore always runs
W8A8, unconditionally, regardless of the pin. Whether SenseNova tolerates
rotated W8A8 at all is unmeasured -- the confirmed regression below was
characterized on the PLAIN (unrotated) int8 path only, and ConvRot
deliberately bypasses this pin rather than being silently covered by it.

W8A8 IS PINNED OFF (``disable_int8_mm``) FOR THE PLAIN INT8 LINEARS ABOVE,
unlike Ideogram 4/Krea 2/FLUX.2/Anima -- this pin does NOT reach ConvRot
layers, see the paragraph above. This is a DIFFERENT pin from MiniMax-H3's
declared-semantics mismatch (see that loader for detail): SenseNova's plain
checkpoint has no such markers and unremarkable weight-quantization error, so
this is instead an empirically confirmed W8A8 numerics regression, re-verified
with 5 replays per arm against a backend carrying e77b1dd7 (ruling out that
unrelated probabilistic uninitialized-flash-KV-cache-tail bug as a confound)
-- fully deterministic in both directions. The mechanism is NOT isolated; see
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
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Optional

import torch

from core.models.common.single_file_format import read_state_dict, strip_prefix, TRANSFORMER_PREFIX
from core.models.common.convrot_marker import int8_convrot_layers_from_markers
from core.models.common.quantized_checkpoint_guard import (
    quantized_state_dict_report, refuse_unsupported_quant_semantics,
    scaled_quantization_report, verify_quantized_swap,
)
from core.models.ideogram4.vendor.int8_linear import (
    disable_int8_mm, is_int8_state_dict, swap_linears_to_int8,
)

from .vendor import NEOChatConfig, NEOChatModel

ARCH_LABEL = "SenseNova"

# The six debug ablation groups Task C exposes -- see
# ``_apply_sensenova_convrot_dequant_ablation``. Named by branch (MoT's
# ``_mot_gen``-suffixed generation-branch duplicates vs. the plain
# understanding-branch modules -- see ``modeling_qwen3.Qwen3Attention``) x
# layer kind.
_SENSENOVA_CONVROT_ABLATION_GROUPS = (
    "gen_attn_qkv", "gen_o_proj", "gen_mlp",
    "understanding_attn_qkv", "understanding_o_proj", "understanding_mlp",
)


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


def _int8_convrot_source_layers(sd: Dict[str, torch.Tensor], *, path: str) -> Dict[str, Dict[str, int]]:
    """Adapt the shared (header/handle-shaped) marker validator to a real state dict.

    Unlike MiniMax-H3's lazy safetensors reader, ``read_state_dict`` already
    materializes every tensor, so the "header" the shared validator expects
    (``{key: {"shape": [...], "dtype": <safetensors code>}}``) is built from
    the real tensors' own shape/dtype rather than a parsed safetensors JSON
    header, and ``handle.get_tensor`` is just a dict lookup.
    """
    dtype_codes = {torch.int8: "I8", torch.float32: "F32"}
    header = {
        key: {"shape": list(t.shape), "dtype": dtype_codes.get(t.dtype, str(t.dtype))}
        for key, t in sd.items()
    }
    handle = SimpleNamespace(get_tensor=sd.__getitem__)
    return int8_convrot_layers_from_markers(handle, header, path=path)


def _sensenova_convrot_ablation_group(module_path: str) -> Optional[str]:
    """Classify a ConvRot Linear's dotted module path into one of six debug groups.

    Branch is the ``_mot_gen`` suffix (the MoT generation-branch duplicate;
    see ``modeling_qwen3.Qwen3Attention``/``Qwen3DecoderLayer``), layer kind is
    attention q/k/v, attention o_proj, or MLP. ``None`` for anything else.

    The ``.mlp.``/``.mlp_mot_gen.`` checks are unanchored substring tests, not
    a decoder-layer-only match -- a path like
    ``fm_modules.vision_model_mot_gen....mlp.fc1`` would also match
    ``gen_mlp``. This is unreachable today because ``ARCH_QUANT_POLICY``
    scopes ConvRot markers to the 42 decoder layers only, and this function is
    debug-only; it is not a total, arch-wide classifier.
    """
    if module_path.endswith(".self_attn.o_proj_mot_gen"):
        return "gen_o_proj"
    if module_path.endswith(".self_attn.o_proj"):
        return "understanding_o_proj"
    if any(module_path.endswith(f".self_attn.{p}_proj_mot_gen") for p in ("q", "k", "v")):
        return "gen_attn_qkv"
    if any(module_path.endswith(f".self_attn.{p}_proj") for p in ("q", "k", "v")):
        return "understanding_attn_qkv"
    if ".mlp_mot_gen." in module_path:
        return "gen_mlp"
    if ".mlp." in module_path:
        return "understanding_mlp"
    return None


def _apply_sensenova_convrot_dequant_ablation(model: NEOChatModel) -> None:
    """DEBUG ONLY: force selected ConvRot layer groups onto the dequant path.

    Backend-config-only -- ``SUSHI_SENSENOVA_CONVROT_DEQUANT``, a comma-
    separated subset of ``_SENSENOVA_CONVROT_ABLATION_GROUPS`` (or ``"all"``),
    never an API parameter or a frontend control. No-op when unset.

    ``ConvRotInt8Linear.forward`` only reaches ``_dequant_forward`` when
    ``torch.is_grad_enabled() and x.requires_grad``, which is never true at
    inference, so this sets the module-level ``_force_dequant`` override
    instead of relying on grad state.

    This isolates ACTIVATION quantization error, not weight quantization
    error (the SAME int8 weight runs against a full-precision activation).
    It is also not fp32-exact: ``_dequant_forward`` requests the weight in
    ``x.dtype``, so on bf16 activations the weight is additionally rounded
    to bf16 on top of the int8 weight's own rounding.
    """
    raw = os.environ.get("SUSHI_SENSENOVA_CONVROT_DEQUANT", "")
    requested = {g.strip() for g in raw.split(",") if g.strip()}
    if not requested:
        return
    if requested == {"all"}:
        requested = set(_SENSENOVA_CONVROT_ABLATION_GROUPS)
    unknown = requested - set(_SENSENOVA_CONVROT_ABLATION_GROUPS)
    if unknown:
        raise ValueError(
            f"SUSHI_SENSENOVA_CONVROT_DEQUANT names unknown group(s) {sorted(unknown)}; "
            f"valid groups are {_SENSENOVA_CONVROT_ABLATION_GROUPS} (or 'all')"
        )

    from core.models.common.convrot_int8_linear import ConvRotInt8Linear

    forced = 0
    for name, module in model.named_modules():
        if isinstance(module, ConvRotInt8Linear) and _sensenova_convrot_ablation_group(name) in requested:
            module._force_dequant = True
            forced += 1
    print(f"[SenseNovaLoader] ConvRot dequant ablation (SUSHI_SENSENOVA_CONVROT_DEQUANT): "
          f"forced {forced} Linear(s) in group(s) {sorted(requested)} onto the dequant path")


def _sensenova_quant_dict_views(
    sd: Dict[str, torch.Tensor],
    int8_convrot_source_layers: Dict[str, Dict[str, int]],
) -> "tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]":
    """Three filtered views of ``sd``, split on which layers are ConvRot.

    Returns ``(guard_sd, plain_sd, sd_for_load)``:

    * ``guard_sd`` -- every key EXCEPT a ConvRot layer's own ``.comfy_quant``
      marker. The early marker read already validated it; declared-semantics
      refusal (``refuse_unsupported_quant_semantics``) must not re-reject a
      marker it already accepted (mirrors MiniMax-H3 DiT loader.py's
      ``guard_state_dict``).
    * ``plain_sd`` -- every key under a ConvRot layer's prefix excluded
      ENTIRELY (weight, scale AND marker, not just the marker): leaving a
      ConvRot layer's ``.comfy_quant`` in the dict passed to
      ``is_int8_state_dict``/``quantized_state_dict_report`` would refuse on
      its own "convrot: true" declaration -- the exact bug MiniMax-H3's
      ``_swap_minimax_h3_quantized_linears`` has today (unreachable there
      only because that arch's real files carry no mixed ConvRot +
      plain-scaled layers; see ``core.models.common.convrot_marker``).
    * ``sd_for_load`` -- every ``.comfy_quant`` key dropped EXCEPT a ConvRot
      layer's own (plain provenance markers have served their purpose;
      ConvRot ones are retained as live module state so a state_dict/export
      cannot lose the rotation contract, mirrors MiniMax-H3 DiT loader.py).
    """
    guard_sd = {
        key: value for key, value in sd.items()
        if not (key.endswith(".comfy_quant") and key[: -len(".comfy_quant")] in int8_convrot_source_layers)
    }
    int8_convrot_prefixes = tuple(name + "." for name in int8_convrot_source_layers)
    plain_sd = {
        key: value for key, value in sd.items()
        if not int8_convrot_prefixes or not key.startswith(int8_convrot_prefixes)
    }
    sd_for_load = {
        key: value for key, value in sd.items()
        if not key.endswith(".comfy_quant") or key[: -len(".comfy_quant")] in int8_convrot_source_layers
    }
    return guard_sd, plain_sd, sd_for_load


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

    ``sd`` must have any ConvRot ``.comfy_quant`` marker already excluded (the
    caller passes the convrot-prefix-excluded dict): ``is_int8_state_dict``
    refuses on a ``convrot: true`` marker (a stricter contract than this
    function implements), so a marker-bearing dict would wrongly refuse a
    mixed ConvRot + plain-int8 checkpoint.
    """
    if not is_int8_state_dict(sd):
        return 0
    swapped = swap_linears_to_int8(model, sd, compute_dtype=dtype)
    # ``disable_int8_mm`` is isinstance(Int8Linear)-based, so it also flips
    # ``_allow_int8_mm`` on any already-swapped ``ConvRotInt8Linear`` (a
    # subclass) -- but that flag is inert there: ``ConvRotInt8Linear.forward``
    # never reads it, it always dispatches to comfy-kitchen's fused kernel (or
    # ``_dequant_forward`` under grad / the Task C ablation). Whether
    # SenseNova tolerates rotated W8A8 at all is unmeasured; ConvRot
    # deliberately bypasses this pin rather than silently being covered by it.
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

    # ConvRot markers, read from the still-plain state dict before anything is
    # installed (mirrors MiniMax-H3 DiT loader.py's ordering: the runtime
    # requirement is checked before any payload is swapped in or loaded).
    int8_convrot_source_layers = _int8_convrot_source_layers(sd, path=model_path)
    if int8_convrot_source_layers:
        from core.models.common.convrot_int8_linear import require_convrot_int8_runtime

        require_convrot_int8_runtime()

    with init_empty_weights():
        model = NEOChatModel(config)
        model.to(torch_dtype)

    # The early marker read above validated every supported ConvRot marker;
    # every OTHER declared-semantics marker must still refuse. See
    # ``_sensenova_quant_dict_views`` for what each of the three views means.
    guard_sd, plain_sd, sd_for_load = _sensenova_quant_dict_views(sd, int8_convrot_source_layers)
    refuse_unsupported_quant_semantics(guard_sd, arch=ARCH_LABEL, path=model_path, label="transformer")

    # Census + verify BEFORE the swap+load, same discipline as every other
    # quantized loader in this repo (Anima/Ideogram 4): a scale-less or
    # partially-matched quantized file must refuse rather than silently cast
    # int8 codes into a bf16 parameter.
    census = quantized_state_dict_report(plain_sd, arch=ARCH_LABEL, path=model_path, label="transformer")
    quant_report = scaled_quantization_report(census, arch=ARCH_LABEL, path=model_path, label="transformer")

    swapped = 0
    if int8_convrot_source_layers:
        from core.models.common.convrot_int8_linear import swap_linears_to_convrot_int8

        convrot_swapped = swap_linears_to_convrot_int8(
            model, sd_for_load, int8_convrot_source_layers, torch_dtype
        )
        if convrot_swapped != len(int8_convrot_source_layers):
            raise RuntimeError(
                f"SenseNova ConvRot metadata mapped {len(int8_convrot_source_layers)} Linear(s), "
                f"but only {convrot_swapped} module(s) were replaced"
            )
        swapped += convrot_swapped
    _apply_sensenova_convrot_dequant_ablation(model)

    plain_swapped = _swap_sensenova_quantized_linears(model, plain_sd, torch_dtype)
    verify_quantized_swap(quant_report, plain_swapped, arch=ARCH_LABEL, path=model_path, label="transformer")
    swapped += plain_swapped

    missing, unexpected = model.load_state_dict(sd_for_load, strict=False, assign=True)
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
