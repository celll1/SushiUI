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
W8A8, unconditionally, regardless of the pin. That bypass is deliberate, not
an oversight, and it was gated before being relied on: a fixed-seed A/B/C
sweep over all four generation modes reproduced no late-step burst on the
rotated path (the regression below was characterized on the PLAIN, unrotated
path only), at 1.69-2.02x the plain checkpoint's per-step time. No dequant
group carve-out is needed; see ``docs/guides/MODEL_FACTS.md``'s sensenova row.

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

Full-parameter TRAINING is the one caller that will undo this: see
``materialize_int8_decoder_linears`` below, which dequantizes one MoT half's
294 Linears (or both halves' 588) back to real ``nn.Parameter`` weights.
Inference never calls it, and nothing does yet -- SenseNova full FT is still
refused by ``arch_capabilities`` and ``train_runner`` until U-2-2.

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
from torch import nn

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


def _load_sensenova_config(
    metadata: Dict[str, Any], model_dir: str
) -> "tuple[NEOChatConfig, Dict[str, Any]]":
    """The checkpoint's ``NEOChatConfig``: embedded metadata first, sibling ``config.json`` fallback.

    Returns the config AND the raw dict it was built from. The dict is what a
    later export re-embeds, so a re-save carries the exact block this load
    accepted rather than a re-serialization of the live object
    (``_embeddable_sensenova_config``).
    """
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
    return NEOChatConfig(**cfg_dict), cfg_dict


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


def _reshape_convrot_scales(
    sd: Dict[str, torch.Tensor], int8_convrot_source_layers: Dict[str, Dict[str, int]]
) -> int:
    """Reshape marker-validated ConvRot scales from the file's ``[out, 1]`` to ``(out,)``.

    ``Int8Linear`` registers ``weight_scale`` as ``(out_features,)`` and
    ``load_state_dict`` shape-checks even under ``assign=True``. Only
    marker-validated layers are touched (and the caller asserts each one got a
    ``ConvRotInt8Linear``), so this is not the blanket squeeze
    ``quantized_checkpoint_guard``'s docstring warns about. Same as
    minimax_h3/loader.py's.
    """
    reshaped = 0
    for layer in int8_convrot_source_layers:
        key = f"{layer}.weight_scale"
        scale = sd.get(key)
        if scale is not None and scale.dim() > 1:
            sd[key] = scale.reshape(-1)
            reshaped += 1
    return reshaped


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


# Decoder Linears per MoT half, and both together -- the counts
# ``iter_sensenova_lora_targets`` enumerates for each branch.
SENSENOVA_BRANCH_LINEAR_COUNTS = {"gen": 294, "und": 294, "both": 588}


def materialize_int8_decoder_linears(
    transformer: NEOChatModel,
    *,
    branch: str,
    dtype: torch.dtype = torch.bfloat16,
) -> int:
    """Replace one (or both) MoT half's ``Int8Linear`` decoder layers with ``nn.Linear``.

    TRAINING-ONLY, and only for full fine-tuning: ``Int8Linear`` holds its
    weight and scale as BUFFERS, so ``requires_grad_(True)`` is a no-op on them
    and an optimizer sees nothing -- the run trains no decoder parameter at all
    while the loss falls normally (SENSENOVA_TRAINING_DESIGN.md 6.1). This is
    that document's 6.4 supply route (a): each selected Linear's weight becomes
    ``int8_codes * weight_scale`` in ``dtype`` as an ``nn.Parameter``, spelled
    exactly as ``Int8Linear._dequant_forward`` spells it, so the materialized
    base computes the same function the frozen int8 base did at that dtype.

    PER-LINEAR ORDER IS LOAD-BEARING. Each module's int8 buffers are released
    before the next one is dequantized, so the peak is the resident base plus
    the materialized total plus ONE weight (48 MiB int8 -> 96 MiB bf16 at the
    largest of the real shapes), not the base plus a whole second copy of it.

    Plain int8 only. A ConvRot base is refused: its codes are Hadamard-rotated,
    so dequantizing them without inverting the rotation gives a wrong weight,
    and inverting it would compound with the train/inference activation skew in
    that document's 5.3.

    The class refusals below run over the whole scope BEFORE anything is
    replaced; the per-Linear scale-shape refusal cannot, so it can leave a
    partially materialized tree. That is deliberate -- holding the int8 modules
    alive to be able to roll back is exactly the second copy this ordering
    exists to avoid, and the refusal aborts the run at setup either way.

    Returns the number of Linears materialized.
    """
    from core.models.common.convrot_int8_linear import ConvRotInt8Linear
    from core.models.ideogram4.vendor.int8_linear import Int8Linear

    from .sensenova_lora import iter_sensenova_lora_targets

    expected = SENSENOVA_BRANCH_LINEAR_COUNTS.get(branch)
    if expected is None:
        raise ValueError(
            f"Unknown SenseNova materialization branch {branch!r} "
            f"(expected one of {sorted(SENSENOVA_BRANCH_LINEAR_COUNTS)})"
        )
    if not dtype.is_floating_point:
        raise ValueError(
            f"SenseNova materialization needs a floating-point dtype for the "
            f"trainable weights; got {dtype}"
        )

    targets = list(iter_sensenova_lora_targets(transformer, branch=branch))
    if len(targets) != expected:
        raise RuntimeError(
            f"SenseNova {branch} materialization found {len(targets)} decoder "
            f"Linear(s), expected {expected}. The loaded tree is not the "
            f"42-layer MoT decoder this route was built for; refusing rather "
            f"than materializing a partial half."
        )

    convrot = [path for path, _, _, m in targets if isinstance(m, ConvRotInt8Linear)]
    if convrot:
        raise RuntimeError(
            f"SenseNova full fine-tuning cannot materialize a ConvRot-quantized base: "
            f"{len(convrot)} of {len(targets)} {branch} decoder Linear(s) are "
            f"ConvRotInt8Linear (first: {convrot[0]}). Their int8 codes are "
            f"Hadamard-rotated, so dequantizing them without inverting the rotation "
            f"produces a wrong weight. Remedy: select the plain-int8 SenseNova "
            f"checkpoint for full fine-tuning, or keep training_method='lora', which "
            f"trains on either quantized base."
        )
    other = [path for path, _, _, m in targets if type(m) is not Int8Linear]
    if other:
        raise RuntimeError(
            f"SenseNova {branch} materialization requires every one of the "
            f"{len(targets)} decoder Linears to be a plain Int8Linear, but "
            f"{len(other)} is/are not (first: {other[0]}). This base is not the "
            f"weight-only int8 checkpoint this repo distributes."
        )

    materialized = 0
    released_bytes = 0
    allocated_bytes = 0
    for index in range(len(targets)):
        module_path, parent, attr, module = targets[index]
        # Drop the list's reference so this module dies at the setattr below and
        # its int8 buffers are freed before the next one is dequantized.
        targets[index] = None

        scale = module.weight_scale
        if scale.dim() != 1 or scale.shape[0] != module.out_features:
            raise RuntimeError(
                f"SenseNova {module_path} carries a weight_scale of shape "
                f"{tuple(scale.shape)}; Int8Linear registers it as (out_features,) = "
                f"({module.out_features},). Refusing rather than reshaping -- this is "
                f"the blanket squeeze _reshape_convrot_scales warns against, and a "
                f"mis-shaped scale would broadcast into a silently wrong weight."
            )
        # The spelling Int8Linear._dequant_forward uses, so the materialized
        # weight is bitwise the tensor that layer built on every call.
        weight = module.weight * scale.to(dtype).unsqueeze(1)
        bias = None if module.bias is None else module.bias.detach().to(dtype)
        released_bytes += module.weight.numel() * module.weight.element_size()
        allocated_bytes += weight.numel() * weight.element_size()

        linear = nn.Linear(
            module.in_features, module.out_features, bias=bias is not None, device="meta"
        )
        linear.weight = nn.Parameter(weight)
        if bias is not None:
            linear.bias = nn.Parameter(bias)
        setattr(parent, attr, linear)
        del module, weight, bias
        materialized += 1

    print(
        f"[SenseNovaLoader] materialized {materialized} {branch} decoder Int8Linear(s) "
        f"to {dtype} nn.Linear parameters: released {released_bytes / 2**20:.1f} MiB of "
        f"int8 codes, allocated {allocated_bytes / 2**20:.1f} MiB of weights"
    )
    return materialized


# Loose sibling files ``_load_sensenova_tokenizer`` and the ``config.json``
# fallback need next to a checkpoint. Copied beside a training save so the
# result loads through the same path the distributed checkpoint does.
SENSENOVA_SIBLING_FILES = (
    "tokenizer_config.json", "vocab.json", "merges.txt",
    "special_tokens_map.json", "added_tokens.json", "config.json",
)


def _sensenova_branch_halves(branch: str) -> "tuple[str, ...]":
    if branch == "both":
        return ("gen", "und")
    if branch in ("gen", "und"):
        return (branch,)
    raise ValueError(
        f"Unknown SenseNova branch {branch!r} "
        f"(expected one of {sorted(SENSENOVA_BRANCH_LINEAR_COUNTS)})"
    )


def _assert_scale_weight_conjunction(weight_dtypes: Dict[str, torch.dtype],
                                     scale_stems: "set[str]") -> None:
    """The loader's own per-Linear gate, asserted over what is about to be written.

    ``swap_linears_to_int8`` takes a Linear only when its ``.weight`` is int8 AND
    a ``.weight_scale`` sits beside it, and ``verify_quantized_swap`` then
    demands swapped == scale keys == int8 weight keys. So a bf16 weight that kept
    its stale scale is refused on read -- by a message describing the INVERSE
    defect ("a scale-less (or partially scale-less) file cannot be read back"),
    which is unfindable from the symptom. Checked here instead, where the cause
    is in hand.
    """
    stale = sorted(s for s in scale_stems if weight_dtypes.get(s) is not torch.int8)
    missing = sorted(s for s, d in weight_dtypes.items()
                     if d is torch.int8 and s not in scale_stems)
    if not stale and not missing:
        return
    raise RuntimeError(
        f"SenseNova checkpoint write would violate the per-Linear rule its own "
        f"loader reads back with: {len(stale)} Linear(s) carry a weight_scale "
        f"beside a non-int8 weight (first: {stale[:3] or None}) and "
        f"{len(missing)} carry an int8 weight with no scale (first: "
        f"{missing[:3] or None}). A dequantized weight's scale is meaningless "
        f"and must be dropped with it."
    )


def _embeddable_sensenova_config(
    config: Any, source_dir: Optional[str], raw_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """The geometry block to embed: the block THIS LOAD accepted, verbatim, if there is one.

    ``raw_config`` is what ``load_sensenova_from_path`` parsed -- the
    checkpoint's own ``sensenova_config`` metadata when it had one, the sibling
    ``config.json`` otherwise. Preferring it is exact AND safe: the reader
    prefers embedded metadata over the sibling, so re-deriving from the sibling
    could embed a different dict than the run was built from; and a dict that
    reached here necessarily reconstructed a config and a model already, so it
    cannot be one of the unreadable forms below. The sibling is the fallback,
    and a re-serialization the last resort.

    ``NEOChatConfig.to_dict()`` is not a fixed point of ``NEOChatConfig(**.)`` in
    this vendor tree, for at least two independent reasons, and each one writes a
    file that only fails when it is read back:

    * the ``to_dict`` override skips the base class's dtype normalization, so the
      top-level ``dtype`` serializes as ``"torch.bfloat16"`` and reloads as
      ``getattr(torch, "torch.bfloat16")`` -- see ``_serializable_sensenova_config``;
    * ``configuration_neo_vit.py:38`` assigns ``self.downsample_ratio =
      downsample_ratio,`` -- a trailing comma, so the value is a 1-tuple that the
      ViT reads as ``downsample_ratio[0]``. Serialized it becomes ``[0.5]`` and
      the next construction makes it ``([0.5],)``, whose ``[0]`` is a list.
      NOT repaired here: the tuple is load-bearing at inference.

    A dict that came off disk has neither problem -- it is what the shipped
    loader reads on every load -- so the export carries it through unchanged
    instead of round-tripping the live object. ``link_siblings`` copies the
    sibling ``config.json`` beside the checkpoint too, so the embedded block and
    the sibling fallback agree by construction rather than by luck.
    """
    if raw_config:
        return dict(raw_config)
    if source_dir:
        path = os.path.join(source_dir, "config.json")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as handle:
                return json.load(handle)
    return _serializable_sensenova_config(config)


def _serializable_sensenova_config(config: Any) -> Dict[str, Any]:
    """``config.to_dict()`` with its dtypes turned into the strings a reload accepts.

    The vendored ``NEOChatConfig.to_dict`` OVERRIDES the base implementation and
    copies ``__dict__`` verbatim, so it skips the base class's dtype
    normalization and leaves the top-level ``dtype`` a real ``torch.dtype``.
    ``sensenova_export_metadata`` then JSON-dumps it with ``default=str``, which
    writes ``"torch.bfloat16"`` -- and ``PreTrainedConfig.__init__`` reads that
    back as ``getattr(torch, "torch.bfloat16")`` and raises. The nested
    ``vision_config`` / ``llm_config`` are unaffected because they use the stock
    ``to_dict``, which is why only one key of the three was ever wrong.

    The base class's own normalizer is called rather than reimplemented; it was
    renamed in transformers 5 (``dict_torch_dtype_to_str`` ->
    ``dict_dtype_to_str``), so both names are tried.
    """
    config_dict = config.to_dict() if hasattr(config, "to_dict") else dict(config or {})
    for name in ("dict_dtype_to_str", "dict_torch_dtype_to_str"):
        normalize = getattr(config, name, None)
        if callable(normalize):
            normalize(config_dict)
            break
    return config_dict


def _assert_config_metadata_reloads(metadata: Dict[str, str]) -> None:
    """Refuse to write a checkpoint whose own geometry metadata cannot be read.

    ``sensenova_config`` is the loader's PRIMARY geometry source, and the write
    happens hours into a run: a value that only fails on read turns a completed
    fine-tune into an unloadable file.

    NOT the whole read path: the reader also builds ``NEOChatModel(config)``
    under ``init_empty_weights``, and this checks only the config construction
    plus the one arithmetic that constructor does on a config value. Both known
    round-trip defects live in those two steps, but a third one further inside
    the model constructor would still reach disk. Kept narrow deliberately --
    building the meta-device module graph on every ``save_every`` would be paid
    by every run to guard a class of defect no run has hit.
    """
    raw = metadata.get("sensenova_config")
    if not raw:
        return
    try:
        reconstructed = NEOChatConfig(**json.loads(raw))
        # The one arithmetic NEOChatModel's constructor does on a config value,
        # and the one field this vendor tree does not round-trip
        # (_embeddable_sensenova_config). Reconstructing the config alone does
        # not raise on it; building the model does, 25 GiB later.
        float(1.0 / reconstructed.vision_config.downsample_ratio[0])
    except Exception as exc:
        raise RuntimeError(
            f"SenseNova save produced a 'sensenova_config' metadata block that "
            f"its own loader cannot reconstruct ({type(exc).__name__}: {exc}). "
            f"Refusing to write: this key is the checkpoint's primary geometry "
            f"source, so the file would be unreadable."
        ) from exc


def save_sensenova_full_finetune_checkpoint(
    transformer: NEOChatModel,
    output_path: str,
    *,
    branch: str,
    save_format: str,
    config: Any = None,
    raw_config: Optional[Dict[str, Any]] = None,
    extra_metadata: Optional[Dict[str, str]] = None,
    source_dir: Optional[str] = None,
    max_shard_bytes: Optional[int] = None,
) -> "tuple[str, Dict[str, Any]]":
    """Write a full-fine-tuned SenseNova model this loader can read back.

    ``branch`` is the half (or halves) a full fine-tune materialized;
    ``save_format`` is one of ``param_defaults.SENSENOVA_FULL_FINETUNE_SAVE_FORMATS``
    (SENSENOVA_TRAINING_DESIGN.md 6.4):

    * ``mixed`` -- trained half bf16, untrained half's int8 codes and scales
      passed through untouched. With BOTH halves trained there is no int8 half
      left, so this degenerates into ``bf16``; the effective format is returned
      and recorded in metadata rather than silently mislabelled.
    * ``bf16``   -- both halves floating point. The untrained half is dequantized
      here, by the same spelling ``materialize_int8_decoder_linears`` uses, so it
      computes the function the int8 half computed at this dtype.
    * ``int8``   -- the trained half is requantized with the repo's own
      ``quantize_weight_to_int8``, giving back the distributed layout. LOSSY:
      any update below half a grid step is discarded, and an untouched weight is
      re-rounded too.

    Streamed, never assembled: one shard buffer plus one tensor of host cost,
    not a second copy of a 16.2 B-parameter model.

    COMPLETENESS IS THE WRITER'S JOB, because the read path accepts any subset
    of the 588 as materialized -- a half-written half loads clean and is a
    valid, wrong model. Counted on the live tree, counted again over the emitted
    keys, and committed atomically (provisional shard names, index last).

    Returns ``(written path, census)``.
    """
    from api.param_defaults import SENSENOVA_FULL_FINETUNE_SAVE_FORMATS
    from core.models.common.quantized_export import (
        DEFAULT_EXPORT_SHARD_BYTES, ShardWriter, link_siblings,
        sensenova_export_metadata,
    )
    from core.models.ideogram4.vendor.int8_linear import (
        Int8Linear, quantize_weight_to_int8,
    )

    from .sensenova_lora import iter_sensenova_lora_targets

    if save_format not in SENSENOVA_FULL_FINETUNE_SAVE_FORMATS:
        raise ValueError(
            f"Unknown SenseNova save format {save_format!r}; supported: "
            f"{', '.join(SENSENOVA_FULL_FINETUNE_SAVE_FORMATS)}"
        )
    trained_halves = _sensenova_branch_halves(branch)
    frozen_halves = tuple(h for h in ("gen", "und") if h not in trained_halves)
    effective = "bf16" if (save_format == "mixed" and not frozen_halves) else save_format

    trained: Dict[str, nn.Module] = {}
    for half in trained_halves:
        for path, _parent, _attr, module in iter_sensenova_lora_targets(transformer, branch=half):
            trained[path] = module
    frozen: Dict[str, nn.Module] = {}
    for half in frozen_halves:
        for path, _parent, _attr, module in iter_sensenova_lora_targets(transformer, branch=half):
            frozen[path] = module

    expected_trained = sum(SENSENOVA_BRANCH_LINEAR_COUNTS[h] for h in trained_halves)
    expected_frozen = sum(SENSENOVA_BRANCH_LINEAR_COUNTS[h] for h in frozen_halves)
    if len(trained) != expected_trained or len(frozen) != expected_frozen:
        raise RuntimeError(
            f"SenseNova save enumerated {len(trained)} trained and {len(frozen)} "
            f"frozen decoder Linear(s) on branch {branch!r}, expected "
            f"{expected_trained} and {expected_frozen}. Refusing to write a "
            f"checkpoint from a tree that is not the 42-layer MoT decoder."
        )
    unmaterialized = sorted(
        path for path, module in trained.items()
        if not isinstance(getattr(module, "weight", None), nn.Parameter)
    )
    if unmaterialized:
        raise RuntimeError(
            f"SenseNova save found {len(unmaterialized)} of {len(trained)} "
            f"{branch}-branch decoder Linear(s) still holding an int8 buffer "
            f"(first: {unmaterialized[0]}). Those layers were never trained; "
            f"writing them would produce a file that loads clean and silently "
            f"carries the base weights for part of the half."
        )
    not_int8 = sorted(path for path, module in frozen.items() if type(module) is not Int8Linear)
    if not_int8:
        raise RuntimeError(
            f"SenseNova save expects the untrained half's {len(frozen)} decoder "
            f"Linear(s) to be plain Int8Linear, but {len(not_int8)} is/are not "
            f"(first: {not_int8[0]})."
        )

    output_path = str(output_path)
    if not output_path.endswith(".safetensors"):
        output_path += ".safetensors"
    directory = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(directory, exist_ok=True)

    config_dict = _embeddable_sensenova_config(config, source_dir, raw_config)
    metadata = dict(sensenova_export_metadata(config_dict))
    _assert_config_metadata_reloads(metadata)
    metadata["sensenova_trained_branch"] = branch
    metadata["sensenova_save_format"] = effective
    metadata["sensenova_save_format_requested"] = save_format
    for key, value in (extra_metadata or {}).items():
        metadata[str(key)] = str(value)

    writer = ShardWriter(
        output_path, metadata,
        int(max_shard_bytes or DEFAULT_EXPORT_SHARD_BYTES),
    )
    # What the frozen half is dequantized into, taken from the trained half so
    # both ends of a bf16 file carry one dtype rather than a hardcoded guess.
    float_dtype = next(iter(trained.values())).weight.dtype
    if not float_dtype.is_floating_point:
        raise RuntimeError(
            f"SenseNova save expects the materialized half to hold floating-point "
            f"weights; got {float_dtype}"
        )
    weight_dtypes: Dict[str, torch.dtype] = {}
    scale_stems: "set[str]" = set()
    census = {"trained_bf16": 0, "trained_int8": 0, "frozen_int8": 0,
              "frozen_bf16": 0, "other": 0}
    try:
        for key, tensor in transformer.state_dict().items():
            stem, _, leaf = key.rpartition(".")
            tensor = tensor.detach()
            if stem in trained:
                if leaf == "weight_scale":
                    raise RuntimeError(
                        f"SenseNova save found a weight_scale on the materialized "
                        f"Linear {stem}; a dequantized weight has no scale."
                    )
                if leaf == "weight":
                    if effective == "int8":
                        codes, scale = quantize_weight_to_int8(tensor)
                        writer.add(f"{TRANSFORMER_PREFIX}{key}", codes.cpu().contiguous())
                        writer.add(f"{TRANSFORMER_PREFIX}{stem}.weight_scale",
                                   scale.cpu().contiguous())
                        weight_dtypes[stem] = torch.int8
                        scale_stems.add(stem)
                        census["trained_int8"] += 1
                        continue
                    writer.add(f"{TRANSFORMER_PREFIX}{key}", tensor.cpu().contiguous())
                    weight_dtypes[stem] = tensor.dtype
                    census["trained_bf16"] += 1
                    continue
            elif stem in frozen:
                if effective == "bf16":
                    if leaf == "weight_scale":
                        continue  # meaningless beside the dequantized weight
                    if leaf == "weight":
                        # The spelling materialize_int8_decoder_linears and
                        # Int8Linear._dequant_forward both use, including its
                        # scale-shape refusal: a mis-shaped scale broadcasts
                        # into a silently wrong weight rather than raising.
                        module = frozen[stem]
                        scale = module.weight_scale
                        if scale.dim() != 1 or scale.shape[0] != module.out_features:
                            raise RuntimeError(
                                f"SenseNova {stem} carries a weight_scale of shape "
                                f"{tuple(scale.shape)}; Int8Linear registers it as "
                                f"(out_features,) = ({module.out_features},). Refusing "
                                f"rather than reshaping -- this is the blanket squeeze "
                                f"_reshape_convrot_scales warns against."
                            )
                        weight = tensor * scale.to(float_dtype).unsqueeze(1)
                        writer.add(f"{TRANSFORMER_PREFIX}{key}", weight.cpu().contiguous())
                        weight_dtypes[stem] = weight.dtype
                        census["frozen_bf16"] += 1
                        del weight
                        continue
                elif leaf == "weight_scale":
                    scale_stems.add(stem)
                elif leaf == "weight":
                    weight_dtypes[stem] = tensor.dtype
                    census["frozen_int8"] += 1
            else:
                if leaf == "weight_scale":
                    raise RuntimeError(
                        f"SenseNova save found a quantized Linear outside the 588 "
                        f"decoder targets ({stem}); this tree is not the base this "
                        f"route was built for."
                    )
                census["other"] += 1
            writer.add(f"{TRANSFORMER_PREFIX}{key}", tensor.cpu().contiguous())

        int8_expected = expected_trained if effective == "int8" else 0
        bf16_expected = 0 if effective == "int8" else expected_trained
        frozen_int8_expected = 0 if effective == "bf16" else expected_frozen
        frozen_bf16_expected = expected_frozen if effective == "bf16" else 0
        if (census["trained_int8"], census["trained_bf16"],
                census["frozen_int8"], census["frozen_bf16"]) != (
                int8_expected, bf16_expected, frozen_int8_expected, frozen_bf16_expected):
            raise RuntimeError(
                f"SenseNova {effective} save wrote {census} decoder Linear(s), "
                f"expected trained_int8={int8_expected}, trained_bf16={bf16_expected}, "
                f"frozen_int8={frozen_int8_expected}, frozen_bf16={frozen_bf16_expected} "
                f"for branch {branch!r}. A partial half loads without a warning, so "
                f"the write is refused instead of committed."
            )
        _assert_scale_weight_conjunction(weight_dtypes, scale_stems)
    except BaseException:
        writer.abort()
        raise
    written = writer.close()

    if source_dir and os.path.isdir(source_dir):
        link_siblings(source_dir, directory, names=SENSENOVA_SIBLING_FILES)
    print(
        f"[SenseNovaLoader] saved {branch} full fine-tune checkpoint "
        f"(format={effective}"
        + (f", requested={save_format}" if effective != save_format else "")
        + f"): {census['trained_bf16']} bf16 + {census['trained_int8']} int8 trained "
        f"Linear(s), {census['frozen_bf16']} bf16 + {census['frozen_int8']} int8 "
        f"frozen Linear(s) -> {written}"
    )
    return written, {**census, "effective_format": effective, "branch": branch}


def install_sensenova_state_dict(
    model: nn.Module,
    sd: Dict[str, torch.Tensor],
    int8_convrot_source_layers: Dict[str, Dict[str, int]],
    torch_dtype: torch.dtype,
    *,
    path: Optional[str] = None,
) -> int:
    """Guard, census, quantized-Linear swap and ``assign=True`` load. Returns the swap count.

    The whole of what ``load_sensenova_from_path`` does to a freshly built
    (meta-device) tree once the tensors are in hand, factored out so that
    anything asserting a written checkpoint can be read back exercises THIS
    code rather than a re-spelling of it.
    """
    # The caller's early marker read validated every supported ConvRot marker;
    # every OTHER declared-semantics marker must still refuse. See
    # ``_sensenova_quant_dict_views`` for what each of the three views means.
    guard_sd, plain_sd, sd_for_load = _sensenova_quant_dict_views(sd, int8_convrot_source_layers)
    refuse_unsupported_quant_semantics(guard_sd, arch=ARCH_LABEL, path=path, label="transformer")

    # Census + verify BEFORE the swap+load, same discipline as every other
    # quantized loader in this repo (Anima/Ideogram 4): a scale-less or
    # partially-matched quantized file must refuse rather than silently cast
    # int8 codes into a bf16 parameter.
    census = quantized_state_dict_report(plain_sd, arch=ARCH_LABEL, path=path, label="transformer")
    quant_report = scaled_quantization_report(census, arch=ARCH_LABEL, path=path, label="transformer")

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
    verify_quantized_swap(quant_report, plain_swapped, arch=ARCH_LABEL, path=path, label="transformer")
    swapped += plain_swapped

    missing, unexpected = model.load_state_dict(sd_for_load, strict=False, assign=True)
    if missing:
        print(f"[SenseNovaLoader] WARNING: {len(missing)} missing key(s); first 5: {missing[:5]}")
    if unexpected:
        print(f"[SenseNovaLoader] WARNING: {len(unexpected)} unexpected key(s); first 5: {unexpected[:5]}")
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

    config, config_dict = _load_sensenova_config(metadata, model_dir)

    # ConvRot markers, read from the still-plain state dict before anything is
    # installed (mirrors MiniMax-H3 DiT loader.py's ordering: the runtime
    # requirement is checked before any payload is swapped in or loaded).
    int8_convrot_source_layers = _int8_convrot_source_layers(sd, path=model_path)
    if int8_convrot_source_layers:
        from core.models.common.convrot_int8_linear import require_convrot_int8_runtime

        require_convrot_int8_runtime()

    _reshape_convrot_scales(sd, int8_convrot_source_layers)

    with init_empty_weights():
        model = NEOChatModel(config)
        model.to(torch_dtype)

    swapped = install_sensenova_state_dict(
        model, sd, int8_convrot_source_layers, torch_dtype, path=model_path
    )

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
        # Additive, for two callers that need the file's own words rather than a
        # re-derivation: the training guard names the save format a refused tree
        # was written with, and an export re-embeds the exact geometry block this
        # load accepted.
        "metadata": dict(metadata or {}),
        "config_dict": config_dict,
    }


def _load_sensenova_tokenizer(model_dir: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.encode("validation", add_special_tokens=False)  # sanity encode
    return tokenizer
