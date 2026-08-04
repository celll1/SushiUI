"""Component loader for Ideogram 4 models.

Loads each sub-model to CPU; pipeline.py stages them to GPU per generation phase.
Expects a local diffusers-layout directory (the same layout published on the Hub):

    <model_path>/
        model_index.json
        transformer/                 Ideogram4Transformer2DModel  (conditional branch)
        unconditional_transformer/   Ideogram4Transformer2DModel  (asymmetric-CFG branch)
        text_encoder/                Qwen3VLModel                 (weight-only FP8)
        tokenizer/                   Qwen2Tokenizer
        vae/                         AutoencoderKLFlux2
        scheduler/                   FlowMatchEulerDiscreteScheduler

Ideogram 4 uses asymmetric classifier-free guidance: a conditional `transformer`
and a separate `unconditional_transformer` are both required at inference time.
"""

from __future__ import annotations

import json
import os
import re

import torch

from .vendor import (
    Ideogram4Transformer2DModel,
    is_bnb4bit_state_dict,
    is_fp8_state_dict,
    is_int8_state_dict,
    load_bnb4bit_state_dict,
    load_fp8_state_dict,
    swap_linears_to_bnb4bit,
    swap_linears_to_fp8,
    swap_linears_to_int8,
)
from .vendor.text_encoder import load_ideogram4_text_encoder


# The sharded-component reader was promoted to core.models.common; this thin
# re-export keeps the historical import path (krea2_loader imports it from here).
from core.models.common.single_file_format import (
    load_component_state_dict as _load_component_state_dict,
)


# The optional leading group is a WRAPPER PREFIX, not decoration. The loader
# always hands these bare module paths (it strips prefixes before it converts),
# but the offline quantizer's ``source_transform`` runs BEFORE prefix stripping
# by contract, and a COMBINED Ideogram 4 single file carries every key under
# ``transformer.`` / ``unconditional_transformer.``. An anchored pattern silently
# matched none of them: the fused qkv keys passed through unsplit and 135 of the
# 241 selectable Linears per transformer were quietly left unquantized. The
# prefix is captured and re-emitted, so a transformed key keeps the layout it
# arrived in -- which is exactly what the contract asks for.
_FUSED_QKV_RE = re.compile(r"^(.*?\.)?(layers\.\d+\.attention)\.qkv\.(weight|weight_scale|bias)$")
_FUSED_O_RE = re.compile(r"^(.*?\.)?(layers\.\d+\.attention)\.o\.(weight|weight_scale|bias)$")


def ideogram4_fused_qkv_to_split(key: str, tensor=None):
    """One checkpoint key -> the module path(s) it holds, in the ``source_transform`` shape.

    ``(key, tensor) -> ((key', tensor'), ...)``, per
    ``core.models.common.quantized_export.EXPORT_LAYOUTS``' contract:

    * ``layers.N.attention.qkv.{weight,weight_scale,bias}`` fans out to
      ``to_q`` / ``to_k`` / ``to_v``; the split is along the OUTPUT rows, which
      is exact for the weight, for the per-row ``weight_scale`` and for the bias
      alike -- and is why quantizing the fused tensor and splitting it afterwards
      would give bit-identical results (the rows are independent). The offline
      artifact splits FIRST because a RUNTIME export sees the live module, which
      the loader has already split;
    * ``layers.N.attention.o.*`` renames to ``to_out.0.*``;
    * every other key passes through untouched, so a checkpoint already in the
      split (diffusers) layout is unchanged.

    ``tensor=None`` is the key-enumeration pass and returns the same key set with
    ``None`` tensors. The row count is taken from the tensor itself
    (``shape[0] // 3``) rather than from a config, so the function needs no
    geometry: a fused qkv tensor is 3 x hidden rows by construction.
    """
    m_qkv = _FUSED_QKV_RE.match(key)
    if m_qkv is not None:
        wrapper = m_qkv.group(1) or ""
        prefix, suffix = f"{wrapper}{m_qkv.group(2)}", m_qkv.group(3)
        names = (f"{prefix}.to_q.{suffix}", f"{prefix}.to_k.{suffix}", f"{prefix}.to_v.{suffix}")
        if tensor is None:
            return tuple((n, None) for n in names)
        rows = tensor.shape[0]
        if rows % 3:
            raise ValueError(
                f"[Ideogram4Loader] {key}: a fused qkv tensor must have 3 x hidden rows, "
                f"got {rows}")
        h = rows // 3
        parts = (tensor[:h], tensor[h:2 * h], tensor[2 * h:3 * h])
        return tuple((n, p.contiguous()) for n, p in zip(names, parts))
    m_o = _FUSED_O_RE.match(key)
    if m_o is not None:
        return ((f"{m_o.group(1) or ''}{m_o.group(2)}.to_out.0.{m_o.group(3)}", tensor),)
    return ((key, tensor),)


def _convert_fused_qkv_to_split(state_dict: dict, hidden_size: int) -> dict:
    """Remap native fused-QKV attention keys to the diffusers split layout.

    The ``ideogram-4-fp8`` checkpoint stores attention as a fused
    ``layers.N.attention.qkv`` (out = 3*hidden, ordered q,k,v) plus
    ``layers.N.attention.o``. The vendored diffusers transformer uses split
    ``to_q``/``to_k``/``to_v`` and ``to_out.0``.

    No-op when the checkpoint already uses the split (diffusers) layout.

    The per-key rule lives in ``ideogram4_fused_qkv_to_split`` above, which the
    offline quantizer's ``source_transform`` also calls -- ONE remap table, so an
    offline artifact cannot land on keys this loader would not produce.
    ``hidden_size`` (from the config) is no longer needed to perform the split,
    which the tensor's own row count determines, but it is still CHECKED against
    it: a config whose geometry disagrees with the checkpoint would otherwise
    produce three plausibly-shaped tensors that fit no module.
    """
    if not any(_FUSED_QKV_RE.match(k) for k in state_dict):
        return state_dict

    new_sd: dict = {}
    for k, v in state_dict.items():
        if _FUSED_QKV_RE.match(k) is not None and v.shape[0] != 3 * hidden_size:
            raise ValueError(
                f"[Ideogram4Loader] {k}: fused qkv has {v.shape[0]} rows, but the config's "
                f"geometry implies 3 x {hidden_size}. The checkpoint and the config disagree.")
        for new_key, value in ideogram4_fused_qkv_to_split(k, v):
            new_sd[new_key] = value
    return new_sd


def _swap_ideogram4_quantized_linears(model, state_dict: dict, dtype: torch.dtype,
                                      label: str) -> int:
    """Replace the ``nn.Linear``s that have a quantized saved weight. Returns the count.

    INT8 and e4m3 are detected INDEPENDENTLY and both swaps run, because
    ``quantize_transformer_fp8.py --format int8`` emits a MIXED checkpoint on
    purpose: a layer whose per-row crest factor makes int8 worse than e4m3 falls
    back to e4m3 in the same file. Each detector and each swap helper gates on
    the weight DTYPE as well as the shared ``.weight_scale`` suffix, so neither
    can claim the other's layers and the call order does not matter. Same
    helpers, same reasoning as ``krea2/vendor/single_file.build_krea2_transformer``
    and ``model_loader._swap_flux2_quantized_linears``; the historical Ideogram 4
    path called only the FP8 half, which is correct for the published FP8
    checkpoint and would have silently mis-loaded an int8 one.

    The count is NOT decorative: the caller compares it with the checkpoint's own
    quantized-key census (``verify_quantized_swap``).
    """
    has_int8 = bool(is_int8_state_dict(state_dict))
    has_fp8 = bool(is_fp8_state_dict(state_dict))
    if not (has_int8 or has_fp8):
        return 0
    n_int8 = swap_linears_to_int8(model, state_dict, compute_dtype=dtype) if has_int8 else 0
    n_fp8 = swap_linears_to_fp8(model, state_dict, compute_dtype=dtype) if has_fp8 else 0
    parts = []
    if n_int8:
        parts.append(f"{n_int8} Int8Linear")
    if n_fp8:
        parts.append(f"{n_fp8} Fp8Linear")
    print(f"[Ideogram4Loader] {label}: weight-only quantized checkpoint; swapped "
          f"{' + '.join(parts) or 'no'} Linear(s); the rest load as {dtype}")
    return n_int8 + n_fp8


def _build_ideogram4_transformer(
    model_path: str,
    subfolder: str,
    torch_dtype: torch.dtype,
) -> Ideogram4Transformer2DModel:
    """Build an Ideogram4Transformer2DModel and load its weights.

    Supports the native fused-QKV FP8 checkpoint (``ideogram-4-fp8``), the
    diffusers split-layout nf4 checkpoint (``ideogram-4-nf4-diffusers``), and a
    plain bf16 diffusers checkpoint. Naming and quantization are detected
    independently so any combination loads correctly.
    """
    component_dir = os.path.join(model_path, subfolder)
    with open(os.path.join(component_dir, "config.json"), encoding="utf-8") as f:
        config = json.load(f)

    state_dict = _load_component_state_dict(component_dir, "diffusion_pytorch_model")
    return _build_ideogram4_transformer_from_state(config, state_dict, torch_dtype, subfolder)


def _build_ideogram4_transformer_from_state(
    config: dict,
    state_dict: dict,
    torch_dtype: torch.dtype,
    label: str,
) -> Ideogram4Transformer2DModel:
    """Build an Ideogram4Transformer2DModel from an explicit config + state_dict.

    Shared by the directory loader (``_build_ideogram4_transformer``) and the
    single-file loader (``load_ideogram4_single_file``). Detects fused-QKV,
    nf4 (bitsandbytes) and weight-only FP8 layouts independently.
    """
    model = Ideogram4Transformer2DModel.from_config(config)
    hidden_size = int(config["attention_head_dim"]) * int(config["num_attention_heads"])

    state_dict = _convert_fused_qkv_to_split(state_dict, hidden_size)

    if is_bnb4bit_state_dict(state_dict):
        # bitsandbytes nf4 (4-bit) — requires CUDA; load directly to GPU.
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"[Ideogram4Loader] {label}: nf4 (bitsandbytes) weights require a CUDA device."
            )
        device = torch.device("cuda")
        swapped = swap_linears_to_bnb4bit(model, compute_dtype=torch_dtype)
        print(f"[Ideogram4Loader] {label}: swapped {swapped} Linear(s) to Linear4bit (nf4)")
        load_bnb4bit_state_dict(model, state_dict, device=device, dtype=torch_dtype)
        model.eval()
        return model

    # Weight-only quantized (int8 and/or e4m3, per-row ``.weight_scale``) is
    # DETECTED here and VERIFIED below. The report fires on either piece of
    # evidence -- a scale key or a quantized weight dtype -- while the swap
    # helpers require both, which is exactly the gap that used to let a
    # scale-less quantized file fall through to the plain-load branch: every
    # scale would be an unexpected key and every quantized code would be cast
    # into a bf16 parameter, silently. Run AFTER the nf4 branch above: a
    # bitsandbytes checkpoint stores uint8 weights with ``.absmax`` / ``.quant_state``
    # siblings and no ``.weight_scale`` at all, so it trips the report's dtype
    # half and has nothing this swap could take.
    # ``scaled_quantization_report`` narrows the census to the SCALED case: a
    # checkpoint whose float8 weights carry no scales at all is a plain dtype
    # cast, which the else branch below reads correctly (``load_state_dict``
    # casts into the bf16 parameter), so it must not be refused as though its
    # scales had been stripped.
    from core.models.common.quantized_checkpoint_guard import (
        quantized_state_dict_report, scaled_quantization_report, verify_quantized_swap,
    )

    quant_report = scaled_quantization_report(
        quantized_state_dict_report(state_dict), arch="Ideogram 4", label=label)
    if quant_report is not None:
        # Cast BEFORE the swap and skip any cast after it: ``nn.Module.to(dtype)``
        # rewrites every floating-point buffer, which would turn an e4m3 weight
        # buffer into bf16 -- value-preserving (all 256 codes convert exactly and
        # the dequant path still applies the scale) but it doubles the buffer and
        # drops ``Fp8Linear``'s ``_scaled_mm`` fast path, which gates on the
        # weight dtype. Same ordering and same reason as the FLUX.2 loader.
        model.to(torch_dtype)
        swapped = _swap_ideogram4_quantized_linears(model, state_dict, torch_dtype, label)
        verify_quantized_swap(quant_report, swapped, arch="Ideogram 4", label=label)
        load_fp8_state_dict(model, state_dict, device=torch.device("cpu"), dtype=torch_dtype)
    else:
        print(f"[Ideogram4Loader] {label}: loading plain (unquantized) weights")
        model.load_state_dict(state_dict)
        model.to(dtype=torch_dtype)

    model.eval()
    model.to("cpu")
    return model


def load_ideogram4_components(
    model_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    load_unconditional: bool = True,
) -> dict:
    """Load Ideogram 4 components from a local diffusers directory.

    Returns a component dict consumed by PipelineManager.load_model():
        {
            "type": "ideogram4",
            "transformer": Ideogram4Transformer2DModel,
            "unconditional_transformer": Ideogram4Transformer2DModel | None,
            "text_encoder": Qwen3VLModel,
            "tokenizer": PreTrainedTokenizer,
            "vae": AutoencoderKLFlux2,
            "scheduler": FlowMatchEulerDiscreteScheduler,
        }

    `load_unconditional=False` skips the unconditional transformer (used by
    LoRA training of the conditional branch only, to save ~9 GB).
    """
    from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
    from transformers import AutoTokenizer

    print(f"[Ideogram4Loader] Loading components from: {model_path}")

    print("[Ideogram4Loader] Loading transformer (conditional)...")
    transformer = _build_ideogram4_transformer(model_path, "transformer", torch_dtype)

    unconditional_transformer = None
    if load_unconditional:
        print("[Ideogram4Loader] Loading unconditional_transformer (asymmetric-CFG branch)...")
        unconditional_transformer = _build_ideogram4_transformer(
            model_path, "unconditional_transformer", torch_dtype
        )
    else:
        print("[Ideogram4Loader] Skipping unconditional_transformer (not requested)")

    print("[Ideogram4Loader] Loading text encoder (Qwen3-VL)...")
    text_encoder = load_ideogram4_text_encoder(model_path, torch_dtype=torch_dtype, device="cpu")

    print("[Ideogram4Loader] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, "tokenizer"))
    # Sanity encode: surface corrupted vocabulary files at load time rather than
    # at the first generation step where the error message is less informative.
    try:
        tokenizer.encode("validation", add_special_tokens=False)
    except Exception as e:
        raise RuntimeError(
            f"[Ideogram4Loader] Tokenizer sanity encode failed — vocabulary files may be corrupted "
            f"({model_path}/tokenizer): {e}"
        ) from e

    print("[Ideogram4Loader] Loading VAE (AutoencoderKLFlux2)...")
    vae = AutoencoderKLFlux2.from_pretrained(
        model_path, subfolder="vae", torch_dtype=torch_dtype, low_cpu_mem_usage=True
    )
    vae.eval()
    vae.to("cpu")

    print("[Ideogram4Loader] Loading scheduler (FlowMatchEulerDiscreteScheduler)...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")

    print("[Ideogram4Loader] All components loaded successfully.")
    return {
        "type": "ideogram4",
        "transformer": transformer,
        "unconditional_transformer": unconditional_transformer,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "vae": vae,
        "scheduler": scheduler,
    }


# ---------------------------------------------------------------------------
# Single-file (combined transformers) support
# ---------------------------------------------------------------------------

# Key prefixes used in the combined single-file save. The conditional branch is
# stored under ``transformer.`` and the asymmetric-CFG branch under
# ``unconditional_transformer.`` (both carry bare Ideogram4Transformer2DModel
# state-dict keys underneath).
COND_PREFIX = "transformer."
UNCOND_PREFIX = "unconditional_transformer."


def _resolve_ideogram4_base_dir(file_path: str, base_dir_hint: str = None) -> str:
    """Resolve a base Ideogram 4 diffusers directory (transformer/ + text_encoder/
    + tokenizer/ + vae/ + scheduler/ subfolders) for a combined single-file save.

    The single file bundles only the two transformers; the text encoder, VAE,
    tokenizer and scheduler are completed from a base diffusers directory.

    Search order (mirrors the Lens single-file resolver):
      1. ``base_dir_hint`` (from the file metadata / caller)
      2. ``settings.models_dir`` entries whose name contains "ideogram"
      3. ancestor directories of the file (up to 4 levels)
      4. sibling SUBDIRECTORIES of the file's parent (one level down): a
         root-level single file at ``M:/model/ideogram4/ideogram4_transformers.safetensors``
         thus finds the base dir ``M:/model/ideogram4/ideogram4/``.
    A directory qualifies when it contains ``transformer/config.json`` and a
    ``text_encoder/`` subfolder.
    """
    def _is_ide_dir(d: str) -> bool:
        return bool(d) and os.path.isdir(d) and os.path.isfile(
            os.path.join(d, "transformer", "config.json")
        ) and os.path.isdir(os.path.join(d, "text_encoder"))

    searched = []
    if base_dir_hint:
        searched.append(base_dir_hint)
        if _is_ide_dir(base_dir_hint):
            return base_dir_hint

    models_root = None
    try:
        from config.settings import settings
        models_root = getattr(settings, "models_dir", None)
    except Exception:
        models_root = None
    if models_root and os.path.isdir(models_root):
        for name in os.listdir(models_root):
            if "ideogram" in name.lower():
                cand = os.path.join(models_root, name)
                searched.append(cand)
                if _is_ide_dir(cand):
                    return cand

    p = os.path.abspath(file_path)
    for _ in range(4):
        p = os.path.dirname(p)
        if not p:
            break
        searched.append(p)
        if _is_ide_dir(p):
            return p

    parent = os.path.dirname(os.path.abspath(file_path))
    if parent and os.path.isdir(parent):
        sibling_matches = []
        for name in sorted(os.listdir(parent)):
            cand = os.path.join(parent, name)
            searched.append(cand)
            if _is_ide_dir(cand):
                sibling_matches.append(cand)
        if sibling_matches:
            sibling_matches.sort(key=lambda d: 0 if "ideogram" in os.path.basename(d).lower() else 1)
            return sibling_matches[0]

    raise FileNotFoundError(
        "Ideogram 4 combined single-file requires a base Ideogram 4 diffusers "
        "directory for its text encoder / VAE / tokenizer / scheduler, but none "
        "was found.\n"
        f"  File: {file_path}\n"
        "Searched (need 'transformer/config.json' + 'text_encoder/' inside):\n  - "
        + "\n  - ".join(searched or ["(nothing to search)"])
    )


def load_ideogram4_single_file(
    file_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    base_dir_hint: str = None,
    load_unconditional: bool = True,
) -> dict:
    """Load Ideogram 4 from a combined single-file save (both transformers).

    The two transformers come from ``file_path`` (keys split by the
    ``transformer.`` / ``unconditional_transformer.`` prefixes). Their configs are
    read from the file metadata (``transformer_config`` / ``unconditional_transformer_config``
    JSON) when present, else from the resolved base directory. The text encoder,
    VAE, tokenizer and scheduler are completed from the base diffusers directory
    (see ``_resolve_ideogram4_base_dir``).
    """
    from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
    from transformers import AutoTokenizer

    from core.models.common.single_file_format import read_state_dict, strip_prefix

    raw, md = read_state_dict(file_path)
    md = md or {}
    hint = base_dir_hint or md.get("component.base_dir") or md.get("sushi.base_model_path")
    base_dir = _resolve_ideogram4_base_dir(file_path, hint)
    print(f"[Ideogram4Loader] Combined single-file: {file_path}")
    print(f"[Ideogram4Loader] Resolved base Ideogram 4 directory: {base_dir}")

    cond_sd = strip_prefix(raw, COND_PREFIX)
    uncond_sd = strip_prefix(raw, UNCOND_PREFIX)
    if not cond_sd:
        raise ValueError(
            f"[Ideogram4Loader] No '{COND_PREFIX}*' keys in {file_path}; not a "
            f"combined Ideogram 4 single-file."
        )

    def _config_for(meta_key: str, subfolder: str) -> dict:
        if md.get(meta_key):
            try:
                return json.loads(md[meta_key])
            except Exception:
                pass
        with open(os.path.join(base_dir, subfolder, "config.json"), encoding="utf-8") as f:
            return json.load(f)

    print("[Ideogram4Loader] Building transformer (conditional) from single-file...")
    cond_cfg = _config_for("transformer_config", "transformer")
    transformer = _build_ideogram4_transformer_from_state(
        cond_cfg, cond_sd, torch_dtype, "transformer"
    )

    unconditional_transformer = None
    if load_unconditional and uncond_sd:
        print("[Ideogram4Loader] Building unconditional_transformer from single-file...")
        uncond_cfg = _config_for("unconditional_transformer_config", "unconditional_transformer")
        unconditional_transformer = _build_ideogram4_transformer_from_state(
            uncond_cfg, uncond_sd, torch_dtype, "unconditional_transformer"
        )
    elif not uncond_sd:
        print("[Ideogram4Loader] No unconditional_transformer keys in single-file (skipping)")

    print("[Ideogram4Loader] Loading text encoder (Qwen3-VL) from base directory...")
    text_encoder = load_ideogram4_text_encoder(base_dir, torch_dtype=torch_dtype, device="cpu")

    print("[Ideogram4Loader] Loading tokenizer from base directory...")
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(base_dir, "tokenizer"))

    print("[Ideogram4Loader] Loading VAE (AutoencoderKLFlux2) from base directory...")
    vae = AutoencoderKLFlux2.from_pretrained(
        base_dir, subfolder="vae", torch_dtype=torch_dtype, low_cpu_mem_usage=True
    )
    vae.eval()
    vae.to("cpu")

    print("[Ideogram4Loader] Loading scheduler from base directory...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(base_dir, subfolder="scheduler")

    print("[Ideogram4Loader] Combined single-file loaded successfully.")
    return {
        "type": "ideogram4",
        "transformer": transformer,
        "unconditional_transformer": unconditional_transformer,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "vae": vae,
        "scheduler": scheduler,
    }
