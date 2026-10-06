"""Component loader for LTX-2.3 (joint audio+video MM-DiT + Gemma-3 + LTX2 VAEs).

Phase 1a: make the model LOADABLE into SushiUI's single-in-memory model slot.
Video generation itself is Phase 1b.

Unlike the other DiT archs (Krea2/Ideogram4), the whole LTX-2 stack already
lives in the pinned venv diffusers 0.38.0, so this loader does NOT rebuild the
transformer from a config + bare state_dict. It simply calls
``LTX2Pipeline.from_pretrained(dir, torch_dtype=bf16)`` (bf16 halves the 46GB
fp32 Gemma-3 text encoder) and returns the resolved components in a dict shaped
like the other archs' return value, which ``PipelineManager.load_model``
consumes.

All components are kept on CPU after load (VRAM discipline). GPU staging happens
at generate time in P1b — either manually per phase, or via the pipeline's
``model_cpu_offload_seq`` (text_encoder -> connectors -> transformer -> vae ->
audio_vae -> vocoder).

WEIGHT-ONLY QUANTIZED TRANSFORMERS
----------------------------------
That "just call from_pretrained" shortcut is exactly wrong for ONE input: a
``transformer/`` component whose Linear weights are int8/e4m3 codes with
per-output-row ``.weight_scale`` siblings (this repo's
``subapps/fp8_quantize/quantize_transformer_fp8.py --arch ltx2``, or
``POST /models/export-quantized``). diffusers would drop every scale as an
unexpected key and cast the codes into bf16 parameters -- the silent
103020%-error failure ``core.models.common.quantized_checkpoint_guard``
documents. So the transformer component is CENSUSED FROM ITS HEADERS first (no
tensor bytes), and only when that census shows SCALED quantization is the
transformer built here -- ``init_empty_weights`` +
``swap_linears_to_int8``/``_fp8`` + ``load_state_dict(assign=True)`` -- and
handed to ``from_pretrained`` as a pre-built component. Every other input,
including a plain float8 CAST with no scales at all (the common ComfyUI
distribution shape), takes the untouched original path: diffusers' own
``torch_dtype`` cast reads that correctly and ``scaled_quantization_report``
says so rather than refusing it.

Only the ``transformer`` component is ever quantized. The Gemma-3 text encoder
in ``text_encoder/`` -- a 48-layer ``language_model.*`` plus a vision tower,
12.19 G of 2-D tensors against the DiT's 18.98 G -- is a different component
object in a different directory, and nothing here walks it;
``arch_capabilities`` separately declares ``text_encoder_quantization``
unsupported for ltx2.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# Diffusers component-directory weight basenames, in the order from_pretrained
# probes them. A quantized export written by ``quantized_export.ShardWriter``
# uses the user's chosen stem instead, so the resolver below falls back to "the
# only safetensors in the directory".
_COMPONENT_BASENAMES = ("diffusion_pytorch_model", "model")


def _resolve_transformer_dir(model_path: str) -> Optional[str]:
    """``<model_path>/transformer`` when it exists, else ``None``."""
    candidate = os.path.join(model_path, "transformer")
    return candidate if os.path.isdir(candidate) else None


def _transformer_shards(directory: str) -> Tuple[List[str], Dict[str, str], Optional[str]]:
    """``(shard paths, {key: shard path}, source)`` for an LTX-2.3 transformer component.

    Accepts the diffusers conventions (``diffusion_pytorch_model.safetensors``
    or its ``.index.json``) and, for an export whose stem the user chose, the
    single ``.safetensors`` / single ``.index.json`` present in the directory.
    Returns ``([], {}, None)`` when nothing readable is there, which makes the
    caller fall through to diffusers' own resolution and error message.

    ``source`` is the ONE path this resolution chose -- the index for a shard
    set, the file for a single -- and it is returned rather than re-derived
    because the caller has to read the same tensors the census read. It used to
    re-list the directory and take ``[0]``, so a directory holding two index
    files (a re-export beside the original, say) could census one shard set and
    then LOAD the other: the swap count would be verified against a file that
    was never opened.
    """
    from safetensors import safe_open

    index_path = None
    single_path = None
    for base in _COMPONENT_BASENAMES:
        cand_idx = os.path.join(directory, f"{base}.safetensors.index.json")
        cand_one = os.path.join(directory, f"{base}.safetensors")
        if os.path.isfile(cand_idx):
            index_path = cand_idx
            break
        if os.path.isfile(cand_one):
            single_path = cand_one
            break
    if index_path is None and single_path is None:
        indexes = [f for f in os.listdir(directory) if f.endswith(".safetensors.index.json")]
        singles = [f for f in os.listdir(directory) if f.endswith(".safetensors")]
        if len(indexes) == 1:
            index_path = os.path.join(directory, indexes[0])
        elif not indexes and len(singles) == 1:
            single_path = os.path.join(directory, singles[0])

    if index_path is not None:
        with open(index_path, encoding="utf-8") as fh:
            weight_map = (json.load(fh).get("weight_map") or {})
        key_to_shard = {k: os.path.join(directory, v) for k, v in weight_map.items()}
        return sorted(set(key_to_shard.values())), key_to_shard, index_path
    if single_path is not None:
        with safe_open(single_path, framework="pt", device="cpu") as fh:
            keys = list(fh.keys())
        return [single_path], {k: single_path for k in keys}, single_path
    return [], {}, None


def _header_dtype_table() -> Dict[str, torch.dtype]:
    """safetensors header dtype NAME -> ``torch.dtype``.

    Taken from safetensors' own table where it exposes one, so this cannot drift
    from what the library writes; the literal fallback covers only the dtypes the
    quantized guard cares about (the names are part of the on-disk format, not of
    the library's API, so they are stable). Same approach, same reason as the
    offline tool's ``_quantized_dtype_names``.

    SENTINEL. ``_TYPES`` is a private attribute, so it is accepted only after it
    ANSWERS CORRECTLY for a dtype whose spelling is fixed by the on-disk format:
    a future version that flipped the mapping to dtype -> name would still be a
    non-empty dict, every ``table.get(name)`` would return ``None``, every census
    would see zero quantized weights, and a quantized checkpoint would load as an
    ordinary one -- precisely the silent failure this whole path exists to
    prevent. Failing the sentinel falls back to the literals below.
    """
    try:
        import safetensors.torch as _st

        table = dict(getattr(_st, "_TYPES", None) or {})
        if table.get("BF16") is torch.bfloat16:
            return table
    except Exception:
        pass
    fallback = {"I8": torch.int8, "U8": torch.uint8, "BF16": torch.bfloat16,
                "F16": torch.float16, "F32": torch.float32}
    for name, attr in (("F8_E4M3", "float8_e4m3fn"), ("F8_E5M2", "float8_e5m2")):
        dtype = getattr(torch, attr, None)
        if dtype is not None:
            fallback[name] = dtype
    return fallback


def _quantization_census(shards: List[str], key_to_shard: Dict[str, str]):
    """``quantized_state_dict_report``'s answer, from the shard HEADERS only.

    Reads zero tensor bytes: one ``safe_open`` per shard plus
    ``get_slice(key).get_dtype()`` per key. The census itself is NOT
    reimplemented -- the headers are turned into zero-element tensors carrying
    the right dtype and handed to the shared
    ``quantized_state_dict_report``, so this and every other loader agree by
    construction about what a quantized file looks like.

    This matters here more than anywhere else: the alternative is materialising
    a 37 GB bf16 (or 19 GB int8) state dict just to find out whether it needed
    the quantized branch at all.
    """
    from safetensors import safe_open

    from core.models.common.quantized_checkpoint_guard import (
        COMFY_QUANT_MARKER_SUFFIX, PRE_QUANT_SCALE_SUFFIX, QUANT_SCALE_SUFFIX,
        quantized_state_dict_report,
    )

    # ``.comfy_quant`` / ``.pre_quant_scale`` are carried through as well, or the
    # DECLARED-semantics refusal inside ``quantized_state_dict_report`` would
    # have nothing to see on this arch: a Comfy-quantized shard set is
    # layout-identical to a supported one, and this loader never materialises a
    # full state dict to notice otherwise. The markers are tens of bytes, so
    # they are read FOR REAL (the rest stay zero-element dtype proxies) and the
    # refusal can name the declared format instead of reporting an undecodable
    # marker.
    _SUFFIXES = (QUANT_SCALE_SUFFIX, ".weight", COMFY_QUANT_MARKER_SUFFIX,
                 PRE_QUANT_SCALE_SUFFIX)
    table = _header_dtype_table()
    proxies: Dict[str, torch.Tensor] = {}
    by_shard: Dict[str, List[str]] = {}
    for key, shard in key_to_shard.items():
        if key.endswith(_SUFFIXES):
            by_shard.setdefault(shard, []).append(key)
    for shard, keys in by_shard.items():
        with safe_open(shard, framework="pt", device="cpu") as fh:
            for key in keys:
                if key.endswith(COMFY_QUANT_MARKER_SUFFIX):
                    proxies[key] = fh.get_tensor(key)
                    continue
                name = fh.get_slice(key).get_dtype()
                dtype = table.get(name)
                if dtype is None:
                    continue
                proxies[key] = torch.empty(0, dtype=dtype)
    return quantized_state_dict_report(proxies)


def _swap_ltx2_quantized_linears(model: nn.Module, sd: dict, dtype: torch.dtype) -> int:
    """Replace LTX-2.3 ``nn.Linear``s that have a quantized saved weight. Count.

    INT8 and e4m3 are detected INDEPENDENTLY and both swaps run, because
    ``--format int8`` emits a MIXED checkpoint on purpose: a layer whose measured
    int8 weight error is not better than its e4m3 one falls back to e4m3 in the
    same file. Each detector and each swap helper gates on the weight DTYPE as
    well as the shared ``.weight_scale`` suffix, so neither can claim the other's
    layers and the call order does not matter. Same helpers, same reasoning as
    ``model_loader._swap_flux2_quantized_linears`` and
    ``anima_loader._swap_quantized_linears``; LTX-2.3 needs no prefix argument
    because its component checkpoint carries bare module paths.

    The returned count is NOT decorative: the caller compares it against the
    header census (``verify_quantized_swap``) and refuses the load when they
    disagree, because a quantized layer this helper did not take is a layer whose
    codes ``load_state_dict`` would install (``assign=True``) or cast into a
    plain parameter without a word.
    """
    try:
        from core.models.ideogram4.vendor.int8_linear import (
            is_int8_state_dict, swap_linears_to_int8,
        )
        from core.models.ideogram4.vendor.fp8_linear import (
            is_fp8_state_dict, swap_linears_to_fp8,
        )
    except Exception as e:
        print(f"[LTX2Loader] weight-only quant support unavailable ({e}); "
              f"the checkpoint would load as a silently wrong model")
        raise
    has_int8 = bool(is_int8_state_dict(sd))
    has_fp8 = bool(is_fp8_state_dict(sd))
    if not (has_int8 or has_fp8):
        return 0
    n_int8 = swap_linears_to_int8(model, sd, compute_dtype=dtype) if has_int8 else 0
    n_fp8 = swap_linears_to_fp8(model, sd, compute_dtype=dtype) if has_fp8 else 0
    parts = []
    if n_int8:
        parts.append(f"{n_int8} Int8Linear")
    if n_fp8:
        parts.append(f"{n_fp8} Fp8Linear")
    print(f"[LTX2Loader] weight-only quantized LTX-2.3 transformer: swapped "
          f"{' + '.join(parts) or 'no'} Linear(s); the remaining Linears load as {dtype}")
    return n_int8 + n_fp8


def _transformer_config(directory: str, metadata: dict) -> dict:
    """The DiT geometry for a quantized component directory.

    ``config.json`` next to the weights first -- that is what
    ``from_pretrained`` itself would read, and what the export copies across --
    then the artifact's own ``transformer_config`` metadata blob
    (``quantized_export.ltx2_export_metadata``) as the self-describing fallback.
    There is deliberately NO compiled-in default: LTX-2.3 has published
    variants, and a guessed ``num_layers`` would build 1660 module paths that
    match no weight.
    """
    cand = os.path.join(directory, "config.json")
    if os.path.isfile(cand):
        with open(cand, encoding="utf-8") as fh:
            return json.load(fh)
    blob = (metadata or {}).get("transformer_config")
    if blob:
        config = json.loads(blob)
        if config:
            print(f"[LTX2Loader] no config.json in {directory}; using the artifact's "
                  f"own 'transformer_config' metadata")
            return config
    raise FileNotFoundError(
        f"the LTX-2.3 transformer component at {directory} is weight-only quantized, "
        f"which means it must be rebuilt from a config before its Linear layers can be "
        f"swapped -- but the directory has no config.json and the file carries no "
        f"'transformer_config' metadata. Copy the source model's "
        f"transformer/config.json next to the weights.")


def _source_transformer_config(directory):
    if os.path.isfile(os.path.join(directory, "config.json")):
        return _transformer_config(directory, {})
    from safetensors import safe_open
    _, _, source = _transformer_shards(directory)
    if source is None:
        raise ValueError("LTX-2 companion root requires native transformer config headers")
    if source.endswith(".index.json"):
        with open(source, encoding="utf-8") as handle:
            metadata = json.load(handle).get("metadata") or {}
    else:
        with safe_open(source, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
    return _transformer_config(directory, metadata)


def _load_quantized_ltx2_transformer(model_path: str, torch_dtype: torch.dtype):
    """The pre-built ``transformer`` component, or ``None`` to let diffusers load it.

    ``None`` is returned for every ordinary checkpoint AND for a plain float8
    cast with no scales, which diffusers reads correctly by casting back --
    ``scaled_quantization_report`` draws that line and prints why.
    """
    directory = _resolve_transformer_dir(model_path)
    if directory is None:
        return None
    shards, key_to_shard, source = _transformer_shards(directory)
    if not key_to_shard or source is None:
        return None

    from core.models.common.quantized_checkpoint_guard import (
        scaled_quantization_report, verify_quantized_swap,
    )

    census = _quantization_census(shards, key_to_shard)
    report = scaled_quantization_report(
        census, arch="LTX-2.3", path=directory, label="transformer")
    if report is None:
        return None

    print(f"[LTX2Loader] weight-only QUANTIZED transformer component detected "
          f"({report['scale_keys']} scale key(s), {report['quantized_weight_keys']} "
          f"quantized weight(s)); rebuilding it here rather than letting "
          f"from_pretrained cast the codes into bf16 parameters")

    from accelerate import init_empty_weights
    from diffusers import LTX2VideoTransformer3DModel

    from core.models.common.single_file_format import read_state_dict

    # THE path the census read (the index for a shard set, the file for a
    # single), carried through from the resolution rather than re-derived:
    # ``read_state_dict`` follows an index's weight_map, so a directory with two
    # index files would otherwise let the census and the load disagree about
    # which shards they are talking about.
    sd, metadata = read_state_dict(source)

    config = _transformer_config(directory, metadata)
    with init_empty_weights():
        model = LTX2VideoTransformer3DModel.from_config(config)
        model.to(torch_dtype)

    swapped = _swap_ltx2_quantized_linears(model, sd, torch_dtype)
    verify_quantized_swap(report, swapped, arch="LTX-2.3", path=directory,
                          label="transformer")

    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    if unexpected:
        # Harmless in itself: a key the module tree has no home for was ignored.
        print(f"[LTX2Loader] WARNING: {len(unexpected)} unexpected key(s); first 5: {unexpected[:5]}")
    if missing:
        # NOT a warning. The module was built under ``init_empty_weights``, so
        # every parameter starts on the META device and only ``assign=True``
        # replaces it with a real tensor. A missing key therefore leaves a meta
        # tensor in a live model: nothing fails here, nothing fails at
        # ``.to(device)``, and the first forward raises
        # "Cannot copy out of meta tensor" (or worse, a NotImplementedError deep
        # in an attention kernel) minutes later in an unrelated place. Refusing
        # here names the file and the keys.
        raise RuntimeError(
            f"the LTX-2.3 transformer component at {directory} is missing "
            f"{len(missing)} key(s) required by LTX2VideoTransformer3DModel "
            f"(first 5: {missing[:5]}). The model was built on the meta device, "
            f"so each missing key would stay a meta tensor and detonate at the "
            f"first forward instead of here. Check that the config.json matches "
            f"the weights.")
    # Belt and braces for the same failure arriving another way (a key present
    # but assigned a meta tensor, a buffer no ``load_state_dict`` covers): the
    # walk is over ~4.2 k entries and costs nothing next to a 19-37 GB load.
    stranded = [n for n, t in list(model.named_parameters()) + list(model.named_buffers())
                if getattr(t, "is_meta", False)]
    if stranded:
        raise RuntimeError(
            f"the rebuilt LTX-2.3 transformer from {directory} still holds "
            f"{len(stranded)} meta tensor(s) after loading (first 5: "
            f"{stranded[:5]}); it would fail at the first forward, not here.")
    # Deliberately NOT cast to torch_dtype afterwards: that would double the int8
    # buffers back to bf16 and drop the quantized-GEMM path, which gates on the
    # weight dtype.
    return model.eval().requires_grad_(False)


def load_ltx2_from_diffusers(
    model_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    *, transformer_checkpoint: Optional[str] = None,
) -> dict:
    """Load LTX-2.3 from a diffusers directory (model_index.json + subfolders).

    Returns a component dict consumed by PipelineManager.load_model():
        {
          "type": "ltx2",
          "pipeline": <LTX2Pipeline>,          # the assembled pipeline (P1b entry)
          "transformer": <LTX2VideoTransformer3DModel>,
          "vae": <AutoencoderKLLTX2Video>,
          "audio_vae": <AutoencoderKLLTX2Audio>,
          "text_encoder": <Gemma3ForConditionalGeneration>,
          "tokenizer": <GemmaTokenizerFast>,
          "connectors": <LTX2TextConnectors>,
          "vocoder": <LTX2VocoderWithBWE>,
          "scheduler": <FlowMatchEulerDiscreteScheduler>,
          "vae_scale_factor_spatial": 32,
          "vae_scale_factor_temporal": 8,
          "latent_channels": 128,
          "is_video": True,
        }

    P1b relies on the "pipeline" reference (its __call__ is the generation entry)
    and on the individual component refs for manual GPU staging / offload wiring.
    """
    from core.models.ltx2 import LTX2Pipeline

    print(f"[LTX2Loader] Loading LTX-2.3 diffusers directory: {model_path}")
    print(f"[LTX2Loader] dtype={torch_dtype} (bf16 halves the fp32 Gemma-3 text encoder)")

    # Header-only census first (no tensor bytes): a weight-only QUANTIZED
    # transformer component is rebuilt and swapped here, because from_pretrained
    # would drop its scales and cast its codes. Everything else -- including a
    # plain float8 cast -- returns None and takes the original path unchanged.
    from core.rewiring.artifacts import read_directory_topology
    from .residual_unet import native_config, validate_topology_identity, wrap_transformer
    from .component_sources import validate_sources
    sources = validate_sources(dict(schema_version=1, root=os.path.abspath(model_path)))
    directory = _resolve_transformer_dir(model_path)
    video_codec = None
    video_identity = None
    if transformer_checkpoint is not None:
        from .residual_unet import preflight_checkpoint
        graph, saved_config = preflight_checkpoint(transformer_checkpoint)
        config = _source_transformer_config(directory)
        from core.models.common.single_file_format import read_metadata
        from .video_artifact import read_record, canonical_config, restored_codec
        saved_metadata = read_metadata(transformer_checkpoint)
        record = read_record(saved_metadata, saved_config) if saved_config is not None else None
        comparison = record["source_config"] if record is not None else saved_config
        if comparison is None or canonical_config(comparison) != canonical_config(config):
            raise ValueError("LTX-2 saved transformer and companion root geometry disagree")
        if record is not None:
            video_codec, video_identity = restored_codec(transformer_checkpoint, saved_metadata, record, torch_dtype=torch_dtype)
        restored = load_ltx2_transformer_checkpoint(transformer_checkpoint, torch_dtype)
        quantized_transformer = getattr(restored, "transformer", restored)
    else:
        graph = read_directory_topology(directory) if directory is not None else None
    directory_record = None
    if transformer_checkpoint is None and directory is not None:
        from core.models.common.single_file_format import read_metadata
        from .video_artifact import KEY as VIDEO_ALIGNMENT_KEY, read_record, canonical_config, restored_codec
        _, directory_keys, source = _transformer_shards(directory)
        saved_metadata = read_metadata(source) if source is not None else {}
        if VIDEO_ALIGNMENT_KEY in saved_metadata:
            from .residual_unet import preflight_checkpoint
            graph, saved_config = preflight_checkpoint(source)
            directory_record = read_record(saved_metadata, saved_config)
            config = _source_transformer_config(directory)
            if canonical_config(config) not in (canonical_config(saved_config),
                                               canonical_config(directory_record["source_config"])):
                raise ValueError("LTX directory config disagrees with the saved video alignment")
            video_codec, video_identity = restored_codec(source, saved_metadata, directory_record, torch_dtype=torch_dtype)
            restored = load_ltx2_transformer_checkpoint(source, torch_dtype)
            quantized_transformer = getattr(restored, "transformer", restored)
        elif any("_video_latent_alignment." in key for key in directory_keys):
            raise ValueError("LTX directory has an orphaned video alignment")
    if transformer_checkpoint is None and directory_record is None and graph is not None:
        _, _, source = _transformer_shards(directory)
        config = _source_transformer_config(directory)
        validate_topology_identity(graph, config)
        restored = load_ltx2_transformer_checkpoint(source, torch_dtype)
        if native_config(restored.config) != native_config(config):
            raise ValueError("LTX-2 directory and checkpoint native configs disagree")
        quantized_transformer = restored.transformer
    elif transformer_checkpoint is None and directory_record is None:
        quantized_transformer = _load_quantized_ltx2_transformer(model_path, torch_dtype)
    overrides = {"transformer": quantized_transformer} if quantized_transformer is not None else {}
    if video_codec is not None:
        # from_pretrained type-checks ModelMixin; the normalized facade is installed after assembly.
        video_codec.inner.spatial_compression_ratio = video_codec.spatial_compression_ratio
        video_codec.inner.temporal_compression_ratio = video_codec.temporal_compression_ratio
        overrides["vae"] = video_codec.inner

    pipeline = LTX2Pipeline.from_pretrained(model_path, torch_dtype=torch_dtype, **overrides)
    if video_codec is not None:
        pipeline.register_modules(vae=video_codec)

    # Keep everything on CPU after load; P1b stages to GPU (or uses
    # model_cpu_offload_seq). from_pretrained already leaves modules on CPU.
    for name in (
        "text_encoder", "connectors", "transformer",
        "vae", "audio_vae", "vocoder",
    ):
        comp = getattr(pipeline, name, None)
        if comp is not None and hasattr(comp, "to"):
            try:
                comp.to("cpu")
            except Exception:
                pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    transformer = getattr(pipeline, "transformer", None)
    if graph is not None or getattr(transformer, "_video_latent_alignment", None) is not None:
        transformer = wrap_transformer(transformer, graph)
        pipeline.transformer = transformer
    inner = getattr(transformer, "transformer", transformer)
    inner._ltx2_component_sources = sources
    if video_identity is not None:
        inner._ltx2_vae_identity = video_identity
    vae = getattr(pipeline, "vae", None)

    # VAE scale factors (spatial 32x, temporal 8x) — pull from the transformer
    # config when present, else fall back to the confirmed distilled values.
    vae_scale_spatial = 32
    vae_scale_temporal = 8
    latent_channels = 128
    try:
        tcfg = getattr(transformer, "config", None)
        if tcfg is not None:
            factors = getattr(tcfg, "vae_scale_factors", None)
            if factors and len(factors) == 3:
                vae_scale_temporal = int(factors[0])
                vae_scale_spatial = int(factors[1])
            latent_channels = int(getattr(tcfg, "in_channels", latent_channels))
    except Exception:
        pass

    print(
        f"[LTX2Loader] Loaded LTX-2.3 (latent_channels={latent_channels}, "
        f"spatial={vae_scale_spatial}x, temporal={vae_scale_temporal}x)"
    )

    return {
        "type": "ltx2",
        "pipeline": pipeline,
        "transformer": transformer,
        "vae": vae,
        "audio_vae": getattr(pipeline, "audio_vae", None),
        "text_encoder": getattr(pipeline, "text_encoder", None),
        "tokenizer": getattr(pipeline, "tokenizer", None),
        "connectors": getattr(pipeline, "connectors", None),
        "vocoder": getattr(pipeline, "vocoder", None),
        "scheduler": getattr(pipeline, "scheduler", None),
        "vae_scale_factor_spatial": vae_scale_spatial,
        "vae_scale_factor_temporal": vae_scale_temporal,
        "latent_channels": latent_channels,
        "vae_identity": video_identity.facts() if video_identity is not None else None,
        "is_video": True,
    }


def load_ltx2_transformer_checkpoint(path: str, torch_dtype: torch.dtype = torch.bfloat16):
    """Restore a standalone native AV transformer; companion loading is separate."""
    from accelerate import init_empty_weights
    from diffusers import LTX2VideoTransformer3DModel
    from core.models.common.single_file_format import read_state_dict
    from core.models.common.quantized_checkpoint_guard import quantized_state_dict_report, scaled_quantization_report, verify_quantized_swap
    from .residual_unet import preflight_checkpoint, wrap_transformer, native_config, GRID_POLICY_KEY
    graph, config = preflight_checkpoint(path)
    if config is None:
        raise ValueError("LTX-2 transformer checkpoint requires native transformer_config")
    state, metadata = read_state_dict(path)
    from .video_artifact import read_record, prepare_restore
    record = read_record(metadata, config)
    state = {key.removeprefix("net."):value for key,value in state.items()
             if not (record is not None and key.startswith("vae."))}
    report = scaled_quantization_report(quantized_state_dict_report(state,arch="LTX-2.3",path=path,label="transformer"),
                                        arch="LTX-2.3",path=path,label="transformer")
    construction = native_config(config)
    construction.pop(GRID_POLICY_KEY)
    with init_empty_weights():
        model = LTX2VideoTransformer3DModel.from_config(construction)
        if record is not None:
            prepare_restore(model, record)
        model.to(torch_dtype)
    swapped = _swap_ltx2_quantized_linears(model,state,torch_dtype)
    verify_quantized_swap(report,swapped,arch="LTX-2.3",path=path,label="transformer")
    quantized_weights = {key.removesuffix(".weight_scale")+".weight" for key in state if key.endswith(".weight_scale")}
    state = {key:(value.to(torch_dtype) if value.is_floating_point() and
                  key not in quantized_weights and not key.endswith(".weight_scale") else value)
             for key,value in state.items()}
    model.load_state_dict(state,strict=True,assign=True)
    if record is not None and not model._video_latent_alignment.ready:
        raise ValueError("Restored LTX video alignment is uncalibrated")
    if any(value.is_meta for value in list(model.parameters())+list(model.buffers())):
        raise ValueError("LTX-2 checkpoint left native tensors on the meta device")
    model.eval().requires_grad_(False)
    return wrap_transformer(model,graph) if graph is not None or record is not None else model


def load_ltx2_from_path(model_path: str, torch_dtype: torch.dtype = torch.bfloat16):
    from pathlib import Path
    from .component_sources import read_sources
    source = Path(model_path)
    if source.is_dir():
        return load_ltx2_from_diffusers(str(source), torch_dtype)
    sources = read_sources(source)
    if sources is None:
        raise ValueError("Standalone LTX-2 model requires ltx2_components companion references")
    return load_ltx2_from_diffusers(sources["root"], torch_dtype, transformer_checkpoint=str(source))
