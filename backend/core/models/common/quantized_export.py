"""Write a LIVE, weight-only-quantized transformer back out as a single file.

WHY THIS EXISTS
---------------
``int8_runtime_quantize.quantize_linears_in_place`` converts an ordinary bf16
checkpoint to ``Int8Linear`` / ``Fp8Linear`` IN PLACE at generation time. That
conversion is one-way and lives only in the process: reload the model and it is
gone, and the next session pays for it again. This module writes the converted
module to disk in EXACTLY the layout the offline tool
(``subapps/fp8_quantize/quantize_transformer_fp8.py``) emits, so the production
loader reads it back with no loader change.

WHAT MAKES THAT CHEAP
---------------------
``Int8Linear`` / ``Fp8Linear`` register ``weight`` (int8 / float8_e4m3fn),
``weight_scale`` (float32, per output row) and ``bias`` as plain BUFFERS, so a
live ``transformer.state_dict()`` already IS the on-disk layout -- there is no
serialization step, only a prefix and metadata. ``safetensors.save_file``
handles int8 and float8 natively. The load path
(``is_int8_state_dict`` -> ``swap_linears_to_int8`` -> ``load_fp8_state_dict``
for Krea 2, ``_swap_quantized_linears`` -> ``load_state_dict(assign=True)`` for
Anima) round-trips it.

KEY LAYOUT
----------
``EXPORT_LAYOUTS`` is the ONE table of per-arch on-disk facts, shared with the
offline tool (which imports it) so the two cannot emit different files. The
subtlety it encodes is the prefix: Anima's loader strips a ``net.`` prefix, so a
LIVE state dict has no ``net.`` and the export must re-add it, while the offline
tool reads a source that already carries it. Hence two prefix fields that
compose (see ``check_layout_prefixes``).

MEMORY
------
Written shard-by-shard through ``ShardWriter`` (~4 GB buffer), not through
``single_file_format.save_single_file_state`` -- that one materialises the whole
state dict in host RAM via ``dedup_tensors``, which a 12 GB int8 Krea 2
transformer does not justify.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from safetensors.torch import save_file

from core.models.common.single_file_format import _INDEX_SUFFIX, _SHARD_SUFFIX

__all__ = [
    "DEFAULT_EXPORT_SHARD_BYTES",
    "SIBLING_DIRS",
    "EXPORT_LAYOUTS",
    "ShardWriter",
    "link_siblings",
    "krea2_export_metadata",
    "anima_export_metadata",
    "check_layout_prefixes",
    "identity_source_transform",
    "layout_module_specs",
    "layout_source_transform",
    "primary_live_prefix",
    "resolve_layout_modules",
    "combined_linear_inventory",
    "quantized_linear_inventory",
    "reject_quant_tokens_in_path",
    "export_quantized_transformer",
]


# Output shard threshold. Smaller than the repo default (10 GB) because the
# writer buffers a whole shard in RAM.
DEFAULT_EXPORT_SHARD_BYTES = 4 * 1024 ** 3

# Default companion component directories to junction next to the output. An
# arch whose loader probes different names overrides this with its own
# ``siblings`` entry; entries may be RELATIVE PATHS, not just names.
SIBLING_DIRS = ("text_encoder", "vae", "tokenizer", "scheduler")


# ---------------------------------------------------------------------------
# Per-arch metadata builders
# ---------------------------------------------------------------------------

def krea2_export_metadata(config: dict) -> Dict[str, str]:
    """Metadata block a Krea 2 single-file loader reads."""
    from core.models.krea2.vendor.single_file import KREA2_DEFAULT_CONFIG

    config = dict(config or {})
    return {
        "model_type": "krea2",
        "variant": "raw",
        "is_distilled": "0",
        "krea2_config": json.dumps({k: config[k] for k in KREA2_DEFAULT_CONFIG if k in config}),
        "has_text_encoder": "0",
        "format": "pt",
    }


def flux2_export_metadata(config: dict) -> Dict[str, str]:
    """Metadata block for a FLUX.2 transformer single file.

    ``base_model_repo`` and ``is_distilled`` are the ONLY metadata keys
    ``model_loader.load_flux2_from_safetensors`` reads back, and they decide real
    behaviour: ``is_distilled`` alone flips
    ``do_classifier_free_guidance = guidance_scale > 1.0 and not is_distilled``
    in every FLUX.2 denoise loop (pipeline_backends/flux2.py). They are PROPAGATED
    here, never invented:

    * present in ``config`` -- which is the case for a live export, because
      ``load_flux2_from_safetensors`` puts both into the ``config`` dict it
      returns, and ``quantized_export_job`` passes that dict -- they are written
      verbatim, so the quantized copy loads exactly the way the source did. For a
      vanilla checkpoint that reproduces what the loader's own detection would
      have produced anyway; for a sushiUI full-FT save (the only writer of those
      keys, ``flux2_adapter.py``) it preserves provenance the loader could NOT
      re-derive: its single-layer-count probe knows 24/36/48 and a Klein 4B
      export has 20, so detection would fall back to ``klein-base-4B`` and take
      ``is_distilled`` from that repo -- turning CFG ON for a distilled
      fine-tune;
    * absent -- the offline tool's route, where ``config`` is the pinned geometry
      and genuinely carries no provenance -- they are OMITTED, and the loader's
      detection runs exactly as it does for the unquantized source. Writing a
      guessed value would be worse than not writing one: the pinned config cannot
      tell a distilled variant from a base one, their geometry being identical.

    ``flux2_config`` is recorded for provenance only; nothing reads it back.
    """
    from core.models.flux2.single_file import FLUX2_DEFAULT_CONFIG

    config = dict(config or {})
    metadata = {
        "modelspec.architecture": "flux2",
        "model_type": "flux2",
        "flux2_config": json.dumps({k: config[k] for k in FLUX2_DEFAULT_CONFIG if k in config}),
        "format": "pt",
    }
    base_model_repo = config.get("base_model_repo")
    if base_model_repo:
        metadata["base_model_repo"] = str(base_model_repo)
    if "is_distilled" in config and config["is_distilled"] is not None:
        # Read back as ``metadata["is_distilled"].lower() == "true"``; written the
        # same way flux2_adapter.py writes it, so the two producers agree.
        metadata["is_distilled"] = str(bool(config["is_distilled"])).lower()
    return metadata


def anima_export_metadata(config: dict) -> Dict[str, str]:
    """Metadata block an Anima DiT single-file loader reads.

    ``modelspec.architecture`` is the fast path in ``is_anima_safetensors``; the
    key-signature check behind it also still passes, because quantization
    renames nothing (it only changes weight dtypes and adds ``.weight_scale``).
    """
    return {
        "modelspec.architecture": "anima",
        "model_type": "anima",
        "format": "pt",
    }


# ---------------------------------------------------------------------------
# Source-key transform (offline route only)
# ---------------------------------------------------------------------------

def _flux2_source_transform(key: str, tensor: Optional[torch.Tensor]):
    """FLUX.2's ``source_transform``: BFL keys -> diffusers keys, identity otherwise.

    A thin, DEFERRED wrapper (the implementation reaches for diffusers'
    ``single_file_utils``, which this module has no other reason to import) kept
    at module scope so ``EXPORT_LAYOUTS`` can name a stable function object --
    ``layout_source_transform(arch) is identity_source_transform`` is how
    ``check_layout_prefixes`` decides whether the prefix invariant applies, and a
    lambda rebuilt per call would make that comparison meaningless for anything
    that later wants the same test in the other direction.
    """
    from core.models.flux2.single_file import flux2_bfl_to_diffusers

    return flux2_bfl_to_diffusers(key, tensor)


def identity_source_transform(key: str, tensor: Optional[torch.Tensor]):
    """The default ``source_transform``: the source key IS the module path.

    See ``EXPORT_LAYOUTS``' ``source_transform`` entry for the contract. Kept as
    a named function rather than an inline lambda so a layout can be compared
    against it (``layout_source_transform(arch) is identity_source_transform``),
    which is what ``check_layout_prefixes`` uses to decide whether the mechanical
    prefix invariant even applies.
    """
    return ((key, tensor),)


# ---------------------------------------------------------------------------
# Per-arch on-disk layout
# ---------------------------------------------------------------------------
#
# Keys:
#   modules         ((component name, live prefix), ...) — the live components
#                   this architecture's export writes, IN WRITE ORDER, and the
#                   prefix each one's ``state_dict()`` keys get on disk. Most
#                   archs have exactly one; Ideogram 4 has two transformers of
#                   identical geometry (``transformer.`` and
#                   ``unconditional_transformer.``) that must land in ONE file,
#                   because a single file is the whole point of the feature.
#                   The component name is the key under
#                   ``pipeline_manager.<arch>_components``.
#   offline_prefix  the offline tool's ``prefix``: prepended to every SOURCE key
#                   (which still carries ``source_prefix``).
#   source_prefix   stripped from a SOURCE key before it is matched against a
#                   module path (offline only; a live module has no wrapper).
#   source_transform  (optional, default ``identity_source_transform``) offline
#                   only. ``(key, tensor) -> [(key', tensor'), ...]`` applied to
#                   every tensor STREAMED out of the source checkpoint, before
#                   ``source_prefix`` stripping and before matching against a
#                   module path. It exists because "source key == module path
#                   modulo one prefix" is an architecture-specific fact, not a
#                   general one: FLUX.2's BFL-format single files need a full key
#                   remap plus a fused-qkv split, and Ideogram 4's loader runs
#                   ``_convert_fused_qkv_to_split`` BEFORE the quantized swap, so
#                   its checkpoint keys are not the module paths either.
#                   Contract:
#                     * one input tensor may fan out to several outputs (a fused
#                       qkv weight -> q/k/v), or to none (a key the target module
#                       does not have);
#                     * it is ALSO called with ``tensor=None`` during the
#                       key-enumeration pass and must then return the same KEY
#                       set it would for a real tensor, with ``None`` tensors;
#                     * the output keys are in the arch's canonical SOURCE
#                       layout, i.e. still carrying ``source_prefix``.
#                   Fusing does not change what is quantized: the scales are PER
#                   ROW and rows are independent, so quantizing a fused weight
#                   then splitting it is numerically identical to splitting then
#                   quantizing. The tiebreaker is that the RUNTIME export sees
#                   the live module AFTER the split, so the offline artifact has
#                   to store split keys to stay byte-comparable with it.
#   metadata        fn(config) -> metadata dict.
#   siblings        component directories ``link_siblings`` junctions next to the
#                   output; may be relative paths.
#   sibling_root    where those names are rooted, relative to the OUTPUT's
#                   directory.
#   output_subdir   where inside a chosen export ROOT the file must be written
#                   for the loader's own layout probe to find its companions.
#
# INVARIANT (``check_layout_prefixes``): offline_prefix + source_prefix ==
# the PRIMARY module's live prefix. Both routes must produce the same on-disk key
# for the same module path, and that equality is the whole reason both fields
# exist. It is checked only for layouts that use the identity
# ``source_transform``; once a transform rewrites keys, "the offline key is the
# live key plus a constant prefix" is no longer an equation that can be checked
# by string concatenation, and the transform itself owns the correspondence.
EXPORT_LAYOUTS: Dict[str, Dict[str, object]] = {
    "krea2": {
        # sushiUI single-file layout: transformer weights live under this prefix.
        "modules": (("transformer", "transformer."),),
        "offline_prefix": "transformer.",
        "source_prefix": "",
        "metadata": krea2_export_metadata,
        "siblings": SIBLING_DIRS,
        "sibling_root": ".",
        "output_subdir": "",
    },
    "flux2": {
        # A FLUX.2 transformer single file carries the diffusers module tree with
        # NO prefix at all: ``load_flux2_from_safetensors`` reads the state dict,
        # and a key set that is neither BFL (``double_blocks.*``) nor a sushiUI
        # full-FT save (``model.diffusion_model.*``) falls through to its "already
        # in diffusers format" branch and is loaded as-is. An export must
        # therefore land in that third branch, which an empty prefix does and
        # which the OTHER two must not be able to claim: no exported key starts
        # with ``double_blocks.`` (the source transform below has already
        # rewritten those to ``transformer_blocks.``) or with
        # ``model.diffusion_model.``.
        "modules": (("transformer", ""),),
        "offline_prefix": "",
        "source_prefix": "",
        # BFL-format sources need a full key remap plus a fused-qkv split before
        # a key can be compared with a module path; a diffusers-format source is
        # passed through untouched (the transform decides per key). See
        # ``core.models.flux2.single_file``.
        "source_transform": _flux2_source_transform,
        "metadata": flux2_export_metadata,
        # No sibling junctions: the FLUX.2 loader resolves its VAE from the
        # Apache-2.0 VAE store and its text encoder / tokenizer / scheduler from
        # the detected base repo, and probes nothing next to the checkpoint. A
        # ``text_encoder`` directory beside the file would be inert, so offering
        # to create one would only suggest it mattered.
        "siblings": (),
        "sibling_root": ".",
        "output_subdir": "",
    },
    "anima": {
        # Anima DiT single-files carry the module tree verbatim under ``net.``;
        # the loader strips that prefix, so the file keeps it.
        "modules": (("transformer", "net."),),
        "offline_prefix": "",
        "source_prefix": "net.",
        "metadata": anima_export_metadata,
        # anima_loader.detect_anima_split_layout walks UP from the DiT file to a
        # ``split_files/diffusion_models`` parent and probes these two siblings
        # for the Qwen3 text encoder and the Qwen-Image VAE -- which is why the
        # DiT must be written into that subdirectory, not at the root.
        "siblings": ("split_files/text_encoders", "split_files/vae"),
        "sibling_root": os.path.join("..", ".."),
        "output_subdir": os.path.join("split_files", "diffusion_models"),
    },
}


def layout_module_specs(arch: str) -> Tuple[Tuple[str, str], ...]:
    """``((component name, live prefix), ...)`` for ``arch``, in write order."""
    layout = EXPORT_LAYOUTS.get(arch)
    if layout is None:
        raise ValueError(
            f"no single-file export layout for architecture {arch!r} "
            f"(known: {', '.join(sorted(EXPORT_LAYOUTS))})")
    return tuple((str(name), str(prefix)) for name, prefix in layout["modules"])


def primary_live_prefix(arch: str) -> str:
    """The live-state-dict prefix of ``arch``'s FIRST exported component.

    The offline tool works from a source checkpoint of that primary component,
    so this is the prefix its keys are compared against.
    """
    return layout_module_specs(arch)[0][1]


def layout_source_transform(arch: str) -> Callable:
    """``arch``'s offline source-key transform (``identity_source_transform``)."""
    layout = EXPORT_LAYOUTS.get(arch) or {}
    return layout.get("source_transform") or identity_source_transform


def check_layout_prefixes() -> List[str]:
    """Return a list of arch names whose prefix fields do not compose.

    Empty is the healthy answer. Exposed (rather than asserted at import) so a
    test can state the invariant without a module-load side effect.

    Skips an arch that declares a non-identity ``source_transform``: for those
    the offline key is not the live key plus a constant prefix by construction,
    so concatenating the two fields proves nothing. Everything else is checked
    against its PRIMARY module's live prefix.
    """
    bad = []
    for arch, layout in EXPORT_LAYOUTS.items():
        if layout_source_transform(arch) is not identity_source_transform:
            continue
        if f"{layout['offline_prefix']}{layout['source_prefix']}" != primary_live_prefix(arch):
            bad.append(arch)
    return bad


# ---------------------------------------------------------------------------
# Streaming writer
# ---------------------------------------------------------------------------

class ShardWriter:
    """Buffer tensors and flush diffusers-convention shards + an index.

    Shard naming and the index schema match
    ``core.models.common.single_file_format.save_single_file_state`` exactly, so
    the produced checkpoint is read by ``read_state_dict`` like any other. The
    difference is that nothing here holds the whole state dict at once.
    """

    def __init__(self, out_path: str, metadata: Dict[str, str], max_shard_bytes: int):
        self.directory = os.path.dirname(os.path.abspath(out_path))
        stem = os.path.basename(out_path)
        if stem.endswith(_SHARD_SUFFIX):
            stem = stem[: -len(_SHARD_SUFFIX)]
        self.stem = stem
        self.metadata = {k: str(v) for k, v in metadata.items()}
        self.max_shard_bytes = max_shard_bytes
        self.buffer: Dict[str, torch.Tensor] = {}
        self.buffer_bytes = 0
        self.total_bytes = 0
        self.shards: List[Tuple[str, List[str]]] = []  # (temp name, keys)
        os.makedirs(self.directory, exist_ok=True)

    def add(self, key: str, tensor: torch.Tensor) -> None:
        nbytes = tensor.numel() * tensor.element_size()
        if self.buffer and self.buffer_bytes + nbytes > self.max_shard_bytes:
            self._flush()
        self.buffer[key] = tensor
        self.buffer_bytes += nbytes
        self.total_bytes += nbytes

    def _flush(self) -> None:
        if not self.buffer:
            return
        # Written under a provisional name; renamed once the shard COUNT is known
        # (the diffusers convention encodes the total in every filename).
        tmp = os.path.join(self.directory, f"{self.stem}-part{len(self.shards):05d}.tmp.safetensors")
        save_file(self.buffer, tmp, metadata=self.metadata)
        self.shards.append((tmp, list(self.buffer)))
        print(f"[QuantExport]   wrote shard {len(self.shards)} ({self.buffer_bytes / 2**30:.2f} GB, "
              f"{len(self.buffer)} tensors)")
        self.buffer = {}
        self.buffer_bytes = 0

    def close(self) -> str:
        self._flush()
        n = len(self.shards)
        if n == 1:
            final = os.path.join(self.directory, f"{self.stem}{_SHARD_SUFFIX}")
            os.replace(self.shards[0][0], final)
            return final
        weight_map: Dict[str, str] = {}
        for i, (tmp, keys) in enumerate(self.shards, start=1):
            name = f"{self.stem}-{i:05d}-of-{n:05d}.safetensors"
            os.replace(tmp, os.path.join(self.directory, name))
            for k in keys:
                weight_map[k] = name
        index_path = os.path.join(self.directory, f"{self.stem}{_INDEX_SUFFIX}")
        with open(index_path, "w", encoding="utf-8") as fh:
            json.dump(
                {"metadata": {**self.metadata, "total_size": self.total_bytes}, "weight_map": weight_map},
                fh,
                indent=2,
            )
        return index_path

    def abort(self) -> None:
        """Delete any provisional shards written so far (failed export)."""
        self.buffer = {}
        self.buffer_bytes = 0
        for tmp, _keys in self.shards:
            try:
                os.remove(tmp)
            except OSError:
                pass
        self.shards = []


# ---------------------------------------------------------------------------
# Sibling junctions
# ---------------------------------------------------------------------------

def link_siblings(src_dir: str, dest_dir: str, names=SIBLING_DIRS) -> List[str]:
    """Create directory junctions dest_dir/<name> -> src_dir/<name>.

    ``names`` may contain RELATIVE PATHS (Anima's components live under
    ``split_files/``), so the link's parent directory is created as needed.

    Junctions (``mklink /J``) need no administrator rights and work across local
    volumes; a symlink would need developer mode. POSIX falls back to symlinks.
    """
    made = []
    os.makedirs(dest_dir, exist_ok=True)
    for name in names:
        target = os.path.join(src_dir, name)
        link = os.path.join(dest_dir, name)
        if not os.path.isdir(target):
            continue
        if os.path.exists(link):
            print(f"[QuantExport]   sibling '{name}' already present, leaving as is")
            continue
        os.makedirs(os.path.dirname(link), exist_ok=True)
        if os.name == "nt":
            # cmd parses a leading "/" as a switch, so forward-slash paths must be
            # normalised to backslashes before they reach mklink.
            link, target = os.path.normpath(link), os.path.normpath(target)
            res = subprocess.run(["cmd", "/c", "mklink", "/J", link, target],
                                 capture_output=True, text=True)
            if res.returncode != 0:
                print(f"[QuantExport]   WARNING: could not link '{name}': "
                      f"{res.stdout.strip()} {res.stderr.strip()}")
                continue
        else:
            os.symlink(target, link, target_is_directory=True)
        made.append(name)
        print(f"[QuantExport]   linked {link} -> {target}")
    return made


# ---------------------------------------------------------------------------
# Inventory + guards
# ---------------------------------------------------------------------------

def quantized_linear_inventory(model: nn.Module) -> Dict[str, object]:
    """Per-layer census of a module's Linear population.

    Returns ``{"int8", "e4m3", "plain", "total", "formats"}`` where ``formats``
    maps module path -> ``"int8"`` / ``"e4m3"`` / ``"linear"``. Detection is by
    MODULE TYPE, the same rule ``already_weight_only_quantized`` uses, so it
    cannot be confused by a plain ``nn.Linear`` that merely stores float8.
    """
    from core.models.ideogram4.vendor.fp8_linear import Fp8Linear
    from core.models.ideogram4.vendor.int8_linear import Int8Linear

    formats: Dict[str, str] = {}
    counts = {"int8": 0, "e4m3": 0, "plain": 0}
    for name, mod in model.named_modules():
        if isinstance(mod, Int8Linear):
            formats[name] = "int8"
            counts["int8"] += 1
        elif isinstance(mod, Fp8Linear):
            formats[name] = "e4m3"
            counts["e4m3"] += 1
        elif isinstance(mod, nn.Linear):
            formats[name] = "linear"
            counts["plain"] += 1
    return {
        "int8": counts["int8"],
        "e4m3": counts["e4m3"],
        "plain": counts["plain"],
        "total": len(formats),
        "formats": formats,
    }


def combined_linear_inventory(
    modules: Sequence[Tuple[str, nn.Module]]
) -> Dict[str, object]:
    """``quantized_linear_inventory`` summed over several ``(name, module)`` pairs.

    ``formats`` is keyed by ``"<component>.<module path>"`` so a two-transformer
    architecture cannot collide two identically-shaped module trees into one
    entry, which is exactly the failure mode of summing per-module dicts.
    """
    total = {"int8": 0, "e4m3": 0, "plain": 0, "total": 0}
    formats: Dict[str, str] = {}
    for name, module in modules:
        one = quantized_linear_inventory(module)
        for k in ("int8", "e4m3", "plain", "total"):
            total[k] += int(one[k])
        for path, fmt in one["formats"].items():
            formats[f"{name}.{path}"] = fmt
    return {**total, "formats": formats}


def resolve_layout_modules(
    arch: str,
    model: Union[nn.Module, Mapping[str, nn.Module], Iterable[Tuple[str, nn.Module]]],
) -> List[Tuple[str, str, nn.Module]]:
    """``[(component name, live prefix, module), ...]`` for ``arch``.

    ``model`` may be

    * a bare ``nn.Module`` -- accepted only for a single-component architecture,
      where there is no ambiguity about which prefix it gets. Passing one for a
      multi-component arch raises rather than guessing, because guessing would
      write half a model into a file that claims to be whole;
    * a mapping ``{component name: module}`` (e.g. ``pipeline_manager
      .<arch>_components``); extra entries are ignored, a missing declared one
      raises;
    * a sequence of ``(component name, module)`` pairs.
    """
    specs = layout_module_specs(arch)
    if isinstance(model, nn.Module):
        if len(specs) != 1:
            raise ValueError(
                f"architecture {arch!r} exports {len(specs)} components "
                f"({', '.join(n for n, _ in specs)}); pass a mapping of them, not a "
                f"single module")
        return [(specs[0][0], specs[0][1], model)]

    if isinstance(model, Mapping):
        provided: Dict[str, nn.Module] = dict(model)
    else:
        provided = {str(name): mod for name, mod in model}

    resolved: List[Tuple[str, str, nn.Module]] = []
    for name, prefix in specs:
        module = provided.get(name)
        if module is None:
            raise ValueError(
                f"architecture {arch!r} exports component {name!r}, which was not "
                f"supplied (got: {', '.join(sorted(provided)) or 'nothing'})")
        resolved.append((name, prefix, module))
    return resolved


def reject_quant_tokens_in_path(path: str) -> None:
    """Raise if ``path`` contains a quant token the Krea 2 loader rejects.

    ``krea2/vendor/single_file.reject_unsupported_quant`` matches those tokens
    against the PATH as well as the metadata, so an export written to e.g.
    ``.../nvfp4_test/model.safetensors`` would be refused at load time by a file
    that is in fact perfectly loadable. Caught here, where the user can still
    choose a different destination.
    """
    try:
        from core.models.krea2.vendor.single_file import _REJECTED_QUANT_TOKENS as tokens
    except Exception:  # pragma: no cover - krea2 vendor always importable in-tree
        tokens = ("int8_convrot", "mxfp8", "nvfp4")
    haystack = str(path or "").lower().replace("-", "_")
    for token in tokens:
        if token in haystack:
            raise ValueError(
                f"the destination path contains '{token}', which the loader treats as an "
                f"unsupported quantization layout and refuses to read. Choose a path "
                f"without that word."
            )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_quantized_transformer(
    model: Union[nn.Module, Mapping[str, nn.Module], Iterable[Tuple[str, nn.Module]]],
    arch: str,
    output_path: str,
    *,
    config: Optional[dict] = None,
    audit: Optional[dict] = None,
    audit_note: Optional[str] = None,
    source: Optional[str] = None,
    link_siblings_from: Optional[str] = None,
    max_shard_bytes: int = DEFAULT_EXPORT_SHARD_BYTES,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    overwrite: bool = False,
) -> Dict[str, object]:
    """Write ``model`` (a live, weight-only quantized transformer) to a file.

    ``model`` is whatever ``resolve_layout_modules`` accepts: a bare module for a
    single-component architecture, or a mapping of the components the arch's
    layout declares. ALL of them go into ONE ``ShardWriter`` -- Ideogram 4's two
    transformers are one checkpoint, and splitting them across two files would
    defeat the single-file property this feature exists for.

    ``audit`` is the document ``quantize_linears_in_place`` returned for THIS
    module (``pipeline_manager._runtime_int8_audit``). When it is None -- the
    model came from an already-quantized checkpoint, so no audit was ever
    computed here -- nothing is fabricated: the metadata records
    ``quant_audit="unavailable"`` plus ``audit_note``.

    Returns a summary dict (written path, tensor/byte counts, inventory,
    metadata, audit path, linked siblings).
    """
    layout = EXPORT_LAYOUTS.get(arch)
    if layout is None:
        raise ValueError(
            f"no single-file export layout for architecture {arch!r} "
            f"(known: {', '.join(sorted(EXPORT_LAYOUTS))})")

    resolved = resolve_layout_modules(arch, model)
    inventory = combined_linear_inventory([(n, m) for n, _p, m in resolved])
    if not (inventory["int8"] or inventory["e4m3"]):
        raise ValueError(
            "the loaded transformer owns no weight-only quantized Linear layers, so "
            "an export would just be a copy of the source checkpoint")

    if not output_path.endswith(_SHARD_SUFFIX):
        raise ValueError(f"the destination must end in '{_SHARD_SUFFIX}': {output_path}")
    reject_quant_tokens_in_path(output_path)

    output_path = os.path.abspath(output_path)
    directory = os.path.dirname(output_path)
    stem = os.path.basename(output_path)[: -len(_SHARD_SUFFIX)]
    existing = [output_path, os.path.join(directory, f"{stem}{_INDEX_SUFFIX}")]
    if not overwrite:
        for candidate in existing:
            if os.path.exists(candidate):
                raise FileExistsError(
                    f"{candidate} already exists; choose another destination or "
                    f"enable overwrite")

    metadata: Dict[str, str] = dict(layout["metadata"](config or {}))
    metadata["quantized_linears"] = str(inventory["int8"] + inventory["e4m3"])
    metadata["quantized_int8_linears"] = str(inventory["int8"])
    metadata["quantized_e4m3_linears"] = str(inventory["e4m3"])
    metadata["unquantized_linears"] = str(inventory["plain"])
    # NOT written into a key the Krea 2 loader scans for rejected quant layouts:
    # ``single_file._REJECTED_QUANT_TOKENS`` matches ("int8_convrot", "mxfp8",
    # "nvfp4") against the path plus the serialized metadata, so this label must
    # be none of them. "int8_perrow" is what the offline tool writes.
    metadata["quant_format"] = "int8_perrow" if inventory["int8"] else "fp8_e4m3_perrow"
    metadata["quant_origin"] = "sushiui_runtime_export"
    metadata["quant_exported_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    if source:
        metadata["quant_source"] = str(source)
    metadata["quant_audit"] = "runtime" if audit else "unavailable"
    if not audit and audit_note:
        metadata["quant_audit_note"] = str(audit_note)

    states = [(prefix, mod.state_dict()) for _name, prefix, mod in resolved]
    metadata["exported_components"] = json.dumps(
        [{"component": n, "prefix": p} for n, p, _m in resolved])
    total = sum(len(s) for _p, s in states)
    writer = ShardWriter(output_path, metadata, max_shard_bytes)
    t0 = time.perf_counter()
    # Tied tensors: safetensors rejects two keys sharing storage. The shared
    # ``dedup_tensors`` DROPS the alias and relies on the loader re-tying it --
    # which neither the Anima nor the Krea 2 transformer loader does. A clone
    # keeps the key inventory of the file identical to the module's, which is
    # exactly what a round-trip check compares.
    #
    # ONE ``seen_ptrs`` map across ALL components, not one per component: two
    # transformers that share a buffer would otherwise reach safetensors as two
    # keys over one storage, which it rejects outright.
    seen_ptrs: Dict[int, str] = {}
    cloned_tied: List[str] = []
    written_keys = 0
    try:
        i = -1
        for prefix, state in states:
            for key, tensor in state.items():
                i += 1
                t = tensor.detach()
                if t.device.type != "cpu":
                    t = t.to("cpu")
                ptr = t.data_ptr()
                if ptr in seen_ptrs:
                    t = t.clone()
                    cloned_tied.append(f"{prefix}{key}")
                else:
                    seen_ptrs[ptr] = f"{prefix}{key}"
                writer.add(f"{prefix}{key}", t.contiguous())
                written_keys += 1
                if progress_cb is not None and (i % 25 == 0 or i + 1 == total):
                    try:
                        progress_cb(i + 1, total, f"{prefix}{key}")
                    except Exception:
                        pass
        if cloned_tied:
            # Recorded for transparency; the file itself carries both copies, so
            # nothing on the read side has to know.
            writer.metadata["tied_weights_cloned"] = json.dumps(cloned_tied)
        written = writer.close()
    except BaseException:
        writer.abort()
        raise
    elapsed = time.perf_counter() - t0

    audit_path = None
    if audit:
        audit_path = os.path.join(directory, f"{stem}.int8_audit.json")
        document = dict(audit)
        settings = dict(document.get("settings", {}) or {})
        settings["exported_to"] = written
        settings["export_arch"] = arch
        document["settings"] = settings
        with open(audit_path, "w", encoding="utf-8") as fh:
            json.dump(document, fh, indent=1)

    linked: List[str] = []
    if link_siblings_from:
        sibling_dest = os.path.normpath(
            os.path.join(directory, str(layout.get("sibling_root", "."))))
        linked = link_siblings(link_siblings_from, sibling_dest,
                               names=tuple(layout.get("siblings", SIBLING_DIRS)))

    print(f"[QuantExport] {arch}: wrote {written} "
          f"({writer.total_bytes / 2**30:.2f} GB, {written_keys} tensors) in {elapsed:.1f}s")
    return {
        "arch": arch,
        "components": [{"component": n, "prefix": p} for n, p, _m in resolved],
        "output_path": written,
        "requested_path": output_path,
        "tensors": written_keys,
        "total_bytes": writer.total_bytes,
        "elapsed_s": elapsed,
        "inventory": {k: inventory[k] for k in ("int8", "e4m3", "plain", "total")},
        "metadata": dict(writer.metadata),
        "audit_path": audit_path,
        "linked_siblings": linked,
        "tied_weights_cloned": cloned_tied,
    }
