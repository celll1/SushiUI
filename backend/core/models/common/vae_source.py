"""Where a swapped VAE comes from, and what is true about it.

Implements §7 of ``docs/guides/VAE_SWAP_MIGRATION_DESIGN.md``: one resolver for
the three source forms a VAE swap can name, producing one ``ResolvedVAE`` that
both the training side (§8) and the generation side (§9) consume.

    registry:<key>   a family in the shared table (``common/vae_store.py``)
    file:<path>      a standalone VAE: diffusers directory or single file
    model:<path>     the VAE *inside* another full checkpoint, extracted

The extraction path is reachable from the training-side selector only
(``GET /training/vae-sources``); the generation-side override list is unchanged
(§7.2). It reuses ``split_prefixed_state_dict`` with the ordered prefix list
``("vae.", "first_stage_model.")`` — both bundling conventions, one mechanism.

THE RULE THAT MATTERS (§7.3): observation wins per field, a declaration fills
only the gaps it leaves, and a number that is neither observed nor declared is
refused. In particular a scaling factor is NEVER inferred from the architecture:
``AutoencoderKL.from_single_file`` cannot tell an SDXL VAE from an SD1.5 one and
substitutes 0.18215 (``vae_store.LDM_SINGLE_FILE_DEFAULT_SCALING_FACTOR``), a
1.40x error on SDXL. Refusing is the whole point of this module.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from safetensors import safe_open

from core.models.common.single_file_format import (
    is_index_path,
    parse_component_metadata,
    split_prefixed_state_dict,
)
from core.models.common.vae_store import (
    VAE_REGISTRY,
    canonical_latent_scaling,
    resolve_vae_dir,
    store_dir_for,
)

# Ordered: krea2/minit2i bundle under "vae.", sd15/sdxl/zimage/flux2/anima/lens
# under "first_stage_model." (design §7.2). First match wins.
VAE_EXTRACT_PREFIXES: Tuple[str, ...] = ("vae.", "first_stage_model.")

SOURCE_FORMS: Tuple[str, ...] = ("registry", "file", "model")

_DECODER_CONVIN_SUFFIX = "decoder.conv_in.weight"
_BN_RUNNING_MEAN_SUFFIX = "bn.running_mean"
# Every encoder downsampler halves H and W, in either key convention.
_DOWNSAMPLER_MARKERS = ("downsamplers.0.conv.weight", "downsample.conv.weight")
_INDEX_SUFFIX = ".safetensors.index.json"
_WEIGHT_SUFFIX = ".safetensors"


class VaeSourceError(ValueError):
    """A VAE source that cannot be resolved into a complete, honest answer."""


# ---------------------------------------------------------------------------
# ResolvedVAE
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResolvedVAE:
    """One resolved VAE: its weights (or the means to load them) plus every fact
    §5.2's ``component.vae.*`` block records about it.

    ``struct_native`` / ``identity_native`` are tri-state. ``None`` means "not
    determined here" — no arch to compare against, or (for identity) no hash of
    the architecture's own VAE was supplied. They are never interchangeable:
    a fine-tuned copy of the native VAE is ``struct_native=True,
    identity_native=False`` (§13.7).
    """

    source: str
    form: str                       # one of SOURCE_FORMS
    family: str                     # a vae_store key, or "custom"
    latent_channels: int
    scale_factor: int
    scale_temporal: int
    ndim: int
    norm: str                       # "shift_scale" | "per_channel" | "batchnorm"
    norm_pack: int
    vae_class: str
    config: Dict[str, Any]
    content_hash: str
    provenance: str                 # display only, never resolved through (§13.9)
    locator: Optional[str]          # "registry:<key>" | "path:<abs>" | None
    struct_native: Optional[bool]
    identity_native: Optional[bool]
    scaling_factor: Optional[float]
    shift_factor: Optional[float]
    path: Optional[str] = None
    prefix: Optional[str] = None    # the prefix an extraction came from
    state_dict: Optional[Dict[str, torch.Tensor]] = field(
        default=None, repr=False, compare=False)

    def facts(self) -> Dict[str, Any]:
        """The JSON-able structural subset (what a selector and §5.2 both want)."""
        return {
            "source": self.source,
            "form": self.form,
            "family": self.family,
            "latent_channels": self.latent_channels,
            "scale_factor": self.scale_factor,
            "scale_temporal": self.scale_temporal,
            "ndim": self.ndim,
            "norm": self.norm,
            "norm_pack": self.norm_pack,
            "vae_class": self.vae_class,
            "scaling_factor": self.scaling_factor,
            "shift_factor": self.shift_factor,
            "content_hash": self.content_hash,
            "provenance": self.provenance,
            "locator": self.locator,
            "struct_native": self.struct_native,
            "identity_native": self.identity_native,
        }

    def load_module(self, torch_dtype: Optional[torch.dtype] = None):
        """Materialise the diffusers VAE module for these weights.

        The resolved ``scaling_factor``/``shift_factor`` are written into the
        module's config afterwards, so a single-file load never keeps
        ``from_single_file``'s 0.18215 guess.
        """
        import diffusers

        cls = getattr(diffusers, self.vae_class, None)
        if cls is None:
            raise VaeSourceError(
                f"diffusers has no VAE class '{self.vae_class}' (source {self.source})")

        if self.path and os.path.isdir(self.path):
            module = cls.from_pretrained(self.path, torch_dtype=torch_dtype)
        elif self.state_dict is None:
            raise VaeSourceError(
                f"{self.source} was resolved without weights; re-resolve with "
                "load_weights=True before loading a module")
        elif self.config:
            module = cls.from_config(self.config)
            module.load_state_dict(self.state_dict, strict=True)
            if torch_dtype is not None:
                module = module.to(dtype=torch_dtype)
        else:
            # No config to build from: the original/LDM key layout, which
            # diffusers converts itself. It accepts the state dict directly.
            module = cls.from_single_file(self.state_dict, torch_dtype=torch_dtype)

        overrides = {k: v for k, v in (("scaling_factor", self.scaling_factor),
                                       ("shift_factor", self.shift_factor))
                     if v is not None}
        if overrides:
            try:
                module.register_to_config(**overrides)
            except Exception as e:  # a class carrying no such config key
                print(f"[VAESource] scaling config not applicable to {self.vae_class}: {e}")
        module.eval()
        return module


# ---------------------------------------------------------------------------
# Source strings
# ---------------------------------------------------------------------------

def parse_vae_source(source: str) -> Tuple[str, str]:
    """``"registry:flux1"`` -> ``("registry", "flux1")``. Raises on anything else."""
    if not isinstance(source, str) or not source.strip():
        raise VaeSourceError("empty VAE source")
    text = source.strip()
    form, sep, value = text.partition(":")
    form = form.strip().lower()
    if not sep or form not in SOURCE_FORMS:
        raise VaeSourceError(
            f"unrecognised VAE source '{source}': expected one of "
            f"{', '.join(f'{f}:<...>' for f in SOURCE_FORMS)}")
    value = value.strip()
    if not value:
        raise VaeSourceError(f"VAE source '{source}' names nothing after '{form}:'")
    return form, value


# ---------------------------------------------------------------------------
# Weight access (header first; tensors only when asked for)
# ---------------------------------------------------------------------------

class _WeightFile:
    """Shapes and metadata from a safetensors file or shard index, without
    reading tensor data. ``tensors()`` then materialises only the keys asked for
    — which is what keeps ``model:`` extraction from a 12 GB checkpoint bounded
    by the VAE's own size.
    """

    def __init__(self, path: str):
        self.path = path
        self.metadata: Dict[str, str] = {}
        self.shapes: Dict[str, Tuple[int, ...]] = {}
        self._files: Dict[str, str] = {}

        index_path = self._index_for(path)
        if index_path is not None:
            with open(index_path, encoding="utf-8") as f:
                index = json.load(f)
            self.metadata = {k: str(v) for k, v in (index.get("metadata") or {}).items()}
            directory = os.path.dirname(index_path)
            for key, shard in (index.get("weight_map") or {}).items():
                self._files[key] = os.path.join(directory, shard)
            files = sorted(set(self._files.values()))
        else:
            files = [path]

        for file_path in files:
            if not os.path.isfile(file_path):
                raise VaeSourceError(f"missing weight file: {file_path}")
            with safe_open(file_path, framework="pt", device="cpu") as f:
                if not self.metadata:
                    self.metadata = dict(f.metadata() or {})
                for key in f.keys():
                    self.shapes[key] = tuple(f.get_slice(key).get_shape())
                    self._files.setdefault(key, file_path)

    @staticmethod
    def _index_for(path: str) -> Optional[str]:
        if is_index_path(path):
            return path
        # A single file present next to its own index wins, matching read_state_dict.
        if path.endswith(_WEIGHT_SUFFIX) and not os.path.exists(path):
            sibling = path[: -len(_WEIGHT_SUFFIX)] + _INDEX_SUFFIX
            if os.path.exists(sibling):
                return sibling
        return None

    def tensors(self, keys: Iterable[str]) -> Dict[str, torch.Tensor]:
        by_file: Dict[str, List[str]] = defaultdict(list)
        for key in keys:
            by_file[self._files[key]].append(key)
        out: Dict[str, torch.Tensor] = {}
        for file_path, file_keys in by_file.items():
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for key in file_keys:
                    out[key] = f.get_tensor(key)
        return out


def _diffusers_weight_path(directory: str) -> str:
    """The weights entry point of a diffusers component directory."""
    for base in ("diffusion_pytorch_model", "model"):
        index = os.path.join(directory, f"{base}{_INDEX_SUFFIX}")
        if os.path.isfile(index):
            return index
        single = os.path.join(directory, f"{base}{_WEIGHT_SUFFIX}")
        if os.path.isfile(single):
            return single
    candidates = sorted(
        name for name in os.listdir(directory) if name.endswith(_WEIGHT_SUFFIX))
    if len(candidates) == 1:
        return os.path.join(directory, candidates[0])
    if not candidates:
        raise VaeSourceError(
            f"no .safetensors weights in {directory} (a .bin-only VAE is not supported)")
    raise VaeSourceError(
        f"{directory} holds {len(candidates)} .safetensors files and no index; "
        "cannot tell which one is the VAE")


def _read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _vae_dir(path: str) -> Optional[str]:
    """The diffusers VAE directory at ``path``, or None.

    Same two-branch judgment ``pipeline.load_override_vae`` makes; delegated to
    the one implementation so ``file:`` and the override agree on what a
    standalone VAE directory is.
    """
    from api.generation_overrides import _vae_config_dir
    return _vae_config_dir(path)


# ---------------------------------------------------------------------------
# Observation (§7.3, "observed" column) — shapes only, no tensor data
# ---------------------------------------------------------------------------

def _suffix_key(shapes: Dict[str, Tuple[int, ...]], suffix: str) -> Optional[str]:
    for key in shapes:
        if key == suffix or key.endswith("." + suffix):
            return key
    return None


def observe_vae(shapes: Dict[str, Tuple[int, ...]]) -> Dict[str, Any]:
    """What the weights themselves say. Unknown fields are absent, never guessed.

    Pure over ``{key: shape}``, so a candidate listing can answer from a
    safetensors header alone.
    """
    observed: Dict[str, Any] = {}
    conv_in = _suffix_key(shapes, _DECODER_CONVIN_SUFFIX)
    if conv_in is not None:
        shape = shapes[conv_in]
        if len(shape) >= 2:
            observed["latent_channels"] = int(shape[1])
        # A Conv2d weight is 4-D and faces a [B,C,H,W] latent; a causal Conv3d
        # weight is 5-D and faces [B,C,T,H,W]. The rank carries over.
        observed["ndim"] = len(shape)

    ldm_keys = any(k.startswith("encoder.down.") or ".encoder.down." in k for k in shapes)
    bn_key = _suffix_key(shapes, _BN_RUNNING_MEAN_SUFFIX)
    if bn_key is not None:
        observed["norm"] = "batchnorm"
        channels = observed.get("latent_channels")
        numel = int(shapes[bn_key][0]) if shapes[bn_key] else 0
        if channels:
            packed = numel / channels
            root = int(round(packed ** 0.5))
            if root * root != int(packed) or packed != int(packed):
                raise VaeSourceError(
                    f"batchnorm statistics cover {numel} channels, which is not a "
                    f"square multiple of the {channels} latent channels; the "
                    "normalisation domain cannot be determined")
            observed["norm_pack"] = root
        observed["vae_class"] = "AutoencoderKLFlux2"
    elif ldm_keys:
        observed["vae_class"] = "AutoencoderKL"

    if observed.get("ndim") == 4:
        downsamplers = {
            key for key in shapes
            if any(key.endswith(marker) for marker in _DOWNSAMPLER_MARKERS)
        }
        if downsamplers:
            observed["scale_factor"] = 2 ** len(downsamplers)
        # A 2-D stack has no temporal axis to compress.
        observed["scale_temporal"] = 1
    return observed


def content_hash_for_state_dict(state_dict: Dict[str, torch.Tensor]) -> str:
    """sha256 of the tensor bytes in key order, first 16 hex (§5.2)."""
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key]
        digest.update(key.encode("utf-8"))
        digest.update(f"|{tuple(tensor.shape)}|{tensor.dtype}|".encode("utf-8"))
        flat = tensor.detach().to("cpu").contiguous().reshape(-1)
        digest.update(flat.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()[:16]


# ---------------------------------------------------------------------------
# The architecture's own VAE, for the two native flags
# ---------------------------------------------------------------------------

def arch_native_vae(arch: Optional[str]) -> Optional[Dict[str, Any]]:
    """The declared facts of ``arch``'s own VAE, or None for an unknown arch.

    Read from the wiring spec plus the per-arch VAE class map; a field the two
    do not carry (a video VAE's temporal ratio) is absent rather than assumed.
    """
    if not arch:
        return None
    from api.generation_overrides import VAE_CLASS_BY_ARCH
    from core.models.component_registry import _WIRING_BY_ARCH

    spec = _WIRING_BY_ARCH.get(arch)
    if spec is None:
        return None
    native: Dict[str, Any] = {
        "latent_channels": spec.latent_channels,
        "scale_factor": spec.vae_scale_factor,
        "ndim": spec.latent_ndim,
        "vae_class": VAE_CLASS_BY_ARCH.get(arch),
        "pixel_space": spec.latent_channels == 0,
    }
    if spec.latent_ndim == 4:
        native["scale_temporal"] = 1
    return native


def _native_flags(facts: Dict[str, Any], native: Optional[Dict[str, Any]],
                  native_hash: Optional[str]) -> Tuple[Optional[bool], Optional[bool]]:
    if not native or native.get("pixel_space"):
        return None, None
    structural = ("latent_channels", "scale_factor", "scale_temporal", "ndim",
                  "vae_class")
    unknown = False
    for key in structural:
        mine, theirs = facts.get(key), native.get(key)
        if mine is None or theirs is None:
            unknown = True
            continue
        if mine != theirs:
            return False, False
    struct_native = None if unknown else True
    if native_hash is None:
        return struct_native, None
    identity_native = bool(facts.get("content_hash") == native_hash)
    if identity_native and struct_native is None:
        struct_native = True
    return struct_native, identity_native


# ---------------------------------------------------------------------------
# Resolution (§7.3)
# ---------------------------------------------------------------------------

def _declared_from_metadata(metadata: Dict[str, str]) -> Dict[str, Any]:
    """The ``component.vae.*`` declaration a checkpoint/file carries, if any."""
    try:
        return dict(parse_component_metadata(metadata).get("vae") or {})
    except Exception:
        return {}


def _pick(observed: Dict[str, Any], key: str, *fallbacks, error: str) -> Any:
    """Observation wins; declarations fill the gap; nothing else is invented."""
    value = observed.get(key)
    if value is not None:
        return value
    for candidate in fallbacks:
        if candidate is not None:
            return candidate
    raise VaeSourceError(error)


def _resolve_facts(
    *,
    source: str,
    shapes: Dict[str, Tuple[int, ...]],
    config: Dict[str, Any],
    declared: Dict[str, Any],
    family: str,
) -> Dict[str, Any]:
    observed = observe_vae(shapes)
    if not observed:
        raise VaeSourceError(
            f"{source} carries no VAE weights (no '{_DECODER_CONVIN_SUFFIX}')")

    registry_entry = VAE_REGISTRY.get(family) or {}
    latent_channels = int(_pick(
        observed, "latent_channels",
        config.get("latent_channels"), declared.get("channels"),
        error=f"{source}: latent channel count is neither observable nor declared"))

    ndim = int(_pick(
        observed, "ndim", registry_entry.get("latent_ndim"),
        error=f"{source}: latent rank is neither observable nor declared"))

    scale_factor = int(_pick(
        observed, "scale_factor",
        config.get("spatial_compression_ratio"),
        _from_block_out_channels(config),
        declared.get("scale_factor"),
        registry_entry.get("scale_factor"),
        error=f"{source}: spatial compression ratio is neither observable nor declared"))
    scale_temporal = int(_pick(
        observed, "scale_temporal",
        config.get("temporal_compression_ratio"),
        declared.get("scale_temporal"),
        registry_entry.get("scale_temporal"),
        error=f"{source}: temporal compression ratio is neither observable nor declared"))

    vae_class = str(_pick(
        observed, "vae_class",
        config.get("_class_name"), declared.get("class"),
        registry_entry.get("class"),
        error=f"{source}: VAE class is neither observable nor declared"))

    scaling_factor = config.get("scaling_factor")
    shift_factor = config.get("shift_factor")
    if scaling_factor is None and family in VAE_REGISTRY:
        canonical = canonical_latent_scaling(family)
        if canonical is not None:
            scaling_factor = canonical[0]
            if shift_factor is None:
                shift_factor = canonical[1]

    has_per_channel = (config.get("latents_mean") is not None
                       and config.get("latents_std") is not None)
    if observed.get("norm"):
        norm = observed["norm"]
    elif has_per_channel:
        norm = "per_channel"
    elif declared.get("norm"):
        norm = str(declared["norm"])
    elif registry_entry.get("norm"):
        norm = str(registry_entry["norm"])
    elif scaling_factor is not None:
        norm = "shift_scale"
    else:
        raise VaeSourceError(
            f"{source}: latent normalisation cannot be determined. Neither the "
            "weights nor a config.json say how latents are scaled, and this "
            "resolver never substitutes diffusers' single-file default (see "
            "vae_store.LDM_SINGLE_FILE_DEFAULT_SCALING_FACTOR). Select the VAE "
            "family under registry:, or a source carrying a config.json")

    if norm == "shift_scale" and scaling_factor is None:
        raise VaeSourceError(
            f"{source}: declares shift_scale normalisation but no scaling_factor "
            "is available; refusing to guess one")
    if norm == "per_channel" and not has_per_channel:
        raise VaeSourceError(
            f"{source}: declares per-channel normalisation but no "
            "latents_mean/latents_std are available")

    norm_pack = int(observed.get("norm_pack")
                    or declared.get("norm_pack")
                    or registry_entry.get("norm_pack")
                    or 1)

    return {
        "latent_channels": latent_channels,
        "ndim": ndim,
        "scale_factor": scale_factor,
        "scale_temporal": scale_temporal,
        "vae_class": vae_class,
        "norm": norm,
        "norm_pack": norm_pack,
        "scaling_factor": (float(scaling_factor) if scaling_factor is not None else None),
        "shift_factor": (float(shift_factor) if shift_factor is not None else None),
    }


def _from_block_out_channels(config: Dict[str, Any]) -> Optional[int]:
    """Spatial ratio of a 2-D diffusers VAE, the expression diffusers' own
    pipelines use for ``vae_scale_factor``."""
    blocks = config.get("block_out_channels")
    if isinstance(blocks, (list, tuple)) and blocks:
        return 2 ** (len(blocks) - 1)
    return None


def _registry_source(key: str, download: bool) -> str:
    """The local path a registry family resolves to: a diffusers directory, or
    the single weight file an original/LDM release leaves in the shared store.

    A non-``diffusers_repo`` family is never downloaded here: ``resolve_vae_dir``
    requires a ``config.json`` the release does not have, so the fetch would
    always end in the same refusal.
    """
    if key not in VAE_REGISTRY:
        raise VaeSourceError(
            f"unknown VAE registry key '{key}' (known: {', '.join(VAE_REGISTRY)})")
    entry = VAE_REGISTRY[key]
    if entry.get("diffusers_repo"):
        directory = resolve_vae_dir(key, download=download)
        if directory:
            return directory
    store = store_dir_for(key)
    if store and os.path.isdir(store):
        try:
            return _diffusers_weight_path(store)
        except VaeSourceError:
            pass
    raise VaeSourceError(
        f"registry:{key} is not available locally (default repo "
        f"{entry['default_repo']}); download it into the shared VAE store first")


def resolve_vae_source(
    source: str,
    *,
    arch: Optional[str] = None,
    native_hash: Optional[str] = None,
    download: bool = True,
    load_weights: bool = True,
) -> ResolvedVAE:
    """Resolve ``source`` into a ``ResolvedVAE``, or raise ``VaeSourceError``.

    ``arch`` (when given) decides ``struct_native``; ``identity_native`` also
    needs ``native_hash``, the content hash of that architecture's own VAE —
    without it the answer is None (unknown), never False-by-assumption.
    """
    form, value = parse_vae_source(source)
    family = "custom"
    config: Dict[str, Any] = {}
    declared: Dict[str, Any] = {}
    prefix: Optional[str] = None
    path: Optional[str] = None

    if form == "registry":
        resolved_path = _registry_source(value, download)
        family = value
        path = resolved_path
        provenance = f"registry:{value}"
        locator = f"registry:{value}"
        if os.path.isdir(resolved_path):
            config = _read_json(os.path.join(resolved_path, "config.json"))
            weights = _WeightFile(_diffusers_weight_path(resolved_path))
        else:
            weights = _WeightFile(resolved_path)
            declared = _declared_from_metadata(weights.metadata)
            config = dict(declared.get("config") or {})
        keys = list(weights.shapes)

    elif form == "file":
        abs_path = os.path.abspath(value)
        directory = _vae_dir(abs_path)
        if directory is not None:
            path = directory
            config = _read_json(os.path.join(directory, "config.json"))
            weights = _WeightFile(_diffusers_weight_path(directory))
        elif os.path.isfile(abs_path):
            path = abs_path
            weights = _WeightFile(abs_path)
            declared = _declared_from_metadata(weights.metadata)
            config = dict(declared.get("config") or {})
        else:
            raise VaeSourceError(
                f"file:{value} is neither a diffusers VAE directory nor a file")
        keys = list(weights.shapes)
        provenance = f"file:{os.path.basename(path.rstrip(os.sep))}"
        locator = f"path:{path}"

    else:  # model:
        abs_path = os.path.abspath(value)
        if not os.path.exists(abs_path):
            raise VaeSourceError(f"model:{value} does not exist")
        path = abs_path
        weights = _WeightFile(abs_path)
        # The values are the keys themselves: the split decides WHICH keys and
        # under which prefix without materialising a single tensor, so the
        # extraction stays bounded by the VAE rather than the checkpoint.
        buckets = split_prefixed_state_dict(
            {key: key for key in weights.shapes}, VAE_EXTRACT_PREFIXES)
        prefix = next((p for p in VAE_EXTRACT_PREFIXES if buckets[p]), None)
        if prefix is None:
            raise VaeSourceError(
                f"model:{value} carries no VAE under any of "
                f"{list(VAE_EXTRACT_PREFIXES)}")
        if not buckets[""]:
            raise VaeSourceError(
                f"model:{value} is a standalone VAE, not a full checkpoint; "
                "select it as file:<path>")
        declared = _declared_from_metadata(weights.metadata)
        config = dict(declared.get("config") or {})
        if declared.get("type") in VAE_REGISTRY:
            family = str(declared["type"])
        keys = [f"{prefix}{stripped}" for stripped in buckets[prefix]]
        shapes = {stripped: weights.shapes[f"{prefix}{stripped}"]
                  for stripped in buckets[prefix]}
        provenance = f"extracted:{os.path.splitext(os.path.basename(abs_path))[0]}"
        # An extracted VAE lives only inside its source checkpoint, so it has no
        # locator a reader could resolve later (§8.7 refuses not bundling it).
        locator = None

    if form != "model":
        if declared.get("type") in VAE_REGISTRY and family == "custom":
            family = str(declared["type"])
        shapes = dict(weights.shapes)

    facts = _resolve_facts(source=source, shapes=shapes, config=config,
                           declared=declared, family=family)

    state_dict: Optional[Dict[str, torch.Tensor]] = None
    content_hash = ""
    if load_weights:
        raw = weights.tensors(keys)
        if prefix:
            state_dict = {key[len(prefix):]: value for key, value in raw.items()}
        else:
            state_dict = raw
        content_hash = content_hash_for_state_dict(state_dict)

    facts["content_hash"] = content_hash
    struct_native, identity_native = _native_flags(
        facts, arch_native_vae(arch), native_hash)

    return ResolvedVAE(
        source=source,
        form=form,
        family=family,
        latent_channels=facts["latent_channels"],
        scale_factor=facts["scale_factor"],
        scale_temporal=facts["scale_temporal"],
        ndim=facts["ndim"],
        norm=facts["norm"],
        norm_pack=facts["norm_pack"],
        vae_class=facts["vae_class"],
        config=config,
        content_hash=content_hash,
        provenance=provenance,
        locator=locator,
        struct_native=struct_native,
        identity_native=identity_native,
        scaling_factor=facts["scaling_factor"],
        shift_factor=facts["shift_factor"],
        path=path,
        prefix=prefix,
        state_dict=state_dict,
    )


def describe_vae_source(source: str, *, arch: Optional[str] = None,
                        download: bool = False) -> Dict[str, Any]:
    """The structural facts of ``source`` plus ``compatible``/``reason``.

    Header and config reads only: no tensor data, no content hash. This is what
    a candidate listing calls; ``resolve_vae_source`` is what a run calls.
    """
    try:
        resolved = resolve_vae_source(source, arch=arch, download=download,
                                      load_weights=False)
    except VaeSourceError as e:
        return {"source": source, "compatible": False, "reason": str(e)}
    except Exception as e:  # a malformed file must not take the listing down
        return {"source": source, "compatible": False,
                "reason": f"{type(e).__name__}: {e}"}
    out = resolved.facts()
    out.pop("content_hash", None)
    compatible, reason = check_vae_compatibility(out, arch)
    out["compatible"] = compatible
    out["reason"] = reason
    if arch == "sensenova":
        out.update(sensenova_token_geometry(resolved.scale_factor))
    return out


# ---------------------------------------------------------------------------
# family compatibility gate (§7.4)
# ---------------------------------------------------------------------------

#: Pixel-space architectures that accept ANY spatial compression ratio, because
#: moving to a latent space is the point of the swap there (D13, §10.2). Every
#: other pixel-space arch has no latent geometry to check a candidate against.
_PIXEL_SPACE_EXEMPT = ("sensenova",)


def check_vae_compatibility(facts: Dict[str, Any],
                            arch: Optional[str]) -> Tuple[bool, Optional[str]]:
    """§7.4's hard gate, server-side, over already-resolved facts."""
    native = arch_native_vae(arch)
    if native is None:
        return True, None

    if native.get("pixel_space"):
        if arch in _PIXEL_SPACE_EXEMPT:
            # D13: any spatial ratio is accepted; the token grid absorbs it.
            return True, None
        return False, (
            f"{arch} is pixel-space; its latent migration is not implemented")

    ndim = facts.get("ndim")
    if ndim is not None and native.get("ndim") is not None and ndim != native["ndim"]:
        return False, (
            f"{ndim}-D latents cannot drive {arch}, which expects "
            f"{native['ndim']}-D")

    scale = facts.get("scale_factor")
    if scale is not None and scale != native.get("scale_factor"):
        return False, (
            f"spatial compression {scale}x differs from {arch}'s "
            f"{native.get('scale_factor')}x")

    temporal = facts.get("scale_temporal")
    native_temporal = native.get("scale_temporal")
    if native_temporal is None:
        return False, (
            f"{arch}'s own temporal compression ratio is not declared, so a "
            "replacement cannot be checked against it")
    if temporal is not None and temporal != native_temporal:
        return False, (
            f"temporal compression {temporal}x differs from {arch}'s "
            f"{native_temporal}x")

    # The shared normalisation layer (§8.4) is not built yet, so a swap may not
    # cross normalisation domains in either direction.
    from core.models.component_registry import _WIRING_BY_ARCH
    spec = _WIRING_BY_ARCH.get(arch)
    arch_norm = getattr(spec, "vae_norm", None)
    if facts.get("norm") == "batchnorm" and arch_norm != "batchnorm":
        return False, (
            f"this VAE normalises with a BatchNorm over its packed latent, which "
            f"{arch} does not; blocked until the shared normalisation layer lands")
    if arch_norm == "batchnorm" and facts.get("norm") != "batchnorm":
        return False, (
            f"{arch} normalises latents with the VAE's own BatchNorm, which this "
            "VAE does not carry; blocked until the shared normalisation layer lands")
    return True, None


def sensenova_token_geometry(scale_factor: int) -> Dict[str, Any]:
    """SenseNova's token width and recommended resolution band for a VAE (§10.2).

    The generation-side patch is fixed at P=4 on the latent grid, so one token
    covers ``4 * scale_factor`` pixels and the current 3-5 MP band moves with
    ``(scale_factor / 8) ** 2``.
    """
    scale = int(scale_factor or 1)
    ratio = (scale / 8.0) ** 2
    return {
        "token_pixel_width": 4 * scale,
        "resolution_band_px": [round(3.0e6 * ratio), round(5.0e6 * ratio)],
    }
