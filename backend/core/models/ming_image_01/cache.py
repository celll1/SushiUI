"""Content-bound native Ming tensor caches with atomic CPU persistence."""

import hashlib
import json
import os
from pathlib import Path
import tempfile

from safetensors import safe_open, SafetensorError
from safetensors.torch import save_file
import torch


VERSION = 1
KINDS = {"latents", "conditioning"}


def identity_key(descriptor):
    if not isinstance(descriptor, dict) or not descriptor:
        raise ValueError("Ming cache requires an explicit identity descriptor")
    return hashlib.sha256(json.dumps(descriptor, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def _tensor_hash(value):
    if value.device.type != "cpu" or value.layout != torch.strided or value.is_quantized:
        raise ValueError("Ming cache hashes require plain CPU tensors")
    raw = value.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(raw)).hexdigest()


def module_fingerprint(module):
    """Call before residency changes; content identity must include actual weights."""
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(json.dumps([name, list(value.shape), str(value.dtype), _tensor_hash(value)]).encode())
    return digest.hexdigest()


def latent_identity(*, pixels, vae_weights, vae_config, dtype, execution):
    return identity_key(dict(version=VERSION, kind="latents", pixels=pixels,
        vae_weights=vae_weights, vae_config=vae_config, dtype=str(dtype),
        layout="independent_rgba_frames_v1", posterior="mode", execution=execution))


def conditioning_identity(*, pixels, captions, artifact_config, conditioner_weights, resolution, dtype, topology, execution):
    return identity_key(dict(version=VERSION, kind="conditioning", pixels=pixels,
        captions=captions, artifact_config=artifact_config, conditioner_weights=conditioner_weights,
        resolution=resolution, dtype=str(dtype), topology=topology,
        layout="native_queries_direct_v1", execution=execution))


def _validate(kind, tensors):
    if kind not in KINDS or not tensors:
        raise ValueError("Unknown or empty Ming tensor cache")
    if any(not isinstance(value, torch.Tensor) or not value.is_floating_point()
           or value.device.type != "cpu" or min(value.shape, default=0) < 1
           or not torch.isfinite(value).all() for value in tensors.values()):
        raise ValueError("Ming cache tensors must be nonempty finite CPU floats")
    if kind == "conditioning":
        if (set(tensors) != {"queries", "direct"} or any(value.ndim != 3 for value in tensors.values())
                or tensors["queries"].shape[0] != tensors["direct"].shape[0]
                or tensors["queries"].dtype != tensors["direct"].dtype):
            raise ValueError("Ming conditioning cache requires aligned dual streams")
    else:
        clean = tensors.get("clean")
        if clean is None or clean.ndim != 5 or clean.shape[1] != 16:
            raise ValueError("Ming latent cache requires [B, 16, F, H, W]")
        for name, value in tensors.items():
            if name == "clean": continue
            if not name.startswith("reference_") or not name[10:].isdigit():
                raise ValueError("Unknown Ming latent cache tensor role")
            index = int(name[10:])
            if (name != f"reference_{index}" or index >= clean.shape[0] or value.ndim != 4
                    or value.shape[0] != 16 or value.shape[-2:] != clean.shape[-2:]):
                raise ValueError("Ming cache references disagree with their native batch/grid")


class MingTensorCache:
    def __init__(self, root):
        self.root = Path(root)

    def _path(self, kind, key):
        if kind not in KINDS or not isinstance(key, str) or len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("Invalid Ming cache kind/key")
        return self.root / kind / (key + ".safetensors")

    @staticmethod
    def _frozen(kind, conditioner):
        if kind == "conditioning" and (conditioner is None or not conditioner.cacheable):
            raise ValueError("Ming conditioning persistence requires a completely frozen conditioner")

    def write(self, kind, key, tensors, *, conditioner=None, cancelled=None):
        self._frozen(kind, conditioner)
        _validate(kind, tensors)
        tensors = {name: value.detach().contiguous().clone() for name, value in tensors.items()}
        path = self._path(kind, key)
        metadata = {"version": str(VERSION), "kind": kind, "identity": key,
            "tensors": json.dumps({name: [list(value.shape), str(value.dtype), _tensor_hash(value)]
                                   for name, value in tensors.items()}, sort_keys=True)}
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".ming-cache-", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
            save_file(tensors, str(temporary), metadata=metadata)
            if cancelled is not None and cancelled():
                raise RuntimeError("Ming cache write cancelled")
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists(): temporary.unlink()

    def read(self, kind, key, *, expected, conditioner=None):
        self._frozen(kind, conditioner)
        path = self._path(kind, key)
        if not path.is_file(): return None
        try:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                metadata = handle.metadata()
                if (metadata is None or metadata.get("version") != str(VERSION)
                        or metadata.get("kind") != kind or metadata.get("identity") != key
                        or set(handle.keys()) != set(expected)):
                    return None
                spec = json.loads(metadata["tensors"])
                tensors = {name: handle.get_tensor(name) for name in handle.keys()}
            _validate(kind, tensors)
            if set(spec) != set(tensors): return None
            for name, value in tensors.items():
                shape, dtype = expected[name]
                actual = [list(value.shape), str(value.dtype), _tensor_hash(value)]
                if tuple(value.shape) != tuple(shape) or value.dtype != dtype or spec[name] != actual:
                    return None
            return tensors
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, SafetensorError):
            return None



def expected_conditioning(conditioner, prepared):
    native = conditioner.native
    inputs = prepared.model_inputs
    patch = native.config.llm_config.image_patch_token
    _, labels, mask, _ = native.append_input_ids_with_multiscale_learnable_tokens(
        inputs["input_ids"], inputs["attention_mask"], native.img_gen_scales,
        patch+1, patch+2, patch)
    use_input = (labels < 0).to(mask.dtype) * mask
    length = int(use_input.cumsum(-1).argmax(-1).max()) + 1
    batch = inputs["input_ids"].shape[0]
    dtype = next(conditioner.parameters()).dtype
    return {"queries": ((batch, native.img_gen_scales[-1]**2, native.proj_out.out_features), dtype),
            "direct": ((batch, length, native.proj_directvlm[1].out_features), dtype)}


def snapshot_cache_context(components, cache, *, execution, latent_enabled, conditioning_enabled):
    """Snapshot frozen CPU producer identity before any execution residency changes."""
    if conditioning_enabled and not components.conditioner.cacheable:
        raise ValueError("Persistent Ming conditioning requires frozen ownership")
    return dict(cache=cache, execution=execution, latent_enabled=latent_enabled,
        conditioning_enabled=conditioning_enabled,
        vae_weights=module_fingerprint(components.vae) if latent_enabled else None,
        conditioner_weights=module_fingerprint(components.conditioner) if conditioning_enabled else None,
        vae_config=dict(components.vae.config), artifact_config=components.artifact.config_fingerprint,
        vae_dtype=components.vae.dtype, conditioner_dtype=next(components.conditioner.parameters()).dtype,
        vae_versions=parameter_versions(components.vae),
        conditioner_versions=parameter_versions(components.conditioner))



def parameter_versions(module):
    return tuple((id(value), value._version) for value in module.parameters())


def validate_cache_context(components, context):
    if context["latent_enabled"] and parameter_versions(components.vae) != context["vae_versions"]:
        raise ValueError("Ming VAE weights changed after cache identity capture; recreate the cache context")
    if context["conditioning_enabled"] and parameter_versions(components.conditioner) != context["conditioner_versions"]:
        raise ValueError("Ming conditioner weights changed after cache identity capture; recreate the cache context")



def configured_modes(config):
    from api.param_defaults import TRAINING_DEFAULTS
    modes = {name: config.get(name, TRAINING_DEFAULTS[name])
             for name in ("latent_encoding_mode", "text_encoding_mode")}
    for name, value in modes.items():
        accepted = {"swap_onthefly", "pre_encoded_cache", "onthefly_gpu"}
        if name == "text_encoding_mode": accepted.add("cpu_prefetch")
        if not isinstance(value, str) or value not in accepted:
            raise ValueError(f"Unsupported Ming {name}: {value!r}")
    return modes


def capture_startup_snapshot(trainer):
    import transformers
    components = trainer.ming_image_01_components
    modes = configured_modes(trainer.config)
    latent = modes["latent_encoding_mode"] == "pre_encoded_cache"
    condition = modes["text_encoding_mode"] == "pre_encoded_cache" and components.conditioner.cacheable
    trainer.ming_cache_snapshot = None
    trainer.ming_requested_encoding_modes = modes
    if not latent and not condition:
        return
    execution = {"implementation": "ming_rgba_dual_v1", "torch": str(torch.__version__),
        "transformers": transformers.__version__, "conditioner_attention": "eager",
        "target_device": str(torch.device(getattr(trainer, "device", "cpu")).type),
        "vae_tiling": bool(getattr(components.vae, "use_tiling", False)),
        "vae_slicing": bool(getattr(components.vae, "use_slicing", False))}
    trainer.ming_cache_snapshot = snapshot_cache_context(components, None,
        execution=execution, latent_enabled=latent, conditioning_enabled=condition)


def attach_startup_snapshot(trainer, *, latent_mode, text_mode):
    configured_modes({"latent_encoding_mode": latent_mode, "text_encoding_mode": text_mode})
    components = trainer.ming_image_01_components
    latent = latent_mode == "pre_encoded_cache"
    condition = text_mode == "pre_encoded_cache" and components.conditioner.cacheable
    trainer.ming_cache_context = None
    if not latent and not condition:
        return
    snapshot = getattr(trainer, "ming_cache_snapshot", None)
    if (snapshot is None or (latent and not snapshot["latent_enabled"])
            or (condition and not snapshot["conditioning_enabled"])):
        raise ValueError("Ming cache mode must be configured before the native CPU model load")
    validate_cache_context(components, snapshot)
    if components.vae.dtype != snapshot["vae_dtype"] or next(components.conditioner.parameters()).dtype != snapshot["conditioner_dtype"]:
        raise ValueError("Ming producer precision changed since its startup cache snapshot")
    for name, attribute in (("vae_tiling", "use_tiling"), ("vae_slicing", "use_slicing")):
        if bool(getattr(components.vae, attribute, False)) != snapshot["execution"][name]:
            raise ValueError("Ming VAE execution policy changed since its startup cache snapshot")
    if torch.device(trainer.device).type != snapshot["execution"]["target_device"]:
        raise ValueError("Ming encoder execution device changed since its startup cache snapshot")
    trainer.ming_cache_context = {**snapshot, "cache": MingTensorCache(Path(trainer.output_dir) / "ming_cache"),
                                "latent_enabled": latent, "conditioning_enabled": condition}
