"""Bounded, process-local reuse of frozen encoder work; never stores sampled noise."""
from collections import OrderedDict
from contextlib import contextmanager
from functools import wraps
from hashlib import sha256
import inspect
import json
import threading
from types import SimpleNamespace

import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution


def tree(value, transform):
    if isinstance(value, torch.Tensor):
        return transform(value)
    if isinstance(value, dict):
        return {key: tree(item, transform) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(tree(item, transform) for item in value)
    if isinstance(value, list):
        return [tree(item, transform) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("Unsupported encoder cache payload")


class EncoderReuseCache:
    def __init__(self, max_bytes, max_entries=None):
        if max_entries is None:
            from api.param_defaults import ENCODER_REUSE_DEFAULTS
            max_entries = ENCODER_REUSE_DEFAULTS["max_entries"]
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.entries = OrderedDict()
        self.bytes = 0
        self.hits = self.misses = self.evictions = 0
        self.lock = threading.RLock()

    def get(self, key):
        with self.lock:
            if key not in self.entries:
                self.misses += 1
                return None
            self.hits += 1
            self.entries.move_to_end(key)
            return self.entries[key][0]

    def put(self, key, value):
        size = 0
        def count(tensor):
            nonlocal size
            size += tensor.numel() * tensor.element_size()
            return tensor
        tree(value, count)
        if size > self.max_bytes:
            return
        payload = tree(value, lambda tensor: tensor.detach().to("cpu", copy=True))
        with self.lock:
            previous = self.entries.pop(key, None)
            if previous:
                self.bytes -= previous[1]
            while self.entries and (self.bytes + size > self.max_bytes or len(self.entries) >= self.max_entries):
                _, (_, removed) = self.entries.popitem(last=False)
                self.bytes -= removed
                self.evictions += 1
            self.entries[key] = (payload, size)
            self.bytes += size


def owner_cache(owner):
    cache = getattr(owner, "_encoder_reuse_cache", None)
    if cache is None:
        from api.param_defaults import ENCODER_REUSE_DEFAULTS
        cache = owner._encoder_reuse_cache = EncoderReuseCache(**ENCODER_REUSE_DEFAULTS)
    return cache


def autocast_identity(device):
    kind = torch.device(device).type
    enabled = torch.is_autocast_enabled(kind)
    return kind, enabled, str(torch.get_autocast_dtype(kind)) if enabled else None


def tokenizer_identity(tokenizer):
    if tokenizer is None:
        return None
    settings = {name: getattr(tokenizer, name, None) for name in (
        "init_kwargs", "special_tokens_map", "added_tokens_encoder", "padding_side",
        "truncation_side", "model_max_length", "chat_template")}
    return id(tokenizer), sha256(json.dumps(settings, sort_keys=True, default=str).encode()).digest()


def producer_identity(module):
    if module is None or not isinstance(module, torch.nn.Module):
        return None
    if any(child.training for child in module.modules()):
        return None
    parameters, buffers = list(module.named_parameters()), list(module.named_buffers())
    tensors = [value for _, value in parameters + buffers]
    if any(value.requires_grad for value in tensors):
        return None
    try:
        state = [(name, id(value), value._version, str(value.dtype), tuple(value.shape))
                 for name, value in parameters]
    except RuntimeError:
        return None
    for name, value in buffers:
        # Tiny normalization/RoPE buffers may be replaced by module.to(); hash their values.
        try:
            identity = (input_digest(value) if value.numel() <= 4096 else
                        (id(value), value._version, str(value.device)))
        except RuntimeError:
            return None
        state.append((name, identity, str(value.dtype), tuple(value.shape)))
    state.append(json.dumps(getattr(module, "config", {}), sort_keys=True, default=str))
    state.append(tuple((name, getattr(module, name, None)) for name in (
        "use_tiling", "use_slicing", "tile_sample_min_size", "tile_latent_min_size", "tile_overlap_factor")))
    return id(module), sha256(repr(state).encode()).digest()


@contextmanager
def reuse_vae_posterior(owner, vae):
    """Wrap the raw codec, preserving native posterior sampling and normalization."""
    if getattr(owner, "_latent_encoding_mode", None) != "swap_onthefly":
        yield
        return
    from core.models.components.latent_adapter import NormalizedFrameVAE
    from core.models.sdxl_vae_wrapper import SDXLVAEWrapper
    if isinstance(vae, NormalizedFrameVAE):
        vae = vae.inner
    if isinstance(vae, SDXLVAEWrapper):
        vae = vae.vae
    identity = producer_identity(vae)
    if identity is None:
        yield
        return
    cache = owner_cache(owner)
    encode = vae.encode
    had_local = "encode" in vae.__dict__
    local_encode = vae.__dict__.get("encode")

    def cached(pixels, *args, **kwargs):
        if args or kwargs or producer_identity(vae) != identity:
            return encode(pixels, *args, **kwargs)
        cpu = pixels.detach().contiguous().cpu()
        digest = sha256(cpu.view(torch.uint8).numpy().tobytes()).digest()
        key = ("posterior", identity, autocast_identity(pixels.device), tuple(cpu.shape), str(cpu.dtype), digest)
        del cpu
        saved = cache.get(key)
        if saved is not None:
            posterior = DiagonalGaussianDistribution(saved["parameters"].to(pixels.device, copy=True),
                deterministic=saved["deterministic"])
            return SimpleNamespace(latent_dist=posterior)
        result = encode(pixels)
        posterior = getattr(result, "latent_dist", None)
        if type(posterior) is DiagonalGaussianDistribution:
            cache.put(key, {"parameters": posterior.parameters, "deterministic": posterior.deterministic})
        return result

    vae.encode = cached
    try:
        yield
    finally:
        if had_local:
            vae.encode = local_encode
        else:
            del vae.encode


def caption_context(owner, requires_grad=False, references=None):
    if getattr(getattr(owner, "arch", None), "name", None) not in {
            "anima", "sd15", "sdxl", "qwen_image_21", "zimage", "flux2", "lens", "krea2"}:
        return None
    if requires_grad or references or getattr(owner, "_text_encoding_mode", None) not in ("swap_onthefly", "cpu_prefetch"):
        return None
    modules = [getattr(owner, name, None) for name in ("text_encoder", "text_encoder_2")]
    modules = [module for module in modules if module is not None]
    identities = [producer_identity(module) for module in modules]
    if not identities or any(identity is None for identity in identities):
        return None
    config = sha256(json.dumps(getattr(owner, "config", {}), sort_keys=True, default=str).encode()).digest()
    device = next(modules[0].parameters(), torch.empty(0)).device
    key = ("text", tuple(identities), config, str(getattr(owner, "training_dtype", None)),
           autocast_identity(device), *(tokenizer_identity(getattr(owner, name, None))
           for name in ("tokenizer", "tokenizer_2", "t5_tokenizer")))
    return owner_cache(owner), key, device


def log_reuse(owner):
    cache = getattr(owner, "_encoder_reuse_cache", None)
    if cache is not None:
        print(f"{getattr(owner, 'log_prefix', '[Training]')} Encoder reuse: hits={cache.hits}, misses={cache.misses}, "
              f"evictions={cache.evictions}, CPU={cache.bytes / 1024**2:.1f}MiB")


def input_digest(value):
    digest = sha256()
    def collect(item):
        if isinstance(item, torch.Tensor):
            cpu = item.detach().contiguous().cpu()
            digest.update(repr((tuple(cpu.shape), str(cpu.dtype))).encode())
            digest.update(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item):
                digest.update(repr(key).encode())
                collect(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(repr((type(item).__name__, len(item))).encode())
            for child in item:
                collect(child)
        else:
            digest.update(repr(item).encode())
    collect(value)
    return digest.digest()


def reuse_ming_conditioning(owner, source):
    identity = producer_identity(source.conditioner)
    if identity is None or not source.conditioner.cacheable:
        return source.for_step().detached()
    from core.models.ming_image_01.conditioning import MingConditioning
    prepared = source.prepared
    key = ("ming_text", identity, prepared.width, prepared.height,
           input_digest(prepared.model_inputs),
           sha256(json.dumps(owner.config, sort_keys=True, default=str).encode()).digest())
    cache = owner_cache(owner)
    payload = cache.get(key)
    if payload is not None:
        return MingConditioning(**tree(payload, lambda value: value.clone()))
    result = source.for_step().detached()
    cache.put(key, {"queries": result.queries, "direct": result.direct})
    return result


def reuse_caption_encoding(function):
    signature = inspect.signature(function)
    @wraps(function)
    def wrapped(owner, *args, **kwargs):
        bound = signature.bind(owner, *args, **kwargs)
        bound.apply_defaults()
        params = bound.arguments
        context = caption_context(owner, params["requires_grad"], params["reference_image_paths"])
        if context is None:
            return function(owner, *args, **kwargs)
        cache, prefix, device = context
        key = (prefix, "single", params["caption"], params["lyrics"], params["cfg_null"])
        payload = cache.get(key)
        if payload is not None:
            return tree(payload, lambda value: value.to(device, copy=True))
        result = function(owner, *args, **kwargs)
        try:
            cache.put(key, result)
        except TypeError:
            pass
        return result
    return wrapped


def reuse_caption_batch(function):
    signature = inspect.signature(function)
    @wraps(function)
    def wrapped(owner, *args, **kwargs):
        bound = signature.bind(owner, *args, **kwargs)
        bound.apply_defaults()
        params = bound.arguments
        captions, lyrics = list(params["captions"]), params["lyrics"]
        if lyrics is not None and len(lyrics) != len(captions):
            return function(owner, *args, **kwargs)
        # Other architectures' batch fallback already uses the single-item cache.
        context = (caption_context(owner, params["requires_grad"])
                   if getattr(getattr(owner, "arch", None), "name", None) == "anima" else None)
        if context is None or not captions:
            return function(owner, *args, **kwargs)
        cache, prefix, device = context
        lyrics = list(lyrics) if lyrics is not None else [""] * len(captions)
        keys = [(prefix, "batch", caption, lyric or "", False) for caption, lyric in zip(captions, lyrics)]
        ready, missing = {}, OrderedDict()
        for index, key in enumerate(keys):
            if key in ready or key in missing:
                continue
            payload = cache.get(key)
            if payload is None:
                missing[key] = index
            else:
                ready[key] = tree(payload, lambda value: value.to(device, copy=True))
        if missing:
            indices = list(missing.values())
            results = function(owner, [captions[index] for index in indices],
                requires_grad=params["requires_grad"], lyrics=[lyrics[index] for index in indices])
            if len(results) != len(indices):
                raise ValueError("Batched text encoder returned the wrong item count")
            for key, result in zip(missing, results):
                ready[key] = result
                try:
                    cache.put(key, result)
                except TypeError:
                    pass
        return [tree(ready[key], lambda value: value.clone()) for key in keys]
    return wrapped
