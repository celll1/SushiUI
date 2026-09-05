"""Resolving and applying a run's VAE swap (design §5.3, §7.4, §8.1, §8.3).

One place decides what ``vae_swap_source`` means, so the route preflight, the
train_runner preflight and the loader cannot disagree about whether a run swaps
its VAE and whether the combination is allowed.

``sdxl_vae_type`` survives here as a READ-ONLY alias: an old YAML naming a
registry family keeps loading, and nothing writes it back.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

# Values of the legacy SDXL-only key that mean "no swap".
_LEGACY_NEUTRAL = ("", "none", "sdxl")


def legacy_source_from_config(config: Dict[str, Any]) -> str:
    """``sdxl_vae_type: flux1`` -> ``registry:flux1``; neutral values -> ``""``."""
    legacy = str((config or {}).get("sdxl_vae_type", "") or "").strip().lower()
    if legacy in _LEGACY_NEUTRAL:
        return ""
    return f"registry:{legacy}"


def resolve_vae_swap_source(config: Dict[str, Any]) -> str:
    """This run's VAE source, new key first, legacy alias second (§5.3)."""
    source = str((config or {}).get("vae_swap_source", "") or "").strip()
    return source or legacy_source_from_config(config)


def legacy_vae_type_marker(resolved: Optional[Any]) -> str:
    """What ``sdxl_vae_type`` may hold for this identity (``"sdxl"`` = no swap).

    The SDXL adapter writes this value straight back out as ``sushi.vae_type``,
    which can only name a registry family: a ``file:``/``model:`` VAE has no
    legacy spelling, and writing ``"custom"`` there would make the next load
    resolve ``registry:custom``. Such a run declares itself through
    ``component.vae.*`` alone.
    """
    if (resolved is None or resolved.identity_native
            or resolved.form != "registry"):
        return "sdxl"
    return str(resolved.family)


def check_swap_method(source: str, method: str) -> None:
    """§8.3: a swap rebuilds the latent-facing layers, which only a full
    fine-tune trains and saves. Refuse every other method."""
    if not source:
        return
    if str(method or "").strip().lower() not in ("full", "full_finetune"):
        raise ValueError(
            f"vae_swap_source={source!r} requires training_method='full_finetune' — "
            "LoRA cannot train the resized latent input/output layers, and the "
            "LoRA save path does not persist them. Switch to Full Fine-tune."
        )


def check_capability(source: str, arch: Optional[str], method: str) -> None:
    """The served refusal, enforced. An architecture whose wave has not landed
    is listed in ``api/arch_capabilities`` -- the only place that decision lives
    -- and the frontend hides the control; this is the same answer for a caller
    that posts the field anyway or hand-writes the YAML."""
    from api.arch_capabilities import training_feature_unsupported_reason

    reason = training_feature_unsupported_reason(arch, "vae_swap", method)
    if reason:
        raise ValueError(f"vae_swap_source={source!r} is not supported for "
                         f"{arch or 'this model'}: {reason}")


def check_bundling(source: str, bundle_vae_explicit_false: bool) -> None:
    """D7 / §8.7: an extracted VAE has no locator a reader could resolve, so a
    ``model:`` swap that refuses to bundle would produce a checkpoint nothing can
    load. Refused here, before training starts, not at save time."""
    if source.startswith("model:") and bundle_vae_explicit_false:
        raise ValueError(
            f"vae_swap_source={source!r} extracts the VAE from another checkpoint, "
            "which leaves no reference a later load could resolve; bundle_vae=false "
            "would produce a checkpoint that cannot be loaded. Either bundle the VAE "
            "(the default for a swap run) or select the same VAE as file:/registry:."
        )


def check_inherited_bundling(base_model_path: Optional[str], arch: Optional[str],
                             bundle_vae_explicit_false: bool) -> None:
    """The same refusal for a run that inherits a swapped base rather than
    naming a source: a VAE that exists only inside the base has no locator
    either, so this run's save cannot leave it out."""
    if not bundle_vae_explicit_false or not base_model_path:
        return
    from core.models.common.vae_source import load_declared_latent_io

    try:
        declared = load_declared_latent_io(base_model_path, arch=arch,
                                           load_weights=False, download=False)
    except Exception:
        # An unresolvable declaration is the loader's refusal to make, with the
        # weights in hand; this check only answers the bundling question.
        return
    if declared is not None and not declared.locator:
        raise ValueError(
            f"{base_model_path} carries its own swapped VAE and no reference to "
            "resolve it elsewhere, so bundle_vae=false would produce a checkpoint "
            "that cannot be loaded. Leave bundle_vae unset for this run.")


def check_arch_vae_compatibility(facts: Dict[str, Any], arch: Optional[str], *,
                                 base_model_path: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """The family gate (§7.4) for an arch named by string, with no trainer yet.

    Goes through the arch handler so an architecture whose latent geometry is a
    per-checkpoint fact (MiniT2I) answers from the base checkpoint rather than
    from a wiring constant that describes only one of its shapes.
    """
    from core.models.common.vae_source import check_vae_compatibility
    from core.training.arch import ARCH_REGISTRY

    handler_cls = ARCH_REGISTRY.get(str(arch or ""))
    if handler_cls is None:
        return check_vae_compatibility(facts, arch)
    return handler_cls(None).check_vae_compatibility(
        facts, base_model_path=base_model_path)


def apply_latent_space(trainer, declared) -> None:
    """Fold the base's declared VAE and this run's ``vae_swap_source`` into the
    trainer (design §8.1-8.3). ``declared`` is what the loader resolved from
    ``component.vae.*``, or None for a native checkpoint.

    Every architecture's ``load_components`` calls this once, after the backbone
    and VAE are on the trainer and BEFORE the freeze/optimizer: the resize
    rebinds Parameters.
    """
    from dataclasses import replace as _dc_replace

    from core.training.arch import get_arch_handler
    from core.training.ops.training_method import resolve_training_method

    swap_source = resolve_vae_swap_source(trainer.config)
    check_swap_method(swap_source, resolve_training_method(trainer))

    if declared is not None:
        # The loader already built the module from these weights; keeping the
        # resolver's copy would hold a second VAE in host memory for the run.
        declared = _dc_replace(declared, state_dict=None)
    trainer.base_vae_identity = declared
    if declared is not None:
        get_arch_handler(trainer).apply_vae_swap(trainer, declared,
                                                 module=trainer.vae)
    if swap_source:
        apply_configured_vae_swap(trainer, swap_source)


def preflight_vae_swap(
    config: Dict[str, Any],
    *,
    arch: Optional[str],
    method: str,
    bundle_vae_explicit_false: bool,
    base_model_path: Optional[str] = None,
) -> str:
    """Every refusal a VAE swap can earn before the model loads. Returns the
    resolved source (``""`` when the run does not swap).

    Raises ``ValueError``; callers map it to their own error surface.
    """
    from core.models.common.vae_source import (
        VAE_REGISTRY, VaeSourceError, describe_vae_source, parse_vae_source,
    )

    source = resolve_vae_swap_source(config)
    if not source:
        check_inherited_bundling(base_model_path, arch, bundle_vae_explicit_false)
        return ""
    check_swap_method(source, method)
    check_capability(source, arch, method)
    check_bundling(source, bundle_vae_explicit_false)

    try:
        form, value = parse_vae_source(source)
    except VaeSourceError as exc:
        raise ValueError(str(exc))

    if form == "registry":
        # Answered from the family table alone: a registry VAE that is not in the
        # shared store yet is downloaded by the run, and refusing it here would
        # refuse a legitimate first use.
        entry = VAE_REGISTRY.get(value)
        if entry is None:
            raise ValueError(
                f"unknown VAE registry key {value!r} "
                f"(known: {', '.join(sorted(VAE_REGISTRY))})")
        facts = {
            "latent_channels": entry.get("latent_channels"),
            "ndim": entry.get("latent_ndim"),
            "scale_factor": entry.get("scale_factor"),
            "scale_temporal": entry.get("scale_temporal"),
            "norm": entry.get("norm"),
        }
    else:
        described = describe_vae_source(source, arch=arch)
        if not described.get("compatible") and described.get("ndim") is None:
            # The source could not be resolved at all; that reason is final.
            raise ValueError(
                f"vae_swap_source={source!r} cannot drive {arch or 'this model'}: "
                f"{described.get('reason') or 'incompatible'}")
        facts = described
    compatible, reason = check_arch_vae_compatibility(
        facts, arch, base_model_path=base_model_path)
    if not compatible:
        raise ValueError(
            f"vae_swap_source={source!r} cannot drive {arch or 'this model'}: "
            f"{reason or 'incompatible'}")
    return source


def apply_configured_vae_swap(trainer, source: str) -> Optional[Any]:
    """Resolve ``source``, gate it, and hand it to the arch handler (§8.1).

    Returns the ``ResizeReport``, or None when the resolved VAE turns out to be
    the one the base already carries (§7.4's no-op rule: same channel count and
    same weights is not a swap).
    """
    from core.models.common.vae_source import resolve_vae_source
    from core.training.arch import get_arch_handler

    handler = getattr(trainer, "arch", None) or get_arch_handler(trainer)
    arch = handler.name
    native_hash = _module_hash(getattr(trainer, "vae", None))

    resolved = resolve_vae_source(source, arch=arch)
    compatible, reason = handler.check_vae_compatibility(
        resolved.facts(), trainer=trainer)
    if not compatible:
        raise ValueError(f"vae_swap_source={source!r} cannot drive {arch}: {reason}")

    log = getattr(trainer, "log_prefix", "[VAESwap]")
    print(f"{log} [VAE swap] {resolved.provenance}: {resolved.latent_channels}ch, "
          f"{resolved.scale_factor}x spatial, norm={resolved.norm}")

    report = handler.apply_vae_swap(trainer, resolved)
    from dataclasses import replace
    # A single-file loader infers the full config while converting LDM weights.
    resolved = replace(resolved, config=_materialized_config(trainer.vae, resolved.config))

    # The two hashes are only comparable module-to-module (same key layout, same
    # dtype), which is why the no-op test happens here and not in the resolver.
    identity_native = False
    if native_hash is not None:
        identity_native = _module_hash(trainer.vae) == native_hash
    if _base_is_already_swapped(trainer):
        # §8.3: the base's own declaration wins — a second swap on top of a
        # swapped base can never be back in the architecture's native space.
        identity_native = False
    if identity_native:
        print(f"{log} [VAE swap] resolved VAE is byte-identical to the base's own; "
              "treated as no swap")
    trainer.vae_identity = _with_identity(resolved, identity_native)
    if identity_native:
        return None
    print(f"{log} [VAE swap] latent I/O resized: {report.replaced} "
          f"-> {report.new_channels}ch ({report.copied_elements} elements copied, "
          f"{report.new_elements} zero-initialised)")
    return report


def _with_identity(resolved, identity_native: bool):
    from dataclasses import replace
    return replace(resolved, identity_native=identity_native,
                   struct_native=(True if identity_native else resolved.struct_native))


def _base_is_already_swapped(trainer) -> bool:
    declared = getattr(trainer, "base_vae_identity", None)
    return declared is not None and declared.identity_native is False


def _module_hash(module) -> Optional[str]:
    from core.models.common.vae_source import content_hash_for_state_dict, latent_space_hash
    if module is None:
        return None
    try:
        config = getattr(module, "config", {})
        if not hasattr(config, "get"):
            config = vars(config)
        return latent_space_hash(content_hash_for_state_dict(module.state_dict()), config)
    except Exception as e:
        print(f"[VAESwap] base VAE hash unavailable ({type(e).__name__}: {e}); "
              "treating the swap as a real one")
        return None


def _materialized_config(module, fallback):
    config = getattr(module, "config", None)
    if config is None:
        return dict(fallback or {})
    return dict(config) if hasattr(config, "keys") else dict(vars(config))


def validate_latent_io(trainer) -> list:
    """Everything wrong with this run's latent plumbing (design §8.6, items 1-2).

    Returns a list of human-readable problems; empty means the backbone's
    latent-facing layers and the VAE agree on channel count, rank and
    compression ratio. The caller decides whether a problem is fatal
    (``strict_validation``).

    Only a run whose latents do NOT come from the architecture's own VAE is
    checked. A native run's answer is already fixed by its loader, and probing
    every architecture's VAE here would let an unfamiliar (but correct) encode
    signature abort a run that has nothing to do with a swap.
    """
    from core.models.components.latent_io import verify_latent_io
    from core.training.arch import get_arch_handler

    identity = getattr(trainer, "vae_identity", None)
    if identity is None or identity.identity_native:
        return []
    handler = getattr(trainer, "arch", None) or get_arch_handler(trainer)
    wiring = getattr(trainer, "wiring", None) or handler.resolve_wiring(trainer)
    if wiring is None:
        return []
    problems = []
    spec = getattr(wiring, "latent_io", None)
    root = (trainer if handler.latent_io_root_attr is None
            else getattr(trainer, handler.latent_io_root_attr, None))
    if spec is not None and root is not None and wiring.latent_channels:
        problems.extend(verify_latent_io(root, spec, wiring.latent_channels))
    problems.extend(_verify_vae_geometry(trainer, wiring))
    return problems


def _verify_vae_geometry(trainer, wiring) -> list:
    """Encode one dummy image and check the latent it produces against the
    wiring: channel count, rank, spatial compression ratio."""
    import torch

    vae = getattr(trainer, "vae", None)
    if vae is None or not wiring.latent_channels:
        return []
    if wiring.latent_ndim != 4:
        # A 5-D VAE needs a clip, not an image; those archs arrive in wave 2.
        return []
    scale = int(wiring.vae_scale_factor or 8)
    size = scale * 8
    try:
        device = next(vae.parameters()).device
        dtype = next(vae.parameters()).dtype
        sample = torch.zeros(1, 3, size, size, device=device, dtype=dtype)
        with torch.no_grad():
            encoded = vae.encode(sample)
        latent = getattr(encoded, "latent_dist", None)
        latent = (latent.mean if latent is not None
                  else getattr(encoded, "latent", None))
        if latent is None:
            latent = encoded[0] if isinstance(encoded, (tuple, list)) else None
        if latent is None:
            return ["VAE encode returned no latent this check understands"]
    except Exception as exc:
        return [f"VAE encode of a {size}x{size} probe failed: "
                f"{type(exc).__name__}: {exc}"]

    problems = []
    if latent.ndim != wiring.latent_ndim:
        problems.append(
            f"VAE produces {latent.ndim}-D latents, wiring expects "
            f"{wiring.latent_ndim}-D")
    if latent.shape[1] != wiring.latent_channels:
        problems.append(
            f"VAE produces {latent.shape[1]} latent channels, wiring expects "
            f"{wiring.latent_channels}")
    if latent.shape[-1] and size // latent.shape[-1] != scale:
        problems.append(
            f"VAE compresses {size}px to {latent.shape[-1]}px "
            f"({size / max(1, latent.shape[-1]):g}x), wiring expects {scale}x")
    return problems


def swap_metadata(trainer) -> Tuple[Optional[Any], bool, Dict[str, str]]:
    """``(ResolvedVAE, bundled, component.vae.* metadata)`` for this run's save,
    or ``(None, False, {})`` when the run trains in the arch's own latent space."""
    from core.models.common.single_file_format import build_component_metadata

    resolved = getattr(trainer, "vae_identity", None)
    if resolved is None or resolved.identity_native:
        return None, False, {}
    from api.param_defaults import resolve_bundle_vae
    arch = getattr(getattr(trainer, "arch", None), "name", None) or ""
    bundled = resolve_bundle_vae(getattr(trainer, "bundle_vae", None), arch,
                                 vae_swapped=True)
    check_bundling(resolved.source, getattr(trainer, "bundle_vae", None) is False)
    # The scaling numbers are the ones a reader cannot observe from weights, so
    # they travel in the declared config or the checkpoint is unloadable (§7.3).
    config = _materialized_config(getattr(trainer, "vae", None), resolved.config)
    for key, value in (("scaling_factor", resolved.scaling_factor),
                       ("shift_factor", resolved.shift_factor)):
        if value is not None:
            config.setdefault(key, value)
    metadata = build_component_metadata(
        vae_type=resolved.family,
        vae_channels=resolved.latent_channels,
        vae_embedded=bundled,
        vae_prefix="vae." if bundled else None,
        vae_class=resolved.vae_class,
        vae_config=config or None,
        vae_scale_factor=resolved.scale_factor,
        vae_scale_temporal=resolved.scale_temporal,
        vae_norm=resolved.norm,
        vae_norm_pack=resolved.norm_pack,
        vae_provenance=resolved.provenance,
        vae_locator=None if bundled else resolved.locator,
        vae_hash=resolved.content_hash,
        vae_struct_native=bool(resolved.struct_native),
        vae_identity_native=False,
    )
    return resolved, bundled, metadata
