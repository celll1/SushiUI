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
        VAE_REGISTRY, VaeSourceError, check_vae_compatibility,
        describe_vae_source, parse_vae_source,
    )

    source = resolve_vae_swap_source(config)
    if not source:
        check_inherited_bundling(base_model_path, arch, bundle_vae_explicit_false)
        return ""
    check_swap_method(source, method)
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
        compatible, reason = check_vae_compatibility(facts, arch)
    else:
        described = describe_vae_source(source, arch=arch)
        compatible = bool(described.get("compatible"))
        reason = described.get("reason")
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
    from core.models.common.vae_source import (
        check_vae_compatibility, resolve_vae_source,
    )
    from core.training.arch import get_arch_handler

    handler = getattr(trainer, "arch", None) or get_arch_handler(trainer)
    arch = handler.name
    native_hash = _module_hash(getattr(trainer, "vae", None))

    resolved = resolve_vae_source(source, arch=arch)
    compatible, reason = check_vae_compatibility(resolved.facts(), arch)
    if not compatible:
        raise ValueError(f"vae_swap_source={source!r} cannot drive {arch}: {reason}")

    log = getattr(trainer, "log_prefix", "[VAESwap]")
    print(f"{log} [VAE swap] {resolved.provenance}: {resolved.latent_channels}ch, "
          f"{resolved.scale_factor}x spatial, norm={resolved.norm}")

    report = handler.apply_vae_swap(trainer, resolved)

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
    from core.models.common.vae_source import content_hash_for_state_dict
    if module is None:
        return None
    try:
        return content_hash_for_state_dict(module.state_dict())
    except Exception as e:
        print(f"[VAESwap] base VAE hash unavailable ({type(e).__name__}: {e}); "
              "treating the swap as a real one")
        return None


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
    wiring = getattr(trainer, "wiring", None) or handler.wiring
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
    config = dict(resolved.config or {})
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
