"""Channel-partial resize of the two modules that face the latent.

Implements ``docs/guides/VAE_SWAP_MIGRATION_DESIGN.md`` §6: replace the input and
output layers declared by a ``LatentIOSpec`` with layers of a new latent channel
count, copying the overlapping channels and ZERO-initialising the rest. The
backbone body is untouched.

The two sides are computed separately, each with its OWN declared order: a packed
weight is 3-D in disguise, and slicing ``[:, :P*n]`` off a flat packed axis is
correct for "outer" and silently wrong for "inner" (it keeps every channel of the
first n pack positions instead of the first n channels). anima is the arch where
the two sides disagree, so one shared expression cannot be right for it.

Call this BEFORE the optimizer is built — it rebinds Parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn

from core.models.components.wiring import LatentIOSpec


@dataclass(frozen=True)
class ResizeReport:
    replaced: Tuple[str, ...]
    old_in_channels: Optional[int]
    old_out_channels: Optional[int]
    new_channels: int
    copied_elements: int
    new_elements: int

    @property
    def changed(self) -> bool:
        return bool(self.replaced)


def resize_latent_io(
    module_root: nn.Module,
    spec: LatentIOSpec,
    new_channels: int,
    *,
    new_channel_init: str = "zero",
) -> ResizeReport:
    """Resize ``spec``'s input/output modules under ``module_root`` to ``new_channels``.

    An empty module path skips that side (the two-call form the asymmetric
    ``resize_unet_in_out`` needs). New channels are zero on both sides: step 0
    then means "ignore the unknown input channels / predict 0 into the unknown
    output channels" rather than injecting an out-of-distribution signal.
    """
    if new_channel_init != "zero":
        raise ValueError(
            f"new_channel_init={new_channel_init!r} is not supported; v1 accepts 'zero' only "
            f"(design D3)."
        )
    new_channels = int(new_channels)
    if new_channels <= 0:
        raise ValueError(f"new_channels must be positive, got {new_channels}")

    replaced: List[str] = []
    copied = 0
    fresh = 0
    old_in: Optional[int] = None
    old_out: Optional[int] = None

    if spec.in_module:
        parent, attr, module = _resolve(module_root, spec.in_module)
        old_in = _in_channels(module, spec)
        if old_in != new_channels:
            new_module, c, f = _build_in(module, spec, old_in, new_channels)
            setattr(parent, attr, new_module)
            replaced.append(spec.in_module)
            copied += c
            fresh += f

    if spec.out_module:
        parent, attr, module = _resolve(module_root, spec.out_module)
        old_out = _out_channels(module, spec)
        if old_out != new_channels:
            new_module, c, f = _build_out(module, spec, old_out, new_channels)
            setattr(parent, attr, new_module)
            replaced.append(spec.out_module)
            copied += c
            fresh += f

    if replaced:
        kwargs = {}
        if spec.in_module:
            kwargs["in_channels"] = new_channels
        if spec.out_module:
            kwargs["out_channels"] = new_channels
        for target in _sync_targets(module_root, replaced):
            _sync_channels(target, kwargs)

    return ResizeReport(
        replaced=tuple(replaced),
        old_in_channels=old_in,
        old_out_channels=old_out,
        new_channels=new_channels,
        copied_elements=copied,
        new_elements=fresh,
    )


def verify_latent_io(module_root: nn.Module, spec: LatentIOSpec,
                     channels: int) -> List[str]:
    """What is wrong with the latent-facing modules' widths; empty when they
    match ``channels`` (design §8.6 item 1).

    Same algebra as the resize, so a packed layer is judged by the channel count
    its packed axis encodes, not by its raw feature count.
    """
    problems: List[str] = []
    for path, reader, side in ((spec.in_module, _in_channels, "input"),
                               (spec.out_module, _out_channels, "output")):
        if not path:
            continue
        try:
            _parent, _attr, module = _resolve(module_root, path)
            found = reader(module, spec)
        except (AttributeError, ValueError) as exc:
            problems.append(f"latent {side} '{path}': {exc}")
            continue
        if found != channels:
            problems.append(
                f"latent {side} '{path}' faces {found} channels, expected {channels}")
    return problems


# --- module path resolution -------------------------------------------------

def _resolve(root: Any, path: str) -> Tuple[Any, str, nn.Module]:
    """(parent, attribute name, module) for a dotted path relative to ``root``.

    ``nn.Module.__getattr__`` serves ``_modules`` entries, so this reaches
    ``ModuleDict`` keys ("2-1") and ``Sequential`` indices ("1") too.
    """
    parts = path.split(".")
    obj = root
    try:
        for part in parts[:-1]:
            obj = getattr(obj, part)
        return obj, parts[-1], getattr(obj, parts[-1])
    except AttributeError as exc:
        raise AttributeError(
            f"latent I/O module '{path}' not found under {type(root).__name__}"
        ) from exc


def _sync_targets(root: Any, replaced: List[str]) -> List[Any]:
    """``root`` plus each replaced module's parent, deduped by identity.

    The parent matters because some archs keep the channel count next to the
    layer (minit2i's ``FinalLayer.out_channels``) and some at the root
    (zimage's ``out_channels``, read by ``unpatchify``).
    """
    targets: List[Any] = [root]
    for path in replaced:
        parent, _attr, _mod = _resolve(root, path)
        if not any(parent is t for t in targets):
            targets.append(parent)
    return targets


def _sync_channels(target: Any, kwargs: dict) -> None:
    if hasattr(target, "register_to_config"):
        target.register_to_config(**kwargs)
        return
    for key, value in kwargs.items():
        if hasattr(target, key):
            try:
                setattr(target, key, value)
            except Exception:
                pass
    # "cfg" as well as "config": minit2i's MMJiT keeps its channel count in
    # `self.cfg.in_channels`, which `unpatchify` reads.
    for config_attr in ("config", "cfg"):
        config = getattr(target, config_attr, None)
        if config is None:
            continue
        for key, value in kwargs.items():
            if hasattr(config, key):
                try:
                    setattr(config, key, value)
                except Exception:
                    pass


# --- channel counts ---------------------------------------------------------

def _in_channels(module: nn.Module, spec: LatentIOSpec) -> int:
    if spec.in_kind == "conv":
        total = int(module.in_channels)
        per_block, rem = divmod(total, max(1, spec.in_repeat))
        if rem:
            raise ValueError(
                f"input conv has {total} channels, not divisible by in_repeat={spec.in_repeat}"
            )
        return per_block - spec.extra_in_channels
    if spec.in_kind == "packed_linear":
        total = int(module.in_features)
        per_pos, rem = divmod(total, max(1, spec.pack_elems) * max(1, spec.in_repeat))
        if rem:
            raise ValueError(
                f"input linear has {total} features, not divisible by "
                f"pack_elems={spec.pack_elems} * in_repeat={spec.in_repeat}"
            )
        return per_pos - spec.extra_in_channels
    raise ValueError(f"unknown in_kind {spec.in_kind!r}")


def _out_channels(module: nn.Module, spec: LatentIOSpec) -> int:
    if spec.out_kind == "conv":
        return int(module.out_channels)
    if spec.out_kind == "packed_linear":
        total = int(module.out_features)
        channels, rem = divmod(total, max(1, spec.pack_elems))
        if rem:
            raise ValueError(
                f"output linear has {total} features, not divisible by pack_elems={spec.pack_elems}"
            )
        return channels
    raise ValueError(f"unknown out_kind {spec.out_kind!r}")


# --- input side -------------------------------------------------------------

def _build_in(module: nn.Module, spec: LatentIOSpec, old_c: int, new_c: int):
    if spec.in_kind == "conv":
        return _build_in_conv(module, spec, old_c, new_c)
    return _build_in_packed(module, spec, old_c, new_c)


def _build_in_conv(conv: nn.Module, spec: LatentIOSpec, old_c: int, new_c: int):
    _require_plain_conv(conv, "input")
    w = conv.weight.detach()
    hidden = w.shape[0]
    k = tuple(w.shape[2:])
    r, e = max(1, spec.in_repeat), spec.extra_in_channels
    n = min(old_c, new_c)

    old_view = w.reshape(hidden, r, old_c + e, *k)
    new_w = torch.zeros(hidden, r, new_c + e, *k, device=w.device, dtype=w.dtype)
    new_w[:, :, :n] = old_view[:, :, :n]
    if e:
        new_w[:, :, new_c:new_c + e] = old_view[:, :, old_c:old_c + e]
    new_w = new_w.reshape(hidden, r * (new_c + e), *k)

    new_conv = type(conv)(
        r * (new_c + e), conv.out_channels,
        kernel_size=conv.kernel_size, stride=conv.stride,
        padding=conv.padding, dilation=conv.dilation,
        groups=conv.groups, bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    ).to(device=w.device, dtype=w.dtype)
    with torch.no_grad():
        new_conv.weight.copy_(new_w)
        if conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(conv.bias)  # hidden-dim: fully preserved
    _match_grad_flags(conv, new_conv)

    copied = hidden * r * (n + e) * _prod(k)
    return new_conv, copied, new_conv.weight.numel() - copied


def _build_in_packed(linear: nn.Module, spec: LatentIOSpec, old_c: int, new_c: int):
    _require_plain_linear(linear, "input")
    order = _require_order(spec.in_channel_order, "in_channel_order")
    w = linear.weight.detach()
    hidden = w.shape[0]
    P, e, r = max(1, spec.pack_elems), spec.extra_in_channels, max(1, spec.in_repeat)
    n = min(old_c, new_c)

    if order == "outer":                                   # k = c*P + s
        old_view = w.reshape(hidden, r, old_c + e, P)
        new_w = torch.zeros(hidden, r, new_c + e, P, device=w.device, dtype=w.dtype)
        new_w[:, :, :n, :] = old_view[:, :, :n, :]
        if e:
            new_w[:, :, new_c:new_c + e, :] = old_view[:, :, old_c:old_c + e, :]
    else:                                                  # k = s*(C+e) + c
        old_view = w.reshape(hidden, r, P, old_c + e)
        new_w = torch.zeros(hidden, r, P, new_c + e, device=w.device, dtype=w.dtype)
        new_w[:, :, :, :n] = old_view[:, :, :, :n]
        if e:
            new_w[:, :, :, new_c:new_c + e] = old_view[:, :, :, old_c:old_c + e]
    new_w = new_w.reshape(hidden, r * P * (new_c + e))

    new_linear = nn.Linear(
        r * P * (new_c + e), linear.out_features, bias=linear.bias is not None,
    ).to(device=w.device, dtype=w.dtype)
    with torch.no_grad():
        new_linear.weight.copy_(new_w)
        if linear.bias is not None and new_linear.bias is not None:
            new_linear.bias.copy_(linear.bias)  # hidden-dim: fully preserved
    _match_grad_flags(linear, new_linear)

    copied = hidden * r * P * (n + e)
    return new_linear, copied, new_linear.weight.numel() - copied


# --- output side ------------------------------------------------------------

def _build_out(module: nn.Module, spec: LatentIOSpec, old_c: int, new_c: int):
    if spec.out_kind == "conv":
        return _build_out_conv(module, spec, old_c, new_c)
    return _build_out_packed(module, spec, old_c, new_c)


def _build_out_conv(conv: nn.Module, spec: LatentIOSpec, old_c: int, new_c: int):
    _require_plain_conv(conv, "output")
    w = conv.weight.detach()
    hidden = w.shape[1]
    k = tuple(w.shape[2:])
    n = min(old_c, new_c)

    new_w = torch.zeros(new_c, hidden, *k, device=w.device, dtype=w.dtype)
    new_w[:n] = w[:n]

    new_conv = type(conv)(
        conv.in_channels, new_c,
        kernel_size=conv.kernel_size, stride=conv.stride,
        padding=conv.padding, dilation=conv.dilation,
        groups=conv.groups, bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    ).to(device=w.device, dtype=w.dtype)
    with torch.no_grad():
        new_conv.weight.copy_(new_w)
        if conv.bias is not None and new_conv.bias is not None:
            new_bias = torch.zeros(new_c, device=w.device, dtype=conv.bias.dtype)
            new_bias[:n] = conv.bias.detach()[:n]
            new_conv.bias.copy_(new_bias)
    _match_grad_flags(conv, new_conv)

    copied = n * hidden * _prod(k)
    return new_conv, copied, new_conv.weight.numel() - copied


def _build_out_packed(linear: nn.Module, spec: LatentIOSpec, old_c: int, new_c: int):
    _require_plain_linear(linear, "output")
    order = _require_order(spec.out_channel_order, "out_channel_order")
    w = linear.weight.detach()
    hidden = w.shape[1]
    P = max(1, spec.pack_elems)
    n = min(old_c, new_c)
    bias = linear.bias.detach() if linear.bias is not None else None

    if order == "outer":                                   # k = c*P + s
        old_view = w.reshape(old_c, P, hidden)
        new_w = torch.zeros(new_c, P, hidden, device=w.device, dtype=w.dtype)
        new_w[:n] = old_view[:n]
        if bias is not None:
            old_b = bias.reshape(old_c, P)
            new_b = torch.zeros(new_c, P, device=bias.device, dtype=bias.dtype)
            new_b[:n] = old_b[:n]
    else:                                                  # k = s*C + c
        old_view = w.reshape(P, old_c, hidden)
        new_w = torch.zeros(P, new_c, hidden, device=w.device, dtype=w.dtype)
        new_w[:, :n] = old_view[:, :n]
        if bias is not None:
            old_b = bias.reshape(P, old_c)
            new_b = torch.zeros(P, new_c, device=bias.device, dtype=bias.dtype)
            new_b[:, :n] = old_b[:, :n]
    new_w = new_w.reshape(P * new_c, hidden)

    new_linear = nn.Linear(
        linear.in_features, P * new_c, bias=bias is not None,
    ).to(device=w.device, dtype=w.dtype)
    with torch.no_grad():
        new_linear.weight.copy_(new_w)
        if bias is not None and new_linear.bias is not None:
            new_linear.bias.copy_(new_b.reshape(P * new_c))
    _match_grad_flags(linear, new_linear)

    copied = n * P * hidden
    return new_linear, copied, new_linear.weight.numel() - copied


# --- guards -----------------------------------------------------------------

def _prod(shape) -> int:
    total = 1
    for s in shape:
        total *= int(s)
    return total


def _require_order(order: str, field: str) -> str:
    if order not in ("outer", "inner"):
        raise ValueError(
            f"{field}={order!r} is not declared for a packed_linear side; "
            f"expected 'outer' or 'inner'"
        )
    return order


def _require_plain_linear(module: nn.Module, side: str) -> None:
    if not isinstance(module, nn.Linear):
        raise TypeError(
            f"{side} module {type(module).__name__} is not an nn.Linear; resize the latent "
            f"I/O before quantization/adapter wrapping"
        )


def _require_plain_conv(module: nn.Module, side: str) -> None:
    # ConvTranspose stores weights as [in, out, k]; slicing it on dim 0 would move
    # the wrong axis (acestep's proj_out, design §6.4 — deferred, not wired here).
    if isinstance(module, nn.modules.conv._ConvTransposeNd) or not isinstance(
        module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)
    ):
        raise TypeError(
            f"{side} module {type(module).__name__} is not a plain Conv1d/2d/3d"
        )


def _match_grad_flags(old: nn.Module, new: nn.Module) -> None:
    new.weight.requires_grad_(old.weight.requires_grad)
    if old.bias is not None and new.bias is not None:
        new.bias.requires_grad_(old.bias.requires_grad)
