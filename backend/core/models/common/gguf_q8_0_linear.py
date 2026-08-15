"""Packed GGUF Q8_0 Linear -- design doc phase 12 ("Q8_0 residency").

Full narrative, the numbers (measured error distribution, host RAM / VRAM
table, both arms), and the rejected-then-fixed placement history live in
``docs/guides/MINIMAX_MUSIC3_DESIGN.md``, "Q8_0 residency" -- this docstring
states the invariants a reader of THIS code needs and does not repeat that
account.

WHAT THIS IS. A weight-only quantized ``nn.Linear`` whose weight stays in
GGML's own Q8_0 block layout (``gguf_container.get_q8_0_packed``: an
``(out, in)`` int8 codes tensor + an ``(out, in // 32)`` float16 per-block
scale) instead of being expanded to a dense tensor at load.

WHY DEQUANTIZE ONCE PER DEVICE, NOT ONCE PER FORWARD. Unlike this repo's other
weight-only quantized Linears (``ideogram4.vendor.int8_linear.Int8Linear``,
``common.convrot_int8_linear.ConvRotInt8Linear``), which dequantize on every
forward because their owning DiTs run only 20-60 denoising steps per
generation, this class's owner (the AR stage's language model) is called up
to ~9,000 times per generation (design doc, "Autoregressive stage"). A
per-forward dequant of an 8B-parameter stack was ESTIMATED FROM MEMORY
BANDWIDTH (not benchmarked; the design doc states this is an estimate, not a
measurement) at ~10-25 ms per call against a 40 ms/frame real-time budget --
rejected outright, not deprioritized.

THE PLACEMENT INVARIANT (the property a prior version of this class got
wrong -- see the design doc for that history): ``qweight``/``qscale`` are
PINNED host-resident for the module's whole life. ``_apply`` does not forward
a device-changing call to them; the first ``forward`` on a given
``(device, dtype)`` copies them there as TRANSIENT temporaries, dequantizes,
caches ONLY the dense result, and lets the temporaries free immediately.
``_materialized_weight`` also SELF-HEALS: if a caller reaches the packed
buffers through a path this class does not intercept, they are moved back to
CPU on the next use before a new mirror is built.

KNOWN GAP THIS SELF-HEAL DOES NOT CLOSE: it heals on the NEXT forward, not
instantly, so a mover that both relocates `qweight`/`qscale` AND holds them
resident on the wrong device between forwards still pays that residency for
one step. Two movers in this repo's OWN pinned dependencies bypass `_apply`
entirely and would do exactly that if ever applied to this architecture:
`diffusers/hooks/group_offloading.py` (`buffer.data = buffer.data.to(...)`
over `module.buffers()`) and `accelerate/hooks.py`'s `AlignDevicesHook`
(`set_module_tensor_to_device` on every buffer at hook-attach time, never
reversed). NEITHER IS APPLIED TO THIS ARCHITECTURE TODAY -- the shipped
staged offload is a plain `component.to(device)`, no hooks -- but this class
is therefore INCOMPATIBLE with `diffusers` group offloading and with an
`accelerate` `AlignDevicesHook` being applied to a module holding a
`GGUFQ8_0Linear`; do not wire either onto this architecture's language model
without re-examining this note.

CACHE INVALIDATION, all three triggers: a device/dtype-changing `_apply`
call, `load_state_dict` (via `_load_from_state_dict` -- `Tensor.copy_` writes
new bytes into `qweight`/`qscale` in place and never calls `_apply`, so
without this override a checkpoint reload or a swapped fine-tune would
silently keep generating from the PREVIOUS weights), and the self-heal path
above. The dense mirror is also excluded from `state_dict()` (never a
buffer) and from pickling / `copy.deepcopy` (`__getstate__`, since
`torch.save(module)` pickles the whole object).

Q8_0 is LOSSY (8-bit codes, one fp16 scale per 32 values) -- not claimed
bit-identical to bf16; the measured error distribution is in the design doc.
"""

from __future__ import annotations

from typing import Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models.common.gguf_container import Q8_0_BLOCK_SIZE

__all__ = [
    "Q8_0_BLOCK_SIZE",
    "dequantize_q8_0",
    "GGUFQ8_0Linear",
    "install_packed_q8_0_linears",
]


def dequantize_q8_0(codes: torch.Tensor, scale: torch.Tensor, compute_dtype: torch.dtype) -> torch.Tensor:
    """``(out, in)`` dense weight from Q8_0 ``codes``/``scale`` (see
    ``gguf_container.GGUFStateDict.get_q8_0_packed`` for their exact shapes
    and the block-layout proof).

    Both operands are widened to float32 for the multiply -- the same
    discipline ``int8_linear.py``'s activation quantizer uses and for the
    same reason: ``codes`` (int8, exact in float32) times a float16 scale
    computed in float16 would round the product to float16 precision before
    the caller's own compute-dtype cast gets a chance to, which is a
    different (and needlessly coarser) rounding than doing the multiply at
    float32 and rounding exactly once, at the end, to ``compute_dtype``.
    """
    if codes.dtype is not torch.int8:
        raise ValueError(f"dequantize_q8_0: codes must be torch.int8, got {codes.dtype}")
    if scale.dtype is not torch.float16:
        raise ValueError(f"dequantize_q8_0: scale must be torch.float16, got {scale.dtype}")
    out_features, in_features = codes.shape
    blocks_per_row = scale.shape[-1]
    if scale.shape != (out_features, blocks_per_row) or in_features != blocks_per_row * Q8_0_BLOCK_SIZE:
        raise ValueError(
            f"dequantize_q8_0: codes {tuple(codes.shape)} and scale {tuple(scale.shape)} are not "
            f"a consistent Q8_0 (out, in) / (out, in // {Q8_0_BLOCK_SIZE}) pair."
        )
    # IN-PLACE, BROADCAST form -- deliberately not `scale.repeat_interleave(...)
    # * codes`. The earlier version materialized an explicit (out, in) float32
    # `expanded_scale` tensor (the SAME size as the float32 codes tensor), so
    # the largest real layer (24576 x 4096) held TWO ~402 MB float32 buffers
    # at once (~0.805 GB) purely to multiply by a value that only takes
    # `blocks_per_row` distinct values per row. Reshaping into
    # ``(out, blocks_per_row, Q8_0_BLOCK_SIZE)`` and multiplying by
    # ``scale.unsqueeze(-1)`` (shape ``(out, blocks_per_row, 1)``) lets torch's
    # broadcasting read each scale value in place, without ever writing an
    # expanded copy of it -- halving that transient peak to one ~402 MB
    # buffer (the mutated codes tensor itself). `.view()` (not `.reshape()`)
    # on the freshly-allocated `.to(torch.float32)` result is always safe: a
    # fresh tensor from a widening cast is always contiguous.
    out_shaped = codes.to(torch.float32).view(out_features, blocks_per_row, Q8_0_BLOCK_SIZE)
    out_shaped.mul_(scale.to(torch.float32).unsqueeze(-1))
    weight = out_shaped.view(out_features, in_features)
    return weight.to(compute_dtype)


class GGUFQ8_0Linear(nn.Module):
    """A Linear layer whose weight is stored PACKED (Q8_0), dequantized ONCE
    PER DEVICE (not once per forward), with the packed source PINNED to the
    host -- see this module's docstring for the full reasoning, including the
    earlier placement this class explicitly does NOT use (packed buffers
    riding along to the GPU, which measured worse than plain bf16 loading).

    ``qweight``/``qscale`` are registered buffers (excluded from optimizer/
    grad machinery, loadable via ``state_dict``, matching ``Int8Linear``'s
    and ``ConvRotInt8Linear``'s convention in every respect EXCEPT ONE:
    ``_apply`` below deliberately does NOT forward a device/dtype-changing
    call to them, so ``module.to('cuda')`` (or ``.cuda()``, or being a
    submodule of a language model the pipeline backend stages to GPU) leaves
    them exactly where they were constructed -- the host, since
    ``pruned_text_encoder_q8_0_remap``/``gguf_container.get_q8_0_packed``
    only ever hand this class CPU tensors. ``bias`` (small, dense, unused on
    the real MiniMax Music 3 checkpoint) moves normally.

    The dense mirror is deliberately NOT a buffer: it must never appear in
    ``state_dict()`` (a caller must not be able to accidentally save an
    expanded copy of a checkpoint whose whole point is staying packed on
    disk), it must be rebuilt rather than silently kept stale across a device
    move, and -- the property this class exists to guarantee -- it must be
    the ONLY thing this module ever places on a compute device, never the
    packed source alongside it.
    """

    qweight: torch.Tensor
    qscale: torch.Tensor
    bias: Optional[torch.Tensor]

    def __init__(
        self,
        qweight: torch.Tensor,
        qscale: torch.Tensor,
        bias: Optional[torch.Tensor],
        compute_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if qweight.dtype is not torch.int8 or qweight.dim() != 2:
            raise ValueError(f"GGUFQ8_0Linear: qweight must be a 2-D int8 tensor, got {qweight.dtype} {tuple(qweight.shape)}")
        if qscale.dtype is not torch.float16 or qscale.dim() != 2:
            raise ValueError(f"GGUFQ8_0Linear: qscale must be a 2-D float16 tensor, got {qscale.dtype} {tuple(qscale.shape)}")
        out_features, in_features = qweight.shape
        if in_features % Q8_0_BLOCK_SIZE != 0:
            raise ValueError(
                f"GGUFQ8_0Linear: in_features={in_features} is not a multiple of Q8_0's block "
                f"size ({Q8_0_BLOCK_SIZE})."
            )
        if tuple(qscale.shape) != (out_features, in_features // Q8_0_BLOCK_SIZE):
            raise ValueError(
                f"GGUFQ8_0Linear: qscale {tuple(qscale.shape)} does not match qweight "
                f"{tuple(qweight.shape)} (expected ({out_features}, {in_features // Q8_0_BLOCK_SIZE}))."
            )
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype
        self.register_buffer("qweight", qweight)
        self.register_buffer("qscale", qscale)
        self.register_buffer("bias", bias)
        # NOT a buffer/parameter -- see the class docstring. `None` means
        # "no dense mirror has been materialized for the buffers' CURRENT
        # device"; `_apply` resets this to `None` on every device/dtype move.
        self._dequant_cache: Optional[torch.Tensor] = None

    def _apply(self, fn, recurse: bool = True):
        # Hit for EVERY `.to()`/`.cuda()`/`.cpu()`/`.half()`/`.bfloat16()`-
        # family call, including the staged offload the pipeline backend
        # already runs (LM + depth decoder to GPU for the AR stage, then back
        # to CPU). TWO deliberate departures from the default `nn.Module`
        # buffer walk (which would apply `fn` to every registered buffer,
        # `qweight`/`qscale` included):
        #
        #   1. `fn` is applied to `bias` only (small, dense -- there is
        #      nothing to protect it from). `qweight`/`qscale` are NEVER
        #      touched here, on purpose: this is the placement fix itself --
        #      a caller moving this module (or its owning language_model) to
        #      a GPU must not silently carry the packed source along, which
        #      is what regressed this class to worse-than-bf16 GPU residency
        #      before the fix (see the module docstring).
        #   2. The cache is still dropped unconditionally, exactly as before:
        #      a stale dense mirror built for the OLD device/dtype must never
        #      be handed back after a move, whether or not the packed source
        #      itself moved.
        #
        # `super()._apply()` is intentionally NOT called: this is a leaf
        # module (no children, no `nn.Parameter`), so the only bookkeeping
        # `nn.Module._apply` would otherwise do is exactly the buffer walk
        # this override replaces with the two rules above.
        #
        # LIMITATION, STATED RATHER THAN DISCOVERED LATER: this override only
        # intercepts callers that go through `nn.Module`'s own `.to()`/
        # `.cuda()`/`.cpu()`/`.half()` machinery. Two movers in this repo's
        # OWN pinned dependencies bypass it entirely -- `diffusers/hooks/
        # group_offloading.py` assigns `buffer.data = buffer.data.to(...)`
        # directly over `module.buffers()`, and `accelerate/hooks.py`'s
        # `AlignDevicesHook` calls `set_module_tensor_to_device` on every
        # buffer at hook-attach time and never brings it back. Neither is
        # applied to this architecture today (the shipped staged offload is a
        # plain `component.to(device)`, no hooks), but this class does not
        # rely on that staying true -- see `_materialized_weight`'s
        # self-healing check below, which is the ENFORCEABLE half of this
        # invariant; this docstring note is the other half, so a reader does
        # not have to discover the gap by reading `_apply` and stopping here.
        if self.bias is not None:
            self.bias = fn(self.bias)
        self._dequant_cache = None
        return self

    def _load_from_state_dict(self, *args, **kwargs):
        # `nn.Module.load_state_dict` copies new bytes into `qweight`/
        # `qscale` IN PLACE (`Tensor.copy_` under `torch.no_grad()`) and never
        # calls `_apply` -- so without this override, a packed-checkpoint
        # reload or a swapped fine-tune would leave `_dequant_cache` holding
        # the PREVIOUS weights' dense mirror, and `_materialized_weight`'s
        # cache check (device + dtype only, both unchanged by a state-dict
        # load) would keep returning it. That is a SILENT wrong-output bug --
        # nothing raises, the shapes and dtypes all still match -- caught only
        # by an explicit test (`minimax_music3_gguf_q8_0_linear_test.py`,
        # `test_load_state_dict_invalidates_the_cache`), not by any shape or
        # dtype assertion this class already had.
        super()._load_from_state_dict(*args, **kwargs)
        self._dequant_cache = None

    def _materialized_weight(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # SELF-HEALING placement check (see `_apply`'s docstring note above):
        # if something outside this class's own `.to()`/`.cuda()` interception
        # moved `qweight`/`qscale` off the host directly (mutating `.data` or
        # replacing the buffer), restore the pinning invariant HERE, on the
        # very next use, rather than silently computing a dense mirror next to
        # a stranded packed copy. Cheap (`.device.type` is O(1)) and runs on
        # every call, because a bypassing mover could act between any two
        # forwards, not only around a `.to()` this class would see.
        if self.qweight.device.type != "cpu" or self.qscale.device.type != "cpu":
            self.qweight = self.qweight.to("cpu")
            self.qscale = self.qscale.to("cpu")
            self._dequant_cache = None
        cached = self._dequant_cache
        if (
            cached is not None
            and cached.device == device
            and cached.dtype == dtype
        ):
            return cached
        # `qweight`/`qscale` are copied to `device` here as TRANSIENT
        # temporaries for this one dequantize call -- see `_apply` above and
        # the module docstring: this is the ONLY moment either packed buffer
        # is ever resident on a non-host device, and it is per-LAYER (this
        # one Linear's own weight, never the whole model's), not per-model.
        # `.to(device)` is a no-op (returns the same tensor, no copy) when
        # `device` already IS the host, which is every CPU-only forward.
        qweight = self.qweight if self.qweight.device == device else self.qweight.to(device)
        qscale = self.qscale if self.qscale.device == device else self.qscale.to(device)
        # Dequantize DIRECTLY to the dtype actually being used (`dtype`, from
        # the calling activation), not to `self.compute_dtype` followed by a
        # per-forward `.to(x.dtype)` cast of the cached tensor -- the earlier
        # version did the latter, which reran that cast on EVERY forward
        # whenever a caller's runtime dtype differed from the dtype the
        # loader built this module with (e.g. the LM run in fp16 against a
        # bf16-loaded checkpoint): exactly the per-forward cost this whole
        # design exists to avoid, reintroduced through a dtype mismatch
        # instead of a device one. Caching keyed on `(device, dtype)` makes
        # that cast a one-time cost too.
        weight = dequantize_q8_0(qweight, qscale, dtype)
        self._dequant_cache = weight
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._materialized_weight(x.device, x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, weight, bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, q8_0=packed(block={Q8_0_BLOCK_SIZE})"
        )

    def __getstate__(self):
        # Exclude the (potentially GPU-resident, multi-hundred-MB) dense
        # mirror from BOTH pickling (`torch.save(module)` pickles the WHOLE
        # object, not only `state_dict()`) and `copy.deepcopy` (which uses
        # the same pickle protocol via `__reduce_ex__` when no `__deepcopy__`
        # is defined). `state_dict()` was already clean -- the mirror was
        # never a buffer -- but `_dequant_cache` is a plain instance
        # attribute, so it lives in `self.__dict__` and neither mechanism
        # exempts it by default; without this override, an unrelated
        # `copy.deepcopy(language_model)` or a whole-module `torch.save`
        # would silently carry an expanded copy of a checkpoint whose whole
        # point is staying packed. No `__setstate__` override is needed: the
        # default pickle reconstruction for an object with `__getstate__`
        # returning a dict is `self.__dict__.update(state)`, which is exactly
        # right here.
        state = self.__dict__.copy()
        state["_dequant_cache"] = None
        return state


def install_packed_q8_0_linears(
    root: nn.Module,
    packed: Mapping[str, Tuple[torch.Tensor, torch.Tensor]],
    compute_dtype: torch.dtype,
) -> int:
    """Replace the ``nn.Linear`` at each ``dest_key``'s module path under
    ``root`` with a ``GGUFQ8_0Linear`` holding ``(codes, scale)``.

    ``root`` is expected to already have the target ``nn.Linear`` modules in
    place (typically meta-device placeholders from ``init_empty_weights()``
    + ``from_config``) at every key in ``packed`` -- the LOADER-shaped
    contract ``convrot_int8_linear.swap_linears_to_convrot_int8`` and
    ``int8_linear.swap_linears_to_int8`` also follow, except keyed by an
    explicit dict of destination paths (this caller already computed those
    from ``pruned_text_encoder_q8_0_remap``'s plan) rather than walking every
    child module and testing a state-dict key's presence.

    Takes ``(codes, scale)`` PLAIN TUPLES rather than
    ``pruned_text_encoder_q8_0_remap.PackedQ8_0Weight`` deliberately: this
    module lives under ``core/models/common/`` (design doc phase 11 keeps
    ``gguf_container.py`` here for the same reason -- it is not MiniMax
    Music 3-specific), so it must not import a dataclass defined in an
    architecture package. A caller holding ``PackedQ8_0Weight`` instances
    passes ``{k: (v.codes, v.scale) for k, v in ...}``.

    Raises ``TypeError`` if the module at a key is not an ``nn.Linear``,
    ``ValueError`` on a shape mismatch, and ``NotImplementedError`` if that
    ``nn.Linear`` owns a bias (no packed bias tensor is ever supplied here --
    MiniMax Music 3's Q8_0 checkpoint carries none; every one of its 169
    quantized tensors is a bias-free weight, verified against the real
    file's tensor census). Returns the count installed.
    """
    installed = 0
    for dest_key, (codes, scale) in packed.items():
        if not dest_key.endswith(".weight"):
            raise ValueError(f"install_packed_q8_0_linears: {dest_key!r} does not end in '.weight'")
        dotted = dest_key[: -len(".weight")]
        if "." in dotted:
            parent_path, attr = dotted.rsplit(".", 1)
            parent = root.get_submodule(parent_path)
        else:
            attr = dotted
            parent = root
        child = getattr(parent, attr)
        if not isinstance(child, nn.Linear):
            raise TypeError(
                f"install_packed_q8_0_linears: {dotted!r} under root is {type(child).__name__}, "
                f"not nn.Linear -- cannot install a packed Q8_0 replacement there."
            )
        if (child.out_features, child.in_features) != tuple(codes.shape):
            raise ValueError(
                f"install_packed_q8_0_linears: {dotted!r} is ({child.out_features}, "
                f"{child.in_features}), but the packed weight is {tuple(codes.shape)}."
            )
        if child.bias is not None:
            raise NotImplementedError(
                f"install_packed_q8_0_linears: {dotted!r} has a bias, but no packed bias tensor "
                f"was supplied for it."
            )
        setattr(parent, attr, GGUFQ8_0Linear(codes, scale, None, compute_dtype))
        installed += 1
    return installed
