"""Adapter leaf layers: the wrappers that carry the trainable branch.

The stock LoRA layer relies on ambient ``torch.autocast`` to reconcile its fp32
masters with a bf16 activation; the MiniMax-H3 subclass casts per call because
that architecture's forward runs without autocast. All are usable as branches
of ``CompositeAdapterLayer``.

LoHa/LoKr/DoRA LOAD conventions -- tensor set, scale, magnitude axis,
factorization -- follow LyCORIS 4.0.0 at 03270a38. The initialisations
deliberately do not (upstream zeroes ``lokr_w2_b`` where this zeroes
``lokr_w2_a``): same no-op start, different training dynamics.
"""

import math
from typing import (Dict, FrozenSet, Iterable, Iterator, List, Mapping,
                    Optional, Set, Tuple, Union)

import torch
import torch.nn as nn
import torch.nn.functional as F

from .execution.dispatch import adapter_forward_delta


#: Tucker-decomposed factors. They exist only for a target with kernel dims
#: (upstream ``weight_gen``: ``if k and tucker``), so a Linear branch cannot
#: honour them; their presence also transposes ``hada_w1_a``. Refused, never
#: dropped -- dropping them changes the algebra silently.
TUCKER_TENSOR_NAMES: FrozenSet[str] = frozenset({"hada_t1", "hada_t2", "lokr_t2"})


def refuse_tucker_tensors(names: Iterable[str], where: str = "") -> None:
    present = sorted(TUCKER_TENSOR_NAMES.intersection(names))
    if present:
        raise ValueError(
            f"{where or 'adapter branch'}: Tucker tensors {present} require a "
            f"target with kernel dimensions; a Linear branch cannot honour them")


class _BranchTensorProtocol:
    """save / resume / optimise, all derived from ``branch_tensors()``, plus the
    execution seam every algebra shares.

    ``branch_tensors`` is the single extension point: an algebra with a
    different tensor set overrides it and inherits the rest.

    ``forward_delta`` is the branch protocol the composite executes, so it is
    where an execution backend belongs; each algebra implements
    ``reference_delta`` and inherits the dispatch, and with nothing selected the
    conduit calls ``reference_delta`` directly.
    """

    #: The branch's two-axis algebra identity, for the execution registry and
    #: the probe. ``DoRALinearLayer`` reads its inner branch's.
    ADAPTER_ALGORITHM: str = ""
    WEIGHT_DECOMPOSE: bool = False

    def branch_tensors(self) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def adapter_strength(self) -> float:
        """The request strength currently in force on this branch.

        Every algebra records it in ``set_adapter_strength``, including the LoRA
        layer, which also folds it into ``scale``. The probe needs it to ask the
        fp32 oracle for the same function the branch is computing.
        """
        return float(getattr(self, "strength", 1.0))

    def reference_delta(self, x: torch.Tensor) -> torch.Tensor:
        """The unfused PyTorch contribution of this branch alone."""
        raise NotImplementedError

    def forward_delta(self, x: torch.Tensor) -> torch.Tensor:
        return adapter_forward_delta(self, x)

    def tensor_names(self) -> Tuple[str, ...]:
        """The names ``export_tensors`` produces and ``load_tensors`` consumes."""
        return tuple(self.branch_tensors())

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """The branch's own parameters. The wrapped base is frozen and excluded."""
        seen = set()
        for tensor in self.branch_tensors().values():
            if isinstance(tensor, nn.Parameter) and id(tensor) not in seen:
                seen.add(id(tensor))
                yield tensor

    def export_tensors(self) -> Dict[str, torch.Tensor]:
        """Detached CPU copies of the LIVE tensor set.

        Not yet a LyCORIS-compatible file: a `scalar` still needs folding into
        `w1`/`hada_w1_a` and its key dropping. See ``LoHaLinearLayer.branch_tensors``.
        """
        return {name: weight.detach().cpu()
                for name, weight in self.branch_tensors().items()}

    def spec_constants(self) -> Tuple[str, ...]:
        """Names in ``branch_tensors()`` that are a freshly built VALUE, so
        ``load_tensors`` must ASSIGN them: LoHa/LoKr rebuild ``alpha`` per call,
        and copying into that throwaway dropped the file's scale silently
        (measured 0.5x; design doc, phase 2)."""
        return ()

    def load_spec_constant(self, name: str, value: torch.Tensor) -> None:
        raise KeyError(f"{type(self).__name__} has no spec constant {name!r}")

    def load_tensors(self, tensors: Mapping[str, torch.Tensor]) -> None:
        """Copy a stem-relative slice back in place. Names absent from the
        slice are left alone, which is the tolerance training resume has always
        had for a checkpoint that carries only some of a branch's tensors."""
        refuse_tucker_tensors(tensors, type(self).__name__)
        constants = self.spec_constants()
        for name, weight in self.branch_tensors().items():
            value = tensors.get(name)
            if value is None:
                continue
            if name in constants:
                self.load_spec_constant(name, value)
            else:
                weight.data.copy_(value)

    def adopt_tensors(self, tensors: Mapping[str, torch.Tensor]) -> None:
        """``load_tensors`` for a branch built FROM a file: ASSIGN the file's
        tensor as the parameter's storage rather than copy into a fresh one.

        Not interchangeable. A fresh buffer is 64-byte aligned where a
        safetensors tensor is not, which selects a different BLAS kernel --
        1 ULP on a measured LoRA delta. Resume keeps ``load_tensors``: there an
        optimizer already holds the parameter. Design doc, phase 2.
        """
        refuse_tucker_tensors(tensors, type(self).__name__)
        constants = self.spec_constants()
        for name, weight in self.branch_tensors().items():
            value = tensors.get(name)
            if value is None:
                continue
            if name in constants:
                self.load_spec_constant(name, value)
            else:
                weight.data = value.to(device=weight.device, dtype=weight.dtype)


class _AlphaIsASpecConstant(_BranchTensorProtocol):
    """For the algebras whose ``branch_tensors()`` carries ``alpha``."""

    def spec_constants(self) -> Tuple[str, ...]:
        return ("alpha",)

    def load_spec_constant(self, name: str, value: torch.Tensor) -> None:
        if name != "alpha":
            raise KeyError(f"{type(self).__name__} has no spec constant {name!r}")
        self.set_alpha(float(value.item()) if torch.is_tensor(value) else float(value))

    def set_alpha(self, alpha: float) -> None:
        self.alpha = float(alpha)


def _base_geometry(base: nn.Module) -> Tuple[int, int]:
    out_features = getattr(base, "out_features", None)
    in_features = getattr(base, "in_features", None)
    if out_features is None or in_features is None:
        raise ValueError(f"{type(base).__name__} exposes no in_features/out_features")
    return int(out_features), int(in_features)


def _require_shape(name: str, tensor: torch.Tensor,
                   expected: Tuple[int, ...]) -> torch.Tensor:
    if tuple(tensor.shape) != expected:
        raise ValueError(f"{name} is {tuple(tensor.shape)}, not {expected}")
    return tensor


def _alpha_from_tensors(tensors: Mapping[str, torch.Tensor],
                        rank: int, alpha: Optional[float]) -> float:
    """Alpha precedence, the order Z-Image's codec set: per-key tensor, then
    the caller's (file metadata), then rank."""
    stored = tensors.get("alpha")
    if stored is not None:
        return float(stored.item()) if torch.is_tensor(stored) else float(stored)
    if alpha is not None:
        return float(alpha)
    return float(rank) if rank else 1.0


def fold_scalar_for_export(layer, into: str) -> Dict[str, torch.Tensor]:
    """``export_tensors`` for an algebra with a trained ``scalar``.

    Upstream multiplies ``scalar`` into the saved first factor and writes no
    ``scalar`` key; its reader then forces ``scalar := 1``. A serializer that
    emitted the key bare would leave every other reader ``1/scalar`` too strong.
    """
    tensors = {name: weight.detach().cpu()
               for name, weight in layer.branch_tensors().items()}
    scalar = tensors.pop("scalar", None)
    if scalar is not None:
        tensors[into] = tensors[into] * scalar
    return tensors


def _refuse_use_scalar(cls_name: str) -> None:
    """No file carries ``scalar`` (see ``LoHaLinearLayer``), and a layer built
    from one would multiply the whole delta by zero."""
    raise ValueError(
        f"{cls_name}.from_tensors: use_scalar layers cannot be built from a "
        f"checkpoint -- scalar starts at zero and no file carries the key")


class LoRALinearLayer(_BranchTensorProtocol, nn.Module):
    """
    LoRA layer for Linear modules.

    Formula: output = original_output + (lora_up(lora_down(x))) * scale
    """

    ADAPTER_ALGORITHM = "lora"

    def __init__(
        self,
        original_module: nn.Linear,
        rank: int,
        alpha: float,
        lora_name: str,
        lora_dtype: torch.dtype = torch.float32,
    ):
        """Initialize LoRA layer."""
        super().__init__()
        self.original_module = original_module
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.lora_name = lora_name
        self.lora_dtype = lora_dtype
        # Folded into ``scale`` rather than applied separately; kept because
        # ``adapter_strength()`` cannot recover it from a scale alone.
        self.strength = 1.0

        in_features = original_module.in_features
        out_features = original_module.out_features

        # LoRA matrices (no bias)
        # Use lora_dtype for LoRA weights (can be different from main model dtype)
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)

        # Initialize: Kaiming uniform for down, zeros for up
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        # Move to same device as original, but use lora_dtype
        device = original_module.weight.device
        self.lora_down.to(device=device, dtype=lora_dtype)
        self.lora_up.to(device=device, dtype=lora_dtype)

    @classmethod
    def from_tensors(cls, base: nn.Module,
                     tensors: Mapping[str, torch.Tensor], *,
                     alpha: Optional[float] = None,
                     lora_dtype: torch.dtype = torch.float32,
                     lora_name: str = "") -> "LoRALinearLayer":
        """Geometry from the TENSORS, then load them.

        Unlike LoHa/LoKr this cannot allocate a shape the base disagrees with,
        so it keeps the constructor's initialisation (which the eleven
        generation loaders also pay) rather than skipping it.
        """
        if tensors.get("lora_mid.weight") is not None:
            raise ValueError("lora_mid.weight is a LoCon convolution factor; a "
                             "Linear branch cannot honour it")
        down = tensors["lora_down.weight"]
        up = tensors["lora_up.weight"]
        out_features, in_features = _base_geometry(base)
        rank = int(down.shape[0])
        _require_shape("lora_down.weight", down, (rank, in_features))
        _require_shape("lora_up.weight", up, (out_features, rank))
        layer = cls(base, rank=rank,
                    alpha=_alpha_from_tensors(tensors, rank, alpha),
                    lora_name=lora_name, lora_dtype=lora_dtype)
        layer.adopt_tensors(tensors)
        return layer

    @property
    def weight(self):
        """Expose the wrapped Linear's weight so callers that introspect
        `.weight` (e.g. T5's DenseGatedActDense dtype check) keep working when a
        Linear is wrapped. Read-only delegate; not a trained parameter here."""
        return self.original_module.weight

    @property
    def bias(self):
        return getattr(self.original_module, "bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Base plus branch, via ``forward_delta`` so a bare branch (training) and a
        composite branch (generation) share one delta site.
        """
        return self.original_module(x) + self.forward_delta(x)

    def reference_delta(self, x: torch.Tensor) -> torch.Tensor:
        """The branch contribution alone, so this layer can be a composite branch.

        Must stay bit-identical to the second term of ``forward``;
        ``adapter_composite_layer_cheap_test`` asserts that with ``atol=0.0``.
        """
        return self.lora_up(self.lora_down(x)) * self.scale

    def compute_delta_weight(self) -> torch.Tensor:
        """The merged delta weight, for folding into a base or for a DoRA
        epilogue. ``set_adapter_strength`` folds the strength into ``scale``, so
        it is already included here."""
        return (self.lora_up.weight @ self.lora_down.weight) * self.scale

    def set_adapter_strength(self, strength: float) -> None:
        """Refold a request strength into the scale, exactly as the generation
        loaders do (``(alpha / rank) * strength``), so restrengthening an
        installed branch is not a rebuild."""
        self.strength = float(strength)
        self.scale = self.alpha / self.rank * strength

    def branch_tensors(self) -> Dict[str, torch.Tensor]:
        """Stem-relative name -> the LIVE weight, in checkpoint order.

        ``alpha`` is deliberately absent: it is a spec constant the saving
        adapter owns (Z-Image writes none at all), not a tensor this branch
        trains.
        """
        return {
            "lora_down.weight": self.lora_down.weight,
            "lora_up.weight": self.lora_up.weight,
        }


class MiniMaxH3LoRALinearLayer(LoRALinearLayer):
    """``LoRALinearLayer`` with the branch cast to the activation dtype.

    MiniMax-H3's forward runs without ``torch.autocast`` -- the vendored
    transformer owns its own policy (fp32 I/O heads and AdaLN, bf16 blocks) and
    an autocast context would override it, making training a different function
    from generation.

    The stock layer needs autocast to reconcile fp32 masters with a bf16
    activation. Without it that is a ``RuntimeError`` on the first
    ``F.linear``; building the branch in the activation dtype instead would
    lose the fp32 master silently. So masters stay fp32 and are cast per call.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original_module(x) + self.forward_delta(x)

    def reference_delta(self, x: torch.Tensor) -> torch.Tensor:
        down = F.linear(x, self.lora_down.weight.to(x.dtype))
        up = F.linear(down, self.lora_up.weight.to(x.dtype))
        return up * self.scale


class LoHaLinearLayer(_AlphaIsASpecConstant, nn.Module):
    """LoHa (Low-rank Hadamard Product) layer for Linear modules.

    ``use_scalar`` is upstream's trained ``scalar``: a parameter starting at
    0.0, folded in BEFORE the request strength, which REPLACES the
    zero-initialised ``hada_w2_a`` as the "starts as a no-op" mechanism.

    It is a TRAINING-side tensor only. Upstream's ``custom_state_dict`` folds it
    into the saved ``hada_w1_a`` and emits no ``scalar`` key, and
    ``load_weight_hook`` forces ``scalar := 1`` after any load -- so no file
    carries one, and reading a file without it (``scalar is None``) is right.
    See ``branch_tensors`` for what that costs an exporter.
    """

    ADAPTER_ALGORITHM = "loha"

    def __init__(
        self,
        original_module: nn.Linear,
        rank: int,
        alpha: float,
        lora_name: str,
        lora_dtype: torch.dtype = torch.float32,
        use_scalar: bool = False,
        init: bool = True,
    ):
        super().__init__()
        self.original_module = original_module
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank if rank > 0 else alpha
        self.lora_name = lora_name
        self.lora_dtype = lora_dtype
        self.strength = 1.0

        in_features = original_module.in_features
        out_features = original_module.out_features
        device = original_module.weight.device

        self.hada_w1_a = nn.Parameter(torch.empty((out_features, rank), device=device, dtype=lora_dtype))
        self.hada_w1_b = nn.Parameter(torch.empty((rank, in_features), device=device, dtype=lora_dtype))
        self.hada_w2_a = nn.Parameter(torch.empty((out_features, rank), device=device, dtype=lora_dtype))
        self.hada_w2_b = nn.Parameter(torch.empty((rank, in_features), device=device, dtype=lora_dtype))

        self.scalar = (nn.Parameter(torch.zeros((), device=device, dtype=lora_dtype))
                       if use_scalar else None)
        # ``init=False`` leaves every factor uninitialised for a caller that is
        # about to overwrite all of them; kaiming draws from the GLOBAL rng.
        if init:
            nn.init.kaiming_uniform_(self.hada_w1_a, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.hada_w1_b, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.hada_w2_b, a=math.sqrt(5))
            if use_scalar:
                nn.init.kaiming_uniform_(self.hada_w2_a, a=math.sqrt(5))
            else:
                nn.init.zeros_(self.hada_w2_a)

    @classmethod
    def from_tensors(cls, base: nn.Module,
                     tensors: Mapping[str, torch.Tensor], *,
                     alpha: Optional[float] = None,
                     lora_dtype: torch.dtype = torch.float32,
                     lora_name: str = "",
                     use_scalar: bool = False) -> "LoHaLinearLayer":
        """Geometry from the TENSORS -- the rank is ``hada_w1_a.shape[1]``, not
        a constructor argument -- then load them."""
        if use_scalar:
            _refuse_use_scalar(cls.__name__)
        out_features, in_features = _base_geometry(base)
        rank = int(tensors["hada_w1_a"].shape[1])
        for name in ("hada_w1_a", "hada_w2_a"):
            _require_shape(name, tensors[name], (out_features, rank))
        for name in ("hada_w1_b", "hada_w2_b"):
            _require_shape(name, tensors[name], (rank, in_features))
        layer = cls(base, rank=rank,
                    alpha=_alpha_from_tensors(tensors, rank, alpha),
                    lora_name=lora_name, lora_dtype=lora_dtype, init=False)
        layer.adopt_tensors(tensors)
        return layer

    @property
    def weight(self):
        return self.original_module.weight

    @property
    def bias(self):
        return getattr(self.original_module, "bias", None)

    def set_alpha(self, alpha: float) -> None:
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank if self.rank > 0 else self.alpha

    def set_adapter_strength(self, strength: float) -> None:
        self.strength = float(strength)

    def compute_delta_weight(self) -> torch.Tensor:
        w1 = self.hada_w1_a @ self.hada_w1_b
        w2 = self.hada_w2_a @ self.hada_w2_b
        delta = (w1 * w2) * self.scale
        if self.scalar is not None:
            delta = delta * self.scalar
        return delta * self.strength

    def reference_delta(self, x: torch.Tensor) -> torch.Tensor:
        delta_w = self.compute_delta_weight().to(x.dtype)
        return F.linear(x, delta_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original_module(x) + self.forward_delta(x)

    def branch_tensors(self) -> Dict[str, torch.Tensor]:
        """LIVE tensors, for the optimizer and for resume.

        NOT a serializer contract: ``scalar`` must be FOLDED into
        ``hada_w1_a`` and its key dropped on export, because upstream's reader
        forces ``scalar := 1``. Emitting it bare leaves the adapter
        ``1/scalar`` too strong for every other reader.
        """
        tensors: Dict[str, torch.Tensor] = {
            "hada_w1_a": self.hada_w1_a,
            "hada_w1_b": self.hada_w1_b,
            "hada_w2_a": self.hada_w2_a,
            "hada_w2_b": self.hada_w2_b,
        }
        if self.scalar is not None:
            tensors["scalar"] = self.scalar
        tensors["alpha"] = torch.tensor(self.alpha, dtype=torch.float32)
        return tensors

    def export_tensors(self) -> Dict[str, torch.Tensor]:
        return fold_scalar_for_export(self, "hada_w1_a")


def factorization(dimension: int, factor: int = -1) -> Tuple[int, int]:
    """Upstream LyCORIS ``factorization``: the most balanced divisor pair
    ``(m, n)`` with ``m <= n``, search capped at ``factor``.

    ``factor=-1`` agrees with the balanced ``isqrt`` search it replaced on every
    dimension 2..20000, so adopting it moves no existing LoKr. Upstream's own
    docstring table is stale: the code gives 360 -> (18, 20).

    A loader must not call this. ``factor`` is not stored in a checkpoint, so a
    file's ``(m1, n1)`` comes from the ``lokr_w1`` shape -- which also absorbs
    upstream's ``unbalanced_factorization``, recorded by no tensor.
    """
    if factor > 0 and (dimension % factor) == 0:
        m, n = factor, dimension // factor
        return (m, n) if m <= n else (n, m)
    if factor < 0:
        factor = dimension
    m, n = 1, dimension
    length = m + n
    while m < n:
        new_m = m + 1
        while dimension % new_m != 0:
            new_m += 1
        new_n = dimension // new_m
        if new_m + new_n > length or new_m > factor:
            break
        m, n = new_m, new_n
    return (m, n) if m <= n else (n, m)


class LoKrLinearLayer(_AlphaIsASpecConstant, nn.Module):
    """LoKr (Low-rank Kronecker Product) layer for Linear modules.

    Either operand may be stored full or low-rank: ``rank == 0`` keeps ``w2``
    full, ``decompose_both`` factors ``w1``. ``use_scalar`` is the same trained
    scalar as ``LoHaLinearLayer``.
    """

    ADAPTER_ALGORITHM = "lokr"

    def __init__(
        self,
        original_module: nn.Linear,
        rank: int,
        alpha: float,
        lora_name: str,
        lora_dtype: torch.dtype = torch.float32,
        factor: int = -1,
        decompose_both: bool = False,
        use_scalar: bool = False,
        factors: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
        init: bool = True,
    ):
        super().__init__()
        self.original_module = original_module
        self.rank = rank
        self.alpha = alpha
        self.lora_name = lora_name
        self.lora_dtype = lora_dtype
        self.strength = 1.0
        self.decompose_both = bool(decompose_both)

        if decompose_both and rank <= 0:
            raise ValueError("decompose_both stores w1 low-rank, so it needs rank > 0")

        out_features = original_module.out_features
        in_features = original_module.in_features
        device = original_module.weight.device

        if factors is not None:
            # A loader passes the split read off the tensors: ``factor`` is not
            # stored in a checkpoint, so ``factorization`` cannot recover it.
            (out_l, out_k), (in_m, in_n) = factors
            if out_l * out_k != out_features or in_m * in_n != in_features:
                raise ValueError(
                    f"factors {factors} do not multiply to the base's "
                    f"[{out_features}, {in_features}]")
        else:
            # ``factor`` applies to BOTH dimensions, as upstream does.
            out_l, out_k = factorization(out_features, factor)
            in_m, in_n = factorization(in_features, factor)
        self.factors = ((out_l, out_k), (in_m, in_n))

        if decompose_both:
            self.lokr_w1_a = nn.Parameter(torch.empty((out_l, rank), device=device, dtype=lora_dtype))
            self.lokr_w1_b = nn.Parameter(torch.empty((rank, in_m), device=device, dtype=lora_dtype))
        else:
            self.lokr_w1 = nn.Parameter(torch.empty((out_l, in_m), device=device, dtype=lora_dtype))

        if rank > 0:
            self.lokr_w2_a = nn.Parameter(torch.empty((out_k, rank), device=device, dtype=lora_dtype))
            self.lokr_w2_b = nn.Parameter(torch.empty((rank, in_n), device=device, dtype=lora_dtype))
            zeroed = self.lokr_w2_a
        else:
            self.lokr_w2 = nn.Parameter(torch.empty((out_k, in_n), device=device, dtype=lora_dtype))
            zeroed = self.lokr_w2

        self.scalar = (nn.Parameter(torch.zeros((), device=device, dtype=lora_dtype))
                       if use_scalar else None)
        # See ``LoHaLinearLayer.__init__`` on ``init=False``.
        if init:
            if decompose_both:
                nn.init.kaiming_uniform_(self.lokr_w1_a, a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.lokr_w1_b, a=math.sqrt(5))
            else:
                nn.init.kaiming_uniform_(self.lokr_w1, a=math.sqrt(5))
            if rank > 0:
                nn.init.kaiming_uniform_(self.lokr_w2_b, a=math.sqrt(5))
            if use_scalar:
                nn.init.kaiming_uniform_(zeroed, a=math.sqrt(5))
            else:
                nn.init.zeros_(zeroed)

    @classmethod
    def from_tensors(cls, base: nn.Module,
                     tensors: Mapping[str, torch.Tensor], *,
                     alpha: Optional[float] = None,
                     lora_dtype: torch.dtype = torch.float32,
                     lora_name: str = "",
                     use_scalar: bool = False) -> "LoKrLinearLayer":
        """Geometry from the TENSORS, which is REQUIRED rather than tidier:
        ``__init__`` derives its split from ``factorization(out_features,
        factor)`` and no LyCORIS file stores ``factor``, so a foreign LoKr
        written with a different one allocates other shapes and ``copy_``
        raises.
        """
        if use_scalar:
            _refuse_use_scalar(cls.__name__)
        out_features, in_features = _base_geometry(base)

        w1_full = tensors.get("lokr_w1")
        if w1_full is not None:
            (out_l, in_m), rank_w1, decompose_both = tuple(w1_full.shape), 0, False
        else:
            w1_a, w1_b = tensors["lokr_w1_a"], tensors["lokr_w1_b"]
            out_l, rank_w1, decompose_both = int(w1_a.shape[0]), int(w1_a.shape[1]), True
            in_m = int(w1_b.shape[1]) if w1_b.ndim == 2 else -1
            _require_shape("lokr_w1_b", w1_b, (rank_w1, in_m))

        w2_full = tensors.get("lokr_w2")
        if w2_full is not None:
            (out_k, in_n), rank = tuple(w2_full.shape), 0
        else:
            w2_a, w2_b = tensors["lokr_w2_a"], tensors["lokr_w2_b"]
            out_k, rank = int(w2_a.shape[0]), int(w2_a.shape[1])
            in_n = int(w2_b.shape[1]) if w2_b.ndim == 2 else -1
            _require_shape("lokr_w2_b", w2_b, (rank, in_n))

        if decompose_both and rank_w1 != rank:
            # One ``lora_dim`` serves both operands upstream; w1-factored +
            # w2-full is the stored form this algebra cannot represent at all
            # (``decompose_both`` needs rank > 0). See the design doc.
            raise ValueError(
                f"lokr_w1 rank {rank_w1} and lokr_w2 rank {rank} disagree; a "
                f"w1-factored operand needs a factored w2 of the same rank")
        if out_l * out_k != out_features or in_m * in_n != in_features:
            raise ValueError(
                f"lokr factors ({out_l}x{out_k}, {in_m}x{in_n}) do not match the "
                f"base's [{out_features}, {in_features}]")

        layer = cls(base, rank=rank,
                    alpha=_alpha_from_tensors(tensors, rank, alpha),
                    lora_name=lora_name, lora_dtype=lora_dtype,
                    decompose_both=decompose_both,
                    factors=((out_l, out_k), (in_m, in_n)), init=False)
        layer.adopt_tensors(tensors)
        return layer

    @property
    def scale(self) -> float:
        """Upstream's ``rank_scale``: a function of the tensor set, not of a
        constructor argument.

        The divisor is the rank of whichever operand is factored, so the
        full/full form scales by exactly 1 -- upstream reaches that by
        overriding ``alpha = lora_dim``, which is also what it writes into the
        file. Reading that stored ``alpha`` as ``alpha/rank`` scales the whole
        adapter by ``lora_dim``.

        w1 is consulted first, as upstream does: no representable checkpoint
        tells the orders apart, but an asymmetric writer would, silently.
        """
        w1_a = getattr(self, "lokr_w1_a", None)
        if w1_a is not None:
            return self.alpha / w1_a.shape[1]
        w2_a = getattr(self, "lokr_w2_a", None)
        if w2_a is not None:
            return self.alpha / w2_a.shape[1]
        return 1.0

    @property
    def weight(self):
        return self.original_module.weight

    @property
    def bias(self):
        return getattr(self.original_module, "bias", None)

    def set_adapter_strength(self, strength: float) -> None:
        self.strength = float(strength)

    def compute_delta_weight(self) -> torch.Tensor:
        w1 = self.lokr_w1_a @ self.lokr_w1_b if self.decompose_both else self.lokr_w1
        w2 = self.lokr_w2_a @ self.lokr_w2_b if self.rank > 0 else self.lokr_w2
        delta = torch.kron(w1, w2) * self.scale
        if self.scalar is not None:
            delta = delta * self.scalar
        return delta * self.strength

    def reference_delta(self, x: torch.Tensor) -> torch.Tensor:
        delta_w = self.compute_delta_weight().to(x.dtype)
        return F.linear(x, delta_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original_module(x) + self.forward_delta(x)

    def branch_tensors(self) -> Dict[str, torch.Tensor]:
        """LIVE tensors; see ``LoHaLinearLayer.branch_tensors`` -- an exporter
        must fold ``scalar`` into ``lokr_w1``/``lokr_w1_a`` and drop the key."""
        tensors: Dict[str, torch.Tensor] = {}
        if self.decompose_both:
            tensors["lokr_w1_a"] = self.lokr_w1_a
            tensors["lokr_w1_b"] = self.lokr_w1_b
        else:
            tensors["lokr_w1"] = self.lokr_w1
        if self.rank > 0:
            tensors["lokr_w2_a"] = self.lokr_w2_a
            tensors["lokr_w2_b"] = self.lokr_w2_b
        else:
            tensors["lokr_w2"] = self.lokr_w2
        if self.scalar is not None:
            tensors["scalar"] = self.scalar
        tensors["alpha"] = torch.tensor(self.alpha, dtype=torch.float32)
        return tensors

    def export_tensors(self) -> Dict[str, torch.Tensor]:
        return fold_scalar_for_export(
            self, "lokr_w1_a" if self.decompose_both else "lokr_w1")


def dora_magnitude_axis(dora_scale: torch.Tensor, out_features: int,
                        in_features: int) -> int:
    """Which axis of the weight ``dora_scale`` carries one magnitude per.

    The shape is the only record of upstream's ``wd_on_out``: ``True`` (its
    default) stores ``(out, 1)`` and norms per output row, ``False`` stores
    ``(1, in)`` and norms per input column. Refused rather than reshaped,
    because a ``view(-1, 1)`` on a ``(1, in)`` vector raises only when
    ``in != out`` -- on a square projection, which every attention
    ``to_q``/``to_k``/``to_v``/``to_out`` is, it silently applies column
    magnitudes as row magnitudes.
    """
    shape = tuple(dora_scale.shape)
    if shape in ((out_features,), (out_features, 1)):
        return 1
    if shape == (1, in_features):
        return 0
    raise ValueError(
        f"dora_scale shape {shape} is neither ({out_features}, 1) nor "
        f"(1, {in_features}) for a [{out_features}, {in_features}] weight")


def weight_decompose_refusal(base: nn.Module) -> Optional[str]:
    """Why a weight-decomposed branch must not cover ``base``.

    The magnitude epilogue reads the base weight's direction and norm every
    forward, so a weight-only quantized base would be dequantized per call and
    the fused base GEMM abandoned. Refused until measured (design doc phase 3).

    Keyed on the weight's dtype, not the quantized Linear classes: the legacy
    fp8 path leaves an ordinary ``nn.Linear`` holding a float8 weight.

    A policy predicate, not an invariant: ``DoRALinearLayer`` stays
    constructible so the session can ask the built branch rather than the
    file's label.
    """
    weight = getattr(base, "weight", None)
    if weight is None:
        return (f"{type(base).__name__} exposes no weight, so it has no "
                f"direction or norm to decompose")
    dtype = weight.dtype
    if not dtype.is_floating_point or "float8" in str(dtype):
        return (f"{type(base).__name__} holds a {dtype} weight; a "
                f"weight-decomposed adapter needs the base weight's direction "
                f"and norm, and reconstructing those over a weight-only "
                f"quantized base is a separate design with its own measurement")
    return None


class DoRALinearLayer(_BranchTensorProtocol, nn.Module):
    """DoRA (Weight-Decomposed Low-Rank Adaptation) wrapper for an adapter branch.

    Strength is the interpolation contract ``W_eff(s) = W0 + s * (W_adapter - W0)``
    over an inner branch held at unit strength; "Runtime hazards" item 2 in
    ``docs/guides/LYCORIS_ADAPTER_DESIGN.md`` says why upstream's order is refused.
    """

    WEIGHT_DECOMPOSE = True

    def __init__(
        self,
        original_module: nn.Linear,
        branch: nn.Module,
        dora_scale: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.original_module = original_module
        self.branch = branch
        self.strength = 1.0
        # A strength a loader folded into the branch (the composite's house
        # convention) would enter v before the magnitude epilogue and again here.
        if callable(getattr(branch, "set_adapter_strength", None)):
            branch.set_adapter_strength(1.0)

        device = original_module.weight.device
        # The magnitude is a TRAINED tensor of the branch, so it takes the
        # BRANCH's dtype: the base's would give an fp16 run an fp16 magnitude
        # with no fp32 master while its factors kept one. At load time the two
        # usually coincide (``lora_branch_dtype`` is the base's own float
        # dtype) -- but not by construction: Lens's branch dtype prefers the
        # BIAS, so a base whose bias dtype differs moves the magnitude with
        # its factors, which is the intended dtype either way.
        if dtype is None:
            dtype = getattr(branch, "lora_dtype", None) or original_module.weight.dtype
        self.lora_dtype = dtype

        if dora_scale is not None:
            dora_magnitude_axis(dora_scale, original_module.out_features,
                                original_module.in_features)  # refuse a bad shape here
            self.dora_scale = nn.Parameter(dora_scale.detach().clone().to(device=device, dtype=dtype))
        else:
            # Upstream's wd_on_out=True default, in its 1-D spelling.
            with torch.no_grad():
                init_scale = torch.norm(original_module.weight.detach().to(torch.float32), p=2, dim=1)
            self.dora_scale = nn.Parameter(init_scale.to(device=device, dtype=dtype))

    @property
    def weight(self):
        return self.original_module.weight

    @property
    def bias(self):
        return getattr(self.original_module, "bias", None)

    # rank/alpha/lora_name delegate to the inner branch, both ways: a loader
    # that reads them off a built branch (Z-Image logs them) or WRITES them
    # after building (MiniMax-H3's fused-QKV split) must not meet a wrapper
    # that has none of its own.
    @property
    def rank(self):
        return getattr(self.branch, "rank", None)

    @rank.setter
    def rank(self, value):
        self.branch.rank = value
        self._refresh_inner_scale()

    @property
    def alpha(self):
        return getattr(self.branch, "alpha", None)

    @alpha.setter
    def alpha(self, value):
        self.branch.alpha = float(value)
        self._refresh_inner_scale()

    def _refresh_inner_scale(self) -> None:
        """Re-derive the inner branch's scale through its OWN rule -- never a
        copy of it here. ``set_alpha`` is that rule for LoHa (LoKr's scale is a
        property off its tensors, so the call is a no-op there);
        ``set_adapter_strength(1.0)`` is it for LoRA, and unit strength is
        exactly what the epilogue holds the branch at.
        """
        for name, argument in (("set_alpha", self.branch.alpha),
                               ("set_adapter_strength", 1.0)):
            setter = getattr(self.branch, name, None)
            if callable(setter):
                setter(argument)
                return

    @property
    def lora_name(self):
        return getattr(self.branch, "lora_name", "")

    @property
    def ADAPTER_ALGORITHM(self) -> str:  # noqa: N802 - matches the class attr it shadows
        """The algebra under the decomposition; the pair is (this, True)."""
        return getattr(self.branch, "ADAPTER_ALGORITHM", "")

    def set_adapter_strength(self, strength: float) -> None:
        """Owned here, not delegated to the branch -- see the class docstring."""
        self.strength = float(strength)

    def branch_delta_weight(self) -> torch.Tensor:
        """The additive branch's own delta weight, fp32."""
        if callable(getattr(self.branch, "compute_delta_weight", None)):
            return self.branch.compute_delta_weight().to(torch.float32)
        w0 = self.original_module.weight
        eye = torch.eye(w0.shape[1], device=w0.device, dtype=w0.dtype)
        return self.branch.forward_delta(eye).T.to(torch.float32)

    def compute_delta_weight(self) -> torch.Tensor:
        w0 = self.original_module.weight.to(torch.float32)
        v = w0 + self.branch_delta_weight()
        out_features, in_features = w0.shape
        axis = dora_magnitude_axis(self.dora_scale, out_features, in_features)
        v_norm = torch.norm(v, p=2, dim=axis, keepdim=True).clamp_min(1e-12)
        magnitudes = self.dora_scale.to(torch.float32).reshape(
            (out_features, 1) if axis == 1 else (1, in_features))
        return (magnitudes * (v / v_norm) - w0) * self.strength

    def reference_delta(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.compute_delta_weight().to(x.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original_module(x) + self.forward_delta(x)

    def branch_tensors(self) -> Dict[str, torch.Tensor]:
        tensors = {}
        if callable(getattr(self.branch, "branch_tensors", None)):
            tensors.update(self.branch.branch_tensors())
        tensors["dora_scale"] = self.dora_scale
        return tensors

    def export_tensors(self) -> Dict[str, torch.Tensor]:
        """The inner branch's export -- which folds a ``scalar`` and drops the
        key -- plus the magnitude. The inherited default would emit ``scalar``
        bare and leave every other reader ``1/scalar`` too strong."""
        tensors = dict(self.branch.export_tensors())
        tensors["dora_scale"] = self.dora_scale.detach().cpu()
        return tensors

    def spec_constants(self) -> Tuple[str, ...]:
        """The inner branch's, since its tensors ride in ``branch_tensors``."""
        getter = getattr(self.branch, "spec_constants", None)
        return tuple(getter()) if callable(getter) else ()

    def load_spec_constant(self, name: str, value: torch.Tensor) -> None:
        self.branch.load_spec_constant(name, value)


#: Options ``new_adapter_branch`` forwards, per algebra. Anything else in a
#: run's ``adapter_config`` is refused by name rather than ignored.
FRESH_BRANCH_OPTIONS: Mapping[str, FrozenSet[str]] = {
    "lora": frozenset(),
    "loha": frozenset({"use_scalar"}),
    "lokr": frozenset({"factor", "decompose_both", "use_scalar"}),
}


def validate_adapter_options(algorithm: str,
                             options: Optional[Mapping[str, object]] = None
                             ) -> Dict[str, object]:
    """Refuse an option this algebra does not have, BY NAME rather than ignoring
    it. Shared so a run is refused from its config, before the model loads, in
    the same words the layer would have used."""
    allowed = FRESH_BRANCH_OPTIONS.get(algorithm)
    if allowed is None:
        raise ValueError(f"unknown adapter algorithm {algorithm!r}")
    opts = dict(options or {})
    unknown = sorted(set(opts) - set(allowed))
    if unknown:
        raise ValueError(
            f"adapter_config {unknown} is not an option of {algorithm} "
            f"(accepted: {sorted(allowed) or 'none'})")
    if opts.get("use_scalar"):
        # from_tensors refuses it and no real file carries the key, so a
        # checkpoint trained with one could not be loaded back.
        raise ValueError(
            "use_scalar cannot be trained here: the exporter folds scalar into "
            "the first factor and every reader forces scalar := 1, so a resume "
            "of the saved file would rebuild a different layer")
    return opts


def new_adapter_branch(algorithm: str, base: nn.Module, *, rank: int,
                       alpha: float, name: str = "",
                       dtype: torch.dtype = torch.float32,
                       weight_decompose: bool = False,
                       options: Optional[Mapping[str, object]] = None,
                       lora_cls: Optional[type] = None) -> nn.Module:
    """A FRESH branch of ``algorithm`` over ``base``, for training.

    The load-time counterpart is ``groups.build_adapter_branch``, which reads
    the geometry off a checkpoint's tensors; here the run's rank/alpha decide
    it. ``lora_cls`` lets an architecture keep its own ordinary-LoRA subclass
    (MiniMax-H3 casts per call); the LyCORIS algebras have no such variant.
    """
    opts = validate_adapter_options(algorithm, options)
    if weight_decompose:
        refusal = weight_decompose_refusal(base)
        if refusal is not None:
            raise ValueError(
                f"weight_decompose (DoRA/DoHa/DoKr) cannot be trained here: "
                f"{refusal}")
    if algorithm == "lora":
        cls = lora_cls or LoRALinearLayer
        branch = cls(base, rank, alpha, name, dtype)
    elif algorithm == "loha":
        branch = LoHaLinearLayer(base, rank, alpha, name, dtype)
    else:
        branch = LoKrLinearLayer(base, rank, alpha, name, dtype,
                                 factor=int(opts.get("factor", -1)),
                                 decompose_both=bool(opts.get("decompose_both", False)))
    if not weight_decompose:
        return branch
    # No dora_scale to seed from, so it starts at the base's own row norms --
    # an identity up to fp32 rounding while the branch delta is still zero.
    return DoRALinearLayer(base, branch, dtype=dtype)
