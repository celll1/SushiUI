"""Adapter leaf layers: the wrappers that carry the trainable branch.

Several algebras live here, and the eventual ``AdapterLayer`` protocol has to
accommodate all of them: the stock LoRA layer relies on an ambient
``torch.autocast`` to reconcile its fp32 masters with a bf16 activation, the
MiniMax-H3 subclass casts per call because that architecture's forward runs
without autocast. All are usable as branches of ``CompositeAdapterLayer``,
which is what lets two adapters share one base module. The LoHa/LoKr/DoRA
LOAD conventions -- tensor set, scale, magnitude axis, factorization -- follow
LyCORIS 4.0.0 at 03270a3839102e63b48578c80e7c024036de74d7. The INITIALISATIONS
deliberately do not: upstream zeroes ``lokr_w2_b`` where this zeroes
``lokr_w2_a``, and inits LoHa with ``normal_`` rather than kaiming. Equivalent
as a "starts as a no-op", different training dynamics.

Moved verbatim from ``core.training.adapters.{sd15,minimax_h3}_adapter``;
those modules now import these classes from here like everyone else.
"""

import math
from typing import (Dict, FrozenSet, Iterable, Iterator, List, Mapping,
                    Optional, Set, Tuple, Union)

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """save / resume / optimise, all derived from ``branch_tensors()`` alone.

    ``branch_tensors`` is the single extension point: an algebra with a
    different tensor set overrides that one and inherits the four below.
    """

    def branch_tensors(self) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

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
        """
        Forward pass with LoRA adaptation.

        Uses autocast to automatically handle mixed precision:
        - LoRA weights (fp32) are automatically converted to training dtype during forward
        - Gradients flow back to fp32 master weights correctly
        - GradScaler handles gradient scaling for fp16/bf16 training
        """
        org_out = self.original_module(x)

        # LoRA computation (autocast will handle dtype conversion automatically)
        # If we're in an autocast context (training_dtype), this will run in that dtype
        # Gradients will still flow back to fp32 master weights correctly
        lora_out = self.lora_up(self.lora_down(x))

        return org_out + lora_out * self.scale

    def forward_delta(self, x: torch.Tensor) -> torch.Tensor:
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
    """``LoRALinearLayer`` with the LoRA branch cast to the ACTIVATION dtype.

    MiniMax-H3's training forward runs WITHOUT ``torch.autocast``: the vendored
    transformer owns its own mixed-precision policy (fp32 I/O heads and AdaLN
    projections, bf16 block stack, each activation aligned to its projection's
    parameter dtype), and an autocast context would override those casts and make
    training a different function from generation.

    The stock layer relies on autocast to reconcile its fp32 master weights with
    a bf16 activation. Without autocast that is not a style difference, it is a
    ``RuntimeError`` on the first ``F.linear`` -- and, if the branch happened to
    be built in the activation dtype instead, a silent loss of the fp32 master.
    So the masters stay fp32 and are cast per call; the gradient flows back
    through the cast to the fp32 parameters unchanged. This is exactly the LoRA
    shape Phase 0T measured (bitwise save->reload, 600/600 tensors receiving
    finite gradients).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        org_out = self.original_module(x)
        down = F.linear(x, self.lora_down.weight.to(x.dtype))
        up = F.linear(down, self.lora_up.weight.to(x.dtype))
        return org_out + up * self.scale

    def forward_delta(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward_delta(self, x: torch.Tensor) -> torch.Tensor:
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
    """Upstream LyCORIS ``functional/general.py.factorization``: the most
    balanced divisor pair ``(m, n)`` with ``m <= n``, with the divisor search
    capped at ``factor``.

    ``factor=-1`` agrees with the balanced ``isqrt`` search it replaces on every
    dimension tested (2..20000), so adopting upstream's algorithm moves no
    existing LoKr. Upstream's own docstring table is stale: the CODE gives
    360 -> (18, 20), not the (8, 45) it documents.

    A LOADER MUST NOT CALL THIS. ``factor`` is not stored in the checkpoint, so
    a file's ``(m1, n1)`` comes from the ``lokr_w1`` shape (or
    ``w1_a.shape[0]`` / ``w1_b.shape[1]``) with ``m2 = out/m1``, ``n2 = in/n1``.
    That derivation also absorbs upstream's ``unbalanced_factorization``, which
    swaps ``out_l``/``out_k`` and which no tensor records.
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
        """Upstream's ``rank_scale`` (``kernels/autograd/lokr.py``): a FUNCTION
        OF THE TENSOR SET, not of a constructor argument.

        The divisor is the rank of whichever operand is factored, so the
        full/full form scales by exactly 1 -- upstream reaches that by
        overriding ``alpha = lora_dim`` in that branch, which is also what it
        writes into the file. Reading that stored ``alpha`` as ``alpha/rank``
        scales the whole adapter by ``lora_dim``.

        w1 is consulted first, as upstream does. No representable checkpoint can
        tell the two orders apart (one ``lora_dim`` serves both operands), but a
        future asymmetric writer would, silently.
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

    def forward_delta(self, x: torch.Tensor) -> torch.Tensor:
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
    """Why a weight-decomposed branch (DoRA/DoHa/DoKr) must not cover ``base``.

    The magnitude epilogue reads the base weight's direction and norm on every
    forward, so a weight-only quantized base would have to be dequantized per
    call and the fused base GEMM abandoned. Refused until that has its own
    design and measurement (design doc, phase 3).

    Keyed on the weight's DTYPE, not on the quantized Linear classes: the legacy
    fp8 path leaves an ordinary ``nn.Linear`` holding a float8 weight, and a
    class test would miss it.

    A POLICY predicate, not an invariant of the layer: ``new_adapter_branch``
    raises on it and ``AdapterSession`` refuses on it, while ``DoRALinearLayer``
    stays constructible so the session can ask the BUILT branch rather than the
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

    def forward_delta(self, x: torch.Tensor) -> torch.Tensor:
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


def get_module_slot(parent: nn.Module, slot: Union[str, int]) -> nn.Module:
    """Read a child module from an attribute name or a container index.

    Integer slots are not a corner case: Anima's adaln_modulation_* and
    llm_adapter targets, and several other architectures' MLPs, sit inside an
    ``nn.Sequential``, where the only address a target has is its index.
    """
    if isinstance(slot, int):
        return parent[slot]
    return getattr(parent, slot)


def set_module_slot(parent: nn.Module, slot: Union[str, int], module: nn.Module) -> None:
    """Write a child module to an attribute name or a container index."""
    if isinstance(slot, int):
        parent[slot] = module
    else:
        setattr(parent, slot, module)


class CompositeAdapterLayer(nn.Module):
    """One wrapper per base module, holding an ordered set of NAMED branches.

    THE DEFECT THIS FIXES. ``LoRALinearLayer.__init__`` reads
    ``original_module.in_features`` / ``out_features`` into locals and never
    exposes them, so it cannot wrap a wrapper: a second adapter over the same
    module raises ``AttributeError`` at construction, which is why every
    architecture is first-wins or an honest refusal today. This class owns the
    base ONCE and puts branches beside each other, so adding, removing,
    restrengthening or deactivating one rewraps nothing.

    THE NAME ENDS IN ``Layer``, NOT ``Linear``, deliberately. Every offloader in
    ``core.memory_management.block_offloading`` selects modules to move or swap
    by ``__class__.__name__.endswith("Linear")`` plus a non-None ``.weight``
    (``weighs_to_device``, ``_linear_weight_modules``, ``_build_weight_swap_jobs``).
    The ``.weight`` delegate below would then enrol the base weight a SECOND
    time -- once at this module's path, once at ``<path>.original_module`` --
    and a paired staging swap applied twice restores the outgoing block's
    weights with no error.

    BRANCH PROTOCOL. A branch is any ``nn.Module`` with
    ``forward_delta(x) -> Tensor`` returning its contribution ALONE, already
    scaled; ``set_adapter_strength(strength)`` is required only of a branch
    whose strength is changed after installation. ``LoRALinearLayer`` and
    ``MiniMaxH3LoRALinearLayer`` differ in their forward (ambient autocast
    versus a per-call activation cast) and both satisfy it, so the composite
    never tests a branch's class. Saving, resuming and optimising a branch go
    through the tensor protocol on the branch itself (``branch_tensors`` and
    friends); the composite does not aggregate those, because naming tensors
    across several branches in one file is the codec's job, not this class's.

    NUMERICS. With one active branch the output is ``base(x) + delta`` -- the
    same two operations in the same order as ``LoRALinearLayer.forward``, hence
    bit-identical. With several, the deltas are summed first and the base added
    once, so two branches are order-independent EXACTLY (fp addition commutes)
    and three or more only up to associativity.
    """

    def __init__(self, base_module: nn.Module):
        super().__init__()
        self.original_module = base_module
        self.branches = nn.ModuleList()
        self._names: List[str] = []
        self._active: Dict[str, bool] = {}
        self._strengths: Dict[str, Optional[float]] = {}

    @classmethod
    def attach(cls, parent: nn.Module, slot: Union[str, int]) -> "CompositeAdapterLayer":
        """The composite covering ``parent[slot]``, installing one if absent.

        Idempotent by design: a second adapter over an already-wrapped slot gets
        the SAME composite back and adds a branch to it.
        """
        existing = get_module_slot(parent, slot)
        if isinstance(existing, cls):
            return existing
        composite = cls(existing)
        set_module_slot(parent, slot, composite)
        return composite

    def detach(self, parent: nn.Module, slot: Union[str, int]) -> nn.Module:
        """Put the original base module back in its slot and return it."""
        current = get_module_slot(parent, slot)
        if current is not self:
            raise ValueError(
                f"slot {slot!r} holds {type(current).__name__}, not this composite; "
                f"refusing to overwrite it")
        base = self.original_module
        set_module_slot(parent, slot, base)
        return base

    @property
    def base_module(self) -> nn.Module:
        return self.original_module

    @property
    def in_features(self):
        return self.original_module.in_features

    @property
    def out_features(self):
        return self.original_module.out_features

    @property
    def weight(self):
        """Read-only delegate, same contract as ``LoRALinearLayer.weight``."""
        return self.original_module.weight

    @property
    def bias(self):
        return getattr(self.original_module, "bias", None)

    @property
    def branch_names(self) -> Tuple[str, ...]:
        return tuple(self._names)

    def __len__(self) -> int:
        return len(self._names)

    def has_branch(self, name: str) -> bool:
        return name in self._active

    def get_branch(self, name: str) -> nn.Module:
        return self.branches[self._index(name)]

    def _index(self, name: str) -> int:
        try:
            return self._names.index(name)
        except ValueError:
            raise KeyError(f"no branch named {name!r} on {type(self).__name__}") from None

    def add_branch(self, name: str, branch: nn.Module, *,
                   strength: Optional[float] = None,
                   active: bool = True) -> nn.Module:
        """Install ``branch`` under ``name``.

        ``strength=None`` keeps whatever scale the branch was built with, since
        every generation loader already folds the request strength into it.
        """
        if name in self._active:
            raise ValueError(f"branch {name!r} is already installed")
        if not callable(getattr(branch, "forward_delta", None)):
            raise TypeError(
                f"{type(branch).__name__} is not an adapter branch: it must define "
                f"forward_delta(x) returning its contribution alone")
        owned = getattr(branch, "original_module", None)
        if owned is not None and owned is not self.original_module:
            # A branch built against a different base is the stale-module splice
            # this engine exists to make impossible, not a merge to attempt.
            raise ValueError(
                f"branch {name!r} was built against a different base module "
                f"({type(owned).__name__}); rebuild it against this composite's base")
        self.branches.append(branch)
        self._names.append(name)
        self._active[name] = bool(active)
        self._strengths[name] = None
        if strength is not None:
            self.set_strength(name, strength)
        return branch

    def remove_branch(self, name: str) -> nn.Module:
        index = self._index(name)
        branch = self.branches[index]
        del self.branches[index]
        del self._names[index]
        del self._active[name]
        del self._strengths[name]
        return branch

    def clear_branches(self) -> None:
        for name in list(self._names):
            self.remove_branch(name)

    def set_strength(self, name: str, strength: float) -> None:
        branch = self.get_branch(name)
        setter = getattr(branch, "set_adapter_strength", None)
        if not callable(setter):
            raise TypeError(
                f"branch {name!r} ({type(branch).__name__}) cannot be restrengthened: "
                f"it defines no set_adapter_strength(strength). Applying the strength "
                f"outside the branch would change the arithmetic, not just the scale")
        setter(float(strength))
        self._strengths[name] = float(strength)

    def get_strength(self, name: str) -> Optional[float]:
        self._index(name)
        return self._strengths[name]

    def set_active(self, name: str, active: bool) -> None:
        self._index(name)
        self._active[name] = bool(active)

    def is_active(self, name: str) -> bool:
        self._index(name)
        return self._active[name]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.original_module(x)
        delta = None
        for name, branch in zip(self._names, self.branches):
            if not self._active[name]:
                continue
            contribution = branch.forward_delta(x)
            delta = contribution if delta is None else delta + contribution
        # No active branch returns the base output UNCHANGED rather than
        # base + 0, which would flip the sign of a -0.0 output.
        return out if delta is None else out + delta


_ADAPTER_WRAPPER_CLASS_NAMES = frozenset({
    "LoRALinearLayer",
    "LoHaLinearLayer",
    "LoKrLinearLayer",
    "DoRALinearLayer",
    "CompositeAdapterLayer",
})


def is_adapter_wrapper(module: nn.Module) -> bool:
    """True for a module that HIDES a base Linear behind an adapter.

    Matched by class name so a duplicated class object (the same file reached by
    two import paths) cannot make the test silently false.

    ``MiniMaxH3LoRALinearLayer`` is deliberately absent: the INT8 gates that read
    this have never counted it, and adding it would create a refusal that
    architecture does not have today. When MiniMax-H3 adopts the composite its
    wrapper root becomes a ``CompositeAdapterLayer`` and is covered then.
    """
    return type(module).__name__ in _ADAPTER_WRAPPER_CLASS_NAMES


def is_adapter_covered(module: Optional[nn.Module]) -> bool:
    """True for a slot an adapter already covers -- either wrapper class.

    The predicate the INJECTION sites want: "leave this alone / do not wrap it a
    second time / do not descend into it". ``is_adapter_wrapper`` above answers a
    different question for the INT8 gates -- it matches by class NAME and so
    deliberately excludes ``MiniMaxH3LoRALinearLayer``, whose slots those gates
    have never counted. Here the subclass must match, because MiniMax-H3's
    injector skips a slot it has already wrapped.
    """
    return isinstance(module, (LoRALinearLayer, LoHaLinearLayer, LoKrLinearLayer, DoRALinearLayer, CompositeAdapterLayer))


def named_modules_outside_adapters(
    root: nn.Module,
    prefix: str = "",
    memo: Optional[Set[nn.Module]] = None,
) -> Iterator[Tuple[str, nn.Module]]:
    """``named_modules()`` that yields an adapter-covered slot but not its inside.

    A wrapper's branches are ``nn.Linear`` children and its base is another, so a
    walk that selects Linears by class would otherwise offer the adapter's own
    ``lora_down``/``lora_up`` -- and the hidden base -- as fresh targets. Same
    order and same duplicate suppression as ``nn.Module.named_modules``, so on a
    tree that holds no adapter the two are indistinguishable.
    """
    if memo is None:
        memo = set()
    if root in memo:
        return
    memo.add(root)
    yield prefix, root
    if is_adapter_covered(root):
        return
    for name, child in root.named_children():
        yield from named_modules_outside_adapters(
            child, f"{prefix}.{name}" if prefix else name, memo)


def branch_survives_block_swap(branch: nn.Module) -> bool:
    """Whether a block offloader's name-based walk reaches every tensor
    ``branch`` OWNS.

    The offloaders move ``module.weight`` for modules whose class name ends in
    "Linear", so a LoRA branch's ``lora_down``/``lora_up`` ride with their block
    and a LoHa/LoKr layer's bare ``nn.Parameter`` factors are left behind. Asked
    of the OBJECT, so an algebra added later is classified without a table -- and
    so a checkpoint whose metadata MISLABELS its algebra cannot bypass it.

    The wrapped base is excluded: its weight is the block's own, moved exactly as
    it was before any adapter existed, and a Linear ``bias`` is not moved by that
    walk either -- requiring it here would refuse every LoRA over a biased base.
    """
    base = getattr(branch, "original_module", None)
    skip = {id(p) for p in base.parameters()} if isinstance(base, nn.Module) else set()
    carried = {id(m.weight) for m in branch.modules()
               if m.__class__.__name__.endswith("Linear")
               and getattr(m, "weight", None) is not None}
    return all(id(p) in carried for p in branch.parameters() if id(p) not in skip)


def count_adapter_wrapper_roots(model: nn.Module) -> int:
    """Number of base modules hidden behind an adapter wrapper under ``model``.

    Counts ROOTS: a composite's branches belong to one wrapped slot, not to
    several, so the walk does not descend into a match. For a tree of plain
    ``LoRALinearLayer`` wrappers -- which cannot nest, that being the defect the
    composite fixes -- this returns exactly what a flat class-name count did.
    """
    if is_adapter_wrapper(model):
        return 1
    return sum(count_adapter_wrapper_roots(child) for child in model.children())
