"""LoHa and LoKr, loaded and applied by the architectures that enable them.

The evidence for the ``core/adapters/capability.py`` flips. One row per enabled
architecture -- all eleven that build an ``AdapterSession`` -- each driving that
architecture's REAL generation loader over a synthetic LyCORIS checkpoint
written in its own key spelling, on CPU in a couple of seconds. A flip is legal
only when its architecture passes all five:

1. the file's stems are EXACTLY the set the architecture's own target iterator
   yields, and every one of them is covered -- set equality, since a count
   matches while the sets differ (the Phase 0 lesson);
2. the installed branch's ``compute_delta_weight()`` matches the fp32 oracle in
   ``core/adapters/reference.py`` at the architecture's own branch dtype, with
   ``alpha != rank`` so a regression to the rank fallback shows as a 3x scale;
3. a wrong-shape group refuses ``lora_partial`` and a partial (truncated) group
   refuses ``lora_incompatible``, on the architecture's real session;
4. ``unload`` restores the pre-adapter module BY OBJECT IDENTITY after a
   component swap performed BEFORE the unload -- the only ordering that catches
   the stale-module splice, which is why it is not reordered here;
5. plus, run separately: that architecture's own
   ``<arch>_lora_roundtrip_cheap_test.py`` still passes unchanged.

The stub trees come from those sibling files rather than being copied, so the
two cannot describe different models. Z-Image's is the production transformer.

The last rows are the stacking property nothing had exercised with MIXED
algebras -- one LoRA and one LoHa over the same module, summed at their own
strengths, in either selection order -- FLUX.2's two components, whose
text-encoder half takes the plain request strength while its transformer half
multiplies the per-block weight into it, and MiniMax-H3's fused ``qkv_proj``,
whose three row-sliced pieces must reconstruct the fused delta exactly and
whose LoKr must be REFUSED, by name, when its ``w1`` rows do not divide by
three.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/adapter_lycoris_roundtrip_cheap_test.py -v
"""

import os
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Callable, List, Optional, Tuple

import pytest
import torch
from torch import nn
from safetensors.torch import save_file

from lora_roundtrip_common import module_ids, warning_codes, warning_probe

from core.adapters import (  # noqa: E402
    AdapterIncompatible, CompositeAdapterLayer, get_module_slot,
)
from core.adapters import reference  # noqa: E402
from core.adapters.layers import factorization  # noqa: E402

RANK = 2
ALPHA = 6.0  # alpha/rank == 3, so a fall back to the rank default is a 3x error
STRENGTH = 0.75
STRENGTH_B = 0.4


# --------------------------------------------------------------------------
# Architecture rows
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Arch:
    name: str
    build: Callable[[], nn.Module]
    backend: Callable[[nn.Module], object]
    components: Callable[[object], list]
    stem: Callable[[str], str]
    load: Callable[[object, List[dict]], object]
    unload: Callable[[object], object]
    swap: Callable[[object, nn.Module], None]
    #: The architecture's own branch-dtype policy, which is what row 2 pins.
    branch_dtype: Callable[[nn.Module], torch.dtype]
    session: Callable[[object], object]
    #: The attribute name of the session's ``build_branch`` hook, so a row can
    #: record whether any target was walked.
    branch_hook: str
    #: Put a live block offloader where this backend's own probe looks, or
    #: ``None`` for an architecture that builds one only AFTER adapters install
    #: (or has none at all).
    fake_offloader: Optional[Callable[[object], None]] = None
    #: File metadata this architecture's ``prepare_file`` reads before it looks
    #: at a single tensor -- a format sniff, or a variant it must be told.
    extra_metadata: dict = field(default_factory=dict)


def file_metadata(arch: Arch) -> dict:
    return {"model_type": arch.name, **arch.extra_metadata}


def _live_offloader():
    """What both probes read: an object with a non-zero ``blocks_to_swap``.

    Building a real one needs a real model on a real device; what the mechanism
    depends on is only the two attributes the backends' own probes touch.
    """
    return SimpleNamespace(blocks_to_swap=4)


def _stage_minit2i_offloader(backend) -> None:
    """Exactly what ``_minit2i_stage_transformer`` sets, one line earlier than
    ``_load_lora_minit2i`` -- which is the whole problem."""
    backend._minit2i_offloader = _live_offloader()


def _stage_ltx2_offloader(backend) -> None:
    offloader = _live_offloader()
    backend._ltx2_block_offloader = lambda: offloader


def _stage_anima_offloader(backend) -> None:
    """What ``_anima_stage_transformer`` sets, one call before
    ``_load_lora_anima``."""
    backend._anima_offloader = _live_offloader()


def _stage_lens_offloader(backend) -> None:
    """What ``_lens_stage_transformer`` sets, one call before
    ``_load_lora_lens``."""
    backend._lens_offloader = _live_offloader()


def _stage_ideogram4_offloader(backend) -> None:
    """``_ideogram4_stage_transformers`` builds one offloader per HALF, or none
    at all -- so a single live entry is the whole request being swapped."""
    backend._ideogram4_offloaders = [("transformer", _live_offloader())]


def _zimage() -> Arch:
    from core.adapters import lora_branch_dtype
    from core.pipeline_backends.zimage import ZImageMixin
    import zimage_lora_roundtrip_cheap_test as gate

    class _B(ZImageMixin):
        def __init__(self, model):
            self.zimage_components = {"transformer": model}

    return Arch(
        name="zimage",
        build=gate.build_model,
        backend=_B,
        components=lambda b: b._zimage_lora_components(),
        stem=lambda p: "lora_transformer_" + p.replace(".", "_"),
        load=lambda b, c: b._load_lora_zimage(c),
        unload=lambda b: b._unload_lora_zimage(),
        swap=lambda b, m: b.zimage_components.__setitem__("transformer", m),
        branch_dtype=lora_branch_dtype,
        session=lambda b: b._zimage_lora_session,
        branch_hook="_zimage_build_lora_branch",
    )


def _krea2() -> Arch:
    from core.adapters import lora_branch_dtype
    from core.models.krea2.krea2_lora import flatten_to_key
    from core.pipeline_backends.krea2 import Krea2Mixin
    import krea2_lora_roundtrip_cheap_test as gate

    class _B(Krea2Mixin):
        def __init__(self, model):
            self.krea2_components = {"transformer": model}

    return Arch(
        name="krea2",
        build=gate.build_model,
        backend=_B,
        components=lambda b: b._krea2_lora_components(),
        stem=flatten_to_key,
        load=lambda b, c: b._load_lora_krea2(c),
        unload=lambda b: b._unload_lora_krea2(),
        swap=lambda b, m: b.krea2_components.__setitem__("transformer", m),
        branch_dtype=lora_branch_dtype,
        session=lambda b: b._krea2_lora_session,
        branch_hook="_krea2_build_lora_branch",
    )


def _minit2i() -> Arch:
    from core.models.minit2i.minit2i_lora import branch_dtype, flatten_to_key
    from core.pipeline_backends.minit2i import MiniT2IMixin
    import minit2i_lora_roundtrip_cheap_test as gate

    class _B(MiniT2IMixin):
        def __init__(self, model):
            self.minit2i_components = {"transformer": model}

    def load(backend, configs):
        # The generate path's own order; the TE pass is a no-op with no
        # text encoder loaded, and running it anyway keeps this the real
        # sequence rather than a shortcut through it.
        files = backend._minit2i_prepare_loras(configs)
        backend._apply_te_lora_minit2i(files)
        return backend._load_lora_minit2i(files)

    return Arch(
        name="minit2i",
        build=gate._Transformer,
        backend=_B,
        components=lambda b: b._minit2i_lora_components(),
        stem=flatten_to_key,
        load=load,
        unload=lambda b: b._unload_lora_minit2i(),
        swap=lambda b, m: b.minit2i_components.__setitem__("transformer", m),
        branch_dtype=branch_dtype,
        session=lambda b: b._minit2i_lora_session,
        branch_hook="_minit2i_build_lora_branch",
        fake_offloader=_stage_minit2i_offloader,
    )


def _ltx2() -> Arch:
    from core.adapters import lora_branch_dtype
    from core.models.ltx2.ltx2_lora import flatten_module_path
    from core.pipeline_backends.ltx2 import LTX2Mixin
    import ltx2_lora_roundtrip_cheap_test as gate

    class _B(LTX2Mixin):
        def __init__(self, model):
            self.ltx2_components = {"transformer": model}

    return Arch(
        name="ltx2",
        build=gate.build_dit,
        backend=_B,
        components=lambda b: b._ltx2_lora_components(),
        stem=lambda p: "lora_unet_" + flatten_module_path(p),
        load=lambda b, c: b._load_lora_ltx2(c),
        unload=lambda b: b._unload_lora_ltx2(),
        swap=lambda b, m: b.ltx2_components.__setitem__("transformer", m),
        branch_dtype=lora_branch_dtype,
        session=lambda b: b._ltx2_lora_session,
        branch_hook="_ltx2_build_lora_branch",
        fake_offloader=_stage_ltx2_offloader,
    )


def _anima() -> Arch:
    from core.models.anima.anima_lora import _flatten_to_sdscripts, branch_dtype
    from core.pipeline_backends.anima import AnimaMixin
    import anima_lora_roundtrip_cheap_test as gate

    class _B(AnimaMixin):
        def __init__(self, model):
            self.anima_components = {"transformer": model}

    return Arch(
        name="anima",
        build=gate.build_model,
        backend=_B,
        components=lambda b: b._anima_lora_components(),
        stem=lambda p: "lora_unet_" + _flatten_to_sdscripts(p),
        load=lambda b, c: b._load_lora_anima(c),
        unload=lambda b: b._unload_lora_anima(),
        swap=lambda b, m: b.anima_components.__setitem__("transformer", m),
        # Not ``lora_branch_dtype``: Anima asks the base for its declared
        # compute dtype, then its bias, before the weight.
        branch_dtype=branch_dtype,
        session=lambda b: b._anima_lora_session,
        branch_hook="_anima_build_lora_branch",
        fake_offloader=_stage_anima_offloader,
    )


def _lens() -> Arch:
    from core.models.lens.lens_lora import _flatten_to_sdscripts, branch_dtype
    from core.pipeline_backends.lens import LensMixin
    import lens_lora_roundtrip_cheap_test as gate

    class _B(LensMixin):
        def __init__(self, model):
            self.lens_components = {"transformer": model}

    return Arch(
        name="lens",
        build=gate.build_model,
        backend=_B,
        components=lambda b: b._lens_lora_components(),
        # The fused-QKV targets (img_qkv/txt_qkv) are ordinary Linears here, so
        # one group covers the whole fused stem -- no row split is involved.
        stem=lambda p: "lora_unet_" + _flatten_to_sdscripts(p),
        load=lambda b, c: b._load_lora_lens(c),
        unload=lambda b: b._unload_lora_lens(),
        swap=lambda b, m: b.lens_components.__setitem__("transformer", m),
        branch_dtype=branch_dtype,
        session=lambda b: b._lens_lora_session,
        branch_hook="_lens_build_lora_branch",
        fake_offloader=_stage_lens_offloader,
    )


def _ideogram4() -> Arch:
    from core.models.ideogram4.ideogram4_lora import (_flatten_to_sdscripts,
                                                      branch_dtype)
    from core.pipeline_backends.ideogram4 import Ideogram4Mixin
    import ideogram4_lora_roundtrip_cheap_test as gate

    class _B(Ideogram4Mixin):
        def __init__(self, model):
            # Conditional half only: the two transformers carry identical module
            # paths and are told apart by the key namespace, so a one-half
            # backend is what pins that ``lora_unet_`` reaches the cond branch.
            self.ideogram4_components = {"transformer": model}

    return Arch(
        name="ideogram4",
        build=gate._Stub,
        backend=_B,
        components=lambda b: b._ideogram4_lora_components(),
        stem=lambda p: "lora_unet_" + _flatten_to_sdscripts(p),
        load=lambda b, c: b._load_lora_ideogram4(c),
        unload=lambda b: b._unload_lora_ideogram4(),
        swap=lambda b, m: b.ideogram4_components.__setitem__("transformer", m),
        branch_dtype=branch_dtype,
        session=lambda b: b._ideogram4_lora_session,
        branch_hook="_ideogram4_build_lora_branch",
        fake_offloader=_stage_ideogram4_offloader,
    )


def _flux2() -> Arch:
    from core.adapters import lora_branch_dtype
    from core.pipeline_backends.flux2 import Flux2Mixin
    import flux2_lora_roundtrip_cheap_test as gate

    class _B(Flux2Mixin):
        def __init__(self, model):
            # Transformer only for the shared rows; the text-encoder half is a
            # SECOND component with its own key namespace and its own strength
            # rule, driven by the FLUX.2 rows at the end of this file.
            self.flux2_components = {"transformer": model, "text_encoder": None,
                                     "vae": None}

    return Arch(
        name="flux2",
        build=gate._Transformer,
        backend=_B,
        components=lambda b: b._flux2_lora_components(),
        stem=lambda p: "lora_transformer_" + p.replace(".", "_"),
        load=lambda b, c: b._load_lora_flux2(c),
        unload=lambda b: b._unload_lora_flux2(),
        swap=lambda b, m: b.flux2_components.__setitem__("transformer", m),
        branch_dtype=lora_branch_dtype,
        session=lambda b: b._flux2_lora_session,
        branch_hook="_flux2_build_lora_branch",
    )


def _acestep() -> Arch:
    from core.adapters import lora_branch_dtype
    from core.pipeline_backends.acestep import AceStepMixin
    import acestep_lora_roundtrip_cheap_test as gate

    class _B(AceStepMixin):
        def __init__(self, model):
            self.acestep_components = {"dit": model}

    return Arch(
        name="acestep",
        build=gate.build_dit,
        backend=_B,
        components=lambda b: b._acestep_lora_components(),
        # sd-scripts codec only. The diffusers/PEFT branch bakes
        # ``(lora_A|lora_B)`` into its key regexes, so a LyCORIS file cannot
        # reach a grouper there at all -- it matches no regex, groups nothing,
        # and is refused as a zero-target file.
        stem=lambda p: "lora_unet_" + p.replace(".", "_"),
        load=lambda b, c: b._load_lora_acestep(c),
        unload=lambda b: b._unload_lora_acestep(),
        swap=lambda b, m: b.acestep_components.__setitem__("dit", m),
        branch_dtype=lora_branch_dtype,
        session=lambda b: b._acestep_lora_session,
        branch_hook="_acestep_build_lora_branch",
    )


def _minimax_h3() -> Arch:
    from core.adapters import lora_branch_dtype
    from core.pipeline_backends.minimax_h3 import MiniMaxH3Mixin
    import minimax_h3_lora_roundtrip_cheap_test as gate

    class _B(MiniMaxH3Mixin):
        def __init__(self, model):
            self.minimax_h3_components = {"transformer": model, "variant": "fl2va"}

    return Arch(
        name="minimax_h3",
        build=gate._Stub,
        backend=_B,
        components=lambda b: b._minimax_h3_lora_components(),
        # The NATIVE spelling, which is one-to-one with the vendored module
        # names: no fusion to split, no half swap. The comfy spelling's fused
        # qkv stem is pinned by its own rows at the end of this file.
        stem=lambda p: "lora_unet_" + p.replace(".", "_"),
        load=lambda b, c: b._load_lora_minimax_h3(c, {}),
        unload=lambda b: b._unload_lora_minimax_h3(),
        swap=lambda b, m: b.minimax_h3_components.__setitem__("transformer", m),
        branch_dtype=lora_branch_dtype,
        session=lambda b: b._minimax_h3_lora_session,
        branch_hook="_minimax_h3_build_lora_branch",
        # fl2va and ref2va checkpoints are indistinguishable by key or shape, so
        # a LoRA that names neither is WARNED about; naming the loaded variant
        # is what keeps the no-warning rows about the adapter.
        extra_metadata={"base_model": "MiniMax-H3-fl2va"},
    )


def _sensenova() -> Arch:
    from core.models.sensenova.sensenova_lora import branch_dtype
    from core.pipeline_backends.sensenova import SenseNovaMixin
    import sensenova_lora_roundtrip_cheap_test as gate

    class _B(SenseNovaMixin):
        def __init__(self, model):
            self.sensenova_components = {"transformer": model}

    return Arch(
        name="sensenova",
        build=gate.build_model,
        backend=_B,
        # BOTH MoT halves, as two components over one module; the stems are the
        # module paths verbatim, which is why this is the one architecture that
        # asks the session to canonicalize a PEFT export's keys.
        components=lambda b: b._sensenova_lora_components(),
        stem=lambda p: p,
        load=lambda b, c: b._load_lora_sensenova(c),
        unload=lambda b: b._unload_lora_sensenova(),
        swap=lambda b, m: b.sensenova_components.__setitem__("transformer", m),
        branch_dtype=branch_dtype,
        session=lambda b: b._sensenova_lora_session,
        branch_hook="_sensenova_build_lora_branch",
        extra_metadata={"tensor_kind": "neo_hf_lora"},
    )


ARCHES = {a.name: a for a in (_zimage(), _krea2(), _minit2i(), _ltx2(),
                              _anima(), _lens(), _ideogram4(), _flux2(),
                              _acestep(), _minimax_h3(), _sensenova())}
NAMES = sorted(ARCHES)
ALGEBRAS = ("loha", "lokr")


# --------------------------------------------------------------------------
# Fixtures and helpers
# --------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def resolve_verbatim(monkeypatch):
    """All nine resolve through ``LoRAManager``; these files live in tmp_path."""
    from core.extensions import lora_manager as lm

    monkeypatch.setattr(lm.lora_manager, "_resolve_lora_path",
                        lambda p: str(p) if os.path.exists(str(p)) else None)


@pytest.fixture
def warnings_seen(monkeypatch):
    return warning_probe(monkeypatch)


def live_targets(arch: Arch, backend) -> List[Tuple[str, nn.Module]]:
    """``(module path, base module)`` from the architecture's OWN iterator --
    the same one ``AdapterSession`` plans over."""
    out = []
    for component in arch.components(backend):
        if component.module is None:
            continue
        for parent, slot, module_path in component.iter_targets(component.module):
            out.append((module_path, get_module_slot(parent, slot)))
    return out


def backend_model(arch: Arch, backend):
    return arch.components(backend)[0].module


def composites(model) -> set:
    return {name for name, module in model.named_modules()
            if isinstance(module, CompositeAdapterLayer)}


def sole_branch(composite):
    assert len(composite) == 1, composite.branch_names
    return composite.get_branch(composite.branch_names[0])


def _randn(shape, generator, std=0.2):
    return torch.randn(*shape, generator=generator) * std


def factor_tensors(algorithm: str, out_features: int, in_features: int,
                   generator) -> dict:
    """One target's LyCORIS factor set, at the canonical tensor names."""
    if algorithm == "loha":
        return {
            "hada_w1_a": _randn((out_features, RANK), generator),
            "hada_w1_b": _randn((RANK, in_features), generator),
            "hada_w2_a": _randn((out_features, RANK), generator),
            "hada_w2_b": _randn((RANK, in_features), generator),
            "alpha": torch.tensor(ALPHA),
        }
    if algorithm == "lokr":
        # w1 full, w2 factored: the one form whose scale is alpha/rank rather
        # than 1, so ALPHA != RANK is visible at all. The split is read off the
        # tensors by the loader (no file stores ``factor``), so any valid
        # factorization is a legal file.
        (out_l, out_k) = factorization(out_features)
        (in_m, in_n) = factorization(in_features)
        return {
            "lokr_w1": _randn((out_l, in_m), generator),
            "lokr_w2_a": _randn((out_k, RANK), generator),
            "lokr_w2_b": _randn((RANK, in_n), generator),
            "alpha": torch.tensor(ALPHA),
        }
    if algorithm == "lora":
        return {
            "lora_down.weight": _randn((RANK, in_features), generator),
            "lora_up.weight": _randn((out_features, RANK), generator),
            "alpha": torch.tensor(ALPHA),
        }
    raise AssertionError(algorithm)


_SUFFIX = {"lora_down.weight": ".lora_down.weight",
           "lora_up.weight": ".lora_up.weight",
           "alpha": ".alpha"}


def write_checkpoint(arch: Arch, backend, tmp_path, algorithm, name=None, seed=7):
    """A file covering EXACTLY the live target set, in this architecture's own
    key spelling. Returns ``(path, {module_path: canonical tensor dict})``."""
    generator = torch.Generator().manual_seed(seed)
    raw, per_target = {}, {}
    for module_path, base in live_targets(arch, backend):
        tensors = factor_tensors(algorithm, base.out_features, base.in_features,
                                 generator)
        per_target[module_path] = tensors
        stem = arch.stem(module_path)
        for key, value in tensors.items():
            raw[stem + _SUFFIX.get(key, "." + key)] = value
    path = tmp_path / (name or f"{arch.name}_{algorithm}.safetensors")
    save_file(raw, str(path), metadata=file_metadata(arch))
    return str(path), per_target


def load_one(arch: Arch, backend, path, strength=STRENGTH):
    return arch.load(backend, [{"path": str(path), "strength": strength}])


# --------------------------------------------------------------------------
# 1. set equality against the architecture's own target iterator
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("algorithm", ALGEBRAS)
def test_a_lycoris_file_covers_exactly_the_iterators_target_set(
        name, algorithm, tmp_path, warnings_seen):
    arch = ARCHES[name]
    backend = arch.backend(arch.build())
    expected = {p for p, _base in live_targets(arch, backend)}
    assert expected, f"{name}: the stub yields no targets"

    fresh = arch.backend(arch.build())
    path, _tensors = write_checkpoint(arch, fresh, tmp_path, algorithm)
    load_one(arch, backend, path)

    assert composites(backend_model(arch, backend)) == expected
    assert not warning_codes(warnings_seen)


# --------------------------------------------------------------------------
# 2. the delta against the fp32 oracle, at the architecture's branch dtype
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("algorithm", ALGEBRAS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_the_installed_delta_matches_the_fp32_oracle(name, algorithm, dtype,
                                                     tmp_path, warnings_seen):
    """``compute_delta_weight()`` against ``core/adapters/reference.py``.

    The oracle shares no code with the layer: it sums outer products and
    assembles Kronecker blocks by hand. ``alpha`` is 3x the rank, so a branch
    that fell back to the rank default is off by 3 rather than by a ULP.
    """
    arch = ARCHES[name]
    model = arch.build().to(dtype)
    backend = arch.backend(model)
    path, per_target = write_checkpoint(arch, arch.backend(arch.build().to(dtype)),
                                        tmp_path, algorithm)
    load_one(arch, backend, path)

    modules = dict(model.named_modules())
    rel = 1e-5 if dtype is torch.float32 else 3e-2
    checked = 0
    for module_path, tensors in per_target.items():
        branch = sole_branch(modules[module_path])
        base = modules[module_path].original_module
        want_dtype = arch.branch_dtype(base)
        # Non-vacuous: the bf16 row really is bf16, so the two rows are not the
        # same comparison run twice.
        assert want_dtype == dtype, (name, want_dtype)
        factor = next(t for n, t in branch.branch_tensors().items() if n != "alpha")
        assert factor.dtype == want_dtype, (name, module_path)

        expected = reference.adapter_delta_weight(
            algorithm, tensors, rank=RANK, alpha=ALPHA, strength=STRENGTH)
        actual = branch.compute_delta_weight().float()
        assert actual.shape == expected.shape, (name, module_path)
        tol = max(1e-6, float(expected.abs().max()) * rel)
        assert torch.allclose(actual, expected, rtol=rel, atol=tol), (
            name, algorithm, dtype, module_path,
            float((actual - expected).abs().max()))
        # Non-vacuity: the alpha/rank scale really moved the delta.
        assert float(expected.abs().max()) > 0
        checked += 1
    assert checked == len(per_target)
    assert not warning_codes(warnings_seen)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("algorithm", ALGEBRAS)
def test_dropping_the_alpha_tensor_changes_the_scale(name, algorithm, tmp_path,
                                                     warnings_seen):
    """The other half of ``alpha != rank``: strip ``.alpha`` and the same file
    applies at a different scale, so row 2 is not passing on a coincidence."""
    arch = ARCHES[name]
    with_alpha = arch.backend(arch.build())
    path, per_target = write_checkpoint(arch, with_alpha, tmp_path, algorithm)
    load_one(arch, with_alpha, path)

    from safetensors.torch import load_file
    stripped = tmp_path / f"{arch.name}_{algorithm}_noalpha.safetensors"
    save_file({k: v for k, v in load_file(path).items() if not k.endswith(".alpha")},
              str(stripped), metadata=file_metadata(arch))
    without = arch.backend(arch.build())
    load_one(arch, without, str(stripped))

    a = dict(backend_model(arch, with_alpha).named_modules())
    b = dict(backend_model(arch, without).named_modules())
    target = sorted(per_target)[0]
    assert sole_branch(a[target]).scale != sole_branch(b[target]).scale


# --------------------------------------------------------------------------
# 3. malformed files, on the architecture's real session
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("algorithm", ALGEBRAS)
def test_a_wrong_shape_group_refuses_lora_partial(name, algorithm, tmp_path,
                                                  warnings_seen):
    """One bad target out of many: the branch is skipped as ``SHAPE_MISMATCH``,
    which makes ``applied < declared`` and refuses the whole request."""
    from safetensors.torch import load_file

    arch = ARCHES[name]
    backend = arch.backend(arch.build())
    path, per_target = write_checkpoint(arch, arch.backend(arch.build()),
                                        tmp_path, algorithm)
    raw = load_file(path)
    victim = arch.stem(sorted(per_target)[0])
    bent = "hada_w1_b" if algorithm == "loha" else "lokr_w2_b"
    original = raw[f"{victim}.{bent}"]
    raw[f"{victim}.{bent}"] = torch.zeros(original.shape[0], original.shape[1] + 1)
    broken = tmp_path / f"{arch.name}_{algorithm}_bent.safetensors"
    save_file(raw, str(broken), metadata=file_metadata(arch))

    model = backend_model(arch, backend)
    with pytest.raises(AdapterIncompatible) as excinfo:
        load_one(arch, backend, str(broken))
    assert excinfo.value.code == "lora_partial"
    assert "lora_partial" in warning_codes(warnings_seen)
    assert not composites(model), "a partial application was installed anyway"


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("algorithm", ALGEBRAS)
def test_a_partial_tensor_group_refuses_lora_incompatible(name, algorithm,
                                                          tmp_path, warnings_seen):
    """A truncated group is DECLARED (its algebra is recognised) and applicable
    by nobody, so the file applies nothing and is refused."""
    arch = ARCHES[name]
    backend = arch.backend(arch.build())
    module_path, base = live_targets(arch, backend)[0]
    generator = torch.Generator().manual_seed(3)
    full = factor_tensors(algorithm, base.out_features, base.in_features, generator)
    keep = ("hada_w1_a", "hada_w1_b") if algorithm == "loha" else ("lokr_w1",)
    stem = arch.stem(module_path)
    path = tmp_path / f"{arch.name}_{algorithm}_half.safetensors"
    save_file({f"{stem}.{k}": full[k] for k in keep}, str(path),
              metadata=file_metadata(arch))

    model = backend_model(arch, backend)
    with pytest.raises(AdapterIncompatible) as excinfo:
        load_one(arch, backend, str(path))
    assert excinfo.value.code == "lora_incompatible"
    # The zero-target verdict, NOT the capability gate: with the row reverted
    # this file would be refused for the wrong reason and the row would pass.
    assert "not enabled" not in str(excinfo.value)
    assert "lora_incompatible" in warning_codes(warnings_seen)
    assert not composites(model)


# --------------------------------------------------------------------------
# 4. unload restores by identity, with the swap FIRST
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("algorithm", ALGEBRAS)
def test_unload_after_a_component_swap_restores_by_object_identity(
        name, algorithm, tmp_path, warnings_seen):
    """Swap first, unload second. The reverse order clears the bookkeeping and
    hides exactly the splice this guards: an unload driven by a map that
    outlived the model it described puts model A's Linears into model B.
    """
    arch = ARCHES[name]
    model_a = arch.build()
    backend = arch.backend(model_a)
    path, per_target = write_checkpoint(arch, arch.backend(arch.build()),
                                        tmp_path, algorithm)
    load_one(arch, backend, path)
    assert composites(model_a) == set(per_target)
    a_ids = module_ids(model_a)

    model_b = arch.build()
    b_before = dict(model_b.named_modules())
    assert not (a_ids & module_ids(model_b)), "setup: A and B must not share modules"

    arch.swap(backend, model_b)
    arch.unload(backend)
    assert dict(model_b.named_modules()) == b_before, "model B's graph was modified"

    load_one(arch, backend, path)
    assert composites(model_b) == set(per_target)
    arch.unload(backend)
    after = dict(model_b.named_modules())
    for module_path in per_target:
        assert after[module_path] is b_before[module_path], (name, module_path)
    assert not composites(model_b)
    assert not (module_ids(model_b) & a_ids)


# --------------------------------------------------------------------------
# The stacking row: one LoRA and one LoHa over the same module
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", NAMES)
def test_a_lora_and_a_loha_stack_over_one_module(name, tmp_path, warnings_seen):
    """``CompositeAdapterLayer``'s reason to exist, with MIXED algebras.

    Two branches of different classes over one base, each at its own strength,
    summed before the base is added -- and the same in either selection order,
    which for two branches is EXACT (fp addition commutes; three would only
    hold up to associativity).
    """
    arch = ARCHES[name]
    fresh = arch.backend(arch.build())
    lora_path, lora_t = write_checkpoint(arch, fresh, tmp_path, "lora",
                                         name="stack_lora.safetensors", seed=11)
    loha_path, loha_t = write_checkpoint(arch, fresh, tmp_path, "loha",
                                         name="stack_loha.safetensors", seed=13)
    assert set(lora_t) == set(loha_t)

    forward = arch.backend(arch.build())
    applied = arch.load(forward, [{"path": lora_path, "strength": STRENGTH},
                                  {"path": loha_path, "strength": STRENGTH_B}])
    reverse = arch.backend(arch.build())
    arch.load(reverse, [{"path": loha_path, "strength": STRENGTH_B},
                        {"path": lora_path, "strength": STRENGTH}])
    if applied is not None:
        assert applied == 2 * len(lora_t)

    one = dict(backend_model(arch, forward).named_modules())
    two = dict(backend_model(arch, reverse).named_modules())
    for module_path in sorted(lora_t):
        composite = one[module_path]
        assert len(composite) == 2, (name, module_path, composite.branch_names)
        base = composite.original_module
        x = torch.randn(3, base.in_features)

        delta = (reference.lora_delta_weight(lora_t[module_path], rank=RANK,
                                             alpha=ALPHA, strength=STRENGTH)
                 + reference.loha_delta_weight(loha_t[module_path], rank=RANK,
                                               alpha=ALPHA, strength=STRENGTH_B))
        expected = base(x) + x @ delta.T
        assert torch.allclose(composite(x), expected, atol=1e-5), (name, module_path)
        # Both branches really contribute.
        lora_only = base(x) + x @ reference.lora_delta_weight(
            lora_t[module_path], rank=RANK, alpha=ALPHA, strength=STRENGTH).T
        assert not torch.allclose(composite(x), lora_only, atol=1e-5), (
            name, module_path, "the LoHa branch is inert")

        assert torch.equal(composite(x), two[module_path](x)), (name, module_path)
    assert not warning_codes(warnings_seen)


# --------------------------------------------------------------------------
# ACE-Step: the sd-scripts codec only
# --------------------------------------------------------------------------

@pytest.mark.parametrize("algorithm", ALGEBRAS)
def test_acestep_refuses_a_lycoris_file_in_the_diffusers_spelling(
        algorithm, tmp_path, warnings_seen):
    """ACE-Step's row is enabled for its sd-scripts codec ALONE.

    Its diffusers/PEFT branch selects on ``(lora_A|lora_B)`` baked into three
    key regexes, so a LyCORIS file in that spelling reaches no grouper: it is
    neither ``is_sdscripts`` (no ``lora_unet_decoder_layers_`` prefix) nor
    ``is_diffusers`` (no ``.lora_A.``/``.lora_B.`` key), and ``prepare_file``
    refuses it outright as an unrecognized key format. Migrating that branch is
    its own step -- design doc, phase 2.
    """
    from core.adapters import AdapterIncompatible

    arch = ARCHES["acestep"]
    backend = arch.backend(arch.build())
    _path, base = live_targets(arch, backend)[0]
    generator = torch.Generator().manual_seed(5)
    tensors = factor_tensors(algorithm, base.out_features, base.in_features,
                             generator)
    stem = "transformer_blocks.0.attn.to_q"
    path = tmp_path / f"acestep_diffusers_{algorithm}.safetensors"
    save_file({f"{stem}.{k}": v for k, v in tensors.items()}, str(path),
              metadata={"model_type": "acestep"})

    model = backend_model(arch, backend)
    with pytest.raises(AdapterIncompatible) as excinfo:
        load_one(arch, backend, str(path))
    assert excinfo.value.code == "lora_incompatible"
    assert "unrecognized key format" in str(excinfo.value)
    # The codec's own verdict, not the capability gate's: ACE-Step DOES enable
    # both families, so a "not enabled" message here would be a lie.
    assert "not enabled" not in str(excinfo.value)
    assert not composites(model)


# --------------------------------------------------------------------------
# FLUX.2's two components, and the two different strength rules
# --------------------------------------------------------------------------

def _flux2_pair():
    """A backend holding BOTH FLUX.2 components, plus the file-stem functions.

    The shared rows above run FLUX.2 transformer-only, because a text-encoder
    target's module path (``text_encoder.model.layers...``) is not its path
    inside the encoder module and no generic row can compare the two.
    """
    from core.pipeline_backends.flux2 import (Flux2Mixin, _flux2_te_lora_targets,
                                              _flux2_transformer_lora_targets)
    import flux2_lora_roundtrip_cheap_test as gate

    class _B(Flux2Mixin):
        def __init__(self, transformer, text_encoder):
            self.flux2_components = {"transformer": transformer,
                                     "text_encoder": text_encoder, "vae": None}

    transformer, text_encoder = gate._Transformer(), gate._TextEncoder()
    unet = [("lora_transformer_" + key.replace(".", "_"),
             key, get_module_slot(parent, slot))
            for parent, slot, key in _flux2_transformer_lora_targets(transformer)]
    te = [(lora_name, key, getattr(parent, attr))
          for parent, attr, key, lora_name in _flux2_te_lora_targets(text_encoder)]
    return _B(transformer, text_encoder), transformer, text_encoder, unet, te


@pytest.mark.parametrize("algorithm", ALGEBRAS)
def test_flux2_covers_both_components_at_their_own_strength_rules(
        algorithm, tmp_path, warnings_seen):
    """One file, two key namespaces, two strength rules.

    ``unet_layer_weights`` multiplies the request strength for a transformer
    target -- ``_get_flux2_block_name`` maps ``transformer_blocks.N`` to
    ``DUALnn`` -- while the Qwen3 text-encoder half deliberately takes the plain
    strength. The strength is read back off the composite, which is where
    ``add_branch(strength=)`` folded it into the branch's own scale.
    """
    backend, transformer, text_encoder, unet, te = _flux2_pair()
    assert unet and te, "setup: the fixture must cover both components"

    generator = torch.Generator().manual_seed(23)
    raw = {}
    for stem, _key, base in unet + te:
        for name, value in factor_tensors(algorithm, base.out_features,
                                          base.in_features, generator).items():
            raw[stem + _SUFFIX.get(name, "." + name)] = value
    path = tmp_path / f"flux2_both_{algorithm}.safetensors"
    save_file(raw, str(path), metadata={"model_type": "flux2"})

    weights = {"DUAL00": 0.5, "DUAL01": 0.25}
    backend._load_lora_flux2([{"path": str(path), "strength": STRENGTH,
                               "unet_layer_weights": weights}])

    tf_modules = dict(transformer.named_modules())
    te_modules = dict(text_encoder.named_modules())
    assert composites(transformer) == {key for _s, key, _b in unet}
    assert composites(text_encoder) == {key[len("text_encoder."):]
                                        for _s, key, _b in te}

    for _stem, key, _base in unet:
        composite = tf_modules[key]
        block = backend._get_flux2_block_name(key)
        assert composite.get_strength(composite.branch_names[0]) == pytest.approx(
            STRENGTH * weights[block]), key
    for _stem, key, _base in te:
        composite = te_modules[key[len("text_encoder."):]]
        assert composite.get_strength(composite.branch_names[0]) == pytest.approx(
            STRENGTH), key
    # Non-vacuity: the two rules really differ on this fixture.
    assert set(weights.values()) != {1.0}
    assert not warning_codes(warnings_seen)

    backend._unload_lora_flux2()
    assert not composites(transformer) and not composites(text_encoder)


# --------------------------------------------------------------------------
# The other side of the boundary: the architectures this phase did NOT flip
# --------------------------------------------------------------------------

#: EMPTY: every architecture that builds an ``AdapterSession`` now takes both
#: families. SD1.5 and SDXL are absent because they build no session at all --
#: they load through diffusers, so a kohya LoHa is listed in the UI and reaches
#: diffusers' loader with no ``lora_incompatible`` refusal anywhere. That is a
#: known gap, recorded in docs/guides/LYCORIS_ADAPTER_DESIGN.md, not something
#: this table can express. Kept as a name so a future architecture that lands
#: unflipped has somewhere to go.
NOT_ENABLED = ()


@pytest.mark.parametrize("unenabled", ("sd15", "sdxl"))
@pytest.mark.parametrize("algorithm", ALGEBRAS)
def test_the_capability_gate_still_refuses_a_family_no_row_enables(
        algorithm, unenabled):
    """``NOT_ENABLED`` is empty, so the discriminating negative is no longer an
    architecture with a session -- it is the two rows without one. Told it is
    loading for SD1.5, a real session still refuses, which is what keeps the
    sibling below from passing on a dead check.

    An unknown NAME would not do: ``AdapterSpec.validate`` refuses that first,
    with a message about the architecture rather than the family.
    """
    import adapter_key_normalization_gate_cheap_test as keys

    session = keys._session("zimage")
    raw, _declared = keys.ARCHES["zimage"].variants[algorithm]
    _handed, codec = session._canonicalize(raw, {})
    assert codec.algorithm == algorithm
    assert session._refuse_unsupported_algebra("probe.safetensors", codec) is None
    session._architecture = unenabled
    with pytest.raises(AdapterIncompatible) as excinfo:
        session._refuse_unsupported_algebra("probe.safetensors", codec)
    assert f"{algorithm} adapters are not enabled" in str(excinfo.value)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("algorithm", ALGEBRAS)
def test_a_flipped_architectures_session_does_not_refuse(name, algorithm):
    """The sibling that makes the row above discriminating rather than a
    tautology about the names left in it."""
    import adapter_key_normalization_gate_cheap_test as keys

    session = keys._session(name)
    raw, _declared = keys.ARCHES[name].variants[algorithm]
    _handed, codec = session._canonicalize(raw, {})
    assert session._refuse_unsupported_algebra(f"{name}.safetensors", codec) is None


def test_every_session_architecture_is_on_exactly_one_side_of_the_boundary():
    """With ``NOT_ENABLED`` empty the boundary is no longer inside the session
    architectures -- it IS the session. What is still contingent, and what this
    asserts: every architecture that builds an ``AdapterSession`` is a row in
    this file AND carries both families, and the only two capability rows
    without a session are SD1.5 and SDXL, which are still ordinary LoRA because
    they load through diffusers. A new architecture that lands unflipped fails
    the second loop; one that lands with no row here fails the first.
    """
    from core.adapters.capability import ENABLED_ADAPTER_PAIRS, ORDINARY_LORA
    import adapter_key_normalization_gate_cheap_test as keys

    assert set(keys.ARCHES) == set(NAMES) | set(NOT_ENABLED)
    assert set(ENABLED_ADAPTER_PAIRS) - set(keys.ARCHES) == {"sd15", "sdxl"}
    for name in NAMES:
        assert {(a, False) for a in ALGEBRAS} <= ENABLED_ADAPTER_PAIRS[name], name
    for name in ("sd15", "sdxl"):
        assert ENABLED_ADAPTER_PAIRS[name] == frozenset({ORDINARY_LORA}), name


@pytest.mark.parametrize("name", NAMES)
def test_a_flipped_architecture_still_refuses_the_decomposition_axis(name):
    """DoRA is Phase 3. Enabling the additive algebras must not open the second
    axis, and the refusal must name the decomposition ALONE -- telling a
    Z-Image user that LoHa is unimplemented is now false."""
    import adapter_key_normalization_gate_cheap_test as keys

    session = keys._session(name)
    stem = "lora_unet_probe"
    raw = {f"{stem}.lora_down.weight": torch.zeros(RANK, 8),
           f"{stem}.lora_up.weight": torch.zeros(8, RANK),
           f"{stem}.dora_scale": torch.ones(8)}
    _handed, codec = session._canonicalize(raw, {})
    assert codec.weight_decompose is True
    with pytest.raises(AdapterIncompatible) as excinfo:
        session._refuse_unsupported_algebra("dora.safetensors", codec)
    assert "dora adapters are not enabled" in str(excinfo.value)
    assert "Phase 2" not in str(excinfo.value)


# --------------------------------------------------------------------------
# Block swap: a refusal where it would crash, an advisory where it costs VRAM
# --------------------------------------------------------------------------

#: Split by ``BLOCK_SWAP_ADAPTER_ORDER``, which records an ORDERING fact about
#: each backend's generate function: does it install adapters before or after
#: the offloader splits the blocks?
STRANDS = ("ltx2", "minit2i", "anima", "lens", "ideogram4")  # AFTER  -> refuse
SWEPT = ("zimage", "flux2", "minimax_h3")                   # BEFORE -> advise
# krea2/acestep build no generation-time offloader; SenseNova's blocks_to_swap
# is inert and its MoT phase evictor moves a module's OWN parameters, so it
# carries a LyCORIS branch with the half it sits under.
NO_SWAP = ("krea2", "acestep", "sensenova")


def _recording_backend(arch: Arch, monkeypatch, visited):
    """A backend whose session-level branch builder records every target it is
    asked about, so a row can assert the tree was never walked."""
    backend = arch.backend(arch.build())
    original = getattr(backend, arch.branch_hook)

    def record(request):
        visited.append(request.module_path)
        return original(request)

    monkeypatch.setattr(backend, arch.branch_hook, record)
    return backend


@pytest.mark.parametrize("name", STRANDS)
@pytest.mark.parametrize("algorithm", ALGEBRAS)
def test_a_lycoris_file_is_refused_while_block_swap_is_live(
        name, algorithm, tmp_path, monkeypatch, warnings_seen):
    """LTX-2.3 and MiniT2I install adapters after their offloader has split the
    blocks, so a LyCORIS branch is built on the HOST for every swapped-out
    block and nothing ever moves it -- a device mismatch mid-denoise. Refused
    instead, before anything is installed.

    The verdict is taken on the BUILT branches, so the tree IS walked first:
    planning mutates nothing, and asking the object is what makes a file whose
    metadata mislabels its algebra refusable (see the bypass row below).

    MiniT2I's text-encoder pass runs earlier and is not block-swapped; a
    transformer-pass refusal reaches ``_minit2i_cleanup`` in the generate
    function's outer ``finally``, which unloads both halves.
    """
    arch = ARCHES[name]
    visited = []
    backend = _recording_backend(arch, monkeypatch, visited)
    model = backend_model(arch, backend)
    path, tensors = write_checkpoint(arch, arch.backend(arch.build()), tmp_path,
                                     algorithm)
    arch.fake_offloader(backend)

    with pytest.raises(AdapterIncompatible) as excinfo:
        load_one(arch, backend, path)

    assert excinfo.value.code == "lora_blockswap_unsupported"
    assert "lora_blockswap_unsupported" in warning_codes(warnings_seen)
    assert "block swap is active" in str(excinfo.value)
    # Named from the class that was built, not from the file's label.
    assert ("LoHa" if algorithm == "loha" else "LoKr") in str(excinfo.value)
    assert set(visited) >= set(tensors), "planning did not reach the targets"
    assert not composites(model), "a branch was installed before the refusal"
    assert not arch.session(backend).state("transformer").wrapped


@pytest.mark.parametrize("name", STRANDS)
def test_ordinary_lora_still_loads_while_block_swap_is_live(
        name, tmp_path, monkeypatch, warnings_seen):
    """The combination that works TODAY must keep working: a LoRA branch is two
    ``nn.Linear``s, which every offloader moves. Without this row the refusal
    above could be written as "no adapters under block swap" and still pass."""
    arch = ARCHES[name]
    visited = []
    backend = _recording_backend(arch, monkeypatch, visited)
    model = backend_model(arch, backend)
    path, tensors = write_checkpoint(arch, arch.backend(arch.build()), tmp_path,
                                     "lora")
    arch.fake_offloader(backend)

    load_one(arch, backend, path)
    assert composites(model) == set(tensors)
    assert visited, "the walk never happened"
    assert "lora_blockswap_unsupported" not in warning_codes(warnings_seen)


@pytest.mark.parametrize("name", SWEPT)
@pytest.mark.parametrize("algorithm", ALGEBRAS + ("lora",))
def test_a_swept_architecture_advises_instead_of_refusing(
        name, algorithm, tmp_path, warnings_seen):
    """Z-Image installs adapters BEFORE ``prepare_block_devices``, whose
    ``blocks[i].to(device)`` sweeps every tensor -- bare factors included -- to
    the device and returns only the Linear weights to the host. Correct numbers,
    a permanently resident adapter, so: an advisory from the offloader build
    site, and the generation proceeds."""
    arch = ARCHES[name]
    backend = arch.backend(arch.build())
    model = backend_model(arch, backend)
    path, tensors = write_checkpoint(arch, arch.backend(arch.build()), tmp_path,
                                     algorithm)

    load_one(arch, backend, path)
    assert composites(model) == set(tensors), "the file was refused"

    stranded = arch.session(backend).warn_unoffloaded_branches("transformer")
    codes = warning_codes(warnings_seen)
    if algorithm == "lora":
        assert stranded == 0
        assert codes == []
    else:
        assert stranded == len(tensors)
        assert codes == ["lora_blockswap_not_offloaded"]


def test_the_block_swap_order_table_covers_every_lycoris_architecture():
    from core.adapters.capability import (BLOCK_SWAP_ADAPTER_ORDER,
                                          ENABLED_ADAPTER_PAIRS, ORDINARY_LORA)

    lycoris = {name for name, pairs in ENABLED_ADAPTER_PAIRS.items()
               if pairs != frozenset({ORDINARY_LORA})}
    assert lycoris == set(NAMES)
    assert lycoris - set(BLOCK_SWAP_ADAPTER_ORDER) == set()
    assert set(STRANDS) | set(SWEPT) | set(NO_SWAP) == lycoris


def test_exactly_the_stranding_backends_declare_a_block_swap_probe():
    """AST-only, so it costs no import. The table above is the readable policy;
    this is what keeps it from drifting from the code that enforces it."""
    import ast
    from pathlib import Path

    from core.adapters.capability import AFTER_SPLIT, BLOCK_SWAP_ADAPTER_ORDER

    backends = Path(__file__).resolve().parents[1] / "core" / "pipeline_backends"
    declared = set()
    for path in sorted(backends.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "id", None) == "AdapterComponent"
                    and any(kw.arg == "block_swap_active" for kw in node.keywords)):
                declared.add(path.stem)
    expected = {name for name, order in BLOCK_SWAP_ADAPTER_ORDER.items()
                if order == AFTER_SPLIT}
    assert declared == expected


def test_every_swept_offloader_site_asks_for_the_advisory():
    """The advisory is only true where it is CALLED, and its call site is the
    offloader build -- one per generate entry point. Source-level, because
    driving it needs a real offloader on a real device."""
    import re
    from pathlib import Path

    backends = Path(__file__).resolve().parents[1] / "core" / "pipeline_backends"
    for name in SWEPT:
        source = (backends / f"{name}.py").read_text(encoding="utf-8")
        sites = len(re.findall(r"prepare_block_devices_before_forward\(\)", source))
        asked = len(re.findall(r"warn_unoffloaded_branches\(", source))
        assert sites and asked == sites, (name, sites, asked)


@pytest.mark.parametrize("name", STRANDS)
@pytest.mark.parametrize("label", ["ss_network_module", "sushi.adapter.algorithm"])
def test_a_loha_file_that_claims_to_be_lora_is_still_refused(
        name, label, tmp_path, monkeypatch, warnings_seen):
    """The bypass a label test cannot close.

    ``CodecRegistry.detect`` gives metadata priority over keys, so a file of
    pure ``hada_*`` tensors carrying ``networks.lora`` DETECTS as ordinary LoRA.
    A predicate keyed on that string lets it through, and the per-group builder
    then constructs ``LoHaLinearLayer``s no offloader can move. Asking the built
    object instead is what makes the metadata irrelevant.
    """
    from safetensors.torch import load_file

    arch = ARCHES[name]
    visited = []
    backend = _recording_backend(arch, monkeypatch, visited)
    model = backend_model(arch, backend)
    honest, _tensors = write_checkpoint(arch, arch.backend(arch.build()), tmp_path,
                                        "loha")
    claim = {label: "networks.lora" if label == "ss_network_module" else "lora"}
    liar = tmp_path / f"{arch.name}_liar_{label.replace('.', '_')}.safetensors"
    save_file(load_file(honest), str(liar), metadata=claim)
    arch.fake_offloader(backend)

    # Non-vacuity: the codec really does read this file as ordinary LoRA.
    session = arch.session(backend)
    _handed, codec = session._canonicalize(load_file(liar), claim)
    assert codec.algorithm == "lora", (name, label, codec.algorithm)

    with pytest.raises(AdapterIncompatible) as excinfo:
        load_one(arch, backend, str(liar))
    assert excinfo.value.code == "lora_blockswap_unsupported"
    assert "LoHa" in str(excinfo.value)
    assert not composites(model)


@pytest.mark.parametrize("name", STRANDS)
def test_an_undetectable_file_is_not_refused_by_the_block_swap_gate(
        name, tmp_path, monkeypatch, warnings_seen):
    """The false refusal a label test causes.

    A valid ``lora_bias=True`` PEFT export sniffs as ``unknown``; the sibling
    policy in ``_refuse_unsupported_algebra`` deliberately leaves an
    unrecognised algebra to the architecture's own zero-target verdict, and this
    gate must agree rather than tell the user "unknown adapters cannot be
    applied while block swap is active".
    """
    from core.adapters import codec as codec_module

    arch = ARCHES[name]
    visited = []
    backend = _recording_backend(arch, monkeypatch, visited)
    path, tensors = write_checkpoint(arch, arch.backend(arch.build()), tmp_path,
                                     "lora", name="undetectable.safetensors")
    arch.fake_offloader(backend)

    # How ``unknown`` is really reached: ``_canonicalize`` fabricates it whenever
    # detection RAISES, which it does on shapes it has not validated.
    def boom(*_a, **_k):
        raise IndexError("tuple index out of range")

    monkeypatch.setattr(codec_module.CodecRegistry, "detect", staticmethod(boom))
    session = arch.session(backend)
    _handed, codec = session._canonicalize({}, {})
    assert codec.algorithm == "unknown", (name, codec.algorithm)

    load_one(arch, backend, path)
    assert "lora_blockswap_unsupported" not in warning_codes(warnings_seen)
    assert composites(backend_model(arch, backend)) == set(tensors)


# --------------------------------------------------------------------------
# MiniMax-H3: the fused QKV row split, and the fc1 half swap generalized
# --------------------------------------------------------------------------

_H3_PREFIX = "diffusion_model."
_H3_IN = 16          # minimax_h3_lora_roundtrip_cheap_test._HIDDEN
_H3_HEAD = 8         # ... _INNER, one projection's output rows
_H3_QKV_OUT = 3 * _H3_HEAD
_H3_FFN = 24         # ... _FFN, the fc1 output rows


def _h3_backend():
    arch = ARCHES["minimax_h3"]
    backend = arch.backend(arch.build())
    return arch, backend, backend_model(arch, backend)


def _h3_write(tmp_path, name, stem, tensors):
    """One comfy-spelled stem, which is the only spelling with a fused qkv."""
    raw = {}
    for key, value in tensors.items():
        raw[_H3_PREFIX + stem + _SUFFIX.get(key, "." + key)] = value
    path = tmp_path / name
    save_file(raw, str(path), metadata=file_metadata(ARCHES["minimax_h3"]))
    return str(path)


def _h3_fused_factors(algorithm, out_features, generator, *, w1_rows=None):
    """A fused stem's factors at ``(out_features, _H3_IN)``.

    ``w1_rows`` is the LoKr knob this pair of rows exists for: a contiguous
    piece is another Kronecker product only when the piece count divides it,
    and no file stores the factorization, so a checkpoint may carry any pair
    that multiplies out.
    """
    if algorithm in ("lora", "loha"):
        return factor_tensors(algorithm, out_features, _H3_IN, generator)
    assert algorithm == "lokr"
    in_m, in_n = factorization(_H3_IN)
    assert out_features % w1_rows == 0
    return {
        "lokr_w1": _randn((w1_rows, in_m), generator),
        "lokr_w2_a": _randn((out_features // w1_rows, RANK), generator),
        "lokr_w2_b": _randn((RANK, in_n), generator),
        "alpha": torch.tensor(ALPHA),
    }


@pytest.mark.parametrize("algorithm,w1_rows",
                         [("lora", None), ("loha", None), ("lokr", 3)])
def test_minimax_h3_qkv_pieces_reconstruct_the_fused_delta(
        algorithm, w1_rows, tmp_path, warnings_seen):
    """One fused ``attn.qkv_proj`` stem -> to_q/to_k/to_v, exactly.

    ``delta[rows] = up[rows, :] @ down`` makes the row slice exact for LoRA and
    LoHa; for LoKr it is exact only under the parent's own ``(out_l, out_k)``
    split, so this row uses a ``w1`` whose rows divide by 3 and its sibling
    below uses one that does not.
    """
    arch, backend, model = _h3_backend()
    generator = torch.Generator().manual_seed(31)
    fused = _h3_fused_factors(algorithm, _H3_QKV_OUT, generator, w1_rows=w1_rows)
    path = _h3_write(tmp_path, "h3_qkv_" + algorithm + ".safetensors",
                     "blocks.0.attn.qkv_proj", fused)

    assert arch.load(backend, [{"path": path, "strength": STRENGTH}]) == 3
    modules = dict(model.named_modules())
    pieces = [sole_branch(modules["transformer_blocks.0.attn." + leaf]).compute_delta_weight()
              for leaf in ("to_q", "to_k", "to_v")]
    assert [tuple(p.shape) for p in pieces] == [(_H3_HEAD, _H3_IN)] * 3

    expected = reference.adapter_delta_weight(algorithm, fused, rank=RANK,
                                              alpha=ALPHA, strength=STRENGTH)
    assert expected.shape == (_H3_QKV_OUT, _H3_IN)
    actual = torch.cat(pieces, dim=0).float()
    assert torch.allclose(actual, expected, atol=1e-6), (
        algorithm, float((actual - expected).abs().max()))
    # Non-vacuity: the thirds really differ, so a wrong assignment would show.
    assert not torch.allclose(pieces[0], pieces[1], atol=1e-6)
    assert not warning_codes(warnings_seen)


def test_minimax_h3_qkv_pieces_do_not_share_one_storage(tmp_path):
    """``split_group_on_out_rows`` shares the non-sliced factors BY REFERENCE and
    ``from_tensors`` ADOPTS, so the three branches would otherwise alias one
    buffer -- and any in-place write would reach all three. The split clones."""
    arch, backend, model = _h3_backend()
    generator = torch.Generator().manual_seed(37)
    fused = _h3_fused_factors("loha", _H3_QKV_OUT, generator)
    path = _h3_write(tmp_path, "h3_qkv_shared.safetensors",
                     "blocks.0.attn.qkv_proj", fused)
    arch.load(backend, [{"path": path, "strength": STRENGTH}])

    modules = dict(model.named_modules())
    branches = [sole_branch(modules["transformer_blocks.0.attn." + leaf])
                for leaf in ("to_q", "to_k", "to_v")]
    shared = [b.hada_w1_b for b in branches]
    assert len({t.data_ptr() for t in shared}) == 3
    before = branches[1].compute_delta_weight().clone()
    with torch.no_grad():
        shared[0].add_(1.0)
    assert torch.equal(branches[1].compute_delta_weight(), before)


def test_minimax_h3_refuses_a_fused_qkv_lokr_that_straddles_a_block(
        tmp_path, warnings_seen):
    """A LoKr whose ``w1`` rows are not divisible by 3 is REFUSED, and told why.

    Every matrix is a degenerate Kronecker product of a 1x1 with itself, so
    emitting the slice anyway would be a numerically different adapter, not a
    rounding difference (measured 0.31 off in ``split_group_on_out_rows``). The
    message must not read as a corrupt file: nothing about this one is.
    """
    arch, backend, model = _h3_backend()
    generator = torch.Generator().manual_seed(41)
    fused = _h3_fused_factors("lokr", _H3_QKV_OUT, generator, w1_rows=4)
    path = _h3_write(tmp_path, "h3_qkv_lokr_straddle.safetensors",
                     "blocks.0.attn.qkv_proj", fused)

    with pytest.raises(AdapterIncompatible) as excinfo:
        arch.load(backend, [{"path": path, "strength": STRENGTH}])
    message = str(excinfo.value)
    assert excinfo.value.code == "lora_incompatible"
    assert "lora_incompatible" in warning_codes(warnings_seen)
    assert "3 divides w1's 4 rows" in message
    assert "well formed" in message
    # Not a corrupt file, and not the capability gate: the row IS enabled.
    assert "truncated or corrupt" not in message
    assert "not enabled" not in message
    assert not composites(model)


@pytest.mark.parametrize("algorithm,w1_rows",
                         [("lora", None), ("loha", None), ("lokr", 4)])
def test_minimax_h3_swaps_the_fc1_halves_on_every_algebra(
        algorithm, w1_rows, tmp_path, warnings_seen):
    """Comfy ``[gate; up]`` -> vendored SwiGLU ``[up; gate]``, generalized.

    A permutation of the OUT axis, so it moves ``lora_up``, both LoHa ``*_a``
    factors, or a LoKr's ``w1`` -- and getting it wrong is silent: the shapes
    match, the load is clean, and the gate delta lands in the up path.
    """
    arch, backend, model = _h3_backend()
    generator = torch.Generator().manual_seed(43)
    fused = _h3_fused_factors(algorithm, _H3_FFN, generator, w1_rows=w1_rows)
    path = _h3_write(tmp_path, "h3_fc1_" + algorithm + ".safetensors",
                     "blocks.0.mlp.fc1", fused)

    assert arch.load(backend, [{"path": path, "strength": STRENGTH}]) == 1
    installed = sole_branch(dict(model.named_modules())["transformer_blocks.0.ff.net.0.proj"])
    comfy = reference.adapter_delta_weight(algorithm, fused, rank=RANK,
                                           alpha=ALPHA, strength=STRENGTH)
    gate_half, up_half = comfy.chunk(2, dim=0)
    expected = torch.cat([up_half, gate_half], dim=0)
    actual = installed.compute_delta_weight().float()
    assert torch.allclose(actual, expected, atol=1e-6), algorithm
    # Non-vacuity: the unswapped delta is a different matrix.
    assert not torch.allclose(actual, comfy, atol=1e-6)
    assert not warning_codes(warnings_seen)


def test_minimax_h3_refuses_an_fc1_lokr_whose_halves_straddle_a_block(
        tmp_path, warnings_seen):
    """The same conditional as the qkv split, at 2 pieces instead of 3: the
    half swap is a whole-block permutation only when ``w1``'s rows are even."""
    arch, backend, model = _h3_backend()
    generator = torch.Generator().manual_seed(47)
    fused = _h3_fused_factors("lokr", _H3_FFN, generator, w1_rows=3)
    path = _h3_write(tmp_path, "h3_fc1_lokr_straddle.safetensors",
                     "blocks.0.mlp.fc1", fused)

    with pytest.raises(AdapterIncompatible) as excinfo:
        arch.load(backend, [{"path": path, "strength": STRENGTH}])
    message = str(excinfo.value)
    assert excinfo.value.code == "lora_incompatible"
    assert "w1's 3 rows are even" in message
    assert "well formed" in message
    assert "truncated or corrupt" not in message and "not enabled" not in message
    assert not composites(model)


def test_minimax_h3_native_and_comfy_lycoris_reach_the_same_delta(tmp_path):
    """The two key conventions are one adapter. A native LoHa on the three
    unfused attention leaves must install the same three deltas a comfy file
    reaches through the fused stem -- which is what makes the split a key codec
    rather than a second algebra."""
    arch, backend, model = _h3_backend()
    generator = torch.Generator().manual_seed(53)
    fused = _h3_fused_factors("loha", _H3_QKV_OUT, generator)
    comfy = _h3_write(tmp_path, "h3_pair_comfy.safetensors",
                      "blocks.0.attn.qkv_proj", fused)
    arch.load(backend, [{"path": comfy, "strength": STRENGTH}])
    leaves = ("to_q", "to_k", "to_v")
    modules = dict(model.named_modules())
    from_comfy = {leaf: sole_branch(modules["transformer_blocks.0.attn." + leaf])
                  .compute_delta_weight().clone() for leaf in leaves}

    raw = {}
    for index, leaf in enumerate(leaves):
        rows = slice(index * _H3_HEAD, (index + 1) * _H3_HEAD)
        stem = "lora_unet_transformer_blocks_0_attn_" + leaf
        for name, value in fused.items():
            # .clone(): safetensors refuses to save three stems sharing one
            # storage, which the shared _b factors would otherwise be.
            raw[stem + "." + name] = (value[rows].contiguous()
                                      if name in ("hada_w1_a", "hada_w2_a")
                                      else value.clone())
    native_path = tmp_path / "h3_pair_native.safetensors"
    save_file(raw, str(native_path), metadata=file_metadata(ARCHES["minimax_h3"]))

    other_backend = arch.backend(arch.build())
    arch.load(other_backend, [{"path": str(native_path), "strength": STRENGTH}])
    other = dict(backend_model(arch, other_backend).named_modules())
    for leaf in leaves:
        native = sole_branch(other["transformer_blocks.0.attn." + leaf])
        assert torch.allclose(native.compute_delta_weight(), from_comfy[leaf],
                              atol=1e-6), leaf


# --------------------------------------------------------------------------
# SenseNova: the one architecture that OPTS IN to key canonicalization
# --------------------------------------------------------------------------

@pytest.mark.parametrize("algorithm", ALGEBRAS)
def test_sensenova_canonicalizes_a_peft_prefixed_lycoris_file(
        algorithm, tmp_path, warnings_seen):
    """``canonicalize_foreign_keys=True`` has to carry LyCORIS too.

    SenseNova's parser matches the suffix on a VERBATIM module path, so a PEFT
    export differs from its own spelling only by ``base_model.model.``.
    ``normalize_keys`` strips exactly that; its ``lora_A``/``lora_B`` rewrites
    never fire on ``hada_*``/``lokr_*`` keys, so the same rewrite that made a
    PEFT LoRA loadable makes a PEFT LoHa loadable. Without the strip the stems
    parse and then match no live module, which is a zero-target refusal.
    """
    arch = ARCHES["sensenova"]
    backend = arch.backend(arch.build())
    plain, per_target = write_checkpoint(arch, arch.backend(arch.build()),
                                         tmp_path, algorithm)

    from safetensors.torch import load_file
    prefixed = tmp_path / f"sensenova_peft_{algorithm}.safetensors"
    save_file({"base_model.model." + k: v.clone()
               for k, v in load_file(plain).items()},
              str(prefixed), metadata=file_metadata(arch))

    model = backend_model(arch, backend)
    load_one(arch, backend, str(prefixed))
    assert composites(model) == set(per_target)
    assert not warning_codes(warnings_seen)


# --------------------------------------------------------------------------
# The quantized base: what MiniMax-H3's and SenseNova's targets ACTUALLY are
# --------------------------------------------------------------------------

def _quantized_stub(cls, bias):
    """The SenseNova stub with every ``nn.Linear`` replaced by ``cls``.

    Its 294 targets per MoT half are all ``Int8Linear`` in production and
    MiniMax-H3's block stack is all ``Fp8Linear``, so a gate built entirely on
    ``nn.Linear`` never touches the layer class these two actually carry. What
    changed for them is the shape check: it moved from a
    ``getattr(base, "in_features", None)``-tolerant comparison to
    ``_base_geometry``, which RAISES on a base that answers neither -- caught by
    ``build_adapter_branch`` and returned as ``SHAPE_MISMATCH``, i.e. a target
    silently skipped and then a ``lora_partial`` refusal for the whole file.
    """
    model = ARCHES["sensenova"].build()
    if cls is nn.Linear:
        return model
    for parent in model.modules():
        for attr, child in list(parent.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            setattr(parent, attr, cls(child.in_features, child.out_features,
                                      bias=bias, compute_dtype=torch.bfloat16))
    return model


def _quantized_classes():
    from core.models.ideogram4.vendor.fp8_linear import Fp8Linear
    from core.models.ideogram4.vendor.int8_linear import Int8Linear

    return {"int8": Int8Linear, "fp8": Fp8Linear, "dense": nn.Linear}


@pytest.mark.parametrize("algorithm", ALGEBRAS)
@pytest.mark.parametrize("kind,bias", [("dense", True), ("int8", True),
                                       ("int8", False), ("fp8", False)])
def test_a_lycoris_branch_installs_over_a_quantized_base(
        kind, bias, algorithm, tmp_path, warnings_seen):
    """The declaration these two rows made true.

    `quantized_base_additive_family` is True for MiniMax-H3 and SenseNova alone
    because neither has a dense configuration to ship first. This is the row
    that backs it: the same file, over `Int8Linear` and `Fp8Linear` with and
    without a bias, reaching every target and matching the fp32 oracle. The
    branch dtype is the architecture's own — the bias when there is a real
    floating one, bfloat16 for a quantized weight with none — because an int8
    weight is not floating point at all.
    """
    from core.models.sensenova.sensenova_lora import _is_lora_target

    cls = _quantized_classes()[kind]
    arch = ARCHES["sensenova"]
    backend = arch.backend(_quantized_stub(cls, bias))
    model = backend_model(arch, backend)
    targets = live_targets(arch, backend)
    assert targets and all(isinstance(base, cls) for _p, base in targets), kind
    assert all(_is_lora_target(base) for _p, base in targets), kind

    path, per_target = write_checkpoint(
        arch, arch.backend(_quantized_stub(cls, bias)), tmp_path, algorithm,
        name=f"sensenova_{kind}_{int(bias)}_{algorithm}.safetensors")
    load_one(arch, backend, path)
    assert composites(model) == set(per_target)
    assert not warning_codes(warnings_seen)

    modules = dict(model.named_modules())
    # A quantized weight is float8 or not floating at all, so the bias decides
    # when there is one and bfloat16 is the floor when there is not.
    expect_dtype = torch.float32 if kind == "dense" else torch.bfloat16
    for module_path, tensors in per_target.items():
        branch = sole_branch(modules[module_path])
        base = modules[module_path].original_module
        assert arch.branch_dtype(base) == expect_dtype, (kind, bias)
        factor = next(t for n, t in branch.branch_tensors().items() if n != "alpha")
        assert factor.dtype == expect_dtype, (kind, bias, module_path)
        expected = reference.adapter_delta_weight(
            algorithm, tensors, rank=RANK, alpha=ALPHA, strength=STRENGTH)
        actual = branch.compute_delta_weight().float()
        assert torch.allclose(actual, expected, rtol=3e-2,
                              atol=max(1e-6, float(expected.abs().max()) * 3e-2)), (
            kind, bias, module_path)

    arch.unload(backend)
    assert not composites(model)
