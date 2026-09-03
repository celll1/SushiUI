"""The session's key-normalization contract, for EVERY architecture at once.

``AdapterSession`` may rewrite a Diffusers/PEFT checkpoint's keys into canonical
``lora_down``/``lora_up`` stems before the architecture parses it. Whether that
helps or destroys the file depends on the architecture: most of them parse
``lora_A``/``lora_B`` themselves, so a rewrite leaves their foreign-format
branch matching nothing and the file is refused (``lora_incompatible``, HTTP
400) although it was valid. That is what shipped for ACE-Step, MiniMax-H3,
Anima, Lens and Ideogram 4 when normalization was made unconditional. Where the
branch is chosen differs and does not save them: ACE-Step selects on the
suffix, MiniMax-H3 on the PREFIX (``diffusion_model.`` vs ``lora_unet_``) and
then fails one level down in its own ``_parse_key``.

The contract this file pins is one sentence: **every key format an
architecture declares support for must still be counted by that
architecture's OWN declared-branch counter after the session has handed the
tensors over.** Both halves come from the REAL session the backend builds --
the counter and the canonicalization flag -- so a future change to either is
what fails here, not a restatement of them.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/adapter_key_normalization_gate_cheap_test.py -v
"""

from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Dict, Optional, Tuple

import pytest
import torch

import lora_roundtrip_common  # noqa: F401  (sys.path bootstrap)

from core.adapters.codec import CodecRegistry  # noqa: E402
from core.adapters.session import AdapterFile  # noqa: E402
from core.adapters.spec import FORMAT_PEFT, FORMAT_UNKNOWN  # noqa: E402

RANK = 4
DIM = 8

_GEN = torch.Generator().manual_seed(20260904)


def _pair(stem: str, down: str, up: str, out: int = DIM) -> Dict[str, torch.Tensor]:
    """One down/up pair under ``stem``, spelled with the given suffixes."""
    return {
        f"{stem}.{down}": torch.randn(RANK, DIM, generator=_GEN),
        f"{stem}.{up}": torch.randn(out, RANK, generator=_GEN),
    }


def _canonical(stem: str, out: int = DIM) -> Dict[str, torch.Tensor]:
    return _pair(stem, "lora_down.weight", "lora_up.weight", out)


def _peft(stem: str, out: int = DIM) -> Dict[str, torch.Tensor]:
    return _pair(stem, "lora_A.weight", "lora_B.weight", out)


def _loha(stem: str, out: int = DIM) -> Dict[str, torch.Tensor]:
    """A LyCORIS LoHa factor set: no ``.lora_down.weight`` key anywhere."""
    return {
        f"{stem}.hada_w1_a": torch.randn(out, RANK, generator=_GEN),
        f"{stem}.hada_w1_b": torch.randn(RANK, DIM, generator=_GEN),
        f"{stem}.hada_w2_a": torch.randn(out, RANK, generator=_GEN),
        f"{stem}.hada_w2_b": torch.randn(RANK, DIM, generator=_GEN),
        f"{stem}.alpha": torch.tensor(float(RANK)),
    }


def _half(stem: str) -> Dict[str, torch.Tensor]:
    """A stem carrying ``lora_down`` and no ``lora_up``: a checkpoint truncated
    mid-write, or a converter that dropped half a pair."""
    return {f"{stem}.lora_down.weight": torch.randn(RANK, DIM, generator=_GEN)}


def _lokr(stem: str, out: int = DIM) -> Dict[str, torch.Tensor]:
    """A LyCORIS LoKr factor set: a full ``lokr_w1`` and a decomposed w2."""
    return {
        f"{stem}.lokr_w1": torch.randn(2, 2, generator=_GEN),
        f"{stem}.lokr_w2_a": torch.randn(out // 2, RANK, generator=_GEN),
        f"{stem}.lokr_w2_b": torch.randn(RANK, DIM // 2, generator=_GEN),
        f"{stem}.alpha": torch.tensor(float(RANK)),
    }


@dataclass(frozen=True)
class _Arch:
    module: str
    cls: str
    session: str
    components: Tuple[str, ...]
    #: ``label -> (keys, detector must call it diffusers_peft, pairs declared)``
    formats: Dict[str, Tuple[Dict[str, torch.Tensor], bool, int]]
    #: ``prepare_file`` consults the LOADED components, so it cannot run here.
    parse_hook_needs_model: bool = False
    #: ``label -> (keys, branches declared)`` for a LyCORIS file, in THIS
    #: architecture's own stem spelling. Deliberately NOT in ``formats``: these
    #: are not supported formats, and the only contract they carry is the one
    #: below -- the counter must SEE them, so the session refuses the file as
    #: unapplied instead of declaring 0 branches and passing.
    variants: Dict[str, Tuple[Dict[str, torch.Tensor], int]] = field(
        default_factory=dict)
    #: ``(one whole pair + one half pair, branches declared)``, again in this
    #: architecture's own spelling. Declared must be 2: the half pair is a
    #: target the file names and no builder can apply, so counting it is what
    #: makes ``applied < declared`` refuse a truncated checkpoint instead of
    #: generating from it with one target silently missing.
    truncated: Optional[Tuple[Dict[str, torch.Tensor], int]] = None


# Every backend that builds an AdapterSession, with the key formats its own
# parser accepts. A new architecture joins this table (the completeness gate
# below fails otherwise) and states which spellings it reads.
ARCHES: Dict[str, _Arch] = {
    "acestep": _Arch(
        "core.pipeline_backends.acestep", "AceStepMixin", "_acestep_lora_session",
        ("dit",),
        {
            "sd-scripts": (_canonical("lora_unet_decoder_layers_0_self_attn_q_proj"), False, 1),
            "diffusers": (_peft("transformer_blocks.0.attn.to_q"), True, 1),
        },
        truncated=({**_canonical("lora_unet_decoder_layers_0_self_attn_q_proj"),
                    **_half("lora_unet_decoder_layers_0_self_attn_k_proj")}, 2),
        variants={
            "loha": (_loha("lora_unet_decoder_layers_0_self_attn_q_proj"), 1),
            "lokr": (_lokr("lora_unet_decoder_layers_0_self_attn_q_proj"), 1),
        },
    ),
    "anima": _Arch(
        "core.pipeline_backends.anima", "AnimaMixin", "_anima_lora_session",
        ("transformer",),
        {
            "sd-scripts": (_canonical("lora_unet_blocks_0_self_attn_q_proj"), False, 1),
            "interchange": (_peft("diffusion_model.blocks.0.self_attn.q_proj"), True, 1),
        },
        truncated=({**_canonical("lora_unet_blocks_0_self_attn_q_proj"),
                    **_half("lora_unet_blocks_0_self_attn_k_proj")}, 2),
        variants={
            "loha": (_loha("lora_unet_blocks_0_self_attn_q_proj"), 1),
            "lokr": (_lokr("lora_unet_blocks_0_self_attn_q_proj"), 1),
        },
    ),
    "flux2": _Arch(
        "core.pipeline_backends.flux2", "Flux2Mixin", "_flux2_lora_session",
        ("transformer", "text_encoder"),
        {
            "sd-scripts": (_canonical("lora_transformer_transformer_blocks_0_attn_to_q"), False, 1),
        },
        parse_hook_needs_model=True,
        truncated=({**_canonical("lora_transformer_transformer_blocks_0_attn_to_q"),
                    **_half("lora_transformer_transformer_blocks_0_attn_to_k")}, 2),
        variants={
            "loha": (_loha("lora_transformer_transformer_blocks_0_attn_to_q"), 1),
            "lokr": (_lokr("lora_transformer_transformer_blocks_0_attn_to_q"), 1),
        },
    ),
    "ideogram4": _Arch(
        "core.pipeline_backends.ideogram4", "Ideogram4Mixin", "_ideogram4_lora_session",
        ("transformer",),
        {
            "sd-scripts": (_canonical("lora_unet_layers_0_attention_to_q"), False, 1),
            "interchange": (_peft("diffusion_model.layers.0.attention.to_q"), True, 1),
        },
        truncated=({**_canonical("lora_unet_layers_0_attention_to_q"),
                    **_half("lora_unet_layers_0_attention_to_k")}, 2),
        variants={
            "loha": (_loha("lora_unet_layers_0_attention_to_q"), 1),
            "lokr": (_lokr("lora_unet_layers_0_attention_to_q"), 1),
        },
    ),
    "krea2": _Arch(
        "core.pipeline_backends.krea2", "Krea2Mixin", "_krea2_lora_session",
        ("transformer",),
        {
            "sd-scripts": (_canonical("lora_unet_transformer_blocks__0__attn__to_q"), False, 1),
        },
        truncated=({**_canonical("lora_unet_transformer_blocks__0__attn__to_q"),
                    **_half("lora_unet_transformer_blocks__0__attn__to_k")}, 2),
        variants={
            "loha": (_loha("lora_unet_transformer_blocks__0__attn__to_q"), 1),
            "lokr": (_lokr("lora_unet_transformer_blocks__0__attn__to_q"), 1),
        },
    ),
    "lens": _Arch(
        "core.pipeline_backends.lens", "LensMixin", "_lens_lora_session",
        ("transformer",),
        {
            "sd-scripts": (_canonical("lora_unet_transformer_blocks_0_attn_img_qkv"), False, 1),
            "interchange": (_peft("diffusion_model.transformer_blocks.0.attn.img_qkv"), True, 1),
        },
        truncated=({**_canonical("lora_unet_transformer_blocks_0_attn_img_qkv"),
                    **_half("lora_unet_transformer_blocks_0_attn_txt_qkv")}, 2),
        variants={
            "loha": (_loha("lora_unet_transformer_blocks_0_attn_img_qkv"), 1),
            "lokr": (_lokr("lora_unet_transformer_blocks_0_attn_img_qkv"), 1),
        },
    ),
    "ltx2": _Arch(
        "core.pipeline_backends.ltx2", "LTX2Mixin", "_ltx2_lora_session",
        ("transformer",),
        {
            "sd-scripts": (_canonical("lora_unet_transformer_blocks_0_attn1_to_q"), False, 1),
            "interchange": (_peft("diffusion_model.transformer_blocks.0.attn1.to_q"), True, 1),
        },
        truncated=({**_canonical("lora_unet_transformer_blocks_0_attn1_to_q"),
                    **_half("lora_unet_transformer_blocks_0_attn1_to_k")}, 2),
        variants={
            "loha": (_loha("lora_unet_transformer_blocks_0_attn1_to_q"), 1),
            "lokr": (_lokr("lora_unet_transformer_blocks_0_attn1_to_q"), 1),
        },
    ),
    "minimax_h3": _Arch(
        "core.pipeline_backends.minimax_h3", "MiniMaxH3Mixin", "_minimax_h3_lora_session",
        ("transformer",),
        {
            "sd-scripts": (_canonical("lora_unet_transformer_blocks_0_attn_to_q"), False, 1),
            # Fused qkv: the up half carries all three thirds, so it is 3*DIM rows.
            "comfy": (_peft("diffusion_model.blocks.0.attn.qkv_proj", out=3 * DIM),
                      True, 3),  # one fused stem -> to_q/to_k/to_v
        },
        truncated=({**_canonical("diffusion_model.blocks.0.attn.to_q"),
                    **_half("diffusion_model.blocks.0.attn.to_k")}, 2),
        variants={
            "loha": (_loha("diffusion_model.blocks.0.attn.qkv_proj", out=3 * DIM), 3),
            "lokr": (_lokr("diffusion_model.blocks.0.attn.qkv_proj", out=3 * DIM), 3),
        },
    ),
    "minit2i": _Arch(
        # Its counter answers "pairs THIS PASS could apply", so a transformer
        # fixture must be asked about the transformer pass.
        "core.pipeline_backends.minit2i", "MiniT2IMixin", "_minit2i_lora_session",
        ("transformer",),
        {
            "sd-scripts": (_canonical("lora_unet_blocks__0__attn__to_q"), False, 1),
        },
        parse_hook_needs_model=True,
        truncated=({**_canonical("lora_unet_blocks__0__attn__to_q"),
                    **_half("lora_unet_blocks__0__attn__to_k")}, 2),
        variants={
            "loha": (_loha("lora_unet_blocks__0__attn__to_q"), 1),
            "lokr": (_lokr("lora_unet_blocks__0__attn__to_q"), 1),
        },
    ),
    "sensenova": _Arch(
        "core.pipeline_backends.sensenova", "SenseNovaMixin", "_sensenova_lora_session",
        ("transformer",),
        {
            "verbatim": (_canonical(
                "language_model.model.layers.0.self_attn.q_proj_mot_gen"), False, 1),
            # The only arch that ASKS for canonicalization: its parser matches the
            # suffix on a verbatim module path, which is what PEFT keys become.
            "peft": (_peft(
                "base_model.model.language_model.model.layers.0.self_attn.q_proj_mot_gen"),
                True, 1),
        },
        truncated=({**_canonical("language_model.model.layers.0.self_attn.q_proj_mot_gen"),
                    **_half("language_model.model.layers.0.self_attn.k_proj_mot_gen")}, 2),
        variants={
            "loha": (_loha("language_model.model.layers.0.self_attn.q_proj_mot_gen"), 1),
            "lokr": (_lokr("language_model.model.layers.0.self_attn.q_proj_mot_gen"), 1),
        },
    ),
    "zimage": _Arch(
        "core.pipeline_backends.zimage", "ZImageMixin", "_zimage_lora_session",
        ("transformer",),
        {
            "sd-scripts": (_canonical("lora_unet_layers_0_attn_to_q"), False, 1),
        },
        truncated=({**_canonical("lora_transformer_layers_0_attn_to_q"),
                    **_half("lora_transformer_layers_0_attn_to_k")}, 2),
        variants={
            "loha": (_loha("lora_transformer_layers_0_attn_to_q"), 1),
            "lokr": (_lokr("lora_transformer_layers_0_attn_to_q"), 1),
        },
    ),
}

CASES = [(arch, label) for arch, spec in ARCHES.items() for label in spec.formats]
VARIANT_CASES = [(arch, label)
                 for arch, spec in ARCHES.items() for label in spec.variants]


def _session(arch: str):
    """The REAL session that backend builds.

    ``object.__new__``: every backend's session is a lazy property over bound
    methods only, so it needs no constructed pipeline and loads no model.
    """
    spec = ARCHES[arch]
    mixin = getattr(import_module(spec.module), spec.cls)
    return getattr(object.__new__(mixin), spec.session)


@pytest.mark.parametrize("arch,label", CASES)
def test_every_declared_key_format_survives_the_session(arch, label):
    """The headline contract: what the session hands over still parses."""
    spec = ARCHES[arch]
    session = _session(arch)
    raw, _is_peft, expected = spec.formats[label]

    handed, _codec = session._canonicalize(raw, {})
    counted = session._count_declared_branches(handed, spec.components)

    # Equality, not `> 0`: `_account` refuses with `lora_partial` whenever
    # applied < declared, so an under-count is the same 400 one level along.
    assert counted == expected, (
        f"{arch}/{label}: the session hands over {sorted(handed)[:4]}, which "
        f"{arch}'s own declared-branch counter reads as {counted} pair(s), not "
        f"{expected} -- a valid file would be refused (0 counted) or warned "
        f"partial (under-counted)."
    )


@pytest.mark.parametrize("arch,label", VARIANT_CASES)
def test_a_lycoris_file_is_counted_by_the_same_declared_branch_counter(arch, label):
    """Every counter reads a LyCORIS factor set as the branches it carries.

    A LoHa/LoKr file has ZERO ``.lora_down.weight`` keys, so a counter that
    tallies those reads it as 0 branches. This pins that none of the eleven does
    -- the fused ``qkv_proj`` stem included, which is 3 for LyCORIS exactly as
    it is for LoRA.

    On the four architectures whose capability row now carries LoHa and LoKr
    (``core/adapters/capability.py``) this IS the live count. On the other
    seven ``_refuse_unsupported_algebra`` fires first, in ``_parse``, and these
    rows pin that the count does not silently rot behind that refusal -- the
    number the session would compare ``applied`` against when the matrix opens.
    """
    spec = ARCHES[arch]
    session = _session(arch)
    raw, expected = spec.variants[label]

    # Non-vacuity: with a pair key present, a pre-migration key tally would
    # already have counted the file and this row would pin nothing.
    assert not [k for k in raw
                if k.endswith((".lora_down.weight", ".lora_A.weight"))], (
        f"{arch}/{label}: fixture is not a LyCORIS-only file")

    handed, _codec = session._canonicalize(raw, {})
    counted = session._count_declared_branches(handed, spec.components)

    assert counted == expected, (
        f"{arch}/{label}: {arch}'s declared-branch counter reads "
        f"{sorted(handed)[:4]} as {counted} branch(es), not {expected}; at 0 "
        f"the session would apply nothing and refuse nothing."
    )


@pytest.mark.parametrize("arch", sorted(a for a in ARCHES if ARCHES[a].truncated))
def test_a_truncated_file_declares_the_half_pair_it_cannot_apply(arch):
    """One whole pair plus one half pair must declare TWO.

    At 1 the file applies its one good pair and generates, with the truncated
    target silently missing -- subtly wrong images and no signal. At 2 the
    half pair is unapplied, ``applied < declared_branches`` fires, and the user
    is told. The counter is the ONLY place that can see the difference: by the
    time a builder is asked about the half pair's module there is nothing to
    return but ``None``, which is indistinguishable from "this file does not
    cover that target".
    """
    spec = ARCHES[arch]
    session = _session(arch)
    raw, expected = spec.truncated

    handed, _codec = session._canonicalize(raw, {})
    counted = session._count_declared_branches(handed, spec.components)

    assert counted == expected, (
        f"{arch}: a file with one whole pair and one half pair declares "
        f"{counted}, not {expected} -- at {counted} the truncation is invisible."
    )


def test_every_architecture_declares_a_loha_and_a_lokr_fixture():
    """The counter is the one surface every architecture shares, so no
    architecture may sit out the LyCORIS row."""
    missing = {arch: sorted({"loha", "lokr"} - set(spec.variants))
               for arch, spec in ARCHES.items()
               if {"loha", "lokr"} - set(spec.variants)}
    assert not missing, f"architectures with no LyCORIS fixture: {missing}"
    no_truncated = sorted(a for a, spec in ARCHES.items() if spec.truncated is None)
    assert not no_truncated, f"architectures with no truncated fixture: {no_truncated}"


@pytest.mark.parametrize("arch,label", CASES)
def test_the_architecture_parses_what_the_session_hands_it(arch, label):
    """The counter is not the only surface: ACE-Step's counter falls back to a
    ``.lora_down.weight`` tally that a rewritten file satisfies, while its
    ``prepare_file`` refuses the same file outright. Both must hold.

    Vacuous for two of the eleven and honest about it: zimage has no parse hook
    at all, and SenseNova's has no raise path. Neither hides a gap -- each
    declares only formats the counter above already covers.
    """
    spec = ARCHES[arch]
    session = _session(arch)
    raw, _is_peft, _expected = spec.formats[label]

    handed, codec = session._canonicalize(raw, {})
    if session._prepare_file is None:
        pytest.skip(f"{arch} has no per-file parse hook")
    if spec.parse_hook_needs_model:
        pytest.skip(f"{arch}'s parse hook reads the loaded components")
    file = AdapterFile(
        index=0, name=f"{arch}-{label}.safetensors", path="", strength=1.0,
        config={}, tensors=handed, metadata={}, branch_name=f"0:{arch}",
        declared_branches=0, codec=codec,
    )
    session._prepare_file(file)  # a refusal here is the 400 the user sees


@pytest.mark.parametrize("arch,label", CASES)
def test_a_peft_fixture_is_the_shape_the_codec_would_rewrite(arch, label):
    """Without this the gate above could pass on fixtures the rewrite never
    touches, and would pin nothing."""
    spec = ARCHES[arch]
    raw, is_peft, _expected = spec.formats[label]
    detected = CodecRegistry.detect(raw, {}).format
    if is_peft:
        assert detected == FORMAT_PEFT, f"{arch}/{label}: detected {detected}"
    else:
        assert detected != FORMAT_PEFT, f"{arch}/{label}: detected {detected}"


@pytest.mark.parametrize("arch", sorted(ARCHES))
def test_a_session_that_did_not_ask_for_canonical_keys_gets_the_file_verbatim(arch):
    """Opt-in means opt-in: no key may change under an architecture that parses
    the foreign spelling itself."""
    spec = ARCHES[arch]
    session = _session(arch)
    if session._canonicalize_foreign_keys:
        pytest.skip(f"{arch} asked for canonical stems")
    for label, (raw, _is_peft, _expected) in spec.formats.items():
        handed, _codec = session._canonicalize(raw, {})
        assert sorted(handed) == sorted(raw), f"{arch}/{label}: keys were rewritten"


@pytest.mark.parametrize("arch", sorted(ARCHES))
def test_a_file_with_bias_tensors_does_not_take_the_load_down(arch, monkeypatch):
    """A `lora_bias=True` PEFT export carries a 1-D ``.lora_A.bias``, which the
    codec's rank extraction used to index as 2-D (``IndexError``). ``_parse``
    runs detection outside every try/except the load has AND now refuses on its
    verdict, so a failed sniff would replace the architecture's own 400 with a
    raw 500.

    ``_sniff_rank`` skips a tensor too small for its axis, so this file is now
    read correctly rather than merely survived; the fallback is still the
    contract, so it is exercised below with detection forced to raise."""
    session = _session(arch)
    stem = "transformer_blocks.0.attn.to_q"
    keys = dict(_peft(stem))
    keys[f"{stem}.lora_A.bias"] = torch.randn(RANK, generator=_GEN)
    # `safe_open().keys()` comes back SORTED, and `.lora_A.bias` sorts before
    # `.lora_A.weight` -- which is why the 1-D tensor is the one detect reaches.
    biased = {k: keys[k] for k in sorted(keys)}

    codec = CodecRegistry.detect(biased, {})  # the raw call this fixture is for
    assert codec.format == FORMAT_PEFT and codec.rank == RANK

    handed, codec = session._canonicalize(biased, {})
    assert len(handed) == len(biased), "a failed sniff must not drop tensors"

    def _explode(*_a, **_k):
        raise IndexError("simulated sniff failure")

    monkeypatch.setattr(CodecRegistry, "detect", staticmethod(_explode))
    handed, codec = session._canonicalize(biased, {})
    assert codec.format == FORMAT_UNKNOWN
    assert len(handed) == len(biased), "a failed sniff must not drop tensors"


def test_every_backend_that_builds_an_adapter_session_is_covered_here():
    """The gate that catches the NEXT consumer: a new architecture cannot get a
    session without declaring the key formats it reads.

    Blind spot: it globs ``pipeline_backends/`` only, so an architecture that
    built its session under ``core/models/<arch>/`` would be invisible here.
    None does today.
    """
    backends_dir = Path(__file__).resolve().parents[1] / "core" / "pipeline_backends"
    building = {
        f"core.pipeline_backends.{path.stem}"
        for path in backends_dir.glob("*.py")
        if "AdapterSession(" in path.read_text(encoding="utf-8")
    }
    assert building == {spec.module for spec in ARCHES.values()}
