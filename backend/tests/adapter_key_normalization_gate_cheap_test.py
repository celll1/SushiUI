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

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Dict, Tuple

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
    ),
    "anima": _Arch(
        "core.pipeline_backends.anima", "AnimaMixin", "_anima_lora_session",
        ("transformer",),
        {
            "sd-scripts": (_canonical("lora_unet_blocks_0_self_attn_q_proj"), False, 1),
            "interchange": (_peft("diffusion_model.blocks.0.self_attn.q_proj"), True, 1),
        },
    ),
    "flux2": _Arch(
        "core.pipeline_backends.flux2", "Flux2Mixin", "_flux2_lora_session",
        ("transformer", "text_encoder"),
        {
            "sd-scripts": (_canonical("lora_transformer_transformer_blocks_0_attn_to_q"), False, 1),
        },
        parse_hook_needs_model=True,
    ),
    "ideogram4": _Arch(
        "core.pipeline_backends.ideogram4", "Ideogram4Mixin", "_ideogram4_lora_session",
        ("transformer",),
        {
            "sd-scripts": (_canonical("lora_unet_layers_0_attention_to_q"), False, 1),
            "interchange": (_peft("diffusion_model.layers.0.attention.to_q"), True, 1),
        },
    ),
    "krea2": _Arch(
        "core.pipeline_backends.krea2", "Krea2Mixin", "_krea2_lora_session",
        ("transformer",),
        {
            "sd-scripts": (_canonical("lora_unet_transformer_blocks__0__attn__to_q"), False, 1),
        },
    ),
    "lens": _Arch(
        "core.pipeline_backends.lens", "LensMixin", "_lens_lora_session",
        ("transformer",),
        {
            "sd-scripts": (_canonical("lora_unet_transformer_blocks_0_attn_img_qkv"), False, 1),
            "interchange": (_peft("diffusion_model.transformer_blocks.0.attn.img_qkv"), True, 1),
        },
    ),
    "ltx2": _Arch(
        "core.pipeline_backends.ltx2", "LTX2Mixin", "_ltx2_lora_session",
        ("transformer",),
        {
            "sd-scripts": (_canonical("lora_unet_transformer_blocks_0_attn1_to_q"), False, 1),
            "interchange": (_peft("diffusion_model.transformer_blocks.0.attn1.to_q"), True, 1),
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
    ),
    "zimage": _Arch(
        "core.pipeline_backends.zimage", "ZImageMixin", "_zimage_lora_session",
        ("transformer",),
        {
            "sd-scripts": (_canonical("lora_unet_layers_0_attn_to_q"), False, 1),
        },
    ),
}

CASES = [(arch, label) for arch, spec in ARCHES.items() for label in spec.formats]


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
def test_a_file_the_detector_cannot_read_does_not_take_the_load_down(arch):
    """A `lora_bias=True` PEFT export carries a 1-D ``.lora_A.bias``, which the
    codec's rank extraction indexes as 2-D (``IndexError``). Detection is
    advisory, and ``_parse`` runs it outside every try/except the load has, so
    a failed sniff would replace the architecture's own 400 with a raw 500."""
    session = _session(arch)
    stem = "transformer_blocks.0.attn.to_q"
    keys = dict(_peft(stem))
    keys[f"{stem}.lora_A.bias"] = torch.randn(RANK, generator=_GEN)
    # `safe_open().keys()` comes back SORTED, and `.lora_A.bias` sorts before
    # `.lora_A.weight` -- which is why the 1-D tensor is the one detect reaches.
    biased = {k: keys[k] for k in sorted(keys)}

    with pytest.raises(IndexError):
        CodecRegistry.detect(biased, {})  # the raw call this fixture is here for

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
