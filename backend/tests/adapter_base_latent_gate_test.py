"""The D10 latent-space gate: which adapters a swapped-VAE base accepts.

WHY THIS FILE EXISTS. On SD1.5/SDXL every LoRA-targetable module is a `Linear`
inside a `Transformer2DModel`; the only modules a VAE swap resizes are
`conv_in`/`conv_out`, which are `Conv2d` and structurally unreachable by the
adapter target scan. So a LoRA trained against the standard VAE applies to a
16-channel checkpoint at 100% with ZERO shape mismatches and silently
contributes deltas learned in a different latent space. Nothing shape-based can
see it; this gate is the only thing that can.

Four outcomes, no fifth (`docs/guides/VAE_SWAP_MIGRATION_DESIGN.md` D10):
  * channel mismatch                      -> refuse `lora_incompatible`
  * same channels, different VAE           -> warn `lora_base_vae_mismatch`
  * no identity, `struct_native="0"` base  -> refuse `lora_incompatible`
  * no identity, `struct_native="1"` and
    `identity_native="0"` base             -> warn `lora_base_vae_unknown`
  (no identity, fully native base          -> unchanged, no check)

Each is exercised on BOTH read paths: `core.extensions.lora_manager` (the two
diffusers architectures) and `core.adapters.AdapterSession` (the other eleven).

No model loads, no CUDA, synthetic safetensors only. Run with:
    venv/Scripts/python.exe -m pytest backend/tests/adapter_base_latent_gate_test.py -v
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_BACKEND = os.path.join(_REPO, "backend")
for _p in (_REPO, _BACKEND):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.adapters import AdapterIncompatible, AdapterSession  # noqa: E402
from core.adapters.base_identity import (  # noqa: E402
    METADATA_BASE_LATENT_CHANNELS, METADATA_BASE_VAE_HASH,
    METADATA_BASE_VAE_IDENTITY_NATIVE, METADATA_BASE_VAE_STRUCT_NATIVE,
    METADATA_BASE_VAE_TYPE, REFUSAL_CODE, WARNING_CODE_MISMATCH,
    WARNING_CODE_UNKNOWN, BaseLatentIdentity, check_base_latent)
from core.adapters.spec import AdapterSpec  # noqa: E402
from core.extensions.lora_manager import LoRAManager  # noqa: E402

R, D = 4, 8

NATIVE = BaseLatentIdentity(latent_channels=4)
SWAPPED_16 = BaseLatentIdentity(latent_channels=16, vae_type="flux1",
                                vae_hash="aaaaaaaaaaaaaaaa",
                                struct_native=False, identity_native=False)
#: Same channel layout, different weights -- a fine-tuned SDXL VAE.
RESTRUCK_4 = BaseLatentIdentity(latent_channels=4, vae_type="sdxl",
                                vae_hash="bbbbbbbbbbbbbbbb",
                                struct_native=True, identity_native=False)


def _lora_tensors(stem: str = "lora_unet_a"):
    return {
        f"{stem}.lora_down.weight": torch.zeros(R, D),
        f"{stem}.lora_up.weight": torch.zeros(D, R),
        f"{stem}.alpha": torch.tensor(float(R)),
    }


def _write(path, identity=None, metadata=None):
    meta = dict(metadata or {})
    if identity is not None:
        meta.update(identity.to_metadata())
    save_file(_lora_tensors(), str(path), metadata=meta or None)
    return path


# ---------------------------------------------------------------------------
# The table itself
# ---------------------------------------------------------------------------

def test_channel_mismatch_refuses():
    verdict = check_base_latent(NATIVE, SWAPPED_16, name="x.safetensors")
    assert (verdict.refuse, verdict.code) == (True, REFUSAL_CODE)
    assert "4" in verdict.message and "16" in verdict.message


def test_same_channels_different_vae_warns():
    trained = BaseLatentIdentity(latent_channels=4, vae_type="sdxl",
                                 vae_hash="cccccccccccccccc",
                                 struct_native=True, identity_native=False)
    verdict = check_base_latent(trained, RESTRUCK_4, name="x.safetensors")
    assert (verdict.refuse, verdict.code) == (False, WARNING_CODE_MISMATCH)


def test_native_adapter_on_a_same_shape_replacement_vae_warns():
    """The hash is absent on a natively trained adapter, so the two
    `identity_native` flags are what disagree."""
    verdict = check_base_latent(NATIVE, RESTRUCK_4, name="x.safetensors")
    assert (verdict.refuse, verdict.code) == (False, WARNING_CODE_MISMATCH)


def test_no_identity_on_a_structurally_swapped_base_refuses():
    verdict = check_base_latent(None, SWAPPED_16, name="x.safetensors")
    assert (verdict.refuse, verdict.code) == (True, REFUSAL_CODE)


def test_no_identity_on_a_same_shape_replacement_vae_warns_not_refuses():
    """Refusing here would reject every ordinary SDXL LoRA on any
    fine-tuned-VAE base; the channel layout is identical."""
    verdict = check_base_latent(None, RESTRUCK_4, name="x.safetensors")
    assert (verdict.refuse, verdict.code) == (False, WARNING_CODE_UNKNOWN)


def test_no_identity_on_a_native_base_is_not_checked():
    assert check_base_latent(None, NATIVE, name="x.safetensors").ok
    assert check_base_latent(None, None, name="x.safetensors").ok


def test_matching_identity_passes():
    assert check_base_latent(SWAPPED_16, SWAPPED_16, name="x").ok
    assert check_base_latent(NATIVE, NATIVE, name="x").ok


def test_an_unknown_channel_count_on_either_side_does_not_refuse():
    """`latent_channels == 0` is the pixel-space architectures' wiring
    constant; reading it as "0 channels" would refuse every MiniT2I adapter."""
    unknown = BaseLatentIdentity(latent_channels=None)
    assert check_base_latent(unknown, SWAPPED_16.__class__(
        latent_channels=16, struct_native=True, identity_native=True)).ok
    assert BaseLatentIdentity.from_facts({"latent_channels": 0}).latent_channels is None


# ---------------------------------------------------------------------------
# Write side: metadata round-trip
# ---------------------------------------------------------------------------

def test_metadata_round_trips_through_adapter_spec():
    meta = SWAPPED_16.to_metadata()
    assert meta[METADATA_BASE_LATENT_CHANNELS] == "16"
    assert meta[METADATA_BASE_VAE_TYPE] == "flux1"
    assert meta[METADATA_BASE_VAE_HASH] == "aaaaaaaaaaaaaaaa"
    assert meta[METADATA_BASE_VAE_STRUCT_NATIVE] == "0"
    assert meta[METADATA_BASE_VAE_IDENTITY_NATIVE] == "0"
    assert BaseLatentIdentity.from_metadata(meta) == SWAPPED_16
    assert BaseLatentIdentity.from_metadata({"lora_rank": "4"}) is None

    spec = AdapterSpec.from_metadata({"lora_rank": "4", "lora_alpha": "4",
                                      **meta})
    assert spec.base_latent_identity() == SWAPPED_16
    assert spec.to_metadata()[METADATA_BASE_VAE_HASH] == "aaaaaaaaaaaaaaaa"
    # The identity travels as flat keys, never doubled into the options JSON.
    assert "base_latent" not in spec.to_metadata().get("sushi.adapter.options", "")


def test_the_trainer_side_writes_the_runs_latent_space():
    from core.training.adapters.base_adapter import base_latent_metadata

    native = SimpleNamespace(wiring=None, arch=SimpleNamespace(
        wiring=SimpleNamespace(latent_channels=4)), vae_identity=None)
    assert base_latent_metadata(native) == {
        METADATA_BASE_LATENT_CHANNELS: "4",
        METADATA_BASE_VAE_STRUCT_NATIVE: "1",
        METADATA_BASE_VAE_IDENTITY_NATIVE: "1",
    }

    swapped = SimpleNamespace(
        wiring=SimpleNamespace(latent_channels=16),
        arch=SimpleNamespace(wiring=SimpleNamespace(latent_channels=4)),
        vae_identity=SimpleNamespace(latent_channels=16, family="flux1",
                                     content_hash="aaaaaaaaaaaaaaaa",
                                     struct_native=False,
                                     identity_native=False))
    assert BaseLatentIdentity.from_metadata(
        base_latent_metadata(swapped)) == SWAPPED_16

    # A pixel-space architecture states nothing rather than "0 channels".
    pixel = SimpleNamespace(wiring=None, arch=SimpleNamespace(
        wiring=SimpleNamespace(latent_channels=0)), vae_identity=None)
    assert base_latent_metadata(pixel) == {}


# ---------------------------------------------------------------------------
# Read path 1: core.extensions.lora_manager (sd15 / sdxl, through diffusers)
# ---------------------------------------------------------------------------

def _manager(tmp_path):
    manager = LoRAManager(lora_dir=str(tmp_path))
    manager.seeded_dirs = []
    return manager


def _pipeline(identity=None, latent_channels=4):
    """A stand-in for the loaded diffusers pipeline: the loader attaches the
    resolved `component.vae.*` block as `_sushi_vae_identity`, and a native
    checkpoint carries none."""
    facts = None
    if identity is not None:
        facts = {"latent_channels": identity.latent_channels,
                 "family": identity.vae_type,
                 "content_hash": identity.vae_hash,
                 "struct_native": identity.struct_native,
                 "identity_native": identity.identity_native}
    return SimpleNamespace(
        _sushi_vae_identity=facts,
        vae=SimpleNamespace(config=SimpleNamespace(
            latent_channels=latent_channels)))


def _gate(manager, monkeypatch, pipeline, path):
    warnings = []
    import core.extensions.lora_manager as lm
    monkeypatch.setattr(lm, "_lora_warn",
                        lambda message, code: warnings.append(code))
    manager._check_base_latent(pipeline, path, os.path.basename(str(path)))
    return warnings


def test_manager_refuses_an_unmarked_lora_on_a_swapped_base(tmp_path, monkeypatch):
    path = _write(tmp_path / "third_party.safetensors")
    with pytest.raises(RuntimeError) as excinfo:
        _gate(_manager(tmp_path), monkeypatch, _pipeline(SWAPPED_16), path)
    assert getattr(excinfo.value, "code", None) == REFUSAL_CODE


def test_manager_refuses_a_four_channel_lora_on_a_swapped_base(tmp_path, monkeypatch):
    path = _write(tmp_path / "sdxl_native.safetensors", NATIVE)
    with pytest.raises(RuntimeError) as excinfo:
        _gate(_manager(tmp_path), monkeypatch, _pipeline(SWAPPED_16), path)
    assert getattr(excinfo.value, "code", None) == REFUSAL_CODE
    assert "16" in str(excinfo.value)


def test_manager_accepts_a_lora_trained_on_the_same_swapped_base(tmp_path, monkeypatch):
    path = _write(tmp_path / "swapped.safetensors", SWAPPED_16)
    assert _gate(_manager(tmp_path), monkeypatch,
                 _pipeline(SWAPPED_16), path) == []


def test_manager_warns_on_the_same_channels_with_another_vae(tmp_path, monkeypatch):
    trained = BaseLatentIdentity(latent_channels=4, vae_type="sdxl",
                                 vae_hash="cccccccccccccccc",
                                 struct_native=True, identity_native=False)
    path = _write(tmp_path / "other_vae.safetensors", trained)
    assert _gate(_manager(tmp_path), monkeypatch,
                 _pipeline(RESTRUCK_4), path) == [WARNING_CODE_MISMATCH]


def test_manager_warns_rather_than_refuses_an_unmarked_lora_on_a_refit_vae(
        tmp_path, monkeypatch):
    path = _write(tmp_path / "third_party.safetensors")
    assert _gate(_manager(tmp_path), monkeypatch,
                 _pipeline(RESTRUCK_4), path) == [WARNING_CODE_UNKNOWN]


def test_manager_leaves_a_native_base_alone(tmp_path, monkeypatch):
    path = _write(tmp_path / "third_party.safetensors")
    assert _gate(_manager(tmp_path), monkeypatch, _pipeline(None), path) == []


def test_a_native_pipeline_reports_its_vaes_channel_count():
    from core.extensions.lora_manager import pipeline_latent_identity

    identity = pipeline_latent_identity(_pipeline(None, latent_channels=4))
    assert (identity.latent_channels, identity.struct_native,
            identity.identity_native) == (4, True, True)


def test_the_img2img_and_inpaint_variants_inherit_the_identity():
    """They are new objects built from `base_pipeline.components`, so without
    the copy a swapped model reads as native on two of the three routes."""
    from core.pipeline import DiffusionPipelineManager

    facts = {"latent_channels": 16, "struct_native": False,
             "identity_native": False}
    base = SimpleNamespace(_sushi_vae_identity=facts)
    img2img, inpaint = SimpleNamespace(), SimpleNamespace()
    DiffusionPipelineManager._carry_latent_identity(base, img2img, inpaint)
    assert img2img._sushi_vae_identity is facts
    assert inpaint._sushi_vae_identity is facts


def test_a_saved_lora_carries_the_block_the_gate_reads(tmp_path):
    """The join point: one dict merge in `BaseLoRAAdapter.save_checkpoint`."""
    import torch.nn as nn

    from core.training.adapters.base_adapter import (BaseLoRAAdapter,
                                                     LORA_COMPONENT_UNET)

    class _Adapter(BaseLoRAAdapter):
        def apply_lora_to_unet(self, lora_layers):
            layer = self.build_branch(self.trainer.transformer.to_q, "to_q")
            self.trainer.transformer.to_q = layer
            self.register_lora_layer(lora_layers, "to_q", layer,
                                     LORA_COMPONENT_UNET)
            return 1

        def apply_lora_to_text_encoders(self, lora_layers):
            return 0

        def setup_trainable_parameters(self, lora_layers):
            return self.component_param_groups(
                lora_layers, {LORA_COMPONENT_UNET: lambda: 1e-4})

        def checkpoint_metadata(self, lora_layers, step, epoch):
            return {"model_type": "sdxl", "step": str(step)}

    model = nn.Module()
    model.to_q = nn.Linear(D, D, bias=False)
    trainer = SimpleNamespace(
        transformer=model, config={}, adapter_algorithm="lora",
        weight_decompose=False, adapter_config={},
        wiring=SimpleNamespace(latent_channels=16),
        arch=SimpleNamespace(wiring=SimpleNamespace(latent_channels=4),
                             adapter_capability=None),
        vae_identity=SimpleNamespace(latent_channels=16, family="flux1",
                                     content_hash="aaaaaaaaaaaaaaaa",
                                     struct_native=False,
                                     identity_native=False))
    adapter = _Adapter(trainer, R, R, torch.float32)
    layers = {}
    adapter.apply_lora_to_unet(layers)
    out = tmp_path / "trained.safetensors"
    adapter.save_checkpoint(layers, step=10, epoch=1, output_path=out)

    entry = _manager(tmp_path).get_available_loras(force_rescan=True)[0]
    assert entry["base_latent_channels"] == 16
    assert entry["base_vae_hash"] == "aaaaaaaaaaaaaaaa"
    assert entry["base_vae_identity_native"] is False
    # ... and it is admitted by the base it was trained on, refused by another.
    assert check_base_latent(
        BaseLatentIdentity.from_facts(_adapter_facts(entry)), SWAPPED_16).ok
    assert check_base_latent(
        BaseLatentIdentity.from_facts(_adapter_facts(entry)), NATIVE).refuse


def _adapter_facts(entry):
    from core.extensions.lora_manager import adapter_latent_facts

    return adapter_latent_facts(entry)


def test_the_listing_reports_the_declared_identity(tmp_path):
    _write(tmp_path / "swapped.safetensors", SWAPPED_16)
    _write(tmp_path / "plain.safetensors")
    entries = {e["name"]: e for e in
               _manager(tmp_path).get_available_loras(force_rescan=True)}
    assert entries["swapped.safetensors"]["base_latent_channels"] == 16
    assert entries["swapped.safetensors"]["base_vae_type"] == "flux1"
    assert entries["swapped.safetensors"]["base_vae_struct_native"] is False
    assert entries["swapped.safetensors"]["base_vae_identity_native"] is False
    assert entries["plain.safetensors"]["base_latent_channels"] is None
    assert entries["plain.safetensors"]["base_vae_identity_native"] is None


# ---------------------------------------------------------------------------
# Read path 2: AdapterSession (the other eleven architectures)
# ---------------------------------------------------------------------------

def _session(model_identity, warnings, architecture="zimage"):
    return AdapterSession(
        resolve_path=lambda p: str(p) if os.path.exists(str(p)) else None,
        warn=lambda message, code: warnings.append(code),
        architecture=architecture,
        base_latent=lambda: model_identity,
    )


def _parse(session, path):
    return session._parse(0, {"path": str(path), "strength": 1.0})


def test_session_refuses_an_unmarked_lora_on_a_swapped_base(tmp_path):
    path = _write(tmp_path / "third_party.safetensors")
    warnings = []
    with pytest.raises(AdapterIncompatible) as excinfo:
        _parse(_session(SWAPPED_16, warnings), path)
    assert excinfo.value.code == REFUSAL_CODE
    assert warnings == [REFUSAL_CODE]


def test_session_refuses_a_channel_mismatch(tmp_path):
    path = _write(tmp_path / "sixteen.safetensors", SWAPPED_16)
    warnings = []
    with pytest.raises(AdapterIncompatible) as excinfo:
        _parse(_session(NATIVE, warnings), path)
    assert excinfo.value.code == REFUSAL_CODE


def test_session_warns_on_the_same_channels_with_another_vae(tmp_path):
    trained = BaseLatentIdentity(latent_channels=4, vae_type="sdxl",
                                 vae_hash="cccccccccccccccc",
                                 struct_native=True, identity_native=False)
    path = _write(tmp_path / "other_vae.safetensors", trained)
    warnings = []
    assert _parse(_session(RESTRUCK_4, warnings), path) is not None
    assert warnings == [WARNING_CODE_MISMATCH]


def test_session_warns_rather_than_refuses_an_unmarked_lora_on_a_refit_vae(tmp_path):
    path = _write(tmp_path / "third_party.safetensors")
    warnings = []
    assert _parse(_session(RESTRUCK_4, warnings), path) is not None
    assert warnings == [WARNING_CODE_UNKNOWN]


def test_session_leaves_a_native_base_alone(tmp_path):
    path = _write(tmp_path / "third_party.safetensors")
    warnings = []
    assert _parse(_session(NATIVE, warnings), path) is not None
    assert warnings == []


def test_every_adapter_session_backend_wires_the_gate():
    """The eleven backends look the resolver up with `getattr`, so a rename
    would silently disable the gate on all of them rather than fail. These two
    assertions are what makes that lookup safe."""
    import glob
    import re

    from core.pipeline import DiffusionPipelineManager

    assert callable(DiffusionPipelineManager.base_latent_identity)

    missing = []
    for path in sorted(glob.glob(os.path.join(
            _BACKEND, "core", "pipeline_backends", "*.py"))):
        source = open(path, encoding="utf-8").read()
        for block in re.findall(r"AdapterSession\((.*?)\n\s*\)", source, re.S):
            if "base_latent=" not in block:
                missing.append(os.path.basename(path))
    assert missing == []


def test_session_without_a_base_latent_callback_is_inert(tmp_path):
    """Eleven backends pass one; a caller that does not must not be refused
    for an identity nobody could resolve."""
    path = _write(tmp_path / "sixteen.safetensors", SWAPPED_16)
    session = AdapterSession(
        resolve_path=lambda p: str(p) if os.path.exists(str(p)) else None,
        architecture="zimage")
    assert _parse(session, path) is not None


def test_a_failing_base_latent_callback_does_not_refuse(tmp_path):
    def boom():
        raise RuntimeError("no model")

    path = _write(tmp_path / "third_party.safetensors")
    session = AdapterSession(
        resolve_path=lambda p: str(p) if os.path.exists(str(p)) else None,
        architecture="zimage", base_latent=boom)
    assert _parse(session, path) is not None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
