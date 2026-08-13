import threading
import sys
import json
import struct
from types import SimpleNamespace
from pathlib import Path

import pytest

_BACKEND = str(Path(__file__).resolve().parents[1])
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from core.model_state_coordinator import ModelStateBusyError, ModelStateCoordinator
from core.models.components.component_catalog import build_catalog, build_response
from core.models.components import component_switcher
from core.models.components.component_switcher import (
    ComponentSwitchError,
    ComponentSwitchFailed,
    switch_component,
)
from core.models.minimax_h3 import loader as h3_loader
from core.models.anima.anima_loader import inspect_anima_component_candidate
from core.vision_encoder import inspect_vision_encoder_candidate


class _Manager:
    def __init__(self, arch="sdxl"):
        self.current_model_info = {
            "type": arch,
            "source": "C:/models/current.safetensors",
            "source_type": "safetensors",
        }
        self.model_revision = 4
        self.component_revision = 7
        self.component_health = "ready"
        self._load_model_lock = threading.Lock()
        self.txt2img_pipeline = SimpleNamespace(
            text_encoder=object(), unet=object(), vae=object()
        ) if arch in ("sd15", "sdxl") else None
        self.vision_encoder = None
        self._vision_encoder_path = None
        self._override_te_path = None
        self._override_vae_path = None
        self.events = []

    def unload_vision_encoder(self):
        self.events.append("unload")
        self.vision_encoder = None
        self._vision_encoder_path = None

    def load_vision_encoder(self, path):
        self.events.append(("load", path))
        self.vision_encoder = object()
        self._vision_encoder_path = path


def test_catalog_keeps_current_and_never_switches_full_checkpoint_as_component():
    manager = _Manager()
    catalog = build_catalog(
        manager,
        models=[{
            "name": "another-sdxl",
            "path": "C:/models/another.safetensors",
            "architecture": "sdxl",
            "observed_components": {"backbone": {"kind": "unet"}},
            "size_gb": 6.5,
        }],
    )
    response = build_response(manager, catalog)
    backbone = next(slot for slot in response["slots"] if slot["slot"] == "backbone")

    assert backbone["candidates"][0]["is_current"] is True
    scanned = next(item for item in backbone["candidates"] if item["display_name"] == "another-sdxl")
    assert scanned["compatibility"] == "compatible"
    assert scanned["switchable"] is False
    assert scanned["load_strategy"] == "unsupported"
    assert "model loader" in scanned["switch_reason"]


def test_vision_encoder_switch_releases_old_before_loading_new():
    manager = _Manager()
    candidate = {
        "candidate_id": "vision_encoder:test",
        "slot": "vision_encoder",
        "compatibility": "compatible",
        "switchable": True,
        "_path": "C:/models/vision.safetensors",
    }

    result = switch_component(manager, "vision_encoder", candidate, 4, 7)

    assert manager.events == ["unload", ("load", "C:/models/vision.safetensors")]
    assert manager.component_revision == 8
    assert result["state"] == "succeeded"


def test_minimax_h3_three_verified_te_variants_switchable_other_slots_disabled():
    manager = _Manager("minimax_h3")
    manager.txt2img_pipeline = None
    manager.minimax_h3_components = {
        "text_encoder": object(), "transformer": object(), "vae": object(), "audio_vae": object(),
        "text_encoder_path": "C:/h3/te_bf16.safetensors",
        "dit_path": "C:/h3/dit.safetensors",
        "vae_path": "C:/h3/vae.safetensors",
        "audio_vae_path": "C:/h3/audio_vae.safetensors",
    }
    h3_candidates = [
        {
            "name": "H3 BF16", "path": "C:/h3/te_bf16.safetensors",
            "compatible": True, "variant": "bf16", "reason": "verified", "size_bytes": 51_000_000_000,
        },
        {
            "name": "H3 INT8 ConvRot", "path": "C:/h3/te_int8.safetensors",
            "compatible": True, "variant": "int8_convrot", "reason": "verified", "size_bytes": 27_000_000_000,
        },
        {
            "name": "H3 NVFP4 AWQ", "path": "C:/h3/te_nvfp4.safetensors",
            "compatible": True, "variant": "nvfp4_awq", "reason": "verified", "size_bytes": 15_000_000_000,
        },
    ]
    catalog = build_catalog(manager, h3_text_encoders=h3_candidates)
    response = build_response(manager, catalog)

    te = next(slot for slot in response["slots"] if slot["slot"] == "text_encoder")
    assert te["switchable"] is True
    assert {item.get("variant") for item in te["candidates"]} == {
        "bf16", "int8_convrot", "nvfp4_awq",
    }
    alternatives = [item for item in te["candidates"] if not item["is_current"]]
    assert alternatives and all(item["compatibility"] == "compatible" for item in alternatives)
    assert all(item["switchable"] is True for item in alternatives)

    for slot_name in ("backbone", "vae", "audio_vae"):
        slot = next(item for item in response["slots"] if item["slot"] == slot_name)
        assert slot["switchable"] is False
        assert "MiniMax-H3" in slot["reason"]
    with pytest.raises(ComponentSwitchError):
        switch_component(manager, "backbone", {"switchable": True, "compatibility": "compatible"}, 4, 7)


def test_h3_converted_encoder_is_offered_only_when_its_projection_resolves():
    manager = _Manager("minimax_h3")
    manager.txt2img_pipeline = None
    manager.minimax_h3_components = {
        "text_encoder": object(), "transformer": object(), "vae": object(), "audio_vae": object(),
        "text_encoder_path": "C:/h3/text_encoders/te_bf16.safetensors",
        "dit_path": "C:/h3/dit.safetensors",
    }
    agreement = {"reference": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
                 "projection": "mmh3-4b-clipproj-celeb-mlp.safetensors", "cosine": 0.826,
                 "rel_rms": 0.214, "rel_rms_floor": 0.048, "presentations": 111}
    catalog = build_catalog(manager, h3_text_encoders=[
        {"name": "te_bf16", "path": "C:/h3/text_encoders/te_bf16.safetensors", "compatible": True,
         "variant": "bf16", "reason": "verified", "size_bytes": 51_000_000_000,
         "requires_projection": False, "projection": None, "projection_reason": None,
         "agreement": None},
        {"name": "qwen3vl_4b", "path": "C:/h3/text_encoders/qwen3vl_4b.safetensors",
         "compatible": True, "variant": "converted_small", "reason": "Converted 4B Qwen3-VL",
         "size_bytes": 5_624_901_632, "requires_projection": True,
         "projection": "C:/h3/clip_projections/mmh3-4b-clipproj-celeb-mlp.safetensors",
         "projection_reason": None, "agreement": agreement},
        {"name": "qwen3vl_8b", "path": "C:/h3/text_encoders/qwen3vl_8b.safetensors",
         "compatible": True, "variant": "converted_small", "reason": "Converted 8B Qwen3-VL",
         "size_bytes": 9_000_000_000, "requires_projection": True, "projection": None,
         "projection_reason": "no file in clip_projections/ declares d_in=4096",
         "agreement": None},
    ])
    rows = {item["display_name"]: item for item in catalog["text_encoder"]}

    paired = rows["qwen3vl_4b"]
    assert paired["switchable"] is True
    assert paired["compatibility"] == "compatible"
    assert paired["requires_projection"] is True
    assert paired["projection"] == "mmh3-4b-clipproj-celeb-mlp.safetensors"
    assert paired["agreement"]["cosine"] == 0.826

    unpaired = rows["qwen3vl_8b"]
    assert unpaired["switchable"] is False
    assert unpaired["compatibility"] == "incompatible"
    assert "d_in=4096" in unpaired["switch_reason"]
    assert unpaired["projection"] is None


_H3_DENSE_SHAPES = {
    "self_attn.q_proj": [8192, 5120],
    "self_attn.k_proj": [1024, 5120],
    "self_attn.v_proj": [1024, 5120],
    "self_attn.o_proj": [5120, 8192],
    "mlp.gate_proj": [25600, 5120],
    "mlp.up_proj": [25600, 5120],
    "mlp.down_proj": [5120, 25600],
}


def _h3_header(variant):
    dtype = {"bf16": "BF16", "int8_convrot": "I8", "nvfp4_awq": "U8"}[variant]
    header = {"model.embed_tokens.weight": {"dtype": "BF16", "shape": [151936, 5120]}}
    for layer in range(50):
        for suffix, original_shape in _H3_DENSE_SHAPES.items():
            stem = f"model.layers.{layer}.{suffix}"
            shape = list(original_shape)
            if variant == "nvfp4_awq":
                shape[1] //= 2
            header[f"{stem}.weight"] = {"dtype": dtype, "shape": shape}
            if variant == "int8_convrot":
                header[f"{stem}.weight_scale"] = {"dtype": "F32", "shape": [shape[0], 1]}
                header[f"{stem}.comfy_quant"] = {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}
            elif variant == "nvfp4_awq":
                header[f"{stem}.weight_scale"] = {
                    "dtype": "F8_E4M3", "shape": [shape[0], shape[1] * 2 // 16],
                }
                header[f"{stem}.weight_scale_2"] = {"dtype": "F32", "shape": []}
                header[f"{stem}.comfy_quant"] = {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}
                if suffix.endswith(("self_attn.o_proj", "mlp.down_proj")):
                    header[f"{stem}.pre_quant_scale"] = {"dtype": "F32", "shape": [shape[1] * 2]}
    if variant == "nvfp4_awq":
        header["model.embed_tokens.weight"] = {"dtype": "I8", "shape": [151936, 5120]}
        header["model.embed_tokens.weight_scale"] = {"dtype": "F32", "shape": [151936, 1]}
        header["model.embed_tokens.comfy_quant"] = {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}
    return header


@pytest.mark.parametrize("variant", ["bf16", "int8_convrot", "nvfp4_awq"])
def test_h3_candidate_geometry_and_marker_contract_not_filename(monkeypatch, tmp_path, variant):
    candidate = tmp_path / f"user-renamed-{variant}.safetensors"
    candidate.write_bytes(b"x")
    header = _h3_header(variant)
    monkeypatch.setattr(h3_loader, "read_safetensors_header", lambda _path: header)
    marker_keys = {key: object() for key in header if key.endswith(".comfy_quant")}
    monkeypatch.setattr(h3_loader, "_read_comfy_quant_markers", lambda _path, _header: marker_keys)
    monkeypatch.setattr(h3_loader, "_supported_int8_convrot_marker", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(h3_loader, "_supported_h3_nvfp4_marker", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(h3_loader, "_supported_h3_int8_embedding_marker", lambda *args, **kwargs: {"ok": True})

    inspected = h3_loader.inspect_minimax_h3_text_encoder_candidate(str(candidate))

    assert inspected["compatible"] is True
    assert inspected["variant"] == variant
    header["model.layers.49.mlp.down_proj.weight"]["shape"] = [1, 1]
    assert h3_loader.inspect_minimax_h3_text_encoder_candidate(str(candidate))["compatible"] is False


def test_h3_candidate_with_unverified_marker_payload_is_disabled(monkeypatch, tmp_path):
    candidate = tmp_path / "shape-only-int8.safetensors"
    candidate.write_bytes(b"x")
    header = _h3_header("int8_convrot")
    monkeypatch.setattr(h3_loader, "read_safetensors_header", lambda _path: header)
    monkeypatch.setattr(
        h3_loader,
        "_read_comfy_quant_markers",
        lambda _path, _header: {key: object() for key in header if key.endswith(".comfy_quant")},
    )
    validator_calls = []

    def reject_marker(key, marker, inspected_header, *, path):
        validator_calls.append((key, marker, inspected_header, path))
        return None

    monkeypatch.setattr(h3_loader, "_supported_int8_convrot_marker", reject_marker)

    inspected = h3_loader.inspect_minimax_h3_text_encoder_candidate(str(candidate))

    assert len(validator_calls) == 1
    assert validator_calls[0][0].endswith(".comfy_quant")
    assert validator_calls[0][2] is header
    assert validator_calls[0][3] == str(candidate)
    assert inspected["compatible"] is False
    assert "marker" in inspected["reason"].lower()


def test_h3_candidate_discovery_uses_resolved_tree_fixture(monkeypatch, tmp_path):
    root = tmp_path / "configured-model-root"
    directory = root / "text_encoders"
    directory.mkdir(parents=True)
    variants = ("bf16", "int8_convrot", "nvfp4_awq")
    for variant in variants:
        (directory / f"user-{variant}.safetensors").write_bytes(b"x")
    monkeypatch.setattr(
        h3_loader,
        "detect_minimax_h3_layout",
        lambda _path: {"root": str(root)},
    )
    monkeypatch.setattr(
        h3_loader,
        "inspect_minimax_h3_text_encoder_candidate",
        lambda path: {
            "path": path,
            "compatible": True,
            "variant": next(variant for variant in variants if variant in Path(path).stem),
            "reason": "verified",
            "size_bytes": Path(path).stat().st_size,
        },
    )

    discovered = h3_loader.list_minimax_h3_text_encoder_candidates(str(root / "model.safetensors"))

    assert {item["variant"] for item in discovered} == set(variants)
    assert all(Path(item["path"]).parent == directory for item in discovered)


def _h3_manager(monkeypatch, **components):
    """An H3 manager whose tree resolution is fixed, with no filesystem behind it."""
    manager = _Manager("minimax_h3")
    manager.txt2img_pipeline = None
    manager.current_model_info["source"] = "C:/h3/diffusion_models/dit.safetensors"
    manager.minimax_h3_components = {
        "text_encoder": object(),
        "text_encoder_config": {},
        "text_encoder_path": "C:/h3/old.safetensors",
        "official_dir": "C:/h3/official",
        "dit_path": "C:/h3/diffusion_models/dit.safetensors",
        **components,
    }
    monkeypatch.setattr(h3_loader, "detect_minimax_h3_layout", lambda _path: {"root": "C:/h3"})
    return manager


def test_h3_te_switch_detaches_before_build_and_keeps_other_components(monkeypatch):
    import core.keep_hot as keep_hot

    shared = {name: object() for name in ("transformer", "vae", "audio_vae", "tokenizer", "processor")}
    manager = _h3_manager(monkeypatch, **shared, text_encoder_config=object())
    events = []
    replacement = object()
    monkeypatch.setattr(keep_hot, "clear_resident", lambda _manager: events.append("clear_keep_hot"))
    monkeypatch.setattr(component_switcher, "_release_device_cache", lambda: events.append("release"))

    def assert_detached():
        assert manager.minimax_h3_components["text_encoder"] is None
        assert manager.minimax_h3_components["text_encoder_path"] is None
        events.append("assert_detached")

    def build(path, official, *, root, dit_path, projection_override=None):
        assert path == "C:/h3/new.safetensors"
        assert official == "C:/h3/official"
        assert (root, dit_path) == ("C:/h3", "C:/h3/diffusion_models/dit.safetensors")
        events.append("build_new")
        return {"text_encoder": replacement, "text_encoder_config": {"variant": "int8_convrot"},
                "te_projection": None, "te_text_only": False}

    monkeypatch.setattr(h3_loader, "assert_no_live_text_encoder", assert_detached)
    monkeypatch.setattr(h3_loader, "build_minimax_h3_text_encoder_bundle", build)
    candidate = {
        "compatibility": "compatible", "switchable": True,
        "_path": "C:/h3/new.safetensors",
    }

    switch_component(manager, "text_encoder", candidate, 4, 7)

    # The first release empties the allocator cache once keep-hot's residents
    # have been offloaded; the second is the adapter proving the old mapping is
    # gone before it maps the new file.
    assert events == ["clear_keep_hot", "release", "release", "assert_detached", "build_new"]
    assert manager.minimax_h3_components["text_encoder"] is replacement
    assert manager.minimax_h3_components["text_encoder_path"] == "C:/h3/new.safetensors"
    assert manager.minimax_h3_components["text_encoder_origin"] == "selected_external"
    assert all(manager.minimax_h3_components[name] is value for name, value in shared.items())
    assert manager.component_revision == 8


def _bundle(projection=None, text_only=False, encoder=None):
    return {
        "text_encoder": encoder if encoder is not None else object(),
        "text_encoder_config": {},
        "te_projection": projection,
        "te_text_only": text_only,
    }


def test_h3_switch_to_converted_encoder_installs_its_projection_and_refreshes_model_info(monkeypatch):
    import core.keep_hot as keep_hot

    manager = _h3_manager(monkeypatch)
    manager.current_model_info.update({
        "text_encoder_file": "old.safetensors", "clip_projection_file": None,
        "te_text_only": False,
    })
    projection = {"path": "C:/h3/clip_projections/mmh3-4b.safetensors",
                  "spec": {"d_in": 2560, "d_out": 5120, "tap": 24}, "tensors": {}}
    monkeypatch.setattr(keep_hot, "clear_resident", lambda _manager: None)
    monkeypatch.setattr(component_switcher, "_release_device_cache", lambda: None)
    monkeypatch.setattr(h3_loader, "assert_no_live_text_encoder", lambda: None)
    monkeypatch.setattr(
        h3_loader, "build_minimax_h3_text_encoder_bundle",
        lambda *args, **kwargs: _bundle(projection=projection, text_only=True))

    switch_component(manager, "text_encoder", {
        "compatibility": "compatible", "switchable": True,
        "_path": "C:/h3/text_encoders/qwen3vl_4b_heretic_tap24_bf16.safetensors",
    }, 4, 7)

    components = manager.minimax_h3_components
    assert components["te_projection"] is projection
    assert components["te_text_only"] is True
    assert manager.current_model_info["text_encoder_file"] == "qwen3vl_4b_heretic_tap24_bf16.safetensors"
    assert manager.current_model_info["clip_projection_file"] == "mmh3-4b.safetensors"
    assert manager.current_model_info["te_text_only"] is True


def test_h3_switch_back_to_a_released_encoder_clears_projection_and_text_only(monkeypatch):
    """Both the emptied slot and the installed one must be projection-free.

    The in-build assertion fails if the slot stops being emptied; the trailing
    assertions fail if the install stops writing the released encoder's own
    (None, False). Either alone would leave the released 32B encoder's
    5120-wide hidden state paired with a 2560-wide projection.
    """
    import core.keep_hot as keep_hot

    converted = {"path": "C:/h3/clip_projections/mmh3-4b.safetensors",
                 "spec": {"d_in": 2560, "d_out": 5120, "tap": 24}, "tensors": {}}
    manager = _h3_manager(
        monkeypatch,
        text_encoder_path="C:/h3/text_encoders/qwen3vl_4b_heretic_tap24_bf16.safetensors",
        te_projection=converted, te_text_only=True,
    )
    manager.current_model_info.update({
        "text_encoder_file": "qwen3vl_4b_heretic_tap24_bf16.safetensors",
        "clip_projection_file": "mmh3-4b.safetensors", "te_text_only": True,
    })
    monkeypatch.setattr(keep_hot, "clear_resident", lambda _manager: None)
    monkeypatch.setattr(component_switcher, "_release_device_cache", lambda: None)
    monkeypatch.setattr(h3_loader, "assert_no_live_text_encoder", lambda: None)

    def build(*_args, **_kwargs):
        assert manager.minimax_h3_components["te_projection"] is None
        assert manager.minimax_h3_components["te_text_only"] is False
        return _bundle()

    monkeypatch.setattr(h3_loader, "build_minimax_h3_text_encoder_bundle", build)

    switch_component(manager, "text_encoder", {
        "compatibility": "compatible", "switchable": True,
        "_path": "C:/h3/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
    }, 4, 7)

    components = manager.minimax_h3_components
    assert components["te_projection"] is None
    assert components["te_text_only"] is False
    assert manager.current_model_info["clip_projection_file"] is None
    assert manager.current_model_info["te_text_only"] is False


def test_h3_switch_failure_restores_the_previous_encoder_and_its_projection_together(monkeypatch):
    import core.keep_hot as keep_hot

    old_projection = {"path": "C:/h3/clip_projections/mmh3-4b.safetensors",
                      "spec": {"d_in": 2560, "d_out": 5120, "tap": 24}, "tensors": {}}
    old_encoder = object()
    manager = _h3_manager(
        monkeypatch,
        text_encoder=old_encoder,
        text_encoder_path="C:/h3/text_encoders/qwen3vl_4b_heretic_tap24_bf16.safetensors",
        te_projection=old_projection, te_text_only=True,
    )
    manager.current_model_info.update({
        "text_encoder_file": "qwen3vl_4b_heretic_tap24_bf16.safetensors",
        "clip_projection_file": "mmh3-4b.safetensors", "te_text_only": True,
    })
    monkeypatch.setattr(keep_hot, "clear_resident", lambda _manager: None)
    monkeypatch.setattr(component_switcher, "_release_device_cache", lambda: None)
    monkeypatch.setattr(h3_loader, "assert_no_live_text_encoder", lambda: None)
    calls = []

    def build(path, _official, *, root, dit_path, projection_override=None):
        calls.append((path, projection_override))
        if "32b" in path.lower():
            raise RuntimeError("replacement failed")
        return _bundle(projection=old_projection, text_only=True)

    monkeypatch.setattr(h3_loader, "build_minimax_h3_text_encoder_bundle", build)

    with pytest.raises(ComponentSwitchFailed):
        switch_component(manager, "text_encoder", {
            "compatibility": "compatible", "switchable": True,
            "_path": "C:/h3/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
        }, 4, 7)

    # The restore names the projection that was running rather than re-deriving
    # it, and reinstalls encoder and projection as one pair.
    assert calls[1] == ("C:/h3/text_encoders/qwen3vl_4b_heretic_tap24_bf16.safetensors",
                        "C:/h3/clip_projections/mmh3-4b.safetensors")
    components = manager.minimax_h3_components
    assert components["te_projection"] is old_projection
    assert components["te_text_only"] is True
    assert components["text_encoder_path"].endswith("qwen3vl_4b_heretic_tap24_bf16.safetensors")
    assert manager.current_model_info["clip_projection_file"] == "mmh3-4b.safetensors"
    assert manager.component_health == "ready"
    assert manager.component_revision == 7


def test_h3_switch_without_a_known_dit_is_refused_before_the_slot_is_emptied(monkeypatch):
    import core.keep_hot as keep_hot

    manager = _h3_manager(monkeypatch, dit_path=None)
    encoder = manager.minimax_h3_components["text_encoder"]
    monkeypatch.setattr(keep_hot, "clear_resident", lambda _manager: None)
    monkeypatch.setattr(h3_loader, "assert_no_live_text_encoder", lambda: None)
    monkeypatch.setattr(
        h3_loader, "build_minimax_h3_text_encoder_bundle",
        lambda *args, **kwargs: pytest.fail("must not build without a DiT to gate against"))

    with pytest.raises(ComponentSwitchError):
        switch_component(manager, "text_encoder", {
            "compatibility": "compatible", "switchable": True,
            "_path": "C:/h3/text_encoders/other.safetensors",
        }, 4, 7)

    assert manager.minimax_h3_components["text_encoder"] is encoder
    assert manager.minimax_h3_components["text_encoder_path"] == "C:/h3/old.safetensors"
    assert manager.component_revision == 7


def test_h3_te_failure_serially_reloads_old_without_revision(monkeypatch):
    import core.keep_hot as keep_hot

    manager = _h3_manager(monkeypatch)
    loads = []
    monkeypatch.setattr(keep_hot, "clear_resident", lambda _manager: None)
    monkeypatch.setattr(component_switcher, "_release_device_cache", lambda: None)
    monkeypatch.setattr(h3_loader, "assert_no_live_text_encoder", lambda: None)

    def build(path, _official, **_kwargs):
        loads.append(path)
        if path.endswith("new.safetensors"):
            raise RuntimeError("new failed")
        return _bundle()

    monkeypatch.setattr(h3_loader, "build_minimax_h3_text_encoder_bundle", build)
    candidate = {
        "compatibility": "compatible", "switchable": True,
        "_path": "C:/h3/new.safetensors",
    }

    with pytest.raises(ComponentSwitchFailed):
        switch_component(manager, "text_encoder", candidate, 4, 7)

    assert loads == ["C:/h3/new.safetensors", "C:/h3/old.safetensors"]
    assert manager.minimax_h3_components["text_encoder_path"] == "C:/h3/old.safetensors"
    assert manager.component_health == "ready"
    assert manager.component_revision == 7


def test_switch_offloads_the_components_it_is_not_replacing():
    """Unload-first has to free the memory, not just forget who held it.

    clear_resident is bookkeeping. The components this switch leaves alone are
    often the larger share of the VRAM -- H3 maps a text encoder of tens of
    GiB while the DiT may still be GPU-resident from the last generation.
    """
    from core import keep_hot

    class _Module:
        def __init__(self):
            self.device = "cuda"

        def to(self, device):
            self.device = device
            return self

    manager = _Manager("minimax_h3")
    manager.txt2img_pipeline = None
    transformer, vae, old_te = _Module(), _Module(), _Module()
    manager.minimax_h3_components = {
        "transformer": transformer, "vae": vae, "text_encoder": old_te,
        "text_encoder_config": {}, "text_encoder_path": "C:/h3/old.safetensors",
        "official_dir": "C:/h3/official",
    }
    keep_hot.mark_resident(manager, "transformer", "key")
    keep_hot.mark_resident(manager, "vae", "key")

    component_switcher._offload_resident_components(manager)

    assert transformer.device == "cpu"
    assert vae.device == "cpu"
    # Untracked components are left alone: the adapter owns the slot it is
    # replacing, and moving it here would only touch it before it is released.
    assert old_te.device == "cuda"


def test_scanned_current_component_does_not_become_a_second_option():
    """The scans surface the loaded file too; it must not appear twice.

    Both rows hash the same path into the same candidate_id, and the UI uses
    that id as the option's value. The duplicate is unselectable -- the change
    handler compares it against the current id and sees no change -- and
    find_candidate resolves it to the current row, which refuses to switch.
    """
    manager = _Manager("anima")
    manager.txt2img_pipeline = None
    current_te = "C:/anima/text_encoders/te.safetensors"
    current_vae = "C:/anima/vae/vae.safetensors"
    manager.anima_components = {
        "transformer": object(), "text_encoder": object(), "vae": object(),
        "paths": {"text_encoder": current_te, "vae": current_vae},
        "vae_source": "external",
    }

    catalog = build_catalog(
        manager,
        text_encoders=[
            {"name": "te", "path": current_te, "arch": "anima", "out_dim": 1024,
             "anima_compatible": True, "anima_compatibility_reason": "verified", "size_gb": 1.0},
            {"name": "other-te", "path": "C:/anima/text_encoders/other.safetensors",
             "arch": "anima", "out_dim": 1024,
             "anima_compatible": True, "anima_compatibility_reason": "verified", "size_gb": 1.0},
        ],
        vaes=[
            {"name": "vae", "path": current_vae, "arch": "anima", "latent_channels": 16,
             "anima_compatible": True, "anima_compatibility_reason": "verified", "size_gb": 0.3},
        ],
    )

    for slot in ("text_encoder", "vae"):
        ids = [item["candidate_id"] for item in catalog[slot]]
        assert len(ids) == len(set(ids)), f"{slot} has duplicate candidate_id values"
        current_rows = [item for item in catalog[slot] if item.get("is_current")]
        assert len(current_rows) == 1
    # The row that survived is the current one, and it is still not offered as
    # something to switch to.
    assert catalog["vae"][0]["is_current"] is True
    assert catalog["vae"][0]["switchable"] is False
    # A genuinely different file is untouched by the dedup.
    assert any(item["_path"] == "C:/anima/text_encoders/other.safetensors"
               and item["switchable"] for item in catalog["text_encoder"])


def test_h3_te_switch_detachment_failure_leaves_health_degraded(monkeypatch):
    """A live owner of the old TE must disable generation, not re-enable it.

    The detachment assertion fires exactly when something still holds the old
    mapping. The slot is already empty by then, so reporting the model ready
    would send the next request into a text_encoder of None.
    """
    import core.keep_hot as keep_hot

    manager = _h3_manager(monkeypatch)
    monkeypatch.setattr(keep_hot, "clear_resident", lambda _manager: None)
    monkeypatch.setattr(component_switcher, "_release_device_cache", lambda: None)

    def still_live():
        raise RuntimeError("a live text encoder reference survived")

    def build(path, _official, **_kwargs):
        raise AssertionError("must not map a new file while an owner survives")

    monkeypatch.setattr(h3_loader, "assert_no_live_text_encoder", still_live)
    monkeypatch.setattr(h3_loader, "build_minimax_h3_text_encoder_bundle", build)
    candidate = {
        "compatibility": "compatible", "switchable": True,
        "_path": "C:/h3/new.safetensors",
    }

    with pytest.raises(ComponentSwitchFailed):
        switch_component(manager, "text_encoder", candidate, 4, 7)

    assert manager.minimax_h3_components["text_encoder"] is None
    assert manager.component_health == "degraded"
    assert manager.component_revision == 7


def test_h3_te_restore_detachment_failure_stays_degraded(monkeypatch):
    """If the restore cannot prove detachment either, stay degraded."""
    import core.keep_hot as keep_hot

    manager = _h3_manager(monkeypatch)
    monkeypatch.setattr(keep_hot, "clear_resident", lambda _manager: None)
    monkeypatch.setattr(component_switcher, "_release_device_cache", lambda: None)

    calls = []

    def assert_detached():
        calls.append("assert")
        if len(calls) > 1:
            raise RuntimeError("owner appeared during the failed build")

    def build(path, _official, **_kwargs):
        raise RuntimeError("new failed")

    monkeypatch.setattr(h3_loader, "assert_no_live_text_encoder", assert_detached)
    monkeypatch.setattr(h3_loader, "build_minimax_h3_text_encoder_bundle", build)
    candidate = {
        "compatibility": "compatible", "switchable": True,
        "_path": "C:/h3/new.safetensors",
    }

    with pytest.raises(ComponentSwitchFailed):
        switch_component(manager, "text_encoder", candidate, 4, 7)

    assert manager.component_health == "degraded"


def _write_header(path, header):
    payload = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(payload)) + payload)


def test_anima_geometry_candidates_and_full_reload_adapter(monkeypatch, tmp_path):
    import core.keep_hot as keep_hot

    te_path = tmp_path / "custom-te.safetensors"
    vae_path = tmp_path / "custom-vae.safetensors"
    te_header = {"model.embed_tokens.weight": {"shape": [151936, 1024]}}
    te_header.update({f"model.layers.{layer}.input_layernorm.weight": {"shape": [1024]} for layer in range(28)})
    _write_header(te_path, te_header)
    _write_header(vae_path, {
        "encoder.conv1.weight": {"shape": [96, 3, 3, 3, 3]},
        "decoder.conv1.weight": {"shape": [384, 16, 3, 3, 3]},
        "conv2.weight": {"shape": [16, 16, 1, 1, 1]},
    })
    assert inspect_anima_component_candidate(str(te_path), "text_encoder")["compatible"] is True
    assert inspect_anima_component_candidate(str(vae_path), "vae")["compatible"] is True

    manager = _Manager("anima")
    manager.txt2img_pipeline = None
    old_te = "C:/anima/old-te.safetensors"
    old_vae = "C:/anima/old-vae.safetensors"
    manager.anima_components = {
        "transformer": object(), "text_encoder": object(), "vae": object(),
        "paths": {"text_encoder": old_te, "vae": old_vae},
        "vae_source": "external",
    }
    catalog = build_catalog(manager, text_encoders=[{
        "name": "verified Anima TE", "path": str(te_path), "size_gb": 1.2,
        "arch": "anima", "out_dim": 1024,
        "anima_compatible": True, "anima_compatibility_reason": "verified geometry",
    }], vaes=[{
        "name": "verified Anima VAE", "path": str(vae_path), "size_gb": 0.3,
        "arch": "anima", "latent_channels": 16,
        "anima_compatible": True, "anima_compatibility_reason": "verified geometry",
    }])
    assert any(item["switchable"] for item in catalog["text_encoder"])
    assert any(item["switchable"] for item in catalog["vae"])
    calls = []

    def reload(source_type, source, pipeline_type, **kwargs):
        calls.append((source_type, source, pipeline_type, kwargs))
        manager.anima_components = {
            "transformer": object(), "text_encoder": object(), "vae": object(),
            "paths": {"text_encoder": kwargs["text_encoder_path"], "vae": kwargs["vae_path"]},
            "vae_source": "external",
        }
        manager.current_model_info = {
            "type": "anima", "source": source, "source_type": source_type,
        }

    manager._load_model_locked = reload
    monkeypatch.setattr(keep_hot, "clear_resident", lambda _manager: None)
    candidate = {"compatibility": "compatible", "switchable": True, "_path": str(te_path)}

    switch_component(manager, "text_encoder", candidate, 4, 7)

    assert len(calls) == 1
    source_type, source, pipeline_type, kwargs = calls[0]
    assert (source_type, source, pipeline_type) == (
        "safetensors", "C:/models/current.safetensors", "txt2img",
    )
    assert kwargs == {
        "force_reload": True,
        "text_encoder_path": str(te_path),
        "vae_path": old_vae,
    }
    assert manager.component_revision == 8


def test_anima_switch_failure_reloads_prior_paths_serially(monkeypatch):
    import core.keep_hot as keep_hot

    manager = _Manager("anima")
    manager.txt2img_pipeline = None
    old_te, old_vae = "C:/anima/old-te.safetensors", "C:/anima/old-vae.safetensors"
    manager.anima_components = {
        "transformer": object(), "text_encoder": object(), "vae": object(),
        "paths": {"text_encoder": old_te, "vae": old_vae}, "vae_source": "external",
    }
    calls = []

    def reload(source_type, source, pipeline_type, **kwargs):
        calls.append(kwargs["text_encoder_path"])
        if kwargs["text_encoder_path"] != old_te:
            manager.anima_components = None
            manager.current_model_info = None
            raise RuntimeError("replacement failed")
        manager.anima_components = {
            "transformer": object(), "text_encoder": object(), "vae": object(),
            "paths": {"text_encoder": old_te, "vae": old_vae}, "vae_source": "external",
        }
        manager.current_model_info = {
            "type": "anima", "source": source, "source_type": source_type,
        }

    manager._load_model_locked = reload
    monkeypatch.setattr(keep_hot, "clear_resident", lambda _manager: None)
    monkeypatch.setattr(component_switcher, "_release_device_cache", lambda: None)
    candidate = {
        "compatibility": "compatible", "switchable": True,
        "_path": "C:/anima/new-te.safetensors",
    }

    with pytest.raises(ComponentSwitchFailed):
        switch_component(manager, "text_encoder", candidate, 4, 7)

    assert calls == ["C:/anima/new-te.safetensors", old_te]
    assert manager.component_health == "ready"
    assert manager.component_revision == 7


def test_standard_sdxl_embedded_components_show_source_checkpoint(tmp_path):
    checkpoint = tmp_path / "current-sdxl.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    manager = _Manager("sdxl")
    manager.current_model_info["source"] = str(checkpoint)
    manager.txt2img_pipeline._sushi_vae_source = "embedded (checkpoint)"
    manager.txt2img_pipeline._sushi_te_embedded = True

    response = build_response(manager, build_catalog(manager))

    states = {slot["slot"]: slot for slot in response["slots"]}
    for slot_name in ("text_encoder", "vae"):
        current = states[slot_name]["current"]
        assert current["origin"] == "embedded_checkpoint"
        assert current["embedded"] is True
        assert "current-sdxl" in current["display_name"]
        assert "current-sdxl" in current["path_display"]
        assert states[slot_name]["candidates"][0]["candidate_id"] == current["candidate_id"]


def test_standard_sdxl_external_defaults_show_recorded_identity():
    manager = _Manager("sdxl")
    manager.txt2img_pipeline._sushi_vae_source = "madebyollin/sdxl-vae-fp16-fix"
    manager.txt2img_pipeline._sushi_te_embedded = False
    manager.txt2img_pipeline._sushi_arch = {"te_type": "qwen3_06b"}

    response = build_response(manager, build_catalog(manager))

    states = {slot["slot"]: slot for slot in response["slots"]}
    vae = states["vae"]["current"]
    text_encoder = states["text_encoder"]["current"]
    assert vae["origin"] == "architecture_default"
    assert vae["display_name"] == "madebyollin/sdxl-vae-fp16-fix"
    assert text_encoder["origin"] == "architecture_default"
    assert text_encoder["display_name"] == "qwen3_06b"
    assert "source unavailable" not in vae["display_name"].lower()
    assert "source unavailable" not in text_encoder["display_name"].lower()


def test_standard_single_file_clip_is_embedded_without_custom_te_flag(tmp_path):
    checkpoint = tmp_path / "plain-sd15.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    manager = _Manager("sd15")
    manager.current_model_info["source"] = str(checkpoint)
    manager.txt2img_pipeline._sushi_vae_source = "stabilityai/sd-vae-ft-mse-original"

    response = build_response(manager, build_catalog(manager))

    text_encoder = next(
        slot for slot in response["slots"] if slot["slot"] == "text_encoder"
    )["current"]
    assert text_encoder["origin"] == "embedded_checkpoint"
    assert "plain-sd15" in text_encoder["display_name"]


def test_provenance_without_loader_evidence_is_unavailable_not_embedded():
    manager = _Manager("sdxl")
    manager.current_model_info["source_type"] = "huggingface"

    response = build_response(manager, build_catalog(manager))

    states = {slot["slot"]: slot for slot in response["slots"]}
    assert states["text_encoder"]["current"]["origin"] == "unavailable"
    assert states["vae"]["current"]["origin"] == "unavailable"
    assert states["text_encoder"]["current"]["embedded"] is False
    assert states["vae"]["current"]["embedded"] is False


def _vision_header(hidden=768, channels=3):
    layers = 12 if hidden == 768 else 27
    intermediate = 3072 if hidden == 768 else 4304
    header = {
        "vision_model.embeddings.patch_embedding.weight": {
            "shape": [hidden, channels, 16, 16],
        },
    }
    expected = {
        "self_attn.q_proj.weight": [hidden, hidden],
        "self_attn.k_proj.weight": [hidden, hidden],
        "self_attn.v_proj.weight": [hidden, hidden],
        "self_attn.out_proj.weight": [hidden, hidden],
        "mlp.fc1.weight": [intermediate, hidden],
        "mlp.fc2.weight": [hidden, intermediate],
        "layer_norm1.weight": [hidden],
        "layer_norm2.weight": [hidden],
    }
    for layer in range(layers):
        for suffix, shape in expected.items():
            header[f"vision_model.encoder.layers.{layer}.{suffix}"] = {"shape": shape}
    return header


def test_optional_vision_encoder_requires_verified_channels_and_dimensions(tmp_path):
    valid_path = tmp_path / "valid-ve.safetensors"
    invalid_path = tmp_path / "invalid-ve.safetensors"
    _write_header(valid_path, _vision_header())
    _write_header(invalid_path, _vision_header(channels=4))
    valid = inspect_vision_encoder_candidate(str(valid_path))
    invalid = inspect_vision_encoder_candidate(str(invalid_path))
    assert valid["compatible"] is True
    assert invalid["compatible"] is False

    manager = _Manager("sdxl")
    catalog = build_catalog(manager, vision_encoders=[
        {
            "name": "valid", "path": str(valid_path), "size_gb": 1,
            "compatibility_verified": valid["compatible"],
            "compatibility_reason": valid["reason"],
        },
        {
            "name": "invalid", "path": str(invalid_path), "size_gb": 1,
            "compatibility_verified": invalid["compatible"],
            "compatibility_reason": invalid["reason"],
        },
    ])
    response = build_response(manager, catalog)
    state = next(slot for slot in response["slots"] if slot["slot"] == "vision_encoder")
    assert state["current"]["origin"] == "unused"
    candidates = {item["display_name"]: item for item in state["candidates"]}
    assert candidates["valid"]["switchable"] is True
    assert candidates["invalid"]["switchable"] is False


def test_vision_encoder_ui_is_resident_reference_conditioning_not_override_detail():
    source = (
        Path(__file__).resolve().parents[2]
        / "frontend" / "src" / "components" / "common" / "ModelLoadSection.tsx"
    ).read_text(encoding="utf-8")
    assert 'label="Vision Encoder (reference conditioning)"' in source
    assert "<VisionEncoderSelector" not in source
    assert "Optional. When loaded, this resident encoder conditions reference images" in source


def test_lifecycle_gate_rejects_mutation_during_generation_and_activity():
    coordinator = ModelStateCoordinator()
    coordinator.begin_generation()
    with pytest.raises(ModelStateBusyError):
        with coordinator.mutation("component switch"):
            pass
    coordinator.end_generation()
    coordinator.begin_activity("training run 1")
    with pytest.raises(ModelStateBusyError):
        with coordinator.mutation("component switch"):
            pass
    coordinator.end_activity("training run 1")
    with coordinator.mutation("component switch"):
        with pytest.raises(ModelStateBusyError):
            coordinator.begin_generation()
