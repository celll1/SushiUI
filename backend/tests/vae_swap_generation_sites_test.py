"""The generation sites that assumed the architecture's native channel count
(design §9.3 / §9.6, phase P4).

Each case is paired: a NATIVE model must take the same branch it took before
this phase (same preview decoder, same inpaint gate, same keep-hot key, same
latent shape, no new warning), and only a checkpoint whose declared channel
count differs from its architecture's may behave differently.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_BACKEND = str(Path(__file__).resolve().parents[1])
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from api import generation_utils as gu
from core.models.components.vae_registry import preview_decoder_for


def _manager(info=None, **attrs):
    return SimpleNamespace(current_model_info=info, minit2i_components=None, **attrs)


NATIVE_SDXL = {"type": "sdxl", "latent_channels": 4}
SWAPPED = {
    "type": "sdxl", "latent_channels": 16, "vae_type": "flux1",
    "vae_provenance": "registry:flux1", "vae_hash": "0123456789abcdef",
    "vae_struct_native": False, "vae_identity_native": False,
}


# --- 1. live preview decoder routing ---------------------------------------

class TestPreviewRouting:
    def test_a_family_answers_for_every_registry_vae(self):
        # The old projection dropped the families it cannot from_pretrained,
        # which is exactly where a swap's preview decision would land.
        assert preview_decoder_for("sdxl") == "taesdxl"
        assert preview_decoder_for("sd15") == "taesd"
        assert preview_decoder_for("flux1") == "taef1"
        assert preview_decoder_for("flux2") == "matrix32"
        assert preview_decoder_for("qwen_image") == "matrix16"

    def test_an_unnamed_family_falls_back_to_the_channel_count(self):
        # file:/model: sources resolve to family "custom".
        assert preview_decoder_for("custom", 16) == "matrix16"
        assert preview_decoder_for("custom", 32) == "matrix32"
        assert preview_decoder_for("custom", 8) == ""
        assert preview_decoder_for("custom") == ""

    def test_a_native_model_is_not_rerouted(self):
        assert gu._preview_vae_kind({}) is None
        assert gu._preview_vae_kind(NATIVE_SDXL) is None
        assert gu._preview_vae_kind({**SWAPPED, "vae_identity_native": True}) is None

    def test_a_swapped_model_routes_on_its_vae(self):
        assert gu._preview_vae_kind(SWAPPED) == "taef1"
        assert gu._preview_vae_kind(
            {"type": "sdxl", "latent_channels": 32, "vae_type": "custom",
             "vae_identity_native": False}) == "matrix32"

    def test_the_same_shape_from_another_vae_keeps_the_arch_decoder(self):
        # struct_native, identity_native=False: 4ch on SDXL, TAESD-XL still fits.
        assert gu._preview_vae_kind(
            {"type": "sdxl", "latent_channels": 4, "vae_type": "custom",
             "vae_struct_native": True, "vae_identity_native": False}) is None

    def test_a_latent_space_no_decoder_covers_is_refused(self):
        assert gu._preview_vae_kind(
            {"type": "sdxl", "latent_channels": 8, "vae_type": "custom",
             "vae_identity_native": False}) == "none"

    def test_native_preview_kwargs_are_unchanged(self):
        class StableDiffusionXLPipeline:
            pass

        kwargs = gu.preview_arch_kwargs(_manager(NATIVE_SDXL),
                                        StableDiffusionXLPipeline())
        assert kwargs["vae_preview_kind"] is None
        assert kwargs["is_sdxl"] is True
        assert kwargs["is_zimage"] is False

    def test_a_swapped_model_carries_its_kind_into_the_callback(self):
        pipeline = SimpleNamespace()
        kwargs = gu.preview_arch_kwargs(_manager(SWAPPED), pipeline)
        assert kwargs["vae_preview_kind"] == "taef1"
        # The class-name test still says "not SDXL"; the kind is what decides.
        assert kwargs["is_sdxl"] is False

    def test_an_uncovered_latent_space_warns_once_per_generation(self):
        from api import generation_status as gs

        gen_id = gs.start_generation("txt2img")
        try:
            gu.preview_arch_kwargs(
                _manager({"type": "sdxl", "latent_channels": 8,
                          "vae_type": "custom", "vae_identity_native": False}),
                SimpleNamespace())
            codes = [w["code"] for w in gs.get_warnings(gen_id)]
        finally:
            gs.complete_generation(generation_id=gen_id)
        assert codes.count("preview_unavailable") == 1

    def test_a_native_generation_records_no_preview_warning(self):
        from api import generation_status as gs

        gen_id = gs.start_generation("txt2img")
        try:
            gu.preview_arch_kwargs(_manager(NATIVE_SDXL), SimpleNamespace())
            codes = [w["code"] for w in gs.get_warnings(gen_id)]
        finally:
            gs.complete_generation(generation_id=gen_id)
        assert "preview_unavailable" not in codes


class TestPreviewDecode:
    """`decode_latent` honours the kind over every architecture flag."""

    def _manager_with_stub_taesd(self):
        from core.utils.taesd import TAESDManager

        manager = TAESDManager()
        seen = {}

        def _load(is_sdxl=False, is_zimage=False, is_deus=False,
                  is_zimage_sdxl_vae=False, is_flux2=False):
            seen.update(is_sdxl=is_sdxl, is_zimage=is_zimage, is_deus=is_deus,
                        is_zimage_sdxl_vae=is_zimage_sdxl_vae, is_flux2=is_flux2)
            return None  # stop before any hub download

        manager.load_taesd = _load
        return manager, seen

    def test_a_native_call_keeps_its_flags(self):
        manager, seen = self._manager_with_stub_taesd()
        assert manager.decode_latent(torch.zeros(1, 4, 8, 8), is_sdxl=True) is None
        assert seen["is_sdxl"] is True

    def test_a_flux1_vae_on_sdxl_takes_the_taef1_path(self):
        manager, seen = self._manager_with_stub_taesd()
        manager.decode_latent(torch.zeros(1, 16, 8, 8), is_sdxl=True,
                              vae_preview_kind="taef1")
        assert seen == dict(is_sdxl=False, is_zimage=True, is_deus=False,
                            is_zimage_sdxl_vae=False, is_flux2=False)

    def test_an_sdxl_vae_on_sd15_takes_the_taesdxl_path(self):
        manager, seen = self._manager_with_stub_taesd()
        manager.decode_latent(torch.zeros(1, 4, 8, 8), vae_preview_kind="taesdxl")
        assert seen["is_sdxl"] is True and seen["is_zimage"] is False

    @pytest.mark.parametrize("kind,channels", [("matrix16", 16), ("matrix32", 32)])
    def test_a_projection_preview_decodes_without_a_tiny_decoder(self, kind, channels):
        manager, _ = self._manager_with_stub_taesd()
        preview = manager.decode_latent(torch.zeros(1, channels, 8, 8),
                                        is_sdxl=True, vae_preview_kind=kind)
        assert preview is not None and preview.size == (64, 64)

    def test_an_uncovered_latent_space_decodes_to_nothing(self):
        manager, seen = self._manager_with_stub_taesd()
        assert manager.decode_latent(torch.zeros(1, 8, 8, 8), is_sdxl=True,
                                     vae_preview_kind="none") is None
        assert seen == {}  # never reached a decoder


# --- 2. the inpaint channel test -------------------------------------------

class TestInpaintChannelGate:
    def _pipeline(self, latent_channels):
        return SimpleNamespace(
            vae=SimpleNamespace(config=SimpleNamespace(latent_channels=latent_channels)))

    def test_channels_come_from_the_loaded_wiring(self, monkeypatch):
        from core.inference import custom_sampling as cs

        monkeypatch.setattr(cs, "_loaded_wiring",
                            lambda: SimpleNamespace(latent_channels=16))
        assert cs.latent_channels_for(self._pipeline(4)) == 16

    def test_without_a_wiring_the_pipelines_own_vae_answers(self, monkeypatch):
        from core.inference import custom_sampling as cs

        monkeypatch.setattr(cs, "_loaded_wiring", lambda: None)
        assert cs.latent_channels_for(self._pipeline(4)) == 4
        assert cs.latent_channels_for(SimpleNamespace()) == 4

    def test_the_gate_is_2c_plus_1(self, monkeypatch):
        from core.inference import custom_sampling as cs

        def gate(unet_in_channels, latent_channels):
            monkeypatch.setattr(cs, "_loaded_wiring",
                                lambda: SimpleNamespace(latent_channels=latent_channels))
            return unet_in_channels == 2 * cs.latent_channels_for(SimpleNamespace()) + 1

        assert gate(9, 4) is True       # native SD/SDXL inpaint model, unchanged
        assert gate(4, 4) is False      # native non-inpaint model, unchanged
        assert gate(9, 16) is False     # 9ch means nothing in a 16ch latent space
        assert gate(33, 16) is True     # 16 + 16 + 1

    def test_the_loop_asks_for_the_gate(self):
        # The sampling loop cannot be driven from a unit test; pin its call site.
        source = (Path(_BACKEND) / "core" / "inference" / "custom_sampling.py").read_text(
            encoding="utf-8")
        assert "unet.config.in_channels == 2 * latent_channels_for(pipeline) + 1" in source


# --- 3. keep_hot model key --------------------------------------------------

class TestKeepHotKey:
    def test_a_native_models_key_is_stable(self):
        from core.keep_hot import compute_model_key

        manager = SimpleNamespace(current_model="a.safetensors",
                                  current_model_info={"type": "sdxl", "latent_channels": 4},
                                  _override_vae_path=None)
        params = {"unet_quantization": None}
        assert compute_model_key(manager, params) == compute_model_key(manager, params)

    def test_a_vae_override_invalidates_the_resident_set(self):
        from core.keep_hot import compute_model_key

        manager = SimpleNamespace(current_model="a.safetensors",
                                  current_model_info={}, _override_vae_path=None)
        params = {"unet_quantization": None}
        before = compute_model_key(manager, params)
        manager._override_vae_path = "M:/model/vae/other.safetensors"
        assert compute_model_key(manager, params) != before

    def test_a_different_vae_identity_invalidates_it(self):
        from core.keep_hot import compute_model_key

        manager = SimpleNamespace(current_model="a.safetensors",
                                  current_model_info={"vae_hash": "aaaa"},
                                  _override_vae_path=None)
        params = {"unet_quantization": None}
        before = compute_model_key(manager, params)
        manager.current_model_info = {"vae_hash": "bbbb"}
        assert compute_model_key(manager, params) != before

    def test_a_manager_without_the_attributes_still_keys(self):
        from core.keep_hot import compute_model_key

        assert compute_model_key(SimpleNamespace(), {"unet_quantization": None})


# --- 4. the latent shape constant ------------------------------------------

class TestLatentScaleFactor:
    def test_the_loaded_wiring_answers(self, monkeypatch):
        from core.inference import custom_sampling as cs

        monkeypatch.setattr(cs, "_loaded_wiring",
                            lambda: SimpleNamespace(vae_scale_factor=16))
        assert cs.latent_scale_factor(SimpleNamespace(vae_scale_factor=8)) == 16

    def test_without_a_wiring_the_pipeline_answers(self, monkeypatch):
        from core.inference import custom_sampling as cs

        monkeypatch.setattr(cs, "_loaded_wiring", lambda: None)
        assert cs.latent_scale_factor(SimpleNamespace(vae_scale_factor=8)) == 8
        assert cs.latent_scale_factor(SimpleNamespace()) == 8

    def test_a_native_sd_wiring_still_divides_by_eight(self):
        from core.inference import custom_sampling as cs
        from core.models.component_registry import _WIRING_BY_ARCH

        for arch in ("sd15", "sdxl"):
            spec = _WIRING_BY_ARCH[arch].replace(latent_channels=16)
            assert spec.vae_scale_factor == 8
        source = (Path(_BACKEND) / "core" / "inference" / "custom_sampling.py").read_text(
            encoding="utf-8")
        block = source[source.index("    # Prepare latents\n"):][:600]
        assert "latent_height = height // _scale" in block
        assert "latent_width = width // _scale" in block
        assert "// 8" not in block


# --- 5. load-time warning replay -------------------------------------------

class TestLoadWarningReplay:
    def test_the_queue_dedups_and_drains_once(self):
        from core.pipeline import DiffusionPipelineManager

        manager = DiffusionPipelineManager.__new__(DiffusionPipelineManager)
        manager._sushi_load_warnings = []
        manager.record_load_warning("swapped", code="model_vae_swapped")
        manager.record_load_warning("swapped", code="model_vae_swapped")
        assert manager.consume_load_warnings() == [
            {"code": "model_vae_swapped", "message": "swapped"}]
        assert manager.consume_load_warnings() == []

    def test_a_swapped_load_queues_a_notice(self):
        from core.pipeline import DiffusionPipelineManager

        manager = DiffusionPipelineManager.__new__(DiffusionPipelineManager)
        manager._sushi_load_warnings = []
        pipeline = SimpleNamespace(_sushi_vae_identity={
            "family": "flux1", "latent_channels": 16, "content_hash": "abcd",
            "provenance": "registry:flux1", "struct_native": False,
            "identity_native": False,
        })
        fields = manager._fold_sd_latent_identity(pipeline, "sdxl")
        assert fields["latent_channels"] == 16
        queued = manager.consume_load_warnings()
        assert [w["code"] for w in queued] == ["model_vae_swapped"]
        assert "registry:flux1" in queued[0]["message"]

    def test_a_native_load_queues_nothing(self):
        from core.pipeline import DiffusionPipelineManager

        manager = DiffusionPipelineManager.__new__(DiffusionPipelineManager)
        manager._sushi_load_warnings = []
        fields = manager._fold_sd_latent_identity(SimpleNamespace(), "sdxl")
        assert fields == {"latent_channels": 4}
        assert manager.consume_load_warnings() == []

    def test_the_first_generation_replays_the_queue(self):
        from api import generation_status as gs
        from core.pipeline import pipeline_manager

        pipeline_manager.record_load_warning("swapped", code="model_vae_swapped")
        first = gs.start_generation("txt2img")
        try:
            first_codes = [w["code"] for w in gs.get_warnings(first)]
        finally:
            gs.complete_generation(generation_id=first)
        second = gs.start_generation("txt2img")
        try:
            second_codes = [w["code"] for w in gs.get_warnings(second)]
        finally:
            gs.complete_generation(generation_id=second)

        assert first_codes == ["model_vae_swapped"]
        assert second_codes == []
