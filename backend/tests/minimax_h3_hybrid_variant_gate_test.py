"""`variant="hybrid"` reaches text-to-video and no other MiniMax-H3 surface.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_h3_hybrid_variant_gate_test.py -v

WHY THIS FILE EXISTS
--------------------
The hybrid contract is conservative because the existing H3 variant gates are
not uniform: two of them refused
`ref2va` BY NAME (a denylist) and one endpoint had no variant gate at all, so a
transformer loaded as a third variant walked through them. C1 closed all four
before anything could write `variant="hybrid"`; C7 then opened exactly one of
them, which is why this file pins WHICH surfaces answer a hybrid and how.

A hybrid transformer is an fl2va base carrying ref2va AdaLN blocks: it shares
every key and shape with both released partitions, so a wrong-variant request
cannot be detected from the weights and would return a bad video rather than
fail.

C7 released ONE workflow for it. The A/B of design section 9.3 ran six arms of
plain text-to-video at 672x384, 124 frames, 20 steps -- and nothing else, so
keyframe binding, reference rows, temporal inpaint, reference outpaint and
chaining stay refused, and the txt2vid case below is INVERTED from the H0 file
this grew out of: the request now passes the gate and carries a `warnings[]`
entry saying on what evidence.

The three properties pinned here:

* every generation surface EXCEPT txt2vid refuses `hybrid` -- img2vid, temporal
  inpaint, ref2vid and the shared outpaint reference gate;
* `fl2va` / `ref2va` are untouched, asserted through a DIFFERENT refusal that
  lives immediately after each gate (so "not refused by the variant gate" is a
  positive observation, not the absence of one);
* `chain_context_for("minimax_h3", "hybrid")` does not fall back to the
  architecture-level entry, which advertises fl2va's `pinned_tail` and
  `motion_preroll`.
"""

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.arch_capabilities import chain_context_for, chain_context_payload  # noqa: E402
from api.error_handlers import ValidationError  # noqa: E402
from api.generation_utils import resolve_minimax_h3_outpaint_reference_gate  # noqa: E402


class _StubPipelineManager:
    """The same stub shape `minimax_h3_pinned_tail_test` uses, plus the no-op
    `reset_cancel_flag` /generate/txt2vid calls once its run opens."""

    def __init__(self, arch="minimax_h3", variant="fl2va"):
        self.is_minimax_h3_model = arch == "minimax_h3"
        self.is_ltx2_model = arch == "ltx2"
        self.current_model_info = {"type": arch, "variant": variant}
        self.minimax_h3_components = None

    def reset_cancel_flag(self):
        pass


def _app(monkeypatch, manager, path, handler_name):
    from fastapi import FastAPI

    import api.routes as routes
    from api.error_handlers import register_error_handlers

    monkeypatch.setattr(routes, "pipeline_manager", manager)
    app = FastAPI()
    register_error_handlers(app)
    app.post(path)(getattr(routes, handler_name))
    app.dependency_overrides[routes.get_gallery_db] = lambda: None
    return app


def _post(app, path, **kwargs):
    import httpx

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(path, **kwargs)
            return response.status_code, response.json()

    return asyncio.run(run())


# ---------------------------------------------------------------------------
# /generate/txt2vid -- the ONE surface C7 released for a merged checkpoint
# ---------------------------------------------------------------------------

def _txt2vid(monkeypatch, variant, hybrid_provenance=None, **body):
    manager = _StubPipelineManager(variant=variant)
    if hybrid_provenance is not None:
        manager.current_model_info["hybrid"] = hybrid_provenance
    app = _app(monkeypatch, manager, "/generate/txt2vid", "generate_txt2vid")
    return _post(app, "/generate/txt2vid", json={"prompt": "a cat", **body})


_PROVENANCE = {
    "base_file": "minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
    "overlay_file": "minimax_h3_ref2va_pruned_fp8_scaled.safetensors",
    "hybrid_recipe": {"preset": "block_range_adaln", "block_range_start": 25,
                      "block_range_end": 49, "final_adaln_from_overlay": False},
}


def _recorded_warnings(monkeypatch):
    """`add_warning` is imported from its module inside the handler, so the
    module attribute is what a request reaches."""
    import api.generation_status as status_module

    recorded = []
    monkeypatch.setattr(status_module, "add_warning",
                        lambda message, code=None: recorded.append((code, message)))
    return recorded


def test_txt2vid_admits_hybrid(monkeypatch):
    """INVERTED at C7. NEGATIVE CONTROL for the release: the request is refused
    by the geometry validation that lives past the gate (width 100 is not
    divisible by 32), so "the variant gate let it through" is observed rather
    than inferred from a 200 this stub cannot produce."""
    status, payload = _txt2vid(monkeypatch, "hybrid", width=100, height=768, num_frames=124)
    assert status == 400, payload
    assert "32" in payload["detail"]
    assert "hybrid" not in payload["error"]


def test_a_hybrid_txt2vid_request_carries_the_experimental_warning(monkeypatch):
    """And carries it BEFORE the geometry check -- i.e. it is attached on the
    way in, not somewhere along a path only a full generation reaches."""
    recorded = _recorded_warnings(monkeypatch)
    _txt2vid(monkeypatch, "hybrid", hybrid_provenance=_PROVENANCE,
             width=100, height=768, num_frames=124)
    assert len(recorded) == 1, recorded
    code, message = recorded[0]
    assert code == "minimax_h3_hybrid_experimental"
    # The recipe that produced THIS model, read from the loaded model info.
    assert "Blocks 25..49" in message
    assert "minimax_h3_fl2va_pruned_fp8_scaled.safetensors" in message
    assert "minimax_h3_ref2va_pruned_fp8_scaled.safetensors" in message
    # The scope of what was measured, and the one audio property that was not.
    assert "672x384" in message and "124 frames" in message
    assert "synchronisation with the video was not measured" in message
    # LoRA is named, on the PERMITTED-but-unmeasured side: `loras` rides this
    # very request and takes effect, so the refusal list would otherwise read as
    # exhaustive while leaving out the one thing the request still carries.
    assert "A LoRA is applied if one is sent" in message
    assert "nothing about a LoRA on a merged checkpoint was measured" in message


def test_txt2vid_refuses_every_variant_that_is_not_a_released_or_merged_one(monkeypatch):
    """The allowlist gained `hybrid`, not a prefix match.

    THE MUTANT THIS EXISTS FOR: C7 opens the gate with `startswith("hybrid")`
    or by dropping it entirely. A recipe-shaped label names a merge the A/B did
    not run, and the img2vid and temporal-inpaint allowlists would still refuse
    it, so admitting it here would be released on nothing.
    """
    status, payload = _txt2vid(monkeypatch, "hybrid_25_49")
    assert status == 400, payload
    assert "hybrid_25_49" in payload["error"]


@pytest.mark.parametrize("variant", ["fl2va", "ref2va", None])
def test_txt2vid_is_unchanged_for_the_released_partitions(monkeypatch, variant):
    """NEGATIVE CONTROL: the request gets past the gate and is refused by the
    geometry validation that follows it (width 100 is not divisible by 32), and
    picks up NO warning on the way -- the experimental entry is the merged
    checkpoint's alone.

    `None` is the renamed-checkpoint case (`detect_minimax_h3_layout` returns a
    null variant): it kept passing here, exactly as before this gate existed.
    """
    recorded = _recorded_warnings(monkeypatch)
    status, payload = _txt2vid(monkeypatch, variant, width=100, height=768, num_frames=124)
    assert status == 400, payload
    assert "32" in payload["detail"]
    assert "hybrid" not in payload["error"]
    assert recorded == []


def test_a_non_h3_video_model_is_untouched_by_the_gate(monkeypatch):
    """The gate is inside `is_minimax_h3_model`; LTX-2.3 never reads a variant.
    Refused by the same geometry rule, with no warning recorded."""
    recorded = _recorded_warnings(monkeypatch)
    app = _app(monkeypatch, _StubPipelineManager(arch="ltx2", variant="hybrid"),
               "/generate/txt2vid", "generate_txt2vid")
    status, payload = _post(app, "/generate/txt2vid",
                            json={"prompt": "a cat", "width": 100, "height": 512,
                                  "num_frames": 121})
    assert status == 400, payload
    assert "32" in payload["detail"]
    assert recorded == []


# ---------------------------------------------------------------------------
# /generate/img2vid -- denylist flipped to an allowlist over NAMED variants
# ---------------------------------------------------------------------------

def _img2vid(monkeypatch, variant, files=None):
    app = _app(monkeypatch, _StubPipelineManager(variant=variant),
               "/generate/img2vid", "generate_img2vid")
    return _post(app, "/generate/img2vid", data={"prompt": "a cat"}, files=files or {})


def test_img2vid_refuses_hybrid(monkeypatch):
    status, payload = _img2vid(monkeypatch, "hybrid",
                               files={"image": ("f.png", b"not a real image", "image/png")})
    assert status == 400, payload
    assert "hybrid variant, not fl2va" in payload["error"]
    # Refused BEFORE the upload is read, like this endpoint's other cheap gates.
    assert "measured" in payload["detail"]


def test_img2vid_still_refuses_ref2va_with_its_own_wording(monkeypatch):
    """The historical row, byte-identical: same summary, same detail."""
    status, payload = _img2vid(monkeypatch, "ref2va",
                               files={"image": ("f.png", b"not a real image", "image/png")})
    assert status == 400, payload
    assert payload["error"] == (
        "The loaded MiniMax-H3 transformer is the ref2va variant, not fl2va")
    assert "/generate/ref2vid" in payload["detail"]
    assert "post-reference rotary origin" in payload["detail"]


@pytest.mark.parametrize("variant", ["fl2va", None])
def test_img2vid_is_unchanged_for_fl2va_and_an_unidentified_variant(monkeypatch, variant):
    """NEGATIVE CONTROL: refused by the conditioning-medium check that follows
    the gate, i.e. the gate let it through."""
    status, payload = _img2vid(monkeypatch, variant)
    assert status == 400, payload
    assert payload["error"] == "img2vid needs something to condition on"


# ---------------------------------------------------------------------------
# /generate/inpaint/video -- same flip
# ---------------------------------------------------------------------------

def _inpaint_video(monkeypatch, variant, **fields):
    app = _app(monkeypatch, _StubPipelineManager(variant=variant),
               "/generate/inpaint/video", "generate_inpaint_video")
    data = {"prompt": "a cat", "regenerate_start_frame": "40", "regenerate_end_frame": "85"}
    data.update({k: str(v) for k, v in fields.items()})
    return _post(app, "/generate/inpaint/video", data=data,
                 files={"video": ("clip.mp4", b"not a real clip", "video/mp4")})


def test_temporal_inpaint_refuses_hybrid(monkeypatch):
    """Unchanged by C7: text-to-video is the only released workflow, so the
    mid-clip pin stays refused -- and the refusal now names where the merged
    checkpoint DOES generate instead of claiming it generates nowhere."""
    status, payload = _inpaint_video(monkeypatch, "hybrid")
    assert status == 400, payload
    assert payload["error"] == "The loaded MiniMax-H3 transformer is the hybrid variant"
    assert "/generate/txt2vid and nowhere else" in payload["detail"]
    assert "unmeasured" in payload["detail"]


def test_temporal_inpaint_admits_ref2va(monkeypatch):
    """NEGATIVE CONTROL, and the row that moved since this file was written:
    the variant table now lives in `resolve_minimax_h3_inpaint_reference_gate`
    and `ref2va` was opened there (unmeasured, warned, not refused). It is
    refused by the clip decode that follows the gate."""
    status, payload = _inpaint_video(monkeypatch, "ref2va")
    assert status == 400, payload
    assert payload["error"] == "Failed to decode the uploaded video clip"


@pytest.mark.parametrize("variant", ["fl2va", None])
def test_temporal_inpaint_is_unchanged_for_fl2va_and_an_unidentified_variant(
        monkeypatch, variant):
    """NEGATIVE CONTROL: refused by the audio-mode check that follows the gate,
    before the clip is decoded."""
    status, payload = _inpaint_video(monkeypatch, variant, inpaint_video_audio_mode="nope")
    assert status == 400, payload
    assert payload["error"] == "Invalid inpaint_video_audio_mode"


# ---------------------------------------------------------------------------
# /generate/ref2vid -- already an allowlist; verified, not changed
# ---------------------------------------------------------------------------

def _ref2vid(monkeypatch, variant, files=None):
    app = _app(monkeypatch, _StubPipelineManager(variant=variant),
               "/generate/ref2vid", "generate_ref2vid")
    return _post(app, "/generate/ref2vid", data={"prompt": "a cat"}, files=files or {})


@pytest.mark.parametrize("variant", ["hybrid", "fl2va", None])
def test_ref2vid_admits_only_ref2va(monkeypatch, variant):
    status, payload = _ref2vid(monkeypatch, variant)
    assert status == 400, payload
    assert payload["error"].endswith("variant, not ref2va")
    assert (variant or "unidentified") in payload["error"]


def test_ref2vid_is_unchanged_for_ref2va(monkeypatch):
    """NEGATIVE CONTROL: refused by the reference-count check after the gate."""
    status, payload = _ref2vid(monkeypatch, "ref2va")
    assert status == 400, payload
    assert payload["error"] == "ref2vid needs at least one reference"


# ---------------------------------------------------------------------------
# The shared outpaint reference gate (route + backend re-check)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("placement", ["extend_forward", "extend_backward", "bridge"])
@pytest.mark.parametrize("has_refs", [False, True])
def test_the_outpaint_gate_refuses_hybrid_on_every_row(placement, has_refs):
    """Named explicitly rather than left to the "unidentified variant" clause,
    which only refuses the reference-carrying rows."""
    with pytest.raises(ValidationError) as error:
        resolve_minimax_h3_outpaint_reference_gate(
            "hybrid", has_reference_images=has_refs, placement=placement,
            generated_frames=124)
    assert "hybrid" in str(error.value)


def test_outpaint_refuses_hybrid_before_the_upload_is_read(monkeypatch):
    """The route half of the row above. The hybrid row needs neither the decoded
    clip nor the placement plan, so it is answered where this endpoint's other
    cheap gates live -- not after the whole upload has been decoded."""
    import inspect

    from api.routes import generate_outpaint_video

    app = _app(monkeypatch, _StubPipelineManager(variant="hybrid"),
               "/generate/outpaint/video", "generate_outpaint_video")
    status, payload = _post(app, "/generate/outpaint/video", data={"prompt": "a cat"},
                            files={"video": ("clip.mp4", b"not a real clip", "video/mp4")})
    assert status == 400, payload
    assert "hybrid" in payload["error"]

    # Structurally, so the refusal cannot drift back behind the decode while the
    # assertion above still passes on a clip that happens to be junk.
    source = inspect.getsource(generate_outpaint_video)
    assert source.index('if _h3_variant == "hybrid":') < source.index("await video.read()")


def test_the_outpaint_gate_is_unchanged_for_the_released_partitions():
    """NEGATIVE CONTROL: the rows either side of the new one still answer as
    `minimax_h3_outpaint_reference_gate_test` pins them."""
    assert resolve_minimax_h3_outpaint_reference_gate(
        "fl2va", has_reference_images=False, placement="bridge") is None
    assert resolve_minimax_h3_outpaint_reference_gate(
        "ref2va", has_reference_images=True, placement="extend_forward",
        generated_frames=124) is None
    # ... and an unidentified variant keeps its own asymmetry: references
    # refused, a plain extend allowed.
    assert resolve_minimax_h3_outpaint_reference_gate(
        "", has_reference_images=False, placement="extend_forward") is None
    with pytest.raises(ValidationError):
        resolve_minimax_h3_outpaint_reference_gate(
            "", has_reference_images=True, placement="extend_forward")


# ---------------------------------------------------------------------------
# CHAIN_CONTEXT: the fallback that would advertise fl2va's modes
# ---------------------------------------------------------------------------

def test_hybrid_does_not_inherit_the_architecture_level_chain_entry():
    arch = chain_context_payload()["minimax_h3"]
    hybrid = chain_context_for("minimax_h3", "hybrid")
    assert hybrid is not arch and hybrid != arch
    # The two modes the arch-level entry would have handed it.
    for mode in ("pinned_tail", "motion_preroll"):
        assert mode in arch["chain_continuation_modes"]
        assert mode not in hybrid["chain_continuation_modes"]
    assert hybrid["chain_supports_sparse_motion_anchors"] is False
    assert hybrid["chain_supports_reference_video"] is False
    # NEGATIVE CONTROL: fl2va (no entry of its own) still answers with the
    # architecture-level entry, so the lookup itself is unchanged.
    assert chain_context_for("minimax_h3", "fl2va") == arch
    assert chain_context_for("minimax_h3", "HYBRID") == hybrid


def test_the_chain_planner_refuses_a_pin_on_hybrid():
    """The capability table is consumed, not merely declared."""
    from api.generation_utils import plan_video_continuation_context

    with pytest.raises(ValidationError) as error:
        plan_video_continuation_context("pinned_tail", 5, "minimax_h3", "hybrid")
    assert "chain_context" in str(error.value.detail)
