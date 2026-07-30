"""
Tests for durable per-image generation warnings (``effective_warnings``).

Run with:
    d:\\celll1\\webui_cl\\venv\\Scripts\\python.exe -m pytest backend/tests/test_warning_persistence.py -v

A generation that silently degrades has to say so ON THE ROW, because the row
is the only artifact that survives the request. Two real diagnoses were blocked
by warnings that were RAISED and then lost before anything persisted them:

  A. A VAE override that never reaches the decoder. ``apply_overrides``
     swallows a load failure into a ``vae_override_error`` warning while still
     recording ``vae_override_path``, and ``DiffusionPipelineManager``'s
     ``load_override_vae``
     used to return SILENTLY when the loaded model exposes no VAE slot — so a
     row could claim an override that never ran. ``GeneratedImage.to_dict``
     keys its override-label derivation on the ``vae_override_error`` CODE, so
     the code must survive redaction even when the message quotes a path.

  B. ``attention_kernel_fallback``. The flash/sage/TQ kernels catch a bare
     ``Exception`` (including OOM) and rerun the call on native SDPA — flash
     runs bf16, native runs the model dtype, so it is NOT equivalent. The
     warning existed but was deduped ONCE PER PROCESS, so only the first
     generation after a backend start could ever record it.

  C. Mis-attribution under a QUEUE. Requests do not serialize at the handler:
     ``start_generation()`` runs at the top, the GPU slot is taken much later,
     so a queued second request sits inside its own start/complete window for
     the whole of the first request's denoise. Any scheme that picks the
     "newest started" generation therefore files the RUNNING generation's
     warnings onto the WAITING one — the normal path with two queued requests,
     not a narrow race. Identity travels with the emitter (a ContextVar set in
     ``start_generation`` and carried into the executor by
     ``contextvars.copy_context().run``), so the tests below model a request as
     its own ``contextvars.Context``, which is what an asyncio Task gives each
     handler in production.

These tests drive the real code paths (no mock of ``generation_status``), with
constructed inputs — no GPU, no model, no server.
"""

from __future__ import annotations

import contextvars
import json
import os
import sys
import tempfile
import threading
import unittest

# ── path setup ───────────────────────────────────────────────────────────────
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_BACKEND = os.path.join(_REPO, "backend")
for _p in (_REPO, _BACKEND):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

from api import generation_status as gs  # noqa: E402
from api.generation_overrides import apply_overrides  # noqa: E402
from utils.path_redaction import redact_params_for_sharing  # noqa: E402

# A path that must never reach a shared PNG, and is not this machine's.
_FAKE_VAE_PATH = r"M:\models\private\someone\vae_dec_run\diffusion_pytorch_model.safetensors"


def _codes(warnings):
    return [w.get("code") for w in warnings]


class _Request:
    """One in-flight generation request.

    Each FastAPI handler runs in its own asyncio Task, i.e. its own
    ``contextvars.Context``; this models that. ``run_in_executor`` mirrors
    ``routes._run_generation_in_executor`` exactly (copy THIS request's
    context, run the blocking work inside the copy on another thread), which is
    the only reason the sampling thread knows which generation it belongs to.
    """

    def __init__(self, kind: str):
        self.ctx = contextvars.copy_context()
        self.kind = kind
        self.gid = self.run(lambda: gs.start_generation(kind))

    def run(self, fn):
        """Run ``fn`` on this request's own context (the handler body)."""
        return self.ctx.run(fn)

    def run_in_executor(self, fn):
        """Run ``fn`` on a worker thread that inherited this request's context."""
        box = {}

        def _outer():
            inner = contextvars.copy_context()   # copy of THIS request's context

            def _work():
                box["result"] = inner.run(fn)

            t = threading.Thread(target=_work)
            t.start()
            t.join()

        self.run(_outer)
        return box.get("result")

    def warnings(self):
        return gs.get_warnings(self.gid)

    def complete(self, last_result=None, pass_id: bool = True):
        """Finish, using the REAL route call shape by default.

        ``pass_id=False`` exercises the no-id default path that any future
        caller could take (routes.py passes ``generation_id=_gen_id`` at all 39
        call sites; the default must still not retire someone else's run).
        """
        if pass_id:
            self.run(lambda: gs.complete_generation(last_result, generation_id=self.gid))
        else:
            self.run(lambda: gs.complete_generation(last_result))


class _FakePM:
    """Minimal stand-in for PipelineManager's override surface."""

    def __init__(self, vae_exc=None):
        self._vae_exc = vae_exc

    def load_override_vae(self, path, **kwargs):
        if self._vae_exc is not None:
            raise self._vae_exc

    def load_override_te(self, path):
        return None

    def override_vae_identity(self):
        return ("override: someone_vae", "deadbeef")


class TestAccumulatorScoping(unittest.TestCase):
    """The accumulator is a module global; generations must not bleed."""

    def test_warning_outside_a_generation_is_dropped(self):
        gs.complete_generation()
        gs.add_warning("stray background notice", code="stray")
        self.assertEqual(gs.current_generation_id(), 0)
        gid = gs.start_generation("txt2img")
        self.assertNotIn("stray", _codes(gs.get_warnings(gid)))
        gs.complete_generation(generation_id=gid)

    def test_same_warning_twice_is_recorded_once(self):
        gid = gs.start_generation("txt2img")
        gs.add_warning("duplicated notice", code="dup")
        gs.add_warning("duplicated notice", code="dup")
        self.assertEqual(_codes(gs.get_warnings(gid)), ["dup"])
        gs.complete_generation(generation_id=gid)

    def test_warnings_readable_after_complete(self):
        """Routes read get_warnings() for the response AFTER complete_generation()
        in some paths; the bucket must survive being retired."""
        gid = gs.start_generation("txt2img")
        gs.add_warning("kept", code="kept")
        gs.complete_generation({"image_id": 1}, generation_id=gid)
        self.assertEqual(_codes(gs.get_warnings(gid)), ["kept"])

    def test_executor_thread_warnings_land(self):
        """Sampling runs in a thread-pool executor; a bare worker thread with no
        inherited context must still land somewhere sane (the oldest active
        generation, i.e. the one holding the GPU slot)."""
        gid = gs.start_generation("txt2img")
        t = threading.Thread(target=gs.add_warning, args=("from a worker thread",),
                             kwargs={"code": "worker"})
        t.start()
        t.join()
        self.assertIn("worker", _codes(gs.get_warnings(gid)))
        gs.complete_generation(generation_id=gid)


class TestQueuedRequests(unittest.TestCase):
    """Two requests in flight: A holds the GPU slot, B is queued behind it.

    This is the shape the fix exists for. ``start_generation`` runs at the top
    of B's handler while A is mid-denoise, so B is the newest-STARTED
    generation for the whole time A is computing.
    """

    def test_running_generation_keeps_its_own_denoise_warnings(self):
        a = _Request("txt2img")            # holds the slot
        b = _Request("img2img")            # queued: started, blocked, silent

        a.run_in_executor(lambda: gs.add_warning(
            "flash_attn error (CUDA out of memory); falling back to native attention",
            code="attention_kernel_fallback"))

        self.assertEqual(_codes(a.warnings()), ["attention_kernel_fallback"],
                         "the running generation lost its own denoise warning")
        self.assertEqual(b.warnings(), [],
                         "the QUEUED generation was handed a fallback it never experienced")
        a.complete()
        b.run_in_executor(lambda: gs.add_warning("B's own downgrade", code="attention_downgrade"))
        self.assertEqual(_codes(b.warnings()), ["attention_downgrade"])
        b.complete()

    def test_current_generation_id_matches_where_the_warning_lands(self):
        """The attention dedup keys on current_generation_id(); if that named a
        different generation than add_warning() writes to, the running
        generation's warning would be suppressed as the queued one's duplicate."""
        a = _Request("txt2img")
        b = _Request("img2img")

        seen = a.run_in_executor(gs.current_generation_id)
        self.assertEqual(seen, a.gid)
        self.assertEqual(b.run(gs.current_generation_id), b.gid)
        a.complete()
        b.complete()

    def test_attention_dedup_rearms_per_generation_under_a_queue(self):
        """Drives the real backend helper from inside each request's context."""
        from core.attention import backends

        message = "unit-test queued kernel fallback"
        a = _Request("txt2img")
        b = _Request("img2img")
        a.run_in_executor(lambda: backends._warn_kernel_fallback(message))
        b.run_in_executor(lambda: backends._warn_kernel_fallback(message))

        self.assertIn(message, [w["message"] for w in a.warnings()])
        self.assertIn(message, [w["message"] for w in b.warnings()],
                      "the queued generation's own fallback was suppressed as a duplicate")
        a.complete()
        b.complete()

    def test_finishing_route_retires_only_its_own_generation(self):
        """SEV-2: `_finish` defaulted to the newest active id, so A completing
        retired B — and A's later warnings then landed in B's bucket."""
        a = _Request("txt2img")
        b = _Request("img2img")

        a.run(lambda: gs.add_warning("A degraded", code="a_code"))
        a.complete({"image_id": 1}, pass_id=False)   # no-id default path

        self.assertIn(b.gid, gs._active_ids, "A's completion retired B")
        self.assertNotIn(a.gid, gs._active_ids)
        self.assertEqual(_codes(gs.get_snapshot()["last_result"]["warnings"]), ["a_code"],
                         "last_result carried the wrong generation's warnings")

        b.run(lambda: gs.add_warning("B degraded", code="b_code"))
        a.run(lambda: gs.add_warning("A after complete", code="a_late"))
        self.assertEqual(_codes(a.warnings()), ["a_code"],
                         "a finished generation kept accepting warnings")
        self.assertEqual(_codes(b.warnings()), ["b_code"],
                         "A's late warning leaked into the still-running B")
        b.complete(pass_id=False)

    def test_failure_does_not_flip_status_for_a_running_sibling(self):
        """SEV-5."""
        a = _Request("txt2img")
        b = _Request("img2img")
        a.run(lambda: gs.fail_generation("A blew up", generation_id=a.gid))
        self.assertEqual(gs.get_snapshot()["status"], "running",
                         "a sibling failure reported the surviving generation as failed")
        b.complete()
        a_only = _Request("txt2img")
        a_only.run(lambda: gs.fail_generation("boom", generation_id=a_only.gid))
        self.assertEqual(gs.get_snapshot()["status"], "error")


class TestBucketRetention(unittest.TestCase):
    """SEV-4: the retention cap must never drop a bucket still being written."""

    def test_cap_never_evicts_a_live_bucket(self):
        requests = [_Request("txt2img") for _ in range(gs._MAX_BUCKETS + 1)]
        oldest = requests[0]

        oldest.run_in_executor(lambda: gs.add_warning("oldest still running",
                                                      code="quantization_fallback"))
        self.assertEqual(_codes(oldest.warnings()), ["quantization_fallback"],
                         "the oldest LIVE generation's bucket was evicted by the cap")
        for r in requests:
            r.complete()

    def test_finished_buckets_are_reclaimed(self):
        before = len(gs._buckets)
        for _ in range(gs._MAX_BUCKETS + 4):
            r = _Request("txt2img")
            r.complete()
        self.assertLessEqual(len(gs._buckets), gs._MAX_BUCKETS,
                             "finished buckets are not being reclaimed")
        self.assertGreaterEqual(len(gs._buckets), min(before, 1))


class TestVaeOverrideFailurePersists(unittest.TestCase):
    """Motivating case A."""

    def test_load_failure_is_recorded_with_code_intact(self):
        gid = gs.start_generation("txt2img")
        meta = apply_overrides(
            _FakePM(vae_exc=RuntimeError(f"No loadable VAE config.json found under: {_FAKE_VAE_PATH}")),
            {"vae": _FAKE_VAE_PATH, "te": None, "vae_kind": "autoencoder", "warnings": []},
        )
        warnings = gs.get_warnings(gid)
        gs.complete_generation(generation_id=gid)

        self.assertIn("vae_override_error", _codes(warnings))
        # The row still records the REQUESTED override; the warning is what
        # tells a later audit the decoder never saw it.
        self.assertEqual(meta.get("vae_override_path"), _FAKE_VAE_PATH)

    def test_no_vae_slot_is_no_longer_silent(self):
        """DiffusionPipelineManager.load_override_vae returned with no warning at all
        when the loaded model exposes no VAE slot."""
        from core.pipeline import DiffusionPipelineManager

        class _NoSlotPM:
            _override_vae_path = None

            def _vae_override_targets(self):
                return []

        gid = gs.start_generation("txt2img")
        DiffusionPipelineManager.load_override_vae(_NoSlotPM(), _FAKE_VAE_PATH)
        warnings = gs.get_warnings(gid)
        gs.complete_generation(generation_id=gid)
        self.assertIn("vae_override_error", _codes(warnings))

    def test_compat_gate_warnings_survive_the_pre_start_window(self):
        """plan_overrides() deliberately runs BEFORE start_generation() so a HARD
        mismatch is an HTTP 400 with no run opened. Its SOFT warnings used to be
        dropped by add_warning() for exactly that reason, so
        ``vae_override_warning`` never reached a single image row."""
        plan = {
            "vae": _FAKE_VAE_PATH,
            "te": None,
            "vae_kind": "autoencoder",
            "warnings": [{
                "code": "vae_override_warning",
                "message": ("VAE override 'run_x' was fine-tuned WITH ITS ENCODER: it encodes "
                            "to a different latent distribution than the base VAE"),
            }],
        }
        gid = gs.start_generation("txt2img")
        apply_overrides(_FakePM(), plan)
        warnings = gs.get_warnings(gid)
        gs.complete_generation(generation_id=gid)
        self.assertIn("vae_override_warning", _codes(warnings))

    def test_plan_overrides_captures_instead_of_dropping(self):
        """The capture buffer is what makes the replay above possible; verify it
        fills from the real _warn() used by the compat gate."""
        from api import generation_overrides as go

        sink = []
        with go._capture_warnings(sink):
            go._warn("VAE override applied with an unverified property: vae_class unknown",
                     code="vae_override_warning")
        self.assertEqual(_codes(sink), ["vae_override_warning"])
        # And the buffer is unset afterwards (no cross-request accumulation).
        go._warn("not captured", code="ignored")
        self.assertEqual(len(sink), 1)


class TestAttentionFallbackPersists(unittest.TestCase):
    """Motivating case B: the dedup must re-arm for every generation."""

    def _drive_flash_fallback(self):
        """Call the real flash backend with inputs it cannot serve.

        Returns the backend's output (None == it fell back to native), so the
        test exercises the same except: branches the denoise hits.
        """
        from core.attention import backends

        q = torch.zeros(1, 4, 2, 8, dtype=torch.float16)
        return backends._flash_attn(q, q, q)

    def test_fallback_is_recorded_on_every_generation(self):
        first = gs.start_generation("txt2img")
        self.assertIsNone(self._drive_flash_fallback(),
                          "flash backend unexpectedly succeeded on CPU fp16 input")
        w1 = gs.get_warnings(first)
        gs.complete_generation(generation_id=first)

        second = gs.start_generation("txt2img")
        self._drive_flash_fallback()
        w2 = gs.get_warnings(second)
        gs.complete_generation(generation_id=second)

        self.assertIn("attention_kernel_fallback", _codes(w1))
        self.assertIn("attention_kernel_fallback", _codes(w2),
                      "second generation lost the fallback warning to a "
                      "process-lifetime dedup")

    def test_fallback_outside_a_generation_does_not_burn_the_dedup(self):
        from core.attention import backends

        message = "unit-test kernel fallback probe"
        gs.complete_generation()
        backends._warn_kernel_fallback(message)   # no generation open
        gid = gs.start_generation("txt2img")
        backends._warn_kernel_fallback(message)
        warnings = gs.get_warnings(gid)
        gs.complete_generation(generation_id=gid)
        self.assertIn(message, [w["message"] for w in warnings])

    def test_fallback_is_not_repeated_within_one_generation(self):
        gid = gs.start_generation("txt2img")
        for _ in range(50):
            self._drive_flash_fallback()
        warnings = gs.get_warnings(gid)
        gs.complete_generation(generation_id=gid)
        self.assertEqual(
            len([c for c in _codes(warnings) if c == "attention_kernel_fallback"]), 1)


class TestPersistedWarningsAreShareable(unittest.TestCase):
    """What the PNG writer does with the list it just read (image_utils.py:
    ``redact_params_for_sharing(get_warnings())``)."""

    def test_code_survives_redaction_and_the_path_does_not(self):
        gid = gs.start_generation("txt2img")
        apply_overrides(
            _FakePM(vae_exc=RuntimeError(f"No loadable VAE config.json found under: {_FAKE_VAE_PATH}")),
            {"vae": _FAKE_VAE_PATH, "te": None, "vae_kind": "autoencoder", "warnings": []},
        )
        warnings = gs.get_warnings(gid)
        gs.complete_generation(generation_id=gid)

        shared = redact_params_for_sharing(warnings)
        self.assertIn("vae_override_error", _codes(shared))
        blob = repr(shared)
        for fragment in ("M:\\models", "M:/models", "private", "someone"):
            self.assertNotIn(fragment, blob, f"{fragment!r} leaked into a shared PNG chunk")

    def test_attention_fallback_message_survives_verbatim(self):
        """Failure mode B of the redactor: prose must not be rewritten, or the
        PNG asserts something false about what ran."""
        message = "flash_attn error (CUDA out of memory); falling back to native attention"
        shared = redact_params_for_sharing([{"code": "attention_kernel_fallback",
                                             "message": message}])
        self.assertEqual(shared[0]["message"], message)
        self.assertEqual(shared[0]["code"], "attention_kernel_fallback")


class TestPngChunkMatchesTheRow(unittest.TestCase):
    """SEV-3: the PNG is the artifact that travels, so it must not carry a
    concurrently-running generation's warnings — the writer takes an explicit
    generation id instead of reading the accumulator blind."""

    def setUp(self):
        from config.settings import settings
        from PIL import Image

        self._settings = settings
        self._saved_dir = settings.outputs_dir
        self._tmp = tempfile.TemporaryDirectory()
        settings.outputs_dir = self._tmp.name
        self._image = Image.new("RGB", (8, 8), (12, 34, 56))

    def tearDown(self):
        self._settings.outputs_dir = self._saved_dir
        self._tmp.cleanup()

    def _chunks(self, filename):
        from PIL import Image

        with Image.open(os.path.join(self._tmp.name, filename)) as im:
            return dict(im.text)

    def test_png_carries_its_own_generation_only(self):
        from utils.image_utils import save_image_with_metadata

        a = _Request("txt2img")
        b = _Request("img2img")
        a.run_in_executor(lambda: gs.add_warning("A's degradation", code="quantization_fallback"))
        b.run(lambda: gs.add_warning("B's degradation", code="unsupported_param"))

        params = {"prompt": "a test", "seed": 1, "steps": 4, "width": 8, "height": 8}
        filename = a.run(lambda: save_image_with_metadata(
            self._image, dict(params), "txt2img", generation_id=a.gid))
        chunks = self._chunks(filename)

        written = json.loads(chunks["effective_warnings"])
        self.assertEqual(_codes(written), ["quantization_fallback"],
                         "the PNG carried another generation's warnings")
        a.complete()
        b.complete()

    def test_caller_without_a_generation_writes_no_warnings_chunk(self):
        """The training-preview saver has no generation of its own; it used to
        stamp whatever the accumulator happened to hold into its PNG."""
        from utils.image_utils import save_image_with_metadata

        a = _Request("txt2img")
        a.run(lambda: gs.add_warning("A's degradation", code="quantization_fallback"))

        params = {"prompt": "preview", "seed": 2, "steps": 4, "width": 8, "height": 8}
        filename = save_image_with_metadata(self._image, dict(params), "txt2img")
        self.assertNotIn("effective_warnings", self._chunks(filename))
        a.complete()

    def test_png_and_row_agree_for_the_motivating_vae_case(self):
        from utils.image_utils import save_image_with_metadata

        a = _Request("txt2img")
        a.run(lambda: apply_overrides(
            _FakePM(vae_exc=RuntimeError(f"No loadable VAE config.json found under: {_FAKE_VAE_PATH}")),
            {"vae": _FAKE_VAE_PATH, "te": None, "vae_kind": "autoencoder", "warnings": []},
        ))
        row = a.warnings()          # what params_for_db["effective_warnings"] gets
        params = {"prompt": "x", "seed": 3, "steps": 4, "width": 8, "height": 8,
                  "vae_override_path": _FAKE_VAE_PATH}
        filename = a.run(lambda: save_image_with_metadata(
            self._image, dict(params), "txt2img", generation_id=a.gid))
        chunks = self._chunks(filename)
        a.complete()

        written = json.loads(chunks["effective_warnings"])
        self.assertEqual(_codes(row), _codes(written), "PNG and DB row disagree")
        self.assertIn("vae_override_error", _codes(written))
        blob = json.dumps(chunks)
        for fragment in ("M:\\\\models", "M:/models", "private", "someone"):
            self.assertNotIn(fragment, blob,
                             f"{fragment!r} leaked into a PNG text chunk")


if __name__ == "__main__":
    unittest.main(verbosity=2)
