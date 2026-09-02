"""On-demand training samples ("sample now").

The trainer is a subprocess with no stdin, no signal channel and no socket; the
only inbound controls are the `.stop_training` flag file and the preview
file-RPC. This adds a third file-RPC whose requests are executed at the START of
the scheduled-sample block, so an on-demand sample is produced by the same code
as a scheduled one (same save path, same metadata, same TensorBoard write, same
onthefly_gpu text-encoder re-home).

Covered here: the RPC round trip and its atomic / delete-before-process / result
-last properties, the queue cap, at most one request per batch, a request
surviving a phase that never polls, stale files cleared before the next run is
spawned, the on-demand filename not colliding with a scheduled sample at the
same step AND the read API returning both, the seed being concrete, and
`_dispatch_sample` returning None being recorded as failed.

Nothing here loads a model, a checkpoint or a database.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/training_sample_on_demand_test.py -v
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.training import training_sample_rpc as rpc  # noqa: E402

BASE_TRAINER_SRC = (BACKEND / "core" / "training" / "base_trainer.py").read_text(
    encoding="utf-8"
)
ROUTES_SRC = (BACKEND / "api" / "routes.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------

def test_request_round_trip(tmp_path):
    payload = rpc.queue_request(tmp_path, seed=1234)
    assert payload["seed"] == 1234
    assert rpc.request_path(tmp_path, payload["request_id"]).exists()

    claimed = rpc.claim_next_request(tmp_path)
    assert claimed["request_id"] == payload["request_id"]
    assert claimed["seed"] == 1234
    # Delete BEFORE process: the file is gone the moment it is claimed, so a
    # crash mid-generation cannot replay it.
    assert not rpc.request_path(tmp_path, payload["request_id"]).exists()
    assert rpc.claim_next_request(tmp_path) is None


def test_request_write_is_atomic(tmp_path):
    """No partially-written request is ever visible under the real name: the
    payload is written to a .tmp sibling and renamed."""
    rpc.queue_request(tmp_path, seed=7)
    names = sorted(p.name for p in tmp_path.iterdir())
    assert len(names) == 1 and not names[0].endswith(".tmp")
    assert json.loads((tmp_path / names[0]).read_text(encoding="utf-8"))["seed"] == 7


def test_result_is_written_last_and_complete(tmp_path):
    """A reader polling on the result file sees a complete record; the PNG it
    names is already in samples/ under its own name."""
    rpc.write_result(tmp_path, "abc123", {
        "ok": True, "step": 40, "files": ["step_000040_sample_0_ondemand_abc123.png"],
        "seeds": [11], "architecture": "sdxl", "error": None, "notes": [],
    })
    p = rpc.result_path(tmp_path, "abc123")
    assert p.exists() and not any(x.name.endswith(".tmp") for x in tmp_path.iterdir())
    rec = json.loads(p.read_text(encoding="utf-8"))
    assert rec["ok"] is True and rec["request_id"] == "abc123"
    assert "completed_at" in rec
    assert rpc.list_results(tmp_path)[0]["step"] == 40


def test_malformed_request_is_claimed_and_discarded(tmp_path):
    rpc.request_path(tmp_path, "deadbeef").write_text("{not json", encoding="utf-8")
    assert rpc.claim_next_request(tmp_path) is None
    assert not rpc.request_path(tmp_path, "deadbeef").exists()


# ---------------------------------------------------------------------------
# Throttle
# ---------------------------------------------------------------------------

def test_queue_is_capped(tmp_path):
    for _ in range(rpc.MAX_PENDING_REQUESTS):
        rpc.queue_request(tmp_path, seed=1)
    with pytest.raises(rpc.SampleQueueFullError) as exc:
        rpc.queue_request(tmp_path, seed=1)
    assert str(rpc.MAX_PENDING_REQUESTS) in str(exc.value)
    assert len(rpc.list_pending_requests(tmp_path)) == rpc.MAX_PENDING_REQUESTS
    # Room again once the trainer has taken one.
    rpc.claim_next_request(tmp_path)
    rpc.queue_request(tmp_path, seed=1)


def test_claim_takes_exactly_one_and_oldest_first(tmp_path):
    ids = []
    for _ in range(3):
        ids.append(rpc.queue_request(tmp_path, seed=1)["request_id"])
        time.sleep(0.01)
    first = rpc.claim_next_request(tmp_path)
    assert first["request_id"] == ids[0]
    assert len(rpc.list_pending_requests(tmp_path)) == 2


def test_trainer_claims_at_most_one_per_batch():
    """The loop calls the claim helper once per batch and appends AT MOST one
    on-demand job, so N queued requests cannot run back to back."""
    assert BASE_TRAINER_SRC.count("self._claim_on_demand_sample_request()") == 1
    idx = BASE_TRAINER_SRC.index("on_demand_request = self._claim_on_demand_sample_request()")
    block = BASE_TRAINER_SRC[idx:idx + 500]
    assert "sample_jobs.append((global_step, on_demand_request))" in block
    assert "for _" not in block and "while " not in block


def test_claim_helper_rechecks_the_stop_flag():
    idx = BASE_TRAINER_SRC.index("def _claim_on_demand_sample_request")
    body = BASE_TRAINER_SRC[idx:idx + 900]
    assert '".stop_training"' in body
    assert body.index('".stop_training"') < body.index("claim_next_request")


def _stub_trainer(tmp_path, stop_flag=False, arch_name="sdxl", run_id=None):
    from core.training.base_trainer import BaseTrainer
    if stop_flag:
        (tmp_path / ".stop_training").write_text("")
    stub = SimpleNamespace(
        output_dir=tmp_path,
        log_prefix="[Test]",
        run_id=run_id,
        arch=SimpleNamespace(name=arch_name),
    )
    for name in ("_claim_on_demand_sample_request", "_record_on_demand_sample_result"):
        setattr(stub, name,
                (lambda m: lambda *a, **k: getattr(BaseTrainer, m)(stub, *a, **k))(name))
    return stub


def test_nothing_is_claimed_once_a_stop_was_requested(tmp_path):
    """A queued request would delay the stop by a whole generation."""
    rpc.queue_request(tmp_path, seed=1)
    stub = _stub_trainer(tmp_path, stop_flag=True)
    assert stub._claim_on_demand_sample_request() is None
    # And it is still pending, not swallowed.
    assert len(rpc.list_pending_requests(tmp_path)) == 1


def test_claim_returns_the_request_when_no_stop_is_pending(tmp_path):
    rpc.queue_request(tmp_path, seed=99)
    stub = _stub_trainer(tmp_path)
    assert stub._claim_on_demand_sample_request()["seed"] == 99


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def test_request_has_no_ttl_and_survives_a_phase_that_never_polls(tmp_path):
    """Dataset scan / bucketing / latent+TE caching poll only the stop flag, and
    can run for a long time. An hours-old request must still be claimable."""
    payload = rpc.queue_request(tmp_path, seed=5)
    old = time.time() - 6 * 3600
    p = rpc.request_path(tmp_path, payload["request_id"])
    import os
    os.utime(p, (old, old))
    assert rpc.list_pending_requests(tmp_path) == [p]
    assert rpc.claim_next_request(tmp_path)["request_id"] == payload["request_id"]


def test_no_expiry_constant_exists_for_requests():
    src = (BACKEND / "core" / "training" / "training_sample_rpc.py").read_text(
        encoding="utf-8")
    assert "STALE_TIMEOUT" not in src


def test_clear_all_removes_requests_and_results(tmp_path):
    rpc.queue_request(tmp_path, seed=1)
    rpc.write_result(tmp_path, "x1", {"ok": True})
    (tmp_path / "keep.txt").write_text("x")
    assert rpc.clear_all(tmp_path) == 2
    assert rpc.list_pending_requests(tmp_path) == []
    assert rpc.list_results(tmp_path) == []
    assert (tmp_path / "keep.txt").exists()


def test_stale_requests_are_cleared_before_the_next_run_is_spawned():
    """Same place the stale `.stop_training` flag is cleared: pre-spawn, so a
    request queued during a long caching phase of the NEW run is not the one
    that gets wiped."""
    src = (BACKEND / "core" / "training" / "training_process.py").read_text(
        encoding="utf-8")
    assert "from core.training.training_sample_rpc import clear_all" in src
    assert src.index("clear_all") < src.index("await asyncio.create_subprocess_exec")
    # NOT in base_trainer.train(), which reaches its stop-flag cleanup only
    # after dataset loading and caching.
    assert "clear_all" not in BASE_TRAINER_SRC


# ---------------------------------------------------------------------------
# Filenames
# ---------------------------------------------------------------------------

def test_on_demand_filename_does_not_collide_with_a_scheduled_one():
    scheduled = rpc.sample_filename(4210, 0)
    on_demand = rpc.sample_filename(4210, 0, "9f1c2ab4d7e05613")
    assert scheduled == "step_004210_sample_0.png"
    assert on_demand == "step_004210_sample_0_ondemand_9f1c2ab4d7e05613.png"
    assert scheduled != on_demand
    # Both are found by the listing endpoint's glob.
    for name in (scheduled, on_demand):
        assert Path(name).match("step_*_sample_*.png")


def _read_api_pattern():
    """The regex the samples listing endpoint actually compiles."""
    match = re.search(r'pattern = re\.compile\(r"([^"]+)"\)', ROUTES_SRC)
    assert match, "samples listing pattern not found"
    return re.compile(match.group(1))


def test_read_api_parses_both_filename_forms():
    pattern = _read_api_pattern()
    scheduled = pattern.match(rpc.sample_filename(4210, 0))
    on_demand = pattern.match(rpc.sample_filename(4210, 1, "abc123def456"))
    assert scheduled and scheduled.groups() == ("004210", "0", None)
    assert on_demand and on_demand.groups() == ("004210", "1", "abc123def456")


def test_read_api_returns_both_grouped_under_the_same_step(tmp_path):
    """The endpoint's parse+group logic, run over a real samples directory."""
    pattern = _read_api_pattern()
    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()
    for name in (rpc.sample_filename(4210, 0),
                 rpc.sample_filename(4210, 0, "abc123"),
                 ".step0_done"):
        (samples_dir / name).write_text("x")

    by_step = {}
    for file in samples_dir.glob("step_*_sample_*.png"):
        m = pattern.match(file.name)
        assert m, file.name
        by_step.setdefault(int(m.group(1)), []).append(
            {"sample_index": int(m.group(2)), "on_demand": m.group(3) is not None,
             "path": file.name, "request_id": m.group(3)})
    assert set(by_step) == {4210}
    images = sorted(by_step[4210],
                    key=lambda x: (x["on_demand"], x["sample_index"], x["path"]))
    assert [i["on_demand"] for i in images] == [False, True]
    assert images[0]["path"] == "step_004210_sample_0.png"
    assert images[1]["request_id"] == "abc123"


def test_trainer_uses_the_shared_filename_builder():
    """One place decides the name, so the writer and the reader cannot drift."""
    assert "sample_rpc.sample_filename(" in BASE_TRAINER_SRC
    assert 'f"step_{sample_step:06d}_sample_{sample_idx}.png"' not in BASE_TRAINER_SRC


def test_step0_marker_is_not_set_by_an_on_demand_sample():
    """The marker records that THIS run wrote its step-0 verification sample; an
    on-demand sample that happens to land on the first batch must not claim it."""
    loop = BASE_TRAINER_SRC.index("for sample_step, on_demand_request in sample_jobs:")
    idx = BASE_TRAINER_SRC.index("self._mark_step0_sample_done()", loop)
    guard = BASE_TRAINER_SRC[:idx].rsplit("\n", 2)[-2].strip()
    assert guard.startswith("if on_demand_request is None and sample_step == 0")


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("configured", [-1, "-1", None, "nonsense"])
def test_seed_is_resolved_to_a_concrete_value(configured):
    """`seed < 0` reaches the arch ops as generator=None, which draws from the
    global torch RNG (zimage_ops, sd_sdxl ancestral samplers). An on-demand
    sample must not perturb the training stream."""
    for _ in range(20):
        seed = rpc.resolve_seed(configured)
        assert isinstance(seed, int) and 0 <= seed < 2 ** 32


def test_a_configured_seed_is_passed_through():
    assert rpc.resolve_seed(1234) == 1234
    assert rpc.resolve_seed("1234") == 1234
    assert rpc.resolve_seed(0) == 0


def test_queued_request_always_carries_a_concrete_seed(tmp_path):
    payload = rpc.queue_request(tmp_path, seed=rpc.resolve_seed(-1))
    assert payload["seed"] >= 0
    assert rpc.claim_next_request(tmp_path)["seed"] >= 0


def test_a_seed_read_off_disk_is_re_resolved_before_use():
    """The request file is not trusted: a negative value in it would reach the
    arch ops as generator=None and draw from the training RNG."""
    idx = BASE_TRAINER_SRC.index("for sample_step, on_demand_request in sample_jobs:")
    block = BASE_TRAINER_SRC[idx:idx + 2000]
    assert '_req_seed = int(on_demand_request.get("seed", -1))' in block
    assert "actual_seed = self._resolve_sample_seed(_req_seed)" in block


@pytest.mark.parametrize("bad", [-1, -99])
def test_the_sentinel_cannot_survive_the_resolver(bad):
    from core.training.base_trainer import BaseTrainer
    assert BaseTrainer._resolve_sample_seed(bad) >= 0


def test_api_resolves_the_seed_before_writing_the_request():
    assert "seed=resolve_seed(_configured_sample_seed(run))" in ROUTES_SRC


# ---------------------------------------------------------------------------
# Failure contract
# ---------------------------------------------------------------------------

def test_dispatch_returning_none_is_recorded_as_failed(tmp_path):
    stub = _stub_trainer(tmp_path, arch_name="ideogram4")
    stub._record_on_demand_sample_result(
        "r1", step=12, files=[], seeds=[],
        error="the training-sample path for architecture 'ideogram4' returned no image")
    rec = rpc.list_results(tmp_path)[0]
    assert rec["ok"] is False
    assert "ideogram4" in rec["error"]
    assert rec["step"] == 12 and rec["files"] == []


def test_success_is_recorded_with_the_written_filenames(tmp_path):
    stub = _stub_trainer(tmp_path, arch_name="zimage")
    name = rpc.sample_filename(40, 0, "r3")
    stub._record_on_demand_sample_result("r3", step=40, files=[name], seeds=[7],
                                         error=None)
    rec = rpc.list_results(tmp_path)[0]
    assert rec["ok"] is True and rec["files"] == [name] and rec["seeds"] == [7]
    # zimage raises rather than returning a blank image, so no qualification.
    assert rec["notes"] == []


@pytest.mark.parametrize("arch", ["sd15", "sdxl"])
def test_blank_image_architectures_are_stated_in_the_result(arch, tmp_path):
    """sd_sdxl_ops returns a blank white image on failure, so a written PNG is
    not proof of success there. Recorded factually; not fixed here."""
    stub = _stub_trainer(tmp_path, arch_name=arch)
    stub._record_on_demand_sample_result(
        "r4", step=40, files=[rpc.sample_filename(40, 0, "r4")], seeds=[7], error=None)
    rec = rpc.list_results(tmp_path)[0]
    assert rec["ok"] is True
    assert len(rec["notes"]) == 1 and arch in rec["notes"][0]
    assert "blank white image" in rec["notes"][0]


@pytest.mark.parametrize("arch", ["zimage", "flux2", "sensenova", None])
def test_other_architectures_carry_no_such_note(arch):
    assert rpc.blank_on_failure_note(arch) is None


def test_an_on_demand_failure_does_not_kill_the_run_but_a_scheduled_one_still_does():
    """The job body is wrapped: a scheduled sample re-raises (unchanged
    behaviour), an on-demand one is recorded and training continues. There is no
    enclosing try between this block and the batch loop, so without the wrapper
    a button press could abort a multi-hour run."""
    idx = BASE_TRAINER_SRC.index("for sample_step, on_demand_request in sample_jobs:")
    block = BASE_TRAINER_SRC[idx:BASE_TRAINER_SRC.index(
        "# Note: Progress callback is now called per-MNT-iteration", idx)]
    assert "try:" in block and "except Exception as sample_err:" in block
    assert "if on_demand_id is None:\n" in block
    raise_idx = block.index("                                raise\n")
    assert block.index("if on_demand_id is None:") < raise_idx
    # Every generating call sits inside the wrapper.
    for call in ("self._dispatch_sample(", "self._save_sample_with_metadata(",
                 "self.writer.add_image("):
        assert block.index("try:") < block.index(call) < block.index(
            "except Exception as sample_err:")


def test_the_result_is_recorded_in_a_finally():
    """claim_next_request already unlinked the request, so a raise anywhere in
    the job must not leave it neither pending nor resulted."""
    idx = BASE_TRAINER_SRC.index("for sample_step, on_demand_request in sample_jobs:")
    block = BASE_TRAINER_SRC[idx:BASE_TRAINER_SRC.index(
        "# Note: Progress callback is now called per-MNT-iteration", idx)]
    assert block.count("self._record_on_demand_sample_result(") == 1
    assert block.index("finally:") < block.index("self._record_on_demand_sample_result(")


def test_a_request_is_left_in_place_for_another_run_sharing_the_directory(tmp_path):
    """Two runs with the same run_name share an output_dir -- the same reason
    the step-0 marker records a run id instead of trusting the file's existence."""
    a = rpc.queue_request(tmp_path, seed=1, run_id=41)
    b = rpc.queue_request(tmp_path, seed=2, run_id=42)

    claimed = rpc.claim_next_request(tmp_path, 42)
    assert claimed["request_id"] == b["request_id"]
    # Run 41's request is untouched, not consumed and not deleted.
    assert rpc.request_path(tmp_path, a["request_id"]).exists()
    assert rpc.claim_next_request(tmp_path, 42) is None
    assert rpc.claim_next_request(tmp_path, 41)["request_id"] == a["request_id"]


def test_a_request_without_a_run_id_is_claimable_by_anyone(tmp_path):
    """Nothing written before this field existed may wedge the directory."""
    rpc.request_path(tmp_path, "legacy00").write_text(
        json.dumps({"request_id": "legacy00", "seed": 5}), encoding="utf-8")
    assert rpc.claim_next_request(tmp_path, 99)["request_id"] == "legacy00"


def test_the_queue_view_and_the_cap_are_per_run(tmp_path):
    for _ in range(rpc.MAX_PENDING_REQUESTS):
        rpc.queue_request(tmp_path, seed=1, run_id=41)
    # Run 42's queue is empty and its cap is not consumed by run 41.
    assert rpc.pending_requests(tmp_path, 42) == []
    rpc.queue_request(tmp_path, seed=1, run_id=42)
    with pytest.raises(rpc.SampleQueueFullError):
        rpc.queue_request(tmp_path, seed=1, run_id=41)
    rpc.write_result(tmp_path, "x", {"ok": True, "run_id": 41})
    assert rpc.list_results(tmp_path, 42) == []
    assert len(rpc.list_results(tmp_path, 41)) == 1


def test_the_trainer_claims_and_records_under_its_own_run_id(tmp_path):
    rpc.queue_request(tmp_path, seed=3, run_id=41)
    stub = _stub_trainer(tmp_path, run_id=42)
    assert stub._claim_on_demand_sample_request() is None
    stub = _stub_trainer(tmp_path, run_id=41)
    assert stub._claim_on_demand_sample_request()["seed"] == 3
    stub._record_on_demand_sample_result("q1", step=1, files=["f.png"], seeds=[3],
                                         error=None)
    assert rpc.list_results(tmp_path, 41)[0]["run_id"] == 41


def test_arch_detection_is_not_reparsed_on_every_queue_poll(tmp_path):
    """The queue endpoint is polled every 5s while training; opening a multi-GB
    safetensors and listing its keys there is not acceptable."""
    from api import routes
    model = tmp_path / "model.safetensors"
    model.write_bytes(b"x" * 16)
    calls = []

    import core.training.training_config as tc
    original = tc._detect_arch
    tc._detect_arch = lambda p: (calls.append(p), "sdxl")[1]
    try:
        routes._SAMPLE_ARCH_CACHE.clear()
        assert routes._detect_arch_cached(str(model)) == "sdxl"
        for _ in range(10):
            routes._detect_arch_cached(str(model))
        assert len(calls) == 1
        # A replaced checkpoint is re-detected rather than served from cache.
        model.write_bytes(b"y" * 32)
        routes._detect_arch_cached(str(model))
        assert len(calls) == 2
    finally:
        tc._detect_arch = original
        routes._SAMPLE_ARCH_CACHE.clear()


def test_results_are_pruned_but_the_recent_ones_survive(tmp_path):
    for i in range(rpc.MAX_KEPT_RESULTS + 5):
        rpc.write_result(tmp_path, f"r{i:03d}", {"ok": True, "step": i})
        time.sleep(0.002)
    results = rpc.list_results(tmp_path)
    assert len(results) == rpc.MAX_KEPT_RESULTS
    assert results[0]["step"] == rpc.MAX_KEPT_RESULTS + 4   # newest first


# ---------------------------------------------------------------------------
# Injection point and API surface
# ---------------------------------------------------------------------------

def test_on_demand_goes_through_the_scheduled_sample_block():
    """Not a second sampling path: the job loop wraps the existing block, so the
    save call, the metadata, the TensorBoard write and the onthefly_gpu TE
    re-home are the same code for both kinds of sample."""
    idx = BASE_TRAINER_SRC.index("for sample_step, on_demand_request in sample_jobs:")
    block = BASE_TRAINER_SRC[idx:BASE_TRAINER_SRC.index(
        "# Note: Progress callback is now called per-MNT-iteration", idx)]
    assert block.count("self._dispatch_sample(") == 1
    assert block.count("self._save_sample_with_metadata(") == 1
    assert block.count("self.writer.add_image(") == 1
    assert 'if text_encoding_mode == "onthefly_gpu":' in block
    # It runs after the MNT window and the checkpoint save, where the scheduled
    # block already was -- not between MNT iterations.
    assert BASE_TRAINER_SRC.index("del latents, text_embeddings") < idx
    assert BASE_TRAINER_SRC.index("if interval_due(global_step, save_every_n_steps):") < idx


def test_post_is_fire_and_forget():
    """A batch can outlast any sane HTTP timeout, so the route must not wait."""
    idx = ROUTES_SRC.index("async def queue_training_sample(")
    body = ROUTES_SRC[idx:ROUTES_SRC.index("async def get_training_sample_queue(")]
    assert "status_code=202" in ROUTES_SRC[idx - 200:idx]
    for waiting in ("await asyncio.sleep", "_await_", "while "):
        assert waiting not in body, waiting


def test_endpoints_are_documented_in_openapi():
    import yaml as _yaml
    spec = _yaml.safe_load((REPO / "openapi.yaml").read_text(encoding="utf-8"))
    post = spec["paths"]["/training/runs/{run_id}/sample"]["post"]
    assert set(post["responses"]) >= {"202", "400", "404", "409", "429", "500"}
    assert post["responses"]["202"]["content"]["application/json"]["schema"]["$ref"] \
        == "#/components/schemas/TrainingSampleRequestAccepted"
    get = spec["paths"]["/training/runs/{run_id}/sample-queue"]["get"]
    assert get["responses"]["200"]["content"]["application/json"]["schema"]["$ref"] \
        == "#/components/schemas/TrainingSampleQueueResponse"
    schemas = spec["components"]["schemas"]
    assert schemas["TrainingSampleRequestAccepted"]["properties"]["seed"]
    assert schemas["TrainingSampleQueueResponse"]["properties"]["unsupported_reason"]
    # The cap and the latency are stated where a caller will read them.
    assert "429" in post["responses"]
    assert "minutes" in post["description"]


@pytest.mark.parametrize("arch", ["ideogram4", "minimax_h3", "acestep"])
def test_architectures_that_cannot_sample_are_refused(arch):
    from api.arch_capabilities import training_feature_unsupported_reason
    assert training_feature_unsupported_reason(arch, "training_samples", "lora")


def test_vae_decoder_runs_are_refused():
    from api.routes import _training_sample_support
    run = SimpleNamespace(training_method="vae_decoder",
                          base_model_path="x.safetensors", config_yaml=None)
    arch, reason = _training_sample_support(run)
    assert arch is None and "no denoiser" in reason


def test_configured_sample_seed_is_read_from_the_run_yaml():
    from api.routes import _configured_sample_seed
    yaml_text = (
        "config:\n  process:\n    - sample:\n        seed: 4242\n        width: 512\n")
    assert _configured_sample_seed(
        SimpleNamespace(config_yaml=yaml_text)) == 4242
    assert _configured_sample_seed(SimpleNamespace(config_yaml=None)) == -1
    assert _configured_sample_seed(SimpleNamespace(config_yaml="{{ not yaml")) == -1
