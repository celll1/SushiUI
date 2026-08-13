"""MiniMax-H3 text-encoder reference bank, over the API.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_h3_reference_bank_api_test.py -v

A real build loads a 25-48 GB encoder for ~5 minutes and 14-24 GiB of host RSS,
so nothing here builds one: a stub encoder produces 9-wide rows for a 3-prompt
suite, and the "existing bank" cases fabricate one. What is covered is the parts
that would otherwise only be exercised by that expensive run -- the two
refusals, the mutual exclusion with generation in BOTH directions, the progress
the user watches, and that a cancelled build leaves nothing behind.
"""

import asyncio
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest  # noqa: E402
import torch  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from api import minimax_h3_reference_bank_job as job  # noqa: E402
from api import routes  # noqa: E402
from core.models.minimax_h3 import te_agreement as ta  # noqa: E402
from core.models.minimax_h3.te_projection import (  # noqa: E402
    load_te_projection, read_te_projection_spec,
)

RELEASED = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
CONVERTED = "qwen3vl_4b_heretic_tap24_bf16.safetensors"
PROJECTION = "mmh3-4b-clipproj-celeb-mlp.safetensors"

D_IN, TEXT_DIM = 6, 9

_LOAD_SUITE = ta.load_suite


# ---------------------------------------------------------------------------
# Fixtures: a store, a suite, a stub encoder, and a "loaded" H3 model
# ---------------------------------------------------------------------------

class _Tokenizer:
    def __call__(self, text, add_special_tokens=True):
        ids = [(abs(hash(word)) % 900) + 10 for word in text.split()]
        return {"input_ids": [1] + ids + [2] if add_special_tokens else ids}


class _Encoder:
    """One deterministic hidden row per token, at a chosen width."""

    def __init__(self, width):
        self.width = width

    def encode(self, token_ids):
        generator = torch.Generator().manual_seed(sum(token_ids) % 10007)
        return torch.randn(1, len(token_ids), self.width, generator=generator).to(torch.bfloat16)


@pytest.fixture(autouse=True)
def clean_job():
    job._job = None
    job._cancel.clear()
    yield
    job._cancel.set()  # release a worker still blocked in a failed test
    job._job = None
    job._cancel.clear()


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Point the whole engine's store at tmp_path instead of settings.cache_dir."""
    directory = tmp_path / "store"
    monkeypatch.setattr(ta, "store_dir", lambda root=None: directory)
    return directory


@pytest.fixture
def suite(tmp_path, monkeypatch):
    path = tmp_path / "suite.json"
    path.write_text(json.dumps({
        "version": "test-suite-v1", "composite_target_tokens": 6,
        "prompts": ["alpha beta", "gamma delta epsilon", "zeta"]}), encoding="utf-8")
    built = _LOAD_SUITE(str(path))
    monkeypatch.setattr(ta, "load_suite", lambda path=None: built)
    return built


@pytest.fixture
def stub_encode(monkeypatch):
    from core.models.minimax_h3 import h3_pipeline_ops as ops

    monkeypatch.setattr(ops, "encode_presentation",
                        lambda encoder, token_ids, **kwargs: encoder.encode(list(token_ids)))


def _projection_file(directory):
    generator = torch.Generator().manual_seed(7)
    tensors = {
        "W": torch.randn(D_IN, TEXT_DIM, generator=generator),
        "mean_in": torch.randn(D_IN, generator=generator),
        "std_in": torch.rand(D_IN, generator=generator) + 0.5,
        "mean_out": torch.randn(TEXT_DIM, generator=generator),
        "std_out": torch.rand(TEXT_DIM, generator=generator) + 0.5,
        "sink_out": torch.randn(TEXT_DIM, generator=generator),
        "mlp.0.weight": torch.randn(4, D_IN, generator=generator),
        "mlp.0.bias": torch.randn(4, generator=generator),
        "mlp.2.weight": torch.randn(TEXT_DIM, 4, generator=generator),
        "mlp.2.bias": torch.randn(TEXT_DIM, generator=generator),
    }
    path = str(directory / PROJECTION)
    save_file(tensors, path, metadata={"d_in": str(D_IN), "d_out": str(TEXT_DIM), "tap": "24",
                                       "mlp_hidden": "4", "mlp_depth": "1"})
    return path


class _Manager:
    """The three attributes the job reads off the pipeline manager."""

    def __init__(self, components):
        self.current_model_info = {"type": "minimax_h3", "source": "M:/model/minimax_h3"}
        self.minimax_h3_components = components
        self._load_model_lock = threading.Lock()


def _released_components(tmp_path, *, encoder=None):
    path = tmp_path / RELEASED
    if not path.exists():
        save_file({"w": torch.zeros(4, 4)}, str(path))
    return {
        "type": "minimax_h3",
        "text_encoder": encoder if encoder is not None else _Encoder(TEXT_DIM),
        "tokenizer": _Tokenizer(),
        "text_encoder_path": str(path),
        "te_projection": None,
        "transformer_config": {"text_dim": TEXT_DIM},
        "dit_path": None,
        "official_dir": None,
    }


def _substitute_components(tmp_path):
    path = tmp_path / CONVERTED
    if not path.exists():
        save_file({"w": torch.zeros(4, 4)}, str(path))
    return {
        "type": "minimax_h3",
        "text_encoder": _Encoder(D_IN),
        "tokenizer": _Tokenizer(),
        "text_encoder_path": str(path),
        "te_projection": load_te_projection(read_te_projection_spec(_projection_file(tmp_path))),
        "transformer_config": {"text_dim": TEXT_DIM},
        "dit_path": None,
        "official_dir": None,
    }


def _install(monkeypatch, manager):
    monkeypatch.setattr(routes, "pipeline_manager", manager)
    return manager


def _start(path):
    return asyncio.run(routes.start_minimax_h3_reference_bank(
        routes.MiniMaxH3ReferenceBankRequest(text_encoder_path=path)))


def _status(model_path=None):
    return asyncio.run(routes.get_minimax_h3_te_agreement(model_path))


def _await_job(timeout=20.0):
    """Poll until the worker settles, the way a client polls GET."""
    for _ in range(int(timeout / 0.02)):
        snapshot = job._job_snapshot()
        if snapshot.get("state") in {"completed", "failed", "cancelled", "idle"}:
            return snapshot
        time.sleep(0.02)
    return job._job_snapshot()


@pytest.fixture
def no_tree_scan(monkeypatch):
    """The status endpoint's tree scan, without a tree on disk."""
    from core.models.minimax_h3 import loader

    monkeypatch.setattr(loader, "describe_minimax_h3_text_encoder_choices",
                        lambda model_path: {"text_encoders": [], "clip_projections": []})


# ---------------------------------------------------------------------------
# 1. Mutual exclusion with generation, both directions
# ---------------------------------------------------------------------------

def test_a_build_is_refused_while_a_generation_runs(tmp_path, monkeypatch, store, suite):
    from api import generation_status

    _install(monkeypatch, _Manager(_released_components(tmp_path)))
    monkeypatch.setattr(generation_status, "get_snapshot", lambda: {"status": "running"})

    with pytest.raises(HTTPException) as excinfo:
        _start(str(tmp_path / RELEASED))
    assert excinfo.value.status_code == 409
    assert "generation is running" in excinfo.value.detail
    assert job._job_snapshot() == {"state": "idle"}, "a refused build must not create a job"


def test_a_generation_is_refused_while_a_build_runs(tmp_path, monkeypatch, store, suite,
                                                    stub_encode):
    """The worker holds the lifecycle gate, which is what refuses the generation."""
    from api import generation_status
    from core.models.minimax_h3 import h3_pipeline_ops as ops

    entered, release = threading.Event(), threading.Event()
    encoder = _Encoder(TEXT_DIM)

    def blocking(text_encoder, token_ids, **kwargs):
        entered.set()
        release.wait(20)
        return encoder.encode(list(token_ids))

    monkeypatch.setattr(ops, "encode_presentation", blocking)
    _install(monkeypatch, _Manager(_released_components(tmp_path, encoder=encoder)))

    _start(str(tmp_path / RELEASED))
    assert entered.wait(10), "the build never reached the encoder"
    try:
        with pytest.raises(HTTPException) as excinfo:
            generation_status.start_generation("txt2img")
        assert excinfo.value.status_code == 409
        assert "MiniMax-H3 reference bank build" in excinfo.value.detail
    finally:
        release.set()
    assert _await_job()["state"] == "completed"


def test_a_second_build_is_refused_while_one_runs(tmp_path, monkeypatch, store, suite,
                                                  stub_encode):
    from core.models.minimax_h3 import h3_pipeline_ops as ops

    entered, release = threading.Event(), threading.Event()
    encoder = _Encoder(TEXT_DIM)

    def blocking(text_encoder, token_ids, **kwargs):
        entered.set()
        release.wait(20)
        return encoder.encode(list(token_ids))

    monkeypatch.setattr(ops, "encode_presentation", blocking)
    _install(monkeypatch, _Manager(_released_components(tmp_path, encoder=encoder)))

    _start(str(tmp_path / RELEASED))
    assert entered.wait(10)
    try:
        with pytest.raises(HTTPException) as excinfo:
            _start(str(tmp_path / RELEASED))
        assert excinfo.value.status_code == 409
        assert "already running" in excinfo.value.detail
    finally:
        release.set()
    _await_job()


# ---------------------------------------------------------------------------
# 2. The two refusals the engine owns
# ---------------------------------------------------------------------------

def test_a_substitute_encoder_cannot_be_the_reference(tmp_path, monkeypatch, store, suite):
    _install(monkeypatch, _Manager(_substitute_components(tmp_path)))

    with pytest.raises(HTTPException) as excinfo:
        _start(str(tmp_path / CONVERTED))
    assert excinfo.value.status_code == 400
    assert "cannot be one" in excinfo.value.detail
    assert PROJECTION in excinfo.value.detail


def test_naming_an_encoder_other_than_the_loaded_one_is_refused(tmp_path, monkeypatch,
                                                                store, suite):
    _install(monkeypatch, _Manager(_released_components(tmp_path)))

    with pytest.raises(HTTPException) as excinfo:
        _start("M:/somewhere/else/qwen3vl_32b_other.safetensors")
    assert excinfo.value.status_code == 400
    assert "Load the encoder you are naming" in excinfo.value.detail
    assert not (store / "banks").exists()


def test_a_non_h3_model_has_no_bank_to_build(monkeypatch, store, suite):
    manager = _Manager(None)
    manager.current_model_info = {"type": "sdxl", "source": "M:/model/sdxl/x.safetensors"}
    _install(monkeypatch, manager)

    with pytest.raises(HTTPException) as excinfo:
        _start("whatever.safetensors")
    assert excinfo.value.status_code == 400
    assert "'sdxl'" in excinfo.value.detail
    assert _status()["supported"] is False


# ---------------------------------------------------------------------------
# 3. Progress and completion
# ---------------------------------------------------------------------------

def test_a_build_reports_progress_and_stores_the_bank(tmp_path, monkeypatch, store, suite,
                                                      stub_encode, no_tree_scan):
    seen = []
    real = ta.build_reference_bank

    def watched(components, **kwargs):
        progress = kwargs.pop("progress")

        def record(done, total, name):
            seen.append((done, total, name))
            progress(done, total, name)

        return real(components, progress=record, **kwargs)

    monkeypatch.setattr(ta, "build_reference_bank", watched)
    components = _released_components(tmp_path)
    _install(monkeypatch, _Manager(components))

    started = _start(str(tmp_path / RELEASED))
    assert started["state"] == "running" and started["reference"] == RELEASED
    finished = _await_job()

    assert finished["state"] == "completed"
    corpus = len(ta.build_corpus(_Tokenizer(), suite))
    assert finished["processed"] == finished["total"] == corpus
    assert [entry[0] for entry in seen] == list(range(1, corpus + 1))
    assert finished["result"]["reference"]["basename"] == RELEASED
    assert finished["result"]["suite_version"] == suite["version"]

    manifest = ta.find_reference_bank(components["text_encoder_path"])
    assert manifest is not None and len(manifest["presentations"]) == corpus

    status = _status("M:/model/minimax_h3")
    assert status["bank"]["reference"] == RELEASED
    assert status["bank"]["is_loaded_encoder"] is True
    assert status["can_build"] is True
    assert status["job"]["state"] == "completed"


# ---------------------------------------------------------------------------
# 4. Cancellation
# ---------------------------------------------------------------------------

def test_a_cancelled_build_leaves_no_bank_behind(tmp_path, monkeypatch, store, suite,
                                                 stub_encode):
    from core.models.minimax_h3 import h3_pipeline_ops as ops

    entered, release = threading.Event(), threading.Event()
    encoder = _Encoder(TEXT_DIM)

    def blocking(text_encoder, token_ids, **kwargs):
        entered.set()
        release.wait(20)
        return encoder.encode(list(token_ids))

    monkeypatch.setattr(ops, "encode_presentation", blocking)
    components = _released_components(tmp_path, encoder=encoder)
    _install(monkeypatch, _Manager(components))

    _start(str(tmp_path / RELEASED))
    assert entered.wait(10), "the build never reached the encoder"
    cancelled = asyncio.run(routes.cancel_minimax_h3_reference_bank())
    assert cancelled["state"] == "running" and cancelled["message"] == "cancelling"
    release.set()

    finished = _await_job()
    assert finished["state"] == "cancelled"
    assert "cancelled after" in finished["message"]
    assert ta.find_reference_bank(components["text_encoder_path"]) is None
    banks = store / "banks"
    assert not banks.is_dir() or list(banks.iterdir()) == [], \
        "a cancelled build must leave no directory claiming to be a bank"


def test_cancelling_with_nothing_running_is_a_no_op(store):
    assert asyncio.run(routes.cancel_minimax_h3_reference_bank()) == {"state": "idle"}


# ---------------------------------------------------------------------------
# 5. The status document
# ---------------------------------------------------------------------------

def test_status_without_a_bank_states_the_cost_and_no_bank(tmp_path, monkeypatch, store,
                                                           suite, no_tree_scan):
    _install(monkeypatch, _Manager(_released_components(tmp_path)))

    status = _status("M:/model/minimax_h3")
    assert status["supported"] is True and status["can_build"] is True
    assert status["reason"] is None
    assert status["bank"] is None and status["banks"] == []
    assert status["measurements"] == [] and status["measurements_reason"] is None
    assert status["loaded"]["text_encoder"] == RELEASED
    assert status["loaded"]["is_substitute"] is False
    assert status["loaded"]["substitution"] is None
    assert status["suite"]["version"] == suite["version"]
    assert status["suite"]["prompts"] == len(suite["prompts"])
    assert status["cost"] == dict(ta.BUILD_COST)
    assert status["job"] == {"state": "idle"}


def test_status_with_no_model_loaded_reports_unsupported(monkeypatch, store, suite):
    manager = _Manager(None)
    manager.current_model_info = {}
    _install(monkeypatch, manager)

    status = _status()
    assert status["supported"] is False
    assert status["can_build"] is False
    assert status["reason"] == "no model is loaded"
    assert status["loaded"] is None
    assert status["bank"] is None


def test_a_bank_from_another_encoder_is_listed_but_does_not_answer_for_this_one(
        tmp_path, monkeypatch, store, suite, stub_encode, no_tree_scan):
    """Two released encoders: the loaded one has no bank, the other does."""
    other = tmp_path / "qwen3vl_32b_other.safetensors"
    save_file({"w": torch.ones(64)}, str(other))
    built = _released_components(tmp_path)
    built["text_encoder_path"] = str(other)
    ta.build_reference_bank(built, reference_basename=other.name)

    _install(monkeypatch, _Manager(_released_components(tmp_path)))
    status = _status("M:/model/minimax_h3")

    assert status["bank"] is None
    assert [entry["reference"] for entry in status["banks"]] == [other.name]
    assert status["banks"][0]["is_loaded_encoder"] is False
    assert status["can_build"] is True


def test_the_automatic_measurement_surfaces_once_a_bank_exists(tmp_path, monkeypatch, store,
                                                               suite, stub_encode):
    """Load a substitute after a bank exists: the cheap half runs and is reported."""
    reference = _released_components(tmp_path)
    ta.build_reference_bank(reference, reference_basename=RELEASED)

    substitute = _substitute_components(tmp_path)
    assert ta.maybe_measure_substitution(substitute) is not None

    from core.models.minimax_h3 import loader

    monkeypatch.setattr(loader, "describe_minimax_h3_text_encoder_choices", lambda model_path: {
        "text_encoders": [{"path": substitute["text_encoder_path"]},
                          {"path": reference["text_encoder_path"]}],
        "clip_projections": [{"path": substitute["te_projection"]["path"]}]})
    _install(monkeypatch, _Manager(substitute))

    status = _status("M:/model/minimax_h3")
    assert len(status["measurements"]) == 1
    measurement = status["measurements"][0]
    assert measurement["encoder"] == CONVERTED
    assert measurement["projection"] == PROJECTION
    assert measurement["reference"] == RELEASED
    assert measurement["stage"] == "raw"  # no DiT was loaded, so no post-refiner view
    assert measurement["cosine"] is not None
    assert measurement["presentations"] == len(ta.build_corpus(_Tokenizer(), suite))

    # A substituted pairing cannot be a reference, and the status says so with
    # the engine's own wording rather than a second one.
    assert status["can_build"] is False
    assert "cannot be one" in status["reason"]
    assert "Measured on this installation" in status["loaded"]["substitution"]
    assert status["loaded"]["is_substitute"] is True


def test_measurements_from_another_tree_are_not_reported(tmp_path, monkeypatch, store, suite,
                                                         stub_encode, no_tree_scan):
    """A measurement whose files are in no scanned tree belongs to no tree here."""
    reference = _released_components(tmp_path)
    ta.build_reference_bank(reference, reference_basename=RELEASED)
    assert ta.maybe_measure_substitution(_substitute_components(tmp_path)) is not None

    _install(monkeypatch, _Manager(reference))
    assert _status("M:/model/other_tree")["measurements"] == []


def test_a_tree_that_cannot_be_scanned_says_so(tmp_path, monkeypatch, store, suite):
    from core.models.minimax_h3 import loader

    def explode(model_path):
        raise ValueError("does not resolve to a MiniMax-H3 model tree")

    monkeypatch.setattr(loader, "describe_minimax_h3_text_encoder_choices", explode)
    _install(monkeypatch, _Manager(_released_components(tmp_path)))

    status = _status("M:/not/a/tree")
    assert status["measurements"] == []
    assert "could not be scanned" in status["measurements_reason"]
