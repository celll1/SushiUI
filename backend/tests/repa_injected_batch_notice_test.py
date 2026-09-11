"""A run whose injected batches get no REPA must say so once, at the start.

``swap_onthefly`` frees an injected item's pixels as soon as its latent enters
the swap buffer, so the REPA teacher has nothing to encode and those batches
train on the diffusion loss alone. That is the decided behaviour (carrying the
pixels costs host RAM proportional to the refill window, and refusing the run
would close the only mode a multi-million-item dataset can use), but the only
thing the operator saw was a load-failure line indistinguishable from breakage.

  (a) the notice exists for swap_onthefly + repa_enable + active injection, and
      states the skip, the reason, and that it is intended;
  (b) it is absent for repa off, injection off, and onthefly_gpu;
  (c) it is emitted once, outside every loop, from train();
  (d) it rides the training-event channel at warning level, so it survives on
      the run row rather than only on the console;
  (e) the per-item paths still log at most once each -- no per-step logging is
      added, and none is left repeating;
  (f) nothing about the notice touches the loss.

Run:
    venv/Scripts/python.exe -m pytest backend/tests/repa_injected_batch_notice_test.py -v

Static: no model, no GPU, no DB, no network.
"""

import ast
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.training.base_trainer import BaseTrainer  # noqa: E402
from core.training.repa import (  # noqa: E402
    INJECTED_BATCH_SKIP_CODE,
    injected_batch_skip_notice,
)
from core.training.training_events import (  # noqa: E402
    merge_run_warnings,
    parse_training_event,
)

BACKEND = Path(__file__).resolve().parents[1]
BASE_TRAINER_SRC = (BACKEND / "core" / "training" / "base_trainer.py").read_text(
    encoding="utf-8")

ON = dict(latent_encoding_mode="swap_onthefly", repa_enable=True,
          injection_active=True, inject_batch_size=4, inject_interval=4)


def _notice(**over):
    kw = dict(ON)
    kw.update(over)
    return injected_batch_skip_notice(**kw)


# ---------------------------------------------------------------------------
# (a) the combination that skips says so
# ---------------------------------------------------------------------------

def test_notice_states_skip_reason_and_intent():
    text = _notice()
    assert text
    assert "REPA is not applied to Danbooru-injected batches" in text
    assert "freed once its latent enters the swap buffer" in text
    assert "intended behaviour" in text
    assert "onthefly_gpu" in text


def test_notice_reports_the_injection_schedule():
    """The share of batches without a REPA term is arithmetic on the schedule:
    one injected batch is spliced after every `interval` base batches."""
    text = _notice(inject_batch_size=2, inject_interval=7)
    assert "one batch of 2 image(s) after every 7 base batch(es)" in text
    assert "at most one batch in every 8" in text
    assert "the skip reaches no dataset item" in text


def test_notice_fits_the_event_message_bound():
    from core.training.training_events import MAX_EVENT_MESSAGE_CHARS
    assert len(_notice(inject_batch_size=999, inject_interval=999)) < MAX_EVENT_MESSAGE_CHARS



def test_silent_when_repa_is_off():
    assert _notice(repa_enable=False) is None


def test_silent_when_injection_is_off():
    assert _notice(injection_active=False) is None
    # The collector can be dropped after the batch-size/interval fields are set
    # (setup failure), so the fields alone are not the gate -- but a zero in
    # either of them is not a schedule either.
    assert _notice(inject_batch_size=0) is None
    assert _notice(inject_interval=0) is None


@pytest.mark.parametrize("mode", ["onthefly_gpu", "pre_encoded_cache", "cpu_prefetch", "", None])
def test_silent_for_every_other_latent_mode(mode):
    """onthefly_gpu hands the decode to the teacher (c6292216/253de1db), and no
    other mode injects at all."""
    assert _notice(latent_encoding_mode=mode) is None



def _call_sites(name):
    tree = ast.parse(BASE_TRAINER_SRC)
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    sites = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == name]
    return sites, parents


def _enclosing(node, parents):
    chain = []
    while node in parents:
        node = parents[node]
        chain.append(node)
    return chain


def test_notice_is_built_once_outside_any_loop():
    sites, parents = _call_sites("injected_batch_skip_notice")
    assert len(sites) == 1, f"{len(sites)} call sites"
    chain = _enclosing(sites[0], parents)
    assert not any(isinstance(n, (ast.For, ast.While)) for n in chain), \
        "called inside a loop: it would print every iteration"
    assert any(isinstance(n, ast.FunctionDef) and n.name == "train" for n in chain)


def test_notice_is_emitted_on_the_training_event_channel():
    """Guarded by the notice itself and handed to emit_training_warning with the
    shared code, so the run row carries it."""
    tree = ast.parse(BASE_TRAINER_SRC)
    emits = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "emit_training_warning"
             and any(kw.arg == "code" and isinstance(kw.value, ast.Name)
                     and kw.value.id == "INJECTED_BATCH_SKIP_CODE" for kw in n.keywords)]
    assert len(emits) == 1
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    chain = _enclosing(emits[0], parents)
    assert not any(isinstance(n, (ast.For, ast.While)) for n in chain)
    assert any(isinstance(n, ast.If) for n in chain), "emitted unconditionally"



def test_notice_persists_on_the_run_row(capsys):
    from core.training.training_events import emit_training_warning
    emit_training_warning(_notice(), code=INJECTED_BATCH_SKIP_CODE, prefix="[test]")
    lines = [l for l in capsys.readouterr().out.splitlines() if l.strip()]
    events = [e for e in (parse_training_event(l) for l in lines) if e]
    assert len(events) == 1
    assert events[0]["level"] == "warning"          # info is not persisted
    assert events[0]["code"] == INJECTED_BATCH_SKIP_CODE
    kept = merge_run_warnings(None, events[0])
    assert kept and kept[0]["message"] == _notice()
    # Say-once: a second identical event does not grow the row.
    assert merge_run_warnings(kept, events[0]) is None



def _pixel_trainer():
    return SimpleNamespace(log_prefix="[test]", repa_size=16,
                           _repa_pix_cache_off=True)


def test_missing_region_logs_once_not_per_batch(capsys):
    t = _pixel_trainer()
    for _ in range(5):
        assert BaseTrainer._get_repa_pixels_for_item(t, {"image_path": "danbooru://1"},
                                                     None) is None
    assert capsys.readouterr().out.count("source region unavailable") == 1


def test_unreadable_source_logs_once_not_per_batch(capsys):
    t = _pixel_trainer()
    item = {"image_path": "danbooru://1"}
    for _ in range(5):
        assert BaseTrainer._get_repa_pixels_for_item(t, item, (0, 0, 8, 8)) is None
    assert capsys.readouterr().out.count("clean-image load failed") == 1



def test_notice_only_reads_and_only_prints():
    """A pure function of five scalars: no trainer, no tensors, no assignment
    to anything but its own locals."""
    import inspect
    from core.training import repa
    fn = ast.parse(inspect.getsource(repa.injected_batch_skip_notice)).body[0]
    assert [a.arg for a in fn.args.args] == [
        "latent_encoding_mode", "repa_enable", "injection_active",
        "inject_batch_size", "inject_interval"]
    assert not [n for n in ast.walk(fn) if isinstance(n, (ast.Attribute, ast.Global))]
