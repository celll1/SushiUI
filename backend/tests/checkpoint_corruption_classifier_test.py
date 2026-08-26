"""What counts as a corrupted checkpoint on resume, and what does not.

A resume failure routes one of two ways: a corrupted FILE falls back to the
previous checkpoint (and the one before that, ...), while any other failure
aborts the run. The shipped discriminator was a substring match on the
exception's text -- containing ``"safetensor"``, ``"corrupted"``, ``"truncated"``
-- duplicated verbatim at both decision points. A structural refusal that merely
NAMES a ``.safetensors`` file therefore read as corruption and reloaded every
older checkpoint (17-25 GiB apiece on SenseNova) to refuse each for the same
reason.

The NEGATIVE CONTROLS are ``test_negative_control_*``: they run the shipped
substring list against real refusal texts and count the resulting loads.

CPU only; the "checkpoints" here are byte files, and the loader is a counter.

Run:
    venv/Scripts/python.exe -m pytest backend/tests/checkpoint_corruption_classifier_test.py -v
"""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest
import torch
from safetensors import SafetensorError
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.training.base_trainer import BaseTrainer, is_checkpoint_corruption_error

RUN_NAME = "20260101_000000_abcdef"

# The list HEAD~ shipped, twice.
_SHIPPED_MARKERS = [
    "incomplete metadata",
    "file not fully covered",
    "deserializing header",
    "safetensor",
    "corrupted",
    "truncated",
    "unexpected end",
    "invalid header",
]


def _shipped_is_corruption(exc: Exception) -> bool:
    error_str = str(exc).lower()
    return any(x in error_str for x in _SHIPPED_MARKERS)


# ---------------------------------------------------------------------------
# Real refusals, quoted from the paths a resume actually reaches
# ---------------------------------------------------------------------------

def _refusal_naming_the_file(name=f"{RUN_NAME}_step_000030.safetensors"):
    """The SenseNova resume-layout refusal, with the checkpoint named in full.

    ``sensenova_ops.accept_resume_shaped_base`` currently strips the suffix off
    the name for exactly this reason; the discriminator must not need it to.
    """
    return RuntimeError(
        f"SenseNova cannot resume the 'gen' branch from {name}: the und half of "
        f"its decoder is not the shape this run trains in. Expected all 294 of "
        f"its Linears to be plain Int8Linear; got gen half: float=294, int8=0, "
        f"other=0; und half: float=294, int8=0, other=0. A resume of this branch "
        f"is only lossless from a checkpoint written as "
        f"sensenova_full_finetune_save_format='mixed'."
    )


_STRUCTURAL_REFUSALS = {
    "sensenova_layout": _refusal_naming_the_file(),
    "sensenova_stamp": RuntimeError(
        f"SenseNova refuses to resume from {RUN_NAME}_step_000030.safetensors: it "
        f"carries the decoder layout a 'gen'-branch resume needs, but not this "
        f"repo's own save stamp (sensenova_trained_branch / sensenova_save_format)."
    ),
    "blocks_to_swap": ValueError(
        "SenseNova training does not implement blocks_to_swap; set it to 0"),
    "flux2_structure": ValueError(
        "FLUX.2 transformer does not have transformer_blocks/single_transformer_blocks; "
        "Block Swap cannot be set up for 20260101_000000_abcdef_step_000030.safetensors"),
    "unsupported_base": RuntimeError(
        "the safetensors file names a quantization this run cannot train from"),
    "truncated_dataset": RuntimeError(
        "the caption list was truncated at 512 tokens; refusing to resume with a "
        "different tokenizer than the checkpoint was written with"),
}


# The ones the shipped list misfired on. ``blocks_to_swap`` is excluded: its
# text happens to contain no marker, which is the whole problem -- whether a
# structural refusal reloaded 17-25 GiB depended on its prose.
_MISCLASSIFIED_REFUSALS = sorted(set(_STRUCTURAL_REFUSALS) - {"blocks_to_swap"})


@pytest.mark.parametrize("name", _MISCLASSIFIED_REFUSALS)
def test_negative_control_shipped_list_calls_structural_refusals_corruption(name):
    exc = _STRUCTURAL_REFUSALS[name]
    assert _shipped_is_corruption(exc) is True, "fixture no longer reproduces the defect"


def test_negative_control_the_shipped_list_was_prose_dependent():
    """The same class of refusal, classified two different ways by wording."""
    assert _shipped_is_corruption(_STRUCTURAL_REFUSALS["blocks_to_swap"]) is False
    assert _shipped_is_corruption(_STRUCTURAL_REFUSALS["sensenova_layout"]) is True


@pytest.mark.parametrize("name", sorted(_STRUCTURAL_REFUSALS))
def test_structural_refusals_are_not_corruption(name):
    assert is_checkpoint_corruption_error(_STRUCTURAL_REFUSALS[name]) is False


# ---------------------------------------------------------------------------
# Real corruption still routes to the fallback
# ---------------------------------------------------------------------------

def _truncated_safetensors(path: Path) -> Path:
    save_file({"a": torch.zeros(64)}, str(path))
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])
    return path


def _read_error(path: Path) -> Exception:
    try:
        load_file(str(path))
    except Exception as exc:
        return exc
    raise AssertionError(f"{path.name} loaded successfully; not a corruption fixture")


def test_a_truncated_safetensors_file_is_corruption(tmp_path):
    exc = _read_error(_truncated_safetensors(tmp_path / "ckpt.safetensors"))
    assert isinstance(exc, SafetensorError)
    assert is_checkpoint_corruption_error(exc) is True


def test_a_garbage_header_is_corruption(tmp_path):
    path = tmp_path / "ckpt.safetensors"
    path.write_bytes(b"\x08\x00\x00\x00\x00\x00\x00\x00" + b"not-json")
    assert is_checkpoint_corruption_error(_read_error(path)) is True


def test_an_empty_file_is_corruption(tmp_path):
    path = tmp_path / "ckpt.safetensors"
    path.write_bytes(b"")
    assert is_checkpoint_corruption_error(_read_error(path)) is True


def test_a_truncated_shard_index_is_corruption(tmp_path):
    """Exercises the classifier against ``json.loads`` directly, not the
    production loader: core/model_loader.py:667-669 swallows a corrupt shard
    index in a bare ``except Exception`` and returns "sd15" without ever
    letting the ``JSONDecodeError`` reach ``is_checkpoint_corruption_error``."""
    path = tmp_path / "ckpt.safetensors.index.json"
    path.write_text('{"weight_map": {"a": "ckpt-00001-of-00002.safet', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError) as excinfo:
        json.loads(path.read_text(encoding="utf-8"))
    assert is_checkpoint_corruption_error(excinfo.value) is True


def test_a_truncated_torch_file_is_corruption(tmp_path):
    path = tmp_path / "state.pt"
    torch.save({"a": torch.zeros(64)}, str(path))
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])
    with pytest.raises(Exception) as excinfo:
        torch.load(str(path), map_location="cpu", weights_only=False)
    assert is_checkpoint_corruption_error(excinfo.value) is True


def test_a_wrapped_corruption_is_still_corruption(tmp_path):
    """Loaders re-raise as RuntimeError; the reader's type survives the chain."""
    inner = _read_error(_truncated_safetensors(tmp_path / "ckpt.safetensors"))
    try:
        try:
            raise inner
        except Exception as exc:
            raise RuntimeError("Failed to load Z-Image transformer weights") from exc
    except RuntimeError as wrapped:
        assert is_checkpoint_corruption_error(wrapped) is True


def test_an_unchained_wrap_is_caught_by_the_readers_own_text():
    """``raise RuntimeError(f"...: {e}") from None`` loses the type, not the text."""
    exc = RuntimeError(
        "Failed to load checkpoint: Error while deserializing header: "
        "invalid header length")
    assert is_checkpoint_corruption_error(exc) is True


# ---------------------------------------------------------------------------
# What each classification costs, in checkpoint loads
# ---------------------------------------------------------------------------

class _Probe:
    _get_sorted_checkpoints = BaseTrainer._get_sorted_checkpoints
    _try_load_checkpoint_with_fallback = BaseTrainer._try_load_checkpoint_with_fallback

    def __init__(self, output_dir, failures):
        self.log_prefix = "[test]"
        self.output_dir = Path(output_dir)
        self.loads = []
        self._failures = failures

    def _load_checkpoint_as_base(self, path):
        self.loads.append(Path(path).name)
        error = self._failures(Path(path).name)
        if error is not None:
            raise error


def _make_checkpoints(tmp_path, steps=(10, 20, 30)):
    for step in steps:
        save_file({"a": torch.zeros(4)},
                  str(tmp_path / f"{RUN_NAME}_step_{step:06d}.safetensors"))
    return [f"{RUN_NAME}_step_{s:06d}.safetensors" for s in steps]


def _shipped_fallback(probe, checkpoint_path):
    """HEAD~'s ``_try_load_checkpoint_with_fallback`` loop, with its own list."""
    sorted_checkpoints = probe._get_sorted_checkpoints()
    for i, (ckpt, _step) in enumerate(sorted_checkpoints):
        try:
            probe._load_checkpoint_as_base(str(ckpt))
            return True
        except Exception as e:
            if _shipped_is_corruption(e):
                continue
            raise
    return False


def test_negative_control_a_structural_refusal_reloads_every_checkpoint(tmp_path):
    _make_checkpoints(tmp_path)
    probe = _Probe(tmp_path, lambda name: _refusal_naming_the_file(name))

    with redirect_stdout(io.StringIO()):
        result = _shipped_fallback(probe, "latest")

    # Three checkpoints, three full model loads, all refused for the same
    # structural reason -- and then the run aborts anyway.
    assert result is False
    assert probe.loads == [f"{RUN_NAME}_step_{s:06d}.safetensors" for s in (30, 20, 10)]
    assert len(probe.loads) == 3


def test_a_structural_refusal_loads_once_and_propagates(tmp_path):
    _make_checkpoints(tmp_path)
    probe = _Probe(tmp_path, lambda name: _refusal_naming_the_file(name))

    with redirect_stdout(io.StringIO()):
        with pytest.raises(RuntimeError, match="cannot resume"):
            probe._try_load_checkpoint_with_fallback("latest")

    assert probe.loads == [f"{RUN_NAME}_step_000030.safetensors"]
    assert len(probe.loads) == 1


def test_a_corrupt_newest_checkpoint_still_falls_back(tmp_path):
    names = _make_checkpoints(tmp_path)
    _truncated_safetensors(tmp_path / names[-1])

    def failures(name):
        path = tmp_path / name
        try:
            load_file(str(path))
        except Exception as exc:
            return exc
        return None

    probe = _Probe(tmp_path, failures)
    with redirect_stdout(io.StringIO()):
        success, loaded = probe._try_load_checkpoint_with_fallback("latest")

    assert success is True
    assert Path(loaded).name == f"{RUN_NAME}_step_000020.safetensors"
    assert probe.loads == [f"{RUN_NAME}_step_000030.safetensors",
                           f"{RUN_NAME}_step_000020.safetensors"]


def test_every_checkpoint_corrupt_exhausts_the_fallback(tmp_path):
    names = _make_checkpoints(tmp_path)
    for name in names:
        _truncated_safetensors(tmp_path / name)

    def failures(name):
        try:
            load_file(str(tmp_path / name))
        except Exception as exc:
            return exc
        return None

    probe = _Probe(tmp_path, failures)
    with redirect_stdout(io.StringIO()):
        success, loaded = probe._try_load_checkpoint_with_fallback("latest")

    assert (success, loaded) == (False, None)
    assert len(probe.loads) == 3


def test_a_healthy_checkpoint_loads_once(tmp_path):
    _make_checkpoints(tmp_path)
    probe = _Probe(tmp_path, lambda name: None)
    with redirect_stdout(io.StringIO()):
        success, loaded = probe._try_load_checkpoint_with_fallback("latest")
    assert success is True
    assert probe.loads == [f"{RUN_NAME}_step_000030.safetensors"]


# ---------------------------------------------------------------------------
# One discriminator, both decision points
# ---------------------------------------------------------------------------

def _source():
    return Path(sys.modules[BaseTrainer.__module__].__file__).read_text(encoding="utf-8")


def test_the_substring_list_is_gone_from_both_sites():
    source = _source()
    assert '"safetensor",' not in source
    assert source.count('"deserializing header"') == 0
    assert source.count("is_checkpoint_corruption_error(") == 3   # def + two sites


@pytest.mark.parametrize("method", ["__init__(", "_try_load_checkpoint_with_fallback(self"])
def test_each_decision_point_calls_the_shared_discriminator(method):
    source = _source()
    body = source[source.index(f"    def {method}"):]
    body = body[:body.index("\n    def ", 10)]
    assert "is_checkpoint_corruption_error(e)" in body
