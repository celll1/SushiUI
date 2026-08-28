"""BaseTrainer._run_step0_sample_if_due: no duplicate step-0 sample across a
relaunch of THIS run between the sample's own save and its first checkpoint.

The guard covers only that window: a crash AFTER step0_sample_path.save()
but BEFORE any checkpoint. A crash DURING the sample itself leaves no PNG,
so the guard has nothing to key off and the relaunch regenerates (this is
not a bug in the guard, just outside what it can cover).

The guard is keyed on a marker file naming the producing run's DB row
(``run_id``), not on path existence alone: ``routes.py``'s
``output_dir.mkdir(exist_ok=True)`` means a NEW run started under an
EXISTING run_name inherits the previous run's ``samples/`` directory, so
path existence alone would skip a different run's own step-0 verification
sample and silently show it a stale base-model check.

This is exercised against the real, shipped ``BaseTrainer._run_step0_sample_if_due``
(the exact method ``train()`` now calls) rather than a reimplementation.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.training.base_trainer import BaseTrainer


class _ConcreteTrainer(BaseTrainer):
    def setup_trainable_parameters(self):
        return []

    def save_checkpoint(self, *args, **kwargs):
        raise NotImplementedError

    def load_checkpoint(self, *args, **kwargs):
        raise NotImplementedError


def _trainer(tmp_path, *, prompts=None, run_id=1):
    trainer = _ConcreteTrainer.__new__(_ConcreteTrainer)
    trainer.output_dir = tmp_path
    trainer.log_prefix = "[test]"
    trainer.run_id = run_id
    trainer._sample_prompts = prompts or [{"positive": "a cat"}]
    return trainer


def _kwargs(**overrides):
    base = dict(
        sample_every_n_steps=100,
        sample_width=64,
        sample_height=64,
        sample_guidance_scale=4.0,
        sample_steps=4,
        sample_seed=1,
        sample_schedule_type="uniform",
        global_step=0,
    )
    base.update(overrides)
    return base


def test_first_launch_generates_and_saves_the_step0_sample(tmp_path):
    from PIL import Image

    trainer = _trainer(tmp_path)
    image = Image.new("RGB", (8, 8))
    with patch.object(trainer, "_dispatch_sample", return_value=image) as dispatch:
        trainer._run_step0_sample_if_due(**_kwargs())

    dispatch.assert_called_once()
    saved = tmp_path / "samples" / "step_000000_sample_0.png"
    assert saved.exists()


def test_step0_forwards_conditioning_and_embeds_generation_metadata(tmp_path):
    from PIL import Image

    prompts = [{
        "positive": "a cat",
        "negative": "blurry",
        "condition_image_path": "condition.png",
        "reference_image_path": "reference.png",
    }]
    trainer = _trainer(tmp_path, prompts=prompts)
    image = Image.new("RGB", (8, 8))
    with patch.object(trainer, "_dispatch_sample", return_value=image) as dispatch:
        trainer._run_step0_sample_if_due(**_kwargs())

    assert dispatch.call_args.kwargs["negative_prompt"] == "blurry"
    assert dispatch.call_args.kwargs["condition_image_path"] == "condition.png"
    assert dispatch.call_args.kwargs["reference_image_path"] == "reference.png"

    saved = tmp_path / "samples" / "step_000000_sample_0.png"
    with Image.open(saved) as sample:
        assert sample.text == {
            "prompt": "a cat",
            "negative_prompt": "blurry",
            "steps": "4",
            "cfg_scale": "4.0",
            "seed": "1",
            "width": "64",
            "height": "64",
            "schedule_type": "uniform",
            "condition_image_path": "condition.png",
            "reference_image_path": "reference.png",
        }


def test_relaunch_with_no_checkpoint_skips_the_already_saved_sample(tmp_path):
    """Two calls with global_step=0 on the same output_dir -- exactly what a
    crash-before-first-checkpoint relaunch replays -- must dispatch once."""
    from PIL import Image

    trainer = _trainer(tmp_path)
    image = Image.new("RGB", (8, 8))
    with patch.object(trainer, "_dispatch_sample", return_value=image) as dispatch:
        trainer._run_step0_sample_if_due(**_kwargs())
        trainer._run_step0_sample_if_due(**_kwargs())

    assert dispatch.call_count == 1


def test_relaunch_skip_is_logged(tmp_path, capsys):
    from PIL import Image

    trainer = _trainer(tmp_path)
    image = Image.new("RGB", (8, 8))
    with patch.object(trainer, "_dispatch_sample", return_value=image):
        trainer._run_step0_sample_if_due(**_kwargs())
    capsys.readouterr()
    with patch.object(trainer, "_dispatch_sample", return_value=image) as dispatch:
        trainer._run_step0_sample_if_due(**_kwargs())

    dispatch.assert_not_called()
    assert "Skipping sample" in capsys.readouterr().out


def test_sample_every_n_steps_zero_never_dispatches(tmp_path):
    trainer = _trainer(tmp_path)
    with patch.object(trainer, "_dispatch_sample") as dispatch:
        trainer._run_step0_sample_if_due(**_kwargs(sample_every_n_steps=0))

    dispatch.assert_not_called()


def test_nonzero_global_step_never_dispatches(tmp_path):
    """This method only ever fires for the pre-loop step-0 verification call;
    periodic in-loop sampling is a separate code path."""
    trainer = _trainer(tmp_path)
    with patch.object(trainer, "_dispatch_sample") as dispatch:
        trainer._run_step0_sample_if_due(**_kwargs(global_step=1))

    dispatch.assert_not_called()


def test_a_none_sample_is_not_saved_and_leaves_no_file_to_dedup_against(tmp_path):
    """architecture can't sample yet (e.g. ideogram4) -> None; a later launch
    must still be able to try again, not be permanently skipped by a phantom
    file that was never written."""
    trainer = _trainer(tmp_path)
    with patch.object(trainer, "_dispatch_sample", return_value=None) as dispatch:
        trainer._run_step0_sample_if_due(**_kwargs())

    dispatch.assert_called_once()
    assert not (tmp_path / "samples" / "step_000000_sample_0.png").exists()

    with patch.object(trainer, "_dispatch_sample", return_value=None) as dispatch2:
        trainer._run_step0_sample_if_due(**_kwargs())

    dispatch2.assert_called_once()


def test_a_different_run_id_under_the_same_output_dir_regenerates(tmp_path):
    """Negative control for the path-existence-only guard this replaces: a
    NEW run (different run_id) started under an output_dir that inherits a
    PREVIOUS run's samples/ (routes.py's mkdir(exist_ok=True) on a reused
    run_name) must NOT be skipped just because the PNG is already there --
    that file was produced by a different base model checkpoint."""
    from PIL import Image

    old_run = _trainer(tmp_path, run_id=5)
    image = Image.new("RGB", (8, 8))
    with patch.object(old_run, "_dispatch_sample", return_value=image):
        old_run._run_step0_sample_if_due(**_kwargs())

    new_run = _trainer(tmp_path, run_id=9)
    with patch.object(new_run, "_dispatch_sample", return_value=image) as dispatch:
        new_run._run_step0_sample_if_due(**_kwargs())

    dispatch.assert_called_once()


def test_marker_missing_regenerates_even_though_the_png_exists(tmp_path):
    """A PNG present with no marker (e.g. written before this guard existed)
    is not proof it belongs to this run -- must regenerate, not skip."""
    from PIL import Image

    trainer = _trainer(tmp_path, run_id=1)
    (tmp_path / "samples").mkdir(parents=True)
    (tmp_path / "samples" / "step_000000_sample_0.png").write_bytes(b"not a real png")

    image = Image.new("RGB", (8, 8))
    with patch.object(trainer, "_dispatch_sample", return_value=image) as dispatch:
        trainer._run_step0_sample_if_due(**_kwargs())

    dispatch.assert_called_once()


def test_no_run_id_never_skips(tmp_path):
    """run_id is None (no DB row) -- there is nothing to key a marker on, so
    the conservative default is to always regenerate rather than guess."""
    from PIL import Image

    trainer = _trainer(tmp_path, run_id=None)
    image = Image.new("RGB", (8, 8))
    with patch.object(trainer, "_dispatch_sample", return_value=image) as dispatch:
        trainer._run_step0_sample_if_due(**_kwargs())
        trainer._run_step0_sample_if_due(**_kwargs())

    assert dispatch.call_count == 2
