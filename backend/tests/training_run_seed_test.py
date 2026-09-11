"""A run's sample order must be reproducible from its ``seed``.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/training_run_seed_test.py -q

Static: no model, no GPU, no DB.

THE DEFECT
----------
Nothing seeded the trainer subprocess. Both shuffles in the non-bucketed batch
builder and every shuffle in ``bucketing.py`` draw from the ``random`` module,
which starts from OS entropy, so two runs differing only in one config key drew
different samples in a different order and could not be compared.

WHAT IS PINNED
--------------
``apply_run_seed`` seeds the ``random`` MODULE rather than a private
``random.Random``, because that module is the stream both shuffle paths draw
from and the one ``save_training_state`` serializes for mid-epoch resume. These
tests therefore exercise the real production shufflers under that seeding:
``BucketManager.build_batch_indices`` directly, and -- for the two inline
``base_trainer.py`` sites that cannot be unit-invoked -- a comment-stripped copy
plus a source grep, the same arrangement ``no_bucketing_epoch_shuffle_test.py``
uses.

The default stays -1 = drawn, so an unconfigured run keeps its previous
unchosen order; a drawn seed is recorded as an ``info`` notice instead. The last
test pins the one place a drawn seed must NOT reach: the crop plan's seed, whose
fingerprint would then change every launch and make a resume restart the epoch.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import torch
import yaml

# AGENTS.md: keep the import off the GPU the owner's training run is holding.
torch.cuda.get_device_capability = lambda *a, **k: (8, 9)
torch.cuda._lazy_init = lambda *a, **k: None
torch._C._cuda_init = lambda *a, **k: None

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from api.param_defaults import TRAINING_DEFAULTS  # noqa: E402
from api.routes import (  # noqa: E402
    TrainingRunCreateRequest,
    _extract_request_params_from_yaml,
)
from core.training.base_trainer import apply_run_seed, resolve_run_seed  # noqa: E402
from core.training.bucketing import BucketManager  # noqa: E402
from core.training.crop_planner import CropPlanner  # noqa: E402
from core.training.training_config import TrainingConfigGenerator  # noqa: E402

_BASE_TRAINER_SRC = (_BACKEND / "core" / "training" / "base_trainer.py").read_text(
    encoding="utf-8")



def _build_non_bucketed_batches(items, batch_size, priority_items=None):
    if priority_items is not None:
        normal_items = list(items)
        random.shuffle(normal_items)
        return ([priority_items[i:i + batch_size]
                 for i in range(0, len(priority_items), batch_size)]
                + [normal_items[i:i + batch_size]
                   for i in range(0, len(normal_items), batch_size)])
    shuffled = list(items)
    random.shuffle(shuffled)
    return [shuffled[i:i + batch_size] for i in range(0, len(shuffled), batch_size)]


def _order(batches):
    return [item for batch in batches for item in batch]


def _items(n):
    return [f"img_{i}.png" for i in range(n)]


def _bucketed_order(seed, n=64):
    apply_run_seed(seed)
    manager = BucketManager(base_resolutions=[1024])
    for i in range(n):
        manager.assign_image_to_bucket(f"img_{i}.png",
                                       width=1024 + (i % 3) * 128,
                                       height=1024 - (i % 5) * 64)
    manager.shuffle_buckets()
    return [item["image_path"] for batch in manager.build_batch_indices(4) for item in batch]



def test_a_configured_seed_is_used_verbatim():
    assert resolve_run_seed(0) == (0, False)
    assert resolve_run_seed(12345) == (12345, False)
    assert resolve_run_seed("7") == (7, False)
    assert resolve_run_seed(2 ** 31 - 1) == (2 ** 31 - 1, False)


def test_an_out_of_range_seed_folds_instead_of_killing_the_run():
    """Only reachable by hand-editing the YAML; numpy would reject it raw."""
    seed, drawn = resolve_run_seed(10 ** 12)
    assert drawn is False and 0 <= seed < 2 ** 31
    apply_run_seed(10 ** 12)  # would raise if the fold were left to numpy


def test_anything_not_a_usable_seed_is_drawn():
    for configured in (-1, None, "", "nonsense", TRAINING_DEFAULTS["seed"]):
        seed, drawn = resolve_run_seed(configured)
        assert drawn is True, configured
        assert 0 <= seed < 2 ** 31


def test_the_default_is_still_an_unchosen_order():
    """-1 keeps the pre-change behaviour: nothing pins two runs together."""
    assert TRAINING_DEFAULTS["seed"] == -1
    drawn = {resolve_run_seed(-1)[0] for _ in range(20)}
    assert len(drawn) > 1

    orders = {tuple(_order(_build_non_bucketed_batches(
        (apply_run_seed(-1), _items(40))[1], 4))) for _ in range(8)}
    assert len(orders) > 1


# ---------------------------------------------------------------------------
# Same seed -> same order, on both paths
# ---------------------------------------------------------------------------

def test_same_seed_same_order_non_bucketed():
    items = _items(37)  # not a multiple of batch_size, as real datasets are not

    apply_run_seed(42)
    first = _order(_build_non_bucketed_batches(items, 5))
    apply_run_seed(42)
    again = _order(_build_non_bucketed_batches(items, 5))
    apply_run_seed(43)
    other = _order(_build_non_bucketed_batches(items, 5))

    assert first == again
    assert first != other
    assert sorted(first) == sorted(items)


def test_same_seed_same_order_priority_arm():
    """The second shuffle site: priority items keep entry order, normal shuffle."""
    priority = [f"p_{i}.png" for i in range(5)]
    items = _items(40)

    apply_run_seed(9)
    first = _order(_build_non_bucketed_batches(items, 5, priority_items=priority))
    apply_run_seed(9)
    again = _order(_build_non_bucketed_batches(items, 5, priority_items=priority))
    apply_run_seed(10)
    other = _order(_build_non_bucketed_batches(items, 5, priority_items=priority))

    assert first == again
    assert first != other
    assert [p for p in first if p.startswith("p_")] == priority


def test_same_seed_same_order_bucketed():
    """The default path, through the real BucketManager."""
    first = _bucketed_order(42)
    again = _bucketed_order(42)
    other = _bucketed_order(43)

    assert first == again
    assert first != other
    assert sorted(first) == sorted(_items(64))



def test_both_inline_shuffle_sites_draw_from_the_seeded_module():
    assert "random.shuffle(normal_items)" in _BASE_TRAINER_SRC
    assert "random.shuffle(_shuffled_image_items)" in _BASE_TRAINER_SRC
    # A private random.Random would leave both of them, and bucketing.py, unseeded.
    assert "random.seed(seed)" in _BASE_TRAINER_SRC
    assert "self.run_seed, self.run_seed_drawn = apply_run_seed(" in _BASE_TRAINER_SRC


def test_apply_run_seed_seeds_the_module_the_shufflers_use():
    apply_run_seed(5)
    draws = [random.random(), random.randrange(1000)]
    apply_run_seed(5)
    assert [random.random(), random.randrange(1000)] == draws


def test_the_seed_reaches_the_generated_config_and_survives_an_edit():
    request = TrainingRunCreateRequest(training_method="lora",
                                       base_model_path="model.safetensors",
                                       total_steps=100, seed=4242)
    text = TrainingConfigGenerator.generate_lora_config(
        request.model_dump(), run_name="seed_roundtrip",
        base_model_path="model.safetensors", output_dir="out", dataset_path="data")
    process = yaml.safe_load(text)["config"]["process"][0]

    assert process["train"]["seed"] == 4242
    params = _extract_request_params_from_yaml(process, "lora")
    assert TrainingRunCreateRequest(**{**params, "training_method": "lora",
                                       "base_model_path": "m"}).seed == 4242


def test_the_api_refuses_a_seed_no_generator_would_accept():
    for bad in (-2, 2 ** 31):
        try:
            TrainingRunCreateRequest(training_method="lora", base_model_path="m",
                                     seed=bad)
        except Exception:
            continue
        raise AssertionError(f"seed={bad} was accepted")


# ---------------------------------------------------------------------------
# A drawn seed must not reach the crop plan
# ---------------------------------------------------------------------------

def test_a_drawn_seed_leaves_the_crop_plan_seed_alone():
    """Its fingerprint gates resume: a per-launch seed restarts the epoch."""
    assert CropPlanner(config={"seed": -1}, base_resolutions=[1024]).seed == 0
    assert CropPlanner(config={}, base_resolutions=[1024]).seed == 0
    assert CropPlanner(config={"seed": 99}, base_resolutions=[1024]).seed == 99
    assert CropPlanner(config={"seed": -1, "crop_plan_seed": 7},
                       base_resolutions=[1024]).seed == 7
    assert "0 if self.run_seed_drawn else self.run_seed" in _BASE_TRAINER_SRC
