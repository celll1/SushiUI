"""Per-run GPU selection: YAML key -> spawn env, and the start-time guard.

The child is pinned with CUDA_VISIBLE_DEVICES rather than a device argument,
so these tests are the only thing standing between a wrong index and a run that
silently trains on the wrong card (or on no card at all).
"""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.param_defaults import TRAINING_DEFAULTS  # noqa: E402
from core.training.training_config import (  # noqa: E402
    TrainingConfigGenerator,
    train_section_key_vocabulary,
)
from core.training.training_process import (  # noqa: E402
    TrainingProcessManager,
    resolve_cuda_visible_devices,
)


def _train_section(config_yaml: str) -> dict:
    return yaml.safe_load(config_yaml)["config"]["process"][0]["train"]


def test_default_is_none_so_existing_runs_keep_inheriting_every_gpu():
    """MUTANT: default 0. Every pre-existing run would suddenly be pinned to
    device 0 -- correct on a single-GPU box, wrong the moment a second card
    exists, and invisible either way."""
    assert TRAINING_DEFAULTS["gpu_index"] is None


def test_none_leaves_the_parent_environment_untouched():
    assert resolve_cuda_visible_devices(None, None) is None
    assert resolve_cuda_visible_devices("0,1", None) is None


def test_index_is_used_verbatim_when_the_parent_sees_every_gpu():
    assert resolve_cuda_visible_devices(None, 0) == "0"
    assert resolve_cuda_visible_devices("", 2) == "2"


def test_index_composes_with_an_inherited_visible_list():
    """MUTANT: return str(gpu_index) unconditionally. With the backend started
    as CUDA_VISIBLE_DEVICES=2,3, picking index 1 means physical GPU 3; writing
    "1" would hand the child physical GPU 1 -- a card the UI never offered."""
    assert resolve_cuda_visible_devices("2,3", 0) == "2"
    assert resolve_cuda_visible_devices("2,3", 1) == "3"
    assert resolve_cuda_visible_devices(" 4 , 5 ", 1) == "5"


def test_index_past_the_inherited_list_is_refused():
    """An out-of-range index would export a device the parent cannot see, and
    the child would find zero CUDA devices and fall back to CPU."""
    with pytest.raises(ValueError, match="outside the backend's"):
        resolve_cuda_visible_devices("2,3", 2)


@pytest.mark.parametrize(
    "method",
    ["lora", "relora", "full_finetune", "controlnet", "vae_decoder"],
)
def test_every_generator_emits_gpu_index(method):
    """MUTANT: emit it only from _build_train_section. generate_vae_config
    builds its own train literal, so VAE runs would drop the selection with no
    error -- the run just starts on the default GPU."""
    generator = TrainingConfigGenerator()
    params = {"gpu_index": 1, "learning_rate": 1e-4}
    kwargs = dict(
        run_name="t",
        base_model_path="m.safetensors",
        dataset_path="d",
        output_dir="o",
        total_steps=10,
    )
    if method == "vae_decoder":
        params["vae_config"] = {}
        config_yaml = generator.generate_vae_config(params, **kwargs)
    else:
        config_yaml = getattr(generator, f"generate_{method}_config")(params, **kwargs)
    assert _train_section(config_yaml)["gpu_index"] == 1


def test_gpu_index_survives_a_config_regeneration():
    """PUT /training/runs/{id} rebuilds the YAML from the request model and
    drops any train key outside the generators' vocabulary."""
    assert "gpu_index" in train_section_key_vocabulary()


def test_create_process_forwards_the_index_to_the_child():
    manager = TrainingProcessManager()
    process = manager.create_process(
        run_id=1, config_path="c.yaml", output_dir="o", gpu_index=3
    )
    assert process.gpu_index == 3


def test_create_process_defaults_to_no_pinning():
    """MUTANT: make gpu_index a required argument. Every existing caller and
    test breaks, and a run started through an older path would raise instead of
    inheriting the previous behaviour."""
    manager = TrainingProcessManager()
    process = manager.create_process(run_id=1, config_path="c.yaml", output_dir="o")
    assert process.gpu_index is None
