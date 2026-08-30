"""A sample control is written only where the architecture reads it.

`TRAINING_SAMPLE_SUPPORTED_PARAMS` already decided which preview controls each
architecture honors, and the frontend already hides the rest. The config
generator and the sample PNG's metadata did not consult it: a SenseNova run
carried `sampler: euler` / `schedule_type: sgm_uniform` in its YAML and stamped
them into every preview PNG, which is not a harmless leftover but a false
statement about how that image was produced -- SenseNova's sample path has no
sampler and no schedule at all, only `timestep_shift`.

Nothing here loads a checkpoint or a database.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/training_sample_arch_gate_test.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml
from PIL import Image

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from api.arch_capabilities import (  # noqa: E402
    TRAINING_SAMPLE_KEY_PARAM, TRAINING_SAMPLE_SUPPORTED_PARAMS,
    training_sample_key_supported,
)
from core.training.base_trainer import BaseTrainer  # noqa: E402
from core.training.training_config import (  # noqa: E402
    _build_sample_section, _GATED_SAMPLE_KEYS_LEADING,
    _GATED_SAMPLE_KEYS_TRAILING,
)

_SD_ONLY = ("sampler", "schedule_type", "cfg_schedule_type", "nag_enable")
_SENSENOVA_ONLY = ("sensenova_timestep_shift", "sensenova_img_cfg_scale",
                   "sensenova_cfg_norm")
_UNGATED = ("sample_every", "width", "height", "prompts", "neg", "seed",
            "guidance_scale", "sample_steps")


# ---------------------------------------------------------------------------
# The table agrees with the handlers that read it
# ---------------------------------------------------------------------------

def test_the_allowlist_names_the_fields_the_arch_handlers_actually_read():
    """The gate is only safe because the table is not aspirational: every key
    it withholds corresponds to a `sample_ctx` field that architecture's
    `sample()` never passes on. Read that off the handlers."""
    arch_dir = BACKEND / "core" / "training" / "arch"
    for key, param in TRAINING_SAMPLE_KEY_PARAM.items():
        ctx_field = key if key.startswith("sensenova_") else key
        readers = {
            f.stem for f in arch_dir.glob("*.py")
            if f.stem not in ("__init__", "base_arch")
            and f"={'sample_ctx.'}{ctx_field}" in f.read_text(encoding="utf-8")
        }
        declared = {a for a, params in TRAINING_SAMPLE_SUPPORTED_PARAMS.items()
                    if param in params}
        assert readers == declared, (key, sorted(readers), sorted(declared))


# ---------------------------------------------------------------------------
# The gate predicate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", _UNGATED + ("prompt", "steps", "cfg_scale"))
def test_an_ungated_key_is_supported_everywhere(key):
    for arch in ("sensenova", "sdxl", "flux2", "unknown", None):
        assert training_sample_key_supported(arch, key)


def test_sensenova_reads_its_own_three_and_none_of_the_sd_controls():
    for key in _SD_ONLY:
        assert not training_sample_key_supported("sensenova", key)
    for key in _SENSENOVA_ONLY:
        assert training_sample_key_supported("sensenova", key)


def test_sdxl_reads_the_sd_controls_and_none_of_sensenovas():
    for key in _SD_ONLY:
        assert training_sample_key_supported("sdxl", key)
    for key in _SENSENOVA_ONLY:
        assert not training_sample_key_supported("sdxl", key)


def test_an_undeclared_architecture_is_not_stripped():
    """_detect_arch answers "unknown" for a model it cannot identify. Absence
    from the table means "not described here", never "supports nothing" --
    stripping an unidentified model's controls would be a silent downgrade."""
    for key in _SD_ONLY + _SENSENOVA_ONLY:
        assert training_sample_key_supported("unknown", key)
        assert training_sample_key_supported(None, key)


# ---------------------------------------------------------------------------
# The generated YAML section
# ---------------------------------------------------------------------------

def _section(arch):
    return _build_sample_section({}, [{"positive": "p", "negative": ""}], arch)


def test_the_sensenova_section_drops_the_controls_it_cannot_honor():
    section = _section("sensenova")
    for key in _SD_ONLY:
        assert key not in section
    for key in _SENSENOVA_ONLY:
        assert key in section
    for key in _UNGATED:
        assert key in section


def test_the_sdxl_section_is_unchanged_apart_from_the_sensenova_keys():
    section = _section("sdxl")
    for key in _SD_ONLY:
        assert key in section
    for key in _SENSENOVA_ONLY:
        assert key not in section


def test_the_supporting_architecture_keeps_the_original_key_order():
    """A regenerated config for an architecture that reads everything must not
    reshuffle its own section."""
    section = _section("unknown")
    expected = list(_GATED_SAMPLE_KEYS_LEADING) + list(_UNGATED) \
        + list(_GATED_SAMPLE_KEYS_TRAILING)
    assert list(section.keys()) == expected


def test_a_flux2_section_carries_only_the_universal_keys():
    assert list(_section("flux2").keys()) == list(_UNGATED)


def test_the_three_generators_all_go_through_the_gate():
    source = (BACKEND / "core" / "training"
              / "training_config.py").read_text(encoding="utf-8")
    assert source.count('"sample": _build_sample_section(') == 3


# ---------------------------------------------------------------------------
# The sample PNG's metadata
# ---------------------------------------------------------------------------

class _StubArch:
    def __init__(self, name):
        self.name = name


class _StubTrainer:
    _save_sample_with_metadata = BaseTrainer._save_sample_with_metadata

    def __init__(self, arch_name):
        self.arch = _StubArch(arch_name)


def _write_sample(tmp_path, arch_name):
    trainer = _StubTrainer(arch_name)
    path = tmp_path / "s.png"
    trainer._save_sample_with_metadata(
        Image.new("RGB", (8, 8)), path,
        prompt="p", negative_prompt="", steps=50, cfg_scale=4.0, seed=1234,
        width=1024, height=1024,
        sampler="euler", schedule_type="sgm_uniform",
        cfg_schedule_type="constant", cfg_schedule_min=1.0,
        cfg_schedule_max=None, cfg_schedule_power=1.0,
        cfg_rescale_snr_alpha=0.0,
        dynamic_threshold_percentile=0.0, dynamic_threshold_mimic_scale=7.0,
        nag_enable=False, nag_scale=1.0, nag_tau=1.0, nag_alpha=1.0,
        nag_sigma_end=3.0, nag_negative_prompt="",
        sensenova_timestep_shift=3.0, sensenova_img_cfg_scale=1.0,
        sensenova_cfg_norm="global",
    )
    with Image.open(path) as im:
        return dict(im.info)


def test_a_sensenova_preview_does_not_claim_a_sampler(tmp_path):
    info = _write_sample(tmp_path, "sensenova")
    for key in _SD_ONLY:
        assert key not in info, key
    assert info["sensenova_timestep_shift"] == "3.0"
    assert info["sensenova_cfg_norm"] == "global"


def test_a_sensenova_preview_keeps_what_actually_produced_it(tmp_path):
    info = _write_sample(tmp_path, "sensenova")
    assert info["prompt"] == "p"
    assert info["steps"] == "50"
    assert info["cfg_scale"] == "4.0"
    assert info["seed"] == "1234"
    assert info["width"] == "1024" and info["height"] == "1024"
    assert info["negative_prompt"] == ""


def test_an_sdxl_preview_keeps_the_sd_controls(tmp_path):
    info = _write_sample(tmp_path, "sdxl")
    assert info["sampler"] == "euler"
    assert info["schedule_type"] == "sgm_uniform"
    for key in _SENSENOVA_ONLY:
        assert key not in info


def test_an_unidentified_architecture_still_records_everything(tmp_path):
    info = _write_sample(tmp_path, None)
    for key in _SD_ONLY + _SENSENOVA_ONLY:
        assert key in info


def test_the_yaml_and_the_metadata_cannot_disagree_about_a_control(tmp_path):
    """Same predicate on both sides: a control is either written in the config
    and recorded in the PNG, or in neither."""
    for arch in ("sensenova", "sdxl", "flux2"):
        section = _section(arch)
        info = _write_sample(tmp_path, arch)
        for key in TRAINING_SAMPLE_KEY_PARAM:
            assert (key in section) == (key in info), (arch, key)


def test_the_generated_yaml_round_trips(tmp_path):
    """The section is dumped into the run's config file; a dropped key must not
    leave the document malformed."""
    text = yaml.dump({"sample": _section("sensenova")}, sort_keys=False,
                     allow_unicode=True)
    assert yaml.safe_load(text)["sample"] == _section("sensenova")
