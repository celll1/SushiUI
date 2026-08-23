"""Image per-architecture defaults overlay (`IMAGE_GEN_ARCH_OVERLAYS`).

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/image_arch_overlay_test.py -v

WHY THIS FILE EXISTS
--------------------
The entire safety case for the image-defaults overlay mechanism is "pure
no-op for every non-SenseNova architecture": `steps`/`cfg_scale` used to be
the shared 20/7.0 for every image route, and only SenseNova now resolves to
its own upstream operating point (50/4.0). This file pins that claim rather
than merely asserting it, mirroring
`minimax_music3_api_defaults_test.py`'s no-op proof for the audio twin.
"""

import os
import sys
from typing import get_args

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.generation_utils import resolve_image_defaults  # noqa: E402
from api.param_defaults import (  # noqa: E402
    GENERATION_DEFAULTS,
    IMAGE_GEN_ARCH_OVERLAYS,
    OUTPAINT_DEFAULTS,
    SENSENOVA_GENERATION_DEFAULTS,
    image_defaults_for_arch,
)
from core.model_loader import ModelType  # noqa: E402

# Every image-generation architecture other than SenseNova, enumerated from
# the authoritative `ModelType` literal rather than hardcoded, so a future
# architecture addition is covered automatically.
_ALL_ARCHS = get_args(ModelType)
_NON_IMAGE_ARCHS = {"ltx2", "acestep", "minimax_h3", "minimax_music3"}
_NON_SENSENOVA_IMAGE_ARCHS = [
    arch for arch in _ALL_ARCHS if arch not in _NON_IMAGE_ARCHS and arch != "sensenova"
]


# --------------------------------------------------------------------------
# `image_defaults_for_arch`: the no-op proof
# --------------------------------------------------------------------------

@pytest.mark.parametrize("arch", _NON_SENSENOVA_IMAGE_ARCHS + [None, "", "not_an_arch"])
def test_non_sensenova_arch_resolves_to_generation_defaults_unchanged(arch):
    assert image_defaults_for_arch(arch) == GENERATION_DEFAULTS


def test_image_defaults_for_arch_returns_a_copy_not_the_same_object():
    resolved = image_defaults_for_arch("sdxl")
    assert resolved is not GENERATION_DEFAULTS
    resolved["steps"] = 999
    assert GENERATION_DEFAULTS["steps"] != 999


def test_sensenova_overlay_resolves_to_its_own_operating_point():
    resolved = image_defaults_for_arch("sensenova")
    assert resolved["steps"] == SENSENOVA_GENERATION_DEFAULTS["steps"]
    assert resolved["cfg_scale"] == SENSENOVA_GENERATION_DEFAULTS["cfg_scale"]
    # Differs from GENERATION_DEFAULTS in EXACTLY steps/cfg_scale -- every
    # other key is untouched by the overlay.
    diff_keys = {
        key for key in GENERATION_DEFAULTS
        if resolved.get(key) != GENERATION_DEFAULTS.get(key)
    }
    assert diff_keys == {"steps", "cfg_scale"}


def test_sensenova_is_the_only_overlay_entry_today():
    assert set(IMAGE_GEN_ARCH_OVERLAYS.keys()) == {"sensenova"}


def test_outpaint_base_variant_resolves_the_same_way():
    """`resolve_image_defaults`'s `base=OUTPAINT_DEFAULTS` variant (used by
    /generate/outpaint) composes the same overlay on top of a different base.
    """
    resolved = image_defaults_for_arch("sensenova", OUTPAINT_DEFAULTS)
    assert resolved["steps"] == SENSENOVA_GENERATION_DEFAULTS["steps"]
    assert resolved["cfg_scale"] == SENSENOVA_GENERATION_DEFAULTS["cfg_scale"]
    for key, value in OUTPAINT_DEFAULTS.items():
        if key in ("steps", "cfg_scale"):
            continue
        assert resolved[key] == value

    assert image_defaults_for_arch("sdxl", OUTPAINT_DEFAULTS) == OUTPAINT_DEFAULTS


# --------------------------------------------------------------------------
# `resolve_image_defaults`: the route-facing contract
# --------------------------------------------------------------------------

def test_resolve_image_defaults_fills_only_client_omitted_fields():
    params = {"steps": None, "cfg_scale": 9.0, "prompt": "hello"}
    # Client explicitly sent cfg_scale=9.0; steps was omitted (Form(None)).
    provided_keys = {"cfg_scale", "prompt"}
    params["steps"] = SENSENOVA_GENERATION_DEFAULTS["steps"]

    resolved = resolve_image_defaults(params, provided_keys, "sensenova")

    assert resolved["steps"] == SENSENOVA_GENERATION_DEFAULTS["steps"]
    assert resolved["cfg_scale"] == SENSENOVA_GENERATION_DEFAULTS["cfg_scale"]
    # Omitted field got the arch's resolved default...
    assert params["steps"] == SENSENOVA_GENERATION_DEFAULTS["steps"]
    # ...but the explicitly-provided field is untouched.
    assert params["cfg_scale"] == 9.0
    # A field with no entry in the resolved map is never touched at all.
    assert params["prompt"] == "hello"


def test_resolve_image_defaults_is_a_noop_for_non_sensenova_when_nothing_is_provided():
    params = {"steps": GENERATION_DEFAULTS["steps"], "cfg_scale": GENERATION_DEFAULTS["cfg_scale"]}
    before = dict(params)
    resolved = resolve_image_defaults(params, provided_keys=set(), arch="sdxl")
    assert resolved == GENERATION_DEFAULTS
    assert params == before


def test_resolve_image_defaults_never_touches_keys_outside_steps_cfg_scale():
    """Mirrors the routes' actual calling convention (`provided_keys =
    set(params) - omitted`, see routes.py's `/generate/txt2img`): every key
    besides a genuinely-omitted `steps`/`cfg_scale` counts as "provided" and
    must survive untouched, even though `resolved` (the overlay applied on
    top of the FULL `GENERATION_DEFAULTS` shape) technically contains those
    keys too.
    """
    params = {
        "steps": None,
        "cfg_scale": None,
        "width": 1024,
        "height": 1024,
        "seed": -1,
    }
    omitted = {key for key, value in params.items() if value is None}
    resolve_image_defaults(params, set(params) - omitted, arch="sensenova")
    assert params["width"] == 1024
    assert params["height"] == 1024
    assert params["seed"] == -1
    assert params["steps"] == SENSENOVA_GENERATION_DEFAULTS["steps"]
    assert params["cfg_scale"] == SENSENOVA_GENERATION_DEFAULTS["cfg_scale"]
