"""Which VAE decodes a MiniMax-H3 request: the T=1 image VAE preference.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_h3_image_vae_decode_test.py -v

WHY THIS FILE EXISTS
--------------------
`select_minimax_h3_decode_vae` (`core/pipeline_backends/minimax_h3.py`)
chooses which VAE decodes a request: the optional, measurably better
`image_vae` for a still-image (`latent_frames == 1`) request when installed,
falling back to the video VAE's own T=1 branch otherwise, and always the
video VAE for ordinary video. It is a small, named, pure function precisely
so that choice is testable here without a model load.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.pipeline_backends.minimax_h3 import select_minimax_h3_decode_vae  # noqa: E402


def test_still_image_prefers_the_image_vae_when_present():
    vae = object()
    image_vae = object()
    components = {"vae": vae, "image_vae": image_vae}

    selected, name, used_fallback = select_minimax_h3_decode_vae(components, latent_frames=1)

    assert selected is image_vae
    assert name == "image_vae"
    assert used_fallback is False


def test_still_image_falls_back_to_the_video_vae_when_image_vae_absent():
    vae = object()
    components = {"vae": vae, "image_vae": None}

    selected, name, used_fallback = select_minimax_h3_decode_vae(components, latent_frames=1)

    assert selected is vae
    assert name == "vae"
    assert used_fallback is True


def test_still_image_falls_back_when_the_image_vae_key_is_missing_entirely():
    """A component dict built before this feature shipped has no `image_vae`
    key at all (not even `None`) -- `.get` must still resolve, not KeyError."""
    vae = object()
    components = {"vae": vae}

    selected, name, used_fallback = select_minimax_h3_decode_vae(components, latent_frames=1)

    assert selected is vae
    assert name == "vae"
    assert used_fallback is True


def test_ordinary_video_never_uses_the_image_vae_or_warns():
    vae = object()
    image_vae = object()
    components = {"vae": vae, "image_vae": image_vae}

    for latent_frames in (2, 7, 22, 362):
        selected, name, used_fallback = select_minimax_h3_decode_vae(components, latent_frames)
        assert selected is vae
        assert name == "vae"
        assert used_fallback is False
