"""A model-list entry for either H3 partition must dispatch to the H3 loader.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_h3_load_dispatch_test.py -v

WHY THIS FILE EXISTS
--------------------
An urgent-defect investigation (2026-08-08) suspected that `GET /models`
listing a DiT single file (`adf751d1`) handed the loader a shape it could not
consume, since the tree-root scan case was new. That was checked and refuted:
`ModelLoader.detect_model_type` and `detect_minimax_h3_layout` already resolve
a bare DiT `.safetensors` path (the exact `path` a listing entry carries) to
the right component tree and variant -- confirmed against the real on-disk
model (`M:/model/minimax_h3`) and pinned here with synthetic header-only files
so it does not need the real 21 GB weights to run in CI. The actual retry
succeeded, corroborating this.

This only covers plumbing (detection + layout resolution), not weight
loading: nothing here reads tensor bytes or needs a GPU.
"""

import json
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.model_loader import ModelLoader  # noqa: E402
from core.models.minimax_h3.loader import detect_minimax_h3_layout  # noqa: E402


def _write_fake_h3_dit(path: str) -> None:
    """Header-only file carrying the H3 key signature (see the listing test)."""
    header = {
        "token_refiner.0.weight": {"dtype": "F32", "shape": [1, 1], "data_offsets": [0, 0]},
        "adaln_t_table": {"dtype": "F32", "shape": [1], "data_offsets": [0, 0]},
        "__metadata__": {"format": "pt"},
    }
    header_bytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)


def _build_h3_tree(root: str) -> None:
    dit_dir = os.path.join(root, "diffusion_models")
    os.makedirs(dit_dir, exist_ok=True)
    _write_fake_h3_dit(os.path.join(dit_dir, "minimax_h3_fl2va_pruned_fp8_scaled.safetensors"))
    _write_fake_h3_dit(os.path.join(dit_dir, "minimax_h3_ref2va_pruned_fp8_scaled.safetensors"))

    official_dir = os.path.join(root, "official")
    os.makedirs(official_dir, exist_ok=True)
    with open(os.path.join(official_dir, "model_index.json"), "w", encoding="utf-8") as f:
        json.dump({"_class_name": "MiniMaxH3ModularPipeline"}, f)

    os.makedirs(os.path.join(root, "text_encoders"), exist_ok=True)
    os.makedirs(os.path.join(root, "vae"), exist_ok=True)


def test_a_listing_entrys_path_detects_as_minimax_h3(tmp_path):
    """`_expand_minimax_h3_tree`'s `path` field is exactly a DiT file path."""
    root = str(tmp_path / "minimax_h3")
    _build_h3_tree(root)
    for variant in ("fl2va", "ref2va"):
        dit = os.path.join(root, "diffusion_models", f"minimax_h3_{variant}_pruned_fp8_scaled.safetensors")
        assert ModelLoader.detect_model_type(dit) == "minimax_h3"


def test_a_listing_entrys_path_resolves_the_right_variant_and_siblings(tmp_path):
    """Layout resolution walks up from the file to the shared tree correctly."""
    root = str(tmp_path / "minimax_h3")
    _build_h3_tree(root)
    for variant in ("fl2va", "ref2va"):
        dit = os.path.join(root, "diffusion_models", f"minimax_h3_{variant}_pruned_fp8_scaled.safetensors")
        layout = detect_minimax_h3_layout(dit)
        assert layout is not None
        assert layout["variant"] == variant
        assert layout["dit"] == dit
        assert layout["official"] == os.path.join(root, "official")


def test_removing_the_safetensors_dispatch_branch_reproduces_the_defect(tmp_path, monkeypatch):
    """Mutant: `detect_model_type` never returns "minimax_h3" for a DiT file.

    This is the shape the hypothesis under investigation described: a listing
    entry whose `path` the loader cannot recognise, so it falls through to the
    SD1.5/SDXL reconstruction path instead.
    """
    root = str(tmp_path / "minimax_h3")
    _build_h3_tree(root)
    dit = os.path.join(root, "diffusion_models", "minimax_h3_ref2va_pruned_fp8_scaled.safetensors")

    real_looks_like = ModelLoader._looks_like_minimax_h3
    monkeypatch.setattr(ModelLoader, "_looks_like_minimax_h3",
                         staticmethod(lambda path: False if path == dit else real_looks_like(path)))

    assert ModelLoader.detect_model_type(dit) != "minimax_h3"


def _touch(path: str) -> None:
    """A zero-byte stand-in file. Layout resolution only checks existence."""
    with open(path, "wb"):
        pass


def test_image_vae_detected_when_present(tmp_path):
    """The optional still-image VAE resolves into its own layout slot."""
    root = str(tmp_path / "minimax_h3")
    _build_h3_tree(root)
    vae_dir = os.path.join(root, "vae")
    _touch(os.path.join(vae_dir, "minimax_h3_video_vae_fp16.safetensors"))
    _touch(os.path.join(vae_dir, "minimax_h3_audio_vae_fp32.safetensors"))
    image_vae_path = os.path.join(vae_dir, "minimax_h3_t1_image_vae_step1597.safetensors")
    _touch(image_vae_path)

    layout = detect_minimax_h3_layout(root)
    assert layout is not None
    assert layout["image_vae"] == image_vae_path
    assert layout["vae"] == os.path.join(vae_dir, "minimax_h3_video_vae_fp16.safetensors")
    assert layout["audio_vae"] == os.path.join(vae_dir, "minimax_h3_audio_vae_fp32.safetensors")


def test_image_vae_none_when_absent(tmp_path):
    """Every install without the community checkpoint keeps resolving as before."""
    root = str(tmp_path / "minimax_h3")
    _build_h3_tree(root)
    vae_dir = os.path.join(root, "vae")
    _touch(os.path.join(vae_dir, "minimax_h3_video_vae_fp16.safetensors"))
    _touch(os.path.join(vae_dir, "minimax_h3_audio_vae_fp32.safetensors"))

    layout = detect_minimax_h3_layout(root)
    assert layout is not None
    assert layout["image_vae"] is None


def test_image_vae_three_way_collision_guard(tmp_path):
    """A lone file that satisfies the image VAE's own accept predicate must not
    ALSO be silently treated as the video (or audio) VAE, or vice versa.

    Set up so it is: with neither a literal ``minimax_h3_video_vae_*`` nor
    ``minimax_h3_audio_vae_*`` file on disk, ``_find_first``'s generic glob
    fallback (no ``accept`` predicate for those two slots) hands the video and
    audio VAE slots this SAME lone file, because it is the only
    ``*minimax_h3*.safetensors`` file present. The guard must then null the
    ``image_vae`` slot rather than let it double as the video/audio VAE.
    """
    root = str(tmp_path / "minimax_h3")
    _build_h3_tree(root)
    vae_dir = os.path.join(root, "vae")
    image_vae_path = os.path.join(vae_dir, "minimax_h3_t1_image_vae_step1597.safetensors")
    _touch(image_vae_path)

    layout = detect_minimax_h3_layout(root)
    assert layout is not None
    # The generic fallback (no accept predicate) resolved "vae" to this same
    # file -- a pre-existing property of `_find_first`, exercised here on
    # purpose to prove the guard actually fires.
    assert layout["vae"] == image_vae_path
    assert layout["image_vae"] is None
