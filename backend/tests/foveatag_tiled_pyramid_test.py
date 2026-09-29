"""Region-bounded source reads for staged FoveaTag inference."""

import json

import numpy as np
import pytest
from PIL import Image

from core.tagger.foveatag.tiled_pyramid import TiledPyramidSource, build_tiled_pyramid
from core.tagger.foveatag.views import ViewConfig, build_views, local_spec, render_view


def _gradient(width, height):
    y, x = np.mgrid[:height, :width]
    pixels = np.stack((x % 256, y % 256, (x + y) % 256), axis=-1).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def test_tiled_source_preserves_native_local_pixels_and_padding(tmp_path):
    path = tmp_path / "source.png"
    image = _gradient(513, 377)
    image.save(path)
    pyramid = build_tiled_pyramid(
        path,
        tmp_path / "pyramid",
        tile_size=128,
        coarsest_max_side=64,
    )
    source = TiledPyramidSource(pyramid)
    specification = local_spec(
        center_x=127.5,
        center_y=128.5,
        canvas_size=128,
        pixels_per_canvas_pixel=1.0,
    )

    expected = render_view(image, specification)
    actual = source.render_view(specification)
    assert actual.tobytes() == expected.tobytes()
    padded = source.render_view(local_spec(
        center_x=0,
        center_y=0,
        canvas_size=64,
        pixels_per_canvas_pixel=1.0,
    ))
    assert padded.getpixel((0, 0)) == (255, 255, 255)


def test_overview_and_selected_local_read_bounded_tiles(tmp_path, monkeypatch):
    path = tmp_path / "source.png"
    _gradient(1024, 768).save(path)
    source = TiledPyramidSource(build_tiled_pyramid(
        path,
        tmp_path / "pyramid",
        tile_size=128,
        coarsest_max_side=64,
    ))
    reads = []
    original = source._read_tile

    def counted(level, column, row):
        reads.append((level, column, row))
        return original(level, column, row)

    monkeypatch.setattr(source, "_read_tile", counted)
    specification = local_spec(
        center_x=512,
        center_y=384,
        canvas_size=128,
        pixels_per_canvas_pixel=1.0,
    )
    views = build_views(
        source,
        ViewConfig(overview_size=128, local_size=128, local_views=1),
        seed=0,
        training=False,
        specs=(specification,),
    )

    assert views.overview.size == (128, 128)
    assert views.local_views[0].size == (128, 128)
    assert len(reads) <= 5
    assert any(level == 0 for level, _column, _row in reads)
    assert any(level > 0 for level, _column, _row in reads)


def test_tiled_source_rejects_inconsistent_manifest(tmp_path):
    root = tmp_path / "bad"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({
        "format_version": 1,
        "kind": "foveatag_tiled_pyramid",
        "source": {"width": 32, "height": 32, "sha256": "x"},
        "tile_size": 16,
        "levels": [{"index": 0, "width": 31, "height": 32, "columns": 2, "rows": 2}],
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="level zero"):
        TiledPyramidSource(root)
