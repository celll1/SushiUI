"""CPU tests for FoveaTag multi-view batching."""

import torch
from PIL import Image

from core.tagger.foveatag.batch import collate_view_batches
from core.tagger.foveatag.views import ViewConfig, build_views


def test_collator_pads_variable_view_counts_and_preserves_masks():
    first = build_views(
        Image.new("RGB", (80, 40), (255, 0, 0)),
        ViewConfig(overview_size=32, local_size=16, local_views=2),
        seed=1,
        training=False,
    )
    second = build_views(
        Image.new("RGB", (40, 80), (0, 255, 0)),
        ViewConfig(overview_size=32, local_size=16, local_views=1),
        seed=2,
        training=False,
    )
    batch = collate_view_batches([first, second])

    assert batch.overview_images.shape == (2, 3, 32, 32)
    assert batch.overview_geometry.shape == (2, 10)
    assert batch.local_images.shape == (2, 2, 3, 16, 16)
    assert batch.local_geometry.shape == (2, 2, 10)
    assert batch.local_view_mask.tolist() == [[True, True], [True, False]]
    assert torch.count_nonzero(batch.local_images[1, 1]) == 0


def test_overview_geometry_contains_letterbox_content_bounds():
    views = build_views(
        Image.new("RGB", (100, 50)),
        ViewConfig(overview_size=200, local_views=0),
        seed=0,
        training=False,
    )
    batch = collate_view_batches([views])

    torch.testing.assert_close(
        batch.overview_geometry[0, 6:],
        torch.tensor([0.25, 0.375, 0.5, 0.25]),
    )


def test_empty_local_batch_retains_canvas_dimensions():
    views = build_views(
        Image.new("RGB", (32, 32)),
        ViewConfig(overview_size=32, local_size=24, local_views=0),
        seed=0,
        training=False,
    )
    batch = collate_view_batches([views])

    assert batch.local_images.shape == (1, 0, 3, 24, 24)
    assert batch.local_geometry.shape == (1, 0, 10)
