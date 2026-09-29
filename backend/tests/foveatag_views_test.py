"""CPU geometry tests for FoveaTag multi-scale views."""

import math

from PIL import Image

from core.tagger.foveatag.views import (
    ViewConfig,
    build_views,
    local_spec,
    identity_overview_spec,
    overview_spec,
    pil_to_tensor,
    render_view,
    render_overview_placebo_locals,
    sample_local_specs,
    stack_view_tensors,
    without_local_views,
)


def test_overview_never_enlarges_and_round_trips_coordinates():
    spec = overview_spec(100, 50, 448)
    assert spec.content_box == (174.0, 199.0, 100.0, 50.0)
    assert spec.pixels_per_canvas_pixel == 1.0

    canvas = spec.source_to_canvas(23.0, 17.0)
    source = spec.canvas_to_source(*canvas)
    assert source == (23.0, 17.0)
    geometry = spec.normalized_geometry(100, 50)
    assert geometry[6:] == (174 / 448, 199 / 448, 100 / 448, 50 / 448)


def test_identity_overview_matches_unmodified_source_canvas():
    spec = identity_overview_spec(1024, 512)

    assert spec.role == "baseline-overview"
    assert (spec.canvas_width, spec.canvas_height) == (1024, 512)
    assert spec.content_box == (0.0, 0.0, 1024.0, 512.0)
    assert spec.source_to_canvas(31.0, 17.0) == (31.0, 17.0)


def test_local_two_x_magnification_maps_128_source_pixels_to_256_canvas():
    spec = local_spec(
        center_x=100,
        center_y=80,
        canvas_size=256,
        pixels_per_canvas_pixel=0.5,
    )
    assert spec.source_box.width == 128
    assert spec.source_to_canvas(36, 16) == (0, 0)
    assert spec.source_to_canvas(164, 144) == (256, 256)


def test_render_view_pads_outside_source_without_shifting_coordinates():
    image = Image.new("RGB", (32, 32), (10, 20, 30))
    spec = local_spec(
        center_x=0,
        center_y=0,
        canvas_size=32,
        pixels_per_canvas_pixel=1.0,
    )
    rendered = render_view(image, spec, pad_color=(255, 255, 255))

    assert rendered.getpixel((0, 0)) == (255, 255, 255)
    assert rendered.getpixel((16, 16)) == (10, 20, 30)


def test_training_sampler_is_reproducible_and_uses_multiple_scales():
    config = ViewConfig(local_views=6, scale_jitter=(0.8, 1.2))
    first = sample_local_specs(2048, 1024, config, seed=123, training=True)
    second = sample_local_specs(2048, 1024, config, seed=123, training=True)

    assert first == second
    densities = {round(spec.pixels_per_canvas_pixel, 3) for spec in first}
    assert len(densities) >= 3
    assert min(densities) >= 0.5


def test_build_views_and_tensor_stack_have_fixed_budget():
    image = Image.new("RGB", (1024, 512), (64, 128, 255))
    config = ViewConfig(overview_size=448, local_size=256, local_views=4)
    views = build_views(image, config, seed=1, training=False)
    overview, local = stack_view_tensors(views)

    assert overview.shape == (1, 3, 448, 448)
    assert local.shape == (4, 3, 256, 256)
    assert math.isclose(float(pil_to_tensor(views.local_views[0]).max()), 1.0)


def test_zero_local_view_batch_retains_local_canvas_contract():
    image = Image.new("RGB", (64, 32), (0, 0, 0))
    views = build_views(image, ViewConfig(local_size=192, local_views=0), seed=0, training=False)
    _, local = stack_view_tensors(views)
    assert local.shape == (0, 3, 192, 192)


def test_overview_placebo_retains_geometry_without_source_native_detail():
    image = Image.new("RGB", (1024, 1024), (0, 0, 0))
    for y in range(384, 640):
        for x in range(384, 640):
            value = 255 if (x + y) % 2 else 0
            image.putpixel((x, y), (value, value, value))
    config = ViewConfig(overview_size=64, local_size=64, local_views=1)
    specification = local_spec(
        center_x=512,
        center_y=512,
        canvas_size=64,
        pixels_per_canvas_pixel=1.0,
    )
    native = build_views(
        image,
        config,
        seed=0,
        training=False,
        specs=(specification,),
    )

    placebo = render_overview_placebo_locals(native)

    assert placebo.local_specs == native.local_specs
    assert placebo.local_views[0].size == native.local_views[0].size
    assert native.local_views[0].getextrema()[0] == (0, 255)
    assert placebo.local_views[0].getextrema()[0] == (128, 128)
    assert placebo.local_views[0].tobytes() != native.local_views[0].tobytes()


def test_without_local_views_keeps_overview_contract():
    views = build_views(
        Image.new("RGB", (320, 240), (1, 2, 3)),
        ViewConfig(overview_size=64, local_size=32, local_views=2),
        seed=0,
        training=False,
    )

    overview_only = without_local_views(views)

    assert overview_only.overview is views.overview
    assert overview_only.overview_spec == views.overview_spec
    assert overview_only.local_views == ()
    assert overview_only.local_specs == ()
    assert overview_only.local_canvas_size == 32
