"""CPU contracts for the SigLIP2 resolution-ladder runner."""

from PIL import Image

from core.tagger.foveatag.benchmark import BenchmarkItem
from core.tagger.foveatag.siglip2_prediction_runner import resize_benchmark_image


def test_resize_benchmark_image_preserves_native_and_uses_requested_size():
    item = BenchmarkItem.from_dict({
        "sample_id": "x", "image_path": "x.png", "width": 80, "height": 40,
        "positive_tags": [], "fine_detail_tags": [], "coarse_tags": [], "evidence": [],
    })
    image = Image.new("RGB", (80, 40))
    native = resize_benchmark_image(image, item, "native", 80, 40)
    reduced = resize_benchmark_image(image, item, "32", 32, 16)
    assert native.size == (80, 40)
    assert native is not image
    assert reduced.size == (32, 16)
