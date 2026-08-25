import hashlib
from io import BytesIO
from types import SimpleNamespace

import numpy as np
from PIL import Image

from core.training.image_preprocessing import flatten_to_rgb
from core.training.base_trainer import BaseTrainer
from core.training.latent_cache import LatentCache
from core.training.vae.vae_dataset import load_image_tensor


def _transparent_webp() -> BytesIO:
    image = Image.new("RGBA", (2, 2), (19, 37, 83, 0))
    image.putpixel((1, 0), (200, 100, 50, 255))
    data = BytesIO()
    image.save(data, format="WEBP", lossless=True)
    data.seek(0)
    return data


def test_flatten_to_rgb_composites_transparent_webp_over_white():
    with Image.open(_transparent_webp()) as image:
        result = flatten_to_rgb(image)

    assert result.mode == "RGB"
    assert result.getpixel((0, 0)) == (255, 255, 255)
    assert result.getpixel((1, 0)) == (200, 100, 50)


def test_flatten_to_rgb_blends_partial_alpha_instead_of_dropping_it():
    image = Image.new("RGBA", (1, 1), (255, 0, 0, 128))

    assert flatten_to_rgb(image).getpixel((0, 0)) == (255, 127, 127)


def test_vae_dataset_uses_white_for_transparent_webp_pixels(tmp_path):
    path = tmp_path / "transparent.webp"
    path.write_bytes(_transparent_webp().getvalue())

    tensor = load_image_tensor(path, resolution=2, random_crop=False)
    pixels = tensor.permute(1, 2, 0).numpy()

    np.testing.assert_array_equal(pixels[0, 0], np.ones(3, dtype=np.float32))


def test_standard_training_encoder_uses_white_for_transparent_webp_pixels():
    trainer = SimpleNamespace(
        is_minit2i=True,
        is_sensenova=False,
        arch=SimpleNamespace(vae_encode=lambda _trainer, tensor, **_kwargs: tensor),
    )
    with Image.open(_transparent_webp()) as image:
        tensor = BaseTrainer.encode_image(
            trainer, image, target_width=2, target_height=2, bucket_strategy="resize"
        )

    np.testing.assert_array_equal(
        tensor[0, :, 0, 0].numpy(), np.ones(3, dtype=np.float32)
    )


def test_webp_latent_keys_do_not_reuse_the_alpha_dropping_cache():
    legacy = hashlib.md5("image.webp_512_512".encode()).hexdigest()

    assert LatentCache.compute_image_hash("image.webp", 512, 512) != legacy
    assert LatentCache.compute_image_hash("image.png", 512, 512) == hashlib.md5(
        "image.png_512_512".encode()
    ).hexdigest()
