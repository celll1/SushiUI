"""Per-exposure image noise, including nested reference captures."""
from contextlib import contextmanager
from contextvars import ContextVar

_NOISE = ContextVar("codec_image_noise", default=None)


@contextmanager
def noise_scope(noise):
    token = _NOISE.set(noise)
    try:
        yield
    finally:
        _NOISE.reset(token)


def exposure_noise(image):
    noise = _NOISE.get()
    if (noise is None or noise.ndim != 4 or image.ndim != 4 or
            noise.shape[1:] != image.shape[1:] or noise.shape[0] < 1 or
            image.shape[0] % noise.shape[0]):
        raise ValueError("Aligned image prediction requires this exposure's BCHW noise")
    return noise.repeat(image.shape[0] // noise.shape[0], 1, 1, 1).to(image.device)
