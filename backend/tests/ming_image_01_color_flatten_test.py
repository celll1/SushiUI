"""Native decode chroma smoothing keeps alpha and resets request state."""
import numpy as np
import pytest
import torch
from ming_image_01_sampling_test import make_sampler, condition, options
from core.models.ming_image_01.postprocessing import MingImageProcessor
from core.inference.color_flatten import flatten_chroma


def test_shared_rgb_oracle_and_unchanged_alpha():
    pixels = torch.rand(2, 4, 32, 32) * 2 - 1
    processor = MingImageProcessor()
    original = processor.postprocess(pixels, output_type="pt")
    processor.color_flatten_strength = 65
    actual = processor.postprocess(pixels, output_type="pt")
    torch.testing.assert_close(actual[:, :3], flatten_chroma(original[:, :3], 65).clamp(0, 1), rtol=0, atol=0)
    assert torch.equal(actual[:, 3:], original[:, 3:])
    assert not torch.equal(actual[:, :3], original[:, :3])
    processor.color_flatten_strength = 0
    assert torch.equal(processor.postprocess(pixels, output_type="pt"), original)


@pytest.mark.parametrize("profile", ["generation_edit", "layer_decompose"])
def test_native_generation_applies_all_frames_without_alpha_changes(profile):
    sampler = make_sampler(profile)
    kwargs = dict(task="text-to-image" if profile == "generation_edit" else "layer-decompose",
                  conditioning=condition(), **options())
    if profile == "layer_decompose":
        kwargs.update(reference_pixels=torch.randn(1, 4, 32, 32), requested_layers=2)
    baseline = sampler.run(**kwargs)[0]
    flattened = sampler.run(**kwargs, color_flatten_strength=70)[0]
    reset = sampler.run(**kwargs)[0]
    assert flattened.roles == baseline.roles
    for before, after, again in zip(baseline.values, flattened.values, reset.values):
        assert np.array_equal(np.asarray(before)[:, :, 3], np.asarray(after)[:, :, 3])
        assert before.tobytes() == again.tobytes()
    assert any(a.tobytes() != b.tobytes() for a, b in zip(baseline.values, flattened.values))


@pytest.mark.parametrize("strength", [-1, 101, True, 2.5])
def test_invalid_strength_refused(strength):
    with pytest.raises(ValueError, match="Color Flatten"):
        make_sampler("generation_edit").run(task="text-to-image", conditioning=condition(),
                                            color_flatten_strength=strength, **options())
