"""CPU lookahead preserves native pixels, errors and bounded ownership."""
import threading
from contextlib import closing
import pytest
import torch
from ming_image_01_training_buffer_test import setup
from core.models.ming_image_01 import pixel_prefetch


@pytest.mark.parametrize("profile", ["generation_edit", "layer_decompose"])
def test_prefetched_native_latents_match_serial(tmp_path, monkeypatch, profile):
    _, owner, _, buffer, _, _ = setup(tmp_path, profile=profile)
    owner.bucket_strategy = "random_crop"
    buffer.latent_interval = 3
    buffer._fill_latents(0)
    prefetched = {i: (identity, value.clean.clone(), value.reference_latents)
                  for i, (identity, value) in buffer.latents.items()}
    monkeypatch.setattr(pixel_prefetch, "PIXEL_WORKING_BUDGET_BYTES", 0)
    buffer._fill_latents(0)
    for index, (identity, value) in buffer.latents.items():
        previous_identity, clean, references = prefetched[index]
        assert identity == previous_identity
        torch.testing.assert_close(value.clean, clean, rtol=0, atol=0)
        if references is not None:
            for actual, expected in zip(value.reference_latents, references):
                if expected is None:
                    assert actual is None
                else:
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    buffer.close()
    assert not torch.cuda.is_initialized()


def test_next_pixels_overlap_caller_work_and_keep_original_indices(tmp_path):
    _, _, batches, _, _, _ = setup(tmp_path)
    started, release = threading.Event(), threading.Event()
    calls = []
    def load(index, pairs):
        calls.append((index, threading.current_thread().name))
        if index == 1:
            started.set()
            assert release.wait(5)
        return index
    with closing(pixel_prefetch.prefetch_pixels(batches, 0, 3, load, lambda: None)) as producer:
        assert next(producer) == (0, 0)
        try:
            assert started.wait(5)
            assert [i for i, _ in calls] == [0, 1]
        finally:
            release.set()
        assert list(producer) == [(1, 1), (2, 2)]
    assert calls[1][1].startswith("ming-pixels")


def test_oversized_sources_use_serial_preparation(tmp_path):
    _, _, batches, _, _, _ = setup(tmp_path)
    from dataclasses import replace
    for batch in batches:
        item = batch[0][0]
        item["_ming_record"] = replace(item["_ming_record"], source_size=(30000, 30000))
    calls = []
    assert list(pixel_prefetch.prefetch_pixels(batches, 0, 3,
        lambda index, pairs: calls.append(threading.current_thread().name) or index,
        lambda: None)) == [(0, 0), (1, 1), (2, 2)]
    assert calls == [threading.current_thread().name] * 3


def test_worker_errors_propagate_and_threads_join(tmp_path):
    _, _, batches, _, _, _ = setup(tmp_path)
    def load(index, pairs):
        if index == 1:
            raise ValueError("changed source")
        return index
    with pytest.raises(ValueError, match="changed source"):
        list(pixel_prefetch.prefetch_pixels(batches, 0, 3, load, lambda: None))
    assert not any(t.name.startswith("ming-pixels") for t in threading.enumerate())


def test_caller_failure_joins_outstanding_pixel_job(tmp_path):
    _, _, batches, _, _, _ = setup(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        with closing(pixel_prefetch.prefetch_pixels(batches, 0, 3,
                lambda index, pairs: index, lambda: None)) as producer:
            next(producer)
            raise KeyboardInterrupt()
    assert not any(t.name.startswith("ming-pixels") for t in threading.enumerate())
