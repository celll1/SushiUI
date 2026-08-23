"""`frequency_scale_vector`'s two RoPE pair layouts.

The suppression curve damps high-frequency reference-Key content before KV
injection. Which head-dim slots share one frequency depends on how the target
arch applies RoPE, and getting that wrong silently scales the wrong
frequencies rather than crashing -- SenseNova shipped with suppression
disabled outright for exactly that reason.

CPU-only and model-free. The interleaved golden vector is pinned because that
path is shared by 7 architectures whose output must not move; the rotate-half
pairing assertion is per-axis-chunk on purpose, since a cat over the whole
head_dim would pair a t-axis frequency with an h-axis one and still produce a
correctly-shaped vector.
"""

from __future__ import annotations

import os
import sys
import unittest

import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_BACKEND = os.path.join(_REPO, "backend")
for _p in (_REPO, _BACKEND):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.inference.reference_style import (  # noqa: E402
    StyleTransferConfig,
    frequency_scale_vector,
)

# Krea2's real axes_dims_rope, one of the 7 archs on the interleaved default.
KREA2_AXES = (32, 48, 48)
KREA2_HEAD_DIM = 128
# SenseNova: Qwen3Attention splits head_dim into t=d/2, h=d/4, w=d/4.
SENSENOVA_AXES = (64, 32, 32)
SENSENOVA_HEAD_DIM = 128

_CURVE_KW = dict(
    high_scale=1.04, low_scale=1.0, beta=2.5,
    device=torch.device("cpu"), dtype=torch.float32,
)


def _reference_interleaved(head_dim, axes_dims, high_scale, low_scale, beta,
                           device, dtype):
    """The pre-`rope_layout` construction, transcribed verbatim."""
    curves = []
    for dim in axes_dims:
        half = dim // 2
        if half == 0:
            continue
        x = torch.linspace(0.0, 1.0, half, device=device, dtype=torch.float32)
        curve = high_scale + (low_scale - high_scale) * x.pow(beta)
        curve = curve.repeat_interleave(2)
        curves.append(curve)
    return torch.cat(curves, dim=0).to(dtype)


class InterleavedDefaultUnchanged(unittest.TestCase):
    """The default must stay byte-identical: Krea2/SD1.5/SDXL/FLUX.2/LTX-2.3/
    MiniT2I/Ideogram4/Lens all ride it and none of them was re-gated."""

    def test_matches_the_pre_rope_layout_construction(self):
        for high, low in ((1.04, 1.0), (0.5, 1.1), (0.0, 1.10)):
            with self.subTest(high=high, low=low):
                kw = dict(_CURVE_KW, high_scale=high, low_scale=low)
                got = frequency_scale_vector(KREA2_HEAD_DIM, KREA2_AXES, **kw)
                want = _reference_interleaved(KREA2_HEAD_DIM, KREA2_AXES, **kw)
                self.assertTrue(torch.equal(got, want))

    def test_default_argument_is_interleaved(self):
        explicit = frequency_scale_vector(
            KREA2_HEAD_DIM, KREA2_AXES, rope_layout="interleaved", **_CURVE_KW)
        implicit = frequency_scale_vector(KREA2_HEAD_DIM, KREA2_AXES, **_CURVE_KW)
        self.assertTrue(torch.equal(explicit, implicit))
        self.assertEqual(StyleTransferConfig().rope_layout, "interleaved")

    def test_adjacent_slots_share_a_frequency(self):
        vec = frequency_scale_vector(KREA2_HEAD_DIM, KREA2_AXES, **_CURVE_KW)
        self.assertTrue(torch.equal(vec[0::2], vec[1::2]))


class RotateHalfPairing(unittest.TestCase):
    def test_each_axis_chunk_pairs_j_with_j_plus_half(self):
        vec = frequency_scale_vector(
            SENSENOVA_HEAD_DIM, SENSENOVA_AXES,
            rope_layout="rotate_half", **_CURVE_KW)
        self.assertEqual(vec.shape, (SENSENOVA_HEAD_DIM,))
        offset = 0
        for dim in SENSENOVA_AXES:
            chunk = vec[offset:offset + dim]
            with self.subTest(axis_offset=offset, dim=dim):
                self.assertTrue(torch.equal(chunk[: dim // 2], chunk[dim // 2:]))
            offset += dim
        self.assertEqual(offset, SENSENOVA_HEAD_DIM)

    def test_differs_from_interleaved_on_the_same_axes(self):
        """Guards against a "simplify these two branches" edit collapsing them."""
        rotate = frequency_scale_vector(
            SENSENOVA_HEAD_DIM, SENSENOVA_AXES,
            rope_layout="rotate_half", **_CURVE_KW)
        inter = frequency_scale_vector(
            SENSENOVA_HEAD_DIM, SENSENOVA_AXES,
            rope_layout="interleaved", **_CURVE_KW)
        self.assertFalse(torch.equal(rotate, inter))

    def test_a_whole_vector_cat_would_not_satisfy_the_pairing(self):
        """The trap the per-axis loop exists to avoid: concatenating once over
        the full head_dim pairs a t-axis frequency with an h-axis one."""
        half = frequency_scale_vector(
            SENSENOVA_HEAD_DIM // 2, (32, 16, 16),
            rope_layout="interleaved", **_CURVE_KW)
        whole = torch.cat([half, half])
        offset = 0
        mismatched = False
        for dim in SENSENOVA_AXES:
            chunk = whole[offset:offset + dim]
            if not torch.equal(chunk[: dim // 2], chunk[dim // 2:]):
                mismatched = True
            offset += dim
        self.assertTrue(mismatched)


class Validation(unittest.TestCase):
    def test_unknown_layout_raises(self):
        for bad in ("rotate-half", "neox", "", None):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    frequency_scale_vector(
                        KREA2_HEAD_DIM, KREA2_AXES, rope_layout=bad, **_CURVE_KW)

    def test_axes_must_sum_to_head_dim(self):
        with self.assertRaises(ValueError):
            frequency_scale_vector(KREA2_HEAD_DIM, (32, 32, 32), **_CURVE_KW)


class ConfigThreading(unittest.TestCase):
    """`get_freq_scale_vector` must forward `rope_layout`, not drop it -- the
    per-config cache key is the easy place to lose it."""

    def test_config_forwards_rope_layout(self):
        common = dict(axes_dims=SENSENOVA_AXES, high_scale_start=1.04,
                      high_scale_end=0.0, low_scale_start=1.0, low_scale_end=1.10)
        rotate = StyleTransferConfig(rope_layout="rotate_half", **common)
        inter = StyleTransferConfig(**common)
        dev, dt = torch.device("cpu"), torch.float32
        for progress in (0.0, 0.5, 1.0):
            with self.subTest(progress=progress):
                r = rotate.get_freq_scale_vector(SENSENOVA_HEAD_DIM, progress, dev, dt)
                i = inter.get_freq_scale_vector(SENSENOVA_HEAD_DIM, progress, dev, dt)
                offset = 0
                for dim in SENSENOVA_AXES:
                    chunk = r[offset:offset + dim]
                    self.assertTrue(torch.equal(chunk[: dim // 2], chunk[dim // 2:]))
                    offset += dim
                if progress > 0.0:
                    self.assertFalse(torch.equal(r, i))


if __name__ == "__main__":
    unittest.main()
