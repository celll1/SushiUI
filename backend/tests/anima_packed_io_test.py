"""Numerical boundary checks for the experimental reduced-grid Anima IO."""
import unittest
import torch
from torch.nn import functional as F
from core.experiments.anima_packed_io import packed_io_weights, packed_head_noise


class PackedIOTest(unittest.TestCase):
    def test_native_embedding_average_and_covariance(self):
        torch.manual_seed(63)
        c, d = 16, 48
        x = torch.randn(2, c, 16, 24, dtype=torch.float64)
        mask = torch.randn(2, 1, 8, 12, dtype=torch.float64).repeat_interleave(2, -2).repeat_interleave(2, -1)
        iw = torch.randn(d, (c + 1) * 4, dtype=torch.float64)
        ow = torch.randn(c * 4, d, dtype=torch.float64)
        incoming, _ = packed_io_weights(iw, ow, c)
        native = F.conv2d(F.pixel_unshuffle(torch.cat((x, mask), 1), 2), iw[:, :, None, None])
        packed = F.pixel_unshuffle(x, 2)
        target = F.conv2d(F.pixel_unshuffle(torch.cat((packed, mask[:, :, ::2, ::2]), 1), 2), incoming[:, :, None, None])
        torch.testing.assert_close(target, F.avg_pool2d(native, 2), atol=1e-12, rtol=1e-12)
        source_cov = iw[:, :64] @ iw[:, :64].T
        target_cov = incoming[:, :256] @ incoming[:, :256].T
        torch.testing.assert_close(4 * target_cov, source_cov, atol=1e-12, rtol=1e-12)

    def test_output_head_and_noise_complement(self):
        torch.manual_seed(64)
        iw = torch.randn(48, 68, dtype=torch.float64)
        ow = torch.randn(64, 48, dtype=torch.float64)
        _, outgoing = packed_io_weights(iw, ow, 16)
        h = torch.randn(2, 48, 4, 6, dtype=torch.float64)
        original = F.conv2d(h, ow[:, :, None, None]).reshape(2, 4, 16, 4, 6).permute(0, 2, 1, 3, 4)
        expanded = F.conv2d(h, outgoing[:, :, None, None]).reshape(2, 4, 64, 4, 6).permute(0, 2, 1, 3, 4)
        restored = F.pixel_shuffle(F.pixel_shuffle(expanded.reshape(2, 256, 4, 6), 2), 2)
        expected = F.pixel_shuffle(original.reshape(2, 64, 4, 6).repeat_interleave(2, -2).repeat_interleave(2, -1), 2)
        torch.testing.assert_close(restored, expected, atol=0, rtol=0)
        noise = torch.randn(2, 64, 16, 24, dtype=torch.float64)
        predicted = packed_head_noise(2 * noise)
        corrected = predicted + noise - packed_head_noise(2 * noise)
        wrong = predicted + noise - 2 * noise
        torch.testing.assert_close(corrected, noise, atol=1e-12, rtol=1e-12)
        self.assertGreater(float((wrong - noise).square().mean()), 1)

    def test_invalid_geometry(self):
        with self.assertRaises(ValueError):
            packed_head_noise(torch.zeros(1, 64, 15, 24))
        with self.assertRaises(ValueError):
            packed_io_weights(torch.zeros(48, 64), torch.zeros(64, 48), 16)


if __name__ == "__main__":
    unittest.main()
