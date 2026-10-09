"""Independent coordinate checks for a reduced-grid RoPE prototype."""
import unittest
import torch
from core.experiments.anima_source_span import source_span_angles
from core.models.anima.anima_models import VideoRopePosition3DEmb


class SourceSpanTest(unittest.TestCase):
    def setUp(self):
        self.pos = VideoRopePosition3DEmb(head_dim=48, len_h=64, len_w=64, len_t=1,
                                         h_extrapolation_ratio=4, w_extrapolation_ratio=4)

    def test_identity_and_unchanged_buffers(self):
        before = {k: v.clone() for k, v in self.pos.state_dict().items()}
        shape = (2, 1, 16, 24, 96)
        actual = source_span_angles(self.pos.generate_embeddings, shape, (16, 24))
        torch.testing.assert_close(actual, self.pos.generate_embeddings(shape), atol=0, rtol=0)
        for k, v in self.pos.state_dict().items():
            torch.testing.assert_close(v, before[k], atol=0, rtol=0)

    def test_cell_centers_and_relative_phase(self):
        shape = (1, 1, 8, 12, 96)
        actual = source_span_angles(self.pos.generate_embeddings, shape, (32, 24)).reshape(8, 12, 48)
        hf = 1 / ((10000 * self.pos.h_ntk_factor) ** self.pos.dim_spatial_range)
        wf = 1 / ((10000 * self.pos.w_ntk_factor) ** self.pos.dim_spatial_range)
        expected_h = (4 * torch.arange(8) + 1.5)[:, None] * hf[None]
        expected_w = (2 * torch.arange(12) + .5)[:, None] * wf[None]
        torch.testing.assert_close(actual[:, 0, 8:16], expected_h, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(actual[0, :, 16:24], expected_w, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(actual[:, :, :8], torch.zeros_like(actual[:, :, :8]), atol=0, rtol=0)
        torch.testing.assert_close(actual[:, :, :24], actual[:, :, 24:], atol=0, rtol=0)

    def test_invalid_geometry(self):
        for shape, source in (((1, 2, 8, 8, 96), (32, 32)),
                              ((1, 1, 7, 8, 96), (32, 32)),
                              ((1, 1, 8, 8, 96), (0, 32))):
            with self.assertRaises(ValueError):
                source_span_angles(self.pos.generate_embeddings, shape, source)


if __name__ == "__main__":
    unittest.main()
