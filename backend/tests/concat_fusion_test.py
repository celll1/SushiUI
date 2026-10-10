import unittest

import torch

from core.experiments.concat_fusion import CompressedConcatCorrection, parameter_count


class ConcatFusionTest(unittest.TestCase):
    def test_zero_output_preserves_existing_stream(self):
        module = CompressedConcatCorrection(8, 3)
        current = torch.randn(2, 8, 3, 5)
        skip = torch.randn(2, 3, 3, 5)
        torch.testing.assert_close(module(current, skip), torch.zeros_like(current), rtol=0, atol=0)
        torch.testing.assert_close(current + module(current, skip), current, rtol=0, atol=0)

    def test_learning_reaches_both_inputs_and_inner_weights(self):
        torch.manual_seed(18)
        module = CompressedConcatCorrection(8, 3)
        current = torch.randn(1, 8, 3, 5, requires_grad=True)
        skip = torch.randn(1, 3, 3, 5, requires_grad=True)
        module(current, skip).sum().backward()
        self.assertGreater(float(module.output.weight.grad.abs().sum()), 0.)
        self.assertEqual(float(module.mix.weight.grad.abs().sum()), 0.)
        with torch.no_grad():module.output.weight.add_(module.output.weight.grad, alpha=-.01)
        module.zero_grad(set_to_none=True)
        current.grad = skip.grad = None
        module(current, skip).square().mean().backward()
        for gradient in (module.mix.weight.grad, current.grad, skip.grad):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0.)

    def test_projection_count_and_independent_spatial_cells(self):
        module = CompressedConcatCorrection(8, 3)
        self.assertEqual(sum(p.numel() for p in module.parameters()), parameter_count(8, 3))
        with torch.no_grad():module.output.weight.fill_(.1)
        current, skip = torch.randn(1, 8, 4, 6), torch.randn(1, 3, 4, 6)
        full = module(current, skip)
        tiles = torch.cat([module(current[..., :3], skip[..., :3]),
                           module(current[..., 3:], skip[..., 3:])], dim=-1)
        torch.testing.assert_close(full, tiles)

    def test_invalid_contracts_are_refused(self):
        for width, rank in ((True, 1), (8, False), (8, 0), (8, 9)):
            with self.assertRaises(ValueError):CompressedConcatCorrection(width, rank)
        module = CompressedConcatCorrection(8, 3)
        for shape in ((2, 3, 2, 2), (1, 4, 2, 2), (1, 3, 3, 2), (1, 3, 2)):
            with self.assertRaises(ValueError):module(torch.zeros(1, 8, 2, 2), torch.zeros(shape))

    def test_zero_current_has_finite_gradients(self):
        module = CompressedConcatCorrection(8, 1)
        with torch.no_grad():module.output.weight.fill_(.1)
        current = torch.zeros(1, 8, 2, 2, requires_grad=True)
        skip = torch.ones(1, 1, 2, 2, requires_grad=True)
        module(current, skip).sum().backward()
        self.assertTrue(torch.isfinite(current.grad).all())
        self.assertTrue(torch.isfinite(skip.grad).all())

    def test_rank_one_skip_is_not_erased_by_normalization(self):
        module = CompressedConcatCorrection(8, 1)
        with torch.no_grad():
            module.mix.weight.zero_()
            module.mix.weight[:, -1].fill_(1.)
            module.output.weight.fill_(1.)
        current = torch.ones(1, 8, 2, 2)
        positive = module(current, torch.ones(1, 1, 2, 2))
        negative = module(current, -torch.ones(1, 1, 2, 2))
        self.assertGreater(float((positive-negative).detach().abs().sum()), 0.)


if __name__ == '__main__':
    unittest.main()
