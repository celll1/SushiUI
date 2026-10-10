"""Global clipping includes every independently rated optimizer group."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

torch.cuda.get_device_capability = lambda *a, **k: (8, 9)
torch.cuda._lazy_init = lambda *a, **k: None
torch._C._cuda_init = lambda *a, **k: None
from core.training.base_trainer import BaseTrainer
assert not torch.cuda.is_initialized()


class GlobalGradientClipTest(unittest.TestCase):
    def test_two_rates_share_one_norm(self):
        first = torch.nn.Parameter(torch.ones(2))
        second = torch.nn.Parameter(torch.ones(2))
        unused = torch.nn.Parameter(torch.ones(2))
        first.grad = torch.tensor([3., 0.])
        second.grad = torch.tensor([0., 4.])
        optimizer = torch.optim.SGD([
            dict(params=[first], lr=.1),
            dict(params=[second, unused], lr=.01),
        ])
        trainer = SimpleNamespace(optimizer=optimizer)
        norm = BaseTrainer._clip_global_grad_norm(trainer, 1.)
        self.assertAlmostEqual(float(norm), 5.)
        torch.testing.assert_close(first.grad, torch.tensor([.6, 0.]))
        torch.testing.assert_close(second.grad, torch.tensor([0., .8]))
        self.assertIsNone(unused.grad)
        optimizer.step()
        torch.testing.assert_close(first, torch.tensor([.94, 1.]))
        torch.testing.assert_close(second, torch.tensor([1., .992]))

    def test_one_group_matches_torch_and_preserves_small_norm(self):
        for magnitude in (.1, 10.):
            first = torch.nn.Parameter(torch.ones(2))
            reference = torch.nn.Parameter(first.detach().clone())
            first.grad = torch.tensor([magnitude, magnitude])
            reference.grad = first.grad.clone()
            trainer = SimpleNamespace(optimizer=torch.optim.SGD([first], lr=.1))
            actual = BaseTrainer._clip_global_grad_norm(trainer, 1.)
            expected = torch.nn.utils.clip_grad_norm_([reference], 1.)
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(first.grad, reference.grad)

    def test_repeated_parameter_is_counted_once(self):
        parameter = torch.nn.Parameter(torch.ones(1))
        parameter.grad = torch.tensor([2.])
        trainer = SimpleNamespace(optimizer=SimpleNamespace(param_groups=[
            dict(params=[parameter]), dict(params=[parameter])]))
        norm = BaseTrainer._clip_global_grad_norm(trainer, 1.)
        self.assertAlmostEqual(float(norm), 2.)
        torch.testing.assert_close(parameter.grad, torch.ones(1))

    def test_no_gradients(self):
        parameter = torch.nn.Parameter(torch.ones(1))
        trainer = SimpleNamespace(optimizer=torch.optim.SGD([parameter], lr=.1))
        self.assertEqual(float(BaseTrainer._clip_global_grad_norm(trainer, 1.)), 0.)
        self.assertIsNone(parameter.grad)


if __name__ == '__main__':
    unittest.main()
