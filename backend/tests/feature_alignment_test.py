import unittest

import torch

from core.experiments.feature_alignment import teacher_grid, normalized_feature_mse, feature_metrics


class FeatureAlignmentTest(unittest.TestCase):
    def test_pooling_matches_cell_oracle(self):
        teacher = torch.arange(48.).reshape(1, 1, 4, 6, 2)
        student = torch.zeros(1, 1, 2, 3, 2)
        expected = torch.stack([teacher[0, 0, i:i+2, j:j+2].mean((0, 1))
                                for i in (0, 2) for j in (0, 2, 4)]).reshape_as(student)
        torch.testing.assert_close(teacher_grid(teacher, student), expected)

    def test_identity_and_zero_loss(self):
        teacher = torch.ones(2, 1, 3, 4, 5)
        torch.testing.assert_close(teacher_grid(teacher, teacher), teacher)
        self.assertEqual(float(normalized_feature_mse(teacher, teacher)), 0.)
        self.assertEqual(feature_metrics(teacher, teacher)['norm_ratio'], 1.)

    def test_relative_loss_is_scale_invariant(self):
        teacher = torch.ones(1, 1, 2, 2, 3)
        student = teacher * 2
        self.assertEqual(float(normalized_feature_mse(student, teacher)), 1.)
        self.assertEqual(float(normalized_feature_mse(student * 5, teacher * 5)), 1.)

    def test_only_student_receives_gradients(self):
        teacher = torch.ones(1, 1, 4, 4, 3, requires_grad=True)
        student = torch.zeros(1, 1, 2, 2, 3, requires_grad=True)
        normalized_feature_mse(student, teacher).backward()
        self.assertIsNone(teacher.grad)
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertGreater(float(student.grad.abs().sum()), 0.)

    def test_invalid_shapes_are_refused(self):
        teacher = torch.ones(1, 1, 4, 4, 3)
        for shape in ((1, 2, 2, 2, 3), (1, 1, 3, 2, 3), (1, 1, 8, 8, 3),
                      (1, 1, 2, 2, 4), (2, 1, 2, 2, 3), (1, 1, 0, 2, 3), (1, 2, 3)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                teacher_grid(teacher, torch.zeros(shape))

    def test_nonfinite_is_refused_and_zero_is_finite(self):
        zeros = torch.zeros(1, 1, 2, 2, 3)
        self.assertEqual(float(normalized_feature_mse(zeros, zeros)), 0.)
        with self.assertRaises(ValueError):
            teacher_grid(zeros, zeros + float('nan'))


if __name__ == '__main__':
    unittest.main()
