"""Batch-assembly text collation must pad the ARCH's sequence axis.

The shared collation used to read ``emb.shape[1]`` as the sequence length. That
is right for ``[1, L, D]`` but wrong for Lens and Ideogram 4, whose per-item
tensor is ``[1, num_layers, L, D]`` -- axis 1 is the layer stack, so every item
in a batch reported the SAME length, the padding branch was skipped, and
``torch.cat`` raised as soon as two captions tokenised to different L. Neither
arch pads its tokenisation to a fixed length (Lens tokenises with
``padding=True`` on a single prompt, Ideogram 4 does not pad at all), so mixed
L is the normal case, not a corner case.

Krea 2 is the third 4-D producer, with the OTHER layout (``[1, L, layers, D]``,
sequence at axis 1). Its length read was already right, but the old padding
branch built a 3-D zero block and would have failed on a ragged batch too.

Synthetic tensors only -- the property under test is shape/mask bookkeeping.

Run (from backend/, which is what puts ``core`` on sys.path):
    ..\\venv\\Scripts\\python.exe -m pytest \\
        tests/text_embedding_collation_axis_test.py -q
"""

import inspect
import unittest

import torch

from core.training.arch import ARCH_REGISTRY
from core.training.base_trainer import BaseTrainer


collate_embeddings = BaseTrainer._collate_text_embeddings
collate_masks = BaseTrainer._collate_text_masks


def _layered_item(num_layers, length, dim, fill):
    """Lens / Ideogram 4 layout: [1, num_layers, L, D]."""
    return torch.full((1, num_layers, length, dim), float(fill))


def _krea2_item(length, num_layers, dim, fill):
    """Krea 2 layout: [1, L, num_layers, D]."""
    return torch.full((1, length, num_layers, dim), float(fill))


def _plain_item(length, dim, fill):
    """[1, L, D] -- Z-Image, MiniT2I, SDXL, FLUX.2, MiniMax-H3, ..."""
    return torch.full((1, length, dim), float(fill))


def _mask(active_len, dtype=torch.bool):
    return torch.ones(active_len, dtype=dtype)


class TheArchDeclaresItsSequenceAxis(unittest.TestCase):
    def test_lens_and_ideogram4_declare_axis_2(self):
        self.assertEqual(ARCH_REGISTRY["lens"].text_seq_axis, 2)
        self.assertEqual(ARCH_REGISTRY["ideogram4"].text_seq_axis, 2)

    def test_every_other_arch_keeps_axis_1(self):
        for name, handler in ARCH_REGISTRY.items():
            if name in ("lens", "ideogram4"):
                continue
            self.assertEqual(
                handler.text_seq_axis, 1,
                f"{name} declares text_seq_axis={handler.text_seq_axis}",
            )


class TheLayeredLayoutPadsTheSequenceAxis(unittest.TestCase):
    """Lens / Ideogram 4: [1, num_layers, L, D] -> [B, num_layers, Lmax, D]."""

    def test_lens_mixed_lengths_collate(self):
        items = [_layered_item(4, 7, 8, 1), _layered_item(4, 11, 8, 2),
                 _layered_item(4, 3, 8, 3)]
        batched, seq_len = collate_embeddings(items, seq_axis=2)
        self.assertEqual(seq_len, 11)
        self.assertEqual(tuple(batched.shape), (3, 4, 11, 8))

    def test_ideogram4_mixed_lengths_collate(self):
        items = [_layered_item(13, 19, 16, 1), _layered_item(13, 24, 16, 2)]
        batched, seq_len = collate_embeddings(items, seq_axis=2)
        self.assertEqual(seq_len, 24)
        self.assertEqual(tuple(batched.shape), (2, 13, 24, 16))

    def test_the_layer_and_feature_dims_are_untouched(self):
        items = [_layered_item(4, 7, 8, 1), _layered_item(4, 11, 8, 2)]
        batched, _ = collate_embeddings(items, seq_axis=2)
        self.assertEqual(batched.shape[1], 4)
        self.assertEqual(batched.shape[3], 8)

    def test_real_rows_survive_and_padding_is_zero(self):
        items = [_layered_item(4, 7, 8, 1), _layered_item(4, 11, 8, 2)]
        batched, _ = collate_embeddings(items, seq_axis=2)
        self.assertTrue(torch.equal(batched[0, :, :7], torch.full((4, 7, 8), 1.0)))
        self.assertTrue(torch.all(batched[0, :, 7:] == 0))
        self.assertTrue(torch.equal(batched[1], torch.full((4, 11, 8), 2.0)))

    def test_reading_axis_1_would_have_raised(self):
        """The defect, reproduced: axis 1 is num_layers, so no padding happens."""
        items = [_layered_item(4, 7, 8, 1), _layered_item(4, 11, 8, 2)]
        self.assertEqual(items[0].shape[1], items[1].shape[1])  # both report 4
        with self.assertRaises(RuntimeError):
            torch.cat(items, dim=0)


class TheKrea2LayoutPadsAxisOne(unittest.TestCase):
    """[1, L, num_layers, D]: the length read was right, the pad block was not."""

    def test_mixed_lengths_collate(self):
        items = [_krea2_item(5, 12, 8, 1), _krea2_item(9, 12, 8, 2)]
        batched, seq_len = collate_embeddings(items, seq_axis=1)
        self.assertEqual(seq_len, 9)
        self.assertEqual(tuple(batched.shape), (2, 9, 12, 8))
        self.assertTrue(torch.all(batched[0, 5:] == 0))
        self.assertTrue(torch.all(batched[0, :5] == 1))


class ThePlainLayoutStillWorks(unittest.TestCase):
    """Z-Image / MiniT2I / SDXL / FLUX.2 / MiniMax-H3: [1, L, D]."""

    def test_mixed_lengths_collate(self):
        items = [_plain_item(6, 32, 1), _plain_item(10, 32, 2)]
        batched, seq_len = collate_embeddings(items, seq_axis=1)
        self.assertEqual(seq_len, 10)
        self.assertEqual(tuple(batched.shape), (2, 10, 32))
        self.assertTrue(torch.all(batched[0, 6:] == 0))

    def test_equal_lengths_are_a_plain_cat(self):
        items = [_plain_item(6, 32, 1), _plain_item(6, 32, 2)]
        batched, seq_len = collate_embeddings(items, seq_axis=1)
        self.assertEqual(seq_len, 6)
        self.assertTrue(torch.equal(batched, torch.cat(items, dim=0)))


class TheMaskIsPaddedWithTheFeatures(unittest.TestCase):
    def test_mask_reaches_the_feature_length(self):
        items = [_layered_item(4, 7, 8, 1), _layered_item(4, 11, 8, 2)]
        _, seq_len = collate_embeddings(items, seq_axis=2)
        mask = collate_masks([_mask(7), _mask(11)], seq_len)
        self.assertEqual(tuple(mask.shape), (2, seq_len))

    def test_padded_positions_are_inactive(self):
        mask = collate_masks([_mask(7), _mask(11)], 11)
        self.assertTrue(torch.all(mask[0, :7]))
        self.assertFalse(torch.any(mask[0, 7:]))
        self.assertTrue(torch.all(mask[1]))

    def test_dtype_is_preserved_for_long_masks(self):
        mask = collate_masks([_mask(3, torch.long), _mask(5, torch.long)], 5)
        self.assertEqual(mask.dtype, torch.long)
        self.assertEqual(mask[0].tolist(), [1, 1, 1, 0, 0])

    def test_a_longer_feature_length_still_wins(self):
        """Features padded past the longest mask must not leave the mask short."""
        mask = collate_masks([_mask(3), _mask(4)], target_len=9)
        self.assertEqual(tuple(mask.shape), (2, 9))
        self.assertFalse(torch.any(mask[:, 4:]))

    def test_none_entries_are_dropped_as_before(self):
        mask = collate_masks([_mask(4), None, _mask(4)], 4)
        self.assertEqual(tuple(mask.shape), (2, 4))


class ASingleItemBatchIsUnchanged(unittest.TestCase):
    def test_layered_single_item(self):
        item = _layered_item(4, 7, 8, 1)
        batched, seq_len = collate_embeddings([item], seq_axis=2)
        self.assertEqual(seq_len, 7)
        self.assertEqual(tuple(batched.shape), (1, 4, 7, 8))
        self.assertTrue(torch.equal(batched, item))

    def test_plain_single_item(self):
        item = _plain_item(6, 32, 1)
        batched, seq_len = collate_embeddings([item], seq_axis=1)
        self.assertEqual(seq_len, 6)
        self.assertTrue(torch.equal(batched, item))

    def test_single_mask(self):
        mask = collate_masks([_mask(6)], 6)
        self.assertEqual(tuple(mask.shape), (1, 6))
        self.assertTrue(torch.all(mask))


class BothCopiesOfTheCollationAreFixed(unittest.TestCase):
    """The MNT re-encode path has its own copy; fixing one only is worse.

    Source-level, because `BaseTrainer.train` cannot be driven without a model.
    """

    def setUp(self):
        self.src = inspect.getsource(BaseTrainer.train)

    def test_both_sites_call_the_shared_helper(self):
        self.assertEqual(self.src.count("self._collate_text_embeddings("), 2)

    def test_both_sites_pass_the_arch_declared_axis(self):
        self.assertEqual(self.src.count("self.arch.text_seq_axis"), 2)

    def test_no_hardcoded_sequence_axis_read_remains(self):
        self.assertNotIn("emb.shape[1] for emb in", self.src)

    def test_the_tensor_mask_stacks_go_through_the_padding_helper(self):
        self.assertNotIn(
            "torch.stack([aux for aux in auxiliary_data_list if aux is not None]",
            self.src,
        )
        self.assertNotIn(
            "torch.stack([aux for aux in mnt_auxiliary_data_list if aux is not None]",
            self.src,
        )
        self.assertEqual(self.src.count("self._collate_text_masks("), 3)


if __name__ == "__main__":
    unittest.main()
