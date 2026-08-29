"""The latent-size filter must reindex `batch` along with the item lists.

A batch whose items land in different latent buckets is filtered down to the
first item's spatial shape. That filter reindexes latents_list and its parallel
lists -- but `batch` itself feeds several PER-ITEM derivations that happen
after it: batch_captions, the LTX-2.3 per-clip fps tensor, the MiniMax-H3
per-clip audio latent, reference paths, and the VE reconstruction mask. If
`batch` keeps its pre-filter length, those pair item k's value with item k+1's
latent, or hand the train step a tensor one row too long.

Source-level assertions plus a positional simulation: BaseTrainer.train()
cannot be driven without a model, and the property under test is alignment.

Run:
    "d:\\celll1\\webui_cl\\venv\\Scripts\\python.exe" -m pytest \\
        backend/tests/batch_filter_alignment_test.py -q
"""

import inspect
import re
import unittest

from core.training.base_trainer import BaseTrainer


def _filter_block() -> str:
    src = inspect.getsource(BaseTrainer.train)
    start = src.index("if len(valid_indices) < len(latents_list):")
    return src[start : start + 2000]


class TheFilterReindexesBatchItself(unittest.TestCase):
    def test_batch_is_filtered_in_the_same_block(self):
        self.assertIn("batch = [batch[i] for i in valid_indices]", _filter_block())

    def test_it_uses_the_same_valid_indices_as_the_item_lists(self):
        block = _filter_block()
        reindexed = set(re.findall(r"(\w+) = \[\w+\[i\] for i in valid_indices\]", block))
        for name in ("latents_list", "text_embeddings_list", "auxiliary_data_list", "batch"):
            self.assertIn(name, reindexed, f"{name} is not reindexed by valid_indices")


class EveryPerItemDerivationOfBatchIsDownstreamOfTheFilter(unittest.TestCase):
    """Guards against a future per-item read of `batch` moving above the filter."""

    CONSUMERS = (
        "batch_captions = [item.get(\"caption\", \"\") for item, dataset in batch]",
        "self._ltx2_batch_fps_tensor(batch)",
        "self._minimax_h3_batch_audio(batch)",
        "ref_paths = [_item.get(\"reference_images\", [None])[0] for _item, _ in batch]",
    )

    def test_each_consumer_runs_after_batch_is_reindexed(self):
        src = inspect.getsource(BaseTrainer.train)
        filtered_at = src.index("batch = [batch[i] for i in valid_indices]")
        for consumer in self.CONSUMERS:
            with self.subTest(consumer=consumer):
                self.assertLess(
                    filtered_at,
                    src.index(consumer),
                    "this reads `batch` per item before the filter reindexes it",
                )


class ThePositionalPairingSurvivesAFilteredBatch(unittest.TestCase):
    def test_captions_still_line_up_with_latents(self):
        batch = [("cap0", "ds"), ("cap1", "ds"), ("cap2", "ds")]
        latents_list = ["lat0", "lat1", "lat2"]
        valid_indices = [0, 2]  # item 1 landed in a different bucket

        latents_list = [latents_list[i] for i in valid_indices]
        batch = [batch[i] for i in valid_indices]

        captions = [item for item, _ds in batch]
        self.assertEqual(len(captions), len(latents_list))
        self.assertEqual(captions, ["cap0", "cap2"])
        self.assertEqual(latents_list, ["lat0", "lat2"])

    def test_the_unfiltered_pairing_is_wrong(self):
        """Negative control: what the code did before this fix."""
        batch = [("cap0", "ds"), ("cap1", "ds"), ("cap2", "ds")]
        latents_list = ["lat0", "lat1", "lat2"]
        valid_indices = [0, 2]

        latents_list = [latents_list[i] for i in valid_indices]
        # batch left untouched -- the bug

        captions = [item for item, _ds in batch]
        self.assertNotEqual(len(captions), len(latents_list))
        self.assertEqual(list(zip(captions, latents_list)), [("cap0", "lat0"), ("cap1", "lat2")])


class TheFilterOnlyRunsWhenItemsAreActuallyDropped(unittest.TestCase):
    def test_the_rebind_is_inside_the_shrink_guard(self):
        src = inspect.getsource(BaseTrainer.train)
        guard = src.index("if len(valid_indices) < len(latents_list):")
        rebind = src.index("batch = [batch[i] for i in valid_indices]")
        nxt = src.index("# Skip batch if no valid latents remain")
        self.assertLess(guard, rebind)
        self.assertLess(rebind, nxt)


if __name__ == "__main__":
    unittest.main()
