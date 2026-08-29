"""The batch-assembly text encode is dead work when the MNT loop will re-encode.

With a trainable text encoder at multi_noise_timesteps > 1, every MNT iteration
rebuilds the conditioning from scratch (each backward frees the graph), so the
encode done during batch assembly is never read -- it is built, held alive for
the whole window, and discarded. These tests pin the bypass: the encode and the
collation that consumes it are skipped together, index alignment survives the
latent-size filter, and SenseNova is untouched.

Source-level assertions, not a live trainer: BaseTrainer.train() cannot be
driven without a model, and the property under test is which branch runs.

Run:
    venv\\Scripts\\python.exe -m pytest \\
        backend/tests/mnt_dead_text_encode_test.py -q
"""

import inspect
import re
import unittest

from core.training.base_trainer import BaseTrainer


def _train_source() -> str:
    return inspect.getsource(BaseTrainer.train)


def _batch_prep_source() -> str:
    """train() from the per-batch preparation block onwards.

    The mode names ("onthefly_gpu") and arch checks ("if self.is_sensenova:")
    also appear in train()'s earlier setup/validation blocks, so every branch
    assertion below must be anchored past them.
    """
    src = _train_source()
    return src[src.index("_te_recompute_per_mnt = ("):]


class ThePredicateIsDefinedOncePerBatch(unittest.TestCase):
    def setUp(self):
        self.src = _train_source()

    def test_the_flag_carries_all_three_conditions(self):
        m = re.search(
            r"_te_recompute_per_mnt = \((.*?)\)\n", self.src, re.S
        )
        self.assertIsNotNone(m, "_te_recompute_per_mnt is not defined in train()")
        body = m.group(1)
        for term in (
            "text_encoder_trainable",
            "multi_noise_timesteps > 1",
            'text_encoding_mode == "onthefly_gpu"',
        ):
            self.assertIn(term, body, f"predicate is missing {term!r}")

    def test_sensenova_is_excluded_from_the_bypass(self):
        m = re.search(r"_te_recompute_per_mnt = \((.*?)\)\n", self.src, re.S)
        self.assertIn("not self.is_sensenova", m.group(1))

    def test_it_is_defined_before_the_per_item_loop_that_reads_it(self):
        define = self.src.index("_te_recompute_per_mnt = (")
        first_read = self.src.index("if _te_recompute_per_mnt:")
        self.assertLess(define, first_read)

    def test_the_predicate_agrees_with_need_recompute_text_embeddings(self):
        """The MNT loop's own predicate must not drift from the assembly one.

        They differ only by the SenseNova exclusion, which is unreachable from
        the branch that reads need_recompute_text_embeddings.
        """
        m = re.search(
            r"need_recompute_text_embeddings = \((.*?)\)\n", self.src, re.S
        )
        self.assertIsNotNone(m)
        body = m.group(1)
        for term in (
            "text_encoder_trainable",
            "multi_noise_timesteps > 1",
            'text_encoding_mode == "onthefly_gpu"',
        ):
            self.assertIn(term, body)


class TheOnTheFlyEncodeIsSkipped(unittest.TestCase):
    def setUp(self):
        self.src = _batch_prep_source()

    def test_the_onthefly_branch_is_guarded(self):
        i = self.src.index('elif text_encoding_mode == "onthefly_gpu":')
        window = self.src[i : i + 1200]
        self.assertIn("if _te_recompute_per_mnt:", window)
        guard = window.index("if _te_recompute_per_mnt:")
        encode = window.index("self.encode_caption(")
        self.assertLess(
            guard, encode, "the encode must sit under the guard, not before it"
        )

    def test_it_appends_placeholders_rather_than_nothing(self):
        i = self.src.index('elif text_encoding_mode == "onthefly_gpu":')
        window = self.src[i : i + 1200]
        guarded = window[window.index("if _te_recompute_per_mnt:") :]
        guarded = guarded[: guarded.index("else:")]
        self.assertIn("text_embeddings_list.append(None)", guarded)
        self.assertIn("auxiliary_data_list.append(None)", guarded)


class TheCollationIsBypassedToo(unittest.TestCase):
    """Placeholders must never reach the collation that dereferences them."""

    def setUp(self):
        self.src = _train_source()

    def test_the_embedding_collation_is_bypassed_first(self):
        bypass = self.src.index("if _te_recompute_per_mnt:\n                        # Assembly encode was skipped")
        collate = self.src.index("elif text_embeddings_list:")
        self.assertLess(bypass, collate)

    def test_the_aux_collation_is_bypassed_first(self):
        chain = self.src.index("attention_mask = None\n                    pooled_embeddings = None")
        window = self.src[chain : chain + 900]
        bypass = window.index("if _te_recompute_per_mnt:")
        first_arch = window.index("elif self.is_zimage")
        self.assertLess(bypass, first_arch)

    def test_no_unguarded_dereference_of_the_lists_remains(self):
        """Every `.shape`/`cat`/`stack` over the assembly lists is downstream of a bypass."""
        for probe in (
            "seq_lengths = [emb.shape[1] for emb in text_embeddings_list]",
            "torch.cat(text_embeddings_list, dim=0)",
        ):
            at = self.src.index(probe)
            before = self.src[:at]
            self.assertIn("if _te_recompute_per_mnt:", before)


class ThePlaceholdersKeepTheListsIndexAligned(unittest.TestCase):
    """The latent-size filter reindexes these lists positionally.

    Appending nothing (instead of None) would make that filter raise IndexError
    on any batch that drops an item -- rare, but a hard run-killer.
    """

    def test_the_filter_survives_placeholder_entries(self):
        latents_list = ["a", "b", "c"]
        text_embeddings_list = [None, None, None]
        auxiliary_data_list = [None, None, None]
        valid_indices = [0, 2]

        latents_list = [latents_list[i] for i in valid_indices]
        text_embeddings_list = [text_embeddings_list[i] for i in valid_indices]
        auxiliary_data_list = [auxiliary_data_list[i] for i in valid_indices]

        self.assertEqual(latents_list, ["a", "c"])
        self.assertEqual(len(text_embeddings_list), len(latents_list))
        self.assertEqual(len(auxiliary_data_list), len(latents_list))

    def test_an_empty_list_would_have_raised(self):
        with self.assertRaises(IndexError):
            _ = [[][i] for i in [0, 2]]


class SenseNovaIsUnaffected(unittest.TestCase):
    def test_its_prefix_branch_still_encodes_during_assembly(self):
        src = _batch_prep_source()
        i = src.index("if self.is_sensenova:")
        window = src[i : i + 1400]
        self.assertIn("self.encode_caption(", window)
        self.assertNotIn("_te_recompute_per_mnt", window)


if __name__ == "__main__":
    unittest.main()
