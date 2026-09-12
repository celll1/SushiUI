import torch

from backend.core.tagger.tag_labels import build_label_and_mask
from backend.core.tagger.tag_vocabulary import TagVocabulary


def _vocabulary() -> TagVocabulary:
    vocabulary = TagVocabulary()
    vocabulary.add_tags(
        [
            "subject",
            "general",
            "sensitive",
            "masterpiece",
            "best_quality",
            "low_quality",
            "worst_quality",
        ]
    )
    return vocabulary


def test_rating_and_quality_masks_match_training_contract():
    vocabulary = _vocabulary()
    label, mask = build_label_and_mask(
        ["subject", "general", "best_quality"], vocabulary, "intra_group"
    )

    assert label[vocabulary.tag_to_idx["subject"]] == 1
    assert label[vocabulary.tag_to_idx["general"]] == 1
    assert label[vocabulary.tag_to_idx["best quality"]] == 1
    assert mask[vocabulary.tag_to_idx["sensitive"]] == 1
    assert mask[vocabulary.tag_to_idx["masterpiece"]] == 0
    assert mask[vocabulary.tag_to_idx["low quality"]] == 1


def test_missing_special_annotations_mask_all_special_siblings():
    vocabulary = _vocabulary()
    _, mask = build_label_and_mask(["subject"], vocabulary)

    for idx in vocabulary.rating_indices:
        assert mask[idx] == 0
    for indices in vocabulary.quality_indices.values():
        for idx in indices:
            assert mask[idx] == 0


def test_cross_group_mode_keeps_quality_negatives_enabled():
    vocabulary = _vocabulary()
    _, mask = build_label_and_mask(["best_quality"], vocabulary, "cross_group")

    for indices in vocabulary.quality_indices.values():
        for idx in indices:
            assert mask[idx] == 1


def test_aliases_unknown_tags_and_vocabulary_growth_are_supported():
    vocabulary = _vocabulary()

    class Resolver:
        def resolve(self, tag):
            return {"old subject": "subject"}.get(tag, tag)

    before, _ = build_label_and_mask(
        ["old subject", "not present"], vocabulary, alias_resolver=Resolver()
    )
    assert before.shape == (7,)
    assert before[vocabulary.tag_to_idx["subject"]] == 1

    vocabulary.add_tags(["not_present"])
    after, _ = build_label_and_mask(["not_present"], vocabulary)
    assert after.shape == (8,)
    assert torch.equal(after, torch.nn.functional.one_hot(
        torch.tensor(vocabulary.tag_to_idx["not present"]), vocabulary.num_tags
    ).float())
