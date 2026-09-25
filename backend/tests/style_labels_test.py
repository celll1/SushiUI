"""Style label category precedence (design §5, test T9)."""

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.style_encoder.style_labels import StyleLabelResolver
from core.tagger.comma_tag_resolver import CommaTagResolver
from core.tagger.tag_alias_resolver import TagAliasResolver


class _Manager:
    def __init__(self, groups):
        self.groups = groups

    def get_tag_group(self, tag):
        return self.groups.get(tag)


def _resolver(groups, aliases=None, comma=None, signature=("signature",), non_copyright=("original",)):
    return StyleLabelResolver(_Manager(groups), comma, TagAliasResolver(aliases or {}), signature, non_copyright)


def test_explicit_artist_unknown_to_taglist_is_kept():
    resolver = _resolver({})
    labels = resolver.resolve(json.dumps([{"tag": "Brand_New_Artist", "category": "ARTIST"}]), None)
    assert labels.artists == {"brand new artist"}
    assert resolver.stats.artist_tags_explicit == 1
    assert resolver.stats.artist_tags_manager == 0


def test_missing_or_placeholder_category_goes_to_manager_not_general():
    resolver = _resolver({"known artist": "Artist", "hero": "Character", "series": "Copyright"})
    labels = resolver.resolve([
        {"tag": "known_artist"},
        {"tag": "hero", "category": "Unknown"},
        {"tag": "series", "category": "__lookup__"},
        {"tag": "mystery tag"},
    ], None)
    assert labels.artists == {"known artist"}
    assert labels.characters == {"hero"}
    assert labels.copyrights == {"series"}
    assert resolver.stats.artist_tags_manager == 1
    assert resolver.stats.unresolved_tags == 1
    assert resolver.stats.category_disagreements == 0


def test_disagreement_keeps_explicit_category_and_is_counted():
    resolver = _resolver({"ambiguous": "Artist", "other": "General"})
    labels = resolver.resolve([
        {"tag": "ambiguous", "category": "character"},
        {"tag": "other", "category": "General"},
    ], None)
    assert labels.artists == frozenset()
    assert labels.characters == {"ambiguous"}
    assert resolver.stats.category_disagreements == 1


def test_explicit_general_yields_to_a_specific_manager_category():
    resolver = _resolver({"backfilled artist": "Artist", "really artist": "General"})
    labels = resolver.resolve([
        {"tag": "backfilled artist", "category": "General"},
        {"tag": "plain tag", "category": "General"},
        {"tag": "really artist", "category": "Artist"},
    ], None)
    assert labels.artists == {"backfilled artist", "really artist"}
    assert resolver.stats.general_overridden_by_manager == 1
    assert resolver.stats.category_disagreements == 1
    assert resolver.stats.artist_tags_manager == 1
    assert resolver.stats.artist_tags_explicit == 1
    assert resolver.stats.unresolved_tags == 0


def test_deprecated_names_resolve_through_their_canonical_form():
    resolver = _resolver({"new name": "Artist", "new hero": "Character", "canon two": "Artist"},
                         aliases={"old_name": "new_name", "old_hero": "new_hero", "old_two": "canon_two"})
    labels = resolver.resolve(None, "old_name, old hero, new name")
    assert labels.artists == {"new name"}
    assert labels.characters == {"new hero"}
    assert resolver.stats.artist_tags_manager == 1
    assert resolver.stats.unresolved_tags == 0
    backfilled = resolver.resolve([{"tag": "old_two", "category": "General"}], None)
    assert backfilled.artists == {"canon two"}
    assert resolver.stats.general_overridden_by_manager == 1


def test_lookup_falls_back_to_the_pre_alias_name():
    resolver = _resolver({"old name": "Artist"}, aliases={"old_name": "new_name"})
    assert resolver.resolve(None, "old_name").artists == {"new name"}
    assert resolver.stats.unresolved_tags == 0


def test_comma_merged_tokens_take_the_comma_category_without_aliasing():
    comma = CommaTagResolver.build_from_category_map({"godzilla, mothra": "Copyright"})
    resolver = _resolver({"godzilla": "Character", "someone": "Artist", "godzilla mothra": "General"},
                         aliases={"godzilla_mothra": "wrong"}, comma=comma)
    labels = resolver.resolve([
        {"tag": "godzilla", "category": "Character"},
        {"tag": "mothra", "category": "General"},
        {"tag": "someone", "category": "Artist"},
        {"tag": "godzilla", "category": "General"},
    ], None)
    assert labels.copyrights == {"godzilla mothra"}
    assert labels.characters == {"godzilla"}
    assert labels.artists == {"someone"}
    assert resolver.stats.category_disagreements == 0


def test_signature_flag():
    resolver = _resolver({}, signature=("signature", "artist name"))
    assert resolver.resolve(None, "Artist_Name").has_signature
    assert not resolver.resolve(None, "sign").has_signature


def test_item_artist_counts():
    resolver = _resolver({"a": "Artist", "b": "Artist"})
    resolver.resolve(None, "a, b")
    resolver.resolve(None, "a")
    resolver.resolve(None, "unrelated")
    assert resolver.stats.items == 3
    assert resolver.stats.items_with_multiple_artists == 1
    assert resolver.stats.items_without_artist == 1


def test_non_copyright_tags_are_dropped_after_canonicalisation():
    resolver = _resolver({"original": "Copyright", "series": "Copyright", "someone": "Artist"},
                         aliases={"original_work": "original"})
    labels = resolver.resolve([{"tag": "Original"}, {"tag": "series"}, {"tag": "someone"}], None)
    assert labels.copyrights == {"series"}
    assert resolver.resolve(None, "original_work, someone").copyrights == frozenset()
    assert resolver.stats.non_copyright_tags_dropped == 2
    kept = _resolver({"original": "Copyright"}, non_copyright=())
    assert kept.resolve(None, "original").copyrights == {"original"}


def test_explicit_category_from_any_caption_wins_in_either_order():
    tagged = ([{"tag": "Brand_New", "category": "Artist"}], None)
    plain = (None, "brand_new, other")
    for captions in ((plain, tagged), (tagged, plain)):
        resolver = _resolver({"brand new": "General"})
        assert resolver.resolve_captions(captions).artists == {"brand new"}
        assert resolver.stats.artist_tags_explicit == 1
    same_caption = [{"tag": "brand_new"}, {"tag": "Brand_New", "category": "Artist"}]
    assert _resolver({}).resolve(same_caption, None).artists == {"brand new"}
