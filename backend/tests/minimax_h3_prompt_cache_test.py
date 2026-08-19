"""The bounded LRU cache for MiniMax-H3's post-projection prompt embedding.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_h3_prompt_cache_test.py -v

WHY THIS FILE EXISTS
--------------------
`core.models.minimax_h3.prompt_cache` is what lets a repeated, IDENTICAL
prompt against the SAME loaded MiniMax-H3 model skip the ~21s Qwen3-VL-32B
streamed encode (a same-prompt sweep, or an ordinary "regenerate with a
different seed"). Its correctness rests entirely on the cache key
((text_encoder_path, te_projection_path, prompt, text_dim)) and the LRU
bound, neither of which needs a model load to exercise -- `get_or_encode_prompt`
takes a plain callable in place of the real `ops.encode_prompt` +
`_minimax_h3_project_prompt_embeds` pair, so this file stubs that callable and
asserts the cache's own bookkeeping, isolated from the model.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402

from core.models.minimax_h3 import prompt_cache  # noqa: E402

TEXT_DIM = 5120


def setup_function(_fn):
    # Every test starts from an empty cache: the module-level `_cache` is a
    # process-wide singleton, matching `generation_timer`'s own justification
    # (one generation at a time behind the GPU coordinator) -- but a test
    # process runs many "generations" back to back, so each test must not see
    # another test's entries.
    prompt_cache.clear()


def _counting_encoder(tokens: int = 4, value: float = 1.0):
    """A stub `encode_fn`: returns a fresh (embeds, num_tokens) pair and counts calls."""
    calls = {"count": 0}

    def _encode():
        calls["count"] += 1
        embeds = torch.full((1, tokens, 8), value, dtype=torch.float32)
        return embeds, tokens

    return _encode, calls


def test_same_key_hits_without_calling_the_encoder_again():
    encode_fn, calls = _counting_encoder()

    (embeds1, tokens1), hit1 = prompt_cache.get_or_encode_prompt(
        "/models/te.safetensors", None, "a cat on a rooftop", TEXT_DIM, encode_fn)
    (embeds2, tokens2), hit2 = prompt_cache.get_or_encode_prompt(
        "/models/te.safetensors", None, "a cat on a rooftop", TEXT_DIM, encode_fn)

    assert hit1 is False
    assert hit2 is True
    assert calls["count"] == 1
    assert tokens1 == tokens2 == 4
    assert torch.equal(embeds1, embeds2)


def test_different_prompt_misses_independently():
    encode_fn, calls = _counting_encoder()

    prompt_cache.get_or_encode_prompt(
        "/models/te.safetensors", None, "prompt A", TEXT_DIM, encode_fn)
    (_embeds, _tokens), hit = prompt_cache.get_or_encode_prompt(
        "/models/te.safetensors", None, "prompt B", TEXT_DIM, encode_fn)

    assert hit is False
    assert calls["count"] == 2


def test_different_text_encoder_path_misses_independently():
    encode_fn, calls = _counting_encoder()

    prompt_cache.get_or_encode_prompt(
        "/models/te_a.safetensors", None, "same prompt", TEXT_DIM, encode_fn)
    (_embeds, _tokens), hit = prompt_cache.get_or_encode_prompt(
        "/models/te_b.safetensors", None, "same prompt", TEXT_DIM, encode_fn)

    assert hit is False
    assert calls["count"] == 2


def test_different_te_projection_path_misses_independently():
    encode_fn, calls = _counting_encoder()

    prompt_cache.get_or_encode_prompt(
        "/models/te.safetensors", "/models/proj_a.safetensors", "same prompt", TEXT_DIM, encode_fn)
    (_embeds, _tokens), hit = prompt_cache.get_or_encode_prompt(
        "/models/te.safetensors", "/models/proj_b.safetensors", "same prompt", TEXT_DIM, encode_fn)

    assert hit is False
    assert calls["count"] == 2


def test_none_projection_is_a_distinct_key_from_a_named_projection():
    """A released encoder (no substitution) must not collide with a
    substituted one that happens to hash/repr similarly."""
    encode_fn, calls = _counting_encoder()

    prompt_cache.get_or_encode_prompt(
        "/models/te.safetensors", None, "same prompt", TEXT_DIM, encode_fn)
    (_embeds, _tokens), hit = prompt_cache.get_or_encode_prompt(
        "/models/te.safetensors", "/models/proj.safetensors", "same prompt", TEXT_DIM, encode_fn)

    assert hit is False
    assert calls["count"] == 2


def test_different_text_dim_misses_independently():
    """A DiT-only reload keeps the encoder/projection identity but can change
    `transformer_config["text_dim"]` -- the cache must not hand the OLD DiT's
    conditioning width to the NEW one (see `prompt_cache._cache_key`)."""
    encode_fn, calls = _counting_encoder()

    prompt_cache.get_or_encode_prompt(
        "/models/te.safetensors", None, "same prompt", TEXT_DIM, encode_fn)
    (_embeds, _tokens), hit = prompt_cache.get_or_encode_prompt(
        "/models/te.safetensors", None, "same prompt", TEXT_DIM + 1, encode_fn)

    assert hit is False
    assert calls["count"] == 2


def test_lru_eviction_at_the_cap():
    encode_fn, calls = _counting_encoder()
    cap = prompt_cache._MAX_ENTRIES

    for i in range(cap + 2):
        prompt_cache.get_or_encode_prompt(
            "/models/te.safetensors", None, f"prompt {i}", TEXT_DIM, encode_fn)

    assert prompt_cache.cache_size() == cap
    assert calls["count"] == cap + 2

    # The two oldest ("prompt 0", "prompt 1") were evicted -- a lookup for
    # either must re-encode (a miss), not resurrect a dropped entry.
    (_embeds, _tokens), hit_oldest = prompt_cache.get_or_encode_prompt(
        "/models/te.safetensors", None, "prompt 0", TEXT_DIM, encode_fn)
    assert hit_oldest is False
    assert calls["count"] == cap + 3

    # The most recent one before this eviction round is still resident.
    (_embeds, _tokens), hit_recent = prompt_cache.get_or_encode_prompt(
        "/models/te.safetensors", None, f"prompt {cap + 1}", TEXT_DIM, encode_fn)
    assert hit_recent is True
    assert calls["count"] == cap + 3


def test_a_hit_moves_the_entry_to_most_recently_used_and_protects_it_from_eviction():
    encode_fn, calls = _counting_encoder()
    cap = prompt_cache._MAX_ENTRIES

    # Fill to the cap.
    for i in range(cap):
        prompt_cache.get_or_encode_prompt(
            "/models/te.safetensors", None, f"prompt {i}", TEXT_DIM, encode_fn)
    assert prompt_cache.cache_size() == cap

    # Touch the OLDEST entry (a hit) so it becomes the MOST recently used.
    (_embeds, _tokens), hit = prompt_cache.get_or_encode_prompt(
        "/models/te.safetensors", None, "prompt 0", TEXT_DIM, encode_fn)
    assert hit is True
    touched_after = calls["count"]

    # One more distinct prompt pushes the cache over the cap by one; the
    # LEAST recently used entry now is "prompt 1" (never touched again),
    # not "prompt 0" (just touched) -- so "prompt 0" must survive.
    prompt_cache.get_or_encode_prompt(
        "/models/te.safetensors", None, "prompt new", TEXT_DIM, encode_fn)

    (_embeds, _tokens), hit_touched = prompt_cache.get_or_encode_prompt(
        "/models/te.safetensors", None, "prompt 0", TEXT_DIM, encode_fn)
    assert hit_touched is True
    assert calls["count"] == touched_after + 1  # +1 for "prompt new" only

    (_embeds, _tokens), hit_untouched = prompt_cache.get_or_encode_prompt(
        "/models/te.safetensors", None, "prompt 1", TEXT_DIM, encode_fn)
    assert hit_untouched is False  # evicted


def test_clear_forces_every_subsequent_lookup_to_re_encode():
    encode_fn, calls = _counting_encoder()

    prompt_cache.get_or_encode_prompt(
        "/models/te.safetensors", None, "a prompt", TEXT_DIM, encode_fn)
    prompt_cache.clear()
    (_embeds, _tokens), hit = prompt_cache.get_or_encode_prompt(
        "/models/te.safetensors", None, "a prompt", TEXT_DIM, encode_fn)

    assert hit is False
    assert calls["count"] == 2
    assert prompt_cache.cache_size() == 1


def test_returned_tensor_is_not_aliased_with_the_cached_or_a_prior_callers_copy():
    """Mutating a tensor handed back on one lookup must not corrupt the value
    a later lookup (same key) returns -- the cache clones on both store and
    hit precisely so no caller can accidentally write through to it."""
    encode_fn, _calls = _counting_encoder(tokens=2, value=3.0)

    (embeds_first, _tokens), _hit1 = prompt_cache.get_or_encode_prompt(
        "/models/te.safetensors", None, "a prompt", TEXT_DIM, encode_fn)
    embeds_first.fill_(999.0)  # mutate the caller's copy in place

    (embeds_second, _tokens), hit2 = prompt_cache.get_or_encode_prompt(
        "/models/te.safetensors", None, "a prompt", TEXT_DIM, encode_fn)

    assert hit2 is True
    assert torch.equal(embeds_second, torch.full((1, 2, 8), 3.0))
    assert not torch.equal(embeds_second, embeds_first)
