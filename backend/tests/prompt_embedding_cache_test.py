import torch

from core.inference.prompt_embedding_cache import (
    PromptEmbeddingCache,
    generation_prompt_cache,
)
from core.pipeline_backends.krea2 import Krea2Mixin
from core.pipeline_backends.minit2i import MiniT2IMixin


class _Encoder:
    pass


def test_hit_returns_independent_tensor_tree():
    cache = PromptEmbeddingCache(max_entries=2)
    encoder = _Encoder()
    original = ({"features": torch.tensor([[1.0, 2.0]])}, torch.tensor([True]))

    cache.put(encoder, ("prompt", 32), original)
    first, hit = cache.get(encoder, ("prompt", 32), "cpu")
    assert hit
    first[0]["features"].zero_()

    second, hit = cache.get(encoder, ("prompt", 32), "cpu")
    assert hit
    assert torch.equal(second[0]["features"], original[0]["features"])
    assert second[0]["features"].data_ptr() != original[0]["features"].data_ptr()


def test_encoder_identity_is_part_of_the_key():
    cache = PromptEmbeddingCache()
    first_encoder = _Encoder()
    second_encoder = _Encoder()
    cache.put(first_encoder, "same prompt", torch.tensor([1]))

    value, hit = cache.get(second_encoder, "same prompt", "cpu")
    assert value is None
    assert not hit


def test_lru_bound_evicts_oldest_entry():
    cache = PromptEmbeddingCache(max_entries=2)
    encoder = _Encoder()
    cache.put(encoder, "a", torch.tensor([1]))
    cache.put(encoder, "b", torch.tensor([2]))
    cache.put(encoder, "c", torch.tensor([3]))

    assert not cache.get(encoder, "a", "cpu")[1]
    assert cache.get(encoder, "b", "cpu")[1]
    assert cache.get(encoder, "c", "cpu")[1]


def test_krea2_hit_skips_encoder_stage_and_forward(monkeypatch):
    from core.models.krea2 import krea2_pipeline_ops

    generation_prompt_cache.clear()
    manager = Krea2Mixin()
    manager.krea2_components = {
        "text_encoder": torch.nn.Linear(1, 1),
        "tokenizer": _Encoder(),
        "text_encoder_select_layers": [1, 2],
    }
    moves = []
    calls = []
    manager._krea2_move = lambda name, device: moves.append((name, device))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def encode_prompt(*args):
        calls.append(args[2])
        return torch.ones(1, 2, 2), torch.ones(1, 2, dtype=torch.bool)

    monkeypatch.setattr(krea2_pipeline_ops, "encode_prompt", encode_prompt)
    cfg = {"max_sequence_length": 32, "guidance": 0.0}

    device = torch.device("cpu")
    first = manager._krea2_encode("p", "", cfg, device, torch.float32,
                                  model_key="model")
    second = manager._krea2_encode("p", "", cfg, device, torch.float32,
                                   model_key="model")

    assert calls == ["p"]
    assert moves == [
        ("text_encoder", device),
        ("text_encoder", "cpu"),
        ("text_encoder", "cpu"),
    ]
    assert torch.equal(first[0], second[0])
    assert first[0].data_ptr() != second[0].data_ptr()


def test_minit2i_cache_covers_positive_negative_and_nag(monkeypatch):
    from core.models.minit2i import minit2i_pipeline_ops

    generation_prompt_cache.clear()
    manager = MiniT2IMixin()
    manager.minit2i_components = {
        "text_encoder": torch.nn.Linear(1, 1),
        "tokenizer": _Encoder(),
    }
    moves = []
    calls = []
    manager._minit2i_move = lambda name, device: moves.append((name, device))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        "core.inference.negpip_minit2i.negpip_eligible", lambda *_args: False,
    )

    def encode_prompt(_encoder, _tokenizer, prompt, _length, _device):
        calls.append(prompt)
        return torch.full((1, 2, 2), len(prompt), dtype=torch.float32), torch.ones(1, 2)

    monkeypatch.setattr(minit2i_pipeline_ops, "encode_prompt", encode_prompt)
    kwargs = dict(
        prompt="positive", negative_prompt="negative", nag_negative_prompt="nag",
        prompt_length=32, device=torch.device("cpu"), dtype=torch.float32,
        model_key="model",
    )

    first = manager._minit2i_encode(**kwargs)
    second = manager._minit2i_encode(**kwargs)

    assert calls == ["positive", "negative", "nag"]
    assert moves == [
        ("text_encoder", torch.device("cpu")),
        ("text_encoder", "cpu"),
        ("text_encoder", "cpu"),
    ]
    assert all(torch.equal(a, b) for a, b in zip(first, second))
