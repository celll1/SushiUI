import gc

import torch

from core.inference.prompt_embedding_cache import (
    PromptEmbeddingCache,
    conditioning_cache_key,
    generation_prompt_cache,
)
from core.keep_hot import mark_resident
from core.pipeline_backends.anima import AnimaMixin
from core.pipeline_backends.flux2 import Flux2Mixin
from core.pipeline_backends.ideogram4 import Ideogram4Mixin
from core.pipeline_backends.krea2 import Krea2Mixin
from core.pipeline_backends.lens import LensMixin
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


def test_encoder_collection_releases_its_cpu_entries():
    cache = PromptEmbeddingCache()
    encoder = _Encoder()
    cache.put(encoder, "prompt", torch.tensor([1]))
    assert len(cache) == 1

    del encoder
    gc.collect()

    assert len(cache) == 0


def test_common_key_covers_tokenizer_and_execution_settings():
    tokenizer = _Encoder()
    tokenizer.name_or_path = "tokenizer-a"
    tokenizer.vocab_size = 10
    tokenizer.model_max_length = 32

    base = conditioning_cache_key(
        "arch", "model", tokenizer, "cpu", torch.float32, "prompt", 32,
    )
    assert base != conditioning_cache_key(
        "arch", "model", tokenizer, "cpu", torch.float32, "changed", 32,
    )
    tokenizer.vocab_size = 11
    assert base != conditioning_cache_key(
        "arch", "model", tokenizer, "cpu", torch.float32, "prompt", 32,
    )


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


def test_flux2_cache_owns_all_text_variants_and_truthful_residency(monkeypatch):
    import core.vram_optimization as vram

    generation_prompt_cache.clear()
    manager = Flux2Mixin()
    manager.device = torch.device("cpu")
    encoder = torch.nn.Linear(1, 1)
    manager.flux2_components = {"text_encoder": encoder}
    stages = []
    forwards = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        vram,
        "move_flux2_text_encoder_to_gpu",
        lambda module, *_args, **_kwargs: stages.append(module) or module,
    )

    def encode(_encoder, _tokenizer, prompt, _max_length):
        forwards.append(prompt)
        value = torch.full((1, 2, 2), len(prompt), dtype=torch.float32)
        return value, torch.arange(8).reshape(1, 2, 4)

    manager._flux2_encode_prompt = encode
    tokenizer = _Encoder()
    params = {"nag_enable": True, "nag_scale": 2.0, "nag_negative_prompt": "nag"}
    args = (
        encoder, tokenizer, "positive", "negative", 32, True, params,
        "model", False, None,
    )

    first, kept_first, nag_active, nag_prompt = manager._flux2_encode_conditioning(*args)
    second, kept_second, _, _ = manager._flux2_encode_conditioning(*args)

    assert forwards == ["positive", "negative", "nag"]
    assert stages == [encoder]
    assert not kept_first and not kept_second
    assert nag_active and nag_prompt == "nag"
    for first_value, second_value in zip(first, second):
        assert torch.equal(first_value, second_value)
        assert first_value.data_ptr() != second_value.data_ptr()

    keep_args = args[:-2] + (True, None)
    _, kept_without_residency, _, _ = manager._flux2_encode_conditioning(*keep_args)
    assert not kept_without_residency

    mark_resident(manager, "text_encoder", "model")
    _, kept_with_residency, _, _ = manager._flux2_encode_conditioning(*keep_args)
    assert kept_with_residency


def test_anima_cache_owns_cfg_and_nag_encodes(monkeypatch):
    from core.models.anima import anima_pipeline_ops

    generation_prompt_cache.clear()
    manager = AnimaMixin()
    encoder = torch.nn.Linear(1, 1)
    manager.anima_components = {"text_encoder": encoder}
    moves = []
    forwards = []
    manager._anima_move = lambda name, device, *_args: moves.append((name, device)) or encoder
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr("core.inference.nag_dit.nag_active", lambda *_args: True)

    def encode_prompt(_encoder, _qwen, _t5, prompt, **_kwargs):
        forwards.append(prompt)
        return {"prompt_embeds": torch.full((1, 2, 2), len(prompt), dtype=torch.float32)}

    monkeypatch.setattr(anima_pipeline_ops, "encode_prompt", encode_prompt)
    params = {"nag_enable": True, "nag_scale": 2.0, "nag_negative_prompt": "nag"}
    args = (
        encoder, _Encoder(), _Encoder(), "positive", "negative", 4.0, params,
        torch.device("cpu"), torch.device("cpu"), torch.float32,
        "model", False, False, None,
    )

    first, kept_first = manager._anima_encode_conditioning(*args)
    second, kept_second = manager._anima_encode_conditioning(*args)

    assert forwards == ["positive", "negative", "nag"]
    assert moves == [
        ("text_encoder", torch.device("cpu")),
        ("text_encoder", "cpu"),
        ("text_encoder", "cpu"),
    ]
    assert not kept_first and not kept_second
    for first_value, second_value in zip(first, second):
        assert torch.equal(first_value["prompt_embeds"], second_value["prompt_embeds"])
        assert first_value["prompt_embeds"].data_ptr() != second_value["prompt_embeds"].data_ptr()


def test_lens_hit_skips_disposable_encoder_reload(monkeypatch):
    from core.models.lens import lens_pipeline_ops

    generation_prompt_cache.clear()
    manager = LensMixin()
    manager.model_revision = 3
    manager.lens_components = {"text_encoder": None, "tokenizer": _Encoder()}
    reloads = []
    forwards = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def reload_encoder():
        reloads.append(True)
        manager.lens_components["text_encoder"] = torch.nn.Linear(1, 1)

    manager._reload_lens_text_encoder = reload_encoder
    manager._lens_move = lambda name, _device, *_args: manager.lens_components[name]

    def encode_prompt(*_args, **_kwargs):
        forwards.append(True)
        return [torch.ones(1, 2, 2)], torch.ones(1, 2, dtype=torch.bool)

    monkeypatch.setattr(lens_pipeline_ops, "encode_prompt", encode_prompt)
    args = (
        {}, "positive", "negative", torch.device("cpu"), torch.device("cpu"),
        torch.float32, 32, "model", False, None,
    )
    first = manager._lens_encode_conditioning(*args)
    second = manager._lens_encode_conditioning(*args)

    assert len(reloads) == 1
    assert len(forwards) == 1
    assert manager.lens_components["text_encoder"] is None
    assert torch.equal(first[0][0], second[0][0])
    assert first[0][0].data_ptr() != second[0][0].data_ptr()

    manager.model_revision += 1
    manager._lens_encode_conditioning(*args)
    assert len(reloads) == 2
    assert len(forwards) == 2


def test_ideogram4_cache_owns_main_and_nag_encodes(monkeypatch):
    from core.models.ideogram4 import ideogram4_pipeline_ops

    generation_prompt_cache.clear()
    manager = Ideogram4Mixin()
    encoder = torch.nn.Linear(1, 1)
    manager.ideogram4_components = {
        "text_encoder": encoder,
        "tokenizer": _Encoder(),
    }
    moves = []
    forwards = []
    manager._ideogram4_move = lambda name, device: moves.append((name, device))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def encode_prompt(_encoder, _tokenizer, prompt, **_kwargs):
        forwards.append(prompt)
        value = torch.full((1, 2, 2), len(prompt), dtype=torch.float32)
        return {"llm_features": value, "neg_llm_features": -value}

    monkeypatch.setattr(ideogram4_pipeline_ops, "encode_prompt", encode_prompt)
    params = {
        "prompt": "positive",
        "negative_prompt": "negative",
        "nag_enable": True,
        "nag_scale": 2.0,
        "nag_negative_prompt": "nag",
    }
    cfg = {
        "prompt": "positive",
        "grid_h": 8,
        "grid_w": 8,
        "max_sequence_length": 32,
    }
    args = (params, cfg, torch.device("cpu"), torch.float32, "model", False)

    (first, first_nag, kept_first) = manager._ideogram4_encode_conditioning(*args)
    (second, second_nag, kept_second) = manager._ideogram4_encode_conditioning(*args)

    assert forwards == ["positive", "nag"]
    assert moves == [
        ("text_encoder", torch.device("cpu")),
        ("text_encoder", "cpu"),
        ("text_encoder", "cpu"),
    ]
    assert not kept_first and not kept_second
    assert first_nag == second_nag == {"nag_scale": 2.0, "nag_tau": 2.5, "nag_alpha": 0.25}
    for name in ("llm_features", "neg_llm_features", "nag_llm_features"):
        assert torch.equal(first[name], second[name])
        assert first[name].data_ptr() != second[name].data_ptr()

    keep_args = args[:-1] + (True,)
    assert not manager._ideogram4_encode_conditioning(*keep_args)[2]
    mark_resident(manager, "text_encoder", "model")
    assert manager._ideogram4_encode_conditioning(*keep_args)[2]
