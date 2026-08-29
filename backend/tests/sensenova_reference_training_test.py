"""Phase 3-1/3-2: SenseNova reference-conditioned training prefix.

The reference path is exercised against the REAL vendor ``get_thw_indexes`` /
``_build_it2i_inputs`` and the REAL ``_splice_reference_image_tokens`` /
``_embed_reference_images``; only the tokenizer, the ViT and the prefix forward
are faked, so the t-extent arithmetic under test is the model's, not the test's.
"""

import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.models.sensenova.vendor.modeling_neo_chat import NEOChatModel
from core.training.ops.sensenova_ops import encode_prompt

_IMG_START = "<img>"
_IMG_END = "</img>"
_IMG_CONTEXT = "<IMG_CONTEXT>"
_SPECIAL = {_IMG_START: 100, _IMG_END: 101, _IMG_CONTEXT: 102}
_EMBED_DIM = 4
_REF_GRID = (4, 4)  # ViT patches per reference; /2 merge -> 4 context tokens.


class _CacheLayer:
    def __init__(self):
        self.keys = torch.ones(1, 1, 2, 1)
        self.values = torch.ones(1, 1, 2, 1)
        self.flash_k_cache = None
        self.flash_v_cache = None


class _Cache:
    def __init__(self):
        self.layers = [_CacheLayer()]
        self._kv_cache_streamer = None
        self._kv_cache_streamer_branch = None


class _Tokenizer:
    """Special tokens keep their id; every other character is one id."""

    pattern = re.compile("|".join(re.escape(tok) for tok in _SPECIAL))

    def __call__(self, query, return_tensors=None):
        ids, cursor = [], 0
        for match in self.pattern.finditer(query):
            ids.extend([1] * (match.start() - cursor))
            ids.append(_SPECIAL[match.group(0)])
            cursor = match.end()
        ids.extend([1] * (len(query) - cursor))
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    def convert_tokens_to_ids(self, token):
        return _SPECIAL[token]


class _Transformer:
    patch_size = 16
    downsample_ratio = 0.5
    device = torch.device("cpu")
    img_start_token_id = _SPECIAL[_IMG_START]
    img_context_token_id = None

    # Vendor code under test, bound verbatim.
    get_thw_indexes = NEOChatModel.get_thw_indexes
    _build_it2i_inputs = NEOChatModel._build_it2i_inputs

    def __init__(self):
        self.cache = _Cache()
        self.embedding = nn.Embedding(200, _EMBED_DIM)
        self.language_model = SimpleNamespace(
            model=SimpleNamespace(layers=[object()]),
            get_input_embeddings=lambda: self.embedding,
        )
        self.calls = []

    def _build_t2i_query(self, text, system_message=None, append_text=None):
        self.calls.append("query")
        return text + (append_text or "")

    def _build_t2i_text_inputs(self, tokenizer, query):
        self.calls.append("t2i_inputs")
        input_ids = tokenizer(query)["input_ids"]
        t_idx = torch.arange(input_ids.shape[1], dtype=torch.long)
        zeros = torch.zeros_like(t_idx)
        return input_ids, torch.stack([t_idx, zeros, zeros], dim=0), {}

    def _t2i_prefix_forward(self, input_ids, indexes, attention_mask):
        self.calls.append("t2i_prefix_forward")
        return self.cache, torch.zeros(1, input_ids.shape[1], _EMBED_DIM)

    def _it2i_prefix_forward(self, input_embeds, indexes, attention_mask):
        self.calls.append("it2i_prefix_forward")
        return self.cache, torch.zeros(1, input_embeds.shape[1], _EMBED_DIM)

    def extract_feature(self, pixel_values, grid_hw=None):
        self.calls.append("extract_feature")
        merged = int(1 / self.downsample_ratio) ** 2
        count = int((grid_hw[:, 0] * grid_hw[:, 1]).sum()) // merged
        return torch.zeros(count, _EMBED_DIM)


def _make_trainer():
    transformer = _Transformer()
    return SimpleNamespace(transformer=transformer, tokenizer=_Tokenizer()), transformer


def _write_images(tmp_path, count):
    from PIL import Image

    paths = []
    for index in range(count):
        path = tmp_path / f"ref{index}.png"
        Image.new("RGB", (64, 64), (index * 10, 0, 0)).save(path)
        paths.append(str(path))
    return paths


def _fake_load_image_native(image, patch_size, downsample_ratio, **kwargs):
    grid_h, grid_w = _REF_GRID
    return (
        torch.zeros(grid_h * grid_w, 8),
        torch.tensor([[grid_h, grid_w]], dtype=torch.long),
    )


def _encode_with_references(trainer, prompt, ref_paths, loader=None):
    target = "core.models.sensenova.sensenova_pipeline_ops"
    with patch(f"{target}.load_image_native", side_effect=loader or _fake_load_image_native) as load, patch(
        f"{target}.raise_if_cancelled"
    ):
        prefix = encode_prompt(trainer, prompt, reference_image_paths=ref_paths)
    return prefix, load


# --- 3-2: prefix construction ------------------------------------------------


@pytest.mark.parametrize(
    "prompt,count",
    [
        ("cat", 1),  # implicit single placeholder
        ("cat", 2),  # implicit "Image-N:<image>" prefixes
        ("a <image> cat", 1),  # explicit placeholder in the caption
    ],
)
def test_reference_prefix_builds_for_every_placeholder_form(tmp_path, prompt, count):
    from core.training.ops.sensenova_ops import _assert_immutable_prefix_cache

    trainer, transformer = _make_trainer()
    paths = _write_images(tmp_path, count)

    prefix, load = _encode_with_references(trainer, prompt, paths)

    assert load.call_count == count
    assert transformer.calls.count("extract_feature") == 1
    assert "it2i_prefix_forward" in transformer.calls
    assert "t2i_prefix_forward" not in transformer.calls
    # Missing this leaves the vendor `assert selected.sum() != 0` to fire.
    assert transformer.img_context_token_id == _SPECIAL[_IMG_CONTEXT]
    _assert_immutable_prefix_cache(prefix.cache, 1)


def test_reference_prefix_text_length_is_the_t_extent_not_the_token_count(tmp_path):
    trainer, transformer = _make_trainer()
    captured = {}
    original = NEOChatModel.get_thw_indexes

    def spy(self, input_ids, grid_hw=None):
        indexes = original(self, input_ids, grid_hw)
        captured["indexes"] = indexes
        captured["tokens"] = int(input_ids.shape[0])
        return indexes

    with patch.object(_Transformer, "get_thw_indexes", spy):
        prefix, _ = _encode_with_references(
            trainer, "cat", _write_images(tmp_path, 1)
        )

    indexes = captured["indexes"]
    assert prefix.text_length == int(indexes[0].max()) + 1
    # The whole point of the generalization: image patches share one t index, so
    # the t extent is strictly shorter than the prefix's token count.
    assert prefix.text_length < captured["tokens"]


def test_text_only_text_length_still_equals_the_input_id_count():
    trainer, transformer = _make_trainer()

    prefix = encode_prompt(trainer, "a caption")

    ids = trainer.tokenizer("a caption<think>\n\n</think>\n\n<img>")["input_ids"]
    assert prefix.text_length == int(ids.shape[1])
    assert transformer.calls == ["query", "t2i_inputs", "t2i_prefix_forward"]
    assert transformer.img_context_token_id is None


@pytest.mark.parametrize("paths", [None, [], [None]])
def test_empty_reference_lists_take_the_unchanged_text_only_path(paths):
    trainer, transformer = _make_trainer()

    encode_prompt(trainer, "a caption", reference_image_paths=paths)

    assert transformer.calls == ["query", "t2i_inputs", "t2i_prefix_forward"]


def test_reference_count_reuses_the_inference_cap(tmp_path):
    from core.pipeline_backends.sensenova import SENSENOVA_MAX_REFERENCE_IMAGES

    trainer, _ = _make_trainer()
    paths = _write_images(tmp_path, SENSENOVA_MAX_REFERENCE_IMAGES + 1)
    with pytest.raises(ValueError, match=str(SENSENOVA_MAX_REFERENCE_IMAGES)):
        _encode_with_references(trainer, "cat", paths)


def test_reference_pixels_never_reach_the_trainer_image_pipeline(tmp_path):
    """The trainer bucket-resizes and normalizes to [-1,1]; the understanding
    tower wants a per-ref smart-resize with ImageNet stats, and the two are
    shape-compatible, so only structure can catch a mix-up."""

    def _boom(*args, **kwargs):
        raise AssertionError("reference reached the trainer image pipeline")

    trainer, _ = _make_trainer()
    trainer.encode_image = _boom
    trainer.vae_encode = _boom
    seen = {}

    def loader(image, patch_size, downsample_ratio, **kwargs):
        seen["normalization"] = kwargs
        seen["patch_size"] = patch_size
        return _fake_load_image_native(image, patch_size, downsample_ratio)

    _encode_with_references(trainer, "cat", _write_images(tmp_path, 1), loader=loader)

    # Reached the understanding tower's own loader, with ITS cap, not a bucket.
    from core.models.sensenova.sensenova_pipeline_ops import (
        REFERENCE_IMAGE_MAX_PIXELS_CAP,
    )

    assert seen["patch_size"] == _Transformer.patch_size
    assert seen["normalization"]["max_pixels"] <= REFERENCE_IMAGE_MAX_PIXELS_CAP
    assert seen["normalization"]["upscale"] is False


def test_reference_encode_still_drives_the_phase_evictor(tmp_path):
    calls = []
    trainer, _ = _make_trainer()
    trainer.sensenova_phase_evictor = SimpleNamespace(
        enter_prefix=lambda: calls.append("prefix"),
        enter_denoise=lambda: calls.append("denoise"),
        assert_generation_resident=lambda: calls.append("resident"),
    )

    _encode_with_references(trainer, "cat", _write_images(tmp_path, 1))

    assert calls == ["prefix", "denoise", "resident"]


# --- 3-3: reference-conditioned sampling during training ---------------------


class _SampleTransformer(nn.Module):
    """Only the state generate_sample touches: training flag + eval()/train()."""

    def forward(self):  # pragma: no cover - never called
        raise AssertionError


def _sample_trainer(with_evictor=False):
    transformer = _SampleTransformer()
    transformer.train()
    trainer = SimpleNamespace(
        transformer=transformer,
        tokenizer=_Tokenizer(),
        log_prefix="[test]",
        attention_backend="auto",
        _resolve_training_backend=lambda backend: "sdpa",
        move_main_model_to_gpu=lambda: None,
    )
    events = []
    if with_evictor:
        trainer.sensenova_phase_evictor = SimpleNamespace(
            enter_prefix=lambda: events.append("prefix"),
            enter_denoise=lambda: events.append("denoise"),
            assert_generation_resident=lambda: events.append("resident"),
        )
    return trainer, transformer, events


class _SampleRecorder:
    """Stands in for the inference ops generate_sample drives."""

    def __init__(self, denoise_error=None):
        self.encode_calls = []
        self.denoise_calls = []
        self.modes = []
        self.cleared = []
        self.denoise_error = denoise_error
        self.prefix = SimpleNamespace(name="prefix")

    def encode_prompt(self, *args, **kwargs):
        self.encode_calls.append((args, kwargs))
        return self.prefix

    def denoise_loop(self, *args, **kwargs):
        self.denoise_calls.append((args, kwargs))
        if self.denoise_error is not None:
            raise self.denoise_error
        return torch.zeros(1, 3, 32, 32)

    def set_attention_backend(self, transformer, backend, mode):
        self.modes.append(mode)
        return 1

    def clear_prefix_caches(self, prefix):
        self.cleared.append(prefix)


def _run_sample(recorder, trainer, **overrides):
    from PIL import Image

    from core.training.ops.sensenova_ops import generate_sample

    target = "core.models.sensenova.sensenova_pipeline_ops"
    kwargs = dict(
        prompt="a cat",
        height=512,
        width=512,
        num_inference_steps=4,
        guidance_scale=4.0,
        seed=7,
    )
    kwargs.update(overrides)
    with patch(f"{target}.encode_prompt", recorder.encode_prompt), patch(
        f"{target}.denoise_loop", recorder.denoise_loop
    ), patch(f"{target}.set_attention_backend", recorder.set_attention_backend), patch(
        f"{target}.clear_prefix_caches", recorder.clear_prefix_caches
    ), patch(
        f"{target}.tensor_to_image", lambda tensor: Image.new("RGB", (8, 8))
    ):
        return generate_sample(trainer, **kwargs)


def test_sample_without_references_keeps_the_text_only_inference_call():
    from api.param_defaults import SENSENOVA_GENERATION_DEFAULTS

    trainer, transformer, _ = _sample_trainer()
    recorder = _SampleRecorder()

    image = _run_sample(recorder, trainer)

    assert image is not None
    (args, kwargs) = recorder.encode_calls[0]
    assert args == (transformer, trainer.tokenizer, "a cat", 512, 512, 4.0)
    assert kwargs["negative_prompt"] == ""
    # Both equal encode_prompt's own defaults, so the text-only branch inside it
    # ("if not ref_images") is entered exactly as before.
    assert kwargs["ref_images"] == []
    assert kwargs["img_cfg_scale"] == SENSENOVA_GENERATION_DEFAULTS["img_cfg_scale"]


def test_sample_with_a_reference_drives_the_inference_reference_path(tmp_path):
    from api.param_defaults import SENSENOVA_GENERATION_DEFAULTS

    trainer, _, _ = _sample_trainer()
    recorder = _SampleRecorder()
    path = _write_images(tmp_path, 1)[0]

    image = _run_sample(recorder, trainer, reference_image_path=path)

    assert image is not None
    kwargs = recorder.encode_calls[0][1]
    refs = kwargs["ref_images"]
    assert len(refs) == 1 and refs[0].size == (64, 64)
    assert kwargs["img_cfg_scale"] == SENSENOVA_GENERATION_DEFAULTS["img_cfg_scale"]


def test_sample_uses_explicit_sensenova_guidance_controls(tmp_path):
    trainer, _, _ = _sample_trainer()
    recorder = _SampleRecorder()
    path = _write_images(tmp_path, 1)[0]

    image = _run_sample(
        recorder,
        trainer,
        reference_image_path=path,
        timestep_shift=5.5,
        img_cfg_scale=1.75,
        cfg_norm="none",
    )

    assert image is not None
    assert recorder.encode_calls[0][1]["img_cfg_scale"] == 1.75
    denoise_kwargs = recorder.denoise_calls[0][1]
    assert denoise_kwargs["timestep_shift"] == 5.5
    assert denoise_kwargs["cfg_norm"] == "none"


def test_sample_condition_image_is_ignored_and_never_becomes_a_reference(tmp_path):
    trainer, _, _ = _sample_trainer()
    recorder = _SampleRecorder()
    path = _write_images(tmp_path, 1)[0]

    _run_sample(recorder, trainer, condition_image_path=path)

    assert recorder.encode_calls[0][1]["ref_images"] == []


def test_reference_sample_restores_training_state_and_frees_the_prefix(tmp_path):
    from core.attention import AttentionMode

    trainer, transformer, _ = _sample_trainer()
    recorder = _SampleRecorder()

    _run_sample(recorder, trainer, reference_image_path=_write_images(tmp_path, 1)[0])

    # An INFERENCE stamp left behind would persist for the whole rest of the run.
    assert recorder.modes == [AttentionMode.INFERENCE, AttentionMode.TRAINING]
    assert transformer.training is True
    assert recorder.cleared == [recorder.prefix]


def test_reference_sample_failure_returns_none_and_still_restores(tmp_path):
    from core.attention import AttentionMode

    trainer, transformer, _ = _sample_trainer()
    recorder = _SampleRecorder(denoise_error=RuntimeError("boom"))

    result = _run_sample(
        recorder, trainer, reference_image_path=_write_images(tmp_path, 1)[0]
    )

    assert result is None
    assert recorder.modes == [AttentionMode.INFERENCE, AttentionMode.TRAINING]
    assert transformer.training is True
    assert recorder.cleared == [recorder.prefix]


def test_reference_sample_keeps_the_phase_evictor_transition_pair(tmp_path):
    trainer, _, events = _sample_trainer(with_evictor=True)
    recorder = _SampleRecorder()

    _run_sample(recorder, trainer, reference_image_path=_write_images(tmp_path, 1)[0])

    assert events == ["prefix", "denoise", "resident"]


def test_sample_unreadable_reference_never_takes_the_run_down(tmp_path):
    """Reference loading happens inside the same guard as the generation."""
    trainer, transformer, _ = _sample_trainer()
    recorder = _SampleRecorder()

    result = _run_sample(
        recorder, trainer, reference_image_path=str(tmp_path / "missing.png")
    )

    assert result is None
    assert recorder.encode_calls == []
    assert transformer.training is True


# --- 3-1: the six flux2 hard gates ------------------------------------------


def _source(relative):
    return (Path(__file__).resolve().parents[1] / relative).read_text(encoding="utf-8")


def test_gate1_runner_no_longer_refuses_reference_runs():
    from unittest.mock import patch as _patch

    from core.model_loader import ModelLoader
    from core.training.train_runner import _apply_sensenova_training_contract

    train = {"batch_size": 1, "use_reference_images": True}
    with _patch.object(ModelLoader, "detect_model_type", return_value="sensenova"):
        assert _apply_sensenova_training_contract("checkpoint", "lora", train, {})
    assert train["use_reference_images"] is True
    assert "deferred to Phase 3" not in _source("core/training/train_runner.py")


def test_gate2_trainer_no_longer_refuses_reference_runs():
    assert "deferred to Phase 3" not in _source("core/training/base_trainer.py")


def test_gate3_sensenova_is_not_warned_as_ignored():
    source = _source("core/training/base_trainer.py")
    assert "self.is_flux2 or self.is_sensenova or _sd_ve_arch" in source
    assert "only supported for FLUX.2, will be ignored" not in source


def test_gate4_reference_separation_covers_sensenova():
    source = _source("core/training/base_trainer.py")
    assert (
        "separate_by_reference = use_reference_images and (self.is_flux2 or self.is_sensenova)"
        in source
    )


def test_gates5_and_6_stay_flux2_only_and_sensenova_is_wired_elsewhere():
    """The remaining two `and self.is_flux2` branches are VAE-latent
    conditioning. SenseNova is released from them by having its own entry
    (the prompt prefix), NOT by being added to them."""
    source = _source("core/training/base_trainer.py")
    assert source.count("use_reference_images and self.is_flux2") == 2
    assert "reference_image_paths=(" in source
    assert "item.get(\"reference_images\") or []" in source


def test_arch_capabilities_declares_all_reference_training_paths():
    from api.arch_capabilities import training_feature_unsupported_reason

    assert training_feature_unsupported_reason("sensenova", "reference_images") is None
    assert training_feature_unsupported_reason("flux2", "reference_images") is None
    assert training_feature_unsupported_reason("sd15", "reference_images") is None
    assert training_feature_unsupported_reason("sdxl", "reference_images") is None
    assert training_feature_unsupported_reason("zimage", "reference_images")
