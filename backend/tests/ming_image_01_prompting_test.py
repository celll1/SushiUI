"""Native chat/vision preparation, RGBA order and full vision gradient oracles."""

import sys
from pathlib import Path

import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def forbidden_cuda(*args, **kwargs):
    raise AssertionError("Ming prompting test attempted CUDA initialization")


torch.cuda.get_device_capability = lambda *a, **k: (8, 9)
torch.cuda._lazy_init = forbidden_cuda
torch._C._cuda_init = forbidden_cuda
torch.set_num_threads(1)

from core.models.ming_image_01.prompting import prepare_inputs, encode_inputs
from core.models.ming_image_01.vendor.tokenization_bailing import BailingTokenizer
from core.models.ming_image_01.vendor.image_processing_bailingmm2 import BailingMM2ImageProcessor
from core.models.ming_image_01.vendor.processing_bailingmm2 import BailingMM2Processor
from ming_image_01_conditioning_test import make_conditioner


def processor():
    vocabulary = {f"word{i}": i for i in range(64)}
    for text, index in (("<imagePatch>", 40), ("<image>", 41), ("</image>", 42)):
        del vocabulary[f"word{index}"]
        vocabulary[text] = index
    tokenizer_impl = Tokenizer(WordLevel(vocabulary, unk_token="word3"))
    tokenizer_impl.pre_tokenizer = Whitespace()
    tokenizer = BailingTokenizer(tokenizer_object=tokenizer_impl, bos_token="word1", eos_token="word2",
                                pad_token="word0", gmask_token="word3", cls_token="word4",
                                additional_special_tokens=["<imagePatch>", "<image>", "</image>"])
    image_processor = BailingMM2ImageProcessor(patch_size=2, temporal_patch_size=1, merge_size=2,
                                              min_pixels=16, max_pixels=64)
    return BailingMM2Processor(image_processor=image_processor, tokenizer=tokenizer)


def options():
    return dict(width=48, height=32, resolution=512)


def test_native_template_tokens_and_fixed_rgba_reference_order():
    native = processor()
    red = Image.new("RGBA", (60, 40), (255, 0, 0, 255))
    green = Image.new("RGBA", (40, 60), (0, 255, 0, 128))
    prepared = prepare_inputs(native, ["word5 word6"], references=[[red, green]], **options())
    expected = native.apply_chat_template([{"role": "HUMAN", "content": [
        {"type": "image", "image": red}, {"type": "image", "image": green},
        {"type": "text", "text": "word5 word6"}]}], add_generation_prompt=True)
    assert prepared.texts == [expected]
    assert prepared.model_inputs["image_grid_thw"].shape[0] == 2
    assert (prepared.model_inputs["input_ids"] == 41).sum() == 2
    refs = prepared.generation_references()
    assert refs.shape == (1, 8, 32, 48)
    assert refs[0, 3].eq(1).all()
    assert refs[0, 0].eq(1).all() and refs[0, 1].eq(-1).all()
    assert refs[0, 4].eq(-1).all() and refs[0, 5].eq(1).all()
    torch.testing.assert_close(refs[0, 7], torch.full((32, 48), 128 / 127.5 - 1))


def test_real_native_prompt_and_vision_backward_keeps_both_ownership_groups():
    model = make_conditioner()
    model.set_trainable({"language", "vision", "queries", "connector", "projections"})
    prepared = prepare_inputs(processor(), ["word5"], references=[[Image.new("RGB", (32, 32), "red")]], **options())
    condition = encode_inputs(model, prepared, requires_grad=True)
    (condition.direct.square().mean() + condition.queries.square().mean()).backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.native.linear_proj.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.native.vision.parameters())
    assert model.native.model.get_input_embeddings().weight.grad.abs().sum() > 0
    inference = encode_inputs(model, prepared, requires_grad=False)
    assert not inference.direct.requires_grad and not inference.queries.requires_grad
    assert not torch.cuda.is_initialized()


def test_mixed_reference_counts_preserved_for_training_and_refused_for_generation():
    prepared = prepare_inputs(processor(), ["word5", "word6 word7"], references=[[], [Image.new("RGBA", (32, 32))]], **options())
    assert prepared.references[0] is None
    assert prepared.references[1].shape == (4, 32, 48)
    with pytest.raises(ValueError, match="equal reference counts"):
        prepared.generation_references()


def test_conditioning_only_skips_rgba_copies_without_changing_native_inputs():
    native = processor()
    references = [[Image.new("RGBA", (60, 40), (70, 80, 90, 100))]]
    original = prepare_inputs(native, ["word5"], references=references, **options())
    compact = prepare_inputs(native, ["word5"], references=references, vae_references=False, **options())
    assert compact.references == [None]
    assert compact.texts == original.texts
    for key, expected in original.model_inputs.items():
        torch.testing.assert_close(compact.model_inputs[key], expected, rtol=0, atol=0)


def test_text_only_native_batch_and_frozen_inference():
    prepared = prepare_inputs(processor(), ["word5", "word6 word7"], references=[[], []], **options())
    assert prepared.generation_references() is None
    assert "pixel_values" not in prepared.model_inputs
    model = make_conditioner().eval()
    model.set_trainable(set())
    result = encode_inputs(model, prepared, requires_grad=False)
    assert result.queries.shape == (2, 4, 8)
    assert not result.direct.requires_grad


@pytest.mark.parametrize("references", [False, True])
def test_frozen_caption_vision_is_staged_only_for_references(monkeypatch, references):
    model = make_conditioner().eval()
    model.set_trainable(set())
    images = [[Image.new("RGB", (32, 32), "red")]] if references else [[]]
    prepared = prepare_inputs(processor(), ["word5"], references=images, **options())
    expected = encode_inputs(model, prepared, requires_grad=False)
    model.stage_training_caption("cpu")
    events = []
    for module in model.component_groups()["vision"]:
        original = module.to
        def tracked(*args, _original=original, **kwargs):
            events.append(str(kwargs["device"]))
            return _original(*args, **kwargs)
        monkeypatch.setattr(module, "to", tracked)
    actual = encode_inputs(model, prepared, requires_grad=False)
    torch.testing.assert_close(actual.queries, expected.queries, rtol=0, atol=0)
    torch.testing.assert_close(actual.direct, expected.direct, rtol=0, atol=0)
    assert events == (["cpu"] * 4 if references else [])
    assert not actual.direct.requires_grad


def test_caption_device_follows_language_when_vision_is_elsewhere():
    model = make_conditioner().eval()
    model.set_trainable(set())
    model.native.vision.to("meta")
    assert next(model.parameters()).device.type == "meta"
    assert model.native.device.type == "cpu"
    prepared = prepare_inputs(processor(), ["word5"], references=[[]], **options())
    result = encode_inputs(model, prepared, requires_grad=False)
    assert result.direct.device.type == "cpu"


@pytest.mark.parametrize("change,match", [
    ({"references": [["https://example.invalid/image"]]}, "decoded PIL"),
    ({"references": []}, "per prompt"),
    ({"width": 33}, "16-pixel"),
])
def test_invalid_preparation_refused_before_processing(change, match):
    kwargs = dict(prompts=["word5"], references=[[]], **options())
    kwargs.update(change)
    with pytest.raises(ValueError, match=match):
        prepare_inputs(processor(), **kwargs)


@pytest.mark.parametrize("profile,task,layer_count", [
    ("generation_edit", "image-edit", 1), ("layer_decompose", "layer-decompose", 2),
])
def test_full_native_prompt_vision_sampler_and_residency_context_release(profile, task, layer_count):
    from contextlib import contextmanager
    from core.models.ming_image_01.generation import MingGenerator
    from ming_image_01_sampling_test import make_sampler
    components = make_sampler(profile).components
    components.processor = processor()
    components.conditioner = make_conditioner().eval()
    components.conditioner.set_trainable(set())
    phases = []
    @contextmanager
    def phase(name):
        phases.append((name, "enter"))
        try:
            yield
        finally:
            phases.append((name, "exit"))
    generator = MingGenerator(components)
    arguments = dict(prompts=["word5"], references=[[Image.new("RGBA", (32, 48), (255, 0, 0, 128))]],
                     task=task, **options(), steps=1, cfg=1., seed=7, requested_layers=layer_count,
                     phase_context=phase)
    result = generator.run(**arguments)
    assert len(result) == 1
    assert result[0].target_frames == (layer_count + 1 if task == "layer-decompose" else 1)
    assert all(image.mode == "RGBA" and image.size == (48, 32) for image in result[0].values)
    assert phases == [("conditioner", "enter"), ("conditioner", "exit"), ("denoising", "enter"), ("denoising", "exit")]
    from core.models.ming_image_01.sampling import MingGenerationCancelled
    phases.clear()
    with pytest.raises(MingGenerationCancelled):
        generator.run(**arguments, cancelled=lambda: True)
    assert phases == []
    def fail_step(*args):
        raise RuntimeError("synthetic step callback failure")
    with pytest.raises(RuntimeError, match="synthetic"):
        generator.run(**arguments, step_callback=fail_step)
    assert phases == [("conditioner", "enter"), ("conditioner", "exit"), ("denoising", "enter"), ("denoising", "exit")]
    assert not torch.cuda.is_initialized()
