"""The Qwen caption path retains pre-norm states while restricting unused logits."""
import copy
import inspect
import textwrap
from types import SimpleNamespace, MethodType

import torch
from transformers import BatchEncoding, Qwen3VLConfig, Qwen3VLForConditionalGeneration

from core.models.qwen_image_21.vendor import pipeline


def test_one_token_logits_preserve_caption_outputs_gradients_and_norm_hook():
    config = Qwen3VLConfig(
        text_config={"vocab_size": 128, "hidden_size": 32, "intermediate_size": 64,
                     "num_hidden_layers": 1, "num_attention_heads": 4,
                     "num_key_value_heads": 4, "head_dim": 8},
        vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64,
                       "num_heads": 4, "out_hidden_size": 32,
                       "num_position_embeddings": 16, "deepstack_visual_indexes": []})
    config._attn_implementation = "eager"
    encoder = Qwen3VLForConditionalGeneration(config).eval()
    original = copy.deepcopy(encoder)
    data = {"input_ids": torch.tensor([[0, 2, 3, 4], [5, 6, 7, 8]]),
            "attention_mask": torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]])}
    def owner(model):
        result = SimpleNamespace(text_encoder=model, processor=lambda **_: BatchEncoding(data),
            prompt_template_t2i="{}", _drop_idx=0, _img_token_id=127)
        result._extract_masked_hidden = MethodType(pipeline.QwenImage21Pipeline._extract_masked_hidden, result)
        return result
    source = textwrap.dedent(inspect.getsource(pipeline.QwenImage21Pipeline._get_qwen_prompt_embeds))
    namespace = dict(vars(pipeline))
    exec(compile(source.replace('"logits_to_keep": 1,', ''), '<full logits oracle>', 'exec'), namespace)
    lengths = []
    handle = encoder.lm_head.register_forward_pre_hook(lambda _, args: lengths.append(args[0].shape[1]))
    actual = pipeline.QwenImage21Pipeline._get_qwen_prompt_embeds(owner(encoder), ['one', 'two'], device='cpu')
    expected = namespace['_get_qwen_prompt_embeds'](owner(original), ['one', 'two'], device='cpu')
    assert lengths == [1]
    handle.remove()
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    actual[0].square().sum().backward()
    expected[0].square().sum().backward()
    for left, right in zip(encoder.parameters(), original.parameters()):
        assert (left.grad is None) == (right.grad is None)
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
    assert not encoder.model.language_model.norm._forward_hooks
