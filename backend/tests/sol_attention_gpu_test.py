"""GPU acceptance tests for the optional official Sol-Attn package."""

import importlib.util
import os
import sys

import pytest
import torch
import torch.nn.functional as F

_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from core.attention import AttentionMode, dispatch_planned_attention
from core.models.minimax_h3.sparse_attention import H3SolAttentionPlan


def _require_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if importlib.util.find_spec("sol_attn") is None:
        pytest.skip("optional sol-attn package is not installed")
    major, _minor = torch.cuda.get_device_capability()
    if major < 8:
        pytest.skip("Sol-Attn requires SM80 or newer")


def _qkv(tokens: int, heads: int = 2):
    generator = torch.Generator(device="cuda").manual_seed(1234 + tokens)
    return tuple(
        torch.randn(
            1, tokens, heads, 128,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        for _ in range(3)
    )


def _dense(q, k, v):
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    ).transpose(1, 2)


@pytest.mark.parametrize("tokens", [63, 64, 65, 127, 128, 129])
def test_full_sink_matches_dense_across_block_boundaries(tokens):
    _require_gpu()
    from sol_attn import sol_attn

    q, k, v = _qkv(tokens)
    with torch.inference_mode():
        expected = _dense(q, k, v)
        actual = sol_attn(q, k, v, sink_start=0, sink_tokens=tokens)
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.03)


def test_sparse_kernel_returns_finite_contiguous_output():
    _require_gpu()
    from sol_attn import sol_attn

    q, k, v = _qkv(257)
    with torch.inference_mode():
        output = sol_attn(q, k, v, tau=1.0, thresh_type="exact", sink_start=0, sink_tokens=65)
    assert output.shape == q.shape
    assert output.dtype == torch.bfloat16
    assert output.is_contiguous()
    assert torch.isfinite(output).all()


def test_h3_adapter_restores_prefix_query_rows_to_dense():
    _require_gpu()
    q, k, v = _qkv(193)
    prefix_tokens = 65
    plan = H3SolAttentionPlan(
        position_ids=torch.zeros(193, 3, device="cuda"),
        target_video_rows=torch.tensor(
            [False] * prefix_tokens + [True] * (193 - prefix_tokens), device="cuda"
        ),
        dense_steps=0,
        dense_layers=0,
    )
    plan.begin_forward()

    with torch.inference_mode():
        expected_prefix = _dense(q[:, :prefix_tokens], k, v)
        actual = dispatch_planned_attention(
            q,
            k,
            v,
            plan,
            dense_backend="native",
            mode=AttentionMode.INFERENCE,
            layer_index=0,
        )
    torch.testing.assert_close(
        actual[:, :prefix_tokens], expected_prefix, rtol=0.03, atol=0.03
    )
    assert torch.isfinite(actual[:, prefix_tokens:]).all()


def test_tiny_h3_forward_stamps_layers_and_runs_sol_path():
    _require_gpu()
    from core.models.minimax_h3.vendor import MiniMaxH3Transformer3DModel
    from core.models.minimax_h3_block_loop_wrapper import MiniMaxH3BlockLoopWrapper

    model = MiniMaxH3Transformer3DModel(
        num_attention_heads=1,
        attention_head_dim=128,
        hidden_size=128,
        num_layers=3,
        num_refiner_layers=1,
        ffn_dim=256,
        in_channels=8,
        audio_in_channels=8,
        patch_size=(1, 1, 1),
        text_dim=32,
        freq_dim=32,
        time_embed_hidden_dim=128,
        time_embed_dim=64,
        rope_freq_dim=16,
        adaln_curve_grid=33,
    ).to(device="cuda", dtype=torch.bfloat16).eval()
    model.adaln_t_table.data = model.adaln_t_table.float()
    num_text, num_audio, num_video = 33, 32, 128
    total = num_text + num_audio + num_video
    text_indices = torch.arange(num_text, device="cuda")
    audio_indices = torch.arange(num_text, num_text + num_audio, device="cuda")
    video_indices = torch.arange(num_text + num_audio, total, device="cuda")
    token_tags = torch.cat((
        torch.ones(num_text, device="cuda", dtype=torch.long),
        torch.full((num_audio,), 2, device="cuda", dtype=torch.long),
        torch.zeros(num_video, device="cuda", dtype=torch.long),
    ))
    plan = H3SolAttentionPlan(
        position_ids=torch.zeros(total, 3, device="cuda"),
        target_video_rows=torch.cat((
            torch.zeros(num_text + num_audio, device="cuda", dtype=torch.bool),
            torch.ones(num_video, device="cuda", dtype=torch.bool),
        )),
        dense_steps=0,
        dense_layers=1,
    )
    model._attention_plan = plan
    model._attn_backend = "native"
    generator = torch.Generator(device="cuda").manual_seed(99)
    kwargs = dict(
        hidden_states=torch.randn(1, num_video, 8, device="cuda", dtype=torch.bfloat16, generator=generator),
        audio_hidden_states=torch.randn(1, num_audio, 8, device="cuda", dtype=torch.bfloat16, generator=generator),
        encoder_hidden_states=torch.randn(1, num_text, 32, device="cuda", dtype=torch.bfloat16, generator=generator),
        timestep=torch.tensor([0.5], device="cuda"),
        timestep_indices=torch.zeros(total, device="cuda", dtype=torch.long),
        token_tags=token_tags,
        position_ids=torch.zeros(total, 3, device="cuda", dtype=torch.long),
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
        return_dict=False,
    )
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        video, audio = model(**kwargs)

    assert plan._step_index == 0
    assert video.shape == (1, num_video, 8)
    assert audio.shape == (1, num_audio, 8)
    assert torch.isfinite(video).all() and torch.isfinite(audio).all()
    assert [block.attn._attention_layer_index for block in model.transformer_blocks] == [0, 1, 2]

    class _ResidentOffloader:
        blocks_to_swap = 1

        def wait_for_block(self, _index):
            pass

        def submit_move_blocks_forward(self, _index):
            pass

    wrapper = MiniMaxH3BlockLoopWrapper(model, block_offloader=_ResidentOffloader())
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        wrapped_video, wrapped_audio = wrapper(**kwargs)
    assert plan._step_index == 1
    assert torch.isfinite(wrapped_video).all() and torch.isfinite(wrapped_audio).all()
