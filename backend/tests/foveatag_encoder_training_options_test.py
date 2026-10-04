"""Optional execution controls must not alter the default encoder contract."""
import torch
from contextlib import contextmanager
from torch.nn.attention import SDPBackend

from core.tagger.foveatag.encoder import FoveaEncoder, FoveaEncoderConfig


def make_encoder():
    return FoveaEncoder(FoveaEncoderConfig(widths=(8, 16, 32, 32), depths=(1, 1, 2, 1),
        window_size=2, attention_heads=(4, 4), mlp_ratio=2, scale_dropout=0))


def inputs():
    return torch.randn(1, 3, 32, 32), torch.tensor([[0.5, 0.5, 1., 1., 0., 0., 0., 0., 1., 1.]])


def test_default_controls_are_off_and_not_checkpoint_state():
    model = make_encoder()
    assert not model.gradient_checkpointing
    assert all(not module.use_sdpa for module in model.modules() if hasattr(module, "use_sdpa"))
    state_keys = set(model.state_dict())
    model.gradient_checkpointing_enable()
    model.set_sdpa(True)
    assert set(model.state_dict()) == state_keys


def test_sdpa_matches_eager_attention():
    model = make_encoder().eval()
    image, geometry = inputs()
    with torch.no_grad():
        eager = model(image, geometry)
        model.set_sdpa(True)
        optimized = model(image, geometry)
    torch.testing.assert_close(optimized.local_map, eager.local_map, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(optimized.global_tokens, eager.global_tokens, rtol=1e-5, atol=1e-5)


def test_checkpointing_preserves_forward_and_gradients():
    eager, checked = make_encoder().train(), make_encoder().train()
    checked.load_state_dict(eager.state_dict())
    checked.gradient_checkpointing_enable()
    image, geometry = inputs()
    first = eager(image, geometry)
    second = checked(image, geometry)
    torch.testing.assert_close(first.global_summary, second.global_summary)
    (first.local_map.square().mean() + first.global_summary.square().mean()).backward()
    (second.local_map.square().mean() + second.global_summary.square().mean()).backward()
    for (name, left), (_, right) in zip(eager.named_parameters(), checked.named_parameters()):
        assert left.grad is not None, name
        torch.testing.assert_close(left.grad, right.grad)


def test_flash_policy_is_applied_to_each_sdpa_call(monkeypatch):
    seen = []

    @contextmanager
    def kernel(backends):
        seen.append(backends)
        yield

    monkeypatch.setattr("core.tagger.foveatag.encoder.sdpa_kernel", kernel)
    model = make_encoder().eval()
    image, geometry = inputs()
    for allowed in (False, True):
        seen.clear()
        model.set_sdpa(True, allow_flash=allowed)
        with torch.no_grad():
            model(image, geometry)
        assert seen
        assert all((SDPBackend.FLASH_ATTENTION in backends) == allowed for backends in seen)
        assert all(SDPBackend.MATH in backends for backends in seen)
