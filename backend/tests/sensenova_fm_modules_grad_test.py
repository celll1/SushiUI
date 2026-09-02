"""`sensenova_train_fm_modules`: the gradient actually has to reach 16 tensors.

`_build_step_context` is inference's per-step embed builder and the training step
reuses it. While it carried an unconditional `@torch.no_grad()`, everything it
calls -- the generation ViT (`extract_feature(gen_model=True)`), the timestep
embedder and the noise-scale embedder, i.e. 12 of the 16 `fm_modules` tensors --
was cut out of the autograd graph, so enabling the option trained only the 4
`fm_head` tensors, which are produced after that call. Two real checkpoints 976
steps apart show exactly that split: `fm_head` moved, the other 12 are
byte-identical.

Every other SenseNova test patches `_build_step_context` out, so this file runs
the real one.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/sensenova_fm_modules_grad_test.py -v
"""

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sensenova_training_core_test import _Cache, _Transformer  # noqa: E402

from core.models.sensenova.sensenova_pipeline_ops import (  # noqa: E402
    _build_step_context,
)
from core.models.sensenova.vendor.modeling_fm_modules import (  # noqa: E402
    ConvDecoder,
    TimestepEmbedder,
)
from core.training.ops.sensenova_ops import (  # noqa: E402
    SenseNovaTrainingPrefix,
    train_step,
)

# The distributed checkpoint's fm_modules index, tensor for tensor.
FM_TENSORS = 16
# ConvDecoder's fixed 4x + 8x pixel shuffle turns one token into a 32x32 patch.
HIDDEN = 64


class _GenVision(nn.Module):
    """`vision_model_mot_gen`'s two trainable tensor pairs, merge included.

    The real tower's per-layer weights are shared with the understanding tower;
    what `fm_modules` carries is the patch and dense embeddings.
    """

    def __init__(self, patch_dim, merged):
        super().__init__()
        self.embeddings = nn.Module()
        self.embeddings.patch_embedding = nn.Linear(patch_dim, HIDDEN)
        self.embeddings.dense_embedding = nn.Linear(HIDDEN * merged, HIDDEN)

    def forward(self, pixel_values, output_hidden_states=False, return_dict=True, grid_hw=None):
        patches = self.embeddings.patch_embedding(pixel_values)
        merged = patches.reshape(1, -1)
        return SimpleNamespace(last_hidden_state=self.embeddings.dense_embedding(merged))


class _FmTransformer(_Transformer):
    """`_Transformer` with all four `fm_modules` entries, real where it matters.

    `timestep_embedder` / `noise_scale_embedder` / `fm_head` are the vendor
    classes at shrunk widths; only the ViT is a double, since the real one needs
    a config file. 16 tensors, like the checkpoint.
    """

    def __init__(self):
        super().__init__()
        self.add_noise_scale_embedding = True
        self.noise_scale = 1.0
        self.noise_scale_mode = "resolution"
        self.noise_scale_base_image_seq_len = 1.0
        self.noise_scale_max_value = 3.0
        merge_size = int(1 / self.downsample_ratio)
        self.fm_modules = nn.ModuleDict({
            "vision_model_mot_gen": _GenVision(3 * self.patch_size ** 2, merge_size ** 2),
            "timestep_embedder": TimestepEmbedder(HIDDEN),
            "fm_head": ConvDecoder(input_dim=HIDDEN, hidden_dim=HIDDEN),
            "noise_scale_embedder": TimestepEmbedder(HIDDEN),
        })

    def extract_feature(self, pixel_values, gen_model=False, grid_hw=None):
        assert gen_model
        return self.fm_modules["vision_model_mot_gen"](
            pixel_values=pixel_values, output_hidden_states=False, return_dict=True, grid_hw=grid_hw
        ).last_hidden_state

    def patchify(self, image, patch, channel_first=False):
        if not channel_first:
            return super().patchify(image, patch)
        batch, channels, height, width = image.shape
        return image.view(
            batch, channels, height // patch, patch, width // patch, patch
        ).permute(0, 2, 4, 1, 3, 5).reshape(batch, -1, channels * patch * patch)


def _shape():
    return SimpleNamespace(
        batch_size=1, merge_size=2, grid_h=2, grid_w=2, token_h=1, token_w=1
    )


def _fm_grads(transformer):
    return {
        name: parameter.grad
        for name, parameter in transformer.fm_modules.named_parameters()
    }


def _run_train_step(transformer):
    trainer = SimpleNamespace(
        transformer=transformer,
        device=torch.device("cpu"),
        training_dtype=torch.float32,
        gradient_checkpointing=False,
    )
    loss, _, _ = train_step(
        trainer,
        images=torch.ones(1, 3, 32, 32),
        prefix=SenseNovaTrainingPrefix(_Cache(), text_length=3),
        timesteps=torch.tensor([0.25]),
    )
    loss.backward()
    return loss


def _train_fm_modules(transformer, on):
    """What the adapter does: freeze the tree, then opt fm_modules back in."""
    transformer.requires_grad_(False)
    transformer.fm_modules.requires_grad_(on)
    return transformer


def test_the_double_carries_the_checkpoint_tensor_count():
    assert len(list(_FmTransformer().fm_modules.parameters())) == FM_TENSORS


# ---------------------------------------------------------------------------
# The function itself
# ---------------------------------------------------------------------------

def test_build_step_context_is_no_grad_by_default_and_differentiable_on_request():
    transformer = _FmTransformer()
    image = torch.ones(1, 3, 32, 32)

    _, image_embeds, timestep_embeddings = _build_step_context(
        transformer, _shape(), image, torch.tensor(0.25), 2.0
    )
    assert not image_embeds.requires_grad and not timestep_embeddings.requires_grad

    _, image_embeds, timestep_embeddings = _build_step_context(
        transformer, _shape(), image, torch.tensor(0.25), 2.0, enable_grad=True
    )
    assert image_embeds.requires_grad and timestep_embeddings.requires_grad


def test_enable_grad_is_keyword_only_and_defaults_off():
    parameter = inspect.signature(_build_step_context).parameters["enable_grad"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is False


def test_build_step_context_carries_no_unconditional_no_grad_decorator():
    """The regression itself: a decorator here overrides the keyword silently."""

    @torch.no_grad()
    def decorated():
        pass

    # The detector detects: torch's decorator functools.wraps its target.
    assert getattr(decorated, "__wrapped__", None) is not None
    assert getattr(_build_step_context, "__wrapped__", None) is None


# ---------------------------------------------------------------------------
# Through the training step
# ---------------------------------------------------------------------------

def test_the_option_on_gives_every_one_of_the_16_tensors_a_gradient():
    transformer = _train_fm_modules(_FmTransformer(), True)
    _run_train_step(transformer)

    grads = _fm_grads(transformer)
    assert len(grads) == FM_TENSORS
    missing = sorted(name for name, grad in grads.items() if grad is None)
    assert not missing
    assert all(torch.isfinite(grad).all() for grad in grads.values())


def test_the_option_off_builds_no_graph_over_the_generation_vit():
    """Frozen fm_modules must cost exactly what it cost before: no graph."""
    transformer = _train_fm_modules(_FmTransformer(), False)
    seen = {}
    real = _build_step_context

    def spy(*args, **kwargs):
        result = real(*args, **kwargs)
        seen.update(enable_grad=kwargs.get("enable_grad"), embeds=result[1])
        return result

    with patch(
        "core.models.sensenova.sensenova_pipeline_ops._build_step_context", side_effect=spy
    ):
        _run_train_step(transformer)

    assert seen["enable_grad"] is False
    assert not seen["embeds"].requires_grad
    assert all(grad is None for grad in _fm_grads(transformer).values())


def test_the_shipped_defect_is_what_the_no_grad_path_would_still_produce():
    """Pre-fix behaviour, pinned: fm_head trains, the other 12 tensors do not.

    Forcing the no-grad path with the 16 tensors trainable reproduces the split
    measured on two real checkpoints, and shows the 4/12 boundary is the
    function call rather than anything about the tensors.
    """
    transformer = _train_fm_modules(_FmTransformer(), True)
    real = _build_step_context

    def no_grad_context(*args, **kwargs):
        return real(*args, **{**kwargs, "enable_grad": False})

    with patch(
        "core.models.sensenova.sensenova_pipeline_ops._build_step_context",
        side_effect=no_grad_context,
    ):
        _run_train_step(transformer)

    got = {name for name, grad in _fm_grads(transformer).items() if grad is not None}
    assert got == {
        "fm_head.conv1.weight", "fm_head.conv1.bias",
        "fm_head.conv2.weight", "fm_head.conv2.bias",
    }


def test_a_flag_on_the_transformer_does_not_open_the_gate():
    """Frozen fm parameters keep the gate shut whatever else is set.

    The gate reads requires_grad, so the cases where the flag is on but the
    adapter collected nothing -- an understanding-only branch, or LoRA -- come
    out no-grad without this call site knowing which case it is in. Those two
    resolutions are the adapter's, and are covered in
    sensenova_fm_modules_training_test.py.
    """
    transformer = _train_fm_modules(_FmTransformer(), False)
    transformer.sensenova_train_fm_modules = True  # nothing reads this
    seen = []
    real = _build_step_context

    with patch(
        "core.models.sensenova.sensenova_pipeline_ops._build_step_context",
        side_effect=lambda *a, **k: (seen.append(k.get("enable_grad")), real(*a, **k))[1],
    ):
        _run_train_step(transformer)

    assert seen == [False]


@pytest.mark.parametrize("on", [True, False])
def test_the_loss_is_unchanged_by_whether_the_graph_is_built(on):
    """Enabling the option must move memory, not numerics."""
    torch.manual_seed(0)
    reference = _train_fm_modules(_FmTransformer(), False)
    torch.manual_seed(0)
    subject = _train_fm_modules(_FmTransformer(), on)

    with patch("torch.randn_like", return_value=torch.full((1, 3, 32, 32), 0.2)):
        expected = _run_train_step(reference)
        actual = _run_train_step(subject)
    assert torch.equal(expected.detach(), actual.detach())
