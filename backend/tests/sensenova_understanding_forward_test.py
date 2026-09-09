from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch


BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


from core.models.sensenova.vendor.modeling_neo_chat import NEOChatModel  # noqa: E402


class _LanguageModel:
    def __init__(self):
        self.embedding = torch.nn.Embedding(32, 4)
        self.call = None

    def get_input_embeddings(self):
        return self.embedding

    def __call__(self, **kwargs):
        self.call = kwargs
        return SimpleNamespace(loss=kwargs["inputs_embeds"].sum())


def _model(visual_count=2):
    language_model = _LanguageModel()
    visual = torch.arange(visual_count * 4, dtype=torch.float32).reshape(visual_count, 4)
    model = SimpleNamespace(
        img_context_token_id=7,
        language_model=language_model,
        get_thw_indexes=lambda ids, grid: torch.full((3, ids.numel()), 9),
        extract_feature=lambda pixels, grid_hw: visual,
    )
    return model, language_model, visual


def test_understanding_forward_replaces_only_image_context_and_passes_labels():
    model, language_model, visual = _model()
    input_ids = torch.tensor([[1, 7, 7, 2]])
    labels = torch.tensor([[-100, -100, -100, 2]])
    attention = torch.ones_like(input_ids)
    before = language_model.embedding(input_ids).detach()

    output = NEOChatModel.forward_understanding(
        model, torch.zeros(2, 3), input_ids, torch.tensor([[1, 2]]), attention, labels
    )

    call = language_model.call
    assert output.loss.requires_grad
    assert torch.equal(call["inputs_embeds"][0, 0], before[0, 0])
    assert torch.equal(call["inputs_embeds"][0, 1:3], visual)
    assert torch.equal(call["inputs_embeds"][0, 3], before[0, 3])
    assert call["labels"] is labels
    assert call["use_cache"] is False
    assert call["indexes"].shape == (3, 4)


def test_understanding_forward_refuses_visual_token_mismatch():
    model, _language_model, _visual = _model(visual_count=1)
    with pytest.raises(ValueError, match="tokens=2, features=1"):
        NEOChatModel.forward_understanding(
            model,
            torch.zeros(1, 3),
            torch.tensor([[1, 7, 7, 2]]),
            torch.tensor([[1, 2]]),
            torch.ones(1, 4, dtype=torch.long),
            torch.tensor([[-100, -100, -100, 2]]),
        )


def test_guarded_generic_forward_remains_guarded():
    source = (BACKEND / "core/models/sensenova/vendor/modeling_neo_chat.py").read_text(encoding="utf-8")
    generic = source[source.index("    def forward(\n"):source.index("    def extract_feature", source.index("    def forward(\n"))]
    assert "raise NotImplementedError('forward')" in generic

