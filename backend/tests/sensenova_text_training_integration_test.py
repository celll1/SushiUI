from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch
from torch import nn


BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.training.adapters.sensenova_adapter import (  # noqa: E402
    SenseNovaFullParameterAdapter,
)
from core.training.ops.sensenova_ops import (  # noqa: E402
    resolve_full_finetune_branch,
    train_i2t_step,
)
from core.training.optimizers.update_census import UpdateCensus  # noqa: E402
from core.training.sensenova_tasks import resolve_generation_caption  # noqa: E402
from core.training.train_runner import _apply_sensenova_task_contract  # noqa: E402


class _Understanding(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.calls = []

    def forward_understanding(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(loss=self.scale * kwargs["pixel_values"].mean())


def _example(value, tokens, weight=1.0, task="i2t_caption"):
    return {
        "task": task,
        "target_tokens": tokens,
        "loss_weight": weight,
        "pixel_values": torch.full((1, 1), float(value)),
        "input_ids": torch.ones(1, 2, dtype=torch.long),
        "grid_hw": torch.ones(1, 2, dtype=torch.long),
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
        "labels": torch.ones(1, 2, dtype=torch.long),
    }


def test_i2t_loss_is_target_token_normalized_and_weighted():
    transformer = _Understanding()
    trainer = SimpleNamespace(
        transformer=transformer,
        train_unet=False,
        train_text_encoder=False,
        device=torch.device("cpu"),
        training_dtype=torch.bfloat16,
        sensenova_phase_evictor=None,
    )

    loss, raw_ce, reconstruction = train_i2t_step(
        trainer,
        examples=[_example(2, 1, 3), _example(4, 3, 0.5)],
    )

    assert raw_ce.item() == pytest.approx((2 * 1 + 4 * 3) / 4)
    assert loss.item() == pytest.approx((2 * 1 * 3 + 4 * 3 * 0.5) / 4)
    assert reconstruction == 0.0
    loss.backward()
    assert transformer.scale.grad is not None
    assert len(transformer.calls) == 2


def test_i2t_step_refuses_a_mixed_task_batch():
    trainer = SimpleNamespace(
        transformer=_Understanding(), device=torch.device("cpu"),
        training_dtype=torch.bfloat16, sensenova_phase_evictor=None,
    )
    with pytest.raises(ValueError, match="task-homogeneous"):
        train_i2t_step(
            trainer,
            examples=[_example(1, 1), _example(1, 1, task="i2t_tags")],
        )


def test_explicit_non_decoder_scope_resolves_without_materializing_a_half():
    trainer = SimpleNamespace(
        train_unet=False,
        train_text_encoder=False,
        config={
            "_sensenova_explicit_tasks": ["i2t_caption"],
            "sensenova_train_scopes": ["understanding_vision"],
        },
    )
    assert resolve_full_finetune_branch(trainer) == "none"


def test_full_ft_native_scope_groups_are_separate():
    class _Language(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.Embedding(4, 3)
            self.output = nn.Linear(3, 4, bias=False)

        def get_input_embeddings(self):
            return self.input

        def get_output_embeddings(self):
            return self.output

    transformer = nn.Module()
    transformer.vision_model = nn.Linear(2, 3)
    transformer.language_model = _Language()
    transformer.fm_modules = nn.Linear(3, 2)
    trainer = SimpleNamespace(
        transformer=transformer,
        train_unet=False,
        train_text_encoder=False,
        config={
            "_sensenova_explicit_tasks": ["i2t_caption", "t2i"],
            "sensenova_train_scopes": [
                "understanding_vision", "shared", "generation_flow",
            ],
        },
        sensenova_train_fm_modules=True,
        learning_rate=1e-5,
        image_encoder_lr=2e-5,
        text_encoder_lr=3e-5,
        text_encoder_1_lr=None,
        unet_lr=4e-5,
    )
    adapter = SenseNovaFullParameterAdapter(trainer)
    transformer.requires_grad_(False)
    scopes = adapter._scope_parameters("none", [])
    for parameters in scopes.values():
        for parameter in parameters:
            parameter.requires_grad_(True)
    groups = adapter.arch_param_groups()

    assert [group["name"] for group in groups] == [
        "understanding_vision", "shared", "generation_flow",
    ]
    assert [group["lr"] for group in groups] == [2e-5, 1e-5, 4e-5]
    assert not ({id(p) for p in groups[0]["params"]}
                & {id(p) for p in groups[1]["params"]})


def test_update_census_checks_only_the_active_task_path():
    text = nn.Parameter(torch.ones(1))
    image = nn.Parameter(torch.ones(1))
    census = UpdateCensus()
    census.expect([text, image], {id(text): "text", id(image): "image"})
    census.begin_step(active_ids={id(text)})
    census.record(text)
    census.assert_complete("text task")
    assert census.unexpected_count() == 0


def test_generation_caption_uses_the_selected_sources_in_order():
    captions = {
        "natural": {"content": "a concise caption"},
        "tags": {"content": "blue_hair, solo"},
    }
    assert resolve_generation_caption(captions, ["tags", "natural"]) == (
        "blue_hair, solo\na concise caption"
    )


def _task_process(task="i2t_caption"):
    return {
        "datasets": [{
            "dataset_id": 25,
            "task_views": [{
                "task": task,
                "target_caption_types": ["natural"],
                "hint_caption_types": [],
                "weight": 1.0,
                "loss_weight": 1.0,
                "hint_dropout": 0.25,
                "prompt_template_version": 1,
            }],
        }],
    }


def test_task_contract_bridges_explicit_lora_scopes(monkeypatch):
    from core.model_loader import ModelLoader

    monkeypatch.setattr(ModelLoader, "detect_model_type", lambda _path: "sensenova")
    train = {
        "sensenova_train_scopes": ["understanding_decoder"],
        "multi_noise_timesteps": 1,
    }
    assert _apply_sensenova_task_contract(
        "model", "lora", train, _task_process()
    )
    assert train["train_unet"] is False
    assert train["train_text_encoder"] is True
    assert train["_sensenova_explicit_tasks"] == ["i2t_caption"]


def test_task_contract_refuses_a_task_with_every_path_scope_frozen(monkeypatch):
    from core.model_loader import ModelLoader

    monkeypatch.setattr(ModelLoader, "detect_model_type", lambda _path: "sensenova")
    train = {
        "sensenova_train_scopes": ["generation_decoder"],
        "multi_noise_timesteps": 1,
    }
    with pytest.raises(ValueError, match="freeze every parameter"):
        _apply_sensenova_task_contract(
            "model", "lora", train, _task_process()
        )


def test_non_decoder_full_ft_scope_requires_mixed_checkpoint(monkeypatch):
    from core.model_loader import ModelLoader

    monkeypatch.setattr(ModelLoader, "detect_model_type", lambda _path: "sensenova")
    train = {
        "sensenova_train_scopes": ["understanding_vision"],
        "sensenova_full_finetune_save_format": "bf16",
        "multi_noise_timesteps": 1,
    }
    with pytest.raises(ValueError, match="save_format='mixed'"):
        _apply_sensenova_task_contract(
            "model", "full_finetune", train, _task_process()
        )
