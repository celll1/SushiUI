from pathlib import Path
import sys

import yaml


BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


from core.training.dataset_params import (  # noqa: E402
    extract_dataset_params,
    read_dataset_params,
)
from core.training.training_config import _build_train_section  # noqa: E402


TASK_VIEW = {
    "task": "i2t_caption_tags",
    "target_caption_types": ["natural_language", "tags"],
    "hint_caption_types": ["tags"],
    "weight": 0.5,
    "loss_weight": 1.25,
    "hint_dropout": 0.4,
    "prompt_template_version": 1,
}


def test_task_views_round_trip_as_dataset_level_config():
    written = extract_dataset_params({"task_views": [TASK_VIEW]})
    assert written == {"task_views": [TASK_VIEW]}
    assert read_dataset_params(written)["task_views"] == [TASK_VIEW]


def test_dataset_defaults_are_not_shared_mutable_lists():
    first = read_dataset_params({})
    second = read_dataset_params({})
    first["task_views"].append(TASK_VIEW)
    assert second["task_views"] == []


def test_explicit_scopes_are_written_to_train_section():
    scopes = ["understanding_vision", "understanding_decoder", "shared"]
    section = _build_train_section(
        {"sensenova_train_scopes": scopes},
        total_steps=10,
        epochs=None,
        train_unet=False,
        train_text_encoder=True,
        include_block_swap=False,
    )
    assert section["sensenova_train_scopes"] == scopes


def test_openapi_has_typed_dataset_task_contract():
    spec = yaml.safe_load((ROOT / "openapi.yaml").read_text(encoding="utf-8"))
    schemas = spec["components"]["schemas"]
    request = schemas["TrainingRunCreateRequest"]["properties"]
    assert request["dataset_configs"]["items"]["$ref"].endswith("/DatasetConfigItem")
    assert schemas["DatasetConfigItem"]["properties"]["task_views"]["items"]["$ref"].endswith("/SenseNovaTaskView")
    assert request["sensenova_train_scopes"]["default"] == []
    assert "generation_norms" in request["sensenova_train_scopes"]["items"]["enum"]
    assert request["sensenova_train_fm_modules"]["default"] is True
    assert request["sensenova_train_generation_norms"]["default"] is True


def test_frontend_exposes_task_views_and_scopes():
    api = (ROOT / "frontend/src/utils/api.ts").read_text(encoding="utf-8")
    panel = (ROOT / "frontend/src/components/training/TrainingConfig.tsx").read_text(encoding="utf-8")
    params = (ROOT / "frontend/src/components/training/trainingParams.ts").read_text(encoding="utf-8")
    assert "task_views?: SenseNovaTaskView[]" in api
    assert "sensenova_train_scopes?: SenseNovaTrainScope[]" in api
    assert "sensenova_train_generation_norms?: boolean" in api
    assert "SenseNova task views" in panel
    assert '"sensenova_train_scopes"' in params
    assert '"sensenova_train_generation_norms"' in params


def test_task_view_api_validation_is_strict():
    # routes imports the full API surface, so this focused source assertion
    # complements the mandatory real-import verification without duplicating it.
    source = (BACKEND / "api/routes.py").read_text(encoding="utf-8")
    assert 'task: Literal["t2i", "ti2i", "i2t_caption", "i2t_tags", "i2t_caption_tags"]' in source
    assert 'hint_dropout: float = Field(default=0.25, ge=0, le=1)' in source
