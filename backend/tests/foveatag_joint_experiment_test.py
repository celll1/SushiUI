"""CPU contracts for the gate-bound FoveaTag joint experiment."""

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn

from core.tagger.foveatag.artifact import FoveaArtifactConfig, save_inference_artifact
from core.tagger.foveatag.benchmark import (
    BenchmarkManifest,
    PredictionSet,
    evaluate_predictions,
)
from core.tagger.foveatag.encoder import FoveaEncoderConfig
from core.tagger.foveatag.joint_experiment import (
    _prepare_output_dir,
    _train_epoch,
    _validate_arguments,
    validate_joint_budget,
    validate_joint_sources,
)
from core.tagger.foveatag.model import FoveaTagger
from core.tagger.foveatag.phase3_comparison import build_phase3_comparison
from core.tagger.foveatag.phase3_gate import record_phase3_gate
from core.tagger.foveatag.router import EvidenceRouter
from core.tagger.foveatag.router_checkpoint import save_router_weights
from core.tagger.foveatag.router_data import FoveaRouterSample
from core.tagger.foveatag.joint_training import FoveaJointTrainResult
from core.tagger.foveatag.views import ViewConfig, build_views


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path.resolve()


def _artifacts(tmp_path, *, learned=0.7, random_baseline=0.4):
    encoder = FoveaEncoderConfig(
        widths=(8, 16, 32, 48),
        depths=(1, 1, 2, 1),
        window_size=4,
        attention_heads=(2, 3),
        mlp_ratio=2.0,
        scale_dropout=0.0,
    )
    config = FoveaArtifactConfig(
        num_tags=2,
        feature_dim=16,
        evidence_slots_per_view=1,
        pool_heads=2,
        encoder=encoder,
        views=ViewConfig(overview_size=32, local_size=16, local_views=2),
        routing_dim=16,
        routing_slots=2,
    )
    model = FoveaTagger(
        2,
        encoder_config=encoder,
        feature_dim=16,
        evidence_slots_per_view=1,
        pool_heads=2,
    )
    source = save_inference_artifact(
        tmp_path / "source.safetensors",
        model,
        config,
        vocabulary={
            "tag_to_idx": {"detail": 0, "scene": 1},
            "idx_to_tag": {"0": "detail", "1": "scene"},
        },
        provenance={
            "initialization": "scratch",
            "external_weights": [],
            "input_mode": "source_native",
        },
    )
    source_metadata = source.with_name("source_metadata.json")
    manifest_payload = {
        "format_version": 1,
        "name": "ground",
        "evidence_mode": "dense_oracle",
        "items": [
            {
                "sample_id": "positive",
                "image_path": "positive.png",
                "width": 32,
                "height": 32,
                "positive_tags": ["detail", "scene"],
                "fine_detail_tags": ["detail"],
                "coarse_tags": ["scene"],
                "evidence": [],
            },
            {
                "sample_id": "negative",
                "image_path": "negative.png",
                "width": 32,
                "height": 32,
                "positive_tags": [],
                "fine_detail_tags": [],
                "coarse_tags": [],
                "evidence": [],
            },
        ],
    }
    manifest = _write(tmp_path / "manifest.json", manifest_payload)
    manifest_value = BenchmarkManifest.from_dict(manifest_payload)
    predictions = {}
    prediction_payloads = {}
    for mode in ("overview_only", "overview_placebo", "source_native"):
        payload = {
            "format_version": 1,
            "model_name": f"foveatag-{mode}",
            "artifact_provenance": {
                "initialization": "scratch",
                "external_weights": [],
                "shared_initialization_sha256": "shared",
                "shared_data_contract_sha256": "d" * 64,
                "input_mode": mode,
                "threshold": 0.5,
            },
            "runtime": {
                "device": "cpu",
                "dtype": "float32",
                "prediction_rows": 2,
                "wall_seconds": 1.0,
                "peak_allocated_mib": None,
                "peak_reserved_mib": None,
            },
            "checkpoint_sha256": _sha256(source) if mode == "source_native" else "a" * 64,
            "checkpoint_metadata_sha256": (
                _sha256(source_metadata) if mode == "source_native" else "b" * 64
            ),
            "predictions": [
                {
                    "sample_id": "positive",
                    "resolution": "native",
                    "scores": {"detail": 0.9, "scene": 0.9},
                    "stage_ms": {"forward": 1.0},
                },
                {
                    "sample_id": "negative",
                    "resolution": "native",
                    "scores": {"detail": 0.1, "scene": 0.1},
                    "stage_ms": {"forward": 1.0},
                },
            ],
        }
        prediction_payloads[mode] = payload
        predictions[mode] = str(_write(tmp_path / f"{mode}.json", payload))
    baseline_payload = evaluate_predictions(
        manifest_value,
        PredictionSet.from_dict({
            "format_version": 1,
            "model_name": "fixed-384",
            "predictions": prediction_payloads["overview_only"]["predictions"],
        }),
    )
    baseline = _write(tmp_path / "baseline.json", baseline_payload)
    report = _write(
        tmp_path / "report.json",
        build_phase3_comparison(manifest_value, prediction_payloads, baseline_payload),
    )
    evaluation = _write(tmp_path / "evaluation.json", {
        "format_version": 1,
        "kind": "foveatag_phase3_evaluation",
        "status": "complete",
        "active_benchmark": None,
        "completed_benchmarks": ["ground"],
        "benchmarks": {
            "ground": {
                "manifest": str(manifest),
                "fixed_baseline_report": str(baseline),
                "comparison_report": str(report),
                "predictions": predictions,
            }
        },
    })
    gate = record_phase3_gate(
        evaluation,
        tmp_path / "gate.json",
        decision="pass",
        rationale="Registered evidence passed.",
        decided_by="repository owner",
    )
    router = EvidenceRouter(feature_dim=32, routing_dim=16, routing_slots=2)
    contract = {
        "phase3_gate_sha256": _sha256(gate),
        "source_native_model_sha256": _sha256(source),
        "source_native_metadata_sha256": _sha256(source_metadata),
        "candidate_count": 4,
        "selection_budget": 2,
    }
    router_path = save_router_weights(
        tmp_path / "router.safetensors",
        router,
        run_contract=contract,
        epoch=0,
        global_step=1,
        validation={
            "budget": 2,
            "candidates_per_sample": 4,
            "mean_utility_recovery": learned,
            "random_mean_utility_recovery": random_baseline,
        },
    )
    return gate, source, router_path


def test_joint_sources_require_matching_gate_model_and_router_with_random_gain(tmp_path):
    gate, source, router = _artifacts(tmp_path)

    loaded_gate, metadata, config, router_metadata = validate_joint_sources(
        gate, source, router
    )

    assert loaded_gate["decision"] == "pass"
    assert metadata["provenance"]["input_mode"] == "source_native"
    assert config.num_tags == 2
    assert router_metadata["validation"]["mean_utility_recovery"] == 0.7


def test_joint_sources_reject_router_that_did_not_beat_random(tmp_path):
    gate, source, router = _artifacts(tmp_path, learned=0.4, random_baseline=0.4)

    with pytest.raises(ValueError, match="did not beat random"):
        validate_joint_sources(gate, source, router)


def test_joint_sources_reject_router_from_another_source_model(tmp_path):
    gate, source, router = _artifacts(tmp_path)
    metadata_path = router.with_name("router_metadata.json")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["run_contract"]["source_native_model_sha256"] = "c" * 64
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match"):
        validate_joint_sources(gate, source, router)


def test_joint_budget_must_match_the_router_acceptance_measurement(tmp_path):
    gate, source, router = _artifacts(tmp_path)
    *_, router_metadata = validate_joint_sources(gate, source, router)

    validate_joint_budget(router_metadata, candidate_count=4, selection_budget=2)
    with pytest.raises(ValueError, match="selection_budget"):
        validate_joint_budget(router_metadata, candidate_count=4, selection_budget=3)
    with pytest.raises(ValueError, match="candidate_count"):
        validate_joint_budget(router_metadata, candidate_count=8, selection_budget=2)


def test_joint_output_refuses_stale_artifacts_without_explicit_resume(tmp_path):
    (tmp_path / "best_joint.safetensors").write_bytes(b"old")

    with pytest.raises(ValueError, match="use --resume"):
        _prepare_output_dir(tmp_path, resume=None)

    _prepare_output_dir(tmp_path, resume=tmp_path / "latest.pt")


@pytest.mark.parametrize("random_count", [0, 2, 3])
def test_joint_arguments_keep_random_coverage_within_fixed_budget(random_count):
    values = {
        "train_counts": {25: 10}, "validation_counts": {25: 2},
        "domain_weights": {25: 1.0}, "epochs": 1, "epoch_samples": 10,
        "batch_size": 1, "workers": 0, "candidate_count": 4,
        "selection_budget": 2, "random_count": random_count, "oracle_interval": 4,
        "oracle_chunk_size": 2, "learning_rate": 1e-4, "weight_decay": 1e-4,
        "warmup_steps": 0, "checkpoint_every_steps": 1, "log_every": 1,
        "photo_loss_weight": 0.25, "max_source_pixels": 40_000_000,
        "router_weight": 1.0, "diversity_weight": 0.05,
        "stop_after_steps": None,
    }

    with pytest.raises(ValueError, match="random_count"):
        _validate_arguments(**values)


class _Samples:
    def __init__(self, count):
        self.values = []
        config = ViewConfig(
            overview_size=16,
            local_size=8,
            local_views=2,
            field_levels=(0.5, 1.0),
            scale_jitter=(1.0, 1.0),
        )
        for index in range(count):
            views = build_views(
                Image.new("RGB", (16, 16), (index, 2, 3)),
                config,
                seed=index,
                training=False,
            )
            boxes = torch.tensor([
                [
                    spec.source_box.center[0] / 16,
                    spec.source_box.center[1] / 16,
                    spec.source_box.width / 16,
                    spec.source_box.height / 16,
                ]
                for spec in views.local_specs
            ])
            self.values.append(FoveaRouterSample(
                views=views,
                candidate_boxes=boxes,
                label=torch.tensor([1.0, 0.0]),
                loss_mask=torch.ones(2),
            ))

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]


class _Trainer:
    def __init__(self):
        self.model = nn.Linear(2, 2)
        self.router = nn.Linear(2, 2)
        self.optimizer = torch.optim.AdamW(
            [*self.model.parameters(), *self.router.parameters()], lr=0.1
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=0.5
        )

    def train_batch(self, batch, *, global_step, random_seed):
        self.optimizer.zero_grad(set_to_none=True)
        loss = (
            self.model(batch.labels).square().mean()
            + self.router(batch.labels).square().mean()
        )
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return FoveaJointTrainResult(
            loss=float(loss.detach()),
            classification=0.3,
            router_ranking=0.2,
            router_diversity=0.1,
            oracle_used=global_step % 2 == 0,
            selected_views=int(batch.labels.shape[0]) * 2,
        )


def _run_epoch(trainer, dataset, tmp_path, **overrides):
    arguments = {
        "epoch": 0, "sample_offset": 0, "batch_size": 2, "workers": 0,
        "checkpoint_every_steps": 1, "log_every": 10, "global_step": 0,
        "output_dir": tmp_path, "best_f1": 0.2, "best_threshold": 0.5,
        "run_contract": {"lineage": "fixed"}, "history": [],
        "initial_accumulator": None, "remaining_step_budget": 2, "seed": 7,
        "progress_callback": None, "run_control": None,
    }
    arguments.update(overrides)
    return _train_epoch(trainer, dataset, **arguments)


def test_joint_epoch_stops_and_checkpoints_at_exact_sample_offset(tmp_path):
    result, offset, step, added, stopped, accumulator = _run_epoch(
        _Trainer(), _Samples(5), tmp_path
    )

    assert stopped == "stopped_after_requested_steps"
    assert (offset, step, added) == (4, 2, 2)
    assert result["samples"] == 4
    assert result["optimizer_steps"] == 2
    assert result["oracle_steps"] == 1
    assert accumulator["selected_views"] == 8
    payload = torch.load(tmp_path / "latest.pt", map_location="cpu", weights_only=False)
    assert (payload["sample_offset"], payload["global_step"]) == (4, 2)
    assert payload["epoch_accumulator"]["samples"] == 4


def test_joint_epoch_honors_stop_before_loading_a_batch(tmp_path):
    trainer = _Trainer()
    control = SimpleNamespace(should_stop=lambda: True)

    result, offset, step, added, stopped, _ = _run_epoch(
        trainer,
        _Samples(3),
        tmp_path,
        run_control=control,
        remaining_step_budget=None,
    )

    assert stopped == "stopped_by_request"
    assert (offset, step, added) == (0, 0, 0)
    assert result["samples"] == 0
    assert (tmp_path / "latest.pt").is_file()
