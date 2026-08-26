"""Source-level guards for the zero-hot-loop-cost training speed monitor."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT / "frontend/src/components/training/TrainingMonitor.tsx").read_text(
    encoding="utf-8"
)


def test_speed_is_derived_from_status_polls_not_trainer_instrumentation():
    assert "performance.now()" in SOURCE
    assert "status.current_step - first.step" in SOURCE
    assert "20_000" in SOURCE
    assert "getTrainingRun(currentRun.id)" in SOURCE


def test_monitor_explains_iteration_batch_and_optimizer_cadence():
    assert "One iteration is one forward/backward at one MNT timestep" in SOURCE
    assert 'yamlInt(yaml, "multi_noise_timesteps", 1)' in SOURCE
    assert 'yamlInt(yaml, "gradient_accumulation_steps", 1)' in SOURCE
    assert "fused ? 1 : accumulation" in SOURCE
    assert "Input-batch wall:" in SOURCE
    assert "Wall/update:" in SOURCE
    assert "Sample passes/update:" in SOURCE
    assert "Eviction included" in SOURCE
