"""The Training list must not eagerly load every detail/editor bundle."""

from pathlib import Path


SOURCE = (
    Path(__file__).resolve().parents[2]
    / "frontend" / "src" / "app" / "training" / "page.tsx"
).read_text(encoding="utf-8")


def test_heavy_training_panels_are_dynamic_imports():
    modules = (
        "@/components/training/TrainingConfig",
        "@/components/training/TrainingMonitor",
        "@/components/training/tagger/TaggerTrainingConfig",
        "@/components/training/tagger/TaggerTrainingMonitor",
        "@/components/training/vae/VaeTrainingConfig",
    )
    for module in modules:
        assert f'dynamic(() => import("{module}"))' in SOURCE
        assert f'import {module.rsplit("/", 1)[-1]} from "{module}"' not in SOURCE
