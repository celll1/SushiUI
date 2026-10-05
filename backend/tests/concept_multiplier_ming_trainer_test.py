"""Exposure rebatching through native conditioning, MNT, GAS and resume."""

import json
import shutil
from types import SimpleNamespace

from PIL import Image
import pytest
import torch

from ming_image_01_conditioning_test import write_tiny_package
from core.training.lora_trainer import LoRATrainer


@pytest.mark.parametrize("mode,randomized", [("swap_onthefly", False), ("cpu_prefetch", True)])
def test_real_ming_multiplier_resume_matches_continuous(tmp_path, monkeypatch, mode, randomized):
    from core.training import latent_cache
    from core.training.ops import ming_image_01_ops
    monkeypatch.setattr(latent_cache, "get_cache_base_dir", lambda: str(tmp_path / "cache"))
    root = tmp_path / "package"
    root.mkdir()
    write_tiny_package(root)
    caption_config = {"shuffle_tokens": True, "shuffle_per_epoch": True, "normalize_tags": False}
    items = []
    for index in range(4):
        path = tmp_path / f"image_{index}.png"
        Image.new("RGBA", (64, 64), (80 + index * 20, 60, 40, 255)).save(path)
        items.append({"image_path": str(path), "caption": "word5, word6, word7", "raw_caption": "word5, word6, word7",
            "width": 64, "height": 64, "is_tags_format": True, "_caption_processing_config": caption_config,
            "tag_data": [{"tag": "word5", "category": "Character"},
                         {"tag": "word6", "category": "General"}, {"tag": "word7", "category": "General"}]})
    ds = SimpleNamespace(unique_id="multiplier-ming", dataset_root=tmp_path, items=items,
                         caption_config=caption_config, reload_for_epoch=lambda **kw: None)
    concept = {"enabled": True, "min_items_per_concept": 1, "focus_batches": 2,
               "local_swap_window": 0, "batch_multiplier": 2,
               "batch_multiplier_mode": "uniform_int" if randomized else "fixed",
               "batch_multiplier_min": 2, "batch_multiplier_max": 3}
    config = {"ming_conditioner_groups": [], "seed": 17, "cfg_uncond_drop_rate": 0.5,
              "latent_encoding_mode": "swap_onthefly", "text_encoding_mode": mode,
              "concept_batch_order": concept}
    def trainer(folder, resume=None):
        return LoRATrainer(model_path=str(root), output_dir=str(folder), lora_rank=2, lora_alpha=2,
            train_text_encoder=False, device="cpu", weight_dtype="fp32", training_dtype="fp32",
            vae_dtype="fp32", mixed_precision=False, attention_backend="native",
            train_config=config.copy(), resume_from_checkpoint=resume)
    def train(owner, resume=None):
        owner.train([ds], num_epochs=1, total_steps=8, batch_size=2, multi_noise_timesteps=2,
            gradient_accumulation_steps=3, save_every_n_steps=2, sample_every_n_steps=0,
            max_step_saves_to_keep=4, max_optimizer_saves_to_keep=4, enable_bucketing=False,
            base_resolutions=[64], latent_encoding_mode="swap_onthefly", text_encoding_mode=mode,
            latent_encoding_swap_interval=2, text_encoding_swap_interval=2,
            concept_batch_order=concept, resume_dataset_change_policy="rebase_remaining",
            resume_from_checkpoint=resume)
    traces = []
    original = ming_image_01_ops.flow_loss
    def record(transformer, **kw):
        traces.append(tuple(kw[key].detach().clone() for key in ("clean", "noise", "sigma")) +
                      (kw["conditioning"].queries.detach().clone(), kw["conditioning"].direct.detach().clone()))
        return original(transformer, **kw)
    monkeypatch.setattr(ming_image_01_ops, "flow_loss", record)
    baseline = trainer(tmp_path / "baseline" / "run")
    train(baseline)
    expected = traces.copy()
    state = json.loads((baseline.output_dir / "run_step_000002_state.json").read_text())
    assert state["concept_batch_plan"]["version"] == 2
    assert state["resume_batch_plan"]["version"] == 2
    destination = tmp_path / "resumed" / "run"
    destination.mkdir(parents=True)
    for path in [*baseline.output_dir.glob("run_step_000002*"), *baseline.output_dir.glob("run_epoch_*batch_plan*.json.gz")]:
        if path.is_dir():
            shutil.copytree(path, destination / path.name)
        else:
            shutil.copy2(path, destination / path.name)
    traces.clear()
    resumed = trainer(destination, "latest")
    train(resumed, "latest")
    assert len(traces) == 6
    assert all(all(torch.equal(a, b) for a, b in zip(wanted, actual))
               for wanted, actual in zip(expected[2:], traces))
    assert any(row[3].count_nonzero() == 0 or (row[3] == 0).all(dim=-1).any() for row in expected)
    for component in ("transformer", "text_encoder"):
        wanted = getattr(baseline, component).state_dict()
        actual = getattr(resumed, component).state_dict()
        assert all(torch.equal(value.view(torch.uint8), actual[key].view(torch.uint8)) for key, value in wanted.items())
