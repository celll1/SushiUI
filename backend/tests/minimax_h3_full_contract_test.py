"""Aligned H3 Full ownership and pre-load checkpoint refusals."""

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from core.models.minimax_h3.full_training_contract import preflight_full_training


@pytest.mark.parametrize("field,value,message", [
    ("train_unet", False, "train_unet=true"),
    ("train_text_encoder", True, "train_text_encoder=false"),
    ("train_image_encoder", True, "train_image_encoder=false"),
    ("bundle_vae", False, "bundled target codec"),
    ("vae_swap_new_channel_init", "zero", "paired_spatial"),
])
def test_invalid_ownership_refuses_before_checkpoint_access(field, value, message):
    with pytest.raises(ValueError, match=message):
        preflight_full_training({field: value}, "missing.safetensors")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_dense_checkpoint_requires_a_swap_and_reads_headers_only(tmp_path, dtype, monkeypatch):
    path = tmp_path / "base.safetensors"
    save_file({"blocks.0.attn.qkv_proj.weight": torch.ones(6, 2, dtype=dtype)}, str(path))
    with pytest.raises(ValueError, match="vae_swap_source"):
        preflight_full_training({}, str(path))
    from core.models.minimax_h3 import video_artifact
    def forbidden(*args, **kwargs):
        raise AssertionError("Full preflight must not load checkpoint weights")
    monkeypatch.setattr(video_artifact, "load_checkpoint", forbidden)
    preflight_full_training({"vae_swap_source": "registry:ltx2"}, str(path))


@pytest.mark.parametrize("kind", ["int8", "fp8", "marker", "packed"])
def test_quantized_checkpoint_refuses_before_model_load(tmp_path, kind):
    state = {"blocks.0.attn.qkv_proj.weight": torch.ones(6, 2)}
    if kind in ("int8", "fp8"):
        state[next(iter(state))] = state[next(iter(state))].to(
            torch.int8 if kind == "int8" else torch.float8_e4m3fn)
    elif kind == "marker":
        state["blocks.0.attn.qkv_proj.comfy_quant"] = torch.ones(2, dtype=torch.uint8)
    else:
        state["blocks.0.attn.qkv_proj.qweight"] = torch.ones(2, dtype=torch.int32)
    path = tmp_path / "base.safetensors"
    save_file(state, str(path))
    with pytest.raises(ValueError, match="unquantized"):
        preflight_full_training({"vae_swap_source": "registry:ltx2"}, str(path))


def test_saved_alignment_uses_strict_native_preflight(tmp_path):
    from minimax_h3_video_artifact_test import fixture
    from core.models.minimax_h3.video_artifact import save_checkpoint
    trainer = fixture(tmp_path, 4)
    path = save_checkpoint(trainer, tmp_path / "full.safetensors", step=0, epoch=0)
    preflight_full_training({}, str(path))
    with pytest.raises(ValueError, match="bundled target codec"):
        preflight_full_training({}, str(path), bundle_vae_explicit_false=True)


def test_full_factory_selects_aligned_adapter_without_loading_weights():
    from core.training.full_parameter_trainer import FullParameterTrainer
    from core.training.adapters.minimax_h3_adapter import MiniMaxH3AlignedFullParameterAdapter
    names = ("ming_image_01", "zimage", "flux2", "anima", "lens", "ideogram4", "minit2i",
             "krea2", "qwen_image_21", "ltx2", "minimax_h3")
    trainer = SimpleNamespace(**{"is_" + name: name == "minimax_h3" for name in names},
                              log_prefix="[test]")
    FullParameterTrainer._create_adapter(trainer)
    assert isinstance(trainer.adapter, MiniMaxH3AlignedFullParameterAdapter)


def test_adapter_rejects_unsupported_encoder_ownership():
    from core.training.adapters.minimax_h3_adapter import MiniMaxH3AlignedFullParameterAdapter
    adapter = MiniMaxH3AlignedFullParameterAdapter(SimpleNamespace(train_text_encoder=True))
    with pytest.raises(ValueError, match="train_text_encoder=false"):
        adapter.prepare_models_for_training()


def test_unreadable_fresh_checkpoint_reports_a_config_error(tmp_path):
    with pytest.raises(ValueError, match="Cannot inspect H3 Full checkpoint"):
        preflight_full_training({"vae_swap_source": "registry:ltx2"}, str(tmp_path / "missing.safetensors"))
