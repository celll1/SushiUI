"""Real Anima region backward composes U-Net, global context and target-native VAE IO."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.cuda.get_device_capability = lambda *a, **k: (8, 9)
torch.cuda._lazy_init = lambda *a, **k: None
torch._C._cuda_init = lambda *a, **k: None
torch.set_num_threads(1)

from core.models.anima.anima_models import Anima, LLMAdapter
from core.models.anima.partition_training import training_plan, backward_regions
from core.models.anima.partition_global import configure_training, NAME
from core.models.anima.residual_unet import restore_topology, native_config
from core.models.anima.checkpointing import configure
from core.rewiring.spec import build_spec, OPERATORS


@pytest.mark.parametrize("swap", [False, True])
def test_real_nested_unet_global_backward_and_stitched_debug(tmp_path, swap):
    torch.manual_seed(113)
    channels = 64 if swap else 4
    model = Anima(max_img_h=64, max_img_w=64, max_frames=1,
        in_channels=channels, out_channels=channels, model_channels=24, num_blocks=7,
        num_heads=2, mlp_ratio=1., crossattn_emb_channels=12,
        use_adaln_lora=True, adaln_lora_dim=8, use_llm_adapter=False)
    model.llm_adapter = LLMAdapter(12, 12, 12, num_layers=1, num_heads=2, self_attn=True)
    model.use_llm_adapter = True
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim >= 2:
                parameter.normal_(0, .02)
    if swap:
        from core.models.components.spatial_swap import install
        from backend.tests.latent_adapter_contract_test import declaration, identity
        install(model, declaration(16, 64, scale=16))
        identity(model.vae_latent_bridge.input, .2)
        identity(model.vae_latent_bridge.output, .3)
    spec = build_spec("anima", [{"name": "main", "block_count": 7}], [
        dict(id="outer", stream="main", parent=None, start_block=1, end_block=6,
             down_factor=[2, 2], **OPERATORS),
        dict(id="inner", stream="main", parent="outer", start_block=2, end_block=5,
             down_factor=[2, 2], **OPERATORS)], native_config(model))
    restore_topology(model, spec)
    config = dict(dit_partition_training_enabled=True, dit_partition_mode="fixed",
        qwen_debug_latent_view="pixel",
        dit_partition_fixed_count=4, dit_partition_halo_tokens=2,
        dit_partition_global_adapter_enabled=True, dit_partition_global_rank=4,
        dit_partition_global_tokens=2, dit_gradient_checkpointing_blocks=3,
        dit_partition_gradient_checkpointing_blocks=5)
    adapter = configure_training(model, config)
    with torch.no_grad():
        adapter.out.weight.normal_(0, .02)
    model.train().enable_gradient_checkpointing()
    configure(model, config)
    owner = SimpleNamespace(config=config, transformer=model, device=torch.device("cpu"),
        training_dtype=torch.float32, reconstruction_loss_weight=.2, mixed_precision=False,
        _collect_concept_sample_loss=True, _current_epoch=2, _current_batch_position=5)
    metrics = {}
    owner.log_extra_metric = lambda name, value: metrics.__setitem__(name, value)
    clean = torch.randn(2, channels, 32, 48)
    noise, sigma = torch.randn_like(clean), torch.tensor([.3, .7])
    text = torch.randn(2, 3, 12, requires_grad=True)
    aux = {name: torch.ones(2, 3, dtype=torch.long)
           for name in ("source_mask", "t5_input_ids", "t5_attn_mask")}
    seen = []
    handle = adapter.register_forward_pre_hook(lambda module, args: seen.append((args[0].shape, args[1])))
    try:
        result = backward_regions(owner, clean=clean, noise=noise, sigma=sigma,
            prompt_embeds=text, anima_aux=aux, plan=training_plan(owner, clean),
            backward_scale=.5, debug_save_path=tmp_path, debug_captions=["general synthetic", "second"])
    finally:
        handle.remove()
    assert all(torch.isfinite(torch.tensor(value)) for value in result)
    assert text.grad is not None and torch.isfinite(text.grad).all() and text.grad.abs().sum() > 0
    assert len(seen) == 4 and all(shape[-2:] == (32, 48) for shape, _ in seen)
    for parameter in adapter.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
               for name, p in model.named_parameters() if not name.startswith(NAME))
    assert [block.gradient_checkpoint_mode for block in model.blocks] == ["standard"]*3 + ["none"]*4
    data = torch.load(tmp_path / "latents_t0.3000.pt", weights_only=True)
    assert data["predicted_velocity"].shape == (1, channels, 32, 48)
    assert data["partition_count"] == 4 and data["caption"] == "general synthetic"
    torch.testing.assert_close(data["predicted_latent"], data["noisy_latents"] - .3*data["predicted_velocity"])
    assert torch.isfinite(data["predicted_latent"]).all()
    pending = owner._pending_anima_debug_previews
    assert [name for name, _ in pending["previews"]] == ["target", "noisy", "pred_x0"]
    assert all(latent.device.type == "cpu" and not latent.requires_grad
               for _, latent in pending["previews"])
    assert owner._pending_concept_sample_loss.shape == (2,)
    assert metrics["anima_partition_count"] == 4
    assert metrics["anima_partition_checkpoint_blocks"] == 5
    assert metrics["anima_partition_wall_ms"] > 0
    assert "anima_partition_forward_ms" not in metrics
