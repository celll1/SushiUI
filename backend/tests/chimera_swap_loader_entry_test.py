"""The production loader must install a requested swap before optimizer setup."""
from types import SimpleNamespace

import torch


def test_fresh_load_components_installs_configured_codec(tmp_path, monkeypatch):
    from diffusers import AutoencoderKL
    from sensenova_sdxl_chimera_p1_test import _build
    from core.models.sensenova_sdxl_chimera.loader import load_chimera_artifact
    from core.training.ops.sensenova_sdxl_chimera_ops import load_components
    from core.training.adapters.sensenova_sdxl_chimera_adapter import SenseNovaSDXLChimeraFullParameterAdapter
    from core.models.components.wiring import SENSENOVA_SDXL_CHIMERA_WIRING

    _, _, _, artifact = _build(tmp_path, initialization="scratch")
    components = load_chimera_artifact(artifact["directory"])
    components["understanding"] = dict(transformer=torch.nn.Linear(6,6),tokenizer=None)
    target = AutoencoderKL(in_channels=4, out_channels=4, latent_channels=6,
        block_out_channels=(4,)*5, down_block_types=("DownEncoderBlock2D",)*5,
        up_block_types=("UpDecoderBlock2D",)*5, norm_num_groups=4, layers_per_block=1)
    folder = tmp_path/"target"
    target.save_pretrained(folder)
    monkeypatch.setattr("core.models.sensenova_sdxl_chimera.loader.load_chimera_artifact",lambda *a,**k:components)
    trainer = SimpleNamespace(model_path=artifact["directory"],weight_dtype=torch.float32,
        vae_dtype=torch.float32,gradient_checkpointing=False,log_prefix="test",device="cpu",
        learning_rate=1e-4,unet_lr=1e-4,
        is_sensenova_sdxl_chimera=True,training_method="full_finetune",bundle_vae=True,
        wiring=SENSENOVA_SDXL_CHIMERA_WIRING,config=dict(training_method="full_finetune",
            chimera_training_stage="unet",chimera_allow_unaligned_scratch=True,
            vae_swap_source=f"file:{folder}",vae_swap_new_channel_init="paired_spatial"))
    load_components(trainer)
    pair = trainer.unet.vae_latent_bridge
    assert pair.declaration["new_channels"] == 6
    assert pair.declaration["new_scale"] == 16
    assert pair.declaration["input_channels"] == 4
    adapter = SenseNovaSDXLChimeraFullParameterAdapter(trainer)
    adapter.prepare_models_for_training()
    optimizer_ids = {id(p) for group in adapter.arch_param_groups() for p in group["params"]}
    assert all(id(p) in optimizer_ids and p.requires_grad for p in pair.parameters())
