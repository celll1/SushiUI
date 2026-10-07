"""Plain dual-DiT ownership and full-artifact CPU contracts."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from core.models.ideogram4.vendor.transformer import Ideogram4Transformer2DModel
from core.models.ideogram4.ideogram4_pipeline_ops import _prepare_ids
from core.training.adapters.ideogram4_adapter import Ideogram4FullParameterAdapter


def _trainer(both=False):
    def model():
        return Ideogram4Transformer2DModel(in_channels=8,num_layers=2,attention_head_dim=12,
            num_attention_heads=2,intermediate_size=32,adaln_dim=8,llm_features_dim=12,mrope_section=(2,2,2))
    return SimpleNamespace(transformer=model(),transformer_uncond=model(),vae=nn.Linear(3,3),
        text_encoder=nn.Linear(3,3),train_unet=True,train_text_encoder=False,
        ideogram4_train_uncond=both,unet_lr=2e-4,config={"ideogram4_lr_factor":.5},
        ideogram4_base_dir="synthetic-companion")


@pytest.mark.parametrize("both", [False,True])
def test_full_optimizer_updates_only_selected_native_branches_and_restores_both(tmp_path,both):
    from core.models.common.single_file_format import read_state_dict,strip_prefix
    from core.models.ideogram4.ideogram4_loader import _build_ideogram4_transformer_from_state
    import json
    torch.manual_seed(70)
    trainer=_trainer(both)
    adapter=Ideogram4FullParameterAdapter(trainer)
    adapter.prepare_models_for_training()
    groups=adapter.arch_param_groups()
    assert groups[0]["lr"] == 1e-4
    assert not any(p.requires_grad for module in (trainer.vae,trainer.text_encoder) for p in module.parameters())
    old={name:{k:v.clone() for k,v in model.state_dict().items()}
        for name,model in (("transformer",trainer.transformer),("unconditional_transformer",trainer.transformer_uncond))}
    optimizer=torch.optim.AdamW(groups)
    positions,segments,indicator=_prepare_ids([2],2,2,2,torch.device("cpu"))
    image=torch.randn(1,6,8)
    features=torch.randn(1,6,12)
    loss=trainer.transformer(hidden_states=image,timestep=torch.tensor([.3]),encoder_hidden_states=features,
        position_ids=positions,segment_ids=segments,indicator=indicator,return_dict=False)[0].square().mean()
    if both:
        loss=loss+trainer.transformer_uncond(hidden_states=image[:,2:],timestep=torch.tensor([.7]),
            encoder_hidden_states=features[:,2:],position_ids=positions[:,2:],segment_ids=segments[:,2:],
            indicator=indicator[:,2:],return_dict=False)[0].square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in groups[0]["params"])
    optimizer.step()
    for name,model in (("transformer",trainer.transformer),("unconditional_transformer",trainer.transformer_uncond)):
        changed=any(not torch.equal(v,old[name][k]) for k,v in model.state_dict().items())
        assert changed == (name=="transformer" or both)
    saved=adapter.write_checkpoint(1,0,tmp_path/"full.safetensors")
    state,metadata=read_state_dict(str(saved))
    assert metadata["component.base_dir"]=="synthetic-companion"
    for name,model in (("transformer",trainer.transformer),("unconditional_transformer",trainer.transformer_uncond)):
        restored=_build_ideogram4_transformer_from_state(json.loads(metadata[name+"_config"]),
            strip_prefix(state,name+"."),torch.float32,name)
        for key,value in model.state_dict().items():
            assert torch.equal(value,restored.state_dict()[key])


def test_full_quantized_branch_refusal_happens_before_any_ownership_mutation():
    from core.models.ideogram4.vendor.fp8_linear import Fp8Linear
    trainer=_trainer(True)
    trainer.transformer_uncond.input_proj=Fp8Linear(8,24,bias=True,compute_dtype=torch.float32)
    before=[p.requires_grad for p in trainer.transformer.parameters()]
    adapter=Ideogram4FullParameterAdapter(trainer)
    for operation in (adapter.prepare_models_for_training,adapter.arch_param_groups):
        with pytest.raises(NotImplementedError,match="unconditional DiT"):
            operation()
    assert before==[p.requires_grad for p in trainer.transformer.parameters()]


def test_full_dispatch_selects_ideogram_adapter_instead_of_sd_fallthrough():
    from core.training.full_parameter_trainer import FullParameterTrainer
    trainer=_trainer()
    for arch in ("ming_image_01","zimage","flux2","anima","lens"):
        setattr(trainer,"is_"+arch,False)
    trainer.is_ideogram4=True
    trainer.log_prefix="[CPU]"
    FullParameterTrainer._create_adapter(trainer)
    assert isinstance(trainer.adapter,Ideogram4FullParameterAdapter)


@pytest.mark.parametrize("suffix", [".safetensors", ".safetensors.index.json"])
def test_checkpoint_resume_routes_both_layouts_to_native_loader(monkeypatch,suffix):
    from core.model_loader import ModelLoader
    from core.training.base_trainer import BaseTrainer
    from core.training.ops import ideogram4_ops
    path = "saved_ideogram"+suffix
    monkeypatch.setattr(ModelLoader,"detect_model_type",lambda candidate:"ideogram4")
    seen = []
    monkeypatch.setattr(ideogram4_ops,"load_components",lambda trainer:seen.append(trainer.model_path))
    trainer = SimpleNamespace(model_path="original_companion")
    BaseTrainer._load_checkpoint_as_base(trainer,path)
    assert seen == [path]
    assert trainer.is_ideogram4 and not trainer.is_sdxl
