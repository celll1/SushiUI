"""Native BF16 admission, frozen-half verification, and lossless export."""
import pytest
import torch
from torch import nn
from safetensors.torch import save_file

from backend.tests.sensenova_full_finetune_resume_base_test import _MoTTree, _float
from backend.tests.sensenova_full_finetune_save_test import _Decoder, _save, _load_back
from core.training.ops.sensenova_ops import _assert_supported_quantized_training_base
from core.models.sensenova.loader import restore_sensenova_frozen_half_from_base
from core.models.sensenova.sensenova_lora import iter_sensenova_lora_targets


@pytest.mark.parametrize('method', ['lora', 'full'])
def test_complete_native_bf16_admitted_and_incomplete_or_wrong_dtype_refused(method):
    model = _MoTTree(gen_factory=_float, und_factory=_float)
    assert _assert_supported_quantized_training_base(model, training_method=method) == 'bf16'
    model.language_model.model.layers[-1].self_attn.q_proj.weight.data = (
        model.language_model.model.layers[-1].self_attn.q_proj.weight.float())
    with pytest.raises(RuntimeError):
        _assert_supported_quantized_training_base(model, training_method=method)
    with pytest.raises(RuntimeError):
        _assert_supported_quantized_training_base(
            _MoTTree(gen_factory=_float, und_factory=_float, layers=41), training_method=method)


@pytest.mark.parametrize('prefixed', [False, True])
def test_native_frozen_half_is_verified_without_quantization(tmp_path, prefixed):
    model = _MoTTree(gen_factory=_float, und_factory=_float)
    prefix = 'transformer.' if prefixed else ''
    state = {prefix+key:value.clone() for key,value in model.state_dict().items()}
    source = tmp_path/'base.safetensors'
    save_file(state, str(source))
    targets = list(iter_sensenova_lora_targets(model, branch='und'))
    modules = [module for _,_,_,module in targets]
    assert restore_sensenova_frozen_half_from_base(model, frozen_half='und',
        base_path=str(source), compute_dtype=torch.bfloat16) == 294
    assert all(current is previous for (_,_,_,current),previous in zip(
        iter_sensenova_lora_targets(model,branch='und'), modules))
    key = prefix + targets[0][0] + '.weight'
    state[key] += 1
    save_file(state, str(source))
    with pytest.raises(RuntimeError, match='differs from native BF16 base'):
        restore_sensenova_frozen_half_from_base(model,frozen_half='und',
            base_path=str(source),compute_dtype=torch.bfloat16)


def test_native_bf16_scale_marker_is_refused(tmp_path):
    model = _MoTTree(gen_factory=_float, und_factory=_float)
    state = dict(model.state_dict())
    key = next(iter_sensenova_lora_targets(model, branch='und'))[0]
    state[key+'.weight_scale'] = torch.ones(8)
    source=tmp_path/'stale.safetensors';save_file(state,str(source))
    with pytest.raises(RuntimeError,match='quantization scales'):
        restore_sensenova_frozen_half_from_base(model,frozen_half='und',
            base_path=str(source),compute_dtype=torch.bfloat16)


@pytest.mark.parametrize('branch', ['gen', 'und', 'both'])
def test_mixed_export_of_native_bf16_is_lossless_bf16(tmp_path, branch):
    model = _Decoder(factory=lambda seed: nn.Linear(8,4,bias=False).to(torch.bfloat16))
    before = {key:value.clone() for key,value in model.state_dict().items()}
    path, census = _save(model,tmp_path,branch,'mixed')
    restored, metadata = _load_back(path)
    assert census['effective_format']=='bf16' and metadata['sensenova_save_format']=='bf16'
    for key,value in before.items():
        assert torch.equal(restored.state_dict()[key],value), key


@pytest.mark.parametrize('branch', ['gen', 'und', 'both'])
def test_explicit_int8_export_quantizes_both_native_halves_without_mutating_base(tmp_path, branch):
    from core.models.ideogram4.vendor.int8_linear import Int8Linear, quantize_weight_to_int8
    model = _Decoder(factory=lambda seed: nn.Linear(8,4,bias=False).to(torch.bfloat16))
    before = {key:value.clone() for key,value in model.state_dict().items()}
    path, census = _save(model,tmp_path,branch,'int8')
    restored, metadata = _load_back(path)
    assert census['effective_format']=='int8' and metadata['sensenova_save_format']=='int8'
    assert census['trained_int8']+census['frozen_int8']==588
    for key,value in before.items():
        assert torch.equal(model.state_dict()[key],value), key
    for name,_,_,module in iter_sensenova_lora_targets(restored,branch='both'):
        assert type(module) is Int8Linear
        codes, scale=quantize_weight_to_int8(before[name+'.weight'])
        assert torch.equal(module.weight,codes)
        assert torch.equal(module.weight_scale,scale)


def test_native_unprefixed_load_requires_all_state_and_inspects_sibling_config(tmp_path, monkeypatch):
    import json
    from backend.tests.dit_rewiring_sensenova_test import SenseNovaRuntimeContracts
    from core.models.sensenova import loader
    from core.models.sensenova.vendor.configuration_neo_chat import NEOChatConfig
    from core.models.sensenova.vendor.modeling_neo_chat import NEOChatModel
    from core.models.sensenova.residual_unet import inspect_artifact
    raw=SenseNovaRuntimeContracts.full_native_config()
    raw['llm_config']['num_hidden_layers']=7
    model=NEOChatModel(NEOChatConfig(**raw)).to(torch.bfloat16).eval()
    state={key:value.clone() for key,value in model.state_dict().items()}
    path=tmp_path/'model.safetensors'
    save_file(state,str(path));(tmp_path/'config.json').write_text(json.dumps(raw))
    monkeypatch.setattr(loader,'_load_sensenova_tokenizer',lambda *a,**k:object())
    restored=loader.load_sensenova_from_path(str(path),torch_dtype=torch.bfloat16)['transformer']
    assert not any(type(module).__name__ in ('Int8Linear','Fp8Linear') for module in restored.modules())
    assert state.keys()==restored.state_dict().keys()
    for key,value in state.items():assert torch.equal(value,restored.state_dict()[key]),key
    config, topology, stride=inspect_artifact(path)
    assert config['llm_config']['num_hidden_layers']==7 and topology is None and stride==32
    missing=next(key for key in state if key.startswith('fm_modules.fm_head.') and key.endswith('weight'))
    for damaged in ({key:value for key,value in state.items() if key!=missing},dict(state,unexpected=torch.ones(1))):
        save_file(damaged,str(path))
        with pytest.raises((ValueError,RuntimeError)):
            loader.load_sensenova_from_path(str(path),torch_dtype=torch.bfloat16)
