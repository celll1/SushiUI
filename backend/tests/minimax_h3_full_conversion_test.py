"""Scaled-FP8 expansion agrees with loader mapping and quantized compute."""

import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file,load_file

from minimax_h3_hybrid_reader_test import _dit_tensors,INNER,HEADS,HEAD_DIM


def source(tmp_path,per_row):
    state = _dit_tensors(2,offset=.1,quant_marker=False)
    state.pop("blocks.0.mlp.fc2.weight_scale")
    for key in ("blocks.0.attn.qkv_proj.weight","blocks.0.mlp.fc1.weight","blocks.0.mlp.fc2.weight"):
        state[key] = state[key].to(torch.float8_e4m3fn)
        stem = key[:-len(".weight")]
        state[stem+".weight_scale"] = torch.linspace(.1,.9,state[key].shape[0]) if per_row else torch.tensor(.3)
        marker = dict(format="float8_e4m3fn")
        if key.endswith("fc2.weight"):
            marker["full_precision_matrix_mult"] = True
        state[stem+".comfy_quant"] = torch.tensor(list(json.dumps(marker).encode()),dtype=torch.uint8)
    path = tmp_path/"source.safetensors"
    save_file(state,str(path))
    directory = tmp_path/"official"/"transformer"
    directory.mkdir(parents=True)
    (directory/"config.json").write_text(json.dumps(dict(num_attention_heads=HEADS,attention_head_dim=HEAD_DIM)),encoding="utf-8")
    return path,state,str(directory.parent)


@pytest.mark.parametrize("per_row",[False,True])
def test_bounded_dense_conversion_preserves_qkv_and_swiglu_semantics(tmp_path,per_row):
    from core.models.minimax_h3.full_conversion import convert_checkpoint,plan_conversion
    from core.models.minimax_h3.loader import _guard_component_file,_synthesize_transformer_config,_map_dit_state_dict
    source_path,state,official = source(tmp_path,per_row)
    output = tmp_path/"dense.safetensors"
    progress = []
    convert_checkpoint(source_path,output,official_dir=official,work_bytes=256,
        progress=lambda done,total:progress.append((done,total)))
    dense = load_file(str(output))
    expected = {}
    for key,value in state.items():
        if key.endswith((".weight_scale",".comfy_quant",".input_scale")):
            continue
        if value.dtype == torch.float8_e4m3fn:
            scale = state[key[:-len(".weight")]+".weight_scale"].bfloat16()
            value = value.bfloat16()* (scale if scale.numel() == 1 else scale[:,None])
        expected[key] = value
        torch.testing.assert_close(dense[key],value,rtol=0,atol=0)
    assert set(dense) == set(expected)
    assert progress[-1][0] == progress[-1][1] and len(progress) > len(dense)
    header,_ = _guard_component_file(str(output),label="dense test")
    config = _synthesize_transformer_config(header,official)
    with safe_open(str(output),framework="pt",device="cpu") as handle:
        mapped,_ = _map_dit_state_dict(handle,header,config,torch.bfloat16)
        record = json.loads(handle.metadata()["sushi.h3_dequantization"])
    assert record["converted_matrices"] == 3
    qkv = expected["blocks.0.attn.qkv_proj.weight"]
    for i,name in enumerate(("to_q","to_k","to_v")):
        torch.testing.assert_close(mapped["transformer_blocks.0.attn."+name+".weight"],qkv[i*INNER:(i+1)*INNER],rtol=0,atol=0)
    gate,up = expected["blocks.0.mlp.fc1.weight"].chunk(2)
    torch.testing.assert_close(mapped["transformer_blocks.0.ff.net.0.proj.weight"],torch.cat((up,gate)),rtol=0,atol=0)
    weights = mapped["transformer_blocks.0.attn.to_q.weight"]
    layer = torch.nn.Linear(INNER,INNER,bias=False,dtype=torch.bfloat16)
    layer.weight.data.copy_(weights)
    value = torch.randn(3,INNER,dtype=torch.bfloat16,requires_grad=True)
    torch.testing.assert_close(layer(value),torch.nn.functional.linear(value,qkv[:INNER]),rtol=0,atol=0)
    layer(value).float().square().mean().backward()
    assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all()
    before = source_path.read_bytes()
    with pytest.raises(FileExistsError):
        convert_checkpoint(source_path,output,official_dir=official)
    assert source_path.read_bytes() == before
    assert len(plan_conversion(source_path,official_dir=official)[2]) == 3


@pytest.mark.parametrize("corruption",["scale","nan","marker","no_scale"])
def test_failed_conversion_never_publishes_partial_model(tmp_path,corruption):
    from core.models.minimax_h3.full_conversion import convert_checkpoint
    path,state,official = source(tmp_path,True)
    stem = "blocks.0.mlp.fc2"
    if corruption == "scale":
        state[stem+".weight_scale"].fill_(0)
    elif corruption == "nan":
        state[stem+".weight"] = torch.full_like(state[stem+".weight"],float("nan"))
    elif corruption == "marker":
        state[stem+".comfy_quant"] = torch.tensor(list(b'{"format":"float8_e4m3fn","unknown":true}'),dtype=torch.uint8)
    else:
        del state[stem+".weight_scale"]
    save_file(state,str(path))
    output = tmp_path/"bad.safetensors"
    with pytest.raises((ValueError,RuntimeError)):
        convert_checkpoint(path,output,official_dir=official,work_bytes=256)
    assert not output.exists() and not list(tmp_path.glob("*.partial"))
