"""Hybrid/global adapter generation retains native partition coordinates."""
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from hybrid_qwen_test import model_and_spec
from hybrid_adapter_payload_test import file_for
from qwen_partition_generation_test import payload, strips
from qwen_partition_full_checkpoint_test import _forward
from core.inference.qwen_image_21_partition import PartitionExecutor, resolve_partition_plan
from core.models.qwen_image_21.partition_global import QwenPartitionGlobalAdapter
from core.models.qwen_image_21.residual_unet import restore_topology
from core.pipeline_backends.qwen_image_21 import QwenImage21Mixin


def test_real_hybrid_partition_global_matches_native_tile_execution():
    model, spec, _ = model_and_spec()
    restore_topology(model, spec)
    with torch.no_grad():
        for parameter in model.rewiring_modules.parameters(): parameter.fill_(.01)
    saved = payload(model)
    plan = resolve_partition_plan(dict(mode='manual', layout=strips(2), halo_tokens=0), 8, 16)
    executor = PartitionExecutor(model, plan, [(saved, None)], use_cache=True)
    assert not executor.use_cache
    grid = torch.randn(1, 8, 16, 8)
    adapter = QwenPartitionGlobalAdapter(8, model.inner_dim, rank=4, summary_tokens=3)
    adapter.load_tensors(saved['state'])
    args = dict(hidden_states=grid.flatten(1, 2), timestep=torch.tensor([.5]),
                encoder_hidden_states=torch.ones(1, 2, 32), img_shapes=[[(1, 8, 16)]],
                img_mask=torch.cat([torch.zeros(1, 2, dtype=torch.bool),
                                    torch.ones(1, 32, dtype=torch.bool)], 1), return_dict=False)
    with torch.no_grad():
        actual = executor.predict(branch='cond', **args).reshape_as(grid)
        expected = torch.empty_like(grid)
        for box in plan.inputs:
            tile = _forward(model, grid, box, adapter(grid, box))[:, -box.tokens:].reshape(1, box.height, box.width, 8)
            expected[:, box.top:box.bottom, box.left:box.right] = tile
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert not executor.caches
    executor.close()
    assert not executor.modules and not executor.payloads
    invalid = resolve_partition_plan(dict(mode='manual', layout=strips(2), halo_tokens=2), 8, 16)
    with pytest.raises(ValueError): PartitionExecutor(model, invalid)


@pytest.mark.parametrize('fail', [False, True])
def test_backend_restores_combined_hybrid_lora_and_global_payload(tmp_path, monkeypatch, fail):
    model, spec, _ = model_and_spec()
    restore_topology(model, spec)
    file = file_for(model, spec)
    saved = payload(model)
    file.metadata.update(saved['metadata'], model_type='qwen_image_21')
    file.tensors.update({'qwen_partition_global_adapter.' + key: value for key, value in saved['state'].items()})
    target = model.transformer_blocks[0].attn.to_q
    stem = 'lora_unet_transformer_blocks__0__attn__to_q'
    file.tensors.update({stem + '.lora_down.weight': torch.ones(2, target.in_features),
                         stem + '.lora_up.weight': torch.zeros(target.out_features, 2)})
    path = tmp_path / 'combined.safetensors'
    save_file(file.tensors, path, metadata=file.metadata)
    restore_topology(model, None)
    owner = QwenImage21Mixin()
    owner.qwen_image_21_components = {'transformer': model}
    owner.cancel_requested = False
    monkeypatch.setattr(owner, '_qwen21_resolve_lora_path', str)
    received = []
    def pipe(**kwargs):
        executor = kwargs['partition_executor']
        received.append(executor)
        assert model._residual_unet_spec == spec
        assert len(executor.payloads) == 1 and not executor.use_cache
        assert kwargs['use_kv_cache'] is False
        if fail: raise RuntimeError('test failure')
        return SimpleNamespace(images=['image'])
    monkeypatch.setattr(owner, '_qwen_image_21_pipe', lambda: pipe)
    params = dict(seed=4, width=256, height=128, loras=[dict(path=str(path), strength=1)],
                  qwen_image_21_partition=dict(mode='manual', layout=strips(2), halo_tokens=0))
    if fail:
        with pytest.raises(RuntimeError, match='test failure'): owner._qwen_image_21_run(params)
    else:
        assert owner._qwen_image_21_run(params)[0] == 'image'
        assert params['dit_rewiring_actual']['spec'] == spec
        assert len(params['qwen_image_21_partition_actual']['global_adapters']) == 1
    assert len(received) == 1 and not received[0].payloads and not received[0].modules
    assert not hasattr(model, 'rewiring_modules') and not hasattr(model, '_residual_unet_spec')
    assert model.transformer_blocks[0].attn.to_q is target
    assert not owner._qwen21_loaded_files
