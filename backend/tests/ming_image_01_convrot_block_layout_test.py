"""CPU-only cache topology checks; no CUDA kernel or hardware claim."""
from types import SimpleNamespace, MethodType
import pytest
import torch
from torch import nn
from core.models.common import quantized_frozen_training as qft

MING_GROUPS = ('noise_refiner', 'context_refiner', 'layers')

@pytest.mark.parametrize('profile', ['generation_edit', 'layer_decompose'])
def test_native_ming_groups_cover_every_refiner_and_main_block(profile):
    from ming_image_01_sampling_test import make_sampler
    model = make_sampler(profile).components.transformer
    before = {key: value.clone() for key, value in model.state_dict().items()}
    paths = qft.convrot_block_paths(model, MING_GROUPS)
    for prefix, ordinal in [('noise_refiner.0', 0), ('context_refiner.0', 1), ('layers.0', 2), ('layers.1', 3)]:
        assert paths[prefix] == ordinal
        assert all(paths[f'{prefix}.{name}'] == ordinal for name, _ in model.get_submodule(prefix).named_modules() if name)
    assert not any(name.startswith('cap_embedder') for name in paths)
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
    assert not torch.cuda.is_initialized()


def test_qwen_default_ordinals_and_outside_projection_are_unchanged():
    model = nn.Module()
    model.transformer_blocks = nn.ModuleList([nn.Sequential(nn.Linear(2, 2)) for _ in range(4)])
    model.proj_out = nn.Linear(2, 2)
    paths = qft.convrot_block_paths(model)
    assert [paths[f'transformer_blocks.{i}.0'] for i in range(4)] == list(range(4))
    assert 'proj_out' not in paths
    assert qft.convrot_block_paths(nn.Module()) == {}


@pytest.mark.parametrize('groups', [(), ('layers', 'layers'), ('',), ('missing',), 'layers', (None,), ('projection',)])
def test_invalid_explicit_layout_is_refused(groups):
    model = nn.Module()
    model.layers = nn.ModuleList([nn.Linear(2, 2)])
    model.projection = nn.Linear(2, 2)
    with pytest.raises(ValueError): qft.convrot_block_paths(model, groups)


def test_shared_blocks_are_refused_instead_of_ambiguous_ordinal():
    model = nn.Module()
    block = nn.Linear(2, 2)
    model.layers = nn.ModuleList([block, block])
    with pytest.raises(ValueError, match='shared'): qft.convrot_block_paths(model, ('layers',))


def test_bounded_cache_crosses_ming_group_boundary_with_ordinal_lookup(monkeypatch):
    # Exercise the actual residency/eviction policy with a CPU scheduling double.
    cache = object.__new__(qft.ConvRotBlockPrefetchCache)
    names = ['noise_refiner.0.q', 'context_refiner.0.q', 'layers.0.q', 'layers.1.q']
    cache.path_to_block = {name: i for i, name in enumerate(names)}
    cache.blocks = {i: [(name, None)] for i, name in enumerate(names)}
    cache.outside = {}; cache.weights = {}; cache.events = {}; cache.resident_blocks = []
    cache.cache_blocks = 2; cache.prefetch_depth = 1
    cache.dtype = torch.bfloat16; cache.device = torch.device('cpu')
    cache.wait_count = cache.prefetch_hits = cache.prefetch_misses = 0
    cache._last_acquired_block = None
    scheduled = []
    class Weight:
        def __init__(self, ordinal): self.ordinal = ordinal
        def record_stream(self, stream): pass
    def schedule(owner, block):
        if block in owner.events or block not in owner.blocks: return
        scheduled.append(block); owner.events[block] = block
        owner.weights[names[block]] = Weight(block); owner.resident_blocks.append(block)
    cache._schedule = MethodType(schedule, cache)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda device: SimpleNamespace(wait_event=lambda event: None))
    for ordinal in [3, 2, 1, 0, 3, 2, 1, 0]:
        weight = cache.acquire(names[ordinal], cache.device, cache.dtype)
        assert weight.ordinal == ordinal
        assert len(cache.resident_blocks) <= 2
        assert len(cache.weights) <= 2
    assert scheduled == [3, 2, 1, 0, 3, 2, 1, 0]
    assert cache.prefetch_hits == 6 and cache.prefetch_misses == 2
    assert not torch.cuda.is_initialized()


def test_named_sequential_children_preserve_actual_paths():
    from collections import OrderedDict
    model = nn.Module()
    model.layers = nn.Sequential(OrderedDict([('first', nn.Linear(2, 2)), ('second', nn.Linear(2, 2))]))
    assert qft.convrot_block_paths(model, ('layers',)) == {'layers.first': 0, 'layers.second': 1}


def test_prefetch_enable_forwards_layout_without_changing_frozen_flags(monkeypatch):
    model = nn.Module()
    captured = {}
    fake = SimpleNamespace(resident_limit_bytes=0)
    def cache_factory(module, **kwargs):
        assert module is model
        captured.update(kwargs)
        return fake
    monkeypatch.setattr(qft, 'ConvRotBlockPrefetchCache', cache_factory)
    assert qft.enable_frozen_training_prefetch_backward(model, dtype=torch.bfloat16, block_groups=MING_GROUPS) == (0, 0, fake)
    assert captured['block_groups'] == MING_GROUPS
    assert captured['cache_blocks'] == 2 and captured['prefetch_depth'] == 1
    assert not torch.cuda.is_initialized()
