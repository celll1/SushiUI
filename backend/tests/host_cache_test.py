from types import SimpleNamespace
import pytest
import torch
from core.memory_management.host_cache import release_unused_host_cache


@pytest.mark.parametrize("initialized,public,private,expected", [
    (False, True, True, None), (True, True, True, "public"),
    (True, False, True, "private"), (True, False, False, None),
])
def test_host_cache_release_without_initializing_cuda(monkeypatch, initialized, public, private, expected):
    called = []
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: initialized)
    monkeypatch.setattr(torch, "accelerator", SimpleNamespace(
        empty_host_cache=(lambda: called.append("public")) if public else None))
    monkeypatch.setattr(torch._C, "_host_emptyCache",
        (lambda: called.append("private")) if private else None, raising=False)
    monkeypatch.setattr(torch.cuda, "init", lambda: pytest.fail("must not initialize CUDA"))
    assert release_unused_host_cache() is (expected is not None)
    assert called == ([] if expected is None else [expected])
