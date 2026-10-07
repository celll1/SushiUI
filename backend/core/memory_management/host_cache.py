"""Release unused pinned staging buffers at model lifecycle boundaries."""
import torch


def release_unused_host_cache() -> bool:
    if not torch.cuda.is_initialized():
        return False
    release = getattr(getattr(torch, "accelerator", None), "empty_host_cache", None)
    if not callable(release):
        release = getattr(torch._C, "_host_emptyCache", None)
    if not callable(release):
        return False
    release()
    return True
