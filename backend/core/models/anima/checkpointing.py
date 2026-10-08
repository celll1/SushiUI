"""Prefix checkpoint limits preserve Anima's selected activation-offload mode."""
from contextlib import contextmanager

from core.models.ming_image_01.checkpointing import block_limit


def configure(model, config):
    modes = tuple(block.gradient_checkpoint_mode for block in model.blocks)
    model._anima_checkpoint_modes = modes
    limit = block_limit(config, len(modes))
    if limit is not None:
        for index, block in enumerate(model.blocks):
            block.gradient_checkpoint_mode = modes[index] if index < limit else "none"


@contextmanager
def partition_scope(model, config):
    original = tuple(block.gradient_checkpoint_mode for block in model.blocks)
    configured = getattr(model, "_anima_checkpoint_modes", original)
    if len(configured) != len(original):
        raise ValueError("Anima checkpoint configuration disagrees with model depth")
    limit = block_limit(config, len(original), partition=True)
    try:
        if limit is not None:
            for index, block in enumerate(model.blocks):
                block.gradient_checkpoint_mode = configured[index] if index < limit else "none"
        yield
    finally:
        for block, mode in zip(model.blocks, original):
            block.gradient_checkpoint_mode = mode
