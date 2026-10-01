"""Main-stream checkpoint limits resolved against native checkpoint depth."""
from contextlib import contextmanager
from api.param_defaults import TRAINING_DEFAULTS


def block_limit(config,depth,*,partition=False):
    name="partition_gradient_checkpointing_blocks" if partition else "gradient_checkpointing_blocks"
    canonical,legacy="dit_"+name,"qwen_"+name
    a,b=config.get(canonical),config.get(legacy)
    if a is not None and b is not None and a != b:
        raise ValueError(f"{canonical} conflicts with {legacy}")
    value=a if a is not None else b if b is not None else TRAINING_DEFAULTS[canonical]
    if value is not None and (type(value) is not int or not 0 <= value <= depth):
        raise ValueError(f"{canonical} must be an integer between 0 and {depth}")
    return value


def configure(model,config):
    limit=block_limit(config,len(model.layers))
    if limit is not None:model._training_gradient_checkpointing_blocks=limit
    elif hasattr(model,"_training_gradient_checkpointing_blocks"):del model._training_gradient_checkpointing_blocks


@contextmanager
def partition_scope(model,config):
    missing=object();original=getattr(model,"_training_gradient_checkpointing_blocks",missing)
    limit=block_limit(config,len(model.layers),partition=True)
    if limit is not None:model._training_gradient_checkpointing_blocks=limit
    try:yield
    finally:
        if original is missing:
            if hasattr(model,"_training_gradient_checkpointing_blocks"):del model._training_gradient_checkpointing_blocks
        else:model._training_gradient_checkpointing_blocks=original
