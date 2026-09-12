"""Per-parameter LR warmup for fused optimizer update seams."""

from contextlib import contextmanager


_ATTR = "_sushi_fresh_param_warmup"


def arm_fresh_param_warmup(optimizer, scheduler, param_ids, anchor, warmup):
    """Attach one scheduler-position cohort without creating parameter groups."""
    ids = set(param_ids)
    if not ids or warmup <= 0:
        return 0
    setattr(optimizer, _ATTR, {
        "ids": ids,
        "scheduler": scheduler,
        "anchor": int(anchor),
        "warmup": int(warmup),
    })
    return len(ids)


def fresh_param_warmup_factor(optimizer, parameter):
    """Return the CPU-scalar multiplier for this parameter's current update."""
    cohort = getattr(optimizer, _ATTR, None)
    if not cohort:
        return 1.0
    position = int(getattr(cohort["scheduler"], "last_epoch", cohort["anchor"]))
    progress = (position - cohort["anchor"]) / float(cohort["warmup"])
    if progress >= 1.0:
        delattr(optimizer, _ATTR)
        return 1.0
    if id(parameter) not in cohort["ids"]:
        return 1.0
    return 0.0 if progress <= 0.0 else progress


@contextmanager
def parameter_warmup_lr(optimizer, parameter, group):
    """Temporarily scale a captured group's scalar LR for one fused update."""
    factor = fresh_param_warmup_factor(optimizer, parameter)
    if factor == 1.0:
        yield
        return
    original = group["lr"]
    group["lr"] = float(original) * factor
    try:
        yield
    finally:
        group["lr"] = original
