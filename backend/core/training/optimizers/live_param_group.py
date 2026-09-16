"""Resolve a parameter's param group at update time, for per-parameter fused hooks.

A hook must never hold a group dict: ``load_state_dict`` (torch's and the
ring-buffer uint8 loaders') and any rewrite of ``param_groups`` replace the dicts,
after which the scheduler writes LRs the held dict never sees.
"""

from typing import Tuple

_ATTR = "_sushi_live_param_group_index"


def live_param_group(optimizer, param) -> Tuple[dict, int, int]:
    """``(group, gindex, pindex)`` of ``param`` in ``optimizer.param_groups`` now.

    O(1) per call: a cached position is used only if the live groups still hold
    this exact tensor there; any rewrite rebuilds the index once.
    """
    groups = optimizer.param_groups
    index = getattr(optimizer, _ATTR, None)
    if index is not None:
        cached = index.get(id(param))
        if cached is not None:
            gindex, pindex = cached
            if gindex < len(groups):
                params = groups[gindex]["params"]
                if pindex < len(params) and params[pindex] is param:
                    return groups[gindex], gindex, pindex

    index = {
        id(p): (gindex, pindex)
        for gindex, group in enumerate(groups)
        for pindex, p in enumerate(group["params"])
    }
    setattr(optimizer, _ATTR, index)
    cached = index.get(id(param))
    if cached is None:
        raise RuntimeError(
            f"A fused-backward update fired for a parameter {tuple(param.shape)} that is "
            f"in no param_group of {type(optimizer).__name__}. Under the fused backward "
            f"pass nothing else would update it."
        )
    gindex, pindex = cached
    return groups[gindex], gindex, pindex
