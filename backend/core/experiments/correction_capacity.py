"""Independent factory-described boundary-capacity candidates; no live migration."""
from copy import deepcopy

import torch

from core.rewiring.hybrid_modules import HybridModules
from core.rewiring.hybrid_spec import build_hybrid_spec, legacy_topology, validate_hybrid_spec


def widen_boundaries(source, ranks):
    """Preserve fitted coordinates and initialize new output columns to zero."""
    if not isinstance(source, HybridModules) or not isinstance(ranks, dict) or not ranks:
        raise ValueError("Capacity candidate requires hybrid modules and boundary ranks")
    declaration = validate_hybrid_spec(source.declaration)
    boundaries = deepcopy(declaration['boundaries'])
    items = {item['id']: item for item in boundaries}
    if not set(ranks) <= set(items):
        raise ValueError("Unknown boundary capacity ID")
    for name, rank in ranks.items():
        if type(rank) is not int or rank <= items[name]['rank']:
            raise ValueError("Boundary capacity must increase its integer rank")
        items[name]['rank'] = rank
    candidate_spec = build_hybrid_spec(legacy_topology(declaration),
        stream_widths=declaration['stream_widths'], boundaries=boundaries,
        refiners=declaration['refiners'], skips=declaration['skips'])
    reference = next(source.parameters())
    candidate = HybridModules(candidate_spec).to(device=reference.device, dtype=reference.dtype)
    old, new = source.state_dict(), candidate.state_dict()
    if set(old) != set(new):
        raise ValueError("Capacity candidate requires the unextended public factory state")
    with torch.no_grad():
        for name, value in new.items():
            previous = old[name]
            if value.shape == previous.shape:
                value.copy_(previous)
            elif name.endswith('.reduce.weight'):
                if value.shape[1:] != previous.shape[1:] or value.shape[0] <= previous.shape[0]:
                    raise ValueError("Capacity input projection layout changed")
                value[:previous.shape[0]].copy_(previous)
            elif name.endswith('.expand.weight'):
                if value.shape[0] != previous.shape[0] or value.shape[2:] != previous.shape[2:] or value.shape[1] <= previous.shape[1]:
                    raise ValueError("Capacity output projection layout changed")
                value.zero_()
                value[:, :previous.shape[1]].copy_(previous)
            else:
                raise ValueError("Unsupported capacity tensor layout")
    return candidate
