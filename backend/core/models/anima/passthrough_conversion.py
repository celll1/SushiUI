"""Function-preserving conversion of a legacy Anima spatial swap to full passthrough.

Legacy backbone channel k >= old (an index copy of target channel k) becomes channel
old + k. Channels old..2*old-1 are new identity paths for target channels 0..old-1,
with zero latent-IO weights, so the converted model computes the same function.
"""
import hashlib
import json

import torch

from core.models.components.latent_adapter import backbone_width, validate_declaration
from core.models.components.spatial_swap import META_KEY, read_declaration
from .partition_global import METADATA_KEY as GLOBAL_KEY

PREFIX = "net."
CONVERSION_KEY = "sushi.anima.passthrough_conversion"
# Latent-facing tensors the conversion reshapes; their saved optimizer moments no longer fit.
RESHAPED = ("x_embedder.proj.1.weight", "final_layer.linear.weight",
            "vae_latent_bridge.input.projection.weight", "vae_latent_bridge.input.projection.bias",
            "vae_latent_bridge.output.projection.weight",
            "anima_partition_global_adapter.key.weight", "anima_partition_global_adapter.query.weight",
            "anima_partition_global_adapter.value.weight")


def strip_reshaped_moments(payload, *, owner="transformer."):
    """Drop saved moments of reshaped tensors so a named optimizer restore marks them fresh."""
    names = payload.get("_sushi_param_names")
    if not names or len(names) != len(payload["param_groups"]):
        raise ValueError("Optimizer state lacks parameter names; a partial restore cannot be targeted")
    wanted = {owner + name for name in RESHAPED}
    state, dropped = dict(payload["state"]), []
    for group_names, group in zip(names, payload["param_groups"]):
        for name, saved_id in zip(group_names, group["params"]):
            if name in wanted:
                for key in (saved_id, str(saved_id)):
                    if state.pop(key, None) is not None:
                        dropped.append(name)
    missing = {owner + name for name in RESHAPED[:5]} - set(dropped)
    if missing:
        raise ValueError(f"Optimizer state has no moments for {sorted(missing)}")
    return dict(payload, state=state), dropped


def _channel_index(old, legacy_width):
    return torch.tensor([c if c < old else old + c for c in range(legacy_width)])


def _remap(value, axis, index, size):
    shape = list(value.shape)
    shape[axis] = size
    result = value.new_zeros(shape)
    result.index_copy_(axis, index.to(value.device), value)
    return result


def convert(state, metadata):
    """Return converted (state, metadata); inputs are not modified."""
    declaration = read_declaration(metadata)
    if declaration is None or declaration.get("passthrough") is not None:
        raise ValueError("Conversion requires a legacy index-passthrough spatial swap")
    if declaration.get("noise_policy") is None:
        raise ValueError("Conversion supports Gaussian-aligned swaps only")
    from core.rewiring.spec import META_SPEC
    from core.rewiring.provenance import META_RECORD
    if META_SPEC in metadata or META_RECORD in metadata:
        raise ValueError("Rewired or hybrid-initialized Anima checkpoints are not supported")
    old, new = declaration["old_channels"], declaration["new_channels"]
    full = dict(validate_declaration(dict(declaration, passthrough="full")))
    legacy_width, width = backbone_width(declaration), backbone_width(full)
    config = json.loads(metadata["transformer_config"])
    if (config["in_channels"], config["out_channels"], config["patch_spatial"], config["patch_temporal"]) != (
            legacy_width, legacy_width, 2, 1) or not config["concat_padding_mask"]:
        raise ValueError("Checkpoint geometry disagrees with its legacy swap declaration")
    index = _channel_index(old, legacy_width)
    phases = 4
    state = dict(state)
    key = lambda name: PREFIX + name

    embed = state[key("x_embedder.proj.1.weight")]
    # The padding-mask channel follows the latent channels.
    columns = (torch.cat([index, torch.tensor([width])])[:, None]*phases + torch.arange(phases)).flatten()
    if embed.shape[1] != (legacy_width + 1)*phases:
        raise ValueError("Unexpected Anima patch-embedding width")
    state[key("x_embedder.proj.1.weight")] = _remap(embed, 1, columns, (width + 1)*phases)

    head = state[key("final_layer.linear.weight")]
    rows = (torch.arange(phases)[:, None]*width + index[None, :]).flatten()
    state[key("final_layer.linear.weight")] = _remap(head, 0, rows, phases*width)

    for name in ("key", "query", "value"):
        weight = state.get(key(f"anima_partition_global_adapter.{name}.weight"))
        if weight is not None:
            features = torch.cat([(index[:, None]*phases + torch.arange(phases)).flatten(),
                                  width*phases + torch.arange(2)])
            state[key(f"anima_partition_global_adapter.{name}.weight")] = _remap(weight, 1, features, width*phases + 2)

    bridge = "vae_latent_bridge."
    weight = state[key(bridge + "input.projection.weight")]
    bias = state[key(bridge + "input.projection.bias")]
    weight, bias = _remap(weight, 0, index, width), _remap(bias, 0, index, width)
    for c in range(old):
        weight[old + c, c, 1, 1] = 1
    state[key(bridge + "input.projection.weight")], state[key(bridge + "input.projection.bias")] = weight, bias
    weight = _remap(state[key(bridge + "output.projection.weight")], 1, index, width)
    for c in range(old):
        weight[c, old + c, 1, 1] = 1
    state[key(bridge + "output.projection.weight")] = weight
    spec = state[key(bridge + "input.bridge_spec")]
    state[key(bridge + "input.bridge_spec")] = torch.tensor([1, new, width, 1, 1], dtype=spec.dtype)
    state[key(bridge + "output.bridge_spec")] = torch.tensor([1, width, new, 1, 1], dtype=spec.dtype)
    identity = hashlib.sha256(json.dumps(full, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).digest()
    state[key(bridge + "noise_alignment.identity")] = torch.tensor(list(identity), dtype=torch.uint8)

    metadata = dict(metadata)
    metadata[META_KEY] = json.dumps(full, sort_keys=True, allow_nan=False)
    metadata[CONVERSION_KEY] = json.dumps(dict(from_width=legacy_width, to_width=width), sort_keys=True)
    metadata["transformer_config"] = json.dumps(dict(config, in_channels=width, out_channels=width))
    if GLOBAL_KEY in metadata:
        metadata[GLOBAL_KEY] = json.dumps(dict(json.loads(metadata[GLOBAL_KEY]), latent_channels=width), sort_keys=True)
    return state, metadata
