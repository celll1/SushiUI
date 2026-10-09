"""Finite spatial context required by hybrid convolution and interpolation paths."""
from .hybrid_spec import validate_hybrid_spec


def convolution_halo(declaration, stream):
    declaration = validate_hybrid_spec(declaration)
    if stream not in declaration["stream_widths"]:
        raise ValueError("Unknown hybrid partition stream")
    count = next(item["block_count"] for item in declaration["streams"] if item["name"] == stream)
    starts = {item["start_block"]: item for item in declaration["regions"] if item["stream"] == stream}
    boundaries = {item["region"]: item for item in declaration["boundaries"]}
    refiners = [item for item in declaration["refiners"] if item["stream"] == stream]
    skips = [item for item in declaration["skips"] if item["stream"] == stream]
    radius, scale, stack, captured = (0, 0), (1, 1), [], {}
    for index in range(count):
        while stack and index == stack[-1]["region"]["end_block"]:
            entry = stack.pop()
            region = entry["region"]
            radius = tuple(max(radius[axis], entry["low_radius"][axis]) + scale[axis]
                           for axis in range(2))
            scale = entry["scale"]
            if region["id"] in boundaries:
                kernel_radius = boundaries[region["id"]]["kernel"] // 2
                radius = tuple(radius[axis] + kernel_radius * scale[axis] for axis in range(2))
            radius = tuple(max(radius[axis], entry["radius"][axis]) for axis in range(2))
        region = starts.get(index)
        if region is not None:
            factor = region["down_factor"]
            kernel_radius = boundaries[region["id"]]["kernel"] // 2 if region["id"] in boundaries else 0
            old_radius = radius
            radius = tuple(radius[axis] + max(factor[axis] - 1, kernel_radius) * scale[axis]
                           for axis in range(2))
            stack.append(dict(region=region, radius=old_radius, low_radius=radius, scale=scale))
            scale = tuple(scale[axis] * factor[axis] for axis in range(2))
        for skip in skips:
            if skip["destination_before"] == index:
                radius = tuple(max(radius[axis], captured[skip["id"]][axis]) for axis in range(2))
        for item in refiners:
            if item["after_block"] == index:
                radius = tuple(radius[axis] + item["kernel"] // 2 * scale[axis] for axis in range(2))
        for skip in skips:
            if skip["source_after"] == index:
                captured[skip["id"]] = radius
    return radius


def resolved_partition_halo(declaration, stream, configured, *, codec_radius=(0, 0)):
    radius = max(a + b for a, b in zip(convolution_halo(declaration, stream), codec_radius))
    return max(configured, (radius + 1) // 2 * 2)


def spatial_codec_radius(pair, patch_size):
    if pair is None:
        return (0, 0)
    radius = [0, 0]
    for name, pack_name in (("input", "source_pack"), ("output", "target_pack")):
        bridge = getattr(pair, name)
        pack = getattr(bridge, pack_name)
        for axis, kernel in enumerate(bridge.projection.kernel_size):
            radius[axis] += (kernel // 2 * pack + patch_size - 1) // patch_size
    # Gaussian complements run in parallel on the external latent grid.
    noise = getattr(pair, "noise_alignment", None)
    if noise is not None:
        for axis in range(2):
            noise_radius = sum(value.shape[-2 + axis] // 2 for value in (noise.incoming, noise.outgoing))
            radius[axis] = max(radius[axis], (noise_radius + patch_size - 1) // patch_size)
    return tuple(radius)
