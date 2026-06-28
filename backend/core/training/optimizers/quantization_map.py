"""
Dynamic Quantization Map Creation

Ported from bitsandbytes (MIT License)
https://github.com/TimDettmers/bitsandbytes/blob/main/bitsandbytes/functional.py

This module creates non-uniform quantization maps with higher precision near zero,
which is critical for accurate 8-bit optimizer state quantization.
"""

import torch


def create_dynamic_map(signed=True, max_exponent_bits=7, total_bits=8):
    """
    Creates dynamic quantization map with more precision near zero.

    Ported from bitsandbytes/functional.py:280-330

    Args:
        signed: If True, create signed quantization map [-1, 1]
                If False, create unsigned map [0, 1]
        max_exponent_bits: Number of bits for exponent (default: 7)
        total_bits: Total bits for quantization (default: 8)

    Returns:
        torch.Tensor: Quantization map of shape [256] with FP32 values
    """
    data = []
    # These are additional items that come from the case where all exponent bits are zero
    non_sign_bits = total_bits - 1
    additional_items = 2 ** (non_sign_bits - max_exponent_bits) - 1

    # Iterate through exponents (smaller to larger magnitude)
    for i in range(max_exponent_bits):
        fraction_items = int(
            2 ** (i + non_sign_bits - max_exponent_bits) + 1
            if signed
            else 2 ** (i + non_sign_bits - max_exponent_bits + 1) + 1
        )
        boundaries = torch.linspace(0.1, 1, fraction_items, dtype=torch.float32)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        data += ((10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()
        if signed:
            data += (-(10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()

    # Additional items for exponent=0 case
    if additional_items > 0:
        boundaries = torch.linspace(0.1, 1, additional_items + 1, dtype=torch.float32)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        data += ((10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()
        if signed:
            data += (-(10 ** (-(max_exponent_bits - 1) + i)) * means).tolist()

    # Add zero and 1.0
    data.append(0)
    data.append(1.0)

    assert len(data) == 2**total_bits, f"Expected {2**total_bits} values, got {len(data)}"

    # Pad with zeros if needed (should not happen for total_bits=8)
    gap = 256 - len(data)
    for i in range(gap):
        data.append(0)

    # CRITICAL: sort the map into a monotonically increasing grid. The CUDA
    # quantize_value() does a BINARY SEARCH over this map, which requires it to
    # be sorted; the construction above groups values by exponent (and
    # interleaves +/- for signed), so without sorting binary search returns the
    # wrong code -> corrupted optimizer state -> divergence/instability. bitsand-
    # bytes sorts here for the same reason; sorting also makes our grid match
    # bnb's exactly.
    data.sort()

    return torch.tensor(data, dtype=torch.float32)


def create_quantization_map(signed=True):
    """
    Convenience function to create quantization map with default settings.

    Args:
        signed: If True, create signed map for exp_avg (momentum)
                If False, create unsigned map for exp_avg_sq (variance)

    Returns:
        torch.Tensor: Quantization map [256] FP32
    """
    return create_dynamic_map(signed=signed, max_exponent_bits=7, total_bits=8)


if __name__ == "__main__":
    # Test quantization map creation
    print("Testing quantization map creation...")

    # Signed map for exp_avg
    qmap_signed = create_quantization_map(signed=True)
    print(f"Signed map shape: {qmap_signed.shape}")
    print(f"Signed map range: [{qmap_signed.min():.6f}, {qmap_signed.max():.6f}]")
    print(f"Signed map[0] (most negative): {qmap_signed[0]:.6f}")
    print(f"Signed map[127] (zero): {qmap_signed[127]:.6f}")
    print(f"Signed map[255] (most positive): {qmap_signed[255]:.6f}")

    # Unsigned map for exp_avg_sq
    qmap_unsigned = create_quantization_map(signed=False)
    print(f"\nUnsigned map shape: {qmap_unsigned.shape}")
    print(f"Unsigned map range: [{qmap_unsigned.min():.6f}, {qmap_unsigned.max():.6f}]")
    print(f"Unsigned map[0] (zero): {qmap_unsigned[0]:.6f}")
    print(f"Unsigned map[255] (max): {qmap_unsigned[255]:.6f}")

    # Check precision near zero (signed)
    near_zero_indices = [125, 126, 127, 128, 129]
    print(f"\nPrecision near zero (signed map):")
    for idx in near_zero_indices:
        print(f"  map[{idx}] = {qmap_signed[idx]:.8f}")

    print("\nQuantization map creation test PASSED")
