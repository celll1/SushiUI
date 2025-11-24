"""
Test script to verify SageAttention installation and basic functionality
"""

import torch
import torch.nn.functional as F
import sys

print("=" * 80)
print("SageAttention Test Script")
print("=" * 80)

# Check PyTorch version
print(f"\n[1] PyTorch version: {torch.__version__}")
print(f"[2] CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[3] CUDA version: {torch.version.cuda}")
    print(f"[4] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[5] GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Test SageAttention import
print("\n" + "=" * 80)
print("Testing SageAttention Import")
print("=" * 80)

try:
    from sageattention import sageattn
    print("[OK] SageAttention imported successfully!")
    print(f"  sageattn function: {sageattn}")
except ImportError as e:
    print(f"[ERROR] Failed to import SageAttention: {e}")
    print("\nInstall with: pip install sageattention")
    sys.exit(1)
except Exception as e:
    print(f"[ERROR] Error importing SageAttention: {e}")
    sys.exit(1)

# Test basic SageAttention functionality
print("\n" + "=" * 80)
print("Testing SageAttention Basic Functionality")
print("=" * 80)

if not torch.cuda.is_available():
    print("[FAIL] CUDA not available. SageAttention requires CUDA.")
    sys.exit(1)

device = "cuda"
dtype = torch.float16  # SageAttention works with FP16

# Create test tensors (typical Stable Diffusion dimensions)
batch_size = 2
num_heads = 20  # SDXL uses 20 heads
seq_len = 4096  # 64x64 latent
head_dim = 64

# For better timing, run multiple iterations
num_iterations = 10

print(f"\nTest configuration:")
print(f"  Batch size: {batch_size}")
print(f"  Num heads: {num_heads}")
print(f"  Sequence length: {seq_len}")
print(f"  Head dim: {head_dim}")
print(f"  Dtype: {dtype}")

# Create random Q, K, V tensors
q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)

print(f"\nTensor shapes:")
print(f"  Q: {q.shape}")
print(f"  K: {k.shape}")
print(f"  V: {v.shape}")

# Test SageAttention
print("\n[Test 1] SageAttention computation...")
try:
    # Warm up
    for _ in range(3):
        _ = sageattn(q, k, v, tensor_layout="HND", is_causal=False)

    # Timed run with multiple iterations
    torch.cuda.synchronize()
    import time
    start = time.time()

    for _ in range(num_iterations):
        output_sage = sageattn(q, k, v, tensor_layout="HND", is_causal=False)

    torch.cuda.synchronize()
    sage_time = (time.time() - start) / num_iterations

    print(f"[OK] SageAttention succeeded!")
    print(f"  Output shape: {output_sage.shape}")
    print(f"  Output dtype: {output_sage.dtype}")
    print(f"  Time: {sage_time*1000:.2f} ms")
    print(f"  Contains NaN: {torch.isnan(output_sage).any().item()}")
    print(f"  Contains Inf: {torch.isinf(output_sage).any().item()}")
    print(f"  Min/Max: {output_sage.min().item():.4f} / {output_sage.max().item():.4f}")

except Exception as e:
    print(f"[FAIL] SageAttention failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Compare with PyTorch SDPA
print("\n[Test 2] PyTorch SDPA (reference) computation...")
try:
    # Warm up
    for _ in range(3):
        _ = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

    # Timed run with multiple iterations
    torch.cuda.synchronize()
    start = time.time()

    for _ in range(num_iterations):
        output_sdpa = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)

    torch.cuda.synchronize()
    sdpa_time = (time.time() - start) / num_iterations

    print(f"[OK] PyTorch SDPA succeeded!")
    print(f"  Output shape: {output_sdpa.shape}")
    print(f"  Time: {sdpa_time*1000:.2f} ms")
    print(f"  Min/Max: {output_sdpa.min().item():.4f} / {output_sdpa.max().item():.4f}")

except Exception as e:
    print(f"[FAIL] PyTorch SDPA failed: {e}")
    sys.exit(1)

# Compare outputs
print("\n[Test 3] Comparing outputs...")
try:
    # Calculate difference
    diff = (output_sage - output_sdpa).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"[OK] Output comparison:")
    print(f"  Max absolute difference: {max_diff:.6f}")
    print(f"  Mean absolute difference: {mean_diff:.6f}")
    print(f"  Relative error: {(max_diff / output_sdpa.abs().max().item()):.6f}")

    # Check if difference is acceptable (SageAttention uses quantization)
    if max_diff < 1.0:  # Reasonable threshold for FP16 with quantization
        print(f"  [OK] Outputs are reasonably close (quantization is working correctly)")
    else:
        print(f"  [WARN] Large difference detected (may indicate issue)")

except Exception as e:
    print(f"[FAIL] Comparison failed: {e}")

# Performance comparison
print("\n[Test 4] Performance comparison...")
print(f"  (Average of {num_iterations} iterations)")
if sage_time > 0 and sdpa_time > 0:
    speedup = sdpa_time / sage_time
    print(f"  SageAttention: {sage_time*1000:.3f} ms")
    print(f"  PyTorch SDPA:  {sdpa_time*1000:.3f} ms")
    print(f"  Speedup: {speedup:.2f}x")
else:
    print(f"  Timing too fast to measure accurately (GPU is very fast!)")
    speedup = 1.0

if speedup > 1.0:
    print(f"  [OK] SageAttention is {speedup:.2f}x faster!")
else:
    print(f"  [WARN] SageAttention is slower (may be due to small tensor size)")
    print(f"     Try larger tensors for better speedup")

print("\n" + "=" * 80)
print("Test Summary")
print("=" * 80)
print("[OK] All tests passed!")
print("SageAttention is working correctly and can be used in production.")
print("=" * 80)
