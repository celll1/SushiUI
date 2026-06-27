/*
 * Lion 8-bit Schedule-Free CUDA Kernel
 *
 * Based on:
 * - bitsandbytes 8-bit quantization (MIT License)
 * - Schedule-Free learning (Facebook Research, arXiv:2405.15682)
 * - Lion optimizer (Google Brain, arXiv:2302.06675)
 *
 * Algorithm:
 * 1. Dequantize z (momentum state)
 * 2. Compute y from z (training parameters)
 * 3. Lion sign-based update on y
 * 4. Update z (main sequence)
 * 5. Re-quantize z
 * 6. Compute x from z (iterate averaging)
 * 7. Write x to param
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cub/cub.cuh>
#include <stdint.h>
#include <stdio.h>
#include <type_traits>

// CUDA 13 (CCCL) removed cub::Max; provide a version-independent max functor.
namespace {
struct CubMaxOp {
    __host__ __device__ __forceinline__
    float operator()(const float &a, const float &b) const { return a > b ? a : b; }
};
}  // namespace

// Quantization block size (must match bitsandbytes)
#define QUANTIZATION_BLOCKSIZE 256
#define THREADS_PER_BLOCK 256

// ============================================================
// Dynamic Quantization Maps (Device Constants)
// ============================================================

// Signed map for z: [-1.0, 1.0] (momentum can be negative)
__device__ __constant__ float d_qmap_signed[256];


// ============================================================
// Quantization/Dequantization (from bitsandbytes)
// ============================================================

/*
 * Quantize a normalized value [-1, 1] to uint8 index.
 * Uses binary search in quantization map.
 */
__device__ inline uint8_t quantize_value(float value, const float* qmap) {
    // Binary search for closest quantization level
    int left = 0;
    int right = 255;

    while (left < right - 1) {
        int mid = (left + right) / 2;
        if (value < qmap[mid]) {
            right = mid;
        } else {
            left = mid;
        }
    }

    // Check both neighbors for closest match
    float dist_left = fabsf(value - qmap[left]);
    float dist_right = fabsf(value - qmap[right]);

    return (dist_left < dist_right) ? (uint8_t)left : (uint8_t)right;
}

/*
 * Dequantize uint8 code to float value.
 */
__device__ inline float dequantize_value(uint8_t code, const float* qmap, float absmax) {
    return qmap[code] * absmax;
}


// ============================================================
// Lion 8-bit Schedule-Free Update Kernel
// ============================================================

template<typename T>
__global__ void lion_8bit_schedulefree_update_kernel(
    T* __restrict__ param,                  // [numel] Parameters (will write x-sequence, FP32/FP16/BF16, GPU)
    const T* __restrict__ grad,             // [numel] Gradients (same dtype as param, GPU)
    uint8_t* __restrict__ state_z,          // [numel] z sequence quantized (momentum, UINT8, GPU)
    float* __restrict__ absmax_z,           // [num_blocks] z absmax (FP32, GPU)
    const float beta1,                      // Lion beta1 (interpolation, 0.9)
    const float beta2,                      // Lion beta2 (momentum EMA, 0.99)
    const float eps,                        // Unused in Lion (kept for API compatibility)
    const float lr,                         // Scheduled learning rate (lr * rect for RAdam)
    const float weight_decay,               // Weight decay (0.0 or 0.01)
    const float ckp1,                       // Averaging coefficient: (k+1)/(k+r)
    const float gnorm_scale,                // Gradient norm scaling (1.0 if no clipping)
    const bool cautious,                    // Cautious masking (sign alignment check)
    const int numel                         // Total number of elements
) {
    // Thread and block indices
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_tid = threadIdx.x;

    // Quantization block index (256 elements per quantization block)
    const int qblock_id = tid / QUANTIZATION_BLOCKSIZE;

    if (tid >= numel) return;

    // ============================================================
    // Step 1: Dequantize optimizer state (z, momentum)
    // ============================================================

    float current_absmax_z = absmax_z[qblock_id];
    uint8_t qz = state_z[tid];

    // Dequantize z (uses signed map)
    float z = dequantize_value(qz, d_qmap_signed, current_absmax_z);

    // ============================================================
    // Step 2: Load param (will compute y from it)
    // ============================================================

    float param_val;
    if constexpr (std::is_same<T, float>::value) {
        param_val = param[tid];
    } else if constexpr (std::is_same<T, __half>::value) {
        param_val = __half2float(param[tid]);
    } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
        param_val = __bfloat162float(param[tid]);
    }

    // ============================================================
    // Step 3: Compute y (training parameters)
    // ============================================================
    // y_t = (1 - ckp1) * z_{t-1} + ckp1 * x_{t-1}
    // Note: param currently holds x_{t-1}
    float y = (1.0f - ckp1) * z + ckp1 * param_val;

    // ============================================================
    // Step 4: Convert gradient to FP32
    // ============================================================

    float g;
    if constexpr (std::is_same<T, float>::value) {
        g = grad[tid] * gnorm_scale;
    } else if constexpr (std::is_same<T, __half>::value) {
        g = __half2float(grad[tid]) * gnorm_scale;
    } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
        g = __bfloat162float(grad[tid]) * gnorm_scale;
    }

    // ============================================================
    // Step 5: Lion Update Algorithm (on y)
    // ============================================================

    // 5.1. Interpolate: c_t = β1 * z_{t-1} + (1 - β1) * g_t
    float c_t = beta1 * z + (1.0f - beta1) * g;

    // 5.2. Cautious masking (if enabled)
    if (cautious) {
        // Binary mask: check sign alignment between c_t and gradient
        float mask_val = (c_t * g > 0.0f) ? 1.0f : 0.0f;

        // Apply mask to interpolated momentum
        c_t = c_t * mask_val;
    }

    // 5.3. Sign-based update with weight decay
    // update = sign(c_t) + weight_decay * y
    float update = (c_t > 0.0f ? 1.0f : (c_t < 0.0f ? -1.0f : 0.0f));
    if (weight_decay > 0.0f) {
        update += weight_decay * y;
    }

    // 5.4. Apply update to y
    // y_new = y - lr * update
    float y_new = y - lr * update;

    // ============================================================
    // Step 6: Update z (momentum EMA)
    // ============================================================
    // z_t = β2 * z_{t-1} + (1 - β2) * g_t
    float z_new = beta2 * z + (1.0f - beta2) * g;

    // ============================================================
    // Step 7: Compute block-level absmax for z
    // ============================================================

    // Shared memory for CUB BlockReduce
    typedef cub::BlockReduce<float, THREADS_PER_BLOCK> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage_z;

    // Compute abs value
    float abs_z = fabsf(z_new);

    // Block-level reduce (max)
    float block_absmax_z = BlockReduce(temp_storage_z).Reduce(abs_z, CubMaxOp{});

    // Update absmax (first thread in block writes)
    if (local_tid == 0) {
        absmax_z[qblock_id] = block_absmax_z;
    }
    __syncthreads();

    // Reload updated absmax
    float new_absmax_z = absmax_z[qblock_id];

    // ============================================================
    // Step 8: Re-quantize z
    // ============================================================

    // Normalize to [-1, 1]
    float normalized_z = (new_absmax_z > 0.0f) ? (z_new / new_absmax_z) : 0.0f;

    // Quantize using binary search
    uint8_t new_qz = quantize_value(normalized_z, d_qmap_signed);

    // Write back quantized state
    state_z[tid] = new_qz;

    // ============================================================
    // Step 9: Compute x (iterate averaging)
    // ============================================================
    // x_t = (1 - ckp1) * z_t + ckp1 * y_new
    float x = (1.0f - ckp1) * z_new + ckp1 * y_new;

    // ============================================================
    // Step 10: Write x to param
    // ============================================================

    if constexpr (std::is_same<T, float>::value) {
        param[tid] = x;
    } else if constexpr (std::is_same<T, __half>::value) {
        param[tid] = __float2half(x);
    } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
        param[tid] = __float2bfloat16(x);
    }
}


// ============================================================
// Note: Kernel instantiation is done in launcher
// ============================================================
// __global__ function templates cannot be explicitly instantiated
// in CUDA. The launcher (lion8bit_schedulefree_launcher.cu) will
// call the kernel template for each dtype (float, __half, __nv_bfloat16).
