/*
 * AdamW 8-bit Schedule-Free CUDA Kernel
 *
 * Based on:
 * - bitsandbytes 8-bit quantization (MIT License)
 * - Schedule-Free learning (Facebook Research, arXiv:2405.15682)
 *
 * Algorithm:
 * 1. Dequantize z and exp_avg_sq
 * 2. Update exp_avg_sq (second moment)
 * 3. Compute normalized gradient
 * 4. Update y (training parameters, in-place on param)
 * 5. Update z (main sequence)
 * 6. Re-quantize z and exp_avg_sq
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cub/cub.cuh>
#include <stdint.h>
#include <stdio.h>
#include <type_traits>

// Quantization block size (must match bitsandbytes)
#define QUANTIZATION_BLOCKSIZE 256
#define THREADS_PER_BLOCK 256

// ============================================================
// Dynamic Quantization Maps (Device Constants)
// ============================================================

// Signed map for z: [-1.0, 1.0] (z can be negative like weights)
__device__ __constant__ float d_qmap_signed[256];

// Unsigned map for exp_avg_sq (variance): [0.0, 1.0]
__device__ __constant__ float d_qmap_unsigned[256];


// ============================================================
// Quantization/Dequantization (from bitsandbytes)
// ============================================================

/*
 * Quantize a normalized value [-1, 1] or [0, 1] to uint8 index.
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
// AdamW 8-bit Schedule-Free Update Kernel
// ============================================================

template<typename T>
__global__ void adamw_8bit_schedulefree_update_kernel(
    T* __restrict__ param,                  // [numel] y (training parameters, FP32/FP16/BF16, GPU)
    const T* __restrict__ grad,             // [numel] Gradients (same dtype as param, GPU)
    uint8_t* __restrict__ state_z,          // [numel] z sequence quantized (UINT8, GPU)
    uint8_t* __restrict__ state_exp_avg_sq, // [numel] exp_avg_sq quantized (UINT8, GPU)
    float* __restrict__ absmax_z,           // [num_blocks] z absmax (FP32, GPU)
    float* __restrict__ absmax2,            // [num_blocks] exp_avg_sq absmax (FP32, GPU)
    const float beta1,                      // AdamW beta1 (0.9)
    const float beta2,                      // AdamW beta2 (0.999)
    const float eps,                        // AdamW epsilon (1e-8)
    const float lr,                         // Scheduled learning rate
    const float weight_decay,               // Weight decay (0.01)
    const float ckp1,                       // Averaging coefficient
    const float gnorm_scale,                // Gradient norm scaling (1.0 if no clipping)
    const float bias_correction2,           // Bias correction for exp_avg_sq
    const int numel                         // Total number of elements
) {
    // Thread and block indices
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_tid = threadIdx.x;

    // Quantization block index (256 elements per quantization block)
    const int qblock_id = tid / QUANTIZATION_BLOCKSIZE;

    if (tid >= numel) return;

    // ============================================================
    // Step 1: Dequantize optimizer states (z and exp_avg_sq)
    // ============================================================

    float current_absmax_z = absmax_z[qblock_id];
    float current_absmax2 = absmax2[qblock_id];

    uint8_t qz = state_z[tid];
    uint8_t q2 = state_exp_avg_sq[tid];

    // Dequantize (z uses signed map, exp_avg_sq uses unsigned map)
    float z = dequantize_value(qz, d_qmap_signed, current_absmax_z);
    float exp_avg_sq = dequantize_value(q2, d_qmap_unsigned, current_absmax2);

    // ============================================================
    // Step 2: Load y (param is y in Schedule-Free notation)
    // ============================================================

    float y;
    if constexpr (std::is_same<T, float>::value) {
        y = param[tid];
    } else if constexpr (std::is_same<T, __half>::value) {
        y = __half2float(param[tid]);
    } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
        y = __bfloat162float(param[tid]);
    }

    // ============================================================
    // Step 3: Update exp_avg_sq (second moment)
    // ============================================================

    // Convert gradient to FP32
    float g;
    if constexpr (std::is_same<T, float>::value) {
        g = grad[tid] * gnorm_scale;
    } else if constexpr (std::is_same<T, __half>::value) {
        g = __half2float(grad[tid]) * gnorm_scale;
    } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
        g = __bfloat162float(grad[tid]) * gnorm_scale;
    }

    exp_avg_sq = beta2 * exp_avg_sq + (1.0f - beta2) * (g * g);

    // ============================================================
    // Step 4: Compute normalized gradient
    // ============================================================

    // Bias-corrected denominator
    float denom = sqrtf(exp_avg_sq / bias_correction2) + eps;

    // Normalize gradient
    float grad_normalized = g / denom;

    // Weight decay at y
    if (weight_decay > 0.0f) {
        grad_normalized += weight_decay * y;
    }

    // ============================================================
    // Step 5: Update y (training parameters)
    // ============================================================

    // y = (1 - ckp1) * y + ckp1 * z + lr * (beta1 * (1 - ckp1) - 1) * grad_normalized
    y = (1.0f - ckp1) * y + ckp1 * z + lr * (beta1 * (1.0f - ckp1) - 1.0f) * grad_normalized;

    // ============================================================
    // Step 6: Update z (main sequence)
    // ============================================================

    z = z - lr * grad_normalized;

    // ============================================================
    // Step 7: Compute block-level absmax for z and exp_avg_sq
    // ============================================================

    // Shared memory for CUB BlockReduce
    typedef cub::BlockReduce<float, THREADS_PER_BLOCK> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage_z;
    __shared__ typename BlockReduce::TempStorage temp_storage_sq;

    // Compute abs values
    float abs_z = fabsf(z);
    float abs_sq = fabsf(exp_avg_sq);

    // Block-level reduce (max)
    float block_absmax_z = BlockReduce(temp_storage_z).Reduce(abs_z, cub::Max());
    __syncthreads();  // Reuse shared memory
    float block_absmax_sq = BlockReduce(temp_storage_sq).Reduce(abs_sq, cub::Max());

    // Update absmax (first thread in block writes)
    if (local_tid == 0) {
        absmax_z[qblock_id] = block_absmax_z;
        absmax2[qblock_id] = block_absmax_sq;
    }
    __syncthreads();

    // Reload updated absmax
    float new_absmax_z = absmax_z[qblock_id];
    float new_absmax_sq = absmax2[qblock_id];

    // ============================================================
    // Step 8: Re-quantize z and exp_avg_sq
    // ============================================================

    // Normalize to [-1, 1] (z) or [0, 1] (exp_avg_sq)
    float normalized_z = (new_absmax_z > 0.0f) ? (z / new_absmax_z) : 0.0f;
    float normalized_sq = (new_absmax_sq > 0.0f) ? (exp_avg_sq / new_absmax_sq) : 0.0f;

    // Quantize using binary search
    uint8_t new_qz = quantize_value(normalized_z, d_qmap_signed);
    uint8_t new_q2 = quantize_value(normalized_sq, d_qmap_unsigned);

    // Write back quantized states
    state_z[tid] = new_qz;
    state_exp_avg_sq[tid] = new_q2;

    // ============================================================
    // Step 9: Write back updated y to param
    // ============================================================

    if constexpr (std::is_same<T, float>::value) {
        param[tid] = y;
    } else if constexpr (std::is_same<T, __half>::value) {
        param[tid] = __float2half(y);
    } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
        param[tid] = __float2bfloat16(y);
    }
}


// ============================================================
// Kernel Launcher (Template Instantiation)
// ============================================================

// Explicit instantiation for FP32
template __global__ void adamw_8bit_schedulefree_update_kernel<float>(
    float*, const float*, uint8_t*, uint8_t*, float*, float*,
    const float, const float, const float, const float, const float,
    const float, const float, const float, const int
);

// Explicit instantiation for FP16
template __global__ void adamw_8bit_schedulefree_update_kernel<__half>(
    __half*, const __half*, uint8_t*, uint8_t*, float*, float*,
    const float, const float, const float, const float, const float,
    const float, const float, const float, const int
);

// Explicit instantiation for BF16
template __global__ void adamw_8bit_schedulefree_update_kernel<__nv_bfloat16>(
    __nv_bfloat16*, const __nv_bfloat16*, uint8_t*, uint8_t*, float*, float*,
    const float, const float, const float, const float, const float,
    const float, const float, const float, const int
);
