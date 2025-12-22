/*
 * AdamW 8-bit Blockwise Quantization CUDA Kernels
 *
 * Ported from bitsandbytes (MIT License)
 * https://github.com/TimDettmers/bitsandbytes
 *
 * Modified for Ring Buffer compatibility:
 * - Supports CPU-allocated tensors (optimizer states)
 * - Maintains exact bitsandbytes quantization algorithm
 * - Uses dynamic quantization map and CUB BlockReduce
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cub/cub.cuh>
#include <stdint.h>
#include <stdio.h>

// Quantization block size (must match bitsandbytes)
#define QUANTIZATION_BLOCKSIZE 256
#define THREADS_PER_BLOCK 256

// ============================================================
// Dynamic Quantization Maps (Device Constants)
// ============================================================

// Signed map for exp_avg (momentum): [-1.0, 1.0]
__device__ __constant__ float d_qmap_signed[256];

// Unsigned map for exp_avg_sq (variance): [0.0, 1.0]
__device__ __constant__ float d_qmap_unsigned[256];


// ============================================================
// Quantization/Dequantization (from bitsandbytes)
// ============================================================

/*
 * Quantize a normalized value [-1, 1] or [0, 1] to uint8 index.
 * Uses binary search in quantization map.
 *
 * Ported from bitsandbytes quantize_2D()
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
// AdamW 8-bit Update Kernel
// ============================================================

template<typename T>
__global__ void adamw_8bit_update_kernel(
    T* __restrict__ param,                  // [numel] Parameters (FP32/FP16/BF16, GPU)
    const T* __restrict__ grad,             // [numel] Gradients (same dtype as param, GPU)
    uint8_t* __restrict__ state1,           // [numel] exp_avg quantized (UINT8, GPU)
    uint8_t* __restrict__ state2,           // [numel] exp_avg_sq quantized (UINT8, GPU)
    float* __restrict__ absmax1,            // [num_blocks] exp_avg absmax (FP32, GPU)
    float* __restrict__ absmax2,            // [num_blocks] exp_avg_sq absmax (FP32, GPU)
    const float beta1,                      // AdamW beta1 (0.9)
    const float beta2,                      // AdamW beta2 (0.999)
    const float eps,                        // AdamW epsilon (1e-8)
    const float lr,                         // Learning rate
    const float weight_decay,               // Weight decay (0.01)
    const float gnorm_scale,                // Gradient norm scaling (1.0 if no clipping)
    const int step,                         // Current step (for bias correction)
    const int numel                         // Total number of elements
) {
    // Thread and block indices
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_tid = threadIdx.x;

    // Quantization block index (256 elements per quantization block)
    const int qblock_id = tid / QUANTIZATION_BLOCKSIZE;

    if (tid >= numel) return;

    // ============================================================
    // Step 1: Dequantize optimizer states
    // ============================================================

    float current_absmax1 = absmax1[qblock_id];
    float current_absmax2 = absmax2[qblock_id];

    uint8_t q1 = state1[tid];
    uint8_t q2 = state2[tid];

    float exp_avg = dequantize_value(q1, d_qmap_signed, current_absmax1);
    float exp_avg_sq = dequantize_value(q2, d_qmap_unsigned, current_absmax2);

    // ============================================================
    // Step 2: Update momentum (FP32 for numerical stability)
    // ============================================================

    float g = (float)grad[tid] * gnorm_scale;

    exp_avg = beta1 * exp_avg + (1.0f - beta1) * g;
    exp_avg_sq = beta2 * exp_avg_sq + (1.0f - beta2) * (g * g);

    // ============================================================
    // Step 3: Compute block-level absmax using CUB BlockReduce
    // ============================================================

    // Each quantization block (256 elements) is processed by one CUDA block
    // This ensures no race conditions

    typedef cub::BlockReduce<float, THREADS_PER_BLOCK> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage1;
    __shared__ typename BlockReduce::TempStorage temp_storage2;

    float local_absmax1 = fabsf(exp_avg);
    float local_absmax2 = fabsf(exp_avg_sq);

    // Block-level reduction (NO ATOMIC OPERATIONS!)
    float block_absmax1 = BlockReduce(temp_storage1).Reduce(local_absmax1, cub::Max());
    __syncthreads();
    float block_absmax2 = BlockReduce(temp_storage2).Reduce(local_absmax2, cub::Max());
    __syncthreads();

    // Only first thread in block writes absmax (NO RACE CONDITION!)
    if (local_tid == 0) {
        // Clamp to avoid division by zero
        absmax1[qblock_id] = fmaxf(block_absmax1, 1e-12f);
        absmax2[qblock_id] = fmaxf(block_absmax2, 1e-12f);
    }
    __syncthreads();

    // ============================================================
    // Step 4: Quantize with updated absmax
    // ============================================================

    float new_absmax1 = absmax1[qblock_id];
    float new_absmax2 = absmax2[qblock_id];

    // Normalize to [-1, 1] / [0, 1]
    float normalized1 = exp_avg / new_absmax1;
    float normalized2 = exp_avg_sq / new_absmax2;

    // Clamp to valid range
    normalized1 = fmaxf(-1.0f, fminf(1.0f, normalized1));
    normalized2 = fmaxf(0.0f, fminf(1.0f, normalized2));

    // Quantize using dynamic map
    state1[tid] = quantize_value(normalized1, d_qmap_signed);
    state2[tid] = quantize_value(normalized2, d_qmap_unsigned);

    // ============================================================
    // Step 5: Bias correction (AdamW)
    // ============================================================

    float bias_correction1 = 1.0f - powf(beta1, (float)step);
    float bias_correction2 = sqrtf(1.0f - powf(beta2, (float)step));

    float corrected_exp_avg = exp_avg / bias_correction1;
    float corrected_exp_avg_sq_sqrt = sqrtf(exp_avg_sq) / bias_correction2;

    // ============================================================
    // Step 6: Update parameter (AdamW with decoupled weight decay)
    // ============================================================

    float denom = corrected_exp_avg_sq_sqrt + eps;
    float update = corrected_exp_avg / denom;

    float param_val = (float)param[tid];

    // Decoupled weight decay (applied to parameter directly)
    if (weight_decay > 0.0f) {
        param_val = param_val * (1.0f - lr * weight_decay);
    }

    // Apply AdamW update
    param_val = param_val - lr * update;

    param[tid] = (T)param_val;
}


// ============================================================
// Host Functions (C++ Callable)
// ============================================================

extern "C" {

// FP32 version
void adamw_8bit_update_fp32(
    float* param,
    const float* grad,
    uint8_t* state1,
    uint8_t* state2,
    float* absmax1,
    float* absmax2,
    float beta1,
    float beta2,
    float eps,
    float lr,
    float weight_decay,
    float gnorm_scale,
    int step,
    int numel,
    cudaStream_t stream
) {
    // Launch with QUANTIZATION_BLOCKSIZE threads per block
    // This ensures each quantization block is processed by one CUDA block
    int num_cuda_blocks = (numel + QUANTIZATION_BLOCKSIZE - 1) / QUANTIZATION_BLOCKSIZE;

    adamw_8bit_update_kernel<float><<<num_cuda_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        param, grad, state1, state2, absmax1, absmax2,
        beta1, beta2, eps, lr, weight_decay, gnorm_scale, step, numel
    );
}

// FP16 version
void adamw_8bit_update_fp16(
    __half* param,
    const __half* grad,
    uint8_t* state1,
    uint8_t* state2,
    float* absmax1,
    float* absmax2,
    float beta1,
    float beta2,
    float eps,
    float lr,
    float weight_decay,
    float gnorm_scale,
    int step,
    int numel,
    cudaStream_t stream
) {
    int num_cuda_blocks = (numel + QUANTIZATION_BLOCKSIZE - 1) / QUANTIZATION_BLOCKSIZE;

    adamw_8bit_update_kernel<__half><<<num_cuda_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        param, grad, state1, state2, absmax1, absmax2,
        beta1, beta2, eps, lr, weight_decay, gnorm_scale, step, numel
    );
}

// BF16 version
void adamw_8bit_update_bf16(
    __nv_bfloat16* param,
    const __nv_bfloat16* grad,
    uint8_t* state1,
    uint8_t* state2,
    float* absmax1,
    float* absmax2,
    float beta1,
    float beta2,
    float eps,
    float lr,
    float weight_decay,
    float gnorm_scale,
    int step,
    int numel,
    cudaStream_t stream
) {
    int num_cuda_blocks = (numel + QUANTIZATION_BLOCKSIZE - 1) / QUANTIZATION_BLOCKSIZE;

    adamw_8bit_update_kernel<__nv_bfloat16><<<num_cuda_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        param, grad, state1, state2, absmax1, absmax2,
        beta1, beta2, eps, lr, weight_decay, gnorm_scale, step, numel
    );
}

// Initialize quantization maps on device
void init_quantization_maps(const float* host_qmap_signed, const float* host_qmap_unsigned) {
    cudaMemcpyToSymbol(d_qmap_signed, host_qmap_signed, 256 * sizeof(float));
    cudaMemcpyToSymbol(d_qmap_unsigned, host_qmap_unsigned, 256 * sizeof(float));
}

}  // extern "C"
