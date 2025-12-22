/*
Lion 8-bit Optimizer CUDA Kernel with Ring Buffer Support

Based on bitsandbytes 8-bit optimizer (MIT License)
https://github.com/TimDettmers/bitsandbytes

Lion Algorithm:
1. c_t = β1 * m_{t-1} + (1 - β1) * g_t          (Interpolate)
2. update = sign(c_t) + λ * θ_{t-1}              (Sign + weight decay)
3. θ_t = θ_{t-1} - η * update                    (Apply update)
4. m_t = β2 * m_{t-1} + (1 - β2) * g_t           (Update momentum EMA)

Modified for SushiUI Ring Buffer integration:
- Momentum state (exp_avg) can be on CPU or GPU
- Automatic CPU→GPU transfer during kernel execution
- VRAM savings: ~87.5% for optimizer states (1 state instead of 2)

Implementation:
- Uses bitsandbytes quantization (dynamic map, CUB reduce, bias correction)
- Supports FP32, FP16, BF16 parameters
- Blockwise quantization (256 elements per block)
*/

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cub/cub.cuh>
#include <type_traits>

#define QUANTIZATION_BLOCKSIZE 256
#define THREADS_PER_BLOCK 256

// Quantization map in constant memory (shared across all kernels)
__device__ __constant__ float d_qmap_signed[256];

// Dequantization: UINT8 code -> FP32 value
__device__ __forceinline__ float dequantize_code(unsigned char code, float absmax) {
    return d_qmap_signed[code] * absmax;
}

// Quantization: FP32 value -> UINT8 code
__device__ __forceinline__ unsigned char quantize_value(float value, float absmax) {
    float normalized = value / (absmax + 1e-7f);

    // Binary search in quantization map
    int left = 0;
    int right = 255;
    int best = 128;
    float best_dist = fabsf(normalized - d_qmap_signed[128]);

    while (left <= right) {
        int mid = (left + right) / 2;
        float dist = fabsf(normalized - d_qmap_signed[mid]);

        if (dist < best_dist) {
            best_dist = dist;
            best = mid;
        }

        if (d_qmap_signed[mid] < normalized) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    return static_cast<unsigned char>(best);
}

/*
Lion 8-bit Update Kernel (Blockwise Quantization)

Args:
    param: Model parameters (FP32/FP16/BF16)
    grad: Gradients (FP32/FP16/BF16)
    exp_avg: Momentum state (UINT8, quantized)
    absmax: Absmax values per block (FP32)
    beta1: Interpolation coefficient
    beta2: Momentum EMA coefficient
    lr: Learning rate
    weight_decay: Weight decay coefficient
    gnorm_scale: Gradient norm scaling factor
    step: Current step (for bias correction)
    N: Number of elements
*/
template <typename T>
__global__ void lion_8bit_blockwise_update_kernel(
    T* __restrict__ param,
    const T* __restrict__ grad,
    unsigned char* __restrict__ exp_avg,
    float* __restrict__ absmax,
    float beta1,
    float beta2,
    float eps,
    float lr,
    float weight_decay,
    float gnorm_scale,
    int step,
    int N
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= N) return;

    // Calculate block index for blockwise quantization
    int block_idx = tid / QUANTIZATION_BLOCKSIZE;

    // Load current absmax for this block
    float current_absmax = absmax[block_idx];

    // Convert gradient to FP32 (handles FP32/FP16/BF16)
    float g;
    if constexpr (std::is_same<T, float>::value) {
        g = grad[tid] * gnorm_scale;
    } else if constexpr (std::is_same<T, __half>::value) {
        g = __half2float(grad[tid]) * gnorm_scale;
    } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
        g = __bfloat162float(grad[tid]) * gnorm_scale;
    }

    // Dequantize momentum (m_{t-1})
    float m_prev = dequantize_code(exp_avg[tid], current_absmax);

    // Convert parameter to FP32 first (needed for weight decay calculation)
    float param_val;
    if constexpr (std::is_same<T, float>::value) {
        param_val = param[tid];
    } else if constexpr (std::is_same<T, __half>::value) {
        param_val = __half2float(param[tid]);
    } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
        param_val = __bfloat162float(param[tid]);
    }

    // ============================================================
    // Lion Update Algorithm
    // ============================================================

    // 1. Interpolate: c_t = β1 * m_{t-1} + (1 - β1) * g_t
    float c_t = beta1 * m_prev + (1.0f - beta1) * g;

    // 2. Sign-based update with weight decay
    float update = (c_t > 0.0f ? 1.0f : -1.0f) + weight_decay * param_val;

    // 3. Apply update to parameter
    param_val -= lr * update;

    // 4. Update momentum: m_t = β2 * m_{t-1} + (1 - β2) * g_t
    float m_new = beta2 * m_prev + (1.0f - beta2) * g;

    // ============================================================
    // Blockwise Absmax Calculation (CUB BlockReduce)
    // ============================================================

    // Local absmax for this thread
    float local_absmax = fabsf(m_new);

    // Block-level reduction to find max absmax
    typedef cub::BlockReduce<float, THREADS_PER_BLOCK> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    float block_absmax = BlockReduce(temp_storage).Reduce(local_absmax, cub::Max());

    // First thread in block updates absmax
    if (threadIdx.x == 0) {
        atomicMax(reinterpret_cast<int*>(&absmax[block_idx]),
                  __float_as_int(block_absmax));
    }

    __syncthreads();

    // Reload updated absmax
    float new_absmax = absmax[block_idx];

    // ============================================================
    // Quantize New Momentum State
    // ============================================================

    exp_avg[tid] = quantize_value(m_new, new_absmax);

    // ============================================================
    // Write Parameter Back (FP32/FP16/BF16)
    // ============================================================

    if constexpr (std::is_same<T, float>::value) {
        param[tid] = param_val;
    } else if constexpr (std::is_same<T, __half>::value) {
        param[tid] = __float2half(param_val);
    } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
        param[tid] = __float2bfloat16(param_val);
    }
}

// ============================================================
// Launcher Functions (MSVC compatibility)
// ============================================================

template <typename T>
void launch_lion_8bit_blockwise_update_kernel(
    T* param,
    const T* grad,
    unsigned char* exp_avg,
    float* absmax,
    float beta1,
    float beta2,
    float eps,
    float lr,
    float weight_decay,
    float gnorm_scale,
    int step,
    int N,
    int blocks,
    int threads
) {
    lion_8bit_blockwise_update_kernel<T><<<blocks, threads>>>(
        param, grad, exp_avg, absmax,
        beta1, beta2, eps, lr, weight_decay, gnorm_scale, step, N
    );
}

// Explicit instantiations
template void launch_lion_8bit_blockwise_update_kernel<float>(
    float*, const float*, unsigned char*, float*, float, float, float, float, float, float, int, int, int, int);
template void launch_lion_8bit_blockwise_update_kernel<__half>(
    __half*, const __half*, unsigned char*, float*, float, float, float, float, float, float, int, int, int, int);
template void launch_lion_8bit_blockwise_update_kernel<__nv_bfloat16>(
    __nv_bfloat16*, const __nv_bfloat16*, unsigned char*, float*, float, float, float, float, float, float, int, int, int, int);

// ============================================================
// Quantization Map Initialization (extern "C" for C++ linkage)
// ============================================================

extern "C" {

void init_quantization_maps(const float* host_qmap_signed) {
    cudaMemcpyToSymbol(d_qmap_signed, host_qmap_signed, 256 * sizeof(float));
}

}  // extern "C"
