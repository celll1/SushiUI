/*
 * CUDA Kernel Launcher for AdamW 8-bit Schedule-Free
 *
 * Handles kernel launch configuration and grid/block dimensions.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

#define THREADS_PER_BLOCK 256

// Forward declare kernel template
template<typename T>
__global__ void adamw_8bit_schedulefree_update_kernel(
    T* param, const T* grad,
    uint8_t* state_z, uint8_t* state_exp_avg_sq,
    float* absmax_z, float* absmax2,
    const float beta1, const float beta2, const float eps, const float lr,
    const float weight_decay, const float ckp1, const float gnorm_scale,
    const float bias_correction2, const int numel
);


// ============================================================
// Kernel Launchers (C API for Python binding)
// ============================================================

extern "C" {

void adamw_8bit_schedulefree_update_fp32(
    float* param, const float* grad,
    uint8_t* state_z, uint8_t* state_exp_avg_sq,
    float* absmax_z, float* absmax2,
    float beta1, float beta2, float eps, float lr,
    float weight_decay, float ckp1, float gnorm_scale,
    float bias_correction2, int numel,
    cudaStream_t stream
) {
    int num_blocks = (numel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    adamw_8bit_schedulefree_update_kernel<float><<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        param, grad, state_z, state_exp_avg_sq, absmax_z, absmax2,
        beta1, beta2, eps, lr, weight_decay, ckp1, gnorm_scale, bias_correction2, numel
    );
}

void adamw_8bit_schedulefree_update_fp16(
    __half* param, const __half* grad,
    uint8_t* state_z, uint8_t* state_exp_avg_sq,
    float* absmax_z, float* absmax2,
    float beta1, float beta2, float eps, float lr,
    float weight_decay, float ckp1, float gnorm_scale,
    float bias_correction2, int numel,
    cudaStream_t stream
) {
    int num_blocks = (numel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    adamw_8bit_schedulefree_update_kernel<__half><<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        param, grad, state_z, state_exp_avg_sq, absmax_z, absmax2,
        beta1, beta2, eps, lr, weight_decay, ckp1, gnorm_scale, bias_correction2, numel
    );
}

void adamw_8bit_schedulefree_update_bf16(
    __nv_bfloat16* param, const __nv_bfloat16* grad,
    uint8_t* state_z, uint8_t* state_exp_avg_sq,
    float* absmax_z, float* absmax2,
    float beta1, float beta2, float eps, float lr,
    float weight_decay, float ckp1, float gnorm_scale,
    float bias_correction2, int numel,
    cudaStream_t stream
) {
    int num_blocks = (numel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    adamw_8bit_schedulefree_update_kernel<__nv_bfloat16><<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        param, grad, state_z, state_exp_avg_sq, absmax_z, absmax2,
        beta1, beta2, eps, lr, weight_decay, ckp1, gnorm_scale, bias_correction2, numel
    );
}

}  // extern "C"
