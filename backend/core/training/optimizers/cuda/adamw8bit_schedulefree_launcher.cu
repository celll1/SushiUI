/*
 * CUDA Kernel Launcher for AdamW 8-bit Schedule-Free
 *
 * Handles kernel launch configuration and grid/block dimensions.
 *
 * NOTE: We include the kernel definition file here because CUDA template
 * functions must have their definitions visible at the call site.
 */

// Include kernel implementation (contains template definition)
#include "adamw8bit_schedulefree_kernel.cu"


// ============================================================
// Kernel Launchers (C API for Python binding)
// ============================================================

extern "C" {

/*
 * Fill THIS translation unit's quantization maps.
 *
 * ``__constant__`` symbols are per translation unit. adamw8bit_kernel.cu
 * declares its own ``d_qmap_signed`` / ``d_qmap_unsigned`` and its
 * ``init_quantization_maps()`` writes only those, so the copies the
 * Schedule-Free kernel reads (declared in adamw8bit_schedulefree_kernel.cu,
 * included above) stayed ZERO for the life of the process. Every
 * ``dequantize_value()`` in the Schedule-Free kernel therefore returned 0: the
 * z sequence and exp_avg_sq were read back as zero on every step, whatever had
 * been stored in them. Verified on a real device -- one step collapsed
 * absmax_z from 6.8e-2 to 1.0e-5 (i.e. exactly ``lr``, the update applied to a
 * z of zero) and drove every z code to 255.
 *
 * The wrapper calls this alongside the plain-AdamW one so both copies are
 * initialised from the same host maps.
 */
void init_schedulefree_quantization_maps(
    const float* host_qmap_signed, const float* host_qmap_unsigned
) {
    cudaMemcpyToSymbol(d_qmap_signed, host_qmap_signed, 256 * sizeof(float));
    cudaMemcpyToSymbol(d_qmap_unsigned, host_qmap_unsigned, 256 * sizeof(float));
}

void adamw_8bit_schedulefree_update_fp32(
    float* param, const float* grad,
    uint8_t* state_z, uint8_t* state_exp_avg_sq,
    float* absmax_z, float* absmax2,
    float beta1, float beta2, float eps, float lr,
    float weight_decay, float ckp1, float gnorm_scale,
    float bias_correction2, int numel,
    int stochastic_z, unsigned int seed,
    cudaStream_t stream
) {
    int num_blocks = (numel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    adamw_8bit_schedulefree_update_kernel<float><<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        param, grad, state_z, state_exp_avg_sq, absmax_z, absmax2,
        beta1, beta2, eps, lr, weight_decay, ckp1, gnorm_scale, bias_correction2, numel,
        stochastic_z, seed
    );
}

void adamw_8bit_schedulefree_update_fp16(
    __half* param, const __half* grad,
    uint8_t* state_z, uint8_t* state_exp_avg_sq,
    float* absmax_z, float* absmax2,
    float beta1, float beta2, float eps, float lr,
    float weight_decay, float ckp1, float gnorm_scale,
    float bias_correction2, int numel,
    int stochastic_z, unsigned int seed,
    cudaStream_t stream
) {
    int num_blocks = (numel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    adamw_8bit_schedulefree_update_kernel<__half><<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        param, grad, state_z, state_exp_avg_sq, absmax_z, absmax2,
        beta1, beta2, eps, lr, weight_decay, ckp1, gnorm_scale, bias_correction2, numel,
        stochastic_z, seed
    );
}

void adamw_8bit_schedulefree_update_bf16(
    __nv_bfloat16* param, const __nv_bfloat16* grad,
    uint8_t* state_z, uint8_t* state_exp_avg_sq,
    float* absmax_z, float* absmax2,
    float beta1, float beta2, float eps, float lr,
    float weight_decay, float ckp1, float gnorm_scale,
    float bias_correction2, int numel,
    int stochastic_z, unsigned int seed,
    cudaStream_t stream
) {
    int num_blocks = (numel + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    adamw_8bit_schedulefree_update_kernel<__nv_bfloat16><<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        param, grad, state_z, state_exp_avg_sq, absmax_z, absmax2,
        beta1, beta2, eps, lr, weight_decay, ckp1, gnorm_scale, bias_correction2, numel,
        stochastic_z, seed
    );
}

}  // extern "C"
