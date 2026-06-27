/*
 * PyTorch C++ Extension for AdamW 8-bit CUDA Kernels
 *
 * Ported from bitsandbytes (MIT License)
 * Modified for Ring Buffer compatibility (CPU-allocated optimizer states)
 *
 * Key features:
 * - Automatic CPU→GPU transfer for Ring Buffer-allocated states
 * - Exact bitsandbytes quantization algorithm
 * - FP32/FP16/BF16 parameter support
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

// Forward declarations of CUDA kernel wrappers
extern "C" {
    void adamw_8bit_update_fp32(
        float* param, const float* grad,
        uint8_t* state1, uint8_t* state2,
        float* absmax1, float* absmax2,
        float beta1, float beta2, float eps, float lr,
        float weight_decay, float gnorm_scale, int step, bool cautious, int numel,
        cudaStream_t stream
    );

    void adamw_8bit_update_fp16(
        at::Half* param, const at::Half* grad,
        uint8_t* state1, uint8_t* state2,
        float* absmax1, float* absmax2,
        float beta1, float beta2, float eps, float lr,
        float weight_decay, float gnorm_scale, int step, bool cautious, int numel,
        cudaStream_t stream
    );

    void adamw_8bit_update_bf16(
        at::BFloat16* param, const at::BFloat16* grad,
        uint8_t* state1, uint8_t* state2,
        float* absmax1, float* absmax2,
        float beta1, float beta2, float eps, float lr,
        float weight_decay, float gnorm_scale, int step, bool cautious, int numel,
        cudaStream_t stream
    );

    void init_quantization_maps(const float* host_qmap_signed, const float* host_qmap_unsigned);

    // Schedule-Free kernels
    void adamw_8bit_schedulefree_update_fp32(
        float* param, const float* grad,
        uint8_t* state_z, uint8_t* state_exp_avg_sq,
        float* absmax_z, float* absmax2,
        float beta1, float beta2, float eps, float lr,
        float weight_decay, float ckp1, float gnorm_scale,
        float bias_correction2, int numel,
        cudaStream_t stream
    );

    void adamw_8bit_schedulefree_update_fp16(
        at::Half* param, const at::Half* grad,
        uint8_t* state_z, uint8_t* state_exp_avg_sq,
        float* absmax_z, float* absmax2,
        float beta1, float beta2, float eps, float lr,
        float weight_decay, float ckp1, float gnorm_scale,
        float bias_correction2, int numel,
        cudaStream_t stream
    );

    void adamw_8bit_schedulefree_update_bf16(
        at::BFloat16* param, const at::BFloat16* grad,
        uint8_t* state_z, uint8_t* state_exp_avg_sq,
        float* absmax_z, float* absmax2,
        float beta1, float beta2, float eps, float lr,
        float weight_decay, float ckp1, float gnorm_scale,
        float bias_correction2, int numel,
        cudaStream_t stream
    );
}


// ============================================================
// PyTorch Wrapper: AdamW 8-bit Update
// ============================================================

void adamw_8bit_update(
    torch::Tensor param,
    torch::Tensor grad,
    torch::Tensor state1,
    torch::Tensor state2,
    torch::Tensor absmax1,
    torch::Tensor absmax2,
    float beta1,
    float beta2,
    float eps,
    float lr,
    float weight_decay,
    float gnorm_scale,
    int step,
    bool cautious
) {
    // Input validation
    TORCH_CHECK(param.is_cuda(), "Param must be on CUDA device");
    TORCH_CHECK(grad.is_cuda(), "Grad must be on CUDA device");
    TORCH_CHECK(param.dtype() == grad.dtype(), "Param and Grad must have same dtype");
    TORCH_CHECK(state1.dtype() == torch::kUInt8, "State1 must be UINT8");
    TORCH_CHECK(state2.dtype() == torch::kUInt8, "State2 must be UINT8");
    TORCH_CHECK(absmax1.dtype() == torch::kFloat32, "Absmax1 must be FP32");
    TORCH_CHECK(absmax2.dtype() == torch::kFloat32, "Absmax2 must be FP32");
    TORCH_CHECK(absmax1.is_cuda(), "Absmax1 must be on CUDA device");
    TORCH_CHECK(absmax2.is_cuda(), "Absmax2 must be on CUDA device");

    int numel = param.numel();

    // ============================================================
    // Ring Buffer Support: CPU→GPU Transfer
    // ============================================================

    torch::Tensor state1_gpu = state1;
    torch::Tensor state2_gpu = state2;

    bool state1_is_cpu = !state1.is_cuda();
    bool state2_is_cpu = !state2.is_cuda();

    // Move states to GPU if on CPU (Ring Buffer case)
    if (state1_is_cpu) {
        state1_gpu = state1.to(param.device(), /*non_blocking=*/true);
    }
    if (state2_is_cpu) {
        state2_gpu = state2.to(param.device(), /*non_blocking=*/true);
    }

    // Get CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(param.device().index()).stream();

    // ============================================================
    // Dispatch based on parameter dtype
    // ============================================================

    if (param.dtype() == torch::kFloat32) {
        adamw_8bit_update_fp32(
            param.data_ptr<float>(),
            grad.data_ptr<float>(),
            state1_gpu.data_ptr<uint8_t>(),
            state2_gpu.data_ptr<uint8_t>(),
            absmax1.data_ptr<float>(),
            absmax2.data_ptr<float>(),
            beta1, beta2, eps, lr, weight_decay, gnorm_scale, step, cautious, numel,
            stream
        );
    } else if (param.dtype() == torch::kFloat16) {
        adamw_8bit_update_fp16(
            reinterpret_cast<at::Half*>(param.data_ptr<at::Half>()),
            reinterpret_cast<const at::Half*>(grad.data_ptr<at::Half>()),
            state1_gpu.data_ptr<uint8_t>(),
            state2_gpu.data_ptr<uint8_t>(),
            absmax1.data_ptr<float>(),
            absmax2.data_ptr<float>(),
            beta1, beta2, eps, lr, weight_decay, gnorm_scale, step, cautious, numel,
            stream
        );
    } else if (param.dtype() == torch::kBFloat16) {
        adamw_8bit_update_bf16(
            reinterpret_cast<at::BFloat16*>(param.data_ptr<at::BFloat16>()),
            reinterpret_cast<const at::BFloat16*>(grad.data_ptr<at::BFloat16>()),
            state1_gpu.data_ptr<uint8_t>(),
            state2_gpu.data_ptr<uint8_t>(),
            absmax1.data_ptr<float>(),
            absmax2.data_ptr<float>(),
            beta1, beta2, eps, lr, weight_decay, gnorm_scale, step, cautious, numel,
            stream
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype. Must be FP32, FP16, or BF16");
    }

    // ============================================================
    // Copy updated states back to CPU (Ring Buffer case)
    // ============================================================
    //
    // H2D, the update kernel and D2H are all issued on the SAME current CUDA
    // stream, so they execute in order; the writeback also precedes the next
    // step's H2D of the same (pinned) CPU buffer on that stream. The previous
    // per-parameter cudaStreamSynchronize() therefore was not needed for
    // correctness -- it forced a CPU<->GPU lockstep on every parameter,
    // draining the whole pipeline thousands of times per step. Issue the
    // writeback asynchronously (pinned CPU buffers) and let it overlap.
    if (state1_is_cpu) {
        state1.copy_(state1_gpu, /*non_blocking=*/true);
    }
    if (state2_is_cpu) {
        state2.copy_(state2_gpu, /*non_blocking=*/true);
    }
}


// ============================================================
// PyTorch Wrapper: Initialize Quantization Maps
// ============================================================

void init_quantization_maps_wrapper(torch::Tensor qmap_signed, torch::Tensor qmap_unsigned) {
    // Validation
    TORCH_CHECK(qmap_signed.dtype() == torch::kFloat32, "Signed qmap must be FP32");
    TORCH_CHECK(qmap_unsigned.dtype() == torch::kFloat32, "Unsigned qmap must be FP32");
    TORCH_CHECK(qmap_signed.numel() == 256, "Signed qmap must have 256 elements");
    TORCH_CHECK(qmap_unsigned.numel() == 256, "Unsigned qmap must have 256 elements");
    TORCH_CHECK(qmap_signed.is_contiguous(), "Signed qmap must be contiguous");
    TORCH_CHECK(qmap_unsigned.is_contiguous(), "Unsigned qmap must be contiguous");
    TORCH_CHECK(qmap_signed.device().is_cpu(), "Signed qmap must be on CPU");
    TORCH_CHECK(qmap_unsigned.device().is_cpu(), "Unsigned qmap must be on CPU");

    // Copy quantization maps to device constant memory
    init_quantization_maps(
        qmap_signed.data_ptr<float>(),
        qmap_unsigned.data_ptr<float>()
    );
}


// ============================================================
// PyTorch Wrapper: AdamW 8-bit Schedule-Free Update
// ============================================================

void adamw_8bit_schedulefree_update(
    torch::Tensor param,
    torch::Tensor grad,
    torch::Tensor state_z,
    torch::Tensor state_exp_avg_sq,
    torch::Tensor absmax_z,
    torch::Tensor absmax2,
    float beta1,
    float beta2,
    float eps,
    float lr,
    float weight_decay,
    float ckp1,
    float gnorm_scale,
    float bias_correction2
) {
    // Input validation
    TORCH_CHECK(param.is_cuda(), "Param must be on CUDA device");
    TORCH_CHECK(grad.is_cuda(), "Grad must be on CUDA device");
    TORCH_CHECK(param.dtype() == grad.dtype(), "Param and Grad must have same dtype");
    TORCH_CHECK(state_z.dtype() == torch::kUInt8, "State z must be UINT8");
    TORCH_CHECK(state_exp_avg_sq.dtype() == torch::kUInt8, "State exp_avg_sq must be UINT8");
    TORCH_CHECK(absmax_z.dtype() == torch::kFloat32, "Absmax z must be FP32");
    TORCH_CHECK(absmax2.dtype() == torch::kFloat32, "Absmax2 must be FP32");
    TORCH_CHECK(absmax_z.is_cuda(), "Absmax z must be on CUDA device");
    TORCH_CHECK(absmax2.is_cuda(), "Absmax2 must be on CUDA device");

    int numel = param.numel();

    // ============================================================
    // Ring Buffer Support: CPU→GPU Transfer
    // ============================================================

    torch::Tensor state_z_gpu = state_z;
    torch::Tensor state_exp_avg_sq_gpu = state_exp_avg_sq;

    bool state_z_is_cpu = !state_z.is_cuda();
    bool state_exp_avg_sq_is_cpu = !state_exp_avg_sq.is_cuda();

    // Move states to GPU if on CPU (Ring Buffer case)
    if (state_z_is_cpu) {
        state_z_gpu = state_z.to(param.device(), /*non_blocking=*/true);
    }
    if (state_exp_avg_sq_is_cpu) {
        state_exp_avg_sq_gpu = state_exp_avg_sq.to(param.device(), /*non_blocking=*/true);
    }

    // Get CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // ============================================================
    // Call CUDA kernel (dtype dispatch)
    // ============================================================

    if (param.dtype() == torch::kFloat32) {
        adamw_8bit_schedulefree_update_fp32(
            param.data_ptr<float>(),
            grad.data_ptr<float>(),
            state_z_gpu.data_ptr<uint8_t>(),
            state_exp_avg_sq_gpu.data_ptr<uint8_t>(),
            absmax_z.data_ptr<float>(),
            absmax2.data_ptr<float>(),
            beta1, beta2, eps, lr, weight_decay, ckp1, gnorm_scale, bias_correction2,
            numel, stream
        );
    } else if (param.dtype() == torch::kFloat16) {
        adamw_8bit_schedulefree_update_fp16(
            reinterpret_cast<at::Half*>(param.data_ptr<at::Half>()),
            reinterpret_cast<const at::Half*>(grad.data_ptr<at::Half>()),
            state_z_gpu.data_ptr<uint8_t>(),
            state_exp_avg_sq_gpu.data_ptr<uint8_t>(),
            absmax_z.data_ptr<float>(),
            absmax2.data_ptr<float>(),
            beta1, beta2, eps, lr, weight_decay, ckp1, gnorm_scale, bias_correction2,
            numel, stream
        );
    } else if (param.dtype() == torch::kBFloat16) {
        adamw_8bit_schedulefree_update_bf16(
            reinterpret_cast<at::BFloat16*>(param.data_ptr<at::BFloat16>()),
            reinterpret_cast<const at::BFloat16*>(grad.data_ptr<at::BFloat16>()),
            state_z_gpu.data_ptr<uint8_t>(),
            state_exp_avg_sq_gpu.data_ptr<uint8_t>(),
            absmax_z.data_ptr<float>(),
            absmax2.data_ptr<float>(),
            beta1, beta2, eps, lr, weight_decay, ckp1, gnorm_scale, bias_correction2,
            numel, stream
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype (must be FP32, FP16, or BF16)");
    }

    // ============================================================
    // Ring Buffer Support: GPU→CPU Copy-back
    // ============================================================

    if (state_z_is_cpu) {
        state_z.copy_(state_z_gpu, /*non_blocking=*/true);
    }
    if (state_exp_avg_sq_is_cpu) {
        state_exp_avg_sq.copy_(state_exp_avg_sq_gpu, /*non_blocking=*/true);
    }

    // No per-parameter cudaStreamSynchronize: H2D, kernel and D2H are all on the
    // current stream and thus ordered with the surrounding backward and with the
    // next step's H2D of the same pinned buffer. Removing the per-parameter
    // lockstep recovers pipeline throughput. (See note in adamw_8bit_update.)
}


// ============================================================
// Python Bindings
// ============================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "adamw_8bit_update",
        &adamw_8bit_update,
        "AdamW update with 8-bit blockwise quantized states (CUDA, Ring Buffer compatible)"
    );
    m.def(
        "adamw_8bit_schedulefree_update",
        &adamw_8bit_schedulefree_update,
        "AdamW Schedule-Free update with 8-bit quantized z and exp_avg_sq (CUDA, Ring Buffer compatible)"
    );
    m.def(
        "init_quantization_maps",
        &init_quantization_maps_wrapper,
        "Initialize quantization maps on device (signed and unsigned)"
    );
}
