/*
Lion 8-bit CUDA Extension (C++ Wrapper)

Based on bitsandbytes 8-bit optimizer (MIT License)
PyTorch C++ extension for Lion 8-bit optimizer with Ring Buffer support.

Functions:
- init_quantization_maps: Initialize quantization map in constant memory
- lion_8bit_update: Perform Lion update with CPU→GPU state transfer
*/

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Forward declarations of CUDA functions (defined in .cu files)
extern "C" {
    void init_quantization_maps(const float* host_qmap_signed);

    // Schedule-Free launchers
    void lion_8bit_schedulefree_update_fp32(
        float* param, const float* grad, uint8_t* state_z, float* absmax_z,
        float beta1, float beta2, float eps, float lr, float weight_decay,
        float ckp1, float gnorm_scale, bool cautious, int numel, cudaStream_t stream
    );
    void lion_8bit_schedulefree_update_fp16(
        __half* param, const __half* grad, uint8_t* state_z, float* absmax_z,
        float beta1, float beta2, float eps, float lr, float weight_decay,
        float ckp1, float gnorm_scale, bool cautious, int numel, cudaStream_t stream
    );
    void lion_8bit_schedulefree_update_bf16(
        __nv_bfloat16* param, const __nv_bfloat16* grad, uint8_t* state_z, float* absmax_z,
        float beta1, float beta2, float eps, float lr, float weight_decay,
        float ckp1, float gnorm_scale, bool cautious, int numel, cudaStream_t stream
    );
}

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
    bool cautious,
    int N,
    int blocks,
    int threads
);

/*
Initialize Quantization Maps in Constant Memory

Args:
    qmap_signed_cpu: Signed quantization map (256 values)
*/
void init_quantization_maps_wrapper(torch::Tensor qmap_signed_cpu) {
    TORCH_CHECK(qmap_signed_cpu.dim() == 1 && qmap_signed_cpu.size(0) == 256,
                "qmap_signed must be 1D tensor with 256 elements");
    TORCH_CHECK(qmap_signed_cpu.dtype() == torch::kFloat32,
                "qmap_signed must be FP32");
    TORCH_CHECK(qmap_signed_cpu.is_contiguous(),
                "qmap_signed must be contiguous");
    TORCH_CHECK(qmap_signed_cpu.device().is_cpu(),
                "qmap_signed must be on CPU");

    // Copy to constant memory via extern "C" function
    init_quantization_maps(qmap_signed_cpu.data_ptr<float>());
}

/*
Lion 8-bit Update with Ring Buffer Support

Supports CPU-allocated states (Ring Buffer) with automatic GPU transfer.

Args:
    param: Model parameters (GPU, FP32/FP16/BF16)
    grad: Gradients (GPU, same dtype as param)
    exp_avg: Momentum state (CPU or GPU, UINT8)
    absmax: Absmax values (GPU, FP32)
    beta1: Interpolation coefficient
    beta2: Momentum EMA coefficient
    eps: Epsilon (unused in Lion, kept for compatibility)
    lr: Learning rate
    weight_decay: Weight decay coefficient
    gnorm_scale: Gradient norm scaling
    step: Current step
*/
void lion_8bit_update(
    torch::Tensor param,
    torch::Tensor grad,
    torch::Tensor exp_avg,
    torch::Tensor absmax,
    float beta1,
    float beta2,
    float eps,
    float lr,
    float weight_decay,
    float gnorm_scale,
    int step,
    bool cautious
) {
    // ============================================================
    // Validation
    // ============================================================

    TORCH_CHECK(param.is_cuda(), "Param must be on CUDA device");
    TORCH_CHECK(grad.is_cuda(), "Grad must be on CUDA device");
    TORCH_CHECK(absmax.is_cuda(), "Absmax must be on CUDA device");

    TORCH_CHECK(exp_avg.dtype() == torch::kUInt8, "Exp_avg must be UINT8");
    TORCH_CHECK(absmax.dtype() == torch::kFloat32, "Absmax must be FP32");

    TORCH_CHECK(param.numel() == grad.numel(), "Param and grad size mismatch");
    TORCH_CHECK(param.numel() == exp_avg.numel(), "Param and exp_avg size mismatch");

    int N = param.numel();
    int blocksize = 256;
    int num_blocks = (N + blocksize - 1) / blocksize;
    TORCH_CHECK(absmax.numel() == num_blocks, "Absmax size mismatch");

    // ============================================================
    // Ring Buffer Support: CPU→GPU Transfer
    // ============================================================

    torch::Tensor exp_avg_gpu = exp_avg;
    bool exp_avg_is_cpu = !exp_avg.is_cuda();

    // Transfer state to GPU if on CPU (Ring Buffer case)
    if (exp_avg_is_cpu) {
        exp_avg_gpu = exp_avg.to(param.device(), /*non_blocking=*/true);
    }

    // ============================================================
    // Launch CUDA Kernel
    // ============================================================

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    auto param_dtype = param.dtype();

    if (param_dtype == torch::kFloat32) {
        launch_lion_8bit_blockwise_update_kernel<float>(
            param.data_ptr<float>(),
            grad.data_ptr<float>(),
            exp_avg_gpu.data_ptr<unsigned char>(),
            absmax.data_ptr<float>(),
            beta1, beta2, eps, lr, weight_decay, gnorm_scale, step, cautious, N,
            blocks, threads
        );
    } else if (param_dtype == torch::kFloat16) {
        launch_lion_8bit_blockwise_update_kernel<__half>(
            reinterpret_cast<__half*>(param.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(grad.data_ptr<at::Half>()),
            exp_avg_gpu.data_ptr<unsigned char>(),
            absmax.data_ptr<float>(),
            beta1, beta2, eps, lr, weight_decay, gnorm_scale, step, cautious, N,
            blocks, threads
        );
    } else if (param_dtype == torch::kBFloat16) {
        launch_lion_8bit_blockwise_update_kernel<__nv_bfloat16>(
            reinterpret_cast<__nv_bfloat16*>(param.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(grad.data_ptr<at::BFloat16>()),
            exp_avg_gpu.data_ptr<unsigned char>(),
            absmax.data_ptr<float>(),
            beta1, beta2, eps, lr, weight_decay, gnorm_scale, step, cautious, N,
            blocks, threads
        );
    } else {
        TORCH_CHECK(false, "Unsupported parameter dtype");
    }

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "CUDA kernel launch failed: ", cudaGetErrorString(err));

    // Synchronize to ensure kernel completion before CPU→GPU copy back
    cudaDeviceSynchronize();

    // ============================================================
    // Ring Buffer Support: GPU→CPU Transfer
    // ============================================================

    // Copy updated state back to CPU if needed
    if (exp_avg_is_cpu) {
        exp_avg.copy_(exp_avg_gpu, /*non_blocking=*/false);
    }
}

/*
Lion 8-bit Schedule-Free Update with Ring Buffer Support

Supports CPU-allocated states (Ring Buffer) with automatic GPU transfer.

Args:
    param: Model parameters (GPU, FP32/FP16/BF16)
    grad: Gradients (GPU, same dtype as param)
    state_z: z-sequence momentum state (CPU or GPU, UINT8)
    absmax_z: Absmax values for z (GPU, FP32)
    beta1: Interpolation coefficient
    beta2: Momentum EMA coefficient
    eps: Epsilon (unused in Lion, kept for compatibility)
    lr: Scheduled learning rate (lr * rect for RAdam)
    weight_decay: Weight decay coefficient
    ckp1: Averaging coefficient (k+1)/(k+r)
    gnorm_scale: Gradient norm scaling
    cautious: Cautious masking (sign alignment check)
*/
void lion_8bit_schedulefree_update(
    torch::Tensor param,
    torch::Tensor grad,
    torch::Tensor state_z,
    torch::Tensor absmax_z,
    float beta1,
    float beta2,
    float eps,
    float lr,
    float weight_decay,
    float ckp1,
    float gnorm_scale,
    bool cautious
) {
    // ============================================================
    // Validation
    // ============================================================

    TORCH_CHECK(param.is_cuda(), "Param must be on CUDA device");
    TORCH_CHECK(grad.is_cuda(), "Grad must be on CUDA device");
    TORCH_CHECK(absmax_z.is_cuda(), "Absmax_z must be on CUDA device");

    TORCH_CHECK(state_z.dtype() == torch::kUInt8, "State_z must be UINT8");
    TORCH_CHECK(absmax_z.dtype() == torch::kFloat32, "Absmax_z must be FP32");

    TORCH_CHECK(param.numel() == grad.numel(), "Param and grad size mismatch");
    TORCH_CHECK(param.numel() == state_z.numel(), "Param and state_z size mismatch");

    int N = param.numel();
    int blocksize = 256;
    int num_blocks = (N + blocksize - 1) / blocksize;
    TORCH_CHECK(absmax_z.numel() == num_blocks, "Absmax_z size mismatch");

    // ============================================================
    // Ring Buffer Support: CPU→GPU Transfer
    // ============================================================

    torch::Tensor state_z_gpu = state_z;
    bool state_z_is_cpu = !state_z.is_cuda();

    // Transfer state to GPU if on CPU (Ring Buffer case)
    if (state_z_is_cpu) {
        state_z_gpu = state_z.to(param.device(), /*non_blocking=*/true);
    }

    // ============================================================
    // Launch CUDA Kernel
    // ============================================================

    auto param_dtype = param.dtype();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (param_dtype == torch::kFloat32) {
        lion_8bit_schedulefree_update_fp32(
            param.data_ptr<float>(),
            grad.data_ptr<float>(),
            state_z_gpu.data_ptr<uint8_t>(),
            absmax_z.data_ptr<float>(),
            beta1, beta2, eps, lr, weight_decay, ckp1, gnorm_scale, cautious, N, stream
        );
    } else if (param_dtype == torch::kFloat16) {
        lion_8bit_schedulefree_update_fp16(
            reinterpret_cast<__half*>(param.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(grad.data_ptr<at::Half>()),
            state_z_gpu.data_ptr<uint8_t>(),
            absmax_z.data_ptr<float>(),
            beta1, beta2, eps, lr, weight_decay, ckp1, gnorm_scale, cautious, N, stream
        );
    } else if (param_dtype == torch::kBFloat16) {
        lion_8bit_schedulefree_update_bf16(
            reinterpret_cast<__nv_bfloat16*>(param.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(grad.data_ptr<at::BFloat16>()),
            state_z_gpu.data_ptr<uint8_t>(),
            absmax_z.data_ptr<float>(),
            beta1, beta2, eps, lr, weight_decay, ckp1, gnorm_scale, cautious, N, stream
        );
    } else {
        TORCH_CHECK(false, "Unsupported parameter dtype");
    }

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "CUDA kernel launch failed: ", cudaGetErrorString(err));

    // Synchronize to ensure kernel completion before GPU→CPU copy back
    cudaDeviceSynchronize();

    // ============================================================
    // Ring Buffer Support: GPU→CPU Transfer
    // ============================================================

    // Copy updated state back to CPU if needed
    if (state_z_is_cpu) {
        state_z.copy_(state_z_gpu, /*non_blocking=*/false);
    }
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init_quantization_maps", &init_quantization_maps_wrapper,
          "Initialize quantization maps in constant memory");
    m.def("lion_8bit_update", &lion_8bit_update,
          "Lion 8-bit optimizer update with Ring Buffer support");
    m.def("lion_8bit_schedulefree_update", &lion_8bit_schedulefree_update,
          "Lion 8-bit Schedule-Free optimizer update with Ring Buffer support");
}
