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
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Optional.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <array>

// ============================================================
// Dedicated per-device transfer stream for Ring Buffer state I/O
// ============================================================
// Cached once per device from the stream pool so it is STABLE across calls.
// Both the H2D and the D2H of a parameter's optimizer state use this same
// stream, which keeps cross-step coherence of the pinned CPU buffer (this
// step's D2H is ordered before next step's H2D on the same stream). Running
// the state writeback here lets it overlap with subsequent backward work on
// the compute (current) stream instead of serialising with it.
static at::cuda::CUDAStream get_xfer_stream(c10::DeviceIndex device) {
    static std::array<c10::optional<at::cuda::CUDAStream>, 64> cache;
    if (device >= 0 && static_cast<size_t>(device) < cache.size()) {
        if (!cache[device].has_value()) {
            cache[device].emplace(at::cuda::getStreamFromPool(/*isHighPriority=*/false, device));
        }
        return cache[device].value();
    }
    return at::cuda::getStreamFromPool(/*isHighPriority=*/false, device);
}

// Forward declarations of CUDA functions (defined in .cu files)
extern "C" {
    void init_quantization_maps(const float* host_qmap_signed);

    // Same map, for the Schedule-Free translation unit's own __constant__ copy.
    void init_schedulefree_quantization_maps(const float* host_qmap_signed);

    // Schedule-Free launchers
    void lion_8bit_schedulefree_update_fp32(
        float* param, const float* grad, uint8_t* state_z, float* absmax_z,
        float beta1, float beta2, float eps, float lr, float weight_decay,
        float ckp1, float gnorm_scale, bool cautious, int numel,
        int stochastic_z, unsigned int seed, cudaStream_t stream
    );
    void lion_8bit_schedulefree_update_fp16(
        __half* param, const __half* grad, uint8_t* state_z, float* absmax_z,
        float beta1, float beta2, float eps, float lr, float weight_decay,
        float ckp1, float gnorm_scale, bool cautious, int numel,
        int stochastic_z, unsigned int seed, cudaStream_t stream
    );
    void lion_8bit_schedulefree_update_bf16(
        __nv_bfloat16* param, const __nv_bfloat16* grad, uint8_t* state_z, float* absmax_z,
        float beta1, float beta2, float eps, float lr, float weight_decay,
        float ckp1, float gnorm_scale, bool cautious, int numel,
        int stochastic_z, unsigned int seed, cudaStream_t stream
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
    int threads,
    cudaStream_t stream
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

    // Copy to constant memory via extern "C" functions. BOTH translation units:
    // __constant__ symbols are not shared between them, and the Schedule-Free
    // kernel reads its own copy (see init_schedulefree_quantization_maps).
    init_quantization_maps(qmap_signed_cpu.data_ptr<float>());
    init_schedulefree_quantization_maps(qmap_signed_cpu.data_ptr<float>());
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

    c10::DeviceIndex dev = param.device().index();
    at::cuda::CUDAStream current = at::cuda::getCurrentCUDAStream(dev);
    at::cuda::CUDAStream xfer = exp_avg_is_cpu ? get_xfer_stream(dev) : current;

    // H2D of the pinned Ring Buffer state on the transfer stream. The update
    // kernel (current stream) must wait for it -> order via an event.
    if (exp_avg_is_cpu) {
        c10::cuda::CUDAStreamGuard guard(xfer);
        exp_avg_gpu = exp_avg.to(param.device(), /*non_blocking=*/true);
        at::cuda::CUDAEvent e_h2d;
        e_h2d.record(xfer);
        e_h2d.block(current);
    }

    // ============================================================
    // Launch CUDA Kernel
    // ============================================================

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    // The update kernel runs on the current (compute) stream, so the in-place
    // parameter update stays naturally ordered with the surrounding backward.
    cudaStream_t stream = current.stream();

    auto param_dtype = param.dtype();

    if (param_dtype == torch::kFloat32) {
        launch_lion_8bit_blockwise_update_kernel<float>(
            param.data_ptr<float>(),
            grad.data_ptr<float>(),
            exp_avg_gpu.data_ptr<unsigned char>(),
            absmax.data_ptr<float>(),
            beta1, beta2, eps, lr, weight_decay, gnorm_scale, step, cautious, N,
            blocks, threads, stream
        );
    } else if (param_dtype == torch::kFloat16) {
        launch_lion_8bit_blockwise_update_kernel<__half>(
            reinterpret_cast<__half*>(param.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(grad.data_ptr<at::Half>()),
            exp_avg_gpu.data_ptr<unsigned char>(),
            absmax.data_ptr<float>(),
            beta1, beta2, eps, lr, weight_decay, gnorm_scale, step, cautious, N,
            blocks, threads, stream
        );
    } else if (param_dtype == torch::kBFloat16) {
        launch_lion_8bit_blockwise_update_kernel<__nv_bfloat16>(
            reinterpret_cast<__nv_bfloat16*>(param.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(grad.data_ptr<at::BFloat16>()),
            exp_avg_gpu.data_ptr<unsigned char>(),
            absmax.data_ptr<float>(),
            beta1, beta2, eps, lr, weight_decay, gnorm_scale, step, cautious, N,
            blocks, threads, stream
        );
    } else {
        TORCH_CHECK(false, "Unsupported parameter dtype");
    }

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "CUDA kernel launch failed: ", cudaGetErrorString(err));

    // ============================================================
    // Ring Buffer Support: GPU→CPU Transfer (on the transfer stream)
    // ============================================================

    // D2H writeback on the dedicated transfer stream so it overlaps with the
    // next parameters' backward on the current stream. Ordered after the update
    // kernel via an event; record_stream keeps the GPU staging buffer alive
    // until both streams are done with it. Same-stream ordering on the transfer
    // stream keeps this step's D2H before next step's H2D of the same pinned
    // buffer; there is no per-parameter sync.
    if (exp_avg_is_cpu) {
        at::cuda::CUDAEvent e_kernel;
        e_kernel.record(current);
        e_kernel.block(xfer);
        c10::cuda::CUDAStreamGuard guard(xfer);
        exp_avg.copy_(exp_avg_gpu, /*non_blocking=*/true);
        exp_avg_gpu.record_stream(current);
        exp_avg_gpu.record_stream(xfer);
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
    bool cautious,
    bool stochastic_z,
    int64_t seed
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

    c10::DeviceIndex dev = param.device().index();
    at::cuda::CUDAStream current = at::cuda::getCurrentCUDAStream(dev);
    at::cuda::CUDAStream xfer = state_z_is_cpu ? get_xfer_stream(dev) : current;

    // H2D of the pinned Ring Buffer state on the transfer stream; the kernel
    // (current stream) waits for it via an event.
    if (state_z_is_cpu) {
        c10::cuda::CUDAStreamGuard guard(xfer);
        state_z_gpu = state_z.to(param.device(), /*non_blocking=*/true);
        at::cuda::CUDAEvent e_h2d;
        e_h2d.record(xfer);
        e_h2d.block(current);
    }

    // ============================================================
    // Launch CUDA Kernel
    // ============================================================

    auto param_dtype = param.dtype();
    // Update kernel runs on the current (compute) stream.
    cudaStream_t stream = current.stream();

    if (param_dtype == torch::kFloat32) {
        lion_8bit_schedulefree_update_fp32(
            param.data_ptr<float>(),
            grad.data_ptr<float>(),
            state_z_gpu.data_ptr<uint8_t>(),
            absmax_z.data_ptr<float>(),
            beta1, beta2, eps, lr, weight_decay, ckp1, gnorm_scale, cautious, N,
            stochastic_z ? 1 : 0, static_cast<unsigned int>(seed), stream
        );
    } else if (param_dtype == torch::kFloat16) {
        lion_8bit_schedulefree_update_fp16(
            reinterpret_cast<__half*>(param.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(grad.data_ptr<at::Half>()),
            state_z_gpu.data_ptr<uint8_t>(),
            absmax_z.data_ptr<float>(),
            beta1, beta2, eps, lr, weight_decay, ckp1, gnorm_scale, cautious, N,
            stochastic_z ? 1 : 0, static_cast<unsigned int>(seed), stream
        );
    } else if (param_dtype == torch::kBFloat16) {
        lion_8bit_schedulefree_update_bf16(
            reinterpret_cast<__nv_bfloat16*>(param.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(grad.data_ptr<at::BFloat16>()),
            state_z_gpu.data_ptr<uint8_t>(),
            absmax_z.data_ptr<float>(),
            beta1, beta2, eps, lr, weight_decay, ckp1, gnorm_scale, cautious, N,
            stochastic_z ? 1 : 0, static_cast<unsigned int>(seed), stream
        );
    } else {
        TORCH_CHECK(false, "Unsupported parameter dtype");
    }

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "CUDA kernel launch failed: ", cudaGetErrorString(err));

    // ============================================================
    // Ring Buffer Support: GPU→CPU Transfer (on the transfer stream)
    // ============================================================

    // D2H writeback on the dedicated transfer stream (overlaps with subsequent
    // backward), ordered after the kernel via an event; record_stream keeps the
    // staging buffer alive across both streams. (See note in lion_8bit_update.)
    if (state_z_is_cpu) {
        at::cuda::CUDAEvent e_kernel;
        e_kernel.record(current);
        e_kernel.block(xfer);
        c10::cuda::CUDAStreamGuard guard(xfer);
        state_z.copy_(state_z_gpu, /*non_blocking=*/true);
        state_z_gpu.record_stream(current);
        state_z_gpu.record_stream(xfer);
    }
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init_quantization_maps", &init_quantization_maps_wrapper,
          "Initialize quantization maps in constant memory");
    m.def("lion_8bit_update", &lion_8bit_update,
          "Lion 8-bit optimizer update with Ring Buffer support");
    m.def("lion_8bit_schedulefree_update", &lion_8bit_schedulefree_update,
          "Lion 8-bit Schedule-Free optimizer update with Ring Buffer support",
          py::arg("param"), py::arg("grad"), py::arg("state_z"), py::arg("absmax_z"),
          py::arg("beta1"), py::arg("beta2"), py::arg("eps"), py::arg("lr"),
          py::arg("weight_decay"), py::arg("ckp1"), py::arg("gnorm_scale"),
          py::arg("cautious"), py::arg("stochastic_z") = false, py::arg("seed") = 0);
}
