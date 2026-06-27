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
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAStreamGuard.h>
#include <c10/util/Optional.h>
#include <cuda_runtime.h>
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
    // Ring Buffer Support: CPU->GPU Transfer (on a dedicated stream)
    // ============================================================

    torch::Tensor state1_gpu = state1;
    torch::Tensor state2_gpu = state2;

    bool state1_is_cpu = !state1.is_cuda();
    bool state2_is_cpu = !state2.is_cuda();
    bool any_cpu = state1_is_cpu || state2_is_cpu;

    c10::DeviceIndex dev = param.device().index();
    at::cuda::CUDAStream current = at::cuda::getCurrentCUDAStream(dev);
    at::cuda::CUDAStream xfer = any_cpu ? get_xfer_stream(dev) : current;

    // H2D of the pinned Ring Buffer states on the transfer stream. The update
    // kernel (current stream) must wait for it -> order via an event.
    if (any_cpu) {
        at::cuda::CUDAStreamGuard guard(xfer);
        if (state1_is_cpu) state1_gpu = state1.to(param.device(), /*non_blocking=*/true);
        if (state2_is_cpu) state2_gpu = state2.to(param.device(), /*non_blocking=*/true);
        at::cuda::CUDAEvent e_h2d;
        e_h2d.record(xfer);
        e_h2d.block(current);
    }

    // The update kernel runs on the current (compute) stream, so the in-place
    // parameter update stays naturally ordered with the surrounding backward
    // and the next forward.
    cudaStream_t stream = current.stream();

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
    // Copy updated states back to CPU on the transfer stream (Ring Buffer)
    // ============================================================
    //
    // The D2H writeback runs on the dedicated transfer stream so it overlaps
    // with the next parameters' backward on the current stream, rather than
    // serialising the (large) state writeback into the compute pipeline. It is
    // ordered after the update kernel via an event, and record_stream keeps the
    // GPU staging buffers alive until both streams are done with them. Same-
    // stream ordering on the transfer stream keeps this step's D2H before next
    // step's H2D of the same pinned buffer; there is no per-parameter sync.
    if (any_cpu) {
        at::cuda::CUDAEvent e_kernel;
        e_kernel.record(current);
        e_kernel.block(xfer);
        at::cuda::CUDAStreamGuard guard(xfer);
        if (state1_is_cpu) {
            state1.copy_(state1_gpu, /*non_blocking=*/true);
            state1_gpu.record_stream(current);
            state1_gpu.record_stream(xfer);
        }
        if (state2_is_cpu) {
            state2.copy_(state2_gpu, /*non_blocking=*/true);
            state2_gpu.record_stream(current);
            state2_gpu.record_stream(xfer);
        }
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
    bool any_cpu = state_z_is_cpu || state_exp_avg_sq_is_cpu;

    c10::DeviceIndex dev = param.device().index();
    at::cuda::CUDAStream current = at::cuda::getCurrentCUDAStream(dev);
    at::cuda::CUDAStream xfer = any_cpu ? get_xfer_stream(dev) : current;

    // H2D of the pinned Ring Buffer states on the transfer stream; the kernel
    // (current stream) waits for it via an event.
    if (any_cpu) {
        at::cuda::CUDAStreamGuard guard(xfer);
        if (state_z_is_cpu) state_z_gpu = state_z.to(param.device(), /*non_blocking=*/true);
        if (state_exp_avg_sq_is_cpu) state_exp_avg_sq_gpu = state_exp_avg_sq.to(param.device(), /*non_blocking=*/true);
        at::cuda::CUDAEvent e_h2d;
        e_h2d.record(xfer);
        e_h2d.block(current);
    }

    // Update kernel runs on the current (compute) stream.
    cudaStream_t stream = current.stream();

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

    // D2H writeback on the transfer stream (overlaps with subsequent backward),
    // ordered after the kernel via an event; record_stream keeps the staging
    // buffers alive across both streams. (See note in adamw_8bit_update.)
    if (any_cpu) {
        at::cuda::CUDAEvent e_kernel;
        e_kernel.record(current);
        e_kernel.block(xfer);
        at::cuda::CUDAStreamGuard guard(xfer);
        if (state_z_is_cpu) {
            state_z.copy_(state_z_gpu, /*non_blocking=*/true);
            state_z_gpu.record_stream(current);
            state_z_gpu.record_stream(xfer);
        }
        if (state_exp_avg_sq_is_cpu) {
            state_exp_avg_sq.copy_(state_exp_avg_sq_gpu, /*non_blocking=*/true);
            state_exp_avg_sq_gpu.record_stream(current);
            state_exp_avg_sq_gpu.record_stream(xfer);
        }
    }
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
