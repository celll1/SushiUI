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

/*
 * Counter-based bit mixer (Murmur3-style finalizer). Stateless per element, so
 * no curand state has to be allocated or carried across steps.
 */
__device__ inline uint32_t sr_mix(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

/*
 * Stochastically quantize a normalized value to a uint8 index.
 *
 * quantize_value() above rounds to the NEAREST grid point, which discards every
 * change smaller than half a quantization step. The z sequence is read back and
 * re-quantized on every step, so under round-to-nearest a sub-half-step update
 * to z is discarded for the whole run and the code never moves -- measured on a
 * real Krea 2 tensor, 0 of 16384 z codes changed in 300 steps at lr 1e-5.
 *
 * Rounding up with probability equal to the position between the two bracketing
 * grid points instead makes E[dequantize(code)] == value, so those updates
 * survive in expectation. Same argument as BF16 stochastic rounding, one grid
 * coarser (see stochastic_rounding.py).
 */
__device__ inline uint8_t quantize_value_stochastic(float value, const float* qmap, float u) {
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

    const float lo = qmap[left];
    const float hi = qmap[right];
    const float span = hi - lo;
    float t = (span > 0.0f) ? ((value - lo) / span) : 0.0f;
    t = fminf(fmaxf(t, 0.0f), 1.0f);

    return (u < t) ? (uint8_t)right : (uint8_t)left;
}

/*
 * Uniform in [0, 1) for this element and step.
 */
__device__ inline float sr_uniform(int tid, unsigned int seed) {
    const uint32_t h = sr_mix(((uint32_t)tid) ^ sr_mix(seed));
    return (float)(h >> 8) * (1.0f / 16777216.0f);  // 24 bits
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
    const int numel,                        // Total number of elements
    const int stochastic_z,                 // 1: stochastically quantize z (0: round-to-nearest)
    const unsigned int seed                 // Per-step seed for the stochastic path
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

    // Block-level reduce (max). NOTE: cub::BlockReduce::Reduce returns the
    // aggregate in THREAD 0 ONLY -- every other thread gets a partial (its
    // warp's / raking segment's max). Anything read by all threads has to be
    // broadcast through shared memory; s_block_absmax_z below does that, and the
    // absmax writer under `local_tid == 0` is already safe.
    float block_absmax_z = BlockReduce(temp_storage_z).Reduce(abs_z, CubMaxOp{});
    __syncthreads();  // Reuse shared memory
    float block_absmax_sq = BlockReduce(temp_storage_sq).Reduce(abs_sq, CubMaxOp{});

    __shared__ float s_block_absmax_z;
    if (local_tid == 0) {
        s_block_absmax_z = block_absmax_z;
    }
    __syncthreads();

    // Update absmax (first thread in block writes)
    //
    // z's scale carries SYMMETRIC HEADROOM. The signed dynamic map is not
    // symmetric -- it ends at +1.000000000 but at -0.992968738 -- so a block
    // whose extreme element is NEGATIVE normalizes to exactly -1.0, is stored as
    // the nearest code (-0.992968738), and dequantizes 0.7031% smaller on the
    // next step. absmax is then recomputed from that smaller value, so the
    // block's scale shrinks by 0.7031% EVERY step and takes the whole block with
    // it: 0.992969^3000 = 4.6e-10. Measured over 3000 steps at lr 1e-5 with
    // zero-mean gradients, mean|z| fell to 0.48 of its reference (0.25 with
    // stochastic rounding, which moves codes and so tracks the sinking scale
    // faster). Round-to-nearest merely hid it -- it is a scale defect, not a
    // rounding one.
    //
    // Dividing by the largest magnitude representable in BOTH directions makes
    // the extreme element land exactly on a grid point in either sign, so
    // dequantize -> recompute absmax -> requantize is idempotent and the scale
    // only moves when the DATA moves. Cost: a 0.7% coarser grid.
    //
    // exp_avg_sq is deliberately not given headroom: it is non-negative and the
    // unsigned map ends at exactly +1.0, so its extreme is already exact.
    const float qmax_symmetric = fminf(-d_qmap_signed[0], d_qmap_signed[255]);
    if (local_tid == 0) {
        absmax_z[qblock_id] = block_absmax_z / qmax_symmetric;
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

    // Quantize using binary search. z carries the optimization sequence and is
    // re-quantized every step, so it is the one that freezes under
    // round-to-nearest; exp_avg_sq is a second moment whose accumulation is
    // deliberately left as-is here.
    // The element that DEFINES the block's absmax is stored exactly and never
    // stochastically: with the headroom above it normalizes to exactly
    // +/-qmax_symmetric, which is a grid point in either sign, so rounding it to
    // nearest is lossless.
    //
    // This is what keeps the scale out of the rounding-noise feedback loop.
    // absmax is a MAX over the block's own stored values, so if the extreme
    // element carries unbiased noise, the block's scale inherits the POSITIVE
    // bias of a maximum over noisy values -- and a larger scale means a larger
    // quantization step, hence a larger excess next step. Measured: +0.63% per
    // step, a factor of 1.5e8 over 3000 steps. Forcing that element down one
    // code instead (the obvious guard) inverts the sign of the same feedback and
    // sinks the block to 0.37x. With the extreme exact, no element can exceed
    // the block maximum (every other element's upper neighbour is at most that
    // maximum), so absmax tracks the DATA and nothing else.
    // Compared on the PRE-quantization magnitudes, which are bitwise equal for
    // the element the CUB reduction took its maximum from. Comparing the
    // normalized value against qmax_symmetric instead misses it: the two
    // divisions leave it an ulp below the constant.
    //
    // Against the BROADCAST maximum, not BlockReduce's return value: that is the
    // aggregate only in thread 0, so every other thread was comparing against a
    // partial max over its own warp/raking segment and exempting itself whenever
    // it was the biggest element in that subset. Measured on the built kernel,
    // 12.73% of elements were exempted instead of 0.39% (one per block) --
    // biased toward the large magnitudes, which are exactly the ones that need
    // stochastic rounding most. Not a stability bug (a partial max is <= the
    // true max, so the true extreme was always among those flagged, and the
    // fixed point held), but it silently withheld the fix from an eighth of the
    // tensor.
    const bool defines_absmax = (abs_z >= s_block_absmax_z);

    uint8_t new_qz = (stochastic_z && !defines_absmax)
        ? quantize_value_stochastic(normalized_z, d_qmap_signed, sr_uniform(tid, seed))
        : quantize_value(normalized_z, d_qmap_signed);

    // Belt and braces: the map has a code for +1.0 with no negative counterpart,
    // and storing z there would put it above its own block's absmax.
    if (d_qmap_signed[new_qz] > qmax_symmetric) {
        new_qz -= 1;
    }

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
// Note: Kernel instantiation is done in launcher
// ============================================================
// __global__ function templates cannot be explicitly instantiated
// in CUDA. The launcher (adamw8bit_schedulefree_launcher.cu) will
// call the kernel template for each dtype (float, __half, __nv_bfloat16).
