# Anima run 176 attention backward precision

Date: 2026-10-12. Checkpoint: step 123701. No optimizer updates were performed.
The owner run was normally stopped and restarted with unchanged parameters.

## Conclusion

The sampled attention gradients do not reproduce the large-logit backward
failure described by [KohakuFA](https://github.com/KohakuBlueleaf/KohakuFA/blob/main/docs/precision.md).
This does not prove all heads, examples, training steps, or optimizer updates
are numerically correct, and does not establish the cause of the jagged samples.
The explicit global-summary attention has a smaller, measurable BF16 precision
issue that largely disappears with an FP32 replay.

## Method

- RTX 6000 Ada, SM89; torch 2.10.0+cu130; flash-attn 2.8.3.
- One API-revalidated rating:general image, fitted to 3072x1536. Its content,
  captions, and derived images are not included in this audit.
- The size gives 4608 image tokens and crosses the configured 4096-token
  threshold, so the existing training plan uses two regions. The sample-size
  1472x896 input does not cross that threshold; it cannot test partition backward.
- Real checkpoint, Qwen 2.1 codec, global adapter, caption bridge, checkpointed
  blocks, region loss and backward. Crop decode loss remains disabled.
- Capture kernel-boundary Q/K/V and the gradient arriving at attention output
  from the actual region velocity loss. No synthetic output gradient is used.
- DiT self/cross attention in blocks 0, 14, 27 and self/cross attention in all
  six caption-bridge blocks; sigma 0.1, 0.2, 0.5, 0.9: 72 cases. For each, retain
  every query/key token in the first region and heads 0/middle/last. Head selection
  preserves attention semantics because these heads are independent.
- FP64 reference: explicitly compute scaled QK, masked softmax and PV, then
  autograd with the same captured output gradient. Cast the **same BF16 inputs**
  to FP64: this separates arithmetic/kernel error from input quantization.
- Replay direct FA2 where unmasked, automatic SDPA and forced math,
  memory-efficient and cuDNN SDPA. Compare dQ/dK/dV relative L2, cosine, max
  absolute error, reference magnitude and finiteness. Also record the BF16
  rounding floor obtained by rounding the reference gradients themselves.
- Capture both explicit global-adapter attention products separately, using
  their real upstream gradients, and compare BF16 and FP32 arithmetic against
  the same FP64 reference. Eight cases per capture; all rank-64 coordinates
  and all tokens retained. Its softmax is FP32, but surrounding einsum products
  and stored probabilities run in BF16 under autocast.

## Results

The largest absolute scaled logit in the selected DiT/bridge heads was 23.786,
well below the reported pathological range around 1e5 and above. The global
adapter's logits were below 1. All gradients were finite.

| Replay | Maximum dQ relative L2 | Maximum dK | Maximum dV | Lowest gradient cosine |
|---|---:|---:|---:|---:|
| Direct FA2, 24 unmasked cases | 1.448% | 0.400% | 1.424% | 0.999895 |
| Automatic SDPA, 72 cases | 1.712% | 0.666% | 1.424% | 0.999855 |
| Forced cuDNN, 72 cases | 2.009% | 0.700% | 1.424% | 0.999799 |
| Forced math, BF16 input/output gradients | 0.252% | 0.231% | 0.184% | 0.999998 |

The profiler identifies automatic SDPA as **memory-efficient**, including the
masked caption bridge. The installed PyTorch build cannot run its forced flash
SDPA backend; this is distinct from the working external flash-attn FA2 package.
cuDNN above is a forced comparison, not the observed automatic kernel.

Global local-query attention shows more cancellation sensitivity: the final
capture has maximum dQ/dK/dV errors of 5.241%/9.831%/0.526%, with minimum cosine
0.995160. At the worst dK case, sigma 0.5, reference norm is 5.889e-6 and maximum
absolute error 8.730e-8. A preceding independent capture measured 11.250% dK
error and cosine 0.993702; upstream gradients are not bitwise identical between
captures. These relative errors must not be treated as zero merely because
the absolute gradients are small.

On identical inputs/upstream gradients, FP32 replay reduces the global maximum
dQ/dK/dV error to **0.052716%/0.000516%/0.000036%**. This implicates the explicit
BF16 attention arithmetic rather than an FA/cuDNN large-logit failure. Keeping
global attention products/probabilities in FP32 is a plausible bounded follow-up;
this audit does not enable it or claim it repairs image quality.

## Limits and evidence

No broad BF16-to-FP32 training change is justified by this measurement. The main
sampled attention gradients have near-identical direction, unlike the quoted
near-zero-cosine failure. Math SDPA is more precise here, but no throughput or
long-term quality comparison was performed. This also does not test BF16 weight
update rounding or optimizer state precision. Earlier FP32 denoising-state tests
address a separate inference integration error.

Machine-readable case records and aggregate measurements:
[anima_owner176_attention_precision_2026-10.json](data/anima_owner176_attention_precision_2026-10.json).
Private reproducibility scripts and complete replay logs are under
`local/verification/owner176_attention_audit*` and
`local/verification/owner176_global_attention_audit.py`. No image or Q/K/V
tensor payload is committed. The default backend and run configuration remain
unchanged; resume uses `latest`, step 123701.
