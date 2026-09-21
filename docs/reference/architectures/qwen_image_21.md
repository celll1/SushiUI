# Qwen-Image 2.1 (`qwen_image_21`)

Qwen-Image 2.1 is a 32-block, single-stream flow-matching DiT with block-causal
attention. It consumes unpatched 64-channel image latents and frozen Qwen3-VL
conditioning. The native VAE is RGBA, spatially 16x compressed, and uses
64-element mean/std normalization vectors.

## Code map

| Responsibility | Symbol / file |
|---|---|
| Detection and load dispatch | `core/model_loader.py::ModelLoader` |
| Split artifact contract | `core/models/qwen_image_21/artifact.py` |
| Source/artifact component loader | `core/models/qwen_image_21/loader.py` |
| Transformer, VAE, pipeline | `core/models/qwen_image_21/vendor/` |
| Generation integration | `core/pipeline_backends/qwen_image_21.py::QwenImage21Mixin` |
| LoRA key/target codec | `core/models/qwen_image_21/lora.py` |
| Training handler | `core/training/arch/qwen_image_21.py` |
| Training math | `core/training/ops/qwen_image_21_ops.py` |
| LoRA/full adapters | `core/training/adapters/qwen_image_21_adapter.py` |
| Artifact converter | `subapps/qwen_image_21_convert.py` |

## Runtime contract

The selected path is either a source Diffusers directory or a prepared variant
directory containing `manifest.json`. Standard local variants are `original`
and `int8_convrot`; each keeps DiT, text encoder, and VAE in separate single
safetensors files so component offload does not require mapping the other large
component.

Generation supports txt2img, true CFG, native image editing with up to ten
ordered images, inpaint/outpaint protected-pixel compositing, ordinary LoRA,
progress/cancellation, and condition-prefix KV caching. The cache is controlled
by `qwen_image_21_kv_cache` and defaults on. Attention selection, runtime
quantization, FBCache, Spectrum, NAG, ControlNet, SDEdit strength, generation
block swap, keep-hot, and VAE/TE replacement are not claimed; capability data
marks the shared controls unsupported.

Training supports DiT LoRA on Original or a frozen INT8 ConvRot base, and full
DiT fine-tuning on Original only. Qwen3-VL and the VAE remain frozen. The flow
objective is `xt = (1-sigma)*x0 + sigma*noise`, target `noise-x0`; loss is taken
only from the target latent tail. ReLoRA, ControlNet, text-encoder training,
VAE swap, and reference-conditioned edit datasets are refused.

See `docs/guides/QWEN_IMAGE_21_DESIGN.md` for exact artifact layout, measured
smokes, deferred gates, and provenance.

## Prompt upsampling

Generation panels optionally rewrite the prompt before queueing. Text-only
requests use the T2I enhancer; edit, inpaint, outpaint, and reference-image
requests use the I2I enhancer with the actual conditioning images. The bundled
enhancers are on-demand INT8 ConvRot single-file artifacts under
`prompt_enhancer/t2i` and `prompt_enhancer/i2i`. Only one is CPU-resident at a
time and it moves to CUDA only for the rewrite.

LM Studio and Ollama are alternative loopback-only engines. They reuse the
shared local-model discovery and load/unload lifecycle. Returned aspect-ratio
advice is displayed but never changes the selected generation canvas.
