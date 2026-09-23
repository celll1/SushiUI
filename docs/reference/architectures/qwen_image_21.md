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
latent Reference Guide, segmented-attention Style Transfer, progress/
cancellation, matrix live previews, optional TAE live previews, and condition-
prefix KV caching. The cache is controlled by `qwen_image_21_kv_cache` and
defaults on; Style Transfer disables it for that request. Attention selection,
runtime quantization, FBCache, Spectrum, NAG, ControlNet, SDEdit strength,
generation block swap, keep-hot, and VAE/TE replacement are not claimed;
capability data marks the shared controls unsupported.

Training supports DiT LoRA on Original or a frozen INT8 ConvRot base, and full
DiT fine-tuning on Original only. `train_adapter: true` adds LoRA to both
`txt_in` Linear layers; `adapter_lr` gives those branches an independent
optimizer group. The default LoRA inventory remains 128 attention projections.
Full fine-tuning includes the complete `txt_in` by default and can freeze it
with `train_adapter: false`. Qwen3-VL and the VAE remain frozen. The flow
objective is `xt = (1-sigma)*x0 + sigma*noise`, target `noise-x0`; loss is taken
only from the target latent tail. ReLoRA, ControlNet, text-encoder training,
VAE swap, and reference-conditioned edit datasets are refused.

For new runs, an omitted guidance-loss weight resolves to 1.0 (100% guided
target); an explicit 0.0 selects ordinary MSE. Existing run YAML retains its
stored value. Custom mixtures default to a stochastic per-image choice between
the two targets; the legacy weighted blend remains available. The optional
CFG-null sigma schedule applies dropout only after ordinary MSE is selected,
while the independent fixed-drop mode remains available for comparisons.

Real 256x256 gates passed for a three-step rank-4 ConvRot LoRA run (including
checkpoint/optimizer resume) and a two-step Original full-DiT run resumed
between steps. The full checkpoint is a four-shard 14.23 GB tensor set; its
step-1 index was reloaded by the production loader as 32 blocks and
7,115,124,736 parameters. Exact measurements and the disk-space caveat are in
`MODEL_FACTS.md`.

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
