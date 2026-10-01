# Ming-Image 0.1 source audit

Audit date: 2026-10-01. Scope: upstream source/configuration and SushiUI's
Qwen-Image 2.1 integration at baseline `2b28c3b4`. No weights were downloaded,
no model was loaded, and no running SushiUI API or GPU was accessed. This is
an implementation-planning audit, not numerical or memory acceptance.

## Pinned evidence

- [Source](https://github.com/inclusionAI/Ming-Image/tree/3515cd7b1e3049fc517eed7d35020251cd06519a), commit `3515cd7b1e3049fc517eed7d35020251cd06519a`.
- [Design artifact](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design/tree/208087ada1486931692c1896f38d4cd16ff3df82), revision `208087ada1486931692c1896f38d4cd16ff3df82`.
- [Layer artifact](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design-Layer/tree/9fabca8b62a67f1f53a957d46389c00451c11e52), revision `9fabca8b62a67f1f53a957d46389c00451c11e52`.

Model configurations, scheduler/tokenizer/processor configurations and file
inventories were inspected, alongside `infer.py`, `inference_profile.py`,
`modeling_bailingmm2.py`, `diffusion/transformer.py` and `diffusion/pipeline.py`.
These revisions define this audit; future upstream changes require a focused
comparison rather than silently changing the loader contract.

## Architecture facts

| Component | Released configuration / behavior |
|---|---|
| DiT | 30 main blocks, width 3840, 30 heads, two image/noise refiners and two context refiners; patch size 2, 16 latent channels, three-axis RoPE |
| MLLM | Custom BailingMM2/BailingMoeV2, 20 language blocks, width 2048, 256 experts / eight selected per token; custom Qwen2.5 vision tower |
| Connector | Qwen2 width 1536, 28 blocks; upstream disables causal attention and supplies a bidirectional mask |
| Conditioning | 256 learned queries through connector to width 2560; separate concatenated MLLM hidden states from layers 5/12/20 projected to width 3840 |
| VAE | AutoencoderKLQwenImage, four pixel channels, 16 latent channels, spatial compression 8; scale 8.0064, shift 0; reference encoding uses posterior mode |
| Scheduler | FlowMatchEulerDiscreteScheduler, 1000 training timesteps, configured shift 6.0, dynamic shifting false; pipeline also adjusts sequence-length shift configuration |
| Design profile | `zero_masked` alignment padding, `multi_frame_output=false`; text-to-image or native reference edit |
| Layer profile | `learned` alignment padding, `multi_frame_output=true`; reference image required; outputs simultaneous ordered RGBA layers |

The components are separate `mllm/`, `connector/`, `mlp/`, `transformer/`,
`vae/`, and `scheduler/` packages. Neither inventory declares a Diffusers
`model_index.json`. Detecting this as Qwen 2.1, or loading only the DiT with a
standard text encoder, would omit required learned conditioning.

The native sequence is image then caption. Image tokens contain target
frame(s), references, and alignment padding; caption conditioning has its own
padding. Alignment is to 32 tokens. Learned padding in Layer is live model
state and participates in attention. Main-block joint attention updates image
and text tokens; it is not Qwen's invariant block-causal condition prefix.

`infer.py` requests `N + 1` frames for N layers, verifies that count, and
omits frame zero from its layer-file export. SushiUI must retain the output
roles explicitly rather than mistake that slot for another requested layer
or treat F as video time. The upstream CLI's layer-count text parsing can
supersede its argument; a typed UI/API needs explicit conflict validation.

Generation uses FP32 noise and timestep handling with BF16 model execution.
The sampler supplies `(1000 - scheduler_t) / 1000` to the model. With scale
greater than one, upstream computes `positive + scale * (positive - zero)`
and negates the model output before the scheduler step. The negative branch
is zero tensors in both conditioning streams, not an encoded negative prompt.
The training velocity sign/time contract must be established by a small
forward/scheduler oracle before applying shared training objectives.

Upstream Design sizing selects 1024/2048 buckets and can resize to requested
positive dimensions, with the documented aspect-ratio boundary 1:4 to 4:1.
Layer uses its own 512/1024 behavior. Record bucket and final dimensions
separately; do not infer compute cost from the resized gallery image.

## Compatibility and provenance

Root source and model license files declare MIT. This does not remove
file-level notices: `diffusion/transformer.py`,
`diffusion/autoencoder_kl_qwenimage.py` and `qwen2_5_vit.py` explicitly carry
Apache-2.0 headers. Other files include Ant Group copyright notices. Before
vendoring, record every copied/adapted file in `docs/legal/`, preserve its
notices, and include the applicable license copies. No upstream implementation
has been added to SushiUI by this audit.

The upstream requirements pin Torch 2.4.0, Transformers 4.57.1 and Diffusers
0.36.0, with Transformer Engine listed. Inspect optional native fallbacks and
adapt version-specific interfaces in an isolated implementation environment;
do not replace SushiUI's working dependencies wholesale. The MLLM exposes
its own eager/FlashAttention path, while DiT uses SDPA. A single shared
attention selector cannot assume their capabilities match. Device-specific
CUDA autocast sites must be adapted without global monkeypatches.

Official model cards document inference on an 80 GiB GPU. This is a tested
configuration, not a proven minimum. No VRAM, latency, parameter census or
training-quality measurement was made here. Active MoE experts affect compute,
but all resident expert weights still affect storage and memory.

## Findings that constrain implementation

1. A native Ming loader, conditioner, VAE adapter and training handler are
   required; Qwen's loader, 64-channel latent assumptions and velocity target
   cannot be reused wholesale.
2. Exact Qwen prefix K/V reuse is inapplicable to Ming's updating joint
   sequence. Conditioning-output caching remains possible. Approximate
   denoising caches require separate opt-in semantics and measurements.
3. Qwen style-KV's exact segmented block-causal semantics do not transfer.
   An architecture-specific style path needs a new operator and acceptance
   tests; simply injecting Qwen's hook is not parity.
4. U-Net transport must distinguish targets, references, caption and live
   padding; spatial reduction acts per target layer, never across F.
5. Full text-encoder training parity includes the custom MLLM, connector,
   projections and learned queries as explicit trainable component groups.
   Freezing them permanently would drop a currently available Qwen feature.
6. Layer data, output grouping, alpha preservation and resume/cache identities
   need additional contracts beyond the ordinary one-image request.

See [integration contract](../guides/MING_IMAGE_01_DESIGN.md) for the required
parity surface and acceptance gates. Implementation scheduling stays under
ignored `local/strategy/` per the documentation policy.
