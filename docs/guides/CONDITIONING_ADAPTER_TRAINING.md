# Conditioning adapter training controls

`train_adapter` and `adapter_lr` are shared training keys for a model-owned
conditioning bridge between an encoder and its denoiser. They are **not** the
LoRA algorithm selector (`adapter_algorithm`), a license to train the text
encoder, or a VAE swap control. Both default to `null`: an existing run keeps
its architecture's historical train/freeze policy and LR. An explicit
`adapter_lr: 0` is a valid zero learning rate, not an unset value.
Qwen and Anima also accept an explicit `train_adapter: true` with
`train_unet: false` for bridge-only training; a null adapter choice does not
silently convert an old denoiser-off run into bridge-only training.

| Architecture | Bridge | Current common-key implementation |
|---|---|---|
| Qwen-Image 2.1 | DiT `txt_in`: RMSNorm, two 4096-wide Linears, GELU | LoRA: opt-in two Linear branches; full DiT: included by default, opt-out; independent optimizer group when trained |
| Anima | DiT `llm_adapter`: Qwen3-to-DiT adapter transformer | LoRA/full: common train override; independent LR override. Null retains `train_llm_adapter`, `anima_lora_scope`, and the full-FT LR factor. |

Qwen LoRA files with `txt_in` branches use the existing
`lora_unet_txt_in__{in_layer,out_layer}` key codec; the generation loader
enumerates both slots even for an old attention-only file. The frozen
Qwen3-VL encoder receives no LoRA. Qwen's partition-global sidecar is a
separate training mechanism and stays in the denoiser LR group. Expanding the
Qwen target inventory from 128 to 130 adds `4 * 4096 * rank` trainable
parameters (2,097,152 at rank 128), not the full 33,558,528 `txt_in` weights.
Changing the number of optimizer groups is a new-run/resume contract: do not toggle
`train_adapter` or introduce `adapter_lr` mid-run and expect an old optimizer
state to map index-for-index. Qwen full fine-tuning and Anima LoRA retain their
legacy optimizer group layout when the common LR remains null.

## Inventory of other conditioning boundaries

The presence of a projection alone does not make it safe to map to the common
switch. Each future integration needs a target inventory, checkpoint codec,
optimizer grouping, and a generation round trip.

| Architecture | Candidate and current boundary |
|---|---|
| Krea 2 | `txt_in` and `text_fusion`; the existing LoRA `proj`/`text_fusion` scopes have different breadth, so a bridge-only target set must be specified first. |
| Lens | `txt_in` sits in the full-FT `other` group; a shared adapter group would require splitting the existing group without losing its LR factors. |
| Flux2 | `context_embedder` is a text-side DiT projection; existing LoRA scopes/grouping do not identify it as a distinct component. |
| MiniMax-H3 | `context_embedder` and token refiner form the text-conditioning entry; a bridge-only group would need to preserve the joint audio/video training contract. |
| ACE-Step 1.5 | DiT `text_projector` is a candidate, subject to its text/lyrics routing and adapter target codec. |
| MiniT2I | `txt_embedder` and `pooled_embedder` already have a `txt_embed` LoRA scope; its two outputs need one explicit bridge-component grouping policy. |
| LTX-2.3 | Separate `LTX2TextConnectors` bridge Gemma-3 to the denoiser, but both encoder and connectors are currently frozen and not checkpointed by training. |
| Ideogram 4 | The conditional DiT consumes packed multi-layer Qwen3-VL features; a separable text bridge is not established by the current LoRA target inventory. |
| SDXL with custom text encoder | `te_adapters` is already a separately trained dimension bridge with TE LR fallback; it can adopt the common LR after legacy precedence is specified. Ordinary SDXL CLIP has no such external bridge. |
| SenseNova SDXL Chimera | `condition_bridge` has an existing stage-dependent `chimera_bridge_lr`; stage alignment semantics prohibit an independent on/off switch without revisiting the training plan. |
| SenseNova U1.5 VAE swap | **Not a conditioning adapter.** It rebuilds latent patch input/output layers and requires full fine-tuning of `fm_modules`; `train_adapter` must not stand in for that safety gate. |

SD1.5/ordinary SDXL use their existing cross-attention input projections,
not a separately owned text bridge. Z-Image has no separately registered
bridge component in its current training adapter. These remain on the ordinary
denoiser/TE controls rather than being guessed into `train_adapter`.

The API rejects a non-null common-key request outside the implemented Qwen
and Anima pair. This avoids a UI/config setting that appears to work while an
architecture silently ignores it.
