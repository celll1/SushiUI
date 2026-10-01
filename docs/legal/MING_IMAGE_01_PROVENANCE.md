# Ming-Image source provenance

Source: [inclusionAI/Ming-Image](https://github.com/inclusionAI/Ming-Image/tree/3515cd7b1e3049fc517eed7d35020251cd06519a), commit `3515cd7b1e3049fc517eed7d35020251cd06519a`.

Vendored files live under `backend/core/models/ming_image_01/vendor/`. Original file headers are retained. Root MIT covers files without a separate license declaration; explicitly Apache-2.0 files retain that license. The applicable texts are [MIT](licenses/MIT-Ming-Image.txt) and [Apache-2.0](licenses/Apache-2.0.txt). No model weights or remote-code downloads are bundled. Both model snapshots declare MIT; verify the acquired snapshot separately.

| File | Declared applicable license | Upstream SHA-256 | Local changes |
|---|---|---|---|
| `bailingmm_utils.py` | MIT (root license) | `daf0aca163ee9332e538edea50ab4e5385c8928d030687ebefadae52fba5f401` | AST-equivalent trailing-whitespace cleanup |
| `configuration_bailing_moe_v2.py` | MIT (root license) | `5c8d0c050ab8dba0465918fa365527406ea6717aa33fdb6642b4395653201f7d` | AST-equivalent trailing-whitespace cleanup |
| `configuration_bailingmm2.py` | Apache-2.0 | `71eb6de8580c0999487b2e92617b40093321fe4927aa40cdd049b5a9aff16775` | package-relative imports |
| `diffusion/__init__.py` | MIT (root license) | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | AST-equivalent trailing-whitespace cleanup |
| `diffusion/autoencoder_kl_qwenimage.py` | Apache-2.0 | `5550326b4a24bd091a536b6130e34cc5c278697b6f5b0007fbb04f89b2fd93c8` | Unmodified |
| `diffusion/generator.py` | MIT (root license) | `8c6bab1f288931edfac8a09c823cc1fc54adbc8f39c517400ff41df9511c96f3` | package-relative imports; AST-equivalent trailing-whitespace cleanup |
| `diffusion/padding.py` | MIT (root license) | `009b5b7c22858827f0de132dd30cdbd5b51788cebb1401ee150ee4d6ef664d29` | Unmodified |
| `diffusion/pipeline.py` | Apache-2.0 | `4ec14e6b7f9d541bd5f10c6c53495bc4c383bd8416f8236150d02c6282be364f` | AST-equivalent trailing-whitespace cleanup |
| `diffusion/transformer.py` | Apache-2.0 | `bec8b892e1b08f0fb6d5c305cec3047527858a032b4ddfaef0f9198365217de4` | package-relative imports; device-aware FP32 rotary/timestep computation; optional target-only residual U-Net dispatch; explicit full-canvas target coordinates and independent full-reference patch packing for spatial partitions; bounded partial main-stream gradient checkpointing; AST-equivalent trailing-whitespace cleanup |
| `image_processing_bailingmm2.py` | Apache-2.0 | `ae6275c6aaae742b32230ff9a84475b535cf882af95a0ef7bca9ebfe93b20d0c` | package-relative imports |
| `inference_profile.py` | MIT (root license) | `0ac0424173f65c143b4ff7b7004e06db9af287f456031e0a9c58995533b6329d` | Unmodified |
| `mllm_device_map.py` | MIT (root license) | `83caa5c9894f6179b4819251457bdd41c1b9ce469ffecc60bed96f0f3a09a6ba` | Unmodified |
| `modeling_bailing_moe_v2.py` | Apache-2.0 | `f0d4f0fd2cf4f068848f78405ebcaff10f8be8fba1077fb9ebac3fd6d84694a6` | package-relative imports; use native Torch FX after Transformers removed its availability helper; AST-equivalent trailing-whitespace cleanup; preserve native default/partial RoPE across Transformers APIs; map tied-weight source/target keys; packed Linear compute-dtype fallback for FlashAttention recasting |
| `modeling_bailingmm2.py` | MIT (root license) | `fd28059a066a12a4e12965204f8a26821744dfbdaea77c97570a74ed347ed59a` | package-relative imports; device-aware autocast (CUDA BF16 retained, CPU autocast disabled); AST-equivalent trailing-whitespace cleanup; explicit live-conditioning flag preserves full-TE direct-stream gradients |
| `modeling_utils.py` | MIT (root license) | `265b74f3e2246a6d01988e4408939a82157360abe73aa474aa3cb6cfb08db9bc` | Unmodified |
| `processing_bailingmm2.py` | Apache-2.0 | `001b316bc82b5c3e05739223d0e0349c2e79258e6638bf9f664df8ee261d93c1` | package-relative imports; AST-equivalent trailing-whitespace cleanup |
| `qwen2_5_vit.py` | Apache-2.0 | `587148f9d9e26ed94e2e0aee277f181bc20de8b1a2b2992cf75362c816bf7f24` | declare gradient checkpointing support for the existing native block checkpoint path; packed Linear compute-dtype fallback for vision activation inputs |
| `tokenization_bailing.py` | MIT (root license) | `4b13ec7a5796a1d04e721f61791d086d7f7fb9f63912948700e5691625b432fb` | initialize BOS property after base tokenizer construction |

Integration changes after this import must update this ledger and the vendor note. Runtime loading uses package-local classes, not arbitrary `trust_remote_code`. CLI/prompt-service implementations are not vendored. The separately listed prompt templates below are the only copied repository assets.

`backend/core/models/ming_image_01/attention.py` adapts the Apache-2.0 native single-stream processor in `vendor/diffusion/transformer.py` to SushiUI’s attention conduit. Its original Ant Group copyright and Apache-2.0 notice are retained; Q/K normalization, complex RoPE and output projection semantics are preserved. The pinned vendor source remains unchanged.

Prompt rewriting templates in `backend/core/extensions/ming_image_01_prompt_templates/` derive from the same pinned MIT source: `design.txt` is an unmodified byte copy of `assets/t2i_rewriter_system_prompt.txt` (SHA-256 `a6107c0f225f755f8ed293e616ef4bbc0e3a61b935893d0e6d3075d4b98200b1`); `layer.txt` extracts the README `GUIDED_PROMPT` literal without changing its contents. No remote prompt service is bundled or automatically contacted.
