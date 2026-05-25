"""Anima (Cosmos-Predict2-based DiT) architecture support for SushiUI.

Vendored from kohya-ss/sd-scripts (Apache-2.0) and adapted for inference.

Model architecture: NVIDIA Cosmos-Predict2 derived DiT
- 28 transformer blocks with AdaLN-LoRA modulation, 3D RoPE
- Qwen3-0.6B text encoder + 6-layer LLM Adapter (Qwen3 -> T5-compatible space)
- Qwen-Image VAE (16ch latents, 8x spatial downscale)
- Rectified Flow / Flow Matching scheduler

Distribution formats supported:
  1. Split files (HuggingFace official):
       <root>/split_files/diffusion_models/*.safetensors
       <root>/split_files/text_encoders/qwen_3_06b_*.safetensors
       <root>/split_files/vae/qwen_image_vae*.safetensors
  2. Single DiT safetensors file (derivative models like OOO_Anima):
       text encoder + VAE must be auto-discovered under
       models/text_encoders/ and models/vae/, or provided via settings.
"""
