"""One-off real SDXL forward/backward check for the style IP-Adapter bridge."""

import sys
from pathlib import Path

import torch
from diffusers import DDPMScheduler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from core.style_encoder.sdxl_ip_adapter import SDXLStyleIPAdapter
from core.style_encoder.sdxl_ip_runtime import load_sdxl_style_pipeline
from core.style_encoder.sdxl_ip_training import install_style_ip_training_processors, sdxl_style_ip_denoising_loss


def main() -> None:
    model_path = Path(__file__).resolve().parents[4] / "models" / "NoobAI-XL-Vpred-v1.0.safetensors"
    print(f"Loading {model_path}", flush=True)
    pipe = load_sdxl_style_pipeline(str(model_path))
    unet = pipe.unet.to("cuda").eval().requires_grad_(False)
    pipe.text_encoder.to("cuda")
    pipe.text_encoder_2.to("cuda")
    prompt_embeds, _, pooled, _ = pipe.encode_prompt(
        "1girl, portrait, pink hair", device="cuda", do_classifier_free_guidance=False,
    )
    adapter = SDXLStyleIPAdapter(unet).to(device="cuda", dtype=torch.bfloat16)
    print(f"cross_attention_layers={len(adapter.layer_names)} bridge_parameters="
          f"{sum(parameter.numel() for parameter in adapter.parameters())}", flush=True)
    install_style_ip_training_processors(unet)
    scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
        num_train_timesteps=1000, clip_sample=False, prediction_type="v_prediction",
    )
    latent = torch.randn(1, 4, 32, 32, device="cuda", dtype=torch.bfloat16)
    noise = torch.randn_like(latent)
    text = prompt_embeds
    cond = {
        "text_embeds": pooled,
        "time_ids": torch.tensor([[256, 256, 0, 0, 256, 256]], device="cuda", dtype=torch.bfloat16),
    }
    loss = sdxl_style_ip_denoising_loss(
        unet, scheduler, adapter, clean_latents=latent,
        style_embedding=torch.randn(1, 512, device="cuda", dtype=torch.bfloat16),
        prompt_embeds=text, added_cond_kwargs=cond, timesteps=torch.tensor([500], device="cuda"),
        noise=noise,
    )
    loss.backward()
    print(f"loss={float(loss):.5f} finite_grad={bool(torch.isfinite(adapter.projection.weight.grad).all())} "
          f"peak_vram_gb={torch.cuda.max_memory_allocated() / 2**30:.2f}", flush=True)


if __name__ == "__main__":
    main()
