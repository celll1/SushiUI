"""
Tiny AutoEncoder for Stable Diffusion (TAESD)
Provides fast latent preview decoding during generation
"""
from typing import Optional
import torch
from diffusers import AutoencoderTiny
from PIL import Image
import numpy as np

class TAESDManager:
    def __init__(self):
        self.taesd = None
        self.taesd_xl = None
        self.taef1 = None  # For Z-Image (FLUX-based)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_taesd(self, is_sdxl: bool = False, is_zimage: bool = False, is_deus: bool = False, is_zimage_sdxl_vae: bool = False, is_flux2: bool = False):
        """Load appropriate TAESD model

        Args:
            is_sdxl: True for SDXL models
            is_zimage: True for Z-Image models (uses TAEF1 for 16ch FLUX VAE)
            is_deus: True for DEUS models (uses TAESD-XL, same as SDXL)
            is_zimage_sdxl_vae: True for Z-Image models using SDXL VAE (4ch, uses TAESD-XL)
            is_flux2: True for FLUX.2 models (32ch latent, no TAESD available yet)
        """
        # FLUX.2 uses 32-channel latents, no compatible TAESD available
        if is_flux2:
            print("[TAESD] FLUX.2 models use 32-channel latents - no compatible preview decoder available")
            return None
        # DEUS uses SDXL VAE (same scaling factor 0.13025), so use TAESD-XL
        if is_deus:
            is_sdxl = True

        # Z-Image with SDXL VAE (4ch) uses TAESD-XL instead of TAEF1
        if is_zimage_sdxl_vae:
            is_zimage = False
            is_sdxl = True

        if is_zimage:
            if self.taef1 is None:
                print("Loading TAEF1 for Z-Image preview...")
                try:
                    self.taef1 = AutoencoderTiny.from_pretrained(
                        "madebyollin/taef1",
                        torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32
                    ).to(self.device)
                    print("TAEF1 loaded successfully")
                except Exception as e:
                    print(f"Failed to load TAEF1: {e}")
            return self.taef1
        elif is_sdxl:
            if self.taesd_xl is None:
                print("Loading TAESD-XL for preview...")
                try:
                    self.taesd_xl = AutoencoderTiny.from_pretrained(
                        "madebyollin/taesdxl",
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                    ).to(self.device)
                    print("TAESD-XL loaded successfully")
                except Exception as e:
                    print(f"Failed to load TAESD-XL: {e}")
            return self.taesd_xl
        else:
            if self.taesd is None:
                print("Loading TAESD for preview...")
                try:
                    self.taesd = AutoencoderTiny.from_pretrained(
                        "madebyollin/taesd",
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                    ).to(self.device)
                    print("TAESD loaded successfully")
                except Exception as e:
                    print(f"Failed to load TAESD: {e}")
            return self.taesd

    def decode_latent(self, latent: torch.Tensor, is_sdxl: bool = False, is_zimage: bool = False, is_deus: bool = False, is_zimage_sdxl_vae: bool = False, is_flux2: bool = False) -> Optional[Image.Image]:
        """Decode latent to preview image

        Args:
            latent: Latent tensor to decode
            is_sdxl: True for SDXL models
            is_zimage: True for Z-Image models (16ch FLUX VAE)
            is_deus: True for DEUS models (uses TAESD-XL, same as SDXL)
            is_zimage_sdxl_vae: True for Z-Image models using SDXL VAE (4ch)
            is_flux2: True for FLUX.2 models (32ch latent, no preview available)
        """
        import time
        decode_start_time = time.time()

        # DEUS uses SDXL VAE (same scaling factor 0.13025), so use TAESD-XL
        if is_deus:
            is_sdxl = True

        # Z-Image with SDXL VAE (4ch) uses TAESD-XL and SDXL scaling factor
        if is_zimage_sdxl_vae:
            is_zimage = False
            is_sdxl = True

        try:
            # Load TAESD model (may be cached, so this should be fast if already loaded)
            load_start_time = time.time()
            decoder = self.load_taesd(is_sdxl, is_zimage, is_deus, is_zimage_sdxl_vae, is_flux2)
            load_time = (time.time() - load_start_time) * 1000
            
            if decoder is None:
                return None
            
            # Log load time only if significant (first load)
            if load_time > 10:  # More than 10ms indicates actual loading
                print(f"[TAESD] Model load time: {load_time:.2f}ms")

            # Decode latent
            decode_internal_start_time = time.time()
            with torch.no_grad():
                # Move latent to correct device and dtype
                # TAEF1 uses BF16, TAESD/TAESD-XL use FP16 or FP32
                if is_zimage:
                    # TAEF1 expects BF16 (Z-Image uses FLUX VAE)
                    latent = latent.to(device=self.device, dtype=torch.bfloat16)
                else:
                    # TAESD/TAESD-XL expect FP16 on GPU, FP32 on CPU
                    target_dtype = torch.float16 if self.device == "cuda" else torch.float32
                    latent = latent.to(device=self.device, dtype=target_dtype)

                # TAESD expects latents to be scaled
                if is_zimage:
                    # Z-Image (FLUX-based) use scaling factor 0.3611
                    # Same as FLUX.1: https://huggingface.co/black-forest-labs/FLUX.1-dev
                    scaled_latent = latent / 0.3611
                elif is_sdxl:
                    # SDXL uses scaling factor 0.13025
                    scaled_latent = latent / 0.13025
                else:
                    # SD1.5 uses scaling factor 0.18215
                    scaled_latent = latent / 0.18215

                # Decode using the decode method
                image = decoder.decode(scaled_latent).sample

                # Convert to PIL Image
                image = (image / 2 + 0.5).clamp(0, 1)
                # NumPy doesn't support BFloat16, convert to FP32 first for Z-Image
                if is_zimage:
                    image = image.cpu().to(torch.float32).permute(0, 2, 3, 1).numpy()
                else:
                    image = image.cpu().permute(0, 2, 3, 1).numpy()
                image = (image[0] * 255).astype(np.uint8)
                
                decode_internal_time = (time.time() - decode_internal_start_time) * 1000
                total_decode_time = (time.time() - decode_start_time) * 1000
                
                # Log decode timing for Step1 debug (only if significant)
                if total_decode_time > 50:  # More than 50ms
                    print(f"[TAESD] Decode time: {decode_internal_time:.2f}ms (total: {total_decode_time:.2f}ms)")
                
                return Image.fromarray(image)

        except Exception as e:
            print(f"Failed to decode latent: {e}")
            return None

# Global instance
taesd_manager = TAESDManager()
