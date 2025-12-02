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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_taesd(self, is_sdxl: bool = False):
        """Load appropriate TAESD model"""
        if is_sdxl:
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

    def decode_latent(self, latent: torch.Tensor, is_sdxl: bool = False) -> Optional[Image.Image]:
        """Decode latent to preview image"""
        try:
            decoder = self.load_taesd(is_sdxl)
            if decoder is None:
                return None

            # Decode latent
            with torch.no_grad():
                # Move latent to correct device
                latent = latent.to(self.device)

                # TAESD expects latents to be scaled
                if is_sdxl:
                    # SDXL uses scaling factor 0.13025
                    scaled_latent = latent / 0.13025
                else:
                    # SD1.5 uses scaling factor 0.18215
                    scaled_latent = latent / 0.18215

                # Decode using the decode method
                image = decoder.decode(scaled_latent).sample

                # Convert to PIL Image
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).numpy()
                image = (image[0] * 255).astype(np.uint8)
                return Image.fromarray(image)

        except Exception as e:
            print(f"Failed to decode latent: {e}")
            return None

# Global instance
taesd_manager = TAESDManager()
