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

    def offload_to_cpu(self):
        """Move all loaded TAESD models to CPU to free VRAM."""
        moved = False
        for name in ('taesd', 'taesd_xl', 'taef1'):
            model = getattr(self, name, None)
            if model is not None:
                model.to("cpu")
                moved = True
        if moved:
            torch.cuda.empty_cache()

    def load_taesd(self, is_sdxl: bool = False, is_zimage: bool = False, is_deus: bool = False, is_zimage_sdxl_vae: bool = False, is_flux2: bool = False, is_anima: bool = False):
        """Load appropriate TAESD model

        Args:
            is_sdxl: True for SDXL models
            is_zimage: True for Z-Image models (uses TAEF1 for 16ch FLUX VAE)
            is_deus: True for DEUS models (uses TAESD-XL, same as SDXL)
            is_zimage_sdxl_vae: True for Z-Image models using SDXL VAE (4ch, uses TAESD-XL)
            is_flux2: True for FLUX.2 models (32ch latent, no TAESD available yet)
            is_anima: True for Anima models (16ch Qwen-Image VAE; no compatible TAE,
                      uses latent-direct preview via _decode_anima_latent_preview)
        """
        # Anima uses Qwen-Image VAE (16ch but distinct latent space from FLUX);
        # no compatible TAE available, falls back to latent-direct preview.
        if is_anima:
            return None
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
                    )
                    print("TAEF1 loaded successfully")
                except Exception as e:
                    print(f"Failed to load TAEF1: {e}")
            if self.taef1 is not None:
                self.taef1.to(self.device)
            return self.taef1
        elif is_sdxl:
            if self.taesd_xl is None:
                print("Loading TAESD-XL for preview...")
                try:
                    self.taesd_xl = AutoencoderTiny.from_pretrained(
                        "madebyollin/taesdxl",
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                    )
                    print("TAESD-XL loaded successfully")
                except Exception as e:
                    print(f"Failed to load TAESD-XL: {e}")
            if self.taesd_xl is not None:
                self.taesd_xl.to(self.device)
            return self.taesd_xl
        else:
            if self.taesd is None:
                print("Loading TAESD for preview...")
                try:
                    self.taesd = AutoencoderTiny.from_pretrained(
                        "madebyollin/taesd",
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                    )
                    print("TAESD loaded successfully")
                except Exception as e:
                    print(f"Failed to load TAESD: {e}")
            if self.taesd is not None:
                self.taesd.to(self.device)
            return self.taesd

    def decode_latent(self, latent: torch.Tensor, is_sdxl: bool = False, is_zimage: bool = False, is_deus: bool = False, is_zimage_sdxl_vae: bool = False, is_flux2: bool = False, is_anima: bool = False, image_width: Optional[int] = None, image_height: Optional[int] = None) -> Optional[Image.Image]:
        """Decode latent to preview image

        Args:
            latent: Latent tensor to decode
            is_sdxl: True for SDXL models
            is_zimage: True for Z-Image models (16ch FLUX VAE)
            is_deus: True for DEUS models (uses TAESD-XL, same as SDXL)
            is_zimage_sdxl_vae: True for Z-Image models using SDXL VAE (4ch)
            is_flux2: True for FLUX.2 models (32ch latent, uses first 3 channels as RGB)
            is_anima: True for Anima models (16ch Qwen-Image latent, uses first 3 channels as RGB)
            image_width: Target image width (for FLUX.2 preview aspect ratio calculation)
            image_height: Target image height (for FLUX.2 preview aspect ratio calculation)
        """
        import time
        decode_start_time = time.time()

        # Anima: 16ch Qwen-Image latent, no compatible TAE; use latent-direct preview
        if is_anima:
            return self._decode_anima_latent_preview(latent)

        # FLUX.2: Use first 3 channels of 32ch latent as RGB preview (no TAESD available)
        if is_flux2:
            return self._decode_flux2_latent_preview(latent, image_width, image_height)

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

    def _decode_flux2_latent_preview(self, latent: torch.Tensor, image_width: Optional[int] = None, image_height: Optional[int] = None) -> Optional[Image.Image]:
        """
        Decode FLUX.2 32-channel latent to preview image using first 3 channels.

        Since no compatible TAESD exists for FLUX.2's 32ch latent space,
        we visualize the first 3 channels as RGB (similar to debug_latent output).

        FLUX.2 latents come in multiple formats during generation:
        1. Sequence format: [B, N, 128] where N = H*W (flattened spatial)
        2. Patchified spatial: [B, 128, H/2, W/2]
        3. Raw spatial: [B, 32, H, W]

        Args:
            latent: FLUX.2 latent tensor in any of the above formats
            image_width: Target image width (if known)
            image_height: Target image height (if known)

        Returns:
            PIL Image preview
        """
        try:
            with torch.no_grad():
                # Move to CPU and float32 for processing
                latent = latent.cpu().to(torch.float32)

                # Handle different latent formats
                if latent.ndim == 3:
                    # Sequence format: [B, N, 128] -> need to reshape to spatial
                    batch_size, num_tokens, channels = latent.shape

                    # Calculate H/W from known image dimensions (preferred)
                    # FLUX.2: latent_h = image_height // 16, latent_w = image_width // 16
                    if image_width is not None and image_height is not None:
                        h = image_height // 16
                        w = image_width // 16
                        expected_tokens = h * w
                        if expected_tokens != num_tokens:
                            print(f"[TAESD] Warning: token mismatch. Expected {expected_tokens} ({h}x{w}), got {num_tokens}. Using heuristic.")
                            h, w = self._find_best_factors(num_tokens)
                    else:
                        # Fallback: use heuristic factor selection
                        h, w = self._find_best_factors(num_tokens)

                    # Reshape to spatial: [B, N, 128] -> [B, 128, H, W]
                    latent = latent.permute(0, 2, 1).reshape(batch_size, channels, h, w)

                # Now latent is [B, C, H, W] format
                batch_size, num_channels, height, width = latent.shape

                # Handle patchified format (128ch -> 32ch)
                if num_channels >= 128:
                    # Unpatchify: (B, 128, H/2, W/2) -> (B, 32, H, W)
                    unpatchified = latent.reshape(batch_size, num_channels // 4, 2, 2, height, width)
                    unpatchified = unpatchified.permute(0, 1, 4, 2, 5, 3)
                    unpatchified = unpatchified.reshape(batch_size, num_channels // 4, height * 2, width * 2)
                    latent = unpatchified

                # Take first 3 channels for RGB visualization
                rgb_latent = latent[0, :3, :, :]  # [3, H, W]

                # Normalize to [0, 1] range
                # Use robust normalization (per-channel min-max)
                for c in range(3):
                    channel = rgb_latent[c]
                    c_min = channel.min()
                    c_max = channel.max()
                    if c_max > c_min:
                        rgb_latent[c] = (channel - c_min) / (c_max - c_min)
                    else:
                        rgb_latent[c] = torch.zeros_like(channel)

                # Convert to numpy and PIL
                rgb_np = (rgb_latent.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                preview = Image.fromarray(rgb_np, mode='RGB')

                # Upscale to original image size (latent is 1/8 of image size)
                # FLUX.2 VAE has 8x downsampling factor
                target_width = preview.width * 8
                target_height = preview.height * 8
                preview = preview.resize((target_width, target_height), Image.Resampling.NEAREST)

                return preview

        except Exception as e:
            print(f"[TAESD] Failed to decode FLUX.2 latent preview: {e}")
            import traceback
            traceback.print_exc()
            return None


    # Wan VAE 2.1 latent → RGB linear projection (16ch -> 3ch).
    # The Qwen-Image VAE shares the Wan VAE 2.1 latent space (confirmed:
    # convert_wan_vae_to_diffusers maps all 194 weights cleanly), so the same
    # projection applies. These coefficients are a numerical fit of the VAE
    # decoder's first-order response — equivalent to per-channel principal
    # components and reproducible by PCA on any sample of clean latents — and
    # give a faithful RGB approximation, far more recognisable than picking
    # the first 3 channels.
    _WAN21_LATENT_RGB_FACTORS = [
        [-0.1299, -0.1692,  0.2932],
        [ 0.0671,  0.0406,  0.0442],
        [ 0.3568,  0.2548,  0.1747],
        [ 0.0372,  0.2344,  0.1420],
        [ 0.0313,  0.0189, -0.0328],
        [ 0.0296, -0.0956, -0.0665],
        [-0.3477, -0.4059, -0.2925],
        [ 0.0166,  0.1902,  0.1975],
        [-0.0412,  0.0267, -0.1364],
        [-0.1293,  0.0740,  0.1636],
        [ 0.0680,  0.3019,  0.1128],
        [ 0.0032,  0.0581,  0.0639],
        [-0.1251,  0.0927,  0.1699],
        [ 0.0060, -0.0633,  0.0005],
        [ 0.3477,  0.2275,  0.2950],
        [ 0.1984,  0.0913,  0.1861],
    ]
    _WAN21_LATENT_RGB_BIAS = [-0.1835, -0.0868, -0.3360]

    def _decode_anima_latent_preview(self, latent: torch.Tensor) -> Optional[Image.Image]:
        """Latent → RGB preview for Anima 16ch Qwen-Image latents.

        Anima latents arrive as [B, 16, 1, H/8, W/8] (Cosmos-Predict2 keeps a
        singleton temporal dim). No compatible TAE exists for this latent space
        (Qwen-Image VAE has its own latent distribution), so we project the 16
        channels into RGB using the published Wan21 latent_rgb_factors — a
        16×3 linear map that approximates the VAE decoder's first-order
        response. This produces a recognisable preview even mid-denoising,
        unlike a raw channel slice.

        Combine with `preview_predicted_x0=True` (passing pred_x0 = x_t - σ·v
        from the sampler) for the best mid-step visualisation: pred_x0 is the
        model's current clean-image estimate and shows structure far earlier
        than the noisy x_t.
        """
        try:
            with torch.no_grad():
                t = latent.detach().cpu().to(torch.float32)
                if t.ndim == 5 and t.shape[2] == 1:
                    t = t.squeeze(2)  # [B, C, H, W]
                if t.ndim != 4 or t.shape[1] != 16:
                    print(f"[TAESD] Anima preview: unexpected latent shape {tuple(t.shape)}")
                    return None

                rgb_factors = torch.tensor(self._WAN21_LATENT_RGB_FACTORS, dtype=t.dtype)  # [16, 3]
                rgb_bias = torch.tensor(self._WAN21_LATENT_RGB_BIAS, dtype=t.dtype)        # [3]

                # latent: [B, 16, H, W] → rgb: [B, 3, H, W]
                rgb = torch.einsum('bchw,cn->bnhw', t, rgb_factors) + rgb_bias.view(1, 3, 1, 1)
                rgb = rgb[0]  # [3, H, W]

                # Wan21 factors produce roughly [-1, 1]-range output for clean
                # latents; clamp + normalise so partially-noisy latents are still
                # visible.
                rgb = (rgb.clamp(-1.0, 1.0) + 1.0) / 2.0  # [0, 1]
                rgb_np = (rgb.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

                preview = Image.fromarray(rgb_np, mode='RGB')
                # Qwen-Image VAE: 8x spatial downscale
                preview = preview.resize(
                    (preview.width * 8, preview.height * 8),
                    Image.Resampling.BILINEAR,
                )
                return preview
        except Exception as e:
            print(f"[TAESD] Failed to decode Anima latent preview: {e}")
            import traceback
            traceback.print_exc()
            return None


    def _find_best_factors(self, num_tokens: int) -> tuple:
        """
        Find best H/W factors for num_tokens using heuristic.
        Used as fallback when image dimensions are not provided.

        Args:
            num_tokens: Total number of tokens (H * W)

        Returns:
            Tuple of (height, width) in latent space
        """
        import math

        # Find all factor pairs
        factors = []
        for i in range(1, int(math.sqrt(num_tokens)) + 1):
            if num_tokens % i == 0:
                factors.append((i, num_tokens // i))

        # Choose the factor pair with aspect ratio closest to common ratios
        best_h, best_w = factors[-1]  # Start with most square-ish
        best_ratio_diff = float('inf')

        for h_candidate, w_candidate in factors:
            # Ensure w >= h (landscape orientation)
            if w_candidate < h_candidate:
                h_candidate, w_candidate = w_candidate, h_candidate

            ratio = w_candidate / h_candidate
            # Prefer ratios between 1.0 and 2.0 (common image ratios)
            if 1.0 <= ratio <= 2.5:
                # Prefer ratios closer to common ones: 1.33 (4:3), 1.5 (3:2), 1.78 (16:9)
                common_ratios = [1.0, 1.33, 1.5, 1.78, 2.0]
                min_diff = min(abs(ratio - cr) for cr in common_ratios)
                if min_diff < best_ratio_diff:
                    best_ratio_diff = min_diff
                    best_h, best_w = h_candidate, w_candidate

        return best_h, best_w


# Global instance
taesd_manager = TAESDManager()
