"""
Tiny AutoEncoder for Stable Diffusion (TAESD)
Provides fast latent preview decoding during generation
"""
from typing import Dict, Optional
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
        # Decode-error rate limiter: full traceback once per `tag`, then
        # collapse repeats to a counted one-liner so a broken preview path
        # doesn't spam the log nor look like the run has frozen.
        self._decode_error_counts: Dict[str, int] = {}

    def _log_decode_error(self, tag: str, exc: BaseException) -> None:
        n = self._decode_error_counts.get(tag, 0) + 1
        self._decode_error_counts[tag] = n
        if n == 1:
            print(f"[TAESD] Failed to decode {tag} preview: {exc}")
            import traceback
            traceback.print_exc()
            print(f"[TAESD] (suppressing further {tag} preview-decode errors; "
                  f"count will be reported at the end of generation)")
        elif n in (10, 100, 1000):
            print(f"[TAESD] {tag} preview-decode error count: {n}")

    def reset_decode_error_counts(self) -> Dict[str, int]:
        """Return and clear the accumulated decode-error counts.
        Call at the end of a generation to emit a final summary line."""
        counts = dict(self._decode_error_counts)
        self._decode_error_counts.clear()
        return counts

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

    def load_taesd(self, is_sdxl: bool = False, is_zimage: bool = False, is_deus: bool = False, is_zimage_sdxl_vae: bool = False, is_flux2: bool = False, is_anima: bool = False, is_lens: bool = False):
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
        # Lens uses AutoencoderKLFlux2 (128-ch patchified flat sequence); no TAESD.
        if is_lens:
            return None
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

    def decode_latent(self, latent: torch.Tensor, is_sdxl: bool = False, is_zimage: bool = False, is_deus: bool = False, is_zimage_sdxl_vae: bool = False, is_flux2: bool = False, is_anima: bool = False, is_lens: bool = False, image_width: Optional[int] = None, image_height: Optional[int] = None) -> Optional[Image.Image]:
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

        # Lens: 128-ch patchified flat-sequence latent (AutoencoderKLFlux2)
        if is_lens:
            return self._decode_lens_latent_preview(latent, image_width, image_height)

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
            self._log_decode_error("TAESD", e)
            return None

    def _decode_lens_latent_preview(self, latent: torch.Tensor, image_width: Optional[int] = None, image_height: Optional[int] = None) -> Optional[Image.Image]:
        """Decode Lens flat-sequence latent [B, N, 128] to preview image.

        Lens uses AutoencoderKLFlux2 with 128-ch patchified flat sequences
        (32ch VAE output → 2×2 patchify → 128ch per token at latent_h×latent_w).
        Snaps image dimensions to the nearest Lens resolution bucket so the
        preview is always correct even when the user requested an unregistered
        resolution.
        """
        try:
            with torch.no_grad():
                latent = latent.cpu().to(torch.float32)

                if latent.ndim != 3:
                    return None

                batch_size, num_tokens, channels = latent.shape
                if channels != 128:
                    return None

                # Snap to nearest Lens bucket to get correct latent_h / latent_w.
                # This handles the case where params.width/height haven't been snapped yet.
                if image_width is not None and image_height is not None:
                    try:
                        from core.models.lens.lens_resolution import find_nearest_bucket
                        snapped_w, snapped_h = find_nearest_bucket(image_width, image_height)
                        latent_h = snapped_h // 16
                        latent_w = snapped_w // 16
                    except Exception:
                        latent_h, latent_w = self._find_best_factors(num_tokens)
                    if latent_h * latent_w != num_tokens:
                        latent_h, latent_w = self._find_best_factors(num_tokens)
                else:
                    latent_h, latent_w = self._find_best_factors(num_tokens)

                # [B, N, 128] → [B, 128, latent_h, latent_w]
                latent = latent.permute(0, 2, 1).reshape(batch_size, 128, latent_h, latent_w)

                # Unpatchify: [B, 128, H, W] → [B, 32, H*2, W*2]
                # mirrors vae_decode rearrange "b (h w) (c p1 p2) -> b c (h p1) (w p2)"
                unpatch = latent.reshape(batch_size, 32, 2, 2, latent_h, latent_w)
                unpatch = unpatch.permute(0, 1, 4, 2, 5, 3)
                unpatch = unpatch.reshape(batch_size, 32, latent_h * 2, latent_w * 2)

                # RGB projection using FLUX.2 factors (shared AutoencoderKLFlux2 latent space)
                rgb_factors = torch.tensor(self._FLUX2_LATENT_RGB_FACTORS, dtype=unpatch.dtype)
                rgb_bias = torch.tensor(self._FLUX2_LATENT_RGB_BIAS, dtype=unpatch.dtype)
                rgb = torch.einsum('bchw,cn->bnhw', unpatch, rgb_factors) + rgb_bias.view(1, 3, 1, 1)
                rgb = rgb[0]  # [3, H, W]
                rgb = (rgb.clamp(-1.0, 1.0) + 1.0) / 2.0
                rgb_np = (rgb.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

                preview = Image.fromarray(rgb_np, mode='RGB')
                # Lens VAE: 8x spatial downscale from image to latent_h*2 / latent_w*2
                preview = preview.resize(
                    (preview.width * 8, preview.height * 8),
                    Image.Resampling.BILINEAR,
                )
                return preview
        except Exception as e:
            self._log_decode_error("Lens", e)
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

                # FLUX.2 VAE 32ch -> RGB linear projection. The coefficients
                # are a numerical fit of the VAE decoder's first-order response
                # — equivalent to per-channel principal components and
                # reproducible by PCA on any sample of clean FLUX.2 latents.
                # Far more recognisable mid-denoising than a raw channel slice.
                # `latent` here is [B, 32, H, W] after the unpatchify above.
                rgb_factors = torch.tensor(self._FLUX2_LATENT_RGB_FACTORS, dtype=latent.dtype)
                rgb_bias = torch.tensor(self._FLUX2_LATENT_RGB_BIAS, dtype=latent.dtype)
                rgb = torch.einsum('bchw,cn->bnhw', latent, rgb_factors) + rgb_bias.view(1, 3, 1, 1)
                rgb = rgb[0]  # [3, H, W]
                rgb = (rgb.clamp(-1.0, 1.0) + 1.0) / 2.0
                rgb_np = (rgb.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                preview = Image.fromarray(rgb_np, mode='RGB')

                # Upscale to original image size (latent is 1/8 of image size)
                target_width = preview.width * 8
                target_height = preview.height * 8
                preview = preview.resize((target_width, target_height),
                                          Image.Resampling.BILINEAR)
                return preview

        except Exception as e:
            self._log_decode_error("FLUX.2", e)
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

    # FLUX.2 VAE 32ch -> RGB linear projection. Same numerical-fit /
    # PCA-style coefficients as above, sized for the 32-channel FLUX.2 base
    # latent (applied AFTER 128->32 unpatchify).
    _FLUX2_LATENT_RGB_FACTORS = [
        [ 0.0058,  0.0113,  0.0073],
        [ 0.0495,  0.0443,  0.0836],
        [-0.0099,  0.0096,  0.0644],
        [ 0.2144,  0.3009,  0.3652],
        [ 0.0166, -0.0039, -0.0054],
        [ 0.0157,  0.0103, -0.0160],
        [-0.0398,  0.0902, -0.0235],
        [-0.0052,  0.0095,  0.0109],
        [-0.3527, -0.2712, -0.1666],
        [-0.0301, -0.0356, -0.0180],
        [-0.0107,  0.0078,  0.0013],
        [ 0.0746,  0.0090, -0.0941],
        [ 0.0156,  0.0169,  0.0070],
        [-0.0034, -0.0040, -0.0114],
        [ 0.0032,  0.0181,  0.0080],
        [-0.0939, -0.0008,  0.0186],
        [ 0.0018,  0.0043,  0.0104],
        [ 0.0284,  0.0056, -0.0127],
        [-0.0024, -0.0022, -0.0030],
        [ 0.1207, -0.0026,  0.0065],
        [ 0.0128,  0.0101,  0.0142],
        [ 0.0137, -0.0072, -0.0007],
        [ 0.0095,  0.0092, -0.0059],
        [ 0.0000, -0.0077, -0.0049],
        [-0.0465, -0.0204, -0.0312],
        [ 0.0095,  0.0012, -0.0066],
        [ 0.0290, -0.0034,  0.0025],
        [ 0.0220,  0.0169, -0.0048],
        [-0.0332, -0.0457, -0.0468],
        [-0.0085,  0.0389,  0.0609],
        [-0.0076,  0.0003, -0.0043],
        [-0.0111, -0.0460, -0.0614],
    ]
    _FLUX2_LATENT_RGB_BIAS = [-0.0329, -0.0718, -0.0851]

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
            self._log_decode_error("Anima", e)
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
