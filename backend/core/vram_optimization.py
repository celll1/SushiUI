"""VRAM Optimization utilities for sequential model loading

This module implements sequential VRAM loading:
- Text Encoder → CPU after encoding
- U-Net → GPU only during inference
- VAE → GPU only during decode

Also supports on-demand U-Net FP8 quantization for VRAM reduction.
Note: torchao/UINT quantization has been removed due to compatibility issues.
"""

import torch
from typing import Dict, Optional
import copy


def _add_generation_warning(message: str, code: str = None) -> None:
    """Best-effort: record a feature-degradation warning for the current generation.

    Lazily imported so this module never hard-depends on the api package at
    import time. Never raises.
    """
    try:
        from api.generation_status import add_warning
        add_warning(message, code=code)
    except Exception:
        pass


def log_device_status(stage: str, pipeline, show_details: bool = False, zimage_components: dict = None, vision_encoder=None):
    """Log device status of all pipeline components

    Args:
        stage: Description of current stage (e.g., "After moving to GPU")
        pipeline: The diffusers pipeline (or None for Z-Image)
        show_details: Show detailed submodule information
        zimage_components: Dict with Z-Image components (text_encoder, transformer, vae)
        vision_encoder: Optional SigLIP2VisionEncoderWrapper instance
    """
    print(f"\n{'='*60}")
    print(f"[VRAM] Device Status: {stage}")
    print(f"{'='*60}")

    def get_dtype_info(module):
        """Get dtype information from module parameters"""
        try:
            param = next(module.parameters())
            return param.dtype
        except:
            return "unknown"

    def check_quantization(module):
        """Check if module is quantized (FP8, etc.)"""
        # Check first few Linear layers for quantization
        checked_count = 0
        max_check = 5  # Check first 5 linear layers

        for name, submodule in module.named_modules():
            if not isinstance(submodule, torch.nn.Linear):
                continue

            checked_count += 1
            if checked_count > max_check:
                break

            # Check weight attributes for quantization
            if hasattr(submodule, 'weight'):
                weight = submodule.weight

                # Check weight dtype for FP8
                if hasattr(weight, 'dtype'):
                    dtype_str = str(weight.dtype)
                    if 'float8' in dtype_str:
                        dtype_name = dtype_str.replace('torch.', '').upper()
                        return f"quantized ({dtype_name})"

        return None

    # Text Encoder
    if hasattr(pipeline, 'text_encoder') and pipeline.text_encoder is not None:
        try:
            device = next(pipeline.text_encoder.parameters()).device
            dtype = get_dtype_info(pipeline.text_encoder)
            print(f"  Text Encoder:   {device} ({dtype})")
        except:
            print(f"  Text Encoder:   no parameters")

    # Text Encoder 2
    if hasattr(pipeline, 'text_encoder_2') and pipeline.text_encoder_2 is not None:
        try:
            device = next(pipeline.text_encoder_2.parameters()).device
            dtype = get_dtype_info(pipeline.text_encoder_2)
            print(f"  Text Encoder 2: {device} ({dtype})")
        except:
            print(f"  Text Encoder 2: no parameters")

    # Vision Encoder (SigLIP2, if provided)
    if vision_encoder is not None:
        try:
            device = next(vision_encoder.model.parameters()).device
            dtype = get_dtype_info(vision_encoder.model)
            print(f"  Vision Encoder: {device} ({dtype})")
        except Exception:
            print(f"  Vision Encoder: no parameters")

    # U-Net
    if hasattr(pipeline, 'unet') and pipeline.unet is not None:
        try:
            device = next(pipeline.unet.parameters()).device
            dtype = get_dtype_info(pipeline.unet)
            quant_info = check_quantization(pipeline.unet)

            if quant_info:
                print(f"  U-Net:          {device} ({dtype}, {quant_info})")
            else:
                print(f"  U-Net:          {device} ({dtype})")

            if show_details:
                # Check for any CPU submodules
                cpu_modules = []
                for name, module in pipeline.unet.named_modules():
                    try:
                        mod_device = next(module.parameters()).device
                        if mod_device.type == 'cpu':
                            cpu_modules.append(name)
                    except StopIteration:
                        pass

                if cpu_modules:
                    print(f"    WARNING: {len(cpu_modules)} submodules on CPU")
                    for name in cpu_modules[:3]:
                        print(f"      - {name}")
        except:
            print(f"  U-Net:          no parameters")

    # VAE
    if pipeline and hasattr(pipeline, 'vae') and pipeline.vae is not None:
        try:
            device = next(pipeline.vae.parameters()).device
            dtype = get_dtype_info(pipeline.vae)
            print(f"  VAE:            {device} ({dtype})")
        except:
            print(f"  VAE:            no parameters")

    # Z-Image components (if provided)
    if zimage_components:
        if 'text_encoder' in zimage_components and zimage_components['text_encoder'] is not None:
            try:
                device = next(zimage_components['text_encoder'].parameters()).device
                dtype = get_dtype_info(zimage_components['text_encoder'])
                print(f"  Text Encoder (Z-Image): {device} ({dtype})")
            except:
                print(f"  Text Encoder (Z-Image): no parameters")

        # Transformer (equivalent to U-Net)
        if 'transformer' in zimage_components and zimage_components['transformer'] is not None:
            try:
                device = next(zimage_components['transformer'].parameters()).device
                dtype = get_dtype_info(zimage_components['transformer'])
                quant_info = check_quantization(zimage_components['transformer'])

                if quant_info:
                    print(f"  Transformer (Z-Image):  {device} ({dtype}, {quant_info})")
                else:
                    print(f"  Transformer (Z-Image):  {device} ({dtype})")
            except:
                print(f"  Transformer (Z-Image):  no parameters")

        # VAE
        if 'vae' in zimage_components and zimage_components['vae'] is not None:
            try:
                device = next(zimage_components['vae'].parameters()).device
                dtype = get_dtype_info(zimage_components['vae'])
                print(f"  VAE (Z-Image):          {device} ({dtype})")
            except:
                print(f"  VAE (Z-Image):          no parameters")

    # VRAM usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"\n  VRAM: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    print(f"{'='*60}\n")


def _quantize_unet(unet, quantization: str):
    """Create a quantized copy of the U-Net

    Args:
        unet: Original U-Net model (should be on CPU)
        quantization: Quantization type - 'fp8_e4m3fn', 'fp8_e5m2'

    Returns:
        Quantized U-Net model

    Supported quantization types:
        - fp8_e4m3fn, fp8_e5m2: FP8 quantization (via .to(), ~50% VRAM reduction)
          * Weight: FP8, Activation: FP16 (via autocast)
    """
    try:
        if quantization in ['fp8_e4m3fn', 'fp8_e5m2']:
            # Determine FP8 dtype
            if quantization == 'fp8_e4m3fn':
                fp8_dtype = torch.float8_e4m3fn
                dtype_name = "FP8 E4M3FN"
            else:  # fp8_e5m2
                fp8_dtype = torch.float8_e5m2
                dtype_name = "FP8 E5M2"

            print(f"[Quantization] Applying {dtype_name} quantization...")

            # Check PyTorch version
            if not hasattr(torch, 'float8_e4m3fn'):
                print(f"[Quantization] ERROR: PyTorch version {torch.__version__} does not support FP8")
                print(f"[Quantization] FP8 requires PyTorch >= 2.1.0")
                print(f"[Quantization] Falling back to original model without quantization")
                _add_generation_warning(
                    f"U-Net quantization '{quantization}' unavailable (PyTorch {torch.__version__} "
                    f"lacks FP8 support); falling back to full precision",
                    code="quantization_fallback",
                )
                return copy.deepcopy(unet)

            try:
                # Clone the model
                quantized_unet = copy.deepcopy(unet)

                # Convert to FP8 - approach based on kohya-ss
                # Note: nn.Embedding layers don't support FP8, but .to() handles this gracefully
                quantized_unet = quantized_unet.to(dtype=fp8_dtype)

                print(f"[Quantization] Successfully converted U-Net to {dtype_name}")
                print(f"[Quantization] Note: Compute will use mixed precision automatically (autocast)")
                print(f"[Quantization] Estimated memory reduction: ~50%")

                return quantized_unet

            except Exception as e:
                print(f"[Quantization] ERROR during {dtype_name} conversion: {e}")
                import traceback
                traceback.print_exc()
                print(f"[Quantization] Falling back to original model without quantization")
                _add_generation_warning(
                    f"U-Net quantization '{quantization}' failed during conversion ({e}); "
                    f"falling back to full precision",
                    code="quantization_fallback",
                )
                return copy.deepcopy(unet)

        else:
            print(f"[Quantization] Unsupported quantization type: {quantization}")
            print(f"[Quantization] Supported types: fp8_e4m3fn, fp8_e5m2")
            print(f"[Quantization] Falling back to original model without quantization")
            _add_generation_warning(
                f"U-Net quantization '{quantization}' is not a supported type "
                f"(fp8_e4m3fn, fp8_e5m2); falling back to full precision",
                code="quantization_fallback",
            )
            return copy.deepcopy(unet)

    except Exception as e:
        print(f"[Quantization] Error during quantization: {e}")
        print(f"[Quantization] Falling back to original model without quantization")
        _add_generation_warning(
            f"U-Net quantization '{quantization}' failed ({e}); falling back to full precision",
            code="quantization_fallback",
        )
        import traceback
        traceback.print_exc()
        return copy.deepcopy(unet)


def move_text_encoders_to_gpu(pipeline):
    """Move text encoders to GPU for encoding"""
    print("[VRAM] Moving Text Encoders to GPU for encoding...")

    if hasattr(pipeline, 'text_encoder') and pipeline.text_encoder is not None:
        pipeline.text_encoder.to('cuda:0', non_blocking=False)

    if hasattr(pipeline, 'text_encoder_2') and pipeline.text_encoder_2 is not None:
        pipeline.text_encoder_2.to('cuda:0', non_blocking=False)

    # Note: torch.cuda.empty_cache() can cause sync delays over VPN
    # Removed to reduce latency - cache will be freed naturally


def move_text_encoders_to_cpu(pipeline):
    """Move text encoders to CPU to free VRAM"""
    print("[VRAM] Moving Text Encoders to CPU to free VRAM...")

    if hasattr(pipeline, 'text_encoder') and pipeline.text_encoder is not None:
        pipeline.text_encoder.to('cpu', non_blocking=False)

    if hasattr(pipeline, 'text_encoder_2') and pipeline.text_encoder_2 is not None:
        pipeline.text_encoder_2.to('cpu', non_blocking=False)

    # Note: torch.cuda.empty_cache() removed to reduce VPN latency


def move_unet_to_gpu(pipeline, quantization: Optional[str] = None, use_torch_compile: bool = False):
    """Move U-Net to GPU for inference, optionally with quantization and torch.compile

    Args:
        pipeline: The diffusers pipeline
        quantization: Quantization type - None, 'none', 'fp8_e4m3fn', 'fp8_e5m2', etc.
        use_torch_compile: Whether to apply torch.compile for speedup (recommended with quantization)
    """
    # Normalize quantization parameter
    if quantization in [None, "", "none"]:
        quantization = None

    # INT8 is a DiT-only path (RUNTIME_INT8_ARCHS: anima / krea2 / flux2; see
    # _refuse_runtime_int8_elsewhere, whose message renders that tuple).
    # Refused HERE rather than in _quantize_unet so the request never pays for
    # the copy.deepcopy that the unsupported-type branch would do first.
    if _refuse_runtime_int8_elsewhere(quantization, "U-Net"):
        quantization = None

    if hasattr(pipeline, 'unet') and pipeline.unet is not None:
        # Fast path: No quantization and no torch.compile (most common case)
        if not quantization and not use_torch_compile:
            print("[VRAM] Moving U-Net to GPU for inference...")
            # Restore original unet if quantization was used before
            if hasattr(pipeline, '_original_unet'):
                pipeline.unet = pipeline._original_unet
            pipeline.unet.to('cuda:0', non_blocking=False)
            # Note: torch.cuda.empty_cache() removed to reduce VPN latency
            return

        # Complex path: quantization or torch.compile requested
        if quantization:
            print(f"[VRAM] Moving U-Net to GPU with {quantization} quantization...")
            # Store original unet on CPU if not already stored
            if not hasattr(pipeline, '_original_unet'):
                print(f"[VRAM] Storing original U-Net on CPU...")
                pipeline._original_unet = pipeline.unet
                # Ensure original is on CPU
                pipeline._original_unet.to('cpu')

            # Check if we have a cached quantized model
            if not hasattr(pipeline, '_quantized_unet_cache'):
                pipeline._quantized_unet_cache = {}

            # Cache key includes both quantization and compile status
            cache_key = f"{quantization}_compile" if use_torch_compile else quantization

            # Use cached quantized model if available
            if cache_key in pipeline._quantized_unet_cache:
                print(f"[VRAM] Using cached {quantization} quantized U-Net...")
                pipeline.unet = pipeline._quantized_unet_cache[cache_key]
            else:
                # Create new quantized copy and cache it
                print(f"[VRAM] Creating {quantization} quantized U-Net...")
                quantized_unet = _quantize_unet(pipeline._original_unet, quantization)

                # Apply torch.compile if requested
                if use_torch_compile:
                    print(f"[torch.compile] Compiling quantized U-Net for optimized inference...")
                    print(f"[torch.compile] Note: First inference will be slower due to compilation")
                    print(f"[torch.compile] Subsequent inferences will be significantly faster (1.3-2x speedup)")
                    try:
                        quantized_unet = torch.compile(
                            quantized_unet,
                            mode="max-autotune",  # Maximum optimization
                            fullgraph=False,  # Allow graph breaks for compatibility
                        )
                        print(f"[torch.compile] Successfully compiled U-Net")
                    except Exception as e:
                        print(f"[torch.compile] Warning: Compilation failed: {e}")
                        print(f"[torch.compile] Continuing without torch.compile")

                # Keep quantized model on CPU for caching
                quantized_unet.to('cpu')
                pipeline._quantized_unet_cache[cache_key] = quantized_unet
                pipeline.unet = quantized_unet

            # Move to GPU
            pipeline.unet.to('cuda:0')
        else:
            # Restore original unet if quantization was used before
            if hasattr(pipeline, '_original_unet'):
                print(f"[VRAM] Restoring original (non-quantized) U-Net...")
                pipeline.unet = pipeline._original_unet
                # Keep cache but don't delete it (for future use)

            # Apply torch.compile to original model if requested
            if use_torch_compile and not hasattr(pipeline.unet, '_compiled'):
                print(f"[torch.compile] Compiling U-Net for optimized inference...")
                print(f"[torch.compile] Note: First inference will be slower due to compilation")
                try:
                    pipeline.unet = torch.compile(
                        pipeline.unet,
                        mode="max-autotune",
                        fullgraph=False,
                    )
                    pipeline.unet._compiled = True
                    print(f"[torch.compile] Successfully compiled U-Net")
                except Exception as e:
                    print(f"[torch.compile] Warning: Compilation failed: {e}")
                    print(f"[torch.compile] Continuing without torch.compile")

            pipeline.unet.to('cuda:0', non_blocking=False)

    # Note: torch.cuda.empty_cache() removed to reduce VPN latency


def move_unet_to_cpu(pipeline):
    """Move U-Net to CPU to free VRAM"""
    print("[VRAM] Moving U-Net to CPU to free VRAM...")

    if hasattr(pipeline, 'unet') and pipeline.unet is not None:
        pipeline.unet.to('cpu', non_blocking=False)

    # Note: torch.cuda.empty_cache() removed to reduce VPN latency


def move_vae_to_gpu(pipeline):
    """Move VAE to GPU for decode"""
    print("[VRAM] Moving VAE to GPU for decode...")

    if hasattr(pipeline, 'vae') and pipeline.vae is not None:
        pipeline.vae.to('cuda:0', non_blocking=False)

    # Note: torch.cuda.empty_cache() removed to reduce VPN latency


def move_vae_to_cpu(pipeline):
    """Move VAE to CPU to free VRAM"""
    print("[VRAM] Moving VAE to CPU to free VRAM...")

    if hasattr(pipeline, 'vae') and pipeline.vae is not None:
        pipeline.vae.to('cpu', non_blocking=False)

    # Note: torch.cuda.empty_cache() removed to reduce VPN latency


# ============================================================
# Z-Image VRAM Optimization
# ============================================================

def move_zimage_text_encoder_to_gpu(text_encoder, quantization=None):
    """Move Z-Image text encoder to GPU for encoding (with optional quantization)

    Args:
        text_encoder: Z-Image text encoder model
        quantization: Optional quantization type (fp8_e4m3fn, fp8_e5m2, uint2-uint8, etc.)

    Returns:
        text_encoder (potentially quantized copy if quantization is enabled)
    """
    if text_encoder is None:
        return None

    # Fast path: No quantization
    if not quantization or quantization == "none":
        print("[VRAM] Moving Z-Image Text Encoder to GPU for encoding...")
        text_encoder.to('cuda:0', non_blocking=False)
        return text_encoder

    # Quantization path: Create quantized copy and move to GPU
    print(f"[VRAM] Moving Z-Image Text Encoder to GPU with {quantization} quantization...")
    print(f"[Quantization] Creating quantized Text Encoder ({quantization})...")

    # Text Encoder must be on CPU for quantization
    if next(text_encoder.parameters()).device.type != 'cpu':
        print(f"[Quantization] Moving Text Encoder to CPU for quantization...")
        text_encoder.to('cpu')

    # Quantize (creates a copy)
    quantized_text_encoder = _quantize_text_encoder(text_encoder, quantization)

    # Move quantized copy to GPU
    print(f"[Quantization] Moving quantized Text Encoder to GPU...")
    quantized_text_encoder.to('cuda:0', non_blocking=False)

    print(f"[Quantization] Text Encoder quantization complete ({quantization})")

    return quantized_text_encoder


def move_zimage_text_encoder_to_cpu(text_encoder):
    """Move Z-Image text encoder to CPU to free VRAM

    Args:
        text_encoder: Z-Image text encoder model
    """
    print("[VRAM] Moving Z-Image Text Encoder to CPU to free VRAM...")
    if text_encoder is not None:
        text_encoder.to('cpu', non_blocking=False)
        torch.cuda.empty_cache()


def move_zimage_transformer_to_gpu(transformer, quantization: Optional[str] = None):
    """Move Z-Image transformer to GPU for inference, optionally with quantization

    Note: Z-Image transformer does not support torch.compile yet

    Args:
        transformer: Z-Image transformer model
        quantization: Quantization type - None, 'none', 'fp8_e4m3fn', 'fp8_e5m2', etc.

    Returns:
        transformer: Transformer on GPU (may be quantized)
    """
    # Normalize quantization parameter
    if quantization in [None, "", "none"]:
        quantization = None

    if _refuse_runtime_int8_elsewhere(quantization, "Z-Image Transformer"):
        quantization = None

    if transformer is None:
        return transformer

    # Fast path: No quantization (most common case)
    if not quantization:
        print("[VRAM] Moving Z-Image Transformer to GPU for inference...")
        transformer.to('cuda:0', non_blocking=False)
        return transformer

    # Quantization path
    print(f"[VRAM] Moving Z-Image Transformer to GPU with {quantization} quantization...")
    print(f"[VRAM] Note: Quantization for Z-Image is experimental")

    # Store original transformer reference if not already stored
    if not hasattr(transformer, '_original_state'):
        transformer._original_state = True

    # Apply quantization (similar to U-Net quantization)
    try:
        quantized_transformer = _quantize_transformer(transformer, quantization)
        quantized_transformer.to('cuda:0', non_blocking=False)
        return quantized_transformer
    except Exception as e:
        print(f"[VRAM] Warning: Quantization failed: {e}")
        print(f"[VRAM] Falling back to non-quantized transformer")
        _add_generation_warning(
            f"Transformer quantization '{quantization}' failed ({e}); "
            f"falling back to full precision",
            code="quantization_fallback",
        )
        transformer.to('cuda:0', non_blocking=False)
        return transformer


def move_zimage_transformer_to_cpu(transformer):
    """Move Z-Image transformer to CPU to free VRAM

    Args:
        transformer: Z-Image transformer model
    """
    print("[VRAM] Moving Z-Image Transformer to CPU to free VRAM...")
    if transformer is not None:
        transformer.to('cpu', non_blocking=False)


def move_zimage_vae_to_gpu(vae):
    """Move Z-Image VAE to GPU for decode

    Args:
        vae: Z-Image VAE model
    """
    print("[VRAM] Moving Z-Image VAE to GPU for decode...")
    if vae is not None:
        vae.to('cuda:0')


def move_zimage_vae_to_cpu(vae):
    """Move Z-Image VAE to CPU to free VRAM

    Args:
        vae: Z-Image VAE model
    """
    print("[VRAM] Moving Z-Image VAE to CPU to free VRAM...")
    if vae is not None:
        vae.to('cpu')


def _quantize_transformer(transformer, quantization: str):
    """Create a quantized copy of Z-Image transformer

    Z-Image transformer requires special FP8 handling:
    - FP8 must only be applied to Linear layer WEIGHTS, not buffers
    - Standard .to() converts everything (weights + buffers), causing dtype mismatch
    - Solution: Manually iterate through Linear layers and convert only weights

    Args:
        transformer: Original Z-Image transformer model
        quantization: Quantization type - 'fp8_e4m3fn', 'fp8_e5m2'

    Returns:
        Quantized transformer model
    """
    print(f"[Quantization] Applying {quantization} to Z-Image Transformer...")

    # FP8 quantization: weight-only conversion (manual)
    if quantization in ['fp8_e4m3fn', 'fp8_e5m2']:
        # Determine FP8 dtype
        if quantization == 'fp8_e4m3fn':
            fp8_dtype = torch.float8_e4m3fn
            dtype_name = "FP8 E4M3FN"
        else:
            fp8_dtype = torch.float8_e5m2
            dtype_name = "FP8 E5M2"

        print(f"[Quantization] Applying {dtype_name} quantization (weight-only)...")

        # Check PyTorch version
        if not hasattr(torch, 'float8_e4m3fn'):
            print(f"[Quantization] ERROR: PyTorch version {torch.__version__} does not support FP8")
            print(f"[Quantization] FP8 requires PyTorch >= 2.1.0")
            print(f"[Quantization] Falling back to original model without quantization")
            _add_generation_warning(
                f"Transformer quantization '{quantization}' unavailable (PyTorch "
                f"{torch.__version__} lacks FP8 support); falling back to full precision",
                code="quantization_fallback",
            )
            return copy.deepcopy(transformer)

        try:
            # Clone the model
            quantized_transformer = copy.deepcopy(transformer)

            # Convert only Linear layer weights to FP8 (leave buffers in BF16)
            converted_count = 0
            for name, module in quantized_transformer.named_modules():
                if isinstance(module, torch.nn.Linear):
                    # Convert weight parameter only
                    if hasattr(module, 'weight') and module.weight is not None:
                        module.weight.data = module.weight.data.to(fp8_dtype)
                        converted_count += 1
                    # Keep bias in original dtype (if exists)
                    # Buffers are automatically preserved

            print(f"[Quantization] Successfully converted {converted_count} Linear layers to {dtype_name}")
            print(f"[Quantization] Buffers (x_pad_token, etc.) kept in BF16")
            print(f"[Quantization] Note: Compute will use mixed precision automatically (autocast)")
            print(f"[Quantization] Estimated memory reduction: ~50%")

            return quantized_transformer

        except Exception as e:
            print(f"[Quantization] ERROR during {dtype_name} conversion: {e}")
            import traceback
            traceback.print_exc()
            print(f"[Quantization] Falling back to original model without quantization")
            _add_generation_warning(
                f"Transformer quantization '{quantization}' failed during conversion ({e}); "
                f"falling back to full precision",
                code="quantization_fallback",
            )
            return copy.deepcopy(transformer)

    # Unknown quantization type
    print(f"[Quantization] ERROR: Unknown quantization type: {quantization}")
    print(f"[Quantization] Supported types: fp8_e4m3fn, fp8_e5m2")
    print(f"[Quantization] Falling back to non-quantized transformer")
    _add_generation_warning(
        f"Transformer quantization '{quantization}' is not a supported type "
        f"(fp8_e4m3fn, fp8_e5m2); falling back to full precision",
        code="quantization_fallback",
    )
    return copy.deepcopy(transformer)


def _quantize_text_encoder(text_encoder, quantization: str):
    """Create a quantized copy of Z-Image text encoder

    Uses same weight-only FP8 quantization as Transformer.
    Z-Image text encoder (Qwen 3.4B) is large, so quantization can significantly reduce VRAM.

    Args:
        text_encoder: Original Z-Image text encoder model (Qwen)
        quantization: Quantization type - 'fp8_e4m3fn', 'fp8_e5m2'

    Returns:
        Quantized text encoder model
    """
    print(f"[Quantization] Applying {quantization} to Z-Image Text Encoder (Qwen)...")

    # FP8 quantization: weight-only conversion (same as Transformer)
    if quantization in ['fp8_e4m3fn', 'fp8_e5m2']:
        # Determine FP8 dtype
        if quantization == 'fp8_e4m3fn':
            fp8_dtype = torch.float8_e4m3fn
            dtype_name = "FP8 E4M3FN"
        else:
            fp8_dtype = torch.float8_e5m2
            dtype_name = "FP8 E5M2"

        print(f"[Quantization] Applying {dtype_name} quantization (weight-only)...")

        # Check PyTorch version
        if not hasattr(torch, 'float8_e4m3fn'):
            print(f"[Quantization] ERROR: PyTorch version {torch.__version__} does not support FP8")
            print(f"[Quantization] FP8 requires PyTorch >= 2.1.0")
            print(f"[Quantization] Falling back to original model without quantization")
            _add_generation_warning(
                f"Text encoder quantization '{quantization}' unavailable (PyTorch "
                f"{torch.__version__} lacks FP8 support); falling back to full precision",
                code="quantization_fallback",
            )
            return copy.deepcopy(text_encoder)

        try:
            # Clone the model
            quantized_text_encoder = copy.deepcopy(text_encoder)

            # Convert Linear and Embedding layer weights to FP8 (leave buffers in BF16)
            linear_count = 0
            embedding_count = 0
            for name, module in quantized_text_encoder.named_modules():
                if isinstance(module, torch.nn.Linear):
                    # Convert weight parameter only
                    if hasattr(module, 'weight') and module.weight is not None:
                        module.weight.data = module.weight.data.to(fp8_dtype)
                        linear_count += 1
                    # Keep bias in original dtype (if exists)

                    # Register forward hook to convert FP8 output to BF16
                    # This ensures compatibility with RMSNorm which expects BF16 input
                    def fp8_to_bf16_hook(module, input, output):
                        if output.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                            return output.to(torch.bfloat16)
                        return output

                    module.register_forward_hook(fp8_to_bf16_hook)

                # Convert Embedding layers to FP8 as well (consistent with Linear layers)
                elif isinstance(module, torch.nn.Embedding):
                    if hasattr(module, 'weight') and module.weight is not None:
                        module.weight.data = module.weight.data.to(fp8_dtype)
                        embedding_count += 1

                    # Register forward hook to convert FP8 output to BF16
                    def embedding_fp8_to_bf16_hook(module, input, output):
                        if output.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                            return output.to(torch.bfloat16)
                        return output

                    module.register_forward_hook(embedding_fp8_to_bf16_hook)

            print(f"[Quantization] Successfully converted {linear_count} Linear layers and {embedding_count} Embedding layers to {dtype_name}")
            print(f"[Quantization] Added forward hooks to convert FP8 outputs to BF16")
            print(f"[Quantization] Buffers kept in BF16")

            return quantized_text_encoder

        except Exception as e:
            print(f"[Quantization] ERROR during {dtype_name} conversion: {e}")
            import traceback
            traceback.print_exc()
            print(f"[Quantization] Falling back to original model without quantization")
            _add_generation_warning(
                f"Text encoder quantization '{quantization}' failed during conversion ({e}); "
                f"falling back to full precision",
                code="quantization_fallback",
            )
            return copy.deepcopy(text_encoder)

    # Unknown quantization type
    print(f"[Quantization] ERROR: Unknown quantization type: {quantization}")
    print(f"[Quantization] Supported types: fp8_e4m3fn, fp8_e5m2")
    print(f"[Quantization] Falling back to non-quantized text encoder")
    _add_generation_warning(
        f"Text encoder quantization '{quantization}' is not a supported type "
        f"(fp8_e4m3fn, fp8_e5m2); falling back to full precision",
        code="quantization_fallback",
    )
    return copy.deepcopy(text_encoder)


# ============================================================
# FLUX.2-Specific VRAM Optimization
# ============================================================

def move_flux2_text_encoder_to_gpu(text_encoder, quantization=None):
    """Move FLUX.2 text encoder (Qwen3) to GPU for encoding (with optional quantization)

    Args:
        text_encoder: FLUX.2 text encoder model (Qwen3)
        quantization: Optional quantization type (fp8_e4m3fn, fp8_e5m2, uint2-uint8, etc.)

    Returns:
        text_encoder (potentially quantized copy if quantization is enabled)
    """
    if text_encoder is None:
        return None

    # Fast path: No quantization
    if not quantization or quantization == "none":
        print("[VRAM] Moving FLUX.2 Text Encoder (Qwen3) to GPU for encoding...")
        text_encoder.to('cuda:0', non_blocking=False)
        return text_encoder

    # Quantization path: Create quantized copy and move to GPU
    print(f"[VRAM] Moving FLUX.2 Text Encoder (Qwen3) to GPU with {quantization} quantization...")
    print(f"[Quantization] Creating quantized Text Encoder ({quantization})...")

    # Text Encoder must be on CPU for quantization
    if next(text_encoder.parameters()).device.type != 'cpu':
        print(f"[Quantization] Moving Text Encoder to CPU for quantization...")
        text_encoder.to('cpu')

    # Quantize (creates a copy)
    quantized_text_encoder = _quantize_text_encoder(text_encoder, quantization)

    # Move quantized copy to GPU
    print(f"[Quantization] Moving quantized Text Encoder to GPU...")
    quantized_text_encoder.to('cuda:0', non_blocking=False)

    print(f"[Quantization] FLUX.2 Text Encoder quantization complete ({quantization})")

    return quantized_text_encoder


def move_flux2_transformer_to_gpu(transformer, quantization: Optional[str] = None):
    """Move FLUX.2 transformer to GPU for inference, optionally with quantization

    Args:
        transformer: FLUX.2 transformer model (Flux2Transformer2DModel)
        quantization: Quantization type - None, 'none', 'fp8_e4m3fn', 'fp8_e5m2', etc.

    Returns:
        transformer: Transformer on GPU (may be quantized)
    """
    # Normalize quantization parameter
    if quantization in [None, "", "none"]:
        quantization = None

    # INT8 is NOT handled here. FLUX.2 is one of RUNTIME_INT8_ARCHS, and its
    # conversion runs IN PLACE from Flux2Mixin._flux2_runtime_int8 before the
    # transformer is staged -- necessarily so, because block swap never calls
    # this function at all (it streams weights per block) and would otherwise
    # never quantize. By the time the module reaches here there is nothing left
    # to do but move it, so the request degrades to a plain move rather than to
    # the "unsupported on this architecture" refusal it used to get.
    if _normalize_quantization(quantization) == RUNTIME_INT8_VALUE:
        quantization = None

    if transformer is None:
        return transformer

    # Legacy FP8 path, superseded for an already-quantized module. Reaching
    # _quantize_transformer with weight-only quantized Linears would deep-copy the
    # whole transformer (3.6 GB bf16 for Klein 4B, more for a 9B) and then convert
    # nothing: Int8Linear / Fp8Linear are not nn.Linear subclasses, so its
    # isinstance loop skips every one of them. The copy would be the only effect.
    # Same condition and same warning code as _anima_quantize_fp8's.
    if quantization:
        owned = _already_weight_only_quantized(transformer)
        if owned:
            print(f"[Quantization] FLUX.2 Transformer already holds {owned} weight-only "
                  f"quantized Linear layer(s); ignoring the runtime '{quantization}' request "
                  f"(they already store their weights in 8 bits, with per-row scales).")
            _add_generation_warning(
                f"FLUX.2 Transformer quantization '{quantization}' was ignored: the "
                f"transformer is already weight-only quantized ({owned} layers).",
                code="quantization_superseded",
            )
            transformer.to('cuda:0', non_blocking=False)
            return transformer

    # Fast path: No quantization (most common case)
    if not quantization:
        print("[VRAM] Moving FLUX.2 Transformer to GPU for inference...")
        transformer.to('cuda:0', non_blocking=False)
        return transformer

    # Quantization path
    print(f"[VRAM] Moving FLUX.2 Transformer to GPU with {quantization} quantization...")

    # Ensure transformer is on CPU for quantization
    if next(transformer.parameters()).device.type != 'cpu':
        print(f"[Quantization] Moving Transformer to CPU for quantization...")
        transformer.to('cpu')

    # Apply quantization (similar to U-Net quantization)
    try:
        quantized_transformer = _quantize_transformer(transformer, quantization)
        quantized_transformer.to('cuda:0', non_blocking=False)
        print(f"[Quantization] FLUX.2 Transformer quantization complete ({quantization})")
        return quantized_transformer
    except Exception as e:
        print(f"[VRAM] Warning: Quantization failed: {e}")
        print(f"[VRAM] Falling back to non-quantized transformer")
        _add_generation_warning(
            f"FLUX.2 Transformer quantization '{quantization}' failed ({e}); "
            f"falling back to full precision",
            code="quantization_fallback",
        )
        transformer.to('cuda:0', non_blocking=False)
        return transformer


# ============================================================
# Anima-Specific VRAM Optimization (Cosmos-Predict2 DiT + Qwen3 + Qwen-Image VAE)
# ============================================================

def _anima_patch_linear_fp8(linear, fp8_dtype):
    """Patch one nn.Linear: cast weight to FP8 in-place + replace forward with
    on-the-fly dequantization so the matmul runs in the input's dtype.

    Why: PyTorch's F.linear cannot multiply FP8 weight by BF16 input directly.
    The cheapest fix that still saves persistent VRAM is to keep the weight in
    FP8 and dequant just-in-time for each forward pass — the temporary BF16
    weight tensor is freed immediately after the matmul.
    """
    import torch.nn.functional as F

    linear.weight.data = linear.weight.data.to(fp8_dtype)
    weight_param = linear.weight
    bias_param = linear.bias

    def patched_forward(x):
        w = weight_param.data.to(x.dtype)
        b = bias_param.data.to(x.dtype) if bias_param is not None else None
        return F.linear(x, w, b)

    linear.forward = patched_forward


def _already_weight_only_quantized(model) -> int:
    """Count ``Int8Linear`` / ``Fp8Linear`` modules under ``model``.

    ARCH-AGNOSTIC despite its Anima-shaped history: it is now also what
    ``move_flux2_transformer_to_gpu`` asks before deep-copying for the legacy FP8
    path. Non-zero means the module owns weight-only quantized Linears -- from an
    offline-quantized checkpoint (per-output-row scales, swapped in by
    ``anima_loader._swap_quantized_linears`` /
    ``model_loader._swap_flux2_quantized_linears``) or from an in-place runtime
    conversion -- and already owns the
    superior path: a real W8A8 GEMM where the shape gates allow, and a dequant
    matmul otherwise. Detection is by module type, not by weight dtype, so it
    cannot be confused by a checkpoint that merely happens to store float8.
    """
    try:
        from core.models.ideogram4.vendor.int8_linear import Int8Linear
        from core.models.ideogram4.vendor.fp8_linear import Fp8Linear
    except Exception:
        return 0
    return sum(1 for m in model.modules() if isinstance(m, (Int8Linear, Fp8Linear)))


# ---------------------------------------------------------------------------
# Runtime INT8: quantize an ordinary bf16 checkpoint IN PLACE, once per load
# ---------------------------------------------------------------------------
#
# ``unet_quantization="int8"`` on an UNQUANTIZED model of any arch in
# RUNTIME_INT8_ARCHS (anima / krea2 / flux2) converts the
# loaded transformer to the same MIXED int8/e4m3 layout the offline tool
# (``subapps/fp8_quantize/quantize_transformer_fp8.py --format int8``) writes,
# using the SAME selection rule -- both import it from
# ``core.models.common.int8_runtime_quantize``.
#
# IN PLACE, deliberately. The SD1.5/SDXL path above keeps a second, unquantized
# copy on CPU (``pipeline._original_unet``) so the request can be undone; that
# costs 1x the model and is not viable against a 26 GB bf16 Krea 2 transformer.
# Here each layer's source weight is dropped as its quantized replacement is
# built, so no second MODULE is ever constructed -- but the process still holds
# the source checkpoint's mapping (the skipped Linears and every non-Linear
# parameter reference it), which measures as ~1.6x the source in host RSS for
# the session: 6.16 GB for Anima's 3.90 GB DiT, ~36 GB extrapolated for Krea 2.
# See quantize_linears_in_place's docstring and docs/guides/MODEL_FACTS.md.
# The conversion is ONE-WAY until the model is reloaded.

RUNTIME_INT8_VALUE = "int8"

# The recovery action, in ONE place, because it is quoted in several warnings,
# in openapi.yaml and in all four generation panels -- and because it was WRONG
# until POST /models/load grew `force`: `_load_model_locked` early-returns when
# the requested model id equals the loaded one, so re-selecting the same
# checkpoint used to be a silent no-op that undid nothing. "Load Selected Model"
# on the currently-loaded model now sends force=true and really does reload.
_RUNTIME_INT8_RECOVERY = (
    "Load the model again from the model selector (loading the model that is already "
    "loaded now forces a real reload) to get an unquantized transformer."
)
_RUNTIME_INT8_RECOVERY_LOWER = (
    "load the model again from the model selector (loading the model that is already "
    "loaded now forces a real reload) to get an unquantized transformer."
)
_RUNTIME_INT8_RECOVERY_LOG = (
    "Load the model again (POST /models/load with force=true, or the model selector's "
    "Load button on the currently loaded model) for an unquantized transformer."
)


def _runtime_int8_arch_phrase() -> str:
    """"the Anima and Krea 2 transformers", rendered from ``RUNTIME_INT8_ARCHS``.

    Imported lazily and defensively for the same reason the converter itself is
    (``apply_runtime_int8_quantization`` below): a build where the vendor
    quantized-Linear modules cannot be imported must still be able to say that
    int8 is unavailable, rather than raising while composing the message.
    """
    try:
        from core.models.common.int8_runtime_quantize import (
            RUNTIME_INT8_ARCHS, arch_names,
        )
        return f"the {arch_names(RUNTIME_INT8_ARCHS)} transformers"
    except Exception:
        return "a subset of the DiT architectures"


def _refuse_runtime_int8_elsewhere(quantization, label: str) -> bool:
    """True (with a warning emitted) when ``quantization`` asks for INT8 on a path
    that cannot do it.

    ``unet_quantization: "int8"`` is advertised for every architecture by the
    request schema, but the in-place converter is wired only for the archs in
    ``RUNTIME_INT8_ARCHS``. Every other quantization entry point would otherwise
    treat "int8" as an unknown type and reach that branch only AFTER paying for
    a full ``copy.deepcopy`` of the U-Net (SD1.5/SDXL). Refuse up front instead,
    and say which architectures do support it -- the arch names are DERIVED from
    ``RUNTIME_INT8_ARCHS`` rather than written out, so this message cannot go on
    naming two archs after a third is wired up.
    """
    if _normalize_quantization(quantization) != RUNTIME_INT8_VALUE:
        return False
    supported = _runtime_int8_arch_phrase()
    print(f"[Quantization] '{RUNTIME_INT8_VALUE}' is not supported for {label}; the in-place "
          f"INT8 conversion is wired for {supported} only. "
          f"Running at full precision.")
    _add_generation_warning(
        f"{label} quantization 'int8' is not supported on this architecture (the in-place "
        f"INT8 conversion is implemented for {supported} only); "
        f"running at full precision.",
        code="quantization_fallback",
    )
    return True


def _normalize_quantization(quantization) -> Optional[str]:
    """``None`` for every spelling of "no quantization", else the string."""
    if quantization in (None, "", "none"):
        return None
    return str(quantization)


def runtime_int8_requested(quantization) -> bool:
    """True when the per-generation request asks for the runtime INT8 path."""
    return _normalize_quantization(quantization) == RUNTIME_INT8_VALUE


def _runtime_int8_progress_adapter(progress_callback):
    """Wrap a generation progress callback for the converter's (done, total, name).

    Uses the SAME decoupled-phase channel the PiD decode uses
    (``progress_callback(step, total, None, phase_label=...)``) rather than
    inventing a mechanism: the conversion happens inside the first generation, so
    it belongs on that generation's progress channel. Throttled to ~2 Hz because
    a conversion is 230-260 layers and every callback is a WebSocket send.
    """
    if progress_callback is None:
        return None
    import time as _time
    last = [0.0]

    def _cb(done: int, total: int, name: str) -> None:
        now = _time.monotonic()
        if done != total and (now - last[0]) < 0.5:
            return
        last[0] = now
        try:
            progress_callback(done, total, None,
                              phase_label="Quantizing transformer weights (INT8, one-time)")
        except Exception:
            pass

    return _cb


def _multi_progress_adapter(progress_callback, index: int, count: int,
                            totals: Optional[Dict[int, int]] = None):
    """The conversion progress callback for component ``index`` of ``count``.

    One bar across the whole request rather than one per component: an
    architecture with two transformers (Ideogram 4) would otherwise show the
    progress restart half way through a single one-time conversion.

    A component's SELECTED-layer count is only known once its first callback
    arrives (the selection runs inside ``quantize_linears_in_place``), so
    ``totals`` -- a dict SHARED across the whole set by the caller -- accumulates
    the ones that have been seen, and the current component's total stands in for
    the ones that have not. The offset is then the sum of the REAL totals of the
    components already finished rather than ``index * total``.

    For identical geometry -- Ideogram 4's two transformers, the only
    multi-component case that exists -- this is exact and identical to what the
    ``index * total`` form produced. For components of DIFFERENT sizes the step
    count stays monotonic and the bar still ends at exactly 100%; only the
    denominator refines (once per component) as each real total becomes known.
    The ``index * total`` form instead moved the step count BACKWARDS at every
    component boundary, e.g. 40/80 -> 11/20 on a 40+10 pair. Pre-computing the
    true grand total would need the converter's selection to run twice (once to
    count, once to convert) or its gate arguments to be duplicated here, where
    they could drift from the ones that actually decide; a denominator that
    refines is the cheaper honest answer.
    """
    base = _runtime_int8_progress_adapter(progress_callback)
    if base is None or count <= 1:
        return base
    if totals is None:
        totals = {}

    def _cb(done: int, total: int, name: str) -> None:
        totals[index] = total
        estimated = [totals.get(i, total) for i in range(count)]
        offset = sum(estimated[:index])
        grand = sum(estimated)
        base(min(done + offset, grand), grand, name)

    return _cb


def _merge_runtime_int8_documents(documents):
    """One audit document describing a multi-component conversion.

    Same SHAPE as a single-component one (``settings`` / ``format_counts`` /
    ``geomean_advantage`` / ``layers``), because it is stored on the manager and
    written next to a ``POST /models/export-quantized`` artifact, and a reader
    must not need to know how many transformers the architecture has. The layer
    rows are already namespaced by component (see ``_qualify``), and the extra
    ``components`` key names them in write order.
    """
    from core.models.common.int8_runtime_quantize import audit_document

    rows = [r for _c, d in documents for r in (d.get("layers", []) or [])]
    settings = dict((documents[0][1].get("settings", {}) or {}))
    settings["components"] = [c for c, _d in documents]
    settings["skipped"] = [s for _c, d in documents
                           for s in ((d.get("settings", {}) or {}).get("skipped", []) or [])]
    merged = audit_document(rows, settings)
    merged["elapsed_s"] = sum(float(d.get("elapsed_s", 0.0) or 0.0) for _c, d in documents)
    converted: Dict[str, int] = {}
    for _c, d in documents:
        for fmt, n in (d.get("converted", {}) or {}).items():
            converted[fmt] = converted.get(fmt, 0) + int(n)
    merged["converted"] = converted
    merged["oom_fallback_layers"] = [
        f"{c}.{name}" for c, d in documents
        for name in (d.get("oom_fallback_layers", []) or [])]
    return merged


def apply_runtime_int8_quantization(manager, model, arch: str, quantization,
                                    label: str = "Transformer",
                                    progress_callback=None, precheck=None):
    """Convert ``model`` to the mixed int8/e4m3 layout in place, once.

    The single-component spelling of ``apply_runtime_int8_quantization_multi``
    below, which holds the implementation and the full contract. Returns
    ``(model, converted)``; ``model`` is the SAME object either way (the
    conversion replaces child modules in place).
    """
    models, converted = apply_runtime_int8_quantization_multi(
        manager, [("transformer", label, model)], arch, quantization,
        progress_callback=progress_callback, precheck=precheck)
    return (models[0] if models else model), converted


def apply_runtime_int8_quantization_multi(manager, components, arch: str, quantization,
                                          progress_callback=None, precheck=None):
    """Convert EVERY module in ``components`` to the mixed int8/e4m3 layout, as one unit.

    ``components`` is a sequence of ``(component name, label, module)``; a
    ``None`` module is dropped. Returns ``(list of modules, converted)``, where
    ``converted`` is True only when the whole set was converted.

    WHY A SET AND NOT A LOOP OVER THE SINGLE-MODULE FUNCTION. The bookkeeping is
    per MANAGER, not per module: ``_runtime_int8_converted`` latches the moment a
    conversion completes, so calling the single-module function twice would
    convert the first transformer, latch, and return the second one untouched --
    silently, and with no warning, because "already converted" is a legitimate
    state. On Ideogram 4 that is not a half-quantized model but something worse:
    a quantized conditional branch and a bf16 unconditional one, i.e. the two
    halves of an asymmetric-CFG denoise step computed at different precisions.
    Everything that decides is therefore evaluated over the whole set, and the
    latch is set once, at the end.

    The modules are returned UNCHANGED (and ``converted`` is False) for every
    case that is not a fresh int8 conversion:

    * the request is not ``"int8"`` -- the caller's own fp8/none handling stands;
    * ``arch`` is not one of ``RUNTIME_INT8_ARCHS``;
    * ANY module is already weight-only quantized, or already holds float8
      weights. Refused for the whole set rather than per module, for the reason
      above: converting the rest would leave the components at different
      precisions;
    * the model was ALREADY converted at runtime. A later ``null``/fp8 request
      then proceeds with the quantized model and emits
      ``runtime_quantization_persistent`` -- the conversion cannot be undone
      without reloading the checkpoint, and pretending otherwise would be worse
      than saying so;
    * the CHECKPOINT is already weight-only quantized (offline artifact). That is
      the same condition ``_anima_quantize_fp8`` refuses on, and it emits the
      same ``quantization_superseded`` code;
    * the weights are ALREADY float8, because an fp8 generation ran earlier in
      this session and ``_anima_patch_linear_fp8`` cast them in place (it leaves
      the module an ``nn.Linear``, so the type-based check above cannot see it).
      Quantizing e4m3-rounded weights to int8 measured 11.2x the weight error of
      a direct int8 conversion -- worse than either format alone -- so this
      refuses instead, with ``quantization_superseded``;
    * a PREVIOUS attempt failed part-way (``manager._runtime_int8_partial``) and
      this request is not another int8 request. A partial model is reported for
      what it is; another int8 request RESUMES it (see below).

    PARTIAL CONVERSIONS. A failure part-way (a CUDA OOM at layer 120 of 263 is
    the realistic one) leaves the module half converted, which cannot be undone.
    It is therefore NOT latched as converted: ``manager._runtime_int8_partial``
    is set instead, the user is told how many layers were converted, and the next
    int8 request resumes the remainder -- selection walks ``nn.Linear`` and a
    converted layer is no longer one. While partial, the "already quantized
    CHECKPOINT" branch is suppressed: those modules are ours, not the
    checkpoint's, and claiming otherwise both asserts a false provenance and
    would strand the unconverted layers forever.

    A completed conversion sets ``manager._runtime_int8_converted = True``, which
    ``keep_hot.compute_model_key`` reads so the resident-component key does not
    flip between generations, and which ``pipeline._load_model_locked`` clears.

    ``precheck`` -- CALLER-OWNED INVARIANTS THAT ONLY APPLY TO A REAL CONVERSION.
    An optional zero-argument callable invoked exactly ONCE, after every refusal
    above has been evaluated and immediately before the first layer of the first
    component is touched; it may raise to abort. It exists because "is this
    request going to convert anything?" is decided HERE, by a dozen conditions
    (already converted, already weight-only quantized, float8 weights, LoRA
    wrappers, arch not wired, ...), and a caller that guards its own invariant
    before calling cannot know the answer. LTX-2.3 is the case that forced it:
    its block offloader is PERSISTENT state on the pipeline wrapper (it survives
    a generation, unlike FLUX.2's ``_flux2_active_block_offloader``, which is
    cleared in a ``finally``), so a guard raised before this function was
    consulted fired on the *second* block-swap generation of a session even when
    no quantization was requested at all. Passing the guard in means it can only
    fire for the violation it describes.

    A model that was ALREADY quantized in its checkpoint sets a SEPARATE latch,
    ``manager._runtime_int8_from_checkpoint``. Keep-hot keys the two identically
    (both mean "the resident transformer is quantized"), and only the runtime one
    carries the one-way persistence message: the checkpoint's quantization is not
    something this session did and not something a reload would undo, so saying
    so would be a false statement, and on an architecture whose published
    checkpoints are all quantized it would be the message the user sees every
    generation.
    """
    resolved = [(str(name), str(lbl), mod) for name, lbl, mod in components if mod is not None]
    models = [mod for _n, _l, mod in resolved]
    if not resolved:
        return models, False
    # One label for the messages, whatever the component count. "X and Y" reads
    # correctly for two and degrades to the single label for one, so every
    # existing single-module warning is byte-identical.
    label = resolved[0][1] if len(resolved) == 1 else \
        f"{', '.join(l for _n, l, _m in resolved[:-1])} and {resolved[-1][1]}"

    requested = _normalize_quantization(quantization)
    already_converted = bool(getattr(manager, "_runtime_int8_converted", False))
    partial = bool(getattr(manager, "_runtime_int8_partial", False))

    if already_converted:
        if requested != RUNTIME_INT8_VALUE:
            print(f"[RuntimeInt8] {label} was quantized to INT8 earlier in this session; "
                  f"the request '{quantization}' cannot undo it (the source weights were "
                  f"dropped). {_RUNTIME_INT8_RECOVERY_LOG}")
            _add_generation_warning(
                f"{label} stays INT8: it was quantized in place earlier in this session and "
                f"the conversion is one-way. {_RUNTIME_INT8_RECOVERY}",
                code="runtime_quantization_persistent",
            )
        return models, False

    if requested != RUNTIME_INT8_VALUE:
        if partial:
            done = int(getattr(manager, "_runtime_int8_partial_done", 0) or 0)
            print(f"[RuntimeInt8] {label} is PARTIALLY INT8 ({done} layer(s) converted before an "
                  f"earlier failure); the request '{quantization}' cannot undo it. "
                  f"{_RUNTIME_INT8_RECOVERY_LOG}")
            _add_generation_warning(
                f"{label} is partially INT8: an earlier conversion failed after {done} layer(s) "
                f"and the conversion is one-way. Request INT8 again to convert the remaining "
                f"layers, or {_RUNTIME_INT8_RECOVERY_LOWER}",
                code="runtime_quantization_persistent",
            )
        return models, False

    try:
        from core.models.common.int8_runtime_quantize import (
            LoraWrappedError, RUNTIME_INT8_ARCHS, already_weight_only_quantized,
            float8_weight_linear_count, lora_wrapped_count, quantize_linears_in_place,
        )
    except Exception as e:
        print(f"[RuntimeInt8] unavailable ({e}); leaving {label} unquantized")
        _add_generation_warning(
            f"{label} INT8 quantization unavailable ({e}); running at full precision",
            code="quantization_fallback",
        )
        return models, False

    if arch not in RUNTIME_INT8_ARCHS:
        # The capability table already warns for an arch that ignores
        # unet_quantization; nothing to add here.
        return models, False

    if not partial:
        # Suppressed while partial: the quantized modules present would then be
        # OUR OWN half-finished work, not a checkpoint property.
        owned = sum(already_weight_only_quantized(mod) for _n, _l, mod in resolved)
        if owned:
            print(f"[RuntimeInt8] {label} already holds {owned} weight-only quantized Linear(s) "
                  f"from the checkpoint; the runtime '{quantization}' request is a no-op.")
            _add_generation_warning(
                f"{label} quantization '{quantization}' was ignored: the checkpoint is already "
                f"weight-only quantized ({owned} layers).",
                code="quantization_superseded",
            )
            # Keyed for keep-hot exactly like a runtime conversion -- the model
            # IS quantized, so the resident-set key must not flip when the next
            # request omits the parameter -- but under its OWN latch. Setting
            # ``_runtime_int8_converted`` here would make every subsequent
            # non-int8 request emit ``runtime_quantization_persistent`` ("it was
            # quantized in place earlier in this session and the conversion is
            # one-way"), which is false: nothing was converted, the checkpoint
            # arrived this way, and reloading it would produce the same model.
            # On an architecture whose published checkpoints are all quantized
            # (Ideogram 4: FP8/nf4) that false warning would be the NORMAL path
            # after a single int8 request, not an edge case.
            manager._runtime_int8_from_checkpoint = True
            return models, False

    fp8_weights = sum(float8_weight_linear_count(mod) for _n, _l, mod in resolved)
    if fp8_weights:
        print(f"[RuntimeInt8] {label} holds {fp8_weights} Linear layer(s) whose weights are "
              f"already float8 from an FP8 generation earlier in this session; refusing to "
              f"quantize rounded weights again (measured 11.2x the weight error of a direct "
              f"INT8 conversion). {_RUNTIME_INT8_RECOVERY_LOG}")
        _add_generation_warning(
            f"{label} quantization '{quantization}' was ignored: {fp8_weights} Linear layer(s) "
            f"already hold FP8 weights (an FP8 generation earlier in this session casts them "
            f"in place), and quantizing already-rounded weights to INT8 is worse than either "
            f"format alone. The "
            f"transformer stays FP8. To convert from the original weights instead, "
            f"{_RUNTIME_INT8_RECOVERY_LOWER[:-1]} and request INT8 without an FP8 "
            f"generation in between.",
            code="quantization_superseded",
        )
        return models, False

    # LoRA pre-flight over the WHOLE set, before the first layer of the first
    # component is touched. ``quantize_linears_in_place`` makes the same refusal
    # per module, but discovering it on the SECOND component would already have
    # left the first one converted -- the mixed-precision state this function
    # exists to prevent.
    wrapped = [(lbl, lora_wrapped_count(mod)) for _n, lbl, mod in resolved]
    if any(n for _l, n in wrapped):
        detail = ", ".join(f"{l}: {n}" for l, n in wrapped if n)
        print(f"[RuntimeInt8] {label} conversion refused: LoRA wrappers present ({detail})")
        _add_generation_warning(
            f"{label} INT8 quantization was not applied because LoRAs are loaded: the LoRA "
            f"wrappers hide the Linear layers, so the conversion would select a different "
            f"set than it should. The model is unchanged and runs at full precision. "
            f"Remove the LoRAs (or reload the model without them) to convert.",
            code="quantization_fallback",
        )
        return models, False

    if partial:
        done = int(getattr(manager, "_runtime_int8_partial_done", 0) or 0)
        print(f"[RuntimeInt8] resuming a partial {label} conversion "
              f"({done} layer(s) already converted)")

    # Every refusal has been evaluated; from here a conversion really happens.
    # This is the only point at which a caller-owned "must not convert now"
    # invariant can be checked without firing on requests that convert nothing
    # (see the ``precheck`` paragraph in the docstring). Deliberately NOT wrapped
    # in try/except: it raises to abort, and swallowing it would restore exactly
    # the silent-corruption case it exists to prevent.
    if precheck is not None:
        precheck()

    work_device = torch.device("cuda:0") if torch.cuda.is_available() else None
    multi = len(resolved) > 1

    def _qualify(rows, component: str):
        """Namespace a component's audit rows when there is more than one.

        Two transformers of IDENTICAL geometry produce identical module paths, so
        an un-namespaced merge would silently collapse 558 rows into 279 -- and
        the de-duplication in the partial-resume path below would then treat the
        second transformer's layers as already done. Single-component archs are
        left exactly as they were, so their audit documents stay diffable against
        the committed offline artifacts.
        """
        if not multi:
            return rows
        for row in rows:
            row["name"] = f"{component}.{row.get('name')}"
        return rows

    documents = []
    component, comp_label = resolved[0][0], resolved[0][1]
    # Shared across the components so the one bar can use each finished
    # component's REAL Linear count as the next one's offset.
    progress_totals: Dict[int, int] = {}
    try:
        for index, (component, comp_label, mod) in enumerate(resolved):
            document = quantize_linears_in_place(
                mod,
                arch=arch,
                compute_dtype=torch.bfloat16,
                work_device=work_device,
                progress_cb=_multi_progress_adapter(progress_callback, index,
                                                    len(resolved), progress_totals),
                label=comp_label,
            )
            _qualify(document.get("layers", []) or [], component)
            _qualify((document.get("settings", {}) or {}).get("skipped", []) or [], component)
            documents.append((component, document))
    except LoraWrappedError as e:
        # The pre-flight above already refused this case, so reaching it means the
        # module changed under us. Nothing was touched IN THIS component -- but an
        # earlier one may already be converted, which is a partial model, not an
        # unchanged one.
        print(f"[RuntimeInt8] {comp_label} conversion refused: {e}")
        if documents:
            manager._runtime_int8_partial = True
            # Rows from an EARLIER partial pass are carried forward, exactly as
            # the generic handler below does: this request may be the resume of
            # one, in which case the layers that pass converted are already
            # quantized and dropping their rows would under-report the model (and
            # make the next resume's de-duplication miss them).
            rows = list(getattr(manager, "_runtime_int8_partial_rows", []) or [])
            rows.extend(r for _c, d in documents for r in (d.get("layers", []) or []))
            seen_names = set()
            unique_rows = []
            for row in rows:
                name = row.get("name")
                if name in seen_names:
                    continue
                seen_names.add(name)
                unique_rows.append(row)
            manager._runtime_int8_partial_rows = unique_rows
            manager._runtime_int8_partial_done = len(
                [r for r in unique_rows if r.get("chosen") in ("int8", "e4m3")])
        _add_generation_warning(
            f"{label} INT8 quantization was not applied to {comp_label} because LoRAs are "
            f"loaded: the LoRA wrappers hide the Linear layers, so the conversion would "
            f"select a different set than it should. "
            + (f"{len(documents)} of {len(resolved)} component(s) were already converted and "
               f"that cannot be undone; " if documents else "The model is unchanged and ")
            + f"the model runs at "
            f"{'mixed precision' if documents else 'full precision'}. "
            f"Remove the LoRAs (or reload the model without them) to convert.",
            code="quantization_fallback" if not documents else "quantization_partial",
        )
        return models, False
    except Exception as e:
        import traceback; traceback.print_exc()
        doc = getattr(e, "_int8_partial_document", None) or {}
        # ``component`` is the one that failed; its partial document's rows need
        # the same namespacing the completed ones got.
        _qualify(doc.get("layers", []) or [], component)
        rows = list(getattr(manager, "_runtime_int8_partial_rows", []) or [])
        for _c, completed in documents:
            rows.extend(completed.get("layers", []) or [])
        rows.extend(doc.get("layers", []) or [])
        unique_rows = []
        seen_names = set()
        for row in rows:
            name = row.get("name")
            if name in seen_names:
                continue
            seen_names.add(name)
            unique_rows.append(row)
        rows = unique_rows
        done = len([r for r in rows if r.get("chosen") in ("int8", "e4m3")])
        remaining = int(doc.get("remaining", 0) or 0)
        # Components not started at all are also outstanding; without them the
        # "left" count would understate a two-transformer failure by a whole
        # transformer.
        untouched = len(resolved) - len(documents) - 1
        # NOT latched as converted: the model is neither bf16 nor fully INT8.
        manager._runtime_int8_partial = True
        manager._runtime_int8_partial_rows = rows
        manager._runtime_int8_partial_done = done
        scope = f" (in {comp_label})" if multi else ""
        untouched_note = (
            f" and {untouched} further component(s) were not started" if untouched > 0 else "")
        print(f"[RuntimeInt8] {label} conversion failed after {done} layer(s)"
              f"{scope} ({remaining} left{untouched_note}): {e}")
        _add_generation_warning(
            f"{label} INT8 quantization failed after converting {done} layer(s){scope} "
            f"({remaining} left{untouched_note}): {e}. The transformer is now partially INT8 "
            f"and that cannot be undone. "
            f"Generate again with INT8 to convert the remaining layers, or "
            f"{_RUNTIME_INT8_RECOVERY_LOWER}",
            code="quantization_partial",
        )
        return models, False

    document = _merge_runtime_int8_documents(documents) if multi else documents[0][1]
    counts = document.get("converted", {})
    converted_n = counts.get("int8", 0) + counts.get("e4m3", 0)
    prev_rows = list(getattr(manager, "_runtime_int8_partial_rows", []) or [])
    if prev_rows:
        # Merge an earlier partial pass in, so the stored audit describes the
        # whole module rather than only the resumed remainder.
        seen = {r.get("name") for r in document.get("layers", [])}
        document["layers"] = [r for r in prev_rows if r.get("name") not in seen] + \
            list(document.get("layers", []))
        merged: Dict[str, int] = {}
        for r in document["layers"]:
            merged[r["chosen"]] = merged.get(r["chosen"], 0) + 1
        document["format_counts"] = merged
        document["converted"] = merged
        document["resumed_after_partial"] = True
        counts = merged
        converted_n = merged.get("int8", 0) + merged.get("e4m3", 0)

    if converted_n == 0:
        # Nothing qualified (every candidate filtered out by the shape gates, or
        # every one rejected in favour of leaving it alone). The model is exactly
        # what it was, so the one-way flag must NOT be latched.
        print(f"[RuntimeInt8] {label}: no layer was converted; leaving the model unchanged")
        _add_generation_warning(
            f"{label} INT8 quantization converted no layers (none of this model's Linear "
            f"layers qualified); the transformer runs at full precision.",
            code="quantization_fallback",
        )
        return models, False

    manager._runtime_int8_converted = True
    manager._runtime_int8_partial = False
    manager._runtime_int8_partial_rows = []
    manager._runtime_int8_partial_done = 0
    manager._runtime_int8_audit = document
    scope = f" across {len(resolved)} components" if multi else ""
    print(f"[RuntimeInt8] {label} converted in place{scope}: "
          f"{counts.get('int8', 0)} int8 + {counts.get('e4m3', 0)} e4m3 Linear(s), "
          f"{document.get('elapsed_s', 0.0):.1f}s. One-way until the model is reloaded.")
    return models, True


def _anima_quantize_fp8(model, quantization: str, label: str):
    """Apply FP8 weight-only quantization with on-the-fly dequant to all
    nn.Linear modules in `model`. Embeddings are left untouched (they're
    looked up, not matmul'd, so we can't safely dequant on lookup).

    Returns the model with Linear forward methods replaced.

    SUPERSEDED BY AN OFFLINE-QUANTIZED CHECKPOINT. This is the legacy runtime
    path: it deep-copies the module and replaces every ``nn.Linear.forward``
    with a full-weight dequantization per call. On an Anima checkpoint that
    already carries ``Int8Linear`` / ``Fp8Linear`` layers that would be strictly
    worse -- it would deep-copy 2.5 GB, then leave those modules alone anyway
    (they are not ``nn.Linear`` and the loop would skip them), so the only
    effect would be the copy. It returns the model untouched in that case, and
    says so.

    The "already quantized" condition is now reachable by a SECOND route: a
    model converted in place by ``apply_runtime_int8_quantization`` above. The
    ANIMA staging path short-circuits before reaching here in that case
    (``_anima_runtime_int8`` passes ``None`` once a runtime conversion has
    happened or is being requested, and the user gets a
    ``runtime_quantization_persistent`` warning explaining that the conversion is
    one-way). LENS does NOT: ``move_lens_transformer_to_gpu`` calls this function
    directly and has no runtime-int8 hook at all, so a Lens transformer reaches
    here with whatever the request said. That is why the int8 refusal below is
    part of this function rather than of the Anima caller, and why the
    already-quantized refusal is kept type-based rather than
    provenance-based: any caller that reaches here with quantized Linears gets
    the same answer.
    """
    # int8 is the in-place converter's value (RUNTIME_INT8_ARCHS); it never means anything
    # here (Lens and the text encoders have no int8 path).
    if _refuse_runtime_int8_elsewhere(quantization, f"Anima/Lens {label}"):
        return model

    already = _already_weight_only_quantized(model)
    if already:
        print(f"[Quantization] Anima {label} is already weight-only quantized "
              f"({already} Int8Linear/Fp8Linear layer(s) from the checkpoint); "
              f"ignoring the runtime '{quantization}' request. The per-row-scaled "
              f"layers already hold their weights in 8 bits.")
        _add_generation_warning(
            f"Anima {label} quantization '{quantization}' was ignored: the checkpoint is "
            f"already weight-only quantized ({already} layers).",
            code="quantization_superseded",
        )
        return model

    if quantization == 'fp8_e4m3fn':
        fp8_dtype = torch.float8_e4m3fn
        dtype_name = "FP8 E4M3FN"
    elif quantization == 'fp8_e5m2':
        fp8_dtype = torch.float8_e5m2
        dtype_name = "FP8 E5M2"
    else:
        print(f"[Quantization] Unsupported {quantization} for Anima; only fp8_e4m3fn/fp8_e5m2 supported")
        _add_generation_warning(
            f"Anima {label} quantization '{quantization}' is not a supported type "
            f"(fp8_e4m3fn, fp8_e5m2); falling back to full precision",
            code="quantization_fallback",
        )
        return model

    if not hasattr(torch, 'float8_e4m3fn'):
        print(f"[Quantization] PyTorch {torch.__version__} does not support FP8. Skipping.")
        _add_generation_warning(
            f"Anima {label} quantization '{quantization}' unavailable (PyTorch "
            f"{torch.__version__} lacks FP8 support); falling back to full precision",
            code="quantization_fallback",
        )
        return model

    print(f"[Quantization] Applying {dtype_name} to Anima {label} (on-the-fly dequant)...")
    quantized = copy.deepcopy(model)
    converted = 0
    for name, module in quantized.named_modules():
        if isinstance(module, torch.nn.Linear):
            _anima_patch_linear_fp8(module, fp8_dtype)
            converted += 1
    print(f"[Quantization] Patched {converted} Linear layers in Anima {label} ({dtype_name})")
    print(f"[Quantization] VRAM reduction ~50%, compute remains BF16 via on-the-fly cast")
    return quantized


def move_anima_text_encoder_to_gpu(text_encoder, quantization: Optional[str] = None):
    """Move Anima text encoder (Qwen3-0.6B) to GPU for encoding (optional quantization).

    Args:
        text_encoder: Qwen3 model
        quantization: None | 'none' | 'fp8_e4m3fn' | 'fp8_e5m2'

    Returns:
        text_encoder (potentially quantized copy if quantization is enabled)
    """
    if text_encoder is None:
        return None

    if not quantization or quantization == "none":
        print("[VRAM] Moving Anima Text Encoder (Qwen3) to GPU for encoding...")
        text_encoder.to('cuda:0', non_blocking=False)
        return text_encoder

    print(f"[VRAM] Moving Anima Text Encoder (Qwen3) to GPU with {quantization} quantization...")
    if next(text_encoder.parameters()).device.type != 'cpu':
        text_encoder.to('cpu')

    quantized = _anima_quantize_fp8(text_encoder, quantization, "Text Encoder")
    quantized.to('cuda:0', non_blocking=False)
    return quantized


def move_anima_text_encoder_to_cpu(text_encoder):
    """Move Anima text encoder back to CPU."""
    if text_encoder is None:
        return
    text_encoder.to('cpu', non_blocking=False)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def move_anima_transformer_to_gpu(transformer, quantization: Optional[str] = None):
    """Move Anima DiT to GPU for inference (optional FP8 quantization).

    Args:
        transformer: Anima DiT (Cosmos-Predict2-based)
        quantization: None | 'none' | 'fp8_e4m3fn' | 'fp8_e5m2'

    Returns:
        transformer (potentially quantized copy)
    """
    if quantization in [None, "", "none"]:
        quantization = None

    if transformer is None:
        return transformer

    if not quantization:
        print("[VRAM] Moving Anima Transformer (DiT) to GPU for inference...")
        transformer.to('cuda:0', non_blocking=False)
        return transformer

    print(f"[VRAM] Moving Anima Transformer (DiT) to GPU with {quantization} quantization...")
    if next(transformer.parameters()).device.type != 'cpu':
        transformer.to('cpu')

    try:
        quantized = _anima_quantize_fp8(transformer, quantization, "Transformer")
        quantized.to('cuda:0', non_blocking=False)
        return quantized
    except Exception as e:
        print(f"[VRAM] Warning: Anima quantization failed: {e}")
        import traceback; traceback.print_exc()
        print(f"[VRAM] Falling back to non-quantized transformer")
        _add_generation_warning(
            f"Anima Transformer quantization '{quantization}' failed ({e}); "
            f"falling back to full precision",
            code="quantization_fallback",
        )
        transformer.to('cuda:0', non_blocking=False)
        return transformer


def move_anima_transformer_to_cpu(transformer):
    if transformer is None:
        return
    transformer.to('cpu', non_blocking=False)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def move_anima_vae_to_gpu(vae):
    if vae is None:
        return
    vae.to('cuda:0')


def move_anima_vae_to_cpu(vae):
    if vae is None:
        return
    vae.to('cpu')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Vision Encoder VRAM helpers ────────────────────────────────────────────────

def move_vision_encoder_to_gpu(vision_encoder, device: str = "cuda:0"):
    """Move SigLIP2VisionEncoderWrapper to GPU for encoding."""
    if vision_encoder is None:
        return
    try:
        vision_encoder.to(device)
        print(f"[VRAM] Vision Encoder moved to {device}")
    except Exception as e:
        print(f"[VRAM] Warning: Could not move Vision Encoder to {device}: {e}")


def move_vision_encoder_to_cpu(vision_encoder):
    """Move SigLIP2VisionEncoderWrapper back to CPU to free VRAM."""
    if vision_encoder is None:
        return
    try:
        vision_encoder.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[VRAM] Vision Encoder moved to CPU")
    except Exception as e:
        print(f"[VRAM] Warning: Could not move Vision Encoder to CPU: {e}")


# ============================================================
# Lens-Specific VRAM Optimization
# ============================================================

def move_lens_text_encoder_to_gpu(text_encoder, quantization: Optional[str] = None):
    """Move Lens GPT-OSS text encoder to GPU (with optional FP8 quantization)."""
    if text_encoder is None:
        return text_encoder
    if quantization in (None, "", "none"):
        quantization = None
    if quantization:
        if next(text_encoder.parameters()).device.type != "cpu":
            text_encoder.to("cpu")
        try:
            text_encoder = _anima_quantize_fp8(text_encoder, quantization, "Lens TextEncoder")
        except Exception as e:
            print(f"[VRAM] Lens text encoder quantization failed: {e}; using bf16")
    text_encoder.to("cuda:0")
    return text_encoder


def move_lens_text_encoder_to_cpu(text_encoder):
    if text_encoder is None:
        return
    text_encoder.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def move_lens_transformer_to_gpu(transformer, quantization: Optional[str] = None):
    """Move Lens MMDiT transformer to GPU (with optional FP8 quantization)."""
    if transformer is None:
        return transformer
    if quantization in (None, "", "none"):
        quantization = None
    if quantization:
        if next(transformer.parameters()).device.type != "cpu":
            transformer.to("cpu")
        try:
            transformer = _anima_quantize_fp8(transformer, quantization, "Lens Transformer")
        except Exception as e:
            print(f"[VRAM] Lens transformer quantization failed: {e}; using bf16")
    transformer.to("cuda:0")
    return transformer


def move_lens_transformer_to_cpu(transformer):
    if transformer is None:
        return
    transformer.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def move_lens_vae_to_gpu(vae):
    if vae is None:
        return
    vae.to("cuda:0")


def move_lens_vae_to_cpu(vae):
    if vae is None:
        return
    vae.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
