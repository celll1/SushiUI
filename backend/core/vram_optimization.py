"""VRAM Optimization utilities for sequential model loading

This module implements sequential VRAM loading:
- Text Encoder → CPU after encoding
- U-Net → GPU only during inference
- VAE → GPU only during decode

Also supports on-demand U-Net FP8 quantization for VRAM reduction.
Note: torchao/UINT quantization has been removed due to compatibility issues.
"""

import torch
from typing import Optional
import copy


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
                return copy.deepcopy(unet)

        else:
            print(f"[Quantization] Unsupported quantization type: {quantization}")
            print(f"[Quantization] Supported types: fp8_e4m3fn, fp8_e5m2")
            print(f"[Quantization] Falling back to original model without quantization")
            return copy.deepcopy(unet)

    except Exception as e:
        print(f"[Quantization] Error during quantization: {e}")
        print(f"[Quantization] Falling back to original model without quantization")
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
            return copy.deepcopy(transformer)

    # Unknown quantization type
    print(f"[Quantization] ERROR: Unknown quantization type: {quantization}")
    print(f"[Quantization] Supported types: fp8_e4m3fn, fp8_e5m2")
    print(f"[Quantization] Falling back to non-quantized transformer")
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
            return copy.deepcopy(text_encoder)

    # Unknown quantization type
    print(f"[Quantization] ERROR: Unknown quantization type: {quantization}")
    print(f"[Quantization] Supported types: fp8_e4m3fn, fp8_e5m2")
    print(f"[Quantization] Falling back to non-quantized text encoder")
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

    if transformer is None:
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


def _anima_quantize_fp8(model, quantization: str, label: str):
    """Apply FP8 weight-only quantization with on-the-fly dequant to all
    nn.Linear modules in `model`. Embeddings are left untouched (they're
    looked up, not matmul'd, so we can't safely dequant on lookup).

    Returns the model with Linear forward methods replaced.
    """
    if quantization == 'fp8_e4m3fn':
        fp8_dtype = torch.float8_e4m3fn
        dtype_name = "FP8 E4M3FN"
    elif quantization == 'fp8_e5m2':
        fp8_dtype = torch.float8_e5m2
        dtype_name = "FP8 E5M2"
    else:
        print(f"[Quantization] Unsupported {quantization} for Anima; only fp8_e4m3fn/fp8_e5m2 supported")
        return model

    if not hasattr(torch, 'float8_e4m3fn'):
        print(f"[Quantization] PyTorch {torch.__version__} does not support FP8. Skipping.")
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
