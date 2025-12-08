"""VRAM Optimization utilities for sequential model loading

This module implements sequential VRAM loading:
- Text Encoder → CPU after encoding
- U-Net → GPU only during inference
- VAE → GPU only during decode

Also supports on-demand U-Net quantization for VRAM reduction.
"""

import torch
from typing import Optional, Literal
import copy


def log_device_status(stage: str, pipeline, show_details: bool = False, zimage_components: dict = None):
    """Log device status of all pipeline components

    Args:
        stage: Description of current stage (e.g., "After moving to GPU")
        pipeline: The diffusers pipeline (or None for Z-Image)
        show_details: Show detailed submodule information
        zimage_components: Dict with Z-Image components (text_encoder, transformer, vae)
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
        """Check if module is quantized (torchao, bitsandbytes, etc.)"""
        # Check first few Linear layers for quantization
        checked_count = 0
        max_check = 5  # Check first 5 linear layers

        for name, submodule in module.named_modules():
            if not isinstance(submodule, torch.nn.Linear):
                continue

            checked_count += 1
            if checked_count > max_check:
                break

            submodule_type = type(submodule).__name__

            # torchao AffineQuantizedTensor or related classes in module type
            if 'AffineQuantized' in submodule_type or 'Quantized' in submodule_type:
                return f"quantized (torchao: {submodule_type})"

            # Check weight attributes for quantization
            if hasattr(submodule, 'weight'):
                weight = submodule.weight
                weight_type = type(weight).__name__

                # torchao quantized weights (AffineQuantizedTensor, etc.)
                if 'AffineQuantized' in weight_type:
                    # Try to extract layout/dtype info
                    if hasattr(weight, 'layout_type'):
                        try:
                            layout = weight.layout_type
                            layout_name = layout.__name__ if hasattr(layout, '__name__') else str(layout)
                            # Extract dtype if possible
                            if hasattr(layout, 'dtype'):
                                dtype = str(layout.dtype).replace('torch.', '')
                                return f"quantized (torchao {dtype.upper()})"
                            return f"quantized (torchao: {layout_name})"
                        except:
                            pass
                    return f"quantized (torchao: {weight_type})"

                # Check weight dtype for FP8
                if hasattr(weight, 'dtype'):
                    dtype_str = str(weight.dtype)
                    if 'float8' in dtype_str:
                        dtype_name = dtype_str.replace('torch.', '').upper()
                        return f"quantized ({dtype_name})"

            # Legacy quantization checks
            if hasattr(submodule, '_packed_params'):
                return "quantized (qint8)"

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
        # Text Encoder
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
        quantization: Quantization type - 'fp8_e4m3fn', 'fp8_e5m2', 'uint2'-'uint8', etc.

    Returns:
        Quantized U-Net model

    Supported quantization types:
        - fp8_e4m3fn, fp8_e5m2: FP8 quantization (via .to(), ~50% VRAM reduction)
          * Weight: FP8, Activation: FP16 (via autocast)
        - uint2-uint8: UintX weight-only quantization (via torchao, for future training support)
          * Weight: UINTX (internally), Activation: FP16 (via autocast)
        - int4, nf4: 4-bit quantization (via bitsandbytes, not recommended)
        - int8: INT8 quantization (not recommended, causes slowdown)
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

        elif quantization == 'int8':
            # INT8 quantization is not recommended for inference
            print(f"[Quantization] WARNING: INT8 quantization is not recommended")
            print(f"[Quantization] INT8 causes significant slowdown with minimal VRAM savings")
            print(f"[Quantization] Recommendation: Use FP8 (Ada/Hopper GPUs) or disable quantization")
            print(f"[Quantization] Sequential offloading already provides efficient VRAM usage")
            print(f"[Quantization] Falling back to original model without quantization")
            return copy.deepcopy(unet)

        elif quantization in ['uint2', 'uint3', 'uint4', 'uint5', 'uint6', 'uint7', 'uint8']:
            # UintX quantization using torchao (for future training support)
            print(f"[Quantization] Applying {quantization.upper()} weight-only quantization...")

            try:
                from torchao.quantization.quant_api import quantize_, UIntXWeightOnlyConfig

                # Map string to torch dtype
                uint_dtypes = {
                    'uint2': torch.uint2,
                    'uint3': torch.uint3,
                    'uint4': torch.uint4,
                    'uint5': torch.uint5,
                    'uint6': torch.uint6,
                    'uint7': torch.uint7,
                    'uint8': torch.uint8,
                }

                if not hasattr(torch, quantization):
                    print(f"[Quantization] ERROR: PyTorch does not support {quantization}")
                    print(f"[Quantization] This may require a newer PyTorch version")
                    print(f"[Quantization] Falling back to original model without quantization")
                    return copy.deepcopy(unet)

                # Clone the model
                quantized_unet = copy.deepcopy(unet)

                # Create quantization config
                # group_size: controls quantization granularity (smaller = more fine-grained)
                config = UIntXWeightOnlyConfig(
                    dtype=uint_dtypes[quantization],
                    group_size=64,  # Default from torchao
                )

                # Apply quantization (modifies model in-place)
                print(f"[Quantization] Quantizing model weights to {quantization}...")

                # Calculate model size before quantization (on CPU)
                def get_model_size(model):
                    """Calculate model size in bytes"""
                    total_size = 0
                    for param in model.parameters():
                        total_size += param.nelement() * param.element_size()
                    return total_size

                size_before = get_model_size(quantized_unet) / (1024**3)  # GB
                print(f"[Quantization] Model size before: {size_before:.3f} GB")

                quantize_(quantized_unet, config)

                size_after = get_model_size(quantized_unet) / (1024**3)  # GB
                actual_reduction_pct = (1 - size_after / size_before) * 100

                print(f"[Quantization] Model size after:  {size_after:.3f} GB")
                print(f"[Quantization] Actual reduction:   {actual_reduction_pct:.1f}%")

                # Estimate memory reduction based on bit width
                bit_width = int(quantization.replace('uint', ''))
                expected_reduction_pct = (1 - bit_width / 16) * 100  # Assuming FP16 baseline

                print(f"[Quantization] Expected reduction: {expected_reduction_pct:.0f}%")
                print(f"[Quantization] Successfully quantized U-Net to {quantization.upper()}")
                print(f"[Quantization] Note: This is weight-only quantization")
                print(f"[Quantization] Activations will use FP16 (via autocast)")

                if actual_reduction_pct < expected_reduction_pct * 0.5:
                    print(f"[Quantization] WARNING: Actual reduction is much lower than expected")
                    print(f"[Quantization] This may indicate fake quantization (model still in high precision)")

                # Mark as UINT quantized for autocast detection
                # Store as a custom attribute since AffineQuantizedTensor reports dtype as float32
                quantized_unet._is_uint_quantized = True

                return quantized_unet

            except ImportError:
                print(f"[Quantization] ERROR: torchao library not installed")
                print(f"[Quantization] Install with: pip install torchao")
                print(f"[Quantization] Falling back to original model without quantization")
                return copy.deepcopy(unet)
            except Exception as e:
                print(f"[Quantization] ERROR during {quantization.upper()} conversion: {e}")
                import traceback
                traceback.print_exc()
                print(f"[Quantization] Falling back to original model without quantization")
                return copy.deepcopy(unet)

        elif quantization in ['int4', 'nf4']:
            # INT4/NF4 quantization using bitsandbytes if available
            print(f"[Quantization] Applying {quantization.upper()} quantization...")
            try:
                import bitsandbytes as bnb
                from transformers import BitsAndBytesConfig

                # Note: This is a simplified approach
                # Full bitsandbytes integration would require model reload
                quantized_unet = copy.deepcopy(unet)

                # Convert Linear layers to 4-bit
                for name, module in quantized_unet.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        # Replace with bnb Linear4bit layer
                        parent_name = '.'.join(name.split('.')[:-1])
                        child_name = name.split('.')[-1]

                        if parent_name:
                            parent = dict(quantized_unet.named_modules())[parent_name]
                        else:
                            parent = quantized_unet

                        # Create 4-bit linear layer
                        quant_type = "nf4" if quantization == 'nf4' else "int4"
                        new_module = bnb.nn.Linear4bit(
                            module.in_features,
                            module.out_features,
                            bias=module.bias is not None,
                            compute_dtype=torch.float16,
                            quant_type=quant_type
                        )

                        # Copy weights
                        new_module.weight.data = module.weight.data
                        if module.bias is not None:
                            new_module.bias.data = module.bias.data

                        setattr(parent, child_name, new_module)

                return quantized_unet

            except ImportError:
                print(f"[Quantization] Warning: bitsandbytes not available, falling back to INT8")
                return _quantize_unet(unet, 'int8')

        else:
            raise ValueError(f"Unsupported quantization type: {quantization}")

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
        vae.to('cuda:0', non_blocking=False)


def move_zimage_vae_to_cpu(vae):
    """Move Z-Image VAE to CPU to free VRAM

    Args:
        vae: Z-Image VAE model
    """
    print("[VRAM] Moving Z-Image VAE to CPU to free VRAM...")
    if vae is not None:
        vae.to('cpu', non_blocking=False)


def _quantize_transformer(transformer, quantization: str):
    """Create a quantized copy of Z-Image transformer

    Z-Image transformer requires special FP8 handling:
    - FP8 must only be applied to Linear layer WEIGHTS, not buffers
    - Standard .to() converts everything (weights + buffers), causing dtype mismatch
    - Solution: Manually iterate through Linear layers and convert only weights

    Args:
        transformer: Original Z-Image transformer model
        quantization: Quantization type - 'fp8_e4m3fn', 'fp8_e5m2', 'uint2'-'uint8'

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

    # UINT quantization is supported (weight-only, doesn't affect buffers)
    if quantization in ['uint2', 'uint3', 'uint4', 'uint5', 'uint6', 'uint7', 'uint8']:
        # Reuse U-Net quantization logic for UINT
        return _quantize_unet(transformer, quantization)

    # Unknown quantization type
    print(f"[Quantization] ERROR: Unknown quantization type: {quantization}")
    print(f"[Quantization] Falling back to non-quantized transformer")
    return copy.deepcopy(transformer)


def _quantize_text_encoder(text_encoder, quantization: str):
    """Create a quantized copy of Z-Image text encoder

    Uses same weight-only FP8/UINT quantization as Transformer.
    Z-Image text encoder (Qwen 3.4B) is large, so quantization can significantly reduce VRAM.

    Args:
        text_encoder: Original Z-Image text encoder model (Qwen)
        quantization: Quantization type - 'fp8_e4m3fn', 'fp8_e5m2', 'uint2'-'uint8', etc.

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

            # Convert only Linear layer weights to FP8 (leave buffers/embeddings in BF16)
            converted_count = 0
            for name, module in quantized_text_encoder.named_modules():
                if isinstance(module, torch.nn.Linear):
                    # Convert weight parameter only
                    if hasattr(module, 'weight') and module.weight is not None:
                        module.weight.data = module.weight.data.to(fp8_dtype)
                        converted_count += 1
                    # Keep bias in original dtype (if exists)

            print(f"[Quantization] Successfully converted {converted_count} Linear layers to {dtype_name}")
            print(f"[Quantization] Embeddings and buffers kept in BF16")
            print(f"[Quantization] Note: Compute will use mixed precision automatically (autocast)")

            return quantized_text_encoder

        except Exception as e:
            print(f"[Quantization] ERROR during {dtype_name} conversion: {e}")
            import traceback
            traceback.print_exc()
            print(f"[Quantization] Falling back to original model without quantization")
            return copy.deepcopy(text_encoder)

    # UINT quantization
    if quantization in ['uint2', 'uint3', 'uint4', 'uint5', 'uint6', 'uint7', 'uint8']:
        # Reuse U-Net quantization logic for UINT
        return _quantize_unet(text_encoder, quantization)

    # Unknown quantization type
    print(f"[Quantization] ERROR: Unknown quantization type: {quantization}")
    print(f"[Quantization] Falling back to non-quantized text encoder")
    return copy.deepcopy(text_encoder)
