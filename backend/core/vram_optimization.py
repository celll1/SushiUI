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


def log_device_status(stage: str, pipeline, show_details: bool = False):
    """Log device status of all pipeline components

    Args:
        stage: Description of current stage (e.g., "After moving to GPU")
        pipeline: The diffusers pipeline
        show_details: Show detailed submodule information
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
        """Check if module is quantized"""
        # Check for quantized linear layers
        for name, submodule in module.named_modules():
            if hasattr(submodule, '_packed_params'):
                return "quantized (qint8)"
            if 'quantized' in str(type(submodule)).lower():
                return f"quantized ({type(submodule).__name__})"
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
    if hasattr(pipeline, 'vae') and pipeline.vae is not None:
        try:
            device = next(pipeline.vae.parameters()).device
            dtype = get_dtype_info(pipeline.vae)
            print(f"  VAE:            {device} ({dtype})")
        except:
            print(f"  VAE:            no parameters")

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
        - uint2-uint8: UintX weight-only quantization (via torchao, for future training support)
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
                quantize_(quantized_unet, config)

                # Estimate memory reduction based on bit width
                bit_width = int(quantization.replace('uint', ''))
                reduction_pct = (1 - bit_width / 16) * 100  # Assuming FP16 baseline

                print(f"[Quantization] Successfully quantized U-Net to {quantization.upper()}")
                print(f"[Quantization] Note: This is weight-only quantization")
                print(f"[Quantization] Estimated memory reduction: ~{reduction_pct:.0f}%")
                print(f"[Quantization] Quality: Lower bits = higher compression but more quality loss")

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
        pipeline.text_encoder.to('cuda:0')

    if hasattr(pipeline, 'text_encoder_2') and pipeline.text_encoder_2 is not None:
        pipeline.text_encoder_2.to('cuda:0')

    torch.cuda.empty_cache()


def move_text_encoders_to_cpu(pipeline):
    """Move text encoders to CPU to free VRAM"""
    print("[VRAM] Moving Text Encoders to CPU to free VRAM...")

    if hasattr(pipeline, 'text_encoder') and pipeline.text_encoder is not None:
        pipeline.text_encoder.to('cpu')

    if hasattr(pipeline, 'text_encoder_2') and pipeline.text_encoder_2 is not None:
        pipeline.text_encoder_2.to('cpu')

    torch.cuda.empty_cache()


def move_unet_to_gpu(pipeline, quantization: Optional[str] = None):
    """Move U-Net to GPU for inference, optionally with quantization

    Args:
        pipeline: The diffusers pipeline
        quantization: Quantization type - None, 'none', 'fp8_e4m3fn', 'fp8_e5m2', etc.
    """
    # Normalize quantization parameter
    if quantization in [None, "", "none"]:
        quantization = None

    if quantization:
        print(f"[VRAM] Moving U-Net to GPU with {quantization} quantization...")
    else:
        print("[VRAM] Moving U-Net to GPU for inference...")

    if hasattr(pipeline, 'unet') and pipeline.unet is not None:
        if quantization:
            # Store original unet on CPU if not already stored
            if not hasattr(pipeline, '_original_unet'):
                print(f"[VRAM] Storing original U-Net on CPU...")
                pipeline._original_unet = pipeline.unet
                # Ensure original is on CPU
                pipeline._original_unet.to('cpu')

            # Check if we have a cached quantized model
            if not hasattr(pipeline, '_quantized_unet_cache'):
                pipeline._quantized_unet_cache = {}

            # Use cached quantized model if available
            if quantization in pipeline._quantized_unet_cache:
                print(f"[VRAM] Using cached {quantization} quantized U-Net...")
                pipeline.unet = pipeline._quantized_unet_cache[quantization]
            else:
                # Create new quantized copy and cache it
                print(f"[VRAM] Creating {quantization} quantized U-Net...")
                quantized_unet = _quantize_unet(pipeline._original_unet, quantization)
                # Keep quantized model on CPU for caching
                quantized_unet.to('cpu')
                pipeline._quantized_unet_cache[quantization] = quantized_unet
                pipeline.unet = quantized_unet

            # Move to GPU
            pipeline.unet.to('cuda:0')
        else:
            # Restore original unet if quantization was used before
            if hasattr(pipeline, '_original_unet'):
                print(f"[VRAM] Restoring original (non-quantized) U-Net...")
                pipeline.unet = pipeline._original_unet
                # Keep cache but don't delete it (for future use)

            pipeline.unet.to('cuda:0')

    torch.cuda.empty_cache()


def move_unet_to_cpu(pipeline):
    """Move U-Net to CPU to free VRAM"""
    print("[VRAM] Moving U-Net to CPU to free VRAM...")

    if hasattr(pipeline, 'unet') and pipeline.unet is not None:
        pipeline.unet.to('cpu')

    torch.cuda.empty_cache()


def move_vae_to_gpu(pipeline):
    """Move VAE to GPU for decode"""
    print("[VRAM] Moving VAE to GPU for decode...")

    if hasattr(pipeline, 'vae') and pipeline.vae is not None:
        pipeline.vae.to('cuda:0')

    torch.cuda.empty_cache()


def move_vae_to_cpu(pipeline):
    """Move VAE to CPU to free VRAM"""
    print("[VRAM] Moving VAE to CPU to free VRAM...")

    if hasattr(pipeline, 'vae') and pipeline.vae is not None:
        pipeline.vae.to('cpu')

    torch.cuda.empty_cache()
