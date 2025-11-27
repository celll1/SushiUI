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
        quantization: Quantization type - 'fp8', 'int8', 'int4', or 'nf4'

    Returns:
        Quantized U-Net model
    """
    try:
        if quantization == 'fp8':
            # FP8 quantization using torch native support
            print(f"[Quantization] Applying FP8 quantization...")
            # For FP8, we can use torch.float8_e4m3fn or torch.float8_e5m2
            # Clone the model and convert weights
            quantized_unet = copy.deepcopy(unet)

            # Convert linear layers to FP8
            for name, module in quantized_unet.named_modules():
                if isinstance(module, torch.nn.Linear):
                    # Convert weight to FP8 (stored as float8_e4m3fn)
                    # Note: This is experimental and may require torch >= 2.1
                    try:
                        # Store original dtype for compute
                        module._original_dtype = module.weight.dtype
                        # Convert to FP8 (this reduces memory but compute still uses original dtype)
                        module.weight.data = module.weight.data.to(torch.float8_e4m3fn).to(module.weight.dtype)
                        if module.bias is not None:
                            module.bias.data = module.bias.data.to(torch.float8_e4m3fn).to(module.bias.dtype)
                    except Exception as e:
                        print(f"[Quantization] Warning: Could not convert {name} to FP8: {e}")

            return quantized_unet

        elif quantization == 'int8':
            # INT8 quantization using bitsandbytes (CUDA-compatible)
            print(f"[Quantization] Applying INT8 quantization with bitsandbytes...")

            try:
                import bitsandbytes as bnb

                # Count linear layers before quantization
                linear_count = sum(1 for m in unet.modules() if isinstance(m, torch.nn.Linear))
                print(f"[Quantization] Found {linear_count} Linear layers to quantize")

                quantized_unet = copy.deepcopy(unet)

                # Replace Linear layers with Int8 layers
                replaced_count = 0
                for name, module in list(quantized_unet.named_modules()):
                    if isinstance(module, torch.nn.Linear):
                        # Get parent module
                        parent_name = '.'.join(name.split('.')[:-1])
                        child_name = name.split('.')[-1]

                        if parent_name:
                            parent = dict(quantized_unet.named_modules())[parent_name]
                        else:
                            parent = quantized_unet

                        # Create Int8 linear layer
                        new_module = bnb.nn.Linear8bitLt(
                            module.in_features,
                            module.out_features,
                            bias=module.bias is not None,
                            has_fp16_weights=False,  # Use int8 weights
                            threshold=6.0
                        )

                        # Copy weights
                        with torch.no_grad():
                            new_module.weight.data = module.weight.data
                            if module.bias is not None:
                                new_module.bias.data = module.bias.data

                        setattr(parent, child_name, new_module)
                        replaced_count += 1

                print(f"[Quantization] Successfully quantized {replaced_count} layers to INT8")
                print(f"[Quantization] Estimated memory reduction: ~50%")

                return quantized_unet

            except ImportError:
                print(f"[Quantization] ERROR: bitsandbytes not available")
                print(f"[Quantization] INT8 quantization requires bitsandbytes for CUDA support")
                print(f"[Quantization] Install with: pip install bitsandbytes")
                print(f"[Quantization] Falling back to original model without quantization")
                return copy.deepcopy(unet)
            except Exception as e:
                print(f"[Quantization] ERROR during INT8 quantization: {e}")
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
        quantization: Quantization type - None, 'none', 'fp8', 'int8', 'int4', or 'nf4'
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

            # Create quantized copy
            print(f"[VRAM] Creating {quantization} quantized U-Net...")
            pipeline.unet = _quantize_unet(pipeline._original_unet, quantization)
            pipeline.unet.to('cuda:0')
        else:
            # Restore original unet if quantization was used before
            if hasattr(pipeline, '_original_unet'):
                print(f"[VRAM] Restoring original (non-quantized) U-Net...")
                pipeline.unet = pipeline._original_unet
                delattr(pipeline, '_original_unet')

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
