"""VRAM Optimization utilities for sequential model loading

This module implements sequential VRAM loading:
- Text Encoder → CPU after encoding
- U-Net → GPU only during inference
- VAE → GPU only during decode
"""

import torch
from typing import Optional


def log_device_status(stage: str, pipeline, show_details: bool = False):
    """Log device status of all pipeline components

    Args:
        stage: Description of current stage (e.g., "Before text encoding")
        pipeline: The diffusers pipeline
        show_details: Show detailed submodule information
    """
    print(f"\n{'='*60}")
    print(f"[VRAM] Device Status: {stage}")
    print(f"{'='*60}")

    # Text Encoder
    if hasattr(pipeline, 'text_encoder') and pipeline.text_encoder is not None:
        try:
            device = next(pipeline.text_encoder.parameters()).device
            print(f"  Text Encoder:   {device}")
        except:
            print(f"  Text Encoder:   no parameters")

    # Text Encoder 2
    if hasattr(pipeline, 'text_encoder_2') and pipeline.text_encoder_2 is not None:
        try:
            device = next(pipeline.text_encoder_2.parameters()).device
            print(f"  Text Encoder 2: {device}")
        except:
            print(f"  Text Encoder 2: no parameters")

    # U-Net
    if hasattr(pipeline, 'unet') and pipeline.unet is not None:
        try:
            device = next(pipeline.unet.parameters()).device
            print(f"  U-Net:          {device}")

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
            print(f"  VAE:            {device}")
        except:
            print(f"  VAE:            no parameters")

    # VRAM usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"\n  VRAM: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    print(f"{'='*60}\n")


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


def move_unet_to_gpu(pipeline):
    """Move U-Net to GPU for inference"""
    print("[VRAM] Moving U-Net to GPU for inference...")

    if hasattr(pipeline, 'unet') and pipeline.unet is not None:
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
