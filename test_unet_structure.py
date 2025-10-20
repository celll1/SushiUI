"""Test script to check U-Net structure and channel dimensions"""
import torch
from diffusers import StableDiffusionXLPipeline

print("Loading SDXL pipeline to inspect U-Net structure...")

# Use a small model or just load the U-Net config
model_path = "stabilityai/stable-diffusion-xl-base-1.0"

try:
    from diffusers import UNet2DConditionModel
    print(f"Loading U-Net from {model_path}...")
    unet = UNet2DConditionModel.from_pretrained(
        model_path,
        subfolder="unet",
        torch_dtype=torch.float16,
        variant="fp16"
    )

    print("\n=== DOWN BLOCKS ===")
    for down_idx, block in enumerate(unet.down_blocks):
        print(f"\ndown_blocks[{down_idx}]: {type(block).__name__}")

        if hasattr(block, 'resnets'):
            print(f"  resnets: {len(block.resnets)}")
            if len(block.resnets) > 0:
                # Check output channels of first resnet
                if hasattr(block.resnets[0], 'conv2'):
                    out_ch = block.resnets[0].conv2.out_channels
                    print(f"    output channels: {out_ch}")

        if hasattr(block, 'attentions') and block.attentions is not None:
            print(f"  attentions: {len(block.attentions)}")
            for attn_idx, attention in enumerate(block.attentions):
                if hasattr(attention, 'transformer_blocks'):
                    # Get input dimension from first transformer block
                    trans_block = attention.transformer_blocks[0]
                    if hasattr(trans_block, 'attn1'):
                        if hasattr(trans_block.attn1, 'to_q'):
                            in_features = trans_block.attn1.to_q.in_features
                            print(f"    attentions[{attn_idx}]: {len(attention.transformer_blocks)} transformer_blocks, {in_features} channels")

    print("\n=== MIDDLE BLOCK ===")
    print(f"mid_block: {type(unet.mid_block).__name__}")
    if hasattr(unet.mid_block, 'attentions'):
        for attn_idx, attention in enumerate(unet.mid_block.attentions):
            if hasattr(attention, 'transformer_blocks'):
                trans_block = attention.transformer_blocks[0]
                if hasattr(trans_block, 'attn1'):
                    if hasattr(trans_block.attn1, 'to_q'):
                        in_features = trans_block.attn1.to_q.in_features
                        print(f"  attentions[{attn_idx}]: {len(attention.transformer_blocks)} transformer_blocks, {in_features} channels")

    print("\n=== UP BLOCKS ===")
    for up_idx, block in enumerate(unet.up_blocks):
        print(f"\nup_blocks[{up_idx}]: {type(block).__name__}")

        if hasattr(block, 'attentions') and block.attentions is not None:
            print(f"  attentions: {len(block.attentions)}")
            for attn_idx, attention in enumerate(block.attentions):
                if hasattr(attention, 'transformer_blocks'):
                    trans_block = attention.transformer_blocks[0]
                    if hasattr(trans_block, 'attn1'):
                        if hasattr(trans_block.attn1, 'to_q'):
                            in_features = trans_block.attn1.to_q.in_features
                            print(f"    attentions[{attn_idx}]: {len(attention.transformer_blocks)} transformer_blocks, {in_features} channels")

except Exception as e:
    print(f"Error: {e}")
    print("\nTrying to load config only...")
    from diffusers import UNet2DConditionModel
    config = UNet2DConditionModel.load_config(model_path, subfolder="unet")
    print(f"Config: {config}")
