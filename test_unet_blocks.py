"""Test script to check actual U-Net block structure and channel dimensions"""
import torch
from diffusers import StableDiffusionXLPipeline

print("Loading SDXL model to inspect U-Net structure...")

model_path = "E:\\sd_forge\\stable-diffusion-webui-forge\\models\\Stable-diffusion\\animagine-xl-3.1.safetensors"

print(f"Loading model: {model_path}")
try:
    from diffusers import StableDiffusionXLPipeline

    # Load pipeline
    pipeline = StableDiffusionXLPipeline.from_single_file(
        model_path,
        torch_dtype=torch.float16,
        use_safetensors=True
    )

    unet = pipeline.unet

    print("\n=== DOWN BLOCKS ===")
    kohya_idx = 0  # kohya-ss style cumulative index
    for down_idx, block in enumerate(unet.down_blocks):
        print(f"\ndown_blocks[{down_idx}]: {type(block).__name__}")

        if hasattr(block, 'attentions') and block.attentions is not None:
            print(f"  has {len(block.attentions)} attentions")
            for attn_idx, attention in enumerate(block.attentions):
                if hasattr(attention, 'transformer_blocks') and len(attention.transformer_blocks) > 0:
                    # Get input dimension from first transformer block
                    trans_block = attention.transformer_blocks[0]
                    if hasattr(trans_block, 'attn1') and hasattr(trans_block.attn1, 'to_q'):
                        in_features = trans_block.attn1.to_q.in_features
                        print(f"    down_blocks[{down_idx}].attentions[{attn_idx}]: {in_features}ch, {len(attention.transformer_blocks)} transformer_blocks")
                        print(f"      -> kohya-ss index: input_blocks_{kohya_idx}")
                        kohya_idx += 1
        else:
            print(f"  NO attentions")

    print("\n=== MIDDLE BLOCK ===")
    print(f"mid_block: {type(unet.mid_block).__name__}")
    if hasattr(unet.mid_block, 'attentions'):
        for attn_idx, attention in enumerate(unet.mid_block.attentions):
            if hasattr(attention, 'transformer_blocks') and len(attention.transformer_blocks) > 0:
                trans_block = attention.transformer_blocks[0]
                if hasattr(trans_block, 'attn1') and hasattr(trans_block.attn1, 'to_q'):
                    in_features = trans_block.attn1.to_q.in_features
                    print(f"  mid_block.attentions[{attn_idx}]: {in_features}ch, {len(attention.transformer_blocks)} transformer_blocks")
                    print(f"    -> kohya-ss: middle_block_1")

    print("\n=== Comparison with LLLite model ===")
    print("LLLite model has:")
    print("  input_blocks_4: 640ch")
    print("  input_blocks_5: 640ch")
    print("  input_blocks_7: 1280ch")
    print("  input_blocks_8: 1280ch")
    print("  middle_block_1: 1280ch")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
