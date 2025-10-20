"""Test script to check LLLite model keys"""
import torch
from safetensors.torch import load_file

model_path = r"E:\sd_forge\stable-diffusion-webui-forge\models\ControlNet\kohya_controllllite_xl_canny_anime.safetensors"

print("Loading LLLite model...")
state_dict = load_file(model_path)

print(f"\nTotal keys: {len(state_dict)}")

# Find all unique base module names
base_names = set()
for key in state_dict.keys():
    # Extract base name (everything before .conditioning1/.down/.mid/.up)
    parts = key.split('.')
    base_parts = []
    for part in parts:
        if part in ['conditioning1', 'down', 'mid', 'up']:
            break
        base_parts.append(part)
    base_name = '.'.join(base_parts)
    base_names.add(base_name)

print(f"\nUnique base modules: {len(base_names)}")

# Group by block type
input_blocks = sorted([n for n in base_names if 'input_blocks' in n])
middle_blocks = sorted([n for n in base_names if 'middle_block' in n])
output_blocks = sorted([n for n in base_names if 'output_blocks' in n])

print(f"\nInput blocks ({len(input_blocks)}):")
for name in input_blocks:
    # Check down weight shape
    down_key = f"{name}.down.0.weight"
    if down_key in state_dict:
        shape = state_dict[down_key].shape
        print(f"  {name}: down.0.weight shape = {shape}")

print(f"\nMiddle blocks ({len(middle_blocks)}):")
for name in middle_blocks:
    down_key = f"{name}.down.0.weight"
    if down_key in state_dict:
        shape = state_dict[down_key].shape
        print(f"  {name}: down.0.weight shape = {shape}")

print(f"\nOutput blocks ({len(output_blocks)}):")
for name in output_blocks:
    down_key = f"{name}.down.0.weight"
    if down_key in state_dict:
        shape = state_dict[down_key].shape
        print(f"  {name}: down.0.weight shape = {shape}")

# Check specific problematic modules
print("\n--- Checking specific modules ---")
test_modules = [
    'lllite_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q',
    'lllite_unet_input_blocks_5_1_transformer_blocks_0_attn1_to_q',
    'lllite_unet_input_blocks_6_1_transformer_blocks_0_attn1_to_q',
    'lllite_unet_input_blocks_7_1_transformer_blocks_0_attn1_to_q',
    'lllite_unet_input_blocks_8_1_transformer_blocks_0_attn1_to_q',
]

for test_mod in test_modules:
    down_key = f"{test_mod}.down.0.weight"
    if down_key in state_dict:
        shape = state_dict[down_key].shape
        print(f"✓ {test_mod}: {shape}")
    else:
        print(f"✗ {test_mod}: NOT FOUND")
