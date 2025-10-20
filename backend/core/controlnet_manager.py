"""
ControlNet Manager for loading and applying ControlNet models.

Supports:
- Standard ControlNet models
- ControlNet-LLLite models
- Layer-wise weight control
- Step range control
"""

import os
import torch
from typing import Optional, Dict, List, Any
from pathlib import Path
from diffusers import ControlNetModel
from PIL import Image
import numpy as np
from config.settings import settings


class ControlNetConfig:
    """Configuration for a single ControlNet application"""
    def __init__(
        self,
        model_path: str,
        image: Image.Image,
        strength: float = 1.0,
        start_step: int = 0,
        end_step: int = 1000,
        layer_weights: Optional[Dict[str, float]] = None,
        prompt: Optional[str] = None,
        is_lllite: bool = False,
    ):
        self.model_path = model_path
        self.image = image
        self.strength = strength
        self.start_step = start_step  # 0-1000 step range
        self.end_step = end_step      # 0-1000 step range
        self.layer_weights = layer_weights  # Per-layer weights like {"IN00": 1.0, ..., "MID": 1.0}
        self.prompt = prompt  # Optional separate prompt for this ControlNet
        self.is_lllite = is_lllite


class ControlNetManager:
    """Manages ControlNet models and their application"""

    def __init__(self, controlnet_dir: Optional[str] = None):
        if controlnet_dir is None:
            controlnet_dir = settings.controlnet_dir
        self.controlnet_dir = Path(controlnet_dir)
        self.additional_dirs: List[Path] = []  # User-configured additional directories
        print(f"[ControlNetManager] ControlNet directory: {self.controlnet_dir}")

        self.loaded_controlnets: Dict[str, ControlNetModel] = {}
        self.loaded_lllites: Dict[str, Any] = {}

        # Track patched layers for cleanup
        self.lllite_patched_layers: List[tuple] = []  # [(layer, original_forward), ...]

        # Track LLLite data for cleanup
        self.current_lllite_modules: Optional[dict] = None  # Cached LLLite weight modules
        self.current_lllite_embeddings: Optional[dict] = None  # Cached conditioning embeddings

    def set_additional_dirs(self, dirs: List[str]):
        """Set additional directories to scan for ControlNets"""
        self.additional_dirs = [Path(d) for d in dirs if d.strip()]
        print(f"[ControlNetManager] Additional directories set: {self.additional_dirs}")

    def _resolve_controlnet_path(self, controlnet_path: str) -> Optional[Path]:
        """Resolve controlnet path, checking default and additional directories"""
        # Try default directory first
        full_path = self.controlnet_dir / controlnet_path
        if full_path.exists():
            return full_path

        # Try additional directories
        for additional_dir in self.additional_dirs:
            full_path = additional_dir / controlnet_path
            if full_path.exists():
                return full_path

        return None

    def get_available_controlnets(self) -> List[str]:
        """Get list of available ControlNet models from default and additional directories"""
        controlnets = []

        # Combine default directory with additional directories
        all_dirs = [self.controlnet_dir] + self.additional_dirs

        for controlnet_dir in all_dirs:
            if not controlnet_dir.exists():
                print(f"[ControlNetManager] Skipping non-existent directory: {controlnet_dir}")
                continue

            print(f"[ControlNetManager] Scanning directory: {controlnet_dir}")
            # Look for .safetensors and .pth files
            for file in controlnet_dir.glob("**/*"):
                if file.suffix in [".safetensors", ".pth", ".pt", ".bin"]:
                    relative_path = file.relative_to(controlnet_dir)
                    controlnets.append(str(relative_path))

        print(f"[ControlNetManager] Total ControlNet models found: {len(controlnets)}")
        return sorted(list(set(controlnets)))  # Remove duplicates

    def is_lllite_model(self, controlnet_path: str) -> bool:
        """Check if a ControlNet model is LLLite format

        LLLite models have characteristic keys like:
        - lllite_unet_input_blocks_X_1_transformer_blocks_0_attn1_to_q.lora_down.weight
        - lllite_unet_middle_block_1_transformer_blocks_0_attn1_to_q.lora_down.weight

        Standard ControlNet models have keys like:
        - input_blocks.X.1.proj_in.weight
        - middle_block.1.proj_in.weight
        """
        full_path = self._resolve_controlnet_path(controlnet_path)

        if full_path is None:
            return False

        try:
            # Load state dict keys
            if full_path.suffix == '.safetensors':
                from safetensors import safe_open
                with safe_open(str(full_path), framework="pt", device="cpu") as f:
                    keys = list(f.keys())
            else:
                import torch
                state_dict = torch.load(str(full_path), map_location="cpu")
                keys = list(state_dict.keys())

            # Check for LLLite-specific keys
            lllite_indicators = [
                'lllite_unet',
                'lllite_mid',
                'lora_down.weight',
                'lora_up.weight'
            ]

            # If any key contains LLLite indicators, it's likely LLLite
            for key in keys[:50]:  # Check first 50 keys
                key_lower = key.lower()
                if any(indicator in key_lower for indicator in lllite_indicators):
                    print(f"[ControlNetManager] Detected LLLite model: {controlnet_path}")
                    print(f"[ControlNetManager] Sample key: {key}")
                    return True

            # Standard ControlNet has different structure
            print(f"[ControlNetManager] Detected standard ControlNet model: {controlnet_path}")
            return False

        except Exception as e:
            print(f"[ControlNetManager] Error detecting model type: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_controlnet_layers(self, controlnet_path: str) -> List[str]:
        """Get layer structure for a ControlNet model

        ControlNet uses Zero Convolution and has:
        - down_blocks (IN00-IN11): Input blocks
        - mid_block (MID): Middle block
        - NO up_blocks: ControlNet doesn't have output blocks (zero convolution ends at mid)

        Returns list like: ['IN00', 'IN01', ..., 'IN11', 'MID']
        """
        full_path = self._resolve_controlnet_path(controlnet_path)

        if full_path is None:
            print(f"[ControlNetManager] ControlNet not found: {controlnet_path}")
            return []

        try:
            # Standard ControlNet has 12 input blocks + 1 middle block
            # IN00-IN11 correspond to down_blocks.0-11
            # MID corresponds to mid_block
            layers = []

            # Add input blocks (IN00-IN11)
            for i in range(12):
                layers.append(f"IN{i:02d}")

            # Add middle block
            layers.append("MID")

            print(f"[ControlNetManager] ControlNet layers for {controlnet_path}: {layers}")
            return layers

        except Exception as e:
            print(f"[ControlNetManager] Error getting ControlNet layers: {e}")
            import traceback
            traceback.print_exc()
            return []

    def load_controlnet(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        is_lllite: bool = False
    ) -> Optional[ControlNetModel]:
        """Load a ControlNet model"""

        # Check if already loaded
        if model_path in self.loaded_controlnets:
            return self.loaded_controlnets[model_path]

        full_path = self._resolve_controlnet_path(model_path)

        if full_path is None:
            print(f"[ControlNetManager] ControlNet model not found: {model_path}")
            return None

        try:
            if is_lllite:
                # Load ControlNet-LLLite
                print(f"Loading ControlNet-LLLite from {full_path}")
                lllite_model = self._load_lllite_model(full_path, device, dtype)
                if lllite_model is not None:
                    self.loaded_lllites[model_path] = lllite_model
                    print(f"ControlNet-LLLite loaded successfully: {model_path}")
                return lllite_model
            else:
                # Load standard ControlNet
                print(f"Loading ControlNet from {full_path}")

                # Try to load as diffusers ControlNet
                if full_path.is_dir():
                    controlnet = ControlNetModel.from_pretrained(
                        str(full_path),
                        torch_dtype=dtype
                    )
                else:
                    # Load from single file
                    controlnet = ControlNetModel.from_single_file(
                        str(full_path),
                        torch_dtype=dtype
                    )

                controlnet = controlnet.to(device)
                self.loaded_controlnets[model_path] = controlnet

                print(f"ControlNet loaded successfully: {model_path}")
                return controlnet

        except Exception as e:
            print(f"Failed to load ControlNet {model_path}: {e}")
            return None

    def _load_lllite_model(self, model_path: Path, device: str, dtype: torch.dtype):
        """Load ControlNet-LLLite model

        ControlNet-LLLite is a lightweight adaptation method that modifies
        intermediate features without a separate ControlNet model.
        """
        try:
            from safetensors.torch import load_file

            # Load state dict
            if model_path.suffix == '.safetensors':
                state_dict = load_file(str(model_path), device=device)
            else:
                state_dict = torch.load(str(model_path), map_location=device)

            # LLLite models are applied differently - they modify U-Net intermediate layers
            # Store the state dict for later application to the U-Net
            lllite_model = {
                'state_dict': state_dict,
                'dtype': dtype,
                'device': device,
                'model_path': str(model_path)
            }

            return lllite_model

        except Exception as e:
            print(f"Failed to load LLLite model: {e}")
            import traceback
            traceback.print_exc()
            return None

    def unload_controlnet(self, model_path: str):
        """Unload a ControlNet model to free memory"""
        if model_path in self.loaded_controlnets:
            del self.loaded_controlnets[model_path]
            torch.cuda.empty_cache()
            print(f"ControlNet unloaded: {model_path}")

    def unload_all(self):
        """Unload all ControlNet models"""
        self.loaded_controlnets.clear()
        self.loaded_lllites.clear()
        torch.cuda.empty_cache()
        print("All ControlNets unloaded")

    def apply_lllite_to_unet(self, unet, lllite_model: dict, control_image: torch.Tensor):
        """Apply LLLite ControlNet to U-Net

        LLLite works by injecting LoRA-like modules into U-Net attention layers.
        The control image is processed and the LLLite weights modulate the attention.

        Args:
            unet: The U-Net model to modify
            lllite_model: Dict containing LLLite state_dict and metadata
            control_image: Preprocessed control image tensor
        """
        print(f"[ControlNetManager] Applying LLLite to U-Net")

        # Remove any existing LLLite patches first
        self.remove_lllite_patches()

        # Store control image in the lllite model for later use
        lllite_model['control_image'] = control_image

        # Get layer weights if available
        layer_weights = lllite_model.get('_layer_weights', None)

        # Apply LLLite weights to U-Net
        # LLLite modifies attention layers, we need to patch the forward hooks
        self._patch_unet_with_lllite(unet, lllite_model, layer_weights)

        print(f"[ControlNetManager] LLLite applied to U-Net")

    def _patch_unet_with_lllite(self, unet, lllite_model: dict, layer_weights=None):
        """Patch U-Net forward hooks to apply LLLite conditioning

        LLLite uses LoRA-like modules to condition the U-Net attention layers.
        Each attention layer (attn1 q/k/v, attn2 q) gets additional LoRA-style weights
        that are conditioned on the control image.

        Architecture:
        - Conditioning modules process control image into embeddings
        - Down/Mid/Up modules apply LoRA-style transformations
        - Applied to specific transformer blocks in input_blocks
        """
        state_dict = lllite_model['state_dict']
        control_image = lllite_model.get('control_image')
        device = lllite_model.get('device', 'cuda')
        dtype = lllite_model.get('dtype', torch.float16)

        # Log VRAM before processing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            reserved = torch.cuda.memory_reserved(device) / 1024**3
            print(f"[ControlNetManager VRAM] Before LLLite: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")

        # Build LLLite modules from state dict
        lllite_modules = self._build_lllite_modules(state_dict, device, dtype)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            reserved = torch.cuda.memory_reserved(device) / 1024**3
            print(f"[ControlNetManager VRAM] After building modules: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")

        # IMPORTANT: Only process conditioning for modules that will actually be used
        # First, determine which modules match U-Net structure
        matched_modules = self._find_matching_modules(unet, lllite_modules)
        print(f"[ControlNetManager] Found {len(matched_modules)} matching U-Net attention layers")

        # Debug: Check if problematic modules exist in LLLite model
        # test_modules = ['lllite_unet_input_blocks_5_1_transformer_blocks_0_attn1_to_q',
        #                'lllite_unet_input_blocks_6_1_transformer_blocks_0_attn1_to_q']
        # for test_mod in test_modules:
        #     if test_mod in lllite_modules:
        #         down_w = lllite_modules[test_mod].get('down.0', {}).get('weight')
        #         if down_w is not None:
        #             print(f"[ControlNetManager DEBUG] {test_mod}: down_weight shape = {down_w.shape}")
        #         else:
        #             print(f"[ControlNetManager DEBUG] {test_mod}: exists but no down.0 weight")
        #     else:
        #         print(f"[ControlNetManager DEBUG] {test_mod}: NOT in lllite_modules")

        # Process control image ONLY for matched modules
        cond_emb = self._process_control_image_lllite_selective(
            control_image, lllite_modules, matched_modules, device, dtype
        )

        # Store for cleanup
        self.current_lllite_modules = lllite_modules
        self.current_lllite_embeddings = cond_emb

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            reserved = torch.cuda.memory_reserved(device) / 1024**3
            print(f"[ControlNetManager VRAM] After conditioning: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")

        # Patch U-Net attention layers
        self._apply_lllite_patches(unet, lllite_modules, cond_emb, layer_weights)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            reserved = torch.cuda.memory_reserved(device) / 1024**3
            print(f"[ControlNetManager VRAM] After patching: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")

        print(f"[ControlNetManager] LLLite patches applied to {len(lllite_modules)} attention layers")

    def _find_matching_modules(self, unet, lllite_modules: dict) -> set:
        """Find which LLLite modules match actual U-Net structure

        Returns set of module base names that have corresponding U-Net layers.
        kohya-ss uses cumulative indexing: input_blocks_0, 1, 2, ... (across all down_blocks)
        """
        matched = set()
        available_keys = set(lllite_modules.keys())

        # print(f"[ControlNetManager DEBUG] Scanning U-Net structure...")

        # Hardcoded mapping based on actual LLLite model structure
        # From test: input_blocks_4,5 (640ch), input_blocks_7,8 (1280ch)
        # Note: down_blocks[1].attentions[1] uses 1280ch but input_blocks_5 has 640ch weights
        # Therefore input_blocks_5 should NOT be mapped (it's likely an unused module in the LLLite model)
        known_mappings = {
            ('down', 1, 0): 4,  # down_blocks[1].attentions[0] -> input_blocks_4 (640ch)
            # ('down', 1, 1): 5,  # SKIP: down_blocks[1].attentions[1] is 1280ch but input_blocks_5 is 640ch
            ('down', 2, 0): 7,  # down_blocks[2].attentions[0] -> input_blocks_7 (1280ch)
            ('down', 2, 1): 8,  # down_blocks[2].attentions[1] -> input_blocks_8 (1280ch)
        }

        # Scan down_blocks
        if hasattr(unet, 'down_blocks'):
            for down_idx, block in enumerate(unet.down_blocks):
                # print(f"[ControlNetManager DEBUG] down_blocks[{down_idx}]: {type(block).__name__}")

                if hasattr(block, 'attentions') and block.attentions is not None:
                    for attn_idx, attention in enumerate(block.attentions):
                        if hasattr(attention, 'transformer_blocks'):
                            kohya_idx = known_mappings.get(('down', down_idx, attn_idx))

                            if kohya_idx is not None:
                                # print(f"[ControlNetManager DEBUG]   down_blocks[{down_idx}].attentions[{attn_idx}] = input_blocks_{kohya_idx}, {len(attention.transformer_blocks)} transformer_blocks")

                                for trans_idx in range(len(attention.transformer_blocks)):
                                    pattern = f"lllite_unet_input_blocks_{kohya_idx}_1_transformer_blocks_{trans_idx}"

                                    for key in available_keys:
                                        if key.startswith(pattern):
                                            matched.add(key)
                                            # print(f"[ControlNetManager DEBUG]     Matched: {key}")
                            # else:
                                # print(f"[ControlNetManager DEBUG]   down_blocks[{down_idx}].attentions[{attn_idx}] = NO MAPPING (skipped)")

        # Scan mid_block (middle_block)
        if hasattr(unet, 'mid_block'):
            # print(f"[ControlNetManager DEBUG] mid_block: {type(unet.mid_block).__name__}")

            if hasattr(unet.mid_block, 'attentions'):
                for attn_idx, attention in enumerate(unet.mid_block.attentions):
                    if hasattr(attention, 'transformer_blocks'):
                        # print(f"[ControlNetManager DEBUG]   mid_block.attentions[{attn_idx}] = middle_block_1, {len(attention.transformer_blocks)} transformer_blocks")

                        for trans_idx in range(len(attention.transformer_blocks)):
                            # kohya-ss naming: middle_block_1_transformer_blocks_Y
                            pattern = f"lllite_unet_middle_block_1_transformer_blocks_{trans_idx}"

                            for key in available_keys:
                                if key.startswith(pattern):
                                    matched.add(key)
                                    # print(f"[ControlNetManager DEBUG]     Matched: {key}")

        # Scan up_blocks (output_blocks)
        # Note: LLLite models typically don't have output_blocks modules
        # Test showed: NO output_blocks in kohya_controllllite_xl_canny_anime.safetensors
        # Skip scanning up_blocks since they won't match
        # if hasattr(unet, 'up_blocks'):
        #     print(f"[ControlNetManager DEBUG] Skipping up_blocks scan - LLLite models typically have no output_blocks modules")

        return matched

    def _process_control_image_lllite_selective(self, control_image: torch.Tensor,
                                                 lllite_modules: dict, matched_modules: set,
                                                 device: str, dtype: torch.dtype) -> dict:
        """Process control image ONLY for matched modules to save VRAM

        This prevents creating 136 conditioning embeddings when only ~10 are actually used.
        """
        if control_image is None:
            return {}

        # Move control image to device and ensure correct format (in-place where possible)
        if control_image.device != torch.device(device) or control_image.dtype != dtype:
            control_image = control_image.to(device=device, dtype=dtype)

        # Normalize from [0, 1] to [-1, 1] (in-place)
        control_image = control_image.mul(2.0).sub_(1.0)

        # Process ONLY matched modules
        cond_embeddings = {}

        for base_name in matched_modules:
            if base_name not in lllite_modules:
                continue

            submodules = lllite_modules[base_name]

            # Find conditioning1 submodules
            conditioning_modules = {k: v for k, v in submodules.items() if k.startswith('conditioning1')}

            if not conditioning_modules:
                continue

            # Apply conditioning convolutions sequentially
            # Use clone() to avoid in-place modification of control_image
            x = control_image.clone()

            # Sort conditioning module layers by their numeric suffix (0, 2, 4)
            cond_layers = sorted(conditioning_modules.items(), key=lambda item: item[0])

            for layer_name, params in cond_layers:
                if 'weight' in params and 'bias' in params:
                    weight = params['weight']
                    bias = params['bias']

                    # kohya-ss implementation uses specific stride/padding values:
                    # - kernel_size=4, stride=4, padding=0 (most common)
                    # - kernel_size=2, stride=2, padding=0 (for final layer in some depths)
                    # We infer from the weight shape
                    kernel_size = weight.shape[2]

                    if kernel_size == 4:
                        stride = 4
                        padding = 0
                    elif kernel_size == 2:
                        stride = 2
                        padding = 0
                    else:
                        # Fallback for unexpected kernel sizes
                        stride = kernel_size
                        padding = 0

                    # Apply conv2d with kohya-ss parameters
                    x_new = torch.nn.functional.conv2d(x, weight, bias, stride=stride, padding=padding)

                    # Free previous x immediately
                    if x is not control_image:
                        del x

                    x = x_new

                    # Apply ReLU activation (except for last layer)
                    if layer_name != cond_layers[-1][0]:
                        x = torch.nn.functional.relu(x)

            # Detach and store conditioning embedding (breaks gradient tracking)
            # This is the only tensor we keep
            cond_embeddings[base_name] = x.detach()

            # Free x immediately
            del x

        # Force garbage collection to free intermediate tensors
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[ControlNetManager] Processed {len(cond_embeddings)} conditioning embeddings (only matched modules)")

        # Log conditioning embedding sizes
        # if cond_embeddings:
        #     first_key = next(iter(cond_embeddings))
        #     first_emb = cond_embeddings[first_key]
        #     emb_size_mb = first_emb.element_size() * first_emb.nelement() / 1024**2
        #     total_size_mb = emb_size_mb * len(cond_embeddings)
        #     print(f"[ControlNetManager DEBUG] Each conditioning embedding: {first_emb.shape}, {emb_size_mb:.2f}MB")
        #     print(f"[ControlNetManager DEBUG] Total conditioning embeddings size: {total_size_mb:.2f}MB")

        return cond_embeddings

    def _build_lllite_modules(self, state_dict: dict, device: str, dtype: torch.dtype) -> dict:
        """Build LLLite module structure from state dict

        Returns dict mapping base module names to their submodules.
        Base module: lllite_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q
        Submodules: conditioning1, down, mid, up

        Moves all weights to device ONCE to avoid duplication.
        """
        import torch.nn as nn

        modules = {}

        # Group keys by base module name and move to device immediately
        for key in state_dict.keys():
            # Parse key structure: lllite_unet_..._to_q.conditioning1.0.weight
            # Split by dots to separate base module, submodule, layer, param
            parts = key.split('.')

            # Find the split point (before conditioning1/down/mid/up)
            base_parts = []
            submodule_parts = []
            found_submodule = False

            for part in parts:
                if part in ['conditioning1', 'down', 'mid', 'up']:
                    found_submodule = True
                    submodule_parts.append(part)
                elif found_submodule:
                    submodule_parts.append(part)
                else:
                    base_parts.append(part)

            if not found_submodule:
                continue

            base_name = '.'.join(base_parts)
            submodule_path = '.'.join(submodule_parts[:-1])  # Exclude 'weight'/'bias'
            param_name = submodule_parts[-1]

            if base_name not in modules:
                modules[base_name] = {}

            if submodule_path not in modules[base_name]:
                modules[base_name][submodule_path] = {}

            # Move to device once and store
            # All closures will share these same GPU tensors
            modules[base_name][submodule_path][param_name] = state_dict[key].to(device=device, dtype=dtype)

        print(f"[ControlNetManager] Built {len(modules)} LLLite base modules (moved to {device})")
        return modules

    def _apply_lllite_patches(self, unet, lllite_modules: dict, cond_embeddings: dict,
                              layer_weights=None):
        """Apply LLLite patches to U-Net attention layers

        Patches specific transformer blocks to add LLLite conditioning.
        """
        # Store LLLite info on U-Net for access during forward pass
        if not hasattr(unet, '_lllite_data'):
            unet._lllite_data = []

        lllite_data = {
            'modules': lllite_modules,
            'cond_embeddings': cond_embeddings,
            'layer_weights': layer_weights
        }

        unet._lllite_data.append(lllite_data)

        # Patch U-Net transformer blocks
        self._patch_unet_transformers(unet, lllite_data)

    def _patch_unet_transformers(self, unet, lllite_data: dict):
        """Patch U-Net transformer blocks to apply LLLite

        Wraps the forward method of attention projection layers (to_q, to_k, to_v)
        to add LLLite conditioning.

        Uses cumulative indexing to match kohya-ss convention.
        """
        lllite_modules = lllite_data['modules']
        cond_embeddings = lllite_data['cond_embeddings']

        available_keys = set(lllite_modules.keys())
        # print(f"[ControlNetManager DEBUG] Total available LLLite modules: {len(available_keys)}")

        patched_count = 0
        input_block_idx = 0

        # Initial conv counts as input_blocks_0
        input_block_idx += 1

        # Patch down_blocks (input_blocks)
        # Use hardcoded mapping instead of cumulative indexing
        if hasattr(unet, 'down_blocks'):
            for down_idx, block in enumerate(unet.down_blocks):
                if hasattr(block, 'attentions') and block.attentions is not None:
                    for attn_idx, attention in enumerate(block.attentions):
                        # Use known_mappings from _find_matching_modules
                        kohya_idx = {
                            (1, 0): 4,
                            (2, 0): 7,
                            (2, 1): 8,
                        }.get((down_idx, attn_idx))

                        if kohya_idx is None:
                            continue  # Skip unmapped blocks

                        if hasattr(attention, 'transformer_blocks'):
                            for trans_idx, transformer_block in enumerate(attention.transformer_blocks):
                                # kohya-ss naming: input_blocks_X_1_transformer_blocks_Y
                                block_name = f"input_blocks_{kohya_idx}_1_transformer_blocks_{trans_idx}"

                                # Patch attn1
                                if hasattr(transformer_block, 'attn1'):
                                    patched_count += self._patch_attention_layer_flexible(
                                        transformer_block.attn1,
                                        f"{block_name}_attn1",
                                        lllite_modules, cond_embeddings, available_keys
                                    )

                                # Patch attn2
                                if hasattr(transformer_block, 'attn2'):
                                    patched_count += self._patch_attention_layer_flexible(
                                        transformer_block.attn2,
                                        f"{block_name}_attn2",
                                        lllite_modules, cond_embeddings, available_keys
                                    )

        # Patch mid_block (middle_block)
        if hasattr(unet, 'mid_block') and hasattr(unet.mid_block, 'attentions'):
            for attn_idx, attention in enumerate(unet.mid_block.attentions):
                if hasattr(attention, 'transformer_blocks'):
                    for trans_idx, transformer_block in enumerate(attention.transformer_blocks):
                        # kohya-ss naming: middle_block_1_transformer_blocks_Y
                        block_name = f"middle_block_1_transformer_blocks_{trans_idx}"

                        # Patch attn1
                        if hasattr(transformer_block, 'attn1'):
                            patched_count += self._patch_attention_layer_flexible(
                                transformer_block.attn1,
                                f"{block_name}_attn1",
                                lllite_modules, cond_embeddings, available_keys
                            )

                        # Patch attn2
                        if hasattr(transformer_block, 'attn2'):
                            patched_count += self._patch_attention_layer_flexible(
                                transformer_block.attn2,
                                f"{block_name}_attn2",
                                lllite_modules, cond_embeddings, available_keys
                            )

        # Patch up_blocks (output_blocks)
        output_block_idx = 0
        if hasattr(unet, 'up_blocks'):
            for up_idx, block in enumerate(unet.up_blocks):
                num_resnets = len(block.resnets) if hasattr(block, 'resnets') else 0

                if hasattr(block, 'attentions') and block.attentions is not None:
                    for attn_idx, attention in enumerate(block.attentions):
                        if hasattr(attention, 'transformer_blocks'):
                            for trans_idx, transformer_block in enumerate(attention.transformer_blocks):
                                # kohya-ss naming: output_blocks_X_1_transformer_blocks_Y
                                block_name = f"output_blocks_{output_block_idx}_1_transformer_blocks_{trans_idx}"

                                # Patch attn1
                                if hasattr(transformer_block, 'attn1'):
                                    patched_count += self._patch_attention_layer_flexible(
                                        transformer_block.attn1,
                                        f"{block_name}_attn1",
                                        lllite_modules, cond_embeddings, available_keys
                                    )

                                # Patch attn2
                                if hasattr(transformer_block, 'attn2'):
                                    patched_count += self._patch_attention_layer_flexible(
                                        transformer_block.attn2,
                                        f"{block_name}_attn2",
                                        lllite_modules, cond_embeddings, available_keys
                                    )

                        output_block_idx += 1
                else:
                    output_block_idx += num_resnets

        print(f"[ControlNetManager] Patched {patched_count} attention projections with LLLite")

    def _patch_attention_layer_flexible(self, attention, block_name: str, lllite_modules: dict, cond_embeddings: dict, available_keys: set) -> int:
        """Patch a single attention layer with flexible key matching

        Returns the number of projections patched.
        """
        patched = 0

        for proj_name in ['to_q', 'to_k', 'to_v']:
            if not hasattr(attention, proj_name):
                continue

            proj_layer = getattr(attention, proj_name)

            # Build the full module name for LLLite lookup
            lllite_name = f"lllite_unet_{block_name}_{proj_name}"

            if lllite_name not in lllite_modules:
                continue

            # Get the conditioning embedding and LoRA modules
            cond_emb = cond_embeddings.get(lllite_name)
            modules = lllite_modules[lllite_name]

            if cond_emb is None:
                # print(f"[ControlNetManager DEBUG] No conditioning for {lllite_name}")
                continue

            # Save original forward for cleanup
            original_forward = proj_layer.forward

            # Track this patch for later cleanup
            self.lllite_patched_layers.append((proj_layer, original_forward))

            def create_lllite_forward(orig_forward, cond_embedding, mods, module_name):
                """Create LLLite forward function based on kohya-ss implementation

                cond_embedding: Pre-computed conditioning embedding (B, C, H, W)
                module_name: Name of the LLLite module for debugging
                """
                # Get LoRA-like weights (already on GPU)
                down_weight = mods.get('down.0', {}).get('weight')
                down_bias = mods.get('down.0', {}).get('bias')
                mid_weight = mods.get('mid.0', {}).get('weight')
                mid_bias = mods.get('mid.0', {}).get('bias')
                up_weight = mods.get('up.0', {}).get('weight')
                up_bias = mods.get('up.0', {}).get('bias')

                # Create a counter for logging (only log first call)
                first_call = [True]

                def lllite_forward(hidden_states):
                    x = hidden_states

                    # Apply LLLite if we have all required components
                    if cond_embedding is not None and all([down_weight is not None,
                                                           mid_weight is not None,
                                                           up_weight is not None]):
                        try:
                            # 1. Reshape pre-computed conditioning: (B, C, H, W) -> (B, H*W, C)
                            n, c, h, w = cond_embedding.shape
                            cx = cond_embedding.view(n, c, h * w).permute(0, 2, 1)  # (B, H*W, C)

                            # 2. Apply down projection to hidden_states
                            down_x = torch.nn.functional.linear(x, down_weight, down_bias)

                            # 3. Match spatial dimensions if needed
                            seq_len = x.shape[1]
                            if cx.shape[1] != seq_len:
                                cx = torch.nn.functional.interpolate(
                                    cx.permute(0, 2, 1),  # (B, C, H*W)
                                    size=seq_len,
                                    mode='linear',
                                    align_corners=False
                                ).permute(0, 2, 1)  # (B, seq_len, C)

                            # 4. Match batch size (for CFG, batch size is 2x)
                            batch_size = x.shape[0]
                            if cx.shape[0] != batch_size:
                                # Repeat conditioning for each batch item
                                cx = cx.repeat(batch_size, 1, 1)

                            # 5. Concatenate: [conditioning, down(x)]
                            cx = torch.cat([cx, down_x], dim=2)

                            # 6. Mid layer
                            cx = torch.nn.functional.linear(cx, mid_weight, mid_bias)

                            # 7. Up layer
                            cx = torch.nn.functional.linear(cx, up_weight, up_bias)

                            # 8. Add to input (LoRA-style residual)
                            x = x + cx

                            # Log first successful execution
                            # if first_call[0]:
                            #     print(f"[LLLite Forward DEBUG] {module_name}: Successfully applied, hidden_states={hidden_states.shape}, down_weight={down_weight.shape if down_weight is not None else None}")
                            #     first_call[0] = False

                        except Exception as e:
                            # Log error on first call (keep error logging)
                            if first_call[0]:
                                print(f"[LLLite Forward ERROR] {module_name}: Failed with hidden_states={hidden_states.shape}, down_weight={down_weight.shape if down_weight is not None else None}")
                                print(f"  Error: {e}")
                                first_call[0] = False

                    # Apply original projection
                    output = orig_forward(x)
                    return output

                return lllite_forward

            # Replace the forward method
            proj_layer.forward = create_lllite_forward(original_forward, cond_emb, modules, lllite_name)
            patched += 1

        return patched

    def _patch_attention_layer(self, attention, block_name: str, lllite_modules: dict, cond_embeddings: dict) -> int:
        """Patch a single attention layer's projection layers (to_q, to_k, to_v)

        Returns the number of projections patched.
        """
        patched = 0

        for proj_name in ['to_q', 'to_k', 'to_v']:
            if not hasattr(attention, proj_name):
                continue

            proj_layer = getattr(attention, proj_name)

            # Build the full module name for LLLite lookup
            lllite_name = f"lllite_unet_{block_name}_{proj_name}"

            # Debug: Show what we're looking for vs what exists
            # if patched == 0:  # Only log once to avoid spam
            #     print(f"[ControlNetManager DEBUG] Looking for: {lllite_name}")
            #     sample_keys = list(lllite_modules.keys())[:3]
            #     print(f"[ControlNetManager DEBUG] Sample available keys: {sample_keys}")

            if lllite_name not in lllite_modules:
                continue

            # Get the conditioning embedding and LoRA modules
            cond_emb = cond_embeddings.get(lllite_name)
            modules = lllite_modules[lllite_name]

            # Wrap the projection layer's forward method
            original_forward = proj_layer.forward

            def create_lllite_forward(orig_forward, cond, mods):
                # Get weights (already on correct device from _build_lllite_modules)
                down_weight = mods.get('down.0', {}).get('weight')
                down_bias = mods.get('down.0', {}).get('bias')
                mid_weight = mods.get('mid.0', {}).get('weight')
                mid_bias = mods.get('mid.0', {}).get('bias')
                up_weight = mods.get('up.0', {}).get('weight')
                up_bias = mods.get('up.0', {}).get('bias')

                def lllite_forward(hidden_states):
                    # Call original projection
                    output = orig_forward(hidden_states)

                    # Apply LLLite modification: down -> concat with cond -> mid -> up
                    try:
                        if down_weight is not None:
                            # Reshape hidden_states for linear layer if needed
                            batch, seq_len, channels = hidden_states.shape
                            down_out = torch.nn.functional.linear(hidden_states, down_weight, down_bias)

                            # Prepare conditioning embedding
                            if cond is not None:
                                # Reshape cond to match sequence
                                # cond shape: (batch, cond_channels, height, width)
                                # Need to reshape to (batch, seq_len, cond_channels)
                                cond_reshaped = cond.flatten(2).permute(0, 2, 1)  # (B, H*W, C)

                                # Interpolate or pool to match seq_len
                                if cond_reshaped.shape[1] != seq_len:
                                    cond_reshaped = torch.nn.functional.adaptive_avg_pool1d(
                                        cond_reshaped.permute(0, 2, 1), seq_len
                                    ).permute(0, 2, 1)

                                # Concatenate conditioning with down output
                                mid_input = torch.cat([down_out, cond_reshaped], dim=-1)
                            else:
                                mid_input = down_out

                            # Mid layer: process combined features
                            if mid_weight is not None:
                                mid_out = torch.nn.functional.linear(mid_input, mid_weight, mid_bias)

                                # Up layer: restore original dimensions
                                if up_weight is not None:
                                    up_out = torch.nn.functional.linear(mid_out, up_weight, up_bias)

                                    # Add LLLite modification to original output
                                    output = output + up_out

                    except Exception as e:
                        print(f"[ControlNetManager] Warning: LLLite application failed for {lllite_name}: {e}")
                        pass

                    return output

                return lllite_forward

            # Replace the forward method
            proj_layer.forward = create_lllite_forward(original_forward, cond_emb, modules, lllite_name)
            patched += 1

        return patched

    def prepare_controlnet_image(
        self,
        image: Image.Image,
        width: int,
        height: int
    ) -> torch.Tensor:
        """Prepare ControlNet conditioning image"""
        # Resize image to target dimensions
        image = image.convert("RGB")
        image = image.resize((width, height), Image.LANCZOS)

        # Convert to tensor
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)

        return image_tensor

    def apply_layer_weights(
        self,
        controlnet,
        layer_weights: Dict[str, float]
    ):
        """Apply layer-wise weights to ControlNet

        ControlNet architecture:
        - down_blocks (IN00-IN11): 12 input blocks
        - mid_block (MID): 1 middle block

        Layer weights map like:
        - IN00-IN11 -> down_block_res_samples[0-11]
        - MID -> mid_block_res_sample
        """
        if not layer_weights:
            return

        print(f"[ControlNetManager] Applying layer weights to ControlNet")

        # Convert layer names (IN00, IN01, ..., MID) to indices
        # Store as a list for easier indexing during forward pass
        down_weights = []
        mid_weight = 1.0

        for i in range(12):
            layer_name = f"IN{i:02d}"
            weight = layer_weights.get(layer_name, 1.0)
            down_weights.append(weight)

        mid_weight = layer_weights.get("MID", 1.0)

        # Store weights in a format compatible with ControlNet output
        # ControlNet returns (down_block_res_samples, mid_block_res_sample)
        layer_weight_data = {
            'down': down_weights,  # List of 12 weights
            'mid': mid_weight      # Single weight
        }

        print(f"[ControlNetManager] Layer weights applied: down={down_weights}, mid={mid_weight}")

        # Handle LLLite models (stored as dict) vs standard ControlNet models
        if isinstance(controlnet, dict):
            # LLLite model - store weights in the dict
            controlnet['_layer_weights'] = layer_weight_data
        else:
            # Standard ControlNet model - store as attribute
            controlnet._layer_weights = layer_weight_data
            # Monkey-patch the forward method to apply weights
            self._patch_controlnet_forward(controlnet)

    def _patch_controlnet_forward(self, controlnet: ControlNetModel):
        """Patch ControlNet forward method to apply layer weights"""

        # Check if already patched
        if hasattr(controlnet, '_original_forward'):
            return

        # Save original forward method
        controlnet._original_forward = controlnet.forward

        def weighted_forward(*args, **kwargs):
            # Call original forward
            output = controlnet._original_forward(*args, **kwargs)

            # Apply layer weights if set
            if hasattr(controlnet, '_layer_weights'):
                weights = controlnet._layer_weights

                # Output is either:
                # - tuple: (down_block_res_samples, mid_block_res_sample)
                # - ControlNetOutput object with .down_block_res_samples and .mid_block_res_sample

                if isinstance(output, tuple):
                    down_samples, mid_sample = output
                else:
                    # ControlNetOutput object
                    down_samples = output.down_block_res_samples
                    mid_sample = output.mid_block_res_sample

                # Apply down block weights
                weighted_down_samples = []
                for i, sample in enumerate(down_samples):
                    if i < len(weights['down']):
                        weight = weights['down'][i]
                        weighted_down_samples.append(sample * weight)
                    else:
                        weighted_down_samples.append(sample)

                # Apply mid block weight
                weighted_mid_sample = mid_sample * weights['mid']

                # Return in same format as input
                if isinstance(output, tuple):
                    return (weighted_down_samples, weighted_mid_sample)
                else:
                    # Reconstruct ControlNetOutput
                    output.down_block_res_samples = weighted_down_samples
                    output.mid_block_res_sample = weighted_mid_sample
                    return output

            return output

        # Replace forward method
        controlnet.forward = weighted_forward

    def remove_lllite_patches(self):
        """Remove all LLLite patches and restore original forward methods"""
        if not self.lllite_patched_layers and self.current_lllite_modules is None:
            print(f"[ControlNetManager] No LLLite patches to remove")
            return

        print(f"[ControlNetManager] Removing {len(self.lllite_patched_layers)} LLLite patches")

        # Check VRAM before cleanup
        if torch.cuda.is_available():
            allocated_before = torch.cuda.memory_allocated() / 1024**3
            reserved_before = torch.cuda.memory_reserved() / 1024**3
            print(f"[ControlNetManager VRAM] Before cleanup: Allocated={allocated_before:.2f}GB, Reserved={reserved_before:.2f}GB")

        # Restore original forward methods
        for layer, original_forward in self.lllite_patched_layers:
            layer.forward = original_forward

        self.lllite_patched_layers.clear()

        # Delete cached LLLite data to free GPU memory
        if self.current_lllite_modules is not None:
            print(f"[ControlNetManager] Clearing {len(self.current_lllite_modules)} LLLite modules from GPU")
            for module_name, module_data in self.current_lllite_modules.items():
                for key in list(module_data.keys()):
                    if isinstance(module_data[key], dict):
                        for subkey in list(module_data[key].keys()):
                            del module_data[key][subkey]
                        module_data[key].clear()
            self.current_lllite_modules.clear()
            self.current_lllite_modules = None

        if self.current_lllite_embeddings is not None:
            print(f"[ControlNetManager] Clearing {len(self.current_lllite_embeddings)} conditioning embeddings from GPU")
            self.current_lllite_embeddings.clear()
            self.current_lllite_embeddings = None

        # Force garbage collection and clear CUDA cache
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            allocated_after = torch.cuda.memory_allocated() / 1024**3
            reserved_after = torch.cuda.memory_reserved() / 1024**3
            print(f"[ControlNetManager VRAM] After cleanup: Allocated={allocated_after:.2f}GB, Reserved={reserved_after:.2f}GB")

        print(f"[ControlNetManager] LLLite patches removed")

    def offload_controlnets_to_cpu(self):
        """Move all loaded ControlNet models to CPU to free GPU memory"""
        if not self.loaded_controlnets:
            return

        print(f"[ControlNetManager] Offloading {len(self.loaded_controlnets)} ControlNet(s) to CPU")

        if torch.cuda.is_available():
            allocated_before = torch.cuda.memory_allocated() / 1024**3
            print(f"[ControlNetManager VRAM] Before offload: Allocated={allocated_before:.2f}GB")

        for model_path, controlnet in self.loaded_controlnets.items():
            controlnet.to('cpu')

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            allocated_after = torch.cuda.memory_allocated() / 1024**3
            print(f"[ControlNetManager VRAM] After offload: Allocated={allocated_after:.2f}GB")

        print(f"[ControlNetManager] ControlNets offloaded to CPU")

# Global ControlNet manager instance
controlnet_manager = ControlNetManager()
