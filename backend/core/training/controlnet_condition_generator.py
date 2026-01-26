"""
ControlNet condition image generator.

Uses controlnet-aux preprocessors to automatically generate condition images
from source images for ControlNet training when reference images are not provided.

Supported preprocessors:
- canny: Canny edge detection
- hed: HED edge detection
- lineart: Line art extraction
- lineart_anime: Anime-style line art
- lineart_standard: Standard line art (CPU only, no model download)
- depth_midas: Monocular depth estimation (MiDaS)
- depth_zoe: Monocular depth estimation (ZoeDepth)
- depth_leres: Monocular depth estimation (LeReS)
- normal_bae: Surface normal estimation
- mlsd: Line segment detection (M-LSD)
- openpose: Human pose estimation
- pidi: Soft edge detection (PiDiNet)
- shuffle: Content shuffle
- teed: Thin edge detection (TEED)
- anyline: AnyLine edge detection

Author: Claude (2026-01-26)
"""

import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

# Lazy imports for controlnet_aux processors (loaded on first use)
_PROCESSOR_CACHE: Dict[str, object] = {}


# Mapping from user-friendly names to controlnet_aux classes
PREPROCESSOR_REGISTRY = {
    "canny": {
        "class": "CannyDetector",
        "module": "controlnet_aux",
        "call_kwargs": {},  # Uses default thresholds
    },
    "hed": {
        "class": "HEDdetector",
        "module": "controlnet_aux",
        "from_pretrained": True,
    },
    "lineart": {
        "class": "LineartDetector",
        "module": "controlnet_aux",
        "from_pretrained": True,
    },
    "lineart_anime": {
        "class": "LineartAnimeDetector",
        "module": "controlnet_aux",
        "from_pretrained": True,
    },
    "lineart_standard": {
        "class": "LineartStandardDetector",
        "module": "controlnet_aux",
        "call_kwargs": {},  # CPU only, no model
    },
    "depth_midas": {
        "class": "MidasDetector",
        "module": "controlnet_aux",
        "from_pretrained": True,
    },
    "depth_zoe": {
        "class": "ZoeDetector",
        "module": "controlnet_aux",
        "from_pretrained": True,
    },
    "depth_leres": {
        "class": "LeresDetector",
        "module": "controlnet_aux",
        "from_pretrained": True,
    },
    "normal_bae": {
        "class": "NormalBaeDetector",
        "module": "controlnet_aux",
        "from_pretrained": True,
    },
    "mlsd": {
        "class": "MLSDdetector",
        "module": "controlnet_aux",
        "from_pretrained": True,
    },
    "openpose": {
        "class": "OpenposeDetector",
        "module": "controlnet_aux",
        "from_pretrained": True,
    },
    "pidi": {
        "class": "PidiNetDetector",
        "module": "controlnet_aux",
        "from_pretrained": True,
    },
    "shuffle": {
        "class": "ContentShuffleDetector",
        "module": "controlnet_aux",
        "call_kwargs": {},
    },
    "teed": {
        "class": "TEEDdetector",
        "module": "controlnet_aux",
        "from_pretrained": True,
    },
    "anyline": {
        "class": "AnylineDetector",
        "module": "controlnet_aux",
        "from_pretrained": True,
    },
}


def get_available_preprocessors() -> List[str]:
    """Return list of available preprocessor names."""
    return sorted(PREPROCESSOR_REGISTRY.keys())


def _load_processor(preprocessor_type: str) -> object:
    """
    Load and cache a controlnet_aux preprocessor.

    Args:
        preprocessor_type: Name of the preprocessor (e.g., "canny", "hed")

    Returns:
        Loaded preprocessor instance
    """
    if preprocessor_type in _PROCESSOR_CACHE:
        return _PROCESSOR_CACHE[preprocessor_type]

    if preprocessor_type not in PREPROCESSOR_REGISTRY:
        raise ValueError(
            f"Unknown preprocessor type: '{preprocessor_type}'. "
            f"Available types: {', '.join(get_available_preprocessors())}"
        )

    config = PREPROCESSOR_REGISTRY[preprocessor_type]

    import importlib
    module = importlib.import_module(config["module"])
    cls = getattr(module, config["class"])

    if config.get("from_pretrained"):
        processor = cls.from_pretrained("lllyasviel/Annotators")
    else:
        processor = cls()

    _PROCESSOR_CACHE[preprocessor_type] = processor
    print(f"[ConditionGenerator] Loaded preprocessor: {preprocessor_type}")

    return processor


class ControlNetConditionGenerator:
    """
    Generates condition images for ControlNet training using controlnet-aux.

    Supports multiple preprocessor types with random selection per image.
    This allows training a ControlNet that generalizes across condition types.

    Usage:
        generator = ControlNetConditionGenerator(["canny", "hed", "lineart"])

        # Generate condition for a single image
        condition = generator.generate_condition("path/to/image.jpg", 512, 512)

        # Pre-generate all conditions for a dataset
        generator.pre_generate_all(dataset_items, cache_dir)
    """

    def __init__(
        self,
        preprocessor_types: List[str],
        random_select: bool = True,
    ):
        """
        Initialize condition generator.

        Args:
            preprocessor_types: List of preprocessor types to use
            random_select: If True, randomly select preprocessor per image.
                          If False, use the first preprocessor for all images.
        """
        if not preprocessor_types:
            raise ValueError("At least one preprocessor type must be specified")

        # Validate preprocessor types
        for ptype in preprocessor_types:
            if ptype not in PREPROCESSOR_REGISTRY:
                raise ValueError(
                    f"Unknown preprocessor type: '{ptype}'. "
                    f"Available types: {', '.join(get_available_preprocessors())}"
                )

        self.preprocessor_types = preprocessor_types
        self.random_select = random_select

        print(f"[ConditionGenerator] Initialized with preprocessors: {preprocessor_types}")

    def generate_condition(
        self,
        image_path: str,
        width: int,
        height: int,
        preprocessor_type: Optional[str] = None,
    ) -> Image.Image:
        """
        Generate a condition image from a source image.

        Args:
            image_path: Path to source image
            width: Target width
            height: Target height
            preprocessor_type: Specific preprocessor to use (overrides random selection)

        Returns:
            Condition image as PIL Image (RGB, [0, 255])
        """
        # Select preprocessor
        if preprocessor_type is not None:
            ptype = preprocessor_type
        elif self.random_select and len(self.preprocessor_types) > 1:
            ptype = random.choice(self.preprocessor_types)
        else:
            ptype = self.preprocessor_types[0]

        # Load source image
        source_image = Image.open(image_path).convert("RGB")

        # Resize to target dimensions
        source_image = source_image.resize((width, height), Image.LANCZOS)

        # Load processor
        processor = _load_processor(ptype)

        # Generate condition
        config = PREPROCESSOR_REGISTRY[ptype]
        call_kwargs = config.get("call_kwargs", {})

        # Call processor
        condition = processor(source_image, **call_kwargs)

        # Ensure output is PIL Image
        if isinstance(condition, np.ndarray):
            condition = Image.fromarray(condition)

        # Ensure RGB
        condition = condition.convert("RGB")

        # Ensure correct size
        if condition.size != (width, height):
            condition = condition.resize((width, height), Image.LANCZOS)

        return condition

    def pre_generate_all(
        self,
        items: List[Dict],
        cache_dir: str,
        width: int = 512,
        height: int = 512,
    ) -> Dict[str, str]:
        """
        Pre-generate condition images for all dataset items.

        Args:
            items: List of dataset items with "image_path" key
            cache_dir: Directory to save generated condition images
            width: Target width for condition images
            height: Target height for condition images

        Returns:
            Dict mapping image_path to generated condition_path
        """
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        condition_map = {}
        total = len(items)

        print(f"[ConditionGenerator] Pre-generating {total} condition images...")

        for i, item in enumerate(items):
            image_path = item.get("image_path", "")
            if not image_path:
                continue

            # Generate cache filename
            source_name = Path(image_path).stem
            condition_filename = f"{source_name}_condition.png"
            condition_path = cache_path / condition_filename

            # Skip if already cached
            if condition_path.exists():
                condition_map[image_path] = str(condition_path)
                continue

            try:
                condition = self.generate_condition(image_path, width, height)
                condition.save(str(condition_path))
                condition_map[image_path] = str(condition_path)
            except Exception as e:
                print(f"[ConditionGenerator] Error processing {image_path}: {e}")
                continue

            if (i + 1) % 100 == 0 or (i + 1) == total:
                print(f"[ConditionGenerator] Progress: {i + 1}/{total}")

        print(f"[ConditionGenerator] Pre-generation complete: {len(condition_map)}/{total} images")
        return condition_map

    def cleanup(self):
        """Release loaded preprocessor models from memory."""
        global _PROCESSOR_CACHE
        for ptype in list(_PROCESSOR_CACHE.keys()):
            del _PROCESSOR_CACHE[ptype]
        _PROCESSOR_CACHE.clear()

        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[ConditionGenerator] Cleaned up preprocessor models")
