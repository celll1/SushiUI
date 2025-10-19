"""
ControlNet Preprocessor Manager

Supports various preprocessors for ControlNet models:
- Canny edge detection
- Depth estimation (Midas, Zoe, Leres)
- OpenPose detection
- Normal map estimation
- Soft edge detection (HED, PIDI)
- Lineart extraction
- Segmentation
- MLSD line detection
"""

import cv2
import numpy as np
from PIL import Image
from typing import Optional, Literal
import torch

# Preprocessor type definitions
PreprocessorType = Literal[
    "none",
    "canny",
    "depth_midas",
    "depth_zoe",
    "depth_leres",
    "openpose",
    "openpose_hand",
    "openpose_face",
    "openpose_full",
    "normal_bae",
    "softedge_hed",
    "softedge_pidi",
    "lineart",
    "lineart_anime",
    "segment_ofade20k",
    "mlsd",
    "tile",
    "blur"
]


class ControlNetPreprocessor:
    """Handles preprocessing of images for ControlNet models"""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.loaded_preprocessors = {}

    def detect_preprocessor_from_model_name(self, model_path: str) -> Optional[PreprocessorType]:
        """Detect which preprocessor to use based on ControlNet model name

        Examples:
        - control_v11p_sd15_canny -> canny
        - control_v11f1p_sd15_depth -> depth_midas
        - control_v11p_sd15_openpose -> openpose
        - control_v11p_sd15_normalbae -> normal_bae
        - control_v11p_sd15_softedge -> softedge_hed
        - control_v11p_sd15_lineart -> lineart
        - control_v11p_sd15s2_lineart_anime -> lineart_anime
        - control_v11f1e_sd15_tile -> tile
        """
        model_name_lower = model_path.lower()

        # Check for specific model types
        if "canny" in model_name_lower:
            return "canny"
        elif "depth" in model_name_lower:
            return "depth_midas"  # Default depth preprocessor
        elif "openpose" in model_name_lower:
            return "openpose"
        elif "normal" in model_name_lower:
            return "normal_bae"
        elif "softedge" in model_name_lower or "hed" in model_name_lower or "pidi" in model_name_lower:
            return "softedge_hed"
        elif "lineart_anime" in model_name_lower or "anime" in model_name_lower:
            return "lineart_anime"
        elif "lineart" in model_name_lower:
            return "lineart"
        elif "seg" in model_name_lower or "segment" in model_name_lower:
            return "segment_ofade20k"
        elif "mlsd" in model_name_lower:
            return "mlsd"
        elif "tile" in model_name_lower:
            return "tile"
        elif "blur" in model_name_lower:
            return "blur"

        # Default: no preprocessing
        return "none"

    def preprocess(
        self,
        image: Image.Image,
        preprocessor_type: PreprocessorType,
        **kwargs
    ) -> Image.Image:
        """Apply preprocessing to image

        Args:
            image: Input PIL Image
            preprocessor_type: Type of preprocessing to apply
            **kwargs: Additional parameters for specific preprocessors
                - canny: low_threshold (default: 100), high_threshold (default: 200)
                - depth: None required
                - openpose: include_hand (default: False), include_face (default: False)

        Returns:
            Preprocessed PIL Image
        """
        if preprocessor_type == "none":
            return image

        # Convert PIL to numpy
        image_np = np.array(image)

        # Apply appropriate preprocessor
        if preprocessor_type == "canny":
            result = self._preprocess_canny(image_np, **kwargs)
        elif preprocessor_type.startswith("depth"):
            result = self._preprocess_depth(image_np, preprocessor_type, **kwargs)
        elif preprocessor_type.startswith("openpose"):
            result = self._preprocess_openpose(image_np, preprocessor_type, **kwargs)
        elif preprocessor_type == "normal_bae":
            result = self._preprocess_normal(image_np, **kwargs)
        elif preprocessor_type.startswith("softedge"):
            result = self._preprocess_softedge(image_np, preprocessor_type, **kwargs)
        elif preprocessor_type.startswith("lineart"):
            result = self._preprocess_lineart(image_np, preprocessor_type, **kwargs)
        elif preprocessor_type.startswith("segment"):
            result = self._preprocess_segment(image_np, **kwargs)
        elif preprocessor_type == "mlsd":
            result = self._preprocess_mlsd(image_np, **kwargs)
        elif preprocessor_type == "tile":
            result = image_np  # No preprocessing for tile
        elif preprocessor_type == "blur":
            result = self._preprocess_blur(image_np, **kwargs)
        else:
            print(f"[Preprocessor] Unknown preprocessor type: {preprocessor_type}, returning original image")
            result = image_np

        # Convert back to PIL
        return Image.fromarray(result)

    def _preprocess_canny(self, image_np: np.ndarray, low_threshold: int = 100, high_threshold: int = 200) -> np.ndarray:
        """Apply Canny edge detection"""
        # Convert to grayscale if needed
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np

        # Apply Canny edge detection
        edges = cv2.Canny(gray, low_threshold, high_threshold)

        # Convert back to 3-channel for ControlNet
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        return edges_rgb

    def _preprocess_depth(self, image_np: np.ndarray, depth_type: str, **kwargs) -> np.ndarray:
        """Apply depth estimation

        Requires controlnet_aux library:
        pip install controlnet-aux
        """
        try:
            from controlnet_aux import MidasDetector, ZoeDetector, LeresDetector

            # Load appropriate depth detector
            if depth_type == "depth_midas" or depth_type == "depth":
                if "depth_midas" not in self.loaded_preprocessors:
                    print("[Preprocessor] Loading Midas depth detector...")
                    self.loaded_preprocessors["depth_midas"] = MidasDetector.from_pretrained("lllyasviel/Annotators")
                detector = self.loaded_preprocessors["depth_midas"]
            elif depth_type == "depth_zoe":
                if "depth_zoe" not in self.loaded_preprocessors:
                    print("[Preprocessor] Loading Zoe depth detector...")
                    self.loaded_preprocessors["depth_zoe"] = ZoeDetector.from_pretrained("lllyasviel/Annotators")
                detector = self.loaded_preprocessors["depth_zoe"]
            elif depth_type == "depth_leres":
                if "depth_leres" not in self.loaded_preprocessors:
                    print("[Preprocessor] Loading Leres depth detector...")
                    self.loaded_preprocessors["depth_leres"] = LeresDetector.from_pretrained("lllyasviel/Annotators")
                detector = self.loaded_preprocessors["depth_leres"]
            else:
                print(f"[Preprocessor] Unknown depth type: {depth_type}, using Midas")
                if "depth_midas" not in self.loaded_preprocessors:
                    self.loaded_preprocessors["depth_midas"] = MidasDetector.from_pretrained("lllyasviel/Annotators")
                detector = self.loaded_preprocessors["depth_midas"]

            # Convert numpy to PIL for detector
            image_pil = Image.fromarray(image_np)
            depth_map = detector(image_pil)
            return np.array(depth_map)

        except ImportError:
            print("[Preprocessor] controlnet-aux not installed, falling back to simple grayscale for depth")
            # Simple fallback: convert to grayscale as pseudo-depth
            if len(image_np.shape) == 3:
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            return image_np

    def _preprocess_openpose(self, image_np: np.ndarray, pose_type: str, **kwargs) -> np.ndarray:
        """Apply OpenPose detection

        Requires controlnet_aux library
        """
        try:
            from controlnet_aux import OpenposeDetector

            if "openpose" not in self.loaded_preprocessors:
                print("[Preprocessor] Loading OpenPose detector...")
                self.loaded_preprocessors["openpose"] = OpenposeDetector.from_pretrained("lllyasviel/Annotators")

            detector = self.loaded_preprocessors["openpose"]

            # Determine detection mode
            include_hand = "hand" in pose_type or "full" in pose_type or kwargs.get("include_hand", False)
            include_face = "face" in pose_type or "full" in pose_type or kwargs.get("include_face", False)

            # Convert numpy to PIL
            image_pil = Image.fromarray(image_np)
            pose_map = detector(image_pil, hand_and_face=include_hand or include_face)
            return np.array(pose_map)

        except ImportError:
            print("[Preprocessor] controlnet-aux not installed, returning original image")
            return image_np

    def _preprocess_normal(self, image_np: np.ndarray, **kwargs) -> np.ndarray:
        """Apply normal map estimation

        Requires controlnet_aux library
        """
        try:
            from controlnet_aux import NormalBaeDetector

            if "normal_bae" not in self.loaded_preprocessors:
                print("[Preprocessor] Loading Normal BAE detector...")
                self.loaded_preprocessors["normal_bae"] = NormalBaeDetector.from_pretrained("lllyasviel/Annotators")

            detector = self.loaded_preprocessors["normal_bae"]
            image_pil = Image.fromarray(image_np)
            normal_map = detector(image_pil)
            return np.array(normal_map)

        except ImportError:
            print("[Preprocessor] controlnet-aux not installed, returning original image")
            return image_np

    def _preprocess_softedge(self, image_np: np.ndarray, edge_type: str, **kwargs) -> np.ndarray:
        """Apply soft edge detection (HED or PIDI)

        Requires controlnet_aux library
        """
        try:
            from controlnet_aux import HEDdetector, PidiNetDetector

            if "hed" in edge_type:
                if "softedge_hed" not in self.loaded_preprocessors:
                    print("[Preprocessor] Loading HED detector...")
                    self.loaded_preprocessors["softedge_hed"] = HEDdetector.from_pretrained("lllyasviel/Annotators")
                detector = self.loaded_preprocessors["softedge_hed"]
            else:  # PIDI
                if "softedge_pidi" not in self.loaded_preprocessors:
                    print("[Preprocessor] Loading PIDI detector...")
                    self.loaded_preprocessors["softedge_pidi"] = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
                detector = self.loaded_preprocessors["softedge_pidi"]

            image_pil = Image.fromarray(image_np)
            edge_map = detector(image_pil)
            return np.array(edge_map)

        except ImportError:
            print("[Preprocessor] controlnet-aux not installed, falling back to Canny")
            return self._preprocess_canny(image_np)

    def _preprocess_lineart(self, image_np: np.ndarray, lineart_type: str, **kwargs) -> np.ndarray:
        """Apply lineart extraction

        Requires controlnet_aux library
        """
        try:
            from controlnet_aux import LineartDetector, LineartAnimeDetector

            if "anime" in lineart_type:
                if "lineart_anime" not in self.loaded_preprocessors:
                    print("[Preprocessor] Loading Anime Lineart detector...")
                    self.loaded_preprocessors["lineart_anime"] = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
                detector = self.loaded_preprocessors["lineart_anime"]
            else:
                if "lineart" not in self.loaded_preprocessors:
                    print("[Preprocessor] Loading Lineart detector...")
                    self.loaded_preprocessors["lineart"] = LineartDetector.from_pretrained("lllyasviel/Annotators")
                detector = self.loaded_preprocessors["lineart"]

            image_pil = Image.fromarray(image_np)
            lineart_map = detector(image_pil)
            return np.array(lineart_map)

        except ImportError:
            print("[Preprocessor] controlnet-aux not installed, falling back to Canny")
            return self._preprocess_canny(image_np)

    def _preprocess_segment(self, image_np: np.ndarray, **kwargs) -> np.ndarray:
        """Apply segmentation

        Requires controlnet_aux library
        """
        try:
            from controlnet_aux import SamDetector

            if "segment" not in self.loaded_preprocessors:
                print("[Preprocessor] Loading Segmentation detector...")
                self.loaded_preprocessors["segment"] = SamDetector.from_pretrained("ybelkada/segment-anything", subfolder="checkpoints")

            detector = self.loaded_preprocessors["segment"]
            image_pil = Image.fromarray(image_np)
            segment_map = detector(image_pil)
            return np.array(segment_map)

        except ImportError:
            print("[Preprocessor] controlnet-aux not installed, returning original image")
            return image_np

    def _preprocess_mlsd(self, image_np: np.ndarray, **kwargs) -> np.ndarray:
        """Apply MLSD line detection

        Requires controlnet_aux library
        """
        try:
            from controlnet_aux import MLSDdetector

            if "mlsd" not in self.loaded_preprocessors:
                print("[Preprocessor] Loading MLSD detector...")
                self.loaded_preprocessors["mlsd"] = MLSDdetector.from_pretrained("lllyasviel/Annotators")

            detector = self.loaded_preprocessors["mlsd"]
            image_pil = Image.fromarray(image_np)
            mlsd_map = detector(image_pil)
            return np.array(mlsd_map)

        except ImportError:
            print("[Preprocessor] controlnet-aux not installed, falling back to Canny")
            return self._preprocess_canny(image_np)

    def _preprocess_blur(self, image_np: np.ndarray, kernel_size: int = 15, **kwargs) -> np.ndarray:
        """Apply Gaussian blur (for tile/blur models)"""
        return cv2.GaussianBlur(image_np, (kernel_size, kernel_size), 0)


# Global preprocessor instance
controlnet_preprocessor = ControlNetPreprocessor()
