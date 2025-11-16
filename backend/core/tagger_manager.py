"""
Image Tagger Manager using ONNX models

Based on cl_tagger: https://huggingface.co/cella110n/cl_tagger
"""

import os
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional
import onnxruntime as ort
import json


class TaggerManager:
    """Manages image tagging using ONNX models"""

    def __init__(self):
        self.session = None
        self.labels = None
        self.idx_to_tag = None
        self.tag_to_category = None
        self.model_path = None
        self.tag_mapping_path = None
        self.model_version = None
        self.loaded = False

    def _download_from_huggingface(
        self,
        repo_id: str = "cella110n/cl_tagger",
        model_version: str = "cl_tagger_1_02"
    ) -> Tuple[str, str]:
        """Download model files from Hugging Face Hub

        Args:
            repo_id: Hugging Face repository ID
            model_version: Model version subdirectory (e.g., cl_tagger_1_02)

        Returns:
            Tuple of (model_path, tag_mapping_path)
        """
        try:
            from huggingface_hub import hf_hub_download

            print(f"[Tagger] Downloading model from Hugging Face: {repo_id}/{model_version}")

            # Download model from subdirectory
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=f"{model_version}/model.onnx",
                cache_dir=None  # Use default cache directory
            )
            print(f"[Tagger] Model downloaded to: {model_path}")

            # Download tag mapping from subdirectory
            tag_mapping_path = hf_hub_download(
                repo_id=repo_id,
                filename=f"{model_version}/tag_mapping.json",
                cache_dir=None
            )
            print(f"[Tagger] Tag mapping downloaded to: {tag_mapping_path}")

            return model_path, tag_mapping_path

        except ImportError:
            raise RuntimeError("huggingface_hub is not installed. Please install it with: pip install huggingface_hub")
        except Exception as e:
            raise RuntimeError(f"Failed to download from Hugging Face: {e}")

    def load_model(
        self,
        model_path: str = None,
        tag_mapping_path: str = None,
        use_gpu: bool = True,
        use_huggingface: bool = True,
        repo_id: str = "cella110n/cl_tagger",
        model_version: str = "cl_tagger_1_02"
    ):
        """Load ONNX tagger model

        Args:
            model_path: Path to ONNX model file (optional if use_huggingface=True)
            tag_mapping_path: Path to tag mapping JSON file (optional if use_huggingface=True)
            use_gpu: Whether to use GPU acceleration
            use_huggingface: Whether to download from Hugging Face Hub
            repo_id: Hugging Face repository ID (default: cella110n/cl_tagger)
            model_version: Model version subdirectory (default: cl_tagger_1_02)
        """
        try:
            # Download from Hugging Face if paths not provided
            if use_huggingface and (model_path is None or tag_mapping_path is None):
                model_path, tag_mapping_path = self._download_from_huggingface(repo_id, model_version)

            if model_path is None or tag_mapping_path is None:
                raise ValueError("model_path and tag_mapping_path must be provided or use_huggingface must be True")

            print(f"[Tagger] Loading ONNX model: {model_path}")

            # Check if model is FP16
            is_fp16_model = False
            try:
                import onnx
                model = onnx.load(model_path)
                for tensor in model.graph.initializer:
                    if tensor.data_type == 10:  # FLOAT16
                        is_fp16_model = True
                        break
                print(f"[Tagger] Model is {'FP16' if is_fp16_model else 'FP32'}")
            except Exception as e:
                print(f"[Tagger] Failed to check model precision: {e}")

            # Setup providers
            available_providers = ort.get_available_providers()
            print(f"[Tagger] Available providers: {available_providers}")

            if use_gpu:
                providers = []
                if 'CUDAExecutionProvider' in available_providers:
                    if is_fp16_model:
                        cuda_options = {
                            'device_id': 0,
                            'arena_extend_strategy': 'kNextPowerOfTwo',
                            'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                            'cudnn_conv_algo_search': 'EXHAUSTIVE',
                            'do_copy_in_default_stream': True,
                        }
                        providers.append(('CUDAExecutionProvider', cuda_options))
                    else:
                        providers.append('CUDAExecutionProvider')
                elif 'DmlExecutionProvider' in available_providers:
                    providers.append('DmlExecutionProvider')

                providers.append('CPUExecutionProvider')

                if len(providers) > 1:
                    print(f"[Tagger] Using providers: {providers}")
                    self.session = ort.InferenceSession(model_path, providers=providers)
                    print(f"[Tagger] Active provider: {self.session.get_providers()[0]}")
                else:
                    print("[Tagger] No GPU providers available, using CPU")
                    self.session = ort.InferenceSession(model_path)
            else:
                self.session = ort.InferenceSession(model_path)
                print("[Tagger] Using CPU for inference")

            # Load tag mapping
            print(f"[Tagger] Loading tag mapping: {tag_mapping_path}")
            self.labels, self.idx_to_tag, self.tag_to_category = self._load_tag_mapping(tag_mapping_path)

            self.model_path = model_path
            self.tag_mapping_path = tag_mapping_path
            self.model_version = model_version
            self.loaded = True
            print(f"[Tagger] Model loaded successfully (version: {model_version})")

        except Exception as e:
            print(f"[Tagger] Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            self.loaded = False

    def _load_tag_mapping(self, mapping_path: str) -> Tuple:
        """Load tag mapping from JSON file"""
        with open(mapping_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Parse the new format: {"0": {"tag": "...", "category": "..."}, ...}
        names = []
        rating_indices = []
        general_indices = []
        artist_indices = []
        character_indices = []
        copyright_indices = []
        meta_indices = []
        quality_indices = []
        model_indices = []

        # Category name mapping (case-insensitive)
        category_map = {
            'rating': 'rating',
            'general': 'general',
            'artist': 'artist',
            'character': 'character',
            'copyright': 'copyright',
            'meta': 'meta',
            'quality': 'quality',
            'model': 'model'
        }

        # Sort by index to ensure correct order
        for idx_str in sorted(data.keys(), key=int):
            idx = int(idx_str)
            item = data[idx_str]
            tag = item['tag']
            category = item['category'].lower()

            names.append(tag)

            # Map to category lists
            if category == 'rating':
                rating_indices.append(idx)
            elif category == 'general':
                general_indices.append(idx)
            elif category == 'artist':
                artist_indices.append(idx)
            elif category == 'character':
                character_indices.append(idx)
            elif category == 'copyright':
                copyright_indices.append(idx)
            elif category == 'meta':
                meta_indices.append(idx)
            elif category == 'quality':
                quality_indices.append(idx)
            elif category == 'model':
                model_indices.append(idx)

        # Create LabelData structure
        class LabelData:
            def __init__(self, names, rating, general, artist, character, copyright, meta, quality, model):
                self.names = names
                self.rating = rating
                self.general = general
                self.artist = artist
                self.character = character
                self.copyright = copyright
                self.meta = meta
                self.quality = quality
                self.model = model

        labels = LabelData(
            names,
            np.array(rating_indices, dtype=np.int64),
            np.array(general_indices, dtype=np.int64),
            np.array(artist_indices, dtype=np.int64),
            np.array(character_indices, dtype=np.int64),
            np.array(copyright_indices, dtype=np.int64),
            np.array(meta_indices, dtype=np.int64),
            np.array(quality_indices, dtype=np.int64),
            np.array(model_indices, dtype=np.int64)
        )

        # Create idx_to_tag mapping
        idx_to_tag = {i: tag for i, tag in enumerate(labels.names)}

        # Create tag_to_category mapping
        tag_to_category = {}
        for idx in labels.rating:
            tag_to_category[labels.names[idx]] = "rating"
        for idx in labels.general:
            tag_to_category[labels.names[idx]] = "general"
        for idx in labels.artist:
            tag_to_category[labels.names[idx]] = "artist"
        for idx in labels.character:
            tag_to_category[labels.names[idx]] = "character"
        for idx in labels.copyright:
            tag_to_category[labels.names[idx]] = "copyright"
        for idx in labels.meta:
            tag_to_category[labels.names[idx]] = "meta"
        for idx in labels.quality:
            tag_to_category[labels.names[idx]] = "quality"
        for idx in labels.model:
            tag_to_category[labels.names[idx]] = "model"

        return labels, idx_to_tag, tag_to_category

    def _ensure_rgb(self, image: Image.Image) -> Image.Image:
        """Convert image to RGB format"""
        if image.mode not in ["RGB", "RGBA"]:
            image = image.convert("RGBA") if "transparency" in image.info else image.convert("RGB")

        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background

        return image

    def _pad_square(self, image: Image.Image) -> Image.Image:
        """Pad image to square"""
        width, height = image.size
        if width == height:
            return image

        new_size = max(width, height)
        new_image = Image.new(image.mode, (new_size, new_size), (255, 255, 255))
        paste_position = ((new_size - width) // 2, (new_size - height) // 2)
        new_image.paste(image, paste_position)

        return new_image

    def _preprocess_image(self, image: Image.Image, target_size: Tuple[int, int] = (448, 448)) -> np.ndarray:
        """Preprocess image for model input"""
        # Ensure RGB
        image = self._ensure_rgb(image)

        # Pad to square
        image = self._pad_square(image)

        # Resize
        image = image.resize(target_size, Image.BICUBIC)

        # Convert to numpy array (HWC format, RGB)
        image_array = np.array(image, dtype=np.float32) / 255.0

        # Convert to CHW format
        image_array = image_array.transpose(2, 0, 1)  # HWC to CHW

        # Convert RGB to BGR (model expects BGR)
        image_array = image_array[::-1, :, :]

        # Normalize with mean=0.5, std=0.5
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
        image_array = (image_array - mean) / std

        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)

        return image_array

    def _stable_sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid function"""
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )

    def _normalize_tag(self, tag: str) -> str:
        """Normalize tag format: underscores to spaces, unescape parentheses"""
        # Replace underscores with spaces
        tag = tag.replace('_', ' ')
        # Unescape parentheses: \( -> (, \) -> )
        tag = tag.replace('\\(', '(').replace('\\)', ')')
        return tag

    def _get_tags(
        self,
        probs: np.ndarray,
        gen_threshold: float = 0.45,
        char_threshold: float = 0.45
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Extract tags from probabilities"""
        results = {
            "rating": [],
            "general": [],
            "artist": [],
            "character": [],
            "copyright": [],
            "meta": [],
            "quality": [],
            "model": []
        }

        # Rating tags - select only the highest confidence tag
        if len(self.labels.rating) > 0:
            rating_probs = [(idx, probs[idx]) for idx in self.labels.rating]
            max_idx, max_prob = max(rating_probs, key=lambda x: x[1])
            normalized_tag = self._normalize_tag(self.labels.names[max_idx])
            results["rating"].append((normalized_tag, float(max_prob)))

        # Quality tags - select only the highest confidence tag
        if len(self.labels.quality) > 0:
            quality_probs = [(idx, probs[idx]) for idx in self.labels.quality]
            max_idx, max_prob = max(quality_probs, key=lambda x: x[1])
            normalized_tag = self._normalize_tag(self.labels.names[max_idx])
            results["quality"].append((normalized_tag, float(max_prob)))

        # Character tags
        for idx in self.labels.character:
            if probs[idx] >= char_threshold:
                normalized_tag = self._normalize_tag(self.labels.names[idx])
                results["character"].append((normalized_tag, float(probs[idx])))
        results["character"].sort(key=lambda x: x[1], reverse=True)

        # General tags
        for idx in self.labels.general:
            if probs[idx] >= gen_threshold:
                normalized_tag = self._normalize_tag(self.labels.names[idx])
                results["general"].append((normalized_tag, float(probs[idx])))
        results["general"].sort(key=lambda x: x[1], reverse=True)

        # Copyright tags
        for idx in self.labels.copyright:
            if probs[idx] >= char_threshold:
                normalized_tag = self._normalize_tag(self.labels.names[idx])
                results["copyright"].append((normalized_tag, float(probs[idx])))
        results["copyright"].sort(key=lambda x: x[1], reverse=True)

        # Artist tags
        for idx in self.labels.artist:
            if probs[idx] >= char_threshold:
                normalized_tag = self._normalize_tag(self.labels.names[idx])
                results["artist"].append((normalized_tag, float(probs[idx])))
        results["artist"].sort(key=lambda x: x[1], reverse=True)

        # Meta tags
        for idx in self.labels.meta:
            if probs[idx] >= gen_threshold:
                normalized_tag = self._normalize_tag(self.labels.names[idx])
                results["meta"].append((normalized_tag, float(probs[idx])))
        results["meta"].sort(key=lambda x: x[1], reverse=True)

        # Model tags
        for idx in self.labels.model:
            if probs[idx] >= gen_threshold:
                normalized_tag = self._normalize_tag(self.labels.names[idx])
                results["model"].append((normalized_tag, float(probs[idx])))
        results["model"].sort(key=lambda x: x[1], reverse=True)

        return results

    def predict(
        self,
        image: Image.Image,
        gen_threshold: float = 0.45,
        char_threshold: float = 0.45,
        model_version: str = "cl_tagger_1_02",
        auto_unload: bool = True
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Predict tags for an image

        Args:
            image: PIL Image object
            gen_threshold: Threshold for general tags
            char_threshold: Threshold for character/copyright/artist tags
            model_version: Model version to use (loads if not already loaded)
            auto_unload: Whether to unload model after prediction to free VRAM

        Returns:
            Dictionary with categorized tags and confidences
        """
        # Auto-load model if not loaded or different version
        if not self.loaded or self.model_version != model_version:
            print(f"[Tagger] Auto-loading model version: {model_version}")
            self.load_model(
                use_huggingface=True,
                repo_id="cella110n/cl_tagger",
                model_version=model_version
            )

        # Preprocess image
        input_data = self._preprocess_image(image)

        # Check expected input type
        expected_input_type = self.session.get_inputs()[0].type
        if "float16" in expected_input_type:
            input_data = input_data.astype(np.float16)
        else:
            input_data = input_data.astype(np.float32)

        # Run inference
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name

        outputs = self.session.run([output_name], {input_name: input_data})[0]

        # Handle NaN and inf
        outputs = np.nan_to_num(outputs, nan=0.0, posinf=100.0, neginf=-100.0)

        # Apply sigmoid
        outputs = self._stable_sigmoid(outputs)

        # Get tags
        predictions = self._get_tags(outputs[0], gen_threshold, char_threshold)

        # Auto-unload to free VRAM
        if auto_unload:
            print("[Tagger] Auto-unloading model to free VRAM")
            self.unload_model()

        return predictions

    def unload_model(self):
        """Unload model from memory"""
        if self.session is not None:
            self.session = None
        self.loaded = False
        print("[Tagger] Model unloaded")


# Global tagger manager instance
tagger_manager = TaggerManager()
