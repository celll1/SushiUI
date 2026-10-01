"""Caption inference borrowing the tagger encoder without duplicating it."""

import hashlib
import io
import threading
import time

from PIL import Image, ImageOps
import torch

from .artifact import load_bundle, tagger_identity
from .features import SigLIP2FeatureProvider
from .storage import canonical_hash


class CaptionInferenceManager:
    def __init__(self, tagger_manager):
        self.tagger = tagger_manager
        self._lock = threading.RLock()
        self.model = None
        self.manifest = None
        self.path = None

    def load(self, path):
        with self.tagger.borrow_caption_tagger(), self._lock:
            model, tokenizer, selector, manifest, _ = load_bundle(path)
            identity = tagger_identity(self.tagger, max_num_patches=manifest["identity"]["max_num_patches"])
            if canonical_hash(identity) != canonical_hash(manifest["identity"]):
                raise ValueError("Tagger weights, vocabulary, or processor do not match captioner")
            self.model, self.tokenizer, self.selector = model, tokenizer, selector
            self.manifest, self.path = manifest, str(path)
            return self.status()

    def unload(self):
        with self._lock:
            self.model = self.manifest = self.path = None
            self.tokenizer = self.selector = None

    def status(self):
        with self._lock:
            if self.model is None:
                return {"loaded": False}
            return {"loaded": True, "path": self.path, "backend": "siglip2",
                    "language": self.manifest["language"], "max_new_tokens": self.model.config.max_length,
                    "max_num_patches": self.manifest["identity"]["max_num_patches"],
                    "max_concepts": self.selector.max_concepts,
                    "vocabulary_hash": self.selector.vocabulary_hash,
                    "tagger_weights_hash": self.manifest["identity"]["weights_hash"],
                    "supported_sources": ["predicted", "provided", "merged"]}

    def predict(self, image_bytes, *, tag_source, tags, max_new_tokens, device, cancelled):
        started = time.perf_counter()
        with self.tagger.borrow_caption_tagger(), self._lock:
            if self.model is None:
                raise ValueError("Load a captioner bundle first")
            if device not in {"cpu", "cuda:0"}:
                raise ValueError("Captioner supports cpu or cuda:0")
            parameter = next(self.tagger.model.parameters())
            if parameter.device != torch.device(device):
                raise ValueError("Caption device must match the loaded tagger device")
            identity = tagger_identity(self.tagger, max_num_patches=self.manifest["identity"]["max_num_patches"])
            if canonical_hash(identity) != canonical_hash(self.manifest["identity"]):
                raise ValueError("Loaded tagger no longer matches captioner")
            if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= self.model.config.max_length:
                raise ValueError("Generation limit exceeds model config")
            if tag_source not in {"predicted", "provided", "merged"}:
                raise ValueError("Unknown tag_source")
            if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
                raise ValueError("tags must be a list of strings")
            if tag_source == "predicted" and tags:
                raise ValueError("predicted mode does not accept provided tags")
            if cancelled():
                raise InterruptedError("Caption generation cancelled")
            with Image.open(io.BytesIO(image_bytes)) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            kwargs = {"images": [image], "return_tensors": "pt"}
            if self.tagger.is_naflex:
                kwargs["max_num_patches"] = identity["max_num_patches"]
            inputs = self.tagger.processor(**kwargs)
            prepared = time.perf_counter()
            inputs = {key: value.to(parameter.device) for key, value in inputs.items()}
            features = SigLIP2FeatureProvider(self.tagger.model).extract(inputs["pixel_values"],
                inputs.get("pixel_attention_mask"), inputs.get("spatial_shapes"))
            concepts = self.selector.select(features.tag_logits, mode=tag_source,
                provided=[tags] if tag_source != "predicted" else None)
            encoded = time.perf_counter()
            # The external GPU slot owns this request's decoder residency.
            try:
                self.model.to(device)
                result = self.model.generate(features, concepts, max_new_tokens=max_new_tokens, cancelled=cancelled)
                ids = result.token_ids[0, :int(result.lengths[0])].tolist()
                caption = self.tokenizer.decode(ids)
                selected = concepts.mask[0].nonzero().flatten().tolist()
                used = [{"name": self.selector.names[int(concepts.ids[0, index])],
                         "source": "provided" if int(concepts.sources[0, index]) else "predicted",
                         "score": float(concepts.scores[0, index]) if not int(concepts.sources[0, index]) else None,
                         "score_calibrated": False} for index in selected]
                finished = time.perf_counter()
                response = {"caption": caption, "used_tags": used, "warnings": list(concepts.warnings[0]),
                        "termination": "eos" if bool(result.reached_eos[0]) else "length_limit",
                        "token_count": len(ids), "image_hash": hashlib.sha256(image_bytes).hexdigest(),
                        "model_identity": self.manifest["files"]["model.safetensors"],
                        "timings_ms": {"preprocess": (prepared - started) * 1000,
                                       "encoder_concepts": (encoded - prepared) * 1000,
                                       "decoder": (finished - encoded) * 1000,
                                       "total": (finished - started) * 1000}}
            finally:
                self.model.to("cpu")
            response["timings_ms"]["total"] = (time.perf_counter() - started) * 1000
            return response
