"""Adapt project-owned joint artifacts to the common tagger response services."""
from dataclasses import asdict
from pathlib import Path
import json

import torch
from torch import nn

from .artifact import FoveaArtifactConfig, load_inference_artifact
from .model import FoveaTagger
from .router import EvidenceRouter


def normalize_metadata(metadata, path):
    if metadata.get("kind") in ("foveatag_direct_joint_weights", "foveatag_onnx_stages"):
        return metadata
    if metadata.get("provenance", {}).get("stage") != "direct_joint_owner_review":
        return metadata
    from api.param_defaults import TAGGER_TRAINING_DEFAULTS
    cfg = FoveaArtifactConfig.from_dict(metadata["config"])
    plan_path = Path(path).with_name("review_plan.json")
    plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.is_file() else {}
    return {**metadata, "architecture":"foveatag", "kind":"foveatag_direct_joint_weights",
        "fovea_view_config":asdict(cfg.views), "fovea_seed":plan.get("seed", TAGGER_TRAINING_DEFAULTS["fovea_seed"]),
        "max_source_pixels":plan.get("max_source_pixels", TAGGER_TRAINING_DEFAULTS["fovea_max_source_pixels"]),
        "mixed_precision":"fp32", "step":metadata["provenance"]["global_step"]}


class DirectJointInferenceAdapter(nn.Module):
    architecture = "foveatag"
    training_method = "full"

    def __init__(self, path):
        super().__init__()
        metadata=json.loads(Path(path).with_name(f"{Path(path).stem}_metadata.json").read_text(encoding="utf-8"))
        self.config=FoveaArtifactConfig.from_dict(metadata["config"])
        cfg=self.config
        self.core=FoveaTagger(cfg.num_tags, encoder_config=cfg.encoder, feature_dim=cfg.feature_dim,
            evidence_slots_per_view=cfg.evidence_slots_per_view,pool_heads=cfg.pool_heads,temperature=cfg.temperature)
        self.router=EvidenceRouter(cfg.encoder.widths[2],routing_dim=cfg.routing_dim,routing_slots=cfg.routing_slots)
        self._ood_tap=nn.Identity()
        load_inference_artifact(path,self.core,router=self.router)

    @property
    def head(self):
        return self.core.head

    @property
    def vision_encoder(self):
        return self.core.encoder

    def register_ood_feature_hook(self,hook):
        return self._ood_tap.register_forward_pre_hook(hook)

    @torch.inference_mode()
    def predict_native(self,image,processor):
        from .inference import predict_image
        if self.training:
            raise ValueError("joint prediction requires evaluation mode")
        if processor.max_source_pixels is not None and image.width*image.height > processor.max_source_pixels:
            raise ValueError("native source image exceeds the pixel cap")
        dtype=torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else None
        mode="balanced" if "balanced" in self.config.inference_modes else next(iter(self.config.inference_modes))
        result=predict_image(image,self.core,self.router,self.config,mode=mode,mixed_precision=dtype)
        self._ood_tap(result.ood_features)
        return result.logits[None]
