"""Strict source, target-codec and bridge identity for LTX video swaps."""
import json

from core.models.components.latent_adapter import ImageCodecContract, SpatialAlignmentPair, validate_declaration
from core.models.components.video_latent_adapter import VideoCodecContract, NormalizedVideoVAE

KEY = "ltx2.video_alignment"


def canonical_config(config):
    from .residual_unet import native_config
    return json.loads(json.dumps(native_config(config), allow_nan=False))


def codec_contract(record):
    d = validate_declaration(record["declaration"])
    if not isinstance(record["source_config"], dict) or any(
            not isinstance(d.get(key), str) or not d[key] for key in ("source_identity", "target_identity")):
        raise ValueError("LTX alignment source configuration or latent identity is missing")
    return VideoCodecContract(ImageCodecContract(d["new_channels"], d["new_scale"], d["ndim"],
        d["norm"], d["norm_pack"], d["input_channels"]), record["temporal_scale"], record["temporal_policy"])


def read_record(metadata, config):
    raw = (metadata or {}).get(KEY)
    if raw is None:
        return None
    if not isinstance(raw, str) or len(raw.encode("utf-8")) > 65536:
        raise ValueError("Invalid LTX video alignment metadata")
    record = json.loads(raw)
    if not isinstance(record, dict) or set(record) != {
            "version", "declaration", "source_config", "temporal_scale", "temporal_policy"} or (
                type(record["version"]) is not int or record["version"] != 1):
        raise ValueError("Unsupported LTX video alignment record")
    d = validate_declaration(record["declaration"])
    contract = codec_contract(record)
    source, saved = canonical_config(record["source_config"]), canonical_config(config)
    if (source["in_channels"] != d["old_channels"] or source["out_channels"] != d["old_channels"] or
            tuple(source["vae_scale_factors"][1:]) != (d["old_scale"],)*2):
        raise ValueError("LTX alignment source geometry disagrees")
    expected = dict(source, in_channels=max(d["old_channels"], d["new_channels"]),
        out_channels=max(d["old_channels"], d["new_channels"]),
        vae_scale_factors=(contract.temporal_scale, d["new_scale"], d["new_scale"]))
    if canonical_config(expected) != saved:
        raise ValueError("LTX alignment backbone geometry disagrees")
    if metadata.get("component.vae.embedded") != "1" or metadata.get("component.vae.prefix") != "vae.":
        raise ValueError("Aligned LTX checkpoint must bundle its video codec")
    if metadata.get("component.vae.identity_native") != "0":
        raise ValueError("Aligned LTX checkpoint must declare a changed latent identity")
    for field, value in (("channels", d["new_channels"]), ("scale_factor", d["new_scale"]),
                         ("scale_temporal", contract.temporal_scale), ("norm_pack", d["norm_pack"])):
        if int(metadata.get("component.vae."+field, 0)) != value:
            raise ValueError("LTX video codec metadata geometry disagrees")
    if metadata.get("component.vae.norm") != d["norm"]:
        raise ValueError("LTX video codec normalization disagrees")
    codec_config = json.loads(metadata.get("component.vae.config", "null"))
    if not isinstance(codec_config, dict) or not metadata.get("component.vae.class"):
        raise ValueError("LTX video codec configuration is missing")
    if not metadata.get("component.vae.hash"):
        raise ValueError("LTX video codec weight identity is missing")
    from core.models.common.vae_source import latent_space_hash
    if latent_space_hash(metadata.get("component.vae.hash", ""), codec_config) != d.get("target_identity"):
        raise ValueError("LTX target latent identity disagrees")
    return record


def export_record(model, vae, metadata):
    inner = getattr(model, "transformer", model)
    pair = getattr(inner, "_video_latent_alignment", None)
    if pair is None:
        return {}
    if not pair.ready or not isinstance(vae, NormalizedVideoVAE) or vae.contract != inner._video_codec_contract:
        raise ValueError("Cannot save an uncalibrated or mismatched LTX video swap")
    if (json.loads(metadata.get("component.vae.config", "null")) != json.loads(json.dumps(dict(vae.inner.config))) or
            metadata.get("component.vae.class") != type(vae.inner).__name__):
        raise ValueError("LTX video codec metadata differs from its actual codec")
    record = dict(version=1, declaration=dict(pair.declaration), source_config=inner._video_source_config,
        temporal_scale=vae.contract.temporal_scale, temporal_policy=vae.contract.temporal_policy)
    result = {KEY: json.dumps(record, sort_keys=True, allow_nan=False)}
    read_record({**metadata, **result}, inner.config)
    return result


def prepare_restore(model, record):
    model._video_latent_alignment = SpatialAlignmentPair(record["declaration"])
    model._video_codec_contract = codec_contract(record)
    model._video_source_config = record["source_config"]


def restored_codec(path, metadata, record, *, torch_dtype):
    from dataclasses import replace
    from core.models.common.vae_source import load_declared_latent_io
    resolved = load_declared_latent_io(path, arch="ltx2", load_weights=True, download=False)
    contract = VideoCodecContract.from_resolved(resolved, temporal_policy=record["temporal_policy"])
    if contract != codec_contract(record):
        raise ValueError("Restored LTX video codec disagrees with its alignment")
    module = resolved.load_module(torch_dtype=torch_dtype).requires_grad_(False).eval()
    return NormalizedVideoVAE(module, contract), replace(resolved, state_dict=None)
