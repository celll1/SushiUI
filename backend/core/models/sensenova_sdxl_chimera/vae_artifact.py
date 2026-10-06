"""Strict target-codec and prediction identity for aligned Chimera artifacts."""
import json

from .flow import FLOW_V1_PREDICTION, FLOW_V2_PREDICTION, FLOW_V2_VELOCITY_PREDICTION, FLOW_V3_PREDICTION, FLOW_V4_PREDICTION

POLAR_DRIFT_CONTRACT = dict(version=1, tangent_drift_units="dt",
    loss="coordinate_tangent_plus_time_drift_v1", integrator="mixed_dt_dd_drift_v1")


def _contract(value, channels):
    from .artifact import prediction_contract
    if not isinstance(value, dict) or value.get("type") not in (
            FLOW_V1_PREDICTION, FLOW_V2_PREDICTION, FLOW_V2_VELOCITY_PREDICTION, FLOW_V3_PREDICTION, FLOW_V4_PREDICTION):
        raise ValueError("Aligned Chimera artifact requires a supported prediction contract")
    result = prediction_contract(value["type"], latent_channels=channels,
        latent_mean=value.get("latent_mean"),
        latent_centered_second_moment=value.get("latent_centered_second_moment"),
        **{key: value[key] for key in (
            "angular_schedule", "angular_endpoint_slope", "radius_floor",
            "angular_singularity_threshold", "angular_step_limit") if key in value})
    if result != value:
        raise ValueError("Aligned Chimera prediction fields differ from their canonical contract")
    return result


def validate_documents(manifest, runtime):
    from core.models.components.latent_adapter import validate_declaration
    record = manifest.get("vae_alignment")
    if record is None:
        if runtime.get("vae_alignment") is not None:
            raise ValueError("Chimera runtime has an orphaned VAE alignment")
        return None
    if not isinstance(record, dict) or record.get("version") != 1 or record != runtime.get("vae_alignment"):
        raise ValueError("Chimera VAE alignment documents disagree")
    if len(json.dumps(record, allow_nan=False).encode("utf-8")) > 65536:
        raise ValueError("Chimera VAE alignment record exceeds its bounded size")
    declaration = validate_declaration(record.get("declaration"))
    width = max(declaration["old_channels"], declaration["new_channels"])
    if declaration["old_channels"] != 4 or declaration["input_channels"] != 3:
        raise ValueError("Chimera alignment requires its four-channel source and an RGB target codec")
    _contract(record.get("source_prediction"), width)
    target = _contract(manifest.get("prediction"), declaration["new_channels"])
    if target != record.get("target_prediction"):
        raise ValueError("Chimera target prediction differs from its saved VAE alignment")
    if target["type"] != record["source_prediction"]["type"]:
        raise ValueError("Chimera VAE swap cannot reinterpret its source prediction type")
    expected_boundary = POLAR_DRIFT_CONTRACT if target["type"] == FLOW_V4_PREDICTION else None
    if record.get("polar_boundary") != expected_boundary:
        raise ValueError("Chimera polar boundary drift contract is missing or incompatible")
    if runtime["unet"].get("in_channels") != width or runtime["unet"].get("out_channels") != width:
        raise ValueError("Chimera native UNet width disagrees with its spatial bridge")
    if manifest["vae"].get("latent_channels") != declaration["new_channels"] or manifest["vae"].get("scale_factor") != declaration["new_scale"]:
        raise ValueError("Chimera target codec geometry disagrees with its alignment")
    statistics = record.get("calibration") or {}
    if any(type(statistics.get(key)) is not int or statistics[key] < 1
           for key in ("images", "positions_per_channel")):
        raise ValueError("Chimera aligned artifact requires completed calibration statistics")
    from .artifact import prediction_contract
    moments = prediction_contract(FLOW_V2_VELOCITY_PREDICTION,
        latent_channels=declaration["new_channels"], latent_mean=statistics.get("latent_mean"),
        latent_centered_second_moment=statistics.get("latent_centered_second_moment"))
    if target["type"] != FLOW_V1_PREDICTION and any(target[key] != moments[key]
            for key in ("latent_mean", "latent_centered_second_moment")):
        raise ValueError("Chimera target prediction differs from its calibrated statistics")
    return declaration


def prepare_export(manifest, runtime, unet, vae, *, metadata, prediction, calibration):
    from copy import deepcopy
    from .artifact import config_hash
    from core.models.components.spatial_swap import artifact_metadata
    from core.models.components.latent_adapter import NormalizedImageVAE
    pair = getattr(unet, "vae_latent_bridge", None)
    if pair is None:
        if manifest.get("vae_alignment") is not None:
            raise ValueError("Chimera aligned export is missing its spatial bridge")
        return {}
    alignment_metadata = artifact_metadata(unet)
    if not isinstance(vae, NormalizedImageVAE) or vae.declaration != pair.declaration:
        raise ValueError("Chimera aligned export requires its matching normalized target codec")
    metadata = dict(metadata or {})
    if metadata.get("component.vae.embedded") != "1" or metadata.get("component.vae.prefix") != "vae.":
        raise ValueError("Chimera aligned checkpoints require a bundled target codec declaration")
    config = dict(vae.inner.config)
    if metadata.get("component.vae.class") != type(vae.inner).__name__:
        raise ValueError("Chimera target codec class differs from its weight metadata")
    if json.loads(metadata.get("component.vae.config", "null")) != config:
        raise ValueError("Chimera codec metadata disagrees with its native configuration")
    width = max(pair.declaration["old_channels"], pair.declaration["new_channels"])
    source = _contract(getattr(unet, "_chimera_source_prediction", None), width)
    target = _contract(dict(prediction or {}), pair.declaration["new_channels"])
    prior = manifest.get("vae_alignment") or {}
    manifest["vae_alignment"] = dict(version=1, declaration=deepcopy(pair.declaration),
        source_prediction=deepcopy(source), target_prediction=deepcopy(target),
        source_vae=deepcopy(prior.get("source_vae", manifest["vae"])),
        calibration=deepcopy(calibration or prior.get("calibration") or {}))
    if target["type"] == FLOW_V4_PREDICTION:
        manifest["vae_alignment"]["polar_boundary"] = dict(POLAR_DRIFT_CONTRACT)
    runtime["vae_alignment"] = deepcopy(manifest["vae_alignment"])
    runtime["unet"], runtime["vae"] = dict(unet.config), config
    manifest["unet"].update(config_hash=config_hash(runtime["unet"]),
                           parameter_count=sum(p.numel() for p in unet.parameters()))
    manifest["vae"].update(config_hash=config_hash(config),
        latent_channels=pair.declaration["new_channels"], scale_factor=pair.declaration["new_scale"],
        scale_temporal=int(metadata.get("component.vae.scale_temporal", 1)),
        embedded=True, ndim=pair.declaration["ndim"],
        source=pair.declaration["source"], form=pair.declaration["source"].split(":", 1)[0],
        family=metadata.get("component.vae.type"), vae_class=metadata.get("component.vae.class"),
        norm=pair.declaration["norm"], norm_pack=pair.declaration["norm_pack"],
        scaling_factor=config.get("scaling_factor"), shift_factor=config.get("shift_factor"),
        struct_native=metadata.get("component.vae.struct_native") == "1",
        identity_native=False, locator=None, latent_hash=pair.declaration["target_identity"],
        provenance=metadata.get("component.vae.provenance", pair.declaration["source"]))
    manifest["prediction"] = target
    validate_documents(manifest, runtime)
    metadata.update(alignment_metadata)
    metadata["chimera.vae_alignment"] = json.dumps(manifest["vae_alignment"], sort_keys=True, allow_nan=False)
    validate_metadata(manifest, runtime, metadata)
    return metadata


def validate_metadata(manifest, runtime, metadata):
    from core.models.components.spatial_swap import read_declaration
    declaration = validate_documents(manifest, runtime)
    if read_declaration(metadata) != declaration:
        raise ValueError("Chimera weight metadata disagrees with its alignment documents")
    if declaration is not None:
        raw = metadata.get("chimera.vae_alignment")
        if not isinstance(raw, str) or len(raw.encode("utf-8")) > 65536 or json.loads(raw) != manifest["vae_alignment"]:
            raise ValueError("Chimera alignment statistics differ from their weight metadata")
        if metadata.get("component.vae.embedded") != "1" or metadata.get("component.vae.prefix") != "vae.":
            raise ValueError("Aligned Chimera weights must bundle their target codec")
        if metadata.get("component.vae.identity_native") != "0":
            raise ValueError("Aligned Chimera must declare its changed latent identity")
        if metadata.get("component.vae.class") != manifest["vae"].get("vae_class"):
            raise ValueError("Chimera target codec class differs from its weight metadata")
        if json.loads(metadata.get("component.vae.config", "null")) != runtime["vae"]:
            raise ValueError("Chimera target codec metadata disagrees with its runtime configuration")
        from core.models.common.vae_source import latent_space_hash
        for field, key in (("channels", "new_channels"), ("scale_factor", "new_scale"),
                           ("norm_pack", "norm_pack")):
            if int(metadata.get("component.vae."+field, 0)) != declaration[key]:
                raise ValueError("Chimera target codec metadata geometry disagrees with its alignment")
        if metadata.get("component.vae.norm") != declaration["norm"]:
            raise ValueError("Chimera target codec normalization disagrees with its alignment")
        if latent_space_hash(metadata.get("component.vae.hash", ""), runtime["vae"]) != declaration.get("target_identity"):
            raise ValueError("Chimera target latent identity disagrees with its alignment")
    return declaration
