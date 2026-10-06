"""Bundled target Oobleck codec and split ACE decoder configuration."""

from dataclasses import asdict
import json
from math import prod
import re
from types import SimpleNamespace

import torch

from core.models.components.audio_latent_adapter import AudioCodecContract, AudioAlignmentPair
from core.models.common.vae_source import content_hash_for_state_dict

KEY = "acestep.audio_alignment"
DECODER_KEY = "acestep_decoder_config"
_DTYPES = {str(v): v for v in (torch.float32, torch.float16, torch.bfloat16)}


def _contract(config):
    if not isinstance(config, dict):
        raise ValueError("Audio codec config must be an object")
    ratios = config.get("downsampling_ratios", ())
    if not isinstance(ratios, (list, tuple)) or not ratios or any(type(n) is not int for n in ratios):
        raise ValueError("Audio codec downsampling config is missing")
    return AudioCodecContract.from_oobleck(SimpleNamespace(config=config, hop_length=prod(ratios)))


def read_record(metadata, owner_config):
    from .residual_unet import native_config, AUDIO_CODEC_KEY, AUDIO_TIME_KEY
    raw = (metadata or {}).get(KEY)
    if raw is None:
        if DECODER_KEY in (metadata or {}):
            raise ValueError("ACE decoder config has no audio alignment declaration")
        return None
    if not isinstance(raw, str) or len(raw.encode("utf-8")) > 65536:
        raise ValueError("Invalid ACE audio alignment metadata")
    record = json.loads(raw)
    fields = {
            "version", "method", "token_policy", "normalization", "source", "target",
            "source_config", "target_config", "source_hash", "target_hash", "source_dtype"}
    if isinstance(record, dict) and record.get("version") == 2:
        fields.add("position_policy")
    if not isinstance(record, dict) or set(record) != fields:
        raise ValueError("Invalid ACE audio alignment record fields")
    if type(record["version"]) is not int or record["version"] not in (1, 2) or (
            record["method"], record["token_policy"], record["normalization"]) != (
                "paired_temporal", "target_native", "identity"):
        raise ValueError("Unsupported ACE audio alignment contract")
    policy = record.get("position_policy", "target_index_v1")
    if policy not in ("source_hop_v1", "target_index_v1"):
        raise ValueError("Unsupported ACE audio position policy")
    source, target = _contract(record["source_config"]), _contract(record["target_config"])
    source.check_pair(target)
    if not isinstance(record["source"], dict) or not isinstance(record["target"], dict):
        raise ValueError("ACE audio codec geometry must be an object")
    if set(record["source"]) != set(asdict(source)) or set(record["target"]) != set(asdict(target)):
        raise ValueError("ACE audio codec geometry fields differ")
    if AudioCodecContract(**record["source"]) != source or AudioCodecContract(**record["target"]) != target:
        raise ValueError("ACE audio codec config and geometry differ")
    for key in ("source_hash", "target_hash"):
        if not isinstance(record[key], str) or re.fullmatch(r"[0-9a-f]{16}", record[key]) is None:
            raise ValueError("ACE audio codec identity is missing")
    if record["source_dtype"] not in _DTYPES:
        raise ValueError("Unsupported ACE source codec dtype")
    source_native = native_config(owner_config)
    if source_native["audio_acoustic_hidden_dim"] != source.latent_channels or source_native["in_channels"] != 3*source.latent_channels:
        raise ValueError("ACE conditioning config and source codec differ")
    decoder = json.loads(metadata.get(DECODER_KEY, "null"))
    if not isinstance(decoder, dict):
        raise ValueError("ACE aligned decoder config is missing")
    expected = dict(source_native, audio_acoustic_hidden_dim=max(source.latent_channels, target.latent_channels),
                    in_channels=3*max(source.latent_channels, target.latent_channels),
                    **{AUDIO_CODEC_KEY: asdict(target)})
    if policy == "source_hop_v1":
        expected[AUDIO_TIME_KEY] = dict(policy=policy, source_hop=source.hop_length)
    if native_config(decoder) != expected:
        raise ValueError("ACE aligned decoder config differs from the audio boundary")
    if metadata.get("component.audio_vae.embedded") != "1" or metadata.get("component.audio_vae.prefix") != "vae.":
        raise ValueError("ACE aligned checkpoint must bundle its target audio codec")
    return record


def export_bundle(model, codec, source_codec=None):
    from .residual_unet import decoder_identity_config
    from diffusers import AutoencoderOobleck
    decoder = getattr(model, "decoder", model)
    pair = getattr(decoder, "_audio_latent_alignment", None)
    if pair is None:
        return {}, {}
    decoder_identity_config(model)
    if not pair.ready or not isinstance(codec, AutoencoderOobleck) or AudioCodecContract.from_oobleck(codec) != pair.target:
        raise ValueError("Cannot export an uncalibrated or mismatched ACE audio swap")
    source_identity = getattr(model, "_audio_source_codec_identity", None)
    if source_codec is not None:
        if not isinstance(source_codec, AutoencoderOobleck) or AudioCodecContract.from_oobleck(source_codec) != pair.source:
            raise ValueError("ACE source audio codec differs from the calibration geometry")
        source_identity = dict(config=dict(source_codec.config), hash=content_hash_for_state_dict(source_codec.state_dict()),
                               dtype=str(next(source_codec.parameters()).dtype))
        previous = getattr(model, "_audio_source_codec_identity", None)
        if previous is not None and source_identity != previous:
            raise ValueError("ACE source codec identity changed since calibration/restoration")
    if source_identity is None:
        raise ValueError("ACE audio export requires the original conditioning codec identity")
    state = {key: value.detach().cpu().contiguous() for key, value in codec.state_dict().items()}
    record = dict(version=2, method="paired_temporal", token_policy="target_native", normalization="identity",
        position_policy=getattr(pair, "position_policy", "target_index_v1"),
        source=asdict(pair.source), target=asdict(pair.target), source_config=source_identity["config"],
        target_config=dict(codec.config), source_hash=source_identity["hash"], target_hash=content_hash_for_state_dict(state),
        source_dtype=source_identity["dtype"])
    metadata = {KEY: json.dumps(record, sort_keys=True, allow_nan=False),
                DECODER_KEY: json.dumps(decoder.config.to_dict(), sort_keys=True, allow_nan=False),
                "component.audio_vae.embedded": "1", "component.audio_vae.prefix": "vae."}
    read_record(metadata, model.config)
    return metadata, {"vae."+key: value for key, value in state.items()}


def preflight_headers(handle, record):
    old, new = record["source"]["latent_channels"], record["target"]["latent_channels"]
    width = max(old, new)
    keys = set(handle.keys())
    prefix = "net." if any(key.startswith("net.") for key in keys) else ""
    bridge = prefix+"decoder._audio_latent_alignment."
    shapes = {"input.projection.weight": [width,new,3], "input.projection.bias": [width],
              "output.projection.weight": [new,width,3], "output.projection.bias": [new],
              "codec_spec": [9], "input.bridge_spec": [3], "output.bridge_spec": [3],
              "input.calibrated": [], "output.calibrated": []}
    for key, shape in shapes.items():
        if bridge+key not in keys or list(handle.get_slice(bridge+key).get_shape()) != shape:
            raise ValueError("ACE audio bridge headers disagree: "+key)
    specs = {"codec_spec": [1,old,record["source"]["sample_rate"],record["source"]["waveform_channels"],
        record["source"]["hop_length"],new,record["target"]["sample_rate"],record["target"]["waveform_channels"],record["target"]["hop_length"]],
        "input.bridge_spec": [1,new,width], "output.bridge_spec": [1,width,new]}
    for key, expected in specs.items():
        value = handle.get_tensor(bridge+key)
        if value.dtype != torch.int64 or value.tolist() != expected:
            raise ValueError("ACE audio bridge tensor contract differs: "+key)
    if not any(key.startswith("vae.encoder.") for key in keys) or not any(key.startswith("vae.decoder.") for key in keys):
        raise ValueError("ACE audio checkpoint is missing its bundled codec")
    for key in ("input.calibrated", "output.calibrated"):
        flag = handle.get_tensor(bridge+key)
        if flag.dtype != torch.bool or not bool(flag):
            raise ValueError("ACE audio checkpoint bridge is uncalibrated")
    from .residual_unet import native_config
    decoder = native_config(json.loads(handle.metadata()[DECODER_KEY]))
    hidden, patch = decoder["hidden_size"], decoder["patch_size"]
    for key, shape in {"decoder.proj_in.1.weight": [hidden,3*width,patch],
                       "decoder.proj_out.1.weight": [hidden,width,patch],
                       "decoder.proj_in.1.bias": [hidden], "decoder.proj_out.1.bias": [width]}.items():
        if prefix+key not in keys or list(handle.get_slice(prefix+key).get_shape()) != shape:
            raise ValueError("ACE aligned decoder headers disagree: "+key)


def prepare_restore(model, record):
    from .vae_boundary import install_decoder_alignment
    pair = AudioAlignmentPair(AudioCodecContract(**record["source"]), AudioCodecContract(**record["target"]))
    install_decoder_alignment(model, pair, position_policy=record.get("position_policy", "target_index_v1"))
    model._audio_source_codec_identity = dict(config=record["source_config"], hash=record["source_hash"], dtype=record["source_dtype"])


def resolved_codec_identity(model, codec, path):
    from core.models.common.vae_source import ResolvedVAE
    pair = getattr(model.decoder, "_audio_latent_alignment", None)
    if pair is None:
        return None
    if AudioCodecContract.from_oobleck(codec) != pair.target or not pair.ready:
        raise ValueError("ACE codec identity requires its restored calibrated boundary")
    return ResolvedVAE(source="model:"+str(path), form="model", family="custom",
        latent_channels=pair.target.latent_channels, scale_factor=1, scale_temporal=pair.target.hop_length,
        ndim=3, norm="identity", norm_pack=1, vae_class="AutoencoderOobleck",
        config=dict(codec.config, _class_name="AutoencoderOobleck"), content_hash=content_hash_for_state_dict(codec.state_dict()),
        provenance="bundled ACE audio codec", locator=None, struct_native=False, identity_native=False,
        scaling_factor=None, shift_factor=None, path=str(path), prefix="vae.")


def restore_codec(record, state, dtype):
    from diffusers import AutoencoderOobleck
    if content_hash_for_state_dict(state) != record["target_hash"]:
        raise ValueError("ACE bundled target codec weight identity differs")
    codec = AutoencoderOobleck.from_config(record["target_config"])
    codec.load_state_dict(state, strict=True)
    if AudioCodecContract.from_oobleck(codec) != AudioCodecContract(**record["target"]):
        raise ValueError("Restored ACE target codec geometry differs")
    return codec.to(dtype=dtype).requires_grad_(False).eval()


def verify_source_codec(model, codec):
    identity = model._audio_source_codec_identity
    if dict(codec.config) != identity["config"] or content_hash_for_state_dict(codec.state_dict()) != identity["hash"]:
        raise ValueError("ACE original conditioning codec identity differs from calibration")


def source_dtype(model):
    return _DTYPES[model._audio_source_codec_identity["dtype"]]
