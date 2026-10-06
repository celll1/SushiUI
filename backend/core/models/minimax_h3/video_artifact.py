"""Native H3 Full checkpoint with an embedded target codec and spatial bridges."""

from dataclasses import asdict, replace
import json
import re
from pathlib import Path

import torch

from core.models.components.latent_adapter import ImageCodecContract,SpatialAlignmentPair,validate_declaration,_Config
from core.models.components.video_latent_adapter import VideoCodecContract,NormalizedVideoVAE
from .residual_unet import native_config,validate_topology_identity,restore_topology,VIDEO_CODEC_KEY,GRID_POLICY_KEY
from .video_alignment import AlignedMiniMaxH3BlockLoopWrapper,VIDEO_POSITION_POLICY_KEY,VIDEO_POSITION_POLICY

KEY = "minimax_h3.video_alignment"
FORMAT = "sushi_h3_native_full_v1"


def canonical_config(config):
    return json.loads(json.dumps(native_config(config),allow_nan=False))


def constructor_config(config):
    value = native_config(config)
    for key in (VIDEO_CODEC_KEY,GRID_POLICY_KEY,VIDEO_POSITION_POLICY_KEY):
        value.pop(key,None)
    return value


def codec_contract(record):
    d = validate_declaration(record["declaration"])
    return VideoCodecContract(ImageCodecContract(d["new_channels"],d["new_scale"],d["ndim"],
        d["norm"],d["norm_pack"],d["input_channels"]),record["temporal_scale"],record["temporal_policy"])


def read_record(metadata, config):
    raw = metadata.get(KEY)
    if not isinstance(raw,str) or len(raw.encode("utf-8")) > 65536:
        raise ValueError("Native H3 Full checkpoint requires a video alignment record")
    record = json.loads(raw)
    if (not isinstance(record,dict) or set(record) != {"version","declaration","source_config",
            "temporal_scale","temporal_policy","position_policy"} or type(record["version"]) is not int
            or record["version"] != 1 or record["position_policy"] != VIDEO_POSITION_POLICY):
        raise ValueError("Unsupported H3 video alignment record")
    d = validate_declaration(record["declaration"])
    if any(not isinstance(d.get(key),str) or not d[key] for key in ("source_identity","target_identity")):
        raise ValueError("H3 source/target latent identity is missing")
    if not isinstance(record["source_config"],dict) or VIDEO_CODEC_KEY in record["source_config"]:
        raise ValueError("H3 alignment requires the original source configuration")
    source = canonical_config(record["source_config"])
    if source != record["source_config"]:
        raise ValueError("H3 alignment source configuration is not canonical")
    contract = codec_contract(record)
    if source["in_channels"] != d["old_channels"] or d["old_scale"] != 16 or source["patch_size"] != [1,2,2]:
        raise ValueError("H3 alignment source geometry disagrees")
    expected = dict(source,in_channels=max(d["old_channels"],d["new_channels"]),
                    **{VIDEO_CODEC_KEY:asdict(contract),VIDEO_POSITION_POLICY_KEY:VIDEO_POSITION_POLICY})
    if canonical_config(expected) != canonical_config(config):
        raise ValueError("H3 alignment backbone geometry disagrees")
    if (metadata.get("component.vae.embedded") != "1" or metadata.get("component.vae.prefix") != "vae."
            or metadata.get("component.vae.identity_native") != "0"):
        raise ValueError("Aligned H3 checkpoint must bundle its changed video codec")
    for key,value in (("channels",d["new_channels"]),("scale_factor",d["new_scale"]),
                      ("scale_temporal",contract.temporal_scale),("norm_pack",d["norm_pack"])):
        if metadata.get("component.vae."+key) != str(value):
            raise ValueError("H3 codec metadata geometry disagrees")
    codec_config = json.loads(metadata.get("component.vae.config","null"))
    if (not isinstance(codec_config,dict) or not metadata.get("component.vae.class")
            or metadata.get("component.vae.norm") != d["norm"] or not metadata.get("component.vae.hash")):
        raise ValueError("H3 codec configuration/normalization/weight identity is missing")
    from core.models.common.vae_source import latent_space_hash
    if latent_space_hash(metadata["component.vae.hash"],codec_config) != d["target_identity"]:
        raise ValueError("H3 target latent identity disagrees")
    return record


def export_record(model, vae, metadata):
    inner = getattr(model,"transformer",model)
    pair = getattr(inner,"_video_latent_alignment",None)
    if (pair is None or not pair.ready or not isinstance(vae,NormalizedVideoVAE)
            or vae.contract != inner._video_codec_contract):
        raise ValueError("Cannot save an uncalibrated or mismatched H3 video swap")
    from core.models.common.vae_source import content_hash_for_state_dict
    if (content_hash_for_state_dict(vae.inner.state_dict()) != metadata.get("component.vae.hash")
            or json.loads(metadata.get("component.vae.config","null")) != json.loads(json.dumps(dict(vae.inner.config)))
            or metadata.get("component.vae.class") != type(vae.inner).__name__):
        raise ValueError("H3 codec metadata differs from its actual codec")
    record = dict(version=1,declaration=dict(pair.declaration),source_config=inner._video_source_config,
        temporal_scale=vae.contract.temporal_scale,temporal_policy=vae.contract.temporal_policy,
        position_policy=VIDEO_POSITION_POLICY)
    result = {KEY:json.dumps(record,sort_keys=True,allow_nan=False)}
    read_record({**metadata,**result},inner.config)
    return result


def prepare_restore(model,record):
    model._video_latent_alignment = SpatialAlignmentPair(record["declaration"])
    model._video_codec_contract = codec_contract(record)
    model._video_source_config = record["source_config"]


def save_checkpoint(trainer,path,*,step,epoch,max_shard_bytes=None):
    from core.training.vae_swap import swap_metadata
    from core.training.adapters.base_adapter import reject_quantized_base
    from core.models.common.single_file_format import save_single_file_state
    from .residual_unet import model_topology_metadata
    inner = getattr(trainer.transformer,"transformer",trainer.transformer)
    reject_quantized_base(inner,model_label="MiniMax H3")
    resolved,bundled,metadata = swap_metadata(trainer)
    if resolved is None or not bundled:
        raise ValueError("H3 aligned Full checkpoint requires a bundled codec")
    metadata.update(export_record(inner,trainer.vae,metadata))
    metadata.update(model_topology_metadata(inner))
    metadata.update(model_type="minimax_h3",**{"modelspec.architecture":"minimax_h3"},
        format=FORMAT,transformer_config=json.dumps(canonical_config(inner.config),allow_nan=False),step=str(step),epoch=str(epoch))
    state = {"net."+key:value.detach().to("cpu").contiguous() for key,value in inner.state_dict().items()}
    state.update({"vae."+key:value.detach().to("cpu").contiguous() for key,value in trainer.vae.inner.state_dict().items()})
    path = Path(path)
    if path.is_dir():
        path = path/f"minimax_h3_step_{step}.safetensors"
    elif not str(path).endswith(".safetensors"):
        path = Path(str(path)+".safetensors")
    path.parent.mkdir(parents=True,exist_ok=True)
    options = {} if max_shard_bytes is None else dict(max_shard_bytes=max_shard_bytes)
    return Path(save_single_file_state(state,metadata,str(path),**options))


def _artifact_path(path):
    source = Path(path)
    if not source.exists() and Path(str(source)+".index.json").is_file():
        source = Path(str(source)+".index.json")
    return source


def _headers(path):
    from safetensors import safe_open
    source = _artifact_path(path)
    mapping = None
    if str(source).endswith(".safetensors.index.json"):
        index = json.loads(source.read_text(encoding="utf-8"))
        metadata,mapping = index.get("metadata") or {},index.get("weight_map")
        if not isinstance(mapping,dict) or not mapping or any(
                not isinstance(name,str) or Path(name).name != name for name in mapping.values()):
            raise ValueError("Invalid H3 checkpoint shard mapping")
        shards = {source.parent/name for name in mapping.values()}
    else:
        metadata = None
        shards = {source}
    shapes,dtypes = {},{}
    for shard in shards:
        with safe_open(str(shard),framework="pt",device="cpu") as handle:
            md = handle.metadata() or {}
            if metadata is None:
                metadata = md
            if {key:str(value) for key,value in metadata.items() if key != "total_size"} != md:
                raise ValueError("H3 index and shard metadata disagree")
            for key in handle.keys():
                if key in shapes or (mapping is not None and mapping.get(key) != shard.name):
                    raise ValueError("H3 checkpoint shard tensor ownership disagrees")
                view = handle.get_slice(key)
                shapes[key],dtypes[key] = tuple(view.get_shape()),view.get_dtype()
    if mapping is not None and set(shapes) != set(mapping):
        raise ValueError("H3 checkpoint index is missing tensors")
    return metadata,shapes,dtypes


def preflight_checkpoint(path):
    from accelerate import init_empty_weights
    from core.rewiring.spec import read_checkpoint_metadata
    from core.models.common.vae_source import load_declared_latent_io
    from .vendor import MiniMaxH3Transformer3DModel
    path = _artifact_path(path)
    metadata,shapes,dtypes = _headers(path)
    if metadata.get("format") != FORMAT or metadata.get("model_type") != "minimax_h3":
        raise ValueError("Not a native H3 aligned Full checkpoint")
    config = json.loads(metadata.get("transformer_config","null"))
    if not isinstance(config,dict):
        raise ValueError("H3 native transformer configuration is missing")
    if config != canonical_config(config):
        raise ValueError("H3 Full checkpoint requires an explicit canonical native configuration")
    record = read_record(metadata,config)
    graph = read_checkpoint_metadata(metadata)
    if graph is not None:
        graph = validate_topology_identity(graph,config)
    blocks = {int(match.group(1)) for key in shapes if
              (match := re.fullmatch(r"net\.transformer_blocks\.(\d+)\.attn\.to_q\.weight",key))}
    if len(blocks) != config["num_layers"] or sorted(blocks) != list(range(config["num_layers"])):
        raise ValueError("H3 native block headers disagree with their declared depth")
    hidden = config["hidden_size"]
    heads = config["num_attention_heads"]*config["attention_head_dim"]
    early_shapes = {"net.proj_in.weight":(hidden,4*config["in_channels"]),
                    "net.proj_out.weight":(4*config["in_channels"],hidden),
                    "net.audio_proj_in.weight":(hidden,config["audio_in_channels"]),
                    "net.audio_proj_out.weight":(config["audio_in_channels"],hidden),
                    "net.context_embedder.weight":(hidden,config["text_dim"])}
    early_shapes.update({f"net.transformer_blocks.{index}.attn.to_q.weight":(heads,hidden) for index in blocks})
    if config["adaln_curve_grid"] is not None:
        early_shapes["net.adaln_t_table"] = (config["adaln_curve_grid"],config["time_embed_dim"])
    if any(shapes.get(key) != value for key,value in early_shapes.items()):
        raise ValueError("H3 native header geometry disagrees with its declared configuration")
    with init_empty_weights():
        model = MiniMaxH3Transformer3DModel(**constructor_config(config))
        prepare_restore(model,record)
    expected = {"net."+key:tuple(value.shape) for key,value in model.state_dict().items()}
    native = {key:shape for key,shape in shapes.items() if key.startswith("net.")}
    if native != expected or any(not key.startswith(("net.","vae.")) for key in shapes):
        raise ValueError("H3 native backbone/bridge headers are incomplete or have invalid geometry")
    for key,value in model.state_dict().items():
        dtype = dtypes["net."+key]
        if (value.is_floating_point() and dtype not in ("F32","F16","BF16")) or (
                not value.is_floating_point() and dtype != {torch.bool:"BOOL",torch.int64:"I64"}.get(value.dtype)):
            raise ValueError("H3 Full checkpoint has unsupported native tensor dtypes")
    codec = load_declared_latent_io(str(path),arch="minimax_h3",load_weights=False,download=False)
    if codec is None or VideoCodecContract.from_resolved(codec,temporal_policy=record["temporal_policy"]) != codec_contract(record):
        raise ValueError("H3 bundled codec header geometry disagrees")
    return metadata,config,record,graph


def restored_codec(path,record):
    from core.models.common.vae_source import resolve_vae_source
    resolved = resolve_vae_source("model:"+str(path),arch="minimax_h3",load_weights=True,download=False,require_backbone=False)
    if resolved.latent_hash != record["declaration"]["target_identity"]:
        raise ValueError("H3 bundled codec weight identity disagrees")
    contract = VideoCodecContract.from_resolved(resolved,temporal_policy=record["temporal_policy"])
    if contract != codec_contract(record):
        raise ValueError("Restored H3 codec disagrees with its alignment")
    module = resolved.load_module(assign=True).requires_grad_(False).eval()
    if any(value.is_meta for value in list(module.parameters())+list(module.buffers())):
        raise ValueError("H3 restored codec left meta tensors")
    return NormalizedVideoVAE(module,contract),replace(resolved,state_dict=None,identity_native=False)


def load_checkpoint(path,torch_dtype=torch.bfloat16):
    from accelerate import init_empty_weights
    from core.models.common.vae_source import _WeightFile
    from .loader import _dit_target_dtype
    from .vendor import MiniMaxH3Transformer3DModel
    path = _artifact_path(path)
    metadata,config,record,graph = preflight_checkpoint(path)
    codec,identity = restored_codec(path,record)
    weights = _WeightFile(str(path))
    if {key:str(value) for key,value in weights.metadata.items()} != {key:str(value) for key,value in metadata.items()}:
        raise ValueError("H3 checkpoint metadata changed during restoration")
    raw = weights.tensors(key for key in weights.shapes if key.startswith("net."))
    with init_empty_weights():
        model = MiniMaxH3Transformer3DModel(**constructor_config(config))
        prepare_restore(model,record)
    state = {}
    for key in list(raw):
        value = raw.pop(key)
        name = key.removeprefix("net.")
        dtype = torch.float32 if name.startswith("_video_latent_alignment.") else _dit_target_dtype(name,torch_dtype,config.get("adaln_curve_grid") is not None)
        state[name] = value.to(dtype) if value.is_floating_point() else value
    model.load_state_dict(state,strict=True,assign=True)
    if not model._video_latent_alignment.ready:
        raise ValueError("Restored H3 video alignment is uncalibrated")
    if any(value.is_meta for value in list(model.parameters())+list(model.buffers())):
        raise ValueError("H3 restored backbone left meta tensors")
    model.register_to_config(**{VIDEO_CODEC_KEY:asdict(codec_contract(record)),VIDEO_POSITION_POLICY_KEY:VIDEO_POSITION_POLICY})
    restore_topology(model,graph)
    wrapper = AlignedMiniMaxH3BlockLoopWrapper(model)
    wrapper.config = _Config(dict(model.config),in_channels=record["declaration"]["new_channels"])
    return wrapper.eval().requires_grad_(False),codec,identity
