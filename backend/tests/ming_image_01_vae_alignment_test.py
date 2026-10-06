"""Coarse-grid VAE swap training, portable restoration and token accounting."""
from types import SimpleNamespace
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch
from torch import nn
from safetensors.torch import save_file, load_file

torch.cuda.get_device_capability = lambda *a, **k: (8, 9)
torch.cuda._lazy_init = lambda *a, **k: None
torch._C._cuda_init = lambda *a, **k: None
torch.set_num_threads(1)

from core.models.ming_image_01.vae_alignment import (MingVAEAlignment,
    NormalizedFrameVAE, calibration_grid, restore, resize_backbone, WEIGHTS)
from core.models.ming_image_01.model_definition import create_transformer
from core.models.ming_image_01.training import flow_loss
from core.models.ming_image_01.conditioning import MingConditioning
from ming_image_01_rewiring_test import config, spec_for
from core.models.ming_image_01.residual_unet import restore_topology
from ming_image_01_artifact_test import package


def declaration(channels=64, ndim=5, input_channels=4):
    return dict(version=1, method="paired_spatial", token_policy="target_native",
        old_channels=16, new_channels=channels, old_scale=8, new_scale=16,
        source="model:example", input_channels=input_channels, ndim=ndim,
        norm="shift_scale", norm_pack=1)


def initialize(alignment):
    g = torch.Generator().manual_seed(754)
    for layer in (alignment.input, alignment.output):
        ins, outs = layer.projection.in_channels, layer.projection.out_channels
        weight = torch.randn(ins*9, outs, generator=g)*.02
        bias = torch.randn(outs, generator=g)*.01
        layer.initialize(weight, bias)
    return alignment


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rewired", [False, True])
def test_full_forward_backward_uses_reduced_target_grid(dtype, rewired):
    model = create_transformer(config("zero_masked"), device="cpu", dtype=dtype)
    if rewired:
        restore_topology(model, spec_for(model))
    resize_backbone(model, 64)
    model.add_module("vae_latent_bridge", initialize(MingVAEAlignment(declaration())).to(dtype=dtype))
    clean = torch.randn(1, 64, 1, 8, 8, dtype=dtype)
    conditioning = MingConditioning(torch.randn(1, 3, 8, dtype=dtype), torch.randn(1, 2, 24, dtype=dtype))
    seen = []
    hook = model.all_x_embedder["2-1"].register_forward_pre_hook(lambda _m, args: seen.append(args[0].shape))
    result = flow_loss(model, clean=clean, noise=torch.randn_like(clean), sigma=torch.tensor([.4]),
        conditioning=conditioning, frame_weights=torch.ones(1, 1))
    hook.remove()
    assert result.prediction.shape == clean.shape
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert model.vae_latent_bridge.input.projection.weight.grad.abs().sum() > 0
    assert model.vae_latent_bridge.output.projection.weight.grad.abs().sum() > 0
    from core.models.ming_image_01.vendor.diffusion.transformer import SEQ_MULTI_OF
    assert seen[0][0] == 16 + (-16) % SEQ_MULTI_OF
    assert seen[0][-1] == 256
    assert model.all_x_embedder["2-1"].weight.grad.reshape(-1,4,64)[:,:,16:].abs().sum() > 0
    assert not torch.cuda.is_initialized()


def test_safetensors_restoration_preserves_learned_weights_and_geometry(tmp_path):
    model = create_transformer(config("zero_masked"), device="cpu", dtype=torch.float32)
    alignment = initialize(MingVAEAlignment(declaration()))
    with torch.no_grad():
        alignment.input.projection.weight.add_(.4)
    save_file(alignment.state_dict(), str(tmp_path/WEIGHTS))
    artifact = SimpleNamespace(root=tmp_path, configs={"transformer":{"vae_alignment":declaration()}})
    restore(model, artifact, torch.float32)
    assert model.config.vae_alignment["token_policy"] == "target_native"
    torch.testing.assert_close(model.vae_latent_bridge.input.projection.weight, alignment.input.projection.weight)
    assert model.vae_latent_bridge.ready


def test_calibration_resamples_only_old_clean_latents():
    old = torch.randn(2,16,3,16,24)
    result = calibration_grid(old, (8,12))
    assert result.shape == (2,16,3,8,12)
    for frame in range(3):
        torch.testing.assert_close(result[:,:,frame], torch.nn.functional.avg_pool2d(old[:,:,frame],2))
    pair = MingVAEAlignment(declaration())
    assert pair.input.source_pack == pair.input.target_pack == 1


class ToyVAE(nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.rank = rank
        from diffusers.configuration_utils import FrozenDict
        self.config = FrozenDict(scaling_factor=2., shift_factor=.5, in_channels=3)

    def encode(self, value):
        assert value.ndim == self.rank and value.shape[1] == 3
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: value))

    def decode(self, value, return_dict=False):
        assert value.ndim == self.rank
        return (value,)


@pytest.mark.parametrize("rank", [4,5])
def test_normalized_frame_facade_keeps_frames_and_handles_rgb_alpha(rank):
    vae = NormalizedFrameVAE(ToyVAE(rank), declaration(3, rank, 3))
    rgba = torch.randn(2,4,3,8,12)
    z = vae.encode(rgba).latent_dist.mode()
    torch.testing.assert_close(z, (rgba[:,:3]-.5)*2)
    pixels = vae.decode(z, return_dict=False)[0]
    torch.testing.assert_close(pixels[:,:3], rgba[:,:3])
    torch.testing.assert_close(pixels[:,3], torch.ones_like(pixels[:,3]))
    assert "temperal_downsample" in vae.config


def test_invalid_preserved_token_policy_is_rejected():
    value = declaration()
    value["token_policy"] = "preserve_original"
    with pytest.raises(ValueError, match="target VAE token grid"):
        MingVAEAlignment(value)


def test_lifted_calibration_retains_extra_channel_capacity():
    from core.models.components.vae_alignment import SpatialLatentBridge
    incoming, outgoing = SpatialLatentBridge(64,16), SpatialLatentBridge(16,64)
    for layer in (incoming, outgoing):
        layer.initialize(torch.randn(layer.projection.in_channels*9,layer.projection.out_channels)*.01,
                         torch.randn(layer.projection.out_channels)*.01)
    pair = MingVAEAlignment(declaration())
    pair.initialize_from_pair(incoming,outgoing)
    source = torch.randn(1,64,1,8,8)
    torch.testing.assert_close(pair.input(source)[:,:16],incoming(source))
    torch.testing.assert_close(pair.input(source)[:,16:],source[:,16:])
    legacy = torch.randn(1,16,1,8,8)
    wide = torch.cat((legacy,torch.zeros(1,48,1,8,8)),dim=1)
    torch.testing.assert_close(pair.output(wide),outgoing(legacy))


def test_global_adapter_resize_preserves_coordinates_and_patch_order():
    from core.models.ming_image_01.partition_global import MingPartitionGlobalAdapter,resize_channels
    adapter = MingPartitionGlobalAdapter(16,24,rank=4,summary_tokens=2)
    resized = resize_channels(adapter,64)
    old = torch.randn(1,16,1,8,8)
    wide = torch.cat((old,torch.randn(1,48,1,8,8)),dim=1)
    for name in ("key","value","query"):
        torch.testing.assert_close(getattr(adapter,name)(adapter._features(old)),
                                   getattr(resized,name)(resized._features(wide)))


def test_runtime_calibration_uses_one_time_subset_and_restores_residency(tmp_path):
    from PIL import Image
    import numpy as np
    from core.models.ming_image_01.vae_alignment import calibrate
    class SyntheticVAE(nn.Module):
        def __init__(self,channels,scale):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))
            self.device = torch.device("cpu")
            self.dtype = torch.float32
            self.config = SimpleNamespace(shift_factor=0.,scaling_factor=1.)
            self.channels, self.scale = channels,scale
        def encode(self,pixels):
            z = torch.nn.functional.avg_pool2d(pixels.squeeze(2),self.scale)
            z = z.repeat(1,self.channels//4,1,1).unsqueeze(2)
            return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda:z))
    model = create_transformer(config("zero_masked"),device="cpu",dtype=torch.float32)
    resize_backbone(model,64)
    model.add_module("vae_latent_bridge",MingVAEAlignment(declaration()))
    paths = []
    for i in range(2):
        path = tmp_path/f"synthetic_{i}.png"
        Image.fromarray(np.random.default_rng(i).integers(0,256,(128,128,4),dtype=np.uint8),"RGBA").save(path)
        paths.append({"image_path":str(path)})
    target = SyntheticVAE(64,16)
    components = SimpleNamespace(transformer=model,conditioner=nn.Linear(2,2))
    events = []
    trainer = SimpleNamespace(config={"vae_swap_calibration_samples":2,"vae_swap_calibration_resolution":128},
        vae=target,device="cpu",ming_image_01_components=components,
        _ming_vae_calibration_source=SyntheticVAE(16,8),_check_stop_requested=lambda:None,
        _vae_alignment_progress_callback=lambda **event:events.append(event))
    calibrate(trainer,[SimpleNamespace(items=paths)])
    assert model.vae_latent_bridge.ready and not hasattr(trainer,"_ming_vae_calibration_source")
    assert next(model.parameters()).device.type == "cpu"
    assert [e["step"] for e in events] == [1,2,3,4]
    calibrate(trainer,[])  # Restoration never opens the old calibration subset again.


def test_artifact_requires_portable_bridge_and_expanded_io(package):
    import json
    from core.models.ming_image_01.artifact import inspect_artifact
    dit_path,vae_path = package/"transformer"/"config.json",package/"vae"/"config.json"
    dit = json.loads(dit_path.read_text())
    dit.update(in_channels=64,vae_alignment=declaration())
    dit_path.write_text(json.dumps(dit))
    vae = json.loads(vae_path.read_text())
    vae.update(z_dim=64)
    vae_path.write_text(json.dumps(vae))
    pair = initialize(MingVAEAlignment(declaration()))
    save_file(pair.state_dict(),str(package/WEIGHTS))
    artifact = inspect_artifact(package)
    model = create_transformer(config("zero_masked"),device="cpu",dtype=torch.float32)
    resize_backbone(model,64)
    native_state = model.state_dict()
    fresh = create_transformer(dict(model.config),device="cpu",dtype=torch.float32)
    fresh.load_state_dict(native_state,strict=True)
    restore(fresh,artifact,torch.float32)
    assert fresh.vae_latent_bridge.ready and fresh.in_channels == 64
    (package/WEIGHTS).unlink()
    with pytest.raises((ValueError,FileNotFoundError)):
        inspect_artifact(package)
