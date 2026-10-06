"""Validate real VAE loading/calibration/restoration using generated pixel patterns."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"backend"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-vae",required=True)
    parser.add_argument("--target",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--execute",action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print("Dry run: generated RGBA patterns only; no dataset access")
        return
    import torch
    from core.models.common.vae_source import resolve_vae_source
    from core.models.ming_image_01.vendor.diffusion.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
    from core.models.ming_image_01.vae_alignment import (MingVAEAlignment,NormalizedFrameVAE,
        calibration_grid,load_saved_vae)
    from core.models.ming_image_01.training_data import encode_frames
    from core.models.components.vae_alignment import SpatialLatentBridge,SpatialBridgeCalibrator
    torch.manual_seed(419)
    x = torch.linspace(-1,1,256)
    y,x = torch.meshgrid(x,x,indexing="ij")
    patterns = [torch.stack((torch.sin(x*(i+1)),torch.cos(y*(i+2)),torch.sin((x+y)*(i+3)),
                            torch.ones_like(x))) for i in range(8)]
    old = AutoencoderKLQwenImage.from_pretrained(args.source_vae,torch_dtype=torch.float32,
        local_files_only=True).requires_grad_(False).eval().to("cuda")
    with torch.inference_mode():
        legacy = [encode_frames(old,p[None,:,None]).cpu() for p in patterns]
    del old
    torch.cuda.empty_cache()
    resolved = resolve_vae_source(args.target)
    declaration = dict(version=1,method="paired_spatial",token_policy="target_native",source=args.target,
        old_channels=16,new_channels=resolved.latent_channels,old_scale=8,new_scale=resolved.scale_factor,
        input_channels=int(resolved.config.get("in_channels",resolved.config.get("input_channels",3))),
        ndim=resolved.ndim,norm=resolved.norm,norm_pack=resolved.norm_pack,target_identity=resolved.latent_hash)
    target = NormalizedFrameVAE(resolved.load_module(torch_dtype=torch.float32),declaration).eval().to("cuda")
    del resolved
    incoming = SpatialLatentBridge(declaration["new_channels"],16)
    outgoing = SpatialLatentBridge(16,declaration["new_channels"])
    infit,outfit = SpatialBridgeCalibrator(incoming,ridge=1e-2),SpatialBridgeCalibrator(outgoing,ridge=1e-2)
    with torch.inference_mode():
        for i,p in enumerate(patterns):
            z = encode_frames(target,p[None,:,None]).cpu()
            coarse = calibration_grid(legacy[i],z.shape[-2:])
            infit.add(z,coarse)
            outfit.add(coarse,z)
            print(f"Synthetic pair {i+1}/{len(patterns)}",flush=True)
    infit.finish(); outfit.finish()
    pair = MingVAEAlignment(declaration)
    pair.initialize_from_pair(incoming,outgoing)
    with torch.inference_mode():
        pixels = target.decode(z.to("cuda"),return_dict=False)[0]
        assert pixels.shape == (1,4,1,256,256) and torch.isfinite(pixels).all()
    target.cpu()
    torch.cuda.empty_cache()
    local = Path(args.output).resolve().parent
    local.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vae-roundtrip-",dir=local) as directory:
        target.save_pretrained(directory)
        restored = load_saved_vae(directory,declaration,torch.float32).eval().to("cuda")
        with torch.inference_mode():
            restored_z = encode_frames(restored,patterns[-1][None,:,None]).cpu()
        torch.testing.assert_close(restored_z,z)
    report = dict(target_channels=declaration["new_channels"],target_scale=declaration["new_scale"],
        target_latent_shape=list(z.shape),calibrated=pair.ready,portable_vae_roundtrip=True,
        finite_rgba_decode=True,scope="synthetic pixel patterns, real VAE weights; no image quality acceptance")
    Path(args.output).write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__": main()
