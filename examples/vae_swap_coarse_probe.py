"""Synthetic-latent timing probe; never reads dataset images or changes runs."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not args.execute:
        print("Dry run: synthetic 1024px-equivalent grids, native 4096 vs coarse 1024 tokens")
        return
    import torch
    from core.models.ming_image_01.vendor.diffusion.transformer import DiffusionTransformer
    from core.models.ming_image_01.loader import _strict_loading_info
    from core.models.ming_image_01.residual_unet import restore_topology
    from core.rewiring.artifacts import read_directory_topology
    from core.models.ming_image_01.attention import install_attention
    from core.models.ming_image_01.vae_alignment import MingVAEAlignment, resize_backbone
    from core.models.components.vae_alignment import SpatialLatentBridge
    torch.manual_seed(379)
    directory = Path(args.checkpoint) / "transformer"
    print("Loading saved bf16 DiT only; no conditioner or VAE", flush=True)
    model, info = DiffusionTransformer.from_pretrained(str(directory), torch_dtype=torch.bfloat16,
        local_files_only=True, low_cpu_mem_usage=True, output_loading_info=True)
    _strict_loading_info(info, "transformer")
    restore_topology(model, read_directory_topology(directory))
    model.requires_grad_(False).eval().to("cuda")
    install_attention(model, "sdpa")
    cap = [torch.randn(256, model.config.cap_feat_dim, device="cuda", dtype=torch.bfloat16)]
    direct = [torch.randn(256, model.dim, device="cuda", dtype=torch.bfloat16)]
    t = torch.tensor([.5], device="cuda", dtype=torch.bfloat16)

    def measure(channels, grid):
        latent = [torch.randn(channels, 1, grid, grid, device="cuda", dtype=torch.bfloat16)]
        timings = []
        with torch.inference_mode():
            for i in range(5):
                torch.cuda.synchronize()
                start = time.perf_counter()
                output = model(latent, t, cap, cap_feats_2=direct).sample[0]
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-start
                if i >= 2: timings.append(elapsed)
                assert output.shape == latent[0].shape and torch.isfinite(output).all()
        return dict(image_tokens=(grid//2)**2, forward_seconds=statistics.median(timings),
            peak_allocated_bytes=torch.cuda.max_memory_allocated())

    torch.cuda.reset_peak_memory_stats()
    baseline = measure(16,128)
    declaration = dict(version=1, method="paired_spatial", token_policy="target_native",
        source="synthetic:timing", old_channels=16,new_channels=64,old_scale=8,new_scale=16,
        input_channels=4,ndim=5,norm="shift_scale",norm_pack=1)
    resize_backbone(model,64)
    pair = MingVAEAlignment(declaration)
    incoming, outgoing = SpatialLatentBridge(64,16), SpatialLatentBridge(16,64)
    for layer in (incoming,outgoing):
        weight = torch.zeros(layer.projection.out_channels,layer.projection.in_channels,3,3)
        for c in range(16): weight[c,c,1,1] = 1
        layer.initialize(weight.flatten(1).T,torch.zeros(layer.projection.out_channels))
    pair.initialize_from_pair(incoming,outgoing)
    model.add_module("vae_latent_bridge", pair.to(device="cuda",dtype=torch.bfloat16))
    torch.cuda.reset_peak_memory_stats()
    coarse = measure(64,64)
    from core.models.ming_image_01.training import flow_loss
    from core.models.ming_image_01.conditioning import MingConditioning
    model.vae_latent_bridge.requires_grad_(True)
    model.all_x_embedder["2-1"].requires_grad_(True)
    model.all_final_layer["2-1"].requires_grad_(True)
    model.enable_gradient_checkpointing()
    clean = torch.randn(1,64,1,16,16,device="cuda",dtype=torch.bfloat16)
    with torch.autocast("cuda",dtype=torch.bfloat16):
        result = flow_loss(model,clean=clean,noise=torch.randn_like(clean),sigma=t,
            conditioning=MingConditioning(torch.stack(cap),torch.stack(direct)),
            frame_weights=torch.ones(1,1,device="cuda"))
    result.loss.backward()
    extra_grad = model.all_x_embedder["2-1"].weight.grad.reshape(-1,4,64)[:,:,16:]
    assert torch.isfinite(result.loss) and torch.isfinite(extra_grad).all() and extra_grad.abs().sum() > 0
    report = dict(baseline=baseline, coarse=coarse,
        speedup=baseline["forward_seconds"]/coarse["forward_seconds"],
        gpu_backward_finite=True, extra_channel_gradient_nonzero=True,
        scope="saved DiT forward only, synthetic latents/conditioning, SDPA, existing U-Net topology; no quality or full-iteration claim")
    output = Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    main()
