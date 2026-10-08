"""Synthetic image-resolution ablation of DiT training; excludes real VAE encode/decode."""
import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import psutil
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, required=True)
    args = parser.parse_args()
    if args.samples < 1:
        raise ValueError("samples must be positive")
    from core.models.anima.anima_loader import load_anima_dit
    from core.models.anima.residual_unet import restore_topology, default_topology, native_config
    from core.models.anima.partition_global import configure_training
    from core.models.anima.checkpointing import configure
    from core.models.components.latent_io import resize_latent_io
    from core.models.components.wiring import ANIMA_LATENT_IO
    from core.models.components.spatial_swap import install
    import bitsandbytes as bnb

    config = json.loads(Path(args.params).read_text(encoding="utf-8-sig"))
    results = []
    for rewired, swapped in ((False, False), (True, False), (False, True), (True, True)):
        torch.manual_seed(174)
        model = load_anima_dit(args.model, dtype=torch.bfloat16)
        channels, scale = (64, 16) if swapped else (16, 8)
        if swapped:
            resize_latent_io(model, ANIMA_LATENT_IO, channels)
            install(model, dict(version=1, method="paired_spatial", token_policy="target_native",
                source="synthetic:channel-smoke", old_channels=16, new_channels=64,
                old_scale=8, new_scale=16, input_channels=3, ndim=5, norm="identity", norm_pack=1))
            for bridge in (model.vae_latent_bridge.input, model.vae_latent_bridge.output):
                weight = torch.zeros(bridge.projection.in_channels*9, bridge.projection.out_channels)
                for channel in range(min(bridge.source_channels, bridge.target_channels)):
                    weight[channel*9+4, channel] = 1
                bridge.initialize(weight, torch.zeros(bridge.projection.out_channels))
        if rewired:
            restore_topology(model, default_topology(native_config(model)))
        configure_training(model, config)
        model.to(device="cuda", dtype=torch.bfloat16).train().requires_grad_(True)
        model.enable_gradient_checkpointing(
            cpu_offload=bool(config.get("cpu_offload_checkpointing")),
            async_offload=bool(config.get("async_cpu_offload_checkpointing")))
        configure(model, config)
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-5)
        text = torch.randn(1, 5, 1024, device="cuda", dtype=torch.bfloat16)
        aux = dict(target_input_ids=torch.tensor([[1, 2, 3]], device="cuda"),
            target_attention_mask=torch.ones(1, 3, device="cuda", dtype=torch.long),
            source_attention_mask=torch.ones(1, 5, device="cuda", dtype=torch.long))
        for pixels in (512, 1024):
            side = pixels // scale
            measurements = []
            for iteration in range(args.samples+1):
                optimizer.zero_grad(set_to_none=True)
                clean = torch.randn(1, channels, 1, side, side, device="cuda", dtype=torch.bfloat16)
                noise = torch.randn_like(clean)
                noisy, target = .5*clean + .5*noise, noise-clean
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
                start = time.perf_counter()
                events[0].record()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    predicted = model(noisy, torch.tensor([.5], device="cuda", dtype=torch.bfloat16), text,
                        padding_mask=torch.zeros(1, 1, side, side, device="cuda", dtype=torch.bfloat16), **aux)
                    loss = (predicted.float()-target.float()).square().mean()
                events[1].record()
                loss.backward()
                events[2].record()
                optimizer.step()
                events[3].record()
                torch.cuda.synchronize()
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite smoke loss")
                if iteration:
                    measurements.append(dict(forward_ms=events[0].elapsed_time(events[1]),
                        backward_ms=events[1].elapsed_time(events[2]),
                        optimizer_ms=events[2].elapsed_time(events[3]),
                        iteration_ms=(time.perf_counter()-start)*1000,
                        peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                        peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
                        rss_gib=psutil.Process().memory_info().rss/1024**3))
                del loss, predicted, noisy, target, clean, noise, events
            row = dict(unet=rewired, vae_swap=swapped, pixels=pixels,
                latent_channels=channels, latent_scale=scale, image_tokens=(side//2)**2,
                parameters=sum(p.numel() for p in model.parameters()),
                median={key: statistics.median(m[key] for m in measurements) for key in measurements[0]},
                samples=measurements)
            results.append(row)
            print(json.dumps(row), flush=True)
        del model, optimizer, text, aux
        gc.collect()
        torch.cuda.empty_cache()
    report = dict(model=args.model, synthetic=True, real_vae_calibration=False,
        excludes=["VAE encoding/decoding", "caption encoding", "IO", "spatial partition training",
            "calibrated Gaussian noise complement"],
        dtype="bf16", batch_size=1, warmup_per_resolution=1,
        checkpoint_blocks=config.get("dit_gradient_checkpointing_blocks"),
        optimizer="AdamW8bit", gpu=torch.cuda.get_device_name(), results=results)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
