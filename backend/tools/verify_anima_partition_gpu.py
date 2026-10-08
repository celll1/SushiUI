"""Real-weight synthetic GPU smoke; never starts a saved training run or reads images."""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import psutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from core.models.anima.anima_loader import load_anima_dit
    from core.models.anima.residual_unet import restore_topology, default_topology, native_config
    from core.models.anima.partition_global import configure_training, NAME
    from core.models.anima.partition_training import backward_regions, training_plan
    from core.models.anima.checkpointing import configure
    from core.models.components.latent_io import resize_latent_io
    from core.models.components.wiring import ANIMA_LATENT_IO
    from core.models.components.spatial_swap import install
    import bitsandbytes as bnb

    torch.manual_seed(174)
    config = json.loads(Path(args.params).read_text(encoding="utf-8-sig"))
    model = load_anima_dit(args.model, dtype=torch.bfloat16)
    resize_latent_io(model, ANIMA_LATENT_IO, 64)
    install(model, dict(version=1, method="paired_spatial", token_policy="target_native",
        source="synthetic:channel-smoke", old_channels=16, new_channels=64,
        old_scale=8, new_scale=16, input_channels=3, ndim=5, norm="identity", norm_pack=1))
    for bridge in (model.vae_latent_bridge.input, model.vae_latent_bridge.output):
        weight = torch.zeros(bridge.projection.in_channels*9, bridge.projection.out_channels)
        for channel in range(min(bridge.source_channels, bridge.target_channels)):
            weight[channel*9+4, channel] = 1
        bridge.initialize(weight, torch.zeros(bridge.projection.out_channels))
    restore_topology(model, default_topology(native_config(model)))
    adapter = configure_training(model, config)
    model = model.to(device="cuda", dtype=torch.bfloat16).train().requires_grad_(True)
    model.enable_gradient_checkpointing(
        cpu_offload=bool(config.get("cpu_offload_checkpointing")),
        async_offload=bool(config.get("async_cpu_offload_checkpointing")))
    configure(model, config)
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-5)
    metrics = {}
    owner = SimpleNamespace(config=config, transformer=model, device=torch.device("cuda"),
        training_dtype=torch.bfloat16, mixed_precision=True, reconstruction_loss_weight=0.,
        _current_epoch=0, _current_batch_position=0,
        log_extra_metric=lambda key, value: metrics.__setitem__(key, value))
    text = torch.randn(1, 5, 1024, device="cuda", dtype=torch.bfloat16)
    aux = dict(source_mask=torch.ones(1, 5, device="cuda", dtype=torch.long),
        t5_input_ids=torch.tensor([[1, 2, 3]], device="cuda"),
        t5_attn_mask=torch.ones(1, 3, device="cuda", dtype=torch.long))
    results = []
    for count in (2, 4):
        config.update(dit_partition_mode="fixed", dit_partition_fixed_count=count,
                      dit_partition_profile=True)
        clean = torch.randn(1, 64, 32, 48, device="cuda", dtype=torch.bfloat16)
        noise = torch.randn_like(clean)
        optimizer.zero_grad(set_to_none=True)
        start = time.perf_counter()
        losses = backward_regions(owner, clean=clean, noise=noise,
            sigma=torch.tensor([.5], device="cuda", dtype=torch.bfloat16),
            prompt_embeds=text, anima_aux=aux, plan=training_plan(owner, clean), backward_scale=1.)
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise RuntimeError(f"Nonfinite gradient: {name}")
        global_grad = float(adapter.out.weight.grad.float().norm())
        if not global_grad > 0:
            raise RuntimeError("Global output projection received no gradient")
        if count == 4:
            for name, parameter in adapter.named_parameters():
                if parameter.grad is None or not float(parameter.grad.float().norm()) > 0:
                    raise RuntimeError(f"Learned global summary received no gradient: {name}")
        optimizer.step()
        torch.cuda.synchronize()
        results.append(dict(count=count, losses=losses, wall_seconds=time.perf_counter()-start,
            global_output_grad_norm=global_grad, metrics=dict(metrics),
            rss_gib=psutil.Process().memory_info().rss/1024**3))
        print(json.dumps(results[-1]), flush=True)
    report = dict(model=args.model, synthetic=True, real_vae_calibration=False,
        topology=model._residual_unet_spec, global_adapter=getattr(model, NAME).metadata(),
        target_channels=64, target_scale=16, optimizer="AdamW8bit", results=results)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
