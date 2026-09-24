"""Train the SDXL style bridge on dataset 39 without changing either base model."""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from diffusers import DDPMScheduler
from PIL import Image, ImageOps
from transformers import Siglip2ImageProcessor

WORKTREE = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(WORKTREE / "backend"))

from core.style_encoder.sdxl_ip_adapter import (
    SDXLStyleIPAdapter, load_sdxl_style_ip_adapter, save_sdxl_style_ip_adapter,
)
from core.style_encoder.sdxl_ip_runtime import load_sdxl_style_pipeline
from core.style_encoder.sdxl_ip_training import install_style_ip_training_processors, sdxl_style_ip_denoising_loss
from core.style_encoder.style_artifacts import load_style_encoder


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def image_at(path: str, size: int) -> torch.Tensor:
    with Image.open(path) as source:
        image = ImageOps.fit(source.convert("RGB"), (size, size), method=Image.Resampling.LANCZOS)
    values = np.asarray(image, dtype=np.float32).copy()
    return torch.from_numpy(values).permute(2, 0, 1).unsqueeze(0).to("cuda", dtype=torch.bfloat16) / 127.5 - 1


def reference_embedding(path: str, model, processor, patch_budget: int) -> torch.Tensor:
    with Image.open(path) as source:
        image = source.convert("RGB")
        inputs = processor(images=[image], max_num_patches=patch_budget, return_tensors="pt")
    inputs = {key: value.to("cuda") for key, value in inputs.items()}
    with torch.no_grad():
        return model.embed(inputs["pixel_values"].to(dtype=torch.bfloat16),
                           inputs["pixel_attention_mask"], inputs["spatial_shapes"]).float()


def select_pair(train: list[dict], by_artist: dict[int, list[dict]], step: int, seed: int):
    epoch, position = divmod(step, len(train))
    order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(len(train))
    target = train[int(order[position])]
    candidates = [item for item in by_artist[target["artist"]] if item["row"] != target["row"]]
    content = set(target["copyrights"]) | {1000000 + value for value in target["characters"]}
    different = [item for item in candidates if content.isdisjoint(
        set(item["copyrights"]) | {1000000 + value for value in item["characters"]}
    )]
    if different:
        candidates = different
    rng = np.random.default_rng(np.random.SeedSequence([seed, step, 97]))
    return target, candidates[int(rng.integers(len(candidates)))]


def atomic_json(path: Path, value: dict) -> None:
    staged = path.with_name(path.name + ".tmp")
    staged.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(staged, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.steps < 1 or args.resolution < 256 or args.resolution % 64 or args.save_every < 1:
        parser.error("steps/save-every must be positive and resolution must be a multiple of 64 >= 256")

    model_path = REPO / "models" / "NoobAI-XL-Vpred-v1.0.safetensors"
    encoder_path = WORKTREE / "outputs/style_hsic_ab_dataset39/baseline/best_style_encoder.safetensors"
    metadata_path = encoder_path.with_name("best_metadata.json")
    manifest_path = WORKTREE / "outputs/sdxl_style_ip_dataset39/manifest.jsonl"
    output = WORKTREE / "outputs/sdxl_style_ip_dataset39"
    output.mkdir(parents=True, exist_ok=True)
    model_sha, encoder_sha = digest(model_path), digest(encoder_path)
    manifest = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    train = [item for item in manifest if item["split"] == 0 and item["eligible"]]
    by_artist = defaultdict(list)
    for item in train:
        by_artist[item["artist"]].append(item)
    if len(train) != 1477 or any(len(items) < 2 for items in by_artist.values()):
        raise ValueError(f"Expected 1477 eligible training images, found {len(train)}")
    print(f"Training from {len(train)} eligible images, {len(by_artist)} artists", flush=True)

    pipe = load_sdxl_style_pipeline(str(model_path))
    pipe.to("cuda")
    pipe.unet.eval().requires_grad_(False)
    pipe.vae.eval().requires_grad_(False)
    pipe.text_encoder.eval().requires_grad_(False)
    pipe.text_encoder_2.eval().requires_grad_(False)
    with metadata_path.open(encoding="utf-8") as handle:
        encoder_metadata = json.load(handle)
    style_model = load_style_encoder(str(encoder_path), "sdpa").to("cuda", dtype=torch.bfloat16).eval().requires_grad_(False)
    style_processor = Siglip2ImageProcessor.from_dict(encoder_metadata["preprocessing"])
    patch_budget = int(encoder_metadata["style_max_num_patches"])
    scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                              num_train_timesteps=1000, clip_sample=False, prediction_type="v_prediction")
    install_style_ip_training_processors(pipe.unet)
    state_path = output / "latest_state.json"
    weights_path = output / "latest_style_ip.safetensors"
    optimizer_path = output / "latest_optimizer.pt"
    identity = {"model_sha256": model_sha, "style_encoder_sha256": encoder_sha,
                "manifest_sha256": digest(manifest_path), "resolution": args.resolution, "seed": args.seed}
    if state_path.exists():
        saved = json.loads(state_path.read_text(encoding="utf-8"))
        if any(saved.get(key) != value for key, value in identity.items()):
            raise ValueError("Saved SDXL style bridge run identity does not match")
        adapter = load_sdxl_style_ip_adapter(pipe.unet, str(weights_path),
                                             style_encoder_sha256=encoder_sha, sdxl_model_sha256=model_sha)
        start = int(saved["step"])
    else:
        torch.manual_seed(args.seed)
        adapter = SDXLStyleIPAdapter(pipe.unet)
        start = 0
    adapter.to("cuda", dtype=torch.float32).train()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4, weight_decay=0.01)
    if start:
        optimizer.load_state_dict(torch.load(optimizer_path, map_location="cuda", weights_only=True))
    print(f"SDXL bridge start={start} target={args.steps}", flush=True)

    def save(step: int) -> None:
        staged = optimizer_path.with_name(optimizer_path.name + ".tmp")
        torch.save(optimizer.state_dict(), staged)
        os.replace(staged, optimizer_path)
        save_sdxl_style_ip_adapter(adapter, str(weights_path), style_encoder_sha256=encoder_sha,
                                   sdxl_model_sha256=model_sha)
        atomic_json(state_path, {**identity, "step": step})
        print(f"Saved step {step}", flush=True)

    for step in range(start, args.steps):
        began = time.perf_counter()
        target, reference = select_pair(train, by_artist, step, args.seed)
        generator = torch.Generator(device="cuda").manual_seed(args.seed + step)
        with torch.no_grad():
            pixels = image_at(target["path"], args.resolution)
            latent = pipe.vae.encode(pixels).latent_dist.sample(generator=generator)
            latent *= pipe.vae.config.scaling_factor
            style = reference_embedding(reference["path"], style_model, style_processor, patch_budget)
            prompt, _, pooled, _ = pipe.encode_prompt(
                target["caption"], device="cuda", do_classifier_free_guidance=False,
            )
            noise = torch.randn(latent.shape, device="cuda", dtype=latent.dtype, generator=generator)
            timestep = torch.randint(0, scheduler.config.num_train_timesteps, (1,),
                                     device="cuda", generator=generator)
        optimizer.zero_grad(set_to_none=True)
        loss = sdxl_style_ip_denoising_loss(
            pipe.unet, scheduler, adapter, clean_latents=latent, style_embedding=style,
            prompt_embeds=prompt, added_cond_kwargs={
                "text_embeds": pooled,
                "time_ids": torch.tensor([[args.resolution, args.resolution, 0, 0,
                                           args.resolution, args.resolution]], device="cuda", dtype=torch.bfloat16),
            }, timesteps=timestep, noise=noise,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at step {step + 1}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        print(json.dumps({"step": step + 1, "loss": float(loss.detach()),
                          "seconds": round(time.perf_counter() - began, 3),
                          "vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}), flush=True)
        if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
            save(step + 1)


if __name__ == "__main__":
    main()
