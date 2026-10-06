"""Opt-in Ming/Qwen VAE alignment probe; never loads a DiT or edits a run.

Run with the repo venv and --execute after releasing training GPU residency.
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps

from core.experiments.vae_alignment import AffineMap, SpatialMap, approved_image_paths, fit_affine, fit_orthogonal, fit_spatial, metrics, pack, unpack


@torch.no_grad()
def embedding_oracle(transformer_dir, maps_path, output):
    from safetensors import safe_open
    from core.experiments.vae_alignment import fold_input, ming_patch_order
    directory = Path(transformer_dir)
    index = json.loads((directory / "diffusion_pytorch_model.safetensors.index.json").read_text())
    tensors = {}
    for name in ("all_x_embedder.2-1.weight", "all_x_embedder.2-1.bias"):
        with safe_open(directory / index["weight_map"][name], framework="pt", device="cpu") as handle:
            tensors[name] = handle.get_tensor(name).double()
    original = torch.nn.Linear(64, tensors["all_x_embedder.2-1.weight"].shape[0]).double()
    with torch.no_grad():
        original.weight.copy_(tensors["all_x_embedder.2-1.weight"])
        original.bias.copy_(tensors["all_x_embedder.2-1.bias"])
    record = torch.load(maps_path, weights_only=True)["qwen_to_ming"]["paired_linear"]
    mapping = ming_patch_order(AffineMap(**record))
    weight, bias = fold_input(original, mapping)
    features = torch.randn(128, 64, generator=torch.Generator().manual_seed(731)).double()
    expected = original(features @ mapping.weight + mapping.bias)
    actual = features @ weight.T + bias
    error = float((actual - expected).abs().max())
    if error > 1e-9:
        raise AssertionError(f"Fold mismatch: {error}")
    reference32 = original.float()(features.float() @ mapping.weight.float() + mapping.bias.float())
    actual32 = features.float() @ weight.float().T + bias.float()
    error32 = float((actual32 - reference32).abs().max())
    if error32 > 1e-4:
        raise AssertionError(f"FP32 fold mismatch: {error32}")
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    torch.save({"weight": weight.float(), "bias": bias.float(), "patch_size": 1,
                "scope": "experimental input embedding only; no complete checkpoint"}, output / "folded_input.pt")
    (output / "embedding_oracle.json").write_text(json.dumps({"max_abs_error_fp64": error,
        "max_abs_error_fp32": error32,
        "weight_shape": list(weight.shape), "new_patch_size": 1, "token_reduction": False}, indent=2))
    print(f"Actual Ming input embedding fold max error: {error}", flush=True)


def pixels(path, side):
    with Image.open(path) as image:
        image = ImageOps.fit(ImageOps.exif_transpose(image).convert("RGBA"), (side, side),
                             method=Image.Resampling.LANCZOS)
        return torch.from_numpy(np.array(image).copy()).permute(2, 0, 1).float() / 127.5 - 1


def alpha_probe(index, side):
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, side), torch.linspace(-1, 1, side), indexing="ij")
    alpha = ((xx + .3 * torch.sin(yy * (index + 2))) * (index % 3 + 1)).clamp(-1, 1)
    return torch.stack((torch.sin(xx * (index + 1)), torch.cos(yy * (index + 2)),
                        torch.sin((xx + yy) * (index + 3)), alpha))


def normalize(z, vae, kind, inverse=False):
    if kind == "ming":
        shift, scale = float(vae.config.shift_factor), float(vae.config.scaling_factor)
        return z / scale + shift if inverse else (z - shift) * scale
    mean = torch.as_tensor(vae.config.latents_mean, device=z.device, dtype=z.dtype).view(1, -1, 1, 1, 1)
    std = torch.as_tensor(vae.config.latents_std, device=z.device, dtype=z.dtype).view(1, -1, 1, 1, 1)
    return z * std + mean if inverse else (z - mean) / std


def load_vae(args, kind):
    # Some imported pipelines query CUDA capability during import, before execution.
    capability = torch.cuda.get_device_capability
    torch.cuda.get_device_capability = lambda *a, **k: (8, 9)
    try:
        if kind == "ming":
            from core.models.ming_image_01.vendor.diffusion.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
            vae = AutoencoderKLQwenImage.from_pretrained(args.ming_vae, torch_dtype=torch.float32,
                                                        local_files_only=True)
        else:
            from core.models.qwen_image_21.artifact import load_manifest, load_vae as load_artifact_vae
            manifest = load_manifest(args.qwen_model)
            vae = load_artifact_vae(manifest.vae, manifest.vae_config, torch.float32)
    finally:
        torch.cuda.get_device_capability = capability
    vae.requires_grad_(False).eval().to(args.device)
    return vae


@torch.inference_mode()
def encode_all(args, images, kind):
    vae = load_vae(args, kind)
    result = []
    start = time.perf_counter()
    for index, image in enumerate(images):
        value = image[None, :, None].to(args.device)
        z = vae.encode(value).latent_dist.mode()
        result.append(normalize(z, vae, kind).squeeze(2).cpu())
        if index % 8 == 0:
            print(f"{kind} encode {index + 1}/{len(images)}", flush=True)
    torch.cuda.synchronize() if args.device == "cuda" else None
    seconds = time.perf_counter() - start
    del vae
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(result), seconds


def build_maps(source, target, rgb):
    channels = source.shape[1]
    zero = AffineMap(torch.zeros(channels, channels).double(), torch.zeros(channels).double())
    identity = AffineMap(torch.eye(channels).double(), torch.zeros(channels).double())
    return {
        "zero": zero,
        "mean_only": AffineMap(zero.weight, target.double().mean((0, 2, 3))),
        "packed_identity": identity,
        "paired_rgb": fit_affine(source, rgb).then(fit_affine(rgb, target)),
        "paired_linear": fit_affine(source, target),
        "paired_spatial": fit_spatial(source, target),
        "orthogonal": fit_orthogonal(source, target),
    }


def fixed_rgb_map():
    from core.utils.taesd import TAESDManager
    cls = TAESDManager
    q = torch.tensor(cls._QWEN_IMAGE_21_LATENT_RGB_FACTORS).double()
    qb = torch.tensor(cls._QWEN_IMAGE_21_LATENT_RGB_BIAS).double()
    m = torch.tensor(cls._WAN21_LATENT_RGB_FACTORS).double()
    mb = torch.tensor(cls._WAN21_LATENT_RGB_BIAS).double()
    inverse = torch.linalg.pinv(m)
    return AffineMap((q @ inverse).repeat_interleave(4, dim=1),
                     ((qb - mb) @ inverse).repeat_interleave(4))


def image_metrics(prediction, target):
    return {label: float(-10 * torch.log10((prediction[:, sl] - target[:, sl]).square().mean().clamp_min(1e-12) / 4))
            for label, sl in (("rgb_psnr", slice(0, 3)), ("alpha_psnr", slice(3, 4)))}


@torch.inference_mode()
def decode_comparison(args, kind, truth, predicted, selected, images):
    vae = load_vae(args, kind)
    results, scores = {}, {}
    for name, latents in {"vae_self": truth, **predicted}.items():
        decoded = []
        for index in selected:
            z = latents[index:index + 1]
            if kind == "ming":
                z = unpack(z)
            z = normalize(z[:, :, None].to(args.device), vae, kind, inverse=True)
            value = vae.decode(z).sample.squeeze(2).cpu().clamp(-1, 1)
            if not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite {kind}/{name} decode")
            decoded.append(value)
        results[name] = torch.cat(decoded)
        scores[name] = image_metrics(results[name], torch.stack([images[i] for i in selected]))
        print(f"{kind} decode {name}: {scores[name]}", flush=True)
    for name in predicted:
        scores[name]["vs_vae_self"] = image_metrics(results[name], results["vae_self"])
    del vae
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()
    return results, scores


def montage(path, images, selected, decoded):
    columns = {"original": torch.stack([images[i] for i in selected]), **decoded}
    side = 160
    canvas = Image.new("RGB", (side * len(columns), (side + 60) * len(selected)), "#dddddd")
    draw = ImageDraw.Draw(canvas)
    for column, (name, values) in enumerate(columns.items()):
        for row, value in enumerate(values):
            array = ((value.permute(1, 2, 0) + 1) * 127.5).clamp(0, 255).byte().numpy()
            image = Image.fromarray(array, "RGBA").resize((side, side))
            background = Image.new("RGBA", (side, side), "white")
            background.alpha_composite(image)
            x, y = column * side, row * (side + 60)
            canvas.paste(background.convert("RGB"), (x, y + 20))
            canvas.paste(image.getchannel("A").resize((side, 32)).convert("RGB"), (x, y + 180))
            draw.text((x + 3, y + 3), name, fill="black")
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ming-vae", required=True)
    parser.add_argument("--qwen-model", required=True)
    parser.add_argument("--image-list", help="User-approved JSON list: unique general image paths")
    parser.add_argument("--synthetic-only", action="store_true", help="Use only generated nonsexual RGBA patterns; read no dataset images")
    parser.add_argument("--output", required=True)
    parser.add_argument("--side", type=int, default=256)
    parser.add_argument("--train-count", type=int, default=32)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--reuse-maps", help="Use previously fitted maps unchanged, e.g. at another resolution")
    parser.add_argument("--embedding-only", help="CPU oracle: path to a bf16 Ming transformer directory")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(vars(args), indent=2)); return
    if args.embedding_only:
        if not args.reuse_maps:
            raise ValueError("Embedding oracle requires --reuse-maps")
        embedding_oracle(args.embedding_only, args.reuse_maps, args.output)
        return
    if args.side % 32:
        raise ValueError("Side must be divisible by 32")
    torch.set_num_threads(8)
    torch.manual_seed(731)
    if bool(args.image_list) == bool(args.synthetic_only):
        raise ValueError("Select an explicitly approved image list OR synthetic-only")
    records = json.loads(Path(args.image_list).read_text(encoding="utf-8-sig")) if args.image_list else []
    paths = approved_image_paths(records)
    if args.train_count < 1 or (paths and (len(set(paths)) != len(paths) or args.train_count >= len(paths))):
        raise ValueError("Need distinct train and held-out image paths")
    natural = ([pixels(path, args.side) for path in paths] if paths else
               [alpha_probe(i + 20, args.side) for i in range(args.train_count + 16)])
    images = natural[:args.train_count] + [alpha_probe(i, args.side) for i in range(8)]
    split = len(images)
    images += natural[args.train_count:] + [alpha_probe(i, args.side) for i in range(8, 12)]
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    ming, mt = encode_all(args, images, "ming")
    qwen, qt = encode_all(args, images, "qwen")
    old = pack(ming)
    if old.shape != qwen.shape:
        raise ValueError(f"Expected matching 64-channel grids: {old.shape}, {qwen.shape}")
    rgb = F.interpolate(torch.stack(images)[:, :3], size=qwen.shape[-2:], mode="area")
    report = {"scope": "clean posterior-mode alignment; no DiT, no diffusion training or speed claim",
              "side": args.side, "natural_paths": paths, "synthetic_only": args.synthetic_only,
              "rating": "general" if paths else "synthetic",
              "split": split, "heldout": len(images) - split,
              "alpha_probes": {"train": 8, "heldout": 4},
              "fixed_preview_rgb_note": "WAN RGB coefficients are only an approximate proxy for Ming, not a calibrated RGBA inverse",
              "encode_seconds": {"ming": mt, "qwen": qt}, "directions": {}}
    reused = torch.load(args.reuse_maps, map_location="cpu", weights_only=True) if args.reuse_maps else None
    report["reused_maps"] = args.reuse_maps
    all_maps = {}
    selected = list(range(split, min(split + 4, len(natural) + 8))) + list(range(len(images) - 2, len(images)))
    for direction, source, target, kind in (("qwen_to_ming", qwen, old, "ming"),
                                             ("ming_to_qwen", old, qwen, "qwen")):
        if reused is None:
            maps = build_maps(source[:split], target[:split], rgb[:split])
            if direction == "qwen_to_ming":
                maps["fixed_preview_rgb"] = fixed_rgb_map()
        else:
            maps = {name: (SpatialMap if "kernel_size" in record else AffineMap)(**record)
                    for name, record in reused[direction].items()}
        predictions = {name: mapping(source) for name, mapping in maps.items()}
        summary = {name: {"train": metrics(value[:split], target[:split]),
                           "heldout": metrics(value[split:], target[split:]),
                           **maps[name].diagnostics()} for name, value in predictions.items()}
        decoded, scores = decode_comparison(args, kind, target, predictions, selected, images)
        montage(output / f"{direction}.png", images, selected, decoded)
        report["directions"][direction] = {"latent": summary, "decoded": scores}
        all_maps[direction] = {name: {"weight": mapping.weight, "bias": mapping.bias,
                                      **({"kernel_size": mapping.kernel_size} if isinstance(mapping, SpatialMap) else {})}
                               for name, mapping in maps.items()}
    torch.save(all_maps, output / "alignment_maps.pt")
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["directions"], indent=2), flush=True)


if __name__ == "__main__":
    main()
