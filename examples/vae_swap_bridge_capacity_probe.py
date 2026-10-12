"""Decode what a saved paired_spatial bridge can express, without loading the DiT.

Usage: venv/Scripts/python.exe examples/vae_swap_bridge_capacity_probe.py CHECKPOINT IMAGE
IMAGE must be a rating:general picture. Outputs go to ignored local/verification/.
"""
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root/"backend"))
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open
from core.models.common.vae_source import resolve_vae_source
from core.models.components.latent_adapter import NormalizedImageVAE
from core.models.anima.anima_loader import load_qwen_image_vae
from core.models.anima.anima_pipeline_ops import vae_encode_image, vae_decode_latents

checkpoint, image_path = sys.argv[1], sys.argv[2]
out = root/"local/verification/vae_swap_bridge_capacity"
out.mkdir(parents=True, exist_ok=True)
with safe_open(checkpoint, "pt") as f:
    declaration = json.loads(f.metadata()["component.vae.alignment"])
    tensor = lambda k: f.get_tensor("net.vae_latent_bridge."+k).float().cuda()
    a_in, b_in = tensor("input.projection.weight"), tensor("input.projection.bias")
    a_out, b_out = tensor("output.projection.weight"), tensor("output.projection.bias")
old = declaration["old_channels"]
target = NormalizedImageVAE(resolve_vae_source(declaration["source"], arch="anima")
    .load_module(torch_dtype=torch.bfloat16), declaration).eval().requires_grad_(False).cuda()
native = load_qwen_image_vae(str(Path(r"M:\model\anima\split_files\vae\qwen_image_vae.safetensors")),
    dtype=torch.bfloat16).eval().requires_grad_(False).cuda()


def r2(estimate, reference):
    error = ((estimate-reference)**2).mean((0, 2, 3))
    variance = ((reference-reference.mean((2, 3), keepdim=True))**2).mean((0, 2, 3))
    return 1-error/variance


source = Image.open(image_path).convert("RGB")
report = {}
torch.manual_seed(0)
for scale in (1, 2):
    width, height = source.size[0]*scale, source.size[1]*scale
    image = source.resize((width, height), Image.Resampling.LANCZOS)
    with torch.no_grad():
        y = vae_encode_image(target, image, "cuda", torch.bfloat16).float().squeeze(2)
        half = image.resize((width//2, height//2), Image.Resampling.LANCZOS)
        x_source = vae_encode_image(native, half, "cuda", torch.bfloat16).float().squeeze(2)
        backbone = F.conv2d(y, a_in, b_in, padding=1)
        backbone[:, old:] = 0
        bottleneck = F.conv2d(backbone, a_out, b_out, padding=1)
        from_source = F.conv2d(torch.cat([x_source, torch.zeros_like(y[:, old:])], 1), a_out, b_out, padding=1)
        perfect_passthrough = bottleneck.clone()
        perfect_passthrough[:, old:] = y[:, old:]
        arms = dict(codec_roundtrip=y, from_half_resolution_source=from_source,
                    source_channel_bottleneck=bottleneck, perfect_index_passthrough=perfect_passthrough)
        report[f"x{scale}"] = {}
        for name, latent in arms.items():
            score = r2(latent, y)
            report[f"x{scale}"][name] = dict(r2_first=score[:old].mean().item(), r2_rest=score[old:].mean().item(),
                                            relative_error=((latent-y).norm()/y.norm()).item())
            vae_decode_latents(target, latent.unsqueeze(2).to(torch.bfloat16))[0].save(out/f"x{scale}_{name}.png")
(out/"report.json").write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1))
