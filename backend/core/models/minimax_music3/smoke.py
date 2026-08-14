"""Standalone MiniMax Music 3 phase-1 smoke check (design doc phase plan, item 1).

Verifies the vendored model modules against the REAL checkpoint's configs and safetensors headers, without loading
the checkpoint's actual weights and without running a generation. Two things this deliberately does NOT do, per the
repo's rules for this phase:

  * it does not load the ~22GB of real checkpoint weights (the language model alone is ~16-32GB depending on dtype)
    -- only `config.json` (tiny JSON) and safetensors HEADERS (metadata only, no tensor bytes) are read for the
    census, and every forward pass in the numeric smoke runs on small RANDOMLY INITIALIZED weights built from the
    real config's *shape* parameters, not the real checkpoint's tensors;
  * it does not run a text-to-music generation end to end -- that needs the LM/depth-decoder resident together
    (~22GB) and is explicitly gated on the loader commit (design doc phase plan, item 2).

This script does NOT run automatically: it is not named `*_test.py` (pytest will not collect it), and it is only
invoked via `python -m core.models.minimax_music3.smoke --model-root <path>` or `python smoke.py --model-root <path>`.
`--model-root` has no default -- per the repo rule against embedding machine-specific paths in tracked files, the
caller must supply the checkpoint location explicitly (e.g. `M:/model/minimax-music3/official` on this machine).

Per the repo's GPU-probe host-RAM rule ("announce HOST RAM peak before it runs"), this prints available host RAM
before touching any file, even though every step here is cheap (config JSON, safetensors headers, and small random
tensors -- expected additional RSS is well under 1 GiB).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)


def _announce_host_ram() -> None:
    try:
        import psutil

        vm = psutil.virtual_memory()
        print(
            f"[minimax_music3.smoke] host RAM: {vm.available / 2**30:.1f} GiB available / "
            f"{vm.total / 2**30:.1f} GiB total. Expected additional usage for this script: well under 1 GiB "
            f"(config JSON + safetensors headers + small random-weight tensors; no real checkpoint weights are "
            f"loaded, no generation is run)."
        )
    except Exception as exc:
        print(f"[minimax_music3.smoke] could not query host RAM via psutil ({exc!r}); proceeding anyway.")


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.pop("_class_name", None)
    cfg.pop("_diffusers_version", None)
    return cfg


def _census(cls, config_path: str, safetensors_paths: List[str], name: str) -> bool:
    """Meta-device `from_config` vs the real safetensors headers: parameter-name and shape parity, no tensor bytes."""
    import torch
    from safetensors import safe_open

    cfg = _load_config(config_path)
    with torch.device("meta"):
        model = cls.from_config(cfg) if hasattr(cls, "from_config") else cls(**cfg)
    model_keys = {k: tuple(v.shape) for k, v in model.state_dict().items()}

    st_keys: Dict[str, Tuple[int, ...]] = {}
    for path in safetensors_paths:
        with safe_open(path, framework="pt") as f:
            for k in f.keys():
                st_keys[k] = tuple(f.get_slice(k).get_shape())

    missing_in_st = sorted(set(model_keys) - set(st_keys))
    missing_in_model = sorted(set(st_keys) - set(model_keys))
    shape_mismatch = [
        (k, model_keys[k], st_keys[k]) for k in set(model_keys) & set(st_keys) if model_keys[k] != st_keys[k]
    ]

    ok = not missing_in_st and not missing_in_model and not shape_mismatch
    status = "PASS" if ok else "FAIL"
    print(f"[minimax_music3.smoke] census[{name}]: {status} "
          f"(model={len(model_keys)} tensors, safetensors={len(st_keys)} tensors)")
    if missing_in_st:
        print(f"  missing in safetensors (model has, checkpoint lacks): {missing_in_st[:10]}"
              + (" ..." if len(missing_in_st) > 10 else ""))
    if missing_in_model:
        print(f"  missing in model (checkpoint has, model lacks): {missing_in_model[:10]}"
              + (" ..." if len(missing_in_model) > 10 else ""))
    if shape_mismatch:
        print(f"  shape mismatches: {shape_mismatch[:10]}" + (" ..." if len(shape_mismatch) > 10 else ""))
    return ok


def _numeric_smoke(model_root: str) -> bool:
    """Tiny random-weight forward passes at the REAL config's shape parameters, on CPU."""
    import torch

    from core.models.minimax_music3.vendor import (
        MiniMaxMusic3ConditionEncoder,
        MiniMaxMusic3RVQDepthDecoder,
        MiniMaxMusic3Transformer1DModel,
        MiniMaxMusic3Vocoder,
    )

    ok = True

    ce_cfg = _load_config(os.path.join(model_root, "condition_encoder", "config.json"))
    ce = MiniMaxMusic3ConditionEncoder(**ce_cfg).eval()
    frames = 40
    x = torch.randn(1, frames, ce_cfg["num_condition_layers"] * ce_cfg["condition_hidden_dim"])
    with torch.no_grad():
        out = ce(x)
    finite = bool(torch.isfinite(out).all())
    print(f"[minimax_music3.smoke] condition_encoder: in {tuple(x.shape)} -> out {tuple(out.shape)}, "
          f"finite={finite}")
    ok = ok and finite

    voc_cfg = _load_config(os.path.join(model_root, "vocoder", "config.json"))
    voc = MiniMaxMusic3Vocoder(**voc_cfg).eval()
    length = 20
    lat = torch.randn(1, voc_cfg["latent_channels"], length)
    with torch.no_grad():
        wav = voc(lat)
    expected_samples = length
    for ratio in voc_cfg["upsampling_ratios"]:
        expected_samples *= ratio
    finite = bool(torch.isfinite(wav).all())
    shape_ok = tuple(wav.shape) == (1, 2, expected_samples)
    range_ok = bool((wav.abs() <= 1.0001).all())
    print(f"[minimax_music3.smoke] vocoder: in {tuple(lat.shape)} -> out {tuple(wav.shape)} "
          f"(expected samples={expected_samples}), finite={finite}, in_range={range_ok}")
    ok = ok and finite and shape_ok and range_ok

    dd_cfg = _load_config(os.path.join(model_root, "rvq_depth_decoder", "config.json"))
    dd = MiniMaxMusic3RVQDepthDecoder(**dd_cfg).eval()
    steps = 3
    seq = torch.randn(2, steps, dd_cfg["hidden_size"])
    with torch.no_grad():
        dd_out = dd(seq)
    finite = bool(torch.isfinite(dd_out).all())
    print(f"[minimax_music3.smoke] rvq_depth_decoder: in {tuple(seq.shape)} -> out {tuple(dd_out.shape)}, "
          f"finite={finite}")
    ok = ok and finite

    tr_cfg = _load_config(os.path.join(model_root, "transformer", "config.json"))
    tr_cfg_small = dict(tr_cfg, num_layers=2)  # keep host RAM tiny; not the real 36-layer weight count
    tr = MiniMaxMusic3Transformer1DModel(**tr_cfg_small).eval()
    batch, length = 1, 12
    lat = torch.randn(batch, tr_cfg["in_channels"], length)
    cond = torch.randn(batch, length, tr_cfg["condition_dim"])
    timestep = torch.tensor([0.5])
    with torch.no_grad():
        tr_out = tr(hidden_states=lat, timestep=timestep, encoder_hidden_states=cond, return_dict=False)[0]
    finite = bool(torch.isfinite(tr_out).all())
    shape_ok = tuple(tr_out.shape) == tuple(lat.shape)
    print(f"[minimax_music3.smoke] transformer (num_layers overridden to 2): in {tuple(lat.shape)} -> "
          f"out {tuple(tr_out.shape)}, finite={finite}")
    ok = ok and finite and shape_ok

    return ok


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model-root", required=True,
        help="Path to the MiniMax Music 3 'official' checkpoint tree "
             "(the directory containing transformer/, condition_encoder/, rvq_depth_decoder/, vocoder/).",
    )
    parser.add_argument(
        "--skip-census", action="store_true",
        help="Skip the config-vs-safetensors-header census (still runs the numeric smoke).",
    )
    args = parser.parse_args(argv)

    _announce_host_ram()

    components = {
        "transformer": None,
        "condition_encoder": None,
        "rvq_depth_decoder": None,
        "vocoder": None,
    }
    for name in components:
        directory = os.path.join(args.model_root, name)
        if not os.path.isdir(directory):
            print(f"[minimax_music3.smoke] FAILED: expected component directory not found: {directory}",
                  file=sys.stderr)
            return 1

    from core.models.minimax_music3.vendor import (
        MiniMaxMusic3ConditionEncoder,
        MiniMaxMusic3RVQDepthDecoder,
        MiniMaxMusic3Transformer1DModel,
        MiniMaxMusic3Vocoder,
    )

    ok = True
    if not args.skip_census:
        transformer_dir = os.path.join(args.model_root, "transformer")
        transformer_shards = sorted(
            os.path.join(transformer_dir, f) for f in os.listdir(transformer_dir) if f.endswith(".safetensors")
        )
        ok = _census(
            MiniMaxMusic3Transformer1DModel, os.path.join(transformer_dir, "config.json"), transformer_shards,
            "transformer",
        ) and ok
        ok = _census(
            MiniMaxMusic3ConditionEncoder,
            os.path.join(args.model_root, "condition_encoder", "config.json"),
            [os.path.join(args.model_root, "condition_encoder", "diffusion_pytorch_model.safetensors")],
            "condition_encoder",
        ) and ok
        ok = _census(
            MiniMaxMusic3RVQDepthDecoder,
            os.path.join(args.model_root, "rvq_depth_decoder", "config.json"),
            [os.path.join(args.model_root, "rvq_depth_decoder", "diffusion_pytorch_model.safetensors")],
            "rvq_depth_decoder",
        ) and ok
        ok = _census(
            MiniMaxMusic3Vocoder,
            os.path.join(args.model_root, "vocoder", "config.json"),
            [os.path.join(args.model_root, "vocoder", "diffusion_pytorch_model.safetensors")],
            "vocoder",
        ) and ok
    else:
        print("[minimax_music3.smoke] --skip-census: skipping the config/safetensors header census.")

    ok = _numeric_smoke(args.model_root) and ok

    print(f"[minimax_music3.smoke] RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
