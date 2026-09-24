"""Generate matched baseline and legacy reference-style SDXL images through the live API."""

import argparse
import base64
import json
from pathlib import Path

import httpx


WORKTREE = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[4]
OUTPUT = WORKTREE / "outputs/sdxl_style_ip_dataset39"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000/api/v1")
    parser.add_argument("--pairs", type=int, default=24)
    parser.add_argument("--stop-after", type=int, default=None)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--style-strength", type=float, default=0.5)
    parser.add_argument("--cohort", choices=("primary", "confirm"), default="primary")
    args = parser.parse_args()
    if args.pairs < 1 or args.pairs > 24 or args.steps < 1 or not 0 <= args.style_strength <= 1:
        parser.error("pairs must be 1..24, steps positive, and style strength 0..1")
    if args.stop_after is not None and not 1 <= args.stop_after <= args.pairs:
        parser.error("stop-after must be between 1 and pairs")
    suffix = "_confirm" if args.cohort == "confirm" else ""
    planned = json.loads((OUTPUT / f"evaluation_strength_0p05{suffix}.json").read_text(encoding="utf-8"))["pairs"]
    manifest = {row["item_key"]: row for row in map(
        json.loads, (OUTPUT / "manifest.jsonl").read_text(encoding="utf-8").splitlines())}
    target_model = str(REPO / "models/NoobAI-XL-Vpred-v1.0.safetensors")
    record_path = OUTPUT / f"reference_route_{args.pairs}pairs{suffix}.json"
    records = []
    if record_path.exists():
        records = json.loads(record_path.read_text(encoding="utf-8"))["pairs"]
    completed = {row["pair"] for row in records}

    with httpx.Client(base_url=args.base_url, timeout=600) as client:
        active = client.get("/training/active").raise_for_status().json()
        if active["is_running"]:
            raise RuntimeError("Stop backend-managed training before SDXL reference comparison")
        original = client.get("/models/current").raise_for_status().json()["model_info"]
        if not original or not original.get("source"):
            raise RuntimeError("A loaded model is required so this script can restore it")
        try:
            loaded = client.post("/models/load", data={"source_type": "safetensors", "source": target_model})
            loaded.raise_for_status()
            for pair in planned[:args.pairs]:
                n = pair["pair"]
                if args.stop_after is not None and n >= args.stop_after:
                    break
                if n in completed:
                    continue
                target = manifest[pair["target_item_key"]]
                reference = manifest[pair["reference_item_key"]]
                common = {"prompt": target["caption"], "negative_prompt": "", "width": "512",
                          "height": "512", "steps": str(args.steps), "cfg_scale": "5", "seed": str(94117 + n),
                          "sampler": "euler", "schedule_type": "uniform", "skip_gallery": "true"}
                baseline = client.post("/generate/txt2img", data={**common, "controlnets": "[]"})
                baseline.raise_for_status()
                style_config = [{"model_path": "", "image_base64": base64.b64encode(
                    Path(reference["path"]).read_bytes()).decode("ascii"),
                    "is_style_transfer": True, "style_transfer_type": "style",
                    "strength": args.style_strength}]
                styled = client.post("/generate/txt2img", data={
                    **common, "controlnets": json.dumps(style_config, separators=(",", ":"))})
                styled.raise_for_status()
                row = {"pair": n, "reference_item_key": pair["reference_item_key"],
                       "target_item_key": pair["target_item_key"],
                       "baseline_path": str(REPO / "outputs" / baseline.json()["filename"]),
                       "reference_route_path": str(REPO / "outputs" / styled.json()["filename"]),
                       "baseline_warnings": baseline.json().get("warnings", []),
                       "reference_route_warnings": styled.json().get("warnings", [])}
                records.append(row)
                record_path.write_text(json.dumps({"model": target_model, "steps": args.steps,
                                                   "style_strength": args.style_strength,
                                                   "pairs": records}, indent=2) + "\n", encoding="utf-8")
                print(json.dumps({"pair": n, "baseline": baseline.json()["filename"],
                                  "styled": styled.json()["filename"]}), flush=True)
        finally:
            restored = client.post("/models/load", data={"source_type": original["source_type"],
                                                          "source": original["source"]})
            restored.raise_for_status()
            print(f"Restored {original['type']} model", flush=True)


if __name__ == "__main__":
    main()
