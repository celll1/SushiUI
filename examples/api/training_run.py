"""
Example: create -> start -> poll status -> get metrics for a LoRA training run.

Endpoints (backend/api/routes.py), all JSON (unlike /generate/*, which use
Form/multipart -- see txt2img_minimal.py):
  POST /api/v1/training/runs                 body: TrainingRunCreateRequest (JSON)
      required fields: training_method ("lora" | "full_finetune"), base_model_path,
                        and EITHER dataset_id OR dataset_configs,
                        and EITHER total_steps OR epochs (mutually exclusive, exactly one).
      response: TrainingRun.to_dict() (includes "id", "run_name", "status": "pending", ...)

  POST /api/v1/training/runs/{run_id}/start   body: none
      response: (kicks off the training subprocess; see routes.py ~line 7679)

  GET  /api/v1/training/runs/{run_id}/status  no body
      response: {"status", "progress", "current_step", "total_steps", "loss",
                 "learning_rate", "phase", "phase_progress", "phase_detail",
                 "process_status"}

  GET  /api/v1/training/runs/{run_id}/metrics query: since_step, max_points (default 1000)
      response: {"loss": [...], "recon_loss": [...], "learning_rate": [...]}
      (read from TensorBoard event files under {output_dir}/tensorboard/)

DRY-RUN IS THE DEFAULT AND THIS SCRIPT WILL NOT DISABLE IT ON ITS OWN.
Starting a real training run spins up a GPU-resident subprocess, writes to
disk under the configured training base dir, and mutates the training.db --
it is heavy and stateful, unlike the read-only gallery/health examples. Only
pass --no-dry-run if you have a real dataset_id and base_model_path ready and
intend to actually train.
"""
import argparse
import json
import requests

BASE_URL = "http://localhost:8000/api/v1"


def build_create_payload(dataset_id: int, base_model_path: str, total_steps: int) -> dict:
    """Minimal required fields per TrainingRunCreateRequest (routes.py line ~6768).
    Everything else falls back to TRAINING_DEFAULTS server-side."""
    return {
        "training_method": "lora",
        "base_model_path": base_model_path,
        "dataset_id": dataset_id,
        "total_steps": total_steps,
        # epochs is intentionally omitted -- total_steps/epochs are mutually exclusive
    }


def print_plan(method: str, url: str, body=None, params=None):
    print(f"--- PLAN: {method} {url} ---")
    if params:
        print(f"  query params: {json.dumps(params)}")
    if body is not None:
        print(f"  json body: {json.dumps(body, indent=2)}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Create/start/poll a LoRA training run")
    parser.add_argument("--dataset-id", type=int, default=1, help="Dataset ID to train on")
    parser.add_argument("--base-model-path", default="C:/path/to/base_model.safetensors",
                         help="Absolute path to the base checkpoint")
    parser.add_argument("--total-steps", type=int, default=1000)
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false",
                         help="Actually create/start/poll a real training run (heavy, stateful)")
    parser.set_defaults(dry_run=True)
    args = parser.parse_args()

    create_body = build_create_payload(args.dataset_id, args.base_model_path, args.total_steps)
    create_url = f"{BASE_URL}/training/runs"

    if args.dry_run:
        print("=== DRY RUN: 4 request plans, nothing sent ===\n")
        print_plan("POST", create_url, body=create_body)
        print_plan("POST", f"{BASE_URL}/training/runs/{{run_id}}/start")
        print_plan("GET", f"{BASE_URL}/training/runs/{{run_id}}/status")
        print_plan("GET", f"{BASE_URL}/training/runs/{{run_id}}/metrics",
                    params={"max_points": 1000})
        print("Re-run with --no-dry-run (plus valid --dataset-id/--base-model-path) "
              "to actually create and start a training run.")
        return

    print(f"POST {create_url} ...")
    resp = requests.post(create_url, json=create_body, timeout=30)
    resp.raise_for_status()
    run = resp.json()
    run_id = run["id"]
    print(f"Created run id={run_id} status={run.get('status')}")

    start_url = f"{BASE_URL}/training/runs/{run_id}/start"
    print(f"POST {start_url} ...")
    resp = requests.post(start_url, timeout=30)
    resp.raise_for_status()
    print(json.dumps(resp.json(), indent=2)[:1000])

    status_url = f"{BASE_URL}/training/runs/{run_id}/status"
    print(f"GET {status_url} ...")
    resp = requests.get(status_url, timeout=10)
    resp.raise_for_status()
    print(json.dumps(resp.json(), indent=2))

    metrics_url = f"{BASE_URL}/training/runs/{run_id}/metrics"
    print(f"GET {metrics_url} ...")
    resp = requests.get(metrics_url, params={"max_points": 1000}, timeout=10)
    resp.raise_for_status()
    print(json.dumps(resp.json(), indent=2)[:1000])


if __name__ == "__main__":
    main()
