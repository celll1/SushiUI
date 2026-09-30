"""Read topology declarations without allocating checkpoint tensors."""
import json
from pathlib import Path

from safetensors import safe_open

from .spec import read_checkpoint_metadata


def read_artifact_topology(path):
    path = Path(path)
    if str(path).endswith(".safetensors") and not path.exists():
        sibling = Path(str(path) + ".index.json")
        if sibling.is_file():
            path = sibling
    if not str(path).endswith(".safetensors.index.json"):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            return read_checkpoint_metadata(handle.metadata() or {})
    with path.open(encoding="utf-8") as handle:
        index = json.load(handle)
    if not isinstance(index, dict):
        raise ValueError("Checkpoint index must be an object")
    spec = read_checkpoint_metadata(index.get("metadata") or {})
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Checkpoint index requires a nonempty weight_map")
    filenames = set()
    for filename in weight_map.values():
        if not isinstance(filename, str) or not filename.endswith(".safetensors"):
            raise ValueError("Invalid checkpoint shard filename")
        shard = (path.parent / filename).resolve()
        if shard.parent != path.parent.resolve():
            raise ValueError("Checkpoint shard must be in its index directory")
        filenames.add(shard)
    for shard in filenames:
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            shard_spec = read_checkpoint_metadata(handle.metadata() or {})
        if shard_spec != spec:
            raise ValueError("Checkpoint index and shard topology declarations disagree")
    return spec


def read_directory_topology(path):
    declarations = [read_artifact_topology(candidate) for candidate in sorted(Path(path).iterdir())
                    if candidate.is_file() and str(candidate).endswith(
                        (".safetensors", ".safetensors.index.json"))]
    topology = declarations[0] if declarations else None
    if any(value != topology for value in declarations):
        raise ValueError("Directory transformer topology declarations disagree")
    return topology
