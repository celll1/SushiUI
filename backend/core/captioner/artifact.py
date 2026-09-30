"""Immutable caption bundles with verified inference and exact-resume files."""

from dataclasses import asdict
import hashlib
from pathlib import Path
import shutil
import tempfile
import weakref

import torch
from safetensors.torch import load_file, save_file

from .concepts import CONCEPT_CATEGORIES, ConceptSelector
from .model import CaptionConfig, CompactCaptioner
from .storage import canonical_hash, file_hash, read_json, write_json
from .tokenizer import CaptionTokenizer


def tagger_identity(manager, *, max_num_patches):
    if manager.model is None or manager.model_type == "onnx":
        raise ValueError("Captioner requires a loaded PyTorch SigLIP2 tagger")
    cached = getattr(manager, "_caption_weight_identity", None)
    if cached is None or cached[0]() is not manager.model:
        digest = hashlib.sha256()
        for name, tensor in sorted(manager.model.state_dict().items()):
            digest.update(name.encode())
            digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
            digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        cached = (weakref.ref(manager.model), digest.hexdigest())
        manager._caption_weight_identity = cached
    idx = manager.vocabulary["idx_to_tag"]
    if sorted(idx) != list(range(len(idx))):
        raise ValueError("Tagger vocabulary indices must be contiguous")
    names = [idx[i] for i in range(len(idx))]
    categories = [manager.vocabulary["tag_to_category"].get(name, "Unknown") for name in names]
    processor = manager.processor.image_processor.to_dict()
    return {"backend": "siglip2", "feature_schema": 1, "weights_hash": cached[1],
            "processor_hash": canonical_hash(processor), "processor_config": processor,
            "is_naflex": manager.is_naflex, "max_num_patches": max_num_patches,
            "names": names, "categories": categories}


def save_bundle(path, *, model, tokenizer, selector, identity, training_state, snapshot_hash, aliases, language):
    path = Path(path)
    if path.exists():
        raise ValueError("Caption bundles are immutable; choose a new checkpoint name")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".caption-bundle-", dir=path.parent))
    try:
        save_file({key: value.detach().cpu().contiguous().clone()
                   for key, value in model.state_dict().items()}, str(temp / "model.safetensors"))
        (temp / "tokenizer.json").write_text(tokenizer.to_json(), encoding="utf-8")
        write_json(temp / "config.json", asdict(model.config))
        manifest = {"format": "sushi-compact-captioner", "version": 1,
                    "identity": identity, "vocabulary_hash": selector.vocabulary_hash,
                    "selector_version": selector.VERSION, "max_concepts": selector.max_concepts,
                    "allowed_categories": sorted({selector.categories[i] for i in selector.allowed}),
                    "aliases": aliases, "language": language, "snapshot_hash": snapshot_hash}
        if training_state is not None:
            torch.save(training_state, temp / "resume.pt")
        manifest["files"] = {p.name: file_hash(p) for p in temp.iterdir() if p.is_file()}
        write_json(temp / "manifest.json", manifest)
        temp.rename(path)
    finally:
        if temp.exists():
            shutil.rmtree(temp)


def load_bundle(path, *, expected_identity=None, require_resume=False):
    path = Path(path)
    manifest = read_json(path / "manifest.json")
    if manifest.get("format") != "sushi-compact-captioner" or manifest.get("version") != 1:
        raise ValueError("Unsupported captioner artifact")
    if manifest.get("selector_version") != ConceptSelector.VERSION:
        raise ValueError("Unsupported concept selector version")
    required = {"model.safetensors", "tokenizer.json", "config.json"}
    if require_resume:
        required.add("resume.pt")
    files = manifest.get("files", {})
    if not required <= set(files) or set(files) - (required | {"resume.pt"}):
        raise ValueError("Incomplete captioner artifact")
    for name, digest in files.items():
        if file_hash(path / name) != digest:
            raise ValueError(f"Captioner artifact checksum mismatch: {name}")
    identity = manifest["identity"]
    if identity.get("backend") != "siglip2" or identity.get("feature_schema") != 1:
        raise ValueError("Unsupported caption feature backend")
    if expected_identity is not None and canonical_hash(identity) != canonical_hash(expected_identity):
        raise ValueError("Tagger weights, vocabulary, or processor do not match captioner")
    from core.tagger.tag_alias_resolver import TagAliasResolver
    resolver = TagAliasResolver(manifest["aliases"])
    selector = ConceptSelector(identity["names"], identity["categories"], max_concepts=manifest["max_concepts"],
                               allowed_categories=manifest["allowed_categories"], alias_resolver=resolver.resolve)
    if selector.vocabulary_hash != manifest["vocabulary_hash"]:
        raise ValueError("Caption vocabulary identity mismatch")
    config = CaptionConfig(**read_json(path / "config.json"))
    tokenizer = CaptionTokenizer.from_json((path / "tokenizer.json").read_text(encoding="utf-8"))
    if (config.vocab_size != tokenizer.vocab_size or config.num_tags != len(identity["names"])
            or config.num_categories != len(CONCEPT_CATEGORIES)
            or (config.pad_id, config.bos_id, config.eos_id) != (tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id)):
        raise ValueError("Caption configuration and tokenizer/vocabulary disagree")
    model = CompactCaptioner(config)
    model.load_state_dict(load_file(str(path / "model.safetensors")), strict=True)
    state = torch.load(path / "resume.pt", map_location="cpu", weights_only=True) if require_resume else None
    return model.eval(), tokenizer, selector, manifest, state


def export_bundle(source, destination):
    model, tokenizer, selector, manifest, _ = load_bundle(source)
    save_bundle(destination, model=model, tokenizer=tokenizer, selector=selector,
                identity=manifest["identity"], training_state=None,
                snapshot_hash=manifest["snapshot_hash"], aliases=manifest["aliases"],
                language=manifest["language"])
