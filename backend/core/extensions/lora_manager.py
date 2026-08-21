"""
LoRA (Low-Rank Adaptation) Manager
Handles loading and applying multiple LoRAs with fine-grained control
"""
from typing import Dict, List, Optional, Any, Tuple
import hashlib
import os
import re
from pathlib import Path
from config.settings import settings


class LoRAAmbiguousIdentifierError(Exception):
    """Raised by ``_resolve_lora_path`` when a bare (untagged) LoRA identifier
    matches files in more than one registered directory. Previously this
    silently returned the first match, which is a real mis-application hazard
    once multiple per-model ``loras/`` directories are auto-registered
    alongside the user's configured directories -- two different files could
    share the same relative path across roots."""

    def __init__(self, identifier: str, matches: List[Path]):
        self.identifier = identifier
        self.matches = matches
        super().__init__(
            f"LoRA identifier '{identifier}' is ambiguous: it resolves to "
            f"{len(matches)} different files across registered directories: "
            f"{[str(m) for m in matches]}. Refusing to guess -- use the "
            f"disambiguated 'tag::relative_path' identifier returned by "
            f"GET /loras instead."
        )


# Subdirectory names that hold a single model component rather than the model
# root itself. Used by the sibling-`loras/` probe below: when the loaded
# model's path lands inside one of these, the probe also tries the PARENT
# directory (the actual model root) so `<model_root>/loras` is found even
# when the caller only ever sees `<model_root>/diffusion_models/foo.safetensors`.
_MODEL_COMPONENT_DIR_NAMES = {
    "diffusion_models", "diffusion_model", "text_encoders", "text_encoder",
    "text_encoder_2", "transformer", "unet", "vae", "tokenizer", "tokenizer_2",
    "scheduler", "official",
}


def classify_lora_keys(keys) -> Dict[str, Any]:
    """Single source of truth for LoRA architecture + block-structure detection
    from a safetensors key list. Reused by both ``LoRAManager._is_valid_lora_file``
    (arch tag at scan time) and ``LoRAManager.get_lora_layers`` (block list for
    the UI) -- do not add a second signature table elsewhere; extend HERE.

    Returns ``{"arch": str, "blocks": List[str]}``. ``arch`` is one of
    "sd15", "sdxl", "zimage", "flux2", "minimax_h3", "sensenova", "unknown"
    ("unknown" is a first-class value, not an error).
    """
    keys = list(keys)
    blocks = set()
    arch = "unknown"

    # --- SenseNova-U1.5-8B-MoT (distillation LoRA over the Qwen3-as-denoiser
    # gen branch) --------------------------------------------------------
    # Keys: language_model.model.layers.{N}.self_attn.{q,k,v,o}_proj_mot_gen.
    # <lora_down.weight|lora_up.weight|alpha> and
    # language_model.model.layers.{N}.mlp_mot_gen.{gate,up,down}_proj.<...>.
    # No other architecture's LoRA keys ever start with
    # "language_model.model.layers." (every other arch here is a diffusion
    # U-Net/DiT, spelling "lora_unet_"/"diffusion_model."/"transformer.*");
    # the "_mot_gen" (Mixture-of-Transformers generation branch) suffix is a
    # second, independent marker checked alongside it so a same-prefixed key
    # from some future non-MoT LLM-backed arch is not misclassified here.
    is_sensenova = any(
        key.startswith('language_model.model.layers.') and 'mot_gen' in key
        for key in keys
    )
    if is_sensenova:
        arch = "sensenova"
        for key in keys:
            match = re.search(r'language_model\.model\.layers\.(\d+)\.', key)
            if match:
                blocks.add(f"L{int(match.group(1)):02d}")
        if not blocks:
            blocks.add("BASE")
        return {"arch": arch, "blocks": _sort_lora_blocks(blocks)}

    # --- MiniMax-H3 (ComfyUI-exported LoRA) ---------------------------------
    # Keys: diffusion_model.blocks.{N}.<attn.qkv_proj|attn.out_proj|mlp.fc1|
    # mlp.fc2|adaln_proj.linear>.<lora_A|lora_B|alpha>, plus
    # diffusion_model.final_layer.* and diffusion_model.token_refiner.blocks.*.
    # Checked before every other signature: it never contains "transformer_blocks_"
    # (the FLUX.2 signature below) or any of the SD/SDXL/Z-Image substrings, but
    # is checked first regardless so a future overlapping format can't silently
    # steal these keys.
    is_minimax_h3 = any(
        key.startswith('diffusion_model.blocks.')
        or key.startswith('diffusion_model.final_layer.')
        or key.startswith('diffusion_model.token_refiner.')
        for key in keys
    )
    if is_minimax_h3:
        arch = "minimax_h3"
        for key in keys:
            match = re.search(r'diffusion_model\.blocks\.(\d+)\.', key)
            if match:
                blocks.add(f"MMB{int(match.group(1)):02d}")
            match = re.search(r'diffusion_model\.token_refiner\.blocks\.(\d+)\.', key)
            if match:
                blocks.add(f"TREF{int(match.group(1)):02d}")
            if key.startswith('diffusion_model.final_layer.'):
                blocks.add("FINAL")
        if not blocks:
            blocks.add("BASE")
        return {"arch": arch, "blocks": _sort_lora_blocks(blocks)}

    # --- SD1.5 / SDXL (kohya-ss "lora_unet_*"/"lora_te*_*" or diffusers dot format) ---
    has_te2 = any(key.startswith('lora_te2_') or key.startswith('text_encoder_2.') for key in keys)
    for key in keys:
        if 'input_blocks' in key:
            match = re.search(r'input_blocks[_.](\d+)', key)
            if match:
                blocks.add(f"IN{int(match.group(1)):02d}")
        elif 'middle_block' in key:
            blocks.add("MID")
        elif 'output_blocks' in key:
            match = re.search(r'output_blocks[_.](\d+)', key)
            if match:
                blocks.add(f"OUT{int(match.group(1)):02d}")
        elif 'down_blocks' in key:
            match = re.search(r'down_blocks[_.](\d+)[._]attentions[_.](\d+)', key)
            if match:
                i, j = int(match.group(1)), int(match.group(2))
                blocks.add(f"IN{3 * i + j + 1:02d}")
        elif 'mid_block' in key:
            blocks.add("MID")
        elif 'up_blocks' in key:
            match = re.search(r'up_blocks[_.](\d+)[._]attentions[_.](\d+)', key)
            if match:
                i, j = int(match.group(1)), int(match.group(2))
                blocks.add(f"OUT{3 * i + j:02d}")

    if blocks or any(k.startswith('lora_unet_') or k.startswith('lora_te') for k in keys):
        arch = "sdxl" if has_te2 else "sd15"
        if not blocks:
            blocks.add("BASE")
        return {"arch": arch, "blocks": _sort_lora_blocks(blocks)}

    # --- Z-Image (transformer-based) ---------------------------------------
    for key in keys:
        if 'noise_refiner' in key:
            match = re.search(r'noise_refiner[_.](\d+)', key)
            if match:
                blocks.add(f"NRef{int(match.group(1))}")
        elif 'context_refiner' in key:
            match = re.search(r'context_refiner[_.](\d+)', key)
            if match:
                blocks.add(f"CRef{int(match.group(1))}")
        elif 'transformer.layers.' in key:
            match = re.search(r'layers[_.](\d+)', key)
            if match:
                blocks.add(f"FDiT{int(match.group(1)):02d}")

    if blocks or any('noise_refiner' in k or 'context_refiner' in k or 'transformer.layers.' in k for k in keys):
        arch = "zimage"
        if not blocks:
            blocks.add("BASE")
        return {"arch": arch, "blocks": _sort_lora_blocks(blocks)}

    # --- FLUX.2 (dual + single stream transformer blocks) ------------------
    for key in keys:
        if 'transformer_blocks_' in key or 'single_transformer_blocks_' in key:
            match_dual = re.search(r'transformer_blocks_(\d+)', key)
            if match_dual and 'single_transformer_blocks' not in key:
                blocks.add(f"DUAL{int(match_dual.group(1)):02d}")
            match_single = re.search(r'single_transformer_blocks_(\d+)', key)
            if match_single:
                blocks.add(f"SING{int(match_single.group(1)):02d}")

    if blocks:
        arch = "flux2"
        return {"arch": arch, "blocks": _sort_lora_blocks(blocks)}

    # --- Unknown / unrecognized structure ------------------------------------
    blocks.add("BASE")
    return {"arch": "unknown", "blocks": _sort_lora_blocks(blocks)}


def _sort_lora_blocks(blocks) -> List[str]:
    """Sort block labels: BASE, IN00-IN.., MID, OUT00-.., NRef/CRef/FDiT
    (Z-Image), DUAL/SING (FLUX.2), MMB/TREF/FINAL (MiniMax-H3), L00-.. (SenseNova)."""
    def sort_key(block):
        if block == "BASE":
            return (0, 0)
        elif block == "MID":
            return (2, 0)
        elif block.startswith("MID"):
            return (2, int(block[3:]) if len(block) > 3 else 0)
        elif block.startswith("IN"):
            return (1, int(block[2:]))
        elif block.startswith("OUT"):
            return (3, int(block[3:]))
        elif block.startswith("NRef"):
            return (1, int(block[4:]))
        elif block.startswith("CRef"):
            return (2, int(block[4:]))
        elif block.startswith("FDiT"):
            return (3, int(block[4:]))
        elif block.startswith("DUAL"):
            return (1, int(block[4:]))
        elif block.startswith("SING"):
            return (2, int(block[4:]))
        elif block.startswith("TREF"):
            return (1, int(block[4:]))
        elif block == "FINAL":
            return (3, 0)
        elif block.startswith("MMB"):
            return (2, int(block[3:]))
        elif block.startswith("L"):
            return (1, int(block[1:]))
        return (9, 0)

    return sorted(list(blocks), key=sort_key)


class LoRAConfig:
    """Configuration for a single LoRA"""
    def __init__(
        self,
        path: str,
        strength: float = 1.0,
        apply_to_text_encoder: bool = True,
        apply_to_unet: bool = True,
        unet_layer_weights: Optional[Dict[str, float]] = None,
        step_range: Optional[List[int]] = None
    ):
        self.path = path
        self.strength = strength
        self.apply_to_text_encoder = apply_to_text_encoder
        self.apply_to_unet = apply_to_unet
        self.unet_layer_weights = unet_layer_weights or {}
        self.step_range = step_range or [0, 1000]  # 0 = start, 1000 = end

    def is_active_at_step(self, current_step: int, total_steps: int) -> bool:
        """Check if LoRA should be active at current step"""
        # Convert normalized range [0-1000] to actual step range
        start_step = int((self.step_range[0] / 1000) * total_steps)
        end_step = int((self.step_range[1] / 1000) * total_steps)
        return start_step <= current_step <= end_step

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LoRAConfig":
        """Create LoRAConfig from dictionary"""
        return cls(
            path=data.get("path", ""),
            strength=data.get("strength", 1.0),
            apply_to_text_encoder=data.get("apply_to_text_encoder", True),
            apply_to_unet=data.get("apply_to_unet", True),
            unet_layer_weights=data.get("unet_layer_weights"),
            step_range=data.get("step_range", [0, 1000])
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "path": self.path,
            "strength": self.strength,
            "apply_to_text_encoder": self.apply_to_text_encoder,
            "apply_to_unet": self.apply_to_unet,
            "unet_layer_weights": self.unet_layer_weights,
            "step_range": self.step_range
        }


class LoRAManager:
    """Manages multiple LoRAs for Stable Diffusion pipelines"""

    def __init__(self, lora_dir: Optional[str] = None):
        if lora_dir is None:
            lora_dir = settings.lora_dir
        self.lora_dir = Path(lora_dir)
        # User-configured additional directories (settings textarea). Set via
        # set_additional_dirs(), which REPLACES this list only -- it must never
        # touch seeded_dirs below (that was the bug: set_additional_dirs used to
        # replace the single additional_dirs list wholesale, silently dropping
        # the training/ dir seeded at startup on every settings save).
        self.additional_dirs: List[Path] = []
        # System-discovered directories: the training/ dir seeded here, plus any
        # per-model `<model_root>/loras` sibling directories registered later via
        # register_model_sibling_loras(). Composed with additional_dirs, never
        # replaced by it.
        self.seeded_dirs: List[Path] = []
        self.loaded_loras: List[LoRAConfig] = []

        # Cache for validated LoRA files (to avoid re-validation on every API call)
        # Each entry: {"path": identifier, "name": str, "arch": str}
        self._lora_cache: Optional[List[Dict[str, Any]]] = None
        self._cache_timestamp: float = 0.0

        # Add training directory to search paths (for trained LoRAs)
        training_dir = Path(settings.root_dir) / "training"
        if training_dir.exists():
            self.seeded_dirs.append(training_dir)
            print(f"[LoRAManager] Added training directory to search paths: {training_dir}")

        print(f"[LoRAManager] LoRA directory: {self.lora_dir}")

    def set_additional_dirs(self, dirs: List[str]):
        """Set the USER-configured additional directories to scan for LoRAs.
        Composes with (never replaces) seeded_dirs -- see __init__ / register_
        seeded_lora_dir for why that distinction matters."""
        self.additional_dirs = [Path(d) for d in dirs if d.strip()]
        print(f"[LoRAManager] Additional directories set: {self.additional_dirs}")
        # Invalidate cache when directories change
        self._lora_cache = None

    def register_seeded_lora_dir(self, directory) -> bool:
        """Register a system-discovered (non-user-configured) LoRA directory,
        e.g. a per-model `<model_root>/loras` sibling. Composes with both the
        training/ dir seeded at __init__ and any user-configured
        additional_dirs (set_additional_dirs no longer wipes this list).

        Returns True if newly added (False if it doesn't exist, isn't a
        directory, or is already registered)."""
        d = Path(directory)
        if not d.is_dir():
            return False
        resolved = d.resolve()
        if any(existing.resolve() == resolved for existing in self.seeded_dirs):
            return False
        self.seeded_dirs.append(d)
        print(f"[LoRAManager] Registered seeded LoRA directory: {d}")
        self._lora_cache = None
        return True

    def register_model_sibling_loras(self, source_path: str) -> bool:
        """Auto-discover a per-model `loras/` directory next to the currently
        loaded model and register it as a seeded search directory.

        Follows the sibling-probe pattern used by the Krea 2 / Lens component
        loaders (`core/models/krea2/krea2_loader.py::_probe_sibling`,
        `core/models/lens/lens_loader.py`): given a model source that may be a
        single file (`<root>/diffusion_models/foo.safetensors`) or a component
        directory (`<root>/transformer/`), walk up through known component
        subdirectory names to the model root and probe `<root>/loras`.

        Never touches additional_dirs (the user's manual override) -- only
        composes via register_seeded_lora_dir().
        """
        if not source_path:
            return False
        try:
            p = Path(source_path)
        except Exception:
            return False
        if not p.exists():
            return False

        start = p.parent if p.is_file() else p

        candidates: List[Path] = []
        seen: set = set()

        def _add_candidate(d: Path):
            try:
                rd = d.resolve()
            except Exception:
                rd = d
            if rd not in seen:
                seen.add(rd)
                candidates.append(d)

        _add_candidate(start)
        cur = start
        for _ in range(4):
            if cur.name.lower() in _MODEL_COMPONENT_DIR_NAMES and cur.parent != cur:
                cur = cur.parent
                _add_candidate(cur)
            else:
                break

        registered = False
        for cand in candidates:
            loras_dir = cand / "loras"
            if loras_dir.is_dir():
                if self.register_seeded_lora_dir(loras_dir):
                    registered = True
        return registered

    def _effective_extra_dirs(self) -> List[Path]:
        """seeded_dirs followed by user-configured additional_dirs, deduped by
        resolved path, stable order. This is the priority order used both when
        scanning (get_available_loras) and when resolving a bare identifier
        (_resolve_lora_path) -- seeded (system-curated) directories are checked
        before the user's manual override list."""
        result: List[Path] = []
        seen: set = set()
        for d in list(self.seeded_dirs) + list(self.additional_dirs):
            try:
                rd = d.resolve()
            except Exception:
                rd = d
            if rd not in seen:
                seen.add(rd)
                result.append(d)
        return result

    def _dir_tag(self, directory: Path) -> str:
        """Short, stable, order-independent identifier for a search directory
        (used to disambiguate LoRA identifiers that collide across roots)."""
        try:
            resolved = str(Path(directory).resolve())
        except Exception:
            resolved = str(directory)
        return hashlib.sha1(resolved.encode('utf-8')).hexdigest()[:8]

    def _tag_to_dir(self, tag: str) -> Optional[Path]:
        for d in [self.lora_dir] + self._effective_extra_dirs():
            if self._dir_tag(d) == tag:
                return d
        return None

    def _resolve_lora_path(self, lora_path: str) -> Optional[Path]:
        """Resolve a LoRA identifier to an absolute Path.

        Two identifier shapes are accepted:
          - "tag::relative/path.safetensors" -- the disambiguated form served
            by get_available_loras() when a relative path collides across
            more than one registered directory. Resolves directly against the
            tagged directory; never ambiguous.
          - "relative/path.safetensors" (legacy / common case) -- checked
            against self.lora_dir first (unchanged priority, so every
            identifier saved before this change and every non-colliding
            identifier since still resolves exactly as before), then against
            every seeded/additional directory. If it matches in MORE THAN ONE
            of those directories, that is a genuine ambiguity and this raises
            LoRAAmbiguousIdentifierError instead of silently returning the
            first match.
        """
        tag, sep, rel = lora_path.partition("::")
        if sep:
            tagged_dir = self._tag_to_dir(tag)
            if tagged_dir is not None:
                full_path = tagged_dir / rel
                return full_path if full_path.exists() else None
            # Unrecognized tag: fall through and try the whole string as a
            # literal (bare) relative path -- keeps odd-but-real filenames
            # containing "::" resolvable instead of erroring outright.

        # Try default directory first (unchanged legacy priority).
        full_path = self.lora_dir / lora_path
        if full_path.exists():
            return full_path

        # Try every other registered directory; collect ALL matches so a
        # genuine cross-directory collision can be detected instead of
        # silently taking the first one.
        matches: List[Path] = []
        for extra_dir in self._effective_extra_dirs():
            candidate = extra_dir / lora_path
            if candidate.exists():
                matches.append(candidate)

        if not matches:
            return None
        if len(matches) > 1:
            raise LoRAAmbiguousIdentifierError(lora_path, matches)
        return matches[0]

    def _is_valid_lora_file(self, file_path: Path) -> Optional[str]:
        """
        Validate if a file is a valid LoRA model file and, if so, detect its
        architecture via classify_lora_keys() (the single signature table
        shared with get_lora_layers()).

        Checks:
        1. File extension (.safetensors only - .pt/.bin excluded to avoid debug latents)
        2. File contains LoRA-specific keys (lora_down, lora_up, etc.)
        3. Excludes training artifacts (optimizer states, debug latents, etc.)

        Returns:
            The detected arch string ("sd15" | "sdxl" | "zimage" | "flux2" |
            "minimax_h3" | "sensenova" | "unknown") if this is a valid LoRA
            file, else None.
        """
        # Exclude known training artifacts by filename patterns
        filename = file_path.name.lower()
        exclude_patterns = [
            'optimizer',           # optimizer states
            'debug_latent',        # debug latent images
            'scheduler',           # scheduler states
            'ema',                 # EMA states
        ]

        for pattern in exclude_patterns:
            if pattern in filename:
                print(f"[LoRAManager] Excluding training artifact: {file_path.name}")
                return None

        # Check file extension (only .safetensors - exclude .pt to avoid debug latents)
        if file_path.suffix not in ['.safetensors']:
            return None

        # Verify .safetensors files contain LoRA keys
        try:
            from safetensors import safe_open

            with safe_open(file_path, framework="pt", device="cpu") as f:
                keys = list(f.keys())

                # LoRA architecture detection:
                # LoRA files have lora_down AND lora_up weights (rank decomposition)
                # Full parameter fine-tune has only full weights (unet.*.weight without lora)

                has_lora_down = any('lora_down' in key for key in keys)
                has_lora_up = any('lora_up' in key for key in keys)

                # Alternative LoRA formats (diffusers, kohya-ss variants)
                has_lora_A = any('.lora_A.' in key for key in keys)
                has_lora_B = any('.lora_B.' in key for key in keys)
                has_lora_unet = any('lora_unet' in key for key in keys)
                has_lora_te = any('lora_te' in key for key in keys)

                # Z-Image LoRA format (transformer-based)
                # Keys: transformer.layers.0.attn1.to_q.lora_down.weight
                has_lora_transformer = any('transformer.' in key and ('lora_down' in key or 'lora_up' in key) for key in keys)

                # Valid LoRA must have BOTH lora_down AND lora_up (or lora_A AND lora_B)
                is_lora = (has_lora_down and has_lora_up) or \
                          (has_lora_A and has_lora_B) or \
                          (has_lora_unet or has_lora_te) or \
                          has_lora_transformer

                if not is_lora:
                    print(f"[LoRAManager] Excluding non-LoRA file (full parameter fine-tune): {file_path.name}")
                    if len(keys) > 0:
                        print(f"[LoRAManager]   Sample keys: {keys[:5]}")
                        print(f"[LoRAManager]   has_lora_down={has_lora_down}, has_lora_up={has_lora_up}")
                    return None

                arch = classify_lora_keys(keys).get("arch", "unknown")

        except Exception as e:
            print(f"[LoRAManager] Could not validate {file_path.name}: {e}")
            # If we can't read it, exclude it to be safe
            return None

        return arch

    def get_available_loras(self, force_rescan: bool = False) -> List[Dict[str, Any]]:
        """
        Get list of available LoRA files from default and additional/seeded
        directories.

        Uses cache to avoid expensive validation on every API call.

        Args:
            force_rescan: Force re-scanning and validation (ignores cache)

        Returns:
            List of {"path": identifier, "name": str, "arch": str} dicts.
            "path" is a bare relative path for the common (non-colliding)
            case -- unchanged from before, so existing stored identifiers
            keep working -- or a disambiguated "tag::relative/path" identifier
            when the same relative path exists under more than one registered
            directory (see _resolve_lora_path's docstring).
        """
        import time

        # Return cached result if available and not forcing rescan
        if not force_rescan and self._lora_cache is not None:
            return self._lora_cache

        print(f"[LoRAManager] Scanning and validating LoRA files...")
        scan_start = time.time()

        # Combine default directory with seeded + user-configured directories,
        # in priority order (default dir wins any collision, matching the
        # pre-existing _resolve_lora_path priority).
        all_dirs = [self.lora_dir] + self._effective_extra_dirs()

        # rel_path -> list of (dir, abs_path, arch), in scan (= priority) order
        records_by_rel: Dict[str, List[Tuple[Path, Path, str]]] = {}

        for lora_dir in all_dirs:
            print(f"[LoRAManager] Checking directory: {lora_dir}")
            print(f"[LoRAManager] Directory exists: {lora_dir.exists()}")

            if not lora_dir.exists():
                if lora_dir == self.lora_dir:
                    print(f"[LoRAManager] Creating default directory: {lora_dir}")
                    lora_dir.mkdir(parents=True, exist_ok=True)
                else:
                    print(f"[LoRAManager] Skipping non-existent directory: {lora_dir}")
                continue

            # Only scan .safetensors files (exclude .pt to avoid debug latents and training artifacts)
            for ext in [".safetensors"]:
                found = list(lora_dir.rglob(f"*{ext}"))
                print(f"[LoRAManager] Found {len(found)} files with extension {ext} in {lora_dir}")

                for f in found:
                    arch = self._is_valid_lora_file(f)
                    if arch is not None:
                        rel = str(f.relative_to(lora_dir))
                        records_by_rel.setdefault(rel, []).append((lora_dir, f, arch))

        result: List[Dict[str, Any]] = []
        for rel, recs in records_by_rel.items():
            for idx, (lora_dir, abs_path, arch) in enumerate(recs):
                if len(recs) > 1 and idx > 0:
                    # Real collision: this root's copy is NOT the highest-
                    # priority owner of the bare identifier -- disambiguate
                    # instead of letting it silently shadow/collide.
                    identifier = f"{self._dir_tag(lora_dir)}::{rel}"
                else:
                    identifier = rel
                result.append({
                    "path": identifier,
                    "name": os.path.basename(rel),
                    "arch": arch,
                })

        result.sort(key=lambda e: e["path"].lower())
        scan_duration = time.time() - scan_start

        print(f"[LoRAManager] Total valid LoRA files found: {len(result)}")
        print(f"[LoRAManager] Scan completed in {scan_duration:.2f}s")

        # Cache result
        self._lora_cache = result
        self._cache_timestamp = time.time()

        return result

    def invalidate_cache(self):
        """Invalidate cached LoRA list (call when files are added/removed)"""
        print(f"[LoRAManager] Cache invalidated")
        self._lora_cache = None

    def load_loras(self, pipeline: Any, lora_configs: List[Dict[str, Any]]) -> Any:
        """
        Load multiple LoRAs into the pipeline

        Args:
            pipeline: Diffusers pipeline
            lora_configs: List of LoRA configurations

        Returns:
            Modified pipeline with LoRAs loaded
        """
        print(f"[LoRAManager] load_loras called with {len(lora_configs) if lora_configs else 0} configs")
        print(f"[LoRAManager] lora_configs: {lora_configs}")

        if not lora_configs:
            print("[LoRAManager] No LoRA configs provided, skipping")
            return pipeline

        # Parse configs
        self.loaded_loras = [LoRAConfig.from_dict(cfg) for cfg in lora_configs]
        print(f"[LoRAManager] Parsed {len(self.loaded_loras)} LoRA configs")

        # Load LoRAs using diffusers' native support
        try:
            for i, lora_config in enumerate(self.loaded_loras):
                lora_path = self._resolve_lora_path(lora_config.path)

                if lora_path is None:
                    print(f"[LoRAManager] WARNING: LoRA file not found: {lora_config.path}")
                    continue

                print(f"[LoRAManager] Attempting to load LoRA from: {lora_path}")
                print(f"[LoRAManager] LoRA config: strength={lora_config.strength}, apply_to_text_encoder={lora_config.apply_to_text_encoder}, apply_to_unet={lora_config.apply_to_unet}")

                print(f"[LoRAManager] Loading LoRA {i+1}/{len(self.loaded_loras)}: {lora_config.path}")

                # Detect LoRA format and convert if needed
                from safetensors import safe_open
                import tempfile
                import os

                adapter_name = f"lora_{i}"

                # Check LoRA format
                with safe_open(str(lora_path), framework="pt", device="cpu") as f:
                    sample_keys = list(f.keys())[:5]
                    print(f"[LoRAManager] Sample keys from LoRA: {sample_keys}")

                    # Detect format: SD format uses underscores (lora_unet_*, lora_te1_*)
                    # Diffusers format uses dots (unet.*, text_encoder.*)
                    is_sd_format = any(k.startswith("lora_") for k in sample_keys)
                    is_diffusers_format = any("." in k and not k.startswith("lora_") for k in sample_keys)

                    print(f"[LoRAManager] LoRA format detected: SD={is_sd_format}, Diffusers={is_diffusers_format}")

                # If LoRA is in diffusers format (dots), convert to SD format for load_lora_weights
                if is_diffusers_format and not is_sd_format:
                    print(f"[LoRAManager] Converting diffusers format to SD format...")
                    converted_state_dict = {}

                    with safe_open(str(lora_path), framework="pt", device="cpu") as f:
                        for key in f.keys():
                            tensor = f.get_tensor(key)

                            # Convert key format:
                            # unet.down_blocks.0.xxx.lora_down.weight -> lora_unet_down_blocks_0_xxx.lora_down.weight
                            # text_encoder.xxx.lora_down.weight -> lora_te1_xxx.lora_down.weight
                            # text_encoder_2.xxx.lora_up.weight -> lora_te2_xxx.lora_up.weight
                            # IMPORTANT: Keep .lora_down.weight, .lora_up.weight, .alpha as-is (dots)

                            # Separate the suffix (.lora_down.weight, .lora_up.weight, .alpha)
                            if ".lora_down.weight" in key:
                                suffix = ".lora_down.weight"
                                base_key = key.replace(suffix, "")
                            elif ".lora_up.weight" in key:
                                suffix = ".lora_up.weight"
                                base_key = key.replace(suffix, "")
                            elif ".alpha" in key:
                                suffix = ".alpha"
                                base_key = key.replace(suffix, "")
                            else:
                                # Unknown key format, keep as-is
                                new_key = key
                                converted_state_dict[new_key] = tensor
                                continue

                            # Convert the base key (module path) to SD format
                            if base_key.startswith("unet."):
                                # unet.down_blocks.0.xxx -> lora_unet_down_blocks_0_xxx
                                new_base = "lora_" + base_key.replace(".", "_")
                            elif base_key.startswith("text_encoder_2."):
                                # text_encoder_2.text_model.xxx -> lora_te2_text_model_xxx
                                new_base = "lora_te2_" + base_key.replace("text_encoder_2.", "").replace(".", "_")
                            elif base_key.startswith("text_encoder."):
                                # text_encoder.text_model.xxx -> lora_te1_text_model_xxx
                                new_base = "lora_te1_" + base_key.replace("text_encoder.", "").replace(".", "_")
                            else:
                                # Unknown prefix, keep as-is
                                new_key = key
                                converted_state_dict[new_key] = tensor
                                continue

                            # Combine base + suffix
                            new_key = new_base + suffix
                            converted_state_dict[new_key] = tensor

                    # Save converted LoRA to temporary file
                    from safetensors.torch import save_file
                    temp_dir = tempfile.gettempdir()
                    temp_lora_path = os.path.join(temp_dir, f"converted_lora_{adapter_name}.safetensors")
                    save_file(converted_state_dict, temp_lora_path)

                    print(f"[LoRAManager] Converted LoRA saved to: {temp_lora_path}")
                    print(f"[LoRAManager] Calling pipeline.load_lora_weights with adapter_name={adapter_name}")

                    # Load converted LoRA
                    pipeline.load_lora_weights(
                        temp_dir,
                        weight_name=f"converted_lora_{adapter_name}.safetensors",
                        adapter_name=adapter_name
                    )

                    # Clean up temporary file
                    os.remove(temp_lora_path)
                    print(f"[LoRAManager] Temporary file removed")
                else:
                    # SD format (Kohya-ss format: lora_te1_*, lora_unet_*) - load directly
                    # diffusers' pipeline.load_lora_weights natively supports SD/Kohya-ss format
                    print(f"[LoRAManager] SD/Kohya-ss format detected - loading directly")
                    print(f"[LoRAManager] Calling pipeline.load_lora_weights with adapter_name={adapter_name}")
                    pipeline.load_lora_weights(
                        str(lora_path.parent),
                        weight_name=lora_path.name,
                        adapter_name=adapter_name
                    )

                print(f"[LoRAManager] Successfully loaded LoRA weights")

                # Set adapter with strength
                # Note: Step ranges will be handled in callback
                if hasattr(pipeline, 'set_adapters'):
                    print(f"[LoRAManager] Setting adapter with strength={lora_config.strength}")
                    pipeline.set_adapters(adapter_name, adapter_weights=lora_config.strength)

                    # Debug: Check if adapter is actually active
                    if hasattr(pipeline, 'get_active_adapters'):
                        active_adapters = pipeline.get_active_adapters()
                        print(f"[LoRAManager] Active adapters after set_adapters: {active_adapters}")

                    # Debug: Check UNet's LoRA modules
                    print(f"[LoRAManager] Checking UNet for LoRA modules...")
                    lora_module_count = 0
                    for name, module in pipeline.unet.named_modules():
                        if hasattr(module, 'lora_A') or hasattr(module, 'lora_B') or hasattr(module, 'scaling'):
                            lora_module_count += 1
                            if lora_module_count <= 3:  # Show first 3
                                print(f"[LoRAManager]   LoRA module found: {name}")
                                if hasattr(module, 'scaling'):
                                    print(f"[LoRAManager]     scaling: {module.scaling}")
                    print(f"[LoRAManager] Total LoRA modules in UNet: {lora_module_count}")

                    # Apply per-layer weights if specified
                    if lora_config.unet_layer_weights and hasattr(pipeline, 'unet'):
                        print(f"[LoRAManager] Applying per-layer weights: {len(lora_config.unet_layer_weights)} layers")
                        self._apply_layer_weights(pipeline, adapter_name, lora_config)
                else:
                    print(f"[LoRAManager] WARNING: Pipeline does not have set_adapters method")

            print(f"[LoRAManager] Successfully loaded {len(self.loaded_loras)} LoRA(s)")

        except Exception as e:
            print(f"[LoRAManager] ERROR loading LoRAs: {e}")
            import traceback
            traceback.print_exc()

        return pipeline

    def _apply_layer_weights(self, pipeline: Any, adapter_name: str, lora_config: LoRAConfig):
        """
        Apply per-block weights to the LoRA adapter

        This modifies the LoRA adapter weights in the UNet directly by scaling them according to
        the block-specific weights (IN00-IN11, MID, OUT00-OUT11) specified in the config.
        """
        try:
            import re

            # Access the UNet's LoRA layers directly
            if not hasattr(pipeline, 'unet'):
                print("[LoRAManager] Pipeline does not have unet attribute")
                return

            unet = pipeline.unet

            # Check if UNet has peft_config (PEFT-based LoRA)
            if not hasattr(unet, 'peft_config'):
                print("[LoRAManager] UNet does not have peft_config, trying alternative method")
                # Try alternative method for non-PEFT LoRAs
                self._apply_layer_weights_alternative(pipeline, adapter_name, lora_config)
                return

            # Iterate through all named modules in the UNet
            modified_count = 0
            for name, module in unet.named_modules():
                # Check if this module has LoRA adapters
                if hasattr(module, 'lora_A') or hasattr(module, 'lora_B'):
                    # Determine which block this module belongs to
                    block_weight = self._get_block_weight_for_module(name, lora_config.unet_layer_weights)

                    if block_weight != 1.0:  # Only modify if weight is not default
                        # Scale the LoRA weights
                        if hasattr(module, 'scaling') and adapter_name in module.scaling:
                            # Modify the scaling factor
                            original_scaling = module.scaling[adapter_name]
                            module.scaling[adapter_name] = original_scaling * block_weight
                            modified_count += 1

            if modified_count > 0:
                print(f"[LoRAManager] Applied block weights to {modified_count} LoRA layers")
            else:
                print(f"[LoRAManager] WARNING: No LoRA layers were modified with block weights")

        except Exception as e:
            print(f"[LoRAManager] WARNING: Failed to apply per-block weights: {e}")
            import traceback
            traceback.print_exc()

    def _get_block_weight_for_module(self, module_name: str, block_weights: dict) -> float:
        """
        Determine the block weight for a given module name

        Args:
            module_name: Full name of the module (e.g., "down_blocks.0.attentions.0")
            block_weights: Dictionary of block_id -> weight

        Returns:
            Weight value for this module (default 1.0)
        """
        # Parse module name to determine block
        if 'down_blocks' in module_name or 'input_blocks' in module_name:
            # Extract block number
            import re
            match = re.search(r'(down_blocks|input_blocks)[._](\d+)', module_name)
            if match:
                block_num = int(match.group(2))
                block_id = f"IN{block_num:02d}"
                return block_weights.get(block_id, 1.0)

        elif 'mid_block' in module_name or 'middle_block' in module_name:
            return block_weights.get("MID", 1.0)

        elif 'up_blocks' in module_name or 'output_blocks' in module_name:
            import re
            match = re.search(r'(up_blocks|output_blocks)[._](\d+)', module_name)
            if match:
                block_num = int(match.group(2))
                block_id = f"OUT{block_num:02d}"
                return block_weights.get(block_id, 1.0)

        # Check for BASE
        return block_weights.get("BASE", 1.0)

    def _apply_layer_weights_alternative(self, pipeline: Any, adapter_name: str, lora_config: LoRAConfig):
        """
        Alternative method for applying block weights (for older diffusers versions)
        """
        print("[LoRAManager] Using alternative block weight application method")
        # This is a fallback - in practice, the main method should work for most cases

    def create_step_callback(self, pipeline: Any, total_steps: int, original_callback=None):
        """
        Create a callback that handles step-based LoRA activation

        Args:
            pipeline: The diffusion pipeline
            total_steps: Total number of generation steps
            original_callback: Original progress callback to chain

        Returns:
            Callback function for step-based LoRA control
        """
        def callback(pipe, step: int, timestep: float, callback_kwargs: dict):
            # Check which LoRAs should be active at this step
            active_adapters = []
            adapter_weights = []

            for i, lora_config in enumerate(self.loaded_loras):
                if lora_config.is_active_at_step(step, total_steps):
                    adapter_name = f"lora_{i}"
                    active_adapters.append(adapter_name)
                    adapter_weights.append(lora_config.strength)

            # Update active adapters for this step
            if hasattr(pipeline, 'set_adapters'):
                if active_adapters:
                    pipeline.set_adapters(active_adapters, adapter_weights=adapter_weights)
                else:
                    # Disable all LoRAs if none are active
                    pipeline.disable_lora()

            # Call original callback if provided
            if original_callback:
                return original_callback(pipe, step, timestep, callback_kwargs)

            return callback_kwargs

        return callback

    def unload_loras(self, pipeline: Any) -> Any:
        """Unload all LoRAs from pipeline"""
        try:
            if hasattr(pipeline, 'unload_lora_weights'):
                pipeline.unload_lora_weights()
                print("Unloaded all LoRAs")
        except Exception as e:
            print(f"Error unloading LoRAs: {e}")

        self.loaded_loras = []
        return pipeline

    def get_lora_info(self, lora_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific LoRA file"""
        # Use _resolve_lora_path to check both lora/ and training/ directories
        # (may raise LoRAAmbiguousIdentifierError -- callers should let that
        # propagate rather than treating it as "not found").
        lora_path = self._resolve_lora_path(lora_name)

        if lora_path is None:
            return None

        # Get layer + arch information (single read of the file's keys)
        arch, layers = self._read_lora_keys_info(lora_path)
        recommended = self._parse_recommended_metadata(lora_path)

        return {
            "name": lora_name,
            "path": str(lora_path),
            "size": lora_path.stat().st_size,
            "exists": True,
            "arch": arch,
            "layers": layers,
            "recommended": recommended,
        }

    def get_lora_layers(self, lora_name: str) -> List[str]:
        """
        Extract U-Net/transformer block structure from a LoRA file.
        Returns blocks in format: BASE, IN00-IN11, MID, OUT00-OUT11 (SD/SDXL),
        NRef/CRef/FDiT (Z-Image), DUAL/SING (FLUX.2), MMB/TREF/FINAL (MiniMax-H3).
        """
        # Use _resolve_lora_path to check both lora/ and training/ directories
        lora_path = self._resolve_lora_path(lora_name)

        if lora_path is None:
            return []

        _, blocks = self._read_lora_keys_info(lora_path)
        return blocks

    def _read_lora_keys_info(self, lora_path: Path) -> Tuple[str, List[str]]:
        """Read a LoRA file's keys once and classify arch + blocks via the
        shared classify_lora_keys() signature table."""
        try:
            from safetensors import safe_open

            with safe_open(lora_path, framework="pt", device="cpu") as f:
                keys = list(f.keys())

            classification = classify_lora_keys(keys)
            arch = classification["arch"]
            blocks = classification["blocks"]
            print(f"[LoRAManager] {lora_path.name}: arch={arch}, {len(blocks)} blocks: {blocks}")
            return arch, blocks

        except Exception as e:
            print(f"[LoRAManager] Error reading LoRA keys: {e}")
            import traceback
            traceback.print_exc()
            return "unknown", []

    def _parse_recommended_metadata(self, lora_path: Path) -> Optional[Dict[str, Any]]:
        """Parse a step-distillation recommendation from the LoRA's safetensors
        `__metadata__`, when present. Only recognizes fields the file itself
        declares -- never invents a recommendation.

        Currently recognized: `student_steps` (a step-distillation LoRA's
        student evaluation count). The repo's `num_inference_steps` counts
        sigma grid points INCLUDING the terminal 0 (one more than the number
        of model evaluations -- see
        core/models/minimax_h3/h3_pipeline_ops.py:1202-1203), so
        num_inference_steps = student_steps + 1. Cache-across-steps features
        (FBCache/Spectrum forecasting) are recommended off: they assume dozens
        of evaluations to amortize their own bookkeeping, meaningless at ~4-8.
        """
        try:
            from safetensors import safe_open

            with safe_open(lora_path, framework="pt", device="cpu") as f:
                metadata = f.metadata() or {}
        except Exception as e:
            print(f"[LoRAManager] Could not read metadata from {lora_path.name}: {e}")
            return None

        student_steps_raw = metadata.get("student_steps")
        if student_steps_raw is None:
            return None

        try:
            student_steps = int(float(student_steps_raw))
        except (TypeError, ValueError):
            print(f"[LoRAManager] {lora_path.name}: unparseable student_steps={student_steps_raw!r}")
            return None

        return {
            "num_inference_steps": student_steps + 1,
            "fbcache_enable": False,
            "spectrum_enable": False,
            "source": "student_steps",
        }


# Global instance
lora_manager = LoRAManager()
