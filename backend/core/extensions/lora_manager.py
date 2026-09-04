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


def dir_tag(directory) -> str:
    """Short, stable, order-independent identifier for a LoRA search directory
    (used to disambiguate LoRA identifiers that collide across roots)."""
    try:
        resolved = str(Path(directory).resolve())
    except Exception:
        resolved = str(directory)
    return hashlib.sha1(resolved.encode('utf-8')).hexdigest()[:8]


def _colliding_dir_tags(identifier: str, matches: List[Path]) -> List[str]:
    """Tags of the registered directories the matches came from.

    Each match is ``<registered dir> / identifier``, so stripping as many
    trailing components as the identifier has recovers the root without
    naming it.
    """
    depth = len([p for p in str(identifier).replace("\\", "/").split("/") if p])
    tags: List[str] = []
    for match in matches:
        try:
            root = Path(match)
            for _ in range(depth):
                root = root.parent
            tags.append(dir_tag(root))
        except Exception:
            continue
    return tags


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
        # Raised outside every loader's try, so this text reaches the API
        # response verbatim: name the file and the directory tags (which is
        # what the caller types back as 'tag::relative_path'), never a path.
        name = str(identifier).replace("\\", "/").rsplit("/", 1)[-1]
        tags = ", ".join(_colliding_dir_tags(identifier, matches)) or "unavailable"
        super().__init__(
            f"LoRA identifier '{name}' is ambiguous: it resolves to "
            f"{len(matches)} different files across registered directories "
            f"(directory tags: {tags}). Refusing to guess -- use the "
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

    Returns ``{"arch": str, "blocks": List[str]}``. ``arch`` is one of the 13
    ``core.training.arch.ARCH_REGISTRY`` keys -- "sd15", "sdxl", "zimage",
    "anima", "lens", "ideogram4", "minit2i", "krea2", "flux2", "ltx2",
    "minimax_h3", "acestep", "sensenova" -- or "unknown" ("unknown" is a
    first-class value, not an error).

    ORDERING RULE: each architecture's signature is ANCHORED on the key prefix
    its own training adapter writes, and every one of them is tested BEFORE the
    sd-scripts ``lora_unet_`` / ``lora_te*`` catch-all at the bottom. Eight
    architectures besides SD1.5/SDXL write ``lora_unet_*`` stems, and an
    SD1.5/SDXL U-Net stem contains their spellings as substrings
    (``lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q``),
    so a DiT check that is unanchored, or ordered after the catch-all, either
    steals SD files or is shadowed by them. The anchored roots are mutually
    exclusive; ``backend/tests/lora_key_classification_test.py`` drives the real
    adapters and asserts the full cross-product.
    """
    keys = list(keys)
    blocks = set()

    def classified(arch: str) -> Dict[str, Any]:
        if not blocks:
            blocks.add("BASE")
        return {"arch": arch, "blocks": _sort_lora_blocks(blocks)}

    # --- SenseNova-U1.5-8B-MoT (LoRA over either MoT half) -----------------
    # Keys are plain module paths (sensenova_adapter.py:49-53):
    # language_model.model.layers.{N}.self_attn.{q,k,v,o}_proj[_mot_gen] and
    # .{mlp_mot_gen|mlp}.{gate,up,down}_proj. "_mot_gen" marks the generation
    # branch; the understanding half carries the un-suffixed names, so its leaf
    # spelling is a second accepted marker -- today only ever seen ALONGSIDE the
    # gen half (save_checkpoint refuses an understanding-only file,
    # sensenova_adapter.py:192), so it is a guard, not a producible artefact.
    # No other architecture here writes keys under "language_model.model.layers.",
    # and requiring one of the two markers keeps a future LLM-backed arch out.
    is_sensenova = any(
        key.startswith('language_model.model.layers.')
        and ('mot_gen' in key
             or re.search(r'\.(?:self_attn|mlp)\.(?:[qkvo]|gate|up|down)_proj\.', key))
        for key in keys
    )
    if is_sensenova:
        for key in keys:
            match = re.search(r'language_model\.model\.layers\.(\d+)\.', key)
            if match:
                blocks.add(f"L{int(match.group(1)):02d}")
        return classified("sensenova")

    # --- MiniMax-H3, ComfyUI/interchange layout ----------------------------
    # Keys: diffusion_model.blocks.{N}.<attn.qkv_proj|attn.out_proj|mlp.fc1|
    # mlp.fc2|adaln_proj.linear>.<lora_A|lora_B|alpha>, plus
    # diffusion_model.final_layer.* and diffusion_model.token_refiner.blocks.*.
    # (The sd-scripts layout THIS repo's trainer writes is handled with Lens and
    # LTX-2.3 further down -- minimax_h3_lora.py documents both conventions.)
    is_minimax_h3 = any(
        key.startswith('diffusion_model.blocks.')
        or key.startswith('diffusion_model.final_layer.')
        or key.startswith('diffusion_model.token_refiner.')
        for key in keys
    )
    if is_minimax_h3:
        for key in keys:
            match = re.search(r'diffusion_model\.blocks\.(\d+)\.', key)
            if match:
                blocks.add(f"MMB{int(match.group(1)):02d}")
            match = re.search(r'diffusion_model\.token_refiner\.blocks\.(\d+)\.', key)
            if match:
                blocks.add(f"TREF{int(match.group(1)):02d}")
            if key.startswith('diffusion_model.final_layer.'):
                blocks.add("FINAL")
        return classified("minimax_h3")

    # --- Z-Image (zimage_adapter.py:99/117) --------------------------------
    # lora_transformer_{layers|noise_refiner|context_refiner}_<N>_attention_*.
    # "transformer.layers." is the dotted spelling the generation loader also
    # accepts (repaired load-side in 1b0a192c; the save side deliberately keeps
    # writing the flattened form, so BOTH must yield FDiT blocks here).
    is_zimage = any(
        'noise_refiner' in key or 'context_refiner' in key
        or key.startswith('lora_transformer_layers_')
        or 'transformer.layers.' in key
        for key in keys
    )
    if is_zimage:
        for key in keys:
            match = re.search(r'noise_refiner[_.](\d+)', key)
            if match:
                blocks.add(f"NRef{int(match.group(1))}")
            match = re.search(r'context_refiner[_.](\d+)', key)
            if match:
                blocks.add(f"CRef{int(match.group(1))}")
            match = (re.match(r'lora_transformer_layers_(\d+)_', key)
                     or re.search(r'transformer\.layers\.(\d+)\.', key))
            if match:
                blocks.add(f"FDiT{int(match.group(1)):02d}")
        return classified("zimage")

    # --- FLUX.2 (flux2_adapter.py:84/122, TE at :210) ----------------------
    # lora_transformer_(single_)transformer_blocks_<N>_*, plus
    # lora_te_model_layers_<N>_* for a text-encoder-only run (train_unet=False).
    # The TE root is narrow on purpose: kohya SD1.5 files spell their single
    # text encoder "lora_te_text_model_encoder_layers_<N>_*".
    is_flux2 = any(
        key.startswith('lora_transformer_transformer_blocks_')
        or key.startswith('lora_transformer_single_transformer_blocks_')
        or key.startswith('lora_te_model_layers_')
        for key in keys
    )
    if is_flux2:
        for key in keys:
            match = re.match(r'lora_transformer_transformer_blocks_(\d+)_', key)
            if match:
                blocks.add(f"DUAL{int(match.group(1)):02d}")
            match = re.match(r'lora_transformer_single_transformer_blocks_(\d+)_', key)
            if match:
                blocks.add(f"SING{int(match.group(1)):02d}")
        return classified("flux2")

    # --- Ideogram 4 (ideogram4_adapter.py:60, iter_ideogram4_lora_targets) --
    # lora_unet_layers_<N>_{attention_to_*,feed_forward_w*,adaln_modulation};
    # the optional unconditional twin repeats the same stem under lora_uncond_.
    if any(re.match(r'lora_(?:unet|uncond)_layers_\d+_', key) for key in keys):
        for key in keys:
            match = re.match(r'lora_unet_layers_(\d+)_', key)
            if match:
                blocks.add(f"FDiT{int(match.group(1)):02d}")
            match = re.match(r'lora_uncond_layers_(\d+)_', key)
            if match:
                blocks.add(f"UDiT{int(match.group(1)):02d}")
        return classified("ideogram4")

    # --- Anima (anima_adapter.py:114, anima_lora._flatten_to_sdscripts) ----
    # lora_unet_blocks_<N>_{self_attn,cross_attn,mlp,adaln_modulation_*}_* and
    # lora_unet_llm_adapter_{blocks_<N>_*,in_proj,out_proj}.
    if any(re.match(r'lora_unet_(?:blocks_\d+_|llm_adapter_)', key) for key in keys):
        for key in keys:
            match = re.match(r'lora_unet_blocks_(\d+)_', key)
            if match:
                blocks.add(f"DIT{int(match.group(1)):02d}")
                continue
            match = re.match(r'lora_unet_llm_adapter_blocks_(\d+)_', key)
            if match:
                blocks.add(f"LAD{int(match.group(1)):02d}")
            elif key.startswith('lora_unet_llm_adapter_'):
                blocks.add("LAPROJ")
        return classified("anima")

    # --- ACE-Step 1.5 (acestep_adapter.py:150, iter_acestep_lora_targets) --
    # lora_unet_decoder_layers_<N>_{self_attn,cross_attn,mlp}_*_proj.
    if any(re.match(r'lora_unet_decoder_layers_\d+_', key) for key in keys):
        for key in keys:
            match = re.match(r'lora_unet_decoder_layers_(\d+)_', key)
            if match:
                blocks.add(f"L{int(match.group(1)):02d}")
        return classified("acestep")

    # --- MiniT2I (minit2i_lora.flatten_to_key / flatten_to_te_key) ---------
    # Flattens "." to "__", so the roots are lora_unet_model__net__* and
    # lora_te_encoder__block__* (FLAN-T5, train_text_encoder) -- disjoint from
    # every single-underscore root by construction.
    if any(key.startswith('lora_unet_model__net__')
           or key.startswith('lora_te_encoder__block__') for key in keys):
        for key in keys:
            match = re.match(r'lora_unet_model__net__double_blocks__(\d+)__', key)
            if match:
                blocks.add(f"MMB{int(match.group(1)):02d}")
                continue
            match = re.match(r'lora_unet_model__net__txt_preamble_blocks__(\d+)__', key)
            if match:
                blocks.add(f"TPRE{int(match.group(1)):02d}")
            elif re.match(r'lora_unet_model__net__(?:txt|pooled)_embedder', key):
                blocks.add("EMB")
        return classified("minit2i")

    # --- Krea 2 (krea2_lora.flatten_to_key) --------------------------------
    # Also "__"-flattened: lora_unet_transformer_blocks__<N>__{attn,ff}__*, plus
    # the opt-in text_fusion and projection scopes. Tested before the
    # single-underscore transformer_blocks family below, whose regex these stems
    # cannot match anyway (the char after "transformer_blocks_" is "_", not a digit).
    krea2_proj_roots = ('lora_unet_img_in', 'lora_unet_txt_in__',
                        'lora_unet_final_layer__', 'lora_unet_time_embed__',
                        'lora_unet_time_mod_proj')
    if any(key.startswith(('lora_unet_transformer_blocks__',
                           'lora_unet_text_fusion__') + krea2_proj_roots)
           for key in keys):
        for key in keys:
            match = re.match(r'lora_unet_transformer_blocks__(\d+)__', key)
            if match:
                blocks.add(f"MMB{int(match.group(1)):02d}")
                continue
            match = re.match(r'lora_unet_text_fusion__layerwise_blocks__(\d+)__', key)
            if match:
                blocks.add(f"TFL{int(match.group(1)):02d}")
                continue
            match = re.match(r'lora_unet_text_fusion__refiner_blocks__(\d+)__', key)
            if match:
                blocks.add(f"TFR{int(match.group(1)):02d}")
            elif key.startswith('lora_unet_text_fusion__projector'):
                blocks.add("TFP")
            elif key.startswith(krea2_proj_roots):
                blocks.add("PROJ")
        return classified("krea2")

    # --- Lens / LTX-2.3 / MiniMax-H3 (sd-scripts native) -------------------
    # All three write lora_unet_transformer_blocks_<N>_<leaf> and are told apart
    # by the leaf, because their target iterators are disjoint:
    #   lens_lora.iter_lens_lora_targets      -> attn_{img,txt}_qkv, attn_to_add_out,
    #                                            {img,txt}_mlp_w*, {img,txt}_mod_*
    #   ltx2_adapter.iter_ltx2_lora_targets   -> attn1/attn2, audio_*, *_to_*_attn_
    #   minimax_h3_adapter.iter_..._targets   -> attn_to_{q,k,v,out_0}, ff_net_*
    # LTX-2.3's opt-in ff leaves are spelled exactly like MiniMax-H3's
    # (ff_net_0_proj / ff_net_2), so an ff-ONLY LTX-2.3 file would read as
    # minimax_h3; it is unreachable because lora_trainer.py:231 resolves ltx2
    # "attention" to True for every scope string, leaving attn1/attn2 to break
    # the tie. A stem matching none of the three falls through to the catch-all
    # rather than being guessed at.
    dit_blocks = [
        (int(match.group(1)), match.group(2))
        for match in (re.match(r'lora_unet_transformer_blocks_(\d+)_(.+)$', key)
                      for key in keys)
        if match is not None
    ]
    if dit_blocks:
        leaves = [leaf for _, leaf in dit_blocks]
        families = (
            ("lens", "DUAL", r'(?:attn_(?:img|txt)_qkv|attn_to_add_out'
                             r'|(?:img|txt)_mlp_w|(?:img|txt)_mod_)'),
            ("ltx2", "MMB", r'(?:attn[12]_|audio_attn[12]_'
                            r'|audio_to_video_attn_|video_to_audio_attn_)'),
            ("minimax_h3", "MMB", r'(?:attn_to_[qkv]|attn_to_out_|ff_net_)'),
        )
        for arch, label, leaf_pattern in families:
            if any(re.match(leaf_pattern, leaf) for leaf in leaves):
                for index, _ in dit_blocks:
                    blocks.add(f"{label}{index:02d}")
                return classified(arch)

    # --- SD1.5 / SDXL (kohya-ss "lora_unet_*"/"lora_te*_*" or diffusers dot
    # format). The catch-all, and it must stay LAST: it accepts ANY lora_unet_/
    # lora_te prefix, which is why every signature above is anchored and tested
    # first.
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
        return classified("sdxl" if has_te2 else "sd15")

    # --- Unknown / unrecognized structure ------------------------------------
    return classified("unknown")


# ---------------------------------------------------------------------------
# Adapter (LoRA family) detection for the LISTING path
# ---------------------------------------------------------------------------
# One detector, shared with generation: core.adapters.codec.CodecRegistry.
# What is reported here has to PREDICT what AdapterSession does with the same
# file, so this mirrors its rule rather than inventing a second one -- ordinary
# LoRA and an unnamed algebra are not validated there, so neither can be
# reported invalid here.

ADAPTER_STATE_OK = "ok"
ADAPTER_STATE_UNKNOWN = "unknown"
ADAPTER_STATE_INVALID = "invalid"

_UNNAMED_ALGEBRA = ("no metadata key and no tensor-key signature names this "
                    "file's adapter algebra")


class _HeaderTensor:
    """Shape-only stand-in for one safetensors entry.

    Detection reads SHAPES; materializing the tensors of every file in a LoRA
    directory to read them would read the whole directory off disk. ``item()``
    fetches the one tensor it is asked for -- only a scalar ``.alpha`` reaches
    it -- and is valid only while the owning ``safe_open`` handle is open.
    """

    __slots__ = ("shape", "_key", "_handle")

    def __init__(self, shape, key: str, handle):
        self.shape = tuple(int(d) for d in shape)
        self._key = key
        self._handle = handle

    def numel(self) -> int:
        count = 1
        for dim in self.shape:
            count *= dim
        return count

    def item(self):
        return self._handle.get_tensor(self._key).item()


def _header_shape(handle, key: str):
    """A key's shape, or ``()`` if this file's header will not give one -- a
    shape read must degrade the DETECTION, never drop the file from the list.
    """
    try:
        return handle.get_slice(key).get_shape()
    except Exception:
        return ()


def _unknown_adapter_fields(reason: str) -> Dict[str, Any]:
    return {
        "adapter_type": "unknown",
        "adapter_algorithm": "unknown",
        "weight_decompose": False,
        "adapter_format": "unknown",
        "adapter_state": ADAPTER_STATE_UNKNOWN,
        "adapter_state_reason": reason,
        "adapter_rank": None,
        "adapter_alpha": None,
    }


def detect_adapter_fields(tensors, metadata,
                          architecture: Optional[str] = None) -> Dict[str, Any]:
    """The adapter description `GET /loras` reports for one checkpoint.

    ``tensors`` may be shape-only views (`_HeaderTensor`). ``architecture`` is
    the one classify_lora_keys() read off the KEYS, used only to keep a bogus
    "unknown" out of the spec; the ARCHITECTURE axis of ``validate()`` is
    neutralised below, and only the ALGEBRA axes decide `adapter_state`.
    """
    from core.adapters.codec import CodecRegistry
    from core.adapters.session import AdapterRefusal
    from core.adapters.spec import (ALGORITHM_UNKNOWN, AdapterSpec,
                                    KNOWN_ARCHITECTURES)

    try:
        codec = CodecRegistry.detect(tensors, dict(metadata or {}))
    except Exception as e:
        # The same carve-out AdapterSession._canonicalize makes: detection
        # indexes shapes it has not validated, and a valid `lora_bias=True`
        # PEFT export's 1-D `.lora_A.bias` used to raise here. Unknown is a
        # report, never a refusal.
        return _unknown_adapter_fields(
            f"adapter detection failed ({type(e).__name__})")

    fields = {
        "adapter_algorithm": codec.algorithm,
        "weight_decompose": bool(codec.weight_decompose),
        "adapter_format": codec.format,
        "adapter_rank": None if codec.rank is None else int(codec.rank),
        "adapter_alpha": None if codec.alpha is None else float(codec.alpha),
    }

    if architecture not in KNOWN_ARCHITECTURES:
        architecture = None
    spec = AdapterSpec.from_codec(codec, architecture=architecture)
    fields["adapter_type"] = spec.family

    state, reason = ADAPTER_STATE_OK, None
    if codec.algorithm == ALGORITHM_UNKNOWN:
        state, reason = ADAPTER_STATE_UNKNOWN, _UNNAMED_ALGEBRA
    elif (codec.algorithm, bool(codec.weight_decompose)) != ("lora", False):
        try:
            # `known_architectures={spec.architecture}` makes that one check
            # pass by construction. It is not answerable here: the listing has
            # no loaded model, and `from_codec` falls back to the file's own
            # `model_type` when the caller passes none -- so a kohya file
            # declaring `sdxl_base_v1-0` would be reported broken while
            # generating fine on every enabled architecture. AdapterSession
            # never reaches that arm (it always passes the loaded arch, so the
            # fallback never fires), which is why the axis is a listing-only
            # false positive.
            spec.validate(known_architectures={spec.architecture})
        except AdapterRefusal as error:
            state = ADAPTER_STATE_INVALID
            reason = getattr(error, "message", None) or str(error)
    fields["adapter_state"] = state
    fields["adapter_state_reason"] = reason
    return fields


def _has_adapter_keys(keys, adapter: Dict[str, Any]) -> bool:
    """Whether this file is an ADAPTER checkpoint rather than a full fine-tune.

    The four historical key-prefix arms are unchanged. The fifth admits a
    LyCORIS file whose stems satisfy none of them: Z-Image's flattened
    ``lora_transformer_layers_0_attn_to_q.hada_w1_a`` was filtered out of the
    list entirely on an architecture that loads and generates it.
    """
    has_lora_down = any('lora_down' in key for key in keys)
    has_lora_up = any('lora_up' in key for key in keys)
    has_lora_A = any('.lora_A.' in key for key in keys)
    has_lora_B = any('.lora_B.' in key for key in keys)
    has_lora_unet = any('lora_unet' in key for key in keys)
    has_lora_te = any('lora_te' in key for key in keys)
    # Z-Image LoRA format: transformer.layers.0.attn1.to_q.lora_down.weight
    has_lora_transformer = any(
        'transformer.' in key and ('lora_down' in key or 'lora_up' in key)
        for key in keys)

    return ((has_lora_down and has_lora_up)
            or (has_lora_A and has_lora_B)
            or has_lora_unet or has_lora_te
            or has_lora_transformer
            or adapter.get("adapter_algorithm") in ("loha", "lokr"))


def _recommended_from_metadata(name: str,
                               metadata: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """A step-distillation recommendation the file itself declares, or None.

    Only `student_steps` is recognized. `num_inference_steps` counts sigma grid
    points INCLUDING the terminal 0, one more than the model evaluations the
    file names, hence `+ 1`. FBCache/Spectrum are recommended off because both
    amortize bookkeeping across dozens of steps.
    """
    student_steps_raw = (metadata or {}).get("student_steps")
    if student_steps_raw is None:
        return None

    try:
        student_steps = int(float(student_steps_raw))
    except (TypeError, ValueError):
        print(f"[LoRAManager] {name}: unparseable student_steps={student_steps_raw!r}")
        return None

    return {
        "num_inference_steps": student_steps + 1,
        "fbcache_enable": False,
        "spectrum_enable": False,
        "source": "student_steps",
    }


def _sort_lora_blocks(blocks) -> List[str]:
    """Sort block labels: BASE, IN00-IN.., MID, OUT00-.. (SD/SDXL); NRef/CRef/
    FDiT (Z-Image); FDiT/UDiT (Ideogram 4 cond/uncond twins); DUAL/SING (FLUX.2),
    DUAL alone (Lens); DIT/LAD/LAPROJ (Anima DiT blocks, LLM-adapter blocks,
    LLM-adapter projections); TPRE/MMB/EMB (MiniT2I); MMB/TFL/TFR/TFP/PROJ
    (Krea 2); MMB (LTX-2.3); TREF/MMB/FINAL (MiniMax-H3); L00-.. (SenseNova
    layers, ACE-Step decoder layers).

    Each architecture's own labels carry distinct group indices, so its list is
    totally ordered; indices are reused ACROSS architectures, which is harmless
    because a file is only ever labelled by one of them.
    """
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
        elif block.startswith("UDiT"):
            return (4, int(block[4:]))
        elif block.startswith("DUAL"):
            return (1, int(block[4:]))
        elif block.startswith("SING"):
            return (2, int(block[4:]))
        elif block.startswith("DIT"):
            return (1, int(block[3:]))
        elif block.startswith("TREF"):
            return (1, int(block[4:]))
        elif block.startswith("TPRE"):
            return (1, int(block[4:]))
        elif block.startswith("TFL"):
            return (3, int(block[3:]))
        elif block.startswith("TFR"):
            return (4, int(block[3:]))
        elif block == "TFP":
            return (5, 0)
        elif block == "EMB":
            return (3, 0)
        elif block == "PROJ":
            return (6, 0)
        elif block == "FINAL":
            return (3, 0)
        elif block.startswith("MMB"):
            return (2, int(block[3:]))
        elif block == "LAPROJ":
            return (3, 0)
        elif block.startswith("LAD"):
            return (2, int(block[3:]))
        elif block.startswith("L"):
            return (1, int(block[1:]))
        return (9, 0)

    return sorted(list(blocks), key=sort_key)


# Components a diffusers pipeline can install a LoRA into. Tokenizers,
# schedulers and the VAE are never LoRA targets on this path, so walking
# `pipeline.components` would only add work and None entries.
_LORA_COMPONENT_ATTRS = ("unet", "transformer", "text_encoder",
                         "text_encoder_2", "text_encoder_3")

# Where a PEFT layer keeps its per-adapter branch: `lora_A` for Linear/Conv
# targets, `lora_embedding_A` for Embedding ones.
_LORA_BRANCH_ATTRS = ("lora_A", "lora_embedding_A")


def _lora_warn(message: str, code: str) -> None:
    """Record a user-visible generation warning (best effort)."""
    try:
        from api.generation_status import add_warning
        add_warning(message, code=code)
    except Exception:
        pass


def _count_applied_lora_targets(pipeline: Any, adapter_name: str) -> Tuple[int, bool]:
    """Read back how much of ``adapter_name`` diffusers/PEFT actually installed.

    ``load_lora_into_unet`` / ``load_lora_into_text_encoder`` return nothing and
    no-op silently when the checkpoint names no module of the component, so the
    only count available is the one left inside the model.

    ``registered`` is the weaker witness that PEFT accepted the adapter at all
    (it is in a component's ``peft_config``, which diffusers pops again when
    injection fails). It is there so a PEFT layer class whose branch is under
    neither name in ``_LORA_BRANCH_ATTRS`` costs a count, not a false refusal.
    """
    targets = 0
    registered = False
    for attr in _LORA_COMPONENT_ATTRS:
        component = getattr(pipeline, attr, None)
        if component is None or not hasattr(component, "named_modules"):
            continue
        peft_config = getattr(component, "peft_config", None)
        try:
            registered = registered or (peft_config is not None and adapter_name in peft_config)
        except TypeError:
            pass
        for _name, module in component.named_modules():
            for branch_attr in _LORA_BRANCH_ATTRS:
                container = getattr(module, branch_attr, None)
                if container is None:
                    continue
                try:
                    if adapter_name in container:
                        targets += 1
                        break
                except TypeError:
                    continue
    return targets, registered


def _count_lora_branch_pairs(keys) -> int:
    """Down/up pairs present in the checkpoint, in either naming convention."""
    return sum(1 for key in keys
               if key.endswith(".lora_down.weight") or key.endswith(".lora_A.weight"))


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
        # Each entry: {"path": identifier, "name": str, "arch": str, adapter fields}
        self._lora_cache: Optional[List[Dict[str, Any]]] = None
        self._cache_timestamp: float = 0.0

        # Per-FILE header probe, keyed by path -> ((mtime_ns, size), record).
        # Survives invalidate_cache()/force_rescan: a rescan exists to notice
        # added, removed or edited files, and re-reading the unchanged ones is
        # what made it linear in the whole directory.
        self._probe_cache: Dict[str, Tuple[Tuple[int, int], Optional[Dict[str, Any]]]] = {}

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
        return dir_tag(directory)

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

    def _read_lora_header(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """One HEADER read of one file: keys, shapes, metadata -- never tensor
        data. ``None`` when the file cannot be read at all.

        Everything the list, the details endpoint and the block graph report
        comes from here, so a file is opened once rather than three times.
        """
        try:
            from safetensors import safe_open

            with safe_open(file_path, framework="pt", device="cpu") as f:
                keys = list(f.keys())
                metadata = f.metadata() or {}
                header = {k: _HeaderTensor(_header_shape(f, k), k, f)
                          for k in keys}
                classification = classify_lora_keys(keys)
                arch = classification.get("arch", "unknown")
                adapter = detect_adapter_fields(header, metadata, arch)
        except Exception as e:
            print(f"[LoRAManager] Could not read {file_path.name}: {e}")
            return None

        return {
            "arch": arch,
            "blocks": classification.get("blocks", []),
            "adapter": adapter,
            "is_adapter": _has_adapter_keys(keys, adapter),
            "recommended": _recommended_from_metadata(file_path.name, metadata),
        }

    @staticmethod
    def _probe_key(file_path: Path) -> str:
        return os.path.normcase(str(file_path))

    def _probe_lora_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """`_read_lora_header`, cached on (path, mtime, size).

        A rescan exists to notice added, removed and edited files; re-reading
        the unchanged ones is what made it cost the whole directory every time.
        A ``None`` record (unreadable file) is cached too.
        """
        try:
            stat = file_path.stat()
        except OSError:
            return None
        key = self._probe_key(file_path)
        stamp = (stat.st_mtime_ns, stat.st_size)
        cached = self._probe_cache.get(key)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        record = self._read_lora_header(file_path)
        self._probe_cache[key] = (stamp, record)
        return record

    def _is_valid_lora_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Validate if a file is an adapter (LoRA-family) checkpoint and, if so,
        describe it: architecture via classify_lora_keys() (the single signature
        table shared with get_lora_layers()) and adapter family via the one
        detector the generation path uses.

        Checks:
        1. File extension (.safetensors only - .pt/.bin excluded to avoid debug latents)
        2. File contains adapter tensor keys (lora_down/lora_up, hada_*, lokr_*, ...)
        3. Excludes training artifacts (optimizer states, debug latents, etc.)

        Returns:
            The probe record (see `_read_lora_header`) if this is an adapter
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

        record = self._probe_lora_file(file_path)
        if record is None:
            # If we can't read it, exclude it to be safe
            return None
        if not record["is_adapter"]:
            print(f"[LoRAManager] Excluding non-LoRA file (full parameter fine-tune): {file_path.name}")
            return None
        return record

    def get_available_loras(self, force_rescan: bool = False) -> List[Dict[str, Any]]:
        """
        Get list of available LoRA files from default and additional/seeded
        directories.

        Uses cache to avoid expensive validation on every API call.

        Args:
            force_rescan: Force re-scanning and validation (ignores cache)

        Returns:
            List of {"path": identifier, "name": str, "arch": str, plus the
            detected adapter fields of `detect_adapter_fields()`} dicts.
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

        # rel_path -> list of (dir, abs_path, probe record), in scan
        # (= priority) order
        records_by_rel: Dict[str, List[Tuple[Path, Path, Dict[str, Any]]]] = {}
        # Every candidate this scan saw, probed or not, for the cache prune
        # below.
        seen: set = set()

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
                    seen.add(self._probe_key(f))
                    record = self._is_valid_lora_file(f)
                    if record is not None:
                        rel = str(f.relative_to(lora_dir))
                        records_by_rel.setdefault(rel, []).append((lora_dir, f, record))

        result: List[Dict[str, Any]] = []
        for rel, recs in records_by_rel.items():
            for idx, (lora_dir, abs_path, record) in enumerate(recs):
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
                    "arch": record["arch"],
                    **record["adapter"],
                })

        # Drop probe entries for files this scan did not find. The training
        # output directory is a search path, so a long run writing checkpoints
        # would otherwise grow the cache for the life of the process.
        self._probe_cache = {k: v for k, v in self._probe_cache.items()
                             if k in seen}

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

    def _refuse_weight_decomposed(self, lora_path, lora_file: str) -> None:
        """Refuse a weight-decomposed adapter BEFORE it reaches diffusers.

        `lora_state_dict` DROPS every `dora_scale` key with a log line and
        nothing else, so a DoRA here applies and reports SUCCESS as an ordinary
        LoRA -- the one silent-wrong-answer case on this path. LoHa/LoKr are
        deliberately left alone: the Kohya converter already raises on their
        unrenamed keys. Measured on diffusers 0.38.0; see the design doc,
        phase 3.
        """
        from api.error_handlers import with_error_code

        record = self._probe_lora_file(Path(lora_path))
        adapter = (record or {}).get("adapter") or {}
        if not adapter.get("weight_decompose"):
            return
        family = adapter.get("adapter_type") or "weight-decomposed"
        message = (
            f"LoRA '{lora_file}' is a {family} adapter (it carries per-target "
            f"dora_scale magnitude vectors). This model loads adapters through "
            f"diffusers, which discards every dora_scale key before applying "
            f"the file -- it would run as an ordinary LoRA at the wrong "
            f"numbers rather than fail. Use it on an architecture whose "
            f"capability row enables it."
        )
        print(f"[LoRAManager] ERROR: {message}")
        _lora_warn(message, code="lora_incompatible")
        raise with_error_code(RuntimeError(message), "lora_incompatible")

    def load_loras(self, pipeline: Any, lora_configs: List[Dict[str, Any]]) -> Any:
        """
        Load multiple LoRAs into the pipeline

        Args:
            pipeline: Diffusers pipeline
            lora_configs: List of LoRA configurations

        Returns:
            Modified pipeline with LoRAs loaded

        Raises:
            FileNotFoundError / RuntimeError when a requested LoRA cannot be
            applied at all. A requested-but-ineffective LoRA must not produce a
            successful generation. The caller unloads in a ``finally``, so a
            refusal here leaves no adapter behind.
        """
        from api.error_handlers import with_error_code

        print(f"[LoRAManager] load_loras called with {len(lora_configs) if lora_configs else 0} configs")
        print(f"[LoRAManager] lora_configs: {lora_configs}")

        if not lora_configs:
            print("[LoRAManager] No LoRA configs provided, skipping")
            return pipeline

        # Parse configs
        self.loaded_loras = [LoRAConfig.from_dict(cfg) for cfg in lora_configs]
        print(f"[LoRAManager] Parsed {len(self.loaded_loras)} LoRA configs")

        # Load LoRAs using diffusers' native support
        for i, lora_config in enumerate(self.loaded_loras):
            # Warnings ride into the PNG metadata chunk and the API response,
            # so they name the basename and never a path.
            lora_file = os.path.basename(str(lora_config.path))
            lora_path = self._resolve_lora_path(lora_config.path)

            if lora_path is None:
                message = (
                    f"LoRA '{lora_file}' was requested but no such file exists in the "
                    f"registered LoRA directories -- refusing to generate without it."
                )
                print(f"[LoRAManager] ERROR: {message}")
                print(f"[LoRAManager]   Searched in: {self.lora_dir}")
                print(f"[LoRAManager]   Additional dirs: {self.additional_dirs}")
                _lora_warn(message, code="lora_not_found")
                raise with_error_code(FileNotFoundError(message), "lora_not_found")

            # Outside the try below on purpose: that block re-wraps everything
            # it catches as lora_load_failed, and this refusal has its own code.
            self._refuse_weight_decomposed(lora_path, lora_file)

            adapter_name = f"lora_{i}"
            file_pairs = 0
            sample_keys: List[str] = []
            applied, registered = 0, False
            try:
                print(f"[LoRAManager] Attempting to load LoRA from: {lora_path}")
                print(f"[LoRAManager] LoRA config: strength={lora_config.strength}, apply_to_text_encoder={lora_config.apply_to_text_encoder}, apply_to_unet={lora_config.apply_to_unet}")

                print(f"[LoRAManager] Loading LoRA {i+1}/{len(self.loaded_loras)}: {lora_config.path}")

                # Detect LoRA format and convert if needed
                # (`os` is module-scope: a local `import os` here would make the
                # name local to the whole method and break the basename above.)
                from safetensors import safe_open
                import tempfile

                # Check LoRA format
                with safe_open(str(lora_path), framework="pt", device="cpu") as f:
                    file_keys = list(f.keys())
                    file_pairs = _count_lora_branch_pairs(file_keys)
                    sample_keys = file_keys[:5]
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

                    # Load converted LoRA. The cleanup is in a finally because a
                    # load failure now raises rather than falling through.
                    try:
                        pipeline.load_lora_weights(
                            temp_dir,
                            weight_name=f"converted_lora_{adapter_name}.safetensors",
                            adapter_name=adapter_name
                        )
                    finally:
                        try:
                            os.remove(temp_lora_path)
                            print(f"[LoRAManager] Temporary file removed")
                        except OSError:
                            pass
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

                applied, registered = _count_applied_lora_targets(pipeline, adapter_name)
                print(f"[LoRAManager] Adapter '{adapter_name}' installed on {applied} module(s) "
                      f"(file has {file_pairs} down/up pair(s), peft_config registered={registered})")

            except Exception as e:
                print(f"[LoRAManager] ERROR loading LoRA {lora_path}: {e}")
                import traceback
                traceback.print_exc()
                # Type + basename only: this text rides into the PNG chunk and
                # the API response, and an OSError's str() carries the absolute
                # resolved path. PEFT also raises here when the checkpoint names
                # no module of a component ("Target modules ... not found").
                message = (f"LoRA '{lora_file}' could not be applied "
                           f"({type(e).__name__}); see the server log for details")
                _lora_warn(message, code="lora_load_failed")
                raise with_error_code(RuntimeError(message), "lora_load_failed") from e

            # Refuse BEFORE set_adapters: with zero targets installed that call
            # raises too, and its generic failure would mask the real reason.
            # diffusers no-ops silently when the checkpoint names no module of a
            # component, so the read-back count is the only signal there is.
            if applied == 0 and not registered:
                message = (
                    f"LoRA '{lora_file}': 0 of {file_pairs} down/up pair(s) applied to the "
                    f"loaded model -- unrecognized key format or a LoRA for a different "
                    f"architecture. Expected stems like 'lora_unet_*' / 'lora_te1_*' "
                    f"(or the diffusers 'unet.*' / 'text_encoder.*' spelling). "
                    f"Sample keys in file: {sample_keys}"
                )
                print(f"[LoRAManager] ERROR: {message}")
                _lora_warn(message, code="lora_incompatible")
                raise with_error_code(RuntimeError(message), "lora_incompatible")

            if 0 < applied < file_pairs:
                _lora_warn(
                    f"LoRA '{lora_file}': applied {applied} of {file_pairs} down/up pair(s); "
                    f"the rest name modules the loaded model does not have.",
                    code="lora_partial",
                )

            try:
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
                    if getattr(pipeline, "unet", None) is not None:
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

                else:
                    print(f"[LoRAManager] WARNING: Pipeline does not have set_adapters method")

            except Exception as e:
                print(f"[LoRAManager] ERROR activating LoRA {lora_path}: {e}")
                import traceback
                traceback.print_exc()
                message = (f"LoRA '{lora_file}' could not be applied "
                           f"({type(e).__name__}); see the server log for details")
                _lora_warn(message, code="lora_load_failed")
                raise with_error_code(RuntimeError(message), "lora_load_failed") from e

        # The activation above names ONE adapter, and set_adapters REPLACES the
        # active set rather than adding to it, so without this every LoRA but
        # the last was installed, counted, reported -- and silently inactive.
        self.activate_adapters(
            pipeline,
            [f"lora_{i}" for i in range(len(self.loaded_loras))],
            [cfg.strength for cfg in self.loaded_loras],
        )

        print(f"[LoRAManager] Successfully loaded {len(self.loaded_loras)} LoRA(s)")

        return pipeline

    def activate_adapters(self, pipeline: Any, names: List[str],
                          weights: List[float]) -> None:
        """Make exactly ``names`` the active set, then re-apply per-block weights.

        ``set_adapters`` recomputes each named adapter's ``scaling`` from its
        weight, so ``unet_layer_weights`` has to be folded in AFTERWARDS or it
        is silently discarded -- which is what happened to a LoRA carrying both
        block weights and a ``step_range``, because the step callback reactivates
        every step. ``_apply_layer_weights`` multiplies the current scaling, so
        it is correct exactly once per activation.
        """
        if not names or not hasattr(pipeline, 'set_adapters'):
            return
        pipeline.set_adapters(names, adapter_weights=weights)
        if getattr(pipeline, "unet", None) is None:
            return
        for name in names:
            index = int(name.rsplit("_", 1)[1])
            config = self.loaded_loras[index]
            if config.unet_layer_weights:
                self._apply_layer_weights(pipeline, name, config)

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
                    self.activate_adapters(pipeline, active_adapters, adapter_weights)
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

        # arch, blocks, adapter family and the file's own recommendation all
        # come from ONE cached header read.
        record = self._probe_lora_file(lora_path) or {}

        return {
            "name": lora_name,
            "path": str(lora_path),
            "size": lora_path.stat().st_size,
            "exists": True,
            "arch": record.get("arch", "unknown"),
            "layers": record.get("blocks", []),
            "recommended": record.get("recommended"),
            **(record.get("adapter")
               or _unknown_adapter_fields("file could not be read")),
        }

    def adapter_report(self, lora_name: str) -> Optional[Dict[str, Any]]:
        """The detected adapter fields for a LoRA IDENTIFIER (the same ones
        `GET /loras` reports), or None when it does not resolve unambiguously
        or cannot be read -- those are the generation path's own refusals.
        """
        try:
            lora_path = self._resolve_lora_path(lora_name)
        except LoRAAmbiguousIdentifierError:
            return None
        if lora_path is None:
            return None
        record = self._probe_lora_file(lora_path)
        return None if record is None else record["adapter"]

    def get_lora_layers(self, lora_name: str) -> List[str]:
        """
        Extract U-Net/transformer block structure from a LoRA file.
        Returns the labels _sort_lora_blocks() documents, for whichever
        architecture classify_lora_keys() detected (BASE when a file carries no
        block-structured keys, e.g. a text-encoder-only checkpoint).
        """
        # Use _resolve_lora_path to check both lora/ and training/ directories
        lora_path = self._resolve_lora_path(lora_name)

        if lora_path is None:
            return []

        _, blocks = self._read_lora_keys_info(lora_path)
        return blocks

    def _read_lora_keys_info(self, lora_path: Path) -> Tuple[str, List[str]]:
        """A file's (arch, block labels), from the cached header probe."""
        record = self._probe_lora_file(lora_path)
        if record is None:
            return "unknown", []
        arch, blocks = record["arch"], record["blocks"]
        print(f"[LoRAManager] {lora_path.name}: arch={arch}, {len(blocks)} blocks: {blocks}")
        return arch, blocks

    def _parse_recommended_metadata(self, lora_path: Path) -> Optional[Dict[str, Any]]:
        """The file's own step-distillation recommendation, or None.
        See `_recommended_from_metadata` for what is recognized."""
        record = self._probe_lora_file(lora_path)
        return None if record is None else record["recommended"]


# Global instance
lora_manager = LoRAManager()
