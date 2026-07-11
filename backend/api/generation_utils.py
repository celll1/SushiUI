"""
Generation endpoint shared utilities
生成エンドポイント共通ユーティリティ

このモジュールは、txt2img/img2img/inpaintエンドポイント間のコード重複を削減します。
"""
from typing import List, Dict, Any, Optional, Callable
from PIL import Image
import base64
from io import BytesIO
import os


# ====================
# Priority 1: 最も重複が多い関数群
# ====================

def process_controlnet_configs(
    controlnet_configs: List[Dict],
    generation_type: str = "txt2img"
) -> "tuple[List[Dict], Optional[Dict], List[Dict], str]":
    """
    ControlNet設定を処理し、base64画像をデコード

    重複削減: 105行 → 35行（70行削減）

    An entry with ``is_style_transfer: true`` is NOT a ControlNet: it carries a
    training-free reference-style-transfer request (StyleAligned/VSP-style KV
    injection, see ``core.inference.reference_style``). It is extracted into
    a separate style-config dict (arch-agnostic keys) instead of being appended
    to the returned ControlNet image list.

    Multi-reference (N-ref) support: MULTIPLE ``is_style_transfer`` entries are
    now collected into ``style_transfers`` (one dict per reference, in the
    order the caller supplied them), each still built with its own knobs
    (``ref_k_strength``, ``block_range``, freq curve, etc. -- see the dict
    below). ``style_transfer`` (singular) stays populated with
    ``style_transfers[0]`` for back-compat with the architectures that only
    read the single key (this is a FIRST-wins semantic; the pre-multi-ref
    code was last-wins by accident when multiple ``is_style_transfer`` entries
    were present, which never happened in practice since only one was ever
    sent). ``style_combine_mode`` (``"stack"`` or ``"common_concept"``) is
    read from the FIRST style entry's ``style_combine_mode`` key (frontend
    multi-ref selector, not yet wired at intake time as of this change) and
    defaults to ``"stack"``.

    Args:
        controlnet_configs: ControlNet設定のリスト
        generation_type: 生成タイプ（ログ用）

    Returns:
        (処理済みのControlNet画像リスト, style_transfer dict or None,
         style_transfers list (0+ entries), style_combine_mode str)
    """
    controlnet_images = []
    style_transfer = None
    style_transfers: List[Dict] = []
    style_combine_mode = "stack"
    if not controlnet_configs:
        return controlnet_images, style_transfer, style_transfers, style_combine_mode

    print(f"Processing {len(controlnet_configs)} ControlNet(s)...")

    for idx, cn_config in enumerate(controlnet_configs):
        if cn_config.get("is_style_transfer"):
            if not cn_config.get("image_base64"):
                print(f"[ControlNet {idx}] WARNING: is_style_transfer entry has no image_base64; skipping.")
                continue
            try:
                image_data = base64.b64decode(cn_config["image_base64"])
                image = Image.open(BytesIO(image_data)).convert("RGB")
                print(f"[StyleTransfer {idx}] Reference image decoded successfully: {image.size}")
                entry = {
                    "image": image,
                    # transfer_type selects the injection recipe: "style" (default,
                    # appearance/texture) or "character" (identity — early blocks +
                    # raw reference Value + minimal AdaIN). Carried through to
                    # StyleTransferConfig; the arch injection path is shared.
                    "transfer_type": cn_config.get("style_transfer_type", "style"),
                    # No literal fallback here: let None propagate so
                    # style_config_from_dict applies the StyleTransferConfig
                    # dataclass default (0.75). A hardcoded 1.0 here would be a
                    # second, stale source of truth that masks the dataclass
                    # default (matches how adain_strength below is wired).
                    "ref_k_strength": cn_config.get("strength"),
                    "adain_strength": cn_config.get("style_adain_strength"),
                    "block_range": cn_config.get("style_blocks"),
                    "start_step": cn_config.get("start_step", 0),
                    "end_step": cn_config.get("end_step", 1000),
                    # Frequency-scale curve (RoPE-freq content suppression). Emit the
                    # exact keys style_config_from_dict reads (start/end for both
                    # high and low) so they actually plumb -- the old "high_scale"
                    # key was silently dropped (style_config_from_dict reads
                    # high_scale_start/high_scale_end).
                    "high_scale_start": cn_config.get("style_high_scale_start"),
                    "high_scale_end": cn_config.get("style_high_scale_end"),
                    "low_scale_start": cn_config.get("style_low_scale_start"),
                    "low_scale_end": cn_config.get("style_low_scale_end"),
                    "beta": cn_config.get("style_beta"),
                    "value_mode": cn_config.get("style_value_mode"),
                    "value_adain_strength": cn_config.get("style_value_adain_strength"),
                    "ref_value_mix": cn_config.get("style_ref_value_mix"),
                    "late_release": cn_config.get("style_late_release"),
                    "rope_offset": cn_config.get("style_rope_offset"),
                }
                style_transfers.append(entry)
                if style_transfer is None:
                    style_transfer = entry
                    style_combine_mode = str(cn_config.get("style_combine_mode", "stack") or "stack")
            except Exception as e:
                print(f"[StyleTransfer {idx}] Error decoding reference image: {e}")
            continue

        print(f"[ControlNet {idx}] model_path: {cn_config.get('model_path')}, "
              f"has_image_base64: {bool(cn_config.get('image_base64'))}")

        if cn_config.get("image_base64"):
            try:
                image_data = base64.b64decode(cn_config["image_base64"])
                image = Image.open(BytesIO(image_data))
                print(f"[ControlNet {idx}] Image decoded successfully: {image.size}")

                controlnet_images.append({
                    "model_path": cn_config["model_path"],
                    "image": image,
                    "strength": cn_config.get("strength", 1.0),
                    "start_step": cn_config.get("start_step", 0.0),
                    "end_step": cn_config.get("end_step", 1.0),
                    "layer_weights": cn_config.get("layer_weights"),
                    "prompt": cn_config.get("prompt"),
                    "is_lllite": cn_config.get("is_lllite", False),
                    "is_reference_guide": cn_config.get("is_reference_guide", False),
                })
            except Exception as e:
                print(f"[ControlNet {idx}] Error decoding image: {e}")
        else:
            print(f"[ControlNet {idx}] WARNING: No image_base64 provided for {generation_type}. "
                  "ControlNet will be skipped.")

    print(f"[Routes] Total controlnet_images added to params: {len(controlnet_images)}")
    return controlnet_images, style_transfer, style_transfers, style_combine_mode


def create_progress_callback_factory(
    taesd_manager,
    websocket_manager,
    is_sdxl: bool,
    is_zimage: bool = False,
    is_deus: bool = False,
    is_zimage_sdxl_vae: bool = False,
    is_flux2: bool = False,
    is_anima: bool = False,
    is_lens: bool = False,
    is_ideogram4: bool = False,
    is_minit2i: bool = False,
    minit2i_vae_type: str = "none",
    is_krea2: bool = False,
    img2img_fix_steps: Optional[bool] = None,
    steps: Optional[int] = None,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    preview_predicted_x0: bool = False,
    preview_enabled: bool = True,
    preview_interval: int = 1,
    preview_decoder: str = "matrix"
) -> Callable:
    """
    WebSocketプログレスコールバックを生成

    重複削減: 63行 → 21行（42行削減）

    Args:
        taesd_manager: TAESD preview生成マネージャー
        websocket_manager: WebSocketマネージャー
        is_sdxl: SDXLモデルかどうか
        is_zimage: Z-Imageモデルかどうか
        is_deus: DEUSモデルかどうか
        is_zimage_sdxl_vae: Z-ImageでSDXL VAE（4ch）を使用しているかどうか
        is_flux2: FLUX.2モデルかどうか（32chLatent、TAESDプレビュー不可）
        img2img_fix_steps: img2img/inpaintの"Do full steps"オプション
        steps: ステップ数（display_total計算用）
        image_width: 生成画像の幅（FLUX.2プレビューのアスペクト比計算用）
        image_height: 生成画像の高さ（FLUX.2プレビューのアスペクト比計算用）
        preview_predicted_x0: Trueの場合、現在のlatentではなくpred_original_sample（推定x0）をプレビュー
        preview_enabled: プレビュー画像を生成するかどうか（転送量削減用）
        preview_interval: プレビューを送信するステップ間隔（1=毎ステップ、4=4ステップごと）

    Returns:
        プログレスコールバック関数
    """
    def progress_callback(step, total_steps, latents, cfg_metrics=None, pred_original_sample=None):
        # Calculate display_total for img2img/inpaint "Do full steps"
        if img2img_fix_steps is not None and steps is not None:
            display_total = steps if img2img_fix_steps else total_steps
        else:
            display_total = total_steps

        # Generate preview image from latent (based on preview_interval)
        preview_image = None
        send_metrics = None

        # Determine if we should generate preview for this step
        # Always generate for: initial (-1), first step (0), last step, or at interval
        is_last_step = (step == total_steps - 1)
        should_generate_preview = preview_enabled and (
            step == -1 or
            step == 0 or
            is_last_step or
            (step > 0 and step % preview_interval == 0)
        )

        if should_generate_preview:
            try:
                # Debug: Log model type being used for preview
                if step == -1 or step == 0:
                    print(f"[ProgressCallback] Using TAESD preview: is_sdxl={is_sdxl}, is_zimage={is_zimage}, is_deus={is_deus}, is_zimage_sdxl_vae={is_zimage_sdxl_vae}, is_flux2={is_flux2}, is_anima={is_anima}, is_lens={is_lens}, image_size={image_width}x{image_height}, preview_predicted_x0={preview_predicted_x0}, preview_interval={preview_interval}")

                # Choose which latent to decode based on preview_predicted_x0 option
                # If preview_predicted_x0 is True and pred_original_sample is available, use it
                # Otherwise fall back to current latents
                if preview_predicted_x0 and pred_original_sample is not None:
                    latent_to_decode = pred_original_sample
                else:
                    latent_to_decode = latents

                preview_pil = taesd_manager.decode_latent(
                    latent_to_decode,
                    is_sdxl=is_sdxl,
                    is_zimage=is_zimage,
                    is_deus=is_deus,
                    is_zimage_sdxl_vae=is_zimage_sdxl_vae,
                    is_flux2=is_flux2,
                    is_anima=is_anima,
                    is_lens=is_lens,
                    is_ideogram4=is_ideogram4,
                    is_minit2i=is_minit2i,
                    minit2i_vae_type=minit2i_vae_type,
                    is_krea2=is_krea2,
                    image_width=image_width,
                    image_height=image_height,
                    preview_decoder=preview_decoder
                )
                if preview_pil:
                    buffered = BytesIO()
                    # Use quality 75 for better compression (was 85)
                    preview_pil.save(buffered, format="JPEG", quality=75)
                    preview_image = base64.b64encode(buffered.getvalue()).decode()
            except Exception as e:
                print(f"Preview generation error: {e}")

        # Send CFG metrics (only when preview is generated to reduce transfer)
        if should_generate_preview:
            send_metrics = cfg_metrics

        # Handle step=-1 (initial noise) display
        if step == -1:
            # Initial noise: display as step 0
            display_step = 0
            status_text = f"Step 0/{display_total} (Initial Noise)"
        else:
            display_step = step + 1
            status_text = f"Step {display_step}/{display_total}"

        # Send synchronously from callback thread
        websocket_manager.send_progress_sync(
            display_step,
            display_total,
            status_text,
            preview_image=preview_image,
            cfg_metrics=send_metrics
        )

        # Additive: mirror the same step info into the in-memory polling
        # status store (GET /generation/status), for clients that don't
        # want to hold a WebSocket connection open. Does not affect the
        # WS broadcast above. Guarded like the preview block above: status
        # bookkeeping must never abort the sampling loop.
        try:
            from api.generation_status import update_progress
            update_progress(display_step, display_total, phase=status_text)
        except Exception as e:
            print(f"Generation status update error: {e}")

    return progress_callback


def create_db_image_record(
    db_image_class,
    filename: str,
    params: Dict[str, Any],
    actual_seed: int,
    generation_type: str,
    image_hash: str,
    lora_names: Optional[List[str]],
    model_name: str,
    model_hash: str,
    result_image: Optional[Image.Image] = None,
    source_image_hash: Optional[str] = None,
    mask_data_base64: Optional[str] = None
):
    """
    データベースレコードを作成

    重複削減: 63行 → 21行（42行削減）

    Args:
        db_image_class: GeneratedImageクラス
        filename: 保存されたファイル名
        params: 生成パラメータ
        actual_seed: 実際に使用されたシード
        generation_type: 生成タイプ（txt2img/img2img/inpaint）
        image_hash: 生成画像のハッシュ
        lora_names: LoRA名リスト
        model_name: モデル名
        model_hash: モデルハッシュ
        result_image: 生成された画像（img2img/inpaintの場合、width/height取得用）
        source_image_hash: ソース画像ハッシュ（img2img/inpaint）
        mask_data_base64: マスクデータbase64（inpaint）

    Returns:
        GeneratedImageインスタンス
    """
    # For img2img/inpaint, use result image dimensions
    if result_image:
        width = result_image.width
        height = result_image.height
    else:
        width = params.get("width", 512)
        height = params.get("height", 512)

    # Get sampler and ancestral seed
    sampler = params.get("sampler", "euler")
    ancestral_seed_value = params.get("ancestral_seed", -1)

    # Base record
    record = db_image_class(
        filename=filename,
        prompt=params.get("prompt", ""),
        negative_prompt=params.get("negative_prompt", ""),
        model_name=model_name,
        sampler=f"{sampler} ({params.get('schedule_type', 'uniform')})",
        steps=params.get("steps", 20),
        cfg_scale=params.get("cfg_scale", 7.0),
        seed=actual_seed,
        ancestral_seed=ancestral_seed_value,
        width=width,
        height=height,
        generation_type=generation_type,
        parameters=params,
        image_hash=image_hash,
        lora_names=lora_names if lora_names else None,
        model_hash=model_hash if model_hash else None,
    )

    # Add img2img/inpaint specific fields
    if source_image_hash:
        record.source_image_hash = source_image_hash
    if mask_data_base64:
        record.mask_data = mask_data_base64

    return record


def load_loras_for_generation(
    lora_manager,
    pipeline,
    lora_configs: List[Dict],
    pipeline_name: str = "txt2img"
) -> tuple:
    """
    生成用のLoRAをロード

    重複削減: 39行 → 13行（26行削減）

    Args:
        lora_manager: LoRAマネージャー
        pipeline: 対象パイプライン
        lora_configs: LoRA設定リスト
        pipeline_name: パイプライン名（ログ用）

    Returns:
        (updated_pipeline, has_step_range_loras)
    """
    has_step_range_loras = False

    if lora_configs and pipeline:
        print(f"Loading {len(lora_configs)} LoRA(s) for {pipeline_name}...")
        pipeline = lora_manager.load_loras(pipeline, lora_configs)

        # Check if any LoRA has non-default step range
        has_step_range_loras = any(
            lora.get("step_range", [0, 1000]) != [0, 1000]
            for lora in lora_configs
        )

    return pipeline, has_step_range_loras


def prepare_params_for_db(params: Dict[str, Any], calculate_image_hash) -> Dict[str, Any]:
    """
    データベース保存用にパラメータを準備（ControlNet画像・ref_imagesをハッシュに変換）

    重複削減: 30行 → 10行（20行削減）

    Args:
        params: 生成パラメータ
        calculate_image_hash: 画像ハッシュ計算関数

    Returns:
        データベース保存用パラメータ
    """
    params_for_db = params.copy()

    if "controlnet_images" in params_for_db:
        params_for_db["controlnet_images"] = [
            {
                k: (calculate_image_hash(v) if k == "image" else v)
                for k, v in cn.items()
            }
            for cn in params_for_db["controlnet_images"]
        ]

    # Style transfer: params["style_transfer"] carries the decoded PIL reference
    # image under the "image" key (consumed at generation time). A raw PIL Image is
    # NOT JSON-serializable and would make the generated_images.parameters DB commit
    # raise "Object of type Image is not JSON serializable" -> HTTP 500. Replace it
    # with a stable hash for the DB record (mirrors the controlnet_images handling).
    _st = params_for_db.get("style_transfer")
    if isinstance(_st, dict):
        params_for_db["style_transfer"] = {
            k: (calculate_image_hash(v) if k == "image" else v)
            for k, v in _st.items()
        }

    # Multi-reference (N-ref): params["style_transfers"] is the plural LIST of
    # style entries (one dict per reference), each also carrying a raw PIL Image
    # under "image". routes.py sets this whenever ANY style entry is present
    # (single- or multi-ref), so it must be hashed here too -- otherwise the DB
    # commit raises "Object of type Image is not JSON serializable" -> HTTP 500,
    # which would break the single-ref path as well.
    _sts = params_for_db.get("style_transfers")
    if isinstance(_sts, list):
        params_for_db["style_transfers"] = [
            {
                k: (calculate_image_hash(v) if k == "image" else v)
                for k, v in entry.items()
            } if isinstance(entry, dict) else entry
            for entry in _sts
        ]

    # FLUX.2 Image Edit: Convert ref_images to hashes
    if "ref_images" in params_for_db and params_for_db["ref_images"]:
        params_for_db["ref_images"] = [
            calculate_image_hash(img) for img in params_for_db["ref_images"]
        ]

    return params_for_db


# ====================
# Priority 2: 中程度の重複
# ====================

def create_lora_step_callback(
    lora_manager,
    pipeline,
    total_steps: int,
    original_callback: Optional[Callable] = None
) -> Optional[Callable]:
    """
    LoRAステップコールバックを作成

    重複削減: 24行 → 8行（16行削減）

    Args:
        lora_manager: LoRAマネージャー
        pipeline: 対象パイプライン
        total_steps: 総ステップ数
        original_callback: 元のコールバック

    Returns:
        ステップコールバック（不要な場合はNone）
    """
    return lora_manager.create_step_callback(
        pipeline,
        total_steps,
        original_callback=original_callback
    )


def extract_model_info(pipeline_manager) -> tuple:
    """
    現在のモデル情報を抽出

    重複削減: 21行 → 7行（14行削減）

    Args:
        pipeline_manager: パイプラインマネージャー

    Returns:
        (model_name, model_hash)
    """
    model_name = ""
    model_hash = ""

    if pipeline_manager.current_model_info:
        model_source = pipeline_manager.current_model_info.get("source", "")
        if model_source:
            model_name = os.path.basename(model_source)
        model_hash = pipeline_manager.current_model_info.get("model_hash", "")

    return model_name, model_hash


def extract_vision_encoder_info(pipeline_manager) -> tuple:
    """現在ロード済みの VE 情報を抽出。未ロードなら ("", "") を返す。"""
    from utils.hash_cache import get_cached_file_hash

    ve_path = getattr(pipeline_manager, '_vision_encoder_path', None)
    print(f"[VE Metadata] _vision_encoder_path = {ve_path!r}")
    if not ve_path:
        return "", ""

    ve_name = os.path.basename(ve_path)
    try:
        ve_hash = get_cached_file_hash(ve_path)
    except Exception as e:
        print(f"[VE Metadata] Hash calculation failed: {e}")
        ve_hash = ""
    print(f"[VE Metadata] ve_name={ve_name!r}, ve_hash={ve_hash[:16] if ve_hash else ''!r}")
    return ve_name, ve_hash


def extract_vae_info(pipeline_manager) -> tuple:
    """Extract the effective VAE identity used for decode. Returns (vae_name, vae_hash).

    The VAE always participates in the final decode, so this is recorded for every
    generation where it can be determined. ``vae_name`` is a source description (a
    resolved directory/repo id, ``"embedded (checkpoint)"``, or ``"none (pixel-space)"``);
    ``vae_hash`` is the cached hash of a concrete local weight file when one is
    identifiable, else "" (embedded VAEs are already covered by the model hash).
    """
    from utils.hash_cache import get_cached_file_hash

    info = getattr(pipeline_manager, "current_model_info", None) or {}
    model_type = info.get("type", "")

    # type -> per-arch components dict attribute holding "vae_source"/"vae_path".
    comp_attr = {
        "flux2": "flux2_components",
        "anima": "anima_components",
        "lens": "lens_components",
        "ideogram4": "ideogram4_components",
        "minit2i": "minit2i_components",
        "krea2": "krea2_components",
        "zimage": "zimage_components",
    }.get(model_type)

    vae_source = None
    vae_path = None
    if comp_attr:
        comps = getattr(pipeline_manager, comp_attr, None) or {}
        vae_source = comps.get("vae_source")
        vae_path = comps.get("vae_path")
        # Fallback for arch dicts without an explicit note (e.g. ideogram4).
        if vae_source is None and comps.get("vae") is not None:
            try:
                from core.models.common.vae_store import vae_identity
                vae_source, vae_path = vae_identity(comps.get("vae"))
            except Exception:
                pass
    else:
        # Standard SD1.5 / SDXL: the identity is stashed on the loaded pipeline.
        for attr in ("txt2img_pipeline", "img2img_pipeline", "inpaint_pipeline"):
            pipe = getattr(pipeline_manager, attr, None)
            if pipe is not None and getattr(pipe, "_sushi_vae_source", None):
                vae_source = pipe._sushi_vae_source
                break

    if not vae_source:
        return "", ""

    vae_hash = ""
    if vae_path:
        try:
            weight_file = _resolve_primary_vae_weight(vae_path)
            if weight_file:
                vae_hash = get_cached_file_hash(weight_file)
        except Exception as e:
            print(f"[VAE Metadata] Hash calculation failed: {e}")
            vae_hash = ""
    print(f"[VAE Metadata] vae_name={vae_source!r}, vae_hash={vae_hash[:16] if vae_hash else ''!r}")
    return vae_source, vae_hash


def _resolve_primary_vae_weight(vae_path: str) -> Optional[str]:
    """Return a concrete VAE weight file to hash from a dir or file path (or None).

    Multi-file diffusers VAE dirs are hashed by their primary weight file
    (``diffusion_pytorch_model.safetensors``), never by every shard.
    """
    if not vae_path or not os.path.exists(vae_path):
        return None
    if os.path.isfile(vae_path):
        return vae_path
    for name in ("diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin"):
        cand = os.path.join(vae_path, name)
        if os.path.isfile(cand):
            return cand
    # Fall back to the first safetensors shard in the directory.
    try:
        for name in sorted(os.listdir(vae_path)):
            if name.endswith(".safetensors"):
                return os.path.join(vae_path, name)
    except OSError:
        pass
    return None


def sanitize_params_for_logging(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    ログ出力用にパラメータをサニタイズ（大きなデータを隠す）

    重複削減: 24行 → 8行（16行削減）

    Args:
        params: 生成パラメータ

    Returns:
        サニタイズされたパラメータ
    """
    params_for_log = params.copy()

    if "controlnets" in params_for_log and params_for_log["controlnets"]:
        params_for_log["controlnets"] = [
            {k: ("<base64_data>" if k == "image_base64" else v) for k, v in cn.items()}
            for cn in params_for_log["controlnets"]
        ]

    return params_for_log


# ====================
# Priority 3: 低頻度の重複だが簡単に抽出可能
# ====================

def set_prompt_chunking_settings(
    pipeline_manager,
    prompt_chunking_mode: str = "a1111",
    max_prompt_chunks: int = 0
):
    """
    プロンプトチャンキング設定を適用

    重複削減: 12行 → 4行（8行削減）

    Args:
        pipeline_manager: パイプラインマネージャー
        prompt_chunking_mode: チャンキングモード
        max_prompt_chunks: 最大チャンク数
    """
    pipeline_manager.prompt_chunking_mode = prompt_chunking_mode
    pipeline_manager.max_prompt_chunks = max_prompt_chunks


def calculate_generation_metadata(
    image: Image.Image,
    lora_configs: List[Dict],
    extract_lora_names_func,
    calculate_image_hash_func,
    source_image: Optional[Image.Image] = None,
    mask_image: Optional[Image.Image] = None,
    encode_mask_func: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    生成メタデータを計算

    重複削減: 9行 → 3行（6行削減）

    Args:
        image: 生成画像
        lora_configs: LoRA設定
        extract_lora_names_func: LoRA名抽出関数
        calculate_image_hash_func: 画像ハッシュ計算関数
        source_image: ソース画像（img2img/inpaint）
        mask_image: マスク画像（inpaint）
        encode_mask_func: マスクエンコード関数

    Returns:
        メタデータ辞書
    """
    metadata = {
        "image_hash": calculate_image_hash_func(image),
        "lora_names": extract_lora_names_func(lora_configs),
    }

    if source_image:
        metadata["source_image_hash"] = calculate_image_hash_func(source_image)

    if mask_image and encode_mask_func:
        metadata["mask_data_base64"] = encode_mask_func(mask_image)

    return metadata


def apply_generation_timings(params: Dict[str, Any], total_seconds: float) -> None:
    """Merge generation timing (total wall time + recorded phases) into ``params``.

    Records the total wall time measured around the generation call plus whatever
    phase breakdown the pipeline layer populated in the process-wide
    ``generation_timer`` (text encode / denoise / VAE decode). Mutates ``params``
    in place so the values flow into PNG chunks (``save_image_with_metadata``) and
    DB parameters (``prepare_params_for_db`` copies ``params``).

    Timing is informational (not reproducibility-affecting); values are seconds
    rounded to 3 decimals. Phases are only present for architectures/paths that
    instrument them — total is always recorded.
    """
    from core.inference.generation_timing import generation_timer

    params["generation_time"] = round(float(total_seconds), 3)
    # Phase keys already come back as time_text_encode / time_denoise / time_vae_decode.
    for key, value in generation_timer.phases_dict().items():
        params[key] = value
