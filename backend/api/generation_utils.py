"""
Generation endpoint shared utilities
生成エンドポイント共通ユーティリティ

このモジュールは、txt2img/img2img/inpaintエンドポイント間のコード重複を削減します。
"""
from typing import List, Dict, Any, Optional, Callable, Tuple
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
                    # CFG-decoupled style guidance (SDXL/SD1.5 prototype only):
                    # None/<=0 = disabled (classic 2-pass, unchanged cost/behavior).
                    "style_guidance_scale": cn_config.get("style_guidance_scale"),
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

    # Multi-reference default: AdaIN (the Q/K distribution pull toward the
    # reference) is the DOMINANT content-destroying knob when 2+ references are
    # combined (stack OR common_concept). At the single-ref default (0.6) the
    # target collapses into an abstract blob when 2+ refs are supplied
    # (GPU-validated: a clean 2x2 sweep showed adain=0.2 blobs at every strength
    # while adain=0.0 stays legible). So when the caller supplies 2+ style
    # entries and did NOT explicitly set an AdaIN strength on an entry, default
    # that entry's AdaIN to 0.0 (content-preserving) instead of the single-ref
    # 0.6. An explicit per-entry style_adain_strength always wins, and single-ref
    # (len<=1) is untouched (keeps the 0.6 default that single-ref was tuned for).
    if len(style_transfers) > 1:
        for entry in style_transfers:
            if entry.get("adain_strength") is None:
                entry["adain_strength"] = 0.0

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
    def progress_callback(step, total_steps, latents, cfg_metrics=None, pred_original_sample=None, phase_label: Optional[str] = None):
        # Decoupled decode-phase progress (e.g. PiD decode): no denoise latent to
        # preview, just forward (step, total_steps, phase_label) as-is.
        if phase_label is not None:
            websocket_manager.send_progress_sync(
                step,
                total_steps,
                phase_label,
                preview_image=None,
                cfg_metrics=None
            )
            try:
                from api.generation_status import update_progress
                update_progress(step, total_steps, phase=phase_label)
            except Exception as e:
                print(f"Generation status update error: {e}")
            return

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

    # Explicit capability marker (avoids a fragile try/except TypeError probe at
    # call sites, e.g. the PiD decode-progress adapter in custom_sampling.py):
    # this closure supports the decoupled `phase_label` kwarg.
    progress_callback._supports_phase_label = True

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

    # params["controlnets"] is the RAW frontend ControlNet config list (set
    # verbatim at routes.py, e.g. `params["controlnets"] = controlnet_configs`)
    # -- unlike controlnet_images/style_transfer(s) above, these entries were
    # never decoded/hashed and still carry the full base64-encoded reference
    # image under "image_base64" (up to several MB per entry). Left in place,
    # this bloats both the DB `parameters` column and every serialization of
    # it (gallery list/detail JSON, PNG metadata). Strip it here so it never
    # reaches storage; model_path/strength/mode stay intact and the decoded
    # copy already lives in the hashed controlnet_images/style_transfers
    # above, so nothing reproducibility-relevant is lost. style_transfers is
    # covered defensively too (its "image" key is already hashed above, but
    # if a caller ever sets it to a raw config carrying image_base64 instead,
    # this keeps it capped as well).
    for _list_key in ("controlnets", "style_transfers"):
        _lst = params_for_db.get(_list_key)
        if isinstance(_lst, list):
            params_for_db[_list_key] = [
                {k: v for k, v in entry.items() if k != "image_base64"}
                if isinstance(entry, dict) else entry
                for entry in _lst
            ]
    if isinstance(params_for_db.get("style_transfer"), dict):
        params_for_db["style_transfer"] = {
            k: v for k, v in params_for_db["style_transfer"].items() if k != "image_base64"
        }

    return params_for_db


def record_attention_backend(params: Dict[str, Any],
                             generation_id: Optional[int] = None) -> Optional[str]:
    """Record the attention backend(s) that ACTUALLY ran, and warn on a downgrade.

    ``params["attention_type"]`` is the REQUESTED string; it is not evidence of
    what executed, because the conduit downgrades per call (capability guard,
    or a kernel that failed and fell back to native). This writes the observed
    backend(s) under ``params["attention_backend"]`` -- one value, or several
    joined with "+" when a generation genuinely ran more than one (e.g. a mixed
    head_dim model where one attention shape is refused by the requested
    backend) -- so the gallery row and the PNG metadata can never name a backend
    that did not run.

    Nothing is written when nothing was observed. That is the honest state for
    an architecture that does not route attention through the conduit (LTX-2.3,
    the diffusers-dispatch paths), and it is preferable to echoing the request.

    Also files ONE ``attention_downgrade`` warning when the requested backend is
    absent from what ran -- the conduit's own per-call warnings already cover
    the guard and kernel-failure cases, but this one is stated per generation in
    terms of the request, and covers any future path that degrades silently.

    Returns the recorded label, or None. Never raises.
    """
    try:
        from core.attention import normalize_backend, observed_backends

        used = observed_backends(generation_id)
        if not used:
            return None
        label = "+".join(used)
        params["attention_backend"] = label
        requested = params.get("attention_type")
        if requested is not None and normalize_backend(requested) not in used:
            try:
                from api.generation_status import add_warning

                add_warning(
                    f"attention_type={requested!r} was requested but the attention "
                    f"conduit ran {label} for this generation; the row records the "
                    f"backend that actually ran.",
                    code="attention_downgrade",
                )
            except Exception:
                pass
        return label
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[Attention] Could not record the resolved attention backend: {exc}")
        return None


def record_model_variant(params: Dict[str, Any], pipeline_manager) -> Optional[str]:
    """Record which MiniMax-H3 partition (fl2va / ref2va) is actually loaded.

    Same precedent as ``record_attention_backend``: the filename is the only
    thing that distinguishes the two checkpoints, so a row that only stores
    ``model_name`` becomes unreadable the moment either file is renamed. The
    loader already resolves and carries ``variant`` in
    ``pipeline_manager.current_model_info`` (see ``minimax_h3/loader.py``), so
    this reads that resolved value rather than inferring anything from the
    request.

    Writes ``params["model_variant"]`` only when a MiniMax-H3 model is loaded
    and its variant is known; a no-op (and no key written) for every other
    architecture. Never raises.
    """
    try:
        if not getattr(pipeline_manager, "is_minimax_h3_model", False):
            return None
        variant = (pipeline_manager.current_model_info or {}).get("variant")
        if not variant:
            return None
        params["model_variant"] = variant
        return variant
    except Exception:
        return None


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


def describe_vae_override(pipeline_manager) -> tuple:
    """Describe the ACTIVE per-generation VAE override. Returns (name, path).

    Returns ``(None, None)`` when no override is applied. ``name`` is prefixed
    with ``"override: "`` so it can never be confused with the model's own VAE,
    and — for a SushiUI VAE fine-tune export — carries the provenance that the
    export DIRECTORY alone cannot express: the run name, the training step, and
    whether these are the EMA or the live weights. Both are needed because a
    single run re-exports to the SAME two paths as it progresses, so the path is
    not a stable identifier of the weights, and the EMA/live pair is written to
    two sibling directories at the identical step.

    PRIVACY: ``name`` flows into ``params["vae_name"]``, which the PNG writer
    records in the file's text chunks — i.e. it leaves this machine. It
    therefore carries only the override's DISPLAY NAME
    (``_friendly_component_name``: the basename, or ``<model folder>/vae`` for a
    generic diffusers component directory), never the directory it lives in,
    the drive, or anything else about this filesystem. That is the same rule
    the accepted ``model_name`` precedent follows (basename + content hash).
    The absolute path stays in ``params["vae_override_path"]`` for the local
    gallery row. Do NOT assume the PNG writer cannot see that key: it is only
    PARTLY an allowlist. Its per-key ``add_text`` calls are one, but it also
    writes a ``sushi_parameters`` chunk containing the WHOLE ``params`` dict —
    that is how ``vae_override_path`` reached shared PNGs for 56 rows before
    this was fixed. What keeps it out now is the redaction applied there
    (``utils/path_redaction.redact_params_for_sharing``), not the shape of the
    writer. See the header of ``utils/image_utils.py``.

    A VAE that this repo did not train has no ``sushi_vae_training.json``
    sidecar (a downloaded diffusers directory, a bare ``.safetensors``, a PiD
    ``.pth``). That is a normal case, not a degraded one: such a label is the
    display name alone, which identifies the file exactly as ``model_name``
    identifies a checkpoint. The name is never empty and never says "unknown".

    Every provenance field is optional and tri-state at the source
    (``read_vae_training_sidecar``): an unknown value is reported as unknown,
    never silently defaulted.

    The label is built from the override PATH, never from the loaded module's
    ``config._name_or_path``: diffusers copies that field verbatim out of
    ``config.json``, so a fine-tune export inherits its BASE VAE's value (the
    run-113 exports both carry ``"../sdxl-vae/"``) and a bare ``.safetensors``
    override carries nothing at all. Reporting it would name the stock VAE for an
    image the fine-tune decoded — the exact confusion this function exists to
    remove. ``override_vae_identity()`` is consulted only for a PiD decoder, where
    it deliberately returns a description rather than a path.
    """
    path = getattr(pipeline_manager, "_override_vae_path", None)
    if not path:
        return None, None

    parts = []
    if _override_vae_is_pid(pipeline_manager):
        # PiD wraps the model's OWN VAE (its encoder is still the model's); only
        # the decode is the PiD net at ``path``.
        parts.append("PiD pixel-diffusion super-resolution decoder")

    try:
        from api.generation_overrides import read_vae_training_sidecar
        prov = read_vae_training_sidecar(path)
    except Exception:
        prov = None
    if prov:
        parts.append(f"run {prov['run_name']}" if prov.get("run_name")
                     else "SushiUI VAE fine-tune")
        if prov.get("step") is not None:
            parts.append(f"step {prov['step']}")
        ema = prov.get("ema_applied")
        parts.append("EMA weights" if ema is True
                     else "live weights" if ema is False
                     else "EMA state unknown")
        enc = prov.get("encoder_trained")
        parts.append("encoder+decoder trained" if enc is True
                     else "decoder only" if enc is False
                     else "trained scope unknown")
    elif _has_vae_training_sidecar(path):
        # The export carries a sidecar that could not be parsed as an object.
        # Distinguishing this from a VAE that never had one keeps the label
        # honest in both directions: it does not silently demote a SushiUI
        # export to "no provenance", and it does not claim provenance for a
        # third-party VAE that legitimately has none.
        parts.append("SushiUI VAE fine-tune")
        parts.append("provenance file unreadable")

    try:
        from api.generation_overrides import _friendly_component_name
        display = _friendly_component_name(path)
    except Exception:
        # Same shape as the helper's own fallback: a name, never a path.
        display = os.path.basename(str(path).rstrip("/\\")) or "unnamed file"

    name = f"override: {display}"
    if parts:
        name = f"{name} ({', '.join(parts)})"
    return name, path


def _has_vae_training_sidecar(path) -> bool:
    """True when ``path`` is a directory holding a SushiUI VAE-training sidecar
    file, regardless of whether that file could be parsed."""
    try:
        from api.generation_overrides import VAE_TRAINING_SIDECAR
        return bool(path) and os.path.isfile(os.path.join(str(path), VAE_TRAINING_SIDECAR))
    except Exception:
        return False


def _override_vae_is_pid(pipeline_manager) -> bool:
    """True when the active override VAE slot holds a ``PidVaeWrapper``."""
    try:
        from core.models.pid.pid_vae_wrapper import PidVaeWrapper
        for kind, container, key in pipeline_manager._vae_override_targets():
            active = getattr(container, key) if kind == "attr" else container.get(key)
            return isinstance(active, PidVaeWrapper)
    except Exception:
        pass
    return False


def extract_vae_info(pipeline_manager) -> tuple:
    """Extract the effective VAE identity used for decode. Returns (vae_name, vae_hash).

    The VAE always participates in the final decode, so this is recorded for every
    generation where it can be determined. ``vae_name`` is a source description (a
    resolved directory/repo id, ``"embedded (checkpoint)"``, ``"none (pixel-space)"``,
    or an ``"override: ..."`` description when a per-generation VAE override is
    active — see ``describe_vae_override``); ``vae_hash`` is the cached hash of a
    concrete local weight file when one is identifiable, else "" (embedded VAEs are
    already covered by the model hash).

    An active override is reported FIRST: it replaces the VAE object in the model's
    slots without touching the ``vae_source`` notes recorded at model load, so those
    notes still describe the checkpoint's own VAE and would misreport the decode.
    """
    from utils.hash_cache import get_cached_file_hash

    override_name, override_path = describe_vae_override(pipeline_manager)
    if override_name:
        # Hash of the override's own weight file. For a PiD override this is the
        # PiD .pth, so such a row records no hash for the model's own VAE (whose
        # encoder still ran) — the model hash remains its only anchor.
        vae_hash = ""
        try:
            weight_file = _resolve_primary_vae_weight(override_path)
            if weight_file:
                vae_hash = get_cached_file_hash(weight_file)
        except Exception as e:
            print(f"[VAE Metadata] Override hash calculation failed: {e}")
        print(f"[VAE Metadata] vae_name={override_name!r}, vae_hash={vae_hash[:16] if vae_hash else ''!r}")
        return override_name, vae_hash

    info = getattr(pipeline_manager, "current_model_info", None) or {}
    model_type = info.get("type", "")

    # The per-arch components dict holding "vae_source"/"vae_path" is DERIVED,
    # never listed: ``PipelineManager`` declares one ``<arch>_components``
    # attribute per component-based architecture in its ``__init__`` (so
    # ``hasattr`` is True from construction, whether or not a model is loaded)
    # and none for SD1.5/SDXL, whose identity is stashed on the loaded pipeline
    # instead. A written-out map is the exact defect that broke
    # ``extract_fp8_gemm_info`` for FLUX.2, and it had already broken this one:
    # the list here was missing ``ltx2`` and ``acestep``, so every LTX-2.3 video
    # fell into the SD1.5/SDXL branch below, found no ``_sushi_vae_source`` and
    # recorded NO vae_name/vae_hash at all -- although ``ltx2_components["vae"]``
    # exists and the ``vae_identity`` fallback below would have named it.
    # ``quantized_capability_parity_test.HandWrittenArchMapTest`` pins the
    # derivation functionally, for every component-based arch.
    comp_attr = f"{model_type}_components" if model_type else ""
    if comp_attr and not hasattr(pipeline_manager, comp_attr):
        comp_attr = ""

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


def extract_fp8_gemm_info(pipeline_manager) -> str:
    """Describe the quantized GEMM path that served this generation, or "" if N/A.

    Only weight-only quantized checkpoints have anything to report, and only on
    the architectures whose loaders swap in the quantized Linear classes
    (``QUANTIZED_LINEAR_ARCHS``) -- either class on every one of them, since each
    of those loaders detects int8 and fp8 independently and a converted artifact
    is mixed by design (Ideogram 4 was ``Fp8Linear`` only until its loader gained
    the int8 half). A bf16 checkpoint on any of them, or any other architecture,
    returns "" and records nothing.

    The arch -> component-dict attribute map is DERIVED from
    ``QUANTIZED_LINEAR_ARCHS`` rather than written out. It was written out, and
    it drifted: FLUX.2 joined the tuple (and this docstring) while the map kept
    three entries, so every FLUX.2 generation recorded no ``fp8_gemm`` and every
    ``quantized_gemm_mode="w8a8"`` request there reported "the loaded checkpoint
    carries no weight-only quantized Linear layers" -- on a checkpoint full of
    them. LTX-2.3 would have been the second. ``PipelineManager`` names every one
    of these dicts ``<arch>_components``, which is what makes the derivation
    exact rather than a guess.

    WHAT THE TESTS ACTUALLY PIN, stated precisely because an earlier version of
    this paragraph claimed more than was true (it said a future arch breaking
    the naming convention "fails a test", while the only test that existed
    grepped ``pipeline.py`` for the string ``<arch>_components`` -- which pins
    that the ATTRIBUTE exists, not that this function derives its name, so
    reverting the derivation to the shipped-broken three-entry map left the
    whole suite green). ``quantized_capability_parity_test`` now pins BOTH:
    ``test_every_quantized_arch_has_a_component_dict_the_gemm_reporter_can_find``
    for the attribute, and
    ``test_the_gemm_reporter_finds_every_quantized_arch``, which calls THIS
    function for every arch in ``QUANTIZED_LINEAR_ARCHS`` over a fake manager
    holding one real ``Int8Linear`` and requires a non-empty label. A hand-written
    map that omits an arch fails the second one.

    The RESOLVED path is what is recorded, not the flag: a W8A8 path can be
    enabled while the per-device probe finds nothing usable, in which case every
    layer runs the dequantized matmul. The paths are numerically different (a
    W8A8 path additionally quantizes the activation), so an image is not
    reproducible without knowing which one ran.

    VOCABULARY. The ``fp8_gemm`` metadata field is declared an opaque mechanism
    label, and this function's job is to keep it one field rather than growing a
    second. Its vocabulary now includes the INT8 stems
    (``w8a8_int_mm(...)`` / ``int8_dequant...``) alongside the FP8 ones
    (``w8a8_scaled_mm(...)`` / ``dequant...``). A MIXED checkpoint -- which is
    what the int8 conversion tool produces, since high-crest layers fall back to
    e4m3 in the same file -- owns both module types and is reported as both
    labels joined with "+", ALWAYS in FP8-then-INT8 order because that is the
    order the describers are collected in below, e.g.
    ``"dequant+w8a8_int_mm(int_mm+fused)"`` for a mixed checkpoint with the FP8
    toggle off and the INT8 one on. The stems are distinct precisely so that join
    stays unambiguous. Packed MiniMax-H3 checkpoints add
    ``w4a8_int8(comfy-kitchen)``; that fixed operator is independent of the
    FP8/INT8 W8A8 toggle.
    """
    from core.models.common.int8_runtime_quantize import QUANTIZED_LINEAR_ARCHS

    info = getattr(pipeline_manager, "current_model_info", None) or {}
    arch = info.get("type", "")
    if arch not in QUANTIZED_LINEAR_ARCHS:
        return ""
    comp_attr = f"{arch}_components"
    comps = getattr(pipeline_manager, comp_attr, None) or {}
    describers = []
    for module_path, attr in (
        ("core.models.ideogram4.vendor.fp8_linear", "describe_gemm_path"),
        ("core.models.ideogram4.vendor.int8_linear", "describe_gemm_path"),
        ("core.models.common.w4a8_linear", "describe_gemm_path"),
    ):
        try:
            describers.append(getattr(__import__(module_path, fromlist=[attr]), attr))
        except Exception:
            pass
    if not describers:
        return ""
    # The transformer is the bulk of the Linear work; Ideogram 4's unconditional
    # branch and both text encoders are swapped by the same loader, so the
    # conditional transformer is a faithful witness for the checkpoint's format.
    #
    # The component KEY is not "transformer" on every arch -- ACE-Step's DiT
    # lives under "dit" -- so the names are DERIVED from the same table that
    # declares what an export writes (``EXPORT_LAYOUTS[arch]["modules"]``, via
    # ``layout_module_specs``), which is by construction the set of components
    # holding this arch's quantized Linears. A hand-written tuple here is the
    # fourth occurrence of the arch-map defect this function's docstring is
    # about; the legacy names stay as a fallback for an arch with quantized
    # Linears but no export layout.
    names = []
    try:
        from core.models.common.quantized_export import layout_module_specs

        names = [component for component, _prefix in layout_module_specs(arch)]
    except Exception:
        names = []
    for name in list(dict.fromkeys(
            names + ["transformer", "unconditional_transformer", "text_encoder"])):
        module = comps.get(name)
        if module is None:
            continue
        labels = []
        for describe in describers:
            try:
                label = describe(module)
            except Exception as e:
                print(f"[Quant Metadata] Could not resolve the GEMM path: {e}")
                return ""
            if label:
                labels.append(label)
        if labels:
            return "+".join(labels)
    return ""


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


# ---------------------------------------------------------------------------
# Per-architecture video request resolution (design sec.8 / sec.9)
# ---------------------------------------------------------------------------

def resolve_video_defaults(params: Dict[str, Any], provided_keys, arch: Optional[str],
                           base: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Fill every OMITTED video field from the loaded arch's video defaults.

    ``VIDEO_GEN_DEFAULTS`` is LTX-2.3-shaped, so a second video architecture
    needs different geometry. Rather than special-casing a route, the Pydantic /
    ``Form()`` declared defaults stay the base values (they are what the schema
    documents) and this helper replaces them for the fields the client did NOT
    send, using ``param_defaults.video_defaults_for_arch``.

    Args:
        params: the request dict, MUTATED in place.
        provided_keys: the keys the client actually sent — Pydantic's
            ``model_fields_set`` on a JSON body, or the set of ``Form(None)``
            sentinels that came back non-None on a multipart one.
        arch: the loaded architecture (``pipeline_manager.current_model_info``'s
            ``type``). Unknown/None resolves to the base defaults, i.e. today's
            behaviour.
        base: the base default map, default ``VIDEO_GEN_DEFAULTS``.

    Returns:
        The RESOLVED default map — which the caller passes on to
        ``check_arch_capabilities(..., defaults=...)`` so "the user set this to
        a non-default value" is judged against the same numbers the request was
        filled from.
    """
    from api.param_defaults import video_defaults_for_arch

    resolved = video_defaults_for_arch(arch, base)
    provided = set(provided_keys or ())
    for key, value in resolved.items():
        if key in params and key not in provided:
            params[key] = value
    return resolved


def validate_video_steps(params: Dict[str, Any], arch: Optional[str],
                         *, steps_key: str = "num_inference_steps") -> None:
    """Refuse a step count the arch's SCHEDULER cannot build a schedule from.

    Spec-driven, exactly like ``validate_video_geometry``: the bound comes from
    the arch's ``TemporalSpec.min_inference_steps`` rather than from a literal
    at the route, because it is arch-specific. LTX-2.3's
    ``FlowMatchEulerDiscreteScheduler`` runs N evaluations for N steps and
    accepts N=1; MiniMax-H3's scheduler counts sigma GRID POINTS with the
    terminal 0 included, so N drives N-1 evaluations and N=1 drives none — its
    ``set_timesteps`` raises for N < 2.

    Without this, an under-minimum request paid for a full text encode (tens of
    seconds) and then died as a 500 from inside the sampler. Called next to
    ``validate_video_geometry``, i.e. before any weight is touched, so the
    answer is a fast 400 instead.

    Non-video / unrecognised archs, and an arch whose minimum is 1 (nothing to
    enforce beyond the positivity every sampler needs), are left alone.
    """
    from api.error_handlers import ValidationError
    from core.models.components.wiring import temporal_spec_for_arch

    spec = temporal_spec_for_arch(arch)
    if spec is None:
        return

    steps = int(params.get(steps_key, spec.min_inference_steps))
    if steps >= spec.min_inference_steps:
        return

    if spec.steps_are_sigma_grid_points:
        detail = (
            f"Got {steps_key}={steps}. On this model {steps_key} counts sigma schedule grid "
            f"points with the terminal 0 included, so it drives {steps_key} - 1 model "
            f"evaluations; {spec.min_inference_steps} is the smallest count that runs any."
        )
    else:
        detail = (f"Got {steps_key}={steps}. This model's scheduler needs at least "
                  f"{spec.min_inference_steps}.")
    raise ValidationError(
        f"{steps_key} must be at least {spec.min_inference_steps} for this model",
        detail=detail,
    )


def validate_video_geometry(params: Dict[str, Any], arch: Optional[str],
                            *, frame_key: Optional[str] = "num_frames") -> List[str]:
    """Validate (and, where the arch says so, SNAP) a video request's geometry.

    Spec-driven: every rule comes from the arch's ``TemporalSpec``
    (``core.models.components.wiring``), so adding a video architecture does not
    add a branch here.

    * spatial axes must be multiples of ``pixel_align`` and must fit
      ``max_pixel_hw`` (orientation-agnostic) — both are hard 400s, because a
      canvas is not something to silently change under a caller;
    * ``frame_key`` must be a valid clip length in the production range —
      unless it is None, which skips the clip-length rule entirely and leaves
      only the spatial and frame-rate rules. That is not a convenience: on
      ``/generate/outpaint/video`` the length the client sends is the OUTPUT
      timeline's, while the length that has to be on the grid is the GENERATED
      span's (the preserved frames are pasted, never sampled), and that span is
      resolved by ``plan_video_outpaint_placement`` instead. An
      invalid one is a hard 400 on an arch whose spec sets
      ``snap_invalid_length=False`` (LTX-2.3 — this is its documented, shipped
      behaviour) and is SNAPPED to the nearest valid length on an arch that sets
      it True (MiniMax-H3), rounding UP to the next encodable length, with a
      warning. The warning fires whether the offending value was sent by the
      client or resolved from a default — deliberately, because a client that
      asked for 130 frames and got 141 must be told either way — which is why
      this function needs no knowledge of which keys the client actually set;
    * ``frame_rate`` is forced to ``fps_fixed`` where the arch has one, with a
      warning.

    Mutates ``params`` for the snapped/forced values. Returns the warning
    messages, already emitted through ``add_warning`` (so they reach the
    response's ``warnings[]``), for the caller to log or assert on.

    An arch with no ``TemporalSpec`` (not a video arch, or an unrecognised one)
    is left completely alone: the caller's own checks still apply.
    """
    from api.error_handlers import ValidationError
    from core.models.components.wiring import temporal_spec_for_arch

    spec = temporal_spec_for_arch(arch)
    warnings: List[str] = []
    if spec is None:
        return warnings

    try:
        from api.generation_status import add_warning
    except ImportError:  # pragma: no cover - status module always present in-process
        add_warning = None

    def warn(message: str) -> None:
        warnings.append(message)
        if add_warning is not None:
            add_warning(message, code="video_constraint")

    width = int(params.get("width", 0))
    height = int(params.get("height", 0))
    if width % spec.pixel_align or height % spec.pixel_align:
        raise ValidationError(
            f"width and height must both be divisible by {spec.pixel_align}",
            detail=f"Got width={width}, height={height}. Round each to the nearest multiple of "
                   f"{spec.pixel_align}.",
        )
    if spec.max_pixel_hw is not None:
        short_cap, long_cap = min(spec.max_pixel_hw), max(spec.max_pixel_hw)
        short_edge, long_edge = min(width, height), max(width, height)
        if short_edge > short_cap or long_edge > long_cap:
            raise ValidationError(
                f"the canvas exceeds this model's {short_cap}x{long_cap} envelope",
                detail=f"Got {width}x{height}. The released checkpoint generates with a short edge "
                       f"of at most {short_cap} and a long edge of at most {long_cap}, in either "
                       f"orientation.",
            )

    # The smoke gate lowers the PRODUCTION floor to the VAE's decodable floor,
    # so a short clip can be generated deliberately (a smoke test, a preview)
    # without that length being reachable by an ordinary API caller. It never
    # lowers `min_decodable_frames`, which the decoder cannot go below at all.
    smoke = bool(os.environ.get(spec.smoke_override_env))
    floor = spec.floor(smoke)
    if frame_key is None:
        num_frames = None
    else:
        num_frames = int(params.get(frame_key, 0))
    in_range = num_frames is not None and floor <= num_frames and (
        spec.max_frames is None or num_frames <= spec.max_frames)
    if num_frames is None:
        pass
    elif not (spec.is_valid_length(num_frames) and in_range):
        if not spec.snap_invalid_length:
            raise ValidationError(
                f"{frame_key} must satisfy ({frame_key} - {spec.frame_offset}) % "
                f"{spec.frame_multiple} == 0",
                detail=f"Got {frame_key}={num_frames}. Valid values are "
                       f"{spec.suggested_lengths(6)}, ...",
            )
        snapped = spec.snap_length(num_frames, smoke)
        params[frame_key] = snapped
        warn(f"{frame_key}={num_frames} is not a length this model can generate; using {snapped}. "
             f"Valid lengths are {spec.frame_multiple} * n + {spec.frame_offset} "
             f"(n >= 1), between {floor} and {spec.max_frames}.")
    elif smoke and num_frames < spec.min_frames:
        warn(f"{frame_key}={num_frames} is below this model's trained range "
             f"({spec.min_frames}-{spec.max_frames}) and was accepted only because "
             f"{spec.smoke_override_env} is set.")

    if spec.fps_fixed is not None:
        frame_rate = float(params.get("frame_rate", spec.fps_fixed))
        if abs(frame_rate - spec.fps_fixed) > 1e-6:
            params["frame_rate"] = spec.fps_fixed
            warn(f"frame_rate={frame_rate} is not supported by this model, which generates at a "
                 f"fixed {spec.fps_fixed} fps; using {spec.fps_fixed}.")

    return warnings


def plan_video_outpaint_placement(
    params: Dict[str, Any],
    arch: Optional[str],
    *,
    head_frames: int,
    tail_frames: Optional[int] = None,
) -> Dict[str, Any]:
    """Resolve a temporal-outpaint request against the arch's placement rule.

    PURE (no warnings, no mutation) so the route can call it for a fast 400
    before the GPU slot and the backend can recompute the same numbers without
    trusting a caller-supplied plan. Both call THIS function; the arithmetic
    exists once.

    ``head_frames`` is the length of the (already trimmed) uploaded clip and
    ``tail_frames`` the length of the optional second (bridge) clip, or None
    when there is none.

    Returns, for an arch whose ``TemporalSpec.outpaint_placements`` is
    ``("free",)`` (LTX-2.3) or which has no spec at all::

        {"placement": "free"}

    and for a boundary-conditioned arch (MiniMax-H3)::

        {"placement": "extend_forward" | "extend_backward" | "bridge",
         "head_frames": int, "tail_frames": int,       # preserved, exactly
         "generated_frames": int,                      # a VALID clip length
         "total_frames": int,                          # effective output length
         "requested_total_frames": int,
         "shared_anchor_frames": int}                  # 1 for an extend, 2 for a bridge

    The generated span -- not the total -- is what has to be a length the model
    can generate, because the preserved frames are pasted rather than sampled.
    Its first (and, for a bridge, last) frame is the anchor, i.e. the SAME
    instant as the preserved frame it was taken from, so it is not emitted
    twice: ``total = head + tail + generated - shared_anchor_frames``.

    Raises ``ValidationError`` for a placement the architecture cannot anchor,
    naming the reason and the offsets that would work.
    """
    from api.error_handlers import ValidationError
    from core.models.components.wiring import temporal_spec_for_arch

    spec = temporal_spec_for_arch(arch)
    placements = tuple(spec.outpaint_placements) if spec is not None else ("free",)

    if "free" in placements:
        if tail_frames is not None:
            raise ValidationError(
                "this model has no bridge placement",
                detail="bridge_video adds a SECOND preserved clip at the end of the timeline, "
                       "which is a placement only an architecture that conditions on boundary "
                       "frames needs. The loaded model places one clip at an arbitrary offset "
                       "instead; upload a single `video` and set input_offset_frames.",
            )
        return {"placement": "free"}

    if head_frames < 1:
        raise ValidationError(
            "video outpaint needs at least one input frame",
            detail=f"The (trimmed) input clip has {head_frames} frames.",
        )

    requested_total = int(params.get("total_frames") or 0)
    offset = int(params.get("input_offset_frames") or 0)
    # WHAT THIS REASON SAYS, AND WHAT IT USED TO SAY. It used to claim the
    # architecture has no index-addressable conditioning. That is false: an
    # anchor's rotary position is `num_text_tokens + (5/3)*f` for any pixel
    # frame f, `build_packed_layout` takes an integer index, and
    # /generate/img2vid places anchors with it. What THIS endpoint has no
    # measured behaviour for is the mid-timeline OUTPAINT shape -- a preserved
    # clip's two boundary frames anchored at their own indices inside one
    # generated span, with exact preservation around them. The refusal is
    # unchanged; only its reason is now the true one.
    boundary_reason = (
        "This endpoint conditions on the boundary frames of the span it generates, so the "
        "preserved clip has to abut one. Mid-timeline placement would anchor its two boundary "
        "frames at their own indices inside one generated span; MiniMax-H3 can address an "
        "arbitrary frame (that is what /generate/img2vid's keyframe placement uses), but that "
        "outpaint shape is unmeasured and is not offered here."
    )

    if tail_frames is not None:
        if "bridge" not in placements:
            raise ValidationError(
                "this model has no bridge placement",
                detail=boundary_reason,
            )
        if tail_frames < 1:
            raise ValidationError(
                "the bridge clip has no frames",
                detail=f"The (trimmed) bridge clip has {tail_frames} frames.",
            )
        if offset != 0:
            raise ValidationError(
                "a bridge places the first clip at the start of the timeline",
                detail=f"Got input_offset_frames={offset}. In a bridge the uploaded `video` is "
                       f"preserved at the head and `bridge_video` at the tail, so the only valid "
                       f"offset for the head clip is 0.",
            )
        placement = "bridge"
        shared = 2
        tail = int(tail_frames)
    else:
        tail = 0
        shared = 1
        if offset == 0:
            placement = "extend_forward"
        elif offset + head_frames == requested_total:
            placement = "extend_backward"
        else:
            raise ValidationError(
                "this model cannot place the clip at that offset",
                detail=f"{boundary_reason} Got input_offset_frames={offset} with a "
                       f"{head_frames}-frame clip in a {requested_total}-frame timeline. Use 0 "
                       f"(extend forward, the clip's last frame anchors the generated span) or "
                       f"{max(0, requested_total - head_frames)} (extend backward, its first "
                       f"frame anchors it), or upload a second clip as bridge_video to generate "
                       f"the span between two clips.",
            )
        if placement not in placements:
            raise ValidationError(
                f"this model does not support the {placement.replace('_', '-')} placement",
                detail=boundary_reason,
            )

    preserved = head_frames + tail
    smoke = bool(os.environ.get(spec.smoke_override_env))
    requested_generated = max(1, requested_total - preserved + shared)
    generated = spec.snap_length(requested_generated, smoke)
    return {
        "placement": placement,
        "head_frames": int(head_frames),
        "tail_frames": tail,
        "generated_frames": int(generated),
        "total_frames": int(preserved + generated - shared),
        "requested_total_frames": requested_total,
        "shared_anchor_frames": shared,
    }


def resolve_minimax_h3_outpaint_reference_gate(
    variant: Optional[str],
    *,
    has_reference_images: bool,
    placement: str,
    generated_frames: Optional[int] = None,
) -> None:
    """The ``ref2va`` reference gate on ``/generate/outpaint/video`` -- the
    decision table of ``minimax_h3_outpaint_refs_design.md`` §3, minus its
    "not MiniMax-H3" row (the caller checks that itself; it needs no
    variant/placement context). PURE (raises or returns None), shared by the
    route (fast 400, called once the clip is decoded and the placement is
    planned) and the backend (defensive re-check for a caller that bypasses
    the route) -- the arithmetic exists once.

    * ``fl2va`` serves every placement; refuses ``reference_images`` outright
      (that partition was never trained to read reference rows).
    * ``ref2va`` serves ONLY ``extend_forward`` (the source clip is always
      auto-referenced there, so this is checked even with
      ``has_reference_images=False``) and refuses a generated span shorter
      than the reference-video floor. ``extend_backward``/``bridge`` are
      refused outright: unmeasured, and structurally suspect (a ref2va
      layout places every reference before the target span on the rotary
      clock, which is a continuation order a backward extend contradicts).
    * Any other/unidentified variant refuses ``reference_images`` (a
      mismatch cannot be detected from the weights, so the safe default is
      to refuse rather than silently run reference conditioning through
      untrained weights).

    No row of this table ever reroutes to ``/generate/ref2vid``.
    """
    from api.error_handlers import ValidationError
    from core.models.minimax_h3.h3_references import MIN_REFERENCE_VIDEO_FRAMES

    variant = (variant or "").lower()
    if variant == "fl2va":
        if has_reference_images:
            raise ValidationError(
                "reference_images requires the MiniMax-H3 ref2va transformer, not fl2va",
                detail="fl2va was never trained to read reference rows (mirror of "
                       "/generate/ref2vid's own partition gate). Load "
                       "diffusion_models/minimax_h3_ref2va_pruned_fp8_scaled.safetensors, or omit "
                       "reference_images to keep using fl2va's exact-preserving extend.",
            )
        return
    if variant == "ref2va":
        if placement != "extend_forward":
            raise ValidationError(
                "MiniMax-H3 ref2va only serves the extend_forward placement",
                detail=f"Got placement={placement!r}. A ref2va layout places every reference "
                       f"before the target span on the rotary clock (continuation order); "
                       f"extend_backward would ask the generated span to precede the source's own "
                       f"content, and bridge composes two boundary anchors with references, "
                       f"neither of which is measured. Load the fl2va checkpoint for "
                       f"extend_backward/bridge, or use extend_forward here.",
            )
        if generated_frames is not None and generated_frames < MIN_REFERENCE_VIDEO_FRAMES:
            raise ValidationError(
                "The generated span is too short to carry the source clip as a reference",
                detail=f"ref2va auto-references the preserved clip's trailing frames, which needs "
                       f"a generated span of at least {MIN_REFERENCE_VIDEO_FRAMES} frames; this "
                       f"request's generated span is {generated_frames}. Request a longer extend, "
                       f"or load the fl2va checkpoint.",
            )
        return
    if has_reference_images:
        raise ValidationError(
            f"reference_images needs the MiniMax-H3 ref2va transformer, not "
            f"{variant or 'an unidentified variant'}",
            detail="The two released MiniMax-H3 files are otherwise indistinguishable, so a "
                   "mismatch cannot be detected from the weights and running this would silently "
                   "produce a bad video rather than fail.",
        )


def latent_frame_spans(spec, num_latent_frames: int) -> List[Tuple[int, int]]:
    """``[(pixel_start, pixel_end_exclusive), ...]`` per latent frame.

    Cycles ``spec.latent_chunk_pattern``; empty pattern means the arch has not
    declared its chunking and there is nothing to compute.
    """
    pattern = tuple(spec.latent_chunk_pattern)
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for index in range(int(num_latent_frames)):
        width = pattern[index % len(pattern)]
        spans.append((cursor, cursor + width))
        cursor += width
    return spans


def plan_video_inpaint_span(
    params: Dict[str, Any],
    arch: Optional[str],
    *,
    clip_frames: int,
) -> Dict[str, Any]:
    """Resolve a temporal-inpaint request against the arch's latent chunking.

    PURE (no warnings, no mutation), for the same reason
    :func:`plan_video_outpaint_placement` is: the route calls it for a fast 400
    before the GPU slot and the backend calls it again for the numbers it
    generates from, so the arithmetic exists once.

    ``clip_frames`` is the length of the TRIMMED uploaded clip, which is also
    the output length -- every frame of it has a row in the packed sequence, so
    it is the clip and not a generated span that must be a valid length. It is
    never snapped: snapping a clip length here means deleting frames the caller
    said to keep, so an invalid length is a 400 naming the trims that reach the
    nearest valid ones.

    Returns::

        {"clip_frames": int, "latent_frames": int,
         "requested_start": int, "requested_end": int,      # as asked, pixels
         "start_frame": int, "end_frame": int,              # after the snap
         "snapped": bool,
         "regenerate_latent_frames": (int, ...),
         "pinned_latent_frames": (int, ...)}                # everything else

    The requested range is expanded OUTWARD to latent-frame boundaries (a latent
    frame is pinned or generated whole), never shrunk: at a boundary the
    caller's "regenerate this" wins over "keep that", which is what an image
    inpaint mask's dilation already means.

    Raises ``ValidationError`` for an architecture with no declared chunking, an
    invalid clip length, an empty/out-of-range range, or a range that leaves
    nothing preserved.
    """
    from api.error_handlers import ValidationError
    from core.models.components.wiring import temporal_spec_for_arch

    spec = temporal_spec_for_arch(arch)
    if spec is None or not spec.latent_chunk_pattern:
        raise ValidationError(
            "this model has no temporal inpaint",
            detail="Regenerating a time range in place needs the architecture's video-VAE "
                   "temporal chunking, which decides the smallest range that can be addressed. "
                   f"'{arch or 'the loaded model'}' does not declare one, so this endpoint cannot "
                   "serve it.",
        )

    clip_frames = int(clip_frames)
    smoke = bool(os.environ.get(spec.smoke_override_env))
    floor = spec.floor(smoke)
    in_range = floor <= clip_frames and (spec.max_frames is None or clip_frames <= spec.max_frames)
    if not (spec.is_valid_length(clip_frames) and in_range):
        why = (f"The trimmed clip is {clip_frames} frame(s). Temporal inpaint samples the WHOLE "
               f"clip -- every frame has a row in the packed sequence -- so the clip itself must "
               f"be a valid length: {spec.frame_multiple} * n + {spec.frame_offset}, between "
               f"{floor} and {spec.max_frames}. The length is not snapped here, because snapping "
               f"it would silently delete frames you asked to keep. ")
        if clip_frames < floor:
            raise ValidationError(
                "the trimmed clip is shorter than this model's shortest clip",
                detail=why + f"A shorter clip cannot be trimmed into range; it needs at least "
                             f"{floor} frames.",
            )
        # The valid length at or below the clip: reachable by trimming.
        below = spec.snap_length(clip_frames, smoke)
        if below > clip_frames or (spec.max_frames is not None and below > spec.max_frames):
            below -= spec.frame_multiple
        below = min(below, spec.max_frames) if spec.max_frames is not None else below
        raise ValidationError(
            "the trimmed clip is not a length this model can generate",
            detail=why + f"Trim {clip_frames - below} more frame(s) (input_trim_start_frames + "
                         f"input_trim_end_frames) to reach {below}.",
        )

    requested_start = int(params.get("regenerate_start_frame") or 0)
    requested_end = int(params.get("regenerate_end_frame") or 0)
    if not (0 <= requested_start < requested_end <= clip_frames):
        raise ValidationError(
            "the regenerate range is not inside the clip",
            detail=f"Got regenerate_start_frame={requested_start}, "
                   f"regenerate_end_frame={requested_end} for a {clip_frames}-frame trimmed clip. "
                   f"The range is [start, end) -- start inclusive, end exclusive -- so it needs "
                   f"0 <= start < end <= {clip_frames}.",
        )

    num_latent_frames = int(spec.latent_frames(clip_frames))
    spans = latent_frame_spans(spec, num_latent_frames)
    regenerate = tuple(index for index, (lo, hi) in enumerate(spans)
                       if lo < requested_end and hi > requested_start)
    start_frame = spans[regenerate[0]][0]
    end_frame = spans[regenerate[-1]][1]
    pinned = tuple(index for index in range(num_latent_frames) if index not in set(regenerate))
    if not pinned:
        raise ValidationError(
            "nothing is preserved",
            detail=f"Frames {start_frame}..{end_frame} cover the whole {clip_frames}-frame clip "
                   f"after the range was expanded to latent-frame boundaries, so there is no "
                   f"preserved content to condition on or to paste back. Generating a whole clip "
                   f"from a prompt is /generate/txt2vid; conditioning it on stills is "
                   f"/generate/img2vid.",
        )

    return {
        "clip_frames": clip_frames,
        "latent_frames": num_latent_frames,
        "requested_start": requested_start,
        "requested_end": requested_end,
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "snapped": bool(start_frame != requested_start or end_frame != requested_end),
        "regenerate_latent_frames": regenerate,
        "pinned_latent_frames": pinned,
    }


# ---------------------------------------------------------------------------
# Keyframe placement (MiniMax-H3 `fl2va`, POST /generate/img2vid)
# ---------------------------------------------------------------------------

# The model card's scope, quoted once and reused by every message that has to
# state it. MiniMax documents `fl2va` for zero, one or two input images at the
# first and last frame.
MINIMAX_H3_DOCUMENTED_ANCHOR_SCOPE = (
    "MiniMax's model card documents this workflow for first- and last-frame "
    "conditioning with up to two images"
)


def plan_keyframe_placements(requests, num_frames: int):
    """Resolve one img2vid request's keyframe placements onto the clip.

    ``requests`` is a sequence of ``(source, requested_index)`` in SUBMISSION
    order, where ``source`` is a human-readable name used in error messages
    (``"image"``, ``"keyframe_images[0]"``, ``"last_frame_image"``).

    Returns ``{"anchors": [...], "undocumented": [...]}`` where each anchor is::

        {"source": str, "requested": int, "frame": int,
         "anchor": "first" | "last" | int}

    sorted ASCENDING by resolved frame, which is the packed order.

    Three decisions live here rather than at the route:

    * **``-1`` is resolved against the SNAPPED clip length.** ``num_frames`` is
      what route validation left in ``params`` after snapping to the arch's
      grid, so the caller cannot compute the last index itself and the sentinel
      is the only reliable way to name it. (``0`` needs no sentinel and is not
      given one.)
    * **The two ENDS resolve to the string anchors.** Frame 0 becomes
      ``"first"`` and the last frame becomes ``"last"``, so a request that only
      uses the ends produces the byte-identical layout it produced before
      placement existed -- see ``h3_pipeline_ops._anchor_rotary_time`` for why
      the string branches are not the integer formula.
    * **Ascending order, not upload order.** The layout gives an anchor the same
      rows wherever it sits, so ordering is free; sorting makes the packed order
      a function of the placements alone, which is what keeps a legacy
      ``image`` + ``last_frame_image`` request identical no matter which part
      the client sent first.

    ``undocumented`` lists the shapes of this request that are outside the model
    card (an anchor at an intermediate frame, more than two anchors). Empty for
    every request expressible before this phase. The caller emits the warning;
    this function stays pure so it is testable without a generation context.

    Raises ``ValidationError`` for an out-of-range index or two anchors that
    resolve to the same frame, with the frame arithmetic in the detail.
    """
    from api.error_handlers import ValidationError

    num_frames = int(num_frames)
    if num_frames < 1:
        raise ValidationError(
            "the clip has no frames to place a keyframe on",
            detail=f"Got num_frames={num_frames}.",
        )
    last_index = num_frames - 1

    anchors = []
    seen = {}
    for source, requested in requests:
        requested = int(requested)
        frame = last_index if requested == -1 else requested
        if frame < 0 or frame > last_index:
            raise ValidationError(
                f"keyframe placement {requested} is outside the clip",
                detail=f"`{source}` asked for frame {requested}. This clip is {num_frames} "
                       f"frames long after the server snapped num_frames to the model's own "
                       f"grid, so its addressable frames are 0..{last_index}. Use -1 for the "
                       f"last frame: the snap happens after the request is sent, so -1 is the "
                       f"only index that always means the end of the clip.",
            )
        if frame in seen:
            raise ValidationError(
                f"two keyframes were placed on frame {frame}",
                detail=f"`{seen[frame]}` and `{source}` both resolve to frame {frame} of a "
                       f"{num_frames}-frame clip (-1 resolves to {last_index}). One frame holds "
                       f"one anchor; give them different indices or send only one.",
            )
        seen[frame] = source
        if frame == 0:
            anchor = "first"
        elif frame == last_index:
            anchor = "last"
        else:
            anchor = frame
        anchors.append({"source": source, "requested": requested,
                        "frame": frame, "anchor": anchor})

    anchors.sort(key=lambda entry: entry["frame"])

    undocumented = []
    intermediate = [entry["frame"] for entry in anchors if isinstance(entry["anchor"], int)]
    if intermediate:
        undocumented.append(
            "an anchor at frame " + ", ".join(str(f) for f in intermediate)
            + f", which is neither the first nor the last frame of the "
              f"{num_frames}-frame clip")
    if len(anchors) > 2:
        undocumented.append(f"{len(anchors)} anchors in one request")

    return {"anchors": anchors, "undocumented": undocumented}
