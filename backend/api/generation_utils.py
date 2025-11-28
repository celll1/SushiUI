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
) -> List[Dict]:
    """
    ControlNet設定を処理し、base64画像をデコード

    重複削減: 105行 → 35行（70行削減）

    Args:
        controlnet_configs: ControlNet設定のリスト
        generation_type: 生成タイプ（ログ用）

    Returns:
        処理済みのControlNet画像リスト
    """
    controlnet_images = []
    if not controlnet_configs:
        return controlnet_images

    print(f"Processing {len(controlnet_configs)} ControlNet(s)...")

    for idx, cn_config in enumerate(controlnet_configs):
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
                })
            except Exception as e:
                print(f"[ControlNet {idx}] Error decoding image: {e}")
        else:
            print(f"[ControlNet {idx}] WARNING: No image_base64 provided for {generation_type}. "
                  "ControlNet will be skipped.")

    print(f"[Routes] Total controlnet_images added to params: {len(controlnet_images)}")
    return controlnet_images


def create_progress_callback_factory(
    taesd_manager,
    websocket_manager,
    is_sdxl: bool,
    img2img_fix_steps: Optional[bool] = None,
    steps: Optional[int] = None
) -> Callable:
    """
    WebSocketプログレスコールバックを生成

    重複削減: 63行 → 21行（42行削減）

    Args:
        taesd_manager: TAESD preview生成マネージャー
        websocket_manager: WebSocketマネージャー
        is_sdxl: SDXLモデルかどうか
        img2img_fix_steps: img2img/inpaintの"Do full steps"オプション
        steps: ステップ数（display_total計算用）

    Returns:
        プログレスコールバック関数
    """
    def progress_callback(step, total_steps, latents, cfg_metrics=None):
        # Calculate display_total for img2img/inpaint "Do full steps"
        if img2img_fix_steps is not None and steps is not None:
            display_total = steps if img2img_fix_steps else total_steps
        else:
            display_total = total_steps

        # Generate preview image from latent (every 5 steps to reduce overhead)
        preview_image = None
        send_metrics = None

        if step % 5 == 0 or step == total_steps - 1:
            try:
                preview_pil = taesd_manager.decode_latent(latents, is_sdxl=is_sdxl)
                if preview_pil:
                    buffered = BytesIO()
                    preview_pil.save(buffered, format="JPEG", quality=85)
                    preview_image = base64.b64encode(buffered.getvalue()).decode()
            except Exception as e:
                print(f"Preview generation error: {e}")
            # Only send CFG metrics when preview is generated
            send_metrics = cfg_metrics

        # Send synchronously from callback thread
        websocket_manager.send_progress_sync(
            step + 1,
            display_total,
            f"Step {step + 1}/{display_total}",
            preview_image=preview_image,
            cfg_metrics=send_metrics
        )

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

    # Calculate ancestral seed
    sampler = params.get("sampler", "euler")
    ancestral_seed = params.get("ancestral_seed", -1)
    if ancestral_seed != -1 and sampler in ["euler_a", "dpm2_a"]:
        ancestral_seed_value = ancestral_seed
    else:
        ancestral_seed_value = None

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
    データベース保存用にパラメータを準備（ControlNet画像をハッシュに変換）

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
