# 開発ドキュメント

Stable Diffusion WebUI の開発ガイド

## 目次

1. [プロジェクト構成](#プロジェクト構成)
2. [セットアップ](#セットアップ)
3. [アーキテクチャ](#アーキテクチャ)
4. [新機能の追加方法](#新機能の追加方法)
5. [API仕様](#api仕様)
6. [主要モジュール](#主要モジュール)
7. [トラブルシューティング](#トラブルシューティング)

---

## プロジェクト構成

### 技術スタック

**フロントエンド**:
- Next.js 14 (App Router)
- React 18
- TypeScript
- TailwindCSS
- Axios

**バックエンド**:
- Python 3.11
- FastAPI
- diffusers (Hugging Face)
- PyTorch 2.1.0+
- SQLAlchemy
- SQLite

### ディレクトリ構造

```
webui_cl/
├── frontend/
│   ├── src/
│   │   ├── app/              # Next.js App Router
│   │   │   ├── page.tsx      # メインページ
│   │   │   └── gallery/      # ギャラリーページ
│   │   ├── components/       # React コンポーネント
│   │   │   ├── generation/  # 生成パネル
│   │   │   │   ├── Txt2ImgPanel.tsx
│   │   │   │   ├── Img2ImgPanel.tsx
│   │   │   │   └── InpaintPanel.tsx
│   │   │   ├── viewer/      # ギャラリー関連
│   │   │   │   ├── ImageGrid.tsx
│   │   │   │   └── GalleryFilter.tsx
│   │   │   └── common/      # 共通コンポーネント
│   │   │       ├── Button.tsx
│   │   │       ├── Input.tsx
│   │   │       ├── Select.tsx
│   │   │       └── Slider.tsx
│   │   └── utils/           # ユーティリティ
│   │       └── api.ts       # API client
│   ├── package.json
│   └── next.config.js
├── backend/
│   ├── api/
│   │   └── routes.py        # FastAPI routes
│   ├── core/                # コアロジック
│   │   ├── pipeline.py      # パイプライン管理
│   │   ├── custom_sampling.py  # カスタムサンプリング
│   │   ├── vram_optimization.py  # VRAM最適化
│   │   ├── model_loader.py  # モデルロード
│   │   ├── lora_manager.py  # LoRA管理
│   │   └── nag_processor.py # NAG実装
│   ├── database/            # データベース
│   │   ├── models.py        # SQLAlchemy models
│   │   └── db.py            # DB接続
│   ├── utils/               # ユーティリティ
│   │   └── image_utils.py   # 画像保存・メタデータ
│   ├── config/              # 設定
│   │   └── settings.py      # アプリケーション設定
│   └── main.py              # エントリポイント
├── outputs/                 # 生成画像
├── thumbnails/              # サムネイル
├── models/                  # Stable Diffusionモデル
├── lora/                    # LoRAモデル
├── controlnet/              # ControlNetモデル
├── webui.db                 # SQLiteデータベース
└── README.md
```

---

## セットアップ

### 必要要件

- Python 3.11+
- Node.js 18+
- CUDA 11.8+ (GPU使用時)
- 16GB+ RAM
- 8GB+ VRAM（推奨）

### インストール

1. **バックエンド**:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **フロントエンド**:
```bash
cd frontend
npm install
```

3. **起動**:

バックエンド:
```bash
cd backend
python main.py
```

フロントエンド:
```bash
cd frontend
npm run dev
```

4. ブラウザで http://localhost:3000 を開く

---

## アーキテクチャ

### 画像生成フロー

```
[フロントエンド] → [API Client] → [FastAPI] → [Pipeline Manager] → [Custom Sampling] → [diffusers]
                                        ↓
                                  [VRAM Optimization]
                                        ↓
                                  [Image Save] → [Database]
```

詳細フロー:

1. **ユーザー入力**: パラメータ設定 → 生成ボタンクリック
2. **API送信**: `api.ts` → FastAPI エンドポイント
3. **パイプライン実行**:
   - モデルロード（初回のみ）
   - Text Encoder実行（プロンプトエンコード）
   - U-Net実行（ノイズ除去）
   - VAE Decode（潜在変数 → 画像）
4. **VRAM最適化**: 各コンポーネントを順次GPU/CPU移動
5. **画像保存**: PNGメタデータ付きで保存
6. **DB保存**: SQLiteに生成情報を保存
7. **結果返却**: 画像パスをフロントエンドに返す

### VRAM最適化（Sequential Offloading）

メモリ効率を最大化するため、Text Encoder → U-Net → VAE を順次実行：

```
┌─────────────┐
│Text Encoder │ GPU  ← プロンプトエンコード
└─────────────┘
       ↓ CPU移動
┌─────────────┐
│   U-Net     │ GPU  ← ノイズ除去（36ステップ）
└─────────────┘
       ↓ CPU移動
┌─────────────┐
│    VAE      │ GPU  ← 潜在変数をRGB画像に変換
└─────────────┘
```

**メリット**:
- 1コンポーネントずつGPUに配置 → VRAM削減
- 8GB VRAMで1024x1024が安定動作

**実装**: `backend/core/vram_optimization.py`

---

## 新機能の追加方法

### パラメータ追加の完全ガイド

新しいパラメータ `example_param` を追加する場合：

#### 1. フロントエンド型定義

**`frontend/src/utils/api.ts`**:
```typescript
export interface GenerationParams {
  prompt: string;
  negative_prompt?: string;
  // ... 既存のパラメータ
  example_param?: number;  // ← 追加
}
```

**各パネル型定義**（`Img2ImgParams`, `InpaintParams`も同様）:
```typescript
interface Img2ImgParams {
  // ... 既存のパラメータ
  example_param?: number;  // ← 追加
}
```

#### 2. フロントエンドUI

**`frontend/src/components/generation/Txt2ImgPanel.tsx`**:

デフォルト値:
```typescript
const DEFAULT_PARAMS: GenerationParams = {
  prompt: "",
  // ... 既存のデフォルト値
  example_param: 1.0,  // ← 追加
};
```

UI要素:
```tsx
<Slider
  label="Example Parameter"
  min={0}
  max={2.0}
  step={0.1}
  value={params.example_param || 1.0}
  onChange={(e) => setParams({
    ...params,
    example_param: parseFloat(e.target.value)
  })}
/>
```

**`Img2ImgPanel.tsx` と `InpaintPanel.tsx` にも同様に追加**

#### 3. API送信（重要！）

**`frontend/src/utils/api.ts`**:

**txt2img** (自動送信):
```typescript
export const generateTxt2Img = async (params: GenerationParams) => {
  // paramsに含まれているので追加作業不要
  const response = await api.post("/generate/txt2img", paramsWithImages);
  return response.data;
};
```

**img2img** (手動追加が必要):
```typescript
export const generateImg2Img = async (params: Img2ImgParams, image: File | string) => {
  const formData = new FormData();

  // ... 既存のformData.append

  // ★★★ 必ず追加 ★★★
  formData.append("example_param", String(paramsWithImages.example_param || 1.0));

  const response = await api.post("/generate/img2img", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return response.data;
};
```

**inpaint** (手動追加が必要):
```typescript
export const generateInpaint = async (params: InpaintParams, image: File | string, mask: File | string) => {
  const formData = new FormData();

  // ... 既存のformData.append

  // ★★★ 必ず追加 ★★★
  formData.append("example_param", String(paramsWithImages.example_param || 1.0));

  const response = await api.post("/generate/inpaint", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return response.data;
};
```

**なぜtxt2imgだけ自動？**
- txt2img: JSON POSTなので全パラメータが自動送信
- img2img/inpaint: 画像ファイルを含むためFormData使用 → 個別に`append`が必要

#### 4. バックエンドパラメータ受け取り

**`backend/api/routes.py`**:
```python
class GenerationParams(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    # ... 既存のパラメータ
    example_param: Optional[float] = 1.0  # ← 追加
```

#### 5. バックエンド処理

**`backend/core/pipeline.py`** (例):
```python
def generate_txt2img(self, params, progress_callback=None, step_callback=None):
    # パラメータ取得
    example_param = params.get("example_param", 1.0)

    # パラメータログ出力
    print(f"[Pipeline] example_param: {example_param}")

    # パラメータを使用した処理
    # ...
```

#### 6. メタデータ保存（オプション）

画像メタデータに保存したい場合：

**`backend/utils/image_utils.py`**:
```python
def save_image_with_metadata(image, params, generation_type, model_info):
    metadata = PngImagePlugin.PngInfo()

    # ... 既存のメタデータ

    # example_paramを保存
    if "example_param" in params:
        metadata.add_text("example_param", str(params["example_param"]))

    image.save(filepath, pnginfo=metadata)
```

#### 7. ギャラリー表示（オプション）

**`backend/database/models.py`**:
```python
class GeneratedImage(Base):
    # ... カラム定義

    def to_dict(self):
        result = {
            # ... 既存のフィールド
        }

        # parametersから抽出
        if self.parameters:
            if "example_param" in self.parameters:
                result["example_param"] = str(self.parameters["example_param"])

        return result
```

**`frontend/src/utils/api.ts`**:
```typescript
export interface GeneratedImage {
  // ... 既存のフィールド
  example_param?: string;
}
```

**`frontend/src/components/viewer/ImageGrid.tsx`**:
```tsx
{selectedImage.example_param && (
  <div>
    <span className="text-gray-400">Example Param:</span> {selectedImage.example_param}
  </div>
)}
```

### チェックリスト

新しいパラメータを追加する際は必ずチェック：

- [ ] `frontend/src/utils/api.ts` の型定義に追加
- [ ] 各パネル（Txt2Img, Img2Img, Inpaint）の型定義に追加
- [ ] 各パネルのDEFAULT_PARAMSに追加
- [ ] 各パネルのUIコンポーネント追加
- [ ] **`generateImg2Img()` の FormData に追加**
- [ ] **`generateInpaint()` の FormData に追加**
- [ ] `backend/api/routes.py` のPydanticモデルに追加
- [ ] バックエンド処理ロジックに追加
- [ ] （オプション）メタデータ保存に追加
- [ ] （オプション）ギャラリー表示に追加

---

## API仕様

### エンドポイント一覧

#### 画像生成

**POST `/generate/txt2img`**

リクエスト:
```json
{
  "prompt": "1girl, anime style",
  "negative_prompt": "bad quality",
  "steps": 20,
  "cfg_scale": 7.0,
  "sampler": "euler",
  "schedule_type": "uniform",
  "seed": -1,
  "width": 1024,
  "height": 1024,
  "unet_quantization": null
}
```

レスポンス:
```json
{
  "image": "txt2img_20250127_120000_12345.png",
  "seed": 12345
}
```

**POST `/generate/img2img`**

リクエスト: FormData
- `image`: File
- `prompt`: string
- `denoising_strength`: float (0.0-1.0)
- （その他txt2imgと同じパラメータ）

**POST `/generate/inpaint`**

リクエスト: FormData
- `image`: File
- `mask`: File
- `prompt`: string
- `denoising_strength`: float
- `mask_blur`: int
- （その他img2imgと同じパラメータ）

#### ギャラリー

**GET `/images`**

クエリパラメータ:
- `skip`: int (オフセット)
- `limit`: int (取得数)
- `search`: string (プロンプト検索)
- `generation_types`: string (カンマ区切り)
- `date_from`, `date_to`: ISO日時
- `width_min`, `width_max`: int
- `height_min`, `height_max`: int

レスポンス:
```json
{
  "images": [...],
  "total": 100,
  "skip": 0,
  "limit": 50
}
```

**GET `/images/{image_id}`**

単一画像の詳細を取得

**DELETE `/images/{image_id}`**

画像を削除

#### モデル管理

**GET `/models`**

利用可能なモデル一覧を取得

**POST `/models/load`**

モデルをロード

FormData:
- `source_type`: "huggingface" | "local" | "safetensors"
- `source`: モデルID or パス

#### LoRA

**GET `/loras`**

利用可能なLoRA一覧

#### ControlNet

**GET `/controlnets`**

利用可能なControlNet一覧

#### サンプラー

**GET `/samplers`**

利用可能なサンプラー一覧

**GET `/schedule_types`**

利用可能なスケジュールタイプ一覧

---

## 主要モジュール

### 1. パイプライン管理 (`backend/core/pipeline.py`)

**クラス**: `PipelineManager`

**主要メソッド**:
- `load_model(source_type, source, pipeline_type)`: モデルをロードしてパイプラインを作成
- `generate_txt2img(params, progress_callback, step_callback)`: txt2img生成
- `generate_img2img(params, init_image, progress_callback, step_callback)`: img2img生成
- `generate_inpaint(params, init_image, mask_image, progress_callback, step_callback)`: inpaint生成

**特徴**:
- 3つのパイプライン（txt2img, img2img, inpaint）を管理
- LoRA、ControlNet、量子化対応
- Sequential offloadingによるVRAM最適化

### 2. VRAM最適化 (`backend/core/vram_optimization.py`)

**主要関数**:

```python
def move_text_encoders_to_gpu(pipeline):
    """Text EncoderをGPUに移動"""

def move_text_encoders_to_cpu(pipeline):
    """Text EncoderをCPUに移動してVRAM解放"""

def move_unet_to_gpu(pipeline, quantization: Optional[str] = None):
    """U-NetをGPUに移動（量子化オプション付き）"""

def move_unet_to_cpu(pipeline):
    """U-NetをCPUに移動してVRAM解放"""

def move_vae_to_gpu(pipeline):
    """VAEをGPUに移動"""

def move_vae_to_cpu(pipeline):
    """VAEをCPUに移動してVRAM解放"""

def log_device_status(stage: str, pipeline, show_details: bool = False):
    """デバイス状態をログ出力"""
```

**量子化サポート**:
- FP8 E4M3FN (推奨): ~50% VRAM削減
- FP8 E5M2: 代替FP8形式
- 量子化モデルはCPUにキャッシュして再利用

### 3. カスタムサンプリング (`backend/core/custom_sampling.py`)

**主要関数**:

```python
def custom_sampling_loop(
    pipeline,
    prompt_embeds,
    negative_prompt_embeds,
    pooled_prompt_embeds,
    negative_pooled_prompt_embeds,
    num_inference_steps,
    guidance_scale,
    width,
    height,
    generator,
    ancestral_generator=None,
    latents=None,
    prompt_embeds_callback=None,
    progress_callback=None,
    step_callback=None,
    # Advanced CFG
    cfg_schedule_type="constant",
    cfg_schedule_min=1.0,
    cfg_schedule_max=None,
    cfg_schedule_power=2.0,
    cfg_rescale_snr_alpha=0.0,
    dynamic_threshold_percentile=0.0,
    dynamic_threshold_mimic_scale=7.0,
    # NAG
    nag_enable=False,
    nag_scale=5.0,
    nag_tau=3.5,
    nag_alpha=0.25,
    nag_sigma_end=3.0,
    nag_negative_prompt_embeds=None,
    nag_negative_pooled_prompt_embeds=None,
    attention_type="normal",
    # ControlNet
    controlnet_images=None,
    controlnet_conditioning_scale=None,
    control_guidance_start=None,
    control_guidance_end=None,
    developer_mode=False,
) -> Image.Image:
    """txt2img用カスタムサンプリングループ"""
```

**対応機能**:
- プロンプト編集（ステップごとに変更可能）
- CFG動的スケジューリング（Linear, Quadratic, Cosine, SNR-based）
- Dynamic Thresholding
- NAG (Normalized Attention Guidance)
- ControlNet
- 進捗コールバック

**img2img/inpaint版**も同様のシグネチャ

### 4. モデルローダー (`backend/core/model_loader.py`)

**クラス**: `ModelLoader`

**主要メソッド**:
```python
@staticmethod
def load_model(
    source_type: ModelSource,
    source: str,
    device: str = "cuda",
    torch_dtype=torch.float16,
    **kwargs
) -> StableDiffusionPipeline:
    """モデルをロード"""
```

**対応形式**:
- Hugging Face Hub (`source_type="huggingface"`)
- ローカルdiffusersディレクトリ (`source_type="local"`)
- Safetensorsファイル (`source_type="safetensors"`)

### 5. LoRA管理 (`backend/core/lora_manager.py`)

**クラス**: `LoRAManager`

**主要メソッド**:
```python
def load_loras(self, pipeline, lora_configs: List[Dict]) -> StableDiffusionPipeline:
    """複数のLoRAをパイプラインに適用"""

def unload_loras(self, pipeline) -> StableDiffusionPipeline:
    """LoRAをアンロード"""
```

**LoRA設定**:
```python
lora_config = {
    "path": "/path/to/lora.safetensors",
    "weight": 0.8,
    "trigger_words": "special_style"
}
```

### 6. 画像ユーティリティ (`backend/utils/image_utils.py`)

**主要関数**:

```python
def save_image_with_metadata(
    image: Image.Image,
    params: Dict[str, Any],
    generation_type: str = "txt2img",
    model_info: Optional[Dict[str, Any]] = None
) -> str:
    """画像をPNGメタデータ付きで保存"""

def extract_metadata_from_image(image_path: str) -> Dict[str, Any]:
    """画像からメタデータを抽出"""

def calculate_file_hash(file_path: str, algorithm: str = "sha256") -> str:
    """ファイルのハッシュを計算"""
```

**保存されるメタデータ**:
- プロンプト、ネガティブプロンプト
- ステップ数、CFG、サンプラー、スケジュールタイプ
- シード、ancestral_seed
- サイズ（width, height）
- 生成タイプ（txt2img, img2img, inpaint）
- Advanced CFGパラメータ
- NAGパラメータ
- U-Net量子化設定
- モデル名、モデルハッシュ
- LoRA情報

---

## トラブルシューティング

### パラメータが反映されない

**症状**: UIで設定したパラメータがバックエンドに届かない

**確認手順**:

1. **ブラウザのDevTools確認**:
   - Networkタブを開く
   - 生成リクエストを確認
   - Payload/FormDataに該当パラメータが含まれているか確認

2. **FormData追加確認**（img2img/inpaintのみ）:
   - `frontend/src/utils/api.ts` の `generateImg2Img()` / `generateInpaint()`
   - `formData.append("parameter_name", ...)` があるか確認

3. **バックエンドログ確認**:
   ```python
   print(f"[Debug] parameter_name: {params.get('parameter_name')}")
   ```

4. **Pydanticモデル確認**:
   - `backend/api/routes.py` の `GenerationParams`
   - 該当フィールドが定義されているか確認

### VRAM不足エラー

**症状**: `CUDA out of memory` エラー

**対策**:

1. **Sequential offloading確認**:
   - ログに `[VRAM] Moving ...` が表示されているか確認
   - 表示されていない場合はコードのバグ

2. **FP8量子化を試す**:
   - UIで "U-Net Quantization" → "FP8 E4M3" を選択
   - ~50% VRAM削減

3. **画像サイズを小さくする**:
   - 1024x1024 → 768x768
   - VRAMは解像度の2乗に比例

4. **バッチサイズ確認**:
   - バッチ生成していないか確認（このUIは基本的に1枚ずつ）

### モデルロードエラー

**症状**: モデルロードに失敗する

**確認手順**:

1. **パスの確認**:
   ```bash
   # モデルディレクトリを確認
   ls models/
   ```

2. **ディスク容量**:
   ```bash
   df -h  # Linux/Mac
   ```

3. **PyTorchバージョン**:
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```
   - 2.1.0以上推奨（FP8サポート）

4. **ログ確認**:
   - バックエンドのターミナル出力を確認
   - エラーメッセージから原因を特定

### FP8量子化エラー

**症状**: FP8量子化で `autocast` や `dtype` エラー

**確認**:

1. **PyTorchバージョン**:
   - 2.1.0以上が必須
   - `torch.float8_e4m3fn` が存在するか確認

2. **GPU対応**:
   - Ada（RTX 40シリーズ）以降推奨
   - それ以前のGPUでは動作が遅い可能性

3. **autocast確認**:
   - `custom_sampling.py` でU-Net呼び出し時に `torch.autocast` が使われているか確認

### 生成速度が遅い

**原因と対策**:

1. **量子化の影響**:
   - INT8量子化は推論を遅くする → FP8に変更 or 無効化

2. **Sequential offloading**:
   - GPU/CPU移動のオーバーヘッド
   - VRAMに余裕があれば無効化を検討

3. **サンプラー選択**:
   - DPM++ 2M: 高速
   - Euler: 標準
   - DDIM: 低速

4. **ステップ数**:
   - 20-30ステップが標準
   - 50ステップ以上は時間がかかる

---

## コーディング規約

### TypeScript/React

- **コンポーネント**: 関数コンポーネント
- **Hooks**: useState, useEffect, useCallback, useMemo
- **型定義**: 明示的に（`any` 禁止）
- **ファイル名**: PascalCase（例: `Txt2ImgPanel.tsx`）
- **フォーマット**: Prettier

### Python

- **スタイル**: PEP 8
- **型ヒント**: 可能な限り使用
- **Docstring**: 主要関数に記述
- **ファイル名**: snake_case（例: `vram_optimization.py`）
- **フォーマット**: Black（推奨）

### Git コミットメッセージ

```
簡潔な要約（50文字以内）

詳細な説明（必要に応じて）:
- 変更内容1
- 変更内容2

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

---

## パフォーマンス最適化

### フロントエンド

1. **React.memo**:
   - 頻繁に再レンダリングされるコンポーネントをメモ化
   - 例: `GalleryFilter`, `ImageList`

2. **useCallback/useMemo**:
   - コールバック関数と計算コストの高い値をメモ化

3. **画像の遅延読み込み**:
   - サムネイルを使用
   - Intersection Observer API

### バックエンド

1. **Sequential Offloading**:
   - VRAM効率を最大化
   - Text Encoder → U-Net → VAE

2. **FP8量子化**:
   - ~50% VRAM削減
   - Ada/Hopper GPUで推奨

3. **量子化モデルキャッシュ**:
   - 同じ量子化設定なら再利用
   - CPU上にキャッシュ

4. **バッチ処理**（将来実装）:
   - 複数画像を一度に生成
   - スループット向上

---

## テスト

### フロントエンド（未実装）

```bash
cd frontend
npm test
```

### バックエンド（未実装）

```bash
cd backend
pytest
```

---

## デプロイ

### 開発環境

上記のセットアップを参照

### プロダクション環境（未実装）

- Docker対応予定
- Nginx + Gunicorn構成を検討中

---

## 貢献

プルリクエストを歓迎します。

1. Forkする
2. フィーチャーブランチを作成 (`git checkout -b feature/amazing-feature`)
3. コミット (`git commit -m 'Add amazing feature'`)
4. プッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

---

## ライセンス

（ライセンス情報を記載）

---

## 参考リンク

- [diffusers Documentation](https://huggingface.co/docs/diffusers)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Next.js Documentation](https://nextjs.org/docs)
- [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) - FP8実装の参考

---

## 変更履歴

- **2025-11-27**: 初版作成
  - FP8量子化実装完了
  - Sequential VRAM offloading実装
  - txt2img/img2img/inpaint対応
