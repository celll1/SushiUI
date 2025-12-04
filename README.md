# 🍣 SushiUI

Stable Diffusion 1.5/XL対応の画像生成Webアプリケーション

## 特徴

- **txt2img**: テキストから画像を生成
- **img2img**: 画像から画像を生成（デノイジング強度調整可能）
- **Inpainting**: マスク領域の再生成
- **Loop Generation**: 同一パラメータで連続生成、ステップ範囲指定可能
- **画像ビューワー**: 生成画像の閲覧とメタデータ検索
- **Advanced CFG**: CFG Scheduling、SNR-Based Adaptive CFG、Dynamic Thresholding
- **高度な機能**: プロンプト編集、マルチLoRA（ステップ範囲指定）、マルチControlNet

## 技術スタック

### バックエンド
- Python 3.10+
- FastAPI
- PyTorch
- Diffusers
- SQLAlchemy
- WebSocket (進捗表示)

### フロントエンド
- Next.js 14
- TypeScript
- Tailwind CSS
- Axios
- WebSocket Client

## セットアップ

### 必要要件
- Python 3.10+
- Node.js 18+
- CUDA対応GPU（推奨、8GB VRAM以上）

### 自動セットアップ（推奨）

1. `start.bat` を実行
```bash
start.bat
```

このスクリプトは以下を自動的に実行します：
- Python仮想環境の作成
- バックエンド依存関係のインストール
- フロントエンド依存関係のインストール
- 必要なディレクトリの作成
- バックエンド・フロントエンドサーバーの起動

**重要**: PyTorchのGPU版は手動でインストールする必要があります：

```bash
# 仮想環境をアクティベート
venv\Scripts\activate

# PyTorch GPU版をインストール（CUDA 12.1の例）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# または CUDA 11.8の場合
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

お使いのCUDAバージョンに応じて、[PyTorch公式サイト](https://pytorch.org/get-started/locally/)から適切なコマンドを確認してください。

### 手動セットアップ

<details>
<summary>クリックして手動セットアップ手順を表示</summary>

#### バックエンドのセットアップ

1. プロジェクトディレクトリに移動
```bash
cd webui_cl
```

2. Python仮想環境を作成
```bash
python -m venv venv
```

3. 仮想環境を有効化
- Windows:
```bash
venv\Scripts\activate
```
- Linux/Mac:
```bash
source venv/bin/activate
```

4. PyTorch GPU版をインストール
```bash
# CUDA 12.1の例
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

5. その他の依存関係をインストール
```bash
cd backend
pip install -r requirements.txt
```

6. 必要なディレクトリを作成
```bash
mkdir ../models ../lora ../controlnet ../vae ../outputs ../thumbnails
```

7. Stable Diffusionモデルをダウンロード
```bash
# models/ ディレクトリにsafetensorsまたはdiffusersフォーマットのモデルを配置
# LoRAは lora/ ディレクトリに配置
# ControlNetは controlnet/ ディレクトリに配置
```

8. バックエンドサーバーを起動
```bash
python main.py
```

サーバーは `http://localhost:8000` で起動します。

#### フロントエンドのセットアップ

1. フロントエンドディレクトリに移動
```bash
cd ../frontend
```

2. 依存関係をインストール
```bash
npm install
```

3. 開発サーバーを起動
```bash
npm run dev
```

フロントエンドは `http://localhost:3000` で起動します。

</details>

## プロジェクト構造

```
webui_cl/
├── backend/
│   ├── api/
│   │   └── routes.py            # APIエンドポイント（txt2img, img2img, inpaint）
│   ├── core/
│   │   ├── pipeline.py          # Diffusersパイプライン管理
│   │   ├── custom_sampling.py   # カスタムサンプリングループ
│   │   ├── lora_manager.py      # LoRA管理（動的ロード）
│   │   ├── controlnet_manager.py      # ControlNet管理
│   │   ├── controlnet_preprocessor.py # ControlNetプリプロセッサー
│   │   └── cfg_utils.py         # Advanced CFG ユーティリティ
│   ├── database/
│   │   ├── models.py            # SQLAlchemyモデル（GeneratedImage, UserSettings）
│   │   └── db.py                # データベース接続・マイグレーション
│   ├── utils/                   # ユーティリティ
│   ├── config/                  # 設定
│   ├── main.py                  # エントリーポイント
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx                # Txt2Img
│   │   │   ├── img2img/page.tsx        # Img2Img
│   │   │   ├── inpaint/page.tsx        # Inpaint
│   │   │   ├── loop-generation/page.tsx # Loop Generation
│   │   │   ├── gallery/page.tsx        # Gallery
│   │   │   └── settings/page.tsx       # Settings
│   │   ├── components/
│   │   │   ├── generation/             # 生成パネルコンポーネント
│   │   │   ├── common/                 # 共通コンポーネント
│   │   │   └── settings/               # 設定コンポーネント
│   │   ├── utils/
│   │   │   ├── api.ts                  # API クライアント
│   │   │   └── sendHelpers.ts          # Send機能ヘルパー
│   │   └── lib/                        # ユーティリティ
│   └── package.json
├── models/               # Stable Diffusionモデル
├── lora/                 # LoRAモデル
├── controlnet/           # ControlNetモデル
├── vae/                  # VAEモデル
├── outputs/              # 生成画像
├── thumbnails/           # サムネイル
├── start.bat             # 自動起動スクリプト
└── README.md
```

## 使用方法

### 基本的な画像生成

1. ブラウザで `http://localhost:3000` を開く
2. "Generate" ページでプロンプトを入力
3. パラメータを調整（Steps, CFG Scale, Sampler等）
4. "Generate" ボタンをクリック
5. WebSocket経由でリアルタイム進捗とプレビューを確認
6. "Gallery" ページで生成した画像を閲覧

### 高度な機能

#### Advanced CFG Features

##### 1. CFG Scheduling (Sigma-based)
CFGスケールを生成プロセス全体で動的に変化させる機能:
- **Constant**: 固定CFG（デフォルト）
- **Linear**: 線形補間（min → max）
- **Quadratic**: 2次補間（より滑らかな変化、power パラメータで調整）
- **Cosine**: コサイン補間（序盤と終盤で緩やかに変化）

パラメータ:
- `cfg_schedule_type`: スケジュールタイプ
- `cfg_schedule_min`: 最小CFG値（デフォルト: 1.0）
- `cfg_schedule_max`: 最大CFG値（デフォルト: main CFG scale）
- `cfg_schedule_power`: Quadraticモード時の累乗値（デフォルト: 2.0）

##### 2. SNR-Based Adaptive CFG
Signal-to-Noise Ratio に基づいてCFGを自動調整:
- `cfg_rescale_snr_alpha`: 0.0 = 無効、0.1-0.5 が一般的
- SNRが高い（クリアな画像）ほどCFGを下げてアーティファクトを抑制
- SNRが低い（ノイズが多い）ほどCFGを上げてプロンプト従属性を強化

##### 3. Dynamic Thresholding
生成された潜在変数の値を動的にクランプしてアーティファクトを抑制:
- `dynamic_threshold_percentile`: 0.0 = 無効、99.5 が一般的
- `dynamic_threshold_mimic_scale`: クランプ値（1～30、推奨 5～7）
- 高CFG使用時のアーティファクト、色飽和を軽減

#### プロンプト編集
プロンプトに `[prompt1:prompt2:0.5]` 形式で記述することで、生成途中でプロンプトを切り替えられます。

- 例: `[cat:dog:0.3]` → 最初の30%はcat、その後dog

#### プロンプト強調構文
- `(word)` - 1.1倍強調
- `((word))` - 1.21倍強調
- `(word:1.5)` - 1.5倍強調
- `[word]` - 0.9倍弱体化

#### LoRA
- LoRAタブで最大5つまで同時に追加可能
- 重み調整（-2.0 〜 2.0）
- ステップ範囲指定（0-1000）で特定のステップ範囲でのみLoRAを適用
- 動的読み込み: 生成時に必要なLoRAのみロード

#### ControlNet
- マルチControlNet対応（複数同時適用可能）
- ControlNetタブで画像をアップロード
- Conditioning Scale調整（0.0 〜 2.0）
- ガイダンス範囲指定（開始・終了ステップ: 0-1000）
- プリプロセッサー自動検出（モデル名から推定）
- LLLiteサポート

#### Loop Generation
- 同一パラメータで連続生成
- ステップ範囲指定（Start Step / End Step）
- Advanced CFG、LoRA、ControlNet をすべてサポート
- 各画像の生成時間を表示

## サポート機能

### モデル対応
- ✅ Stable Diffusion 1.5
- ✅ Stable Diffusion XL
- ✅ v-prediction models（自動検出、guidance rescale適用）
- ✅ Safetensors / Diffusersフォーマット
- ✅ LoRA（マルチLoRA、ステップ範囲指定）
- ✅ ControlNet (SD1.5/SDXL)

### サンプラー
- Euler
- Euler a
- DPM++ 2M
- DPM++ SDE
- DPM2
- DPM2 a
- Heun
- DDIM
- DDPM
- PNDM
- LMS
- UniPC

### スケジュール
- Uniform
- Karras
- Exponential

### TAESD（高速プレビュー）
- リアルタイム生成プレビュー（5ステップごと）
- SD1.5/SDXL自動切り替え

## 実装済み機能

### コア機能
- [x] txt2img基本機能
- [x] img2img機能
- [x] Inpainting機能
- [x] Loop Generation（連続生成、ステップ範囲指定）
- [x] カスタムサンプリングループ
- [x] プロンプト編集（ステップベース切り替え）
- [x] プロンプト強調構文
- [x] WebSocketリアルタイム進捗表示（Server-Sent Events）
- [x] TAESDプレビュー
- [x] 自動データベースマイグレーション

### Advanced CFG Features
- [x] CFG Scheduling (Sigma-based)
  - [x] Linear
  - [x] Quadratic (power パラメータ対応)
  - [x] Cosine
- [x] SNR-Based Adaptive CFG
- [x] Dynamic Thresholding (percentile + mimic scale)
- [x] Developer Mode（CFGメトリクス可視化）

### モデル・拡張機能
- [x] LoRA（マルチLoRA、ステップ範囲指定）
- [x] ControlNet（マルチControlNet対応）
  - [x] ステップ範囲指定（start_step, end_step）
  - [x] プリプロセッサー自動検出
  - [x] LLLiteサポート
  - [x] Canny, Depth, OpenPose, LineArt 等のプリプロセッサー

### UI機能
- [x] 画像ビューワー・ギャラリー
- [x] メタデータ検索・フィルタリング
- [x] 生成パラメータ表示（Advanced CFG含む）
- [x] プレビュー画像からパラメータ再利用
- [x] Send機能（ギャラリー → 各パネル）
- [x] Settings画面
  - [x] モデルディレクトリ登録
  - [x] Advanced CFG表示切替
  - [x] パネル表示切替（LoRA, ControlNet, プリセット）
  - [x] 送信サイズモード設定（absolute/scale）
  - [x] Developer Mode切替
  - [x] localStorage管理
  - [x] 一時画像クリーンアップ
  - [x] サーバー再起動機能

### 高度な機能
- [x] v-prediction モデル対応（guidance rescale自動適用）
- [x] SDXL対応（プール済みエンベディング、time_ids）
- [x] img2img fix steps（ステップ数固定モード）
- [x] タグサジェスト機能
- [x] 複数スケジュールタイプ（Karras, Exponential）
- [x] 確率的サンプラー向けAncestral Seed
- [x] 画像ハッシュ（SHA256）
- [x] 生成画像メタデータ埋め込み（parameters JSON）
- [x] アスペクト比・固定解像度プリセット

## 技術的な詳細

### カスタムサンプリングループ
Diffusersパイプラインをベースにしたカスタム実装により、以下の高度な機能をサポート:
- プロンプト編集（ステップベース切り替え）
- LoRAステップ範囲指定（動的ロード/アンロード）
- ControlNetステップ範囲指定
- Advanced CFG Features（Scheduling, SNR-based, Dynamic Thresholding）
- Ancestral Seed（確率的サンプラーの再現性）

実装ファイル: [backend/core/custom_sampling.py](backend/core/custom_sampling.py)

### Advanced CFG実装

#### CFG Scheduling
シグマベースのスケジューリングにより、ノイズレベルに応じてCFGを動的調整:
- `t = sigma / sigma_max` でタイムステップを正規化（0.0～1.0）
- Linear: `cfg = min + (max - min) * t`
- Quadratic: `cfg = min + (max - min) * (t ** power)`
- Cosine: `cfg = min + (max - min) * ((1 - cos(t * π)) / 2)`

#### SNR-Based Adaptive CFG
各ステップのSNRを計算し、CFGを自動調整:
```python
snr = (sigma_max ** 2) / (sigma ** 2 + 1e-8)
snr_normalized = snr / (snr + 1.0)  # 0.0～1.0に正規化
cfg_adjusted = cfg_base * (1.0 - alpha * snr_normalized)
```
- SNRが高い（終盤）→ CFGを下げてアーティファクト抑制
- SNRが低い（序盤）→ CFGを上げてプロンプト従属性強化

#### Dynamic Thresholding
生成された潜在変数の値をパーセンタイルベースでクランプ:
```python
percentile_value = torch.quantile(abs_values, dynamic_threshold_percentile / 100.0)
clamp_value = max(percentile_value, dynamic_threshold_mimic_scale)
noise_pred = noise_pred.clamp(-clamp_value, clamp_value)
```

実装ファイル: [backend/core/cfg_utils.py](backend/core/cfg_utils.py)

### v-prediction対応
- `prediction_type="v_prediction"` の自動検出
- guidance_rescale=0.7 の自動適用（論文: Common Diffusion Noise Schedules and Sample Steps are Flawed）
- `timestep_spacing="trailing"` の適用

### データベース
- SQLite + SQLAlchemy
- 自動マイグレーション（起動時にスキーマ差分を検出・適用）
- 生成画像メタデータ、パラメータ、モデル情報を保存
- Advanced CFGパラメータの保存・表示対応

実装ファイル: [backend/database/models.py](backend/database/models.py)

### リアルタイムプレビュー
- WebSocket（Server-Sent Events）によるストリーミング配信
- TAESD（Tiny AutoEncoder for Stable Diffusion）による高速デコード
- 5ステップごとに更新（設定可能）
- Developer Mode時はCFGメトリクス（SNR, CFG scale）も配信

## 今後の実装予定

- [ ] バッチ生成（現状はLoop Generationで代替可能）
- [ ] 画像編集機能の強化
- [ ] より高度な検索・フィルタリング（タグベース、日付範囲）
- [ ] APIドキュメント整備（OpenAPI/Swagger）
- [ ] Upscaler統合（RealESRGAN, SwinIR等）
- [ ] その他のプリプロセッサー対応
- [ ] カスタムスケジューラーの追加

## 未実装機能

以下の機能はUI/パラメータとして存在しますが、バックエンドでの実装は未完了です：

### Inpaint at full resolution
- **説明**: マスク領域のみを切り出して高解像度で処理し、元画像に合成する機能
- **理由**: 局所的な再生成はモデルの学習方法を考慮すると困難なため
- **状態**: UIでコメントアウト済み、パラメータは送信されるが処理されない
- **関連パラメータ**:
  - `inpaint_full_res` (boolean)
  - `inpaint_full_res_padding` (integer): 切り出し時の余白サイズ

将来的に実装する場合は、マスク領域のバウンディングボックス検出、切り出し、リサイズ、処理、貼り付けの一連の処理が必要です。

## 謝辞

このプロジェクトは以下のオープンソースプロジェクトを参考に開発されました：

- **[ostris/ai-toolkit](https://github.com/ostris/ai-toolkit)** (MIT License) - LoRA学習アーキテクチャ、量子化アプローチ
- **[kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)** (Apache-2.0) - FP8量子化実装、学習パイプライン構造

## 注意事項

- GPUメモリが不足する場合は、画像サイズやバッチサイズを調整してください
- モデルファイルは含まれていません。別途ダウンロードが必要です
- SDXL使用時は12GB以上のVRAM推奨
- v-predictionモデルは自動検出されますが、一部のモデルで調整が必要な場合があります
- 使用するモデル（Stable Diffusion, LoRA等）のライセンスは各モデルの規約に従ってください
