# 🍣 SushiUI

Stable Diffusion 1.5/XL対応の画像生成Webアプリケーション

## 特徴

- **txt2img**: テキストから画像を生成
- **img2img**: 画像から画像を生成（デノイジング強度調整可能）
- **Inpainting**: マスク領域の再生成
- **画像ビューワー**: 生成画像の閲覧とメタデータ検索
- **高度な機能**: プロンプト編集、LoRA、複数サンプラー対応

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
│   ├── api/              # APIエンドポイント
│   ├── core/             # コア機能（パイプライン、サンプリングループ）
│   ├── extensions/       # 拡張機能システム（将来実装予定）
│   ├── utils/            # ユーティリティ
│   ├── database/         # データベースモデル、自動マイグレーション
│   ├── config/           # 設定
│   ├── main.py           # エントリーポイント
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── app/          # Next.js App Router
│   │   ├── components/   # Reactコンポーネント
│   │   ├── utils/        # API クライアント
│   │   └── lib/          # ユーティリティ
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

#### プロンプト編集
プロンプトに `[prompt1:prompt2:0.5]` 形式で記述することで、生成途中でプロンプトを切り替えられます。

- 例: `[cat:dog:0.3]` → 最初の30%はcat、その後dog

#### プロンプト強調構文
- `(word)` - 1.1倍強調
- `((word))` - 1.21倍強調
- `(word:1.5)` - 1.5倍強調
- `[word]` - 0.9倍弱体化

#### LoRA
- LoRAタブで複数のLoRAを追加可能
- 重み調整（-2.0 〜 2.0）
- ステップ範囲指定で特定のステップ範囲でのみLoRAを適用

#### ControlNet
- ControlNetタブで画像をアップロード
- Conditioning Scale調整
- ガイダンス範囲指定（開始・終了ステップ）

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
- [x] カスタムサンプリングループ
- [x] プロンプト編集（ステップベース切り替え）
- [x] プロンプト強調構文
- [x] WebSocketリアルタイム進捗表示
- [x] TAESDプレビュー
- [x] 自動データベースマイグレーション

### モデル・拡張機能
- [x] LoRA（マルチLoRA、ステップ範囲指定）
- [x] ControlNet（マルチControlNet対応）

### UI機能
- [x] 画像ビューワー・ギャラリー
- [x] メタデータ検索・フィルタリング
- [x] 生成パラメータ表示
- [x] プレビュー画像からパラメータ再利用
- [x] Settings画面（モデルディレクトリ登録）

### 高度な機能
- [x] v-prediction モデル対応（guidance rescale自動適用）
- [x] SDXL対応（プール済みエンベディング、time_ids）
- [x] img2img fix steps（ステップ数固定モード）
- [x] タグサジェスト機能
- [x] 複数スケジュールタイプ（Karras, Exponential）
- [x] 確率的サンプラー向けAncestral Seed

## 技術的な詳細

### カスタムサンプリングループ
Diffusersパイプラインをベースにしたカスタム実装により、プロンプト編集、ControlNet、LoRAステップ範囲などの高度な機能をサポートしています。

### v-prediction対応
- `prediction_type="v_prediction"` の自動検出
- guidance_rescale=0.7 の自動適用（論文: Common Diffusion Noise Schedules and Sample Steps are Flawed）
- `timestep_spacing="trailing"` の適用

### データベース
- SQLite + SQLAlchemy
- 自動マイグレーション（起動時にスキーマ差分を検出・適用）
- 生成画像メタデータ、パラメータ、モデル情報を保存

## 今後の実装予定

- [ ] バッチ生成
- [ ] 画像編集機能の強化
- [ ] より高度な検索・フィルタリング
- [ ] APIドキュメント整備

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

## 注意事項

- GPUメモリが不足する場合は、画像サイズやバッチサイズを調整してください
- モデルファイルは含まれていません。別途ダウンロードが必要です
- SDXL使用時は12GB以上のVRAM推奨
- v-predictionモデルは自動検出されますが、一部のモデルで調整が必要な場合があります
- 使用するモデル（Stable Diffusion, LoRA等）のライセンスは各モデルの規約に従ってください
