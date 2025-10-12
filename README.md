# Stable Diffusion WebUI

Stable Diffusion 1.5/XL対応の画像生成Webアプリケーション

## 特徴

- **txt2img**: テキストから画像を生成
- **img2img**: 画像から画像を生成（未実装）
- **Inpainting**: マスク領域の再生成（未実装）
- **拡張機能**: Hires Fix, ControlNet, Tiled VAE対応予定
- **画像ビューワー**: 生成画像の閲覧とメタデータ検索

## 技術スタック

### バックエンド
- Python
- FastAPI
- PyTorch
- Diffusers
- SQLAlchemy

### フロントエンド
- Next.js 14
- TypeScript
- Tailwind CSS
- Axios

## セットアップ

### 必要要件
- Python 3.10+
- Node.js 18+
- CUDA対応GPU（推奨）

### バックエンドのセットアップ

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

4. 依存関係をインストール
```bash
cd backend
pip install -r requirements.txt
```

5. モデルディレクトリを作成
```bash
mkdir ../models ../outputs ../thumbnails
```

6. Stable Diffusionモデルをダウンロード
```bash
# 例: Hugging Faceから
# models/ ディレクトリにモデルを配置
```

7. バックエンドサーバーを起動
```bash
python main.py
```

サーバーは `http://localhost:8000` で起動します。

### フロントエンドのセットアップ

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

## プロジェクト構造

```
webui_cl/
├── backend/
│   ├── api/              # APIエンドポイント
│   ├── core/             # コア機能（パイプライン）
│   ├── extensions/       # 拡張機能システム
│   ├── utils/            # ユーティリティ
│   ├── database/         # データベースモデル
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
├── outputs/              # 生成画像
├── thumbnails/           # サムネイル
└── README.md
```

## 使用方法

1. ブラウザで `http://localhost:3000` を開く
2. "Generate" ページでプロンプトを入力
3. パラメータを調整
4. "Generate" ボタンをクリック
5. "Gallery" ページで生成した画像を閲覧

## 開発状況

### ✅ 実装済み
- [x] プロジェクト構造
- [x] バックエンドAPI基盤
- [x] フロントエンドUI基盤
- [x] txt2img基本機能
- [x] 画像ビューワー基本機能
- [x] データベース連携

### 🚧 実装予定
- [ ] img2img機能
- [ ] Inpainting機能
- [ ] Hires Fix拡張
- [ ] ControlNet拡張
- [ ] Tiled VAE拡張
- [ ] WebSocketによる進捗表示
- [ ] 高度な画像検索・フィルタリング
- [ ] 設定画面
- [ ] モデル管理機能
## 注意事項

- 開発中のため、機能が不完全な場合があります
- GPUメモリが不足する場合は、画像サイズやバッチサイズを調整してください
- モデルファイルは含まれていません。別途ダウンロードが必要です
