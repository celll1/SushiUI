# SushiUI

SushiUI は、画像・動画・音声の生成と学習をひとつのローカル UI で扱うためのプロジェクトです。FastAPI バックエンドと Next.js フロントエンドで構成され、生成、編集、データセット管理、学習、メトリクス確認までを同じワークフローから操作できます。

## 対応アーキテクチャ

生成側の正本は `backend/core/model_loader.py` の `ModelType`、学習側の正本は `backend/core/training/arch/__init__.py` の `ARCH_REGISTRY` です。現在は 16 の生成アーキテクチャを扱います。

| 種別 | アーキテクチャ |
|---|---|
| 画像 | SD 1.5、SDXL、Z-Image、Flux2、Anima、Lens、Krea2、Ideogram4、MiniT2I、SenseNova U1.5、SenseNova SDXL Chimera |
| 動画＋音声 | LTX-2.3、MiniMax-H3 |
| 音声 | ACE-Step 1.5、MiniMax Music 3、YuE2 |

学習レジストリには 15 アーキテクチャが登録されています。MiniMax Music 3 は生成専用で、YuE2 の学習は Phase-A ABC-planner LoRA に限定されます。方式や重み形式、CFG、VAE、attention、学習可否などの詳細は [Model facts](docs/guides/MODEL_FACTS.md) と [アーキテクチャ別リファレンス](docs/reference/architectures/README.md) を参照してください。

## 主な機能

- txt2img、img2img、inpaint、outpaint、upscale と参照画像条件付け
- 動画・音声生成、および対応モデルでの動画と音声の同時生成
- LoRA、フルパラメータ、tagger、VAE decoder の学習
- 学習再開、サンプル生成、debug latent、リアルタイムメトリクス
- モデル、LoRA、ControlNet、データセット、タグ、生成履歴の管理
- architecture capability に基づく adapter 適用と学習可否判定
- attention backend、block swap、CPU offload、量子化モデルなどのメモリ最適化
- `/api/v1` の REST API、OpenAPI 仕様、WebSocket による進捗通知

機能の対応範囲はモデルごとに異なります。実装上の境界は [Architecture map](docs/guides/ARCHITECTURE_MAP.md)、API は [openapi.yaml](openapi.yaml)、学習パラメータは [Training parameters guide](backend/core/training/TRAINING_PARAMS_GUIDE.md) が正本です。

## セットアップ

現行ランチャーは Windows 向けです。Python、Node.js、対応する GPU ドライバと PyTorch/CUDA 環境を用意してください。必要な VRAM はモデル、解像度、量子化、offload、学習方式によって大きく異なるため、共通の最低値は設けていません。

1. リポジトリを取得します。
2. 使用する PyTorch/CUDA の組み合わせを環境に合わせて準備します。
3. ルートの `start.bat` を実行します。

`start.bat` は `venv` がなければ作成し、`backend/requirements.txt` と `frontend/package.json` の変更を検出して依存関係を更新した後、バックエンドとフロントエンドを起動します。

起動後の既定 URL:

- UI: `http://localhost:3000`
- API: `http://localhost:8000`

モデルファイルは配布物に含まれません。各モデルの提供元から取得し、その利用条件を確認したうえで UI から設定してください。

## API と自動化

REST API は `/api/v1` 以下にあり、仕様は [openapi.yaml](openapi.yaml) に同期されています。直接 API を利用する場合は [API testing guide](docs/guides/API_TESTING.md) と [サンプルスクリプト](examples/api/) を参照してください。

## リポジトリ構成

```text
SushiUI/
├── backend/                 FastAPI、推論、モデル、学習
├── frontend/                Next.js UI
├── docs/                    設計、運用、アーキテクチャ資料
├── examples/api/            REST API の利用例
├── models/                  ローカルモデル置き場
├── lora/                    LoRA 置き場
├── controlnet/              ControlNet 置き場
├── training/                学習設定と成果物
├── outputs/                 生成出力
├── openapi.yaml             API 仕様
└── start.bat                Windows ランチャー
```

詳細なコード所有範囲は [Architecture map](docs/guides/ARCHITECTURE_MAP.md)、文書索引は [Documentation index](docs/README.md) を参照してください。

## ライセンスと第三者表示

現時点で、SushiUI 独自のコードに対するプロジェクトレベルのライセンスは宣言されていません。リポジトリ全体を MIT、Apache-2.0、その他のオープンソースライセンスの下で利用・再配布できるとはみなさないでください。

同梱・派生した第三者コードには、それぞれのライセンスと notice 条件が適用されます。[Third-party code provenance](docs/legal/THIRD_PARTY_PROVENANCE.md) と `docs/legal/licenses/` は、それらの出典・表示要件・監査状況を記録するためのものです。この表示は、SushiUI 独自のコードに対するライセンス付与を意味しません。

第三者コンポーネントの再配布を検討する場合は、同文書の未完了の notice/file audit と redistribution gate を確認してください。これらの条件を満たすことと、SushiUI 独自のコードについて許諾を得ることは別です。

モデル重みはこのリポジトリに含まれず、SushiUI のコードライセンスの対象でもありません。利用者が取得元の利用条件を確認してください。

## 開発資料

- [Documentation index](docs/README.md)
- [Architecture map](docs/guides/ARCHITECTURE_MAP.md)
- [Model facts](docs/guides/MODEL_FACTS.md)
- [Request lifecycle](docs/guides/REQUEST_LIFECYCLE.md)
- [Training API reference](backend/core/training/API_REFERENCE.md)
- [Third-party provenance](docs/legal/THIRD_PARTY_PROVENANCE.md)

このプロジェクトは活発に開発中です。モデルごとの制約や既知の境界は、README の概略より上記の正本文書を優先してください。
