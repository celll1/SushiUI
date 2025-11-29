# Dataset Management Feature - Requirements Document

**Project**: SushiUI - Stable Diffusion WebUI
**Feature**: Dataset Management for Model Fine-tuning
**Date**: 2025-11-29
**Version**: 1.3.0
**Last Updated**: 2025-11-29

---

## 1. 概要

### 1.1 目的

Stable Diffusion モデル（SD1.5/SDXL）の微調整（Full fine-tuning, LoRA, etc.）を行うために、データセットの管理・準備機能をWebUIに統合する。

### 1.2 背景

- 現在のUIは推論（画像生成）のみに対応
- モデルのトレーニング機能を追加するには、高品質なデータセットの準備が必須
- ai-toolkitを参考にした、効率的なデータセット管理システムの構築

### 1.3 スコープ

**含まれるもの**:
- データセット登録・管理
- 画像とキャプション（タグ）の関連付け
- タグの追加・削除・編集
- タグ検索・フィルタリング
- Auto-tagging（ML推論）
- データセットのプレビュー・検証

**含まれないもの**（将来的な拡張）:
- 実際のモデルトレーニング実行（別フェーズで実装予定）
- データセットの自動拡張（augmentation）実行

---

## 2. 機能要件

### 2.1 データセットの基本構造

#### 2.1.1 サポートするファイル形式

**画像ファイル**:
- `.jpg`, `.jpeg`, `.png`, `.webp`

**キャプションファイル（タグ）**:
- 同名のテキストファイル（例: `image001.png` → `image001.txt`）
- カンマ区切り形式（例: `1girl, long hair, smile, outdoor, cherry blossoms`）

**メタデータソース**:
- テキストファイル（`.txt`）- 標準
- 画像EXIF/XMPメタデータ - サポート予定（xsaver形式など）

#### 2.1.2 データセット構造パターン

**重要**: すべてのデータセットはサブディレクトリ構造を持つことを前提とする。以下のパターンは、キャプション形式とファイル命名規則の違いを示す。

**パターンA: 単一画像 + 単一キャプション**
```
dataset/
  └── subdir/
      ├── image001.png
      ├── image001.txt
      ├── image002.jpg
      ├── image002.txt
      └── ...
```

**パターンB: 複数キャプション（suffix付き）**
```
dataset/
  └── subdir/
      ├── image001.png
      ├── image001_main.txt       # メインキャプション
      ├── image001_alt.txt        # 代替キャプション
      ├── image002.jpg
      ├── image002_main.txt
      └── ...
```

**パターンC: 参照画像を伴うペア（ControlNet, img2img用）**

実例: `<PRIVATE_PATH>`

```
dataset/
  └── batch_20251026_012849_01k8/
      ├── 20251026_01k8e370_01k8e370_source.webp    # 入力画像
      ├── 20251026_01k8e370_01k8e370_target.webp    # 出力画像（正解）
      ├── 20251026_01k8e370_01k8e370_instruction.txt # プロンプト
      ├── 20251026_01k8e370_01k8e371_source.webp
      ├── 20251026_01k8e370_01k8e371_target.webp
      ├── 20251026_01k8e370_01k8e371_instruction.txt
      └── ...
```

**ファイル命名パターンの認識**:
- ベース名 + suffix: `{base}_{suffix}.{ext}`
  - `source`, `target`, `cref`, `mask` などの suffix を認識
  - `instruction`, `caption`, `tags` などのテキストsuffix
- 同一ベース名の画像をグループ化（source/target/crefペア）

**パターンD: EXIF/XMPメタデータ内蔵（xsaver形式など）**
```
dataset/
  └── subdir/
      ├── image001.jpg  # EXIF/XMPにタグ・プロンプトを含む
      ├── image002.jpg
      └── ...
```

**サポート範囲**:
- **サブディレクトリの再帰的検索**: デフォルト有効、深度制限なし（ユーザー設定可能）
- **画像とキャプションの自動ペアリング**: ファイル名ベースで自動認識
- **複数suffixのサポート**: `source`, `target`, `cref`, `mask`, `instruction`, `caption` など
- **EXIFメタデータ読み込み**: 優先度低（Phase 2以降）
- **画像ペアの認識**: source/target/crefなどの関連画像を自動グループ化

---

### 2.2 データセット登録・管理

#### 2.2.1 新規データセット登録

**UI要素**: `/dataset` ページ

**入力項目**:
- **データセット名**: ユーザー定義（例: "my_character_dataset_v1"）
- **データセットパス**: ディレクトリパス（ファイル選択ダイアログ）
- **データセットタイプ**: ドロップダウン
  - `Single Image` - 単一画像 + キャプション
  - `Image Pairs` - 画像ペア（source/target, cref/target, etc.）
  - `Auto Detect` - ファイル名から自動判定（デフォルト）
- **キャプションサフィックス**: カンマ区切り（例: `main,alt,instruction` または空白で単一）
- **画像サフィックス**: カンマ区切り（例: `source,target,cref,mask`、Image Pairsの場合）
- **サブディレクトリ検索**: ON/OFF（デフォルト: ON）
- **検索深度**: 数値入力（空白で無制限、デフォルト: 空白）
- **EXIF/XMP読み込み**: ON/OFF（デフォルト: OFF）

**処理フロー**:
1. 指定ディレクトリを再帰的にスキャン
2. 画像ファイルを検出
3. ファイル名からベース名とsuffixを抽出
4. 同一ベース名のファイルをグループ化（画像ペアの場合）
5. 対応するキャプションファイルを検索
6. EXIF/XMPメタデータを読み込み（設定がONの場合）
7. データベースにインデックス化

**ファイル名パターン認識ロジック**:
```python
# 例: "20251026_01k8e370_01k8e370_source.webp"
# → base_name: "20251026_01k8e370_01k8e370"
# → suffix: "source"

# パターン: {base}_{suffix}.{ext}
# 既知のsuffix: source, target, cref, mask, instruction, caption, main, alt, tags

# グループ化:
# - "20251026_01k8e370_01k8e370_source.webp"
# - "20251026_01k8e370_01k8e370_target.webp"
# - "20251026_01k8e370_01k8e370_instruction.txt"
# → 1つのDatasetItemとして登録、related_imagesに全画像パスを保存
```

**データベーススキーマ（新規テーブル）**:

```python
# backend/database/models.py

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)  # ユーザー定義名
    path = Column(String)  # ルートディレクトリパス

    # Dataset configuration
    caption_suffixes = Column(JSON)  # ["main", "alt", "instruction"] or []
    image_suffixes = Column(JSON)  # ["source", "target", "cref", "mask"] or []
    recursive = Column(Boolean, default=True)
    max_depth = Column(Integer, nullable=True)  # None = 無制限
    read_exif = Column(Boolean, default=False)  # EXIF/XMPメタデータを読み込むか

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Statistics
    total_images = Column(Integer, default=0)
    total_items = Column(Integer, default=0)  # グループ化されたアイテム数（ペア含む）
    total_captions = Column(Integer, default=0)
    indexed_at = Column(DateTime, nullable=True)


class DatasetItem(Base):
    __tablename__ = "dataset_items"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), index=True)

    # Item type and grouping
    item_type = Column(String, default="single")  # "single", "pair", "group"
    base_name = Column(String, index=True)  # ファイルのベース名（suffix除く）
    group_id = Column(String, nullable=True, index=True)  # 同一グループのアイテムをまとめるID

    # File paths (primary image)
    image_path = Column(String, index=True)  # 絶対パス（メイン画像）
    relative_path = Column(String)  # データセットルートからの相対パス
    image_suffix = Column(String, nullable=True)  # "source", "target", "cref", "mask", etc.

    # Related images (for paired datasets)
    related_images = Column(JSON)  # {"source": "path/to/source.webp", "target": "path/to/target.webp", ...}

    # Image metadata
    width = Column(Integer)
    height = Column(Integer)
    file_size = Column(Integer)  # bytes
    image_hash = Column(String, index=True)  # SHA256

    # Caption metadata
    caption_paths = Column(JSON)  # {"main": "path/to/main.txt", "instruction": "path/to/instruction.txt", ...}
    exif_data = Column(JSON, nullable=True)  # EXIF/XMPメタデータ（read_exif=True時）

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DatasetCaption(Base):
    __tablename__ = "dataset_captions"

    id = Column(Integer, primary_key=True, index=True)
    item_id = Column(Integer, ForeignKey("dataset_items.id"), index=True)

    # Caption type and content
    caption_type = Column(String, index=True)  # "tags", "natural_language", "social_media", "instruction", etc.
    caption_subtype = Column(String, nullable=True)  # "main", "alt", "x_post", "description", etc.
    content = Column(Text)  # キャプション全体のテキスト
    language = Column(String, nullable=True)  # "en", "ja", etc.

    # Source metadata
    source = Column(String, default="manual")  # "manual", "txt_file", "exif", "auto_wd14", "auto_joytag", etc.
    source_field = Column(String, nullable=True)  # EXIF/XMPフィールド名（例: "ImageDescription", "UserComment"）

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DatasetTag(Base):
    __tablename__ = "dataset_tags"

    id = Column(Integer, primary_key=True, index=True)
    item_id = Column(Integer, ForeignKey("dataset_items.id"), index=True)
    caption_id = Column(Integer, ForeignKey("dataset_captions.id"), nullable=True, index=True)  # どのキャプションから来たか

    # Tag content
    tag = Column(String, index=True)  # 個別タグ（例: "1girl"）
    position = Column(Integer)  # タグの順序（0-indexed）

    # Tag metadata
    tag_group = Column(String, nullable=True, index=True)  # "Character", "Quality", etc.
    confidence = Column(Float, nullable=True)  # Auto-tagging時の信頼度（0.0-1.0）
    source = Column(String, default="manual")  # "manual", "auto_wd14", "auto_joytag", "parsed_from_caption", etc.

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class TagAlias(Base):
    __tablename__ = "tag_aliases"

    id = Column(Integer, primary_key=True, index=True)
    source_tag = Column(String, unique=True, index=True)  # 元のタグ
    target_tag = Column(String, index=True)  # 置き換え先タグ

    created_at = Column(DateTime, default=datetime.utcnow)


class TagGroup(Base):
    __tablename__ = "tag_groups"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)  # "Character", "Quality", "Meta", etc.
    color = Column(String, nullable=True)  # UI表示用カラー（例: "#FF5733"）
    tags = Column(JSON)  # このグループに属するタグのリスト

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), index=True)

    # Run identification
    run_name = Column(String, index=True)  # "lora_character_v1_run_001"
    run_number = Column(Integer, index=True)  # 同一設定での連番（resume用）
    parent_run_id = Column(Integer, ForeignKey("training_runs.id"), nullable=True, index=True)  # resume元のrun

    # Training configuration
    model_type = Column(String)  # "lora", "full", "dreambooth", etc.
    base_model = Column(String)  # "animagine-xl-3.1"
    caption_type = Column(String)  # "tags", "natural_language", etc.
    training_config = Column(JSON)  # 全学習パラメータ（lr, epochs, batch_size, etc.）

    # Run status
    status = Column(String, default="pending")  # "pending", "running", "completed", "failed", "cancelled"
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Training statistics
    total_epochs = Column(Integer)  # 予定エポック数
    completed_epochs = Column(Integer, default=0)  # 完了エポック数
    total_steps = Column(Integer)  # 予定ステップ数
    completed_steps = Column(Integer, default=0)  # 完了ステップ数
    total_samples_seen = Column(Integer, default=0)  # 学習に使用された総サンプル数（重複含む）

    # Output
    output_path = Column(String, nullable=True)  # 出力モデルのパス
    checkpoint_dir = Column(String, nullable=True)  # チェックポイントディレクトリ

    # Metadata export
    metadata_file = Column(String, nullable=True)  # エクスポートされたメタデータファイルのパス
    config_file = Column(String, nullable=True)  # ai-toolkit設定ファイルのパス

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class TrainingItemUsage(Base):
    __tablename__ = "training_item_usage"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("training_runs.id"), index=True)
    item_id = Column(Integer, ForeignKey("dataset_items.id"), index=True)

    # Usage tracking
    times_seen = Column(Integer, default=0)  # このrunで何回学習に使われたか
    first_seen_epoch = Column(Integer, nullable=True)  # 最初に使われたエポック
    last_seen_epoch = Column(Integer, nullable=True)  # 最後に使われたエポック
    first_seen_step = Column(Integer, nullable=True)  # 最初に使われたステップ
    last_seen_step = Column(Integer, nullable=True)  # 最後に使われたステップ

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Composite unique index: 1つのrunで同じitemは1レコードのみ
    __table_args__ = (
        Index('idx_run_item_unique', 'run_id', 'item_id', unique=True),
    )


class TrainingItemStats(Base):
    __tablename__ = "training_item_stats"

    id = Column(Integer, primary_key=True, index=True)
    item_id = Column(Integer, ForeignKey("dataset_items.id"), unique=True, index=True)

    # Cumulative statistics (全run合計)
    total_times_seen = Column(Integer, default=0)  # 全runで何回学習に使われたか
    total_runs_participated = Column(Integer, default=0)  # いくつのrunで使用されたか
    first_trained_at = Column(DateTime, nullable=True)  # 初めて学習に使われた日時
    last_trained_at = Column(DateTime, nullable=True)  # 最後に学習に使われた日時

    # Usage distribution (run別の使用回数の分散・偏り検出用)
    usage_variance = Column(Float, nullable=True)  # 使用回数の分散
    usage_std_dev = Column(Float, nullable=True)  # 使用回数の標準偏差

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
```

#### 2.2.2 キャプションタイプの詳細

**重要**: 1つの画像に対して複数のキャプションタイプを保持できる柔軟な設計。トレーニング時に使用するキャプションタイプを選択可能。

**サポートするキャプションタイプ**:

1. **`tags`** - Danbooruスタイルタグ（カンマ区切り）
   - 例: `1girl, long hair, smile, outdoor, cherry blossoms, masterpiece, best quality`
   - ソース: `.txt`ファイル、Auto-tagging（WD14, JoyTag）
   - 用途: SD1.5/SDXL標準学習

2. **`natural_language`** - 自然言語キャプション（主に英語）
   - 例: `A beautiful anime girl with long flowing hair stands in a cherry blossom garden, smiling warmly at the viewer. The scene is bathed in soft sunlight with pink petals floating in the air.`
   - ソース: `.txt`ファイル、EXIF（ImageDescription）、Auto-captioning（BLIP, LLaVA）
   - 用途: Flux学習、より詳細な構図学習

3. **`social_media`** - SNS投稿文（X/Twitter、Pixivなど）
   - 例（X本文）: `新作イラスト完成！桜の下で微笑む美少女を描いてみました🌸✨ #オリジナル #イラスト #桜`
   - ソース: EXIF（UserComment、XPComment）、手動入力
   - 用途: カジュアルな表現の学習、文脈理解

4. **`instruction`** - 指示文（ControlNet、img2imgペア用）
   - 例: `Make the character smile more brightly and add cherry blossoms in the background`
   - ソース: `_instruction.txt`ファイル
   - 用途: Instruction-following model学習

5. **`description`** - 詳細説明（複数行）
   - 例:
     ```
     Character: Original anime girl
     Hair: Long, flowing, light brown
     Expression: Gentle smile
     Setting: Cherry blossom garden in spring
     Lighting: Soft afternoon sunlight
     Mood: Peaceful and serene
     ```
   - ソース: `_description.txt`ファイル、EXIF
   - 用途: 細かい属性制御の学習

**データベーススキーマの使用例**:

```python
# 同一画像に対して複数のキャプションを保存
item = DatasetItem(image_path="image001.png", ...)

# タグキャプション（.txtファイル）
caption_tags = DatasetCaption(
    item_id=item.id,
    caption_type="tags",
    caption_subtype="main",
    content="1girl, long hair, smile, outdoor, cherry blossoms, masterpiece",
    source="txt_file"
)

# 自然言語キャプション（EXIFから）
caption_natural = DatasetCaption(
    item_id=item.id,
    caption_type="natural_language",
    caption_subtype="description",
    content="A beautiful anime girl with long flowing hair...",
    language="en",
    source="exif",
    source_field="ImageDescription"
)

# SNS投稿文（EXIFから）
caption_social = DatasetCaption(
    item_id=item.id,
    caption_type="social_media",
    caption_subtype="x_post",
    content="新作イラスト完成！桜の下で微笑む美少女を描いてみました🌸✨",
    language="ja",
    source="exif",
    source_field="UserComment"
)

# タグは個別にDatasetTagテーブルへ
for i, tag in enumerate(caption_tags.content.split(", ")):
    DatasetTag(
        item_id=item.id,
        caption_id=caption_tags.id,
        tag=tag.strip(),
        position=i,
        source="txt_file"
    )
```

**EXIF/XMPメタデータマッピング**:

| EXIFフィールド | キャプションタイプ | 説明 |
|--------------|-----------------|------|
| `ImageDescription` | `natural_language` | 画像の説明文 |
| `UserComment` | `social_media` | ユーザーコメント（SNS投稿文など） |
| `XPComment` | `social_media` | Windowsコメント |
| `XPKeywords` | `tags` | Windowsキーワード |
| `Keywords` | `tags` | IPTCキーワード |
| `Caption-Abstract` | `description` | IPTC詳細説明 |

**トレーニング時のキャプション選択**:

UI上で以下を選択可能:
- 使用するキャプションタイプ（tags, natural_language, social_media, etc.）
- キャプションサブタイプ（main, alt, x_post, etc.）
- 複数キャプションの混合（例: tags + natural_language）
- フォールバック設定（第一優先がない場合、第二優先を使用）

#### 2.2.3 データセット一覧表示

**UI要素**: `/dataset` ページ（データセットリスト）

**表示項目**:
- データセット名
- 画像数
- 最終更新日時
- アクション（編集、削除、インデックス再構築）

**操作**:
- **編集**: 設定の変更（サフィックス、検索深度など）
- **削除**: データセットとインデックスの削除（確認ダイアログ）
- **再インデックス**: ディレクトリを再スキャンして更新

---

### 2.3 データセットアイテム表示・編集

#### 2.3.1 画像グリッド表示

**UI要素**: `/dataset/[dataset_id]` ページ

**レイアウト**:
```
+--------------------------------------------------+
| [Dataset: my_character_dataset_v1]               |
| [Search: ____] [Filter by tag: ____] [Sort: ▼]  |
+--------------------------------------------------+
| [Image Grid - 4 columns, responsive]             |
| +-------+  +-------+  +-------+  +-------+       |
| | img1  |  | img2  |  | img3  |  | img4  |       |
| | tags  |  | tags  |  | tags  |  | tags  |       |
| +-------+  +-------+  +-------+  +-------+       |
| ...                                              |
+--------------------------------------------------+
| [Pagination: < 1 2 3 4 5 >]                      |
+--------------------------------------------------+
```

**機能**:
- サムネイル表示（lazy loading）
- タグのプレビュー表示（最大3タグ、残りは "..."）
- クリックで詳細モーダル表示

#### 2.3.2 画像詳細モーダル

**UI要素**: モーダルダイアログ（3カラムレイアウト）

**左側パネル: 画像プレビュー**
- メイン画像のフルサイズ表示
- 画像ペアの場合: source/target/crefの切り替えタブ
- 解像度、ファイルサイズ表示
- 前/次の画像ナビゲーション（キーボード: ←/→）

**中央パネル: キャプション管理**

**キャプションタイプタブ**:
- `Tags` - タグ編集（デフォルト）
- `Natural Language` - 自然言語キャプション
- `Social Media` - SNS投稿文
- `Instruction` - 指示文
- `Description` - 詳細説明
- `All Captions` - すべてのキャプションを一覧表示

**タグ編集タブ（`Tags`）**:
- **サブタイプ選択**: ドロップダウン（main, alt, auto_wd14, etc.）
- **タグリスト表示**: ピルUI（色付きバッジでグループ化）
- **タグ追加**: 入力フィールド + オートコンプリート
- **タグ削除**: 各タグの "×" ボタン
- **タグ並べ替え**: ドラッグ&ドロップ（react-beautiful-dnd）
- **一括操作**:
  - "Copy Tags" - タグをクリップボードにコピー
  - "Paste Tags" - クリップボードからタグを貼り付け
  - "Clear All" - すべてのタグを削除

**自然言語キャプション編集タブ（`Natural Language`）**:
- **テキストエリア**: 複数行入力
- **ソース表示**: txtファイル/EXIF/Auto-captioningのどれから来たか
- **言語選択**: en/ja/auto
- **AI生成ボタン**: BLIP/LLaVAで自動生成（オプション）

**SNS投稿文編集タブ（`Social Media`）**:
- **テキストエリア**: 複数行入力
- **ソース表示**: EXIF（UserComment）/手動入力
- **言語選択**: ja/en/auto
- **文字数カウント**: Twitter/X形式（280文字）

**指示文編集タブ（`Instruction`）** - 画像ペアの場合のみ表示:
- **テキストエリア**: source→targetへの変換指示
- 例: "Make the character smile more brightly"

**すべてのキャプション表示タブ（`All Captions`）**:
- 画像に関連するすべてのキャプションを一覧表示
- キャプションタイプ・ソース・作成日時を表示
- 各キャプションの編集・削除ボタン

**右側パネル: メタデータとアクション**

**画像情報**:
- ファイル名、パス
- 解像度、ファイルサイズ
- ハッシュ値
- 作成日時、更新日時

**関連画像** - 画像ペアの場合:
- Source画像のサムネイル
- Target画像のサムネイル
- Cref画像のサムネイル（あれば）
- クリックで左側パネルに表示

**EXIF/XMPメタデータ** - read_exif=True の場合:
- 折りたたみ可能なセクション
- すべてのEXIFフィールドを表示
- キャプションにマッピングされたフィールドをハイライト

**アクション**:
- "Save" - 変更を保存（DB + txtファイル）
- "Revert" - 変更を破棄
- "Auto-tag" - AI推論でタグを追加
- "Export Caption" - キャプションをファイルにエクスポート

**タグ入力のオートコンプリート**:
- データセット内の既存タグから候補表示（頻度順）
- グローバルタググループから候補表示
- 入力中に動的フィルタリング
- キーボードナビゲーション（↑/↓/Enter）

**保存処理**:
- **リアルタイム保存**: タグ追加/削除時に自動保存（デバウンス500ms）
- **手動保存**: "Save" ボタンで明示的に保存
- **保存先**:
  - データベース（DatasetCaption, DatasetTag）
  - txtファイル（対応するキャプションタイプのみ）
  - EXIF/XMP（オプション、対応フィールドのみ）

---

### 2.4 タグ検索・フィルタリング

#### 2.4.1 検索機能

**検索バー**: `/dataset/[dataset_id]` ページ上部

**サポートする検索方法**:

1. **タグ検索**:
   - 単一タグ: `1girl`
   - 複数タグ（AND検索）: `1girl, long hair`
   - 複数タグ（OR検索）: `1girl | smile`
   - 除外検索: `1girl, -short hair`

2. **グループ検索**:
   - `group:Character` - Characterグループのタグを持つ画像
   - `group:Quality` - Qualityグループのタグを持つ画像

3. **メタデータ検索**:
   - `width:>=1024` - 幅1024px以上
   - `height:<512` - 高さ512px未満
   - `tags:>10` - タグ数10個以上

**実装**:
- バックエンドで SQLAlchemy のクエリビルダーを使用
- フロントエンドで検索クエリをパース

#### 2.4.2 フィルター機能

**フィルターパネル**: サイドバーまたはドロップダウン

**フィルター項目**:
- **タググループ**: 複数選択（Character, Quality, etc.）
- **画像サイズ**: レンジスライダー（width, height）
- **タグ数**: レンジスライダー（最小-最大）
- **キャプションタイプ**: main, alt, etc.

**ソート機能**:
- 作成日時（新しい順/古い順）
- ファイル名（昇順/降順）
- 画像サイズ（大きい順/小さい順）
- タグ数（多い順/少ない順）

---

### 2.5 タグ管理

#### 2.5.1 タググループ管理

**UI要素**: `/dataset/tags` ページ

**機能**:
- タググループ一覧表示（Character, Quality, Meta, etc.）
- 新規グループ作成
- グループへのタグ追加/削除
- グループの色設定（UI表示用）

**初期グループ**（ai-toolkit参考）:
- **Character**: キャラクター名（例: "example_character", "rem (re:zero)"）
- **Copyright**: 作品名（例: "vocaloid", "re:zero"）
- **Artist**: アーティスト名（例: "wlop", "artgerm"）
- **General**: 一般タグ（例: "1girl", "long hair", "smile"）
- **Quality**: 品質タグ（例: "masterpiece", "best quality", "absurdres"）
- **Rating**: レーティング（例: "safe", "sensitive", "nsfw"）
- **Meta**: メタ情報（例: "commentary", "translation request"）

**JSONファイルからのインポート**:
- ai-toolkitの `taggroup/*.json` を参考に初期データを読み込み

#### 2.5.2 タグエイリアス管理

**UI要素**: `/dataset/tags/aliases` ページ

**機能**:
- エイリアスの一覧表示（source_tag → target_tag）
- 新規エイリアス追加
- エイリアス削除

**使用例**:
- `girl` → `1girl`
- `masterwork` → `masterpiece`
- `ultra_detailed` → `extremely detailed`

**適用タイミング**:
- タグ追加時に自動置換
- 既存タグへのバッチ適用（オプション）

#### 2.5.3 高速タグ振り分けUI

**UI要素**: `/dataset/[dataset_id]/quick-tag` ページ

**レイアウト**:
```
+--------------------------------------------------+
| [Current Image (large preview)]                  |
| Image 1/100                                      |
+--------------------------------------------------+
| Quick Tag Buttons:                               |
| [1girl] [2girls] [multiple girls]                |
| [solo] [group]                                   |
| [masterpiece] [best quality] [high quality]      |
| [safe] [sensitive] [nsfw]                        |
+--------------------------------------------------+
| [Custom Tag Input: ____]  [Add]                  |
+--------------------------------------------------+
| Current Tags: [1girl] [long hair] [smile]        |
+--------------------------------------------------+
| [< Previous]  [Save & Next >]  [Skip >]          |
+--------------------------------------------------+
```

**機能**:
- 画像を1枚ずつ表示
- プリセットボタンでワンクリックタグ付け
- カスタムタグ入力
- キーボードショートカット対応:
  - `1-9`: プリセットボタン1-9
  - `Enter`: Save & Next
  - `→`: Skip
  - `←`: Previous

**プリセット設定**:
- ユーザーが頻繁に使うタグをプリセットとして保存
- データセットごとに異なるプリセット設定可能

#### 2.5.4 タグ辞書管理（Tag Dictionary Management）

**目的**: Danbooruタグリストのようなカテゴリ別タグ辞書を管理し、オートコンプリートや推論に活用

**現在のタグリスト構造**:
```
taglist/
  ├── Character.json     # 347,519 タグ
  ├── Artist.json        # ~13.4MB
  ├── Copyright.json     # ~1.5MB
  ├── General.json       # ~14.8MB
  ├── Meta.json          # ~32KB
  └── Model.json         # ~34KB
```

**JSON形式**:
```json
{
  "example_character": 123034,
  "example_character_a": 86568,
  "kirisame_marisa": 76868,
  "custom_character_name": 42
}
```

キー: タグ名（アンダースコア区切り）
値: カウント数（Danbooruでの出現回数、またはユーザー定義値）

##### 2.5.4.1 データベーススキーマ

**新規テーブル**: `tag_dictionary`

```python
# backend/database/models.py

class TagDictionary(Base):
    __tablename__ = "tag_dictionary"

    id = Column(Integer, primary_key=True, index=True)
    tag = Column(String, unique=True, index=True)  # タグ名（例: "example_character"）
    category = Column(String, index=True)  # カテゴリ（"character", "artist", "copyright", "general", "meta", "model"）
    count = Column(Integer, default=0)  # 出現回数（Danbooruカウントまたはユーザー定義）

    # Tag metadata
    display_name = Column(String, nullable=True)  # 表示名（例: "example_character"）
    aliases = Column(JSON, nullable=True)  # エイリアス（例: ["ExampleCharacter", "例示キャラクター"]）
    description = Column(Text, nullable=True)  # 説明文
    wiki_url = Column(String, nullable=True)  # Danbooru Wiki URL

    # Source tracking
    source = Column(String, default="danbooru")  # "danbooru", "user_custom", "auto_detected"
    is_official = Column(Boolean, default=True)  # Danbooru公式タグかどうか
    is_deprecated = Column(Boolean, default=False)  # 非推奨タグかどうか

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Composite index: category + count (ソート・検索用)
    __table_args__ = (
        Index('idx_category_count', 'category', 'count'),
    )
```

##### 2.5.4.2 タグ辞書管理UI

**UI要素**: `/dataset/tags/dictionary` ページ

**レイアウト**:
```
+----------------------------------------------------------+
| Tag Dictionary Management                                 |
+----------------------------------------------------------+
| [Category: All ▼] [Search: ____] [+ Add New Tag]         |
+----------------------------------------------------------+
| Filter:                                                   |
| [ ] Official Only   [ ] User Custom Only                  |
| [ ] Show Deprecated                                       |
|                                                           |
| Sort by: [Count (Desc) ▼]                                |
+----------------------------------------------------------+
| Tag Name          | Category  | Count    | Source | Actions|
|-------------------|-----------|----------|--------|--------|
| example_character      | Character | 123,034  | DB     | [Edit] |
| example_character_a     | Character | 86,568   | DB     | [Edit] |
| my_oc_character   | Character | 42       | Custom | [Edit] |
| ...                                                       |
+----------------------------------------------------------+
| [Pagination: < 1 2 3 ... 100 >]                          |
+----------------------------------------------------------+
```

**機能**:

1. **カテゴリ選択**
   - All, Character, Artist, Copyright, General, Meta, Model

2. **検索**
   - タグ名で部分一致検索
   - 前方一致、後方一致、完全一致オプション

3. **フィルター**
   - 公式タグのみ表示
   - ユーザーカスタムタグのみ表示
   - 非推奨タグの表示/非表示

4. **ソート**
   - カウント数（降順/昇順）
   - タグ名（辞書順）
   - 作成日時（新しい順/古い順）

##### 2.5.4.3 タグの追加・編集・削除

**新規タグ追加UI**:

モーダルダイアログ:
```
+----------------------------------------------------------+
| Add New Tag                                               |
+----------------------------------------------------------+
| Tag Name: [_______________]  (required)                   |
|   Example: "my_original_character"                        |
|                                                           |
| Display Name: [_______________]  (optional)               |
|   Example: "My Original Character"                        |
|                                                           |
| Category: [Character ▼]                                   |
|                                                           |
| Count: [42]                                               |
|   Tip: Use any number to indicate usage frequency        |
|                                                           |
| Aliases: [_______________]  (optional, comma-separated)   |
|   Example: "my_oc, original_char"                         |
|                                                           |
| Description: [________________]  (optional)               |
|   Example: "Original character for my story"             |
|                                                           |
| [ ] Mark as deprecated                                    |
|                                                           |
| [Cancel]  [Save]                                          |
+----------------------------------------------------------+
```

**バリデーション**:
- タグ名: 必須、英数字とアンダースコアのみ、重複不可
- カテゴリ: 必須、6種類から選択
- カウント: 必須、0以上の整数

**タグ編集UI**:

同様のモーダルダイアログで、既存値を表示して編集可能。

**追加フィールド**（編集時のみ表示）:
- Source: 公式/カスタムの表示（変更不可）
- Created At: 作成日時
- Updated At: 更新日時

**タグ削除**:

確認ダイアログ:
```
+----------------------------------------------------------+
| Delete Tag                                                |
+----------------------------------------------------------+
| Are you sure you want to delete this tag?                |
|                                                           |
| Tag: "my_original_character"                              |
| Category: Character                                       |
| Count: 42                                                 |
|                                                           |
| Warning: This action cannot be undone.                    |
|                                                           |
| [ ] Also delete this tag from all datasets                |
|                                                           |
| [Cancel]  [Delete]                                        |
+----------------------------------------------------------+
```

**削除オプション**:
- タグ辞書からのみ削除（データセット内のタグは保持）
- データセット内のタグも一括削除

##### 2.5.4.4 バルクインポート・エクスポート

**インポート機能**:

**UI要素**: `/dataset/tags/dictionary` の "Import" ボタン

**サポート形式**:
1. **JSON形式**（既存のtaglist形式）
   ```json
   {
     "tag_name_1": 12345,
     "tag_name_2": 67890
   }
   ```

2. **CSV形式**
   ```csv
   tag,category,count,display_name,aliases,description
   example_character,character,123034,example_character,"ExampleCharacter,例示キャラクター",Vocaloid character
   ```

**インポート設定**:
- カテゴリの自動割り当て（ファイル名から推測）
- 既存タグの処理:
  - Skip（スキップ）
  - Update（カウント更新）
  - Merge（エイリアス追加）

**エクスポート機能**:

**UI要素**: `/dataset/tags/dictionary` の "Export" ボタン

**エクスポート設定**:
- カテゴリ選択（All, Character, etc.）
- フォーマット選択（JSON, CSV）
- フィルター適用（公式のみ、カスタムのみ）

**出力例**:
```json
// Character.json
{
  "example_character": 123034,
  "my_oc_character": 42
}
```

##### 2.5.4.5 タグ辞書の活用

**オートコンプリート**:
```typescript
// frontend/src/components/dataset/TagInput.tsx

const searchTags = async (query: string, category?: string) => {
  const response = await api.get("/tag-dictionary/search", {
    params: {
      query: query,
      category: category,
      limit: 20,
      sort_by: "count_desc"  // 人気順
    }
  });

  return response.data.tags;
};
```

**表示順序**:
1. カウント数が高い順（人気タグ優先）
2. 前方一致を優先
3. カテゴリでグループ化（オプション）

**Auto-tagging結果のマッピング**:
```python
# backend/core/tagger.py

def map_tagger_output_to_dictionary(predicted_tags: List[str], db: Session):
    """
    Auto-tagger出力をタグ辞書にマッピング
    - 未知のタグは自動的にGeneral/user_customとして追加（オプション）
    - エイリアスを適用
    """
    mapped_tags = []

    for tag in predicted_tags:
        # タグ辞書で検索
        dict_tag = db.query(TagDictionary).filter_by(tag=tag).first()

        if dict_tag:
            # エイリアス適用
            if dict_tag.is_deprecated and dict_tag.aliases:
                mapped_tags.append(dict_tag.aliases[0])
            else:
                mapped_tags.append(tag)
        else:
            # 未知のタグ
            if auto_add_unknown_tags:
                new_tag = TagDictionary(
                    tag=tag,
                    category="general",
                    count=1,
                    source="auto_detected"
                )
                db.add(new_tag)
                db.commit()

            mapped_tags.append(tag)

    return mapped_tags
```

##### 2.5.4.6 タグ統計と分析

**UI要素**: `/dataset/tags/stats` ページ

**表示内容**:

1. **カテゴリ別統計**
   ```
   Character: 347,519 tags
   General:   245,123 tags
   Artist:    89,234 tags
   Copyright: 23,456 tags
   Meta:      1,234 tags
   Model:     567 tags
   ```

2. **ソース別統計**
   ```
   Danbooru Official: 705,432 tags (98.5%)
   User Custom:       10,234 tags (1.4%)
   Auto Detected:     567 tags (0.1%)
   ```

3. **使用頻度分析**
   - データセット内で実際に使用されているタグ数
   - 未使用タグの数
   - 最も使用されているタグ Top 100

4. **トレンド分析**（将来的な拡張）
   - 最近追加されたタグ
   - 急増しているタグ

##### 2.5.4.7 API エンドポイント

**タグ辞書管理**:
```yaml
# openapi.yaml

/tag-dictionary:
  get:
    summary: List tags from dictionary
    parameters:
      - name: category
        in: query
        schema:
          type: string
          enum: [all, character, artist, copyright, general, meta, model]
      - name: search
        in: query
        schema:
          type: string
      - name: source
        in: query
        schema:
          type: string
          enum: [all, danbooru, user_custom, auto_detected]
      - name: page
        in: query
        schema:
          type: integer
          default: 1
      - name: limit
        in: query
        schema:
          type: integer
          default: 100
      - name: sort_by
        in: query
        schema:
          type: string
          enum: [count_desc, count_asc, name_asc, name_desc, created_desc]
          default: count_desc
    responses:
      '200':
        description: Tag list
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/TagDictionaryListResponse'

  post:
    summary: Add new tag to dictionary
    requestBody:
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/CreateTagRequest'
    responses:
      '201':
        description: Tag created

/tag-dictionary/{tag_id}:
  get:
    summary: Get tag details
    responses:
      '200':
        description: Tag details

  put:
    summary: Update tag
    requestBody:
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/UpdateTagRequest'
    responses:
      '200':
        description: Tag updated

  delete:
    summary: Delete tag
    parameters:
      - name: delete_from_datasets
        in: query
        schema:
          type: boolean
          default: false
    responses:
      '204':
        description: Tag deleted

/tag-dictionary/search:
  get:
    summary: Search tags (for autocomplete)
    parameters:
      - name: query
        in: query
        required: true
        schema:
          type: string
      - name: category
        in: query
        schema:
          type: string
      - name: limit
        in: query
        schema:
          type: integer
          default: 20
    responses:
      '200':
        description: Search results

/tag-dictionary/import:
  post:
    summary: Import tags from JSON/CSV
    requestBody:
      content:
        multipart/form-data:
          schema:
            type: object
            properties:
              file:
                type: string
                format: binary
              category:
                type: string
              conflict_resolution:
                type: string
                enum: [skip, update, merge]
                default: skip
    responses:
      '200':
        description: Import successful
        content:
          application/json:
            schema:
              type: object
              properties:
                imported_count:
                  type: integer
                skipped_count:
                  type: integer
                updated_count:
                  type: integer

/tag-dictionary/export:
  post:
    summary: Export tags to JSON/CSV
    requestBody:
      content:
        application/json:
          schema:
            type: object
            properties:
              category:
                type: string
              format:
                type: string
                enum: [json, csv]
              source_filter:
                type: string
    responses:
      '200':
        description: Export file
        content:
          application/json:
            schema:
              type: object
          text/csv:
            schema:
              type: string

/tag-dictionary/stats:
  get:
    summary: Get tag dictionary statistics
    responses:
      '200':
        description: Statistics
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/TagDictionaryStats'
```

##### 2.5.4.8 初期データのロード

**実装方針**:

```python
# backend/utils/tag_dictionary_loader.py

import json
from pathlib import Path
from sqlalchemy.orm import Session

TAGLIST_DIR = Path(__file__).parent.parent.parent / "taglist"

def load_tag_dictionary_from_json(db: Session, force_reload: bool = False):
    """
    taglist/*.json からタグ辞書をロード
    """
    # 既にロード済みか確認
    existing_count = db.query(TagDictionary).count()
    if existing_count > 0 and not force_reload:
        print(f"Tag dictionary already loaded ({existing_count} tags)")
        return

    category_mapping = {
        "Character.json": "character",
        "Artist.json": "artist",
        "Copyright.json": "copyright",
        "General.json": "general",
        "Meta.json": "meta",
        "Model.json": "model"
    }

    total_imported = 0

    for filename, category in category_mapping.items():
        filepath = TAGLIST_DIR / filename

        if not filepath.exists():
            print(f"Warning: {filepath} not found, skipping")
            continue

        print(f"Loading {filename}...")

        with open(filepath, 'r', encoding='utf-8') as f:
            tags = json.load(f)

        # バッチインサート（高速化）
        batch = []
        for tag_name, count in tags.items():
            batch.append({
                "tag": tag_name,
                "category": category,
                "count": count,
                "source": "danbooru",
                "is_official": True
            })

            # 1000件ごとにコミット
            if len(batch) >= 1000:
                db.bulk_insert_mappings(TagDictionary, batch)
                db.commit()
                total_imported += len(batch)
                batch = []

        # 残りをコミット
        if batch:
            db.bulk_insert_mappings(TagDictionary, batch)
            db.commit()
            total_imported += len(batch)

        print(f"  → Imported {len(tags)} tags from {filename}")

    print(f"Total imported: {total_imported} tags")
```

**起動時の自動ロード**:
```python
# backend/main.py

@app.on_event("startup")
async def startup_event():
    # データベース初期化
    create_db_and_tables()

    # タグ辞書のロード（初回のみ）
    with SessionLocal() as db:
        load_tag_dictionary_from_json(db, force_reload=False)
```

##### 2.5.4.9 実装の優先度

**Phase 1（基本実装）**:
- `TagDictionary` テーブル作成
- JSONファイルからの初期ロード
- 基本的なCRUD API（追加、編集、削除）
- 一覧表示UI（検索、フィルター、ソート）

**Phase 2（編集機能）**:
- タグ追加・編集モーダルUI
- バリデーション
- オートコンプリートでのタグ辞書活用

**Phase 3（高度な機能）**:
- バルクインポート・エクスポート
- タグ統計と分析
- Auto-tagging結果とのマッピング
- 非推奨タグの自動置換

---

### 2.6 Auto-Tagging（タガー推論）

#### 2.6.1 使用するタガーモデル

**cl_tagger** - 既存実装を活用

**モデル情報**:
- リポジトリ: `cella110n/cl_tagger`
- 現在のバージョン:
  - v1.00
  - v1.01
  - v1.02（最新、デフォルト）
- 形式: ONNX（高速推論）
- Hugging Face Hubから自動ダウンロード

**サポートするカテゴリ**:
- **Rating**: safe, sensitive, nsfw, etc.
- **General**: 一般タグ（1girl, long hair, smile, etc.）
- **Artist**: アーティスト名
- **Character**: キャラクター名
- **Copyright**: 作品名
- **Meta**: メタ情報
- **Quality**: 品質タグ（masterpiece, best quality, etc.）
- **Model**: モデル名

**既存実装の場所**:
- バックエンド: `backend/core/tagger_manager.py`
- フロントエンド: `frontend/src/components/common/ImageTaggerPanel.tsx`
- API: `backend/api/routes.py` (`/tag-image`, `/tag-batch` エンドポイント）

#### 2.6.2 Auto-Tagging UI（データセット用）

**UI要素**: `/dataset/[dataset_id]` ページの "Auto-Tag" ボタン

**処理フロー**:
1. ユーザーが画像を選択（単一または複数）
2. "Auto-Tag" ボタンをクリック
3. モーダルダイアログ表示:
   - **モデルバージョン選択**: v1.00, v1.01, v1.02（最新）
   - **信頼度しきい値**: 0.0-1.0（デフォルト: 0.35）
   - **カテゴリフィルター**:
     - [ ] General
     - [ ] Character
     - [ ] Copyright
     - [ ] Artist
     - [ ] Quality
     - [ ] Rating
     - [ ] Meta
     - [ ] Model
   - **既存タグの処理**:
     - Replace（置き換え）
     - Add（追加）
     - Skip（スキップ）
   - **キャプションタイプ**: main, alt, etc.
   - **タグ辞書マッピング**: ON/OFF（既知のタグに変換）
4. バックエンドに推論リクエスト送信
5. WebSocket経由で進捗表示
6. 完了後、タグが自動追加される

**既存UIとの統合**:
- 既存の `ImageTaggerPanel` コンポーネントを再利用
- データセット用にラップして、結果をデータセットDBに保存

**バックエンド実装**:

```python
# backend/api/routes.py

@router.post("/dataset/{dataset_id}/auto-tag")
async def auto_tag_dataset_items(
    dataset_id: int,
    item_ids: List[int] = Body(...),
    model_version: str = Body("cl_tagger_1_02"),
    threshold: float = Body(0.35),
    category_filters: List[str] = Body(["general", "character", "quality"]),
    mode: str = Body("add"),  # "replace", "add", "skip"
    caption_type: str = Body("main"),
    apply_tag_dictionary_mapping: bool = Body(True),
    db: Session = Depends(get_db)
):
    """
    Auto-tag dataset items using cl_tagger

    Args:
        dataset_id: Dataset ID
        item_ids: List of item IDs to tag
        model_version: cl_tagger version (cl_tagger_1_00, cl_tagger_1_01, cl_tagger_1_02)
        threshold: Confidence threshold (0.0-1.0)
        category_filters: Categories to include (general, character, copyright, artist, quality, rating, meta, model)
        mode: How to handle existing tags (replace, add, skip)
        caption_type: Caption type to save tags to (main, alt, etc.)
        apply_tag_dictionary_mapping: Apply tag dictionary mapping (aliases, deprecated tags)
    """
    from core.tagger_manager import TaggerManager

    # Load cl_tagger
    tagger = TaggerManager()
    tagger.load_model(
        use_huggingface=True,
        repo_id="cella110n/cl_tagger",
        model_version=model_version
    )

    results = []

    for item_id in item_ids:
        item = db.query(DatasetItem).filter_by(id=item_id).first()
        if not item:
            continue

        # Run inference
        predictions = tagger.predict(
            image_path=item.image_path,
            threshold=threshold
        )

        # Filter by category
        filtered_tags = []
        for tag, confidence, category in predictions:
            if category.lower() in category_filters:
                filtered_tags.append((tag, confidence, category))

        # Apply tag dictionary mapping
        if apply_tag_dictionary_mapping:
            filtered_tags = map_tagger_output_to_dictionary(
                [tag for tag, _, _ in filtered_tags],
                db
            )

        # Save to database
        save_tags_to_dataset_item(
            item_id=item_id,
            tags=filtered_tags,
            caption_type=caption_type,
            mode=mode,
            source=f"auto_{model_version}",
            db=db
        )

        results.append({
            "item_id": item_id,
            "tags_added": len(filtered_tags),
            "tags": [tag for tag, _, _ in filtered_tags]
        })

    return {
        "success": True,
        "results": results,
        "total_items": len(results)
    }
```

**既存の`TaggerManager`の活用**:

```python
# backend/core/tagger_manager.py (既存コード)

class TaggerManager:
    def predict(
        self,
        image_path: str = None,
        image: Image.Image = None,
        threshold: float = 0.35,
        excluded_tags: List[str] = None
    ) -> List[Tuple[str, float, str]]:
        """
        Predict tags for an image

        Returns:
            List of (tag, confidence, category) tuples
        """
        # 既存の推論ロジック
        # ...

        # Returns format: [("1girl", 0.95, "general"), ("example_character", 0.87, "character"), ...]
```

#### 2.6.3 バッチ処理とキャッシュ

**バッチ処理**:
- 大量画像のタグ付けは非同期処理（Celery or Background Tasks）
- WebSocket経由で進捗通知

**キャッシュ**:
- 画像ハッシュ別にタグ推論結果をキャッシュ
- 同じ画像の再推論を避ける

---

### 2.7 データセットのエクスポート

#### 2.7.1 エクスポート形式

**サポートする形式**:

1. **テキストファイル（.txt）** - デフォルト
   - 各画像に対応する `.txt` ファイルを生成
   - タグをカンマ区切りで出力

2. **JSON形式**
   - データセット全体のメタデータを含む
   - ai-toolkitの学習設定ファイルとして使用可能

3. **ai-toolkit形式**
   - ai-toolkitの `config/dataset.yaml` フォーマット

**エクスポート設定**:
- キャプションタイプ選択（main, alt, etc.）
- タググループフィルター（Qualityタグのみ除外、など）
- エイリアス適用: ON/OFF

**UI要素**: `/dataset/[dataset_id]` ページの "Export" ボタン

---

### 2.8 トレーニングトラッキングとバランスドサンプリング

**目的**: 継続学習（resume）時に、特定の画像の学習回数に偏りが出ないようにする。

#### 2.8.1 トレーニングRunの追跡

**新規テーブル**: `training_runs`, `training_item_usage`, `training_item_stats`（上記スキーマ参照）

**トラッキングフロー**:

```python
# 1. トレーニング開始時
training_run = TrainingRun(
    dataset_id=123,
    run_name="lora_character_v1_run_003",
    run_number=3,
    parent_run_id=previous_run.id,  # resume元
    model_type="lora",
    base_model="animagine-xl-3.1",
    caption_type="tags",
    total_epochs=10,
    status="running"
)

# 2. 各バッチ処理後にコールバック
def on_batch_end(batch_indices, current_epoch, current_step):
    """各バッチ処理後に呼ばれるコールバック"""
    for item_idx in batch_indices:
        # TrainingItemUsageテーブルを更新
        usage = db.query(TrainingItemUsage).filter_by(
            run_id=training_run.id,
            item_id=item_idx
        ).first()

        if not usage:
            usage = TrainingItemUsage(
                run_id=training_run.id,
                item_id=item_idx,
                times_seen=0,
                first_seen_epoch=current_epoch,
                first_seen_step=current_step
            )
            db.add(usage)

        usage.times_seen += 1
        usage.last_seen_epoch = current_epoch
        usage.last_seen_step = current_step

        # TrainingItemStatsテーブルも更新
        update_item_stats(item_idx)

    db.commit()

# 3. エポック終了時に統計更新
def on_epoch_end(current_epoch):
    training_run.completed_epochs = current_epoch
    training_run.total_samples_seen = sum_all_times_seen(training_run.id)
    db.commit()
```

#### 2.8.2 バランスドサンプリング戦略

**問題**: 通常のシャッフルでは、resume時に一部の画像が過学習される可能性

**解決策**: 過去のrun履歴を考慮したバランスドサンプリング

**アルゴリズム1: 最小使用回数優先（Min-Usage-First）**

```python
# backend/training/balanced_sampler.py

class BalancedSampler:
    """過去の学習回数を考慮したサンプラー"""

    def __init__(self, dataset_id: int, current_run_id: int, db: Session):
        self.dataset_id = dataset_id
        self.current_run_id = current_run_id
        self.db = db

        # 全アイテムの累計学習回数を取得
        self.item_stats = self._load_item_stats()

    def _load_item_stats(self) -> Dict[int, int]:
        """各アイテムの累計学習回数を取得"""
        stats = {}
        items = self.db.query(DatasetItem).filter_by(
            dataset_id=self.dataset_id
        ).all()

        for item in items:
            item_stat = self.db.query(TrainingItemStats).filter_by(
                item_id=item.id
            ).first()

            if item_stat:
                stats[item.id] = item_stat.total_times_seen
            else:
                stats[item.id] = 0  # 未使用

        return stats

    def get_weighted_sample_probabilities(self) -> np.ndarray:
        """各アイテムのサンプリング確率を計算"""
        item_ids = list(self.item_stats.keys())
        usage_counts = np.array([self.item_stats[id] for id in item_ids])

        # 逆数で重み付け（使用回数が少ないほど高確率）
        # +1 は 0回使用のアイテムでゼロ除算を防ぐため
        weights = 1.0 / (usage_counts + 1.0)

        # 正規化
        probabilities = weights / weights.sum()

        return probabilities

    def sample_epoch_indices(self, num_samples: int) -> List[int]:
        """1エポック分のインデックスをバランスドサンプリング"""
        item_ids = list(self.item_stats.keys())
        probabilities = self.get_weighted_sample_probabilities()

        # 重み付けサンプリング
        sampled_indices = np.random.choice(
            item_ids,
            size=num_samples,
            replace=True,  # 1エポックで複数回使用可能
            p=probabilities
        )

        return sampled_indices.tolist()
```

**アルゴリズム2: 階層化サンプリング（Stratified Sampling）**

```python
class StratifiedBalancedSampler:
    """使用回数でグループ化してバランスドサンプリング"""

    def stratify_items_by_usage(self) -> Dict[str, List[int]]:
        """使用回数でアイテムを階層化"""
        strata = {
            "never_used": [],      # 0回
            "low_usage": [],       # 1-10回
            "medium_usage": [],    # 11-50回
            "high_usage": []       # 51回以上
        }

        for item_id, usage_count in self.item_stats.items():
            if usage_count == 0:
                strata["never_used"].append(item_id)
            elif usage_count <= 10:
                strata["low_usage"].append(item_id)
            elif usage_count <= 50:
                strata["medium_usage"].append(item_id)
            else:
                strata["high_usage"].append(item_id)

        return strata

    def sample_epoch_indices(self, num_samples: int) -> List[int]:
        """各層から均等にサンプリング"""
        strata = self.stratify_items_by_usage()

        # 各層からのサンプル数を決定
        # 例: never_used から 40%, low_usage から 30%, medium から 20%, high から 10%
        samples_per_stratum = {
            "never_used": int(num_samples * 0.4),
            "low_usage": int(num_samples * 0.3),
            "medium_usage": int(num_samples * 0.2),
            "high_usage": int(num_samples * 0.1)
        }

        sampled_indices = []
        for stratum_name, target_count in samples_per_stratum.items():
            stratum_items = strata[stratum_name]

            if len(stratum_items) == 0:
                continue

            # この層からサンプリング
            sampled = np.random.choice(
                stratum_items,
                size=min(target_count, len(stratum_items)),
                replace=(target_count > len(stratum_items))
            )
            sampled_indices.extend(sampled)

        # シャッフル
        np.random.shuffle(sampled_indices)

        return sampled_indices.tolist()
```

**アルゴリズム3: 標準偏差最小化（Variance Minimization）**

```python
class VarianceMinimizingSampler:
    """学習回数の分散を最小化するサンプラー"""

    def calculate_usage_variance(self) -> float:
        """現在の使用回数の分散を計算"""
        usage_counts = np.array(list(self.item_stats.values()))
        return np.var(usage_counts)

    def sample_epoch_indices(self, num_samples: int) -> List[int]:
        """分散を最小化するようにサンプリング"""
        item_ids = list(self.item_stats.keys())
        usage_counts = np.array([self.item_stats[id] for id in item_ids])

        # 目標: 全アイテムの使用回数を均等にする
        target_usage = usage_counts.mean()

        # 目標使用回数との差が大きいほど高確率
        usage_deficit = np.maximum(target_usage - usage_counts, 0)
        weights = usage_deficit + 0.1  # 最低限のランダム性

        probabilities = weights / weights.sum()

        sampled_indices = np.random.choice(
            item_ids,
            size=num_samples,
            replace=True,
            p=probabilities
        )

        return sampled_indices.tolist()
```

#### 2.8.3 トレーニング設定UIでのサンプリング戦略選択

**UI要素**: `/training/new` ページ

**サンプリング戦略の選択**:
```tsx
<Select
  label="Sampling Strategy"
  value={params.sampling_strategy}
  options={[
    { value: "random", label: "Random Shuffle (Default)" },
    { value: "balanced_min_usage", label: "Balanced: Min-Usage-First" },
    { value: "balanced_stratified", label: "Balanced: Stratified Sampling" },
    { value: "balanced_variance_min", label: "Balanced: Variance Minimization" },
  ]}
/>

{params.sampling_strategy !== "random" && (
  <Checkbox
    label="Consider all previous runs (not just parent run)"
    checked={params.consider_all_runs}
  />
)}
```

**トレーニング開始時の設定**:
```python
# backend/api/routes.py

@router.post("/training/jobs")
async def create_training_job(job: TrainingJobRequest, db: Session):
    # バランスドサンプラーの初期化
    if job.sampling_strategy != "random":
        sampler = create_sampler(
            strategy=job.sampling_strategy,
            dataset_id=job.dataset_id,
            current_run_id=training_run.id,
            db=db
        )

        # サンプリングインデックスを事前生成
        num_samples = len(dataset_items) * job.epochs
        sampled_indices = sampler.sample_epoch_indices(num_samples)

        # メタデータに含める
        metadata["sampled_indices"] = sampled_indices
        metadata["sampling_strategy"] = job.sampling_strategy

    # メタデータエクスポート
    export_training_metadata(training_run.id, metadata)
```

#### 2.8.4 トレーニング統計の可視化

**UI要素**: `/training/runs/{run_id}` ページ

**表示項目**:

1. **Run概要**
   - Run名、ステータス、開始日時、完了日時
   - 完了エポック数 / 総エポック数
   - 使用された総サンプル数

2. **データセット使用状況**
   - ヒートマップ: 各画像の使用回数
   - ヒストグラム: 使用回数の分布
   - 統計: 平均、中央値、標準偏差、最小、最大

3. **使用回数の偏り検出**
   - 分散、標準偏差
   - 最も使用された画像 Top 10
   - 未使用画像のリスト

4. **Resume推奨アルゴリズム**
   - 現在の偏り度合いに基づいて推奨サンプリング戦略を表示
   - 例: "標準偏差が高いため、Variance Minimization を推奨"

**可視化例**:
```tsx
// frontend/src/components/training/TrainingRunStats.tsx

<div className="grid grid-cols-2 gap-4">
  {/* Usage Heatmap */}
  <div>
    <h3>Item Usage Heatmap</h3>
    <HeatmapChart
      data={itemUsageData}
      colorScale={["#E0F7FA", "#006064"]}
      tooltip={(d) => `Item ${d.id}: ${d.times_seen} times`}
    />
  </div>

  {/* Usage Distribution */}
  <div>
    <h3>Usage Distribution</h3>
    <HistogramChart
      data={usageDistribution}
      xLabel="Times Seen"
      yLabel="Number of Items"
    />
  </div>

  {/* Statistics */}
  <div className="col-span-2">
    <h3>Statistics</h3>
    <table>
      <tr>
        <td>Mean Usage:</td>
        <td>{stats.mean.toFixed(2)}</td>
      </tr>
      <tr>
        <td>Std Dev:</td>
        <td>{stats.std_dev.toFixed(2)}</td>
      </tr>
      <tr>
        <td>Variance:</td>
        <td>{stats.variance.toFixed(2)}</td>
      </tr>
      <tr>
        <td>Min Usage:</td>
        <td>{stats.min}</td>
      </tr>
      <tr>
        <td>Max Usage:</td>
        <td>{stats.max}</td>
      </tr>
    </table>
  </div>

  {/* Recommendation */}
  <div className="col-span-2">
    <Alert status="info">
      <AlertIcon />
      <AlertTitle>Recommendation for Resume</AlertTitle>
      <AlertDescription>
        Standard deviation is {stats.std_dev.toFixed(2)}.
        Consider using <strong>Variance Minimization</strong> sampling
        strategy to reduce bias in the next run.
      </AlertDescription>
    </Alert>
  </div>
</div>
```

#### 2.8.5 API エンドポイント

**トレーニングRun管理**:
```yaml
# openapi.yaml

/training/runs:
  get:
    summary: List all training runs
    parameters:
      - name: dataset_id
        in: query
        schema:
          type: integer
    responses:
      '200':
        description: Training runs list

  post:
    summary: Create new training run
    requestBody:
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/CreateTrainingRunRequest'
    responses:
      '201':
        description: Training run created

/training/runs/{run_id}:
  get:
    summary: Get training run details
    responses:
      '200':
        description: Training run details
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/TrainingRun'

/training/runs/{run_id}/stats:
  get:
    summary: Get training run statistics
    responses:
      '200':
        description: Usage statistics
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/TrainingRunStats'

/training/runs/{run_id}/item-usage:
  get:
    summary: Get item usage details for a run
    responses:
      '200':
        description: Item usage list

  post:
    summary: Update item usage (called during training)
    requestBody:
      content:
        application/json:
          schema:
            type: object
            properties:
              batch_indices:
                type: array
                items:
                  type: integer
              current_epoch:
                type: integer
              current_step:
                type: integer
```

#### 2.8.6 実装の優先度

**Phase 1（初期実装）**:
- `TrainingRun`, `TrainingItemUsage`, `TrainingItemStats` テーブル作成
- 基本的なトラッキング（times_seen のカウント）
- ランダムシャッフル（デフォルト）

**Phase 2（バランスドサンプリング実装）**:
- Min-Usage-First サンプラー実装
- トレーニング設定UIでのサンプリング戦略選択
- 基本的な統計表示

**Phase 3（高度な機能）**:
- Stratified Sampling, Variance Minimization サンプラー
- ヒートマップ・ヒストグラムでの可視化
- Resume推奨アルゴリズム

---

## 3. 非機能要件

### 3.1 パフォーマンス

**大規模データセット対応**:
- 10,000枚以上の画像でも快適に動作
- ページネーション（1ページ50-100枚）
- 遅延読み込み（Lazy loading）

**データベース最適化**:
- インデックス作成（tag, dataset_id, item_id）
- クエリの最適化

**データローダーの検討**:
- メモ: "データローダーをTSに持たせるかも？(ブラウザ切ったら死ぬので別途バックに別の高速でSQLを扱える言語を立てるかどうか)"
- 初期実装: FastAPI + SQLAlchemy (Python)
- 将来的な拡張: 必要に応じて高速データベース（PostgreSQL, Redis）を検討

### 3.2 データ整合性

**ファイルとDBの同期**:
- タグ変更時に `.txt` ファイルとDBを同時更新
- ファイル削除時のエラーハンドリング

**トランザクション管理**:
- 複数タグの一括更新時にトランザクション使用

### 3.3 ユーザビリティ

**レスポンシブデザイン**:
- モバイル対応（タブレット以上推奨）
- タッチ操作対応

**エラーハンドリング**:
- ファイルアクセスエラー時の適切なメッセージ表示
- 存在しない画像ファイルの検出

---

## 4. 技術スタック

### 4.1 フロントエンド

**フレームワーク**:
- Next.js 14 (App Router)
- React 18
- TypeScript

**UI コンポーネント**:
- 既存のコンポーネントを流用（Select, Input, Button, etc.）
- react-beautiful-dnd（ドラッグ&ドロップ）
- react-virtualized（大規模リスト表示）

**状態管理**:
- React hooks (useState, useEffect, useContext)
- 必要に応じて Zustand または Context API

### 4.2 バックエンド

**フレームワーク**:
- FastAPI
- SQLAlchemy
- SQLite（初期実装）

**新規ライブラリ**:
- `transformers`（Auto-tagging用）
- `Pillow`（画像処理）
- `watchdog`（ファイル監視、オプション）

**Python環境**:
- **必須**: `<PRIVATE_PATH>` を使用（CLAUDE.mdルール）

### 4.3 データベース

**初期実装**: SQLite（既存のwebui.dbに追加）

**将来的な拡張**:
- PostgreSQL（大規模データセット向け）
- Redis（キャッシュ）

---

## 5. API設計（OpenAPI優先）

**重要**: すべてのAPIエンドポイントは `openapi.yaml` に先に定義すること（CLAUDE.mdルール）

### 5.1 データセット管理エンドポイント

```yaml
# openapi.yaml に追加

paths:
  /datasets:
    get:
      tags: [datasets]
      summary: List all datasets
      responses:
        '200':
          description: Dataset list
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/DatasetSummary'

    post:
      tags: [datasets]
      summary: Create new dataset
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateDatasetRequest'
      responses:
        '201':
          description: Dataset created
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Dataset'

  /datasets/{dataset_id}:
    get:
      tags: [datasets]
      summary: Get dataset details
      parameters:
        - name: dataset_id
          in: path
          required: true
          schema:
            type: integer
      responses:
        '200':
          description: Dataset details
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Dataset'

    delete:
      tags: [datasets]
      summary: Delete dataset
      responses:
        '204':
          description: Dataset deleted

  /datasets/{dataset_id}/items:
    get:
      tags: [datasets]
      summary: List dataset items (paginated)
      parameters:
        - name: dataset_id
          in: path
          required: true
          schema:
            type: integer
        - name: page
          in: query
          schema:
            type: integer
            default: 1
        - name: limit
          in: query
          schema:
            type: integer
            default: 50
        - name: search
          in: query
          schema:
            type: string
        - name: tag_filter
          in: query
          schema:
            type: string
      responses:
        '200':
          description: Dataset items
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/DatasetItemsResponse'

  /datasets/{dataset_id}/items/{item_id}/tags:
    get:
      tags: [datasets]
      summary: Get tags for item
      responses:
        '200':
          description: Tag list

    post:
      tags: [datasets]
      summary: Add tag to item
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/AddTagRequest'
      responses:
        '201':
          description: Tag added

    delete:
      tags: [datasets]
      summary: Remove tag from item
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/RemoveTagRequest'
      responses:
        '204':
          description: Tag removed

  /datasets/{dataset_id}/auto-tag:
    post:
      tags: [datasets]
      summary: Auto-tag dataset items
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/AutoTagRequest'
      responses:
        '202':
          description: Auto-tagging started (async)

  /datasets/{dataset_id}/export:
    post:
      tags: [datasets]
      summary: Export dataset
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ExportDatasetRequest'
      responses:
        '200':
          description: Export successful
          content:
            application/zip:
              schema:
                type: string
                format: binary

  /datasets/{dataset_id}/reindex:
    post:
      tags: [datasets]
      summary: Reindex dataset
      responses:
        '202':
          description: Reindexing started (async)

  /tag-groups:
    get:
      tags: [datasets]
      summary: List tag groups
      responses:
        '200':
          description: Tag group list

    post:
      tags: [datasets]
      summary: Create tag group
      responses:
        '201':
          description: Tag group created

  /tag-aliases:
    get:
      tags: [datasets]
      summary: List tag aliases
      responses:
        '200':
          description: Tag alias list

    post:
      tags: [datasets]
      summary: Create tag alias
      responses:
        '201':
          description: Tag alias created
```

### 5.2 スキーマ定義

```yaml
# openapi.yaml の components/schemas に追加

components:
  schemas:
    Dataset:
      type: object
      required: [id, name, path]
      properties:
        id:
          type: integer
          example: 1
        name:
          type: string
          example: "my_character_dataset_v1"
        path:
          type: string
          example: "<PRIVATE_PATH>"
        caption_suffixes:
          type: array
          items:
            type: string
          example: ["main", "alt"]
        recursive:
          type: boolean
          default: true
        max_depth:
          type: integer
          default: 3
        total_images:
          type: integer
          example: 150
        total_captions:
          type: integer
          example: 150
        created_at:
          type: string
          format: date-time
        updated_at:
          type: string
          format: date-time
        indexed_at:
          type: string
          format: date-time

    DatasetSummary:
      type: object
      properties:
        id:
          type: integer
        name:
          type: string
        total_images:
          type: integer
        updated_at:
          type: string
          format: date-time

    CreateDatasetRequest:
      type: object
      required: [name, path]
      properties:
        name:
          type: string
          minLength: 1
          maxLength: 100
        path:
          type: string
        caption_suffixes:
          type: array
          items:
            type: string
          default: []
        recursive:
          type: boolean
          default: true
        max_depth:
          type: integer
          minimum: 1
          maximum: 10
          default: 3

    DatasetItem:
      type: object
      properties:
        id:
          type: integer
        dataset_id:
          type: integer
        image_path:
          type: string
        relative_path:
          type: string
        width:
          type: integer
        height:
          type: integer
        file_size:
          type: integer
        image_hash:
          type: string
        caption_paths:
          type: object
          additionalProperties:
            type: string
        tags:
          type: object
          description: "Tags by caption type"
          additionalProperties:
            type: array
            items:
              $ref: '#/components/schemas/Tag'
        created_at:
          type: string
          format: date-time

    Tag:
      type: object
      properties:
        id:
          type: integer
        tag:
          type: string
        position:
          type: integer
        tag_group:
          type: string
          nullable: true
        confidence:
          type: number
          format: float
          nullable: true
        source:
          type: string
          enum: [manual, auto_wd14, auto_joytag]

    DatasetItemsResponse:
      type: object
      properties:
        items:
          type: array
          items:
            $ref: '#/components/schemas/DatasetItem'
        total:
          type: integer
        page:
          type: integer
        limit:
          type: integer
        total_pages:
          type: integer

    AddTagRequest:
      type: object
      required: [tag, caption_type]
      properties:
        tag:
          type: string
        caption_type:
          type: string
          example: "main"
        position:
          type: integer
          nullable: true
        tag_group:
          type: string
          nullable: true

    RemoveTagRequest:
      type: object
      required: [tag_id]
      properties:
        tag_id:
          type: integer

    AutoTagRequest:
      type: object
      required: [item_ids]
      properties:
        item_ids:
          type: array
          items:
            type: integer
        model_name:
          type: string
          enum: [wd14, joytag]
          default: "wd14"
        threshold:
          type: number
          format: float
          minimum: 0.0
          maximum: 1.0
          default: 0.35
        mode:
          type: string
          enum: [replace, add, skip]
          default: "add"
        caption_type:
          type: string
          default: "main"

    ExportDatasetRequest:
      type: object
      properties:
        format:
          type: string
          enum: [txt, json, ai-toolkit]
          default: "txt"
        caption_type:
          type: string
          default: "main"
        apply_aliases:
          type: boolean
          default: true
        exclude_tag_groups:
          type: array
          items:
            type: string
          example: ["Meta"]

    TagGroup:
      type: object
      properties:
        id:
          type: integer
        name:
          type: string
        color:
          type: string
          nullable: true
        tags:
          type: array
          items:
            type: string

    TagAlias:
      type: object
      properties:
        id:
          type: integer
        source_tag:
          type: string
        target_tag:
          type: string
```

---

## 6. UI/UXデザイン

### 6.1 ページ構成

**新規ページ**:
1. `/dataset` - データセット一覧
2. `/dataset/new` - 新規データセット作成
3. `/dataset/[dataset_id]` - データセットアイテム一覧
4. `/dataset/[dataset_id]/quick-tag` - 高速タグ付けUI
5. `/dataset/tags` - タググループ管理
6. `/dataset/tags/aliases` - タグエイリアス管理

**既存ページとの統合**:
- Sidebarに "Datasets" リンク追加

### 6.2 カラースキーム

**タググループの色**（例）:
- Character: `#FF6B9D` (ピンク)
- Copyright: `#4ECDC4` (シアン)
- Artist: `#FFE66D` (イエロー)
- General: `#95E1D3` (ミント)
- Quality: `#A8E6CF` (ライトグリーン)
- Rating: `#FF6F69` (レッド)
- Meta: `#C7CEEA` (ラベンダー)

### 6.3 アイコン

**推奨**: Lucide React（既存UIで使用済み）

- Dataset: `FolderOpen`
- Tag: `Tag`
- Auto-tag: `Sparkles`
- Export: `Download`
- Reindex: `RefreshCw`

---

## 7. 実装フェーズ

### Phase 1: 基盤構築（Week 1-2）

**目標**: データベースとAPI基盤の構築

- [ ] OpenAPI仕様書に全エンドポイントを定義
- [ ] データベーステーブル作成（SQLAlchemy models）
- [ ] マイグレーション実行
- [ ] 基本的なCRUD APIの実装（datasets, dataset_items）
- [ ] テスト用データセットの準備

### Phase 2: UI基本機能（Week 3-4）

**目標**: データセット登録とアイテム表示

- [ ] `/dataset` ページ（一覧表示）
- [ ] `/dataset/new` ページ（新規作成）
- [ ] `/dataset/[dataset_id]` ページ（グリッド表示）
- [ ] 画像詳細モーダル
- [ ] タグ追加/削除機能

### Phase 3: タグ管理（Week 5-6）

**目標**: タググループとエイリアス

- [ ] タググループ管理UI
- [ ] タグエイリアス管理UI
- [ ] タグのオートコンプリート
- [ ] タグ検索・フィルター機能

### Phase 4: Auto-Tagging（Week 7-8）

**目標**: WD14 Taggerの統合

- [ ] WD14 Taggerのバックエンド実装
- [ ] Auto-tag UI（モーダル）
- [ ] バッチ処理と進捗表示
- [ ] 推論結果のキャッシュ

### Phase 5: 高度な機能（Week 9-10）

**目標**: 高速タグ付けとエクスポート

- [ ] 高速タグ付けUI
- [ ] キーボードショートカット
- [ ] エクスポート機能（txt, JSON）
- [ ] 再インデックス機能

### Phase 6: パフォーマンス最適化（Week 11-12）

**目標**: 大規模データセット対応

- [ ] ページネーション最適化
- [ ] Lazy loading実装
- [ ] データベースクエリ最適化
- [ ] フロントエンドのバンドルサイズ削減

---

## 8. テスト計画

### 8.1 ユニットテスト

**バックエンド**:
- データベースモデルのテスト
- API エンドポイントのテスト（FastAPI TestClient）
- Auto-tagging ロジックのテスト

**フロントエンド**:
- コンポーネントのテスト（React Testing Library）
- API client のテスト

### 8.2 統合テスト

- エンドツーエンドのフロー（データセット作成 → タグ付け → エクスポート）
- 大規模データセット（10,000枚）での動作確認

### 8.3 手動テスト

- UI/UXの確認
- エラーハンドリングの確認
- モバイルデバイスでの表示確認

---

## 9. リスクと対策

### リスク1: パフォーマンス問題

**リスク**: 大規模データセット（10,000枚以上）で動作が遅い

**対策**:
- 初期段階でページネーション実装
- データベースインデックスの最適化
- 必要に応じてPostgreSQLへの移行

### リスク2: ファイルシステムとDBの同期ずれ

**リスク**: `.txt` ファイルの手動編集でDBと不一致

**対策**:
- 再インデックス機能の実装
- ファイル監視（watchdog）でリアルタイム同期（オプション）

### リスク3: Auto-tagging の精度

**リスク**: タガーモデルの推論精度が低い

**対策**:
- 信頼度しきい値の調整
- 複数モデルのサポート（WD14, JoyTag）
- 手動レビュー機能

### リスク4: データローダーの選択

**リスク**: TSでデータローダーを実装した場合、ブラウザクローズで状態が失われる

**対策**:
- 初期実装はバックエンド（FastAPI + SQLAlchemy）で実装
- 将来的に高速化が必要な場合、別言語（Rust, Go）でマイクロサービス化を検討

---

## 10. 将来的な拡張

### 10.1 モデルトレーニング機能

**目標**: データセットを使った実際のLoRA/モデル学習

**実装方針**:
- ai-toolkitの学習パイプラインを統合
- UI上でトレーニング設定（epochs, lr, etc.）
- WebSocket経由でトレーニング進捗表示

### 10.2 データセット拡張（Augmentation）

**目標**: データセットの自動拡張

**機能**:
- 反転、回転、クロッピング
- 色調整、ノイズ追加
- albumentations ライブラリの活用

### 10.3 マルチモーダル対応

**目標**: 動画データセットのサポート

**機能**:
- 動画ファイルの読み込み（.mp4, .avi, etc.）
- フレーム抽出
- キーフレームの自動選択

### 10.4 コラボレーション機能

**目標**: 複数ユーザーでのデータセット共有

**機能**:
- ユーザー管理（既存の認証システムを拡張）
- データセットの権限管理（読み取り専用、編集可能）
- タグ変更履歴の表示

---

## 11. 参考資料

### ai-toolkit 参考ファイル

**データローダー**:
- `<PRIVATE_PATH>`
- `<PRIVATE_PATH>`

**メタデータ管理**:
- `<PRIVATE_PATH>`

**タググループ**:
- `<PRIVATE_PATH>`

**データセットツール**:
- `<PRIVATE_PATH>`

### 外部ライブラリ

**Auto-tagging**:
- [WD14 Tagger](https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3)
- [JoyTag](https://huggingface.co/fancyfeast/joytag)

**UI コンポーネント**:
- [react-beautiful-dnd](https://github.com/atlassian/react-beautiful-dnd)
- [react-virtualized](https://github.com/bvaughn/react-virtualized)

---

## 12. まとめ

本要件定義書は、SushiUI WebUIにデータセット管理機能を追加するための包括的なドキュメントです。

**重要なポイント**:
1. **OpenAPI駆動開発**: すべてのAPI変更は `openapi.yaml` を経由
2. **段階的な実装**: 6つのフェーズで12週間を想定
3. **ai-toolkit参考**: 既存のベストプラクティスを活用
4. **パフォーマンス重視**: 大規模データセット対応を前提
5. **将来的な拡張性**: トレーニング機能への展開を考慮

**次のステップ**:
- 本要件定義書のレビューとフィードバック
- Phase 1（基盤構築）の開始準備
- OpenAPI仕様書の詳細設計

---

## 13. 変更履歴

### Version 1.3.0 (2025-11-29)

**主な変更**:

1. **タグ辞書管理機能の追加**
   - 新規テーブル: `TagDictionary` - 70万件以上のDanbooruタグを管理
   - カテゴリ: Character, Artist, Copyright, General, Meta, Model
   - タグの追加・編集・削除機能
   - カウント数の管理（Danbooru出現回数またはユーザー定義値）

2. **タグ辞書編集UI**
   - タグ一覧表示（検索、フィルター、ソート）
   - タグ追加・編集モーダル
   - バリデーション（タグ名、カテゴリ、カウント）
   - タグ削除（データセットから一括削除オプション）

3. **バルクインポート・エクスポート**
   - JSON形式（既存taglist形式）
   - CSV形式（詳細メタデータ付き）
   - 既存タグの競合解決（Skip, Update, Merge）
   - カテゴリ別エクスポート

4. **タグ辞書の活用**
   - オートコンプリート（人気順、カテゴリ別）
   - Auto-tagging結果のマッピング
   - 非推奨タグの自動置換
   - 未知タグの自動検出・追加

5. **タグ統計と分析**
   - カテゴリ別統計（タグ数、使用頻度）
   - ソース別統計（公式/カスタム/自動検出）
   - 使用頻度分析（データセット内で実際に使用中のタグ）

6. **Auto-Tagging実装の更新**
   - 既存の `cl_tagger` 実装を使用（WD14/JoyTagから変更）
   - モデル: `cella110n/cl_tagger` (v1.00, v1.01, v1.02)
   - ONNX形式、8カテゴリ対応（rating, general, artist, character, copyright, meta, quality, model）
   - 既存ファイル参照: `backend/core/tagger_manager.py`, `frontend/src/components/common/ImageTaggerPanel.tsx`
   - タグ辞書マッピング機能と統合

7. **API エンドポイント追加**
   - `/tag-dictionary` - CRUD操作
   - `/tag-dictionary/search` - オートコンプリート用
   - `/tag-dictionary/import` - バルクインポート
   - `/tag-dictionary/export` - バルクエクスポート
   - `/tag-dictionary/stats` - 統計情報

8. **初期データのロード**
   - 起動時に `taglist/*.json` からタグ辞書を自動ロード
   - バッチインサート（1000件ごと）で高速化
   - 既にロード済みの場合はスキップ

**実装の優先度**:
- Phase 1: テーブル作成、初期ロード、基本CRUD
- Phase 2: 編集UI、バリデーション、オートコンプリート
- Phase 3: バルクインポート・エクスポート、統計、Auto-taggingマッピング

### Version 1.2.0 (2025-11-29)

**主な変更**:

1. **トレーニングトラッキング機能の追加**
   - 新規テーブル: `TrainingRun`, `TrainingItemUsage`, `TrainingItemStats`
   - 各画像のトレーニング使用回数を記録
   - run毎の統計情報（完了エポック、総サンプル数など）
   - 親run追跡（resume時の継承）

2. **バランスドサンプリング戦略**
   - 3つのサンプリングアルゴリズム実装設計:
     - Min-Usage-First: 使用回数が少ない画像を優先
     - Stratified Sampling: 使用回数で階層化してバランスサンプリング
     - Variance Minimization: 使用回数の分散を最小化
   - UI上でサンプリング戦略を選択可能
   - 継続学習（resume）時の偏り防止

3. **トレーニング統計の可視化**
   - ヒートマップ: 各画像の使用回数
   - ヒストグラム: 使用回数の分布
   - 統計情報: 平均、標準偏差、分散など
   - Resume推奨アルゴリズム（偏り検出に基づく）

4. **API エンドポイント追加**
   - `/training/runs` - トレーニングrun管理
   - `/training/runs/{run_id}/stats` - 統計情報取得
   - `/training/runs/{run_id}/item-usage` - アイテム使用状況

5. **実装フェーズの明確化**
   - Phase 1: 基本トラッキング
   - Phase 2: バランスドサンプリング
   - Phase 3: 高度な可視化とアルゴリズム

**設計思想**:
- トレーニング開始前にメタデータをエクスポート（JSON/Parquet）
- 学習中はファイルベースでアクセス（DB依存なし）
- バッチ処理後にコールバックでDB更新
- WebSocket経由で進捗通知

### Version 1.1.0 (2025-11-29)

**主な変更**:

1. **データセット構造パターンの拡張**
   - 画像ペア（source/target/cref）のサポート追加
   - ファイル命名パターンの認識ロジック追加
   - サブディレクトリ構造を標準前提として明記

2. **複数キャプションタイプのサポート**
   - 新規テーブル: `DatasetCaption` - 複数種類のキャプション管理
   - キャプションタイプ: tags, natural_language, social_media, instruction, description
   - EXIF/XMPメタデータの読み込み対応
   - トレーニング時のキャプション選択機能

3. **データベーススキーマの改善**
   - `Dataset`: `image_suffixes`, `read_exif`, `max_depth` (nullable) 追加
   - `DatasetItem`: `item_type`, `base_name`, `group_id`, `image_suffix`, `related_images`, `exif_data` 追加
   - `DatasetCaption`: 新規テーブル追加
   - `DatasetTag`: `caption_id` フィールド追加

4. **UI/UX の詳細化**
   - 画像詳細モーダルを3カラムレイアウトに変更
   - 複数キャプションタイプの編集インターフェース
   - 画像ペアの表示・編集機能
   - EXIFメタデータの表示機能

5. **新規データセット登録の強化**
   - データセットタイプ選択（Single Image, Image Pairs, Auto Detect）
   - 画像suffixの設定
   - 検索深度の無制限オプション
   - EXIF読み込みオプション

**参考ケース追加**:
- `<PRIVATE_PATH>` - 画像ペアの実例
- `<PRIVATE_PATH>` - EXIFメタデータの実例

### Version 1.0.0 (2025-11-29)

**初版作成**:
- 基本的な要件定義
- データセット登録・管理機能
- タグ管理機能
- Auto-tagging機能
- 検索・フィルタリング機能
- OpenAPI仕様設計
- 実装フェーズ計画

---

**作成日**: 2025-11-29
**最終更新**: 2025-11-29
**バージョン**: 1.3.0
**作成者**: Claude Code
**レビュー**: 未実施
