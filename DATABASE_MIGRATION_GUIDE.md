# Database Migration Guide

## 概要

バージョン 2025-11-30 から、データベース構造が変更されました：

**旧構造**（単一DB）:
- `webui.db` または `sd_webui.db` - すべてのデータ

**新構造**（分離DB）:
- `gallery.db` - 生成画像とユーザー設定
- `datasets.db` - データセット関連
- `training.db` - トレーニング関連

## 移行が必要な場合

以下のいずれかに該当する場合、移行スクリプトを実行してください：

- ✅ 既存の `webui.db` または `sd_webui.db` がある
- ✅ 過去に生成した画像のメタデータを保持したい
- ✅ 既存のデータセットやトレーニング履歴を引き継ぎたい

## 移行手順

### 前提条件

1. **バックエンドサーバーを停止**
   ```bash
   # バックエンドが起動している場合は Ctrl+C で停止
   ```

2. **データベースのバックアップ作成**（推奨）
   ```bash
   # Windows
   copy webui.db webui.db.manual_backup

   # Linux/Mac
   cp webui.db webui.db.manual_backup
   ```

### 移行スクリプトの実行

```bash
cd backend
python migrate_to_separated_dbs.py
```

### スクリプトの動作

1. **自動検出**: `webui.db` または `sd_webui.db` を自動検出
2. **新DBスキーマ作成**: `gallery.db`, `datasets.db`, `training.db` を作成
3. **データコピー**: すべてのテーブルを適切なDBにコピー
4. **検証**: 各DBのテーブル数とレコード数を表示
5. **バックアップ**: 旧DBを `webui.db.backup_YYYYMMDD_HHMMSS` にリネーム

### 出力例

```
============================================================
DATABASE MIGRATION: Single DB → Separated DBs
============================================================

[Step 1/6] Detecting old database...
  [OK] Found old database: webui.db
  [INFO] File size: 12.45 MB

[Step 2/6] Checking new databases...
  [OK] No existing databases found

[Step 3/6] Creating new database schemas...
  [OK] Creating gallery.db schema...
  [OK] Creating datasets.db schema...
  [OK] Creating training.db schema...

[Step 4/6] Migrating data...

  Migrating gallery.db tables...
  [OK] Copied 5880 rows to generated_images
  [OK] Copied 1 rows to user_settings
  [INFO] Gallery total: 5881 rows

  Migrating datasets.db tables...
  [OK] Copied 3 rows to datasets
  [OK] Copied 1245 rows to dataset_items
  [INFO] Datasets total: 1248 rows

  Migrating training.db tables...
  [OK] Copied 2 rows to training_runs
  [OK] Copied 5 rows to training_checkpoints
  [INFO] Training total: 7 rows

[Step 5/6] Verifying migration...
  [gallery.db] Tables: 2
    - generated_images: 5880 rows
    - user_settings: 1 rows
  [datasets.db] Tables: 4
    - datasets: 3 rows
    - dataset_items: 1245 rows
  [training.db] Tables: 3
    - training_runs: 2 rows
    - training_checkpoints: 5 rows

[Step 6/6] Backing up old database...
  [OK] Old database backed up to: webui.db.backup_20251130_165430

============================================================
MIGRATION COMPLETED SUCCESSFULLY!
============================================================

Summary:
  Gallery rows:  5881
  Dataset rows:  1248
  Training rows: 7
  Total rows:    7136
```

## 移行後の確認

1. **バックエンド再起動**
   ```bash
   cd backend
   python main.py
   ```

2. **動作確認**
   - ギャラリーで過去の画像が表示されるか
   - データセットが読み込めるか
   - トレーニング履歴が表示されるか

3. **すべて正常なら旧DBを削除**
   ```bash
   # 確認後、旧DBを削除（任意）
   del webui.db.backup_20251130_165430
   ```

## トラブルシューティング

### 1. 旧DBが見つからない

**エラー**: `[ERROR] No old database found!`

**原因**: `webui.db` または `sd_webui.db` が存在しない

**解決策**:
- DBファイルが別の名前の場合、`webui.db` にリネーム
- 既に新DB構造の場合、移行不要

### 2. 新DBが既に存在

**警告**: `[WARN] gallery.db already exists`

**動作**: データを追記（重複はスキップ）

**解決策**:
- 続行: 既存データは保持され、新データが追加される
- クリーンインストール: 新DBを削除してから再実行

### 3. 重複エラー

**警告**: `[WARN] Skipped duplicate row: UNIQUE constraint failed`

**原因**: 同じデータが既に新DBに存在

**影響**: なし（重複は自動的にスキップされる）

### 4. バックアップ失敗

**エラー**: `[ERROR] Failed to backup old database`

**原因**: ディスク容量不足、権限不足

**解決策**:
- 手動でバックアップを作成
- ディスク容量を確保

## 新規インストールの場合

新規インストール（既存DBなし）の場合、移行は不要です。

バックエンド起動時に自動的に新DBが作成されます：

```bash
cd backend
python main.py
```

以下のログが表示されれば正常です：

```
[Database] Initializing gallery.db...
[Database] Initializing datasets.db...
[Database] Initializing training.db...
```

## 手動移行（上級者向け）

自動移行スクリプトが使えない場合の手動手順：

1. **新DB作成**
   ```bash
   cd backend
   python -c "from database import init_db; init_db()"
   ```

2. **データコピー**（SQLite CLI使用）
   ```sql
   -- gallery.db にコピー
   ATTACH DATABASE 'webui.db' AS old;
   INSERT INTO generated_images SELECT * FROM old.generated_images;
   INSERT INTO user_settings SELECT * FROM old.user_settings;
   DETACH DATABASE old;
   ```

3. **検証**
   ```bash
   python -c "import sqlite3; conn = sqlite3.connect('gallery.db'); print(conn.execute('SELECT COUNT(*) FROM generated_images').fetchone())"
   ```

## サポート

問題が解決しない場合は、以下の情報を含めて報告してください：

- エラーメッセージ全文
- `webui.db` のファイルサイズ
- 実行環境（Windows/Linux/Mac）
- Pythonバージョン

---

**重要**: 移行スクリプトは冪等性があるため、何度実行しても安全です（重複データは自動的にスキップされます）。
