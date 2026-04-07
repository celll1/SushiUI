# トレーニングパラメータ改修ガイド

このドキュメントはトレーニングパラメータ（training config / dataset config）を追加・変更・削除する際に**どのファイルのどこを編集すべきか**を示すリファレンスです。

## パラメータの2分類

トレーニングパラメータは構造的に2種類に分かれます。追加する前にどちらに属するか判断してください。

| 種類 | スコープ | YAML上の位置 | 例 |
|------|---------|-------------|-----|
| **Top-level（run-level）** | トレーニング実行全体で1つ | `config.process[0].train` などのトップレベル | `learning_rate`, `optimizer`, `batch_size`, `blocks_to_swap` |
| **Dataset-level** | 各データセットごとに個別 | `config.process[0].datasets[N]` 内 | `caption_types`, `ve_reconstruction_mode` |

---

## Case A: Dataset-level パラメータの追加

**編集箇所は3ヶ所のみ。** それ以外は`dataset_params.py`の一元管理により自動伝播します。

### 1. `backend/api/routes.py` — `DatasetConfigItem` Pydanticモデル

API契約の定義（バリデーションとOpenAPIスキーマ生成に必要）。

```python
class DatasetConfigItem(BaseModel):
    dataset_id: int
    caption_types: List[str] = []
    filters: Dict[str, Any] = {}
    ve_reconstruction_mode: Optional[bool] = False
    your_new_param: Optional[bool] = False  # ← 追加
```

### 2. `backend/core/training/dataset_params.py` — `DATASET_LEVEL_PARAMS`

ここに1行追加すれば、routes.py / training_config.py / train_runner.py の全伝播経路に自動反映されます。

```python
DATASET_LEVEL_PARAMS: Dict[str, Any] = {
    "caption_types": [],
    "ve_reconstruction_mode": False,
    "your_new_param": False,  # ← 追加（キー名とデフォルト値）
}
```

**注意**: デフォルト値はYAML書き込み時の省略判定に使われます。`extract_dataset_params()`はデフォルトと等しい値をYAMLから除外します。

### 3. 消費ポイント — `base_trainer.py` などの実装箇所

実際に該当パラメータを使うコードを追加します。この箇所はパラメータごとに意味が違うため自動化されていません。

現在の`ve_reconstruction_mode`は以下で消費されています（参考）:
- `backend/core/training/train_runner.py` L1010-1030: `item["_ve_reconstruction_mode"]` にフラグ注入
- `backend/core/training/base_trainer.py` L6900-6930: bucketing時の処理
- `backend/core/training/base_trainer.py` L8220付近: トレーニングステップでの挙動変更

### 4. フロントエンド（UIを出す場合）

- `frontend/src/components/training/TrainingConfig.tsx`
  - `DatasetConfig` インターフェース: フィールド追加
  - `datasetConfigs` mapping内のUI要素追加（チェックボックス/入力欄）
- `frontend/src/utils/api.ts`
  - `DatasetConfigItem` インターフェース: フィールド追加

### ❌ 触ってはいけない箇所（自動で処理される）

以下は`dataset_params.py`の関数経由で自動伝播するため、**手動で追加してはいけません**:

| ファイル | 箇所 | 使用関数 |
|---------|------|---------|
| `routes.py` L4640付近 create_training_run | yaml_config構築 | `extract_dataset_params()` |
| `routes.py` L5350付近 update_training_run | yaml_config構築 | `extract_dataset_params()` |
| `routes.py` L5190付近 get_training_run_params | YAML読み戻し | `read_dataset_params()` |
| `training_config.py` L200付近 `generate_lora_config()` | dataset_entry構築 | `extract_dataset_params()` |
| `training_config.py` L820付近 `generate_full_finetune_config()` | dataset_entry構築 | `extract_dataset_params()` |
| `training_config.py` L1086付近 `generate_controlnet_config()` | dataset_entry構築 | `extract_dataset_params()` |
| `train_runner.py` L942付近 | YAML→ds_config構築 | `read_dataset_params()` |
| `train_runner.py` L1009付近 | ds_config読み取り | `read_dataset_params()` |
| `train_runner.py` L1101付近 | ds_config読み取り（epoch reload用） | `read_dataset_params()` |

---

## Case B: Top-level パラメータの追加

Top-levelパラメータも**Pydanticモデル(`TrainingRunCreateRequest`)を単一の真実の情報源(SSoT)**として一元管理されています。

### バックエンドの編集箇所（最小2箇所）

#### 1. `backend/api/routes.py` — `TrainingRunCreateRequest` Pydanticモデル（必須）
SSoTです。フィールドを追加するだけで以下が自動的に処理されます:
- `create_training_run` / `update_training_run`: `request.model_dump()`経由でgeneratorに渡る
- `get_training_run_params`: `_extract_request_params_from_yaml()`が`model_fields`を走査して自動抽出

```python
class TrainingRunCreateRequest(BaseModel):
    # ... 既存のフィールド
    your_new_param: int = 10  # ← この1行で完結
```

#### 2. 実際の使用箇所（trainer / adapter）
- `backend/core/training/base_trainer.py` または対応するtrainerクラス
- 必要に応じて `backend/core/training/adapters/*.py`

`_build_train_section()`ヘルパーが自動でYAMLに書き込みます（フィールド名がそのままYAMLキーになる場合）。trainer側で `self.config.get("your_new_param", 10)` のように消費してください。

### 特殊なYAML配置が必要な場合

新パラメータが`process_config.train`セクション以外（`dtype`, `save`, `sample`, `network`等）に保存されるべき場合、追加で2箇所:

#### 3a. `backend/core/training/training_config.py` の `_build_train_section()`
新しいセクションへの書き込み処理を追加（例: `sample`セクションに保存する場合は対応するgeneratorの`sample`辞書に追加）。

#### 3b. `backend/api/routes.py` の `_YAML_FIELD_LOCATIONS`
YAML位置を明示することで読み戻しが自動化されます:
```python
_YAML_FIELD_LOCATIONS = {
    # ...
    "your_new_param": ("sample", "your_yaml_key"),  # sample.your_yaml_key
}
```

### フロントエンド（UIを追加する場合）
- `frontend/src/utils/api.ts` の `TrainingRunCreateRequest` インターフェース: フィールド追加
- `frontend/src/components/training/TrainingConfig.tsx`:
  - state変数 / UI入力欄
  - `handleSubmit`の`requestData`に含める
  - Edit Config読み戻し処理（自動的にバックエンドからparams辞書として返る）

### ❌ 触ってはいけない箇所（自動処理される）

| ファイル | 関数 | 自動処理の理由 |
|---------|------|---------------|
| `routes.py` create_training_run | `request.model_dump()`で全Pydanticフィールドが流れる | params_dict 経由 |
| `routes.py` update_training_run | 同上 | 同上 |
| `routes.py` get_training_run_params | `_extract_request_params_from_yaml()`がmodel_fieldsを走査 | スキーマ駆動 |
| `training_config.py` `generate_*_config` | `_build_train_section()`が共通dict処理 | LoRA/Full FT/ControlNet全て同じヘルパー |

### Top-level改修時の典型的なミスパターン（旧版・現在は防止済み）

| 旧ミスパターン | 現在の状態 |
|-----|---------|
| 4つのgenerate_*_config関数のうち1つに追加忘れ | ✅ `_build_train_section()`で一元化 |
| `handleSubmit`の`requestData`に追加忘れ | ⚠️ フロントエンドはまだ手動（Phase 3未完） |
| `get_training_run_params`の読み戻し追加忘れ | ✅ Pydanticスキーマ駆動 |
| Pydanticモデルに追加忘れ | ⚠️ 手動だがSSoTなので一目瞭然 |

---

## 改修前・改修後のセルフチェック

### Dataset-levelパラメータの場合
- [ ] `DatasetConfigItem` (routes.py) にフィールド追加
- [ ] `DATASET_LEVEL_PARAMS` (dataset_params.py) に1行追加
- [ ] 消費ポイントで`item["_your_flag"]`注入 or 使用ロジック追加
- [ ] （必要なら）フロントエンドUI追加
- [ ] 新規ランを作成→YAMLファイルに値が書き込まれているか確認
- [ ] Edit Config→保存→再度開いて値保持を確認
- [ ] `py_compile` で全変更ファイルの構文チェック

### Top-levelパラメータの場合
- [ ] `TrainingRunCreateRequest` (routes.py) にフィールド追加
- [ ] `create_training_run` でパラメータ受け取り→`generate_*_config`に渡す
- [ ] `update_training_run` でパラメータ受け取り→YAML更新
- [ ] `get_training_run_params` でYAMLから読み戻し
- [ ] **4つの`generate_*_config`関数すべてで処理**（最頻出ミス）
- [ ] `train_runner.py` / `base_trainer.py` で実装
- [ ] （必要なら）DBマイグレーション
- [ ] フロントエンドtype, state, UI, requestData, 読み戻し
- [ ] 新規ラン作成→YAMLファイル→実行→Edit Config→保存 の一連フローを検証
- [ ] `py_compile` で全変更ファイルの構文チェック

---

## 参考: 関連ファイル一覧

### バックエンド
- `backend/core/training/dataset_params.py` — dataset-level一元管理（新しいdataset-levelパラメータはここに追加）
- `backend/core/training/training_config.py` — YAML生成ロジック（4つの`generate_*_config`関数）
- `backend/core/training/train_runner.py` — サブプロセスでのトレーニング実行
- `backend/core/training/base_trainer.py` — トレーニングロジック本体
- `backend/core/training/adapters/*.py` — モデルアーキテクチャごとの差分
- `backend/api/routes.py` — API エンドポイント、Pydanticモデル

### フロントエンド
- `frontend/src/components/training/TrainingConfig.tsx` — トレーニング設定UI
- `frontend/src/utils/api.ts` — APIクライアント・型定義

---

## 将来的な改善案

Top-levelパラメータにもdataset_params.pyと同様の一元管理（`TRAINING_RUN_PARAMS`辞書）を導入すべきです。ただしPydanticモデルとの二重定義を避けるため、Pydanticモデルから自動抽出する仕組みが望ましいです。現状は手動管理のため、**特にTop-levelパラメータ追加時はチェックリストを必ず実行してください**。
