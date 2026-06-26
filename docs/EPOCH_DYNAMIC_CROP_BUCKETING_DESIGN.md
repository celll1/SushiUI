# Epoch-Dynamic Crop & Bucketing 設計書

SDXL 学習において、**epoch ごとに各画像のクロップ/スケールを変化させ、それに応じて bucket 割当も
epoch ごとに変動させる**ための設計書。狙いは、様々なクロップ範囲・拡大縮小率での学習による
**外挿性（extrapolation）の獲得**。

- 対象アーキテクチャ: **SDXL のみ**（time_ids を持つのは SDXL のみ）
- micro-conditioning 意味論: **kohya 式**（`original_size` = 元画像全体サイズ）
- 本書は設計のみ。実装は別途。

---

## 1. 背景と要件

### 1.1 モチベーション
- 学習させたい拡大縮小率も様々、元画像に対するクロップ範囲も様々。
- 出力意図の大半は「全体画像」だが、一定割合で「下限付きランダムクロップ」を混ぜたい。
  - 例: **面積 25% 以上 かつ 短辺 512px 以上**（いずれも設定可能）の範囲でランダムクロップ。
- 真の外挿性のためには、データごとに入る bucket を固定すべきではなく、**epoch ごとに
  bucketing が変化する**ことに対応する必要がある。

### 1.2 機能要件（2×2 マトリクス）

各 (item, epoch) を **2 つの独立軸**で決める。各軸の確率で 4 象限の割合を制御する。

| (crop 軸 \ bucket 軸) | 最大fitバケット（縮小最小） | より小さいバケット（追加縮小） |
|---|---|---|
| 全体画像（最小crop のみ） | ① 全体→最大bucket（原寸で入れば原寸） | ③ 全体→小bucket |
| ランダムcrop（面積≥指定） | ② crop→最大bucket（cropがbucket級なら原寸） | ④ crop→小bucket |

- `crop_full_image_prob` = P(全体画像) → ①③
- `crop_max_bucket_prob` = P(最大fitバケット) → ①②

その他の機能要件:
1. クロップ結果の解像度に応じて bucket を **epoch ごとに**再割当（小クロップ→小 bucket へ移動）。
2. クロップに整合する SDXL time_ids（kohya 式）を per-item で付与。
3. **resume で完全再現**（クロップ計画・bucket 割当・バッチ順・global_step が一致）。
4. 既存機能（priority training / VE reconstruction / マルチデータセット / Danbooru augmentation /
   per-epoch caption shuffle）と共存。

### 1.3 非対象
- SD1.5 / Z-Image / FLUX.2 / Anima / MiniT2I（time_ids 非対応）。
- epoch 内でのクロップ変動（クロップは epoch 単位で固定。1 epoch 内の同一 item は同一クロップ）。

---

## 2. 現状アーキテクチャ（実装済みで再利用する箇所）

| 機構 | 場所 | 本設計での扱い |
|------|------|----------------|
| バッチを epoch 毎に再構築 | `base_trainer.py:9648` `build_batch_indices` | **そのまま流用**（bucket を作り直せば動く） |
| epoch 毎データセット再読込 | `base_trainer.py:9537` `reload_for_epoch` | CropPlanner はこの直後に走らせる |
| no-upscale 最大 bucket 割当 | `bucketing.py:298-312` | クロップ寸法→bucket 割当で**再利用** |
| bucket 形状変化時の VRAM 解放 | `base_trainer.py:9947` | 多 bucket 化で重要、既存で対応済み |
| micro-cond 導出（resize/crop/random_crop） | `base_trainer.py:4289-4348` | CropPlanner 経由に**置換・統合** |
| `_recompute_sdxl_micro_cond` | `base_trainer.py:1675` | CropPlanner に統合（(0,0) 近似バグを解消） |
| latent cache key `{path}_{w}_{h}` | `latent_cache.py:109` | クロップ非対応 → onthefly 強制で回避 |
| training_state 保存（epoch/batch_idx/random_state/fingerprint） | `base_trainer.py:2837-2845` | crop 関連フィールドを追加 |

### 2.1 現状の制約（本設計で解消する）
- bucket 割当は setup 時の 1 回（`base_trainer.py:8942`）→ item ごとに固定。
- latent cache はクロップ情報を持たない → 同一画像は常に同一（center-crop）latent。
- step 会計は「steps_per_epoch 一定」前提（`base_trainer.py:9919`）、resume fallback は
  `global_step // steps_per_epoch`（`base_trainer.py:9099`）→ 可変 step 数で破綻。

---

## 3. パラメータ（`TRAINING_DEFAULTS` = Single Source of Truth）

`backend/api/param_defaults.py` に追加。全て**最初にここへ追加**してから routes / config / frontend へ展開。

```python
# Epoch-dynamic crop augmentation (SDXL only). Two independent axes (crop, bucket size)
# pick how the image is presented; re-bucketed each epoch. Requires onthefly_gpu encoding.
"crop_augment_enable": False,             # master switch
# Mix proportions (2x2 axes):
"crop_full_image_prob": 0.7,              # P(full image, minimal crop only) -> (1)(3)
"crop_max_bucket_prob": 0.7,              # P(largest-fitting bucket = least downscale) -> (1)(2)
# Random-crop controls:
"crop_min_area_ratio": 0.25,              # crop area >= ratio * original area
"crop_min_short_side_px": 512,            # crop short side (original px) >= this (also bounds aspect)
"crop_aspect_mode": "source",             # "source" (keep image aspect) | "free" (any aspect)
"crop_position_mode": "random",           # "random" (any point) | "corner" (touch a corner)
# Smaller-bucket controls:
"crop_smaller_bucket_mode": "base_res",   # "base_res" (smaller base_resolution) | "scale_range"
"crop_smaller_scale_range": [0.5, 0.9],   # downscale range for scale_range / single base_res fallback
# Full-image (minimal crop) position:
"full_crop_position_mode": "center",      # "center" | "fixed_corner" | "random"
# Conditioning + seed:
"crop_microcond_mode": "kohya",           # time_ids semantics. "kohya" = original_size is full image
"crop_plan_seed": 0,                      # 0 = derive from global training seed
```

### 3.1 パラメータ意味論
- **混合割合**: `crop_full_image_prob`（全体 vs crop）, `crop_max_bucket_prob`（最大fit vs 小bucket）の
  独立 2 確率で 4 象限（①②③④）の割合を制御。
- **ランダムcrop**:
  - `crop_min_area_ratio` / `crop_min_short_side_px`: 窓の面積・短辺の下限（短辺下限がアスペクト比の
    極端化も抑制）。満たせない小画像は全体画像へフォールバック。
  - `crop_aspect_mode`: `source`=元画像と同アスペクト、`free`=任意アスペクト（例 2048²→1024×2048）。
  - `crop_position_mode`: `random`=任意点、`corner`=四隅のいずれかを含む。
- **小bucket選択**: `crop_smaller_bucket_mode`=`base_res`（`base_resolutions`の小解像度を一様選択。
  単一base_resのときは scale_range にフォールバック）/ `scale_range`（連続縮小率
  `crop_smaller_scale_range` を最大bucket解像度に適用し /64 量子化）。
- **全体画像の最小crop位置**: `full_crop_position_mode`=`center`/`fixed_corner`/`random`。
  アスペクト維持の cover 後、はみ出しをこの位置で crop。
- `crop_microcond_mode`: `"kohya"` 固定。

### 3.2 配線チェックリスト（CLAUDE.md 準拠）
1. `param_defaults.py` の `TRAINING_DEFAULTS` に crop_* キー追加。
2. `routes.py` の `TrainingRunCreateRequest`（Pydantic）に `= TRAINING_DEFAULTS["..."]` 参照で追加。
3. `training_config.py` の YAML 生成に追加（`p.get(...)`）。
4. `TrainingConfig.tsx` の `DEFAULT_CONFIG` / 送信処理 / UI セクション追加。
5. `api.ts` の型定義に追加。
6. **追加漏れ検査**: `TaggerTrainingRunCreateRequest` 同様、Pydantic フィールドと `TRAINING_DEFAULTS`
   キーの突合を行う。

---

## 4. micro-conditioning 意味論（kohya 式）

SDXL time_ids = `[original_h, original_w, crop_top, crop_left, target_h, target_w]`。

定義（元画像 `(ow, oh)`、クロップ窓 左上 `(cx, cy)`・サイズ `(cw, ch)`、出力 bucket `(bw, bh)`）:

| 成分 | 値 | 説明 |
|------|----|------|
| `original_size` | `(oh, ow)` | **元画像全体の実サイズ**（kohya 式の核心） |
| `crop_top_left` | `(cy, cx)` | クロップ窓左上の**元画像ピクセル座標** |
| `target_size` | `(bh, bw)` | 出力 bucket サイズ |

- **全体画像ケース**: `(cx, cy) = (0, 0)`、`(cw, ch) = (ow, oh)` → `original=(oh,ow)`,
  `crop=(0,0)`, `target=bucket`。これは**現状の micro-conditioning と完全一致**（後方互換）。
- クロップ窓のアスペクト比は、bucket 割当（§5.2）で選ばれた bucket のアスペクト比に一致させる
  （窓を bucket アスペクトに合わせてから配置）。これにより resize 歪みを避ける。

> 補足: pixel 座標系の厳密な規約（元画像座標 vs resized 座標）は実装時に kohya-ss/sd-scripts の
> `crop_ltrb` と突合して確定する。本書では「全体画像時に現状と一致する」ことを不変条件とする。

---

## 5. 新規モジュール `CropPlanner`

`backend/core/training/crop_planner.py`（新規・1 ファイル）。

### 5.1 役割
- 全 epoch 分のクロップ計画を**学習開始前に事前計算**（決定論的・画像ヘッダのみ使用）。
- 各 (item, epoch) → `CropSpec` を返す純粋関数。
- step 会計用に epoch 毎の bucket 分布 → バッチ数 → 累積 step オフセット表を提供。

### 5.2 データ構造
```python
@dataclass(frozen=True)
class CropSpec:
    is_full: bool                  # True = 全体画像
    crop_box: Tuple[int,int,int,int]  # (cx, cy, cw, ch) 元画像ピクセル
    bucket_w: int
    bucket_h: int
    time_ids: Tuple[int,int,int,int,int,int]  # kohya 式 (oh,ow,ct,cl,bh,bw)
```

### 5.3 決定論シード（resume 再現の核心）
クロップ決定は**グローバル RNG ストリームから独立**させる。

```python
# Per-(epoch, item) independent RNG. Pure function of (seed, epoch, image_path),
# so resume regenerates identical crops regardless of interruption point.
def _item_rng(seed: int, epoch: int, image_path: str) -> random.Random:
    h = hashlib.sha256(f"{seed}|{epoch}|{image_path}".encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))
```
- これにより「長い RNG ストリームの再現」を回避（最難問の解消）。
- バッチ並び順は従来通りグローバル `random` + 既存 `random.getstate()` 保存/復元で担保
  （クロップ用 RNG とは**別系統**）。

### 5.4 計画アルゴリズム（per item, per epoch）
```
rng = _item_rng(seed, epoch, image_path)
ow, oh = header_size(image_path)              # _get_original_size_for_item を流用
full     = rng.random() < crop_full_image_prob    # crop 軸
use_max  = rng.random() < crop_max_bucket_prob    # bucket 軸（描画順固定で決定論）

if full:
    bucket = select_bucket(ow, oh, use_max, rng)   # 画像アスペクトから最大fit/小bucket
    cw, ch = max_window_for_aspect(ow, oh, bucket)  # アスペクト維持 cover
    cx, cy = place(rng, ow, oh, cw, ch, full_crop_position_mode)
else:
    win = sample_crop_window(rng, ow, oh)          # area≥ratio・short≥min、source/free
    if win is None: return full_spec(..., fallback=True)
    cw, ch = win
    cx, cy = place(rng, ow, oh, cw, ch, crop_position_mode)  # random/corner
    bucket = select_bucket(cw, ch, use_max, rng)    # crop窓から最大fit/小bucket
spec = CropSpec(is_full=full, crop_box=(cx,cy,cw,ch),
                bucket_w=bucket.w, bucket_h=bucket.h, time_ids=kohya(oh,ow,cy,cx,bucket))
```

**`select_bucket(rw, rh, use_max, rng)`**（base_res 主軸）:
- 各 base_res で `get_bucket_for_image_size` → no-upscale で収まる候補を抽出。
- `use_max`: 収まる中で最大 base_res のバケット（原寸で入れば原寸）。
- 非 use_max: `base_res` モード=最大より小さい base_res を一様選択（無ければ scale_range）。
  `scale_range` モード/フォールバック=最大bucket解像度 × `uniform(smaller_scale)` を /64 量子化して再バケット。

**`sample_crop_window`**: 面積 `A∈[max(min_area·img, min_short²), img]` を一様、`source`=元アスペクト、
`free`=`A` から定まる有効アスペクト範囲で log-uniform（短辺下限が極端アスペクトを排除）。

- 制約を満たせない小画像は **full_image にフォールバック**（`fallback=True` でログ可能、サイレント禁止）。
- ネイティブ crop（リサイズ無し）は「② crop→最大bucket かつ crop 窓 ≒ bucket」のとき自然に発生
  （例 4096²から2048²窓→2048bucket）。割合を増やすには `crop_min_area_ratio` を調整。

### 5.5 事前計算 API
```python
class CropPlanner:
    def __init__(self, config, base_resolutions, multi_resolution_mode): ...
    def precompute(self, items_by_dataset, num_epochs) -> None: ...
    def spec_for(self, epoch: int, image_path: str) -> CropSpec: ...
    def bucket_assignment_for_epoch(self, epoch: int) -> Dict[image_path, (bw,bh)]: ...
    def steps_per_epoch(self, epoch: int, batch_size: int, mnt: int) -> int: ...
    def cumulative_step_offsets(self, batch_size, mnt) -> List[int]: ...
    def fingerprint(self) -> str:    # hash(crop params + num_epochs + seed + dataset fp)
```

---

## 6. 処理フロー（変更点）

### 6.1 学習開始前（setup）
1. 既存の bucket setup（`base_trainer.py:8794`）は **CropPlanner.precompute()** に置換／併設。
2. `crop_augment_enable=True` のとき **`latent_encoding_mode='onthefly_gpu'` を強制**（warn ログ）。
   - 既存の強制パターン（`base_trainer.py:8986-8990`）に倣う。
   - disk cache / swap buffer はクロップ非整合のため使用不可。
3. `total_steps = sum(steps_per_epoch(e) for e in range(num_epochs))` を `cumulative_step_offsets`
   から厳密に確定 → DB / progress に反映（`update_total_steps_callback`）。

### 6.2 epoch ループ冒頭（`base_trainer.py:9528` 内）
`reload_for_epoch` の直後、バッチ構築の直前に挿入:
```
1. CropPlanner.bucket_assignment_for_epoch(epoch) を取得
2. bucket_manager.buckets をクリアし、この epoch の (item -> bucket) で再構築
   - priority / VE reconstruction / Danbooru 注入の **後** に実行（item 集合確定後）
3. 各 item に CropSpec を添付（item["_crop_spec"] = spec）
4. 既存 build_batch_indices(batch_size) → batches（変更不要）
```

### 6.3 バッチ消費時（encode）
- `encode_image` に **per-item crop_box 経路**を追加:
  - `item["_crop_spec"].crop_box` で元画像から該当領域を切出し → bucket へ resize。
  - time_ids は `item["_crop_spec"].time_ids` を直接使用（再導出しない）。
- `micro_cond_list`（`base_trainer.py:10200`）は CropSpec.time_ids を収集するだけに簡素化。
- swap/cache 経路は無効化済み（onthefly 強制）なので `_recompute_sdxl_micro_cond` の (0,0) 近似
  バグ経路は通らない。

---

## 7. step / progress 会計

- **可変 steps_per_epoch** に対応するため、`cumulative_step_offsets`（長さ `num_epochs+1`）を保持。
  - `epoch_start_step(epoch) = offsets[epoch]`
  - `total_steps = offsets[num_epochs]`
- `base_trainer.py:9908-9926` の「初回 epoch で total_steps 補正」ロジックは offsets 参照に置換。
- resume の epoch 算出（`base_trainer.py:9099, 9177` の `global_step // steps_per_epoch`）は
  **offsets による検索**に置換: `epoch = bisect_right(offsets, global_step) - 1`。

---

## 8. resume 設計

### 8.1 training_state 追加フィールド（`save_training_state`）
```python
state += {
    "crop_plan_seed": self.crop_plan_seed,
    "crop_plan_fingerprint": crop_planner.fingerprint(),
    "step_offsets": offsets,   # 可変 step 会計の再現用
}
```

### 8.2 resume 手順
1. `crop_plan_fingerprint` 照合（クロップ設定 or num_epochs or dataset 変化で不一致）。
   - 不一致 → 既存 dataset_fingerprint と同じく **fresh fallback**（クロップ計画を作り直し、
     現 epoch を batch 0 から、`global_step` は保持）。
2. 一致時:
   - `crop_plan_seed` で CropPlanner を再構築 → `start_epoch` の bucket 割当を**決定論的に再生成**
     （中断位置に依存しない）。
   - グローバル `random.setstate(random_state)`（既存）でバッチ並びを復元。
   - bucket 再構築 → `build_batch_indices` → 同一 batches → `batches[resume_batch_idx:]` で正しくスキップ。
3. `step_offsets` を復元し、可変 step 会計を継続。

### 8.3 不変条件（テストで保証）
- `spec_for(epoch, path)` は **(seed, epoch, path) のみの関数**（実行回数・順序に非依存）。
- 全体画像ケースの time_ids は現状 micro-conditioning と一致（後方互換）。
- 同一 seed・同一設定で 2 回計画 → 完全一致。

---

## 9. 既存機能との干渉

| 機能 | 注意点 |
|------|--------|
| priority training | item 分類・注入後に CropPlanner の bucket 再構築を実行。priority/normal 双方に CropSpec 付与。 |
| VE reconstruction mode | reference_images 注入後に実行。クロップは reference にも同一適用するか要検討（初期は full 推奨）。 |
| マルチデータセット | `_item_rng` のキーに dataset_unique_id を含め衝突回避。 |
| Danbooru augmentation | 動的注入 item は image_path/bytes が必要。header サイズ取得経路（`_danbooru_image_bytes`）を流用。 |
| per-epoch caption shuffle | `reload_for_epoch` と独立。順序は CropPlanner→batch 構築で固定。 |
| 既存 `bucket_strategy` | `crop_augment_enable=True` 時は CropPlanner が優先。OFF 時は従来動作を完全保持。 |

---

## 10. 影響ファイル一覧

| ファイル | 変更内容 |
|----------|----------|
| `backend/api/param_defaults.py` | `TRAINING_DEFAULTS` に 8 キー追加（最初） |
| `backend/core/training/crop_planner.py` | **新規**。CropPlanner / CropSpec / 決定論 RNG / 事前計算 |
| `backend/core/training/base_trainer.py` | setup（onthefly 強制・precompute）、epoch ループ冒頭（bucket 再構築）、`encode_image`（crop_box 経路）、`micro_cond_list` 簡素化、step 会計（offsets）、resume（fingerprint/seed 再生成）、`save/load_training_state` 拡張 |
| `backend/core/training/bucketing.py` | `assign_bucket_for_crop` 相当の薄いヘルパ追加（既存ロジック流用） |
| `backend/core/training/training_config.py` | YAML 生成に 8 キー追加 |
| `backend/api/routes.py` | Pydantic に 8 フィールド追加（`= TRAINING_DEFAULTS[...]`） |
| `frontend/src/components/training/TrainingConfig.tsx` | Crop Augmentation UI セクション・`DEFAULT_CONFIG`・送信処理 |
| `frontend/src/utils/api.ts` | 型定義追加 |

---

## 11. テスト項目

### 11.1 ユニット（CropPlanner）
- [ ] `spec_for` 決定論性: 同 (seed, epoch, path) で N 回 → 完全一致。
- [ ] epoch 非依存独立性: epoch を変えると crop が変わる／同 epoch 内は不変。
- [ ] 制約遵守: 全クロップで area ≥ ratio かつ short ≥ min_px（or full フォールバック）。
- [ ] 全体画像 time_ids が現状 micro-conditioning と一致（後方互換）。
- [ ] `crop_full_image_prob` の実測割合が設定値に統計的に一致。
- [ ] 小画像（制約満たせない）→ full フォールバック＋ログ。

### 11.2 統合
- [ ] `crop_augment_enable=True` で `latent_encoding_mode` が onthefly_gpu に強制される。
- [ ] epoch 間で bucket 分布が変化する（同一 item が別 bucket に移動するケースを確認）。
- [ ] `total_steps` が可変 steps_per_epoch の総和と一致。
- [ ] バケット形状変化時の VRAM 解放が機能（既存 9947 経路）。

### 11.3 resume
- [ ] mid-epoch 中断→resume で batches とクロップが完全一致（loss 連続性で確認）。
- [ ] epoch 境界 resume で `bisect` による epoch 算出が正しい。
- [ ] クロップ設定変更後の resume → fingerprint 不一致で fresh fallback（global_step 保持）。
- [ ] dataset 変更後の resume → 既存 fallback と整合。

### 11.4 コンパイル
```
"d:\celll1\webui_cl\venv\Scripts\python.exe" -m py_compile \
  backend/core/training/crop_planner.py \
  backend/core/training/base_trainer.py \
  backend/core/training/bucketing.py \
  backend/api/routes.py backend/api/param_defaults.py \
  backend/core/training/training_config.py
```
- `python -c "import core.training.crop_planner"` でモジュールロード検証（py_compile が見逃す
  NameError 検出）。

---

## 12. 実装フェーズ提案

1. **Phase 1**: `param_defaults.py` + CropPlanner（事前計算・決定論 RNG・full フォールバック）+ ユニットテスト。
2. **Phase 2**: `encode_image` の crop_box 経路 + time_ids（kohya）+ onthefly 強制。
3. **Phase 3**: epoch ループへの bucket 再構築接続 + 可変 step 会計（offsets）。
4. **Phase 4**: resume 拡張（fingerprint / seed 再生成 / step_offsets）。
5. **Phase 5**: frontend UI + 配線 + 既存機能干渉の確認。
6. **Phase 6**: 統合テスト・ドキュメント更新。

---

## 13. 未決事項（実装着手前に確認）

- pixel 座標規約（元画像座標 vs resized 座標）の kohya 突合（§4 補足）。
- VE reconstruction mode 時のクロップ適用方針（初期は full 推奨）。
- ランダムクロップの面積サンプリング分布（現状は面積を一様サンプル、free アスペクトは
  有効範囲で log 一様）。ネイティブクロップ（窓＝バケット寸）を確実に一定割合で混ぜる
  専用ノブの要否。
