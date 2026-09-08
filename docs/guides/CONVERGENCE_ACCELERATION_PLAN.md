# 収束加速: 実装プラン

**対象設計: [CONVERGENCE_ACCELERATION_DESIGN.md](CONVERGENCE_ACCELERATION_DESIGN.md)**
**状態: Phase 0・Phase 1 実装済み**  日付: 2026-09-08

設計側の結論は「本稿だけではどの案も収束加速策として採用決定できない」である。
本プランはその制約を継承する。**Phase 1 までは診断の追加であり、収束改善の主張を含まない。**
Phase 2 以降は各ゲートの通過を条件とする分岐であり、事前に採用が決まった作業ではない。

---

## 0. 設計監査で変わった前提（プランの土台）

実装に直接効く 5 点。いずれも当初案から結論を変えている。

| 監査結果 | プランへの影響 |
|---|---|
| `aesthetic_loss.py` の forward は `torch.no_grad()` 内 | **微分可能な補助損失の前例は存在しない**。新規に autograd 経路を作る必要がある。既存コードのコピーは不可 |
| VAE decode は非局所。padding 項の消滅点は **14–16 latent cells**（`VAE_DECODE_BEHAVIOR.md`） | 8×8 latent の単独 decode は無効。**context margin 付き crop が必須**。ただし GroupNorm 統計と mid-block attention の項は margin では消えない |
| `context_tiled_decode.py` に margin=16 の geometry が既にある | crop decode の**幾何は再実装不要**（`iter_tiles` / `TileRect` / `resolve_geometry`）。不足は autograd 対応のみ |
| REPA は MiniT2I 専用。latent セルと DiT トークンは一般に 1 対 1 でない | REPA 系タスクは**独立ライン**。他アーキへの展開は別スコープ |
| x̂₀ の復元式は予測形式依存（eps / v / flow）。t の向き規約も未確定 | **全 Phase の前提**として復元ヘルパと規約の明文化を Phase 0 に置く |

再利用できる既存資産（実装確認済み）:

- `core/inference/context_tiled_decode.py` — `iter_tiles`, `TileRect`, `resolve_geometry`, margin 既定 16
- `core/training/vae/vae_losses.py::VaeLossBank` — LPIPS(frozen/eval, `[-1,1]`), YCbCr Charbonnier。`total` は graph 保持
- `base_trainer.py::log_extra_metric` + `metric_registry.EXTRA_METRIC_DEFS` — DB カラム・API 追加なしで指標追加
- `base_trainer.py::generate_sample` / `_run_step0_sample_if_due` — 学習中サンプル生成の既存経路
- `core/training/latent_debug_dump.py::channel_stats` — チャネル統計
- `core/training/latent_cache.py` — `vae_type` で名前空間分離済み

---

## 1. Phase 構成と依存

```
Phase 0 前提の是正           （必須・単独で価値あり）
   └─ Phase 1 診断計測        （必須・「流し続けるか」の材料）
         ├─ Phase 2 G-C 測定  → Phase 3 案1' opt-in loss
         │                        └─(費用不成立時のみ)→ Phase 4 案1 paired critic / G-1
         └─ Phase 5 REPA 系   （G-B → G-A、Phase 2-4 と独立・並行可）
```

案 0 / 2 / 3 は着手しない。Phase 1 の診断で足りるか、Phase 4 に吸収される。

---

## Phase 0 — 前提の是正

**目的**: 後続すべてが依存する規約と、そもそもの設定不整合を潰す。収束加速とは独立に正しさの問題。

### 0-1. 設定・キャッシュ整合の監査（調査タスク、コード変更は発見次第）

- VAE 差し替え時の latent 正規化（`scaling_factor` / `shift_factor` / チャネル別統計）の由来を追う。
  出所: `core/training/vae_swap.py`, 各 `adapters/*_adapter.py`, `latent_cache.py`
- **既存 latent キャッシュが旧 encoder 由来でないかを確認**。`latent_cache.py` は `vae_type` で
  名前空間を切っているが、任意 VAE 差し替え時にこのトークンが実際に変わるかを検証する。
  変わらないなら、差し替え後の学習が旧 latent を読む**サイレントな不整合**になる
- 予測形式・sampler・CFG・VAE が「学習時」と「debug 生成時」で一致しているかを確認

**成果物**: 監査メモ（不整合ゼロならその旨）。不整合があれば修正 1 件 = 1 コミット。
**注意**: これは「まず確認すべき安いこと」であり、**Phase 1 より先に終わらせる**。

### 0-2. x̂₀ 復元ヘルパと t 規約の明文化

- `predict_x0(noise_process, z_t, model_out, t_or_sigma, scheduler)` を 1 箇所に集約。
  eps: `(z_t − σ ε̂)/α`（α≠0 をガード）／ flow: `x̂₀ = z_t − t v̂`（`z_t=(1−t)x₀+tε`）／ v 予測は別式
- **t の向き規約をコメントで固定**。**リポジトリ全体で一つの向きに揃ってはいない**:
  どちらの端が clean かはアーキごとに `ArchHandler.timestep_convention`
  （`core/training/arch/base_arch.py`）が宣言する。`t0`（t=0 が clean）が既定で、
  **`t1`（t=1 が clean）は SenseNova と MiniT2I**。SD15/SDXL は `resolve_timestep_convention()`
  で run の `noise_process` により両者を切り替える。
  ヘルパ自身の引数がどちらの座標を取るかは、この宣言とは**別の語彙**なので混同しないこと
- 帯域指定は t の生値ではなく **SNR** で行う API にする

**配置**: `core/training/ops/` 配下（既存の ops レイヤ慣習に合わせる）
**検証**: 各 arch の既存 scheduler で往復テスト（`x₀ → z_t → predict_x0 → x₀` の一致）。
実 GPU 不要、`pytest` で完結。
**コミット**: 1 件。

---

## ✅ Phase 1 — 診断計測（実装済み）

**目的**: 「長期に流すか捨てるか」の材料を最短で得る。**症状 A/B が実際に再現するかの確認もここで初めて可能になる。**

### 1-1. 固定条件スナップショット

学習中 N step ごとに、固定 seed / prompt / sampler / CFG / VAE / 正規化で 3 種を保存:

1. **単発予測** — `predict_x0` による x̂₀（学習と同じ経路）
2. **通し生成** — 実運用の step 数。案 4 の短縮 rollout を使う場合は通常 step 数とも比較して solver の粗さを分離
3. **GT roundtrip** — データセット画像の encode→decode

既存 `generate_sample` / `_run_step0_sample_if_due` の経路に相乗りする。新しい生成機構は作らない。

### 1-2. 指標を `log_extra_metric` に流す

| 指標 | 目的 | 対照 |
|---|---|---|
| チャネル別 mean/std と参照分布の差 | 症状 A（大域ずれ） | GT latent の同統計 |
| 低周波パワー比 | 症状 A | GT roundtrip |
| decode 後の輝度・彩度・分散 | 症状 A（画素側） | GT roundtrip |
| latent セル周期のパワー | 症状 B | **周期ずらし・非周期ノイズ・正常画像を必ず対照に置く**（正常な模様も 8px に反応するため） |
| 単発 x̂₀ と通し生成の同指標の差 | A が軌跡固有かの切り分け | 同一 checkpoint |

`channel_stats` を流用。`EXTRA_METRIC_DEFS` に登録してチャートに出す。

### 1-3. 明示する限界

- 統計一致は配置・意味・多様性・将来の収束を保証しない
- **この指標だけで収束可否を断定しない**。画像と併せて判断する旨をメトリクス説明文に書く

**変更ファイル**: `base_trainer.py`（フック）, `metric_registry.py`（定義）, 新規 1 ファイル（統計計算）
**パラメータ**: 有効化フラグと間隔のみ。`param_defaults.py::TRAINING_DEFAULTS` → Pydantic → フロントの順で追加
**検証**: 3 step smoke（有限値が出ること、学習が止まらないこと）。収束実験は行わない
**コミット**: 1-1/1-2 で 2 件程度

---

## Phase 2 — G-C: crop decode の妥当性測定（loss 化しない）

**目的**: 案 1' が成立するかを、**実装前に**測る。

### 2-1. autograd 対応の context crop decode

```
decode_crop_with_context(vae, latent, rect, margin_cells) -> pixels(内側のみ)
```

- 幾何は `context_tiled_decode.iter_tiles` / `TileRect` を**そのまま再利用**
- **VAE パラメータは凍結。しかし入力 latent への autograd は維持する**
  （`no_grad` / `detach` で予測側を囲むと拡散モデルが学習できない ← `aesthetic_loss` の轍）
- 内側領域のみ loss 対象。margin は捨てる

### 2-2. 測定 probe

`backend/core/training/probes/` に追加（既存慣習に合わせる）。測る項目:

| 項目 | 内容 |
|---|---|
| 値の一致 | 全画像 decode の同一領域 vs crop 内側。margin を 0/4/8/12/16/20 cells で掃引 |
| **勾配方向の一致** | 入力 latent への勾配の cosine 類似度。**値の一致だけでは不十分** |
| 残存非局所誤差 | GroupNorm 統計項・mid-block attention 項は margin で消えないため、その寄与を分離 |
| 費用 | 実行時間とピーク VRAM。全画像 decode との比 |

`vae_tile_global_norm` 相当の統計移植が勾配経路でも使えるかを併せて確認する。

**許容差は測定前に決めて記録する。**

**注意（GPU probe）**: モデルをロードする probe なので、実行前に**ホスト RAM のピーク見積もりを提示**し、
ユーザーの学習が走っていないことを確認してから実行する。VAE のみのロードなので小さい想定。

**ゲート G-C**: 事前登録した許容差を、値・勾配方向の両方で満たすこと。費用も許容内であること。
**不合格時**: 案 1' を破棄し Phase 4（案 1 paired critic）へ分岐。破棄理由を設計書に追記。
**コミット**: 2-1 と 2-2 で 2 件。測定結果は設計書に追記して 1 件。

---

## Phase 3 — 案 1' を opt-in loss として実装（G-C 合格が前提）

**目的**: 症状 B に対する候補を、**既定 off の opt-in** として入れる。採用決定ではない。

### 3-1. パラメータ（CLAUDE.md の必須順序に従う）

`param_defaults.py::TRAINING_DEFAULTS` → OpenAPI → Pydantic → フロント。**キー数は最小に抑える**:

| キー | 既定 | 内容 |
|---|---|---|
| `crop_decode_loss_enable` | `false` | 有効化 |
| `crop_decode_loss_weight` | `0.0` | 主 loss に対する重み |
| `crop_decode_loss_margin_cells` | `16` | G-C の測定結果で決める |
| `crop_decode_loss_out_cells` | 測定で決定 | 内側領域のサイズ |
| `crop_decode_loss_metric` | `lpips` | `VaeLossBank` の項から選択 |
| `crop_decode_loss_snr_range` | 帯域 | **t 生値ではなく SNR で指定** |

### 3-2. 損失経路

```
model_out ─ predict_x0(Phase 0-2) ─ crop 選択 ─ decode_crop_with_context(Phase 2-1)
                                                      │
GT latent ────────────────────────────────────────────┤
                                                      ▼
                                        VaeLossBank の指標（frozen, graph 保持）
                                                      ▼
                                        total_loss += w * metric
```

- crop 位置はステップごとにランダム。1 サンプル 1 crop から始める
- `VaeLossBank` の LPIPS は既に frozen/eval かつ `total` が graph を保つので**そのまま使える**

### 3-3. 係数決定を測定可能にする

- **主 loss と補助 loss の勾配ノルム比**を `log_extra_metric` に出す
- 設計書のとおり「値の比だけで係数を決めない」。勾配ノルム比と画像を見て決める
- patch-wise 化は**無条件の上位互換として扱わない**。必要が示されてから

### 3-4. 検証

- 3 step smoke: 有限 loss、**勾配が transformer に到達していること**（`p.grad` の非ゼロ確認）
- fp16/bf16 の両方で通す（`verify-in-production-dtype` の教訓）
- `python -c "import ..."` による実インポート確認（`py_compile` だけでは不足）
- **収束実験は行わない**

**コミット**: 3-1（パラメータ配線）、3-2/3-3（損失経路）で 2 件。

---

## Phase 4 — 案 1 paired critic（Phase 3 が費用面で不成立の場合のみ）

**分岐条件**: G-C 不合格、または Phase 3 の実測費用が許容外だった場合**のみ**着手する。
Phase 3 が成立するなら Phase 4 は不要（蒸留誤差を持ち込む理由がない）。

1. **データ生成** — 複数 checkpoint の実予測 + 合成劣化。**画像単位とチェックポイント単位で保留集合を分離**
2. **教師値** — LPIPS 単独ではなく、解釈可能な量のベクトル（周波数パワー / 高周波比 / mean-std / 局所分散）を併記
3. **critic** — `LatentCNN` を素材にするが、入力は `(x̂₀, x₀)` ペア。`aesthetic_loss` の `no_grad` 構造は踏襲しない
4. **ゲート G-1** — 順位相関（目安 ≥0.9）だけでなく、
   **critic を下げる更新が真の decode 指標と画像を改善するか**を確認する。
   分布外入力・同 L2 で画質が異なるペア・低周波/高周波の両方を含める

不合格は「現モデル/データでの棄却」であり、理論上の予測可能性の否定ではない（設計書の表現に従う）。

---

## Phase 5 — REPA 系（Phase 2-4 と独立・並行可）

### 5-1. G-B を先に回す（モデル新規作成不要・安い）

**教師の前処理だけを変える対照実験**。latent 化とは独立に、正方形スカッシュの実害を測る。

- 固定: モデル・データ・seed・その他すべて
- 変数: 教師画像の前処理（正方形 squash vs アスペクト保持）
- 測定: **非正方形バケツ**での空間対応と生成品質。
  **教師が異なる系の REPA loss の大小だけでは判定しない**（設計書の指示）

**G-B 不合格 = 品質側の動機は消える**。その場合 latent 化はコスト削減のみが根拠となり、優先度は下がる。

### 5-2. latent stem 蒸留（G-B 合格時）

- `(latent, 対応画像)` から pixel tagger の patch 特徴を蒸留
- **latent セルと DiT トークンの対応を明示的に定義する**（1 対 1 を仮定しない）。
  patch size / packing / 圧縮率 / crop・flip の対応を設計に書く
- 凍結 trunk のどの層に stem を接続するか、位置埋め込みとグリッドの扱いを決める
- **歪んだ教師をそのまま蒸留しても前処理問題は解決しない** — 5-1 の結果を反映した教師を使う
- **ゲート G-A**: 保留 patch cosine（目安 ≥0.9）**に加えて**、
  固定予算・複数 seed の学習 A/B で生成品質・収束・時間・VRAM を比較。
  特徴類似度だけで品質維持を保証しない

**スコープ注意**: REPA は現状 MiniT2I 専用。他アーキへの展開は本プランの範囲外。

---

## 2. 横断ルール

| ルール | 根拠 |
|---|---|
| 新規機能はすべて **opt-in・既定 off** | 既存学習への影響ゼロを保証する |
| パラメータは `param_defaults.py` → OpenAPI → Pydantic → フロントの順 | CLAUDE.md 必須手順 |
| **収束実験は行わない**。理論 + 3 step smoke で十分とする | 長期実行の可否判断が本件の痛点そのもの |
| `py_compile` に加えて**実インポート**で検証 | モジュールロード時 NameError は `py_compile` を通る |
| fp16 / bf16 の両方で通す | fp32 検証済みコードが本番 dtype で落ちた前例 |
| **バックエンドを勝手に再起動しない** | ユーザーの学習が走っている可能性がある |
| GPU probe は**ホスト RAM ピークを事前申告**し、1 度に 1 本 | 過去に pagefile 枯渇 |
| コミットは Phase 内の単位ごと。**必ず明示 pathspec で** | `git add <file> && commit` は他セッションのステージ済み作業も巻き込む |
| ツリーは他セッションと共有。`stash` / `checkout` / `reset` / `add -A` を使わない | 共有ワークツリー |

---

## 3. 実行順序（推奨）

| 順 | 作業 | 依存 | 中止条件 |
|---|---|---|---|
| 1 | Phase 0-1 設定・キャッシュ監査 | なし | — |
| 2 | Phase 0-2 x̂₀ ヘルパ + t 規約 | なし | — |
| 3 | Phase 1 診断計測 | 0-2 | — |
| 4 | Phase 5-1 G-B（並行可） | なし | G-B 不合格 → Phase 5 凍結 |
| 5 | Phase 2 G-C 測定 | 2-1 | G-C 不合格 → Phase 4 へ分岐 |
| 6 | Phase 3 案 1' opt-in loss | G-C 合格 | 費用許容外 → Phase 4 へ分岐 |
| 7 | Phase 4 / 5-2 | 各ゲート | — |

**Phase 0〜1 だけでも独立した価値がある**（設定不整合の是正 + 継続判断の材料）。
Phase 2 以降は測定結果次第で消える可能性があり、それは失敗ではなく想定内の分岐である。

---

## 4. このプランが答えないこと

- どの案が収束を加速するか。**すべてのゲートは「採用してよいか」ではなく「次に進んでよいか」を判定する**
- 症状 A と B が独立した原因かどうか（Phase 1 の再現確認で初めて扱える）
- 各 Phase の所要時間。設計書 §7 のとおり、条件を固定した実測なしに見積もらない
