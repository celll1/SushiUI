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
| latent セルと DiT トークンは一般に 1 対 1 でない | REPA 系タスクは**独立ライン** |
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

### 1-4. 実装後に判明した限界（2026-09-09 時点）

- `_last_predicted_latent` は本番経路では誰も代入しておらず、単発 x̂₀ 由来の指標は
  全アーキで到達不能だった。`860d196c` で `compute_crop_decode_loss` に捕捉点を置いて解消。
  供給範囲は `ArchHandler.supplies_predicted_latent` が宣言し、crop decode loss の消費範囲と同じ
  9 / 4 の内訳（ControlNet 学習は全アーキで非供給）。
- latent 側の指標には `_diag_gt_latent` も要る。これは最初のサンプルプロンプトの
  condition/reference 画像から埋まるため、それを持たない run では出ない。
- **`diag_trajectory_gap` は名前どおりのものを測っていない**: 単発 x̂₀ は学習バッチの
  最終マイクロバッチ、通し生成はサンプルプロンプトのもので、被写体が別。差は内容差に支配される。
  1-2 の表の「A が軌跡固有かの切り分け」はこの指標では成立しない。

**変更ファイル**: `base_trainer.py`（フック）, `metric_registry.py`（定義）, 新規 1 ファイル（統計計算）
**パラメータ**: 有効化フラグと間隔のみ。`param_defaults.py::TRAINING_DEFAULTS` → Pydantic → フロントの順で追加
**検証**: 3 step smoke（有限値が出ること、学習が止まらないこと）。収束実験は行わない
**コミット**: 1-1/1-2 で 2 件程度

---

## ✅ Phase 2 — G-C: crop decode の妥当性測定（測定完了・合格）

**目的**: 案 1' が成立するかを、**実装前に**測る。

### 2-1. autograd 対応の context crop decode（実装完了）

- 実装: `core/training/ops/crop_decode.py`（`make_crop_rect`, `decode_crop_with_context`）
- 幾何は `context_tiled_decode.iter_tiles` / `TileRect` をそのまま再利用
- **VAE パラメータは凍結。しかし入力 latent への autograd は維持**（`p.requires_grad=False` でも `latent` に勾配伝播）
- 内側領域のみ loss 対象。margin は捨てる

### 2-2. 測定 probe（実装完了・測定結果）

- 実装: `backend/core/training/probes/probe_crop_decode_autograd.py`
- 単体テスト: `backend/tests/crop_decode_autograd_test.py`（4 passed）

**測定結果（SD1.5 実 VAE: 512x512 canvas, 16x16 cells = 128x128 px interior）**:

| Margin (cells) | Window (cells) | MAE (/255) | Max (/255) | Grad Cosine | Time % | Gate G-C 判定 |
|---:|:---:|---:|---:|---:|---:|:---:|
| 0 | 16x16 | 5.30 | 55.84 | 0.71047 | 6.4% | FAIL（境界不連続性） |
| 4 | 24x24 | 1.56 | 30.87 | 0.90790 | 14.5% | FAIL |
| 8 | 32x32 | 0.96 | 16.74 | 0.95011 | 23.6% | **PASS** |
| 12 | 40x40 | 0.73 | 15.01 | 0.98248 | 36.6% | **PASS** |
| 16 | 48x48 | 0.41 | 8.88 | **0.98886** | 54.5% | **PASS** |

**ゲート G-C 判定結果: 合格 (ACCEPTED)**
1. **勾配方向の一致**: $k = 16$ で $\cos = 0.98886 \ge 0.95$（$k=8$ でも $0.95011$ で到達）。
2. **受容野境界誤差の減衰**: MAE は $5.30 \to 0.41$（減衰比 $0.078 \le 0.10$）。残存 $0.41 /255$ は GroupNorm 統計差による非局所成分だが、勾配方向への影響は微小（cos 0.989）。
3. **計算費用比率**: $k = 16$ で全体 decode の $54.5\%$（$1024\times 1024$ での $256\times 256$ crop では面積比 $25\%$ となりさらに削減）。
4. **推奨運用パラメータ**: `margin_cells = 16`（厳密性重視）または `8`（高速性重視）。
   → **この 4 番は下記の再測定で撤回された。`8` は本番作用点では FAIL する。**

**結論**: 案 1'（context crop decode loss）は勾配方向・値の精度・計算効率のすべてにおいて成立することを確認。Phase 4 への分岐は不要とし、**Phase 3（opt-in 補助損失の実装）へ進行可能**。

#### 再測定（本番作用点、2026-09-08）

初回の sweep は `torch.randn` を decoder に直接与えており、振幅が **正規化済み latent の側**
（σ=1）だった。本番の decoder 入力は denormalize 後（SD1.5/SDXL なら σ≈5.5）である。
probe に `--latent-amplitude production`（`scaling_factor` で割る）を追加して再測定した。
条件は初回と同一: SD1.5 実 VAE / 512px canvas / 内側 16 セル / cuda fp32。

| Margin | Window | MAE (/255) | Grad Cosine | Time % | 判定 |
|---:|:---:|---:|---:|---:|:---:|
| 0 | 16x16 | 26.67 | 0.55237 | 7.8% | FAIL |
| 4 | 24x24 | 7.52 | 0.61298 | 13.7% | FAIL |
| 8 | 32x32 | 6.49 | **0.80720** | 20.5% | **FAIL** |
| 12 | 40x40 | 7.12 | **0.90213** | 34.8% | **FAIL** |
| **16** | 48x48 | **2.37** | **0.99287** | 55.5% | **PASS** |

**ゲート G-C は margin 16 で合格のまま**（cos 0.99287、減衰比 0.0890、費用 55.5%）。

**ただし margin 8 は本番作用点では使えない。** 上の初回表（σ=1）では 8 が cos 0.95011 で
PASS だったが、本番振幅では **0.80720** に落ちる。「高速性重視なら 8」という初回の推奨は
撤回する。**採用できる margin は 16 のみ**。

初回表を σ=1 で再現した対照（`--latent-amplitude unit`）は MAE 5.41 / cos 0.98784 で、
元の 5.30 / 0.98886 と一致する。したがって差は振幅由来であって probe の変更由来ではない。

**性質の違い**: 勾配方向の一致は振幅に頑健ではなかった。受容野の議論から
「margin さえ確保すれば振幅に依らない」と予想していたが、実測はそれを否定している。
decoder は非線形（GroupNorm の空間統計と mid-block attention）なので、
作用点が変われば境界の寄与も変わる。

---

## ✅ Phase 3 — 案 1' を opt-in loss として実装（実装・検証完了）

**目的**: 症状 B に対する候補を、**既定 off の opt-in** として入れる。採用決定ではない。

### 3-1. パラメータ（CLAUDE.md の必須順序に従う） [完了: `3fbb31c3`]

`param_defaults.py::TRAINING_DEFAULTS` → OpenAPI → Pydantic → フロント。**キー数は最小に抑える**:

| キー | 既定 | 内容 |
|---|---|---|
| `crop_decode_loss_enable` | `false` | 有効化 |
| `crop_decode_loss_weight` | `0.0` | 主 loss に対する重み |
| `crop_decode_loss_margin_cells` | `16` | Gate G-C 測定結果より 16 |
| `crop_decode_loss_out_cells` | `32` | 内側領域のサイズ（latent 32 = pixel 256） |
| `crop_decode_loss_metric` | `lpips` | `VaeLossBank` の項から選択 |
| `crop_decode_loss_snr_range` | `""` | SNR 帯域（例: `-5.0,5.0`）。空文字で全帯域 |

### 3-2. 損失経路 [完了: `216a96e4`]

```
model_out ─ predict_x0(Phase 0-2) ─ crop 選択 ─ decode_crop_with_context(Phase 2-1)
                                                      │
GT latent ────────────────────────────────────────────┤
                                                      ▼
                                        VaeLossBank の指標（frozen, graph 保持）
                                                      ▼
                                        total_loss += w * metric
```

- `backend/core/training/ops/crop_decode_loss.py` に `CropDecodeLossModule` および `compute_crop_decode_loss` を実装。
- crop 位置はステップごとに一様ランダム。
- `VaeLossBank` の LPIPS/MSE/L1 指標を frozen/eval かつ autograd graph 保持で評価。
- 消費するアーキは `ArchHandler.consumes_crop_decode_loss` が宣言し、`base_trainer.py::crop_decode_loss_is_consumed()`
  が唯一の判定経路（`032cf977`）。現在 True が 9 件（SD1.5 / SDXL / Z-Image / FLUX.2 / Anima / Lens /
  Krea 2 / Ideogram 4 / SenseNova U1.5）、False が 4 件（ACE-Step / LTX-2.3 / MiniT2I / MiniMax-H3）。
  **ControlNet 学習は全アーキで非消費**（forward が `train_step_controlnet`）。
  非消費の run で weight を設定した場合は drop として警告される。

### 3-3. 係数決定を測定可能にする [完了: `216a96e4`]

- 主 loss と補助 loss の勾配ノルム比 `crop_decode_grad_norm_ratio` を `torch.autograd.grad` でモデル出力層勾配から算出し、`log_extra_metric` に記録。
- `crop_decode_loss` の生値も同時に `log_extra_metric` に出力。

### 3-4. 検証 [完了: `216a96e4`]

- `backend/tests/crop_decode_loss_smoke_test.py` 作成・実行:
  - 3-step smoke test: 有限 loss、`p.grad` への勾配到達（非ゼロ・有限値）、勾配ノルム比の有限性を確認。
  - fp32, fp16, bf16 の 3 種すべてで完全パス。
  - SNR 帯域フィルタリングの正常動作を確認。
- 実インポート確認: `core.training.base_trainer`, `core.training.ops.sd_sdxl_ops`, `core.training.ops.crop_decode_loss` （CUDA 未初期化）。
- 回帰テスト: `lr_group_schedules_and_layer_decay_test.py` 105 件全パス。

**コミット**:
- `3fbb31c3`: `Add Phase 3-1 crop decode loss configuration and parameter wiring`
- `216a96e4`: `Add Phase 3-2 and 3-3 crop decode loss pathway and grad ratio diagnostic`

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

### ✅ 5-1. G-B（測定完了・不合格。GPU・学習 A/B とも不要だった）

当初は**教師の前処理だけを変える対照実験**を計画していた（**実行していない**。下の判定で不要になった）:

- 固定: モデル・データ・seed・その他すべて
- 変数: 教師画像の前処理（正方形 squash vs アスペクト保持）
- 測定: **非正方形バケツ**での生成品質。
  **教師が異なる系の REPA loss の大小だけでは判定しない**（設計書の指示）

**空間対応は測定対象から外した（解決済み・2026-09-09）。** 教師画像は
`base_trainer.py::_get_repa_pixels_for_item` が S×S へ squash し（アスペクト破壊）、
教師特徴グリッドは `repa.py::encode_repa_targets` が student のトークングリッド (gh, gw) へ
bilinear（`align_corners=False`）で引き伸ばす。この 2 段は逆変換の関係にある:
student トークン列 j の参照元座標は `(j+0.5)*g/gw - 0.5`、元画像の横位置 `(j+0.5)/gw` を
squash した先の座標も同じ式になり、圧縮と拡大が厳密に相殺する。行方向も同様で、
アスペクト比に依存しない。**対応はずれていない。**

これにより G-B に残る仮説は 1 つになった:
**歪んだ画像の上で教師の特徴そのものが劣化しているか**。これが次項で決着した。

#### ゲート G-B 判定結果: 不合格 (REJECTED)（2026-09-09）

**判定は教師 encoder の processor 設定の読み取りだけで確定し、学習 A/B も GPU も使っていない。**
G-B の仮説「正方形 squash が教師特徴を劣化させる」は、教師自身が正方形 384² で学習されている
という事実に反証された。**REPA の squash は教師の学習ドメイン外ではなく、学習ドメインそのもの。**

再現に必要な設定値（すべて実ファイルから確認）:

| 項目 | 値 | 出所 |
|---|---|---|
| 既定教師の選択規則 | `tagger_models/*/` のうち `base_model_metadata.json` を持ち safetensors mtime が最大のディレクトリ | `base_trainer.py:3168-3183` `_discover_default_tagger_dir` |
| 上記が選ぶディレクトリ（現時点） | `tagger_models/cca72ce1-7420-4164-9f24-c30ae77cdf2f/` (mtime 2026-06-21) | 同上 |
| その `vision_encoder_repo` | `google/siglip2-so400m-patch14-384` | `cca72ce1-.../base_model_metadata.json` |
| naflex か | `"is_naflex": false` | `cca72ce1-.../best_f1_metadata.json` |
| processor 種別 | `"image_processor_type": "SiglipImageProcessor"` | HF キャッシュ `models--google--siglip2-so400m-patch14-384/.../preprocessor_config.json` |
| 教師の入力形状 | `"size": {"height": 384, "width": 384}`。`do_center_crop` / `crop_size` キーなし = 正方形へのリサイズのみ | 同上 |
| 教師の正規化 | `image_mean = image_std = [0.5,0.5,0.5]` = REPA が渡す `[-1,1]` と一致 | 同上（`repa.py:146-155` の docstring と同じ） |
| 空間対応 | 上記の半ピクセル導出のとおり厳密 | `repa.py:181-185`, `base_trainer.py:3394` |

**帰結（5-1 冒頭の規定に従う）**: 品質側の動機は消える。**5-2 の根拠は現在コスト削減のみ。**
これは 5-2 の棄却ではなく、根拠の変更である。

#### G-B の結論が及ばない範囲（残存条件 2 件）

1. **固定解像度系の教師についての結論である。** NaFlex 系の教師を
   `repa_tagger_model_dir` で明示指定した場合は別で、本物の不一致になる。
   `tagger_models/048122a9-e381-4c16-b221-1ddb8c96f92a/` は
   `google/siglip2-so400m-patch16-naflex`。その `preprocessor_config.json` は
   **`size` を持たず** `max_num_patches: 256` / `patch_size: 16` でアスペクト保持の
   可変グリッドを取る。さらに naflex の vision config に `image_size` が無いため
   `load_repa_encoder` の `native_size` は `None`（`repa.py:100-102, 136-137`）となり、
   `repa_size` は `base_trainer.py:3229-3230` の `(native or 384)` で **384 に落ちる**。
   結果、アスペクト保持で学習されたモデルに固定正方形を、256 patch で学習された
   モデルに 24×24 = 576 token を渡していた。**`2e3ffb64` でこの経路は `_setup_repa` の時点で
   拒否されるようになった**（`Siglip2VisionModel.forward` は `pixel_attention_mask` /
   `spatial_shapes` も必須引数で、そもそも呼び出しが成立しないため）。
   したがって G-B の判定範囲外の教師は現在そもそも走らせられない。
2. **リサイズフィルタが一致していない。** REPA 側は `base_trainer.py:3394` で
   `Image.BICUBIC`、教師の processor は `"resample": 2`（PIL BILINEAR）。
   **差の大きさは未測定。**（これも別作業で判断中。）

### 5-2. latent stem 蒸留（G-B 合格時）

- `(latent, 対応画像)` から pixel tagger の patch 特徴を蒸留
- **latent セルと DiT トークンの対応を明示的に定義する**（1 対 1 を仮定しない）。
  patch size / packing / 圧縮率 / crop・flip の対応を設計に書く
- 凍結 trunk のどの層に stem を接続するか、位置埋め込みとグリッドの扱いを決める
- **根拠はコスト削減のみ**（G-B 不合格により品質側の動機は消えた。5-1 の判定結果を参照）。
  「歪んだ教師をそのまま蒸留しても前処理問題は解決しない」という当初の前提は、
  固定解像度系の教師については前処理問題が存在しないため適用されない
- **ゲート G-A**: 保留 patch cosine（目安 ≥0.9）**に加えて**、
  固定予算・複数 seed の学習 A/B で生成品質・収束・時間・VRAM を比較。
  特徴類似度だけで品質維持を保証しない

**スコープ注意**: 「REPA は MiniT2I 専用」は 2026-09-08 の横断化で解消済み。対応範囲は
`ArchHandler.repa_tap()`（`arch/base_arch.py`）を実装したアーキで、現在 8 件:
MiniT2I / Anima / Lens / Krea 2 / Ideogram 4 / SenseNova U1.5 / SD1.5 / SDXL。
未実装アーキでは `repa_enable` は**無視されず拒否**される（`repa.py::refuse_repa`）。
未配線のまま保留している Z-Image / FLUX.2 の理由は設計書 §2 の表が持つ。

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
| 補助 loss / 診断のアーキ別の可否は、列挙ではなく `ArchHandler` の**宣言**（`consumes_crop_decode_loss`, `supplies_predicted_latent`, `repa_tap`）で固定し、真値を各 ops モジュールから **AST で導出するテスト**で縛る | 表示と実際の消費が食い違った前例（`032cf977`, `860d196c`） |

---

## 3. 実行順序（推奨）

| 順 | 作業 | 依存 | 中止条件 |
|---|---|---|---|
| 1 | Phase 0-1 設定・キャッシュ監査 | なし | — |
| 2 | Phase 0-2 x̂₀ ヘルパ + t 規約 | なし | — |
| 3 | Phase 1 診断計測 | 0-2 | — |
| 4 | ✅ Phase 5-1 G-B（測定完了・**不合格**） | なし | 判定済み。帰結は 5-1 の規定（5-2 の根拠はコスト削減のみ）による |
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
