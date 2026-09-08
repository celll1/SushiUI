# Phase 1 診断計測: SenseNova + 差し替え 4ch VAE の実条件

**対象プラン: [CONVERGENCE_ACCELERATION_PLAN.md](CONVERGENCE_ACCELERATION_PLAN.md) の Phase 1**
**状態: ブリーフ（実装未着手）**  日付: 2026-09-08

プランの Phase 1 は汎用の診断として書かれていた。対象ランが
`sensenova_fullft_sdxlvae_v1`（`training.db` id=127）に確定したので、その実条件に落とす。
**「4ch という一致だけで SDXL の較正を流用しない」**というオーナー指示に従い、
以下はすべて実装から読んだ値である。

---

## 1. 確定した実条件

| 項目 | 値 | 出所 |
|---|---|---|
| VAE | **madebyollin/sdxl-vae-fp16-fix**（`registry:sdxl` の解決先） | `models/common/vae_store.py:84-100` |
| 空間圧縮率 | **8**（宣言ではなく `observe_vae` の観測結果。SenseNova は比率チェック免除 `_PIXEL_SPACE_EXEMPT`） | `vae_source.py:312-356`, `:906` |
| チャネル | **4**。起動時に 64×64 ダミーで実測検証済み（`strict_validation: true`） | `vae_swap.py:338-381` |
| 正規化 | **`shift_scale`、spec 経由**（`trainer.wiring`）。`sensenova_ops` は spec を渡している | `ops/sensenova_ops.py:1853, 1937, 2402` |
| 正規化定数 | `scaling_factor = 0.13025`、`shift_factor = None → 0.0` | snapshot `config.json` |
| `gen_patch` | **8**（native 4）。1 トークン = **8×8 = 64 latent セル = 64px 四方** | `models/sensenova/latent_space.py:110-144` |
| latent セル ⇔ トークン | **1 対 1 ではない**（1:64） | 同上 |
| 予測形式 | **モデルは x0 を直接出す**。損失ターゲットのみ velocity | `ops/sensenova_ops.py:2167-2176` |
| t の向き | **t=1 が clean、t=0 が純ノイズ** | `ops/sensenova_ops.py:2101`, `arch/sensenova.py:52` |
| 損失重み | `loss = recon_loss / max(1-t, 0.05)²`、**上限 400** | `ops/sensenova_ops.py:2180-2184` |
| latent キャッシュ | **不使用**（`onthefly_gpu`） | 実ログ |

### noise_scale の式（σ の項は存在しない）

```
noise_scale = min( sqrt(N_tokens / 64) · transformer.noise_scale , 16.0 )
```
`sensenova_pipeline_ops.py:177-188`。**トークン数の項しかなく、データスケール σ の項は式に無い**
（`RMS(x0)` は定数に焼き込み済み。`latent_space.py:319-332`）。

`sensenova_noise_scale_gain` は式の**外側の定数** `transformer.noise_scale` に掛かる
（`latent_space.py:335-371`）。つまり **gain は σ のずれだけを補い、patch=8 による
トークン数減は補わない**。ログの `NEITHER is recalibrated` はこの二重性を指している。

| 条件 | トークン数 | noise_scale |
|---|---|---|
| このckptの校正帯（3–5MP @ 32px/token） | 2930–4883 | **6.77–8.74** |
| このランの学習バケット | 496–576 | **2.78–3.00** |
| gain 1.4616 適用後（未実行） | 同上 | 4.07–4.39 |
| **サンプル生成**（896×1536 に snap） | **336** | **2.29** |

---

## 2. Phase 1 の前に潰すべきこと（優先順）

### F-1【最優先】step 12,335 の未クリップ勾配爆発。モデルは回復していない

DB 実測（run 127）:

```
step 12334  loss 2.16  recon 0.447  grad_norm    4.5
step 12335  loss 7.65  recon 6.605  grad_norm 5216.2   ← 単一ステップ
step 12336  loss 1.03  recon 0.467  grad_norm    4.5
```
1000 step ビンでの `recon_loss`: 4000 台 **0.445**（底）→ 11000 台 0.461 →
**12000 台 0.838** → 6,400 step 後の 18000 台でも **0.693**。**底に戻っていない。**
同ビンの `grad_norm` 最大は **17,334.5**。

`max_grad_norm: 1.0` は fused backward pass 下で**無視されている**
（`base_trainer.py:7506-7545`, 実ログの警告）。17,334 の更新がそのまま重みに入った。

**この事象はリポジトリが既に名指しで記録している**:
`core/training/grad_spike_log.py:5-7` が「run127 は step 12,355 で中央値 3.8 に対し
global norm 17,334 に達し、**never recovered**」と書いている。

現行 config には `fused_grad_clip_factor` のキーが無く、`TRAINING_DEFAULTS` の
既定は **0.0 = クリップ無効**（`api/param_defaults.py:2192`）。**このまま再開すれば再発しうる。**

**Phase 1 への含意**: 後半 6,400 step は**損傷後の状態を測っている**。
劣化前（〜11,000）と劣化後（12,400〜）を混ぜて解釈してはならない。

#### 実測（プラン §4 の解析。学習は回していない）

**単発ではなくバーストだった。** `grad_norm > 100` は 3,300〜12,246 の 8,900 step で **0 件**、
そこから 12,247〜12,370 の 124 step に集中する:

```
12247   172     12343  2670     12348   376
12335  5216     12344   498     12355 17334   ← 最大
12340   180     12346   301     12370   987
```
resume 境界は step 5479 / 9304 / 9534 で、**バーストのどれとも一致しない**（resume 由来ではない）。

**セグメント別分布**

| 区間 | n | recon_loss p10 / 中央 / p90 / 平均 | grad_norm 中央 / p99 / 最大 |
|---|---|---|---|
| early 1–3,999 | 3,999 | 0.241 / 0.618 / 0.916 / 0.622 | 9.70 / 173.7 / 1537.5 |
| **healthy 4,000–12,246** | 8,230 | **0.348 / 0.464 / 0.598 / 0.466** | **5.29 / 21.8 / 60.9** |
| burst 12,247–12,370 | 124 | 0.371 / 0.496 / 1.283 / 0.819 | 3.78 / 4630.6 / 17334.5 |
| post 12,371–18,777 | 6,407 | 0.560 / 0.731 / 0.995 / 0.756 | 5.53 / 27.0 / 100.6 |

**勾配は回復し、損失は回復していない。** post の grad_norm 分布は healthy とほぼ同じ
（中央 5.53 対 5.29、p99 27.0 対 21.8）。モデルは不安定なのではなく、**より悪い盆地に安定して座っている**。

**回復は頭打ち。** 1000 step 平均: 0.937 → 0.777 → 0.729 → 0.714 → 0.694 → 0.704 → 0.690。
14,400 以降 5 ビンの回帰勾配は **−0.0087 / 1000 step**。この率で healthy 平均 0.466 に戻るには
**あと約 26,000 step** かかる計算になり、しかも曲線は平坦化している。

**Phase 1 の基準線は healthy 窓 4,000–12,246 から取る。** post の 6,407 step は基準線から除外する。

### F-2 `prediction_target: velocity` は実装と食い違う

config・DB・`trainer.prediction_target`・起動ログはすべて `velocity` を主張するが、
`ops/sensenova_ops.py` と `arch/sensenova.py` は `prediction_target` / `noise_process` を
**一度も読まない**。ネットワークの出力は x0。

**`ops/x0_recovery.predict_x0` に SenseNova を通してはならない。** 黙って誤る。
同モジュールが SenseNova を明示除外しているのは正しく、専用ヘルパを足す理由もない
（`x0_pred` がそのまま x̂₀ である）。

### F-3 サンプル解像度が学習バケット帯の外

sample 設定 864×1536 は 64 の倍数でないため 896×1536 に snap され、**336 トークン /
noise_scale 2.29**。学習は 496–576 トークン / 2.78–3.00。
**この条件のまま「単発予測 vs 通し生成」を撮っても、差が軌跡由来か帯域由来か分離できない。**
sample を 1216×1728 か 1536×1536 に変えれば揃う。

### F-4 gain 1.4616 は一度も実行されておらず、かつ実測値ではない

- 全 8 ログに再較正行 `generation noise scale recalibrated` が**1 件も無い**
- `debug/step_018700/latents_t0.3014.pt` の `noise_scale = 3.0` は `transformer.noise_scale = 1.0` を意味する（gain が効いていれば 4.385）
- config の `.bak-20260907-114655` から、gain は**最後の実行セグメントより後に追加された**
- `sensenova_noise_scale_auto: false` なので `measure_latent_rms` は走らない。
  1.4616 は「latent RMS = 1.0000」を仮定した値（1.4616 × 0.6842 ≒ 1.000）であって**実測ではない**

実測経路は既にある（`ops/sensenova_ops.py:1864-1925`）。

---

## 3. 既に存在するもの（作らない）

プランが Phase 1 で作ろうとしていた指標の一部は既にある。

| 既存 | 実体 |
|---|---|
| **`recon_loss` 専用 DB カラム** | `MSE(x0_pred, x0)` = **重みなしの x̂₀ 誤差そのもの**。run 127 の全 18,760 step に存在 |
| **真の VAE decode 画像** | `debug/step_*/decode_t*_{noisy,target,pred_x0}.webp` を**実 VAE で**書いている（`ops/sensenova_ops.py:1999-2010`）。188 step 分がディスク上にある |
| **GT roundtrip** | 上記 `_target.webp` が `decode(encode(GT))` そのもの。**新規実装は不要** |
| encode/decode 経路 | `sensenova_ops.vae_encode` / `vae_decode` |
| データセット標本ループ | `measure_latent_rms`（`:1864-1925`）を GT roundtrip にも流用可 |

**プランのガードレール記述を 1 つ訂正する。** 「debug latent 表示は min/max 引き伸ばしで
VAE デコードをしない」は他アーキの話で、**SenseNova には当てはまらない**。
`routes.py:17961-17978` は webp を優先し、`latent_to_image` には到達しない。

### 報告されている `loss` は症状 A に構造的に鈍い

t 分布 logit-normal(0, 1.4) と重み `1/max(1-t,0.05)²` のモンテカルロ（4M サンプル）:
`P(t>0.95) = 1.77%`、`E[w] = 24.3`、`median w = 4.00`。
**チャートの `loss` の重みの約 61% は t>0.90 の 10% のサンプルから来ている**
（t=1 が clean 側なので、これは**低ノイズ側専用の指標**）。
高ノイズ帯の学習不足＝症状 A を見るには `recon_loss` の方が中立。

---

## 4. Phase 1 で測るもの

### 4-A 追加コードゼロ（過去データに遡及可能）

- `recon_loss` / `loss` / `grad_norm` の時系列を **F-1 の前後で分けて**読む
- `cfg_guidance_rel` / `cfg_guidance_cos`（既存 extra_metrics）
- `debug/step_*/decode_t*_{target,pred_x0}.webp` の目視と大域画素統計
  （**webp quality 80 の非可逆圧縮が乗るので周期解析には使えない**）
- `samples/step_*_sample_0.png`（ただし F-3 の帯域ずれつき）

### 4-B 小さな追加で測れるもの

| 指標 | 追加箇所 | 対照 |
|---|---|---|
| latent チャネル別 mean/std（x0 と x0_pred、4ch） | `_save_pixel_debug` の呼び出し側 | 同 step の GT latent |
| **t 帯別 `recon_loss`** | `log_extra_metric` | 自身の時系列。**SNR ではなく t で分けてよい**（t が線形混合係数そのもの） |
| **トークン数別 `recon_loss`** | `log_extra_metric("tokens", N)` | バケットが 496–576 に散るので交絡を分離できる |
| `noise_scale` の実値 | `log_extra_metric` | 校正帯 6.77–8.74 |
| GT roundtrip の latent 側誤差 | 既存 2 関数 | — |

### 4-C 測定条件の既知差分（受け入れた上で測る）

**noise_scale は校正帯の外にある。** gain 1.4616 を適用しても学習帯 496–576 トークンで
**4.07–4.38、校正帯 6.77–8.74 の下端の約 60%**（embedder 入力では 0.254–0.274 対 0.42–0.55）。

gain は式の外側の定数に掛かるので σ のずれだけを補い、`patch=8` によるトークン数減
（`sqrt(N/64)` 項）は補わない。**これは patch=8 を選んだことの帰結として受け入れる。**
帯外であることが実際に品質へ効いているかどうかを、Phase 1 の測定対象とする。

### 4-D 対照条件の注意（このラン固有）

- **通し生成を学習と同じトークン数で撮る**（F-3）。揃えるまで単発 vs 通しの差は解釈できない
- **ブロック周期の候補が 2 つある**: **8px（VAE セル）と 64px（トークン境界）**。
  設計書の「8px 周期」は VAE 側だけを指しており、**SenseNova ではトークン境界 64px の方が新規リスク**。
  **測定項目: 8px と 64px の両方を FFT で分けて見る。** 一方のピークをもう一方に帰属させない。
  正常な模様も周期に反応するため、周期ずらし・非周期ノイズ・正常画像の対照も要る
- `.pt` に latent テンソルが無いため、**過去 188 step に遡って周期解析はできない**

---

## 5. 測れないもの

1. **gain 1.4616 下の挙動**（一度も実行されていない）
2. **step 18,782 の状態**（保存が 1/9 シャードで中断、ファイルは削除済み。再開可能な最新は step 9,533）
3. **過去の latent 周期パワー・チャネル統計**（`.pt` に入っていない）
4. `gnorm_null` / `loss_null` は `cfg_null_frac` が 0 か 1 の step のみ（batch 4 の混在 step では欠測）

プラン Phase 0-1 の **latent キャッシュ監査項目はこのランについては空振り**である
（`onthefly_gpu`、全 dataset `cache_latents_to_disk: false`）。

---

## 6. 実装順序

| 順 | 作業 | 理由 |
|---|---|---|
| 1 | **F-1: `fused_grad_clip_factor` を有効化して再開する**か、少なくとも `grad_spike_log` を有効にする | 再発すれば以降の測定がすべて無意味になる |
| 2 | **F-4: `measure_latent_rms` を 1 度走らせて gain を実測値に置き換える** | 仮定値のまま測っても較正の議論ができない |
| 3 | **F-3: sample 解像度を学習バケット帯に揃える** | 揃うまで単発 vs 通しは比較不能 |
| 4 | 4-A を読む（コード変更ゼロ、F-1 前後で分割） | 既存データだけで判断材料が出る |
| 5 | 4-B を実装（既定 off、`log_extra_metric`） | — |

**1〜3 は Phase 1 の測定条件を整えるための前提であって、Phase 1 の成果物ではない。**

---

## 7. この文書が答えないこと

- 症状 A/B がこのランで実際に再現するか（4-A を読んで初めて分かる）
- F-1 の損傷が回復可能か、step 9,533 からやり直すべきか
- patch=8 を native 4 に戻すべきか（較正帯には近づくが、コストが 4 倍になる）
