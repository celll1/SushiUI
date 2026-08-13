# MiniMax H3 継続・追加学習設計案

> Status: design only（実装前）
> Date: 2026-08-14（rev.2 — 現行コード監査を反映）
> Scope: MiniMax H3 LoRA training

> rev.2 の変更点: 現行コードの静的調査により、初版が前提としていた「unconditional probe」「block swapとの併用」「非有限検出機構」「audio cache key構成」「LoRA profile の module 分離」が実装上成立しないことが判明した。これに伴い CGL を既定目的関数から opt-in 実験へ降格し、retention を前倒しした。本文中の file:line は調査時点のもの。

## 1. Executive decision

MiniMax H3 の元 teacher は公開されておらず、SushiUI に DPO 用データもない。このため、初期設計では teacher matching と DPO/PSO を必須にしない。

採用する基本方針は次の通り。

1. 既定の目的関数は現行の flow-matching velocity target（modality mask 付き）に、LoRA を無効化した公開 H3 base を参照する function-space retention を加えたものとする。
2. CGL（guidance-distillation target）は opt-in の実験項目とする。§5 の null-conditioning 妥当性ゲートを通過するまで既定にしない。
3. video、video-only、image-only、audio-only を modality mask 付きで扱う。
4. audio-only は native zero-video packed layout を確定路線とし、dummy video scaffold は forward プローブが失敗した場合の保険に留める。
5. H3 の shared DiT block による cross-modal contamination を、sampling、retention、adapter 運用単位、診断値で抑制する。

推奨する初期目的関数は次の通り。

\[
L = L_{\mathrm{FM}} + \lambda_{\mathrm{keep}}L_{\mathrm{base\_keep}}
\]

ここで `L_FM` は現行実装の velocity MSE に modality mask を掛けたもの、`base_keep` は元 teacher ではなく、LoRA を無効化した公開 H3 base の出力保持である。LoRA parameter L2 \(L_{\mathrm{param}}\) は補助で既定 0 とする。CGL 実験時のみ \(L_{\mathrm{FM}}\) を \(L_{\mathrm{CGL}}\) に置き換える。

今回は実装・ファイル変更を行わない。

### 1.1 CGL を既定にしない理由

H3 は既に guidance-distilled であり、その出力は元 teacher の \(u_T + s_T(c_T-u_T)\) の近似である。ここへ「モデル自身の null-condition 出力からの外挿」を教師として課すと、蒸留済み出力への guidance 二重掛けと、後述する OOD anchor という二重の不定性が入る。musubi/AI Toolkit の当該実装は参照するが、目的関数を公開コードから完全には確認できず、plain flow matching より優れるという根拠は SushiUI 側に存在しない。現行コード [`minimax_h3_ops.py`](../../backend/core/training/ops/minimax_h3_ops.py) の `:860-867` にも「CGL/weighting を勝手に発明しない」方針が既にコメントとして残っている。

蒸留済みモデルの継続学習で理論的に正当化できるのは base からの function-space 逸脱の制御であり、それは §8 の retention に相当する。したがって retention を先に入れ、CGL は §12.1 の `s=1` 退行テストを安全網として実験する。

## 2. Constraints and non-goals

### Constraints

- H3 の元 teacher/checkpoint は公開されていない。
- DPO 用の pairwise preference data は存在しない。
- H3 は video/audio joint の single-stream DiT で、内部 block は modality 別に分離されていない。
- H3 は CFG-distilled で、推論時に通常の CFG や negative prompt を使わない。パイプラインに unconditional 経路は存在せず、空プロンプトは拒否される。
- H3 は 33B dense DiT、weight-only FP8 base、single-GPU 前提であり、full fine-tune は対象外である。
- 現行学習は batch size 1 を前提とする。
- H3 の training preview は未実装で、学習後に通常の generation path で検証する。
- 追加の no_grad forward は block swap と併用できない（§5）。
- text encoder は 1 prompt あたり 13.5 秒、RSS 49.82 GB を要し、学習中も CPU 常駐で移動禁止である。毎 step の encode は不可能で、事前キャッシュが必須となる。

### Non-goals

- 元 teacher の再現や厳密な teacher matching を初期リリースの前提にしない。
- DPO、Diffusion-DPO、PSO を初期 loss に含めない。
- H3 の full-parameter fine-tuning を設計しない。
- dummy modality を実データ教師として扱わない。
- LoRA の module scope による modality 分離を目指さない（§9）。

## 3. Current-state gap

現行の主な実装は次のファイルにある。

- [`minimax_h3_ops.py`](../../backend/core/training/ops/minimax_h3_ops.py)
- [`minimax_h3.py`](../../backend/core/training/arch/minimax_h3.py)
- [`h3_pipeline_ops.py`](../../backend/core/models/minimax_h3/h3_pipeline_ops.py)
- [`base_trainer.py`](../../backend/core/training/base_trainer.py)
- [`minimax_h3_adapter.py`](../../backend/core/training/adapters/minimax_h3_adapter.py)
- [`minimax_h3_training_test.py`](../../backend/tests/minimax_h3_training_test.py)

| データ種別 | 現行状態 | 設計上の扱い |
|---|---|---|
| video + audio | 対応 | video/audio の両方を学習 |
| video only | 対応 | video loss のみ。audio rows は構造用で loss mask |
| image only | 対応。T_lat<=1 では audio rows を 0 行にする（noise rows を作らない、`minimax_h3_ops.py:878-901`） | T=1 の 5D video latent。audio loss なし |
| audio only | 明示的に拒否 | standalone audio encode/cache と video-free layout を追加 |

現行の `train_step` は video latent を必須にし（`minimax_h3_ops.py:689-696`）、video loss を常に計算する（`:840`）。video 側の mask `m_v` も重み `λ_v` も存在せず、video 重みは 1.0 固定である。一方、音声がない video では audio rows を noise で構成し（`:732-743`、`:782-796`）、`audio_present` で audio loss を除外する（`:842-849`）。[`_refuse_unsupported_audio_only_items`](../../backend/core/training/base_trainer.py) は standalone audio item を現在拒否している（`base_trainer.py:6975-6995`、呼び出しは `:8616` の 1 箇所）。

sigma は 1 回の draw `u` から両 modality へ shift して作られる（`:766-775`）。符号規約は \(x_t=(1-\sigma)x_0+\sigma\epsilon\)、\(v=x_0-\epsilon\)、\(t=1-\sigma\) である（`:31-32`、`:779-780`、`:799`）。

`build_packed_layout` は `num_audio_latents=0` を扱えることが既存テストで確認されている（`minimax_h3_training_test.py:567-581`）。ただしこれは layout レベルの検証であり、実 transformer forward を通した zero-audio 検証は存在しない。次の設計課題は `num_latent_frames=0, num_audio_latents>0` である（§7）。

## 4. Modality data contract

各サンプルは、少なくとも次の論理情報を持つ。

```text
has_video
has_audio
video_latent
audio_latent
duration
fps
spatial_geometry
caption
modality_kind
```

現行の item dict はこのうち `_clip_audio_latent` と collate 後の `audio_present` しか持たない（`base_trainer.py:5091-5117`）。`has_video` / `modality_kind` は新規に導入する。

### video + audio

- `video_loss`: 有効
- `audio_loss`: 有効
- video/audio は同じ時間窓から encode する。
- CGL を有効にした場合は両 modality へ適用する。

### video only

- `video_loss`: 有効
- `audio_loss`: 無効
- 現行の noise audio rows は互換モードとして維持できる。
- zero-audio-row 方式を比較し、品質が同等以上ならそちらを推奨する。

zero-audio-row 方式は image-only で既に本番稼働しており、layout テストも通っている（`minimax_h3_training_test.py:567-581`）。したがって video-only への拡張はこの実績を根拠にできる。ただし実 transformer forward を通した zero-audio 検証は未実施であり、Phase 1 のプローブに含める。

noise rows は教師信号ではないが、single-stream attention を通じて video 出力へ影響し得る。したがって、audio loss をゼロにしただけでは完全な video-only とはみなさない。

### image only

- `video_loss`: 有効
- `audio_loss`: 無効
- image は T=1 の video latent として扱う。
- 現行実装では audio rows が 0 行になり、noise rows も作られない。
- temporal/audio generalization を保持するため、強い base retention を適用する。
- image-only 学習は appearance 用の adapter として別ファイルで管理することを推奨する（§9）。

### audio only

- `video_loss`: 無効
- `audio_loss`: 有効
- standalone audio VAE encode/cache を追加する。
- native zero-video packed layout を確定路線にする。

`has_video=false && has_audio=false` は入力エラーとして拒否する。現行にはこのチェックが存在しない（video latent 必須のため `has_video=false` 自体が表現不能）ため、新規に追加する。

## 5. CGL objective（opt-in 実験）

**前提注記**: H3 パイプラインに unconditional 経路は存在しない。`h3_pipeline_ops.py:1483-1484` の denoise docstring が「guidance-distilled なので unconditional branch は無い」と明記しており、`negative_prompt` / `guidance_scale` / `do_classifier_free_guidance` は pipeline backend にヒットしない。空プロンプトは `h3_pipeline_ops.py:1166-1173` で ValueError となる（text 行が 0 になると rotary clock が未検証パスに入るため）。したがって本節の \(u_r\) は新規定義する null conditioning への応答であり、蒸留時の unconditional branch を近似する保証はない。この妥当性検証を通過するまで CGL は実験扱いとする。

modality `r`（video または audio）の通常 target は、H3 の現行 convention と一致して、

\[
v_r = x_{0,r} - \epsilon_r
\]

である（`minimax_h3_ops.py:779-780`、`:799` で確認済み）。unconditional probe による anchor を \(u_r\) とし、CGL target は、

\[
y_r = u_r + s_{\mathrm{eff},r}(v_r-u_r)
\]

とする。

AI Toolkit 互換の schedule は、

\[
s_{\mathrm{eff},r}=1+(s_r-1)\sigma_r
\]

である。

H3 は video shift 12、audio shift 3 を持つため（`h3_pipeline_ops.py:102-103`）、設計上は実際に各 modality へ適用した post-shift sigma を使うことを推奨する。現行 `train_step` は pre-shift `u` と post-shift `sigma_v` / `sigma_a` の両方をローカル変数として保持しているため（`minimax_h3_ops.py:766-775`）、どちらの方式も数行で実装できる。pre-shift draw を使う方式は互換・ablation 用に残す。

scale 3.5 や 4.0 は既存ツールの実装例であり、H3 元 teacher が非公開である以上、正解値とはみなさない。calibration と ablation で決める。

### Null conditioning の定義

unconditional 経路が無いため、\(u_r\) を得る条件を新規に定義する。候補は次の 2 つ。

- **(a) 固定 caption**: 中立的な固定 caption を run 開始時に 1 回 encode してキャッシュする。text encoder は 13.5 秒/prompt、RSS 49.82 GB を要し学習中も CPU 常駐で移動禁止のため（`base_trainer.py:4523-4531`）、毎 step の encode はできない。
- **(b) ゼロ埋め embedding**: 条件 embeds と同一シーケンス長 `S` のゼロ埋め `[S, 5120]` を使う。packed layout と rotary が条件側と完全に一致し、text encoder の実行も不要である。

採用前に、\(u_r\) の RMS と \(v_r-u_r\) の step 間方向安定性を計測する。これが不安定なら CGL は採用しない。

### Anchor mode

#### live anchor

Musubi/AI Toolkit に近い方式。

- LoRA 有効状態の unconditional prediction
- no-grad
- 比較的安価
- 学習とともに anchor 自体が移動する

#### frozen-base anchor

SushiUI で推奨する方式。

- LoRA 無効の凍結 base で unconditional prediction
- anchor が学習中に移動しない
- base retention と同じ参照モデルを使える
- forward コストは増える

実装方式として、base の複製（19.71 GB）は不要である。H3 の LoRA wrapper は `forward = org_out + up * self.scale` であり `self.scale` は単なる float 属性で（`minimax_h3_adapter.py:89-93`）、全 wrapper が `trainer.lora_layers` に登録されている（`lora_trainer.py:85`、`:275`）。したがって `scale=0` への一時切り替えで base 出力が得られる。追加 VRAM はほぼゼロで、同一重み・同一 forward 経路を通るため bf16 の丸めに起因する比較ノイズも生じない。SushiUI には既存の `disable_adapter` 系機構が無く（PEFT 非使用）、他アーキにも frozen-base retention/teacher/EMA distillation の前例が無いため、この切り替え helper は本設計で新規に導入する。

初期の安定性を優先する場合は frozen-base anchor を標準候補とし、live anchor は互換比較用とする。

### Block swap 制約

追加の no_grad forward は block swap と併用できない。`LayerOffloadConductor` は `register_full_backward_hook` でのみ offload するため（`core/memory_management/layer_offload_conductor.py:352-358`）、no_grad forward は backward hook を発火させず、swap 対象ブロックが GPU に載ったまま残る。block swap の削減効果が消え、最悪 OOM になる。

したがって CGL（frozen_base）および base retention は `blocks_to_swap=0` でのみ許可し、`blocks_to_swap>0` との併用指定は起動時に ValueError とする。これは CLAUDE.md の Block Swap + 8bit optimizer 拒否と同型の扱いである。現行の計測値では 384x640x22 で peak 23.08 GB、512x768x39 で 25.63 GB（base 常駐 19.71 GB）であり、swap 無しで成立する解像度帯が存在するため当面は制約として運用できる。forward-hook ベースの offload 拡張は §16 の open question とする。

### Forward 回数とコスト

anchor（uncond）と retention（cond）は条件が異なるため 1 回の forward で兼用できない。CGL と retention を同時に有効化すると step あたり 3 forward になる。gradient checkpointing 下では 1 step がおよそ 4 回分の forward 相当であるため、推測: anchor 追加で +25%、retention も加えると +50% 程度になる。H3 の forward 単体の計測値はリポジトリに存在しないため、この見積りは Phase 1 の実測で置き換える。

既定構成（CGL 無効、retention を interval 実行）では追加は retention forward のみで、`h3_base_keep_interval=4` なら推測: +6% 程度に償却される。

## 6. Modality-aware loss

サンプルごとに、

\[
m_v,m_a\in\{0,1\}
\]

を持たせる。

\[
L_{\mathrm{FM}} =
m_v\lambda_v\operatorname{MSE}(\hat v,y_v)
+m_a\lambda_a\operatorname{MSE}(\hat a,y_a)
\]

現行実装では `m_v` と `λ_v` が存在せず video 重みは 1.0 固定である（`minimax_h3_ops.py:840`）。`λ_a` は `audio_loss_weight`（既定 1.0、[`param_defaults.py`](../../backend/api/param_defaults.py) `:2211-2230`）が担う。本設計では `m_v` を導入するが、`λ_v` は既定 1.0 固定として新規パラメータを追加しない（`audio_loss_weight` との非対称を維持する）。対称化が必要になった場合は別途 SSOT に追加する。

実際の平均は active rows だけで行う。

- video absent なら video target を作らない。
- audio absent なら audio target を作らない。
- dummy/noise rows を教師信号にしない。
- active modality だけが loss gradient を持つ。
- 両方 absent は事前に拒否する。

現行の平均は両 modality とも要素平均で row 数に依存しない（`minimax_h3_ops.py:837-839` にその設計意図がコメント済み）。ただし短尾ウィンドウの zero パディング行はサンプル平均に含まれている（`base_trainer.py:5107-5114`）。この扱いを維持するか除外するかは Phase 1 で決める。

shared block には active modality から gradient が流れるため、次を併用する。

- modality-balanced sampling
- modality ごとの loss normalization
- base function-space retention
- modality 別 gradient norm ログ（debug フラグ + interval 実行に限定、§11）
- image/audio-only 用の専用 adapter 運用単位（§9）
- paired replay の定期挿入

同一 batch は batch size 1 のため、異なる caption や modality を一 step 内に混在させない（`minimax_h3_ops.py:150-159`、`:708-716`）。dataset 間の mix は step 単位で行う。

## 7. Audio-only packed layout

### 確定路線: native zero-video layout

次の構造を追加する。

```text
num_latent_frames = 0
num_audio_latents > 0
```

静的解析により、必要な修正は次の 3 箇所と確定している。

1. `_temporal_position_grid`（`h3_pipeline_ops.py:196-202`）— n=0 のとき `cat([zeros(1), spans[:-1].cumsum(0)])` が長さ 1 を返す。長さ 0 を返すよう明示化する。
2. `unpatchify_video_rows`（`h3_pipeline_ops.py:158`）— `reshape(-1, 0, ...)` は既知次元に 0 を含むため -1 が曖昧になり RuntimeError となる。0 フレームの early-return を追加する。
3. 学習側 video loss（`minimax_h3_ops.py:840`、`:851`）— `F.mse_loss(empty, empty)` は mean reduction で NaN を返し、loss 全体を汚染する。audio 側の zero-branch（`:846-849`）と対称な video mask を入れる。

一方、次はコード上そのまま通る見込みである。

- `video_indices` が空になる（`h3_pipeline_ops.py:613-614`）。
- `audio_indices` は video frame 数に依存せず channel-major で正しい（`:601-603`、`:205-225`）。
- `build_row_timesteps` は空代入で問題ない（`:941-952`）。
- transformer の `proj_in` / `index_copy` / `index_select` は 0-length で合法（`transformer_minimax_h3.py:812`、`:819`、`:856-861`）。
- video output が `[B, 0, C]` になる。
- 形状 validation（`transformer_minimax_h3.py:797-804`）は modality 行数ゼロを弾かない。

制約として、`latent_height` / `latent_width` は 0 にできない（`h3_pipeline_ops.py:174-193` で sqrt_area=0 が nan になる）。zero-video でも有効な空間 geometry を渡す必要がある。

上記 3 箇所を修正したうえで、実 transformer forward を通したプローブを Phase 1 の exit criteria とする。

### Fallback: dummy video scaffold

修正後の forward プローブが失敗した場合のみ使う保険とする。native 側の修正範囲が 3 箇所と小さいことが判明したため、初版より優先度を下げる。

- audio duration から時間長を決める。
- 最小の valid spatial/temporal geometry を作る。
- video latent は固定 seed の normalized Gaussian または null latent とする。
- video loss は完全に mask する。
- main forward と unconditional anchor forward で同じ scaffold を使う。
- scaffold から教師 gradient を流さない。
- base retention を強くする。

scaffold 方式では transformer と `build_packed_layout` は無改造で済む。必要なのは拒否の緩和、standalone encode、scaffold latent 生成、video loss mask である。

これは本当の video-free 学習ではなく、dummy visual context 上の audio 学習である。native 方式とは別の quality gate で判定する。

## 8. Teacher-free retention

元 teacher の代わりに、LoRA を無効化した公開 H3 base を参照する。

\[
L_{\mathrm{base\_keep}}
=
\sum_r m_r\left\|f_\theta(x_t,c)-f_{\theta_0}(x_t,c)\right\|^2
\]

- `f_theta0` は凍結 base H3。実装は §5 と同じく全 LoRA wrapper の `scale=0` 一時切り替えで行い、base の複製 19.71 GB は不要である。
- base 側に gradient を流さない。
- latent、sigma、packed layout、modality presence を揃える。同一 step 内で評価するため構造的に自動一致する。
- retention forward は毎 step ではなく `h3_base_keep_interval` step ごとに実行する。
- active output だけでなく、定期的な paired replay では video/audio 両方を保持する。

LoRA parameter L2、

\[
L_{\mathrm{param}}=\|\Delta W\|^2
\]

は補助的に使えるが、function-space retention より弱いため中心には置かない。既定 0 とする。

### Replay

base retention 用には次の 2 種類を使う。

1. 学習データ上の同一 noising state。追加コストは retention forward のみ。
2. 学習開始前にオフラインで事前生成した on-policy 中間 state バッファ。固定 prompt 集合と固定 NFE で生成し、latent を disk cache に置く。

in-loop での on-policy 生成は禁止する。text encoder 13.5 秒/prompt と 20-step denoise 62 秒（`docs/guides/MODEL_FACTS.md:1082`）により、1 回の state 生成が十数 step 分の学習時間に相当し、single GPU の実運用では割に合わない。

これは preference 学習ではなく、蒸留済み base の挙動保持である。

## 9. LoRA and adapter policy

現行の LoRA scope は `attention,ff` の 2 グループのみで（`minimax_h3_adapter.py:51-67`、`lora_trainer.py:233-247`）、leaves は `to_q/to_k/to_v/to_out.0` と `net.0.proj/net.2`、全 50 block に一律適用される（`:134-166`）。`proj_in` / `audio_proj_in` / `proj_out` / `audio_proj_out` / `token_refiner` / AdaLN は恒久的に除外されており scope 文字列から到達できない（`:15-23`）。

したがって profile を module scope の分割として定義することはできない。本設計では profile を、**学習データ構成・retention 強度・保存 adapter ファイルの管理単位**として定義する。

| Profile | 主用途 | 差別化の実体 |
|---|---|---|
| `joint` | video+audio | paired データ、標準 retention |
| `appearance` | image-only | 静止画データ、強い retention、別ファイル保存 |
| `video` | video-only | 無音 video データ、標準 retention |
| `audio` | audio-only | standalone audio データ、強い retention、別ファイル保存 |

H3 の block は共有されるため完全分離ではない。しかし運用単位を分けることで、image-only 学習で temporal 能力を壊す、audio-only 学習で video 品質を壊す、といったリスクを下げられる。適用時に adapter を選択・除外できることが分離の実体である。

block 範囲選択（浅層のみ／深層のみ等）による分化は §16 の open question とする。

AI Toolkit の training adapter は比較対象とするが、公開コードから目的関数を完全には確認できないため、SushiUI の必須依存にはしない。

## 10. Future configuration/API principles

実装時の設定候補は次の通り。

```text
h3_training_objective: plain | plain_keep | cgl | cgl_keep
h3_cgl_anchor_mode: frozen_base | live_lora
h3_cgl_null_conditioning: zero_embeds | fixed_caption
h3_cgl_scale_video
h3_cgl_scale_audio
h3_cgl_schedule
h3_cgl_sigma_min
h3_base_keep_weight
h3_base_keep_interval
h3_missing_modality_policy: empty_rows | noise_rows | null_scaffold
h3_modality_sampling
h3_lora_profile
h3_debug_modality_grad_norm
```

既定は `plain_keep` とする。

`h3_cgl_anchor_mode: frozen_base` と base retention は `blocks_to_swap=0` を必須とし、併用指定は起動時に ValueError とする。CGL を有効にすると step あたり最大 3 forward（推測: +50%）になることをパラメータ説明に明記する。

将来追加する default は [`param_defaults.py`](../../backend/api/param_defaults.py) を single source of truth とする。現状 `TRAINING_DEFAULTS` に `h3_` プレフィックスのキーは存在せず、関連するのは汎用キー `audio_loss_weight`（`:2211-2230`）と別辞書 `TIMESTEP_SAMPLING_DEFAULTS_BY_ARCH["minimax_h3"]`（`:2327-2343`）だけである。API 変更は `openapi.yaml` を先に更新する。frontend は [`TrainingConfig.tsx`](../../frontend/src/components/training/TrainingConfig.tsx) がアーキ判定のみを行っており H3 固有の training パラメータ UI は未実装であるため、各 Phase の完了条件に `param_defaults.py` → `openapi.yaml` → Pydantic → frontend の配線を含める。

## 11. Failure safety and observability

少なくとも次をログする。既に実装済みのものと新規のものを区別する。

| 項目 | 状態 |
|---|---|
| `h3_video_loss` | 実装済み（`minimax_h3_ops.py:856-858`、`metric_registry.py:111-120`） |
| `h3_audio_loss` | 実装済み |
| `h3_audio_present` | 実装済み（右軸） |
| `h3_video_present` | 新規 |
| modality 別 target RMS | 新規 |
| `video_gap_rms`、`audio_gap_rms` | 新規 |
| CGL scale と実効 sigma | 新規 |
| base retention loss | 新規 |
| modality 別 gradient norm | 新規、debug フラグ + interval 実行に限定 |
| finite check 結果 | 新規（下記の別経路） |
| dummy scaffold 使用数 | 新規 |
| native zero-row/fallback 使用数 | 新規 |

新規メトリクスの大半は `log_extra_metric` を 1 行呼び、`metric_registry.py` の `EXTRA_METRIC_DEFS` に 1 エントリ足すだけで足りる。DB カラム追加も API 配線も不要である。

modality 別 gradient norm だけは例外で、既存の grad norm が unet/TE 単位でしか取れないため（`base_trainer.py:11944`）modality ごとの個別 backward が必要になり step コストが跳ねる。常時ではなく debug フラグ有効時の interval 実行に限定する。常時の代替指標としては、既に実装済みの `h3_video_loss` / `h3_audio_loss` の比と target RMS を用いる。

### 非有限値の扱い

大きな外挿 target、mixed precision、joint AV loss は Inf/NaN の原因になり得る。

非有限検出は**新規に実装する**。現行の `base_trainer.py` に loss の isfinite チェックは存在せず（`:12074` は metrics 直列化のみ）、既存の step skip 機構は CUDA error/OOM 専用である（`:11182`、`:11216-11246`、`:11442-11449`）。

仕様は次の通り。

- H3 の `train_step` 内で loss と target の isfinite を検査する。
- 非有限なら黙って clamp せず、その step の optimizer step をスキップする。
- scale、sigma、modality、最大絶対値をコンソールログに出す。`log_extra_metric` は非有限値を黙って捨てるため（`base_trainer.py:12070-12076`）この用途には使用禁止とする。
- 連続 N 回で run を abort する。前例は [`vae_trainer.py`](../../backend/core/training/vae/vae_trainer.py) `:938-943` の abort である。

## 12. Test plan

### 12.1 Math and loss unit tests

- `s=1` で CGL target が通常 target と一致する。
- `s>1` で unconditional 方向から外挿される。
- uncond anchor へ gradient が流れない。
- frozen-base anchor で base parameter が変化しない。
- `scale=0` 切り替えの前後で LoRA の寄与が完全に消え、切り替え後に元の scale へ復帰する。
- pre-shift/post-shift sigma が仕様通り計算される。
- absent modality の loss が 0 になる。
- active modality の gradient が非 zero になる。
- modality loss の平均が row 数に不当に依存しない。
- 非有限 target/loss が検出される（§11 の検出機構の実装が前提。それ以前はテスト対象が存在しない）。

### 12.2 Packed layout tests

| 項目 | 現状 |
|---|---|
| paired | 既存カバー（`minimax_h3_layout_test.py:69-135`） |
| video-only | 現行動作（audio 行は noise + mask） |
| image-only | 既存カバー（`minimax_h3_training_test.py:436-517`） |
| audio-only | 新規 |
| zero audio rows | 既存カバー（`minimax_h3_training_test.py:567`）。ただし layout レベルのみ |
| zero video rows | 新規 |
| audio channel-major order | 既存カバー（`minimax_h3_layout_test.py:135`、`:227`） |
| index の重複・欠落がない | 既存カバー（`minimax_h3_layout_test.py:80`、`:977`） |
| `timestep_indices` の shape | 既存カバー（`minimax_h3_training_test.py:580`） |
| video shift 12、audio shift 3 | 既存カバー |
| empty video rows を実際の transformer forward に通せる | 新規。zero-audio ですら forward 検証は存在しない |

実 forward を伴うプローブは 33B のロードを要するため、host RAM の事前見積りと単一 arm 実行のルールを適用する。

### 12.3 Dataset and cache tests

- video+audio が同一時間窓で encode される。
- video-only が音声ファイルを誤って読み込まない。
- image-only が T=1 の 5D latent になる。
- audio-only が PIL 画像経路へ落ちない。
- cache provenance に modality が記録される。現行は `is_window_record` / `has_audio` のみで `modality_kind` が無い（`latent_cache.py:475-483`）。
- cache hit/miss で shape が一致する。
- silent video と audio-only が混同されない。

standalone audio-only cache は新設する。現行の H3 audio latent は video clip record 内に同居し、key は `compute_clip_hash`（`latent_cache.py:203-290`）＝ video_path/w/h/window/stride/fps/start_time/tiling_policy に `audio_prep_version="h3-32k-stereo-v1"` を加えたもので、音声 path・sample rate・duration は独立要素として key に入らない（sample rate と channel 数は版文字列に畳み込まれている、`minimax_h3_ops.py:56-61`）。

新設する audio-only cache は ACE-Step の `compute_audio_hash`（`latent_cache.py:601-626`）を前例とし、key = audio_path + duration + `audio_prep_version`（sample rate/channel/VAE 設定は版文字列に畳み込む）とする。既存の video 同居 record と provenance で区別できることを要件とする。

### 12.4 Modality mask tests

fake transformer を使い、次を検証する。

- paired：video/audio 両方に gradient
- video-only：video のみ gradient
- image-only：video のみ gradient
- audio-only：audio のみ gradient
- 両方なし：事前拒否
- dummy rows の予測値を変えても inactive loss が変わらない
- scaffold が active modality の教師信号にならない

最後の 2 項目は §15 の汚染判定の定量ゲートとしても使う。

### 12.5 LoRA and retention tests

- LoRA save/load 後に出力が再現する。
- LoRA 無効時に base 出力が変わらない。
- base retention forward に gradient が流れない。
- profile ごとの adapter ファイルが独立に save/load できる。
- image/audio profile を外すと base 生成へ戻る。
- quantized base へ training adapter を merge しない（`api/arch_capabilities.py:445-447` が relora を拒否済み）。
- `blocks_to_swap>0` と retention/CGL の併用が起動時に拒否される。

### 12.6 Integration smoke tests

次の各ケースで最小 1 step を実行する。

1. video + audio
2. video only
3. image only
4. audio only native zero-video
5. audio only dummy scaffold fallback

各ケースで、loss が finite、active loss が計算され、inactive loss が 0、adapter 保存・load・通常 generation が成功することを確認する。

### 12.7 Regression tests

既存の次の挙動を維持する。

- silent video
- T=1 still latent（audio rows 0 行）
- zero audio layout
- audio presence metrics
- batch size 1 制約（`minimax_h3_ops.py:150-159`、`:708-716`）
- caption token 長制約（`:708-716`、`:718-721`）
- H3 full fine-tune 拒否（`api/arch_capabilities.py:442-447`、`adapters/minimax_h3_adapter.py:25-29`、`full_parameter_trainer.py:119-141`）
- H3 preview 未実装（`minimax_h3_ops.py:908-924`、`arch/minimax_h3.py:89-92`）
- bf16、gradient checkpointing、FP8 dequant-only 経路（`minimax_h3_ops.py:93-131`、`:221-228`、`:211-215`）

H3 関連テストは `backend/tests/minimax_h3_*.py` に 47 ファイルあり、training に直結するのは `minimax_h3_training_test.py`（42 関数）と `minimax_h3_layout_test.py`（40 関数）である。運用上の注意として、`pytest -k minimax_h3` は host RAM 38 GB を踏むため、テストはファイルパスを明示して個別に実行する。

既存の audio-only 拒否テスト（`minimax_h3_training_test.py:854` 以降）は、仕様変更時に standalone audio route の成功テストへ置き換える。ACE-Step と非 temporal arch の exemption テストは維持する。

## 13. Quality evaluation matrix

固定するものは seed、NFE、scheduler、解像度、frame 数、audio duration、prompt 集合とする。

比較対象:

1. frozen base
2. plain LoRA
3. plain LoRA + base retention（既定構成）
4. live-anchor CGL
5. frozen-base CGL
6. CGL + base retention
7. CGL + external training adapter
8. profile 別運用

評価項目:

- prompt adherence
- subject/identity 保持
- temporal consistency
- motion quality
- audio semantic quality
- audio artifact
- audio-video synchronization
- diversity
- long-run degradation
- NFE/scheduler 変更への頑健性

画像だけの評価で H3 の採否を決めてはならない。video/audio joint、temporal、AV sync を必ず含める。

### audio-only の評価プロトコル

推論側には現在 video-free 生成の経路が無く、`unpatchify_video_rows` は 0 フレームで RuntimeError となる（`h3_pipeline_ops.py:158`）。したがって audio-only 学習の評価方法を先に決める必要がある。次のいずれかを Phase 1 で選択する。

- **(A) joint 生成内評価**: 固定の video prompt で joint 生成し、その中の audio 品質のみを評価する。推論側の変更が不要。
- **(B) 推論側 zero-video 生成**: §7 の修正 2 箇所を推論経路にも適用し、audio-only 生成を正式にサポートする。

どちらかを決めないと Phase 4 の quality gate が実行できない。

H3 の training preview は未実装のため、評価は学習後に通常の generation path で行う。品質基準は実験前に baseline-relative tolerance を登録する。根拠のない絶対品質値を後付けで採用しない。

## 14. Rollout phases

### Phase 0: design validation（GPU 不要）

- CGL target と mask の unit test（fake transformer）
- §7 で静的に確定した zero-video 修正 3 箇所の設計取り込み
- audio-only cache key 設計
- null conditioning の定義選定（zero embeds / fixed caption）
- 非有限検出機構の仕様確定
- base retention の VRAM・時間の見積り（実測は Phase 1）

初版では「zero-video layout の forward 可否確認」を Phase 0 に置いていたが、静的解析で破綻箇所が 3 つ確定しており、未修正のままのプローブは確実に失敗する。可否判定は修正実装を前提とするため Phase 1 へ移した。

### Phase 1: modality routing

- 4 modality の data contract（`has_video` / `modality_kind` の導入）
- `item_type=="audio"` ガード群のアーキ非依存化。現在 `is_acestep` で分岐している箇所（`base_trainer.py:6866`、`:6872`、`:8465`、`:8552`、`:8608-8614`、`:9375`、`:10361-10368`、`:7396-7460`）を arch capability 化する。緩和しないと H3 の audio item が `Image.open` 経路へ落ちる。
- `arch/minimax_h3.py` への `vae_encode_audio` seam 実装。現行の唯一の音声入口 `vae_encode_audio_window`（`minimax_h3_ops.py:418-461`）は video_path 前提で、`.wav` 直接経路が無い。
- standalone audio encode/cache
- loss mask（`m_v` の導入、video 側 zero-branch）
- video-only/image-only 回帰
- **exit criteria**: zero-video / zero-audio の実 transformer forward プローブ成功。base retention の VRAM・step 時間の実測。audio-only 評価プロトコル（§13 の A/B）の決定。

### Phase 2: retention（初版の Phase 3 を前倒し）

- LoRA `scale=0` 切り替え helper
- base function-space retention
- `h3_base_keep_interval` による償却
- オフライン replay バッファ
- block swap 併用の拒否ゲート
- gradient/activation diagnostics

### Phase 3: CGL（実験、null-conditioning ゲート通過が前提）

- null conditioning の妥当性検証（\(u_r\) の RMS、\(v_r-u_r\) の方向安定性）
- live-anchor compatibility mode
- frozen-base anchor
- pre/post-shift sigma 比較
- scale calibration
- `s=1` での plain 退行テスト

### Phase 4: quality gate

- profile 別 adapter 運用
- long-run 学習
- 通常 generation path による検証
- external adapter との比較

## 15. Go / No-Go criteria

各基準に測定手続きを紐付ける。

### Go

| 基準 | 測定手続き |
|---|---|
| 4 modality すべてで active loss が計算される | §12.6 の smoke test 5 ケース |
| absent modality へ教師 gradient が流れない | §12.4 の fake transformer テスト |
| base retention で長時間学習の品質低下が抑制される | §13 の比較対象 2 と 3 を同一 step 数で比較、事前登録した tolerance 内 |
| audio-only native layout または fallback が再現可能である | §12.6 のケース 4/5 が固定 seed で再現 |
| 事前登録した quality tolerance 内に収まる | §13 の baseline-relative tolerance |
| Inf/NaN が発生しない | §11 の新規検出機構のカウンタが 0。検出機構の実装が前提条件 |
| adapter を外した base 生成が破壊されない | 固定 seed/prompt で学習前の base 出力と一致 |

### No-Go

| 基準 | 測定手続き |
|---|---|
| audio-only が dummy video scaffold に強く依存し、native audio 品質を再現できない | §13 で決めた評価プロトコル（A または B）で native と scaffold を比較 |
| image-only 学習で temporal 品質が tolerance を超えて低下する | adapter 装着時の temporal consistency を §13 の tolerance と比較。可逆性は adapter 分離で構造的に保証されるため「不可逆」を基準にしない |
| audio-only 学習で video 品質が大きく低下する | 同上、video 側の評価項目で判定 |
| scale や sigma convention で結果が不安定になる | Phase 3 の pre/post-shift 比較と scale calibration の分散 |
| base retention が長時間学習の崩壊を抑えられない | Go 基準 3 の裏返し |
| dummy modality が active modality の出力を大きく汚染する | §12.4 の dummy-row 摂動テストを定量化。dummy rows を再サンプルしたときの active 出力 RMS 変化が事前登録した ε を超える |
| AMP/FP8 経路で Inf/NaN が頻発する | §11 のカウンタ |

## 16. Open questions

- H3 の null conditioning（zero embeds / fixed caption）が元蒸留時の unconditional branch をどの程度近似するか。近似しない場合、CGL は永久に実験扱いのままとするか破棄するか。
- 元 teacher の guidance scale を公開 checkpoint から推定できるか。
- H3 の通常 generation が audio-only 入力・video-free 条件を意味的にサポートするか（§13 の A/B 選択）。
- video-only の zero-audio rows と現行 noise-row 方式の品質差。
- block 範囲選択 LoRA scope の実装是非。現行は全 50 block 一律で、modality I/O head は恒久除外。
- forward-hook ベースの offload 拡張により、no_grad forward と block swap を互換化できるか。
- 短尾ウィンドウの zero パディング行を audio のサンプル平均から除外すべきか。

初版で open question としていた「H3 transformer が empty video rows を全経路で安全に処理できるか」は、静的解析で破綻箇所と修正範囲（§7 の 3 箇所）が確定したため削除した。残る不確実性は実 forward プローブのみで、Phase 1 の exit criteria に移した。

## 17. References

- [Musubi Tuner H3 guidance-distillation PR](https://github.com/kohya-ss/musubi-tuner/pull/1045)
- [Musubi Tuner implementation commit](https://github.com/kohya-ss/musubi-tuner/commit/a892a044e5cc03ad7da00d7e2c259fedb1da358a)
- [Musubi Tuner H3 guide](https://raw.githubusercontent.com/kohya-ss/musubi-tuner/refs/heads/dev/docs/minimax_h3.md)
- [Ostris AI Toolkit](https://github.com/ostris/ai-toolkit)
- [AI Toolkit training configuration](https://raw.githubusercontent.com/ostris/ai-toolkit/main/toolkit/config_modules.py)
- [AI Toolkit SDTrainer](https://raw.githubusercontent.com/ostris/ai-toolkit/main/extensions_built_in/sd_trainer/SDTrainer.py)
- [AI Toolkit H3 training adapter code](https://github.com/ostris/ai-toolkit/blob/main/extensions_built_in/diffusion_models/minimax_h3/minimax_h3.py#L261-L356)
- [Guided Distillation for Classifier-Free Guidance](https://arxiv.org/abs/2210.03142)
- [Consistency Models](https://arxiv.org/abs/2303.01469)
- [Latent Consistency Models](https://arxiv.org/abs/2310.04378)
- [Pairwise Sample Optimization](https://arxiv.org/abs/2410.03190)
