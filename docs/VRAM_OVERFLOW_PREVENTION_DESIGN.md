# 大画像バッチの VRAM オーバーフロー対策 — 検討メモ（保留中）

ステータス: **検討のみ・実装保留**。SDXL フルFT（aspect-ratio bucketing + crop augmentation）で
大解像度バケットのバッチ処理時に VRAM が溢れる問題への対策を整理する。

---

## 1. 問題

- バッチの activation VRAM は**バケット解像度に強く依存**する。base_resolutions=[1024,1280,1536,1792,2048]
  ではバケット面積は base² に比例し、**2048バケットは1024バケットの約4倍**の activation。
  - 1024²=1.05M, 1280²=1.64M, 1536²=2.36M, 1792²=3.21M, 2048²=4.19M px
- batch_size は全バケット共通（例: 4）。平均は収まっても、**大バケットのバッチでスパイク → 溢れ**。
- crop augmentation は1 epoch で **256 バケット**を生成し、大バケット出現と形状断片化の両方を増幅する。

## 2. 現状とその限界（重要）

### 2.1 既存のリアクティブ OOM 回復
`base_trainer.py::_forward_backward_with_oom_recovery` は OOM 発生時にバッチ次元を半分割→逐次処理→
勾配累積（結果は同一）し、size 1 まで再帰分割する。

### 2.2 Windows WDDM では機能しない
**Windows の NVIDIA ドライバ（WDDM）は専用VRAMを使い切ると例外を出さず「共有GPUメモリ」(ホストRAM)
へ自動スピルする。** その結果:
- `cudaMalloc` が成功してしまい `torch.cuda.OutOfMemoryError` が発生しない
- 例外捕捉型のリアクティブ回復は**一度も発火しない**
- OOMで止まらず共有メモリに溢れて**激遅化**（GPU使用率は0%付近のまま「停止」に見える）

→ **Windows では検出（リアクティブ）方式は原理的に当てにできない。** 先制防止が必須。

### 2.3 アロケータ設定
`PYTORCH_CUDA_ALLOC_CONF` / `expandable_segments` は未設定。256バケットの多形状による予約メモリ
断片化は「形状変化ごとの empty_cache」で対症療法的に対応しているのみ。

## 3. 対策案

### ◎ 案1: 先制マイクロバッチ（size-budgeted gradient accumulation）— 本命
forward/backward の**前に**空きVRAMを見てマイクロバッチサイズを決め、超過なら**事前に**分割して
勾配累積する。OOM検出に一切依存しない。

- 空き容量は `torch.cuda.mem_get_info()`（free, total を返す）で取得。
- バッチのコスト ≈ `batch_size × bucket_area × per_pixel_cost`。`per_pixel_cost` は形状ごとに初回実測して
  キャッシュ（各バケット1回だけ計測）。
- `micro_bs = max(1, floor(free_headroom / per_sample_cost))` のように、専用VRAMに収まる最大microへ。
- **結果は完全に同一**（同じサンプルを累積するだけ）＝設定不変・学習dynamics不変。
- 既存の分割スライス機構（`_forward_backward_with_oom_recovery` のテンソル slice）を**OOM起因ではなく
  サイズ起因で先制発火**させる形で再利用できる。
- 分割は **MNTループの各イテレーション内**でバッチ次元を割る（既存OOM回復と同じ層）。

### ○ 案1-b: `torch.cuda.set_per_process_memory_fraction(f)`（補助）
キャッシュアロケータの上限を専用VRAM以下（f=0.90〜0.95程度）に固定 → スピル前に
`OutOfMemoryError` を**強制発生** → 既存のリアクティブ回復が保険として復活する。
- 通常運用ではコストゼロ（単なる上限）。
- **WDDM での効きは要実測**（環境差あり）。案1の保険として併用する位置づけ。

### ○ 案2: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（補完）
256バケットの多形状による**予約メモリ断片化を低減** → 実使用は収まるのに断片化で溢れるケースを防ぐ。
- 学習結果に影響なし。起動時のアロケータ設定。
- 稀に回帰例あり → opt-in・要実測。現状の per-shape empty_cache の本来の解。

### △ ユーザー側ドライバ設定（コード外・最も確実に実OOMを得る）
NVIDIAコントロールパネル → 「CUDA - システムメモリフォールバックポリシー」→
「システムメモリフォールバックを優先しない」（python.exe 個別）。
- ドライバレベルでスピルを禁止 → CUDAが本当にOOMする。テスト時の早期発見にも有効。

### × 採用しにくい案
- バケット別 batch_size 可変（勾配累積と組み合わせないと実効バッチが変わる＝設定変更）。
- 最大解像度の制限（学習解像度を下げる＝設定変更）。
- さらに細粒度の gradient checkpointing（既に適用済み・効果逓減）。

## 4. iter コストへの影響

| 施策 | iterコスト |
|------|-----------|
| 案1 先制マイクロバッチ | **大バケットのバッチのみ**増加（分割でGPU並列度低下＋カーネル起動増）。理想のフルバッチ比では遅いが、**現状のスピル(10〜100倍遅い)比では大幅高速**。小バケットはオーバーヘッドゼロ。総FLOPs・勾配は不変。 |
| 案1-b memory_fraction | 通常運用ゼロ。超過時にスピルでなくOOMを出すのみ。 |
| 案2 expandable_segments | 中立〜やや高速（断片化低減で empty_cache スラッシング減）。稀に回帰→要実測。 |
| ドライバ設定 | ゼロ（挙動のみ変更）。 |

要点: **学習結果（実効バッチ・勾配）は不変**。コストは「大バケット時のスループット」のみで、その大バケットは
現状スピルで激遅なため実質は改善。

## 5. 推奨

1. **案1（先制マイクロバッチ, `mem_get_info` ベース）** を本命として実装 → 検出に頼らずスピルを根絶。
2. **案1-b（`set_per_process_memory_fraction`）** を併用 → 想定外スパイク時に実OOM→既存回復が保険。
3. **案2（expandable_segments）** を起動時に opt-in で併用 → 断片化低減。
4. テスト環境では**ドライバ設定**でスピル禁止にしておくと早期発見に有効。

## 6. 実装時の留意点（再開時の手掛かり）

- 分割は MNT ループ内のバッチ次元分割。既存 `_forward_backward_with_oom_recovery` のテンソル slice を流用し、
  「サイズ予算 → micro 数」を先に決めて呼び出す形に。
- `per_pixel_cost` は (bucket_w × bucket_h) → 実測ピークのマップとして形状ごと初回キャッシュ。MNT・VE・crop
  併用時の実測値で較正する。
- 予算は GPU 容量に対する相対値（`mem_get_info` の total/free）で自動スケール（24GB/48GB対応）。
- crop augmentation の多形状下では案2の効果が特に大きい。
- 関連: VAEのtrain_step中CPU退避、VE/VE-optimizerの遊休オフロード（実装済み）と組み合わせると、
  大バケット時のピークがさらに下がる。

## 7. 関連実装（既存）
- `_forward_backward_with_oom_recovery`（リアクティブ分割／Windowsでは発火しない点に注意）
- per-shape `torch.cuda.empty_cache()`（断片化対策の対症療法）
- gradient checkpointing（U-Net / TE1 / TE2 / Vision Encoder）

---

## 8. batch_size を変えずに大画像の VRAM を下げる（activation削減）— 保留中

マイクロバッチ（§3案1）とは別に、**batch_size を1回のforwardで保ったまま** activation メモリを
下げる手段。後で議論する。

| 手段 | 効果 | batch維持 | 結果 | 状態 |
|------|------|----------|------|------|
| ① **SDXL VAE tiling/slicing** | 大画像のVAEエンコードのピークを tile サイズで上限化（数十GB→数GB） | ✅ | ほぼ同一（overlapブレンド、小画像は閾値以下で非tile＝無影響） | **未適用** |
| ② FlashAttention | 大画像 attention を O(N²)→O(N) | ✅ | 同一 | **既にON**（設定 use_flash_attention） |
| ③ Activation CPU offload (`torch.autograd.graph.save_on_cpu`) | U-Net forward activation をCPU退避→backwardで戻す | ✅ | 同一 | 未実装・PCIeで iter遅延（大バケット限定なら可） |
| ④ gradient checkpointing | block単位で activation 再計算 | ✅ | 同一 | **既にON**（U-Net/TE/VE） |
| ⑤ channels_last | conv効率化で多少削減+高速 | ✅ | 同一 | 未適用・効果小 |

### 本命: ① SDXL VAE tiling/slicing
- コードベースは **MiniT2I で実装済み**（`base_trainer.py:1446-1458`）。コメントにこの問題が明記:
  「~2048px の単一fp32エンコードは早期フル解像度conv段 + ボトルネック空間self-attentionで**数十GBにピーク**。
  Tiled encode/decode は VAE メモリを画像サイズではなく tile サイズで上限化」。
- **SDXLの `encode_image` 経路には未適用** → 大画像のVAEエンコードが数十GBにスパイク（大画像バッチ溢れの主因の一つ）。
- 実装は MiniT2I と同じ `vae.enable_tiling()/enable_slicing()` を SDXL VAE ロード時に `hasattr` ガード付きで
  追加するだけ。`SDXLVAEWrapper` でも安全。iterコストはほぼゼロ（VAEエンコードは元々高速・no_grad）。

### 補助
- ③ activation offload は U-Net 本体の activation を削れるが PCIe で iter が遅くなる → 大バケット限定発火が前提。
- ⑤ channels_last は速度寄りの小改善。

### 推奨
②④は既にON。**残る大きな未活用レバーは①（SDXL VAE tiling）**。低リスク・batch不変・結果ほぼ不変。
さらに削るなら③（大バケット限定・速度トレードオフ）。

---

## 9. 高解像度バッチの activation を batch を保ったまま削減する（本格検討）

§3案1（バッチ次元のマイクロ分割＝勾配累積）は**採用しない**方針。ここでは
**batch を 1 回の forward で保ったまま、高解像度の activation を削る**方向を整理する。

### 9.1 まず問題の所在を正確に
学習時 VRAM = 重み + 勾配 + optimizer state + **activation**。
- 重み/勾配/optimizer state は**解像度に依存しない（固定）**。
- **activation のみが解像度に比例**（U-Net: `B·C·H·W`、DiT: `B·seq·d`、`seq=(H/p)(W/p)`）。
- ⇒ **高解像度バッチは activation 支配**。

既存の **block swap / ring buffer / musubi の H2D-only block swap は「重み」を対象**（krea2.md:
H2D-only は推論・凍結重み前提で Host→Device のみ・CPUマスター・D2H無し、ring_size 2 で転送/計算
オーバーラップ）。**重みは解像度非依存**なので、高解像度の activation には**原理的にほとんど効かない**
（ご指摘の通り）。`iter` も遅くなる。

⇒ 必要なのは **activation 側の削減**で、かつ **iter を遅くしない（転送を計算でオーバーラップ）**こと。

### 9.2 「バッチ分割」ではなく「次元分割」
ご提案の「高次元バッチを分割して逆伝搬」は、**batch 次元の分割（＝案1）ではなく、
spatial / sequence 次元の分割**として実装するのが本質。batch は保たれ、結果も同一。

### 9.3 提案手法（U-Net / DiT 両対応）

#### A. 非同期 activation オフロード（オーバーラップ前提）— 本命・汎用
gradient checkpointing の**境界 activation（backward用に保持する入力テンソル）を CPU へ退避**し、
backward で**計算とオーバーラップして prefetch** する。
- **activation を対象**＝**解像度依存**＝高解像度に効く（H2D-only block swap の activation 版）。
- **二重バッファ＋専用CUDA streamで転送を backward 計算の裏に隠す**と iter 低下は最小（§3案で
  既存の `LayerOffloadConductor`（`memory_management`）の prefetch 機構が流用候補）。
- `torch.autograd.graph.save_on_cpu()` は同期版で遅い → **非同期オーバーラップ版を自前実装**するのが鍵。
- U-Net / DiT 双方の block 境界に適用可能。

#### B. Selective activation recomputation（Megatron 方式）— 補完
full checkpointing（全再計算）でも no-checkpoint（全保持）でもなく、
**「保持コストが高く再計算が安い op（attention softmax 等）だけ再計算、安いものは保持」**。
- full checkpointing より**再計算コストが小さく**、no-checkpoint より**メモリが小さい**中間点。
- DiT/U-Net の attention 周りに特に有効。

#### C. sequence / spatial チャンク再計算（新規・「次元分割逆伝搬」の核）
checkpointed block の **backward 内の再計算を、モデルが独立に扱える次元でチャンク処理**し、
**再計算時のピーク activation をチャンク係数だけ下げる**。batch・結果は不変。
- **DiT（本命）**: 1 block = attention（全結合）+ MLP/Norm/residual（**トークン毎＝seq方向に独立**）。
  - MLP/Norm/residual は **seq をチャンク分割して再計算+backprop すれば厳密**（トークン間混合なし）。
    高解像度で最大の activation である MLP 中間（`B·seq·d_ff`）をチャンク係数で削減。
  - attention は全結合だが **FlashAttention が既に O(seq) メモリ**で吸収（材料化しない）。
  - ⇒ **DiT は seq チャンク再計算がクリーンに効く**（halo 不要）。
- **U-Net（conv）**: spatial に分割すると conv の受容野で **halo（のりしろ）重複**が必要 → 厳密化は可能
  だが実装は重い。低解像度 block の attention は小さいので、効くのは高解像度 conv 段。
- **出力/loss 段の早期分割（簡易版）**: 最終 proj + loss は**ピクセル毎**なので、
  最終数層の backward を seq/spatial チャンクで回すだけでも、高解像度で大きい最終段 activation を削れる
  （実装が軽く効果がある“入口”）。

#### D. 既存重み手法との直交合成
- **重み**: H2D-only block swap（解像度非依存・大パラメータDiT向け）。
- **activation**: A（非同期オフロード）+ C（次元チャンク再計算）。
- 両者は**直交**＝**併用で「重みは block swap、activation は次元分割/オフロード」**と分担できる。

### 9.4 トレードオフ / 期待効果

| 手法 | 対象 | 高解像度効果 | iter コスト | 実装難度 | batch/結果 |
|------|------|-------------|------------|----------|-----------|
| A 非同期activationオフロード | activation | 大（解像度依存） | 低（オーバーラップ前提）／高（同期だと） | 中（stream/二重buffer） | 不変 |
| B selective recompute | activation | 中 | 低（再計算減） | 中 | 不変 |
| C seq/spatialチャンク再計算 | activation | 大（DiTのMLP） | 低〜中（再計算分） | DiT:中 / U-Net(halo):高 | 不変 |
| D + H2D-only block swap | 重み | 小（解像度非依存） | 低（overlap） | 中 | 不変 |

### 9.5 推奨アプローチ
1. **DiT 優先で C（seq チャンク再計算）**＝新規価値が最大・halo 不要・MLP に直撃。まず
   **出力/loss 段の seq チャンク（簡易版）**から入り、効果を測ってから block 全体へ展開。
2. **A（非同期 activation オフロード）を汎用基盤**として、既存 `LayerOffloadConductor` の prefetch を
   **重みだけでなく activation にも拡張**（オーバーラップで iter を守る）。U-Net/DiT 共通。
3. **B（selective recompute）**を attention 周りに適用し、A/C の再計算コストを相殺。
4. 重み側は **H2D-only block swap**（DiT 大パラメータ時）を直交併用。

### 9.6 留意点
- A/C はいずれも**「転送・再計算を backward 計算でオーバーラップ」できるか**が iter を守る鍵。
  非同期化（専用 stream・二重バッファ・prefetch）を最初から設計に入れる。
- C の DiT seq チャンクは **attention を跨がない**こと（attention は FlashAttention に委ね、
  チャンク化は MLP/Norm/residual の per-token 区間に限定）。
- U-Net の spatial halo タイル化は厳密だが重い → 効果の大きい最高解像度 conv 段に限定して段階導入。
- いずれも **batch 不変・勾配同一**（数値は再計算/オフロードで bit-exact、halo は重複計算で厳密）。

---

## 10. 実装計画（推奨ステップ順・コード根拠付き）

§9.5 の推奨順（C → A → B → D）を、実コードの構造に基づいて段階実装計画化する。
**この §10 は計画のみ。コード実装は各ステップごとにユーザー承認後に着手する。**

### 既存コードの前提（調査確定事項）

| 項目 | 場所 | 内容 |
|------|------|------|
| Z-Image DiT block | `backend/core/models/zimage_transformer.py:143` `ZImageTransformerBlock` | Pre-Norm。attention と FeedForward(SwiGLU, hidden=10240) が分離。各々独立 norm + residual。AdaLN modulation あり |
| Z-Image checkpoint 単位 | 同上 `:587-596` | `torch.utils.checkpoint.checkpoint(layer, unified, mask, freqs_cis, adaln_input, use_reentrant=False)` ＝ **1ブロック=1単位** |
| Z-Image 形状 | 同上 | block 内 `[B, seq, dim=3840]`、seq は SEQ_MULTI_OF=32 倍数 padding、attention は `dispatch_attention()`（flash/sage/native）で O(seq) |
| Z-Image 出力段 | 同上 `:216` `FinalLayer` / `:604` | AdaLN + Linear(3840→64)。**per-token 独立**。直後 unpatchify で `[C,F,H,W]` |
| **既存の非同期offload基盤** | `backend/core/memory_management/layer_offload_conductor.py` | `_offload_activation`/`_restore_activation`(`:305-332`)、`DynamicActivationAllocator`、`transfer_stream`/`compute_stream`(`:88-90`)、pinned memory、Ring Buffer。**`enable_activation_offload=False` で現在無効**(base_trainer.py:1005) |
| **DiT の既存 async ckpt 前例** | base_trainer.py:1077-1088(Anima)/1222-1230(Lens) | `enable_gradient_checkpointing(cpu_offload=, async_offload=)` の3モード(standard/cpu_offload/async_cpu_offload)が**既に存在**。Z-Image/FLUX.2 は標準GCのみ |
| U-Net GC | base_trainer.py:2559-2574 | diffusers `unet.enable_gradient_checkpointing()`（境界は diffusers 内部）|
| backward 呼び出し | base_trainer.py:5027-5047(SD/SDXL train_step) / 5742(zimage) | autocast 内 forward → `loss.backward()`。grad accum / GradScaler(FP16のみ) |
| param SSoT | `backend/api/param_defaults.py` `TRAINING_DEFAULTS` | 全新規パラメータはまずここに追加（CLAUDE.md 厳守）|

---

### ステップ1（最優先）: C — DiT seq チャンク再計算

**狙い**: 高解像度 DiT で最大の activation である FeedForward 中間 `B·seq·10240` を、
checkpoint の**再計算時のみ** seq 方向チャンクで分割し、ピークを 1/N に下げる。batch・勾配 bit-exact。
attention は触らない（FlashAttention に委譲）。

#### Phase 1-a: 出力/loss 段の seq チャンク（簡易版・効果検証の入口）
- **対象**: `zimage_transformer.py` `FinalLayer.forward`(`:216-230`) と forward 末尾(`:604`)、
  および loss 計算（base_trainer の train_step_zimage `:5742-`）。
- **変更**: `FinalLayer`（AdaLN+Linear、per-token独立）を **seq をチャンクに割って逐次適用**する
  ヘルパ `_chunked_final_layer(x, c, chunk)` を追加。`chunk` 指定時のみ分割、未指定は現状動作。
- **正当性**: per-token なのでチャンク間依存ゼロ → 出力連結で完全一致（bit-exact）。
- **効果**: 最終段 `[B,seq,3840]→[B,seq,64]` と loss の中間を chunk 係数で削減。最高解像度で効く軽量入口。
- **リスク**: 低。autograd は連結で正しく勾配を流す。
- **検証**: chunk=1（=現状）と chunk=4 で同一入力→loss値・grad norm が一致することをユーザーが確認。

#### Phase 1-b: FeedForward の seq チャンク再計算（本命）
- **対象**: `ZImageTransformerBlock.forward`(`:189-211`) の FeedForward 区間
  `feed_forward(ffn_norm1(x) * scale_mlp)`。
- **設計**: block 全体の checkpoint は維持しつつ、**block forward 内の FF 適用を seq チャンク化**した
  `_chunked_feed_forward(x_normed, chunk)` に差し替え。norm・residual・AdaLN gate も per-token なので同様に分割可。
  - 重要: **attention をチャンク化しない**（全結合・RoPE 跨ぎ。FlashAttention が既に O(seq)）。
    分割対象は `attention_norm` 後の residual 加算〜FFN 区間の **per-token 部分のみ**。
- **正当性**: FFN/norm/residual はトークン独立 → チャンク連結で厳密一致。
- **効果**: FF 中間 `B·seq·10240`（block内最大の activation）をピーク 1/N。30 layers 全てに効く。
- **リスク**: 中。SEQ_MULTI_OF=32 境界・padding mask・`unified`(画像+caption結合 seq) の扱いに注意。
  チャンク境界は padding を跨いでも per-token なので問題ないが、テストで mask 整合を確認。
- **新規パラメータ**（param_defaults.py → OpenAPI → Pydantic → frontend）:
  - `dit_seq_chunk_size: int = 0`（0=無効。>0 でチャンクトークン数。例 4096）
- **適用条件**: `is_zimage`（将来 flux2/anima/lens/ideogram4/minit2i へ横展開可。U-Net は対象外）。
- **検証**: chunk=0 と chunk>0 で loss・grad 一致＋VRAM 低下をユーザー確認。

---

### ステップ2: A — 非同期 activation オフロード（汎用基盤）

**狙い**: checkpoint 境界で backward 用に保持する入力 activation を CPU 退避し、
backward の再計算/計算と **オーバーラップして prefetch** → iter 低下を最小化。U-Net/DiT 共通。

#### Phase 2-a: Z-Image/FLUX.2 を Anima 方式の async ckpt に揃える
- **発見**: Anima/Lens は既に `enable_gradient_checkpointing(cpu_offload=, async_offload=)` を持つ
  (base_trainer.py:1077-1088, 1222-1230)。**Z-Image/FLUX.2 だけ標準GC**(`:960-962`, `:1882-1884`)。
- **変更**: Z-Image transformer(`zimage_transformer.py:587-596`)の checkpoint 呼び出しに
  **境界 activation の非同期 CPU offload/prefetch オプション**を追加し、
  base_trainer 側で `enable_gradient_checkpointing(cpu_offload=, async_offload=)` 相当の引数を渡す。
  - 既存の `tensor_utils.async_copy_to_device` / `torch.cuda.Event` / pinned memory を流用。
  - double-buffer + `transfer_stream` で backward 計算の裏に転送を隠す。
- **正当性**: 退避/復元は値不変 → bit-exact。
- **リスク**: 中。stream 同期ミスは hang/競合 → Event 待ちを厳密に。`save_on_cpu()` の同期版は遅いので使わない。

#### Phase 2-b: LayerOffloadConductor の activation offload を有効化
- **対象**: `layer_offload_conductor.py` の `enable_activation_offload`（現状 `False` 固定; base_trainer.py:1005）。
- **変更**: block swap 併用時に activation offload も選択可能化（`DynamicActivationAllocator` 既存）。
  prefetch を「重み」だけでなく「activation」にも拡張。
- **新規パラメータ**: `activation_offload_enable: bool = False`, `activation_offload_async: bool = True`。
- **リスク**: 中〜高（既存 block swap との相互作用）。block swap OFF でも単独で効くよう独立フラグ化。
- **検証**: 同一 seed で loss/grad 一致、VRAM 低下、iter 低下幅（オーバーラップ効果）をユーザー測定。

---

### ステップ3: B — selective activation recomputation（attention 周り）

**狙い**: full checkpointing の**再計算コストを削る**。保持高コスト・再計算安価な op だけ再計算、
安い線形層などは保持。A/C で増えた再計算コストを相殺。

- **対象**: `ZImageTransformerBlock` の attention 区間（`dispatch_attention`）と FFN の切り分け。
- **設計**: block を「全再計算」する現状から、**attention 出力（O(seq²) 材料化を避けたい部分）は再計算、
  線形射影など安価で大きい中間は保持**、の選択的方針へ。FlashAttention 使用時は既に材料化回避済みなので、
  主眼は「どの中間を保持し、どれを捨てて再計算するか」のポリシー化。
- **新規パラメータ**: `selective_recompute_policy: str = "full"`（`full`|`attention_only`|`none`）。
- **リスク**: 中。diffusers U-Net は内部実装のため U-Net 側は限定的（DiT 優先）。
- **検証**: ポリシー別に loss 一致＋VRAM/iter のトレードオフをユーザー測定。

---

### ステップ4: D — H2D-only block swap（重み・直交併用）

**狙い**: 大パラメータ DiT の**重み** VRAM を Host→Device のみのストリーミングで削減（解像度非依存）。
A/C（activation）と直交し併用可能。

- **対象**: 既存 `block_offloading.py`（推論用 weights-only offloader）/ `LayerOffloadConductor`（学習用）。
- **設計**: musubi krea2.md の H2D-only（CPU マスター・D2H 無し・ring_size 2 で overlap）を学習の
  **凍結重み部分**（例: 凍結 TE/VE、LoRA 学習時の凍結 base）に適用。学習対象重みは勾配で更新されるため D2H が要る点に注意（フル H2D-only は凍結部限定）。
- **新規パラメータ**: `block_swap_h2d_only: bool = False`, `block_swap_ring_size: int = 2`。
- **リスク**: 中。学習可能重みには非適用（凍結部のみ）の境界を明確化。
- **検証**: 凍結部の重み VRAM 低下と iter 影響をユーザー測定。

---

### 横断的な実装規約（全ステップ共通・CLAUDE.md 準拠）
1. **param_defaults.py が SSoT**: 新規パラメータは `TRAINING_DEFAULTS` にまず追加 → OpenAPI(`openapi.yaml`)
   → `TrainingRunCreateRequest`(Pydantic, routes.py) → `training_config.py` 出力 → frontend
   (`api.ts` 型 / `TrainingConfig.tsx` UI / `StartupContext`)。漏れ防止チェックリストを各PRで実施。
2. **既定 OFF**: 全機能はデフォルト無効（chunk=0 / enable=False / policy="full"）。既存挙動を一切変えない。
3. **bit-exact 検証**: 各機能 ON/OFF で同一 seed の loss 値・grad norm 一致をユーザーがテスト
   （Claude はバックエンド/トレーニングを起動しない — CLAUDE.md）。
4. **段階コミット**: Phase 単位でコミット。大規模変更時は `git diff --cached` 比較レビューを記録。
5. **アーキ分岐**: `is_zimage`/`is_flux2`/`is_anima`/`is_lens`/`is_ideogram4`/`is_minit2i`/`is_sdxl` で
   対象を明示。U-Net(diffusers内部)は C/B の適用が限定的 → DiT 優先、U-Net は A（境界offload）中心。

### 実装順序の依存関係
```
ステップ1 C (DiT seq chunk)  ── 独立・最優先（Phase1-a → 1-b）
        │
ステップ2 A (async offload)  ── 1と独立だが基盤。Phase2-a(Z/FLUX async ckpt) → 2-b(conductor activation)
        │
ステップ3 B (selective)      ── C/A の再計算コスト相殺。1・2 の後が効果的
        │
ステップ4 D (H2D-only swap)  ── 重み側・直交。いつでも追加可（凍結部限定）
```
**最初の着手推奨**: ステップ1 Phase 1-a（出力/loss 段 seq チャンク）— 最小・低リスク・効果測定の入口。

---

## 11. バケット予測ディスパッチ（SDXL・activation offload の精密振り分け）

aspect-ratio bucketing では 3072×1536 のような非正方・大面積バケットが、各バケット固有の bs で
混在する。activation offload は**全解像度常時ON では低中解像度で iter を払って VRAM をほぼ削れず逆効果**
（§11.2 実測）。よって **バケット×bs に応じて offload を先制ディスパッチ**する機構が要る。
OOM 検出には頼れない（Windows WDDM はスピルして例外を出さない・§2.2）ため、**forward 前に予測**する。

### 11.1 予測モデル（実測で確立）
PoC: `tmp/test_B4_aspect_dispatch.py`（fp16 / FA2 / **GC ON** = 現行実装条件）。

```
predicted_peak(bucket, bs) = static + coef × (bs × lat_h × lat_w)      lat = px / 8
```

| 知見 | 実測 |
|------|------|
| **activation は `bs × latent面積` に比例**（aspect 非依存） | coef のばらつき **1.03倍**（23.68〜24.33 ×1e-6 GB/pixel） |
| **アスペクト比は無関係・面積×bsが全て** | 2048×1024 bs4 ≡ 2048² bs2（共に bs×面積=131072）→ peak 12.7GB で**完全一致** |
| 3072×1536 も同一直線に乗る | bs1=11.3GB（面積73728）, bs2=13.1GB（147456） |
| `static` は解像度非依存（重み+勾配+optim state） | fp16 SDXL で約 9.6GB（測定モデル） |

⇒ **forward 前に peak を正確に予測でき、先制ディスパッチが成立**（WDDM 安全）。

### 11.2 offload の効きと iter コスト（実測・現行GC前提）
PoC: `tmp/test_B2_offload_gc.py`（peak）, `tmp/test_B3_offload_timing.py`（iter）。

- **GC 下では offload の効きは activation の約 20%**（`save/act ≈ 0.2`）。GC が既に大半の activation を
  捨てており、offload が動かせるのは残りの境界テンソルのみ。⇒ offload の追加ヘッドルームには上限がある。
- **VRAM 削減 vs iter コスト（同期 offload）**：
  | バケット帯 | VRAM 削減 | iter コスト(同期) | 損得 |
  |-----------|----------|-----------------|------|
  | 1024〜1536px | +0.3〜0.8GB（僅か） | +5〜10% | ✗ 常時ONは逆効果 |
  | 2048px bs4 級(superlinear) | **−3.9GB(−21%)** | **+3%** のみ | ◎ 限定発火で得 |
  - 同期でも +3〜10%（高解像度ほど計算支配で減速率は下がる）。**非同期stream化（§10ステップ2-A）で≒0** に。

### 11.3 三段ディスパッチ（バッチ毎・forward前）
PoC: `tmp/test_B5_dispatcher_poc.py`（`ActivationDispatcher` クラス）。

```python
free, total = torch.cuda.mem_get_info()      # 実空きVRAM（24/48GB を自動スケール）
avail = free - margin                          # margin はスピル回避の安全帯

if predict(bucket, bs, offload=False) <= avail:
    mode = "fast"          # offload OFF（iterコスト0・小〜中バケット）
elif predict(bucket, bs, offload=True) <= avail:
    mode = "offload"       # activation offload ON（大バケット限定発火）
else:
    mode = "escalate"      # offloadでも不足 → micro-batch分割(+offload) / tiling(C)
```

**escalate の micro-batch 分割（実装済み）**: `plan_micro_bs(lh,lw,bs,free)` が
`M = floor((avail - static) / (coef·lat_area·residual_frac))` で**収まる最大の micro バッチ M**を
求め、バッチを `ceil(bs/M)` チャンクに分割して**勾配累積**する。各チャンクの loss を `chunk/bs` で
スケールするため、累積勾配は full-batch の mean 勾配と**等価**（実効バッチ不変）。
- **重要**: 分割は **escalate と判定されたバケットのみ**。fast/offload は分割しない＝**収まるバッチを
  分割して遅くすることはない**（先制予測ゆえに過剰分割が起きない）。
- **勾配等価性の実測**（`tmp/test_microbatch_sdxl.py`, 実 SDXL）: full vs micro+accum の勾配差は
  **fp32 で rel 1.9e-4**（算法的に等価、残差は浮動小数点の縮約順序ノイズ）。**fp16 では L2 ~6%**の
  精度ノイズが出る（fp16 学習と同質）。厳密一致が要る場合は累積を fp32 grad で行う拡張を検討。
- **VRAM 効果**（実測）: 2048px bs4 で 18.2→13.1GB、3072px **bs8 は full で OOM→micro で 17.2GB**に収まる。
- 既存 `_forward_backward_with_oom_recovery` のリアクティブ分割は **loss を未スケール**で累積する潜在バグ
  （勾配が約2倍）があった → 修正済み（`effective_batch_size` を再帰伝播し `chunk/B_eff` スケール）。

**on-the-fly TE/VE 学習との両立（二段構成・実装済み）**: TE/VE を on-the-fly 学習すると埋め込みは
**TE forward の共有グラフを持つ非リーフ**になり、バッチをスライスして複数回 backward すると
「backward a second time」で落ちる（最初の backward が共有グラフを解放）。これを**埋め込み境界の
detach 二段構成**で解決する:
1. 各 chunk は **detach した埋め込みリーフ**で U-Net を回す（backward は U-Net で止まり TE グラフ不変）。
   chunk 毎に埋め込み勾配 `leaf.grad` を `chunk/B_eff` スケール込みで回収・累積。
2. 全 chunk 後、累積した埋め込み勾配で **`torch.autograd.backward(emb_full, emb_grad)` を1回**実行し
   TE/VE グラフを1回だけ辿る → エンコーダ勾配が正しく入る。
- 対象は graph を持つ入力（`mnt_latents`/`mnt_text_embeddings`/`mnt_pooled_embeddings`/`mnt_repa_pixels`）を
  自動検出。リーフ（pre-encoded/cached）なら従来通り素のスライス。
- **実測**（`tmp/test_twostage_micro.py`）: U-Net 勾配 rel 1.0e-7、**エンコーダ勾配 rel 2.2e-7** で
  full-batch と等価 → U-Net activation を削りつつ TE/VE も正しく学習。
- **非対応**: fused backward（Block Swap 時の per-param step）— backward 中に optimizer.step するため
  micro 分割自体と非互換。この場合は分割を無効化し offload のみ（既存ガード）。

**情報源は2つ**：
1. `torch.cuda.mem_get_info()` の実空き（GPU容量に自動スケール）。
2. **オンライン・バケット校正キャッシュ** `key=(lat_h, lat_w, bs)`：各バケット初出現時に
   `max_memory_allocated()` で **フルバッチの base/offload 両 peak を実測**しキャッシュ。以降は O(1) 判定。
   未知バケットの初回のみ §11.1 の coef を保守的初期値に使う。
   - **理由**: 線形 coef は通常域で 1.03倍と正確だが、最大級バケット（2048² bs4 で 8.6GB）は
     transient/attention で **superlinear 化**する。実測校正がこの非線形テールを捉える。
   - **校正の注意（PoC で発見した実装上の罠）**: 校正は必ず**フルバッチ**の base/offload を測ること。
     escalate 実行時の micro-batched peak を「フルバッチ offload peak」として保存すると、
     次回同バケットで過小評価し誤って offload 判定 → 溢れる（PoC で再現・修正済み）。

### 11.4 PoC 実証結果（混在バケット列・budget=14GB で全分岐を誘発）
```
bucket     bs  decision  pre-pred  measured  fits?  note
1024^2      4   fast      11.1G    11.2G     YES   calibrated
1536^2      4   offload   12.6G    12.3G     YES   calibrated
3072x1536   1   fast      11.3G    11.3G     YES   calibrated
2048x1024   4   fast      12.7G    12.7G     YES   calibrated
3072x1536   2   offload   12.6G    12.3G     YES   calibrated
2048^2      4   escalate  14.9G    12.4G     YES   micro_bs=2
1024^2      4   fast      11.2G    11.2G     YES   cache       ← 再出現は O(1)
2048^2      4   escalate  14.3G    12.4G     YES   micro_bs=2  ← cacheがsuperlinear捕捉
```
- 全判定を **forward 前に予測**で実施（OOM 検出ゼロ）。全バケットが budget 内に収束。
- 非正方バケット（3072×1536）も面積×bs で正しく分岐（bs1=fast / bs2=offload）。
- 再出現バケットは校正キャッシュで即決＋ superlinear テールを正しく escalate。

### 11.5 実装の置き場所と SSoT
- **置き場所**: バッチ取得直後・`train_step` 呼び出し**前**（`base_trainer.py` のループ内）。
  bucket sampler が `(lat_h, lat_w, bs)` を持つのでキー化は容易。既存の per-shape `empty_cache()` と同じ層。
- **SSoT**: しきい値（`margin`, 初期 `coef`, offload 残存率, escalate の micro 分割方針）は
  `backend/api/param_defaults.py::TRAINING_DEFAULTS` に集約 → OpenAPI → Pydantic → frontend。
- **既定 OFF / 後方互換**: ディスパッチャ自体を `activation_dispatch_enable: bool = False` でゲート。
  OFF 時は現行挙動（GC のみ）を一切変えない。
- **依存**: §10 ステップ2-A（非同期 offload）が入ると `offload` モードの iter コストが ≒0 になり、
  ディスパッチの発火閾値を下げられる（より積極的に offload 可能）。escalate の micro 分割は §3案1 を
  「OOM 起因ではなくサイズ予算起因で先制発火」させる形（§3 既述）で再利用。

### 11.6 PoC ファイル一覧（再開時の手掛かり）
| ファイル | 役割 |
|---------|------|
| `tmp/profile_highres.py` / `profile_fa.py` | activation が高解像度で支配的・FA 必須を実証 |
| `tmp/test_B2_offload_gc.py` | **現行実装（GC on）基準**の offload peak 削減（旧 test_B は GC 欠落の瑕疵） |
| `tmp/test_B3_offload_timing.py` | 同期 offload の iter コスト（+3〜10%、高解像度ほど小） |
| `tmp/test_B4_aspect_dispatch.py` | 予測モデル `bs×面積` の検証（coef 1.03倍・aspect 非依存） |
| `tmp/test_B5_dispatcher_poc.py` | 三段ディスパッチャ PoC（予測＋校正キャッシュ＋fast/offload/escalate） |

注意: 旧 `tmp/test_B_offload.py` は **gradient checkpointing 未有効・bf16** のため現行実装と条件が
異なり、1536px bs4 を偽 OOM と誤判定していた（実機は GC on で 13.1GB に収まる）。比較は B2 以降を使うこと。
