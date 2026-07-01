# Block Swap 実装 監査・設計ドキュメント

本ドキュメントは SushiUI の block swap（ブロック単位の重み CPU オフロード）機構について、
推論・学習の両面から実装の正しさ・効率・堅牢性・他機能との衝突を監査し、改善ロードマップ
（Tier 1〜3 + H2D-only）をまとめたものである。

対象日時: 2026-07（flux2 ブランチ時点）

---

## 0. 実装マップ

### 推論側

| アーキ | Block Swap | 実装形態 | 使用 offloader |
|---|---|---|---|
| FLUX.2 | ✅ | 専用 `Flux2BlockSwapWrapper.forward`（forward 再実装） | `FluxBlockOffloader`（dual/single 2 リスト） |
| Z-Image | ✅ | vendored forward インライン（`self._block_offloader`） | 共有 `TransformerBlockOffloader` |
| Ideogram4 | ✅ | vendored forward インライン（cond/uncond の 2 transformer） | 共有 `TransformerBlockOffloader` ×2 |
| Anima / Lens / MiniT2I | ❌ 未実装 | — | — |

- **U-Net 系（SD1.5/SDXL）は対象外**（本監査のスコープ外）。
- 共有機構: 先頭 `N = num_blocks − blocks_to_swap` を GPU 常駐、残りを stream。移動対象は
  **`nn.*Linear` の `weight` のみ**（bias・norm・buffer は常駐）。
- 駆動: `wait_for_block(idx)`（実行前）→ compute →`submit_move_blocks_forward(idx)`
  （実行後、`idx+1` をプリフェッチ）。cross-stream 順序は
  `current_stream().wait_event(sync_event)`。

### 学習側（推論とは別経路）

| 経路 | 実装 | 対象アーキ | 品質 |
|---|---|---|---|
| Path A | `FluxBlockOffloader`（`create_flux_block_offloader(supports_backward=True)`） | FLUX.2 | 良好（双方向 async prefetch） |
| Path B | `LayerOffloadConductor`（`RingBufferAllocator` + module hooks） | Anima/Lens/Ideogram4/MiniT2I | 要改善（下記） |

主要ファイル:
- `backend/core/memory_management/block_offloading.py`（`TransformerBlockOffloader`, `swap_weight_devices`, `weighs_to_device`）
- `backend/core/memory_management/flux_block_offloading.py`（`FluxBlockOffloader`）
- `backend/core/memory_management/layer_offload_conductor.py`（学習 Path B）
- `backend/core/memory_management/fused_block_swap.py`, `layer_offload_strategy.py`, `transformer_registry.py`
- `backend/core/models/flux2_block_swap_wrapper.py`
- `backend/core/pipeline_backends/{flux2,zimage,ideogram4}.py`

---

## 1. 正しさ（推論）: 3 アーキとも動作する

- CPU 上で block が実行される経路は存在しない。重み未着なら `wait_for_block` の同期
  フォールバックが GPU へ強制ロードするため、結果は常に正しい。
- first-forward 配置も正しい（FLUX.2/Z-Image は共有 `_move_auxiliary_modules_to_gpu`、
  Ideogram4 は pipeline 側 `_ideogram4_setup_block_swap` で明示移動）。

### 検出した堅牢性の穴（クラッシュはしないが要修正）

1. **FLUX.2: dual↔single 境界の class assert 欠落**
   基底 `block_offloading.py` の `assert block_to_cpu.__class__ == block_to_cuda.__class__`
   が FLUX.2 版で欠落。`blocks_to_swap > single_block 数` のとき dual と single を
   name/shape で誤ペアリング → swap 空振り → `wait_for_block` の同期フォールバックで救済
   されるが**黙って直列化**。unguarded。
2. **Z-Image: `blocks_to_swap` 未クランプ**（`zimage.py:473`）。Ideogram4 は
   `[0, num_layers-1]` にクランプ済み。過大値で範囲外の可能性。
3. **`_move_auxiliary_modules_to_gpu` が Z-Image のモジュール名ハードコード**。Ideogram4 は
   自前で回避しているが、将来 `create_block_offloader_for_model` 経由の新アーキが aux 移動を
   忘れると aux が CPU に残る潜在結合。

---

## 2. 効率・オーバーヘッド（推論）: Partial（改善余地大）

**良い点**: 専用 CUDA stream + worker thread + pinned staging（フラグ OFF でも staging は
pinned）で真の async DMA。プリフェッチあり。

**律速要因**:
- **(致命) swap ごとの `current_stream().synchronize()`**
  （`block_offloading.py:326` / `flux_block_offloading.py:432`）— 1 step あたり ~20+ 回、
  毎回 compute stream を全ドレイン。overlap を最も潰している。
- **per-Linear の host `event.synchronize()`**（staging path `:353,:359`）— block 内 ~7 Linear
  を host 同期 ping-pong（A→wait→CPU→B→wait→GPU）で直列化。
- **プリフェッチ深さ = 1** のみ。
- 副次: `swap_weight_devices` が毎 swap ごとに `named_modules()` 全走査（Python オーバーヘッド）。

---

## 3. 効率・堅牢性（学習）

### Path A（FluxBlockOffloader / FLUX.2）: 効率・堅牢とも良好
- backward 方向 prefetch あり（`_create_backward_hook`, `flux_block_offloading.py:576-598`）。
- async stream + event、pinned、`non_blocking=True`。
- 重みのみオフロード（activation は非対象、kohya 流）。
- CLAUDE.md 記載の非互換ガード（Block Swap + Fused Optimizer Groups + 8bit）は
  `base_trainer.py:3504-3514` で `ValueError` として実装済み。✅

### Path B（LayerOffloadConductor / Anima・Lens・Ideogram4・MiniT2I）: 要改善
- **配線ギャップ**: conductor は `transformer._layer_offload_conductor` に付くが、モデル forward は
  `self._block_offloader` しか読まない。→ conductor の in-forward swap 呼び出しは inert、
  **hook 経由でのみ動作**。hook 正しさが load-bearing。
- **backward pre-load hook が無い**: `register_full_backward_hook` は module backward の**後**に
  発火。offload された層を backward 前に GPU へ戻す機構が無く、device mismatch の懸念。
- **forward load が同期**（forward-pre-hook が即 `sync_layer`）→ async stream の overlap 効果ゼロ、
  実効プリフェッチ深さ 0。
- **dead code**: activation offload するが `_restore_activation` に呼び出し無し（復元されない）。
- **dirty-tracking 無し**: 毎 step D2H 無条件退避。

### 横断的リスク（両経路・fused-backward + 8bit）
- grad hook 発火時に CPU 常駐だった param は**黙ってスキップ**
  （`adamw8bit_ringbuffer.py:607, :905`）。「GPU に戻ったとき適用」というコメントは
  deferred-update 未実装。swap された block が grad-hook 時に常に GPU 常駐であることの検証、
  または deferred-update 機構の追加を推奨。
- mixed GPU/CPU optimizer state（ring-buffer commits）は防御的に正しく処理済み
  （CPU のみ pin、条件付き `.cuda()`/copy-back、absmax は CUDA 固定）。✅

---

## 4. 他機能との衝突（推論）

| 機能 | FLUX.2 | Z-Image | Ideogram4 | 判定 |
|---|---|---|---|---|
| **NAG** | 排他（`flux2.py:600-603` で block swap 強制 OFF） | 併用可 | 併用可 | ✅ 衝突は FLUX.2 のみ・enforce 済 |
| **NegPip** | 同上の排他 | 併用可 | 併用可 | ✅ SAFE |
| **Spectrum** | SAFE | SAFE | SAFE | forward 丸ごと skip → swap cycle も atomically skip |
| **FP8 量子化** | SAFE | SAFE | SAFE | fp8 Linear を dtype 非依存で stream + autocast |
| **torchao uint** | ⚠️ RISK | ⚠️ RISK | ⚠️ RISK | subclass weight が `"Linear"` filter を外れ**黙って未 offload**（クラッシュ無し） |
| **LoRA** | SAFE | SAFE | SAFE | LoRA 子 Linear が base block と一緒に offload/復帰 |

- **FLUX.2 の NAG は forward を丸ごと再実装**（swap hook を持たない）ため block swap と両立不可
  → コードで強制排他（正しい）。**Z-Image/Ideogram4 の NAG は swap 統合済み forward に委譲する
  層構造なので併用可能**。→ 「NAG on / block swap on」follow-up は Z-Image/Ideogram4 では既に達成、
  FLUX.2 のみ排他を解くか判断が必要。
- torchao uint のみ未ガード（silent under-offload）。フィルタを class 名から
  「`.weight` を持つ量子化 Linear」を含める形へ拡張するか、明示的にガードすべき。

---

## 5. 改善ロードマップ（CUDA/低レベル）

`copy_(non_blocking=True)` は pinned からなら既に真の `cudaMemcpyAsync`。cupy/ctypes 独自 memcpy
は帯域を増やさない。**勝ち筋は copy primitive 差し替えではなく構造改善**。

### Tier 1（高効果・低コスト）— 本 PR で着手
- **(A) hot path の `current_stream().synchronize()` を撤去** → block ごとに event を record +
  swap stream 側 `wait_event` で順序保証。compute stream 全ドレインを解消。**最大の効果**。
- **(B) staging path の per-tensor host `event.synchronize()` を `stream.wait_event` 順序へ**、
  または default を full-pinned path に切替。約 280 回/step の host↔GPU 往復を削減。

### Tier 2（高効果・中コスト）
- **(C) block 内 per-Linear コピーを 1 個の連続 flat buffer に coalesce**（~7 DMA → 1 DMA）。
  launch overhead ~7× 削減 + PCIe 帯域を飽和。
- **(D) プリフェッチ深さ 2–3（triple buffering）**。
- **(E) D2H/H2D を別 stream に分けて双方向 DMA エンジンを同時使用**（最大 ~2×）。

### Tier 3（状況依存）
- **(F) CPU 側 weight を fp8/int8 保持 → PCIe バイト半減、GPU で upcast**（既存 fp8 と相性良）。
- CUDA Graph は非推奨（動的 event + threadpool で脆く効果薄）。既存 `RingBufferAllocator`
  （8bit optimizer 由来）を primary path のアリーナに流用する方が筋が良い。

---

## 6. Ring-Buffer 8-bit Optimizer との整合性・連携

SushiUI 独自の `adamw8bit_ringbuffer` / `lion8bit_ringbuffer`（`backend/core/training/optimizers/`）と
block swap の関係を整理する。

### 6.1 用語の注意: 2 つの無関係な「ring buffer」
- **`memory_management/ring_buffer_allocator.py::RingBufferAllocator`**: `LayerOffloadConductor` が使う
  **weight（層）用**の CPU バイトアリーナ（学習 Path B）。
- **Optimizer の "Ring Buffer"**: `get_state_buffer` コールバック機構による **optimizer state 用**。

両者は名前と概念が似るだけで**コード上の接続は無い**。混同しないこと。

### 6.2 Ring-buffer optimizer とは（設計）
- オフロードするのは **optimizer state（`exp_avg`/`exp_avg_sq`/`z`）であって weight ではない**。
  state を **UINT8 blockwise 量子化**（blocksize 256）。Lion は `exp_avg` のみ（~87.5% 削減）、
  AdamW は ~75% 削減。
- `absmax1/absmax2`（FP32、dequant メタ）は **CPU offload 時も常に GPU 常駐**が不変条件
  （CUDA kernel 要件、`adamw8bit_ringbuffer.py:301-318`, load 時も強制 `:407-409`）。
- **state 配置は `get_state_buffer` で決定**（`:248`）。`None` → 純 GPU（bitsandbytes 相当、PCIe 無し）。
  callable → CPU buffer は `.pin_memory()`。commit `12bb584` で pin を `.is_cpu` ガード化 →
  **常駐 param は GPU tensor、溢れ分は CPU tensor** という **partial residency（GPU/CPU 混在 state）**が可能。
- **転送 stream は C++ 側**（commit `5a4b71f`, `adamw8bit_cuda.cpp`）: 専用 xfer stream で CPU-resident
  state を H2D、update kernel は compute stream で `CUDAEvent` 待ち、D2H writeback も xfer stream で
  event 順序化。**CPU-resident state のみ**を stream（`any_cpu` ガード）→ GPU 常駐分は転送スキップ。

### 6.3 Block Swap との連携: **独立サブシステム（協調していない）**
- block swap 側（`flux_block_offloading.py` / `block_offloading.py`）は **weight のみ**移動し、
  optimizer state を一切知らない（grep で `optimizer/exp_avg/state/grad` は 0 ヒット）。
- **重要**: shipped code では `get_state_buffer` に non-None を渡す呼び出しが**どこにも無い**
  （`RING_BUFFER_OPTIMIZER.md` も「現状 `get_state_buffer=None` → Fallback」と明記）。つまり
  CPU オフロード機構は実装・ベンチ済みだが **block swap とは自動連携していない**。現状 state は GPU 常駐。
- **fused hook の発火順序**: optimizer 更新は per-param `register_post_accumulate_grad_hook`
  （`:945`）、block swap-out は per-block `register_full_backward_hook`（`flux:579`）。full-backward は
  param の grad hook の**後**に発火するので、**backward 中の当該 block の更新は swap-out より前**に走る
  （その block については順序は正しい）。
- **ただし両 hook とも `p.is_cuda` で CPU param を黙ってスキップ**（step `:607`, fused hook `:905`）。
  「GPU に戻ったとき適用する」というコメントは**aspirational で未実装**（deferred-update queue は無い）。
  grad 完了時に CPU 常駐だった param は**その step の更新が黙って drop**される。

### 6.4 非互換ガードと ring-buffer の位置づけ
- CLAUDE.md 記載の非互換（Block Swap + Fused Optimizer Groups + 8bit）は `base_trainer.py:3504-3514`
  で `raise`。理由: 8bit kernel は `param.is_cuda` 必須（`adamw8bit_cuda.cpp:126`）、fused groups は
  batched `optimizer.step()` を呼ぶが swap で一部 param が CPU へ移動済み → device mismatch。
- ring-buffer 型は **`_setup_fused_backward_pass`（per-param hook）へ特別ルート**され
  （`:3551-3566`）、fused groups を使わない設計。
- **ただし `raise` のリストは literal `"adamw8bit"/"lion8bit"/"adafactor8bit"` のみで
  `"adamw8bit_ringbuffer"` に一致しない** → ring-buffer + block_swap + `num_optimizer_groups>0` は
  **ブロックされず**、`_setup_fused_optimizer_groups` に落ちるが `create_optimizer_groups` が
  `get_state_buffer`/cautious/schedule_free を forward しない → 未テストの設定穴。
- 命名の罠: `adamw8bit_fused.py` は bnb patch で **state は FP32**（真の 8-bit state ではない）。

### 6.5 互換性マトリクス（optimizer × block_swap × fused_groups）

| Optimizer | block_swap | fused_groups | fused_backward(per-param) | 判定 |
|---|---|---|---|---|
| Adafactor | ✅ | ❌（8bit名なら raise） | ✅ | 最小 VRAM 推奨 |
| AdamW / Lion (FP32) | ✅ | ✅ | — | FP32 state・VRAM 増 |
| bnb AdamW8bit/Lion8bit | ✅ | ❌ **raise** | ✅(adamw8bitのみ,FP32 state) | groups 禁止 |
| **AdamW8bit_RingBuffer** | ✅(per-param hook, CPU param skip) | ⚠️未ブロック・未テスト | ✅ 想定経路 | groups=0 で使う |
| **Lion8bit_RingBuffer** | ✅ 同上 | ⚠️同穴 | ✅ | 同上 |

### 6.6 Correctness リスク（ring-buffer + block swap 固有）
1. **"GPU 復帰時に適用" 未実装** → CPU 常駐時に grad 完了した param の更新が silent drop（§6.3）。
2. **state/param の device desync**: state（optimizer）と weight（block swap）は別ライフサイクル。
   現状 state は GPU 常駐・weight のみ CPU へ → fused hook が CPU param を skip して整合を回避
   （正しく扱うのではなく skip で回避）。`get_state_buffer` を CPU に配線しても kernel が
   `param.is_cuda` 必須なので結局 skip。
3. **absmax always-GPU 不変条件**: checkpoint ロードで CPU に落ちると kernel `TORCH_CHECK` 発火
   （custom loader で強制済みだが fragile）。
4. **async D2H とチェックポイント整合**: commit `5a4b71f` が「host で CPU state を読む保存前に
   `torch.cuda.synchronize()` すべき」と警告。save 経路（`base_trainer.py:7654,7889,8172` が
   ring-buffer state を動かさない特別扱い）が sync しているか要確認。

### 6.7 まとめ（設計 vs 実装）
- **設計**: 8-bit CPU-offload optimizer state + 専用 xfer stream overlap + partial residency で、
  per-param fused hook により block swap と共存する意図。
- **実装（現状）**: `get_state_buffer` 未配線で state は GPU 常駐、「GPU 復帰時適用」はコメントのみ
  （CPU 常駐 param は silent skip）、weight ring と state ring は完全独立。→ **実運用は
  `blocks_to_swap>0` + ring-buffer optimizer + `num_optimizer_groups=0`（per-param fused backward）**
  を推奨経路とし、`num_optimizer_groups>0` との併用は避ける（ガード追加が望ましい）。

---

## 7. H2D-Only Block Swap（musubi-tuner 由来の概念）

参照: kohya-ss/musubi-tuner README.ja.md
（`--block_swap_h2d_only` / `--gradient_checkpointing` 必須 / `--block_swap_ring_size` デフォルト 2）。

### 概念
標準 block swap は 1 block/step あたり **D2H（退避）+ H2D（ロード）= 2× block bytes** を PCIe に
流す。しかし推論・ベース凍結学習では対象 block の重みは**読み取り専用**（optimizer が更新しない）
であり、CPU と GPU に同一の重みが存在する。したがって **D2H（GPU→CPU 退避）は完全な無駄**。

H2D-only は **CPU に永続 pinned master を保持し、常に host→device のみ転送**することで D2H を完全に
排除する。PCIe トラフィックを約半減し、per-tensor D2H の event/sync ハンドシェイクと staging-B leg を
削除できる。

> 「ベース凍結の学習ではCPUとGPU上のベース重みが同一のため、従来のブロックスワップにおける
> device→host（D2H）コピーは完全に無駄になります」（musubi-tuner README）

### 現行コードの確認（D2H は冗長）
両 offloader は pointer-swap 設計で、使用済み block の GPU weight を CPU へ書き戻す:
- staging path: `sbuf_a.copy_(cuda_data_view.data, non_blocking=True)`
  （`block_offloading.py:348`）→ `cpu_data_view.copy_(sbuf_a)`（`:367`）。
- pinned path: `module_pin_buf.copy_(cuda_data_view, non_blocking=True)`（`:396`）。
- FLUX.2 も同型（`flux_block_offloading.py:447/466/489`）。

`supports_backward=False`（推論）では forward が weight を変更しないため、この書き戻しは
byte-identical な冗長コピー。

### 適用可否

| シナリオ | 可否 | 理由 |
|---|---|---|
| **推論** | ✅ 強く推奨 | 重みは完全に読み取り専用。D2H 削除で PCIe ≈ 半減・stream 片方向化 |
| **学習 LoRA（ベース凍結）** | ✅ 有利 | swap 対象の base block は凍結 = 読み取り専用。musubi と同条件 |
| **学習 Full-FT** | ⚠️ 条件付き | optimizer が GPU 常駐 param を毎 step 更新 → 更新された重みの D2H 永続化が必要。naive H2D-only は更新を喪失 |

### 必要な実装変更（pointer-swap → fixed GPU ring）
現行は `weight.data` ポインタを 2 block 間で交換し GPU buffer が CPU home へ移動するため、
D2H 行を消すだけでは pinned master を破壊してしまう。正しい H2D-only 設計:
- swap 可能 block 数ぶんの**永続 GPU weight buffer リング**（`ring_size` 個、デフォルト 2）を確保。
- block ごとに**永続 pinned CPU master**を確保し、**一切書き込まない**。
- 各 swap: 空いた GPU ring buffer へ次 block の CPU master を H2D コピーのみ、
  `next_block.weight.data = gpu_ring_buf`。退避 block は CPU master へポインタを戻すだけ（コピー無し）。
- 影響範囲: `swap_weight_devices`（両ファイル）+ `prepare_block_devices_before_forward` の buffer 確保。
  `wait_for_block` / `submit_move_blocks_forward` / futures / streams は不変。
- Full-FT で使う場合は「更新された block のみ D2H」の dirty-tracking か、CPU-master 更新モデルへの
  optimizer 側変更が必要（別 PR）。gradient checkpointing 必須（musubi と同じく、backward 中の
  再計算で weight を再読するため）。

### リスク
- gradient checkpointing 再計算: weight は backward 中も不変なので H2D-only の読み取りは安全
  （リスクは Full-FT の更新永続化のみ）。
- LoRA: adapter は更新されるが小さく常駐。swap される base block は凍結 = 読み取り専用 →
  **学習で H2D-only が無条件に安全な唯一のシナリオ**。
- 量子化: 読み取り専用の量子化重みは安全（weight-only-FP8 は weight を buffer に持つ点に注意、
  ring rewrite でも同じ属性処理を保持すること）。

---

## 8. 結論と次アクション

1. **正しさ**: 推論 3 アーキとも動作。堅牢性の穴（FLUX.2 assert / Z-Image クランプ /
   aux-mover ハードコード / torchao 未ガード）は現状フォールバックで救済されるが要修正。
2. **効率**: overlap+pinned で naive `.to()` より良いが、per-swap 全同期・per-Linear host 同期・
   深さ 1 でパイプライン性能を出せていない。Tier 1(A)(B) だけで大きく改善可能。
3. **学習**: FLUX.2(Path A) は良好、Path B（Conductor）は backward pre-load 欠落・同期 forward・
   dead code で要改善。
4. **Ring-buffer optimizer**: state offload（8-bit・専用 xfer stream・partial residency）は堅実だが
   block swap とは**独立サブシステム**で、`get_state_buffer` 未配線・deferred-update 未実装
   （CPU 常駐 param は silent skip）。推奨経路は `blocks_to_swap>0` + ring-buffer +
   `num_optimizer_groups=0`。`num_optimizer_groups>0` との併用ガードが `_ringbuffer` 名を
   カバーしていない穴あり。
5. **H2D-only**: 推論と LoRA 学習で有利、Full-FT は条件付き。fixed GPU ring への書き換えが前提。

### 実施順
- **本 PR**: 本ドキュメント(D) → Tier 1(A) 効率改善（per-swap 同期を event-based 順序へ置換）。
- **後続候補**: Tier 1(B)、堅牢性修正（含: ring-buffer + fused_groups ガードの `_ringbuffer` 名拡張）、
  H2D-only（推論 + LoRA）、Path B（Conductor）改善、
  「NAG on / block swap on」FLUX.2 排他解除判断。
