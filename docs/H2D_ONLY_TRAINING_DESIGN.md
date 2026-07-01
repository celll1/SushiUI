# H2D-Only Block Swap for Training (LoRA / frozen base) — 設計ドキュメント

推論向け H2D-only（[[BLOCK_SWAP_AUDIT.md]] §7）を学習時 block swap に拡張するための設計。
実装前レビュー用。**私（Claude）の環境では CUDA も学習も実行できないため数値検証不可** →
本設計は実装時の指針であり、実機検証はユーザーが行う前提。

対象日時: 2026-07（flux2 ブランチ）

---

## 1. ゴールとスコープ

**ゴール**: LoRA 学習（＝ base 凍結）で block swap 時に発生する **凍結 base 重みの D2H 退避を排除**し、
PCIe トラフィックを約半減する。base 重みは optimizer が更新しないため、推論と同じく読み取り専用。

**スコープ（v1）**:
- ✅ **FLUX.2 学習経路（Path A: `FluxBlockOffloader(supports_backward=True)`）** のみ。
- ✅ **LoRA / 凍結 base 前提**（base Linear weight は `requires_grad=False`）。
- ❌ **Full-FT は対象外**（base 重みが更新される → D2H 必須）。検出してフォールバック。
- ❌ **Path B（`LayerOffloadConductor`: Anima/Lens/Ideogram4/MiniT2I 学習）は別 PR**
  （[[BLOCK_SWAP_AUDIT.md]] §3 の通り Path B 自体が要改善）。
- 前提: **gradient checkpointing 必須**（musubi と同条件。理由は §4）。

## 2. 背景: 現行の学習時 block swap 機構

- **作成**: `base_trainer.py:1942/2100` が `create_flux_block_offloader(..., supports_backward=True)`。
- **forward 駆動**: `Flux2BlockSwapWrapper.forward`（`core/models/flux2_block_swap_wrapper.py`）が
  各 block 前に `offloader.wait_for_block(idx)`、後に `offloader.submit_move_blocks_forward(idx)`。
  training 時は wrapper の custom forward が使われる（`blocks_to_swap>0`）。
- **backward 駆動**: `register_backward_hooks()` が各 block に `register_full_backward_hook` を張り、
  `_create_backward_hook(i)`（`flux_block_offloading.py:774`）が backward 中に
  `_submit_block_swap(to_cpu, to_gpu)` + `wait_for_block(to_wait)` を呼ぶ。
  ※ **要確認（実装時）**: training で `register_backward_hooks()` が実際に呼ばれる箇所。
  grep では base_trainer から明示呼び出しが見えないため、呼び出し配線の確認が必要
  （未配線なら backward swap が機能していない可能性 → 現行実装の別課題）。
- **swap 実体**: `submit_move_blocks_forward`/`_submit_block_swap` → `swap_weight_devices`
  （pointer-swap、D2H+H2D）。training では D2H が「更新された重みの永続化」に必要な場合があるが、
  **LoRA では base が凍結なので D2H は冗長**。

## 3. なぜ LoRA 学習で H2D-only が成立するか

- LoRA 学習では `transformer.requires_grad_(False)`（`base_trainer.py:995` 他）で **base 全体が凍結**。
  block swap が動かすのは base の `nn.Linear.weight`（凍結・読み取り専用）。
- LoRA adapter（`lora_down`/`lora_up`）は**別 Linear で trainable**、サイズが小さく、
  optimizer が更新する。これらは **swap 対象にせず GPU 常駐のまま**にする（現行も同様）。
- したがって swap 対象の base 重みは forward/backward/再計算のいずれでも変化しない
  → **D2H は byte-identical な冗長コピー** → 排除可能。

## 4. アクセスパターン（gradient checkpointing 必須の理由）

gradient checkpointing 有効時、backward は各 block の **forward を再計算**してから backward する。
1 ステップの base 重みアクセス列:
- **forward**: block 0 → 1 → … → N-1（各重み 1 回読む）
- **backward**: block N-1 → N-2 → … → 0（**再計算で各重みをもう 1 回読む**）

gradient checkpointing **無効**だと、forward の activation を全保持 → backward で重みは
再計算されず読まれない（勾配は activation から）。しかし block swap で重みを CPU に退避しているため、
backward の勾配計算に必要な base 重みが GPU に無い＝**再計算が前提**。よって musubi 同様
**gradient checkpointing を必須**とし、無効時は H2D-only 学習を拒否（明確なエラー）。

## 5. 設計

### 5.1 凍結重みのみを master 化（LoRA adapter 除外）
`_h2d_linear_modules` の training 版は **`weight.requires_grad == False` の Linear のみ**を対象にする。
- base Linear（凍結）→ H2D master 化・swap 対象。
- LoRA `lora_up`/`lora_down`（trainable）→ **除外**（GPU 常駐のまま、optimizer が通常更新）。
- **Full-FT 検出**: swap 対象 block に `requires_grad=True` の Linear weight が含まれる場合、
  H2D-only 不成立 → 標準 swap にフォールバック（ログ明示）。

### 5.2 双方向リング（overlap 版）
推論は forward 固定順（slot=`i%R`, prefetch `i+R`）。学習は forward→reverse の双方向なので:
- **方向フラグ** `self.h2d_dir ∈ {forward, backward}` を持つ。
  - forward pass 中: `_h2d_submit(idx)` が `idx+1` を prefetch（現行 forward ロジック）。
  - backward pass 中: backward hook 経路が `idx-1` を prefetch。
  - 方向転換（最後の forward block → 最初の backward）は自然に self-heal（§5.3）。
- **slot 割り当て**: 双方向で単純な `i%R` は破綻するため、**block→slot の明示マップ + LRU victim** に
  一般化する（順序非依存で correct）。resident set を `h2d_block_slot: dict[idx→slot]`、
  `h2d_slot_block: list[slot→idx]` で管理。
- **prefetch は最適化**であり、**correctness は §5.3 の同期ロードで保証**（prefetch 失敗・順序ズレでも
  正しい）。

### 5.3 correctness の土台: miss 時同期ロード（順序非依存）
`_h2d_wait(idx)`:
1. `idx` が resident（`h2d_block_slot` にあり、その slot の load 完了）→ weight.data を slot view に向ける。
2. resident でない → LRU victim slot を選び、`idx` の master を**同期 H2D**（cuda.synchronize）→ 割り当て。
これにより **どの順序（forward/backward/再計算）でも常に correct**。prefetch はこの上に載る純最適化。

### 5.4 D2H 完全排除
- `submit_move_blocks_forward` / `_submit_block_swap` の H2D 版は、退避 block を
  **その永続 master に repoint するだけ（コピー無し）**。GPU slot は LRU で再利用。
- backward hook の `_submit_block_swap(to_cpu, to_gpu)` も同様: `to_cpu`→master repoint、
  `to_gpu` は次の `wait_for_block` が（prefetch 済みなら即、未なら同期）ロード。

### 5.5 coalesce（Tier 2C 継承）
推論と同じく block ごとに flat pinned master + flat GPU ring slot（1 DMA/block）。
学習でも凍結 base のみを連結。

## 5.6 実装状況（2026-07）
- ✅ **offloader コア実装済**（`FluxBlockOffloader`, commit 参照）: 学習時 `h2d_only` を許可、frozen-only
  master（LoRA adapter は GPU 常駐で除外）、Full-FT フォールバック、**順序非依存 pull-based residency**
  （block↔slot マップ + LRU、miss 時同期ロード、D2H 無し）。forward/backward/再計算のどの順序でも
  correct。CPU シミュレーションで forward+backward(+recompute) 列を再現し全 config 通過
  （`backend/tmp/test_h2d_only_flux_training.py`）。推論 forward-only パスは無変更。
- ⚠️ **【要対応・既存ギャップ】FLUX 学習の block swap 駆動が未配線**: `base_trainer.py:1942/2100` は
  offloader を生成し `prepare_block_devices_before_forward()` を呼ぶが、**`Flux2BlockSwapWrapper` で
  wrap せず `register_backward_hooks()` も呼ばない**。学習 forward は `self.transformer(...)` を直接呼ぶ
  （`:5986/6219`）。そのため wait_for_block/submit が発火せず、標準 block swap も含め **FLUX 学習の
  block swap は現状駆動されていない**（H2D 以前の問題）。H2D 学習を実際に効かせるには、この配線
  （wrap + backward hooks + grad checkpointing 必須化）を先に整備する必要がある。→ §6 のトレーナー
  配線 follow-up に含める。
- ⏳ SSoT 配線（`TRAINING_DEFAULTS`/OpenAPI/Pydantic/frontend）未実施。

## 6. 統合ポイント（実装時チェックリスト）

1. `FluxBlockOffloader.__init__`: `h2d_only and self.forward_only` のゲートを緩め、
   **training でも h2d_only を許可**（ただし §5.1 の凍結検出を通過した場合のみ）。
2. `_h2d_setup`: training 版は `requires_grad=False` の Linear のみ master 化 + Full-FT フォールバック +
   gradient checkpointing 必須チェック。
3. `_h2d_wait` / `_h2d_submit`: block→slot マップ + LRU に一般化（§5.2/5.3）。推論の固定スロット
   （tested）は forward_only の特例として温存 or 一般版で置換して再テスト。
4. `_submit_block_swap`: h2d_only 時は D2H 無しの master repoint + slot 解放に分岐。
5. `register_backward_hooks` / `_create_backward_hook`: h2d_only 時も同じ hook を使い、内部の
   swap 呼び出しが H2D 経路に分岐することを確認。
6. `base_trainer.py`: FLUX training の `create_flux_block_offloader` に `h2d_only`/`ring_size` を
   config から渡す。gradient checkpointing 無効 + h2d_only ならエラー。
7. **SSoT 配線**: `param_defaults.py TRAINING_DEFAULTS` に
   `block_swap_h2d_only`/`block_swap_ring_size` 追加 → OpenAPI → Pydantic → training_config →
   フロント（CLAUDE.md 手順）。

## 7. correctness 不変条件
- swap 対象 block の重みは 1 ステップ内で不変（凍結）→ D2H 不要。
- `_h2d_wait(idx)` 実行後、`idx` の base weight は必ず GPU slot 上。
- LoRA adapter は常に GPU 常駐で optimizer 更新（H2D master に含めない）。
- 方向転換・prefetch ミスでも §5.3 同期ロードで correct。

## 8. リスク
- **実機検証不可（私の環境）**: autograd backward hook の発火順・grad-ckpt 再計算の相互作用は
  Python シミュレーションで近似検証するが、実 CUDA 学習はユーザー検証必須。
- `register_backward_hooks` の呼び出し配線が現行で機能しているか要確認（§2）。機能していなければ
  それ自体が別バグ。
- LoRA adapter が確実に `requires_grad=True` かつ swap 対象外である前提の検証。
- 混在（一部 trainable base）時のフォールバックが漏れなく効くか。

## 9. 検証計画
1. **Python シミュレーション**（CPU, fake model + LoRA-like trainable adapter）: forward 0→N-1、
   backward N-1→0（再計算相当の 2 回目 read 含む）の呼び出し列を再現し、各アクセスで base weight が
   正しい値・LoRA weight が resident かつ更新反映、を検証。順序非依存 correctness を確認。
2. **フォールバック検証**: Full-FT（base trainable）で標準 swap に落ちること、gradient checkpointing
   無効でエラーになることを単体確認。
3. **ユーザー実機検証**: LoRA 学習 1 run で (a) loss が h2d_only off と一致（bit-exact 近傍）、
   (b) VRAM・iter 速度、(c) 保存 LoRA の健全性、を確認。

## 10. スコープ外（別 PR）
- Path B（Conductor）の H2D-only。
- 双方向 prefetch の高度な最適化（v1 は同期ロード correctness + forward prefetch で可）。
- Full-FT 向けの dirty-tracking 部分 D2H。
