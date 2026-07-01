# FLUX.2: NAG × Block Swap 統合 forward 設計（案2）

FLUX.2 で NAG と Block Swap を併用可能にするための設計。他5アーキと同じ「**swap フックを持つ
forward ループは1つだけ、NAG はその上に載る**」構造へ収束させる。

## 1. 背景

FLUX.2 だけ NAG（`Flux2NAGWrapper`）と Block Swap（`Flux2BlockSwapWrapper`）が**それぞれ
独立に forward 全体を再実装**しており、`flux2.py:602` で相互排他が enforce されている。他アーキ
（Z-Image/Ideogram4/Anima/Lens/MiniT2I）は swap フックを単一 forward に持ち NAG が委譲/層で載る。

両 forward は diffusers Flux2 forward のミラーで**構造が同一**、差分は直交している:

| 段階 | swap wrapper | NAG wrapper |
|---|---|---|
| temb | `timestep*1000`（batch img_b） | `timestep[:1]*1000`（batch1, broadcast） |
| dual ループ | `wait_for_block`/`submit` + grad-ckpt 分岐 | NAG proc が attention 内でバッチ処理（ループは同一呼び出し） |
| dual 後 | そのまま concat | `do_nag` なら `_expand(image)` してから concat |
| single ループ | `wait_for_block(unified)`/`submit` | single proc に `encoder_hidden_states_length`/`origin_img_batch` 設定 |
| controlnet(single) | そのまま加算 | `do_nag` なら `_expand(sample)` |
| 末尾 | text 除去のみ | text 除去 + `[:img_b]` で NAG グループ集約 |

**重要**: dual/single のブロック呼び出しシグネチャは NAG でも swap でも同一。NAG のバッチ倍化は
attention **processor 内部**（`NAGFlux2AttnProcessor` / `NAGFlux2ParallelSelfAttnProcessor`）で
行われ、ループ本体は変わらない。よって swap フックと NAG バッチ処理は同一ループに共存できる。

## 2. 統合 forward の設計

**単一ラッパー**（`Flux2BlockSwapWrapper` を統合ラッパーに拡張。`Flux2NAGWrapper` は processor
設置＋統合ラッパー生成の薄いヘルパーに縮退）:

状態:
- `self._block_offloader`（optional）— swap 有効時のみ。
- `self._nag_single_procs`（optional, list）— NAG 有効時に `set_nag_flux2_processors` が返す
  single-stream processor 群（毎 forward に length を設定する必要があるため保持）。
- NAG scale/tau/alpha は processor に埋め込み済み（forward では不要）。

`forward` ロジック（統合）:
```
swap_on = offloader is not None and offloader.blocks_to_swap > 0
img_b = hidden_states.shape[0]; txt_b = encoder_hidden_states.shape[0]
do_nag = txt_b > img_b            # NAG proc 設置時のみ真になる（pipeline が text を倍化）

temb = time_guidance_embed(timestep[:1]*1000, guidance[:1]*1000)   # batch1 broadcast（両対応）
... x_embedder / context_embedder / concat_rotary_emb（現行同一）...

if self._nag_single_procs:        # NAG 有効時のみ
    for p in self._nag_single_procs: p.encoder_hidden_states_length = num_txt_tokens; p.origin_img_batch = img_b

# dual ループ
for i, block in enumerate(transformer_blocks):
    if swap_on: offloader.wait_for_block(i)
    enc, hs = (grad-ckpt 分岐 or block(...))      # 呼び出しは現行同一
    if swap_on: offloader.submit_move_blocks_forward(i)
    controlnet 加算（現行同一, do_nag 影響なし=dual は image_b のまま）

if do_nag: hidden_states = _expand(hidden_states, img_b, txt_b)
hidden_states = cat([encoder_hidden_states, hidden_states], dim=1)

# single ループ
for i, block in enumerate(single_transformer_blocks):
    uidx = num_dual + i
    if swap_on: offloader.wait_for_block(uidx)
    hs = (grad-ckpt 分岐 or block(...))
    if swap_on: offloader.submit_move_blocks_forward(uidx)
    if controlnet_single: sample = ...; if do_nag: sample = _expand(sample); hs[:,num_txt:] += sample

hidden_states = hidden_states[:, num_txt_tokens:]
if do_nag: hidden_states = hidden_states[:img_b]
norm_out / proj_out（現行同一）
```

## 3. temb を `[:1]` にする妥当性
- 非NAG（img_b==txt_b）でも denoise の各ステップは全バッチ同一 timestep なので、`timestep[:1]` から
  作った modulation（batch1）は broadcast で全 batch に同値適用され、現行 `timestep`（batch img_b）と
  **数値的に同一**。swap-only パスの出力は不変。

## 4. pipeline 配線（`flux2.py` 3経路: txt2img/img2img/inpaint）
- `flux2.py:602` の排他ゲート **`if (nag_active or negpip_active) and enable_block_swap ...` を
  NAG について解除**（NegPip は別途、下記スコープ外）。
- 併用時の順序:
  1. `create_flux_block_offloader` + `prepare_block_devices_before_forward`
  2. NAG 有効なら `set_nag_flux2_processors` で processor 設置（single_procs 取得）
  3. 統合ラッパー生成（offloader + single_procs を渡す）
- swap-only: single_procs なし → do_nag 常に false → 現行 swap forward と同一。
- NAG-only: offloader なし/blocks_to_swap=0 → swap フック無効 → 現行 NAG forward と同一。

## 5. NegPip の扱い（スコープ外だが整合）
NegPip も `Flux2NegPipWrapper` が forward 再実装で、同じ排他ゲートに含まれる。本 PR は **NAG×swap のみ**
統合し、NegPip×swap は後続とする（同じ統合ラッパーに NegPip processor 経路を足せば同様に解決可能）。
ゲート解除は NAG 条件のみに限定し、NegPip 条件は残す。

## 6. 正しさ不変条件（監査項目）
1. **swap-only 不変**: offloader あり・NAG なし → 出力が現行 `Flux2BlockSwapWrapper` と一致
   （temb `[:1]` 等価性含む）。
2. **NAG-only 不変**: NAG あり・swap なし → 出力が現行 `Flux2NAGWrapper` と一致。
3. **NAG+swap 正しさ**: dual/single 両ループで正しい unified index の swap フックが発火し、かつ NAG の
   バッチ倍化・集約が現行 NAG と同じ位置で行われる。weight stream はバッチ非依存なので相互干渉なし。
4. grad-checkpoint 分岐は swap-only（訓練）でのみ意味を持ち、NAG（推論）では通らない。
5. controlnet: dual は image_b 不変で現行同一; single は do_nag 時のみ `_expand(sample)`。

## 7. 検証
- 実機不可（CUDA なし）。統合ラッパーの forward を**構造 diff レビュー**で現行2実装と突き合わせ、
  do_nag=false 経路 / offloader=None 経路が現行と一致することを監査で確認。
- py_compile + import。
- ユーザー実機: (a) block swap のみ（NAG off）で出力が従来と一致、(b) NAG のみで一致、
  (c) NAG+block swap 同時 ON で破綻なく生成・VRAM 削減。

## 8. スコープ
- 対象: FLUX.2 の NAG × Block Swap 併用（txt2img/img2img/inpaint）。
- 非対象（後続）: NegPip × Block Swap、NAG × Block Swap の H2D-only 経路（H2D master は
  processor 設置前に構築される点は他アーキ同様の既知制限）。
