# SenseNova U1.5 学習設計案

> Status: Phase 0 / Phase 1 / Phase 3 / Phase U-0 / Phase U-1 は完了。
> Phase 2b と Phase U-2 / U-3 は未完（U-2-1 = 2b-1 が `cc296e84`、
> U-2-2 の step 1-2 が `601d0271`、U-2-3 が `24220b5c`、U-2-6 が `e6bdcc38` で着地）。
> **【U-2-2 step 3 着地】full FT の受付は開いた** — `TRAINING_UNSUPPORTED
> ["sensenova"]["full_finetune"]` と `train_runner` の `network.type != "lora"`
> 拒否の**両方**を落とし、実 checkpoint 上の smoke run で端から端まで通した
> （gen branch、adafactor、B1、3 step、**census 294/294**、mixed checkpoint
> 25.129 GiB を保存し**本番 reader で 294/294 バイト一致で再ロード**。§13.4
> U-2-2 の実測ボックス）。品質は主張しない。
> **【U-2-4 / U-2-5 着地】** 4 相分割（`071e602b`）に続き、**U-2-5 の exit smoke が
> 3 branch すべてで通った** — und branch **289/294**、both branch **583/588**、
> 動かない 5 個はいずれも `und_gradient_unreachable_paths()` が名前で予測した
> layer 41 の und 側であり、`mixed` の**両向き**と `bf16` を本番 reader で
> バイト一致再ロードした（§13.4 U-2-5）。品質は主張しない。
> **Phase U-2 は offload 合成（2b-4 / §8.3.1）を除いて完了**である。
> それ以外の未測定事項は §13.4 U-2-5 末尾に列挙してある
> （品質・収束・解像度上限・`int8` 形式の往復と resume・保存 checkpoint での生成・
> und × reference・MNT>1 と学習中 sample）。
> Date: 2026-08-25
> Scope: SenseNova-U1.5-8B-MoT の (1) LoRA 学習 / (2) full-parameter fine-tune /
> (3) reference 画像を含むデータセットの混在学習
> 本文中の `file:line` は 2026-08-23 時点の静的調査による（§6.4 と §13.4 U-2 の
> 参照だけは `cc296e84` / `601d0271` 時点で取り直してある）。

この文書は設計判断、その根拠、実装状況を記録する。初期計画のフェーズ順は履歴として
残すが、各節の **DONE / PENDING** が現在の境界である。SenseNova の推論と Phase 1
LoRA は `ARCH_REGISTRY` を含む学習経路へ統合済みで、一般的な追加手順は
[`ADD_A_MODEL_ARCHITECTURE.md`](ADD_A_MODEL_ARCHITECTURE.md)、現行の architecture
facts は [`MODEL_FACTS.md`](MODEL_FACTS.md) を正とする。本文書は SenseNova 固有の
差分だけを扱う。

---

## 1. Executive decision

1. **Phase 1（LoRA）は generation branch のみを対象とし、core/trainer integration
   まで実装済み。** 既存の推論側 LoRA
   (`sensenova_lora.py`) が列挙する 294 個の `_mot_gen` Linear をそのまま学習対象と
   する。understanding branch は凍結し、prefix forward は `no_grad` で回す。
2. **Phase 2a（full FT の拒否ガード）は出荷済み。** 配布されている checkpoint は
   int8 のみで、既知の非対応はモデルロード前に拒否する。
   実装する場合の既定は **gen branch のみの 8.1B**（**【2026-08-24 改訂】** 旧文の
   「both-branch 16.2B は設計対象外」は §6.2 で撤回済み。既定が gen-only なだけで、
   both は閉じていない）。前提条件は「bf16 base の入手」ではなく「配線の実装」で
   （§6.4）、その最初の一片 — int8 base を materialize する経路 (a) — は
   **`cc296e84` で、materialize した half を collect する adapter と契約は
   `601d0271` で、U-2-3（stochastic rounding + dropout guard）は `24220b5c` で
   着地した。ただし受付はまだ開いていない**（残るのは出力 checkpoint format の
   決定と、ユーザーに届く通知経路。§13.4 U-2-2）。
3. **Phase 3（reference 混在）は per-item presence を真とする。**（**DONE**、
   `7a09af52`..`611a4a24`。以下の判断はすべてそのまま実装された。）初版は物理
   `batch_size=1` を強制し、gradient accumulation で effective batch を作る。
   `separate_by_reference` は sampler 整理のため再利用するが、prefix shape の保証には
   使わない。it2i 挙動の学習に understanding branch の解凍は既定では不要とする。
4. **VRAM 機構は Phase 1 が MoT half-eviction、Phase 2b が
   `LayerOffloadConductor`**（学習側の offload 機構。`TransformerBlockOffloader` は
   forward-only inference 用で、以前ここに書かれていたのは誤り — §8.3）という分担に
   する。推論側で記録された block-swap 非対応の判断は generation 固有であり、学習には
   転移しない（§8）。ただし 2 機構の合成にはモジュール粒度の未解決問題がある（§8.3.1）。

§5-§7 の設計判断は fable への設計相談を経て確定したものであり、根拠は各節に併記する。

---

## 2. Constraints and non-goals

### Constraints

- **配布 checkpoint は int8 のみ。** `sensenova_int8.safetensors`
  (17.58 GiB) と ConvRot 版の 2 本。588 個の Linear が `Int8Linear` で、weight と scale は
  `nn.Parameter` ではなく **buffer** として保持される
  ([`loader.py:27-38`](../../backend/core/models/sensenova/loader.py))。
- **`NEOChatModel.forward` は `raise NotImplementedError('forward')`**
  ([`modeling_neo_chat.py:289`](../../backend/core/models/sensenova/vendor/modeling_neo_chat.py))。
  学習 forward は存在しない。推論は `t2i_generate` 相当の helper 群を
  `sensenova_pipeline_ops.py` から直接駆動している。
- **mixed und/gen forward は実装されていない。** attention 階層
  ([`modeling_qwen3.py:983-987`](../../backend/core/models/sensenova/vendor/modeling_qwen3.py))
  と decoder-layer 階層（`:1217-1220`）の両方で `NotImplementedError` を送出する
  （upstream issue #207、parity test なし・production caller なし）。コード自身が
  "Split the sequence at token-type boundaries and use forward_und / forward_gen" と
  指示している。
- **pixel space、VAE なし。** latent cache も VAE 常駐もない代わりに、activation は
  pixel 解像度に比例する。
- 別個の text encoder は存在しない。prompt は denoiser と同じ Qwen3-8B が
  自前の tokenizer / chat template で符号化する。
- 単一 GPU 前提。参照カードは RTX 6000 Ada 48 GB。
- 初版の物理 batch size は 1。effective batch size は gradient accumulation で作る。
- 推論側の最適化（MoT phase eviction / KV cache streaming / ConvRot W8A8 /
  style transfer / FBCache / Spectrum）はすべて generation 専用で、学習には
  持ち込まない（§8 の half-eviction の概念だけが例外）。

### Non-goals

- both-branch（16.2B）の full fine-tune を設計しない（§6.2）。
- understanding-only LoRA を提供しない（推論側で検証する手段がない）。
- int8-resident full FT（int8 code を直接更新する学習）を提案しない。ただし本リポジトリで
  棄却済みだからではなく、**そもそも調査されていない**からである（§6.3.1）。
- upstream issue #207 の mixed forward を修正して 1 パス化することを前提にしない。
  修正できれば単純化されるが、parity test がない経路に設計を依存させない。
- ControlNet / ReLoRA / tagger は本文書の対象外。

---

## 3. Current state

| 領域 | 現状 | 状態 |
|---|---|---|
| `ModelType` 検出 | `"sensenova"` は既に `ModelType` にあり `detect_model_type` も返す（`model_loader.py:13, 643-644`） | 不要 |
| `ComponentWiringSpec` | `SENSENOVA_WIRING` を training shim から re-export 済み | DONE |
| LoRA target / adapter | 実 checkpoint で 294 target を検証し、推論・学習 adapter と保存 round-trip を実装済み | DONE |
| `ARCH_REGISTRY` | SenseNova を含む 13 arch | DONE |
| `arch/sensenova.py` / `ops/sensenova_ops.py` / `adapters/sensenova_adapter.py` | Phase 1 の loader、prefix、pixel-space step、LoRA adapter を実装済み | DONE |
| `base_trainer.py` / `train_runner.py` | B1、単一 flavour の int8 base、no-reference、no-block-swap の初版契約と専用 prefix payload を統合済み | DONE |
| `detect_prediction_config` | `sensenova` を flow / velocity として登録し退行テスト済み | DONE |
| `TRAINING_UNSUPPORTED` | full FT / ReLoRA / ControlNet をロード前に拒否 | DONE |
| real trainer exit smoke | 3 finite steps、runtime strength 0 exact parity、294 apply / restore を実 checkpoint で検証済み | DONE |
| half-eviction | training 専用 driver、opt-in API/UI、実 checkpoint OFF / ON 測定を完了 | DONE |
| 学習中 sample / `debug_latents` | 推論の prefix + Euler loop をそのまま駆動する `generate_sample` と、pixel space の debug dump を実装済み（`dc91bef1`）。`sample_every` の強制 0 は解除 | DONE |
| reference 混在（Phase 3） | ゲート 6 箇所中 4 箇所を解除（残り 2 は意図的に flux2 限定）、prefix への ViT token splice、学習中 sample の ref 対応、実 checkpoint の混在 smoke まで完了（`7a09af52`..`611a4a24`） | DONE |
| full FT（Phase 2b） | gate/loader の method-aware 化（`cc296e84`）、adapter + 契約 + fused backward の decoupling（`601d0271`）、stochastic rounding の強制 + dropout guard（`24220b5c`）は着地。通知経路（`training_log`）も着地。**残るのは出力 checkpoint format の決定と受付の解錠**（§6.4、§13.4 U-2-2） | PENDING |
| understanding branch の LoRA（Phase U-0 / U-1） | `train_text_encoder` で選択（既定 OFF）。微分可能 prefix、branch 対応の単一列挙器、推論側の und 適用、assert 分離、実 checkpoint の exit smoke まで完了（`3d837202`..`327276df`） | DONE |
| understanding branch の Full-FT / reference 併用（U-2 / U-3） | §13.4 | PENDING |

---

## 4. 設計を規定する SenseNova の構造的事実

以降のすべての判断はこの節の事実から導かれる。

### 4.1 MoT による重み二重化

42 の decoder layer それぞれが understanding 半分（素の名前）と generation 半分
（`_mot_gen` 接尾辞）を持ち、比率はちょうど 50/50、layer あたり 386,221,056 bytes、
合計 15.11 GiB / 半分あたり 7.55 GiB
([`mot_phase_eviction.py:11-13`](../../backend/core/models/sensenova/mot_phase_eviction.py))。
int8 で 1 byte/param なので **decoder Linear の総パラメータ数は約 16.2B、片側 8.1B** で
ある。「8B」はあくまで単一 branch 側の規模であり、full FT の見積りを 8B で行うと
2 倍外す。

二重化されるのは q/k/v/o_proj、q/k norm（t 軸・hw 軸）、`mlp` 全体、
`input_layernorm` / `post_attention_layernorm`、および `Qwen3Model.norm`。
**共有されるのは** `self_attn` モジュールオブジェクト自体、`rotary_emb` /
`rotary_emb_hw`、`embed_tokens`、`lm_head`。

接尾辞の位置が attention と MLP で異なる（attention は Linear 自身、MLP は親モジュール）
点は既に `sensenova_lora.py:32-49` が文書化しており、adapter はその列挙器を
再利用することで再実装しない。

### 4.2 branch 選択は per-token mask ではなく 3 分岐

`image_gen_indicators`（bool `[B,S]`）は layer ループに入る前に 2 つの Python bool
へ畳まれ（`modeling_qwen3.py:1319-1328`）、以降は「全部 und」「全部 gen」「混在」の
3 分岐になる。混在は前述のとおり未実装。

結果として production の forward は必ず **2 パス**である。

1. **prefix phase** — text（および it2i の reference 画像）token。全 layer が
   `forward_und`。eager attention 固定。per-layer KV cache を構築。
2. **denoise phase** — image token のみ。全 layer が `forward_gen`。flash 経路、
   image token 間は `causal=False`、`cat[prefix_KV(und), current_KV(gen)]` に attend。

**この 2 パス構造が Phase 1 の設計を事実上決めている。** understanding branch を
凍結すれば phase 1 を `no_grad` で回せ、「2 つの forward をまたいで KV cache に
勾配を通す」という問題自体が消滅する。

### 4.3 目的関数は x0 予測の flow matching

head (`fm_head`、`use_pixel_head: true` なので `ConvDecoder`) が clean pixel `x_pred` を
直接出力し、速度は代数的に導出される：

```
v_pred = (x_pred - z) / (1 - t).clamp_min(t_eps)     # modeling_neo_chat.py:655
z_t    = t * x0 + (1 - t) * noise_scale * eps
```

`t=0` が noise、`t=1` が clean。これは flux2 の sigma 方向とは逆で、**MiniT2I と同一の
規約**である。したがって学習 step の骨格は
[`ops/minit2i_ops.py:281`](../../backend/core/training/ops/minit2i_ops.py) の
`train_step`（pixel space・x0 予測・`v=(x0_pred-x_t)/(1-t)` の velocity loss）が
そのまま構造テンプレートになる。差分は conditioning が prefix KV である点のみ。

### 4.4 noise_scale は解像度依存で、かつモデルに入力される

`compute_noise_scale`（`sensenova_pipeline_ops.py:133-144`）は config の基礎
`noise_scale` に解像度比を掛け、mode に応じて平方根を追加適用し、
`noise_scale_max_value` で clamp する。現在の配布 config
（`noise_scale=1`, `mode=resolution`, max 16）では
`sqrt(grid_h*grid_w/merge^2 / base)` と同値になる。
この値は forward noising に使われるだけでなく、`noise_scale_embedder` を通して
timestep embedding に**加算されて**モデルに渡る（`:625-630`）。

**bucketing との相互作用が実装上の落とし穴になる。** bucket ごとに解像度が違う以上
`noise_scale` はサンプルごとに変わる。学習側は式を再実装せず
`compute_noise_scale()` を再利用する。推論とずれると、
学習時と推論時で条件付けがずれる。これは「よくある定数」ではなく、per-sample に
計算して embedder にも流す必要がある値である。

### 4.5 学習用 timestep 分布は config に既に書かれている

`config.json` の `P_mean: -0.8`, `P_std: 0.8` は推論経路のどこからも読まれていない
不活性フィールドであり、upstream の学習時 lognormal サンプラのパラメータと考えて
まず間違いない。そして本リポジトリの MiniT2I の既定値は
`logit_normal(mean=-0.8, std=0.8)` で**数値が一致する**
(`param_defaults.py:2718-2719`)。したがって
`TIMESTEP_SAMPLING_DEFAULTS_BY_ARCH["sensenova"]` は推測ではなく config 由来の値として
登録できる。

### 4.6 vision tower は 2 つあり、正規化が異なる

- `vision_model`（understanding tower、**ImageNet 正規化**）— it2i の reference 画像
  専用。prefix に `<img><IMG_CONTEXT>*N</img>` として差し込まれる。
- `fm_modules['vision_model_mot_gen']`（generation tower、**0.5/0.5 正規化**）—
  毎 step、現在のノイズ画像 patch に対して走る。

Phase 3 のデータパイプラインはこの 2 つの正規化を取り違えてはならない。同じ
「画像」でも入口が違えば前処理が違う。

### 4.7 学習用 KV / gradient-checkpointing 契約

`update_cache=False` には 2 経路ある。推論が準備する `flash_k_cache` /
`flash_v_cache` または streamer がある場合は `.copy_()` で current K/V を書くため、
autograd と両立しない。一方、それらを**準備しなければ** fallback は
`torch.cat([past, current])`（`modeling_qwen3.py:784-796`）になり、prefix cache を
変更せず current K/V の勾配を保持する。**学習はこの fallback を使う。**

`update_cache=True` は代替にならない。`DynamicCache.update()` が current K/V を
破壊的に append するため、checkpoint backward の再計算で二重 append される。
また stock `GradientCheckpointingLayer.__call__` は checkpointing 中の
`past_key_values` を `None` に置換するため、通常の
`gradient_checkpointing_enable()` は prefix conditioning を静かに消す。

したがって学習は layer 標準 checkpointing を使わず、SenseNova 専用の per-layer
non-reentrant checkpoint loop を持つ。immutable な prefix cache を closure で渡し、
`update_cache=False`、推論用 flash cache/streamer 未準備を不変条件とする。

---

## 5. Phase 1 — LoRA 学習（DONE）

### 5.1 採用する方式

- 学習対象は **generation branch の 294 Linear のみ**
  （42 層 × {q,k,v,o}_proj_mot_gen + mlp_mot_gen.{gate,up,down}_proj）。
- understanding branch、両 vision tower、`embed_tokens` / `lm_head` は凍結。
- 学習 step は 2 パス構造を推論と同じ順序で踏む。
  1. caption（+ Phase 3 では reference token）を `forward_und` で `no_grad` 前進させ、
     prefix KV を得る。
  2. 目標画像 `x0` から `t` を引いて `z_t` を作り、`extract_feature(gen_model=True)` →
     timestep/noise_scale embedding 加算 → `forward_gen` を勾配付きで前進。
  3. `fm_head` の `x_pred` から `v_pred` を作り、target velocity との MSE。
- 物理 `batch_size` は 1 に限定し、effective batch は既存の gradient accumulation で
  作る。prefix-aware batching は padding mask / varlen attention が実装されるまで開かない。
- gen pass は §4.7 の専用 non-reentrant checkpoint loop を使う。prefix cache は
  forward/backward 前後で長さ・tensor identity が変わってはならない。
- timestep sampler は §4.5 の値を既定にする。
- caption embedding のキャッシュ: prefix forward は Qwen3-8B 全体を通す高コスト処理
  なので、他 arch の TE cache と同様に prefix KV を事前計算・キャッシュする余地が
  ある。ただし KV は caption ごとに 42 層 × 全 token 分あり容量が大きい。**初版では
  キャッシュせず毎 step 計算し、実測してから判断する**（§12 の open question）。

### 5.2 確定した設計判断 — LoRA scope（fable 諮問）

**判断: generation branch のみ。scope enum の内部構造だけ用意し、UI には出さない。**

根拠（fable の推論をそのまま記録する）:

1. **und を凍結することは VRAM と忘却対策の話に留まらず、正しさの問題クラスを
   丸ごと削除する。** und を凍結すれば prefix forward は `no_grad` で回せるので、
   prefix KV を定数として扱える。gen forward は §4.7 の非破壊
   `update_cache=False` fallback で current K/V の勾配を保つ。und を学習させるなら、
   2 つの独立した forward をまたぐ微分可能な KV パイプラインを通し、42 層分の
   prefix activation を backward まで保持し、`cat[prefix_KV, gen_KV]` の flash
   attention に勾配を通す必要がある。これは phase 1 の LoRA に対して費用対効果が
   釣り合わない。
   - **【2026-08-24 改訂】この根拠のうち「微分可能な KV パイプラインを*構築*する
     必要がある」は過大評価だった（§13.1）。** 経路は既に微分可能で、必要なのは
     *解錠*である。他方「42 層分の prefix activation を backward まで保持する」
     コストの指摘は正しく、しかも当時の想定より重い — 量子化 base では
     **und 側 294 個の `Int8Linear` が backward 用の dequant 済み bf16 weight を
     autograd に保存する**からである（§13.2）。「フラグではなくサブシステム」という
     評価は、und LoRA 全体ではなく **3 つの具体的な部品**にのみ当てはまる。
2. **und branch はこのアーキテクチャにおける text encoder そのものである。**
   本リポジトリの他 arch はすべて既定で TE を凍結している。SenseNova はその
   「text encoder」が同じ decoder layer の中に交互に格納されているだけで、
   そう捉えれば gen-only は保守的な部分集合ではなく**通常の学習契約**であり、
   und 学習の方が独自の正当化を要する例外である。
3. **checkpoint 互換性。** 推論側 loader と唯一公開されている distillation LoRA が
   294 module の名前空間を定義している。und を含む LoRA は前例ゼロの新方言であり、
   推論側の配線も新規に必要になる。消費者が居ない段階で形式を増やさない。
4. 「und は prefix KV を通じて出力品質に効く」のは事実だが、gen branch は全層・
   全 step でその KV に attend しており、凍結された prefix 表現に**適応する**機会を
   十分に持つ。凍結 TE に cross-attention が適応するのと同じ構図で、公開されている
   gen-only の distillation LoRA がそれが機能することを実証している。

**保留（deferred）→ 【2026-08-24】Phase U として提供する（既定 OFF）。**
`scope: generation | both` は「Phase 3 で reference 忠実度が不足した場合のみ開ける」
保留項目だったが、**「新規コンセプトの追加には understanding 層の学習が重要」という
明示的な機能要求**を受け、ユーザー選択式のオプションとして立てる（§13）。
**この改訂は上の判断と矛盾しない** — 既定は OFF のままで、gen-only が通常の学習契約で
あるという位置づけ（根拠 2）も、既定経路の checkpoint 互換（根拠 3）も変わらない。
変わるのは「選択肢を提供するか」だけである。`understanding-only` は用途が無く
推論側で検証もできないため**恒久的に提供しない**（保存側で gen 0 件を拒否する）。

### 5.3 int8 base 上の LoRA は追加作業なしで成立する

`reject_quantized_base()` の docstring が明示するとおり、LoRA は量子化 base で
意図的に許可されている。`LoRALinearLayer` は量子化モジュールを**包む**だけで、
その weight を微分しない。学習されるのは adapter 自身の float パラメータのみ。
Ideogram 4 / Krea 2 / FLUX.2 / Anima で既に成立している経路であり、SenseNova の
`sensenova_lora.py` はその Ideogram 4 を明示的に踏襲している。

この経路は本リポジトリで既に実証されている。Anima の probe では 341 個の wrapper の
うち 230 個が量子化 Linear の上に載り、`lora_up` に実際の勾配が出ている。
また学習側の `load_components` は他 arch で `disable_int8_mm()` / `disable_scaled_mm()`
を呼び、`_int_mm_forward` の GATE も grad mode が有効なら W8A8 を拒否する。
SenseNova の plain int8 checkpoint は**ロード時点で既に `disable_int8_mm` されている**
（W8A8 は数値退行のため pin off）ため、この点は追加作業なしで整合する。

**ConvRot checkpoint も受理する（`0c9ea86b`）。** 受理条件は「588 個の decoder Linear が
**すべて単一の量子化 flavour**であり、それが `Int8Linear` か `ConvRotInt8Linear` の
いずれかであること」（`ops/sensenova_ops.py` の
`_assert_supported_quantized_training_base`）。census は `isinstance` ではなく
`type(m) is cls` で数える — `ConvRotInt8Linear` は `Int8Linear` の subclass なので、
`isinstance` census は ConvRot を plain-int8 の数に畳み込み、**mixed base を黙って
受理してしまう**。`Fp8Linear` / `W4A8Linear` は診断メッセージ用に census されるが
**受理しない**（該当する SenseNova base が存在せず、受理すれば未検証の経路を出荷する
ことになる）。既知量子化クラスの未知 subclass は名指しで拒否し、mixed・端数・
未量子化 bf16 base も拒否する。

**ただし ConvRot には train / inference の activation 量子化 skew がある（未測定）。**
`ConvRotInt8Linear.forward` は
`self._force_dequant or (torch.is_grad_enabled() and x.requires_grad)` で分岐する
（`convrot_int8_linear.py:72-74`）。学習では LoRA 寄与が activation に入った時点以降、
ほぼ全 Linear が微分可能な dequant 経路（W-int8 / A-bf16）を通る。一方**推論では常に
fused W8A8 kernel が走る** — `disable_int8_mm` はこのクラスに対して inert で、
`forward` はそのフラグを読まない（`loader.py` の QUANTIZATION 節）。したがって
**ConvRot LoRA は、デプロイされるベースとは activation 量子化誤差の分だけ異なる
ベースに対して fit される**。

この skew は**誰も測っていない**（ConvRot で学習した LoRA を fused 推論カーネル下で
A/B した例は無い）。既知の制約として記録するに留め、実害があるとも無いとも断定しない。
**autograd 自体は整合している**: fused 経路は勾配が不要な箇所でしか走らず、§4.7 の
non-reentrant checkpoint の再計算は初回と同じ述語を評価する。plain int8 checkpoint に
この skew は無い — `disable_int8_mm` は真の `Int8Linear` に効き、W8A8 は数値退行のため
既定で pin off なので、学習・推論とも dequant を通る。

ただし `warn_quantized_base_without_checkpointing()` の条件に注意する。gradient
checkpointing を **OFF** にすると、量子化 Linear ごとに backward 用の compute-dtype
weight が autograd に保存され、int8 codes の上に bf16 のモデル全体が実体化する。
G4 の実測（Krea 2 由来）では checkpointing OFF で
`11.94 GiB codes + 23.88 GiB temporaries = 35.81 GiB` 対 bf16 base 23.88 GiB という
逆転が出ている。SenseNova では 588 個すべての decoder Linear が該当するため、
**gradient checkpointing は事実上必須**である。ただし `Qwen3DecoderLayer` が継承する
stock `GradientCheckpointingLayer` は prefix cache を除去するため使えない。
`supports_gradient_checkpointing = True` という宣言だけでは成立せず、§4.7 の専用 loop が
Phase 1 の前提実装になった（DONE）。

### 5.4 実装上の注意

- **`_is_lora_target` と `is_lora_wrappable_linear` の整合。** `sensenova_lora.py:162` の
  述語は `LoRALinearLayer` を含む（再適用・stacking 用）が、`base_adapter.py:45` の
  共有述語は意図的に含まない（二重 wrap 防止）。adapter 側は後者の意味論を使う。
  `Int8Linear` は `nn.Linear` のサブクラスではないため、素朴な
  `isinstance(m, nn.Linear)` は 294 件すべてを黙って取りこぼす。これは既に 4 つの
  arch で踏まれた罠として記録されている。
- **attention backend の mode。** vendored forward は `_attn_backend` / `_attn_mode` を
  読み、既定は `AttentionMode.INFERENCE`。学習では backward 可能な mode を
  stamp する必要がある（`sensenova_pipeline_ops.set_attention_backend` が stamp 器）。
  `forward_und` は eager 固定で `_flash_or_sdpa` に到達しないため、影響を受けるのは
  gen 側のみ。
  - **stamp を戻す責任は呼び出し側にある。** stamp するのは load 時の
    `ops/sensenova_ops.setup_attention_backend` 1 箇所だけで、以後 TRAINING を
    貼り直す場所は存在しない。学習中 sample は生成のあいだ
    `AttentionMode.INFERENCE` を明示的に stamp するため、`finally` で TRAINING を
    貼り直さないと残り全 step が INFERENCE のまま静かに走る。mode を明示するのは
    `set_attention_backend` が省略時に `torch.is_grad_enabled()` から推論するため。
- **style transfer の tripwire。** `forward_und` は style context が armed だと raise する
  (`modeling_qwen3.py:515-525`)。学習経路では style context を必ず未設定にする。
- **noise scale の一般形。** config の現在値だけを展開して式を再実装せず、
  `compute_noise_scale()` を再利用する。`noise_scale` 基礎値の乗算、mode 分岐、
  `dynamic_sqrt`、max clamp をすべて推論と一致させる。
- **`detect_prediction_config`** は Phase 1 統合で `sensenova` を flow / velocity として
  登録し、退行テストで固定した。

---

## 6. Phase 2 — full-parameter fine-tune（guard DONE、本体 PENDING）

### 6.1 拒否ガード（DONE）

**判断: 既知の非対応を共通 preflight でモデルロード前に拒否する。**

配布されている base は int8 のみで、588 個すべての decoder Linear が `Int8Linear` で
ある。`reject_quantized_base()` はここで正しく発火し、その発火は迂回すべきバグでは
なく、まさにこのガードが防ぐために存在する故障モードである — 量子化 Linear は
weight を buffer で持つので `requires_grad_(True)` が no-op になり、
`named_parameters()` にも現れない。結果として full FT は
**何も学習していないのに loss は正常に下がる**。SenseNova の場合、量子化が
スキップした層しか動かないどころか、decoder は文字通り 1 パラメータも動かない。

初期計画では次の 2 層を想定した。

- `SenseNovaFullParameterAdapter` を `prepare_models_for_training` /
  `setup_trainable_parameters` / `save_checkpoint` すべてが `NotImplementedError` を
  投げる形で置く（`ideogram4_adapter.py:131-150` がそのままテンプレート）。
- `arch_capabilities._add_training_unsupported("sensenova", "full_finetune", ...)` を
  追加し、`FullParameterTrainer._refuse_unsupported_full_finetune` が
  **モデルをロードする前に**拒否できるようにする（17.6 GiB のロードを払ってから
  既知の拒否に到達しないため）。

実装は後者の共通 preflight だけで fail-closed になるため、前者の専用 adapter は追加しなかった。
bf16 base を前提とする本体設計は Phase 2b に残る。

**【`601d0271` → U-2-2 step 3】現状はこの初期計画とは別の形になっている。** 専用
adapter は**追加され**、`save_checkpoint` も `22b22f09` で実装されたので
`NotImplementedError` を投げるメソッドはもう無い。**そして受付の拒否も
落ちた**（U-2-2 step 3）— `TRAINING_UNSUPPORTED["sensenova"]["full_finetune"]`
は削除され、`_refuse_unsupported_full_finetune` はこの arch を素通しする。
**残る `reject_quantized_base()` 相当の防御は消えていない**: full FT の
`load_components` は plain int8 base だけを受理し（`cc296e84`）、
materialize されていない Linear が 1 つでも残っていれば adapter が raise する。
本節が防いでいる「静かな 0 件学習」は、拒否ではなく**実体化 + 検証**として
処理されるようになった。

### 6.2 確定した設計判断 — 対象は gen branch のみ（fable 諮問）

**判断: 実装する場合の full FT の既定は gen branch のみの 8.1B。これを SenseNova に
おける「full fine-tune」と呼ぶ。**（**【2026-08-24 改訂】** 旧文は
「both-branch 16.2B はロードマップに載せない」だったが、その根拠だったメモリ算術が
誤りだったので撤回した。下記「改訂」節と §13.4 を参照。）

根拠:

- **座りが悪いのは gen-only ではなく both-branch の方である。** 言語理解を担う branch
  を含む 16.2B を学習するのは「text encoder も一緒に fine-tune する」ことであり、
  本リポジトリはどの arch でもそれを既定にしていない。破滅的忘却の profile が悪い。
  **既定を gen-only にする根拠としてはこれで足りる。**
- **gen-only の算術は厳しいが現実的:** bf16 weight 16.2 GB + gradient 16.2 GB +
  optimizer state（CPU offload）+ stochastic-rounding の per-step scratch + pixel space の
  activation（checkpointing 下）。48 GB で閉じるには gen 半分の offload と
  optimizer state の CPU 常駐が必要になる。つまり **Phase 2 は §8 の
  offload 機構の存在に依存する**。この依存順序自体がガード先行を正当化する。

#### 【2026-08-24 改訂】「メモリで即死」という根拠は撤回する

この節は以前 both-branch を **「bf16 weight 32.4 GB + gradient 32.4 GB だけで 48 GB を
超えるので原理とメモリの両方で落ちる」** として設計対象外にしていた。**この算術は
本リポジトリが既に持つ機構を一つも勘定に入れていない誤りだったので撤回する。**

- `_setup_fused_backward_pass` は per-parameter の
  `register_post_accumulate_grad_hook` で `optimizer.step_param()` を呼び、直後に
  **`tensor.grad = None` する**（[`base_trainer.py:3802-3812`](../../backend/core/training/base_trainer.py)）。
  勾配は 1 パラメータ分ずつ生まれて即消えるので、**32.4 GB の gradient は同時に存在
  しない**。
- Adafactor は 2nd moment を factored に持ち、`patch_adafactor_fused`（`:3763-3764`）
  で fused backward 経路に乗る。

改訂後の予算を下表に置く。**これは構造上の見積もりであって実測ではない**
（48 GB、B1、GC ON、§6.4 経路 (a) を 588 Linear に拡張した場合）。実測値は
prefix KV の 50.5 MiB だけで、これは Phase 0 の計測（§11）からの引用である。

| 項 | 素朴値 | 効く機構 | 適用後（構造上の見積もり） |
|---|---:|---|---:|
| weights bf16 両 half | 32.4 GB | なし | 32.4 GB |
| gradients | 32.4 GB | fused backward（`grad = None`） | ~0.1-0.2 GB |
| optimizer state | AdamW fp32 129.6 GB / adamw8bit 32.4 GB | **Adafactor（factored 2nd moment）**。Ring Buffer 系を代替として併記していたが、出荷された allowlist は Adafactor のみ（§6.5 末尾の訂正） | ~0.1 GB オーダー |
| stochastic rounding scratch | — | 出荷済みの per-step scratch 方式（§6.3） | ~0.2-0.4 GB |
| prefix KV | — | — | **50.5 MiB（Phase 0 実測値の引用）** |
| activations | — | GC ON | ~0.34 GB @1024² + pixel 系一時領域 |

**合計 ~36-38 GB（見積もり）で 48 GB に載る。** ただし成立条件が 6 つある
（旧文は「そのうち 1 は現状のブロッカーである」と続いていた。**その条件 1 は
`601d0271` で解消し、同時に条件 1 の書き方自体が誤りだったので本節末で訂正した**）。

1. ~~**fused backward の gate 解錠（最大のブロッカー）。**~~ **【解決、`601d0271`。
   ただし「解錠」という言葉自体が誤りだったので下記に訂正する】**
   旧文は「`_setup_fused_backward_pass` の呼び出しが `if self.blocks_to_swap > 0:` の
   内側にあり、SenseNova は `blocks_to_swap != 0` を拒否するので fused backward に
   到達できない」と書き、**あたかも Block Swap を有効化することが前提であるかのように
   読める形になっていた。これは機構の取り違えである。**
   本節末の「訂正」を参照。
2. **勾配蓄積の意味論が変わる。** hook は backward ごとに step するので、
   **fused backward 下では effective batch = 物理 batch = 1** になる。
   B1 + gradient accumulation で effective batch を作る現行 SenseNova 契約
   （§11 Phase 1）と衝突する。
   **【`0d843213` で実測確定】これは「accumulation-aware hook が未実装」という
   実装上の欠落ではなく、原理的な非両立である**（詳細と実測値は §13.4 U-2-2）。
   現状は拒否ではなく**警告**として出荷されている。
3. **optimizer は per-parameter seam と state 容量の両方を満たすものに限られる。**
   `torch.optim.AdamW` は fp32 state 129.6 GB かつ per-parameter seam が
   無いので stochastic rounding もかけられず（§6.3）、素の `adamw8bit` は state
   32.4 GB で超過する。**【`601d0271` で確定】出荷された allowlist は
   `("adafactor",)` のみ**である（`SENSENOVA_FULL_FINETUNE_OPTIMIZERS`）。
   Ring Buffer 系を併記していた旧文は誤りだったので撤回した — 根拠は §6.5 末尾の
   「訂正」。
4. **`use_ema` は fused backward と併用拒否**（実装済みの raise、`:3642-3652`）。
   **`601d0271` はこれを SenseNova full FT の契約でも config / 属性の両 channel で
   拒否する**（`assert_full_finetune_contract`）— hook 経路は `optimizer.step()` の
   呼び出し地点を通らないので、EMA shadow が静かに更新されないままになる。
5. **経路 (a) の 588 版は per-Linear に「dequant → int8 解放」の順**で行う。一括だと
   ロード時に int8 15.1 GiB と bf16 32.4 GB が同時実体化する。
6. **解像度上限は実測 gate で引く。** pixel-space の activation が唯一の解像度比例項
   である。

したがって **both-branch full FT は「メモリで不可能」ではなく「未実装の前提が
6 つある」**。ロードマップ上の既定は依然 gen-only（忘却 profile が根拠）だが、
**both-branch を構造的に不可能として閉じることはしない**。実装経路は §13.4（Phase U-2）。

**Phase 1 の実装が既定を裏付けている。** `encode_prompt` は `requires_grad=True` を
即 raise し（`ops/sensenova_ops.py:179-180`）、prefix の immutability は forward の
たびに `_assert_immutable_prefix_cache` で検証される（`:531-558`）。**ただしこれは
「解体しなければ und は学習できない」という意味ではない** — この assert は構造検証と
`requires_grad` 拒否が 1 つの関数に同居しているだけで、**分離すれば済む**（§13.3）。
§5.2 根拠 1 の改訂と合わせて読むこと。

#### 【2026-08-25 訂正】fused backward は Block Swap に依存していない（条件 1）

上の条件 1 の旧文は **「fused backward の gate 解錠」**という言い方をしており、
**Block Swap を有効化することが前提であるかのように読めた。機構の取り違えである。**

`_setup_fused_backward_pass`（[`base_trainer.py:3936-4048`](../../backend/core/training/base_trainer.py)）が
やることは (1) `register_post_accumulate_grad_hook` の登録、(2) optimizer への
`step_param` のパッチ、(3) hook 内での `step_param` → `tensor.grad = None` だけである。
**Block Swap が用意する状態を 1 つも読まない** — `blocks_to_swap` も offloader も
参照しない。`if self.blocks_to_swap > 0:` の内側にあったのは**たまたま**であり、
**Block Swap が「これを必要とする理由」を持つ唯一の機構だった**からにすぎない。

したがって必要だったのは gate の解錠ではなく **decoupling** であり、`601d0271` は
`blocks_to_swap == 0` のままこのルートに fused backward を設置した
（`base_trainer.py:3806-3813`。条件は `is_sensenova` かつ full FT かつ
`num_optimizer_groups == 0` かつ optimizer 名が `FUSED_BACKWARD_OPTIMIZERS` に
あること）。設置に失敗した場合は静かに非 fused へ落ちるのではなく **raise する**
（`:3815-3833`）— このルートは 15.1 GiB 分の勾配が常駐しない前提で予算を組んでおり、
非 fused は劣化モードではなく予算違反だからである。

**SenseNova の Block Swap 拒否は 5 箇所である**（いずれも `blocks_to_swap != 0` を
値として拒否する）。「3 箇所」と書いている記述があれば古い
（`sensenova_full_finetune_adapter_test.py:552` の docstring がそう書いている）。

| 位置 | 経路 |
|---|---|
| `base_trainer.py:1392-1393` | 通常の load（`_load_model_components`） |
| `base_trainer.py:1994-1995` | `_load_checkpoint_as_base` |
| `base_trainer.py:8387-8388` | `train()` の SenseNova 契約 |
| `ops/sensenova_ops.py:353-354` | `load_components` |
| `train_runner.py:175-176` | `_apply_sensenova_training_contract`（run 前） |

種類の違う 6 番目として `arch/sensenova.py:19-20` の `setup_block_swap` が
`NotImplementedError` を投げる。**`base_trainer.py:1333` はこの列に入らない** —
resume 時の checkpoint ロード失敗のエラー処理であって Block Swap とは無関係である。

#### 【`601d0271` で決定】decoder 外の gen 側モジュールは含めない

**直後の節「未決定のサブ論点」の推奨（含める方向）を上書きする決定が実装で下された。**
`SenseNovaFullParameterAdapter` は `fm_head` / gen ViT / timestep・noise_scale embedder /
`*_norm_mot_gen` を **collect しない**。

根拠は品質判断ではなく**同一性**である: これらは量子化されていないので U-2-1 の
`materialize_int8_decoder_linears` が materialize せず、collect すると
**adapter の scope が loader の scope と食い違う**。この食い違いこそ adapter が
存在して防いでいるもの（loader と collector が別々の表を持ち、静かにずれる）である。
adapter と `load_components` は `resolve_full_finetune_branch` と
`iter_sensenova_lora_targets` を共有しているので、**列挙は 1 つしか存在しない**。

**未解決のまま残す問い（新規登録）: x0 を直接出力する `fm_head` を凍結したまま
「full fine-tune」と呼べるのか。** これは本決定が答えていない設計上の問いであり、
ここでは解決しない。答えるとすれば「materialize の scope を量子化の有無から切り離す」
という別の実装が要り、それは同一性を壊さない形で行わなければならない。

#### 未決定のサブ論点 — decoder 外の gen 側モジュール

「gen branch = 8.1B」は **decoder Linear の数え方**である。gen 側には decoder の外にも
モジュールがあり、**これらを trainable に含めるかは未決定**である。

- `fm_modules['fm_head']`（`use_pixel_head` では `ConvDecoder`。x0 を直接出力する）
- `fm_modules['vision_model_mot_gen']`（gen 側 ViT）
- `fm_modules['timestep_embedder']` / `['noise_scale_embedder']`
- 各層の `*_norm_mot_gen`（`q_norm_mot_gen` / `k_norm_mot_gen` など）

`fm_modules` は decoder 層の外にある `nn.ModuleDict` である
（[`vendor/modeling_neo_chat.py:229-255`](../../backend/core/models/sensenova/vendor/modeling_neo_chat.py)）。
したがってこれらは 588 Linear の census にも Phase 1 の 294 LoRA target にも入らず、
量子化対象でもなく、half-eviction の対称性検証の対象でもない。**x0 を直接出力する
`fm_head` を凍結したまま「full fine-tune」と呼べるかは疑わしい**ため含める方向を
推奨するが、これは推奨であって決定ではない。Phase 2b の実装時に明示的に決めること。

> **【`601d0271`】この推奨は上書きされた。決定は「含めない」である。** 理由（loader と
> adapter の scope 同一性）と、**未解決のまま残した問い**（`fm_head` を凍結して
> full FT と呼べるか）は直前の「【`601d0271` で決定】」節にある。
> **この段落は推奨の記録として残す** — 決定を覆すなら、覆す側が同一性を
> どう保つかを示すこと。

### 6.3 master weight dtype 戦略

本リポジトリの既知の欠陥は「bf16 full FT は更新が bf16 の仮数に丸め込まれ、weight が
動かなくなる」というものである。ここで重要なのは、**本リポジトリが実際に出荷した
対策は fp32 master ではなく stochastic rounding であり、しかも永続 fp32 master は
明示的に棄却されている**という点である。

`BaseTrainer._attach_stochastic_rounding`
([`base_trainer.py:3364-3438`](../../backend/core/training/base_trainer.py)) の
docstring が機構をそのまま述べている:

> Full fine-tuning writes optimizer updates straight into BF16 storage with no
> FP32 master, so round-to-nearest deterministically discards every update below
> half a ULP and those weights never move again. Only the two ring-buffer
> optimizers implemented the repair; the shipped full-FT default is `adamw8bit`,
> so a user who changed nothing got the defect.

閉形式は `bf16_stochastic_rounding_test.py` が pin している:
**round-to-nearest 下で weight が動くのは `|w| <= 512 * lr` のときだけ。**
lr 1e-5 なら `|w| <= 5.12e-3` で、DiT の weight の大半が除外される。実測は 3 件あり、
Krea 2 の実 checkpoint で **8.7% しか動かず intended drift の 4.9% しか実現しない**、
bitsandbytes AdamW8bit の実 CUDA kernel 経由で **8.3% / 6.2%**、合成 400 step で
**9.2% / 6.9%**。つまり凍結率は 91% 前後である（「~89%」は概ね正しい）。

対策は `optimizer_stochastic_rounding`（`param_defaults.py:2208`、**既定 False**）で、
per-parameter の更新呼び出しの間だけパラメータと勾配を fp32 image に差し替え、
結果を確率的に bf16 へ丸め戻す。per-parameter の seam を持たない optimizer
（`torch.optim.AdamW` = `optimizer: adamw`）は**カバーできず**、名指しで警告される。

**永続 fp32 master は棄却済みである。** commit `8547f93c` の理由をそのまま引く:
学習要素あたり 4 byte を要し（12.8B の full FT で 51.2 GB、まさに欠陥が問題になる
場面で OOM する）、optimizer checkpoint に毎回直列化され、resume-safe でもない。
代わりに出荷されたのは per-step の scratch buffer（`4 bytes × 最大単一パラメータの
要素数 × slot 数`、optimizer オブジェクトごと）である。

その代わりの**正直な但し書き**も記録されている: scratch 方式は算術的に等価ではなく、
step 間の sub-ULP 蓄積を捨てて不偏性に頼るため、**およそ 1k step 未満では誤差が
信号と同程度**になり、そこから先で round-to-nearest を上回る。

SenseNova への含意:

- SenseNova は 8.1B の bf16 full FT なので**この欠陥をそのまま継承する**。
  アーキテクチャ的に免れる理由は何もない（MoT 二重化も pixel space も、
  仮数の分解能とは無関係）。実装すれば forced-bf16 arch の仲間入りをする。
- **`_attach_stochastic_rounding` は arch を見ないので、追加実装ゼロで適用対象になる。**
  gate は `self.optimizer_stochastic_rounding` と optimizer 名だけで、arch 分岐は
  1 つも無い（`base_trainer.py:3366-3372`）。SenseNova 側に必要なのは実装ではなく
  **契約（既定値と拒否）の決定**だけである。
- **推奨（fable 諮問、決定は Phase 2b 実装時）**: (1) `optimizer: adamw` を SenseNova
  full FT で**拒否する** — 警告で流すと「既定 False」と「非カバー optimizer」の二重経路で
  91% 凍結欠陥を再生産する。`torch.optim.AdamW` は per-parameter seam を持たない
  唯一の optimizer で、非カバー時は名指しで警告されるだけである
  （`base_trainer.py:3426-3431`）。(2) `optimizer_stochastic_rounding` を **SenseNova
  full FT の contract で既定 True に上書きする** — 全 arch 共通の既定
  （`param_defaults.py:2208`、False）は変えず、レガシー利用者のいない新規経路だけを
  正しい既定で開ける。永続 fp32 master は棄却済みのまま（下記）。
  - **【実測で裏付けられた、`5dce52ee`、2026-08-25】** stochastic rounding OFF では、
    測定した**全 optimizer** で **moved@1 == moved@20** — すなわち
    **要素の 84.5% が 20 step を通して一度も動かない**（「遅い」のではなく「凍結」）。
    累積 drift は学習率が要求する量の **18%** にとどまる。ON にすると drift は
    期待値に到達する。**`torch.optim.AdamW` だけは SR でも救えない**（per-parameter
    seam が無い）。合成パラメータと厳密既知の勾配による測定で、モデルは載せていない。
    上の推奨 (1)(2) はこれで実測の裏付けを得た。
- **【`601d0271` / `24220b5c`】推奨 (1) は実装され、(2) も着地した — ただし
  「既定 True」ではなく「ルート要件（強制）」として。**
  - (1) `optimizer: adamw` の拒否は `assert_full_finetune_contract` が
    **新規に実装した**。それまで存在した仕組みは
    `_attach_stochastic_rounding` の警告だけで、しかもそれは
    **`self.optimizer_stochastic_rounding` が既に True のときにしか到達しない**
    （`base_trainer.py:3494-3495` の early return）。既定は False なので
    **出荷時の既定設定では何も言わなかった**。これが「警告で流すと二重経路で
    欠陥を再生産する」の実際の姿である。
  - (2) **【`24220b5c`、U-2-3】`optimizer_stochastic_rounding` はこのルートで
    強制 ON になった。**`enforce_full_finetune_stochastic_rounding`
    （`ops/sensenova_ops.py`）が `setup_optimizer` の optimizer 構築**前**に適用し、
    どの arch がそうするかは `param_defaults.FULL_FINETUNE_FORCED_STOCHASTIC_ROUNDING_BY_ARCH`
    に per-arch 表として置かれている。フラグが立っただけで seam が包まれていない
    状態（ログだけ正しく、書き込みは round-to-nearest）を塞ぐため、
    `assert_full_finetune_stochastic_rounding_attached` が attach 後に
    `step_param` の被覆を**機構として**検証する。
    - **「既定 True」にできなかった理由は transport にある。** request model は
      このフィールドを `Optional[bool]` ではなく **`bool`** として宣言し
      （`routes.py:15145`）、`training_config.py:142` は**値が真のときだけ**
      YAML キーを書く。したがって**明示的な false とキーの省略は、trainer に
      届いた時点で同一**である。false を拒否条件にすると**全リクエストを拒否する**
      ことになる — frontend は既定を false でハードコードし
      （`TrainingConfig.tsx:126`）**常に送る**（`:786`）ので、UI が組める構成は
      すべて落ちる。false を尊重すれば凍結 run を出荷する。どちらも不可なので、
      **ルート要件（強制 + 通知）**を選んだ。通知は `training_log` チャンネルで
      ユーザーに届く（§13.4 警告ボックス (c)）。三値化すれば正直な既定が
      可能になる（未実装）。
    - **強制は full FT ルートだけである。** 同 arch の LoRA 学習は設定を尊重する。
  - **【実測、`24220b5c`】このルート自身の seam での測定。** bf16、
    `N=65536 ~ N(0, 0.02)`、定数勾配、lr 1e-5、Adafactor を
    `adafactor_fused.step_param` 経由で駆動（CPU、モデル無し）:
    **SR OFF は 20 step で 84.5% が一度も動かず、累積 drift は要求量の 18.3%。
    400 step でも凍結率は同じで、moved@1 == moved@400 == 15.46%**（遅いのではなく
    凍結）。**SR ON は 400 step で never-moved 0.0%、drift 100%。**
    独立の監査で seed 非依存・N 非依存も確認した（seed 12345 で 0.8452、
    N=262144 で 0.844）。
    - **但し書き（この 84.5% を scale 非依存と読まないこと）。** 上の
      `5dce52ee` の 84.5% / 18% は **optimizer も device も勾配も tensor サイズも
      違う**測定なので、一致は optimizer と device をまたいでいる。しかし
      **両者は同じ weight scale と同じ lr を固定している**（`PARAM_STD=0.02`、
      `lr=1e-5`）。凍結率を決めているのは実質この 2 定数であり、σ を振ると
      **σ=0.005 / 0.01 / 0.02 / 0.04 で 43.3% / 69.6% / 84.5% / 92.3%** と動く
      （同じ harness での実測）。
- **永続 fp32 master を SenseNova のために復活させる提案はしない。** リポジトリ全体で
  既に棄却された選択肢であり、gen branch 8.1B でも 32.4 GB になる。再提案するなら
  棄却理由（直列化・resume 非安全・OOM）を覆す新しい論拠が要る。
- 短 horizon の誤差特性は SenseNova でも同じなので、**数百 step の短い full FT を
  stochastic rounding で評価してはいけない**。これは SenseNova 固有ではないが、
  評価計画を立てるときに踏みやすい。

### 6.3.1 「int8-resident full FT は棄却済み」という前提は本リポジトリには無い

調査の結果を正確に記録しておく。**本リポジトリには「int8 に常駐したまま full FT
する」（optimizer が int8 code を直接更新する）案の調査・pre-registered gate・
棄却記録は存在しない。** stochastic rounding のノイズ床や error-feedback の state
サイズを論じた文書も無い。`INT8_W8A8_TRAINING_GATE.md` はそれを
**scope 外と明記して除外している**だけである（`:109-116`、"QAT and full fine-tuning
of a quantized base — out of scope"）。

実在するのは次の 2 つで、どちらも別物である。

1. **`reject_quantized_base()` による構造的拒否**（§6.1）。理由は数値的ではなく
   構造的 — weight が buffer なので勾配が付かず、静かに部分学習になる。
   実測: Anima で **weight 要素の 80.6% が凍結**、Krea 2 で
   **12,821.9 M 中 1.4 M しか学習されない**。
2. **gate G3 / G4 の失敗**。G3 は「学習用に勾配対応 INT8 W8A8 forward を作るか」で、
   criterion 2（どのワークロードも 3% 以上退行しない）に違反して失敗
   （256px で -4.53%、512px で -1.75%。DiT forward が compute-bound ではなく
   launch-bound で、activation 量子化 kernel が節約した GEMM 時間を上回る）。
   G4 は「dequant 経路が backward 用に保持する量を減らせるか」で、bitwise・勾配・
   メモリの全基準を通過したうえで**事前登録した step-time 上限 (+12%) に落ちた**。

したがって本文書は「int8-resident full FT は棄却済み」とは書かない。正しくは
**未調査であり、やるなら新しい pre-registered gate と新しい論拠が要る**。
本設計はそれを提案しない — §6.2 の bf16 gen branch 抽出の方が素直だからである。

### 6.4 bf16 base の供給経路（「入手」ではなく「実装」が律速）

**この節の見出しは以前「bf16 base の入手経路」だった。Phase 1 の出荷によって前提条件の
重心が入手から実装へ移ったため改題した。** gate
`_assert_supported_quantized_training_base`（[`ops/sensenova_ops.py`](../../backend/core/training/ops/sensenova_ops.py)、
`cc296e84` 時点で `:134-189`）は **未量子化 bf16 base を依然として拒否する** — full FT
側の例外メッセージも "an unquantized bf16 base is refused because none exists for this
repo" と書いている。したがって upstream から bf16 を入手しても現行 loader 経路では
ロードできず、**gate と loader の method-aware 化という実装作業が、入手とは独立に
必要**であった。

#### 【2026-08-25 改訂】「`load_components` は training method を見ない」は撤回する

旧文はここに **「`load_components`（`:155-156`）は training method を一切見ずに
無条件でこれを呼ぶ」** と書いていた。**`cc296e84` がその配線を実装したので撤回する**
（旧文が指していた `:49-91` / `:155-156` も同 commit で移動したため、この節の行番号は
すべて更新した）。現行の `load_components`（`cc296e84` 時点で `:244-296`）は

- `resolve_training_method(trainer)` で方式を解決し（`:256`）、
- **17.6 GiB のロードより前に** branch を解決して矛盾した switch 対を拒否し（`:259-263`）、
- gate に `training_method=` を渡す（`:267-269`）。gate は full FT のときだけ
  plain int8 を要求し、ConvRot base と bf16 base を名指しで拒否する（`:165-179`）。

`training_method` は train config section に**誰も書き込まない**ため（後述の
「方式の伝達経路」）、`resolve_training_method` は **trainer subclass 名**
（`FullParameterTrainer` を MRO 上で名前照合）を第一の channel とし、
`config["training_method"]` を第二の channel として併用する。

供給経路は 3 つある。

- **(a) ロード時の dequant materialization（外部依存なし）。** 配布 int8 checkpoint を
  ロードし、gen half の 294 Linear だけを `weight.to(bf16) * scale` で bf16 に
  materialize する。und half は int8 のまま凍結でよい — **凍結 int8 の und half の下で
  gen 側に勾配が通ること自体は Phase 0 が実 checkpoint で確認済み**である
  （§11 Phase 0）。ただし**そこで gen 側だったのは LoRA であり、bf16 に materialize した
  294 Linear そのものを学習する形は未試験**である（構造的には同型だが実測は無い）。
  int8 量子化誤差を学習の初期値に焼き付ける。
  - **plain int8 checkpoint に限定することを推奨する。** ConvRot base を dequant 元に
    すると rotation の逆適用という新規の複雑性が入り、しかも §5.3 の train/inference
    skew と重なる。→ **推奨のまま実装された**（下記）。
- **(b) upstream の 46.8 GiB bf16 ソースから gen half を抽出した artifact。** ~16.2 GB。
  §6.2 の推奨形。
- **(c) upstream bf16 をそのまま両 half bf16 でロードする。** 学習対象は gen half のみでも、
  und half が bf16 で常駐するぶん VRAM 要求が上がる。

#### 経路 (a) の 588 版は着地した（U-2-1 / `cc296e84`）

`materialize_int8_decoder_linears`（[`sensenova/loader.py`](../../backend/core/models/sensenova/loader.py)
`:370-`、`SENSENOVA_BRANCH_LINEAR_COUNTS = {gen: 294, und: 294, both: 588}` が `:367`）が
選択された half の `Int8Linear` を `nn.Linear` + 実 `nn.Parameter` に置き換える。dequant の
式は `Int8Linear._dequant_forward` と**同じ綴り**（int8 codes × dtype に落とした
`weight_scale` を out 次元で broadcast）なので、
materialize 後の base はその dtype で凍結 int8 base と同じ関数を計算する。branch は
新規 config key ではなく **`train_unet` = gen half / `train_text_encoder` = und half**
という Phase 1 LoRA と同じ対応で決まり、両方 False は既定ではなく拒否である。

**実装したもの**: 上記 materializer、method-aware な gate と `load_components`、
ConvRot base / mixed base / off-count tree / 非 `(out_features,)` scale の拒否、
および `backend/tests/sensenova_int8_materialize_test.py`（算術の `torch.equal`、
層まるごとの bitwise forward parity、閉形式誤差上界の両側 assert、
「int8 weight は buffer なので `parameters()` に出ない」負の対照、
LoRA 経路が bit 単位で不変であることの証明）。

**意図的に実装していないもの**: `SenseNovaFullParameterAdapter`（U-2-2）、
fused backward の Block Swap からの decoupling（同）、**学習成果物の checkpoint
format の決定（下節は開いたままにしてある。U-2-1 はどれも選んでいない）**、
offload との合成（U-2-4）、stochastic rounding の契約（U-2-3）。
**このうち前 2 つは `601d0271`、U-2-3 は `24220b5c` で着地した**（§13.4 U-2-2）。

~~**そして経路はまだ端から端まで到達しない。**~~ **【U-2-2 step 3 で解消】**
`TRAINING_UNSUPPORTED["sensenova"]["full_finetune"]` と `train_runner` の
`network.type != "lora"` 拒否は**両方とも落ちた**。経路は実 checkpoint 上の
run で端から端まで通っている（§13.4「U-2-2 実測」）。**この節が
「テストで証明されているだけ」と書いていた状態はもう当てはまらない。**

##### メモリ算術は実測である（実 checkpoint の safetensors ヘッダ）

**以下は見積もりではない。** 配布されている plain int8 checkpoint の safetensors
ヘッダ（先頭の長さ + JSON のみ。テンソルは 1 つも実体化していない）を読み、
`iter_sensenova_lora_targets` と同じ命名規則で half ごとに集計した値である。
**gen half と und half は完全に同値**だった。

| 量 | 値 | 出所 |
|---|---:|---|
| half あたりの int8 Linear 数 | 294 | ヘッダの key 集計（`iter_sensenova_lora_targets` の列挙と一致） |
| half あたりの weight 要素数 | 8,103,395,328 | 同上（dtype はすべて `I8`） |
| half あたり int8 | **7.546875 GiB** | 上記要素数 × 1 byte |
| half あたり bf16 | **15.09375 GiB** | 同 × 2 byte |
| 最大の単一 weight | **48.0 MiB (int8) / 96.0 MiB (bf16)** | ヘッダ中の最大 shape（50,331,648 要素） |

per-Linear で「dequant → 直前の int8 を解放」する順序にすると、**int8 base がロード
済みの状態からの追加ピークは `Q + q_max`**（`Q` = その scope の int8 総量、`q_max` =
最大の単一 int8 weight = 48 MiB）になる。列挙した module を全部生かしたまま一括で
materialize すると `2Q`（= bf16 の全複製）になる。

| scope | 追加ピーク: per-Linear 解放（実装） | 一括 |
|---|---:|---:|
| 片 half | **7.5938 GiB** | 15.0938 GiB |
| 両 half | **15.1406 GiB** | 30.1875 GiB |

**これは host RAM であって VRAM ではない。** materialize は transformer が device へ
移る前（`load_components` の `:271-276`、`.to(trainer.device)` は `:294`）に走るので、
GPU が見るのは結果だけである。順序そのものは weakref で assert してあり
（各 int8 module は「自分より後ろに生きている replacement が k 個」の状態で死ぬ。
遅延一括解放にすると毎回 294 と報告される）、コードを読んで確かめる形にはしていない。

##### 構造的に到達不能な und 5 個も materialize される

`und_gradient_unreachable_paths()` が名指しする 5 個（layer 41 の
`self_attn.q_proj` / `self_attn.o_proj` と `mlp.{gate,up,down}_proj`、§13.5）も
**例外なく materialize され、実 `nn.Parameter` を持つ**。列挙は 294 のままだからである
（§13.3 の「und は 289 ではなく 294 を維持する」と同じ理由）。t2i の loss はこの 5 個に
届かないので、**「294 個すべてが動いた」と主張する census はここで落ちるのが正しい**。
exit criterion 側の表現は §13.4 U-2-5 に書いた。

##### 方式の伝達経路（`training_method` は train config に書かれていない）

`resolve_training_method` が trainer subclass を第一の channel にしているのは、
**`training_method` を train config section に書き込むコードが存在しない**からである
（`train_runner` は `network.type` で dispatch して trainer クラスを構成する）。
`config["training_method"]` を第二の channel として併読しているので、後日この key が
配線されても静かに LoRA 扱いになることはない。**この欠落自体は SenseNova 固有ではなく
cross-architecture の欠陥**で、別 commit の担当である。ここに書くのは
**U-2-1 が subclass を channel に選んだ理由の説明としてのみ**であり、
修正の内容は本文書の管轄外とする。

#### 学習成果物の checkpoint format（未決定、新規の設計問題）

Phase 2b は LoRA と違い**モデル本体を出力する**ため、Phase 1 には存在しなかった
決定が要る。選択肢は 3 つで、いずれも推論側 loader との適合が問題になる。

1. **mixed（und int8 + gen bf16）** — 保存量は最小だが、**この形式の推論ロードは一度も
   試験されていない**。§12 に未測定事項として登録する。
2. **両 half bf16** — 推論側の量子化前提から外れる。
3. **gen half を再量子化して int8 に戻す** — 配布形式に一致するが lossy であり、
   学習で得た更新の一部を保存時に捨てる。

どれを既定にするかは Phase 2b 本体の実装時に決める。**現時点でどれかを推奨しない** —
1 の可否が未試験である以上、比較の前提が揃っていない。

> **【`601d0271`】この決定は critical path に乗った。** `SenseNovaFullParameterAdapter.save_checkpoint`
> は上記 3 択を名指しして `NotImplementedError` を raise する。一方 run は
> `save_every`（既定 100 step）ごとに `save_checkpoint` を呼び、その呼び出しを囲む
> except は **`(PermissionError, OSError)` だけ**である（`base_trainer.py:12104`）。
> `NotImplementedError` は捕まらない。
>
> したがって **format を決めないまま U-2-2 step 3（受付の解錠）を行うと、
> ロード前に無料で返っていた拒否が「数時間走って最初の `save_every` で落ちる run」に
> 変わる**。format 決定は step 3 の前提条件である。
>
> **ここで format を選ばない。** 3 択は意図的に開いたままで、選択は利用者の判断である。

##### 【U-2-2 step 3 前提 (b) 着地】3 択は API パラメータになった

`save_checkpoint` は実装され、**3 択は捨てずに設定値として出荷した**
（`sensenova_full_finetune_save_format`、既定 `"mixed"`。
`param_defaults` → `openapi.yaml` → `routes.py` → `training_config.py` →
`base_trainer` → `SenseNovaFullParameterAdapter`）。実体は
`loader.save_sensenova_full_finetune_checkpoint`（reader と同じファイルに置いた）。

- **branch は gen だけではない。** 3 形式はいずれも gen / und / both で意味を持つ。
  唯一の縮退は **both × mixed** — 残す int8 half が存在しないので `bf16` の
  ファイルそのものになる。**黙って改名せず**、metadata に実効 format を書き、
  `training_log` に warning を出す。無効な組み合わせは無い。
- **実サイズは実測表と一致する**（safetensors ヘッダから算術で導出。テンソルは
  1 つも実体化していない）: mixed(gen) 25.1167 GiB / both bf16 32.6575 GiB /
  int8 17.5759 GiB。
- **mixed と bf16 は「再学習の base」にはならない。**
  `_assert_supported_quantized_training_base` は 588 個すべてが単一の量子化
  flavour であることを要求するので、294 個が `nn.Linear` の mixed も、
  量子化ゼロの bf16 も拒否される。**int8 形式だけが再選択できる。**
  これは UI 文言にも書いた。
- **書き手が完全性を保証する。** 読み手は per-Linear 判定 + 3 カウント一致しか
  見ないので、**588 のどの部分集合でも「materialize 済み」として受理する** —
  half の途中で止まった save は無警告でロードでき、valid だが誤ったモデルになる。
  そこで (1) 書き込み前に live tree を数え、(2) 出力キーを数え、
  (3) shard を仮名で書いて index を最後に置く（`ShardWriter`）。3 つは互いを
  代替しない。
- **stale `weight_scale` の罠**（bf16 weight の横に scale を残すと
  `verify_quantized_swap` が **逆の欠陥**「partially scale-less」として拒否する）は、
  **live module tree から書く**ことで構造的に起こり得なくし、さらに書き込み側で
  loader と同じ連言を assert する。負の対照はテストに記録した。
- host RAM は**ストリーミングで 1 shard 分**。既定 4 GiB 閾値、実 checkpoint の
  最大テンソルが 1187.00 MiB なので peak は約 4 GiB + 一時 dequant 96 MiB。

##### 【U-2-2 step 3 で発覚】保存した config が読み戻せなかった（修正済み）

**smoke run の reload arm が捕まえた**。書けることと読めることは別で、
`22b22f09` は前者しか検証していなかった。

埋め込む geometry block（`sensenova_config` metadata。loader の**第一**の
情報源で、sibling `config.json` は fallback）は `config.to_dict()` から作って
いた。**`NEOChatConfig.to_dict()` は `NEOChatConfig(**·)` の不動点ではない** —
独立した理由が 2 つあり、**どちらも書き込み時には何も起きず、読み込み時にだけ
落ちる**。

1. **dtype**: vendor の `to_dict` は基底クラスの実装を**上書き**して
   `__dict__` をそのまま複製するので、基底の dtype 正規化
   （transformers 5 では `dict_dtype_to_str`）を通らない。top-level の
   `dtype` が `torch.dtype` のまま残り、`json.dumps(..., default=str)` が
   `"torch.bfloat16"` と書き、`PreTrainedConfig.__init__` が
   `getattr(torch, "torch.bfloat16")` で `AttributeError` を投げる。
   入れ子の `vision_config` / `llm_config` は素の `to_dict` を使うので
   **3 つのうち 1 つだけが壊れていた**。
2. **`downsample_ratio`**: `configuration_neo_vit.py:38` の
   `self.downsample_ratio = downsample_ratio,` — **末尾のカンマ**で値が 1-tuple に
   なる。ViT は `downsample_ratio[0]` で読むのでこれは load-bearing だが、
   直列化すると `[0.5]`、次の構築で `([0.5],)` になり、`[0]` が list になって
   `NEOChatModel.__init__` の `int(1 / ...)` が `TypeError` を投げる。
   **vendor 側は直さない**（推論が tuple に依存している）。

修正: **source の `config.json` をそのまま埋め込む**
（`_embeddable_sensenova_config`）。配布 checkpoint が積んでいる dict そのもので、
loader が毎回読んでいる形であり、`link_siblings` が同じファイルを checkpoint の
隣にコピーするので **埋め込みと fallback が構造的に一致する**。source が無い
場合だけ dtype を正規化した `to_dict()` に落ちる。加えて書き込み前に
`_assert_config_metadata_reloads` が **reader と同じ構築**をやり、
`NEOChatModel` の constructor が config 値に対して行う唯一の算術
（`1 / vision_config.downsample_ratio[0]`）まで実行する — 不動点でない block は
**25 GiB を書く前に拒否する**。負の対照はテストに入れてある。

Phase 2b 本体を実装して受付を開く際は、loader と利用者向け文書で採用した経路を
明示する。現行ガードは未提供の full FT を広告せず、未実装であることと LoRA 代替だけを
示す。

### 6.5 fused backward 下の optimizer 選択（Adafactor 一択ではない）

§6.2 の予算表は Adafactor を前提に書いてあるが、**それは唯一の選択肢ではない。**
本リポジトリには 8-bit state を持ったまま per-parameter fused-backward hook を自前で
登録する optimizer が 2 つある（`adamw8bit_ringbuffer` / `lion8bit_ringbuffer`）。
ここではその現状と、使える形にするために必要な作業を記録する。

#### 前提事実 1 — Ring Buffer の CPU state モードは、現状どの学習経路からも到達できない

**これは SenseNova 固有の事実ではなくリポジトリ全体の事実である。** 恒久的な置き場所は
`optimizers/RINGBUFFER_OPTIMIZERS.md` と当該 optimizer の docstring であり、ここに
書くのは **§6.2 の予算表がどの列で成立するかを決めてしまう**からである。
**発生源は `b81ac5f1` 以降に訂正済み**（`RINGBUFFER_OPTIMIZERS.md` 冒頭の注記、
`adamw8bit_ringbuffer.py` / `lion8bit_ringbuffer.py` の module docstring、
`vae/vae_config.py` の帰属、`base_trainer._ringbuffer_optimizer_kwargs` の docstring）
で、そちらから本節への参照も張ってある。以降の齟齬は発生源側を正とすること。

`AdamW8bit_RingBuffer` の module docstring は
"Optimizer states (exp_avg, exp_avg_sq) allocated on CPU via Ring Buffer" /
"VRAM savings: ~75%" を主張し、`RINGBUFFER_OPTIMIZERS.md` は
「**99.6% VRAM 削減**（optimizer states について）」と書いていた（**この 2 つの数値は
実測ではなく、同文書の 350M パラメータの仮定例の byte 数から計算した算術値である**:
`1 - 711/2800 = 74.6%`、`1 - 11/2800 = 99.6%`）。しかし
**`get_state_buffer` を optimizer に渡している呼び出し側が存在しない**。参照は
optimizer 実装の内部と `optimizer_factory.py:130`, `:174` の
`kwargs.get("get_state_buffer", None)` だけで、**誰も供給していない** — `BaseTrainer`
の `_ringbuffer_optimizer_kwargs()`（`base_trainer.py:3330-3347`。docstring が
「ユーザーが設定できる全オプションが optimizer に届くよう 1 箇所にまとめる」と
述べている場所）にも入っていない。

したがって常に `None` に解決され、**8-bit state は GPU に確保される**
（`adamw8bit_ringbuffer.py:290-307` の "Ring Buffer disabled: GPU allocation
(bitsandbytes-compatible)" 分岐）。**現状の Ring Buffer optimizer は「GPU state を持つ
fused 8-bit AdamW / Lion」として動いており、名前と文書が主張する CPU 常駐は効いて
いない。** `Lion8bit_RingBuffer` も同構造である（`lion8bit_ringbuffer.py:209-232`）。

> **付随して見つかった不正確な記述**: `vae/vae_config.py:44` は `get_state_buffer` を
> 「only BaseTrainer builds」と書いているが、BaseTrainer も渡していない。VAE 側の
> 記述は「この trainer は渡さない」という結論自体は正しく、帰属先だけが誤っている。

#### 前提事実 2 — fused backward との統合は既に排他で正しく配線されている

`base_trainer.py:3772-3784` に `adamw8bit_ringbuffer` 専用分岐があり、
`patch_adamw8bit_ringbuffer(...)` を呼んだあと **`return` して汎用 `step_param` hook
ループに落ちない**。したがって `step_param` は不要で、二重 step も構造上起きない。
`lion8bit_ringbuffer` も同型の分岐を持つ（`:3784-3793`）。

**CLAUDE.md の「Block Swap + Fused Optimizer Groups + 8bit optimizer は非互換」を
根拠に Ring Buffer 系を外すのは、原因の取り違えである。** 非互換の原因は
**Fused Optimizer Groups の batched `step()` が Block Swap で CPU に退避済みの
parameter に当たる**ことであり（`base_trainer.py:3611-3624`）、state の置き場所の話では
ない。しかも `:3683-3686` のエラーメッセージ自身が
**「(1) 'lion8bit_ringbuffer' か 'adamw8bit_ringbuffer' を使え」と解決策として名指し
している**。`num_optimizer_groups=0` + 自前 hook という Ring Buffer の経路は、その
非互換リストの対象外である。

#### per-parameter state の実測（`5dce52ee`。3 点フィット・残差 0 バイト）

**下の予算表は 16.2B への外挿なので構造上の見積もりだが、その単価は実測に置き換わった。**
モデルをロードせず、合成パラメータと厳密既知の勾配で、1 arm 1 process 測定。

| optimizer / 経路 | 実測 B/param |
|---|---:|
| `torch.optim.AdamW`（bf16 param） | 4.000000 |
| `adamw8bit`（bnb、`step()` 経路） | 2.031250 |
| `adamw8bit`（**fused / Block Swap 経路、`410fe689` 以降**） | **2.031250**（それ以前は **4.000000**） |
| `adamw8bit_ringbuffer`（GPU state） | 2.031250 |
| `lion8bit_ringbuffer`（GPU state） | 1.015625 |
| `adamw8bit_ringbuffer`（HOST state） | GPU **0.031250** / host 2.0 |
| `lion8bit_ringbuffer`（HOST state） | GPU **0.015625** / host 1.0 |
| `adafactor` | 0.002991（**shape 依存**） |

- **2.0 ではなく 2.031250** — 差は absmax である。§6.5 が別途 0.51 GB と書いている
  ぶんがここに含まれている。
- **`adafactor` の値は shape 依存なので、SenseNova の実 shape を測らずに 16.2B へ
  外挿してはならない。** 下の表の「~0.1 GB オーダー」は依然として概算である。
- **`adamw8bit` は Block Swap 下で 8-bit ではなかった**（実測 4.000000 B/param）。
  fused `step_param` が `zeros_like(p)` で moment を確保して手書き AdamW を回しており、
  **名前が約束する量の 2 倍**を、量子化なしで使っていた。`410fe689` が bitsandbytes
  自身の `init_state` / `prefetch_state` / `update_step` に委譲して解決した。
  副次的な利点が 2 つある: **state 形式が `step()` 経路と同一になったので、Block Swap
  の ON/OFF をまたぐ resume に変換アームが不要**であり、**CUDA 拡張のビルド要件も
  増えない**（bitsandbytes は kernel を同梱している）。
- **fused backward の即時解放も直接確認された**: backward 後に residency が残る勾配は
  24 本中 **0 本**（非 fused 経路は 24 本）。

#### 予算比較（16.2B への外挿。**単価は実測、合計は構造上の見積もり**）

both-branch full FT（両 half bf16 32.4 GB）を 48 GB カードで回す場合。

| 項 | Adafactor fused | AdamW8bit_RB **現状の配線**（state GPU） | AdamW8bit_RB **upgrade 後**（state CPU） |
|---|---:|---:|---:|
| weights bf16 | 32.4 GB | 32.4 GB | 32.4 GB |
| grads（fused で即消費） | ~0.1-0.2 GB | ~0.1-0.2 GB | ~0.1-0.2 GB |
| optimizer state (GPU) | ~0.1 GB | **32.4 GB + absmax 0.51 GB** | **absmax 0.51 GB** + 転送中 1 param 分 |
| その他 | ~1-2 GB | ~1-2 GB | ~1-2 GB |
| **GPU 合計** | **~36-38 GB（載る）** | **~68 GB（超過）** | **~37-39 GB（載る）** |

absmax は「param が CPU に移っても**常に GPU に置く**」と実装が明記している
（`adamw8bit_ringbuffer.py:309-310`）。

**ホスト RAM 側（upgrade 後）**: pinned state **32.4 GB**（Lion 版は `exp_avg` 1 本
だけなので 16.2 GB）。§8.3.2 の 4 相 eviction と併用すると evicted half の staging が
加わり、**pinned だけで ~50 GB 級**になる。これは gpu-probe の host-RAM 規律の対象で、
実行ホストの搭載 RAM 次第なので**実測 gate**とする（G-RB2）。

#### 容量問題は帯域問題に変換される

構造上、optimizer step あたり state H2D 32.4 GB + D2H 32.4 GB = **64.8 GB/step**
（Lion 版は半分）。fused backward では optimizer step = backward なので、**この転送は
backward の時間窓に発生する**。4 相 eviction と併用すれば weight 往復 32.4 GB/step が
同じバスに乗り、**PCIe 合計 ~97 GB/step 級**（いずれも構造値）。

> **【G-RB1 実測、`8c13c493`、2026-08-25】この節は以前「隠れず直列加算になる、が
> 実装から言える上限側の事実」と書いていた。実測はこれを覆した — 閾値の上では
> 完全に吸収される。** 以下が実測値である（RTX 6000 Ada、パラメータ数
> 100.66 M 固定、warmup 5、1 arm 1 process）。
>
> - **host/GPU の step wall 比は AdamW で 4096 tokens/step、Lion で 2048 から
>   1.00 に到達し、65536 まで 1.00 のまま**である。「1 に近い」ではなく、
>   **HOST arm の純 compute 超過分が GPU arm の超過分と一致する**（直列加算なら
>   4096 tokens で 47.95 ms のはずが、実測 33.41 ms）。低 token 側では比 4.17
>   （64 tokens、AdamW）まで開く。
> - **転送だけを切り出すと AdamW 15.21 ms / Lion 8.22 ms でちょうど半分**
>   （state 2 本 対 1 本）。26.5 / 24.5 GB/s は PCIe 4.0 x16 の線速で、
>   uint8 連続領域への zero-copy UVA アクセスが帯域効率を落としていないことを示す。
> - **閾値は転送量に厳密に比例し、閉形式で書ける**（実測と ~2% 一致）:
>
>   ```
>   N_tokens >= 2 * (state bytes/param) * achieved_FLOPS / (6 * PCIe bytes/s)
>   ```
>
>   実測定数 **80.7 TFLOP/s**（bf16）・**26.5 GB/s** で
>   **AdamW 2038 tokens / Lion 1019 tokens**。
> - HOST arm と GPU arm が同じ仕事をしていることは仮定ではなく検証済み — 同一 seed・
>   同一入力・10 step で moved fraction / mean drift / parameter checksum /
>   state 占有率が**最終桁まで一致**する。
>
> **以前引用していた「2.8〜11.5 倍」は warmup 無しの暫定値で、11.5 倍は再現しない。
> 確定値として引用しないこと。**
>
> **機構は未解明**: なぜ in-order stream がこれを吸収するのかは特定できていない。
> 壁時計が `sum` ではなく `max(compute, transfer)` の形になるという事実までが射程で、
> プロファイラは取っていない。**断定しないこと。**

**SenseNova は閾値の下側にいる（実測に基づく位置づけ）。** /32 token grid・batch 1
なので **1024² で 1024 image tokens** — AdamW の 2038 を下回り、Lion の 1019 と
ほぼ同じである。1536² で 2304、2048² で 4096。**つまり SenseNova の想定解像度帯では
転送はおおむね直列に乗る。**

**16.2B への投影（実測帯域からの投影であって実測ではない）**: AdamW の 64.8 GB 往復は
約 **2.45 s** を要し、compute 約 **1.64 s** に対して**隠れない**（+0.8 s/step）。
Lion の 32.4 GB は**隠れる**。ただし MoT が image token に対して gen 側 half しか
計算しないなら compute は約 **0.82 s** に落ち、**Lion も隠れなくなる**。
**この投影が乗っている SenseNova の実 step wall は、本リポジトリに実測値が存在しない**
（U-2-4 が測る場所である）。

なお **hook の発火順 = 勾配確定順 = 逆 layer 順で決定的**なので、prefetch schedule は
導出可能である（閾値の下で効かせたい場合の余地）。

#### upgrade に必要な項目（4 → 3 → **残り 1**。U-2-6 で 2 項目が着地、機構の誤りも訂正）

1. ~~`BaseTrainer` が `get_state_buffer`（`RingBufferAllocator` ベース）を
   optimizer_kwargs に渡す配線（**現在ゼロ**）。~~ **【DONE、U-2-6】**
   `_ringbuffer_optimizer_kwargs()` が `optimizer_state_host_resident` のときに
   `HostOptimizerStateAllocator` を渡す。**ただし基底の指定が誤っていた** —
   `RingBufferAllocator` は**再利用されるアリーナへの view** を返す
   （`free_layer` / wrap-around 付き、層パラメータ 1 回分の寿命用）ので、
   run 全体を生きる optimizer state に使うと **2 つのパラメータの moment が同じ
   バイトを共有して静かに壊れる**。**永続・非再利用・パラメータ単位**の専用
   allocator が要る。また **pinned 済みバッファを返すこと**が必須で、返さないと
   optimizer 側の `pin_memory()` が第 2 の確保を作り **host RAM が 2 倍**になる
   （実測: 保持する版 2.04x 対 出荷版 1.04x）。
   **switch は config channel 限定**（run の train_config = YAML の
   `optimizer_state_host_resident` キー。`BaseTrainer.__init__` が読む）。
   **API / UI 面は意図的に張らない** — 対象は Ring Buffer 系 2 つだけで、
   本ルートの allowlist は `("adafactor",)` なので、選べる場所では効かず、
   効く場所では未実測になる。理由は `BaseTrainer.__init__` のコメントと §13.4 U-2-5。
2. ~~**hook 経路への state H2D → 更新 → D2H の移植。**~~ **【撤回、`5dce52ee`】
   不要である。** ただし **【U-2-6 で機構を訂正】理由が誤っていた。**
   ここには「host buffer は pinned なので kernel が UVA 経由で直接アドレスでき、
   ステージングされた転送が無いことがこの経路の動作原理」と書いてあったが、
   **転送はステージングされている。** 素通しなのは Python 層だけで、実際には
   C++ 拡張が行う（`cuda/adamw8bit_cuda.cpp:145-243`、Lion は
   `cuda/lion8bit_cuda.cpp:170-250`）: **device ごとにキャッシュされた専用
   transfer stream** 上での H2D、**CUDA event** による update kernel の順序付け、
   同じ stream 上での D2H writeback。移植が不要なのは「転送が無いから」ではなく
   **「すでに一段下の層に実装されているから」**である。
   G-RB1 の 26.5 GB/s（PCIe 4.0 x16 線速）はこのバルク DMA が出す値で、
   per-element の UVA read が出す値ではない。**`git log 8c13c493..327614d6` が
   空**なので、この転送機構は G-RB1 実測コミットの祖先であり、
   **G-RB1 はステージング経路を測っていた。**
3. ~~専用 transfer stream~~ + **次パラメータ state の prefetch** + ~~event 同期~~。
   **【U-2-6】stream と event 同期は上記のとおり実装済み。残るのは prefetch だけ。**
   狙いは明確で、**H2D は完全に露出している** — hook 発火時に発行され、その直後に
   それを待つ kernel が走るので何とも重ならない（D2H は既に後続の backward と
   重なっている）。hook 発火順は決定的なので schedule は導出可能。
   **閾値の下**（SenseNova の想定解像度帯）でのみ意味を持つ。**未実装。**
4. ~~**`if not param.is_cuda: return` のサイレントスキップを fail-loud にする**
   （`:1082`）~~ **【DONE】hook 側は `3a7c9560` で既に済んでいた**
   （本項が引く `:1082` の記述は当時から stale）。**U-2-6 が見つけた残りは
   `step()` 側**の同型のスキップ（`adamw8bit_ringbuffer.py:751` /
   `lion8bit_ringbuffer.py:525`、同じ嘘のコメント付き）で、こちらを fail-loud
   にした。8-bit は CUDA kernel なので raise、FP32 経路は CPU テンソルで走る
   ので**スキップをやめて実行**する。
   **step ごとの updated-param census == trainable 数**（G-RB3）は実装済み。

#### トレードオフ（「一択」と断定しない）

| 軸 | Adafactor fused | AdamW8bit_RB（upgrade 後） |
|---|---|---|
| GPU state | ~0.1 GB（factored） | absmax 0.51 GB のみ |
| ホスト RAM | 追加なし | pinned 32.4 GB（Lion 16.2 GB） |
| PCIe/step | 追加なし | 64.8 GB（Lion 32.4 GB）。**閾値の上では実測で完全に吸収される**（G-RB1）。SenseNova の想定解像度帯は閾値の**下**なのでおおむね直列 |
| 状態の情報量 | factored 2nd moment（per-element の 1st/2nd moment を持たない） | AdamW の full moment 構造を保つが **8-bit blockwise 量子化誤差**を持つ |
| stochastic rounding | 外付け wrapper 経由（§6.3） | **native 内蔵**（`_NATIVE_STOCHASTIC_ROUNDING_OPTIMIZERS`、`base_trainer.py:3351`） |
| fused seam | 追加実装ゼロ | 内蔵 hook が**そのまま host state で動く**（実測。旧記述「state shuttle 未配線 = upgrade 必須」は撤回） |
| 追加前提 | gate 解錠のみ | gate 解錠 + upgrade **3 項目**（うち allocator 配線が必須、残り 2 は閾値下の最適化と fail-loud 化）+ **CUDA 拡張の JIT ビルド**（CUDA toolkit + ninja。`adamw8bit_cuda.py:191-199` が optimizer 生成時に `load()` する。**`adamw8bit` はこの要件を持たない** — bnb 同梱 kernel に委譲するため） |

**提示形**（**【`601d0271`】実装はこれを採らなかった。本節末の「訂正」を先に読むこと**）:
both-branch Full-FT の許可 optimizer は **Adafactor と Ring Buffer 系の
両方**とし、**初版の既定は Adafactor**。ただしその根拠は品質ではなく
**「追加実装ゼロ・外部前提ゼロで成立する」という事実のみ**である。
AdamW8bit_RB は upgrade 完了と gate 通過を条件に opt-in で開く。
**収束特性の優劣は主張しない** — 本リポジトリに実測が無く、収束実験も行わない方針で
ある（moment 構造の違いという事実の記録に留める）。`Lion8bit_RingBuffer` も同構造で
併記する（state 1 本なので容量・帯域が半分。ただし `schedule_free` は明示的に
非対応で raise する。`lion8bit_ringbuffer.py:98-104`）。

#### 実測 gate（G-RB1 **CLOSED** / G-RB2 / G-RB3）

opt-in を開く条件として事前登録した gate。実施は U-2-6（§13.4）の exit criteria。

| gate | 問い | 状態 |
|---|---|---|
| **G-RB1（帯域隠蔽）** | state の往復は backward に隠れるか、直列加算になるか | **CLOSED（`8c13c493`）**: 閾値の上では完全に吸収され、閾値は閉形式で書ける。SenseNova の想定解像度帯は閾値の下。詳細は上の実測ボックス。**モデルを載せずに「閾値」を測る形にしたのが要点** — 16.2B の step wall は測れないが、閾値は parameter 数に依存しないので合成で測れる |
| **G-RB2（ホスト RAM）** | pinned state が実行ホストに載るか | **CLOSED（U-2-6）**。下の実測ボックス参照。**問い自体は書かれたままでは検証不能**（「載るか」は実行ホスト依存で、16.2 B は確保できない）ので、**測れる内容に分解して消化した**: (1) host 常駐が実際に有効になること（flag ではなく state テンソルの device census で）、(2) **pinned bytes/param の実測**、(3) **peak working set ≒ pinned 実バイト**（＝二重確保が無いこと。ここが実際に落ちうる箇所だった）。(4) fit 判定はこれらの掛け算 |
| **G-RB3（correctness）** | サイレント CPU-skip が起きていないか | **CLOSED（U-2-6）**。`optimizers/update_census.py`。fused backward 実測で **32/32 が更新、negative control は 31/32 を名前付きで捕捉**。コストは **588 パラメータで 47.8 µs/step**（81 ns/param、device 仕事も同期も無し）。U-2-5 の「588 Linear update-nonzero census」に統合する |

> **【G-RB2 実測、U-2-6】RTX 6000 Ada、402,653,184 個の bf16 パラメータ、
> `set_per_process_memory_fraction(0.72)`、announce 付き、`1 case 1 process`。**
> **1 process で 8 case を回した最初の版は host の数字が使い物にならなかった** —
> working set は縮まないので、2 番目以降の case は 0.375 GiB の pinned を確保
> しながら RSS delta ほぼ 0 を報告する。**delta に現れるのはその process が
> 初めて行った確保だけである。**
>
> | optimizer | mode | GPU state B/param | host state B/param | pinned | pinned 実サイズ | peak wset | peak GPU |
> |---|---|---:|---:|---:|---:|---:|---:|
> | `adamw8bit_ringbuffer` | GPU（既定） | 2.031250 | 0 | – | 0 | 1.380 GiB | 4.027 GiB |
> | `adamw8bit_ringbuffer` | HOST | **0.031250** | 2.000000 | 100% | 0.750 GiB | 2.151 GiB | 3.277 GiB |
> | `lion8bit_ringbuffer` | GPU（既定） | 1.015625 | 0 | – | 0 | 1.379 GiB | 3.646 GiB |
> | `lion8bit_ringbuffer` | HOST | **0.015625** | 1.000000 | 100% | 0.375 GiB | 1.776 GiB | 3.271 GiB |
>
> - **単価は §6.5 の HOST 行を桁まで再現**し、GPU state は **98.5% 減**。
> - **二重確保は無い**: peak wset の HOST − GPU 差は AdamW **+0.771 GiB / 0.750 GiB
>   pinned = 1.028x**、Lion **+0.397 / 0.375 = 1.059x**。allocator が pinned 済み
>   バッファを返さないと **2.04x** になる（optimizer 側の `pin_memory()` が
>   第 2 の確保を作るため）。
> - HOST 側の peak GPU が GPU 側より **0.75 GiB 低い**（3.277 対 4.027）のは、
>   ちょうど AdamW の GPU state が消えた分である。
> - **16.2 B への 32.4 GB / ~50 GB は依然として構造上の外挿である。**
>   実測なのは**単価だけ**で、合計は掛け算にすぎない。
> - **announce は 2 回とも GPU 側を低く見積もった**（3.0 GiB と申告して実測 4.78 GiB。
>   weights + grads + GPU state に加えて stochastic rounding の scratch と
>   拡張のステージングバッファが乗る）。probe の announce は実測値に直した。

G-RB3 が最も重要である。upgrade 項目 4 のサイレントスキップは、**loss が正常に下がる
まま一部の half が更新されない**という §6.1 と同型の故障を作るので、
「速いか」ではなく「正しいか」を先に閉じる。

#### 【`601d0271` で訂正】出荷された allowlist は `("adafactor",)` のみ

**本節の「提示形」は「Adafactor と Ring Buffer 系の両方を許可し、既定は Adafactor」
だった。実装はそうしていない。** `SENSENOVA_FULL_FINETUNE_OPTIMIZERS`
（`ops/sensenova_ops.py`）は `("adafactor",)` である。Ring Buffer 系を
allowlist に入れる案は **2 つの独立した理由で誤っていた**ので、再発しないよう
理由を残す。

1. **§13.4 U-2-6 が G-RB2 / G-RB3 を exit gate として事前登録しており、
   どちらも開いている。** allowlist に入れることは gate を通さずに opt-in を
   開けることと同じで、本節自身の「これが通るまで Ring Buffer 系は opt-in を
   開かない」に反する。
2. **`adamw8bit` を排除している state 容量の議論は、`adamw8bit_ringbuffer` を
   同じだけ排除する。** 上の実測表が両者に **同一の 2.031250 B/param** を与えており
   （host state モードは `get_state_buffer` を供給する呼び出し側が存在しないので
   起動しない = 前提事実 1）、GPU 上での単価は文字通り等しい。
   `lion8bit_ringbuffer` だけは **1.015625 B/param** で半分だが、そちらは理由 1 が
   そのまま当たる。

**scope を合わせた算術**（U-2-1 が実 checkpoint ヘッダから取った half あたり
**8,103,395,328** 要素。§6.4 の表）:

| optimizer | B/param（実測） | gen half のみ（このルートの既定 branch） | 両 half |
|---|---:|---:|---:|
| `adamw8bit` / `adamw8bit_ringbuffer`（GPU state） | 2.031250 | **16.5 GB** | **32.9 GB** |
| `lion8bit_ringbuffer`（GPU state） | 1.015625 | 8.2 GB | 16.5 GB |

上の「予算比較」表は **both-branch 16.2B への外挿**なので、gen-only を既定とする
このルートの数字はここで別に置く。**単価は実測、合計は掛け算である。**

---

## 7. Phase 3 — reference 画像を含むデータセットの混在（DONE）

**実装完了（`7a09af52` 3-1/3-2、`d7bd9067` 3-3、`611a4a24` 3-4）。**
§7.1-§7.4 の設計判断は**すべてそのまま実装された** — per-item presence、B1 強制、
`separate_by_reference` の位置づけ（prefix shape の保証には使わない）、und 凍結。
実装の実際の姿と実測値は §7.5 / §11 Phase 3 に置く。

### 7.1 前例は存在する（新規設計ではない）

本リポジトリには reference 画像を使う**実際に学習される** conditioning 経路が
2 つ既にある。

- **FLUX.2**: reference を target と同じ bucket 寸法で VAE encode し、pack して
  position ID を `t_offset = 10 + 10*ref_idx` でずらし、noisy sequence に連結。
  出力を `original_seq_len` で切り戻して loss は target のみに掛ける
  (`ops/flux2_ops.py:470-566`)。
- **SD1.5 / SDXL**: SigLIP2 vision encoder の 257 token を text embedding 列に連結
  (`base_trainer.py:11157-11167`)。VE 自体も学習対象になり得る。

データ基盤も既にある。`Dataset.reference_suffixes` / `target_suffixes` /
`caption_suffixes_for_reference`（`database/models.py:499-504`）、
`DatasetItem.related_images` JSON の `"reference"` キー（`:589`）、
ファイル名 suffix 走査による自動投入（`utils/dataset_scanner.py:282-338`）、
学習 item への受け渡し（`train_runner.py:965-967` が
`processed_item["reference_images"]` を設定）。

**したがって Phase 3 は新規設計ではなく既存基盤の拡張である。** ただし SenseNova の
reference は FLUX.2 とは**まったく別の入口**から入る（noisy 画像列への連結ではなく、
understanding tower を通した ViT token を text prefix に差し込む）ため、
`ops/` 層の実装は流用できない。流用できるのは schema・scanner・item 受け渡し・
bucketing の 4 層である。

### 7.2 確定した設計判断 — 表現と batch 構成（fable 諮問）

**判断 1: per-item presence を真とする。既存の run-global
`use_reference_images` は「ref 経路が armed」であることだけを表す。**

DB と `train_runner.py` は既に per-item で `item["reference_images"]` を保持しており、
それが正しい粒度である。これにより ref 有り dataset と ref 無し dataset が、run 全体の
意味論を新しく発明することなく 1 つの run に共存できる。`use_reference_images` は
FLUX.2 でも既に経路の arm と homogeneous bucketing の有効化に使われており、SenseNova
も同じ意味論を再利用する。**新しい dataset-level parameter、schema、per-item stamp は
追加しない。** item 自身の `reference_images` の有無だけが各 batch の分類を決める。

**判断 2: 初版は物理 `batch_size=1` を強制する。
`separate_by_reference` は再利用するが、shape の正しさを委ねない。**

決定的な論拠は prefix 長の不揃いである。`separate_by_reference` の bucket key は
`(resolution, bool(has_reference))` だけで、caption token 長、reference 枚数、reference
token 数を揃えない。これは ref 無し Phase 1 でも caption 長が違えば起きる。
padding された prefix KV を denoise phase の `causal=False` flash attention に流すには、
連結 cache を跨ぐ padding mask / varlen attention が必要だが、現行実装には無い。

初版は物理 batch 1 とし、既存 gradient accumulation で effective batch を作る。
単純な per-sample forward loop は BaseTrainer が loss return 後に backward する現契約では
全 sample の graph を保持し、VRAM 利点が無いため採らない。batch > 1 は padding-aware
gen mask、varlen attention、または streaming per-sample backward が実装された時だけ開く。

`separate_by_reference` は sampler の整理と将来 batching の入口として残すが、物理
batch 1 では correctness 上は冗長であり、prefix shape の保証とは呼ばない。各 item の
reference は推論と同じ動的 preprocessing を使え、異なる item 間で固定 token 数に
揃える必要はない。

**判断 3: it2i 挙動の学習に understanding branch の解凍は、既定では不要とする。**

凍結された und tower と und branch は既に reference を豊かに符号化している
（完全な VLM である）。gen branch は phase 跨ぎの attention を通じて、その固定表現を
**利用する**ことを学習できる。これは凍結 TE + gen-only LoRA が他のすべての arch で
機能しているのと同じ構図であり、公開されている gen-only の distillation LoRA が
それを示唆している。したがって §5.2 の判断と矛盾しない。

**この点は本文書で最も不確実性が高い箇所である。** 凍結 und での reference 忠実度が
十分かどうかは、どちらの方向にも前例が無い経験的問題である。

### 7.3 忠実度が不足した場合の和解経路

reference 忠実度が実測で不足した場合にのみ、§5.2 で保留した `scope: both` を
**LoRA に限って**追加する（full FT には決して入れない）。und への LoRA なら忘却
リスクは有界であり、微分可能 prefix の実装コストは opt-in したユーザだけが払う。

**設計としては継ぎ目だけを用意し、und 学習の機構は先に作らない。**

### 7.4 データパイプライン上の注意

- reference は understanding tower 用に **ImageNet 正規化**、target は generation
  tower 用に **0.5/0.5 正規化**。同じ item の 2 枚の画像が違う前処理を要求する。
- reference は現状どの arch でも latent cache されず毎 epoch ディスクから読み直される
  (`base_trainer.py:10597`)。SenseNova では reference は ViT token になるので、
  キャッシュするなら token 側でキャッシュするのが自然。初版では実装しない。
- 推論側には `REFERENCE_IMAGE_MAX_PIXELS_CAP = 1024*1024` の encode コスト上限が
  ある。学習側も同じ上限と動的 preprocessing を再利用する。

### 7.5 実装差分（設計判断は §7.1-§7.4 のまま、配線の具体）— DONE

§7.1-§7.4 の設計判断はすべて維持された。以下は Phase 3 の実装で必要になった配線で、
**§9 の統合ポイント一覧に未記載だったもの**である。B1 強制はむしろ必然性を増した —
reference は各 ref の smart-resize で token 数が per-item に変わるため、prefix を
さらに ragged にする。

**差分 1: `use_reference_images` のゲート — 6 箇所中 4 箇所だけ解除した（`7a09af52`）。**
**残る 2 箇所は意図的に flux2 限定のまま**である。

| 箇所 | 内容 | 結果 |
|---|---|---|
| `train_runner.py` | sensenova で `ValueError`（Phase 3 deferral） | **解除**（型正規化だけ残す） |
| `base_trainer.py` train() | 同じ拒否の trainer 側の重複 | **解除** |
| `base_trainer.py` warn | 非 flux2 は warn して無視 | **解除**（`not (is_flux2 or is_sensenova)`） |
| `base_trainer.py` sampler | `separate_by_reference = ... and self.is_flux2` | **解除**（`(is_flux2 or is_sensenova)`） |
| `base_trainer.py` encode | reference latent の encode 分岐 | **flux2 のまま（意図的）** |
| `base_trainer.py` batch | reference latent の batch 引き回し | **flux2 のまま（意図的）** |

**後ろ 2 つを広げてはいけない。** そこは reference を **target の bucket 寸法で VAE
encode する latent 経路**であり、§7.5 差分 4 が「反面テンプレート」と呼んでいるもの
そのものである。SenseNova はそこに**加わることで**ではなく、**自前の入口を持つことで**
解放された（上流の `encode_caption` 分岐へ迂回し、reference は prompt prefix に入る）。
`sensenova_reference_training_test.py` が
`source.count("use_reference_images and self.is_flux2") == 2` を固定しているので、
後から「SenseNova を足し忘れている」と誤解して広げると**テストが落ちる**。これは
仕様であって不足ではない。

なお `separate_by_reference` は SenseNova でも有効化されたが、**位置づけは §7.2 の
まま「sampler の整理」**である（B1 では prefix が per-item なので、これが prefix
shape を保証することは無い）。**この経路は 3-4 の smoke では exercise されていない** —
bucketing 無効時は通らないためである。

**差分 2: `text_length` の意味論変更（最も踏みやすい罠）。** 現在の
`SenseNovaTrainingPrefix.text_length` は `int(input_ids.shape[1])`
（`ops/sensenova_ops.py:207`）で、これが `_build_t2i_image_indexes(token_h, token_w,
text_len, device)` の **t 軸の基点**として使われる（`:353`）。vendor 実装は
`t_image = torch.full((token_h*token_w,), text_len)` であり、text_len は
**「次の t index」であって token 数ではない**
（`vendor/modeling_neo_chat.py:502-507`）。text-only では
`indexes[0].max()+1 == indexes.shape[1] == input_ids.shape[1]` が偶然すべて一致するため
現在の実装は正しい。**reference があると image patch token が h/w 軸に展開されて
t 軸長 ≠ token 数になり、この一致が壊れる。** 推論の reference 経路は一般形の
`indexes_cond[0].max() + 1` を使う（`sensenova_pipeline_ops.py:534-535`、
text-only 経路の `indexes_cond.shape[1]`（`:463-464`）とは別式）。**放置すると位置ずれが
形状エラーなしに静かに起きる。**

**実装（`7a09af52`）**: `text_length = int(indexes[0].max()) + 1` に一般化し、
dataclass のコメントで「token 数ではなく次の t index」と明記した。両方向を pin して
ある — text-only は `input_ids.shape[1]` との一致、reference 有りは t extent との一致
**かつ token 数より真に小さいこと**。後者が要るのは、**前者だけでは退化ケース
（両者が偶然一致する形）を通してしまう**からである。実機での非退化確認は
§11 Phase 3 の実測に記録した。

**差分 3: 再利用する推論側関数（再実装ゼロ）。** cond branch のみでよい
（img_cond / uncond は CFG 用で loss には不要）。

- `_embed_reference_images`（`sensenova_pipeline_ops.py:239-274`）—
  `load_image_native` による ImageNet 正規化 + smart-resize + 1MP cap、
  understanding tower の `extract_feature(..., gen_model=False)` 経由。
- `transformer.img_context_token_id` の設定（`sensenova_pipeline_ops.py:482`）。
  **既定は `None`（`modeling_neo_chat.py:258`）で Phase 1 経路は一度も設定しない。**
  漏らすと `_build_it2i_inputs` の `assert selected.sum() != 0`
  （`modeling_neo_chat.py:672-673`）で落ちる。
- `_splice_reference_image_tokens`（`sensenova_pipeline_ops.py:277-`）、
  `transformer._build_it2i_inputs`（`modeling_neo_chat.py:658-`）、
  `transformer._it2i_prefix_forward`（`modeling_neo_chat.py:518-`）。

`train_step` 本体・専用 non-reentrant checkpoint loop・`update_cache=False` fallback は
**無変更で流用できる**（prefix が長くなるだけである）。

**実装（`7a09af52`）**: 上記をすべてそのまま使い、再実装はゼロ。`img_context_token_id`
は `_build_it2i_inputs` の直前に設定する。`SENSENOVA_MAX_REFERENCE_IMAGES`（推論側の
上限）を超えたら raise し、`REFERENCE_IMAGE_MAX_PIXELS_CAP` は
`_embed_reference_images` の既定引数として自動的に効く。予告どおり `train_step`・
immutability assert・checkpoint loop・phase evictor は**一行も変えていない**。

**差分 4: FLUX.2 の前例は正規化・リサイズの「反面テンプレート」である。** これは §10 の
「正規化の取り違え」リスクの顕在化形態である。FLUX.2 は reference を**target と同じ
bucket 寸法で** trainer 自身の `encode_image` に通して VAE encode する
（`base_trainer.py:10700-10712`、コメントに "Use same bucket dimensions as target
image"）。SenseNova の reference は **bucket に一切参加せず**、per-ref 独立の
smart-resize、ImageNet 正規化、understanding tower 経由である。故障モードが厄介で、
**ref を trainer の画像ロード経路に通してから ViT に渡すと、形状は patchify 段で
偶然合いうるため、エラーなしに誤正規化の条件付けが学習される。** 防御は規律ではなく
構造で行うこと — reference のロード・前処理・encode を丸ごと `sensenova_ops` 内に
閉じ、**PIL path を受け取る設計**にして trainer の画像 pipeline に ref を触らせない。

**実装（`7a09af52`）**: そのとおりに構造で防いだ。trainer には **path しか渡らず**、
`_load_reference_images` が PIL を開いて vendor `load_image_native` に**そのまま**渡す
（`.convert("RGB")` すら呼ばない — RGBA の flatten は vendor の責務なので、ここで
触ると二重処理になる）。さらに `sensenova_reference_training_test.py` が
`encode_image` と `vae_encode` を**例外送出に差し替えて**テストを走らせており、
「trainer の画像 pipeline に ref が触れないこと」を規律ではなく**構造として**
保証している。

---

## 8. VRAM / offload 戦略

### 8.1 記録済みの block-swap 非対応は学習に転移しない

`MODEL_FACTS.md:2005-2018`（sensenova 行の "Generic rolling block-swap was
deliberately not built for this arch"）の判断は次の理由に基づく。

> rewriting the 3-branch denoise loop's layer-outer/branch-inner ordering to
> support it would cost 2-3x more PCIe traffic than this phase-exclusive scheme,
> while activations and KV-cache dominate the peak regardless

**この機構的理由は generation 固有であり、学習 step には存在しない。** 学習 step は
(1) `no_grad` の und 1 パス、(2) 通常の layer 順の gen forward、(3) backward であり、
3 分岐の denoise loop も rolling schedule も無い。構造的には
`LayerOffloadConductor` が既に serve している他 arch の**学習経路**と同型である
（gradient checkpointing 下で backward が block を再計算し、offloader の仕事は
その時点で当該 block の weight を常駐させること）。

`arch_capabilities.py:521-523` に「acestep は generation では block-swap を持たず、
`blocks_to_swap` は TRAINING 経路でのみ読まれる」という前例があり、
generation の非対応が training の非対応を含意しないことは既にこのリポジトリの
既成事実である。

### 8.2 2 パス forward と rolling offloader の両立

**両立する。理由は 2 つの phase が交錯しないからである。**

- phase 1（und）は `no_grad` なので何も保持しない。
- phase 2（gen）が保持する prefix KV は `no_grad` 下で作られた通常の saved tensor で
  あり、autograd から見れば定数である。メモリコストのみで、text prefix なら小さく、
  reference 有りなら大きくなる。

記録しておくべき実装上の不変条件: **prefix forward は checkpointed region の外に
置くこと。** さもないと backward の再計算が、退避したはずの und weight を再要求する。

### 8.3 フェーズ別の担当

| | Phase 1（LoRA） | Phase 2（gen full FT） |
|---|---|---|
| 主機構 | MoT half-eviction（DONE、既定 OFF） | `LayerOffloadConductor`（gen 半分のみ、PENDING、§8.3.1 の未解決問題あり） |
| und 半分 | 凍結済み。prefix cache 成功後から次の prefix まで CPU 退避 | 同左 |
| 根拠 | weight は int8 で 15.1 GiB、圧迫要因は pixel space の activation で block swap では減らない。half-eviction は粗い粒度（7.55 GiB）で phase 境界あたり 2 転送、`kv_cache_streaming.py:27-35` が学習への転移を明示的に是認している | bf16 gen weight 16.2 GB + gradient がボトルネックになり、per-block の rolling window が効く |

7.55 GiB は構造上 CPU 退避の候補になる understanding-side weight 量であり、allocator の
peak allocated / reserved が同量減る保証ではない。Phase 0 で観測した約 15.06 GiB の差は
**gradient checkpointing OFF と ON の差**で、half-eviction の実測値ではない。
half-eviction の exit 判定には、同一 checkpoint・seed・shape・GC 条件で eviction OFF / ON
を別 process で走らせ、peak allocated / reserved と wall time を個別に記録する必要がある。

**機構名の訂正（旧記述は誤り）。** この表は以前 Phase 2 の主機構として
`TransformerBlockOffloader` を挙げていたが、その docstring は
**"Block offloader for Transformer models (forward-only inference)"** であり
（[`memory_management/block_offloading.py:264-272`](../../backend/core/memory_management/block_offloading.py)）、
学習用ではない。学習側は `LayerOffloadConductor`
（"Orchestrates layer offloading for VRAM-efficient training"、
[`layer_offload_conductor.py:24-32`](../../backend/core/memory_management/layer_offload_conductor.py)）で、
他 arch の学習 `setup_block_swap` は全てこちらを使う（anima / krea2 / ideogram4 /
acestep / ltx2 等の `ops/*_ops.py`）。**しかも下に verbatim 引用している
`kv_cache_streaming.py:27-35` 自身が "training-side offload belongs to
LayerOffloadConductor" と書いており、引用を残したまま表の機構名だけが誤っていた** —
文書内で自己矛盾していた形なので、経緯ごとここに残す。

**2 機構の合成は未解決の設計問題である**（旧記述「互いに素な weight 集合を持つため
素直に合成できる」からの格下げ）。詳細は §8.3.1。

`kv_cache_streaming.py:27-35` の verbatim:

> this streamer does NOT apply to training -- a training step is a single-timestep
> forward/backward with no multi-step denoise loop, so no persistent read-many KV
> cache exists to stream; training-side offload belongs to LayerOffloadConductor.
> What DOES transfer is the MoT half-eviction CONCEPT from mot_phase_eviction.py:
> if fine-tuning freezes the understanding branch (likely for image-gen tunes),
> its weight-half can be CPU-evicted during training for a similar VRAM saving.
> Evaluate that when training is built; reuse the layer-selection logic, not this
> module.

**DONE（driver）**: 推論用 callback は再利用せず、学習専用の `full / prefix /
denoise` state machine を実装した。2 周目の `denoise -> prefix` は gen D2H 完了後に
und H2D、`prefix -> denoise` は und D2H 完了後に gen H2D とし、同一 phase は no-op。
転送は correctness 優先の blocking copy とする。H2D 非同期化は未実装で、将来の
opt-in 最適化として残す（下の実測は既定 ON を正当化しなかったが、その測定自体が
機構の有効性を判定していない — §8.3 の gate を参照）。
prefix 失敗時は `prefix` に留まるため retry で余分な転送を起こさず、LoRA の
forward / backward / optimizer / save 中は gen half を GPU に維持する。selector は
LoRA injection 後の live tree から Parameter と永続 buffer を選び、non-persistent
buffer と rotary を除外し、42 layer の両 half の形状・dtype 対称性を fail-closed で
検証する。partial transfer 失敗は再利用不能にし、全 weight の CPU 正規化を試みる。

**DONE（measurement、2026-08-24）**: exit-smoke と同じ plain-int8 checkpoint、
RTX 6000 Ada 48 GB、seed 1234、native attention、64×64、B1、rank 1 / alpha 1、
GC ON、3 step を OFF / ON の別 process で測定した。

| | OFF | ON |
|---|---:|---:|
| peak allocated | 18.09 GiB | 17.59 GiB |
| peak reserved | 18.19 GiB | 18.00 GiB |
| train loop wall | 3.278 s | 18.400 s |
| model load込み wall | 22.759 s | 37.330 s |

**この条件で観測された事実**: ON は peak allocated を 0.50 GiB（2.76%）、reserved を
0.18 GiB（1.01%）下げ、train loop は 5.61 倍遅くなった。両 arm の 3 loss、
live / saved LoRA SHA-256、882-tensor checkpoint は一致し、fresh runtime の strength 0
parity と 294 apply / restore も両方で成立した（correctness は OFF / ON で同一）。
64×64 ではモデルロード時に両 half が同時に載る初期 high-water が peak を支配するため、
この shape では 7.55 GiB の退避候補量が end-to-end peak の削減へ直結しない。

**運用判断（現行）**: 既定 OFF を維持し、VRAM 制約時のみ opt-in とする。根拠は
非対称性ひとつである — ON のコスト（測定した shape での 5.61 倍）は観測されているのに
対し、ON の便益はまだどの shape でも観測されていない。既定 OFF なら、未測定の解像度帯
でユーザが観測済みのコストだけを踏むことはない。これは「機構が無効だ」という判断では
ない（下の gate を参照）。

**未解決の gate**: この測定は機構そのものの有効性を判定していない。64×64 は退避候補量
7.55 GiB に対して活性化・活動集合が小さすぎ、初期 high-water が支配する shape であって、
half-eviction が効くと想定した領域（activation が支配的になる高解像度・長 prefix）を
一切踏んでいない。したがって「half-eviction は VRAM をほとんど下げない」も
「5.61 倍の減速は本質的コストである」も、この測定からは結論できない。exit を主張する
なら、activation が支配する解像度で同一 checkpoint / seed / GC 条件の OFF / ON を
別 process で取り直す必要がある。

**減速の数値は再測定が必要**: 上の ON arm は `sensenova_phase_eviction.py` の
staging 実装、すなわち host 側で `.to("cpu")`（pageable）→ `pin_memory()` の
二重コピーを毎 transfer 行っていた版に対する測定である。この staging 実装は並行作業で
変更されつつあるため、5.61 倍という比は現行実装の値ではない。二重コピーが減速の
支配要因だったかどうかは**未測定であり、断定しない** — blocking copy であること、
転送量が phase 境界あたり 7.55 GiB であること自体も候補として残る。

### 8.3.1 2 機構の合成は未解決の設計問題（旧「素直に合成できる」を格下げ）

以前 §8.3 は「2 つの機構は互いに素な weight 集合を持つため素直に合成できる」と
書いていた。**これは楽観的すぎたので未解決問題に格下げする。** 理由は weight 集合では
なく**モジュール粒度**にある。

- `LayerOffloadConductor` は `layers` の各要素に `register_forward_pre_hook` /
  `register_full_backward_hook` を張り、staging は `layer.to(self.device)` で
  **モジュール丸ごと**行う（`layer_offload_conductor.py:334-359`, `:175-179`）。
- SenseNova の `Qwen3DecoderLayer` は**両 half を同一モジュール内の兄弟属性として持つ**
  （`q_proj_mot_gen` / `mlp_mot_gen` などが und 側と並ぶ。
  `vendor/modeling_qwen3.py:458-490`）。gen half は独立した部分木ではない。
- したがって decoder layer をそのまま conductor に渡すと、`layer.to(device)` が
  **evictor が CPU 退避したはずの und weight を引き戻す**か、device 不整合を起こす。

合成するには conductor に「層ごとの gen half だけ」を渡せる必要があるが、現状の
`select_mot_weight_modules` は層構造を持たない**平坦なモジュール列**を返す
（`training/sensenova_phase_eviction.py:41-45`）ため、そのままでは per-layer の
indexable list にならない。**conductor がサブモジュール粒度のリストを受けられるかは
未調査**であり、受けられたとしても hook が層単位ではなく Linear 単位で発火する点の
検討が要る。§12 に未測定事項として登録する。

**この依存関係は Phase 2b の VRAM 前提に直結する。** Phase 2b は
gen bf16 16.2 GB + gradient 16.2 GB で、weights と gradients だけで 48 GB カードの
32.4 GB を占める構造である（§6.2 の算術）。und half 7.55 GiB の退避が残る唯一の余白で
あるにもかかわらず、**half-eviction の有効性 gate は §8.3 のとおり未解決のまま**である。
したがって **gate の消化（activation が支配する解像度での OFF / ON 再測定）を
Phase 2b の最初の作業項目に置く**（§11 Phase 2b-0）。

### 8.3.2 und 学習と half-eviction（原理的には両立する）

und branch を学習対象にする場合（§13）の eviction 契約をここに置く。

> **und 学習と half-eviction は、単一 `loss.backward()` 実装では両立しない**
> （backward の途中に weight swap を挟む座標が無い）。**原理的には両立する**:
> und / gen の境界は prefix KV cache（**Phase 0 実測 50.5 MiB @258 token**）で、
> 境界勾配も同形なので、**合計 ~100 MiB でグラフを切断し 2 回の backward に
> 分割できる**。したがって contract 側の拒否は「この実装形のスコープ制限」であって
> **不可能宣言ではない**。

分割する場合は既存 evictor の 3 状態（`full` / `prefix` / `denoise`）を 4 相に拡張する。

| 相 | 内容 | 常駐 half |
|---|---|---|
| `prefix` | und forward（勾配付き、境界 KV を葉として保持） | und |
| `denoise` | gen forward + **gen backward**（境界 KV の `.grad` まで） | gen |
| `und_backward` | und forward を**再計算**し `torch.autograd.backward(recomputed_kv, grad_tensors=kv_grad)` | und |
| `full` | 既存 | 両方 |

- 相 3 の「16.2 GB の勾配」問題は fused backward（§6.2 改訂）で消える。
- **weight 同時常駐は 16.2 GB + 境界 + 活性 ≈ ~19-21 GB** となり、**24 GB 級カードでの
  both-branch Full-FT が視野に入る**。**これは構造上の見積もりであって断定ではない** —
  half-eviction の有効性 gate は §8.3 のとおり未解決で、現行 staging 実装の転送コストも
  未知だからである（§8.3 の「減速の数値は再測定が必要」）。
  > **【U-2-4 実測】この見積もりは着地しなかった。** 実 run の peak は
  > **32.66 GiB allocated / 33.91 GiB reserved** で、**19-21 GB ではない**。
  > 原因は分割でも eviction でもなく **placement の順序**である —
  > `load_components` が 588 Linear を materialize して `.to(device)` した時点で
  > **両 half が同時に GPU に載り**、evictor はその後（`_prepare_models` の後）に
  > 作られる。実測でも `model_resident == peak_allocated`（32.66 GiB、完全一致）で、
  > **学習 step は一度もロード時の high-water を超えていない**。
  > すなわち 4 相 eviction が下げるのは **step の常駐量**であって
  > **ロード時の high-water ではない**。24 GB 級を狙うなら、loader 側で
  > half を 1 つずつ置く（§6.2 条件 5 の「per-Linear に dequant → 解放」を
  > **placement にも**適用する）変更が別途要る。**step 中の最小常駐量は測っていない**
  > （peak しか記録していないため）。詳細は §13.4 の U-2-4 実測。
- `und_backward → prefix` は **no-op** にできる（und 常駐のまま次 step へ）ので、
  転送は 1 往復節約できる。
- **MNT > 1 では境界勾配を累積してから相 3 を 1 回だけ回す。** KV 葉の `.grad` は
  backward 間で自然に加算され、und は iteration 間で不変なので数学的に正確である。
  **副作用として「gen は MNT 回 / und は batch あたり 1 回」という更新頻度の非対称**が
  生まれる。これは欠陥ではなく設計事実として記録する。
  > **【U-2-4 訂正】「境界勾配を累積する」が成立しない理由は、形ではなく
  > und の不変性である。**
  > **形は揃う** — SenseNova は契約上 B1 で（`_collate_sensenova_b1_prefix` は
  > prefix を 1 本しか許さない）、MNT ループは**その 1 item の timestep を回し、
  > 毎 iteration `captions[0]` を再 encode する**（`_sensenova_mnt_conditioning`）。
  > したがって窓内の境界 K/V は同形で、素直に加算できる。
  > **成立しないのは設計の厳密性の前提の方である**: MNT ループは
  > **iteration ごとに optimizer を step する**（そもそも graph を retain せず
  > prefix を再計算しているのはそのためである）ので、窓末で流す累積勾配は
  > **既に動いた und weight に対して**逆伝播することになる。
  > 勾配 accumulation なら窓が閉じるまで step しないのでこの前提は成り立つが、
  > その経路は別理由で拒否されている（§6.2 条件 2）。
  > 実装は相 3 を **iteration ごとに** 回す。これは厳密である — 相 3 の再計算は
  > 自分の相 1 と同じ und weight を読み（間に走る gen hook は und weight に
  > 触れない）、gen half の更新は und forward に影響しない。
  > **コストは weight 往復が step あたり 2 回ではなく MNT iteration あたり 2 回**に
  > なることで、これは拒否ではなく `training_log` チャネルで告知する
  > （`sensenova_four_phase_mnt_cost`）。更新頻度の非対称という結論は変わらない。
  > **なお MNT>1 は到達可能である** — `assert_full_finetune_contract` が拒否するのは
  > `gradient_accumulation_steps` であって `multi_noise_timesteps` ではなく、
  > `BaseTrainer` は MNT >= 1 しか要求しない。**両者を同一視していた旧記述は誤り**。
- **相 3 は backward 直後に走らせる（optimizer step 地点ではない）。**
  `_update_census.assert_complete()` は MNT ループ内で呼ばれ、これは
  `should_step_optimizer` ブロックより**上流**である（`base_trainer.py`。
  行番号は動くので symbol で引く — 近傍の行番号を書いた旧版は、同じ節の編集で
  既に 1 度ずれて OOM bucket の print を指していた）。
  相 3 を step 地点まで遅らせると、**正しい run で census が「und half は 1 つも
  更新されていない」と報告する**。§12 が「census × 4 相の順序」を未検証項目として
  挙げていたのはまさにこれで、実装時に踏んだ。
- **4 相は fused backward ルート専用である。** 相 3 は gen half を CPU に置いたまま
  終わるので、その後に `optimizer.step()` が走ると **CPU の parameter に CUDA の
  勾配**が当たる。fused backward には その呼び出し地点が無く、各 half は自分が
  常駐している間に自分の hook で更新される。したがって LoRA では拒否する。
- **und forward 2 回のコストと weight 往復コストは未測定。**
  `probes/text_encode_vs_step.py`（`a67640ed`）には既に sensenova アームがあり、
  prefix forward と DiT step の壁時計を測れるので、**「prefix / step 比の実測」を
  この設計の exit gate に指定する**（§13.4 U-2-4）。

#### 【U-2-4 実測】exit gate（2026-08-25: PASS）

実 checkpoint（`M:/model/sensenova/sensenova_int8.safetensors`）、RTX 6000 Ada、
1024px、prefix 467 token（caption 30 本、実測 p50 218 token）、image token 1024、
B1、GC ON、native attention、und 側の勾配は **both branch の LoRA rank 4** で供給、
`set_per_process_memory_fraction(0.72)`。probe は
`probes/text_encode_vs_step.py --arm sensenova-four-phase`（新設。既存の sensenova
アームは prefix を `no_grad` の vendor 経路で測るので、4 相が必要とする
**勾配付き prefix・境界葉での分割・und backward** を測れなかった）。
**warmup 5 / 計測 25**（probe の既定）、**以下はすべて実測値**。

| 量 | p50 (s) | mean (s) |
|---|---:|---:|
| single backward: prefix forward | 0.1728 | 0.1836 |
| single backward: gen forward + backward（両 half 貫通） | 1.7584 | 1.7826 |
| 4 相: prefix forward | 0.1708 | 0.1811 |
| 4 相: denoise（gen forward + 境界までの backward） | 1.4288 | 1.4266 |
| 4 相: und forward の**再計算** | 0.1897 | 0.2076 |
| 4 相: und backward | 0.3291 | 0.3343 |

**比は p50 と mean を混ぜず、両方で出す**（初稿は p50 の内訳から mean 由来の合計を
引いて、成分の和と 57% 食い違う数字を書いていた）。

| 比 | p50 基準 | mean 基準 |
|---|---:|---:|
| prefix / step | 0.098 | 0.103 |
| 再計算 / single backward 合計 | 0.098 | 0.106 |
| **4 相合計 / single backward 合計** | **1.097** | **1.093** |
| 分解残差（denoise + und bwd を single の gen fwd+bwd と比較） | **−0.03%** | −1.22% |

- **分割の限界コストは +9.3〜+9.7%。** 初稿の「+8.25%」は n=3 で、しかも分母側の
  prefix forward に外れ値（mean 0.2715 対 p50 0.2010）を抱えていた。
  **4 桁目まで主張できる測定ではなかったので撤回する。** 結論は変わらない —
  **再計算は経済的である。**
- **分解が成立している**（p50 基準で残差 −0.03%）ので、各相は名前どおりのものを
  測っている。
- **eviction 自体の転送コストは分割の約 7 倍である。** 同 probe が
  production の staging 経路（`stage_modules_to_pinned_cpu` +
  `_move_modules_to_device`）で und half（**8,161,563,648 byte = 7.60 GiB**、int8）の
  往復を測ると D2H p50 0.3363 s / H2D p50 0.3296 s、**往復 0.666 s**。
  4 相は step あたり 2 往復なので **+1.332 s = single backward の +69.0%（p50）/
  +67.8%（mean）**。
  **この比は §8.3 の未解決 gate（eviction の有効性）に属するのであって、分割の
  コストではない** — 三状態 evictor も同じ 2 往復を払うので、**分割が足す転送は
  ゼロである**。bf16 の both branch では half が 15.09 GiB になるので転送量は倍になる。
  - 初稿の「往復 1.046 s / +103.7% / 分割の 12 倍」は**測定の欠陥である**。
    往復ループに warmup が無く、1 回目が `stage_modules_to_pinned_cpu` 内の
    pinned 確保を含んでいた（以後は torch の caching host allocator が使い回す）。
    warmup を入れた現在は成分の和と一致する（0.3363 + 0.3296 = 0.666）。
- **測っていないもの**: bf16 の und Linear での比（上記は int8 Linear の値である）、
  1024px 以外の解像度、pinned 転送の非同期化。

### 8.4 half-eviction 再利用時の注意

`mot_phase_eviction.py:115-136` の層選択は **Parameter の有無ではなく永続性**を
判別子にしている。`Int8Linear` は Parameter を 1 つも持たないため、
`parameters()` ベースの規則は RMSNorm（約 0.21 GiB）しか選ばず、**2 度にわたって
無害に見える形で不活性なまま出荷された**。学習側で層選択ロジックを再利用する際は
この判別子ごと持ってくる。`rotary_emb` は名前で除外される。

学習中 sample は evictor を学習 step と同じ `enter_prefix` / `enter_denoise` の
遷移ペアで駆動するため、half-eviction ON でも両 half が同時常駐しない。生成後の
evictor は `prefix` / `denoise` のどちらかに置かれたままになるが、次 step の
`encode_prompt` はどちらからも合法に遷移する（§11 Phase 1）。

なお `block_swap` と optimizer の互換性検証（`base_trainer.py:3546-3642`）は
arch 非依存で、`blocks_to_swap` / `num_optimizer_groups` / `optimizer_type` だけを
見る。SenseNova 用の追加は不要だが、CLAUDE.md の Block Swap × 8bit optimizer の
制約はそのまま適用される。

---

## 9. 既存コードベースへの統合ポイント

[`ADD_A_MODEL_ARCHITECTURE.md`](ADD_A_MODEL_ARCHITECTURE.md) の §4 が正規手順。
以下は初期計画を残した実装マップである。DONE は現行コードへ統合済み、PENDING は
§11 の exit criteria をまだ満たしていない。

### DONE — Phase 1 ファイル

- `backend/core/training/arch/sensenova.py` — `name = "sensenova"`,
  `wiring = SENSENOVA_WIRING`, `pixel_align = 32`（patch 16 × merge 2。
  `vae_scale_factor` ではない）, `temporal = None`。8 つの抽象メソッドを
  `ops/sensenova_ops.py` へ委譲するだけの薄い層（`arch/krea2.py` がテンプレート）。
  `vae_encode` は MiniT2I の pixel-space 分岐と同様、共有 VAE staging より**前に**
  dispatch され自己完結する必要がある。`vae_decode` は raise でよい。
- `backend/core/training/ops/sensenova_ops.py` — `load_components`,
  `setup_block_swap`, `setup_attention_backend`, `encode_prompt`（= prefix KV 構築）,
  `vae_encode`（= pixel passthrough）, `train_step`, `generate_sample`。
- `backend/core/training/adapters/sensenova_adapter.py` —
  `SenseNovaLoRAAdapter`（`iter_sensenova_lora_targets` を再利用）。初期案の
  `SenseNovaFullParameterAdapter` は追加せず、full FT は共通のロード前 capability
  guard で拒否する。

### DONE — 登録（漏れると import 時に落ちる = 安全）

1. `arch/__init__.py` — import 追加、`ARCH_REGISTRY` に追加、
   **`_EXPECTED_ARCH_KEYS` にも追加**（module レベルの assert がある）、
   `resolve_arch_name` に `is_sensenova` の分岐を追加。
2. `training/components/wiring.py` — `SENSENOVA_WIRING` を re-export（import 節と
   `__all__` の両方）。
3. `adapters/__init__.py` — import と `__all__`。
4. `lora_trainer.py` — adapter import と `_create_adapter` の分岐、
   SenseNova adapter 選択。
5. `arch_capabilities.py` / 既存の full-parameter・ReLoRA preflight — full FT と ReLoRA
   をモデルロード前に拒否。

### DONE — `base_trainer.py`（漏れると静かに間違う = 危険）

- flag 代入ブロック **2 箇所**（`:1271-1283` と `:1875-1887`。後者は
  `_load_checkpoint_as_base` 側の重複）。
- loader dispatch（`:1293-1322`）。`self.arch` は load 後に bind されるため、
  ここは `ops.load_components` を直接呼ぶ。
- `encode_caption`（`:4373-4426`）— 手書きの if 連鎖のままで、`self.arch.encode_prompt`
  へは routing されていない。`(embeds, aux)` タプル形状を返す arch 群の述語
  （`:7595`, `:7657`, `:7733`）にも追加が要る。
- DiT ファミリ述語 **3 箇所**（`:4592`, `:4601`, `:4654`。コメントが「3 つは
  一致させ続けること」と明記している）。
- `_execute_forward_backward`（`:5631-5790`）に `TrainStepContext` を組む分岐。
- batch collation（`:10853-10881`）と MNT 経路（`:11052-11080`）。
- timestep 既定の連鎖（`:8234-8245`）と
  `param_defaults.TIMESTEP_SAMPLING_DEFAULTS_BY_ARCH["sensenova"]`（§4.5 の値）。

### その他

- **DONE:** `model_loader.detect_prediction_config` の flow / velocity 分類と退行テスト。
- **DONE:** `train_runner.py` の bf16 強制、B1・単一 flavour int8 base・no-reference・no-block-swap
  preflight、on-the-fly prefix/pixel 経路。
- **DONE:** training-method capability による full FT / ReLoRA / ControlNet の UI と
  backend refusal。SenseNova は VAE を持たないが、明示 VAE path/store の decoder
  training は別契約として許可する。
- **DONE:** real trainer の 3-step exit smoke と fresh runtime strength 0 parity。
- **DONE:** Phase 1 half-eviction の OFF / ON 別 process 計測。
- **DONE:** 学習中 sample（`arch/sensenova.py::sample` → `ops.generate_sample`）と
  `debug_latents` の pixel dump（`_execute_forward_backward` の `is_sensenova` 分岐が
  `TrainStepContext` の debug 3 フィールドを渡す）。`train_runner` と
  `base_trainer.train()` の `sample_every` 強制 0 は削除済み。API / フロントの変更は
  不要だった（§11 Phase 1 参照）。

### DONE — Phase 3（`7a09af52`..`611a4a24`）

§9 の初版一覧に未記載だった統合ポイントで、実装時に必要になったもの:

- `use_reference_images` のゲート **6 箇所中 4 箇所**の解除（残り 2 は意図的に
  flux2 限定。§7.5 差分 1）。
- `encode_caption` に `reference_image_paths` を通す配線と、その sensenova 分岐。
  **arch 経由**（`arch.encode_prompt`）で渡すので、他 arch はこの引数を無視する。
- `SenseNovaTrainingPrefix.text_length` の意味論変更（§7.5 差分 2）。
- `arch_capabilities` の `reference_images` 宣言から sensenova を外す。

### PENDING

- Phase 2b full FT 本体（`ops/sensenova_ops.py` の gate と `load_components` の
  method-aware 化は `cc296e84` で、**adapter・契約・fused backward の decoupling は
  `601d0271`**、**stochastic rounding の強制と dropout guard は `24220b5c`** で
  DONE。通知経路（`training_log`）も着地。**残るのは checkpoint format の決定と
  受付の解錠**で、§6.4 と §13.4 U-2-2 に列挙してある）と、
  Phase U（§13）。

### DONE — 登録から自動的に得られたもの

`_build_cache_namespace` は `self.arch.name` を読むだけになっており、
`pixel_align` / `temporal` も handler のクラス属性を読む宣言的な機構なので、
cache namespace と alignment は登録だけで有効になった。

---

## 10. SenseNova 固有のリスク

| リスク | 内容 | 緩和 |
|---|---|---|
| mixed forward の欠落 | issue #207。1 パスで und/gen を混ぜられない | 2 パス構造を設計の前提にする（§4.2）。修正を前提にしない |
| int8 base のみ | weight が buffer なので full FT は 1 パラメータも学習しない（**「不可能」ではない**: `cc296e84` が学習する half を実 Parameter へ materialize する経路を実装した。ただし受付は未解錠） | Phase 2 をガード先行にする（§6.1）。経路 (a) と残作業は §6.4 |
| bf16 丸め欠陥 | 8.1B full FT でそのまま継承する（凍結率 91% 前後、`\|w\| <= 512*lr` でしか動かない） | `optimizer_stochastic_rounding` の既定を SenseNova full FT でどうするか実装時に決定（§6.3, §12）。`optimizer: adamw` は構造的にカバー不能 |
| 短 horizon での評価 | stochastic rounding は 1k step 未満では誤差が信号と同程度 | 数百 step の full FT で品質判断をしない（§6.3） |
| gradient checkpointing OFF | 量子化 base の上に bf16 全体が実体化し、逆に増える | §4.7 の専用 non-reentrant loop を必須にする |
| stock gradient checkpointing | `past_key_values` が `None` に置換され、prefix conditioning が消える | layer 標準 GC を使わず、cache 不変性と ON/OFF parity をテストする |
| batch > 1 の ragged prefix | ref 有無だけでは caption/ref token 長が揃わず、gen flash attention に padding mask が無い | 初版は物理 batch 1。effective batch は gradient accumulation |
| `noise_scale` の再現漏れ | bucket ごとに変わる値をモデルにも渡す必要がある | §4.4。学習 step の必須要素として扱う |
| `Int8Linear` の isinstance 罠 | `nn.Linear` サブクラスでないため 294 件を黙って取りこぼす | 既存の共有述語を使う（§5.4） |
| base census の isinstance 罠（別方向） | `ConvRotInt8Linear` は `Int8Linear` の subclass なので、`isinstance` census は ConvRot を plain の数に畳み込み mixed base を黙って受理する | `type(m) is cls` で数える（§5.3、`0c9ea86b`） |
| ConvRot base の train / inference skew | 学習は dequant 経路（W-int8 / A-bf16）に fit するが、推論は常に fused W8A8。fit 先とデプロイ先が activation 量子化誤差の分だけ異なる。**実害は未測定** | 既知制約として §5.3 に記録。skew を避けるなら plain int8 base を選ぶ |
| 正規化の取り違え | reference は ImageNet、target は 0.5/0.5。**顕在化形態**: FLUX.2 の前例（ref を target と同じ bucket 寸法で VAE encode）をテンプレートにすると、形状が patchify 段で偶然合いエラーなしに誤正規化が学習される | §4.6, §7.4。防御は規律ではなく構造で — ref の前処理を `sensenova_ops` に閉じる（§7.5 差分 4） |
| offload 2 機構の合成 | `LayerOffloadConductor` はモジュール丸ごと staging するが、SenseNova の decoder layer は両 half を同一モジュールに持つ | 未解決。conductor のサブモジュール粒度対応が未調査（§8.3.1、§12） |
| und LoRA の推論側サイレント欠落 | `apply_lora_group` は gen のみ列挙して lookup するため、und キーは applied 件数が減るだけで**エラーが出ない** | 列挙器に `branch` を追加し applied カウントを検証する（§13.3） |
| und 学習のサイレント無効化 | `_assert_immutable_prefix_cache` の `requires_grad` 拒否を単に外すと、prefix が `no_grad` のままでも loss は正常に下がり und は学習されない | 拒否を外すのではなく**逆向きの positive assertion**（全 42 層の K/V が `grad_fn` を持つ）に差し替える（§13.3） |
| optimizer 名と実挙動の乖離 | Ring Buffer optimizer は名前と docstring が CPU state を主張するが、`get_state_buffer` を渡す呼び出し側が無く GPU に確保される。予算を名前で見積もると 32.4 GB 外す | 発生源（`RINGBUFFER_OPTIMIZERS.md` と docstring）を訂正する。設計上の帰結は §6.5 |
| fused hook のサイレント CPU-skip | `if not param.is_cuda: return` は 4 相 eviction と順序が狂うと更新されない half を作るが loss は下がる | fail-loud 化 + step ごとの updated-param census（G-RB3、§6.5） |
| prediction config の退行 | 既存の flow-matching 登録を学習統合時に外すと静かに誤る | §9 の退行テストで固定 |
| half-eviction の層選択 | Parameter ベースの規則は 2 度不活性のまま出荷された | 判別子ごと再利用する（§8.4） |
| prefix forward のコスト | Qwen3-8B 全体を毎 step 通す | 実測後にキャッシュ可否を判断（§12） |
| pixel space の activation | VAE が無いぶん activation が pixel 解像度に比例 | gradient checkpointing + 解像度上限。block swap では減らない |

---

## 11. フェーズ分割

初期計画を残し、現在の DONE / PENDING 境界を明示する。

**Phase U（understanding branch の学習）はこの系列に含めない。** 依存は Phase 1 のみで、
PENDING の Phase 2b / 3 に混ぜると偽の依存が生まれるため、独立フェーズとして §13 に
分離した。

### Phase 0 — 前提確認（DONE）

- `forward_gen` を勾配付きで通す最小 probe。物理 batch 1、64×64、plain int8 base、
  推論用 flash cache/streamer 未準備、`update_cache=False` fallback で image token 側に
  backward が通ることを確認する。
- `no_grad` の prefix forward + 勾配付き gen forward の 2 パスで
  有限の loss が出ることを確認する（収束実験は行わない）。
- stock GC ではなく §4.7 の専用 non-reentrant checkpoint loop を使い、GC ON / OFF を
  別 process で実測する。
- **exit criteria**: 有限 loss、1 backward 目で 294 `lora_up.grad` が finite、optimizer
  step 後の 2 backward 目で `lora_down.grad` にも nonzero が届くこと、GC ON/OFF の
  loss/gradient parity、prefix cache の長さと tensor identity が forward/backward 前後で
  不変であること、peak allocated/reserved VRAM の記録。

#### Phase 0 実測（2026-08-23: PASS）

`backend/core/training/probes/sensenova_real_checkpoint.py` を GC OFF / ON の別 process で
実行した。checkpoint は plain-int8 `sensenova_int8.safetensors`（18,872,241,160 bytes）、
GPU は RTX 6000 Ada 48 GB、native attention、物理 B1、64×64、seed 1234、`t=0.5`、
rank 1 / alpha 1 の fp32 LoRA である。実型 census は plain `Int8Linear` 588、ConvRot 0、
学習 target 294（attention 168 + MLP 126）、decoder/cache layer 42 だった。

| | GC OFF | GC ON |
|---|---:|---:|
| step 1 loss | 1.4259879589 | 1.4259879589 |
| step 1 up grad L2 / nonzero | 0.03093660 / 294 | 0.03093660 / 294 |
| step 1 peak allocated | 33.12 GiB | 18.07 GiB |
| step 1 wall（optimizer state 初期化込み） | 5.069 s | 4.484 s |
| step 2 loss | 1.4267587662 | 1.4267587662 |
| step 2 up / down grad L2 | 0.02673777 / 0.01044420 | 0.02673777 / 0.01044420 |
| step 2 up / down nonzero | 294 / 294 | 294 / 294 |
| step 2 peak allocated | 33.16 GiB | 18.09 GiB |
| step 2 wall | 0.363 s | 0.531 s |
| max reserved | 33.42 GiB | 18.14 GiB |

両 arm で 2 loss、全 grad L2、全 grad tensor の SHA-256 が一致した。1 backward 目の
down grad は zero-up 初期化どおり全 0 で、optimizer step 後は 294/294 が nonzero になった。
prefix cache は sequence length 258 のまま、cache 本体、layers list、全 layer、全 K/V
tensor の object identity / data pointer / shape / value が 2 backward 後も不変だった。
model resident allocated は 17.59 GiB、prefix の live allocated 増分は 50.5 MiB である
（reserved 増分は allocator cache を含む）。専用 non-reentrant loop は数値を変えず、
この条件で GC OFF 比の peak allocated を約 15.06 GiB 下げたため、checkpoint-safe GC
の前提を満たす。この数値は GC OFF / ON 差であり、half-eviction OFF / ON による
削減量ではない（§8.3）。

### Phase 1 — LoRA（DONE）

- `arch/sensenova.py` + `ops/sensenova_ops.py` + `adapters/sensenova_adapter.py`。
- 登録 5 箇所 + `base_trainer.py` の分岐（§9）。
- `TIMESTEP_SAMPLING_DEFAULTS_BY_ARCH` と、既存の flow-matching prediction config の退行テスト。
- `train_step`: MiniT2I の骨格 + prefix KV conditioning + per-sample `noise_scale`。
- 物理 `batch_size=1` の config-time guardと、prefix cache を保持する専用
  non-reentrant checkpoint loop。
- checkpoint 保存形式: 推論側 loader が読む `neo_hf_lora` 方言との round-trip を実装済み
  （`LoRATrainer.load_checkpoint` は arch 非依存で
  `{lora_name}.lora_down.weight` / `.lora_up.weight` を読む）。
- **DONE:** 上記の arch/ops/adapter、294-target と単一 flavour int8 base の gate
  （plain / ConvRot、`0c9ea86b`）、checkpoint-safe
  2-pass core、trainer/runner integration、prediction/defaults、保存と runtime round-trip。
- **DONE:** MoT half-eviction の学習側再利用（opt-in）。
- **DONE:** 学習中 sample と `debug_latents`（`dc91bef1`）。要点:
  - `generate_sample` は `sensenova_pipeline_ops` の `normalize_resolution` →
    `encode_prompt` → `denoise_loop` → `tensor_to_image` をそのまま駆動する。
    denoise は一切再実装していない。`timestep_shift` / `cfg_norm` は
    `SENSENOVA_GENERATION_DEFAULTS`、解像度と step 数は YAML の値をそのまま使う
    （grid 外の解像度は snap して warn）。学習中の LoRA は別途 apply 不要
    （`LoRALinearLayer` wrapper 自体が生成 forward の呼ぶ module である）。
  - `finally` で attention mode の TRAINING 再 stamp（§5.4）、`train()` 状態、
    prefix cache を必ず復元する。生成失敗は traceback を出して `None` を返す。
    `base_trainer` の sample ブロック（`:9070`, `:11749`）に例外ガードが無いため、
    伝播させると run ごと落ちる。
  - half-eviction ON との併用は拒否ではなく対応。生成は学習 step と同じ
    `enter_prefix` / `enter_denoise` の遷移ペアで evictor を駆動するので両 half が
    同時常駐しない。evictor は呼び出し終了時の状態に置かれたままになるが、次 step の
    `encode_prompt` はどちらからも合法に遷移できる（§8.4）。
  - `debug_latents` は pixel space の等価物。SenseNova の「latent」は既に
    `[-1,1]` RGB なので `target` / `noisy` / `pred_x0` を decode 無しで webp 化し、
    `.pt` はスカラーのみ。ファイル名は既存の `visualize_debug_latent` が導出する規約
    （`latents_t<ts>.pt` → `decode_t<ts>_{target,noisy,pred_x0}.webp`）に合わせたため
    API / フロントの変更は無い。
- **DONE:** §8.3 の OFF / ON 別 process 計測（64×64、0.50 GiB 削減 / train loop 5.61 倍）。
  既定 OFF を維持する運用判断もそこに記す。この shape は機構の有効性そのものを
  判定していない未解決の gate であり、減速比は変更中の staging 実装に対する値である
  （§8.3）。
- **DONE exit criteria:** real trainer の 3-step smoke で有限 loss、保存した LoRA が推論側の
  runtime LoRA としてそのままロードでき、strength 0 で base 出力と一致すること。

#### Phase 1 exit-smoke（DONE、2026-08-24）

通常の pytest から収集されない opt-in probe である。repo の venv から次を実行すると、
trainer arm（実画像 64×64、B1、rank 1 / alpha 1、fp32 LoRA、plain-int8 base /
bf16 compute・training、
native attention、専用 non-reentrant GC、3 step、step 3 保存、sample なし）を起動し、
完全終了後に別 process の runtime arm が fresh model をロードする。

```text
<repo>/venv/Scripts/python.exe backend/core/training/probes/sensenova_real_checkpoint.py --model-path <plain-int8-checkpoint> --trainer-exit-smoke
```

trainer arm は 3 個の有限 loss、`neo_hf_lora` metadata、882 tensors / 294 targets、
全 LoRA tensor の有限性・SHA-256・peak allocated/reserved を検証する。runtime arm は
同じ prompt / seed / 64×64 / 1 step の base denoise と saved LoRA の strength 0 を
`torch.equal` で比較し、294 apply / 294 restore と全 module identity の復元を確認する。
各 arm は独立した subprocess として個別の timeout/watchdog 下で順に起動し、共有一時
ディレクトリは両 arm 完了後（例外時も unwind 後）に削除する。

plain-int8 checkpoint、RTX 6000 Ada 48 GB、seed 1234、native attention、64×64、
B1、rank 1 / alpha 1、GC ON で実行した。trainer の loss は
`0.41384110 / 0.37259370 / 0.52550042` で全て有限、training callback は
step `1 / 2 / 3` を報告した。step 3 の保存物は `neo_hf_lora`、294 targets、
882 tensors（588 weight tensors）で、live LoRA と保存 weight の SHA-256 が一致した。
peak allocated は 18.09 GiB、peak reserved は 18.19 GiB だった。

> **【2026-08-24 追記】この 3 つの loss 値は現行実装では再現しない。上の値は当時の
> 正しい実測なので書き換えず、注記として残す。**
>
> - 記録時点（`58637bc5`、08-24 00:11）の tree では再現する。`ba5181cb^`
>   （= `76671a0e`）で再実行すると残差 ≤ 3.2e-09、これは上の 8 桁丸めそのものである。
> - **現行（`611a4a24`）では `0.41466627 / 0.37605366 / 0.52368003`。**
> - 原因は **`ba5181cb`「Stop quantizing the SenseNova training timestep to bf16」**
>   （08-24 07:40、記録の 7.5 時間後）。`t` を bf16 に量子化するのをやめた**意図的な
>   数値変更**である（`t` は活性ではなく条件付けの**値**で、bf16 は ~2e-3 に丸めていた）。
> - **peak allocated / reserved は 18.09 / 18.19 GiB で現行と一致する。** 変わったのは
>   loss 値だけで、メモリ挙動は同一である。
> - grad digest も動くが、**step 1 の down 方向だけは一致する**。矛盾ではなく、
>   その勾配が厳密にゼロ（`lora_up` のゼロ初期化）で dtype に依存しないためである。
> - **帰属の但し書き**: この A/B は `76671a0e..fa0ebbab` の 20 コミットを一括で比較して
>   おり、`ba5181cb` 単独を分離してはいない。この範囲で SenseNova の学習コード配下に
>   触れるのは 5 コミットだが、**算術を変えるのは `ba5181cb` だけ**である
>   （diff は `t` の `dtype=dtype → torch.float32` と fp32 の分母。他の 4 つは
>   sampling / debug dump、grad-norm の集計単位、batch position の記録、docstring）。
> - **Phase 3 とは無関係である。** `fa0ebbab` と `611a4a24` の reference-free arm は
>   **bit-exact に一致する**（§11 Phase 3 実測）。この隣で「exit-smoke の数値が変わった」
>   だけを読むと Phase 3 を疑いたくなるが、その可能性は bit-exact parity が排除している。
>
> 上書きしない理由: (1) 2026-08-24 の値は正しく取られた実測で、消すと「意図的な数値
> 変更があった」証跡まで消える。(2) 上書きは実際に起きた誤診断を再発させる — 日付の
> 無い triple 1 組だけを見て食い違いに気づいた人は「壊れている」と考え、実際にその
> 調査に時間が使われた。(3) かといって旧 triple だけを残すのも不可で、今日実行する
> 人は別の値を得る。

fresh runtime arm は保存物を 294/294 module に適用し、294/294 を復元した。
復元後の全 module identity は元と一致した。base と strength 0 の denoise tensor は
同じ SHA-256 を持ち、`torch.equal` でも一致した。half-eviction の OFF / ON 実測も
§8.3 のとおり完了し、Phase 1 exit criteria を満たした。

half-eviction の効果を測る場合は、同一 checkpoint / seed / shape / GC 条件で
別 process の OFF / ON を個別に起動する。2026-08-24 の結果は §8.3 に記録した。

```text
# OFF
<repo>/venv/Scripts/python.exe backend/core/training/probes/sensenova_real_checkpoint.py --model-path <plain-int8-checkpoint> --trainer-exit-smoke --smoke-phase-eviction off --smoke-json-out <off-result.json>

# ON
<repo>/venv/Scripts/python.exe backend/core/training/probes/sensenova_real_checkpoint.py --model-path <plain-int8-checkpoint> --trainer-exit-smoke --smoke-phase-eviction on --smoke-json-out <on-result.json>
```

trainer arm の JSON には `phase_eviction`、`wall_time_s`（`train()` のみ）、
`wall_time_with_model_load_s`、`model_load_wall_time_s`、
`peak_memory.allocated/reserved` を記録する。

### Phase 2a — full FT ガード（DONE）

- `TRAINING_UNSUPPORTED` と共通 preflight でモデルロード前に拒否する。初期案の
  `SenseNovaFullParameterAdapter` は不要になったため追加していない。

### Phase 2b — full FT 本体（PENDING、律速は bf16 base の「入手」ではなく「実装」）

**前提条件の表現を訂正した。** 旧見出しは「bf16 base 入手が前提条件」だったが、
現行 gate は未量子化 bf16 base も拒否するため、入手しただけでは動かない（§6.4）。
以下は Phase 1 の §11 と同じ粒度の作業分割である。

- **2b-0 — half-eviction gate の消化（最初に行う）。** §8.3 の未解決 gate を、
  activation が支配する解像度で同一 checkpoint / seed / GC 条件の OFF / ON 別 process
  として取り直す。Phase 2b の VRAM 前提が und half 7.55 GiB の退避に依存するため
  （§8.3.1）、これが未解決のままでは後続の作業量が見積もれない。
- **2b-1 — gate と loader の method-aware 化（DONE、`cc296e84`。= U-2-1）。**
  `_assert_supported_quantized_training_base` と `load_components` が training method を
  見るようになり、受理する供給経路は **(a) の plain int8 限定**に決まった（§6.4）。
  **ただし full FT の受付はまだ開いていない**（§6.4 の「端から端まで到達しない」）。
- **2b-2 — `SenseNovaFullParameterAdapter`（DONE、`601d0271`。= U-2-2 step 1-2）。**
  §6.1 で「共通 preflight だけで fail-closed になるため追加しなかった」もの。
  adapter + `assert_full_finetune_contract` + fused backward の decoupling が着地し、
  decoder 外の gen 側モジュールは**含めない**と決まった（§6.2）。
- **2b-2b — 受付の解錠（DONE。= U-2-2 step 3）。** 2 つの gate を落とし、
  `_apply_sensenova_full_finetune_contract` を足し、**実 checkpoint 上の
  gen branch smoke run**（3 step、census 294/294、mixed 25.129 GiB の保存と
  本番 reader での 294/294 バイト一致再ロード）で通した。実測と、その過程で
  見つかった 7 件（全件修正済み）と、`train_unet` 修正の arch 横断的な副作用
  3 件（同じく全件解決）は §13.4。
  **品質は主張しない。**
- **2b-3 — bf16 rounding-defect の契約（DONE、`601d0271` + `24220b5c`。= U-2-3）。**
  §6.3 の推奨 2 点のうち **`optimizer: adamw` 拒否は `601d0271`**、
  **`optimizer_stochastic_rounding` は `24220b5c`** で着地した。後者は
  「contract 既定 True」ではなく**ルート要件（強制 + stdout 通知）**として実装
  されている（transport が「未指定」を表現できないため。§6.3 (2)）。
  同 commit で dropout guard が full FT では branch によらず無条件になった（§13.3）。
- **2b-4 — offload の合成。** §8.3.1 のモジュール粒度問題を解決する。
  `LayerOffloadConductor` がサブモジュール粒度のリストを受けられるかの調査が先行する。
- **2b-5 — exit smoke（DONE。= U-2-5）。** gen / und / both の 3 branch すべてに
  実 checkpoint 上の run が付き、`mixed` の**両向き**（gen 側 int8 残し / und 側
  int8 残し）と `bf16` を本番 reader で読み戻した。update-nonzero census は
  **gen 294/294、und 289/294、both 583/588** で、動かない 5 個はいずれも
  `und_gradient_unreachable_paths()` が名前で予測したものである。
  **prefix forward を checkpointed region の外に置く不変条件のテストもここで着地した**
  （`sensenova_u2_5_exit_smoke_test.py`、負の対照つき）。実測と、残る未測定事項は
  §13.4 の「U-2-5 実測」。**品質は主張しない。**
- **exit criteria**: **「学習が壊れていないこと」だけを主張し、品質は主張しない。**
  短 horizon では stochastic rounding の誤差が信号と同程度で（§6.3）、A/B が測定として
  無効になるためである。

### Phase 3 — reference 混在（DONE、2026-08-24）

- **3-1 — gate 解除と配線（DONE、`7a09af52`）。** §7.5 差分 1 の 6 箇所のうち
  **4 箇所を解除、2 箇所は意図的に flux2 限定のまま**（テストが出現数 2 を固定）。
- **3-2 — reference prefix の構築（DONE、`7a09af52`）。** §7.5 差分 2
  （`text_length` の一般化）と差分 3（推論側関数の再利用）。reference の前処理は
  `sensenova_ops` 内に閉じた（差分 4）。
- **3-3 — 学習中 sample の ref 対応（DONE、`d7bd9067`）。** 推論側
  `encode_prompt(..., ref_images=..., img_cfg_scale=...)` を、生成バックエンドと
  **同じ位置引数順・同じ kwarg** で駆動する。`img_cfg_scale` は sample config に
  該当フィールドが無いため `SENSENOVA_GENERATION_DEFAULTS` から取る
  （`timestep_shift` / `cfg_norm` と同じ扱い。ハードコードや新規パラメータにはしない）。
  `denoise_loop` は無変更 — branch 情報は prefix に載るので、推論側もそう渡している。
  **condition image は引き続き無視する**が、これは正しい: ControlNet 系の条件付けで
  SenseNova には入口が無く（capability で拒否済み）、reference とは別機構である。
  メッセージも分離した（1 文にまとめると reference まで未対応に読めるため）。
  復元契約は不変で、reference のロードは同じ `try` の中にあるため、読めない path は
  他の失敗と同様に `None` を返す。
- **3-4 — 混在 smoke（DONE、`611a4a24`）。** 下記の実測を参照。
- 既存の run-global `use_reference_images` と per-item `reference_images` を再利用した
  （新しい dataset-level parameter / API 変更なし）。
- **exit criteria: 4 項目すべて PASS**（下記）。

#### Phase 3 実測（2026-08-24: PASS）

`611a4a24`。plain int8 checkpoint、CRef（dataset id 23、`M:\dataset_control`、
`_source` / `_target` / `_instruction` の suffix 規約、production の
`related_images["reference"]` 導出経由）と reference なしデータセットの **2 dataset を
1 run に混在**、per-item presence で batch ごとに分岐。B1、3 step、GC ON、
`set_per_process_memory_fraction(0.72)`。**以下はすべて実測値である。**

- **3 step とも有限 loss**、294/294 target に有限 grad。64px と 1 段上の解像度の
  両方で確認したので、smoke のジオメトリに合わせ込んだ実装ではない。
- **reference-free 経路が 3-1 直前（`fa0ebbab`）と bit-exact**: 3 つの loss、
  6 つの grad digest、live / saved の LoRA hash、peak allocated / reserved が
  **バイト単位で一致**。reference 対応は、reference を使わない経路を一切乱していない。
- **ViT 行数 == splice した placeholder 数**: `grid_hw=[[44,80]]`、downsample 0.5 で
  **880 == 880**、`<IMG_CONTEXT>` は token id **151669** に解決。
  **モックでは検証できなかった項目**である。
- **t-extent の非退化ケース**: reference 有りで `text_length = 414 ==
  indexes[0].max()+1`、かつ prefix は **1293 token** なので **414 < 1293**。
  同じ run の reference なし step は `558 == 558`。これで `text_length` が
  「token 数」ではなく本当に t 座標であることが両側から固定された。
- **reference のコスト**: 1280×720 の source 1 枚で **+883 token / +146.6 MiB**
  （同一 caption の A/B）。
- **VRAM**: peak **17.911 GiB allocated / 18.098 GiB reserved**、
  model resident 17.591 GiB。probe が全 arm に敷く 34.55 GiB ゲートの **52.4%**。

**exercise されていない経路**: `separate_by_reference`（bucketing 無効時は通らない）。
配線は入ったが、この smoke では踏んでいない。

**効果は測っていない**: reference 忠実度やプロンプト追従が改善したかは**何も測定して
いない**。上記はすべて「形状と数値が壊れていないこと」の確認である。

なお `_instruction` caption は `tags` と内容が同一で、本番の auto-select は `tags` を
取るため 3-2 と衝突しない。

---

## 12. Open questions（実装時に決めること）

- **prefix KV をキャッシュするか。** caption ごとに 42 層 × 全 token の K/V は容量が
  大きい。毎 step 計算のコストを実測してから決める。Phase 0 の計測項目。
- ~~**`optimizer_stochastic_rounding` を SenseNova full FT で既定 ON にするか、
  OFF を拒否するか。**~~ **決定済み（`24220b5c`）: どちらでもなく「ルート要件と
  して強制する」。** 既定にも拒否にもできないのは transport が「未指定」を
  表現できないためで、理由と実装は §6.3 (2)。（永続 fp32 master は選択肢に
  含めない。棄却済み。）強制を**ユーザーに知らせる経路**は `training_log`
  チャンネルとして着地した（§13.4 警告ボックス (c)）。
- ~~**`optimizer: adamw` を SenseNova full FT で拒否するか。**~~ **決定済み
  （`601d0271`）: 拒否する。** それどころか allowlist は `("adafactor",)` のみで、
  `adamw` はそこから外れた名前の 1 つとして拒否され、加えて §6.3 の理由が
  メッセージに名指しで入る。**この拒否は新規実装である** — 既存の
  `_attach_stochastic_rounding` は警告しかせず、しかも
  `optimizer_stochastic_rounding` が既に True のときにしか到達しないので、
  出荷既定では何も言わなかった。§6.3 / §6.5。
- ~~**full FT の学習成果物をどの checkpoint format で保存するか。**~~
  **決定済み（`22b22f09`）: 選ばずに設定値として出荷した**
  （`sensenova_full_finetune_save_format`、既定 `mixed`）。§6.4。
  **実 run で実証されたのは `mixed`（gen / und の両向き）と `bf16`（both）**である
  （§13.4 の U-2-2 / U-2-5 実測）— それぞれ 25.129 / 25.129 / 32.682 GiB を保存し、
  本番 reader で 294/294・294/294・588/588 バイト一致で読み戻した。
  **`int8` の実 run 往復は未実施**（合成ツリーのテストのみ）。これは
  **再学習の base になれる唯一の形式**なので、resume の実測も同時に空いている。
- ~~**materialize 時の `weight_dtype` 契約が未決定である。**~~ **決定済み
  （`601d0271`）: bf16 のみ。ロードより前に拒否する。**
  `materialize_int8_decoder_linears` は `trainer.weight_dtype` **へ向けて** dequant
  するので、この設定は base を*記述*するのではなく **base が何になるかを決める**。
  したがって fp16 は「更新を stochastic rounding で運べない base」を実体化し、
  fp32 は **30.2 GiB**（8,103,395,328 要素 × 4 byte）の base を実体化する。
  `assert_full_finetune_contract` が `weight_dtype` / `training_dtype` の
  両方を見て、17.6 GiB のロードの前に拒否する。`use_grad_scaler` も別途拒否される
  （hook が勾配を即解放するので GradScaler の inf/NaN 検査自体が走らない）。
  **旧文が指摘していた「BaseTrainer を直接構成する呼び出しからは fp16 に到達する」
  経路は、これで閉じた。**
- ~~**full FT の `train_text_encoder` 既定を SenseNova で明示的に決めること。**~~
  **決定済み（`601d0271`）: SenseNova に限り既定を gen half（False）にする。
  拒否ではなく arch 別既定である。** `param_defaults.FULL_FINETUNE_TRAIN_TEXT_ENCODER_DEFAULTS_BY_ARCH`
  （`_default: True` / `sensenova: False`）と `resolve_full_finetune_train_text_encoder`
  が解決し、`generate_full_finetune_config` はリテラルの代わりにこれを呼ぶ。
  根拠: **他のどの arch でもこの flag は「別個の、凍結されているモデル」を指すが、
  ここでは denoiser の半分を指す**。汎用既定を継承すると、key を省略しただけで
  パラメータ数と host メモリが 2 倍になる。**両 half を明示的に要求する経路は
  そのまま通る** — 決定は既定の変更であって禁止ではない。
- ~~**decoder 外の gen 側モジュール（`fm_head`、gen ViT、embedder、`*_norm_mot_gen`）を
  trainable に含めるか。**~~ **決定済み（`601d0271`）: 含めない。§6.2 の
  「含める方向を推奨」を上書きする。** 根拠は品質ではなく scope 同一性
  （量子化されていない → loader が materialize しない → collect すると
  adapter の scope が loader の scope と食い違う。これは adapter の存在理由そのもの）。
  **代わりに開いたままにする問い: x0 を出力する head を凍結したまま
  「full fine-tune」と呼べるのか。** 本決定はこれに答えていない。§6.2。
- ~~**ConvRot checkpoint を学習対象に含めるか。**~~ **解決済み（`0c9ea86b`）: 含める。**
  受理条件は 588 Linear が単一 flavour で `Int8Linear` か `ConvRotInt8Linear`（§5.3）。
  代わりに**新しい未測定事項**が残る: **ConvRot での学習は dequant 経路
  （W-int8 / A-bf16）に fit するが、推論は常に fused W8A8 を走らせる**という
  train / inference skew の実害。ConvRot 学習 LoRA を fused 推論カーネル下で A/B した
  例が無い。plain int8 にこの skew は無い（§5.3）。
- ~~**dequant-from-int8 起点の full FT が実測で劣るか。**~~ **「測らない」で close する。**
  構造的な理由であって、優先度の問題ではない: この A/B には比較対象として upstream bf16 が
  要るが、**upstream bf16 が入手できるなら dequant 経路を使う理由が消滅する**。dequant が
  必要なのは upstream が入手できない場合だけで、そのとき比較 arm は存在しない
  （実施可能なとき = 不要、必要なとき = 実施不能）。加えて収束規模の run が 2 本要り、
  本リポジトリは収束実験を行わない。
  **代替（upstream が入手できた場合に限り、1 回だけ）**: `dequant(int8)` と upstream bf16 の
  per-tensor 誤差 census（学習なし、shard streaming、GPU 不要）を記録する。これは
  「劣るか」には答えないが、**焼き付く誤差の規模を事実として残せる**。
- **凍結 und での reference 忠実度が十分か。** §7.2 判断 3 の経験的前提。
  **Phase 3 が DONE になっても、この問いは開いたままである** — 3-4 の smoke は形状と
  数値の健全性だけを確認しており、**忠実度は何も測っていない**（§11 Phase 3 実測）。
  不足した場合のみ `scope: both` を開くが、その受け皿は Phase U（§13）として既に
  設計されている。
- **`separate_by_reference` を SenseNova で実際に踏んだときの挙動。** 配線は Phase 3 で
  入ったが（§7.5 差分 1）、3-4 の smoke は bucketing 無効で走ったため
  **この経路は未 exercise** である。
- **batch > 1 をいつ開くか。** padding-aware gen mask / varlen attention または
  streaming per-sample backward が前提。`separate_by_reference` だけでは開かない。
- **upstream issue #207 の mixed forward を検証・修正して 1 パス化する価値があるか。**
  2 パス設計で十分機能する見込みなので優先度は低いが、und 学習を将来入れるなら
  再評価する。

### 未測定事項の一覧（実測が無いもの／構造から推論しただけのもの）

新規（今回の設計再検証で判明したもの。いずれも**構造からの推論であって実測ではない**）:

1. ~~**mixed checkpoint（und int8 + gen bf16）の推論ロード可否。**~~
   **ロードのレベルで解消（U-2-2 step 3、2026-08-25）。** 実 run が保存した
   25.129 GiB の mixed checkpoint を、**本番 reader `load_sensenova_from_path`**
   が別プロセスで読み、gen 294 個が浮動小数の `nn.Linear`、und 294 個が
   `Int8Linear`、**294/294 の weight が SHA-256 でバイト一致**した。
   **生成そのものは走らせていない**ので、「ロードできる」以上は主張しない。
   なおここで **2 件の直列化欠陥**が見つかっている（§6.4 末尾）— 「構造上は
   通りそう」という推論は、実際には**書いた側が読めない config を書いていた**。
   **【U-2-5 で拡張】** mixed の**逆向き**（und 学習 = gen 半分が int8 のまま）も
   **294/294 バイト一致**で読み戻した。加えて **both branch の `bf16`**
   （量子化テンソルが 1 つも無いツリー）も **588/588 バイト一致**で読める。
   **どちらも生成は走らせていない。** 残るのは `int8` 形式の実 run 往復である。
2. **`LayerOffloadConductor` がサブモジュール粒度のリストを受けられるか。** 未調査。
   受けられなければ half-eviction との合成に wrapper か per-layer 選択の新規実装が要る。
   §8.3.1。
3. **dequant 起点 full FT の学習品質影響。** 上記のとおり**測定不能な構造**である
   （比較 arm が存在しうる状況と、経路が必要になる状況が排他）。

Phase U（§13）関連。

**U-0 / U-1 で解決したもの**（`3d837202`..`327276df`。**構造的推論が実測に置き換わった**）:

4. ~~**非 reentrant checkpoint の closure 捕捉 cache テンソルへの勾配伝播が、この
   vendor 経路で実際に und grad を届けるか。**~~ **解決: 届く。** und LoRA
   **289 個が有限かつ非ゼロ**（§13.5）。同時に**残り 5 個は構造的に到達不能**である
   ことも判明し、`und_gradient_unreachable_paths()` が名前で予測する（§13.3）。
5. ~~**prefix 非 checkpoint 時の und dequant 実体化 ~15.1 GiB。**~~ **解決: 当たって
   いた。** 解析値 **15.093 GiB**（設計値と 3 桁一致）+ 実測の傾き 66 MB/層 →
   全深度外挿 **17.65 GiB**、model resident 込みで 35.2 GiB（§13.2、§13.5）。
6. **推論側 `mot_phase_eviction` × und LoRA wrapper の weight 移動整合。**
   **一部解決。** CPU テストで、evictor が und wrapper の
   `lora_down` / `lora_up` / `original_module` を **und 側に分類**し、4 相を一巡させても
   出力が不変であることを確認した（§13.3）。**残るのは実 H2D / D2H 転送と pinning の
   挙動**で、これは CPU テストでは踏めない。

**残るもの**:

7. **und LoRA が reference 忠実度・プロンプト追従を実際に改善するか。** 未測定。
   **U-0 / U-1 は「勾配が届き、壊れていない」ことしか示していない。**
   提供するのは選択肢であって効果の主張ではない（§13）。
   **U-2-5 も同じ位置にある** — und half の full FT は 3 branch とも実 run で
   通ったが、示したのは「壊れていない」ことだけである。
   **さらに U-2-5 が新たに開けた未測定事項**: 解像度上限（全 run が 64px で、
   both branch は既に gate の 94.5%）、`int8` 形式の実 run 往復とそれに載る resume、
   保存 checkpoint での生成、host RSS peak の再現性（同一 arm で 9.7 GiB 動いた）。
   一覧は §13.4 の「U-2-5 測っていないもの」。
8. **旧ビルドが新形式（gen+und）LoRA を無警告で部分適用するバージョン skew。**
   新ビルド側は `check_lora_application` の metadata 突き合わせで検知できるように
   なったが、**逆方向（旧ビルドが新ファイルを読む）は依然として防げない**（§13.3）。
9. ~~**und forward 2 回のコストと weight 往復コスト**（4 相分割）。~~
   **CLOSED（U-2-4、`071e602b`）。** prefix / step 比 0.098（p50）/ 0.103（mean）、
   分割の限界コスト **+9.3〜+9.7%**、eviction の往復は別勘定で **+69.0%**（p50）。
   実測表は §8.3.2 の「U-2-4 実測」。**旧「未測定」は stale だった**（U-2-5 の点検で発見）。
   残るのは同節末尾の 3 点（bf16 und Linear での比、1024px 以外の解像度、
   pinned 転送の非同期化）である。
10. **勾配ノルムの大小関係。** U-0 と U-1 で順序が逆転したが、**測定条件が異なるので
    どちらも一般的主張にならない**（§13.6 の訂正表）。設計判断の根拠に使わないこと。

Ring Buffer optimizer 関連（§6.5）。**事前登録の gate として U-2-6 で消化する**:

11. ~~**G-RB1 — state 往復 64.8 GB/step（Lion 32.4 GB）が backward に隠れるか。**~~
    **CLOSED（`8c13c493`）: 閾値の上では完全に隠れる。** 閾値は閉形式で書け
    （AdamW 2038 tokens / Lion 1019 tokens @ 80.7 TFLOP/s・26.5 GB/s）、実測と 2% 一致。
    **旧記述「隠れず直列加算になる」は閾値の下でのみ真だった**（§6.5）。
    **SenseNova の想定解像度帯は閾値の下側**である（1024² で 1024 image tokens）。
    残る未解明: **なぜ in-order stream が吸収するのかの機構**。壁時計が
    `max(compute, transfer)` の形になるという事実までが射程で、プロファイラは
    取っていない。**断定しないこと。**
    また **16.2B の step wall そのものは未実測**で、上の投影が乗る土台が無い
    （U-2-4 が測る）。
12. ~~**G-RB2 — pinned host RAM が実行ホストに載るか。**~~ **CLOSED（U-2-6）。**
    単価は実測（AdamW host 2.0 / Lion 1.0 B/param、100% pinned、二重確保無し）。
    **16.2 B への 32.4 GB / ~50 GB は依然として構造上の外挿**である。§6.5 の実測ボックス。
13. ~~**G-RB3 — サイレント CPU-skip の不在。**~~ **CLOSED（U-2-6）。**
    census は fused backward 実測で 32/32、negative control 31/32 を捕捉。
    なお本項が引いていた `adamw8bit_ringbuffer.py:1082` の記述は **`3a7c9560`
    以来 stale** だった（hook 側は当時すでに fail-loud）。U-2-6 が直したのは
    `step()` 側の同型のスキップである。
    ~~**残る未検証**: census と **4 相 eviction の順序**の相互作用。~~
    **CLOSED（U-2-4）。しかも「未検証」ではなく実際に踏んだ欠陥だった** —
    `_update_census.assert_complete()` は MNT ループ内、`should_step_optimizer`
    ブロックより**上流**で呼ばれるので、相 3 を optimizer step 地点に置くと
    **正しい run で census が「und half は 1 つも更新されていない」と報告する**。
    相 3 は backward 直後に走る。実 run（both branch、588 個中 583 個が更新、
    5 個は構造的到達不能）で確認済み。§8.3.2 と §13.4 の U-2-4 実測。

**§6.5 前提事実 1（「CPU state はどの学習経路からも有効化されない」）は
U-2-6 で解消された** — `_ringbuffer_optimizer_kwargs()` が allocator を渡す。
**ただし switch は config channel 限定**（YAML キー。API / UI 面は意図的に無い）
なので、**UI から起動した run では依然 GPU 確保**である。残っているのは
**閾値下の prefetch**（stream と event 同期は実装済み。§6.5 upgrade 項目 3）である。

既存（不変）: half-eviction の有効性（§8.3 の gate）、凍結 und での reference 忠実度
（§7.2）、ConvRot base の train / inference skew（§5.3）。

**ただし half-eviction の依存関係は強まった。** Phase 2b は weights + gradients だけで
32.4 GB を占め、und half 7.55 GiB の退避が唯一の余白であるため、この gate は
「Phase 1 の運用判断」から「**Phase 2b の VRAM 前提**」に格上げされた（§8.3.1、
§11 Phase 2b-0）。

---

## 13. Phase U — understanding branch の学習（U-0 / U-1 / U-2 DONE、U-3 PENDING）

**要求**: understanding branch も微調整の対象にするかどうかを、SDXL の TE / U-Net や
他 arch の TE / DiT と同様に**ユーザーが選択できるようにする**（LoRA / Full-FT の
両方）。位置づけは nice-to-have ではなく機能要求である — 「新規のコンセプトを画像生成
モデルに追加する上で understanding 層の学習は重要」という理由が示されている。
§7.2 判断 3 の「凍結 und で reference 忠実度は足りるか」という未解決の経験的
不確実性とも接続する。

**既定は OFF。** 既定 OFF である限り §5.2 / §7.2 の既存判断とは矛盾しない（§5.2 の
改訂注記を参照）。**提供するのは選択肢であって効果の主張ではない** — und 学習が
実際に忠実度やプロンプト追従を改善するかは未測定である（§12）。

**Phase 2b / Phase 3 には折り込まない。** U-1 の依存は Phase 1 のみで、PENDING の
2b / 3 に混ぜると偽の依存が生まれる。

**実装状況（2026-08-24）**: **U-0（`3d837202`）と U-1（`e811e461` 本体、
`327276df` 実機 exit smoke）は DONE。** U-2（und Full-FT）と U-3（und × reference）は
PENDING。実測は §13.5 / §13.6 に置く。**効果は依然として何も測っていない** —
und LoRA が品質・忠実度・プロンプト追従を改善するかは未測定である（§12）。

### 13.1 微分可能経路は「新規構築」ではなく「解錠」である（U-0 で実証済み）

§5.2 根拠 1 は und 学習を「微分可能な KV パイプラインの構築」と評価したが、
**und LoRA についてはこれが過大評価だった。** Phase 1 が自分のために作った機構が、
偶然 und 勾配互換の構造を備えている。以下はコードから読める構造的事実で、
**U-0 が実 checkpoint 上で実証した**（§13.5。当初この節は「実証はまだ無い」と
書いていた）。

- **prefix forward の cache 書き込みは非破壊。** `DynamicLayer.update` は
  `self.keys = torch.cat([self.keys, key_states], dim=-2)` と**再束縛**する
  （`venv/.../transformers/cache_utils.py:120-121`。docstring は "in-place" と書くが
  実装は cat である）。したがって `no_grad` を外せば K/V は grad_fn を持つ。
- **gen pass の `update_cache=False` fallback も `torch.cat`。**
  `key_states = torch.cat([past_k, key_states], dim=2)`
  （`vendor/modeling_qwen3.py:792-793` 付近の training 経路）。prefix 側に grad_fn が
  あれば勾配はそこへ流れる。
- **gen の checkpoint loop は `use_reentrant=False`**（`ops/sensenova_ops.py:602`）。
  非 reentrant checkpoint は closure 捕捉テンソルへの勾配伝播を公式にサポートする。

真にサブシステム級なのは次の 3 つだけである: **(a) prefix pass 専用の checkpoint
loop**（§13.2）、**(b) 推論側の und target 列挙と適用**（§13.3）、
**(c) MNT 再計算・assert 分割・eviction 配線**（§13.3）。**この 3 分類は正しかった** —
U-1 の実装量はほぼこの 3 つに収まった。

**ただし 4 つ目があった（U-0 が実行して初めて判明）**: `LoRALinearLayer` は
**fp32 の adapter を保持し、周囲の ambient autocast に依存する**。`train_step` は
generation pass 用に autocast を張っているが、`encode_prompt` は張っていなかったため、
**und LoRA の最初の prefix pass が layer 0 で dtype 不一致を起こして落ちた**。
U-1 は prefix を autocast で包むことで解決した（`_build_trainable_prefix`）。
コードを読むだけでは出てこなかった項目であり、U-0 を「実行する」probe にした価値が
ここに出た。

### 13.2 (a) prefix 専用 checkpoint loop が前提実装である理由

**prefix を非 checkpoint で勾配付きに回すと VRAM が破綻する。** und 側 294 個の
`Int8Linear` がそれぞれ backward 用に dequant 済み bf16 weight を autograd に保存し、
layer 1 以降は hidden が `requires_grad` なので **294 個すべてが該当して約 15.1 GiB が
同時実体化する**。text prefix の活性そのものより支配的である。機構は §5.3 の
`warn_quantized_base_without_checkpointing` および `INT8_W8A8_TRAINING_GATE.md` の
G4 実測と同一である。

> **【U-0 実測、2026-08-24】この見積もりは当たっていた（当初は「構造上の見積もりで
> あって実測ではない」と書いていた）。** 294 target の dequant weight 保持は解析値
> **15.093 GiB** で、設計時の 15.1 GiB と**3 桁一致**する。さらに実測の per-layer の
> 傾きが **66 MB/層** の活性を上乗せし、全深度に外挿すると **17.65 GiB**。
> model resident と合わせると **35.2 GiB** で probe の上限を超える。
> GC OFF の arm は設計どおり 8 層で直線的に伸びたところで上限に当たって中断した。
> **したがって prefix checkpoint は「推奨」ではなく前提実装である。**

**gen 側の流儀をそのまま流用できない**理由が 2 つある。

1. **checkpoint の再計算が `DynamicCache.update()` を二重 append する。**
   §4.7 が gen 側で回避した問題そのものが、prefix 側では cache 書き込みという形で
   再来する。
2. **cache への書き込みは checkpoint segment の副作用チャネルである。** 勾配を下流に
   繋ぐには K/V を **checkpoint 関数の明示的な出力**として返す必要があるが、
   `forward_und` は attn_output しか返さない（attention 版は
   `return attn_output, attn_weights`、decoder-layer 版は `hidden_states` のみ。
   `vendor/modeling_qwen3.py:591`, `:1155` 付近）。vendor に **opt-in の `return_kv`
   seam** を足す設計になる（vendor 編集の前例は style tripwire に既存）。

**parity gate（必須）**: no-grad モードで、学習 prefix loop と vendor
`_t2i_prefix_forward` の cache K/V が **bitwise 一致**すること。

**実装（`3d837202`）**: 設計どおり vendor に `return_kv` を足した。**keyword-only で
既定 OFF** なので既存の呼び出し側は 2-tuple のまま変わらない。attention 版が
`(attn_output, attn_weights, key_states, value_states)`、decoder-layer 版が
`(hidden_states, key_states, value_states)` を返す。**gen branch では
`NotImplementedError` を送出する** — gen は既存の prefix を読む側で prefix を生成
しないので、推測で通すより落とす方が正しい。学習側の prefix cache は
`_TrainingPrefixLayer` として **checkpoint の「出力」から**組み立て、cache 書き込みは
経由しない。**parity gate は PASS**（§13.5）。

### 13.3 (b)(c) 推論側と学習側の配線

#### 推論側は und キーを無警告で捨てる（実害あり）

`normalise_lora_state_dict` は und キーも**グルーピングする** — `_parse_key` は
「将来 target が広がったときに黙って落とさないため」意図的に任意の module path を
受ける（`sensenova_lora.py:88-92, 96-103`）。ところが `apply_lora_group` は
**gen のみを列挙する `iter_sensenova_lora_targets` を回して `grouped.get(module_path)`
で lookup する**（`:242-243`）ため、und エントリは一度も参照されない。
**applied カウントが減るだけでエラーは出ない。** 対応せずに und LoRA を出荷すると
「学習はできるが黙って部分適用される LoRA」を生産する。

必要な対応（**すべて `e811e461` で実装済み**）:

- `iter_sensenova_lora_targets` に `branch: "gen" | "und" | "both"` を追加する。
  **学習と推論が同じ列挙器を使うこと**（新しい列挙器を作らない。これは §5.4 で
  記録済みの「リゾルバを増やさない」規律と同型）。
  → **DONE。** 列挙器は「これが唯一の target 列挙器である」と docstring で宣言し、
  学習 adapter・推論の apply / restore・U-0 probe がすべてこれを駆動する。
  **probe が持っていた私的コピーは削除した**（drift させないため）。
- **applied カウントの検証**（grouped の件数と applied の件数の突き合わせ）。
  → **DONE。** `check_lora_application()` が 2 つの独立した検査を行う: ファイルが
  持つ全 module が live module に到達したか、および metadata の `lora_targets` が
  宣言する scope と module 数が一致するか（後者が「gen+und の checkpoint を
  gen しか知らないビルドが読んだ」場合を捕まえる）。不足は
  `add_warning(code="sensenova_lora_partially_applied")` で生成レスポンスにも出る。
  **`apply_lora_group` の既定は `branch="both"` になった。**
- **format sniff の硬化**: `load_lora_safetensors` の `looks_like_keys` は
  `"mot_gen" in k` を見るので、gen+und ファイルは通るが und のみのファイルは
  `unknown` に落ちる。→ `understanding-only` は恒久非提供なので実害は無い（下記）。
- `smoke.py` に und ケースを追加。→ **DONE**（`sensenova_und_lora_smoke_test.py`）。
- **推論側 `mot_phase_eviction` × und LoRA wrapper の weight 移動整合を検証するか、
  拒否する。** 未検証のまま出荷すると eviction ON の生成で device mismatch になる。
  → **CPU テストで検証済み、ただし限界あり。** evictor は LoRA 適用**後**に構築され、
  分類は module path で行う。und wrapper の `lora_down` / `lora_up` / `original_module`
  は `_mot_gen` を含まないので **und 側に分類され、実際にそれを呼ぶ half と一緒に
  動く**（prefix では常駐、denoise では CPU へ退避 — denoise で und branch は
  そもそも到達しない）。テストは 4 相を一巡させて出力が不変であることまで確認する。
  **カバーできていないのは実 H2D / D2H 転送と pinning の挙動**であり、この部分は
  §12 に残す。

#### `_assert_immutable_prefix_cache` は分離する（外すのではない）

現状この関数は**構造検証**（layer 数 / 非空 K/V / streamer 不在 / flash buffer 不在）と
**`requires_grad` 拒否**が同居している（`ops/sensenova_ops.py:531-558`。拒否は
`:552-553`）。

- 構造検証は**無条件で維持**する。
- `requires_grad` 拒否だけをフラグで分岐する。
- **単に外すのではなく、und 学習時は逆向きの positive assertion**（全 42 層の K/V が
  `grad_fn` を持つこと）を入れる。**これが無いと「und LoRA を選んだのに prefix が
  `no_grad` で作られ、loss は正常に下がるが und は 1mm も学習されない」という
  §6.1 と同型のサイレント故障**を再生産する。

**実装（`e811e461`）**: 設計どおり 3 つに分けた —
`_assert_prefix_cache_structure`（無条件）、`_assert_prefix_cache_detached`（凍結時）、
`_assert_prefix_cache_differentiable`（学習時、positive）。後者の例外文は
「loss would fall normally and the understanding LoRA would never be trained」と
故障の形まで書いてある。**実機で実際に発火していることも確認済み** — 3-step arm で
9 回、MNT arm で 12 回呼ばれ、census は毎回 `(42, 42, 42)` だった（§13.6）。

#### その他の配線

- **MNT > 1 では `retain_graph` を使わない。** MNT ループは iteration ごとに
  `optimizer.step()` を打つので、graph 再利用は version counter 衝突か旧パラメータ
  勾配になる。**per-iteration で prefix を再計算する。** 前例は
  `need_recompute_text_embeddings`（TE trainable かつ MNT>1 なら re-encode。
  `base_trainer.py:11086-11090`）。
  → **DONE。** `_sensenova_mnt_conditioning` が `mnt_index` と captions を受け取り、
  `train_text_encoder` が真かつ `mnt_index > 0` のときだけ prefix を再構築する
  （凍結時は従来どおり同じ detached prefix を使い回す）。実機で
  **2 batch × 2 MNT = 4 回の prefix build、freed-graph エラー無し**を確認（§13.6）。
- **dropout guard**: `attention_dropout` の既定は 0.0（upstream `Qwen3Config`）だが、
  `dropout=0.0 if not self.training else self.attention_dropout`
  （`vendor/modeling_qwen3.py:591`）の分岐は `transformer.train()` が stamp される
  学習経路で**生きている**。**und 学習経路の有効化時に `attention_dropout != 0` を
  fail-closed で拒否する**（将来の非ゼロ config で再計算が確率的になるのを黙って
  通さないため）。
  → **DONE**（`assert_understanding_training_supported`）。理由も
  「checkpoint された prefix の**再計算**が確率的になり、recompute した K/V が
  forward の K/V と静かに食い違う」と明記されている。既定 0.0 なので今日存在する
  構成は何も拒否しない。
  - **【`24220b5c` で判明・修正】guard は存在したが、その呼び出しが
    und / both branch に限定されていた。** full FT の既定 branch は
    **gen-only** なので、そのままでは素通りしていた。欠陥になる連鎖はこうである:
    (1) und と gen は**同一の decoder** — 1 個の `Qwen3Attention` が
    `q_proj` と `q_proj_mot_gen` を並べて持つ（`vendor/modeling_qwen3.py:455-458`）。
    (2) `load_components` は decoder 全体に `train()` を stamp する
    （`ops/sensenova_ops.py:511`。branch も method も見ない）。
    (3) prompt prefix は **und half が毎 step 構築する**（`encode_prompt` →
    `_t2i_prefix_forward` → `language_model.model`）。これは `no_grad` の下だが、
    **`no_grad` は dropout を止めない**（`dropout=0.0 if not self.training else
    self.attention_dropout`、`:591`）。
    したがって非ゼロの `attention_dropout` は、**loss を計算する相手である
    conditioning 自体を、毎 step 別々に、かつ推論とも別に、無言でランダム化する**。
    → **`assert_full_finetune_dropout_free` を新設し、full FT では branch に
    よらず無条件に呼ぶ**（`sensenova_adapter.py:301`）。
  - **【既知・非 live】LoRA 側に同型の穴が残っている（意図的）。**
    `train()` の stamp は LoRA でも同じ 1 行から来る（上記 (2)）。
    `SenseNovaLoRAAdapter` が und guard を呼ぶのは
    **`train_text_encoder` が真のときだけ**（`sensenova_adapter.py:131-133`）で、
    `encode_prompt` 側の呼び出しも `requires_grad=True`（= und LoRA）に限られる
    （`sensenova_ops.py:737-738`）。つまり **gen-only LoRA（既定）は
    どちらの guard も通らない**。**触っていないのは LoRA の挙動を変えないため**である。
    **live ではない**根拠: upstream `Qwen3Config` の `attention_dropout` 既定は
    **0.0** で、実 checkpoint も一致する — `M:/model/sensenova/config.json` の
    `llm_config.attention_dropout = 0.0`、`vision_config.attention_dropout = 0.0`。
    非ゼロ config を持ち込む日が来たら、ここが先に読まれるように記録しておく。
- **eviction との併用は contract で拒否する（自動無効化ではない）。**
  `train_text_encoder` と `sensenova_mot_phase_eviction` の同時指定は
  `train_runner` が `ValueError` にする。**どちらも opt-in なので、片方を黙って
  落とすとユーザーが設定した契約（VRAM 予算か、学習対象か）を破る**からである。
  メッセージは「**これはこの実装形のスコープ制限であって原理的な非互換ではない**」と
  明言し、4 相分割（§8.3.2、U-2-4）を参照先として指す。

#### und 側 target の命名

und attn は `self_attn.{q,k,v,o}_proj`（サフィックス無し）、und MLP は
`mlp.{gate,up,down}_proj`。**42 層 × 7 = 294** で、gen 側の 294 と合わせて 588。
なお **gen 側の命名は非対称**（attn は Linear 名 `q_proj_mot_gen` を `self_attn` の
下に、MLP は親名 `mlp_mot_gen` を使う。`sensenova_lora.py:154-155`, `:203-215`）だが、
**und 側は attn / MLP ともに素の名前**であり、非対称なのは gen 側だけである。
実装ではこの差を `_BRANCH_LAYOUT`（branch → attn 属性名・MLP 親名・MLP 属性名）に
畳んであり、列挙ループ自体は 1 本である。

#### und は 289 ではなく 294 を維持する

U-0 が「5 個には構造的に勾配が届かない」ことを実測したが（§13.5）、**列挙は 294 の
まま**にした。理由は gen 側との対称性で、checkpoint は 588 × 3 = **1764 tensors** に
なる。届かない 5 個は `lora_up` のゼロ初期化のまま残り、推論時に何も寄与しない。
その代わり **`und_gradient_unreachable_paths()` が 5 個を名前で予測する**ので、
census は「294 個すべてが動いた」ではなく「**294 個中 289 個が到達、残り 5 個は
この名前**」と書ける。「全部動いたはず」と書いた assert は**まさにこの 5 個で落ちる**
のが正しい挙動である。

#### パラメータ・LR・checkpoint

- **パラメータは `train_text_encoder` を流用する**（新設しない）。LR は
  `text_encoder_1_lr`、fallback は `text_encoder_lr` → `unet_lr` の連鎖で、
  SDXL LoRA と同じ（`sdxl_adapter.py:387`）。und が vision も兼ねる件は param ではなく
  **capability reason と UI 説明文**で扱う。
  → **DONE。** adapter は und half を `LORA_COMPONENT_TEXT_ENCODER_1` として登録する
  （`unet` ではなく）ので、LR チェーンにも grad-norm の集計単位にも自然に乗る。
  実機の optimizer group は `[{lr: 1e-4, 588 params}, {lr: 5e-5, 588 params}]` で、
  **und が実際に `text_encoder_1_lr` チェーンを通っている証拠**になっている（§13.6）。
- **checkpoint は gen + und を 1 ファイル。** `neo_hf_lora` の verbatim path 形式は
  und パスをそのまま収容できる。metadata は
  `lora_targets: "generation+understanding"`、tensor 数は **1764**
  （588 target × down/up/alpha。Phase 1 の 882 / 294 と同じ比）。
  **既存の gen-only 蒸留 LoRA の互換は無変更で保たれる** — 適用は lookup 駆動なので
  und スロットが空振りするだけである。
- **バージョン skew を記録する**: 新形式を旧ビルドでロードすると und キーが無警告で
  落ちる。新ビルドは metadata と突き合わせて検知できるが、**逆方向は防げない**（§12）。
- `understanding-only` LoRA は**恒久的に提供しない**（保存側で gen 0 件を拒否する）。

#### capability 宣言

`TRAINING_FEATURE_UNSUPPORTED["sensenova"]["text_encoder_training"]` は**削除ではなく
スコープ化**する: `methods=["full_finetune"]`。これは zimage の逆向きの同型である
（zimage は `methods=["lora", "relora"]` で「LoRA では TE を学習しないが full FT は
する」を宣言している）。U-2 が着地したらこの entry 自体を落とす。
→ **DONE**（`e811e461`）。コメントに「SenseNova is Z-Image's mirror image」と書いて
両者を相互参照させてある。

### 13.4 フェーズ分割と exit criteria

- **U-0 — 前提 probe（DONE、`3d837202`）。** exit: (1) 学習 prefix loop と vendor
  `_t2i_prefix_forward` の K/V が **bitwise 一致**（no-grad モード）、
  (2) 588 LoRA の grad が finite かつ **und 294 が nonzero**（= §13.1 の closure
  勾配伝播の実証。ここが本フェーズの中心）、(3) prefix GC ON / OFF の loss・grad
  parity、(4) **GC OFF arm で ~15.1 GiB の dequant 実体化を数値として観測**
  （§13.2 の見積もりの検証）。→ **全項目 PASS。ただし (2) は 294 ではなく 289 で、
  残り 5 個は構造的に到達不能であることが判明した**（§13.5）。
- **U-1 — und LoRA 本体（text-only）（DONE、`e811e461` + `327276df`）。** exit:
  3-step smoke で有限 loss、1764 tensors、**fresh runtime で 588 適用・588 復元、
  strength 0 が base と `torch.equal`、strength 1 が base と異なること**（後者が
  無いと §13.3 のサイレント部分適用を検出できない）、`train_text_encoder=false` の
  loss・grad SHA-256 が現行実装と一致すること（**回帰していないことの証明**）、
  MNT>1 smoke、既存の蒸留 LoRA の推論ロード回帰。→ **5 項目すべて PASS**（§13.6）。
- **U-2 — und Full-FT（DONE。ただし 2b-4 = offload 合成は別途 PENDING）。**
  U-2-1 〜 U-2-6 のうち **U-2-1 / U-2-2 / U-2-3 / U-2-4 / U-2-5 / U-2-6 が着地**し、
  gen / und / both の 3 branch すべてに実 checkpoint 上の run がある。
  **§8.3.1 の offload 合成（§11 の 2b-4）だけが残る**。
  U-2-1: §6.4 経路 (a) の 588 版 → **DONE（`cc296e84`）。
  ただし経路は端から端まで到達しない**（`arch_capabilities.py:812-814` と
  `train_runner.py:166-167` が拒否を維持している）ので、**テストで証明されただけで
  run では証明されていない**。着地したもの／意図的に着地させなかったもの／
  実 checkpoint ヘッダから取った host RAM の実測値は §6.4。
  U-2-2: adapter + fused backward の Block Swap からの
  **decoupling**（「gate 解錠」ではない。§6.2 の訂正）+ EMA 拒否 +
  **effective batch = 物理 batch = 1 を受け入れる**契約（§6.2 改訂の条件 1-4）。
  **許可 optimizer は `("adafactor",)` のみ**（§6.5 末尾の訂正。Ring Buffer 系を
  併記していた旧文は撤回した）。
  → **step 1-2 は DONE（`601d0271`）、step 3 は意図的に未着地。** 下記
  「U-2-2 の着地状況」。
  U-2-3: stochastic rounding（§6.3）+ dropout guard。→ **DONE（`24220b5c`）。
  ただし「既定 True」ではなく「ルート要件（強制）」として着地した**（transport が
  「未指定」を表現できないため。§6.3 (2)）。dropout guard は full FT で無条件に
  なった（§13.3）。
  U-2-4: §8.3.2 の 4 相分割（exit gate に **prefix / step 比の実測**を含む）。
  → **DONE。exit gate は PASS、ただし ~19-21 GB の見積もりは着地しなかった**（下記）。
  U-2-5: exit smoke — **DONE（2026-08-25。本節末「U-2-5 実測」）。**
  gen / und / both の 3 branch すべてに実 run が付き、`mixed`（gen 側 int8 残し・
  und 側 int8 残しの**両向き**）と `bf16` の再ロードが本番 reader で通った。
  **update-nonzero census** で bf16 丸め欠陥の
  「動かないのに loss は下がる」故障モード（§6.3）を捕まえる。**品質は主張しない。**
  **期待値は「全部動いた」ではない** — und half の 5 個は materialize されて実
  `nn.Parameter` を持つが（§6.4）、t2i の loss が構造的に届かない（§13.5）。したがって
  census は **und branch で 294 個中 289 個、両 half で 588 個中 583 個**が動き、
  **残り 5 個は `und_gradient_unreachable_paths()` が返す名前**（layer 41 の
  `self_attn.q_proj` / `self_attn.o_proj` と `mlp.{gate,up,down}_proj`）である、と書く。
  これは U-0 / U-1 が LoRA で記録したものと**同じ 5 個・同じ数え方**である。
  「294 個すべてが動いた」と主張する assert は**まさにこの 5 個で落ちるのが正しい**。
  - **【U-2-2 step 3 実測】gen branch は 294 個中 294 個が動いた。** 5 個は
    und 側（`layers.41.self_attn.q_proj` / `.o_proj` / `mlp.{gate,up,down}_proj`）で、
    gen の enumeration は `*_mot_gen` / `mlp_mot_gen.*` なので**集合として交わらない** —
    したがって gen だけの census に減算は入らない。**残っているのは und / both の
    census**（289 / 583）で、それが U-2-5 の本体である。
    なお census の機構自体はここで一度壊れているのが見つかっている
    （adafactor が `record_param_update` を呼んでいなかった。§13.4）。
  - **【census の arm 方法】** `optimizer_update_census` は
    **run の train_config（YAML）のキー**として読まれる
    （`BaseTrainer.__init__`。`use_ema` / `gradient_checkpointing` と同じ channel で、
    `train_runner` の 4 箇所すべてが `train_config=` を渡している）。
    以前は `__init__` でリテラル `False` に固定されており、**config からは
    一切 arm できなかった** — U-2-5 の acceptance criterion が、trainer を手で
    構築する以外の方法では使えない状態だった。
    **API / UI 面は意図的に張っていない**: census は不足を検出すると `raise` するので
    checkbox にすると false positive が正しい run を落とすし、結果は stdout の
    census で UI に出す先が無い。`optimizer_state_host_resident`（§6.5）も
    同じ config channel・同じ理由で同じ扱いにした。
    テスト: `backend/tests/optimizer_diagnostic_switch_config_test.py`。
  U-2-6: **Ring Buffer optimizer の upgrade 3 項目**（§6.5。`get_state_buffer` の配線 /
  閾値下向けの専用 stream + prefetch / サイレント CPU-skip の fail-loud 化。
  **state shuttle の移植は実測により不要と判明したので落とした**）。
  **依存は U-2-2、exit は G-RB2 / G-RB3**（G-RB1 は `8c13c493` で CLOSED）。
  これが通るまで Ring Buffer 系は opt-in を開かない。
  - **⚠️ U-2-6 の罠（`24220b5c` が仕掛けた assertion に当たる）。**
    Ring Buffer 系は **`step_param` を定義しない** — 自前で
    post-accumulate-grad hook を登録し、`base_trainer._setup_fused_backward_pass`
    の両 branch 末尾で **early return する**。
    一方 `assert_full_finetune_stochastic_rounding_attached` は
    `step_param` の存在と被覆を検査する。したがって
    **`_NATIVE_STOCHASTIC_ROUNDING_OPTIMIZERS` への所属を「被覆済み」として
    扱わないまま `SENSENOVA_FULL_FINETUNE_OPTIMIZERS` に追加すると、
    正しい構成で assertion が raise する。** しかも両者は
    **stochastic rounding を自前の更新の中でネイティブに実装している唯一の
    optimizer**なので、この assertion は**最もよく被覆された構成を、
    両半分とも彼らには当てはまらないメッセージで拒否する**ことになる。
  - **【`0d843213` で確定】U-2-2 の「effective batch = 1」は選択肢ではなく前提である。**
    fused backward と gradient accumulation は**原理的に両立しない** — accumulation は
    window 終端で全パラメータの総和勾配が同時に存在することを要求し、それは
    fused backward が回避している常駐そのものだからである。実測: 定数勾配・AdamW・
    `accum=4` で、fused は非 fused の **3.88 倍の距離**を移動し、「4 回の独立した step」と
    厳密一致した。**素の SGD では両者が完全一致する**ので、この乖離は harness ではなく
    **optimizer の非線形性による本物の差**である（線形な更新則なら和は一致する）。
    したがって U-2 は **effective batch = 物理 batch = 1 を受け入れる**か、
    別機構（pinned host への accumulator 退避など。**コスト付きの新機構であって
    修正ではない**）を作るかの二択になる。現状は**拒否ではなく警告**として出荷済み。
  - **【`cc296e84` 以後】U-2-2 は「静かな部分学習」の形をひとつ引き受けた。**
    `load_components` は SenseNova に text encoder が無いため
    `trainer.text_encoder = None` を置く（`sensenova_ops.py:281`）。一方 full FT の
    汎用 collector は **`trainer.train_text_encoder and trainer.text_encoder is not None`**
    で分岐する（`sd15_adapter.py:327`、`:363`、`:418`）。この 2 つを突き合わせないまま
    `train_text_encoder=True` で走らせると、**7.5 GiB を払って materialize した und half が
    1 パラメータも収集されない**。したがって U-2-2 は次の 3 点を明示的に処理すること。
    1. `full_parameter_trainer._create_adapter` に **`elif self.is_sensenova:` 分岐を
       `else: SD15FullParameterAdapter` の fallthrough（`full_parameter_trainer.py:202-204`）
       より上に**追加する。無いと上記の静かな 0 件収集になる
       （`self.is_sensenova` は `base_trainer` が既に立てている）。
    2. und half の materialize 済み weight は **`trainer.transformer` から**集め、
       **`trainer.text_encoder` は決して読まない**。LR は
       `text_encoder_1_lr` → `text_encoder_lr` → `unet_lr` の連鎖に載せる。これは
       Phase 1 LoRA adapter が und を `LORA_COMPONENT_TEXT_ENCODER_1` として登録し
       （`sensenova_adapter.py:134`）同じ連鎖を引いている（`:162-166`）のと同型である。
    3. `TRAINING_UNSUPPORTED["sensenova"]["full_finetune"]`（`arch_capabilities.py:812-814`）と
       `network.type != "lora"` 拒否（`train_runner.py:166-167`）の**両方**を落とす。
       片方だけでは到達しない。
  - **【`601d0271`】U-2-2 の着地状況 — 上記 3 点のうち 1 と 2 が DONE、3 は未着地。**

    **着地したもの**:
    - **`SenseNovaFullParameterAdapter`**（`adapters/sensenova_adapter.py`、
      `adapters/__init__.py` に export、`full_parameter_trainer._create_adapter` の
      `elif self.is_sensenova:` は SD1.5 fallthrough より**上**）。
      `trainer.transformer` だけを読み、`trainer.text_encoder` は読まない。
      scope は loader と**共有**（`resolve_full_finetune_branch` +
      `iter_sensenova_lora_targets`）なので、materialize した集合と最適化する集合は
      「一致する 2 つ」ではなく **1 つ**である。target がまだ buffer を持っていれば
      静かに凍結せず **raise** する。LR は
      `text_encoder_1_lr` → `text_encoder_lr` → `unet_lr` の連鎖。
    - **fused backward の Block Swap からの decoupling**（§6.2 の訂正）。
      `blocks_to_swap == 0` のまま設置し、設置に失敗したら raise する。
    - **`assert_full_finetune_contract`**（`ops/sensenova_ops.py`）。
      **run 中 2 回**呼ばれる — `load_components` から config を読んで
      17.6 GiB のロード前に、`setup_optimizer` から**引数として渡された
      optimizer 名**で（config 値と食い違いうるので、そちらが権威）。
      拒否するもの: 非 bf16 の `weight_dtype` / `training_dtype`、`use_grad_scaler`、
      `use_ema`、`num_optimizer_groups != 0`、`gradient_accumulation_steps != 1`、
      allowlist（`("adafactor",)`）外の optimizer。**config channel と属性 channel の
      両方**を見る option は両方で見る。

    **【step 3 着地】受付は開いた。** 下記「⚠️ 順序制約」の 3 前提はすべて
    discharge され、その上で `TRAINING_UNSUPPORTED["sensenova"]["full_finetune"]`
    と `train_runner` の受付拒否を**両方**落とした。実測は本節末の
    「U-2-2 step 3 実測」。

    > ### ⚠️ step 3 を開ける前の順序制約（好みではなく前提条件）— **3 前提すべて discharge、gate は OPEN**
    >
    > **(a) U-2-3（stochastic rounding）— DONE（`24220b5c`）。**
    > `601d0271` 時点では、契約が**許可する**構成（Adafactor + 出荷既定
    > `optimizer_stochastic_rounding: False`）が、同じ契約の `adamw` 拒否
    > メッセージが**名指しで挙げている欠陥**をそのまま再生産していた
    > （bf16 要素の 84.5% が step 数によらず一度も動かず、loss は正常に下がる。
    > §6.3）。現在はこのルートで強制 ON になっている。
    >
    > **(b) §6.4 の checkpoint format の決定 — DONE（`22b22f09`）。** 3 候補は捨てずに
    > `sensenova_full_finetune_save_format`（既定 `mixed`）として出荷した。
    > gen / und / both × 3 形式すべてが有効で、無効な組み合わせは無い
    > （both × mixed のみ bf16 へ縮退し、それを告知する）。§6.4 末尾。
    >
    > **(c) ユーザーに届く通知経路 — DONE（`339790b5`）。** 以前はこの強制が
    > **trainer の stdout にしか出ず**、`routes.py` の `log_callback` が
    > **バックエンドサーバー自身のコンソール**に `print()` するだけだった。
    > 現在は `core/training/training_events.py` 経由で発行され、
    > `TrainingProcess` が stdout から拾い上げて `training_log` WebSocket
    > メッセージとして broadcast し、`training_runs.warnings` に永続化する
    > （切断中・再読み込み・完了後も Training Monitor で見える。
    > `backend/api/WS_PROTOCOL.md` の `training_log` 節）。
    > **checkpoint metadata に有効値を残す仕組みは依然として無い**ので、
    > 三値 transport（下記）は独立した未実施項目のまま。
    > なお三値化は研究ではなく `docs/guides/ADD_A_PARAMETER.md` の通常作業である:
    > `routes.py:15145` を `Optional[bool] = None`、`openapi.yaml:18246` の
    > スキーマに `nullable: true`、`training_config.py:142` を無条件書き込みに、
    > 加えて frontend が「未指定」を送れるようにする変更。
    >
    > **この順序制約は満たされた。以降、この box は履歴である。**

    #### 【step 3】開けた 2 つの gate と、同時に足した契約分岐

    - **`arch_capabilities.py`**: `TRAINING_UNSUPPORTED["sensenova"]["full_finetune"]`
      のエントリを**削除**（`relora` / `controlnet` はそのまま）。UI の method
      dropdown はこの表を filter しているので、frontend の変更は不要である。
    - **`train_runner._apply_sensenova_training_contract`**: `network.type != "lora"`
      の拒否を **allow-list（`lora` / `full_finetune`）**に置き換えた。
      **単に削除してはならない** — この 1 行は SenseNova ControlNet 学習を拒否して
      いる**唯一の場所**でもある（`controlnet` には capability 表のエントリは
      あるが、ReLoRA の `_refuse_unsupported_relora` に相当する trainer 側の
      guard が無い）。したがって他の method は**名指しで**拒否する。
    - **full FT 用の契約分岐 `_apply_sensenova_full_finetune_contract`**（新規）。
      **重複ではなく前倒しである** — 各節は trainer 側でも再チェックされるが、
      ここは torch も checkpoint も読む前で、メッセージが run failure として
      ユーザーに届く。内容は §6.2 / §6.4 の予算条件のうち config だけで決まるもの:
      `gradient_accumulation_steps == 1`（§6.2 条件 2）、`num_optimizer_groups == 0`、
      `use_ema` 拒否（§6.2 条件 4）、optimizer allowlist（§6.5 末尾）、および
      **`sensenova_full_finetune_save_format` の妥当性**（§6.4。adapter は save 時に
      しか解決しないので、既定 `save_every=100` 下では不正値が「数時間走ってから
      落ちる run」になる。API は `Literal` で縛っているので手書き YAML 専用の経路）。
      `batch_size` の拒否メッセージからは **accumulation の勧めを外した**
      （full FT ではそれ自体が拒否されるため。`base_trainer.py:8512` と同型）。
      **`weight_dtype` / `training_dtype` はここでは見ない** — full FT の dispatch が
      `_is_bf16_native_base_model` 経由で両方を bf16 に強制するので、config 値は
      trainer が見る値ではない。

    #### 【step 3】受付経路の点検と監査で見つかった 7 件（全件修正済み）

    1. **`train_unet` が `FullParameterTrainer` に渡っていなかった**（修正済み）。
       `train_runner` の full_finetune 分岐は `train_text_encoder` /
       `train_image_encoder` は config から読むのに `train_unet` は読まず、
       constructor の既定 `True` が常に勝っていた。**arch 非依存の欠陥**だが
       （どの arch でも TE-only full FT が製品から到達不能だった）、SenseNova では
       意味が重い: `train_unet=False` + `train_text_encoder=True` は「und half だけ」
       の要求なのに **`both` になり、dequant する Linear が 294 → 588、host RAM が
       7.59 → 15.14 GiB になる**。`train_config.get('train_unet', True)` を読んで
       渡すよう修正した。
    2. **updated-parameter census が adafactor で機能していなかった**（修正済み。
       **smoke run の 1 step 目が捕まえた**）。`setup_update_census` は
       fused-backward の全 optimizer に対して census を arm するが、
       `record_param_update` を呼んでいたのは **ring buffer 系 2 つだけ**で、
       `adafactor_fused` / `adamw8bit_fused` は呼んでいなかった。結果、
       **このルートが唯一許可する optimizer で、正しい run が「294 個中 294 個が
       更新されていない」と報告された**。U-2-5 の acceptance criterion そのものが
       使えない状態だったことになる。両 `step_param` の末尾（勾配なしの early
       return の**後**）で記録するようにした。
    3. **保存した checkpoint が読み戻せなかった**（修正済み。**smoke run の
       reload arm が捕まえた**）。詳細は §6.4 の「保存した config が読み戻せない」節。
    4. **【監査で格上げ】既定の save format で作った run は、既定の設定のまま
       resume できない。** 旧文はこれを「SenseNova の resume 自体は整合している」と
       書いていた。**過小評価である** — 整合してはいるが、**帰結は「既定同士の
       組み合わせが行き止まりになる」**である。
       - `TRAINING_DEFAULTS["resume_from_checkpoint"] = "latest"`
         （`param_defaults.py:2226`）で、`_build_train_section` は
         **無条件に書き込む**（`training_config.py:417`）。
       - 再起動すると `base_trainer.py:1352-1385` が最新の entry を選び、
         `:2084-2089` が `model_path` をそれに差し替えて
         `sensenova_ops.load_components` を通す。
       - `_assert_supported_quantized_training_base` は 588 個すべてが単一の
         量子化 flavour であることを要求するので、**既定の `mixed`
         （294 `nn.Linear` + 294 `Int8Linear`）も `bf16` も落ちる**。
       - **fail-loud なので安全側**であり、静かに base から学習し直すことはない。
         しかし**再学習の base になるのは `int8` だけ**で、それは openapi が
         lossy と明記している形式である。**resume したい run は作成時に
         `sensenova_full_finetune_save_format='int8'` を選んでおく必要があり、
         後から変更はできない**（他 2 形式の重みは既に loader が base として
         受理しない形で書かれている）。
       - **コード側の対応（実施済み）**: 拒否されたツリーが**このリポジトリ自身の
         writer が書いたもの**だったとき、guard が
         `sensenova_full_finetune_save_format` を**名指しする**ようにした
         （`_own_save_format_remedy`）。実効 format / 要求 format / branch は
         その checkpoint の metadata に書いてあるので、「plain-int8 の checkpoint を
         選べ」という指示と「作成時に自分が選んだ設定」を利用者が自力で
         結びつける必要はもう無い。そのために `load_sensenova_from_path` は
         metadata を返すようになった（追加のみ）。
    5. **`FullParameterTrainer.load_checkpoint` は存在しないモジュールを import する**
       （**修正済み（拒否として着地）**、arch 非依存）。`from core.models.checkpoint_utils
       import load_unified_checkpoint` — `checkpoint_utils.py` はリポジトリのどこにも無い。
       **どの arch もこのメソッドを通らない**（旧文は SenseNova だけが通らないと
       読める書き方だった。**過小評価である**）: `load_checkpoint` は
       `BaseTrainer` の abstractmethod だが**呼び出し側が 1 つも無く**、
       full FT の resume は全 arch とも `resume_from_checkpoint` →
       `BaseTrainer.__init__` → `_load_checkpoint_as_base`（checkpoint を
       base model として読み直す）である。自分の `load_checkpoint` を呼ぶ trainer は
       `ControlNetTrainer` と `VaeTrainer` で、いずれも別クラス・別実装。
       - **対応**: 削除は不可（abstractmethod なのでクラスが instantiate できなくなる）、
         実装は 11 arch 分の full-FT 保存形式の reader を消費者ゼロで発明することに
         なるので、**実 resume 経路を名指しする `NotImplementedError` に置き換えた**。
         同時に消したもう 1 つの枝は diffusers ディレクトリ形式の loader で、
         **本リポジトリのどの full-parameter adapter も書かない**レイアウト向けだった。
         テスト: `backend/tests/full_parameter_resume_path_test.py`。
    6. **grad-norm の bucket 分けは und half を `unet` として数える**（**修正済み**）。
       `_calculate_grad_norms` の full-FT 側は parameter を「walk した module」で
       bucket に入れる。1 module = 1 component の arch では正しいが、SenseNova は
       両 MoT half が `transformer_original` の中にあるので、`both` / `und` branch で
       `MoT-Understanding` の grad norm が独立して出なかった。`_build_component_lr_list`
       は正しく 2 group を返すので **LR 側にずれは無かった**（学習は正しく、表示だけの
       問題だった）。
       - **対応**: LoRA 側と同じ権威に揃えた — **optimizer group を作った adapter が
         自分の parameter を分類する**（`BaseFullParameterAdapter.grad_norm_components()`、
         既定は `{}`）。SenseNova の実装は `iter_sensenova_lora_targets` で駆動し、
         und half を `LORA_COMPONENT_TEXT_ENCODER_1` に入れる（Phase 1 LoRA が
         `sensenova_adapter.py:134` で登録しているのと同じ component）。
         **module path の名前判定は使わない**（`dd0b10c7` が 4 arch で消した形）。
         override しない adapter は `{}` を返すので他 arch の挙動は不変。
         テスト: `backend/tests/sensenova_full_finetune_grad_norm_test.py`。
    7. **`text_encoder_training` の capability 理由が虚偽になった**（修正済み）。
       「full fine-tuning is refused for this architecture as a whole」と書いて
       いたが、それは step 3 で偽になった。**エントリ自体は残した** — 機構は
       あり trainer も拒否しないが、**und half の full FT には実測 run がまだ無い**
       （U-2-5）ので、理由をその事実に置き換えて UI では gen half のみを出す。
       これは trainer 側の拒否ではないので、API から `train_text_encoder=true` を
       送る経路は通る。

    #### 【step 3 の後追い】`train_unet` 修正の arch 横断的な副作用（3 件とも解決）

    項目 1 の修正は SenseNova 固有ではなく **全 arch の full FT の挙動を変える**もの
    だったので、旧文はここに 3 件を「未修正」として開いたまま記録していた。
    **3 件とも閉じた。** 以下は結論と、閉じる過程で**旧文が事実として誤っていた**と
    判明した点である。

    1. **LoRA / ReLoRA 側の同じ穴（閉じた）。** `LoRATrainer(...)` にも
       `ReLoRATrainer(...)` にも `train_unet=` が無く、**同じ UI のチェックボックスが
       full FT では効き LoRA では効かない**非対称になっていた（旧文は LoRA だけを
       挙げていた。**ReLoRA も同じ穴で、`ReLoRATrainer` は `LoRATrainer` の
       サブクラスなので同じ 1 行である**）。両方に渡すようにした。
       - 効かせる場所は adapter ではなく **`LoRATrainer._apply_lora` の
         `if self.train_unet:`** なので、`train_unet` を一度も参照しない 3 つの
         LoRA adapter（sensenova / ideogram4 / minimax_h3）も含めて**全 arch が従う**。
       - **これは 13 arch すべてに対する挙動変更である**（意図的だが、記録が要る）。
         これまで LoRA / ReLoRA では無視されていた `train_unet=false` が意味を持つので、
         **`train_unet=false` + `train_text_encoder=true` を保存している preset や
         過去の run は、U-Net + TE から TE のみに変わる**。両方 false のものは
         ロード前に失敗する（下の項目 3）。そうした run を U-Net LoRA を含む
         checkpoint から resume すると TE 側しか注入されず、
         `lora_trainer.load_checkpoint:370-379` は**一致する key だけを copy する**ので
         エラーではなく**静かに小さい adapter**になる。
    2. **【旧文の誤り】SenseNova の LoRA では `train_unet=False` +
       `train_text_encoder=True` は成立しない。** full FT では
       `resolve_full_finetune_branch` が `und` と名前を付ける正当な branch だが、
       **LoRA では成果物にならない** — `SenseNovaLoRAAdapter.save_checkpoint` は
       **generation branch を含まない LoRA を明示的に拒否する**
       （推論は 1 ファイルから両 branch を適用するので、消費者が存在しない）。
       したがって forwarding を素直に入れると**「100 step 学習してから最初の save で
       落ちる run」が新たに到達可能になる**。`_apply_sensenova_training_contract` の
       LoRA 分岐で **ロード前に名指しで拒否**するようにした。und half の学習は従来どおり
       `train_unet=True` + `train_text_encoder=True`（branch `both`）である。
    3. **`train_unet=False` かつ `train_text_encoder=False`（閉じた、arch 非依存）。**
       旧文は「他 arch は空のパラメータリストを optimizer に渡す」と書いていたが、
       **静かに壊れるのではない** — torch が
       `ValueError: optimizer got an empty parameter list` を出す。問題は
       **それがロード後（checkpoint 常駐後、数分後）である**ことだった。
       `train_runner._assert_training_scope_is_nonempty` が
       `lora` / `relora` / `full_finetune` について **config だけで、ロード前に**拒否する。
       ControlNet は構造上 `train_unet=False` で自分の module を学習するので対象外、
       `vae_decoder` はそもそもこの flag を持たない。
       `train_vision_encoder` + `vision_encoder_path` は**第 4 の学習対象として数える**
       （SigLIP2 のみを学習する LoRA は成立する）。
       同時に、この guard は 4 つの flag を**素の truthiness では読まない**:
       手書き YAML の `train_unet: "false"` は非空文字列なので、
       guard を通ったうえに trainer 側でも truthy に読まれていた。
       bool へ正規化して config に書き戻す（真偽値として解釈できない値は名指しで拒否、
       明示的な `null` は既定として扱う）。API は bool 型なので到達経路は手書き YAML のみ。
    4. **frontend 既定の初回 run 拒否（閉じた。ただし旧文の前提が 2 つ誤り）。**
       - **誤り (a): `batch_size` 4 は死んだリテラルだった。**
         `TRAINING_DEFAULTS["batch_size"] == 1` であり、`TrainingConfig` は起動時に
         `/schema/training-defaults` で `DEFAULT_PARAMS` を上書きする。したがって
         **バックエンドが上がっている限り、full FT の確定的な拒否は optimizer 1 件
         だけ**だった。frontend のリテラル 4 は SSOT に対して古かっただけなので 1 に直した。
       - **誤り (b): `batch_size=1` は full FT だけの契約ではない。**
         `train_runner.py:185-191` は **SenseNova の全 method** に対して要求するので、
         **古い 4 を持ち込んだ利用者は LoRA の初回 run も拒否される**。
         capability 宣言で `batch_size` に method scope を付けなかったのはこのためである。
         **その「持ち込み」の経路は localStorage ではない**（本節の初稿はそう書いていたが、
         `TrainingConfig.tsx` は training 設定を localStorage に**保存しない** —
         唯一の参照 `:1780` は sample prompt のために `txt2img_params` を**読む**だけである）。
         実際の carrier は **preset・copy-from-run・edit mode の YAML 復元**、および
         **バックエンド未起動時のフォールバックリテラル**の 4 つである。
         この 4 つはいずれも arch / method を触らずに値だけを書き換えるので、
         下の UI 実装が「値のドリフト」で収束する必要がある理由でもある。
       - **対応: capability の第 4 軸 `TRAINING_REQUIRED_VALUES`。**
         既存の 3 軸（`ARCH_UNSUPPORTED` / `TRAINING_UNSUPPORTED` /
         `TRAINING_FEATURE_UNSUPPORTED`）はいずれも「**無い**もの」を宣言するが、
         この軸だけは「**その値でなければならない**」を宣言する。
         `arch -> param -> {value, reason, methods?}` で、`GET /schema/arch-capabilities`
         が `training_required_values` として配信する。SenseNova の宣言は
         `batch_size=1`（全 method）、`optimizer=adafactor` /
         `gradient_accumulation_steps=1` / `use_ema=false`（full FT）、
         `train_unet=true`（LoRA、上の項目 2）、
         `text_encoding_mode` / `latent_encoding_mode` = `onthefly_gpu`（全 method）。
         `num_optimizer_groups` は**書かない** — `fused_optimizer_groups` として
         `TRAINING_FEATURE_UNSUPPORTED` が既に所有しており、
         2 つの表が同じ parameter を持つことは import 時の assert で禁止した
         （この assert が「並列表」ではなく「分割」であることを保証している）。
       - **【この軸は「拒否」と「上書き」の両方を運ぶ】** 最後の 2 件は他と性質が違う。
         `train_runner.py:254-255` は**全 SenseNova run** に対して 2 つの encoding mode を
         `onthefly_gpu` に**黙って書き換える**（prompt encoder が und branch そのもので
         別建ての encoder が無く、pixel-space なので latent も無い）。拒否ではないが、
         **フォームが生きた select を 2 つ出し続け、run がそれを黙って捨てる**という
         状態であり、**本軸が防ぐために存在する失敗そのもの**なので宣言に含めた。
         各 entry の `reason` が「refused」か「overwritten rather than refused」かを
         名指しし、**テストが reason の文言と runner の実際の挙動を突き合わせる**
         （`test_every_declared_entry_is_enforced_the_way_its_reason_says`）。
         表の docstring も「全件が拒否である」とは主張しない書き方に直した。
       - **UI**: 該当 control を**その値に固定して disable し、backend が書いた reason を
         その場に出す**。method radio の隣に契約全体を出し、**フォームが値を変更した項目には
         「(changed from &lt;元の値&gt;)」を付ける** — 黙って上書きしない、が前提である
         （arch / method を変えると自動的に unpin される。**元の値に戻しはしない**）。
         **収束は arch/method の identity ではなく値のドリフトで判定する**
         （effect の依存は `[requiredValues, params]`、不一致のときだけ書く）:
         preset・copy-from-run・起動時の `trainingDefaults` 一括差し替えは
         arch も method も変えずに値だけを書くので、identity 依存では
         **disable された control の中に違反値が座り、method radio を往復する以外に
         直す手段が無い**状態になる。
         frontend は値の複製を持たない（テストで `api.ts` と `TrainingConfig.tsx` の
         **両方**に reason 文字列が現れないことを確認している）。
       - テスト: `backend/tests/training_required_values_test.py`（54 件。
         「出荷既定が拒否される」「LoRA path が `train_unet` を無視していた」
         「素の truthiness なら文字列 `"false"` を学習対象として通していた」を
         negative control として記録している）。
    #### 【step 3】U-2-2 実測（2026-08-25: PASS）

    実 checkpoint（`M:/model/sensenova/sensenova_int8.safetensors`、plain int8）上の
    **実 run** である。probe は `core/training/probes/sensenova_full_finetune.py`
    （`--arm train` / `--arm reload`、**別プロセス**。host RAM のピークが重ならない
    ようにするため）。**以下はすべて実測値。**

    構成: `FullParameterTrainer`、gen branch（`train_unet=True` /
    `train_text_encoder=False`）、**adafactor**、B1、accumulation 1、
    `blocks_to_swap=0`、GC ON、bf16、64px、3 step、`save_every_n_steps=3`、
    lr **1e-6**（`generate_full_finetune_config` の出荷既定。census を通すために
    選んだ値ではない）、`set_per_process_memory_fraction(0.72)` = 34.551 GiB。

    - **adapter は `SenseNovaFullParameterAdapter`**（SD1.5 fallthrough ではない）、
      branch `gen`、**294 target**、optimizer group は 1 つ
      （lr 1e-6、294 tensor、**8,103,395,328 要素** = §6.4 のヘッダ実測値と一致）。
    - **3 step とも有限 loss**（0.3868 / 0.4116 / 0.9016）。**下降は主張しない** —
      3 step で 1 枚の画像であり、§11 Phase 2b の exit criteria は
      「壊れていないこと」だけを主張する。
    - **fused backward が設置され**（`use_fused_backward=True`）、
      **stochastic rounding は強制 ON かつ attach 検証を通過**した。
      強制は `training_log` の warning として発行された
      （`sensenova_stochastic_rounding_forced`）。
    - **updated-parameter census: 294 expect / 3 step すべて complete**、
      exempt は `und_gradient_unreachable_paths()` の 5 個
      （layer 41 の und 側。**gen branch なので expectation set には元々入らない**）。
    - **U-2-5 形式の update-nonzero census: gen 294 個中 294 個が動いた（0 個が不動）。**
      §13.4 U-2-5 が「gen なら 294、到達不能な 5 個は und 側 layer 41」と書いて
      いた前提は**コードでも実測でも正しい** — gen の enumeration は
      `*_mot_gen` / `mlp_mot_gen.*` で、5 個の名前（`layers.41.self_attn.q_proj` /
      `.o_proj` / `mlp.{gate,up,down}_proj`）とは**素の集合として交わらない**。
      要素レベルの標本（layer 0 の 4 本、bf16、lr 1e-6、3 step）では
      **3.6% / 4.0% / 4.8% / 5.6% の要素が動いた**。SR OFF なら 3 step 程度では
      ほぼ何も動かないはずの領域である（§6.3。**A/B は取っていない**ので
      これは対照ではなく観測値である）。
    - **保存**: `mixed`（gen bf16 + und int8）、7 shard + index、
      **26,982,323,715 byte = 25.1292 GiB**。§6.4 が safetensors ヘッダから
      算術で出した **25.1167 GiB** に対し **+12.85 MiB（+0.05%）**。
      **この残差の原因は測っていない** — §6.4 の予測は header の 1704 テンソルを
      **すべて**合計しているので「非 decoder テンソル」では説明にならない
      （旧文はそう書いていた。**誤り**）。候補は `state_dict()` にしか現れない
      buffer と 7 つの shard header だが、**どちらも確認していないので原因を
      断定しない**。書き込み前の disk free は **C: 618.0 GiB**。
    - **再ロード（本番 reader、別プロセス）**: `load_sensenova_from_path` で
      **gen 294 個すべてが浮動小数の `nn.Linear`、und 294 個すべてが `Int8Linear`**、
      **294/294 の weight が SHA-256 でバイト一致**。
      §12 の未測定事項 1「**mixed checkpoint の推論ロード可否**」は
      **ロードのレベルでは解消した**（生成そのものは走らせていない）。
    - **VRAM**: model resident **25.1198 GiB**、peak **26.1603 GiB allocated /
      26.2500 GiB reserved**。gate の 34.551 GiB に対して **75.7%**。
      Phase 3 の LoRA smoke の peak（17.911 GiB）との差は **+8.2493 GiB**。
      gen half の int8 → bf16 の増分は **+7.5469 GiB**（15.09375 − 7.546875）
      なので、**0.7024 GiB（差の 9%）が未説明**である。候補は Adafactor の
      factored state、stochastic rounding の per-step scratch（最大単一
      パラメータ 96 MiB × slot 数。§6.3）、allocator の断片化で、**どれも
      分離して測っていない**。「ほぼ一致」と書いていた旧文は、この 0.70 GiB を
      黙って落としていた。
    - **host RAM**: ロード前 0.97 GiB → ロード後 14.53 GiB、**プロセス peak
      32.10 GiB**（materialize の一時ピークと safetensors 読み込みを含む。
      §6.4 の「per-Linear 解放で追加ピーク 7.5938 GiB」はこの内訳の一部であり、
      直接分離して測ってはいない）。reload arm の peak は **16.22 GiB**。
      実行ホストは 93.6 GiB、空き 42.7 GiB。
    - **wall time**: model load 25.87 s、train + save 21.61 s（3 step、64px）。
      **step 単体の壁時計は分離していない**（U-2-4 の担当）。
    - **測っていないもの**（この run の時点。**取り消し線相当の更新は下記**）:
      品質、収束、解像度上限、offload との合成、
      und branch / both branch の run、`int8` / `bf16` 形式の保存と再ロード、
      cold cache からのロード時間（reload arm の 0.70 s は直前に書いた
      ファイルの page cache 上での mmap である）。
      → **このうち `both` branch の run は U-2-4 が、`und` branch の run と
      `bf16` の再ロードは U-2-5 が埋めた。** 残るのは品質・収束・解像度上限・
      offload 合成・`int8` 形式の往復・cold cache のロード時間である。
    #### 【U-2-4】4 相分割の実装と実測（2026-08-25: PASS）

    **exit gate（prefix / step 比）は §8.3.2 の「U-2-4 実測」に置いた。結論は
    prefix / step 比 0.098（p50）/ 0.103（mean）、分割の限界コスト **+9.3〜+9.7%** で、
    再計算は経済的である。**

    着地したもの:

    - **`core/training/sensenova_four_phase.py`**（新規）。境界の切断・境界勾配の
      capture・相 3 の replay を持つ。相 1 は **`no_grad` で回す** — 相 3 が
      どのみち再計算するので、相 1 のグラフは作った端から捨てることになり、
      その活性こそ分割が避けたい常駐だからである。数値は同一（同じ関数・同じ入力・
      同じ重み、`attention_dropout` は 0 が assert 済み）。
    - **`SenseNovaTrainingPhaseEvictor` の 4 相化**（既存の 3 状態機械の拡張であって
      並列機構ではない）。`und_backward` 状態、`enter_und_backward`、
      `assert_understanding_resident`、および **`und_backward → prefix` の no-op**
      （設計どおり往復を 1 回節約する）。
    - **層選択の判別子は「永続性」のまま**（§8.4）。U-2-1 以後、学習する側の Linear は
      再び Parameter を持つが、**凍結側は `Int8Linear` のままで Parameter を 1 つも
      持たない**。`parameters()` 規則は RMSNorm しか選ばず、しかも `Int8Linear` の
      scale buffer を落として **1 モジュールのテンソルを 2 デバイスに分割する**。
      4 相で新しいのは判別子ではなく、**退避する half が勾配を持つようになったこと**で、
      `_assert_grad_free` が「hook が消費していない `.grad` を持つ half の退避」を
      拒否する（§8.3 の表が名指しする「片方が更新されないまま loss だけ下がる」故障）。
    - **`_assert_prefix_cache_boundary_leaf`**（新規）。既存の
      `_assert_prefix_cache_differentiable` は **`grad_fn` の存在**を要求するので、
      4 相の境界 K/V（葉）を**必ず拒否する**。実装中に実際に落ちた。葉であることを
      **肯定的に** assert する（`requires_grad` かつ `is_leaf` かつ `grad_fn is None`）：
      `requires_grad=False` なら何も学習されず、`grad_fn` が残っていれば切断が
      起きておらず gen backward が und half に流れ込む。単一 backward 経路の
      assertion は**そのまま厳格に残す**（そこでの葉は静かな不学習である）。
    - **配線は opt-in**。`sensenova_four_phase_eviction`（既定 `False`、
      `param_defaults.TRAINING_DEFAULTS`）。`train_text_encoder` +
      `sensenova_mot_phase_eviction` + `full_finetune` の 3 つを要求し、
      `train_runner` がロード前に、`assert_four_phase_contract` が trainer 側で、
      `assert_four_phase_fused_backward` が fused backward の設置後に検査する。
      `train_runner` の「`train_text_encoder` × eviction」拒否は**削除ではなく分岐**に
      なり、拒否メッセージは lift する設定を名指しするようになった。

    **勾配パリティ（acceptance criterion）**: 合成木上で **分割の勾配と単一 backward の
    勾配は bitwise 一致**（float64 と float32 の両方）。**許容誤差はゼロで、選んだ値では
    なく導出した値である** — 分割は同じ値に同じ演算を同じ順序で当てる（再計算は決定的で
    自分自身の forward を再現し、`leaf.grad` への堆積は空バッファへの加算＝値そのもの、
    それを `autograd.backward` に渡すのは単一呼び出しが走らせるのと同じ und backward）。
    **negative control 付き**: 境界を detached で渡すと **loss は単一 backward と
    完全一致したまま und half の `.grad` が 1 つも生えない**。相 3 を飛ばした場合も同様。
    テスト: `backend/tests/sensenova_four_phase_test.py`（35 件）。
    **この bitwise パリティは合成木（CPU、fp64 / fp32）上のものであって、実グラフ
    （bf16 autocast + gradient checkpointing + 実 attention kernel）上のものではない。
    この限定を落として要約しないこと。** 実グラフ側で言えるのは実 run の結果
    （583/588 が動き、動かない 5 個が予告どおりの名前である）までである。

    **実 run（`--branch both --four-phase`、実 checkpoint、64px、3 step、adafactor、
    lr 1e-6、`mixed` 要求）。以下はすべて実測値。**

    - adapter は `SenseNovaFullParameterAdapter`、branch `both`、**588 target**、
      optimizer group 2 つ（各 294 tensor / 8,103,395,328 要素、合計
      **16,206,790,656 要素**）。fused backward 設置済み、stochastic rounding 強制 ON。
    - **3 step とも有限 loss**（1.4558 / 0.4096 / 0.4010）。**下降は主張しない。**
    - **updated-parameter census: 583 expect / 3 step すべて complete**、
      exempt は `und_gradient_unreachable_paths()` の 5 個。
    - **U-2-5 形式の update-nonzero census: 588 個中 583 個が動き、動かなかった 5 個は
      `layers.41.self_attn.{q,o}_proj` と `layers.41.mlp.{gate,up,down}_proj`** ——
      §13.4 U-2-5 が予告した名前と数え方に一致する。要素レベル標本（layer 0 の
      gen 側 4 本）は **2.75% / 2.78% / 4.00% / 3.77%**。
    - **grad norm は 2 half に分かれて出た**（`MoT-Understanding` 35.544 /
      `MoT-Generation` 30.948）。**大小は主張しない。**
    - **4 相が実際に回ったことの証拠**: step 境界の `assert_understanding_resident()` が
      通っている。この assert は状態が `und_backward` か `prefix` であることを要求し、
      `und_backward` は `denoise` からしか、`denoise` は `prefix` からしか到達できず、
      各遷移が実際の D2H / H2D を行う。
    - **VRAM: peak 32.66 GiB allocated / 33.91 GiB reserved、gate 34.55 GiB の 94.5%。
      §8.3.2 の ~19-21 GB には着地しなかった。** 理由は §8.3.2 に転記した placement の
      順序であり、`model_resident == peak_allocated`（32.66 GiB が完全一致）が
      「学習 step はロード時 high-water を一度も超えていない」ことを示している。
      **step 中の最小常駐量は測っていない。**
    - **host RAM: ロード前 0.98 GiB → ロード後 26.07 GiB、プロセス peak 61.67 GiB**
      （実行ホスト 93.6 GiB、開始時 空き 56.6 GiB）。gen only の U-2-2（32.10 GiB）に
      対し、もう 1 つの half を materialize する分だけ増えている。
    - **wall time**: model load 24.77 s、train + save 58.81 s（3 step、64px、保存込み）。
    - **保存**: `mixed` を要求したが **both branch では int8 に残す half が無い**ので
      `bf16` へ縮退し、それを告知した（§6.4 の既知挙動）。9 shard + index、
      **32.68 GiB**。
    - **測っていないもの**: 品質、収束、解像度上限、この checkpoint の再ロード
      （`--arm reload` は回していない）、4 相 OFF との A/B（64px では両 arm とも
      ロード時 high-water が peak を支配するので、この shape では差が出ない）。

    #### 【U-2-4】監査で見つかった 6 件（全件修正済み）

    1. **`FullParameterTrainer.__init__` 末尾の `print(... Training U-Net ...)` が
       `train()` の中へ孤立していた**（**arch 非依存の回帰**）。新メソッドを
       その行の上に挿入したため、**必ず return する `try` の後ろ**に落ち、
       **どの arch の full-FT run でもこのログが出なくなっていた**。`__init__` に戻した。
       「他 arch には何も影響しない」という当方の主張はこれで反証されている。
    2. **転送コストの測定に warmup が無かった**（上の §8.3.2 に経緯ごと記載）。
       さらに **p50 の内訳から mean 由来の合計を引いていた**ので、引用した成分の和と
       57% 食い違っていた。probe は 2 統計を混ぜずに両方出すようにした。
    3. **`sensenova_four_phase_eviction` が `TRAINING_DEFAULTS` にだけ在った。**
       `/schema/training-defaults` はこれを配信するのに `TrainingRunCreateRequest`
       にも `openapi.yaml` にも無く、**製品からは一切有効化できなかった**
       （`_build_train_section` は Pydantic 経由で保存された run params を読む）。
       前例として引いた `255a3ab5` は**逆のことを書いている** — 「TRAINING_DEFAULTS
       で公開すればこの API 面ができてしまう」ので**あえて避けた**、である。
       本項は診断ではなく VRAM ノブで、正しい run で raise もしないので、
       Pydantic + openapi 側に寄せた。**frontend の control は張っていない**ので、
       現状の到達経路は API と YAML である。
    4. **MNT に関する当方の訂正が誤りだった**（§8.3.2 に訂正を反映）。形は揃う。
       成立しないのは und の不変性の方で、しかも **MNT>1 は到達可能**である
       （契約が拒否するのは accumulation であって MNT ではない）。
       コストは拒否せず `sensenova_four_phase_mnt_cost` として告知する。
    5. **回復可能 OOM の batch skip が `four_phase.discard()` を呼んでいなかった。**
       中断した batch の prefix は既に切られているので、次 batch の `cut()` が
       raise し、それは OOM に分類されないので **この skip 経路が生かそうとしている
       run を殺す**。`zero_grad` の隣に置いた。
    6. **`install_training_phase_eviction` が method を見ずに flag だけを見ていた。**
       `sensenova_four_phase_eviction` は `BaseTrainer.__init__` が全 trainer に立て、
       `LoRATrainer` もこの installer を呼ぶ。`train_runner` の guard を迂回されると
       **LoRA run が対称性 backstop を緩めてしまう** — その backstop は
       まさに前段の check が走らなかった場合のために在る。installer 側でも
       full fine-tune を要求するようにした。

    #### 【U-2-5】exit smoke 実測（2026-08-25: PASS）

    実 checkpoint（`M:/model/sensenova/sensenova_int8.safetensors`、plain int8）上の
    実 run。probe は `probes/sensenova_full_finetune.py`、arm ごとに別プロセス、
    `set_per_process_memory_fraction(0.72)` = 34.551 GiB、64px、3 step、adafactor、
    lr 1e-6、B1、accumulation 1、`blocks_to_swap=0`、GC ON、bf16。
    **以下はすべて実測値。品質・収束は一切主張しない。**

    **U-2-2（gen）/ U-2-4（both）が既に着地させていたものは再測していない。**
    本節が新たに埋めたのは (1) **und branch の run が存在しなかったこと**、
    (2) **`und` × `mixed`（gen 半分を int8 に残す、gen run とは向きが逆の形式）の
    再ロード**、(3) **both branch の `bf16` checkpoint の再ロード**（`071e602b` は
    書いただけで読み戻していない）の 3 点である。

    | | und branch（新規） | both branch + 4 相（`071e602b` の arm を再実行） |
    |---|---|---|
    | adapter / branch / target | `SenseNovaFullParameterAdapter` / `und` / **294** | 同 / `both` / **588** |
    | optimizer group | 1 群（lr 1e-6、294 tensor、**8,103,395,328 要素**） | 2 群（各 294 tensor、計 16,206,790,656 要素） |
    | loss（3 step） | 1.9414 / 0.3786 / 1.4027 | 0.4108 / 0.5655 / 0.3779 |
    | fused backward / SR 強制 | 設置済み / ON・attach 検証通過 | 同 |
    | updated-parameter census | **289 expect、3/3 complete**、exempt 5 | **583 expect、3/3 complete**、exempt 5 |
    | **update-nonzero census** | **294 個中 289 個が動いた** | **588 個中 583 個が動いた** |
    | 動かなかったもの | `layers.41` の `self_attn.{q,o}_proj` と `mlp.{gate,up,down}_proj` | 同じ 5 個 |
    | 要素レベル標本（layer 0 の 4 本） | und 側 **2.43 / 2.47 / 2.79 / 2.94%** | gen 側 **3.35 / 3.21 / 4.78 / 4.24%** |
    | 保存 | `mixed` 実効 `mixed`、7 shard + index、**26,982,061,395 B = 25.12900 GiB** | `mixed` 要求 → `bf16` へ縮退（告知あり）、**35,091,856,594 B = 32.68184 GiB** |
    | VRAM peak allocated / reserved | **26.2571 / 26.5508 GiB**（gate の **76.0%**） | **32.6606 / 33.9063 GiB**（**94.5%**） |
    | model resident | 25.1198 GiB | 32.6606 GiB（= peak allocated、完全一致） |
    | host RSS peak | **32.101 GiB** | **51.965 GiB** |
    | wall（model load / train+save） | 17.12 s / 23.11 s | 24.71 s / 45.98 s |

    - **und の census は 289 であって 294 ではない、が実測でも成立した。**
      動かなかった 5 個は `und_gradient_unreachable_paths()` が名前で予測する
      layer 41 の 5 本と**集合として完全一致**する。exempt 集合（census が
      期待集合から落とす 5 個）と unmoved 集合が同じであることは、
      **「hook が発火しなかった」と「勾配が構造的に届かない」を混同していない**
      ことの確認でもある。同じ層の `k_proj` / `v_proj` は**動いた**（gen 側 layer 41 が
      その K/V を消費する）ので、層まるごとを除外する criterion なら誤りである。
    - **criterion は読む数値ではなく assertion になった。** probe は
      `u2_5_unmoved_expectation()` で予測集合を作り、unmoved 集合と一致しなければ
      **JSON を書いてから raise する**（25 GiB 書いた run が自分の数値まで失う理由が無い）。
      gen branch で予測が空集合になるのは `*_mot_gen` 命名により交わらないからで、
      これも同じ 1 本の関数から出る。
    - **再ロード（本番 reader `load_sensenova_from_path`、trainer を持たない別プロセス）。**
      - `und` × `mixed`: **gen 294 個すべてが `Int8Linear`、und 294 個すべてが浮動小数の
        `nn.Linear`、und 294/294 が SHA-256 でバイト一致**。gen run（`22b22f09` 以来
        唯一検証されていた形）と**向きが逆の形式**で、これが初回である。
        load 0.573 s、host RSS peak **16.225 GiB**。
      - `both` × `bf16`: **588 個すべてが浮動小数、int8 は 0 個、588/588 が SHA-256 で
        バイト一致**。**量子化テンソルが 1 つも無いツリーを reader が受理する**ことは
        ここで初めて実測された（`verify_quantized_swap` は 0 対 0 で通る）。
        load 0.676 s、host RSS peak **31.309 GiB**。
      - どちらの arm も **checkpoint の metadata（`sensenova_trained_branch` /
        `sensenova_save_format`）を権威として読み**、train arm の申告と突き合わせている。
        以前の reload arm は **gen × mixed の答えをハードコードしていた**ので、
        上の 2 形式は**この arm では検査できなかった**（下記「見つかったもの」1）。
    - **`und` × `mixed` は gen × mixed より 262,320 byte 小さい**（26,982,061,395 対
      26,982,323,715）。両者は同じテンソル総量なので差は shard header 側にあり、
      und の key 名が短い（`mlp.` 対 `mlp_mot_gen.`）ことと整合する。
      **これは U-2-2 が残した「+12.85 MiB の残差」を説明しない** — 桁が 2 つ違う。
      残差の原因は依然として測っていない。
    - **both branch の host RSS peak は再現しなかった。原因は構造にある。**
      同一コマンド・同一 checkpoint で `071e602b` は「ロード後 **26.07** GiB /
      peak **61.67** GiB」、本 run は「ロード後 **9.04** GiB / peak **51.97** GiB」。
      **差は peak が 9.70 GiB、ロード後が 17.03 GiB で、同じではない**
      （初稿は「どちらも約 9.7 GiB」と書いていた。**誤り**。しかも
      **大きい方＝ロード後の 17.03 GiB の方が情報量が多い** — 差がロード時点で
      既に開いていることを示す）。
      - **測っている量が悪い。** probe の `_host_peak_bytes()` は
        `psutil.memory_info().peak_wset`、すなわち **peak working set** である。
        Windows の working set は (1) **常駐している mmap 済みファイルページを含む**
        （17.6 GiB の base に対する page cache が warm か cold かで動く）、
        (2) **メモリ圧の下で OS がトリムできる**（＝下がりうる）。したがって
        **プロセスが所有する何かの単調な high-water ではない**。
        `071e602b` の run は空き 56.6 GiB から始まり、本 run は**同じセッションで
        32 GiB の und run の直後**に走った。これは 9.70 と 17.03 の**両方**を説明し、
        この量を「謎めいて再現しない」ではなく**構造的に再現しない**ものにする。
      - **対応（実施済み）**: probe は `peak_pagefile`（peak commit charge、
        本 venv で存在を確認）を `host_rss.peak_commit_gib` として併記するようにした。
        **これは常駐ファイルマッピングに膨らまされず OS にトリムもされない**ので、
        **host RAM 予算を書くならこちらである**。
        `_host_peak_bytes()` の docstring に両者の違いを書いた。
      - **運用**: 依然として **61.67 GiB の側を上限として扱う**。
        追試のために 32 GiB の arm を回し直す価値は無い。
      - **VRAM peak は再現した** — `071e602b` と本 run はともに
        **記録されている精度（32.66 GiB）で一致**する
        （`071e602b` の実測ボックスは 2 桁までしか残しておらず、当時の JSON は
        削除済みなので、「バイト単位で一致」とは**言えない**）。
        `model_resident == peak_allocated` も両 run で成立している。
    - **2 つの both run は決定的ではない。** loss は
      `071e602b` が 1.4558 / 0.4096 / 0.4010、本 run が 0.4108 / 0.5655 / 0.3779 で、
      **probe が seed を固定しているのは `sample_seed` だけ**（noise と timestep
      サンプリングは固定していない）。「同一コマンド」は字義どおり真だが
      **「他がすべて同じ」ではない**。loss 列がその可視の証拠であり、
      host RSS の差をこの非決定性だけで説明することも**できない**
      （上の working-set の性質の方が支配的だが、**切り分けは測っていない**）。

    #### 【U-2-5】「und branch に 4 相分割は要るか」— 要らない（コードからの結論、run で確認）

    **要求していない。** `assert_four_phase_contract` が要求するのは
    `train_text_encoder`（4 相が存在する理由）であって、その逆ではない。
    und-only の run は **単一 backward 経路**を通る — `encode_prompt` の
    `requires_grad and four_phase is None` 分岐で、prefix は `grad_fn` を持つ
    生きたグラフとして構築され、`_assert_prefix_cache_differentiable` が
    それを**肯定的に**要求する。実測 run はこの経路で走った
    （`four_phase_eviction: false`、`evictor_states_during_run: []`）。

    **要る場合は 1 つだけある: und 学習と MoT half-eviction を同時に使うとき。**
    `encode_prompt` は「requires_grad + evictor あり + 4 相なし」を実行時に
    `RuntimeError` で拒否し、`train_runner` は同じ組を**ロード前に**拒否して
    `sensenova_four_phase_eviction` を名指しする。すなわち
    **4 相は und 学習の前提ではなく、und 学習 × eviction の唯一の合法な形**である。
    eviction を使わないなら 4 相は「second backward と再計算 forward を足すだけ」で、
    契約自身がそう書いて拒否する。

    #### 【U-2-5】§11 Phase 2b-5 が要求していた不変条件テストを足した

    §11 の 2b-5 は exit smoke に「**prefix forward を checkpointed region の外に
    置く不変条件のテスト**」を含めることを要求していたが、`b2694674` /
    `071e602b` のどちらもこれを着地させていない。
    `backend/tests/sensenova_u2_5_exit_smoke_test.py` に着地させた:
    prefix を**自分の checkpoint 無し**で構築し、gen loop を `checkpoint_layers=True`
    で回して backward すると、**und の 3 層はフォワードの 1 回しか呼ばれない**
    （gen backward は自分の segment を再計算するが und 側には入らない）。
    **負の対照つき**: prefix 構築を gen の checkpoint segment の**内側**に置くと
    呼び出し数が `[0,1,2]` から `[0,1,2,0,1,2]` に倍増する。
    加えて gen の再計算が**同じ cache オブジェクト**（`id` 一致）を読むことと、
    `no_grad` prefix が `trainable_prefix=True` の下で依然拒否されることを固定した。

    #### 【U-2-5】見つかったもの（2 件、いずれも probe 側。trainer 側には無し）

    1. **reload arm が gen × mixed をハードコードしていた**（修正済み）。
       「294 個の gen が浮動小数で、294 個の und が `Int8Linear`」だけを問う実装で、
       これは 3 形式 × 3 branch のうち**ちょうど 1 つ**の答えである。
       したがって **U-2-5 が閉じるべき 2 つの形式は、この arm では検査できなかった** —
       `und` × `mixed` には全項目が偽（向きが逆）、`both` × `bf16` には
       「und が int8」が偽（int8 half が存在しない）になる。
       **さらに形状判定だけの問題ではなかった**（初出の記述はここを過小評価していた）:
       **SHA-256 の digest 比較も gen 側からしか作っていなかった**ので、
       `und` × `mixed` では形状の 2 件に加えて **294/294 が digest 不一致**として
       報告されることになる。すなわち「学習した half がバイト一致で読み戻せるか」
       という**この arm の中心的な問い自体が、gen 以外の branch では機能していなかった**。
       `expected_read_shape(branch, effective_format)` に規則を出し、
       arm は checkpoint metadata から branch と実効 format を読むようにした。
       テストは**規則を書き写さず**、9 通りすべてを本番 writer で書いて本番
       read sequence で読み戻し、出てきた形と突き合わせる。
    2. **U-2-5 の criterion が「読む数値」でしかなかった**（修正済み）。
       moved / unmoved を報告するだけで、289 という期待値はどこにも無かった。
       上記のとおり assertion 化した。負の対照として
       「294 個すべてが動いたと主張する assert はここで落ちるのが正しい」を
       **実行可能な形で**テストに入れてある。

    **trainer 側の欠陥は 1 件も出なかった。** gen run が 3 件、both run が 3 件
    見つけたのとは対照的だが、それは驚くことではない — und branch が通る配線
    （branch 解決 / adapter / fused backward / census / 保存）は
    その 6 件がすべて修正した後の同じコードであり、branch 依存の分岐は
    `resolve_full_finetune_branch` と列挙器の中だけに閉じている。

    #### 【U-2-5】und full FT は U-1 の text-only スコープをそのまま継承する

    **`train_text_encoder=true` + reference データセットは、25-32 GiB をロードしてから
    最初の item で落ちる。** `encode_prompt` は `requires_grad` が立っている限り
    reference 条件付き item を `NotImplementedError` で拒否し
    （`sensenova_ops.py:919-924` と `:948-953`。単一 backward 経路と 4 相経路の**両方**）、
    一方 `use_reference_images` は `train_runner.py:211` で
    「**Normalized, not gated**」— すなわちロード前に拒否する gate が無い。

    **これは U-2-5 が作った欠陥ではない**（und LoRA でも同一で、U-1 以来同じ形である）。
    ここに書くのは、**full FT の下で und branch に到達できるようになったのが初めて**
    だからである。§13 の U-3（und × reference）はまさにこの穴を埋めるフェーズであり、
    それまでの間、この組み合わせは**ロード前に拒否されるべきもの**として残っている。

    #### 【U-2-5】新しい 2 branch で exercise していない経路

    - **MNT > 1**。両 run とも `multi_noise_timesteps: 1` である。4 相 × MNT>1 は
      `warn_four_phase_mnt_cost` が告知する経路だが、**新しい branch では踏んでいない**
      （U-1 の LoRA arm は MNT>1 を踏んでいる）。
    - **学習中の sample 生成**。両 run とも `sample_every_n_steps: 0`、
      `sample_prompts: []` である。full FT × sample の相互作用
      （`generate_sample` の evictor 駆動と attention mode の再 stamp。§11 Phase 1）は
      **どちらの新 branch でも走っていない**。

    #### 【U-2-5】測っていないもの（U-2 全体として残るもの）

    - **品質・収束**。3 step・64px・画像 1 枚であり、§11 Phase 2b の exit criteria は
      「壊れていないこと」だけを主張する。短 horizon では stochastic rounding の
      誤差が信号と同程度なので（§6.3）、**A/B は測定として無効になる**。
    - **解像度上限**。全 run が 64px である。both branch は 64px で既に gate の 94.5% を
      使っており、しかも `model_resident == peak_allocated` なので
      **解像度を上げたときに step 側が high-water を超える点は測っていない**。
    - **offload との合成（U-2-4 の 2b-4 / §8.3.1）**。`LayerOffloadConductor` の
      サブモジュール粒度は依然として未調査。
    - **4 相 ON / OFF の A/B**。64px では両 arm ともロード時 high-water が peak を
      支配するので、この shape では差が出ない（U-2-4 と同じ理由）。
      **und-only での 4 相 run も回していない**（要らないので回さなかった。
      「要らない」は上記のとおりコードからの結論である）。
    - **`int8` 形式の実 run 往復**。3 形式のうち `mixed`（両向き）と `bf16` は
      実 checkpoint で往復したが、`int8` は合成ツリーのテストだけである。
      これは**再学習の base になれる唯一の形式**なので（§6.4）、
      resume の実測も同時に空いたままである。
    - **保存した checkpoint での生成**。reader が読めることまでで、
      推論そのものは 3 branch とも走らせていない。
    - **`+12.85 MiB` の残差**（U-2-2）。本 run の 262,320 byte 差は桁が違うので
      説明にならない。
    - **host RSS peak の再現性**。上記のとおり同一 arm で peak 9.70 GiB /
      ロード後 17.03 GiB 動いた。機構（working set の性質 + セッション履歴）は
      特定したが、**working set の変動と run 自体の非決定性の切り分けは測っていない**。
      `peak_commit_gib` は今回の 2 run には**存在しない**（今回追加したため）ので、
      **再現量としての host 予算は次の run から取れる**。
    - **step 中の最小常駐量**（peak しか記録していない。U-2-4 から継続）。
    - **grad norm の大小**。both run では 2 half に分かれて出るが、§13.6 の訂正表の
      とおり**どちらが大きいかは設計判断の根拠にしない**。

- **U-3 — und × reference。** 依存は U-1 + Phase 3（**Phase 3 は DONE なので、
  実質の依存は U-1 だけになった**）。**`vision_model`（und tower の
  ViT）自体は学習対象に含めない** — 294 target の外で推論側に検証手段が無く、
  §5.2 根拠 3 の「消費者の居ない形式を作らない」原則が当たる。したがって
  **und 学習 = 「und decoder 層の学習」と定義する**。

**Phase 3 との関係は直交である。** und trainable と reference あり item を組み合わせた
場合、reference の `<IMG_CONTEXT>` token は**同じ prefix pass の und 層を通る**ので、
**追加機構ゼロで reference 条件付け経路も学習される**。これは §7.3 が保留していた
和解経路の実体化にあたる（§7.3 と矛盾しない — 既定 OFF である限り §7.2 判断 3 も
生きている）。

### 13.5 U-0 実測（2026-08-24: PASS、`3d837202`）

実 checkpoint 上での測定である。**以下はすべて実測値。**

- **K/V parity: 42/42 層 bitwise 一致。** 学習 prefix loop と vendor
  `_t2i_prefix_forward` を比較。checkpoint あり / なし / autocast 下の **3 モード
  すべて**で一致した。これは同時に「既定経路が変わっていない」ことの証拠でもある。
- **勾配伝播: und LoRA 289 個が有限かつ非ゼロ。** 実際の flow-matching loss からの
  1 回の backward で到達する。**§13.1 の構造的推論が実証された。**
- **到達しない 5 個は構造であって欠陥ではない**:
  `layers.41.self_attn.q_proj` / `.o_proj` と `layers.41.mlp.{gate,up,down}_proj`。
  prefix は `past_key_values` を残して `last_hidden_state` を捨てるので、**最終層の
  attention 以降が何も生まない**。同じ層の `k_proj` / `v_proj` は**学習される** —
  generation 側の layer 41 がその K/V を消費するからである。**推論もまったく同じ
  テンソルを捨てる**ので、これはモデルの形である。
- **prefix checkpoint は前提実装**（§13.2 の実測ボックス参照）。解析値 15.093 GiB、
  実測の傾き 66 MB/層、全深度外挿 17.65 GiB、model resident 込みで 35.2 GiB。
- **`LoRALinearLayer` の fp32 adapter は ambient autocast に依存する**（§13.1 の
  「4 つ目」）。`encode_prompt` に autocast が無く、最初の und-LoRA prefix pass が
  layer 0 で dtype 不一致を起こした。autocast を張り直しても parity は不変である
  ことも確認した。

### 13.6 U-1 実機 exit smoke（2026-08-24: PASS、`327276df`）

17.6 GiB の plain-int8 checkpoint に対し、**5 アームをそれぞれ独立プロセス**で、
Phase 1 probe と同じ per-process VRAM ゲート下で実行した。**以下はすべて実測値。**

| arm | 結果 |
|---|---|
| 3-step trainer | loss `[0.41467, 0.37538, 0.52401]`、保存 **1764 tensors / 588 targets**、metadata `lora_targets=generation+understanding`、optimizer group `[{lr 1e-4, 588}, {lr 5e-5, 588}]` |
| fresh runtime | **588 適用 / 588 復元**、**strength 0 が base と `torch.equal`**、strength 1 は max abs delta **0.015625** |
| `train_text_encoder=false` 回帰 | `3d837202` に対し 3 loss・6 grad digest・parameter hash・保存 tensor hash が一致、**peak allocated がバイト単位で同一（19,424,865,792 B）** |
| MNT>1 | prefix build **4 回**（2 batch × 2 MNT）、freed-graph エラー無し、各 step で und nonzero = **289** |
| 既存 gen-only 蒸留 LoRA | **294/294 適用・復元**、strength 0 parity |

- **strength 0 と strength 1 の対**が要点である。**片方だけでは意味が無い** —
  strength 0 の一致は「壊していない」ことしか言わず、**strength 1 の差分が
  「ロードしても何も起きない LoRA」ではないことを示す**。これは §13.3 の推論側
  列挙器修正が防いでいる故障そのものである。
- **列挙器を広げても既存フォーマットのコストはゼロ**: 蒸留 LoRA は both-branch 列挙
  下でも 294/294 に到達する（適用は lookup 駆動なので und スロットが空振りするだけ）。
- **positive assertion が実際に発火している**: 3-step arm で 9 回、MNT arm で 12 回、
  census は毎回 `(42, 42, 42)`。
- **autocast wrapper は装飾ではない**: 外す破壊実験を 1 回だけ行い、U-0 と同じ
  dtype エラー（`BFloat16 != float`）が再現することを確認した。
- **到達しない 5 個は U-0 が名指ししたとおりに現れた**（全 step で `.grad` が
  ゼロではなく `None`、adapter はゼロ初期化のままファイルに残る）。したがって census は
  **294 個中 289 個到達**であり、「294 個すべてが動いた」と主張する assert は
  運が悪いのではなく**誤り**である。
- **VRAM**: 全アーム 64×64 で最大 **18.57 GiB**（hard gate 34.55 GiB の 53.7%）。

#### 勾配ノルムの大小は主張しない（U-0 コミットメッセージの訂正）

`3d837202` のコミットメッセージは「und の勾配ノルムは gen より大きい」と書いたが、
**これは一般には成り立たないので訂正する。**

| 測定 | 条件 | 結果 |
|---|---|---|
| U-0 | **合成 x0**（一様乱数）+ **固定 `t=0.5`** | und 3.410e-2 > gen 3.094e-2 |
| U-1 exit smoke | **実画像** + **sampler が引いた timestep** | gen 0.0046 / 0.0204 / 0.0408 > und 0.0030 / 0.0178 / 0.0179 |

**2 つは同一条件の比較ではないので、どちらの順序も一般的主張にはならない。**
どちらが大きいかを設計判断の根拠に使わないこと。この訂正は `327276df` の
コミットメッセージにも記録してある。

---

## 14. References

- 内部（推論側）:
  [`sensenova/loader.py`](../../backend/core/models/sensenova/loader.py),
  [`sensenova_pipeline_ops.py`](../../backend/core/models/sensenova/sensenova_pipeline_ops.py),
  [`sensenova_lora.py`](../../backend/core/models/sensenova/sensenova_lora.py),
  [`mot_phase_eviction.py`](../../backend/core/models/sensenova/mot_phase_eviction.py),
  [`kv_cache_streaming.py`](../../backend/core/models/sensenova/kv_cache_streaming.py),
  [`vendor/modeling_qwen3.py`](../../backend/core/models/sensenova/vendor/modeling_qwen3.py),
  [`vendor/modeling_neo_chat.py`](../../backend/core/models/sensenova/vendor/modeling_neo_chat.py)
- 内部（学習側の型・前例）:
  [`arch/base_arch.py`](../../backend/core/training/arch/base_arch.py),
  [`arch/krea2.py`](../../backend/core/training/arch/krea2.py),
  [`ops/minit2i_ops.py`](../../backend/core/training/ops/minit2i_ops.py)（pixel-space x0 予測の構造テンプレート）,
  [`ops/flux2_ops.py`](../../backend/core/training/ops/flux2_ops.py)（reference conditioning の前例）,
  [`adapters/base_adapter.py`](../../backend/core/training/adapters/base_adapter.py)（`reject_quantized_base` / `warn_quantized_base_without_checkpointing`）,
  [`adapters/ideogram4_adapter.py`](../../backend/core/training/adapters/ideogram4_adapter.py)（full FT ガードのテンプレート）,
  [`base_trainer.py`](../../backend/core/training/base_trainer.py)（`_attach_stochastic_rounding`, `:3313-3386`）,
  [`optimizers/stochastic_rounding.py`](../../backend/core/training/optimizers/stochastic_rounding.py)（bf16 丸め欠陥の機構と永続 master 棄却の根拠）,
  [`optimizers/RINGBUFFER_OPTIMIZERS.md`](../../backend/core/training/optimizers/RINGBUFFER_OPTIMIZERS.md),
  `backend/tests/bf16_stochastic_rounding_test.py` / `bf16_stochastic_rounding_default_optimizer_test.py`（`|w| <= 512*lr` の閉形式）
- 内部（文書）:
  [`ADD_A_MODEL_ARCHITECTURE.md`](ADD_A_MODEL_ARCHITECTURE.md),
  [`MODEL_FACTS.md`](MODEL_FACTS.md) の sensenova 行,
  [`INT8_W8A8_TRAINING_GATE.md`](../../backend/core/training/INT8_W8A8_TRAINING_GATE.md)（G3/G4）,
  [`MINIMAX_H3_CONTINUAL_TRAINING_DESIGN.md`](MINIMAX_H3_CONTINUAL_TRAINING_DESIGN.md)（本文書の書式の前例）
- 外部:
  [SenseNova-U1 (OpenSenseNova)](https://github.com/OpenSenseNova/SenseNova-U1),
  upstream issue #207（mixed und/gen forward の未検証）
