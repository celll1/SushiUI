# SenseNova U1.5 学習設計案

> Status: Phase 0 と Phase 1 は完了。Phase 2b、Phase 3 は未完。
> Date: 2026-08-24
> Scope: SenseNova-U1.5-8B-MoT の (1) LoRA 学習 / (2) full-parameter fine-tune /
> (3) reference 画像を含むデータセットの混在学習
> 本文中の `file:line` は 2026-08-23 時点の静的調査による。

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
   int8 のみで、既知の非対応はモデルロード前に拒否する。実装本体は bf16 base の
   入手を前提条件として Phase 2b に送る。
   実装する場合の対象は **gen branch のみの 8.1B**（both-branch 16.2B は設計対象外）。
3. **Phase 3（reference 混在）は per-item presence を真とする。** 初版は物理
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
| reference / full FT | §11 の後続フェーズ。full FT の律速は bf16 base の入手ではなく gate/loader の method-aware 化（§6.4）、reference は flux2 ハードゲート 6 箇所の解除から（§7.5） | PENDING |

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
   2 つの独立した forward をまたぐ微分可能な KV パイプラインを構築し、42 層分の
   prefix activation を backward まで保持し、`cat[prefix_KV, gen_KV]` の flash
   attention に勾配を通す必要がある。これはフラグではなくサブシステムであり、
   phase 1 の LoRA に対して費用対効果が極端に釣り合わない。
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

**保留（deferred）**: `scope: generation | both` の選択肢。追加のトリガは Phase 3 で
reference 忠実度が実測で不足した場合のみ（§7.3）。`understanding-only` は用途が無く
推論側で検証もできないため**恒久的に提供しない**。

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
現行メッセージは int8 base で full FT が未実装であることと LoRA 代替を示す。bf16 base
を前提とする本体設計は Phase 2b に残る。

### 6.2 確定した設計判断 — 対象は gen branch のみ（fable 諮問）

**判断: 実装する場合の full FT は gen branch のみの 8.1B。これを SenseNova における
「full fine-tune」と呼ぶ。both-branch 16.2B はロードマップに載せない。**

根拠:

- **座りが悪いのは gen-only ではなく both-branch の方である。** 言語理解を担う branch
  を含む 16.2B を学習するのは「text encoder も一緒に fine-tune する」ことであり、
  本リポジトリはどの arch でもそれを既定にしていない。破滅的忘却の profile が
  悪く、同時に bf16 weight 32.4 GB + gradient 32.4 GB だけで 48 GB を超え、
  activation と一時領域を置けない。原理とメモリの両方で落ちるので選択肢から外す。
- **gen-only の算術は厳しいが現実的:** bf16 weight 16.2 GB + gradient 16.2 GB +
  optimizer state（CPU offload）+ stochastic-rounding の per-step scratch + pixel space の
  activation（checkpointing 下）。48 GB で閉じるには gen 半分の offload と
  optimizer state の CPU 常駐が必要になる。つまり **Phase 2 は §8 の
  offload 機構の存在に依存する**。この依存順序自体がガード先行を正当化する。

**Phase 1 の実装がこの判断を強化した。** `encode_prompt` は `requires_grad=True` を
即 raise し（`arch/sensenova.py:27-32` → `ops/sensenova_ops.py`）、prefix の
immutability は forward のたびに `_assert_immutable_prefix_cache` で検証される。
und を学習可能にするには**この不変条件群と 2 パス構造そのものを解体する**必要があり、
§5.2 の「フラグではなくサブシステム」という評価が実装として具体化された形である。

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
重心が入手から実装へ移ったため改題した。** 現行の gate
`_assert_supported_quantized_training_base`（[`ops/sensenova_ops.py:49-91`](../../backend/core/training/ops/sensenova_ops.py)）は
**未量子化 bf16 base を明示的に拒否する** — docstring と例外メッセージの両方が
"and so is an unquantized bf16 base" と書いている。しかも
`load_components`（`:155-156`）は training method を一切見ずに無条件でこれを呼ぶ。
したがって upstream から bf16 を入手しても現行 loader 経路ではロードできず、
**gate と loader の method-aware 化という実装作業が、入手とは独立に必要**である。
これは Phase 2b の前提条件が「artifact の入手」から「配線の実装」に変わったことを意味する
（構造的推論ではなく、gate のコードそのものから読める事実）。

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
    skew と重なる。
- **(b) upstream の 46.8 GiB bf16 ソースから gen half を抽出した artifact。** ~16.2 GB。
  §6.2 の推奨形。
- **(c) upstream bf16 をそのまま両 half bf16 でロードする。** 学習対象は gen half のみでも、
  und half が bf16 で常駐するぶん VRAM 要求が上がる。

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

Phase 2b 本体を実装して受付を開く際は、loader と利用者向け文書で採用した経路を
明示する。現行ガードは未提供の full FT を広告せず、未実装であることと LoRA 代替だけを
示す。

---

## 7. Phase 3 — reference 画像を含むデータセットの混在（PENDING）

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

### 7.5 実装差分（設計判断は §7.1-§7.4 のまま、配線の具体）

§7.1-§7.4 の設計判断はすべて維持する。以下は Phase 3 の実装時に必要になる配線で、
**§9 の統合ポイント一覧に未記載だったもの**である。B1 強制はむしろ必然性を増した —
reference は各 ref の smart-resize で token 数が per-item に変わるため、prefix を
さらに ragged にする。

**差分 1: `use_reference_images` の flux2 ハードゲート解除（必須）。** 現状は 6 箇所で
gate されている。

| 箇所 | 内容 |
|---|---|
| `train_runner.py:202-203` | sensenova で `ValueError`（Phase 3 deferral） |
| `base_trainer.py:8085-8086` | 同じ拒否の trainer 側の重複（§9 の「flag 代入ブロック 2 箇所」と同型） |
| `base_trainer.py:8107-8110` | 非 flux2 は warn して**無視**（"only supported for FLUX.2, will be ignored"） |
| `base_trainer.py:8269` | `separate_by_reference = use_reference_images and self.is_flux2` |
| `base_trainer.py:10700` | reference latent の encode 分岐が `and self.is_flux2` |
| `base_trainer.py:11005` | batch への引き回しが `and self.is_flux2` |

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
text-only 経路の `indexes_cond.shape[1]`（`:463-464`）とは別式）。学習側もこれに揃え、
フィールドを「次の t index」として一般化すること。**放置すると位置ずれが形状エラー
なしに静かに起きる。**

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

### PENDING

- Phase 2b full FT 本体と Phase 3 reference 混在。
- **§9 に未記載だった統合ポイント**（設計再検証で判明）: Phase 3 は
  `use_reference_images` の flux2 ハードゲート 6 箇所（`train_runner.py:202-203`、
  `base_trainer.py:8085`, `:8110`, `:8269`, `:10700`, `:11005`）の解除が必須で、
  `SenseNovaTrainingPrefix.text_length` の意味論変更を伴う。Phase 2b は
  `ops/sensenova_ops.py` の gate と `load_components` を method-aware にする必要が
  ある。詳細は §7.5 / §6.4。

### DONE — 登録から自動的に得られたもの

`_build_cache_namespace` は `self.arch.name` を読むだけになっており、
`pixel_align` / `temporal` も handler のクラス属性を読む宣言的な機構なので、
cache namespace と alignment は登録だけで有効になった。

---

## 10. SenseNova 固有のリスク

| リスク | 内容 | 緩和 |
|---|---|---|
| mixed forward の欠落 | issue #207。1 パスで und/gen を混ぜられない | 2 パス構造を設計の前提にする（§4.2）。修正を前提にしない |
| int8 base のみ | full FT が構造的に不可能 | Phase 2 をガード先行にする（§6.1） |
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
| prediction config の退行 | 既存の flow-matching 登録を学習統合時に外すと静かに誤る | §9 の退行テストで固定 |
| half-eviction の層選択 | Parameter ベースの規則は 2 度不活性のまま出荷された | 判別子ごと再利用する（§8.4） |
| prefix forward のコスト | Qwen3-8B 全体を毎 step 通す | 実測後にキャッシュ可否を判断（§12） |
| pixel space の activation | VAE が無いぶん activation が pixel 解像度に比例 | gradient checkpointing + 解像度上限。block swap では減らない |

---

## 11. フェーズ分割

初期計画を残し、現在の DONE / PENDING 境界を明示する。

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
- **2b-1 — gate と loader の method-aware 化。**
  `_assert_supported_quantized_training_base` と `load_components` が training method を
  見るようにし、§6.4 の供給経路のどれを受理するかを決める。
- **2b-2 — `SenseNovaFullParameterAdapter`。** §6.1 で「共通 preflight だけで
  fail-closed になるため追加しなかった」もの。本体実装時には必要になる。
  decoder 外の gen 側モジュールを含めるかの決定（§6.2）をここで確定する。
- **2b-3 — bf16 rounding-defect の契約。** §6.3 の推奨 2 点
  （`optimizer: adamw` 拒否、`optimizer_stochastic_rounding` の contract 既定 True）を
  決定して実装する。実装量はゼロに近く、決定が本体である。
- **2b-4 — offload の合成。** §8.3.1 のモジュール粒度問題を解決する。
  `LayerOffloadConductor` がサブモジュール粒度のリストを受けられるかの調査が先行する。
- **2b-5 — exit smoke。** prefix forward を checkpointed region の外に置く不変条件の
  テストを含む。
- **exit criteria**: **「学習が壊れていないこと」だけを主張し、品質は主張しない。**
  短 horizon では stochastic rounding の誤差が信号と同程度で（§6.3）、A/B が測定として
  無効になるためである。

### Phase 3 — reference 混在（PENDING）

- **3-1 — gate 解除と配線。** §7.5 差分 1 の 6 箇所。
- **3-2 — reference prefix の構築。** §7.5 差分 2（`text_length` の一般化）と
  差分 3（推論側関数の再利用）。reference の前処理は `sensenova_ops` 内に閉じる
  （差分 4）。
- **3-3 — 学習中 sample の ref 対応。** 現在 `generate_sample` は
  reference/condition image を無視して warn する（`dc91bef1`）。
- **3-4 — 混在 smoke。** ref 有り / 無し dataset を 1 run で混ぜる。
- 既存の run-global `use_reference_images` と per-item `reference_images` を再利用する
  （新しい dataset-level parameter / API 変更は行わない）。
- `separate_by_reference` の SenseNova への適用（bucket key に反映）。
- **exit criteria**: 混在 run で両種類の batch が形状エラーなく通ること、
  **ref 無し step の loss / grad SHA-256 が Phase 1 実装と一致すること**（何も壊して
  いないことの証明）、および **t-extent 検証** — ref 有り prefix で
  `text_length ≠ token 数` になるケースを最低 1 つ含み、image index の t 基点が
  `indexes[0].max()+1` と一致することを確認する（§7.5 差分 2 の静かな位置ずれは
  これでしか捕まらない）。

---

## 12. Open questions（実装時に決めること）

- **prefix KV をキャッシュするか。** caption ごとに 42 層 × 全 token の K/V は容量が
  大きい。毎 step 計算のコストを実測してから決める。Phase 0 の計測項目。
- **`optimizer_stochastic_rounding` を SenseNova full FT で既定 ON にするか、
  OFF を拒否するか。** 既定 False のまま出すと既知の欠陥を再生産する。
  **推奨は contract で既定 True に上書き（全 arch 共通の既定は変えない）。決定は
  Phase 2b-3。** §6.3。（永続 fp32 master は選択肢に含めない。棄却済み。）
- **`optimizer: adamw` を SenseNova full FT で拒否するか。** per-parameter seam が
  無く stochastic rounding をかけられない唯一の optimizer である。
  **推奨は拒否。決定は Phase 2b-3。** §6.3。
- **full FT の学習成果物をどの checkpoint format で保存するか。** mixed
  （und int8 + gen bf16）/ 両 half bf16 / gen half 再量子化 の 3 択。§6.4。
- **decoder 外の gen 側モジュール（`fm_head`、gen ViT、embedder、`*_norm_mot_gen`）を
  trainable に含めるか。** 含める方向を推奨するが未決定。§6.2。
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
  不足した場合のみ `scope: both`（LoRA 限定）を開く。
- **batch > 1 をいつ開くか。** padding-aware gen mask / varlen attention または
  streaming per-sample backward が前提。`separate_by_reference` だけでは開かない。
- **upstream issue #207 の mixed forward を検証・修正して 1 パス化する価値があるか。**
  2 パス設計で十分機能する見込みなので優先度は低いが、und 学習を将来入れるなら
  再評価する。

### 未測定事項の一覧（実測が無いもの／構造から推論しただけのもの）

新規（今回の設計再検証で判明したもの。いずれも**構造からの推論であって実測ではない**）:

1. **mixed checkpoint（und int8 + gen bf16）の推論ロード可否。** 構造上は通りそうだが
   **一度も試験されていない**。§6.4。
2. **`LayerOffloadConductor` がサブモジュール粒度のリストを受けられるか。** 未調査。
   受けられなければ half-eviction との合成に wrapper か per-layer 選択の新規実装が要る。
   §8.3.1。
3. **dequant 起点 full FT の学習品質影響。** 上記のとおり**測定不能な構造**である
   （比較 arm が存在しうる状況と、経路が必要になる状況が排他）。

既存（不変）: half-eviction の有効性（§8.3 の gate）、凍結 und での reference 忠実度
（§7.2）、ConvRot base の train / inference skew（§5.3）。

**ただし half-eviction の依存関係は強まった。** Phase 2b は weights + gradients だけで
32.4 GB を占め、und half 7.55 GiB の退避が唯一の余白であるため、この gate は
「Phase 1 の運用判断」から「**Phase 2b の VRAM 前提**」に格上げされた（§8.3.1、
§11 Phase 2b-0）。

---

## 13. References

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
