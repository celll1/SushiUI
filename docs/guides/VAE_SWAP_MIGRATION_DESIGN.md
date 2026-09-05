# VAE差し替えフルFT（latent移行）設計

Status: **P0〜P8 実装済み**。P8（SenseNova の pixel → latent）は §10.6 の受け入れ条件を
実重みで測定して満たしたため（測定値は §10.6 末尾）、ゲートを解除し
`TRAINING_FEATURE_ADVISORY["sensenova"]["vae_swap"] = "experimental"` に移した。
LoRA 系での拒否は他 arch と同様に残る。
フェーズごとの到達点は §11 の表を参照。

本書は §11 のフェーズ単位で実装・検証・コミットする
前提で書かれており、各フェーズの受け入れ条件を持つ。既存挙動の記述は全て
`file:line` を付す。引用のない記述は設計上の決定であり、「要検証」と付したものは
実装前に確認が必要な事実主張である。

**実装が設計を上書きした箇所は、本文の該当節を実装後の姿に書き換えてある。**
設計時の想定と実装が食い違った点のうち、後続フェーズの前提になるものは
各節に注記として残す（§5.1 末尾の3点、§8.4 末尾の lens、§10.5 など）。

要求（リポジトリ所有者）:

- 画像生成モデルの学習で、VAE を差し替えてフルFTを行えるようにする
  （SenseNova は pixel モデルなので latent モデルへの移行を含む）。
- 出力チャネル数が変わっても既存重みを完全には reset しない。
- VAE は原則、単一 safetensors なら同梱する。ModelLoader は VAE の有無を確認し、
  チャネル数の整合性を検証する。LoRA（亜種含む）の適用にも整合性チェックを入れる。
- 差し替え VAE の出所として、他のフルモデルに部分的に含まれる VAE も選択可能にする
  （モデル構築時に VAE のみを読む）。
- 学習側（フロント配線、対応 arch だけのドロップダウン表示）と生成側
  （ロード・生成）の両方を設計する。

---

## 0. 決定の一覧

| # | 論点 | 決定 |
|---|---|---|
| D1 | 全体方針 | SDXL 専用機構（`sdxl_vae_type`）を **arch 非依存に一般化**する。arch 別 opt-in の積み上げは採らない。arch ごとに宣言するのは「潜在 I/O の構造」（§5.1 `LatentIOSpec`）だけで、変形・出所解決・同梱・宣言・ゲートは共有層に置く |
| D2 | 対象 arch | 第1波: sd15, sdxl。第2波: zimage, krea2, ltx2。第3波: anima, flux2, lens, minit2i。最終波: sensenova（研究段階）。保留: acestep。対象外: ideogram4, minimax_h3, minimax_music3（§2） |
| D3 | 重み保存 | 重複チャネルは**チャネル軸で部分コピー**（現行 SDXL と同じ方針）。packed Linear は reshape→チャネル軸スライス→reshape（§6）。入力側と出力側の C の折込順序は arch ごとに**別々に**宣言する（anima は入力 outer / 出力 inner、§5.1）。**新規チャネルは入力側・出力側ともゼロ初期化**。現行の fresh Kaiming 初期化は廃止（SDXL も統一）。ゼロ初期化が走るのは**swap 適用時のみ**で、保存済みチェックポイントの再ロード・再開では保存済み重みを保持する（§6.2） |
| D4 | スケール補正 | 重みへのスケール補正は行わない。潜在の正規化（`scaling_factor` / `latents_mean,std` / BN）で入力分散を揃える。正規化は VAE 側が宣言する領域（生 C ch、または 2×2 パック後）で適用し、呼び出し側には常に生の C ch 潜在を見せる（§8.4）。効果は未測定である旨を UI には書かない |
| D5 | 縮小率の変更 | 第1〜3波では**空間/時間縮小率が arch 既定と異なる VAE を拒否**する。例外は SenseNova（pixel 1 → latent への移行が目的そのもの。空間縮小率は任意の値を受理する、D13） |
| D6 | VAE の出所 | 3種: (a) レジストリキー、(b) standalone VAE（単一ファイル / diffusers ディレクトリ）、(c) **他フルモデルからの抽出**。(c) は学習側の選択 API でのみ有効化し、生成側の override 候補列挙は現状のまま（§7） |
| D7 | 同梱 | 差し替え後の VAE は **`vae.` prefix・diffusers キー配置で同梱**（全 arch 共通）。ネイティブ VAE の同梱規約（sd15/sdxl の `first_stage_model.` LDM 配置）は変えない。差し替え run では `bundle_vae` の解決値を True にする。明示 False は `registry:` / `file:` 由来に限り尊重し、解決用の locator を書く。**`model:` 由来の非同梱は保存前（preflight）に拒否**する（§8.7） |
| D8 | 宣言メタデータ | 既存の `component.vae.*` ブロックを拡張して SSoT にする。`sushi.vae_type` / `sushi.in_channels` は SDXL アダプタが引き続き書き、リーダーは `component.vae.*` を優先し `sushi.*` にフォールバック。**第3の表は作らない**。VAE の「ネイティブ性」は **構造互換**（`struct_native`）と **潜在空間の同一性**（`identity_native`）の 2 キーに分ける（§5.2）。表示用の `provenance` と解決用の `locator` も別キー |
| D9 | 生成時の VAE 優先順位 | 現行の「ユーザー override > 宣言 > 埋め込み > arch 既定」を維持。ただし宣言値を正直に伝播させ、`_check_vae_compat` が構造不一致（チャネル・縮小率・ndim）で 400、同構造で hash 不一致の VAE で警告を出すようにする |
| D10 | LoRA 整合 | 書き側: 全 arch のアダプタメタデータに base の潜在 identity（C、family、hash、`struct_native`、`identity_native`）を記録。読み側: チャネル不一致は **hard refusal**（`lora_incompatible`）、同チャネルで VAE hash 不一致は **warning**。メタデータ無しのアダプタは、base が `struct_native="0"` なら **refusal**、`struct_native="1"` かつ `identity_native="0"`（同構造・別 hash の VAE）なら **warning**、`identity_native="1"` なら現状どおり無検査 |
| D11 | swap 済み base への LoRA 学習 | **許可**する（base は自己整合しており、LoRA は Linear のみ学習する）。拒否するのは「swap を要求しつつ method が full でない」場合のみ（現行どおり） |
| D12 | latent cache | namespace に `vae-<family>-<hash8>` トークンを**加算的**に追加。トークンの有無は `identity_native` で決める: `"1"` はトークン無し（既存 namespace を壊さない）、`"0"` は同チャネル・同 family の別 VAE でもトークン付き（hash で分離） |
| D13 | SenseNova | 生成側パッチ（潜在格子上）は **`P = 4` 固定**で `vae_scale_factor` に依らない。`fm_head` の `ps1(2)`/`ps2(2)`/`ps3(k)` の総拡大率 `4k` に対する最小の合法値であり、`conv1`・`ps1`・`ps2`・`dense_embedding` を無傷に保つ。ViT patch-embed カーネル = P/2 = 2、最終 PixelShuffle 係数 k = P/4 = 1、`fm_head.conv2` 出力 = C。1 トークンが覆う画素幅は `4 × vae_scale_factor`（8× → 32px、16× → 64px）で、トークン数は `vae_scale_factor` に比例して伸びる解像度で保存される。**任意の `vae_scale_factor` を受理**する。トークン幅・推奨解像度帯は VAE 選択時と生成時に提示し、黙って変えない。patch-embed は小さい標準偏差の切断正規分布で初期化、`fm_head.conv2` はゼロ初期化。`sensenova_train_fm_modules` を必須値化 |

---

## 1. 前提の訂正

設計の前提として、所有者の認識と現状の差分を先に固定する。

1. **「他モデルに含まれる VAE のみを読む仕組みは既にある」は半分だけ正しい。**
   存在するのは「今ロード中の自分のチェックポイント」から `first_stage_model.*` /
   `vae.*` を抜き出す機構で、6 arch にある（zimage/flux2 `model_loader.py:1293-1304`,
   `1592-1599`, `1743-1760`; lens `lens_loader.py:245-249`; anima `anima_loader.py:591-599`;
   krea2 `krea2/vendor/single_file.py:477-479`; minit2i `minit2i/vendor/single_file.py:133-135`;
   sd/sdxl `model_loader.py:297-334`）。汎用ヘルパも既にある:
   `split_prefixed_state_dict`（`common/single_file_format.py:103-122`、順序付き prefix リスト）、
   `reattach_embedded_weights`（`:337-360`、ゼロ件マッチで例外）。
   存在しないのは「ユーザーが選んだ第三者のフルモデル」から VAE だけ取り出す経路で、
   3箇所で意図的に塞がれている（`component_registry.py:426-439`,
   `generation_overrides.py:362-364`, `pipeline.py:1851-1901`）。§7 で扱う。
2. **MiniT2I は既に latent 対応済み。** `MMJiTConfig.vae_type` が `none`/`sdxl`/`flux1`
   を取り（`minit2i/vendor/mmjit.py:333-338`）、ローダが patch-embed conv の shape から
   自動推定する（`vendor/single_file.py:69-79`）。pixel/latent はチェックポイント単位の
   設定値であり、SenseNova 設計の直接の前例になる。
3. **現行 SDXL 機能は VAE を同梱していない**（`sdxl_adapter.py:374-392`）。理由は
   「SDXL の VAE コンバータが 4ch 構造を前提にしており別 VAE を誤変換する」。§8 で解消する。

---

## 2. アーキテクチャ別の実現可能性と採否

| arch | 潜在 C / 縮小率 | 変形対象（実名） | C の折込 | 判定 | 波 |
|---|---|---|---|---|---|
| sd15 | 4 / 8 | `unet.conv_in Conv2d(4→320)`, `unet.conv_out Conv2d(320→4)` | なし | 可能。`resize_unet_in_out` は arch 非依存で配線が無いだけ | 1 |
| sdxl | 4 / 8 | 同上 | なし | 実装済み（`sd_sdxl_ops.py:177-197`） | 1 |
| zimage | 16 / 8 | `all_x_embedder["2-1"] Linear(p²·f·C→dim)`, `all_final_layer["2-1"].linear Linear(dim→p²·f·C)`（`zimage_transformer.py:463-472`, `:518-530`） | あり、**C が最内** | 可能。4ch/16ch 両対応の前例あり（`model_loader.py:1458-1512`） | 2 |
| krea2 | 16 / 8 → 2×2 pack | `img_in Linear(64→6144)`, `final_layer.linear Linear(6144→64)`（`krea2/vendor/transformer.py:405,431-436,465`） | あり、C が最外（`krea2_pipeline_ops.py:140-147`） | 可能。per-channel `latents_mean/std` と `z_dim` を消費側が読む（`krea2_pipeline_ops.py:193-197`） | 2 |
| ltx2 | 128 / 空間32・時間8 | `proj_in Linear(128→inner)`, `proj_out Linear(inner→128)`, `audio_proj_in/out`（diffusers `transformer_ltx2.py:1110-1126,1155-1162,1313-1317`） | なし（`patch_size=1`） | 可能。置換 VAE は video VAE 必須、`17n+5` フレーム算術は時間比率固定 | 2 |
| anima | 16 / 8 | `x_embedder.proj[1] Linear((C+1)·p²·t→2048, bias=False)`, `final_layer.linear Linear(2048→p²·t·C, bias=False)`（`anima_models.py:480-491,507-513,1152-1157`） | あり、**入力側は C が最外、出力側は C が最内**（`anima_models.py:485-489`, `:1208-1212`）、**入力側のみ +1（padding mask）** | 可能・要注意。学習可能 `pos_embedder` が `max_img_h//patch_spatial` でサイズ決定（`:1160-1170`）⇒ 縮小率変更は不可（D5 で拒否） | 3 |
| flux2 | 32 / 8 → 2×2 pack | `x_embedder Linear(128→inner, bias=False)`, `proj_out Linear(inner→128, bias=False)` | あり、C が最外（入出力とも、`base_trainer.py:10482-10497`） | 可能・要注意。trainer が **2×2 パック後の 128ch** に `vae.bn.running_mean/var` を直読で適用（`flux2_ops.py:366-373`）。正規化領域は §8.4 | 3 |
| lens | 32 / 8 → 2×2 pack | `img_in Linear(128→inner)`, `proj_out Linear(inner→128, bias=True)`（`lens/vendor/transformer.py:426-451,533`） | あり、C が最外（入出力とも、`lens_pipeline_ops.py:283-296`） | 可能・要注意。同じく `_patchify` → `_bn_normalize` の順で VAE 内部 BN をパック後に適用（`lens_pipeline_ops.py:283-300`） | 3 |
| minit2i | 3 / 4 / 16 | `img_embedder.proj1 Conv2d(C,128,k=P,s=P,bias=False)`, `final_layer.linear Linear(hidden→P²·C)`（`mmjit.py:104-109,278-284`） | 入力側は conv（折込なし）、出力側は packed で C が最内（`mmjit.py:371-377`） | 実装済み（config 値）。差し替え＝`vae_type` 変更＋変形として同機構に乗せる | 3 |
| sensenova | pixel、VAE なし、格子 /32 | §10 | — | アーキ的に別物。研究段階の変更として最終波 | 4 |
| acestep | audio 64 / 1920 | `decoder.proj_in Conv1d(192→inner)`, `decoder.proj_out ConvTranspose1d(inner→64)`（`modeling_acestep_v15_turbo.py:1268-1305`）、`detokenize.proj_out`（`:894`）、凍結 RVQ | 時間 patchify のみ、C は 3 倍（`:1706`, `:1359`） | 可能だが凍結 RVQ の再学習/置換と `silence_latent` 資産（`acestep_ops.py:189-196,537-548`）の再生成が必要 | 保留 |
| ideogram4 | 32 / 8 | `input_proj`, `final_layer.linear` ×2 transformer | あり | **対象外**: full FT が拒否されている（`arch_capabilities.py:1028-1030`）、`Ideogram4FullParameterAdapter` 不在 | — |
| minimax_h3 | video 24 / 16, audio 32 | `proj_in/out`, `audio_proj_in/out` | あり | **対象外**: full FT が 3層で拒否（`arch_capabilities.py:1004-1006`, `full_parameter_trainer.py:226-246`） | — |
| minimax_music3 | 128 / 512 | — | — | **対象外**: 学習非対応 | — |

対象外 3 arch は `TRAINING_FEATURE_UNSUPPORTED[arch]["vae_swap"]` に理由付きで登録し、
フロントは served 値でコントロールを隠す（§9）。acestep は「保留」として同じ表に載せる。

### 2.1 横断的な制約（設計の根拠）

- **6 arch が `p²·C`（または `p²·t·C`）をトークン次元に折り込む。** 1:1 依存は ltx2 と
  2つの U-Net のみ。折込の順序は **outer / inner の 2 通り**あり、しかも**入力側と出力側で
  一致するとは限らない**（anima は入力 outer / 出力 inner、minit2i は入力 conv / 出力 inner、§5.1）。
- **潜在正規化は 3 方式**: (a) `scaling_factor`/`shift_factor`（sd15/sdxl/zimage）、
  (b) per-channel `latents_mean/std`（anima/krea2/ltx2）、(c) VAE 内部 BatchNorm
  （flux2/lens/ideogram4）。(c) は生の C ch 潜在ではなく **2×2 パック後の 4C ch に対して統計が
  定義されている**（flux2 `bn.running_mean` は 128 次元、`flux2_ops.py:366-373`; lens
  `lens_pipeline_ops.py:283-300`）。共有層の `normalize(latent, vae, spec)` は `spec` を無視して
  `shift_scale` しか実装していない（`vae_registry.py:166-168`, `:113-129`）。
  `_scale_shift` の `or 1.0` は flux2 で scale=1.0 を返し、これは `vae_store.py:55-58` が
  禁じている読み方である。第3波の前提として 3 方式と正規化領域を実装する（§8.4）。
- **縮小率はチャネルより深い依存を持つ。** trainer の形状リテラル `base_trainer.py:14476-14532`
  （`//8`, `//16`, `64`, `128`, `3`）、zimage `calculate_shift`（`zimage_ops.py:656-667`）、
  flux2 `_flux2_compute_empirical_mu_for_sample`（`:1138-1141`）、anima `pos_embedder`、
  ltx2 `17n+5`。チャネル変更は cache namespace `c<n>` で分離される（`latent_cache.py:122-126`）
  が縮小率変更は全キャッシュ再生成を強いる。⇒ D5。

---

## 3. 一般化元: 現行 SDXL VAE 移行機能

新機構が置き換える対象と、その中で保つ性質を列挙する。

- 設定: `sdxl_vae_type`（`param_defaults.py:2569-2571`, `routes.py:15697`, `openapi.yaml:21149-21151`,
  `api.ts:7536`, `TrainingConfig.tsx:4151-4168`）。DB カラムは無く `config_yaml` に一括保存
  （`database/models.py:826`）⇒ **新パラメータにマイグレーションは不要**。
- 変形: `resize_unet_in_out`（`sdxl_custom_arch.py:32-101`）。重複チャネルをコピー、新規は
  `nn.Conv2d` の既定初期化（Kaiming uniform）、縮小は `min()` で切り捨て、本体は無傷。
- 適用: `sd_sdxl_ops.py:177-197`。LoRA 拒否ゲート `:161-175`。
- 書き出し: `sdxl_adapter.py:40-72` `sushi_modelspec_metadata()` → `sushi.vae_type`, `sushi.in_channels`,
  `modelspec.architecture="sdxl-custom"`。カスタム VAE 非同梱（`:374-392`）。
- 読み込み: `model_loader.py:2352-2380`（`model_type == "sdxl"` のときだけ読む）→ `:2390-2396`
  `load_alt_vae` → `:2484-2500` `resize_unet_in_out` + `load_custom_convs_from_single_file`
  （shape 不一致時は静かに `False`、`sdxl_custom_arch.py:132-144`）。
- 再開: `base_trainer.py:3161-3252`。
- レジストリ: `components/vae_registry.py:30-45`（`sdxl`, `flux1` の 2 件、任意パス不可）。

保つ性質: チャネル部分コピー、本体無傷、`register_to_config` で config 同期、
full-FT-only ゲート、宣言メタデータによる再構築、cache namespace 分離。

解消する既知の限界（ブリーフ §1.6）: SDXL 限定、レジストリ 2 件のみ、非同梱、fresh random
初期化、`from_single_file` の `out_channels` 回避策の 3 段構え、再構築失敗のサイレント化、
`load_components` の custom-arch 非対応（`sd_sdxl_ops.py:43-100`）、`strict_validation` が
目的関数しか見ないこと（`train_runner.py:2590-2647`）、重みコピー計算のテスト不在。

---

## 4. 全体構造

```
core/models/components/
  wiring.py         ComponentWiringSpec に latent_io: LatentIOSpec を追加（表は増やさない）
  latent_io.py      [新規] resize_latent_io(): conv / packed_linear の変形（§6）
  vae_source.py     [新規] resolve_vae_source(): registry / standalone / extraction（§7）
  vae_registry.py   normalize(spec) に 3 方式を実装（§8.4）
common/single_file_format.py
                    component.vae.* の拡張キー、同梱 VAE の書き/読み（§8）
core/training/arch/base_arch.py
                    ArchHandler.apply_vae_swap(trainer, resolved) の共通実装（§8.1）
core/model_loader.py / pipeline_backends/*
                    宣言メタデータ → 変形 → 同梱 VAE ロード → wiring 畳み込み（§9）
core/extensions/lora_manager.py, core/adapters/session.py
                    base 潜在 identity ゲート（§9.4）
api/arch_capabilities.py
                    TRAINING_FEATURE_PARAMS["vae_swap"], TRAINING_FEATURE_UNSUPPORTED（§9.6）
```

共有層に置く理由は `wiring.py:8-14` の配置規則（コンポーネント層は生成・学習共有であり
`core/training/` 配下に置かない）に従うため。学習側 `core/training/components/vae_registry.py`
は re-export シムのまま（`:15-28`）。

---

## 5. 宣言データ

### 5.1 `LatentIOSpec`（`ComponentWiringSpec` の新フィールド）

arch ごとに「潜在に面するモジュール」と「C の折込レイアウト」を宣言する。これは
構造的事実であり、`wiring.py:35-52` の既存フィールド（`latent_channels`, `latent_packing`,
`vae_norm`）の隣に置く。**新しい表ではなく既存 spec の拡張**である。

入力側（潜在 → backbone）と出力側（backbone → 潜在）は**別の関数が別の順序でパックする**
ので、`kind` と `channel_order` は入出力で別フィールドに持つ。単一の `kind`/`channel_order` は置かない。

```python
@dataclass(frozen=True)
class LatentIOSpec:
    in_module: str            # 例 "unet.conv_in", "x_embedder", "img_in", "x_embedder.proj.1"
    out_module: str           # 例 "unet.conv_out", "proj_out", "final_layer.linear"
    in_kind: str              # "conv" | "packed_linear"
    out_kind: str             # "conv" | "packed_linear"
    in_channel_order: str     # in_kind == packed_linear のみ: "outer" | "inner"
    out_channel_order: str    # out_kind == packed_linear のみ: "outer" | "inner"
    pack_elems: int           # packed_linear のみ: p² (・t) (・f)。例 flux2/lens/krea2=4, anima=4(空間)×1(時間), zimage=4
    extra_in_channels: int    # 入力側に付く非潜在チャネル数（anima=1、他=0）
    in_repeat: int            # 入力側で C が何倍で入るか（acestep=3、他=1）
    out_bias: bool
```

宣言値（第1〜3波）:

| arch | in_module | out_module | in_kind / in_order | out_kind / out_order | pack_elems | extra | repeat |
|---|---|---|---|---|---|---|---|
| sd15/sdxl | `unet.conv_in` | `unet.conv_out` | conv / — | conv / — | — | 0 | 1 |
| zimage | `all_x_embedder.2-1` | `all_final_layer.2-1.linear` | packed_linear / inner | packed_linear / inner | 4 | 0 | 1 |
| krea2 | `img_in` | `final_layer.linear` | packed_linear / outer | packed_linear / outer | 4 | 0 | 1 |
| ltx2 | `proj_in` | `proj_out` | packed_linear / — | packed_linear / — | 1 | 0 | 1 |
| anima | `x_embedder.proj.1` | `final_layer.linear` | packed_linear / **outer** | packed_linear / **inner** | 4 | 1 | 1 |
| flux2 | `x_embedder` | `proj_out` | packed_linear / outer | packed_linear / outer | 4 | 0 | 1 |
| lens | `img_in` | `proj_out` | packed_linear / outer | packed_linear / outer | 4 | 0 | 1 |
| minit2i | `img_embedder.proj1` | `final_layer.linear` | conv / — | packed_linear / inner | P² | 0 | 1 |

順序の根拠（入力側と出力側を別々に読んで確定した値）:

| arch | 入力側 | 出力側 |
|---|---|---|
| zimage | `view(C,F_t,pF,H_t,pH,W_t,pW).permute(1,3,5,2,4,6,0)`（`zimage_transformer.py:592-597`、学習側 `batched_zimage_wrapper.py:239-244` も同一）⇒ inner | `view(F//pF,H//pH,W//pW,pF,pH,pW,out_ch).permute(6,0,3,1,4,2,5)`（`zimage_transformer.py:526-531`）⇒ inner |
| krea2 | `pack_latents`: `view(b,c,h/p,p,w/p,p).permute(0,2,4,1,3,5)`（`krea2_pipeline_ops.py:140-146`）⇒ outer | `unpack_latents`: `view(b,gh,gw,c,p,p).permute(0,3,1,4,2,5)`（`krea2_pipeline_ops.py:150-157`）⇒ outer |
| lens | `_patchify`: `view(b,c,h/2,2,w/2,2).permute(0,1,3,5,2,4)`（`lens_pipeline_ops.py:283-288`）⇒ outer | `_unpatchify`: `reshape(b,c4//4,2,2,h,w).permute(0,1,4,2,5,3)`（`lens_pipeline_ops.py:291-296`）⇒ outer |
| flux2 | `_flux2_patchify_latents_for_training`: `view(b,c,h/2,2,w/2,2).permute(0,1,3,5,2,4)`（`base_trainer.py:10482-10489`）⇒ outer | `_flux2_unpatchify_latents`: `reshape(b,c//4,2,2,h,w).permute(0,1,4,2,5,3)`（`base_trainer.py:10491-10497`）⇒ outer |
| anima | `Rearrange("b c (t r) (h m) (w n) -> b t h w (c r m n)")`（`anima_models.py:485-489`）⇒ outer | `rearrange(x, "B T H W (p1 p2 t C) -> B C (T t) (H p1) (W p2)")`（`anima_models.py:1208-1212`）⇒ **inner** |
| minit2i | `x_embedder.proj1` は Conv2d、パック無し（`mmjit.py:104-109`） | `unpatchify`: `reshape(b,gh,gw,p,p,c)`（`mmjit.py:371-377`）⇒ inner |
| ltx2 | `patch_size=1` ⇒ `pack_elems=1`、順序は無意味 | 同左 |

入出力の対称性は **arch ごとに両側のコードを読んで確定した事実であり、推定してはならない**。
新 arch を追加するときは、上表に入力側と出力側の両方の引用を足してから宣言を書く
（anima のように片側だけ読むと逆順を宣言する）。§6.6 のテストは入力側・出力側を別ケースで持つ。
ltx2 の in/out は 1:1（`patch_size=1`）なので `pack_elems=1` の packed_linear として同じコードで扱う。

実装で確定した3点:

1. **パス解決のルート。** 表の `unet.conv_in` / `unet.conv_out` は U-Net **の所有者**を
   ルートに取る綴りである。他 arch のパスは transformer 本体をルートに取るので、両者は
   同じルート型ではない。sd15/sdxl で `resize_latent_io` に素の U-Net を渡してはならない
   （`resize_unet_in_out` は `SimpleNamespace(unet=unet)` で包んで渡している）。
2. **`pack_elems` は文書ではなく計算に使う。** `resize_latent_io` は旧チャネル数を引数に取らず
   `C_old = in_features / P − e` で復元するため、`P` が誤っていると黙って別の要素を切り出す。
   **minit2i の `P` はチェックポイント単位の config 値**（latent 2 / pixel 16）であって定数ではなく、
   pixel 版の `final_layer.linear`（768 = 256·3）は `P=4` でも割り切れてしまうので誤りが表に出ない。
   P7 で解決済み: `MiniT2IArchHandler.resolve_wiring` がロード済み config
   （trainer 不在の preflight では `peek_io_config`）から `pack_elems`・
   `latent_channels`・`vae_scale_factor` を解決し、`apply_vae_swap` はその
   run 固有 spec を使う。`ComponentWiringSpec` の固定値は pixel 版のままである。
   `ResizeReport.old_in_channels` / `old_out_channels` を `wiring.latent_channels` と
   突き合わせる検算はその補助であって、代替にはならない。
3. `core/training/components/wiring.py`（再エクスポートのシム）は `LatentIOSpec` を
   再エクスポートしていない。学習側からこの型を import する必要が出た波で足す。

### 5.2 チェックポイント宣言メタデータ（D8）

`build_component_metadata` / `parse_component_metadata`（`single_file_format.py:129-185`）を
拡張する。既存キー `component.vae.type` / `.channels` / `.embedded` は意味を変えない。

| キー | 値 | 意味 |
|---|---|---|
| `component.vae.type` | レジストリキー or `"custom"` | family。cache namespace とプレビューデコーダの選択に使う |
| `component.vae.channels` | int | 潜在チャネル数 C。backbone の in_channels は tensor から sniff する（`component_registry.py:343-346,452-454`）ので別キーは持たない |
| `component.vae.embedded` | `"1"/"0"` | 同梱の有無 |
| `component.vae.prefix` | `"vae."` or `"first_stage_model."` | 同梱時の prefix（リーダーは無くても両方試す） |
| `component.vae.class` | `"AutoencoderKL"` 等 | 同梱 VAE の diffusers クラス名 |
| `component.vae.config` | JSON 文字列 | 同梱 VAE の `config.json` 全体。抽出元に config が無い LDM 形式なら変換後の値 |
| `component.vae.scale_factor` | int（例 `"8"`） | 空間縮小率 |
| `component.vae.scale_temporal` | int | 時間縮小率（画像 VAE は `"1"`） |
| `component.vae.norm` | `"shift_scale"/"per_channel"/"batchnorm"` | 正規化方式。§8.4 の `normalize(spec)` が読む |
| `component.vae.norm_pack` | int（`"1"` or `"2"`） | 正規化統計が定義される領域の空間パック係数。`"1"` = 生の C ch、`"2"` = 2×2 パック後の 4C ch（flux2/lens の BN）。backbone 側の `pack_elems` とは独立（§8.4） |
| `component.vae.provenance` | 文字列 | 出所の**表示用**文字列（`registry:flux1`, `file:<basename>`, `extracted:<model stem>`）。解決には使わない |
| `component.vae.locator` | 文字列 or 空 | 非同梱時の**解決用**参照。`registry:<key>` または `path:<絶対パス>`。同梱時は空。ロード時は locator 先の内容ハッシュを再計算して `component.vae.hash` と照合し、不一致・不在は拒否（§8.7, §9.1） |
| `component.vae.hash` | sha256 先頭 16 hex | 同梱/参照 VAE の**テンソル内容ハッシュ**。locator の重み検証に使う。cache / adapter は正規化設定を加えた `ResolvedVAE.latent_hash` を使う |
| `component.vae.struct_native` | `"1"/"0"` | **構造互換**: `channels` / `scale_factor` / `scale_temporal` / `latent_ndim` / `class` が arch 既定 VAE と全て一致するか。capability 判定、LoRA の hard refusal 境界、生成側 `_check_vae_compat` の hard/soft 判定に使う |
| `component.vae.identity_native` | `"1"/"0"` | **潜在空間の同一性**: 重みと正規化設定が base の VAE と一致するか。`"0"` が「swap 済み」の定義。cache namespace、アダプタ identity、warning 境界に使う。`struct_native="0"` ⇒ `identity_native="0"` は不変条件（逆は成り立たない） |

- 「arch 既定 VAE」は、その base を swap 無しでロードしたときに使われる VAE を指す
  （sd15/sdxl は base に同梱された `first_stage_model.`、他 arch は同梱 VAE またはレジストリ既定）。
  base が既に `identity_native="0"` を宣言している場合、その系列で作る成果物は `"0"` を引き継ぐ（§8.3）。
  宣言を持たず sniff が arch 既定と一致するチェックポイントは両キーとも `"1"` として扱う（現行経路）。
- ハッシュは学習側の VAE 解決時に 1 回だけ、state_dict のテンソルバイト列を安定順序で
  sha256 して得る。同梱時は生成側で再計算せずメタデータを信じる（改竄されない前提）。
  非同梱時は locator 先を読んだ時点で再計算し、`component.vae.hash` と一致しなければロード拒否する
  正規化設定も保存時の config と照合する（表示用 `provenance` は解決に使わない）。
- `ResolvedVAE.latent_hash` はテンソルハッシュに `scaling_factor` / `shift_factor` /
  `latents_mean` / `latents_std` / `batch_norm_eps` を加えて導出する。cache の `hash8` と
  アダプタ・ロード済みモデルの `vae_hash` はこの値を使う。既存の非nativeキャッシュは
  再生成され、旧アダプタのテンソルのみのhashとの差は警告となる。nativeキャッシュは不変。
- 同梱する config は出所の config ではなく、ロード後の VAE モジュールの完全な config。
  これにより config を持たない LDM 形式から変換した VAE も再構築できる。
- `sushi.vae_type` / `sushi.in_channels` は SDXL アダプタが引き続き書く（`sdxl_adapter.py:58-62`）。
  `component_registry._apply_component_hints`（`:375-408`）はこの 2 キーも読むように拡張し、
  `component.vae.*` を優先、`sushi.*` にフォールバックする。ブリーフ §5.3 が指摘する
  「レジストリ由来スタック全体を正す最小の修正」はこれで達成される。
- `modelspec.architecture="sdxl-custom"` は provenance 表示用に残す。読み手は作らない。

### 5.3 学習パラメータ

`backend/core/training/TRAINING_PARAMS_GUIDE.md` の Case B に従い、`TrainingRunCreateRequest`
にフィールドを足す（`_build_train_section` が YAML へ自動書出）。既定値は
`api/param_defaults.py` `TRAINING_DEFAULTS` のみに置く。

| キー | 型 / 既定 | 意味 |
|---|---|---|
| `vae_swap_source` | str / `""` | `""` = 差し替え無し。`registry:<key>` / `file:<path>` / `model:<path>` の 3 形式（§7.1） |
| `vae_swap_new_channel_init` | str / `"zero"` | `"zero"` のみ受理（D3）。将来の実験用に enum を持つが v1 は 1 値。UI には出さない |
| `bundle_vae` | 既存 / None | 既存のまま。resolver（`param_defaults.py:3048-3056`）が `vae_swap_source != ""` のとき True を返す |

`sdxl_vae_type` は**読み取り専用の後方互換エイリアス**とする: trainer の解決は
`config.get("vae_swap_source") or _legacy_from_sdxl_vae_type(config)` で、
`sdxl_vae_type in ("flux1",...)` を `registry:<key>` に写す。UI コントロール
（`TrainingConfig.tsx:4151-4168`）は新セレクタに置き換え、`sdxl_vae_type` は
`PARAM_KEYS` に残して古い run の編集を壊さない。`param_defaults.py` の `sdxl_vae_type` 既定は
保持する（削除すると `model_dump()` が変わり古い YAML の読戻しが変わる）。

「呼び出し側が意図的に設定したか」の判定は `request.model_fields_set` で行う
（`request.model_dump()` は全既定を実体化する: ADD_A_PARAMETER 失敗パターン #7）。

---

## 6. 重み保存アルゴリズム（`latent_io.resize_latent_io`）

### 6.1 契約

```python
def resize_latent_io(module_root: nn.Module, spec: LatentIOSpec,
                     new_channels: int, *, new_channel_init: str = "zero") -> ResizeReport
```

- `in_module` / `out_module` を `spec` に従って**新しい層に置換**し、重複チャネルをコピーする。
  本体には触れない（現行 `resize_unet_in_out` と同じ、`sdxl_custom_arch.py:32-101`）。
- 置換後、`register_to_config(in_channels=..., out_channels=...)` があれば呼ぶ。無い vendor
  モジュール（anima/lens/krea2 等）は各 arch が持つ `in_channels` 属性を更新する。
  zimage は `self.out_channels` を `unpatchify` が読む（`zimage_transformer.py:526-529`）ので必ず更新する。
- optimizer 作成**前**に呼ぶ（置換された Parameter を optimizer が掴む必要がある）。
  SDXL は `load_components` 内で行っており（`sd_sdxl_ops.py:177-197`）同じ位置で行う。
- `ResizeReport` に「コピーした要素数 / 新規要素数 / 置換モジュール名」を返し、trainer の
  ログと `strict_validation`（§8.6）が使う。

### 6.2 `conv`（sd15/sdxl/minit2i 入力/acestep）

現行アルゴリズムを維持し、初期化のみ変更する。

- 入力: `W_in: [hidden, C_old·r, k…]`（`r = in_repeat`）。`view(hidden, r, C_old, *k)` →
  `zeros(hidden, r, C_new, *k)` に `[:, :, :n]` をコピー（`n = min(C_old, C_new)`）→ `view` を戻す。
  `r=1` なら現行の `new_in.weight[:, :n] = conv_in.weight[:, :n]` と一致する。bias は hidden 次元なので全コピー。
- 出力: `W_out: [C_old, hidden, k…]` → `[:n]` をコピー。bias `[:n]` をコピー。
  `ConvTranspose1d`（acestep `proj_out`）は重みが `[in, out, k]` なので dim 1 でスライスする。
- **新規チャネルはゼロ**（weight・bias とも）。現行の Kaiming 初期化は廃止する（D3）。
  理由: 入力側の fresh random は step 0 で本体に学習分布外の信号を注入し、出力側の fresh random は
  予測にバイアスを載せる。ゼロなら step 0 の挙動は「未知チャネルを無視する / 未知チャネルへ 0 を
  予測する」と定義でき、勾配は下流の非ゼロ重みを通じて定義される。リポジトリの出力層ゼロ初期化の
  前例（`modeling_fm_modules.py:293-294, 451-452`）とも整合する。
  却下: チャネル平均複製（VAE 潜在チャネルは互換でない）、fresh random（上記）。
- ゼロ初期化が走るのは **swap を新規に適用する瞬間だけ**である。swap 済みチェックポイントの
  生成ロード・学習ロード・再開では、`resize_latent_io` の後に保存済み state_dict が新しい層を
  上書きし、保存済み重みが保持される（現行の SDXL 経路も resize 後に
  `load_custom_convs_from_single_file` が上書きする形になっている、`model_loader.py:2484-2500`）。
  これを回帰条件にする: 「swap 済みチェックポイントを再ロードした backbone の `in_module`/`out_module`
  重みが保存値と bit 同一」（§6.6-6、§11 P1）。
- SDXL の既存 swap 経路もこの実装に切り替える。**SDXL の挙動変更**であり CHANGELOG に書く。
  既存 run（Kaiming 初期化）と同じ設定で再学習しても学習結果は再現されない。保存済みの既存 run の
  再ロード・再開は上記の回帰条件により影響を受けない。

### 6.3 `packed_linear`（DiT 6 arch + ltx2）

packed 次元では「重複チャネル」は連続スライスではない。`W_in: [hidden, P·C_old]` を
`[:, :P·n]` でスライスすると、`outer` 順では先頭 `n` チャネル分の全要素（正しい）だが、
`inner` 順では先頭 `P·n` 要素＝空間位置 `s < n` の**全チャネル**を拾い、
チャネル `c ≥ n` の重みが `c < n` の位置に混入する。したがって必ず 3-D に戻してから
チャネル軸でスライスする。

入力側と出力側は**別々の順序宣言**（`in_channel_order` / `out_channel_order`）で、別々に演算する。
1 つの式で両側を扱わない（anima は入力 outer / 出力 inner）。

入力側のインデックス（packed index `k_in`、チャネル `c ∈ [0,C+e)`、パック内位置 `s ∈ [0,P)`）:

- `in_channel_order="outer"`: `k_in = c·P + s`（flux2/lens/krea2/anima/ltx2）
- `in_channel_order="inner"`: `k_in = s·(C+e) + c`（zimage）

出力側のインデックス（packed index `k_out`、`c ∈ [0,C)`）:

- `out_channel_order="outer"`: `k_out = c·P + s`（flux2/lens/krea2/ltx2）
- `out_channel_order="inner"`: `k_out = s·C + c`（zimage/anima/minit2i）

入力側 `W_in: [hidden, P·(C_old + e)]`（`e = extra_in_channels`、`order = in_channel_order`）:

```python
if order == "outer":
    Wv = W_in.view(hidden, C_old + e, P)          # [h, c, s]
    Wn = zeros(hidden, C_new + e, P)
    Wn[:, :n, :] = Wv[:, :n, :]                   # 潜在チャネル
    Wn[:, C_new:C_new + e, :] = Wv[:, C_old:C_old + e, :]   # extra（anima の padding mask）
    W_new = Wn.view(hidden, P * (C_new + e))
else:  # inner
    Wv = W_in.view(hidden, P, C_old + e)          # [h, s, c]
    Wn = zeros(hidden, P, C_new + e)
    Wn[:, :, :n] = Wv[:, :, :n]
    Wn[:, :, C_new:C_new + e] = Wv[:, :, C_old:C_old + e]
    W_new = Wn.view(hidden, P * (C_new + e))
```

入力 bias は hidden 次元なので全コピー（flux2/anima は `bias=False`）。

出力側 `W_out: [P·C_old, hidden]`、`b_out: [P·C_old]`（`order = out_channel_order`、入力側の値を流用しない）:

```python
if order == "outer":
    Wv = W_out.view(C_old, P, hidden); bv = b_out.view(C_old, P)
    Wn = zeros(C_new, P, hidden);      bn = zeros(C_new, P)
    Wn[:n] = Wv[:n];                   bn[:n] = bv[:n]
else:
    Wv = W_out.view(P, C_old, hidden); bv = b_out.view(P, C_old)
    Wn = zeros(P, C_new, hidden);      bn = zeros(P, C_new)
    Wn[:, :n] = Wv[:, :n];             bn[:, :n] = bv[:, :n]
W_new = Wn.view(P * C_new, hidden); b_new = bn.view(P * C_new)
```

出力側に `extra` は無い（anima は出力 `p²·t·C`、`anima_models.py:507-513`）。
minit2i は入力側が `conv`（§6.2）、出力側だけがこの `inner` 式を通る。

`pack_elems` の内訳（anima `r·m·n`、zimage `pF·pH·pW`）は `outer`/`inner` の判定にだけ
効き、内部順序はスライスに影響しない（C 軸と直交する軸をまとめて `P` として扱える）。
anima の出力 `(p1 p2 t C)` も同様で、`p1·p2·t` をまとめて `P`、C が最内である。

### 6.4 acestep（保留波の仕様のみ）

`proj_in` の in_channels 192 は `cat([src_latents, chunk_masks])`（`:1706`）→
`cat([context_latents, hidden_states])`（`:1359`）による `[src C | mask C | noisy C]` の
連続 3 ブロックなので、`in_repeat=3` の `conv` として `view(inner, 3, C_old, k)` でスライスする。
`chunk_masks = ones(B,T,64)`（`acestep_ops.py:543`）と `silence_latent [1,750,64]` は C に追従して
再生成が必要、`detokenize.proj_out Linear(hidden→64)`（`:894`）は出力 `conv` 相当、凍結 RVQ は
64 次元固定で置換/再学習が必要。この 3 点の設計は本書の範囲外（保留）。

### 6.5 ideogram4（対象外だが記録）

2 つの transformer それぞれの `input_proj`/`final_layer.linear`（`ideogram4/vendor/transformer.py:479-500,526-530`）
を `outer` packed_linear として変形し、forward の `in_channels` hard-assert（`:546-549`）を config
更新で満たす。full FT が開放されたとき、`LatentIOSpec` を 2 本持つ（`in_module` をリストにする）拡張で対応できる。

### 6.6 検証（重みコピー計算のテスト — 現状不在）

`tests/latent_io_test.py`（新規）に**性質テスト**を置く。これが「strided vs contiguous」の
バグを捕まえる唯一の手段である。

入力側と出力側は**別ケース**にする。入力側だけを見るテストは anima の出力順序の誤りを通してしまう
（入力 outer / 出力 inner）。各ケースは arch ごとにパラメタ化し、**その arch の実際の pack 関数と
unpack 関数**（§5.1 の根拠表に引用したもの）を直接呼ぶ。テスト内で pack/unpack を書き直さない。

1. 入力側・拡張の等価性: 乱数潜在 `x: [B, C_old, H, W]` を用意し `y_old = L_old(pack_arch(x))`。
   `L_new = resize(L_old, C_new ≥ C_old)`、`x' = cat([x, zeros(B, C_new-C_old, H, W)], dim=1)` として
   `y_new = L_new(pack_arch(x'))`。**`y_new == y_old`（allclose、fp32）**。`pack_arch` は
   `pack_latents` / `_patchify` / `_flux2_patchify_latents_for_training` / anima `x_embedder` の
   `Rearrange` / zimage の `view(...).permute(...)` / minit2i は恒等（conv）。
   `in_channel_order` を取り違えると必ず落ちる。
2. 入力側・縮小の等価性: `C_new < C_old` で `y_new == L_old(pack_arch(x[:, :C_new]))` に対応する出力行の一致。
3. 出力側・拡張と縮小: 乱数 `h: [B, T, hidden]` に対し
   `unpack_arch_new(L_new_out(h))[:, :n] == unpack_arch_old(L_old_out(h))[:, :n]`（allclose、fp32）、
   `C_new > C_old` なら残りチャネルは 0。`unpack_arch` は `unpack_latents` / `_unpatchify` /
   `_flux2_unpatchify_latents` / anima `final_layer` 後の `rearrange` / zimage `unpatchify` /
   minit2i `unpatchify`。`out_channel_order` を取り違えると必ず落ちる。anima はこのケースで
   入力側と異なる順序を通ることを確認する。
4. anima: extra チャネル（padding mask）の重みが入力側の `C_new` 位置に移っていること。
5. sd15/sdxl: 現行 `resize_unet_in_out` の重複部分と bit 同一（回帰）。
6. 再ロードの保持: `resize` 後に保存した state_dict を、再度 `resize` した層にロードして
   bit 同一（§6.2 の回帰条件。ゼロ初期化が保存済み重みを上書きしないことの確認）。

いずれも GPU 不要・モデル不要（`nn.Linear`/`nn.Conv2d` を spec 通りの形で生成する）。
pack/unpack 関数がモジュールメソッドである arch（anima、zimage、minit2i）は、当該メソッドを
最小 config のモジュールから呼ぶか、関数として切り出してから使う。

---

## 7. VAE の出所（`vae_source.resolve_vae_source`）

### 7.1 3 形式

`vae_swap_source` の値:

| 形式 | 例 | 解決 |
|---|---|---|
| `registry:<key>` | `registry:flux1` | 表A `vae_store.VAE_REGISTRY`（`vae_store.py:59-111`）で `class/default_repo/scaling` を引く。表B（`components/vae_registry.py:30-45`）は表A へ統合し、`preview` フィールドを表A に移す（表の数を減らす方向の変更） |
| `file:<path>` | `file:M:/model/vae/flux2_vae.safetensors` | standalone: 単一ファイル（LDM 素キー or diffusers キー）/ diffusers ディレクトリ。現行 `load_override_vae`（`pipeline.py:1851-1901`）の 2 分岐と同じ判定を共有関数化して使う |
| `model:<path>` | `model:M:/model/flux2/flux2-dev.safetensors` | **抽出**: ヘッダのみで `has_backbone` を確認し、`split_prefixed_state_dict(sd, ["vae.", "first_stage_model."])` で VAE 部分だけをロードする（§7.2） |

戻り値 `ResolvedVAE(module, latent_channels, scale_factor, scale_temporal, ndim, norm, norm_pack,
vae_class, config_dict, family, content_hash, provenance, locator, struct_native, identity_native)`。
学習・生成の両方がこの 1 型を消費する。`provenance` は表示用文字列、`locator` は解決用
（`registry:<key>` / `path:<絶対パス>`、`model:` 由来は `None`）で、§5.2 の同名キーにそのまま書かれる。
`struct_native` / `identity_native` は §5.2 の定義に従い、解決時に arch 既定 VAE と比較して確定する。

### 7.2 抽出経路と、塞いでいる 3 つのゲートの扱い

| ゲート | 現状の意図 | 新設計 |
|---|---|---|
| `component_registry.py:426-439` | フルモデルのサブ重みを standalone VAE と誤認しない（モデル一覧の分類） | **変えない**。一覧の分類意味論は維持 |
| `generation_overrides.py:362-364` `classify_vae_candidate` | 生成 override 候補からフルモデルを除外 | **変えない**。生成側 override にフルモデル抽出は提供しない（毎生成で数 GB のヘッダ走査＋抽出を行う理由が無い。必要なら学習側で一度抽出して同梱すればよい） |
| `pipeline.py:1851-1901` `load_override_vae` | diffusers dir / 素 safetensors の 2 分岐 | `file:` の共有関数を使うよう置き換える。`model:` 分岐は足さない |

抽出は学習側の新 API `GET /training/vae-sources` が返す第 3 グループ（"フルモデルから抽出"）
からのみ到達する。列挙は `_override_scan_dirs()`（`routes.py:9468-9487`）と同じ根を走査し、
`has_backbone and has_vae` のヘッダを候補にする（sniff は `component_registry.py:339-341,445-447`
と同じ `decoder.conv_in.weight.shape[1]` を再利用）。

抽出時の prefix は `["vae.", "first_stage_model."]` の順で試す（krea2/minit2i は `vae.`、
sdxl/sd15/zimage/flux2/anima/lens は `first_stage_model.`、ブリーフ §3.5）。両方ゼロ件なら
`reattach_embedded_weights` と同様に例外（`single_file_format.py:337-360`）。

### 7.3 スケーリング/正規化メタデータの解決順序

`generation_overrides.py:139-151` の規則「観測がフィールド単位で勝ち、宣言は欠落のみ埋める」を
そのまま採る。ただし観測できるものとできないものを分ける。

| フィールド | 観測（優先） | 宣言（欠落時） | 最終手段 |
|---|---|---|---|
| `latent_channels` | `decoder.conv_in.weight.shape[1]` | `component.vae.channels` / `config.json` | 拒否 |
| `scale_factor` | `encoder` の downsample 段数（`down_blocks` 数から算出）。**未検証**: 対応する VAE クラスごとに、down-block 数と実縮小率の対応を確認してから実装する | `config.json` / 表A | 拒否 |
| `norm` | `bn.running_mean` が在れば `batchnorm`、`config.latents_mean` が在れば `per_channel` | `component.vae.norm` | `shift_scale`（`scaling_factor` が宣言されている場合のみ） |
| `norm_pack` | `batchnorm` のとき `bn.running_mean.numel() / latent_channels` の平方根（flux2: 128/32=4 → 2）。平方数でなければ拒否。`batchnorm` 以外は 1 | `component.vae.norm_pack` | 拒否 |
| `scaling_factor`/`shift_factor` | 観測不能 | `config.json` → `component.vae.*` → 表A `canonical_latent_scaling`（`vae_store.py:120-132`） | **拒否**。`0.18215` の当て推量（`:117`）は使わない |
| `latents_mean/std` | 観測不能 | `config.json` | 拒否 |
| `vae_class` | LDM 素キー → `AutoencoderKL`、`bn.*` → `AutoencoderKLFlux2`。5D conv（video VAE）のクラス判別は**未検証**: 対応クラス（ltx2/anima の VAE）ごとに識別キーを確認してから実装する | `component.vae.class` | 拒否 |
| dtype | ロード地点の既定（`pipeline.py:1856-1861` 踏襲） | `vae_dtype`（学習）/ 現行ロジック（生成） | — |

「拒否」は学習 preflight で `ValueError`、生成ロードで `NotFoundError`/`ValidationError`
（`api/error_handlers.py`）。数値を推測して続行する経路は作らない。

### 7.4 family 互換ゲート（学習 preflight、hard）

| 条件 | 判定 |
|---|---|
| `ndim` が wiring の `latent_ndim` と異なる（画像 VAE を anima/ltx2 に等） | 拒否 |
| `scale_factor` / `scale_temporal` が wiring と異なる | 拒否（D5、SenseNova 除く） |
| wiring `vae_norm="batchnorm"`（flux2/lens）で置換 VAE が BN を持たない | P5（§8.4）まで拒否、P5 以降は解除 |
| 置換 VAE が `norm="batchnorm"`（`norm_pack=2`）で、置換先 arch の wiring が `batchnorm` でない（sd15/sdxl/zimage 等に flux2/lens 由来 VAE） | P5（§8.4）で学習側を解除。sd15/sdxl は生成側 `custom_sampling.py` のスカラ読みのため 556febc6 で一時的に拒否が残っていたが、efe85eda で 12 箇所が共有 `normalize`/`denormalize` 経由になり解除済み（P7 の受入行が挙げていた「sdxl に 32ch BN VAE」はそのコミットが担当した） |
| krea2 で `latents_mean/std` も `scaling_factor` も無い | 拒否 |
| sensenova で `scale_factor` が任意の値 | 受理（D13。`P = 4` 固定なので重み形状は `scale_factor` に依らない。トークン幅 `4 × scale_factor` と推奨解像度帯を候補に添えて返す、§10.2） |
| `vae_swap_source` が `model:` かつ `bundle_vae` が明示 False | 拒否（D7、§8.7。生成時に解決できない成果物を作らない） |
| `latent_channels` が現状と同じかつ重み・正規化設定が同じ | 「差し替え無し」として no-op（swap 扱いにしない、`identity_native="1"`） |
| `latent_channels` が現状と同じで `content_hash` が異なる | swap として扱う。`resize_latent_io` はコピーのみ（新規チャネル無し）、`struct_native="1"`, `identity_native="0"` |

---

## 8. 学習側

### 8.1 arch handler への畳み込み

`ArchHandler`（`base_arch.py`）に共通実装のメソッドを 1 つ足す。arch 別 override は不要。

```python
def apply_vae_swap(self, trainer, resolved: ResolvedVAE) -> None:
    trainer.vae = resolved.module
    report = resize_latent_io(trainer.backbone, self.wiring.latent_io, resolved.latent_channels)
    trainer.wiring = self.wiring.replace(latent_channels=resolved.latent_channels,
                                         vae_norm=resolved.norm,
                                         vae_norm_pack=resolved.norm_pack)
    trainer.vae_identity = resolved   # cache namespace / metadata / strict_validation が読む
```

`vae_norm_pack` は `ComponentWiringSpec` に足す int フィールド（既定 1、flux2/lens は 2）。
`vae_norm` と同じく §8.4 の `normalize(spec)` だけが読む。

- `trainer.backbone` は arch により `unet` / `transformer` を指す既存属性（各 ops が持つ）。
- `wiring.replace` は既存の graft helper（`wiring.py:47-50`）。`wiring.py:22-26` が約束して未実装の
  「ロード時に実際の値を畳み込む」処理はこれで実装される。
- `sd_sdxl_ops.py:177-197` の SDXL 固有ブロックはこの共通実装の呼び出しに置き換える。
- flux2 の `_flux2_patchify_latents_for_training`（`base_trainer.py:10482-10489`）と `//8` バリデータ
  （`:14516-14522`）は C を `trainer.wiring.latent_channels` から読むよう変更（縮小率は D5 で不変）。
  `flux2_ops.py:366-373` が patchify と BN 適用を 1 続きで書いている箇所は、P5 で
  「§8.4 の `normalize`（生 32ch を受け取り生 32ch を返す）→ backbone 用 patchify」の 2 段に分ける。
  置換 VAE が BN を持たない場合（P5 以降）は `normalize` が `resolved.norm` の方式で動く。
  krea2 の `z_dim` 読み（`krea2_pipeline_ops.py:193-197`）、anima の `latents_mean/std`（`anima_ops.py:330-332`）、
  ltx2（`ltx2_ops.py:450-456`）は `resolved` の値を `trainer.vae.config` 経由で読むので、置換 VAE が
  同じ config 属性を持てば変更不要。持たない場合は §7.4 で拒否済み。

### 8.2 swap 済み base からの学習（ブリーフ §1.6-8 の解消）

`sd_sdxl_ops.load_components`（`:43-100`）を含む全 arch の学習ローダは、base の
`component.vae.*`（`sushi.*` フォールバック）を読み、`struct_native="0"` なら**モデル構築前に**
`in_channels` を宣言値で指定して構築する。`identity_native="0"`（swap 済み）なら VAE を
同梱の `vae.` prefix から、非同梱なら `locator` から（hash 照合付き、§8.7）ロードする。
同構造・別 hash の base（`struct_native="1"`, `identity_native="0"`）は構築を変えずに VAE だけ差し替わる。
これは生成側ローダ（§9.1）と同じ関数 `load_declared_latent_io(path) -> ResolvedVAE | None`
を共有する。SDXL の `from_single_file(num_in_channels=C, out_channels=C)` +
`load_custom_convs_from_single_file` の 3 段回避（`model_loader.py:2419-2500`）は、
学習側では `resize_latent_io` 後に state_dict を厳密ロードする 1 段に置き換える。
生成側の diffusers 経路は §9.1。

### 8.3 メソッドゲート（D11）

- 「`vae_swap_source != ""` かつ `training_method != "full_finetune"`」→ `ValueError`（現行
  `sd_sdxl_ops.py:161-175` と同文言の一般化）。テスト `training_method_gate_test.py:285-314` の
  存在しないキー `"flux"`（`:286`）は `registry:flux1` に直す。
- base が既に `identity_native="0"` で swap を要求しない LoRA/full → **許可**。アダプタメタデータに
  base の潜在 identity を書く（§9.4）。
- base が `identity_native="0"` でさらに別の swap を要求 → 許可（2 段階移行）。両キーの基準は
  arch 既定 VAE（§5.2）なので、`identity_native` は `"0"` を引き継ぎ、`struct_native` は新 VAE と
  arch 既定の比較で決め直す。

### 8.4 共有正規化層の完成（第3波の前提）

`vae_registry.normalize(latent, vae, spec)` / `denormalize`（`:166-168`）に `spec.vae_norm` の
3 方式を実装する:

- `shift_scale`: `(z - shift) * scale`（現行 `normalize_latent` `:113-129`）
- `per_channel`: `(z - mean.view(1,-1,…)) / std.view(1,-1,…)`（anima/krea2/ltx2 の各 ops にある式）
- `batchnorm`: `(z - bn.running_mean) / sqrt(bn.running_var + eps)`（`flux2_ops.py:366-373`, `lens_pipeline_ops.py:283-300`）

**正規化が定義される領域（`spec.vae_norm_pack`）**: `batchnorm` の統計は生の C ch ではなく
2×2 パック後の 4C ch に対して定義されている（flux2 の `running_mean` は 128 次元で、trainer は
`_flux2_patchify_latents_for_training` を通してから引いている、`flux2_ops.py:366-373`; lens は
`_patchify` → `_bn_normalize`、`lens_pipeline_ops.py:283-300`）。これは VAE の性質であって
backbone の性質ではない。契約は次のとおり:

- `normalize` / `denormalize` は**常に生の C ch 潜在 `[B, C, H, W]` を受け取り、生の C ch 潜在を返す**。
- `vae_norm_pack == 2` のとき、内部で `view(B,C,H/2,2,W/2,2).permute(0,1,3,5,2,4).reshape(B,4C,H/2,W/2)`
  （C 最外、`base_trainer.py:10482-10489` と同じ順序）→ 統計を適用 → 逆変換
  （`base_trainer.py:10491-10497` と同じ）を行う。統計の並びは VAE がこの pack 順で学習されたことに
  依存するので、`norm_pack` の pack 順は VAE 側の固定仕様として `vae_class` ごとに 1 つに決め、
  `LatentIOSpec` の `in_channel_order` から導かない。
- backbone 側のパック（`LatentIOSpec.pack_elems` / `in_channel_order`）とは**独立**である。
  flux2/lens では両者がともに 2×2 で一致するため現行コードは 1 回の patchify で兼ねているが、
  sdxl に flux2 由来の VAE を当てる場合は backbone がパックしないので、VAE 側で pack → BN → unpack
  して 32ch の正規化済み潜在を得る必要がある。逆に flux2 に `shift_scale` の VAE を当てる場合は
  `normalize` は pack せず、backbone の patchify だけが走る。
- P5 の受け入れ条件（ネイティブ VAE の潜在が置き換え前後で bit 同一）は、pack → 統計 → unpack →
  backbone patchify の往復が `view`/`permute` のみで数値演算を含まないことで満たされる。

flux2/lens/anima/krea2/ltx2 の ops は自前の式をこの関数呼び出しに置き換える。`_scale_shift` の
`or 1.0`（`vae_store.py:55-58` が禁じる読み）は削除し、`scaling_factor is None and norm == "shift_scale"`
を例外にする。これは swap の有無に関係なく正しくなる変更で、単独コミットにする。

**`LENS_WIRING.vae_norm` は現状 `"shift_scale"` だが、lens の実装は BatchNorm である**
（`lens_pipeline_ops.py:299-310, 359`）。宣言が実装と食い違っているため、§7.4 の BN ゲートは
lens に対して発火せず、BN を持たない VAE が構造検査を通過して encode 経路で落ちる。
P5 で `vae_norm="batchnorm"` に直す（`vae_norm_pack=2` は P2b で宣言済み）。それまで lens は
§7.4 の arch ゲートで swap 自体を拒否しているので実害は無いが、wave 3 で lens を開放する前に
必ず直すこと。

この節は **P5** で実装する。P2〜P4 では BN 系 VAE を非 BN arch に当てる swap（およびその逆）を
§7.4 で拒否するので、P2 の第1波は `shift_scale` 系 VAE のみで検証する（§11）。

**実装時の訂正（P5）**: 「flux2 と lens は同じ BN の式」という前提は成り立たなかった。flux2 は
`sqrt(running_var + eps)` を **buffer の dtype（fp32）で計算してから 1 回だけキャスト**し、lens の
`_bn_normalize` は **var を先に潜在の dtype に落としてから** sqrt していた（lens の
`_bn_denormalize` と vendor `_decode` は flux2 側の順序）。fp16/bf16 では両者は一致せず（実測:
チャネルの 12〜18%、bf16 で std の相対差 最大 0.63%）、1 つの関数が両方に bit 同一にはなり得ない。
共有層は精度の高い前者（＝lens 自身の decode・vendor と同じ）を採り、lens の encode だけが
fp16/bf16 で最大 1.0%（fp32 では bit 同一）動く。同じ理由で denormalize は `x * std + mean` に
統一し、vendor の `x / (1/std) - (-mean)` は採らない。per-channel 系は 3 arch とも
`torch.as_tensor(...).to(dtype)` で 3 dtype とも bit 同一（Qwen-Image の定数では二重丸めも起きない）。
ltx2 の `(z - mean) * scaling_factor / std` は演算順ごと維持する。

### 8.5 latent cache namespace（D12）

`base_trainer.py:10554-10557` の

```python
vae_type = getattr(self, "sdxl_vae_type", None) if arch == "sdxl" else None
```

を `vae_type = None if self.vae_identity.identity_native else f"{family}-{hash8}"` に置き換える。
判定キーは **`identity_native`**（潜在空間の同一性、§5.2）であって `struct_native` ではない。
`build_cache_namespace`（`latent_cache.py:72-134`）は変更しない: `identity_native="1"` は `None` で
トークン無し（`base_arch.py:431-433` の不変条件どおり加算的）、`"0"` は `vae-<family>-<hash8>`
＋既存の `c<n>`、`dt<dtype>`。同 family 同 C の別 VAE（fine-tune 版 SDXL VAE など）は
`struct_native="1"` だが `identity_native="0"` なのでトークンが付き、素の SDXL VAE で作った
キャッシュとは hash で分かれる（同一 namespace への混入を起こさない）。
hash 計算は §5.2 の 1 回のみで、`vae_store.vae_identity()`（`:135-165`）の `vae_path` は
provenance 文字列に使う。

### 8.6 `strict_validation` の拡張（ブリーフ §1.6-10）

`train_runner.py:2590-2647` に以下を追加する（full FT では `True` に上書きされる、
`training_config.py:986-989`）:

1. `resize_latent_io` 後の `in_module` の入力幅 == `pack_elems·(C + extra)·repeat`、
   `out_module` の出力幅 == `pack_elems·C`。
2. `trainer.vae` を 1 枚のダミー画像で `encode` した潜在の `shape[1] == C`、`ndim == latent_ndim`、
   空間縮小率 == `vae_scale_factor`。
3. base 再構築（§8.2）で state_dict の欠落/余剰キーがゼロ。**失敗は `sys.exit(1)`**。
   現行の print + traceback のみ（`model_loader.py:2497-2500`, `sdxl_custom_arch.py:132-144`）は廃止。

### 8.7 同梱（D7）

- 差し替え run の full FT 保存: 各 arch アダプタの保存関数で、VAE を **`vae.` prefix・diffusers
  キー配置**で `save_single_file_state`（`single_file_format.py:218-265`）に渡す。LDM 変換は
  通さない（SDXL の 4ch 前提コンバータ問題 `sdxl_adapter.py:374-392` はこれで回避）。
  `component.vae.*` を §5.2 のとおり書く。
- ネイティブ VAE の同梱は現行規約のまま（sd15/sdxl は `first_stage_model.` LDM 配置、
  `BUNDLE_VAE_DEFAULTS_BY_ARCH` `param_defaults.py:2956-2961`）。swap 済み sdxl は A1111/ComfyUI で
  読めない（`conv_in` が 4ch でない）ので LDM 互換を保つ理由が無い。
- `bundle_vae` resolver（`:3048-3056`）: `vae_swap_source != ""` → 既定 True。明示 False の扱いは
  出所で分かれる:
  - `registry:<key>` / `file:<path>` → 尊重する。`component.vae.embedded="0"`、
    `component.vae.locator` に `registry:<key>` / `path:<絶対パス>`（`file:` は解決時に絶対化した
    パス）、`component.vae.hash` を書く。表示用 `provenance` は解決に使わない（basename では元ファイルを
    特定できない）。生成ロード・学習ロードは locator を解決し、内容ハッシュを再計算して `hash` と
    照合する。不在・不一致は**ロード拒否**（黙って 4ch にフォールバックしない）。
  - `model:<path>` → **preflight で拒否**（`ValueError`、run 作成時の検証で返す。§7.4）。
    抽出 VAE は抽出元フルモデルの中にしか存在せず、生成時に抽出元の存在を前提にしないので
    解決可能な locator を持てない。保存が成功して後で読めない成果物を作らない。
    却下: 保存時に standalone VAE を別ファイルとして書き出し locator で指す案 — 成果物が
    2 ファイルになり、片方だけ移動されると同じ「読めない成果物」になる。同梱（既定 True）が
    その要件を 1 ファイルで満たす。
- 10GiB シャーディング（`:57`）: VAE テンソルは他と同じく `dedup_tensors` → シャード分割に乗る。
  リーダー `load_component_state_dict`（`:312-330`）は prefix で全シャードを横断するので変更不要。
- LoRA 保存には VAE を同梱しない（LoRA は base を変えない。§9.4 の identity だけ書く）。

---

## 9. 生成側

### 9.1 ロード

共通関数 `load_declared_latent_io(path)` が単一ファイルのヘッダから `component.vae.*`
（`sushi.*` フォールバック）と backbone の `in_channels`（`_UNET_CONVIN_SUFFIXES` の `shape[1]`、
`component_registry.py:343-346,452-454`）を読み、以下を返す:

- 宣言が無く sniff も arch 既定と一致 → `None`（現行経路、変更なし）。
- 宣言 C と sniff C の不一致（anima は `+1`、inpaint U-Net は `2C+1` を許容）→ **ロード拒否**。
- 一致し `identity_native="0"` → `ResolvedVAE`（同梱なら `vae.`/`first_stage_model.` から、非同梱なら
  `locator` から。locator 先の内容ハッシュが `component.vae.hash` と一致しなければ拒否、§8.7）。
  `struct_native="0"` なら backbone を宣言 C で構築する。`struct_native="1"` かつ
  `identity_native="0"` は構築を変えず VAE だけを置き換える。

各 arch ローダは backbone を C で構築してから state_dict をロードする。diffusers 経由の
sd15/sdxl は現行 `from_single_file(num_in_channels=C, out_channels=C)` + `resize_unet_in_out` +
`load_custom_convs_from_single_file`（`model_loader.py:2419-2500`）を維持するが、
`load_custom_convs_from_single_file` の shape 不一致は例外にする。`model_type == "sdxl"` ゲート
（`:2358`）は `model_type in ("sd15", "sdxl")` へ、その他 arch はそれぞれのローダで同じ関数を呼ぶ。

ロード後、`pipeline._sushi_wiring = _WIRING_BY_ARCH[arch].replace(latent_channels=C, vae_norm=…, vae_norm_pack=…)`
を必ず設定し、`current_model_info`（`pipeline.py:1397-1405`）に `latent_channels`, `vae_type`,
`vae_hash`, `vae_provenance`, `vae_struct_native`, `vae_identity_native` を載せる。`/models/current`（`routes.py:10097-10108`）、
`describe_vae`（`generation_overrides.py:154-224`）、`_fold_baseline`（`component_registry.py:599-604`）
は静的定数ではなく `pipeline._sushi_wiring` を読む。`ModelLoadSection.tsx:196-201` はサーバ値を
既に優先するのでフロント変更は `api.ts:239-255` `ModelInfo` への `latent_channels` 宣言のみ。

### 9.2 arch 判定の堅牢化（ブリーフ §5.1）

`detect_model_type()` の SD/SDXL サイズ閾値（`model_loader.py:961-968`）の**前**に:

1. `component.vae.*` / `sushi.*` / `modelspec.architecture` があれば `model_type` を確定。
   swap 済みモデル（`identity_native="0"`）はこの段で必ず確定する。
2. `model.diffusion_model.label_emb.0.0.weight` の有無で sdxl/sd15 を判定（SDXL の
   added-cond 埋め込み）。**未検証**: このキー名はリポジトリ内のコンバータの対応表
   （`training/adapters/state_dict_converter.py:24`、`label_emb.0.0.weight` ↔
   `add_embedding.linear_1.weight`）にあるが、実チェックポイントのヘッダで prefix 込みの
   完全なキー名を確認していない。P0 の実装時に手元の SDXL / SD1.5 チェックポイントで確認してから
   採用し、確認できなければこの段を入れずに 1 → 3 とする。
3. どちらも無いときだけ現行のサイズ閾値。

差し替え後のファイルサイズは両方向に動く（16ch VAE 同梱で増、fp8 export で減）ため、
サイズ判定に落ちる非標準 SDXL が `sd15` に分類され宣言が読まれない事故を塞ぐ。

### 9.3 生成経路の残り（ブリーフ §5.3）

| 箇所 | 変更 |
|---|---|
| ライブプレビュー `generation_utils.py:193` → `taesd.py:99-104` | `preview_decoder_for(vae_type)`（`vae_registry.py:56-57`）で family からプレビューデコーダを選ぶ。family 不明かつ C∈{16,32} は RGB 射影（`taesd.py:558-615`）、それ以外はプレビュー無効＋`add_warning(code="preview_unavailable")` |
| PiD `generation_overrides.py:457-461` | `!= 4` は事実（PiD は 4ch SDXL 参照 VAE 用デコーダ）なので残す。§9.1 で `latent_channels` が正直になるため 16ch SDXL は 400 で拒否される（現状は無言のゴミデコード） |
| inpaint 9ch ゲート `custom_sampling.py:4534` | `== 9` → `== 2*latent_channels + 1` |
| `keep_hot.compute_model_key`（`:125-146`） | `vae_path` override と `vae_hash` をキーに含める |
| `custom_sampling.py:2173-2189` `height // 8` | `pipeline._sushi_wiring.vae_scale_factor` を読む（値は不変だが定数を 1 箇所に寄せる） |

### 9.4 アダプタ整合ゲート（D10）

**書き側**: `sushi_modelspec_metadata()`（`sdxl_adapter.py:40-72`）を共有層
`core/adapters/base_identity.py` に一般化し、全 arch の LoRA/LyCORIS/full 保存メタデータに次を書く:

| キー | 値 |
|---|---|
| `sushi.base.latent_channels` | int |
| `sushi.base.vae_type` | family |
| `sushi.base.vae_hash` | 重みidentityと正規化設定から導出する `ResolvedVAE.latent_hash`（16 hex）。旧アダプタはテンソルハッシュのみを持つ場合があり、その差は警告となる |
| `sushi.base.vae_struct_native` | `"1"/"0"`（§5.2） |
| `sushi.base.vae_identity_native` | `"1"/"0"`（§5.2） |

`AdapterSpec`（`spec.py:135-143`）には `options["base_latent"]` として載せ、既存規約どおり
明示アクセサ `base_latent_identity()` を付ける。`schema_version` は上げない（任意キーの追加）。

**読み側（diffusers 経路、sd15/sdxl）**: `lora_manager._read_lora_header`（`:911-940`）→
`detect_adapter_fields`（`:425-482`）が上記キーを `fields["base_latent_channels"]` 等に載せ、
`load_lora_weights` の前に DoRA 拒否（`:1120-1139`）と同型のゲートを置く:

| アダプタ側 | ロード済みモデル側 | 判定 |
|---|---|---|
| `latent_channels` あり、不一致 | — | **refuse** `lora_incompatible`（`with_error_code(RuntimeError)`） |
| `latent_channels` 一致、`vae_hash` 不一致 | — | warning `lora_base_vae_mismatch`（`_lora_warn`） |
| メタデータ無し | `vae_struct_native="0"` | **refuse** `lora_incompatible`。理由: SushiUI が作る swap 済み base 用アダプタは必ずメタデータを持つので、無メタデータのものは別の潜在空間で学習された可能性が高い。証明はできないが、サイレント破損（適用率 100%、shape 不一致ゼロ、ブリーフ §6.2）の方が回復不能 |
| メタデータ無し | `vae_struct_native="1"` かつ `vae_identity_native="0"` | warning `lora_base_vae_unknown`。同構造の別 VAE（fine-tune 版 SDXL VAE 等）はチャネル配置が一致しており、無メタデータのアダプタは base VAE で学習された通常のアダプタである可能性が高い。§13-6 の境界（同形状で identity が異なるものは警告）に従う。無言では通さない |
| メタデータ無し | `vae_identity_native="1"` | 現状どおり無検査 |

境界の根拠は 2 キーで分ける: hard refusal は**構造**（`struct_native`、チャネル配置が物理的に異なる）、
warning は**同一性**（`identity_native`、同配置で潜在分布が異なりうる）。

**読み側（`AdapterSession` 経路、他 11 arch）**: `session._canonicalize`（`:304`）で spec を
得た直後、同じ表で `AdapterIncompatible(code="lora_incompatible")`（`:96`）を投げる/警告する。
DiT の patch embedder を LoRA 対象にしている arch（krea2 `proj` スコープ `krea2_lora.py:193-211`、
anima `x_embedder`/`final_layer` `anima_lora.py:42,52`、lens `img_in`/`proj_out`
`lens_adapter.py:130-131`）では shape 不一致が本物として出るので既存の `SHAPE_MISMATCH`
（`session.py:105-111`）も同時に働く。それは「部分適用」ではなく「不一致」として扱われるべきなので、
identity ゲートを shape ゲートの**前**に置く。

`GET /loras` の応答（`detect_adapter_fields`）に `base_latent_channels` / `base_vae_struct_native` /
`base_vae_identity_native` を載せ、フロントの LoRA 一覧はロード済みモデルの `latent_channels` と
比べて非互換を灰色表示する。

### 9.5 override 優先順位（D9）

`plan_overrides`（`generation_overrides.py:609`）の順序は変えない。`_check_vae_compat`
（`:445-510`）はロード側記述を `pipeline._sushi_wiring` から取るので、swap 済みモデルに 4ch override
を当てると 400、正しい 16ch は通る（現状は逆、ブリーフ §5.2）。同 C 別 hash は
`vae_override_warning`。同梱 VAE 付きモデルにユーザー override を当てる場合も警告を 1 件出す
（`code="vae_override_replaces_bundled"`）。

### 9.6 ロード時警告の配達

`ModelLoader` は生成の外で走るので素の `add_warning()` は捨てられる（`generation_status.py:223-224`）。
`generation_overrides.py:403-442` の `_warn` + `_capture_warnings` の型を写し、ロード時警告は
`pipeline._sushi_load_warnings` に溜めて**初回生成の `start_generation` 後に再生**する。

### 9.7 capability と フロントエンド

- `arch_capabilities.py`: `TRAINING_FEATURE_PARAMS["vae_swap"] = ["vae_swap_source"]`、
  `TRAINING_FEATURE_LABELS["vae_swap"]`、`_add_training_feature_unsupported` で
  ideogram4 / minimax_h3 / acestep / sensenova（最終波まで）/ 未実装波の arch に理由を登録。
  第2・3波の arch は実装コミットで登録を外す（ABSENT MEANS SUPPORTED、`:247-250`）。
  SenseNova 波では `_add_training_required_value("sensenova", "sensenova_train_fm_modules", True, …, unless={"vae_swap_source": ""})`。
- `TrainingConfig.tsx`: 新コンポーネント `VaeSwapSourceSelector` を
  `!unsupportedTrainingFeature("vae_swap") && trainingMethod === "full_finetune"` で描画
  （`:855-860` の既存ヘルパ）。`isSDXLModel` 等の述語で capability を判定しない（`:766-816` の規約）。
  現行の「(SDXL only)」ラベル付きコントロール（`:4151-4168`、実際は全 arch で描画）は削除。
- `GET /training/vae-sources?arch=<arch>`（openapi 先行）: 3 グループ
  `registry` / `standalone` / `extract_from_model` を、各候補の `latent_channels`, `scale_factor`,
  `ndim`, `norm`, `compatible: bool`, `reason` 付きで返す。§7.4 の family 互換判定はサーバで行い、
  UI は `compatible=false` を選択不可・理由表示にする。`arch=sensenova` では各候補に
  `token_pixel_width`（= `4 × scale_factor`）と `resolution_band_px` を加え、`VaeSwapSourceSelector` が
  候補名の隣に表示する（§10.2 の決定: 解像度帯を黙って変えない）。
- 生成側 `ModelLoadSection`: `vae_identity_native=false` のとき「VAE: <provenance> (<C>ch)」のバッジ。
  `VaeOverrideSelector.isVaeCompatible()`（`:32-44`）は変更不要（サーバの `latent_channels` が正直になる）。
- `ModelInfo`（`api.ts:239-255`）に `latent_channels?: number`, `vae_struct_native?: boolean`,
  `vae_identity_native?: boolean`, `vae_provenance?: string` を宣言。

---

## 10. SenseNova pixel → latent（最終波・研究段階）

本波は配線ではなく研究段階の変更である。§10.6 の受け入れ条件を満たすまで
`TRAINING_FEATURE_UNSUPPORTED["sensenova"]["vae_swap"]` を外さない。

### 10.1 現状の事実

- pixel-space、VAE なし（`sensenova_pipeline_ops.py:1-2`, `sensenova_ops.py:1155`,
  `wiring.py:185-189`）。サンプリングは `randn(B,3,H,W)` → Euler → `clamp(-1,1)`
  （`sensenova_pipeline_ops.py:117,1111-1112,1141`）。
- 1 ステップに 2 つの patchify（`:619-648`）: `patchify(img, 32)` → `[B,(H/32)(W/32),3072]` が
  flow matching の状態、`patchify(img,16)` → ViT。ViT 側は 16×16 の `patch_embedding` の後に
  `dense_embedding = Conv2d(1024, 4096, k=2, s=2)` で **2×2 マージ**する（`modeling_neo_vit.py:135-137`、
  `sensenova_pipeline_ops.py:621` 経由）。1 トークンが覆う画素幅は ViT パッチ幅 × 2 = 32 で、
  flow matching 側の 32 と一致する。
- 出力側 `fm_head`（`ConvDecoder`）は `ps1(2) → conv1 → ps2(2) → conv2 → ps3(8)`
  （`modeling_fm_modules.py:580-598`）で、総拡大率は `2·2·8 = 32`。
- 形が変わるテンソルは実質 2 つ: `fm_modules.vision_model_mot_gen.embeddings.patch_embedding.weight
  [1024,3,16,16]`（`modeling_neo_vit.py:132-134`）と `fm_modules.fm_head.conv2.weight/bias
  [192,256,3,3]`（`192 = 3·8²`）。約 1.2M パラメータ。588 decoder Linear（約 99.6%）、norm、
  embeddings、両 RoPE、`timestep_embedder`/`noise_scale_embedder`、`dense_embedding`、
  `ConvDecoder.conv1`、`ps1`/`ps2`、理解タワー全体は形状不変（§10.2 の選択で維持される）。
- 学習済み位置テーブルは無い（3 機構すべて解析計算、`modeling_qwen3.py:606-610`,
  `modeling_neo_chat.py:502-507`, `modeling_neo_vit.py:144-167`）。
- `patch_size`/`downsample_ratio` は両タワー共有スカラ（`modeling_neo_chat.py:191-194`）で、
  参照画像経路も読む（`sensenova_pipeline_ops.py:267-271, 298-301`）。`rotary_emb`/`rotary_emb_hw` は両 MoT 半分で共有。
- `sensenova_train_fm_modules` は既定オフで、16 テンソルは本リポジトリで一度も最適化されていない
  （`SENSENOVA_TRAINING_DESIGN.md:105-107`）。

### 10.2 決定: 生成側パッチ P = 4 固定とトークン数の保存

flow matching 側のパッチ幅（潜在格子上）は **`P = 4` で固定**し、`vae_scale_factor` に依らせない。

構造的な制約は 1 つだけである。出力側 `fm_head` の `ps1(2) → conv1 → ps2(2) → ps3(k)`
（`modeling_fm_modules.py:580-598`）は総拡大率が `4k` で、`k` は PixelShuffle の係数なので正の整数
でなければならない。したがって合法な `P` は `4k`（4, 8, 12, …）に限られ、**`P = 4`（`k = 1`）は
その最小値**である。現行経路の 2×2 マージ（`dense_embedding`）と `ps1(2)`/`ps2(2)` を維持すると、
`P` から次が決まり、`P = 4` ではいずれも VAE に依らない定数になる:

- ViT patch-embed カーネル幅（潜在格子上）= `P/2` = **2**（マージ後に 1 トークン = 4 潜在画素）
- `fm_head.ps3` の係数 `k = P/4` = **1**（`ps3` は恒等）
- `fm_head.conv2` の出力チャネル = `C·k²` = **C**

形が変わるテンソルは §10.1 の 2 つ（patch-embed `[1024, C, 2, 2]`、`conv2 [C, 256, 3, 3]`）に
限られ、`conv1 [1024, 1024, 3, 3]`、`ps1`、`ps2`、`dense_embedding [4096, 1024, 2, 2]` は
**どの `vae_scale_factor` でも無傷**である。16× VAE に追加の重み手術は要らない。

VAE によって変わるのは **1 トークンが覆う画素幅** `4 × vae_scale_factor` だけである:
8× VAE で 32px（現行 pixel-space の幾何と同一）、16× VAE で 64px。トークン数は
`vae_scale_factor` に比例して伸びる解像度で保存される: 潜在格子 128×128（ViT patchify 64×64、
マージ後 32×32 = 1024 トークン）は 8× で 1024px、16× で 2048px に対応し、`fm_head` の出力は
32×32 → `ps1` 64×64 → `ps2` 128×128 → `ps3(1)` 128×128 = 潜在格子で不変。上流の
約 4096 トークン帯（生成側の推奨解像度帯、`core/pipeline_backends/sensenova.py:429-440`）は
8× で 2048² 付近、16× で 4096² 付近に移る。**画像に対するタイルの相対サイズは両者で同一**
なので、RoPE の h/w index、`compute_noise_scale`（`sensenova_pipeline_ops.py:133-144`、トークン数の
関数）、`_calculate_dynamic_mu`（`modeling_neo_chat.py:451-479`、`image_seq_len` の関数）は
全て学習済みレンジに留まる。

負担は transformer から VAE に移る: 4×4×C の潜在タイル 1 枚が再構成する画素領域は 8× で
32×32、16× で 64×64 であり、制限要因は VAE の再構成品質であって transformer ではない。
どちらの `vae_scale_factor` についても品質の主張はしない（§10.6-5）。

却下: **1 トークンが覆う画素幅を 32 に固定する**（`P = 32 / vae_scale_factor`）— 8× 以外の
あらゆる縮小率で `k = P/4` が非整数になり（16× で `k = 0.5`）、ヘッドが実際には収容できる
VAE を拒否することになる。「1 トークン = 32px」は設計上の選択であって構造的制約ではない。
却下: `ps1`/`ps2` の係数や ViT のマージ率を可変にして画素幅 32 を保つ — `conv1` の入力チャネル
`input_dim // 4` と `dense_embedding` のカーネルが変わり、保存できる重みが減る。

`NeoChatConfig` に `gen_patch_size`（= 4）/ `gen_in_channels` を追加し、生成側 patchify/unpatchify と
ViT patch-embed（カーネル 2）、`fm_head`（`k = 1`）だけがこれを読む。`patch_size`/`downsample_ratio` は
理解タワー・参照前処理専用に残す（§10.1 の共有スカラを切る）。`rotary_emb_hw` は共有のままでよい
（h/w index は格子から算術で決まり、格子は保存される）。

**決定: 解像度帯は VAE 選択で黙って変えない。** `vae_scale_factor` はモデル選択の帰結であって
ユーザーが表明した意図ではないので、トークン幅 `4 × vae_scale_factor` と、それに対応する
推奨解像度帯（現行 3〜5 MP の帯を `(vae_scale_factor / 8)²` 倍した画素数）を次の 2 箇所で提示する:

- 学習側・VAE 選択時: `GET /training/vae-sources?arch=sensenova`（§9.7）の各候補に
  `token_pixel_width`（= `4 × scale_factor`）と `resolution_band_px`（帯の下限・上限を画素数で）を
  載せ、`VaeSwapSourceSelector` が候補名の隣に表示する。
- 生成側・生成時: `_sensenova_common_params` の帯外警告（`core/pipeline_backends/sensenova.py:429-440`、
  `code="sensenova_resolution"`）の閾値を `3.0 × (s/8)²`〜`5.0 × (s/8)²` MP に置き換え、
  文面にトークン幅とその要求のトークン数を含める。latent 版チェックポイントのロード時には
  `code="sensenova_token_grid"` の警告を 1 件、§9.6 の `pipeline._sushi_load_warnings` に積み、
  初回生成で再生する（`add_warning` を再利用し、新しい配達機構は作らない）。
  `current_model_info`（§9.1）に `token_pixel_width` を載せ、`ModelLoadSection` のバッジに出す。

### 10.3 決定: 初期化

- **patch-embed `[1024, C, 2, 2]`（`P/2 = 2`、`vae_scale_factor` に依らない）: 切断正規分布、
  `std = 1/sqrt(C·2²)`**（anima `PatchEmbed.init_weights` `anima_models.py:494-496` と同じ規約）。
  却下: (C) チャネル平均複製 — 「潜在チャネルは交換可能」という前提が VAE 潜在では成り立たず、
  さらに 16×16→2×2 のカーネル再標本化が重なる。(D) 擬似逆合成 — オフライン当てはめが要り、
  デコーダの非線形性により高ノイズ域で悪化しうる。(B) ゼロ — 全トークンが入力内容に依らず
  bias のみの同一値になり、ヘッドが非ゼロになる step 1 まで本体は入力内容を一切見ない。
  小さい標準偏差の乱数は本体へ入る信号の大きさを bias と同程度に抑えつつ、step 0 から
  内容依存の特徴を本体に流す。上流への勾配が step 0 でゼロなのはヘッドのゼロ初期化に
  よるもので（次項）、patch-embed の初期化方式では変わらない。
- **`fm_head.conv2 [C, 256, 3, 3]` と bias: ゼロ初期化**（`k = P/4 = 1`、`ps3` は恒等、
  出力 `C`。`vae_scale_factor` に依らない）。理由: step 0 で `x_pred = 0` となり、初期状態が入力に依存しない
  定義済みの値になる（fresh random は `x_pred` に入力依存の任意のバイアスを載せる）。
  リポジトリの出力層ゼロ初期化慣習（`modeling_fm_modules.py:293-294, 451-452`）と整合する。
  このとき `v = (x_pred - z)/(1-t) = -z/(1-t)` であり、`t→1` での発散は
  `(1-t).clamp_min(t_eps)`（`modeling_neo_chat.py:655`）でのみ抑えられる。ゼロ初期化は
  この発散を避ける手段では**ない**（`x_pred` の値に依らず `-z/(1-t)` の項は残る）。
  **ゼロヘッドは step 0 で `conv2` より上流（`conv1` 以前、588 Linear、patch-embed）への勾配を
  ゼロにする**。`conv2` 自身の勾配は入力（`ps2` 出力）に比例して非ゼロなので、step 1 以降は
  上流にも勾配が流れる。§10.6 はこれを受け入れ条件で測る。
  却下: (D) ランダム — 初期 `x_pred` が入力依存の無意味な値になり、初期損失の大きさが初期化の
  乱数に依存する。(B) エンコーダ擬似逆 — 当てはめパスと PixelShuffle 展開基底での
  合成が要る。研究的には最有力だが v1 では採らず、`sensenova_latent_head_init: "zero"|"encoder_pinv"`
  の enum を予約して後続実験の口だけ残す（v1 は `"zero"` のみ受理）。(E) ピクセルヘッド保持＋後段
  エンコーダ — latent モデルにならない。
- `ConvDecoder.conv1`、`ps1`、`ps2`、`dense_embedding`、両 embedder、588 Linear は無傷
  （§10.2 で `P = 4` に固定したことで、どの `vae_scale_factor` でも保証される）。

### 10.4 決定: `noise_scale` と潜在の正規化

トークン数が保存されるので `compute_noise_scale` の式と `noise_scale_base_image_seq_len`、
`noise_scale_embedder` は**変更しない**。潜在は VAE の正規化（§8.4）で単位スケール付近に揃える。
RGB `[-1,1]` と単位分散潜在の分散差が `noise_scale` 較正に与える影響は未測定であり、
§10.6 の smoke で `noise_scale` 値が学習済みレンジ（`:733` 「1024px で 4」、latent 版では
1024 トークン、すなわち `128 × vae_scale_factor` px 四方で 4）に入っていることを確認する。
較正の再導出（ブリーフ §4.7 (i)）と `noise_scale_mode` 変更（(iii)）は、smoke 後の実験結果で判断する
（本書では決めない）。

### 10.5 変更点一覧

- 緩めるガード: `_assert_pixel_head_fm_decoder`（`sensenova_ops.py:1048-1084`）、`vae_encode` の
  `shape[1] != 3`（`:1816`）、`train_step` の `images.shape[1] != 3`（`:1905`）、`vae_decode` の
  `NotImplementedError`（`training/arch/sensenova.py:126`）、`SENSENOVA_WIRING.latent_channels=0`
  （`wiring.py:187`）。
- **トークン幅 `4 × vae_scale_factor` 由来の配線**（画素幅 32 のリテラルを全て置き換える。
  リテラル 32 は残さない）:
  - `TOKEN_GRID_ALIGN = 32`（`sensenova_pipeline_ops.py:38`）と、それを既定引数に取る
    `align_to_grid`/`normalize_resolution`（`:68-76`）: ロード済み wiring の `pixel_align` を
    引数で受ける。§9.1 の `pipeline._sushi_wiring = SENSENOVA_WIRING.replace(latent_channels=C,
    vae_scale_factor=s, pixel_align=4·s, …)` が値の出所（`ComponentWiringSpec.pixel_align`、
    `wiring.py:249`）。`smoke.py:426` の重複定数も同じ値を引く。
  - 学習側 `pixel_align = 32`（`core/training/arch/sensenova.py:28`）: `apply_vae_swap`（§8.1）が
    `trainer.wiring.pixel_align = 4·s` を畳み込み、bucketing（`bucketing.py:642, 707`）はそれを読む。
  - `% 32` 整除チェック（`sensenova_ops.py:1818-1819`, `:1914-1915`, `sensenova_pipeline_ops.py:411-414`）:
    画素に対しては `% (4·s)`、潜在格子に対しては `% 4`。
  - 解像度帯の警告（`core/pipeline_backends/sensenova.py:429-440`、現在 3〜5 MP のリテラル）:
    `(s/8)²` 倍にスケールする（§10.2 の決定）。スケールしなければ 16× VAE での 4096² 生成が
    毎回帯外と警告される。
- リテラル `3` の全箇所（ブリーフ §4.2 の編集リスト）を `gen_in_channels` に置換。
- `trainer.vae` を `ResolvedVAE.module` に、`vae_encode`/`vae_decode` を実 VAE に。
- 生成側: `randn(B,C,H/s,W/s)`、Euler 更新は潜在上、最終 `clamp` は廃止して `vae.decode`。
  `_style_capture`（`:120-130, 849-873`）、img2img/inpaint init（`:1224, 1284-1285`）、RePaint
  マスクブレンド（`:1114-1116`）は潜在で行う（マスクは `s` 倍縮小）。
- 参照条件付け経路（理解タワー、ImageNet 正規化、`vendor/utils.py:153-159`）は**触らない**。
- `sensenova_train_fm_modules` を required value 化（§9.7）。再開時のグループ数変化
  （294 vs 310）は既存の per-group 先頭 prefix remap（`SENSENOVA_TRAINING_DESIGN.md:119-121`）で吸収。
- 保存: `fm_modules` の新形状テンソルはそのまま書かれる。`component.vae.*` を書き、VAE を `vae.` で同梱。
  int8 export（`sensenova_full_finetune_save_format`）は decoder Linear のみ量子化するので影響なし。
- 生成ローダ（`pipeline_backends/sensenova.py`）は `component.vae.*` を読んで `gen_patch_size`（= 4）/
  `gen_in_channels` を設定し、`component.vae.scale_factor` を wiring の `vae_scale_factor` と
  `pixel_align = 4·s` に畳み込む（任意の値を受理、D13）。`vae_override` の拒否
  （`arch_capabilities.py:607`）は latent 版のみ解除し、override 候補は `_check_vae_compat`（§9.5）で
  `scale_factor` 一致を要求する（`pixel_align` が変わる override は 400）。
- ViT `patch_embedding` のカーネルを 2、`fm_head.conv2` の出力を `C` で構築する（§10.2、
  `vae_scale_factor` に依らない）。`dense_embedding` と `ps1`/`ps2` は触らない。

**実装で確定した点（P8）**: 実装が §10.5 の想定と食い違った箇所を、後続の実験が
前提にできる形で残す。

1. **共有 `resize_latent_io` は使えない。** 変形はチャネル軸のスライスではなく
   *再構築*（ViT patch-embed はカーネルも 16×16 → 2×2 に変わる）なので、
   `SENSENOVA_WIRING.latent_io` は `None` のままで、`SenseNovaArchHandler.apply_vae_swap`
   が丸ごと override し `core/models/sensenova/latent_space.apply_latent_geometry` を呼ぶ。
   `SENSENOVA_WIRING` の `latent_channels=0` / `vae_scale_factor=1` も pixel 版の記述として
   残し、run 固有の値は P7 の minit2i と同じく `resolve_wiring` が答える（§10.5 は
   「`latent_channels=0` を緩める」と書いていたが、緩めるのは定数ではなく解決経路である）。
2. **`ComponentWiringSpec.pixel_align` は存在しない。** `pixel_align` は `ArchHandler` の
   クラス属性で、still 画像の bucketing は `arch.pixel_align` だけを読む。SenseNova は
   これをロード済みツリーから読む property にした（`gen_patch_size × gen_vae_scale_factor`）。
3. **`vae_scale_factor` は config キーではなくスタンプ。** 数値の住所は
   `component.vae.scale_factor`（§5.2）1 箇所であるべきなので、`NeoChatConfig` に足したのは
   `gen_patch_size`/`gen_in_channels` の 2 つだけで、縮小率はローダが
   `transformer.gen_vae_scale_factor` に書く。ローダは config ブロックと component ブロックの
   不一致（チャネル数・patch）をロード前に拒否する（`_assert_declared_latent_geometry`）。
4. **`_assert_pixel_head_fm_decoder` は緩める必要が無かった。** これはヘッドの*種類*
   （ConvDecoder か deep/plain か）を見ており、latent 版も ConvDecoder のままである。
5. **`_t2i_predict_v` は `image_size` の意味を変えず、任意引数 `token_hw` を足した。**
   `image_size` は画素、生成格子は潜在なので、両者を同じ引数で表すと黙って形が狂う。
   `SenseNovaPrefix` は `gen_size`（潜在格子の W,H）を持ち、denoise ループはそれを使う。
6. **`sensenova_train_fm_modules` の required value を宣言するには `arch_capabilities` の
   不変条件を method 単位にする必要があった。** 同じキーが LoRA では
   `TRAINING_FEATURE_UNSUPPORTED` に載っており、旧来の assert は method スコープを見ずに
   両表の同居を禁じていた。スコープが交わらないなら同居できる、に直した。
   要求は「rebuild が実際に走るとき」だけで、既に latent 版の base を学習し直す run
   （層は学習済み）には課さない。
7. RePaint のマスクは画素解像度でぼかしてから潜在格子へ area-average する
   （§10.5 の「s 倍縮小」の具体化）。羽根の*画素*幅が要求どおりに保たれる。
8. 推奨解像度帯のリテラル（3〜5 MP）は `latent_space.resolution_band_mp` に移し、
   生成側の警告とテストが同じ式を読む。

### 10.6 受け入れ条件（研究段階のゲート）

1. §6.6 相当の性質テストは適用不能（変形が部分コピーでない）。代わりに「形状不変テンソル
   （`conv1`、`dense_embedding`、588 Linear を含む）が bit 同一で保存・再ロードされる」テストを置く。
2. 形状の一致: `128 × vae_scale_factor` px 四方（8×: 1024px、16×: 2048px）の入力で flow matching 側の
   トークン数が 1024、ViT 側のマージ後トークン数が 1024、`fm_head` の出力が `[B, C, 128, 128]`
   （潜在格子）であること。8× と 16× の両方で確認し、形が変わるテンソルが両者で同一形状
   （patch-embed `[1024, C, 2, 2]`、`conv2 [C, 256, 3, 3]`）であること。
3. 3 ステップ smoke: `sensenova_train_fm_modules=True`、bf16、有限 loss、`noise_scale` が学習済みレンジ内
   （トークン数 1024 で 4、§10.4）。
   有限 loss だけでは足りず、次を同じ smoke で記録する:
   - `t` の端点付近（`t = 1 - t_eps` と `t = t_eps`）で `v` のノルムが有限であること
     （`(1-t).clamp_min(t_eps)` の下で `-z/(1-t)` が有界であることの確認）。
   - step 0 で `conv2` の勾配ノルムが非ゼロ、`conv2` より上流（`conv1`、patch-embed、decoder Linear の
     1 つ）の勾配ノルムがゼロであること（§10.3 のゼロヘッドの帰結どおり）。
   - step 1 以降で上流の勾配ノルムが有限かつ非ゼロであること（ヘッドが非ゼロになって勾配が
     届いたことの確認）。
4. 1 枚生成が VAE decode まで通り、出力が NaN でない。`normalize_resolution` が `4·s` の格子に
   スナップし、16× VAE で 4096² の要求に `sensenova_resolution` 警告が出ないこと（§10.5）。
5. 上記を満たしても**品質の主張はしない**。8× と 16× の優劣も主張しない。品質は所有者が実データで判断する。

#### 実測（`M:/model/sensenova/sensenova_int8.safetensors`、42層 MoT、gen 側 294 Linear）

条件 1・2 は CPU テストで担保済み。条件 3・4 を実重みで実行した結果を記録する。
`registry:sdxl`（4ch）と `registry:flux1`（16ch）の両方、いずれも 8×。
変形後: `patch=4`, `vit_patch=2`, `head_shuffle k=1`, `token_pixel_width=32`、
`patch_embedding → [1024, C, 2, 2]`、`fm_head.conv2 → [C, 256, 3, 3]`（初期値の非ゼロ要素 0）。

| 項目 | sdxl (4ch) | flux1 (16ch) |
|---|---|---|
| loss（3 step、t=0.15/0.50/0.85） | 0.864994 / 2.501813 / 28.172358 | 1.142786 / 3.282179 / 37.014191 |
| `noise_scale`（1024 トークン） | **4.0** | **4.0** |
| ‖v_pred‖（t=0.05 → 0.95） | 1022.04 → 3982.81 | 2052.90 → 9077.06 |
| `conv2` 勾配（step 0 / 1 / 2） | 0.99921 / 1.66736 / 112.551 | 1.85672 / 0.039696 / 1.39526 |
| 上流 3 箇所の勾配（step 0） | **すべて 0.0** | **すべて 0.0** |
| 上流 3 箇所の勾配（step 1 以降） | 有限かつ非ゼロ | 有限かつ非ゼロ |
| 生成 → `vae.decode` | `[1,3,1024,1024]`、NaN 0 | 同左、NaN 0 |

- **速度を抑えているのはクランプであってゼロヘッドではない**ことが実測で確認された:
  `199.14 / 0.05 = 3982.81` と一致する。§10.3 の記述どおり。
- 上流勾配は step 0 で厳密に 0、`optimizer.step()` が `conv2` を 0 から動かした step 1 以降で
  有限かつ非ゼロ。ゼロヘッド初期化の帰結（§10.3）が実重みでも同じ形で現れる。
- ピーク: ホスト RSS 32.357 GiB / commit 65.6 GiB、VRAM allocated 27.802 GiB。
  int8 ファイルサイズ（17.58 GB）からの見積もりは誤りで、full FT 経路が gen 側を
  int8 コードから bf16 重みへ実体化する（7,728 MiB 解放 → 15,456 MiB 確保）。
  `SENSENOVA_TRAINING_DESIGN.md` の実測 gen アーム（32.101 GiB）と一致する。

**未達のまま残るもの**: 条件 4 のうち「16× VAE で 4096² に `sensenova_resolution` 警告が出ない」半分。
本ツリーに 16× の**画像** VAE が存在しない（16× は MiniMax-H3 の動画 VAE のみ）ため実重みでは
検証できず、CPU テストのみで担保されている。
また 8× では `4·s = 32` が pixel モデルの align と数値的に一致するため、
スナップ値そのものは「差し替え後の幾何を読んだ」ことの証拠にならない（値の導出元が
`token_pixel_width(transformer)` であることが根拠）。64px のケースも CPU テストのみ。

**smoke で再現していない条件**: 生成は in-memory の変形済みツリーで行っており、
チェックポイント往復（`pipeline_backends/sensenova.py` が `component.vae.*` から幾何を再構築する経路）は
通っていない（保存に約 33 GiB を要するため）。この経路は
`tests/sensenova_latent_migration_test.py` の往復テストと不一致拒否テストで担保されている。
また 294 の gen Linear のうち 292 は凍結したまま測定した（全解凍は bf16 勾配 16.2 GiB を要し
45.4 GiB の空き VRAM に収まらない）。optimizer は素の Adafactor。

---

## 11. 実装フェーズ

各フェーズは独立に検証・コミット可能で、前フェーズの上に積む。ネイティブ VAE の挙動を
変えないことを各フェーズの回帰条件とする。

| Phase | 内容 | 検証 | リスク |
|---|---|---|---|
| ✅ **P0 宣言と正直な伝播**（dc686c0b）| `component.vae.*` 拡張キー（§5.2、`struct_native`/`identity_native`/`provenance`/`locator`/`norm_pack` を含む）、`_apply_component_hints` の `sushi.*` フォールバック、`current_model_info` / `pipeline._sushi_wiring`、`/models/current`・`describe_vae`・`_fold_baseline` の読み替え、arch 判定の順序変更（§9.2）、`ModelInfo` 型宣言 | 既存の swap 済み SDXL チェックポイントで `/models/current.latent_channels == 16`、`_check_vae_compat` が 4ch override を 400、native モデルで全応答が不変。§9.2-2 の `label_emb` キーは実チェックポイントのヘッダで確認できた場合のみ有効化 | 低。読み取り経路のみ。SD/SDXL 判定の新シグナルは実機未検証（確認できなければ入れない） |
| ✅ **P1 変形の一般化**（2aa48f4f）| `latent_io.py` + `LatentIOSpec`（入出力別の kind/order、§5.1, §6）、性質テスト（§6.6）、`resize_unet_in_out` を委譲に変更、新規チャネルゼロ初期化 | `tests/latent_io_test.py`: 入力側は各 arch の実 pack 関数、出力側は実 unpack 関数で**別ケース**として等価性（anima の入力 outer / 出力 inner を含む）、SDXL 回帰 bit 同一、swap 済みチェックポイントの再ロードで `in_module`/`out_module` の重みが保存値と bit 同一（§6.2 の回帰条件） | 低〜中。SDXL の初期化変更は挙動変更（CHANGELOG）。既存 run の再現性は変わる、再ロード・再開は不変 |
| ✅ **P2 出所と第1波（sd15/sdxl）**（eedaf45b / 475b801b / bb884f2f / bf917adb）| `vae_source.py`（registry/file/model、§7）、表B → 表A 統合、`vae_swap_source` の全層配線（§5.3、TRAINING_PARAMS_GUIDE Case B）、`GET /training/vae-sources`（openapi 先行）、`apply_vae_swap`（§8.1）、cache namespace（§8.5）、`strict_validation` 拡張（§8.6）、同梱 `vae.` と locator（§8.7）、生成ロードの同梱読み・locator 解決（§9.1）、swap 済み base の学習ロード（§8.2）、capability 登録と `VaeSwapSourceSelector`（§9.7）。**`shift_scale` 系 VAE のみ**を対象にし、BN 系（`norm=batchnorm`）は §7.4 で拒否 | sdxl: `registry:flux1`（16ch）/ `file:`（16ch standalone）/ `model:` で 3 ステップ smoke → 保存 → 生成ロード → 1 枚生成。`model:` の初回検証は (i) zimage フルモデルから抽出した 16ch `first_stage_model.`（`shift_scale`、C 変更あり）と (ii) 別の SDXL フルモデルから抽出した 4ch VAE（同 C・別 hash: `struct_native="1"`, `identity_native="0"`、resize はコピーのみ、cache namespace が分離、生成ロードで抽出 VAE が使われる）の 2 本。sd15 同様。`registry:`/`file:` + `bundle_vae=False` で locator 解決と hash 不一致時の拒否、`model:` + `bundle_vae=False` が preflight で拒否。flux2 フルモデルからの抽出は §7.4 で拒否されることを確認（解除は P7） | 中。配線層が多い。`model_fields_set` 判定の漏れ、old YAML の `sdxl_vae_type` エイリアス |
| ✅ **P3 アダプタ整合**（db99085e）| 書き側 identity（全 arch）、diffusers 経路ゲート、`AdapterSession` ゲート、`GET /loras` 拡張、フロント灰色表示（§9.4） | 標準 SDXL LoRA を swap 済み SDXL に当てて `lora_incompatible`; swap 済み base で学習した LoRA が通る; 同 C 別 hash で warning | 中。無メタデータ refusal は既存ユーザーの LoRA を swap 済みモデルで拒否する（設計意図） |
| ✅ **P4 生成側の残り**（146ea21b）| TAESD ルーティング、inpaint `2C+1`、`keep_hot` キー、`height//8` 読み替え、ロード時警告の再生（§9.3, §9.6） | 16ch SDXL でプレビューが RGB 射影/`taef1` に切替、inpaint が 33ch ゲートを正しく判定、`vae_path` 変更で hot VAE が無効化 | 低 |
| ✅ **P5 共有正規化層**（93a73f93 / 556febc6）| `normalize(spec)` 3 方式と正規化領域 `vae_norm_pack`（§8.4）、flux2/lens ops の「normalize → backbone patchify」2 段化、anima/krea2/ltx2 ops の置き換え、`or 1.0` 削除、§7.4 の BN 系ゲート解除 | 各 arch でネイティブ VAE の潜在が置き換え前後で bit 同一（同一入力・同一 dtype）。sdxl に flux2 由来の 32ch BN VAE（`registry`/`file`）を当てて `normalize` の出力が 32ch で、flux2 経路で BN 適用後に unpack した値と一致 | 中。5 arch の学習経路に触る。dtype 行列（fp16/bf16）で検証（verify-in-production-dtype） |
| ✅ **P6 第2波（zimage/krea2）**（ltx2 は別理由で拒否継続）| `LatentIOSpec` 宣言、各ローダの宣言読み・同梱読み、capability 解除。**ltx2 は解除しない**: full FT 保存が `net.` prefix の safetensors を書くが、これを読むローダが無い（`detect_model_type` は ltx2 を diffusers ディレクトリとしてのみ認識する）ため、学習できて生成できない成果物になる（556febc6 と同じ判断）。`ComponentWiringSpec.vae_scale_temporal`（P2a が保留した判断）は本波で追加し、ltx2=8 / minimax_h3=4 を宣言する — 宣言の無い 5-D arch（anima）は従来どおり拒否する。krea2 は `in_channels` がパック後の幅（C·p²）なので `LatentIOSpec.config_channels_packed` で区別する | 各 arch で smoke → 保存 → 生成。zimage は 16ch→4ch（既存 4ch 版 VAE）と 4ch→16ch の両方向 | 中。zimage は入出力とも `inner` 順の唯一の実例 |
| ✅ **P7 第3波（anima/flux2/lens/minit2i）** | anima `+1` と出力 `inner`、flux2/lens は P5 前提、minit2i は `vae_type` config と統合。実装で 3 点が設計と食い違った: (a) `config_channels_packed` は**入出力で別フラグ**が要る（lens は `in_channels=128` がパック後・`out_channels=32` が生、`vendor/transformer.py:426-451`）ので `config_in_channels_packed` / `config_out_channels_packed` に分割した。flux2 は両側パック（`patch_size=1`、2×2 パックは transformer の外）。(b) anima の `vae_scale_temporal` を 4 と宣言（`QWEN_IMAGE_VAE_CONFIG.temperal_downsample` の True 2 個）。宣言前は全候補が「比較対象が無い」で拒否されていた。(c) **minit2i の潜在ジオメトリは arch 定数ではなくチェックポイント単位の config 値**（pixel 3ch/patch16 と latent 4-or-16ch/patch2）なので、`MINIT2I_WIRING` を読む `vae_source.arch_native_vae` は latent 版にも pixel の拒否を返す。`ArchHandler.resolve_wiring` / `ArchHandler.check_vae_compatibility` の 2 フックを足し、minit2i だけが base の config（`minit2i_loader.peek_io_config`、preflight では `base_model_path` から）で答える。pixel → latent は patch_size も変わるので従来どおり拒否 | `tests/vae_swap_wave3_test.py`（CPU のみ、実重み不要）: 4 arch の両方向 resize、anima の padding-mask 行移動・`pos_embedder` 不変・**両側同順序なら落ちるケース**、lens の非対称 config 往復、minit2i の P 解決（pixel の 768=256·3 が P=4 でも割り切れることを含む）、各 arch の保存 → `load_declared_latent_io` 再読、preflight の capability 強制 | 中〜高。flux2/lens は BN 以外の VAE を初めて通す |
| **P8 SenseNova**（実装済み・ゲートは閉じたまま）| §10 全体（`P = 4` 固定、任意の `vae_scale_factor`。トークン幅 `4 × vae_scale_factor` 由来の配線 §10.5 を含む）。実装と設計の食い違いは §10.5 末尾の 8 点 | `tests/sensenova_latent_migration_test.py`（CPU のみ、実重み不要）: 形が変わるのは 2 テンソルだけで 8×/16× で同形状、他の全テンソルが bit 同一、ゼロヘッドと切断正規分布、128 セル格子＝1024 トークンで `fm_head` 出力が潜在格子、保存→再ロードで形状不変テンソルが bit 同一＋VAE 同梱、`t` 両端の `v` ノルム、step 0 の上流勾配ゼロ／step 1 で有限非ゼロ、ゲートが閉じていること。**未達**: §10.6-3 の 3 ステップ smoke と §10.6-4 の 1 枚生成は 16.2B の実重みと GPU が要る | 高。研究段階 |
| 保留 | acestep（§6.4） | — | RVQ 再学習を含み本書の範囲外 |

各フェーズのコミット前に `git diff --cached` の比較レビュー（CLAUDE.md 大規模変更手順）と、
`py_compile` に加えて実 import（`python -c "import core.models.components.latent_io"` 等）を行う。
フロントのビルドは所有者が実施する。

---

## 12. 範囲外

- **ideogram4 / minimax_h3**: full FT が拒否されている（§2）。開放されたときの変形仕様は §6.5 と
  `LatentIOSpec` の複数モジュール拡張で足りる。
- **minimax_music3**: 学習非対応。
- **acestep**: 変形自体は §6.4 で定義できるが、凍結 RVQ・`silence_latent`・`chunk_masks` の再設計を
  含むため本書では保留。
- **縮小率の変更**（8× → 16× 等）: D5 で拒否。SenseNova のみ例外で、任意の空間縮小率を受理する（§10.2）。
- **生成側 override でのフルモデル抽出**: 提供しない（§7.2）。
- **重みスケール補正・warmup/freeze スケジュール**: 行わない（D4）。効果の主張は実測後に限る。
- **SenseNova の `noise_scale` 再較正と `encoder_pinv` ヘッド初期化**: 実験結果待ち（§10.3, §10.4）。

---

## 13. 不変条件（実装者向け）

1. 既定値は `api/param_defaults.py` にのみ置く。Pydantic は参照する。
2. API 変更は `openapi.yaml` を先に更新する。
3. capability 判定は `api/arch_capabilities.py` と `adapters/capability.py` のみ。フロントは served
   `archCapabilities` を読み、`isXxxModel` で gate を書かない。
4. VAE のスケーリング数値は表A（`vae_store.py`）と各 VAE の `config.json` にしか存在しない。
   コードに数値リテラルを増やさない。
5. cache namespace への追加は加算的（`base_arch.py:431-433`）。
6. 「拒否」と「警告」の境界: 形状・チャネル・ndim・縮小率の不一致（`struct_native="0"` 側）は拒否、
   同形状で identity が異なるもの（`struct_native="1"`, `identity_native="0"`）は警告、証明できないが
   破損の蓋然性が高いもの（無メタデータ × `struct_native="0"`）は拒否。
7. `struct_native` と `identity_native` を混用しない: 構築・hard refusal・capability は前者、
   cache namespace・アダプタ identity・warning は後者。
8. 正規化は VAE 側の領域（`vae_norm_pack`）で閉じ、呼び出し側は常に生の C ch を扱う。backbone の
   パック（`LatentIOSpec`）と VAE の正規化パックを同じ物として書かない。
9. 表示用（`provenance`）を解決に使わない。解決は `locator` + hash 照合のみ。
10. UI・コミットメッセージ・本書に主観的形容詞と未測定の数値を書かない。
