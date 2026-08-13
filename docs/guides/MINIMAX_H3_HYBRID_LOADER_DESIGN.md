# MiniMax-H3 Hybrid Loader 取り込み設計

> 改訂履歴
> - rev1: 初版（コード未参照で作成）
> - rev2: 実コードに対する監査結果を反映。§3 にゲート極性の実測表を追加、§4.3/§4.4 を実 API に合わせて修正、§5 の capability モデルを既存機構ベースに置換、§9/§10 を既存テスト資産と実ファイル状況に合わせて再スコープ。

## 1. 結論

ComfyUI の `ComfyUI_MinimaxH3HybridLoader` が行うモデルハイブリッド化は、SushiUI の MiniMax-H3 推論へ**条件付きで取り込める**。

ただし、ComfyUI カスタムノードをそのまま移植するのではなく、SushiUI の独自 DiT ローダーに、2 本の safetensors を生キー段階で選択的に読む仕組みを追加する必要がある。SushiUI は、モデル構築前に次の処理を行っているためである。

- pruned AdaLN-curve の形状を checkpoint header から合成する
- Comfy 形式の fused QKV を分割する
- SwiGLU の行順を入れ替える
- FP8、W4A8、INT8 ConvRot の量子化契約を検証し、専用モジュールへ差し替える
- `strict=False, assign=True` で meta-built transformer に state dict を導入する

推奨する最初の構成は次のとおり。

| 項目 | MVP の方針 |
|---|---|
| base | `fl2va` checkpoint |
| overlay | 同じ MiniMax-H3 tree にある `ref2va` checkpoint |
| overlay 対象 | `blocks.N.adaln_proj.linear.weight`（`bias` は両側に存在する場合のみ） |
| block 範囲 | 既定 `25..49`、両端を含む |
| final AdaLN | 既定 OFF |
| 量子化 | base と overlay が同じ形式であることを要求 |
| 形式 | 同じ pruned geometry のみ |
| custom glob / 全量 overlay | MVP 対象外 |
| UI 上の扱い | `fl2va` / `ref2va` のどちらにも偽装せず、`hybrid` として表示 |
| reference workflow | 実測完了までは拒否または experimental 扱い |

upstream が報告する「品質の高い `fl2va` と reference 対応の `ref2va` を組み合わせる」効果は、upstream 作者による主観評価であり、SushiUI のローダー、量子化経路、サンプラー、reference row の配置で同じ結果になる保証はない。実装後の実機 A/B を通過した capability だけを段階的に解禁する。

### 1.1 ライセンス

upstream の `ComfyUI_MinimaxH3HybridLoader` は **MIT ライセンス**である。したがって、キー選択規則・block 範囲の既定値・sidecar 同伴規則などを参照実装として直接引用・移植することは許諾される。

§11 の「ComfyUI ノードの直移植は推奨しない」はライセンス上の制約ではなく、SushiUI 独自の変換・量子化・lifecycle を迂回してしまうというアーキテクチャ上の理由による。

**実装ルール**: upstream 由来のロジックを移植したファイル・関数には、MIT の copyright 表記とライセンス参照をコメントとして残す。SLA (Apache-2.0) の citation 要求と同じ扱いとする。

## 2. upstream の事実

参照元:

- [ComfyUI MiniMax H3 Hybrid Loader README](https://github.com/scottmudge/ComfyUI_MinimaxH3HybridLoader)
- [`minimaxh3.py`](https://raw.githubusercontent.com/scottmudge/ComfyUI_MinimaxH3HybridLoader/master/minimaxh3.py)
- [`minimax_h3_analysis.md`](https://raw.githubusercontent.com/scottmudge/ComfyUI_MinimaxH3HybridLoader/master/minimax_h3_analysis.md)

upstream は、base checkpoint の全キーを起点に、指定されたキーだけを overlay checkpoint から読み、合成 state dict を ComfyUI の標準 diffusion-model loader に渡す。`overlay_preset == "none"` なら base 単体と同じ挙動になる。

2 つの checkpoint は同じキー配置・shape・量子化 layout を持つ前提で、分析では次が報告されている。

- `fl2va` は first/last keyframe 系、`ref2va` は multimodal reference 系の partition。
- attention、MLP、norm、patch projection、token refiner 等の大部分は bit-identical または cosine similarity が非常に高い。
- 大きく異なるのは DiT 各 block の `adaln_proj.linear.*` と `final_layer.adaln_proj.linear.*`。
- 推奨 hybrid は `fl2va` を base にし、`ref2va` の block AdaLN だけを overlay する構成。
- upstream の既定候補は block 範囲 `25..49`。作者の主観的な試験結果であり、一般的な品質保証ではない。
- 両ファイルを `safetensors.safe_open` で mmap-backed に開き、キーごとに 1 つずつ読むことで、2 モデル分の weight data を同時に通常の RAM へ展開しない。
- `.comfy_quant` や scale 系の量子化補助テンソルは、対応する weight と同じ側から読むようにしている。
- 2 checkpoint のキー集合が一致しない場合は merge を拒否する。

upstream は ComfyUI の `load_diffusion_model_state_dict` と `ModelPatcher` の reload/deepclone 契約を利用する。一方、SushiUI は同じ Comfy 単一ファイルを読みつつ、独自の変換と量子化実装を持つため、読み出し選択部分だけを設計要素として取り込む。

## 3. SushiUI の現状と責務境界

主な既存箇所:

| 箇所 | 現在の責務 |
|---|---|
| `backend/core/models/minimax_h3/loader.py` | flat tree の layout 解決、header 検出、DiT/VAE/TE 構築、量子化ガード、Comfy キーから vendored module への変換 |
| `loader.py::_map_dit_state_dict` | 生 DiT キーの読み出し、QKV 分割、SwiGLU 行入替、sidecar の変換 |
| `loader.py::_build_transformer` | header 検証、量子化 module 差し替え、state dict 導入、meta tensor/strict-load 防衛 |
| `loader.py::_int8_convrot_layers_from_markers` | `.comfy_quant` marker の読み出し（**handle を消費する第二の経路**） |
| `loader.py::_guard_component_file` | ファイルごとに独立した `safe_open` による事前 guard |
| `backend/core/models/minimax_h3/reload.py` | 同一 tree 内の DiT-only reload、TE/VAE/scheduler の共有 |
| `backend/core/models/minimax_h3/minimax_h3_lora.py` | LoRA の QKV 分割・SwiGLU 入替・variant 互換判定 |
| `backend/core/model_loader.py` | MiniMax-H3 の load dispatch |
| `backend/core/pipeline.py` | loaded component、`current_model_info`、same-model 判定、reload、last-model 保存 |
| `backend/api/routes.py` | `/models` の partition 列挙、`/models/load`、variant ごとの生成 route gate |
| `backend/api/generation_utils.py` | outpaint reference gate、text-only TE gate などの純粋関数ゲート |
| `backend/api/arch_capabilities.py` | arch 単位の unsupported 表と `CHAIN_CONTEXT` |
| `backend/api/param_defaults.py` | API default の single source of truth |
| `frontend/src/components/common/ModelSelector.tsx` | H3 partition と TE/projection の選択、`/models/load` 呼び出し |
| `frontend/src/utils/api.ts` | typed API client と `ModelInfo` |
| `openapi.yaml` | load endpoint と model schema の API 契約 |

現在の H3 は、`diffusion_models/`、`text_encoders/`、`vae/` と MiniMax の config-only `official/` からなる flat tree を前提とする。DiT はファイル名から `fl2va`/`ref2va` を判定し、TE は大きな mmap-backed component として保持し、transformer/VAE は生成フェーズごとに GPU へ staging する。

### 3.1 既存 route gate の実際の極性（重要）

既存の gate は variant 文字列に依存するが、**その極性は一様ではない**。実測結果は次のとおりで、新しい `variant="hybrid"` を導入した時点で挙動が分かれる。

| gate | 実際の判定 | `variant="hybrid"` の扱い |
|---|---|---|
| `/generate/ref2vid` | `!= "ref2va"` なら拒否（allowlist） | **拒否される（安全）** |
| outpaint `reference_images` | `resolve_minimax_h3_outpaint_reference_gate` の「識別できない variant は拒否」節 | **拒否される（ただし暗黙の副作用に依存）** |
| temporal inpaint | `== "ref2va"` のときだけ拒否（**denylist**） | **通過してしまう** |
| img2vid keyframe conditioning | `== "ref2va"` のときだけ拒否（**denylist**） | **通過してしまう** |
| txt2vid | variant gate なし | **通過してしまう** |
| `chain_context_for` | 未知 variant は arch レベル entry にフォールバック | **`fl2va` 相当の chain 能力が広告されてしまう** |

したがって、**`variant="hybrid"` を書き込むコードより先に、これら 4 面（temporal inpaint / img2vid keyframe / txt2vid / CHAIN_CONTEXT）を修正しなければならない**。denylist は fl2va allowlist に反転し、txt2vid には H0 gate を追加し、`CHAIN_CONTEXT["minimax_h3"]["variants"]` に保守的な `"hybrid"` entry を明示する。

また、outpaint reference gate が hybrid を拒否しているのは「識別できない variant」節による偶発的な保護である。将来の variant 正規化リファクタで消える可能性があるため、**gate 内で `hybrid` を明示的に列挙する**。

`fl2va` と `ref2va` はキーと shape が同じでも、学習された conditioning の意味が同じとは限らない。したがって hybrid 追加では、単一の `variant` だけで base と overlay の意味を表現しない。

## 4. 推奨アーキテクチャ

### 4.1 HybridSpec

runtime load の入力を、単一 `dit_path` ではなく次の論理仕様として扱う。

```text
MiniMaxH3HybridSpec
  schema_version
  base_dit_path
  overlay_dit_path
  preset                 # MVP: block_range_adaln
  block_range_start      # MVP default: 25
  block_range_end        # MVP default: 49
  final_adaln_from_overlay  # MVP default: false
  base_variant           # fl2va
  overlay_variant        # ref2va
  compatibility_digest   # header/contract 検証結果
```

`source` は既存 API の互換性のため base DiT を指してよいが、内部の model identity と provenance は `HybridSpec` 全体から作る。overlay を隠れたグローバル状態や UI の一時状態として扱わない。

### 4.2 preflight と header 検証

モデル構築、TE mmap、既存モデルの破棄より前に、base/overlay の両 header を読む。

必須検証:

1. 両パスが同じ H3 tree に属し、同じ `official/`、VAE、**audio VAE**、TE と解決される。これは `reload.py` の `_SHARED_LAYOUT_KEYS`（`root`, `official`, `vae`, `audio_vae`, `text_encoder`）と一致させる。
2. base が `fl2va`、overlay が `ref2va` である。
3. 両方が `is_minimax_h3_safetensors` を通過する。
4. raw key 集合が完全一致する。
5. 全 key の shape と dtype が一致する。
6. pruned/full geometry が一致する。MVP は両方とも pruned AdaLN-curve に限定する。
7. quantization metadata、marker の種類、groupsize、必要な sidecar の存在が一致する。
8. W4A8 では `_quantization_metadata` と全 layer contract が一致する。
9. `blocks.N.adaln_proj.linear.weight` が指定範囲に存在する。`bias` は**両側に存在する場合のみ** overlay 対象とし、存在を前提としない（出荷ファイルに bias があるかは未確認。geometry 合成は weight しか読まない）。
10. base/overlay が同一形式である。FP8 と BF16、W4A8 と FP8、INT8 ConvRot と BF16 の混在は MVP では拒否する。

`__metadata__` のうちファイル名や variant 表記だけが異なることは許容してよいが、architecture、geometry、quantization semantics に影響する metadata は一致を要求する。検証結果は `compatibility_digest` として保持し、ロード後の provenance とテストログに使う。

**header 消費者の source 固定（契約）**: preflight 項目 4/5/7/8 が両ファイルの等価性を保証するため、以下はすべて **base の header / `__metadata__`** を使う。実装者が個別に判断してはならない。

- `_synthesize_transformer_config(header, official_dir)`
- `_w4a8_layers_from_metadata(metadata, ...)`
- quantization census / `scaled_quantization_report` 系
- `_header_shape(header, ...)` による broadcast 用 shape 参照

### 4.3 RawTensorReader と selector

既存の `_map_dit_state_dict` が期待する「raw key から tensor を取得する」境界を抽象化する。

実 API は次のとおりである。

```text
_map_dit_state_dict(handle, header, config, compute_dtype,
                    w4a8_layers=None, int8_convrot_layers=None)
```

全 tensor バイトは `handle.get_tensor(key)` という**単一のインターフェース**を通る。キー走査は `for key in header`（**header dict の挿入順であり、sorted ではない**）で行われ、走査は `_map_dit_state_dict` 側の責務である。

したがって reader の責務は次の 1 点のみとする。

```text
SingleTensorReader(base_handle)
    get_tensor(key) -> Tensor            # base から読む

HybridTensorReader(base_handle, overlay_handle, HybridSelector)
    get_tensor(key) -> Tensor            # selector が overlay 判定した key のみ overlay から読む
```

- reader は**走査しない**。渡された key に答えるだけの受動的な dispatcher とする。
- `header` は base の header dict をそのまま渡す（preflight 項目 5 が shape 一致を保証するため妥当）。
- 両 `safe_open(..., framework="pt", device="cpu")` を read-only で保持する。
- tensor data を 2 本分の state dict として同時保持しない。

**第二の handle 消費者（rev1 で見落とし）**: `_int8_convrot_layers_from_markers(handle, header, ...)` も `.comfy_quant` marker を handle 経由で読む。これを reader に通さないと、ConvRot marker の provenance が黙って片側に固定される。`_build_transformer` から呼ばれるこの経路も**同じ reader を経由させる**こと。

MVP の selector は glob ではなく構造化された block-range selector とする。

```text
blocks.<N>.adaln_proj.linear.weight
blocks.<N>.adaln_proj.linear.bias   # 両側に存在する場合のみ
```

だけを、`start <= N <= end` のとき overlay から読む。`final_layer.adaln_proj` は別の明示 toggle とし、既定では base に残す。`adaln_t_table`（fp32 固定で導入される curve 側のテーブル）も MVP では base のままとする。custom glob、全量 overlay、複数 overlay の優先順位は後続設計とする。

### 4.4 量子化 sidecar の原子性

upstream の `.comfy_quant`/scale 同伴規則をそのまま流用しない。SushiUI は形式ごとに扱う sidecar が異なるため、**logical weight family 単位**で provenance を決める。実キー名は次のとおり。

| 形式 | 同じ source から一緒に読むもの |
|---|---|
| BF16 | 対応する weight/bias |
| FP8 scaled | weight、`.weight_scale`、`.comfy_quant` |
| INT8 ConvRot | weight、`.weight_scale`、`.comfy_quant`（ConvRot の marker は**この `.comfy_quant` テンソルそのもの**であり、別キーではない） |
| W4A8 mixed | `.weight`、`.weight_s_rel`、`.weight_s_channel`、`.weight_codebook`、`.weight_correction`、および file-level `_quantization_metadata` |

補足（rev1 で欠落）:

- `.input_scale` は FP8 layer の一部に存在するが、ローダーが意図的に drop する（`_DIT_DROPPED_KEYS` 相当の policy）。したがって provenance の対象外である。
- NVFP4 の `.weight_scale_2` / `.pre_quant_scale` は DiT では扱わない。DiT の guard は NVFP4 を許可しない（許可されるのは INT8 ConvRot のみ）。TE 専用である。
- **出荷済み checkpoint では `adaln_proj` は量子化されていない**。量子化対象の 200 個の Linear は qkv / out_proj / fc1 / fc2 であり、`adaln_proj` は curve mode で fp32 強制導入される。つまり MVP selector は実際には sidecar に一度も触れない。本節の原子性規則は将来の拡張に対する防衛であり、MVP の実行時要件ではない。

overlay 対象の weight に対して sidecar が存在する場合、sidecar だけを base から取得してはならない。sidecar が片側にだけある、shape/dtype が違う、quant marker の契約が違う場合は load 前に拒否する。

両ファイルの量子化形式が同じでも、overlay 対象の weight を module 化する処理は既存の `_build_transformer` に任せる。Hybrid 層が量子化 weight を dequantize したり、module 構築後に raw weight を差し替えたりしてはいけない。

### 4.5 既存ローダーへの接続

推奨順序:

1. 両ファイルの header/quant contract を preflight。
2. 両ファイルの marker を既存 guard で個別検証。
3. `HybridTensorReader` を作る。
4. reader を通じて既存の DiT key mapping を実行（ConvRot marker 読み出しも同じ reader を経由）。
5. merged/mapped state dict に対して既存の quantization census と swap-count 検証を実行。
6. meta-built transformer に `strict=False, assign=True` で導入する。
7. unexpected key、**すべての** missing key、stranded meta tensor を拒否する（既存実装は missing を一切許容しない。rev1 の「unexplained missing」という表現は実装より緩かった）。
8. component dict に `hybrid_spec`、`variant="hybrid"`、capability state、provenance を付ける。

この順序は既存 `_build_transformer` の実行順と一致する。

**リファクタ時に壊してはならない不変条件**: `_dit_quantization_policy` / `disable_scaled_mm` の load-time 決定、および `input_scale` を含む drop-key policy は、`_build_transformer` に reader を受け取らせる改修を行っても、そのまま維持されなければならない。

### 4.6 メモリと Windows 動作

2本の safetensors を mmap すること自体は可能だが、現在の TE に対する Windows の mmap 制約と同一視してはならない。実装時には次を測定する。

- base 単体、overlay 単体、hybrid の peak RSS
- load 中の commit/pagefile と file mapping 数
- merged state dict 作成前後の peak RSS
- FP8、W4A8、INT8 ConvRot 各形式の resident footprint
- TE が既に mmap されている状態での DiT hybrid load
- **preflight の window**: `_guard_component_file` はファイルごとに独自の `safe_open` を行うため、hybrid では 2 本の 12〜21GB ファイルに対して一時的に最大 4 マッピングが開く

**mmap の生存期間（rev1 で見落とし）**: `assign=True` は mmap-backed tensor をそのまま live model に導入する。したがって hybrid では、モデルの CPU 側 weight が **2 本のファイルマッピングに支えられたまま**残る（初回 GPU staging まで、あるいはモデルの CPU lifetime 全体）。Windows では該当ファイルが削除・置換不能になる。§7 は load 中のファイル差し替えを扱うが、load 後のロックは別問題として扱う必要がある。

「upstream が約1モデル分」と報告していることは、SushiUI の state-dict mapping と量子化 module 化を含む測定結果ではない。受入基準を満たすまでは hybrid を experimental 扱いとする。

## 5. variant、capability、LoRA、生成メタデータ

### 5.1 variant の表現と capability の伝達

既存の `variant` に `fl2va` または `ref2va` を入れるだけでは、route gate が hybrid を誤って production workflow として扱う。variant は次のように分離する。

```text
variant: "hybrid"
base_variant: "fl2va"
overlay_variant: "ref2va"
hybrid_recipe: {
  preset: "block_range_adaln",
  block_range_start: 25,
  block_range_end: 49,
  final_adaln_from_overlay: false,
}
```

**capability の表現方法（rev1 から変更）**: rev1 は `capabilities: {reference_rows: "refused|experimental|allowed", ...}` という三値辞書と専用 resolver を提案していたが、このリポジトリにその語彙を消費する機構は存在せず、新規 resolver 層の発明を強いる。既存機構で同じ結果を得る。

| 層 | 既存機構 | hybrid での使い方 |
|---|---|---|
| 拒否 | `routes.py` / `generation_utils.py` の variant gate | denylist を fl2va allowlist に反転し、hybrid を明示列挙して拒否 |
| chain 能力の広告 | `arch_capabilities.py` の `CHAIN_CONTEXT[arch]["variants"][variant]` | 保守的な `"hybrid"` entry を追加 |
| experimental | 生成レスポンスの `warnings[]` / `add_warning(code=...)` | 通過は許すが未実測であることを警告 |

専用の capability resolver モジュールは、H3 hybrid 以外にも利用者が現れた時点で初めて正当化される。MVP では導入しない。

**capability の wire channel**: `GET /schema/arch-capabilities` は arch 単位の静的表であり、ロードごとに変わる hybrid の状態を載せられない。hybrid の provenance と capability は **`current_model_info`（`/models/load` のレスポンスと `/models/current`）** に載せる。フロントエンドは既に `currentModelInfo.model_info.variant` を読んでおり、同じ経路を拡張する。

### 5.2 段階的な capability 解禁

初期ロードでは、hybrid の reference 能力を `ref2va` と同等に扱わない。

- **H0**: load/inspect のみ。生成は拒否。§3.1 の 4 面（temporal inpaint / img2vid keyframe / txt2vid / CHAIN_CONTEXT）の修正が H0 の実体であり、`variant="hybrid"` を書き込む前に完了していなければならない。
- **H1**: reference rows を使わない standard prompt / keyframe のみ、`warnings[]` 付きで許可。
- **H2**: standard generation の A/B と量子化形式ごとの再現性を確認後、keyframe/audio conditioning を個別解禁。
- **H3**: reference rows、`ref2vid`、reference outpaint を専用の実測基準に通過した後に解禁。
- temporal inpaint、chain、outpaint の各組み合わせは、既存の `fl2va`/`ref2va` の測定結果を自動継承せず、必要な shape ごとに判定する。

### 5.3 LoRA

`fl2va` と `ref2va` は key/shape が同じため、LoRA の内容だけでは wrong-variant を検出できない。hybrid ではさらにどの AdaLN recipe を前提にしたかが問題になる。

既存の `check_variant_compatibility` は、`metadata["base_model"]` が `fl2va`/`ref2va` を宣言していて現在の variant がそれ以外の場合、**すでに hard refusal を行う**。したがって `variant="hybrid"` にすれば、variant を宣言済みの LoRA は追加実装なしで拒否される。

MVP で新たに必要なのは次の点のみ。

- metadata に variant 宣言が**ない** LoRA は現在 warning のみで通過する。hybrid では hard refusal、または明示的な experimental override + warning にする。
- hybrid 用の recipe fingerprint を持つ LoRA metadata を将来追加する。
- LoRA の QKV/SwiGLU 変換は既存の `minimax_h3_lora.py` を再利用し、hybrid reader の責務に混ぜない。
- **禁止事項**: `check_variant_compatibility` に `base_variant`（= `fl2va`）を渡してはならない。渡すと宣言済み LoRA への既存の保護が黙って無効化される。

### 5.4 生成メタデータ

生成結果には少なくとも次を記録する。

- `model_variant: "hybrid"`
- base/overlay の basename と canonical identity
- preset、block range、final AdaLN toggle
- base/overlay の header compatibility digest
- quantization format
- capability level と experimental warning

絶対パスや機密情報をそのまま gallery metadata に書かず、既存 metadata の sanitization 方針に合わせる。再現性のための内部 `last_model` 保存には、同じ HybridSpec を復元できる情報を残す。

**gallery 行への露出は C6（C5 では行わない）**: C4 の `model_hybrid_*` キーは DB の `parameters` JSON に入るが、`backend/database/models.py` の `to_dict()` は `model_variant` だけを whitelist しているため、gallery API のレスポンスには出ない。露出させる変更は、それを表示する `ImageGrid.tsx` と同じコミットで行う。バックエンドだけ先に開けると、どのキーをどう表示するか未定のまま API 契約が増える。PNG メタデータと `/models/current` は C4/C5 の時点で既に provenance を持っているため、情報自体は失われていない。

## 6. API、OpenAPI、defaults、frontend

### 6.1 API

`POST /models/load` は既に multipart/form-data であり、H3 専用の optional Form field（`text_encoder_file`、`clip_projection_file`）を持つ。hybrid の field はこの確立済みパターンの延長として追加する。

**命名規約**: 既存の H3 専用 field は**接頭辞なし**で、description に "MiniMax-H3 only" と書く方式である。rev1 の `minimax_h3_*` 接頭辞はこの規約から外れるため、接頭辞なしに揃える。

- `overlay_file`
- `hybrid_preset`
- `hybrid_block_range_start`
- `hybrid_block_range_end`
- `hybrid_final_adaln_from_overlay`

`source` は既存どおり base DiT または H3 tree とし、overlay 未指定なら従来の単一 checkpoint load とする。overlay 指定時は、route で解釈せず loader の preflight が同一 tree、variant、geometry、quant contract を検証する。

**overlay 候補 API**: `GET /models/minimax-h3/text-encoders` が、既にこの用途のテンプレートである（header-only、ロード中でも安全、tree 単位で候補を返す）。overlay 候補 API はその sibling として設計する。新パターンの発明は不要。

### 6.2 API 契約

API 変更時は必ず次を同一変更に含める。

1. `backend/api/param_defaults.py` に H3 hybrid の default bundle（`H3_HYBRID_LOAD_DEFAULTS`）を置く。`VIDEO_CHAIN_DEFAULTS` や `OUTPAINT_VIDEO_DEFAULTS` と同じ、生成 default 以外の bundle の前例がある。
2. `backend/api/routes.py` の `Form(...)` default はそこから参照する。
3. `openapi.yaml` の multipart schema、enum、default、examples、error response を更新する。
4. `ModelInfo` と current-model schema に hybrid provenance/capability を追加する。
5. header 不一致、異形式混在、overlay missing、未検証 capability の error code を定義する。

**未解決事項（実装時に決定すること）**: `/models/load` には現在 defaults を配信する経路がない（既存の Form default はリテラルの `None`/`False`）。block range の既定値 25/49 をフロントエンドへ届ける方法として、(a) overlay 候補 API のレスポンスに含める、(b) 新しい `/schema/model-load-defaults` を追加する、のどちらかを選ぶ。(a) の方が変更が小さい。

**決定（C5）**: (a) を採用。`GET /models/minimax-h3/hybrid-overlays` のレスポンスに `defaults` オブジェクトを載せる。block range は、同じレスポンスが同じ header から読む `base.num_blocks` と並んで初めて意味を持つため、別エンドポイントに分離すると 2 回のフェッチが必要になる。値は `H3_HYBRID_LOAD_DEFAULTS` を経由せず `hybrid_spec` の定数から直接返す（`param_defaults.py` も同じ定数を import する）。

**model_hash の既知の限界（C4 監査）**: hybrid の `model_hash` は base ファイル単体のハッシュであり、同じ base 上の異なる hybrid は `/models/current` でも gallery でも同じ値を報告する。hybrid の同一性判定には `hybrid.compatibility_digest`（および recipe）を使い、`model_hash` に何かを紐付けてはならない。OpenAPI の `LoadedModelInfo.hybrid` にもこの注意を記載済み。

`openapi.yaml` の編集時は、他エージェントの同時編集による重複キー（YAML の last-key-wins）を避けるため、コミット前に重複キー走査を行う。

### 6.3 frontend

`ModelSelector.tsx` は既に、モデルごとの load-time 選択状態を localStorage に保持し（`h3TextEncoder` / `h3ClipProjection`）、arch でゲートされたサブセレクタ（`MiniMaxH3TextEncoderSelector`）を持ち、追加 field を `loadModel` に渡している。hybrid selector は同じ形の拡張であり、書き直しではない。

- base partition を選ぶ
- 同一 tree の overlay partition を選ぶ
- preset と block range を選ぶ
- final AdaLN は advanced/experimental control として表示
- compatibility check が未完了の間は load button を disabled にする
- loaded summary に `fl2va + ref2va / blocks 25..49` を表示する

`frontend/src/utils/api.ts` には load request と `ModelInfo` の型を追加する。generation panel に hybrid 固有の判定を散在させず、**`currentModelInfo.model_info` 経由**で既存の variant ベース gating と同じ経路に載せる（§5.1 参照。`/schema/arch-capabilities` は静的なので使えない）。

## 7. reload、cache、原子性

既存の `build_dit_only_reload` は同一 tree 内で DiT だけを差し替え、TE/VAE/scheduler を共有できる。この利点は hybrid でも維持する。ただし単一 `dit_path` だけでは overlay recipe が失われる。

model identity に含めるもの:

- base canonical path
- overlay canonical path または none
- preset
- block range
- final AdaLN toggle
- compatibility digest
- TE/projection selection

**variant はファイル名由来にしてはならない（C4 の必須要件）**: `loader.py:1045` の variant 判定は**部分文字列マッチ**である（`"ref2va" if "ref2va" in name else ("fl2va" if "fl2va" in name else None)`）。したがって base パスに `fl2va` を含む hybrid は、何もしなければ `fl2va` とラベルされ、C1 で閉じたゲートを**すべて通過する**。C4 は merged component に対して `variant="hybrid"` を**明示的に設定**しなければならず、`layout["variant"]` を継承してはならない。

**variant のラベル形式**: C1 の txt2vid ゲートは「released partition のいずれでもないもの」を拒否する形にしてあるため、`hybrid_25_49` のような recipe 込みラベルを採用しても素通りしない。ただし img2vid / temporal inpaint の allowlist も同じ性質を持つことを C4 で再確認すること。

**修正が必要な具体箇所**: 現在の same-model 判定は `model_id = f"{source_type}:{source}"` の単純な文字列比較であり、`h3_te_selection_changed` と `component_health` によってのみ carve-out されている。**同じ base で overlay/range だけを変えた要求は、この文字列比較で early return され、再構築されない**。DiT-only fast path の同一性判定も同様に hybrid recipe を含める必要がある。`h3_te_selection_changed` が、そのまま真似すべき既存の前例である。

`last_model` の復元は現在 `(source_type, source, pipeline_type, te, projection)` しか渡さないため、HybridSpec の永続化は新規配線となる。

reload の原子性については、既存実装が**すでに要求を満たしている**。

1. 新しい HybridSpec を header-only preflight する。
2. 現在の live components は保持したまま、新 transformer を CPU 側に構築する。（既存: 現 component dict は失敗時も含め一切変更されない）
3. strict load、quantization swap count、meta tensor 検査を完了する。
4. 成功後にのみ `minimax_h3_components` と `current_model_info` を交換する。（既存の動作）
5. 失敗時は旧 transformer、TE、VAE、scheduler、current model info を変更しない。（既存の動作）
6. 旧 transformer の解放は交換後に行う。（既存の動作）

したがって hybrid で新規に必要なのは、identity への recipe 組み込みと `last_model` 永続化のみである。

overlay file が途中で置換された場合に備え、header digest と file identity を preflight と実読込の双方で確認する。ファイルが変化したら partial load を成功扱いにしない。

## 8. MVP の制約と非目標

MVP で扱うもの:

- 同一 SushiUI MiniMax-H3 tree 内の 2 DiT
- `fl2va` base + `ref2va` overlay
- 同じ pruned AdaLN-curve geometry
- 同じ quantization format
- block AdaLN の固定または範囲指定 overlay
- `safe_open` による read-only streaming
- 既存の transformer mapping/quantization/runtime reload への接続

MVP で扱わないもの:

- 異なる tree の merge
- BF16/FP8/W4A8/INT8 ConvRot の異形式混在
- full checkpoint の自動書き出し、元ファイルの書き換え
- custom glob、複数 overlay、重複 rule の優先順位
- final AdaLN を既定 ON にすること
- `adaln_t_table` の overlay 化
- TE、VAE、scheduler の hybrid 化
- hybrid に対する既存 LoRA の無条件適用
- 専用 capability resolver モジュールの導入
- upstream の結果だけを根拠にした reference workflow の production 解禁

異形式混在を将来検討する場合も、単に同じ key/shape だから許可してはならない。quantized module の contract と sidecar 変換を形式ごとに設計し直す。

## 9. テスト計画と実機受入基準

### 9.1 単体・契約テスト

既存のテスト資産を前提とする。`backend/tests/` には 43 本の `minimax_h3_*` テストがあり、`minimax_h3_model_listing_test.py` には**header-only の偽 safetensors を書き出すフィクスチャビルダーが既に存在する**（データ部ゼロ、struct-pack した JSON header、`diffusion_models/ official/ text_encoders/ vae/` の完全な偽 tree）。新規作成ではなく拡張する。

追加するテスト:

- header-only の key set mismatch、shape mismatch、dtype mismatch を拒否する。
- pruned/full 混在を拒否する。
- base/overlay の variant 方向を逆にした場合を拒否する。
- 同一形式でない場合を拒否する。
- overlay 未指定が単体 base load と同じ source 選択になる。
- `25..49` が block 25〜49 の AdaLN weight（と存在すれば bias）だけを overlay から読む。
- range の境界、空 range、範囲外を検証する。
- 選択された weight の全 sidecar が同じ source から読まれることを fake reader で検証する。
- ConvRot marker の読み出しが reader を経由することを fake reader で検証する。
- custom glob、final AdaLN、full overlay は MVP で明示的に拒否または feature flag 下に置く。
- 既存の QKV split、SwiGLU swap、FP8/W4A8/ConvRot の swap count が単体 base と同じ契約で閉じる。

### 9.2 pipeline/reload/API テスト

既存テストの多くは拡張対象である（`minimax_h3_dit_reload_test.py` には失敗時に現行 component が変更されないことを確認するテストが既にある。他に `minimax_h3_model_variant_record_test.py`、`minimax_h3_outpaint_reference_gate_test.py`、`minimax_h3_temporal_inpaint_route_test.py`、`minimax_h3_load_dispatch_test.py`）。

- 同一 tree の base-only、hybrid、range 変更がそれぞれ正しい model identity になる（= same-model early return を通さない）。
- hybrid DiT-only reload が TE/VAE/processor/scheduler を同一 object として保持する。
- preflight または strict load 失敗時に現行 model が維持される。
- **`variant="hybrid"` が temporal inpaint / img2vid keyframe / txt2vid / ref2vid / outpaint reference のすべてで拒否される（H0 の回帰テスト）。**
- `chain_context_for("minimax_h3", "hybrid")` が fl2va の entry にフォールバックしない。
- `/models` の partition 列挙と overlay candidate の表示が一致する。
- `/models/load` の default、OpenAPI、実装の値が一致する。
- `/models/current` と generation metadata に hybrid provenance が入る。
- existing `fl2va`/`ref2va`/non-H3 の挙動が変わらない。

### 9.3 実機 A/B

**利用可能な checkpoint（実測確認済み）**: `M:\model\minimax_h3\diffusion_models\` に両 variant × 3 形式が揃っている。

| 形式 | ファイル | サイズ |
|---|---|---|
| FP8 scaled | `minimax_h3_{fl2va,ref2va}_pruned_fp8_scaled.safetensors` | 各 20.96 GB |
| INT8 ConvRot | `minimax_h3_{fl2va,ref2va}_pruned_int8_convrot.safetensors` | 各 20.97 GB |
| W4A8 mixed | `minimax_h3_{fl2va,ref2va}_pruned_w4a8_mixed.safetensors` | 各 12.54 GB |

**BF16 は disk 上に存在しない**（ローダーのコメントにも「ここにはダウンロードされていない」と明記）。したがって Phase 1 の load parity は BF16 ではなく **fp8_scaled** で行うか、合成した小さな BF16 フィクスチャで行う。

最低限、同一 seed、同一 prompt、同一解像度、同一 frame count、同一 steps で次を比較する。

1. `fl2va` base-only
2. `ref2va` base-only
3. hybrid `25..49`
4. hybrid 全 block
5. hybrid `25..49` + final AdaLN

比較項目:

- standard text-to-video の画質と破綻率
- keyframe binding
- audio-conditioned video の挙動
- reference image/video/audio の binding と干渉
- video/audio の finite 値、shape、同期、decode 成功率
- denoise/decode の peak VRAM、load peak RSS、pagefile/commit、load 時間
- FP8、W4A8、INT8 ConvRot の各同形式ペアでの出力再現性
- LoRA なし/明示的に許可した LoRA の挙動

受入基準は実測前に固定する。少なくとも「既存 base-only の standard workflow を壊さない」「hybrid load が quantization guard を迂回しない」「失敗時に旧 model を保持する」「未検証 reference workflow が route gate をすり抜けない」を必須とする。reference capability の解禁基準は、upstream の主観評価をそのまま採用せず、SushiUI の実機 A/B 結果で決める。

## 10. 段階的な実装手順（コミット単位）

このリポジトリは feature/進捗単位でコミットする。各コミットは独立に検証可能でなければならない。

| コミット | 内容 | 検証 |
|---|---|---|
| **C1** | ゲート極性の修正。temporal inpaint と img2vid keyframe の denylist を fl2va allowlist に反転、txt2vid に H3 variant gate を追加、outpaint reference gate に `hybrid` を明示列挙、`CHAIN_CONTEXT["minimax_h3"]["variants"]` に保守的な `"hybrid"` entry。**まだ hybrid を生成する経路はないので、この時点では既存挙動不変が受入条件。** | 既存 `fl2va`/`ref2va` テストが全て緑。`hybrid` 文字列に対する新規拒否テスト。 |
| **C2** | preflight + compatibility digest + 全拒否パス。HybridSpec のデータ構造。 | header-only フィクスチャによる拒否テスト（§9.1 前半）。 |
| **C3** | `HybridTensorReader` + structured selector + `_map_dit_state_dict` / `_int8_convrot_layers_from_markers` への接続。 | fake reader による source 選択テスト、fp8_scaled での base-only vs hybrid load parity。 |
| **C4** | component lifecycle。model identity への recipe 組み込み、same-model early return の修正、`last_model` への HybridSpec 永続化、DiT-only reload の hybrid 対応。 | reload/identity テスト（§9.2）。 |
| **C5** | API 層。`param_defaults.py` の `H3_HYBRID_LOAD_DEFAULTS`、`routes.py` の Form field、overlay 候補 API、`openapi.yaml`（重複キー走査込み）、`api.ts` の型。 | OpenAPI parity 検証、API テスト。 |
| **C6** | frontend。ModelSelector の hybrid selector と loaded summary。 | ユーザーによるビルド・型チェック。 |
| **C7+** | 実測に基づく capability の個別解禁。H1 → H2 → H3 をそれぞれ別コミットに。 | §9.3 の A/B 結果。 |

**C1 は C4 より前でなければならない。** C4 で `variant="hybrid"` が `current_model_info` に書き込まれるようになるため、それ以前にゲートが閉じている必要がある。rev1 は capability を Phase 4 に置いていたが、それでは未検証の hybrid が temporal inpaint と txt2vid に到達する窓ができる。

### 変更候補ファイル

- `backend/core/models/minimax_h3/loader.py`
- `backend/core/models/minimax_h3/reload.py`
- `backend/core/models/minimax_h3/minimax_h3_lora.py`
- `backend/core/model_loader.py`
- `backend/core/pipeline.py`
- `backend/api/routes.py`
- `backend/api/generation_utils.py`
- `backend/api/arch_capabilities.py`
- `backend/api/param_defaults.py`
- `frontend/src/components/common/ModelSelector.tsx`
- `frontend/src/utils/api.ts`
- `openapi.yaml`
- `backend/tests/minimax_h3_*hybrid*` 相当の新規テスト（既存フィクスチャビルダーを再利用）
- `docs/guides/MODEL_FACTS.md`（実装・実測後の事実追記）

## 11. リスクと代替案

### リスク

- **ゲート極性リスク（最優先）**: 既存の temporal inpaint / img2vid keyframe gate は denylist であり、txt2vid にはゲートがない。`variant="hybrid"` を先に導入すると未検証モデルがこれらの経路に到達する。
- **品質リスク**: AdaLN の差し替えは block ごとに大きな差があり、reference が有効になる一方で standard quality や音声品質を損なう可能性がある。
- **量子化リスク**: weight と sidecar の source がずれると、load は成功しても推論結果が壊れる可能性がある。ConvRot marker を reader に通し忘れるのが最も起きやすい経路。
- **メモリリスク**: 2 mmap、mapped tensors、quantized module の staging が Windows の commit/pagefile 制約に触れる可能性がある。加えて `assign=True` により 2 本のファイルがモデルの CPU lifetime 中ロックされ続ける。
- **variant リスク**: hybrid を `fl2va` または `ref2va` として扱うと、未検証の route が通る。
- **reload リスク**: overlay recipe を model identity に含めないと、別 recipe の要求が same-model early return（単純な文字列比較）で消える。
- **LoRA リスク**: variant が同じでも AdaLN recipe が違うため、既存 LoRA を無条件に適用できない。`check_variant_compatibility` に `base_variant` を渡すと既存の保護が消える。
- **TE 置換との交差**: text-only TE gate による小型 TE 変換と hybrid を同時に使うと、2 つの実験的置換が重なる。capability の判定はこの組み合わせを別扱いにする。
- **variant のファイル名由来リスク**: 上記 §7 参照。C1 のゲートは variant 文字列を信頼しており、その文字列がファイル名から推測されている以上、C4 が明示設定を怠るとゲートは無力化される。
- **`chain_supports_exact_prefix`**: C1 が追加した hybrid の CHAIN_CONTEXT entry で、このフィールドだけは能力を「保留」ではなく「主張」している。route が拒否している間は不活性だが、C7 で hybrid の生成を解禁する際は継承せず再導出すること。

### 代替案

1. **offline merged checkpoint**: 実行時の lifecycle は単純になるが、2本分の保存容量、recipe の再生成、量子化形式ごとの書き出しが必要になる。
2. **model 構築後の module weight 差し替え**: raw key の量子化 sidecar、QKV split、SwiGLU swap と整合しないため推奨しない。
3. **ComfyUI ノードの直移植**: MIT ライセンス上は可能だが、SushiUI の loader、quantization、pipeline/reload、API 契約を迂回するため推奨しない。ロジックの参照・部分移植は帰属表記付きで行う。
4. **hybrid を ref2va として先行実装**: reference route を早く試せるが、未検証能力を production capability と誤認するため採用しない。

最も安全な方針は、SushiUI の既存 loader の変換・量子化・lifecycle を維持し、raw tensor の source selection だけを追加し、`hybrid` を独立した実験 variant として実測後に capability を解禁することである。
