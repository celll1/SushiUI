# Style embedding encoder 設計書

Status: 設計確定・未実装（2026-09-23）

## 1. 目的と要件

SigLIP2 tagger は、run の設定（`excluded_categories: ["Artist"]`）で artist タグを意図的に外して学習している。理由は2つある。流行の画風に寄った偽陽性が多いこと、そして誤った権利主張や判定の道具にされかねないことである。

本機能では、tag 判定とは独立に **style embedding** を出力するモデルを作る。

- 同じ artist タグを持つ画像同士の cosine 類似度が高くなるように学習する。
- 画風が似た別の作者同士が近くなることは許容する（作者を識別する分類器ではない）。
- 想定用途は、画風の判定・検索と、生成側の conditioning（ControlNet / IP-Adapter 系）。
- 成果物の配布は制限しない。作者を特定する用途は、artist を対象にした tagger でも実現できるため、style モデルの配布を控えても抑止にならない。

## 2. 決定事項

| 論点 | 決定 | 理由 |
|---|---|---|
| 構成 | **独立した style encoder**（B2）。tagger とは重みを共有しない | 単独で配布でき、tagger の再学習に引きずられない。サイズも独立に選べる |
| 学習 | **tagger とは別の run / 別の学習モード** | 重みを共有しないため同時学習の利点がない。batch 構成（tagger はランダム、style は P×K）が両立しない |
| 共通化の範囲 | 学習の仕組み（取り込み、offload、checkpoint、UI）を共通化し、モデルは共有しない | 保守性と分離の両立 |
| 作者ラベル | 既存の artist タグ | 3M 以上のデータと、追加中の artist データセットで十分 |
| 主 loss | **重み付き multi-positive SupCon** | §8.1 |
| 内容の shortcut 対策 | 同じ作品の作者で batch を組む ＋ 正例の軽い重み付け ＋ 入力パッチの block mask ＋ 凍結した tagger による HSIC ペナルティ | §6〜§8 |
| 採用しない案 | 途中層からの分岐（B1）、tagger との同時学習（C）、トークン単位で routing する MoE、encoder の凍結（A） | A は、artist を外して調整した重みを使うため不適 |
| 保留 | サイン位置の判定、作者ごとの複数中心（sub-center）、tagger 特徴による hard negative 探索、prototype 補助 loss | §15 |

## 3. 全体構成

```
backend/core/style_encoder/          # 新規パッケージ（既存に同名なし）
  style_model.py        # encoder + masked pooling + projector
  style_dataset.py      # 作者インデックス、分割、view 生成
  style_sampler.py      # P×K / 同じ作品 batch の sampler
  style_loss.py         # 重み付き SupCon、nHSIC
  style_trainer.py      # 学習ループ（共通基盤を import して使う）
  style_eval.py         # 検索評価
```

tagger 側の `tagger_trainer.py` は単体のファイルが大きい（約2,800行）。共通化は **継承ではなく、独立した関数やモジュールへの切り出し**で行う。切り出しの候補は次のとおり。

- optimizer と scheduler の構築
- checkpoint の保持ルール
- GPU 共有時の offload（`tagger_offload.py`）
- dataset drift の検査
- caption の取り込み（`resolve_caption_tags`）
- `gpu_index` の適用

切り出しは tagger の挙動を変えないリファクタリングとし、CLAUDE.md の比較レビューの対象とする。

## 4. モデル

- **encoder**: SigLIP2 NaFlex（Apache-2.0）。初期値は tagger 用に調整する前の Google 公開重み。
  - サイズの候補は `google/siglip2-base-patch16-naflex` と `google/siglip2-so400m-patch16-naflex`。Phase 0 の測定で選ぶ。
- **学習方式**: `full` / `lora`（既存の `LoRALinear` を使う）。
- **pooling**: 学習可能な single-query attention pooling。**padding と mask 対象の token を `key_padding_mask` で除外すること**（不変条件 I-1）。修正済みの `CustomAttentionPooling` を流用してよい。
  - 入力は `style_feature_layers` で指定した層の token 列を連結し、線形変換したもの（デフォルト `[-1]`）。
- **projector**: 2層 MLP → L2 正規化。出力次元は `style_embed_dim`（初期値 512）。
- **出力**:
  - `embed(x) -> [B, D]`：画風の埋め込み
  - `token_features(x) -> [B, N, D_tok]`：将来の cross-attention conditioning 用

**不変条件 I-1**: 出力は、mask 対象または padding の位置の画素値に依存しない。

SigLIP2 の encoder の attention は、mask された位置を key から除外する。しかし mask された token 自身の hidden state には、residual 経路を通って自身の画素情報が残る。したがって、pooling で除外しない限り情報が漏れる。

> 参考：tagger の `CustomAttentionPooling`（`cls_dim` 指定時）は、以前 padding token も attention の対象にしていた。`main` の `277d827e` で `key_padding_mask` を適用するよう修正済みである（テスト: `backend/tests/tagger_custom_pooler_mask_test.py`）。style encoder の pooling も同じ不変条件を満たすこと。

## 5. データ

- **入力**: `dataset_configs`（tagger と同じ形式）。caption の解釈には既存の取り込み処理を使う。
- **カテゴリ判定**: 必ず `get_tag_group_manager().get_tag_group()` を使う（独自の判定は禁止）。artist の別名は既存の `TagAliasResolver` で正規の名前にまとめる。
- **作者ラベル**: artist タグがちょうど1つの item だけを使う。共作（2つ以上）と作者不明（0）は除外し、件数をログに出す。
- **学習対象の作者**: 画像が `style_min_images_per_artist`（初期値 4）枚以上ある作者。
- **インデックス**（学習開始時に一度構築）:
  - artist → items
  - copyright → artist → items
  - item → 内容タグの集合（Character ∪ Copyright）
  - item → サイン系タグの有無

  3M 規模に対応するため、Python の dict of list ではなく numpy の CSR 配列で持つ。規模は item 数に対して O(n)。
- **分割**:
  - **学習に出さない作者（主な評価用）**: 正規化した artist 名のハッシュで決める（`style_val_artist_ratio`、初期値 0.05）。データが増えても既存の作者の所属が変わらない。
  - **学習済み作者の、学習に使わない画像（参考値）**: `style_val_image_ratio`。ほぼ同じ画像が学習側と検証側にまたがる可能性があるため、この値は参考扱いとする。
- **dataset drift の検査と rescan**: tagger と同じ仕組みを使う。

## 6. サンプリング

- 1 batch = 作者 P 人 × 各 K 枚。1枚から view を2つ作るので、埋め込みは 2PK 個になる（`style_artists_per_batch` P、`style_images_per_artist` K）。
- **同じ作品の batch**（割合 `style_same_copyright_batch_ratio`、初期値 0.5）:
  1. 学習対象の作者が P 人以上いて、各作者がその作品で K 枚以上持っている copyright を選ぶ。
  2. その中から P 人を選び、その作品の画像を各 K 枚選ぶ。
  3. 条件を満たす copyright がない場合は、通常の batch に戻す。
- **通常の batch**: 作者を n^α に比例する確率で選ぶ（n は作者の画像数、`style_artist_sampling_power` α、初期値 0.5）。作者ごとの K 枚は、その作者の中でキャラが分散するように選ぶ。
- **epoch**: 学習対象の画像数 ÷ (P·K) step と定義する。

## 7. View 生成

- **global view**: 画像全体。tagger と同じ NaFlex の前処理を使う。
- **masked view**: 同じパッチ grid に対する **block mask**。
  - 矩形の block を、有効パッチのうち mask された割合が比率 r に達するまで配置する。
  - mask は `pixel_attention_mask` を 0 にすることで行う。V1 では token を物理的に削除しない。
- **mask 比率**: r_a = r_min + (r_max − r_min)·s_a（`style_mask_ratio_min` 初期値 0.3、`style_mask_ratio_max` 初期値 0.75）。
- **作品との交絡度 s_a** = c_a · e_a
  - c_a（作品の集中度）: その作者の画像のうち、最も多い作品が占める割合
  - e_a（作品の独占度）: その作品の画像のうち、その作者が占める割合
- **スケールについて**: 切り抜きと違い、2つの view の間で縮尺が一致する。画面全体の解像度はパッチ数の上限（`style_max_num_patches`）で決まる。
  - tagger のデフォルトである 256 パッチでは、大きな画像ほど強く縮小され、線幅などの細部が失われる可能性がある。
  - 上限値は Phase 0 で測って決める。

## 8. Loss

### 8.1 主 loss：重み付き multi-positive SupCon

anchor i に対して、正例の集合 P(i) を「同じ作者の他の埋め込み（同じ画像の別 view を含む）」とする。

```
L_i = − Σ_{j∈P(i)} (w_ij / Σ_{k∈P(i)} w_ik) · log( exp(s_ij/τ) / Σ_{a≠i} exp(s_ia/τ) )
```

- s は cosine 類似度。τ は `style_temperature`（初期値 0.1。Khosla et al. 2020 の値）。
- 正例の重み:
  - 同じ画像の別 view: `style_view_positive_weight`（初期値 1.0）
  - 同じ作者の別の画像: w_ij = max(w_min, 1 − J_ij)
    - J_ij は内容タグ集合（Character ∪ Copyright）の Jaccard 係数。
    - w_min は `style_positive_weight_min`（初期値 0.3）。下限があるので、同じ作品しか描かない作者の正例も必ず学習に寄与する。
    - 両方の集合が空のときは J=0 とする。

**prototype 型（作者ごとの学習ベクトル）を主 loss にしない理由**

1. 同じ作品の batch による対策は、batch 内の比較にしか効かない。prototype 型は毎 step 全作者と比較するので、内容で見分けられる負例を毎回持ち込み、shortcut を再び生む。
2. 作者の表が作者数に比例して大きくなり、データの追加に応じて伸ばす必要がある。
3. 作者の分類器そのものになる。

以上から、prototype 型は保留の補助 loss とする。

### 8.2 内容の追い出し：凍結した tagger による nHSIC

- **tagger**: `style_hsic_tagger_path` で指定する、artist を外した SigLIP2 tagger の checkpoint。凍結して推論のみに使う。tagger 自身の前処理で、画像1枚につき1回 forward する。
- **対象の列**: checkpoint の語彙にある `tag_to_category` を使って、カテゴリごとに出力確率（sigmoid）の列を選ぶ。
- **ペナルティ**: カテゴリ g ごとに、正規化 HSIC（CKA 形式）nHSIC(K_Z, K_{T_g}) を global view の埋め込みに対して計算する。
  - kernel は両側とも Gaussian で、bandwidth は median heuristic で決める。
  - 勾配は style 側にのみ流す。
- **重み** `style_hsic_weights`:

  | カテゴリ | 初期値 | 補足 |
  |---|---|---|
  | Character | 0.1 | |
  | Copyright | 0.1 | |
  | Rating | 0.1 | |
  | Quality | 0.0 | 作者と強く相関し、主 loss とぶつかる可能性があるため opt-in |
  | General | 対象外 | 画風系のタグ（`monochrome`、`sketch` など）を保護するため |
  | Meta の一部（`signature`、`highres` など） | — | 将来の候補 |

- **refuse 条件**: 重みが 0 より大きいカテゴリの列が語彙に1つもない場合、学習の開始を拒否する（警告だけで進めない）。
- **合計の loss**: L = mean_i L_i + Σ_g λ_g · nHSIC_g

**既知の緊張関係**: 作品との交絡度が高い作者では、SupCon（同じ作者を近づける）と HSIC（作品の情報を消す）が原理的にぶつかる。釣り合いは λ_g で決まる。§10 の評価で監視する。

## 9. 学習ループと共通基盤

- **optimizer**: `adamw` / `adamw8bit` / `lion8bit`。scheduler は warmup → cosine。
- **その他**: bf16、gradient checkpointing、FlashAttention-2（任意）、`gpu_index`。
- **GPU 共有の offload**: 凍結した tagger も、退避と復帰の対象に含める。
- **resume**: sampler の状態（RNG、epoch 内の位置）、分割に使う seed、作者インデックスのハッシュを保存する。インデックスのハッシュが変わった場合は、sampler を作り直したうえで続行する。
- **optimizer step ゼロ**: 一度も optimizer step が完了しなかった run は、完了扱いにせず重みも保存しない（tagger と同じ規約）。

## 10. 評価

`validate_every` epoch ごとに、次を計算する。

| 指標 | 内容 | 用途 |
|---|---|---|
| (a) 学習に出てこない作者での検索 | Recall@1/5、mAP@R。各画像を query にし、残りを検索対象にする | **主指標**。`best` の選定に使う |
| (b) 同じ作品に限った検索 | 検索対象を query と同じ copyright に限る。該当する作品に、学習に出てこない作者が2人以上いる query のみで計算する | 内容の shortcut の検出 |
| (c) サインの有無による分割 | サイン系タグ（`style_signature_tags`、初期値 `signature, artist name, watermark, twitter username, web address`）の有無で分け、(a) をそれぞれ計算する | サインによる精度の水増しの検出 |
| (d) masked view 同士の検索 | (a) を masked view で計算する | mask に対して頑健かの確認 |
| (e) 学習済み作者の、学習に使わない画像での検索 | (a) と同じ計算 | 参考値（ほぼ同じ画像による水増しの可能性あり） |

## 11. 成果物

- **配布用**: `{name}_style_encoder.safetensors`（encoder、pooling、projector）と `{name}_metadata.json`。
  - metadata に含める情報：
    - base repo id
    - `style_embed_dim`、`style_feature_layers`、`style_max_num_patches`
    - 前処理の設定
    - HSIC に使った tagger checkpoint のパスとハッシュ
    - データセットの snapshot 情報
  - 作者の一覧や作者 ID の対応表は、配布用には**含めない**。
- **学習状態**（resume 用）: optimizer、sampler の状態、作者インデックス、分割 seed。
- **LoRA**: adapter 単体を保存し、merge したものを出力することもできる。
- **保持ルール**: tagger と同じ（step checkpoint はまとめて削除し、`latest` と `best` は対象外）。

## 12. API / DB / UI

- **API**: OpenAPI を先に更新する。
  - `/api/v1/style-training/runs`：作成・一覧・取得・更新・start・stop・delete・metrics
  - `/api/v1/schema/style-training-defaults`
- **デフォルト値**: `STYLE_TRAINING_DEFAULTS` を `backend/api/param_defaults.py` に追加する（唯一の情報源）。
- **DB**: `training.db` に `style_training_runs` と `style_training_metrics` を新設する。
  - `tagger_training_runs` は tagger 固有の列（`best_f1`、`tag_vocabulary` など）を持つため、流用しない。
- **UI**: 学習タブに style モードを追加する（設定と学習モニタ）。
- **推論 API**（Phase 4）: `/api/v1/style/embed`、`/api/v1/style/similarity`。

## 13. フェーズ

| Phase | 内容 | 完了条件 |
|---|---|---|
| 0 | 測定（推論のみ）。素の SigLIP2 base / so400m、パッチ上限 256 / 1024 について、層ごとの特徴で (a)(b) を測る。GPU を使う前に使用状況を確認し、host RAM のピークを見積もって告知する | サイズ、層、パッチ上限を決定 |
| 1 | 作者インデックス、分割、交絡度、sampler（CPU のみ） | §14 のテスト T3〜T5 が通る |
| 2 | モデル、view、SupCon、学習ループ、checkpoint、評価、REST / DB / デフォルト値 | 約3 step の smoke で loss が有限、評価が完走 |
| 3 | HSIC、凍結 tagger の offload 対応 | T6、T7 が通る。HSIC の有無で A/B 比較 |
| 4 | UI、推論 API | |
| 5 | 生成側での利用、保留事項の再検討 | |

## 14. テスト（CPU での契約テスト）

- **T1**: masked / padding 位置の画素値を変えても出力が変わらない（I-1）。
- **T2**: 同じ画像で padding の長さだけが違う場合、出力が一致する。
- **T3**: 同じ作品の batch では、作者が P 人の別人で、全画像が同じ copyright に属する。
- **T4**: 作者のハッシュ分割が、データの追加前後で既存の作者について変わらない。
- **T5**: 正例の重みと交絡度の計算。
- **T6**: nHSIC が、独立な入力でほぼ 0、依存する入力で正になる。
- **T7**: 重みが 0 より大きいカテゴリが tagger の語彙にないとき、学習の開始を拒否する。
- **T8**: 成果物の中身（配布用に作者一覧が含まれないこと）。

## 15. 保留事項と既知のリスク

**保留**

- サインの位置判定（当面はランダムな block mask で部分的に対応）
- 作者ごとの複数中心（sub-center）
- tagger 特徴の近傍検索による hard negative の探索
- prototype 補助 loss
- 共作の扱い
- tagger の予測確率を使った Jaccard（タグの付け漏れの補完）
- HSIC の対象に Meta の一部を加えること
- Danbooru のオンライン取得への対応（`tag_string_artist` が使える）

**リスク**

- HSIC の推定値は batch サイズに依存し、B が小さいと偏りが出る。
- 凍結した tagger の forward の分、1 step の計算とVRAMが増える。
- パッチ上限によって、細部の情報が失われる。
- Danbooru で alias として統合されていない同一作者（複数名義）は、偽の負例になる。
