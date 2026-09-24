# Style embedding encoder 設計書

Status: 実装中（Phase 0・1・2 完了。Phase 3 の HSIC 実装と短期 GPU A/B 比較は完了、効果の再現性は未確定。Phase 4 の学習 UI と CPU 推論 API は実装済みで、実 checkpoint の API 推論も確認済み。フロントエンド build は未検証。Phase 5 は SDXL の学習可能な IP-Adapter 型 attention bridge の基礎実装まで。学習済み重み・生成 API は未実装）。2026-09-23 の外部監査の指摘 4 件を §4・§5・§9・§10・§14 に反映済み

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
  style_labels.py       # caption → 作者・作品・キャラ（§5 の判定規則）
  style_model.py        # encoder + masked pooling + projector
  style_dataset.py      # 作者インデックス、分割、交絡度
  style_sampler.py      # P×K / 同じ作品 batch の sampler
  style_resume.py       # resume の判定（§9）
  style_views.py        # block mask の view（§7）
  style_ingest.py       # datasets.db からの読み込みと item キー
  style_loader.py       # torch Dataset / collate / batch キー
  style_artifacts.py    # 成果物・学習状態の保存と読み込み、保持ルール（§11）
  style_loss.py         # 重み付き SupCon（nHSIC は Phase 3）
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
  - `token_features(x) -> (features [B, N, D_tok], valid_mask [B, N])`：将来の cross-attention conditioning 用。mask 対象と padding の位置は **features を 0 にし**、`valid_mask` を 0 にして返す。利用側は `valid_mask` を attention mask として使う

**不変条件 I-1**: `embed` と `token_features` のどちらの出力も、mask 対象または padding の位置の画素値に依存しない。

SigLIP2 の encoder の attention は、mask された位置を key から除外する。しかし mask された token 自身の hidden state には、residual 経路を通って自身の画素情報が残る。したがって、pooling で除外するだけでなく、token 列を返す出力でもその位置を 0 にしない限り情報が漏れる。

> 参考：tagger の `CustomAttentionPooling`（`cls_dim` 指定時）は、以前 padding token も attention の対象にしていた。`main` の `277d827e` で `key_padding_mask` を適用するよう修正済みである（テスト: `backend/tests/tagger_custom_pooler_mask_test.py`）。style encoder の pooling も同じ不変条件を満たすこと。

## 5. データ

- **入力**: `dataset_configs`（tagger と同じ形式）。caption の解釈には既存の取り込み処理を使う。
- **カテゴリ判定**: 次の優先順位で決める。独自の判定規則は追加しない。
  1. caption の `tag_data` の要素に `category` が**明示されている**場合は、その値を使う（大文字小文字を正規化する）。追加中の作者データのように、taglist にない作者名でも `Artist` と明記されていれば学習対象になる。
  2. `category` が無い、または `Unknown` / `__lookup__` の場合は、`get_tag_group_manager().get_tag_group()` で判定する。
  - 既存の `_parse_caption_tags`（`tag_vocabulary.py`）は、`category` が無い要素を `General` として返す。このため、style 側ではこの関数の戻り値をそのまま使わず、「明示あり」と「無し」を区別して読む。
  - 1 と 2 の結果が食い違った場合は 1 を採用し、件数をログに出す。
  - **例外**: 明示された値が `General` で、manager が別のカテゴリ（None 以外）を返した場合は、manager の値を採用する。`backend/api/batch_operations.py` の tag_data backfill は、taglist に無いタグに一律 `General` を書き込むため、明示された `General` と backfill の既定値を区別できないからである。この件数は、食い違いとは別に数える。
  - Character / Copyright / Rating など、作者以外の内容タグの判定にも同じ規則を使う。
  - 同じ方針は `CONCEPT_BATCH_ORDER_DESIGN_2026-09.md` の Artist 抽出でも採られている。
- **作品として扱わないタグ**: `style_non_copyright_tags`（初期値 `original`）に含まれるタグは、Copyright の集合から外す。Danbooru の `original` は共通の作品を表さないので、同じ作品の batch、正例の Jaccard、交絡度、同じ作品に限った評価のどれにも使わない。外した件数はログに出す。
- **別名とカンマ入りタグ**: tagger の語彙構築（`tag_vocabulary.py`）と同じ順序で処理する。
  1. 既存の `CommaTagResolver` で、カンマで分割された断片を元のタグに結合する。結合されたタグは、`category_of()` の結果を確定したカテゴリとして扱う。
  2. `TagAliasResolver` で正規の名前に変換してから、カテゴリを調べる。正規の名前で見つからなければ、元の名前でも調べる。
  3. 作者・キャラ・作品のいずれの集合にも、正規の名前を入れる。

  manager（`get_tag_group`）は別名を解決しない。そのため、別名解決の前にカテゴリを調べると、taglist に旧名が無い作者（taglist の作者を指す別名のほとんど）を取りこぼす。
- **作者ラベル**: artist タグがちょうど1つの item だけを使う。
- **除外件数のログ**: 学習開始時に次の件数を出す。
  - 共作（作者が2人以上）
  - 作者不明（0人）
  - 判定の出どころ別の作者タグ数（`tag_data` の明示 / manager）
  - 判定の食い違い
  - どちらでも判定できなかったタグ
- **学習対象の作者**: 検証用の画像を除いた学習用の画像が `style_min_images_per_artist`（初期値 4）枚以上ある作者。これにより K ≤ `style_min_images_per_artist` であれば必ず K 枚を選べる。
- **インデックス**（学習開始時に一度構築）:
  - artist → items
  - copyright → artist → items
  - item → 内容タグの集合（Character ∪ Copyright）
  - item → サイン系タグの有無

  3M 規模に対応するため、Python の dict of list ではなく numpy の CSR 配列で持つ。規模は item 数に対して O(n)。
- **分割**:
  - **学習に出さない作者（主な評価用）**: 正規化した artist 名のハッシュで決める（`style_val_artist_ratio`、初期値 0.05）。データが増えても既存の作者の所属が変わらない。
  - **学習済み作者の、学習に使わない画像（参考値）**: `style_val_image_ratio`。§9 の安定した item キー（(dataset_id, 相対画像パス) 由来）のハッシュで決める。DB の id は再利用されるので使わない。ほぼ同じ画像が学習側と検証側にまたがる可能性があるため、この値は参考扱いとする。
- **dataset drift の検査と rescan**: tagger と同じ仕組みを使う。**未実装（後続の作業）**：tagger の仕組みは開始 route の中に `scope="tagger"` 固定で書かれているため、先に切り出しが必要。
- **重複パスの除去**: 設定した複数のデータセットで同じ画像ファイル（正規化した絶対パスが同じもの）が現れる場合は、1つだけ残す。学習に使った（pin された）コピーがあればそれを、無ければ dataset_id が最小のコピーを残す。データセットのルートが入れ子になっている場合に、同じ画像を二重に学習したり、学習用と検証用に分かれたりするのを防ぐ。

## 6. サンプリング

- 1 batch = 作者 P 人 × 各 K 枚。1枚から view を2つ作るので、埋め込みは 2PK 個になる（`style_artists_per_batch` P、`style_images_per_artist` K）。
- **同じ作品の batch**（割合 `style_same_copyright_batch_ratio`、初期値 0.5）:
  1. 学習対象の作者が P 人以上いて、各作者がその作品で K 枚以上持っている copyright を選ぶ。
  2. その中から P 人を選び、その作品の画像を各 K 枚選ぶ。
  3. 条件を満たす copyright がない場合は、通常の batch に戻す。
- **通常の batch**: 作者を n^α に比例する確率で選ぶ（n は作者の画像数、`style_artist_sampling_power` α、初期値 0.5）。作者ごとの K 枚は、その作者の中でキャラが分散するように選ぶ。
- **epoch**: 学習対象の画像数 ÷ (P·K) step と定義する。
- **run の長さ**: 学習開始時に `total_steps` = epochs × epoch 長を確定し、state に保存する。scheduler の horizon もこの値に固定する。データが増減して epoch 長が変わっても、`total_steps` と LR の曲線は変えない（最後の epoch が途中で終わることがある）。

## 7. View 生成

- **global view**: 画像全体。tagger と同じ NaFlex の前処理を使う。
- **masked view**: 同じパッチ grid に対する **block mask**。
  - 矩形の block を、有効パッチのうち mask された割合が比率 r に達するまで配置する。
  - mask は `pixel_attention_mask` を 0 にすることで行う。V1 では token を物理的に削除しない。
- **mask 比率**: r_a = r_min + (r_max − r_min)·s_a（`style_mask_ratio_min` 初期値 0.3、`style_mask_ratio_max` 初期値 0.75）。
- **作品との交絡度 s_a** = c_a · e_a
  - c_a（作品の集中度）: その作者の画像のうち、最も多い作品が占める割合
  - e_a（作品の独占度）: その作品の画像のうち、その作者が占める割合
  - 集計対象は、インデックスに入った全画像（学習用と検証用の画像の両方）。作品が複数付いた画像は、それぞれの作品に数える。どの画像にも作品が付いていない作者は s_a = 0。
  - 最も多い作品が複数あるときは、作品名の辞書順で最初のものを使う。
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
  - パスが空なら HSIC を無効にする。checkpoint 固有の vocabulary snapshot を必須とし、配布 metadata には checkpoint のファイル名と SHA-256 のみを記録する。
  - tagger の NaFlex 前処理は推論と同じ 256 パッチ上限を使う。tagger の出力は global view の style embedding とだけ比較する。
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
- **その他**: bf16、gradient checkpointing、`gpu_index`。attention の実装は `sdpa` / `eager`。transformers 5.17 の `Siglip2VisionModel` は FlashAttention-2 に対応していないため、`flash_attention_2` を指定した場合は黙って別の実装に切り替えず、学習の開始を拒否する。
- **GPU 共有の offload**: 凍結した tagger も、退避と復帰の対象に含める。
- **resume**: 次の値を state に保存する。
  - 重み、optimizer、scheduler、`global_step`、`total_steps`
  - epoch 番号と epoch 内の step 位置。batch は (seed, epoch, step) の純関数なので、RNG の状態そのものは保存しない。epoch の最後の step の後で保存した場合は、(epoch+1, 0) として扱う
  - **インデックスのハッシュ**: インデックスに入った全 item（検証用を含む）の ID、作者名、作品・キャラの集合、train / val の所属から計算する。入力の順序に依存しない。配列と名前はそれぞれ長さを前置して符号化する
  - **分割パラメータ**: `style_val_artist_ratio`、`style_val_image_ratio`、`style_min_images_per_artist`、分割 seed、作者名の正規化の版
  - **学習に使った item の snapshot**（`.npz`）: 学習用で、かつその時点で学習対象の作者に属していた item の集合。resume のたびに OR で合成し、縮小しない（削除された item も残す）。
    - キーは `DatasetItem.id` ではなく、(dataset_id, 正規化した相対画像パス) から作る安定した 64bit キー。SQLite は削除後に最大の id を再利用するため。
    - 別名表、カンマ結合の表、`normalize_tag` が更新されると、作者名が変わって作者の所属が移ることがある。実際の別名表では、作者の別名の約 4.7% が「学習側に振り分けられる名前」を「検証側に振り分けられる名前」に変える。そこで、snapshot にある item を1つでも持つ作者は、**名前のハッシュに関係なく学習側に固定する**（pin）。これにより、学習に使った画像が検証に混ざることを防ぎつつ、表の更新で resume が止まらない。固定した作者数と item 数はログに出す。
    - **限界**:
      - 固定は (dataset_id, 相対画像パス) で決まる item キーに付く。データセットの登録し直し（dataset_id が変わる）やファイルの移動・改名をすると、キーが変わって固定が外れ、学習に使った画像が検証に入り得る。run の途中ではこれらの操作をしない。
      - 学習に使った画像がすべてデータから削除され、その後に作者名が検証側の名前へ変わった場合、その作者の残りの画像（学習には未使用）は検証用に移る。
      - 同じ (dataset, 相対パス) を持つ item が2つあると、インデックスの構築は拒否される。Phase 2 の学習ループは、衝突した item を示すこと。
  - **sampler パラメータ**: P、K、`style_same_copyright_batch_ratio`、`style_artist_sampling_power`、sampler の seed、sampler のアルゴリズムの版、numpy のバージョン（完全一致。NEP 19 により、numpy の乱数列はリリース間で同一性が保証されないため）

  resume 時は、次の3通りで扱いを決める。

  | 状況 | 扱い |
  |---|---|
  | ハッシュと sampler パラメータがどちらも一致 | 保存した epoch 内の位置から、そのまま続行する |
  | ハッシュ（データの追加・削除）または sampler パラメータが不一致で、分割パラメータは一致 | **epoch の境界から再開する**。中断した epoch は終わったものとして扱う。次の epoch の先頭から、新しいインデックスで sampler を作り直す。sampler の RNG は (seed, epoch 番号) から決める。重み、optimizer、scheduler、`global_step`、`total_steps` はそのまま引き継ぐ |
  | 分割パラメータが不一致 | **resume を拒否する**。既存の作者の train / val 所属が変わり、検証用の作者が学習に混ざるため |
  | 名前の変更などで、学習に使った item の作者が検証側に振り分けられる | 拒否しない。その作者を学習側に固定する（上記の pin）。所属が変わるのでハッシュが変わり、epoch の境界から再開する |

  - ハッシュ分割（§5）により、データが増えても既存の作者の所属は変わらない。新しく加わった作者も、名前から一意に train / val のどちらかに入る。
  - インデックスが変わると評価の集合も変わる。metrics の行にインデックスの版を記録し、チャートで不連続点として区別できるようにする。
- **optimizer step ゼロ**: 一度も optimizer step が完了しなかった run は、完了扱いにせず重みも保存しない（tagger と同じ規約）。

## 10. 評価

`validate_every` epoch ごとに、次を計算する。

| 指標 | 内容 | 用途 |
|---|---|---|
| (a) 学習に出てこない作者での検索 | Recall@1/5、mAP@R。各画像を query にし、残りを検索対象にする | **主指標**。`best` の選定に使う |
| (b) 同じ作品に限った検索 | 検索対象を、query と copyright を1つ以上共有する画像に限る。対象条件は下記 | 内容の shortcut の検出 |
| (c) サインの有無による分割 | サイン系タグ（`style_signature_tags`、初期値 `signature, artist name, watermark, twitter username, web address`）の有無で **query を**分け、(a) をそれぞれ計算する。検索対象は分けない | サインによる精度の水増しの検出 |
| (d) masked view 同士の検索 | (a) を masked view で計算する | mask に対して頑健かの確認 |
| (e) 学習済み作者の、学習に使わない画像での検索 | (a) と同じ計算 | 参考値（ほぼ同じ画像による水増しの可能性あり） |

**query の対象条件**: どの指標でも、検索対象の中に次の2つがそろう query だけを数える。
- 同じ作者の別の画像（正解が1枚以上）
- 別の作者の画像（不正解が1枚以上）

mAP@R の R は、検索対象の中にある正解の枚数とする。(b) では、作品内に学習に出てこない作者が2人いても、各作者の画像がその作品に1枚ずつしかなければ、条件を満たさず対象外になる。指標ごとに、対象になった query の数と除外した数を併せて記録する。

## 11. 成果物

- **配布用**: `{name}_style_encoder.safetensors`（encoder、pooling、projector）と `{name}_metadata.json`。
  - metadata に含める情報：
    - base repo id
    - `style_embed_dim`、`style_feature_layers`、`style_max_num_patches`
    - 前処理の設定
    - HSIC に使った tagger checkpoint の公開可能な識別子とハッシュ（Phase 3）。ローカルパスは含めない
    - データセットの snapshot 情報
  - 作者の一覧や作者 ID の対応表は、配布用には**含めない**。
- **学習状態**（resume 用）: §9 で挙げた値（optimizer、epoch と step の位置、インデックスのハッシュ、分割パラメータ、sampler パラメータ、学習に使った item の snapshot）。
- **LoRA**: adapter 単体を保存し、merge したものを出力することもできる。
- **LoRA の step checkpoint**: adapter、optimizer、split snapshot、state、metadata のみを保存する。配布用の merged encoder は `latest` / `best` に保存する。旧形式のフル encoder 付き step bundle も resume できる。
- **保持ルール**: tagger と同じ（step checkpoint はまとめて削除し、`latest` と `best` は対象外）。

## 12. API / DB / UI

- **API**: OpenAPI を先に更新する。
  - `/api/v1/style-training/runs`：作成・一覧・取得・更新・start・stop・delete・metrics
  - `/api/v1/schema/style-training-defaults`
- **デフォルト値**: `STYLE_TRAINING_DEFAULTS` を `backend/api/param_defaults.py` に追加する（唯一の情報源）。
- **DB**: `training.db` に `style_training_runs` と `style_training_metrics` を新設する。
  - `tagger_training_runs` は tagger 固有の列（`best_f1`、`tag_vocabulary` など）を持つため、流用しない。
- **UI**: 学習タブの style モードからデータセット選択、主要設定、run の開始・停止・編集・削除、進捗・評価指標を扱う。フォームにない設定は API で指定する。
- **推論 API**（Phase 4）: `/api/v1/style/embed`、`/api/v1/style/similarity`。配布用 encoder と隣接 metadata を読み込み、CPU で推論する。
  - 2026-09-24、実 GPU run の `latest_style_encoder.safetensors` を API 経由で読み込み、512 次元・L2 norm 1・同一画像の cosine 1 を確認した。

## 13. フェーズ

| Phase | 内容 | 完了条件 |
|---|---|---|
| 0 | 測定（推論のみ）。素の SigLIP2 base / so400m、パッチ上限 256 / 1024 について、層ごとの特徴で (a)(b) を測る。GPU を使う前に使用状況を確認し、host RAM のピークを見積もって告知する | サイズ、層、パッチ上限を決定 |
| 1 | 作者インデックス、分割、交絡度、sampler（CPU のみ） | §14 のテスト T3〜T5、T9 が通る |
| 2 | モデル、view、SupCon、学習ループ、checkpoint、評価、REST / DB / デフォルト値 | T1、T2、T10、T11 が通り、約3 step の smoke で loss が有限、評価が完走 |
| 3 | HSIC、凍結 tagger の offload 対応 | T6、T7 が通る。HSIC の有無で A/B 比較 |
| 4 | UI、推論 API | |
| 5 | 生成側での利用、保留事項の再検討 | 対象の生成 architecture と変換器の学習契約を決め、strength 0 の既存出力一致・style と内容の両評価を通す |

## 14. テスト（CPU での契約テスト）

- **T1**: masked / padding 位置の画素値を変えても、`embed` と `token_features` のどちらの出力も変わらない。`token_features` の該当位置が 0 で、`valid_mask` が 0 である（I-1）。
- **T2**: 同じ画像で padding の長さだけが違う場合、出力が一致する。
- **T3**: 同じ作品の batch では、作者が P 人の別人で、全画像が同じ copyright に属する。
- **T4**: 作者のハッシュ分割が、データの追加前後で既存の作者について変わらない。
- **T5**: 正例の重みと交絡度の計算。
- **T6**: nHSIC が、独立な入力でほぼ 0、依存する入力で正になる。
- **T7**: 重みが 0 より大きいカテゴリが tagger の語彙にないとき、学習の開始を拒否する。
- **T8**: 成果物の中身（配布用に作者一覧が含まれないこと）。
- **T9**: カテゴリ判定の優先順位。
  - taglist にない名前でも、`tag_data` に `Artist` と明示されていれば作者として採用される。
  - `category` が無い要素は manager で判定される（`General` 扱いにならない）。
  - 食い違いは `tag_data` 側が採用され、件数に数えられる。
- **T10**: resume の境界。
  - ハッシュが一致すれば、epoch 内の位置から続行する。
  - ハッシュが不一致なら、次の epoch の先頭から再開し、`global_step` と `total_steps` が保たれる。
  - 分割パラメータが不一致なら、resume を拒否する。
- **T11**: 評価の対象条件。正解または不正解が検索対象にない query が除外され、除外数が記録される（(b) で、作者ごとの画像が作品内に1枚ずつの場合を含む）。

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

## 16. Phase 0 の測定結果（2026-09-23）

凍結した素の SigLIP2 NaFlex で、各層の出力を有効 token で平均 pooling し、作者検索の精度を測った。学習後の性能ではなく、層・サイズ・パッチ上限の比較のための代理指標である。

- **データ**: datasets.db の全データセット（画像が存在しない 3 データセットを除く 25 件）、3,756,255 件を取り込み。インデックスは 3,386,541 item、作者 327,532 人（検証用 16,342 人）。
- **評価セット**: 検証用の作者から 400 人・3,957 枚（1人最大 16 枚、seed 0）。(a) の対象 query は 3,957 件、(b) の対象 query は 2,434 件。
- **chance 補正**: (b) は検索対象が約 70 分の 1 と小さいため、生の値は (a) と比較できない。乱数の埋め込みで測った chance（(a) mAP@R 0.0008、(b) mAP@R 0.147）で補正した lift を併記した。

| 構成 | (a) mAP@R の最大（層） | (a) R@1 の最大（層） | (b) mAP@R の最大（層） | 最終層の (a) mAP@R | GPU forward（img/s） | 最大 VRAM |
|---|---|---|---|---|---|---|
| base・256 | 0.1252（L12） | 0.4122（L12） | 0.4439（L12） | 0.1252 | 880 | 0.76 GB |
| base・1024 | 0.1330（L12） | 0.4380（L12） | 0.4557（L12） | 0.1330 | 305 | 1.57 GB |
| so400m・256 | 0.2021（L21） | 0.5620（L21） | 0.5602（L21） | 0.1973 | 327 | 2.88 GB |
| so400m・1024 | 0.2335（L21） | 0.6027（L20） | 0.5922（L20） | 0.2255 | 58.5 | 3.85 GB |

層番号は L0 = embeddings。so400m は全 27 層。

- **artist 単位の paired bootstrap**（2,000 回、95% 区間、(a) mAP@R の差）:
  - so400m L21 − base L12: 256 で +0.077 [0.068, 0.086]、1024 で +0.101 [0.090, 0.112]
  - so400m L21 の 1024 − 256: +0.031 [0.027, 0.036]
  - base L12 の 1024 − 256: +0.008 [0.005, 0.010]
- **so400m の層**: L18〜L25 は最大値から 0.014 以内に収まる。最終層 L27 との差は、1024 で +0.008（区間は 0 を含まない）、R@1 では 256・1024 とも +0.033。
- **内容の shortcut の指標**: (b) の lift を、同じ query での (a) の lift で割った比。どの構成でも (b) が (a) を下回ることはなく、比は深い層ほど小さくなった（so400m・1024 で L0 の 5.00 → L21 の 2.08）。gallery の大きさの影響が残るため、ゼロ点は定まっていない。
- **限界**: 凍結特徴による代理指標であり、学習後の順位が同じになるとは限らない。評価セットは seed 1 つ。学習時の backward のコストは測っていない。

**この結果に基づく決定**: encoder は `google/siglip2-so400m-patch16-naflex`、`style_feature_layers` は `[21]`（L22 以降の計算を省ける）、`style_token_dim` は 1152 とする。パッチ上限（256 / 1024）は、精度とコストの兼ね合いで repo owner が決める。

### 16.1 学習の smoke と学習コスト（2026-09-23）

- **smoke**: so400m-naflex、LoRA、デフォルト設定、1 データセット（学習用 1,097 item、1 epoch = 17 step）。step 3 で停止 → 「そのまま続行」で再開して step 5 → さらに再開して step 17（epoch の終わり）で評価。17 step の loss はすべて有限（4.53 → 3.41）。配布用 metadata に作者名とローカルパスは含まれなかった。
- **学習コスト**（学習時、backward を含む。step 4〜13 の平均、P16×K4×2 view、`num_workers` 4）:

| 構成 | 秒/step | 画像/s | 最大 VRAM（allocated / reserved） | 1 epoch の時間（学習用 2,867,931 item ÷ 64 = 44,811 step） |
|---|---|---|---|---|
| LoRA・256 | 1.31 | 48.8 | 6.5 / 9.2 GB | 16.3 h |
| full・256 | 1.55 | 41.2 | 7.2 / 10.4 GB | 19.3 h |
| LoRA・1024 | 6.78 | 9.4 | 20.7 / 24.4 GB | 84.4 h |
| full・1024 | 7.91 | 8.1 | 22.1 / 24.9 GB | 98.5 h |

- 1024 パッチは、PyTorch の既定のメモリ確保設定では 30 GB の上限内で断片化により OOM になった（allocated は約 20 GB）。Windows では `expandable_segments` が使えず、`PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.8` で収まった。上の 1024 の数値はこの設定での測定。
- `num_workers` 2 では、データ読み込みの待ち時間が step 時間の 23〜37% を占めた（4 では 0.6% 以下）。

## 17. Phase 3 の実データ確認と短期 A/B（2026-09-24）

dataset 39（`image-tag-caption-dataset`）の 33,099 item を調べ、tags caption のある 18,634 件から 9,926 item・7,444 作者を index 化した。学習対象は 209 作者・1,477 item で、16 作者×4 枚なら 1 epoch は 23 step。条件を満たす copyright が 1 件あり、seed 0 の 3 epoch・69 batch 中 30 batch が同じ作品の batch になった。各 batch で作者 16 人が異なり、64 枚すべてに選ばれた copyright があることを実データで確認した。

同じ index hash `9df70dad080a8b3dfa3eb5b000371830`、初期重み、seed 0、so400m・LoRA・256 パッチ、3 epoch で HSIC の有無だけを変えた。`num_workers=0` は Windows の実験スクリプトで子プロセスを使わないため。評価は検証作者の 490 枚から検索可能な 176 query（同じ作品内では 58 query）。HSIC 側は Artist タグを含まない tagger の `best_f1.safetensors`（SHA-256 `bd48dd2e60717678434c62f71941a55d54d3823db1ac9994a3c12bcd9152c626`）を凍結し、Character / Copyright / Rating を各 0.1、Quality を 0 とした。

| 設定 | step 23 mAP@R | step 46 mAP@R | step 69 mAP@R | step 69・同じ作品内 mAP@R |
|---|---:|---:|---:|---:|
| HSIC なし | 0.2511 | 0.1603 | 0.1581 | 0.5932 |
| HSIC あり | 0.2207 | 0.2093 | 0.1734 | 0.5327 |

両 run の 69 step の loss はすべて有限で、HSIC loss は 0.082〜0.204。HSIC ありの GPU 使用量は学習中の観測時点で 16.7 GB。主指標は step 23 では HSIC なしが高く、step 46・69 では HSIC ありが高い。同じ作品内の最終指標は逆に HSIC なしが高い。単一 seed・短い学習・有効 query 176 件なので、効果が安定したとは判断しない。HSIC は opt-in のままとし、採用判断には複数 seed とより大きい評価集合が必要。

## 18. Phase 5 の接続条件（実装前の確認）

style encoder は画像から正規化された 512 次元のベクトルを出す。一方、既存の `core.inference.reference_style` は参照画像を VAE latent にして denoise step ごとに生成モデルの Key/Value を取り出す。両者は入力形式も学習した表現空間も異なり、ベクトルをそのまま既存の参照画像経路へ渡すことはできない。`controlnet_manager.py` の IP-Adapter 重み判定はファイル検出であり、style ベクトル用の投影器や学習経路ではない。

最初の対象として SDXL を仮定する場合は、style encoder と生成モデルを凍結し、512 次元から SDXL の条件付けに必要な token / attention Key/Value へ写す変換器を別 artifact として学習する。参照画像と生成対象には同じ作者の異なる画像を使い、内容の異なる組を含める。style 条件の dropout と強度 0 の経路を設け、強度 0 では既存の生成結果が一致することを確認する。評価は未学習作者での style 類似度、内容保持、同じ作品に限った指標を既存の画像参照 style transfer と比較する。変換器の重み形式、resume、API/default/UI は対象 architecture が決まってから OpenAPI-first で定義する。

実装順は SDXL の IP-Adapter 型、次に Qwen-Image 2.1 の既存参照画像経路。SDXL の `sdxl_ip_adapter.py` は 512 次元から画像 token と各 cross-attention 層の専用 K/V を学習し、`UnifiedAttnProcessor` で通常のテキスト attention 出力に独立加算する。CFG の負側には加算せず、強度 0 は既存出力と一致する。重み artifact は style encoder と SDXL model の SHA-256 および U-Net 層構成で照合する。`sdxl_ip_training.py` には別画像・同一作者で内容の異なる参照を優先するペア選択と、凍結 U-Net の denoising loss を用意した。CPU の小型モデルで勾配と強度 0 を確認済み。実データの学習 runner、resume、学習済み artifact、未学習作者での生成比較が揃うまで生成 API には公開しない。

Qwen-Image 2.1 の既存 style transfer は segmented attention の参照画像 K/V を使う。参照画像が backend mixin から vendor pipeline に届き、capture/inject が出力を変えることを既存 CPU テストで確認した。この経路は画像そのものを入力とし、style encoder の 512 次元を直接消費しない。SDXL の投影器も流用しない。SDXL の学習・評価を先に終えた後、Qwen では既存参照画像経路を基準に style encoder を使った参照画像選択と生成評価を進める。
