# SenseNova und ブランチは訓練が必要か — 実測調査

日付 2026-09-02 / ベース `M:/model/sensenova/sensenova_int8.safetensors`（plain int8、588 Int8Linear、weight-only dequant）
プローブ `backend/core/training/probes/sensenova_und_discrimination.py`
材料 dataset 37 = run 121/122 が実際に学習した kouyoku_senki_exs-tia、28ペア
（キャプション集合ビルダと生 JSON は `tmp/sensenova_und/` に残置。tmp/ は追跡外なので
再現するときはプローブ docstring の手順で作り直す）

## 結論

**und ブランチに情報欠落はない。表現ボトルネックを理由に und を訓練する必要はない。**

## 1. 構造（コード確定）

生成ブランチはキャプション文字列を一切見ない。見るのは und ブランチが作る 42層ぶんの
prefix K/V (`[B, H_kv=8, S_text, D=128]`) だけである。その経路に：

- プーリングなし、truncation なし、固定長ボトルネックなし、要約トークンなし、
  リサンプラ／Q-Former なし。`tokenizer(query, return_tensors="pt")` に
  `max_length` も `truncation` も渡っていない。
- 唯一ボトルネックが隠れうる last_hidden_state は捨てられる（`del prefix_hidden`）。
- 画像トークンは `causal=False` で prefix 全位置を層対応で全参照する。
  sliding window は flash 経路で一切参照されない。

つまり「設計上の情報欠落」は存在しない。残る実測対象は顕著性と int8 量子化の床。

## 2. KV アーム — 分離性

| family | K rel.Frobenius | K pooled cos-dist | V pooled cos-dist |
|---|---|---|---|
| identical（健全性） | **0.00000** | 0.0 | 0.0 |
| minimal_edit（83タグ中1タグ置換） | 0.0377 | 1.6e-5 | 1.3e-4 |
| real_pair（実データ、3-6タグ差） | 0.355 | 1.9e-4 | 1.0e-3 |
| unrelated（天井） | — | 7.5e-3 | 3.9e-2 |

- **編集位置より前の差分が全42層で厳密に 0.00000。** 因果 prefix として整合し、
  位置アラインメントが正しいことの検証になる。
- **編集トークン位置の K の相対移動 = 0.32〜0.58。** 83タグ中1語を替えるだけで
  その位置の K が32〜58%動く。埋もれていない。
- **深さ方向で減衰しない。** layer 0 で 0.016〜0.030 → layer 30-40 で 0.05〜0.074。
  「深層で細部が洗い流される」は成立しない。
- **希釈ラダーがフラット。** 同一の髪色編集を 5/11/21/41/83 タグの中に置いた場合の
  K relF = 0.042/0.047/0.043/0.044/0.040、K@edit = 0.48→0.55（微増）。
  タグ数が増えても1タグの KV 信号は薄まらない。
- pooled cosine の順序は identical < minimal_edit < real_pair < unrelated で単調。

注意：reorder 統制（同一タグ集合の並べ替え＝意味不変）の raw 距離のほうが
detail 編集より大きい（0.077/0.304/0.499 対 0.038）。タグ丸ごとの位置移動が下流の
多数の位置を書き換えるためで、raw 距離が表層形に支配されていることを示す。
raw 距離単独では意味的分離の証明にならない → 3節と4節。

## 3. readout アーム — 生成側が実際に読むか

512px、固定ノイズ1サンプル、t=0.5、CFGなし。ペア間で変わるのはプロンプトだけ。

| family | x0 rel.Frobenius | x0 cos-dist |
|---|---|---|
| identical | 0.00000 | 0.0 |
| reorder（意味不変） | 0.0453 | 9.8e-4 |
| minimal_edit | 0.0467 | 9.9e-4 |
| real_pair | 0.0658 | 1.9e-3 |
| unrelated | 0.302 | 4.1e-2 |

生の値では detail 編集と reorder がほぼ同じに見える。だが **KV の移動量で正規化する**
と分かれる（x0 relF / KV relF）：

| | 比 |
|---|---|
| minimal_edit（意味変化） | 0.66 – 1.72（平均 1.08） |
| reorder（純粋な並べ替え） | 0.56 / 0.14 / **0.10** |

**K/V の移動1単位あたり、意味的な1タグ編集は純粋な並べ替えの約7〜10倍、画像予測を動かす。**
生成ブランチは prefix の意味成分を選択的に読んでいる。

希釈は readout 側にだけ現れる：同じ髪色編集の x0 relF が 5タグで 0.086、11タグで 0.229、
83タグで 0.039。KV 信号は薄まらないが、画像予測全体への寄与は
「83分の1のタグ」相応に縮む。これは欠陥ではなく期待どおりの挙動。

## 4. QA アーム — 意味が正しく載っているか

und ブランチ自身の LM ヘッドに、同じタグ列について質問した（text-only、greedy）。

| 編集 | 髪色 | 目色 | 髪長 | 人数 |
|---|---|---|---|---|
| identical | blonde/blonde | blue/blue | long/long | 1/1 |
| blonde_hair→black_hair | **blonde→black** | blue/blue | long/long | 1/1 |
| blue_eyes→green_eyes | blonde/blonde | **blue→green** | long/long | 1/1 |
| long_hair→short_hair | blonde/blonde | blue/blue | long/long | 1/1 |
| 1girl→2girls | blonde/blonde | blue/blue | long/long | **1→2** |
| white_panties→black_panties | blonde/blonde | blue/blue | long/long | 1/1 |
| thighhighs→pantyhose | blonde/blonde | blue/blue | long/long | 1/1 |
| +smile / −on_side | blonde/blonde | blue/blue | long/long | 1/1 |

**変わるべき軸だけが変わり、他軸への漏れがゼロ。**
`long_hair→short_hair` で答えが変わらなかったのは失敗ではない：編集後のキャプションには
`very_long_hair` が残っており（`['blonde_hair','hair_between_eyes','hair_bow','short_hair','very_long_hair']`）、
矛盾したタグ列に対して 'long' と答えたのは正しい解決である。

und ブランチは danbooru 形式のアンダースコア付きカンマ区切りタグ列を、特別な前処理
なしにそのまま正しく解釈している。自然言語で事前学習された LLM をタグ列に適応させる
ための訓練も、この証拠に照らせば不要。

## 5. 限界・注意

- 測定したのは**分離性と可読性**であって、生成ブランチがそれを**描けるか**ではない。
  描画の失敗は gen 側の問題として切り分けられる。
- int8 weight-only 量子化を含んだ本番条件で測っている（プローブの人工物ではない）。
- readout は CFG なし・ノイズ1サンプル・t=0.5・512px の1点。CFG は cond−uncond 差を
  増幅するので、実生成での差は**これより大きくなる**方向。
- トークン長が変わるペアは画像トークンの RoPE `t`（= text_length）も動くため、
  位置整合メトリクスからは除外し、順序非依存の pooled cosine のみ報告した。
- 全てベースモデルの測定。full-FT 後の und が壊れていないかは別問題。

## 6. 示唆

und を訓練する動機が「細部が KV の時点で潰れている」ことなら、その前提は成立しない。
むしろ既存の診断（3 run すべてで CFG 増幅された cond ドリフト）と合わせると、
**正常に機能しているコンポーネントを訓練して壊すリスクのほうが大きい**。
`train_text_encoder` は既定 off のままが妥当で、訓練予算は gen 側に置くべき。
