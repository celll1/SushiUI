# コンセプト巡回バッチ順序 設計書

Status: 提案・未実装（2026-09-23）

## 1. 目的と検証すべき仮説

大規模な画像データセットで Character / Artist の画像が少数ずつ散在すると、通常のランダム順では同じコンセプトの次の画像まで長い間隔が空く。対象画像を一時的に集中して学習しつつ、エポック内で全データを巡回するバッチ順序を追加する。途中チェックポイントでも多くのコンセプトに学習機会を与えることが主目的である。

集中学習が適応速度、保持、あるいは contrastive loss 使用時の unconditional 分岐の安定性を改善するかは未検証の仮説である。順序だけを変えても画像ごとの露出回数は増えない。再提示を有効にする場合は追加の optimizer step として会計し、順序効果と露出回数効果を分けて評価する。

この機能は生成モデルの通常の画像学習を対象とする。タグ分類器や style encoder のサンプラーは対象外とする。

## 2. 現状と変更境界

- `backend/core/training/priority_training.py` は手指定エントリを先頭一致で分類し、エントリ順に priority batch を作る。`multiplier` はその全体をエポック先頭で繰り返す。タグ照合は `tag_data` を優先し、なければキャプションの部分文字列にフォールバックする。
- `backend/core/training/base_trainer.py` は各エポックで `all_items` から batch を作る。通常の bucket path は `BucketManager.build_batch_indices()` がバッチ単位でシャッフルし、priority path は priority → normal の順に結合する。解像度 bucket、参照画像、動画・音声、SenseNova のタスク分割、途中再開などの後段処理がある。
- `backend/core/training/train_runner.py` は学習用 snapshot に主キャプションの `raw_caption`、`tag_data`、`is_tags_format` を保持する。追加キャプション形式は要求されたものだけ `_captions_by_type` に保持する。
- `backend/core/training/base_trainer.py::_compute_dataset_fingerprint` は画像パスを主に検査し、キャプション変更を意図的に無視する。コンセプト分類が順序を決める場合、この既存 fingerprint だけでは途中再開を保証できない。

新機能は `priority_training` の既定挙動を変えない、独立した `concept_batch_order` 設定とする。同時指定は初版で明示的に拒否する。優先度、重複回数、エポック先頭固定という意味が異なり、黙って合成すると学習量と順序が読めなくなるためである。

## 3. ユーザー向け設定案

API の `concept_batch_order` は省略時 `null`、または以下の object とする。値は**提案値**であり、実装時に `backend/api/param_defaults.py` の `TRAINING_DEFAULTS` を唯一の既定値として定義する。`openapi.yaml` は具体的な schema、制約、例を持たせ、`backend/api/routes.py`、`frontend/src/utils/api.ts` と学習画面を同期する。

| フィールド | 提案既定値 | 意味 |
|---|---:|---|
| `enabled` | `false` | 機能の有効化 |
| `category` | `character` | `character` / `artist`。両カテゴリ同時巡回は初版の対象外 |
| `min_items_per_concept` | `8` | 所属画像がこれ未満のタグは通常プールへ戻す |
| `include` / `exclude` | `[]` | 正規化済みタグ名による対象制限。`include=[]` は該当カテゴリ全体 |
| `focus_batches` | `8` | 同じコンセプトを続ける最大バッチ数。残りは次の巡回で扱う。`0` はそのコンセプトを最後まで扱う |
| `local_swap_window` | `2` | 集中バッチの元位置から前後に動ける最大数。`0` なら揺らぎなし |
| `background_interval` | `8` | 集中バッチ何個ごとに通常プールのバッチを1個挟むか。`0` は挿入しない |
| `replay_interval` | `0` | 既出コンセプトのバッチを再提示する間隔。`0` は追加露出なし |
| `caption_aliases` | `{}` | コンセプト名から自然言語表記の明示的な別名列への写像 |
| `match_natural_language` | `false` | 自然言語キャプションを照合対象に含める |

数値の範囲と上限は API で検証する。実装時には上記の複合設定を typed schema として定義し、未知フィールドを拒否する。既定値を frontend や trainer に重複記載しない。

初版では Character と Artist を別 run で試せるようにする。どちらも選べる複合モードを後から追加する場合、カテゴリ間の優先順位、同一画像の割当、露出回数を別途定義する。

## 4. コンセプトの抽出と割当

1. 学習開始時、または dataset reload 時に、学習 snapshot の各画像を走査する。`is_tags_format=true` かつ `tag_data` があれば `category` が `Character` / `Artist` のタグだけを抽出する。カテゴリ名は大文字小文字を正規化し、タグ名には既存の `normalize_tag_for_matching` を使う。
2. `tag_data` がないタグ形式は、既存 tag group manager によるカテゴリ判定が可能な場合だけ候補にする。タグ列の単純な部分文字列検索でカテゴリを推定しない。カテゴリが確定できない画像は通常プールへ置き、件数を報告する。
3. 対象制限と `min_items_per_concept` を適用する。件数は**各画像が持つ候補タグ**で数え、最終割当件数とは区別してログに出す。
4. 複数の対象タグを持つ画像はエポック内の基本巡回では一度だけ使う。候補のうち、現在の割当枚数が最小のコンセプトへ置き、同数なら run seed と安定した画像 ID から決める。割当はエポックごとに再計算できるが、同じ seed と snapshot から同じ結果になることを保証する。共有画像だけから成る概念は独立した集中区間を保証できないため、その件数を表示する。
5. 対象外の画像は通常プールへ置く。全画像がちょうど一方に属することを不変条件とする。`include` に未出現タグがあっても silent no-op にせず、0 件として報告する。

大規模データでは全組み合わせのタグ対を作らない。画像ごとの既存 item 参照とコンセプト ID の索引を使い、抽出は画像数と実際のタグ数に比例する処理にする。対象数・メモリ使用量を初期化時に計測する。

## 5. 自然言語キャプションの照合

初版は曖昧な名前推定を行わない。`caption_aliases` に列挙した表記だけを照合し、表記の空白・アンダースコア、Unicode 正規化、大文字小文字を揃える。英字列は単語境界で照合し、`Ann` が `Anna` に一致しないようにする。姓名の逆順、ミドルネーム省略、愛称、他言語表記は明示的な別名として登録する。括弧付き作品名を自動で落とさない。

`match_natural_language=true` のときは、選択された主キャプションが自然言語形式ならその原文を照合する。タグ形式が主キャプションで、自然言語も照合したい場合は `train_runner.py` の snapshot に必要な `natural_language` キャプションを保持する。後段の caption dropout / shuffle 後の `caption` は照合に使わない。自然言語しかないデータでは、`caption_aliases` のキーを対象コンセプトとして使う。カテゴリを自然文から自動推測しないため、キーは選択した `category` の名前に限る。複数の別名が一致したときも §4 の一画像一割当を適用し、曖昧一致の件数を報告する。

既存 priority training の `caption_contains` の緩い部分文字列検索は変更しない。

## 6. エポック内スケジューラ

### 6.1 基本巡回

- 各画像を、現在の解像度 bucket と、モデルが要求する追加の同質条件（参照画像の有無など）を守って batch 化する。コンセプトごとに同じ bucket の画像をシャッフルして batch 化し、サイズ不足の最終 batch も保持する。別コンセプトを混ぜて batch を満たすことはしない。
- エポックごとにコンセプト順と各コンセプト内の batch 順を変える。一つのコンセプトから最大 `focus_batches` 個を続けて取り、次のコンセプトへ進む。残りは次の巡回で扱い、巡回ごとにコンセプト順を変える。これにより途中チェックポイントで扱ったコンセプト数を増やす。`focus_batches=0` なら各コンセプトを最後まで扱ってから次へ進む。
- `background_interval` ごとに通常プールから1 batch を挟む。通常プールが尽きた場合は挿入をやめ、残りの集中 batch を続ける。最後に残った通常 batch も同じエポック内で消化する。
- `local_swap_window` は、**batch 化後**、隣り合うコンセプトの境界付近の集中バッチに適用する。各バッチの移動幅を元位置から最大指定数に制限し、両側の画像が少し交じることを許す。通常プールと再提示 batch の配置は先に確定し、それらは動かさない。bucket の異なるバッチは交換できるが、個々の batch の中身は変更しない。揺らぎ適用後も各コンセプトの大部分が連続することを確認する。
- 基本巡回では、unfittable と判定されたものなど既存の除外条件を除き、全入力画像をエポックに一度だけ出す。追加の optimizer step は作らない。`max_steps` 等でエポック途中終了する場合は、未訪問コンセプトがあることを UI / ログで分かるようにする。

### 6.2 任意の再提示

`replay_interval > 0` のときだけ、既に基本巡回で提示済みのコンセプト batch から1 batch を追加で挟む。候補はコンセプト間で偏らないように選び、同じ画像を短い間隔で連続提示しない。再提示 batch は基本巡回から消さず、総 batch 数、追加 batch 数、コンセプトごとの露出回数を表示する。露出増による効果を分離できるよう、評価では `replay_interval=0` を必ず基準にする。

再提示の選択範囲、選択 RNG、および必要な履歴は checkpoint から復元できるようにする。初版の実装を簡単にするなら、エポック開始時に完全な batch plan を決定し、plan 自体を保存する代わりに同じ snapshot・設定・RNG state から再生成する。

## 7. 既存機能との接続・再開

| 接点 | 実装条件 |
|---|---|
| 解像度 bucket / 動的 crop | `CropPlanner` による当該エポックの crop spec と bucket 割当を先に確定し、その結果で concept batch を作る。priority path の現行「crop 無効化」を流用しない。 |
| 参照画像 / VE | `separate_by_reference` を保持し、後段の VE 用全体 shuffle が集中順序を壊さないようにする。混合 batch が生じたら分割後も位置を維持する。 |
| SenseNova | タスク別 batch 化が順序を変えたり画像を落としたりするため、初版は対象外として設定時に明示的に拒否する。対応時はタスク割当を先に確定してから concept plan を作る。 |
| 動画・音声 | 初版は対象外。画像と併用する run では、既存の動画・音声 batch を通常プールに含めるのではなく、設定を拒否する。 |
| online Danbooru 注入 | 追加 batch を定期挿入する既存処理は維持する。基本巡回の出現回数と注入 batch 数を別々に表示する。 |
| priority training | 両方が有効ならエラー。既存の priority 設定と UI は維持する。 |
| 勾配累積 / LR schedule | 既存の batch と optimizer step の会計を使う。再提示有効時の step 増を epoch 長と scheduler に反映する。 |
| 途中再開 | `base_trainer.py` の batch 生成前 RNG snapshot を利用し、dataset fingerprint に加えて設定 hash と分類に使ったキャプション・タグ情報の hash を照合する。異なれば旧 batch index を適用せず、そのエポックの先頭から再計画する。 |

無効時は既存 batch path の順序と乱数消費を変えない。新しい順序計画には run seed と epoch から導く専用 RNG を使い、他の dropout / sampling 用 RNG を余計に進めない。再開時は batch plan の digest をログと state に残し、一致を確認する。

## 8. API・UI・実装箇所

- `backend/core/training/concept_batch_order.py`（新規）: 設定検証、索引、分類、batch plan、digest。大きな `base_trainer.py` には呼び出しと既存機能との接続だけを置く。
- `backend/core/training/train_runner.py`: 必要な自然言語キャプションの snapshot 取り込み。`priority_training` の既存経路は変更しない。
- `backend/api/param_defaults.py`、`backend/api/routes.py`、`openapi.yaml`: 設定の唯一の既定値、入力検証、API 契約。
- `frontend/src/utils/api.ts`、`frontend/src/components/training/TrainingConfig.tsx`: カテゴリ、集中長、揺らぎ幅、通常データ間隔、再提示間隔、自然言語別名の入力と設定保存・復元。priority training との同時有効化は UI でも説明する。
- 学習ログまたは事前プレビュー: 対象コンセプト数、対象画像数、通常画像数、重複候補数、0 件の指定、batch 数、追加再提示数、予定巡回順の先頭数件を表示する。百万枚規模で全順序を UI に送らない。

API 追加時には `openapi.yaml` を先に更新し、training request、保存・再開、frontend の送受信を揃える。学習パラメータの既定値を別ファイルへ複製しない。

## 9. 受け入れ条件と評価

### 実装の受け入れ条件

1. `replay_interval=0` では、除外条件を適用した各画像が各エポックにちょうど一度現れ、基本 batch 数と総露出回数がログから検証できる。
2. Character / Artist のカテゴリ、別名、大文字小文字、姓名別表記、境界一致、複数対象タグ、タグ情報のない画像を fixture で検証する。
3. 異なる解像度・参照画像の有無を含むデータでも不正な混合 batch がない。動的 crop を有効にしたエポックごとの bucket 変化を維持する。
4. 同じ seed / snapshot / 設定なら、通常実行と途中再開後の未実行 batch 列および optimizer step 数が一致する。キャプション・設定変更は再開順序の無効化として検出する。
5. 無効時の既存順序、priority training、前処理結果は変化しない。百万枚規模の項目数で、索引構築時間と追加メモリを測定する。

### 学習効果の評価

同じデータ、seed、optimizer step、キャプション処理で、通常シャッフルと `replay_interval=0` の巡回順序を比較する。別実験で再提示を有効にし、**露出回数を揃えた対照群**を置く。対象は頻出・希少な Character と Artist、既存知識の保持用プロンプトを含める。途中と最終 checkpoint でタグ応答、類似キャラクターへの混同、既存タグの劣化を評価する。contrastive loss / unconditional 関連の効果は、その設定を変えた実験で別に測る。改善を前提に既定有効化しない。
