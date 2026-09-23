# コンセプト巡回バッチ順序 設計書

Status: 提案・未実装（2026-09-23）

## 1. 目的と検証すべき仮説

大規模な画像データセットで Character / Artist の画像が少数ずつ散在すると、通常のランダム順では同じコンセプトの次の画像まで長い間隔が空く。対象画像を一時的に集中して学習しつつ、エポック内で全データを巡回するバッチ順序を追加する。途中チェックポイントでも多くのコンセプトに学習機会を与えることが主目的である。

集中学習が適応速度、保持、あるいは contrastive loss 使用時の unconditional 分岐の安定性を改善するかは未検証の仮説である。順序だけを変えても画像ごとの露出回数は増えない。再提示を有効にする場合は追加の batch と MNT iteration として会計し、勾配累積を考慮した optimizer update 数も示す。順序効果と露出回数効果は分けて評価する。

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
| `background_placement` | `front` | `front` は集中学習を早期に消化、`spread` は集中区間をエポック全体へ分散 |
| `background_interval` | `8` | `front` 専用。集中バッチ何個ごとに通常プールのバッチを1個挟むか。`0` は挿入しない |
| `replay_interval` | `0` | 既出コンセプトのバッチを再提示する間隔。`0` は追加露出なし |
| `caption_aliases` | `{}` | コンセプト名から自然言語表記の明示的な別名列への写像 |
| `match_natural_language` | `false` | 自然言語キャプションを照合対象に含める |

数値の範囲と上限は API で検証する。実装時には上記の複合設定を typed schema として定義し、未知フィールドを拒否する。既定値を frontend や trainer に重複記載しない。

初版では Character と Artist を別 run で試せるようにする。どちらも選べる複合モードを後から追加する場合、カテゴリ間の優先順位、同一画像の割当、露出回数を別途定義する。

## 4. コンセプトの抽出と割当

1. 学習開始時、または dataset reload 時に、学習 snapshot の各画像を走査する。`is_tags_format=true` かつ `tag_data` があれば `category` が `Character` / `Artist` のタグだけを抽出する。カテゴリ名は大文字小文字を正規化し、タグ名には既存の `normalize_tag_for_matching` を使う。
2. `tag_data` がないタグ形式は、既存 tag group manager によるカテゴリ判定が可能な場合だけ候補にする。タグ列の単純な部分文字列検索でカテゴリを推定しない。カテゴリが確定できない画像は通常プールへ置き、件数を報告する。
3. 対象制限と `min_items_per_concept` を適用する。件数は**各画像が持つ候補タグ**で数え、最終割当件数とは区別してログに出す。
4. 複数の対象タグを持つ画像はエポック内の基本巡回では一度だけ使う。割当処理は `(dataset_unique_id, image_path)` の安定した順序で走査し、候補のうち現在の割当枚数が最小のコンセプトへ置く。同数なら run seed とこの画像 ID から決める。割当はエポックごとに再計算できるが、同じ seed と snapshot から同じ結果になることを保証する。共有画像だけから成る概念は独立した集中区間を保証できないため、その件数を表示する。
5. 対象外の画像は通常プールへ置く。全画像がちょうど一方に属することを不変条件とする。`include` に未出現タグがあっても silent no-op にせず、0 件として報告する。`tag_data` 欠落・カテゴリ判定不能・通常プールへの回帰件数も報告し、必要なデータセットでは既存の `/api/v1/datasets/{dataset_id}/backfill-tag-data` を案内する。

大規模データでは全組み合わせのタグ対を作らない。画像ごとの既存 item 参照とコンセプト ID の索引を使い、抽出は画像数と実際のタグ数に比例する処理にする。対象数・メモリ使用量を初期化時に計測する。

## 5. 自然言語キャプションの照合

初版は曖昧な名前推定を行わない。`caption_aliases` に列挙した表記だけを照合し、表記の空白・アンダースコア、Unicode 正規化、大文字小文字を揃える。英字列は単語境界で照合し、`Ann` が `Anna` に一致しないようにする。姓名の逆順、ミドルネーム省略、愛称、他言語表記は明示的な別名として登録する。括弧付き作品名を自動で落とさない。

`match_natural_language=true` のときは、選択された主キャプションが自然言語形式ならその原文を照合する。タグ形式が主キャプションで、自然言語も照合したい場合は `train_runner.py` の snapshot に必要な `natural_language` キャプションを保持する。後段の caption dropout / shuffle 後の `caption` は照合に使わない。自然言語しかないデータでは、`caption_aliases` のキーを対象コンセプトとして使う。カテゴリを自然文から自動推測しないため、キーは選択した `category` の名前に限る。複数の別名が一致したときも §4 の一画像一割当を適用し、曖昧一致の件数を報告する。

既存 priority training の `caption_contains` の緩い部分文字列検索は変更しない。

## 6. エポック内スケジューラ

### 6.1 基本巡回

- 各画像を、現在の解像度 bucket と、モデルが要求する追加の同質条件（参照画像の有無など）を守って batch 化する。コンセプトごとに同じ bucket の画像をシャッフルして batch 化し、サイズ不足の最終 batch も保持する。別コンセプトを混ぜて batch を満たすことはしない。
- エポックごとにコンセプト順と各コンセプト内の batch 順を変える。一つのコンセプトから最大 `focus_batches` 個を続けて取り、次のコンセプトへ進む。残りは次の巡回で扱い、巡回ごとにコンセプト順を変える。これにより途中チェックポイントで扱ったコンセプト数を増やす。`focus_batches=0` なら各コンセプトを最後まで扱ってから次へ進む。
- `background_placement=front`（既定）では `background_interval` 個の集中 batch ごとに通常プールから1 batch を挟み、集中 batch が終わったら残りの通常 batch を消化する。途中で停止する run が、早い段階でコンセプト学習を終えられることを優先する。通常プールが尽きたら挿入をやめる。
- `background_placement=spread` は任意設定で、集中区間（最大 `focus_batches` batch）を単位としてエポック全体に配置する。集中区間数を `F`、通常 batch 数を `N` とし、最初は集中区間から始め、各区間の後へ `floor(i*N/F)-floor((i-1)*N/F)` 個（`i=1..F`）の通常 batch を置く。`F=0` なら通常 batch のみ。`background_interval` はこのモードで使用せず、UI では無効表示にする。途中停止時には未訪問のコンセプトが多くなり得るため、両モードの予定巡回率を表示する。
- `local_swap_window` は、**batch 化後**、隣り合うコンセプトの境界付近の集中バッチに適用する。各バッチの移動幅を元位置から最大指定数に制限し、両側の画像が少し交じることを許す。通常プールと再提示 batch の配置は先に確定し、それらは動かさない。bucket の異なるバッチは交換できるが、個々の batch の中身は変更しない。揺らぎ適用後も各コンセプトの大部分が連続することを確認する。
- 基本巡回では、unfittable と判定されたものなど既存の除外条件を除き、全入力画像をエポックに一度だけ出す。順序変更だけによる追加画像露出は作らない。ただしコンセプト別・bucket 別の端数 batch は batch 数と実効 batch size を変え得る。`max_steps` 等でエポック途中終了する場合は、訪問済み・未訪問コンセプト数、画像消化率、予定されていた集中区間の残数を UI / ログで分かるようにする。

### 6.2 任意の再提示

`replay_interval > 0` のときだけ、既に基本巡回で提示済みのコンセプト batch から1 batch を追加で挟む。候補はコンセプト間で偏らないように選び、同じ画像を短い間隔で連続提示しない。再提示 batch は基本巡回から消さず、総 batch 数、追加 batch 数、コンセプトごとの露出回数を表示する。露出増による効果を分離できるよう、評価では `replay_interval=0` を必ず基準にする。

再提示の選択範囲、選択 RNG、および必要な履歴は checkpoint から復元できるようにする。エポック開始時に完全な batch plan を決定し、同じ入力から再生成する。百万枚規模の plan 全体を checkpoint に複製しない。再生成と照合の契約は §7.1 とする。

## 7. 既存機能との接続・再開

| 接点 | 実装条件 |
|---|---|
| 解像度 bucket / 動的 crop | `CropPlanner` による当該エポックの crop spec と bucket 割当を先に確定し、その結果で concept batch を作る。priority path の現行「crop 無効化」を流用しない。 |
| 参照画像 / VE | `separate_by_reference` を保持し、後段の VE 用全体 shuffle が集中順序を壊さないようにする。混合 batch が生じたら分割後も位置を維持する。 |
| SenseNova | タスク別 batch 化が順序を変えたり画像を落としたりするため、初版は対象外として設定時に明示的に拒否する。対応時はタスク割当を先に確定してから concept plan を作る。 |
| 動画・音声 | 初版は対象外。画像と併用する run では、既存の動画・音声 batch を通常プールに含めるのではなく、設定を拒否する。 |
| online Danbooru 注入 | 初版は同時指定を拒否する。非同期 collector の到着内容と挿入位置を再現するには、画像を含む確定済み注入計画と checkpoint cursor が別途必要。 |
| priority training | 両方が有効ならエラー。既存の priority 設定と UI は維持する。 |
| 勾配累積 / LR schedule | コンセプト・bucket ごとの端数と再提示を含む実際の batch 数から、MNT iteration 数、optimizer update 数、epoch 長、scheduler 位置を計算する。初期の `ceil(total_items/batch_size)` をそのまま使わない。crop 有効時の `CropPlanner.step_offsets` もコンセプト別 batch 数を含む方式へ拡張する。 |
| 途中再開 | §7.1 の plan 再生成・照合を行い、一致した場合だけ保存された `batch_idx` を適用する。設定や分類情報が変わった場合は途中再開を拒否する。 |

無効時は既存 batch path の順序と乱数消費を変えない。新しい順序計画には run seed と epoch から導く専用 RNG を使い、他の dropout / sampling 用 RNG を余計に進めない。

### 7.1 中断・再開のバッチ順契約

現行の `save_training_state()` は model checkpoint と同じ step の state JSON に、エポック番号、**次に処理する** batch のエポック内絶対 index、batch 生成前のグローバル RNG state、dataset fingerprint、crop plan fingerprint を保存する。再開時は batch 列を作り直してから、その index までを切り落とす。この方式は batch 列が完全一致したときだけ安全である。

本機能の有効時は、以下を追加する。

1. **計画入力の確定**: 選択された checkpoint に結び付く学習 snapshot、`(dataset_unique_id, image_path)` 順の分類入力、正規化した設定、実効 run seed、epoch、batch size、MNT、解像度 bucket と参照画像の条件を使う。`CropPlanner.spec_for()` は seed・epoch・画像パス・元画像寸法などから決定論的に得られるものとし、既存の crop plan fingerprint も照合する。caption dropout 後の文では分類しない。
2. **計画生成**: 専用 RNG を `(実効 run seed, epoch, plan version)` から作る。画像割当、bucket 内 batch 化、コンセプト巡回、`background_placement`、境界の入れ替え、再提示、unfittable 除外、VE の分割まで含めて最終 batch 列を固定する。VE の後段全体 shuffle は本機能では行わない。途中で必要になる非同期 batch 挿入は初版で併用しない。
3. **state 保存**: `concept_batch_plan` に version、実効 seed、設定 hash、分類入力 hash、crop plan fingerprint、当該エポックの最終 batch 数、最終 batch 列の digest を保存する。digest は各 batch の順序付き `(dataset ID, image_path, bucket key, concept ID, base/replay)` から作り、crop spec と参照画像条件も含める。計画本体は保存しない。既存のエポック内 `batch_idx` は最終 batch 列に対する次バッチ位置のまま使う。
4. **再開前の照合**: model checkpoint と**同じ step** の state JSON を読み、分類入力・設定・crop fingerprint を先に比較する。再生成した最終 batch 数と digest が保存値に一致し、`0 <= batch_idx <= batch_count` のときだけ `batches[batch_idx:]` を適用する。`batch_idx == batch_count` は次のエポックへ進める。既存のグローバル RNG state も通常どおり復元する。
5. **不一致・欠落時**: 本機能を有効にした checkpoint の `concept_batch_plan` が欠ける、version が非対応、計画が不一致、または対応する state JSON が欠ける場合、途中再開はエラーにする。現在の既定のように旧 index のまま切り落としたり、無言でエポック先頭からやり直したりしない。データや設定を変えて続行したい場合は、別途「新しいエポックから再開」を明示的に選ぶ。この場合は既学習画像の再提示と scheduler 位置の変化を表示する。

エポックごとの元画像寸法、bucket、端数 batch 数が変わる場合、総 step 数を初回エポックから外挿しない。step 指定 run は目標 step に達するまで必要なエポック計画を作り、epoch 指定 run は全エポックの計画 batch 数を集計する。作成した計画の digest と、先頭・末尾の batch ID をログに出す。checkpoint 保存は batch の全 MNT iteration 完了後に行う既存位置を維持する。

## 8. API・UI・実装箇所

- `backend/core/training/concept_batch_order.py`（新規）: 設定検証、索引、分類、batch plan、digest。大きな `base_trainer.py` には呼び出しと既存機能との接続だけを置く。
- `backend/core/training/train_runner.py`: 必要な自然言語キャプションの snapshot 取り込み。`priority_training` の既存経路は変更しない。
- `backend/api/param_defaults.py`、`backend/api/routes.py`、`openapi.yaml`: 設定の唯一の既定値、入力検証、API 契約。
- `frontend/src/utils/api.ts`、`frontend/src/components/training/TrainingConfig.tsx`: カテゴリ、集中長、揺らぎ幅、通常データの配置モードと固定間隔、再提示間隔、自然言語別名の入力と設定保存・復元。`front` は早期消化、`spread` は途中停止時に未訪問コンセプトが残り得ることを説明する。priority training との同時有効化も UI で説明する。
- 学習ログまたは事前プレビュー: 対象コンセプト数、対象画像数、通常画像数、重複候補数、0 件の指定、batch 数、追加再提示数、予定巡回順の先頭数件を表示する。百万枚規模で全順序を UI に送らない。

API 追加時には `openapi.yaml` を先に更新し、training request、保存・再開、frontend の送受信を揃える。学習パラメータの既定値を別ファイルへ複製しない。

## 9. 受け入れ条件と評価

### 実装の受け入れ条件

1. `replay_interval=0` では、除外条件を適用した各画像が各エポックにちょうど一度現れ、基本 batch 数と総露出回数がログから検証できる。
2. Character / Artist のカテゴリ、別名、大文字小文字、姓名別表記、境界一致、複数対象タグ、タグ情報のない画像を fixture で検証する。
3. 異なる解像度・参照画像の有無を含むデータでも不正な混合 batch がない。動的 crop を有効にしたエポックごとの bucket 変化を維持する。
4. 同じ seed / snapshot / 設定なら、`front` / `spread`、crop、VE、再提示の各条件で通常実行と途中再開後の未実行 batch 列・optimizer update 数が一致する。エポック末端の `batch_idx`、複数回の途中再開、選択 checkpoint と state JSON の step 不一致も検証する。キャプション・設定変更、計画 version の不一致は途中再開拒否として検出する。
5. 無効時の既存順序、priority training、前処理結果は変化しない。百万枚規模の項目数で、索引構築時間と追加メモリを測定する。

### 学習効果の評価

同じデータ、seed、キャプション処理で、通常シャッフルと `replay_interval=0` の巡回順序を比較する。端数 batch による batch 数・実効 batch size・optimizer update 数の差を測り、比較条件を揃える。別実験で再提示を有効にし、**露出回数を揃えた対照群**を置く。`front` と `spread` は同じ早期停止 step でも訪問済みコンセプト数が違うため、途中 checkpoint の訪問率と品質を別々に示す。対象は頻出・希少な Character と Artist、既存知識の保持用プロンプトを含める。途中と最終 checkpoint でタグ応答、類似キャラクターへの混同、既存タグの劣化を評価する。contrastive loss / unconditional 関連の効果は、その設定を変えた実験で別に測る。改善を前提に既定有効化しない。
